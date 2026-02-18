#!/usr/bin/env python3
"""
Download CogVideoX-2B (and optionally 5B) videos from all VideoPhy datasets.

Scans:
  1. videophysics/videophy_test_public   (VideoPhy v1 test, ~3360 rows)
  2. videophysics/videophy_train_public  (VideoPhy v1 train, ~4600 rows)
  3. videophysics/videophy2_test         (VideoPhy-2 test, ~3400 rows)
  4. videophysics/videophy2_train        (VideoPhy-2 train)

Outputs:
  <output_dir>/
  ├── videos/
  │   ├── v1_test_<idx>.mp4
  │   ├── v1_train_<idx>.mp4
  │   ├── v2_test_<idx>.mp4
  │   └── ...
  └── metadata.json   (list of dicts with all fields + local path)

Usage:
    pip install datasets requests --break-system-packages
    python download_videophy_cogx.py --output-dir ~/scratch/physics/videophy_cogx
    python download_videophy_cogx.py --output-dir ~/scratch/physics/videophy_cogx --include-5b
"""

import argparse
import json
import logging
import os
import re
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

# ============================================================
# Source name matching
# ============================================================

COGX_2B_PATTERNS = [
    "cogvideox-2b",
    "cogvideox_2b",
    "cogvideo-2b",
    "cogvideo_2b",
    "CogVideoX-2B",
    "CogVideoX_2B",
    "cogvideox2b",
]

COGX_5B_PATTERNS = [
    "cogvideox-5b",
    "cogvideox_5b",
    "cogvideo-5b",
    "cogvideo_5b",
    "CogVideoX-5B",
    "CogVideoX_5B",
    "cogvideox5b",
]


def is_cogx_2b(source: str) -> bool:
    s = source.strip().lower().replace(" ", "").replace("-", "").replace("_", "")
    return "cogvideox2b" in s or "cogvideo2b" in s


def is_cogx_5b(source: str) -> bool:
    s = source.strip().lower().replace(" ", "").replace("-", "").replace("_", "")
    return "cogvideox5b" in s or "cogvideo5b" in s


def is_cogx(source: str, include_5b: bool = False) -> bool:
    if is_cogx_2b(source):
        return True
    if include_5b and is_cogx_5b(source):
        return True
    return False


# ============================================================
# Dataset loading
# ============================================================


def load_hf_dataset(dataset_name: str):
    """Load a HuggingFace dataset, return list of dicts."""
    try:
        from datasets import load_dataset

        ds = load_dataset(
            dataset_name, split="test" if "test" in dataset_name else "train"
        )
        logger.info(f"Loaded {dataset_name}: {len(ds)} rows")
        return [dict(row) for row in ds]
    except Exception as e:
        logger.warning(f"Failed to load {dataset_name}: {e}")
        # Try alternative splits
        try:
            from datasets import load_dataset

            ds = load_dataset(dataset_name)
            # Get whatever split exists
            for split_name in ds:
                logger.info(
                    f"Loaded {dataset_name} split={split_name}: {len(ds[split_name])} rows"
                )
                return [dict(row) for row in ds[split_name]]
        except Exception as e2:
            logger.error(f"Cannot load {dataset_name} at all: {e2}")
            return []


def get_source_field(row: dict) -> str:
    """Extract source/model name from a row (handles both v1 and v2 schemas)."""
    # v1: 'source' field
    if "source" in row and row["source"]:
        return str(row["source"])
    # v2: 'model_name' field
    if "model_name" in row and row["model_name"]:
        s = str(row["model_name"])
        # v2 uses "cogvideo" which is actually CogVideoX-5B (check URL to confirm)
        if s.lower() == "cogvideo":
            url = row.get("video_url", "").lower()
            if "5b" in url:
                return "cogvideox-5b"
            elif "2b" in url:
                return "cogvideox-2b"
            else:
                return "cogvideox-5b"  # v2 default is 5B
        return s
    # Try to infer from video_url
    url = row.get("video_url", "")
    if "CogVideoX" in url or "cogvideox" in url.lower():
        if "2b" in url.lower() or "2B" in url:
            return "cogvideox-2b"
        if "5b" in url.lower() or "5B" in url:
            return "cogvideox-5b"
    return "unknown"


# ============================================================
# Filename generation
# ============================================================

MAX_CAPTION_LEN = 80


def caption_to_slug(caption: str, max_len: int = MAX_CAPTION_LEN) -> str:
    """Convert caption to filesystem-safe slug."""
    slug = re.sub(r"[^a-zA-Z0-9\s]", "", caption)
    slug = re.sub(r"\s+", "_", slug.strip())
    if len(slug) > max_len:
        slug = slug[:max_len].rstrip("_")
    return slug


def make_video_name(source: str, caption: str, suffix: str = "") -> str:
    """Generate canonical video name: cogvideox-2b_A_ball_bounces."""
    src_lower = (
        source.strip().lower().replace(" ", "").replace("-", "").replace("_", "")
    )
    if "cogvideox2b" in src_lower or "cogvideo2b" in src_lower:
        prefix = "cogvideox-2b"
    elif "cogvideox5b" in src_lower or "cogvideo5b" in src_lower:
        prefix = "cogvideox-5b"
    elif "cogvideo" in src_lower:
        prefix = "cogvideox-5b"
    else:
        prefix = source.lower().replace(" ", "_")

    slug = caption_to_slug(caption)
    name = f"{prefix}_{slug}"
    if suffix:
        name = f"{name}{suffix}"
    return name


def normalize_row(row: dict, dataset_tag: str, idx: int) -> dict:
    """Normalize a row from any VideoPhy dataset into a common schema."""
    source = get_source_field(row)

    # Physics label: v1 uses majority_pc, v2 uses pc
    physics = None
    if "majority_pc" in row:
        physics = int(row["majority_pc"])
    elif "pc" in row:
        val = row["pc"]
        if isinstance(val, (int, float)):
            # v2 uses 1-5 scale; v1 train uses 0/1
            if val in (0, 1):
                physics = int(val)
            else:
                # v2: threshold at >= 4 for "good physics"
                physics = 1 if val >= 4 else 0

    # Semantic adherence
    sa = None
    if "majority_sa" in row:
        sa = int(row["majority_sa"])
    elif "sa" in row:
        val = row["sa"]
        if isinstance(val, (int, float)):
            if val in (0, 1):
                sa = int(val)
            else:
                sa = 1 if val >= 4 else 0

    return {
        "dataset": dataset_tag,
        "original_index": idx,
        "source": source,
        "model_variant": "2b"
        if is_cogx_2b(source)
        else ("5b" if is_cogx_5b(source) else "other"),
        "caption": row.get("caption", ""),
        "video_url": row.get("video_url", ""),
        "physics": physics,
        "sa": sa,
        "states_of_matter": row.get("states_of_matter", None),
        "complexity": row.get("complexity", None),
        "category": row.get("category", None),
        "action": row.get("action", None),
        "is_hard": row.get("is_hard", None),
        # v2 extra fields
        "joint": row.get("joint", None),
        "physics_rules_followed": row.get("physics_rules_followed", None),
        "physics_rules_unfollowed": row.get("physics_rules_unfollowed", None),
        "human_violated_rules": row.get("human_violated_rules", None),
    }


# ============================================================
# Video downloading
# ============================================================


def download_video(url: str, save_path: Path, max_retries: int = 3) -> bool:
    """Download a video file with retries."""
    if save_path.exists() and save_path.stat().st_size > 1000:
        return True  # Already downloaded

    for attempt in range(max_retries):
        try:
            resp = requests.get(url, timeout=60, stream=True)
            resp.raise_for_status()
            save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
            if save_path.stat().st_size > 1000:
                return True
            else:
                save_path.unlink(missing_ok=True)
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2**attempt)
            else:
                logger.warning(f"Failed to download {url}: {e}")
    return False


# ============================================================
# Main
# ============================================================

DATASETS_TO_SCAN = [
    ("videophysics/videophy_test_public", "v1_test"),
    ("videophysics/videophy_train_public", "v1_train"),
    ("videophysics/videophy2_test", "v2_test"),
    ("videophysics/videophy2_train", "v2_train"),
]


def main():
    parser = argparse.ArgumentParser(
        description="Download CogVideoX videos from all VideoPhy datasets"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory (e.g. ~/scratch/physics/videophy_cogx)",
    )
    parser.add_argument(
        "--include-5b", action="store_true", help="Also download CogVideoX-5B videos"
    )
    parser.add_argument(
        "--workers", type=int, default=8, help="Number of parallel download workers"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only scan datasets and report counts, don't download",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="*",
        default=None,
        help="Only scan specific datasets (e.g. v1_test v2_test)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir).expanduser()
    videos_dir = output_dir / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)

    # =========================================================
    # Phase 1: Scan all datasets, collect CogVideoX entries
    # =========================================================
    all_entries = []
    all_sources_seen = set()
    name_counter = {}  # Track duplicate caption slugs

    for hf_name, tag in DATASETS_TO_SCAN:
        if args.datasets and tag not in args.datasets:
            logger.info(f"Skipping {hf_name} (not in --datasets)")
            continue

        logger.info(f"Scanning {hf_name} ...")
        rows = load_hf_dataset(hf_name)

        # Catalog all source names
        source_counts = {}
        cogx_count = 0
        for i, row in enumerate(rows):
            src = get_source_field(row)
            source_counts[src] = source_counts.get(src, 0) + 1
            all_sources_seen.add(src)

            if is_cogx(src, include_5b=args.include_5b):
                entry = normalize_row(row, tag, i)
                # Generate meaningful filename from source + caption
                url = entry["video_url"]
                ext = Path(url).suffix if Path(url).suffix else ".mp4"
                base_name = make_video_name(entry["source"], entry["caption"])
                # Track duplicates within this run
                if base_name not in name_counter:
                    name_counter[base_name] = 0
                name_counter[base_name] += 1
                if name_counter[base_name] > 1:
                    stem = f"{base_name}_{name_counter[base_name]}"
                else:
                    stem = base_name
                entry["local_filename"] = f"{stem}{ext}"
                entry["video_filename"] = stem
                entry["local_path"] = str(videos_dir / entry["local_filename"])
                all_entries.append(entry)
                cogx_count += 1

        logger.info(f"  Sources in {tag}: {json.dumps(source_counts, indent=2)}")
        logger.info(f"  CogVideoX entries found: {cogx_count}")

    logger.info("=" * 60)
    logger.info(f"All source names seen across datasets: {sorted(all_sources_seen)}")
    logger.info(f"Total CogVideoX entries to download: {len(all_entries)}")

    # Breakdown
    by_variant = {}
    by_dataset = {}
    for e in all_entries:
        v = e["model_variant"]
        by_variant[v] = by_variant.get(v, 0) + 1
        d = e["dataset"]
        by_dataset[d] = by_dataset.get(d, 0) + 1

    logger.info(f"  By variant: {by_variant}")
    logger.info(f"  By dataset: {by_dataset}")

    # Physics label distribution
    physics_counts = {}
    for e in all_entries:
        p = e["physics"]
        physics_counts[p] = physics_counts.get(p, 0) + 1
    logger.info(f"  Physics labels: {physics_counts}")

    if args.dry_run:
        logger.info("Dry run - not downloading. Saving metadata only.")
        meta_path = output_dir / "metadata.json"
        with open(meta_path, "w") as f:
            json.dump(all_entries, f, indent=2, default=str)
        logger.info(f"Saved metadata preview to {meta_path}")
        return

    # =========================================================
    # Phase 2: Download videos
    # =========================================================
    logger.info(f"Downloading {len(all_entries)} videos with {args.workers} workers...")

    success = 0
    failed = 0

    def _download_one(entry):
        url = entry["video_url"]
        path = Path(entry["local_path"])
        return entry["local_filename"], download_video(url, path)

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(_download_one, e): e for e in all_entries}
        for i, future in enumerate(as_completed(futures)):
            fname, ok = future.result()
            if ok:
                success += 1
            else:
                failed += 1
            if (i + 1) % 50 == 0 or (i + 1) == len(all_entries):
                logger.info(
                    f"  Progress: {i + 1}/{len(all_entries)} (success={success}, failed={failed})"
                )

    logger.info(f"Download complete: {success} success, {failed} failed")

    # =========================================================
    # Phase 3: Save metadata
    # =========================================================
    # Mark download status
    for entry in all_entries:
        path = Path(entry["local_path"])
        entry["downloaded"] = path.exists() and path.stat().st_size > 1000

    meta_path = output_dir / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(all_entries, f, indent=2, default=str)
    logger.info(f"Saved metadata to {meta_path}")

    # Summary
    downloaded = sum(1 for e in all_entries if e.get("downloaded"))
    logger.info("=" * 60)
    logger.info(f"Summary:")
    logger.info(f"  Total entries:  {len(all_entries)}")
    logger.info(f"  Downloaded:     {downloaded}")
    logger.info(f"  Physics=1:      {sum(1 for e in all_entries if e['physics'] == 1)}")
    logger.info(f"  Physics=0:      {sum(1 for e in all_entries if e['physics'] == 0)}")
    logger.info(
        f"  Physics=None:   {sum(1 for e in all_entries if e['physics'] is None)}"
    )
    logger.info(f"  Videos dir:     {videos_dir}")
    logger.info(f"  Metadata:       {meta_path}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
