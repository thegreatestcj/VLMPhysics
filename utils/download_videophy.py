#!/usr/bin/env python3
"""
Download ALL VideoPhy + VideoPhy-2 Videos

Combines 4 splits into one unified metadata.json:
  - videophysics/videophy_train_public   (~4587, binary labels)
  - videophysics/videophy_test_public    (~3360, binary labels)
  - videophysics/videophy2_train         (~3340, 1-5 scale -> binarize)
  - videophysics/videophy2_test          (~3400, 1-5 scale -> binarize)

Binarization for v2: physics = (pc >= 4), sa = (sa >= 4)

Output:
    videophy_data/
    ├── metadata.json   <- single source of truth
    └── videos/         <- all mp4 files (named {source}_{basename})

Usage:
    python utils/download_videophy.py --metadata-only
    python utils/download_videophy.py
    python utils/download_videophy.py --max-videos 50
"""

import os
import json
import argparse
import requests
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from urllib.parse import urlparse
from collections import Counter

try:
    from datasets import load_dataset

    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False
    print("Warning: 'datasets' not installed. pip install datasets")


def download_video(url: str, output_path: Path, timeout: int = 60) -> bool:
    """Download a single video file."""
    try:
        if output_path.exists():
            return True
        response = requests.get(url, timeout=timeout, stream=True)
        response.raise_for_status()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return True
    except Exception as e:
        print(f"Failed: {url}: {e}")
        return False


def get_video_basename(url: str) -> str:
    """Extract raw filename from URL (without source prefix)."""
    return os.path.basename(urlparse(url).path)


def make_video_filename(source: str, url: str) -> str:
    """
    Create a unique video filename by prefixing source model name.

    Without this, different models generating the same prompt share
    identical filenames (e.g. ball_rolls.mp4), causing ~6000 videos
    to be lost during deduplication.

    Examples:
        lavie_ball_rolls.mp4
        pika_ball_rolls.mp4
        cogvideoX5b_ball_rolls.mp4
    """
    basename = get_video_basename(url)
    # Sanitize source name: lowercase, replace spaces/slashes
    safe_source = (
        source.lower()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("(", "")
        .replace(")", "")
    )
    return f"{safe_source}_{basename}"


# =========================================================================
# VideoPhy v1 loaders (binary labels)
# =========================================================================


def load_v1_train():
    print("  Loading videophysics/videophy_train_public ...")
    raw = [
        dict(x)
        for x in load_dataset("videophysics/videophy_train_public", split="train")
    ]
    entries = []
    for d in raw:
        entries.append(
            {
                "video_url": d.get("video_url", ""),
                "caption": d.get("caption", ""),
                "physics": int(d.get("physics", 0)),
                "sa": int(d.get("sa", 0)),
                "source": d.get("source", "unknown"),
                "states_of_matter": d.get("states_of_matter", ""),
                "complexity": int(d.get("complexity", 0)),
                "dataset_split": "v1_train",
            }
        )
    print(f"  -> {len(entries)} entries")
    return entries


def load_v1_test():
    print("  Loading videophysics/videophy_test_public ...")
    raw = [
        dict(x) for x in load_dataset("videophysics/videophy_test_public", split="test")
    ]
    entries = []
    for d in raw:
        entries.append(
            {
                "video_url": d.get("video_url", ""),
                "caption": d.get("caption", ""),
                "physics": int(d.get("majority_pc", 0)),  # majority_pc -> physics
                "sa": int(d.get("majority_sa", 0)),  # majority_sa -> sa
                "source": d.get("source", "unknown"),
                "states_of_matter": d.get("states_of_matter", ""),
                "complexity": int(d.get("complexity", 0)),
                "dataset_split": "v1_test",
            }
        )
    print(f"  -> {len(entries)} entries")
    return entries


# =========================================================================
# VideoPhy-2 loaders (1-5 scale -> binarize: >= 4 means positive)
# =========================================================================


def _extract_model_from_url(url: str) -> str:
    """Try to extract model name from v2 video URL path."""
    # e.g. .../cogvideoX5b_hard_train_upsampled/video.mp4
    parts = urlparse(url).path.split("/")
    for p in parts:
        for prefix in [
            "cogvideo",
            "hunyuan",
            "wan",
            "ray2",
            "dream_machine",
            "pika",
            "kling",
            "veo",
            "sora",
            "mochi",
            "ltx",
            "cosmos",
        ]:
            if prefix in p.lower():
                return p.split("_")[0] if "_" in p else p
    return "unknown"


def load_v2_train():
    print("  Loading videophysics/videophy2_train ...")
    raw = [dict(x) for x in load_dataset("videophysics/videophy2_train", split="train")]
    entries = []
    for d in raw:
        pc = int(d.get("pc", 0))
        sa = int(d.get("sa", 0))
        source = _extract_model_from_url(d.get("video_url", ""))
        entries.append(
            {
                "video_url": d.get("video_url", ""),
                "caption": d.get("caption", ""),
                "physics": int(pc >= 4),  # binarize
                "sa": int(sa >= 4),  # binarize
                "source": source,
                "states_of_matter": "",
                "complexity": 0,
                "dataset_split": "v2_train",
            }
        )
    print(f"  -> {len(entries)} entries")
    return entries


def load_v2_test():
    print("  Loading videophysics/videophy2_test ...")
    raw = [dict(x) for x in load_dataset("videophysics/videophy2_test", split="test")]
    entries = []
    for d in raw:
        pc = int(d.get("pc", 0))
        sa = int(d.get("sa", 0))
        source = d.get("model_name", "unknown")
        entries.append(
            {
                "video_url": d.get("video_url", ""),
                "caption": d.get("caption", ""),
                "physics": int(pc >= 4),  # binarize
                "sa": int(sa >= 4),  # binarize
                "source": source,
                "states_of_matter": "",
                "complexity": 0,
                "dataset_split": "v2_test",
            }
        )
    print(f"  -> {len(entries)} entries")
    return entries


# =========================================================================
# Main
# =========================================================================


def main():
    parser = argparse.ArgumentParser(description="Download ALL VideoPhy v1+v2 data")
    parser.add_argument(
        "--data-dir", type=str, default="/users/ctang33/scratch/physics/videophy_data"
    )
    parser.add_argument("--metadata-only", action="store_true")
    parser.add_argument("--max-videos", type=int, default=None)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--v1-only", action="store_true", help="Skip VideoPhy-2")
    parser.add_argument("--v2-only", action="store_true", help="Skip VideoPhy v1")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    videos_dir = data_dir / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)

    # Load splits
    print("=" * 60)
    print("Loading metadata from HuggingFace")
    print("=" * 60)

    all_entries = []
    if not args.v2_only:
        all_entries += load_v1_train()
        all_entries += load_v1_test()
    if not args.v1_only:
        all_entries += load_v2_train()
        all_entries += load_v2_test()

    # Build unified metadata with source-prefixed filenames
    # This prevents filename collisions across different models
    seen = set()
    metadata = []
    collision_count = 0

    for d in all_entries:
        # Use source-prefixed filename to avoid collisions
        fn = make_video_filename(d["source"], d["video_url"])

        if fn in seen:
            collision_count += 1
            continue
        seen.add(fn)

        d["video_filename"] = fn
        d["video_path"] = str(videos_dir / fn)
        d["index"] = len(metadata)
        metadata.append(d)

    # Stats
    total = len(metadata)
    pos = sum(1 for m in metadata if m["physics"] == 1)
    neg = total - pos
    sa_pos = sum(1 for m in metadata if m["sa"] == 1)
    sources = Counter(m["source"] for m in metadata)
    splits = Counter(m["dataset_split"] for m in metadata)

    print(f"\n{'=' * 60}")
    print(f"Total:      {total} unique videos (collisions skipped: {collision_count})")
    print(f"Splits:     {dict(sorted(splits.items()))}")
    print(f"Physics:    {pos} pos / {neg} neg  (pos_weight ~ {neg / max(pos, 1):.3f})")
    print(f"SA:         {sa_pos} pos / {total - sa_pos} neg")
    print(f"Sources ({len(sources)} models):")
    for src, count in sorted(sources.items(), key=lambda x: -x[1]):
        print(f"  {src:25s}: {count}")
    print(f"{'=' * 60}")

    # Save metadata.json
    with open(data_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\nSaved metadata.json ({total} entries)")

    if args.metadata_only:
        print("--metadata-only set. Done.")
        return

    # Download videos
    dl_list = metadata[: args.max_videos] if args.max_videos else metadata
    existing = sum(1 for m in dl_list if Path(m["video_path"]).exists())
    to_dl = len(dl_list) - existing
    print(f"\nDownloading: {to_dl} videos ({existing} already exist)")

    if to_dl > 0:
        failed = []
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {}
            for m in dl_list:
                if not Path(m["video_path"]).exists():
                    futs[
                        ex.submit(download_video, m["video_url"], Path(m["video_path"]))
                    ] = m["video_filename"]
            for f in tqdm(as_completed(futs), total=len(futs), desc="Downloading"):
                if not f.result():
                    failed.append(futs[f])
        ok = sum(1 for m in dl_list if Path(m["video_path"]).exists())
        print(f"\nAvailable: {ok}/{len(dl_list)}")
        if failed:
            print(f"Failed: {len(failed)}")
            (data_dir / "failed_downloads.txt").write_text("\n".join(failed))

    total_bytes = sum(
        Path(m["video_path"]).stat().st_size
        for m in dl_list
        if Path(m["video_path"]).exists()
    )
    print(f"Disk: {total_bytes / 1e9:.2f} GB")
    print("Done.")


if __name__ == "__main__":
    main()
