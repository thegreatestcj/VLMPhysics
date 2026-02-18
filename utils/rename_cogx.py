#!/usr/bin/env python3
"""
Rename CogVideoX videos and feature folders to meaningful names.

Before: v1_test_2674.mp4, v1_test_2674/
After:  cogvideox-2b_A_ball_bounces_on_the_ground.mp4, cogvideox-2b_A_ball_bounces_on_the_ground/

Updates metadata.json and all feature directories accordingly.

Usage:
    # Dry run (shows what would be renamed)
    python data/rename_cogx.py \
        --data-dir ~/scratch/physics/videophy_cogx \
        --feature-dir ~/scratch/physics/videophy_cogx_features \
        --dry-run

    # Actually rename
    python data/rename_cogx.py \
        --data-dir ~/scratch/physics/videophy_cogx \
        --feature-dir ~/scratch/physics/videophy_cogx_features
"""

import argparse
import json
import logging
import os
import re
import shutil
from pathlib import Path
from typing import Optional

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


# ============================================================
# Naming convention (shared with download script)
# ============================================================

MAX_CAPTION_LEN = 80  # Max characters for caption portion of filename


def caption_to_slug(caption: str, max_len: int = MAX_CAPTION_LEN) -> str:
    """
    Convert a caption to a filesystem-safe slug.

    'A ball bounces on the ground.' -> 'A_ball_bounces_on_the_ground'
    """
    # Remove special characters, keep alphanumeric + spaces
    slug = re.sub(r"[^a-zA-Z0-9\s]", "", caption)
    # Collapse whitespace and convert to underscores
    slug = re.sub(r"\s+", "_", slug.strip())
    # Truncate
    if len(slug) > max_len:
        slug = slug[:max_len].rstrip("_")
    return slug


def make_video_name(source: str, caption: str, suffix: str = "") -> str:
    """
    Generate a canonical video name.

    Args:
        source: e.g. 'CogVideoX-2B', 'cogvideo', etc.
        caption: Video caption text
        suffix: Optional disambiguator (e.g. '_1') for duplicate captions

    Returns:
        e.g. 'cogvideox-2b_A_ball_bounces_on_the_ground'
    """
    # Normalize source prefix
    src_lower = (
        source.strip().lower().replace(" ", "").replace("-", "").replace("_", "")
    )
    if "cogvideox2b" in src_lower or "cogvideo2b" in src_lower:
        prefix = "cogvideox-2b"
    elif "cogvideox5b" in src_lower or "cogvideo5b" in src_lower:
        prefix = "cogvideox-5b"
    elif "cogvideo" in src_lower:
        # v2 test uses "cogvideo" without version — check URL later, default to 5b
        prefix = "cogvideox-5b"
    else:
        prefix = source.lower().replace(" ", "_")

    slug = caption_to_slug(caption)
    name = f"{prefix}_{slug}"
    if suffix:
        name = f"{name}{suffix}"
    return name


# ============================================================
# Rename logic
# ============================================================


def build_rename_map(metadata: list) -> dict:
    """
    Build old_stem -> new_stem mapping from metadata.

    Handles duplicate captions by appending _2, _3, etc.
    """
    # Track new names to detect duplicates
    name_counts = {}
    rename_map = {}

    for item in metadata:
        # Get old stem
        old_local = item.get("local_path", item.get("video_path", ""))
        old_stem = Path(old_local).stem  # e.g. 'v1_test_2674'
        if not old_stem:
            continue

        source = item.get("source", "unknown")
        caption = item.get("caption", "")
        if not caption:
            logger.warning(f"No caption for {old_stem}, skipping")
            continue

        base_name = make_video_name(source, caption)

        # Handle duplicates
        if base_name in name_counts:
            name_counts[base_name] += 1
            new_stem = f"{base_name}_{name_counts[base_name]}"
        else:
            name_counts[base_name] = 1
            new_stem = base_name

        rename_map[old_stem] = new_stem

    return rename_map


def rename_videos(videos_dir: Path, rename_map: dict, dry_run: bool = True) -> int:
    """Rename video files in videos/ directory."""
    count = 0
    for old_stem, new_stem in sorted(rename_map.items()):
        # Find the actual file (could be .mp4, .webm, etc.)
        matches = list(videos_dir.glob(f"{old_stem}.*"))
        for old_path in matches:
            new_path = old_path.parent / f"{new_stem}{old_path.suffix}"
            if old_path == new_path:
                continue
            if dry_run:
                logger.info(f"  [video] {old_path.name} -> {new_path.name}")
            else:
                old_path.rename(new_path)
            count += 1
    return count


def rename_feature_dirs(
    feature_dir: Path, rename_map: dict, dry_run: bool = True
) -> int:
    """Rename feature subdirectories."""
    if not feature_dir or not feature_dir.exists():
        return 0

    count = 0
    for old_stem, new_stem in sorted(rename_map.items()):
        old_dir = feature_dir / old_stem
        new_dir = feature_dir / new_stem
        if not old_dir.exists():
            continue
        if old_dir == new_dir:
            continue
        if dry_run:
            logger.info(f"  [feat]  {old_stem}/ -> {new_stem}/")
        else:
            old_dir.rename(new_dir)
        count += 1
    return count


def update_metadata(metadata: list, rename_map: dict, data_dir: Path) -> list:
    """Update metadata with new paths and video_filename."""
    for item in metadata:
        old_local = item.get("local_path", item.get("video_path", ""))
        old_stem = Path(old_local).stem
        old_ext = Path(old_local).suffix or ".mp4"

        if old_stem in rename_map:
            new_stem = rename_map[old_stem]
            new_filename = f"{new_stem}{old_ext}"
            item["video_filename"] = new_stem
            item["local_path"] = f"videos/{new_filename}"
            item["video_path"] = str(data_dir / "videos" / new_filename)
        else:
            # Ensure video_filename is set even if not renamed
            item["video_filename"] = old_stem

    return metadata


def main():
    parser = argparse.ArgumentParser(
        description="Rename CogVideoX videos and features to meaningful names"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="CogVideoX data directory (contains videos/ and metadata.json)",
    )
    parser.add_argument(
        "--feature-dir",
        type=str,
        default=None,
        help="Feature directory to also rename (optional)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Only print what would be renamed"
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir).expanduser()
    feature_dir = Path(args.feature_dir).expanduser() if args.feature_dir else None
    videos_dir = data_dir / "videos"
    meta_path = data_dir / "metadata.json"

    if not meta_path.exists():
        logger.error(f"metadata.json not found at {meta_path}")
        return

    # Load metadata
    with open(meta_path) as f:
        metadata = json.load(f)
    logger.info(f"Loaded {len(metadata)} entries from metadata.json")

    # Build rename map
    rename_map = build_rename_map(metadata)
    logger.info(f"Built rename map: {len(rename_map)} entries")

    # Show sample
    for i, (old, new) in enumerate(sorted(rename_map.items())):
        if i >= 5:
            logger.info(f"  ... ({len(rename_map) - 5} more)")
            break
        logger.info(f"  {old} -> {new}")

    if args.dry_run:
        logger.info("\n=== DRY RUN ===\n")

    # Rename videos
    logger.info("Renaming videos...")
    n_vid = rename_videos(videos_dir, rename_map, dry_run=args.dry_run)
    logger.info(f"  Videos: {n_vid} renamed")

    # Rename features
    if feature_dir:
        logger.info("Renaming feature directories...")
        n_feat = rename_feature_dirs(feature_dir, rename_map, dry_run=args.dry_run)
        logger.info(f"  Feature dirs: {n_feat} renamed")

    # Update and save metadata
    if not args.dry_run:
        metadata = update_metadata(metadata, rename_map, data_dir)
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Updated metadata.json")

        # Also update feature metadata/labels if they exist
        if feature_dir:
            for json_file in feature_dir.glob("*.json"):
                try:
                    with open(json_file) as f:
                        data = json.load(f)

                    # Handle labels.json: {old_stem: label} -> {new_stem: label}
                    if isinstance(data, dict) and all(
                        isinstance(v, (int, float)) for v in data.values()
                    ):
                        new_data = {}
                        for k, v in data.items():
                            new_data[rename_map.get(k, k)] = v
                        with open(json_file, "w") as f:
                            json.dump(new_data, f, indent=2)
                        logger.info(f"  Updated {json_file.name}")

                    # Handle metadata_shard*.json: {videos: {old_stem: ...}}
                    elif isinstance(data, dict) and "videos" in data:
                        new_videos = {}
                        for k, v in data["videos"].items():
                            new_videos[rename_map.get(k, k)] = v
                        data["videos"] = new_videos
                        with open(json_file, "w") as f:
                            json.dump(data, f, indent=2)
                        logger.info(f"  Updated {json_file.name}")

                    # Handle enriched_labels.json: {old_stem: {label, caption}}
                    elif isinstance(data, dict):
                        new_data = {}
                        for k, v in data.items():
                            new_data[rename_map.get(k, k)] = v
                        with open(json_file, "w") as f:
                            json.dump(new_data, f, indent=2)
                        logger.info(f"  Updated {json_file.name}")

                except Exception as e:
                    logger.warning(f"  Could not update {json_file.name}: {e}")

    logger.info("Done!")


if __name__ == "__main__":
    main()
