#!/usr/bin/env python3
"""
Download VideoPhy Training Dataset

Downloads the VideoPhy training dataset with physics labels for training
the physics discriminator head.

Dataset: https://huggingface.co/datasets/videophysics/videophy_train_public
Videos: https://huggingface.co/videophysics/videophy-train-videos

Usage:
    # Download metadata only (quick)
    python download_videophy.py --metadata-only --output-dir data/videophy

    # Download videos (takes longer, ~10-20GB)
    python download_videophy.py --output-dir data/videophy --max-videos 500

    # Download all videos
    python download_videophy.py --output-dir data/videophy
"""

import os
import json
import argparse
import requests
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from urllib.parse import urlparse

try:
    from datasets import load_dataset

    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False
    print(
        "Warning: 'datasets' library not installed. Install with: pip install datasets"
    )


def download_video(url: str, output_path: Path, timeout: int = 60) -> bool:
    """Download a single video file."""
    try:
        if output_path.exists():
            return True  # Skip if already downloaded

        response = requests.get(url, timeout=timeout, stream=True)
        response.raise_for_status()

        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        return True
    except Exception as e:
        print(f"Failed to download {url}: {e}")
        return False


def get_video_filename(url: str) -> str:
    """Extract filename from URL."""
    parsed = urlparse(url)
    return os.path.basename(parsed.path)


def main():
    parser = argparse.ArgumentParser(description="Download VideoPhy Training Dataset")
    parser.add_argument(
        "--output-dir", type=str, default="data/videophy", help="Output directory"
    )
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Download metadata only, skip videos",
    )
    parser.add_argument(
        "--max-videos",
        type=int,
        default=None,
        help="Maximum number of videos to download",
    )
    parser.add_argument(
        "--workers", type=int, default=8, help="Number of parallel download workers"
    )
    parser.add_argument(
        "--source-filter",
        type=str,
        default=None,
        help="Filter by source (e.g., 'lavie', 'pika', 'videocrafter2')",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    videos_dir = output_dir / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # Step 1: Load dataset metadata from HuggingFace
    # =========================================================================
    print("=" * 60)
    print("Step 1: Loading dataset metadata from HuggingFace...")
    print("=" * 60)

    if HAS_DATASETS:
        dataset = load_dataset("videophysics/videophy_train_public", split="train")
        data = [dict(item) for item in dataset]
    else:
        # Fallback: download CSV directly
        print("Using fallback method (downloading CSV)...")
        import pandas as pd

        url = "https://huggingface.co/datasets/videophysics/videophy_train_public/resolve/main/videophy_train.csv"
        df = pd.read_csv(url)
        data = df.to_dict("records")

    print(f"Loaded {len(data)} samples")

    # =========================================================================
    # Step 2: Filter and analyze data
    # =========================================================================
    print("\n" + "=" * 60)
    print("Step 2: Analyzing dataset...")
    print("=" * 60)

    # Apply source filter if specified
    if args.source_filter:
        data = [d for d in data if d.get("source") == args.source_filter]
        print(f"Filtered to source '{args.source_filter}': {len(data)} samples")

    # Analyze distribution
    physics_pos = sum(1 for d in data if d.get("physics", 0) == 1)
    physics_neg = len(data) - physics_pos

    sources = {}
    for d in data:
        src = d.get("source", "unknown")
        sources[src] = sources.get(src, 0) + 1

    print(f"\nPhysics label distribution:")
    print(
        f"  Positive (physics=1): {physics_pos} ({physics_pos / len(data) * 100:.1f}%)"
    )
    print(
        f"  Negative (physics=0): {physics_neg} ({physics_neg / len(data) * 100:.1f}%)"
    )

    print(f"\nSource distribution:")
    for src, count in sorted(sources.items(), key=lambda x: -x[1]):
        print(f"  {src}: {count}")

    # =========================================================================
    # Step 3: Save metadata
    # =========================================================================
    print("\n" + "=" * 60)
    print("Step 3: Saving metadata...")
    print("=" * 60)

    # Create clean metadata with local paths
    metadata = []
    for i, d in enumerate(data):
        video_url = d.get("video_url", "")
        video_filename = get_video_filename(video_url)
        local_path = str(videos_dir / video_filename)

        metadata.append(
            {
                "index": i,
                "video_url": video_url,
                "video_path": local_path,
                "video_filename": video_filename,
                "caption": d.get("caption", ""),
                "physics": d.get("physics", 0),  # This is the label we need!
                "sa": d.get("sa", 0),
                "source": d.get("source", "unknown"),
                "states_of_matter": d.get("states_of_matter", ""),
                "complexity": d.get("complexity", 0),
            }
        )

    # Save metadata
    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to {metadata_path}")

    # Also save as CSV for easy viewing
    try:
        import pandas as pd

        df = pd.DataFrame(metadata)
        csv_path = output_dir / "metadata.csv"
        df.to_csv(csv_path, index=False)
        print(f"Saved CSV to {csv_path}")
    except ImportError:
        pass

    # Save labels separately (for training)
    labels = {m["video_filename"]: m["physics"] for m in metadata}
    labels_path = output_dir / "labels.json"
    with open(labels_path, "w") as f:
        json.dump(labels, f, indent=2)
    print(f"Saved labels to {labels_path}")

    if args.metadata_only:
        print("\n--metadata-only specified, skipping video download.")
        print(f"\nDone! Metadata saved to {output_dir}")
        return

    # =========================================================================
    # Step 4: Download videos
    # =========================================================================
    print("\n" + "=" * 60)
    print("Step 4: Downloading videos...")
    print("=" * 60)

    # Limit number of videos if specified
    download_list = metadata[: args.max_videos] if args.max_videos else metadata
    print(f"Downloading {len(download_list)} videos...")

    # Check existing
    existing = sum(1 for m in download_list if Path(m["video_path"]).exists())
    print(f"Already downloaded: {existing}")
    print(f"Remaining: {len(download_list) - existing}")

    # Download with progress bar
    failed = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {}
        for m in download_list:
            if not Path(m["video_path"]).exists():
                future = executor.submit(
                    download_video, m["video_url"], Path(m["video_path"])
                )
                futures[future] = m["video_filename"]

        if futures:
            for future in tqdm(
                as_completed(futures), total=len(futures), desc="Downloading"
            ):
                filename = futures[future]
                try:
                    success = future.result()
                    if not success:
                        failed.append(filename)
                except Exception as e:
                    failed.append(filename)
                    print(f"Error downloading {filename}: {e}")

    # Summary
    print("\n" + "=" * 60)
    print("Download Summary")
    print("=" * 60)

    downloaded = sum(1 for m in download_list if Path(m["video_path"]).exists())
    print(f"Successfully downloaded: {downloaded}/{len(download_list)}")

    if failed:
        print(f"Failed: {len(failed)}")
        failed_path = output_dir / "failed_downloads.txt"
        with open(failed_path, "w") as f:
            f.write("\n".join(failed))
        print(f"Failed list saved to {failed_path}")

    # Calculate disk usage
    total_size = sum(
        Path(m["video_path"]).stat().st_size
        for m in download_list
        if Path(m["video_path"]).exists()
    )
    print(f"Total disk usage: {total_size / 1e9:.2f} GB")

    print(f"\nDone! Dataset saved to {output_dir}")
    print(f"\nNext steps:")
    print(
        f"  1. Extract DiT features: python -m src.models.extract_features --data_dir {output_dir}"
    )
    print(
        f"  2. Train physics head: python -m trainer.train_physics_head --feature_dir ..."
    )


if __name__ == "__main__":
    main()
