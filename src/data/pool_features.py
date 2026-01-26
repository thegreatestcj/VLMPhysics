#!/usr/bin/env python3
"""
Pool Extracted DiT Features for Fast Training

Converts full spatial features [num_patches, D] to temporal-pooled [T, D].
This reduces file size from ~67MB to ~100KB per file, making training 100x faster.

The output directory structure mirrors the input structure exactly:
    input:  physion_features/video_id/t200/layer_15.pt  (67MB, [17550, 1920])
    output: physion_features_pooled/video_id/t200/layer_15.pt  (100KB, [13, 1920])

This means minimal changes needed to feature_dataset.py - just point to the
new directory.

Usage:
    python -m utils.pool_features \
        --input_dir ~/scratch/physics/physion_features \
        --output_dir ~/scratch/physics/physion_features_pooled
"""

import argparse
import shutil
import logging
from pathlib import Path
from typing import List, Optional

import torch
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def pool_spatial(features: torch.Tensor, num_frames: int = 13) -> torch.Tensor:
    """
    Pool spatial dimensions, keeping temporal dimension.

    Args:
        features: Full features with shape:
            - [num_patches, D] where num_patches = T * H * W
            - [T, H, W, D]
            - [T, H*W, D]
        num_frames: Number of frames (T), default 13 for CogVideoX

    Returns:
        Temporal features [T, D]
    """
    if features.dim() == 2:
        # [num_patches, D] -> [T, spatial, D] -> [T, D]
        num_patches, hidden_dim = features.shape
        spatial_patches = num_patches // num_frames

        # Reshape: [T*H*W, D] -> [T, H*W, D]
        features = features.view(num_frames, spatial_patches, hidden_dim)

        # Pool spatial: [T, H*W, D] -> [T, D]
        pooled = features.mean(dim=1)

    elif features.dim() == 3:
        # [T, H*W, D] -> [T, D]
        pooled = features.mean(dim=1)

    elif features.dim() == 4:
        # [T, H, W, D] -> [T, D]
        pooled = features.mean(dim=(1, 2))

    else:
        raise ValueError(
            f"Unexpected feature dim: {features.dim()}, shape: {features.shape}"
        )

    return pooled


def process_single_file(
    input_path: Path, output_path: Path, num_frames: int = 13, save_fp16: bool = True
) -> bool:
    """Process a single feature file."""
    try:
        # Load full features
        features = torch.load(input_path, map_location="cpu", weights_only=True)

        # Pool spatial dimensions
        pooled = pool_spatial(features, num_frames=num_frames)

        # Convert to fp16 to save space
        if save_fp16:
            pooled = pooled.half()

        # Create output directory
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save pooled features
        torch.save(pooled, output_path)

        return True

    except Exception as e:
        logger.warning(f"Failed to process {input_path}: {e}")
        return False


def pool_features(
    input_dir: str,
    output_dir: str,
    layers: Optional[List[int]] = None,
    num_frames: int = 13,
    save_fp16: bool = True,
    skip_existing: bool = True,
):
    """
    Pool all features in input directory.

    Preserves the directory structure:
        video_id/t{timestep}/layer_{layer}.pt
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    logger.info(f"Input directory: {input_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Layers to process: {layers if layers else 'all'}")

    # Copy labels.json if exists
    for label_file in ["labels.json", "labels.csv"]:
        src = input_dir / label_file
        if src.exists():
            dst = output_dir / label_file
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(src, dst)
            logger.info(f"Copied {label_file}")

    # Find all video directories (folders directly under input_dir)
    video_dirs = [
        d
        for d in sorted(input_dir.iterdir())
        if d.is_dir() and not d.name.startswith(".")
    ]

    logger.info(f"Found {len(video_dirs)} video directories")

    # Counters
    total_files = 0
    processed_files = 0
    skipped_files = 0
    failed_files = 0

    # Process each video
    for video_dir in tqdm(video_dirs, desc="Processing videos"):
        video_id = video_dir.name

        # Iterate over timestep directories (t200, t400, t600, t800)
        for timestep_dir in sorted(video_dir.iterdir()):
            if not timestep_dir.is_dir() or not timestep_dir.name.startswith("t"):
                continue

            # Iterate over layer files
            for layer_file in sorted(timestep_dir.glob("layer_*.pt")):
                total_files += 1

                # Extract layer number
                layer_name = layer_file.stem  # "layer_15"
                layer_num = int(layer_name.split("_")[1])

                # Skip if not in requested layers
                if layers and layer_num not in layers:
                    skipped_files += 1
                    continue

                # Construct output path (same structure)
                rel_path = layer_file.relative_to(input_dir)
                output_path = output_dir / rel_path

                # Skip if already exists
                if skip_existing and output_path.exists():
                    skipped_files += 1
                    continue

                # Process file
                success = process_single_file(
                    input_path=layer_file,
                    output_path=output_path,
                    num_frames=num_frames,
                    save_fp16=save_fp16,
                )

                if success:
                    processed_files += 1
                else:
                    failed_files += 1

    # Summary
    logger.info("=" * 50)
    logger.info("Pooling complete!")
    logger.info(f"  Total files found: {total_files}")
    logger.info(f"  Processed: {processed_files}")
    logger.info(f"  Skipped: {skipped_files}")
    logger.info(f"  Failed: {failed_files}")

    # Size comparison
    if processed_files > 0:
        sample_in = list(input_dir.glob("*/t200/layer_15.pt"))[:3]
        sample_out = list(output_dir.glob("*/t200/layer_15.pt"))[:3]

        if sample_in and sample_out:
            avg_in = sum(f.stat().st_size for f in sample_in) / len(sample_in)
            avg_out = sum(f.stat().st_size for f in sample_out) / len(sample_out)
            ratio = avg_in / avg_out if avg_out > 0 else 0

            logger.info(f"  Avg input size: {avg_in / 1024 / 1024:.1f} MB")
            logger.info(f"  Avg output size: {avg_out / 1024:.1f} KB")
            logger.info(f"  Compression: {ratio:.0f}x")

    logger.info("=" * 50)


def main():
    parser = argparse.ArgumentParser(
        description="Pool DiT features for faster training"
    )

    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Path to original features directory",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Path for pooled features output"
    )
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=None,
        help="Layers to process (default: all)",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=13,
        help="Number of frames in video (default: 13)",
    )
    parser.add_argument(
        "--fp32", action="store_true", help="Save as float32 instead of float16"
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing files"
    )

    args = parser.parse_args()

    pool_features(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        layers=args.layers,
        num_frames=args.num_frames,
        save_fp16=not args.fp32,
        skip_existing=not args.overwrite,
    )


if __name__ == "__main__":
    main()
