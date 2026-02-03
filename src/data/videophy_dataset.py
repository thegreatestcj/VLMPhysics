#!/usr/bin/env python3
"""
VideoPhy Dataset Loader for Physics Head Training

This loader reads VideoPhy videos and prepares them for DiT feature extraction
and physics head training.

Usage:
    from src.data.videophy_dataset import VideoPhyDataset

    dataset = VideoPhyDataset(
        data_root="data/videophy",
        split="train",
    )
"""

import os
import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
from torch.utils.data import Dataset, DataLoader

try:
    import torchvision.io as tvio
except ImportError:
    tvio = None

logger = logging.getLogger(__name__)


class VideoPhyDataset(Dataset):
    """
    Dataset for loading VideoPhy videos with physics labels.

    Labels:
        physics=1: Video follows physical commonsense
        physics=0: Video violates physical commonsense
    """

    def __init__(
        self,
        data_root: str,
        split: str = "all",
        num_frames: int = 49,
        frame_size: Tuple[int, int] = (480, 720),
        val_ratio: float = 0.1,
        seed: int = 42,
        source_filter: Optional[List[str]] = None,
        max_samples: Optional[int] = None,
    ):
        """
        Args:
            data_root: Path to VideoPhy data folder
            split: "train", "val", or "all"
            num_frames: Number of frames to sample (CogVideoX uses 49)
            frame_size: Target frame size (H, W)
            val_ratio: Fraction for validation split
            seed: Random seed for reproducibility
            source_filter: Filter by video source (e.g., ["lavie", "pika"])
            max_samples: Maximum number of samples to load
        """
        self.data_root = Path(data_root)
        self.split = split
        self.num_frames = num_frames
        self.frame_size = frame_size
        self.val_ratio = val_ratio
        self.seed = seed

        # Load metadata
        self.samples = []  # List of (video_path, label, metadata)
        self._load_metadata(source_filter)

        # Limit samples if specified
        if max_samples and len(self.samples) > max_samples:
            self.samples = self.samples[:max_samples]

        # Apply train/val split
        self._apply_split()

        logger.info(f"VideoPhyDataset [{split}]: {len(self.samples)} samples")
        self._print_label_distribution()

    def _load_metadata(self, source_filter: Optional[List[str]] = None):
        """Load metadata.json and filter samples."""
        metadata_path = self.data_root / "metadata.json"

        if not metadata_path.exists():
            raise FileNotFoundError(
                f"Metadata not found: {metadata_path}\n"
                f"Run: python download_videophy.py --output-dir {self.data_root}"
            )

        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        found = 0
        missing = 0

        for item in metadata:
            video_path = Path(item["video_path"])

            # Check if video exists
            if not video_path.exists():
                missing += 1
                continue

            # Apply source filter
            if source_filter and item.get("source") not in source_filter:
                continue

            self.samples.append(
                (
                    video_path,
                    item["physics"],  # 0 or 1
                    {
                        "caption": item.get("caption", ""),
                        "source": item.get("source", "unknown"),
                        "sa": item.get("sa", 0),
                        "states_of_matter": item.get("states_of_matter", ""),
                        "video_filename": item.get("video_filename", ""),
                    },
                )
            )
            found += 1

        logger.info(f"Loaded {found} samples ({missing} videos missing)")

    def _apply_split(self):
        """Apply train/val split."""
        if self.split == "all":
            return

        rng = np.random.RandomState(self.seed)
        indices = np.arange(len(self.samples))
        rng.shuffle(indices)

        n_val = int(len(self.samples) * self.val_ratio)

        if self.split == "train":
            selected = indices[n_val:]
        elif self.split == "val":
            selected = indices[:n_val]
        else:
            return

        self.samples = [self.samples[i] for i in selected]

    def _print_label_distribution(self):
        """Print label distribution."""
        if not self.samples:
            logger.warning("No samples loaded!")
            return

        labels = [s[1] for s in self.samples]
        pos = sum(labels)
        neg = len(labels) - pos
        logger.info(
            f"  Physics label: {pos} positive ({pos / len(labels) * 100:.1f}%), "
            f"{neg} negative ({neg / len(labels) * 100:.1f}%)"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def _load_video(self, video_path: Path) -> torch.Tensor:
        """Load and preprocess video frames."""
        if tvio is None:
            raise ImportError("torchvision is required")

        video, _, _ = tvio.read_video(str(video_path), pts_unit="sec")
        # video: [T, H, W, C] uint8

        total_frames = video.shape[0]

        indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
        video = video[indices]  # [num_frames, H, W, C]

        # Convert to float [0, 1]
        video = video.float() / 255.0

        # Resize if needed
        if video.shape[1:3] != self.frame_size:
            video = video.permute(0, 3, 1, 2)  # [T, C, H, W]
            video = torch.nn.functional.interpolate(
                video,
                size=self.frame_size,
                mode="bilinear",
                align_corners=False,
            )
            video = video.permute(0, 2, 3, 1)  # [T, H, W, C]

        # Final: [T, C, H, W]
        video = video.permute(0, 3, 1, 2)

        return video

    def __getitem__(self, idx: int) -> Dict:
        video_path, label, meta = self.samples[idx]

        video = self._load_video(video_path)

        return {
            "video": video,  # [T, C, H, W]
            "label": label,  # 0 or 1
            "video_name": meta["video_filename"],
            "source": meta.get("source", "unknown"),
            "caption": meta.get("caption", ""),
        }


def get_dataloaders(
    data_root: str,
    batch_size: int = 1,
    num_workers: int = 4,
    num_frames: int = 49,
    **dataset_kwargs,
) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation dataloaders."""
    train_dataset = VideoPhyDataset(
        data_root, split="train", num_frames=num_frames, **dataset_kwargs
    )
    val_dataset = VideoPhyDataset(
        data_root, split="val", num_frames=num_frames, **dataset_kwargs
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader


# ============ Test Code ============
if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="data/videophy")
    args = parser.parse_args()

    print("=" * 60)
    print("Testing VideoPhyDataset")
    print("=" * 60)

    dataset = VideoPhyDataset(
        data_root=args.data_root,
        split="all",
        num_frames=49,
    )

    print(f"\nTotal samples: {len(dataset)}")

    # Show source distribution
    sources = {}
    for _, _, meta in dataset.samples:
        src = meta["source"]
        sources[src] = sources.get(src, 0) + 1

    print("\nSamples per source:")
    for src, count in sorted(sources.items(), key=lambda x: -x[1]):
        print(f"  {src}: {count}")

    if len(dataset) > 0:
        print("\nLoading first sample...")
        sample = dataset[0]
        print(f"  Video shape: {sample['video'].shape}")
        print(
            f"  Label: {sample['label']} ({'physical' if sample['label'] else 'non-physical'})"
        )
        print(f"  Source: {sample['source']}")
        print(f"  Caption: {sample['caption'][:80]}...")

    print("\n" + "=" * 60)
    print("Test complete!")
    print("=" * 60)
