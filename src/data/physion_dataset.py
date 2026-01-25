"""
Physion Dataset Loader for Physics Discriminator Training

Loads videos from Physion test set with OCP (Object Contact Prediction) labels.
Label: True = red object contacts yellow object, False = no contact
"""

import os
import csv
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional
import numpy as np

# For video loading
try:
    import decord
    from decord import VideoReader, cpu

    decord.bridge.set_bridge("torch")
    USE_DECORD = True
except ImportError:
    USE_DECORD = False
    print("Warning: decord not found, falling back to imageio")
    import imageio


class PhysionDataset(Dataset):
    """
    Physion dataset for physics discriminator training.

    Args:
        data_root: Path to Physion directory (containing labels.csv and scenario folders)
        split: 'train' or 'val' (will split based on ratio)
        train_ratio: Fraction of data for training (default 0.85)
        video_dir: Which video directory to use ('mp4s-redyellow' for model, 'mp4s' for human)
        num_frames: Number of frames to sample from each video
        seed: Random seed for reproducible splits
    """

    def __init__(
        self,
        data_root: str,
        split: str = "train",
        train_ratio: float = 0.85,
        video_dir: str = "mp4s-redyellow",  # Use red-yellow marked videos for model
        num_frames: int = 49,  # CogVideoX default
        seed: int = 42,
    ):
        self.data_root = data_root
        self.split = split
        self.video_dir = video_dir
        self.num_frames = num_frames

        # Load labels
        self.samples = self._load_labels()

        # Train/val split
        np.random.seed(seed)
        indices = np.random.permutation(len(self.samples))
        split_idx = int(len(indices) * train_ratio)

        if split == "train":
            self.indices = indices[:split_idx]
        else:
            self.indices = indices[split_idx:]

        print(f"PhysionDataset [{split}]: {len(self.indices)} samples")
        self._print_label_distribution()

    def _load_labels(self) -> List[Dict]:
        """Load labels.csv and match with video files."""
        labels_path = os.path.join(self.data_root, "labels.csv")
        samples = []

        with open(labels_path, "r") as f:
            reader = csv.reader(f)
            next(reader)  # Skip header: ",ground truth outcome"

            for row in reader:
                if len(row) < 2:
                    continue

                video_name = row[0].strip()
                label_str = row[1].strip()
                label = 1 if label_str == "True" else 0

                # Parse scenario from video name (e.g., "pilot_dominoes_..." -> "Dominoes")
                scenario = self._parse_scenario(video_name)

                # Find video file
                video_path = self._find_video_path(scenario, video_name)

                if video_path and os.path.exists(video_path):
                    samples.append(
                        {
                            "video_name": video_name,
                            "video_path": video_path,
                            "label": label,
                            "scenario": scenario,
                        }
                    )

        print(f"Loaded {len(samples)} samples from labels.csv")
        return samples

    def _parse_scenario(self, video_name: str) -> str:
        """Extract scenario name from video filename."""
        # Format: pilot_<scenario>_... or test_<scenario>_...
        scenarios = [
            "collide",
            "contain",
            "dominoes",
            "drape",
            "drop",
            "link",
            "roll",
            "support",
        ]
        video_name_lower = video_name.lower()

        for scenario in scenarios:
            if scenario in video_name_lower:
                # Return with proper capitalization for folder name
                return scenario.capitalize()

        return "Unknown"

    def _find_video_path(self, scenario: str, video_name: str) -> Optional[str]:
        """Find the video file path."""
        # Try different naming patterns
        possible_paths = [
            os.path.join(self.data_root, scenario, self.video_dir, f"{video_name}.mp4"),
            os.path.join(
                self.data_root, scenario, self.video_dir, f"{video_name}_img.mp4"
            ),
            os.path.join(
                self.data_root, scenario.lower(), self.video_dir, f"{video_name}.mp4"
            ),
        ]

        for path in possible_paths:
            if os.path.exists(path):
                return path

        # Search in directory
        scenario_dir = os.path.join(self.data_root, scenario, self.video_dir)
        if os.path.exists(scenario_dir):
            for fname in os.listdir(scenario_dir):
                if video_name in fname and fname.endswith(".mp4"):
                    return os.path.join(scenario_dir, fname)

        return None

    def _print_label_distribution(self):
        """Print label distribution for current split."""
        labels = [self.samples[i]["label"] for i in self.indices]
        pos = sum(labels)
        neg = len(labels) - pos
        print(
            f"  Label distribution: {pos} positive ({pos / len(labels) * 100:.1f}%), "
            f"{neg} negative ({neg / len(labels) * 100:.1f}%)"
        )

    def _load_video(self, video_path: str) -> torch.Tensor:
        """Load video frames."""
        if USE_DECORD:
            vr = VideoReader(video_path, ctx=cpu(0))
            total_frames = len(vr)

            # Sample frames uniformly
            if total_frames >= self.num_frames:
                indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
            else:
                # If video is shorter, repeat last frame
                indices = list(range(total_frames))
                indices += [total_frames - 1] * (self.num_frames - total_frames)

            frames = vr.get_batch(indices)  # [T, H, W, C]
            frames = frames.permute(0, 3, 1, 2)  # [T, C, H, W]
        else:
            # Fallback to imageio
            reader = imageio.get_reader(video_path)
            frames = []
            for frame in reader:
                frames.append(torch.from_numpy(frame).permute(2, 0, 1))
            reader.close()

            frames = torch.stack(frames)  # [T, C, H, W]

            # Sample frames
            total_frames = len(frames)
            if total_frames >= self.num_frames:
                indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
                frames = frames[indices]
            else:
                # Pad with last frame
                padding = frames[-1:].repeat(self.num_frames - total_frames, 1, 1, 1)
                frames = torch.cat([frames, padding], dim=0)

        # Normalize to [0, 1]
        frames = frames.float() / 255.0

        return frames

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[self.indices[idx]]

        # Load video
        video = self._load_video(sample["video_path"])

        return {
            "video": video,  # [T, C, H, W]
            "label": sample["label"],
            "scenario": sample["scenario"],
            "video_name": sample["video_name"],
        }


def get_dataloaders(
    data_root: str, batch_size: int = 4, num_workers: int = 4, **dataset_kwargs
) -> Tuple[DataLoader, DataLoader]:
    """Get train and validation dataloaders."""

    train_dataset = PhysionDataset(data_root, split="train", **dataset_kwargs)
    val_dataset = PhysionDataset(data_root, split="val", **dataset_kwargs)

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

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="data/Physion")
    parser.add_argument("--num_frames", type=int, default=49)
    args = parser.parse_args()

    print("=" * 50)
    print("Testing PhysionDataset")
    print("=" * 50)

    # Test dataset
    dataset = PhysionDataset(
        data_root=args.data_root, split="train", num_frames=args.num_frames
    )

    print(f"\nTotal samples: {len(dataset)}")

    # Test loading one sample
    print("\nLoading first sample...")
    sample = dataset[0]
    print(f"  Video shape: {sample['video'].shape}")
    print(f"  Label: {sample['label']}")
    print(f"  Scenario: {sample['scenario']}")
    print(f"  Video name: {sample['video_name']}")

    # Test dataloader
    print("\nTesting DataLoader...")
    train_loader, val_loader = get_dataloaders(
        args.data_root,
        batch_size=2,
        num_workers=0,  # 0 for testing
        num_frames=args.num_frames,
    )

    batch = next(iter(train_loader))
    print(f"  Batch video shape: {batch['video'].shape}")
    print(f"  Batch labels: {batch['label']}")

    print("\n" + "=" * 50)
    print("Dataset test passed!")
    print("=" * 50)
