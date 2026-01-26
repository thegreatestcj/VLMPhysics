"""
Feature Dataset for Loading Pre-extracted (and Pooled) DiT Features

Supports two feature formats:
1. Original: [num_patches, D] = [17550, 1920] (~67MB per file)
2. Pooled:   [T, D] = [13, 1920] (~100KB per file) <- RECOMMENDED

Directory structure (same for both formats):
    feature_dir/
    ├── video_id_1/
    │   ├── t200/
    │   │   ├── layer_5.pt
    │   │   ├── layer_10.pt
    │   │   └── layer_15.pt
    │   ├── t400/
    │   └── ...
    ├── video_id_2/
    └── labels.json

Usage:
    # For pooled features (fast training)
    dataset = FeatureDataset(
        feature_dir="~/scratch/physics/physion_features_pooled",
        label_file="~/scratch/physics/physion_features_pooled/labels.json",
        layer=15,
        is_pooled=True  # Set this to True for pooled features
    )

    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        feature_dir="~/scratch/physics/physion_features_pooled",
        layer=15,
        batch_size=32,
        is_pooled=True
    )
"""

import json
import random
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureDataset(Dataset):
    """
    Dataset for loading pre-extracted DiT features.

    Supports both original (full spatial) and pooled (temporal only) formats.
    The pooled format is ~700x smaller and much faster to load.
    """

    def __init__(
        self,
        feature_dir: str,
        label_file: str,
        layer: int = 15,
        timesteps: Optional[List[int]] = None,
        split: str = "train",
        train_ratio: float = 0.85,
        is_pooled: bool = True,
        num_frames: int = 13,
        seed: int = 42,
    ):
        """
        Args:
            feature_dir: Path to features directory
            label_file: Path to labels.json
            layer: DiT layer to use (5, 10, 15, 20, 25)
            timesteps: List of timesteps to use (default: [200, 400, 600, 800])
            split: "train" or "val"
            train_ratio: Ratio of data for training (default: 0.85)
            is_pooled: True if features are pre-pooled [T, D], False for original [num_patches, D]
            num_frames: Number of frames T (default: 13, only used if is_pooled=False)
            seed: Random seed for train/val split
        """
        self.feature_dir = Path(feature_dir)
        self.layer = layer
        self.timesteps = timesteps or [200, 400, 600, 800]
        self.split = split
        self.is_pooled = is_pooled
        self.num_frames = num_frames

        # Load labels
        with open(label_file, "r") as f:
            self.all_labels = json.load(f)

        # Find available videos by scanning directory
        self.video_ids = self._find_available_videos()

        # Train/val split
        random.seed(seed)
        shuffled = self.video_ids.copy()
        random.shuffle(shuffled)

        split_idx = int(len(shuffled) * train_ratio)
        if split == "train":
            self.video_ids = shuffled[:split_idx]
        else:
            self.video_ids = shuffled[split_idx:]

        # Build sample list: (video_id, timestep) pairs
        self.samples = []
        for video_id in self.video_ids:
            for t in self.timesteps:
                # Check if feature file exists
                feature_path = self._get_feature_path(video_id, t)
                if feature_path.exists():
                    self.samples.append((video_id, t))

        logger.info(
            f"FeatureDataset [{split}]: {len(self.samples)} samples "
            f"({len(self.video_ids)} videos × {len(self.timesteps)} timesteps), "
            f"layer={layer}, is_pooled={is_pooled}"
        )

    def _find_available_videos(self) -> List[str]:
        """Find all video IDs that have features and labels."""
        video_ids = []

        for video_dir in sorted(self.feature_dir.iterdir()):
            if not video_dir.is_dir():
                continue

            video_id = video_dir.name

            # Check if video has label
            if video_id not in self.all_labels:
                continue

            # Check if at least one timestep exists
            has_features = False
            for t in self.timesteps:
                feature_path = self._get_feature_path(video_id, t)
                if feature_path.exists():
                    has_features = True
                    break

            if has_features:
                video_ids.append(video_id)

        return video_ids

    def _get_feature_path(self, video_id: str, timestep: int) -> Path:
        """Get path to feature file."""
        return self.feature_dir / video_id / f"t{timestep}" / f"layer_{self.layer}.pt"

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        video_id, timestep = self.samples[idx]

        # Load features
        feature_path = self._get_feature_path(video_id, timestep)
        features = torch.load(feature_path, map_location="cpu", weights_only=True)

        # Ensure float32 for training
        features = features.float()

        # Handle shape based on format
        if self.is_pooled:
            # Already pooled: [T, D] = [13, 1920]
            # No additional processing needed
            pass
        else:
            # Original format: [num_patches, D] = [17550, 1920]
            # Need to pool spatial dimensions to get [T, D]
            if features.dim() == 2:
                num_patches, hidden_dim = features.shape
                spatial_patches = num_patches // self.num_frames
                features = features.view(self.num_frames, spatial_patches, hidden_dim)
                features = features.mean(dim=1)  # [T, D]

        # Get label
        label = float(self.all_labels[video_id])

        return {
            "features": features,  # [T, D] = [13, 1920]
            "label": torch.tensor(label),  # scalar
            "timestep": torch.tensor(timestep),  # scalar (200, 400, 600, 800)
            "video_id": video_id,  # string
        }


class FeatureDatasetRandomTimestep(Dataset):
    """
    Dataset that randomly samples one timestep per video per epoch.

    This is useful for training when you want each video to appear
    once per epoch with a random timestep (data augmentation).
    """

    def __init__(
        self,
        feature_dir: str,
        label_file: str,
        layer: int = 15,
        timesteps: Optional[List[int]] = None,
        split: str = "train",
        train_ratio: float = 0.85,
        is_pooled: bool = True,
        num_frames: int = 13,
        seed: int = 42,
    ):
        self.feature_dir = Path(feature_dir)
        self.layer = layer
        self.timesteps = timesteps or [200, 400, 600, 800]
        self.split = split
        self.is_pooled = is_pooled
        self.num_frames = num_frames

        # Load labels
        with open(label_file, "r") as f:
            self.all_labels = json.load(f)

        # Find available videos
        self.video_ids = self._find_available_videos()

        # Train/val split
        random.seed(seed)
        shuffled = self.video_ids.copy()
        random.shuffle(shuffled)

        split_idx = int(len(shuffled) * train_ratio)
        if split == "train":
            self.video_ids = shuffled[:split_idx]
        else:
            self.video_ids = shuffled[split_idx:]

        # Store available timesteps per video
        self.video_timesteps = {}
        for video_id in self.video_ids:
            available = []
            for t in self.timesteps:
                if self._get_feature_path(video_id, t).exists():
                    available.append(t)
            self.video_timesteps[video_id] = available

        logger.info(
            f"FeatureDatasetRandomTimestep [{split}]: {len(self.video_ids)} videos, "
            f"layer={layer}"
        )

    def _find_available_videos(self) -> List[str]:
        """Find all video IDs that have features and labels."""
        video_ids = []

        for video_dir in sorted(self.feature_dir.iterdir()):
            if not video_dir.is_dir():
                continue

            video_id = video_dir.name
            if video_id in self.all_labels:
                video_ids.append(video_id)

        return video_ids

    def _get_feature_path(self, video_id: str, timestep: int) -> Path:
        return self.feature_dir / video_id / f"t{timestep}" / f"layer_{self.layer}.pt"

    def __len__(self) -> int:
        return len(self.video_ids)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        video_id = self.video_ids[idx]

        # Randomly select timestep
        timestep = random.choice(self.video_timesteps[video_id])

        # Load features
        feature_path = self._get_feature_path(video_id, timestep)
        features = torch.load(feature_path, map_location="cpu", weights_only=True)
        features = features.float()

        # Pool if needed
        if not self.is_pooled and features.dim() == 2:
            num_patches, hidden_dim = features.shape
            spatial_patches = num_patches // self.num_frames
            features = features.view(self.num_frames, spatial_patches, hidden_dim)
            features = features.mean(dim=1)

        label = float(self.all_labels[video_id])

        return {
            "features": features,
            "label": torch.tensor(label),
            "timestep": torch.tensor(timestep),
            "video_id": video_id,
        }


def create_dataloaders(
    feature_dir: str,
    label_file: Optional[str] = None,
    layer: int = 15,
    timesteps: Optional[List[int]] = None,
    batch_size: int = 32,
    num_workers: int = 4,
    is_pooled: bool = True,
    random_timestep: bool = False,
    train_ratio: float = 0.85,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and val dataloaders.

    Args:
        feature_dir: Path to features directory
        label_file: Path to labels.json (default: feature_dir/labels.json)
        layer: DiT layer to use
        timesteps: List of timesteps
        batch_size: Batch size
        num_workers: Number of dataloader workers
        is_pooled: Whether features are pre-pooled
        random_timestep: If True, randomly sample one timestep per video per epoch
        train_ratio: Ratio of data for training
        seed: Random seed

    Returns:
        (train_loader, val_loader)
    """
    feature_dir = Path(feature_dir)

    # Find labels file
    if label_file is None:
        label_file = feature_dir / "labels.json"

    # Choose dataset class
    DatasetClass = FeatureDatasetRandomTimestep if random_timestep else FeatureDataset

    # Create datasets
    train_dataset = DatasetClass(
        feature_dir=str(feature_dir),
        label_file=str(label_file),
        layer=layer,
        timesteps=timesteps,
        split="train",
        train_ratio=train_ratio,
        is_pooled=is_pooled,
        seed=seed,
    )

    val_dataset = DatasetClass(
        feature_dir=str(feature_dir),
        label_file=str(label_file),
        layer=layer,
        timesteps=timesteps,
        split="val",
        train_ratio=train_ratio,
        is_pooled=is_pooled,
        seed=seed,
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m src.data.feature_dataset <feature_dir>")
        print(
            "Example: python -m src.data.feature_dataset ~/scratch/physics/physion_features_pooled"
        )
        sys.exit(1)

    feature_dir = sys.argv[1]
    label_file = Path(feature_dir) / "labels.json"

    print(f"Testing with feature_dir: {feature_dir}")
    print()

    # Test FeatureDataset
    train_loader, val_loader = create_dataloaders(
        feature_dir=feature_dir,
        label_file=str(label_file),
        layer=15,
        batch_size=4,
        num_workers=0,
        is_pooled=True,
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print()

    # Get a sample batch
    batch = next(iter(train_loader))
    print("Sample batch:")
    print(f"  features shape: {batch['features'].shape}")
    print(f"  labels: {batch['label']}")
    print(f"  timesteps: {batch['timestep']}")
    print(f"  video_ids: {batch['video_id']}")
