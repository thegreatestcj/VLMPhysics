"""
Feature Dataset for Loading Pre-extracted (and Pooled) DiT Features

Reads metadata.json directly as the single source of truth for labels,
SA annotations, captions, and source info. No labels.json or
enriched_labels.json needed.

Supports two feature formats:
1. Original: [num_patches, D] = [17550, 1920] (~67MB per file)
2. Pooled:   [T, D] = [13, 1920] (~100KB per file) <- RECOMMENDED

Directory structure:
    feature_dir/
    ├── video_id_1/
    │   ├── t200/
    │   │   └── layer_15.pt
    │   ├── t400/
    │   └── ...
    └── video_id_2/

    metadata lives separately (e.g. videophy_data/metadata.json)

Usage:
    train_loader, val_loader = create_dataloaders(
        feature_dir="~/scratch/physics/videophy_features_pooled",
        metadata_file="~/scratch/physics/videophy_data/metadata.json",
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


def _load_metadata(metadata_file: str) -> Tuple[Dict[str, int], Dict[str, dict]]:
    """
    Build label and enriched lookups from a single metadata file.

    Supports two formats:
      - list-of-dicts  (new metadata.json from download_videophy_all.py)
      - dict {video_id: label}  (legacy labels.json, backward compatible)

    Returns:
        all_labels: {video_filename: 0/1}
        enriched:   {video_filename: {label, sa, caption, source, ...}}
    """
    with open(metadata_file, "r") as f:
        data = json.load(f)

    if isinstance(data, list):
        # New unified metadata.json format
        all_labels = {}
        enriched = {}
        for m in data:
            vid = m["video_filename"]
            all_labels[vid] = m["physics"]
            enriched[vid] = {
                "label": m["physics"],
                "sa": m.get("sa", -1),
                "caption": m.get("caption", ""),
                "source": m.get("source", "unknown"),
                "states_of_matter": m.get("states_of_matter", ""),
                "dataset_split": m.get("dataset_split", "unknown"),
            }
        logger.info(f"Loaded metadata.json: {len(all_labels)} entries")
    else:
        # Legacy labels.json: {video_id: 0/1}
        all_labels = {k: int(v) for k, v in data.items()}
        enriched = {}
        logger.info(f"Loaded legacy labels: {len(all_labels)} entries")

    return all_labels, enriched


def _find_metadata_file(feature_dir: Path, metadata_file: Optional[str]) -> str:
    """
    Resolve metadata file path with fallback chain:
      1. Explicit metadata_file argument
      2. feature_dir/metadata.json
      3. feature_dir/labels.json  (legacy)
    """
    if metadata_file is not None:
        p = Path(metadata_file)
        if p.exists():
            return str(p)
        raise FileNotFoundError(f"Metadata file not found: {metadata_file}")

    # Auto-detect in feature_dir
    for name in ["metadata.json", "labels.json"]:
        p = feature_dir / name
        if p.exists():
            logger.info(f"Auto-detected: {p}")
            return str(p)

    raise FileNotFoundError(
        f"No metadata.json or labels.json found in {feature_dir}. "
        f"Pass --metadata_file explicitly."
    )


class FeatureDataset(Dataset):
    """
    Dataset for loading pre-extracted DiT features.

    Supports both original (full spatial) and pooled (temporal only) formats.
    The pooled format is ~700x smaller and much faster to load.
    """

    def __init__(
        self,
        feature_dir: str,
        metadata_file: str,
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
            metadata_file: Path to metadata.json (single source of truth)
            layer: DiT layer to use (5, 10, 15, 20, 25)
            timesteps: List of timesteps to use (default: [200, 400, 600, 800])
            split: "train" or "val"
            train_ratio: Ratio of data for training (default: 0.85)
            is_pooled: True if features are pre-pooled [T, D]
            num_frames: Number of frames T (default: 13)
            seed: Random seed for train/val split
        """
        self.feature_dir = Path(feature_dir)
        self.layer = layer
        self.timesteps = timesteps or [200, 400, 600, 800]
        self.split = split
        self.is_pooled = is_pooled
        self.num_frames = num_frames

        # Load metadata — single source of truth
        self.all_labels, self.enriched = _load_metadata(metadata_file)

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
                feature_path = self._get_feature_path(video_id, t)
                if feature_path.exists():
                    self.samples.append((video_id, t))

        logger.info(
            f"FeatureDataset [{split}]: {len(self.samples)} samples "
            f"({len(self.video_ids)} videos x {len(self.timesteps)} timesteps), "
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

        # Pool features to [T, D] based on input shape
        ndim = features.dim()

        if ndim == 2:
            if features.shape[0] == self.num_frames:
                pass  # Already pooled: [T, D]
            else:
                # Unpooled: [num_patches, D] -> [T, D]
                num_patches, hidden_dim = features.shape
                spatial_patches = num_patches // self.num_frames
                features = features.view(self.num_frames, spatial_patches, hidden_dim)
                features = features.mean(dim=1)
        elif ndim == 3:
            features = features.mean(dim=1)  # [T, H*W, D] -> [T, D]
        elif ndim == 4:
            features = features.mean(dim=(1, 2))  # [T, H, W, D] -> [T, D]
        elif ndim == 5:
            features = features[0].mean(dim=(1, 2))  # [B, T, H, W, D] -> [T, D]

        # Get label
        label = float(self.all_labels[video_id])

        meta = self.enriched.get(video_id, {})
        sa = float(meta.get("sa", -1))  # -1 = unknown (for masking in loss)

        return {
            "features": features,
            "labels": torch.tensor(label),
            "sa": torch.tensor(sa),
            "timesteps": timestep,
            "video_id": video_id,
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
        metadata_file: str,
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

        # Load metadata — single source of truth
        self.all_labels, self.enriched = _load_metadata(metadata_file)

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

        # Pool features to [T, D] based on input shape
        ndim = features.dim()

        if ndim == 2:
            if features.shape[0] == self.num_frames:
                pass
            else:
                num_patches, hidden_dim = features.shape
                spatial_patches = num_patches // self.num_frames
                features = features.view(self.num_frames, spatial_patches, hidden_dim)
                features = features.mean(dim=1)
        elif ndim == 3:
            features = features.mean(dim=1)
        elif ndim == 4:
            features = features.mean(dim=(1, 2))
        elif ndim == 5:
            features = features[0].mean(dim=(1, 2))

        label = float(self.all_labels[video_id])

        meta = self.enriched.get(video_id, {})
        sa = float(meta.get("sa", -1))

        return {
            "features": features,
            "labels": torch.tensor(label),
            "sa": torch.tensor(sa),
            "timesteps": timestep,
            "video_id": video_id,
        }


def create_dataloaders(
    feature_dir: str,
    metadata_file: Optional[str] = None,
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
        metadata_file: Path to metadata.json.
            If None, auto-detects metadata.json or labels.json in feature_dir.
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
    feature_dir_path = Path(feature_dir)

    # Resolve metadata file with fallback chain
    resolved = _find_metadata_file(feature_dir_path, metadata_file)

    DatasetClass = FeatureDatasetRandomTimestep if random_timestep else FeatureDataset

    train_dataset = DatasetClass(
        feature_dir=str(feature_dir),
        metadata_file=resolved,
        layer=layer,
        timesteps=timesteps,
        split="train",
        train_ratio=train_ratio,
        is_pooled=is_pooled,
        seed=seed,
    )

    val_dataset = DatasetClass(
        feature_dir=str(feature_dir),
        metadata_file=resolved,
        layer=layer,
        timesteps=timesteps,
        split="val",
        train_ratio=train_ratio,
        is_pooled=is_pooled,
        seed=seed,
    )

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
        print("Usage: python -m src.data.feature_dataset <feature_dir> [metadata_file]")
        print(
            "Example: python -m src.data.feature_dataset "
            "~/scratch/physics/videophy_features_pooled "
            "~/scratch/physics/videophy_data/metadata.json"
        )
        sys.exit(1)

    feature_dir = sys.argv[1]
    metadata_file = sys.argv[2] if len(sys.argv) > 2 else None

    print(f"Testing with feature_dir: {feature_dir}")
    if metadata_file:
        print(f"  metadata_file: {metadata_file}")
    print()

    train_loader, val_loader = create_dataloaders(
        feature_dir=feature_dir,
        metadata_file=metadata_file,
        layer=15,
        batch_size=4,
        num_workers=0,
        is_pooled=True,
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print()

    batch = next(iter(train_loader))
    print("Sample batch:")
    print(f"  features shape: {batch['features'].shape}")
    print(f"  labels: {batch['labels']}")
    print(f"  sa: {batch['sa']}")
    print(f"  timesteps: {batch['timesteps']}")
    print(f"  video_ids: {batch['video_id']}")
