"""
Feature Dataset for Loading Pre-extracted DiT Features

This dataset loads features that were extracted by extract_features.py.
Since features are pre-computed, training is extremely fast (~3 min for 100 epochs).

Expected directory structure:
    feature_dir/
    ├── metadata.json           # Contains layers, timesteps info
    ├── labels.json             # {video_id: 0 or 1}
    ├── pilot_dominoes_xxx/     # One folder per video
    │   ├── t200/
    │   │   └── layer_15.pt     # Feature tensor (various shapes)
    │   ├── t400/
    │   │   └── layer_15.pt
    │   └── ...
    └── test10_xxx/
        └── ...

Supported feature shapes:
    - [T, h, w, D] - Original spatial format (e.g., [13, 30, 45, 1920])
    - [1, seq, D]  - Flattened with batch dim (e.g., [1, 17550, 1920])
    - [seq, D]     - Flattened without batch dim (e.g., [17550, 1920])

Usage:
    dataset = FeatureDataset(
        feature_dir="/users/xxx/scratch/physion_features",
        layer=15,
        timesteps=[200, 400, 600, 800],
        pool_spatial=True  # Pool to [D] for classification
    )

    loader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_features)
    for batch in loader:
        features = batch['features']   # [B, D] when pooled
        labels = batch['labels']       # [B]
        timesteps = batch['timesteps'] # List[int]
"""

import os
import json
import csv
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from torch.utils.data import Dataset, DataLoader
import logging

logger = logging.getLogger(__name__)


class FeatureDataset(Dataset):
    """
    Dataset for loading pre-extracted DiT features.

    Key design decisions:
    1. Load features on-the-fly (not all in memory) - handles large datasets
    2. Support multiple timesteps - essential for noise-robust discriminator
    3. Optional spatial pooling - reduces memory, pooled features work well
    4. Timestep embedding support - model learns noise-level awareness
    5. Flexible shape handling - works with various feature formats
    """

    def __init__(
        self,
        feature_dir: str,
        label_file: Optional[str] = None,
        layer: int = 15,
        timesteps: Optional[List[int]] = None,
        pool_spatial: bool = True,
        pool_temporal: bool = False,
        split: Optional[str] = None,  # 'train', 'test', or None for all
        dtype: torch.dtype = torch.float32,
        max_samples: Optional[int] = None,  # For debugging
    ):
        """
        Args:
            feature_dir: Directory containing extracted features
            label_file: Path to labels.json (optional, will auto-detect)
            layer: Which DiT layer to use (e.g., 15)
            timesteps: List of timesteps to use. None = use all available
            pool_spatial: If True, global average pool to [D]
            pool_temporal: If True, also pool temporal (only matters if not already pooled)
            split: 'train', 'test', or None (all data)
            dtype: Data type for features (default float32)
            max_samples: Limit samples (for debugging)
        """
        self.feature_dir = Path(feature_dir)
        self.label_file = Path(label_file) if label_file else None
        self.layer = layer
        self.pool_spatial = pool_spatial
        self.pool_temporal = pool_temporal
        self.dtype = dtype

        # Load metadata
        metadata_path = self.feature_dir / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path) as f:
                self.metadata = json.load(f)
            logger.debug(
                f"Loaded metadata: layers={self.metadata.get('layers')}, "
                f"timesteps={self.metadata.get('timesteps')}"
            )
        else:
            self.metadata = {}

        # Determine timesteps
        if timesteps is None:
            self.timesteps = self.metadata.get("timesteps", [200, 400, 600, 800])
        else:
            self.timesteps = timesteps

        # Load labels
        self.labels = self._load_labels()

        # Find all video directories with extracted features
        self.video_ids = self._find_videos()

        # Apply split filter
        if split == "train":
            self.video_ids = [v for v in self.video_ids if not v.startswith("test")]
        elif split == "test":
            self.video_ids = [v for v in self.video_ids if v.startswith("test")]

        # Limit samples for debugging
        if max_samples is not None:
            self.video_ids = self.video_ids[:max_samples]

        # Create (video_id, timestep) pairs for indexing
        # Each sample is one (video, timestep) combination
        self.samples = []
        for vid in self.video_ids:
            for t in self.timesteps:
                # Only add if feature exists
                feature_path = (
                    self.feature_dir / vid / f"t{t}" / f"layer_{self.layer}.pt"
                )
                if feature_path.exists():
                    self.samples.append((vid, t))

        logger.info(
            f"FeatureDataset: {len(self.video_ids)} videos, layer={layer}, split={split}"
        )
        logger.info(
            f"Total samples: {len(self.samples)} "
            f"({len(self.video_ids)} videos × {len(self.timesteps)} timesteps)"
        )

    def _load_labels(self) -> Dict[str, int]:
        """Load labels from labels.json or labels.csv."""
        labels = {}

        # Try explicit label file first
        if self.label_file and self.label_file.exists():
            if self.label_file.suffix == ".json":
                with open(self.label_file) as f:
                    raw_labels = json.load(f)
                for vid, label in raw_labels.items():
                    if isinstance(label, bool):
                        labels[vid] = 1 if label else 0
                    else:
                        labels[vid] = int(label)
                return labels

        # Try JSON in feature dir
        json_path = self.feature_dir / "labels.json"
        if json_path.exists():
            with open(json_path) as f:
                raw_labels = json.load(f)
            for vid, label in raw_labels.items():
                if isinstance(label, bool):
                    labels[vid] = 1 if label else 0
                else:
                    labels[vid] = int(label)
            return labels

        # Try CSV in feature dir
        csv_path = self.feature_dir / "labels.csv"
        if csv_path.exists():
            with open(csv_path) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    vid = row.get("video_id") or row.get("stimulus_name")
                    label = row.get("label") or row.get("contacted_zone")
                    if vid and label is not None:
                        if isinstance(label, str):
                            label = label.lower() in ("true", "1", "yes")
                        labels[vid] = int(label)
            return labels

        # Try parent directory (Physion labels)
        parent_csv = self.feature_dir.parent / "Physion" / "labels.csv"
        if parent_csv.exists():
            with open(parent_csv) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    vid = row.get("stimulus_name", "")
                    label = row.get("contacted_zone", "")
                    if vid:
                        labels[vid] = 1 if label.lower() == "true" else 0
            return labels

        raise FileNotFoundError(
            f"No labels found. Tried:\n"
            f"  - {self.label_file}\n"
            f"  - {json_path}\n"
            f"  - {csv_path}\n"
            f"  - {parent_csv}"
        )

    def _find_videos(self) -> List[str]:
        """Find all video directories that have extracted features."""
        videos = []

        for item in self.feature_dir.iterdir():
            if not item.is_dir():
                continue
            if item.name.endswith(".json") or item.name.endswith(".csv"):
                continue

            video_id = item.name

            # Check if this video has labels
            if video_id not in self.labels:
                # Try fuzzy matching
                found_label = False
                for label_key in self.labels.keys():
                    if label_key in video_id or video_id in label_key:
                        self.labels[video_id] = self.labels[label_key]
                        found_label = True
                        break
                if not found_label:
                    continue

            # Check if features exist for at least one timestep
            has_features = False
            for t in self.timesteps:
                feature_path = item / f"t{t}" / f"layer_{self.layer}.pt"
                if feature_path.exists():
                    has_features = True
                    break

            if has_features:
                videos.append(video_id)

        return sorted(videos)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        video_id, timestep = self.samples[idx]

        # Load feature file
        feature_path = (
            self.feature_dir / video_id / f"t{timestep}" / f"layer_{self.layer}.pt"
        )

        if not feature_path.exists():
            raise FileNotFoundError(f"Feature not found: {feature_path}")

        # Load features
        features = torch.load(feature_path, map_location="cpu", weights_only=True)

        # ================================================================
        # Handle different feature shapes and restore structure:
        # ================================================================
        # Expected:  [T, h, w, D] = [13, 30, 45, 1920]  (4D spatial)
        # Common:    [1, seq, D]  = [1, 17550, 1920]    (3D with batch, flattened)
        # Also:      [seq, D]    = [17550, 1920]       (2D flattened)
        #
        # Key insight: 17550 = 13 × 30 × 45 = T × h × w
        # We can reshape back to recover temporal structure!
        # ================================================================

        # Known dimensions from CogVideoX feature extraction
        # These are typical values; adjust if your extraction used different settings
        KNOWN_T = 13  # Number of frames (temporal)
        KNOWN_H = 30  # Height after patchification
        KNOWN_W = 45  # Width after patchification
        KNOWN_SEQ = KNOWN_T * KNOWN_H * KNOWN_W  # 17550

        # Step 1: Remove spurious batch dimension if present
        if features.dim() == 3 and features.shape[0] == 1:
            # [1, seq, D] -> [seq, D]
            features = features.squeeze(0)

        # Step 2: Try to restore [T, h, w, D] structure from flattened features
        if features.dim() == 2:
            seq_len, D = features.shape

            # Check if this matches known flattened sequence length
            if seq_len == KNOWN_SEQ:
                # Reshape back to [T, h, w, D]
                features = features.reshape(KNOWN_T, KNOWN_H, KNOWN_W, D)
            elif seq_len % KNOWN_T == 0:
                # Sequence length is divisible by T, assume [T, spatial, D]
                spatial = seq_len // KNOWN_T
                features = features.reshape(KNOWN_T, spatial, D)

        # Step 3: Apply pooling based on current shape
        if self.pool_spatial:
            if features.dim() == 4:
                # [T, h, w, D] -> [T, D] (pool only spatial, keep temporal!)
                features = features.mean(dim=(1, 2))
            elif features.dim() == 3:
                # [T, spatial, D] -> [T, D]
                features = features.mean(dim=1)
            elif features.dim() == 2:
                # Already [T, D] or [seq, D] - check if we should pool
                if features.shape[0] > 100:  # Likely still flattened, pool everything
                    features = features.mean(dim=0)
                # Otherwise keep as [T, D]
            # dim == 1: already [D], no change
        else:
            # No spatial pooling - keep full resolution
            if features.dim() == 4:
                # [T, h, w, D] -> [T*h*w, D] for sequence models
                T, h, w, D = features.shape
                features = features.reshape(T * h * w, D)
            # [seq, D] is fine as-is

        # Step 4: Apply temporal pooling if requested
        if self.pool_temporal and features.dim() >= 2:
            # [T, D] -> [D] or [T, h, w, D] -> [D]
            features = features.mean(dim=tuple(range(features.dim() - 1)))

        # Convert to target dtype
        features = features.to(self.dtype)

        # Get label
        label = self.labels.get(video_id, 0)

        return {
            "video_id": video_id,
            "features": features,
            "labels": torch.tensor(label, dtype=torch.float32),  # Note: plural 'labels'
            "timesteps": timestep,  # Return as int, collate will handle
        }


def collate_features(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function for FeatureDataset.

    Handles variable-length sequences by padding or stacking.

    Returns:
        video_ids: List[str]
        features: [B, ...] tensor (stacked if same shape, padded if different)
        labels: [B] tensor
        timesteps: List[int]
    """
    video_ids = [item["video_id"] for item in batch]
    labels = torch.stack([item["labels"] for item in batch])
    timesteps = [item["timesteps"] for item in batch]  # Keep as list of ints

    # Stack features (assumes same shape within batch)
    features = torch.stack([item["features"] for item in batch])

    return {
        "video_ids": video_ids,
        "features": features,
        "labels": labels,
        "timesteps": timesteps,
    }


class MultiTimestepFeatureDataset(Dataset):
    """
    Dataset that returns all timesteps for a video in one sample.

    Useful for models that need to see features across multiple noise levels.

    Returns:
        features: [num_timesteps, D] (if pooled)
        labels: scalar
        timesteps: [num_timesteps]
    """

    def __init__(
        self,
        feature_dir: str,
        label_file: Optional[str] = None,
        layer: int = 15,
        timesteps: List[int] = [200, 400, 600, 800],
        pool_spatial: bool = True,
        pool_temporal: bool = True,
        split: Optional[str] = None,
        dtype: torch.dtype = torch.float32,
        max_samples: Optional[int] = None,
    ):
        self.feature_dir = Path(feature_dir)
        self.label_file = Path(label_file) if label_file else None
        self.layer = layer
        self.timesteps = sorted(timesteps)
        self.pool_spatial = pool_spatial
        self.pool_temporal = pool_temporal
        self.dtype = dtype

        # Load labels
        self.labels = self._load_labels()

        # Find videos
        self.video_ids = self._find_videos()

        # Apply split filter
        if split == "train":
            self.video_ids = [v for v in self.video_ids if not v.startswith("test")]
        elif split == "test":
            self.video_ids = [v for v in self.video_ids if v.startswith("test")]

        if max_samples:
            self.video_ids = self.video_ids[:max_samples]

        logger.info(
            f"MultiTimestepDataset: {len(self.video_ids)} videos, "
            f"{len(self.timesteps)} timesteps each"
        )

    def _load_labels(self) -> Dict[str, int]:
        """Load labels from various sources."""
        labels = {}

        # Try explicit label file
        if self.label_file and self.label_file.exists():
            with open(self.label_file) as f:
                raw = json.load(f)
            for vid, label in raw.items():
                labels[vid] = (
                    int(label) if not isinstance(label, bool) else (1 if label else 0)
                )
            return labels

        # Try feature_dir/labels.json
        json_path = self.feature_dir / "labels.json"
        if json_path.exists():
            with open(json_path) as f:
                raw = json.load(f)
            for vid, label in raw.items():
                labels[vid] = (
                    int(label) if not isinstance(label, bool) else (1 if label else 0)
                )
            return labels

        return labels

    def _find_videos(self) -> List[str]:
        """Find videos that have ALL required timesteps."""
        videos = []
        for item in self.feature_dir.iterdir():
            if not item.is_dir() or item.name.endswith(".json"):
                continue

            video_id = item.name
            if video_id not in self.labels:
                continue

            # Must have all timesteps
            has_all = all(
                (item / f"t{t}" / f"layer_{self.layer}.pt").exists()
                for t in self.timesteps
            )
            if has_all:
                videos.append(video_id)

        return sorted(videos)

    def __len__(self) -> int:
        return len(self.video_ids)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        video_id = self.video_ids[idx]

        # Known dimensions from CogVideoX feature extraction
        KNOWN_T = 13
        KNOWN_H = 30
        KNOWN_W = 45
        KNOWN_SEQ = KNOWN_T * KNOWN_H * KNOWN_W  # 17550

        features_list = []
        for t in self.timesteps:
            path = self.feature_dir / video_id / f"t{t}" / f"layer_{self.layer}.pt"
            feat = torch.load(path, map_location="cpu", weights_only=True)

            # Handle various shapes - same logic as FeatureDataset
            if feat.dim() == 3 and feat.shape[0] == 1:
                feat = feat.squeeze(0)  # [1, seq, D] -> [seq, D]

            # Try to restore structure
            if feat.dim() == 2:
                seq_len, D = feat.shape
                if seq_len == KNOWN_SEQ:
                    feat = feat.reshape(KNOWN_T, KNOWN_H, KNOWN_W, D)

            if self.pool_spatial:
                if feat.dim() == 4:
                    feat = feat.mean(dim=(1, 2))  # [T, h, w, D] -> [T, D]
                elif feat.dim() == 2 and feat.shape[0] > 100:
                    feat = feat.mean(dim=0)  # [seq, D] -> [D]

            if self.pool_temporal:
                if feat.dim() >= 2:
                    feat = feat.mean(dim=tuple(range(feat.dim() - 1)))  # -> [D]

            features_list.append(feat.to(self.dtype))

        # Stack: [num_timesteps, T, D] or [num_timesteps, D]
        features = torch.stack(features_list, dim=0)

        return {
            "video_id": video_id,
            "features": features,
            "labels": torch.tensor(self.labels[video_id], dtype=torch.float32),
            "timesteps": torch.tensor(self.timesteps, dtype=torch.long),
        }


def create_dataloaders(
    feature_dir: str,
    label_file: Optional[str] = None,
    layer: int = 15,
    timesteps: List[int] = [200, 400, 600, 800],
    batch_size: int = 32,
    num_workers: int = 4,
    pool_spatial: bool = True,
    pool_temporal: bool = False,
    multi_timestep: bool = False,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders.

    Args:
        feature_dir: Directory with extracted features
        label_file: Path to labels.json
        layer: DiT layer to use
        timesteps: List of diffusion timesteps
        batch_size: Batch size
        num_workers: DataLoader workers
        pool_spatial: Pool spatial dimensions
        pool_temporal: Pool temporal dimension
        multi_timestep: If True, use MultiTimestepFeatureDataset
        val_ratio: Validation split ratio
        seed: Random seed

    Returns:
        (train_loader, val_loader)
    """
    DatasetClass = MultiTimestepFeatureDataset if multi_timestep else FeatureDataset

    # Create full dataset first, then split
    full_dataset = DatasetClass(
        feature_dir=feature_dir,
        label_file=label_file,
        layer=layer,
        timesteps=timesteps,
        pool_spatial=pool_spatial,
        pool_temporal=pool_temporal,
        split="train",  # Use train split (excludes test videos)
    )

    # Manual train/val split
    n_samples = len(full_dataset)
    indices = list(range(n_samples))

    rng = np.random.RandomState(seed)
    rng.shuffle(indices)

    n_val = int(n_samples * val_ratio)
    train_indices = indices[n_val:]
    val_indices = indices[:n_val]

    from torch.utils.data import Subset

    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_features,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_features,
    )

    return train_loader, val_loader


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--feature_dir", type=str, required=True)
    parser.add_argument("--layer", type=int, default=15)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    print("=" * 60)
    print("Testing FeatureDataset")
    print("=" * 60)

    dataset = FeatureDataset(
        feature_dir=args.feature_dir,
        layer=args.layer,
        pool_spatial=True,
        max_samples=10,
    )

    print(f"\nDataset size: {len(dataset)}")

    sample = dataset[0]
    print(f"Sample keys: {sample.keys()}")
    print(f"Features shape: {sample['features'].shape}")
    print(f"Features dtype: {sample['features'].dtype}")
    print(f"Features mean: {sample['features'].mean():.4f}")
    print(f"Features std: {sample['features'].std():.4f}")
    print(f"Label: {sample['labels']}")
    print(f"Timestep: {sample['timesteps']}")
    print(f"Video ID: {sample['video_id']}")

    print("\n" + "=" * 60)
    print("Testing collate_features")
    print("=" * 60)

    batch = [dataset[i] for i in range(min(4, len(dataset)))]
    collated = collate_features(batch)
    print(f"Batch features shape: {collated['features'].shape}")
    print(f"Batch labels shape: {collated['labels'].shape}")
    print(f"Batch timesteps: {collated['timesteps']}")

    print("\n" + "=" * 60)
    print("Testing DataLoader")
    print("=" * 60)

    train_loader, val_loader = create_dataloaders(
        feature_dir=args.feature_dir,
        layer=args.layer,
        batch_size=4,
        num_workers=0,
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    for batch in train_loader:
        print(f"\nBatch features: {batch['features'].shape}")
        print(f"Batch labels: {batch['labels'].shape}")
        print(f"Batch timesteps: {batch['timesteps']}")
        break

    print("\n✓ All tests passed!")
