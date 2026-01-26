"""
Feature Dataset for loading pre-extracted DiT features.

This is used AFTER running extract_features.py.
Training with this dataset is fast since no DiT forward is needed!

Directory structure:
    data/Physion/features/
    ├── video_001/
    │   ├── t200/
    │   │   └── layer_15.pt    # [T, h, w, D] = [13, 30, 45, 1920]
    │   ├── t400/
    │   └── t800/
    ├── video_002/
    └── metadata.json

Usage:
    dataset = FeatureDataset(
        feature_dir="data/Physion/features",
        label_file="data/Physion/labels.json",
        layer=15,
        timesteps=[200, 400, 600, 800]
    )

    for batch in DataLoader(dataset, batch_size=32):
        features = batch["features"]  # [B, T, h, w, D]
        labels = batch["labels"]      # [B]
        # No DiT forward needed!
"""

import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import logging
import random

logger = logging.getLogger(__name__)


class FeatureDataset(Dataset):
    """
    Dataset for loading pre-extracted DiT features.

    Features are stored as [T, h, w, D] tensors.
    """

    def __init__(
        self,
        feature_dir: str,
        label_file: str,
        layer: int = 15,
        timesteps: Optional[List[int]] = None,
        split: str = "train",
        pool_spatial: bool = False,
        pool_temporal: bool = False,
        max_samples: Optional[int] = None,
        dtype: torch.dtype = torch.float16,
    ):
        """
        Args:
            feature_dir: Directory with extracted features
            label_file: Path to labels JSON
            layer: Which DiT layer to load
            timesteps: Which timesteps to use (if None, use all)
            split: "train" or "test"
            pool_spatial: Average pool over (h, w) -> [T, D]
            pool_temporal: Average pool over T
            max_samples: Limit samples (for debugging)
            dtype: Data type to load as
        """
        self.feature_dir = Path(feature_dir)
        self.layer = layer
        self.timesteps = timesteps
        self.split = split
        self.pool_spatial = pool_spatial
        self.pool_temporal = pool_temporal
        self.dtype = dtype

        # Load labels
        with open(label_file) as f:
            all_labels = json.load(f)

        # Load or infer metadata
        metadata_path = self.feature_dir / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path) as f:
                self.metadata = json.load(f)
            videos_meta = self.metadata.get("videos", self.metadata)
        else:
            videos_meta = self._infer_metadata()

        # Build sample list
        self.samples = []
        for video_id, video_meta in videos_meta.items():
            # Check label exists
            if video_id not in all_labels:
                continue

            # Check split
            video_split = video_meta.get("split", split)
            if video_split != split:
                continue

            # Get available timesteps
            if "timesteps" in video_meta:
                available_t = [int(t) for t in video_meta["timesteps"].keys()]
            else:
                # Infer from directory
                video_dir = self.feature_dir / video_id
                available_t = [
                    int(d.name[1:])
                    for d in video_dir.iterdir()
                    if d.is_dir() and d.name.startswith("t")
                ]

            # Filter timesteps
            if timesteps:
                valid_t = [t for t in timesteps if t in available_t]
            else:
                valid_t = available_t

            if not valid_t:
                continue

            self.samples.append(
                {
                    "video_id": video_id,
                    "label": all_labels[video_id],
                    "timesteps": valid_t,
                }
            )

        if max_samples:
            self.samples = self.samples[:max_samples]

        logger.info(
            f"FeatureDataset: {len(self.samples)} videos, layer={layer}, split={split}"
        )

    def _infer_metadata(self) -> Dict:
        """Infer metadata from directory structure."""
        metadata = {}
        for video_dir in self.feature_dir.iterdir():
            if not video_dir.is_dir():
                continue
            video_id = video_dir.name
            metadata[video_id] = {
                "timesteps": {},
                "split": "train",  # Default
            }
            for t_dir in video_dir.iterdir():
                if t_dir.is_dir() and t_dir.name.startswith("t"):
                    t = t_dir.name
                    metadata[video_id]["timesteps"][t] = str(t_dir)
        return metadata

    def _get_feature_path(self, video_id: str, timestep: int) -> Path:
        """Get path to feature file."""
        return self.feature_dir / video_id / f"t{timestep}" / f"layer_{self.layer}.pt"

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]
        video_id = sample["video_id"]
        label = sample["label"]

        # Randomly select timestep (data augmentation)
        timestep = random.choice(sample["timesteps"])

        # Load features
        feature_path = self._get_feature_path(video_id, timestep)
        features = torch.load(feature_path).to(self.dtype)

        # features shape: [T, h, w, D]

        # Optional pooling
        if self.pool_spatial:
            # [T, h, w, D] -> [T, D]
            features = features.mean(dim=(1, 2))

        if self.pool_temporal:
            # [T, h, w, D] -> [h, w, D] or [T, D] -> [D]
            features = features.mean(dim=0)

        return {
            "video_id": video_id,
            "features": features,
            "label": torch.tensor(label, dtype=torch.float32),
            "timestep": timestep,
        }


class FeatureDatasetMultiLayer(Dataset):
    """
    Load features from multiple layers simultaneously.
    Useful for experiments combining layer features.
    """

    def __init__(
        self,
        feature_dir: str,
        label_file: str,
        layers: List[int] = [10, 15, 20],
        **kwargs,
    ):
        self.layers = layers
        self._base = FeatureDataset(
            feature_dir=feature_dir, label_file=label_file, layer=layers[0], **kwargs
        )
        self.feature_dir = self._base.feature_dir
        self.samples = self._base.samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]
        video_id = sample["video_id"]
        label = sample["label"]
        timestep = random.choice(sample["timesteps"])

        # Load all layers
        features = {}
        for layer in self.layers:
            path = self.feature_dir / video_id / f"t{timestep}" / f"layer_{layer}.pt"
            features[layer] = torch.load(path)

        return {
            "video_id": video_id,
            "features": features,
            "label": torch.tensor(label, dtype=torch.float32),
            "timestep": timestep,
        }


# =============================================================================
# Collate functions
# =============================================================================


def collate_features(batch: List[Dict]) -> Dict:
    """Standard collate for FeatureDataset."""
    return {
        "video_ids": [b["video_id"] for b in batch],
        "features": torch.stack([b["features"] for b in batch]),
        "labels": torch.stack([b["label"] for b in batch]),
        "timesteps": [b["timestep"] for b in batch],
    }


def collate_features_multi_layer(batch: List[Dict]) -> Dict:
    """Collate for FeatureDatasetMultiLayer."""
    layers = list(batch[0]["features"].keys())
    return {
        "video_ids": [b["video_id"] for b in batch],
        "features": {
            layer: torch.stack([b["features"][layer] for b in batch])
            for layer in layers
        },
        "labels": torch.stack([b["label"] for b in batch]),
        "timesteps": [b["timestep"] for b in batch],
    }


# =============================================================================
# Utilities
# =============================================================================


def create_dataloader(
    feature_dir: str,
    label_file: str,
    layer: int = 15,
    timesteps: Optional[List[int]] = None,
    split: str = "train",
    batch_size: int = 32,
    num_workers: int = 4,
    **dataset_kwargs,
) -> DataLoader:
    """Convenience function to create DataLoader."""
    dataset = FeatureDataset(
        feature_dir=feature_dir,
        label_file=label_file,
        layer=layer,
        timesteps=timesteps,
        split=split,
        **dataset_kwargs,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        collate_fn=collate_features,
        pin_memory=True,
    )


def get_feature_shape(feature_dir: str, layer: int = 15) -> Tuple[int, ...]:
    """Get shape of features from a sample file."""
    feature_dir = Path(feature_dir)

    for video_dir in feature_dir.iterdir():
        if not video_dir.is_dir():
            continue
        for t_dir in video_dir.iterdir():
            if not t_dir.is_dir():
                continue
            feature_file = t_dir / f"layer_{layer}.pt"
            if feature_file.exists():
                features = torch.load(feature_file)
                return tuple(features.shape)

    raise FileNotFoundError(f"No feature files found in {feature_dir}")
