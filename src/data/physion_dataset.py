"""
Physion Dataset Loader with Fixed Video Name Matching

Key insight:
- pilot_* videos: Located in {Scenario}/mp4s-redyellow/, filename has -redyellow suffix
- test* videos (Drape): Located in Drape/mp4s/, filename is simply {video_id}.mp4

The labels.csv uses names WITHOUT -redyellow suffix for pilot videos.
"""

import os
import re
import csv
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


class PhysionDataset(Dataset):
    """
    Dataset for loading Physion videos with physics labels.

    Handles the complex filename mapping between labels.csv and actual video files.
    """

    SCENARIOS = [
        "Collide",
        "Contain",
        "Dominoes",
        "Drape",
        "Drop",
        "Link",
        "Roll",
        "Support",
    ]

    # Mapping from labels.csv scenario prefix to folder name
    SCENARIO_MAPPING = {
        "collision": "Collide",
        "containment": "Contain",
        "dominoes": "Dominoes",
        "drape": "Drape",
        "drop": "Drop",
        "link": "Link",
        "roll": "Roll",
        "support": "Support",
    }

    def __init__(
        self,
        data_root: str,
        split: str = "all",
        num_frames: int = 49,
        frame_size: Tuple[int, int] = (480, 720),
        val_ratio: float = 0.1,
        seed: int = 42,
        debug: bool = False,
    ):
        """
        Args:
            data_root: Path to Physion data folder (contains Collide, Contain, etc.)
            split: Data split - "train", "val", "test", or "all"
            num_frames: Number of frames to sample from each video
            frame_size: Target frame size (H, W)
            val_ratio: Fraction of data to use for validation
            seed: Random seed for split
            debug: Enable debug output
        """
        self.data_root = Path(data_root)
        self.split = split
        self.num_frames = num_frames
        self.frame_size = frame_size
        self.val_ratio = val_ratio
        self.seed = seed
        self.debug = debug

        # Load labels and find corresponding video files
        self.samples = []  # List of (video_path, label, scenario, video_name)
        self._load_labels()

        # Apply train/val split
        self._apply_split()

        logger.info(f"PhysionDataset [{split}]: {len(self.samples)} samples")
        self._print_label_distribution()

    def _detect_scenario_from_name(self, video_name: str) -> Optional[str]:
        """
        Detect scenario from video name.

        Examples:
            pilot_it2_collision_xxx -> Collide
            pilot-containment-xxx -> Contain
            pilot_dominoes_xxx -> Dominoes
            testNN_xxxx_img -> Drape
            pilot_it2_drop_xxx -> Drop
            pilot_it2_linking_xxx -> Link
            pilot_it2_rolling_xxx -> Roll
            pilot_towers_xxx -> Support
        """
        video_name_lower = video_name.lower()

        # Test videos are all Drape scenario
        if video_name_lower.startswith("test"):
            return "Drape"
        # IMPORTANT: Check rollingSliding BEFORE collision!
        # pilot_it2_rollingSliding_simple_collision_* contains "collision" 
        # but belongs to Roll scenario
        if "rollingsliding" in video_name_lower:
            return "Roll"

        # Check for scenario keywords
        if "collision" in video_name_lower or "_collide" in video_name_lower:
            return "Collide"
        if "containment" in video_name_lower or "-containment" in video_name_lower:
            return "Contain"
        if "dominoes" in video_name_lower:
            return "Dominoes"
        if "_drop_" in video_name_lower:
            return "Drop"
        if "linking" in video_name_lower or "_link_" in video_name_lower:
            return "Link"
        if "rolling" in video_name_lower or "_roll_" in video_name_lower:
            return "Roll"
        if "towers" in video_name_lower or "_support" in video_name_lower:
            return "Support"

        return None

    def _find_video_path(self, scenario: str, video_name: str) -> Optional[Path]:
        """
        Find the actual video file path for a video ID.

        Handles two cases:
        1. test* videos (Drape): in mp4s/ folder with simple filename
        2. pilot* videos: in mp4s-redyellow/ folder with -redyellow suffix
        """
        # Case 1: Test videos (Drape scenario)
        # test videos are in Drape/mp4s/ with simple filenames like test10_0007_img.mp4
        if video_name.startswith("test"):
            # Try mp4s folder first (NOT mp4s-redyellow)
            mp4s_dir = self.data_root / scenario / "mp4s"
            if mp4s_dir.exists():
                # Direct match: test10_0007_img -> test10_0007_img.mp4
                direct_path = mp4s_dir / f"{video_name}.mp4"
                if direct_path.exists():
                    if self.debug:
                        logger.debug(f"  Found test video: {direct_path}")
                    return direct_path

                # Search in directory
                for fname in mp4s_dir.iterdir():
                    if fname.suffix == ".mp4" and video_name in fname.stem:
                        if self.debug:
                            logger.debug(f"  Found test video via search: {fname}")
                        return fname

            # Also try mp4s-redyellow in case test videos are there too
            redyellow_dir = self.data_root / scenario / "mp4s-redyellow"
            if redyellow_dir.exists():
                # Try with -redyellow suffix
                match = re.match(r"^(.+?)_(\d{4})_(img)$", video_name)
                if match:
                    prefix, num, suffix = match.groups()
                    redyellow_name = f"{prefix}-redyellow_{num}_{suffix}.mp4"
                    redyellow_path = redyellow_dir / redyellow_name
                    if redyellow_path.exists():
                        if self.debug:
                            logger.debug(
                                f"  Found test video in redyellow: {redyellow_path}"
                            )
                        return redyellow_path

            if self.debug:
                logger.warning(f"  Test video not found: {video_name}")
            return None

        # Case 2: Pilot videos (all other scenarios)
        # Located in {Scenario}/mp4s-redyellow/ with -redyellow in filename
        scenario_dir = self.data_root / scenario / "mp4s-redyellow"

        if not scenario_dir.exists():
            # Try lowercase scenario name
            scenario_dir = self.data_root / scenario.lower() / "mp4s-redyellow"

        if not scenario_dir.exists():
            if self.debug:
                logger.warning(f"  Scenario dir not found: {scenario_dir}")
            return None

        # Method 1: Transform name by inserting -redyellow before the 4-digit number
        # labels.csv:  pilot_dominoes_0mid_xxx_tdwroom_0018_img
        # actual file: pilot_dominoes_0mid_xxx_tdwroom-redyellow_0018_img.mp4
        match = re.match(r"^(.+?)_(\d{4})_(img)$", video_name)
        if match:
            prefix, num, suffix = match.groups()
            transformed_name = f"{prefix}-redyellow_{num}_{suffix}.mp4"
            transformed_path = scenario_dir / transformed_name
            if transformed_path.exists():
                if self.debug:
                    logger.debug(f"  Found via transform: {transformed_path}")
                return transformed_path

        # Method 2: Search by comparing normalized names (remove -redyellow)
        for fname in scenario_dir.iterdir():
            if fname.suffix != ".mp4":
                continue

            # Remove -redyellow from filename and compare
            fname_normalized = fname.stem.replace("-redyellow", "")
            if fname_normalized == video_name:
                if self.debug:
                    logger.debug(f"  Found via normalized search: {fname}")
                return fname

        if self.debug:
            logger.warning(f"  Pilot video not found: {video_name} in {scenario_dir}")
        return None

    def _load_labels(self):
        """Load labels.csv and match with video files."""
        labels_path = self.data_root / "labels.csv"

        if not labels_path.exists():
            raise FileNotFoundError(f"Labels file not found: {labels_path}")

        found_count = 0
        missing_count = 0
        missing_by_scenario = {}

        with open(labels_path, "r") as f:
            reader = csv.reader(f)
            next(reader)  # Skip header: ",ground truth outcome"

            for row_idx, row in enumerate(reader):
                if len(row) < 2:
                    continue

                video_name = row[0].strip()
                label_str = row[1].strip()
                label = 1 if label_str == "True" else 0

                # Detect scenario from video name
                scenario = self._detect_scenario_from_name(video_name)

                if scenario is None:
                    if self.debug:
                        logger.warning(f"Cannot detect scenario for: {video_name}")
                    missing_count += 1
                    continue

                # Find video file
                video_path = self._find_video_path(scenario, video_name)

                if video_path is not None:
                    self.samples.append((video_path, label, scenario, video_name))
                    found_count += 1
                else:
                    missing_count += 1
                    missing_by_scenario[scenario] = (
                        missing_by_scenario.get(scenario, 0) + 1
                    )

        logger.info(f"Loaded {found_count} samples from labels.csv")

        if missing_count > 0:
            logger.warning(f"Missing {missing_count} videos")
            for scenario, count in sorted(missing_by_scenario.items()):
                logger.warning(f"  {scenario}: {count} missing")

    def _apply_split(self):
        """Apply train/val/test split."""
        if self.split == "all":
            return

        # Shuffle samples with fixed seed for reproducibility
        rng = np.random.RandomState(self.seed)
        indices = np.arange(len(self.samples))
        rng.shuffle(indices)

        n_val = int(len(self.samples) * self.val_ratio)

        if self.split == "train":
            selected_indices = indices[n_val:]
        elif self.split == "val":
            selected_indices = indices[:n_val]
        else:
            # For "test", use all data (same as "all" for now)
            return

        self.samples = [self.samples[i] for i in selected_indices]

    def _print_label_distribution(self):
        """Print label distribution info."""
        if len(self.samples) == 0:
            logger.warning("No samples loaded!")
            return

        labels = [s[1] for s in self.samples]
        pos = sum(labels)
        neg = len(labels) - pos
        logger.info(
            f"  Label distribution: {pos} positive ({pos / len(labels) * 100:.1f}%), "
            f"{neg} negative ({neg / len(labels) * 100:.1f}%)"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def _load_video(self, video_path: Path) -> torch.Tensor:
        """Load and preprocess video frames."""
        if tvio is None:
            raise ImportError("torchvision is required for video loading")

        # Read video
        video, audio, info = tvio.read_video(str(video_path), pts_unit="sec")
        # video shape: [T, H, W, C] in uint8

        # Sample frames uniformly
        total_frames = video.shape[0]
        if total_frames >= self.num_frames:
            indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
        else:
            # Repeat last frame if video is too short
            indices = list(range(total_frames))
            indices.extend([total_frames - 1] * (self.num_frames - total_frames))

        video = video[indices]  # [num_frames, H, W, C]

        # Convert to float and normalize to [0, 1]
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

        # Rearrange to [T, C, H, W]
        video = video.permute(0, 3, 1, 2)

        return video

    def __getitem__(self, idx: int) -> Dict:
        video_path, label, scenario, video_name = self.samples[idx]

        video = self._load_video(video_path)

        return {
            "video": video,  # [T, C, H, W]
            "label": label,
            "scenario": scenario,
            "video_name": video_name,
        }


def get_dataloaders(
    data_root: str,
    batch_size: int = 1,
    num_workers: int = 4,
    num_frames: int = 49,
    **dataset_kwargs,
) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation dataloaders."""
    train_dataset = PhysionDataset(
        data_root, split="train", num_frames=num_frames, **dataset_kwargs
    )
    val_dataset = PhysionDataset(
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
    parser.add_argument("--data_root", type=str, default="data/Physion")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    print("=" * 60)
    print("Testing PhysionDataset with Fixed Video Matching")
    print("=" * 60)

    # Enable debug logging if requested
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    # Test dataset
    dataset = PhysionDataset(
        data_root=args.data_root,
        split="all",
        debug=args.debug,
    )

    print(f"\nTotal samples loaded: {len(dataset)}")

    # Show samples per scenario
    scenario_counts = {}
    for _, _, scenario, _ in dataset.samples:
        scenario_counts[scenario] = scenario_counts.get(scenario, 0) + 1

    print("\nSamples per scenario:")
    for scenario in sorted(scenario_counts.keys()):
        print(f"  {scenario}: {scenario_counts[scenario]}")

    # Show expected vs actual
    print(f"\nExpected: 1200 (150 per scenario × 8 scenarios)")
    print(f"Actual: {len(dataset)}")
    print(f"Missing: {1200 - len(dataset)}")

    if len(dataset) > 0:
        print("\nLoading first sample...")
        sample = dataset[0]
        print(f"  Video shape: {sample['video'].shape}")
        print(f"  Label: {sample['label']}")
        print(f"  Scenario: {sample['scenario']}")
        print(f"  Video name: {sample['video_name']}")

    print("\n" + "=" * 60)
    print("Test complete!")
    print("=" * 60)
