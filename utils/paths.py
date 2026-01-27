#!/usr/bin/env python3
"""
Centralized Path Configuration for VLMPhysics

Provides consistent directory structure for all experiment outputs:
- Generation results (videos)
- Evaluation results (metrics)
- Training results (checkpoints, logs)
- Figures

Directory Structure:
    results/
    ├── generation/                              # Video generation
    │   └── phygenbench/
    │       └── {model}_{method}_{YYYYMMDD_HHMMSS}/
    │           ├── config.json                 # Generation config
    │           ├── log.json                    # Generation log
    │           └── videos/                     # Generated videos
    │               ├── output_video_1.mp4
    │               └── ...
    │
    ├── evaluation/                             # Evaluation results
    │   └── phygenbench/
    │       └── {exp_name}_{stages}_{YYYYMMDD_HHMMSS}/
    │           ├── config.json                 # Evaluation config
    │           ├── results.json                # Full results
    │           └── summary.json                # Summary statistics
    │
    ├── training/                               # Training results
    │   └── physics_head/
    │       └── {ablation}_{YYYYMMDD_HHMMSS}/
    │           ├── config.json                 # Training config
    │           ├── summary.json                # Ablation summary
    │           └── {variant}/                  # Per-variant results
    │               ├── best.pt
    │               ├── latest.pt
    │               └── results.json
    │
    └── figures/                                # Visualizations
        └── {exp_name}_{YYYYMMDD_HHMMSS}/

Usage:
    from utils.paths import ResultsManager

    # For generation
    rm = ResultsManager(exp_type="generation", exp_name="baseline", model="cogvideox-2b")
    video_dir = rm.get_output_dir()  # results/generation/phygenbench/cogvideox-2b_baseline_20260127_143022

    # For training
    rm = ResultsManager(exp_type="training", exp_name="head_ablation")
    output_dir = rm.get_output_dir()  # results/training/physics_head/head_ablation_20260127_160000

    # Resume existing experiment
    rm = ResultsManager.from_existing("results/training/physics_head/head_ablation_20260127_160000")
"""

import os
import json
import shutil
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, Union, List
from dataclasses import dataclass, asdict, field


# =============================================================================
# Constants
# =============================================================================

# Base results directory (relative to project root)
DEFAULT_RESULTS_BASE = "results"

# Experiment types and their subdirectories
EXP_TYPES = {
    "generation": "generation/phygenbench",
    "evaluation": "evaluation/phygenbench",
    "training": "training/physics_head",
    "training_seed": "training/seed",
    "figures": "figures",
}

# Config filename for each experiment
CONFIG_FILENAME = "config.json"
SUMMARY_FILENAME = "summary.json"


# =============================================================================
# Experiment Config
# =============================================================================


@dataclass
class ExperimentConfig:
    """Configuration for an experiment run."""

    # Basic info
    exp_type: str
    exp_name: str
    timestamp: str

    # Optional metadata
    model: Optional[str] = None
    description: Optional[str] = None

    # Training specific
    ablation_type: Optional[str] = None  # "heads", "layers", "timesteps"
    layer: Optional[int] = None
    head_type: Optional[str] = None

    # Generation specific
    num_frames: Optional[int] = None
    num_steps: Optional[int] = None
    seed: Optional[int] = None

    # Evaluation specific
    stages: Optional[List[int]] = None

    # Additional parameters
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, excluding None values."""
        d = asdict(self)
        return {k: v for k, v in d.items() if v is not None}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ExperimentConfig":
        """Create from dictionary."""
        # Extract known fields
        known_fields = {
            "exp_type",
            "exp_name",
            "timestamp",
            "model",
            "description",
            "ablation_type",
            "layer",
            "head_type",
            "num_frames",
            "num_steps",
            "seed",
            "stages",
            "extra",
        }

        kwargs = {k: v for k, v in d.items() if k in known_fields}

        # Put unknown fields in extra
        extra = kwargs.get("extra", {})
        for k, v in d.items():
            if k not in known_fields:
                extra[k] = v
        kwargs["extra"] = extra

        return cls(**kwargs)


# =============================================================================
# Results Manager
# =============================================================================


class ResultsManager:
    """
    Centralized manager for experiment output directories.

    Creates structured output directories with timestamps and configs.
    """

    def __init__(
        self,
        exp_type: str,
        exp_name: str,
        model: Optional[str] = None,
        timestamp: Optional[str] = None,
        results_base: Optional[str] = None,
        create: bool = True,
        **kwargs,
    ):
        """
        Initialize results manager.

        Args:
            exp_type: Type of experiment ("generation", "evaluation", "training", "figures")
            exp_name: Name of the experiment (e.g., "baseline", "head_ablation")
            model: Model name for generation (e.g., "cogvideox-2b")
            timestamp: Optional timestamp (auto-generated if None)
            results_base: Base directory for results (default: "results")
            create: Whether to create directories
            **kwargs: Additional config parameters
        """
        if exp_type not in EXP_TYPES:
            raise ValueError(
                f"Unknown exp_type: {exp_type}. Must be one of {list(EXP_TYPES.keys())}"
            )

        self.exp_type = exp_type
        self.exp_name = exp_name
        self.model = model
        self.timestamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_base = Path(results_base or DEFAULT_RESULTS_BASE)

        # Build config - filter kwargs into known fields vs extra
        known_config_fields = {
            "description",
            "ablation_type",
            "layer",
            "head_type",
            "num_frames",
            "num_steps",
            "seed",
            "stages",
        }
        config_kwargs = {k: v for k, v in kwargs.items() if k in known_config_fields}
        extra_kwargs = {k: v for k, v in kwargs.items() if k not in known_config_fields}

        self.config = ExperimentConfig(
            exp_type=exp_type,
            exp_name=exp_name,
            timestamp=self.timestamp,
            model=model,
            extra=extra_kwargs,
            **config_kwargs,
        )

        # Compute paths
        self._compute_paths()

        # Create directories
        if create:
            self._create_directories()

    def _compute_paths(self):
        """Compute all relevant paths."""
        # Base path for this experiment type
        type_subdir = EXP_TYPES[self.exp_type]
        self.type_dir = self.results_base / type_subdir

        # Experiment directory name
        if self.model and self.exp_type == "generation":
            self.exp_dir_name = f"{self.model}_{self.exp_name}_{self.timestamp}"
        else:
            self.exp_dir_name = f"{self.exp_name}_{self.timestamp}"

        # Full experiment directory
        self.exp_dir = self.type_dir / self.exp_dir_name

        # Standard subdirectories based on exp_type
        if self.exp_type == "generation":
            self.videos_dir = self.exp_dir / "videos"
        elif self.exp_type == "training":
            # Will be created per-ablation
            pass

    def _create_directories(self):
        """Create necessary directories."""
        self.exp_dir.mkdir(parents=True, exist_ok=True)

        if self.exp_type == "generation":
            self.videos_dir.mkdir(exist_ok=True)

    def get_output_dir(self) -> Path:
        """Get the main output directory."""
        return self.exp_dir

    def get_videos_dir(self) -> Path:
        """Get videos directory (for generation)."""
        if self.exp_type != "generation":
            raise ValueError("videos_dir only available for generation experiments")
        return self.videos_dir

    def get_video_path(self, index: int) -> Path:
        """
        Get path for a specific video file.

        Args:
            index: Video index (0-based, saved as 1-based for PhyGenBench)

        Returns:
            Path to video file
        """
        if self.exp_type != "generation":
            raise ValueError("get_video_path only available for generation experiments")
        # PhyGenBench uses 1-based indexing
        return self.videos_dir / f"output_video_{index + 1}.mp4"

    def get_ablation_dir(
        self, ablation_type: str, ablation_value: Union[str, int]
    ) -> Path:
        """
        Get directory for a specific ablation run.

        Args:
            ablation_type: "heads", "layers", or "timesteps"
            ablation_value: The specific value (e.g., "mean", 15, "200")

        Returns:
            Path to ablation subdirectory
        """
        if self.exp_type != "training":
            raise ValueError("ablation_dir only available for training experiments")

        # Map ablation types to directory prefixes
        prefix_map = {
            "heads": "",
            "layers": "layer_",
            "timesteps": "t_",
            "seeds": "seed_",
        }

        prefix = prefix_map.get(ablation_type, "")
        subdir_name = f"{prefix}{ablation_value}"

        ablation_dir = self.exp_dir / subdir_name
        ablation_dir.mkdir(parents=True, exist_ok=True)

        return ablation_dir

    def save_config(self, extra: Optional[Dict[str, Any]] = None):
        """Save experiment configuration."""
        config_dict = self.config.to_dict()
        if extra:
            config_dict["extra"].update(extra)

        config_path = self.exp_dir / CONFIG_FILENAME
        with open(config_path, "w") as f:
            json.dump(config_dict, f, indent=2)

    def save_summary(self, summary: Dict[str, Any]):
        """Save experiment summary."""
        summary_path = self.exp_dir / SUMMARY_FILENAME
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

    def save_results(self, results: Dict[str, Any], filename: str = "results.json"):
        """Save results to a JSON file."""
        results_path = self.exp_dir / filename
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)

    @classmethod
    def from_existing(cls, path: Union[str, Path]) -> "ResultsManager":
        """
        Load from an existing experiment directory.

        Args:
            path: Path to existing experiment directory

        Returns:
            ResultsManager instance
        """
        path = Path(path)
        config_path = path / CONFIG_FILENAME

        if not config_path.exists():
            raise ValueError(f"No config.json found in {path}")

        with open(config_path, "r") as f:
            config_dict = json.load(f)

        config = ExperimentConfig.from_dict(config_dict)

        # Extract results_base from path
        # Path structure: results_base/exp_type_subdir/exp_dir_name
        parts = path.parts
        results_base = Path(*parts[:-2]) if len(parts) > 2 else Path(".")

        return cls(
            exp_type=config.exp_type,
            exp_name=config.exp_name,
            model=config.model,
            timestamp=config.timestamp,
            results_base=str(results_base),
            create=False,
            **config.extra,
        )

    @staticmethod
    def list_experiments(
        exp_type: str, results_base: str = DEFAULT_RESULTS_BASE
    ) -> List[Path]:
        """
        List all experiments of a given type.

        Args:
            exp_type: Type of experiment
            results_base: Base results directory

        Returns:
            List of experiment directories, sorted by timestamp (newest first)
        """
        type_subdir = EXP_TYPES.get(exp_type)
        if not type_subdir:
            return []

        type_dir = Path(results_base) / type_subdir
        if not type_dir.exists():
            return []

        # Get all subdirectories
        exp_dirs = [d for d in type_dir.iterdir() if d.is_dir()]

        # Sort by modification time (newest first)
        exp_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)

        return exp_dirs

    @staticmethod
    def get_latest_experiment(
        exp_type: str,
        exp_name: Optional[str] = None,
        results_base: str = DEFAULT_RESULTS_BASE,
    ) -> Optional[Path]:
        """
        Get the most recent experiment directory.

        Args:
            exp_type: Type of experiment
            exp_name: Optional filter by experiment name
            results_base: Base results directory

        Returns:
            Path to latest experiment, or None
        """
        exp_dirs = ResultsManager.list_experiments(exp_type, results_base)

        if exp_name:
            exp_dirs = [d for d in exp_dirs if exp_name in d.name]

        return exp_dirs[0] if exp_dirs else None


# =============================================================================
# Helper Functions
# =============================================================================


def get_timestamp() -> str:
    """Get current timestamp string."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_dir(path: Union[str, Path]) -> Path:
    """Ensure directory exists and return Path."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def clean_old_results(
    exp_type: str,
    keep_n: int = 5,
    results_base: str = DEFAULT_RESULTS_BASE,
    dry_run: bool = True,
) -> List[Path]:
    """
    Clean old experiment results, keeping only the most recent N.

    Args:
        exp_type: Type of experiment to clean
        keep_n: Number of experiments to keep
        results_base: Base results directory
        dry_run: If True, only print what would be deleted

    Returns:
        List of directories that were/would be deleted
    """
    exp_dirs = ResultsManager.list_experiments(exp_type, results_base)

    to_delete = exp_dirs[keep_n:]

    for d in to_delete:
        print(f"{'Would delete' if dry_run else 'Deleting'}: {d}")
        if not dry_run:
            shutil.rmtree(d)

    return to_delete


def create_generation_manager(
    exp_name: str = "baseline",
    model: str = "cogvideox-2b",
    results_base: str = DEFAULT_RESULTS_BASE,
    **kwargs,
) -> ResultsManager:
    """
    Convenience function to create a generation ResultsManager.

    Args:
        exp_name: Experiment name (e.g., "baseline", "trajectory_pruning")
        model: Model name (e.g., "cogvideox-2b", "cogvideox-5b")
        results_base: Base results directory
        **kwargs: Additional config parameters (num_frames, num_steps, seed, etc.)

    Returns:
        ResultsManager configured for generation

    Example:
        rm = create_generation_manager("baseline", "cogvideox-2b", seed=42, num_steps=50)
        rm.get_video_path(0)  # results/generation/phygenbench/cogvideox-2b_baseline_.../videos/output_video_1.mp4
    """
    return ResultsManager(
        exp_type="generation",
        exp_name=exp_name,
        model=model,
        results_base=results_base,
        **kwargs,
    )


def create_evaluation_manager(
    exp_name: str = "eval",
    stages: Optional[List[int]] = None,
    results_base: str = DEFAULT_RESULTS_BASE,
    **kwargs,
) -> ResultsManager:
    """
    Convenience function to create an evaluation ResultsManager.

    Args:
        exp_name: Experiment name (e.g., "baseline_eval")
        stages: Evaluation stages (e.g., [1, 2, 3])
        results_base: Base results directory
        **kwargs: Additional config parameters

    Returns:
        ResultsManager configured for evaluation
    """
    # Include stages in exp_name for clarity
    if stages:
        stage_str = "stage" + "".join(str(s) for s in sorted(stages))
        full_name = f"{exp_name}_{stage_str}"
    else:
        full_name = exp_name

    return ResultsManager(
        exp_type="evaluation",
        exp_name=full_name,
        results_base=results_base,
        stages=stages,
        **kwargs,
    )


def create_training_manager(
    exp_name: str = "train",
    ablation_type: Optional[str] = None,
    results_base: str = DEFAULT_RESULTS_BASE,
    **kwargs,
) -> ResultsManager:
    """
    Convenience function to create a training ResultsManager.

    Args:
        exp_name: Experiment name (e.g., "single_run")
        ablation_type: Type of ablation ("heads", "layers", "timesteps", "seeds")
        results_base: Base results directory
        **kwargs: Additional config parameters (layer, head_type, lr, etc.)

    Returns:
        ResultsManager configured for training
    """
    # Include ablation type in exp_name
    if ablation_type:
        full_name = f"{ablation_type}_ablation"
    else:
        full_name = exp_name

    # Use separate directory for seed ablation
    exp_type = "training_seed" if ablation_type == "seeds" else "training"

    return ResultsManager(
        exp_type=exp_type,
        exp_name=full_name,
        results_base=results_base,
        ablation_type=ablation_type,
        **kwargs,
    )


def create_figures_manager(
    exp_name: str = "figures", results_base: str = DEFAULT_RESULTS_BASE, **kwargs
) -> ResultsManager:
    """
    Convenience function to create a figures ResultsManager.

    Args:
        exp_name: Experiment name (e.g., "head_ablation_figures")
        results_base: Base results directory
        **kwargs: Additional parameters

    Returns:
        ResultsManager configured for figures
    """
    return ResultsManager(
        exp_type="figures", exp_name=exp_name, results_base=results_base, **kwargs
    )


def get_videos_from_generation(gen_dir: Union[str, Path]) -> List[Path]:
    """
    Get list of video files from a generation directory.

    Args:
        gen_dir: Path to generation experiment directory

    Returns:
        Sorted list of video paths
    """
    gen_dir = Path(gen_dir)
    videos_dir = gen_dir / "videos"

    if not videos_dir.exists():
        # Check if videos are directly in gen_dir (old structure)
        videos = list(gen_dir.glob("output_video_*.mp4"))
    else:
        videos = list(videos_dir.glob("output_video_*.mp4"))

    # Sort by index
    def get_idx(p: Path) -> int:
        try:
            return int(p.stem.split("_")[-1])
        except ValueError:
            return 0

    return sorted(videos, key=get_idx)


# =============================================================================
# CLI
# =============================================================================


def main():
    """CLI for managing results."""
    import argparse

    parser = argparse.ArgumentParser(description="Manage experiment results")
    subparsers = parser.add_subparsers(dest="command", help="Command")

    # List command
    list_parser = subparsers.add_parser("list", help="List experiments")
    list_parser.add_argument("--type", choices=list(EXP_TYPES.keys()), required=True)
    list_parser.add_argument("--base", default=DEFAULT_RESULTS_BASE)

    # Clean command
    clean_parser = subparsers.add_parser("clean", help="Clean old experiments")
    clean_parser.add_argument("--type", choices=list(EXP_TYPES.keys()), required=True)
    clean_parser.add_argument("--keep", type=int, default=5, help="Number to keep")
    clean_parser.add_argument("--base", default=DEFAULT_RESULTS_BASE)
    clean_parser.add_argument("--execute", action="store_true", help="Actually delete")

    args = parser.parse_args()

    if args.command == "list":
        exp_dirs = ResultsManager.list_experiments(args.type, args.base)
        if exp_dirs:
            print(f"\n{args.type.upper()} experiments ({len(exp_dirs)} total):\n")
            for d in exp_dirs:
                config_path = d / CONFIG_FILENAME
                if config_path.exists():
                    with open(config_path) as f:
                        config = json.load(f)
                    print(f"  {d.name}")
                    print(f"    Created: {config.get('timestamp', 'unknown')}")
                    if config.get("model"):
                        print(f"    Model: {config['model']}")
                else:
                    print(f"  {d.name} (no config)")
        else:
            print(f"No {args.type} experiments found.")

    elif args.command == "clean":
        deleted = clean_old_results(
            args.type,
            keep_n=args.keep,
            results_base=args.base,
            dry_run=not args.execute,
        )
        if deleted:
            print(
                f"\n{'Deleted' if args.execute else 'Would delete'} {len(deleted)} experiment(s)"
            )
        else:
            print("Nothing to clean")


if __name__ == "__main__":
    main()
