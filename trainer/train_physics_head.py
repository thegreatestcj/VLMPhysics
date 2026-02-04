#!/usr/bin/env python3
"""
Training Script for Physics Discriminator Head

Trains lightweight MLP heads on pre-extracted (and pooled) DiT features.
Since features are pre-computed, training is extremely fast (~5 min for 100 epochs).

Output Structure:
    results/training/physics_head/{ablation}_{timestamp}/
    ├── config.json         # Training configuration
    ├── summary.json        # Ablation summary (best AUC per variant)
    └── {variant}/          # Per-variant results (e.g., mean/, causal_adaln/)
        ├── best.pt         # Best checkpoint
        ├── latest.pt       # Latest checkpoint  
        └── results.json    # Training history and metrics

Usage:
    # Basic training with pooled features
    python -m trainer.train_physics_head \
        --feature_dir ~/scratch/physics/physion_features_pooled \
        --layer 15 \
        --is_pooled \
        --exp-name single_run

    # Head type ablation
    python -m trainer.train_physics_head \
        --feature_dir ~/scratch/physics/physion_features_pooled \
        --ablation heads \
        --layer 15 \
        --is_pooled

    # Layer ablation  
    python -m trainer.train_physics_head \
        --feature_dir ~/scratch/physics/physion_features_pooled \
        --ablation layers \
        --layers 5 10 15 20 25 \
        --is_pooled
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

import numpy as np
from tqdm import tqdm

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.paths import (
    ResultsManager,
    create_training_manager,
)

# Try to import project modules
try:
    from src.data.feature_dataset import create_dataloaders
    from src.models.physics_head import create_physics_head, list_heads
except ImportError:
    print("Warning: Could not import project modules. Using stubs.")

    # Stub functions for standalone testing
    def create_dataloaders(*args, **kwargs):
        raise NotImplementedError("Please install project dependencies")

    def create_physics_head(*args, **kwargs):
        raise NotImplementedError("Please install project dependencies")

    def list_heads():
        return [
            "mean",
            "mean_adaln",
            "causal_adaln",
            "temporal_adaln",
            "multiview_adaln",
        ]


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# =============================================================================
# Metrics
# =============================================================================


@dataclass
class Metrics:
    """Evaluation metrics."""

    loss: float = 0.0
    accuracy: float = 0.0
    auc_roc: float = 0.0
    auc_pr: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    auc_per_timestep: Dict[int, float] = None

    def __post_init__(self):
        if self.auc_per_timestep is None:
            self.auc_per_timestep = {}


def compute_metrics(
    all_logits: torch.Tensor,
    all_labels: torch.Tensor,
    all_timesteps: torch.Tensor,
    loss: float = 0.0,
) -> Metrics:
    """Compute evaluation metrics."""
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        roc_auc_score,
        average_precision_score,
    )

    # Convert to numpy
    logits_np = all_logits.cpu().numpy()
    labels_np = all_labels.cpu().numpy()
    timesteps_np = all_timesteps.cpu().numpy()

    # Predictions
    probs = 1 / (1 + np.exp(-logits_np))
    preds = (probs > 0.5).astype(int)

    # Basic metrics
    accuracy = accuracy_score(labels_np, preds)
    precision = precision_score(labels_np, preds, zero_division=0)
    recall = recall_score(labels_np, preds, zero_division=0)
    f1 = f1_score(labels_np, preds, zero_division=0)

    # AUC metrics
    try:
        auc_roc = roc_auc_score(labels_np, probs)
    except ValueError:
        auc_roc = 0.5

    try:
        auc_pr = average_precision_score(labels_np, probs)
    except ValueError:
        auc_pr = 0.5

    # Per-timestep AUC
    auc_per_timestep = {}
    for t in np.unique(timesteps_np):
        mask = timesteps_np == t
        if mask.sum() > 10 and len(np.unique(labels_np[mask])) > 1:
            try:
                auc_per_timestep[int(t)] = roc_auc_score(labels_np[mask], probs[mask])
            except ValueError:
                auc_per_timestep[int(t)] = 0.5

    return Metrics(
        loss=loss,
        accuracy=accuracy,
        auc_roc=auc_roc,
        auc_pr=auc_pr,
        precision=precision,
        recall=recall,
        f1=f1,
        auc_per_timestep=auc_per_timestep,
    )


# =============================================================================
# Trainer
# =============================================================================


class PhysicsHeadTrainer:
    """Trainer for physics discriminator head."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        output_dir: str,
        head_type: str = "mean",
        device: str = "cuda",
        lr: float = 1e-3,
        weight_decay: float = 0.01,
        num_epochs: int = 100,
        early_stopping_patience: int = 15,
        train_sa: bool = False,  # NEW
        sa_weight: float = 0.3,
    ):
        if train_sa and not isinstance(model, MultiTaskWrapper):
            from src.models.physics_head import MultiTaskWrapper
            model = MultiTaskWrapper(model)
            logger.info(f"Auto-wrapped with MultiTaskWrapper (sa_weight={sa_weight})")
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.head_type = head_type
        self.device = device
        self.num_epochs = num_epochs
        self.early_stopping_patience = early_stopping_patience

        # Setup optimizer
        # Setup optimizer
        # Compute pos_weight from training data to handle class imbalance
        # VideoPhy: ~35% positive, ~65% negative → pos_weight ≈ 1.83
        train_labels = []
        for batch in train_loader:
            train_labels.append(batch["labels"])
        train_labels = torch.cat(train_labels)
        num_pos = train_labels.sum().item()
        num_neg = len(train_labels) - num_pos
        if num_pos > 0 and num_neg > 0:
            pw = torch.tensor([num_neg / num_pos], device=device)
            logger.info(
                f"Class imbalance: {num_pos:.0f} pos / {num_neg:.0f} neg, pos_weight={pw.item():.3f}"
            )
        else:
            pw = None
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pw)
        # Multi-task: SA prediction
        self.train_sa = train_sa
        self.sa_weight = sa_weight
        if train_sa:
            self.sa_criterion = nn.BCEWithLogitsLoss()
            logger.info(f"Multi-task training enabled: SA weight={sa_weight}")
        self.optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

        total_steps = len(train_loader) * num_epochs
        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=lr,
            total_steps=total_steps,
            pct_start=0.1,
            anneal_strategy="cos",
        )

        # Tracking
        self.best_auc = 0.0
        self.best_epoch = 0
        self.no_improve_count = 0
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "val_auc": [],
            "val_accuracy": [],
            "val_f1": [],
            "lr": [],
        }

    def _requires_timestep(self) -> bool:
        """Check if head requires timestep input."""
        return self.head_type != "mean"

    def _forward(self, batch: Dict) -> Dict[str, torch.Tensor]:
        """Forward pass. Returns dict with logits, labels, timesteps, and optionally SA."""
        features = batch["features"].to(self.device)
        labels = batch["labels"].to(self.device).float()
        timesteps = batch["timesteps"]

        if isinstance(timesteps, list):
            timesteps = torch.tensor(timesteps, device=self.device)
        elif isinstance(timesteps, int):
            timesteps = torch.tensor([timesteps], device=self.device)
        else:
            timesteps = timesteps.to(self.device)

        if self._requires_timestep():
            output = self.model(features, timesteps)
        else:
            output = self.model(features)

        result = {"timesteps": timesteps, "labels": labels}

        if self.train_sa and isinstance(output, tuple):
            # MultiTaskWrapper returns (physics_logits, sa_logits)
            physics_logits, sa_logits = output
            result["logits"] = physics_logits.squeeze(-1)
            result["sa_logits"] = sa_logits.squeeze(-1)
            # SA labels from batch (-1 means unknown, mask those)
            if "sa" in batch:
                result["sa_labels"] = batch["sa"].to(self.device).float()
        else:
            # Single-task: just physics
            if isinstance(output, tuple):
                output = output[0]  # safety: unwrap if tuple
            result["logits"] = output.squeeze(-1)

        return result

    def train_epoch(self) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in self.train_loader:
            fwd = self._forward(batch)
            loss = self.criterion(fwd["logits"], fwd["labels"])

            # Multi-task: add SA loss
            if self.train_sa and "sa_logits" in fwd and "sa_labels" in fwd:
                sa_labels = fwd["sa_labels"]
                sa_logits = fwd["sa_logits"]
                # Mask unknown SA values (sa == -1)
                sa_mask = sa_labels >= 0
                if sa_mask.any():
                    sa_loss = self.sa_criterion(
                        sa_logits[sa_mask], sa_labels[sa_mask]
                    )
                    loss = loss + self.sa_weight * sa_loss

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.scheduler.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / max(num_batches, 1)

    @torch.no_grad()
    def evaluate(self) -> Metrics:
        """Evaluate on validation set."""
        self.model.eval()

        all_logits, all_labels, all_timesteps = [], [], []
        total_loss = 0.0
        num_batches = 0

        for batch in self.val_loader:
            fwd = self._forward(batch)
            loss = self.criterion(fwd["logits"], fwd["labels"])

            total_loss += loss.item()
            num_batches += 1

            all_logits.append(fwd["logits"].cpu())
            all_labels.append(fwd["labels"].cpu())
            all_timesteps.append(fwd["timesteps"].cpu())

    def save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_auc": self.best_auc,
            "best_epoch": self.best_epoch,
            "head_type": self.head_type,
        }

        torch.save(checkpoint, self.output_dir / "latest.pt")
        if is_best:
            torch.save(checkpoint, self.output_dir / "best.pt")

    def train(self) -> Dict:
        """Full training loop."""
        logger.info(f"Starting training for {self.num_epochs} epochs...")
        logger.info(
            f"Train batches: {len(self.train_loader)}, Val batches: {len(self.val_loader)}"
        )

        start_time = time.time()

        for epoch in range(self.num_epochs):
            train_loss = self.train_epoch()
            metrics = self.evaluate()

            # Track history
            current_lr = self.optimizer.param_groups[0]["lr"]
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(metrics.loss)
            self.history["val_auc"].append(metrics.auc_roc)
            self.history["val_accuracy"].append(metrics.accuracy)
            self.history["val_f1"].append(metrics.f1)
            self.history["lr"].append(current_lr)

            # Check for improvement
            if metrics.auc_roc > self.best_auc:
                self.best_auc = metrics.auc_roc
                self.best_epoch = epoch
                self.no_improve_count = 0
                self.save_checkpoint(is_best=True)
            else:
                self.no_improve_count += 1

            self.save_checkpoint(is_best=False)

            # Log progress
            logger.info(
                f"Epoch {epoch:3d} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {metrics.loss:.4f} | "
                f"Val AUC: {metrics.auc_roc:.4f} | "
                f"Val Acc: {metrics.accuracy:.4f} | "
                f"Best: {self.best_auc:.4f} (ep {self.best_epoch})"
            )

            # Early stopping
            if self.no_improve_count >= self.early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break

        training_time = str(timedelta(seconds=int(time.time() - start_time)))

        # Final evaluation with best model
        best_ckpt = torch.load(self.output_dir / "best.pt", weights_only=True)
        self.model.load_state_dict(best_ckpt["model_state_dict"])
        final_metrics = self.evaluate()

        # Save results
        results = {
            "head_type": self.head_type,
            "best_auc": self.best_auc,
            "best_epoch": self.best_epoch,
            "final_metrics": asdict(final_metrics),
            "history": self.history,
            "training_time": training_time,
            "num_params": sum(p.numel() for p in self.model.parameters()),
        }

        with open(self.output_dir / "results.json", "w") as f:
            json.dump(results, f, indent=2)

        return results


# =============================================================================
# Ablation Functions
# =============================================================================


def run_layer_ablation(
    feature_dir: str,
    layers: List[int],
    output_dir: str,
    head_type: str = "mean",
    is_pooled: bool = True,
    **train_kwargs,
) -> Dict[int, Dict]:
    """Run ablation over different DiT layers."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    for layer in layers:
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Layer Ablation: layer={layer}")
        logger.info(f"{'=' * 60}")

        layer_output = output_dir / f"layer_{layer}"

        train_loader, val_loader = create_dataloaders(
            feature_dir=feature_dir,
            layer=layer,
            is_pooled=is_pooled,
            **{
                k: v
                for k, v in train_kwargs.items()
                if k in ["batch_size", "num_workers", "timesteps"]
            },
        )

        sample = next(iter(train_loader))
        hidden_dim = sample["features"].shape[-1]

        model = create_physics_head(head_type, hidden_dim=hidden_dim)

        trainer = PhysicsHeadTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            output_dir=str(layer_output),
            head_type=head_type,
            **{
                k: v
                for k, v in train_kwargs.items()
                if k not in ["batch_size", "num_workers", "timesteps"]
            },
        )

        results[layer] = trainer.train()

    # Summary
    logger.info(f"\n{'=' * 60}")
    logger.info("LAYER ABLATION SUMMARY")
    logger.info(f"{'=' * 60}")
    for layer, res in sorted(results.items()):
        logger.info(
            f"Layer {layer:2d}: AUC={res['best_auc']:.4f} (epoch {res['best_epoch']})"
        )

    # Save summary
    summary = {
        layer: {"best_auc": r["best_auc"], "best_epoch": r["best_epoch"]}
        for layer, r in results.items()
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return results


def run_head_ablation(
    feature_dir: str,
    layer: int,
    output_dir: str,
    head_types: Optional[List[str]] = None,
    is_pooled: bool = True,
    **train_kwargs,
) -> Dict[str, Dict]:
    """Run ablation over different head architectures."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if head_types is None:
        head_types = list_heads()

    results = {}

    for head_type in head_types:
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Head Ablation: {head_type}")
        logger.info(f"{'=' * 60}")

        head_output = output_dir / head_type

        train_loader, val_loader = create_dataloaders(
            feature_dir=feature_dir,
            layer=layer,
            is_pooled=is_pooled,
            **{
                k: v
                for k, v in train_kwargs.items()
                if k in ["batch_size", "num_workers", "timesteps"]
            },
        )

        sample = next(iter(train_loader))
        hidden_dim = sample["features"].shape[-1]

        model = create_physics_head(head_type, hidden_dim=hidden_dim)
        logger.info(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

        trainer = PhysicsHeadTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            output_dir=str(head_output),
            head_type=head_type,
            **{
                k: v
                for k, v in train_kwargs.items()
                if k not in ["batch_size", "num_workers", "timesteps"]
            },
        )

        results[head_type] = trainer.train()

    # Summary
    logger.info(f"\n{'=' * 60}")
    logger.info("HEAD ABLATION SUMMARY")
    logger.info(f"{'=' * 60}")
    for head_type, res in results.items():
        logger.info(
            f"{head_type:15s}: AUC={res['best_auc']:.4f} (epoch {res['best_epoch']})"
        )

    # Save summary
    summary = {
        ht: {"best_auc": r["best_auc"], "best_epoch": r["best_epoch"]}
        for ht, r in results.items()
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return results


def run_seed_ablation(
    feature_dir: str,
    layer: int,
    output_dir: str,
    seeds: List[int],
    head_type: str = "mean",
    is_pooled: bool = True,
    **train_kwargs,
) -> Dict[int, Dict]:
    """Run ablation over different random seeds."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    for seed in seeds:
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Seed Ablation: seed={seed}")
        logger.info(f"{'=' * 60}")

        # Set seed for reproducibility
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        seed_output = output_dir / f"seed_{seed}"

        train_loader, val_loader = create_dataloaders(
            feature_dir=feature_dir,
            layer=layer,
            is_pooled=is_pooled,
            **{
                k: v
                for k, v in train_kwargs.items()
                if k in ["batch_size", "num_workers", "timesteps"]
            },
        )

        sample = next(iter(train_loader))
        hidden_dim = sample["features"].shape[-1]

        model = create_physics_head(head_type, hidden_dim=hidden_dim)

        trainer = PhysicsHeadTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            output_dir=str(seed_output),
            head_type=head_type,
            **{
                k: v
                for k, v in train_kwargs.items()
                if k not in ["batch_size", "num_workers", "timesteps"]
            },
        )

        results[seed] = trainer.train()

    # Summary
    logger.info(f"\n{'=' * 60}")
    logger.info("SEED ABLATION SUMMARY")
    logger.info(f"{'=' * 60}")

    aucs = [res["best_auc"] for res in results.values()]
    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)

    for seed, res in sorted(results.items()):
        logger.info(
            f"Seed {seed:3d}: AUC={res['best_auc']:.4f} (epoch {res['best_epoch']})"
        )
    logger.info(f"{'=' * 60}")
    logger.info(f"Mean AUC: {mean_auc:.4f} +/- {std_auc:.4f}")

    # Save summary
    summary = {
        "per_seed": {
            seed: {"best_auc": r["best_auc"], "best_epoch": r["best_epoch"]}
            for seed, r in results.items()
        },
        "mean_auc": mean_auc,
        "std_auc": std_auc,
        "num_seeds": len(seeds),
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return results


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Train Physics Discriminator Head",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Output directory structure:
    results/training/physics_head/{exp_name}_{timestamp}/
    ├── config.json
    ├── summary.json
    └── {variant}/
        ├── best.pt
        ├── latest.pt
        └── results.json
        """,
    )

    # Data arguments
    parser.add_argument(
        "--feature_dir", type=str, required=True, help="Features directory"
    )
    parser.add_argument("--metadata_file", type=str, default=None, help="Metadata file")
    parser.add_argument("--layer", type=int, default=15, help="DiT layer (default: 15)")
    parser.add_argument(
        "--timesteps", type=int, nargs="+", default=None, help="Timesteps"
    )

    # Model arguments
    parser.add_argument("--head_type", type=str, default="mean", help=f"Head type")

    # Training arguments
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--early_stopping", type=int, default=15)
    parser.add_argument(
        "--train-sa",
        action="store_true",
        default=False,
        help="Multi-task training: predict both physics and SA. "
        "Requires enriched_labels.json in feature directory.",
    )
    parser.add_argument(
        "--sa-weight",
        type=float,
        default=0.3,
        help="Weight for SA loss in multi-task training (default: 0.3)",
    )

    # Feature format
    parser.add_argument(
        "--is_pooled", action="store_true", help="Features are pre-pooled"
    )
    parser.add_argument("--no_pooled", dest="is_pooled", action="store_false")
    parser.set_defaults(is_pooled=True)

    # Ablation arguments
    parser.add_argument(
        "--ablation", type=str, choices=["layers", "heads", "seeds", None], default=None
    )
    parser.add_argument("--layers", type=int, nargs="+", default=[5, 10, 15, 20, 25])
    parser.add_argument("--heads", type=str, nargs="+", default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456, 789, 1024])

    # Output (new structure)
    parser.add_argument("--exp-name", type=str, default="train", help="Experiment name")
    parser.add_argument(
        "--results-base", type=str, default="results", help="Results base dir"
    )

    # Legacy output (deprecated)
    parser.add_argument(
        "--output_dir", type=str, default=None, help="[DEPRECATED] Use --exp-name"
    )

    args = parser.parse_args()

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    # Common training kwargs
    train_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "timesteps": args.timesteps,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "num_epochs": args.num_epochs,
        "early_stopping_patience": args.early_stopping,
        "device": device,
    }

    # Setup output directory
    if args.output_dir:
        # Legacy mode
        print("[WARNING] --output_dir is deprecated. Use --exp-name instead.")
        output_dir = args.output_dir
        rm = None
    else:
        # New unified structure
        rm = create_training_manager(
            exp_name=args.exp_name,
            ablation_type=args.ablation,
            results_base=args.results_base,
            layer=args.layer,
            head_type=args.head_type,
            lr=args.lr,
            batch_size=args.batch_size,
        )
        output_dir = str(rm.get_output_dir())
        logger.info(f"Output directory: {output_dir}")

    # Save config
    if rm:
        rm.save_config(
            {
                "feature_dir": args.feature_dir,
                "layer": args.layer,
                "head_type": args.head_type,
                "ablation": args.ablation,
                "is_pooled": args.is_pooled,
                **train_kwargs,
            }
        )

    # Run ablation or single training
    if args.ablation == "layers":
        run_layer_ablation(
            feature_dir=args.feature_dir,
            layers=args.layers,
            output_dir=output_dir,
            head_type=args.head_type,
            is_pooled=args.is_pooled,
            **train_kwargs,
        )

    elif args.ablation == "heads":
        run_head_ablation(
            feature_dir=args.feature_dir,
            layer=args.layer,
            output_dir=output_dir,
            head_types=args.heads,
            is_pooled=args.is_pooled,
            **train_kwargs,
        )

    elif args.ablation == "seeds":
        run_seed_ablation(
            feature_dir=args.feature_dir,
            layer=args.layer,
            output_dir=output_dir,
            seeds=args.seeds,
            head_type=args.head_type,
            is_pooled=args.is_pooled,
            **train_kwargs,
        )

    else:
        # Single training
        logger.info(f"Feature dir: {args.feature_dir}")
        logger.info(f"Layer: {args.layer}")
        logger.info(f"Head type: {args.head_type}")
        logger.info(f"Is pooled: {args.is_pooled}")

        train_loader, val_loader = create_dataloaders(
            feature_dir=args.feature_dir,
            metadata_file=args.metadata_file,
            layer=args.layer,
            timesteps=args.timesteps,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            is_pooled=args.is_pooled,
        )

        sample = next(iter(train_loader))
        feature_shape = sample["features"].shape
        hidden_dim = feature_shape[-1]
        logger.info(f"Feature shape: {feature_shape}")
        logger.info(f"Hidden dim: {hidden_dim}")

        model = create_physics_head(args.head_type, hidden_dim=hidden_dim)
        num_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Model params: {num_params:,}")

        # For single training, use a subdirectory
        if rm:
            single_output = rm.get_ablation_dir("heads", args.head_type)
        else:
            single_output = Path(output_dir) / args.head_type
            single_output.mkdir(parents=True, exist_ok=True)

        trainer = PhysicsHeadTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            output_dir=str(single_output),
            head_type=args.head_type,
            device=device,
            lr=args.lr,
            weight_decay=args.weight_decay,
            num_epochs=args.num_epochs,
            early_stopping_patience=args.early_stopping,
        )

        results = trainer.train()

        # Final summary
        logger.info(f"\n{'=' * 60}")
        logger.info("TRAINING COMPLETE")
        logger.info(f"{'=' * 60}")
        logger.info(
            f"Best AUC: {results['best_auc']:.4f} (epoch {results['best_epoch']})"
        )
        logger.info(f"Training time: {results['training_time']}")

        fm = results["final_metrics"]
        logger.info(f"Final Metrics:")
        logger.info(f"  AUC-ROC:   {fm['auc_roc']:.4f}")
        logger.info(f"  Accuracy:  {fm['accuracy']:.4f}")
        logger.info(f"  F1:        {fm['f1']:.4f}")

        if fm.get("auc_per_timestep"):
            logger.info(f"Per-timestep AUC:")
            for t, auc in sorted(fm["auc_per_timestep"].items()):
                logger.info(f"  t={t}: {auc:.4f}")


if __name__ == "__main__":
    main()
