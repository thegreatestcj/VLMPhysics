#!/usr/bin/env python3
"""
Training Script for Physics Discriminator Head

Trains lightweight MLP heads on pre-extracted (and pooled) DiT features.
Since features are pre-computed, training is extremely fast (~5 min for 100 epochs).

Usage:
    # Basic training with pooled features (FAST!)
    python -m trainer.train_physics_head \
        --feature_dir ~/scratch/physics/physion_features_pooled \
        --layer 15 \
        --is_pooled \
        --output_dir results/physics_head

    # Training with original features (SLOW, not recommended)
    python -m trainer.train_physics_head \
        --feature_dir ~/scratch/physics/physion_features \
        --layer 15 \
        --output_dir results/physics_head

    # Layer ablation
    python -m trainer.train_physics_head \
        --feature_dir ~/scratch/physics/physion_features_pooled \
        --ablation layers \
        --layers 5 10 15 20 25 \
        --is_pooled \
        --output_dir results/layer_ablation

    # Head type ablation  
    python -m trainer.train_physics_head \
        --feature_dir ~/scratch/physics/physion_features_pooled \
        --ablation heads \
        --layer 15 \
        --is_pooled \
        --output_dir results/head_ablation
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

from src.data.feature_dataset import create_dataloaders
from src.models.physics_head import create_physics_head, list_heads

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
    """
    Compute evaluation metrics.

    Args:
        all_logits: [N] tensor of logits
        all_labels: [N] tensor of labels (0 or 1)
        all_timesteps: [N] tensor of timesteps
        loss: Average loss value

    Returns:
        Metrics dataclass
    """
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
    probs = 1 / (1 + np.exp(-logits_np))  # sigmoid
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
    """
    Trainer for physics discriminator head.

    Features are loaded from disk (pre-extracted), making training very fast.
    """

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
        warmup_ratio: float = 0.1,
        early_stopping_patience: int = 15,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.output_dir = Path(output_dir)
        self.head_type = head_type
        self.device = device
        self.num_epochs = num_epochs
        self.early_stopping_patience = early_stopping_patience

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Loss function
        self.criterion = nn.BCEWithLogitsLoss()

        # Optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

        # Scheduler: OneCycleLR with warmup
        total_steps = len(train_loader) * num_epochs
        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=lr,
            total_steps=total_steps,
            pct_start=warmup_ratio,
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
        # Only "mean" (MeanPool) doesn't require timestep
        # All other heads use timestep in some form (AdaLN, concat, bias, etc.)
        return self.head_type != "mean"

    def _forward(self, batch: Dict) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass handling different head types.

        Returns:
            (logits, labels, timesteps) all as tensors
        """
        features = batch["features"].to(self.device)  # [B, T, D]
        labels = batch["labels"].to(self.device).float()  # [B]
        timesteps = batch["timesteps"]  # [B]
        if isinstance(timesteps, list):
            timesteps = torch.tensor(timesteps, device=self.device)
        elif isinstance(timesteps, int):
            timesteps = torch.tensor([timesteps], device=self.device)
        else:
            timesteps = timesteps.to(self.device)

        # Forward pass
        if self._requires_timestep():
            logits = self.model(features, timesteps)  # [B, 1]
        else:
            logits = self.model(features)  # [B, 1]

        logits = logits.squeeze(-1)  # [B]

        return logits, labels, timesteps

    def train_epoch(self) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in self.train_loader:
            logits, labels, _ = self._forward(batch)

            loss = self.criterion(logits, labels)

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

        all_logits = []
        all_labels = []
        all_timesteps = []
        total_loss = 0.0
        num_batches = 0

        for batch in self.val_loader:
            logits, labels, timesteps = self._forward(batch)

            loss = self.criterion(logits, labels)
            total_loss += loss.item()
            num_batches += 1

            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())
            all_timesteps.append(timesteps.cpu())

        # Concatenate
        all_logits = torch.cat(all_logits)
        all_labels = torch.cat(all_labels)
        all_timesteps = torch.cat(all_timesteps)

        # Compute metrics
        avg_loss = total_loss / max(num_batches, 1)
        metrics = compute_metrics(all_logits, all_labels, all_timesteps, avg_loss)

        return metrics

    def save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_auc": self.best_auc,
            "best_epoch": self.best_epoch,
            "head_type": self.head_type,
        }

        # Save latest
        torch.save(checkpoint, self.output_dir / "latest.pt")

        # Save best
        if is_best:
            torch.save(checkpoint, self.output_dir / "best.pt")

    def train(self) -> Dict:
        """
        Full training loop.

        Returns:
            Dictionary with training results
        """
        logger.info(f"Starting training for {self.num_epochs} epochs...")
        logger.info(
            f"Train batches: {len(self.train_loader)}, Val batches: {len(self.val_loader)}"
        )

        start_time = time.time()

        for epoch in range(self.num_epochs):
            # Train
            train_loss = self.train_epoch()

            # Evaluate
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

            # Save latest
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

        # Training time
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

        # Create dataloaders
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

        # Get hidden dim from first batch
        sample = next(iter(train_loader))
        hidden_dim = sample["features"].shape[-1]

        # Create model
        model = create_physics_head(head_type, hidden_dim=hidden_dim)

        # Train
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
    with open(output_dir / "layer_summary.json", "w") as f:
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

    # Default head types
    if head_types is None:
        head_types = list_heads()

    results = {}

    for head_type in head_types:
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Head Ablation: {head_type}")
        logger.info(f"{'=' * 60}")

        head_output = output_dir / head_type

        # Create dataloaders
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

        # Get hidden dim
        sample = next(iter(train_loader))
        hidden_dim = sample["features"].shape[-1]

        # Create model
        model = create_physics_head(head_type, hidden_dim=hidden_dim)
        logger.info(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

        # Train
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
    with open(output_dir / "head_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return results


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Train Physics Discriminator Head",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Data arguments
    parser.add_argument(
        "--feature_dir", type=str, required=True, help="Path to features directory"
    )
    parser.add_argument(
        "--label_file",
        type=str,
        default=None,
        help="Path to labels.json (default: feature_dir/labels.json)",
    )
    parser.add_argument(
        "--layer", type=int, default=15, help="DiT layer to use (default: 15)"
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        nargs="+",
        default=None,
        help="Timesteps to use (default: 200 400 600 800)",
    )

    # Model arguments
    parser.add_argument(
        "--head_type",
        type=str,
        default="mean",
        help=f"Head architecture (choices: {list_heads()})",
    )

    # Training arguments
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--early_stopping", type=int, default=15)

    # Feature format
    parser.add_argument(
        "--is_pooled",
        action="store_true",
        help="Features are pre-pooled [T,D] (use with pooled features)",
    )
    parser.add_argument(
        "--no_pooled",
        dest="is_pooled",
        action="store_false",
        help="Features are original [num_patches,D]",
    )
    parser.set_defaults(is_pooled=True)

    # Ablation arguments
    parser.add_argument(
        "--ablation",
        type=str,
        choices=["layers", "heads", None],
        default=None,
        help="Run ablation study",
    )
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=[5, 10, 15, 20, 25],
        help="Layers for layer ablation",
    )
    parser.add_argument(
        "--heads",
        type=str,
        nargs="+",
        default=None,
        help="Head types for head ablation",
    )

    # Output
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/physics_head",
        help="Output directory",
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

    # Run ablation or single training
    if args.ablation == "layers":
        run_layer_ablation(
            feature_dir=args.feature_dir,
            layers=args.layers,
            output_dir=args.output_dir,
            head_type=args.head_type,
            is_pooled=args.is_pooled,
            **train_kwargs,
        )

    elif args.ablation == "heads":
        run_head_ablation(
            feature_dir=args.feature_dir,
            layer=args.layer,
            output_dir=args.output_dir,
            head_types=args.heads,
            is_pooled=args.is_pooled,
            **train_kwargs,
        )

    else:
        # Single training
        logger.info(f"Feature dir: {args.feature_dir}")
        logger.info(f"Layer: {args.layer}")
        logger.info(f"Head type: {args.head_type}")
        logger.info(f"Is pooled: {args.is_pooled}")

        # Create dataloaders
        train_loader, val_loader = create_dataloaders(
            feature_dir=args.feature_dir,
            label_file=args.label_file,
            layer=args.layer,
            timesteps=args.timesteps,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            is_pooled=args.is_pooled,
        )

        # Get hidden dim
        sample = next(iter(train_loader))
        feature_shape = sample["features"].shape
        hidden_dim = feature_shape[-1]
        logger.info(f"Feature shape: {feature_shape}")
        logger.info(f"Hidden dim: {hidden_dim}")

        # Create model
        model = create_physics_head(args.head_type, hidden_dim=hidden_dim)
        num_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Model params: {num_params:,}")

        # Train
        trainer = PhysicsHeadTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            output_dir=args.output_dir,
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
