"""
Training Script for Physics Discriminator Head (AdaLN Version)

This script trains physics discriminator heads on pre-extracted DiT features.
Features are loaded from disk, making training extremely fast (~5 min for 100 epochs).

Supported head types (with AdaLN timestep conditioning):
    - mean:           Global mean pooling baseline (no timestep)
    - mean_adaln:     Mean pooling + AdaLN timestep modulation
    - temporal_adaln: Bidirectional attention + AdaLN-Zero
    - causal_adaln:   Causal attention + AdaLN-Zero (physics-aware)
    - multiview_adaln: Multi-view pooling + AdaLN (most comprehensive)

Usage:
    # Single training run
    python -m trainer.train_physics_head \
        --feature_dir /path/to/physion_features \
        --label_file /path/to/labels.json \
        --head_type causal_adaln \
        --output_dir results/causal_adaln

    # Head ablation (compare all heads)
    python -m trainer.train_physics_head \
        --feature_dir /path/to/physion_features \
        --label_file /path/to/labels.json \
        --ablation heads \
        --output_dir results/head_ablation

Evaluation Metrics:
    - AUC-ROC: Primary metric (threshold-independent ranking ability)
    - AUC-PR:  Average precision (handles class imbalance)
    - Accuracy, Precision, Recall, F1
    - Per-timestep AUC (understanding noise-level performance)

Author: VLMPhysics Project
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingLR

import numpy as np
from tqdm import tqdm

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.feature_dataset import (
    FeatureDataset,
    collate_features,
)
from src.models.physics_head import (
    create_physics_head,
    list_heads,
    get_head_info,
)

# Configure logging
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
class EvaluationMetrics:
    """Container for all evaluation metrics."""

    # Core metrics
    auc_roc: float = 0.0
    auc_pr: float = 0.0
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    loss: float = 0.0

    # Per-timestep breakdown
    auc_per_timestep: Dict[int, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    def summary_str(self) -> str:
        """Return a summary string."""
        return (
            f"AUC-ROC: {self.auc_roc:.4f} | "
            f"AUC-PR: {self.auc_pr:.4f} | "
            f"Acc: {self.accuracy:.4f} | "
            f"F1: {self.f1:.4f}"
        )


def compute_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    timesteps: Optional[List[int]] = None,
    threshold: float = 0.5,
) -> EvaluationMetrics:
    """
    Compute comprehensive evaluation metrics.

    Args:
        logits: [N] model outputs (before sigmoid)
        labels: [N] ground truth (0 or 1)
        timesteps: [N] optional timestep values (as list) for per-timestep analysis
        threshold: Classification threshold

    Returns:
        EvaluationMetrics dataclass
    """
    # Move to CPU and convert to numpy
    logits_np = logits.detach().cpu().numpy()
    labels_np = labels.detach().cpu().numpy()
    probs_np = 1 / (1 + np.exp(-logits_np))  # sigmoid
    preds_np = (probs_np > threshold).astype(np.float32)

    metrics = EvaluationMetrics()

    # === Core Metrics ===

    # Accuracy
    metrics.accuracy = (preds_np == labels_np).mean()

    # Precision, Recall, F1
    tp = ((preds_np == 1) & (labels_np == 1)).sum()
    fp = ((preds_np == 1) & (labels_np == 0)).sum()
    fn = ((preds_np == 0) & (labels_np == 1)).sum()

    metrics.precision = tp / (tp + fp + 1e-8)
    metrics.recall = tp / (tp + fn + 1e-8)
    metrics.f1 = (
        2
        * metrics.precision
        * metrics.recall
        / (metrics.precision + metrics.recall + 1e-8)
    )

    # AUC-ROC and AUC-PR
    try:
        from sklearn.metrics import roc_auc_score, average_precision_score

        if len(np.unique(labels_np)) > 1:  # Need both classes
            metrics.auc_roc = roc_auc_score(labels_np, probs_np)
            metrics.auc_pr = average_precision_score(labels_np, probs_np)
        else:
            metrics.auc_roc = metrics.accuracy
            metrics.auc_pr = metrics.accuracy

    except ImportError:
        logger.warning("sklearn not available, using accuracy as AUC proxy")
        metrics.auc_roc = metrics.accuracy
        metrics.auc_pr = metrics.accuracy

    # === Per-Timestep AUC ===

    if timesteps is not None and len(timesteps) > 0:
        timesteps_np = np.array(timesteps)
        unique_timesteps = np.unique(timesteps_np)

        for t in unique_timesteps:
            mask = timesteps_np == t
            if mask.sum() > 10 and len(np.unique(labels_np[mask])) > 1:
                try:
                    from sklearn.metrics import roc_auc_score

                    metrics.auc_per_timestep[int(t)] = roc_auc_score(
                        labels_np[mask], probs_np[mask]
                    )
                except:
                    pass

    return metrics


# =============================================================================
# Data Loading Utilities
# =============================================================================


def create_train_val_loaders(
    feature_dir: str,
    label_file: str,
    layer: int = 15,
    timesteps: Optional[List[int]] = None,
    batch_size: int = 32,
    num_workers: int = 4,
    val_ratio: float = 0.15,
    seed: int = 42,
    pool_spatial: bool = True,
    pool_temporal: bool = False,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders with proper splitting.

    FeatureDataset doesn't have built-in train/val split (only train/test),
    so we manually split the "train" set into train/val by video indices.

    Args:
        feature_dir: Directory with extracted features
        label_file: Path to labels JSON file
        layer: DiT layer to use
        timesteps: List of timesteps to use
        batch_size: Batch size
        num_workers: DataLoader workers
        val_ratio: Fraction for validation
        seed: Random seed for reproducible splits
        pool_spatial: Pool spatial dimensions
        pool_temporal: Pool temporal dimension

    Returns:
        (train_loader, val_loader)
    """
    # Create full dataset with split="train"
    # This loads all training data, we'll manually split into train/val
    full_dataset = FeatureDataset(
        feature_dir=feature_dir,
        label_file=label_file,
        layer=layer,
        timesteps=timesteps,
        split="train",
        pool_spatial=pool_spatial,
        pool_temporal=pool_temporal,
    )

    # Split by sample indices
    num_samples = len(full_dataset)
    indices = list(range(num_samples))

    # Shuffle with seed
    rng = np.random.RandomState(seed)
    rng.shuffle(indices)

    # Split
    n_val = int(num_samples * val_ratio)
    val_indices = indices[:n_val]
    train_indices = indices[n_val:]

    logger.info(
        f"Dataset split: {len(train_indices)} train, {len(val_indices)} val samples"
    )

    # Create subset datasets
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_features,
        pin_memory=True,
        drop_last=len(train_indices) > batch_size,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_features,
        pin_memory=True,
    )

    return train_loader, val_loader


# =============================================================================
# Trainer
# =============================================================================


class PhysicsHeadTrainer:
    """
    Trainer for physics discriminator head with AdaLN.

    Training is fast because features are pre-extracted:
    - ~1200 videos total
    - ~40 iterations per epoch (batch_size=32)
    - ~5 minutes for 100 epochs
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str = "cuda",
        lr: float = 1e-3,
        weight_decay: float = 0.01,
        num_epochs: int = 100,
        warmup_epochs: int = 5,
        output_dir: Optional[str] = None,
        head_type: str = "causal_adaln",
        early_stopping_patience: int = 20,
    ):
        """
        Args:
            model: Physics head model
            train_loader: Training data loader
            val_loader: Validation data loader
            device: Device to train on
            lr: Learning rate
            weight_decay: Weight decay for AdamW
            num_epochs: Number of epochs
            warmup_epochs: Warmup epochs for scheduler
            output_dir: Directory to save checkpoints
            head_type: Type of head (for logging)
            early_stopping_patience: Stop if no improvement for N epochs
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.num_epochs = num_epochs
        self.head_type = head_type
        self.early_stopping_patience = early_stopping_patience

        # Output directory
        self.output_dir = Path(output_dir) if output_dir else None
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)

        # Loss function
        self.criterion = nn.BCEWithLogitsLoss()

        # Optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
        )

        # Learning rate scheduler
        total_steps = num_epochs * len(train_loader)
        warmup_steps = warmup_epochs * len(train_loader)

        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=lr,
            total_steps=max(total_steps, 1),
            pct_start=min(warmup_steps / max(total_steps, 1), 0.3),
            anneal_strategy="cos",
        )

        # Tracking
        self.best_auc = 0.0
        self.best_epoch = 0
        self.epochs_without_improvement = 0

        self.history = {
            "train_loss": [],
            "val_loss": [],
            "val_auc": [],
            "val_accuracy": [],
            "val_f1": [],
            "lr": [],
        }

    def _requires_timestep(self) -> bool:
        """Check if the head type requires timestep input."""
        return self.head_type != "mean"

    def _forward(self, batch: Dict) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
        """
        Forward pass handling different head types.

        collate_features returns:
            - video_ids: List[str]
            - features: [B, T, D] or [B, T, h, w, D] tensor
            - labels: [B] tensor (注意是复数 labels!)
            - timesteps: List[int] (注意是列表，不是 tensor!)

        Returns:
            logits: [B] model predictions
            labels: [B] ground truth
            timesteps: List[int] timestep values
        """
        # Convert features to float32 (model uses float32, features are stored as float16)
        features = batch["features"].to(self.device, dtype=torch.float32)
        labels = batch["labels"].to(self.device)  # 复数 'labels'
        timesteps_list = batch["timesteps"]  # 列表，不是 tensor

        # Convert timesteps list to tensor for model
        timesteps_tensor = torch.tensor(
            timesteps_list, device=self.device, dtype=torch.long
        )

        # Forward pass
        if self._requires_timestep():
            logits = self.model(features, timesteps_tensor)
        else:
            logits = self.model(features)

        return logits.squeeze(-1), labels, timesteps_list

    def train_epoch(self) -> float:
        """Train for one epoch. Returns average loss."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(self.train_loader, desc="Training", leave=False)
        for batch in pbar:
            self.optimizer.zero_grad()

            logits, labels, _ = self._forward(batch)
            loss = self.criterion(logits, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.scheduler.step()

            total_loss += loss.item()
            num_batches += 1

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        return total_loss / max(num_batches, 1)

    @torch.no_grad()
    def evaluate(self) -> EvaluationMetrics:
        """Evaluate on validation set."""
        self.model.eval()

        all_logits = []
        all_labels = []
        all_timesteps = []
        total_loss = 0.0
        num_batches = 0

        for batch in self.val_loader:
            logits, labels, timesteps_list = self._forward(batch)
            loss = self.criterion(logits, labels)

            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())
            all_timesteps.extend(timesteps_list)  # extend 列表
            total_loss += loss.item()
            num_batches += 1

        if num_batches == 0:
            return EvaluationMetrics()

        # Concatenate tensors
        all_logits = torch.cat(all_logits)
        all_labels = torch.cat(all_labels)

        # Compute metrics
        metrics = compute_metrics(
            all_logits,
            all_labels,
            timesteps=all_timesteps,
        )
        metrics.loss = total_loss / num_batches

        return metrics

    def save_checkpoint(
        self, epoch: int, metrics: EvaluationMetrics, is_best: bool = False
    ):
        """Save model checkpoint."""
        if self.output_dir is None:
            return

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": metrics.to_dict(),
            "head_type": self.head_type,
            "history": self.history,
        }

        # Save latest
        torch.save(checkpoint, self.output_dir / "latest.pt")

        # Save best
        if is_best:
            torch.save(checkpoint, self.output_dir / "best.pt")

    def train(self) -> Dict[str, Any]:
        """
        Full training loop.

        Returns:
            Dictionary with training results
        """
        logger.info(f"Starting training: {self.head_type}")
        logger.info(f"  - Epochs: {self.num_epochs}")
        logger.info(f"  - Train batches: {len(self.train_loader)}")
        logger.info(f"  - Val batches: {len(self.val_loader)}")
        logger.info(f"  - Parameters: {self.model.get_num_params():,}")

        start_time = datetime.now()

        for epoch in range(1, self.num_epochs + 1):
            # Train
            train_loss = self.train_epoch()

            # Evaluate
            metrics = self.evaluate()

            # Track history
            current_lr = self.scheduler.get_last_lr()[0]
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(metrics.loss)
            self.history["val_auc"].append(metrics.auc_roc)
            self.history["val_accuracy"].append(metrics.accuracy)
            self.history["val_f1"].append(metrics.f1)
            self.history["lr"].append(current_lr)

            # Check for improvement
            is_best = metrics.auc_roc > self.best_auc
            if is_best:
                self.best_auc = metrics.auc_roc
                self.best_epoch = epoch
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1

            # Save checkpoint
            self.save_checkpoint(epoch, metrics, is_best)

            # Logging
            if epoch % 10 == 0 or epoch == 1 or is_best:
                logger.info(
                    f"Epoch {epoch:3d}/{self.num_epochs} | "
                    f"Train Loss: {train_loss:.4f} | "
                    f"Val Loss: {metrics.loss:.4f} | "
                    f"AUC: {metrics.auc_roc:.4f} {'*' if is_best else ''} | "
                    f"Acc: {metrics.accuracy:.4f}"
                )

            # Early stopping
            if self.epochs_without_improvement >= self.early_stopping_patience:
                logger.info(
                    f"Early stopping at epoch {epoch} (no improvement for {self.early_stopping_patience} epochs)"
                )
                break

        # Training complete
        elapsed = datetime.now() - start_time
        logger.info(f"Training completed in {elapsed}")
        logger.info(f"Best AUC: {self.best_auc:.4f} (epoch {self.best_epoch})")

        # Final evaluation with best model
        if self.output_dir and (self.output_dir / "best.pt").exists():
            best_ckpt = torch.load(self.output_dir / "best.pt", weights_only=False)
            self.model.load_state_dict(best_ckpt["model_state_dict"])

        final_metrics = self.evaluate()

        # Save final results
        results = {
            "head_type": self.head_type,
            "best_auc": self.best_auc,
            "best_epoch": self.best_epoch,
            "final_metrics": final_metrics.to_dict(),
            "history": self.history,
            "training_time": str(elapsed),
            "num_params": self.model.get_num_params(),
        }

        if self.output_dir:
            with open(self.output_dir / "results.json", "w") as f:
                json.dump(results, f, indent=2)

        return results


# =============================================================================
# Ablation Studies
# =============================================================================


def run_head_ablation(
    feature_dir: str,
    label_file: str,
    layer: int,
    output_dir: str,
    head_types: Optional[List[str]] = None,
    **train_kwargs,
) -> Dict[str, Dict]:
    """
    Run ablation study comparing different head architectures.
    """
    if head_types is None:
        head_types = list_heads()

    results = {}
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("HEAD ARCHITECTURE ABLATION")
    logger.info("=" * 70)
    logger.info(f"Heads to test: {head_types}")
    logger.info(f"Layer: {layer}")

    for head_type in head_types:
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Training: {head_type}")
        logger.info(f"{'=' * 60}")

        try:
            # Create dataloaders
            train_loader, val_loader = create_train_val_loaders(
                feature_dir=feature_dir,
                label_file=label_file,
                layer=layer,
                batch_size=train_kwargs.get("batch_size", 32),
                num_workers=train_kwargs.get("num_workers", 4),
                pool_spatial=True,
                val_ratio=train_kwargs.get("val_ratio", 0.15),
                seed=train_kwargs.get("seed", 42),
            )

            # Get feature dimension from a sample
            sample_batch = next(iter(train_loader))
            feature_shape = sample_batch["features"].shape
            hidden_dim = feature_shape[-1]
            logger.info(f"Feature shape: {feature_shape}")

            # Create model
            model = create_physics_head(head_type, hidden_dim=hidden_dim)
            logger.info(f"Parameters: {model.get_num_params():,}")

            # Train
            trainer = PhysicsHeadTrainer(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                output_dir=output_path / head_type,
                head_type=head_type,
                device=train_kwargs.get("device", "cuda"),
                lr=train_kwargs.get("lr", 1e-3),
                weight_decay=train_kwargs.get("weight_decay", 0.01),
                num_epochs=train_kwargs.get("num_epochs", 100),
                warmup_epochs=train_kwargs.get("warmup_epochs", 5),
                early_stopping_patience=train_kwargs.get("early_stopping_patience", 20),
            )

            head_results = trainer.train()
            results[head_type] = head_results

        except Exception as e:
            logger.error(f"Head {head_type} failed: {e}")
            import traceback

            traceback.print_exc()
            results[head_type] = {"error": str(e)}

    # Print summary
    logger.info("\n" + "=" * 70)
    logger.info("HEAD ABLATION SUMMARY")
    logger.info("=" * 70)
    logger.info(
        f"{'Head':<20} {'AUC-ROC':>10} {'Accuracy':>10} {'F1':>10} {'Params':>12}"
    )
    logger.info("-" * 65)

    for head_type in head_types:
        r = results.get(head_type, {})
        if "error" in r:
            logger.info(f"{head_type:<20} ERROR: {r['error'][:30]}")
        else:
            fm = r.get("final_metrics", {})
            logger.info(
                f"{head_type:<20} "
                f"{r.get('best_auc', 0):>10.4f} "
                f"{fm.get('accuracy', 0):>10.4f} "
                f"{fm.get('f1', 0):>10.4f} "
                f"{r.get('num_params', 0):>12,}"
            )

    # Save summary
    with open(output_path / "head_ablation_summary.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


def run_layer_ablation(
    feature_dir: str,
    label_file: str,
    layers: List[int],
    output_dir: str,
    head_type: str = "causal_adaln",
    **train_kwargs,
) -> Dict[int, Dict]:
    """
    Run ablation study comparing different DiT layers.
    """
    results = {}
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("LAYER ABLATION")
    logger.info("=" * 70)
    logger.info(f"Layers to test: {layers}")
    logger.info(f"Head type: {head_type}")

    for layer in layers:
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Training: Layer {layer}")
        logger.info(f"{'=' * 60}")

        try:
            train_loader, val_loader = create_train_val_loaders(
                feature_dir=feature_dir,
                label_file=label_file,
                layer=layer,
                batch_size=train_kwargs.get("batch_size", 32),
                num_workers=train_kwargs.get("num_workers", 4),
                pool_spatial=True,
                val_ratio=train_kwargs.get("val_ratio", 0.15),
                seed=train_kwargs.get("seed", 42),
            )

            sample_batch = next(iter(train_loader))
            hidden_dim = sample_batch["features"].shape[-1]

            model = create_physics_head(head_type, hidden_dim=hidden_dim)

            trainer = PhysicsHeadTrainer(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                output_dir=output_path / f"layer_{layer}",
                head_type=head_type,
                device=train_kwargs.get("device", "cuda"),
                lr=train_kwargs.get("lr", 1e-3),
                weight_decay=train_kwargs.get("weight_decay", 0.01),
                num_epochs=train_kwargs.get("num_epochs", 100),
                warmup_epochs=train_kwargs.get("warmup_epochs", 5),
                early_stopping_patience=train_kwargs.get("early_stopping_patience", 20),
            )

            layer_results = trainer.train()
            results[layer] = layer_results

        except Exception as e:
            logger.error(f"Layer {layer} failed: {e}")
            results[layer] = {"error": str(e)}

    # Print summary
    logger.info("\n" + "=" * 70)
    logger.info("LAYER ABLATION SUMMARY")
    logger.info("=" * 70)
    logger.info(f"{'Layer':<10} {'AUC-ROC':>10} {'Accuracy':>10}")
    logger.info("-" * 35)

    for layer in sorted(results.keys()):
        r = results[layer]
        if "error" in r:
            logger.info(f"{layer:<10} ERROR")
        else:
            fm = r.get("final_metrics", {})
            logger.info(
                f"{layer:<10} "
                f"{r.get('best_auc', 0):>10.4f} "
                f"{fm.get('accuracy', 0):>10.4f}"
            )

    # Save summary
    with open(output_path / "layer_ablation_summary.json", "w") as f:
        json.dump({str(k): v for k, v in results.items()}, f, indent=2)

    return results


def run_timestep_ablation(
    feature_dir: str,
    label_file: str,
    layer: int,
    output_dir: str,
    timestep_configs: Optional[List[List[int]]] = None,
    head_type: str = "causal_adaln",
    **train_kwargs,
) -> Dict[str, Dict]:
    """
    Run ablation study comparing different timestep configurations.
    """
    if timestep_configs is None:
        timestep_configs = [
            [200],
            [400],
            [600],
            [800],
            [200, 400, 600, 800],
        ]

    results = {}
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("TIMESTEP ABLATION")
    logger.info("=" * 70)

    for timesteps in timestep_configs:
        config_name = "_".join(map(str, timesteps))
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Training: timesteps={timesteps}")
        logger.info(f"{'=' * 60}")

        try:
            train_loader, val_loader = create_train_val_loaders(
                feature_dir=feature_dir,
                label_file=label_file,
                layer=layer,
                timesteps=timesteps,
                batch_size=train_kwargs.get("batch_size", 32),
                num_workers=train_kwargs.get("num_workers", 4),
                pool_spatial=True,
                val_ratio=train_kwargs.get("val_ratio", 0.15),
                seed=train_kwargs.get("seed", 42),
            )

            sample_batch = next(iter(train_loader))
            hidden_dim = sample_batch["features"].shape[-1]

            model = create_physics_head(head_type, hidden_dim=hidden_dim)

            trainer = PhysicsHeadTrainer(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                output_dir=output_path / f"t_{config_name}",
                head_type=head_type,
                device=train_kwargs.get("device", "cuda"),
                lr=train_kwargs.get("lr", 1e-3),
                weight_decay=train_kwargs.get("weight_decay", 0.01),
                num_epochs=train_kwargs.get("num_epochs", 100),
                warmup_epochs=train_kwargs.get("warmup_epochs", 5),
                early_stopping_patience=train_kwargs.get("early_stopping_patience", 20),
            )

            ts_results = trainer.train()
            results[config_name] = ts_results

        except Exception as e:
            logger.error(f"Timesteps {timesteps} failed: {e}")
            results[config_name] = {"error": str(e)}

    # Print summary
    logger.info("\n" + "=" * 70)
    logger.info("TIMESTEP ABLATION SUMMARY")
    logger.info("=" * 70)
    logger.info(f"{'Timesteps':<25} {'AUC-ROC':>10} {'Accuracy':>10}")
    logger.info("-" * 50)

    for config_name in results:
        r = results[config_name]
        if "error" in r:
            logger.info(f"{config_name:<25} ERROR")
        else:
            fm = r.get("final_metrics", {})
            logger.info(
                f"{config_name:<25} "
                f"{r.get('best_auc', 0):>10.4f} "
                f"{fm.get('accuracy', 0):>10.4f}"
            )

    # Save summary
    with open(output_path / "timestep_ablation_summary.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Train physics discriminator head (AdaLN version)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single training run
  python -m trainer.train_physics_head \\
      --feature_dir /path/to/features \\
      --label_file /path/to/labels.json \\
      --head_type causal_adaln

  # Head ablation
  python -m trainer.train_physics_head \\
      --feature_dir /path/to/features \\
      --label_file /path/to/labels.json \\
      --ablation heads

  # Layer ablation
  python -m trainer.train_physics_head \\
      --feature_dir /path/to/features \\
      --label_file /path/to/labels.json \\
      --ablation layers \\
      --layers 5 10 15 20 25
        """,
    )

    # Data arguments
    parser.add_argument(
        "--feature_dir",
        type=str,
        required=True,
        help="Directory containing extracted features",
    )
    parser.add_argument(
        "--label_file", type=str, required=True, help="Path to labels JSON file"
    )
    parser.add_argument(
        "--layer", type=int, default=15, help="DiT layer to use (default: 15)"
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        nargs="+",
        default=None,
        help="Timesteps to use for training (default: all available)",
    )

    # Model arguments
    parser.add_argument(
        "--head_type",
        type=str,
        default="causal_adaln",
        choices=list_heads(),
        help="Physics head architecture",
    )

    # Training arguments
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument(
        "--val_ratio", type=float, default=0.15, help="Validation set ratio"
    )
    parser.add_argument(
        "--early_stopping",
        type=int,
        default=20,
        help="Early stopping patience (epochs)",
    )

    # Ablation arguments
    parser.add_argument(
        "--ablation",
        type=str,
        choices=["heads", "layers", "timesteps"],
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
        "--head_types",
        type=str,
        nargs="+",
        default=None,
        help="Specific heads for head ablation",
    )

    # Output arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/physics_head",
        help="Output directory",
    )
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")
    if device == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name()}")

    # Common training kwargs
    train_kwargs = {
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "num_epochs": args.num_epochs,
        "warmup_epochs": args.warmup_epochs,
        "device": device,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "val_ratio": args.val_ratio,
        "seed": args.seed,
        "early_stopping_patience": args.early_stopping,
    }

    # Run ablation or single training
    if args.ablation == "heads":
        run_head_ablation(
            feature_dir=args.feature_dir,
            label_file=args.label_file,
            layer=args.layer,
            output_dir=args.output_dir,
            head_types=args.head_types,
            **train_kwargs,
        )

    elif args.ablation == "layers":
        run_layer_ablation(
            feature_dir=args.feature_dir,
            label_file=args.label_file,
            layers=args.layers,
            output_dir=args.output_dir,
            head_type=args.head_type,
            **train_kwargs,
        )

    elif args.ablation == "timesteps":
        run_timestep_ablation(
            feature_dir=args.feature_dir,
            label_file=args.label_file,
            layer=args.layer,
            output_dir=args.output_dir,
            head_type=args.head_type,
            **train_kwargs,
        )

    else:
        # Single training run
        logger.info("=" * 70)
        logger.info("SINGLE TRAINING RUN")
        logger.info("=" * 70)
        logger.info(f"Feature dir: {args.feature_dir}")
        logger.info(f"Label file: {args.label_file}")
        logger.info(f"Layer: {args.layer}")
        logger.info(f"Head type: {args.head_type}")
        logger.info(f"Timesteps: {args.timesteps or 'all available'}")

        # Create dataloaders
        train_loader, val_loader = create_train_val_loaders(
            feature_dir=args.feature_dir,
            label_file=args.label_file,
            layer=args.layer,
            timesteps=args.timesteps,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pool_spatial=True,
            val_ratio=args.val_ratio,
            seed=args.seed,
        )

        # Get feature dimension
        sample_batch = next(iter(train_loader))
        feature_shape = sample_batch["features"].shape
        hidden_dim = feature_shape[-1]
        logger.info(f"Feature shape: {feature_shape}")
        logger.info(f"Hidden dim: {hidden_dim}")

        # Create model
        model = create_physics_head(args.head_type, hidden_dim=hidden_dim)
        logger.info(f"Model: {args.head_type}")
        logger.info(f"Parameters: {model.get_num_params():,}")

        # Print head info
        info = get_head_info().get(args.head_type, {})
        logger.info(f"Description: {info.get('description', 'N/A')}")

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
            warmup_epochs=args.warmup_epochs,
            early_stopping_patience=args.early_stopping,
        )

        results = trainer.train()

        # Final summary
        logger.info("\n" + "=" * 70)
        logger.info("TRAINING COMPLETE")
        logger.info("=" * 70)
        logger.info(
            f"Best AUC-ROC: {results['best_auc']:.4f} (epoch {results['best_epoch']})"
        )

        fm = results["final_metrics"]
        logger.info(f"Final Metrics:")
        logger.info(f"  AUC-ROC:   {fm['auc_roc']:.4f}")
        logger.info(f"  AUC-PR:    {fm['auc_pr']:.4f}")
        logger.info(f"  Accuracy:  {fm['accuracy']:.4f}")
        logger.info(f"  F1:        {fm['f1']:.4f}")
        logger.info(f"  Precision: {fm['precision']:.4f}")
        logger.info(f"  Recall:    {fm['recall']:.4f}")

        if fm.get("auc_per_timestep"):
            logger.info(f"Per-timestep AUC:")
            for t, auc in sorted(fm["auc_per_timestep"].items()):
                logger.info(f"  t={t}: {auc:.4f}")

        logger.info(f"\nResults saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
