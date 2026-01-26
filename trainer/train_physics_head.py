"""
Training Script for Physics Discriminator Head

Uses pre-extracted DiT features for fast training.
No DiT forward pass needed - just load .pt files from disk!

Usage:
    # Basic training
    python -m trainer.train_physics_head \
        --feature_dir data/Physion/features \
        --label_file data/Physion/labels.json \
        --layer 15
    
    # Layer ablation
    python -m trainer.train_physics_head \
        --feature_dir data/Physion/features \
        --label_file data/Physion/labels.json \
        --ablation \
        --ablation_layers 5 10 15 20 25
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import argparse
import logging
from pathlib import Path
from typing import Dict, Optional, List
from tqdm import tqdm
import json

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


class PhysicsHeadTrainer:
    """
    Trainer for physics discriminator head.

    Features are loaded from disk (pre-extracted), making training very fast.
    A full epoch on 2000 videos takes ~1 minute instead of hours!
    """

    def __init__(
        self,
        feature_dir: str,
        label_file: str,
        output_dir: str,
        layer: int = 15,
        timesteps: Optional[List[int]] = None,
        head_type: str = "simple",
        hidden_dim: int = 1920,
        mlp_hidden: int = 512,
        batch_size: int = 32,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        num_epochs: int = 100,
        device: str = "cuda",
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.num_epochs = num_epochs

        # Save config
        self.config = {
            "feature_dir": feature_dir,
            "label_file": label_file,
            "layer": layer,
            "timesteps": timesteps,
            "head_type": head_type,
            "hidden_dim": hidden_dim,
            "mlp_hidden": mlp_hidden,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "num_epochs": num_epochs,
        }

        # Create dataloaders
        logger.info("Creating dataloaders...")
        self.train_loader = self._create_dataloader(
            feature_dir,
            label_file,
            layer,
            timesteps,
            split="train",
            batch_size=batch_size,
        )
        self.val_loader = self._create_dataloader(
            feature_dir,
            label_file,
            layer,
            timesteps,
            split="test",
            batch_size=batch_size,
        )

        # Create model
        logger.info(f"Creating {head_type} physics head...")
        from src.models.physics_head import create_physics_head

        self.model = create_physics_head(
            head_type=head_type, hidden_dim=hidden_dim, mlp_hidden=mlp_hidden
        ).to(device)

        num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"Model parameters: {num_params:,}")

        # Optimizer and scheduler
        self.optimizer = AdamW(
            self.model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=num_epochs)

        # Loss function
        self.criterion = nn.BCEWithLogitsLoss()

        # Tracking
        self.best_val_acc = 0.0
        self.history = {"train_loss": [], "val_loss": [], "val_acc": []}

    def _create_dataloader(
        self,
        feature_dir: str,
        label_file: str,
        layer: int,
        timesteps: Optional[List[int]],
        split: str,
        batch_size: int,
    ) -> DataLoader:
        """Create dataloader from pre-extracted features."""
        from src.data.feature_dataset import FeatureDataset, collate_features

        dataset = FeatureDataset(
            feature_dir=feature_dir,
            label_file=label_file,
            layer=layer,
            timesteps=timesteps,
            split=split,
        )

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=4,
            collate_fn=collate_features,
            pin_memory=True,
        )

    def train_epoch(self) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in self.train_loader:
            features = batch["features"].to(self.device)
            labels = batch["labels"].to(self.device)

            # Forward
            logits = self.model(features).squeeze(-1)
            loss = self.criterion(logits, labels)

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / num_batches

    @torch.no_grad()
    def evaluate(self) -> Dict:
        """Evaluate on validation set."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch in self.val_loader:
            features = batch["features"].to(self.device)
            labels = batch["labels"].to(self.device)

            logits = self.model(features).squeeze(-1)
            loss = self.criterion(logits, labels)

            # Accuracy
            preds = (torch.sigmoid(logits) > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            total_loss += loss.item() * labels.size(0)

        return {
            "loss": total_loss / total if total > 0 else 0,
            "accuracy": correct / total if total > 0 else 0,
        }

    def train(self):
        """Full training loop."""
        logger.info("Starting training...")
        logger.info(f"  Train samples: {len(self.train_loader.dataset)}")
        logger.info(f"  Val samples: {len(self.val_loader.dataset)}")

        for epoch in range(self.num_epochs):
            # Train
            train_loss = self.train_epoch()
            self.scheduler.step()

            # Evaluate
            val_metrics = self.evaluate()

            # Log
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_metrics["loss"])
            self.history["val_acc"].append(val_metrics["accuracy"])

            logger.info(
                f"Epoch {epoch + 1}/{self.num_epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f} | "
                f"Val Acc: {val_metrics['accuracy']:.4f}"
            )

            # Save best model
            if val_metrics["accuracy"] > self.best_val_acc:
                self.best_val_acc = val_metrics["accuracy"]
                self.save_checkpoint("best.pt")
                logger.info(f"  ✓ New best! Acc: {self.best_val_acc:.4f}")

            # Periodic checkpoint
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(f"epoch_{epoch + 1}.pt")

        # Save final
        self.save_checkpoint("final.pt")
        self.save_history()

        logger.info(f"Training complete! Best val acc: {self.best_val_acc:.4f}")

    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        path = self.output_dir / filename
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "config": self.config,
                "best_val_acc": self.best_val_acc,
            },
            path,
        )

    def save_history(self):
        """Save training history."""
        with open(self.output_dir / "history.json", "w") as f:
            json.dump(self.history, f, indent=2)

        with open(self.output_dir / "config.json", "w") as f:
            json.dump(self.config, f, indent=2)


def run_layer_ablation(
    feature_dir: str,
    label_file: str,
    output_base: str,
    layers: List[int] = [5, 10, 15, 20, 25],
    **trainer_kwargs,
) -> Dict:
    """
    Run ablation study over different DiT layers.

    With pre-extracted features:
    - Without pre-extraction: 5 layers × 5 hours = 25 hours
    - With pre-extraction: 5 layers × 3 min = 15 minutes!
    """
    results = {}

    for layer in layers:
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Training with layer {layer}")
        logger.info(f"{'=' * 60}\n")

        output_dir = Path(output_base) / f"layer_{layer}"

        trainer = PhysicsHeadTrainer(
            feature_dir=feature_dir,
            label_file=label_file,
            output_dir=str(output_dir),
            layer=layer,
            **trainer_kwargs,
        )

        trainer.train()

        results[layer] = {
            "best_val_acc": trainer.best_val_acc,
            "final_train_loss": trainer.history["train_loss"][-1],
            "final_val_loss": trainer.history["val_loss"][-1],
        }

    # Save ablation results
    output_base = Path(output_base)
    output_base.mkdir(parents=True, exist_ok=True)
    with open(output_base / "ablation_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("ABLATION RESULTS")
    logger.info("=" * 60)
    for layer, metrics in sorted(results.items()):
        logger.info(f"Layer {layer}: Val Acc = {metrics['best_val_acc']:.4f}")

    best_layer = max(results.keys(), key=lambda l: results[l]["best_val_acc"])
    logger.info(
        f"\nBest layer: {best_layer} (acc: {results[best_layer]['best_val_acc']:.4f})"
    )

    return results


def main():
    parser = argparse.ArgumentParser(description="Train Physics Discriminator Head")

    # Data arguments
    parser.add_argument(
        "--feature_dir",
        type=str,
        required=True,
        help="Directory with pre-extracted features",
    )
    parser.add_argument(
        "--label_file", type=str, required=True, help="Path to labels JSON file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/physics_head",
        help="Output directory for checkpoints",
    )

    # Model arguments
    parser.add_argument("--layer", type=int, default=15, help="DiT layer to use")
    parser.add_argument(
        "--head_type",
        type=str,
        default="simple",
        choices=["simple", "temporal", "causal"],
        help="Physics head architecture",
    )
    parser.add_argument(
        "--mlp_hidden", type=int, default=512, help="MLP hidden dimension"
    )

    # Training arguments
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--device", type=str, default="cuda")

    # Ablation mode
    parser.add_argument(
        "--ablation", action="store_true", help="Run layer ablation study"
    )
    parser.add_argument(
        "--ablation_layers",
        type=int,
        nargs="+",
        default=[5, 10, 15, 20, 25],
        help="Layers to test in ablation",
    )

    args = parser.parse_args()

    if args.ablation:
        run_layer_ablation(
            feature_dir=args.feature_dir,
            label_file=args.label_file,
            output_base=args.output_dir,
            layers=args.ablation_layers,
            head_type=args.head_type,
            mlp_hidden=args.mlp_hidden,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            num_epochs=args.num_epochs,
            device=args.device,
        )
    else:
        trainer = PhysicsHeadTrainer(
            feature_dir=args.feature_dir,
            label_file=args.label_file,
            output_dir=args.output_dir,
            layer=args.layer,
            head_type=args.head_type,
            mlp_hidden=args.mlp_hidden,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            num_epochs=args.num_epochs,
            device=args.device,
        )
        trainer.train()


if __name__ == "__main__":
    main()
