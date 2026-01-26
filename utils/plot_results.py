"""
Visualization Script for Physics Head Ablation Results

Generates publication-ready figures for ablation studies.

Usage:
    # Plot head ablation results
    python -m trainer.plot_results \
        --results_dir results/head_ablation \
        --plot_type head_ablation

    # Plot layer ablation results
    python -m trainer.plot_results \
        --results_dir results/layer_ablation \
        --plot_type layer_ablation

    # Plot training curves for a single run
    python -m trainer.plot_results \
        --results_dir results/head_ablation/causal_adaln \
        --plot_type training_curves

    # Plot all (auto-detect)
    python -m trainer.plot_results --results_dir results --plot_type all

Output:
    results/figures/
    ├── head_ablation_comparison.pdf
    ├── layer_ablation_comparison.pdf
    ├── timestep_ablation_comparison.pdf
    ├── training_curves_*.pdf
    └── per_timestep_analysis.pdf
"""

import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np

# Check for matplotlib
try:
    import matplotlib.pyplot as plt
    import matplotlib as mpl

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("WARNING: matplotlib not installed. Install with: pip install matplotlib")

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)


# =============================================================================
# Style Configuration (publication-ready)
# =============================================================================


def setup_style():
    """Setup matplotlib style for publication-quality figures."""
    if not HAS_MATPLOTLIB:
        return

    plt.style.use("seaborn-v0_8-whitegrid")

    mpl.rcParams.update(
        {
            # Font
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 9,
            # Figure
            "figure.figsize": (6, 4),
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.1,
            # Lines
            "lines.linewidth": 1.5,
            "lines.markersize": 6,
            # Axes
            "axes.linewidth": 1,
            "axes.grid": True,
            "grid.alpha": 0.3,
        }
    )


# Color palette for different heads
HEAD_COLORS = {
    "mean": "#1f77b4",  # Blue
    "mean_adaln": "#ff7f0e",  # Orange
    "temporal_adaln": "#2ca02c",  # Green
    "causal_adaln": "#d62728",  # Red
    "multiview_adaln": "#9467bd",  # Purple
}

# Nicer display names
HEAD_NAMES = {
    "mean": "MeanPool",
    "mean_adaln": "MeanPool+AdaLN",
    "temporal_adaln": "Temporal+AdaLN",
    "causal_adaln": "Causal+AdaLN",
    "multiview_adaln": "MultiView+AdaLN",
}


# =============================================================================
# Data Loading
# =============================================================================


def load_results(results_dir: Path) -> Dict[str, Any]:
    """Load results.json from a directory."""
    results_file = results_dir / "results.json"
    if results_file.exists():
        with open(results_file) as f:
            return json.load(f)
    return {}


def load_ablation_summary(results_dir: Path, ablation_type: str) -> Dict[str, Any]:
    """Load ablation summary JSON."""
    summary_file = results_dir / f"{ablation_type}_summary.json"
    if summary_file.exists():
        with open(summary_file) as f:
            return json.load(f)

    # Try to find it
    for f in results_dir.glob("*_summary.json"):
        with open(f) as fp:
            return json.load(fp)

    return {}


# =============================================================================
# Plotting Functions
# =============================================================================


def plot_head_ablation(results_dir: Path, output_dir: Path):
    """
    Plot head architecture comparison.

    Creates a bar chart comparing AUC-ROC across different head types.
    """
    if not HAS_MATPLOTLIB:
        logger.error("matplotlib required for plotting")
        return

    # Load summary
    summary = load_ablation_summary(results_dir, "head_ablation")
    if not summary:
        # Try loading individual results
        summary = {}
        for head_dir in results_dir.iterdir():
            if head_dir.is_dir():
                results = load_results(head_dir)
                if results:
                    summary[head_dir.name] = results

    if not summary:
        logger.error(f"No results found in {results_dir}")
        return

    # Extract data
    heads = []
    aucs = []
    accs = []
    f1s = []
    params = []

    # Order by expected performance
    head_order = [
        "mean",
        "mean_adaln",
        "temporal_adaln",
        "causal_adaln",
        "multiview_adaln",
    ]

    for head in head_order:
        if head in summary and "error" not in summary[head]:
            heads.append(head)
            aucs.append(summary[head].get("best_auc", 0))
            fm = summary[head].get("final_metrics", {})
            accs.append(fm.get("accuracy", 0))
            f1s.append(fm.get("f1", 0))
            params.append(summary[head].get("num_params", 0))

    if not heads:
        logger.error("No valid results to plot")
        return

    # Create figure with 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # === Plot 1: AUC-ROC Comparison ===
    ax1 = axes[0]
    x = np.arange(len(heads))
    colors = [HEAD_COLORS.get(h, "#333333") for h in heads]
    labels = [HEAD_NAMES.get(h, h) for h in heads]

    bars = ax1.bar(x, aucs, color=colors, edgecolor="black", linewidth=0.5)

    # Add value labels on bars
    for bar, auc in zip(bars, aucs):
        height = bar.get_height()
        ax1.annotate(
            f"{auc:.3f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax1.set_xlabel("Head Architecture")
    ax1.set_ylabel("AUC-ROC")
    ax1.set_title("Physics Discriminator: Head Architecture Comparison")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=15, ha="right")
    ax1.set_ylim(0.5, max(aucs) * 1.1)
    ax1.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="Random")

    # === Plot 2: Metrics Comparison ===
    ax2 = axes[1]
    width = 0.25

    bars1 = ax2.bar(x - width, aucs, width, label="AUC-ROC", color="#2ecc71")
    bars2 = ax2.bar(x, accs, width, label="Accuracy", color="#3498db")
    bars3 = ax2.bar(x + width, f1s, width, label="F1", color="#e74c3c")

    ax2.set_xlabel("Head Architecture")
    ax2.set_ylabel("Score")
    ax2.set_title("Comparison of Multiple Metrics")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=15, ha="right")
    ax2.set_ylim(0.4, 1.0)
    ax2.legend(loc="lower right")

    plt.tight_layout()

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "head_ablation_comparison.pdf"
    plt.savefig(output_path)
    plt.savefig(output_path.with_suffix(".png"))
    logger.info(f"Saved: {output_path}")

    plt.close()

    # === Additional: Parameter vs Performance ===
    if any(p > 0 for p in params):
        fig, ax = plt.subplots(figsize=(8, 5))

        for i, (head, auc, param) in enumerate(zip(heads, aucs, params)):
            color = HEAD_COLORS.get(head, "#333333")
            label = HEAD_NAMES.get(head, head)
            ax.scatter(param / 1e6, auc, s=100, c=color, label=label, edgecolor="black")

        ax.set_xlabel("Parameters (M)")
        ax.set_ylabel("AUC-ROC")
        ax.set_title("Performance vs Model Size")
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)

        output_path = output_dir / "head_params_vs_performance.pdf"
        plt.savefig(output_path)
        plt.savefig(output_path.with_suffix(".png"))
        logger.info(f"Saved: {output_path}")
        plt.close()


def plot_layer_ablation(results_dir: Path, output_dir: Path):
    """
    Plot layer ablation results.

    Creates a line chart showing AUC vs DiT layer.
    """
    if not HAS_MATPLOTLIB:
        return

    # Load summary
    summary = load_ablation_summary(results_dir, "layer_ablation")
    if not summary:
        summary = {}
        for layer_dir in results_dir.iterdir():
            if layer_dir.is_dir() and layer_dir.name.startswith("layer_"):
                layer = int(layer_dir.name.split("_")[1])
                results = load_results(layer_dir)
                if results:
                    summary[str(layer)] = results

    if not summary:
        logger.error(f"No results found in {results_dir}")
        return

    # Extract data
    layers = []
    aucs = []
    accs = []

    for layer_str in sorted(summary.keys(), key=lambda x: int(x)):
        if "error" not in summary[layer_str]:
            layers.append(int(layer_str))
            aucs.append(summary[layer_str].get("best_auc", 0))
            fm = summary[layer_str].get("final_metrics", {})
            accs.append(fm.get("accuracy", 0))

    if not layers:
        logger.error("No valid results to plot")
        return

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(
        layers, aucs, "o-", color="#e74c3c", linewidth=2, markersize=10, label="AUC-ROC"
    )
    ax.plot(
        layers,
        accs,
        "s--",
        color="#3498db",
        linewidth=2,
        markersize=8,
        label="Accuracy",
    )

    # Highlight best layer
    best_idx = np.argmax(aucs)
    ax.scatter(
        [layers[best_idx]],
        [aucs[best_idx]],
        s=200,
        c="gold",
        edgecolor="black",
        zorder=10,
        label=f"Best (Layer {layers[best_idx]})",
    )

    # Add value labels
    for i, (layer, auc) in enumerate(zip(layers, aucs)):
        ax.annotate(
            f"{auc:.3f}",
            (layer, auc),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=9,
        )

    ax.set_xlabel("DiT Layer")
    ax.set_ylabel("Score")
    ax.set_title("Physics Discriminator: DiT Layer Comparison")
    ax.set_xticks(layers)
    ax.set_ylim(0.5, max(aucs) * 1.1)
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "layer_ablation_comparison.pdf"
    plt.savefig(output_path)
    plt.savefig(output_path.with_suffix(".png"))
    logger.info(f"Saved: {output_path}")
    plt.close()


def plot_timestep_ablation(results_dir: Path, output_dir: Path):
    """
    Plot timestep ablation results.
    """
    if not HAS_MATPLOTLIB:
        return

    summary = load_ablation_summary(results_dir, "timestep_ablation")
    if not summary:
        summary = {}
        for ts_dir in results_dir.iterdir():
            if ts_dir.is_dir() and ts_dir.name.startswith("t_"):
                results = load_results(ts_dir)
                if results:
                    summary[ts_dir.name[2:]] = results  # Remove 't_' prefix

    if not summary:
        logger.error(f"No results found in {results_dir}")
        return

    # Extract data
    configs = []
    aucs = []

    # Sort: single timesteps first, then combined
    single_ts = []
    combined_ts = []

    for config in summary.keys():
        if "error" not in summary[config]:
            if "_" in config:
                combined_ts.append(config)
            else:
                single_ts.append(config)

    single_ts.sort(key=int)

    for config in single_ts + combined_ts:
        configs.append(config)
        aucs.append(summary[config].get("best_auc", 0))

    if not configs:
        return

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 5))

    x = np.arange(len(configs))
    colors = ["#3498db"] * len(single_ts) + ["#e74c3c"] * len(combined_ts)

    bars = ax.bar(x, aucs, color=colors, edgecolor="black", linewidth=0.5)

    # Labels
    labels = []
    for config in configs:
        if "_" in config:
            labels.append("All")
        else:
            labels.append(f"t={config}")

    for bar, auc in zip(bars, aucs):
        height = bar.get_height()
        ax.annotate(
            f"{auc:.3f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            fontsize=9,
        )

    ax.set_xlabel("Timestep Configuration")
    ax.set_ylabel("AUC-ROC")
    ax.set_title("Physics Discriminator: Timestep Configuration Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.5, max(aucs) * 1.1)
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)

    # Legend
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="#3498db", label="Single timestep"),
        Patch(facecolor="#e74c3c", label="Combined"),
    ]
    ax.legend(handles=legend_elements, loc="lower right")

    plt.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "timestep_ablation_comparison.pdf"
    plt.savefig(output_path)
    plt.savefig(output_path.with_suffix(".png"))
    logger.info(f"Saved: {output_path}")
    plt.close()


def plot_training_curves(results_dir: Path, output_dir: Path, name: str = None):
    """
    Plot training curves (loss, AUC vs epoch).
    """
    if not HAS_MATPLOTLIB:
        return

    results = load_results(results_dir)
    if not results:
        logger.error(f"No results.json found in {results_dir}")
        return

    history = results.get("history", {})
    if not history:
        logger.error("No training history found")
        return

    name = name or results.get("head_type", results_dir.name)

    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    epochs = range(1, len(history.get("train_loss", [])) + 1)

    # === Plot 1: Loss ===
    ax1 = axes[0]
    if "train_loss" in history:
        ax1.plot(epochs, history["train_loss"], label="Train", color="#3498db")
    if "val_loss" in history:
        ax1.plot(epochs, history["val_loss"], label="Val", color="#e74c3c")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title(f"{name}: Loss Curves")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # === Plot 2: AUC ===
    ax2 = axes[1]
    if "val_auc" in history:
        ax2.plot(epochs, history["val_auc"], color="#2ecc71", linewidth=2)

        # Mark best
        best_epoch = results.get("best_epoch", np.argmax(history["val_auc"]) + 1)
        best_auc = results.get("best_auc", max(history["val_auc"]))
        ax2.scatter(
            [best_epoch], [best_auc], s=100, c="gold", edgecolor="black", zorder=10
        )
        ax2.annotate(
            f"Best: {best_auc:.3f}",
            (best_epoch, best_auc),
            textcoords="offset points",
            xytext=(10, -10),
            fontsize=9,
        )

    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("AUC-ROC")
    ax2.set_title(f"{name}: Validation AUC")
    ax2.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)
    ax2.grid(True, alpha=0.3)

    # === Plot 3: Learning Rate ===
    ax3 = axes[2]
    if "lr" in history:
        ax3.plot(epochs, history["lr"], color="#9b59b6", linewidth=2)
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Learning Rate")
    ax3.set_title(f"{name}: Learning Rate Schedule")
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"training_curves_{name}.pdf"
    plt.savefig(output_path)
    plt.savefig(output_path.with_suffix(".png"))
    logger.info(f"Saved: {output_path}")
    plt.close()


def plot_per_timestep_analysis(results_dir: Path, output_dir: Path):
    """
    Plot per-timestep AUC analysis across different heads.

    Creates a grouped bar chart showing how each head performs at different noise levels.
    """
    if not HAS_MATPLOTLIB:
        return

    # Collect per-timestep data from all heads
    data = {}  # {head: {timestep: auc}}

    for head_dir in results_dir.iterdir():
        if head_dir.is_dir():
            results = load_results(head_dir)
            if results and "final_metrics" in results:
                auc_per_ts = results["final_metrics"].get("auc_per_timestep", {})
                if auc_per_ts:
                    data[head_dir.name] = {int(k): v for k, v in auc_per_ts.items()}

    if not data:
        logger.warning("No per-timestep data found")
        return

    # Get all timesteps
    all_timesteps = sorted(set(t for aucs in data.values() for t in aucs.keys()))

    if not all_timesteps:
        return

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    n_heads = len(data)
    n_timesteps = len(all_timesteps)
    width = 0.8 / n_heads

    head_order = [
        "mean",
        "mean_adaln",
        "temporal_adaln",
        "causal_adaln",
        "multiview_adaln",
    ]
    heads = [h for h in head_order if h in data]

    for i, head in enumerate(heads):
        aucs = [data[head].get(t, 0) for t in all_timesteps]
        x = np.arange(n_timesteps) + i * width
        color = HEAD_COLORS.get(head, "#333333")
        label = HEAD_NAMES.get(head, head)
        ax.bar(
            x, aucs, width, label=label, color=color, edgecolor="black", linewidth=0.5
        )

    ax.set_xlabel("Timestep (Noise Level)")
    ax.set_ylabel("AUC-ROC")
    ax.set_title("Per-Timestep Performance Across Head Architectures")
    ax.set_xticks(np.arange(n_timesteps) + width * (n_heads - 1) / 2)
    ax.set_xticklabels([f"t={t}" for t in all_timesteps])
    ax.set_ylim(0.5, 1.0)
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3, axis="y")

    # Add annotation
    ax.text(
        0.02,
        0.98,
        "Lower timestep = Less noise = Clearer features",
        transform=ax.transAxes,
        fontsize=8,
        verticalalignment="top",
        style="italic",
        color="gray",
    )

    plt.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "per_timestep_analysis.pdf"
    plt.savefig(output_path)
    plt.savefig(output_path.with_suffix(".png"))
    logger.info(f"Saved: {output_path}")
    plt.close()


def plot_all(results_dir: Path, output_dir: Path):
    """Auto-detect and plot all available results."""
    results_dir = Path(results_dir)

    # Check for different ablation types
    if (results_dir / "head_ablation").exists():
        logger.info("Plotting head ablation...")
        plot_head_ablation(results_dir / "head_ablation", output_dir)
        plot_per_timestep_analysis(results_dir / "head_ablation", output_dir)

        # Training curves for each head
        for head_dir in (results_dir / "head_ablation").iterdir():
            if head_dir.is_dir() and (head_dir / "results.json").exists():
                plot_training_curves(head_dir, output_dir, head_dir.name)

    if (results_dir / "layer_ablation").exists():
        logger.info("Plotting layer ablation...")
        plot_layer_ablation(results_dir / "layer_ablation", output_dir)

    if (results_dir / "timestep_ablation").exists():
        logger.info("Plotting timestep ablation...")
        plot_timestep_ablation(results_dir / "timestep_ablation", output_dir)

    # Check if results_dir itself contains ablation results
    if (results_dir / "head_ablation_summary.json").exists():
        plot_head_ablation(results_dir, output_dir)

    if (results_dir / "layer_ablation_summary.json").exists():
        plot_layer_ablation(results_dir, output_dir)


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Visualize physics head ablation results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Plot head ablation
    python -m trainer.plot_results --results_dir results/head_ablation --plot_type head

    # Plot all available results
    python -m trainer.plot_results --results_dir results --plot_type all
    
    # Plot training curves for a specific run
    python -m trainer.plot_results --results_dir results/head_ablation/causal_adaln --plot_type curves
        """,
    )

    parser.add_argument(
        "--results_dir", type=str, required=True, help="Directory containing results"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for figures (default: results_dir/figures)",
    )
    parser.add_argument(
        "--plot_type",
        type=str,
        default="all",
        choices=["head", "layer", "timestep", "curves", "per_timestep", "all"],
        help="Type of plot to generate",
    )

    args = parser.parse_args()

    if not HAS_MATPLOTLIB:
        logger.error("matplotlib is required. Install with: pip install matplotlib")
        return

    setup_style()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir) if args.output_dir else results_dir / "figures"

    logger.info(f"Results directory: {results_dir}")
    logger.info(f"Output directory: {output_dir}")

    if args.plot_type == "head":
        plot_head_ablation(results_dir, output_dir)
        plot_per_timestep_analysis(results_dir, output_dir)

    elif args.plot_type == "layer":
        plot_layer_ablation(results_dir, output_dir)

    elif args.plot_type == "timestep":
        plot_timestep_ablation(results_dir, output_dir)

    elif args.plot_type == "curves":
        plot_training_curves(results_dir, output_dir)

    elif args.plot_type == "per_timestep":
        plot_per_timestep_analysis(results_dir, output_dir)

    elif args.plot_type == "all":
        plot_all(results_dir, output_dir)

    logger.info("Done!")


if __name__ == "__main__":
    main()
