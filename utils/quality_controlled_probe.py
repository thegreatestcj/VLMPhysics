#!/usr/bin/env python3
"""
Quality-Controlled Physics Probing

Tests whether physics signal in DiT features persists after
controlling for visual quality (measured by VQAScore).

Method:
  1. Load DiT features and VQAScore quality scores
  2. Fit linear regression: features → quality_score (find quality direction)
  3. Project out quality direction from features
  4. Re-probe: logistic regression on residual features → physics labels
  5. Compare AUC before vs after quality removal

If residual AUC ≈ original AUC → physics signal is independent of quality
If residual AUC ≈ 0.5 → probe was only detecting quality, not physics

Usage:
    python utils/quality_controlled_probe.py \
        --feature-dir ~/scratch/physics/videophy_cogx_features \
        --metadata ~/scratch/physics/videophy_cogx/metadata.json \
        --vqascores ~/scratch/physics/videophy_cogx/vqascores.json \
        --output figures/quality_control \
        --layer 10 --timestep 200

    # Test all timesteps
    python utils/quality_controlled_probe.py \
        --feature-dir ~/scratch/physics/videophy_cogx_features \
        --metadata ~/scratch/physics/videophy_cogx/metadata.json \
        --vqascores ~/scratch/physics/videophy_cogx/vqascores.json \
        --output figures/quality_control \
        --layer 10 --timesteps 200 400 600
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


# =============================================================================
# Data Loading
# =============================================================================


def load_features_with_quality(
    feature_dir: str,
    metadata_file: str,
    vqascores_file: str,
    layer: int,
    timestep: int,
    is_pooled: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Load DiT features matched with VQAScore quality scores and physics labels.

    Returns:
        features: [N, D] feature matrix
        labels: [N] physics labels (0/1)
        quality: [N] VQAScore quality scores
        video_ids: [N] video filenames
    """
    feature_dir = Path(feature_dir)

    # Load metadata
    with open(metadata_file) as f:
        meta_list = json.load(f)
    meta_lookup = {}
    for m in meta_list:
        vf = m.get("video_filename", "")
        if vf:
            meta_lookup[vf] = m

    # Load VQAScores
    with open(vqascores_file) as f:
        vqascores = json.load(f)

    logger.info(f"Metadata: {len(meta_lookup)} entries")
    logger.info(f"VQAScores: {len(vqascores)} entries")

    features_list = []
    labels_list = []
    quality_list = []
    video_ids = []

    for video_dir in sorted(feature_dir.iterdir()):
        if not video_dir.is_dir():
            continue

        video_id = video_dir.name

        # Need all three: features, metadata (label), and VQAScore
        meta = meta_lookup.get(video_id)
        vqa = vqascores.get(video_id)

        if meta is None or vqa is None:
            continue

        label = meta.get("physics")
        if label is None or label == -1:
            continue

        score = vqa.get("score_max") or vqa.get("score_mean")
        if score is None:
            continue

        # Load features
        feat_path = feature_dir / video_id / f"t{timestep}" / f"layer_{layer}.pt"
        if not feat_path.exists():
            continue

        feat = torch.load(feat_path, map_location="cpu", weights_only=True)
        feat = feat.float().numpy()

        # Pool if needed: flatten to 1D
        feat = feat.reshape(-1)

        features_list.append(feat)
        labels_list.append(int(label))
        quality_list.append(float(score))
        video_ids.append(video_id)

    if not features_list:
        logger.error("No matched samples found!")
        return np.array([]), np.array([]), np.array([]), []

    features = np.stack(features_list)
    labels = np.array(labels_list)
    quality = np.array(quality_list)

    logger.info(
        f"Loaded {len(features)} matched samples "
        f"(layer={layer}, t={timestep}, dim={features.shape[1]})"
    )
    logger.info(f"  Physics: {(labels == 1).sum()} pos / {(labels == 0).sum()} neg")
    logger.info(
        f"  Quality: mean={quality.mean():.4f}, std={quality.std():.4f}, "
        f"range=[{quality.min():.4f}, {quality.max():.4f}]"
    )

    return features, labels, quality, video_ids


# =============================================================================
# Quality Direction Removal
# =============================================================================


def remove_quality_direction(
    features: np.ndarray,
    quality: np.ndarray,
    method: str = "projection",
) -> np.ndarray:
    """
    Remove quality-correlated direction from feature space.

    Methods:
        'projection': Find direction most correlated with quality via
                      linear regression, then project it out.
        'residual':   Regress each feature dimension on quality score,
                      keep residuals.

    Args:
        features: [N, D] feature matrix
        quality: [N] quality scores
        method: 'projection' or 'residual'

    Returns:
        residual_features: [N, D] with quality direction removed
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    N, D = features.shape

    # Standardize features first
    scaler = StandardScaler()
    X = scaler.fit_transform(features)

    # PCA to reduce dimensionality (avoids overfitting in regression)
    n_components = min(128, D, N - 1)
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)

    if method == "projection":
        # Find quality direction via linear regression in PCA space
        # quality = X_pca @ w + b
        # Then project out w from X_pca
        from sklearn.linear_model import Ridge

        q_centered = quality - quality.mean()
        q_std = q_centered.std()
        if q_std < 1e-8:
            logger.warning("Quality scores have near-zero variance, skipping removal")
            return X_pca

        q_norm = q_centered / q_std

        reg = Ridge(alpha=1.0)
        reg.fit(X_pca, q_norm)
        w = reg.coef_  # [n_components]

        # Explained variance of quality by features
        r2 = reg.score(X_pca, q_norm)
        logger.info(f"  Quality R² from features: {r2:.4f}")

        # Normalize direction
        w_norm = w / (np.linalg.norm(w) + 1e-8)  # [n_components]

        # Project out: X_residual = X - (X @ w_hat) * w_hat^T
        projections = X_pca @ w_norm  # [N]
        X_residual = X_pca - np.outer(projections, w_norm)  # [N, n_components]

        # Verify: correlation with quality should be near zero
        reg_check = Ridge(alpha=1.0)
        reg_check.fit(X_residual, q_norm)
        r2_after = reg_check.score(X_residual, q_norm)
        logger.info(f"  Quality R² after removal: {r2_after:.4f}")

        return X_residual

    elif method == "residual":
        # Per-dimension residualization
        from sklearn.linear_model import LinearRegression

        q_2d = quality.reshape(-1, 1)
        X_residual = np.zeros_like(X_pca)

        for d in range(X_pca.shape[1]):
            reg = LinearRegression()
            reg.fit(q_2d, X_pca[:, d])
            X_residual[:, d] = X_pca[:, d] - reg.predict(q_2d)

        return X_residual

    else:
        raise ValueError(f"Unknown method: {method}")


# =============================================================================
# Probing
# =============================================================================


def probe_auc(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    seed: int = 42,
) -> Tuple[float, float]:
    """5-fold CV logistic regression AUC."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler

    n_pos = (y == 1).sum()
    n_neg = (y == 0).sum()
    effective_splits = min(n_splits, n_pos, n_neg)
    if effective_splits < 2:
        return float("nan"), float("nan")

    aucs = []
    skf = StratifiedKFold(n_splits=effective_splits, shuffle=True, random_state=seed)

    for train_idx, test_idx in skf.split(X, y):
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(X[train_idx])
        Xte = scaler.transform(X[test_idx])

        clf = LogisticRegression(max_iter=1000, C=1.0, random_state=seed)
        clf.fit(Xtr, y[train_idx])
        proba = clf.predict_proba(Xte)[:, 1]

        try:
            aucs.append(roc_auc_score(y[test_idx], proba))
        except ValueError:
            continue

    if not aucs:
        return float("nan"), float("nan")

    return float(np.mean(aucs)), float(np.std(aucs))


def quality_auc(
    quality: np.ndarray,
    labels: np.ndarray,
) -> float:
    """AUC of quality score alone for predicting physics label."""
    from sklearn.metrics import roc_auc_score

    try:
        return float(roc_auc_score(labels, quality))
    except ValueError:
        return float("nan")


# =============================================================================
# Visualization
# =============================================================================


def plot_results(
    results: Dict,
    output_path: str,
):
    """Plot quality-controlled probing results."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # --- Panel 1: Quality score distribution by physics label ---
    ax = axes[0]
    quality_by_label = results.get("quality_distribution", {})
    if quality_by_label:
        pos_scores = quality_by_label.get("physics_1", [])
        neg_scores = quality_by_label.get("physics_0", [])
        ax.hist(neg_scores, bins=20, alpha=0.6, label="physics=0", color="#e74c3c")
        ax.hist(pos_scores, bins=20, alpha=0.6, label="physics=1", color="#2ecc71")
        ax.set_xlabel("VQAScore (quality)")
        ax.set_ylabel("Count")
        ax.set_title("Quality Score Distribution")
        ax.legend()

    # --- Panel 2: AUC comparison bar chart ---
    ax = axes[1]
    comparisons = results.get("comparisons", {})
    if comparisons:
        labels_bar = []
        aucs = []
        stds = []
        colors = []

        label_map = {
            "quality_only": ("Quality\nonly", "#f39c12"),
            "original": ("Original\nfeatures", "#3498db"),
            "residual_projection": ("After quality\nremoval", "#2ecc71"),
        }

        for key in ["quality_only", "original", "residual_projection"]:
            if key in comparisons:
                name, color = label_map[key]
                labels_bar.append(name)
                aucs.append(comparisons[key]["auc_mean"])
                stds.append(comparisons[key].get("auc_std", 0))
                colors.append(color)

        x = np.arange(len(labels_bar))
        bars = ax.bar(x, aucs, yerr=stds, capsize=5, color=colors, alpha=0.8, width=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(labels_bar, fontsize=9)
        ax.set_ylabel("AUC")
        ax.set_title("Physics Probing: Quality Control")
        ax.set_ylim(0.4, max(0.8, max(aucs) + 0.1))
        ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="chance")

        # Add value labels
        for bar, auc in zip(bars, aucs):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{auc:.3f}",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )

    # --- Panel 3: Per-timestep comparison (if available) ---
    ax = axes[2]
    per_timestep = results.get("per_timestep", {})
    if per_timestep and len(per_timestep) > 1:
        timesteps = sorted(int(t) for t in per_timestep.keys())
        orig_aucs = [per_timestep[str(t)]["original"]["auc_mean"] for t in timesteps]
        resid_aucs = [per_timestep[str(t)]["residual"]["auc_mean"] for t in timesteps]
        quality_aucs = [
            per_timestep[str(t)].get("quality_only_auc", 0.5) for t in timesteps
        ]

        ax.plot(
            timesteps,
            orig_aucs,
            "o-",
            label="Original features",
            color="#3498db",
            linewidth=2,
        )
        ax.plot(
            timesteps,
            resid_aucs,
            "s-",
            label="After quality removal",
            color="#2ecc71",
            linewidth=2,
        )
        ax.plot(
            timesteps,
            quality_aucs,
            "^--",
            label="Quality score only",
            color="#f39c12",
            linewidth=1.5,
        )
        ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)
        ax.set_xlabel("Timestep")
        ax.set_ylabel("AUC")
        ax.set_title("Quality Control Across Timesteps")
        ax.legend(fontsize=8)
        ax.set_ylim(0.4, 0.85)
    else:
        # Single timestep: show quality R² info
        r2_before = results.get("quality_r2_before", 0)
        r2_after = results.get("quality_r2_after", 0)
        ax.bar(
            ["Before\nremoval", "After\nremoval"],
            [r2_before, r2_after],
            color=["#e74c3c", "#2ecc71"],
            alpha=0.8,
            width=0.5,
        )
        ax.set_ylabel("R² (quality predictability)")
        ax.set_title("Quality Direction Removal")
        ax.set_ylim(0, max(0.3, r2_before + 0.05))

    plt.suptitle("Quality-Controlled Physics Probing", fontsize=14, fontweight="bold")
    plt.tight_layout()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    logger.info(f"Saved figure to {output_path}")
    plt.close()


# =============================================================================
# Main
# =============================================================================


def run_analysis(
    feature_dir: str,
    metadata_file: str,
    vqascores_file: str,
    output_prefix: str,
    layer: int = 10,
    timesteps: Optional[List[int]] = None,
    is_pooled: bool = True,
):
    """Run quality-controlled probing analysis."""

    if timesteps is None:
        timesteps = [200]

    all_results = {
        "config": {
            "feature_dir": feature_dir,
            "metadata_file": metadata_file,
            "vqascores_file": vqascores_file,
            "layer": layer,
            "timesteps": timesteps,
        },
        "per_timestep": {},
    }

    # Run for each timestep
    for t in timesteps:
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Timestep t={t}, Layer {layer}")
        logger.info(f"{'=' * 60}")

        features, labels, quality, video_ids = load_features_with_quality(
            feature_dir=feature_dir,
            metadata_file=metadata_file,
            vqascores_file=vqascores_file,
            layer=layer,
            timestep=t,
            is_pooled=is_pooled,
        )

        if len(features) == 0:
            logger.warning(f"No data for t={t}, skipping")
            continue

        # --- Baseline: quality score alone ---
        q_auc = quality_auc(quality, labels)
        logger.info(f"  Quality-only AUC: {q_auc:.4f}")

        # --- Original features probe ---
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA

        n_components = min(128, features.shape[1], features.shape[0] - 1)
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(StandardScaler().fit_transform(features))

        orig_auc, orig_std = probe_auc(X_pca, labels)
        logger.info(f"  Original features AUC: {orig_auc:.4f} ± {orig_std:.4f}")

        # --- Quality-controlled probe ---
        logger.info(f"  Removing quality direction (projection method)...")
        X_residual = remove_quality_direction(features, quality, method="projection")

        resid_auc, resid_std = probe_auc(X_residual, labels)
        logger.info(f"  Residual features AUC: {resid_auc:.4f} ± {resid_std:.4f}")

        # --- Summary ---
        delta = orig_auc - resid_auc
        logger.info(f"\n  Summary for t={t}:")
        logger.info(f"    Quality-only AUC:  {q_auc:.4f}")
        logger.info(f"    Original AUC:      {orig_auc:.4f} ± {orig_std:.4f}")
        logger.info(f"    After removal AUC: {resid_auc:.4f} ± {resid_std:.4f}")
        logger.info(f"    AUC drop:          {delta:+.4f}")

        if resid_auc > 0.55:
            logger.info(f"    → Physics signal PERSISTS after quality control ✓")
        elif resid_auc > 0.52:
            logger.info(f"    → Weak residual physics signal (borderline)")
        else:
            logger.info(
                f"    → Physics signal DISAPPEARS — probe was detecting quality ✗"
            )

        all_results["per_timestep"][str(t)] = {
            "original": {"auc_mean": orig_auc, "auc_std": orig_std},
            "residual": {"auc_mean": resid_auc, "auc_std": resid_std},
            "quality_only_auc": q_auc,
            "n_samples": len(features),
            "auc_drop": delta,
        }

    # Use first timestep for main comparison plot
    first_t = str(timesteps[0])
    if first_t in all_results["per_timestep"]:
        ts_data = all_results["per_timestep"][first_t]
        all_results["comparisons"] = {
            "quality_only": {"auc_mean": ts_data["quality_only_auc"], "auc_std": 0},
            "original": ts_data["original"],
            "residual_projection": ts_data["residual"],
        }

    # Quality distribution for plotting
    # Re-load for first timestep
    features, labels, quality, _ = load_features_with_quality(
        feature_dir, metadata_file, vqascores_file, layer, timesteps[0], is_pooled
    )
    if len(features) > 0:
        all_results["quality_distribution"] = {
            "physics_1": quality[labels == 1].tolist(),
            "physics_0": quality[labels == 0].tolist(),
        }

    # Save results
    results_path = f"{output_prefix}_results.json"
    Path(results_path).parent.mkdir(parents=True, exist_ok=True)
    # Remove large arrays before saving
    save_results = {k: v for k, v in all_results.items() if k != "quality_distribution"}
    with open(results_path, "w") as f:
        json.dump(save_results, f, indent=2)
    logger.info(f"\nSaved results to {results_path}")

    # Plot
    plot_results(all_results, f"{output_prefix}_quality_control.png")


def main():
    parser = argparse.ArgumentParser(description="Quality-controlled physics probing")
    parser.add_argument(
        "--feature-dir",
        type=str,
        default="/users/ctang33/scratch/physics/videophy_cogx_features",
        help="Path to extracted DiT features",
    )
    parser.add_argument(
        "--metadata",
        type=str,
        default="/users/ctang33/scratch/physics/videophy_cogx/metadata.json",
        help="Path to metadata.json",
    )
    parser.add_argument(
        "--vqascores",
        type=str,
        default="/users/ctang33/scratch/physics/videophy_cogx/vqascores.json",
        help="Path to VQAScore results",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="figures/quality_control/qc",
        help="Output prefix for figures and results",
    )
    parser.add_argument("--layer", type=int, default=10)
    parser.add_argument(
        "--timesteps",
        type=int,
        nargs="+",
        default=[200, 400, 600],
        help="Timesteps to analyze",
    )
    parser.add_argument(
        "--is-pooled",
        action="store_true",
        default=True,
        help="Whether features are pre-pooled",
    )
    args = parser.parse_args()

    run_analysis(
        feature_dir=args.feature_dir,
        metadata_file=args.metadata,
        vqascores_file=args.vqascores,
        output_prefix=args.output,
        layer=args.layer,
        timesteps=args.timesteps,
        is_pooled=args.is_pooled,
    )


if __name__ == "__main__":
    main()
