#!/usr/bin/env python3
"""
Per-Source Conditional Analysis

Quantifies how much physics signal remains after controlling for source (generation model).

Produces:
  1. Conditional Fisher: Fisher ratio within each source (vs global Fisher)
  2. Per-source AUC: Train+eval physics head within each source
  3. Cross-source AUC: Train on source A, test on source B

Usage:
    python utils/analyze_source_confounding.py \
        --feature-dir /path/to/videophy_features_2 \
        --metadata /path/to/videophy_data/metadata.json \
        --output figures/source_analysis \
        --layer 20 --timestep 200
"""

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple

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
# Data Loading (reuse from visualize_features.py)
# =============================================================================


def _sanitize_for_dirname(text: str) -> str:
    for ch in ["/", "\\", ":", "*", "?", '"', "<", ">", "|"]:
        text = text.replace(ch, "_")
    return text


def load_metadata(metadata_path: str) -> Dict[str, Dict]:
    with open(metadata_path) as f:
        items = json.load(f)

    lookup = {}
    for item in items:
        meta = {
            "physics": item.get("physics", -1),
            "sa": item.get("sa", -1),
            "source": item.get("source", "unknown"),
            "states_of_matter": item.get("states_of_matter", "unknown"),
            "caption": item.get("caption", ""),
        }
        source = item.get("source", "")
        caption = item.get("caption", "")

        key1 = _sanitize_for_dirname(f"{source}_{caption}").replace(" ", "_")
        lookup[key1] = meta
        key2 = _sanitize_for_dirname(f"{source}_{caption}")
        lookup[key2] = meta
        if "video_filename" in item:
            lookup[Path(item["video_filename"]).stem] = meta

    return lookup


def match_video_to_metadata(video_dir_name, metadata_lookup):
    if video_dir_name in metadata_lookup:
        return metadata_lookup[video_dir_name]
    normalized = video_dir_name.replace(" ", "_")
    if normalized in metadata_lookup:
        return metadata_lookup[normalized]
    for prefix_len in [80, 60, 40]:
        if len(video_dir_name) >= prefix_len:
            prefix = video_dir_name[:prefix_len]
            for key, meta in metadata_lookup.items():
                if key[:prefix_len] == prefix:
                    return meta
    return None


def load_single_layer_features(
    feature_dir: Path,
    metadata_lookup: Dict,
    layer: int,
    timestep: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load features for a single (layer, timestep).

    Returns:
        features: [N, 24960]
        physics_labels: [N] int
        sources: [N] str
        categories: [N] str (states_of_matter)
    """
    video_dirs = sorted(
        [
            d.name
            for d in feature_dir.iterdir()
            if d.is_dir() and not d.name.startswith((".", "_"))
        ]
    )

    feats_list = []
    labels_list = []
    sources_list = []
    categories_list = []

    for vdir in video_dirs:
        meta = match_video_to_metadata(vdir, metadata_lookup)
        if meta is None or meta["physics"] < 0:
            continue

        pt_path = feature_dir / vdir / f"t{timestep}" / f"layer_{layer}.pt"
        if not pt_path.exists():
            continue

        feat = torch.load(pt_path, map_location="cpu", weights_only=True)
        feat = feat.float().reshape(-1).numpy()

        feats_list.append(feat)
        labels_list.append(meta["physics"])
        sources_list.append(meta["source"])
        categories_list.append(meta.get("states_of_matter", "unknown"))

    features = np.stack(feats_list)
    physics_labels = np.array(labels_list)
    sources = np.array(sources_list)
    categories = np.array(categories_list)

    logger.info(
        f"Loaded {len(features)} videos, layer={layer}, t={timestep}, "
        f"sources={dict(zip(*np.unique(sources, return_counts=True)))}"
    )

    return features, physics_labels, sources, categories


# =============================================================================
# 1. Conditional Fisher Analysis
# =============================================================================


def compute_fisher(features: np.ndarray, labels: np.ndarray) -> float:
    """Compute mean Fisher discriminant ratio across dimensions."""
    pos = features[labels == 1]
    neg = features[labels == 0]
    if len(pos) < 5 or len(neg) < 5:
        return float("nan")
    mu_pos, mu_neg = pos.mean(0), neg.mean(0)
    var_pos, var_neg = pos.var(0), neg.var(0)
    return float(((mu_pos - mu_neg) ** 2 / (var_pos + var_neg + 1e-8)).mean())


def conditional_fisher_analysis(
    features: np.ndarray,
    labels: np.ndarray,
    sources: np.ndarray,
) -> Dict:
    """Compute global Fisher vs per-source conditional Fisher."""
    global_fisher = compute_fisher(features, labels)

    per_source = {}
    for src in sorted(set(sources)):
        mask = sources == src
        n_total = mask.sum()
        n_pos = (labels[mask] == 1).sum()
        n_neg = n_total - n_pos

        fisher = compute_fisher(features[mask], labels[mask])

        per_source[src] = {
            "fisher": fisher,
            "n_total": int(n_total),
            "n_pos": int(n_pos),
            "n_neg": int(n_neg),
            "pos_rate": float(n_pos / n_total) if n_total > 0 else 0,
        }

    # Weighted average of per-source Fisher
    total_n = len(labels)
    conditional_fisher = sum(
        per_source[src]["fisher"] * per_source[src]["n_total"] / total_n
        for src in per_source
        if not np.isnan(per_source[src]["fisher"])
    )

    return {
        "global_fisher": global_fisher,
        "conditional_fisher": conditional_fisher,
        "confounding_ratio": (global_fisher - conditional_fisher) / global_fisher
        if global_fisher > 0
        else 0,
        "per_source": per_source,
    }


# =============================================================================
# 2. Per-Source AUC (Logistic Regression)
# =============================================================================


def compute_auc_logreg(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> float:
    """Train logistic regression and compute AUC."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    pca = PCA(n_components=min(128, X_train.shape[1], X_train.shape[0] - 1))
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_pca)
    X_test_scaled = scaler.transform(X_test_pca)

    clf = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    clf.fit(X_train_scaled, y_train)

    y_prob = clf.predict_proba(X_test_scaled)[:, 1]

    try:
        return float(roc_auc_score(y_test, y_prob))
    except ValueError:
        return float("nan")


def compute_auc_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> float:
    """Train 2-layer MLP and compute AUC (nonlinear probe)."""
    from sklearn.neural_network import MLPClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    pca = PCA(n_components=min(128, X_train.shape[1], X_train.shape[0] - 1))
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_pca)
    X_test_scaled = scaler.transform(X_test_pca)

    clf = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.15,
        random_state=42,
    )
    clf.fit(X_train_scaled, y_train)

    y_prob = clf.predict_proba(X_test_scaled)[:, 1]

    try:
        return float(roc_auc_score(y_test, y_prob))
    except ValueError:
        return float("nan")


def per_source_auc_analysis(
    features: np.ndarray,
    labels: np.ndarray,
    sources: np.ndarray,
    n_splits: int = 5,
) -> Dict:
    """Compute AUC within each source using cross-validation.
    Runs both linear (LogReg) and nonlinear (MLP) probes."""
    from sklearn.model_selection import StratifiedKFold

    results = {}

    def _run_cv(src_features, src_labels, n_cv):
        """Run CV for both probes, return (logreg_aucs, mlp_aucs)."""
        logreg_aucs, mlp_aucs = [], []
        skf = StratifiedKFold(n_splits=n_cv, shuffle=True, random_state=42)
        for train_idx, test_idx in skf.split(src_features, src_labels):
            Xtr, ytr = src_features[train_idx], src_labels[train_idx]
            Xte, yte = src_features[test_idx], src_labels[test_idx]
            logreg_aucs.append(compute_auc_logreg(Xtr, ytr, Xte, yte))
            mlp_aucs.append(compute_auc_mlp(Xtr, ytr, Xte, yte))
        return logreg_aucs, mlp_aucs

    # Global AUC (all sources mixed)
    logreg_aucs, mlp_aucs = _run_cv(features, labels, n_splits)
    results["global"] = {
        "auc_mean": float(np.nanmean(logreg_aucs)),
        "auc_std": float(np.nanstd(logreg_aucs)),
        "mlp_auc_mean": float(np.nanmean(mlp_aucs)),
        "mlp_auc_std": float(np.nanstd(mlp_aucs)),
        "n": len(labels),
    }

    # Per-source AUC
    for src in sorted(set(sources)):
        mask = sources == src
        src_features = features[mask]
        src_labels = labels[mask]

        n_pos = (src_labels == 1).sum()
        n_neg = (src_labels == 0).sum()

        if n_pos < 10 or n_neg < 10:
            results[src] = {
                "auc_mean": float("nan"),
                "auc_std": float("nan"),
                "mlp_auc_mean": float("nan"),
                "mlp_auc_std": float("nan"),
                "n": int(mask.sum()),
                "note": "too few samples for one class",
            }
            continue

        n_cv = min(n_splits, n_pos, n_neg)
        if n_cv < 2:
            n_cv = 2

        logreg_aucs, mlp_aucs = _run_cv(src_features, src_labels, n_cv)

        results[src] = {
            "auc_mean": float(np.nanmean(logreg_aucs)),
            "auc_std": float(np.nanstd(logreg_aucs)),
            "mlp_auc_mean": float(np.nanmean(mlp_aucs)),
            "mlp_auc_std": float(np.nanstd(mlp_aucs)),
            "n": int(mask.sum()),
        }

    return results


# =============================================================================
# 3. Cross-Source Generalization
# =============================================================================


def cross_source_analysis(
    features: np.ndarray,
    labels: np.ndarray,
    sources: np.ndarray,
    min_samples: int = 30,
) -> Dict:
    """Train on source A, test on source B. Returns matrix of AUCs."""
    unique_sources = sorted(set(sources))

    valid_sources = []
    for src in unique_sources:
        mask = sources == src
        n_pos = (labels[mask] == 1).sum()
        n_neg = (labels[mask] == 0).sum()
        if n_pos >= 10 and n_neg >= 10 and mask.sum() >= min_samples:
            valid_sources.append(src)

    logger.info(
        f"Cross-source analysis with {len(valid_sources)} sources: {valid_sources}"
    )

    matrix = {}
    for train_src in valid_sources:
        train_mask = sources == train_src
        X_train = features[train_mask]
        y_train = labels[train_mask]

        for test_src in valid_sources:
            test_mask = sources == test_src
            X_test = features[test_mask]
            y_test = labels[test_mask]

            auc = compute_auc_logreg(X_train, y_train, X_test, y_test)
            matrix[(train_src, test_src)] = auc

    return {"valid_sources": valid_sources, "matrix": matrix}


# =============================================================================
# Plotting
# =============================================================================


def plot_fisher_comparison(fisher_results: Dict, output_path: str):
    """Bar chart: global Fisher vs per-source Fisher."""
    per_source = fisher_results["per_source"]
    sources = sorted(per_source.keys())
    fishers = [per_source[s]["fisher"] for s in sources]
    counts = [per_source[s]["n_total"] for s in sources]
    pos_rates = [per_source[s]["pos_rate"] * 100 for s in sources]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), height_ratios=[2, 1])

    # Top: Fisher ratios
    x = range(len(sources) + 1)
    bars = [fisher_results["global_fisher"]] + fishers
    bar_labels = ["GLOBAL\n(mixed)"] + sources
    colors = ["#2c3e50"] + ["#3498db"] * len(sources)

    ax1.bar(x, bars, color=colors, edgecolor="white", linewidth=0.5)
    ax1.axhline(
        y=fisher_results["conditional_fisher"],
        color="#e74c3c",
        linestyle="--",
        linewidth=2,
        label=f"Conditional avg: {fisher_results['conditional_fisher']:.6f}",
    )
    ax1.set_xticks(x)
    ax1.set_xticklabels(bar_labels, rotation=45, ha="right")
    ax1.set_ylabel("Fisher Discriminant Ratio")
    ax1.set_title(
        f"Physics Signal: Global vs Per-Source\n"
        f"Confounding ratio: {fisher_results['confounding_ratio']:.1%} "
        f"of global Fisher is explained by source",
        fontsize=13,
        fontweight="bold",
    )
    ax1.legend(fontsize=10)

    for i, val in enumerate(bars):
        if not np.isnan(val):
            ax1.text(i, val + max(bars) * 0.02, f"{val:.5f}", ha="center", fontsize=8)

    # Bottom: sample counts and positive rates
    ax2_twin = ax2.twinx()
    ax2.bar(range(len(sources)), counts, color="#3498db", alpha=0.6, label="Count")
    ax2_twin.plot(
        range(len(sources)), pos_rates, "o-", color="#e74c3c", label="Pos rate %"
    )
    ax2.set_xticks(range(len(sources)))
    ax2.set_xticklabels(sources, rotation=45, ha="right")
    ax2.set_ylabel("Sample Count")
    ax2_twin.set_ylabel("Physics=1 Rate (%)")
    ax2.set_title("Dataset Composition per Source", fontsize=11)

    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    logger.info(f"Saved Fisher comparison to {output_path}")
    plt.close()


def plot_auc_comparison(auc_results: Dict, output_path: str):
    """Bar chart: global AUC vs per-source AUC, LogReg and MLP side by side."""
    sources = [k for k in sorted(auc_results.keys()) if k != "global"]

    fig, ax = plt.subplots(figsize=(12, 5))

    labels_list = ["GLOBAL\n(mixed)"] + sources
    n_groups = len(labels_list)
    x = np.arange(n_groups)
    width = 0.35

    # LogReg bars
    lr_aucs = [auc_results["global"]["auc_mean"]] + [
        auc_results[s]["auc_mean"] for s in sources
    ]
    lr_stds = [auc_results["global"]["auc_std"]] + [
        auc_results[s]["auc_std"] for s in sources
    ]

    # MLP bars
    mlp_aucs = [auc_results["global"]["mlp_auc_mean"]] + [
        auc_results[s]["mlp_auc_mean"] for s in sources
    ]
    mlp_stds = [auc_results["global"]["mlp_auc_std"]] + [
        auc_results[s]["mlp_auc_std"] for s in sources
    ]

    counts = [auc_results["global"]["n"]] + [auc_results[s]["n"] for s in sources]

    ax.bar(
        x - width / 2,
        lr_aucs,
        width,
        yerr=lr_stds,
        color="#3498db",
        edgecolor="white",
        linewidth=0.5,
        capsize=3,
        label="LogReg (linear)",
    )
    ax.bar(
        x + width / 2,
        mlp_aucs,
        width,
        yerr=mlp_stds,
        color="#e67e22",
        edgecolor="white",
        linewidth=0.5,
        capsize=3,
        label="MLP (nonlinear)",
    )

    ax.axhline(y=0.5, color="#95a5a6", linestyle=":", linewidth=1, label="Random (0.5)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels_list, rotation=45, ha="right")
    ax.set_ylabel("AUC")
    ax.set_ylim(0.35, 0.85)
    ax.set_title(
        "Physics Classification AUC: Linear vs Nonlinear Probe\n"
        "Global (all sources mixed) vs Per-Source (controlling for source)",
        fontsize=13,
        fontweight="bold",
    )
    ax.legend(fontsize=10, loc="upper right")

    # Annotate with AUC values and count
    for i, (lr_val, mlp_val, n) in enumerate(zip(lr_aucs, mlp_aucs, counts)):
        if not np.isnan(lr_val):
            ax.text(
                i - width / 2,
                lr_val + lr_stds[i] + 0.01,
                f"{lr_val:.3f}",
                ha="center",
                fontsize=7,
                color="#2980b9",
            )
        if not np.isnan(mlp_val):
            ax.text(
                i + width / 2,
                mlp_val + mlp_stds[i] + 0.01,
                f"{mlp_val:.3f}",
                ha="center",
                fontsize=7,
                color="#d35400",
            )
        # Count below x-axis
        ax.text(i, 0.37, f"n={n}", ha="center", fontsize=7, color="#7f8c8d")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    logger.info(f"Saved AUC comparison to {output_path}")
    plt.close()


def plot_cross_source_matrix(cross_results: Dict, output_path: str):
    """Heatmap of train-on-A / test-on-B AUC."""
    sources = cross_results["valid_sources"]
    matrix = cross_results["matrix"]
    n = len(sources)

    mat = np.zeros((n, n))
    for i, train_src in enumerate(sources):
        for j, test_src in enumerate(sources):
            mat[i, j] = matrix.get((train_src, test_src), float("nan"))

    fig, ax = plt.subplots(figsize=(8, 7))

    im = ax.imshow(mat, cmap="RdYlGn", vmin=0.35, vmax=0.75, aspect="auto")
    ax.set_xticks(range(n))
    ax.set_xticklabels(sources, rotation=45, ha="right")
    ax.set_yticks(range(n))
    ax.set_yticklabels(sources)
    ax.set_xlabel("Test Source", fontsize=12)
    ax.set_ylabel("Train Source", fontsize=12)
    ax.set_title(
        "Cross-Source Generalization (AUC)\n"
        "Diagonal = within-source, Off-diagonal = cross-source",
        fontsize=13,
        fontweight="bold",
    )

    for i in range(n):
        for j in range(n):
            val = mat[i, j]
            if not np.isnan(val):
                color = "white" if val < 0.5 else "black"
                weight = "bold" if i == j else "normal"
                ax.text(
                    j,
                    i,
                    f"{val:.3f}",
                    ha="center",
                    va="center",
                    fontsize=9,
                    color=color,
                    fontweight=weight,
                )

    plt.colorbar(im, ax=ax, shrink=0.8, label="AUC")

    diag = [mat[i, i] for i in range(n) if not np.isnan(mat[i, i])]
    off_diag = [
        mat[i, j]
        for i in range(n)
        for j in range(n)
        if i != j and not np.isnan(mat[i, j])
    ]
    if diag and off_diag:
        ax.text(
            0.02,
            0.02,
            f"Diagonal mean: {np.mean(diag):.3f}\n"
            f"Off-diag mean: {np.mean(off_diag):.3f}\n"
            f"Gap: {np.mean(diag) - np.mean(off_diag):.3f}",
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="bottom",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    logger.info(f"Saved cross-source matrix to {output_path}")
    plt.close()


# =============================================================================
# 4. Permutation Test (statistical significance of per-source AUC)
# =============================================================================


def permutation_test_single(
    features: np.ndarray,
    labels: np.ndarray,
    n_permutations: int = 1000,
    n_splits: int = 5,
    seed: int = 42,
) -> Dict:
    """
    Run permutation test: shuffle labels N times, compute AUC each time.
    Returns null distribution + p-value for the real AUC.
    """
    from sklearn.model_selection import StratifiedKFold

    rng = np.random.RandomState(seed)

    # Real AUC (5-fold CV)
    real_aucs = []
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for train_idx, test_idx in skf.split(features, labels):
        auc = compute_auc_logreg(
            features[train_idx],
            labels[train_idx],
            features[test_idx],
            labels[test_idx],
        )
        real_aucs.append(auc)
    real_auc = float(np.nanmean(real_aucs))

    # Null distribution: shuffle labels, compute AUC each time
    null_aucs = []
    for i in range(n_permutations):
        shuffled = rng.permutation(labels)
        perm_aucs = []
        skf_perm = StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=seed + i
        )
        try:
            for train_idx, test_idx in skf_perm.split(features, shuffled):
                auc = compute_auc_logreg(
                    features[train_idx],
                    shuffled[train_idx],
                    features[test_idx],
                    shuffled[test_idx],
                )
                perm_aucs.append(auc)
            null_aucs.append(float(np.nanmean(perm_aucs)))
        except ValueError:
            # Stratified split failed (all one class in a fold)
            continue

    null_aucs = np.array(null_aucs)
    # p-value: fraction of null AUCs >= real AUC
    p_value = float(np.mean(null_aucs >= real_auc))

    return {
        "real_auc": real_auc,
        "null_mean": float(np.mean(null_aucs)),
        "null_std": float(np.std(null_aucs)),
        "null_95": float(np.percentile(null_aucs, 95)),
        "null_99": float(np.percentile(null_aucs, 99)),
        "p_value": p_value,
        "n_permutations": len(null_aucs),
        "null_aucs": null_aucs.tolist(),
    }


def permutation_test_per_source(
    features: np.ndarray,
    labels: np.ndarray,
    sources: np.ndarray,
    n_permutations: int = 1000,
    seed: int = 42,
) -> Dict:
    """Run permutation test within each source + globally."""
    results = {}

    # Global
    logger.info("  Permutation test: global (all sources mixed)...")
    results["global"] = permutation_test_single(
        features, labels, n_permutations=n_permutations, seed=seed
    )
    logger.info(
        f"    AUC={results['global']['real_auc']:.3f}, "
        f"null={results['global']['null_mean']:.3f}+/-{results['global']['null_std']:.3f}, "
        f"p={results['global']['p_value']:.4f}"
    )

    # Per source
    for src in sorted(set(sources)):
        mask = sources == src
        src_labels = labels[mask]
        n_pos = (src_labels == 1).sum()
        n_neg = (src_labels == 0).sum()

        if n_pos < 10 or n_neg < 10:
            results[src] = {
                "real_auc": float("nan"),
                "p_value": float("nan"),
                "note": "too few samples",
            }
            continue

        logger.info(f"  Permutation test: {src} (n={mask.sum()})...")
        results[src] = permutation_test_single(
            features[mask],
            src_labels,
            n_permutations=n_permutations,
            seed=seed,
        )
        logger.info(
            f"    AUC={results[src]['real_auc']:.3f}, "
            f"null={results[src]['null_mean']:.3f}+/-{results[src]['null_std']:.3f}, "
            f"p={results[src]['p_value']:.4f}"
        )

    return results


def plot_permutation_test(perm_results: Dict, output_path: str):
    """Plot permutation test results: null distributions + real AUC markers."""
    sources = [k for k in sorted(perm_results.keys()) if k != "global"]
    all_keys = ["global"] + sources
    valid_keys = [
        k
        for k in all_keys
        if not np.isnan(perm_results[k].get("real_auc", float("nan")))
    ]

    n_plots = len(valid_keys)
    cols = min(4, n_plots)
    rows = (n_plots + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.5 * rows), squeeze=False)

    for idx, key in enumerate(valid_keys):
        ax = axes[idx // cols][idx % cols]
        res = perm_results[key]

        null_aucs = res.get("null_aucs", [])
        if not null_aucs:
            ax.set_visible(False)
            continue

        # Null distribution histogram
        ax.hist(
            null_aucs,
            bins=40,
            color="#bdc3c7",
            edgecolor="white",
            linewidth=0.3,
            density=True,
            alpha=0.8,
            label="Null dist.",
        )

        # Real AUC line
        ax.axvline(
            res["real_auc"],
            color="#e74c3c",
            linewidth=2.5,
            label=f"Real AUC={res['real_auc']:.3f}",
        )

        # 95th percentile of null
        ax.axvline(
            res["null_95"],
            color="#f39c12",
            linewidth=1.5,
            linestyle="--",
            label=f"95th pctl={res['null_95']:.3f}",
        )

        # Significance marker
        p = res["p_value"]
        if p < 0.001:
            sig_text = "p < 0.001 ***"
            sig_color = "#27ae60"
        elif p < 0.01:
            sig_text = f"p = {p:.3f} **"
            sig_color = "#27ae60"
        elif p < 0.05:
            sig_text = f"p = {p:.3f} *"
            sig_color = "#f39c12"
        else:
            sig_text = f"p = {p:.3f} (n.s.)"
            sig_color = "#e74c3c"

        title = "GLOBAL (mixed)" if key == "global" else key
        ax.set_title(
            f"{title}\n{sig_text}", fontsize=11, fontweight="bold", color=sig_color
        )
        ax.set_xlabel("AUC", fontsize=9)
        ax.set_ylabel("Density", fontsize=9)
        ax.legend(fontsize=7, loc="upper left")

        # Consistent x-axis range
        ax.set_xlim(0.35, max(0.8, res["real_auc"] + 0.05))

    # Hide unused subplots
    for idx in range(n_plots, rows * cols):
        axes[idx // cols][idx % cols].set_visible(False)

    fig.suptitle(
        "Permutation Test: Is Per-Source Physics Signal Statistically Significant?\n"
        "Red line = real AUC, gray = null distribution (shuffled labels)",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    logger.info(f"Saved permutation test to {output_path}")
    plt.close()


# =============================================================================
# 5. Per-Category Probing (rules out caption confound)
# =============================================================================


def per_category_auc_analysis(
    features: np.ndarray,
    labels: np.ndarray,
    categories: np.ndarray,
    n_splits: int = 5,
    seed: int = 42,
) -> Dict:
    """
    Compute AUC within each states_of_matter category.

    If physics signal only exists in one category, it's likely a caption confound.
    If signal is consistent across categories, more likely genuine physics.
    """
    from sklearn.model_selection import StratifiedKFold

    results = {}

    # Global AUC (for reference)
    global_aucs = []
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for train_idx, test_idx in skf.split(features, labels):
        auc = compute_auc_logreg(
            features[train_idx],
            labels[train_idx],
            features[test_idx],
            labels[test_idx],
        )
        global_aucs.append(auc)
    results["global"] = {
        "auc_mean": float(np.nanmean(global_aucs)),
        "auc_std": float(np.nanstd(global_aucs)),
        "n": len(labels),
        "pos_rate": float((labels == 1).mean()),
    }

    # Per category
    for cat in sorted(set(categories)):
        if cat in ("unknown", "", "nan"):
            continue
        mask = categories == cat
        cat_labels = labels[mask]
        cat_feats = features[mask]
        n_pos = (cat_labels == 1).sum()
        n_neg = (cat_labels == 0).sum()

        if n_pos < 5 or n_neg < 5:
            results[cat] = {
                "auc_mean": float("nan"),
                "auc_std": float("nan"),
                "n": int(mask.sum()),
                "pos_rate": float(n_pos / mask.sum()) if mask.sum() > 0 else 0,
                "note": "too few samples in one class",
            }
            continue

        # Use fewer folds if sample size is small
        effective_splits = min(n_splits, n_pos, n_neg)
        if effective_splits < 2:
            results[cat] = {
                "auc_mean": float("nan"),
                "auc_std": float("nan"),
                "n": int(mask.sum()),
                "pos_rate": float(n_pos / mask.sum()),
                "note": "cannot stratify",
            }
            continue

        cat_aucs = []
        skf_cat = StratifiedKFold(
            n_splits=effective_splits, shuffle=True, random_state=seed
        )
        for train_idx, test_idx in skf_cat.split(cat_feats, cat_labels):
            auc = compute_auc_logreg(
                cat_feats[train_idx],
                cat_labels[train_idx],
                cat_feats[test_idx],
                cat_labels[test_idx],
            )
            cat_aucs.append(auc)

        results[cat] = {
            "auc_mean": float(np.nanmean(cat_aucs)),
            "auc_std": float(np.nanstd(cat_aucs)),
            "n": int(mask.sum()),
            "pos_rate": float(n_pos / mask.sum()),
        }

    return results


def plot_category_auc(cat_results: Dict, output_path: str):
    """Bar chart of AUC per category."""
    keys = [k for k in sorted(cat_results.keys()) if k != "global"]
    valid = [
        k for k in keys if not np.isnan(cat_results[k].get("auc_mean", float("nan")))
    ]

    if not valid:
        logger.warning("No valid categories to plot")
        return

    all_keys = ["global"] + valid
    aucs = [cat_results[k]["auc_mean"] for k in all_keys]
    stds = [cat_results[k]["auc_std"] for k in all_keys]
    ns = [cat_results[k]["n"] for k in all_keys]

    fig, ax = plt.subplots(figsize=(max(8, len(all_keys) * 1.2), 5))

    colors = ["#2c3e50"] + ["#3498db"] * len(valid)
    bars = ax.bar(
        range(len(all_keys)),
        aucs,
        yerr=stds,
        capsize=4,
        color=colors,
        edgecolor="white",
        linewidth=0.5,
        alpha=0.85,
    )

    ax.axhline(0.5, color="#e74c3c", linestyle="--", linewidth=1.5, label="Chance")

    # Annotate with sample counts
    for i, (auc, n) in enumerate(zip(aucs, ns)):
        ax.text(i, auc + stds[i] + 0.01, f"n={n}", ha="center", fontsize=8)

    ax.set_xticks(range(len(all_keys)))
    labels = ["GLOBAL\n(all)"] + [k.replace("_", "\n") for k in valid]
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("AUC (5-fold CV)", fontsize=11)
    ax.set_ylim(0.35, max(0.85, max(aucs) + 0.1))
    ax.set_title(
        "Per-Category Physics Probing\n"
        "Consistent AUC across categories → signal is NOT caption-specific",
        fontsize=12,
        fontweight="bold",
    )
    ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    logger.info(f"Saved category analysis to {output_path}")
    plt.close()


# =============================================================================
# 6. Timestep Progression (physics emergence during denoising)
# =============================================================================


def load_multi_timestep_features(
    feature_dir: Path,
    metadata_lookup: Dict,
    layer: int,
    timesteps: List[int],
) -> Tuple[Dict[int, np.ndarray], np.ndarray, np.ndarray]:
    """
    Load features for one layer across multiple timesteps.
    Only includes videos that have features at ALL timesteps.

    Returns:
        features_by_t: {timestep: [N, D] array}
        labels: [N]
        sources: [N]
    """
    video_dirs = sorted(
        [
            d.name
            for d in feature_dir.iterdir()
            if d.is_dir() and not d.name.startswith((".", "_"))
        ]
    )

    # First pass: find videos that have all timesteps
    valid_videos = []
    for vdir in video_dirs:
        meta = match_video_to_metadata(vdir, metadata_lookup)
        if meta is None or meta["physics"] < 0:
            continue
        if all(
            (feature_dir / vdir / f"t{t}" / f"layer_{layer}.pt").exists()
            for t in timesteps
        ):
            valid_videos.append((vdir, meta))

    logger.info(
        f"Timestep progression: {len(valid_videos)} videos have all timesteps {timesteps}"
    )

    # Second pass: load features
    features_by_t = {t: [] for t in timesteps}
    labels_list = []
    sources_list = []

    for vdir, meta in valid_videos:
        for t in timesteps:
            pt_path = feature_dir / vdir / f"t{t}" / f"layer_{layer}.pt"
            feat = torch.load(pt_path, map_location="cpu", weights_only=True)
            features_by_t[t].append(feat.float().reshape(-1).numpy())
        labels_list.append(meta["physics"])
        sources_list.append(meta["source"])

    for t in timesteps:
        features_by_t[t] = np.stack(features_by_t[t])

    return features_by_t, np.array(labels_list), np.array(sources_list)


def timestep_progression_analysis(
    features_by_t: Dict[int, np.ndarray],
    labels: np.ndarray,
    sources: np.ndarray,
    n_splits: int = 5,
    seed: int = 42,
) -> Dict:
    """
    Compute per-source AUC at each timestep.

    Expected pattern (if physics emerges during denoising):
        t=800: AUC ≈ 0.5 (pure noise, no physics)
        t=600: AUC slightly above 0.5
        t=400: AUC increasing
        t=200: AUC highest (clearest signal)
    """
    from sklearn.model_selection import StratifiedKFold

    timesteps = sorted(features_by_t.keys(), reverse=True)  # high → low noise
    results = {}

    # Global AUC at each timestep
    results["global"] = {}
    for t in timesteps:
        feats = features_by_t[t]
        aucs = []
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for train_idx, test_idx in skf.split(feats, labels):
            auc = compute_auc_logreg(
                feats[train_idx],
                labels[train_idx],
                feats[test_idx],
                labels[test_idx],
            )
            aucs.append(auc)
        results["global"][t] = {
            "auc_mean": float(np.nanmean(aucs)),
            "auc_std": float(np.nanstd(aucs)),
        }
        logger.info(
            f"  Global t={t}: AUC={np.nanmean(aucs):.3f}+/-{np.nanstd(aucs):.3f}"
        )

    # Per-source AUC at each timestep
    for src in sorted(set(sources)):
        mask = sources == src
        src_labels = labels[mask]
        n_pos = (src_labels == 1).sum()
        n_neg = (src_labels == 0).sum()

        if n_pos < 5 or n_neg < 5:
            continue

        effective_splits = min(n_splits, n_pos, n_neg)
        if effective_splits < 2:
            continue

        results[src] = {}
        for t in timesteps:
            src_feats = features_by_t[t][mask]
            aucs = []
            skf = StratifiedKFold(
                n_splits=effective_splits, shuffle=True, random_state=seed
            )
            for train_idx, test_idx in skf.split(src_feats, src_labels):
                auc = compute_auc_logreg(
                    src_feats[train_idx],
                    src_labels[train_idx],
                    src_feats[test_idx],
                    src_labels[test_idx],
                )
                aucs.append(auc)
            results[src][t] = {
                "auc_mean": float(np.nanmean(aucs)),
                "auc_std": float(np.nanstd(aucs)),
            }

        # Log per-source progression
        progression = " → ".join(
            f"t={t}: {results[src][t]['auc_mean']:.3f}" for t in timesteps
        )
        logger.info(f"  {src}: {progression}")

    return results


def plot_timestep_progression(ts_results: Dict, output_path: str):
    """Line chart: AUC vs timestep, one line per source."""
    fig, ax = plt.subplots(figsize=(8, 5))

    # Color palette
    palette = [
        "#e74c3c",
        "#3498db",
        "#2ecc71",
        "#f39c12",
        "#9b59b6",
        "#1abc9c",
        "#e67e22",
        "#34495e",
    ]

    all_keys = ["global"] + sorted(k for k in ts_results if k != "global")

    for i, key in enumerate(all_keys):
        ts_data = ts_results[key]
        timesteps = sorted(ts_data.keys(), reverse=True)
        aucs = [ts_data[t]["auc_mean"] for t in timesteps]
        stds = [ts_data[t]["auc_std"] for t in timesteps]

        color = palette[i % len(palette)]
        linewidth = 3 if key == "global" else 1.5
        alpha = 1.0 if key == "global" else 0.7
        marker = "o" if key == "global" else "s"
        label = "GLOBAL" if key == "global" else key

        ax.errorbar(
            timesteps,
            aucs,
            yerr=stds,
            marker=marker,
            linewidth=linewidth,
            alpha=alpha,
            color=color,
            capsize=3,
            label=label,
            markersize=6,
        )

    ax.axhline(
        0.5, color="gray", linestyle="--", linewidth=1, alpha=0.5, label="Chance"
    )

    ax.set_xlabel("Timestep t (← denoising direction)", fontsize=11)
    ax.set_ylabel("AUC (5-fold CV)", fontsize=11)
    ax.set_title(
        "Physics Signal vs Denoising Progress\n"
        "Monotonic increase → physics emerges during denoising (hard to explain by confound)",
        fontsize=12,
        fontweight="bold",
    )

    # Reverse x-axis: high t (noisy) on left, low t (clean) on right
    ax.invert_xaxis()
    ax.legend(fontsize=8, loc="upper right", ncol=2)
    ax.set_ylim(0.4, max(0.85, ax.get_ylim()[1]))

    # Annotate denoising direction
    ax.annotate(
        "Noisy",
        xy=(0.02, 0.02),
        xycoords="axes fraction",
        fontsize=9,
        color="gray",
        fontstyle="italic",
    )
    ax.annotate(
        "Clean",
        xy=(0.92, 0.02),
        xycoords="axes fraction",
        fontsize=9,
        color="gray",
        fontstyle="italic",
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    logger.info(f"Saved timestep progression to {output_path}")
    plt.close()


# =============================================================================
# 7. Layer Progression (where in the network is physics encoded?)
# =============================================================================


def load_multi_layer_features(
    feature_dir: Path,
    metadata_lookup: Dict,
    layers: List[int],
    timestep: int,
) -> Tuple[Dict[int, np.ndarray], np.ndarray, np.ndarray]:
    """
    Load features for multiple layers at a fixed timestep.
    Only includes videos that have features at ALL layers.

    Returns:
        features_by_layer: {layer: [N, D] array}
        labels: [N]
        sources: [N]
    """
    video_dirs = sorted(
        [
            d.name
            for d in feature_dir.iterdir()
            if d.is_dir() and not d.name.startswith((".", "_"))
        ]
    )

    # First pass: find videos that have all layers at this timestep
    valid_videos = []
    for vdir in video_dirs:
        meta = match_video_to_metadata(vdir, metadata_lookup)
        if meta is None or meta["physics"] < 0:
            continue
        if all(
            (feature_dir / vdir / f"t{timestep}" / f"layer_{l}.pt").exists()
            for l in layers
        ):
            valid_videos.append((vdir, meta))

    logger.info(
        f"Layer progression: {len(valid_videos)} videos have all layers {layers} at t={timestep}"
    )

    # Second pass: load features
    features_by_layer = {l: [] for l in layers}
    labels_list = []
    sources_list = []

    for vdir, meta in valid_videos:
        for l in layers:
            pt_path = feature_dir / vdir / f"t{timestep}" / f"layer_{l}.pt"
            feat = torch.load(pt_path, map_location="cpu", weights_only=True)
            features_by_layer[l].append(feat.float().reshape(-1).numpy())
        labels_list.append(meta["physics"])
        sources_list.append(meta["source"])

    for l in layers:
        features_by_layer[l] = np.stack(features_by_layer[l])

    return features_by_layer, np.array(labels_list), np.array(sources_list)


def layer_progression_analysis(
    features_by_layer: Dict[int, np.ndarray],
    labels: np.ndarray,
    sources: np.ndarray,
    n_splits: int = 5,
    seed: int = 42,
) -> Dict:
    """
    Compute per-source AUC at each layer (fixed timestep).

    Expected patterns:
        - AUC increases with depth → DiT builds physics understanding layer by layer
        - AUC peaks at middle layers → intermediate representations are most informative
        - AUC flat across layers → signal comes from input (VAE latent), not DiT computation
    """
    from sklearn.model_selection import StratifiedKFold

    layers = sorted(features_by_layer.keys())
    results = {}

    # Global AUC at each layer
    results["global"] = {}
    for l in layers:
        feats = features_by_layer[l]
        aucs = []
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for train_idx, test_idx in skf.split(feats, labels):
            auc = compute_auc_logreg(
                feats[train_idx],
                labels[train_idx],
                feats[test_idx],
                labels[test_idx],
            )
            aucs.append(auc)
        results["global"][l] = {
            "auc_mean": float(np.nanmean(aucs)),
            "auc_std": float(np.nanstd(aucs)),
        }
        logger.info(
            f"  Global layer={l}: AUC={np.nanmean(aucs):.3f}+/-{np.nanstd(aucs):.3f}"
        )

    # Per-source AUC at each layer
    for src in sorted(set(sources)):
        mask = sources == src
        src_labels = labels[mask]
        n_pos = (src_labels == 1).sum()
        n_neg = (src_labels == 0).sum()

        if n_pos < 5 or n_neg < 5:
            continue

        effective_splits = min(n_splits, n_pos, n_neg)
        if effective_splits < 2:
            continue

        results[src] = {}
        for l in layers:
            src_feats = features_by_layer[l][mask]
            aucs = []
            skf = StratifiedKFold(
                n_splits=effective_splits, shuffle=True, random_state=seed
            )
            for train_idx, test_idx in skf.split(src_feats, src_labels):
                auc = compute_auc_logreg(
                    src_feats[train_idx],
                    src_labels[train_idx],
                    src_feats[test_idx],
                    src_labels[test_idx],
                )
                aucs.append(auc)
            results[src][l] = {
                "auc_mean": float(np.nanmean(aucs)),
                "auc_std": float(np.nanstd(aucs)),
            }

        # Log per-source progression
        progression = " → ".join(
            f"L{l}: {results[src][l]['auc_mean']:.3f}" for l in layers
        )
        logger.info(f"  {src}: {progression}")

    return results


def plot_layer_progression(layer_results: Dict, output_path: str):
    """Line chart: AUC vs layer depth, one line per source."""
    fig, ax = plt.subplots(figsize=(8, 5))

    palette = [
        "#e74c3c",
        "#3498db",
        "#2ecc71",
        "#f39c12",
        "#9b59b6",
        "#1abc9c",
        "#e67e22",
        "#34495e",
    ]

    all_keys = ["global"] + sorted(k for k in layer_results if k != "global")

    for i, key in enumerate(all_keys):
        layer_data = layer_results[key]
        layers = sorted(layer_data.keys())
        aucs = [layer_data[l]["auc_mean"] for l in layers]
        stds = [layer_data[l]["auc_std"] for l in layers]

        color = palette[i % len(palette)]
        linewidth = 3 if key == "global" else 1.5
        alpha = 1.0 if key == "global" else 0.7
        marker = "o" if key == "global" else "s"
        label = "GLOBAL" if key == "global" else key

        ax.errorbar(
            layers,
            aucs,
            yerr=stds,
            marker=marker,
            linewidth=linewidth,
            alpha=alpha,
            color=color,
            capsize=3,
            label=label,
            markersize=6,
        )

    ax.axhline(
        0.5, color="gray", linestyle="--", linewidth=1, alpha=0.5, label="Chance"
    )

    ax.set_xlabel("DiT Layer (→ deeper)", fontsize=11)
    ax.set_ylabel("AUC (5-fold CV)", fontsize=11)
    ax.set_title(
        "Physics Signal vs Network Depth\n"
        "Where in the DiT is physics information encoded?",
        fontsize=12,
        fontweight="bold",
    )

    ax.legend(fontsize=8, loc="upper right", ncol=2)
    ax.set_ylim(0.4, max(0.85, ax.get_ylim()[1]))

    # Annotate
    ax.annotate(
        "Shallow",
        xy=(0.02, 0.02),
        xycoords="axes fraction",
        fontsize=9,
        color="gray",
        fontstyle="italic",
    )
    ax.annotate(
        "Deep",
        xy=(0.92, 0.02),
        xycoords="axes fraction",
        fontsize=9,
        color="gray",
        fontstyle="italic",
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    logger.info(f"Saved layer progression to {output_path}")
    plt.close()


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Analyze source confounding in DiT physics features",
    )
    parser.add_argument("--feature-dir", type=str, required=True)
    parser.add_argument("--metadata", type=str, required=True)
    parser.add_argument("--output", type=str, default="figures/source_analysis")
    parser.add_argument("--layer", type=int, default=20)
    parser.add_argument("--timestep", type=int, default=200)
    parser.add_argument(
        "--skip-cross", action="store_true", help="Skip cross-source analysis (slower)"
    )
    parser.add_argument(
        "--skip-permutation",
        action="store_true",
        help="Skip permutation test (slowest)",
    )
    parser.add_argument(
        "--skip-category", action="store_true", help="Skip per-category probing"
    )
    parser.add_argument(
        "--skip-timestep",
        action="store_true",
        help="Skip timestep progression analysis",
    )
    parser.add_argument(
        "--skip-layer-prog", action="store_true", help="Skip layer progression analysis"
    )
    parser.add_argument(
        "--n-permutations",
        type=int,
        default=1000,
        help="Number of permutations for significance test",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        nargs="+",
        default=None,
        help="Timesteps for progression analysis (default: auto-detect)",
    )
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=None,
        help="Layers for layer progression analysis (default: auto-detect)",
    )

    args = parser.parse_args()

    output_prefix = args.output.rstrip("/")
    Path(output_prefix).parent.mkdir(parents=True, exist_ok=True)

    # Load data
    metadata_lookup = load_metadata(args.metadata)
    features, labels, sources, categories = load_single_layer_features(
        Path(args.feature_dir), metadata_lookup, args.layer, args.timestep
    )

    # =========================================================================
    # 1. Conditional Fisher
    # =========================================================================
    logger.info("=" * 60)
    logger.info("1. Conditional Fisher Analysis")
    logger.info("=" * 60)

    fisher_results = conditional_fisher_analysis(features, labels, sources)

    logger.info(f"  Global Fisher:      {fisher_results['global_fisher']:.6f}")
    logger.info(f"  Conditional Fisher: {fisher_results['conditional_fisher']:.6f}")
    logger.info(f"  Confounding ratio:  {fisher_results['confounding_ratio']:.1%}")

    for src, info in sorted(fisher_results["per_source"].items()):
        logger.info(
            f"    {src:12s}: Fisher={info['fisher']:.6f}, "
            f"n={info['n_total']:4d} (pos_rate={info['pos_rate']:.1%})"
        )

    plot_fisher_comparison(fisher_results, f"{output_prefix}_fisher.png")

    # =========================================================================
    # 2. Per-Source AUC
    # =========================================================================
    logger.info("=" * 60)
    logger.info("2. Per-Source AUC (Linear + Nonlinear Probes)")
    logger.info("=" * 60)

    auc_results = per_source_auc_analysis(features, labels, sources)

    for key in sorted(auc_results.keys()):
        info = auc_results[key]
        logger.info(
            f"  {key:12s}: LogReg={info['auc_mean']:.3f}+/-{info['auc_std']:.3f}, "
            f"MLP={info['mlp_auc_mean']:.3f}+/-{info['mlp_auc_std']:.3f}, "
            f"n={info['n']}"
        )

    plot_auc_comparison(auc_results, f"{output_prefix}_auc.png")

    # =========================================================================
    # 3. Cross-Source Generalization
    # =========================================================================
    if not args.skip_cross:
        logger.info("=" * 60)
        logger.info("3. Cross-Source Generalization")
        logger.info("=" * 60)

        cross_results = cross_source_analysis(features, labels, sources)
        plot_cross_source_matrix(cross_results, f"{output_prefix}_cross.png")

        sources_list = cross_results["valid_sources"]
        matrix = cross_results["matrix"]
        diag = [matrix[(s, s)] for s in sources_list]
        off = [matrix[(a, b)] for a in sources_list for b in sources_list if a != b]
        logger.info(f"  Diagonal (within-source) mean AUC: {np.nanmean(diag):.3f}")
        logger.info(f"  Off-diagonal (cross-source) mean AUC: {np.nanmean(off):.3f}")
        logger.info(f"  Gap: {np.nanmean(diag) - np.nanmean(off):.3f}")

    # =========================================================================
    # 4. Permutation Test
    # =========================================================================
    perm_results = None
    if not args.skip_permutation:
        logger.info("=" * 60)
        logger.info(f"4. Permutation Test ({args.n_permutations} permutations)")
        logger.info("=" * 60)

        perm_results = permutation_test_per_source(
            features,
            labels,
            sources,
            n_permutations=args.n_permutations,
        )
        plot_permutation_test(perm_results, f"{output_prefix}_permutation.png")

        # Summary table
        logger.info("")
        logger.info(
            f"  {'Source':<12s} {'AUC':>6s} {'Null mean':>10s} {'p-value':>10s} {'Sig':>6s}"
        )
        logger.info(f"  {'-' * 12} {'-' * 6} {'-' * 10} {'-' * 10} {'-' * 6}")
        for key in ["global"] + sorted(k for k in perm_results if k != "global"):
            res = perm_results[key]
            auc = res.get("real_auc", float("nan"))
            if np.isnan(auc):
                continue
            p = res["p_value"]
            sig = (
                "***"
                if p < 0.001
                else "**"
                if p < 0.01
                else "*"
                if p < 0.05
                else "n.s."
            )
            null_m = res.get("null_mean", float("nan"))
            logger.info(
                f"  {key:<12s} {auc:>6.3f} {null_m:>10.3f} {p:>10.4f} {sig:>6s}"
            )

    # =========================================================================
    # 5. Per-Category Probing
    # =========================================================================
    cat_results = None
    if not args.skip_category:
        logger.info("=" * 60)
        logger.info("5. Per-Category Probing (states_of_matter)")
        logger.info("=" * 60)

        unique_cats = sorted(set(categories) - {"unknown", "", "nan"})
        logger.info(f"  Categories found: {unique_cats}")
        cat_counts = {c: (categories == c).sum() for c in unique_cats}
        logger.info(f"  Counts: {cat_counts}")

        if len(unique_cats) >= 2:
            cat_results = per_category_auc_analysis(features, labels, categories)
            plot_category_auc(cat_results, f"{output_prefix}_category.png")

            logger.info("")
            logger.info(f"  {'Category':<20s} {'AUC':>6s} {'n':>6s} {'pos%':>6s}")
            logger.info(f"  {'-' * 20} {'-' * 6} {'-' * 6} {'-' * 6}")
            for key in ["global"] + sorted(k for k in cat_results if k != "global"):
                res = cat_results[key]
                auc = res.get("auc_mean", float("nan"))
                n = res.get("n", 0)
                pr = res.get("pos_rate", 0)
                logger.info(f"  {key:<20s} {auc:>6.3f} {n:>6d} {pr:>5.1%}")
        else:
            logger.info("  Skipped: fewer than 2 valid categories")

    # =========================================================================
    # 6. Timestep Progression
    # =========================================================================
    ts_results = None
    if not args.skip_timestep:
        logger.info("=" * 60)
        logger.info("6. Timestep Progression Analysis")
        logger.info("=" * 60)

        # Auto-detect available timesteps from feature directory
        if args.timesteps:
            ts_list = sorted(args.timesteps)
        else:
            # Scan first video dir for available timesteps
            feature_dir = Path(args.feature_dir)
            sample_dir = None
            for d in feature_dir.iterdir():
                if d.is_dir() and not d.name.startswith((".", "_")):
                    sample_dir = d
                    break
            if sample_dir:
                ts_list = sorted(
                    int(t.name[1:])
                    for t in sample_dir.iterdir()
                    if t.is_dir() and t.name.startswith("t")
                )
                logger.info(f"  Auto-detected timesteps: {ts_list}")
            else:
                ts_list = []

        if len(ts_list) >= 2:
            features_by_t, ts_labels, ts_sources = load_multi_timestep_features(
                Path(args.feature_dir), metadata_lookup, args.layer, ts_list
            )
            ts_results = timestep_progression_analysis(
                features_by_t, ts_labels, ts_sources
            )
            plot_timestep_progression(ts_results, f"{output_prefix}_timestep.png")
        else:
            logger.info(
                f"  Skipped: need >=2 timesteps, found {ts_list}. "
                f"Extract features at multiple timesteps first."
            )

    # =========================================================================
    # 7. Layer Progression
    # =========================================================================
    layer_results = None
    if not args.skip_layer_prog:
        logger.info("=" * 60)
        logger.info("7. Layer Progression Analysis")
        logger.info("=" * 60)

        # Auto-detect available layers from feature directory
        if args.layers:
            layer_list = sorted(args.layers)
        else:
            feature_dir = Path(args.feature_dir)
            sample_dir = None
            for d in feature_dir.iterdir():
                if d.is_dir() and not d.name.startswith((".", "_")):
                    sample_dir = d
                    break
            if sample_dir:
                # Look inside t{timestep} for layer files
                t_dir = sample_dir / f"t{args.timestep}"
                if t_dir.exists():
                    layer_list = sorted(
                        int(f.stem.split("_")[1])
                        for f in t_dir.iterdir()
                        if f.name.startswith("layer_") and f.suffix == ".pt"
                    )
                    logger.info(f"  Auto-detected layers: {layer_list}")
                else:
                    layer_list = []
            else:
                layer_list = []

        if len(layer_list) >= 2:
            features_by_layer, lp_labels, lp_sources = load_multi_layer_features(
                Path(args.feature_dir), metadata_lookup, layer_list, args.timestep
            )
            layer_results = layer_progression_analysis(
                features_by_layer, lp_labels, lp_sources
            )
            plot_layer_progression(layer_results, f"{output_prefix}_layer.png")
        else:
            logger.info(
                f"  Skipped: need >=2 layers, found {layer_list}. "
                f"Extract features at multiple layers first."
            )

    # =========================================================================
    # Save JSON
    # =========================================================================
    all_results = {
        "config": {"layer": args.layer, "timestep": args.timestep},
        "fisher": {
            "global": fisher_results["global_fisher"],
            "conditional": fisher_results["conditional_fisher"],
            "confounding_ratio": fisher_results["confounding_ratio"],
            "per_source": fisher_results["per_source"],
        },
        "auc": auc_results,
    }
    if not args.skip_cross:
        all_results["cross_source"] = {
            "valid_sources": cross_results["valid_sources"],
            "matrix": {f"{a}->{b}": v for (a, b), v in cross_results["matrix"].items()},
        }

    if perm_results is not None:
        # Save summary (exclude raw null_aucs arrays to keep JSON small)
        all_results["permutation_test"] = {
            key: {k: v for k, v in res.items() if k != "null_aucs"}
            for key, res in perm_results.items()
        }

    if cat_results is not None:
        all_results["category_probing"] = cat_results

    if ts_results is not None:
        all_results["timestep_progression"] = ts_results

    if layer_results is not None:
        all_results["layer_progression"] = layer_results

    json_path = f"{output_prefix}_results.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"Saved results to {json_path}")

    logger.info("=" * 60)
    logger.info("Done!")


if __name__ == "__main__":
    main()
