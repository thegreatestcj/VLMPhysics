#!/usr/bin/env python3
"""
Cross-source generalization: DiT features vs VAE latent.

Key question: Does DiT improve cross-source transfer compared to VAE?
If yes → could use 5B data to augment 2B training.
If no → must train separate heads per model.

Usage:
    python utils/cross_source_dit.py \
        --vae-latents ~/scratch/physics/vae_latents_cogx_2b5b.npz \
        --dit-feature-dir ~/scratch/physics/videophy_cogx_features \
        --metadata ~/scratch/physics/videophy_cogx/metadata.json \
        --dit-layer 10 --dit-timestep 600 \
        --output figures/cross_source_dit_vs_vae.png
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


def cross_source_generalization(
    X: np.ndarray,
    y: np.ndarray,
    sources: np.ndarray,
    max_pca_dim: int = 128,
    seed: int = 42,
) -> Dict[str, Dict[str, float]]:
    """Train on source A, test on source B. Returns matrix[train][test] = AUC."""
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler

    unique_sources = sorted(set(sources))
    matrix = {}

    for train_src in unique_sources:
        matrix[train_src] = {}
        train_mask = sources == train_src
        Xtr, ytr = X[train_mask], y[train_mask]

        if (ytr == 1).sum() < 3 or (ytr == 0).sum() < 3:
            for test_src in unique_sources:
                matrix[train_src][test_src] = float("nan")
            continue

        n_components = min(max_pca_dim, Xtr.shape[1], len(Xtr) - 1)
        pca = PCA(n_components=n_components)
        Xtr_pca = pca.fit_transform(Xtr)
        sc = StandardScaler()
        Xtr_sc = sc.fit_transform(Xtr_pca)

        clf = LogisticRegression(max_iter=1000, C=1.0, random_state=seed)
        clf.fit(Xtr_sc, ytr)

        for test_src in unique_sources:
            if train_src == test_src:
                # Use 5-fold CV for diagonal (avoid train=test overfitting)
                n_pos = (ytr == 1).sum()
                n_neg = (ytr == 0).sum()
                n_splits = min(5, n_pos, n_neg)
                if n_splits < 2:
                    matrix[train_src][test_src] = float("nan")
                    continue
                aucs = []
                skf = StratifiedKFold(
                    n_splits=n_splits, shuffle=True, random_state=seed
                )
                for tr_idx, te_idx in skf.split(Xtr, ytr):
                    pca_cv = PCA(n_components=n_components)
                    X_tr_cv = pca_cv.fit_transform(Xtr[tr_idx])
                    X_te_cv = pca_cv.transform(Xtr[te_idx])
                    sc_cv = StandardScaler()
                    X_tr_cv = sc_cv.fit_transform(X_tr_cv)
                    X_te_cv = sc_cv.transform(X_te_cv)
                    clf_cv = LogisticRegression(max_iter=1000, C=1.0, random_state=seed)
                    clf_cv.fit(X_tr_cv, ytr[tr_idx])
                    proba = clf_cv.predict_proba(X_te_cv)[:, 1]
                    aucs.append(roc_auc_score(ytr[te_idx], proba))
                matrix[train_src][test_src] = float(np.mean(aucs))
            else:
                test_mask = sources == test_src
                Xte, yte = X[test_mask], y[test_mask]
                if (yte == 1).sum() < 2 or (yte == 0).sum() < 2:
                    matrix[train_src][test_src] = float("nan")
                    continue
                Xte_pca = pca.transform(Xte)
                Xte_sc = sc.transform(Xte_pca)
                proba = clf.predict_proba(Xte_sc)[:, 1]
                try:
                    matrix[train_src][test_src] = float(roc_auc_score(yte, proba))
                except ValueError:
                    matrix[train_src][test_src] = float("nan")

    return matrix


def load_dit_features(feature_dir, metadata_path, layer, timestep):
    """Load DiT features matched with metadata."""
    with open(metadata_path) as f:
        meta_list = json.load(f)

    metadata = {}
    for item in meta_list:
        vname = item.get("video_filename", "")
        if not vname:
            vname = Path(item.get("video_path", "")).stem
        metadata[vname] = item

    feats, labels, sources, names = [], [], [], []
    feature_dir = Path(feature_dir)

    for video_dir in sorted(feature_dir.iterdir()):
        if not video_dir.is_dir():
            continue
        feat_path = video_dir / f"t{timestep}" / f"layer_{layer}.pt"
        if not feat_path.exists():
            continue

        meta = metadata.get(video_dir.name, {})
        label = meta.get("physics")
        if label is None:
            continue

        feat = torch.load(feat_path, map_location="cpu", weights_only=True)
        if isinstance(feat, dict):
            feat = feat.get("pooled", feat.get("features"))
        feat = feat.float().numpy().reshape(-1)

        feats.append(feat)
        labels.append(int(label))
        sources.append(meta.get("source", "unknown"))
        names.append(video_dir.name)

    return np.stack(feats), np.array(labels), np.array(sources), np.array(names)


def plot_comparison(vae_matrix, dit_matrix, output_path):
    """Side-by-side heatmaps: VAE vs DiT cross-source."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sources = sorted(vae_matrix.keys())
    n = len(sources)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, matrix, title in [
        (axes[0], vae_matrix, "VAE Latent"),
        (axes[1], dit_matrix, "DiT Features"),
    ]:
        mat = np.zeros((n, n))
        for i, tr in enumerate(sources):
            for j, te in enumerate(sources):
                mat[i, j] = matrix[tr].get(te, float("nan"))

        im = ax.imshow(mat, cmap="RdYlGn", vmin=0.35, vmax=0.75, aspect="auto")
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        short_labels = [s.replace("cogvideox-", "") for s in sources]
        ax.set_xticklabels(short_labels, fontsize=11)
        ax.set_yticklabels(short_labels, fontsize=11)
        ax.set_xlabel("Test Source", fontsize=10)
        ax.set_ylabel("Train Source", fontsize=10)
        ax.set_title(title, fontsize=12, fontweight="bold")

        for i in range(n):
            for j in range(n):
                val = mat[i, j]
                text = f"{val:.3f}" if not np.isnan(val) else "N/A"
                color = "white" if val < 0.5 else "black"
                ax.text(j, i, text, ha="center", va="center", fontsize=14, color=color)

        plt.colorbar(im, ax=ax, label="AUC", shrink=0.8)

    # Compute and annotate summary stats
    for matrix, ax in [(vae_matrix, axes[0]), (dit_matrix, axes[1])]:
        diag = [matrix[s][s] for s in sources if not np.isnan(matrix[s][s])]
        off = [
            matrix[a][b]
            for a in sources
            for b in sources
            if a != b and not np.isnan(matrix[a][b])
        ]
        if diag and off:
            gap = np.mean(diag) - np.mean(off)
            ax.set_xlabel(
                f"Test Source\n"
                f"diag={np.mean(diag):.3f}  off={np.mean(off):.3f}  gap={gap:.3f}",
                fontsize=9,
            )

    plt.suptitle(
        "Cross-Source Generalization: VAE Latent vs DiT Features",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    logger.info(f"Saved to {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vae-latents", required=True)
    parser.add_argument("--dit-feature-dir", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--dit-layer", type=int, default=10)
    parser.add_argument("--dit-timestep", type=int, default=600)
    parser.add_argument("--output", default="figures/cross_source_dit_vs_vae.png")
    args = parser.parse_args()

    # --- VAE cross-source ---
    logger.info("Loading VAE latents...")
    data = np.load(args.vae_latents, allow_pickle=True)
    vae_feats = data["features"]
    vae_labels = data["labels"]
    vae_sources = data["sources"]

    logger.info("Computing VAE cross-source matrix...")
    vae_matrix = cross_source_generalization(vae_feats, vae_labels, vae_sources)

    logger.info("VAE cross-source:")
    for tr in sorted(vae_matrix):
        for te in sorted(vae_matrix[tr]):
            logger.info(f"  train={tr} -> test={te}: {vae_matrix[tr][te]:.3f}")

    # --- DiT cross-source ---
    logger.info(f"\nLoading DiT features (L{args.dit_layer}, t={args.dit_timestep})...")
    dit_feats, dit_labels, dit_sources, _ = load_dit_features(
        args.dit_feature_dir, args.metadata, args.dit_layer, args.dit_timestep
    )
    logger.info(f"Loaded {len(dit_feats)} DiT features")

    logger.info("Computing DiT cross-source matrix...")
    dit_matrix = cross_source_generalization(dit_feats, dit_labels, dit_sources)

    logger.info("DiT cross-source:")
    for tr in sorted(dit_matrix):
        for te in sorted(dit_matrix[tr]):
            logger.info(f"  train={tr} -> test={te}: {dit_matrix[tr][te]:.3f}")

    # --- Summary ---
    logger.info("\n" + "=" * 60)
    logger.info("COMPARISON")
    logger.info("=" * 60)
    sources = sorted(vae_matrix.keys())
    for tr in sources:
        for te in sources:
            if tr == te:
                continue
            v = vae_matrix[tr][te]
            d = dit_matrix[tr].get(te, float("nan"))
            delta = d - v if not np.isnan(d) else float("nan")
            logger.info(
                f"  {tr} -> {te}:  VAE={v:.3f}  DiT={d:.3f}  Delta={delta:+.3f}"
            )

    off_vae = [
        vae_matrix[a][b]
        for a in sources
        for b in sources
        if a != b and not np.isnan(vae_matrix[a][b])
    ]
    off_dit = [
        dit_matrix[a][b]
        for a in sources
        for b in sources
        if a != b and not np.isnan(dit_matrix[a].get(b, float("nan")))
    ]

    if off_vae and off_dit:
        logger.info(
            f"\n  Off-diagonal mean:  VAE={np.mean(off_vae):.3f}  DiT={np.mean(off_dit):.3f}"
        )
        if np.mean(off_dit) > np.mean(off_vae) + 0.05:
            logger.info(
                "  => DiT IMPROVES cross-source transfer. Consider using 5B data for 2B training."
            )
        elif np.mean(off_dit) > np.mean(off_vae):
            logger.info(
                "  => DiT slightly improves transfer, but may not be enough to merge sources."
            )
        else:
            logger.info(
                "  => DiT does NOT improve transfer. Must train separate heads per source."
            )

    # --- Plot ---
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plot_comparison(vae_matrix, dit_matrix, str(output_path))

    # --- Save JSON ---
    json_path = output_path.with_suffix(".json")
    results = {
        "config": {"dit_layer": args.dit_layer, "dit_timestep": args.dit_timestep},
        "vae_cross_source": {
            f"{tr}->{te}": vae_matrix[tr][te] for tr in sources for te in sources
        },
        "dit_cross_source": {
            f"{tr}->{te}": dit_matrix[tr].get(te, None)
            for tr in sources
            for te in sources
        },
    }
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved results to {json_path}")


if __name__ == "__main__":
    main()
