#!/usr/bin/env python3
"""
Combined training experiment: Should we merge 5B data into 2B training?

Experiment design:
  Baseline:  5-fold CV on 2B only (train ~274, test ~69)
  Combined:  5-fold CV, train = 2B_train + ALL 5B, test = 2B_test only

If combined AUC > baseline AUC, merging helps.

Usage:
    python utils/combined_training_test.py \
        --dit-feature-dir ~/scratch/physics/videophy_cogx_features \
        --metadata ~/scratch/physics/videophy_cogx/metadata.json \
        --dit-layer 10 --dit-timestep 600
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


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

    feats, labels, sources = [], [], []
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

    return np.stack(feats), np.array(labels), np.array(sources)


def run_experiment(X, y, sources, max_pca_dim=128, n_splits=5, seed=42):
    """Run 2B-only vs Combined training, both tested on 2B."""
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler

    mask_2b = sources == "cogvideox-2b"
    mask_5b = sources == "cogvideox-5b"

    X_2b, y_2b = X[mask_2b], y[mask_2b]
    X_5b, y_5b = X[mask_5b], y[mask_5b]

    logger.info(
        f"2B: {len(X_2b)} samples ({(y_2b == 1).sum()} pos, {(y_2b == 0).sum()} neg)"
    )
    logger.info(
        f"5B: {len(X_5b)} samples ({(y_5b == 1).sum()} pos, {(y_5b == 0).sum()} neg)"
    )

    n_pos = (y_2b == 1).sum()
    n_neg = (y_2b == 0).sum()
    effective_splits = min(n_splits, n_pos, n_neg)

    skf = StratifiedKFold(n_splits=effective_splits, shuffle=True, random_state=seed)

    aucs_baseline = []
    aucs_combined = []
    aucs_5b_only = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X_2b, y_2b)):
        # Test set: always 2B held-out
        X_test = X_2b[test_idx]
        y_test = y_2b[test_idx]

        # === Baseline: 2B only ===
        X_train_base = X_2b[train_idx]
        y_train_base = y_2b[train_idx]

        n_comp = min(max_pca_dim, X_train_base.shape[1], len(X_train_base) - 1)

        pca = PCA(n_components=n_comp)
        Xtr = pca.fit_transform(X_train_base)
        Xte = pca.transform(X_test)
        sc = StandardScaler()
        Xtr = sc.fit_transform(Xtr)
        Xte = sc.transform(Xte)

        clf = LogisticRegression(max_iter=1000, C=1.0, random_state=seed)
        clf.fit(Xtr, y_train_base)
        proba = clf.predict_proba(Xte)[:, 1]
        aucs_baseline.append(roc_auc_score(y_test, proba))

        # === Combined: 2B train + ALL 5B ===
        X_train_comb = np.vstack([X_2b[train_idx], X_5b])
        y_train_comb = np.concatenate([y_2b[train_idx], y_5b])

        n_comp_c = min(max_pca_dim, X_train_comb.shape[1], len(X_train_comb) - 1)

        pca_c = PCA(n_components=n_comp_c)
        Xtr_c = pca_c.fit_transform(X_train_comb)
        Xte_c = pca_c.transform(X_test)
        sc_c = StandardScaler()
        Xtr_c = sc_c.fit_transform(Xtr_c)
        Xte_c = sc_c.transform(Xte_c)

        clf_c = LogisticRegression(max_iter=1000, C=1.0, random_state=seed)
        clf_c.fit(Xtr_c, y_train_comb)
        proba_c = clf_c.predict_proba(Xte_c)[:, 1]
        aucs_combined.append(roc_auc_score(y_test, proba_c))

        # === 5B only (for reference) ===
        n_comp_5 = min(max_pca_dim, X_5b.shape[1], len(X_5b) - 1)

        pca_5 = PCA(n_components=n_comp_5)
        Xtr_5 = pca_5.fit_transform(X_5b)
        Xte_5 = pca_5.transform(X_test)
        sc_5 = StandardScaler()
        Xtr_5 = sc_5.fit_transform(Xtr_5)
        Xte_5 = sc_5.transform(Xte_5)

        clf_5 = LogisticRegression(max_iter=1000, C=1.0, random_state=seed)
        clf_5.fit(Xtr_5, y_5b)
        proba_5 = clf_5.predict_proba(Xte_5)[:, 1]
        aucs_5b_only.append(roc_auc_score(y_test, proba_5))

        logger.info(
            f"  Fold {fold + 1}: 2B-only={aucs_baseline[-1]:.3f}  "
            f"Combined={aucs_combined[-1]:.3f}  "
            f"5B-only={aucs_5b_only[-1]:.3f}"
        )

    return {
        "2b_only": {
            "mean": float(np.mean(aucs_baseline)),
            "std": float(np.std(aucs_baseline)),
            "folds": [float(a) for a in aucs_baseline],
        },
        "combined": {
            "mean": float(np.mean(aucs_combined)),
            "std": float(np.std(aucs_combined)),
            "folds": [float(a) for a in aucs_combined],
        },
        "5b_only": {
            "mean": float(np.mean(aucs_5b_only)),
            "std": float(np.std(aucs_5b_only)),
            "folds": [float(a) for a in aucs_5b_only],
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dit-feature-dir", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--dit-layer", type=int, default=10)
    parser.add_argument("--dit-timestep", type=int, default=600)
    parser.add_argument("--output", default="figures/combined_training_test.json")
    args = parser.parse_args()

    logger.info(f"Loading DiT features (L{args.dit_layer}, t={args.dit_timestep})...")
    X, y, sources = load_dit_features(
        args.dit_feature_dir, args.metadata, args.dit_layer, args.dit_timestep
    )
    logger.info(f"Loaded {len(X)} features, dim={X.shape[1]}")

    logger.info("=" * 60)
    logger.info("Combined Training Experiment")
    logger.info(f"  All tested on 2B held-out samples")
    logger.info(f"  DiT layer={args.dit_layer}, timestep={args.dit_timestep}")
    logger.info("=" * 60)

    results = run_experiment(X, y, sources)

    # Summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("RESULTS (tested on 2B held-out)")
    logger.info("=" * 60)
    logger.info(
        f"  2B-only:   AUC = {results['2b_only']['mean']:.3f} +/- {results['2b_only']['std']:.3f}"
    )
    logger.info(
        f"  Combined:  AUC = {results['combined']['mean']:.3f} +/- {results['combined']['std']:.3f}"
    )
    logger.info(
        f"  5B-only:   AUC = {results['5b_only']['mean']:.3f} +/- {results['5b_only']['std']:.3f}"
    )

    delta = results["combined"]["mean"] - results["2b_only"]["mean"]
    logger.info("")
    if delta > 0.02:
        logger.info(
            f"  Delta = {delta:+.3f} => MERGE 5B data. Combined training helps."
        )
    elif delta > -0.02:
        logger.info(
            f"  Delta = {delta:+.3f} => Marginal. 5B data neither helps nor hurts much."
        )
    else:
        logger.info(
            f"  Delta = {delta:+.3f} => DO NOT merge. 5B data hurts 2B performance."
        )

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results["config"] = {"layer": args.dit_layer, "timestep": args.dit_timestep}
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
