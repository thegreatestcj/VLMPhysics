#!/usr/bin/env python3
"""
Probe VAE Latent for Physics Signal
====================================
Compare physics-relevant information in VAE latent space vs DiT features.

Pipeline:
    Normal: Video -> VAE encode -> add noise -> DiT forward -> features [13, 1920]
    This:   Video -> VAE encode -> stop                     -> latent  [13, 16, 60, 90]

Features:
    - VAE latent probing (no spatial pooling for fair comparison)
    - DiT feature probing (for comparison)
    - Per-source AUC breakdown
    - UMAP visualization (colored by source + physics label)
    - Cross-source generalization matrix (train on A, test on B)

Usage:
    # Encode and probe (first run)
    python utils/probe_vae_latent.py \\
        --data-dir ~/scratch/physics/videophy_cogx \\
        --save-latents ~/scratch/physics/vae_latents_cogx.npz \\
        --output figures/vae_cogx_2b_vs_5b.png

    # Load saved latents (fast re-analysis)
    python utils/probe_vae_latent.py \\
        --load-latents ~/scratch/physics/vae_latents_cogx.npz \\
        --output figures/vae_cogx_2b_vs_5b.png
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ============================================================
# JSON serialization helper
# ============================================================


def make_json_serializable(obj):
    """Recursively convert numpy types to Python native types."""
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_json_serializable(v) for v in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


# ============================================================
# VAE encoding
# ============================================================


def load_vae(model_id: str = "THUDM/CogVideoX-2b", device: str = "cuda"):
    """Load CogVideoX VAE encoder only."""
    from diffusers import AutoencoderKLCogVideoX

    vae = AutoencoderKLCogVideoX.from_pretrained(model_id, subfolder="vae")
    vae = vae.to(device=device, dtype=torch.float16)
    vae.eval()
    logger.info("VAE loaded")
    return vae


def encode_video_vae(video: torch.Tensor, vae, device: str = "cuda") -> torch.Tensor:
    """
    Encode a video to VAE latent, NO spatial pooling (full resolution).

    Args:
        video: [T, C, H, W] in [0, 1]
        vae: CogVideoX VAE

    Returns:
        [T_latent, C_latent, H_latent, W_latent] tensor (e.g. [13, 16, 60, 90])
    """
    # [T, C, H, W] -> [1, C, T, H, W]
    video = (
        video.unsqueeze(0).permute(0, 2, 1, 3, 4).to(device=device, dtype=torch.float16)
    )
    # Normalize [0,1] -> [-1,1]
    video = video * 2.0 - 1.0

    with torch.no_grad():
        latent = vae.encode(video).latent_dist.mean  # Deterministic (mean, not sample)
        # latent: [1, C, T', H', W'] e.g. [1, 16, 13, 60, 90]

    # Rearrange to [T', C, H', W'] - keep full spatial resolution
    latent = latent.squeeze(0)  # [C, T', H', W']
    latent = latent.permute(1, 0, 2, 3)  # [T', C, H', W']
    # NO spatial pooling — flatten later for fair comparison

    return latent.float().cpu()


# ============================================================
# Data loading
# ============================================================


def load_videos_and_labels(
    data_dir: Path,
    dataset_type: str = "videophy",
    max_videos: Optional[int] = None,
) -> "Dataset":
    """Load video dataset."""
    if dataset_type == "videophy":
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from src.data.videophy_dataset import VideoPhyDataset

        dataset = VideoPhyDataset(data_root=str(data_dir), split="all", num_frames=49)
    elif dataset_type == "physion":
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from src.data.physion_dataset import PhysionDataset

        dataset = PhysionDataset(data_root=str(data_dir), split="all", num_frames=49)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

    if max_videos and max_videos > 0:
        dataset.samples = dataset.samples[:max_videos]

    logger.info(f"Dataset: {len(dataset)} videos")
    return dataset


# ============================================================
# DiT feature loading (for comparison)
# ============================================================


def load_dit_features(
    feature_dir: Path,
    metadata_path: Optional[Path],
    layer: int,
    timestep: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load pre-extracted DiT features and match with metadata."""
    feats_list = []
    labels_list = []
    sources_list = []
    names_list = []

    # Load metadata for labels
    metadata = {}
    if metadata_path and metadata_path.exists():
        with open(metadata_path) as f:
            meta_list = json.load(f)
        for item in meta_list:
            vname = item.get("video_filename", "")
            if not vname:
                vname = Path(item.get("video_path", "")).stem
            metadata[vname] = item

    # Scan feature directory
    feat_key = f"layer_{layer}"

    for video_dir in sorted(feature_dir.iterdir()):
        if not video_dir.is_dir():
            continue
        video_name = video_dir.name

        feat_path = video_dir / f"t{timestep}" / f"{feat_key}.pt"
        if not feat_path.exists():
            continue

        feat = torch.load(feat_path, map_location="cpu")
        if isinstance(feat, dict):
            feat = feat.get("pooled", feat.get("features"))
        if isinstance(feat, torch.Tensor):
            feat = feat.float().numpy()

        # Get metadata
        meta = metadata.get(video_name, {})
        label = meta.get("physics")
        if label is None:
            continue

        source = meta.get("source", "unknown")

        feats_list.append(feat.reshape(-1))
        labels_list.append(int(label))
        sources_list.append(source)
        names_list.append(video_name)

    if not feats_list:
        logger.warning("No DiT features loaded!")
        return np.array([]), np.array([]), np.array([]), np.array([])

    logger.info(f"DiT features: {len(feats_list)} videos, layer={layer}, t={timestep}")
    return (
        np.stack(feats_list),
        np.array(labels_list),
        np.array(sources_list),
        np.array(names_list),
    )


# ============================================================
# Probing
# ============================================================


def probe_auc(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    seed: int = 42,
    max_pca_dim: int = 128,
) -> Tuple[float, float]:
    """5-fold CV logistic regression AUC with PCA + StandardScaler."""
    from sklearn.decomposition import PCA
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
        n_components = min(max_pca_dim, X.shape[1], len(train_idx) - 1)
        pca = PCA(n_components=n_components)
        Xtr = pca.fit_transform(X[train_idx])
        Xte = pca.transform(X[test_idx])

        sc = StandardScaler()
        Xtr = sc.fit_transform(Xtr)
        Xte = sc.transform(Xte)

        clf = LogisticRegression(max_iter=1000, C=1.0, random_state=seed)
        clf.fit(Xtr, y[train_idx])
        proba = clf.predict_proba(Xte)[:, 1]
        aucs.append(roc_auc_score(y[test_idx], proba))

    return float(np.mean(aucs)), float(np.std(aucs))


def probe_by_source(
    X: np.ndarray, y: np.ndarray, sources: np.ndarray
) -> Dict[str, Tuple[float, float, int]]:
    """Run probe globally and per-source."""
    results = {}

    # Global
    m, s = probe_auc(X, y)
    results["GLOBAL"] = (m, s, len(y))
    logger.info(f"  GLOBAL: AUC={m:.3f}+/-{s:.3f} (n={len(y)})")

    # Per-source
    for src in sorted(set(sources)):
        mask = sources == src
        n = mask.sum()
        n_pos = (y[mask] == 1).sum()
        n_neg = (y[mask] == 0).sum()
        if n_pos < 5 or n_neg < 5:
            logger.info(f"  {src}: SKIPPED (n_pos={n_pos}, n_neg={n_neg})")
            continue
        m, s = probe_auc(X[mask], y[mask])
        results[src] = (m, s, int(n))
        logger.info(f"  {src}: AUC={m:.3f}+/-{s:.3f} (n={n})")

    return results


# ============================================================
# Cross-source generalization
# ============================================================


def cross_source_generalization(
    X: np.ndarray,
    y: np.ndarray,
    sources: np.ndarray,
    max_pca_dim: int = 128,
    seed: int = 42,
) -> Dict[str, Dict[str, float]]:
    """
    Train probe on source A, test on source B.
    Returns matrix[train_source][test_source] = AUC.
    """
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler

    unique_sources = sorted(set(sources))
    matrix = {}

    for train_src in unique_sources:
        matrix[train_src] = {}
        train_mask = sources == train_src
        Xtr, ytr = X[train_mask], y[train_mask]

        # Need both classes in training
        if (ytr == 1).sum() < 3 or (ytr == 0).sum() < 3:
            for test_src in unique_sources:
                matrix[train_src][test_src] = float("nan")
            continue

        # PCA + scale on training data
        n_components = min(max_pca_dim, Xtr.shape[1], len(Xtr) - 1)
        pca = PCA(n_components=n_components)
        Xtr_pca = pca.fit_transform(Xtr)
        sc = StandardScaler()
        Xtr_sc = sc.fit_transform(Xtr_pca)

        clf = LogisticRegression(max_iter=1000, C=1.0, random_state=seed)
        clf.fit(Xtr_sc, ytr)

        for test_src in unique_sources:
            test_mask = sources == test_src
            Xte, yte = X[test_mask], y[test_mask]

            if (yte == 1).sum() < 2 or (yte == 0).sum() < 2:
                matrix[train_src][test_src] = float("nan")
                continue

            Xte_pca = pca.transform(Xte)
            Xte_sc = sc.transform(Xte_pca)
            proba = clf.predict_proba(Xte_sc)[:, 1]

            try:
                auc = roc_auc_score(yte, proba)
            except ValueError:
                auc = float("nan")

            matrix[train_src][test_src] = float(auc)

    return matrix


# ============================================================
# Visualization
# ============================================================


def plot_comparison(
    vae_results: Dict,
    dit_results: Optional[Dict],
    output_path: str,
):
    """Bar chart comparing VAE latent AUC vs DiT feature AUC."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sources = sorted(vae_results.keys())

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(sources))
    width = 0.35

    # VAE bars
    vae_aucs = [vae_results[s][0] for s in sources]
    vae_stds = [vae_results[s][1] for s in sources]
    ax.bar(
        x - width / 2,
        vae_aucs,
        width,
        yerr=vae_stds,
        label="VAE Latent",
        color="#3498db",
        capsize=3,
        alpha=0.8,
    )

    # DiT bars (if available)
    if dit_results:
        dit_aucs = [dit_results.get(s, (0.5, 0, 0))[0] for s in sources]
        dit_stds = [dit_results.get(s, (0.5, 0, 0))[1] for s in sources]
        ax.bar(
            x + width / 2,
            dit_aucs,
            width,
            yerr=dit_stds,
            label="DiT Features",
            color="#e74c3c",
            capsize=3,
            alpha=0.8,
        )

    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, alpha=0.5)
    ax.set_ylabel("AUC (5-fold CV)", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(sources, rotation=30, ha="right", fontsize=9)
    ax.legend(fontsize=10)
    ax.set_ylim(0.4, max(0.85, ax.get_ylim()[1]))
    ax.set_title(
        "VAE Latent vs DiT Features: Where Does Physics Signal Originate?",
        fontsize=12,
        fontweight="bold",
    )

    # Annotate sample counts
    for i, s in enumerate(sources):
        n = vae_results[s][2]
        ax.annotate(
            f"n={n}",
            xy=(x[i], 0.41),
            fontsize=7,
            ha="center",
            color="gray",
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    logger.info(f"Saved comparison plot to {output_path}")
    plt.close()


def plot_umap(
    X: np.ndarray,
    y: np.ndarray,
    sources: np.ndarray,
    output_path: str,
    max_pca_dim: int = 50,
    seed: int = 42,
):
    """
    UMAP visualization: 2 panels.
    Left: colored by source (2B vs 5B)
    Right: colored by physics label (0 vs 1)
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    try:
        from umap import UMAP
    except ImportError:
        logger.warning("umap-learn not installed, skipping UMAP plot")
        return

    logger.info("Computing UMAP embedding...")

    # PCA first for speed
    n_components = min(max_pca_dim, X.shape[1], len(X) - 1)
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)

    embedding = UMAP(
        n_components=2,
        n_neighbors=15,
        min_dist=0.1,
        metric="cosine",
        random_state=seed,
    ).fit_transform(X_pca)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: source
    ax = axes[0]
    unique_sources = sorted(set(sources))
    colors_src = {"cogvideox-2b": "#e74c3c", "cogvideox-5b": "#3498db"}
    for src in unique_sources:
        mask = sources == src
        c = colors_src.get(src, "#999999")
        ax.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            c=c,
            label=src,
            s=8,
            alpha=0.5,
        )
    ax.set_title("Colored by Source (2B vs 5B)", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9, markerscale=3)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")

    # Right: physics label
    ax = axes[1]
    colors_phy = {0: "#e74c3c", 1: "#2ecc71"}
    labels_phy = {0: "physics=0", 1: "physics=1"}
    for label_val in [0, 1]:
        mask = y == label_val
        ax.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            c=colors_phy[label_val],
            label=labels_phy[label_val],
            s=8,
            alpha=0.5,
        )
    ax.set_title("Colored by Physics Label", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9, markerscale=3)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")

    plt.suptitle("VAE Latent Space: CogVideoX-2B vs 5B", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    logger.info(f"Saved UMAP plot to {output_path}")
    plt.close()


def plot_cross_source(
    matrix: Dict[str, Dict[str, float]],
    output_path: str,
):
    """Heatmap of cross-source generalization matrix."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sources = sorted(matrix.keys())
    n = len(sources)
    mat = np.zeros((n, n))
    for i, train_src in enumerate(sources):
        for j, test_src in enumerate(sources):
            mat[i, j] = matrix[train_src].get(test_src, float("nan"))

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(mat, cmap="RdYlGn", vmin=0.35, vmax=0.85, aspect="auto")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(sources, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(sources, fontsize=9)
    ax.set_xlabel("Test Source", fontsize=10)
    ax.set_ylabel("Train Source", fontsize=10)
    ax.set_title(
        "Cross-Source Generalization (VAE Latent)", fontsize=11, fontweight="bold"
    )

    # Annotate cells
    for i in range(n):
        for j in range(n):
            val = mat[i, j]
            if np.isnan(val):
                text = "N/A"
            else:
                text = f"{val:.3f}"
            color = "white" if val < 0.5 else "black"
            ax.text(j, i, text, ha="center", va="center", fontsize=11, color=color)

    plt.colorbar(im, ax=ax, label="AUC", shrink=0.8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    logger.info(f"Saved cross-source plot to {output_path}")
    plt.close()

    # Log summary
    diag = [mat[i, i] for i in range(n) if not np.isnan(mat[i, i])]
    off_diag = [
        mat[i, j]
        for i in range(n)
        for j in range(n)
        if i != j and not np.isnan(mat[i, j])
    ]
    if diag:
        logger.info(f"  Diagonal mean (within-source): {np.mean(diag):.3f}")
    if off_diag:
        logger.info(f"  Off-diagonal mean (cross-source): {np.mean(off_diag):.3f}")
    if diag and off_diag:
        logger.info(f"  Gap: {np.mean(diag) - np.mean(off_diag):.3f}")


# ============================================================
# Main
# ============================================================


def main():
    parser = argparse.ArgumentParser(
        description="Probe VAE latent for physics signal (no DiT, no noise)"
    )

    # Data source
    parser.add_argument(
        "--data-dir", type=str, default=None, help="Video dataset directory"
    )
    parser.add_argument(
        "--dataset", type=str, default="videophy", choices=["videophy", "physion"]
    )
    parser.add_argument(
        "--max-videos",
        type=int,
        default=None,
        help="Max videos to encode (0 or None = all)",
    )

    # Save/load latents
    parser.add_argument("--save-latents", type=str, default=None)
    parser.add_argument("--load-latents", type=str, default=None)

    # DiT comparison
    parser.add_argument("--dit-feature-dir", type=str, default=None)
    parser.add_argument("--metadata", type=str, default=None)
    parser.add_argument("--dit-layer", type=int, default=20)
    parser.add_argument("--dit-timestep", type=int, default=200)

    # Analysis options
    parser.add_argument(
        "--skip-umap", action="store_true", help="Skip UMAP visualization"
    )
    parser.add_argument(
        "--skip-cross-source",
        action="store_true",
        help="Skip cross-source generalization",
    )

    # Output
    parser.add_argument("--output", type=str, default="figures/vae_vs_dit.png")
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    # =========================================================
    # Phase 1: Get VAE latents
    # =========================================================
    if args.load_latents:
        logger.info(f"Loading pre-computed VAE latents from {args.load_latents}")
        data = np.load(args.load_latents, allow_pickle=True)
        vae_feats = data["features"]
        vae_labels = data["labels"]
        vae_sources = data["sources"]
        vae_names = data["names"]
        logger.info(f"Loaded {len(vae_feats)} latents, shape={vae_feats.shape}")
    else:
        if not args.data_dir:
            parser.error("--data-dir required when not using --load-latents")

        data_dir = Path(args.data_dir).expanduser()
        dataset = load_videos_and_labels(data_dir, args.dataset, args.max_videos)

        # Load VAE
        vae = load_vae(device=args.device)

        vae_latents = []
        vae_labels_list = []
        vae_sources_list = []
        vae_names_list = []

        logger.info("Encoding videos with VAE...")
        t0 = time.time()

        from tqdm import tqdm

        for idx in tqdm(range(len(dataset)), desc="VAE encoding"):
            try:
                sample = dataset[idx]
                video = sample["video"]  # [T, C, H, W]
                label = sample["label"]
                video_name = sample.get("video_name", f"video_{idx}")
                # Get source from __getitem__ return dict, not from samples tuple
                source = sample.get("source", "unknown")

                latent = encode_video_vae(video, vae, args.device)
                # latent shape: [13, 16, 60, 90] -> flatten to [1123200]
                vae_latents.append(latent.reshape(-1).numpy())
                vae_labels_list.append(label)
                vae_sources_list.append(source)
                vae_names_list.append(video_name)

            except Exception as e:
                logger.warning(f"Failed on video {idx}: {e}")
                continue

        elapsed = time.time() - t0
        if vae_latents:
            logger.info(
                f"VAE encoding done: {len(vae_latents)} videos in {elapsed:.1f}s "
                f"({elapsed / len(vae_latents):.2f}s/video)"
            )
        else:
            logger.error("No videos encoded!")
            return

        vae_feats = np.stack(vae_latents)
        vae_labels = np.array(vae_labels_list)
        vae_sources = np.array(vae_sources_list)
        vae_names = np.array(vae_names_list)

        # Save latents
        if args.save_latents:
            save_path = Path(args.save_latents).expanduser()
            save_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(
                save_path,
                features=vae_feats,
                labels=vae_labels,
                sources=vae_sources,
                names=vae_names,
            )
            logger.info(f"Saved VAE latents to {save_path}")

        # Free GPU memory
        del vae
        torch.cuda.empty_cache()

    # =========================================================
    # Phase 2: Probe VAE latents
    # =========================================================
    logger.info("=" * 60)
    logger.info(f"VAE Latent Probing (shape: {vae_feats.shape})")
    logger.info("=" * 60)

    # Log source distribution
    unique_srcs, counts = np.unique(vae_sources, return_counts=True)
    for s, c in zip(unique_srcs, counts):
        n_pos = int((vae_labels[vae_sources == s] == 1).sum())
        logger.info(f"  {s}: n={c}, physics=1: {n_pos} ({100 * n_pos / c:.0f}%)")

    vae_results = probe_by_source(vae_feats, vae_labels, vae_sources)

    # =========================================================
    # Phase 3: UMAP visualization
    # =========================================================
    output_path = Path(args.output).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not args.skip_umap:
        umap_path = str(output_path.with_name(output_path.stem + "_umap.png"))
        plot_umap(vae_feats, vae_labels, vae_sources, umap_path)

    # =========================================================
    # Phase 4: Cross-source generalization
    # =========================================================
    cross_results = None
    if not args.skip_cross_source and len(set(vae_sources)) >= 2:
        logger.info("=" * 60)
        logger.info("Cross-Source Generalization (VAE Latent)")
        logger.info("=" * 60)

        cross_results = cross_source_generalization(vae_feats, vae_labels, vae_sources)

        # Print matrix
        sources_list = sorted(cross_results.keys())
        train_test = "Train\\Test"
        header = f"{train_test:<20}" + "".join(f"{s:>15}" for s in sources_list)
        logger.info(header)
        for train_src in sources_list:
            row = f"{train_src:<20}"
            for test_src in sources_list:
                val = cross_results[train_src][test_src]
                row += f"{val:>15.3f}" if not np.isnan(val) else f"{'N/A':>15}"
            logger.info(row)

        cross_path = str(output_path.with_name(output_path.stem + "_cross_source.png"))
        plot_cross_source(cross_results, cross_path)

    # =========================================================
    # Phase 5: DiT comparison (optional)
    # =========================================================
    dit_results = None
    if args.dit_feature_dir:
        dit_dir = Path(args.dit_feature_dir).expanduser()
        meta_path = Path(args.metadata).expanduser() if args.metadata else None

        if meta_path is None:
            for candidate in [
                Path(args.data_dir).expanduser() / "metadata.json"
                if args.data_dir
                else None,
                dit_dir / "metadata.json",
            ]:
                if candidate and candidate.exists():
                    meta_path = candidate
                    break

        logger.info("=" * 60)
        logger.info(
            f"DiT Feature Probing (layer={args.dit_layer}, t={args.dit_timestep})"
        )
        logger.info("=" * 60)

        dit_feats, dit_labels, dit_sources, dit_names = load_dit_features(
            dit_dir, meta_path, args.dit_layer, args.dit_timestep
        )

        if len(dit_feats) > 0:
            dit_results = probe_by_source(dit_feats, dit_labels, dit_sources)

    # =========================================================
    # Phase 6: Summary and plots
    # =========================================================
    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)

    header = f"{'Source':<20} {'VAE AUC':>10} {'DiT AUC':>10} {'Δ (DiT-VAE)':>12}"
    logger.info(header)
    logger.info("-" * len(header))

    for src in sorted(vae_results.keys()):
        vae_auc = vae_results[src][0]
        if dit_results and src in dit_results:
            dit_auc = dit_results[src][0]
            delta = dit_auc - vae_auc
            logger.info(f"{src:<20} {vae_auc:>10.3f} {dit_auc:>10.3f} {delta:>+12.3f}")
        else:
            logger.info(f"{src:<20} {vae_auc:>10.3f} {'N/A':>10} {'N/A':>12}")

    # Interpretation
    logger.info("")
    logger.info("Interpretation:")
    if dit_results:
        global_vae = vae_results.get("GLOBAL", (0.5,))[0]
        global_dit = dit_results.get("GLOBAL", (0.5,))[0]
        delta = global_dit - global_vae
        if abs(delta) < 0.02:
            logger.info(
                f"  Δ={delta:+.3f} ≈ 0 → Physics signal comes from VAE latent, DiT only preserves it."
            )
        elif delta > 0.02:
            logger.info(
                f"  Δ={delta:+.3f} > 0 → DiT adds physics-relevant information beyond VAE latent."
            )
        else:
            logger.info(
                f"  Δ={delta:+.3f} < 0 → VAE latent is MORE informative. DiT may dilute physics signal."
            )

    if cross_results:
        sources_list = sorted(cross_results.keys())
        diag = [
            cross_results[s][s]
            for s in sources_list
            if not np.isnan(cross_results[s][s])
        ]
        off_diag = [
            cross_results[a][b]
            for a in sources_list
            for b in sources_list
            if a != b and not np.isnan(cross_results[a][b])
        ]
        if diag and off_diag:
            gap = np.mean(diag) - np.mean(off_diag)
            if gap > 0.1:
                logger.info(
                    f"  Cross-source gap={gap:.3f} → 2B and 5B encode physics differently, DO NOT merge training."
                )
            else:
                logger.info(
                    f"  Cross-source gap={gap:.3f} → 2B and 5B share physics encoding, SAFE to merge training."
                )

    # Save comparison plot
    plot_comparison(vae_results, dit_results, str(output_path))

    # Save JSON results
    json_path = output_path.with_suffix(".json")
    results_json = make_json_serializable(
        {
            "vae": {
                k: {"auc": v[0], "std": v[1], "n": v[2]} for k, v in vae_results.items()
            },
        }
    )
    if dit_results:
        results_json["dit"] = make_json_serializable(
            {k: {"auc": v[0], "std": v[1], "n": v[2]} for k, v in dit_results.items()}
        )
    if cross_results:
        results_json["cross_source"] = make_json_serializable(cross_results)

    with open(json_path, "w") as f:
        json.dump(results_json, f, indent=2)
    logger.info(f"Saved results to {json_path}")


if __name__ == "__main__":
    main()
