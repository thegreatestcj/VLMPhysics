#!/usr/bin/env python3
"""
Feature Space Visualization: t-SNE / UMAP for Physics Probing

Generates:
  1. Separation heatmap (Fisher ratio + cosine distance)
  2. UMAP/t-SNE grid (layers × timesteps), colored by physics label
  3. Category scatter plots, colored by any metadata field

Directory structure expected:
    videophy_features_2/
      {video_id}/
        t200/
          layer_10.pt   # [13, 1920] pooled features
          ...
        t400/ ...
        t600/ ...

Usage:
    # Full pipeline (heatmap + grid + category plots)
    python utils/visualize_features.py \
        --feature-dir /path/to/videophy_features_2 \
        --metadata /path/to/videophy_data/metadata.json \
        --output figures/features \
        --layers 10 15 20 25 --timesteps 200 400 600

    # Category plots only, colored by source, at specific (layer, timestep)
    python utils/visualize_features.py \
        --feature-dir /path/to/videophy_features_2 \
        --metadata /path/to/videophy_data/metadata.json \
        --output figures/features \
        --color-by source \
        --category-layers 20 --category-timesteps 200 \
        --skip-grid --skip-heatmap

    # Multiple color-by fields at once
    python utils/visualize_features.py \
        --feature-dir ... --metadata ... --output figures/features \
        --color-by physics sa source states_of_matter \
        --category-layers 15 20 --category-timesteps 200
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")  # Non-interactive backend for cluster
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

# All supported color-by fields from metadata
SUPPORTED_COLOR_FIELDS = ["physics", "sa", "source", "states_of_matter"]


# =============================================================================
# Data Loading
# =============================================================================


def _sanitize_for_dirname(text: str) -> str:
    """
    Reproduce the sanitization used when creating feature directory names.
    Replaces characters that are invalid in directory names.
    """
    for ch in ["/", "\\", ":", "*", "?", '"', "<", ">", "|"]:
        text = text.replace(ch, "_")
    return text


def load_metadata(metadata_path: str) -> Dict[str, Dict]:
    """
    Load metadata.json and build a lookup dict.

    Feature directories are named as: {source}_{caption}

    Returns:
        Dict mapping possible directory names to metadata dict.
    """
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

        # Primary key: {source}_{caption} with spaces as underscores
        key1 = _sanitize_for_dirname(f"{source}_{caption}").replace(" ", "_")
        lookup[key1] = meta

        # Also try with spaces kept
        key2 = _sanitize_for_dirname(f"{source}_{caption}")
        lookup[key2] = meta

        # Fallback: video_filename stem
        if "video_filename" in item:
            stem = Path(item["video_filename"]).stem
            lookup[stem] = meta

    logger.info(f"Loaded metadata for {len(items)} videos ({len(lookup)} lookup keys)")
    return lookup


def match_video_to_metadata(
    video_dir_name: str, metadata_lookup: Dict[str, Dict]
) -> Optional[Dict]:
    """
    Match a feature directory name to metadata.
    Tries exact match, normalized match, then prefix match.
    """
    # Exact match
    if video_dir_name in metadata_lookup:
        return metadata_lookup[video_dir_name]

    # Normalized match
    normalized = video_dir_name.replace(" ", "_")
    if normalized in metadata_lookup:
        return metadata_lookup[normalized]

    # Prefix match for truncated directory names
    for prefix_len in [80, 60, 40]:
        if len(video_dir_name) >= prefix_len:
            prefix = video_dir_name[:prefix_len]
            for key, meta in metadata_lookup.items():
                if key[:prefix_len] == prefix:
                    return meta

    return None


def discover_structure(feature_dir: Path) -> Tuple[List[str], List[int], List[int]]:
    """Auto-discover available video dirs, timesteps, and layers."""
    video_dirs = sorted(
        [
            d.name
            for d in feature_dir.iterdir()
            if d.is_dir() and not d.name.startswith((".", "_"))
        ]
    )

    if not video_dirs:
        raise ValueError(f"No video directories found in {feature_dir}")

    sample_dir = feature_dir / video_dirs[0]
    timesteps = sorted(
        [
            int(d.name[1:])
            for d in sample_dir.iterdir()
            if d.is_dir() and d.name.startswith("t")
        ]
    )

    sample_t_dir = sample_dir / f"t{timesteps[0]}"
    layers = sorted(
        [int(f.stem.split("_")[1]) for f in sample_t_dir.glob("layer_*.pt")]
    )

    logger.info(
        f"Discovered: {len(video_dirs)} videos, timesteps={timesteps}, layers={layers}"
    )
    return video_dirs, timesteps, layers


def load_features(
    feature_dir: Path,
    video_dirs: List[str],
    timesteps: List[int],
    layers: List[int],
    metadata_lookup: Dict[str, Dict],
) -> Tuple[Dict[Tuple[int, int], np.ndarray], List[Dict], List[str]]:
    """
    Load all features into arrays organized by (layer, timestep).

    Returns:
        features_dict: {(layer, timestep): np.array [N, D]}
        valid_metas: List of metadata dicts for matched videos
        valid_videos: List of matched video directory names
    """
    valid_videos = []
    valid_metas = []

    for vdir in video_dirs:
        meta = match_video_to_metadata(vdir, metadata_lookup)
        if meta is None or meta["physics"] < 0:
            continue

        # Check all (layer, timestep) files exist
        all_exist = True
        for t in timesteps:
            for l in layers:
                pt_path = feature_dir / vdir / f"t{t}" / f"layer_{l}.pt"
                if not pt_path.exists():
                    all_exist = False
                    break
            if not all_exist:
                break

        if all_exist:
            valid_videos.append(vdir)
            valid_metas.append(meta)

    n = len(valid_videos)
    n_unmatched = len(video_dirs) - n
    logger.info(
        f"Matched {n} / {len(video_dirs)} videos with metadata and complete features"
    )

    if n_unmatched > 0:
        unmatched = [v for v in video_dirs if v not in valid_videos][:5]
        logger.warning(f"  Unmatched examples (first 5): {unmatched}")
        matched_examples = valid_videos[:3]
        logger.info(f"  Matched examples (first 3): {matched_examples}")

    if n == 0:
        raise ValueError(
            "No videos matched. Check metadata path and feature directory names."
        )

    # Load features
    features_dict = {}
    for l in layers:
        for t in timesteps:
            feats = []
            for vdir in valid_videos:
                pt_path = feature_dir / vdir / f"t{t}" / f"layer_{l}.pt"
                feat = torch.load(pt_path, map_location="cpu", weights_only=True)
                # [13, 1920] → flatten → [24960], preserving temporal info
                feat = feat.float().reshape(-1).numpy()
                feats.append(feat)
            features_dict[(l, t)] = np.stack(feats)
            logger.info(
                f"  Loaded layer={l}, t={t}: shape={features_dict[(l, t)].shape}"
            )

    return features_dict, valid_metas, valid_videos


def get_color_labels(metas: List[Dict], field: str) -> np.ndarray:
    """
    Extract color labels from metadata for a given field.

    Args:
        metas: List of metadata dicts
        field: One of SUPPORTED_COLOR_FIELDS

    Returns:
        np.array of string labels
    """
    if field not in SUPPORTED_COLOR_FIELDS:
        raise ValueError(
            f"Unsupported color field: {field}. "
            f"Supported: {SUPPORTED_COLOR_FIELDS}"
        )

    raw = [m.get(field, "unknown") for m in metas]

    # Convert numeric fields to readable strings
    if field == "physics":
        return np.array(["physics=1" if v == 1 else "physics=0" for v in raw])
    elif field == "sa":
        return np.array(["SA=1" if v == 1 else "SA=0" for v in raw])
    else:
        return np.array([str(v) for v in raw])


# =============================================================================
# Dimensionality Reduction
# =============================================================================


def run_reduction(
    features: np.ndarray,
    method: str = "both",
    random_state: int = 42,
    pca_dim: int = 256,
) -> Dict[str, np.ndarray]:
    """
    Run t-SNE and/or UMAP on features.
    PCA is applied first for high-dimensional inputs.
    """
    from sklearn.decomposition import PCA

    if features.shape[1] > pca_dim:
        logger.info(f"  PCA: {features.shape[1]} → {pca_dim}")
        pca = PCA(n_components=pca_dim, random_state=random_state)
        features = pca.fit_transform(features)

    results = {}

    if method in ("umap", "both"):
        try:
            import umap

            logger.info(f"  Running UMAP on {features.shape}...")
            reducer = umap.UMAP(
                n_components=2,
                n_neighbors=15,
                min_dist=0.1,
                metric="cosine",
                random_state=random_state,
            )
            results["UMAP"] = reducer.fit_transform(features)
        except ImportError:
            logger.warning("umap-learn not installed. pip install umap-learn")

    if method in ("tsne", "both"):
        from sklearn.manifold import TSNE

        logger.info(f"  Running t-SNE on {features.shape}...")
        reducer = TSNE(
            n_components=2,
            perplexity=min(30, len(features) // 4),
            random_state=random_state,
            init="pca",
            learning_rate="auto",
        )
        results["t-SNE"] = reducer.fit_transform(features)

    return results


# =============================================================================
# Plotting
# =============================================================================

# Distinct colors for categorical plots
DISTINCT_COLORS = [
    "#e74c3c",  # red
    "#2980b9",  # blue
    "#27ae60",  # green
    "#f39c12",  # orange
    "#8e44ad",  # purple
    "#1abc9c",  # teal
    "#d35400",  # dark orange
    "#c0392b",  # dark red
    "#2c3e50",  # navy
    "#7f8c8d",  # gray
]


def plot_grid(
    features_dict: Dict[Tuple[int, int], np.ndarray],
    labels: np.ndarray,
    layers: List[int],
    timesteps: List[int],
    method: str = "umap",
    output_path: str = "feature_grid.png",
    random_state: int = 42,
):
    """
    Plot a (layers × timesteps) grid of scatter plots.
    Color = physics label (green=plausible, red=implausible)
    """
    n_rows = len(layers)
    n_cols = len(timesteps)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4.5 * n_cols, 4 * n_rows),
        squeeze=False,
    )

    logger.info(f"Computing {method} for {n_rows * n_cols} (layer, timestep) pairs...")

    for i, l in enumerate(layers):
        for j, t in enumerate(timesteps):
            ax = axes[i][j]
            feats = features_dict[(l, t)]

            reductions = run_reduction(feats, method=method, random_state=random_state)
            method_name = list(reductions.keys())[0]
            coords = reductions[method_name]

            mask_pos = labels == 1
            mask_neg = labels == 0

            ax.scatter(
                coords[mask_neg, 0], coords[mask_neg, 1],
                c="#e74c3c", alpha=0.35, s=8, label="Physics ✗",
                rasterized=True,
            )
            ax.scatter(
                coords[mask_pos, 0], coords[mask_pos, 1],
                c="#2ecc71", alpha=0.35, s=8, label="Physics ✓",
                rasterized=True,
            )

            ax.set_xticks([])
            ax.set_yticks([])

            if i == 0:
                ax.set_title(f"t = {t}", fontsize=14, fontweight="bold")
            if j == 0:
                ax.set_ylabel(f"Layer {l}", fontsize=14, fontweight="bold")

    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#2ecc71",
               markersize=8, label="Physics ✓ (plausible)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#e74c3c",
               markersize=8, label="Physics ✗ (implausible)"),
    ]
    fig.legend(
        handles=legend_elements, loc="lower center",
        ncol=2, fontsize=12, frameon=True, bbox_to_anchor=(0.5, -0.02),
    )

    n_pos = int(labels.sum())
    n_neg = len(labels) - n_pos
    fig.suptitle(
        f"DiT Feature Space Visualization ({method_name})\n"
        f"N={len(labels)} videos ({n_pos} plausible, {n_neg} implausible)",
        fontsize=16, fontweight="bold", y=1.02,
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    logger.info(f"Saved grid plot to {output_path}")
    plt.close()


def plot_by_category(
    features_dict: Dict[Tuple[int, int], np.ndarray],
    color_labels: np.ndarray,
    layer: int,
    timestep: int,
    color_field: str = "states_of_matter",
    method: str = "umap",
    output_path: str = "feature_by_category.png",
    random_state: int = 42,
):
    """
    Single scatter plot at a given (layer, timestep), colored by any field.
    """
    feats = features_dict[(layer, timestep)]
    reductions = run_reduction(feats, method=method, random_state=random_state)
    method_name = list(reductions.keys())[0]
    coords = reductions[method_name]

    unique_labels = sorted(set(color_labels))
    colors = DISTINCT_COLORS[: len(unique_labels)]

    fig, ax = plt.subplots(figsize=(10, 8))

    for idx, label in enumerate(unique_labels):
        mask = color_labels == label
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            c=colors[idx], alpha=0.5, s=12, label=label,
            rasterized=True,
        )

    ax.set_xticks([])
    ax.set_yticks([])

    # Human-readable title for each field
    field_titles = {
        "physics": "Physics Label",
        "sa": "Semantic Alignment",
        "source": "Generation Model",
        "states_of_matter": "States of Matter",
    }
    title_str = field_titles.get(color_field, color_field)

    ax.set_title(
        f"Feature Space by {title_str} ({method_name})\n"
        f"Layer {layer}, t={timestep}, N={len(color_labels)}",
        fontsize=14, fontweight="bold",
    )
    ax.legend(
        loc="center left", bbox_to_anchor=(1.0, 0.5),
        fontsize=10, markerscale=2,
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    logger.info(f"Saved {color_field} plot to {output_path}")
    plt.close()


# =============================================================================
# Quantitative separation metric
# =============================================================================


def compute_separation_metrics(
    features_dict: Dict[Tuple[int, int], np.ndarray],
    labels: np.ndarray,
    layers: List[int],
    timesteps: List[int],
) -> Dict[Tuple[int, int], Dict[str, float]]:
    """
    Compute feature distribution separation metrics (no training needed).
    Uses physics binary labels for Fisher ratio and cosine distance.
    """
    from numpy.linalg import norm

    results = {}

    for l in layers:
        for t in timesteps:
            feats = features_dict[(l, t)]
            pos_feats = feats[labels == 1]
            neg_feats = feats[labels == 0]

            mu_pos = pos_feats.mean(axis=0)
            mu_neg = neg_feats.mean(axis=0)
            var_pos = pos_feats.var(axis=0)
            var_neg = neg_feats.var(axis=0)

            denom = var_pos + var_neg + 1e-8
            fisher = ((mu_pos - mu_neg) ** 2 / denom).mean()

            cos_sim = np.dot(mu_pos, mu_neg) / (norm(mu_pos) * norm(mu_neg) + 1e-8)
            cos_dist = 1.0 - cos_sim

            results[(l, t)] = {
                "fisher": float(fisher),
                "cosine_dist": float(cos_dist),
            }

            logger.info(
                f"  Layer {l}, t={t}: Fisher={fisher:.6f}, CosDist={cos_dist:.6f}"
            )

    return results


def plot_separation_heatmap(
    metrics: Dict[Tuple[int, int], Dict[str, float]],
    layers: List[int],
    timesteps: List[int],
    output_path: str = "separation_heatmap.png",
):
    """Plot heatmaps of separation metrics."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, metric_name, title in zip(
        axes,
        ["fisher", "cosine_dist"],
        ["Fisher Discriminant Ratio ↑", "Cosine Distance Between Centroids ↑"],
    ):
        matrix = np.zeros((len(layers), len(timesteps)))
        for i, l in enumerate(layers):
            for j, t in enumerate(timesteps):
                matrix[i, j] = metrics[(l, t)][metric_name]

        im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto")
        ax.set_xticks(range(len(timesteps)))
        ax.set_xticklabels([f"t={t}" for t in timesteps])
        ax.set_yticks(range(len(layers)))
        ax.set_yticklabels([f"Layer {l}" for l in layers])
        ax.set_xlabel("Timestep (noisy → clean)")
        ax.set_ylabel("Layer (shallow → deep)")
        ax.set_title(title, fontsize=12, fontweight="bold")

        for i in range(len(layers)):
            for j in range(len(timesteps)):
                val = matrix[i, j]
                ax.text(
                    j, i, f"{val:.4f}", ha="center", va="center",
                    fontsize=9,
                    color="black" if val < matrix.max() * 0.7 else "white",
                )

        plt.colorbar(im, ax=ax, shrink=0.8)

    fig.suptitle(
        "Feature Separation: Physics Plausible vs Implausible",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    logger.info(f"Saved separation heatmap to {output_path}")
    plt.close()


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Visualize DiT feature space for physics probing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Supported --color-by fields:
  physics           Physics plausibility (0/1)
  sa                Semantic alignment (0/1)
  source            Generation model (lavie, pika, modelscope, ...)
  states_of_matter  Material interaction type (fluid_fluid, solid_solid, ...)

Examples:
  # Full pipeline
  python utils/visualize_features.py --feature-dir ... --metadata ... \\
      --output figures/features --layers 10 15 20 25 --timesteps 200 400 600

  # Category plots only, colored by source, at Layer 20 t=200
  python utils/visualize_features.py --feature-dir ... --metadata ... \\
      --output figures/features --color-by source \\
      --category-layers 20 --category-timesteps 200 \\
      --skip-grid --skip-heatmap

  # All fields at multiple (layer, timestep) pairs
  python utils/visualize_features.py --feature-dir ... --metadata ... \\
      --output figures/features \\
      --color-by physics sa source states_of_matter \\
      --category-layers 15 20 --category-timesteps 200 400 \\
      --skip-grid --skip-heatmap
        """,
    )

    # Required
    parser.add_argument(
        "--feature-dir", type=str, required=True,
        help="Path to extracted features directory",
    )
    parser.add_argument(
        "--metadata", type=str, required=True,
        help="Path to metadata.json",
    )
    parser.add_argument(
        "--output", type=str, default="figures/features",
        help="Output prefix for all generated figures",
    )

    # Method
    parser.add_argument(
        "--method", type=str, default="umap", choices=["tsne", "umap", "both"],
        help="Dimensionality reduction method (default: umap)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # Layer/timestep for heatmap + grid
    parser.add_argument(
        "--layers", type=int, nargs="+", default=None,
        help="Layers for heatmap and grid (default: auto-discover)",
    )
    parser.add_argument(
        "--timesteps", type=int, nargs="+", default=None,
        help="Timesteps for heatmap and grid (default: auto-discover)",
    )

    # Category plot controls
    parser.add_argument(
        "--color-by", type=str, nargs="+",
        default=["states_of_matter"],
        choices=SUPPORTED_COLOR_FIELDS,
        help="Field(s) to color category plots by (default: states_of_matter)",
    )
    parser.add_argument(
        "--category-layers", type=int, nargs="+", default=None,
        help="Layer(s) for category plots (default: best by Fisher ratio)",
    )
    parser.add_argument(
        "--category-timesteps", type=int, nargs="+", default=None,
        help="Timestep(s) for category plots (default: best by Fisher ratio)",
    )

    # Skip flags
    parser.add_argument(
        "--skip-heatmap", action="store_true",
        help="Skip separation heatmap generation",
    )
    parser.add_argument(
        "--skip-grid", action="store_true",
        help="Skip UMAP/t-SNE grid generation",
    )
    parser.add_argument(
        "--skip-category", action="store_true",
        help="Skip category plot generation",
    )

    args = parser.parse_args()

    feature_dir = Path(args.feature_dir)
    output_prefix = args.output.rstrip("/")
    Path(output_prefix).parent.mkdir(parents=True, exist_ok=True)

    # Discover structure
    video_dirs, all_timesteps, all_layers = discover_structure(feature_dir)

    # Grid layers/timesteps
    grid_layers = args.layers if args.layers else all_layers
    grid_timesteps = args.timesteps if args.timesteps else all_timesteps

    # Category layers/timesteps
    cat_layers = args.category_layers
    cat_timesteps = args.category_timesteps

    # Collect all unique layers/timesteps we need to load
    load_layers = set(grid_layers)
    load_timesteps = set(grid_timesteps)
    if cat_layers:
        load_layers.update(cat_layers)
    if cat_timesteps:
        load_timesteps.update(cat_timesteps)
    load_layers = sorted(load_layers)
    load_timesteps = sorted(load_timesteps)

    logger.info(f"Grid: layers={grid_layers}, timesteps={grid_timesteps}")
    logger.info(f"Category: layers={cat_layers}, timesteps={cat_timesteps}")
    logger.info(f"Color-by: {args.color_by}")
    logger.info(f"Loading features for layers={load_layers}, timesteps={load_timesteps}")

    # Load metadata & features
    metadata_lookup = load_metadata(args.metadata)
    features_dict, valid_metas, video_ids = load_features(
        feature_dir, video_dirs, load_timesteps, load_layers, metadata_lookup
    )

    # Physics labels (binary, for heatmap + grid)
    physics_labels = np.array([m["physics"] for m in valid_metas])

    # =========================================================================
    # 1. Separation heatmap
    # =========================================================================
    metrics = None
    if not args.skip_heatmap:
        logger.info("=" * 50)
        logger.info("Computing separation metrics...")
        metrics = compute_separation_metrics(
            features_dict, physics_labels, grid_layers, grid_timesteps
        )
        plot_separation_heatmap(
            metrics, grid_layers, grid_timesteps,
            f"{output_prefix}_heatmap.png",
        )

    # =========================================================================
    # 2. Grid plot (physics labels)
    # =========================================================================
    if not args.skip_grid:
        logger.info("=" * 50)
        logger.info("Generating grid visualization...")

        methods = ["umap", "tsne"] if args.method == "both" else [args.method]
        for m in methods:
            plot_grid(
                features_dict, physics_labels, grid_layers, grid_timesteps,
                method=m,
                output_path=f"{output_prefix}_grid_{m}.png",
                random_state=args.seed,
            )

    # =========================================================================
    # 3. Category plots
    # =========================================================================
    if not args.skip_category:
        logger.info("=" * 50)
        logger.info("Generating category visualizations...")

        # Determine (layer, timestep) pairs for category plots
        if cat_layers and cat_timesteps:
            cat_pairs = [(l, t) for l in cat_layers for t in cat_timesteps]
        elif metrics is not None:
            # Auto-select best by Fisher ratio
            best_key = max(metrics, key=lambda k: metrics[k]["fisher"])
            cat_pairs = [best_key]
            logger.info(
                f"Auto-selected best (layer, timestep) = {best_key} "
                f"(Fisher={metrics[best_key]['fisher']:.6f})"
            )
        else:
            # Default: first grid layer and timestep
            cat_pairs = [(grid_layers[0], grid_timesteps[0])]

        pick_method = "umap" if args.method != "tsne" else "tsne"

        for field in args.color_by:
            color_labels = get_color_labels(valid_metas, field)

            for layer, timestep in cat_pairs:
                out_path = f"{output_prefix}_{field}_l{layer}_t{timestep}.png"
                logger.info(f"  Plotting {field} at Layer {layer}, t={timestep}")

                plot_by_category(
                    features_dict, color_labels, layer, timestep,
                    color_field=field,
                    method=pick_method,
                    output_path=out_path,
                    random_state=args.seed,
                )

    logger.info("=" * 50)
    logger.info("Done!")


if __name__ == "__main__":
    main()