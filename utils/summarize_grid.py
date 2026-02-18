#!/usr/bin/env python3
"""
Summarize layer x timestep grid search results.

Reads all figures/grid_l*_t*.json files and produces:
  1. Console summary table
  2. Heatmap of DiT AUC and Delta(DiT-VAE) for 2B and 5B

Usage:
    python utils/summarize_grid.py
    python utils/summarize_grid.py --fig-dir figures --output figures/grid_summary.png
"""

import argparse
import json
import glob
import re
from pathlib import Path

import numpy as np


def load_grid_results(fig_dir: str) -> list:
    """Load all grid_l*_t*.json files."""
    pattern = str(Path(fig_dir) / "grid_l*_t*.json")
    results = []
    for path in sorted(glob.glob(pattern)):
        match = re.search(r"grid_l(\d+)_t(\d+)", path)
        if not match:
            continue
        layer = int(match.group(1))
        timestep = int(match.group(2))
        with open(path) as f:
            data = json.load(f)
        results.append({"layer": layer, "timestep": timestep, "data": data})
    return results


def print_summary(results: list):
    """Print formatted summary tables."""
    layers = sorted(set(r["layer"] for r in results))
    timesteps = sorted(set(r["timestep"] for r in results))

    # Build lookup
    lookup = {}
    for r in results:
        lookup[(r["layer"], r["timestep"])] = r["data"]

    sources = ["cogvideox-2b", "cogvideox-5b"]

    for source in sources:
        # --- DiT AUC table ---
        print(f"\n{'=' * 60}")
        print(f"  {source} - DiT AUC (linear probe, 5-fold CV)")
        print(f"{'=' * 60}")
        header = f"{'Layer':<8}" + "".join(f"{'t=' + str(t):>10}" for t in timesteps)
        print(header)
        print("-" * len(header))
        for layer in layers:
            row = f"L{layer:<7}"
            for t in timesteps:
                data = lookup.get((layer, t))
                if data and "dit" in data and source in data["dit"]:
                    auc = data["dit"][source]["auc"]
                    row += f"{auc:>10.3f}"
                else:
                    row += f"{'---':>10}"
            print(row)

        # --- Delta(DiT-VAE) table ---
        print(f"\n  {source} - Delta(DiT - VAE)")
        print("-" * len(header))
        for layer in layers:
            row = f"L{layer:<7}"
            for t in timesteps:
                data = lookup.get((layer, t))
                if (
                    data
                    and "dit" in data
                    and source in data["dit"]
                    and "vae" in data
                    and source in data["vae"]
                ):
                    dit_auc = data["dit"][source]["auc"]
                    vae_auc = data["vae"][source]["auc"]
                    delta = dit_auc - vae_auc
                    row += f"{delta:>+10.3f}"
                else:
                    row += f"{'---':>10}"
            print(row)

    # --- VAE AUC sanity check (should be ~constant across runs) ---
    print(f"\n{'=' * 60}")
    print(f"  VAE AUC sanity check (should be ~constant)")
    print(f"{'=' * 60}")
    for source in sources:
        vals = []
        for r in results:
            if "vae" in r["data"] and source in r["data"]["vae"]:
                vals.append(r["data"]["vae"][source]["auc"])
        if vals:
            print(
                f"  {source}: mean={np.mean(vals):.3f}, "
                f"std={np.std(vals):.3f}, "
                f"range=[{min(vals):.3f}, {max(vals):.3f}]"
            )

    # --- Best config ---
    print(f"\n{'=' * 60}")
    print(f"  Best configurations")
    print(f"{'=' * 60}")
    for source in sources:
        best_dit = None
        best_delta = None
        for r in results:
            data = r["data"]
            if "dit" not in data or source not in data["dit"]:
                continue
            dit_auc = data["dit"][source]["auc"]
            vae_auc = (
                data["vae"][source]["auc"] if source in data.get("vae", {}) else 0.5
            )
            delta = dit_auc - vae_auc

            if best_dit is None or dit_auc > best_dit[0]:
                best_dit = (dit_auc, r["layer"], r["timestep"])
            if best_delta is None or delta > best_delta[0]:
                best_delta = (delta, r["layer"], r["timestep"])

        if best_dit:
            print(f"  {source}:")
            print(
                f"    Highest DiT AUC:    {best_dit[0]:.3f} @ L{best_dit[1]}, t={best_dit[2]}"
            )
            print(
                f"    Highest Delta:    {best_delta[0]:+.3f} @ L{best_delta[1]}, t={best_delta[2]}"
            )


def plot_heatmaps(results: list, output_path: str):
    """Plot 2x2 heatmap: rows = {2B, 5B}, cols = {DiT AUC, Delta}."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    layers = sorted(set(r["layer"] for r in results))
    timesteps = sorted(set(r["timestep"] for r in results))
    sources = ["cogvideox-2b", "cogvideox-5b"]
    source_labels = ["CogVideoX-2B", "CogVideoX-5B"]

    lookup = {}
    for r in results:
        lookup[(r["layer"], r["timestep"])] = r["data"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    for row_idx, (source, label) in enumerate(zip(sources, source_labels)):
        auc_mat = np.full((len(layers), len(timesteps)), np.nan)
        delta_mat = np.full((len(layers), len(timesteps)), np.nan)

        for i, layer in enumerate(layers):
            for j, t in enumerate(timesteps):
                data = lookup.get((layer, t))
                if data and "dit" in data and source in data["dit"]:
                    auc_mat[i, j] = data["dit"][source]["auc"]
                if (
                    data
                    and "dit" in data
                    and source in data["dit"]
                    and "vae" in data
                    and source in data["vae"]
                ):
                    delta_mat[i, j] = (
                        data["dit"][source]["auc"] - data["vae"][source]["auc"]
                    )

        # Left: DiT AUC
        ax = axes[row_idx, 0]
        im = ax.imshow(auc_mat, cmap="YlGn", vmin=0.50, vmax=0.70, aspect="auto")
        ax.set_xticks(range(len(timesteps)))
        ax.set_xticklabels([f"t={t}" for t in timesteps])
        ax.set_yticks(range(len(layers)))
        ax.set_yticklabels([f"L{l}" for l in layers])
        ax.set_title(f"{label} - DiT AUC", fontsize=11, fontweight="bold")
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Layer")
        for i in range(len(layers)):
            for j in range(len(timesteps)):
                val = auc_mat[i, j]
                if not np.isnan(val):
                    ax.text(
                        j,
                        i,
                        f"{val:.3f}",
                        ha="center",
                        va="center",
                        fontsize=10,
                        color="white" if val > 0.63 else "black",
                    )
        plt.colorbar(im, ax=ax, shrink=0.8)

        # Right: Delta(DiT - VAE) with diverging colormap
        ax = axes[row_idx, 1]
        vmax = max(0.05, np.nanmax(np.abs(delta_mat)))
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
        im = ax.imshow(delta_mat, cmap="RdBu_r", norm=norm, aspect="auto")
        ax.set_xticks(range(len(timesteps)))
        ax.set_xticklabels([f"t={t}" for t in timesteps])
        ax.set_yticks(range(len(layers)))
        ax.set_yticklabels([f"L{l}" for l in layers])
        ax.set_title(f"{label} - Delta(DiT - VAE)", fontsize=11, fontweight="bold")
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Layer")
        for i in range(len(layers)):
            for j in range(len(timesteps)):
                val = delta_mat[i, j]
                if not np.isnan(val):
                    ax.text(
                        j,
                        i,
                        f"{val:+.3f}",
                        ha="center",
                        va="center",
                        fontsize=10,
                        color="white" if abs(val) > vmax * 0.6 else "black",
                    )
        plt.colorbar(im, ax=ax, shrink=0.8)

    plt.suptitle(
        "Layer x Timestep Grid: Physics Signal in CogVideoX DiT Features",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved heatmap to {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Summarize grid search results")
    parser.add_argument(
        "--fig-dir",
        type=str,
        default="figures",
        help="Directory containing grid_l*_t*.json files",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="figures/grid_summary.png",
        help="Output heatmap path",
    )
    args = parser.parse_args()

    results = load_grid_results(args.fig_dir)
    if not results:
        print(f"No grid_l*_t*.json files found in {args.fig_dir}/")
        return

    print(f"Found {len(results)} grid results")
    print_summary(results)
    plot_heatmaps(results, args.output)


if __name__ == "__main__":
    main()
