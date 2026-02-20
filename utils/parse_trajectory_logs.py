#!/usr/bin/env python3
"""
Parse SLURM generation logs to analyze trajectory selection.

Extracts:
  1. Per-video scores at each checkpoint
  2. Whether first-checkpoint argmax matches final winner
  3. Score distributions and pruning statistics

Usage:
    # Parse single log file
    python utils/parse_trajectory_logs.py \
        --logs slurm/phygenbench/physics/traj8-406487.out

    # Parse multiple log files (split jobs)
    python utils/parse_trajectory_logs.py \
        --logs slurm/phygenbench/physics/traj8-406486.out \
               slurm/phygenbench/physics/traj8-406487.out \
        --output figures/trajectory_analysis
"""

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


def parse_keep_prune(text: str) -> List[Tuple[int, float]]:
    """
    Parse 'Keep [(1, '0.998'), (0, '0.980')]' into [(1, 0.998), (0, 0.980)]
    """
    entries = re.findall(r"\((\d+),\s*'([0-9.]+)'\)", text)
    return [(int(idx), float(score)) for idx, score in entries]


def parse_log_file(log_path: str) -> List[Dict]:
    """
    Parse a SLURM generation log file.

    Returns list of per-video records:
    {
        "video_idx": 80,
        "prompt_preview": "A puddle of oil...",
        "n_trajectories": 4,
        "checkpoints": [
            {
                "step": 20,
                "timestep": 599,
                "keep": [(1, 0.998), (0, 0.980)],
                "prune": [(3, 0.870), (2, 0.062)],
                "all_scores": {0: 0.980, 1: 0.998, 2: 0.062, 3: 0.870}
            },
            ...
        ],
        "final_idx": 1,
        "final_score": 1.000,
        "sampling_time": 474.6,
        "total_time": 497.3
    }
    """
    records = []
    current = None

    with open(log_path, "r") as f:
        for line in f:
            line = line.strip()

            # New video: [80] Interference & Diffraction: A puddle of oil...
            match = re.search(r"\[(\d+)\]\s*(.+?)\.{3,}|$", line)
            if match and "Mode:" not in line:
                # Check if this is really a prompt line (has category prefix or text)
                idx_str = match.group(1)
                prompt = match.group(2) if match.group(2) else ""

                # Only start new record if we see a prompt index
                if idx_str:
                    if current is not None:
                        records.append(current)
                    current = {
                        "video_idx": int(idx_str),
                        "prompt_preview": prompt.strip()[:80],
                        "n_trajectories": 4,
                        "checkpoints": [],
                        "final_idx": None,
                        "final_score": None,
                        "sampling_time": None,
                        "total_time": None,
                    }

            if current is None:
                continue

            # Mode line: Mode: prune, Trajectories: 4, Checkpoint steps: [20, 30]
            match = re.search(r"Trajectories:\s*(\d+)", line)
            if match:
                current["n_trajectories"] = int(match.group(1))

            # Checkpoint: Step 20 (t=599): Keep [...], Prune [...]
            match = re.search(
                r"Step\s+(\d+)\s+\(t=(\d+)\):\s+Keep\s+(\[.*?\]),\s+Prune\s+(\[.*?\])",
                line,
            )
            if match:
                step = int(match.group(1))
                timestep = int(match.group(2))
                keep = parse_keep_prune(match.group(3))
                prune = parse_keep_prune(match.group(4))

                all_scores = {}
                for idx, score in keep + prune:
                    all_scores[idx] = score

                current["checkpoints"].append(
                    {
                        "step": step,
                        "timestep": timestep,
                        "keep": keep,
                        "prune": prune,
                        "all_scores": all_scores,
                    }
                )

            # Final: Best trajectory: idx=1, score=1.000, sampling_time=474.6s
            match = re.search(
                r"Best trajectory:\s+idx=(\d+),\s+score=([0-9.]+),\s+sampling_time=([0-9.]+)s",
                line,
            )
            if match:
                current["final_idx"] = int(match.group(1))
                current["final_score"] = float(match.group(2))
                current["sampling_time"] = float(match.group(3))

            # Total time: Done in 497.3s (sample=474.6s, decode=21.1s), score=1.000
            match = re.search(r"Done in ([0-9.]+)s", line)
            if match:
                current["total_time"] = float(match.group(1))

    # Don't forget last record
    if current is not None:
        records.append(current)

    return records


def analyze_selection(records: List[Dict]) -> Dict:
    """
    Analyze trajectory selection patterns.

    Key question: Does multi-step pruning matter, or would
    picking the argmax at the first checkpoint give the same result?
    """
    results = {
        "total_videos": len(records),
        "first_checkpoint_agreement": 0,
        "first_checkpoint_disagreement": 0,
        "disagreement_cases": [],
        "score_distributions": {},
        "per_checkpoint_stats": [],
    }

    first_cp_scores = []  # Score of first-checkpoint argmax
    final_scores = []  # Score of final winner
    first_cp_would_have_scored = []  # What score first-cp winner got at final checkpoint

    for rec in records:
        if not rec["checkpoints"] or rec["final_idx"] is None:
            continue

        # First checkpoint
        cp1 = rec["checkpoints"][0]
        cp1_scores = cp1["all_scores"]
        cp1_argmax = max(cp1_scores, key=cp1_scores.get)
        cp1_max_score = cp1_scores[cp1_argmax]

        final_idx = rec["final_idx"]
        final_score = rec["final_score"]

        first_cp_scores.append(cp1_max_score)
        final_scores.append(final_score)

        if cp1_argmax == final_idx:
            results["first_checkpoint_agreement"] += 1
        else:
            results["first_checkpoint_disagreement"] += 1

            # Track what happened
            case = {
                "video_idx": rec["video_idx"],
                "prompt": rec["prompt_preview"],
                "cp1_winner": cp1_argmax,
                "cp1_winner_score": cp1_max_score,
                "final_winner": final_idx,
                "final_winner_score": final_score,
            }

            # What did the cp1 winner score at cp1?
            # And what did the final winner score at cp1?
            final_winner_cp1_score = cp1_scores.get(final_idx, None)
            case["final_winner_cp1_score"] = final_winner_cp1_score
            case["score_gap_at_cp1"] = cp1_max_score - (final_winner_cp1_score or 0)

            # Check if cp1 winner was pruned at cp1 or survived to cp2
            cp1_kept_ids = [idx for idx, _ in cp1["keep"]]
            case["cp1_winner_survived_cp1"] = cp1_argmax in cp1_kept_ids

            results["disagreement_cases"].append(case)

    total_with_data = (
        results["first_checkpoint_agreement"] + results["first_checkpoint_disagreement"]
    )
    if total_with_data > 0:
        results["agreement_rate"] = (
            results["first_checkpoint_agreement"] / total_with_data
        )
    else:
        results["agreement_rate"] = None

    # Score statistics
    if first_cp_scores:
        results["score_distributions"] = {
            "first_checkpoint_argmax_score": {
                "mean": float(np.mean(first_cp_scores)),
                "std": float(np.std(first_cp_scores)),
                "min": float(np.min(first_cp_scores)),
                "max": float(np.max(first_cp_scores)),
            },
            "final_winner_score": {
                "mean": float(np.mean(final_scores)),
                "std": float(np.std(final_scores)),
                "min": float(np.min(final_scores)),
                "max": float(np.max(final_scores)),
            },
        }

    # Per-checkpoint score spread (how much disagreement is there?)
    for cp_idx in range(max((len(r["checkpoints"]) for r in records), default=0)):
        spreads = []
        for rec in records:
            if cp_idx < len(rec["checkpoints"]):
                scores = list(rec["checkpoints"][cp_idx]["all_scores"].values())
                if len(scores) > 1:
                    spreads.append(max(scores) - min(scores))

        if spreads:
            results["per_checkpoint_stats"].append(
                {
                    "checkpoint_idx": cp_idx,
                    "score_spread_mean": float(np.mean(spreads)),
                    "score_spread_std": float(np.std(spreads)),
                }
            )

    return results


def plot_analysis(records: List[Dict], results: Dict, output_prefix: str):
    """Generate analysis plots."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available, skipping plots")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # --- Panel 1: Agreement pie chart ---
    ax = axes[0]
    agree = results["first_checkpoint_agreement"]
    disagree = results["first_checkpoint_disagreement"]
    if agree + disagree > 0:
        ax.pie(
            [agree, disagree],
            labels=[f"Same\n({agree})", f"Different\n({disagree})"],
            colors=["#2ecc71", "#e74c3c"],
            autopct="%1.0f%%",
            startangle=90,
            textprops={"fontsize": 11},
        )
        ax.set_title(
            "First Checkpoint Argmax\nvs Final Winner",
            fontsize=12,
            fontweight="bold",
        )

    # --- Panel 2: Score distributions at first checkpoint ---
    ax = axes[1]
    all_cp1_scores = []
    for rec in records:
        if rec["checkpoints"]:
            scores = list(rec["checkpoints"][0]["all_scores"].values())
            all_cp1_scores.extend(scores)

    if all_cp1_scores:
        ax.hist(all_cp1_scores, bins=30, color="#3498db", alpha=0.7, edgecolor="white")
        ax.set_xlabel("Physics Head Score")
        ax.set_ylabel("Count")
        ax.set_title(
            "Score Distribution\n(First Checkpoint)", fontsize=12, fontweight="bold"
        )
        ax.axvline(x=0.5, color="gray", linestyle="--", alpha=0.5)

    # --- Panel 3: Score spread per checkpoint ---
    ax = axes[2]
    cp_stats = results.get("per_checkpoint_stats", [])
    if cp_stats:
        cp_indices = [s["checkpoint_idx"] for s in cp_stats]
        spreads = [s["score_spread_mean"] for s in cp_stats]
        spread_stds = [s["score_spread_std"] for s in cp_stats]

        # Use actual timestep labels if available
        timestep_labels = []
        for cp_idx in cp_indices:
            ts_values = set()
            for rec in records:
                if cp_idx < len(rec["checkpoints"]):
                    ts_values.add(rec["checkpoints"][cp_idx]["timestep"])
            if ts_values:
                timestep_labels.append(f"t≈{max(ts_values)}")
            else:
                timestep_labels.append(f"CP {cp_idx + 1}")

        x = range(len(cp_indices))
        ax.bar(x, spreads, yerr=spread_stds, capsize=5, color="#9b59b6", alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(timestep_labels)
        ax.set_ylabel("Score Spread (max - min)")
        ax.set_title(
            "Score Disagreement\nPer Checkpoint", fontsize=12, fontweight="bold"
        )

    plt.suptitle("Trajectory Selection Analysis", fontsize=14, fontweight="bold")
    plt.tight_layout()

    Path(output_prefix).parent.mkdir(parents=True, exist_ok=True)
    fig_path = f"{output_prefix}_trajectory_analysis.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    logger.info(f"Saved figure to {fig_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Parse generation logs and analyze trajectory selection"
    )
    parser.add_argument(
        "--logs",
        type=str,
        nargs="+",
        required=True,
        help="SLURM log file(s) to parse",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="figures/trajectory_analysis/traj",
        help="Output prefix for figures and results",
    )
    args = parser.parse_args()

    # Parse all log files
    all_records = []
    for log_path in args.logs:
        logger.info(f"Parsing {log_path}...")
        records = parse_log_file(log_path)
        logger.info(f"  Found {len(records)} video records")
        all_records.extend(records)

    logger.info(f"\nTotal records: {len(all_records)}")

    if not all_records:
        logger.error("No records found!")
        return

    # Analyze
    results = analyze_selection(all_records)

    # Print summary
    logger.info(f"\n{'=' * 60}")
    logger.info("TRAJECTORY SELECTION ANALYSIS")
    logger.info(f"{'=' * 60}")
    logger.info(f"Total videos: {results['total_videos']}")
    logger.info(
        f"Agreement (1st CP argmax = final): {results['first_checkpoint_agreement']}"
    )
    logger.info(
        f"Disagreement (different winner):   {results['first_checkpoint_disagreement']}"
    )

    if results["agreement_rate"] is not None:
        logger.info(f"Agreement rate: {results['agreement_rate']:.1%}")
        logger.info("")

        if results["agreement_rate"] > 0.9:
            logger.info("→ Multi-step pruning rarely changes outcome.")
            logger.info("  Single checkpoint might be sufficient.")
        elif results["agreement_rate"] > 0.7:
            logger.info("→ Multi-step pruning changes outcome ~20-30% of the time.")
            logger.info("  Progressive pruning provides meaningful refinement.")
        else:
            logger.info("→ Multi-step pruning frequently changes outcome.")
            logger.info("  Later checkpoints carry significant additional signal.")

    # Disagreement case details
    if results["disagreement_cases"]:
        logger.info(f"\nDisagreement cases (showing first 10):")
        for case in results["disagreement_cases"][:10]:
            logger.info(
                f"  Video {case['video_idx']}: "
                f"CP1 winner=traj{case['cp1_winner']} ({case['cp1_winner_score']:.3f}) → "
                f"Final=traj{case['final_winner']} ({case['final_winner_score']:.3f}), "
                f"gap at CP1={case['score_gap_at_cp1']:.3f}"
            )

    # Score distributions
    sd = results.get("score_distributions", {})
    if sd:
        logger.info(f"\nScore statistics:")
        for key, stats in sd.items():
            logger.info(
                f"  {key}: {stats['mean']:.3f} ± {stats['std']:.3f} "
                f"[{stats['min']:.3f}, {stats['max']:.3f}]"
            )

    # Per-checkpoint stats
    for cp in results.get("per_checkpoint_stats", []):
        logger.info(
            f"  Checkpoint {cp['checkpoint_idx']}: "
            f"spread={cp['score_spread_mean']:.3f} ± {cp['score_spread_std']:.3f}"
        )

    # Save results
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    results_path = f"{args.output}_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"\nSaved results to {results_path}")

    # Save full records for further analysis
    records_path = f"{args.output}_records.json"
    with open(records_path, "w") as f:
        json.dump(all_records, f, indent=2, default=str)
    logger.info(f"Saved records to {records_path}")

    # Plot
    plot_analysis(all_records, results, args.output)


if __name__ == "__main__":
    main()
