#!/usr/bin/env python3
"""
PhyGenBench Video Generation Script
Generates videos for PhyGenBench prompts using CogVideoX-2B/5B

Features:
  - Unified output directory structure (results/generation/phygenbench/...)
  - Range selection: --start 0 --end 50
  - Random sampling: --sample 20 --seed 42
  - Stratified sampling: --sample 20 --stratified

Output Structure:
    results/generation/phygenbench/{model}_{method}_{timestamp}/
    ├── config.json         # Generation configuration
    ├── log.json            # Generation log with timing info
    └── videos/             # Generated video files
        ├── output_video_1.mp4
        └── ...

Usage:
    # Generate baseline videos (range)
    python eval/generate_phygenbench.py --model 2b --start 0 --end 10

    # Generate with custom experiment name
    python eval/generate_phygenbench.py --model 2b --exp-name my_experiment --sample 20

    # Stratified sample (even distribution across categories)
    python eval/generate_phygenbench.py --model 2b --sample 20 --stratified

    # Continue from existing experiment directory
    python eval/generate_phygenbench.py --resume results/generation/phygenbench/cogvideox-2b_baseline_20260127_143022
"""

import argparse
import json
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
from diffusers import CogVideoXPipeline
from diffusers.utils import export_to_video
from tqdm import tqdm

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.paths import (
    ResultsManager,
    create_generation_manager,
    get_videos_from_generation,
)


# =============================================================================
# Prompt Loading and Selection
# =============================================================================


def load_prompts(prompts_file: str) -> List[Dict]:
    """Load PhyGenBench prompts from JSON file."""
    with open(prompts_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Normalize format
    if isinstance(data, list):
        if len(data) > 0 and isinstance(data[0], str):
            # Simple string list
            return [
                {"prompt": p, "index": i, "category": "unknown"}
                for i, p in enumerate(data)
            ]

        # Dict list - normalize field names
        normalized = []
        for i, item in enumerate(data):
            norm_item = {
                "prompt": item.get("prompt") or item.get("caption", ""),
                "category": item.get("category")
                or item.get("main_category", "unknown"),
                "sub_category": item.get("sub_category", ""),
                "physical_laws": item.get("physical_laws", ""),
                "index": item.get("index", i),
            }
            normalized.append(norm_item)
        return normalized

    return data


def select_prompts(
    prompts: List[Dict],
    start: Optional[int] = None,
    end: Optional[int] = None,
    sample_k: Optional[int] = None,
    stratified: bool = False,
    indices: Optional[List[int]] = None,
    seed: int = 42,
) -> List[Tuple[int, Dict]]:
    """
    Select prompts based on criteria.

    Returns list of (original_index, prompt_dict) tuples.
    """
    if indices is not None:
        # Specific indices
        return [(i, prompts[i]) for i in indices if i < len(prompts)]

    if sample_k is not None:
        random.seed(seed)

        if stratified:
            # Group by category
            by_category = defaultdict(list)
            for i, p in enumerate(prompts):
                cat = p.get("category", "unknown")
                by_category[cat].append((i, p))

            # Sample evenly from each category
            per_cat = max(1, sample_k // len(by_category))
            selected = []
            for cat, items in by_category.items():
                n = min(per_cat, len(items))
                selected.extend(random.sample(items, n))

            # Fill remainder if needed
            remaining = sample_k - len(selected)
            if remaining > 0:
                all_items = [(i, p) for i, p in enumerate(prompts)]
                available = [x for x in all_items if x not in selected]
                selected.extend(
                    random.sample(available, min(remaining, len(available)))
                )

            return sorted(selected, key=lambda x: x[0])
        else:
            # Random sample
            all_items = [(i, p) for i, p in enumerate(prompts)]
            return sorted(
                random.sample(all_items, min(sample_k, len(all_items))),
                key=lambda x: x[0],
            )

    # Range selection
    s = start or 0
    e = end or len(prompts)
    return [(i, prompts[i]) for i in range(s, min(e, len(prompts)))]


def print_selected_prompts(selected: List[Tuple[int, Dict]], verbose: bool = True):
    """Print summary of selected prompts."""
    # Count by category
    by_cat = defaultdict(int)
    for _, p in selected:
        by_cat[p.get("category", "unknown")] += 1

    print(f"\n{'=' * 70}")
    print(f"SELECTED PROMPTS: {len(selected)} total")
    print(f"{'=' * 70}")
    print("By category:")
    for cat, count in sorted(by_cat.items()):
        print(f"  {cat}: {count}")

    if verbose:
        print(f"\n{'─' * 70}")
        for idx, p in selected:
            prompt_text = p.get("prompt", "")[:60]
            cat = p.get("category", "?")
            print(f"  [{idx:3d}] [{cat:10s}] {prompt_text}...")

    print(f"{'=' * 70}\n")


# =============================================================================
# GPU Utilities
# =============================================================================


def get_gpu_info() -> Dict:
    """Get GPU device information for logging."""
    if not torch.cuda.is_available():
        return {"gpu_available": False}

    device_id = torch.cuda.current_device()
    return {
        "gpu_available": True,
        "device_id": device_id,
        "device_name": torch.cuda.get_device_name(device_id),
        "device_capability": torch.cuda.get_device_capability(device_id),
        "total_memory_gb": round(
            torch.cuda.get_device_properties(device_id).total_memory / 1e9, 2
        ),
        "cuda_version": torch.version.cuda,
        "pytorch_version": torch.__version__,
    }


# =============================================================================
# Video Generation
# =============================================================================


def setup_pipeline(model_size: str, device: str = "cuda"):
    """Initialize CogVideoX pipeline with memory optimization."""
    model_id = f"THUDM/CogVideoX-{model_size}"
    print(f"Loading {model_id}...")

    pipe = CogVideoXPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)

    # Memory optimization for 24GB GPU
    pipe.enable_model_cpu_offload()
    pipe.vae.enable_tiling()

    return pipe


def generate_video(
    pipe,
    prompt: str,
    output_path: str,
    num_frames: int = 49,
    num_steps: int = 50,
    guidance_scale: float = 6.0,
    seed: int = 42,
):
    """Generate a single video from prompt."""
    generator = torch.Generator(device="cuda").manual_seed(seed)

    video = pipe(
        prompt=prompt,
        num_videos_per_prompt=1,
        num_inference_steps=num_steps,
        num_frames=num_frames,
        guidance_scale=guidance_scale,
        generator=generator,
    ).frames[0]

    export_to_video(video, output_path, fps=8)
    return output_path


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Generate PhyGenBench videos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Output directory structure:
    results/generation/phygenbench/{model}_{method}_{timestamp}/
    ├── config.json
    ├── log.json
    └── videos/
        ├── output_video_1.mp4
        └── ...
        """,
    )

    # Model and data
    parser.add_argument(
        "--model",
        type=str,
        default="2b",
        choices=["2b", "5b"],
        help="CogVideoX model size",
    )
    parser.add_argument(
        "--prompts-file",
        type=str,
        default="data/phygenbench/prompts.json",
        help="Path to PhyGenBench prompts.json",
    )

    # Output configuration (new structure)
    parser.add_argument(
        "--exp-name",
        type=str,
        default="baseline",
        help="Experiment name (e.g., 'baseline', 'trajectory_pruning')",
    )
    parser.add_argument(
        "--results-base", type=str, default="results", help="Base directory for results"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume from existing experiment directory",
    )

    # Legacy output option (for backward compatibility)
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="[DEPRECATED] Use --exp-name instead. Direct output directory path.",
    )

    # Selection options
    select_group = parser.add_mutually_exclusive_group()
    select_group.add_argument(
        "--indices",
        type=str,
        default=None,
        help="Comma-separated list of indices (e.g., '0,5,10,15')",
    )
    select_group.add_argument(
        "--sample", type=int, default=None, help="Randomly sample K prompts"
    )
    parser.add_argument(
        "--start", type=int, default=None, help="Start index (with --end)"
    )
    parser.add_argument(
        "--end", type=int, default=None, help="End index (with --start)"
    )
    parser.add_argument(
        "--stratified",
        action="store_true",
        help="Use stratified sampling (even distribution across categories)",
    )

    # Generation options
    parser.add_argument("--num-frames", type=int, default=49, help="Frames per video")
    parser.add_argument("--num-steps", type=int, default=50, help="Inference steps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--skip-existing", action="store_true", help="Skip if exists")
    parser.add_argument("--dry-run", action="store_true", help="Only print selection")
    parser.add_argument("--quiet", action="store_true", help="Less verbose output")

    args = parser.parse_args()

    # Setup output directory
    if args.resume:
        # Resume from existing directory
        rm = ResultsManager.from_existing(args.resume)
        print(f"Resuming from: {rm.get_output_dir()}")
    elif args.output_dir:
        # Legacy: direct path (deprecated but supported)
        print(f"[WARNING] --output-dir is deprecated. Use --exp-name instead.")
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        videos_dir = output_dir
        rm = None
    else:
        # New unified structure
        model_name = f"cogvideox-{args.model}"
        rm = create_generation_manager(
            exp_name=args.exp_name,
            model=model_name,
            results_base=args.results_base,
            num_frames=args.num_frames,
            num_steps=args.num_steps,
            seed=args.seed,
        )
        print(f"Output directory: {rm.get_output_dir()}")

    # Load prompts
    prompts = load_prompts(args.prompts_file)
    print(f"Loaded {len(prompts)} prompts from {args.prompts_file}")

    # Parse indices
    indices = None
    if args.indices:
        indices = [int(x.strip()) for x in args.indices.split(",")]

    # Select prompts
    selected = select_prompts(
        prompts,
        start=args.start,
        end=args.end,
        sample_k=args.sample,
        stratified=args.stratified,
        indices=indices,
        seed=args.seed,
    )

    print_selected_prompts(selected, verbose=not args.quiet)

    # Dry run check
    if args.dry_run:
        print("DRY RUN - No videos generated")
        return

    # Save config
    if rm:
        rm.save_config(
            {
                "prompts_file": str(args.prompts_file),
                "selection": {
                    "method": "indices"
                    if indices
                    else ("sample" if args.sample else "range"),
                    "start": args.start,
                    "end": args.end,
                    "sample_k": args.sample,
                    "stratified": args.stratified,
                    "indices": indices,
                },
                "selected_count": len(selected),
                "selected_indices": [idx for idx, _ in selected],
            }
        )

    # Setup pipeline
    pipe = setup_pipeline(args.model)

    # Print GPU info
    gpu_info = get_gpu_info()
    print(f"\n{'─' * 60}")
    print(
        f"GPU: {gpu_info.get('device_name', 'N/A')} ({gpu_info.get('total_memory_gb', '?')} GB)"
    )
    print(
        f"CUDA: {gpu_info.get('cuda_version', 'N/A')}, PyTorch: {gpu_info.get('pytorch_version', 'N/A')}"
    )
    print(f"{'─' * 60}\n")

    # Generate videos
    results = []
    total_time = 0

    for orig_idx, item in tqdm(selected, desc="Generating videos"):
        # Get output path
        if rm:
            output_path = rm.get_video_path(orig_idx)
        else:
            output_path = videos_dir / f"output_video_{orig_idx + 1}.mp4"

        # Skip if exists
        if args.skip_existing and output_path.exists():
            print(f"Skipping {output_path} (already exists)")
            continue

        prompt_text = item.get("prompt", "")
        category = item.get("category", "unknown")

        print(f"\n[{orig_idx + 1}] [{category}] Generating: {prompt_text[:70]}...")

        try:
            torch.cuda.synchronize()
            start_time = time.time()

            generate_video(
                pipe,
                prompt=prompt_text,
                output_path=str(output_path),
                num_frames=args.num_frames,
                num_steps=args.num_steps,
                seed=args.seed,
            )

            torch.cuda.synchronize()
            elapsed = time.time() - start_time
            total_time += elapsed

            results.append(
                {
                    "index": orig_idx,
                    "category": category,
                    "prompt": prompt_text,
                    "output": str(output_path),
                    "time": elapsed,
                    "status": "success",
                }
            )
            print(f"  -> Saved to {output_path} ({elapsed:.1f}s)")

        except Exception as e:
            print(f"  -> ERROR: {e}")
            results.append(
                {
                    "index": orig_idx,
                    "category": category,
                    "prompt": prompt_text,
                    "output": None,
                    "error": str(e),
                    "status": "failed",
                }
            )

    # Save generation log
    log_data = {
        "model": f"CogVideoX-{args.model}",
        "gpu_info": gpu_info,
        "total_prompts": len(selected),
        "successful": sum(1 for r in results if r["status"] == "success"),
        "total_time_seconds": total_time,
        "avg_time_per_video": total_time / max(1, len(results)),
        "results": results,
    }

    if rm:
        log_path = rm.get_output_dir() / "log.json"
    else:
        log_path = output_dir / "log.json"

    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log_data, f, indent=2, ensure_ascii=False)

    # Print summary
    print(f"\n{'=' * 60}")
    print(f"Generation complete!")
    print(f"  GPU: {gpu_info.get('device_name', 'N/A')}")
    print(f"  Total videos: {sum(1 for r in results if r['status'] == 'success')}")
    print(f"  Total time: {total_time / 60:.1f} min")
    print(f"  Avg time/video: {total_time / max(1, len(results)):.1f}s")
    print(f"  Output: {rm.get_output_dir() if rm else output_dir}")
    print(f"  Log: {log_path}")


if __name__ == "__main__":
    main()
