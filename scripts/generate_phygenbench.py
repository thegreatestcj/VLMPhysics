"""
PhyGenBench Video Generation Script
Generates videos for PhyGenBench prompts using CogVideoX-2B/5B

Features:
  - Range selection: --start 0 --end 50
  - Random sampling: --sample 20 --seed 42
  - Stratified sampling: --sample 20 --stratified (samples evenly from each category)

Usage:
    # Generate specific range
    python scripts/generate_phygenbench.py --model 2b --start 0 --end 10

    # Random sample 20 prompts
    python scripts/generate_phygenbench.py --model 2b --sample 20 --seed 42

    # Stratified sample (even distribution across categories)
    python scripts/generate_phygenbench.py --model 2b --sample 20 --stratified

    # Specific indices
    python scripts/generate_phygenbench.py --model 2b --indices 0,5,10,15,20
"""

import argparse
import json
import os
import random
import time
from collections import defaultdict
from pathlib import Path

import torch
from diffusers import CogVideoXPipeline
from diffusers.utils import export_to_video
from tqdm import tqdm


def load_prompts(prompts_file: str) -> list[dict]:
    """Load PhyGenBench prompts from JSON file."""
    with open(prompts_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Normalize format
    if isinstance(data, list):
        if isinstance(data[0], str):
            return [
                {"prompt": p, "index": i, "category": "unknown"}
                for i, p in enumerate(data)
            ]
        # Ensure index field exists
        for i, item in enumerate(data):
            if "index" not in item:
                item["index"] = i
        return data
    return data


def select_prompts(
    prompts: list[dict],
    start: int = None,
    end: int = None,
    sample_k: int = None,
    stratified: bool = False,
    indices: list[int] = None,
    seed: int = 42,
) -> list[dict]:
    """
    Select prompts based on criteria.

    Returns list of (original_index, prompt_dict) tuples.
    """
    random.seed(seed)

    # Option 1: Specific indices
    if indices is not None:
        selected = [(i, prompts[i]) for i in indices if i < len(prompts)]
        return selected

    # Option 2: Range selection
    if start is not None or end is not None:
        start = start or 0
        end = end or len(prompts)
        selected = [(i, prompts[i]) for i in range(start, min(end, len(prompts)))]
        return selected

    # Option 3: Random/Stratified sampling
    if sample_k is not None:
        if stratified:
            # Group by category
            by_category = defaultdict(list)
            for i, p in enumerate(prompts):
                cat = p.get("category", "unknown")
                by_category[cat].append((i, p))

            # Sample evenly from each category
            selected = []
            categories = list(by_category.keys())
            per_category = max(1, sample_k // len(categories))
            remainder = sample_k - per_category * len(categories)

            for cat in categories:
                items = by_category[cat]
                n = min(per_category, len(items))
                selected.extend(random.sample(items, n))

            # Add remainder from random categories
            all_remaining = [
                (i, p) for i, p in enumerate(prompts) if (i, p) not in selected
            ]
            if remainder > 0 and all_remaining:
                selected.extend(
                    random.sample(all_remaining, min(remainder, len(all_remaining)))
                )

            # Sort by original index for consistency
            selected.sort(key=lambda x: x[0])
            return selected[:sample_k]
        else:
            # Pure random sampling
            indices = random.sample(range(len(prompts)), min(sample_k, len(prompts)))
            indices.sort()
            return [(i, prompts[i]) for i in indices]

    # Default: all prompts
    return [(i, p) for i, p in enumerate(prompts)]


def print_selected_prompts(selected: list[tuple], verbose: bool = True):
    """Print information about selected prompts."""
    print(f"\n{'=' * 70}")
    print(f"SELECTED PROMPTS: {len(selected)} total")
    print(f"{'=' * 70}")

    # Group by category for summary
    by_category = defaultdict(list)
    for idx, p in selected:
        cat = p.get("category", "unknown")
        by_category[cat].append(idx)

    print("\nDistribution by category:")
    for cat, indices in sorted(by_category.items()):
        print(
            f"  {cat:15s}: {len(indices):3d} prompts (indices: {indices[:5]}{'...' if len(indices) > 5 else ''})"
        )

    if verbose:
        print(f"\n{'─' * 70}")
        print("Detailed prompt list:")
        print(f"{'─' * 70}")
        for idx, p in selected:
            prompt_text = p.get("prompt", p) if isinstance(p, dict) else p
            cat = p.get("category", "?") if isinstance(p, dict) else "?"
            law = p.get("physical_law", "") if isinstance(p, dict) else ""
            # Truncate long prompts
            if len(prompt_text) > 60:
                prompt_text = prompt_text[:57] + "..."
            print(f"  [{idx:3d}] [{cat:10s}] {prompt_text}")
            if law:
                print(f"        └─ Law: {law}")

    print(f"{'=' * 70}\n")


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


def main():
    parser = argparse.ArgumentParser(description="Generate PhyGenBench videos")
    parser.add_argument(
        "--model",
        type=str,
        default="2b",
        choices=["2b", "5b"],
        help="CogVideoX model size (2b or 5b)",
    )
    parser.add_argument(
        "--prompts-file",
        type=str,
        default="data/phygenbench/prompts.json",
        help="Path to PhyGenBench prompts.json",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/phygenbench",
        help="Output directory for generated videos",
    )

    # Selection options (mutually exclusive)
    select_group = parser.add_mutually_exclusive_group()
    select_group.add_argument(
        "--indices",
        type=str,
        default=None,
        help="Comma-separated list of indices to generate (e.g., '0,5,10,15')",
    )
    select_group.add_argument(
        "--sample", type=int, default=None, help="Randomly sample K prompts"
    )

    parser.add_argument(
        "--start",
        type=int,
        default=None,
        help="Start index (used with --end for range)",
    )
    parser.add_argument(
        "--end", type=int, default=None, help="End index (used with --start for range)"
    )
    parser.add_argument(
        "--stratified",
        action="store_true",
        help="Use stratified sampling (even distribution across categories)",
    )

    # Generation options
    parser.add_argument(
        "--num-frames", type=int, default=49, help="Number of frames per video"
    )
    parser.add_argument(
        "--num-steps", type=int, default=50, help="Number of inference steps"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for generation and sampling"
    )
    parser.add_argument(
        "--skip-existing", action="store_true", help="Skip if video already exists"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print selected prompts, don't generate",
    )
    parser.add_argument(
        "--quiet", action="store_true", help="Don't print detailed prompt list"
    )
    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load prompts
    prompts = load_prompts(args.prompts_file)
    print(f"Loaded {len(prompts)} prompts from {args.prompts_file}")

    # Parse indices if provided
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

    # Print selected prompts
    print_selected_prompts(selected, verbose=not args.quiet)

    # Dry run - just show what would be generated
    if args.dry_run:
        print("DRY RUN - No videos generated")
        return

    # Setup pipeline
    pipe = setup_pipeline(args.model)

    # Generate videos
    results = []
    total_time = 0

    for orig_idx, item in tqdm(selected, desc="Generating videos"):
        # PhyGenBench naming convention: output_video_{index+1}.mp4
        output_path = output_dir / f"output_video_{orig_idx + 1}.mp4"

        # Skip if exists
        if args.skip_existing and output_path.exists():
            print(f"Skipping {output_path} (already exists)")
            continue

        # Get prompt text
        prompt_text = item["prompt"] if isinstance(item, dict) else item
        category = (
            item.get("category", "unknown") if isinstance(item, dict) else "unknown"
        )

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
    log_path = output_dir / "generation_log.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model": f"CogVideoX-{args.model}",
                "selection": {
                    "method": "indices"
                    if indices
                    else ("sample" if args.sample else "range"),
                    "stratified": args.stratified,
                    "seed": args.seed,
                    "selected_indices": [idx for idx, _ in selected],
                },
                "total_prompts": len(selected),
                "successful": sum(1 for r in results if r["status"] == "success"),
                "total_time_seconds": total_time,
                "avg_time_per_video": total_time / max(1, len(results)),
                "results": results,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"\n{'=' * 60}")
    print(f"Generation complete!")
    print(f"  Total videos: {sum(1 for r in results if r['status'] == 'success')}")
    print(f"  Total time: {total_time / 60:.1f} min")
    print(f"  Log saved to: {log_path}")


if __name__ == "__main__":
    main()
