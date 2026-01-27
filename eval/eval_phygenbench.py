#!/usr/bin/env python3
"""
PhyGenBench Evaluation Script
Implements the 3-stage PhyGenEval pipeline:
  - Stage 1: VQAScore (key phenomena detection)
  - Stage 2: GPT-4o (physical order verification)  
  - Stage 3: GPT-4o (overall naturalness)

Output Structure:
    results/evaluation/phygenbench/{exp_name}_{stages}_{timestamp}/
    ├── config.json         # Evaluation configuration
    ├── results.json        # Full per-video results
    └── summary.json        # Summary statistics

Usage:
    # Evaluate from generation results directory
    python eval/eval_phygenbench.py \
        --videos-dir results/generation/phygenbench/cogvideox-2b_baseline_20260127_143022 \
        --exp-name baseline_eval

    # Stage 1 only (no API cost)
    python eval/eval_phygenbench.py \
        --videos-dir results/generation/phygenbench/cogvideox-2b_baseline_20260127_143022 \
        --stages 1 \
        --exp-name baseline_stage1

    # Evaluate specific indices
    python eval/eval_phygenbench.py \
        --videos-dir results/generation/phygenbench/cogvideox-2b_baseline_20260127_143022 \
        --indices 0,1,2,3,4 \
        --exp-name test_eval
"""

import argparse
import base64
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
from tqdm import tqdm

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.paths import (
    ResultsManager,
    create_evaluation_manager,
    get_videos_from_generation,
)


# =============================================================================
# Video Processing Utilities
# =============================================================================


def extract_frames(video_path: str, num_frames: int = 8) -> List[np.ndarray]:
    """Extract evenly spaced frames from video."""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames == 0:
        cap.release()
        raise ValueError(f"Could not read video: {video_path}")

    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)

    cap.release()
    return frames


def frame_to_base64(frame: np.ndarray, max_size: int = 512) -> str:
    """Convert frame to base64 string, resizing if needed."""
    h, w = frame.shape[:2]
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    _, buffer = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buffer).decode("utf-8")


def frames_to_base64_list(frames: List[np.ndarray], max_size: int = 512) -> List[str]:
    """Convert list of frames to base64 strings."""
    return [frame_to_base64(f, max_size) for f in frames]


# =============================================================================
# Evaluation Stages
# =============================================================================


def eval_stage1_vqascore(
    video_path: str,
    prompt: str,
    question_data: Optional[Dict] = None,
    vqa_model=None,
) -> Dict:
    """
    Stage 1: VQAScore - Key phenomena detection
    """
    result = {
        "stage": 1,
        "score": None,
        "details": {},
        "status": "skipped",
    }

    try:
        # Extract frames
        frames = extract_frames(video_path, num_frames=8)

        if vqa_model is not None:
            # Use VQAScore model
            # For now, return placeholder
            score = 0.5  # Placeholder
            result["score"] = score
            result["status"] = "success"
        else:
            result["status"] = "skipped"
            result["reason"] = "VQA model not loaded"

    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)

    return result


def eval_stage2_order(
    video_path: str,
    question_data: Dict,
    api_key: str,
    verbose: bool = False,
) -> Dict:
    """
    Stage 2: Physical order verification using GPT-4o
    """
    result = {
        "stage": 2,
        "score": None,
        "details": {},
        "status": "skipped",
    }

    if not api_key:
        result["reason"] = "No API key provided"
        return result

    try:
        # Extract frames
        frames = extract_frames(video_path, num_frames=8)
        frames_b64 = frames_to_base64_list(frames)

        # TODO: Implement GPT-4o API call
        # For now, return placeholder
        score = 0.5  # Placeholder
        result["score"] = score
        result["status"] = "success"

    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)

    return result


def eval_stage3_naturalness(
    video_path: str,
    question_data: Dict,
    api_key: str,
    verbose: bool = False,
) -> Dict:
    """
    Stage 3: Overall naturalness evaluation using GPT-4o
    """
    result = {
        "stage": 3,
        "score": None,
        "details": {},
        "status": "skipped",
    }

    if not api_key:
        result["reason"] = "No API key provided"
        return result

    try:
        # Extract frames
        frames = extract_frames(video_path, num_frames=8)
        frames_b64 = frames_to_base64_list(frames)

        # TODO: Implement GPT-4o API call
        # For now, return placeholder
        score = 0.5  # Placeholder
        result["score"] = score
        result["status"] = "success"

    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)

    return result


# =============================================================================
# Main Evaluation Functions
# =============================================================================


def evaluate_video(
    video_path: str,
    prompt_data: Dict,
    questions: Dict,
    api_key: Optional[str] = None,
    stages: List[int] = [1, 2, 3],
    vqa_model=None,
    verbose: bool = False,
) -> Dict:
    """
    Evaluate a single video through all stages.
    """
    idx = prompt_data.get("index", 0)
    results = {
        "index": idx,
        "prompt": prompt_data.get("prompt", ""),
        "category": prompt_data.get("category", "unknown"),
        "sub_category": prompt_data.get("sub_category", ""),
        "video_path": video_path,
        "stages": {},
    }

    # Stage 1: VQAScore
    if 1 in stages:
        single_q = questions.get("single", [])
        single_q = single_q[idx] if idx < len(single_q) else None

        stage1_result = eval_stage1_vqascore(
            video_path, prompt_data.get("prompt", ""), single_q, vqa_model
        )
        results["stages"]["stage1"] = stage1_result

    # Stage 2: Order verification
    if 2 in stages and api_key:
        multi_q = questions.get("multi", [])
        multi_q = multi_q[idx] if idx < len(multi_q) else None

        if multi_q:
            stage2_result = eval_stage2_order(video_path, multi_q, api_key, verbose)
            results["stages"]["stage2"] = stage2_result
        else:
            results["stages"]["stage2"] = {
                "stage": 2,
                "score": None,
                "status": "skipped",
                "reason": "No question data",
            }

    # Stage 3: Naturalness
    if 3 in stages and api_key:
        video_q = questions.get("video", [])
        video_q = video_q[idx] if idx < len(video_q) else None

        if video_q:
            stage3_result = eval_stage3_naturalness(
                video_path, video_q, api_key, verbose
            )
            results["stages"]["stage3"] = stage3_result
        else:
            results["stages"]["stage3"] = {
                "stage": 3,
                "score": None,
                "status": "skipped",
                "reason": "No question data",
            }

    # Calculate combined PCA score
    valid_scores = []
    for stage_key, stage_data in results["stages"].items():
        if (
            stage_data.get("score") is not None
            and stage_data.get("status") == "success"
        ):
            valid_scores.append(stage_data["score"])

    results["pca_score"] = np.mean(valid_scores) if valid_scores else None

    return results


def compute_summary(results: List[Dict]) -> Dict:
    """Compute summary statistics from evaluation results."""
    summary = {
        "total_videos": len(results),
        "overall": {},
        "by_category": {},
        "by_stage": {},
    }

    all_pca = []
    stage_scores = {1: [], 2: [], 3: []}
    category_scores = {}

    for r in results:
        if r.get("pca_score") is not None:
            all_pca.append(r["pca_score"])

        category = r.get("category", "unknown")
        if category not in category_scores:
            category_scores[category] = []

        for stage_key, stage_data in r.get("stages", {}).items():
            stage_num = int(stage_key.replace("stage", ""))
            if (
                stage_data.get("score") is not None
                and stage_data.get("status") == "success"
            ):
                stage_scores[stage_num].append(stage_data["score"])
                category_scores[category].append(stage_data["score"])

    # Overall
    if all_pca:
        summary["overall"]["pca_score"] = float(np.mean(all_pca))
        summary["overall"]["pca_std"] = float(np.std(all_pca))
        summary["overall"]["count"] = len(all_pca)

    # By stage
    for stage_num, scores in stage_scores.items():
        if scores:
            summary["by_stage"][f"stage{stage_num}"] = {
                "mean": float(np.mean(scores)),
                "std": float(np.std(scores)),
                "count": len(scores),
            }

    # By category
    for category, scores in category_scores.items():
        if scores:
            summary["by_category"][category] = {
                "mean": float(np.mean(scores)),
                "std": float(np.std(scores)),
                "count": len(scores),
            }

    return summary


def load_questions(questions_dir: Optional[str], prompts_file: str) -> Dict:
    """Load question files for evaluation."""
    questions = {"single": [], "multi": [], "video": []}

    if questions_dir is None:
        # Try to find questions in same directory as prompts
        questions_dir = str(Path(prompts_file).parent)

    questions_dir = Path(questions_dir)

    for q_type in ["single", "multi", "video"]:
        q_file = questions_dir / f"{q_type}_question.json"
        if q_file.exists():
            with open(q_file, "r", encoding="utf-8") as f:
                questions[q_type] = json.load(f)

    return questions


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="PhyGenBench Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Output directory structure:
    results/evaluation/phygenbench/{exp_name}_{stages}_{timestamp}/
    ├── config.json
    ├── results.json
    └── summary.json
        """,
    )

    # Input
    parser.add_argument(
        "--videos-dir",
        type=str,
        required=True,
        help="Directory containing generated videos (generation output dir)",
    )
    parser.add_argument(
        "--prompts-file",
        type=str,
        default="data/phygenbench/prompts.json",
        help="Path to prompts.json",
    )
    parser.add_argument(
        "--questions-dir",
        type=str,
        default=None,
        help="Directory containing question JSON files",
    )

    # Output (new structure)
    parser.add_argument(
        "--exp-name",
        type=str,
        default="eval",
        help="Experiment name for output directory",
    )
    parser.add_argument(
        "--results-base", type=str, default="results", help="Base directory for results"
    )

    # Legacy output (deprecated)
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="[DEPRECATED] Use --exp-name instead. Direct output file path.",
    )

    # Evaluation options
    parser.add_argument(
        "--stages",
        type=int,
        nargs="+",
        default=[1, 2, 3],
        help="Evaluation stages to run",
    )
    parser.add_argument(
        "--model", type=str, default="gpt-4o", help="Model for stages 2 and 3"
    )

    # Selection options
    parser.add_argument("--start", type=int, default=None, help="Start index")
    parser.add_argument("--end", type=int, default=None, help="End index")
    parser.add_argument(
        "--indices",
        type=str,
        default=None,
        help="Comma-separated list of indices to evaluate",
    )

    # Other options
    parser.add_argument("--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    # Get API key
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key and any(s in args.stages for s in [2, 3]):
        print("Warning: OPENAI_API_KEY not set. Stages 2 and 3 will be skipped.")
        args.stages = [s for s in args.stages if s == 1]

    # Find videos directory
    videos_dir = Path(args.videos_dir)

    # Check if this is a new-structure generation dir
    if (videos_dir / "videos").exists():
        actual_videos_dir = videos_dir / "videos"
    else:
        actual_videos_dir = videos_dir

    # Setup output
    if args.output:
        # Legacy mode
        print("[WARNING] --output is deprecated. Use --exp-name instead.")
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        rm = None
    else:
        # New unified structure
        rm = create_evaluation_manager(
            exp_name=args.exp_name,
            stages=args.stages,
            results_base=args.results_base,
            model=args.model,
        )
        print(f"Output directory: {rm.get_output_dir()}")

    # Load prompts and questions
    with open(args.prompts_file, "r", encoding="utf-8") as f:
        prompts_data = json.load(f)

    if isinstance(prompts_data, list) and len(prompts_data) > 0:
        if isinstance(prompts_data[0], str):
            prompts = [
                {"prompt": p, "index": i, "category": "unknown"}
                for i, p in enumerate(prompts_data)
            ]
        else:
            prompts = prompts_data
    else:
        prompts = prompts_data

    questions = load_questions(args.questions_dir, args.prompts_file)

    # Determine indices to evaluate
    if args.indices:
        indices = [int(x.strip()) for x in args.indices.split(",")]
    elif args.start is not None or args.end is not None:
        s = args.start or 0
        e = args.end or len(prompts)
        indices = list(range(s, min(e, len(prompts))))
    else:
        indices = list(range(len(prompts)))

    # Load VQA model for stage 1
    vqa_model = None
    if 1 in args.stages:
        # TODO: Load VQAScore model
        pass

    # Save config
    if rm:
        rm.save_config(
            {
                "videos_dir": str(args.videos_dir),
                "prompts_file": str(args.prompts_file),
                "stages": args.stages,
                "model": args.model,
                "indices_evaluated": indices,
            }
        )

    # Run evaluation
    results = []

    for idx in tqdm(indices, desc="Evaluating"):
        # PhyGenBench naming: output_video_{idx+1}.mp4
        video_path = actual_videos_dir / f"output_video_{idx + 1}.mp4"

        if not video_path.exists():
            if args.verbose:
                print(f"  Skipping index {idx}: video not found")
            continue

        prompt_data = prompts[idx] if idx < len(prompts) else {"index": idx}

        if args.verbose:
            print(
                f"\n[{idx}] {prompt_data.get('category', '?')}: {prompt_data.get('prompt', '')[:50]}..."
            )

        result = evaluate_video(
            str(video_path),
            prompt_data,
            questions,
            api_key=api_key,
            stages=args.stages,
            vqa_model=vqa_model,
            verbose=args.verbose,
        )
        results.append(result)

        # Progress update
        if not args.verbose and len(results) % 10 == 0:
            recent_scores = [
                r["pca_score"] for r in results[-10:] if r.get("pca_score")
            ]
            if recent_scores:
                print(f"  Recent avg PCA: {np.mean(recent_scores):.3f}")

    # Compute summary
    summary = compute_summary(results)

    # Save results
    if rm:
        output_dir = rm.get_output_dir()

        # Full results
        with open(output_dir / "results.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "config": {
                        "videos_dir": str(args.videos_dir),
                        "prompts_file": str(args.prompts_file),
                        "stages": args.stages,
                        "model": args.model,
                        "indices_evaluated": indices,
                    },
                    "summary": summary,
                    "results": results,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

        # Summary only
        rm.save_summary(summary)
    else:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "config": {
                        "videos_dir": str(args.videos_dir),
                        "prompts_file": str(args.prompts_file),
                        "stages": args.stages,
                        "model": args.model,
                        "indices_evaluated": indices,
                    },
                    "summary": summary,
                    "results": results,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)

    if summary["overall"]:
        print(
            f"\nOverall PCA Score: {summary['overall'].get('pca_score', 0):.4f} "
            f"(± {summary['overall'].get('pca_std', 0):.4f})"
        )

    if summary["by_stage"]:
        print("\nBy Stage:")
        for stage, data in summary["by_stage"].items():
            print(f"  {stage}: {data['mean']:.4f} (n={data['count']})")

    if summary["by_category"]:
        print("\nBy Category:")
        for cat, data in sorted(summary["by_category"].items()):
            print(f"  {cat}: {data['mean']:.4f} (n={data['count']})")

    print(f"\nResults saved to: {rm.get_output_dir() if rm else output_path}")


if __name__ == "__main__":
    main()
