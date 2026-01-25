"""
PhyGenBench Evaluation Script
Implements the 3-stage PhyGenEval pipeline:
  - Stage 1: VQAScore (key phenomena detection)
  - Stage 2: GPT-4o (physical order verification)  
  - Stage 3: GPT-4o (overall naturalness)

Usage:
    # Full evaluation with GPT-4o
    python scripts/eval_phygenbench.py \
        --videos-dir outputs/phygenbench/baseline \
        --prompts-file data/phygenbench/prompts.json \
        --questions-dir data/phygenbench \
        --output results/baseline_eval.json

    # Stage 1 only (no API cost)
    python scripts/eval_phygenbench.py \
        --videos-dir outputs/phygenbench/baseline \
        --prompts-file data/phygenbench/prompts.json \
        --stage 1 \
        --output results/baseline_stage1.json

    # Evaluate specific indices
    python scripts/eval_phygenbench.py \
        --videos-dir outputs/phygenbench/baseline \
        --prompts-file data/phygenbench/prompts.json \
        --questions-dir data/phygenbench \
        --indices 0,1,2,3,4 \
        --output results/test_eval.json
"""

import argparse
import base64
import json
import os
import re
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from tqdm import tqdm


# =============================================================================
# Video Processing Utilities
# =============================================================================


def extract_frames(video_path: str, num_frames: int = 8) -> list[np.ndarray]:
    """Extract evenly spaced frames from video."""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames == 0:
        cap.release()
        raise ValueError(f"Could not read video: {video_path}")

    # Get evenly spaced frame indices
    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            # Convert BGR to RGB
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

    # Convert RGB to BGR for cv2.imencode
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    _, buffer = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buffer).decode("utf-8")


def frames_to_base64_list(frames: list[np.ndarray], max_size: int = 512) -> list[str]:
    """Convert list of frames to base64 strings."""
    return [frame_to_base64(f, max_size) for f in frames]


# =============================================================================
# Data Loading
# =============================================================================


def load_prompts(prompts_file: str) -> list[dict]:
    """Load PhyGenBench prompts from JSON file."""
    with open(prompts_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        normalized = []
        for i, item in enumerate(data):
            if isinstance(item, str):
                norm_item = {"prompt": item, "index": i, "category": "unknown"}
            else:
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


def load_questions(questions_dir: str) -> dict:
    """Load PhyGenBench question files for evaluation."""
    questions = {}

    # Stage 1: single_question.json (for VQAScore)
    single_q_path = Path(questions_dir) / "single_question.json"
    if single_q_path.exists():
        with open(single_q_path, "r", encoding="utf-8") as f:
            questions["single"] = json.load(f)

    # Stage 2: multi_question.json (for GPT-4o order verification)
    multi_q_path = Path(questions_dir) / "multi_question.json"
    if multi_q_path.exists():
        with open(multi_q_path, "r", encoding="utf-8") as f:
            questions["multi"] = json.load(f)

    # Stage 3: video_question.json (for GPT-4o naturalness)
    video_q_path = Path(questions_dir) / "video_question.json"
    if video_q_path.exists():
        with open(video_q_path, "r", encoding="utf-8") as f:
            questions["video"] = json.load(f)

    return questions


# =============================================================================
# Stage 1: VQAScore Evaluation
# =============================================================================


def eval_stage1_vqascore(
    video_path: str,
    prompt: str,
    question: Optional[dict] = None,
    model=None,
) -> dict:
    """
    Stage 1: Evaluate using VQAScore (CLIP-FlanT5).
    Checks if key physical phenomena are present in the video.

    Returns dict with score and details.
    """
    try:
        # Extract frames
        frames = extract_frames(video_path, num_frames=8)

        if model is None:
            # Lazy load t2v-metrics
            import t2v_metrics

            model = t2v_metrics.VQAScore(model="clip-flant5-xxl")

        # If we have specific questions, use them
        if question and "question" in question:
            query = question["question"]
        else:
            # Default: check if the prompt content is present
            query = f"Does this video show: {prompt}? Answer yes or no."

        # VQAScore expects PIL images
        from PIL import Image

        pil_frames = [Image.fromarray(f) for f in frames]

        # Get score (average across frames)
        scores = []
        for frame in pil_frames:
            score = model(images=[frame], texts=[query])
            scores.append(float(score[0][0]))

        avg_score = np.mean(scores)

        return {
            "stage": 1,
            "method": "VQAScore",
            "score": avg_score,
            "query": query,
            "frame_scores": scores,
            "status": "success",
        }

    except Exception as e:
        return {
            "stage": 1,
            "method": "VQAScore",
            "score": 0.0,
            "error": str(e),
            "status": "failed",
        }


# =============================================================================
# Stage 2 & 3: GPT-4o Evaluation
# =============================================================================


def call_gpt4o_vision(
    frames_base64: list[str],
    prompt: str,
    api_key: str,
    max_retries: int = 3,
    model: str = "gpt-4o",
) -> str:
    """Call GPT-4o with video frames."""
    import openai

    client = openai.OpenAI(api_key=api_key)

    # Build message with frames
    content = []

    # Add instruction
    content.append({"type": "text", "text": prompt})

    # Add frames as images
    for i, b64 in enumerate(frames_base64):
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{b64}",
                    "detail": "low",  # Use low detail to save tokens
                },
            }
        )

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": content}],
                max_tokens=500,
                temperature=0.0,
            )
            return response.choices[0].message.content

        except openai.RateLimitError:
            wait_time = 2**attempt * 10  # Exponential backoff
            print(f"  Rate limited, waiting {wait_time}s...")
            time.sleep(wait_time)
        except Exception as e:
            if attempt == max_retries - 1:
                raise e
            time.sleep(5)

    raise Exception("Max retries exceeded")


def eval_stage2_order(
    video_path: str, question_data: dict, api_key: str, verbose: bool = False
) -> dict:
    """
    Stage 2: Verify physical event order using GPT-4o.

    question_data should contain:
    - question: The question about event order
    - options: Multiple choice options (A, B, C, D)
    - answer: Correct answer
    """
    try:
        frames = extract_frames(video_path, num_frames=8)
        frames_b64 = frames_to_base64_list(frames)

        # Build prompt
        question = question_data.get("question", "")
        options = question_data.get("options", [])

        prompt = f"""These 8 frames are extracted from a video in chronological order (frame 1 is earliest, frame 8 is latest).

{question}

Options:
"""
        for i, opt in enumerate(options):
            prompt += f"{chr(65 + i)}. {opt}\n"

        prompt += "\nAnswer with just the letter (A, B, C, or D)."

        if verbose:
            print(f"  Stage 2 Question: {question[:80]}...")

        response = call_gpt4o_vision(frames_b64, prompt, api_key)

        # Parse answer
        response_clean = response.strip().upper()
        predicted = response_clean[0] if response_clean else "X"
        correct_answer = question_data.get("answer", "A").strip().upper()

        is_correct = predicted == correct_answer[0]

        if verbose:
            print(
                f"  Stage 2 Response: {predicted} (correct: {correct_answer[0]}) {'✓' if is_correct else '✗'}"
            )

        return {
            "stage": 2,
            "method": "GPT-4o",
            "score": 1.0 if is_correct else 0.0,
            "predicted": predicted,
            "correct": correct_answer,
            "question": question,
            "response": response,
            "status": "success",
        }

    except Exception as e:
        return {
            "stage": 2,
            "method": "GPT-4o",
            "score": 0.0,
            "error": str(e),
            "status": "failed",
        }


def eval_stage3_naturalness(
    video_path: str, question_data: dict, api_key: str, verbose: bool = False
) -> dict:
    """
    Stage 3: Evaluate overall naturalness/physical plausibility using GPT-4o.

    question_data should contain:
    - question: Question about naturalness
    - (optional) options: If multiple choice
    """
    try:
        frames = extract_frames(video_path, num_frames=8)
        frames_b64 = frames_to_base64_list(frames)

        question = question_data.get("question", "")
        options = question_data.get("options", [])

        if options:
            # Multiple choice format
            prompt = f"""These 8 frames are extracted from a video in chronological order.

{question}

Options:
"""
            for i, opt in enumerate(options):
                prompt += f"{chr(65 + i)}. {opt}\n"
            prompt += "\nAnswer with just the letter."
        else:
            # Yes/No format
            prompt = f"""These 8 frames are extracted from a video in chronological order.

{question}

Answer with Yes or No, then briefly explain why."""

        if verbose:
            print(f"  Stage 3 Question: {question[:80]}...")

        response = call_gpt4o_vision(frames_b64, prompt, api_key)

        # Parse response
        response_lower = response.strip().lower()
        correct_answer = question_data.get("answer", "yes").strip().lower()

        if options:
            # Multiple choice
            predicted = response.strip().upper()[0] if response.strip() else "X"
            is_correct = predicted == correct_answer[0].upper()
        else:
            # Yes/No
            if response_lower.startswith("yes"):
                predicted = "yes"
            elif response_lower.startswith("no"):
                predicted = "no"
            else:
                predicted = "unclear"
            is_correct = predicted == correct_answer

        if verbose:
            print(
                f"  Stage 3 Response: {predicted} (correct: {correct_answer}) {'✓' if is_correct else '✗'}"
            )

        return {
            "stage": 3,
            "method": "GPT-4o",
            "score": 1.0 if is_correct else 0.0,
            "predicted": predicted,
            "correct": correct_answer,
            "question": question,
            "response": response,
            "status": "success",
        }

    except Exception as e:
        return {
            "stage": 3,
            "method": "GPT-4o",
            "score": 0.0,
            "error": str(e),
            "status": "failed",
        }


# =============================================================================
# Main Evaluation Pipeline
# =============================================================================


def evaluate_video(
    video_path: str,
    prompt_data: dict,
    questions: dict,
    api_key: Optional[str] = None,
    stages: list[int] = [1, 2, 3],
    vqa_model=None,
    verbose: bool = False,
) -> dict:
    """Evaluate a single video through specified stages."""

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
        single_q = None
        if "single" in questions and idx < len(questions["single"]):
            single_q = questions["single"][idx]

        stage1_result = eval_stage1_vqascore(
            video_path, prompt_data.get("prompt", ""), single_q, vqa_model
        )
        results["stages"]["stage1"] = stage1_result

    # Stage 2: Order verification (requires API key)
    if 2 in stages and api_key:
        multi_q = None
        if "multi" in questions and idx < len(questions["multi"]):
            multi_q = questions["multi"][idx]

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

    # Stage 3: Naturalness (requires API key)
    if 3 in stages and api_key:
        video_q = None
        if "video" in questions and idx < len(questions["video"]):
            video_q = questions["video"][idx]

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

    # Calculate combined score
    valid_scores = []
    for stage_key, stage_data in results["stages"].items():
        if (
            stage_data.get("score") is not None
            and stage_data.get("status") == "success"
        ):
            valid_scores.append(stage_data["score"])

    results["pca_score"] = np.mean(valid_scores) if valid_scores else None

    return results


def compute_summary(results: list[dict]) -> dict:
    """Compute summary statistics from evaluation results."""
    summary = {
        "total_videos": len(results),
        "overall": {},
        "by_category": {},
        "by_stage": {},
    }

    # Collect scores
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
        summary["overall"]["pca_score"] = np.mean(all_pca)
        summary["overall"]["pca_std"] = np.std(all_pca)
        summary["overall"]["count"] = len(all_pca)

    # By stage
    for stage_num, scores in stage_scores.items():
        if scores:
            summary["by_stage"][f"stage{stage_num}"] = {
                "mean": np.mean(scores),
                "std": np.std(scores),
                "count": len(scores),
            }

    # By category
    for category, scores in category_scores.items():
        if scores:
            summary["by_category"][category] = {
                "mean": np.mean(scores),
                "std": np.std(scores),
                "count": len(scores),
            }

    return summary


def main():
    parser = argparse.ArgumentParser(description="PhyGenBench Evaluation")
    parser.add_argument(
        "--videos-dir",
        type=str,
        required=True,
        help="Directory containing generated videos",
    )
    parser.add_argument(
        "--prompts-file", type=str, required=True, help="Path to prompts.json"
    )
    parser.add_argument(
        "--questions-dir",
        type=str,
        default=None,
        help="Directory containing question JSON files",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/eval.json",
        help="Output file for results",
    )

    # Selection options
    parser.add_argument(
        "--start", type=int, default=None, help="Start index (inclusive)"
    )
    parser.add_argument("--end", type=int, default=None, help="End index (exclusive)")
    parser.add_argument(
        "--indices",
        type=str,
        default=None,
        help="Comma-separated list of indices to evaluate",
    )
    parser.add_argument(
        "--sample", type=int, default=None, help="Randomly sample K videos"
    )

    # Stage selection
    parser.add_argument(
        "--stage",
        type=str,
        default="all",
        help="Stages to run: 'all', '1', '2', '3', or '1,2'",
    )

    # API configuration
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="OpenAI API key (or set OPENAI_API_KEY env var)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o",
        help="OpenAI model to use (default: gpt-4o)",
    )

    # Other options
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Print detailed output"
    )
    parser.add_argument(
        "--skip-stage1",
        action="store_true",
        help="Skip Stage 1 (VQAScore) - useful if not installed",
    )

    args = parser.parse_args()

    # Parse stages
    if args.stage == "all":
        stages = [1, 2, 3]
    else:
        stages = [int(s) for s in args.stage.split(",")]

    if args.skip_stage1 and 1 in stages:
        stages.remove(1)

    # Get API key
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key and (2 in stages or 3 in stages):
        print("Warning: No OpenAI API key provided. Stage 2 and 3 will be skipped.")
        print("Set OPENAI_API_KEY environment variable or use --api-key")
        stages = [s for s in stages if s == 1]

    # Load data
    print(f"Loading prompts from {args.prompts_file}...")
    prompts = load_prompts(args.prompts_file)
    print(f"Loaded {len(prompts)} prompts")

    questions = {}
    if args.questions_dir:
        print(f"Loading questions from {args.questions_dir}...")
        questions = load_questions(args.questions_dir)
        print(f"Loaded questions: {list(questions.keys())}")

    # Select indices
    if args.indices:
        indices = [int(i) for i in args.indices.split(",")]
    elif args.start is not None or args.end is not None:
        start = args.start or 0
        end = args.end or len(prompts)
        indices = list(range(start, min(end, len(prompts))))
    elif args.sample:
        import random

        indices = random.sample(range(len(prompts)), min(args.sample, len(prompts)))
    else:
        indices = list(range(len(prompts)))

    print(f"\nEvaluating {len(indices)} videos")
    print(f"Stages: {stages}")
    print(f"Output: {args.output}")
    print("-" * 60)

    # Initialize VQA model if needed
    vqa_model = None
    if 1 in stages:
        try:
            import t2v_metrics

            print("Loading VQAScore model...")
            vqa_model = t2v_metrics.VQAScore(model="clip-flant5-xxl")
        except ImportError:
            print("Warning: t2v-metrics not installed. Skipping Stage 1.")
            stages = [s for s in stages if s != 1]

    # Run evaluation
    results = []
    videos_dir = Path(args.videos_dir)

    for idx in tqdm(indices, desc="Evaluating"):
        # PhyGenBench naming: output_video_{idx+1}.mp4
        video_path = videos_dir / f"output_video_{idx + 1}.mp4"

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
            stages=stages,
            vqa_model=vqa_model,
            verbose=args.verbose,
        )
        results.append(result)

        # Brief progress update
        if not args.verbose and len(results) % 10 == 0:
            recent_scores = [
                r["pca_score"] for r in results[-10:] if r.get("pca_score")
            ]
            if recent_scores:
                print(f"  Recent avg PCA: {np.mean(recent_scores):.3f}")

    # Compute summary
    summary = compute_summary(results)

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_data = {
        "config": {
            "videos_dir": str(args.videos_dir),
            "prompts_file": str(args.prompts_file),
            "stages": stages,
            "model": args.model,
            "indices_evaluated": indices,
        },
        "summary": summary,
        "results": results,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

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
        for cat, data in summary["by_category"].items():
            print(f"  {cat}: {data['mean']:.4f} (n={data['count']})")

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
