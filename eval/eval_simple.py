#!/usr/bin/env python3
"""
VLM Physics Evaluation Script for PhyGenBench

Evaluates generated videos for physical plausibility using either:
- Local VLM (Qwen2.5-VL-7B) - free but slower
- GPT-4o API - faster but costs money

Usage:
    # Local VLM evaluation
    python eval_vlm_physics.py \
        --videos_dir outputs/phygenbench/baseline \
        --output results/eval/baseline_physics.json \
        --vlm local

    # GPT-4o evaluation
    python eval_vlm_physics.py \
        --videos_dir outputs/phygenbench/physics \
        --output results/eval/physics_eval.json \
        --vlm gpt4o

    # Compare two methods
    python eval_vlm_physics.py \
        --videos_dir outputs/phygenbench/baseline outputs/phygenbench/physics \
        --output results/eval/comparison.json \
        --vlm local
"""

import os
import json
import argparse
import base64
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from tqdm import tqdm
import cv2
import numpy as np

# ============================================================================
# Prompts for Physics Evaluation
# ============================================================================

PHYSICS_EVAL_PROMPT = """Analyze this video for physical plausibility.

The video was generated from this prompt: "{caption}"

Evaluate whether the video follows real-world physics. Consider:
1. Gravity: Do objects fall/move correctly under gravity?
2. Collisions: Do objects interact realistically when they touch?
3. Conservation: Do objects maintain consistent shape/mass/volume?
4. Motion: Are velocities and accelerations natural?
5. Material properties: Do objects behave according to their material (rigid, fluid, soft)?

Answer with ONLY a JSON object (no markdown, no explanation):
{{"physics_score": 0 or 1, "reason": "brief explanation"}}

Where:
- physics_score=1: Video follows physical laws reasonably well
- physics_score=0: Video contains obvious physics violations
"""

# ============================================================================
# Video Processing Utilities
# ============================================================================


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


def frames_to_base64(frames: List[np.ndarray], max_size: int = 512) -> List[str]:
    """Convert frames to base64 strings for API calls."""
    base64_frames = []
    for frame in frames:
        h, w = frame.shape[:2]
        if max(h, w) > max_size:
            scale = max_size / max(h, w)
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        _, buffer = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
        base64_frames.append(base64.b64encode(buffer).decode("utf-8"))

    return base64_frames


# ============================================================================
# Local VLM Evaluator (Qwen2.5-VL-7B)
# ============================================================================


class LocalVLMEvaluator:
    """Evaluate physics using local Qwen2.5-VL-7B model."""

    def __init__(self, model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct"):
        print(f"Loading {model_name}...")

        import torch
        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=torch.float16, device_map="cuda"
        )
        self.processor = AutoProcessor.from_pretrained(model_name)
        print("Model loaded!")

    def evaluate(self, video_path: str, caption: str) -> Dict:
        """Evaluate a single video."""
        from qwen_vl_utils import process_vision_info

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": video_path,
                        "nframes": 8,  # Sample 8 frames
                    },
                    {
                        "type": "text",
                        "text": PHYSICS_EVAL_PROMPT.format(caption=caption),
                    },
                ],
            }
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt",
        ).to("cuda")

        outputs = self.model.generate(**inputs, max_new_tokens=200)
        response = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]

        # Parse response
        return self._parse_response(response)

    def _parse_response(self, response: str) -> Dict:
        """Parse VLM response to extract score."""
        try:
            # Find JSON in response
            start = response.find("{")
            end = response.rfind("}") + 1
            if start != -1 and end > start:
                json_str = response[start:end]
                result = json.loads(json_str)
                return {
                    "physics_score": int(result.get("physics_score", 0)),
                    "reason": result.get("reason", ""),
                    "raw_response": response,
                }
        except:
            pass

        # Fallback: look for yes/no
        response_lower = response.lower()
        if (
            'physics_score": 1' in response_lower
            or 'physics_score":1' in response_lower
        ):
            score = 1
        elif (
            'physics_score": 0' in response_lower
            or 'physics_score":0' in response_lower
        ):
            score = 0
        else:
            score = 1 if "yes" in response_lower else 0

        return {
            "physics_score": score,
            "reason": "parsed from response",
            "raw_response": response,
        }


# ============================================================================
# GPT-4o Evaluator
# ============================================================================


class GPT4oEvaluator:
    """Evaluate physics using OpenAI GPT-4o API."""

    def __init__(self):
        from openai import OpenAI

        self.client = OpenAI()  # Uses OPENAI_API_KEY env var
        self.model = "gpt-4o"
        print(f"Using {self.model} API")

    def evaluate(self, video_path: str, caption: str) -> Dict:
        """Evaluate a single video."""
        # Extract and encode frames
        frames = extract_frames(video_path, num_frames=8)
        base64_frames = frames_to_base64(frames)

        # Build message with images
        content = []
        for i, b64 in enumerate(base64_frames):
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{b64}",
                        "detail": "low",  # Save tokens
                    },
                }
            )

        content.append(
            {
                "type": "text",
                "text": PHYSICS_EVAL_PROMPT.format(caption=caption),
            }
        )

        # API call
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": content}],
            max_tokens=200,
        )

        response_text = response.choices[0].message.content

        # Parse response
        return self._parse_response(response_text)

    def _parse_response(self, response: str) -> Dict:
        """Parse GPT-4o response."""
        try:
            start = response.find("{")
            end = response.rfind("}") + 1
            if start != -1 and end > start:
                json_str = response[start:end]
                result = json.loads(json_str)
                return {
                    "physics_score": int(result.get("physics_score", 0)),
                    "reason": result.get("reason", ""),
                    "raw_response": response,
                }
        except:
            pass

        # Fallback
        score = (
            1
            if 'physics_score": 1' in response or 'physics_score":1' in response
            else 0
        )
        return {
            "physics_score": score,
            "reason": "parsed from response",
            "raw_response": response,
        }


# ============================================================================
# Main Evaluation Functions
# ============================================================================


def load_prompts(prompts_file: str = "data/phygenbench/prompts.json") -> List[Dict]:
    """Load PhyGenBench prompts."""
    with open(prompts_file, "r") as f:
        prompts = json.load(f)

    # Handle different formats
    if isinstance(prompts, list) and len(prompts) > 0:
        if isinstance(prompts[0], str):
            return [{"index": i, "prompt": p} for i, p in enumerate(prompts)]
        return prompts
    return prompts


def evaluate_folder(
    videos_dir: str,
    evaluator,
    prompts: List[Dict],
    output_file: Optional[str] = None,
) -> Dict:
    """Evaluate all videos in a folder."""
    videos_dir = Path(videos_dir)
    results = []

    # Find all videos
    video_files = sorted(videos_dir.glob("output_video_*.mp4"))
    print(f"Found {len(video_files)} videos in {videos_dir}")

    total_time = 0

    for video_file in tqdm(video_files, desc=f"Evaluating {videos_dir.name}"):
        # Extract index from filename (output_video_1.mp4 -> index 0)
        try:
            idx = int(video_file.stem.split("_")[-1]) - 1  # 1-indexed to 0-indexed
        except:
            continue

        if idx >= len(prompts):
            print(f"Warning: {video_file.name} index {idx} out of range")
            continue

        caption = prompts[idx].get("prompt", "")

        start_time = time.time()
        try:
            result = evaluator.evaluate(str(video_file), caption)
            result["video_file"] = video_file.name
            result["index"] = idx
            result["caption"] = caption
            results.append(result)
        except Exception as e:
            print(f"Error evaluating {video_file.name}: {e}")
            results.append(
                {
                    "video_file": video_file.name,
                    "index": idx,
                    "caption": caption,
                    "physics_score": -1,  # Error marker
                    "error": str(e),
                }
            )

        total_time += time.time() - start_time

    # Compute summary
    valid_results = [r for r in results if r.get("physics_score", -1) >= 0]
    if valid_results:
        physics_scores = [r["physics_score"] for r in valid_results]
        summary = {
            "folder": str(videos_dir),
            "total_videos": len(video_files),
            "evaluated": len(valid_results),
            "errors": len(results) - len(valid_results),
            "physics_pass_rate": sum(physics_scores) / len(physics_scores),
            "physics_pass_count": sum(physics_scores),
            "avg_time_per_video": total_time / len(valid_results),
            "total_time": total_time,
        }
    else:
        summary = {
            "folder": str(videos_dir),
            "error": "No valid results",
        }

    output = {
        "summary": summary,
        "results": results,
    }

    # Save results
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"Results saved to {output_file}")

    return output


def compare_methods(
    baseline_results: Dict,
    physics_results: Dict,
) -> Dict:
    """Compare baseline vs physics-guided results."""
    baseline_pass = baseline_results["summary"]["physics_pass_rate"]
    physics_pass = physics_results["summary"]["physics_pass_rate"]

    improvement = physics_pass - baseline_pass
    relative_improvement = improvement / baseline_pass * 100 if baseline_pass > 0 else 0

    return {
        "baseline_pass_rate": baseline_pass,
        "physics_pass_rate": physics_pass,
        "absolute_improvement": improvement,
        "relative_improvement_pct": relative_improvement,
    }


# ============================================================================
# Main
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate generated videos for physics plausibility",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Local VLM (free, ~5s/video)
    python eval_vlm_physics.py --videos_dir outputs/phygenbench/baseline --vlm local

    # GPT-4o (paid, ~2s/video)
    python eval_vlm_physics.py --videos_dir outputs/phygenbench/physics --vlm gpt4o

    # Compare two methods
    python eval_vlm_physics.py \\
        --videos_dir outputs/phygenbench/baseline outputs/phygenbench/physics \\
        --vlm local
        """,
    )

    parser.add_argument(
        "--videos_dir",
        type=str,
        nargs="+",
        required=True,
        help="Directory(ies) containing generated videos",
    )
    parser.add_argument(
        "--prompts_file",
        type=str,
        default="data/phygenbench/prompts.json",
        help="Path to prompts.json",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file for results",
    )
    parser.add_argument(
        "--vlm",
        type=str,
        choices=["local", "gpt4o"],
        default="local",
        help="VLM to use: 'local' (Qwen2.5-VL) or 'gpt4o'",
    )
    parser.add_argument(
        "--max_videos",
        type=int,
        default=None,
        help="Maximum videos to evaluate (for testing)",
    )

    args = parser.parse_args()

    # Load prompts
    prompts = load_prompts(args.prompts_file)
    print(f"Loaded {len(prompts)} prompts")

    # Create evaluator
    if args.vlm == "local":
        evaluator = LocalVLMEvaluator()
    else:
        evaluator = GPT4oEvaluator()

    # Evaluate each folder
    all_results = {}
    for videos_dir in args.videos_dir:
        folder_name = Path(videos_dir).name

        output_file = None
        if args.output:
            if len(args.videos_dir) == 1:
                output_file = args.output
            else:
                output_file = args.output.replace(".json", f"_{folder_name}.json")

        results = evaluate_folder(
            videos_dir=videos_dir,
            evaluator=evaluator,
            prompts=prompts,
            output_file=output_file,
        )

        all_results[folder_name] = results

        # Print summary
        print(f"\n{'=' * 50}")
        print(f"Results for {folder_name}:")
        print(f"{'=' * 50}")
        summary = results["summary"]
        print(
            f"  Evaluated: {summary.get('evaluated', 0)}/{summary.get('total_videos', 0)}"
        )
        print(f"  Physics Pass Rate: {summary.get('physics_pass_rate', 0) * 100:.1f}%")
        print(f"  Avg Time/Video: {summary.get('avg_time_per_video', 0):.2f}s")

    # Compare if multiple folders
    if len(args.videos_dir) == 2:
        names = list(all_results.keys())
        comparison = compare_methods(all_results[names[0]], all_results[names[1]])

        print(f"\n{'=' * 50}")
        print("Comparison:")
        print(f"{'=' * 50}")
        print(f"  {names[0]}: {comparison['baseline_pass_rate'] * 100:.1f}%")
        print(f"  {names[1]}: {comparison['physics_pass_rate'] * 100:.1f}%")
        print(
            f"  Improvement: +{comparison['absolute_improvement'] * 100:.1f}% ({comparison['relative_improvement_pct']:.1f}% relative)"
        )


if __name__ == "__main__":
    main()
