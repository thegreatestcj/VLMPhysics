#!/usr/bin/env python3
"""
Complete PhyGenBench 3-Stage Evaluation Pipeline

This standalone script implements the full PhyGenEval evaluation framework:
- Stage 1: VQAScore (Key Physical Phenomena Detection)
- Stage 2: Multi-frame Physics Order Verification (CLIP + GPT-4o)
- Stage 3: Video-level Overall Naturalness (GPT-4o)
- Overall Score Calculation

Designed to evaluate outputs/phygenbench/baseline videos without external dependencies.

Usage:
    # Full 3-stage evaluation with GPT-4o
    python eval/eval_phygenbench.py --videos-dir outputs/phygenbench/baseline

    # Stage 1 only (VQAScore, no API cost)
    python eval/eval_phygenbench.py --videos-dir outputs/phygenbench/baseline --stage 1

    # Stages 2+3 only (GPT-4o)
    python eval/eval_phygenbench.py --videos-dir outputs/phygenbench/baseline --stage 2 3

    # Dry run (just count videos and verify setup)
    python eval/eval_phygenbench.py --videos-dir outputs/phygenbench/baseline --dry-run

Requirements:
    - t2v_metrics (for VQAScore): pip install t2v-metrics
    - openai (for GPT-4o): pip install openai
    - OPENAI_API_KEY environment variable

Cost estimate: ~$3-5 for 160 videos (Stages 2+3)
Time estimate: ~30-60 minutes for full evaluation
"""

import argparse
import base64
import json
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict

import cv2
import numpy as np
from tqdm import tqdm


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class EvalConfig:
    """Configuration for PhyGenBench evaluation."""

    # Paths
    videos_dir: str = "outputs/phygenbench/physics"
    prompts_file: str = "data/phygenbench/prompts.json"
    single_question_file: str = "data/phygenbench/single_question.json"
    multi_question_file: str = "data/phygenbench/multi_question.json"
    video_question_file: str = "data/phygenbench/video_question.json"
    output_dir: str = "results/evaluation/phygenbench"

    # Stages to run
    stages: List[int] = field(default_factory=lambda: [1, 2, 3])

    # Frame extraction
    num_frames_stage1: int = 8  # Frames for VQAScore
    num_frames_stage2: int = 4  # Frames for multi-image QA
    num_frames_stage3: int = 8  # Frames for video QA

    # VQAScore model
    vqascore_model: str = "clip-flant5-xxl"  # Options: clip-flant5-xxl, clip-flant5-xl

    # GPT-4o settings
    gpt_model: str = "gpt-4o"
    gpt_max_tokens: int = 200

    # Resume settings
    resume: bool = True

    # Testing
    max_videos: Optional[int] = None
    dry_run: bool = False


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
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)

    cap.release()
    return frames


def extract_keyframe_by_clip(
    video_path: str, retrieval_prompt: str, num_frames: int = 16
) -> np.ndarray:
    """
    Extract keyframe using CLIP similarity to retrieval prompt.
    Falls back to middle frame if CLIP fails.
    """
    frames = extract_frames(video_path, num_frames)

    if retrieval_prompt.lower() in ["middle frame", "last frame", "first frame"]:
        if "last" in retrieval_prompt.lower():
            return frames[-1]
        elif "first" in retrieval_prompt.lower():
            return frames[0]
        else:
            return frames[len(frames) // 2]

    # Try CLIP-based retrieval
    try:
        import torch
        import clip

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, preprocess = clip.load("ViT-B/32", device=device)

        # Encode text
        text = clip.tokenize([retrieval_prompt]).to(device)
        with torch.no_grad():
            text_features = model.encode_text(text)

        # Encode frames and find best match
        best_idx = 0
        best_sim = -float("inf")

        for i, frame in enumerate(frames):
            # Convert to PIL and preprocess
            from PIL import Image

            img = Image.fromarray(frame)
            img_input = preprocess(img).unsqueeze(0).to(device)

            with torch.no_grad():
                img_features = model.encode_image(img_input)
                sim = (img_features @ text_features.T).item()

            if sim > best_sim:
                best_sim = sim
                best_idx = i

        return frames[best_idx]

    except Exception as e:
        # Fallback to middle frame
        return frames[len(frames) // 2]


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


def frame_to_base64(frame: np.ndarray, max_size: int = 512) -> str:
    """Convert single frame to base64 string."""
    return frames_to_base64([frame], max_size)[0]


# =============================================================================
# Data Loading
# =============================================================================


def load_questions(config: EvalConfig) -> Dict[str, List[Dict]]:
    """Load all question files."""
    questions = {}

    # Load single questions (Stage 1)
    if os.path.exists(config.single_question_file):
        with open(config.single_question_file, "r") as f:
            questions["single"] = json.load(f)
    else:
        print(f"Warning: {config.single_question_file} not found")
        questions["single"] = []

    # Load multi questions (Stage 2)
    if os.path.exists(config.multi_question_file):
        with open(config.multi_question_file, "r") as f:
            questions["multi"] = json.load(f)
    else:
        print(f"Warning: {config.multi_question_file} not found")
        questions["multi"] = []

    # Load video questions (Stage 3)
    if os.path.exists(config.video_question_file):
        with open(config.video_question_file, "r") as f:
            questions["video"] = json.load(f)
    else:
        print(f"Warning: {config.video_question_file} not found")
        questions["video"] = []

    return questions


def get_video_files(
    videos_dir: str, max_videos: Optional[int] = None
) -> List[Tuple[int, str]]:
    """
    Get list of video files with their indices.

    Returns: List of (index, video_path) tuples sorted by index
    """
    videos = []
    videos_dir = Path(videos_dir)

    for f in videos_dir.glob("output_video_*.mp4"):
        try:
            # Extract index from filename (1-based in filename, 0-based for questions)
            idx = int(f.stem.split("_")[-1]) - 1
            videos.append((idx, str(f)))
        except:
            continue

    # Sort by index
    videos.sort(key=lambda x: x[0])

    if max_videos:
        videos = videos[:max_videos]

    return videos


# =============================================================================
# Stage 1: VQAScore
# =============================================================================


class VQAScoreEvaluator:
    """Stage 1: VQAScore for Key Physical Phenomena Detection."""

    def __init__(self, model_name: str = "clip-flant5-xxl"):
        self.model_name = model_name
        self.model = None

    def _load_model(self):
        """Lazy load VQAScore model."""
        if self.model is None:
            try:
                import t2v_metrics

                self.model = t2v_metrics.VQAScore(model=self.model_name)
                print(f"Loaded VQAScore model: {self.model_name}")
            except ImportError:
                print(
                    "Warning: t2v_metrics not installed. Install with: pip install t2v-metrics"
                )
                raise

    def evaluate_video(
        self, video_path: str, questions: List[Dict], num_frames: int = 8
    ) -> Dict:
        """
        Evaluate a video using VQAScore.

        Questions should have format:
        {
            "Retrieval Prompt": "...",
            "Statement": "...",
            "Antonym": "..."
        }
        """
        self._load_model()

        frames = extract_frames(video_path, num_frames)

        scores = []
        details = []

        for q in questions:
            retrieval_prompt = q.get("Retrieval Prompt", "Middle Frame")
            statement = q.get("Statement", "")
            antonym = q.get("Antonym", "")

            # Get keyframe
            if "last" in retrieval_prompt.lower():
                keyframe = frames[-1]
            elif "first" in retrieval_prompt.lower():
                keyframe = frames[0]
            else:
                keyframe = frames[len(frames) // 2]

            # Compute VQAScore
            from PIL import Image
            import tempfile

            img = Image.fromarray(keyframe)

            try:
                # VQAScore expects file paths, not PIL Images
                # Save to temp file
                with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                    img.save(tmp.name, "JPEG", quality=95)
                    tmp_path = tmp.name

                try:
                    score_pos = self.model([tmp_path], [statement]).item()
                    score_neg = (
                        self.model([tmp_path], [antonym]).item() if antonym else 0
                    )
                finally:
                    # Clean up temp file
                    import os

                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)

                # Binary decision: positive statement should score higher
                binary_score = 1 if score_pos > score_neg else 0

                scores.append(binary_score)
                details.append(
                    {
                        "statement": statement,
                        "score_pos": score_pos,
                        "score_neg": score_neg,
                        "binary": binary_score,
                    }
                )
            except Exception as e:
                print(f"VQAScore error: {e}")
                scores.append(0)
                details.append({"error": str(e)})

        # Stage 1 score: proportion of correct answers
        final_score = sum(scores) / len(scores) if scores else 0

        return {
            "stage1_score": final_score,
            "num_questions": len(questions),
            "correct": sum(scores),
            "details": details,
        }


# =============================================================================
# Stage 2: Multi-frame Physics Order Verification (GPT-4o)
# =============================================================================


class MultiFrameEvaluator:
    """Stage 2: Multi-frame Physics Order Verification using GPT-4o."""

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o"):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.model = model
        self.client = None

    def _init_client(self):
        """Initialize OpenAI client."""
        if self.client is None:
            from openai import OpenAI

            self.client = OpenAI(api_key=self.api_key)

    def evaluate_video(
        self, video_path: str, multi_question: Dict, num_frames: int = 4
    ) -> Dict:
        """
        Evaluate using multi-frame comparison.

        multi_question format:
        {
            "Retrieval Prompt": "...",
            "Description1": "...",  # Wrong/before
            "Description2": "..."   # Correct/after
        }
        """
        self._init_client()

        retrieval_prompt = multi_question.get("Retrieval Prompt", "Middle Frame")
        desc1 = multi_question.get("Description1", "")
        desc2 = multi_question.get("Description2", "")

        # Extract frames around keyframe
        frames = extract_frames(video_path, num_frames)
        base64_frames = frames_to_base64(frames)

        # Build prompt
        prompt = f"""You are evaluating a physics video for temporal order verification.

Look at these {num_frames} frames from a video and determine which description better matches what happens:

Description A: {desc1}
Description B: {desc2}

Based on the temporal sequence shown in the frames, which description better matches the physics shown?
Answer with ONLY "A" or "B"."""

        # Build message with images
        content = []
        for i, b64 in enumerate(base64_frames):
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{b64}",
                        "detail": "low",
                    },
                }
            )
        content.append({"type": "text", "text": prompt})

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": content}],
                max_tokens=50,
            )

            answer = response.choices[0].message.content.strip().upper()

            # Description2 is typically the correct physics outcome
            # Score 1 if model chooses B (correct physics)
            score = 1 if "B" in answer else 0

            return {
                "stage2_score": score,
                "answer": answer,
                "desc1": desc1,
                "desc2": desc2,
            }

        except Exception as e:
            print(f"GPT-4o error (Stage 2): {e}")
            return {"stage2_score": 0, "error": str(e)}


# =============================================================================
# Stage 3: Video-level Overall Naturalness (GPT-4o)
# =============================================================================


class VideoNaturalnessEvaluator:
    """Stage 3: Overall Naturalness Evaluation using GPT-4o."""

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o"):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.model = model
        self.client = None

    def _init_client(self):
        """Initialize OpenAI client."""
        if self.client is None:
            from openai import OpenAI

            self.client = OpenAI(api_key=self.api_key)

    def evaluate_video(
        self, video_path: str, video_question: Dict, caption: str, num_frames: int = 8
    ) -> Dict:
        """
        Evaluate video naturalness on 4-level scale.

        video_question format:
        {
            "Description1": "...",  # Most unrealistic (score 0)
            "Description2": "...",  # Unrealistic (score 0.33)
            "Description3": "...",  # Slightly unrealistic (score 0.67)
            "Description4": "..."   # Realistic (score 1)
        }
        """
        self._init_client()

        desc1 = video_question.get("Description1", "")
        desc2 = video_question.get("Description2", "")
        desc3 = video_question.get("Description3", "")
        desc4 = video_question.get("Description4", "")

        frames = extract_frames(video_path, num_frames)
        base64_frames = frames_to_base64(frames)

        prompt = f"""You are evaluating a physics video for overall naturalness and physical correctness.

Video prompt: "{caption}"

Look at these {num_frames} frames and choose which description best matches the video:

1 (Completely unrealistic): {desc1}

2 (Mostly unrealistic): {desc2}

3 (Slightly unrealistic): {desc3}

4 (Realistic): {desc4}

Based on how well the video follows real-world physics, answer with ONLY a number: 1, 2, 3, or 4."""

        content = []
        for b64 in base64_frames:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{b64}",
                        "detail": "low",
                    },
                }
            )
        content.append({"type": "text", "text": prompt})

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": content}],
                max_tokens=50,
            )

            answer = response.choices[0].message.content.strip()

            # Parse level (1-4) and convert to score (0, 0.33, 0.67, 1)
            level = 1
            for ch in answer:
                if ch.isdigit() and 1 <= int(ch) <= 4:
                    level = int(ch)
                    break

            score_map = {1: 0, 2: 0.33, 3: 0.67, 4: 1}
            score = score_map.get(level, 0)

            return {"stage3_score": score, "level": level, "answer": answer}

        except Exception as e:
            print(f"GPT-4o error (Stage 3): {e}")
            return {"stage3_score": 0, "error": str(e)}


# =============================================================================
# Overall Score Calculation
# =============================================================================


def compute_overall_score(stage1: float, stage2: float, stage3: float) -> float:
    """
    Compute overall PhyGenBench score.

    According to PhyGenEval paper:
    - Score = Stage1 * Stage2 * Stage3
    - Each stage contributes multiplicatively
    """
    return stage1 * stage2 * stage3


# =============================================================================
# Main Evaluation Pipeline
# =============================================================================


class PhyGenBenchEvaluator:
    """Complete PhyGenBench evaluation pipeline."""

    def __init__(self, config: EvalConfig):
        self.config = config
        self.questions = load_questions(config)

        # Initialize evaluators lazily
        self._vqascore_evaluator = None
        self._multiframe_evaluator = None
        self._naturalness_evaluator = None

    @property
    def vqascore_evaluator(self):
        if self._vqascore_evaluator is None:
            self._vqascore_evaluator = VQAScoreEvaluator(self.config.vqascore_model)
        return self._vqascore_evaluator

    @property
    def multiframe_evaluator(self):
        if self._multiframe_evaluator is None:
            self._multiframe_evaluator = MultiFrameEvaluator(
                model=self.config.gpt_model
            )
        return self._multiframe_evaluator

    @property
    def naturalness_evaluator(self):
        if self._naturalness_evaluator is None:
            self._naturalness_evaluator = VideoNaturalnessEvaluator(
                model=self.config.gpt_model
            )
        return self._naturalness_evaluator

    def evaluate_single_video(self, video_idx: int, video_path: str) -> Dict:
        """Evaluate a single video through all configured stages."""
        result = {
            "index": video_idx,
            "video_path": video_path,
            "stage1_score": None,
            "stage2_score": None,
            "stage3_score": None,
            "overall_score": None,
        }

        # Get questions for this video index
        single_q = (
            self.questions["single"][video_idx]
            if video_idx < len(self.questions["single"])
            else None
        )
        multi_q = (
            self.questions["multi"][video_idx]
            if video_idx < len(self.questions["multi"])
            else None
        )
        video_q = (
            self.questions["video"][video_idx]
            if video_idx < len(self.questions["video"])
            else None
        )

        caption = single_q.get("caption", "") if single_q else ""

        # Stage 1: VQAScore
        if 1 in self.config.stages and single_q:
            try:
                vqa_questions = single_q.get(
                    "vqa_question", single_q.get("singleimage_question", [])
                )
                stage1_result = self.vqascore_evaluator.evaluate_video(
                    video_path, vqa_questions, self.config.num_frames_stage1
                )
                result["stage1_score"] = stage1_result["stage1_score"]
                result["stage1_details"] = stage1_result
            except Exception as e:
                print(f"Stage 1 error for video {video_idx}: {e}")
                result["stage1_score"] = 0
                result["stage1_error"] = str(e)

        # Stage 2: Multi-frame QA
        if 2 in self.config.stages and multi_q:
            try:
                multi_question = multi_q.get("multiimage_question", {})
                stage2_result = self.multiframe_evaluator.evaluate_video(
                    video_path, multi_question, self.config.num_frames_stage2
                )
                result["stage2_score"] = stage2_result["stage2_score"]
                result["stage2_details"] = stage2_result
            except Exception as e:
                print(f"Stage 2 error for video {video_idx}: {e}")
                result["stage2_score"] = 0
                result["stage2_error"] = str(e)

        # Stage 3: Video Naturalness
        if 3 in self.config.stages and video_q:
            try:
                video_question = video_q.get("video_question", {})
                stage3_result = self.naturalness_evaluator.evaluate_video(
                    video_path, video_question, caption, self.config.num_frames_stage3
                )
                result["stage3_score"] = stage3_result["stage3_score"]
                result["stage3_details"] = stage3_result
            except Exception as e:
                print(f"Stage 3 error for video {video_idx}: {e}")
                result["stage3_score"] = 0
                result["stage3_error"] = str(e)

        # Compute overall score
        s1 = result["stage1_score"] if result["stage1_score"] is not None else 1
        s2 = result["stage2_score"] if result["stage2_score"] is not None else 1
        s3 = result["stage3_score"] if result["stage3_score"] is not None else 1

        result["overall_score"] = compute_overall_score(s1, s2, s3)

        return result

    def run(self) -> Dict:
        """Run full evaluation pipeline."""
        print("=" * 60)
        print("PhyGenBench Evaluation Pipeline")
        print("=" * 60)

        # Setup
        videos = get_video_files(self.config.videos_dir, self.config.max_videos)
        print(f"Videos found: {len(videos)}")
        print(f"Stages to run: {self.config.stages}")
        print(f"Output dir: {self.config.output_dir}")

        if self.config.dry_run:
            print("\n[DRY RUN] Would evaluate:")
            for idx, path in videos[:5]:
                print(f"  Video {idx + 1}: {path}")
            if len(videos) > 5:
                print(f"  ... and {len(videos) - 5} more")
            return {}

        # Create output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(self.config.output_dir) / f"eval_{timestamp}"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Load existing results if resuming
        results_path = output_dir / "results.json"
        existing_results = {}
        if self.config.resume and results_path.exists():
            with open(results_path, "r") as f:
                existing_data = json.load(f)
                existing_results = {
                    r["index"]: r for r in existing_data.get("per_video", [])
                }
            print(f"Resuming: found {len(existing_results)} existing results")

        # Evaluate each video
        results = []
        for video_idx, video_path in tqdm(videos, desc="Evaluating"):
            # Skip if already evaluated
            if video_idx in existing_results:
                results.append(existing_results[video_idx])
                continue

            try:
                result = self.evaluate_single_video(video_idx, video_path)
                results.append(result)

                # Save incrementally
                self._save_results(results, output_dir)

            except Exception as e:
                print(f"\nError evaluating video {video_idx}: {e}")
                results.append(
                    {"index": video_idx, "video_path": video_path, "error": str(e)}
                )

        # Final save
        final_results = self._compute_summary(results)
        self._save_results(results, output_dir, final_results)

        return final_results

    def _compute_summary(self, results: List[Dict]) -> Dict:
        """Compute summary statistics."""
        # Filter valid results
        valid = [
            r for r in results if "error" not in r or r.get("overall_score") is not None
        ]

        # Per-stage averages
        stage1_scores = [
            r["stage1_score"] for r in valid if r.get("stage1_score") is not None
        ]
        stage2_scores = [
            r["stage2_score"] for r in valid if r.get("stage2_score") is not None
        ]
        stage3_scores = [
            r["stage3_score"] for r in valid if r.get("stage3_score") is not None
        ]
        overall_scores = [
            r["overall_score"] for r in valid if r.get("overall_score") is not None
        ]

        # Per-category breakdown
        category_scores = defaultdict(list)
        for r in valid:
            idx = r["index"]
            if idx < len(self.questions["single"]):
                cat = self.questions["single"][idx].get("main_category", "Unknown")
                if r.get("overall_score") is not None:
                    category_scores[cat].append(r["overall_score"])

        summary = {
            "total_videos": len(results),
            "valid_videos": len(valid),
            "stages_evaluated": self.config.stages,
            "stage1_avg": sum(stage1_scores) / len(stage1_scores)
            if stage1_scores
            else None,
            "stage2_avg": sum(stage2_scores) / len(stage2_scores)
            if stage2_scores
            else None,
            "stage3_avg": sum(stage3_scores) / len(stage3_scores)
            if stage3_scores
            else None,
            "overall_avg": sum(overall_scores) / len(overall_scores)
            if overall_scores
            else None,
            "per_category": {
                cat: sum(scores) / len(scores)
                for cat, scores in category_scores.items()
            },
        }

        return summary

    def _save_results(
        self, results: List[Dict], output_dir: Path, summary: Optional[Dict] = None
    ):
        """Save results to JSON."""
        data = {
            "config": asdict(self.config),
            "timestamp": datetime.now().isoformat(),
            "per_video": results,
        }

        if summary:
            data["summary"] = summary

        results_path = output_dir / "results.json"
        with open(results_path, "w") as f:
            json.dump(data, f, indent=2, default=str)

        # Also save summary separately
        if summary:
            summary_path = output_dir / "summary.json"
            with open(summary_path, "w") as f:
                json.dump(summary, f, indent=2)


# =============================================================================
# CLI
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="PhyGenBench 3-Stage Evaluation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Full evaluation
    python eval/eval_phygenbench_full.py --videos-dir outputs/phygenbench/baseline
    
    # Stage 1 only (no API cost)
    python eval/eval_phygenbench_full.py --videos-dir outputs/phygenbench/baseline --stage 1
    
    # GPT-4o stages only
    python eval/eval_phygenbench_full.py --videos-dir outputs/phygenbench/baseline --stage 2 3
    
    # Test with first 5 videos
    python eval/eval_phygenbench_full.py --videos-dir outputs/phygenbench/baseline --max-videos 5
""",
    )

    # Paths
    parser.add_argument(
        "--videos-dir",
        type=str,
        default="outputs/phygenbench/baseline",
        help="Directory containing videos (output_video_*.mp4)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/evaluation/phygenbench",
        help="Output directory for results",
    )
    parser.add_argument(
        "--prompts-file",
        type=str,
        default="data/phygenbench/prompts.json",
        help="Path to prompts.json",
    )

    # Stage selection
    parser.add_argument(
        "--stage",
        type=int,
        nargs="+",
        default=[1, 2, 3],
        help="Stages to run (1, 2, 3)",
    )

    # VQAScore options
    parser.add_argument(
        "--vqascore-model",
        type=str,
        default="clip-flant5-xxl",
        choices=["clip-flant5-xxl", "clip-flant5-xl"],
        help="VQAScore model",
    )

    # GPT-4o options
    parser.add_argument(
        "--gpt-model", type=str, default="gpt-4o", help="GPT model for stages 2 & 3"
    )

    # Testing options
    parser.add_argument(
        "--max-videos",
        type=int,
        default=None,
        help="Maximum videos to evaluate (for testing)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Just show what would be done"
    )
    parser.add_argument(
        "--no-resume", action="store_true", help="Don't resume from existing results"
    )

    args = parser.parse_args()

    # Check API key for GPT-4o stages
    if (2 in args.stage or 3 in args.stage) and not args.dry_run:
        if not os.environ.get("OPENAI_API_KEY"):
            print("ERROR: OPENAI_API_KEY not set!")
            print("Set it with: export OPENAI_API_KEY='sk-xxx'")
            sys.exit(1)

    # Create config
    config = EvalConfig(
        videos_dir=args.videos_dir,
        output_dir=args.output_dir,
        prompts_file=args.prompts_file,
        stages=args.stage,
        vqascore_model=args.vqascore_model,
        gpt_model=args.gpt_model,
        max_videos=args.max_videos,
        dry_run=args.dry_run,
        resume=not args.no_resume,
    )

    # Run evaluation
    evaluator = PhyGenBenchEvaluator(config)
    results = evaluator.run()

    # Print summary
    if results and "summary" not in results:
        results = {"summary": results}

    if results.get("summary"):
        summary = results["summary"]
        print("\n" + "=" * 60)
        print("EVALUATION SUMMARY")
        print("=" * 60)
        print(f"Total videos: {summary.get('total_videos', 'N/A')}")
        print(f"Valid videos: {summary.get('valid_videos', 'N/A')}")
        print(f"\nPer-Stage Averages:")
        if summary.get("stage1_avg") is not None:
            print(f"  Stage 1 (VQAScore): {summary['stage1_avg']:.3f}")
        if summary.get("stage2_avg") is not None:
            print(f"  Stage 2 (Multi-frame): {summary['stage2_avg']:.3f}")
        if summary.get("stage3_avg") is not None:
            print(f"  Stage 3 (Naturalness): {summary['stage3_avg']:.3f}")
        if summary.get("overall_avg") is not None:
            print(f"\n  Overall Score: {summary['overall_avg']:.3f}")

        if summary.get("per_category"):
            print(f"\nPer-Category Scores:")
            for cat, score in sorted(summary["per_category"].items()):
                print(f"  {cat}: {score:.3f}")


if __name__ == "__main__":
    main()
