#!/usr/bin/env python3
"""
PhyGenBench Evaluation Pipeline — 100% Faithful to Official PhyGenEval

Replicates the EXACT evaluation protocol from the official PhyGenBench repo.
Every branch traced line-by-line against:
  - PhyGenEval/single/vqascore.py           (Stage 1)
  - PhyGenEval/multi/multiimage_clip.py      (Stage 2 CLIP pre-step)
  - PhyGenEval/multi/GPT4o.py               (Stage 2 GPT-4o)
  - PhyGenEval/video/GPT4o.py               (Stage 3)
  - PhyGenEval/overall.py                   (aggregation)

Stage 2 is split into two sub-stages to handle OSCAR cluster constraints
(GPU nodes have no external network access):
  - 2clip: CLIP + VQAScore preprocessing for 24 non-Middle videos (GPU, no network)
  - 2gpt:  GPT-4o yes/no evaluation for all 160 videos (API, no GPU)

Usage:
    # GPU job (Stage 1 + Stage 2 CLIP preprocessing, ~16-18GB VRAM, ~2.5h)
    python eval/eval_phygenbench.py --videos-dir outputs/phygenbench/baseline --stage 1 2clip

    # API job (Stage 2 GPT + Stage 3 + Overall, no GPU needed, ~$8, ~2h)
    python eval/eval_phygenbench.py --videos-dir outputs/phygenbench/baseline --stage 2gpt 3 overall
"""

import argparse
import base64
import io
import json
import os
import sys
import tempfile
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class EvalConfig:
    videos_dir: str = "outputs/phygenbench/baseline"
    output_dir: str = ""  # Auto-set: <videos_dir>/eval
    stages: List = field(default_factory=lambda: [1, "2clip", "2gpt", 3, "overall"])

    # Data files
    single_question_file: str = "data/phygenbench/single_question.json"
    multi_question_file: str = "data/phygenbench/multi_question.json"
    video_question_file: str = "data/phygenbench/video_question.json"
    explicit_prompts_file: str = "data/phygenbench/explicit_prompts.json"
    prompts_file: str = "data/phygenbench/prompts.json"

    # Models
    vqascore_model: str = "clip-flant5-xxl"
    clip_model: str = "openai/clip-vit-large-patch14"
    gpt_model: str = "gpt-4o"

    # Frame sampling counts (match official)
    num_frames_clip: int = 32  # CLIP retrieval sampling (Stage 1 & 2)
    num_frames_video: int = 26  # GPT-4o video frames (Stage 3)

    # Options
    max_videos: Optional[int] = None
    resume: bool = True
    dry_run: bool = False


# =============================================================================
# Video / Image Utilities
# =============================================================================


def sample_frames(video_path: str, num_frames: int) -> List[Tuple[Image.Image, int]]:
    """Sample frames uniformly from video (matching official sample_frames)."""
    video = cv2.VideoCapture(video_path)
    total = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.linspace(0, total - 1, num_frames, dtype=int)
    frames = []
    for idx in indices:
        video.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = video.read()
        if ok:
            frames.append(
                (Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), int(idx))
            )
    video.release()
    return frames


def get_frame(video_path: str, position: str) -> Image.Image:
    """Get first / middle / last frame."""
    video = cv2.VideoCapture(video_path)
    total = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    idx = {"first": 0, "middle": total // 2, "last": total - 1}[position]
    video.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = video.read()
    video.release()
    if ok:
        return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    raise ValueError(f"Failed to read {position} frame from {video_path}")


def pil_to_base64(img: Image.Image, max_size: int = 512) -> str:
    """Convert PIL Image to base64 JPEG."""
    w, h = img.size
    if max(w, h) > max_size:
        s = max_size / max(w, h)
        img = img.resize((int(w * s), int(h * s)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def pil_to_tmpfile(img: Image.Image) -> str:
    """Save PIL Image to temp file, return path (caller must clean up)."""
    tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    img.save(tmp.name, "JPEG", quality=95)
    return tmp.name


def extract_video_frames_b64(video_path: str, num_frames: int) -> List[str]:
    """Extract frames for GPT-4o Stage 3 (matching official process_video)."""
    video = cv2.VideoCapture(video_path)
    total = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video.get(cv2.CAP_PROP_FPS) or 24.0
    duration = total / fps
    step = int(fps * duration / num_frames)
    b64_frames = []
    curr = 0
    while curr < total - 1 and len(b64_frames) < num_frames:
        video.set(cv2.CAP_PROP_POS_FRAMES, curr)
        ok, frame = video.read()
        if not ok:
            break
        _, buf = cv2.imencode(".jpg", frame)
        b64_frames.append(base64.b64encode(buf).decode("utf-8"))
        curr += step
    video.release()
    return b64_frames


# =============================================================================
# Data Loading
# =============================================================================


def load_questions(config: EvalConfig) -> Dict:
    """Load all question files."""
    qs = {}
    for key, path in [
        ("single", config.single_question_file),
        ("multi", config.multi_question_file),
        ("video", config.video_question_file),
    ]:
        if os.path.exists(path):
            with open(path) as f:
                qs[key] = json.load(f)
        else:
            print(f"WARNING: {path} not found")
            qs[key] = []

    if os.path.exists(config.explicit_prompts_file):
        with open(config.explicit_prompts_file) as f:
            qs["explicit"] = json.load(f)
    else:
        print(
            f"WARNING: {config.explicit_prompts_file} not found — Stage 2 Q3 will be empty!"
        )
        qs["explicit"] = []

    return qs


def get_video_files(
    videos_dir: str, max_n: Optional[int] = None
) -> List[Tuple[int, str]]:
    """Return sorted (0-based index, path) list."""
    vids = []
    for f in Path(videos_dir).glob("output_video_*.mp4"):
        try:
            idx = int(f.stem.split("_")[-1]) - 1  # 1-indexed filename -> 0-indexed
            vids.append((idx, str(f)))
        except (ValueError, IndexError):
            continue
    vids.sort()
    return vids[:max_n] if max_n else vids


# =============================================================================
# Stage 1: VQAScore + CLIP Retrieval
#
# Faithfully replicates: PhyGenEval/single/vqascore.py + overall.py lines 35-56
#
# Per question:
#   "Last Frame" -> VQAScore(last_frame, Statement) -> raw score
#   Otherwise    -> CLIP retrieval (32 frames) -> window ±2 (up to 5 frames)
#                   -> per frame: score_re + pos/(pos+neg) -> take max
#
# Per video:
#   1 question  -> score = max_combined - clip_retrieval_of_LAST_window_frame
#                  (original stores last iteration's score_re, line 206)
#                  -> discretize [0,0.25)->0, [0.25,0.5)->1, [0.5,0.75)->2, >=0.75->3
#   2 questions -> sum of max_combined scores
#                  -> discretize <=1->0, <=1.5->1, <=2.25->2, >2.25->3
#
# VRAM: ~16-18GB (VQAScore clip-flant5-xxl ~12-15GB + CLIP ~1GB)
# Time: ~2h per 160 videos on RTX 3090
# =============================================================================


class Stage1Evaluator:
    def __init__(self, vqascore_model: str, clip_model: str):
        self._vqa_name = vqascore_model
        self._clip_name = clip_model
        self._vqa = None
        self._clip_model = None
        self._clip_proc = None

    def _load_vqa(self):
        if self._vqa is None:
            import t2v_metrics

            self._vqa = t2v_metrics.VQAScore(model=self._vqa_name)
            print(f"[Stage1] Loaded VQAScore: {self._vqa_name}")

    def _load_clip(self):
        if self._clip_model is None:
            from transformers import CLIPProcessor, CLIPModel

            self._clip_model = CLIPModel.from_pretrained(self._clip_name)
            self._clip_proc = CLIPProcessor.from_pretrained(self._clip_name)
            print(f"[Stage1] Loaded CLIP: {self._clip_name}")

    def _clip_scores(
        self, frames: List[Tuple[Image.Image, int]], prompt: str
    ) -> List[float]:
        """CLIP similarity scores (matching official calculate_clip_scores)."""
        import torch

        self._load_clip()
        imgs = [f[0] for f in frames]
        inputs = self._clip_proc(
            text=[prompt], images=imgs, return_tensors="pt", padding=True
        )
        with torch.no_grad():
            out = self._clip_model(**inputs)
        logits = out.logits_per_image.squeeze().detach().cpu()
        del inputs, out
        return logits.tolist() if logits.dim() > 0 else [logits.item()]

    def _vqascore(self, img: Image.Image, text: str) -> float:
        """VQAScore for one image-text pair."""
        import torch

        self._load_vqa()
        path = pil_to_tmpfile(img)
        try:
            score = self._vqa([path], [text]).item()
            torch.cuda.empty_cache()
            return score
        finally:
            os.remove(path)

    def _get_window(self, frames, scores):
        """Match official save_surrounding_frames: ±2 around argmax.
        Returns (start, end) as half-open [start, end)."""
        max_idx = int(np.argmax(scores))
        start = max(max_idx - 2, 0)
        end = min(max_idx + 3, len(frames))
        return start, end

    def evaluate_video(
        self, video_path: str, questions: List[Dict], num_frames: int = 32
    ) -> Dict:
        """Evaluate one video for Stage 1. Returns stage1_score (0-3)."""
        sampled = sample_frames(video_path, num_frames)
        score_final_all = 0.0
        per_question = []

        for j, q in enumerate(questions):
            rp = q.get("Retrieval Prompt", "")
            statement = q.get("Statement", "")
            antonym = q.get("Antonym", "")

            if rp == "Last Frame":
                # === Last Frame branch (vqascore.py lines 148-159) ===
                # Official: get_last_frame() reads frame at total_frames-1 directly
                # NOT sampled[-1] — avoids edge case if sampling missed last frame
                last_img = get_frame(video_path, "last")
                score1 = self._vqascore(last_img, statement)
                score_re = 0.0
                score_final = score1
                clip_retrieval_stored = 0.0
                per_question.append(
                    {
                        "j": j,
                        "type": "last_frame",
                        "score_final": score_final,
                        "clip_retrieval_stored": clip_retrieval_stored,
                    }
                )
            else:
                # === CLIP retrieval branch (vqascore.py lines 164-206) ===
                clip_scores = self._clip_scores(sampled, rp)
                start, end = self._get_window(sampled, clip_scores)

                scores_res = []
                score_re = 0.0  # Holds LAST iteration's score_re (official line 206)

                for k in range(start, end):
                    frame_img = sampled[k][0]
                    # VQAScore(frame, retrieval_prompt)
                    score_re = self._vqascore(frame_img, rp)
                    # VQAScore(frame, statement)
                    score_pos = self._vqascore(frame_img, statement)
                    # VQAScore(frame, antonym)
                    score_neg = self._vqascore(frame_img, antonym) if antonym else 0.0
                    # Normalize (official line 194)
                    normalized = (
                        score_pos / (score_pos + score_neg)
                        if (score_pos + score_neg) > 0
                        else 0.5
                    )
                    # Combined (official line 198)
                    combined = score_re + normalized
                    scores_res.append(combined)

                score_final = max(scores_res)
                # clip_retrieval is LAST iteration's score_re (official line 206)
                clip_retrieval_stored = score_re

                per_question.append(
                    {
                        "j": j,
                        "type": "clip_retrieval",
                        "score_final": score_final,
                        "clip_retrieval_stored": clip_retrieval_stored,
                        "window": [start, end],
                    }
                )

            score_final_all += score_final

        # === Discretization (overall.py lines 35-56) ===
        num_q = len(questions)
        if num_q == 1:
            pq = per_question[0]
            if pq["type"] == "last_frame":
                raw = pq["score_final"]  # overall.py line 43
            else:
                raw = (
                    pq["score_final"] - pq["clip_retrieval_stored"]
                )  # overall.py line 41
            if raw < 0.25:
                discrete = 0
            elif raw < 0.5:
                discrete = 1
            elif raw < 0.75:
                discrete = 2
            else:
                discrete = 3
        else:
            # 2-question (vqascore.py lines 222-229)
            s = score_final_all
            if s <= 1.0:
                discrete = 0
            elif s <= 1.5:
                discrete = 1
            elif s <= 2.25:
                discrete = 2
            else:
                discrete = 3

        return {
            "stage1_score": discrete,
            "raw_score": float(score_final_all),
            "num_questions": num_q,
            "per_question": per_question,
        }


# =============================================================================
# Stage 2 CLIP Preprocessing (GPU, no network)
#
# For the 24 non-Middle videos, run CLIP retrieval + VQAScore to get:
#   - window indices (start, max, end)
#   - per-frame retrieval scores
# Save to stage2_clip_results.json for later use by Stage 2 GPT.
#
# Replicates: PhyGenEval/multi/multiimage_clip.py
#
# VRAM: ~16-18GB (shared with Stage 1 if run together)
# Time: ~15 min for 24 videos
# =============================================================================


class Stage2ClipPreprocessor:
    """CLIP + VQAScore preprocessing for Stage 2 non-Middle videos (GPU only)."""

    def __init__(self, vqascore_model: str, clip_model: str):
        self._vqa_name = vqascore_model
        self._clip_name = clip_model
        self._vqa = None
        self._clip_model = None
        self._clip_proc = None

    def _load_vqa(self):
        if self._vqa is None:
            import t2v_metrics

            self._vqa = t2v_metrics.VQAScore(model=self._vqa_name)
            print(f"[Stage2Clip] Loaded VQAScore: {self._vqa_name}")

    def _load_clip(self):
        if self._clip_model is None:
            from transformers import CLIPProcessor, CLIPModel

            self._clip_model = CLIPModel.from_pretrained(self._clip_name)
            self._clip_proc = CLIPProcessor.from_pretrained(self._clip_name)
            print(f"[Stage2Clip] Loaded CLIP: {self._clip_name}")

    def _clip_scores(self, frames, prompt):
        import torch

        self._load_clip()
        imgs = [f[0] for f in frames]
        inputs = self._clip_proc(
            text=[prompt], images=imgs, return_tensors="pt", padding=True
        )
        with torch.no_grad():
            out = self._clip_model(**inputs)
        logits = out.logits_per_image.squeeze().detach().cpu()
        del inputs, out
        return logits.tolist() if logits.dim() > 0 else [logits.item()]

    def _vqascore(self, img, text):
        import torch

        self._load_vqa()
        path = pil_to_tmpfile(img)
        try:
            score = self._vqa([path], [text]).item()
            torch.cuda.empty_cache()
            return score
        finally:
            os.remove(path)

    def preprocess_video(
        self, video_path: str, retrieval_prompt: str, num_frames: int = 32
    ) -> Dict:
        """CLIP retrieval + VQAScore for one non-Middle video.
        Replicates multiimage_clip.py lines 243-290."""
        sampled = sample_frames(video_path, num_frames)
        clip_scores = self._clip_scores(sampled, retrieval_prompt)

        # Window: ±1 around argmax (multiimage_clip.py save_surrounding_frames)
        max_index = int(np.argmax(clip_scores))
        start_index = max(max_index - 1, 0)
        end_index = min(max_index + 2, len(sampled))

        # VQAScore retrieval scores (multiimage_clip.py lines 259-288)
        # retrieval_1: frames for Q1 loop [start, max+1)
        retrieval_1 = {}
        for k in range(start_index, max_index + 1):
            retrieval_1[str(k)] = self._vqascore(sampled[k][0], retrieval_prompt)

        # retrieval_2: frames for Q2 loop [max, end)
        retrieval_2 = {}
        for m in range(max_index, end_index):
            retrieval_2[str(m)] = self._vqascore(sampled[m][0], retrieval_prompt)

        # Save frame images as base64 for later GPT-4o calls
        frame_b64 = {}
        for k in range(start_index, end_index):
            frame_b64[str(k)] = pil_to_base64(sampled[k][0])

        first_b64 = pil_to_base64(get_frame(video_path, "first"))
        last_b64 = pil_to_base64(get_frame(video_path, "last"))

        return {
            "start_index": start_index,
            "max_index": max_index,
            "end_index": end_index,
            "retrieval_1": retrieval_1,  # str(k) -> float
            "retrieval_2": retrieval_2,  # str(m) -> float
            "frame_b64": frame_b64,  # str(k) -> base64 string
            "first_b64": first_b64,
            "last_b64": last_b64,
        }


# =============================================================================
# Stage 2 GPT Evaluation (API, no GPU)
#
# Two paths:
#   Middle (136/160): first+mid+last -> 3 GPT-4o yes/no -> 0-3
#   Non-Middle (24/160): Load pre-computed CLIP data from stage2_clip_results.json
#     Q1: per frame in [start, max+1): GPT(first+frame, desc1) -> combined = retrieval+gpt
#     Q2: per frame in [max, end):     GPT(frame+last, desc2)  -> combined = retrieval+gpt
#     Q3: GPT(all_frames, explicit_caption) -> max_retrieval + gpt
#     Each sub-score >= 1.5 -> 1, else 0 -> total 0-3
#
# Replicates: PhyGenEval/multi/GPT4o.py + overall.py lines 61-100
#
# API cost: ~$5 for 160 videos
# Time: ~1h
# =============================================================================

STAGE2_FORMAT = 'Answer me in Format:{"Choice":"Yes or No","Reason":"the reason"} '

STAGE2_Q_TMPL = (
    "Look carefully at the picture. Please check if the temporal sequence "
    "depicted in these two images matches the following description. "
    "First, answer Yes or No, then explain the reason. Think step-by-step.\n"
    "Description: {description}\n" + STAGE2_FORMAT
)


class Stage2GptEvaluator:
    """GPT-4o evaluation for Stage 2 (API only, no GPU needed)."""

    def __init__(self, gpt_model: str):
        self.gpt_model = gpt_model
        self._client = None

    def _init_client(self):
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI()

    def _gpt4o_yesno(
        self,
        images_b64: List[str],
        description: str,
        temperature: float = 0,
        top_p: float = 1,
    ) -> Tuple[int, Dict]:
        """GPT-4o yes/no. Returns (0 or 1, details). NEVER raises."""
        self._init_client()
        question = STAGE2_Q_TMPL.format(description=description)
        content = [{"type": "text", "text": question}]
        for b64 in images_b64:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                }
            )
        try:
            resp = self._client.chat.completions.create(
                model=self.gpt_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an assistant that only outputs valid JSON format. "
                        "Always use double quotes for keys and values, and never use "
                        "single quotes or any extra text.",
                    },
                    {"role": "user", "content": content},
                ],
                max_tokens=200,
                temperature=temperature,
                top_p=top_p,
            )
            raw = resp.choices[0].message.content.strip()
            raw = raw.lstrip("```json").rstrip("```").strip()
            # Parse JSON — use .get() exclusively, NEVER bare dict["key"]
            choice = raw  # fallback
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, dict):
                    choice = str(parsed.get("Choice", parsed.get("choice", raw)))
                else:
                    choice = str(parsed)
            except (json.JSONDecodeError, TypeError, ValueError):
                choice = raw
            score = 1 if "yes" in choice.lower() else 0
            return score, {"choice": choice, "raw": raw}
        except Exception as e:
            print(f"  GPT-4o error: {e}")
            return 0, {"error": str(e)}

    def evaluate_video_middle(
        self, video_path: str, desc1: str, desc2: str, desc3: str
    ) -> Dict:
        """Middle Frame path (official GPT4o.py lines 224-378)."""
        first_b64 = pil_to_base64(get_frame(video_path, "first"))
        mid_b64 = pil_to_base64(get_frame(video_path, "middle"))
        last_b64 = pil_to_base64(get_frame(video_path, "last"))

        s1, d1 = self._gpt4o_yesno([first_b64, mid_b64], desc1)
        time.sleep(0.3)
        s2, d2 = self._gpt4o_yesno([mid_b64, last_b64], desc2)
        time.sleep(0.3)
        s3, d3 = self._gpt4o_yesno([first_b64, mid_b64, last_b64], desc3)

        return {
            "stage2_score": s1 + s2 + s3,
            "path": "middle",
            "q1": {"score": s1, **d1},
            "q2": {"score": s2, **d2},
            "q3": {"score": s3, **d3},
        }

    def evaluate_video_nonmiddle(
        self, clip_data: Dict, desc1: str, desc2: str, desc3: str
    ) -> Dict:
        """Non-Middle path using pre-computed CLIP data (GPT4o.py lines 382-623).

        Args:
            clip_data: Pre-computed from Stage2ClipPreprocessor, containing:
                - start_index, max_index, end_index
                - retrieval_1, retrieval_2 (str(k) -> float)
                - frame_b64 (str(k) -> base64)
                - first_b64, last_b64
        """
        # Validate required keys (clip entries can be error-only dicts)
        required = [
            "start_index",
            "max_index",
            "end_index",
            "retrieval_1",
            "retrieval_2",
            "frame_b64",
            "first_b64",
            "last_b64",
        ]
        missing = [k for k in required if k not in clip_data]
        if missing:
            raise ValueError(
                f"Incomplete clip_data, missing keys: {missing}. "
                f"Re-run '--stage 2clip' on GPU."
            )

        start_index = clip_data["start_index"]
        max_index = clip_data["max_index"]
        end_index = clip_data["end_index"]
        retrieval_1 = clip_data["retrieval_1"]  # str(k) -> float
        retrieval_2 = clip_data["retrieval_2"]  # str(m) -> float
        frame_b64 = clip_data["frame_b64"]  # str(k) -> base64
        first_b64 = clip_data["first_b64"]
        last_b64 = clip_data["last_b64"]

        # --- Q1: first + window frame -> desc1 (GPT4o.py lines 406-471) ---
        scores1 = []
        q1_details = []
        for k in range(start_index, max_index + 1):
            gpt_s, det = self._gpt4o_yesno([first_b64, frame_b64[str(k)]], desc1)
            time.sleep(0.3)
            combined = retrieval_1[str(k)] + gpt_s
            scores1.append(combined)
            q1_details.append(
                {"k": k, "ret": retrieval_1[str(k)], "gpt": gpt_s, "comb": combined}
            )

        gpt4o_1_score = max(scores1) if scores1 else 0.0

        # --- Q2: window frame + last -> desc2 (GPT4o.py lines 476-540) ---
        scores2 = []
        q2_details = []
        for m in range(max_index, end_index):
            gpt_s, det = self._gpt4o_yesno([frame_b64[str(m)], last_b64], desc2)
            time.sleep(0.3)
            combined = retrieval_2[str(m)] + gpt_s
            scores2.append(combined)
            q2_details.append(
                {"m": m, "ret": retrieval_2[str(m)], "gpt": gpt_s, "comb": combined}
            )

        gpt4o_2_score = max(scores2) if scores2 else 0.0

        # --- Q3: all frames -> desc3 (GPT4o.py lines 550-610) ---
        # NOTE: Official code collects frame_middle_output_paths from BOTH Q1 and Q2
        # loops. Since max_index appears in both [start, max+1) and [max, end),
        # the max_index frame is included TWICE. We replicate this faithfully.
        all_ret = list(retrieval_1.values()) + list(retrieval_2.values())
        score_re3 = max(all_ret) if all_ret else 0.0

        all_b64 = [first_b64]
        # Q1 loop frames: [start_index, max_index+1)
        for k in range(start_index, max_index + 1):
            all_b64.append(frame_b64[str(k)])
        # Q2 loop frames: [max_index, end_index) — max_index sent again
        for m in range(max_index, end_index):
            all_b64.append(frame_b64[str(m)])
        all_b64.append(last_b64)

        gpt3_s, d3 = self._gpt4o_yesno(all_b64, desc3)
        gpt4o_3_score = score_re3 + gpt3_s

        # --- Discretize (overall.py lines 84-100): each >= 1.5 -> 1 ---
        total = 0
        total += 1 if gpt4o_1_score >= 1.5 else 0
        total += 1 if gpt4o_2_score >= 1.5 else 0
        total += 1 if gpt4o_3_score >= 1.5 else 0

        return {
            "stage2_score": total,
            "path": "nonmiddle",
            "clip_window": [start_index, max_index, end_index],
            "gpt4o_1_score": float(gpt4o_1_score),
            "gpt4o_2_score": float(gpt4o_2_score),
            "gpt4o_3_score": float(gpt4o_3_score),
            "q1_details": q1_details,
            "q2_details": q2_details,
            "q3": {"ret_max": float(score_re3), "gpt": gpt3_s},
        }

    def evaluate_video(
        self,
        video_path: str,
        multi_q: Dict,
        explicit_caption: str,
        clip_data: Optional[Dict] = None,
    ) -> Dict:
        """Route to Middle or Non-Middle path."""
        mq = multi_q.get("multiimage_question", multi_q)
        rp = mq.get("Retrieval Prompt", "Middle Frame")
        desc1 = mq.get("Description1", "")
        desc2 = mq.get("Description2", "")
        desc3 = explicit_caption  # Q3 uses explicit_caption

        if rp == "Middle Frame":
            return self.evaluate_video_middle(video_path, desc1, desc2, desc3)
        else:
            if clip_data is None:
                raise ValueError(
                    f"Non-Middle video requires pre-computed CLIP data. "
                    f"Run '--stage 2clip' first (GPU job)."
                )
            return self.evaluate_video_nonmiddle(clip_data, desc1, desc2, desc3)


# =============================================================================
# Stage 3: Video-level Naturalness (GPT-4o)
#
# Official generate_question.py produces:
#   Description1 = Completely Fantastical -> 0
#   Description2 = Clearly Unrealistic    -> 1
#   Description3 = Slightly Unrealistic   -> 2
#   Description4 = Almost Realistic       -> 3
#
# NOTE: Official GPT4o.py references Description4->"Slightly Unrealistic" and
# Description5->"Almost Realistic", but Description5 doesn't exist in JSON.
# We use Description3/4 matching the data (confirmed bug in original code).
#
# VRAM: 0GB (API only)
# API cost: ~$3 for 160 videos
# Time: ~30 min
# =============================================================================


class Stage3Evaluator:
    def __init__(self, gpt_model: str):
        self.gpt_model = gpt_model
        self._client = None

    def _init_client(self):
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI()

    def evaluate_video(
        self, video_path, video_q, caption, physical_laws, num_frames=26
    ):
        """Evaluate one video for Stage 3. Returns stage3_score (0-3)."""
        self._init_client()
        b64_frames = extract_video_frames_b64(video_path, num_frames)

        desc1 = video_q.get("Description1", "")
        desc2 = video_q.get("Description2", "")
        desc3 = video_q.get("Description3", "")
        desc4 = video_q.get("Description4", "")

        # Prompt (official GPT4o.py lines 66-108)
        task_prompt = """### Task Overview:

    Your task is to analyze an input video to determine whether it conforms to real-world physical laws. You will receive the T2V prompt corresponding to this video, as well as the physical law it primarily reflects. Besides, you will be provided with four different descriptions (Completely Fantastical, Highly Unrealistic, Slightly Unrealistic, Almost Realistic) that offer varying levels of detail or focus. Your goal is to select the most appropriate description to evaluate the extent to which this video conforms to the emphasized physical law.

    ### Task Requirements:

    1. **Selection**: Choose the description that best suits the purpose of assessing the video's physical realism.
    2. **Explanation**: Provide a reason for your selection, explaining why this description is the most relevant for the task.

    ### Expected Output Format:

    {
    "Choice": "<Selected_Description>",
    "Reason": "<Explanation>"
    }

    ### Special Notes:

    - Exercise caution when assigning choices, especially when considering the Almost Realistic.
    - Do not easily give the choice of Almost Realistic.
    - Use step-by-step reasoning to make your selection, considering the relevance and specificity of each description.
    - The explanation should be concise but comprehensive, highlighting key factors that influenced your choice.
    - You need to focus on whether the video reflects the emphasized physical law.

    """

        input_prompt = f"""
    Here is the T2V prompt and the physical law it primarily reflects:
    Prompt:{caption}
    Physical_Law:{physical_laws}
    Here is the different descriptions:
    Completely Fantastical:{desc1}
    Highly Unrealistic:{desc2}
    Slightly Unrealistic:{desc3}
    Almost Realistic:{desc4}
    """

        try:
            resp = self._client.chat.completions.create(
                model=self.gpt_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an assistant that only outputs valid JSON format. "
                        "Always use double quotes for keys and values, and never use "
                        "single quotes or any extra text.",
                    },
                    {"role": "user", "content": task_prompt + input_prompt},
                    {
                        "role": "user",
                        "content": [
                            "These are the frames from the video.",
                            *[
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpg;base64,{x}",
                                        "detail": "low",
                                    },
                                }
                                for x in b64_frames
                            ],
                        ],
                    },
                ],
                temperature=0,
            )
            raw = resp.choices[0].message.content.strip()
            raw = raw.lstrip("```json").rstrip("```").strip()
            parsed = json.loads(raw)
            choice = parsed.get("Choice", "")

            # Map (official GPT4o.py lines 143-150)
            if "Completely Fantastical" in choice:
                score = 0
            elif "Highly Unrealistic" in choice or "Clearly Unrealistic" in choice:
                score = 1
            elif "Slightly Unrealistic" in choice:
                score = 2
            elif "Almost Realistic" in choice:
                score = 3
            else:
                score = 0
                for s, kw in [(3, "realistic"), (2, "slightly"), (1, "unrealistic")]:
                    if kw in choice.lower():
                        score = s
                        break

            return {
                "stage3_score": score,
                "choice": choice,
                "reason": parsed.get("Reason", ""),
                "raw": raw,
            }

        except Exception as e:
            print(f"  GPT-4o Stage 3 error: {e}")
            return {"stage3_score": 0, "error": str(e)}


# =============================================================================
# Overall Score (official overall.py line 197)
# =============================================================================


def compute_overall(s1: int, s2: int, s3: int) -> int:
    return round((s1 + s2 + s3) / 3)


# =============================================================================
# Main Pipeline
# =============================================================================


class PhyGenBenchEvaluator:
    def __init__(self, config: EvalConfig):
        self.config = config
        self.questions = load_questions(config)
        if not config.output_dir:
            config.output_dir = os.path.join(config.videos_dir, "eval")
        self._s1 = None
        self._s2clip = None
        self._s2gpt = None
        self._s3 = None

    @property
    def stage1(self):
        if self._s1 is None:
            self._s1 = Stage1Evaluator(
                self.config.vqascore_model, self.config.clip_model
            )
        return self._s1

    @property
    def stage2clip(self):
        if self._s2clip is None:
            self._s2clip = Stage2ClipPreprocessor(
                self.config.vqascore_model, self.config.clip_model
            )
            # Share model instances from Stage 1 if already loaded
            # This avoids loading VQAScore (~12GB) twice on 24GB GPUs
            if self._s1 is not None:
                if self._s1._vqa is not None:
                    self._s2clip._vqa = self._s1._vqa
                    print("[Stage2Clip] Sharing VQAScore instance from Stage 1")
                if self._s1._clip_model is not None:
                    self._s2clip._clip_model = self._s1._clip_model
                    self._s2clip._clip_proc = self._s1._clip_proc
                    print("[Stage2Clip] Sharing CLIP instance from Stage 1")
        return self._s2clip

    @property
    def stage2gpt(self):
        if self._s2gpt is None:
            self._s2gpt = Stage2GptEvaluator(self.config.gpt_model)
        return self._s2gpt

    @property
    def stage3(self):
        if self._s3 is None:
            self._s3 = Stage3Evaluator(self.config.gpt_model)
        return self._s3

    def _load_existing(self, stage):
        p = Path(self.config.output_dir) / f"{stage}_results.json"
        if p.exists() and self.config.resume:
            with open(p) as f:
                return {r["index"]: r for r in json.load(f)}
        return {}

    def _save(self, stage, results):
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
        p = Path(self.config.output_dir) / f"{stage}_results.json"
        with open(p, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    # ---- Stage 1 (GPU) ----

    def run_stage1(self, videos):
        print("\n" + "=" * 60)
        print("Stage 1: VQAScore + CLIP Retrieval")
        print("  VRAM: ~16-18GB | Time: ~2h / 160 videos")
        print("=" * 60)
        existing = self._load_existing("stage1")
        if existing:
            print(f"  Resuming from {len(existing)} existing results")
        results = list(existing.values())

        for idx, vpath in tqdm(videos, desc="Stage 1"):
            if idx in existing:
                continue
            sq = (
                self.questions["single"][idx]
                if idx < len(self.questions["single"])
                else None
            )
            if not sq:
                continue
            qs = sq.get("singleimage_question", sq.get("vqa_question", []))
            try:
                r = self.stage1.evaluate_video(vpath, qs, self.config.num_frames_clip)
                r["index"] = idx
                results.append(r)
                self._save("stage1", results)
            except Exception as e:
                print(f"\n  ERROR video {idx}: {e}")
                results.append({"index": idx, "stage1_score": 0, "error": str(e)})

        self._save("stage1", results)
        scores = [r["stage1_score"] for r in results if "stage1_score" in r]
        print(f"\nStage 1 done: mean={np.mean(scores):.2f}/3 (n={len(scores)})")

    # ---- Stage 2 CLIP (GPU, no network) ----

    def run_stage2clip(self, videos):
        print("\n" + "=" * 60)
        print("Stage 2 CLIP: Preprocessing for 24 non-Middle videos")
        print("  VRAM: ~16-18GB (shared with Stage 1) | Time: ~15 min")
        print("  No network access needed")
        print("=" * 60)

        existing = self._load_existing("stage2_clip")
        if existing:
            print(f"  Resuming from {len(existing)} existing results")
        results = list(existing.values())

        # Identify non-Middle videos
        nonmiddle_indices = set()
        for idx, _ in videos:
            if idx < len(self.questions["multi"]):
                mq = self.questions["multi"][idx]
                rp = mq.get("multiimage_question", mq).get(
                    "Retrieval Prompt", "Middle Frame"
                )
                if rp != "Middle Frame":
                    nonmiddle_indices.add(idx)

        print(f"  Non-Middle videos to process: {len(nonmiddle_indices)}")

        for idx, vpath in tqdm(videos, desc="Stage 2 CLIP"):
            if idx not in nonmiddle_indices or idx in existing:
                continue
            mq = self.questions["multi"][idx]
            rp = mq.get("multiimage_question", mq).get("Retrieval Prompt", "")
            try:
                r = self.stage2clip.preprocess_video(
                    vpath, rp, self.config.num_frames_clip
                )
                r["index"] = idx
                results.append(r)
                self._save("stage2_clip", results)
            except Exception as e:
                print(f"\n  ERROR video {idx}: {e}")
                results.append({"index": idx, "error": str(e)})

        self._save("stage2_clip", results)
        ok = sum(1 for r in results if "start_index" in r)
        print(
            f"\nStage 2 CLIP done: {ok}/{len(nonmiddle_indices)} non-Middle videos processed"
        )

    # ---- Stage 2 GPT (API, no GPU) ----

    def run_stage2gpt(self, videos):
        print("\n" + "=" * 60)
        print("Stage 2 GPT: Multi-frame Physics Verification (GPT-4o)")
        print("  Middle: 136 videos x 3 API calls (no GPU)")
        print("  Non-Middle: 24 videos x ~5 API calls (uses pre-computed CLIP data)")
        print("  API cost: ~$5 | Time: ~1h")
        print("=" * 60)

        # Fail-fast if explicit_prompts.json is missing
        if not self.questions["explicit"]:
            print("ERROR: explicit_prompts.json not loaded!")
            print("  Stage 2 Q3 requires this file for all 160 videos.")
            print("  Run: bash eval/eval_phygenbench_tools.sh setup")
            print(
                "  Or:  cp PhyGenBench/PhyGenEval/multi/explicit_prompts.json data/phygenbench/"
            )
            return

        # Load pre-computed CLIP data for non-Middle videos
        clip_data_raw = self._load_existing("stage2_clip")
        clip_data = {}
        clip_errors = 0
        for idx, entry in clip_data_raw.items():
            if "start_index" in entry:
                clip_data[idx] = entry
            else:
                clip_errors += 1
        if clip_data:
            print(f"  Loaded CLIP data for {len(clip_data)} non-Middle videos")
            if clip_errors:
                print(
                    f"  WARNING: {clip_errors} non-Middle videos had CLIP errors (OOM?)"
                )
                print(f"  These will be SKIPPED. Re-run 2clip on GPU to fix.")
        else:
            print("  WARNING: No valid stage2_clip_results.json found.")
            print("  Non-Middle videos will be SKIPPED.")
            print("  Run '--stage 2clip' on GPU first!")

        existing = self._load_existing("stage2")
        if existing:
            print(f"  Resuming from {len(existing)} existing results")
        results = list(existing.values())

        for idx, vpath in tqdm(videos, desc="Stage 2 GPT"):
            if idx in existing:
                continue
            mq = (
                self.questions["multi"][idx]
                if idx < len(self.questions["multi"])
                else None
            )
            eq = (
                self.questions["explicit"][idx]
                if idx < len(self.questions["explicit"])
                else None
            )
            if not mq:
                continue
            explicit_caption = eq.get("explicit_caption", "") if eq else ""

            # Check if this is a non-Middle video that needs CLIP data
            rp = mq.get("multiimage_question", mq).get(
                "Retrieval Prompt", "Middle Frame"
            )
            video_clip_data = clip_data.get(idx) if rp != "Middle Frame" else None

            if rp != "Middle Frame" and video_clip_data is None:
                print(
                    f"\n  SKIP video {idx}: non-Middle but no CLIP data (run 2clip first)"
                )
                results.append(
                    {
                        "index": idx,
                        "stage2_score": 0,
                        "error": "missing clip data",
                        "path": "nonmiddle_skipped",
                    }
                )
                continue

            try:
                r = self.stage2gpt.evaluate_video(
                    vpath, mq, explicit_caption, video_clip_data
                )
                r["index"] = idx
                results.append(r)
                self._save("stage2", results)
            except Exception as e:
                print(f"\n  ERROR video {idx}: {e}")
                results.append({"index": idx, "stage2_score": 0, "error": str(e)})
            time.sleep(0.2)

        self._save("stage2", results)
        scores = [r["stage2_score"] for r in results if "stage2_score" in r]
        nm = sum(1 for r in results if r.get("path") == "nonmiddle")
        print(
            f"\nStage 2 GPT done: mean={np.mean(scores):.2f}/3 (n={len(scores)}, non-middle={nm})"
        )

    # ---- Stage 3 (API, no GPU) ----

    def run_stage3(self, videos):
        print("\n" + "=" * 60)
        print("Stage 3: Video Naturalness (GPT-4o, 4-level)")
        print("  API cost: ~$3 | Time: ~30min")
        print("=" * 60)
        existing = self._load_existing("stage3")
        if existing:
            print(f"  Resuming from {len(existing)} existing results")
        results = list(existing.values())

        for idx, vpath in tqdm(videos, desc="Stage 3"):
            if idx in existing:
                continue
            vq = (
                self.questions["video"][idx]
                if idx < len(self.questions["video"])
                else None
            )
            if not vq:
                continue
            caption = vq.get("caption", "")
            plaws = vq.get("physical_laws", "")
            vquest = vq.get("video_question", {})
            try:
                r = self.stage3.evaluate_video(
                    vpath, vquest, caption, plaws, self.config.num_frames_video
                )
                r["index"] = idx
                results.append(r)
                self._save("stage3", results)
            except Exception as e:
                print(f"\n  ERROR video {idx}: {e}")
                results.append({"index": idx, "stage3_score": 0, "error": str(e)})
            time.sleep(0.2)

        self._save("stage3", results)
        scores = [r["stage3_score"] for r in results if "stage3_score" in r]
        print(f"\nStage 3 done: mean={np.mean(scores):.2f}/3 (n={len(scores)})")

    # ---- Overall ----

    def run_overall(self):
        print("\n" + "=" * 60)
        print("Overall Aggregation")
        print("=" * 60)
        s1r = self._load_existing("stage1")
        s2r = self._load_existing("stage2")
        s3r = self._load_existing("stage3")

        if not s1r or not s2r or not s3r:
            print("ERROR: missing stage results. Run all stages first.")
            print(f"  Stage 1: {len(s1r)}, Stage 2: {len(s2r)}, Stage 3: {len(s3r)}")
            return {}

        prompts = []
        if os.path.exists(self.config.prompts_file):
            with open(self.config.prompts_file) as f:
                prompts = json.load(f)

        cat_map = {
            "Force": "Mechanics",
            "Light": "Optics",
            "Heat": "Thermal",
            "Physical Properties": "Material",
            "Chemical Properties": "Material",
        }

        indices = sorted(set(s1r) & set(s2r) & set(s3r))
        print(f"Videos with all 3 stages: {len(indices)}")

        per_video = []
        paper_cats = defaultdict(list)

        for idx in indices:
            s1 = s1r[idx]["stage1_score"]
            s2 = s2r[idx]["stage2_score"]
            s3 = s3r[idx]["stage3_score"]
            ov = compute_overall(s1, s2, s3)

            cat_raw = (
                prompts[idx].get("main_category", "?") if idx < len(prompts) else "?"
            )
            cat = cat_map.get(cat_raw, cat_raw)

            per_video.append(
                {
                    "index": idx,
                    "stage1": s1,
                    "stage2": s2,
                    "stage3": s3,
                    "overall": ov,
                    "category": cat,
                    "category_raw": cat_raw,
                }
            )
            paper_cats[cat].append(ov)

        overall_scores = [v["overall"] for v in per_video]
        final = np.mean(overall_scores) / 3.0 if overall_scores else 0.0

        summary = {
            "total_videos": len(indices),
            "final_score": float(final),
            "stage1_avg": float(np.mean([v["stage1"] for v in per_video])),
            "stage2_avg": float(np.mean([v["stage2"] for v in per_video])),
            "stage3_avg": float(np.mean([v["stage3"] for v in per_video])),
            "overall_avg_0to3": float(np.mean(overall_scores)),
            "per_category": {
                cat: float(np.mean(paper_cats[cat]) / 3.0)
                for cat in ["Mechanics", "Optics", "Thermal", "Material"]
                if cat in paper_cats
            },
        }

        print(f"\n{'=' * 60}")
        print("RESULTS (PhyGenEval Protocol — Closed-Source Only)")
        print(f"{'=' * 60}")
        print(f"Final Score: {summary['final_score']:.3f}")
        print(f"\nPer-Stage (0-3 scale):")
        print(f"  S1 VQAScore:    {summary['stage1_avg']:.2f}")
        print(f"  S2 Multi-frame: {summary['stage2_avg']:.2f}")
        print(f"  S3 Naturalness: {summary['stage3_avg']:.2f}")
        print(f"  Overall:        {summary['overall_avg_0to3']:.2f}")
        print(f"\nPer-Category (0-1 scale):")
        for c in ["Mechanics", "Optics", "Thermal", "Material"]:
            if c in summary["per_category"]:
                print(f"  {c:20s}: {summary['per_category'][c]:.3f}")
        print(f"  {'Average':20s}: {summary['final_score']:.3f}")
        print(
            f"\nRef (paper CogVideoX-2B ENSEMBLE): Mech=0.38 Opti=0.43 Ther=0.34 Mate=0.39 Avg=0.37"
        )
        print(f"Note: closed-source only -> compare within our methods")

        out = {
            "summary": summary,
            "per_video": per_video,
            "config": {
                "videos_dir": self.config.videos_dir,
                "date": datetime.now().isoformat(),
            },
        }
        outpath = Path(self.config.output_dir) / "final_results.json"
        with open(outpath, "w") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f"\nSaved: {outpath}")
        return out

    # ---- Run ----

    def run(self):
        videos = get_video_files(self.config.videos_dir, self.config.max_videos)
        print(f"Found {len(videos)} videos in {self.config.videos_dir}")
        if self.config.dry_run:
            print("[DRY RUN]")
            for i, p in videos[:5]:
                print(f"  Video {i + 1}: {p}")
            return

        stages = self.config.stages
        if 1 in stages:
            self.run_stage1(videos)
            self._clear_gpu_cache("Stage 1")
        if "2clip" in stages:
            self.run_stage2clip(videos)
            self._clear_gpu_cache("Stage 2 CLIP")
        if "2gpt" in stages:
            self.run_stage2gpt(videos)
        if 3 in stages:
            self.run_stage3(videos)
        if "overall" in stages:
            return self.run_overall()

    @staticmethod
    def _clear_gpu_cache(stage_name: str):
        """Clear GPU cache between stages to reduce fragmentation."""
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                import gc

                gc.collect()
                free = torch.cuda.mem_get_info()[0] / 1e9
                print(f"\n[{stage_name}] GPU cache cleared. Free VRAM: {free:.1f}GB")
        except ImportError:
            pass


# =============================================================================
# CLI
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="PhyGenBench Evaluation (faithful to official PhyGenEval)"
    )
    parser.add_argument("--videos-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="")
    parser.add_argument(
        "--stage",
        nargs="+",
        default=["1", "2clip", "2gpt", "3", "overall"],
        help="Stages to run. Options: 1, 2clip, 2gpt, 3, overall. "
        "GPU stages: 1, 2clip. API stages: 2gpt, 3.",
    )

    parser.add_argument(
        "--single-question-file",
        type=str,
        default="data/phygenbench/single_question.json",
    )
    parser.add_argument(
        "--multi-question-file",
        type=str,
        default="data/phygenbench/multi_question.json",
    )
    parser.add_argument(
        "--video-question-file",
        type=str,
        default="data/phygenbench/video_question.json",
    )
    parser.add_argument(
        "--explicit-prompts-file",
        type=str,
        default="data/phygenbench/explicit_prompts.json",
    )
    parser.add_argument(
        "--prompts-file", type=str, default="data/phygenbench/prompts.json"
    )

    parser.add_argument("--vqascore-model", type=str, default="clip-flant5-xxl")
    parser.add_argument(
        "--clip-model", type=str, default="openai/clip-vit-large-patch14"
    )
    parser.add_argument("--gpt-model", type=str, default="gpt-4o")

    parser.add_argument("--max-videos", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-resume", action="store_true")

    args = parser.parse_args()

    stages = []
    for s in args.stage:
        if s in ("2clip", "2gpt", "overall"):
            stages.append(s)
        else:
            stages.append(int(s))

    # Only check API key for stages that need network
    api_stages = {"2gpt", 3}
    if any(s in api_stages for s in stages) and not args.dry_run:
        if not os.environ.get("OPENAI_API_KEY"):
            print("ERROR: OPENAI_API_KEY not set!")
            print("  Stages 2gpt and 3 require OpenAI API access.")
            print("  GPU-only stages (1, 2clip) do not need API key.")
            sys.exit(1)

    config = EvalConfig(
        videos_dir=args.videos_dir,
        output_dir=args.output_dir,
        stages=stages,
        single_question_file=args.single_question_file,
        multi_question_file=args.multi_question_file,
        video_question_file=args.video_question_file,
        explicit_prompts_file=args.explicit_prompts_file,
        prompts_file=args.prompts_file,
        vqascore_model=args.vqascore_model,
        clip_model=args.clip_model,
        gpt_model=args.gpt_model,
        max_videos=args.max_videos,
        resume=not args.no_resume,
        dry_run=args.dry_run,
    )
    PhyGenBenchEvaluator(config).run()


if __name__ == "__main__":
    main()
