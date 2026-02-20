#!/usr/bin/env python3
"""
Compute VQAScore for VideoPhy Training Videos

For each video, extracts keyframes and computes VQAScore(frame, caption)
as a proxy for visual quality / text-video alignment.

Output:
    {output}/vqascores.json
    {
        "video_filename_1": {"score": 0.72, "caption": "..."},
        "video_filename_2": {"score": 0.45, "caption": "..."},
        ...
    }

Usage:
    python utils/compute_vqascore.py \
        --data-dir ~/scratch/physics/videophy_cogx \
        --output ~/scratch/physics/videophy_cogx/vqascores.json \
        --num-frames 4

    # Quick test with 10 videos
    python utils/compute_vqascore.py \
        --data-dir ~/scratch/physics/videophy_cogx \
        --output ~/scratch/physics/videophy_cogx/vqascores.json \
        --max-videos 10
"""

import argparse
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


def extract_frames(video_path: str, num_frames: int = 4) -> List[np.ndarray]:
    """Extract evenly spaced frames from video."""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames == 0:
        cap.release()
        return []

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


def compute_vqascores(
    data_dir: str,
    output_path: str,
    model_name: str = "clip-flant5-xxl",
    num_frames: int = 4,
    max_videos: Optional[int] = None,
    resume: bool = True,
):
    """
    Compute VQAScore for each video in the dataset.

    For each video, computes VQAScore(frame, caption) for multiple frames
    and takes the max score (best frame alignment).
    """
    data_dir = Path(data_dir)
    metadata_path = data_dir / "metadata.json"
    videos_dir = data_dir / "videos"

    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata.json not found in {data_dir}")

    # Load metadata
    with open(metadata_path) as f:
        metadata = json.load(f)

    logger.info(f"Loaded {len(metadata)} entries from metadata.json")

    # Load existing results if resuming
    existing = {}
    if resume and Path(output_path).exists():
        with open(output_path) as f:
            existing = json.load(f)
        logger.info(f"Resuming: {len(existing)} videos already scored")

    # Filter to videos that exist and need scoring
    to_process = []
    for entry in metadata:
        video_filename = entry.get("video_filename", "")
        caption = entry.get("caption", "")
        physics = entry.get("physics", -1)

        if not video_filename or not caption:
            continue

        if video_filename in existing:
            continue

        # Find video file
        video_path = None
        for ext in [".mp4", ""]:
            candidate = videos_dir / f"{video_filename}{ext}"
            if candidate.exists():
                video_path = str(candidate)
                break

        if video_path is None:
            continue

        to_process.append(
            {
                "video_filename": video_filename,
                "video_path": video_path,
                "caption": caption,
                "physics": physics,
            }
        )

    if max_videos:
        to_process = to_process[:max_videos]

    logger.info(f"Videos to process: {len(to_process)}")

    if len(to_process) == 0:
        logger.info("Nothing to process. Done.")
        return

    # Load VQAScore model
    logger.info(f"Loading VQAScore model: {model_name}")
    import t2v_metrics

    model = t2v_metrics.VQAScore(model=model_name)
    logger.info("VQAScore model loaded.")

    # Process videos
    results = dict(existing)  # Start with existing results
    errors = 0

    for entry in tqdm(to_process, desc="Computing VQAScore"):
        video_filename = entry["video_filename"]
        video_path = entry["video_path"]
        caption = entry["caption"]

        try:
            frames = extract_frames(video_path, num_frames)
            if not frames:
                logger.warning(f"No frames extracted: {video_filename}")
                errors += 1
                continue

            # Compute VQAScore for each frame, take max
            frame_scores = []
            for frame in frames:
                from PIL import Image

                img = Image.fromarray(frame)

                with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                    img.save(tmp.name, "JPEG", quality=95)
                    tmp_path = tmp.name

                try:
                    score = model([tmp_path], [caption]).item()
                    frame_scores.append(score)
                finally:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)

            if frame_scores:
                results[video_filename] = {
                    "score_max": float(max(frame_scores)),
                    "score_mean": float(np.mean(frame_scores)),
                    "score_per_frame": [float(s) for s in frame_scores],
                    "caption": caption,
                    "physics": entry["physics"],
                }

        except Exception as e:
            logger.warning(f"Error processing {video_filename}: {e}")
            errors += 1
            continue

        # Periodic save (every 50 videos)
        if len(results) % 50 == 0:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(results, f, indent=2)

    # Final save
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    # Summary
    scored = [v for v in results.values() if "score_max" in v]
    if scored:
        scores = [v["score_max"] for v in scored]
        logger.info(f"\nResults:")
        logger.info(f"  Scored: {len(scored)} videos")
        logger.info(f"  Errors: {errors}")
        logger.info(f"  Score range: [{min(scores):.4f}, {max(scores):.4f}]")
        logger.info(f"  Score mean:  {np.mean(scores):.4f}")
        logger.info(f"  Score std:   {np.std(scores):.4f}")

        # Score by physics label
        pos_scores = [v["score_max"] for v in scored if v.get("physics") == 1]
        neg_scores = [v["score_max"] for v in scored if v.get("physics") == 0]
        if pos_scores and neg_scores:
            logger.info(
                f"  physics=1 mean: {np.mean(pos_scores):.4f} (n={len(pos_scores)})"
            )
            logger.info(
                f"  physics=0 mean: {np.mean(neg_scores):.4f} (n={len(neg_scores)})"
            )
            logger.info(f"  Gap: {np.mean(pos_scores) - np.mean(neg_scores):.4f}")

    logger.info(f"\nSaved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Compute VQAScore for VideoPhy training videos"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="/users/ctang33/scratch/physics/videophy_cogx",
        help="Path to videophy_cogx data directory",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON path (default: {data-dir}/vqascores.json)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="clip-flant5-xxl",
        choices=["clip-flant5-xxl", "clip-flant5-xl"],
        help="VQAScore model",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=4,
        help="Number of frames to extract per video",
    )
    parser.add_argument(
        "--max-videos",
        type=int,
        default=None,
        help="Max videos to process (for testing)",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Don't resume from existing results",
    )
    args = parser.parse_args()

    output_path = args.output or str(Path(args.data_dir) / "vqascores.json")

    compute_vqascores(
        data_dir=args.data_dir,
        output_path=output_path,
        model_name=args.model,
        num_frames=args.num_frames,
        max_videos=args.max_videos,
        resume=not args.no_resume,
    )


if __name__ == "__main__":
    main()
