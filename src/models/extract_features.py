"""
Batch Feature Extraction Script

Extract DiT features from Physion videos and save to disk.
Run this ONCE before training - then training is super fast!

Usage:
    python -m src.models.extract_features \
        --data_dir data/Physion \
        --output_dir data/Physion/features \
        --layers 15 \
        --timesteps 200 400 600 800 \
        --use-8bit

Output structure:
    output_dir/
    ├── video_001/
    │   ├── t200/
    │   │   └── layer_15.pt    # [T, h, w, D] tensor
    │   ├── t400/
    │   └── t800/
    ├── video_002/
    └── metadata.json
"""

import torch
import argparse
import json
import logging
import time
from pathlib import Path
from typing import List, Dict, Optional
from tqdm import tqdm
from datetime import datetime

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


def setup_models(model_id: str, layers: List[int], use_8bit: bool, device: str):
    """Load DiT extractor and noise scheduler."""
    from src.models.dit_extractor import create_extractor

    logger.info(f"Loading DiT extractor for layers {layers}...")
    extractor = create_extractor(
        model_id=model_id, layers=layers, use_8bit=use_8bit, device=device
    )
    extractor.load_model()

    # Load noise scheduler
    logger.info("Loading noise scheduler...")
    from diffusers import CogVideoXDDIMScheduler

    scheduler = CogVideoXDDIMScheduler.from_pretrained(model_id, subfolder="scheduler")

    return extractor, scheduler


def add_noise(
    latents: torch.Tensor, timestep: int, scheduler, device: str
) -> torch.Tensor:
    """Add noise to latents at specified timestep."""
    noise = torch.randn_like(latents)
    timestep_tensor = torch.tensor([timestep], device=device)
    noisy_latents = scheduler.add_noise(latents, noise, timestep_tensor)
    return noisy_latents


def extract_single_video(
    video_id: str,
    latents: torch.Tensor,
    text_embeds: torch.Tensor,
    timesteps: List[int],
    extractor,
    scheduler,
    output_dir: Path,
    save_dtype: torch.dtype = torch.float16,
) -> Dict:
    """
    Extract features for a single video at multiple timesteps.

    Args:
        video_id: Video identifier
        latents: Clean VAE latents [T, C, H, W] or [1, T, C, H, W]
        text_embeds: Text embeddings [1, 226, 4096]
        timesteps: List of diffusion timesteps
        extractor: DiTFeatureExtractor instance
        scheduler: Noise scheduler
        output_dir: Where to save features
        save_dtype: Data type for saving

    Returns:
        Tuple of (extraction info dict, forward time in seconds)
    """
    device = extractor.config.device
    video_output_dir = output_dir / video_id
    forward_time = 0.0  # Track DiT forward time only

    # Ensure batch dimension: [T, C, H, W] -> [1, T, C, H, W]
    if latents.dim() == 4:
        latents = latents.unsqueeze(0)

    latents = latents.to(device)
    text_embeds = text_embeds.to(device)

    extraction_info = {
        "video_id": video_id,
        "timesteps": {},
        "latent_shape": list(latents.shape),
        "extracted_at": datetime.now().isoformat(),
    }

    for t in timesteps:
        t_dir = video_output_dir / f"t{t}"
        t_dir.mkdir(parents=True, exist_ok=True)

        # Add noise at timestep t
        timestep_tensor = torch.tensor([t], device=device)
        noisy_latents = add_noise(latents, t, scheduler, device)
        noisy_latents = noisy_latents.to(latents.dtype)

        # Extract features (time this - it's the DiT forward pass)
        torch.cuda.synchronize()  # Ensure previous ops complete
        forward_start = time.perf_counter()
        features = extractor.extract(noisy_latents, timestep_tensor, text_embeds)
        torch.cuda.synchronize()  # Wait for forward to complete
        forward_time += time.perf_counter() - forward_start

        # Get video shape for reshaping
        T, h, w = extractor.get_video_shape(noisy_latents.shape)
        video_seq_len = T * h * w

        # Save each layer
        layer_info = {}
        for layer_idx, feat in features.items():
            # Handle text+video vs video-only output
            seq_len = feat.shape[1]
            if seq_len == video_seq_len:
                video_feat = feat
            elif seq_len == 226 + video_seq_len:
                video_feat = feat[:, 226:, :]  # Skip text tokens
            else:
                logger.warning(f"Unexpected seq_len {seq_len} for {video_id} t={t}")
                video_feat = feat

            # Reshape to 3D: [1, seq, D] -> [T, h, w, D]
            if video_feat.shape[1] == video_seq_len:
                video_feat = video_feat.view(1, T, h, w, -1)
                video_feat = video_feat.squeeze(0)  # Remove batch -> [T, h, w, D]

            # Save
            save_path = t_dir / f"layer_{layer_idx}.pt"
            torch.save(video_feat.cpu().to(save_dtype), save_path)

            layer_info[layer_idx] = {
                "shape": list(video_feat.shape),
                "dtype": str(save_dtype),
            }

        extraction_info["timesteps"][t] = layer_info

    return extraction_info, forward_time


def extract_dataset(
    data_dir: str,
    output_dir: str,
    layers: List[int],
    timesteps: List[int],
    model_id: str = "THUDM/CogVideoX-2b",
    use_8bit: bool = True,
    device: str = "cuda",
    split: str = "train",
    max_videos: Optional[int] = None,
    resume: bool = True,
):
    """
    Extract features for entire Physion dataset.
    """
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load models
    extractor, scheduler = setup_models(model_id, layers, use_8bit, device)

    # Load dataset
    from src.data.physion_dataset import PhysionDataset

    dataset = PhysionDataset(
        data_dir=str(data_dir),
        split=split,
        use_precomputed_latents=True,
        max_samples=max_videos,
    )

    # Prepare dummy text embeddings
    # For physics discrimination, text content doesn't matter much
    text_embeds = torch.zeros(1, 226, 4096, device=device, dtype=torch.float16)

    # Track progress
    metadata = {
        "videos": {},
        "config": {
            "layers": layers,
            "timesteps": timesteps,
            "model_id": model_id,
            "split": split,
        },
    }
    skipped = 0
    total_forward_time = 0.0  # Track DiT forward time only

    logger.info(f"Extracting features for {len(dataset)} videos...")
    logger.info(f"  Layers: {layers}")
    logger.info(f"  Timesteps: {timesteps}")

    for idx in tqdm(range(len(dataset)), desc=f"Extracting ({split})"):
        sample = dataset[idx]
        video_id = sample["video_id"]
        latents = sample["latents"]

        # Check if already extracted (resume mode)
        if resume:
            video_dir = output_dir / video_id
            if video_dir.exists():
                all_done = all(
                    (video_dir / f"t{t}" / f"layer_{layers[0]}.pt").exists()
                    for t in timesteps
                )
                if all_done:
                    skipped += 1
                    continue

        try:
            info, fwd_time = extract_single_video(
                video_id=video_id,
                latents=latents,
                text_embeds=text_embeds,
                timesteps=timesteps,
                extractor=extractor,
                scheduler=scheduler,
                output_dir=output_dir,
            )

            total_forward_time += fwd_time

            # Add split info
            info["split"] = split
            metadata["videos"][video_id] = info

        except Exception as e:
            logger.error(f"Failed to extract {video_id}: {e}")
            continue

        # Periodically save metadata
        if (idx + 1) % 50 == 0:
            _save_metadata(output_dir, metadata)

    # Final save
    _save_metadata(output_dir, metadata)

    logger.info(f"Extraction complete!")
    logger.info(f"  Extracted: {len(metadata['videos'])} videos")
    logger.info(f"  Skipped: {skipped} videos")
    logger.info(f"  Output: {output_dir}")
    logger.info(
        f"  Total DiT forward time: {total_forward_time:.2f}s ({total_forward_time / 60:.2f} min)"
    )
    if len(metadata["videos"]) > 0:
        avg_time = total_forward_time / len(metadata["videos"])
        logger.info(f"  Avg forward time per video: {avg_time:.2f}s")


def _save_metadata(output_dir: Path, metadata: Dict):
    """Save metadata.json."""
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Extract DiT features")

    # Data arguments
    parser.add_argument(
        "--data_dir", type=str, required=True, help="Physion data directory"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Output directory for features"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "test"],
        help="Dataset split",
    )

    # Extraction arguments
    parser.add_argument(
        "--layers", type=int, nargs="+", default=[15], help="DiT layers to extract"
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        nargs="+",
        default=[200, 400, 600, 800],
        help="Diffusion timesteps",
    )

    # Model arguments
    parser.add_argument("--model_id", type=str, default="THUDM/CogVideoX-2b")
    parser.add_argument(
        "--use-8bit", action="store_true", help="Use 8-bit quantization"
    )
    parser.add_argument("--device", type=str, default="cuda")

    # Other
    parser.add_argument(
        "--max_videos", type=int, default=None, help="Limit videos (for debugging)"
    )
    parser.add_argument(
        "--no-resume", action="store_true", help="Don't skip already extracted"
    )

    args = parser.parse_args()

    extract_dataset(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        layers=args.layers,
        timesteps=args.timesteps,
        model_id=args.model_id,
        use_8bit=args.use_8bit,
        device=args.device,
        split=args.split,
        max_videos=args.max_videos,
        resume=not args.no_resume,
    )


if __name__ == "__main__":
    main()
