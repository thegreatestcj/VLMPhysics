"""
Batch Feature Extraction Script

Extract DiT features from Physion videos and save to disk.
Run this ONCE before training - then training is super fast!

Pipeline: Video → VAE encode → Latents → Add noise → DiT forward → Features

Usage:
    python -m src.models.extract_features \
        --data_dir data/Physion \
        --output_dir /users/$USER/scratch/physion_features \
        --layers 15 \
        --timesteps 400 \
        --use-8bit
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
    """Load VAE, DiT extractor, and noise scheduler."""
    from src.models.dit_extractor import create_extractor
    from diffusers import AutoencoderKLCogVideoX, CogVideoXDDIMScheduler

    # Load VAE for encoding videos to latents
    logger.info("Loading VAE...")
    vae = AutoencoderKLCogVideoX.from_pretrained(
        model_id, subfolder="vae", torch_dtype=torch.float16
    )
    vae = vae.to(device)
    vae.eval()
    vae.requires_grad_(False)

    # Load DiT extractor
    logger.info(f"Loading DiT extractor for layers {layers}...")
    extractor = create_extractor(
        model_id=model_id, layers=layers, use_8bit=use_8bit, device=device
    )
    extractor.load_model()

    # Load noise scheduler
    logger.info("Loading noise scheduler...")
    scheduler = CogVideoXDDIMScheduler.from_pretrained(model_id, subfolder="scheduler")

    return vae, extractor, scheduler


def encode_video(video: torch.Tensor, vae, device: str) -> torch.Tensor:
    """
    Encode video frames to VAE latents.

    Args:
        video: [T, C, H, W] in range [0, 1]
        vae: CogVideoX VAE

    Returns:
        latents: [1, T_latent, C_latent, H_latent, W_latent]
    """
    # CogVideoX VAE expects [B, C, T, H, W]
    # Input video is [T, C, H, W], need to transpose
    video = video.unsqueeze(0)  # [1, T, C, H, W]
    video = video.permute(0, 2, 1, 3, 4)  # [1, C, T, H, W]
    video = video.to(device, dtype=torch.float16)

    # Normalize from [0, 1] to [-1, 1]
    video = 2.0 * video - 1.0

    with torch.no_grad():
        latent_dist = vae.encode(video).latent_dist
        latents = latent_dist.sample()
        latents = latents * vae.config.scaling_factor

    # Output shape: [1, C, T_latent, H_latent, W_latent]
    # CogVideoX DiT expects [B, T, C, H, W]
    latents = latents.permute(0, 2, 1, 3, 4)  # [1, T, C, H, W]

    return latents


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
    video: torch.Tensor,
    vae,
    text_embeds: torch.Tensor,
    timesteps: List[int],
    extractor,
    scheduler,
    output_dir: Path,
    device: str,
    save_dtype: torch.dtype = torch.float16,
) -> tuple:
    """
    Extract features for a single video at multiple timesteps.

    Returns:
        (extraction_info, forward_time)
    """
    video_output_dir = output_dir / video_id
    forward_time = 0.0

    # Step 1: VAE encode video to latents
    latents = encode_video(video, vae, device)
    # latents shape: [1, T, C, H, W]

    extraction_info = {
        "video_id": video_id,
        "timesteps": {},
        "latent_shape": list(latents.shape),
        "extracted_at": datetime.now().isoformat(),
    }

    for t in timesteps:
        t_dir = video_output_dir / f"t{t}"
        t_dir.mkdir(parents=True, exist_ok=True)

        # Step 2: Add noise at timestep t
        timestep_tensor = torch.tensor([t], device=device)
        noisy_latents = add_noise(latents, t, scheduler, device)
        noisy_latents = noisy_latents.to(latents.dtype)

        # Step 3: Extract features (time this - it's the DiT forward pass)
        torch.cuda.synchronize()
        forward_start = time.perf_counter()
        features = extractor.extract(noisy_latents, timestep_tensor, text_embeds)
        torch.cuda.synchronize()
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
                video_feat = video_feat.squeeze(0)  # [T, h, w, D]

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
    max_videos: Optional[int] = None,
    resume: bool = True,
):
    """
    Extract features for Physion dataset.
    """
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load models (VAE + DiT + scheduler)
    vae, extractor, scheduler = setup_models(model_id, layers, use_8bit, device)

    # Load Physion dataset (returns raw videos)
    from src.data.physion_dataset import PhysionDataset

    logger.info("Loading Physion dataset...")
    dataset = PhysionDataset(
        data_root=str(data_dir),  # FIXED: was data_dir
        split="all",  # Use all data
        num_frames=49,  # CogVideoX default
    )

    if max_videos is not None:
        # Limit for debugging
        dataset.indices = dataset.indices[:max_videos]
        logger.info(f"Limited to {max_videos} videos for debugging")

    # Prepare dummy text embeddings (physics discrimination doesn't need real text)
    text_embeds = torch.zeros(1, 226, 4096, device=device, dtype=torch.float16)

    # Track progress
    metadata = {
        "videos": {},
        "config": {
            "layers": layers,
            "timesteps": timesteps,
            "model_id": model_id,
        },
    }
    skipped = 0
    total_forward_time = 0.0

    logger.info(f"Extracting features for {len(dataset)} videos...")
    logger.info(f"  Layers: {layers}")
    logger.info(f"  Timesteps: {timesteps}")
    logger.info(f"  Output: {output_dir}")

    for idx in tqdm(range(len(dataset)), desc="Extracting"):
        sample = dataset[idx]
        video = sample["video"]  # [T, C, H, W] in [0, 1]
        video_id = sample["video_name"]  # Use video_name as ID
        label = sample["label"]

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
                video=video,
                vae=vae,
                text_embeds=text_embeds,
                timesteps=timesteps,
                extractor=extractor,
                scheduler=scheduler,
                output_dir=output_dir,
                device=device,
            )

            total_forward_time += fwd_time
            info["label"] = label
            info["scenario"] = sample["scenario"]
            metadata["videos"][video_id] = info

        except Exception as e:
            logger.error(f"Failed to extract {video_id}: {e}")
            import traceback

            traceback.print_exc()
            continue

        # Periodically save metadata
        if (idx + 1) % 50 == 0:
            _save_metadata(output_dir, metadata)

    # Final save
    _save_metadata(output_dir, metadata)

    # Also save labels.json for trainer
    labels = {vid: info["label"] for vid, info in metadata["videos"].items()}
    with open(output_dir / "labels.json", "w") as f:
        json.dump(labels, f, indent=2)

    logger.info("Extraction complete!")
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
    parser = argparse.ArgumentParser(description="Extract DiT features from Physion")

    # Data arguments
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Physion data directory (contains labels.csv)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for features (use scratch!)",
    )

    # Extraction arguments
    parser.add_argument(
        "--layers", type=int, nargs="+", default=[15], help="DiT layers to extract"
    )
    parser.add_argument(
        "--timesteps", type=int, nargs="+", default=[400], help="Diffusion timesteps"
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
        max_videos=args.max_videos,
        resume=not args.no_resume,
    )


if __name__ == "__main__":
    main()
