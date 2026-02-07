"""
Batch Feature Extraction Script with Sharding Support

Extract DiT features from Physion or VideoPhy videos and save to disk.
Supports parallel extraction across multiple GPUs using sharding.

Pipeline: Video → VAE encode → Latents → Add noise → DiT forward → Features

Usage (Physion - single GPU):
    python -m src.models.extract_features \
        --data_dir data/Physion \
        --output_dir /users/$USER/scratch/physics/physion_features \
        --dataset physion \
        --layers 10 \
        --timesteps 200 400 600 800

Usage (VideoPhy - single GPU):
    python -m src.models.extract_features \
        --data_dir ~/scratch/physics/videophy_data \
        --output_dir ~/scratch/physics/videophy_features \
        --dataset videophy \
        --layers 10 \
        --timesteps 200 400 600 800 \
        --use-8bit

Usage (parallel with 2 GPUs):
    # Terminal 1 / SLURM job 1:
    python -m src.models.extract_features ... --shard 0 --num_shards 2
    
    # Terminal 2 / SLURM job 2:
    python -m src.models.extract_features ... --shard 1 --num_shards 2
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


def setup_models(
    model_id: str,
    layers: List[int],
    use_8bit: bool,
    device: str,
    use_text: bool = False,
):
    """Load VAE, DiT extractor, noise scheduler, and optionally T5 text encoder."""
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

    # Optionally load T5 text encoder for real caption embeddings
    text_encoder = None
    tokenizer = None
    if use_text:
        from transformers import T5EncoderModel, AutoTokenizer

        logger.info("Loading T5 text encoder for caption embeddings...")
        tokenizer = AutoTokenizer.from_pretrained(model_id, subfolder="tokenizer")
        text_encoder = T5EncoderModel.from_pretrained(
            model_id, subfolder="text_encoder", torch_dtype=torch.float16
        )
        text_encoder = text_encoder.to(device)
        text_encoder.eval()
        text_encoder.requires_grad_(False)
        logger.info("T5 text encoder loaded successfully")

    return vae, extractor, scheduler, text_encoder, tokenizer


def encode_video(video: torch.Tensor, vae, device: str) -> torch.Tensor:
    """
    Encode video frames to VAE latents.

    Args:
        video: [T, C, H, W] tensor in [0, 1] range
        vae: CogVideoX VAE model
        device: Target device

    Returns:
        latents: [1, C, T', H', W'] latent tensor
    """
    # video: [T, C, H, W] -> [1, T, C, H, W]
    video = video.unsqueeze(0).to(device=device, dtype=torch.float16)

    # Rearrange: [B, T, C, H, W] -> [B, C, T, H, W] for VAE
    video = video.permute(0, 2, 1, 3, 4)

    # Normalize from [0, 1] to [-1, 1]
    video = video * 2.0 - 1.0

    with torch.no_grad():
        latents = vae.encode(video).latent_dist.sample()
        latents = latents * vae.config.scaling_factor

    # latents shape: [1, C, T', H', W'] -> [1, T', C, H', W']
    latents = latents.permute(0, 2, 1, 3, 4)

    return latents

def encode_caption(
    caption: str,
    tokenizer,
    text_encoder,
    device: str,
    max_length: int = 226,
) -> torch.Tensor:
    """
    Encode a caption string into T5 text embeddings for CogVideoX.

    Args:
        caption: Text caption string
        tokenizer: T5 tokenizer
        text_encoder: T5 encoder model
        device: Target device
        max_length: Max token length (CogVideoX uses 226)

    Returns:
        text_embeds: [1, max_length, 4096] tensor
    """
    inputs = tokenizer(
        caption,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    input_ids = inputs.input_ids.to(device)

    with torch.no_grad():
        text_embeds = text_encoder(input_ids)[0]  # [1, seq_len, 4096]

    return text_embeds.to(torch.float16)


def pool_spatial_inline(features: torch.Tensor, num_frames: int = 13) -> torch.Tensor:
    """
    Pool spatial dims inline during extraction.

    CogVideoX-2B layer 15 shapes:
        Input:  [1, 17550, 1920]   (batch=1, patches=13*30*45, hidden=1920)
        Output: [13, 1920]         (T=13 latent frames, D=1920)

    Args:
        features: Raw DiT hidden states [1, T*H*W, D]
        num_frames: Temporal frames in latent space (13 for CogVideoX)

    Returns:
        [T, D] pooled tensor
    """
    if features.dim() == 3 and features.shape[0] == 1:
        features = features.squeeze(0)  # [1, 17550, 1920] -> [17550, 1920]

    if features.dim() == 2:
        num_patches, hidden_dim = features.shape
        spatial = num_patches // num_frames
        # [T*H*W, D] -> [T, H*W, D] -> mean(H*W) -> [T, D]
        return features.view(num_frames, spatial, hidden_dim).mean(dim=1)
    elif features.dim() == 3:
        return features.mean(dim=1)  # [T, H*W, D] -> [T, D]
    else:
        raise ValueError(f"pool_spatial_inline: unexpected shape {features.shape}")


def extract_single_video(
    video_id: str,
    video: torch.Tensor,
    label: int,
    vae,
    extractor,
    scheduler,
    timesteps: List[int],
    layers: List[int],
    output_dir: Path,
    device: str,
    text_embeds=None,
    pool: bool = False,
) -> tuple:
    """
    Extract features for a single video at multiple timesteps.

    Returns:
        info: Dict with extraction metadata
        forward_time: Time spent on DiT forward passes
    """
    video_dir = output_dir / video_id
    video_dir.mkdir(parents=True, exist_ok=True)

    info = {"label": label, "timesteps": {}}
    forward_time = 0.0

    # Encode video to latents
    latents = encode_video(video, vae, device)

    if text_embeds is None:
        text_embeds = torch.zeros(1, 226, 4096, device=device, dtype=torch.float16)

    for t in timesteps:
        t_dir = video_dir / f"t{t}"
        t_dir.mkdir(exist_ok=True)

        # Check if already extracted for this timestep
        if all((t_dir / f"layer_{l}.pt").exists() for l in layers):
            continue

        # Add noise at timestep t
        timestep_tensor = torch.tensor([t], device=device, dtype=torch.long)
        noise = torch.randn_like(latents)
        noisy_latents = scheduler.add_noise(latents, noise, timestep_tensor)

        # Reshape for DiT: [1, C, T, H, W] -> [1, T*H*W, C] (simplified)
        # Actually CogVideoX expects [B, C, T, H, W] directly
        batch_size, channels, num_frames, height, width = noisy_latents.shape

        # DiT forward with feature extraction
        torch.cuda.synchronize()
        forward_start = time.perf_counter()

        with torch.no_grad():
            features = extractor.extract(noisy_latents, timestep_tensor, text_embeds)

        torch.cuda.synchronize()
        forward_time += time.perf_counter() - forward_start

        # Save features for each layer (with optional inline pooling)
        layer_shapes = {}
        for layer_idx, feat in features.items():
            feat_path = t_dir / f"layer_{layer_idx}.pt"
            raw_shape = list(feat.shape)
            if pool:
                feat = pool_spatial_inline(feat.cpu(), num_frames=13).half()
            else:
                feat = feat.cpu()
            torch.save(feat, feat_path)
            layer_shapes[layer_idx] = list(feat.shape)

        # Debug: log shapes for the very first timestep of this video
        if not info["timesteps"]:
            for layer_idx in layer_shapes:
                raw = features[layer_idx].shape
                saved = layer_shapes[layer_idx]
                size_kb = (t_dir / f"layer_{layer_idx}.pt").stat().st_size / 1024
                logger.debug(
                    f"  [{video_id}] layer {layer_idx}: "
                    f"{list(raw)} -> {saved} "
                    f"({'pooled fp16' if pool else 'raw'}, {size_kb:.0f} KB)"
                )

        info["timesteps"][t] = {"layer_shapes": layer_shapes}

    return info, forward_time


def get_shard_indices(total_samples: int, shard: int, num_shards: int) -> List[int]:
    """
    Get indices for this shard.

    Example with 1200 videos and 2 shards:
        shard=0: indices [0, 2, 4, 6, ...]  (600 videos)
        shard=1: indices [1, 3, 5, 7, ...]  (600 videos)

    This interleaved approach ensures both shards process similar scenarios
    (since videos are typically sorted by scenario in labels.csv).
    """
    return list(range(shard, total_samples, num_shards))


def extract_dataset(
    data_dir: str,
    output_dir: str,
    model_id: str = "THUDM/CogVideoX-2b",
    dataset_type: str = "physion",  # NEW: dataset type parameter
    layers: List[int] = [5, 10, 15, 20, 25],
    timesteps: List[int] = [400],
    use_8bit: bool = True,
    use_text: bool = False,
    device: str = "cuda",
    max_videos: Optional[int] = None,
    resume: bool = True,
    shard: int = 0,
    num_shards: int = 1,
    pool: bool = False,
):
    """
    Extract features for all videos in dataset.

    Args:
        data_dir: Path to dataset directory
        output_dir: Output directory for features
        model_id: CogVideoX model ID
        dataset_type: "physion" or "videophy"
        layers: DiT layers to extract
        timesteps: Diffusion timesteps
        use_8bit: Use 8-bit quantization
        device: Target device
        max_videos: Limit number of videos (for debugging)
        resume: Skip already extracted videos
        shard: Shard index (0-indexed) for parallel extraction
        num_shards: Total number of shards (set to 2 for 2-GPU parallel)
    """
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load models (VAE + DiT + scheduler)
    vae, extractor, scheduler, text_encoder, tokenizer = setup_models(
            model_id, layers, use_8bit, device, use_text=use_text
        )

    # =========================================================================
    # Load dataset based on type (NEW: dataset selection logic)
    # =========================================================================
    if dataset_type == "physion":
        from src.data.physion_dataset import PhysionDataset

        logger.info("Loading Physion dataset...")
        dataset = PhysionDataset(
            data_root=str(data_dir),
            split="all",
            num_frames=49,
        )
    elif dataset_type == "videophy":
        from src.data.videophy_dataset import VideoPhyDataset

        logger.info("Loading VideoPhy dataset...")
        dataset = VideoPhyDataset(
            data_root=str(data_dir),
            split="all",
            num_frames=49,
        )
    else:
        raise ValueError(
            f"Unknown dataset type: {dataset_type}. Use 'physion' or 'videophy'."
        )

    # Apply sharding BEFORE any other filtering
    if num_shards > 1:
        original_count = len(dataset.samples)
        shard_indices = get_shard_indices(original_count, shard, num_shards)
        dataset.samples = [dataset.samples[i] for i in shard_indices]
        logger.info(
            f"Shard {shard}/{num_shards}: Processing {len(dataset.samples)}/{original_count} videos"
        )

    if max_videos is not None:
        dataset.samples = dataset.samples[:max_videos]
        logger.info(f"Limited to {max_videos} videos for debugging")


    # If use_text=False, use zeros (backward compatible)
    # If use_text=True, encode per-video captions below
    default_text_embeds = torch.zeros(1, 226, 4096, device=device, dtype=torch.float16)
    
    # Track progress (per-shard metadata)
    metadata = {
        "videos": {},
        "config": {
            "layers": layers,
            "timesteps": timesteps,
            "model_id": model_id,
            "use_text": use_text,
            "dataset_type": dataset_type,  # NEW: save dataset type in metadata
            "shard": shard,
            "num_shards": num_shards,
        },
    }
    skipped = 0
    total_forward_time = 0.0

    logger.info(f"Extracting features for {len(dataset)} videos...")
    logger.info(f"  Dataset: {dataset_type}")
    logger.info(f"  Layers: {layers}")
    logger.info(f"  Timesteps: {timesteps}")
    logger.info(f"  Output: {output_dir}")
    logger.info(f"  Pool inline: {pool}  (will save [13, 1920] fp16 ~100KB/file)")
    if num_shards > 1:
        logger.info(f"  Shard: {shard}/{num_shards}")

    for idx in tqdm(range(len(dataset)), desc=f"Extracting (shard {shard})"):
        sample = dataset[idx]
        video = sample["video"]
        video_id = sample["video_name"]
        label = sample["label"]

        # Check if already extracted (resume mode)
        if resume:
            video_dir = output_dir / video_id
            if video_dir.exists():
                all_done = all(
                    (video_dir / f"t{t}" / f"layer_{l}.pt").exists()
                    for t in timesteps
                    for l in layers
                )
                if all_done:
                    skipped += 1
                    continue

        try:
            # Encode caption if use_text is enabled
            if use_text and text_encoder is not None:
                caption = sample.get("caption", "")
                if caption:
                    text_embeds = encode_caption(
                        caption, tokenizer, text_encoder, device
                    )
                else:
                    text_embeds = default_text_embeds
            else:
                text_embeds = default_text_embeds

            info, fwd_time = extract_single_video(
                video_id=video_id,
                video=video,
                label=label,
                vae=vae,
                extractor=extractor,
                scheduler=scheduler,
                timesteps=timesteps,
                layers=layers,
                output_dir=output_dir,
                device=device,
                text_embeds=text_embeds,
                pool=pool,
            )
            metadata["videos"][video_id] = info
            # Save caption for enriched labels reconstruction
            if use_text:
                metadata["videos"][video_id]["caption"] = sample.get("caption", "")
            total_forward_time += fwd_time

        except Exception as e:
            logger.error(f"Failed to extract {video_id}: {e}")
            continue

    # Save metadata (per-shard)
    metadata_file = (
        output_dir / f"metadata_shard{shard}.json"
        if num_shards > 1
        else output_dir / "metadata.json"
    )
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)

    # Save labels (only if shard 0 or single-shard mode to avoid conflicts)
    if shard == 0:
        labels = {}
        # Load all existing metadata to merge labels
        for meta_file in output_dir.glob("metadata*.json"):
            with open(meta_file) as f:
                meta = json.load(f)
                for vid, info in meta.get("videos", {}).items():
                    labels[vid] = info.get("label", 0)

        with open(output_dir / "labels.json", "w") as f:
            json.dump(labels, f, indent=2)
            
        # Also save enriched labels if captions are available
        if use_text:
            enriched = {}
            for meta_file in output_dir.glob("metadata*.json"):
                with open(meta_file) as f:
                    meta = json.load(f)
                    for vid, info in meta.get("videos", {}).items():
                        enriched[vid] = {
                            "label": info.get("label", 0),
                            "caption": info.get("caption", ""),
                        }
            with open(output_dir / "enriched_labels.json", "w") as f:
                json.dump(enriched, f, indent=2)
            logger.info(f"Saved enriched_labels.json with captions")

    logger.info(f"Extraction complete!")
    logger.info(f"  Shard: {shard}/{num_shards}")
    logger.info(f"  Extracted: {len(metadata['videos'])} videos")
    logger.info(f"  Skipped (already done): {skipped} videos")
    logger.info(f"  Output: {output_dir}")
    logger.info(
        f"  Total DiT forward time: {total_forward_time:.2f}s ({total_forward_time / 60:.2f} min)"
    )
    if len(metadata["videos"]) > 0:
        avg_time = total_forward_time / len(metadata["videos"])
        logger.info(f"  Avg forward time per video: {avg_time:.2f}s")


def main():
    parser = argparse.ArgumentParser(
        description="Extract DiT features from video datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Physion dataset
    python -m src.models.extract_features \\
        --data_dir data/Physion \\
        --output_dir ~/scratch/physics/physion_features \\
        --dataset physion

    # VideoPhy dataset
    python -m src.models.extract_features \\
        --data_dir ~/scratch/physics/videophy_data \\
        --output_dir ~/scratch/physics/videophy_features \\
        --dataset videophy \\
        --use-8bit

    # Parallel extraction with 2 GPUs
    python -m src.models.extract_features ... --shard 0 --num_shards 2
    python -m src.models.extract_features ... --shard 1 --num_shards 2
        """,
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to dataset directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for extracted features",
    )
    # NEW: dataset type argument
    parser.add_argument(
        "--dataset",
        type=str,
        default="physion",
        choices=["physion", "videophy"],
        help="Dataset type: 'physion' or 'videophy' (default: physion)",
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="THUDM/CogVideoX-2b",
        help="CogVideoX model ID",
    )
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=[5, 10, 15, 20, 25],
        help="DiT layers to extract features from",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        nargs="+",
        default=[400],
        help="Diffusion timesteps to extract features at",
    )
    parser.add_argument(
        "--use-8bit",
        action="store_true",
        help="Use 8-bit quantization for DiT (saves memory)",
    )
    parser.add_argument(
        "--use-text",
        action="store_true",
        default=False,
        help="Encode real captions with T5 instead of zero embeddings. "
        "Produces text-conditioned DiT features (recommended for v2).",
    )
    parser.add_argument(
        "--max_videos",
        type=int,
        default=None,
        help="Maximum number of videos to process (for debugging)",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Don't skip already extracted videos",
    )
    parser.add_argument(
        "--shard",
        type=int,
        default=0,
        help="Shard index (0-indexed) for parallel extraction",
    )
    parser.add_argument(
        "--num_shards",
        type=int,
        default=1,
        help="Total number of shards. Set to 2 for 2-GPU parallel extraction.",
    )

    parser.add_argument(
        "--pool",
        action="store_true",
        default=False,
        help="Pool features inline during extraction. "
        "Saves [13, 1920] fp16 (~100KB) instead of "
        "[1, 17550, 1920] fp32 (~67MB). "
        "Output is directly usable with --is_pooled in training.",
    )

    args = parser.parse_args()

    extract_dataset(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        model_id=args.model_id,
        dataset_type=args.dataset,  # NEW: pass dataset type
        layers=args.layers,
        timesteps=args.timesteps,
        use_8bit=args.use_8bit,
        use_text=args.use_text,
        max_videos=args.max_videos,
        resume=not args.no_resume,
        shard=args.shard,
        num_shards=args.num_shards,
        pool=args.pool,
    )


if __name__ == "__main__":
    main()
