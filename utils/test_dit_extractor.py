"""
Test script for DiT Feature Extractor

Usage:
    # Mock test (no GPU/model needed)
    python -m utils.test_dit_extractor --mock

    # Real test with single layer
    python -m utils.test_dit_extractor --layers 15 --use-8bit

    # Real test with multiple layers
    python -m utils.test_dit_extractor --layers 10 15 20 --use-8bit

    # All layers
    python -m utils.test_dit_extractor --all-layers --use-8bit
"""

import torch
import argparse
import logging
import time
from typing import Dict, List, Tuple, Optional

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


# =============================================================================
# Test Utilities
# =============================================================================


def get_dummy_inputs(
    batch_size: int = 1,
    num_frames: int = 13,
    height: int = 60,
    width: int = 90,
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Create dummy inputs matching CogVideoX expected shapes.

    CogVideoX expects latent shape: [B, T, C, H, W]  (NOT [B, C, T, H, W]!)
    - B: batch size
    - T: temporal frames = 13 (49 video frames compressed 4x by VAE)
    - C: channels = 16 (CogVideoX VAE latent channels)
    - H: height = 60 (480 pixels compressed 8x)
    - W: width = 90 (720 pixels compressed 8x)

    Text embeddings: [B, 226, 4096] from T5 encoder
    """
    # IMPORTANT: CogVideoX uses [B, T, C, H, W], not [B, C, T, H, W]
    latents = torch.randn(
        batch_size,
        num_frames,
        16,
        height,
        width,  # [B, T, C, H, W]
        device=device,
        dtype=dtype,
    )
    timestep = torch.randint(0, 1000, (batch_size,), device=device)
    text_embeds = torch.randn(batch_size, 226, 4096, device=device, dtype=dtype)
    return latents, timestep, text_embeds


# =============================================================================
# Mock Test (No GPU/Model needed)
# =============================================================================


def test_mock():
    """
    Test extractor logic without loading real model.
    """
    logger.info("=" * 60)
    logger.info("Running MOCK test (no GPU/model needed)")
    logger.info("=" * 60)

    from src.models.dit_extractor import DiTFeatureExtractor, ExtractorConfig

    # Test 1: Config parsing
    logger.info(f"\n[1] Config Test")
    config = ExtractorConfig(
        model_id="THUDM/CogVideoX-2b", extract_layers=[10, 15, 20], device="cpu"
    )
    extractor = DiTFeatureExtractor(config)

    assert extractor.num_layers == 30, (
        f"2B should have 30 layers, got {extractor.num_layers}"
    )
    assert extractor.hidden_dim == 1920, (
        f"2B should have hidden_dim 1920, got {extractor.hidden_dim}"
    )
    assert extractor.extract_layer_indices == [10, 15, 20]
    logger.info("  ✓ Config parsed correctly")
    logger.info(f"    - num_layers: {extractor.num_layers}")
    logger.info(f"    - hidden_dim: {extractor.hidden_dim}")
    logger.info(f"    - extract_layers: {extractor.extract_layer_indices}")

    # Test 2: Layer validation
    logger.info(f"\n[2] Layer Validation Test")
    try:
        bad_config = ExtractorConfig(
            extract_layers=[50]
        )  # Out of range for 2B (30 layers)
        bad_extractor = DiTFeatureExtractor(bad_config)
        logger.error("  ✗ Should have raised ValueError")
    except ValueError as e:
        logger.info(f"  ✓ Correctly rejected invalid layer")

    # Test 3: Video shape calculation
    logger.info(f"\n[3] Video Shape Test")
    # CogVideoX latent shape: [B, T, C, H, W] = [1, 13, 16, 60, 90]
    latent_shape = (1, 13, 16, 60, 90)  # [B, T, C, H, W]
    T, h, w = extractor.get_video_shape(latent_shape)

    expected = (13, 30, 45)  # T unchanged, H/2, W/2
    assert (T, h, w) == expected, f"Expected {expected}, got {(T, h, w)}"
    logger.info(f"  ✓ get_video_shape: {latent_shape} → (T={T}, h={h}, w={w})")

    # Test 4: Reshape test
    logger.info(f"\n[4] Reshape Test")
    video_seq_len = T * h * w  # 13 * 30 * 45 = 17550
    dummy_video_feat = torch.randn(1, video_seq_len, 1920)
    reshaped = extractor.reshape_video_features(dummy_video_feat, T, h, w)

    expected_shape = (1, 13, 30, 45, 1920)
    assert reshaped.shape == expected_shape, (
        f"Expected {expected_shape}, got {reshaped.shape}"
    )
    logger.info(f"  ✓ reshape: [1, {video_seq_len}, 1920] → {list(reshaped.shape)}")

    # Test 5: 5B model config
    logger.info(f"\n[5] 5B Model Config Test")
    config_5b = ExtractorConfig(model_id="THUDM/CogVideoX-5b", extract_layers=[21])
    extractor_5b = DiTFeatureExtractor(config_5b)
    assert extractor_5b.num_layers == 42, (
        f"5B should have 42 layers, got {extractor_5b.num_layers}"
    )
    assert extractor_5b.hidden_dim == 3072, (
        f"5B should have hidden_dim 3072, got {extractor_5b.hidden_dim}"
    )
    logger.info(
        f"  ✓ 5B config: {extractor_5b.num_layers} layers, hidden_dim={extractor_5b.hidden_dim}"
    )

    logger.info("\n" + "=" * 60)
    logger.info("✓ All mock tests passed!")
    logger.info("=" * 60)


# =============================================================================
# Real Test (Requires GPU + Model)
# =============================================================================


def test_real(layers: List[int], use_8bit: bool = False, device: str = "cuda"):
    """
    Test extractor with real CogVideoX model.
    """
    logger.info("=" * 60)
    logger.info("Running REAL test (requires GPU + model)")
    logger.info("=" * 60)

    from src.models.dit_extractor import create_extractor

    # Check CUDA
    if device == "cuda" and not torch.cuda.is_available():
        logger.error("CUDA not available! Use --mock for CPU testing.")
        return False

    if device == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(
            f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
        )

    # Step 1: Create extractor
    logger.info(f"\n[1] Creating extractor for layers: {layers}")
    extractor = create_extractor(
        model_id="THUDM/CogVideoX-2b", layers=layers, use_8bit=use_8bit, device=device
    )
    logger.info(f"  Extract layers: {extractor.extract_layer_indices}")
    logger.info(f"  Hidden dim: {extractor.hidden_dim}")

    # Step 2: Load model
    logger.info(f"\n[2] Loading model...")
    start_time = time.time()
    extractor.load_model()
    load_time = time.time() - start_time
    logger.info(f"  ✓ Loaded in {load_time:.1f}s")

    if device == "cuda":
        vram_used = torch.cuda.memory_allocated() / 1e9
        logger.info(f"  VRAM used: {vram_used:.2f} GB")

    # Step 3: Create dummy inputs
    logger.info(f"\n[3] Creating dummy inputs...")
    latents, timestep, text_embeds = get_dummy_inputs(device=device)
    logger.info(f"  Latents shape: {latents.shape}  # [B, T, C, H, W]")
    logger.info(f"  Timestep: {timestep.item()}")
    logger.info(f"  Text embeds shape: {text_embeds.shape}")

    # Step 4: Extract features
    logger.info(f"\n[4] Extracting features...")
    start_time = time.time()
    features = extractor.extract(latents, timestep, text_embeds)
    extract_time = time.time() - start_time
    logger.info(f"  ✓ Extracted in {extract_time:.3f}s")

    # Step 5: Validate features
    logger.info(f"\n[5] Validating extracted features...")
    T, h, w = extractor.get_video_shape(latents.shape)
    video_seq_len = T * h * w  # Expected: 13 * 30 * 45 = 17550
    text_len = 226

    logger.info(f"  Expected video_seq_len: {video_seq_len} ({T}×{h}×{w})")

    for layer_idx in layers:
        feat = features[layer_idx]
        seq_len = feat.shape[1]

        # CogVideoX intermediate layers may output:
        # - video only: [B, 17550, D]
        # - text + video: [B, 17776, D]
        if seq_len == video_seq_len:
            logger.info(f"  Layer {layer_idx}: {feat.shape} (video only)")
        elif seq_len == text_len + video_seq_len:
            logger.info(f"  Layer {layer_idx}: {feat.shape} (text + video)")
        else:
            logger.warning(f"  Layer {layer_idx}: {feat.shape} (unexpected seq_len!)")

    # Step 6: Test stacked output
    if len(layers) > 1:
        logger.info(f"\n[6] Testing stacked output...")
        stacked = extractor.extract_stacked(latents, timestep, text_embeds)
        logger.info(f"  Stacked shape: {stacked.shape}  # [B, num_layers, seq, D]")

    # Step 7: Test reshape to 3D
    logger.info(f"\n[7] Testing reshape to 3D spatiotemporal...")
    first_layer = layers[0]
    feat = features[first_layer]
    seq_len = feat.shape[1]

    # Handle both cases: video-only or text+video
    if seq_len == video_seq_len:
        # CogVideoX intermediate layer outputs video tokens only
        video_feat = feat
        logger.info(f"  Features are video-only: {feat.shape}")
    elif seq_len == text_len + video_seq_len:
        # Has both text and video tokens
        video_feat = feat[:, text_len:, :]  # Skip first 226 (text)
        logger.info(f"  Features have text+video, extracted video: {video_feat.shape}")
    else:
        logger.error(f"  Unexpected sequence length: {seq_len}")
        return False

    # Reshape to [B, T, h, w, D]
    video_3d = extractor.reshape_video_features(video_feat, T, h, w)
    expected_3d_shape = (1, T, h, w, extractor.hidden_dim)

    assert video_3d.shape == expected_3d_shape, (
        f"Shape mismatch: {video_3d.shape} != {expected_3d_shape}"
    )
    logger.info(f"  Video 3D: {video_3d.shape}  # [B, T, h, w, D]")
    logger.info(f"  ✓ Reshape successful")

    # Step 8: Cleanup
    logger.info(f"\n[8] Cleanup...")
    extractor.unload()
    if device == "cuda":
        torch.cuda.empty_cache()
        vram_after = torch.cuda.memory_allocated() / 1e9
        logger.info(f"  VRAM after unload: {vram_after:.2f} GB")

    logger.info("\n" + "=" * 60)
    logger.info("✓ All real tests passed!")
    logger.info("=" * 60)

    return True


def test_all_layers(use_8bit: bool = False, device: str = "cuda"):
    """Test extracting from all 30 layers."""
    logger.info("=" * 60)
    logger.info("Running ALL-LAYERS test")
    logger.info("=" * 60)

    from src.models.dit_extractor import create_extractor_all_layers

    if device == "cuda" and not torch.cuda.is_available():
        logger.error("CUDA not available!")
        return False

    # Create extractor for all layers
    logger.info("\n[1] Creating extractor for ALL layers...")
    extractor = create_extractor_all_layers(use_8bit=use_8bit, device=device)
    logger.info(f"  Total layers to extract: {len(extractor.extract_layer_indices)}")

    # Load model
    logger.info("\n[2] Loading model...")
    start = time.time()
    extractor.load_model()
    logger.info(f"  ✓ Loaded in {time.time() - start:.1f}s")

    if device == "cuda":
        logger.info(f"  VRAM: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    # Extract features
    logger.info("\n[3] Extracting from all layers...")
    latents, timestep, text_embeds = get_dummy_inputs(device=device)

    start = time.time()
    features = extractor.extract(latents, timestep, text_embeds)
    logger.info(f"  ✓ Extracted {len(features)} layers in {time.time() - start:.3f}s")

    # Show shapes
    logger.info("\n[4] Feature shapes per layer:")
    for layer_idx in sorted(features.keys()):
        logger.info(f"  Layer {layer_idx:2d}: {features[layer_idx].shape}")

    # Stacked output
    logger.info("\n[5] Stacked output:")
    stacked = extractor.extract_stacked(latents, timestep, text_embeds)
    logger.info(f"  Shape: {stacked.shape}  # [B, num_layers, seq, D]")

    # Memory info
    if device == "cuda":
        logger.info("\n[6] Memory analysis:")
        single_size = features[0].numel() * 2 / 1e6  # fp16 = 2 bytes
        total_size = single_size * len(features)
        logger.info(f"  Single layer: {single_size:.1f} MB")
        logger.info(f"  All layers:   {total_size:.1f} MB")

    extractor.unload()

    logger.info("\n" + "=" * 60)
    logger.info("✓ All-layers test passed!")
    logger.info("=" * 60)

    return True


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description="Test DiT Feature Extractor")

    parser.add_argument(
        "--mock", action="store_true", help="Run mock test (no GPU/model needed)"
    )
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=None,
        help="Layer indices to extract (default: [15])",
    )
    parser.add_argument(
        "--all-layers", action="store_true", help="Extract from all layers"
    )
    parser.add_argument(
        "--use-8bit", action="store_true", help="Use 8-bit quantization to save VRAM"
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device: cuda or cpu"
    )

    args = parser.parse_args()

    if args.mock:
        test_mock()
    elif args.all_layers:
        test_all_layers(use_8bit=args.use_8bit, device=args.device)
    else:
        layers = args.layers or [15]
        test_real(layers=layers, use_8bit=args.use_8bit, device=args.device)


if __name__ == "__main__":
    main()
