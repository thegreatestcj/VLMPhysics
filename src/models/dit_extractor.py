"""
DiT Feature Extractor for VLM-Physics

Extracts intermediate features from CogVideoX DiT backbone during diffusion sampling.
Supports multi-layer extraction for downstream physics causal attention modeling.

Usage:
    # Single layer (default: middle layer)
    python dit_extractor.py

    # Multiple layers
    python dit_extractor.py --layers 10 15 20

    # All layers (for analysis)
    python dit_extractor.py --all-layers

Author: VLM-Physics Project
Target: ECCV 2026 Submission
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, List, Dict, Union
from dataclasses import dataclass, field
import logging
import argparse

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class ExtractorConfig:
    """Configuration for DiT feature extraction."""

    model_id: str = "THUDM/CogVideoX-2b"
    extract_layers: List[int] = field(default_factory=lambda: [15])
    dtype: torch.dtype = torch.float16
    device: str = "cuda"
    use_8bit: bool = False

    def __post_init__(self):
        # Ensure extract_layers is a list
        if isinstance(self.extract_layers, int):
            self.extract_layers = [self.extract_layers]


# =============================================================================
# Feature Hook
# =============================================================================


class FeatureHook:
    """Forward hook to capture intermediate features from a single layer."""

    def __init__(self, layer_idx: int):
        self.layer_idx = layer_idx
        self.features: Optional[torch.Tensor] = None
        self._handle = None

    def __call__(self, module, input, output):
        feat = output[0] if isinstance(output, tuple) else output
        self.features = feat.detach()

    def register(self, module: nn.Module):
        self._handle = module.register_forward_hook(self)

    def remove(self):
        if self._handle:
            self._handle.remove()
            self._handle = None

    def clear(self):
        self.features = None


class MultiLayerHookManager:
    """Manages hooks for multiple layers."""

    def __init__(self):
        self.hooks: Dict[int, FeatureHook] = {}

    def register(self, layer_idx: int, module: nn.Module):
        """Register a hook on a specific layer."""
        hook = FeatureHook(layer_idx)
        hook.register(module)
        self.hooks[layer_idx] = hook

    def register_layers(
        self, layer_indices: List[int], transformer_blocks: nn.ModuleList
    ):
        """Register hooks on multiple layers."""
        for idx in layer_indices:
            self.register(idx, transformer_blocks[idx])

    def get_features(self) -> Dict[int, torch.Tensor]:
        """Get all captured features as a dict: {layer_idx: tensor}."""
        return {
            idx: hook.features
            for idx, hook in self.hooks.items()
            if hook.features is not None
        }

    def clear(self):
        """Clear all stored features."""
        for hook in self.hooks.values():
            hook.clear()

    def remove_all(self):
        """Remove all hooks."""
        for hook in self.hooks.values():
            hook.remove()
        self.hooks.clear()

    @property
    def layer_indices(self) -> List[int]:
        """Get list of hooked layer indices."""
        return sorted(self.hooks.keys())


# =============================================================================
# Main Extractor
# =============================================================================


class DiTFeatureExtractor:
    """
    Extract raw intermediate features from CogVideoX DiT.

    Supports extracting from multiple layers simultaneously.
    Returns features with full spatiotemporal structure for causal modeling.

    Usage:
        extractor = DiTFeatureExtractor(config)
        extractor.load_model()

        # Extract from multiple layers
        features = extractor.extract(latents, timestep, text_embeds)
        # features = {15: tensor[B, seq, D], 20: tensor[B, seq, D], ...}

        # Or get as stacked tensor
        stacked = extractor.extract_stacked(latents, timestep, text_embeds)
        # stacked = [B, num_layers, seq, D]
    """

    def __init__(self, config: Optional[ExtractorConfig] = None):
        self.config = config or ExtractorConfig()
        self.transformer = None
        self.hook_manager = MultiLayerHookManager()
        self._loaded = False

        # Model specs
        self._is_5b = "5b" in self.config.model_id.lower()
        self._num_layers = 42 if self._is_5b else 30
        self._hidden_dim = 3072 if self._is_5b else 1920

        # Validate layer indices
        for layer_idx in self.config.extract_layers:
            if not (0 <= layer_idx < self._num_layers):
                raise ValueError(
                    f"Layer {layer_idx} out of range [0, {self._num_layers})"
                )

    def load_model(self) -> None:
        """Load CogVideoX transformer and register hooks."""
        if self._loaded:
            return

        from diffusers import CogVideoXTransformer3DModel

        logger.info(f"Loading transformer from {self.config.model_id}...")

        load_kwargs = {"subfolder": "transformer", "torch_dtype": self.config.dtype}

        if self.config.use_8bit:
            try:
                from transformers import BitsAndBytesConfig

                load_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_8bit=True
                )
            except ImportError:
                logger.warning("bitsandbytes not available, using fp16")

        self.transformer = CogVideoXTransformer3DModel.from_pretrained(
            self.config.model_id, **load_kwargs
        )

        if not self.config.use_8bit:
            self.transformer = self.transformer.to(self.config.device)

        self.transformer.eval()
        for p in self.transformer.parameters():
            p.requires_grad = False

        # Register hooks on all specified layers
        self.hook_manager.register_layers(
            self.config.extract_layers, self.transformer.transformer_blocks
        )

        self._loaded = True
        logger.info(
            f"Loaded. Extracting from layers: {self.config.extract_layers} "
            f"(total {self._num_layers} layers)"
        )

    @torch.no_grad()
    def extract(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Dict[int, torch.Tensor]:
        """
        Extract raw features from multiple DiT layers.

        Args:
            hidden_states: Latent video [B, C, T, H, W]
            timestep: Diffusion timestep [B]
            encoder_hidden_states: Text embeddings [B, 226, 4096]
            image_rotary_emb: Optional 3D RoPE

        Returns:
            Dict mapping layer_idx -> features [B, seq_len, hidden_dim]
            where seq_len = 226 (text) + T*(H/2)*(W/2) (video)
        """
        if not self._loaded:
            raise RuntimeError("Call load_model() first")

        self.hook_manager.clear()

        _ = self.transformer(
            hidden_states=hidden_states,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            image_rotary_emb=image_rotary_emb,
            return_dict=False,
        )

        features = self.hook_manager.get_features()
        self.hook_manager.clear()

        return features

    @torch.no_grad()
    def extract_stacked(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Extract features and stack them into a single tensor.

        Returns:
            Tensor [B, num_layers, seq_len, hidden_dim]
            Layers are sorted in ascending order.
        """
        features_dict = self.extract(
            hidden_states, timestep, encoder_hidden_states, image_rotary_emb
        )

        # Stack in sorted layer order
        sorted_indices = sorted(features_dict.keys())
        stacked = torch.stack([features_dict[idx] for idx in sorted_indices], dim=1)

        return stacked

    def add_layer(self, layer_idx: int) -> None:
        """Dynamically add a layer to extract from."""
        if not (0 <= layer_idx < self._num_layers):
            raise ValueError(f"Layer {layer_idx} out of range")

        if layer_idx in self.hook_manager.hooks:
            logger.warning(f"Layer {layer_idx} already registered")
            return

        if self._loaded:
            self.hook_manager.register(
                layer_idx, self.transformer.transformer_blocks[layer_idx]
            )

        if layer_idx not in self.config.extract_layers:
            self.config.extract_layers.append(layer_idx)
            self.config.extract_layers.sort()

    def remove_layer(self, layer_idx: int) -> None:
        """Dynamically remove a layer from extraction."""
        if layer_idx in self.hook_manager.hooks:
            self.hook_manager.hooks[layer_idx].remove()
            del self.hook_manager.hooks[layer_idx]

        if layer_idx in self.config.extract_layers:
            self.config.extract_layers.remove(layer_idx)

    def set_layers(self, layer_indices: List[int]) -> None:
        """Replace all extraction layers with new set."""
        # Validate first
        for idx in layer_indices:
            if not (0 <= idx < self._num_layers):
                raise ValueError(f"Layer {idx} out of range")

        # Remove old hooks
        self.hook_manager.remove_all()

        # Update config
        self.config.extract_layers = sorted(layer_indices)

        # Re-register if loaded
        if self._loaded:
            self.hook_manager.register_layers(
                self.config.extract_layers, self.transformer.transformer_blocks
            )

    # =========================================================================
    # Utility methods
    # =========================================================================

    def get_video_shape(self, latent_shape: Tuple[int, ...]) -> Tuple[int, int, int]:
        """Get video token dimensions (T, h, w) from latent shape [B, C, T, H, W]."""
        _, T, _, H, W = latent_shape
        return T, H // 2, W // 2

    def reshape_video_features(
        self, video_features: torch.Tensor, T: int, h: int, w: int
    ) -> torch.Tensor:
        """Reshape [B, T*h*w, D] -> [B, T, h, w, D]."""
        B, seq, D = video_features.shape
        assert seq == T * h * w, f"Mismatch: {seq} != {T}*{h}*{w}={T * h * w}"
        return video_features.view(B, T, h, w, D)

    def split_text_video(
        self,
        features: torch.Tensor,
        text_len: int = 226,
        video_seq_len: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Split features into text and video parts.
        
        CogVideoX intermediate layers may or may not include text tokens.
        This method auto-detects based on sequence length.
        """
        seq_len = features.shape[-2]
        
        # If video_seq_len is provided, use it to detect
        if video_seq_len is not None:
            if seq_len == video_seq_len:
                # No text tokens, features is video-only
                return None, features
            elif seq_len == text_len + video_seq_len:
                # Has text tokens
                text_feat = features[..., :text_len, :]
                video_feat = features[..., text_len:, :]
                return text_feat, video_feat
            else:
                raise ValueError(
                    f"Unexpected seq_len={seq_len}, "
                    f"expected {video_seq_len} (video only) or {text_len + video_seq_len} (text+video)"
                )
        
        # Fallback: assume text is present
        text_feat = features[..., :text_len, :]
        video_feat = features[..., text_len:, :]
        return text_feat, video_feat

    @property
    def hidden_dim(self) -> int:
        return self._hidden_dim

    @property
    def num_layers(self) -> int:
        return self._num_layers

    @property
    def text_seq_len(self) -> int:
        return 226

    @property
    def extract_layer_indices(self) -> List[int]:
        return sorted(self.config.extract_layers)

    def unload(self) -> None:
        """Free memory."""
        self.hook_manager.remove_all()
        if self.transformer is not None:
            del self.transformer
            self.transformer = None
        self._loaded = False
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# =============================================================================
# Factory Functions
# =============================================================================


def create_extractor(
    model_id: str = "THUDM/CogVideoX-2b",
    layers: Optional[Union[int, List[int]]] = None,
    use_8bit: bool = False,
    device: str = "cuda",
) -> DiTFeatureExtractor:
    """
    Create a DiT feature extractor.

    Args:
        model_id: HuggingFace model ID
        layers: Layer(s) to extract from.
                - None: use optimal default (middle layer)
                - int: single layer
                - List[int]: multiple layers
        use_8bit: Use 8-bit quantization
        device: Target device
    """
    is_5b = "5b" in model_id.lower()
    num_layers = 42 if is_5b else 30

    if layers is None:
        # Default: middle layer
        layers = [num_layers // 2]
    elif isinstance(layers, int):
        layers = [layers]

    config = ExtractorConfig(
        model_id=model_id, extract_layers=layers, use_8bit=use_8bit, device=device
    )
    return DiTFeatureExtractor(config)


def create_extractor_all_layers(
    model_id: str = "THUDM/CogVideoX-2b", use_8bit: bool = False, device: str = "cuda"
) -> DiTFeatureExtractor:
    """Create extractor that captures ALL layers (for analysis)."""
    is_5b = "5b" in model_id.lower()
    num_layers = 42 if is_5b else 30
    all_layers = list(range(num_layers))

    return create_extractor(model_id, all_layers, use_8bit, device)


# =============================================================================
# CLI
# =============================================================================


def parse_args():
    parser = argparse.ArgumentParser(
        description="DiT Feature Extractor for CogVideoX",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default (middle layer)
  python dit_extractor.py
  
  # Specific layers
  python dit_extractor.py --layers 10 15 20
  
  # All layers (analysis mode)
  python dit_extractor.py --all-layers
  
  # With 8-bit quantization
  python dit_extractor.py --layers 15 --use-8bit
        """,
    )

    parser.add_argument(
        "--model",
        type=str,
        default="THUDM/CogVideoX-2b",
        help="Model ID (default: THUDM/CogVideoX-2b)",
    )
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=None,
        help="Layer indices to extract from (default: middle layer)",
    )
    parser.add_argument(
        "--all-layers",
        action="store_true",
        help="Extract from all layers (for analysis)",
    )
    parser.add_argument(
        "--use-8bit", action="store_true", help="Use 8-bit quantization to save memory"
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device (default: cuda)"
    )

    return parser.parse_args()


def main():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
    )

    args = parse_args()

    print("=" * 60)
    print("DiT Feature Extractor Test")
    print("=" * 60)

    # Determine layers
    if args.all_layers:
        extractor = create_extractor_all_layers(
            model_id=args.model, use_8bit=args.use_8bit, device=args.device
        )
    else:
        extractor = create_extractor(
            model_id=args.model,
            layers=args.layers,
            use_8bit=args.use_8bit,
            device=args.device,
        )

    print(f"\nConfiguration:")
    print(f"  Model: {extractor.config.model_id}")
    print(f"  Total layers: {extractor.num_layers}")
    print(f"  Extract layers: {extractor.extract_layer_indices}")
    print(f"  Hidden dim: {extractor.hidden_dim}")
    print(f"  8-bit: {extractor.config.use_8bit}")

    # Load model
    print("\nLoading model...")
    extractor.load_model()

    # Create dummy inputs
    print("\nCreating dummy inputs...")
    B, C, T, H, W = 1, 16, 13, 60, 90

    latents = torch.randn(B, C, T, H, W, device=args.device, dtype=torch.float16)
    timestep = torch.tensor([500], device=args.device)
    text_embeds = torch.randn(B, 226, 4096, device=args.device, dtype=torch.float16)

    print(f"  Latents: {latents.shape}")
    print(f"  Timestep: {timestep.item()}")
    print(f"  Text embeds: {text_embeds.shape}")

    # Extract features (dict)
    print("\nExtracting features...")
    features_dict = extractor.extract(latents, timestep, text_embeds)

    print(f"\nExtracted features (dict):")
    for layer_idx, feat in sorted(features_dict.items()):
        print(f"  Layer {layer_idx:2d}: {feat.shape}")

    # Extract features (stacked)
    if len(extractor.extract_layer_indices) > 1:
        stacked = extractor.extract_stacked(latents, timestep, text_embeds)
        print(f"\nStacked features: {stacked.shape}  # [B, num_layers, seq, D]")

    # Split and reshape demo
    print("\n" + "-" * 40)
    print("Split and reshape demo (using first extracted layer):")
    first_layer = extractor.extract_layer_indices[0]
    feat = features_dict[first_layer]

    text_feat, video_feat = extractor.split_text_video(feat)
    print(f"  Text features:  {text_feat.shape}")
    print(f"  Video features: {video_feat.shape}")

    t, h, w = extractor.get_video_shape(latents.shape)
    video_3d = extractor.reshape_video_features(video_feat, t, h, w)
    print(f"  Video 3D:       {video_3d.shape}  # [B, T, h, w, D]")

    # Dynamic layer modification demo
    print("\n" + "-" * 40)
    print("Dynamic layer modification demo:")
    original_layers = extractor.extract_layer_indices.copy()
    print(f"  Original layers: {original_layers}")

    # Add a layer
    new_layer = 5 if 5 not in original_layers else 25
    if new_layer < extractor.num_layers:
        extractor.add_layer(new_layer)
        print(f"  After add_layer({new_layer}): {extractor.extract_layer_indices}")

    # Remove a layer
    if len(extractor.extract_layer_indices) > 1:
        to_remove = extractor.extract_layer_indices[-1]
        extractor.remove_layer(to_remove)
        print(f"  After remove_layer({to_remove}): {extractor.extract_layer_indices}")

    # Set layers
    extractor.set_layers([10, 15, 20])
    print(f"  After set_layers([10,15,20]): {extractor.extract_layer_indices}")

    # Cleanup
    print("\nCleaning up...")
    extractor.unload()

    print("\n✓ All tests passed!")


if __name__ == "__main__":
    main()
