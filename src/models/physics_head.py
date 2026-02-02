"""
Physics Discriminator Heads with AdaLN Timestep Conditioning

This module implements physics discriminator heads using Adaptive Layer Normalization
(AdaLN) for timestep conditioning, following the design principles from DiT
(Diffusion Transformers).

Key Design Decision: AdaLN vs Concatenation
============================================

Previous approach (concat):
    features ──┬── attention ── concat([features, t_emb]) ── classifier
               │
    timestep ──┘ (only used at the end)

New approach (AdaLN):
    features ── AdaLN(t_emb) ── attention ── AdaLN(t_emb) ── classifier
                   ↑                            ↑
    timestep ──────┴────────────────────────────┘ (modulates every layer)

Why AdaLN?
----------
1. Timestep affects the entire computation, not just final classification
2. Features at t=200 (low noise) vs t=800 (high noise) need different processing
3. This is how DiT handles timestep, and our features come from DiT
4. AdaLN allows the same network to behave differently at different timesteps

AdaLN Formula:
    Standard LayerNorm: y = LayerNorm(x) * γ + β        (γ, β are learned constants)
    AdaLN:              y = LayerNorm(x) * (1 + scale) + shift   (scale, shift from t_emb)

Head Variants:
    1. MeanPool:         Global mean pooling baseline (no timestep)
    2. MeanPoolAdaLN:    Mean pooling + AdaLN timestep conditioning
    3. TemporalAdaLN:    Bidirectional attention with AdaLN
    4. CausalAdaLN:      Causal attention with AdaLN
    5. MultiViewAdaLN:   Multi-view pooling with AdaLN

Author: VLMPhysics Project
"""

import math
from typing import Optional, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Timestep Embedding
# =============================================================================


def get_timestep_embedding(timesteps: torch.Tensor, dim: int = 256) -> torch.Tensor:
    """
    Create sinusoidal timestep embeddings.

    Args:
        timesteps: [B] tensor of timestep values (e.g., 200, 400, 600, 800)
        dim: Embedding dimension

    Returns:
        [B, dim] sinusoidal embedding
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=timesteps.device) / half
    )
    args = timesteps[:, None].float() * freqs[None, :]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = F.pad(embedding, (0, 1))
    return embedding


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.

    Architecture: sinusoidal_embed → Linear → SiLU → Linear

    This follows DiT's design where timestep embeddings are projected
    to a higher dimension before being used for AdaLN modulation.
    """

    def __init__(self, hidden_size: int, frequency_dim: int = 256):
        """
        Args:
            hidden_size: Output dimension (should match model hidden size)
            frequency_dim: Dimension of sinusoidal embedding
        """
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.frequency_dim = frequency_dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Args:
            timesteps: [B] integer timesteps

        Returns:
            [B, hidden_size] timestep embeddings
        """
        t_freq = get_timestep_embedding(timesteps, self.frequency_dim)
        t_emb = self.mlp(t_freq)
        return t_emb


# =============================================================================
# Adaptive Layer Normalization (AdaLN)
# =============================================================================


class AdaLN(nn.Module):
    """
    Adaptive Layer Normalization.

    Instead of learning fixed scale (γ) and shift (β) parameters,
    AdaLN generates them dynamically from a conditioning signal (timestep).

    Formula:
        y = LayerNorm(x) * (1 + scale) + shift

        where scale, shift = Linear(t_emb)

    The (1 + scale) formulation ensures that when scale ≈ 0 (at initialization),
    the layer behaves like standard LayerNorm.
    """

    def __init__(self, hidden_size: int, conditioning_size: int):
        """
        Args:
            hidden_size: Dimension of input features
            conditioning_size: Dimension of conditioning signal (t_emb)
        """
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False)

        # Project conditioning to scale and shift
        # Output: [scale, shift] each of dimension hidden_size
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(conditioning_size, 2 * hidden_size),
        )

        # Initialize to output near-zero, so initially behaves like standard LayerNorm
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, ..., hidden_size] input features
            t_emb: [B, conditioning_size] timestep embedding

        Returns:
            [B, ..., hidden_size] modulated features
        """
        # Generate scale and shift from timestep
        modulation = self.adaLN_modulation(t_emb)  # [B, 2 * hidden_size]
        scale, shift = modulation.chunk(2, dim=-1)  # [B, hidden_size] each

        # Expand for broadcasting if x has more dimensions
        # x: [B, T, D] or [B, D], t_emb: [B, D]
        while scale.dim() < x.dim():
            scale = scale.unsqueeze(1)  # [B, 1, D]
            shift = shift.unsqueeze(1)  # [B, 1, D]

        # Apply adaptive normalization
        x = self.norm(x)
        x = x * (1 + scale) + shift

        return x


class AdaLNZero(nn.Module):
    """
    AdaLN-Zero: AdaLN with additional gate for residual connections.

    Used in DiT for gating the output of attention and FFN blocks.

    Formula:
        y = LayerNorm(x) * (1 + scale) + shift
        output = gate * y  (applied to block output before residual add)

    The gate starts at 0, meaning the block initially does nothing,
    allowing for stable training of deep networks.
    """

    def __init__(self, hidden_size: int, conditioning_size: int):
        """
        Args:
            hidden_size: Dimension of input features
            conditioning_size: Dimension of conditioning signal
        """
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False)

        # Project conditioning to scale, shift, and gate
        # Output: [scale, shift, gate] each of dimension hidden_size
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(conditioning_size, 3 * hidden_size),
        )

        # Initialize to zero for stable training
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(
        self, x: torch.Tensor, t_emb: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, ..., hidden_size] input features
            t_emb: [B, conditioning_size] timestep embedding

        Returns:
            normalized: [B, ..., hidden_size] normalized and modulated features
            gate: [B, ..., hidden_size] gate values for residual connection
        """
        modulation = self.adaLN_modulation(t_emb)  # [B, 3 * hidden_size]
        scale, shift, gate = modulation.chunk(3, dim=-1)

        # Expand for broadcasting
        while scale.dim() < x.dim():
            scale = scale.unsqueeze(1)
            shift = shift.unsqueeze(1)
            gate = gate.unsqueeze(1)

        # Normalize and modulate
        normalized = self.norm(x) * (1 + scale) + shift

        return normalized, gate


# =============================================================================
# Helper Functions
# =============================================================================


def _init_weights(module: nn.Module):
    """Initialize weights with truncated normal."""
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


# =============================================================================
# Head 1: MeanPool (Baseline, No Timestep)
# =============================================================================


class MeanPool(nn.Module):
    """
    Simplest baseline: Global mean pooling without timestep conditioning.

    This serves as the lower bound for ablation studies.

    Pipeline:
        [B, T, H, W, D] → mean(T, H, W) → [B, D] → MLP → [B, 1]

    Parameters: ~1.0M
    """

    def __init__(
        self,
        hidden_dim: int = 1920,
        mlp_dim: int = 512,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, mlp_dim),
            nn.LayerNorm(mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, mlp_dim // 2),
            nn.GELU(),
            nn.Linear(mlp_dim // 2, 1),
        )
        _init_weights(self)

    def forward(
        self,
        features: torch.Tensor,
        timestep: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            features: [B, T, H, W, D] DiT features
            timestep: ignored

        Returns:
            [B, 1] logits
        """
        if features.dim() == 5:
            x = features.mean(dim=(1, 2, 3))
        elif features.dim() == 3:
            x = features.mean(dim=1)
        else:
            x = features
        return self.classifier(x)

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# =============================================================================
# Head 2: MeanPoolAdaLN (Mean Pooling + AdaLN)
# =============================================================================


class MeanPoolAdaLN(nn.Module):
    """
    Mean pooling with AdaLN timestep conditioning.

    The timestep modulates the pooled features before classification,
    allowing the classifier to interpret features differently based on noise level.

    Pipeline:
        [B, T, H, W, D] → mean(T, H, W) → [B, D]
        timestep → TimestepEmbedder → [B, D]
        AdaLN(features, t_emb) → MLP → [B, 1]

    Parameters: ~1.5M
    """

    def __init__(
        self,
        hidden_dim: int = 1920,
        mlp_dim: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Timestep embedding
        self.t_embedder = TimestepEmbedder(hidden_dim)

        # AdaLN for feature modulation
        self.adaln = AdaLN(hidden_dim, hidden_dim)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, mlp_dim // 2),
            nn.GELU(),
            nn.Linear(mlp_dim // 2, 1),
        )

        _init_weights(self)

    def forward(
        self,
        features: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            features: [B, T, H, W, D] DiT features
            timestep: [B] diffusion timesteps

        Returns:
            [B, 1] logits
        """
        # Global mean pooling: [B, T, H, W, D] → [B, D]
        if features.dim() == 5:
            x = features.mean(dim=(1, 2, 3))
        elif features.dim() == 3:
            x = features.mean(dim=1)
        else:
            x = features

        # Timestep embedding: [B] → [B, D]
        t_emb = self.t_embedder(timestep)

        # Apply AdaLN: modulate features based on timestep
        x = self.adaln(x, t_emb)

        return self.classifier(x)

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# =============================================================================
# Head 3: TemporalAdaLN (Bidirectional Attention + AdaLN)
# =============================================================================


class TemporalAdaLN(nn.Module):
    """
    Bidirectional temporal attention with AdaLN timestep conditioning.

    AdaLN is applied:
        1. Before attention (modulates queries/keys/values)
        2. Before FFN (modulates attention output)

    This allows the attention mechanism to behave differently at different
    noise levels, e.g., attending to different temporal patterns.

    Pipeline:
        [B, T, H, W, D] → spatial_mean → [B, T, D]
        → proj → [B, T, d] + pos_embed
        → AdaLN → Attention → AdaLN → FFN (× num_layers)
        → mean(T) → [B, d] → classifier → [B, 1]

    Parameters: ~3.0M
    """

    def __init__(
        self,
        hidden_dim: int = 1920,
        attn_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 2,
        max_frames: int = 50,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.attn_dim = attn_dim
        self.num_layers = num_layers

        # Input projection
        self.input_proj = nn.Linear(hidden_dim, attn_dim)

        # Positional embedding
        self.pos_embed = nn.Parameter(torch.randn(1, max_frames, attn_dim) * 0.02)

        # Timestep embedding (projects to attn_dim for AdaLN)
        self.t_embedder = TimestepEmbedder(attn_dim)

        # Transformer layers with AdaLN
        self.attn_layers = nn.ModuleList()
        self.ffn_layers = nn.ModuleList()
        self.adaln_attn = nn.ModuleList()  # AdaLN before attention
        self.adaln_ffn = nn.ModuleList()  # AdaLN before FFN

        for _ in range(num_layers):
            # Multi-head attention
            self.attn_layers.append(
                nn.MultiheadAttention(
                    embed_dim=attn_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    batch_first=True,
                )
            )

            # Feed-forward network
            self.ffn_layers.append(
                nn.Sequential(
                    nn.Linear(attn_dim, attn_dim * 4),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(attn_dim * 4, attn_dim),
                    nn.Dropout(dropout),
                )
            )

            # AdaLN-Zero for gated residual connections
            self.adaln_attn.append(AdaLNZero(attn_dim, attn_dim))
            self.adaln_ffn.append(AdaLNZero(attn_dim, attn_dim))

        # Final classifier
        self.final_norm = nn.LayerNorm(attn_dim)
        self.classifier = nn.Sequential(
            nn.Linear(attn_dim, attn_dim // 2),
            nn.GELU(),
            nn.Linear(attn_dim // 2, 1),
        )

        _init_weights(self)

    def forward(
        self,
        features: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            features: [B, T, H, W, D] DiT features
            timestep: [B] diffusion timesteps

        Returns:
            [B, 1] logits
        """
        # Spatial pooling: [B, T, H, W, D] → [B, T, D]
        if features.dim() == 5:
            x = features.mean(dim=(2, 3))
        else:
            x = features

        B, T, D = x.shape

        # Project and add position: [B, T, D] → [B, T, attn_dim]
        x = self.input_proj(x)
        x = x + self.pos_embed[:, :T, :]

        # Timestep embedding: [B] → [B, attn_dim]
        t_emb = self.t_embedder(timestep)

        # Apply transformer layers with AdaLN
        for attn, ffn, adaln_a, adaln_f in zip(
            self.attn_layers, self.ffn_layers, self.adaln_attn, self.adaln_ffn
        ):
            # AdaLN before attention
            x_norm, gate_a = adaln_a(x, t_emb)
            attn_out, _ = attn(x_norm, x_norm, x_norm)
            x = x + gate_a * attn_out  # Gated residual

            # AdaLN before FFN
            x_norm, gate_f = adaln_f(x, t_emb)
            ffn_out = ffn(x_norm)
            x = x + gate_f * ffn_out  # Gated residual

        # Temporal mean pooling: [B, T, d] → [B, d]
        x = x.mean(dim=1)
        x = self.final_norm(x)

        return self.classifier(x)

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# =============================================================================
# Head 4: CausalAdaLN (Causal Attention + AdaLN)
# =============================================================================


class CausalAdaLN(nn.Module):
    """
    Causal temporal attention with AdaLN timestep conditioning.

    Each frame can only attend to itself and previous frames, respecting
    physical causality (past determines future). AdaLN allows the attention
    to interpret temporal patterns differently based on noise level.

    Pipeline:
        [B, T, H, W, D] → spatial_mean → [B, T, D]
        → proj → [B, T, d] + pos_embed
        → AdaLN → CausalAttention → AdaLN → FFN (× num_layers)
        → take last frame → [B, d] → classifier → [B, 1]

    Why last frame?
        In causal attention, the last frame has attended to all previous frames,
        so it contains a summary of the entire causal history.

    Parameters: ~3.0M
    """

    def __init__(
        self,
        hidden_dim: int = 1920,
        attn_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 2,
        max_frames: int = 50,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.attn_dim = attn_dim
        self.num_layers = num_layers

        # Input projection
        self.input_proj = nn.Linear(hidden_dim, attn_dim)

        # Positional embedding
        self.pos_embed = nn.Parameter(torch.randn(1, max_frames, attn_dim) * 0.02)

        # Timestep embedding
        self.t_embedder = TimestepEmbedder(attn_dim)

        # Transformer layers with AdaLN
        self.attn_layers = nn.ModuleList()
        self.ffn_layers = nn.ModuleList()
        self.adaln_attn = nn.ModuleList()
        self.adaln_ffn = nn.ModuleList()

        for _ in range(num_layers):
            self.attn_layers.append(
                nn.MultiheadAttention(
                    embed_dim=attn_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    batch_first=True,
                )
            )

            self.ffn_layers.append(
                nn.Sequential(
                    nn.Linear(attn_dim, attn_dim * 4),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(attn_dim * 4, attn_dim),
                    nn.Dropout(dropout),
                )
            )

            self.adaln_attn.append(AdaLNZero(attn_dim, attn_dim))
            self.adaln_ffn.append(AdaLNZero(attn_dim, attn_dim))

        # Final layers
        self.final_norm = nn.LayerNorm(attn_dim)
        self.classifier = nn.Sequential(
            nn.Linear(attn_dim, attn_dim // 2),
            nn.GELU(),
            nn.Linear(attn_dim // 2, 1),
        )

        _init_weights(self)

    def _get_causal_mask(self, T: int, device: torch.device) -> torch.Tensor:
        """
        Create causal attention mask.

        Returns:
            [T, T] mask where True = cannot attend (upper triangular)
        """
        return torch.triu(torch.ones(T, T, device=device), diagonal=1).bool()

    def forward(
        self,
        features: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            features: [B, T, H, W, D] DiT features
            timestep: [B] diffusion timesteps

        Returns:
            [B, 1] logits
        """
        # Spatial pooling: [B, T, H, W, D] → [B, T, D]
        if features.dim() == 5:
            x = features.mean(dim=(2, 3))
        else:
            x = features

        B, T, D = x.shape

        # Project and add position
        x = self.input_proj(x)
        x = x + self.pos_embed[:, :T, :]

        # Timestep embedding
        t_emb = self.t_embedder(timestep)

        # Causal mask
        causal_mask = self._get_causal_mask(T, x.device)

        # Apply transformer layers with AdaLN
        for attn, ffn, adaln_a, adaln_f in zip(
            self.attn_layers, self.ffn_layers, self.adaln_attn, self.adaln_ffn
        ):
            # AdaLN before causal attention
            x_norm, gate_a = adaln_a(x, t_emb)
            attn_out, _ = attn(x_norm, x_norm, x_norm, attn_mask=causal_mask)
            x = x + gate_a * attn_out

            # AdaLN before FFN
            x_norm, gate_f = adaln_f(x, t_emb)
            ffn_out = ffn(x_norm)
            x = x + gate_f * ffn_out

        # Take last frame (contains full causal history): [B, T, d] → [B, d]
        x = x[:, -1, :]
        x = self.final_norm(x)

        return self.classifier(x)

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# =============================================================================
# Head 5: MultiViewAdaLN (Multi-View Pooling + AdaLN)
# =============================================================================


class MultiViewAdaLN(nn.Module):
    """
    Multi-view pooling with AdaLN timestep conditioning.

    Combines 6 complementary views of the video, each modulated by AdaLN
    to interpret features appropriately for the current noise level.

    Views:
        1. mean_pool:   Global average (overall statistics)
        2. max_pool:    Peak activations (salient features)
        3. first_frame: Initial conditions
        4. last_frame:  Final state
        5. diff_pool:   Temporal dynamics (velocity/motion)
        6. causal_pool: Causal aggregation (trajectory coherence)

    Each view is processed with its own AdaLN, allowing timestep to affect
    how each type of information is weighted.

    Pipeline:
        [B, T, H, W, D] → spatial_mean → [B, T, D]

        ├── mean(T) → AdaLN → [B, d]
        ├── max(T) → AdaLN → [B, d]
        ├── first_frame → AdaLN → [B, d]
        ├── last_frame → AdaLN → [B, d]
        ├── diff_mean → AdaLN → [B, d]
        └── causal_attn → AdaLN → [B, d]

        concat → classifier → [B, 1]

    Parameters: ~7.0M
    """

    def __init__(
        self,
        hidden_dim: int = 1920,
        view_dim: int = 256,
        attn_dim: int = 256,
        num_heads: int = 4,
        mlp_dim: int = 512,
        max_frames: int = 50,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.view_dim = view_dim
        self.attn_dim = attn_dim

        # Timestep embedding (shared across views)
        self.t_embedder = TimestepEmbedder(view_dim)

        # =============================================
        # View projections (one for each pooling type)
        # =============================================
        self.mean_proj = nn.Linear(hidden_dim, view_dim)
        self.max_proj = nn.Linear(hidden_dim, view_dim)
        self.first_proj = nn.Linear(hidden_dim, view_dim)
        self.last_proj = nn.Linear(hidden_dim, view_dim)
        self.diff_proj = nn.Linear(hidden_dim, view_dim)

        # =============================================
        # AdaLN for each view (allows timestep-specific weighting)
        # =============================================
        self.adaln_mean = AdaLN(view_dim, view_dim)
        self.adaln_max = AdaLN(view_dim, view_dim)
        self.adaln_first = AdaLN(view_dim, view_dim)
        self.adaln_last = AdaLN(view_dim, view_dim)
        self.adaln_diff = AdaLN(view_dim, view_dim)
        self.adaln_causal = AdaLN(attn_dim, view_dim)

        # =============================================
        # Causal attention branch
        # =============================================
        self.causal_proj = nn.Linear(hidden_dim, attn_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, max_frames, attn_dim) * 0.02)
        self.causal_attn = nn.MultiheadAttention(
            embed_dim=attn_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # =============================================
        # Final classifier
        # =============================================
        # 5 views (5 * view_dim) + causal (attn_dim)
        total_dim = 5 * view_dim + attn_dim

        self.classifier = nn.Sequential(
            nn.LayerNorm(total_dim),
            nn.Linear(total_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, mlp_dim // 2),
            nn.GELU(),
            nn.Linear(mlp_dim // 2, 1),
        )

        _init_weights(self)

    def _get_causal_mask(self, T: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.ones(T, T, device=device), diagonal=1).bool()

    def forward(
        self,
        features: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            features: [B, T, H, W, D] DiT features
            timestep: [B] diffusion timesteps

        Returns:
            [B, 1] logits
        """
        # Spatial pooling: [B, T, H, W, D] → [B, T, D]
        if features.dim() == 5:
            x = features.mean(dim=(2, 3))
        else:
            x = features

        B, T, D = x.shape

        # Timestep embedding: [B] → [B, view_dim]
        t_emb = self.t_embedder(timestep)

        # =============================================
        # Multi-view pooling with AdaLN
        # =============================================

        # View 1: Mean pool + AdaLN
        mean_pool = self.mean_proj(x.mean(dim=1))  # [B, view_dim]
        mean_pool = self.adaln_mean(mean_pool, t_emb)

        # View 2: Max pool + AdaLN
        max_pool = self.max_proj(x.max(dim=1)[0])  # [B, view_dim]
        max_pool = self.adaln_max(max_pool, t_emb)

        # View 3: First frame + AdaLN
        first_pool = self.first_proj(x[:, 0, :])  # [B, view_dim]
        first_pool = self.adaln_first(first_pool, t_emb)

        # View 4: Last frame + AdaLN
        last_pool = self.last_proj(x[:, -1, :])  # [B, view_dim]
        last_pool = self.adaln_last(last_pool, t_emb)

        # View 5: Temporal diff + AdaLN
        diff = x[:, 1:, :] - x[:, :-1, :]  # [B, T-1, D]
        diff_pool = self.diff_proj(diff.mean(dim=1))  # [B, view_dim]
        diff_pool = self.adaln_diff(diff_pool, t_emb)

        # View 6: Causal attention + AdaLN
        causal_x = self.causal_proj(x)  # [B, T, attn_dim]
        causal_x = causal_x + self.pos_embed[:, :T, :]
        causal_mask = self._get_causal_mask(T, x.device)
        causal_out, _ = self.causal_attn(
            causal_x, causal_x, causal_x, attn_mask=causal_mask
        )
        causal_pool = causal_out[:, -1, :]  # [B, attn_dim]
        causal_pool = self.adaln_causal(causal_pool, t_emb)

        # =============================================
        # Concatenate and classify
        # =============================================
        combined = torch.cat(
            [
                mean_pool,  # global statistics
                max_pool,  # salient features
                first_pool,  # initial state
                last_pool,  # final state
                diff_pool,  # dynamics
                causal_pool,  # causal aggregation
            ],
            dim=-1,
        )  # [B, 5*view_dim + attn_dim]

        return self.classifier(combined)

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# =============================================================================
# Factory Functions
# =============================================================================

HEAD_REGISTRY: Dict[str, type] = {
    "mean": MeanPool,
    "mean_adaln": MeanPoolAdaLN,
    "temporal_adaln": TemporalAdaLN,
    "causal_adaln": CausalAdaLN,
    "multiview_adaln": MultiViewAdaLN,
}


def create_physics_head(
    head_type: str = "multiview_adaln",
    hidden_dim: int = 1920,
    **kwargs,
) -> nn.Module:
    """
    Factory function to create a physics discriminator head.

    Args:
        head_type: One of "mean", "mean_adaln", "temporal_adaln",
                   "causal_adaln", "multiview_adaln"
        hidden_dim: DiT hidden dimension (1920 for CogVideoX-2B)
        **kwargs: Additional arguments for the head

    Returns:
        Physics head module

    Example:
        head = create_physics_head("causal_adaln", hidden_dim=1920)
        logits = head(features, timestep)
    """
    if head_type not in HEAD_REGISTRY:
        raise ValueError(
            f"Unknown head_type: '{head_type}'. Available: {list(HEAD_REGISTRY.keys())}"
        )
    return HEAD_REGISTRY[head_type](hidden_dim=hidden_dim, **kwargs)


def list_heads() -> List[str]:
    """Return list of available head types."""
    return list(HEAD_REGISTRY.keys())


def get_head_info() -> Dict[str, Dict]:
    """Return detailed information about each head."""
    return {
        "mean": {
            "name": "MeanPool",
            "description": "Global mean pooling baseline (no timestep)",
            "timestep": "None",
            "params": "~1.0M",
            "expected_auc": "0.55-0.65",
        },
        "mean_adaln": {
            "name": "MeanPoolAdaLN",
            "description": "Mean pooling with AdaLN timestep modulation",
            "timestep": "AdaLN",
            "params": "~1.5M",
            "expected_auc": "0.60-0.70",
        },
        "temporal_adaln": {
            "name": "TemporalAdaLN",
            "description": "Bidirectional attention with AdaLN",
            "timestep": "AdaLN-Zero (every layer)",
            "params": "~3.0M",
            "expected_auc": "0.65-0.75",
        },
        "causal_adaln": {
            "name": "CausalAdaLN",
            "description": "Causal attention with AdaLN (physics-aware)",
            "timestep": "AdaLN-Zero (every layer)",
            "params": "~3.0M",
            "expected_auc": "0.68-0.78",
        },
        "multiview_adaln": {
            "name": "MultiViewAdaLN",
            "description": "Multi-view pooling with AdaLN (most comprehensive)",
            "timestep": "AdaLN (per view)",
            "params": "~7.0M",
            "expected_auc": "0.72-0.82",
        },
    }
    
# =============================================================================
# SIMPLIFIED VARIANTS (append to physics_head.py)
# =============================================================================
#
# These are lighter versions of the underperforming heads for ablation study.
# Add these classes BEFORE the HEAD_REGISTRY definition.
#
# Results context:
#   - mean:           AUC 0.749, epoch 28 (BEST - ignores timestep!)
#   - mean_adaln:     AUC 0.610, epoch 1  (immediate overfit)
#   - causal_adaln:   AUC 0.664, epoch 3  (early overfit)
#
# Key insight: MeanPool ignores timestep entirely and performs best,
# suggesting DiT features already encode timestep information implicitly.
# =============================================================================


# =============================================================================
# Simplified MeanPoolAdaLN Variants
# =============================================================================


class MeanPoolConcat(nn.Module):
    """
    MeanPool with timestep CONCATENATION instead of AdaLN.

    Hypothesis: AdaLN's multiplicative modulation (scale * x + shift) may cause
    training instability. Simple concat is more stable.

    Changes from MeanPoolAdaLN:
        - Remove AdaLN entirely
        - Concat [features, t_emb] before classifier
        - Smaller t_emb dimension (256 vs 1920)

    Pipeline:
        [B, T, H, W, D] → mean → [B, D]
        timestep → embed → [B, 256]
        concat → [B, D+256] → MLP → [B, 1]

    Parameters: ~1.2M (vs ~1.5M original)
    """

    def __init__(
        self,
        hidden_dim: int = 1920,
        t_dim: int = 256,
        mlp_dim: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.t_dim = t_dim

        # Timestep embedding (smaller dimension)
        self.t_embedder = TimestepEmbedder(t_dim, t_dim)

        # Classifier takes concatenated input
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim + t_dim),
            nn.Linear(hidden_dim + t_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, mlp_dim // 2),
            nn.GELU(),
            nn.Linear(mlp_dim // 2, 1),
        )

        _init_weights(self)

    def forward(
        self,
        features: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        # Mean pool: [B, T, H, W, D] → [B, D]
        if features.dim() == 5:
            x = features.mean(dim=(1, 2, 3))
        elif features.dim() == 3:
            x = features.mean(dim=1)
        else:
            x = features

        # Timestep embedding
        t_emb = self.t_embedder(timestep)  # [B, t_dim]

        # Simple concat (no multiplicative interaction)
        x = torch.cat([x, t_emb], dim=-1)  # [B, D + t_dim]

        return self.classifier(x)

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class MeanPoolBias(nn.Module):
    """
    MeanPool with timestep as BIAS only (no scale).

    Hypothesis: The scale parameter in AdaLN may cause gradient instability.
    Using only shift/bias is more stable.

    Changes from MeanPoolAdaLN:
        - Remove scale, keep only shift: x = norm(x) + bias(t_emb)
        - This is additive, not multiplicative

    Pipeline:
        [B, T, H, W, D] → mean → [B, D]
        timestep → embed → bias [B, D]
        norm(x) + bias → MLP → [B, 1]

    Parameters: ~1.0M
    """

    def __init__(
        self,
        hidden_dim: int = 1920,
        mlp_dim: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Timestep → bias (same dim as features for addition)
        self.t_embedder = TimestepEmbedder(256, hidden_dim)

        # LayerNorm (no affine, we add bias from timestep)
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, mlp_dim // 2),
            nn.GELU(),
            nn.Linear(mlp_dim // 2, 1),
        )

        _init_weights(self)
        # Initialize bias projection to near-zero for stable start
        nn.init.zeros_(self.t_embedder.mlp[-1].weight)
        nn.init.zeros_(self.t_embedder.mlp[-1].bias)

    def forward(
        self,
        features: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        # Mean pool
        if features.dim() == 5:
            x = features.mean(dim=(1, 2, 3))
        elif features.dim() == 3:
            x = features.mean(dim=1)
        else:
            x = features

        # Timestep as bias only (additive, not multiplicative)
        bias = self.t_embedder(timestep)  # [B, D]
        x = self.norm(x) + bias  # No scale!

        return self.classifier(x)

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class MeanPoolAdaLNLite(nn.Module):
    """
    MeanPoolAdaLN with REDUCED capacity.

    Hypothesis: Original AdaLN has too many parameters for ~20k samples,
    causing immediate overfitting at epoch 1.

    Changes from MeanPoolAdaLN:
        - Project to smaller dim first: 1920 → 256
        - Smaller AdaLN parameters
        - Simpler classifier

    Parameters: ~0.5M (vs ~1.5M original)
    """

    def __init__(
        self,
        hidden_dim: int = 1920,
        proj_dim: int = 256,
        mlp_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.proj_dim = proj_dim

        # Project to smaller dimension first
        self.input_proj = nn.Linear(hidden_dim, proj_dim)

        # Timestep embedding (smaller)
        self.t_embedder = TimestepEmbedder(proj_dim, proj_dim)

        # Lightweight AdaLN
        self.norm = nn.LayerNorm(proj_dim, elementwise_affine=False)
        self.adaln_proj = nn.Linear(proj_dim, proj_dim * 2)  # scale + shift
        nn.init.zeros_(self.adaln_proj.weight)
        nn.init.zeros_(self.adaln_proj.bias)

        # Simpler classifier
        self.classifier = nn.Sequential(
            nn.Linear(proj_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, 1),
        )

        _init_weights(self)

    def forward(
        self,
        features: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        # Mean pool
        if features.dim() == 5:
            x = features.mean(dim=(1, 2, 3))
        elif features.dim() == 3:
            x = features.mean(dim=1)
        else:
            x = features

        # Project to smaller dim
        x = self.input_proj(x)  # [B, proj_dim]

        # Timestep embedding
        t_emb = self.t_embedder(timestep)

        # Lightweight AdaLN
        scale, shift = self.adaln_proj(t_emb).chunk(2, dim=-1)
        x = self.norm(x) * (1 + scale) + shift

        return self.classifier(x)

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# =============================================================================
# Simplified CausalAdaLN Variants
# =============================================================================


class CausalAdaLN1Layer(nn.Module):
    """
    CausalAdaLN with SINGLE transformer layer.

    Hypothesis: 2 layers with 4 AdaLN modules is too complex for limited data.

    Changes from CausalAdaLN:
        - num_layers: 2 → 1
        - AdaLN modules: 4 → 2

    Parameters: ~1.5M (vs ~3.0M original)
    """

    def __init__(
        self,
        hidden_dim: int = 1920,
        attn_dim: int = 512,
        num_heads: int = 8,
        max_frames: int = 50,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.attn_dim = attn_dim

        # Input projection
        self.input_proj = nn.Linear(hidden_dim, attn_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, max_frames, attn_dim) * 0.02)

        # Timestep embedding
        self.t_embedder = TimestepEmbedder(attn_dim, attn_dim)

        # Single attention layer
        self.attn = nn.MultiheadAttention(
            embed_dim=attn_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Single FFN
        self.ffn = nn.Sequential(
            nn.Linear(attn_dim, attn_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(attn_dim * 4, attn_dim),
            nn.Dropout(dropout),
        )

        # AdaLN-Zero for single layer (2 instead of 4)
        self.adaln_attn = AdaLNZero(attn_dim, attn_dim)
        self.adaln_ffn = AdaLNZero(attn_dim, attn_dim)

        # Classifier
        self.final_norm = nn.LayerNorm(attn_dim)
        self.classifier = nn.Sequential(
            nn.Linear(attn_dim, attn_dim // 2),
            nn.GELU(),
            nn.Linear(attn_dim // 2, 1),
        )

        _init_weights(self)

    def _get_causal_mask(self, T: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.ones(T, T, device=device), diagonal=1).bool()

    def forward(
        self,
        features: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        # Spatial pool: [B, T, H, W, D] → [B, T, D]
        if features.dim() == 5:
            x = features.mean(dim=(2, 3))
        else:
            x = features

        B, T, D = x.shape

        # Project and add position
        x = self.input_proj(x)
        x = x + self.pos_embed[:, :T, :]

        # Timestep embedding
        t_emb = self.t_embedder(timestep)

        # Causal mask
        causal_mask = self._get_causal_mask(T, x.device)

        # Single layer with AdaLN-Zero
        x_norm, gate_a = self.adaln_attn(x, t_emb)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, attn_mask=causal_mask)
        x = x + gate_a * attn_out

        x_norm, gate_f = self.adaln_ffn(x, t_emb)
        x = x + gate_f * self.ffn(x_norm)

        # Take last frame (contains causal history)
        x = x[:, -1, :]
        x = self.final_norm(x)

        return self.classifier(x)

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class CausalConcat(nn.Module):
    """
    Causal attention WITHOUT AdaLN, timestep CONCAT at the end.

    Hypothesis: AdaLN + causal mask combination is overly restrictive.
    Separating them may help.

    Changes from CausalAdaLN:
        - Remove all AdaLN modules
        - Use standard Pre-LN transformer
        - Concat timestep only at the final classifier

    Parameters: ~2.0M
    """

    def __init__(
        self,
        hidden_dim: int = 1920,
        attn_dim: int = 512,
        t_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 2,
        max_frames: int = 50,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.attn_dim = attn_dim
        self.num_layers = num_layers

        # Input projection
        self.input_proj = nn.Linear(hidden_dim, attn_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, max_frames, attn_dim) * 0.02)

        # Timestep embedding (for concat at end)
        self.t_embedder = TimestepEmbedder(t_dim, t_dim)

        # Standard transformer layers (NO AdaLN)
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                nn.ModuleDict(
                    {
                        "norm1": nn.LayerNorm(attn_dim),
                        "attn": nn.MultiheadAttention(
                            embed_dim=attn_dim,
                            num_heads=num_heads,
                            dropout=dropout,
                            batch_first=True,
                        ),
                        "norm2": nn.LayerNorm(attn_dim),
                        "ffn": nn.Sequential(
                            nn.Linear(attn_dim, attn_dim * 4),
                            nn.GELU(),
                            nn.Dropout(dropout),
                            nn.Linear(attn_dim * 4, attn_dim),
                            nn.Dropout(dropout),
                        ),
                    }
                )
            )

        # Classifier (concat timestep here)
        self.final_norm = nn.LayerNorm(attn_dim)
        self.classifier = nn.Sequential(
            nn.Linear(attn_dim + t_dim, attn_dim // 2),
            nn.GELU(),
            nn.Linear(attn_dim // 2, 1),
        )

        _init_weights(self)

    def _get_causal_mask(self, T: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.ones(T, T, device=device), diagonal=1).bool()

    def forward(
        self,
        features: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        # Spatial pool
        if features.dim() == 5:
            x = features.mean(dim=(2, 3))
        else:
            x = features

        B, T, D = x.shape

        # Project and add position
        x = self.input_proj(x)
        x = x + self.pos_embed[:, :T, :]

        # Causal mask
        causal_mask = self._get_causal_mask(T, x.device)

        # Standard Pre-LN transformer (no AdaLN)
        for layer in self.layers:
            # Attention
            x_norm = layer["norm1"](x)
            attn_out, _ = layer["attn"](x_norm, x_norm, x_norm, attn_mask=causal_mask)
            x = x + attn_out

            # FFN
            x_norm = layer["norm2"](x)
            x = x + layer["ffn"](x_norm)

        # Take last frame
        x = x[:, -1, :]
        x = self.final_norm(x)

        # Concat timestep at the end only
        t_emb = self.t_embedder(timestep)
        x = torch.cat([x, t_emb], dim=-1)

        return self.classifier(x)

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class CausalSimple(nn.Module):
    """
    MINIMAL causal temporal attention, NO timestep.

    Key differences from CausalSimple:
        - Completely removes timestep embedding and concat
        - Even simpler classifier

    Design rationale:
        - Causal mask enforces temporal ordering (physics causality)
        - Last frame aggregates all previous frames
        - No timestep needed if features already encode it

    Pipeline:
        [B, T, H, W, D] → spatial_mean → [B, T, D]
        → proj → [B, T, 256] + pos_embed
        → LayerNorm → Causal Attention (lower triangular mask)
        → take last frame → [B, 256] → classifier → [B, 1]

    Parameters: ~0.55M (vs ~0.8M CausalSimple with timestep)
    """

    def __init__(
        self,
        hidden_dim: int = 1920,
        attn_dim: int = 256,
        num_heads: int = 4,
        max_frames: int = 50,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.attn_dim = attn_dim

        # Input projection
        self.input_proj = nn.Linear(hidden_dim, attn_dim)

        # Positional embedding
        self.pos_embed = nn.Parameter(torch.randn(1, max_frames, attn_dim) * 0.02)

        # Single causal attention layer
        self.norm = nn.LayerNorm(attn_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=attn_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Simple classifier (NO timestep)
        self.classifier = nn.Sequential(
            nn.LayerNorm(attn_dim),
            nn.Linear(attn_dim, 128),
            nn.GELU(),
            nn.Linear(128, 1),
        )

        _init_weights(self)

    def _get_causal_mask(self, T: int, device: torch.device) -> torch.Tensor:
        """
        Create causal attention mask (upper triangular = True = masked).

        Frame i can only attend to frames 0..i (itself and past).
        """
        return torch.triu(torch.ones(T, T, device=device), diagonal=1).bool()

    def forward(
        self,
        features: torch.Tensor,
        timestep: Optional[torch.Tensor] = None,  # ignored
    ) -> torch.Tensor:
        """
        Args:
            features: [B, T, H, W, D] or [B, T, D] DiT features
            timestep: ignored (kept for API compatibility)

        Returns:
            [B, 1] logits
        """
        # Spatial pooling
        if features.dim() == 5:
            x = features.mean(dim=(2, 3))
        else:
            x = features

        B, T, D = x.shape

        # Project and add position
        x = self.input_proj(x)
        x = x + self.pos_embed[:, :T, :]

        # Causal attention
        x_norm = self.norm(x)
        causal_mask = self._get_causal_mask(T, x.device)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, attn_mask=causal_mask)
        x = x + attn_out

        # Take LAST frame (contains causal aggregation of all frames)
        x = x[:, -1, :]  # [B, attn_dim]

        return self.classifier(x)

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class TemporalSimple(nn.Module):
    """
    MINIMAL bidirectional temporal attention, NO timestep.

    Design rationale:
        - Single attention layer (vs 2 in TemporalAdaLN)
        - Small dimensions (256 vs 512)
        - No FFN layer
        - No timestep embedding at all
        - Mean pooling over temporal dimension

    Pipeline:
        [B, T, H, W, D] → spatial_mean → [B, T, D]
        → proj → [B, T, 256] + pos_embed
        → LayerNorm → Bidirectional Attention
        → mean(T) → [B, 256] → classifier → [B, 1]

    Parameters: ~0.6M (vs ~3.0M TemporalAdaLN)
    """

    def __init__(
        self,
        hidden_dim: int = 1920,
        attn_dim: int = 256,
        num_heads: int = 4,
        max_frames: int = 50,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.attn_dim = attn_dim

        # Input projection to smaller dimension
        self.input_proj = nn.Linear(hidden_dim, attn_dim)

        # Learnable positional embedding
        self.pos_embed = nn.Parameter(torch.randn(1, max_frames, attn_dim) * 0.02)

        # Single bidirectional attention layer
        self.norm = nn.LayerNorm(attn_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=attn_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Simple classifier (no timestep input)
        self.classifier = nn.Sequential(
            nn.LayerNorm(attn_dim),
            nn.Linear(attn_dim, 128),
            nn.GELU(),
            nn.Linear(128, 1),
        )

        _init_weights(self)

    def forward(
        self,
        features: torch.Tensor,
        timestep: Optional[torch.Tensor] = None,  # ignored
    ) -> torch.Tensor:
        """
        Args:
            features: [B, T, H, W, D] or [B, T, D] DiT features
            timestep: ignored (kept for API compatibility)

        Returns:
            [B, 1] logits
        """
        # Spatial pooling: [B, T, H, W, D] → [B, T, D]
        if features.dim() == 5:
            x = features.mean(dim=(2, 3))
        else:
            x = features

        B, T, D = x.shape

        # Project to smaller dimension and add positional embedding
        x = self.input_proj(x)  # [B, T, attn_dim]
        x = x + self.pos_embed[:, :T, :]

        # Single bidirectional attention (NO causal mask)
        x_norm = self.norm(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out  # residual connection

        # Temporal mean pooling: [B, T, attn_dim] → [B, attn_dim]
        x = x.mean(dim=1)

        return self.classifier(x)

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
class MultiViewSimple(nn.Module):
    """
    Multi-view pooling WITHOUT timestep, combining:
        1. Mean pooling (global statistics)
        2. Temporal attention (bidirectional context)
        3. Causal attention (physics-aware aggregation)

    Design rationale:
        - Each view captures different aspects of video features
        - Mean: overall feature magnitude
        - Temporal: frame interactions (bidirectional)
        - Causal: temporal ordering (unidirectional, physics causality)
        - No timestep: DiT features already encode noise level

    Pipeline:
        [B, T, H, W, D] → spatial_mean → [B, T, D]

        View 1 (Mean):     mean(T) → proj → [B, d]
        View 2 (Temporal): bidirectional_attn → mean(T) → [B, d]
        View 3 (Causal):   causal_attn → last_frame → [B, d]

        concat([v1, v2, v3]) → [B, 3d] → classifier → [B, 1]

    Parameters: ~1.2M (vs ~7.0M MultiViewAdaLN)
    """

    def __init__(
        self,
        hidden_dim: int = 1920,
        view_dim: int = 256,
        num_heads: int = 4,
        max_frames: int = 50,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.view_dim = view_dim

        # ===========================================
        # View 1: Mean Pooling (simplest baseline)
        # ===========================================
        self.mean_proj = nn.Linear(hidden_dim, view_dim)

        # ===========================================
        # View 2: Temporal (Bidirectional) Attention
        # ===========================================
        self.temporal_proj = nn.Linear(hidden_dim, view_dim)
        self.temporal_pos = nn.Parameter(torch.randn(1, max_frames, view_dim) * 0.02)
        self.temporal_norm = nn.LayerNorm(view_dim)
        self.temporal_attn = nn.MultiheadAttention(
            embed_dim=view_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # ===========================================
        # View 3: Causal Attention
        # ===========================================
        self.causal_proj = nn.Linear(hidden_dim, view_dim)
        self.causal_pos = nn.Parameter(torch.randn(1, max_frames, view_dim) * 0.02)
        self.causal_norm = nn.LayerNorm(view_dim)
        self.causal_attn = nn.MultiheadAttention(
            embed_dim=view_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # ===========================================
        # Final Classifier (3 views concatenated)
        # ===========================================
        total_dim = view_dim * 3  # mean + temporal + causal
        self.classifier = nn.Sequential(
            nn.LayerNorm(total_dim),
            nn.Linear(total_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 1),
        )

        _init_weights(self)

    def _get_causal_mask(self, T: int, device: torch.device) -> torch.Tensor:
        """Create causal attention mask."""
        return torch.triu(torch.ones(T, T, device=device), diagonal=1).bool()

    def forward(
        self,
        features: torch.Tensor,
        timestep: Optional[torch.Tensor] = None,  # ignored
    ) -> torch.Tensor:
        """
        Args:
            features: [B, T, H, W, D] or [B, T, D] DiT features
            timestep: ignored (kept for API compatibility)

        Returns:
            [B, 1] logits
        """
        # Spatial pooling: [B, T, H, W, D] → [B, T, D]
        if features.dim() == 5:
            x = features.mean(dim=(2, 3))
        else:
            x = features

        B, T, D = x.shape

        # ===========================================
        # View 1: Mean Pooling
        # ===========================================
        mean_pool = x.mean(dim=1)  # [B, D]
        mean_pool = self.mean_proj(mean_pool)  # [B, view_dim]

        # ===========================================
        # View 2: Temporal (Bidirectional) Attention
        # ===========================================
        temporal_x = self.temporal_proj(x)  # [B, T, view_dim]
        temporal_x = temporal_x + self.temporal_pos[:, :T, :]
        temporal_norm = self.temporal_norm(temporal_x)
        temporal_attn_out, _ = self.temporal_attn(
            temporal_norm, temporal_norm, temporal_norm
        )
        temporal_x = temporal_x + temporal_attn_out
        temporal_pool = temporal_x.mean(dim=1)  # [B, view_dim]

        # ===========================================
        # View 3: Causal Attention
        # ===========================================
        causal_x = self.causal_proj(x)  # [B, T, view_dim]
        causal_x = causal_x + self.causal_pos[:, :T, :]
        causal_norm = self.causal_norm(causal_x)
        causal_mask = self._get_causal_mask(T, x.device)
        causal_attn_out, _ = self.causal_attn(
            causal_norm, causal_norm, causal_norm, attn_mask=causal_mask
        )
        causal_x = causal_x + causal_attn_out
        causal_pool = causal_x[:, -1, :]  # [B, view_dim] (last frame)

        # ===========================================
        # Concatenate all views and classify
        # ===========================================
        combined = torch.cat([mean_pool, temporal_pool, causal_pool], dim=-1)
        # [B, view_dim * 3]

        return self.classifier(combined)

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# =============================================================================
# UPDATE HEAD_REGISTRY (replace the existing one)
# =============================================================================

HEAD_REGISTRY: Dict[str, type] = {
    # Original heads
    "mean": MeanPool,
    "mean_adaln": MeanPoolAdaLN,
    "temporal_adaln": TemporalAdaLN,
    "causal_adaln": CausalAdaLN,
    "multiview_adaln": MultiViewAdaLN,
    # Simplified variants for ablation
    "mean_concat": MeanPoolConcat,
    "mean_bias": MeanPoolBias,
    "mean_adaln_lite": MeanPoolAdaLNLite,
    "causal_1layer": CausalAdaLN1Layer,
    "causal_concat": CausalConcat,
    "causal_simple": CausalSimple,
    "temporal_simple": TemporalSimple,
    "multiview_simple": MultiViewSimple,
}


# =============================================================================
# UPDATE get_head_info() (replace the existing one)
# =============================================================================


def get_head_info() -> Dict[str, Dict]:
    """Return detailed information about each head."""
    return {
        # Original heads
        "mean": {
            "name": "MeanPool",
            "description": "Global mean pooling (BEST - ignores timestep!)",
            "timestep": "None",
            "params": "~1.0M",
            "auc": "0.749",
        },
        "mean_adaln": {
            "name": "MeanPoolAdaLN",
            "description": "Mean pooling with AdaLN timestep modulation",
            "timestep": "AdaLN",
            "params": "~1.5M",
            "auc": "0.610 (epoch 1)",
        },
        "temporal_adaln": {
            "name": "TemporalAdaLN",
            "description": "Bidirectional attention with AdaLN",
            "timestep": "AdaLN-Zero (every layer)",
            "params": "~3.0M",
            "auc": "0.725",
        },
        "causal_adaln": {
            "name": "CausalAdaLN",
            "description": "Causal attention with AdaLN",
            "timestep": "AdaLN-Zero (every layer)",
            "params": "~3.0M",
            "auc": "0.664 (epoch 3)",
        },
        "multiview_adaln": {
            "name": "MultiViewAdaLN",
            "description": "Multi-view pooling with AdaLN",
            "timestep": "AdaLN (per view)",
            "params": "~7.0M",
            "auc": "0.727",
        },
        # Simplified variants
        "mean_concat": {
            "name": "MeanPoolConcat",
            "description": "Mean pooling + timestep CONCAT (no AdaLN)",
            "timestep": "Concat",
            "params": "~1.2M",
            "base": "mean_adaln",
        },
        "mean_bias": {
            "name": "MeanPoolBias",
            "description": "Mean pooling + timestep as BIAS only (no scale)",
            "timestep": "Bias only",
            "params": "~1.0M",
            "base": "mean_adaln",
        },
        "mean_adaln_lite": {
            "name": "MeanPoolAdaLNLite",
            "description": "MeanPoolAdaLN with reduced capacity",
            "timestep": "AdaLN (lite)",
            "params": "~0.5M",
            "base": "mean_adaln",
        },
        "causal_1layer": {
            "name": "CausalAdaLN1Layer",
            "description": "CausalAdaLN with 1 layer instead of 2",
            "timestep": "AdaLN-Zero",
            "params": "~1.5M",
            "base": "causal_adaln",
        },
        "causal_concat": {
            "name": "CausalConcat",
            "description": "Causal attention + timestep CONCAT (no AdaLN)",
            "timestep": "Concat at end",
            "params": "~2.0M",
            "base": "causal_adaln",
        },
        "causal_simple": {
            "name": "CausalSimple",
            "description": "Minimal causal: 1 layer, NO timestep",
            "timestep": "None",
            "params": "~0.55M",
            "auc": "TBD",
        },
        # NEW: Simplified variants WITHOUT timestep
        "temporal_simple": {
            "name": "TemporalSimple",
            "description": "Minimal bidirectional: 1 layer, no timestep",
            "timestep": "None",
            "params": "~0.6M",
            "auc": "TBD",
        },
        "multiview_simple": {
            "name": "MultiViewSimple",
            "description": "Multi-view (mean+temporal+causal), NO timestep",
            "timestep": "None",
            "params": "~1.2M",
            "auc": "TBD",
        },
    }


# =============================================================================
# Testing
# =============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("Physics Discriminator Heads with AdaLN Timestep Conditioning")
    print("=" * 80)

    # Test input
    B, T, H, W, D = 4, 13, 30, 45, 1920
    features = torch.randn(B, T, H, W, D)
    timestep = torch.tensor([200, 400, 600, 800])

    print(f"\nTest Input:")
    print(f"  Features: [{B}, {T}, {H}, {W}, {D}]")
    print(f"  Timestep: {timestep.tolist()}")
    print()

    # Test each head
    print(f"{'Head':<20} {'Params':>12} {'Timestep':>15} {'Output':>12}")
    print("-" * 65)

    for name in HEAD_REGISTRY:
        head = create_physics_head(name, hidden_dim=D)
        params = head.get_num_params()

        with torch.no_grad():
            if name == "mean":
                out = head(features)
                t_type = "None"
            else:
                out = head(features, timestep)
                t_type = "AdaLN"

        print(f"{name:<20} {params:>12,} {t_type:>15} {str(list(out.shape)):>12}")

    print()
    print("=" * 80)
    print("AdaLN vs Concatenation Comparison:")
    print("=" * 80)
    print("""
    Old (concat):
        features ── attention ── concat([out, t_emb]) ── classifier
                                        ↑
        timestep ───────────────────────┘ (only at end)
    
    New (AdaLN):
        features ── AdaLN(t) ── attention ── AdaLN(t) ── classifier
                       ↑                        ↑
        timestep ──────┴────────────────────────┘ (modulates every layer)
    
    Benefits of AdaLN:
        1. Timestep affects entire computation, not just final decision
        2. Different noise levels trigger different attention patterns
        3. Matches DiT's design (our features come from DiT)
        4. Gate mechanism (AdaLN-Zero) enables stable deep training
    """)
    print("=" * 80)
