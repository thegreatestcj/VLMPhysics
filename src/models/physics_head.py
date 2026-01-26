"""
Physics Discriminator Head

Lightweight MLP that predicts physics plausibility from DiT features.
Target: ~0.5M-1M parameters for fast training and inference.

Usage:
    head = PhysicsHead(hidden_dim=1920)
    score = head(features)  # features: [B, T, h, w, D]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class PhysicsHead(nn.Module):
    """
    Simple physics discriminator: Pool -> MLP -> Score

    Pooling: Average over all dimensions -> [B, D]
    MLP: D -> hidden -> 1
    """

    def __init__(
        self,
        hidden_dim: int = 1920,
        mlp_hidden: int = 512,
        mlp_layers: int = 2,
        dropout: float = 0.1,
        pool_type: str = "mean",
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.pool_type = pool_type

        # Project down from DiT dimension
        self.proj = nn.Linear(hidden_dim, mlp_hidden)

        # MLP layers
        layers = []
        in_dim = mlp_hidden
        for i in range(mlp_layers - 1):
            out_dim = max(mlp_hidden // (2 ** (i + 1)), 64)
            layers.extend([nn.Linear(in_dim, out_dim), nn.GELU(), nn.Dropout(dropout)])
            in_dim = out_dim

        layers.append(nn.Linear(in_dim, 1))
        self.mlp = nn.Sequential(*layers)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: [B, T, h, w, D] or [B, seq, D]

        Returns:
            score: [B, 1] physics plausibility logit
        """
        # Flatten spatial/temporal dims
        if features.dim() == 5:
            B, T, h, w, D = features.shape
            features = features.view(B, T * h * w, D)

        # Pool: [B, seq, D] -> [B, D]
        if self.pool_type == "mean":
            pooled = features.mean(dim=1)
        elif self.pool_type == "max":
            pooled = features.max(dim=1)[0]
        else:
            raise ValueError(f"Unknown pool_type: {self.pool_type}")

        # Project and classify
        x = self.proj(pooled)
        x = F.gelu(x)
        score = self.mlp(x)

        return score

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class PhysicsHeadTemporal(nn.Module):
    """
    Physics head with temporal attention.

    1. Spatial pool: [B, T, h, w, D] -> [B, T, D]
    2. Temporal transformer: model frame dependencies
    3. Output: [B, 1]
    """

    def __init__(
        self,
        hidden_dim: int = 1920,
        mlp_hidden: int = 512,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1,
        causal: bool = True,
    ):
        super().__init__()

        self.causal = causal

        # Project to smaller dim
        self.input_proj = nn.Linear(hidden_dim, mlp_hidden)

        # Temporal transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=mlp_hidden,
            nhead=num_heads,
            dim_feedforward=mlp_hidden * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # Output head
        self.output_head = nn.Sequential(
            nn.Linear(mlp_hidden, mlp_hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden // 2, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _get_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        return mask.masked_fill(mask == 1, float("-inf"))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: [B, T, h, w, D]

        Returns:
            score: [B, 1]
        """
        B, T, h, w, D = features.shape

        # Spatial pool
        x = features.mean(dim=(2, 3))  # [B, T, D]

        # Project
        x = self.input_proj(x)

        # Temporal attention
        if self.causal:
            mask = self._get_causal_mask(T, x.device)
            x = self.temporal_encoder(x, mask=mask)
        else:
            x = self.temporal_encoder(x)

        # Use last frame (has seen all previous)
        x = x[:, -1, :]

        # Output
        return self.output_head(x)

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class PhysicsHeadCausalPerFrame(nn.Module):
    """
    Per-frame physics scoring with causal attention.

    Output: Score for each frame based on frames 0..t (causal).
    Useful for trajectory pruning at checkpoints.
    """

    def __init__(
        self,
        hidden_dim: int = 1920,
        mlp_hidden: int = 512,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.input_proj = nn.Linear(hidden_dim, mlp_hidden)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=mlp_hidden,
            nhead=num_heads,
            dim_feedforward=mlp_hidden * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        self.frame_head = nn.Linear(mlp_hidden, 1)

    def _get_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        return mask.masked_fill(mask == 1, float("-inf"))

    def forward(
        self, features: torch.Tensor, return_all_frames: bool = False
    ) -> torch.Tensor:
        """
        Args:
            features: [B, T, h, w, D]
            return_all_frames: If True, return [B, T, 1]; else [B, 1]
        """
        B, T, h, w, D = features.shape

        # Spatial pool
        x = features.mean(dim=(2, 3))

        # Project
        x = self.input_proj(x)

        # Causal attention
        mask = self._get_causal_mask(T, x.device)
        x = self.temporal_encoder(x, mask=mask)

        # Per-frame scores
        scores = self.frame_head(x)  # [B, T, 1]

        if return_all_frames:
            return scores
        else:
            return scores[:, -1, :]

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# =============================================================================
# Factory
# =============================================================================


def create_physics_head(
    head_type: str = "simple", hidden_dim: int = 1920, **kwargs
) -> nn.Module:
    """
    Create physics head by type.

    Args:
        head_type: "simple", "temporal", "causal"
        hidden_dim: DiT hidden dimension (1920 for CogVideoX-2B)
    """
    if head_type == "simple":
        return PhysicsHead(hidden_dim=hidden_dim, **kwargs)
    elif head_type == "temporal":
        return PhysicsHeadTemporal(hidden_dim=hidden_dim, **kwargs)
    elif head_type == "causal":
        return PhysicsHeadCausalPerFrame(hidden_dim=hidden_dim, **kwargs)
    else:
        raise ValueError(f"Unknown head_type: {head_type}")


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    B, T, h, w, D = 2, 13, 30, 45, 1920
    features = torch.randn(B, T, h, w, D)

    print("=" * 60)
    print("Testing PhysicsHead (simple)")
    print("=" * 60)
    head = PhysicsHead(hidden_dim=D)
    out = head(features)
    print(f"  Input:  {list(features.shape)}")
    print(f"  Output: {list(out.shape)}")
    print(f"  Params: {head.get_num_params():,}")

    print("\n" + "=" * 60)
    print("Testing PhysicsHeadTemporal")
    print("=" * 60)
    head = PhysicsHeadTemporal(hidden_dim=D)
    out = head(features)
    print(f"  Input:  {list(features.shape)}")
    print(f"  Output: {list(out.shape)}")
    print(f"  Params: {head.get_num_params():,}")

    print("\n" + "=" * 60)
    print("Testing PhysicsHeadCausalPerFrame")
    print("=" * 60)
    head = PhysicsHeadCausalPerFrame(hidden_dim=D)
    out_all = head(features, return_all_frames=True)
    out_last = head(features, return_all_frames=False)
    print(f"  Input:       {list(features.shape)}")
    print(f"  Output (all): {list(out_all.shape)}")
    print(f"  Output (last): {list(out_last.shape)}")
    print(f"  Params: {head.get_num_params():,}")
