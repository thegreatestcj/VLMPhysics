"""
Physics Discriminator Heads for Ablation Study

5 variants testing different hypotheses:
1. MeanPool: Simplest baseline
2. MeanPool+T: Add timestep conditioning  
3. TempAttn+T: Temporal attention (bidirectional)
4. CausalAttn+T: Causal temporal attention
5. MultiView+T: Multi-view pooling combination

Input: [B, T, H, W, D] = [B, 13, 30, 45, 1920]
Output: [B, 1] physics plausibility logit
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# =============================================================================
# Timestep Embedding (shared across heads)
# =============================================================================

def get_timestep_embedding(timesteps: torch.Tensor, dim: int = 128) -> torch.Tensor:
    """Sinusoidal timestep embedding, same as diffusion models."""
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=timesteps.device) / half)
    args = timesteps[:, None].float() * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


# =============================================================================
# 1. MeanPool: Simplest Baseline
# =============================================================================

class MeanPoolHead(nn.Module):
    """
    最简单的 baseline: 全部 mean pooling
    
    测试假设: DiT 特征是否本身就包含物理信息
    
    流程:
        [B, T, H, W, D] → mean(T,H,W) → [B, D] → MLP → [B, 1]
    """
    
    def __init__(self, hidden_dim: int = 1920, mlp_dim: int = 512):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, mlp_dim),
            nn.LayerNorm(mlp_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(mlp_dim, mlp_dim // 2),
            nn.GELU(),
            nn.Linear(mlp_dim // 2, 1)
        )
    
    def forward(self, features: torch.Tensor, timestep: Optional[torch.Tensor] = None):
        # [B, T, H, W, D] → [B, D]
        if features.dim() == 5:
            x = features.mean(dim=(1, 2, 3))
        elif features.dim() == 3:
            x = features.mean(dim=1)
        else:
            x = features
        
        return self.classifier(x)


# =============================================================================
# 2. MeanPool + Timestep: Add timestep conditioning
# =============================================================================

class MeanPoolTimestepHead(nn.Module):
    """
    Mean pooling + timestep conditioning
    
    测试假设: 知道噪声水平是否有帮助
    
    流程:
        [B, T, H, W, D] → mean → [B, D]
        timestep → embedding → [B, 128]
        concat → [B, D+128] → MLP → [B, 1]
    """
    
    def __init__(self, hidden_dim: int = 1920, mlp_dim: int = 512, t_dim: int = 128):
        super().__init__()
        self.t_dim = t_dim
        
        # Timestep MLP
        self.t_mlp = nn.Sequential(
            nn.Linear(t_dim, t_dim),
            nn.SiLU(),
            nn.Linear(t_dim, t_dim)
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim + t_dim, mlp_dim),
            nn.LayerNorm(mlp_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(mlp_dim, mlp_dim // 2),
            nn.GELU(),
            nn.Linear(mlp_dim // 2, 1)
        )
    
    def forward(self, features: torch.Tensor, timestep: torch.Tensor):
        # Spatial-temporal pooling: [B, T, H, W, D] → [B, D]
        if features.dim() == 5:
            x = features.mean(dim=(1, 2, 3))
        elif features.dim() == 3:
            x = features.mean(dim=1)
        else:
            x = features
        
        # Timestep embedding
        t_emb = get_timestep_embedding(timestep, self.t_dim)
        t_emb = self.t_mlp(t_emb)
        
        # Concat and classify
        x = torch.cat([x, t_emb], dim=-1)
        return self.classifier(x)


# =============================================================================
# 3. Temporal Attention + Timestep: Bidirectional temporal modeling
# =============================================================================

class TemporalAttnHead(nn.Module):
    """
    时序注意力 (双向) + timestep conditioning
    
    测试假设: 显式建模帧间关系是否有帮助
    
    流程:
        [B, T, H, W, D] → spatial_mean → [B, T, D]
        → temporal self-attention (bidirectional)
        → mean → [B, D]
        + timestep → MLP → [B, 1]
    """
    
    def __init__(
        self, 
        hidden_dim: int = 1920, 
        mlp_dim: int = 512, 
        t_dim: int = 128,
        num_heads: int = 8,
        num_layers: int = 2
    ):
        super().__init__()
        self.t_dim = t_dim
        
        # Project to smaller dim for attention
        self.input_proj = nn.Linear(hidden_dim, mlp_dim)
        
        # Learnable temporal position embedding
        self.pos_embed = nn.Parameter(torch.randn(1, 50, mlp_dim) * 0.02)
        
        # Temporal transformer (bidirectional)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=mlp_dim,
            nhead=num_heads,
            dim_feedforward=mlp_dim * 2,
            dropout=0.1,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.temporal_attn = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Timestep MLP
        self.t_mlp = nn.Sequential(
            nn.Linear(t_dim, t_dim),
            nn.SiLU(),
            nn.Linear(t_dim, t_dim)
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(mlp_dim + t_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, 1)
        )
    
    def forward(self, features: torch.Tensor, timestep: torch.Tensor):
        # Spatial pooling: [B, T, H, W, D] → [B, T, D]
        if features.dim() == 5:
            x = features.mean(dim=(2, 3))
        else:
            x = features  # assume already [B, T, D]
        
        B, T, D = x.shape
        
        # Project
        x = self.input_proj(x)  # [B, T, mlp_dim]
        
        # Add position embedding
        x = x + self.pos_embed[:, :T, :]
        
        # Temporal attention (bidirectional - no mask)
        x = self.temporal_attn(x)  # [B, T, mlp_dim]
        
        # Temporal pooling
        x = x.mean(dim=1)  # [B, mlp_dim]
        
        # Timestep
        t_emb = get_timestep_embedding(timestep, self.t_dim)
        t_emb = self.t_mlp(t_emb)
        
        # Classify
        x = torch.cat([x, t_emb], dim=-1)
        return self.classifier(x)


# =============================================================================
# 4. Causal Attention + Timestep: Causal temporal modeling
# =============================================================================

class CausalAttnHead(nn.Module):
    """
    因果时序注意力 + timestep conditioning
    
    测试假设: 物理因果性 (过去决定未来) 是否重要
    
    流程:
        [B, T, H, W, D] → spatial_mean → [B, T, D]
        → causal self-attention (每帧只能看过去)
        → last_frame → [B, D]  (因果: 最后帧包含所有历史)
        + timestep → MLP → [B, 1]
    """
    
    def __init__(
        self, 
        hidden_dim: int = 1920, 
        mlp_dim: int = 512, 
        t_dim: int = 128,
        num_heads: int = 8,
        num_layers: int = 2,
        max_frames: int = 50
    ):
        super().__init__()
        self.t_dim = t_dim
        
        # Project
        self.input_proj = nn.Linear(hidden_dim, mlp_dim)
        
        # Position embedding
        self.pos_embed = nn.Parameter(torch.randn(1, max_frames, mlp_dim) * 0.02)
        
        # Causal self-attention
        self.attn_layers = nn.ModuleList([
            nn.MultiheadAttention(mlp_dim, num_heads, dropout=0.1, batch_first=True)
            for _ in range(num_layers)
        ])
        self.ffn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(mlp_dim, mlp_dim * 2),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(mlp_dim * 2, mlp_dim),
                nn.Dropout(0.1)
            )
            for _ in range(num_layers)
        ])
        self.norms1 = nn.ModuleList([nn.LayerNorm(mlp_dim) for _ in range(num_layers)])
        self.norms2 = nn.ModuleList([nn.LayerNorm(mlp_dim) for _ in range(num_layers)])
        
        # Timestep
        self.t_mlp = nn.Sequential(
            nn.Linear(t_dim, t_dim),
            nn.SiLU(),
            nn.Linear(t_dim, t_dim)
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(mlp_dim + t_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, 1)
        )
    
    def _get_causal_mask(self, T: int, device: torch.device) -> torch.Tensor:
        """下三角 mask: 每帧只能 attend 到过去 (包括自己)"""
        mask = torch.triu(torch.ones(T, T, device=device), diagonal=1).bool()
        return mask  # True = 不能看
    
    def forward(self, features: torch.Tensor, timestep: torch.Tensor):
        # Spatial pooling
        if features.dim() == 5:
            x = features.mean(dim=(2, 3))  # [B, T, D]
        else:
            x = features
        
        B, T, D = x.shape
        
        # Project
        x = self.input_proj(x)
        
        # Position
        x = x + self.pos_embed[:, :T, :]
        
        # Causal mask
        causal_mask = self._get_causal_mask(T, x.device)
        
        # Causal attention layers
        for attn, ffn, norm1, norm2 in zip(
            self.attn_layers, self.ffn_layers, self.norms1, self.norms2
        ):
            # Self-attention with causal mask
            x_norm = norm1(x)
            attn_out, _ = attn(x_norm, x_norm, x_norm, attn_mask=causal_mask)
            x = x + attn_out
            
            # FFN
            x = x + ffn(norm2(x))
        
        # Take last frame (contains all causal history)
        x = x[:, -1, :]  # [B, mlp_dim]
        
        # Timestep
        t_emb = get_timestep_embedding(timestep, self.t_dim)
        t_emb = self.t_mlp(t_emb)
        
        # Classify
        x = torch.cat([x, t_emb], dim=-1)
        return self.classifier(x)


# =============================================================================
# 5. MultiView + Timestep: Combine multiple pooling strategies
# =============================================================================

class MultiViewHead(nn.Module):
    """
    多视角 pooling 组合 + timestep conditioning
    
    测试假设: 不同 pooling 捕捉互补信息
    
    组合:
        - mean_pool: 全局统计
        - max_pool: 最显著特征
        - first_frame: 初始状态
        - last_frame: 最终状态
        - diff_pool: 动态变化 (帧间差分)
        - causal_pool: 因果聚合
    
    流程:
        [B, T, H, W, D] → spatial_mean → [B, T, D]
        → 6 种 pooling → concat → [B, 6*D']
        + timestep → MLP → [B, 1]
    """
    
    def __init__(
        self, 
        hidden_dim: int = 1920, 
        mlp_dim: int = 512, 
        t_dim: int = 128,
        num_heads: int = 8
    ):
        super().__init__()
        self.t_dim = t_dim
        self.mlp_dim = mlp_dim
        
        # Project each view to smaller dim
        self.proj = nn.Linear(hidden_dim, mlp_dim)
        
        # Diff projection
        self.diff_proj = nn.Linear(hidden_dim, mlp_dim)
        
        # Causal attention for causal pooling
        self.causal_attn = nn.MultiheadAttention(
            mlp_dim, num_heads, dropout=0.1, batch_first=True
        )
        self.causal_norm = nn.LayerNorm(mlp_dim)
        
        # Timestep
        self.t_mlp = nn.Sequential(
            nn.Linear(t_dim, t_dim),
            nn.SiLU(),
            nn.Linear(t_dim, t_dim)
        )
        
        # 6 views * mlp_dim + t_dim
        concat_dim = mlp_dim * 6 + t_dim
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(concat_dim, mlp_dim),
            nn.LayerNorm(mlp_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(mlp_dim, mlp_dim // 2),
            nn.GELU(),
            nn.Linear(mlp_dim // 2, 1)
        )
    
    def _get_causal_mask(self, T: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.ones(T, T, device=device), diagonal=1).bool()
    
    def forward(self, features: torch.Tensor, timestep: torch.Tensor):
        # Spatial pooling: [B, T, H, W, D] → [B, T, D]
        if features.dim() == 5:
            x = features.mean(dim=(2, 3))
        else:
            x = features
        
        B, T, D = x.shape
        
        # ============================================
        # 6 种 Pooling
        # ============================================
        
        # 1. Mean pool: 全局统计
        mean_pool = self.proj(x.mean(dim=1))  # [B, mlp_dim]
        
        # 2. Max pool: 最显著特征
        max_pool = self.proj(x.max(dim=1)[0])  # [B, mlp_dim]
        
        # 3. First frame: 初始状态
        first_frame = self.proj(x[:, 0, :])  # [B, mlp_dim]
        
        # 4. Last frame: 最终状态
        last_frame = self.proj(x[:, -1, :])  # [B, mlp_dim]
        
        # 5. Diff pool: 动态变化
        diff = x[:, 1:, :] - x[:, :-1, :]  # [B, T-1, D]
        diff_pool = self.diff_proj(diff.mean(dim=1))  # [B, mlp_dim]
        
        # 6. Causal pool: 因果聚合
        x_proj = self.proj(x)  # [B, T, mlp_dim]
        causal_mask = self._get_causal_mask(T, x.device)
        causal_out, _ = self.causal_attn(x_proj, x_proj, x_proj, attn_mask=causal_mask)
        causal_pool = self.causal_norm(causal_out[:, -1, :])  # [B, mlp_dim]
        
        # ============================================
        # Timestep
        # ============================================
        t_emb = get_timestep_embedding(timestep, self.t_dim)
        t_emb = self.t_mlp(t_emb)
        
        # ============================================
        # Concat all views
        # ============================================
        combined = torch.cat([
            mean_pool,      # 全局
            max_pool,       # 显著
            first_frame,    # 初始
            last_frame,     # 最终
            diff_pool,      # 动态
            causal_pool,    # 因果
            t_emb           # 噪声水平
        ], dim=-1)  # [B, 6*mlp_dim + t_dim]
        
        return self.classifier(combined)


# =============================================================================
# Factory Function
# =============================================================================

HEAD_REGISTRY = {
    'mean': MeanPoolHead,
    'mean_t': MeanPoolTimestepHead,
    'temporal_t': TemporalAttnHead,
    'causal_t': CausalAttnHead,
    'multiview_t': MultiViewHead,
}


def create_head(head_type: str, **kwargs) -> nn.Module:
    """Create physics head by type."""
    if head_type not in HEAD_REGISTRY:
        raise ValueError(f"Unknown head: {head_type}. Choose from {list(HEAD_REGISTRY.keys())}")
    return HEAD_REGISTRY[head_type](**kwargs)


def get_num_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Physics Head Ablation Variants")
    print("=" * 70)
    
    # Test input: [B, T, H, W, D]
    B, T, H, W, D = 4, 13, 30, 45, 1920
    features = torch.randn(B, T, H, W, D)
    timestep = torch.randint(200, 800, (B,))
    
    print(f"\nInput: features {features.shape}, timestep {timestep.shape}\n")
    print(f"{'Head':<20} {'Params':>12} {'Output':>15} {'Needs timestep'}")
    print("-" * 60)
    
    for name, HeadClass in HEAD_REGISTRY.items():
        head = HeadClass()
        
        # Check if needs timestep
        if name == 'mean':
            out = head(features)
        else:
            out = head(features, timestep)
        
        needs_t = "No" if name == 'mean' else "Yes"
        print(f"{name:<20} {get_num_params(head):>12,} {str(list(out.shape)):>15} {needs_t}")
    
    print("\n" + "=" * 70)
    print("All heads tested successfully!")
    print("=" * 70)


