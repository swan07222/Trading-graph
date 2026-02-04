"""
State-of-the-Art Neural Network Architectures for Stock Prediction
Implements latest research for maximum prediction accuracy
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict
import math

from .advanced_layers import (
    FlashAttention, GatedMLP, MixtureOfExperts,
    TemporalFusionDecoder, UncertaintyHead, AdaptiveComputationTime
)
from .layers import AttentionPooling, ConfidenceHead


class MambaBlock(nn.Module):
    """
    Mamba-style State Space Model Block
    Linear time complexity - handles long sequences efficiently
    State-of-the-art for sequential data
    """
    def __init__(self, dim: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.dim = dim
        self.d_state = d_state
        d_inner = dim * expand
        
        self.in_proj = nn.Linear(dim, d_inner * 2, bias=False)
        
        self.conv1d = nn.Conv1d(
            d_inner, d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=d_inner
        )
        
        # SSM parameters
        self.x_proj = nn.Linear(d_inner, d_state * 2 + 1, bias=False)
        self.dt_proj = nn.Linear(1, d_inner, bias=True)
        
        # Initialize dt bias
        dt_init_std = 0.001
        nn.init.uniform_(self.dt_proj.bias, -dt_init_std, dt_init_std)
        
        self.A = nn.Parameter(torch.randn(d_inner, d_state))
        self.D = nn.Parameter(torch.ones(d_inner))
        
        self.out_proj = nn.Linear(d_inner, dim, bias=False)
        self.norm = nn.LayerNorm(dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        
        x_norm = self.norm(x)
        xz = self.in_proj(x_norm)
        x_inner, z = xz.chunk(2, dim=-1)
        
        # Conv
        x_conv = self.conv1d(x_inner.transpose(1, 2))[:, :, :L].transpose(1, 2)
        x_conv = F.silu(x_conv)
        
        # SSM
        x_ssm = self._ssm(x_conv)
        
        # Gate
        y = x_ssm * F.silu(z)
        
        return x + self.out_proj(y)
    
    def _ssm(self, x: torch.Tensor) -> torch.Tensor:
        """Selective State Space Model"""
        B, L, D = x.shape
        
        # Project to get B, C, dt
        x_proj = self.x_proj(x)
        delta, B_proj, C_proj = x_proj[..., :1], x_proj[..., 1:self.d_state+1], x_proj[..., self.d_state+1:]
        
        delta = F.softplus(self.dt_proj(delta))
        
        # Discretize
        A = -torch.exp(self.A.float())
        dA = torch.exp(delta * A.unsqueeze(0).unsqueeze(0))
        dB = delta * B_proj.unsqueeze(-1) * x.unsqueeze(-1)
        
        # Scan
        h = torch.zeros(B, D, self.d_state, device=x.device, dtype=x.dtype)
        ys = []
        
        for i in range(L):
            h = dA[:, i] * h + dB[:, i]
            y = (h * C_proj[:, i].unsqueeze(1)).sum(-1)
            ys.append(y)
        
        y = torch.stack(ys, dim=1)
        
        return y + x * self.D


class AdvancedLSTMModel(nn.Module):
    """
    Enhanced LSTM with modern improvements:
    - Layer normalization
    - Highway connections
    - Multi-scale temporal attention
    - Uncertainty estimation
    """
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 256,
        num_layers: int = 4,
        num_classes: int = 3,
        dropout: float = 0.3
    ):
        super().__init__()
        self.hidden_size = hidden_size
        
        # Input projection with normalization
        self.input_proj = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Multi-scale LSTM
        self.lstm_short = nn.LSTM(
            hidden_size, hidden_size // 2, 
            num_layers=2, batch_first=True, 
            dropout=dropout, bidirectional=True
        )
        
        self.lstm_long = nn.LSTM(
            hidden_size, hidden_size // 2,
            num_layers=2, batch_first=True,
            dropout=dropout, bidirectional=True
        )
        
        # Attention fusion
        self.fusion_attention = FlashAttention(hidden_size, num_heads=8, dropout=dropout)
        
        # Highway connection
        self.highway = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Sigmoid()
        )
        self.highway_h = nn.Linear(hidden_size * 2, hidden_size)
        
        # Temporal Fusion
        self.temporal_fusion = TemporalFusionDecoder(hidden_size, dropout=dropout)
        
        # Pooling
        self.pool = AttentionPooling(hidden_size)
        
        # Uncertainty-aware output
        self.uncertainty_head = UncertaintyHead(hidden_size, num_classes)
        
        # Standard confidence
        self.confidence = ConfidenceHead(hidden_size)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Input projection
        h = self.input_proj(x)
        
        # Multi-scale LSTM
        h_short, _ = self.lstm_short(h)
        
        # Downsample for long-term patterns
        h_down = h[:, ::2, :]  # Take every other step
        h_long, _ = self.lstm_long(h_down)
        # Upsample back
        h_long = F.interpolate(h_long.transpose(1, 2), size=h.size(1), mode='linear').transpose(1, 2)
        
        # Combine scales
        h_combined = torch.cat([h_short, h_long], dim=-1)
        
        # Highway connection
        gate = self.highway(h_combined)
        h_highway = gate * self.highway_h(h_combined) + (1 - gate) * h
        
        # Temporal fusion
        h_fused = self.temporal_fusion(h_highway)
        
        # Attention fusion
        h_attn = self.fusion_attention(h_fused)
        
        # Pool
        context = self.pool(h_attn)
        
        # Output with uncertainty
        logits, log_var, epistemic_conf = self.uncertainty_head(context)
        aleatoric_conf = self.confidence(context)
        
        # Combined confidence
        confidence = epistemic_conf * 0.5 + aleatoric_conf * 0.5
        
        return logits, confidence


class AdvancedTransformerModel(nn.Module):
    """
    State-of-the-art Transformer with:
    - Rotary Position Embeddings
    - Flash Attention
    - Mixture of Experts
    - Adaptive Computation Time
    """
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        num_classes: int = 3,
        dropout: float = 0.3,
        use_moe: bool = True
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.use_moe = use_moe
        
        # Input embedding
        self.input_proj = nn.Linear(input_size, hidden_size)
        self.input_norm = nn.LayerNorm(hidden_size)
        
        # Transformer blocks
        self.blocks = nn.ModuleList()
        for i in range(num_layers):
            self.blocks.append(
                AdvancedTransformerBlock(
                    hidden_size, num_heads, dropout,
                    use_moe=(use_moe and i % 2 == 1)  # MoE every other layer
                )
            )
        
        # Adaptive computation
        self.act = AdaptiveComputationTime(hidden_size)
        
        # Output
        self.output_norm = nn.LayerNorm(hidden_size)
        self.pool = AttentionPooling(hidden_size)
        self.uncertainty_head = UncertaintyHead(hidden_size, num_classes)
        self.confidence = ConfidenceHead(hidden_size)
        
        # Auxiliary loss accumulator
        self.aux_loss = 0.0
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Input projection
        h = self.input_norm(self.input_proj(x))
        
        # Transformer blocks with MoE
        self.aux_loss = 0.0
        for block in self.blocks:
            h, aux = block(h)
            self.aux_loss += aux
        
        # Adaptive computation
        h, ponder_cost = self.act(h)
        self.aux_loss += ponder_cost * 0.01
        
        # Output
        h = self.output_norm(h)
        context = self.pool(h)
        
        logits, log_var, epistemic_conf = self.uncertainty_head(context)
        aleatoric_conf = self.confidence(context)
        
        confidence = epistemic_conf * 0.5 + aleatoric_conf * 0.5
        
        return logits, confidence


class AdvancedTransformerBlock(nn.Module):
    """Single transformer block with optional MoE"""
    def __init__(self, dim: int, num_heads: int, dropout: float, use_moe: bool = False):
        super().__init__()
        self.use_moe = use_moe
        
        self.norm1 = nn.LayerNorm(dim)
        self.attn = FlashAttention(dim, num_heads, dropout)
        
        self.norm2 = nn.LayerNorm(dim)
        if use_moe:
            self.ffn = MixtureOfExperts(dim, num_experts=4, dropout=dropout)
        else:
            self.ffn = GatedMLP(dim, dropout=dropout)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, float]:
        # Attention
        x = x + self.dropout(self.attn(self.norm1(x)))
        
        # FFN (with or without MoE)
        aux_loss = 0.0
        if self.use_moe:
            ffn_out, aux_loss = self.ffn(self.norm2(x))
        else:
            ffn_out = self.ffn(self.norm2(x))
        
        x = x + self.dropout(ffn_out)
        
        return x, aux_loss


class MambaModel(nn.Module):
    """
    Mamba State Space Model
    Linear time complexity, excellent for long sequences
    """
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 256,
        num_layers: int = 6,
        num_classes: int = 3,
        dropout: float = 0.3
    ):
        super().__init__()
        
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size)
        )
        
        # Mamba blocks
        self.blocks = nn.ModuleList([
            MambaBlock(hidden_size) for _ in range(num_layers)
        ])
        
        # Output
        self.output_norm = nn.LayerNorm(hidden_size)
        self.pool = AttentionPooling(hidden_size)
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes)
        )
        
        self.confidence = ConfidenceHead(hidden_size)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.input_proj(x)
        
        for block in self.blocks:
            h = block(h)
        
        h = self.output_norm(h)
        context = self.pool(h)
        
        logits = self.classifier(context)
        conf = self.confidence(context)
        
        return logits, conf


class HybridMambaTransformer(nn.Module):
    """
    Hybrid model combining Mamba and Transformer
    Best of both worlds: efficiency and expressiveness
    """
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 256,
        num_classes: int = 3,
        dropout: float = 0.3
    ):
        super().__init__()
        
        # Shared input
        self.input_proj = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU()
        )
        
        # Mamba branch (local patterns)
        self.mamba_blocks = nn.ModuleList([
            MambaBlock(hidden_size) for _ in range(3)
        ])
        
        # Transformer branch (global patterns)
        self.transformer_blocks = nn.ModuleList([
            AdvancedTransformerBlock(hidden_size, 8, dropout, use_moe=False)
            for _ in range(3)
        ])
        
        # Cross-attention fusion
        self.cross_attn = FlashAttention(hidden_size, num_heads=8, dropout=dropout)
        
        # Gated fusion
        self.gate = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Sigmoid()
        )
        self.proj = nn.Linear(hidden_size * 2, hidden_size)
        
        # Output
        self.output_norm = nn.LayerNorm(hidden_size)
        self.pool = AttentionPooling(hidden_size)
        self.uncertainty_head = UncertaintyHead(hidden_size, num_classes)
        self.confidence = ConfidenceHead(hidden_size)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.input_proj(x)
        
        # Mamba branch
        h_mamba = h
        for block in self.mamba_blocks:
            h_mamba = block(h_mamba)
        
        # Transformer branch
        h_trans = h
        for block in self.transformer_blocks:
            h_trans, _ = block(h_trans)
        
        # Cross-attention fusion (Mamba attends to Transformer)
        h_cross = self.cross_attn(h_mamba + h_trans)
        
        # Gated fusion
        h_concat = torch.cat([h_mamba, h_trans], dim=-1)
        gate = self.gate(h_concat)
        h_fused = gate * self.proj(h_concat) + (1 - gate) * h_cross
        
        # Output
        h_out = self.output_norm(h_fused)
        context = self.pool(h_out)
        
        logits, log_var, epistemic_conf = self.uncertainty_head(context)
        aleatoric_conf = self.confidence(context)
        
        confidence = epistemic_conf * 0.5 + aleatoric_conf * 0.5
        
        return logits, confidence