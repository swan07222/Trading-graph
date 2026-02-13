# models/layers.py
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer models."""

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        if d_model <= 0:
            raise ValueError(f"d_model must be positive, got {d_model}")
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)

        # Handle both even and odd d_model correctly
        num_pairs = d_model // 2
        div_term = torch.exp(
            torch.arange(0, num_pairs * 2, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: (d_model + 1) // 2])

        # (1, max_len, d_model)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, d_model)"""
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Self-Attention.

    Uses PyTorch's scaled_dot_product_attention when available (torch >= 2.0)
    for flash-attention / memory-efficient kernels.
    """

    def __init__(self, d_model: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
            )

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.attn_dropout = dropout
        self.resid_dropout = nn.Dropout(dropout)

        self._use_sdpa = hasattr(F, "scaled_dot_product_attention")

    def forward(
        self,
        x: torch.Tensor,
        causal: bool = False,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Args:
            x: (batch, seq_len, d_model)
            causal: if True, apply causal (lower-triangular) mask
            mask: optional additive or boolean mask

        Returns:
            output: (batch, seq_len, d_model)
            attention_weights: (batch, heads, seq, seq) or None when using SDPA
        """
        B, S, _ = x.shape

        Q = self.q_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        attn_weights = None
        drop_p = self.attn_dropout if self.training else 0.0

        if self._use_sdpa:
            # Use fused kernel (flash / memory-efficient)
            attn_mask = None
            if mask is not None:
                attn_mask = mask
            context = F.scaled_dot_product_attention(
                Q, K, V,
                attn_mask=attn_mask,
                dropout_p=drop_p,
                is_causal=causal and attn_mask is None,
            )
        else:
            scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
            if causal:
                causal_mask = torch.triu(
                    torch.ones(S, S, device=x.device, dtype=torch.bool), diagonal=1
                )
                scores = scores.masked_fill(causal_mask, float("-inf"))
            if mask is not None:
                scores = scores.masked_fill(mask == 0, float("-inf"))
            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = F.dropout(attn_weights, p=drop_p, training=self.training)
            context = torch.matmul(attn_weights, V)

        context = context.transpose(1, 2).contiguous().view(B, S, self.d_model)
        output = self.resid_dropout(self.out_proj(context))
        return output, attn_weights

class FeedForward(nn.Module):
    """Position-wise Feed-Forward Network with SwiGLU option."""

    def __init__(
        self,
        d_model: int,
        d_ff: int | None = None,
        dropout: float = 0.1,
        use_glu: bool = False,
    ):
        super().__init__()
        d_ff = d_ff or d_model * 4

        if use_glu:
            self.net = nn.Sequential(
                _SwiGLU(d_model, d_ff),
                nn.Dropout(dropout),
                nn.Linear(d_ff, d_model),
                nn.Dropout(dropout),
            )
        else:
            self.net = nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_ff, d_model),
                nn.Dropout(dropout),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class _SwiGLU(nn.Module):
    """SwiGLU activation: SiLU(xW1) * xW2"""

    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.w1 = nn.Linear(d_in, d_out)
        self.w2 = nn.Linear(d_in, d_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.silu(self.w1(x)) * self.w2(x)

class TransformerBlock(nn.Module):
    """Pre-norm Transformer encoder block with optional causal masking."""

    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        causal: bool = False,
    ):
        super().__init__()
        self.causal = causal

        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, dropout=dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        attn_out, _ = self.attention(self.norm1(x), causal=self.causal, mask=mask)
        x = x + attn_out

        ff_out = self.feed_forward(self.norm2(x))
        x = x + ff_out

        return x

class LSTMBlock(nn.Module):
    """LSTM block with layer normalization.

    Args:
        bidirectional: Set False for strictly causal mode.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True,
    ):
        super().__init__()
        self.bidirectional = bidirectional
        self.output_size = hidden_size * (2 if bidirectional else 1)

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        self.norm = nn.LayerNorm(self.output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.lstm(x)
        output = self.norm(output)
        output = self.dropout(output)
        return output

class TemporalConvBlock(nn.Module):
    """Strictly causal temporal convolution block with dilated convolutions.

    Uses left-padding only so output at time t depends only on inputs <= t.
    GroupNorm is used instead of BatchNorm for stability with small / variable batch sizes.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.padding = (kernel_size - 1) * dilation

        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size, padding=0, dilation=dilation
        )
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size, padding=0, dilation=dilation
        )

        # GroupNorm — works with batch_size=1 unlike BatchNorm
        num_groups1 = _pick_num_groups(out_channels)
        num_groups2 = _pick_num_groups(out_channels)
        self.norm1 = nn.GroupNorm(num_groups1, out_channels)
        self.norm2 = nn.GroupNorm(num_groups2, out_channels)

        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

        self.residual = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, channels, seq_len)"""
        residual = self.residual(x)

        out = F.pad(x, (self.padding, 0))  # causal left-pad
        out = self.activation(self.norm1(self.conv1(out)))
        out = self.dropout(out)

        out = F.pad(out, (self.padding, 0))  # causal left-pad
        out = self.activation(self.norm2(self.conv2(out)))
        out = self.dropout(out)

        return self.activation(out + residual)

class AttentionPooling(nn.Module):
    """Attention-based pooling to aggregate a sequence into a single vector."""

    def __init__(self, hidden_size: int):
        super().__init__()
        bottleneck = max(1, hidden_size // 4)
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, bottleneck),
            nn.Tanh(),
            nn.Linear(bottleneck, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, hidden) -> (batch, hidden)"""
        weights = self.attention(x)  # (batch, seq, 1)
        weights = F.softmax(weights, dim=1)
        return torch.sum(weights * x, dim=1)

def _pick_num_groups(channels: int, preferred: int = 8) -> int:
    """Pick a valid num_groups for GroupNorm (must divide channels)."""
    for g in (preferred, 4, 2, 1):
        if channels % g == 0:
            return g
    return 1