# models/networks.py
"""
Neural Network Architectures for Time-Series Stock Prediction.

Design principles:
    - All architectures are CAUSAL — no future information leakage.
    - Every model returns (logits, confidence) for a unified interface.
    - Head dimensions are validated at init time to prevent runtime crashes.
    - Xavier / Kaiming init applied for stable training.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

from .layers import (
    PositionalEncoding,
    TransformerBlock,
    LSTMBlock,
    TemporalConvBlock,
    AttentionPooling,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _safe_num_heads(dim: int, preferred: int = 8) -> int:
    """Return the largest num_heads <= preferred that divides dim."""
    for h in range(preferred, 0, -1):
        if dim % h == 0:
            return h
    return 1


def _init_weights(module: nn.Module):
    """Apply sensible weight initialization."""
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Conv1d):
        nn.init.kaiming_normal_(module.weight, nonlinearity="linear")
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm1d)):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


def _count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class _ClassifierHead(nn.Module):
    """Shared classification + confidence head."""

    def __init__(self, in_features: int, hidden: int, num_classes: int, dropout: float):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )
        self.confidence = nn.Sequential(
            nn.Linear(in_features, max(1, hidden // 4)),
            nn.GELU(),
            nn.Linear(max(1, hidden // 4), 1),
            nn.Sigmoid(),
        )

    def forward(self, pooled: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.classifier(pooled), self.confidence(pooled)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class LSTMModel(nn.Module):
    """Bidirectional LSTM with self-attention pooling."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 256,
        num_classes: int = 3,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_size, hidden_size)

        self.lstm = LSTMBlock(
            hidden_size, hidden_size, num_layers=2,
            dropout=dropout, bidirectional=True,
        )

        lstm_out = hidden_size * 2  # bidirectional
        self.pool = AttentionPooling(lstm_out)
        self.head = _ClassifierHead(lstm_out, hidden_size, num_classes, dropout)

        self.apply(_init_weights)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.input_proj(x)
        x = self.lstm(x)
        pooled = self.pool(x)
        return self.head(pooled)


class TransformerModel(nn.Module):
    """Causal Transformer encoder for sequence classification."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 256,
        num_classes: int = 3,
        dropout: float = 0.3,
        num_layers: int = 4,
        num_heads: int = 8,
    ):
        super().__init__()
        # Ensure head count is valid
        num_heads = _safe_num_heads(hidden_size, num_heads)

        self.input_proj = nn.Linear(input_size, hidden_size)
        self.pos_encoding = PositionalEncoding(hidden_size, dropout=dropout)

        self.layers = nn.ModuleList(
            [
                TransformerBlock(hidden_size, num_heads, dropout, causal=True)
                for _ in range(num_layers)
            ]
        )

        self.norm = nn.LayerNorm(hidden_size)
        self.pool = AttentionPooling(hidden_size)
        self.head = _ClassifierHead(hidden_size, hidden_size // 2, num_classes, dropout)

        self.apply(_init_weights)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.input_proj(x)
        x = self.pos_encoding(x)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)
        pooled = self.pool(x)
        return self.head(pooled)


class GRUModel(nn.Module):
    """Bidirectional GRU with attention pooling."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 256,
        num_classes: int = 3,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_size, hidden_size)

        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
            bidirectional=True,
        )

        gru_out = hidden_size * 2
        self.norm = nn.LayerNorm(gru_out)
        self.pool = AttentionPooling(gru_out)
        self.head = _ClassifierHead(gru_out, hidden_size, num_classes, dropout)

        self.apply(_init_weights)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.input_proj(x)
        x, _ = self.gru(x)
        x = self.norm(x)
        pooled = self.pool(x)
        return self.head(pooled)


class TCNModel(nn.Module):
    """Temporal Convolutional Network (strictly causal).

    Uses the LAST timestep output (not global average pool) to preserve
    the causal property of the dilated convolutions.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 256,
        num_classes: int = 3,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_size, hidden_size)

        self.tcn_blocks = nn.ModuleList(
            [
                TemporalConvBlock(hidden_size, hidden_size, dilation=2**i, dropout=dropout)
                for i in range(4)  # dilations: 1, 2, 4, 8
            ]
        )

        self.head = _ClassifierHead(hidden_size, hidden_size // 2, num_classes, dropout)

        self.apply(_init_weights)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.input_proj(x)
        x = x.transpose(1, 2)  # (batch, hidden, seq)

        for block in self.tcn_blocks:
            x = block(x)

        # Take LAST timestep to preserve causality
        pooled = x[:, :, -1]  # (batch, hidden)
        return self.head(pooled)


class HybridModel(nn.Module):
    """Causal CNN + LSTM hybrid model.

    CNN uses causal (left-only) padding to prevent future leakage.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 256,
        num_classes: int = 3,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_size, hidden_size)

        # Causal convolutions (left-padding only)
        self.conv1_pad = 2  # kernel_size - 1 for k=3
        self.conv1 = nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=0)
        self.conv2_pad = 4  # kernel_size - 1 for k=5
        self.conv2 = nn.Conv1d(hidden_size, hidden_size, kernel_size=5, padding=0)

        self.conv_norm = nn.GroupNorm(_pick_num_groups(hidden_size), hidden_size)

        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size // 2,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
            bidirectional=True,
        )

        self.norm = nn.LayerNorm(hidden_size)
        self.pool = AttentionPooling(hidden_size)
        self.head = _ClassifierHead(hidden_size, hidden_size // 2, num_classes, dropout)

        self.apply(_init_weights)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.input_proj(x)

        # Causal CNN path
        x_t = x.transpose(1, 2)  # (batch, hidden, seq)
        c1 = F.gelu(self.conv1(F.pad(x_t, (self.conv1_pad, 0))))
        c2 = F.gelu(self.conv2(F.pad(x_t, (self.conv2_pad, 0))))
        conv_out = self.conv_norm(c1 + c2).transpose(1, 2)

        # Residual connection
        x = x + conv_out

        # LSTM path
        x, _ = self.lstm(x)
        x = self.norm(x)

        pooled = self.pool(x)
        return self.head(pooled)


# Re-export helper used by HybridModel
def _pick_num_groups(channels: int, preferred: int = 8) -> int:
    for g in (preferred, 4, 2, 1):
        if channels % g == 0:
            return g
    return 1