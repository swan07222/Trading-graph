# models/networks.py
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _safe_num_heads(dim: int, preferred: int = 8) -> int:
    """Return the largest num_heads <= preferred that divides dim."""
    for h in range(preferred, 0, -1):
        if dim % h == 0:
            return h
    return 1

def _pick_num_groups(channels: int, preferred: int = 8) -> int:
    """Pick valid num_groups for GroupNorm."""
    # FIX GROUPNORM: Guard against channels <= 0
    if channels <= 0:
        return 1
    for g in (preferred, 4, 2, 1):
        if channels % g == 0:
            return g
    return 1

def _init_weights(module: nn.Module):
    """Apply sensible weight initialization.

    FIX INIT: Skip LSTM/GRU modules — they use orthogonal initialization
    by default in PyTorch which is better for recurrent networks.
    """
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Conv1d):
        nn.init.kaiming_normal_(module.weight, nonlinearity="linear")
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm1d)):
        if module.weight is not None:
            nn.init.ones_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    # Skip LSTM, GRU — PyTorch default orthogonal init is preferred

def _count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Building blocks (previously in layers.py)

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for Transformer models.
    Strictly causal — only encodes position, no future information.
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 512):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model > 1:
            pe[:, 1::2] = torch.cos(position * div_term[:d_model // 2])

        # Register as buffer (not parameter, not trained, moves with .to())
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args:
            x: (batch, seq_len, d_model)
        Returns:
            (batch, seq_len, d_model) with positional encoding added
        """
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)

class TransformerBlock(nn.Module):
    """Single Transformer encoder block with optional causal masking.

    FIX ATTN_MASK: Causal mask is created dynamically to handle
    variable sequence lengths, and shaped correctly for
    nn.MultiheadAttention (which expects (seq, seq) or
    (batch*heads, seq, seq)).
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1,
        causal: bool = True,
        ff_mult: int = 4,
    ):
        super().__init__()
        self.causal = causal

        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * ff_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * ff_mult, d_model),
            nn.Dropout(dropout),
        )

        self._mask_cache: torch.Tensor | None = None
        self._mask_cache_size: int = 0

    def _get_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Get or create causal attention mask.

        Returns upper-triangular mask where True means "do not attend".
        Shape: (seq_len, seq_len) — broadcast over batch and heads.
        """
        if self._mask_cache is not None and self._mask_cache_size >= seq_len:
            return self._mask_cache[:seq_len, :seq_len]

        # Create upper-triangular mask (True = masked/blocked)
        mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device, dtype=torch.bool),
            diagonal=1,
        )
        self._mask_cache = mask
        self._mask_cache_size = seq_len
        return mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args:
            x: (batch, seq_len, d_model)
        Returns:
            (batch, seq_len, d_model)
        """
        attn_mask = None
        if self.causal:
            attn_mask = self._get_causal_mask(x.size(1), x.device)

        # Pre-norm architecture (more stable training)
        normed = self.norm1(x)
        attn_out, _ = self.attn(
            normed, normed, normed,
            attn_mask=attn_mask,
            need_weights=False,
        )
        x = x + attn_out

        normed = self.norm2(x)
        x = x + self.ff(normed)

        return x

class LSTMBlock(nn.Module):
    """LSTM block with proper dropout handling.

    FIX LSTM_DROPOUT: When num_layers=1, PyTorch warns that dropout
    has no effect. We set dropout=0 in that case.
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

        # FIX LSTM_DROPOUT: No dropout for single layer
        effective_dropout = dropout if num_layers > 1 else 0.0

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=effective_dropout,
            bidirectional=bidirectional,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args:
            x: (batch, seq_len, input_size)
        Returns:
            (batch, seq_len, hidden_size * num_directions)
        """
        output, _ = self.lstm(x)
        return self.dropout(output)

class TemporalConvBlock(nn.Module):
    """Causal temporal convolution block with residual connection.

    Uses left-padding to ensure strict causality: output at time t
    depends only on inputs at times <= t.

    FIX TCN_RESIDUAL: Uses 1x1 convolution for residual path when
    input and output channels differ, instead of silently dropping
    the residual connection.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation  # Left padding for causality

        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            dilation=dilation, padding=0,
        )
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size,
            dilation=dilation, padding=0,
        )
        self.norm1 = nn.GroupNorm(_pick_num_groups(out_channels), out_channels)
        self.norm2 = nn.GroupNorm(_pick_num_groups(out_channels), out_channels)
        self.dropout = nn.Dropout(dropout)

        # FIX TCN_RESIDUAL: 1x1 conv when channels change
        if in_channels != out_channels:
            self.residual_proj = nn.Conv1d(in_channels, out_channels, 1)
        else:
            self.residual_proj = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args:
            x: (batch, channels, seq_len)
        Returns:
            (batch, out_channels, seq_len)
        """
        residual = x
        if self.residual_proj is not None:
            residual = self.residual_proj(residual)

        # Causal padding (left only)
        out = F.pad(x, (self.padding, 0))
        out = self.conv1(out)
        out = self.norm1(out)
        out = F.gelu(out)
        out = self.dropout(out)

        out = F.pad(out, (self.padding, 0))
        out = self.conv2(out)
        out = self.norm2(out)
        out = F.gelu(out)
        out = self.dropout(out)

        # Ensure sequence lengths match (should be same due to causal padding)
        min_len = min(out.size(2), residual.size(2))
        out = out[:, :, :min_len] + residual[:, :, :min_len]

        return out

class AttentionPooling(nn.Module):
    """Attention-based pooling over the sequence dimension.

    Learns to weight different time steps, producing a fixed-size
    representation from variable-length sequences.
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args:
            x: (batch, seq_len, hidden_size)
        Returns:
            (batch, hidden_size) — weighted average over time
        """
        scores = self.attention(x)  # (batch, seq_len, 1)
        weights = F.softmax(scores, dim=1)  # (batch, seq_len, 1)

        pooled = (x * weights).sum(dim=1)  # (batch, hidden_size)
        return pooled

class _ClassifierHead(nn.Module):
    """Shared classification + confidence head."""

    def __init__(self, in_features: int, hidden: int, num_classes: int, dropout: float):
        super().__init__()
        # Ensure hidden dimension is at least 1
        hidden = max(1, hidden)

        self.classifier = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

        conf_hidden = max(1, hidden // 4)
        self.confidence = nn.Sequential(
            nn.Linear(in_features, conf_hidden),
            nn.GELU(),
            nn.Linear(conf_hidden, 1),
            nn.Sigmoid(),
        )

    def forward(self, pooled: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.classifier(pooled), self.confidence(pooled)

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

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Args:
            x: (batch, seq_len, input_size)
        Returns:
            (logits, confidence) where logits is (batch, num_classes)
            and confidence is (batch, 1)
        """
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
        self.head = _ClassifierHead(
            hidden_size, max(1, hidden_size // 2), num_classes, dropout
        )

        self.apply(_init_weights)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Args:
            x: (batch, seq_len, input_size)
        Returns:
            (logits, confidence)
        """
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
        num_layers: int = 2,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_size, hidden_size)

        # FIX LSTM_DROPOUT: Same fix for GRU — no dropout for single layer
        effective_dropout = dropout if num_layers > 1 else 0.0

        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=effective_dropout,
            bidirectional=True,
        )

        gru_out = hidden_size * 2
        self.norm = nn.LayerNorm(gru_out)
        self.pool = AttentionPooling(gru_out)
        self.head = _ClassifierHead(gru_out, hidden_size, num_classes, dropout)

        self.apply(_init_weights)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Args:
            x: (batch, seq_len, input_size)
        Returns:
            (logits, confidence)
        """
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
        num_blocks: int = 4,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_size, hidden_size)

        self.tcn_blocks = nn.ModuleList(
            [
                TemporalConvBlock(
                    hidden_size, hidden_size,
                    dilation=2**i, dropout=dropout,
                )
                for i in range(num_blocks)  # dilations: 1, 2, 4, 8
            ]
        )

        self.head = _ClassifierHead(
            hidden_size, max(1, hidden_size // 2), num_classes, dropout
        )

        self.apply(_init_weights)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Args:
            x: (batch, seq_len, input_size)
        Returns:
            (logits, confidence)
        """
        x = self.input_proj(x)
        x = x.transpose(1, 2)  # (batch, hidden, seq)

        for block in self.tcn_blocks:
            x = block(x)

        pooled = x[:, :, -1]  # (batch, hidden)
        return self.head(pooled)

class HybridModel(nn.Module):
    """Causal CNN + LSTM hybrid model.

    CNN uses causal (left-only) padding to prevent future leakage.

    FIX HYBRID_RESIDUAL: Causal padding preserves sequence length,
    so the residual addition is always valid. Explicit length check
    added as safety guard.
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
        self.conv_dropout = nn.Dropout(dropout)

        # FIX LSTM_DROPOUT: num_layers=2 so dropout is fine
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
        self.head = _ClassifierHead(
            hidden_size, max(1, hidden_size // 2), num_classes, dropout
        )

        self.apply(_init_weights)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Args:
            x: (batch, seq_len, input_size)
        Returns:
            (logits, confidence)
        """
        x = self.input_proj(x)

        x_t = x.transpose(1, 2)  # (batch, hidden, seq)
        c1 = F.gelu(self.conv1(F.pad(x_t, (self.conv1_pad, 0))))
        c2 = F.gelu(self.conv2(F.pad(x_t, (self.conv2_pad, 0))))

        # Both c1 and c2 should have same seq length as x_t due to causal padding
        conv_out = self.conv_norm(c1 + c2)
        conv_out = self.conv_dropout(conv_out)
        conv_out = conv_out.transpose(1, 2)  # (batch, seq, hidden)

        # FIX HYBRID_RESIDUAL: Safety check for sequence length match
        min_len = min(x.size(1), conv_out.size(1))
        x = x[:, :min_len, :] + conv_out[:, :min_len, :]

        x, _ = self.lstm(x)
        x = self.norm(x)

        pooled = self.pool(x)
        return self.head(pooled)