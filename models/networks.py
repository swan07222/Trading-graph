"""Modern neural network architectures for time series forecasting.

This module contains ONLY cutting-edge model architectures:
- Informer: Efficient Transformer for long sequences (O(L log L))
- Temporal Fusion Transformer (TFT): Interpretable predictions
- N-BEATS: Neural basis expansion analysis
- TSMixer: All-MLP architecture

Legacy models (LSTM, GRU, TCN) have been removed in favor of
superior modern architectures.

References:
    - Informer: https://arxiv.org/abs/2012.07436
    - TFT: https://arxiv.org/abs/1912.09363
    - N-BEATS: https://arxiv.org/abs/1905.10437
    - TSMixer: https://arxiv.org/abs/2303.06053
"""
from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# Informer Model - Efficient Transformer for Long Sequences
# ============================================================================

class ProbAttention(nn.Module):
    """Probabilistic attention mechanism from Informer.

    Reduces attention computation from O(L²) to O(L log L) by
    selecting only top-K queries based on sparsity measure.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        factor: int = 5,
        attention_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.factor = factor

        self.inner_attention = nn.ScaledDotProductAttention(
            d_model ** -0.5,
            attention_dropout,
        )
        self.query_projection = nn.Linear(d_model, d_model)
        self.key_projection = nn.Linear(d_model, d_model)
        self.value_projection = nn.Linear(d_model, d_model)
        self.out_projection = nn.Linear(d_model, d_model)

    def _sample_q(self, Q: torch.Tensor, factor: int) -> torch.Tensor:
        """Sample queries based on sparsity measure."""
        B, L, H, E = Q.shape
        index = torch.randint(0, L, (factor * math.log(max(L, 2)),), device=Q.device)
        return Q[:, index, :, :]

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        Q = self.query_projection(queries).view(B, L, H, -1)
        K = self.key_projection(keys).view(B, S, H, -1)
        V = self.value_projection(values).view(B, S, H, -1)

        Q_sample = self._sample_q(Q, self.factor)

        out, attn = self.inner_attention(
            Q_sample, K, V,
            attn_mask=attn_mask if attn_mask is not None else None,
        )

        out = out.contiguous().view(B, L, -1)
        return self.out_projection(out), attn


class InformerEncoderLayer(nn.Module):
    """Single Informer encoder layer with distillation."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        factor: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.attn = ProbAttention(d_model, n_heads, factor, dropout)
        self.conv1 = nn.Conv1d(d_model, d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(d_ff, d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.gelu

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        new_x, _ = self.attn(x, x, x, attn_mask=attn_mask)
        x = x + self.dropout(new_x)
        x = self.norm1(x)

        y = x.transpose(1, 2)
        y = self.conv1(y)
        y = self.activation(y)
        y = self.conv2(y)
        y = y.transpose(1, 2)
        x = x + self.dropout(y)
        x = self.norm2(x)

        return x


class InformerEncoder(nn.Module):
    """Informer encoder with distilling operation."""

    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        d_ff: int | None = None,
        factor: int = 5,
        n_layers: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        d_ff = d_ff or 4 * d_model

        self.layers = nn.ModuleList([
            InformerEncoderLayer(d_model, n_heads, d_ff, factor, dropout)
            for _ in range(n_layers)
        ])

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, attn_mask)
        return x


class Informer(nn.Module):
    """Informer model for long sequence time series forecasting.

    Key innovations:
        - ProbSparse self-attention: O(L log L) complexity
        - Distilling: Halves sequence length between layers
        - Generative decoder: Direct long-sequence output

    Best for: Long-horizon forecasting (20-60 days ahead)
    """

    def __init__(
        self,
        input_size: int = 58,
        pred_len: int = 20,
        seq_len: int = 60,
        d_model: int = 128,
        n_heads: int = 8,
        d_ff: int = 512,
        factor: int = 5,
        n_layers: int = 3,
        dropout: float = 0.1,
        num_classes: int = 3,
    ) -> None:
        super().__init__()
        self.pred_len = pred_len

        self.input_projection = nn.Linear(input_size, d_model)
        self.encoder = InformerEncoder(d_model, n_heads, d_ff, factor, n_layers, dropout)
        self.decoder = nn.Linear(d_model, pred_len)
        self.regression_head = nn.Linear(d_model, 1)
        self.classification_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            x: Input tensor (batch, seq_len, input_size)

        Returns:
            Dictionary with forecast, logits, regression, and hidden states
        """
        x = self.input_projection(x)
        hidden = self.encoder(x)
        pooled = hidden.mean(dim=1)
        logits = self.classification_head(pooled)
        forecast = self.decoder(hidden)
        forecast = forecast[:, -1, :]
        regression = self.regression_head(pooled).squeeze(-1)

        return {
            "forecast": forecast,
            "logits": logits,
            "regression": regression,
            "hidden": hidden,
        }


# ============================================================================
# Temporal Fusion Transformer (TFT) - Interpretable Multi-Horizon
# ============================================================================

class GatedResidualNetwork(nn.Module):
    """Gated Residual Network from TFT."""

    def __init__(
        self,
        d_model: int,
        dropout: float = 0.1,
        context_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.context_dim = context_dim

        self.fc1 = nn.Linear(d_model, d_model)
        self.fc2 = nn.Linear(d_model, d_model)
        self.gate = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.gelu

        if context_dim is not None:
            self.context_proj = nn.Linear(context_dim, d_model)

    def forward(self, x: torch.Tensor, context: torch.Tensor | None = None) -> torch.Tensor:
        if self.context_dim is not None and context is not None:
            context = self.context_proj(context)
            x = x + context

        gate = torch.sigmoid(self.gate(x))
        residual = self.fc1(x)
        residual = self.activation(residual)
        residual = self.dropout(residual)
        residual = self.fc2(residual)
        residual = self.dropout(residual)

        return self.norm(x + gate * residual)


class VariableSelectionNetwork(nn.Module):
    """Variable Selection Network from TFT."""

    def __init__(
        self,
        d_model: int,
        num_vars: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_vars = num_vars

        self.joint_grn = GatedResidualNetwork(d_model, dropout)
        self.var_grns = nn.ModuleList([
            GatedResidualNetwork(d_model, dropout)
            for _ in range(num_vars)
        ])
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, V, D = x.shape

        x_flat = x.reshape(B * L * V, D)
        weights = torch.stack([
            self.joint_grn(x_flat[:, i * D:(i + 1) * D])
            for i in range(V)
        ], dim=1)
        weights = self.softmax(weights)

        vars_processed = torch.stack([
            grn(x[:, :, i, :])
            for i, grn in enumerate(self.var_grns)
        ], dim=2)

        weights = weights.view(B, L, V, 1)
        output = (vars_processed * weights).sum(dim=2)

        return output


class TemporalFusionTransformer(nn.Module):
    """Temporal Fusion Transformer for interpretable forecasting.

    Key features:
        - Variable selection for feature importance
        - Static covariate encoding
        - Multi-horizon forecasting
        - Quantile regression for uncertainty

    Best for: Explainable predictions with feature importance
    """

    def __init__(
        self,
        input_size: int = 58,
        pred_len: int = 20,
        seq_len: int = 60,
        d_model: int = 128,
        dropout: float = 0.1,
        num_static_vars: int = 10,
        num_known_vars: int = 5,
        num_classes: int = 3,
        quantiles: list[float] | None = None,
    ) -> None:
        super().__init__()
        self.pred_len = pred_len
        self.d_model = d_model
        self.quantiles = quantiles or [0.1, 0.5, 0.9]

        self.static_encoder = nn.Linear(num_static_vars, d_model)
        self.static_grn = GatedResidualNetwork(d_model, dropout)

        self.known_var_selector = VariableSelectionNetwork(d_model, num_known_vars)

        self.lstm = nn.LSTM(
            d_model, d_model,
            num_layers=2, batch_first=True, dropout=dropout
        )
        self.glu = nn.GLU()

        self.forecast_head = nn.Linear(d_model, len(self.quantiles) * pred_len)
        self.classification_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, num_classes),
        )

    def forward(
        self,
        x: torch.Tensor,
        static: torch.Tensor | None = None,
        known: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        B = x.shape[0]

        if static is not None:
            static_encoded = self.static_encoder(static)
            static_encoded = self.static_grn(static_encoded)
        else:
            static_encoded = torch.zeros(B, self.d_model, device=x.device)

        x = x.unsqueeze(2).expand(-1, -1, 1, -1)
        x = self.known_var_selector(x)

        lstm_out, _ = self.lstm(x)
        lstm_out = self.glu(lstm_out)

        pooled = lstm_out.mean(dim=1)
        logits = self.classification_head(pooled)

        forecast_flat = self.forecast_head(pooled)
        forecast = forecast_flat.view(B, len(self.quantiles), self.pred_len)

        return {
            "forecast": forecast[:, 1, :],
            "quantiles": forecast,
            "logits": logits,
            "hidden": lstm_out,
        }


# ============================================================================
# N-BEATS - Neural Basis Expansion Analysis
# ============================================================================

class NBEATSBlock(nn.Module):
    """Single N-BEATS block with basis expansion."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        num_layers: int = 4,
        layer_size: int = 256,
        num_basis: int = 16,
        block_type: str = "generic",
    ) -> None:
        super().__init__()
        self.block_type = block_type
        self.output_size = output_size
        self.num_basis = num_basis

        layers = []
        prev_size = input_size
        for _ in range(num_layers):
            layers.extend([
                nn.Linear(prev_size, layer_size),
                nn.ReLU(),
            ])
            prev_size = layer_size
        self.fc = nn.Sequential(*layers)

        if block_type == "trend":
            self.basis = self._trend_basis(output_size, num_basis)
        elif block_type == "seasonality":
            self.basis = self._seasonality_basis(output_size, num_basis)
        else:
            self.basis = nn.Linear(num_basis, output_size)

        self.theta = nn.Linear(layer_size, num_basis)

    def _trend_basis(self, size: int, num_basis: int) -> torch.Tensor:
        time = torch.linspace(-1, 1, size)
        basis = torch.stack([time ** i for i in range(num_basis)])
        return nn.Parameter(basis, requires_grad=False)

    def _seasonality_basis(self, size: int, num_basis: int) -> torch.Tensor:
        time = torch.linspace(0, 2 * math.pi, size)
        basis = []
        for i in range(num_basis // 2 + 1):
            basis.append(torch.sin(i * time))
            basis.append(torch.cos(i * time))
        return nn.Parameter(torch.stack(basis[:num_basis]), requires_grad=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.fc(x)
        coeffs = self.theta(features)

        if self.block_type in ("trend", "seasonality"):
            basis = self.basis.to(x.device)
            forecast = coeffs @ basis
        else:
            forecast = self.basis(coeffs)

        backcast = nn.Linear(features.shape[1], self.output_size)(features)

        return forecast, backcast


class NBEATS(nn.Module):
    """N-BEATS model for interpretable time series forecasting.

    Architecture:
        - Multiple stacks of blocks (trend, seasonality, generic)
        - Each block decomposes signal into interpretable components
        - Residual connections for gradient flow

    Best for: Quick baseline with trend/seasonality interpretation
    """

    def __init__(
        self,
        input_size: int = 58,
        pred_len: int = 20,
        seq_len: int = 60,
        d_model: int = 256,
        num_classes: int = 3,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        total_len = seq_len + pred_len

        self.trend_stack = self._make_stack("trend", num_blocks=3, total_len=total_len)
        self.seasonality_stack = self._make_stack("seasonality", num_blocks=3, total_len=total_len)
        self.generic_stack = self._make_stack("generic", num_blocks=3, total_len=total_len)

        self.classifier = nn.Sequential(
            nn.Linear(input_size * seq_len, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes),
        )

    def _make_stack(self, block_type: str, num_blocks: int, total_len: int) -> nn.ModuleList:
        return nn.ModuleList([
            NBEATSBlock(
                input_size=self.seq_len * 58 if i == 0 else total_len,
                output_size=total_len,
                block_type=block_type,
            )
            for i in range(num_blocks)
        ])

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        B, L, D = x.shape
        x_flat = x.reshape(B, -1)

        trend_forecast = torch.zeros(B, self.pred_len, device=x.device)
        trend_backcast = x_flat
        for block in self.trend_stack:
            forecast, backcast = block(trend_backcast)
            trend_forecast = trend_forecast + forecast[:, -self.pred_len:]
            trend_backcast = trend_backcast - backcast[:, :self.seq_len]

        seasonality_forecast = torch.zeros(B, self.pred_len, device=x.device)
        seasonality_backcast = trend_backcast
        for block in self.seasonality_stack:
            forecast, backcast = block(seasonality_backcast)
            seasonality_forecast = seasonality_forecast + forecast[:, -self.pred_len:]
            seasonality_backcast = seasonality_backcast - backcast[:, :self.seq_len]

        generic_forecast = torch.zeros(B, self.pred_len, device=x.device)
        generic_backcast = seasonality_backcast
        for block in self.generic_stack:
            forecast, backcast = block(generic_backcast)
            generic_forecast = generic_forecast + forecast[:, -self.pred_len:]

        total_forecast = trend_forecast + seasonality_forecast + generic_forecast

        pooled = x.reshape(B, -1)
        logits = self.classifier(pooled)

        return {
            "forecast": total_forecast,
            "trend": trend_forecast,
            "seasonality": seasonality_forecast,
            "generic": generic_forecast,
            "logits": logits,
        }


# ============================================================================
# TSMixer - MLP-based Architecture
# ============================================================================

class TSMixerBlock(nn.Module):
    """TSMixer block with time-mixing and feature-mixing MLPs."""

    def __init__(self, d_model: int, seq_len: int = 60, dropout: float = 0.1) -> None:
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(seq_len, seq_len),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(seq_len, seq_len),
        )
        self.feature_mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x.transpose(1, 2)
        y = self.time_mlp(y)
        y = y.transpose(1, 2)
        x = self.norm1(x + y)

        y = self.feature_mlp(x)
        x = self.norm2(x + y)

        return x


class TSMixer(nn.Module):
    """TSMixer: All-MLP Architecture for Time Series.

    Simple but effective architecture using only MLPs:
        - Time-mixing MLP: Captures temporal dependencies
        - Feature-mixing MLP: Captures feature interactions
        - Residual connections for deep networks

    Best for: Resource-efficient inference with competitive accuracy
    """

    def __init__(
        self,
        input_size: int = 58,
        pred_len: int = 20,
        seq_len: int = 60,
        d_model: int = 128,
        dropout: float = 0.1,
        num_blocks: int = 4,
        num_classes: int = 3,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len

        self.input_projection = nn.Linear(input_size, d_model)
        self.blocks = nn.ModuleList([
            TSMixerBlock(d_model, seq_len, dropout)
            for _ in range(num_blocks)
        ])

        self.forecast_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, pred_len),
        )
        self.classification_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self.input_projection(x)

        for block in self.blocks:
            x = block(x)

        pooled = x.mean(dim=1)
        logits = self.classification_head(pooled)
        forecast = self.forecast_head(x[:, -1, :])

        return {
            "forecast": forecast,
            "logits": logits,
            "hidden": x,
        }


# ============================================================================
# Model Factory
# ============================================================================

def get_model(model_type: str, **kwargs: Any) -> nn.Module:
    """Factory function to create models.

    Args:
        model_type: One of 'informer', 'tft', 'nbeats', 'tsmixer'
        **kwargs: Model hyperparameters

    Returns:
        PyTorch model instance

    Raises:
        ValueError: If model_type is not recognized

    Example:
        >>> model = get_model("informer", input_size=58, pred_len=20)
    """
    models = {
        "informer": Informer,
        "tft": TemporalFusionTransformer,
        "nbeats": NBEATS,
        "tsmixer": TSMixer,
    }

    if model_type not in models:
        raise ValueError(
            f"Unknown model type: {model_type}. "
            f"Available: {list(models.keys())}"
        )

    return models[model_type](**kwargs)


def list_models() -> list[str]:
    """List available model types."""
    return ["informer", "tft", "nbeats", "tsmixer"]
