from __future__ import annotations

import threading

import numpy as np
import torch
import torch.nn as nn

from config.settings import CONFIG
from models.ensemble import EnsembleModel


class _TinyLogitNet(nn.Module):
    def __init__(self, input_size: int, num_classes: int) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=0.2)
        self.proj = nn.Linear(input_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Consume the latest step as a compact per-sample representation.
        return self.proj(self.dropout(x[:, -1, :]))


def test_predict_batch_mc_dropout_uses_grad_safe_numpy(monkeypatch) -> None:
    input_size = 4
    num_classes = int(getattr(CONFIG.model, "num_classes", 3))

    ensemble = EnsembleModel.__new__(EnsembleModel)
    ensemble.input_size = int(input_size)
    ensemble.device = "cpu"
    ensemble._lock = threading.RLock()
    ensemble.temperature = 1.0
    ensemble.models = {"tiny": _TinyLogitNet(input_size=input_size, num_classes=num_classes)}
    ensemble.weights = {"tiny": 1.0}

    monkeypatch.setattr(
        CONFIG.model,
        "uncertainty_quantification_enabled",
        True,
        raising=False,
    )
    monkeypatch.setattr(
        CONFIG.model,
        "monte_carlo_dropout_samples",
        2,
        raising=False,
    )

    X = np.random.randn(5, 6, input_size).astype(np.float32)
    preds = ensemble.predict_batch(
        X,
        batch_size=2,
        include_quality_report=False,
    )

    assert len(preds) == 5
    assert all(np.isfinite(float(p.confidence)) for p in preds)
    assert all(np.isfinite(float(p.raw_confidence)) for p in preds)
    assert all(np.isfinite(float(p.uncertainty)) for p in preds)
