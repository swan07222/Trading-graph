from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from config.settings import CONFIG
from models.trainer import Trainer


def test_fetch_raw_data_rejects_invalid_ohlc():
    trainer = Trainer()

    bad_df = pd.DataFrame(
        {
            "open": [10.0, 10.0, 10.0],
            "high": [9.0, 9.0, 9.0],
            "low": [11.0, 11.0, 11.0],
            "close": [10.0, 10.0, 10.0],
            "volume": [1000, 1200, 900],
        }
    )

    class _Fetcher:
        def get_history(self, code, **kwargs):  # noqa: ARG002
            return bad_df

    trainer.fetcher = _Fetcher()

    out = trainer._fetch_raw_data(
        stocks=["000001"],
        interval="1m",
        bars=240,
        verbose=False,
    )

    assert out == {}
    summary = trainer._last_data_quality_summary
    assert int(summary["symbols_checked"]) == 1
    assert int(summary["symbols_rejected"]) == 1
    assert "invalid_ohlc_relations" in set(summary["top_reject_reasons"])


def test_quality_gate_blocks_tail_stress_failure_only():
    trainer = Trainer()
    test_metrics = {
        "accuracy": 0.72,
        "trading": {
            "profit_factor": 1.25,
            "max_drawdown": 0.12,
            "trades": 24,
            "sharpe_ratio": 1.30,
            "excess_return": 8.0,
            "win_rate": 0.58,
            "trade_coverage": 0.25,
            "avg_trade_confidence": 0.74,
        },
        "stress_tests": {
            "tail_guard_passed": False,
            "cost_resilience_passed": True,
        },
    }

    out = trainer._build_quality_gate(
        test_metrics=test_metrics,
        walk_forward={"enabled": False},
        overfit_report={"detected": False},
        drift_guard={"action": "deploy_ok"},
        data_quality={"symbols_checked": 10, "valid_symbol_ratio": 0.9},
        incremental_guard={"blocked": False},
    )

    assert out["passed"] is False
    assert out["checks"]["tail_stress"] is False
    assert "tail_stress_failure" in set(out["failed_reasons"])


def test_quality_gate_blocks_insufficient_trade_count():
    trainer = Trainer()
    test_metrics = {
        "accuracy": 0.86,
        "trading": {
            "profit_factor": 1.45,
            "max_drawdown": 0.08,
            "trades": 1,
            "sharpe_ratio": 1.70,
            "excess_return": 12.0,
            "win_rate": 0.66,
            "trade_coverage": 0.20,
            "avg_trade_confidence": 0.78,
        },
        "stress_tests": {
            "tail_guard_passed": True,
            "cost_resilience_passed": True,
        },
    }

    out = trainer._build_quality_gate(
        test_metrics=test_metrics,
        walk_forward={"enabled": False},
        overfit_report={"detected": False},
        drift_guard={"action": "deploy_ok"},
        data_quality={"symbols_checked": 10, "valid_symbol_ratio": 0.9},
        incremental_guard={"blocked": False},
    )

    assert out["passed"] is False
    assert out["checks"]["trade_count"] is False
    assert "insufficient_trade_count" in set(out["failed_reasons"])


def test_train_blocks_incremental_mode_on_high_regime(monkeypatch, tmp_path: Path):
    import models.trainer as trainer_mod

    trainer = trainer_mod.Trainer()
    seq_len = int(CONFIG.SEQUENCE_LENGTH)
    n_feat = 3

    sample_df = pd.DataFrame({"close": [1.0, 2.0, 3.0]})
    monkeypatch.setattr(
        trainer,
        "_fetch_raw_data",
        lambda *_a, **_k: {"AAA": sample_df},
    )
    monkeypatch.setattr(
        trainer.feature_engine,
        "get_feature_columns",
        lambda: [f"f{i}" for i in range(n_feat)],
    )
    monkeypatch.setattr(
        trainer,
        "_split_and_fit_scaler",
        lambda *_a, **_k: (
            {
                "AAA": {
                    "train": pd.DataFrame(),
                    "val": pd.DataFrame(),
                    "test": pd.DataFrame(),
                }
            },
            True,
        ),
    )
    monkeypatch.setattr(
        trainer,
        "_create_sequences_from_splits",
        lambda *_a, **_k: {
            "train": {"X": [1], "y": [1], "r": [1]},
            "val": {"X": [1], "y": [1], "r": [1]},
            "test": {"X": [1], "y": [1], "r": [1]},
        },
    )

    arrays = [
        (
            np.random.randn(24, seq_len, n_feat).astype(np.float32),
            np.random.randint(0, 3, 24).astype(np.int64),
            np.zeros(24, dtype=np.float32),
        ),
        (
            np.random.randn(8, seq_len, n_feat).astype(np.float32),
            np.random.randint(0, 3, 8).astype(np.int64),
            np.zeros(8, dtype=np.float32),
        ),
        (
            np.random.randn(6, seq_len, n_feat).astype(np.float32),
            np.random.randint(0, 3, 6).astype(np.int64),
            np.zeros(6, dtype=np.float32),
        ),
    ]
    monkeypatch.setattr(trainer, "_combine_arrays", lambda *_a, **_k: arrays.pop(0))
    monkeypatch.setattr(
        trainer,
        "_summarize_regime_shift",
        lambda *_a, **_k: {
            "level": "high",
            "score": 0.92,
            "volatility_ratio": 1.8,
            "confidence_boost": 0.1,
        },
    )
    monkeypatch.setattr(trainer, "_train_forecaster", lambda *_a, **_k: False)
    monkeypatch.setattr(
        trainer,
        "_evaluate",
        lambda *_a, **_k: {
            "accuracy": 0.62,
            "trading": {
                "profit_factor": 1.20,
                "max_drawdown": 0.10,
                "trades": 14,
                "sharpe_ratio": 1.00,
                "excess_return": 5.0,
                "win_rate": 0.55,
                "trade_coverage": 0.20,
                "avg_trade_confidence": 0.72,
            },
            "stress_tests": {
                "tail_guard_passed": True,
                "cost_resilience_passed": True,
            },
        },
    )
    monkeypatch.setattr(
        trainer,
        "_assess_overfitting",
        lambda *_a, **_k: {"detected": False},
    )
    monkeypatch.setattr(
        trainer,
        "_walk_forward_validate",
        lambda *_a, **_k: {"enabled": False},
    )
    monkeypatch.setattr(
        trainer,
        "_run_drift_guard",
        lambda *_a, **_k: {"action": "deploy_ok", "baseline_updated": False},
    )
    monkeypatch.setattr(trainer.processor, "save_scaler", lambda *_a, **_k: None)

    model_dir = tmp_path / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(trainer_mod.CONFIG, "MODEL_DIR", model_dir, raising=False)
    (model_dir / "ensemble_1m_30.pt").write_bytes(b"stub")

    class _DummyEnsemble:
        load_calls = 0

        def __init__(self, input_size, model_names=None):  # noqa: ARG002
            self.input_size = int(input_size)
            self.models = {"dummy": object()}
            self.interval = "1m"
            self.prediction_horizon = 30
            self.trained_stock_codes = []

        def load(self, path):  # noqa: ARG002
            _DummyEnsemble.load_calls += 1
            return True

        def train(
            self,
            X_train,  # noqa: ARG002
            y_train,  # noqa: ARG002
            X_val,  # noqa: ARG002
            y_val,  # noqa: ARG002
            epochs=None,  # noqa: ARG002
            batch_size=None,  # noqa: ARG002
            callback=None,  # noqa: ARG002
            stop_flag=None,  # noqa: ARG002
            learning_rate=None,  # noqa: ARG002
        ):
            return {"dummy": {"val_acc": [0.63]}}

        def save(self, path):  # noqa: ARG002
            return None

    monkeypatch.setattr(trainer_mod, "EnsembleModel", _DummyEnsemble)

    out = trainer.train(
        stock_codes=["AAA"],
        epochs=1,
        batch_size=4,
        incremental=True,
        save_model=False,
        interval="1m",
        prediction_horizon=30,
    )

    assert out["status"] == "complete"
    assert out["incremental_guard"]["requested"] is True
    assert out["incremental_guard"]["blocked"] is True
    assert out["incremental_guard"]["effective"] is False
    assert _DummyEnsemble.load_calls == 0


def test_rebalance_train_samples_downsamples_noise_and_upsamples_tails():
    trainer = Trainer()
    n = 400
    seq_len = int(CONFIG.SEQUENCE_LENGTH)
    n_feat = 4

    X = np.random.randn(n, seq_len, n_feat).astype(np.float32)
    y = np.random.randint(0, 3, n).astype(np.int64)
    low_noise = np.random.normal(0.0, 0.02, 360)
    tails = np.random.normal(0.0, 0.90, 40)
    r = np.concatenate([low_noise, tails]).astype(np.float32)

    X_out, y_out, r_out, report = trainer._rebalance_train_samples(X, y, r)

    assert report["enabled"] is True
    assert int(report["low_signal_downsampled"]) > 0
    assert int(report["tail_upsampled"]) > 0
    assert len(X_out) == int(report["output_samples"])
    assert len(y_out) == len(X_out)
    assert r_out is not None
    assert len(r_out) == len(X_out)
