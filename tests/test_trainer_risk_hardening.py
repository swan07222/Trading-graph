from __future__ import annotations

import builtins
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from config.settings import CONFIG
from models.trainer import Trainer


def test_evaluate_falls_back_when_sklearn_metrics_missing(monkeypatch) -> None:
    trainer = Trainer.__new__(Trainer)
    trainer.ensemble = SimpleNamespace(
        predict_batch=lambda X: [
            SimpleNamespace(
                predicted_class=int(i % 3),
                confidence=0.7,
                agreement=0.8,
                entropy=0.2,
                margin=0.1,
                prob_up=0.6,
                prob_down=0.2,
            )
            for i in range(len(X))
        ]
    )
    trainer._effective_confidence_floor = lambda *_a, **_k: 0.6
    trainer._trade_quality_thresholds = lambda *_a, **_k: {}
    trainer._trade_masks = lambda *_a, **_k: {}
    trainer._simulate_trading = lambda *_a, **_k: {}
    trainer._build_trading_stress_tests = lambda *_a, **_k: {}
    trainer._build_explainability_samples = lambda *_a, **_k: []
    trainer._risk_adjusted_score = lambda *_a, **_k: 0.0

    original_import = builtins.__import__

    def _fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "sklearn.metrics":
            raise ModuleNotFoundError("No module named 'sklearn.metrics'")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _fake_import)

    X = np.zeros((9, int(CONFIG.SEQUENCE_LENGTH), 3), dtype=np.float32)
    y = np.asarray([0, 1, 2, 0, 1, 2, 0, 1, 2], dtype=np.int64)
    r = np.zeros(9, dtype=np.float32)

    out = trainer._evaluate(X, y, r, regime_profile={"level": "normal"})

    assert out["metrics_backend"] == "numpy"
    assert isinstance(out["confusion_matrix"], list)
    assert len(out["confusion_matrix"]) == 3
    assert "up_precision" in out
    assert "up_recall" in out
    assert "up_f1" in out


def test_fetch_raw_data_rejects_invalid_ohlc() -> None:
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


def test_fetch_raw_data_skips_codes_with_pending_reconcile() -> None:
    trainer = Trainer()

    idx = pd.date_range("2026-02-18 09:30:00", periods=300, freq="min")
    good_df = pd.DataFrame(
        {
            "open": [10.0] * len(idx),
            "high": [10.1] * len(idx),
            "low": [9.9] * len(idx),
            "close": [10.0] * len(idx),
            "volume": [100] * len(idx),
            "amount": [1000.0] * len(idx),
        },
        index=idx,
    )

    class _Fetcher:
        def __init__(self) -> None:
            self.reconcile_calls = 0

        @staticmethod
        def clean_code(code):
            return str(code).zfill(6)

        def reconcile_pending_cache_sync(self, codes=None, interval="1m"):  # noqa: ARG002
            self.reconcile_calls += 1
            return {"reconciled": 0, "remaining": 1}

        @staticmethod
        def get_pending_reconcile_codes(interval="1m"):  # noqa: ARG002
            return ["000001"]

        @staticmethod
        def get_history(code, **kwargs):  # noqa: ARG002
            return good_df

    fetcher = _Fetcher()
    trainer.fetcher = fetcher

    out = trainer._fetch_raw_data(
        stocks=["000001", "000002"],
        interval="1m",
        bars=300,
        verbose=False,
    )

    assert "000001" not in out
    assert "000002" in out
    assert fetcher.reconcile_calls >= 1

    summary = trainer._last_data_quality_summary
    assert "pending_reconcile_consistency" in set(summary["top_reject_reasons"])
    guard = dict(summary.get("consistency_guard", {}) or {})
    assert int(guard.get("pending_count", 0)) == 1


def test_fetch_raw_data_uses_cache_fallback_when_online_empty() -> None:
    trainer = Trainer()
    idx = pd.date_range("2026-02-18 09:30:00", periods=300, freq="min")
    good_df = pd.DataFrame(
        {
            "open": [10.0] * len(idx),
            "high": [10.1] * len(idx),
            "low": [9.9] * len(idx),
            "close": [10.0] * len(idx),
            "volume": [100] * len(idx),
            "amount": [1000.0] * len(idx),
        },
        index=idx,
    )

    class _Fetcher:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        @staticmethod
        def clean_code(code):  # noqa: ANN001
            return str(code).zfill(6)

        def get_history(self, code, **kwargs):  # noqa: ANN001
            _ = code
            self.calls.append(dict(kwargs))
            if bool(kwargs.get("use_cache", False)):
                return good_df
            return pd.DataFrame()

    fetcher = _Fetcher()
    trainer.fetcher = fetcher

    out = trainer._fetch_raw_data(
        stocks=["000001"],
        interval="1m",
        bars=300,
        verbose=False,
    )

    assert "000001" in out
    assert any(bool(c.get("use_cache", False)) for c in fetcher.calls)
    summary = dict(trainer._last_data_quality_summary or {})
    assert int(summary.get("cache_fallback_hits", 0)) >= 1


def test_train_empty_raw_data_error_includes_quality_context(monkeypatch) -> None:
    trainer = Trainer()

    monkeypatch.setattr(
        trainer.feature_engine,
        "get_feature_columns",
        lambda: ["f1", "f2"],
    )
    monkeypatch.setattr(
        trainer,
        "_fetch_raw_data",
        lambda *_a, **_k: {},
    )
    trainer._last_data_quality_summary = {
        "symbols_checked": 7,
        "symbols_passed": 0,
        "cache_fallback_hits": 0,
        "top_reject_reasons": ["pending_reconcile_consistency"],
    }

    with pytest.raises(ValueError) as exc_info:
        trainer.train(
            stock_codes=["000001"],
            epochs=1,
            save_model=False,
            interval="1m",
            prediction_horizon=5,
        )

    msg = str(exc_info.value)
    assert "quality_passed=0/7" in msg
    assert "pending_reconcile_consistency" in msg


def test_quality_gate_blocks_tail_stress_failure_only() -> None:
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


def test_quality_gate_blocks_insufficient_trade_count() -> None:
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


def test_train_blocks_incremental_mode_on_high_regime(monkeypatch, tmp_path: Path) -> None:
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

        def __init__(self, input_size, model_names=None) -> None:  # noqa: ARG002
            self.input_size = int(input_size)
            self.models = {"dummy": object()}
            self.interval = "1m"
            self.prediction_horizon = 30
            self.trained_stock_codes = []

        def load(self, path) -> bool:  # noqa: ARG002
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

        def save(self, path) -> None:  # noqa: ARG002
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


def test_rebalance_train_samples_downsamples_noise_and_upsamples_tails() -> None:
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
