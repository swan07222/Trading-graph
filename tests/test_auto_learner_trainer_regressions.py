from __future__ import annotations

from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from config.settings import CONFIG
from models.auto_learner import ContinuousLearner, LearningProgress
from models.trainer import Trainer
from utils.cancellation import CancelledException


def test_auto_cycle_small_batch_threshold_is_capped():
    learner = ContinuousLearner.__new__(ContinuousLearner)
    learner.progress = LearningProgress()
    learner.progress.training_mode = "auto"

    class _Rotator:
        def __init__(self):
            self.processed_count = 0
            self.pool_size = 2

        def discover_new(
            self, max_stocks, min_market_cap, stop_check, progress_cb  # noqa: ARG002
        ):
            return ["000001", "000002"][: int(max_stocks)]

        def mark_failed(self, code):  # noqa: ARG002
            return None

        def mark_processed(self, codes):  # noqa: ARG002
            return None

    class _Replay:
        def __len__(self):
            return 0

        def sample(self, n):  # noqa: ARG002
            return []

    class _Fetcher:
        def fetch_batch(
            self, codes, interval, lookback, min_bars, stop_check, progress_cb  # noqa: ARG002
        ):
            return list(codes), []

    learner._rotator = _Rotator()
    learner._replay = _Replay()
    learner._fetcher = _Fetcher()
    learner._guardian = SimpleNamespace(backup_current=lambda *_a, **_k: True)
    learner._lr_scheduler = SimpleNamespace(
        get_lr=lambda cycle, incremental: 1e-3  # noqa: ARG005
    )
    learner._metrics = SimpleNamespace(
        trend="stable",
        record=lambda acc: None,  # noqa: ARG005
    )

    finalized = {"count": 0}
    learner._ensure_holdout = lambda *_a, **_k: None
    learner._get_holdout_set = lambda: set()
    learner._should_stop = lambda: False
    learner._update = lambda **_kw: None
    learner._prioritize_codes_by_news = (
        lambda codes, interval, max_probe=16: list(codes)  # noqa: ARG005
    )
    learner._train = lambda *_a, **_k: {
        "status": "complete",
        "best_accuracy": 0.61,
    }
    learner._validate_and_decide = lambda *_a, **_k: True
    learner._finalize_cycle = (
        lambda *_a, **_k: finalized.__setitem__("count", finalized["count"] + 1)
    )

    ok = learner._run_cycle(
        max_stocks=2,
        epochs=1,
        min_market_cap=0.0,
        interval="1m",
        horizon=30,
        lookback=240,
        incremental=True,
        cycle_number=1,
    )

    assert ok is True
    assert finalized["count"] == 1


def test_auto_cycle_rejects_when_trainer_quality_gate_fails():
    learner = ContinuousLearner.__new__(ContinuousLearner)
    learner.progress = LearningProgress()
    learner.progress.training_mode = "auto"

    class _Rotator:
        def __init__(self):
            self.processed_count = 0
            self.pool_size = 2

        def discover_new(
            self, max_stocks, min_market_cap, stop_check, progress_cb  # noqa: ARG002
        ):
            return ["000001", "000002"][: int(max_stocks)]

        def mark_failed(self, code):  # noqa: ARG002
            return None

        def mark_processed(self, codes):  # noqa: ARG002
            return None

    class _Replay:
        def __len__(self):
            return 0

        def sample(self, n):  # noqa: ARG002
            return []

    class _Fetcher:
        def fetch_batch(
            self, codes, interval, lookback, min_bars, stop_check, progress_cb  # noqa: ARG002
        ):
            return list(codes), []

    restore_calls = {"count": 0}

    learner._rotator = _Rotator()
    learner._replay = _Replay()
    learner._fetcher = _Fetcher()
    learner._guardian = SimpleNamespace(
        backup_current=lambda *_a, **_k: True,
        restore_backup=lambda *_a, **_k: restore_calls.__setitem__(
            "count", restore_calls["count"] + 1
        ),
    )
    learner._lr_scheduler = SimpleNamespace(
        get_lr=lambda cycle, incremental: 1e-3  # noqa: ARG005
    )
    learner._metrics = SimpleNamespace(
        trend="stable",
        record=lambda acc: None,  # noqa: ARG005
    )

    finalized = {"accepted": None}

    def _should_not_validate(*_a, **_k):
        raise AssertionError("holdout validation should be skipped")

    learner._ensure_holdout = lambda *_a, **_k: None
    learner._get_holdout_set = lambda: set()
    learner._should_stop = lambda: False
    learner._update = lambda **_kw: None
    learner._prioritize_codes_by_news = (
        lambda codes, interval, max_probe=16: list(codes)  # noqa: ARG005
    )
    learner._train = lambda *_a, **_k: {
        "status": "complete",
        "best_accuracy": 0.61,
        "quality_gate": {
            "passed": False,
            "recommended_action": "shadow_mode_recommended",
            "failed_reasons": ["risk_adjusted_score_below_threshold"],
        },
        "deployment": {
            "deployed": False,
            "reason": "quality_gate_block:shadow_mode_recommended",
        },
    }
    learner._validate_and_decide = _should_not_validate
    learner._finalize_cycle = lambda accepted, *_a, **_k: finalized.__setitem__(
        "accepted", bool(accepted)
    )

    ok = learner._run_cycle(
        max_stocks=2,
        epochs=1,
        min_market_cap=0.0,
        interval="1m",
        horizon=30,
        lookback=240,
        incremental=True,
        cycle_number=1,
    )

    assert ok is False
    assert finalized["accepted"] is False
    assert restore_calls["count"] == 1
    assert learner.progress.model_was_rejected is True


def test_trainer_metadata_uses_split_survivors(monkeypatch):
    import models.trainer as trainer_mod

    trainer = trainer_mod.Trainer()
    seq_len = int(CONFIG.SEQUENCE_LENGTH)
    n_feat = 4
    sample_df = pd.DataFrame({"close": [1.0, 2.0, 3.0]})

    monkeypatch.setattr(
        trainer,
        "_fetch_raw_data",
        lambda *_a, **_k: {"AAA": sample_df, "BBB": sample_df},
    )

    split_data = {
        "AAA": {
            "train": pd.DataFrame(),
            "val": pd.DataFrame(),
            "test": pd.DataFrame(),
        }
    }
    monkeypatch.setattr(
        trainer, "_split_and_fit_scaler", lambda *_a, **_k: (split_data, True)
    )
    monkeypatch.setattr(
        trainer.feature_engine,
        "get_feature_columns",
        lambda: [f"f{i}" for i in range(n_feat)],
    )
    monkeypatch.setattr(trainer.processor, "save_scaler", lambda *_a, **_k: None)
    monkeypatch.setattr(
        trainer,
        "_create_sequences_from_splits",
        lambda *_a, **_k: {
            "train": {"X": [1], "y": [1], "r": [1]},
            "val": {"X": [1], "y": [1], "r": [1]},
            "test": {"X": [], "y": [], "r": []},
        },
    )

    arrays = [
        (
            np.random.randn(12, seq_len, n_feat).astype(np.float32),
            np.random.randint(0, 3, 12).astype(np.int64),
            np.zeros(12, dtype=np.float32),
        ),
        (
            np.random.randn(4, seq_len, n_feat).astype(np.float32),
            np.random.randint(0, 3, 4).astype(np.int64),
            np.zeros(4, dtype=np.float32),
        ),
        (
            np.zeros((0, seq_len, n_feat), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=np.float32),
        ),
    ]

    monkeypatch.setattr(trainer, "_combine_arrays", lambda *_a, **_k: arrays.pop(0))
    monkeypatch.setattr(trainer, "_train_forecaster", lambda *_a, **_k: False)

    class _DummyEnsemble:
        def __init__(self, input_size, model_names=None):  # noqa: ARG002
            self.input_size = int(input_size)
            self.models = {"dummy": object()}
            self.interval = "1m"
            self.prediction_horizon = 30
            self.trained_stock_codes = []

        def load(self, path):  # noqa: ARG002
            return False

        def train(
            self,
            X_train,
            y_train,
            X_val,
            y_val,
            epochs=None,
            batch_size=None,
            callback=None,
            stop_flag=None,
            learning_rate=None,
        ):  # noqa: ARG002
            return {"dummy": {"val_acc": [0.62]}}

        def save(self, path):  # noqa: ARG002
            return None

    monkeypatch.setattr(trainer_mod, "EnsembleModel", _DummyEnsemble)

    result = trainer.train(
        stock_codes=["AAA", "BBB"],
        epochs=1,
        batch_size=4,
        save_model=False,
        interval="1m",
        prediction_horizon=30,
    )

    assert result["status"] == "complete"
    assert result["trained_stock_count"] == 1
    assert result["trained_stock_codes"] == ["AAA"]
    assert trainer.ensemble is not None
    assert trainer.ensemble.trained_stock_codes == ["AAA"]


def test_split_and_fit_scaler_refits_on_feature_mismatch(monkeypatch):
    trainer = Trainer()
    trainer._skip_scaler_fit = True

    fit_calls = []

    class _Processor:
        is_fitted = True
        n_features = 99

        def fit_scaler(self, features, interval=None, horizon=None):
            fit_calls.append((features.shape, interval, horizon))

    trainer.processor = _Processor()

    split_train = pd.DataFrame(
        {
            "f1": [1.0, 2.0, 3.0],
            "f2": [1.5, 2.5, 3.5],
            "label": [0.0, 1.0, 2.0],
        }
    )
    fake_split = {
        "train": split_train,
        "val": split_train.iloc[:0].copy(),
        "test": split_train.iloc[:0].copy(),
    }

    monkeypatch.setattr(trainer, "_split_single_stock", lambda *_a, **_k: fake_split)

    split_data, ok = trainer._split_and_fit_scaler(
        raw_data={"AAA": pd.DataFrame({"close": [1.0, 2.0, 3.0]})},
        feature_cols=["f1", "f2"],
        horizon=30,
        interval="1m",
    )

    assert ok is True
    assert "AAA" in split_data
    assert len(fit_calls) == 1


def test_train_forecaster_raises_when_stop_requested():
    trainer = Trainer()

    class _Stop:
        is_cancelled = True

    with pytest.raises(CancelledException):
        trainer._train_forecaster(
            split_data={},
            feature_cols=[],
            horizon=30,
            interval="1m",
            batch_size=16,
            epochs=2,
            stop_flag=_Stop(),
            save_model=False,
            learning_rate=1e-3,
        )


def test_trainer_prepare_data_forces_1m_and_minimum_lookback(monkeypatch):
    trainer = Trainer()
    seq_len = int(CONFIG.SEQUENCE_LENGTH)
    n_feat = 3

    captured: dict[str, int | str] = {}

    def _fake_fetch(stocks, interval, bars, **kwargs):  # noqa: ARG001
        captured["interval"] = str(interval)
        captured["bars"] = int(bars)
        return {"AAA": pd.DataFrame({"close": [1.0, 2.0, 3.0]})}

    monkeypatch.setattr(trainer, "_fetch_raw_data", _fake_fetch)
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
            "test": {"X": [], "y": [], "r": []},
        },
    )
    monkeypatch.setattr(trainer.processor, "save_scaler", lambda *_a, **_k: None)

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
            np.zeros((0, seq_len, n_feat), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=np.float32),
        ),
    ]
    monkeypatch.setattr(trainer, "_combine_arrays", lambda *_a, **_k: arrays.pop(0))

    trainer.prepare_data(
        stock_codes=["AAA"],
        interval="15m",
        lookback_bars=300,
        verbose=False,
    )

    assert captured["interval"] == "1m"
    assert int(captured["bars"]) >= 10080


def test_trainer_normalize_model_names_uses_all_five_defaults():
    expected = ["lstm", "gru", "tcn", "transformer", "hybrid"]
    assert Trainer._normalize_model_names(None) == expected
    assert Trainer._normalize_model_names(["LSTM", "gru", "lstm", "  "]) == [
        "lstm",
        "gru",
    ]


def test_trainer_incremental_adds_missing_default_models(monkeypatch, tmp_path):
    import models.trainer as trainer_mod

    trainer = trainer_mod.Trainer()
    seq_len = int(CONFIG.SEQUENCE_LENGTH)
    n_feat = 4
    horizon = 30

    monkeypatch.setattr(
        trainer,
        "_fetch_raw_data",
        lambda *_a, **_k: {"AAA": pd.DataFrame({"close": [1.0, 2.0, 3.0]})},
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
            "test": {"X": [], "y": [], "r": []},
        },
    )
    monkeypatch.setattr(trainer.processor, "save_scaler", lambda *_a, **_k: None)
    monkeypatch.setattr(trainer, "_train_forecaster", lambda *_a, **_k: False)

    arrays = [
        (
            np.random.randn(20, seq_len, n_feat).astype(np.float32),
            np.random.randint(0, 3, 20).astype(np.int64),
            np.zeros(20, dtype=np.float32),
        ),
        (
            np.random.randn(6, seq_len, n_feat).astype(np.float32),
            np.random.randint(0, 3, 6).astype(np.int64),
            np.zeros(6, dtype=np.float32),
        ),
        (
            np.zeros((0, seq_len, n_feat), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=np.float32),
        ),
    ]
    monkeypatch.setattr(trainer, "_combine_arrays", lambda *_a, **_k: arrays.pop(0))

    model_dir = tmp_path / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / f"ensemble_1m_{horizon}.pt").write_bytes(b"stub")
    monkeypatch.setattr(trainer_mod.CONFIG, "MODEL_DIR", model_dir, raising=False)

    class _DummyEnsemble:
        def __init__(self, input_size, model_names=None):  # noqa: ARG002
            self.input_size = int(input_size)
            self.models = {"lstm": object()}
            self.weights = {"lstm": 1.0}
            self.interval = "1m"
            self.prediction_horizon = horizon
            self.trained_stock_codes = []
            self.added: list[str] = []
            self.normalized = False

        def load(self, path):  # noqa: ARG002
            return True

        def _init_model(self, name):
            self.models[str(name)] = object()
            self.weights[str(name)] = 1.0
            self.added.append(str(name))

        def _normalize_weights(self):
            self.normalized = True

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
            return {"dummy": {"val_acc": [0.6]}}

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
        prediction_horizon=horizon,
    )

    assert out["status"] == "complete"
    assert trainer.ensemble is not None
    assert set(trainer.ensemble.models.keys()) >= {
        "lstm",
        "gru",
        "tcn",
        "transformer",
        "hybrid",
    }
    assert trainer.ensemble.normalized is True


def _setup_minimal_trainer_for_artifact_gate(
    monkeypatch,
    tmp_path: Path,
    quality_passed: bool,
):
    import models.trainer as trainer_mod

    trainer = trainer_mod.Trainer()
    seq_len = int(CONFIG.SEQUENCE_LENGTH)
    n_feat = 4
    horizon = 30

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
            np.random.randn(20, seq_len, n_feat).astype(np.float32),
            np.random.randint(0, 3, 20).astype(np.int64),
            np.zeros(20, dtype=np.float32),
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
    monkeypatch.setattr(trainer, "_train_forecaster", lambda *_a, **_k: False)
    monkeypatch.setattr(
        trainer,
        "_evaluate",
        lambda *_a, **_k: {
            "accuracy": 0.58,
            "trading": {
                "profit_factor": 1.20,
                "max_drawdown": 0.10,
                "trades": 10,
                "sharpe_ratio": 1.10,
                "excess_return": 4.0,
                "win_rate": 0.55,
                "trade_coverage": 0.20,
                "avg_trade_confidence": 0.72,
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
        lambda *_a, **_k: {
            "action": "deploy_ok",
            "baseline_updated": False,
        },
    )
    monkeypatch.setattr(
        trainer,
        "_build_quality_gate",
        lambda *_a, **_k: {
            "passed": bool(quality_passed),
            "recommended_action": (
                "deploy_ok"
                if quality_passed
                else "shadow_mode_recommended"
            ),
            "failed_reasons": (
                [] if quality_passed else ["risk_adjusted_score_below_threshold"]
            ),
            "risk_adjusted_score": 0.6 if quality_passed else 0.3,
        },
    )

    model_dir = tmp_path / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(trainer_mod.CONFIG, "MODEL_DIR", model_dir, raising=False)

    monkeypatch.setattr(
        trainer.processor,
        "save_scaler",
        lambda path, *_a, **_k: Path(path).write_text(
            "candidate_scaler",
            encoding="utf-8",
        ),
    )

    class _DummyEnsemble:
        def __init__(self, input_size, model_names=None):  # noqa: ARG002
            self.input_size = int(input_size)
            self.models = {"dummy": object()}
            self.interval = "1m"
            self.prediction_horizon = horizon
            self.trained_stock_codes = []

        def load(self, path):  # noqa: ARG002
            return False

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
            return {"dummy": {"val_acc": [0.62]}}

        def save(self, path):
            Path(path).write_text("candidate_ensemble", encoding="utf-8")

    monkeypatch.setattr(trainer_mod, "EnsembleModel", _DummyEnsemble)

    return trainer, model_dir, horizon


def test_trainer_keeps_live_artifacts_when_quality_gate_fails(monkeypatch, tmp_path):
    trainer, model_dir, horizon = _setup_minimal_trainer_for_artifact_gate(
        monkeypatch,
        tmp_path,
        quality_passed=False,
    )

    live_ensemble = model_dir / f"ensemble_1m_{horizon}.pt"
    live_scaler = model_dir / f"scaler_1m_{horizon}.pkl"
    live_ensemble.write_text("live_ensemble", encoding="utf-8")
    live_scaler.write_text("live_scaler", encoding="utf-8")

    out = trainer.train(
        stock_codes=["AAA"],
        epochs=1,
        batch_size=4,
        save_model=True,
        interval="1m",
        prediction_horizon=horizon,
    )

    candidate_ensemble = model_dir / f"ensemble_1m_{horizon}.candidate.pt"
    candidate_scaler = model_dir / f"scaler_1m_{horizon}.candidate.pkl"

    assert out["status"] == "complete"
    assert out["quality_gate"]["passed"] is False
    assert out["deployment"]["deployed"] is False
    assert live_ensemble.read_text(encoding="utf-8") == "live_ensemble"
    assert live_scaler.read_text(encoding="utf-8") == "live_scaler"
    assert candidate_ensemble.read_text(encoding="utf-8") == "candidate_ensemble"
    assert candidate_scaler.read_text(encoding="utf-8") == "candidate_scaler"


def test_trainer_promotes_live_artifacts_when_quality_gate_passes(
    monkeypatch,
    tmp_path,
):
    trainer, model_dir, horizon = _setup_minimal_trainer_for_artifact_gate(
        monkeypatch,
        tmp_path,
        quality_passed=True,
    )

    live_ensemble = model_dir / f"ensemble_1m_{horizon}.pt"
    live_scaler = model_dir / f"scaler_1m_{horizon}.pkl"
    live_ensemble.write_text("live_ensemble", encoding="utf-8")
    live_scaler.write_text("live_scaler", encoding="utf-8")

    out = trainer.train(
        stock_codes=["AAA"],
        epochs=1,
        batch_size=4,
        save_model=True,
        interval="1m",
        prediction_horizon=horizon,
    )

    candidate_ensemble = model_dir / f"ensemble_1m_{horizon}.candidate.pt"
    candidate_scaler = model_dir / f"scaler_1m_{horizon}.candidate.pkl"

    assert out["status"] == "complete"
    assert out["quality_gate"]["passed"] is True
    assert out["deployment"]["deployed"] is True
    assert live_ensemble.read_text(encoding="utf-8") == "candidate_ensemble"
    assert live_scaler.read_text(encoding="utf-8") == "candidate_scaler"
    assert candidate_ensemble.exists() is False
    assert candidate_scaler.exists() is False


def test_drift_guard_skips_weak_initial_baseline(monkeypatch, tmp_path):
    trainer = Trainer()
    monkeypatch.setattr(CONFIG, "DATA_DIR", tmp_path, raising=False)

    weak_metrics = {
        "accuracy": 0.52,
        "trading": {
            "profit_factor": 0.95,
            "max_drawdown": 0.45,
            "trades": 2,
            "sharpe_ratio": -0.2,
            "excess_return": -3.0,
        },
    }
    weak_out = trainer._run_drift_guard(
        interval="1m",
        horizon=30,
        test_metrics=weak_metrics,
        risk_adjusted_score=0.30,
    )
    baseline_path = tmp_path / "training_drift_baseline_1m_30.json"

    assert weak_out["baseline_updated"] is False
    assert weak_out["meets_quality_floor"] is False
    assert weak_out["baseline_update_block_reason"] == "quality_floor_not_met"
    assert baseline_path.exists() is False

    strong_metrics = {
        "accuracy": 0.60,
        "trading": {
            "profit_factor": 1.20,
            "max_drawdown": 0.12,
            "trades": 12,
            "sharpe_ratio": 1.00,
            "excess_return": 6.0,
        },
    }
    strong_out = trainer._run_drift_guard(
        interval="1m",
        horizon=30,
        test_metrics=strong_metrics,
        risk_adjusted_score=0.68,
    )

    assert strong_out["baseline_updated"] is True
    assert strong_out["meets_quality_floor"] is True
    assert baseline_path.exists() is True


def test_auto_learner_forces_1m_and_10080_lookback_floor():
    learner = ContinuousLearner.__new__(ContinuousLearner)

    iv, horizon, lookback, min_bars = learner._resolve_interval(
        "30m", 30, 300
    )
    assert iv == "1m"
    assert horizon == 30
    assert lookback >= 10080
    assert min_bars >= max(int(CONFIG.SEQUENCE_LENGTH) + 20, 80)

    assert learner._compute_lookback_bars("1m") >= 10080


def test_finalize_cycle_updates_rejection_streak():
    learner = ContinuousLearner.__new__(ContinuousLearner)
    learner.progress = LearningProgress()
    learner.progress.training_mode = "auto"

    class _Replay:
        def __init__(self):
            self._count = 0

        def add(self, codes, confidence=0.5):  # noqa: ARG002
            self._count += len(list(codes or []))

        def __len__(self):
            return int(self._count)

    learner._replay = _Replay()
    learner._rotator = SimpleNamespace(mark_processed=lambda *_a, **_k: None)
    learner._guardian = SimpleNamespace(save_as_best=lambda *_a, **_k: None)
    learner._cache_training_sequences = lambda *_a, **_k: None
    learner._update = lambda **_k: None
    learner._log_cycle = lambda *_a, **_k: None

    start = datetime.now()
    learner._finalize_cycle(
        accepted=False,
        ok_codes=["000001"],
        new_batch=["000001"],
        replay_batch=[],
        interval="1m",
        horizon=30,
        lookback=10080,
        acc=0.45,
        cycle_number=1,
        start_time=start,
    )

    assert learner.progress.model_was_rejected is True
    assert learner.progress.consecutive_rejections == 1

    learner._finalize_cycle(
        accepted=True,
        ok_codes=["000001"],
        new_batch=["000001"],
        replay_batch=[],
        interval="1m",
        horizon=30,
        lookback=10080,
        acc=0.62,
        cycle_number=2,
        start_time=start,
    )

    assert learner.progress.model_was_rejected is False
    assert learner.progress.consecutive_rejections == 0
