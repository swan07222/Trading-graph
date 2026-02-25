from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from config.settings import CONFIG
from models.trainer import Trainer
from scripts.profit_quality_benchmark import (
    _make_synthetic_dataset,
    _SyntheticEnsemble,
    main as benchmark_main,
)


def _build_trainer() -> Trainer:
    trainer = Trainer.__new__(Trainer)
    trainer.ensemble = _SyntheticEnsemble()
    trainer.prediction_horizon = int(max(1, CONFIG.PREDICTION_HORIZON))
    trainer.interval = "1m"
    return trainer


def _split_dataset(samples: int = 1200, seed: int = 11):
    X, y, r = _make_synthetic_dataset(
        samples=samples,
        seed=seed,
        seq_len=int(CONFIG.SEQUENCE_LENGTH),
        features=6,
    )
    a = int(samples * 0.60)
    b = int(samples * 0.80)
    return (
        X[a:b],
        y[a:b],
        r[a:b],
        X[b:],
        y[b:],
        r[b:],
    )


def test_build_confidence_calibration_returns_monotonic_map() -> None:
    trainer = _build_trainer()
    X_val, y_val, _r_val, _X_test, _y_test, _r_test = _split_dataset()

    report = trainer._build_confidence_calibration(X_val, y_val)

    assert report["enabled"] is True
    assert int(report["sample_count"]) >= 120
    x_pts = np.asarray(report["x_points"], dtype=np.float64)
    y_pts = np.asarray(report["y_points"], dtype=np.float64)
    assert len(x_pts) >= 2
    assert len(y_pts) == len(x_pts)
    assert np.all(np.diff(x_pts) >= -1e-8)
    assert np.all(np.diff(y_pts) >= -1e-8)


def test_walk_forward_reports_selection_metrics_with_calibration() -> None:
    trainer = _build_trainer()
    X_val, y_val, r_val, X_test, y_test, r_test = _split_dataset(samples=1500, seed=23)
    calibration = trainer._build_confidence_calibration(X_val, y_val)

    out = trainer._walk_forward_validate(
        X_val,
        y_val,
        r_val,
        X_test,
        y_test,
        r_test,
        regime_profile={},
        calibration_map=calibration,
    )

    assert out["enabled"] is True
    assert "selection_score" in out
    assert "downside_stability" in out
    assert "mean_selection_score" in out
    assert "std_selection_score" in out
    assert "recency_weighted_risk_adjusted_score" in out
    assert 0.0 <= float(out["selection_score"]) <= 1.0
    assert 0.0 <= float(out["downside_stability"]) <= 1.0
    assert 0.0 <= float(out["mean_selection_score"]) <= 1.0
    assert float(out["std_selection_score"]) >= 0.0


def test_profit_quality_benchmark_script_smoke(tmp_path: Path) -> None:
    out_path = tmp_path / "profit_quality_benchmark.json"
    rc = benchmark_main(
        [
            "--samples",
            "1000",
            "--seed",
            "7",
            "--output",
            str(out_path),
        ]
    )

    assert rc == 0
    assert out_path.exists() is True

    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert "before" in payload
    assert "after" in payload
    assert "deltas" in payload
    assert "calibration_report" in payload
