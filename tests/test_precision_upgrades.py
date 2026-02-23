from __future__ import annotations

import pandas as pd

from data.processor import DataProcessor
from models.auto_learner import ContinuousLearner, LearningProgress


def test_profit_aware_labels_use_wider_neutral_band() -> None:
    p = DataProcessor()
    base = pd.DataFrame(
        {
            "close": [100.0, 100.15, 100.30, 100.45, 100.60, 100.75],
        }
    )
    base["open"] = base["close"]
    base["high"] = base["close"] * 1.001
    base["low"] = base["close"] * 0.999
    base["volume"] = 1000

    normal = p.create_labels(
        base,
        horizon=1,
        up_thresh=0.10,
        down_thresh=-0.10,
        profit_aware=False,
    )
    aware = p.create_labels(
        base,
        horizon=1,
        up_thresh=0.10,
        down_thresh=-0.10,
        profit_aware=True,
    )

    # Profit-aware mode should not increase UP labels on tiny moves.
    normal_up = int((normal["label"] == 2.0).sum())
    aware_up = int((aware["label"] == 2.0).sum())
    assert aware_up <= normal_up


def test_threshold_score_prefers_clear_edges() -> None:
    samples = [
        {
            "predicted": 2,
            "actual": 2,
            "confidence": 0.82,
            "agreement": 0.78,
            "entropy": 0.22,
            "prob_up": 0.88,
            "prob_down": 0.05,
            "future_return": 1.6,
        },
        {
            "predicted": 0,
            "actual": 0,
            "confidence": 0.79,
            "agreement": 0.74,
            "entropy": 0.25,
            "prob_up": 0.06,
            "prob_down": 0.83,
            "future_return": -1.3,
        },
        {
            "predicted": 2,
            "actual": 0,
            "confidence": 0.58,
            "agreement": 0.52,
            "entropy": 0.62,
            "prob_up": 0.43,
            "prob_down": 0.33,
            "future_return": -0.9,
        },
    ]

    strict = ContinuousLearner._score_thresholds(
        samples, min_conf=0.70, min_agree=0.65, max_entropy=0.40, min_edge=0.10
    )
    loose = ContinuousLearner._score_thresholds(
        samples, min_conf=0.50, min_agree=0.50, max_entropy=0.80, min_edge=0.02
    )

    assert strict["precision"] >= loose["precision"]


def _make_validation_learner(post_val: dict):
    learner = ContinuousLearner.__new__(ContinuousLearner)
    learner.progress = LearningProgress()
    restore_calls = {"count": 0}
    tuned_calls = {"count": 0}

    class _Guardian:
        def validate_model(self, *_a, **_k):
            return post_val

        def restore_backup(self, *_a, **_k) -> None:
            restore_calls["count"] += 1

    learner._guardian = _Guardian()
    learner._get_holdout_set = lambda: {"000001", "000002", "000004"}
    learner._maybe_tune_precision_thresholds = (
        lambda *_a, **_k: tuned_calls.__setitem__("count", tuned_calls["count"] + 1)
    )
    return learner, restore_calls, tuned_calls


def test_validate_and_decide_rejects_weak_lower_bound_without_baseline() -> None:
    samples = [
        {
            "predicted": 2,
            "actual": 2,
            "confidence": 0.60,
            "agreement": 0.57,
            "entropy": 0.50,
            "prob_up": 0.62,
            "prob_down": 0.20,
            "future_return": 0.9,
        },
        {
            "predicted": 0,
            "actual": 2,
            "confidence": 0.59,
            "agreement": 0.56,
            "entropy": 0.53,
            "prob_up": 0.28,
            "prob_down": 0.58,
            "future_return": 0.8,
        },
        {
            "predicted": 2,
            "actual": 0,
            "confidence": 0.58,
            "agreement": 0.55,
            "entropy": 0.55,
            "prob_up": 0.60,
            "prob_down": 0.24,
            "future_return": -0.7,
        },
        {
            "predicted": 0,
            "actual": 0,
            "confidence": 0.61,
            "agreement": 0.58,
            "entropy": 0.51,
            "prob_up": 0.18,
            "prob_down": 0.62,
            "future_return": -0.6,
        },
        {
            "predicted": 2,
            "actual": 0,
            "confidence": 0.57,
            "agreement": 0.54,
            "entropy": 0.56,
            "prob_up": 0.59,
            "prob_down": 0.26,
            "future_return": -0.5,
        },
        {
            "predicted": 0,
            "actual": 2,
            "confidence": 0.56,
            "agreement": 0.53,
            "entropy": 0.57,
            "prob_up": 0.30,
            "prob_down": 0.56,
            "future_return": 0.6,
        },
    ]
    post_val = {
        "accuracy": 0.50,
        "avg_confidence": 0.59,
        "predictions_made": 6,
        "samples": samples,
    }
    learner, restore_calls, tuned_calls = _make_validation_learner(post_val)

    accepted = learner._validate_and_decide(
        "1m",
        30,
        360,
        pre_val=None,
        new_acc=0.72,
    )

    assert accepted is False
    assert restore_calls["count"] == 1
    assert tuned_calls["count"] == 0


def test_validate_and_decide_relaxes_degradation_in_high_vol_regime() -> None:
    hi_vol_samples = []
    for i in range(80):
        pred = 2 if (i % 2 == 0) else 0
        actual = pred if i < 40 else (0 if pred == 2 else 2)
        hi_vol_samples.append(
            {
                "predicted": pred,
                "actual": actual,
                "confidence": 0.74 + (0.02 if i % 3 == 0 else -0.01),
                "agreement": 0.70 + (0.02 if i % 4 == 0 else -0.01),
                "entropy": 0.32 + (0.03 if i % 5 == 0 else 0.0),
                "prob_up": 0.86 if pred == 2 else 0.10,
                "prob_down": 0.08 if pred == 2 else 0.82,
                "future_return": 2.6 if pred == 2 else -2.3,
            }
        )

    post_val = {
        "accuracy": 0.46,  # would exceed fixed 15% degradation vs pre=0.55
        "avg_confidence": 0.53,
        "predictions_made": 80,
        "samples": hi_vol_samples,
    }
    pre_val = {
        "accuracy": 0.55,
        "avg_confidence": 0.58,
        "predictions_made": 80,
    }
    learner, restore_calls, tuned_calls = _make_validation_learner(post_val)

    accepted = learner._validate_and_decide(
        "1m",
        30,
        360,
        pre_val=pre_val,
        new_acc=0.63,
    )

    assert accepted is True
    assert restore_calls["count"] == 0
    assert tuned_calls["count"] == 1


def test_tune_precision_thresholds_returns_adaptive_metadata() -> None:
    samples = []
    for i in range(120):
        is_good = i < 80
        pred = 2 if i % 2 == 0 else 0
        actual = pred if is_good else (0 if pred == 2 else 2)
        samples.append(
            {
                "predicted": pred,
                "actual": actual,
                "confidence": 0.90 if is_good else 0.58,
                "agreement": 0.84 if is_good else 0.55,
                "entropy": 0.20 if is_good else 0.62,
                "prob_up": 0.92 if pred == 2 else 0.04,
                "prob_down": 0.03 if pred == 2 else 0.90,
                "future_return": 1.8 if is_good else -0.9,
            }
        )

    learner = ContinuousLearner.__new__(ContinuousLearner)
    tuned = learner._tune_precision_thresholds(samples)

    assert tuned is not None
    assert tuned["search_space_size"] > 0
    assert tuned["search_space_size"] > 400
    assert tuned["min_required_trades"] >= ContinuousLearner._MIN_TUNED_TRADES
    assert tuned["regime"] in {"trend", "high_vol", "low_signal", "unknown"}
    assert (
        tuned["min_confidence"] >= 0.75
        or tuned["min_agreement"] >= 0.72
        or tuned["min_edge"] >= 0.20
    )
