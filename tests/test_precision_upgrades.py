from __future__ import annotations

import pandas as pd

from data.processor import DataProcessor
from models.auto_learner import ContinuousLearner


def test_profit_aware_labels_use_wider_neutral_band():
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


def test_threshold_score_prefers_clear_edges():
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
