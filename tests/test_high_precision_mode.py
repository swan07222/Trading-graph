from __future__ import annotations

from config.settings import PrecisionConfig
from models.predictor import Prediction, Predictor, Signal


def _make_predictor(cfg: dict) -> Predictor:
    p = Predictor.__new__(Predictor)
    p._high_precision = cfg
    return p


def test_precision_mode_default_is_disabled() -> None:
    assert PrecisionConfig().enabled is False


def test_high_precision_disabled_keeps_signal() -> None:
    p = _make_predictor({"enabled": 0.0})
    pred = Prediction(
        stock_code="600519",
        signal=Signal.BUY,
        signal_strength=0.8,
        confidence=0.55,
        prob_up=0.56,
        prob_down=0.41,
        model_agreement=0.51,
        entropy=0.70,
    )
    p._apply_high_precision_gate(pred)
    assert pred.signal == Signal.BUY


def test_high_precision_enabled_filters_weak_signal() -> None:
    p = _make_predictor(
        {
            "enabled": 1.0,
            "min_confidence": 0.7,
            "min_agreement": 0.65,
            "max_entropy": 0.45,
            "min_edge": 0.1,
        }
    )
    pred = Prediction(
        stock_code="600519",
        signal=Signal.SELL,
        signal_strength=0.86,
        confidence=0.61,
        prob_up=0.31,
        prob_down=0.39,
        model_agreement=0.52,
        entropy=0.62,
    )
    p._apply_high_precision_gate(pred)
    assert pred.signal == Signal.HOLD
    assert pred.signal_strength <= 0.49
    assert any("High Precision Mode filtered signal" in w for w in pred.warnings)


def test_high_precision_enabled_keeps_strong_signal() -> None:
    p = _make_predictor(
        {
            "enabled": 1.0,
            "min_confidence": 0.7,
            "min_agreement": 0.65,
            "max_entropy": 0.45,
            "min_edge": 0.1,
        }
    )
    pred = Prediction(
        stock_code="600519",
        signal=Signal.STRONG_BUY,
        signal_strength=0.92,
        confidence=0.82,
        prob_up=0.88,
        prob_down=0.05,
        model_agreement=0.80,
        entropy=0.20,
    )
    p._apply_high_precision_gate(pred)
    assert pred.signal == Signal.STRONG_BUY


def test_runtime_quality_gate_filters_weak_signal() -> None:
    p = _make_predictor({"enabled": 0.0})
    pred = Prediction(
        stock_code="600519",
        signal=Signal.BUY,
        signal_strength=0.74,
        confidence=0.62,
        prob_up=0.43,
        prob_down=0.39,
        model_agreement=0.44,
        entropy=0.82,
        trend="SIDEWAYS",
        atr_pct_value=0.046,
    )
    p._apply_runtime_signal_quality_gate(pred)
    assert pred.signal == Signal.HOLD
    assert pred.signal_strength <= 0.49
    assert any("Runtime quality gate filtered signal" in w for w in pred.warnings)


def test_runtime_quality_gate_keeps_strong_aligned_signal() -> None:
    p = _make_predictor({"enabled": 0.0})
    pred = Prediction(
        stock_code="600519",
        signal=Signal.STRONG_BUY,
        signal_strength=0.93,
        confidence=0.90,
        prob_up=0.86,
        prob_down=0.07,
        model_agreement=0.89,
        entropy=0.20,
        trend="UPTREND",
        atr_pct_value=0.021,
    )
    p._apply_runtime_signal_quality_gate(pred)
    assert pred.signal == Signal.STRONG_BUY
