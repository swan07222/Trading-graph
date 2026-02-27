from __future__ import annotations

from models.predictor import Prediction, Predictor, Signal, TradingLevels


def _make_predictor(capital: float = 100000.0) -> Predictor:
    p = Predictor.__new__(Predictor)
    p.capital = float(capital)
    return p


def test_position_sizing_allows_positive_expected_edge() -> None:
    p = _make_predictor()
    pred = Prediction(
        stock_code="600519",
        signal=Signal.STRONG_BUY,
        confidence=0.82,
        signal_strength=0.90,
        model_agreement=0.80,
        prob_up=0.78,
        prob_down=0.10,
        current_price=100.0,
        levels=TradingLevels(
            entry=100.0,
            stop_loss=98.0,
            target_1=103.0,
            target_2=106.0,
        ),
    )

    pos = p._calculate_position(pred)
    assert pos.shares > 0
    assert pos.expected_edge_pct > 0
    assert pos.risk_reward_ratio > 1.1


def test_position_sizing_blocks_negative_expected_edge() -> None:
    p = _make_predictor()
    pred = Prediction(
        stock_code="600519",
        signal=Signal.BUY,
        confidence=0.61,
        signal_strength=0.60,
        model_agreement=0.55,
        prob_up=0.45,
        prob_down=0.40,
        current_price=100.0,
        levels=TradingLevels(
            entry=100.0,
            stop_loss=98.0,
            target_1=101.0,
            target_2=102.0,
        ),
    )

    pos = p._calculate_position(pred)
    assert pos.shares == 0
    assert pos.expected_edge_pct <= 0


def test_position_sizing_skips_trade_if_min_lot_exceeds_risk_budget() -> None:
    p = _make_predictor(capital=10000.0)
    pred = Prediction(
        stock_code="600519",
        signal=Signal.BUY,
        confidence=0.82,
        signal_strength=0.84,
        model_agreement=0.80,
        prob_up=0.90,
        prob_down=0.05,
        current_price=100.0,
        levels=TradingLevels(
            entry=100.0,
            stop_loss=92.0,
            target_1=110.0,
            target_2=120.0,
        ),
        tail_risk_score=0.20,
        uncertainty_score=0.20,
    )

    pos = p._calculate_position(pred)
    assert pos.shares == 0
    assert pos.expected_edge_pct > 0


def test_position_sizing_blocks_high_tail_risk_setup() -> None:
    p = _make_predictor()
    pred = Prediction(
        stock_code="600519",
        signal=Signal.BUY,
        confidence=0.84,
        signal_strength=0.88,
        model_agreement=0.82,
        prob_up=0.80,
        prob_down=0.10,
        current_price=100.0,
        levels=TradingLevels(
            entry=100.0,
            stop_loss=98.0,
            target_1=104.0,
            target_2=108.0,
        ),
        tail_risk_score=0.82,
        uncertainty_score=0.40,
    )

    pos = p._calculate_position(pred)
    assert pos.shares == 0
