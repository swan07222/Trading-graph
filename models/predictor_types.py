from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from core.types import Signal


@dataclass
class TradingLevels:
    """Trading price levels."""

    entry: float = 0.0
    stop_loss: float = 0.0
    stop_loss_pct: float = 0.0
    target_1: float = 0.0
    target_1_pct: float = 0.0
    target_2: float = 0.0
    target_2_pct: float = 0.0
    target_3: float = 0.0
    target_3_pct: float = 0.0


@dataclass
class PositionSize:
    """Position sizing information."""

    shares: int = 0
    value: float = 0.0
    risk_amount: float = 0.0
    risk_pct: float = 0.0
    expected_edge_pct: float = 0.0
    risk_reward_ratio: float = 0.0


@dataclass
class Prediction:
    """Complete prediction result."""

    stock_code: str
    stock_name: str = ""
    timestamp: datetime = field(default_factory=datetime.now)

    signal: Signal = Signal.HOLD
    signal_strength: float = 0.0
    confidence: float = 0.0
    raw_confidence: float = 0.0

    prob_up: float = 0.33
    prob_neutral: float = 0.34
    prob_down: float = 0.33

    current_price: float = 0.0
    price_history: list[float] = field(default_factory=list)
    predicted_prices: list[float] = field(default_factory=list)

    rsi: float = 50.0
    macd_signal: str = "NEUTRAL"
    trend: str = "NEUTRAL"

    levels: TradingLevels = field(default_factory=TradingLevels)
    position: PositionSize = field(default_factory=PositionSize)

    reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    interval: str = "1m"
    horizon: int = 30

    # Extra fields for UI/signal generator.
    model_agreement: float = 1.0
    entropy: float = 0.0
    model_margin: float = 0.0
    brier_score: float = 0.0
    uncertainty_score: float = 0.5
    tail_risk_score: float = 0.5

    # ATR from features (used internally for levels).
    atr_pct_value: float = 0.02

    # Forecast uncertainty bands (same length as predicted_prices).
    predicted_prices_low: list[float] = field(default_factory=list)
    predicted_prices_high: list[float] = field(default_factory=list)

    # News-aware context (used by UI/details and forecast tilt).
    news_sentiment: float = 0.0
    news_confidence: float = 0.0
    news_count: int = 0
