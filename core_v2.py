"""
Trading Graph v2.0 - Streamlined Core Module

This module provides simplified, high-quality core functionality.
Focus: Correctness, performance, and prediction accuracy.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any


class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"


class Signal(Enum):
    BUY = "BUY"
    HOLD = "HOLD"
    SELL = "SELL"


@dataclass
class Bar:
    """OHLCV bar data."""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    amount: float = 0.0
    
    def __post_init__(self):
        self.amount = self.amount or (self.close * self.volume)


@dataclass 
class Quote:
    """Real-time quote."""
    code: str
    price: float
    timestamp: datetime
    name: str = ""


@dataclass
class Order:
    """Trading order."""
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: int
    price: float = 0.0
    id: str = ""
    status: str = "PENDING"
    
    def __post_init__(self):
        if not self.id:
            self.id = f"ORD-{datetime.now().strftime('%Y%m%d%H%M%S')}"


@dataclass
class Position:
    """Stock position."""
    symbol: str
    quantity: int
    avg_cost: float
    current_price: float = 0.0
    
    @property
    def market_value(self) -> float:
        return self.quantity * self.current_price
    
    @property
    def pnl(self) -> float:
        return (self.current_price - self.avg_cost) * self.quantity
    
    @property
    def pnl_pct(self) -> float:
        if self.avg_cost <= 0:
            return 0.0
        return (self.current_price / self.avg_cost - 1) * 100


@dataclass
class Prediction:
    """Model prediction output."""
    stock_code: str
    stock_name: str
    signal: Signal
    confidence: float  # 0.0 to 1.0
    current_price: float
    target_price: float = 0.0
    stop_loss: float = 0.0
    regime: str = "UNKNOWN"  # Market regime
    
    def __post_init__(self):
        self.confidence = max(0.0, min(1.0, self.confidence))


# Type aliases
Bars = list[Bar]
Quotes = dict[str, Quote]
Positions = dict[str, Position]
