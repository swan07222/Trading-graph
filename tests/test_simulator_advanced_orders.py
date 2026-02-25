from __future__ import annotations

import random
from types import SimpleNamespace

import pandas as pd
import pytest

from core.types import Order, OrderSide, OrderStatus, OrderType, Position

try:
    from trading.broker import SimulatorBroker

    _EXECUTION_STACK_AVAILABLE = True
except ImportError:
    _EXECUTION_STACK_AVAILABLE = False
    SimulatorBroker = None  # type: ignore[assignment]

pytestmark = pytest.mark.skipif(
    not _EXECUTION_STACK_AVAILABLE,
    reason="Execution stack modules are removed in analysis-only build.",
)


def test_simulator_quote_falls_back_to_recent_cache(monkeypatch) -> None:
    broker = SimulatorBroker(initial_capital=100000.0)

    class _Fetcher:
        def __init__(self) -> None:
            self.calls = 0

        def get_realtime(self, symbol):  # noqa: ARG002
            self.calls += 1
            if self.calls == 1:
                return SimpleNamespace(price=12.34)
            return None

    fetcher = _Fetcher()
    monkeypatch.setattr(broker, "_get_fetcher", lambda: fetcher)

    px1 = broker.get_quote("600519")
    px2 = broker.get_quote("600519")

    assert float(px1 or 0.0) == 12.34
    assert float(px2 or 0.0) == 12.34


def test_simulator_quote_falls_back_to_history_close(monkeypatch) -> None:
    broker = SimulatorBroker(initial_capital=100000.0)

    class _Fetcher:
        def get_realtime(self, symbol) -> None:  # noqa: ARG002
            return None

        def get_history(
            self,
            symbol,  # noqa: ARG002
            interval="1m",  # noqa: ARG002
            bars=2,  # noqa: ARG002
            use_cache=True,  # noqa: ARG002
            update_db=False,  # noqa: ARG002
            allow_online=False,  # noqa: ARG002
        ):
            idx = pd.date_range("2026-02-16 09:30:00", periods=2, freq="min")
            return pd.DataFrame({"close": [10.5, 10.8]}, index=idx)

    monkeypatch.setattr(broker, "_get_fetcher", lambda: _Fetcher())
    px = broker.get_quote("000001")
    assert float(px or 0.0) == 10.8


def test_stop_order_waits_for_trigger_then_fills(monkeypatch) -> None:
    broker = SimulatorBroker(initial_capital=1_000_000.0)
    assert broker.connect() is True

    monkeypatch.setattr(random, "random", lambda: 0.99)

    order = Order(
        symbol="600519",
        side=OrderSide.BUY,
        order_type=OrderType.STOP,
        quantity=100,
        price=0.0,
        stop_price=10.0,
    )
    order.status = OrderStatus.ACCEPTED
    order.tags = {
        "requested_order_type": "stop",
        "trigger_price": 10.0,
        "max_immediate_fill_ratio": 1.0,
    }

    broker._execute_order(order, market_price=9.8)
    assert order.status == OrderStatus.ACCEPTED
    assert order.filled_qty == 0

    broker._execute_order(order, market_price=10.2)
    assert order.filled_qty == 100
    assert order.status in {OrderStatus.FILLED, OrderStatus.PARTIAL}
    assert bool(order.tags.get("_conditional_triggered", False)) is True


def test_trailing_sell_waits_then_triggers_on_pullback(monkeypatch) -> None:
    broker = SimulatorBroker(initial_capital=1_000_000.0)
    assert broker.connect() is True
    broker._positions["600519"] = Position(
        symbol="600519",
        name="Kweichow Moutai",
        quantity=200,
        available_qty=200,
        avg_cost=10.0,
        current_price=10.0,
    )

    monkeypatch.setattr(random, "random", lambda: 0.99)

    order = Order(
        symbol="600519",
        side=OrderSide.SELL,
        order_type=OrderType.TRAIL_MARKET,
        quantity=100,
        price=0.0,
    )
    order.status = OrderStatus.ACCEPTED
    order.tags = {
        "requested_order_type": "trail_market",
        "trailing_stop_pct": 2.0,
        "max_immediate_fill_ratio": 1.0,
    }

    broker._execute_order(order, market_price=10.0)
    assert order.filled_qty == 0
    assert order.status == OrderStatus.ACCEPTED

    broker._execute_order(order, market_price=10.5)
    assert order.filled_qty == 0
    assert order.status == OrderStatus.ACCEPTED

    broker._execute_order(order, market_price=10.2)
    assert order.filled_qty == 100
    assert bool(order.tags.get("_conditional_triggered", False)) is True
    assert broker._positions["600519"].quantity == 100


def test_trailing_limit_validation_accepts_offset_without_price() -> None:
    broker = SimulatorBroker(initial_capital=1_000_000.0)
    order = Order(
        symbol="600519",
        side=OrderSide.BUY,
        order_type=OrderType.TRAIL_LIMIT,
        quantity=100,
        price=0.0,
    )
    order.tags = {
        "requested_order_type": "trail_limit",
        "trail_limit_offset_pct": 0.5,
        "trailing_stop_pct": 2.0,
    }

    ok, reason = broker._validate_order(order, price=10.0)
    assert ok is True
    assert reason == "OK"


def test_non_marketable_limit_day_order_waits_not_rejects() -> None:
    broker = SimulatorBroker(initial_capital=1_000_000.0)
    assert broker.connect() is True

    order = Order(
        symbol="600519",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        quantity=100,
        price=10.0,
    )
    order.status = OrderStatus.ACCEPTED
    order.tags = {
        "requested_order_type": "limit",
        "time_in_force": "day",
    }

    broker._execute_order(order, market_price=10.3)
    assert order.status == OrderStatus.ACCEPTED
    assert order.filled_qty == 0
    assert "Waiting limit BUY" in str(order.message)


def test_non_marketable_limit_ioc_cancels() -> None:
    broker = SimulatorBroker(initial_capital=1_000_000.0)
    assert broker.connect() is True

    order = Order(
        symbol="600519",
        side=OrderSide.SELL,
        order_type=OrderType.LIMIT,
        quantity=100,
        price=10.0,
    )
    order.status = OrderStatus.ACCEPTED
    order.tags = {
        "requested_order_type": "limit",
        "time_in_force": "ioc",
    }

    broker._execute_order(order, market_price=9.8)
    assert order.status == OrderStatus.CANCELLED
    assert order.filled_qty == 0
