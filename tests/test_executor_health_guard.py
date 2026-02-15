import queue
import threading
from types import SimpleNamespace

from config.settings import CONFIG, TradingMode
from core.types import AutoTradeMode, Fill, Order, OrderSide, TradeSignal
from trading.executor import ExecutionEngine
from trading.health import HealthStatus


def test_submit_blocks_when_unhealthy():
    eng = ExecutionEngine.__new__(ExecutionEngine)
    eng._running = True
    eng.mode = TradingMode.SIMULATION
    eng._health_monitor = SimpleNamespace(
        get_health=lambda: SimpleNamespace(status=HealthStatus.UNHEALTHY)
    )
    eng._kill_switch = SimpleNamespace(can_trade=True)
    eng.risk_manager = SimpleNamespace(check_order=lambda *a, **k: (True, ""))
    eng._queue = queue.Queue()
    eng._alert_manager = SimpleNamespace(risk_alert=lambda *a, **k: None)
    eng._reject_signal = lambda sig, reason: setattr(sig, "_rejected_reason", reason)

    sig = TradeSignal(symbol="600519", side=OrderSide.BUY, quantity=100, price=10.0)
    ok = eng.submit(sig)
    assert ok is False
    assert "unhealthy" in getattr(sig, "_rejected_reason", "").lower()


def test_degraded_auto_pause_switches_to_manual():
    class DummyAutoTrader:
        def __init__(self):
            self._mode = AutoTradeMode.AUTO
            self.set_calls = []

        def get_mode(self):
            return self._mode

        def set_mode(self, mode):
            self._mode = mode
            self.set_calls.append(mode)

    eng = ExecutionEngine.__new__(ExecutionEngine)
    eng.auto_trader = DummyAutoTrader()
    eng._alert_manager = SimpleNamespace(risk_alert=lambda *a, **k: None)

    health = SimpleNamespace(status=HealthStatus.DEGRADED)
    eng._on_health_degraded(health)

    assert eng.auto_trader.get_mode() == AutoTradeMode.MANUAL
    assert eng.auto_trader.set_calls


def test_submit_blocks_on_best_exec_quote_deviation():
    import trading.executor as exec_mod

    eng = ExecutionEngine.__new__(ExecutionEngine)
    eng._running = True
    eng.mode = TradingMode.SIMULATION
    eng._health_monitor = SimpleNamespace(
        get_health=lambda: SimpleNamespace(status=HealthStatus.HEALTHY)
    )
    eng._kill_switch = SimpleNamespace(can_trade=True)
    eng.risk_manager = SimpleNamespace(check_order=lambda *a, **k: (True, ""))
    eng._queue = queue.Queue()
    eng._alert_manager = SimpleNamespace(risk_alert=lambda *a, **k: None)
    eng._reject_signal = lambda sig, reason: setattr(sig, "_rejected_reason", reason)
    eng._require_fresh_quote = lambda symbol, max_age_seconds=15.0: (True, "OK", 12.0)

    old = getattr(CONFIG.risk, "max_quote_deviation_bps", 80.0)
    old_is_open = CONFIG.is_market_open
    old_access = exec_mod.get_access_control
    old_audit = exec_mod.get_audit_log
    CONFIG.is_market_open = lambda: True
    exec_mod.get_access_control = lambda: SimpleNamespace(check=lambda *_a, **_k: True)
    exec_mod.get_audit_log = lambda: SimpleNamespace(log_risk_event=lambda *a, **k: None)
    CONFIG.risk.max_quote_deviation_bps = 50.0
    try:
        sig = TradeSignal(symbol="600519", side=OrderSide.BUY, quantity=100, price=10.0)
        ok = eng.submit(sig)
        assert ok is False
        assert "deviation" in getattr(sig, "_rejected_reason", "").lower()
    finally:
        CONFIG.risk.max_quote_deviation_bps = old
        CONFIG.is_market_open = old_is_open
        exec_mod.get_access_control = old_access
        exec_mod.get_audit_log = old_audit


def test_execution_snapshot_includes_broker_routing():
    eng = ExecutionEngine.__new__(ExecutionEngine)
    eng._running = True
    eng.mode = TradingMode.SIMULATION
    eng.auto_trader = None
    eng.broker = SimpleNamespace(
        name="router",
        is_connected=True,
        get_health_snapshot=lambda: {"active_venue": "ths", "venues": []},
    )

    snap = eng._build_execution_snapshot()
    assert snap["running"] is True
    assert snap["broker"]["name"] == "router"
    assert snap["broker"]["routing"]["active_venue"] == "ths"


def test_execution_quality_snapshot_tracks_slippage():
    eng = ExecutionEngine.__new__(ExecutionEngine)
    eng._exec_quality_lock = threading.RLock()
    eng._exec_quality = {
        "fills": 0,
        "slippage_bps_sum": 0.0,
        "slippage_bps_abs_sum": 0.0,
        "by_reason": {},
        "last_update": "",
    }

    order = Order(symbol="600519", side=OrderSide.BUY, quantity=100, price=10.0)
    order.tags["arrival_price"] = 10.0
    fill = Fill(order_id=order.id, symbol=order.symbol, side=OrderSide.BUY, quantity=100, price=10.1)

    eng._record_execution_quality(order, fill)
    snap = eng._get_execution_quality_snapshot()

    assert snap["fills"] == 1
    assert snap["avg_signed_slippage_bps"] > 0
    assert snap["avg_abs_slippage_bps"] > 0
    assert snap["by_reason"].get("entry", 0) == 1


def test_synthetic_exit_plan_triggers_and_clears():
    eng = ExecutionEngine.__new__(ExecutionEngine)
    eng._synthetic_exit_lock = threading.RLock()
    eng._synthetic_exits = {}
    eng._get_quote_snapshot = lambda symbol: (12.0, None, "test")

    calls = []

    def _fake_submit(plan, trigger_price, reason):
        calls.append((plan["symbol"], trigger_price, reason))
        return True

    eng._submit_synthetic_exit = _fake_submit

    order = Order(
        symbol="600519",
        side=OrderSide.BUY,
        quantity=100,
        price=10.0,
        stop_loss=9.2,
        take_profit=11.0,
    )
    fill = Fill(order_id=order.id, symbol=order.symbol, side=OrderSide.BUY, quantity=100, price=10.0)

    eng._maybe_register_synthetic_exit(order, fill)
    assert order.id in eng._synthetic_exits

    eng._evaluate_synthetic_exits()
    assert calls, "synthetic exit should submit when TP is crossed"
    assert order.id not in eng._synthetic_exits


def test_synthetic_exit_state_roundtrip(tmp_path):
    eng = ExecutionEngine.__new__(ExecutionEngine)
    eng._synthetic_exit_lock = threading.RLock()
    eng._synthetic_exits = {
        "ORD_X": {
            "plan_id": "ORD_X",
            "source_order_id": "ORD_X",
            "symbol": "600519",
            "side": "long",
            "open_qty": 100,
            "stop_loss": 9.5,
            "take_profit": 11.2,
            "trailing_stop_pct": 1.5,
            "highest_price": 10.2,
            "armed_at": "2026-01-01T09:30:00",
        }
    }
    eng._synthetic_exit_state_path = tmp_path / "synthetic_exits_state.json"
    eng._last_synthetic_persist_ts = 0.0
    eng._synthetic_persist_min_interval_s = 0.0

    eng._persist_synthetic_exits(force=True)
    eng._synthetic_exits = {}
    eng._restore_synthetic_exits()

    assert "ORD_X" in eng._synthetic_exits
    restored = eng._synthetic_exits["ORD_X"]
    assert restored["symbol"] == "600519"
    assert int(restored["open_qty"]) == 100
