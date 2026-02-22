import queue
import threading
from collections import deque
from types import SimpleNamespace

import pytest

from config.settings import CONFIG, TradingMode
from core.types import AutoTradeMode, Fill, Order, OrderSide, OrderStatus, TradeSignal
from trading.executor import ExecutionEngine
from trading.health import ComponentType, HealthStatus


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


def test_submit_blocks_when_delayed_quote_guard_enabled():
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
    eng._recent_submit_keys = {}
    eng._recent_submissions = {}
    eng._recent_rejections = deque()

    captured = {"block_delayed": None}

    def _fake_require(symbol, max_age_seconds=15.0, block_delayed=False):  # noqa: ARG001
        captured["block_delayed"] = bool(block_delayed)
        if block_delayed:
            return False, "Quote delayed/stale (source=fetcher:last_good)", 0.0
        return True, "OK", 10.0

    eng._require_fresh_quote = _fake_require

    old_guard = bool(getattr(CONFIG.auto_trade, "block_on_stale_realtime", True))
    old_is_open = CONFIG.is_market_open
    old_access = exec_mod.get_access_control
    old_audit = exec_mod.get_audit_log
    CONFIG.auto_trade.block_on_stale_realtime = True
    CONFIG.is_market_open = lambda: True
    exec_mod.get_access_control = lambda: SimpleNamespace(check=lambda *_a, **_k: True)
    exec_mod.get_audit_log = lambda: SimpleNamespace(log_risk_event=lambda *a, **k: None)
    try:
        sig = TradeSignal(
            symbol="600519",
            side=OrderSide.BUY,
            quantity=100,
            price=10.0,
            order_type="market",
        )
        ok = eng.submit(sig)
        assert ok is False
        assert captured["block_delayed"] is True
        assert "delayed" in str(getattr(sig, "_rejected_reason", "")).lower()
    finally:
        CONFIG.auto_trade.block_on_stale_realtime = old_guard
        CONFIG.is_market_open = old_is_open
        exec_mod.get_access_control = old_access
        exec_mod.get_audit_log = old_audit


def test_trigger_model_drift_alarm_forces_live_auto_manual():
    import trading.executor as exec_mod

    class DummyAutoTrader:
        def __init__(self):
            self._mode = AutoTradeMode.AUTO
            self.pause_calls = []
            self.mode_calls = []

        def get_mode(self):
            return self._mode

        def set_mode(self, mode):
            self._mode = mode
            self.mode_calls.append(mode)

        def pause(self, reason, duration_seconds=0):
            self.pause_calls.append((str(reason), int(duration_seconds)))

    eng = ExecutionEngine.__new__(ExecutionEngine)
    eng.mode = TradingMode.LIVE
    eng.auto_trader = DummyAutoTrader()
    eng._alert_manager = SimpleNamespace(risk_alert=lambda *a, **k: None)

    old_enabled = bool(CONFIG.auto_trade.enabled)
    old_pause = int(getattr(CONFIG.auto_trade, "model_drift_pause_seconds", 3600))
    old_disable = bool(getattr(CONFIG.auto_trade, "auto_disable_on_model_drift", True))
    old_audit = exec_mod.get_audit_log
    exec_mod.get_audit_log = lambda: SimpleNamespace(log_risk_event=lambda *a, **k: None)
    CONFIG.auto_trade.enabled = True
    CONFIG.auto_trade.model_drift_pause_seconds = 600
    CONFIG.auto_trade.auto_disable_on_model_drift = True

    with ExecutionEngine._ACTIVE_ENGINES_LOCK:
        ExecutionEngine._ACTIVE_ENGINES.add(eng)

    try:
        handled = ExecutionEngine.trigger_model_drift_alarm("unit_test_drift")
        assert handled >= 1
        assert eng.auto_trader.get_mode() == AutoTradeMode.MANUAL
        assert eng.auto_trader.mode_calls
        assert eng.auto_trader.pause_calls
        assert CONFIG.auto_trade.enabled is False
    finally:
        with ExecutionEngine._ACTIVE_ENGINES_LOCK:
            ExecutionEngine._ACTIVE_ENGINES.discard(eng)
        CONFIG.auto_trade.enabled = old_enabled
        CONFIG.auto_trade.model_drift_pause_seconds = old_pause
        CONFIG.auto_trade.auto_disable_on_model_drift = old_disable
        exec_mod.get_audit_log = old_audit


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
    eng._recent_submit_keys = {}
    eng._recent_submissions = {}
    eng._recent_rejections = deque()

    old = getattr(CONFIG.risk, "max_quote_deviation_bps", 80.0)
    old_is_open = CONFIG.is_market_open
    old_access = exec_mod.get_access_control
    old_audit = exec_mod.get_audit_log
    CONFIG.is_market_open = lambda: True
    exec_mod.get_access_control = lambda: SimpleNamespace(check=lambda *_a, **_k: True)
    exec_mod.get_audit_log = lambda: SimpleNamespace(log_risk_event=lambda *a, **k: None)
    CONFIG.risk.max_quote_deviation_bps = 50.0
    try:
        sig = TradeSignal(
            symbol="600519",
            side=OrderSide.BUY,
            quantity=100,
            price=10.0,
            order_type="market",
        )
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


def test_submit_limit_keeps_limit_price_and_tracks_arrival_quote():
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
    eng._recent_submit_keys = {}
    eng._recent_submissions = {}
    eng._recent_rejections = deque()

    old_is_open = CONFIG.is_market_open
    old_access = exec_mod.get_access_control
    old_audit = exec_mod.get_audit_log
    CONFIG.is_market_open = lambda: True
    exec_mod.get_access_control = lambda: SimpleNamespace(check=lambda *_a, **_k: True)
    exec_mod.get_audit_log = lambda: SimpleNamespace(log_risk_event=lambda *a, **k: None)
    try:
        sig = TradeSignal(
            symbol="600519",
            side=OrderSide.BUY,
            quantity=100,
            price=10.0,
            order_type="limit",
        )
        ok = eng.submit(sig)
        assert ok is True
        assert sig.price == 10.0
        assert float(getattr(sig, "_arrival_price", 0.0)) == 12.0
        queued = eng._queue.get_nowait()
        assert queued is sig
    finally:
        CONFIG.is_market_open = old_is_open
        exec_mod.get_access_control = old_access
        exec_mod.get_audit_log = old_audit


def test_submit_rejects_stop_order_without_trigger():
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
    eng._recent_submit_keys = {}
    eng._recent_submissions = {}
    eng._recent_rejections = deque()

    old_is_open = CONFIG.is_market_open
    old_access = exec_mod.get_access_control
    old_audit = exec_mod.get_audit_log
    CONFIG.is_market_open = lambda: True
    exec_mod.get_access_control = lambda: SimpleNamespace(check=lambda *_a, **_k: True)
    exec_mod.get_audit_log = lambda: SimpleNamespace(log_risk_event=lambda *a, **k: None)
    try:
        sig = TradeSignal(
            symbol="600519",
            side=OrderSide.BUY,
            quantity=100,
            price=0.0,
            order_type="stop",
            trigger_price=0.0,
        )
        ok = eng.submit(sig)
        assert ok is False
        assert "trigger" in str(getattr(sig, "_rejected_reason", "")).lower()
    finally:
        CONFIG.is_market_open = old_is_open
        exec_mod.get_access_control = old_access
        exec_mod.get_audit_log = old_audit


def test_submit_blocks_live_advanced_order_without_native_support():
    import trading.executor as exec_mod

    eng = ExecutionEngine.__new__(ExecutionEngine)
    eng._running = True
    eng.mode = TradingMode.LIVE
    eng.broker = SimpleNamespace()  # no advanced-order capability surface
    eng._health_monitor = SimpleNamespace(
        get_health=lambda: SimpleNamespace(status=HealthStatus.HEALTHY)
    )
    eng._kill_switch = SimpleNamespace(can_trade=True)
    eng.risk_manager = SimpleNamespace(check_order=lambda *a, **k: (True, ""))
    eng._queue = queue.Queue()
    eng._alert_manager = SimpleNamespace(risk_alert=lambda *a, **k: None)
    eng._reject_signal = lambda sig, reason: setattr(sig, "_rejected_reason", reason)
    eng._require_fresh_quote = (
        lambda symbol, max_age_seconds=15.0, block_delayed=False: (True, "OK", 12.0)
    )
    eng._recent_submit_keys = {}
    eng._recent_submissions = {}
    eng._recent_rejections = deque()

    old_is_open = CONFIG.is_market_open
    old_access = exec_mod.get_access_control
    old_audit = exec_mod.get_audit_log
    old_allow_live = bool(
        getattr(CONFIG.security, "allow_live_order_type_emulation", False)
    )
    CONFIG.is_market_open = lambda: True
    CONFIG.security.allow_live_order_type_emulation = False
    exec_mod.get_access_control = lambda: SimpleNamespace(check=lambda *_a, **_k: True)
    exec_mod.get_audit_log = lambda: SimpleNamespace(log_risk_event=lambda *a, **k: None)
    try:
        sig = TradeSignal(
            symbol="600519",
            side=OrderSide.BUY,
            quantity=100,
            price=10.0,
            order_type="trail_market",
        )
        ok = eng.submit(sig)
        assert ok is False
        assert "requires broker-native support" in str(
            getattr(sig, "_rejected_reason", "")
        ).lower()
    finally:
        CONFIG.is_market_open = old_is_open
        CONFIG.security.allow_live_order_type_emulation = old_allow_live
        exec_mod.get_access_control = old_access
        exec_mod.get_audit_log = old_audit


def test_submit_with_retry_escalates_programming_error_when_strict_policy_enabled():
    eng = ExecutionEngine.__new__(ExecutionEngine)
    eng.mode = TradingMode.SIMULATION
    eng._wait_or_stop = lambda _seconds: False

    def _boom_submit(_order):
        raise TypeError("bug-like")

    eng.broker = SimpleNamespace(submit_order=_boom_submit)

    old_strict = bool(getattr(CONFIG.security, "strict_runtime_exception_policy", True))
    CONFIG.security.strict_runtime_exception_policy = True
    try:
        order = Order(symbol="600519", side=OrderSide.BUY, quantity=100, price=10.0)
        with pytest.raises(TypeError):
            eng._submit_with_retry(order, attempts=2)
    finally:
        CONFIG.security.strict_runtime_exception_policy = old_strict


def test_oco_sibling_cancel_on_fill():
    eng = ExecutionEngine.__new__(ExecutionEngine)
    cancelled = []

    def _cancel(order_id):
        cancelled.append(order_id)
        return True

    eng.broker = SimpleNamespace(cancel_order=_cancel)

    filled_order = Order(symbol="600519", side=OrderSide.BUY, quantity=100, price=10.0)
    filled_order.status = OrderStatus.FILLED
    filled_order.tags["oco_group"] = "G1"

    sibling = Order(symbol="600519", side=OrderSide.SELL, quantity=100, price=11.0)
    sibling.status = OrderStatus.ACCEPTED
    sibling.broker_id = "BRK_1"
    sibling.tags["oco_group"] = "G1"

    other = Order(symbol="600519", side=OrderSide.SELL, quantity=100, price=12.0)
    other.status = OrderStatus.ACCEPTED
    other.tags["oco_group"] = "G2"

    updates: list[tuple[str, OrderStatus, str]] = []
    oms = SimpleNamespace(
        get_orders=lambda symbol=None: [filled_order, sibling, other],
        update_order_status=lambda oid, status, message="": updates.append((oid, status, message)),
    )

    fill = Fill(order_id=filled_order.id, symbol="600519", side=OrderSide.BUY, quantity=100, price=10.1)
    eng._cancel_oco_siblings(oms, filled_order, fill)

    assert "BRK_1" in cancelled or sibling.id in cancelled
    assert any(u[0] == sibling.id and u[1] == OrderStatus.CANCELLED for u in updates)
    assert all(u[0] != other.id for u in updates)


def test_market_fingerprint_ignores_price_noise():
    s1 = TradeSignal(
        symbol="600519",
        side=OrderSide.BUY,
        quantity=100,
        price=10.0,
        order_type="market",
    )
    s2 = TradeSignal(
        symbol="600519",
        side=OrderSide.BUY,
        quantity=100,
        price=10.2,
        order_type="market",
    )
    assert ExecutionEngine._make_submit_fingerprint(s1) == ExecutionEngine._make_submit_fingerprint(s2)


def test_live_readiness_check_skips_for_non_live_mode():
    eng = ExecutionEngine.__new__(ExecutionEngine)
    eng.mode = TradingMode.SIMULATION

    ok, msg = eng._evaluate_live_start_readiness()
    assert ok is True
    assert msg == ""


def test_live_readiness_non_strict_allows_start_check(monkeypatch):
    import utils.institutional as institutional

    eng = ExecutionEngine.__new__(ExecutionEngine)
    eng.mode = TradingMode.LIVE

    monkeypatch.setattr(
        institutional,
        "collect_institutional_readiness",
        lambda: {
            "pass": False,
            "failed_required_controls": [
                "strict_live_governance",
                "runtime_lease_enabled",
            ],
        },
    )
    old_strict = bool(getattr(CONFIG.security, "strict_live_governance", False))
    CONFIG.security.strict_live_governance = False
    try:
        ok, msg = eng._evaluate_live_start_readiness()
    finally:
        CONFIG.security.strict_live_governance = old_strict

    assert ok is True
    assert "Institutional readiness failed" in msg


def test_start_blocks_when_live_readiness_fails_strict(monkeypatch):
    import utils.institutional as institutional

    eng = ExecutionEngine.__new__(ExecutionEngine)
    eng._running = False
    eng.mode = TradingMode.LIVE
    health_reports: list[tuple[ComponentType, HealthStatus, str]] = []
    eng._health_monitor = SimpleNamespace(
        report_component_health=lambda comp, status, error="": health_reports.append(
            (comp, status, str(error))
        )
    )
    eng._acquire_runtime_lease = lambda: (_ for _ in ()).throw(
        AssertionError("runtime lease must not be checked when readiness fails")
    )

    monkeypatch.setattr(
        institutional,
        "collect_institutional_readiness",
        lambda: {
            "pass": False,
            "failed_required_controls": ["strict_live_governance"],
        },
    )
    old_strict = bool(getattr(CONFIG.security, "strict_live_governance", False))
    CONFIG.security.strict_live_governance = True
    try:
        ok = eng.start()
    finally:
        CONFIG.security.strict_live_governance = old_strict

    assert ok is False
    assert health_reports
    comp, status, err = health_reports[0]
    assert comp == ComponentType.RISK_MANAGER
    assert status == HealthStatus.UNHEALTHY
    assert "Institutional readiness failed" in err

