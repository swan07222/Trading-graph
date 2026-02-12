import queue
from types import SimpleNamespace

from config.settings import TradingMode
from core.types import TradeSignal, OrderSide, AutoTradeMode
from trading.executor import ExecutionEngine
from trading.health import HealthStatus
from config.settings import CONFIG


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
