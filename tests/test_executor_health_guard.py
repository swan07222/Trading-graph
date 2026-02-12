import queue
from types import SimpleNamespace

from config.settings import TradingMode
from core.types import TradeSignal, OrderSide, AutoTradeMode
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
