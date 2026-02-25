import threading

import pytest

try:
    from trading.executor import AutoTrader, ExecutionEngine
    from trading.executor_core_ops import _start_engine_thread

    _EXECUTION_STACK_AVAILABLE = True
except ImportError:
    _EXECUTION_STACK_AVAILABLE = False
    AutoTrader = ExecutionEngine = None  # type: ignore[assignment]
    _start_engine_thread = None  # type: ignore[assignment]

pytestmark = pytest.mark.skipif(
    not _EXECUTION_STACK_AVAILABLE,
    reason="Execution stack modules are removed in analysis-only build.",
)


class _DummyBroker:
    is_connected = False


class _DummyEngine:
    def __init__(self) -> None:
        self._running = False
        self.broker = _DummyBroker()
        self.risk_manager = None


def test_auto_trader_worker_thread_is_non_daemon() -> None:
    trader = AutoTrader(engine=_DummyEngine(), predictor=None, watch_list=[])
    trader._start_loop()
    try:
        assert trader._thread is not None
        assert trader._thread.daemon is False
    finally:
        trader._stop_loop()


def test_execution_engine_join_worker_threads_clears_refs() -> None:
    eng = ExecutionEngine.__new__(ExecutionEngine)
    t = threading.Thread(target=lambda: None, name="dummy-thread")
    eng._exec_thread = t
    eng._fill_sync_thread = t
    eng._status_sync_thread = t
    eng._recon_thread = t
    eng._reconnect_thread = t
    eng._watchdog_thread = t
    eng._checkpoint_thread = t

    eng._join_worker_threads(timeout_seconds=0.1)

    assert eng._exec_thread is None
    assert eng._fill_sync_thread is None
    assert eng._status_sync_thread is None
    assert eng._recon_thread is None
    assert eng._reconnect_thread is None
    assert eng._watchdog_thread is None
    assert eng._checkpoint_thread is None


def test_start_engine_thread_skips_duplicate_live_thread() -> None:
    class DummyEngine:
        def __init__(self) -> None:
            self._running = True
            self._thread_hb_lock = threading.RLock()
            self._thread_heartbeats = {}
            self._kill_switch = type("K", (), {"can_trade": False})()
            self._health_monitor = type(
                "HM",
                (),
                {"report_component_health": staticmethod(lambda *a, **k: None)},
            )()
            self._alert_manager = type(
                "AM",
                (),
                {"risk_alert": staticmethod(lambda *a, **k: None)},
            )()
            self._exec_thread = None

        def _heartbeat(self, name: str) -> None:
            with self._thread_hb_lock:
                self._thread_heartbeats[str(name)] = 1.0

    eng = DummyEngine()
    started = threading.Event()
    release = threading.Event()

    def _target() -> None:
        started.set()
        release.wait(timeout=2.0)

    _start_engine_thread(eng, "_exec_thread", _target, "exec")
    assert isinstance(eng._exec_thread, threading.Thread)
    assert started.wait(timeout=1.0)
    first = eng._exec_thread

    # Duplicate start should be ignored while the first worker is alive.
    _start_engine_thread(eng, "_exec_thread", lambda: None, "exec")
    assert eng._exec_thread is first

    release.set()
    first.join(timeout=2.0)
    assert not first.is_alive()
