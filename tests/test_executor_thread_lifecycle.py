import threading

from trading.executor import AutoTrader, ExecutionEngine


class _DummyBroker:
    is_connected = False


class _DummyEngine:
    def __init__(self) -> None:
        self._running = False
        self.broker = _DummyBroker()
        self.risk_manager = None


def test_auto_trader_worker_thread_is_non_daemon():
    trader = AutoTrader(engine=_DummyEngine(), predictor=None, watch_list=[])
    trader._start_loop()
    try:
        assert trader._thread is not None
        assert trader._thread.daemon is False
    finally:
        trader._stop_loop()


def test_execution_engine_join_worker_threads_clears_refs():
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
