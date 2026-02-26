import time
from types import SimpleNamespace

from data.feeds import FeedManager, FeedStatus, PollingFeed, WebSocketFeed


def _is_watchdog_alive(thread_obj) -> bool:
    try:
        return bool(thread_obj is not None and thread_obj.is_alive())
    except Exception:
        return False


def test_feed_manager_watchdog_stop_terminates_thread() -> None:
    fm = FeedManager()
    fm._stop_health_watchdog()

    old_interval = fm._health_poll_interval_s
    try:
        fm._health_poll_interval_s = 3.0
        fm._start_health_watchdog()
        time.sleep(0.05)
        thread = fm._health_thread
        assert _is_watchdog_alive(thread)

        fm._stop_health_watchdog()
        assert not _is_watchdog_alive(thread)
    finally:
        fm._health_poll_interval_s = old_interval
        fm._stop_health_watchdog()


def test_websocket_feed_dns_failures_trigger_cooldown() -> None:
    ws = WebSocketFeed()
    ws._ws_client_installed = True
    ws._running = True

    for _ in range(int(ws._DNS_FAILURE_DISABLE_THRESHOLD)):
        ws._on_error(None, OSError("[Errno 11001] getaddrinfo failed"))

    assert ws.status == FeedStatus.ERROR
    assert ws._running is False
    assert ws.supports_websocket() is False
    assert float(ws._ws_disabled_until_ts) > time.monotonic()


def test_websocket_feed_support_recovers_after_cooldown() -> None:
    ws = WebSocketFeed()
    ws._ws_client_installed = True
    ws._ws_disabled_until_ts = time.monotonic() - 0.1
    assert ws.supports_websocket() is True


def test_websocket_feed_on_close_honors_temp_disable() -> None:
    ws = WebSocketFeed()
    ws._ws_client_installed = True
    ws._running = False
    ws._ws_disabled_until_ts = time.monotonic() + 60.0
    ws.status = FeedStatus.RECONNECTING
    before = int(ws._reconnect_count)

    ws._on_close(None, None, None)

    assert ws.status == FeedStatus.ERROR
    assert int(ws._reconnect_count) == before


def test_websocket_feed_on_error_ignores_late_callbacks_while_disabled() -> None:
    ws = WebSocketFeed()
    ws._ws_client_installed = True
    ws._running = False
    ws._ws_disabled_until_ts = time.monotonic() + 60.0
    ws._consecutive_dns_failures = 0
    ws.status = FeedStatus.DISCONNECTED

    for _ in range(int(ws._DNS_FAILURE_DISABLE_THRESHOLD) + 3):
        ws._on_error(None, OSError("[Errno 11001] getaddrinfo failed"))

    assert ws.status == FeedStatus.DISCONNECTED
    assert int(ws._consecutive_dns_failures) == 0
    assert ws.supports_websocket() is False


def test_websocket_feed_max_reconnect_attempts_enters_cooldown() -> None:
    ws = WebSocketFeed()
    ws._ws_client_installed = True
    ws._running = True
    ws._reconnect_count = int(ws._MAX_RECONNECT_ATTEMPTS)

    ws._on_close(None, None, None)

    assert ws.status == FeedStatus.ERROR
    assert ws._running is False
    assert ws.supports_websocket() is False
    assert float(ws._ws_disabled_until_ts) > time.monotonic()


def test_websocket_feed_can_be_forced_off_with_env(monkeypatch) -> None:
    monkeypatch.setenv("TRADING_DISABLE_WEBSOCKET", "1")
    ws = WebSocketFeed()
    ws._ws_client_installed = True
    assert ws.supports_websocket() is False
    assert ws.connect() is False


def test_websocket_feed_connect_allowed_on_china_direct(monkeypatch) -> None:
    """WebSocket is allowed in China-only mode."""
    class _Env:
        is_china_direct = True

    monkeypatch.setattr("core.network.peek_network_env", lambda: _Env())

    ws = WebSocketFeed()
    ws._ws_client_installed = True

    # In China-only mode, WebSocket is allowed (DNS check will determine actual connection)
    # This test verifies the network check passes
    allowed, reason = ws._network_allows_websocket()
    assert allowed is True


def test_websocket_feed_connect_blocked_when_ws_host_dns_fails(monkeypatch) -> None:
    class _Env:
        is_china_direct = True

    monkeypatch.setattr("core.network.peek_network_env", lambda: _Env())
    monkeypatch.setattr("socket.getaddrinfo", lambda *args, **kwargs: (_ for _ in ()).throw(OSError("dns fail")))

    ws = WebSocketFeed()
    ws._ws_client_installed = True

    assert ws.connect() is False
    assert ws._running is False
    assert ws.supports_websocket() is False


def test_websocket_feed_host_resolves_timeout_returns_unknown_as_true(monkeypatch) -> None:
    ws = WebSocketFeed()
    ws._ws_client_installed = True
    ws._last_dns_probe_host = ""

    monkeypatch.setattr(
        WebSocketFeed,
        "_resolve_host_with_timeout",
        staticmethod(lambda host, timeout_s: None),
    )

    assert ws._host_resolves("push.sina.cn") is True


def test_websocket_feed_host_resolves_uses_cache(monkeypatch) -> None:
    ws = WebSocketFeed()
    ws._ws_client_installed = True
    ws._DNS_PRECHECK_CACHE_SECONDS = 60.0

    calls = {"n": 0}

    def _fake_probe(host: str, timeout_s: float) -> bool:
        calls["n"] += 1
        return False

    monkeypatch.setattr(
        WebSocketFeed,
        "_resolve_host_with_timeout",
        staticmethod(_fake_probe),
    )

    assert ws._host_resolves("push.sina.cn") is False
    assert ws._host_resolves("push.sina.cn") is False
    assert int(calls["n"]) == 1


def test_polling_feed_batch_invalid_quote_falls_back_to_single_fetch(monkeypatch) -> None:
    feed = PollingFeed(interval=1.0)

    class _Fetcher:
        def __init__(self) -> None:
            self.single_calls = 0

        def get_realtime_batch(self, symbols):
            return {str(symbols[0]): SimpleNamespace(price=0.0)}

        def get_realtime(self, symbol):
            self.single_calls += 1
            return SimpleNamespace(code=str(symbol), price=15.2)

    class _SpotCache:
        @staticmethod
        def get_quote(symbol) -> None:  # noqa: ARG004
            return None

    fetcher = _Fetcher()
    monkeypatch.setattr(feed, "_get_fetcher", lambda: fetcher)
    monkeypatch.setattr("data.fetcher.get_spot_cache", lambda: _SpotCache())

    out = feed._fetch_batch_quotes(["600519"])
    assert "600519" in out
    assert float(getattr(out["600519"], "price", 0) or 0) == 15.2
    assert int(fetcher.single_calls) == 1
