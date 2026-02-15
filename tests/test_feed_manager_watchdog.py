import time

from data.feeds import FeedManager, FeedStatus, WebSocketFeed


def _is_watchdog_alive(thread_obj) -> bool:
    try:
        return bool(thread_obj is not None and thread_obj.is_alive())
    except Exception:
        return False


def test_feed_manager_watchdog_stop_terminates_thread():
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


def test_websocket_feed_dns_failures_trigger_cooldown():
    ws = WebSocketFeed()
    ws._ws_client_installed = True
    ws._running = True

    for _ in range(int(ws._DNS_FAILURE_DISABLE_THRESHOLD)):
        ws._on_error(None, OSError("[Errno 11001] getaddrinfo failed"))

    assert ws.status == FeedStatus.ERROR
    assert ws._running is False
    assert ws.supports_websocket() is False
    assert float(ws._ws_disabled_until_ts) > time.monotonic()


def test_websocket_feed_support_recovers_after_cooldown():
    ws = WebSocketFeed()
    ws._ws_client_installed = True
    ws._ws_disabled_until_ts = time.monotonic() - 0.1
    assert ws.supports_websocket() is True


def test_websocket_feed_on_close_honors_temp_disable():
    ws = WebSocketFeed()
    ws._ws_client_installed = True
    ws._running = False
    ws._ws_disabled_until_ts = time.monotonic() + 60.0
    ws.status = FeedStatus.RECONNECTING
    before = int(ws._reconnect_count)

    ws._on_close(None, None, None)

    assert ws.status == FeedStatus.ERROR
    assert int(ws._reconnect_count) == before


def test_websocket_feed_on_error_ignores_late_callbacks_while_disabled():
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


def test_websocket_feed_max_reconnect_attempts_enters_cooldown():
    ws = WebSocketFeed()
    ws._ws_client_installed = True
    ws._running = True
    ws._reconnect_count = int(ws._MAX_RECONNECT_ATTEMPTS)

    ws._on_close(None, None, None)

    assert ws.status == FeedStatus.ERROR
    assert ws._running is False
    assert ws.supports_websocket() is False
    assert float(ws._ws_disabled_until_ts) > time.monotonic()


def test_websocket_feed_can_be_forced_off_with_env(monkeypatch):
    monkeypatch.setenv("TRADING_DISABLE_WEBSOCKET", "1")
    ws = WebSocketFeed()
    ws._ws_client_installed = True
    assert ws.supports_websocket() is False
    assert ws.connect() is False


def test_websocket_feed_connect_blocked_on_vpn(monkeypatch):
    class _Env:
        is_vpn_active = True

    monkeypatch.setattr("core.network.peek_network_env", lambda: _Env())

    ws = WebSocketFeed()
    ws._ws_client_installed = True

    assert ws.connect() is False
    assert ws._running is False
    assert ws.supports_websocket() is False


def test_websocket_feed_connect_blocked_when_ws_host_dns_fails(monkeypatch):
    class _Env:
        is_vpn_active = False

    monkeypatch.setattr("core.network.peek_network_env", lambda: _Env())
    monkeypatch.setattr("socket.getaddrinfo", lambda *args, **kwargs: (_ for _ in ()).throw(OSError("dns fail")))

    ws = WebSocketFeed()
    ws._ws_client_installed = True

    assert ws.connect() is False
    assert ws._running is False
    assert ws.supports_websocket() is False


def test_websocket_feed_host_resolves_timeout_returns_unknown_as_true(monkeypatch):
    ws = WebSocketFeed()
    ws._ws_client_installed = True
    ws._last_dns_probe_host = ""

    monkeypatch.setattr(
        WebSocketFeed,
        "_resolve_host_with_timeout",
        staticmethod(lambda host, timeout_s: None),
    )

    assert ws._host_resolves("push.sina.cn") is True


def test_websocket_feed_host_resolves_uses_cache(monkeypatch):
    ws = WebSocketFeed()
    ws._ws_client_installed = True
    ws._DNS_PRECHECK_CACHE_SECONDS = 60.0

    calls = {"n": 0}

    def _fake_probe(host: str, timeout_s: float):
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
