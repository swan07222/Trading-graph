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
