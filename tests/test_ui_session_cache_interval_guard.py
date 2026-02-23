from __future__ import annotations

from ui.app import MainApp


class _RecorderCache:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, dict]] = []

    def append_bar(self, symbol, interval, bar) -> bool:
        self.calls.append((str(symbol), str(interval), dict(bar)))
        return True


class _DummyApp:
    _persist_session_bar = MainApp._persist_session_bar
    _normalize_interval_token = MainApp._normalize_interval_token

    def __init__(self) -> None:
        self._session_bar_cache = _RecorderCache()
        self._last_session_cache_write_ts: dict[str, float] = {}


def test_persist_session_bar_skips_non_1m_interval() -> None:
    app = _DummyApp()

    app._persist_session_bar(
        "600519",
        "15m",
        {
            "timestamp": "2026-02-17T10:00:00+08:00",
            "open": 100.0,
            "high": 100.5,
            "low": 99.8,
            "close": 100.2,
            "final": True,
        },
    )

    assert app._session_bar_cache.calls == []


def test_persist_session_bar_writes_1m_interval() -> None:
    app = _DummyApp()

    app._persist_session_bar(
        "600519",
        "1m",
        {
            "timestamp": "2026-02-17T10:01:00+08:00",
            "open": 101.0,
            "high": 101.3,
            "low": 100.9,
            "close": 101.1,
            "final": True,
        },
    )

    assert len(app._session_bar_cache.calls) == 1
    symbol, interval, payload = app._session_bar_cache.calls[0]
    assert symbol == "600519"
    assert interval == "1m"
    assert payload.get("interval") == "1m"
