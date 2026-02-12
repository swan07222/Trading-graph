from __future__ import annotations

import time

from data.session_cache import SessionBarCache


def test_performance_session_cache_append_smoke(tmp_path):
    cache = SessionBarCache(root=tmp_path / "session_bars")
    symbol = "600519"
    interval = "1m"

    started = time.perf_counter()
    for i in range(2000):
        ok = cache.append_bar(
            symbol,
            interval,
            {
                "timestamp": f"2026-02-12T09:{i // 60:02d}:{i % 60:02d}+00:00",
                "open": 100.0 + (i * 0.01),
                "high": 101.0 + (i * 0.01),
                "low": 99.0 + (i * 0.01),
                "close": 100.5 + (i * 0.01),
                "volume": 1000 + i,
                "final": True,
            },
        )
        assert ok
    elapsed = time.perf_counter() - started

    # Smoke threshold: catches severe regressions without being machine-sensitive.
    assert elapsed < 8.0, f"append_bar too slow: {elapsed:.2f}s for 2000 bars"


def test_performance_session_cache_read_smoke(tmp_path):
    cache = SessionBarCache(root=tmp_path / "session_bars")
    symbol = "600519"
    interval = "1m"

    for i in range(3000):
        cache.append_bar(
            symbol,
            interval,
            {
                "timestamp": f"2026-02-12T10:{i // 60:02d}:{i % 60:02d}+00:00",
                "open": 100.0 + (i * 0.01),
                "high": 101.0 + (i * 0.01),
                "low": 99.0 + (i * 0.01),
                "close": 100.5 + (i * 0.01),
                "volume": 1000 + i,
                "final": True,
            },
        )

    started = time.perf_counter()
    df = cache.read_history(symbol, interval, bars=500)
    elapsed = time.perf_counter() - started

    assert not df.empty
    assert len(df) == 500
    assert elapsed < 2.0, f"read_history too slow: {elapsed:.2f}s for 3000-bar file"

