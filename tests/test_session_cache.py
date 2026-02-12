from data.session_cache import SessionBarCache


def test_session_cache_append_and_read(tmp_path):
    cache = SessionBarCache(root=tmp_path / "session_bars")
    symbol = "600519"
    interval = "1m"

    for i in range(3):
        cache.append_bar(
            symbol,
            interval,
            {
                "timestamp": f"2026-02-12T09:3{i}:00+00:00",
                "open": 100 + i,
                "high": 101 + i,
                "low": 99 + i,
                "close": 100.5 + i,
                "volume": 1000 + i,
                "final": True,
            },
        )

    df = cache.read_history(symbol, interval, bars=10)
    assert not df.empty
    assert len(df) == 3
    assert "close" in df.columns

    symbols = cache.get_recent_symbols(interval=interval, min_rows=1)
    assert symbol in symbols
