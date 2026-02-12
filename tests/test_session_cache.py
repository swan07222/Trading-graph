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


def test_session_cache_epoch_milliseconds_timestamp(tmp_path):
    cache = SessionBarCache(root=tmp_path / "session_bars")

    ok = cache.append_bar(
        "600519",
        "1m",
        {
            "timestamp": 1700000000000,  # 2023-11-14T22:13:20Z
            "open": 10,
            "high": 11,
            "low": 9,
            "close": 10.5,
            "volume": 100,
            "final": True,
        },
    )
    assert ok

    df = cache.read_history("600519", "1m", bars=10)
    assert not df.empty
    assert df.index[-1].year == 2023


def test_session_cache_epoch_seconds_timestamp(tmp_path):
    cache = SessionBarCache(root=tmp_path / "session_bars")

    ok = cache.append_bar(
        "600519",
        "1m",
        {
            "timestamp": 1700000000,  # 2023-11-14T22:13:20Z
            "open": 20,
            "high": 21,
            "low": 19,
            "close": 20.5,
            "volume": 100,
            "final": True,
        },
    )
    assert ok

    df = cache.read_history("600519", "1m", bars=10)
    assert not df.empty
    assert df.index[-1].year == 2023
