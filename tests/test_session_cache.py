import csv

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


def test_session_cache_skips_immediate_duplicate_rows(tmp_path):
    cache = SessionBarCache(root=tmp_path / "session_bars")
    bar = {
        "timestamp": "2026-02-12T09:30:00+00:00",
        "open": 100,
        "high": 101,
        "low": 99,
        "close": 100.5,
        "volume": 1000,
        "final": True,
    }
    assert cache.append_bar("600519", "1m", bar) is True
    assert cache.append_bar("600519", "1m", bar) is False

    df = cache.read_history("600519", "1m", bars=10)
    assert len(df) == 1


def test_session_cache_normalizes_bad_numeric_inputs(tmp_path):
    cache = SessionBarCache(root=tmp_path / "session_bars")
    ok = cache.append_bar(
        "600519",
        "1m",
        {
            "timestamp": "2026-02-12T09:35:00+00:00",
            "open": "oops",
            "high": None,
            "low": "nan",
            "close": "10.5",
            "volume": "nan",
            "amount": "inf",
            "final": True,
        },
    )
    assert ok is True

    df = cache.read_history("600519", "1m", bars=10)
    assert not df.empty
    row = df.iloc[-1]
    assert row["close"] == 10.5
    assert row["open"] == 10.5
    assert row["high"] >= row["close"]
    assert row["low"] <= row["close"]


def test_session_cache_read_history_scrubs_outlier_jumps(tmp_path):
    cache = SessionBarCache(root=tmp_path / "session_bars")
    symbol = "600519"
    interval = "1m"

    bars = [
        {
            "timestamp": "2026-02-12T09:30:00+00:00",
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.0,
            "final": True,
        },
        {
            "timestamp": "2026-02-12T09:31:00+00:00",
            "open": 250.0,
            "high": 255.0,
            "low": 245.0,
            "close": 250.0,
            "final": True,
        },
        {
            "timestamp": "2026-02-12T09:32:00+00:00",
            "open": 100.5,
            "high": 101.5,
            "low": 100.0,
            "close": 101.0,
            "final": True,
        },
    ]

    for row in bars:
        assert cache.append_bar(symbol, interval, row) is True

    df = cache.read_history(symbol, interval, bars=10)
    assert not df.empty
    assert len(df) == 2
    assert float(df["close"].max()) < 200.0


def test_session_cache_prefers_recent_segment_after_scale_regime_jump(tmp_path):
    cache = SessionBarCache(root=tmp_path / "session_bars")
    symbol = "601318"
    interval = "1m"

    path = cache.root / f"{symbol}_{interval}.csv"
    rows = [
        ["2026-02-17T09:30:00+08:00", 1.48, 1.49, 1.47, 1.48, 100, 0.0, True],
        ["2026-02-17T09:31:00+08:00", 1.48, 1.49, 1.47, 1.485, 100, 0.0, True],
        ["2026-02-17T10:05:00+08:00", 79.05, 79.12, 79.00, 79.08, 100, 0.0, True],
        ["2026-02-17T10:06:00+08:00", 79.08, 79.15, 79.02, 79.10, 100, 0.0, True],
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["timestamp", "open", "high", "low", "close", "volume", "amount", "is_final"]
        )
        for row in rows:
            writer.writerow(row)

    df = cache.read_history(symbol, interval, bars=50, final_only=True)
    assert not df.empty
    assert float(df["close"].median()) > 10.0
    assert float(df["close"].iloc[-1]) > 70.0


def test_session_cache_rejects_outlier_append_write(tmp_path):
    cache = SessionBarCache(root=tmp_path / "session_bars")
    symbol = "600519"
    interval = "1m"

    assert cache.append_bar(
        symbol,
        interval,
        {
            "timestamp": "2026-02-17T09:30:00+08:00",
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.0,
            "final": True,
        },
    ) is True

    # Huge jump should be rejected by append-time guard.
    assert cache.append_bar(
        symbol,
        interval,
        {
            "timestamp": "2026-02-17T09:31:00+08:00",
            "open": 1.50,
            "high": 1.60,
            "low": 1.40,
            "close": 1.50,
            "final": True,
        },
    ) is False

    df = cache.read_history(symbol, interval, bars=10)
    assert not df.empty
    assert len(df) == 1
    assert abs(float(df["close"].iloc[-1]) - 100.0) < 1e-9


def test_session_cache_final_only_salvages_legacy_non_final_rows(tmp_path):
    cache = SessionBarCache(root=tmp_path / "session_bars")
    symbol = "000333"
    interval = "1m"

    rows = [
        {
            "timestamp": "2026-02-17T09:30:00+08:00",
            "open": 79.00,
            "high": 79.06,
            "low": 78.98,
            "close": 79.05,
            "final": False,
        },
        {
            "timestamp": "2026-02-17T09:31:00+08:00",
            "open": 79.05,
            "high": 79.10,
            "low": 79.02,
            "close": 79.08,
            "final": False,
        },
        {
            "timestamp": "2026-02-17T09:32:00+08:00",
            "open": 79.08,
            "high": 79.12,
            "low": 79.04,
            "close": 79.10,
            "final": False,
        },
    ]
    for row in rows:
        assert cache.append_bar(symbol, interval, row) is True

    # Legacy files with only non-final rows should still yield stable history.
    df = cache.read_history(symbol, interval, bars=10, final_only=True)
    assert not df.empty
    assert len(df) == 2
    assert abs(float(df["close"].iloc[-1]) - 79.08) < 1e-9


def test_session_cache_keeps_new_day_opening_gap_rows(tmp_path):
    cache = SessionBarCache(root=tmp_path / "session_bars")
    symbol = "600519"
    interval = "1m"

    rows = [
        {
            "timestamp": "2026-02-17T15:00:00+08:00",
            "open": 100.0,
            "high": 100.3,
            "low": 99.8,
            "close": 100.0,
            "final": True,
        },
        {
            # Next trading day open: valid overnight gap > intraday jump cap.
            "timestamp": "2026-02-18T09:30:00+08:00",
            "open": 112.0,
            "high": 112.4,
            "low": 111.7,
            "close": 112.1,
            "final": True,
        },
        {
            "timestamp": "2026-02-18T09:31:00+08:00",
            "open": 112.1,
            "high": 112.5,
            "low": 111.9,
            "close": 112.3,
            "final": True,
        },
    ]
    for row in rows:
        assert cache.append_bar(symbol, interval, row) is True

    df = cache.read_history(symbol, interval, bars=10)
    assert not df.empty
    assert len(df) == 3
    assert abs(float(df["close"].iloc[1]) - 112.1) < 1e-9
