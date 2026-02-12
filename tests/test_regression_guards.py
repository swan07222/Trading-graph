from __future__ import annotations

from analysis.strategy_marketplace import StrategyMarketplace
from data.session_cache import SessionBarCache


def test_regression_marketplace_skips_mismatch_hash(tmp_path):
    strategies_dir = tmp_path / "strategies"
    strategies_dir.mkdir(parents=True, exist_ok=True)

    script = strategies_dir / "demo.py"
    script.write_text(
        "def generate_signal(df, indicators, context):\n"
        "    return {'action': 'hold', 'score': 0.5}\n",
        encoding="utf-8",
    )

    # Deliberately wrong hash to verify mismatch is blocked.
    (strategies_dir / "marketplace.json").write_text(
        (
            "{"
            "\"version\": 1,"
            "\"strategies\": ["
            "{"
            "\"id\": \"demo\","
            "\"name\": \"Demo\","
            "\"version\": \"1.0\","
            "\"file\": \"demo.py\","
            "\"sha256\": \"deadbeef\","
            "\"enabled_by_default\": true"
            "}"
            "]"
            "}"
        ),
        encoding="utf-8",
    )

    marketplace = StrategyMarketplace(strategies_dir=strategies_dir)
    marketplace.save_enabled_ids(["demo"])
    files = marketplace.get_enabled_files()
    assert files == []


def test_regression_session_cache_keeps_latest_duplicate_timestamp(tmp_path):
    cache = SessionBarCache(root=tmp_path / "session_bars")
    symbol = "600519"
    interval = "1m"
    ts = "2026-02-12T09:30:00+00:00"

    cache.append_bar(
        symbol,
        interval,
        {
            "timestamp": ts,
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.0,
            "volume": 1000,
            "final": True,
        },
    )
    cache.append_bar(
        symbol,
        interval,
        {
            "timestamp": ts,
            "open": 102.0,
            "high": 103.0,
            "low": 101.0,
            "close": 102.5,
            "volume": 2000,
            "final": True,
        },
    )

    df = cache.read_history(symbol, interval, bars=10)
    assert len(df) == 1
    assert float(df["close"].iloc[0]) == 102.5

