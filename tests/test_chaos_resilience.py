from __future__ import annotations

from analysis.strategy_marketplace import StrategyMarketplace
from data.session_cache import SessionBarCache


def test_chaos_marketplace_invalid_json_falls_back(tmp_path):
    strategies_dir = tmp_path / "strategies"
    strategies_dir.mkdir(parents=True, exist_ok=True)

    (strategies_dir / "marketplace.json").write_text("{invalid-json", encoding="utf-8")
    (strategies_dir / "enabled.json").write_text("{also-invalid", encoding="utf-8")

    marketplace = StrategyMarketplace(strategies_dir=strategies_dir)
    assert marketplace.list_entries() == []
    assert marketplace.get_enabled_ids() == []
    assert marketplace.get_enabled_files() == []


def test_chaos_session_cache_invalid_rows_are_ignored(tmp_path):
    cache = SessionBarCache(root=tmp_path / "session_bars")
    symbol = "600519"
    interval = "1m"

    path = cache.root / f"{symbol}_{interval}.csv"
    path.write_text(
        "timestamp,open,high,low,close,volume,amount,is_final\n"
        "bad-ts,1,1,1,1,1,1,true\n"
        "2026-02-12T09:30:00+00:00,10,11,9,0,100,1000,true\n"
        "2026-02-12T09:31:00+00:00,10,11,9,10.5,100,1000,true\n",
        encoding="utf-8",
    )

    df = cache.read_history(symbol, interval, bars=10)
    assert len(df) == 1
    assert float(df["close"].iloc[0]) == 10.5


def test_chaos_session_cache_rejects_nondict_bar(tmp_path):
    cache = SessionBarCache(root=tmp_path / "session_bars")
    assert cache.append_bar("600519", "1m", None) is False
    assert cache.read_history("600519", "1m", bars=10).empty

