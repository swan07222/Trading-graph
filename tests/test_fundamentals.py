from __future__ import annotations

import pandas as pd

from data.fundamentals import FundamentalDataService, _safe_float


def test_safe_float_parses_percent_and_cn_units() -> None:
    assert _safe_float("15.5%") == 15.5
    assert _safe_float("1,234.5\u4ebf") == 1234.5 * 1e8
    assert _safe_float("3500\u4e07") == 3500 * 1e4
    assert _safe_float("--") is None


def test_fundamental_service_proxy_snapshot_scores_are_bounded(monkeypatch) -> None:
    monkeypatch.setenv("TRADING_FUNDAMENTALS_ONLINE", "0")

    class _Fetcher:
        def get_history(
            self,
            _symbol: str,
            interval: str = "1d",
            bars: int = 150,
            use_cache: bool = True,
            update_db: bool = False,
            allow_online: bool = False,
        ) -> pd.DataFrame:
            n = int(bars)
            idx = pd.date_range("2026-01-01", periods=n, freq="1D")
            close = pd.Series(range(100, 100 + n), index=idx, dtype=float)
            volume = pd.Series([1_000_000 + (i * 1_000) for i in range(n)], index=idx, dtype=float)
            return pd.DataFrame({"close": close, "volume": volume}, index=idx)

    import data.fetcher as fetcher_mod

    monkeypatch.setattr(fetcher_mod, "get_fetcher", lambda: _Fetcher())
    svc = FundamentalDataService(cache_ttl_seconds=600.0)
    snap = svc.get_snapshot("600519", force_refresh=True)

    assert snap.symbol == "600519"
    assert snap.source == "proxy"
    assert 0.0 <= snap.value_score <= 1.0
    assert 0.0 <= snap.quality_score <= 1.0
    assert 0.0 <= snap.growth_score <= 1.0
    assert 0.0 <= snap.composite_score <= 1.0


def test_fundamental_service_cache_reuses_snapshot(monkeypatch) -> None:
    monkeypatch.setenv("TRADING_FUNDAMENTALS_ONLINE", "0")
    calls = {"n": 0}

    class _Fetcher:
        def get_history(
            self,
            _symbol: str,
            interval: str = "1d",
            bars: int = 150,
            use_cache: bool = True,
            update_db: bool = False,
            allow_online: bool = False,
        ) -> pd.DataFrame:
            calls["n"] += 1
            idx = pd.date_range("2026-01-01", periods=80, freq="1D")
            close = pd.Series([100 + i for i in range(80)], index=idx, dtype=float)
            volume = pd.Series([2_000_000 for _ in range(80)], index=idx, dtype=float)
            return pd.DataFrame({"close": close, "volume": volume}, index=idx)

    import data.fetcher as fetcher_mod

    monkeypatch.setattr(fetcher_mod, "get_fetcher", lambda: _Fetcher())
    svc = FundamentalDataService(cache_ttl_seconds=600.0)

    first = svc.get_snapshot("000001")
    second = svc.get_snapshot("000001")

    assert first is second
    assert calls["n"] == 1
