from __future__ import annotations

import threading
from datetime import datetime

import pandas as pd

from data.fetcher import DataFetcher, Quote


class _DummyCache:
    def __init__(self):
        self._store = {}

    def get(self, key, ttl):  # noqa: ARG002
        return self._store.get(key)

    def set(self, key, value):
        self._store[key] = value


def _mk_quote(code: str, px: float, source: str = "dummy") -> Quote:
    return Quote(
        code=str(code).zfill(6),
        name="",
        price=float(px),
        open=float(px),
        high=float(px),
        low=float(px),
        close=float(px),
        volume=0,
        amount=0.0,
        source=source,
        is_delayed=False,
        latency_ms=1.0,
    )


def _make_fetcher_for_realtime() -> DataFetcher:
    f = DataFetcher.__new__(DataFetcher)
    f._rt_cache_lock = threading.RLock()
    f._rt_batch_microcache = {"ts": 0.0, "key": None, "data": {}}
    f._last_good_lock = threading.RLock()
    f._last_good_quotes = {}
    f._last_network_force_refresh_ts = 0.0
    f._network_force_refresh_cooldown_s = 20.0
    f._cache = _DummyCache()
    return f


def test_realtime_batch_merges_partial_results_across_sources(monkeypatch):
    monkeypatch.setenv("TRADING_OFFLINE", "0")
    fetcher = _make_fetcher_for_realtime()

    class _S1:
        def get_realtime_batch(self, codes):
            if "600519" in codes:
                return {"600519": _mk_quote("600519", 101.0, "s1")}
            return {}

    class _S2:
        def get_realtime_batch(self, codes):
            if "000001" in codes:
                return {"000001": _mk_quote("000001", 22.5, "s2")}
            return {}

    fetcher._get_active_sources = lambda: [_S1(), _S2()]
    fetcher._fill_from_spot_cache = lambda missing, result: None
    fetcher._fill_from_single_source_quotes = lambda missing, result, sources: None
    fetcher._maybe_force_network_refresh = lambda: False
    fetcher._fallback_last_good = lambda codes: {}
    fetcher._fallback_last_close_from_db = lambda codes: {}

    out = fetcher.get_realtime_batch(["600519", "000001"])

    assert set(out.keys()) == {"600519", "000001"}
    assert out["600519"].source == "s1"
    assert out["000001"].source == "s2"


def test_realtime_batch_falls_back_to_localdb_last_close_when_live_unavailable(monkeypatch):
    monkeypatch.setenv("TRADING_OFFLINE", "0")
    fetcher = _make_fetcher_for_realtime()

    class _DB:
        def get_bars(self, code: str, limit: int = 1):  # noqa: ARG002
            if str(code).zfill(6) != "600519":
                return pd.DataFrame()
            idx = pd.DatetimeIndex([pd.Timestamp("2026-02-10 15:00:00")])
            return pd.DataFrame(
                {
                    "open": [100.0],
                    "high": [103.0],
                    "low": [99.0],
                    "close": [101.5],
                    "volume": [1000],
                    "amount": [101500.0],
                },
                index=idx,
            )

    fetcher._db = _DB()
    fetcher._get_active_sources = lambda: []
    fetcher._fill_from_spot_cache = lambda missing, result: None
    fetcher._fill_from_single_source_quotes = lambda missing, result, sources: None
    fetcher._maybe_force_network_refresh = lambda: False
    fetcher._fallback_last_good = lambda codes: {}

    out = fetcher.get_realtime_batch(["600519"])

    assert "600519" in out
    assert out["600519"].price == 101.5
    assert out["600519"].source == "localdb_last_close"
    assert out["600519"].is_delayed is True


def test_realtime_batch_partial_missing_uses_last_good_then_localdb(monkeypatch):
    monkeypatch.setenv("TRADING_OFFLINE", "0")
    fetcher = _make_fetcher_for_realtime()

    class _S1:
        def get_realtime_batch(self, codes):
            if "600519" in codes:
                return {"600519": _mk_quote("600519", 101.0, "s1")}
            return {}

    class _DB:
        def get_bars(self, code: str, limit: int = 1):  # noqa: ARG002
            if str(code).zfill(6) != "000002":
                return pd.DataFrame()
            idx = pd.DatetimeIndex([pd.Timestamp("2026-02-10 15:00:00")])
            return pd.DataFrame(
                {
                    "open": [10.0],
                    "high": [10.2],
                    "low": [9.8],
                    "close": [10.1],
                    "volume": [1000],
                    "amount": [10100.0],
                },
                index=idx,
            )

    fetcher._db = _DB()
    fetcher._get_active_sources = lambda: [_S1()]
    fetcher._fill_from_spot_cache = lambda missing, result: None
    fetcher._fill_from_single_source_quotes = lambda missing, result, sources: None
    fetcher._maybe_force_network_refresh = lambda: False
    fetcher._last_good_quotes = {"000001": _mk_quote("000001", 22.5, "last_good")}

    out = fetcher.get_realtime_batch(["600519", "000001", "000002"])

    assert set(out.keys()) == {"600519", "000001", "000002"}
    assert out["600519"].source == "s1"
    assert out["000001"].source == "last_good"
    assert out["000002"].source == "localdb_last_close"


def test_fetch_history_with_depth_retry_uses_larger_windows():
    fetcher = DataFetcher.__new__(DataFetcher)

    calls: list[int] = []

    def _fake_fetch(inst, days, interval):  # noqa: ARG001
        calls.append(int(days))
        rows = 20 if int(days) <= 5 else 220
        idx = pd.date_range("2026-01-01", periods=rows, freq="min")
        return pd.DataFrame(
            {
                "open": [1.0] * rows,
                "high": [1.0] * rows,
                "low": [1.0] * rows,
                "close": [1.0] * rows,
                "volume": [1] * rows,
            },
            index=idx,
        )

    fetcher._fetch_from_sources_instrument = _fake_fetch

    out = fetcher._fetch_history_with_depth_retry(
        inst={"market": "CN", "asset": "EQUITY", "symbol": "600519"},
        interval="1m",
        requested_count=200,
        base_fetch_days=5,
    )

    assert len(out) >= 200
    assert calls[0] == 5
    assert any(x > 5 for x in calls)


def test_get_history_session_shortcut_only_for_small_intraday_windows():
    fetcher = DataFetcher.__new__(DataFetcher)
    fetcher._cache = _DummyCache()

    idx = pd.date_range("2026-02-12 09:30:00", periods=800, freq="min")
    session_df = pd.DataFrame(
        {
            "open": [10.0] * 800,
            "high": [10.0] * 800,
            "low": [10.0] * 800,
            "close": [10.0] * 800,
            "volume": [1] * 800,
        },
        index=idx,
    )

    fetcher._get_session_history = lambda symbol, interval, bars: session_df.tail(bars)  # noqa: ARG005
    called = {"intraday": 0}

    def _fake_intraday(inst, count, fetch_days, interval, cache_key, offline, session):  # noqa: ARG001
        called["intraday"] += 1
        return session.tail(count)

    fetcher._get_history_cn_intraday = _fake_intraday

    out_small = fetcher.get_history(
        "600519",
        bars=200,
        interval="1m",
        instrument={"market": "CN", "asset": "EQUITY", "symbol": "600519"},
    )
    assert not out_small.empty
    assert len(out_small) == 200
    assert called["intraday"] == 0

    out_large = fetcher.get_history(
        "600519",
        bars=700,
        interval="1m",
        instrument={"market": "CN", "asset": "EQUITY", "symbol": "600519"},
    )
    assert not out_large.empty
    assert len(out_large) == 700
    assert called["intraday"] == 1


def test_get_history_normalizes_interval_alias_before_source_routing():
    fetcher = DataFetcher.__new__(DataFetcher)
    fetcher._cache = _DummyCache()
    fetcher._get_session_history = lambda symbol, interval, bars: pd.DataFrame()  # noqa: ARG005

    captured: dict[str, str] = {}

    def _fake_intraday(inst, count, fetch_days, interval, cache_key, offline, session):  # noqa: ARG001
        captured["interval"] = str(interval)
        idx = pd.date_range("2026-02-12 10:00:00", periods=int(count), freq="h")
        return pd.DataFrame(
            {
                "open": [10.0] * int(count),
                "high": [10.0] * int(count),
                "low": [10.0] * int(count),
                "close": [10.0] * int(count),
                "volume": [1] * int(count),
            },
            index=idx,
        )

    fetcher._get_history_cn_intraday = _fake_intraday

    out = fetcher.get_history(
        "600519",
        bars=50,
        interval="1h",
        instrument={"market": "CN", "asset": "EQUITY", "symbol": "600519"},
    )

    assert not out.empty
    assert captured.get("interval") == "60m"


def test_network_force_refresh_is_rate_limited(monkeypatch):
    fetcher = _make_fetcher_for_realtime()
    fetcher._last_network_force_refresh_ts = 0.0
    fetcher._network_force_refresh_cooldown_s = 20.0

    calls = {"n": 0}

    def _fake_get_network_env(force_refresh: bool = False):  # noqa: ARG001
        calls["n"] += 1
        return {"ok": True, "ts": datetime.now().isoformat()}

    monkeypatch.setattr("core.network.get_network_env", _fake_get_network_env)
    t = {"now": 100.0}
    monkeypatch.setattr("data.fetcher.time.time", lambda: float(t["now"]))

    assert fetcher._maybe_force_network_refresh() is True
    assert calls["n"] == 1

    t["now"] = 105.0
    assert fetcher._maybe_force_network_refresh() is False
    assert calls["n"] == 1
