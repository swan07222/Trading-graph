from __future__ import annotations

import threading
from datetime import datetime, timedelta

import pandas as pd

from data.fetcher import DataFetcher, DataSource, Quote


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


def test_last_good_fallback_marks_quote_delayed():
    fetcher = _make_fetcher_for_realtime()
    fetcher._last_good_quotes = {"600519": _mk_quote("600519", 101.0, "tencent")}

    out = fetcher._fallback_last_good(["600519"])

    assert "600519" in out
    assert out["600519"].price == 101.0
    assert out["600519"].source == "tencent"
    assert out["600519"].is_delayed is True


def test_data_source_half_open_probe_during_cooldown(monkeypatch):
    src = DataSource()
    src.status.disabled_until = datetime.now() + timedelta(seconds=60)
    src.status.consecutive_errors = 8
    src._next_half_open_probe_ts = 100.0

    now = {"t": 99.0}
    monkeypatch.setattr("data.fetcher.time.monotonic", lambda: float(now["t"]))

    assert src.is_available() is False
    now["t"] = 100.0
    assert src.is_available() is True
    # Immediate re-check should be blocked until next probe interval.
    assert src.is_available() is False


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


def test_get_history_post_close_exact_refresh_bypasses_session_shortcut():
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
    fetcher._should_refresh_intraday_exact = lambda **kwargs: True

    called = {"exact": 0, "intraday": 0}

    def _fake_exact(inst, count, fetch_days, interval, cache_key, offline):  # noqa: ARG001
        called["exact"] += 1
        return session_df.tail(count)

    def _fake_intraday(inst, count, fetch_days, interval, cache_key, offline, session):  # noqa: ARG001
        called["intraday"] += 1
        return session.tail(count)

    fetcher._get_history_cn_intraday_exact = _fake_exact
    fetcher._get_history_cn_intraday = _fake_intraday

    out = fetcher.get_history(
        "600519",
        bars=200,
        interval="1m",
        instrument={"market": "CN", "asset": "EQUITY", "symbol": "600519"},
        refresh_intraday_after_close=True,
    )

    assert not out.empty
    assert len(out) == 200
    assert called["exact"] == 1
    assert called["intraday"] == 0


def test_get_history_intraday_market_open_skips_db_persist(monkeypatch):
    fetcher = DataFetcher.__new__(DataFetcher)
    fetcher._cache = _DummyCache()
    fetcher._get_session_history = lambda symbol, interval, bars: pd.DataFrame()  # noqa: ARG005
    fetcher._should_refresh_intraday_exact = lambda **kwargs: False

    captured = {"persist": None}

    def _fake_intraday(
        inst,  # noqa: ARG001
        count,  # noqa: ARG001
        fetch_days,  # noqa: ARG001
        interval,  # noqa: ARG001
        cache_key,  # noqa: ARG001
        offline,  # noqa: ARG001
        session,  # noqa: ARG001
        *,
        persist_intraday_db=True,
    ):
        captured["persist"] = bool(persist_intraday_db)
        idx = pd.date_range("2026-02-12 10:00:00", periods=30, freq="min")
        return pd.DataFrame(
            {
                "open": [10.0] * 30,
                "high": [10.0] * 30,
                "low": [10.0] * 30,
                "close": [10.0] * 30,
                "volume": [1] * 30,
            },
            index=idx,
        )

    fetcher._get_history_cn_intraday = _fake_intraday
    monkeypatch.setattr("data.fetcher.CONFIG.is_market_open", lambda: True)

    out = fetcher.get_history(
        "600519",
        bars=30,
        interval="1m",
        update_db=True,
        instrument={"market": "CN", "asset": "EQUITY", "symbol": "600519"},
    )

    assert not out.empty
    assert captured["persist"] is False


def test_get_history_intraday_market_closed_allows_db_persist(monkeypatch):
    fetcher = DataFetcher.__new__(DataFetcher)
    fetcher._cache = _DummyCache()
    fetcher._get_session_history = lambda symbol, interval, bars: pd.DataFrame()  # noqa: ARG005
    fetcher._should_refresh_intraday_exact = lambda **kwargs: False

    captured = {"persist": None}

    def _fake_intraday(
        inst,  # noqa: ARG001
        count,  # noqa: ARG001
        fetch_days,  # noqa: ARG001
        interval,  # noqa: ARG001
        cache_key,  # noqa: ARG001
        offline,  # noqa: ARG001
        session,  # noqa: ARG001
        *,
        persist_intraday_db=True,
    ):
        captured["persist"] = bool(persist_intraday_db)
        idx = pd.date_range("2026-02-12 10:00:00", periods=30, freq="min")
        return pd.DataFrame(
            {
                "open": [10.0] * 30,
                "high": [10.0] * 30,
                "low": [10.0] * 30,
                "close": [10.0] * 30,
                "volume": [1] * 30,
            },
            index=idx,
        )

    fetcher._get_history_cn_intraday = _fake_intraday
    monkeypatch.setattr("data.fetcher.CONFIG.is_market_open", lambda: False)

    out = fetcher.get_history(
        "600519",
        bars=30,
        interval="1m",
        update_db=True,
        instrument={"market": "CN", "asset": "EQUITY", "symbol": "600519"},
    )

    assert not out.empty
    assert captured["persist"] is True


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


def test_get_history_cn_weekly_routes_to_daily_handler():
    fetcher = DataFetcher.__new__(DataFetcher)
    fetcher._cache = _DummyCache()
    fetcher._get_session_history = (
        lambda symbol, interval, bars: pd.DataFrame()  # noqa: ARG005
    )

    called = {"daily": 0, "intraday": 0, "interval": ""}

    def _fake_daily(  # noqa: ARG001
        inst,
        count,
        fetch_days,
        cache_key,
        offline,
        update_db,
        session_df=None,
        interval="1d",
    ):
        called["daily"] += 1
        called["interval"] = str(interval)
        idx = pd.date_range("2026-01-02", periods=int(count), freq="W-FRI")
        return pd.DataFrame(
            {
                "open": [10.0] * int(count),
                "high": [10.5] * int(count),
                "low": [9.5] * int(count),
                "close": [10.0] * int(count),
                "volume": [100] * int(count),
                "amount": [1000.0] * int(count),
            },
            index=idx,
        )

    def _fake_intraday(  # noqa: ARG001
        inst,
        count,
        fetch_days,
        interval,
        cache_key,
        offline,
        session,
        *,
        persist_intraday_db=True,
    ):
        called["intraday"] += 1
        return pd.DataFrame()

    fetcher._get_history_cn_daily = _fake_daily
    fetcher._get_history_cn_intraday = _fake_intraday

    out = fetcher.get_history(
        "600519",
        bars=12,
        interval="1wk",
        instrument={"market": "CN", "asset": "EQUITY", "symbol": "600519"},
    )

    assert not out.empty
    assert called["daily"] == 1
    assert called["intraday"] == 0
    assert called["interval"] == "1wk"


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


def test_accept_online_intraday_snapshot_rejects_stale_online():
    fetcher = DataFetcher.__new__(DataFetcher)

    idx = pd.date_range("2026-02-18 09:30:00", periods=180, freq="min")
    online = pd.DataFrame(
        {
            "open": [40.73] * len(idx),
            "high": [40.73] * len(idx),
            "low": [40.73] * len(idx),
            "close": [40.73] * len(idx),
            "volume": [0] * len(idx),
            "amount": [0.0] * len(idx),
        },
        index=idx,
    )
    base_close = pd.Series([40.50 + (i * 0.0015) for i in range(len(idx))], index=idx)
    baseline = pd.DataFrame(
        {
            "open": base_close.shift(1).fillna(base_close.iloc[0]),
            "high": base_close + 0.03,
            "low": base_close - 0.03,
            "close": base_close,
            "volume": [120] * len(idx),
            "amount": (base_close * 120.0).values,
        },
        index=idx,
    )

    ok = fetcher._accept_online_intraday_snapshot(
        symbol="603014",
        interval="1m",
        online_df=online,
        baseline_df=baseline,
    )
    assert ok is False


def test_accept_online_intraday_snapshot_accepts_clean_online():
    fetcher = DataFetcher.__new__(DataFetcher)

    idx_base = pd.date_range("2026-02-18 09:30:00", periods=120, freq="min")
    idx_online = pd.date_range("2026-02-18 09:30:00", periods=180, freq="min")
    base_close = pd.Series([50.0 + (i * 0.0010) for i in range(len(idx_base))], index=idx_base)
    online_close = pd.Series([50.0 + (i * 0.0016) for i in range(len(idx_online))], index=idx_online)

    baseline = pd.DataFrame(
        {
            "open": base_close.shift(1).fillna(base_close.iloc[0]),
            "high": base_close + 0.02,
            "low": base_close - 0.02,
            "close": base_close,
            "volume": [80] * len(idx_base),
            "amount": (base_close * 80.0).values,
        },
        index=idx_base,
    )
    online = pd.DataFrame(
        {
            "open": online_close.shift(1).fillna(online_close.iloc[0]),
            "high": online_close + 0.02,
            "low": online_close - 0.02,
            "close": online_close,
            "volume": [140] * len(idx_online),
            "amount": (online_close * 140.0).values,
        },
        index=idx_online,
    )

    ok = fetcher._accept_online_intraday_snapshot(
        symbol="603014",
        interval="1m",
        online_df=online,
        baseline_df=baseline,
    )
    assert ok is True


def test_depth_retry_keeps_probing_when_first_intraday_window_is_bad():
    fetcher = DataFetcher.__new__(DataFetcher)

    calls: list[int] = []

    def _fake_fetch(inst, days, interval, include_localdb=True):  # noqa: ARG001
        calls.append(int(days))
        idx = pd.date_range("2026-02-18 09:30:00", periods=240, freq="min")
        if len(calls) == 1:
            return pd.DataFrame(
                {
                    "open": [40.73] * len(idx),
                    "high": [40.73] * len(idx),
                    "low": [40.73] * len(idx),
                    "close": [40.73] * len(idx),
                    "volume": [0] * len(idx),
                    "amount": [0.0] * len(idx),
                },
                index=idx,
            )
        closes = pd.Series([40.50 + (i * 0.002) for i in range(len(idx))], index=idx)
        return pd.DataFrame(
            {
                "open": closes.shift(1).fillna(closes.iloc[0]),
                "high": closes + 0.03,
                "low": closes - 0.03,
                "close": closes,
                "volume": [120] * len(idx),
                "amount": (closes * 120.0).values,
            },
            index=idx,
        )

    fetcher._fetch_from_sources_instrument = _fake_fetch

    out = fetcher._fetch_history_with_depth_retry(
        inst={"market": "CN", "asset": "EQUITY", "symbol": "603014"},
        interval="1m",
        requested_count=200,
        base_fetch_days=5,
    )

    assert len(calls) >= 2
    q = fetcher._intraday_frame_quality(out, "1m")
    assert float(q["stale_ratio"]) < 0.20
