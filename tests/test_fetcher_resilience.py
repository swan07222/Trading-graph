from __future__ import annotations

import threading
from datetime import datetime, timedelta
from types import SimpleNamespace
from typing import NoReturn

import pandas as pd

from data.fetcher import DataFetcher, DataSource, Quote, SinaHistorySource


class _DummyCache:
    def __init__(self) -> None:
        self._store = {}

    def get(self, key, ttl):  # noqa: ARG002
        return self._store.get(key)

    def set(self, key, value) -> None:
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


def test_realtime_batch_uses_non_tencent_batch_fallback_when_tencent_partial(monkeypatch) -> None:
    monkeypatch.setenv("TRADING_OFFLINE", "0")
    fetcher = _make_fetcher_for_realtime()

    class _Tencent:
        name = "tencent"

        def get_realtime_batch(self, codes):
            if "600519" in codes:
                return {"600519": _mk_quote("600519", 101.0, "tencent")}
            return {}

    class _Other:
        name = "akshare"

        def get_realtime_batch(self, codes):
            if "000001" in codes:
                return {"000001": _mk_quote("000001", 22.5, "akshare")}
            return {}

    fetcher._get_active_sources = lambda: [_Tencent(), _Other()]
    fetcher._fill_from_spot_cache = lambda missing, result: None
    fetcher._fill_from_single_source_quotes = lambda missing, result, sources: None
    fetcher._maybe_force_network_refresh = lambda: False
    fetcher._fallback_last_good = lambda codes: {}
    fetcher._fallback_last_close_from_db = lambda codes: {}

    out = fetcher.get_realtime_batch(["600519", "000001"])

    assert set(out.keys()) == {"600519", "000001"}
    assert out["600519"].source == "tencent"
    assert out["000001"].source == "akshare"


def test_realtime_batch_uses_non_tencent_single_quote_for_missing(monkeypatch) -> None:
    monkeypatch.setenv("TRADING_OFFLINE", "0")
    fetcher = _make_fetcher_for_realtime()

    class _Tencent:
        name = "tencent"

        def get_realtime_batch(self, codes):  # noqa: ARG002
            return {}

    class _Yahoo:
        name = "yahoo"

        @staticmethod
        def get_realtime(code):
            if str(code).zfill(6) == "000001":
                return _mk_quote("000001", 12.3, "yahoo")
            return None

    fetcher._get_active_sources = lambda: [_Tencent(), _Yahoo()]
    fetcher._fill_from_spot_cache = lambda missing, result: None
    fetcher._maybe_force_network_refresh = lambda: False
    fetcher._fallback_last_good = lambda codes: {}
    fetcher._fallback_last_close_from_db = lambda codes: {}

    out = fetcher.get_realtime_batch(["000001"])

    assert set(out.keys()) == {"000001"}
    assert out["000001"].source == "yahoo"


def test_realtime_batch_drops_stale_localdb_last_close_by_default(monkeypatch) -> None:
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

    assert "600519" not in out


def test_realtime_batch_partial_missing_keeps_fresh_last_good_only(monkeypatch) -> None:
    monkeypatch.setenv("TRADING_OFFLINE", "0")
    fetcher = _make_fetcher_for_realtime()

    class _S1:
        name = "tencent"

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

    assert set(out.keys()) == {"600519", "000001"}
    assert out["600519"].source == "s1"
    assert out["000001"].source == "last_good"


def test_realtime_batch_allows_stale_localdb_when_explicitly_opted_in(monkeypatch) -> None:
    monkeypatch.setenv("TRADING_OFFLINE", "0")
    fetcher = _make_fetcher_for_realtime()
    fetcher._allow_stale_realtime_fallback = True

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


def test_realtime_batch_uses_spot_cache_when_tencent_missing(monkeypatch) -> None:
    import data.fetcher as fetcher_mod

    monkeypatch.setenv("TRADING_OFFLINE", "0")
    fetcher = _make_fetcher_for_realtime()

    class _Spot:
        @staticmethod
        def get_quote(code):
            if str(code).zfill(6) != "600519":
                return None
            return {
                "name": "KWEICHOW MOUTAI",
                "price": 1888.8,
                "open": 1870.0,
                "high": 1899.0,
                "low": 1866.0,
                "close": 1880.0,
                "volume": 1200,
                "amount": 2266560.0,
                "change": 8.8,
                "change_pct": 0.47,
            }

    fetcher._get_active_sources = lambda: []
    fetcher._maybe_force_network_refresh = lambda: False
    fetcher._fallback_last_good = lambda codes: {}
    fetcher._fallback_last_close_from_db = lambda codes: {}

    monkeypatch.setattr(fetcher_mod, "get_spot_cache", lambda: _Spot())

    out = fetcher.get_realtime_batch(["600519"])

    assert "600519" in out
    assert out["600519"].source == "spot_cache"
    assert float(out["600519"].price) == 1888.8


def test_realtime_batch_spot_cache_tolerates_malformed_values(monkeypatch) -> None:
    import data.fetcher as fetcher_mod

    monkeypatch.setenv("TRADING_OFFLINE", "0")
    fetcher = _make_fetcher_for_realtime()

    class _Spot:
        @staticmethod
        def get_quote(code):
            code6 = str(code).zfill(6)
            if code6 == "600519":
                return {
                    "name": "KWEICHOW MOUTAI",
                    "price": "1888.8",
                    "open": "1870.0",
                    "high": "1899.0",
                    "low": "1866.0",
                    "close": "1880.0",
                    "volume": "1200",
                    "amount": "2266560.0",
                }
            if code6 == "000001":
                return {
                    "name": "BROKEN",
                    "price": "12.3",
                    "volume": "not-a-number",
                    "amount": object(),
                }
            return None

    fetcher._get_active_sources = lambda: []
    fetcher._maybe_force_network_refresh = lambda: False
    fetcher._fallback_last_good = lambda codes: {}
    fetcher._fallback_last_close_from_db = lambda codes: {}

    monkeypatch.setattr(fetcher_mod, "get_spot_cache", lambda: _Spot())

    out = fetcher.get_realtime_batch(["600519", "000001"])

    assert "600519" in out
    assert "000001" in out
    assert out["600519"].source == "spot_cache"
    assert float(out["600519"].price) == 1888.8
    assert out["000001"].source == "spot_cache"
    assert int(out["000001"].volume) == 0


def test_realtime_batch_drops_stale_spot_cache_snapshot(monkeypatch) -> None:
    import data.fetcher as fetcher_mod

    monkeypatch.setenv("TRADING_OFFLINE", "0")
    fetcher = _make_fetcher_for_realtime()

    class _Spot:
        _cache_time = pd.Timestamp("2026-02-01 09:30:00", tz="UTC").timestamp()

        @staticmethod
        def get_quote(code):  # noqa: ARG002
            return {
                "name": "KWEICHOW MOUTAI",
                "price": 1888.8,
                "open": 1870.0,
                "high": 1899.0,
                "low": 1866.0,
                "close": 1880.0,
                "volume": 1200,
                "amount": 2266560.0,
                "change": 8.8,
                "change_pct": 0.47,
            }

    fetcher._get_active_sources = lambda: []
    fetcher._maybe_force_network_refresh = lambda: False
    fetcher._fallback_last_good = lambda codes: {}
    fetcher._fallback_last_close_from_db = lambda codes: {}
    fetcher._realtime_quote_max_age_s = 5.0
    fetcher._allow_stale_realtime_fallback = False

    monkeypatch.setattr(fetcher_mod, "get_spot_cache", lambda: _Spot())

    out = fetcher.get_realtime_batch(["600519"])

    assert "600519" not in out


def test_last_good_fallback_marks_quote_delayed() -> None:
    fetcher = _make_fetcher_for_realtime()
    fetcher._last_good_quotes = {"600519": _mk_quote("600519", 101.0, "tencent")}

    out = fetcher._fallback_last_good(["600519"])

    assert "600519" in out
    assert out["600519"].price == 101.0
    assert out["600519"].source == "tencent"
    assert out["600519"].is_delayed is True


def test_data_source_half_open_probe_during_cooldown(monkeypatch) -> None:
    src = DataSource()
    src.status.disabled_until = datetime.now() + timedelta(seconds=60)
    src.status.consecutive_errors = 8
    src._next_half_open_probe_ts = 100.0

    now = {"t": 99.0}
    monkeypatch.setattr("data.fetcher_sources.time.monotonic", lambda: float(now["t"]))

    assert src.is_available() is False
    now["t"] = 100.0
    assert src.is_available() is True
    # Immediate re-check should be blocked until next probe interval.
    assert src.is_available() is False


def test_fetch_history_with_depth_retry_uses_larger_windows() -> None:
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


def test_fetch_history_policy_tries_online_before_localdb() -> None:
    fetcher = DataFetcher.__new__(DataFetcher)
    fetcher._rate_limiter = threading.Semaphore(1)
    fetcher._rate_limit = lambda source, interval: None  # noqa: ARG005

    idx = pd.date_range("2026-01-01", periods=40, freq="D")
    frame = pd.DataFrame(
        {
            "open": [10.0] * len(idx),
            "high": [10.2] * len(idx),
            "low": [9.8] * len(idx),
            "close": [10.0 + (i * 0.1) for i in range(len(idx))],
            "volume": [100] * len(idx),
            "amount": [1010.0] * len(idx),
        },
        index=idx,
    )
    call_order: list[str] = []

    class _Tencent:
        name = "tencent"

        def __init__(self) -> None:
            self.called = 0

        def get_history_instrument(self, inst, days, interval="1d"):  # noqa: ARG002
            self.called += 1
            call_order.append(self.name)
            return frame.copy()

    class _Akshare:
        name = "akshare"

        def __init__(self) -> None:
            self.called = 0

        def get_history_instrument(self, inst, days, interval="1d"):  # noqa: ARG002
            self.called += 1
            call_order.append(self.name)
            return frame.copy()

    class _Local:
        name = "localdb"

        def __init__(self) -> None:
            self.called = 0

        def get_history_instrument(self, inst, days, interval="1d"):  # noqa: ARG002
            self.called += 1
            call_order.append(self.name)
            return frame.copy()

    tencent = _Tencent()
    akshare = _Akshare()
    localdb = _Local()

    fetcher._get_active_sources = lambda: [localdb, tencent, akshare]
    fetcher._all_sources = [localdb, tencent, akshare]

    out = fetcher._fetch_from_sources_instrument(
        inst={"market": "CN", "asset": "EQUITY", "symbol": "600519"},
        days=200,
        interval="1d",
        include_localdb=True,
    )

    assert not out.empty
    assert call_order
    assert call_order[0] == "tencent"


def test_history_policy_keeps_nonlocal_fallback_when_single_online_source() -> None:
    fetcher = DataFetcher.__new__(DataFetcher)
    fetcher._rate_limiter = threading.Semaphore(1)
    fetcher._rate_limit = lambda source, interval: None  # noqa: ARG005

    idx = pd.date_range("2026-01-01", periods=20, freq="D")
    frame = pd.DataFrame(
        {
            "open": [10.0] * len(idx),
            "high": [10.2] * len(idx),
            "low": [9.8] * len(idx),
            "close": [10.1] * len(idx),
            "volume": [100] * len(idx),
            "amount": [1010.0] * len(idx),
        },
        index=idx,
    )

    class _Tencent:
        name = "tencent"

        def get_history_instrument(self, inst, days, interval="1d"):  # noqa: ARG002
            return frame.copy()

    src = _Tencent()
    fetcher._get_active_sources = lambda: [src]
    fetcher._all_sources = [src]

    out = fetcher._fetch_from_sources_instrument(
        inst={"market": "CN", "asset": "EQUITY", "symbol": "600519"},
        days=20,
        interval="1d",
        include_localdb=False,
    )
    assert not out.empty


def test_fetch_history_daily_collects_multiple_sources_for_consensus() -> None:
    fetcher = DataFetcher.__new__(DataFetcher)
    fetcher._rate_limiter = threading.Semaphore(1)
    fetcher._rate_limit = lambda source, interval: None  # noqa: ARG005

    idx = pd.date_range("2024-01-01", periods=420, freq="D")
    close_a = pd.Series([20.0 + (i * 0.03) for i in range(len(idx))], index=idx)
    close_b = close_a * 1.001

    frame_a = pd.DataFrame(
        {
            "open": close_a.shift(1).fillna(close_a.iloc[0]),
            "high": close_a + 0.2,
            "low": close_a - 0.2,
            "close": close_a,
            "volume": [1000] * len(idx),
            "amount": (close_a * 1000.0).values,
        },
        index=idx,
    )
    frame_b = pd.DataFrame(
        {
            "open": close_b.shift(1).fillna(close_b.iloc[0]),
            "high": close_b + 0.2,
            "low": close_b - 0.2,
            "close": close_b,
            "volume": [1100] * len(idx),
            "amount": (close_b * 1100.0).values,
        },
        index=idx,
    )

    class _Tencent:
        name = "tencent"

        def __init__(self) -> None:
            self.called = 0

        def get_history_instrument(self, inst, days, interval="1d"):  # noqa: ARG002
            self.called += 1
            return frame_a.copy()

    class _Sina:
        name = "sina"

        def __init__(self) -> None:
            self.called = 0

        def get_history_instrument(self, inst, days, interval="1d"):  # noqa: ARG002
            self.called += 1
            return frame_b.copy()

    tencent = _Tencent()
    sina = _Sina()
    fetcher._get_active_sources = lambda: [tencent, sina]
    fetcher._all_sources = [tencent, sina]

    out = fetcher._fetch_from_sources_instrument(
        inst={"market": "CN", "asset": "EQUITY", "symbol": "600519"},
        days=420,
        interval="1d",
        include_localdb=False,
    )

    assert not out.empty
    assert tencent.called == 1
    assert sina.called == 1


def test_get_history_cn_daily_skips_db_upsert_when_quorum_fails() -> None:
    fetcher = DataFetcher.__new__(DataFetcher)
    fetcher._cache = _DummyCache()

    idx = pd.date_range("2026-01-01", periods=40, freq="D")
    online_df = pd.DataFrame(
        {
            "open": [10.0] * len(idx),
            "high": [10.2] * len(idx),
            "low": [9.8] * len(idx),
            "close": [10.1] * len(idx),
            "volume": [1000] * len(idx),
            "amount": [10100.0] * len(idx),
        },
        index=idx,
    )

    class _DB:
        def __init__(self) -> None:
            self.upsert_calls = 0

        @staticmethod
        def get_bars(code, limit=1000):  # noqa: ARG002
            return pd.DataFrame()

        def upsert_bars(self, code, df) -> None:  # noqa: ARG002
            self.upsert_calls += 1

    db = _DB()
    fetcher._db = db
    fetcher._fetch_history_with_depth_retry = lambda **kwargs: (  # type: ignore[method-assign]
        online_df.copy(),
        {
            "quorum_passed": False,
            "agreeing_points": 3,
            "compared_points": 20,
            "agreeing_ratio": 0.15,
            "required_sources": 2,
            "sources": ["tencent", "akshare", "sina"],
            "reason": "insufficient_consensus",
        },
    )

    out = fetcher._get_history_cn_daily(
        inst={"market": "CN", "asset": "EQUITY", "symbol": "600519"},
        count=20,
        fetch_days=30,
        cache_key="history:test:600519:1d",
        offline=False,
        update_db=True,
        session_df=pd.DataFrame(),
        interval="1d",
    )

    assert not out.empty
    assert db.upsert_calls == 0


def test_get_history_cn_daily_upserts_when_quorum_passes() -> None:
    fetcher = DataFetcher.__new__(DataFetcher)
    fetcher._cache = _DummyCache()

    idx = pd.date_range("2026-01-01", periods=30, freq="D")
    online_df = pd.DataFrame(
        {
            "open": [10.0] * len(idx),
            "high": [10.2] * len(idx),
            "low": [9.8] * len(idx),
            "close": [10.1] * len(idx),
            "volume": [1000] * len(idx),
            "amount": [10100.0] * len(idx),
        },
        index=idx,
    )

    class _DB:
        def __init__(self) -> None:
            self.upsert_calls = 0

        @staticmethod
        def get_bars(code, limit=1000):  # noqa: ARG002
            return pd.DataFrame()

        def upsert_bars(self, code, df) -> None:  # noqa: ARG002
            self.upsert_calls += 1

    db = _DB()
    fetcher._db = db
    fetcher._fetch_history_with_depth_retry = lambda **kwargs: (  # type: ignore[method-assign]
        online_df.copy(),
        {
            "quorum_passed": True,
            "agreeing_points": 24,
            "compared_points": 27,
            "agreeing_ratio": 0.89,
            "required_sources": 2,
            "sources": ["tencent", "akshare", "sina"],
            "reason": "",
        },
    )

    out = fetcher._get_history_cn_daily(
        inst={"market": "CN", "asset": "EQUITY", "symbol": "600519"},
        count=20,
        fetch_days=30,
        cache_key="history:test:600519:1d",
        offline=False,
        update_db=True,
        session_df=pd.DataFrame(),
        interval="1d",
    )

    assert not out.empty
    assert db.upsert_calls == 1


def test_sina_source_parses_kline_payload(monkeypatch) -> None:
    monkeypatch.setattr(
        "core.network.get_network_env",
        lambda: SimpleNamespace(is_china_direct=True),
    )
    src = SinaHistorySource()

    class _Resp:
        status_code = 200
        text = (
            '{"result":{"data":['
            '{"day":"2026-02-10 09:31:00","open":"10.0","high":"10.5","low":"9.8","close":"10.2","volume":"1000"},'
            '{"day":"2026-02-11 09:31:00","open":"10.2","high":"10.6","low":"10.1","close":"10.4","volume":"900"}'
            "]}}"
        )

    monkeypatch.setattr(src._session, "get", lambda *args, **kwargs: _Resp())

    out = src.get_history_instrument(
        {"market": "CN", "asset": "EQUITY", "symbol": "600519"},
        days=3,
        interval="1d",
    )

    assert not out.empty
    assert "close" in out.columns
    assert float(out["close"].iloc[-1]) > 0


def test_sina_source_not_available_off_china_direct(monkeypatch) -> None:
    monkeypatch.setattr(
        "core.network.get_network_env",
        lambda: SimpleNamespace(is_china_direct=False),
    )
    src = SinaHistorySource()
    assert src.is_available() is False


def test_sina_source_parses_jsonp_wrapper(monkeypatch) -> None:
    monkeypatch.setattr(
        "core.network.get_network_env",
        lambda: SimpleNamespace(is_china_direct=True),
    )
    src = SinaHistorySource()

    class _Resp:
        status_code = 200
        text = (
            'cb123(['
            '{"day":"2026-02-10 09:31:00","open":"9.0","high":"9.8","low":"8.9","close":"9.6","volume":"700"},'
            '{"day":"2026-02-10 09:32:00","open":"9.6","high":"9.9","low":"9.5","close":"9.7","volume":"600"}'
            ']);'
        )

    monkeypatch.setattr(src._session, "get", lambda *args, **kwargs: _Resp())

    out = src.get_history_instrument(
        {"market": "CN", "asset": "EQUITY", "symbol": "600519"},
        days=1,
        interval="2m",
    )
    assert not out.empty
    assert "close" in out.columns


def test_get_history_cn_intraday_does_not_use_session_shortcut() -> None:
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

    def _fake_intraday(inst, count, fetch_days, interval, cache_key, offline, session, *, persist_intraday_db=True):  # noqa: ARG001
        called["intraday"] += 1
        idx_local = pd.date_range("2026-02-12 09:30:00", periods=int(count), freq="min")
        return pd.DataFrame(
            {
                "open": [10.0] * int(count),
                "high": [10.0] * int(count),
                "low": [10.0] * int(count),
                "close": [10.0] * int(count),
                "volume": [1] * int(count),
            },
            index=idx_local,
        )

    fetcher._get_history_cn_intraday = _fake_intraday

    out_small = fetcher.get_history(
        "600519",
        bars=200,
        interval="1m",
        instrument={"market": "CN", "asset": "EQUITY", "symbol": "600519"},
    )
    assert not out_small.empty
    assert len(out_small) == 200
    assert called["intraday"] == 1

    out_large = fetcher.get_history(
        "600519",
        bars=700,
        interval="1m",
        instrument={"market": "CN", "asset": "EQUITY", "symbol": "600519"},
    )
    assert not out_large.empty
    assert len(out_large) == 700
    assert called["intraday"] == 2


def test_get_history_post_close_exact_refresh_bypasses_session_shortcut() -> None:
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


def test_get_history_intraday_market_open_skips_db_persist(monkeypatch) -> None:
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


def test_get_history_intraday_market_closed_allows_db_persist(monkeypatch) -> None:
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


def test_get_history_cn_intraday_offline_filters_non_session_rows() -> None:
    fetcher = DataFetcher.__new__(DataFetcher)
    fetcher._cache = _DummyCache()

    idx = pd.to_datetime(
        ["2026-02-18 09:00:00", "2026-02-18 09:35:00", "2026-02-18 15:30:00"]
    )
    raw_df = pd.DataFrame(
        {
            "open": [10.0, 10.1, 10.2],
            "high": [10.2, 10.3, 10.4],
            "low": [9.8, 9.9, 10.0],
            "close": [10.0, 10.1, 10.2],
            "volume": [100, 120, 90],
        },
        index=idx,
    )

    class _DB:
        @staticmethod
        def get_intraday_bars(code, interval="1m", limit=1000):  # noqa: ARG002
            return raw_df.tail(int(limit)).copy()

    fetcher._db = _DB()

    out = fetcher._get_history_cn_intraday(
        inst={"market": "CN", "asset": "EQUITY", "symbol": "600519"},
        count=100,
        fetch_days=1,
        interval="1m",
        cache_key="history:cn:offline",
        offline=True,
        session_df=pd.DataFrame(),
        persist_intraday_db=False,
    )

    assert len(out) == 1
    assert out.index[0].strftime("%H:%M") == "09:35"


def test_get_history_cn_intraday_rejects_weak_online_snapshot() -> None:
    fetcher = DataFetcher.__new__(DataFetcher)
    fetcher._cache = _DummyCache()

    idx_db = pd.date_range("2026-02-18 09:30:00", periods=30, freq="min")
    db_df = pd.DataFrame(
        {
            "open": [10.0 + (i * 0.01) for i in range(30)],
            "high": [10.05 + (i * 0.01) for i in range(30)],
            "low": [9.95 + (i * 0.01) for i in range(30)],
            "close": [10.0 + (i * 0.01) for i in range(30)],
            "volume": [100] * 30,
            "amount": [1000.0] * 30,
        },
        index=idx_db,
    )
    idx_online = pd.date_range("2026-02-18 10:00:00", periods=30, freq="min")
    weak_online = pd.DataFrame(
        {
            "open": [8.0] * 30,
            "high": [8.0] * 30,
            "low": [8.0] * 30,
            "close": [8.0] * 30,
            "volume": [0] * 30,
            "amount": [0.0] * 30,
        },
        index=idx_online,
    )

    class _DB:
        @staticmethod
        def get_intraday_bars(code, interval="1m", limit=1000):  # noqa: ARG002
            return db_df.tail(int(limit)).copy()

    fetcher._db = _DB()
    fetcher._fetch_history_with_depth_retry = lambda **kwargs: weak_online.copy()

    out = fetcher._get_history_cn_intraday(
        inst={"market": "CN", "asset": "EQUITY", "symbol": "600519"},
        count=200,
        fetch_days=2,
        interval="1m",
        cache_key="history:cn:weak_online",
        offline=False,
        session_df=pd.DataFrame(),
        persist_intraday_db=False,
    )

    assert len(out) == len(db_df)
    assert float(out["close"].min()) >= 10.0


def test_get_history_normalizes_interval_alias_before_source_routing() -> None:
    fetcher = DataFetcher.__new__(DataFetcher)
    fetcher._cache = _DummyCache()
    fetcher._get_session_history = lambda symbol, interval, bars: pd.DataFrame()  # noqa: ARG005

    captured: dict[str, str] = {}

    def _fake_intraday(inst, count, fetch_days, interval, cache_key, offline, session, *, persist_intraday_db=True):  # noqa: ARG001
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


def test_get_history_cn_weekly_routes_to_daily_handler() -> None:
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


def test_network_force_refresh_is_rate_limited(monkeypatch) -> None:
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


def test_accept_online_intraday_snapshot_rejects_stale_online() -> None:
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


def test_accept_online_intraday_snapshot_accepts_clean_online() -> None:
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


def test_depth_retry_keeps_probing_when_first_intraday_window_is_bad() -> None:
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


def test_refresh_trained_stock_history_uses_last_saved_increment(monkeypatch) -> None:
    monkeypatch.setenv("TRADING_OFFLINE", "0")
    fetcher = DataFetcher.__new__(DataFetcher)

    now = datetime(2026, 2, 18, 14, 30, 0)
    monkeypatch.setattr(
        DataFetcher,
        "_now_shanghai_naive",
        staticmethod(lambda: now),
    )
    base_idx = pd.date_range(
        end=(now - timedelta(minutes=20)),
        periods=240,
        freq="min",
    )
    base_df = pd.DataFrame(
        {
            "open": [10.0] * len(base_idx),
            "high": [10.1] * len(base_idx),
            "low": [9.9] * len(base_idx),
            "close": [10.0] * len(base_idx),
            "volume": [100] * len(base_idx),
            "amount": [1000.0] * len(base_idx),
        },
        index=base_idx,
    )

    class _DB:
        def __init__(self, seed: pd.DataFrame) -> None:
            self.df = seed.copy()

        def get_intraday_bars(self, code, interval="1m", limit=1000):  # noqa: ARG002
            return self.df.tail(int(limit)).copy()

        def upsert_intraday_bars(self, code, interval, df) -> None:  # noqa: ARG002
            merged = pd.concat([self.df, df], axis=0)
            merged = merged[~merged.index.duplicated(keep="last")].sort_index()
            self.df = merged

        def get_bars(self, code, limit=1000):  # noqa: ARG002
            return pd.DataFrame()

        def upsert_bars(self, code, df) -> None:  # noqa: ARG002
            return None

    fetcher._db = _DB(base_df)
    calls: list[int] = []

    def _fake_fetch(inst, days, interval="1m", include_localdb=False):  # noqa: ARG001
        calls.append(int(days))
        idx = pd.date_range(end=now, periods=30, freq="min")
        return pd.DataFrame(
            {
                "open": [10.2] * len(idx),
                "high": [10.3] * len(idx),
                "low": [10.1] * len(idx),
                "close": [10.2] * len(idx),
                "volume": [120] * len(idx),
                "amount": [1224.0] * len(idx),
            },
            index=idx,
        )

    fetcher._fetch_from_sources_instrument = _fake_fetch

    out = fetcher.refresh_trained_stock_history(
        ["600519"],
        interval="1m",
        window_days=29,
        allow_online=True,
    )

    assert int(out["total"]) == 1
    assert int(out["updated"]) == 1
    assert calls
    assert int(calls[0]) <= 2
    rows = dict(out.get("rows", {}) or {})
    assert int(rows.get("600519", 0)) > 0


def test_refresh_trained_stock_history_replaces_realtime_cache_after_close(monkeypatch) -> None:
    import data.fetcher as fetcher_mod

    monkeypatch.setenv("TRADING_OFFLINE", "0")
    monkeypatch.setattr(fetcher_mod.CONFIG, "is_market_open", lambda: False, raising=False)

    now = datetime(2026, 2, 18, 15, 30, 0)
    monkeypatch.setattr(
        DataFetcher,
        "_now_shanghai_naive",
        staticmethod(lambda: now),
    )

    fetcher = DataFetcher.__new__(DataFetcher)

    class _DB:
        def __init__(self) -> None:
            self.df = pd.DataFrame()

        def get_intraday_bars(self, code, interval="1m", limit=1000):  # noqa: ARG002
            if self.df.empty:
                return pd.DataFrame()
            return self.df.tail(int(limit)).copy()

        def upsert_intraday_bars(self, code, interval, df) -> None:  # noqa: ARG002
            merged = pd.concat([self.df, df], axis=0)
            merged = merged[~merged.index.duplicated(keep="last")].sort_index()
            self.df = merged

        def get_bars(self, code, limit=1000):  # noqa: ARG002
            return pd.DataFrame()

        def upsert_bars(self, code, df) -> None:  # noqa: ARG002
            return None

    fetcher._db = _DB()
    calls: list[int] = []

    def _fake_fetch(inst, days, interval="1m", include_localdb=False):  # noqa: ARG001
        calls.append(int(days))
        idx = pd.date_range(end=now, periods=40, freq="min")
        return pd.DataFrame(
            {
                "open": [11.0] * len(idx),
                "high": [11.1] * len(idx),
                "low": [10.9] * len(idx),
                "close": [11.0] * len(idx),
                "volume": [200] * len(idx),
                "amount": [2200.0] * len(idx),
            },
            index=idx,
        )

    fetcher._fetch_from_sources_instrument = _fake_fetch

    class _SessionCache:
        def __init__(self) -> None:
            self.purged_calls = 0
            self.upsert_calls = []
            self.purge_kwargs = []

        def describe_symbol_interval(self, symbol, interval):  # noqa: ARG002
            return {
                "rows": 100,
                "first_ts": now - timedelta(days=2),
                "last_ts": now - timedelta(minutes=1),
                "first_realtime_ts": now - timedelta(days=2),
                "first_realtime_after_akshare_ts": now - timedelta(hours=2),
                "last_akshare_ts": now - timedelta(hours=1),
            }

        def purge_realtime_rows(self, symbol, interval, *, since_ts=None) -> int:  # noqa: ARG002
            self.purged_calls += 1
            self.purge_kwargs.append({"since_ts": since_ts})
            return 100

        def upsert_history_frame(
            self,
            symbol,
            interval,
            frame,
            source="official_history",
            is_final=True,  # noqa: ARG002
        ):
            self.upsert_calls.append((str(symbol), str(interval), str(source), int(len(frame))))
            return int(len(frame))

    cache = _SessionCache()
    monkeypatch.setattr(fetcher_mod, "get_session_bar_cache", lambda: cache)

    out = fetcher.refresh_trained_stock_history(
        ["600519"],
        interval="1m",
        window_days=29,
        allow_online=True,
        sync_session_cache=True,
        replace_realtime_after_close=True,
    )

    assert int(out["total"]) == 1
    assert calls
    assert int(calls[0]) >= 2
    assert cache.purged_calls == 1
    assert cache.upsert_calls
    assert cache.upsert_calls[0][2] == "official_history"
    assert cache.purge_kwargs
    since_ts = cache.purge_kwargs[0]["since_ts"]
    assert since_ts is not None
    assert pd.Timestamp(since_ts) >= pd.Timestamp(now - timedelta(hours=1))
    purged = dict(out.get("purged_realtime_rows", {}) or {})
    assert int(purged.get("600519", 0)) == 100
    used = dict(out.get("replacement_anchor_used", {}) or {})
    assert str(used.get("600519", "")).strip() != ""


def test_refresh_trained_stock_history_retries_pending_cache_sync(monkeypatch, tmp_path) -> None:
    import data.fetcher as fetcher_mod

    monkeypatch.setenv("TRADING_OFFLINE", "0")
    monkeypatch.setattr(fetcher_mod.CONFIG, "is_market_open", lambda: True, raising=False)

    now = datetime(2026, 2, 18, 14, 30, 0)
    monkeypatch.setattr(
        DataFetcher,
        "_now_shanghai_naive",
        staticmethod(lambda: now),
    )

    fetcher = DataFetcher.__new__(DataFetcher)
    fetcher._refresh_reconcile_lock = threading.RLock()
    fetcher._refresh_reconcile_path = tmp_path / "refresh_reconcile_queue.json"

    class _DB:
        def __init__(self) -> None:
            self.df = pd.DataFrame()

        def get_intraday_bars(self, code, interval="1m", limit=1000):  # noqa: ARG002
            if self.df.empty:
                return pd.DataFrame()
            return self.df.tail(int(limit)).copy()

        def upsert_intraday_bars(self, code, interval, df) -> None:  # noqa: ARG002
            merged = pd.concat([self.df, df], axis=0)
            merged = merged[~merged.index.duplicated(keep="last")].sort_index()
            self.df = merged

        def get_bars(self, code, limit=1000):  # noqa: ARG002
            return pd.DataFrame()

        def upsert_bars(self, code, df) -> None:  # noqa: ARG002
            return None

    fetcher._db = _DB()

    def _fake_fetch(inst, days, interval="1m", include_localdb=False):  # noqa: ARG001
        idx = pd.date_range(end=now, periods=30, freq="min")
        return pd.DataFrame(
            {
                "open": [10.0] * len(idx),
                "high": [10.1] * len(idx),
                "low": [9.9] * len(idx),
                "close": [10.0] * len(idx),
                "volume": [100] * len(idx),
                "amount": [1000.0] * len(idx),
            },
            index=idx,
        )

    fetcher._fetch_from_sources_instrument = _fake_fetch

    class _FailCache:
        def describe_symbol_interval(self, symbol, interval):  # noqa: ARG002
            return {
                "rows": 0,
                "first_ts": None,
                "last_ts": None,
                "first_realtime_ts": None,
                "first_realtime_after_akshare_ts": None,
                "last_akshare_ts": None,
            }

        def upsert_history_frame(
            self,
            symbol,
            interval,
            frame,
            source="official_history",
            is_final=True,  # noqa: ARG002
        ) -> NoReturn:
            raise RuntimeError("cache write failed")

        def purge_realtime_rows(self, symbol, interval, *, since_ts=None) -> int:  # noqa: ARG002
            return 0

    monkeypatch.setattr(fetcher_mod, "get_session_bar_cache", lambda: _FailCache())

    out1 = fetcher.refresh_trained_stock_history(
        ["600519"],
        interval="1m",
        window_days=29,
        allow_online=True,
        sync_session_cache=True,
    )
    sync_errors_1 = dict(out1.get("cache_sync_errors", {}) or {})
    assert "600519" in sync_errors_1
    assert int(out1.get("pending_reconcile_after", 0)) == 1
    assert fetcher._refresh_reconcile_path.exists()

    class _GoodCache:
        def __init__(self) -> None:
            self.upsert_calls = 0

        def describe_symbol_interval(self, symbol, interval):  # noqa: ARG002
            return {
                "rows": 0,
                "first_ts": None,
                "last_ts": None,
                "first_realtime_ts": None,
                "first_realtime_after_akshare_ts": None,
                "last_akshare_ts": None,
            }

        def upsert_history_frame(
            self,
            symbol,
            interval,
            frame,
            source="official_history",
            is_final=True,  # noqa: ARG002
        ):
            self.upsert_calls += 1
            return int(len(frame))

        def purge_realtime_rows(self, symbol, interval, *, since_ts=None) -> int:  # noqa: ARG002
            return 0

    good_cache = _GoodCache()
    monkeypatch.setattr(fetcher_mod, "get_session_bar_cache", lambda: good_cache)

    out2 = fetcher.refresh_trained_stock_history(
        ["600519"],
        interval="1m",
        window_days=29,
        allow_online=False,
        sync_session_cache=True,
    )
    assert good_cache.upsert_calls >= 1
    assert int(out2.get("pending_reconcile_after", 0)) == 0


def test_reconcile_pending_cache_sync_clears_queue_on_success(monkeypatch, tmp_path) -> None:
    import data.fetcher as fetcher_mod

    fetcher = DataFetcher.__new__(DataFetcher)
    fetcher._refresh_reconcile_lock = threading.RLock()
    fetcher._refresh_reconcile_path = tmp_path / "refresh_reconcile_queue.json"

    idx = pd.date_range("2026-02-18 09:30:00", periods=30, freq="min")
    db_df = pd.DataFrame(
        {
            "open": [10.0] * len(idx),
            "high": [10.1] * len(idx),
            "low": [9.9] * len(idx),
            "close": [10.0] * len(idx),
            "volume": [100] * len(idx),
            "amount": [1000.0] * len(idx),
        },
        index=idx,
    )

    class _DB:
        def get_intraday_bars(self, code, interval="1m", limit=1000):  # noqa: ARG002
            return db_df.tail(int(limit)).copy()

        def get_bars(self, code, limit=1000):  # noqa: ARG002
            return pd.DataFrame()

    fetcher._db = _DB()

    fetcher._save_refresh_reconcile_queue(
        {
            "600519:1m": {
                "code": "600519",
                "interval": "1m",
                "pending_since": "2026-02-18T15:00:00",
                "attempts": 1,
                "last_attempt_at": "",
                "last_error": "cache write failed",
            }
        }
    )

    class _Cache:
        def __init__(self) -> None:
            self.upsert_calls = 0

        def upsert_history_frame(
            self,
            symbol,
            interval,
            frame,
            source="official_history",
            is_final=True,  # noqa: ARG002
        ):
            self.upsert_calls += 1
            assert str(symbol) == "600519"
            assert str(interval) == "1m"
            assert str(source) == "official_history"
            return int(len(frame))

        def describe_symbol_interval(self, symbol, interval):  # noqa: ARG002
            return {
                "first_realtime_after_akshare_ts": None,
            }

        def purge_realtime_rows(self, symbol, interval, *, since_ts=None) -> int:  # noqa: ARG002
            return 0

    cache = _Cache()
    monkeypatch.setattr(fetcher_mod, "get_session_bar_cache", lambda: cache)
    monkeypatch.setattr(fetcher_mod.CONFIG, "is_market_open", lambda: True, raising=False)

    out = fetcher.reconcile_pending_cache_sync(codes=["600519"], interval="1m")

    assert int(out.get("targeted", 0)) == 1
    assert int(out.get("reconciled", 0)) == 1
    assert int(out.get("failed", 0)) == 0
    assert int(out.get("remaining", 0)) == 0
    assert cache.upsert_calls == 1
    assert fetcher.get_pending_reconcile_codes(interval="1m") == []


def test_get_multiple_parallel_allows_short_requests_under_global_history_floor() -> None:
    fetcher = DataFetcher.__new__(DataFetcher)
    fetcher._all_sources = []

    idx = pd.date_range("2026-01-01", periods=50, freq="D")
    small_df = pd.DataFrame(
        {
            "open": [10.0] * len(idx),
            "high": [10.2] * len(idx),
            "low": [9.8] * len(idx),
            "close": [10.1] * len(idx),
            "volume": [100] * len(idx),
        },
        index=idx,
    )
    fetcher.get_history = lambda code, days=500, interval="1d", **kwargs: small_df.copy()  # type: ignore[method-assign]

    out = fetcher.get_multiple_parallel(
        ["600519", "000001"],
        days=20,
        interval="1d",
        max_workers=2,
    )

    assert set(out.keys()) == {"600519", "000001"}
    assert all(len(df) == 50 for df in out.values())


def test_get_multiple_parallel_clamps_invalid_negative_worker_count() -> None:
    fetcher = DataFetcher.__new__(DataFetcher)
    fetcher._all_sources = []

    idx = pd.date_range("2026-01-01", periods=20, freq="D")
    df = pd.DataFrame(
        {
            "open": [10.0] * len(idx),
            "high": [10.2] * len(idx),
            "low": [9.8] * len(idx),
            "close": [10.1] * len(idx),
            "volume": [100] * len(idx),
        },
        index=idx,
    )
    fetcher.get_history = lambda code, days=500, interval="1d", **kwargs: df.copy()  # type: ignore[method-assign]

    out = fetcher.get_multiple_parallel(
        ["600519"],
        days=20,
        interval="1d",
        max_workers=-1,
    )

    assert set(out.keys()) == {"600519"}
