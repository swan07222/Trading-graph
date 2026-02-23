from datetime import datetime

from data.feeds import BarAggregator


def _sample_bar(ts: datetime) -> dict:
    return {
        "timestamp": ts,
        "open": 10.0,
        "high": 10.2,
        "low": 9.9,
        "close": 10.1,
        "volume": 100,
    }


def test_emit_bar_attaches_canonical_interval_for_one_day() -> None:
    agg = BarAggregator(interval_seconds=86400)
    seen: list[dict] = []
    agg.add_callback(lambda _sym, bar: seen.append(dict(bar)))

    agg._emit_bar("000001", _sample_bar(datetime(2026, 2, 16, 10, 0, 0)), final=False)

    assert seen
    out = seen[-1]
    assert out["interval"] == "1d"
    assert int(out["interval_seconds"]) == 86400
    assert out["final"] is False


def test_emit_bar_attaches_canonical_interval_for_sixty_minutes() -> None:
    agg = BarAggregator(interval_seconds=3600)
    seen: list[dict] = []
    agg.add_callback(lambda _sym, bar: seen.append(dict(bar)))

    agg._emit_bar("000001", _sample_bar(datetime(2026, 2, 16, 10, 0, 0)), final=False)

    assert seen
    out = seen[-1]
    assert out["interval"] == "60m"
    assert int(out["interval_seconds"]) == 3600


def test_emit_bar_market_open_writes_session_only(monkeypatch) -> None:
    agg = BarAggregator(interval_seconds=60)

    class _Cache:
        def __init__(self) -> None:
            self.calls = 0

        def append_bar(self, symbol, interval, bar) -> bool:  # noqa: ARG002
            self.calls += 1
            return True

    class _DB:
        def __init__(self) -> None:
            self.calls = 0

        def upsert_intraday_bars(self, symbol, interval, df) -> None:  # noqa: ARG002
            self.calls += 1

    cache = _Cache()
    db = _DB()

    monkeypatch.setattr("data.feeds.CONFIG.is_market_open", lambda: True)
    monkeypatch.setattr("data.session_cache.get_session_bar_cache", lambda: cache)
    monkeypatch.setattr("data.database.get_database", lambda: db)

    agg._emit_bar("000001", _sample_bar(datetime(2026, 2, 16, 10, 0, 0)), final=True)

    assert cache.calls == 1
    assert db.calls == 0


def test_emit_bar_market_closed_persists_to_db(monkeypatch) -> None:
    agg = BarAggregator(interval_seconds=60)

    class _Cache:
        def __init__(self) -> None:
            self.calls = 0

        def append_bar(self, symbol, interval, bar) -> bool:  # noqa: ARG002
            self.calls += 1
            return True

    class _DB:
        def __init__(self) -> None:
            self.calls = 0

        def upsert_intraday_bars(self, symbol, interval, df) -> None:  # noqa: ARG002
            self.calls += 1

    cache = _Cache()
    db = _DB()

    monkeypatch.setattr("data.feeds.CONFIG.is_market_open", lambda: False)
    monkeypatch.setattr("data.session_cache.get_session_bar_cache", lambda: cache)
    monkeypatch.setattr("data.database.get_database", lambda: db)

    agg._emit_bar("000001", _sample_bar(datetime(2026, 2, 16, 15, 1, 0)), final=True)

    assert cache.calls == 1
    assert db.calls == 1
