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


def test_emit_bar_attaches_canonical_interval_for_one_day():
    agg = BarAggregator(interval_seconds=86400)
    seen: list[dict] = []
    agg.add_callback(lambda _sym, bar: seen.append(dict(bar)))

    agg._emit_bar("000001", _sample_bar(datetime(2026, 2, 16, 10, 0, 0)), final=False)

    assert seen
    out = seen[-1]
    assert out["interval"] == "1d"
    assert int(out["interval_seconds"]) == 86400
    assert out["final"] is False


def test_emit_bar_attaches_canonical_interval_for_sixty_minutes():
    agg = BarAggregator(interval_seconds=3600)
    seen: list[dict] = []
    agg.add_callback(lambda _sym, bar: seen.append(dict(bar)))

    agg._emit_bar("000001", _sample_bar(datetime(2026, 2, 16, 10, 0, 0)), final=False)

    assert seen
    out = seen[-1]
    assert out["interval"] == "60m"
    assert int(out["interval_seconds"]) == 3600
