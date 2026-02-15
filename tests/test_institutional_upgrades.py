from datetime import datetime, timedelta
from types import SimpleNamespace

from config.settings import CONFIG
from core.types import AutoTradeMode
from data.news import NewsAggregator, NewsItem
from trading.executor import AutoTrader


class _DummyEngine:
    pass


def test_auto_trader_get_state_is_snapshot_copy():
    at = AutoTrader(engine=_DummyEngine(), predictor=None, watch_list=[])
    at.state.mode = AutoTradeMode.AUTO
    at.state.trades_today = 5

    snap = at.get_state()
    snap.trades_today = 99

    assert at.state.trades_today == 5
    assert snap.mode == AutoTradeMode.AUTO


def test_auto_trader_precision_quality_gate():
    at = AutoTrader(engine=_DummyEngine(), predictor=None, watch_list=[])

    pred_bad_entropy = SimpleNamespace(entropy=0.99, prob_up=0.7, prob_down=0.1)
    ok, reason = at._passes_precision_quality_gate(pred_bad_entropy)
    assert ok is False
    assert "entropy" in reason.lower()

    old_min_edge = float(getattr(CONFIG.precision, "min_edge", 0.14))
    try:
        CONFIG.precision.min_edge = 0.30
        pred_bad_edge = SimpleNamespace(entropy=0.05, prob_up=0.53, prob_down=0.49)
        ok2, reason2 = at._passes_precision_quality_gate(pred_bad_edge)
        assert ok2 is False
        assert "edge" in reason2.lower()
    finally:
        CONFIG.precision.min_edge = old_min_edge


def test_news_aggregator_stale_cache_fallback_and_source_health(monkeypatch):
    agg = NewsAggregator()
    stale_item = NewsItem(
        title="stale cached headline",
        source="cache",
        publish_time=datetime.now() - timedelta(minutes=15),
        category="market",
    )
    agg._cache["market_1"] = [stale_item]
    agg._cache_time["market_1"] = 0.0  # force stale

    class _Env:
        tencent_ok = True
        is_china_direct = False
        eastmoney_ok = False

    monkeypatch.setattr("core.network.get_network_env", lambda: _Env())
    monkeypatch.setattr(
        agg._tencent,
        "fetch_market_news",
        lambda count: (_ for _ in ()).throw(RuntimeError("simulated provider outage")),
    )

    out = agg.get_market_news(count=1, force_refresh=True)
    assert len(out) == 1
    assert out[0].title == "stale cached headline"

    health = agg.get_source_health()
    assert "tencent" in health
    assert health["tencent"]["failed_calls"] >= 1
    assert health["tencent"]["success_rate"] < 1.0


def test_news_aggregator_institutional_snapshot_shape(monkeypatch):
    agg = NewsAggregator()

    class _Env:
        tencent_ok = True
        is_china_direct = False
        eastmoney_ok = False

    now = datetime.now()
    items = [
        NewsItem(
            title="alpha",
            source="tencent",
            publish_time=now - timedelta(minutes=2),
            category="market",
        ),
        NewsItem(
            title="beta",
            source="tencent",
            publish_time=now - timedelta(minutes=8),
            category="market",
        ),
    ]

    monkeypatch.setattr("core.network.get_network_env", lambda: _Env())
    monkeypatch.setattr(agg._tencent, "fetch_market_news", lambda count: items[:count])

    snap = agg.get_institutional_snapshot(stock_code=None, hours_lookback=24)
    assert snap["scope"] == "market"
    assert isinstance(snap["source_mix"], dict)
    assert "freshness" in snap
    assert "sentiment" in snap
    assert "features" in snap
    assert "source_health" in snap


def test_news_aggregator_stock_fallback_from_market_pool(monkeypatch):
    agg = NewsAggregator()

    class _Env:
        tencent_ok = False
        is_china_direct = False
        eastmoney_ok = False

    now = datetime.now()
    market_items = [
        NewsItem(
            title="600519 gains after policy support",
            source="cache",
            publish_time=now - timedelta(minutes=3),
            category="market",
        ),
        NewsItem(
            title="macro headline",
            source="cache",
            publish_time=now - timedelta(minutes=6),
            category="market",
        ),
    ]

    monkeypatch.setattr("core.network.get_network_env", lambda: _Env())
    monkeypatch.setattr(agg, "get_market_news", lambda *a, **k: market_items)
    monkeypatch.setattr(agg._sina, "fetch_stock_news", lambda *a, **k: [])
    monkeypatch.setattr(agg._eastmoney, "fetch_stock_news", lambda *a, **k: [])

    out = agg.get_stock_news("600519", count=5, force_refresh=True)
    assert len(out) == 1
    assert "600519" in out[0].title
