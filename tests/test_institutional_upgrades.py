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


def test_auto_trader_caps_buy_quantity_with_risk_manager():
    from models.predictor import Signal

    quick_pred = SimpleNamespace(
        stock_code="600519",
        signal=Signal.STRONG_BUY,
        confidence=0.92,
        signal_strength=0.88,
        model_agreement=0.90,
    )
    full_pred = SimpleNamespace(
        stock_code="600519",
        stock_name="Kweichow Moutai",
        signal=Signal.STRONG_BUY,
        confidence=0.93,
        signal_strength=0.90,
        model_agreement=0.91,
        current_price=100.0,
        position=SimpleNamespace(shares=1000),
        levels=SimpleNamespace(stop_loss=97.0),
        atr_pct_value=0.02,
        entropy=0.08,
        prob_up=0.82,
        prob_down=0.10,
    )

    class _Predictor:
        ensemble = object()

        def predict_quick_batch(self, *args, **kwargs):  # noqa: ARG002
            return [quick_pred]

        def predict(self, *args, **kwargs):  # noqa: ARG002
            return full_pred

    account = SimpleNamespace(positions={}, equity=1_000_000.0)
    engine = SimpleNamespace(
        get_account=lambda: account,
        risk_manager=SimpleNamespace(
            calculate_position_size=lambda **_k: 200
        ),
    )

    at = AutoTrader(engine=engine, predictor=_Predictor(), watch_list=["600519"])
    at.state.mode = AutoTradeMode.SEMI_AUTO

    at._run_scan_cycle()

    assert len(at.state.pending_approvals) == 1
    pending = at.state.pending_approvals[0]
    assert pending.stock_code == "600519"
    assert pending.quantity == 200


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
        tencent_ok = False
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
        tencent_ok = True
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


def test_news_aggregator_stock_news_prefers_direct_over_context(monkeypatch):
    agg = NewsAggregator()

    class _Env:
        tencent_ok = True
        is_china_direct = False
        eastmoney_ok = False

    now = datetime.now()
    direct_item = NewsItem(
        title="600519 earnings beat expectations",
        source="sina",
        publish_time=now - timedelta(minutes=2),
        category="company",
    )
    context_item = NewsItem(
        title="macro headline",
        source="cache",
        publish_time=now - timedelta(minutes=1),
        category="market",
    )

    monkeypatch.setattr("core.network.get_network_env", lambda: _Env())
    monkeypatch.setattr(agg._sina, "fetch_stock_news", lambda *a, **k: [direct_item])
    monkeypatch.setattr(agg._eastmoney, "fetch_stock_news", lambda *a, **k: [])
    monkeypatch.setattr(agg, "get_market_news", lambda *a, **k: [context_item])

    out = agg.get_stock_news("600519", count=1, force_refresh=True)
    assert len(out) == 1
    assert "600519" in out[0].title


def test_news_aggregator_stock_news_does_not_mutate_shared_market_items(monkeypatch):
    agg = NewsAggregator()

    class _Env:
        tencent_ok = False
        is_china_direct = False
        eastmoney_ok = False

    now = datetime.now()
    shared_market_item = NewsItem(
        title="macro headline",
        source="cache",
        publish_time=now - timedelta(minutes=4),
        category="market",
    )
    market_items = [shared_market_item]

    monkeypatch.setattr("core.network.get_network_env", lambda: _Env())
    monkeypatch.setattr(agg._sina, "fetch_stock_news", lambda *a, **k: [])
    monkeypatch.setattr(agg._eastmoney, "fetch_stock_news", lambda *a, **k: [])
    monkeypatch.setattr(agg, "get_market_news", lambda *a, **k: market_items)

    out_a = agg.get_stock_news("600519", count=1, force_refresh=True)
    out_b = agg.get_stock_news("000001", count=1, force_refresh=True)

    assert out_a and "600519" in out_a[0].stock_codes
    assert out_b and "000001" in out_b[0].stock_codes
    assert shared_market_item.stock_codes == []
