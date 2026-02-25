import time
from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest

from config.settings import CONFIG
from core.types import AutoTradeMode
from data.news import NewsAggregator, NewsItem

try:
    from trading.executor import AutoTrader

    _EXECUTION_STACK_AVAILABLE = True
except ImportError:
    _EXECUTION_STACK_AVAILABLE = False
    AutoTrader = None  # type: ignore[assignment]

pytestmark = pytest.mark.skipif(
    not _EXECUTION_STACK_AVAILABLE,
    reason="Execution stack modules are removed in analysis-only build.",
)


class _DummyEngine:
    pass


def test_auto_trader_get_state_is_snapshot_copy() -> None:
    at = AutoTrader(engine=_DummyEngine(), predictor=None, watch_list=[])
    at.state.mode = AutoTradeMode.AUTO
    at.state.trades_today = 5

    snap = at.get_state()
    snap.trades_today = 99

    assert at.state.trades_today == 5
    assert snap.mode == AutoTradeMode.AUTO


def test_auto_trader_precision_quality_gate() -> None:
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


def test_auto_trader_precision_quality_gate_blocks_short_history_fallback() -> None:
    at = AutoTrader(engine=_DummyEngine(), predictor=None, watch_list=[])

    old_block = bool(
        getattr(CONFIG.precision, "block_auto_trade_on_short_history_fallback", True)
    )
    try:
        CONFIG.precision.block_auto_trade_on_short_history_fallback = True
        pred = SimpleNamespace(
            short_history_fallback=True,
            entropy=0.01,
            prob_up=0.70,
            prob_down=0.10,
        )
        ok, reason = at._passes_precision_quality_gate(pred)
        assert ok is False
        assert "short-history" in reason.lower()
    finally:
        CONFIG.precision.block_auto_trade_on_short_history_fallback = old_block


def test_auto_trader_precision_quality_gate_fail_closed_on_error() -> None:
    at = AutoTrader(engine=_DummyEngine(), predictor=None, watch_list=[])

    old_fail_closed = bool(
        getattr(CONFIG.precision, "fail_closed_on_quality_gate_error", True)
    )
    old_max_entropy = float(getattr(CONFIG.precision, "max_entropy", 0.35))
    try:
        CONFIG.precision.fail_closed_on_quality_gate_error = True
        CONFIG.precision.max_entropy = "bad_value"  # type: ignore[assignment]
        pred = SimpleNamespace(
            short_history_fallback=False,
            entropy=0.05,
            prob_up=0.70,
            prob_down=0.10,
        )
        ok, reason = at._passes_precision_quality_gate(pred)
        assert ok is False
        assert "fail-closed" in reason.lower()
    finally:
        CONFIG.precision.fail_closed_on_quality_gate_error = old_fail_closed
        CONFIG.precision.max_entropy = old_max_entropy


def test_auto_trader_caps_buy_quantity_with_risk_manager() -> None:
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


def test_auto_trader_stop_does_not_block_on_scan_lock() -> None:
    old_enabled = bool(CONFIG.auto_trade.enabled)
    old_scan_interval = int(CONFIG.auto_trade.scan_interval_seconds)
    try:
        # Force the scan loop into the no-scan branch.
        CONFIG.auto_trade.enabled = False
        CONFIG.auto_trade.scan_interval_seconds = 60

        at = AutoTrader(engine=_DummyEngine(), predictor=None, watch_list=["600519"])
        at.state.mode = AutoTradeMode.AUTO
        at.start()

        deadline = time.time() + 1.0
        while (not at._is_loop_running()) and time.time() < deadline:
            time.sleep(0.01)
        time.sleep(0.05)

        t0 = time.perf_counter()
        at.stop()
        elapsed = time.perf_counter() - t0
        assert elapsed < 1.5
    finally:
        CONFIG.auto_trade.enabled = old_enabled
        CONFIG.auto_trade.scan_interval_seconds = old_scan_interval


def test_news_aggregator_stale_cache_fallback_and_source_health(monkeypatch) -> None:
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


def test_news_aggregator_institutional_snapshot_shape(monkeypatch) -> None:
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


def test_news_aggregator_stock_fallback_from_market_pool(monkeypatch) -> None:
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


def test_news_aggregator_stock_news_prefers_direct_over_context(monkeypatch) -> None:
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


def test_news_aggregator_stock_news_does_not_mutate_shared_market_items(monkeypatch) -> None:
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
