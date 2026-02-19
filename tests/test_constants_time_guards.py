import builtins
from datetime import date, datetime, timedelta, timezone

from core.constants import get_price_limit, is_st_stock, is_trading_day
from data.fetcher import DataFetcher


def test_is_st_stock_handles_cn_prefix_forms():
    cn_st = "ST\u4e2d\u73e0"
    cn_star_st = "*ST\u957f\u836f"
    assert is_st_stock(cn_st) is True
    assert is_st_stock(cn_star_st) is True
    assert is_st_stock("st abc") is True
    assert is_st_stock("BEST") is False
    assert is_st_stock("FASTEST") is False


def test_get_price_limit_uses_st_band_for_cn_st_names():
    cn_st = "ST\u4e2d\u73e0"
    cn_star_st = "*ST\u957f\u836f"
    assert float(get_price_limit("600000", cn_st)) == 0.05
    assert float(get_price_limit("600000", cn_star_st)) == 0.05


def test_is_trading_day_marks_2026_new_year_closed():
    # 2026-01-01 should not be considered a trading day.
    assert is_trading_day(date(2026, 1, 1)) is False


def test_fetcher_now_shanghai_fallback_keeps_utc_plus_8(monkeypatch):
    original_import = builtins.__import__

    def _fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "zoneinfo":
            raise ImportError("simulated missing zoneinfo")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _fake_import)

    got = DataFetcher._now_shanghai_naive()
    expected = datetime.now(tz=timezone(timedelta(hours=8))).replace(tzinfo=None)
    assert abs((got - expected).total_seconds()) < 2.0
