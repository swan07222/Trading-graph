from __future__ import annotations

from data.discovery import DiscoveredStock, UniversalStockDiscovery
from data.fetcher import TencentQuoteSource


def test_tencent_daily_kline_parser():
    payload = (
        '{"code":0,"msg":"","data":{"sh600519":{"qfqday":['
        '["2026-02-10","1000","1010","1015","995","12345"],'
        '["2026-02-11","1010","1020","1030","1005","15000"]'
        "]}}}"
    )
    df = TencentQuoteSource._parse_daily_kline(payload, "sh600519")
    assert not df.empty
    assert len(df) == 2
    assert float(df["close"].iloc[-1]) == 1020.0


def test_tencent_daily_kline_parser_jsonp_wrapper():
    payload = (
        'jQuery12345({"code":0,"msg":"","data":{"sz000858":{"qfqday":['
        '["2026-02-11","1200","1210","1220","1190","20000"]'
        "]}}});"
    )
    df = TencentQuoteSource._parse_daily_kline(payload, "sz000858")
    assert not df.empty
    assert len(df) == 1
    assert float(df["close"].iloc[0]) == 1210.0


def test_discovery_tencent_semicolon_response(monkeypatch):
    class _Resp:
        status_code = 200
        text = (
            'v_sh600519="51~A~600519~1500~1490~0~0~0~0~0";'
            'v_sz000858="51~B~000858~1200~1190~0~0~0~0~0";'
        )

    class _RequestsStub:
        @staticmethod
        def get(url, timeout=10):
            return _Resp()

    monkeypatch.setattr("data.discovery.requests", _RequestsStub)

    monkeypatch.setattr(
        UniversalStockDiscovery,
        "_get_fallback_stocks",
        staticmethod(
            lambda: [
                DiscoveredStock(code="600519", name="A", source="fallback"),
                DiscoveredStock(code="000858", name="B", source="fallback"),
            ]
        ),
    )

    d = UniversalStockDiscovery()
    out = d._discover_via_tencent()
    codes = sorted(s.code for s in out)
    assert codes == ["000858", "600519"]
