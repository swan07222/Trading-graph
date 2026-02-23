from __future__ import annotations

from data.discovery import DiscoveredStock, UniversalStockDiscovery
from data.fetcher import TencentQuoteSource


def test_tencent_daily_kline_parser() -> None:
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


def test_tencent_daily_kline_parser_jsonp_wrapper() -> None:
    payload = (
        'jQuery12345({"code":0,"msg":"","data":{"sz000858":{"qfqday":['
        '["2026-02-11","1200","1210","1220","1190","20000"]'
        "]}}});"
    )
    df = TencentQuoteSource._parse_daily_kline(payload, "sz000858")
    assert not df.empty
    assert len(df) == 1
    assert float(df["close"].iloc[0]) == 1210.0


def test_discovery_tencent_semicolon_response(monkeypatch) -> None:
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


def test_tencent_realtime_bse_30pct_move_not_filtered() -> None:
    src = TencentQuoteSource()

    parts = ["0"] * 40
    parts[1] = "BSE_SAMPLE"
    parts[3] = "13.00"   # latest
    parts[4] = "10.00"   # prev close (+30%)
    parts[5] = "10.10"   # open
    parts[6] = "2.00"    # lots
    parts[9] = "12.95"   # bid
    parts[19] = "13.05"  # ask
    parts[33] = "13.20"  # high
    parts[34] = "10.00"  # low
    parts[37] = "26000"  # amount
    payload = "~".join(parts)

    class _Resp:
        status_code = 200
        text = f'v_bj430001="{payload}";\\n'

    src._session.get = lambda *args, **kwargs: _Resp()  # type: ignore[method-assign]

    out = src.get_realtime_batch(["430001"])
    assert "430001" in out
    assert float(out["430001"].price) == 13.0


def test_tencent_realtime_batch_parses_semicolon_single_line_payload() -> None:
    src = TencentQuoteSource()

    def _make_payload(name: str, price: float, prev_close: float) -> str:
        parts = ["0"] * 40
        parts[1] = name
        parts[3] = f"{price:.2f}"
        parts[4] = f"{prev_close:.2f}"
        parts[5] = f"{prev_close:.2f}"
        parts[6] = "1.00"
        parts[9] = f"{price:.2f}"
        parts[19] = f"{price:.2f}"
        parts[33] = f"{max(price, prev_close):.2f}"
        parts[34] = f"{min(price, prev_close):.2f}"
        parts[37] = "10000"
        return "~".join(parts)

    p1 = _make_payload("A", 1500.0, 1490.0)
    p2 = _make_payload("B", 12.0, 11.8)

    class _Resp:
        status_code = 200
        text = f'v_sh600519="{p1}";v_sz000001="{p2}";'

    src._session.get = lambda *args, **kwargs: _Resp()  # type: ignore[method-assign]

    out = src.get_realtime_batch(["600519", "000001"])
    assert "600519" in out
    assert "000001" in out
