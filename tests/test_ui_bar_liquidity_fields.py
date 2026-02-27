from __future__ import annotations

from ui import app_bar_ops as _app_bar_ops
from ui import app_feed_ops as _app_feed_ops
from ui.app_common import MainAppCommonMixin


class _Combo:
    def __init__(self, value: str) -> None:
        self._value = str(value)

    def currentText(self) -> str:
        return self._value


class _LineEdit:
    def __init__(self, value: str) -> None:
        self._value = str(value)

    def text(self) -> str:
        return self._value


class _Spin:
    @staticmethod
    def value() -> int:
        return 30


class _DummyBarApp:
    _on_bar_ui = _app_feed_ops._on_bar_ui
    _ui_norm = MainAppCommonMixin._ui_norm
    _normalize_interval_token = MainAppCommonMixin._normalize_interval_token
    _interval_token_from_seconds = _app_bar_ops._interval_token_from_seconds
    _interval_seconds = _app_bar_ops._interval_seconds
    _bar_bucket_epoch = _app_bar_ops._bar_bucket_epoch
    _ts_to_epoch = _app_bar_ops._ts_to_epoch
    _epoch_to_iso = _app_bar_ops._epoch_to_iso
    _bar_safety_caps = _app_bar_ops._bar_safety_caps
    _sanitize_ohlc = _app_bar_ops._sanitize_ohlc
    _is_intraday_day_boundary = _app_bar_ops._is_intraday_day_boundary
    _bar_trading_date = _app_bar_ops._bar_trading_date
    _is_outlier_tick = _app_bar_ops._is_outlier_tick

    def __init__(self, ui_interval: str = "1m") -> None:
        self._bars_by_symbol: dict[str, list[dict]] = {}
        self._last_bar_feed_ts: dict[str, float] = {}
        self._last_session_cache_write_ts: dict[str, float] = {}
        self._session_bar_cache = None
        self.interval_combo = _Combo(ui_interval)
        self.stock_input = _LineEdit("600519")
        self.forecast_spin = _Spin()

    @staticmethod
    def _is_market_session_timestamp(_ts_raw, _interval: str) -> bool:
        return True

    @staticmethod
    def _history_window_bars(_interval: str) -> int:
        return 400

    @staticmethod
    def _debug_console(*_args, **_kwargs) -> None:
        return

    @staticmethod
    def _resolve_chart_prediction_series(*_args, **_kwargs):
        return [], "1m"

    @staticmethod
    def _render_chart_state(**kwargs):
        return list(kwargs.get("bars") or [])


def test_on_bar_ui_derives_amount_from_volume_and_close() -> None:
    app = _DummyBarApp(ui_interval="1m")

    app._on_bar_ui(
        "600519",
        {
            "timestamp": "2026-02-18T09:31:00+08:00",
            "open": 10.0,
            "high": 10.3,
            "low": 9.9,
            "close": 10.2,
            "volume": 1200,
            "interval": "1m",
            "final": True,
        },
    )

    bars = app._bars_by_symbol.get("600519") or []
    assert len(bars) == 1
    row = bars[-1]
    assert abs(float(row.get("volume", 0.0)) - 1200.0) < 1e-9
    assert abs(float(row.get("amount", 0.0)) - 12240.0) < 1e-6


def test_on_bar_ui_aggregates_amount_when_resampling_to_ui_interval() -> None:
    app = _DummyBarApp(ui_interval="5m")

    app._on_bar_ui(
        "600519",
        {
            "timestamp": "2026-02-18T09:31:00+08:00",
            "open": 10.0,
            "high": 10.1,
            "low": 9.9,
            "close": 10.0,
            "volume": 100.0,
            "amount": 1000.0,
            "interval": "1m",
            "final": True,
        },
    )
    app._on_bar_ui(
        "600519",
        {
            "timestamp": "2026-02-18T09:32:00+08:00",
            "open": 10.0,
            "high": 10.2,
            "low": 9.95,
            "close": 10.1,
            "volume": 60.0,
            "amount": 606.0,
            "interval": "1m",
            "final": True,
        },
    )

    bars = app._bars_by_symbol.get("600519") or []
    assert len(bars) == 1
    row = bars[0]
    assert abs(float(row.get("volume", 0.0)) - 160.0) < 1e-9
    assert abs(float(row.get("amount", 0.0)) - 1606.0) < 1e-9


def test_on_bar_ui_does_not_double_count_same_source_minute_updates() -> None:
    app = _DummyBarApp(ui_interval="5m")

    # First partial update for source minute 09:31.
    app._on_bar_ui(
        "600519",
        {
            "timestamp": "2026-02-18T09:31:00+08:00",
            "open": 10.0,
            "high": 10.1,
            "low": 9.9,
            "close": 10.0,
            "volume": 100.0,
            "amount": 1000.0,
            "interval": "1m",
            "final": False,
        },
    )
    # Same source minute arrives again with larger finalized totals.
    app._on_bar_ui(
        "600519",
        {
            "timestamp": "2026-02-18T09:31:00+08:00",
            "open": 10.0,
            "high": 10.2,
            "low": 9.9,
            "close": 10.1,
            "volume": 120.0,
            "amount": 1212.0,
            "interval": "1m",
            "final": True,
        },
    )
    # Lower duplicate update for the same source minute must not decrease or add again.
    app._on_bar_ui(
        "600519",
        {
            "timestamp": "2026-02-18T09:31:00+08:00",
            "open": 10.0,
            "high": 10.15,
            "low": 9.95,
            "close": 10.05,
            "volume": 110.0,
            "amount": 1105.5,
            "interval": "1m",
            "final": True,
        },
    )
    # Next source minute in the same 5m bucket should add on top.
    app._on_bar_ui(
        "600519",
        {
            "timestamp": "2026-02-18T09:32:00+08:00",
            "open": 10.1,
            "high": 10.2,
            "low": 10.0,
            "close": 10.1,
            "volume": 60.0,
            "amount": 606.0,
            "interval": "1m",
            "final": True,
        },
    )

    bars = app._bars_by_symbol.get("600519") or []
    assert len(bars) == 1
    row = bars[0]
    # same-source duplicate minute should contribute max(100,120,110) = 120 once
    assert abs(float(row.get("volume", 0.0)) - 180.0) < 1e-9
    assert abs(float(row.get("amount", 0.0)) - 1818.0) < 1e-9


class _DummyScrubApp:
    _scrub_chart_bars = _app_bar_ops._scrub_chart_bars
    _ui_norm = MainAppCommonMixin._ui_norm

    @staticmethod
    def _normalize_interval_token(value: str, fallback: str = "1m") -> str:
        v = str(value or fallback).strip().lower()
        return v or str(fallback)

    @staticmethod
    def _prepare_chart_bars_for_interval(
        bars: list[dict], _interval: str, *, symbol: str = ""
    ) -> list[dict]:
        _ = symbol
        return list(bars or [])

    @staticmethod
    def _rescale_chart_bars_to_anchor(
        bars: list[dict], *, anchor_price=None, interval: str, symbol: str = ""
    ) -> list[dict]:
        _ = (anchor_price, interval, symbol)
        return list(bars or [])

    @staticmethod
    def _bar_safety_caps(_interval: str) -> tuple[float, float]:
        return (0.08, 0.006)

    @staticmethod
    def _recover_chart_bars_from_close(
        bars: list[dict], *, interval: str, symbol: str = "", anchor_price=None
    ) -> list[dict]:
        _ = (bars, interval, symbol, anchor_price)
        return [
            {
                "open": 10.0,
                "high": 10.1,
                "low": 9.9,
                "close": 10.0,
                "interval": "1m",
                "timestamp": "2026-02-20T09:31:00+08:00",
            }
        ]

    @staticmethod
    def _debug_console(*_args, **_kwargs) -> None:
        return


def test_scrub_chart_bars_uses_recovery_when_extreme_ratio_high() -> None:
    app = _DummyScrubApp()
    bars: list[dict] = []
    for i in range(30):
        close = 10.0 + (0.01 * i)
        if i % 6 == 0:
            # Inject an extreme malformed bar.
            bars.append(
                {
                    "open": close,
                    "high": close * 2.0,
                    "low": close * 0.2,
                    "close": close,
                    "interval": "1m",
                }
            )
        else:
            bars.append(
                {
                    "open": close * 0.999,
                    "high": close * 1.002,
                    "low": close * 0.998,
                    "close": close,
                    "interval": "1m",
                }
            )

    out = app._scrub_chart_bars(bars, "1m", symbol="600519", anchor_price=10.0)

    assert len(out) == 1
    assert abs(float(out[0]["close"]) - 10.0) < 1e-9


def test_scrub_chart_bars_recovers_when_prepare_returns_empty() -> None:
    class _EmptyPrepApp(_DummyScrubApp):
        @staticmethod
        def _prepare_chart_bars_for_interval(
            bars: list[dict], _interval: str, *, symbol: str = ""
        ) -> list[dict]:
            _ = (bars, symbol)
            return []

    app = _EmptyPrepApp()
    raw = [
        {
            "open": 10.0,
            "high": 10.2,
            "low": 9.8,
            "close": 10.1,
            "interval": "1m",
        }
    ]
    out = app._scrub_chart_bars(raw, "1m", symbol="600519", anchor_price=10.0)

    assert len(out) == 1
    assert abs(float(out[0]["close"]) - 10.0) < 1e-9


def test_sanitize_ohlc_intraday_compacts_extreme_vendor_wicks() -> None:
    app = _DummyBarApp(ui_interval="1m")

    out = app._sanitize_ohlc(
        229.800,
        300.000,
        120.000,
        229.420,
        interval="1m",
        ref_close=229.700,
    )

    assert out is not None
    o, h, low, c = out
    anchor = 229.700
    span_pct = abs(float(h) - float(low)) / anchor
    body_pct = abs(float(o) - float(c)) / anchor
    # Prevent barcode-like candles after sanitization.
    assert span_pct < 0.005
    assert span_pct < (body_pct + 0.004)


def test_sanitize_ohlc_5m_compacts_thin_spike_wicks() -> None:
    app = _DummyBarApp(ui_interval="5m")

    out = app._sanitize_ohlc(
        229.95,
        250.00,
        200.00,
        229.90,
        interval="5m",
        ref_close=229.92,
    )

    assert out is not None
    o, h, low, c = out
    anchor = 229.92
    span_pct = abs(float(h) - float(low)) / anchor
    body_pct = abs(float(o) - float(c)) / anchor
    # 5m bars may be wider than 1m, but should still avoid tall spike bars.
    assert span_pct < 0.008
    assert span_pct < (body_pct + 0.0055)
