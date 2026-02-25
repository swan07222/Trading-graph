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
