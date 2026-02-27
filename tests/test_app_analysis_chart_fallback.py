from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace

from ui import app_analysis_ops


class _ActionStub:
    def __init__(self) -> None:
        self.enabled_calls: list[bool] = []

    def setEnabled(self, value: bool) -> None:  # noqa: N802
        self.enabled_calls.append(bool(value))


class _ProgressStub:
    def __init__(self) -> None:
        self.hide_calls = 0

    def hide(self) -> None:
        self.hide_calls += 1


class _StatusStub:
    def __init__(self) -> None:
        self.values: list[str] = []

    def setText(self, value: str) -> None:  # noqa: N802
        self.values.append(str(value))


class _ChartStub:
    def __init__(self) -> None:
        self.calls: list[tuple[list[float], dict[str, object]]] = []

    def update_data(
        self,
        actual_prices: list[float],
        predicted_prices: list[float] | None = None,
        predicted_prices_low: list[float] | None = None,
        predicted_prices_high: list[float] | None = None,
        levels: dict[str, float] | None = None,
    ) -> None:
        self.calls.append(
            (
                list(actual_prices or []),
                {
                    "predicted_prices": list(predicted_prices or []),
                    "predicted_prices_low": list(predicted_prices_low or []),
                    "predicted_prices_high": list(predicted_prices_high or []),
                    "levels": levels,
                },
            )
        )


class _ChartUpdateChartStub:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def update_chart(
        self,
        bars: list[dict[str, object]],
        predicted_prices: list[float] | None = None,
        predicted_prices_low: list[float] | None = None,
        predicted_prices_high: list[float] | None = None,
        levels: dict[str, float] | None = None,
    ) -> None:
        self.calls.append(
            {
                "bars": list(bars or []),
                "predicted_prices": list(predicted_prices or []),
                "predicted_prices_low": list(predicted_prices_low or []),
                "predicted_prices_high": list(predicted_prices_high or []),
                "levels": levels,
            }
        )


def test_on_analysis_done_uses_price_history_fallback_chart_when_history_empty() -> None:
    chart = _ChartStub()
    latest_calls: list[tuple[str, float | None]] = []
    render_calls: list[dict[str, object]] = []
    logs: list[tuple[str, str]] = []

    stub = SimpleNamespace(
        _analyze_request_seq=7,
        workers={"analyze": SimpleNamespace(_request_seq=7)},
        _ui_norm=lambda x: str(x or "").strip(),
        _debug_console=lambda *_a, **_k: None,
        stock_input=SimpleNamespace(text=lambda: "001325"),
        analyze_action=_ActionStub(),
        progress=_ProgressStub(),
        status_label=_StatusStub(),
        _startup_loading_active=False,
        signal_panel=SimpleNamespace(update_prediction=lambda _pred: None),
        interval_combo=SimpleNamespace(currentText=lambda: "1m"),
        lookback_spin=SimpleNamespace(value=lambda: 480),
        forecast_spin=SimpleNamespace(value=lambda: 60),
        _seven_day_lookback=lambda _iv: 480,
        _load_chart_history_bars=lambda *_a, **_k: [],
        _filter_bars_to_market_session=lambda rows, _iv: list(rows or []),
        _render_chart_state=lambda **kwargs: render_calls.append(dict(kwargs)),
        chart=chart,
        _prepare_chart_predicted_prices=lambda **_kwargs: [53.05, 53.10],
        _build_chart_prediction_bands=lambda **_kwargs: ([52.95, 53.00], [53.15, 53.20]),
        _get_levels_dict=lambda: None,
        _update_chart_latest_label=lambda symbol, bar=None, price=None: latest_calls.append(
            (str(symbol), float(price) if isinstance(price, (int, float)) else None)
        ),
        _normalize_interval_token=lambda iv, fallback="1m": str(iv or fallback or "1m").strip().lower(),
        _bars_by_symbol={},
        _update_details=lambda _pred: None,
        _add_to_history=lambda _pred: None,
        _ensure_feed_subscription=lambda _code: None,
        monitor=None,
        monitor_action=SimpleNamespace(setChecked=lambda _checked: None),
        _start_monitoring=lambda: None,
        log=lambda msg, level="info": logs.append((str(msg), str(level))),
        _last_analysis_log={},
    )

    pred = SimpleNamespace(
        stock_code="001325",
        stock_name="Test",
        timestamp=datetime.now(),
        signal=SimpleNamespace(value="HOLD"),
        confidence=0.38,
        prob_up=0.30,
        prob_neutral=0.40,
        prob_down=0.30,
        current_price=52.99,
        price_history=[52.80, 52.88, 52.99],
        predicted_prices=[53.00, 53.05, 53.10],
        interval="1m",
        warnings=[],
        levels=SimpleNamespace(stop_loss=0.0, target_1=0.0, target_2=0.0, target_3=0.0),
    )

    app_analysis_ops._on_analysis_done(stub, pred, request_seq=7)

    assert render_calls == []
    assert len(chart.calls) == 1
    actual_prices, payload = chart.calls[0]
    assert actual_prices == [52.80, 52.88, 52.99]
    assert payload["predicted_prices"] == [53.05, 53.10]
    assert payload["predicted_prices_low"] == [52.95, 53.00]
    assert payload["predicted_prices_high"] == [53.15, 53.20]
    assert latest_calls
    assert latest_calls[-1][0] == "001325"
    assert latest_calls[-1][1] == 52.99
    assert "001325" in stub._bars_by_symbol
    assert len(stub._bars_by_symbol["001325"]) == 3
    assert "analyze" not in stub.workers
    assert logs


def test_on_analysis_done_fallback_prefers_update_chart_with_synthetic_bars() -> None:
    chart = _ChartUpdateChartStub()
    latest_calls: list[tuple[str, float | None]] = []

    stub = SimpleNamespace(
        _analyze_request_seq=8,
        workers={"analyze": SimpleNamespace(_request_seq=8)},
        _ui_norm=lambda x: str(x or "").strip(),
        _debug_console=lambda *_a, **_k: None,
        stock_input=SimpleNamespace(text=lambda: "001325"),
        analyze_action=_ActionStub(),
        progress=_ProgressStub(),
        status_label=_StatusStub(),
        _startup_loading_active=False,
        signal_panel=SimpleNamespace(update_prediction=lambda _pred: None),
        interval_combo=SimpleNamespace(currentText=lambda: "1m"),
        lookback_spin=SimpleNamespace(value=lambda: 480),
        forecast_spin=SimpleNamespace(value=lambda: 60),
        _seven_day_lookback=lambda _iv: 480,
        _load_chart_history_bars=lambda *_a, **_k: [],
        _filter_bars_to_market_session=lambda rows, _iv: list(rows or []),
        _render_chart_state=lambda **_kwargs: [],
        chart=chart,
        _prepare_chart_predicted_prices=lambda **_kwargs: [53.05, 53.10],
        _build_chart_prediction_bands=lambda **_kwargs: ([52.95, 53.00], [53.15, 53.20]),
        _get_levels_dict=lambda: None,
        _update_chart_latest_label=lambda symbol, bar=None, price=None: latest_calls.append(
            (str(symbol), float(price) if isinstance(price, (int, float)) else None)
        ),
        _normalize_interval_token=lambda iv, fallback="1m": str(iv or fallback or "1m").strip().lower(),
        _bars_by_symbol={},
        _update_details=lambda _pred: None,
        _add_to_history=lambda _pred: None,
        _ensure_feed_subscription=lambda _code: None,
        monitor=None,
        monitor_action=SimpleNamespace(setChecked=lambda _checked: None),
        _start_monitoring=lambda: None,
        log=lambda *_a, **_k: None,
        _last_analysis_log={},
    )

    pred = SimpleNamespace(
        stock_code="001325",
        stock_name="Test",
        timestamp=datetime.now(),
        signal=SimpleNamespace(value="HOLD"),
        confidence=0.38,
        prob_up=0.30,
        prob_neutral=0.40,
        prob_down=0.30,
        current_price=52.99,
        price_history=[52.80, 52.88, 52.99],
        predicted_prices=[53.00, 53.05, 53.10],
        interval="1m",
        warnings=[],
        levels=SimpleNamespace(stop_loss=0.0, target_1=0.0, target_2=0.0, target_3=0.0),
    )

    app_analysis_ops._on_analysis_done(stub, pred, request_seq=8)

    assert len(chart.calls) == 1
    payload = chart.calls[0]
    bars = list(payload.get("bars") or [])
    assert len(bars) == 3
    assert bool(bars[0].get("_close_only_fallback", False)) is True
    assert str(bars[0].get("timestamp", "")) != ""
    assert float(bars[1]["open"]) == float(bars[0]["close"])
    assert "001325" in stub._bars_by_symbol
    assert len(stub._bars_by_symbol["001325"]) == 3
    assert latest_calls
