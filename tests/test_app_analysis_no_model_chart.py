from __future__ import annotations

from types import SimpleNamespace

from ui import app_analysis_ops


def test_analyze_stock_no_model_loads_online_chart_before_render() -> None:
    loaded_calls: list[tuple[str, str, int]] = []
    render_calls: list[dict[str, object]] = []
    logs: list[tuple[str, str]] = []

    stub = SimpleNamespace(
        stock_input=SimpleNamespace(text=lambda: "600519"),
        signal_panel=SimpleNamespace(reset=lambda: None),
        current_prediction=None,
        interval_combo=SimpleNamespace(currentText=lambda: "1m"),
        lookback_spin=SimpleNamespace(value=lambda: 20),
        forecast_spin=SimpleNamespace(value=lambda: 30),
        _bars_by_symbol={},
        _ui_norm=lambda value: str(value or "").strip(),
        _predictor_runtime_ready=lambda: False,
        _normalize_interval_token=lambda value, fallback="1m": str(
            value or fallback or "1m"
        ).strip().lower(),
        _recommended_lookback=lambda _interval: 120,
        _load_chart_history_bars=lambda sym, iv, lb: loaded_calls.append(
            (sym, iv, int(lb))
        )
        or [
            {
                "open": 10.0,
                "high": 10.2,
                "low": 9.9,
                "close": 10.1,
                "timestamp": "2026-02-25T09:31:00+08:00",
                "interval": "1m",
            }
        ],
        _filter_bars_to_market_session=lambda rows, _iv: list(rows),
        _render_chart_state=lambda **kwargs: render_calls.append(dict(kwargs)),
        _debug_console=lambda *_a, **_k: None,
        log=lambda msg, level="info": logs.append((str(msg), str(level))),
    )

    app_analysis_ops._analyze_stock(stub)

    assert loaded_calls == [("600519", "1m", 120)]
    assert render_calls
    assert render_calls[-1].get("symbol") == "600519"
    assert bool(render_calls[-1].get("bars"))
    assert logs
    assert logs[-1][0] == "No model loaded. Please train a model first."
