from __future__ import annotations

from types import SimpleNamespace

from ui import app_model_chart_ops as chart_ops


class _ChartRecorder:
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


class _DummyRenderApp:
    _render_chart_state = chart_ops._render_chart_state
    _resolve_chart_prediction_series = chart_ops._resolve_chart_prediction_series

    def __init__(self) -> None:
        self.chart = _ChartRecorder()
        self._chart_symbol = ""
        self._bars_by_symbol: dict[str, list[dict[str, object]]] = {}
        self.current_prediction = None
        self.forecast_spin = SimpleNamespace(value=lambda: 2)
        self.predictor = None

    @staticmethod
    def _ui_norm(value: object) -> str:
        return str(value or "").strip()

    @staticmethod
    def _normalize_interval_token(value: object, fallback: str = "1m") -> str:
        iv = str(value or "").strip().lower()
        return iv or str(fallback or "1m").strip().lower()

    @staticmethod
    def _safe_list(values: object) -> list[object]:
        if values is None:
            return []
        try:
            return list(values)
        except TypeError:
            return []

    @staticmethod
    def _effective_anchor_price(_symbol: str, candidate: float | None = None) -> float:
        try:
            px = float(candidate or 0.0)
        except (TypeError, ValueError):
            px = 0.0
        return px if px > 0 else 10.0

    @staticmethod
    def _scrub_chart_bars(
        bars: list[dict[str, object]] | None,
        _interval: str,
        *,
        symbol: str = "",
        anchor_price: float | None = None,
    ) -> list[dict[str, object]]:
        _ = (symbol, anchor_price)
        return list(bars or [])

    @staticmethod
    def _stabilize_chart_depth(
        _symbol: str,
        _interval: str,
        candidate: list[dict[str, object]] | None,
    ) -> list[dict[str, object]]:
        return list(candidate or [])

    @staticmethod
    def _debug_candle_quality(**_kwargs: object) -> None:
        return

    @staticmethod
    def _prepare_chart_predicted_prices(**_kwargs: object) -> list[float]:
        return []

    @staticmethod
    def _build_chart_prediction_bands(**_kwargs: object) -> tuple[list[float], list[float]]:
        return [], []

    @staticmethod
    def _debug_forecast_quality(**_kwargs: object) -> None:
        return

    @staticmethod
    def _debug_chart_state(**_kwargs: object) -> None:
        return

    @staticmethod
    def _get_levels_dict() -> None:
        return None

    @staticmethod
    def _update_chart_latest_label(
        _symbol: str,
        *,
        bar: dict[str, object] | None = None,
        price: float | None = None,
    ) -> None:
        _ = (bar, price)
        return


def test_render_chart_state_filters_invalid_prepared_predictions() -> None:
    app = _DummyRenderApp()
    bars = [
        {
            "open": 10.0,
            "high": 10.2,
            "low": 9.9,
            "close": 10.1,
            "interval": "1m",
            "timestamp": "2026-02-20T09:31:00+08:00",
        }
    ]

    app._render_chart_state(
        symbol="600519",
        interval="1m",
        bars=bars,
        context="unit_test",
        current_price=10.1,
        predicted_prices=[10.2, "bad", float("nan"), -1.0, 10.3, 10.4],
        source_interval="1m",
        target_steps=2,
        predicted_prepared=True,
    )

    assert app.chart.calls
    payload = app.chart.calls[-1]
    assert payload["predicted_prices"] == [10.2, 10.3]
