from __future__ import annotations

from types import SimpleNamespace

from ui import app_model_chart_ops as chart_ops


class _DummyChartApp:
    _resolve_chart_prediction_series = chart_ops._resolve_chart_prediction_series

    def __init__(self, stock_code: str) -> None:
        self.current_prediction = SimpleNamespace(
            stock_code=stock_code,
            predicted_prices=[10.1, 10.2, 10.3],
            interval="1m",
        )

    @staticmethod
    def _ui_norm(value: object) -> str:
        return str(value or "").strip()

    @staticmethod
    def _normalize_interval_token(value: object, fallback: str = "1m") -> str:
        text = str(value or "").strip().lower()
        return text or str(fallback)


def test_resolve_chart_prediction_series_matches_normalized_symbol() -> None:
    app = _DummyChartApp(" 600519 ")

    vals, src_iv = app._resolve_chart_prediction_series(
        symbol="600519",
        fallback_interval="1m",
    )

    assert vals == [10.1, 10.2, 10.3]
    assert src_iv == "1m"


def test_resolve_chart_prediction_series_rejects_other_symbol() -> None:
    app = _DummyChartApp("600519")

    vals, src_iv = app._resolve_chart_prediction_series(
        symbol="600520",
        fallback_interval="1m",
    )

    assert vals == []
    assert src_iv == "1m"
