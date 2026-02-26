from __future__ import annotations

from types import SimpleNamespace

from models.predictor import Predictor
from ui.app_common import MainAppCommonMixin


class _DummyApp(MainAppCommonMixin):
    def __init__(self, predictor: object) -> None:
        self.predictor = predictor


def test_forecast_runtime_ready_requires_scaler_and_core_components() -> None:
    predictor = Predictor.__new__(Predictor)
    predictor.processor = SimpleNamespace(is_fitted=True)
    predictor.feature_engine = object()
    predictor.fetcher = object()
    predictor.ensemble = None
    predictor.forecaster = None

    assert predictor._forecast_ready_for_runtime() is True

    predictor.feature_engine = None
    assert predictor._forecast_ready_for_runtime() is False


def test_full_runtime_ready_still_requires_signal_models() -> None:
    predictor = Predictor.__new__(Predictor)
    predictor.processor = SimpleNamespace(is_fitted=True)
    predictor.feature_engine = object()
    predictor.fetcher = object()
    predictor.ensemble = None
    predictor.forecaster = None

    assert predictor._forecast_ready_for_runtime() is True
    assert predictor._models_ready_for_runtime() is False


def test_app_common_model_summary_reports_forecast_only_mode() -> None:
    predictor = SimpleNamespace(
        _forecast_ready_for_runtime=lambda: True,
        _models_ready_for_runtime=lambda: False,
        processor=SimpleNamespace(is_fitted=True),
        ensemble=None,
        forecaster=None,
    )
    app = _DummyApp(predictor)

    summary = app._predictor_model_summary()
    assert summary["runtime_ready"] is False
    assert summary["forecast_ready"] is True
