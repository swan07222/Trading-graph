from __future__ import annotations

import threading

from core.types import Signal
from models.predictor_forecast_ops import get_top_picks
from models.predictor_types import Prediction


class _DummyPredictor:
    def __init__(self, predictions: list[Prediction]) -> None:
        self._predict_lock = threading.RLock()
        self._predictions = list(predictions)

    def predict_quick_batch(self, _stock_codes: list[str]) -> list[Prediction]:
        return list(self._predictions)


def test_get_top_picks_uses_screener_overlay(monkeypatch) -> None:
    p1 = Prediction(stock_code="600519", signal=Signal.BUY, confidence=0.82)
    p2 = Prediction(stock_code="000001", signal=Signal.BUY, confidence=0.80)
    predictor = _DummyPredictor([p1, p2])

    class _FakeScreener:
        def rank_predictions(self, predictions, *, top_n: int, include_fundamentals: bool = True):
            assert include_fundamentals is True
            assert top_n == 1
            return [predictions[1], predictions[0]]

    import analysis.screener as screener_mod

    monkeypatch.setattr(screener_mod, "build_default_screener", lambda: _FakeScreener())
    out = get_top_picks(predictor, ["600519", "000001"], n=1, signal_type="buy")

    assert len(out) == 1
    assert out[0].stock_code == "000001"
