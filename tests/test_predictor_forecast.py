import threading
from types import SimpleNamespace

import numpy as np
import torch

from models.predictor import Prediction, Predictor


def _touch(path):
    path.write_bytes(b"x")


def test_forecaster_checkpoint_prefers_same_interval(tmp_path):
    _touch(tmp_path / "forecast_1d_5.pt")
    _touch(tmp_path / "forecast_1m_30.pt")
    _touch(tmp_path / "forecast_1m_120.pt")

    predictor = Predictor.__new__(Predictor)
    predictor.interval = "1m"
    predictor.horizon = 100

    fp = predictor._find_best_forecaster_checkpoint(tmp_path)
    assert fp is not None
    assert fp.name == "forecast_1m_120.pt"


class _DummyForecaster:
    def eval(self):
        return self

    def __call__(self, _x):
        # 3-step output only, horizon in caller can be larger.
        returns = torch.tensor([[0.4, -0.2, 0.1]], dtype=torch.float32)
        conf = torch.zeros((1, 1), dtype=torch.float32)
        return returns, conf


def test_generate_forecast_forecaster_path_matches_requested_horizon():
    predictor = Predictor.__new__(Predictor)
    predictor.forecaster = _DummyForecaster()
    predictor.ensemble = None

    out = predictor._generate_forecast(
        X=np.zeros((1, 60, 8), dtype=np.float32),
        current_price=100.0,
        horizon=10,
        atr_pct=0.02,
    )

    assert len(out) == 10
    assert all(float(v) > 0 for v in out)


class _DummyEnsemble:
    def predict(self, _x):
        return SimpleNamespace(probabilities=[0.2, 0.3, 0.5])


def test_generate_forecast_fallback_uses_sequence_signature():
    predictor = Predictor.__new__(Predictor)
    predictor.forecaster = None
    predictor.ensemble = _DummyEnsemble()

    recent = [100.0 + (0.03 * i) for i in range(80)]
    out_a = predictor._generate_forecast(
        X=np.zeros((1, 60, 8), dtype=np.float32),
        current_price=102.0,
        horizon=16,
        atr_pct=0.02,
        sequence_signature=1.0,
        recent_prices=recent,
    )
    out_b = predictor._generate_forecast(
        X=np.zeros((1, 60, 8), dtype=np.float32),
        current_price=102.0,
        horizon=16,
        atr_pct=0.02,
        sequence_signature=2.0,
        recent_prices=recent,
    )

    assert len(out_a) == 16
    assert len(out_b) == 16
    assert not np.allclose(out_a, out_b)


class _NeutralEnsemble:
    def predict(self, _x):
        return SimpleNamespace(probabilities=[0.32, 0.36, 0.32])


class _ConstantUpForecaster:
    def eval(self):
        return self

    def __call__(self, _x):
        returns = torch.full((1, 30), 0.6, dtype=torch.float32)
        conf = torch.zeros((1, 1), dtype=torch.float32)
        return returns, conf


def test_generate_forecast_neutral_mode_stays_close_to_price():
    predictor = Predictor.__new__(Predictor)
    predictor.forecaster = None
    predictor.ensemble = _NeutralEnsemble()

    recent = [100.0 + np.sin(i / 6.0) * 0.25 for i in range(120)]
    out = predictor._generate_forecast(
        X=np.zeros((1, 60, 8), dtype=np.float32),
        current_price=100.0,
        horizon=30,
        atr_pct=0.02,
        sequence_signature=7.0,
        recent_prices=recent,
    )

    assert len(out) == 30
    max_dev = max(abs((float(v) / 100.0) - 1.0) for v in out)
    assert max_dev <= 0.015


def test_generate_forecast_neutral_forecaster_avoids_one_way_tail():
    predictor = Predictor.__new__(Predictor)
    predictor.forecaster = _ConstantUpForecaster()
    predictor.ensemble = _NeutralEnsemble()

    recent = [100.0 + np.sin(i / 7.0) * 0.20 for i in range(140)]
    out = predictor._generate_forecast(
        X=np.zeros((1, 60, 8), dtype=np.float32),
        current_price=100.0,
        horizon=30,
        atr_pct=0.02,
        sequence_signature=3.14,
        recent_prices=recent,
    )

    assert len(out) == 30
    max_dev = max(abs((float(v) / 100.0) - 1.0) for v in out)
    assert max_dev <= 0.015
    diffs = np.diff(np.array(out, dtype=float))
    assert np.any(diffs < 0)
    # Guard against synthetic template waves (strict up/down alternation).
    if diffs.size >= 3:
        signs = np.sign(diffs)
        flips = np.sum(signs[1:] != signs[:-1])
        flip_ratio = flips / float(max(1, signs.size - 1))
        assert flip_ratio < 0.80


def test_prediction_cache_is_contextual_and_immutable():
    predictor = Predictor.__new__(Predictor)
    predictor._cache_lock = threading.Lock()
    predictor._pred_cache = {}

    pred = Prediction(
        stock_code="600519",
        interval="1m",
        horizon=30,
        predicted_prices=[1.0, 2.0, 3.0],
    )
    predictor._set_cached_prediction("600519:1m:30", pred)

    cached = predictor._get_cached_prediction("600519:1m:30")
    assert cached is not None
    cached.predicted_prices.append(9.0)

    cached_again = predictor._get_cached_prediction("600519:1m:30")
    assert cached_again is not None
    assert cached_again.predicted_prices == [1.0, 2.0, 3.0]
    assert predictor._get_cached_prediction("600519:1d:5") is None
