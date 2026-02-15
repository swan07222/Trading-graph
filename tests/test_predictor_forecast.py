import threading
from types import SimpleNamespace

import numpy as np
import pandas as pd
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


def test_prediction_cache_ttl_is_shorter_for_realtime_intraday():
    predictor = Predictor.__new__(Predictor)
    predictor._cache_lock = threading.Lock()
    predictor._pred_cache = {}
    predictor._CACHE_TTL = 5.0
    predictor._CACHE_TTL_REALTIME = 1.2

    pred = Prediction(
        stock_code="600519",
        interval="1m",
        horizon=30,
        predicted_prices=[1.0, 2.0, 3.0],
    )
    predictor._set_cached_prediction("600519:1m:30:rt", pred)

    ts, payload = predictor._pred_cache["600519:1m:30:rt"]
    predictor._pred_cache["600519:1m:30:rt"] = (ts - 2.0, payload)

    assert predictor._get_cached_prediction("600519:1m:30:rt", ttl=5.0) is not None
    assert predictor._get_cached_prediction("600519:1m:30:rt", ttl=1.2) is None


def test_cache_ttl_profile_prefers_short_realtime_window():
    predictor = Predictor.__new__(Predictor)
    predictor._CACHE_TTL = 5.0
    predictor._CACHE_TTL_REALTIME = 1.2

    assert predictor._get_cache_ttl(use_realtime=False, interval="1m") == 5.0
    assert predictor._get_cache_ttl(use_realtime=True, interval="1m") == 1.2
    assert predictor._get_cache_ttl(use_realtime=True, interval="1d") == 2.0


class _BatchOnlyEnsemble:
    def __init__(self):
        self.batch_calls = 0
        self.single_calls = 0

    def predict_batch(self, X, batch_size=1024):
        self.batch_calls += 1
        out = []
        for _ in range(len(X)):
            out.append(
                SimpleNamespace(
                    probabilities=np.array([0.15, 0.25, 0.60], dtype=float),
                    predicted_class=2,
                    confidence=0.82,
                    agreement=0.90,
                    entropy=0.20,
                )
            )
        return out

    def predict(self, _x):
        self.single_calls += 1
        raise AssertionError("single predict should not be called")


class _BatchFailSingleOkEnsemble:
    def __init__(self):
        self.batch_calls = 0
        self.single_calls = 0

    def predict_batch(self, X, batch_size=1024):
        self.batch_calls += 1
        raise RuntimeError("batch failure")

    def predict(self, _x):
        self.single_calls += 1
        return SimpleNamespace(
            probabilities=np.array([0.20, 0.30, 0.50], dtype=float),
            predicted_class=2,
            confidence=0.75,
            agreement=0.85,
            entropy=0.25,
        )


class _DummyFeatureEngine:
    MIN_ROWS = 10

    def create_features(self, df):
        return df


class _DummyProcessor:
    def prepare_inference_sequence(self, df, _feature_cols):
        val = float(df["close"].iloc[-1])
        out = np.zeros((1, 60, 8), dtype=np.float32)
        out[:, :, 0] = val
        return out


def _mk_df(price: float) -> pd.DataFrame:
    rows = 120
    base = np.linspace(price - 1.0, price, rows, dtype=float)
    return pd.DataFrame(
        {
            "open": base,
            "high": base + 0.5,
            "low": base - 0.5,
            "close": base,
            "volume": np.full(rows, 1000, dtype=float),
        }
    )


def _mk_quick_predictor(ensemble) -> Predictor:
    predictor = Predictor.__new__(Predictor)
    predictor._predict_lock = threading.RLock()
    predictor.interval = "1m"
    predictor.ensemble = ensemble
    predictor._high_precision = {"enabled": 0.0}
    predictor.feature_engine = _DummyFeatureEngine()
    predictor.processor = _DummyProcessor()
    predictor._feature_cols = []
    predictor._clean_code = lambda code: str(code).strip()
    predictor._fetch_data = (
        lambda code, interval, lookback, use_realtime: _mk_df(100.0 + len(str(code)))
    )
    return predictor


def test_predict_quick_batch_uses_batched_ensemble_path():
    ensemble = _BatchOnlyEnsemble()
    predictor = _mk_quick_predictor(ensemble)

    out = predictor.predict_quick_batch(["600519", "000001", "300750"])

    assert len(out) == 3
    assert ensemble.batch_calls == 1
    assert ensemble.single_calls == 0
    assert all(float(p.confidence) > 0.7 for p in out)


def test_predict_quick_batch_falls_back_to_single_predict():
    ensemble = _BatchFailSingleOkEnsemble()
    predictor = _mk_quick_predictor(ensemble)

    out = predictor.predict_quick_batch(["600519", "000001", "300750"])

    assert len(out) == 3
    assert ensemble.batch_calls == 1
    assert ensemble.single_calls == 3
    assert all(float(p.confidence) > 0.7 for p in out)


def test_apply_ensemble_result_clips_and_normalizes_probabilities():
    predictor = Predictor.__new__(Predictor)
    predictor._high_precision = {"enabled": 0.0}

    pred = Prediction(stock_code="600519")
    ensemble_pred = SimpleNamespace(
        probabilities=np.array([1.2, -0.1, 0.3], dtype=float),
        predicted_class=2,
        confidence=0.88,
        agreement=0.80,
        entropy=0.25,
    )

    predictor._apply_ensemble_result(ensemble_pred, pred)

    assert 0.0 <= pred.prob_down <= 1.0
    assert 0.0 <= pred.prob_neutral <= 1.0
    assert 0.0 <= pred.prob_up <= 1.0
    assert abs((pred.prob_down + pred.prob_neutral + pred.prob_up) - 1.0) < 1e-9
