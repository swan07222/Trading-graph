import threading
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from config.settings import CONFIG
from models.predictor import Prediction, Predictor, Signal


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


def test_generate_forecast_forecaster_tail_extension_is_not_flat():
    predictor = Predictor.__new__(Predictor)
    predictor.forecaster = _DummyForecaster()
    predictor.ensemble = None

    out = predictor._generate_forecast(
        X=np.zeros((1, 60, 8), dtype=np.float32),
        current_price=100.0,
        horizon=24,
        atr_pct=0.02,
        sequence_signature=11.0,
        seed_context="600519:1m",
        recent_prices=[100.0 + (0.02 * i) for i in range(120)],
    )

    assert len(out) == 24
    tail = np.array(out[6:], dtype=float)
    assert np.std(tail) > 1e-5


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


def test_generate_forecast_last_resort_without_models_is_not_flat():
    predictor = Predictor.__new__(Predictor)
    predictor.forecaster = None
    predictor.ensemble = None

    recent = [100.0 + (0.04 * np.sin(i / 5.0)) for i in range(120)]
    out = predictor._generate_forecast(
        X=np.zeros((1, 60, 8), dtype=np.float32),
        current_price=100.0,
        horizon=20,
        atr_pct=0.02,
        sequence_signature=5.0,
        seed_context="600519:1m",
        recent_prices=recent,
    )

    assert len(out) == 20
    assert np.std(np.array(out, dtype=float)) > 1e-6


def test_generate_forecast_seed_context_differentiates_same_signature():
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
        seed_context="600519:1m",
        recent_prices=recent,
    )
    out_b = predictor._generate_forecast(
        X=np.zeros((1, 60, 8), dtype=np.float32),
        current_price=102.0,
        horizon=16,
        atr_pct=0.02,
        sequence_signature=1.0,
        seed_context="000001:1m",
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


def test_generate_forecast_news_bias_tilts_curve_direction():
    predictor = Predictor.__new__(Predictor)
    predictor.interval = "1m"
    predictor.forecaster = None
    predictor.ensemble = _NeutralEnsemble()

    recent = [100.0 + np.sin(i / 8.0) * 0.2 for i in range(140)]
    out_pos = predictor._generate_forecast(
        X=np.zeros((1, 60, 8), dtype=np.float32),
        current_price=100.0,
        horizon=24,
        atr_pct=0.02,
        sequence_signature=9.0,
        seed_context="600519:1m",
        recent_prices=recent,
        news_bias=0.12,
    )
    out_neg = predictor._generate_forecast(
        X=np.zeros((1, 60, 8), dtype=np.float32),
        current_price=100.0,
        horizon=24,
        atr_pct=0.02,
        sequence_signature=9.0,
        seed_context="600519:1m",
        recent_prices=recent,
        news_bias=-0.12,
    )

    assert len(out_pos) == 24
    assert len(out_neg) == 24
    assert float(out_pos[-1]) > float(out_neg[-1])


def test_apply_news_influence_rebalances_probs_and_can_upgrade_hold():
    predictor = Predictor.__new__(Predictor)
    predictor.interval = "1m"
    predictor._get_news_sentiment = lambda *_args, **_kwargs: (0.8, 0.9, 12)

    pred = Prediction(
        stock_code="600519",
        signal=Signal.HOLD,
        confidence=0.70,
        prob_up=0.30,
        prob_neutral=0.35,
        prob_down=0.35,
    )

    news_bias = predictor._apply_news_influence(pred, "600519", "1m")

    assert news_bias > 0
    assert pred.news_count == 12
    assert pred.prob_up > 0.30
    assert pred.prob_down < 0.35
    assert pred.signal in (Signal.BUY, Signal.STRONG_BUY)
    assert abs((pred.prob_up + pred.prob_neutral + pred.prob_down) - 1.0) < 1e-9


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


def test_sanitize_history_df_repairs_zero_open_rows():
    predictor = Predictor.__new__(Predictor)
    idx = pd.to_datetime(
        [
            "2026-02-13 09:31:00",
            "2026-02-13 09:32:00",
            "2026-02-13 09:33:00",
        ]
    )
    df = pd.DataFrame(
        {
            "open": [0.0, 0.0, 39.05],
            "high": [39.10, 39.12, 39.08],
            "low": [39.00, 39.02, 39.01],
            "close": [39.04, 39.06, 39.03],
            "volume": [100, 120, 110],
        },
        index=idx,
    )

    out = predictor._sanitize_history_df(df, "1m")

    assert len(out) == 3
    assert (out["open"] > 0).all()
    assert (out["high"] >= out[["open", "close"]].max(axis=1)).all()
    assert (out["low"] <= out[["open", "close"]].min(axis=1)).all()


def test_sanitize_history_df_drops_non_session_intraday_rows():
    predictor = Predictor.__new__(Predictor)
    idx = pd.to_datetime(
        [
            "2026-02-13 08:59:00",  # pre-open
            "2026-02-13 09:31:00",  # session
            "2026-02-13 12:30:00",  # lunch break
            "2026-02-13 13:15:00",  # session
            "2026-02-13 15:10:00",  # after close
            "2026-02-14 09:35:00",  # weekend
        ]
    )
    base = pd.DataFrame(
        {
            "open": [39.0, 39.0, 39.0, 39.0, 39.0, 39.0],
            "high": [39.1, 39.1, 39.1, 39.1, 39.1, 39.1],
            "low": [38.9, 38.9, 38.9, 38.9, 38.9, 38.9],
            "close": [39.0, 39.0, 39.0, 39.0, 39.0, 39.0],
            "volume": [10, 10, 10, 10, 10, 10],
        },
        index=idx,
    )

    out = predictor._sanitize_history_df(base, "1m")

    kept = [ts.strftime("%H:%M") for ts in out.index]
    assert kept == ["09:31", "13:15"]


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


def test_realtime_forecast_curve_defaults_to_30_on_latest_1680_bars(monkeypatch):
    class _Fetcher:
        def __init__(self):
            self.last_bars = 0

        def get_history(self, _code, interval, bars, use_cache=True, update_db=True):
            self.last_bars = int(bars)
            n = int(bars)
            base = np.linspace(99.0, 101.0, n, dtype=float)
            return pd.DataFrame(
                {
                    "open": base,
                    "high": base + 0.3,
                    "low": base - 0.3,
                    "close": base,
                    "volume": np.full(n, 1000.0, dtype=float),
                    "interval": [interval] * n,
                }
            )

        def get_realtime(self, _code):
            return None

    class _EmptyCache:
        def read_history(self, symbol, interval, bars=500, final_only=True):  # noqa: ARG002
            return pd.DataFrame()

    monkeypatch.setattr(
        "data.session_cache.get_session_bar_cache",
        lambda: _EmptyCache(),
        raising=True,
    )

    predictor = Predictor.__new__(Predictor)
    predictor._predict_lock = threading.RLock()
    predictor.interval = "1m"
    predictor.horizon = 120
    predictor._clean_code = lambda code: str(code).strip()
    predictor.feature_engine = _DummyFeatureEngine()
    predictor.processor = _DummyProcessor()
    predictor._feature_cols = []
    predictor.fetcher = _Fetcher()
    predictor._get_atr_pct = lambda _df: 0.02

    captured = {"horizon": 0, "recent_len": 0}

    def _fake_generate(
        X,
        current_price,
        horizon,
        atr_pct=0.02,
        sequence_signature=0.0,
        seed_context="",
        recent_prices=None,
    ):
        captured["horizon"] = int(horizon)
        captured["recent_len"] = len(recent_prices or [])
        return [float(current_price)] * int(horizon)

    predictor._generate_forecast = _fake_generate

    actual, predicted = predictor.get_realtime_forecast_curve(
        stock_code="600519",
        interval="1m",
        horizon_steps=None,
        lookback_bars=None,
        use_realtime_price=False,
    )

    assert predictor.fetcher.last_bars == 1680
    assert len(actual) == 1680
    assert captured["horizon"] == 30
    assert captured["recent_len"] == 1680
    assert len(predicted) == 30


def test_realtime_forecast_curve_merges_partial_session_bars(monkeypatch):
    class _Fetcher:
        def get_history(self, _code, interval, bars, use_cache=True, update_db=True):  # noqa: ARG002
            n = int(bars)
            idx = pd.date_range("2026-01-05 09:30:00", periods=n, freq="1min")
            base = np.linspace(99.0, 100.0, n, dtype=float)
            return pd.DataFrame(
                {
                    "open": base,
                    "high": base + 0.2,
                    "low": base - 0.2,
                    "close": base,
                    "volume": np.full(n, 1000.0, dtype=float),
                    "interval": [interval] * n,
                },
                index=idx,
            )

        def get_realtime(self, _code):
            return None

    captured = {"final_only": True}

    class _FakeCache:
        def read_history(self, symbol, interval, bars=500, final_only=True):  # noqa: ARG002
            captured["final_only"] = bool(final_only)
            idx = pd.date_range("2026-01-05 09:30:00", periods=int(bars), freq="1min")
            base = np.linspace(99.0, 100.0, int(bars), dtype=float)
            if len(base) > 330:
                # 15:00 is inside CN session for this synthetic stream.
                base[330] = 101.0
            return pd.DataFrame(
                {
                    "open": base,
                    "high": base + 0.2,
                    "low": base - 0.2,
                    "close": base,
                    "volume": np.full(int(bars), 1200.0, dtype=float),
                    "is_final": [False] * int(bars),
                },
                index=idx,
            )

    monkeypatch.setattr(
        "data.session_cache.get_session_bar_cache",
        lambda: _FakeCache(),
        raising=True,
    )

    predictor = Predictor.__new__(Predictor)
    predictor._predict_lock = threading.RLock()
    predictor.interval = "1m"
    predictor.horizon = 30
    predictor._clean_code = lambda code: str(code).strip()
    predictor.feature_engine = _DummyFeatureEngine()
    predictor.processor = _DummyProcessor()
    predictor._feature_cols = []
    predictor.fetcher = _Fetcher()
    predictor._get_atr_pct = lambda _df: 0.02
    predictor._generate_forecast = lambda *args, **kwargs: [100.0] * 30

    actual, predicted = predictor.get_realtime_forecast_curve(
        stock_code="600519",
        interval="1m",
        horizon_steps=30,
        lookback_bars=560,
        use_realtime_price=False,
    )

    assert captured["final_only"] is False
    assert 200 <= len(actual) <= 560
    assert abs(float(actual[-1]) - 101.0) < 1e-9
    assert len(predicted) == 30


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


def test_predict_uses_short_history_bootstrap_when_intraday_history_is_short():
    predictor = Predictor.__new__(Predictor)
    predictor._predict_lock = threading.RLock()
    predictor.interval = "1m"
    predictor.horizon = 30
    predictor._clean_code = lambda code: str(code).strip()
    predictor.feature_engine = _DummyFeatureEngine()
    predictor.processor = _DummyProcessor()
    predictor._feature_cols = []
    predictor.ensemble = None
    predictor._extract_technicals = lambda _df, _pred: None
    predictor._generate_forecast = (
        lambda _X, current_price, horizon, *_args, **_kwargs:
        [float(current_price)] * int(horizon)
    )
    predictor._calculate_levels = lambda pred: pred.levels
    predictor._calculate_position = lambda pred: pred.position
    predictor._generate_reasons = lambda _pred: None
    predictor._sequence_signature = lambda _X: 0.0
    predictor._set_cached_prediction = lambda *_args, **_kwargs: None
    predictor.fetcher = SimpleNamespace(
        get_realtime=lambda _code: SimpleNamespace(price=123.0, name="Demo")
    )
    predictor._get_stock_name = lambda _code, _df: "Demo"

    calls = []

    def _fake_fetch(code, interval, lookback, use_realtime, history_allow_online=True):  # noqa: ARG001
        calls.append((str(interval), int(lookback), bool(use_realtime), bool(history_allow_online)))
        return _mk_df(100.0).tail(20)  # intentionally insufficient for full path

    predictor._fetch_data = _fake_fetch

    pred = predictor.predict(
        "300059",
        use_realtime_price=True,
        interval="1m",
        forecast_minutes=12,
        lookback_bars=560,
        skip_cache=True,
    )

    assert len(calls) == 1
    assert calls[0][0] == "1m"
    assert pred.interval == "1m"
    assert pred.current_price > 0
    assert len(pred.predicted_prices) == 12
    assert any("Short-history fallback used" in w for w in pred.warnings)
    assert not any("Interval fallback applied" in w for w in pred.warnings)
    assert not any("Insufficient data:" in w for w in pred.warnings)


def test_predict_populates_minimal_snapshot_when_all_history_paths_are_short():
    predictor = Predictor.__new__(Predictor)
    predictor._predict_lock = threading.RLock()
    predictor.interval = "1m"
    predictor.horizon = 30
    predictor._clean_code = lambda code: str(code).strip()
    predictor.feature_engine = _DummyFeatureEngine()
    predictor.processor = _DummyProcessor()
    predictor._feature_cols = []
    predictor.ensemble = None
    predictor._set_cached_prediction = lambda *_args, **_kwargs: None
    predictor.fetcher = SimpleNamespace(
        get_realtime=lambda _code: SimpleNamespace(price=88.5, name="Demo"),
        get_history=lambda *_args, **_kwargs: pd.DataFrame(),
    )
    predictor._get_stock_name = lambda _code, _df: ""
    predictor._fetch_data = (
        lambda _code, _interval, _lookback, _use_realtime, history_allow_online=True: _mk_df(100.0).tail(10)  # noqa: ARG005,E501
    )

    pred = predictor.predict(
        "002594",
        use_realtime_price=True,
        interval="1m",
        forecast_minutes=10,
        lookback_bars=560,
        skip_cache=True,
    )

    assert abs(float(pred.current_price) - 88.5) < 1e-9
    assert pred.price_history == [88.5]
    assert any("Insufficient data:" in w for w in pred.warnings)


def test_predict_uses_short_history_bootstrap_before_minimal_snapshot():
    predictor = Predictor.__new__(Predictor)
    predictor._predict_lock = threading.RLock()
    predictor.interval = "1m"
    predictor.horizon = 30
    predictor._clean_code = lambda code: str(code).strip()
    predictor.feature_engine = _DummyFeatureEngine()
    predictor.processor = _DummyProcessor()
    predictor._feature_cols = []
    predictor.ensemble = None
    predictor._set_cached_prediction = lambda *_args, **_kwargs: None
    predictor.fetcher = SimpleNamespace(
        get_realtime=lambda _code: None,
        get_history=lambda *_args, **_kwargs: pd.DataFrame(),
    )
    predictor._get_stock_name = lambda _code, _df: "Demo"
    predictor._calculate_levels = lambda pred: pred.levels
    predictor._calculate_position = lambda pred: pred.position
    predictor._generate_reasons = lambda _pred: None
    predictor._fetch_data = (
        lambda _code, _interval, _lookback, _use_realtime, history_allow_online=True: _mk_df(100.0).tail(40)  # noqa: ARG005,E501
    )

    pred = predictor.predict(
        "300059",
        use_realtime_price=True,
        interval="1m",
        forecast_minutes=10,
        lookback_bars=560,
        skip_cache=True,
    )

    assert pred.current_price > 0
    assert len(pred.predicted_prices) == 10
    assert float(pred.confidence) > 0
    assert any("Short-history fallback used:" in w for w in pred.warnings)
    assert not any("Insufficient data:" in w for w in pred.warnings)


def test_short_history_bootstrap_marks_fallback_and_suppresses_direction_by_default():
    predictor = Predictor.__new__(Predictor)
    predictor._get_stock_name = lambda _code, _df: "Demo"
    predictor._refresh_prediction_uncertainty = lambda _pred: None
    predictor._apply_tail_risk_guard = lambda _pred: None
    predictor._build_prediction_bands = lambda _pred: None
    predictor._calculate_levels = lambda pred: pred.levels
    predictor._calculate_position = lambda pred: pred.position
    predictor._generate_reasons = lambda _pred: None

    old_flag = bool(
        getattr(CONFIG.precision, "allow_short_history_directional_signals", False)
    )
    try:
        CONFIG.precision.allow_short_history_directional_signals = False
        pred = Prediction(stock_code="600519")
        ok = predictor._bootstrap_short_history_prediction(
            pred,
            _mk_df(100.0).tail(40),
            horizon=12,
            required_rows=120,
        )
        assert ok is True
        assert pred.signal == Signal.HOLD
        assert any(
            "directional signal suppressed" in w.lower()
            for w in pred.warnings
        )
    finally:
        CONFIG.precision.allow_short_history_directional_signals = old_flag


def test_short_history_bootstrap_allows_direction_when_policy_opted_in():
    predictor = Predictor.__new__(Predictor)
    predictor._get_stock_name = lambda _code, _df: "Demo"
    predictor._refresh_prediction_uncertainty = lambda _pred: None
    predictor._apply_tail_risk_guard = lambda _pred: None
    predictor._build_prediction_bands = lambda _pred: None
    predictor._calculate_levels = lambda pred: pred.levels
    predictor._calculate_position = lambda pred: pred.position
    predictor._generate_reasons = lambda _pred: None

    old_flag = bool(
        getattr(CONFIG.precision, "allow_short_history_directional_signals", False)
    )
    try:
        CONFIG.precision.allow_short_history_directional_signals = True
        close = np.linspace(100.0, 130.0, 40, dtype=float)
        df = pd.DataFrame(
            {
                "open": close,
                "high": close + 0.5,
                "low": close - 0.5,
                "close": close,
                "volume": np.full(len(close), 1000.0, dtype=float),
            }
        )
        pred = Prediction(stock_code="600519")
        ok = predictor._bootstrap_short_history_prediction(
            pred,
            df,
            horizon=12,
            required_rows=120,
        )
        assert ok is True
        assert pred.signal in (Signal.BUY, Signal.STRONG_BUY)
    finally:
        CONFIG.precision.allow_short_history_directional_signals = old_flag


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


def test_determine_signal_requires_edge_in_sideways_regime():
    predictor = Predictor.__new__(Predictor)
    pred = Prediction(
        stock_code="600519",
        trend="SIDEWAYS",
        prob_up=0.41,
        prob_down=0.37,
    )
    ensemble_pred = SimpleNamespace(
        confidence=0.95,
        predicted_class=2,
    )

    out = predictor._determine_signal(ensemble_pred, pred)
    assert out == Signal.HOLD


def test_generate_reasons_keeps_existing_gate_warnings():
    predictor = Predictor.__new__(Predictor)
    pred = Prediction(
        stock_code="600519",
        signal=Signal.HOLD,
        confidence=0.50,
        prob_up=0.40,
        prob_down=0.35,
        model_agreement=0.55,
        entropy=0.82,
    )
    pred.warnings.append("Runtime quality gate filtered signal BUY -> HOLD")

    predictor._generate_reasons(pred)

    assert any(
        "Runtime quality gate filtered signal" in item
        for item in pred.warnings
    )


def test_uncertainty_profile_builds_prediction_bands():
    predictor = Predictor.__new__(Predictor)
    pred = Prediction(
        stock_code="600519",
        signal=Signal.BUY,
        confidence=0.66,
        raw_confidence=0.74,
        prob_up=0.52,
        prob_neutral=0.21,
        prob_down=0.27,
        model_agreement=0.58,
        entropy=0.66,
        model_margin=0.04,
        atr_pct_value=0.03,
        predicted_prices=[100.0 + (0.12 * i) for i in range(20)],
        signal_strength=0.62,
    )

    predictor._refresh_prediction_uncertainty(pred)
    predictor._build_prediction_bands(pred)

    assert 0.0 <= float(pred.uncertainty_score) <= 1.0
    assert 0.0 <= float(pred.tail_risk_score) <= 1.0
    assert len(pred.predicted_prices_low) == len(pred.predicted_prices)
    assert len(pred.predicted_prices_high) == len(pred.predicted_prices)
    assert all(
        float(lo) < float(px) < float(hi)
        for lo, px, hi in zip(
            pred.predicted_prices_low,
            pred.predicted_prices,
            pred.predicted_prices_high,
            strict=False,
        )
    )


def test_tail_risk_guard_filters_fragile_actionable_signal():
    predictor = Predictor.__new__(Predictor)
    pred = Prediction(
        stock_code="600519",
        signal=Signal.BUY,
        signal_strength=0.78,
        confidence=0.62,
        prob_up=0.34,
        prob_neutral=0.20,
        prob_down=0.46,
        model_agreement=0.48,
        entropy=0.82,
        model_margin=0.02,
        atr_pct_value=0.055,
    )

    predictor._refresh_prediction_uncertainty(pred)
    predictor._apply_tail_risk_guard(pred)

    assert pred.signal == Signal.HOLD
    assert any("Tail-risk guard filtered signal" in x for x in pred.warnings)
