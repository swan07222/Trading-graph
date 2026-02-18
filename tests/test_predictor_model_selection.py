from models.predictor import Predictor


def _touch(path):
    path.write_bytes(b"x")


def test_model_pair_prefers_same_interval_when_exact_horizon_missing(tmp_path):
    _touch(tmp_path / "ensemble_1d_5.pt")
    _touch(tmp_path / "scaler_1d_5.pkl")
    _touch(tmp_path / "ensemble_1m_30.pt")
    _touch(tmp_path / "scaler_1m_30.pkl")

    predictor = Predictor.__new__(Predictor)
    predictor.interval = "1m"
    predictor.horizon = 120

    ens, scl = predictor._find_best_model_pair(tmp_path)

    assert ens is not None
    assert ens.name == "ensemble_1m_30.pt"
    assert scl is not None
    assert scl.name == "scaler_1m_30.pkl"


def test_model_pair_prefers_exact_interval_and_horizon(tmp_path):
    _touch(tmp_path / "ensemble_1m_120.pt")
    _touch(tmp_path / "scaler_1m_120.pkl")
    _touch(tmp_path / "ensemble_1m_30.pt")
    _touch(tmp_path / "scaler_1m_30.pkl")

    predictor = Predictor.__new__(Predictor)
    predictor.interval = "1m"
    predictor.horizon = 120

    ens, scl = predictor._find_best_model_pair(tmp_path)

    assert ens is not None
    assert ens.name == "ensemble_1m_120.pt"
    assert scl is not None
    assert scl.name == "scaler_1m_120.pkl"


def test_model_pair_intraday_request_prefers_intraday_fallback(tmp_path):
    _touch(tmp_path / "ensemble_1d_5.pt")
    _touch(tmp_path / "scaler_1d_5.pkl")
    _touch(tmp_path / "ensemble_1m_30.pt")
    _touch(tmp_path / "scaler_1m_30.pkl")

    predictor = Predictor.__new__(Predictor)
    predictor.interval = "5m"
    predictor.horizon = 30

    ens, scl = predictor._find_best_model_pair(tmp_path)

    # Interval matching is strict: no cross-interval fallback.
    assert ens is None
    assert scl is None


def test_best_scaler_checkpoint_prefers_same_interval_nearest_horizon(tmp_path):
    _touch(tmp_path / "scaler_1d_5.pkl")
    _touch(tmp_path / "scaler_1m_5.pkl")
    _touch(tmp_path / "scaler_1m_30.pkl")

    predictor = Predictor.__new__(Predictor)
    predictor.interval = "1m"
    predictor.horizon = 22

    sp = predictor._find_best_scaler_checkpoint(tmp_path)

    assert sp is not None
    assert sp.name == "scaler_1m_30.pkl"


def test_model_checkpoint_selection_accepts_candidate_suffix(tmp_path):
    _touch(tmp_path / "ensemble_1m_5.candidate.pt")
    _touch(tmp_path / "forecast_1m_5.candidate.pt")
    _touch(tmp_path / "scaler_1m_5.candidate.pkl")

    predictor = Predictor.__new__(Predictor)
    predictor.interval = "1m"
    predictor.horizon = 30

    ens, scl = predictor._find_best_model_pair(tmp_path)
    fore = predictor._find_best_forecaster_checkpoint(tmp_path)

    assert ens is not None
    assert ens.name == "ensemble_1m_5.candidate.pt"
    assert scl is not None
    assert scl.name == "scaler_1m_5.candidate.pkl"
    assert fore is not None
    assert fore.name == "forecast_1m_5.candidate.pt"


def test_trained_stock_fallback_uses_matching_learner_state(tmp_path):
    import json

    from config.settings import CONFIG

    state_path = tmp_path / "learner_state.json"
    state_path.write_text(
        json.dumps(
            {
                "_data": {
                    "last_interval": "1m",
                    "last_horizon": 30,
                    "replay": {"buffer": ["600519", "000001", "abc"]},
                }
            }
        ),
        encoding="utf-8",
    )

    old_cached = CONFIG._data_dir_cached
    try:
        CONFIG._data_dir_cached = tmp_path

        predictor = Predictor.__new__(Predictor)
        codes = predictor._load_trained_stocks_from_learner_state("1m", 30)
        assert codes == ["600519", "000001"]

        no_match = predictor._load_trained_stocks_from_learner_state("1d", 5)
        assert no_match == []
    finally:
        CONFIG._data_dir_cached = old_cached
