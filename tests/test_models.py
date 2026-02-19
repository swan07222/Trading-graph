
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import CONFIG


class TestNetworkArchitectures:
    """Test neural network architectures"""

    @pytest.fixture
    def sample_input(self):
        """Create sample input tensor"""
        batch_size = 4
        seq_len = CONFIG.SEQUENCE_LENGTH
        features = 35

        return torch.randn(batch_size, seq_len, features)

    def test_lstm_model(self, sample_input):
        """Test LSTM model forward pass"""
        from models.networks import LSTMModel

        model = LSTMModel(
            input_size=sample_input.shape[2],
            hidden_size=64,
            num_classes=3
        )

        logits, conf = model(sample_input)

        assert logits.shape == (4, 3)
        assert conf.shape == (4, 1)
        assert not torch.isnan(logits).any()

    def test_transformer_model(self, sample_input):
        """Test Transformer model forward pass"""
        from models.networks import TransformerModel

        model = TransformerModel(
            input_size=sample_input.shape[2],
            hidden_size=64,
            num_classes=3
        )

        logits, conf = model(sample_input)

        assert logits.shape == (4, 3)
        assert not torch.isnan(logits).any()

    def test_gru_model(self, sample_input):
        """Test GRU model forward pass"""
        from models.networks import GRUModel

        model = GRUModel(
            input_size=sample_input.shape[2],
            hidden_size=64,
            num_classes=3
        )

        logits, conf = model(sample_input)

        assert logits.shape == (4, 3)

    def test_tcn_model(self, sample_input):
        """Test TCN model forward pass"""
        from models.networks import TCNModel

        model = TCNModel(
            input_size=sample_input.shape[2],
            hidden_size=64,
            num_classes=3
        )

        logits, conf = model(sample_input)

        assert logits.shape == (4, 3)

class TestEnsemble:
    """Test ensemble model"""

    @pytest.fixture
    def sample_data(self):
        """Create sample training data"""
        n_samples = 100
        seq_len = CONFIG.SEQUENCE_LENGTH
        features = 35

        X = np.random.randn(n_samples, seq_len, features).astype(np.float32)
        y = np.random.randint(0, 3, n_samples).astype(np.int64)

        return X, y

    def test_ensemble_creation(self, sample_data):
        """Test ensemble creation"""
        from models.ensemble import EnsembleModel

        X, y = sample_data
        input_size = X.shape[2]

        ensemble = EnsembleModel(input_size, model_names=['lstm', 'gru'])

        assert len(ensemble.models) == 2
        assert 'lstm' in ensemble.models
        assert 'gru' in ensemble.models

    def test_ensemble_prediction(self, sample_data):
        """Test ensemble prediction"""
        from models.ensemble import EnsembleModel

        X, y = sample_data
        input_size = X.shape[2]

        ensemble = EnsembleModel(input_size, model_names=['lstm'])

        pred = ensemble.predict(X[0:1])

        assert pred.predicted_class in [0, 1, 2]
        assert 0 <= pred.confidence <= 1
        assert len(pred.probabilities) == 3
        assert 0 <= pred.raw_confidence <= 1
        assert 0 <= pred.margin <= 1
        assert pred.brier_score >= 0

    def test_ensemble_training(self, sample_data):
        """Test ensemble traini`ng (short)"""
        from models.ensemble import EnsembleModel

        X, y = sample_data
        input_size = X.shape[2]

        ensemble = EnsembleModel(input_size, model_names=['lstm'])

        history = ensemble.train(
            X[:80], y[:80],
            X[80:], y[80:],
            epochs=2
        )

        assert 'lstm' in history
        # FIXED: Early stopping can make this shorter
        assert len(history['lstm']['val_acc']) >= 1
        assert len(history['lstm']['val_acc']) <= 2

    def test_partial_weight_update_preserves_untrained_mass(
        self, sample_data
    ):
        """Partial training updates should not fabricate scores for untrained models."""
        from models.ensemble import EnsembleModel

        X, _y = sample_data
        input_size = X.shape[2]
        ensemble = EnsembleModel(
            input_size, model_names=["lstm", "gru", "tcn"]
        )

        ensemble.weights = {"lstm": 0.2, "gru": 0.5, "tcn": 0.3}
        ensemble._update_weights({"lstm": 0.90})

        assert abs(sum(ensemble.weights.values()) - 1.0) < 1e-9
        assert ensemble.weights["gru"] > 0.0
        assert ensemble.weights["tcn"] > 0.0
        # Untrained models should keep substantial share, not be overwritten by placeholders.
        assert (ensemble.weights["gru"] + ensemble.weights["tcn"]) >= 0.50

    def test_full_weight_update_reflects_validation_ranking(
        self, sample_data
    ):
        """When all models are trained, higher validation accuracy should dominate."""
        from models.ensemble import EnsembleModel

        X, _y = sample_data
        input_size = X.shape[2]
        ensemble = EnsembleModel(
            input_size, model_names=["lstm", "gru", "tcn"]
        )

        ensemble._update_weights({"lstm": 0.55, "gru": 0.75, "tcn": 0.65})

        assert abs(sum(ensemble.weights.values()) - 1.0) < 1e-9
        assert ensemble.weights["gru"] > ensemble.weights["tcn"]
        assert ensemble.weights["tcn"] > ensemble.weights["lstm"]

    def test_train_skips_calibration_when_stop_requested(
        self, monkeypatch
    ):
        """If stop is requested before training, calibration must be skipped."""
        from models.ensemble import EnsembleModel

        input_size = 35
        ensemble = EnsembleModel(input_size, model_names=["lstm"])

        X_train = np.random.randn(64, CONFIG.SEQUENCE_LENGTH, input_size).astype(
            np.float32
        )
        y_train = np.random.randint(0, 3, 64).astype(np.int64)
        X_val = np.random.randn(160, CONFIG.SEQUENCE_LENGTH, input_size).astype(
            np.float32
        )
        y_val = np.random.randint(0, 3, 160).astype(np.int64)

        calls = {"calibrate": 0}

        def _fake_calibrate(*_args, **_kwargs):
            calls["calibrate"] += 1

        monkeypatch.setattr(ensemble, "calibrate", _fake_calibrate, raising=True)

        class _Stop:
            is_cancelled = True

        hist = ensemble.train(
            X_train,
            y_train,
            X_val,
            y_val,
            epochs=1,
            stop_flag=_Stop(),
        )

        assert hist == {}
        assert calls["calibrate"] == 0

    def test_train_skips_calibration_after_midcycle_stop(
        self, monkeypatch
    ):
        """If stop is triggered during a cycle, calibration must be skipped."""
        from models.ensemble import EnsembleModel

        input_size = 35
        ensemble = EnsembleModel(input_size, model_names=["lstm"])

        X_train = np.random.randn(96, CONFIG.SEQUENCE_LENGTH, input_size).astype(
            np.float32
        )
        y_train = np.random.randint(0, 3, 96).astype(np.int64)
        X_val = np.random.randn(160, CONFIG.SEQUENCE_LENGTH, input_size).astype(
            np.float32
        )
        y_val = np.random.randint(0, 3, 160).astype(np.int64)

        calls = {"calibrate": 0}

        def _fake_calibrate(*_args, **_kwargs):
            calls["calibrate"] += 1

        monkeypatch.setattr(ensemble, "calibrate", _fake_calibrate, raising=True)

        class _ToggleStop:
            def __init__(self):
                self.calls = 0

            def __call__(self):
                self.calls += 1
                return self.calls >= 3

        stop = _ToggleStop()
        hist = ensemble.train(
            X_train,
            y_train,
            X_val,
            y_val,
            epochs=2,
            batch_size=32,
            stop_flag=stop,
        )

        assert "lstm" in hist
        assert stop.calls >= 3
        assert calls["calibrate"] == 0

    def test_ensemble_save_load_preserves_trained_stock_codes(
        self, sample_data, tmp_path
    ):
        """Trained stock metadata should round-trip through save/load."""
        from models.ensemble import EnsembleModel

        X, _y = sample_data
        input_size = X.shape[2]

        ensemble = EnsembleModel(input_size, model_names=['lstm'])
        ensemble.interval = "1m"
        ensemble.prediction_horizon = 20
        ensemble.trained_stock_codes = ["600519", "000001", "300750"]
        ensemble.trained_stock_last_train = {
            "600519": "2026-02-19T10:00:00",
            "000001": "2026-02-18T09:30:00",
        }

        model_path = tmp_path / "ensemble_1m_20.pt"
        ensemble.save(str(model_path))

        loaded = EnsembleModel(input_size, model_names=['lstm'])
        assert loaded.load(str(model_path)) is True
        info = loaded.get_model_info()
        assert info.get("trained_stock_count", 0) == 3
        assert info.get("trained_stock_codes", []) == [
            "600519", "000001", "300750"
        ]
        assert info.get("trained_stock_last_train", {}) == {
            "600519": "2026-02-19T10:00:00",
            "000001": "2026-02-18T09:30:00",
        }

class TestTrainer:
    """Test trainer module"""

    def test_trainer_initialization(self):
        """Test trainer initialization"""
        from models.trainer import Trainer

        trainer = Trainer()

        assert trainer.fetcher is not None
        assert trainer.processor is not None
        assert trainer.feature_engine is not None

class TestPredictor:
    """Test predictor module"""

    def test_predictor_initialization(self):
        """Test predictor initialization"""
        from models.predictor import Predictor

        predictor = Predictor()

        assert predictor.fetcher is not None
        assert predictor.feature_engine is not None

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
