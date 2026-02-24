"""Tests for uncertainty quantification module."""
import numpy as np

from models.uncertainty_quantification import (
    ConformalPrediction,
    ConformalPredictor,
    DeepEnsemble,
    MonteCarloDropout,
    UncertaintyDecomposition,
    UncertaintyQuantifier,
)


class TestUncertaintyDecomposition:
    """Test uncertainty decomposition."""

    def test_create_decomposition(self) -> None:
        """Test creating uncertainty decomposition."""
        decomp = UncertaintyDecomposition(
            total_uncertainty=0.20,
            epistemic_uncertainty=0.12,
            aleatoric_uncertainty=0.08,
        )
        
        assert decomp.total_uncertainty == 0.20
        assert decomp.epistemic_uncertainty == 0.12
        assert decomp.aleatoric_uncertainty == 0.08

    def test_ratio_calculation(self) -> None:
        """Test ratio calculation."""
        decomp = UncertaintyDecomposition(
            total_uncertainty=0.25,
            epistemic_uncertainty=0.15,
            aleatoric_uncertainty=0.10,
        )
        
        assert abs(decomp.epistemic_ratio - 0.60) < 0.01
        assert abs(decomp.aleatoric_ratio - 0.40) < 0.01

    def test_zero_uncertainty(self) -> None:
        """Test zero uncertainty case."""
        decomp = UncertaintyDecomposition(
            total_uncertainty=0.0,
            epistemic_uncertainty=0.0,
            aleatoric_uncertainty=0.0,
        )
        
        assert decomp.epistemic_ratio == 0.0
        assert decomp.aleatoric_ratio == 0.0


class TestConformalPrediction:
    """Test conformal prediction."""

    def test_create_conformal_prediction(self) -> None:
        """Test creating conformal prediction."""
        cp = ConformalPrediction(
            prediction=100.0,
            lower_bound=95.0,
            upper_bound=105.0,
            coverage=0.90,
            efficiency=0.1,
            calibration_set_size=100,
            is_valid=True,
        )
        
        assert cp.prediction == 100.0
        assert cp.lower_bound == 95.0
        assert cp.upper_bound == 105.0
        assert cp.coverage == 0.90


class TestMonteCarloDropout:
    """Test Monte Carlo Dropout."""

    def test_init(self) -> None:
        """Test initialization."""
        mc = MonteCarloDropout(n_samples=30, dropout_rate=0.3)
        
        assert mc.n_samples == 30
        assert mc.dropout_rate == 0.3

    def test_fallback_predict(self) -> None:
        """Test fallback prediction without PyTorch."""
        mc = MonteCarloDropout()
        
        class MockModel:
            def predict(self, X: np.ndarray) -> np.ndarray:
                return np.array([0.6])
        
        X = np.array([0.5, 0.3, 0.8])
        pred, decomp = mc._fallback_predict(MockModel(), X)
        
        assert pred[0] == 0.6
        assert decomp.total_uncertainty == 0.1


class TestDeepEnsemble:
    """Test Deep Ensemble."""

    def test_init(self) -> None:
        """Test initialization."""
        ensemble = DeepEnsemble(n_models=5, aggregation="mean")
        
        assert ensemble.n_models == 5
        assert ensemble.aggregation == "mean"
        assert ensemble.ensemble_size == 0

    def test_add_model(self) -> None:
        """Test adding model to ensemble."""
        ensemble = DeepEnsemble()
        
        class MockModel:
            def predict(self, X: np.ndarray) -> np.ndarray:
                return np.array([0.7])
        
        ensemble.add_model(MockModel(), validation_accuracy=0.85)
        
        assert ensemble.ensemble_size == 1

    def test_predict_with_uncertainty_empty(self) -> None:
        """Test prediction with empty ensemble."""
        ensemble = DeepEnsemble()
        
        X = np.array([0.5, 0.3, 0.8])
        pred, decomp, individual = ensemble.predict_with_uncertainty(X)
        
        assert pred[0] == 0.5
        assert decomp.total_uncertainty == 1.0
        assert individual == []

    def test_predict_with_uncertainty(self) -> None:
        """Test prediction with ensemble."""
        ensemble = DeepEnsemble(n_models=3, aggregation="mean")
        
        class MockModel:
            def __init__(self, offset: float) -> None:
                self.offset = offset
            
            def predict(self, X: np.ndarray) -> np.ndarray:
                return np.array([0.6 + self.offset])
        
        # Add models with different predictions
        ensemble.add_model(MockModel(-0.1), 0.80)
        ensemble.add_model(MockModel(0.0), 0.85)
        ensemble.add_model(MockModel(0.1), 0.82)
        
        X = np.array([0.5, 0.3, 0.8])
        pred, decomp, individual = ensemble.predict_with_uncertainty(X)
        
        assert len(individual) == 3
        assert abs(pred[0] - 0.6) < 0.01  # Mean of 0.5, 0.6, 0.7

    def test_model_agreement(self) -> None:
        """Test model agreement metric."""
        ensemble = DeepEnsemble()
        
        # Single model - perfect agreement
        assert ensemble.get_model_agreement() == 1.0


class TestConformalPredictor:
    """Test Conformal Predictor."""

    def test_init(self) -> None:
        """Test initialization."""
        cp = ConformalPredictor(coverage=0.95)
        
        assert cp.coverage == 0.95
        assert cp._quantile is None

    def test_calibrate(self) -> None:
        """Test calibration."""
        cp = ConformalPredictor(coverage=0.90)
        
        class MockModel:
            def predict(self, X: np.ndarray) -> np.ndarray:
                return X * 1.1  # Slight overprediction
        
        X_cal = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
        y_cal = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        
        cp.calibrate(MockModel(), X_cal, y_cal)
        
        assert cp._quantile is not None
        assert cp._quantile >= 0

    def test_predict_with_interval(self) -> None:
        """Test prediction with interval."""
        cp = ConformalPredictor(coverage=0.90)
        
        class MockModel:
            def predict(self, X: np.ndarray) -> np.ndarray:
                return np.array([100.0])
        
        # Calibrate first
        X_cal = np.array([[1.0], [2.0], [3.0]])
        y_cal = np.array([1.0, 2.0, 3.0])
        cp.calibrate(MockModel(), X_cal, y_cal)
        
        # Predict
        X = np.array([100.0])
        result = cp.predict_with_interval(MockModel(), X)
        
        assert result.prediction == 100.0
        assert result.lower_bound <= 100.0
        assert result.upper_bound >= 100.0
        assert result.coverage == 0.90
        assert result.is_valid is True


class TestUncertaintyQuantifier:
    """Test unified uncertainty quantifier."""

    def test_init(self) -> None:
        """Test initialization."""
        uq = UncertaintyQuantifier(
            use_mc_dropout=False,
            use_ensemble=True,
            use_conformal=False,
        )
        
        assert uq.mc_dropout is None
        assert uq.ensemble is not None
        assert uq.conformal is None

    def test_quantify_uncertainty(self) -> None:
        """Test uncertainty quantification."""
        uq = UncertaintyQuantifier(
            use_mc_dropout=False,
            use_ensemble=True,
            use_conformal=False,
        )
        
        class MockModel:
            def predict(self, X: np.ndarray) -> np.ndarray:
                return np.array([0.65])
        
        # Add a model to ensemble
        uq.ensemble.add_model(MockModel())
        
        X = np.array([0.5, 0.3, 0.8])
        results = uq.quantify_uncertainty(MockModel(), X, 100.0)
        
        assert "timestamp" in results
        assert "current_price" in results
        assert results["current_price"] == 100.0
        assert "ensemble" in results
        assert "combined_uncertainty" in results

    def test_all_methods_disabled(self) -> None:
        """Test with all methods disabled."""
        uq = UncertaintyQuantifier(
            use_mc_dropout=False,
            use_ensemble=False,
            use_conformal=False,
        )
        
        class MockModel:
            def predict(self, X: np.ndarray) -> np.ndarray:
                return np.array([0.5])
        
        X = np.array([0.5])
        results = uq.quantify_uncertainty(MockModel(), X, 100.0)
        
        assert "combined_uncertainty" in results
        assert results["combined_uncertainty"] == 0.1  # Default
