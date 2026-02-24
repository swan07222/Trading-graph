"""Tests for ML model explainability module."""
import tempfile
from pathlib import Path

import numpy as np
import pytest

from models.explainability import (
    FeatureImportance,
    GlobalExplanation,
    LocalExplanation,
    ModelExplainer,
    explain_prediction,
)


class TestFeatureImportance:
    """Test feature importance dataclass."""

    def test_create_feature_importance(self) -> None:
        """Test creating feature importance."""
        fi = FeatureImportance(
            feature_name="rsi_14",
            shap_value=0.15,
            shap_abs_mean=0.20,
            rank=1,
        )
        
        assert fi.feature_name == "rsi_14"
        assert fi.shap_value == 0.15
        assert fi.shap_abs_mean == 0.20
        assert fi.rank == 1

    def test_direction_positive(self) -> None:
        """Test positive direction detection."""
        fi = FeatureImportance(
            feature_name="momentum",
            shap_value=0.05,
            shap_abs_mean=0.05,
        )
        
        assert fi.direction == "positive"

    def test_direction_negative(self) -> None:
        """Test negative direction detection."""
        fi = FeatureImportance(
            feature_name="volatility",
            shap_value=-0.05,
            shap_abs_mean=0.05,
        )
        
        assert fi.direction == "negative"

    def test_direction_mixed(self) -> None:
        """Test mixed direction detection."""
        fi = FeatureImportance(
            feature_name="volume",
            shap_value=0.005,
            shap_abs_mean=0.005,
        )
        
        assert fi.direction == "mixed"


class TestLocalExplanation:
    """Test local explanation dataclass."""

    def test_create_local_explanation(self) -> None:
        """Test creating local explanation."""
        exp = LocalExplanation(
            predicted_value=0.75,
            predicted_class="BUY",
            base_value=0.5,
        )
        
        assert exp.predicted_value == 0.75
        assert exp.predicted_class == "BUY"
        assert exp.base_value == 0.5
        assert exp.prediction_id.startswith("EXP_")

    def test_explanation_text(self) -> None:
        """Test explanation text generation."""
        exp = LocalExplanation(
            predicted_value=0.75,
            predicted_class="BUY",
            base_value=0.5,
            top_positive_features=[("rsi_14", 0.1), ("momentum", 0.05)],
            top_negative_features=[("volatility", -0.05)],
        )
        
        text = exp.explanation_text
        assert "Prediction:" in text
        assert "BUY" in text
        assert "Positive factors:" in text
        assert "rsi_14" in text

    def test_to_dict(self) -> None:
        """Test serialization to dictionary."""
        exp = LocalExplanation(
            predicted_value=0.75,
            predicted_class="BUY",
        )
        
        data = exp.to_dict()
        assert data['predicted_value'] == 0.75
        assert data['predicted_class'] == "BUY"
        assert 'explanation_text' in data


class TestGlobalExplanation:
    """Test global explanation dataclass."""

    def test_create_global_explanation(self) -> None:
        """Test creating global explanation."""
        exp = GlobalExplanation(
            dataset_name="trading_data",
            num_samples=1000,
            num_features=20,
        )
        
        assert exp.dataset_name == "trading_data"
        assert exp.num_samples == 1000
        assert exp.num_features == 20

    def test_feature_importances_sorted(self) -> None:
        """Test feature importances are sorted by importance."""
        exp = GlobalExplanation(
            dataset_name="test",
            num_samples=100,
            num_features=3,
        )
        
        exp.feature_importances = [
            FeatureImportance("feat_1", 0.1, 0.15),
            FeatureImportance("feat_2", 0.3, 0.35),
            FeatureImportance("feat_3", 0.2, 0.25),
        ]
        
        # Should be sorted by abs mean
        assert exp.feature_importances[0].feature_name == "feat_2"
        assert exp.feature_importances[1].feature_name == "feat_3"
        assert exp.feature_importances[2].feature_name == "feat_1"

    def test_top_features(self) -> None:
        """Test getting top features."""
        exp = GlobalExplanation(
            dataset_name="test",
            num_samples=100,
            num_features=20,
        )
        
        # Add 15 features
        for i in range(15):
            exp.feature_importances.append(
                FeatureImportance(f"feat_{i}", 0.01 * i, 0.01 * i)
            )
        
        top = exp.top_features
        assert len(top) == 10  # Top 10

    def test_to_dict(self) -> None:
        """Test serialization to dictionary."""
        exp = GlobalExplanation(
            dataset_name="test",
            num_samples=100,
            num_features=5,
        )
        
        exp.feature_importances = [
            FeatureImportance("rsi_14", 0.2, 0.25, rank=1),
        ]
        
        data = exp.to_dict()
        assert data['num_samples'] == 100
        assert 'top_features' in data


class TestModelExplainer:
    """Test model explainer class."""

    @pytest.fixture
    def sample_model(self) -> object:
        """Create a simple mock model."""
        class MockModel:
            def predict(self, X: np.ndarray) -> np.ndarray:
                return np.array([0.7] * len(X))
            
            def predict_proba(self, X: np.ndarray) -> np.ndarray:
                # Return probabilities for 2 classes
                return np.array([[0.3, 0.7]] * len(X))
        
        return MockModel()

    @pytest.fixture
    def feature_names(self) -> list[str]:
        """Create feature names."""
        return [
            "rsi_14",
            "macd_hist",
            "volatility_20",
            "volume_ratio",
            "momentum_10",
        ]

    def test_init(self, sample_model: object, feature_names: list[str]) -> None:
        """Test explainer initialization."""
        explainer = ModelExplainer(
            model=sample_model,
            feature_names=feature_names,
        )
        
        assert explainer.model is sample_model
        assert explainer.feature_names == feature_names
        assert explainer.model_type == "classifier"

    def test_get_prediction_class(self, sample_model: object, feature_names: list[str]) -> None:
        """Test prediction class mapping."""
        explainer = ModelExplainer(
            model=sample_model,
            feature_names=feature_names,
        )
        
        assert explainer._get_prediction_class(0.8) == "STRONG_BUY"
        assert explainer._get_prediction_class(0.6) == "BUY"
        assert explainer._get_prediction_class(0.5) == "HOLD"
        assert explainer._get_prediction_class(0.25) == "SELL"
        assert explainer._get_prediction_class(0.2) == "STRONG_SELL"

    def test_explain_prediction_basic(
        self,
        sample_model: object,
        feature_names: list[str],
    ) -> None:
        """Test explaining a prediction (without SHAP)."""
        explainer = ModelExplainer(
            model=sample_model,
            feature_names=feature_names,
        )
        
        X_sample = np.array([50.0, 0.5, 0.2, 1.5, 0.1])
        prediction = 0.75
        
        explanation = explainer.explain_prediction(X_sample, prediction)
        
        assert explanation.predicted_value == 0.75
        assert explanation.predicted_class == "BUY"

    def test_explain_dataset(
        self,
        sample_model: object,
        feature_names: list[str],
    ) -> None:
        """Test explaining a dataset."""
        explainer = ModelExplainer(
            model=sample_model,
            feature_names=feature_names,
        )
        
        X = np.random.rand(100, len(feature_names))
        
        global_exp = explainer.explain_dataset(X)
        
        assert global_exp.num_samples == 100
        assert global_exp.num_features == len(feature_names)

    def test_save_explanation(
        self,
        sample_model: object,
        feature_names: list[str],
    ) -> None:
        """Test saving explanation to file."""
        explainer = ModelExplainer(
            model=sample_model,
            feature_names=feature_names,
        )
        
        explanation = LocalExplanation(
            predicted_value=0.75,
            predicted_class="BUY",
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "explanation.json"
            explainer.save_explanation(explanation, output_path)
            
            assert output_path.exists()

    def test_fallback_explanation(
        self,
        feature_names: list[str],
    ) -> None:
        """Test fallback explanation without SHAP."""
        # Model with coefficients
        class LinearModel:
            coef_ = np.array([0.3, -0.2, 0.1, 0.05, -0.15])
            
            def predict(self, X: np.ndarray) -> np.ndarray:
                return np.array([0.6] * len(X))
        
        explainer = ModelExplainer(
            model=LinearModel(),
            feature_names=feature_names,
        )
        
        X_sample = np.array([50.0, 0.5, 0.2, 1.5, 0.1])
        explanation = explainer.explain_prediction(X_sample, 0.6)
        
        # Should have feature contributions from coefficients
        assert len(explanation.feature_contributions) > 0


class TestConvenienceFunction:
    """Test convenience function."""

    def test_explain_prediction_function(self) -> None:
        """Test the explain_prediction convenience function."""
        class MockModel:
            def predict(self, X: np.ndarray) -> np.ndarray:
                return np.array([0.7] * len(X))
        
        X_sample = np.array([0.5, 0.3, 0.8])
        feature_names = ["feat_1", "feat_2", "feat_3"]
        
        explanation = explain_prediction(
            model=MockModel(),
            X_sample=X_sample,
            feature_names=feature_names,
            prediction=0.7,
        )
        
        assert explanation.predicted_value == 0.7
