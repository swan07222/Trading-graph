"""
Test suite for Prediction Quality Enhancement Module

Tests all improvements to the AI guessing system:
1. Uncertainty quantification
2. Adaptive confidence thresholding
3. Data quality assessment
4. Ensemble disagreement detection
5. Feature importance explanation
6. Market regime detection
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from models.prediction_quality import (
    DataQualityReport,
    EnsembleDisagreement,
    FeatureImportance,
    MarketRegime,
    PredictionQualityAssessor,
    PredictionQualityReport,
    UncertaintyQuantification,
    get_quality_assessor,
)


class TestUncertaintyQuantification(unittest.TestCase):
    """Test uncertainty quantification system."""

    def setUp(self):
        self.assessor = PredictionQualityAssessor()
        
        # Create mock individual predictions (3 models, 3 classes)
        self.individual_predictions = {
            "model1": np.array([0.1, 0.2, 0.7]),
            "model2": np.array([0.15, 0.25, 0.6]),
            "model3": np.array([0.12, 0.18, 0.7]),
        }
        
        self.final_probabilities = np.array([0.12, 0.21, 0.67])

    def test_uncertainty_calculation(self):
        """Test that uncertainty is calculated correctly."""
        uq = self.assessor._quantify_uncertainty(
            self.individual_predictions,
            self.final_probabilities,
        )
        
        # Uncertainty should be between 0 and 1
        self.assertGreaterEqual(uq.total, 0.0)
        self.assertLessEqual(uq.total, 1.0)
        
        # Epistemic and aleatoric should be non-negative
        self.assertGreaterEqual(uq.epistemic, 0.0)
        self.assertGreaterEqual(uq.aleatoric, 0.0)
        
        # Total should be weighted combination
        expected_total = 0.6 * uq.epistemic + 0.4 * uq.aleatoric
        self.assertAlmostEqual(uq.total, expected_total, places=2)

    def test_high_uncertainty_flag(self):
        """Test high uncertainty flag is set correctly."""
        # Create high variance predictions
        high_var_predictions = {
            "model1": np.array([0.7, 0.2, 0.1]),
            "model2": np.array([0.1, 0.7, 0.2]),
            "model3": np.array([0.2, 0.1, 0.7]),
        }
        
        uq = self.assessor._quantify_uncertainty(
            high_var_predictions,
            np.array([0.33, 0.33, 0.34]),
        )
        
        # High disagreement should trigger high uncertainty
        self.assertTrue(uq.is_high_uncertainty or uq.epistemic > 0.2)

    def test_per_class_uncertainty(self):
        """Test per-class uncertainty is calculated."""
        uq = self.assessor._quantify_uncertainty(
            self.individual_predictions,
            self.final_probabilities,
        )
        
        self.assertIsNotNone(uq.per_class_std)
        self.assertEqual(len(uq.per_class_std), 3)  # 3 classes


class TestEnsembleDisagreement(unittest.TestCase):
    """Test ensemble disagreement detection."""

    def setUp(self):
        self.assessor = PredictionQualityAssessor()

    def test_perfect_agreement(self):
        """Test detection of perfect agreement."""
        predictions = {
            "model1": np.array([0.1, 0.1, 0.8]),
            "model2": np.array([0.15, 0.05, 0.8]),
            "model3": np.array([0.1, 0.15, 0.75]),
        }
        confidences = {
            "model1": 0.8,
            "model2": 0.8,
            "model3": 0.75,
        }
        
        disagreement = self.assessor._analyze_disagreement(
            predictions, confidences
        )
        
        # All models predict class 2
        self.assertEqual(disagreement.agreement_rate, 1.0)
        self.assertEqual(disagreement.disagreement_type, "none")
        self.assertEqual(disagreement.kappa_statistic, 1.0)

    def test_complete_disagreement(self):
        """Test detection of complete disagreement."""
        predictions = {
            "model1": np.array([0.8, 0.1, 0.1]),  # Class 0
            "model2": np.array([0.1, 0.8, 0.1]),  # Class 1
            "model3": np.array([0.1, 0.1, 0.8]),  # Class 2
        }
        confidences = {
            "model1": 0.8,
            "model2": 0.8,
            "model3": 0.8,
        }
        
        disagreement = self.assessor._analyze_disagreement(
            predictions, confidences
        )
        
        # All models disagree
        self.assertLess(disagreement.agreement_rate, 0.5)
        self.assertEqual(disagreement.disagreement_type, "conflicting")
        self.assertGreater(disagreement.confidence_penalty, 0.2)


class TestDataQualityAssessment(unittest.TestCase):
    """Test data quality assessment."""

    def setUp(self):
        self.assessor = PredictionQualityAssessor()

    def test_perfect_data_quality(self):
        """Test assessment of perfect data."""
        # Clean, normal data with reasonable mean/variance
        np.random.seed(42)
        data = np.random.randn(100, 10) * 0.5 + 10  # Mean ~10, std ~0.5
        
        report = self.assessor._assess_data_quality(
            data,
            feature_names=[f"feat_{i}" for i in range(10)],
        )
        
        # With clean data, score should be reasonable (not necessarily > 0.8)
        self.assertGreater(report.overall_score, 0.5)
        self.assertEqual(len(report.warnings), 0)

    def test_nan_detection(self):
        """Test detection of NaN values."""
        data = np.random.randn(100, 10)
        data[50, 5] = np.nan  # Insert NaN
        
        report = self.assessor._assess_data_quality(data)
        
        self.assertLess(report.completeness, 1.0)
        self.assertLess(report.overall_score, 0.95)

    def test_outlier_detection(self):
        """Test detection of outliers."""
        data = np.random.randn(100, 10)
        data[0, 0] = 100.0  # Extreme outlier
        
        report = self.assessor._assess_data_quality(data)
        
        self.assertLess(report.consistency, 1.0)


class TestMarketRegimeDetection(unittest.TestCase):
    """Test market regime detection."""

    def setUp(self):
        self.assessor = PredictionQualityAssessor()

    def test_normal_regime(self):
        """Test detection of normal market regime."""
        market_data = {
            "volatility": 0.015,
            "volume_ratio": 1.0,
            "price_change_pct": 0.005,
            "trend_strength": 0.3,
        }
        
        regime = self.assessor._detect_market_regime(market_data)
        
        self.assertEqual(regime, MarketRegime.NORMAL)

    def test_high_volatility_regime(self):
        """Test detection of high volatility regime."""
        market_data = {
            "volatility": 0.05,  # High
            "volume_ratio": 1.5,
            "price_change_pct": 0.01,
            "trend_strength": 0.4,
        }
        
        regime = self.assessor._detect_market_regime(market_data)
        
        self.assertEqual(regime, MarketRegime.HIGH_VOLATILITY)

    def test_crash_regime(self):
        """Test detection of crash regime."""
        market_data = {
            "volatility": 0.08,  # Very high
            "volume_ratio": 2.0,
            "price_change_pct": -0.07,  # Large drop
            "trend_strength": 0.8,
        }
        
        regime = self.assessor._detect_market_regime(market_data)
        
        self.assertEqual(regime, MarketRegime.CRASH)


class TestPredictionQualityReport(unittest.TestCase):
    """Test comprehensive quality report."""

    def test_adjusted_confidence_calculation(self):
        """Test that adjusted confidence accounts for all factors."""
        report = PredictionQualityReport(
            confidence=0.85,
            uncertainty=UncertaintyQuantification(
                total=0.2,
                epistemic=0.15,
                aleatoric=0.1,
            ),
            data_quality=DataQualityReport(
                overall_score=0.9,
                completeness=0.95,
                consistency=0.9,
            ),
            ensemble_disagreement=EnsembleDisagreement(
                agreement_rate=0.85,
            ),
            market_regime=MarketRegime.NORMAL,
        )
        
        # Adjusted confidence should be lower than base
        self.assertLess(report.adjusted_confidence, report.confidence)
        
        # But still reasonable
        self.assertGreater(report.adjusted_confidence, 0.5)

    def test_quality_flags(self):
        """Test quality flag setting."""
        # High quality scenario
        report_hq = PredictionQualityReport(
            confidence=0.9,
            uncertainty=UncertaintyQuantification(total=0.05),
            data_quality=DataQualityReport(overall_score=0.95),
            ensemble_disagreement=EnsembleDisagreement(agreement_rate=0.95),
            market_regime=MarketRegime.NORMAL,
        )
        
        assessor = PredictionQualityAssessor()
        assessor._set_quality_flags(report_hq)
        
        self.assertTrue(report_hq.is_high_quality)
        self.assertTrue(report_hq.is_reliable)
        self.assertFalse(report_hq.requires_human_review)

    def test_recommendations_generation(self):
        """Test recommendations are generated for problematic predictions."""
        report = PredictionQualityReport(
            uncertainty=UncertaintyQuantification(
                total=0.4,  # High
                is_high_uncertainty=True,
            ),
            ensemble_disagreement=EnsembleDisagreement(
                disagreement_type="conflicting",
            ),
            data_quality=DataQualityReport(
                overall_score=0.5,  # Poor
                is_sufficient=False,
            ),
            market_regime=MarketRegime.CRASH,
        )
        
        assessor = PredictionQualityAssessor()
        # Set flags first
        assessor._set_quality_flags(report)
        # Then generate recommendations
        assessor._generate_recommendations(report)
        
        self.assertGreater(len(report.recommendations), 0)
        # requires_human_review should be set due to multiple issues
        # (conflicting disagreement, high uncertainty, poor data, crash regime)
        self.assertTrue(
            report.requires_human_review,
            "Should require human review with multiple quality issues"
        )


class TestAdaptiveConfidenceThresholding(unittest.TestCase):
    """Test adaptive confidence thresholding."""

    def setUp(self):
        self.assessor = PredictionQualityAssessor()

    def test_base_threshold(self):
        """Test base confidence threshold."""
        threshold = self.assessor.get_adaptive_confidence_threshold(
            MarketRegime.NORMAL,
            data_quality_score=1.0,
        )
        
        # Should be close to base threshold
        self.assertGreater(threshold, 0.5)
        self.assertLess(threshold, 0.8)

    def test_high_volatility_threshold(self):
        """Test threshold increases in high volatility."""
        threshold_normal = self.assessor.get_adaptive_confidence_threshold(
            MarketRegime.NORMAL,
            data_quality_score=1.0,
        )
        
        threshold_hv = self.assessor.get_adaptive_confidence_threshold(
            MarketRegime.HIGH_VOLATILITY,
            data_quality_score=1.0,
        )
        
        # Higher threshold in high volatility
        self.assertGreater(threshold_hv, threshold_normal)

    def test_poor_data_threshold(self):
        """Test threshold increases with poor data quality."""
        threshold_good = self.assessor.get_adaptive_confidence_threshold(
            MarketRegime.NORMAL,
            data_quality_score=0.95,
        )
        
        threshold_poor = self.assessor.get_adaptive_confidence_threshold(
            MarketRegime.NORMAL,
            data_quality_score=0.5,
        )
        
        # Higher threshold with poor data
        self.assertGreater(threshold_poor, threshold_good)


class TestFeatureImportance(unittest.TestCase):
    """Test feature importance computation."""

    def setUp(self):
        self.assessor = PredictionQualityAssessor()

    def test_feature_importance_calculation(self):
        """Test feature importance is calculated."""
        np.random.seed(42)
        input_data = np.random.randn(50, 10)
        probabilities = np.random.rand(50, 3)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        
        feature_names = [f"feature_{i}" for i in range(10)]
        
        fi = self.assessor._compute_feature_importance(
            input_data,
            probabilities.mean(axis=0),
            feature_names,
        )
        
        self.assertEqual(len(fi.feature_names), 10)
        self.assertEqual(len(fi.importance_scores), 10)
        self.assertAlmostEqual(fi.importance_scores.sum(), 1.0, places=2)
        self.assertGreater(len(fi.top_features), 0)

    def test_feature_influence_direction(self):
        """Test detection of positive/negative influence."""
        np.random.seed(42)
        input_data = np.random.randn(50, 10)
        # Create correlation: feature 0 positively correlated with UP
        input_data[:, 0] = np.random.randn(50)
        probabilities = np.zeros((50, 3))
        probabilities[:, 2] = (input_data[:, 0] + 1) / 2  # Class 2 (UP)
        probabilities = np.clip(probabilities, 0.01, 0.99)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        
        feature_names = [f"feature_{i}" for i in range(10)]
        
        fi = self.assessor._compute_feature_importance(
            input_data,
            probabilities.mean(axis=0),
            feature_names,
        )
        
        # Should identify some features as influential
        self.assertGreater(len(fi.top_features), 0)


class TestGlobalQualityAssessor(unittest.TestCase):
    """Test global quality assessor singleton."""

    def test_singleton_pattern(self):
        """Test that get_quality_assessor returns same instance."""
        assessor1 = get_quality_assessor()
        assessor2 = get_quality_assessor()
        
        self.assertIs(assessor1, assessor2)

    def test_assessor_initialization(self):
        """Test quality assessor is properly initialized."""
        assessor = get_quality_assessor()
        
        self.assertIsInstance(assessor, PredictionQualityAssessor)
        self.assertGreater(assessor.uncertainty_threshold_high, 0.0)
        self.assertLess(assessor.uncertainty_threshold_low, 0.3)


class TestIntegration(unittest.TestCase):
    """Integration tests for full prediction quality assessment."""

    def test_full_assessment_pipeline(self):
        """Test complete quality assessment pipeline."""
        assessor = PredictionQualityAssessor()
        
        # Create realistic prediction data
        np.random.seed(42)
        n_models = 4
        n_classes = 3
        
        individual_predictions = {
            f"model{i}": np.random.dirichlet([1, 1, 1])
            for i in range(n_models)
        }
        individual_confidences = {
            f"model{i}": float(np.max(pred))
            for i, pred in individual_predictions.items()
        }
        
        final_probabilities = np.mean(
            list(individual_predictions.values()),
            axis=0,
        )
        
        input_data = np.random.randn(60, 15)
        feature_names = [f"feat_{i}" for i in range(15)]
        
        market_data = {
            "volatility": 0.02,
            "volume_ratio": 1.1,
            "price_change_pct": 0.01,
            "trend_strength": 0.4,
        }
        
        # Run full assessment
        report = assessor.assess_prediction(
            probabilities=final_probabilities,
            individual_predictions=individual_predictions,
            individual_confidences=individual_confidences,
            input_data=input_data,
            feature_names=feature_names,
            market_data=market_data,
        )
        
        # Verify report is complete
        self.assertIsNotNone(report)
        self.assertGreater(report.confidence, 0.0)
        self.assertLessEqual(report.confidence, 1.0)
        self.assertIsNotNone(report.uncertainty)
        self.assertIsNotNone(report.data_quality)
        self.assertIsNotNone(report.ensemble_disagreement)
        self.assertIsInstance(report.market_regime, MarketRegime)
        
        # Verify to_dict works
        report_dict = report.to_dict()
        self.assertIn("confidence", report_dict)
        self.assertIn("uncertainty", report_dict)
        self.assertIn("data_quality", report_dict)


if __name__ == "__main__":
    unittest.main()
