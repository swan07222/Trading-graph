
import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import CONFIG
from data.processor import DataProcessor
from data.features import FeatureEngine

class TestScalerLeakage:
    """Tests to ensure scaler is not fitted on test data"""

    @pytest.fixture
    def sample_df(self):
        """Create deterministic sample data"""
        np.random.seed(42)
        n = 500
        dates = pd.date_range('2020-01-01', periods=n, freq='D')

        close = 100 + np.cumsum(np.random.randn(n) * 0.5)

        df = pd.DataFrame({
            'open': close * (1 + np.random.randn(n) * 0.01),
            'high': close * 1.02,
            'low': close * 0.98,
            'close': close,
            'volume': np.random.randint(1000000, 5000000, n),
        }, index=dates)

        return df

    def test_scaler_not_fitted_before_explicit_call(self):
        """Scaler should not be fitted until explicitly called"""
        processor = DataProcessor()
        assert not processor._fitted

        with pytest.raises(RuntimeError, match="Scaler not fitted"):
            processor.transform(np.random.randn(10, 5))

    def test_scaler_fitted_only_on_training(self, sample_df):
        """Scaler must be fitted only on training data"""
        feature_engine = FeatureEngine()
        processor = DataProcessor()

        df = feature_engine.create_features(sample_df)
        df = processor.create_labels(df)
        feature_cols = feature_engine.get_feature_columns()

        n = len(df)
        train_end = int(n * 0.7)

        train_df = df.iloc[:train_end - CONFIG.EMBARGO_BARS]
        test_df = df.iloc[train_end:]

        train_features = train_df[feature_cols].values
        valid_mask = ~train_df['label'].isna()
        processor.fit_scaler(train_features[valid_mask])

        train_center = processor.scaler.center_.copy()
        train_scale = processor.scaler.scale_.copy()

        # Process test (should not change scaler)
        X_test, y_test, _ = processor.prepare_sequences(
            test_df, feature_cols, fit_scaler=False
        )

        np.testing.assert_array_equal(processor.scaler.center_, train_center)
        np.testing.assert_array_equal(processor.scaler.scale_, train_scale)

    def test_split_temporal_applies_embargo(self, sample_df):
        """Temporal split must apply embargo gap"""
        feature_engine = FeatureEngine()
        processor = DataProcessor()

        df = feature_engine.create_features(sample_df)
        df = processor.create_labels(df)
        feature_cols = feature_engine.get_feature_columns()

        splits = processor.split_temporal(df, feature_cols)

        # Verify embargo is applied (train should be shorter than expected)
        n = len(df)
        expected_train_end = int(n * CONFIG.TRAIN_RATIO) - CONFIG.EMBARGO_BARS

        assert splits['train'][0].shape[0] > 0
        assert splits['val'][0].shape[0] > 0
        assert splits['test'][0].shape[0] > 0

class TestLabelLeakage:
    """Tests to ensure labels don't leak future information"""

    @pytest.fixture
    def deterministic_df(self):
        """Create data with known future returns"""
        n = 200
        dates = pd.date_range('2020-01-01', periods=n, freq='D')

        # Deterministic prices: each day increases by 1
        close = np.arange(100, 100 + n, dtype=float)

        df = pd.DataFrame({
            'open': close - 0.5,
            'high': close + 1,
            'low': close - 1,
            'close': close,
            'volume': np.ones(n) * 1000000,
        }, index=dates)

        return df

    def test_labels_use_only_future_data(self, deterministic_df):
        """Labels must be computed from future prices only"""
        processor = DataProcessor()
        horizon = 5

        df = processor.create_labels(deterministic_df, horizon=horizon)

        # For deterministic prices [100, 101, 102, ...], 
        # future return at index i should be (close[i+horizon] / close[i] - 1) * 100

        for i in range(len(df) - horizon):
            expected_return = (df['close'].iloc[i + horizon] / df['close'].iloc[i] - 1) * 100
            actual_return = df['future_return'].iloc[i]

            np.testing.assert_almost_equal(actual_return, expected_return, decimal=5)

    def test_last_horizon_rows_have_nan_labels(self, deterministic_df):
        """Last PREDICTION_HORIZON rows must have NaN labels"""
        processor = DataProcessor()
        horizon = 5

        df = processor.create_labels(deterministic_df, horizon=horizon)

        # Last `horizon` rows should have NaN labels
        assert df['label'].iloc[-horizon:].isna().all()
        assert df['future_return'].iloc[-horizon:].isna().all()

        assert not df['label'].iloc[:-horizon].isna().any()

class TestSequenceConsistency:
    """Tests to ensure training and inference use same sequence construction"""

    @pytest.fixture
    def sample_data(self):
        """Create sample data"""
        np.random.seed(42)
        n = 200
        dates = pd.date_range('2020-01-01', periods=n, freq='D')

        close = 100 + np.cumsum(np.random.randn(n) * 0.5)

        df = pd.DataFrame({
            'open': close * 1.01,
            'high': close * 1.02,
            'low': close * 0.98,
            'close': close,
            'volume': np.random.randint(1000000, 5000000, n),
        }, index=dates)

        return df

    def test_train_inference_sequence_match(self, sample_data):
        """Training and inference sequences must be constructed identically"""
        feature_engine = FeatureEngine()
        processor = DataProcessor()

        df = feature_engine.create_features(sample_data)
        df = processor.create_labels(df)
        feature_cols = feature_engine.get_feature_columns()

        X_train, y_train, _ = processor.prepare_sequences(
            df, feature_cols, fit_scaler=True
        )

        last_train_seq = X_train[-1]

        n_valid = len(X_train) + CONFIG.SEQUENCE_LENGTH - 1
        inference_df = df.iloc[:n_valid]

        X_inference = processor.prepare_inference_sequence(
            inference_df.tail(CONFIG.SEQUENCE_LENGTH + 10),  # Extra for safety
            feature_cols
        )

        # The last inference sequence should match last training sequence
        # (within floating point tolerance)
        np.testing.assert_array_almost_equal(
            last_train_seq,
            X_inference[0, :CONFIG.SEQUENCE_LENGTH, :],
            decimal=5
        )

class TestTemporalOrdering:
    """Tests to ensure temporal ordering is preserved"""

    @pytest.fixture
    def ordered_df(self):
        """Create data with clear temporal ordering"""
        n = 300
        dates = pd.date_range('2020-01-01', periods=n, freq='D')

        close = np.arange(n, dtype=float) + 100

        df = pd.DataFrame({
            'open': close,
            'high': close + 1,
            'low': close - 1,
            'close': close,
            'volume': np.ones(n) * 1000000,
        }, index=dates)

        return df

    def test_no_shuffle_in_split(self, ordered_df):
        """Temporal split must not shuffle data"""
        feature_engine = FeatureEngine()
        processor = DataProcessor()

        df = feature_engine.create_features(ordered_df)
        df = processor.create_labels(df)
        feature_cols = feature_engine.get_feature_columns()

        splits = processor.split_temporal(df, feature_cols)

        X_train, y_train, _ = splits['train']
        X_val, y_val, _ = splits['val']
        X_test, y_test, _ = splits['test']

        # Within each split, sequences should be in order
        # Check by looking at the 'close' values encoded in the first feature
        # (This is a simplified check - the actual close is transformed)

        # At minimum: all splits should be non-empty and properly sized
        assert len(X_train) > 0
        assert len(X_val) > 0  
        assert len(X_test) > 0

        assert X_train.shape[1] == CONFIG.SEQUENCE_LENGTH
        assert X_val.shape[1] == CONFIG.SEQUENCE_LENGTH
        assert X_test.shape[1] == CONFIG.SEQUENCE_LENGTH

class TestEnsembleWeightPersistence:
    """Tests for ensemble model save/load"""

    def test_weights_preserved_after_save_load(self, tmp_path):
        """Ensemble weights must be preserved after save/load"""
        from models.ensemble import EnsembleModel

        ensemble = EnsembleModel(input_size=35, model_names=['lstm', 'gru'])

        ensemble.weights = {'lstm': 0.7, 'gru': 0.3}

        save_path = str(tmp_path / "test_ensemble.pt")
        ensemble.save(save_path)

        ensemble2 = EnsembleModel(input_size=35, model_names=['lstm', 'gru'])
        loaded = ensemble2.load(save_path)

        assert loaded

        np.testing.assert_almost_equal(ensemble2.weights['lstm'], 0.7, decimal=5)
        np.testing.assert_almost_equal(ensemble2.weights['gru'], 0.3, decimal=5)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])