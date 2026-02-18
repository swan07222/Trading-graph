
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import CONFIG
from data.features import FeatureEngine
from data.fetcher import DataFetcher, YahooSource
from data.processor import DataProcessor


class TestDataFetcher:
    """Tests for DataFetcher"""

    @pytest.fixture
    def fetcher(self):
        return DataFetcher()

    def test_fetcher_initialization(self, fetcher):
        """Test that fetcher initializes with sources"""
        assert fetcher is not None
        assert len(fetcher._sources) > 0

    def test_clean_code(self, fetcher):
        """Test stock code cleaning"""
        assert DataFetcher.clean_code("600519") == "600519"
        assert DataFetcher.clean_code("sh600519") == "600519"
        assert DataFetcher.clean_code("600519.SS") == "600519"
        assert DataFetcher.clean_code("519") == "000519"

    def test_get_history_returns_dataframe(self, fetcher):
        """Test that get_history returns a DataFrame"""
        df = fetcher.get_history("600519", days=100)

        if not df.empty:
            assert isinstance(df, pd.DataFrame)
            assert 'close' in df.columns
            assert 'volume' in df.columns
            assert len(df) > 0

    def test_cache_functionality(self, fetcher):
        """Test that caching works"""
        df1 = fetcher.get_history("600519", days=50)

        if not df1.empty:
            df2 = fetcher.get_history("600519", days=50, use_cache=True)

            assert len(df1) == len(df2)

    def test_yahoo_intraday_keeps_bar_count(self, monkeypatch):
        """Intraday Yahoo fetch should not be truncated to calendar days."""
        rows = 240
        idx = pd.date_range("2026-01-01", periods=rows, freq="min")
        raw = pd.DataFrame(
            {
                "Open": np.linspace(10.0, 20.0, rows),
                "High": np.linspace(10.5, 20.5, rows),
                "Low": np.linspace(9.5, 19.5, rows),
                "Close": np.linspace(10.2, 20.2, rows),
                "Volume": np.full(rows, 1000),
            },
            index=idx,
        )

        class _Ticker:
            def history(self, **kwargs):
                return raw.copy()

        class _YF:
            def Ticker(self, _):
                return _Ticker()

        src = YahooSource()
        src._yf = _YF()
        monkeypatch.setattr(src, "is_available", lambda: True)

        inst = {"market": "CN", "asset": "EQUITY", "symbol": "000001"}
        out = src.get_history_instrument(inst, days=7, interval="1m")

        assert len(out) == rows

    def test_resolve_intraday_days_to_bar_depth(self):
        bars = DataFetcher._resolve_requested_bar_count(
            days=7,
            bars=None,
            interval="1m",
        )
        assert bars >= 1000

    def test_resolve_weekly_monthly_days_to_bar_depth(self):
        bars_weekly = DataFetcher._resolve_requested_bar_count(
            days=500,
            bars=None,
            interval="1wk",
        )
        bars_monthly = DataFetcher._resolve_requested_bar_count(
            days=500,
            bars=None,
            interval="1mo",
        )
        assert bars_weekly == 100
        assert bars_monthly == 25

class TestDataProcessor:
    """Tests for DataProcessor"""

    @pytest.fixture
    def processor(self):
        return DataProcessor()

    @pytest.fixture
    def sample_df(self):
        """Create sample DataFrame"""
        dates = pd.date_range('2023-01-01', periods=200, freq='D')
        np.random.seed(42)

        close = 100 + np.cumsum(np.random.randn(200) * 0.5)

        df = pd.DataFrame({
            'open': close * (1 + np.random.randn(200) * 0.01),
            'high': close * 1.02,
            'low': close * 0.98,
            'close': close,
            'volume': np.random.randint(1000000, 5000000, 200),
            'amount': close * np.random.randint(1000000, 5000000, 200)
        }, index=dates)

        return df

    def test_create_labels(self, processor, sample_df):
        """Test label creation"""
        df = processor.create_labels(sample_df)

        assert 'label' in df.columns
        assert 'future_return' in df.columns

        # Labels should be 0, 1, or 2, or NaN (for last HORIZON rows)
        valid_labels = df['label'].dropna()
        assert valid_labels.isin([0, 1, 2]).all()

        # Last PREDICTION_HORIZON rows should have NaN labels
        assert df['label'].iloc[-CONFIG.PREDICTION_HORIZON:].isna().all()

        # NOTE: create_labels does NOT drop rows, it sets them to NaN
        # So len(df) == len(sample_df)
        assert len(df) == len(sample_df)

    def test_scaler_fitting(self, processor):
        """Test scaler fitting"""
        features = np.random.randn(100, 10)

        processor.fit_scaler(features)

        assert processor._fitted
        assert processor.scaler is not None

    def test_transform_requires_fit(self, processor):
        """Test that transform fails without fit"""
        features = np.random.randn(100, 10)

        with pytest.raises(RuntimeError):
            processor.transform(features)

    def test_transform_clips_values(self, processor):
        """Test that transform clips extreme values"""
        features = np.random.randn(100, 10)
        processor.fit_scaler(features)

        extreme = np.array([[100, -100] + [0] * 8])
        transformed = processor.transform(extreme)

        assert np.all(transformed >= -5)
        assert np.all(transformed <= 5)

    def test_save_load_scaler(self, processor, tmp_path):
        """Test scaler save/load"""
        features = np.random.randn(100, 10)
        processor.fit_scaler(features)

        path = str(tmp_path / "scaler.pkl")
        processor.save_scaler(path)

        processor2 = DataProcessor()
        loaded = processor2.load_scaler(path)

        assert loaded
        assert processor2._fitted

class TestFeatureEngine:
    """Tests for FeatureEngine"""

    @pytest.fixture
    def engine(self):
        return FeatureEngine()

    @pytest.fixture
    def sample_ohlcv(self):
        """Create sample OHLCV data"""
        dates = pd.date_range('2023-01-01', periods=200, freq='D')
        np.random.seed(42)

        close = 100 + np.cumsum(np.random.randn(200) * 0.5)

        return pd.DataFrame({
            'open': close * (1 + np.random.randn(200) * 0.01),
            'high': close * 1.02,
            'low': close * 0.98,
            'close': close,
            'volume': np.random.randint(1000000, 5000000, 200),
        }, index=dates)

    def test_create_features(self, engine, sample_ohlcv):
        """Test feature creation"""
        df = engine.create_features(sample_ohlcv)

        feature_cols = engine.get_feature_columns()

        for col in feature_cols[:5]:  # Check first 5
            assert col in df.columns, f"Missing feature: {col}"

    def test_no_nan_in_features(self, engine, sample_ohlcv):
        """Test that features don't have NaN after processing"""
        df = engine.create_features(sample_ohlcv)
        feature_cols = engine.get_feature_columns()

        for col in feature_cols:
            if col in df.columns:
                assert not df[col].isna().any(), f"NaN in {col}"

    def test_feature_count(self, engine):
        """Test feature count matches expected"""
        feature_cols = engine.get_feature_columns()
        assert len(feature_cols) >= 30  # Should have many features

class TestIntegration:
    """Integration tests for the data pipeline"""

    def test_full_pipeline(self):
        """Test complete data pipeline"""
        fetcher = DataFetcher()
        processor = DataProcessor()
        engine = FeatureEngine()

        df = fetcher.get_history("600519", days=300)

        if df.empty:
            pytest.skip("No data available")

        df = engine.create_features(df)

        df = processor.create_labels(df)

        feature_cols = engine.get_feature_columns()

        X, y, r = processor.prepare_sequences(df, feature_cols, fit_scaler=True)

        assert len(X) > 0
        assert len(y) == len(X)
        assert X.shape[2] == len(feature_cols)

        assert not np.isnan(X).any()
        assert not np.isnan(y).any()

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
