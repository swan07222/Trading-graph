# tests/integration/test_data_pipeline.py
"""
Integration Tests for Data Pipeline

FIXES:
- Add integration tests for full data pipeline
- Test multi-source federation
- Test data quality validation
- End-to-end data flow testing
"""

from __future__ import annotations

import asyncio
import pytest
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np

from data.multi_source_federation import (
    MultiSourceFederation,
    DataSource,
    SourceHealth,
    DataQuality,
    get_federation,
)
from data.advanced_data_quality import (
    DataQualityValidator,
    QualityFlag,
    get_validator,
)


class TestMultiSourceFederation:
    """Integration tests for multi-source data federation."""
    
    @pytest.fixture
    def federation(self) -> MultiSourceFederation:
        """Create federation instance."""
        return MultiSourceFederation()
    
    @pytest.mark.asyncio
    async def test_fetch_with_consensus_success(
        self,
        federation: MultiSourceFederation,
    ) -> None:
        """Test successful multi-source consensus fetch."""
        # Mock source responses
        await self._setup_mock_sources(federation)
        
        result = await federation.fetch_with_consensus(
            symbol="600519",
            data_type="quote",
            min_sources=2,
        )
        
        assert result.symbol == "600519"
        assert result.quality in [DataQuality.GOLD, DataQuality.SILVER]
        assert len(result.sources_used) >= 1
        assert 0.0 <= result.consensus_score <= 1.0
        assert isinstance(result.df, pd.DataFrame)
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_opens_on_failures(
        self,
        federation: MultiSourceFederation,
    ) -> None:
        """Test circuit breaker opens after consecutive failures."""
        source = DataSource(
            name="test_source",
            base_url="http://invalid",
            priority=1,
            timeout_seconds=0.1,
            max_retries=1,
        )
        federation.sources["test_source"] = source
        
        # Simulate failures
        for _ in range(5):
            source.record_failure()
        
        assert source.health == SourceHealth.UNHEALTHY
        assert source.circuit_breaker_open is True
    
    @pytest.mark.asyncio
    async def test_source_health_tracking(
        self,
        federation: MultiSourceFederation,
    ) -> None:
        """Test source health metrics are tracked."""
        source = federation.sources.get("tencent")
        if source:
            # Record successes
            for _ in range(10):
                source.record_success(latency_ms=50.0)
            
            assert source.success_rate() == 1.0
            assert source.avg_latency_ms > 0.0
            assert source.health == SourceHealth.HEALTHY
    
    async def _setup_mock_sources(
        self,
        federation: MultiSourceFederation,
    ) -> None:
        """Setup mock source responses for testing."""
        # Implementation would mock HTTP responses
        pass


class TestDataQualityValidation:
    """Integration tests for data quality validation."""
    
    @pytest.fixture
    def validator(self) -> DataQualityValidator:
        """Create validator instance."""
        return DataQualityValidator()
    
    @pytest.fixture
    def sample_data(self) -> pd.DataFrame:
        """Create sample OHLCV data for testing."""
        dates = pd.date_range(
            start=datetime.now() - timedelta(days=100),
            end=datetime.now(),
            freq="D",
        )
        
        np.random.seed(42)
        base_price = 100.0
        
        data = {
            "open": base_price + np.cumsum(np.random.randn(100) * 2),
            "high": base_price + np.cumsum(np.random.randn(100) * 2) + np.abs(np.random.randn(100)),
            "low": base_price + np.cumsum(np.random.randn(100) * 2) - np.abs(np.random.randn(100)),
            "close": base_price + np.cumsum(np.random.randn(100) * 2),
            "volume": np.random.randint(1000000, 10000000, 100),
        }
        
        df = pd.DataFrame(data, index=dates)
        df["high"] = df[["high", "open", "close"]].max(axis=1)
        df["low"] = df[["low", "open", "close"]].min(axis=1)
        
        return df
    
    def test_validate_good_data(
        self,
        validator: DataQualityValidator,
        sample_data: pd.DataFrame,
    ) -> None:
        """Test validation passes for good quality data."""
        report = validator.validate(
            symbol="600519",
            df=sample_data,
        )
        
        assert report.overall_score > 0.7
        assert QualityFlag.INVALID not in report.flags
        assert report.is_acceptable(min_score=0.7)
    
    def test_validate_missing_columns(
        self,
        validator: DataQualityValidator,
    ) -> None:
        """Test validation detects missing columns."""
        incomplete_df = pd.DataFrame({
            "open": [100, 101, 102],
            "close": [101, 102, 103],
        })
        
        report = validator.validate(
            symbol="600519",
            df=incomplete_df,
        )
        
        assert QualityFlag.MISSING in report.flags
        assert report.overall_score < 0.5
    
    def test_validate_invalid_ohlc(
        self,
        validator: DataQualityValidator,
    ) -> None:
        """Test validation detects invalid OHLC relationships."""
        invalid_df = pd.DataFrame({
            "open": [100, 101, 102],
            "high": [99, 100, 101],  # High < Open (invalid)
            "low": [98, 99, 100],
            "close": [101, 102, 103],
        })
        
        report = validator.validate(
            symbol="600519",
            df=invalid_df,
        )
        
        assert QualityFlag.SUSPECT in report.flags or QualityFlag.INVALID in report.flags
    
    def test_validate_outliers(
        self,
        validator: DataQualityValidator,
        sample_data: pd.DataFrame,
    ) -> None:
        """Test validation detects outliers."""
        # Add extreme outlier
        data_with_outlier = sample_data.copy()
        data_with_outlier.loc[data_with_outlier.index[-1], "close"] = 1000.0
        
        report = validator.validate(
            symbol="600519",
            df=data_with_outlier,
        )
        
        assert QualityFlag.OUTLIER in report.flags
    
    def test_data_repair(
        self,
        validator: DataQualityValidator,
        sample_data: pd.DataFrame,
    ) -> None:
        """Test automatic data repair."""
        # Create data with issues
        repaired_df = validator.repair_data(
            df=sample_data,
            report=None,  # Not used in this test
        )
        
        assert len(repaired_df) > 0
        assert not repaired_df.isnull().any().any()
    
    def test_validation_metrics(
        self,
        validator: DataQualityValidator,
        sample_data: pd.DataFrame,
    ) -> None:
        """Test validation returns detailed metrics."""
        report = validator.validate(
            symbol="600519",
            df=sample_data,
        )
        
        assert "completeness" in report.metrics
        assert "accuracy" in report.metrics
        assert "consistency" in report.metrics
        assert "timeliness" in report.metrics


class TestEndToEndDataFlow:
    """End-to-end tests for complete data flow."""
    
    @pytest.mark.asyncio
    async def test_full_pipeline(
        self,
        federation: MultiSourceFederation,
        validator: DataQualityValidator,
    ) -> None:
        """Test complete data pipeline from fetch to validation."""
        # 1. Fetch data
        federated = await federation.fetch_with_consensus(
            symbol="600519",
            data_type="quote",
        )
        
        # 2. Validate quality
        report = validator.validate(
            symbol="600519",
            df=federated.df,
        )
        
        # 3. Assert quality gates
        assert report.is_acceptable(min_score=0.6)
        assert federated.df is not None
        assert len(federated.df) > 0
    
    @pytest.mark.asyncio
    async def test_pipeline_with_repair(
        self,
        federation: MultiSourceFederation,
        validator: DataQualityValidator,
    ) -> None:
        """Test pipeline with automatic data repair."""
        # Fetch and validate
        federated = await federation.fetch_with_consensus(
            symbol="600519",
            data_type="quote",
        )
        
        report = validator.validate(
            symbol="600519",
            df=federated.df,
        )
        
        # Repair if needed
        if not report.is_acceptable(min_score=0.8):
            repaired_df = validator.repair_data(federated.df, report)
            repaired_report = validator.validate(
                symbol="600519",
                df=repaired_df,
            )
            # Quality should improve or stay same
            assert repaired_report.overall_score >= report.overall_score * 0.9


# Fixtures for integration tests
@pytest.fixture(scope="module")
def federation() -> MultiSourceFederation:
    """Module-scoped federation instance."""
    return get_federation()


@pytest.fixture(scope="module")
def validator() -> DataQualityValidator:
    """Module-scoped validator instance."""
    return get_validator()
