# tests/test_fetcher_unified.py
"""Comprehensive tests for unified data fetching pipeline.

FIX 2026-02-26: Tests for all improvements:
1. Configuration
2. Source health monitoring
3. Progressive loading
4. Timezone handling
5. Data validation
6. Unified fetcher
"""

import pandas as pd
import pytest
from datetime import datetime, time
from unittest.mock import Mock, patch, MagicMock

# Import new modules
from data.fetcher_config import FetcherConfig, get_config, TimeoutConfig, RetryConfig
from data.source_health import (
    DataSourceHealthMonitor,
    SourceHealthStatus,
    SourceHealthState,
    get_health_monitor,
    reset_health_monitor,
)
from data.timezone_utils import (
    TradingSessionChecker,
    TimezoneConverter,
    is_trading_time,
    filter_trading_hours,
)
from data.progressive_loader import (
    ProgressiveDataLoader,
    LoadResult,
    LoadStatus,
    get_progressive_loader,
)
from data.fetcher_unified import (
    UnifiedDataFetcher,
    FetchOptions,
    FetchResult,
    get_unified_fetcher,
)
import data.fetcher_unified as unified_mod
from data.validator import DataValidator, ValidationResult, get_validator


class TestFetcherConfig:
    """Tests for unified fetcher configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = FetcherConfig()
        
        assert config.timeout.connect_timeout > 0
        assert config.timeout.read_timeout > 0
        assert config.retry.max_retries > 0
        assert config.cache.default_ttl > 0
        assert config.rate_limit.default_rate > 0
        assert config.circuit_breaker.failure_threshold > 0
        assert config.quality.min_quality_score_daily > 0
    
    def test_config_to_dict(self):
        """Test config serialization."""
        config = FetcherConfig()
        config_dict = config.to_dict()
        
        assert "timeout" in config_dict
        assert "retry" in config_dict
        assert "cache" in config_dict
        assert "rate_limit" in config_dict
        assert "circuit_breaker" in config_dict
    
    def test_get_config_singleton(self):
        """Test global config singleton."""
        config1 = get_config()
        config2 = get_config()
        
        assert config1 is config2


class TestSourceHealthMonitor:
    """Tests for data source health monitoring."""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Reset health monitor before each test."""
        reset_health_monitor()
        yield
    
    def test_initial_state(self):
        """Test initial health state."""
        monitor = get_health_monitor()
        summary = monitor.get_summary()
        
        assert summary["total_sources"] == 0
        assert summary["healthy"] == 0
        assert summary["unhealthy"] == 0
    
    def test_record_success(self):
        """Test recording successful requests."""
        monitor = DataSourceHealthMonitor()
        
        monitor.record_success("akshare", response_time=0.5)
        monitor.record_success("akshare", response_time=0.3)
        
        state = monitor._sources.get("akshare")
        assert state is not None
        assert state.success_count == 2
        assert state.consecutive_successes == 2
        assert state.status == SourceHealthStatus.UNKNOWN  # Needs more successes
    
    def test_record_failure(self):
        """Test recording failed requests."""
        monitor = DataSourceHealthMonitor()
        
        for _ in range(5):
            monitor.record_failure("akshare", error="timeout")
        
        state = monitor._sources.get("akshare")
        assert state is not None
        assert state.consecutive_failures >= 5
        assert state.status == SourceHealthStatus.UNHEALTHY
    
    def test_health_score_calculation(self):
        """Test health score calculation."""
        state = SourceHealthState(source="test")
        
        # Initial state
        assert state.health_score == 1.0
        
        # Add some successes
        state.record_success(0.5)
        state.record_success(0.3)
        assert state.health_score > 0.8
        
        # Add failures
        for _ in range(3):
            state.record_failure("error")
        assert state.health_score < 0.5
    
    def test_is_available(self):
        """Test source availability check."""
        state = SourceHealthState(source="test")
        
        # Healthy source is available
        assert state.is_available()
        
        # Unhealthy source with cooldown
        state.status = SourceHealthStatus.UNHEALTHY
        state.cooldown_until = float('inf')
        assert not state.is_available()
    
    def test_auto_failover(self):
        """Test automatic failover to healthy source."""
        monitor = DataSourceHealthMonitor()
        
        # Make primary source unhealthy
        for _ in range(5):
            monitor.record_failure("akshare")
        
        # Make secondary source healthy
        monitor.record_success("sina", response_time=0.3)
        
        # Should failover to sina
        healthy_source = monitor.get_healthy_source(preferred="akshare")
        assert healthy_source != "akshare"
        assert healthy_source == "sina"


class TestTradingSessionChecker:
    """Tests for trading session handling."""
    
    def test_trading_hours_detection(self):
        """Test trading hours detection."""
        checker = TradingSessionChecker()
        
        # During morning session
        morning_ts = datetime(2024, 3, 15, 10, 30)  # Wednesday 10:30
        assert checker.is_trading_time(morning_ts)
        
        # During afternoon session
        afternoon_ts = datetime(2024, 3, 15, 14, 0)  # Wednesday 14:00
        assert checker.is_trading_time(afternoon_ts)
        
        # Lunch break
        lunch_ts = datetime(2024, 3, 15, 12, 0)  # Wednesday 12:00
        assert not checker.is_trading_time(lunch_ts)
        
        # After hours
        evening_ts = datetime(2024, 3, 15, 20, 0)  # Wednesday 20:00
        assert not checker.is_trading_time(evening_ts)
    
    def test_weekend_detection(self):
        """Test weekend detection."""
        checker = TradingSessionChecker()
        
        # Saturday
        saturday = datetime(2024, 3, 16, 10, 0)
        assert not checker.is_trading_time(saturday)
        
        # Sunday
        sunday = datetime(2024, 3, 17, 10, 0)
        assert not checker.is_trading_time(sunday)
    
    def test_filter_trading_hours(self):
        """Test filtering DataFrame to trading hours."""
        checker = TradingSessionChecker()
        
        # Create test data with mixed times
        dates = pd.date_range("2024-03-15 09:00", "2024-03-15 15:30", freq="30min")
        df = pd.DataFrame({"close": range(len(dates))}, index=dates)
        
        filtered = checker.filter_trading_hours(df)
        
        # Should only keep trading hours
        for ts in filtered.index:
            assert checker.is_trading_time(ts.to_pydatetime())


class TestProgressiveLoader:
    """Tests for progressive data loading."""
    
    def test_load_minimum_data(self):
        """Test loading minimum required data."""
        loader = ProgressiveDataLoader(
            min_bars_daily=10,
            chunk_size=5,
        )
        
        # Mock fetch function that returns increasing data
        call_count = [0]
        
        def mock_fetch(bars: int) -> pd.DataFrame:
            call_count[0] += 1
            n = min(bars, 15)  # Return 15 bars max
            dates = pd.date_range("2024-01-01", periods=n, freq="1d")
            return pd.DataFrame({
                "open": range(100, 100 + n),
                "high": range(101, 101 + n),
                "low": range(99, 99 + n),
                "close": range(100, 100 + n),
                "volume": range(1000, 1000 + n),
            }, index=dates)
        
        result = loader.load(mock_fetch, interval="1d", requested_bars=20)
        
        assert result.status in [LoadStatus.COMPLETE, LoadStatus.PARTIAL, LoadStatus.MINIMUM]
        assert result.bars_loaded > 0
        assert result.quality_score > 0
    
    def test_load_insufficient_data(self):
        """Test handling of insufficient data."""
        loader = ProgressiveDataLoader(
            min_bars_daily=100,
            allow_partial=False,
        )
        
        def mock_fetch(bars: int) -> pd.DataFrame:
            # Return very little data
            dates = pd.date_range("2024-01-01", periods=5, freq="1d")
            return pd.DataFrame({
                "open": range(100, 105),
                "high": range(101, 106),
                "low": range(99, 104),
                "close": range(100, 105),
                "volume": range(1000, 1005),
            }, index=dates)
        
        result = loader.load(mock_fetch, interval="1d", requested_bars=100)
        
        assert result.status == LoadStatus.INSUFFICIENT
        assert result.data is not None  # Still returns available data
    
    def test_quality_score_calculation(self):
        """Test quality score calculation."""
        loader = ProgressiveDataLoader()
        
        # Good data
        good_df = pd.DataFrame({
            "open": [100, 101, 102],
            "high": [102, 103, 104],
            "low": [99, 100, 101],
            "close": [101, 102, 103],
            "volume": [1000, 1100, 1200],
        }, index=pd.date_range("2024-01-01", periods=3))
        
        good_score = loader._calculate_quality_score(good_df, "1d")
        assert good_score > 0.8
        
        # Bad data (with NaN)
        bad_df = good_df.copy()
        bad_df.loc[0, "close"] = float('nan')
        
        bad_score = loader._calculate_quality_score(bad_df, "1d")
        assert bad_score < good_score


class TestUnifiedDataFetcher:
    """Tests for unified data fetcher."""
    
    @pytest.fixture
    def fetcher(self):
        """Create test fetcher instance."""
        return UnifiedDataFetcher()
    
    def test_fetch_options_defaults(self):
        """Test default fetch options."""
        options = FetchOptions()
        
        assert options.interval == "1d"
        assert options.validate is True
        assert options.progressive is True
        assert options.allow_partial is True
    
    def test_fetch_result_is_usable(self):
        """Test fetch result usability check."""
        # Usable result
        result = FetchResult(
            success=True,
            data=pd.DataFrame({"close": [1, 2, 3]}),
            bars_loaded=100,
            bars_requested=100,
            quality_score=0.8,
            load_time_ms=50,
            source_used="akshare",
            cache_hit=False,
            validation_result=None,
            load_status=LoadStatus.COMPLETE,
        )
        assert result.is_usable()
        
        # Unusable result
        bad_result = FetchResult(
            success=False,
            data=None,
            bars_loaded=0,
            bars_requested=100,
            quality_score=0,
            load_time_ms=10,
            source_used=None,
            cache_hit=False,
            validation_result=None,
            load_status=LoadStatus.FAILED,
            error="Failed",
        )
        assert not bad_result.is_usable()
    
    def test_fetch_result_to_dict(self):
        """Test fetch result serialization."""
        result = FetchResult(
            success=True,
            data=pd.DataFrame({"close": [1, 2, 3]}),
            bars_loaded=50,
            bars_requested=100,
            quality_score=0.75,
            load_time_ms=100,
            source_used="sina",
            cache_hit=True,
            validation_result=None,
            load_status=LoadStatus.PARTIAL,
        )
        
        result_dict = result.to_dict()
        
        assert result_dict["success"] is True
        assert result_dict["bars_loaded"] == 50
        assert result_dict["completeness"] == 0.5
        assert result_dict["quality_score"] == 0.75
    
    @patch('data.fetcher_unified.get_fetcher')
    def test_unified_fetcher_fallback(self, mock_get_fetcher):
        """Test unified fetcher fallback to standard fetcher."""
        # Setup mock
        mock_inner = Mock()
        mock_inner.get_history.return_value = pd.DataFrame({
            "open": [100, 101],
            "high": [102, 103],
            "low": [99, 100],
            "close": [101, 102],
            "volume": [1000, 1100],
        }, index=pd.date_range("2024-01-01", periods=2))
        mock_get_fetcher.return_value = mock_inner
        
        fetcher = UnifiedDataFetcher(inner_fetcher=mock_inner)
        
        # Disable unified fetcher to force fallback
        df = fetcher.get_history("000001", interval="1d", bars=10)
        
        assert df is not None
        assert len(df) > 0
    
    def test_get_metrics(self):
        """Test fetching metrics."""
        fetcher = UnifiedDataFetcher()
        
        metrics = fetcher.get_metrics()
        
        assert "total_requests" in metrics
        assert "successful_requests" in metrics
        assert "failed_requests" in metrics
        assert "health_monitor" in metrics

    def test_direct_fetch_handles_days_without_bars(self):
        """Direct mode should derive bars from days without raising errors."""
        mock_inner = Mock()
        mock_inner.get_history.return_value = pd.DataFrame(
            {
                "open": [100, 101, 102],
                "high": [101, 102, 103],
                "low": [99, 100, 101],
                "close": [100, 101, 102],
                "volume": [1000, 1100, 1200],
            },
            index=pd.date_range("2024-01-01", periods=3, freq="1d"),
        )
        fetcher = UnifiedDataFetcher(inner_fetcher=mock_inner)

        result = fetcher.fetch_with_options(
            "000001",
            FetchOptions(interval="1d", days=3, progressive=False, validate=False),
        )

        assert result.success
        kwargs = mock_inner.get_history.call_args.kwargs
        assert kwargs["bars"] == 3

    def test_partial_data_rejected_when_allow_partial_false(self):
        """Strict mode should reject incomplete direct-fetch data."""
        mock_inner = Mock()
        mock_inner.get_history.return_value = pd.DataFrame(
            {
                "open": [100 + i for i in range(20)],
                "high": [101 + i for i in range(20)],
                "low": [99 + i for i in range(20)],
                "close": [100 + i for i in range(20)],
                "volume": [1000 + i for i in range(20)],
            },
            index=pd.date_range("2024-01-01", periods=20, freq="1d"),
        )
        fetcher = UnifiedDataFetcher(inner_fetcher=mock_inner)

        result = fetcher.fetch_with_options(
            "000001",
            FetchOptions(
                interval="1d",
                bars=100,
                progressive=False,
                validate=False,
                allow_partial=False,
                partial_threshold=0.4,
            ),
        )

        assert result.success is False
        assert result.load_status == LoadStatus.INSUFFICIENT
        assert "Partial data rejected" in str(result.error)

    def test_result_cache_hits_and_force_refresh(self):
        """Versioned in-memory cache should avoid duplicate fetches unless forced."""
        mock_inner = Mock()
        mock_inner.get_history.return_value = pd.DataFrame(
            {
                "open": [100, 101, 102, 103],
                "high": [101, 102, 103, 104],
                "low": [99, 100, 101, 102],
                "close": [100, 101, 102, 103],
                "volume": [1000, 1100, 1200, 1300],
            },
            index=pd.date_range("2024-01-01", periods=4, freq="1d"),
        )
        fetcher = UnifiedDataFetcher(inner_fetcher=mock_inner)

        options = FetchOptions(
            interval="1d",
            bars=4,
            progressive=False,
            validate=False,
            use_cache=True,
        )
        first = fetcher.fetch_with_options("000001", options)
        second = fetcher.fetch_with_options("000001", options)
        refreshed = fetcher.fetch_with_options(
            "000001",
            FetchOptions(
                interval="1d",
                bars=4,
                progressive=False,
                validate=False,
                use_cache=True,
                force_refresh=True,
            ),
        )

        assert first.success
        assert second.success
        assert second.cache_hit is True
        assert refreshed.success
        assert mock_inner.get_history.call_count == 2

    def test_stale_intraday_cache_triggers_refresh(self):
        """Stale cached intraday frame should be refreshed instead of reused."""
        old_index = pd.date_range("2020-03-16 10:00", periods=20, freq="1min")
        fresh_index = pd.date_range("2024-03-15 10:00", periods=20, freq="1min")
        old_df = pd.DataFrame(
            {
                "open": [100 + i for i in range(20)],
                "high": [101 + i for i in range(20)],
                "low": [99 + i for i in range(20)],
                "close": [100 + i for i in range(20)],
                "volume": [1000 + i for i in range(20)],
            },
            index=old_index,
        )
        fresh_df = pd.DataFrame(
            {
                "open": [200 + i for i in range(20)],
                "high": [201 + i for i in range(20)],
                "low": [199 + i for i in range(20)],
                "close": [200 + i for i in range(20)],
                "volume": [2000 + i for i in range(20)],
            },
            index=fresh_index,
        )
        mock_inner = Mock()
        mock_inner.get_history.side_effect = [old_df, fresh_df]
        fetcher = UnifiedDataFetcher(inner_fetcher=mock_inner)
        fetcher._session_checker.is_market_open = lambda _ts: True  # type: ignore[assignment]

        options = FetchOptions(
            interval="1m",
            bars=20,
            progressive=False,
            validate=False,
            use_cache=True,
        )
        first = fetcher.fetch_with_options("000001", options)
        second = fetcher.fetch_with_options("000001", options)

        assert first.success
        assert second.success
        assert mock_inner.get_history.call_count == 2
        assert float(second.data.iloc[-1]["close"]) == float(fresh_df.iloc[-1]["close"])

    def test_rate_limit_timeout_returns_failure(self, monkeypatch: pytest.MonkeyPatch):
        """Rate-limit gate timeout should fail request when no fallback exists."""
        mock_inner = Mock()
        mock_inner.get_history.return_value = pd.DataFrame()
        fetcher = UnifiedDataFetcher(inner_fetcher=mock_inner)

        monkeypatch.setattr(unified_mod, "_HAS_ENHANCED_RATE_LIMITING", True)
        monkeypatch.setattr(
            unified_mod,
            "_acquire_rate_limit",
            lambda _source, timeout=30.0: False,
        )

        result = fetcher.fetch_with_options(
            "000001",
            FetchOptions(
                interval="1d",
                bars=20,
                progressive=False,
                validate=False,
                preferred_source="akshare",
            ),
        )

        assert result.success is False
        assert "Rate limit acquisition timed out" in str(result.error)


class TestValidator:
    """Tests for data validation."""
    
    def test_validate_good_data(self):
        """Test validation of good data."""
        validator = DataValidator()
        
        df = pd.DataFrame({
            "open": [100, 101, 102],
            "high": [102, 103, 104],
            "low": [99, 100, 101],
            "close": [101, 102, 103],
            "volume": [1000, 1100, 1200],
        }, index=pd.date_range("2024-01-01", periods=3))
        
        result = validator.validate_bars(df, symbol="000001", interval="1d")
        
        assert result.is_valid
        assert result.score > 0.8
    
    def test_validate_bad_ohlc(self):
        """Test validation catches OHLC issues."""
        validator = DataValidator()
        
        # High < Low (invalid)
        df = pd.DataFrame({
            "open": [100],
            "high": [99],  # Invalid: high < low
            "low": [100],
            "close": [100],
            "volume": [1000],
        }, index=pd.date_range("2024-01-01", periods=1))
        
        result = validator.validate_bars(df)
        
        assert not result.is_valid or len(result.issues) > 0 or len(result.warnings) > 0
    
    def test_validate_nan_values(self):
        """Test validation catches NaN values."""
        validator = DataValidator()
        
        df = pd.DataFrame({
            "open": [100, float('nan'), 102],
            "high": [102, 103, 104],
            "low": [99, 100, 101],
            "close": [101, 102, 103],
            "volume": [1000, 1100, 1200],
        }, index=pd.date_range("2024-01-01", periods=3))
        
        result = validator.validate_bars(df)
        
        assert len(result.warnings) > 0 or result.score < 1.0


class TestIntegration:
    """Integration tests for the complete pipeline."""
    
    def test_end_to_end_fetch(self):
        """Test complete fetch pipeline with mock data."""
        # Create mock inner fetcher
        mock_inner = Mock()
        mock_inner.get_history.return_value = pd.DataFrame({
            "open": [100 + i for i in range(50)],
            "high": [102 + i for i in range(50)],
            "low": [99 + i for i in range(50)],
            "close": [101 + i for i in range(50)],
            "volume": [1000 + i * 10 for i in range(50)],
        }, index=pd.date_range("2024-01-01", periods=50, freq="1d"))
        
        fetcher = UnifiedDataFetcher(inner_fetcher=mock_inner)
        
        # Fetch with all features enabled
        options = FetchOptions(
            interval="1d",
            bars=50,
            validate=True,
            progressive=False,  # Use direct fetch for mock
            allow_partial=True,
        )
        
        result = fetcher.fetch_with_options("000001", options)
        
        assert result.success
        assert result.bars_loaded > 0
        assert result.quality_score > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
