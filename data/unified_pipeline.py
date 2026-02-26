"""Unified Data Pipeline with Look-Ahead Bias Prevention.

This module provides:
- Unified pipeline for structured (price) and unstructured (news) data
- Precise timestamp synchronization
- Look-ahead bias prevention for backtesting
- Point-in-time correct data access
- Event-driven data alignment

Fixes:
- Timestamp synchronization between news and price data
- Look-ahead bias in backtesting
- Infrastructure complexity for multi-modal data
"""
from __future__ import annotations

import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config.settings import CONFIG
from utils.logger import get_logger

from .data_quality import DataAdjuster, DataQualityValidator, get_adjuster, get_validator
from .news_processor_enhanced import (
    EnhancedNewsProcessor,
    FilteredNews,
    get_news_processor,
)
from .news_collector import NewsArticle, get_collector

log = get_logger(__name__)


class DataType(Enum):
    """Types of data in the pipeline."""
    PRICE = "price"
    NEWS = "news"
    SENTIMENT = "sentiment"
    FUNDAMENTAL = "fundamental"
    ALTERNATIVE = "alternative"


class TimeAlignmentMethod(Enum):
    """Methods for time alignment."""
    FORWARD_FILL = "forward_fill"  # Use last known value
    BACKWARD_FILL = "backward_fill"  # Use next value (look-ahead!)
    INTERPOLATE = "interpolate"  # Linear interpolation
    DROP = "drop"  # Drop misaligned rows


@dataclass
class AlignedDataPoint:
    """Data point aligned in time."""
    timestamp: datetime
    symbol: str
    data_type: DataType
    value: Any
    quality_score: float = 1.0
    source: str = ""
    is_estimated: bool = False  # True if forward-filled
    original_timestamp: datetime | None = None
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "symbol": self.symbol,
            "data_type": self.data_type.value,
            "value": self.value,
            "quality_score": self.quality_score,
            "source": self.source,
            "is_estimated": self.is_estimated,
            "original_timestamp": self.original_timestamp.isoformat() if self.original_timestamp else None,
        }


@dataclass
class SynchronizedSnapshot:
    """Synchronized snapshot of all data types at a point in time."""
    timestamp: datetime
    symbol: str
    price_data: dict[str, Any] = field(default_factory=dict)
    news_data: list[FilteredNews] = field(default_factory=list)
    sentiment_data: dict[str, float] = field(default_factory=dict)
    fundamental_data: dict[str, Any] = field(default_factory=dict)
    data_quality: dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "symbol": self.symbol,
            "price_data": self.price_data,
            "news_data": [n.to_dict() for n in self.news_data],
            "sentiment_data": self.sentiment_data,
            "fundamental_data": self.fundamental_data,
            "data_quality": self.data_quality,
        }


@dataclass
class BacktestConfig:
    """Configuration for backtesting with bias prevention."""
    symbol: str
    start_date: datetime
    end_date: datetime
    use_news_data: bool = True
    news_lookback_hours: int = 24  # How far back to look for news
    delay_news_minutes: int = 0  # Intentional delay to simulate real-world latency
    allow_forward_fill: bool = True  # Allow forward-filling missing data
    max_forward_fill_bars: int = 5  # Maximum bars to forward-fill
    require_price_quality: float = 0.7  # Minimum price data quality
    require_news_quality: float = 0.5  # Minimum news quality
    trading_hours_only: bool = True  # Only include trading hours


class LookAheadBiasPreventer:
    """Prevents look-ahead bias in backtesting."""
    
    def __init__(self) -> None:
        self._knowledge_cutoff: dict[str, datetime] = {}
        self._lock = threading.RLock()
    
    def set_cutoff(self, symbol: str, cutoff: datetime) -> None:
        """Set the knowledge cutoff for a symbol (current backtest time)."""
        with self._lock:
            self._knowledge_cutoff[symbol] = cutoff
    
    def get_cutoff(self, symbol: str) -> datetime | None:
        """Get the knowledge cutoff for a symbol."""
        with self._lock:
            return self._knowledge_cutoff.get(symbol)
    
    def is_accessible(
        self,
        symbol: str,
        data_timestamp: datetime,
        data_type: DataType = DataType.PRICE,
        delay_minutes: int = 0,
    ) -> bool:
        """Check if data is accessible at current backtest time (no look-ahead).
        
        Args:
            symbol: Stock symbol
            data_timestamp: Timestamp of the data
            data_type: Type of data
            delay_minutes: Additional delay for the data type (e.g., news latency)
        
        Returns:
            True if data is accessible without look-ahead bias
        """
        cutoff = self.get_cutoff(symbol)
        if cutoff is None:
            return True  # No cutoff set, assume accessible
        
        # Apply delay for news data (simulates real-world latency)
        effective_timestamp = data_timestamp + timedelta(minutes=delay_minutes)
        
        # Data is accessible only if its timestamp (plus delay) is before cutoff
        return effective_timestamp <= cutoff
    
    def filter_accessible(
        self,
        symbol: str,
        items: list[Any],
        timestamp_fn: callable,
        data_type: DataType = DataType.PRICE,
        delay_minutes: int = 0,
    ) -> list[Any]:
        """Filter list to only include items accessible at cutoff time.
        
        Args:
            symbol: Stock symbol
            items: List of items to filter
            timestamp_fn: Function to extract timestamp from item
            data_type: Type of data
            delay_minutes: Additional delay for the data type
        
        Returns:
            Filtered list of accessible items
        """
        cutoff = self.get_cutoff(symbol)
        if cutoff is None:
            return items
        
        result = []
        for item in items:
            ts = timestamp_fn(item)
            effective_ts = ts + timedelta(minutes=delay_minutes)
            if effective_ts <= cutoff:
                result.append(item)
        
        return result


class UnifiedDataPipeline:
    """Unified pipeline for price and news data with bias prevention."""
    
    def __init__(
        self,
        quality_validator: DataQualityValidator | None = None,
        data_adjuster: DataAdjuster | None = None,
        news_processor: EnhancedNewsProcessor | None = None,
    ) -> None:
        self.quality_validator = quality_validator or get_validator()
        self.data_adjuster = data_adjuster or get_adjuster()
        self.news_processor = news_processor or get_news_processor()
        
        self.bias_preventer = LookAheadBiasPreventer()
        
        # Data storage
        self._price_data: dict[str, pd.DataFrame] = {}
        self._news_data: dict[str, list[FilteredNews]] = {}
        self._quality_reports: dict[str, Any] = {}
        
        # Cache
        self._snapshot_cache: dict[str, SynchronizedSnapshot] = {}
        self._cache_ttl = 60  # seconds
    
    def load_price_data(
        self,
        symbol: str,
        df: pd.DataFrame,
        adjust_corporate_actions: bool = True,
        validate_quality: bool = True,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Load and validate price data.
        
        Args:
            symbol: Stock symbol
            df: Price DataFrame
            adjust_corporate_actions: Whether to adjust for splits/dividends
            validate_quality: Whether to validate data quality
        
        Returns:
            Tuple of (validated DataFrame, quality report)
        """
        # Validate quality
        quality_report = None
        if validate_quality:
            quality_report = self.quality_validator.validate(df, symbol)
            self._quality_reports[symbol] = quality_report.to_dict()
            
            if not quality_report.is_valid:
                log.warning(f"Price data quality issues for {symbol}: {quality_report.issues}")
        
        # Adjust for corporate actions
        df_adjusted = df.copy()
        adjustments = 0
        if adjust_corporate_actions:
            df_adjusted, adjustments = self.data_adjuster.adjust_for_corporate_actions(
                df_adjusted, symbol
            )
        
        # Ensure datetime index
        if not isinstance(df_adjusted.index, pd.DatetimeIndex):
            if "datetime" in df_adjusted.columns:
                df_adjusted["datetime"] = pd.to_datetime(df_adjusted["datetime"])
                df_adjusted.set_index("datetime", inplace=True)
            elif "date" in df_adjusted.columns:
                df_adjusted["date"] = pd.to_datetime(df_adjusted["date"])
                df_adjusted.set_index("date", inplace=True)
        
        # Sort by timestamp
        df_adjusted = df_adjusted.sort_index()
        
        # Store
        self._price_data[symbol] = df_adjusted
        
        log.info(
            f"Loaded price data for {symbol}: {len(df_adjusted)} rows, "
            f"{adjustments} corporate action adjustments"
        )
        
        return df_adjusted, quality_report.to_dict() if quality_report else {}
    
    def load_news_data(
        self,
        symbol: str,
        articles: list[NewsArticle] | list[FilteredNews],
        keywords: list[str] | None = None,
    ) -> list[FilteredNews]:
        """Load and process news data.
        
        Args:
            symbol: Stock symbol
            articles: List of news articles
            keywords: Keywords for relevance scoring
        
        Returns:
            List of filtered news articles
        """
        # Process articles if they're raw
        if articles and isinstance(articles[0], NewsArticle):
            filtered = self.news_processor.process_batch(
                articles,  # type: ignore
                keywords=keywords or [symbol],
                store=True,
            )
        else:
            filtered = articles  # type: ignore
        
        # Filter by symbol relevance
        symbol_filtered = []
        for news in filtered:
            # Check if symbol is mentioned in entities or content
            if symbol in news.article.entities:
                symbol_filtered.append(news)
            elif symbol in news.article.content or symbol in news.article.title:
                symbol_filtered.append(news)
        
        # Store
        self._news_data[symbol] = symbol_filtered
        
        log.info(f"Loaded {len(symbol_filtered)} relevant news articles for {symbol}")
        
        return symbol_filtered
    
    def get_synchronized_snapshot(
        self,
        symbol: str,
        timestamp: datetime,
        config: BacktestConfig | None = None,
    ) -> SynchronizedSnapshot:
        """Get synchronized snapshot of all data at a point in time.
        
        This is the key method for preventing look-ahead bias:
        - Only returns data that was available at the given timestamp
        - Applies news delay for realistic latency simulation
        - Forward-fills missing data within limits
        
        Args:
            symbol: Stock symbol
            timestamp: Point in time for snapshot
            config: Backtest configuration
        
        Returns:
            SynchronizedSnapshot with all available data
        """
        config = config or BacktestConfig(
            symbol=symbol,
            start_date=timestamp - timedelta(days=30),
            end_date=timestamp,
        )
        
        # Set knowledge cutoff for bias prevention
        self.bias_preventer.set_cutoff(symbol, timestamp)
        
        # Get price data at timestamp
        price_data = self._get_price_at_time(
            symbol, timestamp, config
        )
        
        # Get news data available at timestamp
        news_data = self._get_news_at_time(
            symbol, timestamp, config
        )
        
        # Calculate sentiment from available news
        sentiment_data = self._calculate_sentiment_at_time(
            news_data, timestamp
        )
        
        # Calculate data quality scores
        data_quality = self._calculate_quality_at_time(
            symbol, price_data, news_data, timestamp
        )
        
        snapshot = SynchronizedSnapshot(
            timestamp=timestamp,
            symbol=symbol,
            price_data=price_data,
            news_data=news_data,
            sentiment_data=sentiment_data,
            fundamental_data={},  # Can be extended
            data_quality=data_quality,
        )
        
        # Cache
        cache_key = f"{symbol}_{timestamp.isoformat()}"
        self._snapshot_cache[cache_key] = snapshot
        
        return snapshot
    
    def _get_price_at_time(
        self,
        symbol: str,
        timestamp: datetime,
        config: BacktestConfig,
    ) -> dict[str, Any]:
        """Get price data available at timestamp."""
        df = self._price_data.get(symbol)
        if df is None or df.empty:
            return {}
        
        # Filter to data available at timestamp
        accessible_df = df[df.index <= timestamp].copy()
        
        if accessible_df.empty:
            return {}
        
        # Get latest row
        latest = accessible_df.iloc[-1]
        
        # Check if we need forward-fill
        time_diff = timestamp - accessible_df.index[-1]
        is_estimated = False
        
        if time_diff > timedelta(days=1) and config.allow_forward_fill:
            # Data is stale, but we allow forward-fill
            is_estimated = True
        
        return {
            "open": float(latest.get("open", 0)),
            "high": float(latest.get("high", 0)),
            "low": float(latest.get("low", 0)),
            "close": float(latest.get("close", 0)),
            "volume": float(latest.get("volume", 0)),
            "timestamp": accessible_df.index[-1].isoformat(),
            "is_estimated": is_estimated,
        }
    
    def _get_news_at_time(
        self,
        symbol: str,
        timestamp: datetime,
        config: BacktestConfig,
    ) -> list[FilteredNews]:
        """Get news data available at timestamp."""
        if not config.use_news_data:
            return []
        
        all_news = self._news_data.get(symbol, [])
        if not all_news:
            return []
        
        # Calculate lookback window
        lookback_start = timestamp - timedelta(hours=config.news_lookback_hours)
        
        # Filter by time window AND accessibility (no look-ahead)
        accessible_news = []
        for news in all_news:
            # Apply news delay (simulates real-world latency)
            effective_time = news.timestamp_aligned + timedelta(
                minutes=config.delay_news_minutes
            )
            
            # Check if news is accessible at timestamp
            if effective_time <= timestamp and news.timestamp_aligned >= lookback_start:
                # Check quality threshold
                if news.quality_score >= config.require_news_quality:
                    accessible_news.append(news)
        
        # Sort by timestamp
        accessible_news.sort(key=lambda x: x.timestamp_aligned, reverse=True)
        
        return accessible_news
    
    def _calculate_sentiment_at_time(
        self,
        news_data: list[FilteredNews],
        timestamp: datetime,
    ) -> dict[str, float]:
        """Calculate sentiment from available news."""
        if not news_data:
            return {
                "overall": 0.0,
                "weighted": 0.0,
                "article_count": 0,
            }
        
        # Simple average
        sentiments = [n.bias_adjusted_sentiment for n in news_data]
        overall = np.mean(sentiments) if sentiments else 0.0
        
        # Quality-weighted
        quality_weights = {
            "high": 1.0,
            "medium": 0.7,
            "low": 0.4,
        }
        
        weighted_sum = sum(
            n.bias_adjusted_sentiment * quality_weights.get(n.quality_level.value, 0.5)
            for n in news_data
        )
        total_weight = sum(
            quality_weights.get(n.quality_level.value, 0.5)
            for n in news_data
        )
        weighted = weighted_sum / max(1.0, total_weight)
        
        return {
            "overall": float(overall),
            "weighted": float(weighted),
            "article_count": len(news_data),
            "std": float(np.std(sentiments)) if len(sentiments) > 1 else 0.0,
        }
    
    def _calculate_quality_at_time(
        self,
        symbol: str,
        price_data: dict[str, Any],
        news_data: list[FilteredNews],
        timestamp: datetime,
    ) -> dict[str, float]:
        """Calculate data quality scores at timestamp."""
        quality = {}
        
        # Price data quality
        if price_data:
            quality_report = self._quality_reports.get(symbol, {})
            quality["price"] = quality_report.get("quality_score", 0.5)
            
            # Penalize for estimated data
            if price_data.get("is_estimated"):
                quality["price"] *= 0.8
        else:
            quality["price"] = 0.0
        
        # News data quality
        if news_data:
            quality["news"] = np.mean([n.quality_score for n in news_data])
        else:
            quality["news"] = 0.5  # Neutral when no news
        
        # Overall quality
        quality["overall"] = (quality["price"] * 0.7 + quality["news"] * 0.3)
        
        return quality
    
    def iterate_backtest(
        self,
        config: BacktestConfig,
    ) -> list[SynchronizedSnapshot]:
        """Iterate through backtest period with proper bias prevention.
        
        This is the main method for running backtests:
        - Yields synchronized snapshots at each time step
        - Ensures no look-ahead bias
        - Handles trading hours
        - Applies news delay
        
        Args:
            config: Backtest configuration
        
        Yields:
            SynchronizedSnapshot for each time step
        """
        # Get price data index for iteration
        df = self._price_data.get(config.symbol)
        if df is None or df.empty:
            log.warning(f"No price data for {config.symbol}")
            return []
        
        # Filter to backtest period
        mask = (df.index >= config.start_date) & (df.index <= config.end_date)
        backtest_df = df[mask]
        
        if backtest_df.empty:
            log.warning(f"No price data in backtest period for {config.symbol}")
            return []
        
        snapshots = []
        
        # Iterate through each timestamp
        for timestamp in backtest_df.index:
            # Skip non-trading hours if configured
            if config.trading_hours_only:
                hour = timestamp.hour
                minute = timestamp.minute
                
                # Check if within trading hours
                if hour < 9 or (hour == 9 and minute < 30):
                    continue
                if hour > 15 or (hour == 15 and minute > 0):
                    continue
                # Skip lunch break
                if 11 <= hour <= 12:
                    continue
            
            # Get synchronized snapshot
            snapshot = self.get_synchronized_snapshot(
                config.symbol,
                timestamp,
                config,
            )
            
            # Skip if data quality is too low
            if snapshot.data_quality.get("overall", 0) < config.require_price_quality:
                log.debug(
                    f"Skipping {timestamp} due to low data quality: "
                    f"{snapshot.data_quality.get('overall', 0):.2f}"
                )
                continue
            
            snapshots.append(snapshot)
        
        log.info(
            f"Generated {len(snapshots)} backtest snapshots for {config.symbol} "
            f"from {config.start_date} to {config.end_date}"
        )
        
        return snapshots
    
    def get_feature_matrix(
        self,
        snapshots: list[SynchronizedSnapshot],
    ) -> pd.DataFrame:
        """Convert snapshots to feature matrix for ML.
        
        Args:
            snapshots: List of synchronized snapshots
        
        Returns:
            DataFrame with features for each timestamp
        """
        rows = []
        
        for snapshot in snapshots:
            row = {
                "timestamp": snapshot.timestamp,
                "symbol": snapshot.symbol,
            }
            
            # Price features
            for key, value in snapshot.price_data.items():
                if key not in ["timestamp", "is_estimated"]:
                    row[f"price_{key}"] = value
            
            # Sentiment features
            for key, value in snapshot.sentiment_data.items():
                row[f"sentiment_{key}"] = value
            
            # News count feature
            row["news_count"] = len(snapshot.news_data)
            
            # Quality features
            for key, value in snapshot.data_quality.items():
                row[f"quality_{key}"] = value
            
            # Recent sentiment trend (if enough news)
            if len(snapshot.news_data) >= 3:
                recent = snapshot.news_data[:3]
                older = snapshot.news_data[3:6] if len(snapshot.news_data) > 5 else []
                
                if older:
                    recent_avg = np.mean([n.bias_adjusted_sentiment for n in recent])
                    older_avg = np.mean([n.bias_adjusted_sentiment for n in older])
                    row["sentiment_trend"] = recent_avg - older_avg
            
            rows.append(row)
        
        df = pd.DataFrame(rows)
        
        # Set timestamp as index
        if "timestamp" in df.columns:
            df.set_index("timestamp", inplace=True)
        
        return df


# Singleton instance
_pipeline: UnifiedDataPipeline | None = None


def get_data_pipeline() -> UnifiedDataPipeline:
    """Get singleton data pipeline instance."""
    global _pipeline
    if _pipeline is None:
        _pipeline = UnifiedDataPipeline()
    return _pipeline


def reset_pipeline() -> None:
    """Reset pipeline instance (for testing)."""
    global _pipeline
    _pipeline = None
