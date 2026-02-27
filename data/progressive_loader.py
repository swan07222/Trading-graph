# data/progressive_loader.py
"""Progressive data loading with minimum viable dataset.

FIX 2026-02-26: Addresses disadvantages:
- Minimum data requirements with graceful degradation
- Progressive loading for large datasets
- Chunked processing to reduce memory
- Partial data acceptance with quality scoring

Features:
- Configurable minimum bar requirements
- Progressive fetching in chunks
- Quality-based acceptance
- Memory-bounded loading
"""

import gc
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Generator

import pandas as pd

from config.runtime_env import env_flag, env_float, env_int, env_text
from utils.logger import get_logger

log = get_logger(__name__)


class LoadStatus(Enum):
    """Status of progressive loading."""
    COMPLETE = "complete"
    PARTIAL = "partial"
    MINIMUM = "minimum"
    INSUFFICIENT = "insufficient"
    FAILED = "failed"

    def is_usable(self, min_status: "LoadStatus" = None) -> bool:
        """Check if this status meets a minimum threshold."""
        if min_status is None:
            min_status = LoadStatus.MINIMUM
        status_order = {
            LoadStatus.FAILED: 0,
            LoadStatus.INSUFFICIENT: 1,
            LoadStatus.MINIMUM: 2,
            LoadStatus.PARTIAL: 3,
            LoadStatus.COMPLETE: 4,
        }
        return status_order[self] >= status_order[min_status]


@dataclass
class LoadResult:
    """Result of progressive data loading."""
    status: LoadStatus
    data: pd.DataFrame | None
    bars_loaded: int
    bars_requested: int
    quality_score: float
    load_time_ms: float
    chunks_loaded: int
    error: str | None = None
    
    def is_usable(self, min_status: LoadStatus = LoadStatus.MINIMUM) -> bool:
        """Check if loaded data meets minimum requirements."""
        return self.status.is_usable(min_status)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for reporting."""
        return {
            "status": self.status.value,
            "bars_loaded": self.bars_loaded,
            "bars_requested": self.bars_requested,
            "completeness": round(self.bars_loaded / max(1, self.bars_requested), 2),
            "quality_score": round(self.quality_score, 3),
            "load_time_ms": round(self.load_time_ms, 1),
            "chunks_loaded": self.chunks_loaded,
            "error": self.error,
        }


class ProgressiveDataLoader:
    """Load data progressively with minimum viable dataset.
    
    FIX 2026-02-26:
    - Load minimum data first, then enhance
    - Chunked loading to reduce memory pressure
    - Quality-based acceptance
    - Graceful degradation on failures
    """
    
    def __init__(
        self,
        min_bars_intraday: int = 480,
        min_bars_daily: int = 14,
        min_bars_weekly: int = 8,
        min_bars_monthly: int = 6,
        chunk_size: int = 500,
        max_bars_per_request: int = 2000,
        allow_partial: bool = True,
        partial_threshold: float = 0.4,
        max_memory_mb: float = 500.0,
    ):
        self.min_bars = {
            "1m": min_bars_intraday,
            "2m": min_bars_intraday,
            "3m": min_bars_intraday,
            "5m": min_bars_intraday,
            "15m": min_bars_intraday,
            "30m": min_bars_intraday,
            "60m": min_bars_intraday,
            "1h": min_bars_intraday,
            "1d": min_bars_daily,
            "1wk": min_bars_weekly,
            "1mo": min_bars_monthly,
        }
        self.chunk_size = chunk_size
        self.max_bars_per_request = max_bars_per_request
        self.allow_partial = allow_partial
        self.partial_threshold = partial_threshold
        self.max_memory_mb = max_memory_mb
        
        self._lock = threading.RLock()
        self._memory_warning_threshold = max_memory_mb * 0.8
    
    def _get_min_bars(self, interval: str) -> int:
        """Get minimum bars for interval."""
        interval_key = interval.lower()
        return self.min_bars.get(interval_key, self.min_bars.get("1d", 14))
    
    def _estimate_memory_mb(self, df: pd.DataFrame) -> float:
        """Estimate DataFrame memory usage in MB."""
        try:
            return df.memory_usage(deep=True).sum() / (1024 * 1024)
        except Exception:
            # Rough estimate: rows * cols * 8 bytes
            return len(df) * len(df.columns) * 8 / (1024 * 1024)
    
    def _check_memory_pressure(self) -> bool:
        """Check if system is under memory pressure."""
        try:
            import psutil
            process = psutil.Process()
            memory_mb = process.memory_info().rss / (1024 * 1024)
            return memory_mb > self._memory_warning_threshold
        except ImportError:
            return False
        except Exception:
            return False
    
    def load(
        self,
        fetch_fn: Callable[[int], pd.DataFrame | None],
        interval: str = "1d",
        requested_bars: int | None = None,
    ) -> LoadResult:
        """Load data progressively using fetch function.
        
        FIX 2026-02-26: Progressive loading strategy:
        1. Load minimum required bars first
        2. If successful, load remaining in chunks
        3. Stop on memory pressure or errors
        4. Return best available data with status
        
        Args:
            fetch_fn: Function that takes bars count and returns DataFrame
            interval: Bar interval (for min bars determination)
            requested_bars: Total bars requested (optional)
        
        Returns:
            LoadResult with status and data
        """
        start_time = time.time()
        
        min_bars = self._get_min_bars(interval)
        target_bars = requested_bars or min_bars
        target_bars = min(target_bars, self.max_bars_per_request)
        
        log.debug(
            "[PROGRESSIVE] Loading %s interval: min=%d, target=%d, chunk=%d",
            interval, min_bars, target_bars, self.chunk_size
        )
        
        # Phase 1: Load minimum required data
        min_result = self._load_chunk(fetch_fn, min_bars)
        
        if min_result is None or min_result.empty:
            return LoadResult(
                status=LoadStatus.FAILED,
                data=None,
                bars_loaded=0,
                bars_requested=target_bars,
                quality_score=0.0,
                load_time_ms=(time.time() - start_time) * 1000,
                chunks_loaded=0,
                error="Failed to load minimum data",
            )
        
        # Validate minimum data quality
        quality = self._calculate_quality_score(min_result, interval)
        
        if len(min_result) < min_bars * self.partial_threshold:
            if self.allow_partial:
                log.warning(
                    "Only %d/%d minimum bars loaded, accepting partial data",
                    len(min_result), min_bars
                )
                return LoadResult(
                    status=LoadStatus.MINIMUM,
                    data=min_result,
                    bars_loaded=len(min_result),
                    bars_requested=target_bars,
                    quality_score=quality,
                    load_time_ms=(time.time() - start_time) * 1000,
                    chunks_loaded=1,
                )
            else:
                return LoadResult(
                    status=LoadStatus.INSUFFICIENT,
                    data=min_result,
                    bars_loaded=len(min_result),
                    bars_requested=target_bars,
                    quality_score=quality,
                    load_time_ms=(time.time() - start_time) * 1000,
                    chunks_loaded=1,
                    error=f"Insufficient data: {len(min_result)} < {min_bars}",
                )
        
        # Phase 2: Load remaining data in chunks
        remaining_bars = target_bars - len(min_result)
        chunks_loaded = 1
        all_data = [min_result]
        
        while remaining_bars > 0:
            # Check memory pressure
            if self._check_memory_pressure():
                log.warning("Memory pressure detected, stopping progressive load")
                gc.collect()
                break
            
            # Calculate next chunk size
            chunk_bars = min(self.chunk_size, remaining_bars)
            
            # Load chunk
            chunk_result = self._load_chunk(fetch_fn, target_bars)
            
            if chunk_result is None or chunk_result.empty:
                log.debug("Failed to load additional chunk, using available data")
                break
            
            # Check if we got new data
            if len(chunk_result) <= len(min_result):
                break
            
            # Extract new rows (assuming chronological order)
            new_data = chunk_result.iloc[-(len(chunk_result) - len(min_result)):]
            
            if new_data.empty:
                break
            
            all_data.append(new_data)
            remaining_bars -= len(new_data)
            chunks_loaded += 1
            
            # Update minimum for next iteration
            min_result = pd.concat(all_data, ignore_index=False)
        
        # Combine all chunks
        final_data = pd.concat(all_data, ignore_index=False) if len(all_data) > 1 else all_data[0]
        
        # Remove duplicates (keep last)
        if final_data.index.has_duplicates:
            final_data = final_data[~final_data.index.duplicated(keep="last")]
        
        # Calculate final quality
        final_quality = self._calculate_quality_score(final_data, interval)
        
        # Determine status
        if len(final_data) >= target_bars * 0.9:
            status = LoadStatus.COMPLETE
        elif len(final_data) >= target_bars * self.partial_threshold:
            status = LoadStatus.PARTIAL
        elif len(final_data) >= min_bars * self.partial_threshold:
            status = LoadStatus.MINIMUM
        else:
            status = LoadStatus.INSUFFICIENT
        
        load_time_ms = (time.time() - start_time) * 1000
        
        log.info(
            "[PROGRESSIVE] Loaded %d/%d bars (%s), quality=%.2f, time=%.0fms, chunks=%d",
            len(final_data), target_bars, status.value,
            final_quality, load_time_ms, chunks_loaded
        )
        
        return LoadResult(
            status=status,
            data=final_data,
            bars_loaded=len(final_data),
            bars_requested=target_bars,
            quality_score=final_quality,
            load_time_ms=load_time_ms,
            chunks_loaded=chunks_loaded,
        )
    
    def _load_chunk(
        self,
        fetch_fn: Callable[[int], pd.DataFrame | None],
        bars: int,
    ) -> pd.DataFrame | None:
        """Load a single chunk of data."""
        try:
            return fetch_fn(bars)
        except Exception as e:
            log.debug("Chunk load failed: %s", e)
            return None
    
    def _calculate_quality_score(self, df: pd.DataFrame, interval: str) -> float:
        """Calculate data quality score (0.0 to 1.0)."""
        if df is None or df.empty:
            return 0.0
        
        score = 1.0
        
        # Check for NaN values
        required_cols = ["open", "high", "low", "close"]
        for col in required_cols:
            if col in df.columns:
                nan_ratio = df[col].isna().sum() / len(df)
                score -= nan_ratio * 0.2
        
        # Check for zero/negative prices
        if "close" in df.columns:
            invalid_prices = (df["close"] <= 0).sum() / len(df)
            score -= invalid_prices * 0.3
        
        # Check for extreme values
        if "close" in df.columns:
            close = df["close"].dropna()
            if len(close) > 0:
                median = close.median()
                if median > 0:
                    extreme = (close > median * 2).sum() / len(df)
                    score -= extreme * 0.1
        
        # Check OHLC relationships
        if all(col in df.columns for col in required_cols):
            invalid_ohlc = (df["high"] < df["low"]).sum() / len(df)
            score -= invalid_ohlc * 0.3
        
        return max(0.0, min(1.0, score))
    
    def load_generator(
        self,
        fetch_fn: Callable[[int], pd.DataFrame | None],
        interval: str = "1d",
        target_bars: int = 1000,
    ) -> Generator[LoadResult, None, None]:
        """Generator that yields progressive load results.
        
        Yields:
            LoadResult after each chunk is loaded
        """
        min_bars = self._get_min_bars(interval)
        chunk_bars = min_bars
        
        while chunk_bars <= target_bars:
            result = self._load_chunk(fetch_fn, chunk_bars)
            
            if result is not None and not result.empty:
                quality = self._calculate_quality_score(result, interval)
                yield LoadResult(
                    status=LoadStatus.PARTIAL,
                    data=result,
                    bars_loaded=len(result),
                    bars_requested=target_bars,
                    quality_score=quality,
                    load_time_ms=0,  # Not tracked in generator
                    chunks_loaded=1,
                )
            
            chunk_bars = min(chunk_bars + self.chunk_size, target_bars)


# Global loader instance
_loader: ProgressiveDataLoader | None = None
_loader_lock = threading.Lock()


def get_progressive_loader() -> ProgressiveDataLoader:
    """Get or create global progressive loader."""
    global _loader
    with _loader_lock:
        if _loader is None:
            _loader = ProgressiveDataLoader(
                min_bars_intraday=int(env_int("TRADING_MIN_BARS_INTRADAY", "480")),
                min_bars_daily=int(env_int("TRADING_MIN_BARS_DAILY", "14")),
                chunk_size=int(env_int("TRADING_CHUNK_SIZE", "500")),
                max_bars_per_request=int(env_int("TRADING_MAX_BARS_REQUEST", "2000")),
                allow_partial=env_flag("TRADING_ALLOW_PARTIAL", "1"),
                partial_threshold=float(env_text("TRADING_PARTIAL_THRESHOLD", "0.4")),
                max_memory_mb=float(env_text("TRADING_MAX_MEMORY_MB", "500.0")),
            )
        return _loader


def load_with_progressive(
    fetch_fn: Callable[[int], pd.DataFrame | None],
    interval: str = "1d",
    requested_bars: int | None = None,
) -> LoadResult:
    """Load data using global progressive loader."""
    return get_progressive_loader().load(fetch_fn, interval, requested_bars)


def reset_progressive_loader() -> None:
    """Reset global loader (for testing)."""
    global _loader
    with _loader_lock:
        _loader = None
