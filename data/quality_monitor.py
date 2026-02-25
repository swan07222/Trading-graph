"""
Data Quality Monitoring and Validation

Addresses disadvantages:
- Free data providers (Tencent, AkShare, Sina, Yahoo) - not professional-grade
- Auto-failover suggests unreliable connections
- No direct exchange feed connections
- Data delays possible, especially for China A-shares

Features:
- Multi-source data validation and consensus
- Latency monitoring per provider
- Data quality scoring
- Stale data detection
- Anomaly detection (outliers, gaps, flat lines)
- Provider health tracking
"""
from __future__ import annotations

import threading
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum

import numpy as np

from utils.logger import get_logger

log = get_logger(__name__)


class DataQuality(Enum):
    """Data quality rating."""
    EXCELLENT = "excellent"  # < 100ms latency, all validations pass
    GOOD = "good"  # < 500ms latency, minor issues
    FAIR = "fair"  # < 2000ms latency, some issues
    POOR = "poor"  # > 2000ms latency, multiple issues
    STALE = "stale"  # Data too old
    INVALID = "invalid"  # Failed validation


class ProviderHealth(Enum):
    """Data provider health status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class DataQualityReport:
    """Data quality assessment report."""
    symbol: str
    source: str
    timestamp: datetime
    data_timestamp: datetime
    latency_ms: float
    quality: DataQuality
    price: float
    volume: float
    is_stale: bool
    is_anomaly: bool
    validation_errors: list[str]
    quality_score: float  # 0-100


@dataclass
class ProviderStats:
    """Provider statistics tracking."""
    provider_id: str
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    avg_latency_ms: float = 0.0
    last_success_time: datetime | None = None
    last_failure_time: datetime | None = None
    consecutive_failures: int = 0
    health: ProviderHealth = ProviderHealth.UNKNOWN

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_requests == 0:
            return 0.0
        return self.successful_requests / self.total_requests


class DataQualityMonitor:
    """
    Real-time data quality monitoring.

    Features:
    - Latency tracking
    - Staleness detection
    - Anomaly detection
    - Multi-source validation
    - Quality scoring
    """

    def __init__(
        self,
        stale_threshold_seconds: float = 60.0,
        critical_stale_threshold_seconds: float = 300.0,
        max_price_jump_pct: float = 0.10,
        flat_line_tolerance: int = 10,
    ) -> None:
        self.stale_threshold = stale_threshold_seconds
        self.critical_stale_threshold = critical_stale_threshold_seconds
        self.max_price_jump_pct = max_price_jump_pct
        self.flat_line_tolerance = flat_line_tolerance

        self._lock = threading.RLock()
        self._price_history: dict[str, list[tuple[datetime, float]]] = {}
        self._last_data_time: dict[str, datetime] = {}
        self._quality_reports: list[DataQualityReport] = []

    def validate_data(
        self,
        symbol: str,
        source: str,
        price: float,
        volume: float,
        data_timestamp: datetime,
        fetch_timestamp: datetime = None,
    ) -> DataQualityReport:
        """
        Validate incoming data quality.

        Args:
            symbol: Stock code
            source: Data provider
            price: Current price
            volume: Current volume
            data_timestamp: Timestamp of data
            fetch_timestamp: When data was fetched (default: now)

        Returns:
            DataQualityReport with quality assessment
        """
        if fetch_timestamp is None:
            fetch_timestamp = datetime.now()

        validation_errors = []
        quality_issues = 0

        # Calculate latency
        latency_ms = (fetch_timestamp - data_timestamp).total_seconds() * 1000

        # Check staleness
        age_seconds = (fetch_timestamp - data_timestamp).total_seconds()
        is_stale = age_seconds > self.stale_threshold
        is_critical_stale = age_seconds > self.critical_stale_threshold

        if is_critical_stale:
            validation_errors.append(f"Critical stale: {age_seconds:.0f}s old")
            quality_issues += 3
        elif is_stale:
            validation_errors.append(f"Stale: {age_seconds:.0f}s old")
            quality_issues += 1

        # Check price anomalies
        is_anomaly = False
        with self._lock:
            if symbol in self._price_history:
                history = self._price_history[symbol]
                if history:
                    last_price = history[-1][1]
                    if last_price > 0:
                        price_change_pct = abs(price - last_price) / last_price

                        if price_change_pct > self.max_price_jump_pct:
                            validation_errors.append(
                                f"Price jump: {price_change_pct:.1%}"
                            )
                            is_anomaly = True
                            quality_issues += 2

                        # Check for flat line (suspicious)
                        recent_prices = [p for _, p in history[-self.flat_line_tolerance:]]
                        if len(recent_prices) >= self.flat_line_tolerance:
                            if len(set(recent_prices)) == 1:
                                validation_errors.append("Flat line detected")
                                is_anomaly = True
                                quality_issues += 1

            # Update history
            if symbol not in self._price_history:
                self._price_history[symbol] = []
            self._price_history[symbol].append((fetch_timestamp, price))

            # Keep last 1000 prices
            if len(self._price_history[symbol]) > 1000:
                self._price_history[symbol] = self._price_history[symbol][-1000:]

            self._last_data_time[symbol] = fetch_timestamp

        # Check volume anomalies
        if volume < 0:
            validation_errors.append("Negative volume")
            quality_issues += 2
        elif volume == 0:
            validation_errors.append("Zero volume")
            quality_issues += 1

        # Check price validity
        if price <= 0:
            validation_errors.append("Invalid price")
            quality_issues += 3
        elif price > 1e6:
            validation_errors.append("Suspiciously high price")
            quality_issues += 1

        # Calculate quality score (0-100)
        quality_score = max(0, 100 - quality_issues * 15 - min(latency_ms / 100, 50))

        # Determine quality rating
        if quality_score >= 90 and not validation_errors:
            quality = DataQuality.EXCELLENT
        elif quality_score >= 70 and quality_issues <= 1:
            quality = DataQuality.GOOD
        elif quality_score >= 50 and quality_issues <= 2:
            quality = DataQuality.FAIR
        elif is_critical_stale or quality_issues >= 4:
            quality = DataQuality.STALE if is_stale else DataQuality.INVALID
        else:
            quality = DataQuality.POOR

        report = DataQualityReport(
            symbol=symbol,
            source=source,
            timestamp=fetch_timestamp,
            data_timestamp=data_timestamp,
            latency_ms=latency_ms,
            quality=quality,
            price=price,
            volume=volume,
            is_stale=is_stale,
            is_anomaly=is_anomaly,
            validation_errors=validation_errors,
            quality_score=quality_score,
        )

        with self._lock:
            self._quality_reports.append(report)

            # Keep last 1000 reports
            if len(self._quality_reports) > 1000:
                self._quality_reports = self._quality_reports[-1000:]

        if quality in {DataQuality.POOR, DataQuality.STALE, DataQuality.INVALID}:
            log.warning(
                f"Data quality issue: {symbol} from {source} | "
                f"{quality.value} | {', '.join(validation_errors)}"
            )

        return report

    def is_data_fresh(self, symbol: str, max_age_seconds: float = None) -> bool:
        """Check if data is fresh enough."""
        if max_age_seconds is None:
            max_age_seconds = self.stale_threshold

        with self._lock:
            last_time = self._last_data_time.get(symbol)

        if last_time is None:
            return False

        age = (datetime.now() - last_time).total_seconds()
        return age < max_age_seconds

    def get_quality_statistics(
        self,
        symbol: str = None,
        source: str = None,
        since: datetime = None,
    ) -> dict:
        """Get data quality statistics."""
        with self._lock:
            reports = self._quality_reports.copy()

        if symbol:
            reports = [r for r in reports if r.symbol == symbol]
        if source:
            reports = [r for r in reports if r.source == source]
        if since:
            reports = [r for r in reports if r.timestamp >= since]

        if not reports:
            return {"count": 0}

        quality_counts = {}
        for r in reports:
            quality_counts[r.quality.value] = quality_counts.get(r.quality.value, 0) + 1

        avg_latency = np.mean([r.latency_ms for r in reports])
        avg_score = np.mean([r.quality_score for r in reports])
        stale_count = sum(1 for r in reports if r.is_stale)
        anomaly_count = sum(1 for r in reports if r.is_anomaly)

        return {
            "count": len(reports),
            "quality_distribution": quality_counts,
            "avg_latency_ms": round(avg_latency, 1),
            "avg_quality_score": round(avg_score, 1),
            "stale_count": stale_count,
            "anomaly_count": anomaly_count,
            "fresh_rate": 1.0 - stale_count / len(reports),
        }


class ProviderHealthMonitor:
    """
    Monitor health of multiple data providers.

    Features:
    - Per-provider latency tracking
    - Success/failure rate monitoring
    - Automatic health status updates
    - Provider ranking
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_successes: int = 3,
        latency_threshold_ms: float = 5000.0,
    ) -> None:
        self.failure_threshold = failure_threshold
        self.recovery_successes = recovery_successes
        self.latency_threshold_ms = latency_threshold_ms

        self._lock = threading.RLock()
        self._providers: dict[str, ProviderStats] = {}

    def record_request(
        self,
        provider_id: str,
        success: bool,
        latency_ms: float,
    ) -> None:
        """Record provider request result."""
        with self._lock:
            if provider_id not in self._providers:
                self._providers[provider_id] = ProviderStats(provider_id=provider_id)

            stats = self._providers[provider_id]
            stats.total_requests += 1

            if success:
                stats.successful_requests += 1
                stats.last_success_time = datetime.now()
                stats.consecutive_failures = 0
            else:
                stats.failed_requests += 1
                stats.last_failure_time = datetime.now()
                stats.consecutive_failures += 1

            # Update running average latency
            n = stats.successful_requests
            if n > 0:
                stats.avg_latency_ms = (
                    (stats.avg_latency_ms * (n - 1) + latency_ms) / n
                )

            # Update health status
            self._update_health(stats)

    def _update_health(self, stats: ProviderStats) -> None:
        """Update provider health status."""
        if stats.consecutive_failures >= self.failure_threshold:
            stats.health = ProviderHealth.UNHEALTHY
        elif stats.avg_latency_ms > self.latency_threshold_ms:
            stats.health = ProviderHealth.DEGRADED
        elif stats.success_rate < 0.5:
            stats.health = ProviderHealth.DEGRADED
        elif stats.success_rate > 0.9 and stats.avg_latency_ms < self.latency_threshold_ms:
            stats.health = ProviderHealth.HEALTHY
        else:
            stats.health = ProviderHealth.UNKNOWN

    def get_healthy_providers(self) -> list[str]:
        """Get list of healthy providers."""
        with self._lock:
            return [
                provider_id
                for provider_id, stats in self._providers.items()
                if stats.health == ProviderHealth.HEALTHY
            ]

    def get_best_provider(self) -> str | None:
        """Get best provider based on health and latency."""
        with self._lock:
            healthy = [
                (provider_id, stats.avg_latency_ms, stats.success_rate)
                for provider_id, stats in self._providers.items()
                if stats.health == ProviderHealth.HEALTHY
            ]

        if not healthy:
            return None

        # Score by latency and success rate
        def score(item):
            _, latency, success_rate = item
            return success_rate * 1000 / max(latency, 1)

        return max(healthy, key=score)[0]

    def get_provider_status(self) -> dict:
        """Get all provider status."""
        with self._lock:
            return {
                provider_id: {
                    "health": stats.health.value,
                    "success_rate": round(stats.success_rate, 3),
                    "avg_latency_ms": round(stats.avg_latency_ms, 1),
                    "total_requests": stats.total_requests,
                    "consecutive_failures": stats.consecutive_failures,
                    "last_success": (
                        stats.last_success_time.isoformat()
                        if stats.last_success_time else None
                    ),
                }
                for provider_id, stats in self._providers.items()
            }


class MultiSourceValidator:
    """
    Validate data across multiple sources.

    Features:
    - Cross-source price comparison
    - Consensus building
    - Outlier detection
    - Source reliability scoring
    """

    def __init__(
        self,
        max_price_diff_pct: float = 0.03,
        min_sources_for_consensus: int = 2,
    ) -> None:
        self.max_price_diff_pct = max_price_diff_pct
        self.min_sources_for_consensus = min_sources_for_consensus

        self._lock = threading.RLock()
        self._latest_prices: dict[str, dict[str, tuple[datetime, float]]] = {}
        self._source_reliability: dict[str, float] = {}

    def record_price(
        self,
        symbol: str,
        source: str,
        price: float,
        timestamp: datetime,
    ) -> None:
        """Record price from source."""
        with self._lock:
            if symbol not in self._latest_prices:
                self._latest_prices[symbol] = {}

            self._latest_prices[symbol][source] = (timestamp, price)

    def validate_consensus(
        self,
        symbol: str,
        source: str,
        price: float,
    ) -> tuple[bool, str, float | None]:
        """
        Validate price against other sources.

        Args:
            symbol: Stock code
            source: Data source
            price: Price to validate

        Returns:
            (is_valid, reason, consensus_price)
        """
        with self._lock:
            if symbol not in self._latest_prices:
                return True, "No comparison data", None

            sources = self._latest_prices[symbol]

            if len(sources) < self.min_sources_for_consensus:
                return True, "Insufficient sources", None

            # Collect recent prices from other sources
            cutoff = datetime.now() - timedelta(seconds=60)
            other_prices = []
            # FIX: Create snapshot to prevent "dictionary changed size during iteration"
            sources_snapshot = dict(sources)
            for src, (ts, p) in sources_snapshot.items():
                if src != source and ts > cutoff and p > 0:
                    other_prices.append((src, p))

            if len(other_prices) < self.min_sources_for_consensus - 1:
                return True, "Insufficient recent comparison data", None

            # Calculate consensus price (median)
            other_prices_values = [p for _, p in other_prices]
            consensus_price = float(np.median(other_prices_values))

            # Check deviation from consensus
            if consensus_price > 0:
                deviation = abs(price - consensus_price) / consensus_price

                if deviation > self.max_price_diff_pct:
                    # This source is an outlier
                    self._decrease_reliability(source)
                    return False, f"Price deviates {deviation:.1%} from consensus", consensus_price

            # Increase reliability for matching sources
            self._increase_reliability(source)

            return True, "Consensus validated", consensus_price

    def _increase_reliability(self, source: str) -> None:
        """Increase source reliability score."""
        current = self._source_reliability.get(source, 0.5)
        self._source_reliability[source] = min(1.0, current + 0.05)

    def _decrease_reliability(self, source: str) -> None:
        """Decrease source reliability score."""
        current = self._source_reliability.get(source, 0.5)
        self._source_reliability[source] = max(0.0, current - 0.1)

    def get_consensus_price(self, symbol: str) -> float | None:
        """Get consensus price for symbol."""
        with self._lock:
            if symbol not in self._latest_prices:
                return None

            cutoff = datetime.now() - timedelta(seconds=60)
            recent_prices = [
                p for src, (ts, p) in self._latest_prices[symbol].items()
                if ts > cutoff and p > 0
            ]

            if not recent_prices:
                return None

            return float(np.median(recent_prices))

    def get_source_reliability(self) -> dict[str, float]:
        """Get source reliability scores."""
        with self._lock:
            return self._source_reliability.copy()


@dataclass
class DataQualityConfig:
    """Data quality configuration."""
    stale_threshold_seconds: float = 60.0
    critical_stale_threshold_seconds: float = 300.0
    max_price_jump_pct: float = 0.10
    flat_line_tolerance: int = 10
    provider_failure_threshold: int = 5
    max_price_diff_pct: float = 0.03
    min_sources_for_consensus: int = 2


def create_data_quality_pipeline(
    config: DataQualityConfig = None,
) -> tuple[DataQualityMonitor, ProviderHealthMonitor, MultiSourceValidator]:
    """Create data quality pipeline with given configuration."""
    if config is None:
        config = DataQualityConfig()

    quality_monitor = DataQualityMonitor(
        stale_threshold_seconds=config.stale_threshold_seconds,
        critical_stale_threshold_seconds=config.critical_stale_threshold_seconds,
        max_price_jump_pct=config.max_price_jump_pct,
        flat_line_tolerance=config.flat_line_tolerance,
    )

    provider_monitor = ProviderHealthMonitor(
        failure_threshold=config.provider_failure_threshold,
    )

    validator = MultiSourceValidator(
        max_price_diff_pct=config.max_price_diff_pct,
        min_sources_for_consensus=config.min_sources_for_consensus,
    )

    return quality_monitor, provider_monitor, validator
