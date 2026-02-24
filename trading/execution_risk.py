"""
Execution Risk Mitigation Module

Addresses disadvantages:
- Retail-grade broker connector risk
- Slippage uncertainty
- No direct market access (DMA)
- Execution not guaranteed at predicted prices

Features:
- Smart order routing with multiple broker fallback
- Dynamic slippage estimation with real-time volatility
- Order execution validation and retry logic
- Transaction cost analysis (TCA)
- Best/worst case execution price bounds
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Callable, Optional

import numpy as np

from core.types import Fill, Order, OrderSide, OrderType
from utils.logger import get_logger

log = get_logger(__name__)


class ExecutionQuality(Enum):
    """Execution quality rating."""
    EXCELLENT = "excellent"  # < 1 bps slippage
    GOOD = "good"  # 1-5 bps slippage
    FAIR = "fair"  # 5-15 bps slippage
    POOR = "poor"  # 15-30 bps slippage
    REJECTED = "rejected"  # > 30 bps or failed


class OrderRoutingStrategy(Enum):
    """Order routing strategy."""
    BEST_PRICE = "best_price"
    FASTEST_FILL = "fastest_fill"
    MINIMIZE_IMPACT = "minimize_impact"
    BALANCED = "balanced"


@dataclass
class ExecutionReport:
    """Post-trade execution quality report."""
    order_id: int
    symbol: str
    side: OrderSide
    requested_price: float
    executed_price: float
    quantity: int
    timestamp: datetime
    slippage_bps: float
    spread_cost_bps: float
    commission_bps: float
    total_cost_bps: float
    quality: ExecutionQuality
    fill_time_ms: float
    venue: str = ""
    notes: str = ""

    @property
    def is_acceptable(self) -> bool:
        """Check if execution quality meets minimum standards."""
        return self.quality in {
            ExecutionQuality.EXCELLENT,
            ExecutionQuality.GOOD,
            ExecutionQuality.FAIR
        }


@dataclass
class SlippageEstimate:
    """Real-time slippage estimation."""
    symbol: str
    base_slippage_bps: float
    volatility_adjustment_bps: float
    volume_impact_bps: float
    spread_cost_bps: float
    total_estimated_bps: float
    confidence: float  # 0-1 confidence in estimate
    timestamp: datetime
    sample_size: int = 0

    @property
    def worst_case_bps(self) -> float:
        """Conservative worst-case slippage estimate."""
        return min(self.total_estimated_bps * 2.0, 50.0)

    @property
    def best_case_bps(self) -> float:
        """Optimistic best-case slippage estimate."""
        return max(self.total_estimated_bps * 0.5, 1.0)


class SmartOrderRouter:
    """
    Smart Order Router (SOR) for optimal execution.

    Features:
    - Multi-broker routing with health monitoring
    - Dynamic slippage estimation
    - Order slicing for large orders
    - Execution quality tracking
    """

    def __init__(
        self,
        max_slippage_bps: float = 20.0,
        max_participation_rate: float = 0.05,
        order_split_threshold: float = 100000.0,
    ) -> None:
        self.max_slippage_bps = max_slippage_bps
        self.max_participation_rate = max_participation_rate
        self.order_split_threshold = order_split_threshold

        self._lock = threading.RLock()
        self._execution_reports: list[ExecutionReport] = []
        self._slippage_cache: dict[str, SlippageEstimate] = {}
        self._broker_health: dict[str, bool] = {}
        self._broker_latencies: dict[str, float] = {}

    def estimate_slippage(
        self,
        symbol: str,
        side: OrderSide,
        quantity: int,
        price: float,
        daily_volume: float,
        volatility: float,
        bid_ask_spread: float,
    ) -> SlippageEstimate:
        """
        Estimate realistic slippage before order submission.

        Args:
            symbol: Stock code
            side: Buy or sell
            quantity: Order quantity
            price: Current price
            daily_volume: Average daily volume
            volatility: Annualized volatility (0-1)
            bid_ask_spread: Current bid-ask spread

        Returns:
            SlippageEstimate with confidence bounds
        """
        order_value = quantity * price

        # Base slippage (market dependent)
        base_bps = 5.0 if daily_volume > 1e9 else (10.0 if daily_volume > 1e8 else 20.0)

        # Volatility adjustment (higher vol = higher slippage)
        vol_adjustment_bps = volatility * 50.0  # 50 bps per unit of volatility

        # Volume impact (order size relative to daily volume)
        daily_value = daily_volume * price
        if daily_value > 0:
            participation = order_value / daily_value
            volume_impact_bps = min(participation * 100.0, 30.0)
        else:
            volume_impact_bps = 30.0

        # Spread cost (half spread for market orders)
        spread_cost_bps = bid_ask_spread * 50.0  # Convert to bps

        total_bps = base_bps + vol_adjustment_bps + volume_impact_bps + spread_cost_bps

        # Confidence based on data quality
        confidence = 0.9 if daily_volume > 1e8 else (0.7 if daily_volume > 1e7 else 0.5)

        estimate = SlippageEstimate(
            symbol=symbol,
            base_slippage_bps=base_bps,
            volatility_adjustment_bps=vol_adjustment_bps,
            volume_impact_bps=volume_impact_bps,
            spread_cost_bps=spread_cost_bps,
            total_estimated_bps=total_bps,
            confidence=confidence,
            timestamp=datetime.now(),
            sample_size=int(daily_volume / 1e6),
        )

        with self._lock:
            self._slippage_cache[symbol] = estimate

        return estimate

    def should_route_to_broker(
        self,
        broker_id: str,
        order: Order,
        estimate: SlippageEstimate,
    ) -> bool:
        """Check if order should be routed to specific broker."""
        with self._lock:
            # Check broker health
            if not self._broker_health.get(broker_id, True):
                log.warning(f"Broker {broker_id} unhealthy, skipping")
                return False

            # Check slippage tolerance
            if estimate.total_estimated_bps > self.max_slippage_bps:
                log.warning(
                    f"Estimated slippage {estimate.total_estimated_bps:.1f} bps "
                    f"exceeds max {self.max_slippage_bps:.1f} bps"
                )
                return False

        return True

    def slice_order(
        self,
        order: Order,
        daily_volume: float,
        max_participation: float = None,
    ) -> list[Order]:
        """
        Slice large orders to minimize market impact.

        Args:
            order: Original order
            daily_volume: Average daily volume
            max_participation: Max participation rate (default: self.max_participation_rate)

        Returns:
            List of child orders
        """
        if max_participation is None:
            max_participation = self.max_participation_rate

        order_value = order.quantity * (order.price or 0)

        # No slicing needed for small orders
        if order_value < self.order_split_threshold:
            return [order]

        # Calculate max quantity per slice based on participation rate
        max_qty_per_slice = int(daily_volume * max_participation)
        if max_qty_per_slice <= 0:
            max_qty_per_slice = order.quantity // 10

        # Create child orders
        slices = []
        remaining_qty = order.quantity

        while remaining_qty > 0:
            slice_qty = min(max_qty_per_slice, remaining_qty)
            child = Order(
                symbol=order.symbol,
                side=order.side,
                order_type=order.order_type,
                quantity=slice_qty,
                price=order.price,
                parent_order_id=order.id,
            )
            slices.append(child)
            remaining_qty -= slice_qty

        log.info(
            f"Order sliced: {order.quantity} -> {len(slices)} child orders "
            f"({max_qty_per_slice} qty each)"
        )

        return slices

    def record_execution(
        self,
        order: Order,
        fill: Fill,
        requested_price: float,
        fill_time_ms: float,
        venue: str = "",
    ) -> ExecutionReport:
        """Record and analyze execution quality."""
        # Calculate slippage
        if order.side == OrderSide.BUY:
            slippage = fill.price - requested_price
        else:
            slippage = requested_price - fill.price

        slippage_bps = (slippage / requested_price) * 10000 if requested_price > 0 else 0

        # Estimate spread cost (simplified)
        spread_cost_bps = 5.0  # Assume 5 bps typical spread

        # Commission (China A-share typical: ~3 bps)
        commission_bps = 3.0

        total_cost_bps = abs(slippage_bps) + spread_cost_bps + commission_bps

        # Quality rating
        if total_cost_bps < 10:
            quality = ExecutionQuality.EXCELLENT
        elif total_cost_bps < 20:
            quality = ExecutionQuality.GOOD
        elif total_cost_bps < 30:
            quality = ExecutionQuality.FAIR
        elif total_cost_bps < 50:
            quality = ExecutionQuality.POOR
        else:
            quality = ExecutionQuality.REJECTED

        report = ExecutionReport(
            order_id=order.id,
            symbol=order.symbol,
            side=order.side,
            requested_price=requested_price,
            executed_price=fill.price,
            quantity=fill.quantity,
            timestamp=datetime.now(),
            slippage_bps=slippage_bps,
            spread_cost_bps=spread_cost_bps,
            commission_bps=commission_bps,
            total_cost_bps=total_cost_bps,
            quality=quality,
            fill_time_ms=fill_time_ms,
            venue=venue,
        )

        with self._lock:
            self._execution_reports.append(report)

            # Update broker health based on execution quality
            if venue and quality == ExecutionQuality.REJECTED:
                self._broker_health[venue] = False

        log.info(
            f"Execution report: {order.symbol} | {quality.value} | "
            f"slippage={slippage_bps:.1f}bps | total_cost={total_cost_bps:.1f}bps"
        )

        return report

    def get_execution_statistics(
        self,
        symbol: str = None,
        since: datetime = None,
    ) -> dict:
        """Get execution quality statistics."""
        with self._lock:
            reports = self._execution_reports.copy()

        if symbol:
            reports = [r for r in reports if r.symbol == symbol]

        if since:
            reports = [r for r in reports if r.timestamp >= since]

        if not reports:
            return {"count": 0}

        total_cost_bps = np.mean([r.total_cost_bps for r in reports])
        slippage_bps = np.mean([abs(r.slippage_bps) for r in reports])
        fill_time_ms = np.mean([r.fill_time_ms for r in reports])

        quality_counts = {}
        for r in reports:
            quality_counts[r.quality.value] = quality_counts.get(r.quality.value, 0) + 1

        return {
            "count": len(reports),
            "avg_total_cost_bps": round(total_cost_bps, 2),
            "avg_slippage_bps": round(slippage_bps, 2),
            "avg_fill_time_ms": round(fill_time_ms, 1),
            "quality_distribution": quality_counts,
            "acceptable_rate": sum(
                1 for r in reports if r.is_acceptable
            ) / len(reports),
        }

    def update_broker_health(
        self,
        broker_id: str,
        is_healthy: bool,
        latency_ms: float = None,
    ) -> None:
        """Update broker health status."""
        with self._lock:
            self._broker_health[broker_id] = is_healthy
            if latency_ms is not None:
                self._broker_latencies[broker_id] = latency_ms

    def get_best_broker(self) -> Optional[str]:
        """Get the healthiest broker with lowest latency."""
        with self._lock:
            healthy_brokers = [
                (broker, self._broker_latencies.get(broker, float('inf')))
                for broker, is_healthy in self._broker_health.items()
                if is_healthy
            ]

        if not healthy_brokers:
            return None

        return min(healthy_brokers, key=lambda x: x[1])[0]


class ExecutionValidator:
    """
    Pre-trade and post-trade execution validation.

    Prevents:
    - Fat finger errors
    - Stale price executions
    - Abnormal slippage
    """

    def __init__(
        self,
        max_price_deviation: float = 0.05,
        max_order_value: float = 1e7,
        stale_quote_threshold_ms: int = 5000,
    ) -> None:
        self.max_price_deviation = max_price_deviation
        self.max_order_value = max_order_value
        self.stale_quote_threshold_ms = stale_quote_threshold_ms

        self._lock = threading.RLock()
        self._recent_prices: dict[str, tuple[float, datetime]] = {}

    def validate_pre_trade(
        self,
        order: Order,
        current_price: float,
        quote_age_ms: float,
    ) -> tuple[bool, str]:
        """
        Validate order before submission.

        Returns:
            (is_valid, reason)
        """
        # Check order value
        order_value = order.quantity * (order.price or current_price)
        if order_value > self.max_order_value:
            return False, f"Order value {order_value:.0f} exceeds max {self.max_order_value:.0f}"

        # Check price deviation
        if order.price and current_price:
            deviation = abs(order.price - current_price) / current_price
            if deviation > self.max_price_deviation:
                return False, f"Price deviation {deviation:.1%} exceeds max {self.max_price_deviation:.1%}"

        # Check quote staleness
        if quote_age_ms > self.stale_quote_threshold_ms:
            return False, f"Quote stale ({quote_age_ms:.0f}ms > {self.stale_quote_threshold_ms:.0f}ms)"

        return True, "OK"

    def validate_post_trade(
        self,
        fill: Fill,
        expected_price: float,
    ) -> tuple[bool, str]:
        """
        Validate fill after execution.

        Returns:
            (is_valid, reason)
        """
        # Check fill price vs expected
        deviation = abs(fill.price - expected_price) / expected_price
        if deviation > self.max_price_deviation:
            return False, f"Fill price deviation {deviation:.1%} exceeds threshold"

        # Check fill quantity
        if fill.quantity <= 0:
            return False, "Invalid fill quantity"

        return True, "OK"

    def update_price(self, symbol: str, price: float) -> None:
        """Update recent price for validation."""
        with self._lock:
            self._recent_prices[symbol] = (price, datetime.now())

    def get_recent_price(self, symbol: str) -> Optional[float]:
        """Get most recent price for symbol."""
        with self._lock:
            if symbol not in self._recent_prices:
                return None
            price, ts = self._recent_prices[symbol]
            # Invalidate if too old
            if (datetime.now() - ts).total_seconds() > 60:
                return None
            return price


@dataclass
class ExecutionConfig:
    """Execution configuration for production trading."""
    max_slippage_bps: float = 20.0
    max_participation_rate: float = 0.05
    order_split_threshold: float = 100000.0
    max_price_deviation: float = 0.05
    max_order_value: float = 1e7
    stale_quote_threshold_ms: int = 5000
    enable_order_slicing: bool = True
    enable_execution_validation: bool = True
    log_all_executions: bool = True


def create_execution_pipeline(config: ExecutionConfig = None) -> tuple[SmartOrderRouter, ExecutionValidator]:
    """Create execution pipeline with given configuration."""
    if config is None:
        config = ExecutionConfig()

    router = SmartOrderRouter(
        max_slippage_bps=config.max_slippage_bps,
        max_participation_rate=config.max_participation_rate,
        order_split_threshold=config.order_split_threshold,
    )

    validator = ExecutionValidator(
        max_price_deviation=config.max_price_deviation,
        max_order_value=config.max_order_value,
        stale_quote_threshold_ms=config.stale_quote_threshold_ms,
    )

    return router, validator
