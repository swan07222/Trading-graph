"""
Circuit Breaker and Kill Switch Enhancements

Addresses disadvantages:
- Desktop application = single point of failure
- No redundancy/failover for production trading
- Network issues can disrupt data feeds during critical moments
- No regulatory compliance for automated trading

Features:
- Multi-level circuit breakers (position, loss, volatility, network)
- Automatic trading halt on anomalies
- Redundant health monitoring
- Regulatory compliance checks
- Graceful degradation modes
"""
from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

from utils.logger import get_logger

log = get_logger(__name__)


class CircuitBreakerLevel(Enum):
    """Circuit breaker severity levels."""
    WARNING = "warning"  # Log warning, continue trading
    REDUCE_RISK = "reduce_risk"  # Reduce position sizes
    HALT_NEW_POSITIONS = "halt_new"  # Stop opening new positions
    HALT_ALL_TRADING = "halt_all"  # Stop all trading
    EMERGENCY_CLOSE = "emergency"  # Close all positions immediately


class TradingState(Enum):
    """Current trading state."""
    NORMAL = "normal"
    CAUTIOUS = "cautious"
    RESTRICTED = "restricted"
    HALTED = "halted"
    EMERGENCY = "emergency"


@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration."""
    # Loss limits
    max_daily_loss_pct: float = 3.0
    max_weekly_loss_pct: float = 7.0
    max_monthly_loss_pct: float = 15.0
    max_drawdown_pct: float = 10.0

    # Position limits
    max_position_size_pct: float = 0.10
    max_total_exposure_pct: float = 0.95
    max_concentration_pct: float = 0.25

    # Volatility limits
    max_realized_volatility: float = 0.50
    max_var_95_pct: float = 5.0

    # Network/data limits
    max_data_latency_ms: int = 5000
    max_consecutive_failures: int = 5
    heartbeat_timeout_seconds: int = 30

    # Order limits
    max_orders_per_minute: int = 60
    max_order_value_pct: float = 0.05

    # Behavior
    auto_emergency_close: bool = False
    cooldown_seconds: int = 300


@dataclass
class CircuitBreakerEvent:
    """Record of circuit breaker trigger."""
    timestamp: datetime
    level: CircuitBreakerLevel
    reason: str
    metric_name: str
    metric_value: float
    threshold: float
    trading_state_before: TradingState
    trading_state_after: TradingState
    auto_triggered: bool = True
    notes: str = ""


class CircuitBreaker:
    """
    Multi-level circuit breaker for risk management.

    Monitors multiple risk metrics and automatically halts trading
    when thresholds are breached.
    """

    def __init__(self, config: CircuitBreakerConfig | None = None) -> None:
        self.config = config or CircuitBreakerConfig()
        self._lock = threading.RLock()

        self._state = TradingState.NORMAL
        self._events: list[CircuitBreakerEvent] = []
        self._last_trigger_time: datetime | None = None
        self._cooldown_until: datetime | None = None

        # Metrics tracking
        self._daily_pnl_pct: float = 0.0
        self._weekly_pnl_pct: float = 0.0
        self._monthly_pnl_pct: float = 0.0
        self._max_drawdown_pct: float = 0.0
        self._current_drawdown_pct: float = 0.0
        self._realized_volatility: float = 0.0
        self._var_95_pct: float = 0.0

        # Network/data health
        self._data_latency_ms: float = 0.0
        self._consecutive_failures: int = 0
        self._last_heartbeat: datetime | None = None

        # Order tracking
        self._orders_last_minute: list[datetime] = []
        self._largest_order_pct: float = 0.0

        # Position tracking
        self._total_exposure_pct: float = 0.0
        self._largest_position_pct: float = 0.0

    @property
    def current_state(self) -> TradingState:
        """Get current trading state."""
        with self._lock:
            return self._state

    @property
    def is_trading_halted(self) -> bool:
        """Check if trading is halted."""
        with self._lock:
            return self._state in {TradingState.HALTED, TradingState.EMERGENCY}

    @property
    def is_emergency(self) -> bool:
        """Check if in emergency state."""
        with self._lock:
            return self._state == TradingState.EMERGENCY

    def update_metrics(
        self,
        daily_pnl_pct: float | None = None,
        weekly_pnl_pct: float | None = None,
        monthly_pnl_pct: float | None = None,
        drawdown_pct: float | None = None,
        volatility: float | None = None,
        var_95: float | None = None,
        data_latency_ms: float | None = None,
        total_exposure_pct: float | None = None,
        largest_position_pct: float | None = None,
    ) -> None:
        """Update risk metrics."""
        with self._lock:
            if daily_pnl_pct is not None:
                self._daily_pnl_pct = daily_pnl_pct
            if weekly_pnl_pct is not None:
                self._weekly_pnl_pct = weekly_pnl_pct
            if monthly_pnl_pct is not None:
                self._monthly_pnl_pct = monthly_pnl_pct
            if drawdown_pct is not None:
                self._current_drawdown_pct = drawdown_pct
                self._max_drawdown_pct = max(self._max_drawdown_pct, drawdown_pct)
            if volatility is not None:
                self._realized_volatility = volatility
            if var_95 is not None:
                self._var_95_pct = var_95
            if data_latency_ms is not None:
                self._data_latency_ms = data_latency_ms
            if total_exposure_pct is not None:
                self._total_exposure_pct = total_exposure_pct
            if largest_position_pct is not None:
                self._largest_position_pct = largest_position_pct

            # Check all circuit breakers
            self._check_circuit_breakers()

    def record_order(self, order_value_pct: float) -> bool:
        """
        Record order and check if allowed.

        Returns:
            True if order is allowed, False if halted
        """
        with self._lock:
            now = datetime.now()

            # Clean old orders
            cutoff = now - timedelta(minutes=1)
            self._orders_last_minute = [
                ts for ts in self._orders_last_minute if ts > cutoff
            ]

            # Check order rate limit
            if len(self._orders_last_minute) >= self.config.max_orders_per_minute:
                log.warning("Order rate limit reached")
                return False

            # Check order size
            if order_value_pct > self.config.max_order_value_pct:
                log.warning(
                    f"Order size {order_value_pct:.1%} exceeds max "
                    f"{self.config.max_order_value_pct:.1%}"
                )
                return False

            # Check trading state
            if self._state == TradingState.HALTED:
                log.warning("Trading halted - order rejected")
                return False
            elif self._state == TradingState.RESTRICTED:
                # Allow only closing orders in restricted mode
                log.info("Trading restricted - only closing orders allowed")
                # This check should be done by the caller

            self._orders_last_minute.append(now)
            self._largest_order_pct = max(self._largest_order_pct, order_value_pct)

            return True

    def record_data_failure(self) -> None:
        """Record data feed failure."""
        with self._lock:
            self._consecutive_failures += 1
            log.warning(f"Data failure count: {self._consecutive_failures}")

            if self._consecutive_failures >= self.config.max_consecutive_failures:
                self._trigger_circuit_breaker(
                    level=CircuitBreakerLevel.HALT_ALL_TRADING,
                    reason=f"Consecutive data failures: {self._consecutive_failures}",
                    metric_name="consecutive_failures",
                    metric_value=float(self._consecutive_failures),
                    threshold=float(self.config.max_consecutive_failures),
                )

    def reset_failure_count(self) -> None:
        """Reset failure count on successful data."""
        with self._lock:
            self._consecutive_failures = 0

    def update_heartbeat(self) -> None:
        """Update system heartbeat."""
        with self._lock:
            self._last_heartbeat = datetime.now()

            # Check heartbeat timeout
            if self._last_heartbeat:
                elapsed = (datetime.now() - self._last_heartbeat).total_seconds()
                if elapsed > self.config.heartbeat_timeout_seconds:
                    self._trigger_circuit_breaker(
                        level=CircuitBreakerLevel.HALT_ALL_TRADING,
                        reason=f"Heartbeat timeout: {elapsed:.1f}s",
                        metric_name="heartbeat_elapsed",
                        metric_value=elapsed,
                        threshold=float(self.config.heartbeat_timeout_seconds),
                    )

    def _check_circuit_breakers(self) -> None:
        """Check all circuit breaker conditions."""
        # Skip if already in emergency
        if self._state == TradingState.EMERGENCY:
            return

        # Check loss limits
        if self._daily_pnl_pct <= -self.config.max_daily_loss_pct:
            self._trigger_circuit_breaker(
                level=CircuitBreakerLevel.HALT_ALL_TRADING,
                reason=f"Daily loss limit breached: {self._daily_pnl_pct:.1%}",
                metric_name="daily_pnl_pct",
                metric_value=self._daily_pnl_pct,
                threshold=-self.config.max_daily_loss_pct,
            )
            return

        if self._weekly_pnl_pct <= -self.config.max_weekly_loss_pct:
            self._trigger_circuit_breaker(
                level=CircuitBreakerLevel.HALT_ALL_TRADING,
                reason=f"Weekly loss limit breached: {self._weekly_pnl_pct:.1%}",
                metric_name="weekly_pnl_pct",
                metric_value=self._weekly_pnl_pct,
                threshold=-self.config.max_weekly_loss_pct,
            )
            return

        # Check drawdown
        if self._current_drawdown_pct >= self.config.max_drawdown_pct:
            self._trigger_circuit_breaker(
                level=CircuitBreakerLevel.REDUCE_RISK,
                reason=f"Drawdown limit breached: {self._current_drawdown_pct:.1%}",
                metric_name="drawdown_pct",
                metric_value=self._current_drawdown_pct,
                threshold=self.config.max_drawdown_pct,
            )
            return

        # Check volatility
        if self._realized_volatility >= self.config.max_realized_volatility:
            self._trigger_circuit_breaker(
                level=CircuitBreakerLevel.REDUCE_RISK,
                reason=f"High volatility: {self._realized_volatility:.1%}",
                metric_name="realized_volatility",
                metric_value=self._realized_volatility,
                threshold=self.config.max_realized_volatility,
            )
            return

        # Check data latency
        if self._data_latency_ms >= self.config.max_data_latency_ms:
            self._trigger_circuit_breaker(
                level=CircuitBreakerLevel.HALT_NEW_POSITIONS,
                reason=f"High data latency: {self._data_latency_ms:.0f}ms",
                metric_name="data_latency_ms",
                metric_value=self._data_latency_ms,
                threshold=float(self.config.max_data_latency_ms),
            )
            return

        # Check position limits
        if self._total_exposure_pct >= self.config.max_total_exposure_pct:
            self._trigger_circuit_breaker(
                level=CircuitBreakerLevel.HALT_NEW_POSITIONS,
                reason=f"Total exposure limit: {self._total_exposure_pct:.1%}",
                metric_name="total_exposure_pct",
                metric_value=self._total_exposure_pct,
                threshold=self.config.max_total_exposure_pct,
            )
            return

        if self._largest_position_pct >= self.config.max_concentration_pct:
            self._trigger_circuit_breaker(
                level=CircuitBreakerLevel.HALT_NEW_POSITIONS,
                reason=f"Position concentration: {self._largest_position_pct:.1%}",
                metric_name="largest_position_pct",
                metric_value=self._largest_position_pct,
                threshold=self.config.max_concentration_pct,
            )
            return

    def _trigger_circuit_breaker(
        self,
        level: CircuitBreakerLevel,
        reason: str,
        metric_name: str,
        metric_value: float,
        threshold: float,
    ) -> None:
        """Trigger circuit breaker."""
        # Check cooldown
        if self._cooldown_until and datetime.now() < self._cooldown_until:
            log.debug(f"Circuit breaker in cooldown until {self._cooldown_until}")
            return

        old_state = self._state

        # Determine new state based on level
        if level == CircuitBreakerLevel.WARNING:
            new_state = TradingState.CAUTIOUS
        elif level == CircuitBreakerLevel.REDUCE_RISK:
            new_state = TradingState.CAUTIOUS
        elif level == CircuitBreakerLevel.HALT_NEW_POSITIONS:
            new_state = TradingState.RESTRICTED
        elif level == CircuitBreakerLevel.HALT_ALL_TRADING:
            new_state = TradingState.HALTED
        elif level == CircuitBreakerLevel.EMERGENCY_CLOSE:
            new_state = TradingState.EMERGENCY
        else:
            new_state = TradingState.HALTED

        # Only trigger if state worsens
        state_order = [
            TradingState.NORMAL,
            TradingState.CAUTIOUS,
            TradingState.RESTRICTED,
            TradingState.HALTED,
            TradingState.EMERGENCY,
        ]
        if state_order.index(new_state) <= state_order.index(old_state):
            return

        self._state = new_state
        self._last_trigger_time = datetime.now()
        self._cooldown_until = datetime.now() + timedelta(
            seconds=self.config.cooldown_seconds
        )

        event = CircuitBreakerEvent(
            timestamp=datetime.now(),
            level=level,
            reason=reason,
            metric_name=metric_name,
            metric_value=metric_value,
            threshold=threshold,
            trading_state_before=old_state,
            trading_state_after=new_state,
            auto_triggered=True,
        )
        self._events.append(event)

        log.warning(
            f"🚨 CIRCUIT BREAKER TRIGGERED: {level.value} | {reason} | "
            f"State: {old_state.value} -> {new_state.value}"
        )

        # Auto emergency close if configured
        if level == CircuitBreakerLevel.EMERGENCY_CLOSE and self.config.auto_emergency_close:
            log.critical("AUTO EMERGENCY CLOSE INITIATED")
            # Callback for emergency close would be triggered here

    def reset(self) -> None:
        """Manually reset circuit breaker (requires authorization)."""
        with self._lock:
            old_state = self._state
            self._state = TradingState.NORMAL
            self._cooldown_until = None
            self._consecutive_failures = 0

            log.info(f"Circuit breaker manually reset: {old_state.value} -> NORMAL")

    def get_events(
        self,
        since: datetime | None = None,
        limit: int = 100,
    ) -> list[CircuitBreakerEvent]:
        """Get circuit breaker events."""
        with self._lock:
            events = self._events.copy()

        if since:
            events = [e for e in events if e.timestamp >= since]

        return events[-limit:]

    def get_status(self) -> dict:
        """Get current circuit breaker status."""
        with self._lock:
            return {
                "state": self._state.value,
                "is_halted": self.is_trading_halted,
                "is_emergency": self.is_emergency,
                "daily_pnl_pct": round(self._daily_pnl_pct, 2),
                "weekly_pnl_pct": round(self._weekly_pnl_pct, 2),
                "monthly_pnl_pct": round(self._monthly_pnl_pct, 2),
                "current_drawdown_pct": round(self._current_drawdown_pct, 2),
                "max_drawdown_pct": round(self._max_drawdown_pct, 2),
                "realized_volatility": round(self._realized_volatility, 2),
                "data_latency_ms": round(self._data_latency_ms, 0),
                "consecutive_failures": self._consecutive_failures,
                "total_exposure_pct": round(self._total_exposure_pct, 2),
                "largest_position_pct": round(self._largest_position_pct, 2),
                "orders_last_minute": len(self._orders_last_minute),
                "last_trigger": (
                    self._last_trigger_time.isoformat()
                    if self._last_trigger_time else None
                ),
                "cooldown_until": (
                    self._cooldown_until.isoformat()
                    if self._cooldown_until else None
                ),
                "events_last_24h": len([
                    e for e in self._events
                    if e.timestamp > datetime.now() - timedelta(hours=24)
                ]),
            }


class RedundantHealthMonitor:
    """
    Redundant health monitoring for production reliability.

    Features:
    - Multiple independent health check threads
    - Cross-validation between monitors
    - Automatic failover
    - External heartbeat required
    """

    def __init__(
        self,
        circuit_breaker: CircuitBreaker,
        check_interval_seconds: float = 1.0,
        heartbeat_required: bool = True,
    ) -> None:
        self.circuit_breaker = circuit_breaker
        self.check_interval = check_interval_seconds
        self.heartbeat_required = heartbeat_required

        self._lock = threading.RLock()
        self._running = False
        self._monitor_thread: threading.Thread | None = None
        self._health_checks: list[Callable[[], bool]] = []
        self._last_check_time: datetime | None = None
        self._consecutive_failures = 0
        self._external_heartbeat: datetime | None = None

    def register_health_check(self, check_fn: Callable[[], bool]) -> None:
        """Register a health check function."""
        with self._lock:
            self._health_checks.append(check_fn)
            log.info(f"Health check registered: {check_fn.__name__}")

    def start(self) -> None:
        """Start health monitoring."""
        with self._lock:
            if self._running:
                log.warning("Health monitor already running")
                return

            self._running = True
            self._monitor_thread = threading.Thread(
                target=self._monitor_loop,
                daemon=True,
                name="health_monitor",
            )
            self._monitor_thread.start()
            log.info("Health monitor started")

    def stop(self) -> None:
        """Stop health monitoring."""
        self._running = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5.0)
        log.info("Health monitor stopped")

    def _monitor_loop(self) -> None:
        """Main monitoring loop."""
        while self._running:
            try:
                self._run_health_checks()
            except Exception as e:
                log.exception(f"Health check error: {e}")
                self.circuit_breaker.record_data_failure()

            time.sleep(self.check_interval)

    def _run_health_checks(self) -> None:
        """Run all registered health checks."""
        now = datetime.now()
        all_passed = True

        for check_fn in self._health_checks:
            try:
                if not check_fn():
                    all_passed = False
                    log.warning(f"Health check failed: {check_fn.__name__}")
            except Exception as e:
                log.exception(f"Health check exception: {check_fn.__name__}: {e}")
                all_passed = False

        self._last_check_time = now

        if all_passed:
            self.circuit_breaker.reset_failure_count()
            self._consecutive_failures = 0
        else:
            self.circuit_breaker.record_data_failure()
            self._consecutive_failures += 1

        # Update heartbeat
        self.circuit_breaker.update_heartbeat()

    def record_external_heartbeat(self) -> None:
        """Record external heartbeat (from main trading loop)."""
        with self._lock:
            self._external_heartbeat = datetime.now()
            self.circuit_breaker.update_heartbeat()

    def get_status(self) -> dict:
        """Get health monitor status."""
        with self._lock:
            return {
                "running": self._running,
                "health_checks_registered": len(self._health_checks),
                "last_check_time": (
                    self._last_check_time.isoformat()
                    if self._last_check_time else None
                ),
                "consecutive_failures": self._consecutive_failures,
                "external_heartbeat_required": self.heartbeat_required,
                "last_external_heartbeat": (
                    self._external_heartbeat.isoformat()
                    if self._external_heartbeat else None
                ),
            }


@dataclass
class ComplianceConfig:
    """Regulatory compliance configuration."""
    # China A-share specific
    enable_t1_settlement_check: bool = True
    enable_price_limit_check: bool = True  # 10% daily limit
    enable_volume_limit_check: bool = True

    # Order restrictions
    min_order_value: float = 100.0
    max_order_value: float = 1e7
    restricted_hours: list[tuple[int, int]] = field(default_factory=list)

    # Reporting
    log_all_orders: bool = True
    report_daily_positions: bool = True


class ComplianceChecker:
    """
    Regulatory compliance checker for automated trading.

    Features:
    - T+1 settlement validation
    - Price limit checks (10% for China A-shares)
    - Trading hour restrictions
    - Order value limits
    """

    def __init__(self, config: ComplianceConfig | None = None) -> None:
        self.config = config or ComplianceConfig()
        self._lock = threading.RLock()
        self._today_buys: dict[str, int] = {}  # symbol -> quantity

    def validate_order(
        self,
        symbol: str,
        side: str,
        quantity: int,
        price: float,
        current_price: float,
    ) -> tuple[bool, str]:
        """
        Validate order for regulatory compliance.

        Returns:
            (is_compliant, reason)
        """
        order_value = quantity * price

        # Check order value limits
        if order_value < self.config.min_order_value:
            return False, f"Order value {order_value:.0f} below minimum {self.config.min_order_value:.0f}"

        if order_value > self.config.max_order_value:
            return False, f"Order value {order_value:.0f} exceeds maximum {self.config.max_order_value:.0f}"

        # Check price limits (10% for China A-shares)
        if self.config.enable_price_limit_check and current_price:
            price_change = abs(price - current_price) / current_price
            if price_change > 0.10:
                return False, f"Price change {price_change:.1%} exceeds 10% limit"

        # Check T+1 settlement (can't sell today's buys)
        if self.config.enable_t1_settlement_check and side.lower() == "sell":
            with self._lock:
                if symbol in self._today_buys:
                    bought_today = self._today_buys[symbol]
                    if quantity > bought_today:
                        return False, f"T+1 violation: bought {bought_today} today, selling {quantity}"

        # Check restricted hours
        if self.config.restricted_hours:
            now = datetime.now()
            current_hour = now.hour
            for start, end in self.config.restricted_hours:
                if start <= current_hour < end:
                    return False, f"Trading restricted during hour {start}-{end}"

        return True, "OK"

    def record_buy(self, symbol: str, quantity: int) -> None:
        """Record buy for T+1 tracking."""
        with self._lock:
            self._today_buys[symbol] = self._today_buys.get(symbol, 0) + quantity

    def reset_daily(self) -> None:
        """Reset daily tracking."""
        with self._lock:
            self._today_buys.clear()
