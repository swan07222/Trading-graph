# trading/risk.py

import threading
from collections import deque
from datetime import date, datetime, timedelta

import numpy as np

from config.settings import CONFIG
from core.events import EVENT_BUS, Event, EventType, RiskEvent
from core.types import (
    Account,
    OrderSide,
    OrderStatus,
    Position,
    RiskLevel,
    RiskMetrics,
)
from utils.logger import get_logger

log = get_logger(__name__)

# Audit log — safe import with fallback stub
# FIX(7): utils.security may not exist; provide no-op fallback

class _NoOpAuditLog:
    """Stub audit log when utils.security is unavailable."""

    def log_risk_event(self, risk_type: str, details: dict) -> None:
        pass

    def log_order(self, *args, **kwargs) -> None:
        pass

    def __getattr__(self, name: str):
        return lambda *a, **kw: None

def _get_audit_log():
    """Import real audit log or return stub."""
    try:
        from utils.security import get_audit_log
        return get_audit_log()
    except Exception as e:
        log.debug("Audit log unavailable; using no-op stub: %s", e)
        return _NoOpAuditLog()

# Cost estimation helper (single source of truth)
# FIX(6,14): Validates inputs; always uses CONFIG.trading.X consistently

def _estimate_order_cost(
    quantity: int | None,
    price: float | None,
    side: OrderSide = OrderSide.BUY,
) -> tuple[float, float, float]:
    """Estimate total order cost including slippage, commission, and stamp tax.

    Args:
        quantity: Number of shares (must be > 0)
        price: Price per share (must be > 0)
        side: BUY or SELL

    Returns:
        (estimated_price, notional, total_cost)
        For BUY:  total_cost = notional + fees  (cash needed)
        For SELL: total_cost = fees only         (proceeds = notional - fees)

    Raises:
        ValueError: If quantity <= 0, price <= 0, or either is None
    """
    if quantity is None:
        raise ValueError("quantity must not be None")
    if price is None:
        raise ValueError("price must not be None")
    if quantity <= 0:
        raise ValueError(f"quantity must be > 0, got {quantity}")
    if price <= 0:
        raise ValueError(f"price must be > 0, got {price}")

    slip = float(CONFIG.trading.slippage)
    comm_rate = float(CONFIG.trading.commission)
    comm_min = 5.0

    if side == OrderSide.BUY:
        est_price = price * (1.0 + slip)
    else:
        est_price = price * (1.0 - slip)

    notional = quantity * est_price
    commission = max(comm_min, notional * comm_rate)

    stamp_tax = 0.0
    if side == OrderSide.SELL:
        stamp_tax = notional * float(CONFIG.trading.stamp_tax)

    fees = commission + stamp_tax

    if side == OrderSide.BUY:
        total_cost = notional + fees
    else:
        total_cost = fees

    return est_price, notional, total_cost

_oms_module = None
_kill_switch_module = None
_constants_module = None
_feeds_module = None
_oms_cache_ts: float = 0.0
_oms_cache_val = None
_oms_cache_ttl: float = 5.0  # 5 second cache for OMS

def _get_oms():
    """Cached deferred import for OMS with TTL.

    FIX: Adds TTL-based caching to avoid repeated calls.
    """
    global _oms_module, _oms_cache_ts, _oms_cache_val
    import time

    now = time.time()
    if _oms_cache_val is not None and (now - _oms_cache_ts) < _oms_cache_ttl:
        return _oms_cache_val

    if _oms_module is None:
        from trading import oms as _m
        _oms_module = _m

    _oms_cache_val = _m.get_oms()
    _oms_cache_ts = now
    return _oms_cache_val

def _get_kill_switch():
    """Cached deferred import for kill switch."""
    global _kill_switch_module
    if _kill_switch_module is None:
        from trading import kill_switch as _m
        _kill_switch_module = _m
    return _kill_switch_module.get_kill_switch()

def _get_lot_size(symbol: str) -> int:
    """Cached deferred import for lot size."""
    global _constants_module
    if _constants_module is None:
        from core import constants as _m
        _constants_module = _m
    return _constants_module.get_lot_size(symbol)

def _get_feed_quote(symbol: str):
    """Cached deferred import for feed manager quote."""
    global _feeds_module
    if _feeds_module is None:
        try:
            from data import feeds as _m
            _feeds_module = _m
        except ImportError:
            return None
    try:
        return _feeds_module.get_feed_manager().get_quote(symbol)
    except Exception:
        return None

class RiskManager:
    """Production risk management system.

    Thread-safety
    -------------
    All public methods acquire ``self._lock``.
    Internal methods prefixed with ``_`` assume the lock is already held
    (they are only called from within locked sections).

    The lock is an ``RLock`` so re-entrant calls (e.g. ``_check_risk_breaches``
    → ``get_metrics``) are safe.
    """

    # --- Class-level constants ---
    MIN_RETURNS_FOR_VAR: int = 20
    MAX_RETURNS_HISTORY: int = 252
    MAX_EQUITY_HISTORY_DAYS: int = 400
    DEFAULT_VAR_PCT: float = 0.02
    DEFAULT_ES_PCT: float = 0.03
    MAX_VAR_PCT_THRESHOLD: float = 5.0
    MAX_ADD_PCT_WHEN_HIGH_VAR: float = 2.0
    TOP3_CONCENTRATION_LIMIT: float = 50.0
    MAX_ERRORS_PER_MINUTE: int = 5
    MAX_AFFORDABLE_FACTOR: float = 0.95
    STALENESS_THRESHOLD_DEFAULT: float = 30.0
    RISK_EVENT_THROTTLE_SECONDS: float = 60.0
    MAX_TRACKED_SYMBOLS: int = 500  # FIX(9): bound quote timestamps
    DAILY_RETURN_LOOKBACK_DAYS: int = 30  # FIX(12): extended from 15

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._audit = _get_audit_log()  # FIX(7): safe import

        self._account: Account | None = None
        self._initial_equity: float = 0.0
        self._daily_start_equity: float = 0.0
        self._peak_equity: float = 0.0
        self._max_drawdown_pct: float = 0.0
        self._last_date: date = date.today()

        # FIX(2): deque with maxlen for O(1) append and auto-prune
        self._returns_history: deque = deque(maxlen=self.MAX_RETURNS_HISTORY)

        self._trades_today: int = 0
        self._orders_submitted_today: int = 0

        # FIX(8): deque for O(1) appendleft; we prune manually by time
        self._orders_this_minute: deque = deque()
        self._errors_this_minute: deque = deque()

        # Quote timestamps for staleness — FIX(9): bounded dict
        self._quote_timestamps: dict[str, datetime] = {}
        self._staleness_threshold_seconds: float = self.STALENESS_THRESHOLD_DEFAULT
        self._equity_by_day: dict[date, float] = {}
        self._last_var_day: date | None = None
        self._last_breach: dict[str, datetime] = {}

        # FIX(1): Removed _on_trade (was no-op). Only subscribe to
        # events we actually handle.
        self._event_handlers = {
            EventType.ERROR: self._on_error,
            EventType.TICK: self._on_tick,
        }
        for event_type, handler in self._event_handlers.items():
            EVENT_BUS.subscribe(event_type, handler)

    def shutdown(self) -> None:
        """Unsubscribe from all events. Call before discarding this instance."""
        for event_type, handler in self._event_handlers.items():
            try:
                EVENT_BUS.unsubscribe(event_type, handler)
            except Exception as e:
                log.debug(
                    "RiskManager unsubscribe failed for %s: %s",
                    event_type,
                    e,
                )

    # =========================================================================
    # =========================================================================

    def initialize(self, account: Account) -> None:
        """Initialize with account state."""
        with self._lock:
            self._account = account
            self._initial_equity = account.equity
            self._daily_start_equity = account.equity
            self._peak_equity = account.equity
            self._last_date = date.today()
            self._trades_today = 0
            self._orders_submitted_today = 0
            self._max_drawdown_pct = 0.0
            log.info(f"Risk manager initialized: equity=¥{account.equity:,.2f}")

    def update(self, account: Account) -> None:
        """Update with current account state. Called on every portfolio refresh."""
        if account is None:
            log.warning("RiskManager.update() called with None account")
            return

        with self._lock:
            if self._account is None:
                self.initialize(account)
                return

            old_equity = self._account.equity
            self._account = account

            if account.equity > self._peak_equity:
                self._peak_equity = account.equity

            today = date.today()
            if today != self._last_date:
                self._record_daily_return(old_equity)
                self._new_day(old_equity)
                self._last_date = today

            self._equity_by_day[today] = account.equity
            self._prune_equity_history()
            self._check_risk_breaches()

    def record_trade(self) -> None:
        """Record a trade — called by execution engine.
        This is the ONLY way trades should be counted.
        """
        with self._lock:
            self._trades_today += 1

    # =========================================================================
    # =========================================================================

    def _get_unified_account_view(self) -> Account:
        """Build a unified account view that reflects OMS active-order
        reservations.

        Priority:
        1. OMS account (if OMS is available and has an account)
        2. self._account with OMS reservations applied
        3. self._account as-is
        4. Empty Account (last resort)

        MUST be called with self._lock held.
        """
        try:
            oms = _get_oms()
            acc = oms.get_account()
            if acc is not None:
                return acc
        except Exception as e:
            log.debug("OMS account snapshot unavailable in risk view: %s", e)

        if self._account is None:
            return Account()

        # FIX(3): Deep-copy ALL Position fields
        unified_positions = {}
        for sym, pos in self._account.positions.items():
            unified_positions[sym] = Position(
                symbol=pos.symbol,
                name=pos.name,
                quantity=pos.quantity,
                available_qty=pos.available_qty,
                frozen_qty=pos.frozen_qty,
                pending_buy=pos.pending_buy,
                pending_sell=pos.pending_sell,
                avg_cost=pos.avg_cost,
                current_price=pos.current_price,
                realized_pnl=pos.realized_pnl,
                commission_paid=pos.commission_paid,
                opened_at=pos.opened_at,
                last_updated=pos.last_updated,
            )

        unified = Account(
            broker_name=self._account.broker_name,
            account_id=self._account.account_id,
            cash=self._account.cash,
            available=self._account.available,
            frozen=self._account.frozen,
            positions=unified_positions,
            initial_capital=self._account.initial_capital,
            realized_pnl=self._account.realized_pnl,
            commission_paid=self._account.commission_paid,
            daily_start_equity=self._account.daily_start_equity,
            daily_start_date=self._account.daily_start_date,
            peak_equity=self._account.peak_equity,
        )

        # Apply OMS active-order reservations
        self._apply_oms_reservations(unified)
        return unified

    def _apply_oms_reservations(self, account: Account) -> None:
        """Reduce available cash/shares based on active orders in OMS.
        Mutates ``account`` in place.
        """
        try:
            oms = _get_oms()
            active_orders = oms.get_active_orders()
        except Exception as e:
            log.debug(f"Could not apply OMS reservations: {e}")
            return

        for order in active_orders:
            if order.status not in (
                OrderStatus.PENDING, OrderStatus.SUBMITTED,
                OrderStatus.ACCEPTED, OrderStatus.PARTIAL,
            ):
                continue

            remaining_qty = order.quantity - order.filled_qty
            if remaining_qty <= 0:
                continue

            if order.side == OrderSide.BUY:
                reserved = 0.0
                if order.tags:
                    reserved = float(
                        order.tags.get("reserved_cash_remaining", 0.0)
                    )
                if reserved <= 0.0:
                    try:
                        _, _, reserved = _estimate_order_cost(
                            remaining_qty, order.price, OrderSide.BUY,
                        )
                    except ValueError:
                        continue
                account.available = max(0.0, account.available - reserved)
            else:
                pos = account.positions.get(order.symbol)
                if pos:
                    pos.available_qty = max(
                        0, pos.available_qty - remaining_qty
                    )

    # =========================================================================
    # =========================================================================

    def _new_day(self, last_equity: float) -> None:
        """Reset daily counters for new trading day."""
        log.info("Risk manager: New trading day started")
        self._daily_start_equity = last_equity
        self._trades_today = 0
        self._orders_submitted_today = 0
        self._orders_this_minute.clear()
        self._errors_this_minute.clear()

        if self._initial_equity > 0:
            total_return = (last_equity / self._initial_equity - 1) * 100
            log.info(
                f"Previous day end equity: {last_equity:,.2f} "
                f"(Total return: {total_return:+.2f}%)"
            )

    def _prune_equity_history(self) -> None:
        """Prune old entries from _equity_by_day."""
        if len(self._equity_by_day) <= self.MAX_EQUITY_HISTORY_DAYS:
            return
        cutoff = date.today() - timedelta(days=self.MAX_EQUITY_HISTORY_DAYS)
        self._equity_by_day = {
            d: eq for d, eq in self._equity_by_day.items() if d >= cutoff
        }

    # =========================================================================
    # =========================================================================

    def _on_error(self, event: Event) -> None:
        """Handle error event for error rate monitoring."""
        with self._lock:
            self._errors_this_minute.append(datetime.now())

    def _on_tick(self, event: Event) -> None:
        """Handle tick event for quote freshness tracking."""
        with self._lock:
            symbol = getattr(event, 'symbol', None)
            if symbol:
                ts = getattr(event, 'timestamp', None) or datetime.now()
                self._quote_timestamps[symbol] = ts

                # FIX(9): Bound quote timestamps dict
                if len(self._quote_timestamps) > self.MAX_TRACKED_SYMBOLS:
                    sorted_syms = sorted(
                        self._quote_timestamps,
                        key=self._quote_timestamps.get,
                    )
                    for sym in sorted_syms[:len(sorted_syms) // 4]:
                        del self._quote_timestamps[sym]

    # =========================================================================
    # Minute-window deque cleanup helper
    # FIX(13): Single helper instead of repeated inline code
    # =========================================================================

    @staticmethod
    def _prune_minute_window(dq: deque) -> None:
        """Remove entries older than 1 minute from the left of a
        time-sorted deque.
        """
        cutoff = datetime.now() - timedelta(minutes=1)
        while dq and dq[0] < cutoff:
            dq.popleft()

    # =========================================================================
    # =========================================================================

    def get_metrics(self, _account: Account = None) -> RiskMetrics:
        """Calculate comprehensive risk metrics.

        Args:
            _account: Optional pre-fetched unified account view.
                      If None, fetches internally.
        """
        with self._lock:
            if self._account is None:
                return RiskMetrics()

            # FIX(4): Use provided account to avoid redundant OMS call
            account = _account or self._get_unified_account_view()
            equity = account.equity

            metrics = RiskMetrics()
            warnings: list[str] = []

            metrics.equity = equity
            metrics.cash = account.cash
            metrics.positions_value = account.positions_value
            metrics.total_pnl = equity - self._initial_equity

            # Daily P&L
            if self._daily_start_equity > 0:
                metrics.daily_pnl = equity - self._daily_start_equity
                metrics.daily_pnl_pct = (
                    metrics.daily_pnl / self._daily_start_equity * 100.0
                )

            if self._peak_equity > 0:
                metrics.current_drawdown_pct = (
                    (self._peak_equity - equity) / self._peak_equity * 100.0
                )

            if metrics.current_drawdown_pct > self._max_drawdown_pct:
                self._max_drawdown_pct = metrics.current_drawdown_pct
            metrics.max_drawdown_pct = self._max_drawdown_pct

            metrics.var_1d_95 = self._calculate_var(0.95, equity)
            metrics.var_1d_99 = self._calculate_var(0.99, equity)
            metrics.expected_shortfall = (
                self._calculate_expected_shortfall(0.95, equity)
            )

            positions = list(account.positions.values())
            metrics.position_count = sum(
                1 for p in positions if p.quantity != 0
            )

            largest = max(
                (p.market_value for p in positions), default=0.0,
            )
            metrics.largest_position_pct = (
                (largest / equity * 100.0) if equity > 0 else 0.0
            )

            metrics.long_exposure = sum(
                max(0.0, p.market_value) for p in positions
            )
            metrics.short_exposure = 0.0  # A-share: no shorting
            metrics.net_exposure = metrics.long_exposure
            metrics.gross_exposure = metrics.long_exposure
            metrics.exposure_pct = (
                (metrics.gross_exposure / equity * 100.0)
                if equity > 0 else 0.0
            )

            metrics.daily_loss_remaining_pct = (
                CONFIG.risk.max_daily_loss_pct + metrics.daily_pnl_pct
            )
            metrics.position_limit_remaining = max(
                0, CONFIG.risk.max_positions - metrics.position_count
            )

            metrics.risk_level = self._assess_risk_level(metrics)

            metrics.can_trade = True
            if metrics.daily_pnl_pct <= -CONFIG.risk.max_daily_loss_pct:
                metrics.can_trade = False
                warnings.append("Daily loss limit breached")
            if metrics.current_drawdown_pct >= CONFIG.risk.max_drawdown_pct:
                metrics.can_trade = False
                warnings.append("Max drawdown breached")

            # Kill switch / circuit breaker status
            try:
                ks = _get_kill_switch()
                metrics.kill_switch_active = not ks.can_trade
                metrics.circuit_breaker_active = getattr(
                    ks, 'circuit_breaker_active', False
                )
                if metrics.kill_switch_active:
                    metrics.can_trade = False
                    warnings.append("Kill switch active")
            except Exception as e:
                log.debug("Kill-switch metrics snapshot unavailable: %s", e)

            metrics.warnings = warnings
            metrics.timestamp = datetime.now()
            return metrics

    @staticmethod
    def _assess_risk_level(metrics: RiskMetrics) -> RiskLevel:
        """Determine overall risk level from metrics."""
        if (
            metrics.current_drawdown_pct >= 10.0
            or metrics.daily_pnl_pct <= -2.5
        ):
            return RiskLevel.CRITICAL
        if (
            metrics.current_drawdown_pct >= 5.0
            or metrics.daily_pnl_pct <= -1.5
            or metrics.largest_position_pct >= 20.0
        ):
            return RiskLevel.HIGH
        if (
            metrics.current_drawdown_pct >= 2.0
            or metrics.daily_pnl_pct <= -0.5
            or metrics.largest_position_pct >= 12.0
        ):
            return RiskLevel.MEDIUM
        return RiskLevel.LOW

    # =========================================================================
    # =========================================================================

    def _calculate_var(self, confidence: float, equity: float) -> float:
        """Historical simulation VaR (absolute currency)."""
        if equity <= 0:
            return 0.0
        if len(self._returns_history) < self.MIN_RETURNS_FOR_VAR:
            return equity * self.DEFAULT_VAR_PCT

        returns = np.array(self._returns_history, dtype=np.float64)
        var_pct = np.percentile(returns, (1.0 - confidence) * 100.0)
        return abs(float(var_pct)) * equity

    def _calculate_expected_shortfall(
        self, confidence: float, equity: float,
    ) -> float:
        """Conditional VaR (Expected Shortfall) in currency."""
        if equity <= 0:
            return 0.0
        if len(self._returns_history) < self.MIN_RETURNS_FOR_VAR:
            return equity * self.DEFAULT_ES_PCT

        returns = np.array(self._returns_history, dtype=np.float64)
        var_pct = np.percentile(returns, (1.0 - confidence) * 100.0)
        tail = returns[returns <= var_pct]
        if len(tail) == 0:
            return abs(float(var_pct)) * equity
        return abs(float(np.mean(tail))) * equity

    def _record_daily_return(self, equity: float) -> None:
        """Record daily return for VaR calculation."""
        today = date.today()
        if self._last_var_day == today:
            return

        # FIX(12): Extended lookback for long holidays
        prev_equity = None
        for i in range(1, self.DAILY_RETURN_LOOKBACK_DAYS + 1):
            d = today - timedelta(days=i)
            if d in self._equity_by_day:
                prev_equity = self._equity_by_day[d]
                break

        if prev_equity and prev_equity > 0:
            daily_return = (equity - prev_equity) / prev_equity
            # FIX(2): deque with maxlen auto-prunes
            self._returns_history.append(daily_return)

        self._equity_by_day[today] = equity
        self._last_var_day = today

    # =========================================================================
    # =========================================================================

    def _check_risk_breaches(self) -> None:
        """Check for risk limit breaches and trigger events."""
        if self._account is None:
            return

        # FIX(4): Single account snapshot; pass to get_metrics to avoid
        # redundant _get_unified_account_view call inside get_metrics
        account = self._get_unified_account_view()
        metrics = self.get_metrics(_account=account)

        breach_checks = [
            (
                'daily_loss_limit',
                metrics.daily_pnl_pct <= -CONFIG.risk.max_daily_loss_pct,
                metrics.daily_pnl_pct,
            ),
            (
                'max_drawdown',
                metrics.current_drawdown_pct >= CONFIG.risk.max_drawdown_pct,
                metrics.current_drawdown_pct,
            ),
            (
                'kill_switch_threshold',
                metrics.daily_pnl_pct <= -CONFIG.risk.kill_switch_loss_pct,
                metrics.daily_pnl_pct,
            ),
            (
                'kill_switch_drawdown',
                (
                    metrics.current_drawdown_pct
                    >= CONFIG.risk.kill_switch_drawdown_pct
                ),
                metrics.current_drawdown_pct,
            ),
            (
                'concentration_breach',
                (
                    metrics.largest_position_pct
                    > CONFIG.risk.max_position_pct * 1.1
                ),
                metrics.largest_position_pct,
            ),
        ]

        for risk_type, triggered, value in breach_checks:
            if triggered:
                self._trigger_risk_event(risk_type, value)

    def _trigger_risk_event(self, risk_type: str, value: float) -> None:
        """Trigger risk event with throttling."""
        now = datetime.now()
        last = self._last_breach.get(risk_type)
        if (
            last
            and (now - last).total_seconds()
            < self.RISK_EVENT_THROTTLE_SECONDS
        ):
            return
        self._last_breach[risk_type] = now

        EVENT_BUS.publish(RiskEvent(
            type=EventType.RISK_BREACH,
            risk_type=risk_type,
            current_value=value,
            limit_value=self._get_limit_for_type(risk_type),
            action_taken='alert_triggered',
        ))

        self._audit.log_risk_event(risk_type, {
            'value': value,
            'equity': self._account.equity if self._account else 0,
            'timestamp': now.isoformat(),
        })

        log.warning(f"Risk breach: {risk_type} = {value:.2f}")

    def _get_limit_for_type(self, risk_type: str) -> float:
        """Get the configured limit value for a risk type."""
        limits = {
            'daily_loss_limit': CONFIG.risk.max_daily_loss_pct,
            'max_drawdown': CONFIG.risk.max_drawdown_pct,
            'kill_switch_threshold': CONFIG.risk.kill_switch_loss_pct,
            'kill_switch_drawdown': CONFIG.risk.kill_switch_drawdown_pct,
            'concentration_breach': CONFIG.risk.max_position_pct,
        }
        return limits.get(risk_type, 0.0)

    # =========================================================================
    # =========================================================================

    def _check_rate_limit(self) -> bool:
        """Check order rate limits. Does NOT record the attempt."""
        self._prune_minute_window(self._orders_this_minute)

        if (
            len(self._orders_this_minute)
            >= CONFIG.risk.max_orders_per_minute
        ):
            log.warning(
                f"Rate limit: {len(self._orders_this_minute)} orders/minute "
                f"(max: {CONFIG.risk.max_orders_per_minute})"
            )
            return False

        if self._orders_submitted_today >= CONFIG.risk.max_orders_per_day:
            log.warning(
                f"Daily limit: {self._orders_submitted_today} orders today "
                f"(max: {CONFIG.risk.max_orders_per_day})"
            )
            return False

        return True

    def _check_error_rate(self) -> bool:
        """Check error rate — pause if too many errors in the last minute."""
        self._prune_minute_window(self._errors_this_minute)

        if len(self._errors_this_minute) >= self.MAX_ERRORS_PER_MINUTE:
            log.warning(
                f"High error rate: "
                f"{len(self._errors_this_minute)} errors/minute"
            )
            return False
        return True

    def _record_order_attempt(self) -> None:
        """Record an order attempt AFTER all validations pass."""
        self._orders_this_minute.append(datetime.now())
        self._orders_submitted_today += 1

    # =========================================================================
    # =========================================================================

    def _check_quote_staleness(self, symbol: str) -> tuple[bool, str]:
        """Check if quote is fresh enough for trading."""
        last_quote_time = self._quote_timestamps.get(symbol)

        if last_quote_time is None:
            q = _get_feed_quote(symbol)
            if q and getattr(q, "timestamp", None):
                last_quote_time = q.timestamp

        if last_quote_time is None:
            return True, "OK (first trade - no prior quote)"

        age = (datetime.now() - last_quote_time).total_seconds()
        if age > self._staleness_threshold_seconds:
            return False, (
                f"Quote stale: {age:.0f}s "
                f"(limit {self._staleness_threshold_seconds:.0f}s)"
            )
        return True, "OK"

    # =========================================================================
    # =========================================================================

    def check_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: int,
        price: float,
    ) -> tuple[bool, str]:
        """Comprehensive pre-trade order validation.

        Returns:
            (approved, message) — message is "OK" on success, reason
            on failure.
        """
        with self._lock:
            if price <= 0:
                return False, "Invalid price"
            if quantity <= 0:
                return False, "Invalid quantity"

            if self._account is None:
                try:
                    if not _get_oms().get_account():
                        return False, "Risk manager not initialized"
                except Exception as e:
                    log.debug("RiskManager OMS account lookup failed: %s", e)
                    return False, "Risk manager not initialized"

            # FIX(5): Kill switch failure now BLOCKS trading
            try:
                ks = _get_kill_switch()
                if not ks.can_trade:
                    return False, (
                        "Trading halted - kill switch or "
                        "circuit breaker active"
                    )
            except Exception as e:
                log.error(f"Kill switch check failed: {e}")
                return False, (
                    "Trading halted - cannot verify kill switch status"
                )

            account = self._get_unified_account_view()
            metrics = self.get_metrics(_account=account)

            if not metrics.can_trade:
                reasons = "; ".join(metrics.warnings) if metrics.warnings else (
                    f"daily loss: {metrics.daily_pnl_pct:.1f}%"
                )
                return False, f"Trading disabled - {reasons}"

            if not self._check_rate_limit():
                return False, "Order rate limit exceeded - please wait"

            if not self._check_error_rate():
                return False, (
                    "High error rate detected - trading paused for safety"
                )

            staleness_ok, staleness_msg = self._check_quote_staleness(symbol)
            if not staleness_ok:
                return False, staleness_msg

            # Side-specific validation
            if side == OrderSide.BUY:
                ok, msg = self._validate_buy_order(
                    symbol, quantity, price, metrics, account,
                )
            else:
                ok, msg = self._validate_sell_order(
                    symbol, quantity, account,
                )

            if not ok:
                return False, msg

            self._record_order_attempt()
            return True, "OK"

    def _validate_buy_order(
        self,
        symbol: str,
        quantity: int,
        price: float,
        metrics: RiskMetrics,
        account: Account,
    ) -> tuple[bool, str]:
        """Validate buy order against all risk limits."""
        lot_size = _get_lot_size(symbol)
        if quantity % lot_size != 0:
            return False, f"Quantity must be multiple of {lot_size}"

        try:
            _, _, total_cost = _estimate_order_cost(
                quantity, price, OrderSide.BUY,
            )
        except ValueError as e:
            return False, str(e)

        if total_cost > account.available:
            return False, (
                f"Insufficient funds: need ¥{total_cost:,.2f}, "
                f"have ¥{account.available:,.2f}"
            )

        equity = account.equity
        new_notional = quantity * price

        existing_value = (
            account.positions[symbol].market_value
            if symbol in account.positions else 0.0
        )
        new_position_value = existing_value + new_notional

        if equity > 0:
            position_pct = (new_position_value / equity) * 100.0
            max_pct = CONFIG.risk.max_position_pct
            if position_pct > max_pct:
                return False, (
                    f"Position too large: {position_pct:.1f}% "
                    f"(max: {max_pct}%)"
                )

        if symbol not in account.positions:
            if metrics.position_count >= CONFIG.risk.max_positions:
                return False, (
                    f"Maximum positions reached: "
                    f"{CONFIG.risk.max_positions}"
                )

        max_exposure_pct = CONFIG.risk.max_portfolio_risk_pct
        new_exposure = metrics.gross_exposure + new_notional
        if equity > 0:
            new_exposure_pct = (new_exposure / equity) * 100.0
            if new_exposure_pct > max_exposure_pct:
                return False, (
                    f"Would exceed max exposure: "
                    f"{new_exposure_pct:.1f}% (max: {max_exposure_pct}%)"
                )

        # Top-3 concentration
        conc_ok, conc_msg = self._check_top3_concentration(
            symbol, new_notional, account,
        )
        if not conc_ok:
            return False, conc_msg

        # VaR-based throttling
        if equity > 0:
            current_var_pct = (metrics.var_1d_95 / equity) * 100.0
            if current_var_pct > self.MAX_VAR_PCT_THRESHOLD:
                max_add_value = equity * (
                    self.MAX_ADD_PCT_WHEN_HIGH_VAR / 100.0
                )
                if new_notional > max_add_value:
                    return False, (
                        f"High VaR ({current_var_pct:.1f}%) - "
                        f"reduce position to max ¥{max_add_value:,.0f}"
                    )

        return True, "OK"

    def _validate_sell_order(
        self, symbol: str, quantity: int, account: Account,
    ) -> tuple[bool, str]:
        """Validate sell order."""
        if symbol not in account.positions:
            return False, f"No position in {symbol}"

        pos = account.positions[symbol]
        if quantity > pos.available_qty:
            return False, (
                f"Insufficient available shares: "
                f"have {pos.available_qty}, need {quantity}"
            )

        return True, "OK"

    def _check_top3_concentration(
        self, symbol: str, new_value: float, account: Account,
    ) -> tuple[bool, str]:
        """Check top-3 position concentration limit."""
        equity = account.equity
        if equity <= 0:
            return False, "No equity available"

        position_values = []
        symbol_found = False
        for sym, pos in account.positions.items():
            if sym == symbol:
                position_values.append(pos.market_value + new_value)
                symbol_found = True
            else:
                position_values.append(pos.market_value)

        if not symbol_found:
            position_values.append(new_value)

        position_values.sort(reverse=True)
        top3_value = sum(position_values[:3])
        top3_pct = (top3_value / equity) * 100.0

        if top3_pct > self.TOP3_CONCENTRATION_LIMIT:
            return False, (
                f"Top 3 concentration {top3_pct:.1f}% "
                f"exceeds {self.TOP3_CONCENTRATION_LIMIT}%"
            )

        return True, "OK"

    # =========================================================================
    # =========================================================================

    def calculate_position_size(
        self,
        symbol: str,
        entry_price: float,
        stop_loss: float,
        confidence: float = 1.0,
        signal_strength: float = 1.0,
    ) -> int:
        """Calculate optimal position size using risk-based sizing.

        Uses Kelly-adjusted risk allocation with multiple caps.
        """
        with self._lock:
            if self._account is None:
                return 0

            # FIX(10): Get account snapshot under lock, then compute
            account = self._get_unified_account_view()

        # Compute outside lock to avoid holding lock during calculations
        equity = account.equity
        available = account.available

        if equity <= 0 or available <= 0:
            return 0

        risk_per_share = abs(entry_price - stop_loss)
        if risk_per_share <= 0 or entry_price <= 0:
            return 0

        base_risk_pct = CONFIG.risk.risk_per_trade_pct / 100.0
        adjusted_risk = (
            base_risk_pct
            * max(0.0, min(1.0, confidence))
            * max(0.0, min(2.0, signal_strength))
        )
        kelly_adjusted = adjusted_risk * CONFIG.risk.kelly_fraction
        risk_amount = equity * kelly_adjusted

        shares = int(risk_amount / risk_per_share)

        lot_size = _get_lot_size(symbol)
        shares = (shares // lot_size) * lot_size

        max_position_value = equity * (CONFIG.risk.max_position_pct / 100.0)
        max_shares_by_position = (
            int(max_position_value / entry_price / lot_size) * lot_size
        )
        shares = min(shares, max_shares_by_position)

        max_affordable = int(
            available * self.MAX_AFFORDABLE_FACTOR
            / entry_price / lot_size
        ) * lot_size
        shares = min(shares, max_affordable)

        if shares < lot_size:
            return 0

        return shares

    def get_position_size_recommendation(
        self,
        symbol: str,
        entry_price: float,
        stop_loss: float,
        confidence: float = 1.0,
    ) -> dict:
        """Get detailed position size recommendation."""
        # FIX(11): calculate_position_size already acquires lock;
        # we get account snapshot separately to avoid double-lock
        shares = self.calculate_position_size(
            symbol=symbol,
            entry_price=entry_price,
            stop_loss=stop_loss,
            confidence=confidence,
        )

        if shares == 0:
            return {
                'shares': 0, 'value': 0, 'risk_amount': 0,
                'risk_pct': 0, 'position_pct': 0,
                'reason': 'Position size too small or no funds',
            }

        with self._lock:
            account = self._get_unified_account_view()
            equity = account.equity

        value = shares * entry_price
        risk_per_share = abs(entry_price - stop_loss)
        risk_amount = shares * risk_per_share
        risk_pct = (risk_amount / equity * 100.0) if equity > 0 else 0
        position_pct = (value / equity * 100.0) if equity > 0 else 0

        return {
            'shares': shares,
            'value': round(value, 2),
            'risk_amount': round(risk_amount, 2),
            'risk_pct': round(risk_pct, 2),
            'position_pct': round(position_pct, 2),
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'reason': 'OK',
        }

    # =========================================================================
    # =========================================================================

    def get_account(self) -> Account | None:
        """Get current account snapshot."""
        with self._lock:
            return self._account

    def get_daily_pnl(self) -> tuple[float, float]:
        """Get daily P&L in absolute and percentage."""
        with self._lock:
            if self._account is None or self._daily_start_equity <= 0:
                return 0.0, 0.0
            pnl = self._account.equity - self._daily_start_equity
            pnl_pct = (pnl / self._daily_start_equity) * 100.0
            return pnl, pnl_pct

    def get_total_pnl(self) -> tuple[float, float]:
        """Get total P&L since inception."""
        with self._lock:
            if self._account is None or self._initial_equity <= 0:
                return 0.0, 0.0
            pnl = self._account.equity - self._initial_equity
            pnl_pct = (pnl / self._initial_equity) * 100.0
            return pnl, pnl_pct

    # =========================================================================
    # =========================================================================

    def set_staleness_threshold(self, seconds: float) -> None:
        """Set quote staleness threshold."""
        with self._lock:
            self._staleness_threshold_seconds = max(1.0, seconds)

    def update_quote_timestamp(
        self, symbol: str, timestamp: datetime = None,
    ) -> None:
        """Manually update quote timestamp."""
        with self._lock:
            self._quote_timestamps[symbol] = timestamp or datetime.now()

    def reset_daily(self) -> None:
        """Manually reset daily tracking (for testing)."""
        with self._lock:
            if self._account:
                self._daily_start_equity = self._account.equity
            self._trades_today = 0
            self._orders_submitted_today = 0
            self._orders_this_minute.clear()
            self._errors_this_minute.clear()
            self._last_date = date.today()

# Module-level singleton

_risk_manager: RiskManager | None = None
_risk_manager_lock = threading.Lock()

def get_risk_manager() -> RiskManager:
    """Get global risk manager instance (thread-safe)."""
    global _risk_manager
    if _risk_manager is None:
        with _risk_manager_lock:
            if _risk_manager is None:
                _risk_manager = RiskManager()
    return _risk_manager

def reset_risk_manager() -> None:
    """Reset global risk manager (for testing).
    Properly unsubscribes the old instance before discarding.
    """
    global _risk_manager
    with _risk_manager_lock:
        if _risk_manager is not None:
            _risk_manager.shutdown()
        _risk_manager = None
