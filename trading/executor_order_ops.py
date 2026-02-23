from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

from config import CONFIG
from core.types import Fill, Order, OrderSide, OrderStatus, OrderType, TradeSignal
from trading.alerts import AlertPriority
from trading.executor_error_policy import SOFT_FAIL_EXCEPTIONS
from trading.executor_policy_ops import (
    _allow_order_type_emulation,
    _broker_supports_order_type,
    _is_advanced_order_type,
    _should_escalate_runtime_exception,
)
from trading.health import HealthStatus
from utils.logger import get_logger
from utils.metrics import inc_counter, observe, set_gauge
from utils.policy import get_trade_policy_engine
from utils.security import get_access_control, get_audit_log

log = get_logger(__name__)
_SOFT_FAIL_EXCEPTIONS = SOFT_FAIL_EXCEPTIONS


def _resolve_access_control():
    try:
        from trading import executor as _executor_mod

        factory = getattr(_executor_mod, "get_access_control", None)
        if callable(factory):
            return factory()
    except _SOFT_FAIL_EXCEPTIONS as e:
        log.debug("Executor-scoped access control resolver unavailable: %s", e)
    return get_access_control()


def _resolve_audit_log():
    try:
        from trading import executor as _executor_mod

        factory = getattr(_executor_mod, "get_audit_log", None)
        if callable(factory):
            return factory()
    except _SOFT_FAIL_EXCEPTIONS as e:
        log.debug("Executor-scoped audit log resolver unavailable: %s", e)
    return get_audit_log()


def submit(self, signal: TradeSignal) -> bool:
    """Submit a trading signal for execution with strict quote freshness."""
    if not self._running:
        log.warning("Execution engine not running")
        return False
    hinted_price = float(getattr(signal, "price", 0.0) or 0.0)
    requested_order_type = self._normalize_requested_order_type(
        getattr(signal, "order_type", "limit")
    )
    requested_enum = OrderType(requested_order_type)
    market_style_types = {"market", "ioc", "fok", "stop", "trail_market"}
    limit_style_types = {"limit", "stop_limit", "trail_limit"}

    # Explicit policy gate: advanced order emulation is configurable and
    # defaults to disabled in LIVE mode.
    if _is_advanced_order_type(self, requested_enum) and not _broker_supports_order_type(
        self, requested_enum
    ):
        emulate_ok, emulate_msg = _allow_order_type_emulation(self, requested_enum)
        if not emulate_ok:
            self._reject_signal(signal, emulate_msg)
            try:
                _resolve_audit_log().log_risk_event(
                    "order_type_emulation_blocked",
                    {
                        "symbol": signal.symbol,
                        "signal_id": signal.id,
                        "requested_order_type": requested_enum.value,
                        "mode": str(getattr(self.mode, "value", self.mode)),
                    },
                )
            except _SOFT_FAIL_EXCEPTIONS as e:
                log.warning(
                    "Audit log write failed for emulation block on %s: %s",
                    signal.symbol,
                    e,
                )
            return False

    # Operational guardrails: block/allow trading based on health status.
    try:
        sec_cfg = getattr(CONFIG, "security", None)
        block_unhealthy = bool(
            getattr(sec_cfg, "block_trading_when_unhealthy", True)
        )
        block_degraded = bool(
            getattr(sec_cfg, "block_trading_when_degraded", False)
        )
        h = self._health_monitor.get_health()
        if block_unhealthy and h.status in (HealthStatus.UNHEALTHY, HealthStatus.CRITICAL):
            self._reject_signal(signal, f"System unhealthy: {h.status.value}")
            return False
        if block_degraded and h.status == HealthStatus.DEGRADED:
            self._reject_signal(signal, "System degraded: trading paused by policy")
            return False
    except _SOFT_FAIL_EXCEPTIONS as e:
        log.warning("Health policy guard failed for signal=%s: %s", signal.symbol, e)

    # Institutional controls: permission gating + optional dual-control.
    try:
        sec_cfg = getattr(CONFIG, "security", None)
        access = _resolve_access_control()
        audit = _resolve_audit_log()
        mode_is_live = str(getattr(self.mode, "value", "")).lower() == "live"

        require_perm = bool(
            getattr(sec_cfg, "require_live_trade_permission", True)
        )
        strict_live = bool(
            getattr(sec_cfg, "strict_live_governance", False)
        )
        min_approvals = int(getattr(sec_cfg, "min_live_approvals", 2))

        if mode_is_live and require_perm:
            if not access.check("trade_live"):
                self._reject_signal(
                    signal, "Access denied: trade_live permission required"
                )
                audit.log_risk_event(
                    "access_denied_live_trade",
                    {"symbol": signal.symbol, "signal_id": signal.id},
                )
                return False
        elif not mode_is_live:
            if not access.check("trade_paper"):
                self._reject_signal(
                    signal, "Access denied: trade_paper permission required"
                )
                audit.log_risk_event(
                    "access_denied_paper_trade",
                    {"symbol": signal.symbol, "signal_id": signal.id},
                )
                return False

        if mode_is_live and strict_live and not signal.auto_generated:
            approvals = int(getattr(signal, "approvals_count", 0) or 0)
            approver_ids = getattr(signal, "approver_ids", None)
            if not isinstance(approver_ids, list):
                approver_ids = getattr(signal, "approved_by", None)
            if isinstance(approver_ids, list):
                approvals = max(
                    approvals,
                    len({str(x).strip().lower() for x in approver_ids if str(x).strip()}),
                )
            required = max(1, min_approvals)
            if approvals < required:
                self._reject_signal(
                    signal, f"Insufficient live approvals ({approvals}/{required})"
                )
                audit.log_risk_event(
                    "live_trade_blocked_dual_control",
                    {
                        "symbol": signal.symbol,
                        "signal_id": signal.id,
                        "approvals": approvals,
                        "required": required,
                    },
                )
                return False

        if mode_is_live:
            decision = get_trade_policy_engine().evaluate_live_trade(signal)
            if not decision.allowed:
                self._reject_signal(signal, f"Policy blocked: {decision.reason}")
                audit.log_risk_event(
                    "live_trade_blocked_policy",
                    {
                        "symbol": signal.symbol,
                        "signal_id": signal.id,
                        "reason": decision.reason,
                        "policy_version": decision.policy_version,
                        "metadata": decision.metadata or {},
                    },
                )
                return False
    except _SOFT_FAIL_EXCEPTIONS as e:
        if _should_escalate_runtime_exception(self, e):
            raise
        log.error("Security governance check failed for %s: %s", signal.symbol, e)
        self._reject_signal(signal, "Security governance check unavailable")
        return False

    try:
        if not CONFIG.is_market_open():
            self._reject_signal(signal, "Market closed")
            return False
    except _SOFT_FAIL_EXCEPTIONS as e:
        if _should_escalate_runtime_exception(self, e):
            raise
        log.error("Market-open check failed for signal=%s: %s", signal.symbol, e)
        self._reject_signal(signal, "Market-state check unavailable")
        return False

    if not self._kill_switch.can_trade:
        self._reject_signal(signal, "Trading halted - kill switch active")
        return False

    if not self.risk_manager:
        self._reject_signal(signal, "Risk manager not initialized")
        return False

    try:
        from data.fetcher import DataFetcher

        signal.symbol = DataFetcher.clean_code(signal.symbol)
    except _SOFT_FAIL_EXCEPTIONS as e:
        log.debug("Signal symbol normalization failed for %s: %s", signal.symbol, e)

    max_age = 15.0
    if hasattr(CONFIG, "risk") and hasattr(CONFIG.risk, "quote_staleness_seconds"):
        max_age = float(CONFIG.risk.quote_staleness_seconds)
    block_delayed = bool(
        getattr(
            getattr(CONFIG, "auto_trade", None),
            "block_on_stale_realtime",
            True,
        )
    )

    try:
        ok, msg, fresh_px = self._require_fresh_quote(
            signal.symbol,
            max_age_seconds=max_age,
            block_delayed=block_delayed,
        )
    except TypeError:
        ok, msg, fresh_px = self._require_fresh_quote(
            signal.symbol, max_age_seconds=max_age
        )
    if not ok:
        self._reject_signal(signal, msg)
        return False

    trigger_price = float(getattr(signal, "trigger_price", 0.0) or 0.0)
    if requested_order_type in {"stop", "stop_limit"} and trigger_price <= 0:
        self._reject_signal(signal, "Missing trigger price for stop order")
        return False
    if requested_order_type in limit_style_types and hinted_price <= 0:
        self._reject_signal(
            signal,
            f"{requested_order_type} order requires a positive limit price",
        )
        return False

    execution_ref_price = hinted_price
    if requested_order_type in market_style_types or execution_ref_price <= 0:
        execution_ref_price = float(fresh_px)
    if execution_ref_price <= 0:
        self._reject_signal(signal, "No valid execution reference price")
        return False

    # Best-execution sanity check: avoid large drift from hinted signal price.
    try:
        max_bps = float(getattr(CONFIG.risk, "max_quote_deviation_bps", 80.0))
        if (
            requested_order_type in market_style_types
            and hinted_price > 0
            and max_bps > 0
        ):
            dev_bps = abs((float(fresh_px) / hinted_price - 1.0) * 10000.0)
            if dev_bps > max_bps:
                self._reject_signal(
                    signal,
                    f"Best-exec guard: quote deviation {dev_bps:.1f}bps > {max_bps:.1f}bps",
                )
                try:
                    _resolve_audit_log().log_risk_event(
                        "best_execution_guard_reject",
                        {
                            "symbol": signal.symbol,
                            "hinted_price": hinted_price,
                            "fresh_price": float(fresh_px),
                            "deviation_bps": float(dev_bps),
                            "limit_bps": float(max_bps),
                        },
                    )
                except _SOFT_FAIL_EXCEPTIONS as e:
                    log.warning(
                        "Audit log write failed for best-exec rejection on %s: %s",
                        signal.symbol,
                        e,
                    )
                return False
    except _SOFT_FAIL_EXCEPTIONS as e:
        if _should_escalate_runtime_exception(self, e):
            raise
        mode_is_live = str(getattr(self.mode, "value", self.mode)).lower() == "live"
        if mode_is_live:
            self._reject_signal(signal, "Best-exec guard unavailable")
            return False
        log.warning(
            "Best-exec guard evaluation failed for %s: %s",
            signal.symbol,
            e,
        )

    signal._arrival_price = float(fresh_px)
    signal.price = (
        float(execution_ref_price)
        if requested_order_type in market_style_types
        else float(hinted_price)
    )

    guard_ok, guard_msg = self._check_submission_guardrails(signal)
    if not guard_ok:
        self._reject_signal(signal, guard_msg)
        return False

    # CN limit up/down sanity
    try:
        from core.constants import get_price_limit
        from data.fetcher import get_fetcher

        q = get_fetcher().get_realtime(signal.symbol)
        prev_close = float(getattr(q, "close", 0.0) or 0.0)
        if prev_close > 0:
            lim = float(
                get_price_limit(signal.symbol, getattr(q, "name", None))
            )
            up = prev_close * (1.0 + lim)
            dn = prev_close * (1.0 - lim)

            if signal.side == OrderSide.BUY and signal.price >= up * 0.999:
                self._reject_signal(
                    signal, f"At/near limit-up ({lim * 100:.0f}%)"
                )
                return False
            if signal.side == OrderSide.SELL and signal.price <= dn * 1.001:
                self._reject_signal(
                    signal, f"At/near limit-down ({lim * 100:.0f}%)"
                )
                return False
    except _SOFT_FAIL_EXCEPTIONS as e:
        if _should_escalate_runtime_exception(self, e):
            raise
        mode_is_live = str(getattr(self.mode, "value", self.mode)).lower() == "live"
        if mode_is_live:
            self._reject_signal(signal, "Price-limit sanity check unavailable")
            return False
        log.warning("CN price-limit sanity check failed for %s: %s", signal.symbol, e)

    passed, rmsg = self.risk_manager.check_order(
        signal.symbol, signal.side, int(signal.quantity), float(signal.price)
    )
    if not passed:
        log.warning(f"Risk check failed: {rmsg}")
        self._alert_manager.risk_alert(
            "Order Rejected (Risk)", f"{signal.symbol}: {rmsg}"
        )
        self._reject_signal(signal, rmsg)
        return False

    self._queue.put(signal)
    set_gauge("runtime_queue_depth", float(self._queue.qsize()))
    log.info(
        f"Signal queued: {signal.side.value} {signal.quantity} "
        f"{signal.symbol} @ {signal.price:.2f}"
        f"{' [AUTO]' if signal.auto_generated else ''}"
    )
    return True


def _execute(self, signal: TradeSignal):
    """Execute a single signal - NEVER fabricate fills."""
    from trading.oms import get_oms

    oms = get_oms()
    order: Order | None = None

    try:
        if not self._kill_switch.can_trade:
            log.warning("Trading halted during execution")
            return

        order = Order(
            symbol=signal.symbol,
            name=signal.name,
            side=signal.side,
            quantity=int(signal.quantity),
            price=float(signal.price),
            order_type=OrderType.LIMIT,
            stop_loss=(
                float(signal.stop_loss) if signal.stop_loss else 0.0
            ),
            take_profit=(
                float(signal.take_profit) if signal.take_profit else 0.0
            ),
            signal_id=signal.id,
            strategy=signal.strategy or "",
        )

        requested_order_type = str(
            self._normalize_requested_order_type(
                getattr(signal, "order_type", "limit")
            )
        )
        requested_enum = OrderType(requested_order_type)

        broker_order_type = requested_enum
        emulation_reason = ""
        if _is_advanced_order_type(
            self, requested_enum
        ) and not _broker_supports_order_type(self, requested_enum):
            emulate_ok, emulate_msg = _allow_order_type_emulation(
                self, requested_enum
            )
            if not emulate_ok:
                self._reject_signal(signal, emulate_msg)
                return
            # Broker abstraction is market/limit centric; advanced types are
            # represented in tags and handled by runtime guardrails/simulator.
            if requested_enum in {
                OrderType.STOP,
                OrderType.IOC,
                OrderType.FOK,
                OrderType.TRAIL_MARKET,
            }:
                broker_order_type = OrderType.MARKET
            else:
                broker_order_type = OrderType.LIMIT
            emulation_reason = (
                f"{requested_enum.value}_as_{broker_order_type.value}"
            )

        trigger_px = float(getattr(signal, "trigger_price", 0.0) or 0.0)
        if trigger_px <= 0 and requested_enum in {OrderType.STOP, OrderType.STOP_LIMIT}:
            trigger_px = float(signal.price or 0.0)

        if broker_order_type == OrderType.LIMIT and order.price <= 0:
            fallback_px = float(signal.price or 0.0)
            if fallback_px <= 0 and trigger_px > 0:
                fallback_px = trigger_px
            order.price = fallback_px

        if trigger_px > 0:
            order.stop_price = float(trigger_px)

        order.order_type = broker_order_type

        # Tag auto-trade orders for audit
        if signal.auto_generated:
            order.tags["auto_trade"] = True
            order.tags["auto_trade_action_id"] = signal.auto_trade_action_id
            order.strategy = order.strategy or "auto_trade"
        arrival_price = float(getattr(signal, "_arrival_price", 0.0) or 0.0)
        if arrival_price <= 0:
            arrival_price = float(signal.price or 0.0)
        order.tags["arrival_price"] = arrival_price
        order.tags["requested_order_type"] = requested_enum.value
        order.tags["order_type_emulated"] = bool(
            broker_order_type != requested_enum
        )
        if emulation_reason:
            order.tags["order_type_emulation_reason"] = emulation_reason
        if trigger_px > 0:
            order.tags["trigger_price"] = float(trigger_px)
        oco_group = str(getattr(signal, "oco_group", "") or "").strip()
        if oco_group:
            order.tags["oco_group"] = oco_group

        tif = str(getattr(signal, "time_in_force", "day") or "day").strip().lower()
        tif = tif.replace("-", "_")
        if requested_enum in {OrderType.IOC, OrderType.FOK}:
            tif = requested_enum.value
        if tif not in {"day", "gtc", "ioc", "fok"}:
            tif = "day"
        order.tags["time_in_force"] = tif
        if tif in {"ioc", "fok"}:
            order.tags["strict_time_in_force"] = True

        order.tags["bracket_enabled"] = bool(
            getattr(signal, "bracket", False)
            or order.stop_loss > 0
            or order.take_profit > 0
        )
        trailing_pct = float(getattr(signal, "trailing_stop_pct", 0.0) or 0.0)
        if requested_enum in {OrderType.TRAIL_MARKET, OrderType.TRAIL_LIMIT} and trailing_pct <= 0:
            trailing_pct = float(
                getattr(getattr(CONFIG, "auto_trade", None), "trailing_stop_pct", 0.0) or 0.0
            )
        if trailing_pct > 0:
            order.tags["trailing_stop_pct"] = trailing_pct
        trail_limit_offset_pct = float(
            getattr(signal, "trail_limit_offset_pct", 0.0) or 0.0
        )
        if trail_limit_offset_pct > 0:
            order.tags["trail_limit_offset_pct"] = trail_limit_offset_pct

        order = oms.submit_order(order)

        result = self._submit_with_retry(order, attempts=3)

        inc_counter(
            "orders_submitted_total",
            labels={"side": order.side.value, "symbol": order.symbol},
        )

        # Persist broker_id/status in OMS
        oms.update_order_status(
            order.id,
            result.status,
            message=result.message or "",
            broker_id=result.broker_id or "",
        )

        if result.status == OrderStatus.REJECTED:
            self._alert_manager.risk_alert(
                "Order Rejected (Broker)",
                f"{order.symbol}: {result.message}",
            )
            if self.on_reject:
                self.on_reject(order, result.message or "Rejected")
            return

        # Pull fills immediately (sim) / early sync
        self._process_pending_fills()

        log.info(
            f"Order sent: {order.id} -> broker_id={result.broker_id}, "
            f"status={result.status.value}"
            f"{' [AUTO]' if signal.auto_generated else ''}"
        )

    except _SOFT_FAIL_EXCEPTIONS as e:
        if _should_escalate_runtime_exception(self, e):
            log.exception("Execution fatal exception for %s", signal.symbol)
            raise
        log.error(f"Execution error: {e}")
        if order:
            try:
                oms.update_order_status(
                    order.id, OrderStatus.REJECTED, message=str(e)
                )
            except _SOFT_FAIL_EXCEPTIONS as status_err:
                log.warning(
                    "Failed to mark order rejected in OMS (order_id=%s): %s",
                    getattr(order, "id", ""),
                    status_err,
                )
        self._alert_manager.system_alert(
            "Execution Failed",
            f"{signal.symbol}: {e}",
            AlertPriority.HIGH,
        )


def _startup_sync(self):
    """Run once after broker.connect()."""
    from trading.oms import get_oms

    oms = get_oms()

    self._rebuild_broker_mappings(oms)

    self._process_pending_fills()

    for order in oms.get_active_orders():
        try:
            synced = self.broker.sync_order(order)
            if synced and synced.status and synced.status != order.status:
                oms.update_order_status(
                    order.id,
                    synced.status,
                    message="Startup sync",
                    broker_id=synced.broker_id or order.broker_id or "",
                )
        except _SOFT_FAIL_EXCEPTIONS as e:
            log.debug("Startup order sync failed for order=%s: %s", order.id, e)
            continue


def _process_pending_fills(self):
    """Process pending fills with safe watermark overlap."""
    from trading.oms import get_oms

    oms = get_oms()

    with self._fills_lock:
        try:
            query_start = datetime.now()

            fills = self.broker.get_fills(since=self._last_fill_sync)

            newest_ts = None
            for f in fills:
                ts = getattr(f, "timestamp", None)
                if ts:
                    newest_ts = (
                        ts if newest_ts is None else max(newest_ts, ts)
                    )

            if newest_ts:
                self._last_fill_sync = newest_ts - timedelta(
                    seconds=self._FILL_WATERMARK_OVERLAP_SECONDS
                )
            else:
                self._last_fill_sync = query_start

            for fill in fills:
                fill_id = str(getattr(fill, "id", "") or "").strip()
                if not fill_id:
                    fill_id = (
                        f"{fill.order_id}|{fill.symbol}|"
                        f"{fill.side.value}|{fill.quantity}|"
                        f"{fill.price}|{getattr(fill, 'timestamp', None)}"
                    )

                if fill_id in self._processed_fill_ids:
                    continue
                self._processed_fill_ids.add(fill_id)

                order = oms.get_order(fill.order_id)

                # Fallback: broker may have put broker_id into fill.order_id
                if order is None:
                    try:
                        order = oms.get_order_by_broker_id(fill.order_id)
                        if order:
                            fill.order_id = order.id
                    except _SOFT_FAIL_EXCEPTIONS as e:
                        log.debug(
                            "Fallback order lookup by broker_id failed for %s: %s",
                            fill.order_id,
                            e,
                        )
                        order = None

                if not order:
                    log.warning(
                        f"Fill for unknown order: {fill.order_id}"
                    )
                    continue

                oms.process_fill(order, fill)
                self._record_execution_quality(order, fill)
                self._maybe_register_synthetic_exit(order, fill)
                self._cancel_oco_siblings(oms, order, fill)
                log.info(
                    f"Fill processed: {fill_id} for order {order.id}"
                )

                if self.on_fill:
                    try:
                        self.on_fill(order, fill)
                    except _SOFT_FAIL_EXCEPTIONS as e:
                        log.warning(f"Fill callback error: {e}")

                # Update auto-trader state if this was an auto-trade
                if (
                    self.auto_trader
                    and order.tags.get("auto_trade")
                ):
                    self._update_auto_trade_fill(order, fill)

                inc_counter(
                    "fills_processed_total",
                    labels={"side": fill.side.value},
                )
                observe(
                    "fill_latency_seconds",
                    (
                        (datetime.now() - fill.timestamp).total_seconds()
                        if fill.timestamp
                        else 0.0
                    ),
                )

        except _SOFT_FAIL_EXCEPTIONS as e:
            if _should_escalate_runtime_exception(self, e):
                log.exception("Fill processing fatal exception")
                raise
            log.error(f"Fill processing error: {e}")

        self._prune_processed_fills_unlocked(max_size=50000)


def _cancel_oco_siblings(self, oms, filled_order: Order, fill: Fill) -> None:
    tags = dict(getattr(filled_order, "tags", {}) or {})
    oco_group = str(tags.get("oco_group", "") or "").strip()
    if not oco_group or int(getattr(fill, "quantity", 0) or 0) <= 0:
        return

    try:
        siblings = oms.get_orders(filled_order.symbol)
    except _SOFT_FAIL_EXCEPTIONS as e:
        log.debug(f"OCO sibling scan failed for {filled_order.symbol}: {e}")
        return

    active_statuses = {
        OrderStatus.PENDING,
        OrderStatus.SUBMITTED,
        OrderStatus.ACCEPTED,
        OrderStatus.PARTIAL,
    }
    cancelled = 0
    for sibling in siblings:
        if sibling.id == filled_order.id:
            continue
        if sibling.status not in active_statuses:
            continue
        s_tags = dict(getattr(sibling, "tags", {}) or {})
        sibling_group = str(s_tags.get("oco_group", "") or "").strip()
        if sibling_group != oco_group:
            continue

        cancel_ok = False
        broker_ids = [
            str(getattr(sibling, "broker_id", "") or "").strip(),
            str(getattr(sibling, "id", "") or "").strip(),
        ]
        for oid in broker_ids:
            if not oid:
                continue
            try:
                if bool(self.broker.cancel_order(oid)):
                    cancel_ok = True
                    break
            except _SOFT_FAIL_EXCEPTIONS as e:
                log.debug("OCO sibling cancel attempt failed for %s: %s", oid, e)
                continue

        if not cancel_ok and sibling.status == OrderStatus.PENDING:
            cancel_ok = True

        if not cancel_ok:
            log.warning(
                "OCO sibling cancel failed: %s group=%s source_fill=%s",
                sibling.id,
                oco_group,
                filled_order.id,
            )
            continue

        try:
            oms.update_order_status(
                sibling.id,
                OrderStatus.CANCELLED,
                message=f"OCO cancelled after fill {filled_order.id}",
            )
            cancelled += 1
        except _SOFT_FAIL_EXCEPTIONS as e:
            log.warning(f"OCO sibling OMS cancel update failed: {sibling.id}: {e}")

    if cancelled > 0:
        log.info(
            "OCO siblings cancelled: group=%s symbol=%s count=%s",
            oco_group,
            filled_order.symbol,
            cancelled,
        )


def _record_execution_quality(self, order: Order, fill: Fill) -> None:
    try:
        tags = dict(getattr(order, "tags", {}) or {})
        arrival = float(tags.get("arrival_price", 0.0) or 0.0)
        if arrival <= 0:
            arrival = float(getattr(order, "price", 0.0) or 0.0)
        if arrival <= 0 or float(fill.price) <= 0:
            return

        if fill.side == OrderSide.BUY:
            slip_bps = (float(fill.price) / arrival - 1.0) * 10000.0
        else:
            slip_bps = (arrival / float(fill.price) - 1.0) * 10000.0

        reason = str(tags.get("exit_reason", "entry") or "entry")
        with self._exec_quality_lock:
            self._exec_quality["fills"] = int(
                self._exec_quality.get("fills", 0)
            ) + 1
            self._exec_quality["slippage_bps_sum"] = float(
                self._exec_quality.get("slippage_bps_sum", 0.0)
            ) + float(slip_bps)
            self._exec_quality["slippage_bps_abs_sum"] = float(
                self._exec_quality.get("slippage_bps_abs_sum", 0.0)
            ) + abs(float(slip_bps))
            by_reason = self._exec_quality.setdefault("by_reason", {})
            by_reason[str(reason)] = int(by_reason.get(str(reason), 0)) + 1
            self._exec_quality["last_update"] = datetime.now().isoformat()

        observe(
            "execution_slippage_bps",
            float(slip_bps),
            labels={"side": fill.side.value},
        )
    except _SOFT_FAIL_EXCEPTIONS as e:
        log.debug(f"Execution quality accounting failed: {e}")


def _maybe_register_synthetic_exit(self, order: Order, fill: Fill) -> None:
    # Synthetic bracket/OCO exits for filled entry orders.
    if order.side != OrderSide.BUY:
        return
    tags = dict(getattr(order, "tags", {}) or {})
    if bool(tags.get("synthetic_exit", False)):
        return

    stop_loss = float(getattr(order, "stop_loss", 0.0) or 0.0)
    take_profit = float(getattr(order, "take_profit", 0.0) or 0.0)
    trailing_pct = float(tags.get("trailing_stop_pct", 0.0) or 0.0)
    if stop_loss <= 0 and take_profit <= 0 and trailing_pct <= 0:
        return

    changed = False
    with self._synthetic_exit_lock:
        plan = self._synthetic_exits.get(order.id)
        if plan is None:
            plan = {
                "plan_id": order.id,
                "source_order_id": order.id,
                "symbol": order.symbol,
                "side": "long",
                "open_qty": 0,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "trailing_stop_pct": trailing_pct,
                "highest_price": float(fill.price),
                "armed_at": datetime.now().isoformat(),
            }
            self._synthetic_exits[order.id] = plan
            changed = True
        prev_qty = int(plan.get("open_qty", 0) or 0)
        prev_high = float(plan.get("highest_price", 0.0) or 0.0)
        plan["open_qty"] = prev_qty + int(fill.quantity)
        plan["highest_price"] = max(
            float(plan.get("highest_price", 0.0) or 0.0),
            float(fill.price),
        )
        if plan["open_qty"] != prev_qty or plan["highest_price"] != prev_high:
            changed = True
        if trailing_pct > 0:
            dyn_stop = float(plan["highest_price"]) * (1.0 - trailing_pct / 100.0)
            cur_stop = float(plan.get("stop_loss", 0.0) or 0.0)
            new_stop = max(cur_stop, dyn_stop) if cur_stop > 0 else dyn_stop
            if float(new_stop) != float(cur_stop):
                changed = True
            plan["stop_loss"] = new_stop

    if changed:
        self._persist_synthetic_exits(force=False)


def _evaluate_synthetic_exits(self) -> None:
    with self._synthetic_exit_lock:
        plans = list(self._synthetic_exits.values())

    for plan in plans:
        persist_needed = False
        symbol = str(plan.get("symbol", "") or "").strip()
        qty = int(plan.get("open_qty", 0) or 0)
        if not symbol or qty <= 0:
            continue

        px, _, _, _ = self._coerce_quote_snapshot(
            self._get_quote_snapshot(symbol)
        )
        if px <= 0:
            continue

        trailing_pct = float(plan.get("trailing_stop_pct", 0.0) or 0.0)
        if trailing_pct > 0:
            with self._synthetic_exit_lock:
                cur = self._synthetic_exits.get(str(plan.get("plan_id", "")))
                if cur is not None:
                    old_high = float(cur.get("highest_price", 0.0) or 0.0)
                    old_stop = float(cur.get("stop_loss", 0.0) or 0.0)
                    cur["highest_price"] = max(
                        float(cur.get("highest_price", 0.0) or 0.0),
                        float(px),
                    )
                    dyn_stop = float(cur["highest_price"]) * (
                        1.0 - trailing_pct / 100.0
                    )
                    cur_stop = float(cur.get("stop_loss", 0.0) or 0.0)
                    cur["stop_loss"] = max(cur_stop, dyn_stop) if cur_stop > 0 else dyn_stop
                    if (
                        float(cur.get("highest_price", 0.0) or 0.0) != old_high
                        or float(cur.get("stop_loss", 0.0) or 0.0) != old_stop
                    ):
                        persist_needed = True
                    plan = dict(cur)

        take_profit = float(plan.get("take_profit", 0.0) or 0.0)
        stop_loss = float(plan.get("stop_loss", 0.0) or 0.0)
        reason = ""
        if take_profit > 0 and px >= take_profit:
            reason = "take_profit"
        elif stop_loss > 0 and px <= stop_loss:
            reason = "stop_loss"
        if not reason:
            if persist_needed:
                self._persist_synthetic_exits(force=False)
            continue

        if self._submit_synthetic_exit(plan, trigger_price=float(px), reason=reason):
            with self._synthetic_exit_lock:
                removed = self._synthetic_exits.pop(str(plan.get("plan_id", "")), None)
                if removed is not None:
                    persist_needed = True
        if persist_needed:
            self._persist_synthetic_exits(force=True)


def _submit_synthetic_exit(
    self,
    plan: dict[str, Any],
    trigger_price: float,
    reason: str,
) -> bool:
    from trading.oms import get_oms

    symbol = str(plan.get("symbol", "") or "").strip()
    qty = int(plan.get("open_qty", 0) or 0)
    if not symbol or qty <= 0:
        return False

    order = Order(
        symbol=symbol,
        side=OrderSide.SELL,
        quantity=qty,
        order_type=OrderType.MARKET,
        price=0.0,
        strategy="synthetic_bracket_exit",
        parent_id=str(plan.get("source_order_id", "") or ""),
    )
    order.tags["synthetic_exit"] = True
    order.tags["exit_reason"] = str(reason)
    order.tags["exit_parent_order_id"] = str(plan.get("source_order_id", "") or "")
    order.tags["arrival_price"] = float(trigger_price or 0.0)

    try:
        oms = get_oms()
        order = oms.submit_order(order)
        result = self._submit_with_retry(order, attempts=2)
        oms.update_order_status(
            order.id,
            result.status,
            message=result.message or "",
            broker_id=result.broker_id or "",
        )
        if result.status == OrderStatus.REJECTED:
            self._alert_manager.risk_alert(
                "Synthetic exit rejected",
                f"{symbol}: {result.message or 'rejected'}",
                {"reason": reason, "qty": qty},
            )
            return False
        log.warning(
            "Synthetic %s exit submitted: %s qty=%s trigger=%.2f",
            reason,
            symbol,
            qty,
            float(trigger_price or 0.0),
        )
        return True
    except _SOFT_FAIL_EXCEPTIONS as e:
        log.error(f"Synthetic exit submit failed: {symbol} {reason}: {e}")
        return False


def _status_sync_loop(self):
    """
    Poll broker for order status updates.
    Includes stuck order watchdog.
    """
    from trading.oms import get_oms

    oms = get_oms()
    first_seen: dict[str, datetime] = {}

    stuck_seconds = 60
    if hasattr(CONFIG, "risk") and hasattr(CONFIG.risk, "order_stuck_seconds"):
        stuck_seconds = int(CONFIG.risk.order_stuck_seconds)

    while self._running:
        self._heartbeat("status_sync")
        try:
            if self._wait_or_stop(3.0):
                break
            if not self.broker.is_connected:
                continue

            active_orders = oms.get_active_orders()
            now = datetime.now()

            for order in active_orders:
                first_seen.setdefault(order.id, now)
                age = (now - first_seen[order.id]).total_seconds()

                broker_status = None
                try:
                    broker_status = self.broker.get_order_status(order.id)
                except _SOFT_FAIL_EXCEPTIONS as e:
                    log.debug("Broker order status fetch failed for %s: %s", order.id, e)
                    broker_status = None

                # If broker_status missing, try broker.sync_order
                if broker_status is None:
                    try:
                        synced = self.broker.sync_order(order)
                        broker_status = getattr(synced, "status", None)
                        if (
                            synced
                            and getattr(synced, "broker_id", None)
                            and not order.broker_id
                        ):
                            oms.update_order_status(
                                order.id,
                                order.status,
                                broker_id=synced.broker_id,
                                message="Recovered broker_id",
                            )
                    except _SOFT_FAIL_EXCEPTIONS as e:
                        log.debug("Broker sync_order fallback failed for %s: %s", order.id, e)
                        broker_status = None

                # If broker says FILLED: process fills immediately
                if broker_status == OrderStatus.FILLED:
                    self._process_pending_fills()
                    refreshed = oms.get_order(order.id)
                    if (
                        refreshed
                        and refreshed.status != OrderStatus.FILLED
                        and age > 30
                    ):
                        self._alert_manager.risk_alert(
                            "Missing Fills After Broker FILLED",
                            (
                                f"{order.symbol}: broker FILLED but "
                                f"OMS {refreshed.status.value}"
                            ),
                            details={
                                "order_id": order.id,
                                "broker_id": order.broker_id,
                            },
                        )
                    first_seen.pop(order.id, None)
                    continue

                if broker_status and broker_status != order.status:
                    oms.update_order_status(
                        order.id,
                        broker_status,
                        message=f"Status sync: {broker_status.value}",
                    )
                    if broker_status in (
                        OrderStatus.CANCELLED,
                        OrderStatus.REJECTED,
                        OrderStatus.FILLED,
                        OrderStatus.EXPIRED,
                    ):
                        first_seen.pop(order.id, None)
                    continue

                if age >= stuck_seconds and order.status in (
                    OrderStatus.SUBMITTED,
                    OrderStatus.ACCEPTED,
                    OrderStatus.PARTIAL,
                ):
                    self._alert_manager.risk_alert(
                        "Order Stuck Watchdog",
                        (
                            f"{order.symbol}: {order.status.value} "
                            f"for {int(age)}s"
                        ),
                        details={
                            "order_id": order.id,
                            "broker_id": order.broker_id,
                            "status": order.status.value,
                        },
                    )
                    # FIX: Only auto-cancel if explicitly enabled
                    if self.AUTO_CANCEL_STUCK_ORDERS:
                        try:
                            self.broker.cancel_order(order.id)
                        except _SOFT_FAIL_EXCEPTIONS as cancel_err:
                            log.warning(
                                "Auto-cancel of stuck order failed (order_id=%s): %s",
                                order.id,
                                cancel_err,
                            )

        except _SOFT_FAIL_EXCEPTIONS as e:
            if _should_escalate_runtime_exception(self, e):
                log.exception("Status sync fatal exception")
                raise
            log.error(f"Status sync error: {e}")

