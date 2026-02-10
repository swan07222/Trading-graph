# trading/executor.py
"""
Execution Engine - Production Grade with Full Fill Sync

CRITICAL for Live Trading:
- Proper fill sync loop with broker
- Order status polling
- OMS integration with correct order IDs
- Reconciliation
"""
from __future__ import annotations

import queue
import threading
import time
from datetime import datetime
from typing import Dict, Optional, Callable, Set, List

from config import CONFIG, TradingMode
from core.types import Order, OrderSide, OrderStatus, TradeSignal, Account, Fill
from trading.broker import BrokerInterface, create_broker
from trading.risk import RiskManager, get_risk_manager
from trading.kill_switch import get_kill_switch
from trading.health import get_health_monitor, ComponentType, HealthStatus
from trading.alerts import get_alert_manager, AlertPriority
from utils.logger import get_logger
from utils.metrics import inc_counter, set_gauge, observe

log = get_logger(__name__)


class ExecutionEngine:
    """
    Production execution engine with correct broker synchronization.
    
    DESIGN PRINCIPLES:
    1. Fills are ONLY processed from broker.get_fills() - never fabricated
    2. OMS is the single source of truth for order state
    3. Broker ID mapping is persisted through OMS for crash recovery
    4. Status sync captures previous state before mutation
    """

    def __init__(self, mode: TradingMode = None):
        self.mode = mode or CONFIG.trading_mode
        self.broker: BrokerInterface = create_broker(self.mode.value)
        from trading.risk import get_risk_manager
        self.risk_manager = get_risk_manager()

        self._kill_switch = get_kill_switch()
        self._health_monitor = get_health_monitor()
        self._alert_manager = get_alert_manager()
        self._fills_lock = threading.RLock()

        self._queue: queue.Queue[Optional[TradeSignal]] = queue.Queue()
        self._running = False

        self._exec_thread: Optional[threading.Thread] = None
        self._fill_sync_thread: Optional[threading.Thread] = None
        self._status_sync_thread: Optional[threading.Thread] = None
        self._recon_thread: Optional[threading.Thread] = None

        self._processed_fill_ids: Set[str] = set()

        # ---- NEW: fill polling watermark (reduces repeated broker scans) ----
        self._last_fill_sync: Optional[datetime] = None

        self.on_fill: Optional[Callable[[Order, Fill], None]] = None
        self.on_reject: Optional[Callable[[Order, str], None]] = None

        self._kill_switch.on_activate(self._on_kill_switch)
        self._processed_fill_ids = self._load_processed_fills()

    def start(self) -> bool:
        if self._running:
            return True

        if not self.broker.connect():
            log.error("Broker connection failed")
            self._health_monitor.report_component_health(
                ComponentType.BROKER, HealthStatus.UNHEALTHY, error="Connection failed"
            )
            return False

        from trading.oms import get_oms
        from trading.risk import get_risk_manager  # ADDED: Import here
        
        oms = get_oms()

        # Rebuild broker ID mappings from persisted orders (crash recovery)
        self._rebuild_broker_mappings(oms)

        # FIXED: Initialize risk manager BEFORE using it
        
        # Get account from OMS
        account = oms.get_account()
        
        # Initialize and update risk manager
        if self.risk_manager:
            self.risk_manager.initialize(account)
            self.risk_manager.update(account)

        self._health_monitor.start()
        self._alert_manager.start()

        self._running = True

        self._exec_thread = threading.Thread(target=self._execution_loop, name="exec", daemon=True)
        self._exec_thread.start()

        self._fill_sync_thread = threading.Thread(target=self._fill_sync_loop, name="fill_sync", daemon=True)
        self._fill_sync_thread.start()

        self._status_sync_thread = threading.Thread(target=self._status_sync_loop, name="status_sync", daemon=True)
        self._status_sync_thread.start()

        self._recon_thread = threading.Thread(target=self._reconciliation_loop, name="recon", daemon=True)
        self._recon_thread.start()

        self._health_monitor.attach_broker(self.broker)

        self._health_monitor.report_component_health(ComponentType.BROKER, HealthStatus.HEALTHY)
        log.info(f"Execution engine started ({self.mode.value})")

        self._alert_manager.system_alert(
            "Trading System Started",
            f"Mode: {self.mode.value}, Equity: {account.equity:,.2f}",
            priority=AlertPriority.MEDIUM,
        )
        self._startup_sync()
        self._reconnect_thread = threading.Thread(target=self._broker_reconnect_loop, name="broker_reconnect", daemon=True)
        self._reconnect_thread.start()
        return True

    def _load_processed_fills(self) -> Set[str]:
        """Load already-processed fill IDs from OMS DB (source of truth)."""
        try:
            from trading.oms import get_oms
            oms = get_oms()

            fills = oms.get_fills()  # <-- FIX: fills must come from OMS database
            out = set()
            for f in fills:
                fid = getattr(f, "id", None)
                if fid:
                    out.add(str(fid))
            return out
        except Exception as e:
            log.warning(f"Could not load processed fills: {e}")
            return set()

    def _resolve_price(self, symbol: str, hinted_price: float = 0.0) -> float:
        """
        Resolve one authoritative price for this submission, fast + safe:
        1) hinted_price (if >0)
        2) FeedManager cached quote (DO NOT auto-init feeds here)
        3) broker.get_quote()
        4) fetcher.get_realtime()
        """
        try:
            px = float(hinted_price or 0.0)
            if px > 0:
                return px
        except Exception:
            pass

        # 2) Feed cache without blocking init
        try:
            from data.feeds import get_feed_manager
            fm = get_feed_manager(auto_init=False)  # important: don't block execution path
            q = fm.get_quote(symbol)
            if q and getattr(q, "price", 0) and float(q.price) > 0:
                return float(q.price)
        except Exception:
            pass

        # 3) Broker quote
        try:
            px = self.broker.get_quote(symbol)
            if px and float(px) > 0:
                return float(px)
        except Exception:
            pass

        # 4) Fetcher realtime
        try:
            from data.fetcher import get_fetcher
            q = get_fetcher().get_realtime(symbol)
            if q and getattr(q, "price", 0) and float(q.price) > 0:
                return float(q.price)
        except Exception:
            pass

        return 0.0

    def _rebuild_broker_mappings(self, oms):
        """Rebuild broker ID mappings from persisted orders after restart"""
        try:
            active_orders = oms.get_active_orders()
            for order in active_orders:
                if order.broker_id:
                    self.broker.register_order_mapping(order.id, order.broker_id)
                    log.debug(f"Recovered mapping: {order.id} -> {order.broker_id}")
            log.info(f"Recovered {len(active_orders)} order mappings from DB")
        except Exception as e:
            log.warning(f"Failed to rebuild broker mappings: {e}")

    def stop(self):
        if not self._running:
            return

        self._running = False

        try:
            self._queue.put_nowait(None)
        except Exception:
            pass

        for t in [self._exec_thread, self._fill_sync_thread, self._status_sync_thread, self._recon_thread]:
            if t and t.is_alive():
                t.join(timeout=5)

        try:
            self.broker.disconnect()
        except Exception as e:
            log.warning(f"Broker disconnect error: {e}")

        try:
            self._health_monitor.stop()
            self._alert_manager.stop()
        except Exception:
            pass

        log.info("Execution engine stopped")

    def _get_quote_snapshot(self, symbol: str) -> Tuple[float, Optional[datetime], str]:
        """
        Returns (price, timestamp, source).
        Tries:
        1) FeedManager cached quote (no auto-init)
        2) broker.get_quote (price only)
        3) DataFetcher realtime (has timestamp)
        """
        # 1) feed cache
        try:
            from data.feeds import get_feed_manager
            fm = get_feed_manager(auto_init=False)
            q = fm.get_quote(symbol)
            if q and float(getattr(q, "price", 0.0) or 0.0) > 0:
                ts = getattr(q, "timestamp", None)
                return float(q.price), ts, "feed"
        except Exception:
            pass

        # 2) broker quote (usually no timestamp)
        try:
            px = self.broker.get_quote(symbol)
            if px and float(px) > 0:
                return float(px), None, "broker"
        except Exception:
            pass

        # 3) fetcher realtime
        try:
            from data.fetcher import get_fetcher
            q = get_fetcher().get_realtime(symbol)
            if q and float(getattr(q, "price", 0.0) or 0.0) > 0:
                return float(q.price), getattr(q, "timestamp", None), f"fetcher:{getattr(q,'source','')}"
        except Exception:
            pass

        return 0.0, None, "none"


    def _require_fresh_quote(self, symbol: str, max_age_seconds: float = 15.0) -> Tuple[bool, str, float]:
        """
        Strict quote freshness gate for order submission.
        Returns (ok, message, price).
        """
        px, ts, src = self._get_quote_snapshot(symbol)
        if px <= 0:
            return False, "No valid quote", 0.0

        # if no timestamp, be conservative in LIVE mode
        if ts is None:
            if str(self.mode.value).lower() == "live":
                return False, f"No timestamped quote (source={src})", 0.0
            return True, "OK", px

        try:
            age = (datetime.now() - ts).total_seconds()
        except Exception:
            age = 0.0

        if age > float(max_age_seconds):
            return False, f"Quote stale: {age:.0f}s (source={src})", 0.0

        return True, "OK", px

    def submit(self, signal: TradeSignal) -> bool:
        """Submit a trading signal for execution with strict quote freshness."""
        if not self._running:
            log.warning("Execution engine not running")
            return False

        # Market hours guard (holiday-aware via CONFIG.is_market_open fix)
        try:
            if not CONFIG.is_market_open():
                self._reject_signal(signal, "Market closed")
                return False
        except Exception:
            pass

        if not self._kill_switch.can_trade:
            self._reject_signal(signal, "Trading halted - kill switch active")
            return False

        if not self.risk_manager:
            self._reject_signal(signal, "Risk manager not initialized")
            return False

        # Normalize CN symbol
        try:
            from data.fetcher import DataFetcher
            signal.symbol = DataFetcher.clean_code(signal.symbol)
        except Exception:
            pass

        # STRICT quote freshness for LIVE (and recommended for paper too)
        max_age = float(getattr(CONFIG.risk, "quote_staleness_seconds", 15.0)) if hasattr(CONFIG, "risk") else 15.0
        ok, msg, fresh_px = self._require_fresh_quote(signal.symbol, max_age_seconds=max_age)
        if not ok:
            self._reject_signal(signal, msg)
            return False

        # Use ONE authoritative price for entire pipeline
        signal.price = float(fresh_px)

        # CN limit up/down sanity (single realtime fetch)
        try:
            from core.constants import get_price_limit
            from data.fetcher import get_fetcher
            q = get_fetcher().get_realtime(signal.symbol)
            prev_close = float(getattr(q, "close", 0.0) or 0.0)
            if prev_close > 0:
                lim = float(get_price_limit(signal.symbol, getattr(q, "name", None)))
                up = prev_close * (1.0 + lim)
                dn = prev_close * (1.0 - lim)

                if signal.side == OrderSide.BUY and signal.price >= up * 0.999:
                    self._reject_signal(signal, f"At/near limit-up ({lim*100:.0f}%)")
                    return False
                if signal.side == OrderSide.SELL and signal.price <= dn * 1.001:
                    self._reject_signal(signal, f"At/near limit-down ({lim*100:.0f}%)")
                    return False
        except Exception:
            pass

        # Risk check (same resolved price)
        passed, rmsg = self.risk_manager.check_order(
            signal.symbol, signal.side, int(signal.quantity), float(signal.price)
        )
        if not passed:
            log.warning(f"Risk check failed: {rmsg}")
            self._alert_manager.risk_alert("Order Rejected (Risk)", f"{signal.symbol}: {rmsg}")
            self._reject_signal(signal, rmsg)
            return False

        self._queue.put(signal)
        log.info(f"Signal queued: {signal.side.value} {signal.quantity} {signal.symbol} @ {signal.price:.2f}")
        return True

    def submit_from_prediction(self, pred) -> bool:
        """Submit order from AI prediction"""
        from models.predictor import Signal as UiSignal

        if pred.signal == UiSignal.HOLD or pred.position.shares == 0:
            return False

        side = OrderSide.BUY if pred.signal in (UiSignal.STRONG_BUY, UiSignal.BUY) else OrderSide.SELL

        signal = TradeSignal(
            symbol=pred.stock_code,
            name=pred.stock_name,
            side=side,
            quantity=int(pred.position.shares),
            price=float(pred.levels.entry),
            stop_loss=float(pred.levels.stop_loss) if pred.levels.stop_loss else 0.0,
            take_profit=float(pred.levels.target_2) if pred.levels.target_2 else 0.0,
            confidence=float(pred.confidence),
            reasons=list(pred.reasons),
        )
        return self.submit(signal)

    def _execution_loop(self):
        """Main execution loop"""
        last_risk_update = 0.0  # ← MOVE THIS TO THE TOP
        
        while self._running:
            try:
                signal = self._queue.get(timeout=0.2)
                if signal is None:
                    break
                self._execute(signal)
            except queue.Empty:
                pass
            except Exception as e:
                log.error(f"Execution loop error: {e}")
                self._alert_manager.system_alert("Execution Loop Error", str(e), AlertPriority.HIGH)

            # Risk update check (moved inside main loop, removed nested while)
            now = time.time()
            if self.risk_manager and self.broker.is_connected and (now - last_risk_update) >= 1.0:
                try:
                    account = self.broker.get_account()
                    self.risk_manager.update(account)
                    set_gauge("account_equity", account.equity)
                    set_gauge("account_cash", account.cash)
                    set_gauge("positions_count", len(account.positions))
                    last_risk_update = now
                except Exception as e:
                    log.warning(f"Risk update error: {e}")

            time.sleep(0.05)

    def get_risk_metrics(self):
        if self.risk_manager:
            return self.risk_manager.get_metrics()
        return None

    def _execute(self, signal: TradeSignal):
        """Execute a single signal - NEVER fabricate fills."""
        from trading.oms import get_oms
        oms = get_oms()

        order: Optional[Order] = None

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
                stop_loss=float(signal.stop_loss) if signal.stop_loss else 0.0,
                take_profit=float(signal.take_profit) if signal.take_profit else 0.0,
                signal_id=signal.id,
            )

            # OMS reserves resources
            order = oms.submit_order(order)

            # Submit to broker with retry
            result = self._submit_with_retry(order, attempts=3)

            inc_counter("orders_submitted_total", labels={"side": order.side.value, "symbol": order.symbol})

            # Persist broker_id/status in OMS
            oms.update_order_status(
                order.id,
                result.status,
                message=result.message or "",
                broker_id=result.broker_id or ""
            )

            if result.status == OrderStatus.REJECTED:
                self._alert_manager.risk_alert(
                    "Order Rejected (Broker)",
                    f"{order.symbol}: {result.message}"
                )
                if self.on_reject:
                    self.on_reject(order, result.message or "Rejected")
                return

            # Pull fills immediately (sim) / early sync
            self._process_pending_fills()

            log.info(f"Order sent: {order.id} -> broker_id={result.broker_id}, status={result.status.value}")

        except Exception as e:
            log.error(f"Execution error: {e}")
            if order:
                try:
                    oms.update_order_status(order.id, OrderStatus.REJECTED, message=str(e))
                except Exception:
                    pass
            self._alert_manager.system_alert("Execution Failed", f"{signal.symbol}: {e}", AlertPriority.HIGH)

    
    def _startup_sync(self):
        """
        Run once after broker.connect():
        - rebuild broker mappings
        - pull fills
        - sync active order statuses
        """
        from trading.oms import get_oms
        oms = get_oms()

        # Rebuild mappings (already exists)
        self._rebuild_broker_mappings(oms)

        # Process any fills immediately
        self._process_pending_fills()

        # Sync active orders
        for order in oms.get_active_orders():
            try:
                synced = self.broker.sync_order(order)
                if synced and synced.status and synced.status != order.status:
                    oms.update_order_status(
                        order.id, synced.status,
                        message="Startup sync",
                        broker_id=synced.broker_id or order.broker_id or ""
                    )
            except Exception:
                continue

    def _process_pending_fills(self):
        """Process pending fills with safe watermark overlap (avoid missed fills)."""
        from trading.oms import get_oms
        oms = get_oms()

        with self._fills_lock:
            try:
                query_start = datetime.now()
                fills = self.broker.get_fills(since=self._last_fill_sync)

                newest_ts = None
                for f in fills:
                    if getattr(f, "timestamp", None):
                        newest_ts = f.timestamp if newest_ts is None else max(newest_ts, f.timestamp)

                # OVERLAP: step watermark slightly back to avoid missing same-timestamp fills
                if newest_ts:
                    from datetime import timedelta
                    self._last_fill_sync = newest_ts - timedelta(seconds=1)
                else:
                    self._last_fill_sync = query_start

                for fill in fills:
                    # Robust fill id: if broker doesn't provide one, synthesize composite
                    fill_id = (fill.id or "").strip()
                    if not fill_id:
                        fill_id = f"{fill.order_id}|{fill.symbol}|{fill.side.value}|{fill.quantity}|{fill.price}|{getattr(fill,'timestamp',None)}"

                    if fill_id in self._processed_fill_ids:
                        continue
                    self._processed_fill_ids.add(fill_id)

                    order = oms.get_order(fill.order_id)

                    if order is None and fill.order_id:
                        order = oms.get_order_by_broker_id(fill.order_id)
                        if order:
                            log.info(f"Recovered order {order.id} from broker_id {fill.order_id}")
                            fill.order_id = order.id

                    if not order:
                        log.warning(f"Fill for unknown order: {fill.order_id}")
                        continue

                    oms.process_fill(order, fill)
                    log.info(f"Fill processed: {fill.id} for order {order.id}")

                    if self.on_fill:
                        try:
                            self.on_fill(order, fill)
                        except Exception as e:
                            log.warning(f"Fill callback error: {e}")

                    inc_counter("fills_processed_total", labels={"side": fill.side.value})
                    observe(
                        "fill_latency_seconds",
                        (datetime.now() - fill.timestamp).total_seconds() if fill.timestamp else 0.0
                    )

            except Exception as e:
                log.error(f"Fill processing error: {e}")

    def _fill_sync_loop(self):
        """Poll broker for fills"""
        while self._running:
            try:
                time.sleep(1.0)
                if not self.broker.is_connected:
                    continue
                self._process_pending_fills()
            except Exception as e:
                log.error(f"Fill sync loop error: {e}")

    def _status_sync_loop(self):
        """
        Poll broker for order status updates.
        Adds:
        - stuck order watchdog (submitted/accepted too long)
        - broker sync fallback when get_order_status returns None
        """
        from trading.oms import get_oms
        oms = get_oms()

        first_seen: Dict[str, datetime] = {}

        # configurable threshold
        stuck_seconds = int(getattr(CONFIG.risk, "order_stuck_seconds", 60)) if hasattr(CONFIG, "risk") else 60

        while self._running:
            try:
                time.sleep(3.0)
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
                    except Exception:
                        broker_status = None

                    # If broker_status missing, try broker.sync_order
                    if broker_status is None:
                        try:
                            synced = self.broker.sync_order(order)
                            broker_status = getattr(synced, "status", None)
                            if synced and getattr(synced, "broker_id", None) and not order.broker_id:
                                oms.update_order_status(order.id, order.status, broker_id=synced.broker_id, message="Recovered broker_id")
                        except Exception:
                            broker_status = None

                    # If broker says FILLED: process fills immediately
                    if broker_status == OrderStatus.FILLED:
                        self._process_pending_fills()
                        refreshed = oms.get_order(order.id)
                        if refreshed and refreshed.status != OrderStatus.FILLED and age > 30:
                            self._alert_manager.risk_alert(
                                "Missing Fills After Broker FILLED",
                                f"{order.symbol}: broker FILLED but OMS {refreshed.status.value}",
                                details={"order_id": order.id, "broker_id": order.broker_id}
                            )
                        continue

                    # Normal status changes
                    if broker_status and broker_status != order.status:
                        oms.update_order_status(order.id, broker_status, message=f"Status sync: {broker_status.value}")
                        if broker_status in (OrderStatus.CANCELLED, OrderStatus.REJECTED, OrderStatus.FILLED, OrderStatus.EXPIRED):
                            first_seen.pop(order.id, None)
                        continue

                    # --- Watchdog: stuck too long in non-terminal state ---
                    if age >= stuck_seconds and order.status in (OrderStatus.SUBMITTED, OrderStatus.ACCEPTED, OrderStatus.PARTIAL):
                        self._alert_manager.risk_alert(
                            "Order Stuck Watchdog",
                            f"{order.symbol}: {order.status.value} for {int(age)}s",
                            details={"order_id": order.id, "broker_id": order.broker_id, "status": order.status.value}
                        )
                        # Optionally attempt cancel if too long (conservative)
                        try:
                            self.broker.cancel_order(order.id)
                        except Exception:
                            pass

            except Exception as e:
                log.error(f"Status sync error: {e}")

    def _reconciliation_loop(self):
        """Periodic reconciliation"""
        from trading.oms import get_oms
        oms = get_oms()

        while self._running:
            try:
                time.sleep(300)
                if not self.broker.is_connected:
                    continue

                broker_account = self.broker.get_account()
                broker_positions = self.broker.get_positions()
                discrepancies = oms.reconcile(broker_positions, broker_account.cash)

                if (abs(discrepancies.get('cash_diff', 0.0)) > 1.0 or
                    discrepancies.get('position_diffs') or
                    discrepancies.get('missing_positions') or
                    discrepancies.get('extra_positions')):
                    self._alert_manager.risk_alert(
                        "Reconciliation Discrepancy",
                        f"Cash diff: {discrepancies.get('cash_diff', 0):.2f}",
                        discrepancies
                    )
            except Exception as e:
                log.error(f"Reconciliation error: {e}")

    def _submit_with_retry(self, order: Order, attempts: int = 3) -> Order:
        """
        Retry broker.submit_order for transient failures.
        Does NOT retry validation failures (broker REJECTED).
        """
        delay = 0.5
        last_exc = None
        for i in range(int(attempts)):
            try:
                result = self.broker.submit_order(order)
                # If broker explicitly rejects, do not retry
                if getattr(result, "status", None) == OrderStatus.REJECTED:
                    return result
                return result
            except Exception as e:
                last_exc = e
                time.sleep(delay)
                delay = min(delay * 2.0, 5.0)

        raise last_exc if last_exc else RuntimeError("submit_order failed")

    def _broker_reconnect_loop(self):
        """
        If broker drops, attempt reconnect with exponential backoff.
        On reconnect, run startup sync again.
        """
        backoff = 1.0
        while self._running:
            try:
                time.sleep(2.0)
                if self.broker.is_connected:
                    backoff = 1.0
                    continue

                log.warning(f"Broker disconnected. Reconnecting in {backoff:.0f}s...")
                time.sleep(backoff)

                # reconnect attempt (use CONFIG.BROKER_PATH if broker needs it)
                try:
                    ok = self.broker.connect(exe_path=getattr(CONFIG, "broker_path", "") or getattr(CONFIG, "BROKER_PATH", ""))
                except TypeError:
                    ok = self.broker.connect()

                if ok:
                    log.info("Broker reconnected successfully")
                    backoff = 1.0
                    self._startup_sync()
                else:
                    backoff = min(backoff * 2.0, 60.0)

            except Exception as e:
                log.warning(f"Reconnect loop error: {e}")
                backoff = min(backoff * 2.0, 60.0)

    def _on_kill_switch(self, reason: str):
        """Handle kill switch activation"""
        log.critical(f"Kill switch activated: {reason}")

        try:
            for order in self.broker.get_orders(active_only=True):
                try:
                    self.broker.cancel_order(order.id)
                except Exception:
                    pass
        except Exception as e:
            log.error(f"Failed to cancel orders: {e}")

        try:
            while not self._queue.empty():
                self._queue.get_nowait()
        except Exception:
            pass

        self._alert_manager.critical_alert("KILL SWITCH ACTIVATED", f"All trading halted: {reason}")

    def _reject_signal(self, signal: TradeSignal, reason: str):
        """Handle signal rejection"""
        log.warning(f"Signal rejected: {signal.symbol} - {reason}")
        if self.on_reject:
            order = Order(symbol=signal.symbol, side=signal.side, quantity=signal.quantity, price=signal.price)
            order.status = OrderStatus.REJECTED
            order.message = reason
            self.on_reject(order, reason)

    def get_account(self) -> Account:
        from trading.oms import get_oms
        return get_oms().get_account()

    def get_positions(self):
        from trading.oms import get_oms
        return get_oms().get_positions()

    def get_orders(self):
        return self.broker.get_orders()