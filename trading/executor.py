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
        self.risk_manager: Optional[RiskManager] = None
        
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
        
        self.on_fill: Optional[Callable[[Order, Fill], None]] = None
        self.on_reject: Optional[Callable[[Order, str], None]] = None

        self._kill_switch.on_activate(self._on_kill_switch)
        self._processed_fill_ids: Set[str] = self._load_processed_fills()

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
        self.risk_manager = get_risk_manager()
        
        # Get account from OMS
        account = oms.get_account()
        
        # Initialize and update risk manager
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
        return True

    def _load_processed_fills(self) -> Set[str]:
        """Load already-processed fill IDs from database"""
        try:
            from trading.oms import get_oms
            oms = get_oms()
            fills = oms.get_fills()  # Gets all fills from DB
            return {f.id for f in fills if f.id}
        except Exception as e:
            log.warning(f"Could not load processed fills: {e}")
            return set()

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

    def submit(self, signal: TradeSignal) -> bool:
        """Submit a trading signal for execution"""
        if not self._running:
            log.warning("Execution engine not running")
            return False

        if not self._kill_switch.can_trade:
            self._reject_signal(signal, "Trading halted - kill switch active")
            return False

        if not self.risk_manager:
            self._reject_signal(signal, "Risk manager not initialized")
            return False

        price = float(signal.price or 0.0)
        if price <= 0:
            quote_price = self.broker.get_quote(signal.symbol)
            price = float(quote_price) if quote_price else 0.0

        if price <= 0:
            self._reject_signal(signal, f"Cannot get price for {signal.symbol}")
            return False

        passed, msg = self.risk_manager.check_order(signal.symbol, signal.side, signal.quantity, price)
        if not passed:
            log.warning(f"Risk check failed: {msg}")
            self._alert_manager.risk_alert("Order Rejected (Risk)", f"{signal.symbol}: {msg}")
            self._reject_signal(signal, msg)
            return False

        self._queue.put(signal)
        log.info(f"Signal queued: {signal.side.value} {signal.quantity} {signal.symbol}")
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
        """Execute a single signal - NEVER fabricate fills"""
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

            # OMS validates and reserves resources, sets status to SUBMITTED
            order = oms.submit_order(order)

            # Submit to broker
            result = self.broker.submit_order(order)

            inc_counter("orders_submitted_total", labels={"side": order.side.value, "symbol": order.symbol})

            # Persist broker_id and update status through OMS
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

            # For simulator: immediately pull and process fills
            # This ensures single fill processing path
            if result.status in (OrderStatus.PARTIAL, OrderStatus.FILLED):
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

    def _process_pending_fills(self):
        """Process any pending fills from broker - single processing path"""
        from trading.oms import get_oms
        oms = get_oms()

        try:  # ← ADD THIS TRY
            with self._fills_lock:
                fills = self.broker.get_fills()
                for fill in fills:
                    fill_id = fill.id or ""
                    if fill_id and fill_id in self._processed_fill_ids:
                        continue
                    if fill_id:
                        self._processed_fill_ids.add(fill_id)

                    # Primary lookup by our order_id
                    order = oms.get_order(fill.order_id)
                    
                    # FIXED: Only try broker_id lookup if primary fails AND fill has broker-style ID
                    if not order and fill.order_id:
                        # Check if this looks like a broker ID (not our format)
                        if not fill.order_id.startswith("ORD_"):
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
                    observe("fill_latency_seconds", (datetime.now() - fill.timestamp).total_seconds() if fill.timestamp else 0)

        except Exception as e:  # ← NOW THIS MATCHES THE TRY
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
        """Poll broker for order status updates - STATUS ONLY, not filled_qty"""
        from trading.oms import get_oms
        oms = get_oms()

        while self._running:
            try:
                time.sleep(3.0)
                if not self.broker.is_connected:
                    continue

                active_orders = oms.get_active_orders()
                for order in active_orders:
                    # Get status from broker (don't use filled_qty from broker)
                    broker_status = self.broker.get_order_status(order.id)
                    
                    if broker_status is None:
                        continue
                    
                    # Only update if status actually changed
                    if broker_status == OrderStatus.FILLED:
                        oms.update_order_status(
                            order.id,
                            order.status,  # keep unchanged
                            message="Status sync: broker reports FILLED; waiting fills"
                        )
                        continue

                    if broker_status != order.status:
                        oms.update_order_status(
                            order.id,
                            broker_status,
                            message=f"Status sync: {broker_status.value}"
                        )

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
        return self.broker.get_account()

    def get_positions(self):
        return self.broker.get_positions()

    def get_orders(self):
        return self.broker.get_orders()