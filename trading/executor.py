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

log = get_logger(__name__)


class ExecutionEngine:
    """
    Production execution engine with full broker synchronization.
    
    CRITICAL for Live Trading Correctness:
    - Fill sync loop polls broker.get_fills() and updates OMS
    - Order status sync loop polls broker order statuses
    - Proper order ID mapping (OMS ID != broker entrust number)
    - Reconciliation between OMS and broker state
    """

    def __init__(self, mode: TradingMode = None):
        self.mode = mode or CONFIG.trading_mode

        # Broker
        self.broker: BrokerInterface = create_broker(self.mode.value)

        # Components
        self.risk_manager: Optional[RiskManager] = None
        self._kill_switch = get_kill_switch()
        self._health_monitor = get_health_monitor()
        self._alert_manager = get_alert_manager()

        # Queue / threads
        self._queue: queue.Queue[Optional[TradeSignal]] = queue.Queue()
        self._running = False

        self._exec_thread: Optional[threading.Thread] = None
        self._fill_sync_thread: Optional[threading.Thread] = None
        self._status_sync_thread: Optional[threading.Thread] = None
        self._recon_thread: Optional[threading.Thread] = None

        # Track processed fills to avoid duplicates
        self._processed_fill_ids: Set[str] = set()

        # Callbacks
        self.on_fill: Optional[Callable[[Order], None]] = None
        self.on_reject: Optional[Callable[[Order, str], None]] = None

        # Register kill switch callback
        self._kill_switch.on_activate(self._on_kill_switch)

    # ---------------------------------------------------------------------
    # Lifecycle
    # ---------------------------------------------------------------------
    def start(self) -> bool:
        if self._running:
            return True

        # Connect broker
        if not self.broker.connect():
            log.error("Broker connection failed")
            self._health_monitor.report_component_health(
                ComponentType.BROKER,
                HealthStatus.UNHEALTHY,
                error="Connection failed",
            )
            return False

        # Initialize OMS
        from trading.oms import get_oms
        oms = get_oms()

        # Initialize risk manager with broker account
        account = self.broker.get_account()
        self.risk_manager = get_risk_manager()
        self.risk_manager.initialize(account)

        # Start monitors
        self._health_monitor.start()
        self._alert_manager.start()

        # Start all threads
        self._running = True

        self._exec_thread = threading.Thread(
            target=self._execution_loop, 
            name="exec_loop", 
            daemon=True
        )
        self._exec_thread.start()

        self._fill_sync_thread = threading.Thread(
            target=self._fill_sync_loop, 
            name="fill_sync_loop", 
            daemon=True
        )
        self._fill_sync_thread.start()

        self._status_sync_thread = threading.Thread(
            target=self._order_status_sync_loop, 
            name="status_sync_loop", 
            daemon=True
        )
        self._status_sync_thread.start()

        self._recon_thread = threading.Thread(
            target=self._reconciliation_loop, 
            name="recon_loop", 
            daemon=True
        )
        self._recon_thread.start()

        # Report healthy
        self._health_monitor.report_component_health(ComponentType.BROKER, HealthStatus.HEALTHY)

        log.info(f"Execution engine started ({self.mode.value})")

        self._alert_manager.system_alert(
            "Trading System Started",
            f"Mode: {self.mode.value}, Equity: {account.equity:,.2f}",
            priority=AlertPriority.MEDIUM,
        )

        return True

    def stop(self):
        if not self._running:
            return

        self._running = False

        # Wake execution thread
        try:
            self._queue.put_nowait(None)
        except Exception:
            pass

        # Join all threads
        threads = [
            self._exec_thread,
            self._fill_sync_thread,
            self._status_sync_thread,
            self._recon_thread
        ]
        for t in threads:
            if t and t.is_alive():
                t.join(timeout=5)

        # Disconnect broker
        try:
            self.broker.disconnect()
        except Exception as e:
            log.warning(f"Broker disconnect error: {e}")

        # Stop components
        try:
            self._health_monitor.stop()
        except Exception:
            pass

        try:
            self._alert_manager.stop()
        except Exception:
            pass

        log.info("Execution engine stopped")

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------
    def submit(self, signal: TradeSignal) -> bool:
        """Submit a trading signal for execution"""
        if not self._running:
            log.warning("Execution engine not running")
            return False

        # Kill switch check
        if not self._kill_switch.can_trade:
            msg = "Trading halted - kill switch/circuit breaker active"
            log.warning(msg)
            self._reject_callback(signal, msg)
            return False

        if not self.risk_manager:
            msg = "Risk manager not initialized"
            self._reject_callback(signal, msg)
            return False

        # Get price for validation
        price = float(signal.price or 0.0)
        if price <= 0:
            quote_price = self.broker.get_quote(signal.symbol)
            if quote_price is not None:
                price = float(quote_price)
            else:
                price = 0.0

        if price <= 0:
            msg = f"Cannot get price for {signal.symbol}"
            self._reject_callback(signal, msg)
            return False

        # Risk check
        passed, msg = self.risk_manager.check_order(
            signal.symbol, signal.side, signal.quantity, price
        )
        if not passed:
            log.warning(f"Risk check failed: {msg}")
            self._alert_manager.risk_alert(
                "Order Rejected (Risk)",
                f"{signal.symbol}: {msg}",
                {'symbol': signal.symbol, 'reason': msg},
            )
            self._reject_callback(signal, msg)
            return False

        # Queue for execution
        self._queue.put(signal)
        log.info(f"Signal queued: {signal.side.value} {signal.quantity} {signal.symbol}")
        return True

    def submit_from_prediction(self, pred) -> bool:
        """Submit order from AI prediction"""
        from models.predictor import Signal as UiSignal

        if pred.signal == UiSignal.HOLD:
            return False
        if pred.position.shares == 0:
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

    # ---------------------------------------------------------------------
    # Execution Loop
    # ---------------------------------------------------------------------
    def _execution_loop(self):
        """Main execution loop - processes signals from queue"""
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
                self._alert_manager.system_alert(
                    "Execution Loop Error",
                    str(e),
                    priority=AlertPriority.HIGH,
                )

            # Periodic risk update
            if self.risk_manager and self.broker.is_connected:
                try:
                    account = self.broker.get_account()
                    self.risk_manager.update(account)
                except Exception:
                    pass

            time.sleep(0.05)

    def _execute(self, signal: TradeSignal):
        """Execute a single signal"""
        from trading.oms import get_oms
        oms = get_oms()

        order: Optional[Order] = None

        try:
            if not self._kill_switch.can_trade:
                log.warning("Trading halted during execution")
                return

            # Build Order object
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

            # Submit to OMS first (reserves cash/shares)
            order = oms.submit_order(order)

            # Submit to broker
            result = self.broker.submit_order(order)

            # Handle broker response
            if result.status == OrderStatus.REJECTED:
                # Release OMS reservations
                oms.update_order_status(order.id, OrderStatus.REJECTED, message=result.message or "Rejected")
                
                self._alert_manager.risk_alert(
                    "Order Rejected (Broker)",
                    f"{order.symbol}: {result.message}",
                    {'order_id': order.id, 'symbol': order.symbol, 'reason': result.message},
                )
                
                if self.on_reject:
                    self.on_reject(order, result.message or "Rejected")
                return

            if result.status in (OrderStatus.SUBMITTED, OrderStatus.ACCEPTED):
                # Update OMS status
                oms.update_order_status(order.id, result.status, message=result.message or "")
                log.info(f"Order sent to broker: {order.id} -> broker_id={result.broker_id}")
                return

            if result.status == OrderStatus.FILLED:
                # Immediate fill (simulator)
                fill = Fill(
                    order_id=order.id,  # Our order ID
                    symbol=order.symbol,
                    side=order.side,
                    quantity=int(result.filled_qty or order.quantity),
                    price=float(result.filled_price or order.price),
                    commission=float(result.commission or 0.0),
                    stamp_tax=0.0,
                )
                
                oms.process_fill(order, fill)
                
                log.info(f"Filled: {order.side.value.upper()} {fill.quantity} {order.symbol} @ {fill.price:.2f}")

                if self.on_fill:
                    self.on_fill(order)

                self._alert_manager.trading_alert(
                    "Order Filled",
                    f"{order.side.value.upper()} {fill.quantity} {order.symbol} @ {fill.price:.2f}",
                    {'order_id': order.id, 'symbol': order.symbol, 'price': fill.price, 'qty': fill.quantity},
                )
                return

            log.warning(f"Unknown broker order status: {result.status}")

        except Exception as e:
            log.error(f"Execution error: {e}")
            if order is not None:
                try:
                    oms.update_order_status(order.id, OrderStatus.REJECTED, message=str(e))
                except Exception:
                    pass

            self._alert_manager.system_alert(
                "Execution Failed",
                f"{signal.symbol}: {str(e)}",
                priority=AlertPriority.HIGH,
            )

    # ---------------------------------------------------------------------
    # Fill Sync Loop - CRITICAL for Live Trading
    # ---------------------------------------------------------------------
    def _fill_sync_loop(self):
        """
        Poll broker for fills and update OMS.
        
        CRITICAL: This is what makes live trading work correctly.
        The broker's get_fills() returns Fill objects with our order_id
        (mapped from broker's entrust number), which we then process
        through OMS.
        """
        from trading.oms import get_oms
        oms = get_oms()

        while self._running:
            try:
                time.sleep(2.0)  # Poll every 2 seconds

                if not self.broker.is_connected:
                    continue

                # Get new fills from broker
                fills = self.broker.get_fills()

                for fill in fills:
                    # Skip already processed
                    fill_id = getattr(fill, 'id', '') or ''
                    if fill_id and fill_id in self._processed_fill_ids:
                        continue
                    
                    if fill_id:
                        self._processed_fill_ids.add(fill_id)

                    # Get order from OMS
                    order = oms.get_order(fill.order_id)
                    if not order:
                        log.warning(f"Fill for unknown order: {fill.order_id}")
                        continue

                    # Process fill through OMS
                    oms.process_fill(order, fill)
                    
                    log.info(f"Fill synced: {fill.id} for order {fill.order_id}")

                    # Callback
                    if self.on_fill:
                        try:
                            self.on_fill(order)
                        except Exception as e:
                            log.warning(f"Fill callback error: {e}")

            except Exception as e:
                log.error(f"Fill sync loop error: {e}")

    # ---------------------------------------------------------------------
    # Order Status Sync Loop
    # ---------------------------------------------------------------------
    def _order_status_sync_loop(self):
        """
        Poll broker for order status updates.
        
        For orders that haven't been filled yet, check if their
        status has changed (partial fill, cancelled, etc.)
        """
        from trading.oms import get_oms
        oms = get_oms()

        while self._running:
            try:
                time.sleep(5.0)  # Poll every 5 seconds

                if not self.broker.is_connected:
                    continue

                # Get active orders from OMS
                active_orders = oms.get_active_orders()

                for order in active_orders:
                    # Sync with broker
                    synced = self.broker.sync_order(order)
                    
                    if synced.status != order.status:
                        # Status changed - update OMS
                        oms.update_order_status(
                            order.id, 
                            synced.status, 
                            message=synced.message
                        )
                        log.info(f"Order status synced: {order.id} -> {synced.status.value}")

            except Exception as e:
                log.error(f"Order status sync error: {e}")

    # ---------------------------------------------------------------------
    # Reconciliation Loop
    # ---------------------------------------------------------------------
    def _reconciliation_loop(self):
        """Periodic reconciliation between OMS and broker"""
        from trading.oms import get_oms
        oms = get_oms()

        while self._running:
            try:
                time.sleep(300)  # Every 5 minutes

                if not self.broker.is_connected:
                    continue

                broker_account = self.broker.get_account()
                broker_positions = self.broker.get_positions()

                discrepancies = oms.reconcile(broker_positions, broker_account.cash)

                if (
                    abs(discrepancies.get('cash_diff', 0.0)) > 1.0
                    or discrepancies.get('position_diffs')
                    or discrepancies.get('missing_positions')
                    or discrepancies.get('extra_positions')
                ):
                    self._alert_manager.risk_alert(
                        "Reconciliation Discrepancy",
                        f"Cash diff: {discrepancies.get('cash_diff', 0):.2f}",
                        discrepancies,
                    )

            except Exception as e:
                log.error(f"Reconciliation error: {e}")

    # ---------------------------------------------------------------------
    # Kill Switch Handler
    # ---------------------------------------------------------------------
    def _on_kill_switch(self, reason: str):
        """Handle kill switch activation"""
        log.critical(f"Kill switch activated: {reason}")

        # Cancel active orders
        try:
            for order in self.broker.get_orders(active_only=True):
                try:
                    self.broker.cancel_order(order.id)
                    log.info(f"Cancelled order: {order.id}")
                except Exception:
                    pass
        except Exception as e:
            log.error(f"Failed to cancel orders: {e}")

        # Drain queue
        try:
            while not self._queue.empty():
                self._queue.get_nowait()
        except Exception:
            pass

        self._alert_manager.critical_alert(
            "KILL SWITCH ACTIVATED",
            f"All trading halted: {reason}",
            {'reason': reason},
        )

    # ---------------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------------
    def _reject_callback(self, signal: TradeSignal, reason: str):
        if not self.on_reject:
            return
        try:
            order = Order(
                symbol=signal.symbol, 
                side=signal.side, 
                quantity=signal.quantity, 
                price=signal.price
            )
            order.status = OrderStatus.REJECTED
            order.message = reason
            self.on_reject(order, reason)
        except Exception:
            pass

    def get_account(self) -> Account:
        return self.broker.get_account()

    def get_positions(self):
        return self.broker.get_positions()

    def get_orders(self):
        return self.broker.get_orders()

    def reconcile(self) -> Dict:
        if hasattr(self.broker, "reconcile"):
            return self.broker.reconcile()
        return {}