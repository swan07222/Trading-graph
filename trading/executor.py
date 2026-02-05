# trading/executor.py
"""
Execution Engine - Production Grade

FIXES:
- No syntax errors (sync loop is a proper class method)
- Correct thread lifecycle (start/stop/join)
- OMS integration (submit->broker->update OMS)
- Fill sync loop (requires broker.get_fills(); safe no-op if missing)
- Reconciliation loop separated
- Kill switch cancels orders + drains queue
"""
from __future__ import annotations

import queue
import threading
import time
from typing import Dict, Optional, Callable, Set, List

from config import CONFIG, TradingMode
from core.types import Order, OrderSide, OrderStatus, TradeSignal, Account, Fill
from trading.broker import create_broker
from trading.risk import RiskManager, get_risk_manager
from trading.kill_switch import get_kill_switch
from trading.health import get_health_monitor, ComponentType, HealthStatus
from trading.alerts import get_alert_manager, AlertPriority
from utils.logger import log


class ExecutionEngine:
    """
    Production execution engine with:
    - Kill switch integration
    - Health monitoring
    - Alerting
    - OMS persistence + reconciliation
    - Broker->OMS fill sync loop (required for live correctness)
    """

    def __init__(self, mode: TradingMode = None):
        self.mode = mode or CONFIG.trading_mode

        # Broker
        self.broker = create_broker(self.mode.value)

        # Components
        self.risk_manager: Optional[RiskManager] = None
        self._kill_switch = get_kill_switch()
        self._health_monitor = get_health_monitor()
        self._alert_manager = get_alert_manager()

        # Queue / threads
        self._queue: queue.Queue[Optional[TradeSignal]] = queue.Queue()
        self._running = False

        self._exec_thread: Optional[threading.Thread] = None
        self._sync_thread: Optional[threading.Thread] = None
        self._recon_thread: Optional[threading.Thread] = None

        self._sync_running = False
        self._seen_fill_ids: Set[str] = set()

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

        # Initialize OMS + risk
        from trading.oms import get_oms
        _ = get_oms()

        account = self.broker.get_account()
        self.risk_manager = get_risk_manager()
        self.risk_manager.initialize(account)

        # Start monitors
        self._health_monitor.start()
        self._alert_manager.start()

        # Start loops
        self._running = True

        self._exec_thread = threading.Thread(target=self._run_loop, name="exec_loop", daemon=True)
        self._exec_thread.start()

        self._sync_running = True
        self._sync_thread = threading.Thread(target=self._sync_loop, name="sync_loop", daemon=True)
        self._sync_thread.start()

        self._recon_thread = threading.Thread(target=self._reconciliation_loop, name="recon_loop", daemon=True)
        self._recon_thread.start()

        # Report healthy
        self._health_monitor.report_component_health(ComponentType.BROKER, HealthStatus.HEALTHY)

        log.info(f"Execution engine started ({self.mode.value})")

        self._alert_manager.system_alert(
            "Trading System Started",
            f"Mode: {self.mode.value}, Equity: ¥{account.equity:,.2f}",
            priority=AlertPriority.MEDIUM,
        )

        return True

    def stop(self):
        if not self._running:
            return

        self._running = False
        self._sync_running = False

        # Wake execution thread
        try:
            self._queue.put_nowait(None)
        except Exception:
            pass

        # Join threads
        for t in (self._exec_thread, self._sync_thread, self._recon_thread):
            if t and t.is_alive():
                t.join(timeout=5)

        # Disconnect + stop components
        try:
            self.broker.disconnect()
        except Exception:
            pass

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
        if not self._running:
            log.warning("Execution engine not running")
            return False

        # Kill switch
        if not self._kill_switch.can_trade:
            msg = "Trading halted - kill switch/circuit breaker active"
            log.warning(msg)
            self._reject_callback(signal, msg)
            return False

        if not self.risk_manager:
            msg = "Risk manager not initialized"
            self._reject_callback(signal, msg)
            return False

        # FIXED: Properly handle price extraction from Quote or float
        price = float(signal.price or 0.0)
        if price <= 0:
            quote_result = self.broker.get_quote(signal.symbol)
            # get_quote returns float directly in our interface
            if quote_result is not None:
                price = float(quote_result)
            else:
                price = 0.0

        if price <= 0:
            msg = f"Cannot get price for {signal.symbol}"
            self._reject_callback(signal, msg)
            return False

        passed, msg = self.risk_manager.check_order(signal.symbol, signal.side, signal.quantity, price)
        if not passed:
            log.warning(f"Risk check failed: {msg}")
            self._alert_manager.risk_alert(
                "Order Rejected (Risk)",
                f"{signal.symbol}: {msg}",
                {'symbol': signal.symbol, 'reason': msg},
            )
            self._reject_callback(signal, msg)
            return False

        self._queue.put(signal)
        log.info(f"Signal queued: {signal.side.value} {signal.quantity} {signal.symbol}")
        return True

    def submit_from_prediction(self, pred) -> bool:
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
    # Internal loops
    # ---------------------------------------------------------------------
    def _run_loop(self):
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

            # periodic risk update
            if self.risk_manager and self.broker.is_connected:
                try:
                    account = self.broker.get_account()
                    self.risk_manager.update(account)
                except Exception:
                    pass

            time.sleep(0.05)

    def _sync_loop(self):
        """
        Sync broker fills -> OMS.

        NOTE: For live trading, your broker MUST implement broker.get_fills()
        returning core.types.Fill records with fill.order_id matching OMS order.id.
        """
        from trading.oms import get_oms
        oms = get_oms()

        while self._running and self._sync_running:
            try:
                time.sleep(2.0)

                if not self.broker.is_connected:
                    continue

                get_fills = getattr(self.broker, "get_fills", None)
                if not callable(get_fills):
                    continue

                fills: List[Fill] = []
                try:
                    fills = get_fills()
                except Exception:
                    fills = []

                for fill in fills:
                    fid = getattr(fill, "id", "") or ""
                    if not fid:
                        continue
                    if fid in self._seen_fill_ids:
                        continue
                    self._seen_fill_ids.add(fid)

                    order = oms.get_order(fill.order_id)
                    if order:
                        oms.process_fill(order, fill)

            except Exception as e:
                log.error(f"Sync loop error: {e}")

    def _reconciliation_loop(self):
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

                if (
                    abs(discrepancies.get('cash_diff', 0.0)) > 1.0
                    or discrepancies.get('position_diffs')
                    or discrepancies.get('missing_positions')
                    or discrepancies.get('extra_positions')
                ):
                    self._alert_manager.risk_alert(
                        "Reconciliation Discrepancy",
                        f"Cash diff: ¥{discrepancies.get('cash_diff', 0):.2f}",
                        discrepancies,
                    )

            except Exception as e:
                log.error(f"Reconciliation error: {e}")

    # ---------------------------------------------------------------------
    # Execution + OMS integration
    # ---------------------------------------------------------------------
    def _execute(self, signal: TradeSignal):
        from trading.oms import get_oms
        oms = get_oms()

        order: Optional[Order] = None

        try:
            if not self._kill_switch.can_trade:
                log.warning("Trading halted during execution")
                return

            # Build OMS Order
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

            # Submit to OMS first (reserve cash/shares)
            order = oms.submit_order(order)

            # Submit to broker
            result = self.broker.submit_order(order)

            # Broker reject -> update OMS to release reservations
            if result.status == OrderStatus.REJECTED:
                oms.update_order_status(order.id, OrderStatus.REJECTED, message=result.message or "Rejected")
                self._alert_manager.risk_alert(
                    "Order Rejected (Broker)",
                    f"{order.symbol}: {result.message}",
                    {'order_id': order.id, 'symbol': order.symbol, 'reason': result.message},
                )
                if self.on_reject:
                    self.on_reject(order, result.message or "Rejected")
                return

            # Submitted/accepted/partial -> OMS status update, fills will arrive via sync loop
            if result.status in (OrderStatus.SUBMITTED, OrderStatus.ACCEPTED, OrderStatus.PARTIAL):
                try:
                    oms.update_order_status(order.id, result.status, message=result.message or "")
                except Exception:
                    pass
                log.info(f"Order sent to broker: {order.id} status={result.status.value}")
                return

            # Immediate fill (simulator)
            if result.status == OrderStatus.FILLED:
                fill = Fill(
                    order_id=order.id,
                    symbol=order.symbol,
                    side=order.side,
                    quantity=int(getattr(result, "filled_qty", order.quantity) or order.quantity),
                    price=float(getattr(result, "filled_price", order.price) or order.price),
                    commission=float(getattr(result, "commission", 0.0) or 0.0),
                    stamp_tax=float(getattr(result, "stamp_tax", 0.0) or 0.0),
                )
                oms.process_fill(order, fill)

                log.info(f"Filled: {order.side.value.upper()} {fill.quantity} {order.symbol} @ ¥{fill.price:.2f}")

                if self.on_fill:
                    self.on_fill(order)

                self._alert_manager.trading_alert(
                    "Order Filled",
                    f"{order.side.value.upper()} {fill.quantity} {order.symbol} @ ¥{fill.price:.2f}",
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
    # Kill switch
    # ---------------------------------------------------------------------
    def _on_kill_switch(self, reason: str):
        log.critical(f"Kill switch activated: {reason}")

        # Cancel active orders
        try:
            for o in self.broker.get_orders(active_only=True):
                try:
                    self.broker.cancel_order(o.id)
                    log.info(f"Cancelled order: {o.id}")
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
            o = Order(symbol=signal.symbol, side=signal.side, quantity=signal.quantity, price=signal.price)
            o.status = OrderStatus.REJECTED
            o.message = reason
            self.on_reject(o, reason)
        except Exception:
            pass

    # ---------------------------------------------------------------------
    # Convenience getters
    # ---------------------------------------------------------------------
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