"""
Execution Engine - Production Grade
"""
import threading
import queue
import time
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass

from config import CONFIG, TradingMode
from core.types import Order, OrderSide, OrderStatus, TradeSignal, Account
from trading.broker import BrokerInterface, SimulatorBroker, THSBroker, create_broker
from trading.risk import RiskManager, get_risk_manager
from trading.kill_switch import get_kill_switch
from trading.health import get_health_monitor, ComponentType, HealthStatus
from trading.alerts import get_alert_manager, AlertPriority
from utils.logger import log
from trading.alerts import get_alert_manager, AlertPriority


class ExecutionEngine:
    """
    Production execution engine with:
    - Kill switch integration
    - Health monitoring
    - Alerting
    - Reconciliation
    """
    
    def __init__(self, mode: TradingMode = None):
        self.mode = mode or CONFIG.trading_mode
        
        # Create broker
        self.broker = create_broker(self.mode.value)
        
        # Components
        self.risk_manager: Optional[RiskManager] = None
        self._kill_switch = get_kill_switch()
        self._health_monitor = get_health_monitor()
        self._alert_manager = get_alert_manager()
        
        # Order queue
        self._queue: queue.Queue = queue.Queue()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        
        # Callbacks
        self.on_fill: Optional[Callable[[Order], None]] = None
        self.on_reject: Optional[Callable[[Order, str], None]] = None
        
        # Register kill switch callback
        self._kill_switch.on_activate(self._on_kill_switch)
    
    def start(self) -> bool:
        """Start execution engine"""
        # Connect broker
        if not self.broker.connect():
            log.error("Broker connection failed")
            self._health_monitor.report_component_health(
                ComponentType.BROKER,
                HealthStatus.UNHEALTHY,
                error="Connection failed"
            )
            return False
        
        # Initialize risk manager
        account = self.broker.get_account()
        self.risk_manager = get_risk_manager()
        self.risk_manager.initialize(account)
        
        # Start components
        self._health_monitor.start()
        self._alert_manager.start()
        
        # Start execution loop
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        
        # Report healthy
        self._health_monitor.report_component_health(
            ComponentType.BROKER,
            HealthStatus.HEALTHY
        )
        
        log.info(f"Execution engine started ({self.mode.value})")
        
        # Send startup alert
        self._alert_manager.system_alert(
            "Trading System Started",
            f"Mode: {self.mode.value}, Capital: ¥{account.equity:,.2f}"
        )

        self._reconciliation_thread = threading.Thread(
            target=self._reconciliation_loop, daemon=True
        )
        self._reconciliation_thread.start()
        
        return True
    
    def _reconciliation_loop(self):
        """Periodic reconciliation with broker"""
        while self._running:
            try:
                time.sleep(300)  # Every 5 minutes
                
                if not self.broker.is_connected:
                    continue
                
                from trading.oms import get_oms
                oms = get_oms()
                
                broker_account = self.broker.get_account()
                broker_positions = self.broker.get_positions()
                
                discrepancies = oms.reconcile(broker_positions, broker_account.cash)
                
                if discrepancies.get('cash_diff', 0) > 1 or \
                discrepancies.get('position_diffs') or \
                discrepancies.get('missing_positions') or \
                discrepancies.get('extra_positions'):
                    
                    self._alert_manager.risk_alert(
                        "Reconciliation Discrepancy",
                        f"Cash diff: ¥{discrepancies.get('cash_diff', 0):.2f}",
                        discrepancies
                    )
                    
            except Exception as e:
                log.error(f"Reconciliation error: {e}")

    def stop(self):
        """Stop execution engine"""
        self._running = False
        
        if self._thread:
            self._queue.put(None)
            self._thread.join(timeout=5)
        
        self.broker.disconnect()
        self._health_monitor.stop()
        self._alert_manager.stop()
        
        log.info("Execution engine stopped")
    
    def submit(self, signal: TradeSignal) -> bool:
        """Submit trade signal"""
        # Check kill switch first
        if not self._kill_switch.can_trade:
            log.warning("Trading halted - cannot submit order")
            if self.on_reject:
                order = Order(symbol=signal.symbol, side=signal.side)
                order.status = OrderStatus.REJECTED
                order.message = "Trading halted"
                self.on_reject(order, "Trading halted")
            return False
        
        # Get price for validation
        price = signal.price
        if price is None or price <= 0:
            price = self.broker.get_quote(signal.symbol)
            if price is None or price <= 0:
                log.error(f"Cannot get price for {signal.symbol}")
                return False
        
        # Risk check
        passed, msg = self.risk_manager.check_order(
            signal.symbol, signal.side, signal.quantity, price
        )
        
        if not passed:
            log.warning(f"Risk check failed: {msg}")
            
            self._alert_manager.risk_alert(
                "Order Rejected",
                f"{signal.symbol}: {msg}",
                {'symbol': signal.symbol, 'reason': msg}
            )
            
            if self.on_reject:
                order = Order(symbol=signal.symbol, side=signal.side)
                order.status = OrderStatus.REJECTED
                order.message = msg
                self.on_reject(order, msg)
            return False
        
        self._queue.put(signal)
        log.info(f"Signal queued: {signal.side.value} {signal.quantity} {signal.symbol}")
        return True
    
    def submit_from_prediction(self, pred) -> bool:
        """Create signal from prediction"""
        from models.predictor import Signal
        
        if pred.signal == Signal.HOLD:
            return False
        
        if pred.position.shares == 0:
            return False
        
        side = OrderSide.BUY if pred.signal in [Signal.STRONG_BUY, Signal.BUY] else OrderSide.SELL
        
        signal = TradeSignal(
            symbol=pred.stock_code,
            name=pred.stock_name,
            side=side,
            quantity=pred.position.shares,
            price=pred.levels.entry,
            stop_loss=pred.levels.stop_loss,
            take_profit=pred.levels.target_2,
            confidence=pred.confidence,
            reasons=pred.reasons
        )
        
        return self.submit(signal)
    
    def _run_loop(self):
        """Main execution loop"""
        while self._running:
            try:
                signal = self._queue.get(timeout=0.1)
                
                if signal is None:
                    break
                
                self._execute(signal)
                
            except queue.Empty:
                pass
            except Exception as e:
                log.error(f"Execution loop error: {e}")
                self._alert_manager.system_alert(
                    "Execution Error",
                    str(e),
                    priority=AlertPriority.HIGH
                )
            
            # Update risk manager
            if self.risk_manager and self.broker.is_connected:
                try:
                    account = self.broker.get_account()
                    self.risk_manager.update(account)
                except Exception:
                    pass
            
            time.sleep(0.1)
    
    def _execute(self, signal: TradeSignal):
        """Execute signal"""
        try:
            if not self._kill_switch.can_trade:
                log.warning("Trading halted during execution")
                return
            
            order = Order(
                symbol=signal.symbol,
                name=signal.name,
                side=signal.side,
                quantity=signal.quantity,
                price=signal.price,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit,
                signal_id=signal.id
            )
            
            from trading.oms import get_oms
            oms = get_oms()
            order = oms.submit_order(order)

            result = self.broker.submit_order(order)
            
            if result.status == OrderStatus.FILLED:

                from core.types import Fill
                fill = Fill(
                    order_id=order.id,
                    symbol=order.symbol,
                    side=order.side,
                    quantity=result.filled_qty,
                    price=result.filled_price,
                    commission=result.commission
                )
                oms.process_fill(order, fill)

                log.info(
                    f"Filled: {result.side.value.upper()} {result.filled_qty} "
                    f"@ ¥{result.filled_price:.2f}"
                )
                
                if self.on_fill:
                    self.on_fill(result)
                
                self._alert_manager.trading_alert(
                    "Order Filled",
                    f"{result.side.value.upper()} {result.filled_qty} {result.symbol} @ ¥{result.filled_price:.2f}",
                    result.to_dict()
                )
                    
            elif result.status == OrderStatus.REJECTED:
                log.warning(f"Rejected: {result.message}")
                
                if self.on_reject:
                    self.on_reject(result, result.message)
                    
        except Exception as e:
            log.error(f"Execution error: {e}")
            
            self._alert_manager.system_alert(
                "Execution Failed",
                f"{signal.symbol}: {str(e)}",
                priority=AlertPriority.HIGH
            )
    
    def _on_kill_switch(self, reason: str):
        """Handle kill switch activation"""
        log.critical(f"Kill switch activated: {reason}")
        
        try:
            for order in self.broker.get_orders(active_only=True):
                self.broker.cancel_order(order.id)
                log.info(f"Cancelled order: {order.id}")
        except Exception as e:
            log.error(f"Failed to cancel orders: {e}")
        
        self._alert_manager.critical_alert(
            "KILL SWITCH ACTIVATED",
            f"All trading halted: {reason}",
            {'reason': reason}
        )
    
    def get_account(self) -> Account:
        return self.broker.get_account()
    
    def get_positions(self):
        return self.broker.get_positions()
    
    def get_orders(self):
        return self.broker.get_orders()
    
    def reconcile(self) -> Dict:
        """Reconcile with broker"""
        if hasattr(self.broker, 'reconcile'):
            return self.broker.reconcile()
        return {}