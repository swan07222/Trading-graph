"""
Execution Engine - Handles order execution with unified broker types
"""
import threading
import queue
import time
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass

from config import CONFIG, TradingMode
from .broker import (
    BrokerInterface, SimulatorBroker, THSBroker,
    Order, OrderSide, OrderStatus, create_broker
)
from .risk import RiskManager
from utils.logger import log


@dataclass
class TradeSignal:
    """Trade signal for execution"""
    stock_code: str
    side: OrderSide
    quantity: int
    price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None


class ExecutionEngine:
    """
    Order execution engine with:
    - Risk checks
    - Order queue
    - Async execution
    """
    
    def __init__(self, mode: TradingMode = None):
        self.mode = mode or CONFIG.TRADING_MODE
        
        # Create broker using factory
        self.broker = create_broker(self.mode.value)
        
        self.risk_manager: Optional[RiskManager] = None
        
        # Order queue
        self._queue = queue.Queue()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        
        # Callbacks
        self.on_fill: Optional[Callable[[Order], None]] = None
        self.on_reject: Optional[Callable[[Order, str], None]] = None
    
    def start(self) -> bool:
        """Start execution engine"""
        if not self.broker.connect():
            log.error("Broker connection failed")
            return False
        
        account = self.broker.get_account()
        self.risk_manager = RiskManager(account)
        
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        
        log.info(f"Execution engine started ({self.mode.value})")
        return True
    
    def stop(self):
        """Stop execution engine"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        self.broker.disconnect()
        log.info("Execution engine stopped")
    
    def submit(self, signal: TradeSignal) -> bool:
        """Submit trade signal"""
        # Get price for validation
        price = signal.price
        if price is None or price <= 0:
            price = self.broker.get_quote(signal.stock_code)
            if price is None or price <= 0:
                log.error(f"Cannot get price for {signal.stock_code}")
                return False
        
        # Risk check
        passed, msg = self.risk_manager.check_order(
            signal.stock_code, signal.side, signal.quantity, price
        )
        
        if not passed:
            log.warning(f"Risk check failed: {msg}")
            if self.on_reject:
                order = Order(stock_code=signal.stock_code, side=signal.side)
                order.status = OrderStatus.REJECTED
                order.message = msg
                self.on_reject(order, msg)
            return False
        
        self._queue.put(signal)
        log.info(f"Signal queued: {signal.side.value} {signal.quantity} {signal.stock_code}")
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
            stock_code=pred.stock_code,
            side=side,
            quantity=pred.position.shares,
            price=pred.levels.entry,
            stop_loss=pred.levels.stop_loss,
            take_profit=pred.levels.target_2
        )
        
        return self.submit(signal)
    
    def _run_loop(self):
        """Main execution loop"""
        while self._running:
            try:
                signal = self._queue.get(timeout=0.1)
                self._execute(signal)
            except queue.Empty:
                pass
            
            # Update risk manager
            if self.risk_manager:
                account = self.broker.get_account()
                self.risk_manager.update(account)
            
            time.sleep(0.1)
    
    def _execute(self, signal: TradeSignal):
        """Execute signal"""
        try:
            order = Order(
                stock_code=signal.stock_code,
                side=signal.side,
                quantity=signal.quantity,
                price=signal.price
            )
            
            result = self.broker.submit_order(order)
            
            if result.status == OrderStatus.FILLED:
                log.info(f"Filled: {result.side.value.upper()} {result.filled_qty} @ ¥{result.filled_price:.2f}")
                if self.risk_manager:
                    self.risk_manager.record_trade()
                if self.on_fill:
                    self.on_fill(result)
                    
            elif result.status == OrderStatus.REJECTED:
                log.warning(f"Rejected: {result.message}")
                if self.on_reject:
                    self.on_reject(result, result.message)
                    
        except Exception as e:
            log.error(f"Execution error: {e}")
    
    def get_account(self):
        return self.broker.get_account()
    
    def get_positions(self):
        return self.broker.get_positions()
    
    def get_orders(self):
        return self.broker.get_orders()