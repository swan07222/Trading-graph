"""
Risk Management System

FIXED Issues:
- Automatic daily reset
- Proper date tracking
- Comprehensive risk checks

Author: AI Trading System
Version: 2.0
"""
from typing import Tuple, Dict
from datetime import date, datetime

from config import CONFIG
from .broker_base import Account, OrderSide
from utils.logger import log


class RiskManager:
    """
    Risk Management System
    
    Features:
    - Position size limits
    - Daily loss limits with auto-reset
    - Order validation
    - Risk-based position sizing
    
    Usage:
        risk_mgr = RiskManager(account)
        passed, msg = risk_mgr.check_order(code, side, qty, price)
    """
    
    def __init__(self, account: Account):
        self.account = account
        self.initial_equity = account.equity
        self.daily_start_equity = account.equity
        self.trades_today = 0
        self._last_date = date.today()
    
    def update(self, account: Account):
        """
        Update account state
        Automatically resets on new trading day
        """
        self.account = account
        
        # Auto-reset on new day
        today = date.today()
        if today != self._last_date:
            self._new_day()
            self._last_date = today
    
    def _new_day(self):
        """Reset for new trading day"""
        log.info("Risk manager: New trading day - resetting daily limits")
        self.daily_start_equity = self.account.equity
        self.trades_today = 0
    
    def force_reset(self):
        """Force reset (for manual override)"""
        self.daily_start_equity = self.account.equity
        self.trades_today = 0
        self._last_date = date.today()
        log.info("Risk manager: Forced reset")
    
    def check_order(self,
                    code: str,
                    side: OrderSide,
                    quantity: int,
                    price: float) -> Tuple[bool, str]:
        """
        Check if order passes all risk rules
        
        Args:
            code: Stock code
            side: BUY or SELL
            quantity: Number of shares
            price: Order price
            
        Returns:
            (passed, message) tuple
        """
        # Update for new day if needed
        today = date.today()
        if today != self._last_date:
            self._new_day()
            self._last_date = today
        
        # Check daily loss limit
        if self.account.equity > 0:
            daily_pnl_pct = (self.account.equity - self.daily_start_equity) / self.daily_start_equity * 100
            
            if daily_pnl_pct <= -CONFIG.MAX_DAILY_LOSS_PCT:
                return False, f"Daily loss limit reached: {daily_pnl_pct:.1f}% (limit: -{CONFIG.MAX_DAILY_LOSS_PCT}%)"
        
        if side == OrderSide.BUY:
            return self._check_buy_order(code, quantity, price)
        else:
            return self._check_sell_order(code, quantity)
    
    def _check_buy_order(self, code: str, quantity: int, price: float) -> Tuple[bool, str]:
        """Validate buy order"""
        # Lot size check
        if quantity <= 0:
            return False, "Quantity must be positive"
        
        if quantity % CONFIG.LOT_SIZE != 0:
            return False, f"Quantity must be multiple of {CONFIG.LOT_SIZE}"
        
        # Cash check
        cost = quantity * price
        commission = cost * CONFIG.COMMISSION
        total_cost = cost + commission
        
        if total_cost > self.account.available:
            return False, (
                f"Insufficient funds: need ¥{total_cost:,.2f}, "
                f"available ¥{self.account.available:,.2f}"
            )
        
        # Position size limit
        if self.account.equity > 0:
            position_pct = (quantity * price) / self.account.equity * 100
            
            if position_pct > CONFIG.MAX_POSITION_PCT:
                return False, (
                    f"Position too large: {position_pct:.1f}% "
                    f"(max: {CONFIG.MAX_POSITION_PCT}%)"
                )
        
        # Max positions limit
        if code not in self.account.positions:
            if len(self.account.positions) >= CONFIG.MAX_POSITIONS:
                return False, f"Max positions reached: {CONFIG.MAX_POSITIONS}"
        
        return True, "OK"
    
    def _check_sell_order(self, code: str, quantity: int) -> Tuple[bool, str]:
        """Validate sell order"""
        pos = self.account.positions.get(code)
        
        if not pos:
            return False, f"No position in {code}"
        
        if quantity > pos.available_qty:
            return False, (
                f"Insufficient shares: requested {quantity}, "
                f"available {pos.available_qty}"
            )
        
        return True, "OK"
    
    def calculate_position_size(self,
                                entry: float,
                                stop_loss: float,
                                confidence: float = 1.0) -> int:
        """
        Calculate position size based on risk
        
        Args:
            entry: Entry price
            stop_loss: Stop loss price
            confidence: Confidence multiplier (0-1)
            
        Returns:
            Number of shares (multiple of LOT_SIZE)
        """
        risk_per_share = abs(entry - stop_loss)
        
        if risk_per_share <= 0:
            return 0
        
        # Risk amount based on account and risk per trade
        risk_amount = self.account.equity * (CONFIG.RISK_PER_TRADE / 100)
        
        # Adjust by confidence
        risk_amount *= max(0, min(1, confidence))
        
        # Calculate shares
        shares = int(risk_amount / risk_per_share)
        
        # Round to lot size
        shares = (shares // CONFIG.LOT_SIZE) * CONFIG.LOT_SIZE
        
        # Apply position limit
        max_value = self.account.equity * (CONFIG.MAX_POSITION_PCT / 100)
        max_shares = int(max_value / entry / CONFIG.LOT_SIZE) * CONFIG.LOT_SIZE
        
        shares = min(shares, max_shares)
        
        # Ensure we can afford it
        max_affordable = int(self.account.available / entry / CONFIG.LOT_SIZE) * CONFIG.LOT_SIZE
        shares = min(shares, max_affordable)
        
        return max(0, shares)
    
    def get_status(self) -> Dict:
        """Get current risk status"""
        if self.daily_start_equity > 0:
            daily_pnl = self.account.equity - self.daily_start_equity
            daily_pnl_pct = daily_pnl / self.daily_start_equity * 100
        else:
            daily_pnl = 0
            daily_pnl_pct = 0
        
        loss_limit_remaining = CONFIG.MAX_DAILY_LOSS_PCT + daily_pnl_pct
        
        return {
            'equity': self.account.equity,
            'available': self.account.available,
            'daily_pnl': daily_pnl,
            'daily_pnl_pct': daily_pnl_pct,
            'positions_count': len(self.account.positions),
            'max_positions': CONFIG.MAX_POSITIONS,
            'loss_limit_remaining': max(0, loss_limit_remaining),
            'can_trade': loss_limit_remaining > 0,
            'trades_today': self.trades_today
        }
    
    def record_trade(self):
        """Record a trade execution"""
        self.trades_today += 1