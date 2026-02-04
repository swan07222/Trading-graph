"""
Risk Management System
"""
from typing import Tuple, Dict
from datetime import date

from config import CONFIG
from .broker_base import Account, OrderSide
from utils.logger import log


class RiskManager:
    """
    Risk management with:
    - Position limits
    - Daily loss limits
    - Order validation
    - Position sizing
    """
    
    def __init__(self, account: Account):
        self.account = account
        self.initial_equity = account.equity
        self.daily_start_equity = account.equity
        self.trades_today = 0
        self._last_date = date.today()
    
    def update(self, account: Account):
        """Update account state and check for new day"""
        self.account = account
        
        # Auto-reset on new trading day
        today = date.today()
        if today != self._last_date:
            self.new_day()
            self._last_date = today
    
    def new_day(self):
        """Reset for new trading day"""
        self.daily_start_equity = self.account.equity
        self.trades_today = 0
        log.info("Risk manager: New trading day reset")
    
    def check_order(self, 
                    code: str, 
                    side: OrderSide, 
                    quantity: int, 
                    price: float) -> Tuple[bool, str]:
        """Check if order passes risk rules"""
        
        # Daily loss limit
        daily_pnl_pct = (self.account.equity - self.daily_start_equity) / self.daily_start_equity * 100
        if daily_pnl_pct <= -CONFIG.MAX_DAILY_LOSS_PCT:
            return False, f"Daily loss limit reached: {daily_pnl_pct:.1f}%"
        
        if side == OrderSide.BUY:
            # Lot size
            if quantity % CONFIG.LOT_SIZE != 0:
                return False, f"Must be multiple of {CONFIG.LOT_SIZE}"
            
            # Cash check
            cost = quantity * price * (1 + CONFIG.COMMISSION)
            if cost > self.account.available:
                return False, f"Insufficient funds: need ¥{cost:,.2f}, have ¥{self.account.available:,.2f}"
            
            # Position limit
            position_pct = (quantity * price) / self.account.equity * 100
            if position_pct > CONFIG.MAX_POSITION_PCT:
                return False, f"Position too large: {position_pct:.1f}% > {CONFIG.MAX_POSITION_PCT}%"
            
            # Max positions
            if code not in self.account.positions and len(self.account.positions) >= CONFIG.MAX_POSITIONS:
                return False, f"Max positions: {CONFIG.MAX_POSITIONS}"
        
        else:  # SELL
            pos = self.account.positions.get(code)
            if not pos:
                return False, "No position"
            if quantity > pos.available_qty:
                return False, f"Available: {pos.available_qty}"
        
        return True, "OK"
    
    def calculate_size(self, 
                       entry: float, 
                       stop_loss: float, 
                       confidence: float) -> int:
        """Calculate position size based on risk"""
        risk_per_share = abs(entry - stop_loss)
        
        if risk_per_share <= 0:
            return 0
        
        # Risk amount
        risk_amount = self.account.equity * (CONFIG.RISK_PER_TRADE / 100)
        risk_amount *= confidence  # Adjust by confidence
        
        # Calculate shares
        shares = int(risk_amount / risk_per_share)
        shares = (shares // CONFIG.LOT_SIZE) * CONFIG.LOT_SIZE
        
        # Position limit
        max_value = self.account.equity * (CONFIG.MAX_POSITION_PCT / 100)
        max_shares = int(max_value / entry / CONFIG.LOT_SIZE) * CONFIG.LOT_SIZE
        
        return min(shares, max_shares)
    
    def get_status(self) -> Dict:
        """Get risk status"""
        daily_pnl = self.account.equity - self.daily_start_equity
        daily_pnl_pct = daily_pnl / self.daily_start_equity * 100 if self.daily_start_equity > 0 else 0
        
        return {
            'equity': self.account.equity,
            'daily_pnl': daily_pnl,
            'daily_pnl_pct': daily_pnl_pct,
            'positions': len(self.account.positions),
            'max_positions': CONFIG.MAX_POSITIONS,
            'available': self.account.available,
            'loss_limit_remaining': CONFIG.MAX_DAILY_LOSS_PCT + daily_pnl_pct
        }