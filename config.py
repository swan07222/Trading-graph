"""
Configuration - AI Stock Trading System
Professional trading configuration with risk management
"""
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List
from enum import Enum
from datetime import datetime


class TradingMode(Enum):
    SIMULATION = "simulation"
    LIVE = "live"


@dataclass
class Config:
    """
    Production Trading Configuration
    
    ⚠️ WARNING: This system trades with REAL MONEY in LIVE mode.
    Ensure you understand all risks before using.
    """
    
    # === System Paths ===
    BASE_DIR: Path = field(default_factory=lambda: Path(__file__).parent)
    DATA_DIR: Path = field(default_factory=lambda: Path(__file__).parent / "saved_data")
    MODEL_DIR: Path = field(default_factory=lambda: Path(__file__).parent / "saved_models")
    LOG_DIR: Path = field(default_factory=lambda: Path(__file__).parent / "logs")
    
    # === AI Model Architecture ===
    SEQUENCE_LENGTH: int = 60          # Input sequence (days)
    HIDDEN_SIZE: int = 256             # Neural network hidden size
    NUM_LAYERS: int = 3                # Network depth
    NUM_HEADS: int = 8                 # Attention heads (Transformer)
    DROPOUT: float = 0.3               # Regularization dropout
    NUM_CLASSES: int = 3               # UP, NEUTRAL, DOWN
    EPOCHS = 100
    
    # === Training Parameters ===
    LEARNING_RATE: float = 0.0005
    BATCH_SIZE: int = 64
    EPOCHS: int = 100
    EARLY_STOP_PATIENCE: int = 15
    TRAIN_RATIO: float = 0.7
    VAL_RATIO: float = 0.15
    TEST_RATIO: float = 0.15
    
    # === Prediction Settings ===
    PREDICTION_HORIZON: int = 5        # Days ahead to predict
    UP_THRESHOLD: float = 2.0          # % gain = UP signal
    DOWN_THRESHOLD: float = -2.0       # % loss = DOWN signal
    
    # === Signal Thresholds ===
    STRONG_BUY_THRESHOLD: float = 0.70
    BUY_THRESHOLD: float = 0.55
    SELL_THRESHOLD: float = 0.55
    STRONG_SELL_THRESHOLD: float = 0.70
    MIN_CONFIDENCE: float = 0.50       # Minimum confidence to trade
    
    # === Trading Rules (A-Share Market) ===
    LOT_SIZE: int = 100                # Minimum trade unit (100 shares)
    COMMISSION: float = 0.00025        # 0.025% commission rate
    STAMP_TAX: float = 0.001           # 0.1% stamp tax (sell only)
    SLIPPAGE: float = 0.001            # 0.1% slippage estimate
    T_PLUS_1: bool = True              # T+1 settlement rule
    
    # === Risk Management ===
    MAX_POSITION_PCT: float = 15.0     # Max % of capital per position
    MAX_DAILY_LOSS_PCT: float = 3.0    # Daily stop loss %
    MAX_POSITIONS: int = 10            # Max concurrent positions
    RISK_PER_TRADE: float = 2.0        # Risk % per trade
    CAPITAL: float = 100000.0          # Starting capital
    
    # === Broker Settings ===
    TRADING_MODE: TradingMode = TradingMode.SIMULATION
    BROKER_PATH: str = ""              # Broker executable path
    
    # === Default Stock Pool ===
    STOCK_POOL: List[str] = field(default_factory=lambda: [
        # Blue Chips
        "600519",  # Kweichow Moutai
        "601318",  # Ping An Insurance
        "600036",  # China Merchants Bank
        "000858",  # Wuliangye
        "600900",  # Yangtze Power
        
        # Growth Stocks
        "002594",  # BYD
        "300750",  # CATL
        "002475",  # Luxshare
        "300059",  # East Money
        "002230",  # iFlytek
        
        # Consumer
        "000333",  # Midea
        "000651",  # Gree Electric
        "600887",  # Yili
        "603288",  # Haitian
        
        # Healthcare
        "600276",  # Hengrui Medicine
        "300760",  # Mindray
        "300015",  # Aier Eye
        
        # Finance
        "601166",  # Industrial Bank
        "601398",  # ICBC
        "600030",  # CITIC Securities
    ])
    
    def __post_init__(self):
        """Initialize directories"""
        for d in [self.DATA_DIR, self.MODEL_DIR, self.LOG_DIR]:
            d.mkdir(parents=True, exist_ok=True)
    
    def is_market_open(self) -> bool:
        """Check if A-share market is open"""
        now = datetime.now()
        
        # Weekend check
        if now.weekday() >= 5:
            return False
        
        # Trading hours (China Standard Time)
        # Morning: 9:30 - 11:30
        # Afternoon: 13:00 - 15:00
        t = now.hour * 60 + now.minute
        morning = (9 * 60 + 30 <= t <= 11 * 60 + 30)
        afternoon = (13 * 60 <= t <= 15 * 60)
        
        return morning or afternoon
    
    def enable_live_trading(self, broker_path: str):
        """
        Enable live trading mode
        
        ⚠️ WARNING: This enables real money trading!
        """
        self.TRADING_MODE = TradingMode.LIVE
        self.BROKER_PATH = broker_path
    
    def set_risk_profile(self, profile: str):
        """
        Set risk profile
        
        Args:
            profile: 'conservative', 'moderate', or 'aggressive'
        """
        profiles = {
            'conservative': {
                'MAX_POSITION_PCT': 10.0,
                'MAX_DAILY_LOSS_PCT': 2.0,
                'RISK_PER_TRADE': 1.0,
                'MIN_CONFIDENCE': 0.60,
                'MAX_POSITIONS': 8,
            },
            'moderate': {
                'MAX_POSITION_PCT': 15.0,
                'MAX_DAILY_LOSS_PCT': 3.0,
                'RISK_PER_TRADE': 2.0,
                'MIN_CONFIDENCE': 0.55,
                'MAX_POSITIONS': 10,
            },
            'aggressive': {
                'MAX_POSITION_PCT': 25.0,
                'MAX_DAILY_LOSS_PCT': 5.0,
                'RISK_PER_TRADE': 3.0,
                'MIN_CONFIDENCE': 0.50,
                'MAX_POSITIONS': 15,
            }
        }
        
        if profile in profiles:
            for key, value in profiles[profile].items():
                setattr(self, key, value)
    
    def get_summary(self) -> str:
        """Get configuration summary"""
        return f"""
Configuration Summary:
======================
Trading Mode: {self.TRADING_MODE.value}
Capital: ¥{self.CAPITAL:,.2f}

AI Model:
- Sequence Length: {self.SEQUENCE_LENGTH} days
- Hidden Size: {self.HIDDEN_SIZE}
- Ensemble: LSTM, Transformer, GRU, TCN, Hybrid

Risk Management:
- Max Position: {self.MAX_POSITION_PCT}%
- Max Daily Loss: {self.MAX_DAILY_LOSS_PCT}%
- Risk per Trade: {self.RISK_PER_TRADE}%
- Max Positions: {self.MAX_POSITIONS}

Trading Rules:
- Lot Size: {self.LOT_SIZE} shares
- Commission: {self.COMMISSION * 100}%
- Stamp Tax: {self.STAMP_TAX * 100}%
- T+1 Settlement: {self.T_PLUS_1}
"""


# Global configuration instance
CONFIG = Config()