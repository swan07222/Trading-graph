"""
Configuration - AI Stock Trading System v3.0
Production-grade configuration with proper defaults

FIXED:
- Removed duplicate EPOCHS definition
- Added EMBARGO_BARS for proper train/test separation
- Added ALLOW_SHORT flag
- Cleaner organization
"""
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional
from enum import Enum
from datetime import datetime, time


class TradingMode(Enum):
    SIMULATION = "simulation"
    LIVE = "live"


class RiskProfile(Enum):
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"


@dataclass
class Config:
    """
    Production Trading Configuration
    
    All parameters are validated and have sensible defaults.
    """
    
    # === System Paths ===
    BASE_DIR: Path = field(default_factory=lambda: Path(__file__).parent)
    
    @property
    def DATA_DIR(self) -> Path:
        return self.BASE_DIR / "saved_data"
    
    @property
    def MODEL_DIR(self) -> Path:
        return self.BASE_DIR / "saved_models"
    
    @property
    def LOG_DIR(self) -> Path:
        return self.BASE_DIR / "logs"
    
    # === AI Model Architecture ===
    SEQUENCE_LENGTH: int = 60
    HIDDEN_SIZE: int = 256
    NUM_LAYERS: int = 3
    NUM_HEADS: int = 8
    DROPOUT: float = 0.3
    NUM_CLASSES: int = 3  # UP, NEUTRAL, DOWN
    
    # === Training Parameters ===
    EPOCHS: int = 100
    LEARNING_RATE: float = 0.0005
    BATCH_SIZE: int = 64
    EARLY_STOP_PATIENCE: int = 15
    TRAIN_RATIO: float = 0.70
    VAL_RATIO: float = 0.15
    TEST_RATIO: float = 0.15
    
    # === Prediction Settings ===
    PREDICTION_HORIZON: int = 5
    UP_THRESHOLD: float = 2.0      # % gain = UP
    DOWN_THRESHOLD: float = -2.0   # % loss = DOWN
    
    # === CRITICAL: Embargo to prevent label leakage ===
    # Number of bars to skip between train/val and val/test
    EMBARGO_BARS: int = 5  # Should be >= PREDICTION_HORIZON
    
    # === Signal Thresholds ===
    STRONG_BUY_THRESHOLD: float = 0.70
    BUY_THRESHOLD: float = 0.55
    SELL_THRESHOLD: float = 0.55
    STRONG_SELL_THRESHOLD: float = 0.70
    MIN_CONFIDENCE: float = 0.55
    
    # === Trading Rules (A-Share Market) ===
    LOT_SIZE: int = 100
    COMMISSION: float = 0.00025    # 0.025% per side
    STAMP_TAX: float = 0.001       # 0.1% sell only
    SLIPPAGE: float = 0.001        # 0.1% estimate
    T_PLUS_1: bool = True
    ALLOW_SHORT: bool = False      # A-shares: no shorting
    
    # === Risk Management ===
    MAX_POSITION_PCT: float = 15.0
    MAX_DAILY_LOSS_PCT: float = 3.0
    MAX_POSITIONS: int = 10
    RISK_PER_TRADE: float = 2.0
    CAPITAL: float = 100000.0
    
    # === Broker Settings ===
    TRADING_MODE: TradingMode = TradingMode.SIMULATION
    BROKER_PATH: Optional[str] = None
    
    # === Market Hours (China) ===
    MARKET_OPEN_MORNING: time = time(9, 30)
    MARKET_CLOSE_MORNING: time = time(11, 30)
    MARKET_OPEN_AFTERNOON: time = time(13, 0)
    MARKET_CLOSE_AFTERNOON: time = time(15, 0)
    
    # === Default Stock Pool ===
    STOCK_POOL: List[str] = field(default_factory=lambda: [
        # Blue Chips
        "600519", "601318", "600036", "000858", "600900",
        # Growth
        "002594", "300750", "002475", "300059", "002230",
        # Consumer
        "000333", "000651", "600887", "603288",
        # Healthcare
        "600276", "300760", "300015",
        # Finance
        "601166", "601398", "600030",
    ])
    
    def __post_init__(self):
        """Initialize directories and validate config"""
        for d in [self.DATA_DIR, self.MODEL_DIR, self.LOG_DIR]:
            d.mkdir(parents=True, exist_ok=True)
        
        # Validate
        assert self.EMBARGO_BARS >= self.PREDICTION_HORIZON, \
            f"EMBARGO_BARS ({self.EMBARGO_BARS}) must be >= PREDICTION_HORIZON ({self.PREDICTION_HORIZON})"
        assert self.TRAIN_RATIO + self.VAL_RATIO + self.TEST_RATIO == 1.0, \
            "Split ratios must sum to 1.0"
        assert 0 < self.MIN_CONFIDENCE <= 1.0, "MIN_CONFIDENCE must be in (0, 1]"
    
    def is_market_open(self) -> bool:
        """Check if A-share market is currently open"""
        now = datetime.now()
        
        if now.weekday() >= 5:  # Weekend
            return False
        
        current_time = now.time()
        
        morning = self.MARKET_OPEN_MORNING <= current_time <= self.MARKET_CLOSE_MORNING
        afternoon = self.MARKET_OPEN_AFTERNOON <= current_time <= self.MARKET_CLOSE_AFTERNOON
        
        return morning or afternoon
    
    def set_risk_profile(self, profile: RiskProfile):
        """Apply a predefined risk profile"""
        profiles = {
            RiskProfile.CONSERVATIVE: {
                'MAX_POSITION_PCT': 10.0,
                'MAX_DAILY_LOSS_PCT': 2.0,
                'RISK_PER_TRADE': 1.0,
                'MIN_CONFIDENCE': 0.60,
                'MAX_POSITIONS': 8,
            },
            RiskProfile.MODERATE: {
                'MAX_POSITION_PCT': 15.0,
                'MAX_DAILY_LOSS_PCT': 3.0,
                'RISK_PER_TRADE': 2.0,
                'MIN_CONFIDENCE': 0.55,
                'MAX_POSITIONS': 10,
            },
            RiskProfile.AGGRESSIVE: {
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


# Global singleton
CONFIG = Config()