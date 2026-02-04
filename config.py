"""
Configuration - All settings in one place
"""
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict
from enum import Enum


class TradingMode(Enum):
    SIMULATION = "simulation"
    LIVE = "live"


@dataclass
class Config:
    """Main configuration"""
    
    # === Paths ===
    BASE_DIR: Path = field(default_factory=lambda: Path(__file__).parent)
    DATA_DIR: Path = field(default_factory=lambda: Path(__file__).parent / "saved_data")
    MODEL_DIR: Path = field(default_factory=lambda: Path(__file__).parent / "saved_models")
    LOG_DIR: Path = field(default_factory=lambda: Path(__file__).parent / "logs")
    
    # === Model Architecture ===
    SEQUENCE_LENGTH: int = 60          # Input sequence length
    HIDDEN_SIZE: int = 256             # Hidden layer size
    NUM_LAYERS: int = 3                # Number of layers
    NUM_HEADS: int = 8                 # Attention heads
    DROPOUT: float = 0.3               # Dropout rate
    NUM_CLASSES: int = 3               # UP, NEUTRAL, DOWN
    
    # === Training ===
    LEARNING_RATE: float = 0.0005
    BATCH_SIZE: int = 64
    EPOCHS: int = 100
    EARLY_STOP_PATIENCE: int = 15
    TRAIN_RATIO: float = 0.7
    VAL_RATIO: float = 0.15
    TEST_RATIO: float = 0.15
    
    # === Prediction ===
    PREDICTION_HORIZON: int = 5        # Days ahead to predict
    UP_THRESHOLD: float = 2.0          # % gain = UP
    DOWN_THRESHOLD: float = -2.0       # % loss = DOWN
    
    # === Signals ===
    STRONG_BUY_THRESHOLD: float = 0.75
    BUY_THRESHOLD: float = 0.60
    SELL_THRESHOLD: float = 0.60
    STRONG_SELL_THRESHOLD: float = 0.75
    MIN_CONFIDENCE: float = 0.55
    
    # === Trading Rules (China A-Share) ===
    LOT_SIZE: int = 100                # Minimum trade unit
    COMMISSION: float = 0.0003         # 0.03%
    STAMP_TAX: float = 0.001           # 0.1% (sell only)
    SLIPPAGE: float = 0.001            # 0.1%
    T_PLUS_1: bool = True              # T+1 rule
    
    # === Risk Management ===
    MAX_POSITION_PCT: float = 15.0     # Max single position
    MAX_DAILY_LOSS_PCT: float = 3.0    # Max daily loss
    MAX_POSITIONS: int = 10            # Max concurrent positions
    RISK_PER_TRADE: float = 2.0        # % risk per trade
    CAPITAL: float = 100000.0          # Starting capital
    
    # === Broker ===
    TRADING_MODE: TradingMode = TradingMode.SIMULATION
    BROKER_PATH: str = ""              # Path to broker executable
    
    # === Default Stocks ===
    STOCK_POOL: List[str] = field(default_factory=lambda: [
        "600519",  # 贵州茅台
        "601318",  # 中国平安
        "600036",  # 招商银行
        "000858",  # 五粮液
        "002594",  # 比亚迪
        "600276",  # 恒瑞医药
        "000333",  # 美的集团
        "601888",  # 中国中免
        "600900",  # 长江电力
        "000001",  # 平安银行
    ])
    
    def __post_init__(self):
        """Create directories"""
        for d in [self.DATA_DIR, self.MODEL_DIR, self.LOG_DIR]:
            d.mkdir(parents=True, exist_ok=True)
    
    def is_market_open(self) -> bool:
        """Check if market is open"""
        from datetime import datetime
        now = datetime.now()
        
        if now.weekday() >= 5:
            return False
        
        t = now.hour * 60 + now.minute
        return (9*60+30 <= t <= 11*60+30) or (13*60 <= t <= 15*60)


# Global config instance
CONFIG = Config()