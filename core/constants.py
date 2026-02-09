"""
System Constants - Immutable configuration values
Score Target: 10/10

Central repository for all magic numbers and constant values.
"""
from datetime import time, date
from typing import Dict, List, Set, Tuple
from enum import Enum, auto
from core.types import OrderSide, OrderType, OrderStatus

# =============================================================================
# EXCHANGES
# =============================================================================

class Exchange(Enum):
    """Stock exchanges"""
    SSE = "SSE"      # Shanghai Stock Exchange
    SZSE = "SZSE"    # Shenzhen Stock Exchange
    BSE = "BSE"      # Beijing Stock Exchange
    HKEX = "HKEX"    # Hong Kong
    NYSE = "NYSE"    # New York
    NASDAQ = "NASDAQ"


EXCHANGES = {
    'SSE': {
        'name': 'Shanghai Stock Exchange',
        'timezone': 'Asia/Shanghai',
        'currency': 'CNY',
        'prefix': ['600', '601', '603', '605', '688'],
    },
    'SZSE': {
        'name': 'Shenzhen Stock Exchange',
        'timezone': 'Asia/Shanghai',
        'currency': 'CNY',
        'prefix': ['000', '001', '002', '003', '300', '301'],
    },
    'BSE': {
        'name': 'Beijing Stock Exchange',
        'timezone': 'Asia/Shanghai',
        'currency': 'CNY',
        'prefix': ['83', '87', '43'],
    },
}


# =============================================================================
# TRADING HOURS
# =============================================================================

TRADING_HOURS = {
    'SSE': {
        'morning': (time(9, 30), time(11, 30)),
        'afternoon': (time(13, 0), time(15, 0)),
        'pre_open': (time(9, 15), time(9, 25)),
        'pre_close': (time(14, 57), time(15, 0)),
    },
    'SZSE': {
        'morning': (time(9, 30), time(11, 30)),
        'afternoon': (time(13, 0), time(15, 0)),
        'pre_open': (time(9, 15), time(9, 25)),
        'pre_close': (time(14, 57), time(15, 0)),
    },
}


# =============================================================================
# HOLIDAYS (2024-2025 China)
# =============================================================================

HOLIDAYS_2024: Set[date] = {
    # New Year
    date(2024, 1, 1),
    # Spring Festival
    date(2024, 2, 9), date(2024, 2, 10), date(2024, 2, 11),
    date(2024, 2, 12), date(2024, 2, 13), date(2024, 2, 14),
    date(2024, 2, 15), date(2024, 2, 16), date(2024, 2, 17),
    # Qingming
    date(2024, 4, 4), date(2024, 4, 5), date(2024, 4, 6),
    # Labor Day
    date(2024, 5, 1), date(2024, 5, 2), date(2024, 5, 3),
    date(2024, 5, 4), date(2024, 5, 5),
    # Dragon Boat
    date(2024, 6, 8), date(2024, 6, 9), date(2024, 6, 10),
    # Mid-Autumn
    date(2024, 9, 15), date(2024, 9, 16), date(2024, 9, 17),
    # National Day
    date(2024, 10, 1), date(2024, 10, 2), date(2024, 10, 3),
    date(2024, 10, 4), date(2024, 10, 5), date(2024, 10, 6),
    date(2024, 10, 7),
}

HOLIDAYS_2025: Set[date] = {
    # New Year
    date(2025, 1, 1),
    # Spring Festival (estimated)
    date(2025, 1, 28), date(2025, 1, 29), date(2025, 1, 30),
    date(2025, 1, 31), date(2025, 2, 1), date(2025, 2, 2),
    date(2025, 2, 3), date(2025, 2, 4),
}

HOLIDAYS = HOLIDAYS_2024 | HOLIDAYS_2025


ORDER_SIDES = {s.value: s for s in OrderSide}
ORDER_TYPES = {t.value: t for t in OrderType}
ORDER_STATUS = {s.value: s for s in OrderStatus}


# =============================================================================
# SIGNAL CONSTANTS
# =============================================================================

class SignalType(Enum):
    """Trading signal type"""
    STRONG_BUY = "STRONG_BUY"
    BUY = "BUY"
    HOLD = "HOLD"
    SELL = "SELL"
    STRONG_SELL = "STRONG_SELL"


SIGNAL_TYPES = {s.value: s for s in SignalType}

SIGNAL_COLORS = {
    SignalType.STRONG_BUY: "#00C853",
    SignalType.BUY: "#4CAF50",
    SignalType.HOLD: "#FFC107",
    SignalType.SELL: "#FF5722",
    SignalType.STRONG_SELL: "#D50000",
}


# =============================================================================
# TRADING RULES
# =============================================================================

# Price limits by board type
PRICE_LIMITS = {
    'main_board': 0.10,      # ±10%
    'star_market': 0.20,     # ±20% (科创板)
    'chinext': 0.20,         # ±20% (创业板)
    'st': 0.05,              # ±5%
    'new_listing': 0.44,     # +44% / -36% first day
    'bse': 0.30,             # ±30% (北交所)
}

# Lot sizes by market
LOT_SIZES = {
    'main_board': 100,
    'star_market': 200,
    'chinext': 100,
    'bse': 100,
    'hk': 1,  # Various lot sizes
}

# Transaction costs
TRANSACTION_COSTS = {
    'commission': 0.00025,    # 0.025% (negotiable)
    'commission_min': 5.0,    # Minimum ¥5
    'stamp_tax': 0.001,       # 0.1% (sell only)
    'transfer_fee': 0.00002,  # 0.002% (SSE only)
    'slippage': 0.001,        # 0.1% estimated
}


# =============================================================================
# TECHNICAL ANALYSIS
# =============================================================================

# Common MA periods
MA_PERIODS = [5, 10, 20, 30, 60, 120, 250]

# RSI thresholds
RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70

# MACD parameters
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

# Bollinger Bands
BB_PERIOD = 20
BB_STD = 2


# =============================================================================
# MACHINE LEARNING
# =============================================================================

# Feature groups
FEATURE_GROUPS = {
    'price': ['returns', 'log_returns', 'price_position'],
    'volume': ['volume_ratio', 'vwap_ratio', 'obv_slope'],
    'volatility': ['volatility_5', 'volatility_20', 'atr_pct'],
    'momentum': ['rsi_14', 'macd_hist', 'momentum_10'],
    'trend': ['ma_ratio_5_20', 'adx', 'trend_strength'],
}

# Label definitions
LABEL_UP = 2
LABEL_NEUTRAL = 1
LABEL_DOWN = 0

LABEL_NAMES = {
    LABEL_UP: 'UP',
    LABEL_NEUTRAL: 'NEUTRAL',
    LABEL_DOWN: 'DOWN',
}


# =============================================================================
# RISK MANAGEMENT
# =============================================================================

# Risk levels
class RiskLevel(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


RISK_COLORS = {
    RiskLevel.LOW: "#4CAF50",
    RiskLevel.MEDIUM: "#FFC107",
    RiskLevel.HIGH: "#FF9800",
    RiskLevel.CRITICAL: "#F44336",
}

# Default risk limits
DEFAULT_RISK_LIMITS = {
    'max_position_pct': 15.0,
    'max_daily_loss_pct': 3.0,
    'max_drawdown_pct': 15.0,
    'max_positions': 10,
    'var_confidence': 0.95,
}


# =============================================================================
# UI CONSTANTS
# =============================================================================

# Color scheme
COLORS = {
    'background': '#0d1117',
    'surface': '#161b22',
    'primary': '#58a6ff',
    'secondary': '#8b949e',
    'success': '#3fb950',
    'warning': '#d29922',
    'error': '#f85149',
    'text': '#c9d1d9',
    'text_secondary': '#8b949e',
    'border': '#30363d',
}

# Font sizes
FONTS = {
    'h1': 24,
    'h2': 20,
    'h3': 16,
    'body': 12,
    'small': 10,
    'mono': 'Consolas',
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_exchange(code: str) -> str:
    """Get exchange from stock code"""
    code = str(code).zfill(6)
    
    for exchange, info in EXCHANGES.items():
        for prefix in info['prefix']:
            if code.startswith(prefix):
                return exchange
    
    return 'UNKNOWN'


def get_price_limit(code: str, name: str = None) -> float:
    """
    Get price limit for stock.
    
    Args:
        code: Stock code
        name: Stock name (optional, for ST detection)
    
    Returns:
        Price limit as decimal (e.g., 0.10 for 10%)
    """
    code = str(code).zfill(6)
    
    # Check ST first if name provided
    if name and is_st_stock(name):
        return PRICE_LIMITS['st']
    
    # STAR Market (科创板)
    if code.startswith('688'):
        return PRICE_LIMITS['star_market']
    
    # ChiNext (创业板)
    if code.startswith('30'):
        return PRICE_LIMITS['chinext']
    
    # BSE (北交所)
    if code.startswith(('83', '43', '87')):
        return PRICE_LIMITS['bse']
    
    # Main board
    return PRICE_LIMITS['main_board']


def get_lot_size(code: str) -> int:
    """Get lot size for stock"""
    code = str(code).zfill(6)
    
    # STAR Market
    if code.startswith('688'):
        return LOT_SIZES['star_market']
    
    return LOT_SIZES['main_board']


def is_trading_day(d: date) -> bool:
    """Check if date is a trading day"""
    # Weekend
    if d.weekday() >= 5:
        return False
    
    # Holiday
    if d in HOLIDAYS:
        return False
    
    return True


def is_trading_time(exchange: str = 'SSE') -> bool:
    """Check if current time is trading time"""
    from datetime import datetime
    
    now = datetime.now().time()
    hours = TRADING_HOURS.get(exchange, TRADING_HOURS['SSE'])
    
    morning = hours['morning'][0] <= now <= hours['morning'][1]
    afternoon = hours['afternoon'][0] <= now <= hours['afternoon'][1]
    
    return morning or afternoon


def is_st_stock(name: str) -> bool:
    """Check if stock is ST"""
    if not name:
        return False
    name_upper = name.upper()
    return 'ST' in name_upper or '*ST' in name_upper