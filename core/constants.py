# core/constants.py
import importlib
import re
from datetime import date, datetime, time
from enum import Enum
from functools import lru_cache
from pathlib import Path

from core.types import OrderSide, OrderStatus, OrderType
from utils.logger import get_logger

log = get_logger(__name__)


class Exchange(Enum):
    """Stock exchanges"""
    SSE = "SSE"       # Shanghai Stock Exchange
    SZSE = "SZSE"     # Shenzhen Stock Exchange
    BSE = "BSE"       # Beijing Stock Exchange
    HKEX = "HKEX"     # Hong Kong
    NYSE = "NYSE"     # New York
    NASDAQ = "NASDAQ"

EXCHANGES = {
    "SSE": {
        "name": "Shanghai Stock Exchange",
        "timezone": "Asia/Shanghai",
        "currency": "CNY",
        "prefix": ("600", "601", "603", "605", "688"),
    },
    "SZSE": {
        "name": "Shenzhen Stock Exchange",
        "timezone": "Asia/Shanghai",
        "currency": "CNY",
        "prefix": ("000", "001", "002", "003", "300", "301"),
    },
    "BSE": {
        "name": "Beijing Stock Exchange",
        "timezone": "Asia/Shanghai",
        "currency": "CNY",
        "prefix": ("83", "87", "43"),
    },
}

TRADING_HOURS = {
    "SSE": {
        "morning": (time(9, 30), time(11, 30)),
        "afternoon": (time(13, 0), time(15, 0)),
        "pre_open": (time(9, 15), time(9, 25)),
        "pre_close": (time(14, 57), time(15, 0)),
    },
    "SZSE": {
        "morning": (time(9, 30), time(11, 30)),
        "afternoon": (time(13, 0), time(15, 0)),
        "pre_open": (time(9, 15), time(9, 25)),
        "pre_close": (time(14, 57), time(15, 0)),
    },
}

# HOLIDAYS (2024-2026 China)
# Note: These are approximate holidays based on historical patterns.
# For production use, consider integrating with a dynamic holiday API.

HOLIDAYS_2024: set[date] = {
    date(2024, 1, 1),  # New Year's Day
    date(2024, 2, 9), date(2024, 2, 10), date(2024, 2, 11),
    date(2024, 2, 12), date(2024, 2, 13), date(2024, 2, 14),
    date(2024, 2, 15), date(2024, 2, 16), date(2024, 2, 17),  # Spring Festival
    date(2024, 4, 4), date(2024, 4, 5), date(2024, 4, 6),  # Qingming Festival
    date(2024, 5, 1), date(2024, 5, 2), date(2024, 5, 3),
    date(2024, 5, 4), date(2024, 5, 5),  # Labor Day
    date(2024, 6, 8), date(2024, 6, 9), date(2024, 6, 10),  # Dragon Boat Festival
    date(2024, 9, 15), date(2024, 9, 16), date(2024, 9, 17),  # Mid-Autumn Festival
    date(2024, 10, 1), date(2024, 10, 2), date(2024, 10, 3),
    date(2024, 10, 4), date(2024, 10, 5), date(2024, 10, 6),
    date(2024, 10, 7),  # National Day
}

HOLIDAYS_2025: set[date] = {
    date(2025, 1, 1),  # New Year's Day
    date(2025, 1, 28), date(2025, 1, 29), date(2025, 1, 30),
    date(2025, 1, 31), date(2025, 2, 1), date(2025, 2, 2),
    date(2025, 2, 3), date(2025, 2, 4),  # Spring Festival
    date(2025, 4, 4), date(2025, 4, 5), date(2025, 4, 6),  # Qingming Festival
    date(2025, 5, 1), date(2025, 5, 2), date(2025, 5, 3),
    date(2025, 5, 4), date(2025, 5, 5),  # Labor Day
    date(2025, 5, 31), date(2025, 6, 1), date(2025, 6, 2),  # Dragon Boat Festival
    date(2025, 10, 1), date(2025, 10, 2), date(2025, 10, 3),
    date(2025, 10, 4), date(2025, 10, 5), date(2025, 10, 6),
    date(2025, 10, 7), date(2025, 10, 8),  # National Day + Mid-Autumn
}

HOLIDAYS_2026: set[date] = {
    date(2026, 1, 1), date(2026, 1, 2), date(2026, 1, 3),  # New Year's Day
    date(2026, 2, 15), date(2026, 2, 16), date(2026, 2, 17),
    date(2026, 2, 18), date(2026, 2, 19), date(2026, 2, 20),
    date(2026, 2, 21), date(2026, 2, 22), date(2026, 2, 23),  # Spring Festival
    date(2026, 4, 4), date(2026, 4, 5), date(2026, 4, 6),  # Qingming Festival
    date(2026, 5, 1), date(2026, 5, 2), date(2026, 5, 3),
    date(2026, 5, 4), date(2026, 5, 5),  # Labor Day
    date(2026, 6, 19), date(2026, 6, 20), date(2026, 6, 21),  # Dragon Boat Festival
    date(2026, 9, 25), date(2026, 9, 26), date(2026, 9, 27),  # Mid-Autumn Festival
    date(2026, 10, 1), date(2026, 10, 2), date(2026, 10, 3),
    date(2026, 10, 4), date(2026, 10, 5), date(2026, 10, 6),
    date(2026, 10, 7),  # National Day
}

_HOLIDAYS_BUILTIN = HOLIDAYS_2024 | HOLIDAYS_2025 | HOLIDAYS_2026
_BUILTIN_HOLIDAY_YEARS = frozenset(int(d.year) for d in _HOLIDAYS_BUILTIN)

# Keep the old name for backward compat but don't use it for lookups
HOLIDAYS = _HOLIDAYS_BUILTIN

ORDER_SIDES = {s.value: s for s in OrderSide}
ORDER_TYPES = {t.value: t for t in OrderType}
ORDER_STATUS = {s.value: s for s in OrderStatus}

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

PRICE_LIMITS = {
    "main_board": 0.10,      # ±10%
    "star_market": 0.20,     # ±20% (科创板)
    "chinext": 0.20,         # ±20% (创业板)
    "st": 0.05,              # ±5%
    "new_listing": 0.44,     # +44% / -36% first day
    "bse": 0.30,             # ±30% (北交所)
}

LOT_SIZES = {
    "main_board": 100,
    "star_market": 200,
    "chinext": 100,
    "bse": 100,
    "hk": 1,  # Various lot sizes
}

TRANSACTION_COSTS = {
    "commission": 0.00025,    # 0.025% (negotiable)
    "commission_min": 5.0,    # Minimum ¥5
    "stamp_tax": 0.001,       # 0.1% (sell only)
    "transfer_fee": 0.00002,  # 0.002% (SSE only)
    "slippage": 0.001,        # 0.1% estimated
}

MA_PERIODS = [5, 10, 20, 30, 60, 120, 250]

RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70

MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

BB_PERIOD = 20
BB_STD = 2

FEATURE_GROUPS = {
    "price": ["returns", "log_returns", "price_position"],
    "volume": ["volume_ratio", "vwap_ratio", "obv_slope"],
    "volatility": ["volatility_5", "volatility_20", "atr_pct"],
    "momentum": ["rsi_14", "macd_hist", "momentum_10"],
    "trend": ["ma_ratio_5_20", "adx", "trend_strength"],
}

LABEL_UP = 2
LABEL_NEUTRAL = 1
LABEL_DOWN = 0

LABEL_NAMES = {
    LABEL_UP: "UP",
    LABEL_NEUTRAL: "NEUTRAL",
    LABEL_DOWN: "DOWN",
}

# FIX: Removed duplicate RiskLevel enum — use core.types.RiskLevel instead.
#      Kept RISK_COLORS referencing the canonical enum.

from core.types import RiskLevel  # noqa: E402  (already imported above indirectly)

RISK_COLORS = {
    RiskLevel.LOW: "#4CAF50",
    RiskLevel.MEDIUM: "#FFC107",
    RiskLevel.HIGH: "#FF9800",
    RiskLevel.CRITICAL: "#F44336",
}

DEFAULT_RISK_LIMITS = {
    "max_position_pct": 15.0,
    "max_daily_loss_pct": 3.0,
    "max_drawdown_pct": 15.0,
    "max_positions": 10,
    "var_confidence": 0.95,
}

COLORS = {
    "background": "#0d1117",
    "surface": "#161b22",
    "primary": "#58a6ff",
    "secondary": "#8b949e",
    "success": "#3fb950",
    "warning": "#d29922",
    "error": "#f85149",
    "text": "#c9d1d9",
    "text_secondary": "#8b949e",
    "border": "#30363d",
}

FONTS = {
    "h1": 24,
    "h2": 20,
    "h3": 16,
    "body": 12,
    "small": 10,
    "mono": "Consolas",
}

# FIX Bug 6: Detect ST labels only at name prefix.
_ST_PREFIX_PATTERN = re.compile(r"^\s*\*?\s*ST", re.IGNORECASE)


def get_exchange(code: str) -> str:
    """Get exchange from stock code."""
    code = str(code).zfill(6)

    for exchange, info in EXCHANGES.items():
        for prefix in info["prefix"]:
            if code.startswith(prefix):
                return exchange

    return "UNKNOWN"


@lru_cache(maxsize=1)
def _load_external_holidays() -> frozenset[date]:
    """
    Load optional external holidays file once and cache as frozenset.

    File format: <data_dir>/holidays_cn.json
    Content: ["2026-01-01", "2026-02-10", ...]
    """
    extra: set[date] = set()
    try:
        from config.settings import CONFIG

        path = Path(CONFIG.data_dir) / "holidays_cn.json"
        if path.exists():
            import json

            data = json.loads(path.read_text(encoding="utf-8"))
            rows: list[object]
            if isinstance(data, list):
                rows = list(data)
            elif isinstance(data, dict):
                maybe_rows = data.get("holidays", [])
                rows = list(maybe_rows) if isinstance(maybe_rows, list) else []
            else:
                rows = []
            for s in rows:
                try:
                    y, m, d = map(int, str(s).split("-"))
                    extra.add(date(y, m, d))
                except (TypeError, ValueError):
                    continue
    except (ImportError, OSError, TypeError, ValueError):
        return frozenset()

    return frozenset(extra)


@lru_cache(maxsize=32)
def _load_dynamic_holidays_for_year(year: int) -> frozenset[date]:
    """
    Best-effort dynamic CN holiday provider for years beyond static constants.

    Uses optional `holidays` package when available.
    """
    y = int(year)
    if y < 1990 or y > 2100:
        return frozenset()

    try:
        module = importlib.import_module("holidays")
    except ImportError:
        return frozenset()

    country_holidays = getattr(module, "country_holidays", None)
    if not callable(country_holidays):
        return frozenset()

    try:
        rows = country_holidays("CN", years=[y])
    except (TypeError, ValueError, RuntimeError):
        return frozenset()

    out: set[date] = set()
    keys_attr = getattr(rows, "keys", None)
    if callable(keys_attr):
        try:
            iterator = keys_attr()
        except (TypeError, ValueError):
            return frozenset()
    else:
        iterator = rows

    for item in iterator:
        if isinstance(item, date):
            out.add(item)

    return frozenset(out)


@lru_cache(maxsize=128)
def _holidays_for_year(year: int) -> frozenset[date]:
    y = int(year)
    out = {d for d in _HOLIDAYS_BUILTIN if int(d.year) == y}
    out.update(d for d in _load_external_holidays() if int(d.year) == y)
    if (y not in _BUILTIN_HOLIDAY_YEARS) or (not out):
        out.update(_load_dynamic_holidays_for_year(y))
    return frozenset(out)


def _holiday_window_years(anchor_year: int) -> tuple[int, ...]:
    years = {
        int(anchor_year) - 1,
        int(anchor_year),
        int(anchor_year) + 1,
        int(anchor_year) + 2,
    }
    years.update(int(y) for y in _BUILTIN_HOLIDAY_YEARS)
    years.update(int(d.year) for d in _load_external_holidays())
    return tuple(sorted(y for y in years if y >= 1990))


@lru_cache(maxsize=8)
def _holiday_window(anchor_year: int) -> frozenset[date]:
    out: set[date] = set()
    for year in _holiday_window_years(anchor_year):
        out.update(_holidays_for_year(year))
    return frozenset(out)


def get_holidays() -> frozenset[date]:
    """
    Return holiday set for current runtime window.

    Includes:
    - built-in constants
    - optional external holidays file
    - optional dynamic provider for years outside built-in coverage
    """
    return _holiday_window(datetime.now().year)


def get_price_limit(code: str, name: str | None = None) -> float:
    """
    Get price limit for stock.

    Args:
        code: Stock code
        name: Stock name (optional, for ST detection)

    Returns:
        Price limit as decimal (e.g., 0.10 for 10%)
    """
    code = str(code).zfill(6)

    if name and is_st_stock(name):
        return PRICE_LIMITS["st"]

    # STAR Market (科创板)
    if code.startswith("688"):
        return PRICE_LIMITS["star_market"]

    # ChiNext (创业板)
    if code.startswith("30"):
        return PRICE_LIMITS["chinext"]

    # BSE (北交所)
    if code.startswith(("83", "43", "87")):
        return PRICE_LIMITS["bse"]

    return PRICE_LIMITS["main_board"]


def get_lot_size(code: str) -> int:
    """Get lot size for stock."""
    code = str(code).zfill(6)

    if code.startswith("688"):
        return LOT_SIZES["star_market"]

    return LOT_SIZES["main_board"]


def is_trading_day(d: date) -> bool:
    """
    Check if date is a trading day (weekend + holiday aware).

    Uses year-scoped holiday cache so future years can be resolved via
    optional dynamic providers when static constants are outdated.
    """
    if d.weekday() >= 5:
        return False
    return d not in _holidays_for_year(int(d.year))


def is_trading_time(exchange: str = "SSE") -> bool:
    """
    Check if current time is within trading hours.

    Uses Asia/Shanghai timezone for accurate trading time detection.
    Falls back to local time with a warning if zoneinfo is unavailable.
    """
    try:
        from zoneinfo import ZoneInfo

        now = datetime.now(tz=ZoneInfo("Asia/Shanghai")).time()
    except Exception as e:
        # Log warning once to avoid spam
        if not hasattr(is_trading_time, "_warned"):
            log.warning(
                "zoneinfo unavailable (%s), using local time for trading hours. "
                "This may give incorrect results on non-Chinese servers.",
                e,
            )
            is_trading_time._warned = True  # type: ignore
        now = datetime.now().time()

    hours = TRADING_HOURS.get(exchange, TRADING_HOURS["SSE"])

    morning = hours["morning"][0] <= now <= hours["morning"][1]
    afternoon = hours["afternoon"][0] <= now <= hours["afternoon"][1]

    return morning or afternoon


def is_st_stock(name: str) -> bool:
    """
    Check if stock is ST.

    Detects ST at prefix only, e.g. "ST600000", "*ST600000", "ST ABC".
    Avoids false positives like "BEST" or "FASTEST".
    """
    if not name:
        return False
    return bool(_ST_PREFIX_PATTERN.match(str(name)))
