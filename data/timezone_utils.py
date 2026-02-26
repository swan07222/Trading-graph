# data/timezone_utils.py
"""Unified timezone and trading session handling.

FIX 2026-02-26: Addresses disadvantages:
- Timezone confusion with explicit handling
- Trading session filtering for China A-shares
- Naive/aware timestamp conversion
- Holiday calendar awareness

Features:
- Shanghai timezone utilities
- Trading session detection
- Non-trading time filtering
- Holiday calendar integration
"""

import threading
from datetime import datetime, time, timedelta
from typing import Any

import pandas as pd

from config.runtime_env import env_flag, env_text
from utils.logger import get_logger

log = get_logger(__name__)

# China A-share trading hours
TRADING_MORNING_START = time(9, 30)
TRADING_MORNING_END = time(11, 30)
TRADING_AFTERNOON_START = time(13, 0)
TRADING_AFTERNOON_END = time(15, 0)

# Timezone
ASIA_SHANGHAI = "Asia/Shanghai"


class TradingSessionChecker:
    """Check if a timestamp is within trading hours."""
    
    def __init__(
        self,
        morning_start: time = TRADING_MORNING_START,
        morning_end: time = TRADING_MORNING_END,
        afternoon_start: time = TRADING_AFTERNOON_START,
        afternoon_end: time = TRADING_AFTERNOON_END,
        timezone: str = ASIA_SHANGHAI,
    ):
        self.morning_start = morning_start
        self.morning_end = morning_end
        self.afternoon_start = afternoon_start
        self.afternoon_end = afternoon_end
        self.timezone = timezone
        self._holiday_cache: dict[int, bool] = {}
        self._lock = threading.RLock()
    
    def is_trading_time(self, ts: datetime) -> bool:
        """Check if timestamp is within trading hours."""
        if ts is None:
            return False
        
        # Check if weekend
        if ts.weekday() >= 5:  # Saturday = 5, Sunday = 6
            return False
        
        # Check if holiday (simplified - in production would use full calendar)
        if self.is_holiday(ts):
            return False
        
        # Check time of day
        t = ts.time()
        
        # Morning session
        if self.morning_start <= t <= self.morning_end:
            return True
        
        # Afternoon session
        if self.afternoon_start <= t <= self.afternoon_end:
            return True
        
        return False
    
    def is_holiday(self, ts: datetime) -> bool:
        """Check if date is a holiday (simplified implementation).
        
        FIX 2026-02-26: In production, this should use a proper holiday calendar
        like pandas_market_calendars or similar.
        """
        date_key = ts.date().toordinal()
        
        with self._lock:
            if date_key in self._holiday_cache:
                return self._holiday_cache[date_key]
        
        # Simplified: Check for major Chinese holidays
        # In production, use proper holiday calendar
        month_day = (ts.month, ts.day)
        
        # Major Chinese holidays (approximate)
        holidays = {
            (1, 1),  # New Year
            (5, 1),  # Labor Day
            (10, 1),  # National Day
        }
        
        # Add some variable holidays (simplified)
        # Spring Festival (varies, typically late Jan/early Feb)
        # Mid-Autumn Festival (varies, typically Sept)
        
        is_holiday = month_day in holidays
        
        with self._lock:
            self._holiday_cache[date_key] = is_holiday
        
        return is_holiday
    
    def is_market_open(self, ts: datetime) -> bool:
        """Check if market is currently open (during trading hours)."""
        return self.is_trading_time(ts)
    
    def next_trading_time(self, ts: datetime) -> datetime:
        """Get the next trading time from given timestamp."""
        if ts is None:
            return datetime.now()
        
        # Start from next minute
        candidate = ts.replace(second=0, microsecond=0) + timedelta(minutes=1)
        
        # Search forward for next trading time
        max_iterations = 7 * 24 * 60  # One week max
        for _ in range(max_iterations):
            if self.is_trading_time(candidate):
                return candidate
            candidate += timedelta(minutes=1)
        
        # Fallback
        return candidate
    
    def previous_trading_time(self, ts: datetime) -> datetime:
        """Get the previous trading time from given timestamp."""
        if ts is None:
            return datetime.now()
        
        # Start from previous minute
        candidate = ts.replace(second=0, microsecond=0) - timedelta(minutes=1)
        
        # Search backward for previous trading time
        max_iterations = 7 * 24 * 60  # One week max
        for _ in range(max_iterations):
            if self.is_trading_time(candidate):
                return candidate
            candidate -= timedelta(minutes=1)
        
        # Fallback
        return candidate
    
    def filter_trading_hours(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter DataFrame to only include trading hours.
        
        Args:
            df: DataFrame with DatetimeIndex
        
        Returns:
            Filtered DataFrame with only trading hours
        """
        if df is None or df.empty:
            return df
        
        if not isinstance(df.index, pd.DatetimeIndex):
            log.warning("DataFrame index is not DatetimeIndex, skipping trading hours filter")
            return df
        
        mask = df.index.to_series().apply(self.is_trading_time)
        return df[mask]
    
    def add_session_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add trading session indicator columns to DataFrame.
        
        Args:
            df: DataFrame with DatetimeIndex
        
        Returns:
            DataFrame with added columns:
            - is_trading: bool, within trading hours
            - session: str, 'morning', 'afternoon', or 'closed'
        """
        if df is None or df.empty:
            return df
        
        if not isinstance(df.index, pd.DatetimeIndex):
            return df
        
        df = df.copy()
        
        def get_session(ts: datetime) -> str:
            if not self.is_trading_time(ts):
                return "closed"
            t = ts.time()
            if self.morning_start <= t <= self.morning_end:
                return "morning"
            if self.afternoon_start <= t <= self.afternoon_end:
                return "afternoon"
            return "closed"
        
        df["is_trading"] = df.index.to_series().apply(self.is_trading_time)
        df["session"] = df.index.to_series().apply(get_session)
        
        return df


class TimezoneConverter:
    """Handle timezone conversions for market data."""
    
    def __init__(self, target_timezone: str = ASIA_SHANGHAI, force_naive: bool = True):
        self.target_timezone = target_timezone
        self.force_naive = force_naive
        self._tz_cache: dict[str, Any] = {}
        self._lock = threading.RLock()
    
    def _get_timezone(self, tz: str) -> Any:
        """Get timezone object with caching."""
        with self._lock:
            if tz not in self._tz_cache:
                try:
                    import zoneinfo
                    self._tz_cache[tz] = zoneinfo.ZoneInfo(tz)
                except ImportError:
                    # Python < 3.9, try pytz
                    try:
                        import pytz
                        self._tz_cache[tz] = pytz.timezone(tz)
                    except ImportError:
                        log.warning("No timezone support available, using naive datetimes")
                        self._tz_cache[tz] = None
            return self._tz_cache[tz]
    
    def to_shanghai(self, ts: datetime) -> datetime:
        """Convert timestamp to Shanghai timezone."""
        if ts is None:
            return datetime.now()
        
        if ts.tzinfo is None:
            # Already naive, assume Shanghai
            return ts
        
        try:
            shanghai_tz = self._get_timezone(self.target_timezone)
            if shanghai_tz is None:
                # No timezone support, convert to naive
                return ts.replace(tzinfo=None)
            
            ts_shanghai = ts.astimezone(shanghai_tz)
            
            if self.force_naive:
                return ts_shanghai.replace(tzinfo=None)
            
            return ts_shanghai
            
        except Exception as e:
            log.debug("Timezone conversion failed: %s", e)
            # Fallback to naive
            return ts.replace(tzinfo=None) if ts.tzinfo else ts
    
    def to_utc(self, ts: datetime) -> datetime:
        """Convert timestamp to UTC."""
        if ts is None:
            return datetime.utcnow()
        
        try:
            if ts.tzinfo is None:
                # Assume Shanghai
                shanghai_tz = self._get_timezone(self.target_timezone)
                if shanghai_tz:
                    ts = shanghai_tz.localize(ts) if hasattr(shanghai_tz, 'localize') else ts.replace(tzinfo=shanghai_tz)
            
            utc_tz = self._get_timezone("UTC")
            if utc_tz is None:
                # No timezone support, return naive UTC
                return ts
            
            ts_utc = ts.astimezone(utc_tz)
            
            if self.force_naive:
                return ts_utc.replace(tzinfo=None)
            
            return ts_utc
            
        except Exception as e:
            log.debug("UTC conversion failed: %s", e)
            return ts
    
    def localize_naive(self, ts: datetime, assume_shanghai: bool = True) -> datetime:
        """Add timezone info to naive timestamp."""
        if ts is None or ts.tzinfo is not None:
            return ts
        
        tz = self.target_timezone if assume_shanghai else "UTC"
        tz_obj = self._get_timezone(tz)
        
        if tz_obj is None:
            return ts
        
        try:
            if hasattr(tz_obj, 'localize'):
                return tz_obj.localize(ts)
            else:
                return ts.replace(tzinfo=tz_obj)
        except Exception as e:
            log.debug("Localization failed: %s", e)
            return ts
    
    def ensure_datetime_index(
        self,
        df: pd.DataFrame,
        column: str = "datetime",
        set_index: bool = True,
    ) -> pd.DataFrame:
        """Ensure DataFrame has proper DatetimeIndex in Shanghai time.
        
        FIX 2026-02-26: Handles various input formats and timezones.
        
        Args:
            df: Input DataFrame
            column: Column name containing datetime strings/values
            set_index: Whether to set the datetime as index
        
        Returns:
            DataFrame with proper DatetimeIndex
        """
        if df is None or df.empty:
            return df
        
        df = df.copy()
        
        # Check if index is already DatetimeIndex
        if isinstance(df.index, pd.DatetimeIndex):
            # Convert to Shanghai time
            if df.index.tz is not None:
                df.index = df.index.tz_convert(self.target_timezone)
                if self.force_naive:
                    df.index = df.index.tz_localize(None)
            return df
        
        # Check if column exists
        if column not in df.columns:
            # Try common datetime column names
            for col in ["timestamp", "time", "date", "datetime"]:
                if col in df.columns:
                    column = col
                    break
            else:
                log.warning("No datetime column found, using index")
                return df
        
        # Convert column to datetime
        try:
            df[column] = pd.to_datetime(df[column], errors="coerce")
        except Exception as e:
            log.warning("Datetime conversion failed: %s", e)
            return df
        
        # Drop rows with NaT
        df = df.dropna(subset=[column])
        
        if df.empty:
            return df
        
        # Convert to Shanghai time
        if hasattr(df[column].dt, 'tz') and df[column].dt.tz is not None:
            df[column] = df[column].dt.tz_convert(self.target_timezone)
            if self.force_naive:
                df[column] = df[column].dt.tz_localize(None)
        
        # Set as index
        if set_index:
            df = df.set_index(column)
        
        return df


# Global instances
_session_checker: TradingSessionChecker | None = None
_timezone_converter: TimezoneConverter | None = None
_instances_lock = threading.Lock()


def get_session_checker() -> TradingSessionChecker:
    """Get or create global session checker."""
    global _session_checker
    with _instances_lock:
        if _session_checker is None:
            _session_checker = TradingSessionChecker(
                morning_start=_parse_time(env_text("TRADING_MORNING_START", "09:30")),
                morning_end=_parse_time(env_text("TRADING_MORNING_END", "11:30")),
                afternoon_start=_parse_time(env_text("TRADING_AFTERNOON_START", "13:00")),
                afternoon_end=_parse_time(env_text("TRADING_AFTERNOON_END", "15:00")),
            )
        return _session_checker


def get_timezone_converter() -> TimezoneConverter:
    """Get or create global timezone converter."""
    global _timezone_converter
    with _instances_lock:
        if _timezone_converter is None:
            _timezone_converter = TimezoneConverter(
                target_timezone=env_text("TRADING_TIMEZONE", ASIA_SHANGHAI),
                force_naive=env_flag("TRADING_FORCE_NAIVE", "1"),
            )
        return _timezone_converter


def _parse_time(time_str: str) -> time:
    """Parse time string to time object."""
    try:
        parts = time_str.strip().split(":")
        return time(int(parts[0]), int(parts[1]))
    except (ValueError, IndexError):
        return TRADING_MORNING_START


def is_trading_time(ts: datetime | None = None) -> bool:
    """Check if current or given time is within trading hours."""
    if ts is None:
        ts = datetime.now()
    return get_session_checker().is_trading_time(ts)


def filter_trading_hours(df: pd.DataFrame) -> pd.DataFrame:
    """Filter DataFrame to only include trading hours."""
    return get_session_checker().filter_trading_hours(df)


def ensure_shanghai_datetime(
    df: pd.DataFrame,
    column: str = "datetime",
) -> pd.DataFrame:
    """Ensure DataFrame has proper Shanghai-time DatetimeIndex."""
    return get_timezone_converter().ensure_datetime_index(df, column)


def to_shanghai_naive(ts: datetime) -> datetime:
    """Convert timestamp to naive Shanghai time."""
    return get_timezone_converter().to_shanghai(ts)


def reset_timezone_utils() -> None:
    """Reset global utilities (for testing)."""
    global _session_checker, _timezone_converter
    with _instances_lock:
        _session_checker = None
        _timezone_converter = None
