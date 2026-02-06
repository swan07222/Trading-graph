"""
Data Validation Module
Score Target: 10/10

Comprehensive data validation for all inputs.
"""
import re
from datetime import datetime, date, timedelta
from typing import List, Dict, Optional, Tuple, Any, Union
from dataclasses import dataclass
import pandas as pd
import numpy as np

from core.constants import EXCHANGES, get_exchange, is_trading_day
from core.exceptions import DataValidationError
from utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class ValidationResult:
    """Result of validation"""
    valid: bool
    errors: List[str]
    warnings: List[str]
    data: Any = None
    
    def __bool__(self):
        return self.valid
    
    def raise_if_invalid(self):
        """Raise exception if invalid"""
        if not self.valid:
            raise DataValidationError(
                message="; ".join(self.errors),
                details={'errors': self.errors, 'warnings': self.warnings}
            )


class StockCodeValidator:
    """Validate stock codes"""
    
    # Valid patterns for different markets
    PATTERNS = {
        'a_share': r'^[036]\d{5}$',
        'hk': r'^\d{5}$',
        'us': r'^[A-Z]{1,5}$',
    }
    
    @classmethod
    def validate(cls, code: str, market: str = 'a_share') -> ValidationResult:
        """
        Validate stock code.

        CN (a_share):
        - Accepts SSE/SZSE/BSE via get_exchange()

        HK:
        - 5 digits

        US:
        - allow BRK.B / BF.B style tickers
        """
        errors = []
        warnings = []

        if not code:
            errors.append("Stock code is required")
            return ValidationResult(False, errors, warnings)

        raw = str(code).strip()
        code_u = raw.strip().upper()

        # Strip common prefixes/suffixes (CN)
        for prefix in ['SH', 'SZ', 'BJ', 'SH.', 'SZ.', 'BJ.', 'sh', 'sz', 'bj']:
            if code_u.startswith(prefix.upper()):
                code_u = code_u[len(prefix):]
                break
        for suffix in ['.SS', '.SZ', '.BJ', '.ss', '.sz', '.bj']:
            if code_u.endswith(suffix.upper()):
                code_u = code_u[:-len(suffix)]
                break

        if market == "a_share":
            digits = "".join(ch for ch in code_u if ch.isdigit())
            if not digits:
                errors.append(f"Invalid stock code format: {raw}")
                return ValidationResult(False, errors, warnings)

            code6 = digits.zfill(6)
            exchange = get_exchange(code6)
            if exchange == "UNKNOWN":
                errors.append(f"Unknown/invalid A-share code: {code6}")
                return ValidationResult(False, errors, warnings)

            return ValidationResult(True, errors, warnings, code6)

        if market == "hk":
            if not re.fullmatch(r"^\d{5}$", code_u):
                errors.append(f"Invalid HK stock code format: {raw}")
                return ValidationResult(False, errors, warnings)
            return ValidationResult(True, errors, warnings, code_u)

        if market == "us":
            # allow BRK.B etc.
            if not re.fullmatch(r"^[A-Z0-9]{1,6}(\.[A-Z])?$", code_u):
                errors.append(f"Invalid US ticker format: {raw}")
                return ValidationResult(False, errors, warnings)
            return ValidationResult(True, errors, warnings, code_u)

        errors.append(f"Unknown market: {market}")
        return ValidationResult(False, errors, warnings)
    
    @classmethod
    def validate_many(cls, codes: List[str], market: str = 'a_share') -> ValidationResult:
        """Validate multiple stock codes"""
        errors = []
        warnings = []
        valid_codes = []
        
        for code in codes:
            result = cls.validate(code, market)
            if result.valid:
                valid_codes.append(result.data)
            else:
                errors.extend(result.errors)
            warnings.extend(result.warnings)
        
        if not valid_codes:
            errors.append("No valid stock codes")
            return ValidationResult(False, errors, warnings)
        
        return ValidationResult(True, errors, warnings, valid_codes)


class OHLCVValidator:
    """Validate OHLCV data"""
    
    REQUIRED_COLUMNS = ['open', 'high', 'low', 'close', 'volume']
    
    @classmethod
    def validate(
        cls, 
        df: pd.DataFrame,
        min_rows: int = 10,
        check_prices: bool = True,
        check_volume: bool = True,
        check_dates: bool = True,
        fix_errors: bool = True
    ) -> ValidationResult:
        """
        Validate OHLCV DataFrame
        
        Args:
            df: DataFrame with OHLCV data
            min_rows: Minimum required rows
            check_prices: Validate price relationships
            check_volume: Validate volume
            check_dates: Validate date index
            fix_errors: Attempt to fix minor errors
        
        Returns:
            ValidationResult with cleaned DataFrame
        """
        errors = []
        warnings = []
        
        if df is None or df.empty:
            errors.append("DataFrame is empty")
            return ValidationResult(False, errors, warnings)
        
        df = df.copy()
        
        # Check required columns
        missing = set(cls.REQUIRED_COLUMNS) - set(df.columns)
        if missing:
            errors.append(f"Missing columns: {missing}")
            return ValidationResult(False, errors, warnings)
        
        # Check minimum rows
        if len(df) < min_rows:
            errors.append(f"Insufficient data: {len(df)} rows (min: {min_rows})")
            return ValidationResult(False, errors, warnings)
        
        # Check index is datetime
        if check_dates:
            if not isinstance(df.index, pd.DatetimeIndex):
                try:
                    df.index = pd.to_datetime(df.index)
                except Exception:
                    errors.append("Index is not datetime")
                    return ValidationResult(False, errors, warnings)
            
            # Check for duplicates
            if df.index.duplicated().any():
                warnings.append("Duplicate dates found")
                if fix_errors:
                    df = df[~df.index.duplicated(keep='last')]
            
            # Check order
            if not df.index.is_monotonic_increasing:
                warnings.append("Dates not sorted")
                if fix_errors:
                    df = df.sort_index()
        
        # Convert to numeric
        for col in cls.REQUIRED_COLUMNS:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Check for NaN
        nan_counts = df[cls.REQUIRED_COLUMNS].isna().sum()
        if nan_counts.any():
            warnings.append(f"NaN values: {nan_counts.to_dict()}")
            if fix_errors:
                df = df.dropna(subset=cls.REQUIRED_COLUMNS)
        
        # Validate prices
        if check_prices:
            # Check positive
            for col in ['open', 'high', 'low', 'close']:
                invalid = df[col] <= 0
                if invalid.any():
                    warnings.append(f"Non-positive {col}: {invalid.sum()} rows")
                    if fix_errors:
                        df = df[~invalid]
            
            # Check high >= low
            invalid = df['high'] < df['low']
            if invalid.any():
                warnings.append(f"High < Low: {invalid.sum()} rows")
                if fix_errors:
                    df = df[~invalid]
            
            # Check high >= open, close
            invalid = (df['high'] < df['open']) | (df['high'] < df['close'])
            if invalid.any():
                warnings.append(f"High not highest: {invalid.sum()} rows")
                if fix_errors:
                    df.loc[invalid, 'high'] = df.loc[invalid, ['open', 'high', 'low', 'close']].max(axis=1)
            
            # Check low <= open, close
            invalid = (df['low'] > df['open']) | (df['low'] > df['close'])
            if invalid.any():
                warnings.append(f"Low not lowest: {invalid.sum()} rows")
                if fix_errors:
                    df.loc[invalid, 'low'] = df.loc[invalid, ['open', 'high', 'low', 'close']].min(axis=1)
        
        # Validate volume
        if check_volume:
            invalid = df['volume'] < 0
            if invalid.any():
                warnings.append(f"Negative volume: {invalid.sum()} rows")
                if fix_errors:
                    df = df[~invalid]
            
            # Check for zero volume (might be valid for suspended stocks)
            zero_vol = df['volume'] == 0
            if zero_vol.sum() > len(df) * 0.1:
                warnings.append(f"Many zero volume days: {zero_vol.sum()}")
        
        # Final check
        if len(df) < min_rows:
            errors.append(f"Insufficient valid data: {len(df)} rows")
            return ValidationResult(False, errors, warnings)
        
        return ValidationResult(True, errors, warnings, df)


class FeatureValidator:
    """Validate feature data"""
    
    @classmethod
    def validate(
        cls,
        df: pd.DataFrame,
        feature_cols: List[str],
        max_nan_pct: float = 0.1,
        max_inf_pct: float = 0.01,
        fix_errors: bool = True
    ) -> ValidationResult:
        """
        Validate feature DataFrame
        """
        errors = []
        warnings = []
        
        if df is None or df.empty:
            errors.append("DataFrame is empty")
            return ValidationResult(False, errors, warnings)
        
        df = df.copy()
        
        # Check columns exist
        missing = set(feature_cols) - set(df.columns)
        if missing:
            errors.append(f"Missing feature columns: {missing}")
            return ValidationResult(False, errors, warnings)
        
        features = df[feature_cols]
        
        # Check NaN
        nan_pct = features.isna().sum() / len(features)
        high_nan = nan_pct[nan_pct > max_nan_pct]
        if not high_nan.empty:
            warnings.append(f"High NaN columns: {high_nan.to_dict()}")
        
        # Check Inf
        inf_count = np.isinf(features.values).sum(axis=0)
        inf_pct = inf_count / len(features)
        high_inf = pd.Series(inf_pct, index=feature_cols)[inf_pct > max_inf_pct]
        if not high_inf.empty:
            warnings.append(f"High Inf columns: {high_inf.to_dict()}")
        
        # Fix errors
        if fix_errors:
            # Replace inf with nan, then fill
            for col in feature_cols:
                df[col] = df[col].replace([np.inf, -np.inf], np.nan)
            
            # Drop rows with too many NaN
            nan_per_row = df[feature_cols].isna().sum(axis=1)
            max_nan = len(feature_cols) * 0.5
            df = df[nan_per_row <= max_nan]
            
            # Fill remaining NaN with forward fill then 0
            df[feature_cols] = df[feature_cols].ffill().fillna(0)
        
        if len(df) < 10:
            errors.append("Insufficient valid data after cleaning")
            return ValidationResult(False, errors, warnings)
        
        return ValidationResult(True, errors, warnings, df)


class OrderValidator:
    """Validate trading orders"""
    
    @classmethod
    def validate(
        cls,
        symbol: str,
        side: str,
        quantity: int,
        price: float = None,
        order_type: str = 'limit'
    ) -> ValidationResult:
        """
        Validate order parameters
        """
        errors = []
        warnings = []
        
        # Validate symbol
        code_result = StockCodeValidator.validate(symbol)
        if not code_result.valid:
            errors.extend(code_result.errors)
            return ValidationResult(False, errors, warnings)
        
        symbol = code_result.data
        
        # Validate side
        side = side.lower()
        if side not in ['buy', 'sell']:
            errors.append(f"Invalid order side: {side}")
            return ValidationResult(False, errors, warnings)
        
        # Validate quantity
        if quantity <= 0:
            errors.append("Quantity must be positive")
            return ValidationResult(False, errors, warnings)
        
        # Check lot size
        from core.constants import get_lot_size
        lot_size = get_lot_size(symbol)
        
        if quantity % lot_size != 0:
            errors.append(f"Quantity must be multiple of {lot_size}")
            return ValidationResult(False, errors, warnings)
        
        # Validate price
        if order_type == 'limit':
            if price is None or price <= 0:
                errors.append("Limit orders require positive price")
                return ValidationResult(False, errors, warnings)
            
            # Check price precision (2 decimal places)
            if round(price, 2) != price:
                warnings.append(f"Price rounded to 2 decimals: {round(price, 2)}")
                price = round(price, 2)
        
        return ValidationResult(True, errors, warnings, {
            'symbol': symbol,
            'side': side,
            'quantity': quantity,
            'price': price,
            'order_type': order_type
        })


class ConfigValidator:
    """Validate configuration parameters"""
    
    @classmethod
    def validate_risk_config(cls, config: Dict) -> ValidationResult:
        """Validate risk configuration"""
        errors = []
        warnings = []
        
        # Required fields
        required = [
            'max_position_pct', 'max_daily_loss_pct', 
            'max_positions', 'risk_per_trade_pct'
        ]
        
        for field in required:
            if field not in config:
                errors.append(f"Missing required field: {field}")
        
        if errors:
            return ValidationResult(False, errors, warnings)
        
        # Validate ranges
        if not 0 < config['max_position_pct'] <= 100:
            errors.append("max_position_pct must be between 0 and 100")
        
        if not 0 < config['max_daily_loss_pct'] <= 100:
            errors.append("max_daily_loss_pct must be between 0 and 100")
        
        if not 1 <= config['max_positions'] <= 100:
            errors.append("max_positions must be between 1 and 100")
        
        if not 0 < config['risk_per_trade_pct'] <= 10:
            errors.append("risk_per_trade_pct must be between 0 and 10")
        
        # Warnings for aggressive settings
        if config['max_position_pct'] > 25:
            warnings.append("High max_position_pct - consider reducing")
        
        if config['max_daily_loss_pct'] > 5:
            warnings.append("High max_daily_loss_pct - consider reducing")
        
        return ValidationResult(len(errors) == 0, errors, warnings, config)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def validate_stock_code(code: str) -> str:
    """Validate and return cleaned stock code"""
    result = StockCodeValidator.validate(code)
    result.raise_if_invalid()
    return result.data


def validate_ohlcv(df: pd.DataFrame, min_rows: int = 10) -> pd.DataFrame:
    """Validate and return cleaned OHLCV DataFrame"""
    result = OHLCVValidator.validate(df, min_rows=min_rows)
    result.raise_if_invalid()
    return result.data


def validate_features(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    """Validate and return cleaned feature DataFrame"""
    result = FeatureValidator.validate(df, feature_cols)
    result.raise_if_invalid()
    return result.data