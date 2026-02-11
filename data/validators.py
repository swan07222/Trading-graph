
import re
from datetime import datetime, date, timedelta
from typing import List, Dict, Optional, Tuple, Any, Union
from dataclasses import dataclass
import pandas as pd
import numpy as np

from core.constants import (
    EXCHANGES,
    get_exchange,
    get_lot_size,
    is_trading_day,
)
from core.exceptions import DataValidationError
from utils.logger import get_logger

log = get_logger(__name__)


# =============================================================================
# VALIDATION RESULT
# =============================================================================


@dataclass
class ValidationResult:
    """Result of validation."""

    valid: bool
    errors: List[str]
    warnings: List[str]
    data: Any = None

    def __bool__(self) -> bool:
        return self.valid

    def raise_if_invalid(self) -> None:
        """Raise DataValidationError if invalid."""
        if not self.valid:
            raise DataValidationError(
                message="; ".join(self.errors),
                details={"errors": self.errors, "warnings": self.warnings},
            )


# =============================================================================
# STOCK CODE VALIDATOR
# =============================================================================


class StockCodeValidator:
    """Validate stock codes for A-share, HK, and US markets."""

    @classmethod
    def validate(
        cls, code: str, market: str = "a_share"
    ) -> ValidationResult:
        """
        Validate and normalise a stock code.

        CN (a_share):
            Strips common prefixes/suffixes, zero-pads to 6 digits,
            and verifies the exchange via ``get_exchange()``.

        HK:
            5 digits.

        US:
            1–6 uppercase alphanumerics, optionally followed by ``.X``
            (e.g. BRK.B, BF.B).

        Returns:
            ValidationResult with cleaned code in ``data`` on success.
        """
        errors: List[str] = []
        warnings: List[str] = []

        if not code:
            errors.append("Stock code is required")
            return ValidationResult(False, errors, warnings)

        raw = str(code).strip()
        cleaned = raw.upper()

        # ---- A-share ------------------------------------------------
        if market == "a_share":
            # Strip common exchange prefixes
            for prefix in ("SH.", "SZ.", "BJ.", "SH", "SZ", "BJ"):
                if cleaned.startswith(prefix):
                    cleaned = cleaned[len(prefix) :]
                    break

            # Strip common exchange suffixes
            for suffix in (".SS", ".SZ", ".BJ"):
                if cleaned.endswith(suffix):
                    cleaned = cleaned[: -len(suffix)]
                    break

            digits = "".join(ch for ch in cleaned if ch.isdigit())
            if not digits:
                errors.append(f"Invalid A-share code (no digits): {raw}")
                return ValidationResult(False, errors, warnings)

            code6 = digits.zfill(6)

            # Validate that zero-padding didn't create a bogus code
            # e.g. "12345" → "012345" which starts with "01" — not valid.
            exchange = get_exchange(code6)
            if exchange == "UNKNOWN":
                errors.append(f"Unknown/invalid A-share code: {code6}")
                return ValidationResult(False, errors, warnings)

            return ValidationResult(True, errors, warnings, code6)

        # ---- HK ------------------------------------------------------
        if market == "hk":
            if not re.fullmatch(r"\d{5}", cleaned):
                errors.append(f"Invalid HK stock code (need 5 digits): {raw}")
                return ValidationResult(False, errors, warnings)
            return ValidationResult(True, errors, warnings, cleaned)

        # ---- US ------------------------------------------------------
        if market == "us":
            # Allow BRK.B, BF.B style tickers
            if not re.fullmatch(r"[A-Z0-9]{1,6}(\.[A-Z])?", cleaned):
                errors.append(f"Invalid US ticker format: {raw}")
                return ValidationResult(False, errors, warnings)
            return ValidationResult(True, errors, warnings, cleaned)

        errors.append(f"Unknown market: {market}")
        return ValidationResult(False, errors, warnings)

    @classmethod
    def validate_many(
        cls,
        codes: List[str],
        market: str = "a_share",
    ) -> ValidationResult:
        """Validate multiple stock codes, returning all valid ones."""
        errors: List[str] = []
        warnings: List[str] = []
        valid_codes: List[str] = []

        if not codes:
            errors.append("No stock codes provided")
            return ValidationResult(False, errors, warnings)

        for code in codes:
            result = cls.validate(code, market)
            if result.valid:
                valid_codes.append(result.data)
            else:
                errors.extend(result.errors)
            warnings.extend(result.warnings)

        if not valid_codes:
            errors.append("No valid stock codes found")
            return ValidationResult(False, errors, warnings)

        # Deduplicate while preserving order
        seen: set = set()
        deduped: List[str] = []
        for c in valid_codes:
            if c not in seen:
                seen.add(c)
                deduped.append(c)

        return ValidationResult(True, errors, warnings, deduped)


# =============================================================================
# DATE RANGE VALIDATOR
# =============================================================================


class DateRangeValidator:
    """Validate date ranges for data queries."""

    @classmethod
    def validate(
        cls,
        start_date: Any = None,
        end_date: Any = None,
        max_range_days: int = 3650,
        allow_future: bool = False,
    ) -> ValidationResult:
        """
        Validate and normalise a date range.

        Args:
            start_date: Start date (str, date, or datetime).
            end_date:   End date (str, date, or datetime). Defaults to today.
            max_range_days: Maximum span in calendar days.
            allow_future:   Whether end_date may be in the future.

        Returns:
            ValidationResult with ``data = (start_date, end_date)`` as
            ``date`` objects on success.
        """
        errors: List[str] = []
        warnings: List[str] = []
        today = date.today()

        # Parse end_date
        end = cls._parse_date(end_date, "end_date", errors)
        if end is None:
            end = today

        # Parse start_date
        start = cls._parse_date(start_date, "start_date", errors)
        if errors:
            return ValidationResult(False, errors, warnings)

        if start is None:
            errors.append("start_date is required")
            return ValidationResult(False, errors, warnings)

        # Logical checks
        if start > end:
            errors.append(
                f"start_date ({start}) is after end_date ({end})"
            )
            return ValidationResult(False, errors, warnings)

        if not allow_future and end > today:
            warnings.append(f"end_date ({end}) is in the future; clamping to today")
            end = today

        span = (end - start).days
        if span > max_range_days:
            errors.append(
                f"Date range too large: {span} days (max {max_range_days})"
            )
            return ValidationResult(False, errors, warnings)

        if span == 0:
            warnings.append("start_date equals end_date (single day)")

        return ValidationResult(True, errors, warnings, (start, end))

    @staticmethod
    def _parse_date(
        value: Any, field_name: str, errors: List[str]
    ) -> Optional[date]:
        """Parse a value into a ``date``, appending to *errors* on failure."""
        if value is None:
            return None
        if isinstance(value, datetime):
            return value.date()
        if isinstance(value, date):
            return value
        if isinstance(value, str):
            for fmt in ("%Y-%m-%d", "%Y%m%d", "%Y/%m/%d"):
                try:
                    return datetime.strptime(value, fmt).date()
                except ValueError:
                    continue
            errors.append(f"Cannot parse {field_name}: {value!r}")
            return None
        errors.append(
            f"Invalid type for {field_name}: {type(value).__name__}"
        )
        return None


# =============================================================================
# OHLCV VALIDATOR
# =============================================================================


class OHLCVValidator:
    """Validate OHLCV DataFrames."""

    REQUIRED_COLUMNS = ["open", "high", "low", "close", "volume"]

    @classmethod
    def validate(
        cls,
        df: pd.DataFrame,
        min_rows: int = 10,
        check_prices: bool = True,
        check_volume: bool = True,
        check_dates: bool = True,
        fix_errors: bool = True,
    ) -> ValidationResult:
        """
        Validate an OHLCV DataFrame.

        When ``fix_errors=True`` the method attempts to repair minor
        inconsistencies (duplicate dates, unsorted index, high/low
        violations) and drops irrecoverable rows (negative prices,
        negative volume, NaN in required columns).

        Args:
            df:           DataFrame with at least open/high/low/close/volume.
            min_rows:     Minimum acceptable row count.
            check_prices: Validate OHLC relationships.
            check_volume: Validate volume values.
            check_dates:  Validate datetime index.
            fix_errors:   Attempt to fix minor issues.

        Returns:
            ValidationResult with cleaned DataFrame in ``data``.
        """
        errors: List[str] = []
        warnings: List[str] = []

        if df is None or df.empty:
            errors.append("DataFrame is empty")
            return ValidationResult(False, errors, warnings)

        df = df.copy()

        # Normalise column names to lowercase
        df.columns = [c.lower().strip() for c in df.columns]

        # ---- Required columns ----------------------------------------
        missing = set(cls.REQUIRED_COLUMNS) - set(df.columns)
        if missing:
            errors.append(f"Missing columns: {sorted(missing)}")
            return ValidationResult(False, errors, warnings)

        # ---- Minimum rows --------------------------------------------
        if len(df) < min_rows:
            errors.append(
                f"Insufficient data: {len(df)} rows (min: {min_rows})"
            )
            return ValidationResult(False, errors, warnings)

        # ---- Date index ----------------------------------------------
        if check_dates:
            if not isinstance(df.index, pd.DatetimeIndex):
                try:
                    df.index = pd.to_datetime(df.index)
                except Exception:
                    errors.append("Index cannot be converted to datetime")
                    return ValidationResult(False, errors, warnings)

            dup_count = df.index.duplicated().sum()
            if dup_count:
                warnings.append(f"Duplicate dates: {dup_count}")
                if fix_errors:
                    df = df[~df.index.duplicated(keep="last")]

            if not df.index.is_monotonic_increasing:
                warnings.append("Dates not sorted")
                if fix_errors:
                    df = df.sort_index()

        # ---- Convert to numeric --------------------------------------
        for col in cls.REQUIRED_COLUMNS:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # ---- Drop NaN rows in required columns ----------------------
        nan_counts = df[cls.REQUIRED_COLUMNS].isna().sum()
        total_nan = int(nan_counts.sum())
        if total_nan:
            warnings.append(f"NaN values found: {nan_counts.to_dict()}")
            if fix_errors:
                df = df.dropna(subset=cls.REQUIRED_COLUMNS)

        # ---- Price validation (single-pass) --------------------------
        if check_prices and len(df) > 0:
            df, price_warnings = cls._validate_prices(df, fix_errors)
            warnings.extend(price_warnings)

        # ---- Volume validation ---------------------------------------
        if check_volume and len(df) > 0:
            neg_vol = df["volume"] < 0
            if neg_vol.any():
                warnings.append(f"Negative volume: {neg_vol.sum()} rows")
                if fix_errors:
                    df = df[~neg_vol]

            if len(df) > 0:
                zero_vol = (df["volume"] == 0).sum()
                if zero_vol > len(df) * 0.1:
                    warnings.append(
                        f"Many zero-volume days: {zero_vol}/{len(df)}"
                    )

        # ---- Final row count check -----------------------------------
        if len(df) < min_rows:
            errors.append(
                f"Insufficient valid data after cleaning: "
                f"{len(df)} rows (min: {min_rows})"
            )
            return ValidationResult(False, errors, warnings)

        return ValidationResult(True, errors, warnings, df)

    @classmethod
    def _validate_prices(
        cls,
        df: pd.DataFrame,
        fix_errors: bool,
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Validate OHLC price relationships in a single pass.

        Builds a composite mask of rows with irrecoverable issues
        (non-positive prices), drops them, then fixes repairable
        issues (high/low violations) on the surviving rows.

        Returns:
            (cleaned_df, list_of_warning_strings)
        """
        warnings: List[str] = []
        price_cols = ["open", "high", "low", "close"]

        # ---- Drop rows with non-positive prices ---------------------
        non_positive = (df[price_cols] <= 0).any(axis=1)
        if non_positive.any():
            count = int(non_positive.sum())
            warnings.append(f"Non-positive prices: {count} rows")
            if fix_errors:
                df = df[~non_positive]

        if df.empty:
            return df, warnings

        # ---- Drop rows where high < low (irrecoverable) -------------
        bad_hl = df["high"] < df["low"]
        if bad_hl.any():
            count = int(bad_hl.sum())
            warnings.append(f"High < Low: {count} rows")
            if fix_errors:
                df = df[~bad_hl]

        if df.empty:
            return df, warnings

        # ---- Fix high not being the highest (repairable) ------------
        high_violation = (df["high"] < df["open"]) | (
            df["high"] < df["close"]
        )
        if high_violation.any():
            count = int(high_violation.sum())
            warnings.append(f"High not highest: {count} rows")
            if fix_errors:
                df.loc[high_violation, "high"] = df.loc[
                    high_violation, price_cols
                ].max(axis=1)

        # ---- Fix low not being the lowest (repairable) --------------
        low_violation = (df["low"] > df["open"]) | (
            df["low"] > df["close"]
        )
        if low_violation.any():
            count = int(low_violation.sum())
            warnings.append(f"Low not lowest: {count} rows")
            if fix_errors:
                df.loc[low_violation, "low"] = df.loc[
                    low_violation, price_cols
                ].min(axis=1)

        return df, warnings


# =============================================================================
# FEATURE VALIDATOR
# =============================================================================


class FeatureValidator:
    """Validate feature DataFrames for ML pipeline."""

    @classmethod
    def validate(
        cls,
        df: pd.DataFrame,
        feature_cols: List[str],
        max_nan_pct: float = 0.1,
        max_inf_pct: float = 0.01,
        fix_errors: bool = True,
    ) -> ValidationResult:
        """
        Validate a feature DataFrame.

        When ``fix_errors=True``:
        - Replaces ±inf with NaN.
        - Drops rows where more than 50 % of features are NaN.
        - Forward-fills remaining NaN (does NOT zero-fill to avoid
          creating false signals for financial features).

        Args:
            df:           DataFrame containing feature columns.
            feature_cols: Names of columns to validate.
            max_nan_pct:  Warn if any column exceeds this NaN fraction.
            max_inf_pct:  Warn if any column exceeds this inf fraction.
            fix_errors:   Attempt to repair.

        Returns:
            ValidationResult with cleaned DataFrame.
        """
        errors: List[str] = []
        warnings: List[str] = []

        if df is None or df.empty:
            errors.append("DataFrame is empty")
            return ValidationResult(False, errors, warnings)

        df = df.copy()

        # ---- Check columns exist ------------------------------------
        missing = set(feature_cols) - set(df.columns)
        if missing:
            errors.append(f"Missing feature columns: {sorted(missing)}")
            return ValidationResult(False, errors, warnings)

        # ---- Ensure numeric dtype before inf check ------------------
        for col in feature_cols:
            if not pd.api.types.is_numeric_dtype(df[col]):
                df[col] = pd.to_numeric(df[col], errors="coerce")

        features = df[feature_cols]

        # ---- NaN report ---------------------------------------------
        nan_pct = features.isna().sum() / max(len(features), 1)
        high_nan = nan_pct[nan_pct > max_nan_pct]
        if not high_nan.empty:
            warnings.append(f"High NaN columns: {high_nan.to_dict()}")

        # ---- Inf report (safe for numeric-only frame) ---------------
        inf_mask = np.isinf(features.values)
        inf_count = inf_mask.sum(axis=0)
        inf_pct = inf_count / max(len(features), 1)
        high_inf_mask = inf_pct > max_inf_pct
        if high_inf_mask.any():
            high_inf = {
                feature_cols[i]: float(inf_pct[i])
                for i in range(len(feature_cols))
                if high_inf_mask[i]
            }
            warnings.append(f"High Inf columns: {high_inf}")

        # ---- Fix errors ---------------------------------------------
        if fix_errors:
            # Replace inf with NaN
            for col in feature_cols:
                df[col] = df[col].replace([np.inf, -np.inf], np.nan)

            # Drop rows where >50 % of features are NaN
            nan_per_row = df[feature_cols].isna().sum(axis=1)
            max_nan_allowed = len(feature_cols) * 0.5
            df = df[nan_per_row <= max_nan_allowed]

            # Forward-fill, then backward-fill (NOT zero-fill)
            df[feature_cols] = df[feature_cols].ffill().bfill()

            # If any NaN remains (e.g., entire column was NaN), warn
            remaining_nan = df[feature_cols].isna().sum()
            still_nan = remaining_nan[remaining_nan > 0]
            if not still_nan.empty:
                warnings.append(
                    f"Unfillable NaN columns (filled with 0): "
                    f"{still_nan.to_dict()}"
                )
                df[feature_cols] = df[feature_cols].fillna(0)

        if len(df) < 10:
            errors.append("Insufficient valid data after cleaning")
            return ValidationResult(False, errors, warnings)

        return ValidationResult(True, errors, warnings, df)


# =============================================================================
# ORDER VALIDATOR
# =============================================================================


class OrderValidator:
    """Validate trading order parameters."""

    @classmethod
    def validate(
        cls,
        symbol: str,
        side: str,
        quantity: int,
        price: float = None,
        order_type: str = "limit",
        market: str = "a_share",
    ) -> ValidationResult:
        """
        Validate order parameters.

        Args:
            symbol:     Stock code.
            side:       ``'buy'`` or ``'sell'``.
            quantity:   Number of shares.
            price:      Required for limit orders.
            order_type: ``'limit'`` or ``'market'``.
            market:     Market for symbol validation.

        Returns:
            ValidationResult with normalised order dict in ``data``.
        """
        errors: List[str] = []
        warnings: List[str] = []

        # ---- Symbol --------------------------------------------------
        code_result = StockCodeValidator.validate(symbol, market=market)
        if not code_result.valid:
            errors.extend(code_result.errors)
            return ValidationResult(False, errors, warnings)
        warnings.extend(code_result.warnings)
        symbol = code_result.data

        # ---- Side ----------------------------------------------------
        side = str(side).lower().strip()
        if side not in ("buy", "sell"):
            errors.append(f"Invalid order side: {side!r} (expected buy/sell)")
            return ValidationResult(False, errors, warnings)

        # ---- Order type ----------------------------------------------
        order_type = str(order_type).lower().strip()
        if order_type not in ("limit", "market"):
            errors.append(
                f"Invalid order type: {order_type!r} (expected limit/market)"
            )
            return ValidationResult(False, errors, warnings)

        # ---- Quantity ------------------------------------------------
        if not isinstance(quantity, (int, float)) or quantity <= 0:
            errors.append("Quantity must be a positive number")
            return ValidationResult(False, errors, warnings)

        quantity = int(quantity)

        # Check lot size (only for A-share)
        if market == "a_share":
            try:
                lot_size = get_lot_size(symbol)
            except Exception as exc:
                warnings.append(f"Could not determine lot size: {exc}")
                lot_size = 100  # safe default

            if quantity % lot_size != 0:
                errors.append(
                    f"Quantity ({quantity}) must be a multiple of "
                    f"lot size ({lot_size})"
                )
                return ValidationResult(False, errors, warnings)

        # ---- Price ---------------------------------------------------
        if order_type == "limit":
            if price is None or price <= 0:
                errors.append("Limit orders require a positive price")
                return ValidationResult(False, errors, warnings)

            rounded = round(price, 2)
            if rounded != price:
                warnings.append(
                    f"Price rounded to 2 decimals: {price} → {rounded}"
                )
                price = rounded

        return ValidationResult(
            True,
            errors,
            warnings,
            {
                "symbol": symbol,
                "side": side,
                "quantity": quantity,
                "price": price,
                "order_type": order_type,
            },
        )


# =============================================================================
# CONFIG VALIDATOR
# =============================================================================


class ConfigValidator:
    """Validate configuration parameters."""

    @classmethod
    def _get_field(cls, config: Any, field: str, default: Any = None) -> Any:
        """Get a field from a dict or dataclass."""
        if isinstance(config, dict):
            return config.get(field, default)
        return getattr(config, field, default)

    @classmethod
    def _has_field(cls, config: Any, field: str) -> bool:
        """Check if a field exists in a dict or dataclass."""
        if isinstance(config, dict):
            return field in config
        return hasattr(config, field)

    @classmethod
    def validate_risk_config(
        cls, config: Any
    ) -> ValidationResult:
        """
        Validate risk configuration.

        Accepts either a dict or a RiskConfig dataclass.
        """
        errors: List[str] = []
        warnings: List[str] = []

        required = [
            "max_position_pct",
            "max_daily_loss_pct",
            "max_positions",
            "risk_per_trade_pct",
        ]

        for field in required:
            if not cls._has_field(config, field):
                errors.append(f"Missing required field: {field}")

        if errors:
            return ValidationResult(False, errors, warnings)

        # ---- Range checks -------------------------------------------
        max_pos = cls._get_field(config, "max_position_pct")
        if not (0 < max_pos <= 100):
            errors.append("max_position_pct must be in (0, 100]")

        max_loss = cls._get_field(config, "max_daily_loss_pct")
        if not (0 < max_loss <= 100):
            errors.append("max_daily_loss_pct must be in (0, 100]")

        max_n = cls._get_field(config, "max_positions")
        if not (1 <= max_n <= 100):
            errors.append("max_positions must be in [1, 100]")

        risk_per = cls._get_field(config, "risk_per_trade_pct")
        if not (0 < risk_per <= 10):
            errors.append("risk_per_trade_pct must be in (0, 10]")

        # ---- Warnings for aggressive settings -----------------------
        if max_pos > 25:
            warnings.append(
                f"High max_position_pct ({max_pos}%) — consider reducing"
            )
        if max_loss > 5:
            warnings.append(
                f"High max_daily_loss_pct ({max_loss}%) — consider reducing"
            )

        return ValidationResult(len(errors) == 0, errors, warnings, config)

    @classmethod
    def validate_data_config(cls, config: Any) -> ValidationResult:
        """Validate DataConfig fields."""
        errors: List[str] = []
        warnings: List[str] = []

        ttl = cls._get_field(config, "cache_ttl_hours", 0)
        if ttl <= 0:
            errors.append("cache_ttl_hours must be positive")

        parallel = cls._get_field(config, "parallel_downloads", 0)
        if not (1 <= parallel <= 50):
            errors.append("parallel_downloads must be in [1, 50]")

        timeout = cls._get_field(config, "request_timeout", 0)
        if timeout <= 0:
            errors.append("request_timeout must be positive")

        min_hist = cls._get_field(config, "min_history_days", 0)
        if min_hist < 1:
            errors.append("min_history_days must be >= 1")

        return ValidationResult(len(errors) == 0, errors, warnings, config)

    @classmethod
    def validate_model_config(cls, config: Any) -> ValidationResult:
        """Validate ModelConfig fields."""
        errors: List[str] = []
        warnings: List[str] = []

        seq_len = cls._get_field(config, "sequence_length", 0)
        if seq_len < 1:
            errors.append("sequence_length must be >= 1")

        horizon = cls._get_field(config, "prediction_horizon", 0)
        if horizon < 1:
            errors.append("prediction_horizon must be >= 1")

        embargo = cls._get_field(config, "embargo_bars", 0)
        if embargo < horizon:
            errors.append(
                f"embargo_bars ({embargo}) must be >= "
                f"prediction_horizon ({horizon})"
            )

        lr = cls._get_field(config, "learning_rate", 0)
        if not (0 < lr < 1):
            errors.append("learning_rate must be in (0, 1)")

        train = cls._get_field(config, "train_ratio", 0)
        val = cls._get_field(config, "val_ratio", 0)
        test = cls._get_field(config, "test_ratio", 0)
        ratio_sum = train + val + test
        if abs(ratio_sum - 1.0) >= 0.001:
            errors.append(f"Split ratios must sum to 1.0, got {ratio_sum:.4f}")

        return ValidationResult(len(errors) == 0, errors, warnings, config)


# =============================================================================
# TRADING DAY VALIDATOR
# =============================================================================


class TradingDayValidator:
    """Validate that a date is a trading day."""

    @classmethod
    def validate(
        cls,
        d: Any,
        allow_non_trading: bool = False,
    ) -> ValidationResult:
        """
        Validate that *d* is a trading day.

        Args:
            d: A date, datetime, or date-string.
            allow_non_trading: If True, non-trading days produce a
                               warning instead of an error.

        Returns:
            ValidationResult with ``date`` object in ``data``.
        """
        errors: List[str] = []
        warnings: List[str] = []

        parsed = DateRangeValidator._parse_date(d, "date", errors)
        if parsed is None:
            if not errors:
                errors.append(f"Cannot parse date: {d!r}")
            return ValidationResult(False, errors, warnings)

        if not is_trading_day(parsed):
            msg = f"{parsed} is not a trading day"
            if allow_non_trading:
                warnings.append(msg)
            else:
                errors.append(msg)
                return ValidationResult(False, errors, warnings)

        return ValidationResult(True, errors, warnings, parsed)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def validate_stock_code(code: str, market: str = "a_share") -> str:
    """Validate and return cleaned stock code. Raises on failure."""
    result = StockCodeValidator.validate(code, market=market)
    result.raise_if_invalid()
    return result.data


def validate_ohlcv(
    df: pd.DataFrame, min_rows: int = 10
) -> pd.DataFrame:
    """Validate and return cleaned OHLCV DataFrame. Raises on failure."""
    result = OHLCVValidator.validate(df, min_rows=min_rows)
    result.raise_if_invalid()
    return result.data


def validate_features(
    df: pd.DataFrame, feature_cols: List[str]
) -> pd.DataFrame:
    """Validate and return cleaned feature DataFrame. Raises on failure."""
    result = FeatureValidator.validate(df, feature_cols)
    result.raise_if_invalid()
    return result.data


def validate_date_range(
    start_date: Any,
    end_date: Any = None,
    max_range_days: int = 3650,
) -> Tuple[date, date]:
    """Validate and return (start_date, end_date). Raises on failure."""
    result = DateRangeValidator.validate(
        start_date, end_date, max_range_days=max_range_days
    )
    result.raise_if_invalid()
    return result.data