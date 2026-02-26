# data/validator.py
"""Data validation and quality checks for fetched market data.

This module provides comprehensive validation for:
- OHLCV bar data
- Real-time quotes
- Historical time series
- Cross-source consistency
"""

import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

import pandas as pd

from config.runtime_env import env_float
from utils.logger import get_logger

log = get_logger(__name__)

# Validation thresholds
_MAX_PRICE_RATIO = float(env_float("TRADING_MAX_PRICE_RATIO", "1.5"))
_MIN_PRICE = float(env_float("TRADING_MIN_PRICE", "0.01"))
_MAX_SINGLE_BAR_CHANGE = float(env_float("TRADING_MAX_BAR_CHANGE", "0.25"))
_MAX_WICK_RATIO = float(env_float("TRADING_MAX_WICK_RATIO", "0.15"))
_MAX_VOLUME_SPIKE = float(env_float("TRADING_VOLUME_SPIKE", "10.0"))
_MAX_PRICE_GAP = float(env_float("TRADING_MAX_PRICE_GAP", "0.30"))
_MIN_DATA_POINTS = int(env_float("TRADING_MIN_DATA_POINTS", "5"))


@dataclass
class ValidationResult:
    """Result of data validation."""
    is_valid: bool
    score: float  # 0.0 to 1.0
    issues: list[str]
    warnings: list[str]
    checked_rows: int
    timestamp: datetime

    @classmethod
    def ok(cls, checked_rows: int = 0) -> "ValidationResult":
        return cls(
            is_valid=True,
            score=1.0,
            issues=[],
            warnings=[],
            checked_rows=checked_rows,
            timestamp=datetime.now(),
        )

    @classmethod
    def fail(cls, issues: list[str], checked_rows: int = 0) -> "ValidationResult":
        return cls(
            is_valid=False,
            score=0.0,
            issues=issues,
            warnings=[],
            checked_rows=checked_rows,
            timestamp=datetime.now(),
        )

    def add_warning(self, warning: str) -> None:
        self.warnings.append(warning)

    def merge(self, other: "ValidationResult") -> "ValidationResult":
        """Merge two validation results."""
        return ValidationResult(
            is_valid=self.is_valid and other.is_valid,
            score=(self.score + other.score) / 2,
            issues=self.issues + other.issues,
            warnings=self.warnings + other.warnings,
            checked_rows=self.checked_rows + other.checked_rows,
            timestamp=min(self.timestamp, other.timestamp),
        )


class DataValidator:
    """Validates market data quality."""

    def __init__(
        self,
        max_price_ratio: float = _MAX_PRICE_RATIO,
        min_price: float = _MIN_PRICE,
        max_bar_change: float = _MAX_SINGLE_BAR_CHANGE,
        max_wick_ratio: float = _MAX_WICK_RATIO,
        max_volume_spike: float = _MAX_VOLUME_SPIKE,
        max_price_gap: float = _MAX_PRICE_GAP,
    ):
        self.max_price_ratio = max_price_ratio
        self.min_price = min_price
        self.max_bar_change = max_bar_change
        self.max_wick_ratio = max_wick_ratio
        self.max_volume_spike = max_volume_spike
        self.max_price_gap = max_price_gap

    def validate_bars(
        self,
        df: pd.DataFrame,
        symbol: str = "",
        interval: str = "1d",
    ) -> ValidationResult:
        """Validate OHLCV bar data."""
        issues: list[str] = []
        warnings: list[str] = []
        score = 1.0

        if df is None or df.empty:
            return ValidationResult.fail(["Empty dataframe"], 0)

        required_cols = {"open", "high", "low", "close", "volume"}
        missing = required_cols - set(str(c).lower() for c in df.columns)
        if missing:
            return ValidationResult.fail([f"Missing columns: {missing}"], len(df))

        df_check = df.copy()
        df_check.columns = [str(c).lower() for c in df_check.columns]

        # Check for NaN values
        nan_counts = df_check[list(required_cols)].isna().sum()
        total_nans = int(nan_counts.sum())
        if total_nans > 0:
            nan_ratio = total_nans / (len(df_check) * len(required_cols))
            if nan_ratio > 0.5:
                issues.append(f"Too many NaN values: {nan_ratio:.1%}")
                score -= 0.5
            else:
                warnings.append(f"NaN values present: {nan_ratio:.1%}")
                score -= 0.1 * nan_ratio

        # Validate prices
        price_issues = self._validate_prices(df_check, symbol)
        issues.extend(price_issues.get("issues", []))
        warnings.extend(price_issues.get("warnings", []))
        score -= price_issues.get("penalty", 0.0)

        # Validate OHLC relationships
        ohlc_issues = self._validate_ohlc_relationships(df_check)
        issues.extend(ohlc_issues.get("issues", []))
        warnings.extend(ohlc_issues.get("warnings", []))
        score -= ohlc_issues.get("penalty", 0.0)

        # Validate price changes
        change_issues = self._validate_price_changes(df_check, interval)
        issues.extend(change_issues.get("issues", []))
        warnings.extend(change_issues.get("warnings", []))
        score -= change_issues.get("penalty", 0.0)

        # Validate volume
        volume_issues = self._validate_volume(df_check)
        issues.extend(volume_issues.get("issues", []))
        warnings.extend(volume_issues.get("warnings", []))
        score -= volume_issues.get("penalty", 0.0)

        # Validate index/timestamps
        index_issues = self._validate_index(df_check, interval)
        issues.extend(index_issues.get("issues", []))
        warnings.extend(index_issues.get("warnings", []))
        score -= index_issues.get("penalty", 0.0)

        return ValidationResult(
            is_valid=len(issues) == 0,
            score=max(0.0, score),
            issues=issues,
            warnings=warnings,
            checked_rows=len(df_check),
            timestamp=datetime.now(),
        )

    def _validate_prices(
        self,
        df: pd.DataFrame,
        symbol: str,
    ) -> dict[str, Any]:
        """Validate price values."""
        issues: list[str] = []
        warnings: list[str] = []
        penalty = 0.0

        price_cols = ["open", "high", "low", "close"]
        for col in price_cols:
            if col not in df.columns:
                continue

            prices = df[col].dropna()
            if len(prices) == 0:
                continue

            # Check for negative or zero prices
            invalid_prices = (prices <= 0).sum()
            if invalid_prices > 0:
                ratio = invalid_prices / len(prices)
                if ratio > 0.5:
                    issues.append(f"Column '{col}': {ratio:.1%} non-positive prices")
                    penalty += 0.3
                else:
                    warnings.append(f"Column '{col}': {invalid_prices} non-positive values")
                    penalty += 0.05 * ratio

            # Check for extreme values
            median = prices.median()
            extreme_threshold = median * self.max_price_ratio
            extreme = (prices > extreme_threshold).sum()
            if extreme > 0 and median > 0:
                ratio = extreme / len(prices)
                if ratio > 0.1:
                    warnings.append(f"Column '{col}': {ratio:.1%} extreme values (> {extreme_threshold:.2f})")
                    penalty += 0.1 * ratio

            # Check for minimum price
            min_price = prices.min()
            if min_price < self.min_price and median > self.min_price * 10:
                warnings.append(f"Column '{col}': very low price {min_price:.4f}")
                penalty += 0.05

        return {"issues": issues, "warnings": warnings, "penalty": penalty}

    def _validate_ohlc_relationships(
        self,
        df: pd.DataFrame,
    ) -> dict[str, Any]:
        """Validate OHLC relationships (H >= L, H >= O, H >= C, L <= O, L <= C)."""
        issues: list[str] = []
        warnings: list[str] = []
        penalty = 0.0

        if not all(col in df.columns for col in ["open", "high", "low", "close"]):
            return {"issues": issues, "warnings": warnings, "penalty": penalty}

        # High should be >= Low
        invalid_hl = (df["high"] < df["low"]).sum()
        if invalid_hl > 0:
            ratio = invalid_hl / len(df)
            if ratio > 0.1:
                issues.append(f"High < Low in {ratio:.1%} of rows")
                penalty += 0.3
            else:
                warnings.append(f"High < Low in {invalid_hl} rows")
                penalty += 0.1 * ratio

        # High should be >= Open and Close
        invalid_h = ((df["high"] < df["open"]) | (df["high"] < df["close"])).sum()
        if invalid_h > 0:
            ratio = invalid_h / len(df)
            warnings.append(f"High < Open/Close in {ratio:.1%} of rows")
            penalty += 0.1 * ratio

        # Low should be <= Open and Close
        invalid_l = ((df["low"] > df["open"]) | (df["low"] > df["close"])).sum()
        if invalid_l > 0:
            ratio = invalid_l / len(df)
            warnings.append(f"Low > Open/Close in {ratio:.1%} of rows")
            penalty += 0.1 * ratio

        # Check wick ratio (unusually long wicks can indicate bad data)
        body = abs(df["close"] - df["open"])
        high_wick = df["high"] - df[["open", "close"]].max(axis=1)
        low_wick = df[["open", "close"]].min(axis=1) - df["low"]
        range_total = df["high"] - df["low"]

        # Avoid division by zero
        mask = range_total > 0
        if mask.sum() > 0:
            high_wick_ratio = (high_wick[mask] / range_total[mask]).mean()
            low_wick_ratio = (low_wick[mask] / range_total[mask]).mean()

            if high_wick_ratio > self.max_wick_ratio * 2:
                warnings.append(f"Unusually long upper wicks (avg ratio: {high_wick_ratio:.2f})")
                penalty += 0.05
            if low_wick_ratio > self.max_wick_ratio * 2:
                warnings.append(f"Unusually long lower wicks (avg ratio: {low_wick_ratio:.2f})")
                penalty += 0.05

        return {"issues": issues, "warnings": warnings, "penalty": penalty}

    def _validate_price_changes(
        self,
        df: pd.DataFrame,
        interval: str,
    ) -> dict[str, Any]:
        """Validate price changes between bars."""
        issues: list[str] = []
        warnings: list[str] = []
        penalty = 0.0

        if "close" not in df.columns or len(df) < 2:
            return {"issues": issues, "warnings": warnings, "penalty": penalty}

        closes = df["close"].dropna()
        if len(closes) < 2:
            return {"issues": issues, "warnings": warnings, "penalty": penalty}

        # Calculate price changes
        pct_change = closes.pct_change().abs()

        # Check for extreme single-bar changes
        max_change = pct_change.max()
        if pd.notna(max_change) and max_change > self.max_bar_change:
            extreme_count = (pct_change > self.max_bar_change).sum()
            ratio = extreme_count / len(pct_change)
            if ratio > 0.1:
                issues.append(f"{ratio:.1%} bars have extreme changes (>{self.max_bar_change:.0%})")
                penalty += 0.3
            else:
                warnings.append(f"{extreme_count} bars have extreme changes (max: {max_change:.1%})")
                penalty += 0.1 * ratio

        # Check for price gaps (for daily data)
        if interval in ["1d", "1wk", "1mo"]:
            gaps = self._detect_price_gaps(df)
            if gaps:
                warnings.extend(gaps)
                penalty += 0.05 * len(gaps)

        return {"issues": issues, "warnings": warnings, "penalty": penalty}

    def _detect_price_gaps(
        self,
        df: pd.DataFrame,
    ) -> list[str]:
        """Detect suspicious price gaps."""
        warnings: list[str] = []

        if not all(col in df.columns for col in ["open", "close"]):
            return warnings

        if len(df) < 2:
            return warnings

        # Check gap between previous close and current open
        prev_close = df["close"].shift(1)
        gap = (df["open"] - prev_close) / prev_close

        large_gaps = gap.abs() > self.max_price_gap
        if large_gaps.sum() > 0:
            count = int(large_gaps.sum())
            max_gap = gap.abs().max()
            warnings.append(f"{count} large price gaps (max: {max_gap:.1%})")

        return warnings

    def _validate_volume(
        self,
        df: pd.DataFrame,
    ) -> dict[str, Any]:
        """Validate volume data."""
        issues: list[str] = []
        warnings: list[str] = []
        penalty = 0.0

        if "volume" not in df.columns:
            return {"issues": issues, "warnings": warnings, "penalty": penalty}

        volume = df["volume"].dropna()
        if len(volume) == 0:
            warnings.append("No volume data")
            return {"issues": issues, "warnings": warnings, "penalty": penalty}

        # Check for negative volume
        negative = (volume < 0).sum()
        if negative > 0:
            ratio = negative / len(volume)
            if ratio > 0.1:
                issues.append(f"{ratio:.1%} negative volume values")
                penalty += 0.3
            else:
                warnings.append(f"{negative} negative volume values")
                penalty += 0.1 * ratio

        # Check for volume spikes
        if volume.median() > 0:
            volume_ratio = volume / volume.rolling(window=20, min_periods=1).median()
            spikes = (volume_ratio > self.max_volume_spike).sum()
            if spikes > 0:
                ratio = spikes / len(volume)
                if ratio > 0.05:
                    warnings.append(f"{ratio:.1%} volume spikes (>{self.max_volume_spike}x median)")
                    penalty += 0.05 * ratio * 10

        # Check for zero volume (might be valid for some markets)
        zero_volume = (volume == 0).sum()
        if zero_volume > 0:
            ratio = zero_volume / len(volume)
            if ratio > 0.5:
                warnings.append(f"{ratio:.1%} bars with zero volume")
                penalty += 0.1

        return {"issues": issues, "warnings": warnings, "penalty": penalty}

    def _validate_index(
        self,
        df: pd.DataFrame,
        interval: str,
    ) -> dict[str, Any]:
        """Validate DataFrame index (timestamps)."""
        issues: list[str] = []
        warnings: list[str] = []
        penalty = 0.0

        if not isinstance(df.index, pd.DatetimeIndex):
            warnings.append("Index is not DatetimeIndex")
            penalty += 0.1
            return {"issues": issues, "warnings": warnings, "penalty": penalty}

        # Check for duplicate timestamps
        duplicates = df.index.duplicated().sum()
        if duplicates > 0:
            ratio = duplicates / len(df)
            if ratio > 0.1:
                issues.append(f"{ratio:.1%} duplicate timestamps")
                penalty += 0.2
            else:
                warnings.append(f"{duplicates} duplicate timestamps")
                penalty += 0.05 * ratio

        # Check for future dates
        now = datetime.now()
        future = (df.index > now).sum()
        if future > 0:
            ratio = future / len(df)
            warnings.append(f"{ratio:.1%} future timestamps")
            penalty += 0.1 * ratio

        # Check for reasonable time gaps
        if len(df) > 1:
            diffs = df.index.to_series().diff()
            median_diff = diffs.median()
            if pd.notna(median_diff) and median_diff.total_seconds() > 0:
                # Check for large gaps in the data
                large_gaps = (diffs > median_diff * 10).sum()
                if large_gaps > 0:
                    warnings.append(f"{large_gaps} large time gaps in data")
                    penalty += 0.02 * large_gaps

        return {"issues": issues, "warnings": warnings, "penalty": penalty}

    def validate_quote(
        self,
        quote: Any,
        symbol: str = "",
    ) -> ValidationResult:
        """Validate a single real-time quote."""
        issues: list[str] = []
        warnings: list[str] = []

        if quote is None:
            return ValidationResult.fail(["Quote is None"])

        # Check required attributes
        required = ["code", "price", "open", "high", "low", "close"]
        for attr in required:
            if not hasattr(quote, attr):
                issues.append(f"Missing attribute: {attr}")

        if issues:
            return ValidationResult.fail(issues)

        # Validate price
        try:
            price = float(getattr(quote, "price", 0))
            if price <= 0:
                issues.append(f"Invalid price: {price}")
            if price < self.min_price:
                warnings.append(f"Very low price: {price}")
        except (TypeError, ValueError):
            issues.append(f"Non-numeric price: {getattr(quote, 'price', None)}")

        # Validate OHLC relationships
        try:
            high = float(getattr(quote, "high", price))
            low = float(getattr(quote, "low", price))
            if high < low:
                issues.append(f"High ({high}) < Low ({low})")
        except (TypeError, ValueError) as e:
            warnings.append(f"OHLC validation error: {e}")

        # Check timestamp
        ts = getattr(quote, "timestamp", None)
        if ts is None:
            warnings.append("Missing timestamp")
        elif isinstance(ts, datetime):
            age = (datetime.now() - ts).total_seconds()
            if age > 3600:  # 1 hour
                warnings.append(f"Stale quote: {age:.0f}s old")

        return ValidationResult(
            is_valid=len(issues) == 0,
            score=1.0 if len(issues) == 0 else 0.0,
            issues=issues,
            warnings=warnings,
            checked_rows=1,
            timestamp=datetime.now(),
        )


# Global validator instance
_validator: DataValidator | None = None
_validator_lock = threading.Lock()


def get_validator() -> DataValidator:
    """Get or create global validator instance."""
    global _validator
    with _validator_lock:
        if _validator is None:
            _validator = DataValidator()
        return _validator


def validate_bars(
    df: pd.DataFrame,
    symbol: str = "",
    interval: str = "1d",
) -> ValidationResult:
    """Validate bar data using global validator."""
    return get_validator().validate_bars(df, symbol, interval)


def validate_quote(quote: Any, symbol: str = "") -> ValidationResult:
    """Validate a quote using global validator."""
    return get_validator().validate_quote(quote, symbol)


# Import threading for the lock
import threading
