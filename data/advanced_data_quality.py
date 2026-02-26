# data/advanced_data_quality.py
"""
Advanced Data Quality Framework

FIXES:
- Data quality validation with statistical tests
- Anomaly detection and automatic repair
- Outlier handling with robust statistics
- Data drift detection
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Optional

import numpy as np
import pandas as pd
from scipy import stats

from utils.logger import get_logger

log = get_logger(__name__)


class QualityFlag(Enum):
    """Data quality flags."""
    VALID = "valid"
    SUSPECT = "suspect"
    INVALID = "invalid"
    MISSING = "missing"
    OUTLIER = "outlier"
    STALE = "stale"


@dataclass
class QualityReport:
    """Data quality assessment report."""
    symbol: str
    timestamp: datetime
    overall_score: float
    flags: list[QualityFlag] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    
    def is_acceptable(self, min_score: float = 0.7) -> bool:
        """Check if data quality is acceptable."""
        return self.overall_score >= min_score and QualityFlag.INVALID not in self.flags


class DataQualityValidator:
    """
    Advanced data quality validation with statistical tests.
    
    FIXES IMPLEMENTED:
    1. Multi-dimensional quality scoring
    2. Statistical anomaly detection
    3. Automatic data repair suggestions
    4. Drift detection for data distribution changes
    """
    
    def __init__(
        self,
        z_threshold: float = 3.0,
        iqr_multiplier: float = 1.5,
        min_correlation: float = 0.5,
        staleness_threshold_hours: float = 24.0,
    ):
        self.z_threshold = z_threshold
        self.iqr_multiplier = iqr_multiplier
        self.min_correlation = min_correlation
        self.staleness_threshold_hours = staleness_threshold_hours
        self._historical_profiles: dict[str, dict[str, float]] = {}
    
    def validate(
        self,
        symbol: str,
        df: pd.DataFrame,
        reference_df: Optional[pd.DataFrame] = None,
    ) -> QualityReport:
        """
        Comprehensive data quality validation.
        
        Args:
            symbol: Stock symbol
            df: Data to validate
            reference_df: Optional reference data for comparison
        
        Returns:
            QualityReport with score and issues
        """
        issues = []
        flags = []
        metrics = {}
        recommendations = []
        
        # 1. Completeness check
        completeness_score = self._check_completeness(df, issues, flags)
        metrics["completeness"] = completeness_score
        
        # 2. Accuracy check (statistical tests)
        accuracy_score = self._check_accuracy(df, issues, flags)
        metrics["accuracy"] = accuracy_score
        
        # 3. Consistency check
        consistency_score = self._check_consistency(df, issues, flags)
        metrics["consistency"] = consistency_score
        
        # 4. Timeliness check
        timeliness_score = self._check_timeliness(df, issues, flags)
        metrics["timeliness"] = timeliness_score
        
        # 5. Distribution check (drift detection)
        distribution_score = self._check_distribution(
            df, reference_df, issues, flags
        )
        metrics["distribution"] = distribution_score
        
        # Calculate overall score (weighted average)
        weights = {
            "completeness": 0.20,
            "accuracy": 0.30,
            "consistency": 0.25,
            "timeliness": 0.15,
            "distribution": 0.10,
        }
        
        overall_score = (
            completeness_score * weights["completeness"] +
            accuracy_score * weights["accuracy"] +
            consistency_score * weights["consistency"] +
            timeliness_score * weights["timeliness"] +
            distribution_score * weights["distribution"]
        )
        
        # Generate recommendations
        recommendations = self._generate_recommendations(flags, issues, metrics)
        
        return QualityReport(
            symbol=symbol,
            timestamp=datetime.now(),
            overall_score=overall_score,
            flags=flags,
            issues=issues,
            recommendations=recommendations,
            metrics=metrics,
        )
    
    def _check_completeness(
        self,
        df: pd.DataFrame,
        issues: list[str],
        flags: list[QualityFlag],
    ) -> float:
        """Check data completeness."""
        required_columns = ["open", "high", "low", "close", "volume"]
        missing_cols = [col for col in required_columns if col not in df.columns]
        
        if missing_cols:
            issues.append(f"Missing columns: {missing_cols}")
            flags.append(QualityFlag.MISSING)
            return 0.0
        
        # Check for missing values
        missing_pct = df[required_columns].isnull().sum().sum() / (len(df) * len(required_columns))
        
        if missing_pct > 0.1:
            issues.append(f"High missing value rate: {missing_pct:.2%}")
            flags.append(QualityFlag.MISSING)
            return max(0.0, 1.0 - missing_pct * 2)
        
        return 1.0 - missing_pct
    
    def _check_accuracy(
        self,
        df: pd.DataFrame,
        issues: list[str],
        flags: list[QualityFlag],
    ) -> float:
        """Check data accuracy using statistical tests."""
        if len(df) < 10:
            return 1.0  # Not enough data for statistical tests
        
        accuracy_scores = []
        
        # 1. Check for impossible values (negative prices, etc.)
        if "close" in df.columns:
            negative_prices = (df["close"] < 0).sum()
            if negative_prices > 0:
                issues.append(f"Negative prices detected: {negative_prices}")
                flags.append(QualityFlag.INVALID)
                accuracy_scores.append(0.0)
            else:
                accuracy_scores.append(1.0)
        
        # 2. Check OHLC consistency
        if all(col in df.columns for col in ["open", "high", "low", "close"]):
            invalid_ohlc = (
                (df["high"] < df["low"]).sum() +
                (df["high"] < df["open"]).sum() +
                (df["high"] < df["close"]).sum() +
                (df["low"] > df["open"]).sum() +
                (df["low"] > df["close"]).sum()
            )
            if invalid_ohlc > 0:
                issues.append(f"Invalid OHLC relationships: {invalid_ohlc}")
                flags.append(QualityFlag.SUSPECT)
                accuracy_scores.append(max(0.0, 1.0 - invalid_ohlc / len(df)))
            else:
                accuracy_scores.append(1.0)
        
        # 3. Detect outliers using Z-score
        if "close" in df.columns:
            z_scores = np.abs(stats.zscore(df["close"].dropna()))
            outliers = (z_scores > self.z_threshold).sum()
            if outliers > 0:
                outlier_pct = outliers / len(df)
                if outlier_pct > 0.05:
                    issues.append(f"High outlier rate: {outlier_pct:.2%}")
                    flags.append(QualityFlag.OUTLIER)
                accuracy_scores.append(max(0.0, 1.0 - outlier_pct))
            else:
                accuracy_scores.append(1.0)
        
        # 4. Detect outliers using IQR
        if "close" in df.columns and len(df) > 4:
            q1 = df["close"].quantile(0.25)
            q3 = df["close"].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - self.iqr_multiplier * iqr
            upper_bound = q3 + self.iqr_multiplier * iqr
            iqr_outliers = ((df["close"] < lower_bound) | (df["close"] > upper_bound)).sum()
            if iqr_outliers > 0:
                accuracy_scores.append(max(0.0, 1.0 - iqr_outliers / len(df)))
        
        return np.mean(accuracy_scores) if accuracy_scores else 1.0
    
    def _check_consistency(
        self,
        df: pd.DataFrame,
        issues: list[str],
        flags: list[QualityFlag],
    ) -> float:
        """Check data consistency."""
        consistency_scores = []
        
        # 1. Check for duplicate timestamps
        if isinstance(df.index, pd.DatetimeIndex):
            duplicates = df.index.duplicated().sum()
            if duplicates > 0:
                issues.append(f"Duplicate timestamps: {duplicates}")
                flags.append(QualityFlag.SUSPECT)
                consistency_scores.append(max(0.0, 1.0 - duplicates / len(df)))
            else:
                consistency_scores.append(1.0)
        
        # 2. Check for gaps in time series
        if isinstance(df.index, pd.DatetimeIndex) and len(df) > 1:
            time_diffs = df.index.to_series().diff()
            median_diff = time_diffs.median()
            large_gaps = (time_diffs > median_diff * 2).sum()
            if large_gaps > 0:
                issues.append(f"Time gaps detected: {large_gaps}")
                flags.append(QualityFlag.SUSPECT)
                consistency_scores.append(max(0.0, 1.0 - large_gaps / len(df)))
            else:
                consistency_scores.append(1.0)
        
        # 3. Check price returns consistency
        if "close" in df.columns and len(df) > 2:
            returns = df["close"].pct_change()
            extreme_returns = (np.abs(returns) > 0.2).sum()  # >20% daily move
            if extreme_returns > 0:
                extreme_pct = extreme_returns / len(df)
                if extreme_pct > 0.01:  # More than 1% extreme moves
                    issues.append(f"Extreme price moves: {extreme_returns}")
                    flags.append(QualityFlag.SUSPECT)
                consistency_scores.append(max(0.0, 1.0 - extreme_pct))
            else:
                consistency_scores.append(1.0)
        
        return np.mean(consistency_scores) if consistency_scores else 1.0
    
    def _check_timeliness(
        self,
        df: pd.DataFrame,
        issues: list[str],
        flags: list[QualityFlag],
    ) -> float:
        """Check data timeliness."""
        if len(df) == 0:
            flags.append(QualityFlag.STALE)
            issues.append("Empty dataset")
            return 0.0
        
        # Check last update time
        if isinstance(df.index, pd.DatetimeIndex):
            last_timestamp = df.index[-1]
            age_hours = (datetime.now() - last_timestamp).total_seconds() / 3600
            
            if age_hours > self.staleness_threshold_hours:
                issues.append(f"Data is {age_hours:.1f} hours old")
                flags.append(QualityFlag.STALE)
                return max(0.0, 1.0 - age_hours / (self.staleness_threshold_hours * 2))
        
        return 1.0
    
    def _check_distribution(
        self,
        df: pd.DataFrame,
        reference_df: Optional[pd.DataFrame],
        issues: list[str],
        flags: list[QualityFlag],
    ) -> float:
        """Check for distribution drift."""
        if reference_df is None or len(df) < 30 or len(reference_df) < 30:
            return 1.0  # Not enough data for drift detection
        
        # Store current profile
        current_profile = self._compute_distribution_profile(df)
        
        # Compare with reference
        if "close" in df.columns and "close" in reference_df.columns:
            # Kolmogorov-Smirnov test
            ks_stat, ks_pvalue = stats.ks_2samp(
                df["close"].dropna(),
                reference_df["close"].dropna(),
            )
            
            if ks_pvalue < 0.05:
                issues.append(f"Distribution drift detected (KS p={ks_pvalue:.3f})")
                flags.append(QualityFlag.SUSPECT)
                return max(0.0, 1.0 - ks_stat)
        
        return 1.0
    
    def _compute_distribution_profile(
        self,
        df: pd.DataFrame,
    ) -> dict[str, float]:
        """Compute distribution profile for drift tracking."""
        profile = {}
        
        for col in ["open", "high", "low", "close", "volume"]:
            if col in df.columns:
                data = df[col].dropna()
                if len(data) > 0:
                    profile[f"{col}_mean"] = float(data.mean())
                    profile[f"{col}_std"] = float(data.std())
                    profile[f"{col}_skew"] = float(data.skew())
                    profile[f"{col}_kurtosis"] = float(data.kurtosis())
        
        return profile
    
    def _generate_recommendations(
        self,
        flags: list[QualityFlag],
        issues: list[str],
        metrics: dict[str, Any],
    ) -> list[str]:
        """Generate actionable recommendations."""
        recommendations = []
        
        if QualityFlag.MISSING in flags:
            recommendations.append(
                "Fetch data from alternative sources to fill missing values"
            )
        
        if QualityFlag.INVALID in flags:
            recommendations.append(
                "Remove or correct invalid data points before analysis"
            )
        
        if QualityFlag.OUTLIER in flags:
            recommendations.append(
                "Review outliers - may indicate data errors or significant events"
            )
        
        if QualityFlag.STALE in flags:
            recommendations.append(
                "Refresh data from primary sources"
            )
        
        if QualityFlag.SUSPECT in flags:
            recommendations.append(
                "Cross-validate with multiple sources before trading decisions"
            )
        
        if metrics.get("completeness", 1.0) < 0.9:
            recommendations.append(
                "Implement data imputation strategy for missing values"
            )
        
        if metrics.get("accuracy", 1.0) < 0.9:
            recommendations.append(
                "Add data validation rules at ingestion time"
            )
        
        return recommendations
    
    def repair_data(
        self,
        df: pd.DataFrame,
        report: QualityReport,
    ) -> pd.DataFrame:
        """
        Attempt automatic data repair based on quality report.
        
        FIX: Automatic data quality improvement
        """
        repaired = df.copy()
        
        # 1. Remove duplicates
        if isinstance(repaired.index, pd.DatetimeIndex):
            repaired = repaired[~repaired.index.duplicated(keep="first")]
        
        # 2. Handle missing values
        numeric_cols = repaired.select_dtypes(include=[np.number]).columns
        repaired[numeric_cols] = repaired[numeric_cols].fillna(method="ffill")
        repaired[numeric_cols] = repaired[numeric_cols].fillna(method="bfill")
        
        # 3. Cap extreme outliers
        for col in ["open", "high", "low", "close"]:
            if col in repaired.columns:
                q1 = repaired[col].quantile(0.01)
                q3 = repaired[col].quantile(0.99)
                repaired[col] = repaired[col].clip(lower=q1, upper=q3)
        
        # 4. Fix OHLC inconsistencies
        if all(col in repaired.columns for col in ["open", "high", "low", "close"]):
            # Ensure high >= max(open, close)
            repaired["high"] = repaired[["high", "open", "close"]].max(axis=1)
            # Ensure low <= min(open, close)
            repaired["low"] = repaired[["low", "open", "close"]].min(axis=1)
        
        log.info(f"Data repair completed: {len(df)} -> {len(repaired)} rows")
        return repaired


# Singleton instance
_validator: Optional[DataQualityValidator] = None


def get_validator() -> DataQualityValidator:
    """Get validator singleton."""
    global _validator
    if _validator is None:
        _validator = DataQualityValidator()
    return _validator
