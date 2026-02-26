# models/enhanced_data_quality.py
"""Enhanced data quality validation and improvement utilities.

This module provides:
- Comprehensive data quality assessment
- Advanced outlier detection and handling
- Missing data imputation strategies
- Data drift detection
- Feature quality analysis
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from utils.logger import get_logger

log = get_logger(__name__)

_EPS = 1e-8


class QualityLevel(Enum):
    """Data quality levels."""
    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    POOR = "poor"
    UNACCEPTABLE = "unacceptable"


@dataclass
class QualityReport:
    """Comprehensive data quality report."""
    
    # Overall quality
    overall_score: float = 0.0
    quality_level: QualityLevel = QualityLevel.UNACCEPTABLE
    
    # Completeness
    completeness_score: float = 0.0
    missing_ratio: float = 0.0
    missing_by_column: dict[str, float] = field(default_factory=dict)
    
    # Validity
    validity_score: float = 0.0
    invalid_values: dict[str, int] = field(default_factory=dict)
    outlier_ratio: float = 0.0
    
    # Consistency
    consistency_score: float = 0.0
    duplicate_ratio: float = 0.0
    inconsistency_count: int = 0
    
    # Distribution
    distribution_score: float = 0.0
    skewness_issues: list[str] = field(default_factory=list)
    kurtosis_issues: list[str] = field(default_factory=dict)
    
    # Temporal
    temporal_score: float = 0.0
    gaps_detected: int = 0
    irregular_frequency: bool = False
    
    # Recommendations
    issues: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "overall_score": float(self.overall_score),
            "quality_level": self.quality_level.value,
            "completeness_score": float(self.completeness_score),
            "validity_score": float(self.validity_score),
            "consistency_score": float(self.consistency_score),
            "distribution_score": float(self.distribution_score),
            "temporal_score": float(self.temporal_score),
            "missing_ratio": float(self.missing_ratio),
            "outlier_ratio": float(self.outlier_ratio),
            "duplicate_ratio": float(self.duplicate_ratio),
            "issues": self.issues,
            "recommendations": self.recommendations,
        }


def assess_dataframe_quality(
    df: pd.DataFrame,
    timestamp_col: str | None = None,
    expected_frequency: str | None = None,
) -> QualityReport:
    """Assess comprehensive data quality for a DataFrame.
    
    Args:
        df: DataFrame to assess
        timestamp_col: Name of timestamp column (optional)
        expected_frequency: Expected frequency for time series (e.g., '1min', '1D')
        
    Returns:
        QualityReport object
    """
    report = QualityReport()
    
    if df is None or len(df) == 0:
        report.issues.append("empty_dataframe")
        report.recommendations.append("Provide non-empty DataFrame")
        return report
    
    n_rows, n_cols = df.shape
    
    # 1. Completeness assessment
    _assess_completeness(df, report)
    
    # 2. Validity assessment
    _assess_validity(df, report)
    
    # 3. Consistency assessment
    _assess_consistency(df, report)
    
    # 4. Distribution assessment
    _assess_distribution(df, report)
    
    # 5. Temporal assessment (if applicable)
    if timestamp_col is not None:
        _assess_temporal(df, timestamp_col, expected_frequency, report)
    
    # Calculate overall score
    weights = {
        "completeness": 0.25,
        "validity": 0.25,
        "consistency": 0.20,
        "distribution": 0.15,
        "temporal": 0.15,
    }
    
    report.overall_score = (
        weights["completeness"] * report.completeness_score +
        weights["validity"] * report.validity_score +
        weights["consistency"] * report.consistency_score +
        weights["distribution"] * report.distribution_score +
        weights["temporal"] * report.temporal_score
    )
    
    # Determine quality level
    if report.overall_score >= 0.95:
        report.quality_level = QualityLevel.EXCELLENT
    elif report.overall_score >= 0.85:
        report.quality_level = QualityLevel.GOOD
    elif report.overall_score >= 0.70:
        report.quality_level = QualityLevel.ACCEPTABLE
    elif report.overall_score >= 0.50:
        report.quality_level = QualityLevel.POOR
    else:
        report.quality_level = QualityLevel.UNACCEPTABLE
    
    # Generate recommendations
    _generate_recommendations(report)
    
    return report


def _assess_completeness(df: pd.DataFrame, report: QualityReport) -> None:
    """Assess data completeness."""
    total_cells = df.size
    missing_cells = df.isna().sum().sum()
    
    report.missing_ratio = float(missing_cells / max(1, total_cells))
    report.completeness_score = 1.0 - report.missing_ratio
    
    # Missing by column
    for col in df.columns:
        col_missing = df[col].isna().sum()
        report.missing_by_column[col] = float(col_missing / len(df))
    
    if report.missing_ratio > 0.10:
        report.issues.append(f"high_missing_ratio ({report.missing_ratio:.1%})")
    
    # Check for columns with excessive missing data
    for col, ratio in report.missing_by_column.items():
        if ratio > 0.30:
            report.issues.append(f"column_{col}_excessive_missing ({ratio:.1%})")


def _assess_validity(df: pd.DataFrame, report: QualityReport) -> None:
    """Assess data validity."""
    invalid_counts = {}
    
    # Check for common validity issues
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
        
        col_invalid = 0
        
        # Check for infinite values
        if np.isinf(df[col]).any():
            col_invalid += int(np.isinf(df[col]).sum())
        
        # Check for negative values where inappropriate
        if col.lower() in ["price", "open", "high", "low", "close", "volume"]:
            if (df[col] < 0).any():
                col_invalid += int((df[col] < 0).sum())
        
        # Check for zero values in critical columns
        if col.lower() in ["volume", "amount"] and (df[col] == 0).any():
            # Zero volume might be valid for some periods
            pass
        
        if col_invalid > 0:
            invalid_counts[col] = col_invalid
    
    report.invalid_values = invalid_counts
    total_invalid = sum(invalid_counts.values())
    report.outlier_ratio = float(total_invalid / max(1, df.size))
    report.validity_score = 1.0 - min(1.0, report.outlier_ratio * 10)
    
    if report.outlier_ratio > 0.05:
        report.issues.append(f"high_invalid_value_ratio ({report.outlier_ratio:.1%})")


def _assess_consistency(df: pd.DataFrame, report: QualityReport) -> None:
    """Assess data consistency."""
    # Check for duplicates
    duplicate_rows = df.duplicated().sum()
    report.duplicate_ratio = float(duplicate_rows / len(df))
    
    if report.duplicate_ratio > 0.01:
        report.issues.append(f"high_duplicate_ratio ({report.duplicate_ratio:.1%})")
    
    # Check for OHLC consistency (if applicable)
    ohlc_cols = ["open", "high", "low", "close"]
    if all(col in df.columns for col in ohlc_cols):
        inconsistencies = 0
        
        # High should be >= low, open, close
        high_violations = (
            (df["high"] < df["low"]) |
            (df["high"] < df["open"]) |
            (df["high"] < df["close"])
        ).sum()
        
        # Low should be <= open, close
        low_violations = (
            (df["low"] > df["open"]) |
            (df["low"] > df["close"])
        ).sum()
        
        inconsistencies += int(high_violations + low_violations)
        
        if inconsistencies > 0:
            report.inconsistency_count = inconsistencies
            ohlc_violation_ratio = inconsistencies / len(df)
            report.issues.append(f"OHLC_inconsistencies ({ohlc_violation_ratio:.1%})")
    
    report.consistency_score = 1.0 - min(1.0, report.duplicate_ratio * 5) - min(
        1.0, report.inconsistency_count / max(1, len(df)) * 10
    )


def _assess_distribution(df: pd.DataFrame, report: QualityReport) -> None:
    """Assess data distribution quality."""
    skewness_issues = []
    kurtosis_issues = []
    
    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].nunique() < 3:
            continue
        
        skewness = float(stats.skew(df[col].dropna()))
        kurtosis = float(stats.kurtosis(df[col].dropna()))
        
        # Check for extreme skewness
        if abs(skewness) > 3.0:
            skewness_issues.append(f"{col}: {skewness:.2f}")
        
        # Check for extreme kurtosis
        if abs(kurtosis) > 10.0:
            kurtosis_issues.append(f"{col}: {kurtosis:.2f}")
    
    report.skewness_issues = skewness_issues
    report.kurtosis_issues = kurtosis_issues
    
    # Distribution score
    distribution_penalty = (
        len(skewness_issues) * 0.05 + len(kurtosis_issues) * 0.05
    )
    report.distribution_score = max(0.0, 1.0 - distribution_penalty)
    
    if skewness_issues:
        report.issues.append(f"extreme_skewness ({len(skewness_issues)} columns)")
    if kurtosis_issues:
        report.issues.append(f"extreme_kurtosis ({len(kurtosis_issues)} columns)")


def _assess_temporal(
    df: pd.DataFrame,
    timestamp_col: str,
    expected_frequency: str | None,
    report: QualityReport,
) -> None:
    """Assess temporal data quality."""
    if timestamp_col not in df.columns:
        report.temporal_score = 1.0  # Not applicable
        return
    
    timestamps = pd.to_datetime(df[timestamp_col], errors="coerce")
    if timestamps.isna().all():
        report.issues.append("invalid_timestamps")
        report.temporal_score = 0.0
        return
    
    timestamps = timestamps.dropna().sort_values()
    
    # Check for gaps
    if len(timestamps) > 1:
        diffs = timestamps.diff()
        median_diff = diffs.median()
        
        # Detect gaps (intervals > 3x median)
        if expected_frequency:
            try:
                expected_td = pd.Timedelta(expected_frequency)
                gap_threshold = expected_td * 3
            except ValueError:
                gap_threshold = median_diff * 3
        else:
            gap_threshold = median_diff * 3
        
        gaps = (diffs > gap_threshold).sum()
        report.gaps_detected = int(gaps)
        
        if gaps > 0:
            report.issues.append(f"temporal_gaps_detected ({gaps})")
        
        # Check for irregular frequency
        diff_std = diffs.std()
        if diff_std > median_diff * 0.5:
            report.irregular_frequency = True
            report.issues.append("irregular_frequency")
    
    report.temporal_score = max(
        0.0,
        1.0 - (report.gaps_detected * 0.02) - (0.2 if report.irregular_frequency else 0.0)
    )


def _generate_recommendations(report: QualityReport) -> None:
    """Generate recommendations based on issues."""
    if report.missing_ratio > 0.10:
        report.recommendations.append(
            "Consider imputation or removal of rows with missing values"
        )
    
    if report.outlier_ratio > 0.05:
        report.recommendations.append(
            "Review and handle invalid values (infinite, negative prices, etc.)"
        )
    
    if report.duplicate_ratio > 0.01:
        report.recommendations.append(
            "Remove duplicate rows to ensure data integrity"
        )
    
    if report.skewness_issues:
        report.recommendations.append(
            f"Apply transformations to highly skewed columns: {', '.join(report.skewness_issues[:3])}"
        )
    
    if report.gaps_detected > 0:
        report.recommendations.append(
            "Investigate and fill temporal gaps or adjust sampling frequency"
        )
    
    if report.inconsistency_count > 0:
        report.recommendations.append(
            "Fix OHLC inconsistencies (high < low, etc.)"
        )
    
    if not report.recommendations:
        report.recommendations.append("Data quality is acceptable - no major issues detected")


def detect_outliers(
    df: pd.DataFrame,
    method: str = "isolation_forest",
    contamination: float = 0.05,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Detect outliers in DataFrame.
    
    Args:
        df: DataFrame to analyze
        method: Detection method ('isolation_forest', 'zscore', 'iqr')
        contamination: Expected proportion of outliers
        columns: Columns to analyze (default: all numeric)
        
    Returns:
        DataFrame with outlier flags
    """
    if columns is None:
        columns = list(df.select_dtypes(include=[np.number]).columns)
    
    if len(columns) == 0:
        return pd.DataFrame(index=df.index)
    
    data = df[columns].dropna()
    
    if method == "zscore":
        # Z-score method
        z_scores = np.abs(stats.zscore(data))
        outlier_mask = (z_scores > 3.0).any(axis=1)
    
    elif method == "iqr":
        # IQR method
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        outlier_mask = ((data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR))).any(axis=1)
    
    elif method == "isolation_forest":
        # Isolation Forest
        try:
            from sklearn.ensemble import IsolationForest
            
            clf = IsolationForest(
                contamination=contamination,
                random_state=42,
                n_estimators=100,
            )
            preds = clf.fit_predict(data)
            outlier_mask = preds == -1
        except ImportError:
            log.warning("sklearn not available - falling back to Z-score method")
            return detect_outliers(df, method="zscore", contamination=contamination, columns=columns)
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    result = pd.DataFrame({
        "is_outlier": outlier_mask,
        "outlier_score": getattr(outlier_mask, "astype", lambda x: outlier_mask)(float),
    }, index=data.index)
    
    # Align with original DataFrame
    return result.reindex(df.index, fill_value=False)


def impute_missing_values(
    df: pd.DataFrame,
    method: str = "forward_fill",
    columns: list[str] | None = None,
    max_gap: int = 5,
) -> pd.DataFrame:
    """Impute missing values.
    
    Args:
        df: DataFrame to impute
        method: Imputation method ('forward_fill', 'backward_fill', 'interpolate', 'mean', 'median')
        columns: Columns to impute (default: all)
        max_gap: Maximum gap to fill (for time series methods)
        
    Returns:
        DataFrame with imputed values
    """
    if columns is None:
        columns = list(df.columns)
    
    result = df.copy()
    
    for col in columns:
        if col not in result.columns:
            continue
        
        if method == "forward_fill":
            result[col] = result[col].ffill(limit=max_gap)
        elif method == "backward_fill":
            result[col] = result[col].bfill(limit=max_gap)
        elif method == "interpolate":
            result[col] = result[col].interpolate(method="linear", limit=max_gap)
        elif method == "mean":
            result[col] = result[col].fillna(result[col].mean())
        elif method == "median":
            result[col] = result[col].fillna(result[col].median())
    
    return result


def detect_data_drift(
    reference_data: pd.DataFrame,
    new_data: pd.DataFrame,
    method: str = "ks_test",
    threshold: float = 0.05,
) -> dict[str, Any]:
    """Detect data drift between reference and new data.
    
    Args:
        reference_data: Reference (training) data
        new_data: New (production) data
        method: Drift detection method ('ks_test', 'psi', 'wasserstein')
        threshold: Significance threshold
        
    Returns:
        Drift detection results
    """
    results = {
        "drift_detected": False,
        "drifted_columns": [],
        "column_results": {},
    }
    
    common_columns = set(reference_data.columns) & set(new_data.columns)
    
    for col in common_columns:
        if not pd.api.types.is_numeric_dtype(reference_data[col]):
            continue
        
        ref_data = reference_data[col].dropna()
        new_data_col = new_data[col].dropna()
        
        if len(ref_data) < 10 or len(new_data_col) < 10:
            continue
        
        if method == "ks_test":
            # Kolmogorov-Smirnov test
            stat, p_value = stats.ks_2samp(ref_data, new_data_col)
            drifted = p_value < threshold
            results["column_results"][col] = {
                "statistic": float(stat),
                "p_value": float(p_value),
                "drifted": bool(drifted),
            }
        
        elif method == "psi":
            # Population Stability Index
            psi_value = _calculate_psi(ref_data, new_data_col)
            drifted = psi_value > 0.25  # Common PSI threshold
            results["column_results"][col] = {
                "psi": float(psi_value),
                "drifted": bool(drifted),
                "severity": _psi_severity(psi_value),
            }
        
        elif method == "wasserstein":
            # Wasserstein distance
            distance = stats.wasserstein_distance(ref_data, new_data_col)
            # Normalize by reference std
            normalized_distance = distance / (ref_data.std() + _EPS)
            drifted = normalized_distance > threshold
            results["column_results"][col] = {
                "distance": float(distance),
                "normalized_distance": float(normalized_distance),
                "drifted": bool(drifted),
            }
        
        if results["column_results"][col]["drifted"]:
            results["drifted_columns"].append(col)
            results["drift_detected"] = True
    
    return results


def _calculate_psi(ref: pd.Series, new: pd.Series, buckets: int = 10) -> float:
    """Calculate Population Stability Index."""
    # Create breakpoints
    breakpoints = np.percentile(ref, np.linspace(0, 100, buckets + 1))
    breakpoints = np.unique(breakpoints)
    
    if len(breakpoints) < 2:
        return 0.0
    
    # Calculate distributions
    ref_counts = pd.cut(ref, bins=breakpoints, include_lowest=True).value_counts().sort_index()
    new_counts = pd.cut(new, bins=breakpoints, include_lowest=True).value_counts().sort_index()
    
    # Convert to proportions
    ref_props = ref_counts / len(ref)
    new_props = new_counts / len(new)
    
    # Align indices
    all_indices = ref_props.index.union(new_props.index)
    ref_props = ref_props.reindex(all_indices, fill_value=0.001)  # Avoid log(0)
    new_props = new_props.reindex(all_indices, fill_value=0.001)
    
    # Calculate PSI
    psi = np.sum((new_props - ref_props) * np.log(new_props / ref_props))
    
    return float(psi)


def _psi_severity(psi: float) -> str:
    """Interpret PSI severity."""
    if psi < 0.1:
        return "negligible"
    elif psi < 0.2:
        return "moderate"
    else:
        return "significant"
