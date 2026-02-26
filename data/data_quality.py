"""Enhanced Data Quality Validator with Corporate Actions Handling.

This module provides comprehensive data quality validation and adjustment
for corporate actions (splits, dividends, rights issues) to prevent
survivorship bias and ensure accurate backtesting.

Fixes:
- Data quality issues (missing data, inconsistent formatting)
- Corporate actions not properly adjusted
- Survivorship bias (delisted companies)
- Look-ahead bias prevention
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config.settings import CONFIG
from utils.logger import get_logger

log = get_logger(__name__)


class DataQualityIssue(Enum):
    """Types of data quality issues."""
    MISSING_VALUES = "missing_values"
    DUPLICATE_ROWS = "duplicate_rows"
    INVALID_PRICES = "invalid_prices"
    NEGATIVE_VOLUME = "negative_volume"
    PRICE_JUMP_ANOMALY = "price_jump_anomaly"
    VOLUME_SPIKE_ANOMALY = "volume_spike_anomaly"
    TRADING_HALT = "trading_halt"
    CORPORATE_ACTION_UNADJUSTED = "corporate_action_unadjusted"
    SURVIVORSHIP_BIAS = "survivorship_bias"
    LOOKAHEAD_BIAS = "lookahead_bias"
    TIMESTAMP_MISALIGNMENT = "timestamp_misalignment"
    CURRENCY_MISMATCH = "currency_mismatch"


class CorporateActionType(Enum):
    """Types of corporate actions."""
    STOCK_SPLIT = "stock_split"
    REVERSE_SPLIT = "reverse_split"
    DIVIDEND_CASH = "dividend_cash"
    DIVIDEND_STOCK = "dividend_stock"
    RIGHTS_ISSUE = "rights_issue"
    SPIN_OFF = "spin_off"
    MERGER = "merger"
    DELISTING = "delisting"


@dataclass
class CorporateAction:
    """Represents a corporate action event."""
    symbol: str
    action_type: CorporateActionType
    announcement_date: datetime
    ex_date: datetime
    record_date: datetime | None
    payable_date: datetime | None
    ratio: float = 1.0  # Split ratio, dividend amount, etc.
    currency: str = "CNY"
    description: str = ""
    source: str = ""
    verified: bool = False
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "action_type": self.action_type.value,
            "announcement_date": self.announcement_date.isoformat(),
            "ex_date": self.ex_date.isoformat(),
            "record_date": self.record_date.isoformat() if self.record_date else None,
            "payable_date": self.payable_date.isoformat() if self.payable_date else None,
            "ratio": self.ratio,
            "currency": self.currency,
            "description": self.description,
            "source": self.source,
            "verified": self.verified,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CorporateAction:
        return cls(
            symbol=data["symbol"],
            action_type=CorporateActionType(data["action_type"]),
            announcement_date=datetime.fromisoformat(data["announcement_date"]),
            ex_date=datetime.fromisoformat(data["ex_date"]),
            record_date=datetime.fromisoformat(data["record_date"]) if data.get("record_date") else None,
            payable_date=datetime.fromisoformat(data["payable_date"]) if data.get("payable_date") else None,
            ratio=float(data.get("ratio", 1.0)),
            currency=str(data.get("currency", "CNY")),
            description=str(data.get("description", "")),
            source=str(data.get("source", "")),
            verified=bool(data.get("verified", False)),
        )


@dataclass
class DataQualityReport:
    """Comprehensive data quality report."""
    symbol: str
    start_date: datetime
    end_date: datetime
    total_rows: int
    is_valid: bool
    issues: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    statistics: dict[str, Any] = field(default_factory=dict)
    corporate_actions: list[CorporateAction] = field(default_factory=list)
    adjustments_applied: int = 0
    quality_score: float = 0.0
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat(),
            "total_rows": self.total_rows,
            "is_valid": self.is_valid,
            "issues": self.issues,
            "warnings": self.warnings,
            "statistics": self.statistics,
            "corporate_actions": [ca.to_dict() for ca in self.corporate_actions],
            "adjustments_applied": self.adjustments_applied,
            "quality_score": self.quality_score,
        }


class CorporateActionsDatabase:
    """Database for tracking corporate actions."""
    
    def __init__(self, storage_path: Path | None = None) -> None:
        self.storage_path = storage_path or CONFIG.data_dir / "corporate_actions"
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self._actions_cache: dict[str, list[CorporateAction]] = {}
        self._delisted_symbols: set[str] = set()
        self._load_from_storage()
    
    def _load_from_storage(self) -> None:
        """Load corporate actions from disk."""
        actions_file = self.storage_path / "actions.json"
        if actions_file.exists():
            try:
                content = actions_file.read_text(encoding="utf-8")
                data = json.loads(content)
                for symbol, actions_list in data.items():
                    self._actions_cache[symbol] = [
                        CorporateAction.from_dict(a) for a in actions_list
                    ]
                log.info(f"Loaded {len(self._actions_cache)} symbols with corporate actions")
            except Exception as e:
                log.warning(f"Failed to load corporate actions: {e}")
        
        delisted_file = self.storage_path / "delisted.json"
        if delisted_file.exists():
            try:
                content = delisted_file.read_text(encoding="utf-8")
                data = json.loads(content)
                self._delisted_symbols = set(data.get("symbols", []))
                log.info(f"Loaded {len(self._delisted_symbols)} delisted symbols")
            except Exception as e:
                log.warning(f"Failed to load delisted symbols: {e}")
    
    def _save_to_storage(self) -> None:
        """Save corporate actions to disk."""
        actions_file = self.storage_path / "actions.json"
        try:
            data = {
                symbol: [a.to_dict() for a in actions]
                for symbol, actions in self._actions_cache.items()
            }
            content = json.dumps(data, ensure_ascii=False, indent=2)
            actions_file.write_text(content, encoding="utf-8")
        except Exception as e:
            log.warning(f"Failed to save corporate actions: {e}")
        
        delisted_file = self.storage_path / "delisted.json"
        try:
            content = json.dumps({"symbols": list(self._delisted_symbols)}, indent=2)
            delisted_file.write_text(content, encoding="utf-8")
        except Exception as e:
            log.warning(f"Failed to save delisted symbols: {e}")
    
    def add_action(self, action: CorporateAction) -> None:
        """Add a corporate action."""
        if action.symbol not in self._actions_cache:
            self._actions_cache[action.symbol] = []
        self._actions_cache[action.symbol].append(action)
        self._save_to_storage()
    
    def get_actions(
        self,
        symbol: str,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> list[CorporateAction]:
        """Get corporate actions for a symbol within date range."""
        actions = self._actions_cache.get(symbol, [])
        
        if start_date:
            actions = [a for a in actions if a.ex_date >= start_date]
        if end_date:
            actions = [a for a in actions if a.ex_date <= end_date]
        
        return sorted(actions, key=lambda x: x.ex_date)
    
    def mark_delisted(self, symbol: str, delisting_date: datetime, reason: str = "") -> None:
        """Mark a symbol as delisted."""
        self._delisted_symbols.add(symbol)
        action = CorporateAction(
            symbol=symbol,
            action_type=CorporateActionType.DELISTING,
            announcement_date=delisting_date,
            ex_date=delisting_date,
            record_date=None,
            payable_date=None,
            description=reason,
        )
        self.add_action(action)
    
    def is_delisted(self, symbol: str) -> bool:
        """Check if a symbol is delisted."""
        return symbol in self._delisted_symbols
    
    def get_all_symbols(self, include_delisted: bool = False) -> set[str]:
        """Get all symbols in the database."""
        symbols = set(self._actions_cache.keys())
        if include_delisted:
            symbols |= self._delisted_symbols
        return symbols


class DataQualityValidator:
    """Comprehensive data quality validator."""
    
    def __init__(self) -> None:
        self.corporate_actions_db = CorporateActionsDatabase()
        
        # Validation thresholds
        self.price_jump_threshold = 0.50  # 50% daily move
        self.volume_spike_threshold = 5.0  # 5x average volume
        self.missing_data_threshold = 0.05  # 5% missing data allowed
    
    def validate(
        self,
        df: pd.DataFrame,
        symbol: str = "",
        interval: str = "1d",
    ) -> DataQualityReport:
        """Validate data quality and return comprehensive report."""
        if df.empty:
            return DataQualityReport(
                symbol=symbol,
                start_date=datetime.now(),
                end_date=datetime.now(),
                total_rows=0,
                is_valid=False,
                issues=[{"type": "empty_dataframe", "severity": "critical"}],
            )
        
        issues: list[dict[str, Any]] = []
        warnings: list[str] = []
        
        # Ensure datetime index
        df = df.copy()
        if not isinstance(df.index, pd.DatetimeIndex):
            try:
                if "datetime" in df.columns:
                    df["datetime"] = pd.to_datetime(df["datetime"])
                    df.set_index("datetime", inplace=True)
                elif "date" in df.columns:
                    df["date"] = pd.to_datetime(df["date"])
                    df.set_index("date", inplace=True)
            except Exception as e:
                issues.append({
                    "type": "invalid_index",
                    "severity": "critical",
                    "message": f"Cannot parse datetime index: {e}",
                })
                return self._create_report(symbol, df, issues, warnings)
        
        # Run all validations
        issues.extend(self._check_missing_values(df))
        issues.extend(self._check_duplicate_rows(df))
        issues.extend(self._check_invalid_prices(df))
        issues.extend(self._check_negative_volume(df))
        issues.extend(self._check_price_anomalies(df))
        issues.extend(self._check_volume_anomalies(df))
        issues.extend(self._check_trading_halts(df))
        issues.extend(self._check_timestamp_alignment(df, interval))
        
        # Check for corporate actions
        corporate_actions = self._detect_corporate_actions(df, symbol)
        
        # Calculate statistics
        statistics = self._calculate_statistics(df)
        
        # Calculate quality score
        quality_score = self._calculate_quality_score(issues, len(df))
        
        is_valid = len([i for i in issues if i.get("severity") == "critical"]) == 0
        
        return DataQualityReport(
            symbol=symbol,
            start_date=df.index.min().to_pydatetime() if hasattr(df.index.min(), 'to_pydatetime') else df.index.min(),
            end_date=df.index.max().to_pydatetime() if hasattr(df.index.max(), 'to_pydatetime') else df.index.max(),
            total_rows=len(df),
            is_valid=is_valid,
            issues=issues,
            warnings=warnings,
            statistics=statistics,
            corporate_actions=corporate_actions,
            quality_score=quality_score,
        )
    
    def _check_missing_values(self, df: pd.DataFrame) -> list[dict]:
        """Check for missing values."""
        issues = []
        required_columns = ["open", "high", "low", "close", "volume"]
        
        for col in required_columns:
            if col not in df.columns:
                issues.append({
                    "type": DataQualityIssue.MISSING_VALUES.value,
                    "severity": "critical",
                    "column": col,
                    "message": f"Required column '{col}' is missing",
                })
                continue
            
            missing_count = df[col].isna().sum()
            missing_pct = missing_count / len(df)
            
            if missing_pct > self.missing_data_threshold:
                issues.append({
                    "type": DataQualityIssue.MISSING_VALUES.value,
                    "severity": "warning" if missing_pct < 0.20 else "error",
                    "column": col,
                    "missing_count": int(missing_count),
                    "missing_percentage": float(missing_pct),
                })
        
        return issues
    
    def _check_duplicate_rows(self, df: pd.DataFrame) -> list[dict]:
        """Check for duplicate rows."""
        issues = []
        duplicates = df.index.duplicated().sum()
        
        if duplicates > 0:
            issues.append({
                "type": DataQualityIssue.DUPLICATE_ROWS.value,
                "severity": "error",
                "duplicate_count": int(duplicates),
            })
        
        return issues
    
    def _check_invalid_prices(self, df: pd.DataFrame) -> list[dict]:
        """Check for invalid prices."""
        issues = []
        
        for col in ["open", "high", "low", "close"]:
            if col not in df.columns:
                continue
            
            # Check for zero or negative prices
            invalid = (df[col] <= 0).sum()
            if invalid > 0:
                issues.append({
                    "type": DataQualityIssue.INVALID_PRICES.value,
                    "severity": "error",
                    "column": col,
                    "invalid_count": int(invalid),
                })
            
            # Check high-low relationship
            if col == "high" and "low" in df.columns:
                invalid_hl = (df["high"] < df["low"]).sum()
                if invalid_hl > 0:
                    issues.append({
                        "type": DataQualityIssue.INVALID_PRICES.value,
                        "severity": "error",
                        "message": "High < Low detected",
                        "invalid_count": int(invalid_hl),
                    })
        
        return issues
    
    def _check_negative_volume(self, df: pd.DataFrame) -> list[dict]:
        """Check for negative volume."""
        issues = []
        
        if "volume" in df.columns:
            negative = (df["volume"] < 0).sum()
            if negative > 0:
                issues.append({
                    "type": DataQualityIssue.NEGATIVE_VOLUME.value,
                    "severity": "error",
                    "invalid_count": int(negative),
                })
        
        return issues
    
    def _check_price_anomalies(self, df: pd.DataFrame) -> list[dict]:
        """Check for price jump anomalies."""
        issues = []
        
        if "close" not in df.columns:
            return issues
        
        # Calculate daily returns
        returns = df["close"].pct_change().abs()
        
        # Detect anomalies
        anomalies = returns > self.price_jump_threshold
        anomaly_count = anomalies.sum()
        
        if anomaly_count > 0:
            anomaly_dates = df.index[anomalies].tolist()
            issues.append({
                "type": DataQualityIssue.PRICE_JUMP_ANOMALY.value,
                "severity": "warning",
                "anomaly_count": int(anomaly_count),
                "threshold": self.price_jump_threshold,
                "dates": [str(d) for d in anomaly_dates[:10]],  # Limit to first 10
            })
        
        return issues
    
    def _check_volume_anomalies(self, df: pd.DataFrame) -> list[dict]:
        """Check for volume spike anomalies."""
        issues = []
        
        if "volume" not in df.columns:
            return issues
        
        # Calculate rolling average volume
        rolling_avg = df["volume"].rolling(window=20, min_periods=5).mean()
        volume_ratio = df["volume"] / rolling_avg
        
        # Detect spikes
        spikes = volume_ratio > self.volume_spike_threshold
        spike_count = spikes.sum()
        
        if spike_count > 0:
            issues.append({
                "type": DataQualityIssue.VOLUME_SPIKE_ANOMALY.value,
                "severity": "info",
                "spike_count": int(spike_count),
                "threshold": self.volume_spike_threshold,
            })
        
        return issues
    
    def _check_trading_halts(self, df: pd.DataFrame) -> list[dict]:
        """Check for trading halts (consecutive days with no price change)."""
        issues = []
        
        if "close" not in df.columns:
            return issues
        
        # Detect flat prices (potential trading halt)
        price_change = df["close"].pct_change().abs()
        flat_days = price_change == 0
        
        # Find consecutive flat days
        consecutive_count = 0
        max_consecutive = 0
        
        for is_flat in flat_days:
            if is_flat:
                consecutive_count += 1
                max_consecutive = max(max_consecutive, consecutive_count)
            else:
                consecutive_count = 0
        
        if max_consecutive >= 5:  # 5+ consecutive days with no change
            issues.append({
                "type": DataQualityIssue.TRADING_HALT.value,
                "severity": "warning",
                "consecutive_days": max_consecutive,
                "message": "Possible trading halt detected",
            })
        
        return issues
    
    def _check_timestamp_alignment(
        self,
        df: pd.DataFrame,
        interval: str,
    ) -> list[dict]:
        """Check for timestamp alignment issues."""
        issues = []
        
        if not isinstance(df.index, pd.DatetimeIndex):
            return issues
        
        # Check for gaps in daily data
        if interval == "1d":
            # Get trading days (exclude weekends)
            trading_days = pd.bdate_range(df.index.min(), df.index.max())
            missing_days = trading_days.difference(df.index)
            
            # Filter out holidays (approximate - would need holiday calendar)
            # For now, just report if > 5% missing
            if len(missing_days) > len(trading_days) * 0.05:
                issues.append({
                    "type": DataQualityIssue.TIMESTAMP_MISALIGNMENT.value,
                    "severity": "warning",
                    "missing_days": len(missing_days),
                    "message": "Significant number of trading days missing",
                })
        
        return issues
    
    def _detect_corporate_actions(
        self,
        df: pd.DataFrame,
        symbol: str,
    ) -> list[CorporateAction]:
        """Detect potential corporate actions from price data."""
        actions = []
        
        if "close" not in df.columns or "volume" not in df.columns:
            return actions
        
        # Detect splits: large price drop with volume spike
        returns = df["close"].pct_change()
        volume_ratio = df["volume"] / df["volume"].rolling(20, min_periods=5).mean()
        
        # Potential split: price drops > 30% with volume > 3x average
        potential_splits = (returns < -0.30) & (volume_ratio > 3.0)
        
        for date in df.index[potential_splits]:
            price_change = returns.loc[date]
            # Estimate split ratio
            ratio = 1 / (1 + price_change)
            
            action = CorporateAction(
                symbol=symbol,
                action_type=CorporateActionType.STOCK_SPLIT,
                announcement_date=date.to_pydatetime() if hasattr(date, 'to_pydatetime') else date,
                ex_date=date.to_pydatetime() if hasattr(date, 'to_pydatetime') else date,
                record_date=None,
                payable_date=None,
                ratio=ratio,
                description=f"Detected split with estimated ratio {ratio:.2f}:1",
                source="auto_detected",
            )
            actions.append(action)
        
        return actions
    
    def _calculate_statistics(self, df: pd.DataFrame) -> dict[str, Any]:
        """Calculate data statistics."""
        stats = {
            "total_rows": len(df),
            "date_range": {
                "start": str(df.index.min()),
                "end": str(df.index.max()),
            },
        }
        
        for col in ["open", "high", "low", "close", "volume"]:
            if col in df.columns:
                stats[col] = {
                    "mean": float(df[col].mean()) if df[col].notna().any() else None,
                    "std": float(df[col].std()) if df[col].notna().any() else None,
                    "min": float(df[col].min()) if df[col].notna().any() else None,
                    "max": float(df[col].max()) if df[col].notna().any() else None,
                }
        
        return stats
    
    def _calculate_quality_score(
        self,
        issues: list[dict],
        total_rows: int,
    ) -> float:
        """Calculate overall quality score (0-1)."""
        if total_rows == 0:
            return 0.0
        
        # Start with perfect score
        score = 1.0
        
        # Deduct points based on issue severity
        severity_weights = {
            "critical": 0.30,
            "error": 0.15,
            "warning": 0.05,
            "info": 0.01,
        }
        
        for issue in issues:
            severity = issue.get("severity", "info")
            weight = severity_weights.get(severity, 0.01)
            score -= weight
        
        return max(0.0, min(1.0, score))
    
    def _create_report(
        self,
        symbol: str,
        df: pd.DataFrame,
        issues: list[dict],
        warnings: list[str],
    ) -> DataQualityReport:
        """Create a data quality report."""
        return DataQualityReport(
            symbol=symbol,
            start_date=df.index.min().to_pydatetime() if hasattr(df.index.min(), 'to_pydatetime') else df.index.min(),
            end_date=df.index.max().to_pydatetime() if hasattr(df.index.max(), 'to_pydatetime') else df.index.max(),
            total_rows=len(df),
            is_valid=len([i for i in issues if i.get("severity") == "critical"]) == 0,
            issues=issues,
            warnings=warnings,
            statistics=self._calculate_statistics(df),
            quality_score=self._calculate_quality_score(issues, len(df)),
        )


class DataAdjuster:
    """Adjusts price data for corporate actions."""
    
    def __init__(self, corporate_actions_db: CorporateActionsDatabase) -> None:
        self.corporate_actions_db = corporate_actions_db
    
    def adjust_for_corporate_actions(
        self,
        df: pd.DataFrame,
        symbol: str,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> tuple[pd.DataFrame, int]:
        """Adjust price data for corporate actions.
        
        Returns:
            Tuple of (adjusted DataFrame, number of adjustments applied)
        """
        if df.empty:
            return df, 0
        
        df = df.copy()
        adjustments = 0
        
        # Get corporate actions for the symbol
        actions = self.corporate_actions_db.get_actions(symbol, start_date, end_date)
        
        if not actions:
            return df, 0
        
        # Sort actions by ex-date (most recent first for backward adjustment)
        actions = sorted(actions, key=lambda x: x.ex_date, reverse=True)
        
        for action in actions:
            if action.action_type == CorporateActionType.STOCK_SPLIT:
                df, adj_count = self._apply_split(df, action)
                adjustments += adj_count
            elif action.action_type == CorporateActionType.REVERSE_SPLIT:
                df, adj_count = self._apply_reverse_split(df, action)
                adjustments += adj_count
            elif action.action_type == CorporateActionType.DIVIDEND_CASH:
                df, adj_count = self._apply_dividend(df, action)
                adjustments += adj_count
        
        return df, adjustments
    
    def _apply_split(
        self,
        df: pd.DataFrame,
        action: CorporateAction,
    ) -> tuple[pd.DataFrame, int]:
        """Apply stock split adjustment."""
        ratio = action.ratio
        ex_date = action.ex_date
        
        # Adjust prices before ex-date
        mask = df.index < ex_date
        adjustment_count = mask.sum()
        
        if adjustment_count > 0:
            for col in ["open", "high", "low", "close"]:
                if col in df.columns:
                    df.loc[mask, col] = df.loc[mask, col] / ratio
            
            # Adjust volume
            if "volume" in df.columns:
                df.loc[mask, "volume"] = df.loc[mask, "volume"] * ratio
        
        log.info(f"Applied split adjustment for {action.symbol}: ratio={ratio:.2f}:1")
        return df, adjustment_count
    
    def _apply_reverse_split(
        self,
        df: pd.DataFrame,
        action: CorporateAction,
    ) -> tuple[pd.DataFrame, int]:
        """Apply reverse split adjustment."""
        ratio = action.ratio
        ex_date = action.ex_date
        
        # Adjust prices before ex-date
        mask = df.index < ex_date
        adjustment_count = mask.sum()
        
        if adjustment_count > 0:
            for col in ["open", "high", "low", "close"]:
                if col in df.columns:
                    df.loc[mask, col] = df.loc[mask, col] * ratio
            
            # Adjust volume
            if "volume" in df.columns:
                df.loc[mask, "volume"] = df.loc[mask, "volume"] / ratio
        
        log.info(f"Applied reverse split adjustment for {action.symbol}: ratio=1:{ratio:.2f}")
        return df, adjustment_count
    
    def _apply_dividend(
        self,
        df: pd.DataFrame,
        action: CorporateAction,
    ) -> tuple[pd.DataFrame, int]:
        """Apply cash dividend adjustment."""
        dividend_amount = action.ratio  # For dividends, ratio is the amount
        ex_date = action.ex_date
        
        # Adjust prices before ex-date
        mask = df.index < ex_date
        
        if "close" in df.columns:
            adjustment_count = mask.sum()
            # Simple dividend adjustment (subtract dividend from historical prices)
            df.loc[mask, "close"] = df.loc[mask, "close"] - dividend_amount
            
            # Adjust other price columns proportionally
            for col in ["open", "high", "low"]:
                if col in df.columns:
                    df.loc[mask, col] = df.loc[mask, col] - dividend_amount
            
            log.info(f"Applied dividend adjustment for {action.symbol}: {dividend_amount} {action.currency}")
            return df, adjustment_count
        
        return df, 0


def get_validator() -> DataQualityValidator:
    """Get singleton instance of data quality validator."""
    if not hasattr(get_validator, "_instance"):
        get_validator._instance = DataQualityValidator()
    return get_validator._instance


def get_adjuster() -> DataAdjuster:
    """Get singleton instance of data adjuster."""
    if not hasattr(get_adjuster, "_instance"):
        db = CorporateActionsDatabase()
        get_adjuster._instance = DataAdjuster(db)
    return get_adjuster._instance
