# models/enhanced_evaluation.py
"""Enhanced model evaluation and validation utilities.

This module provides:
- Comprehensive metric calculations
- Statistical significance testing
- Walk-forward validation improvements
- Regime-aware evaluation
- Risk-adjusted performance metrics
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from utils.logger import get_logger

log = get_logger(__name__)

_EPS = 1e-8


@dataclass
class ClassificationMetrics:
    """Comprehensive classification metrics."""
    
    accuracy: float = 0.0
    precision: np.ndarray = field(default_factory=lambda: np.array([]))
    recall: np.ndarray = field(default_factory=lambda: np.array([]))
    f1_score: np.ndarray = field(default_factory=lambda: np.array([]))
    confusion_matrix: np.ndarray = field(default_factory=lambda: np.array([]))
    roc_auc: float = 0.0
    pr_auc: float = 0.0
    cohen_kappa: float = 0.0
    matthews_corr: float = 0.0
    brier_score: float = 0.0
    log_loss: float = 0.0
    
    # Per-class metrics
    class_metrics: dict[int, dict[str, float]] = field(default_factory=dict)
    
    # Calibration metrics
    calibration_error: float = 0.0
    calibration_slope: float = 1.0
    calibration_intercept: float = 0.0
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "accuracy": float(self.accuracy),
            "precision": self.precision.tolist() if len(self.precision) > 0 else [],
            "recall": self.recall.tolist() if len(self.recall) > 0 else [],
            "f1_score": self.f1_score.tolist() if len(self.f1_score) > 0 else [],
            "roc_auc": float(self.roc_auc),
            "pr_auc": float(self.pr_auc),
            "cohen_kappa": float(self.cohen_kappa),
            "matthews_corr": float(self.matthews_corr),
            "brier_score": float(self.brier_score),
            "log_loss": float(self.log_loss),
            "calibration_error": float(self.calibration_error),
            "class_metrics": self.class_metrics,
        }


@dataclass
class TradingMetrics:
    """Trading performance metrics."""
    
    total_return: float = 0.0
    annualized_return: float = 0.0
    volatility: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_duration: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    avg_holding_period: float = 0.0
    tail_ratio: float = 0.0
    skewness: float = 0.0
    kurtosis: float = 0.0
    var_95: float = 0.0
    cvar_95: float = 0.0
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_return": float(self.total_return),
            "annualized_return": float(self.annualized_return),
            "volatility": float(self.volatility),
            "sharpe_ratio": float(self.sharpe_ratio),
            "sortino_ratio": float(self.sortino_ratio),
            "calmar_ratio": float(self.calmar_ratio),
            "max_drawdown": float(self.max_drawdown),
            "max_drawdown_duration": int(self.max_drawdown_duration),
            "win_rate": float(self.win_rate),
            "profit_factor": float(self.profit_factor),
            "avg_win": float(self.avg_win),
            "avg_loss": float(self.avg_loss),
            "largest_win": float(self.largest_win),
            "largest_loss": float(self.largest_loss),
            "trades": int(self.trades),
            "winning_trades": int(self.winning_trades),
            "losing_trades": int(self.losing_trades),
            "avg_holding_period": float(self.avg_holding_period),
            "tail_ratio": float(self.tail_ratio),
            "skewness": float(self.skewness),
            "kurtosis": float(self.kurtosis),
            "var_95": float(self.var_95),
            "cvar_95": float(self.cvar_95),
        }


def calculate_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
    num_classes: int = 3,
) -> ClassificationMetrics:
    """Calculate comprehensive classification metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Predicted probabilities (optional)
        num_classes: Number of classes
        
    Returns:
        ClassificationMetrics object
    """
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        confusion_matrix,
        roc_auc_score,
        average_precision_score,
        cohen_kappa_score,
        matthews_corrcoef,
        brier_score_loss,
        log_loss,
    )
    
    # calibration_error may not be available in all sklearn versions
    try:
        from sklearn.metrics import calibration_error
    except ImportError:
        calibration_error = None
    
    metrics = ClassificationMetrics()
    
    # Basic metrics
    metrics.accuracy = float(accuracy_score(y_true, y_pred))
    metrics.precision = precision_score(y_true, y_pred, average=None, zero_division=0)
    metrics.recall = recall_score(y_true, y_pred, average=None, zero_division=0)
    metrics.f1_score = f1_score(y_true, y_pred, average=None, zero_division=0)
    metrics.confusion_matrix = confusion_matrix(y_true, y_pred)
    
    # Per-class metrics
    for i in range(num_classes):
        metrics.class_metrics[i] = {
            "precision": float(metrics.precision[i]) if i < len(metrics.precision) else 0.0,
            "recall": float(metrics.recall[i]) if i < len(metrics.recall) else 0.0,
            "f1": float(metrics.f1_score[i]) if i < len(metrics.f1_score) else 0.0,
            "support": int(np.sum(y_true == i)),
        }
    
    # Aggregate metrics
    if len(np.unique(y_true)) > 1:
        try:
            metrics.cohen_kappa = float(cohen_kappa_score(y_true, y_pred))
            metrics.matthews_corr = float(matthews_corrcoef(y_true, y_pred))
        except (ValueError, TypeError):
            pass
    
    # Probability-based metrics
    if y_prob is not None:
        try:
            # Multi-class ROC AUC (OvR)
            if num_classes == 3 and y_prob.shape[1] == 3:
                metrics.roc_auc = float(roc_auc_score(
                    y_true, y_prob,
                    multi_class='ovr',
                    average='macro'
                ))
            elif num_classes == 2:
                metrics.roc_auc = float(roc_auc_score(y_true, y_prob[:, 1]))
            
            # Precision-Recall AUC
            metrics.pr_auc = float(average_precision_score(
                y_true, y_prob,
                average='macro'
            ))
        except (ValueError, TypeError):
            pass
        
        try:
            metrics.brier_score = float(brier_score_loss(y_true, y_prob[:, -1].argmax(axis=1) if y_prob.ndim > 1 else y_prob))
        except (ValueError, TypeError):
            pass
        
        try:
            metrics.log_loss = float(log_loss(y_true, y_prob, labels=list(range(num_classes))))
        except (ValueError, TypeError):
            pass
        
        # Calibration
        if calibration_error is not None:
            try:
                metrics.calibration_error = float(calibration_error(y_true, y_prob.max(axis=1)))
            except (ValueError, TypeError):
                pass
    
    return metrics


def calculate_trading_metrics(
    returns: np.ndarray,
    benchmark_returns: np.ndarray | None = None,
    trades: int | None = None,
    risk_free_rate: float = 0.02,
    trading_days_per_year: int = 252,
) -> TradingMetrics:
    """Calculate comprehensive trading performance metrics.
    
    Args:
        returns: Array of returns
        benchmark_returns: Benchmark returns for comparison
        trades: Number of trades
        risk_free_rate: Annual risk-free rate
        trading_days_per_year: Trading days per year
        
    Returns:
        TradingMetrics object
    """
    metrics = TradingMetrics()
    
    if len(returns) == 0:
        return metrics
    
    returns = np.asarray(returns, dtype=np.float64)
    returns = returns[np.isfinite(returns)]
    
    if len(returns) == 0:
        return metrics
    
    # Basic statistics
    metrics.total_return = float(np.prod(1 + returns) - 1)
    metrics.volatility = float(np.std(returns, ddof=1)) if len(returns) > 1 else 0.0
    metrics.annualized_return = float((1 + metrics.total_return) ** (trading_days_per_year / len(returns)) - 1)
    
    # Risk-adjusted returns
    excess_return = metrics.annualized_return - risk_free_rate
    annualized_vol = metrics.volatility * np.sqrt(trading_days_per_year)
    
    if annualized_vol > _EPS:
        metrics.sharpe_ratio = float(excess_return / annualized_vol)
    
    # Sortino ratio (downside deviation)
    downside_returns = returns[returns < 0]
    if len(downside_returns) > 1:
        downside_dev = float(np.std(downside_returns, ddof=1))
        if downside_dev > _EPS:
            metrics.sortino_ratio = float(excess_return / (downside_dev * np.sqrt(trading_days_per_year)))
    
    # Drawdown analysis
    cumulative = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = (cumulative - running_max) / running_max
    metrics.max_drawdown = float(np.min(drawdowns))
    
    # Drawdown duration
    in_drawdown = drawdowns < 0
    if in_drawdown.any():
        drawdown_periods = []
        current_duration = 0
        for is_dd in in_drawdown:
            if is_dd:
                current_duration += 1
            else:
                if current_duration > 0:
                    drawdown_periods.append(current_duration)
                current_duration = 0
        if current_duration > 0:
            drawdown_periods.append(current_duration)
        if drawdown_periods:
            metrics.max_drawdown_duration = int(max(drawdown_periods))
    
    # Calmar ratio
    if abs(metrics.max_drawdown) > _EPS:
        metrics.calmar_ratio = float(metrics.annualized_return / abs(metrics.max_drawdown))
    
    # Trade analysis
    if trades is not None:
        metrics.trades = trades
        winning_returns = returns[returns > 0]
        losing_returns = returns[returns < 0]
        
        metrics.winning_trades = len(winning_returns)
        metrics.losing_trades = len(losing_returns)
        
        if metrics.winning_trades > 0:
            metrics.win_rate = float(metrics.winning_trades / metrics.trades)
            metrics.avg_win = float(np.mean(winning_returns))
            metrics.largest_win = float(np.max(winning_returns))
        
        if metrics.losing_trades > 0:
            metrics.avg_loss = float(np.mean(losing_returns))
            metrics.largest_loss = float(np.min(losing_returns))
        
        # Profit factor
        gross_profit = float(np.sum(winning_returns)) if len(winning_returns) > 0 else 0.0
        gross_loss = float(abs(np.sum(losing_returns))) if len(losing_returns) > 0 else 0.0
        if gross_loss > _EPS:
            metrics.profit_factor = float(gross_profit / gross_loss)
    
    # Distribution statistics
    if len(returns) > 2:
        metrics.skewness = float(stats.skew(returns))
        metrics.kurtosis = float(stats.kurtosis(returns))
    
    # Tail risk
    if len(returns) >= 20:
        metrics.var_95 = float(np.percentile(returns, 5))
        metrics.cvar_95 = float(np.mean(returns[returns <= metrics.var_95]))
    
    # Tail ratio
    if len(returns) >= 20:
        percentile_95 = float(np.percentile(returns, 95))
        percentile_5 = float(np.percentile(returns, 5))
        if abs(percentile_5) > _EPS:
            metrics.tail_ratio = float(percentile_95 / abs(percentile_5))
    
    return metrics


def walk_forward_analysis(
    predictions: list[np.ndarray],
    actuals: list[np.ndarray],
    returns: list[np.ndarray],
    min_samples: int = 50,
) -> dict[str, Any]:
    """Perform walk-forward analysis on time-series predictions.
    
    Args:
        predictions: List of predictions for each fold
        actuals: List of actual values for each fold
        returns: List of returns for each fold
        min_samples: Minimum samples per fold
        
    Returns:
        Walk-forward analysis results
    """
    results = {
        "enabled": False,
        "folds": [],
        "stability_metrics": {},
    }
    
    n_folds = len(predictions)
    if n_folds == 0:
        return results
    
    # Validate inputs
    valid_folds = []
    for i, (pred, act, ret) in enumerate(zip(predictions, actuals, returns, strict=False)):
        if len(pred) >= min_samples and len(act) >= min_samples:
            valid_folds.append((i, pred, act, ret))
    
    if len(valid_folds) < 2:
        results["reason"] = f"insufficient_valid_folds (need>=2, got={len(valid_folds)})"
        return results
    
    results["enabled"] = True
    results["n_folds"] = len(valid_folds)
    
    # Calculate metrics for each fold
    fold_metrics = []
    for fold_idx, pred, act, ret in valid_folds:
        # Classification metrics
        pred_classes = np.argmax(pred, axis=1) if pred.ndim > 1 else pred
        acc = float(np.mean(pred_classes == act))
        
        # Trading metrics
        trading_met = calculate_trading_metrics(np.asarray(ret))
        
        fold_metrics.append({
            "fold": fold_idx,
            "accuracy": acc,
            "sharpe_ratio": trading_met.sharpe_ratio,
            "total_return": trading_met.total_return,
            "max_drawdown": trading_met.max_drawdown,
            "trades": trading_met.trades,
        })
    
    results["folds"] = fold_metrics
    
    # Calculate stability metrics
    accuracies = np.array([m["accuracy"] for m in fold_metrics])
    sharpe_ratios = np.array([m["sharpe_ratio"] for m in fold_metrics])
    returns_arr = np.array([m["total_return"] for m in fold_metrics])
    
    results["stability_metrics"] = {
        "accuracy_mean": float(np.mean(accuracies)),
        "accuracy_std": float(np.std(accuracies)),
        "accuracy_cv": float(np.std(accuracies) / (np.mean(accuracies) + _EPS)),
        "sharpe_mean": float(np.mean(sharpe_ratios)),
        "sharpe_std": float(np.std(sharpe_ratios)),
        "return_mean": float(np.mean(returns_arr)),
        "return_std": float(np.std(returns_arr)),
        "consistency_score": float(1.0 - (np.std(accuracies) / (np.mean(accuracies) + _EPS))),
    }
    
    # Statistical tests
    if len(fold_metrics) >= 3:
        # Test if accuracy is significantly better than random
        t_stat, p_value = stats.ttest_1samp(accuracies, 0.33)  # 3-class random guess
        results["statistical_tests"] = {
            "accuracy_vs_random": {
                "t_statistic": float(t_stat),
                "p_value": float(p_value),
                "significant": bool(p_value < 0.05),
            }
        }
    
    return results


def regime_analysis(
    returns: np.ndarray,
    predictions: np.ndarray,
    market_returns: np.ndarray | None = None,
) -> dict[str, Any]:
    """Analyze model performance across different market regimes.
    
    Args:
        returns: Model returns
        predictions: Model predictions
        market_returns: Market benchmark returns
        
    Returns:
        Regime analysis results
    """
    results = {
        "regimes": {},
        "regime_performance": {},
    }
    
    if len(returns) < 50:
        results["reason"] = "insufficient_samples"
        return results
    
    returns = np.asarray(returns)
    predictions = np.asarray(predictions)
    
    # Define regimes based on volatility and trend
    rolling_vol = pd.Series(returns).rolling(window=20).std()
    rolling_return = pd.Series(returns).rolling(window=20).mean()
    
    # Regime classification
    vol_median = float(np.median(rolling_vol.dropna()))
    ret_median = float(np.median(rolling_return.dropna()))
    
    regimes = []
    for vol, ret in zip(rolling_vol, rolling_return, strict=False):
        if pd.isna(vol) or pd.isna(ret):
            regimes.append("unknown")
        elif vol > vol_median and ret > ret_median:
            regimes.append("high_vol_bull")
        elif vol > vol_median and ret <= ret_median:
            regimes.append("high_vol_bear")
        elif vol <= vol_median and ret > ret_median:
            regimes.append("low_vol_bull")
        else:
            regimes.append("low_vol_bear")
    
    # Performance by regime
    for regime in set(regimes):
        mask = np.array([r == regime for r in regimes])
        if mask.sum() < 10:
            continue
        
        regime_returns = returns[mask]
        regime_preds = predictions[mask]
        
        met = calculate_trading_metrics(regime_returns)
        results["regime_performance"][regime] = met.to_dict()
        
        # Prediction accuracy by regime
        pred_classes = np.argmax(regime_preds, axis=1) if regime_preds.ndim > 1 else regime_preds
        # Note: Would need actual labels for accuracy calculation
    
    results["regimes"] = {
        "high_volatility_periods": int(sum(1 for r in regimes if "high_vol" in r)),
        "low_volatility_periods": int(sum(1 for r in regimes if "low_vol" in r)),
        "bull_periods": int(sum(1 for r in regimes if "bull" in r)),
        "bear_periods": int(sum(1 for r in regimes if "bear" in r)),
    }
    
    return results


def model_comparison_test(
    model1_returns: np.ndarray,
    model2_returns: np.ndarray,
    model1_name: str = "Model 1",
    model2_name: str = "Model 2",
) -> dict[str, Any]:
    """Statistical comparison of two models.
    
    Args:
        model1_returns: Returns from first model
        model2_returns: Returns from second model
        model1_name: Name of first model
        model2_name: Name of second model
        
    Returns:
        Comparison results
    """
    if len(model1_returns) != len(model2_returns):
        return {"error": "Returns arrays must have same length"}
    
    # Paired t-test
    t_stat, p_value = stats.ttest_rel(model1_returns, model2_returns)
    
    # Wilcoxon signed-rank test (non-parametric)
    w_stat, wilcoxon_p = stats.wilcoxon(model1_returns, model2_returns)
    
    # Effect size (Cohen's d)
    diff = model1_returns - model2_returns
    cohen_d = float(np.mean(diff) / (np.std(diff, ddof=1) + _EPS))
    
    # Bootstrap confidence interval for difference
    n_bootstrap = 1000
    bootstrap_diffs = []
    rng = np.random.default_rng(42)
    for _ in range(n_bootstrap):
        indices = rng.choice(len(diff), size=len(diff), replace=True)
        bootstrap_diffs.append(np.mean(diff[indices]))
    
    ci_lower, ci_upper = float(np.percentile(bootstrap_diffs, 2.5)), float(np.percentile(bootstrap_diffs, 97.5))
    
    return {
        "model1_name": model1_name,
        "model2_name": model2_name,
        "model1_mean_return": float(np.mean(model1_returns)),
        "model2_mean_return": float(np.mean(model2_returns)),
        "mean_difference": float(np.mean(diff)),
        "paired_t_test": {
            "t_statistic": float(t_stat),
            "p_value": float(p_value),
            "significant": bool(p_value < 0.05),
        },
        "wilcoxon_test": {
            "w_statistic": float(w_stat),
            "p_value": float(wilcoxon_p),
            "significant": bool(wilcoxon_p < 0.05),
        },
        "effect_size": {
            "cohens_d": float(cohen_d),
            "interpretation": _interpret_cohens_d(cohen_d),
        },
        "bootstrap_ci_95": {
            "lower": float(ci_lower),
            "upper": float(ci_upper),
            "significant": bool(ci_lower > 0 or ci_upper < 0),
        },
    }


def _interpret_cohens_d(d: float) -> str:
    """Interpret Cohen's d effect size."""
    abs_d = abs(d)
    if abs_d < 0.2:
        return "negligible"
    elif abs_d < 0.5:
        return "small"
    elif abs_d < 0.8:
        return "medium"
    else:
        return "large"
