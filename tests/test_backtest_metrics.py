from datetime import datetime, timedelta

import numpy as np

from analysis.backtest import Backtester


def _build_dates(n: int):
    start = datetime(2024, 1, 1)
    return [start + timedelta(days=i) for i in range(n)]


def test_backtest_metrics_include_extended_ratios():
    bt = Backtester.__new__(Backtester)
    daily = np.array([0.2, -0.1, 0.3, 0.1, -0.05], dtype=float)
    bench = np.array([0.1, -0.05, 0.2, 0.05, -0.02], dtype=float)

    result = bt._calculate_metrics(
        trades=[],
        daily_returns=daily,
        benchmark_daily=bench,
        dates=_build_dates(len(daily)),
        capital=100000.0,
        num_folds=2,
        fold_accuracies=[0.6, 0.7],
        fold_results=[],
    )

    assert np.isfinite(result.sortino_ratio)
    assert np.isfinite(result.information_ratio)
    assert np.isfinite(result.alpha)
    assert np.isfinite(result.beta)
    assert "Sortino Ratio" in result.summary()
    assert "Information Ratio" in result.summary()
    assert "Alpha/Beta" in result.summary()


def test_backtest_metrics_handle_flat_benchmark():
    bt = Backtester.__new__(Backtester)
    daily = np.array([0.1, -0.2, 0.15, -0.05], dtype=float)
    bench = np.zeros_like(daily)

    result = bt._calculate_metrics(
        trades=[],
        daily_returns=daily,
        benchmark_daily=bench,
        dates=_build_dates(len(daily)),
        capital=100000.0,
        num_folds=1,
        fold_accuracies=[0.5],
        fold_results=[],
    )

    assert result.beta == 0.0
    assert np.isfinite(result.alpha)


def test_backtest_metrics_sanitize_non_finite_and_alignment():
    bt = Backtester.__new__(Backtester)
    daily = np.array([0.1, np.nan, np.inf, -np.inf, 0.2], dtype=float)
    bench = np.array([0.0, 0.05], dtype=float)

    result = bt._calculate_metrics(
        trades=[],
        daily_returns=daily,
        benchmark_daily=bench,
        dates=_build_dates(len(daily)),
        capital=0.0,
        num_folds=1,
        fold_accuracies=[0.5],
        fold_results=[],
    )

    assert np.isfinite(result.total_return)
    assert np.isfinite(result.excess_return)
    assert np.isfinite(result.sharpe_ratio)
    assert np.isfinite(result.max_drawdown_pct)
    assert len(result.equity_curve) == len(bench)
    assert len(result.dates) == len(bench)
