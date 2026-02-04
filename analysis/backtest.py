"""
Walk-Forward Backtesting System

FIXED Issues:
- Scaler fitted only on training data for each fold
- Proper temporal split
- No look-ahead bias
- Realistic trading simulation

Author: AI Trading System
Version: 2.0
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime

from config import CONFIG
from data.fetcher import DataFetcher
from data.features import FeatureEngine
from data.processor import DataProcessor
from models.ensemble import EnsembleModel
from utils.logger import log


@dataclass
class BacktestResult:
    """Backtest results"""
    total_return: float
    benchmark_return: float
    excess_return: float
    trades: int
    win_rate: float
    profit_factor: float
    sharpe_ratio: float
    max_drawdown: float
    calmar_ratio: float
    num_folds: int
    avg_fold_accuracy: float
    
    def is_profitable(self) -> bool:
        return (
            self.excess_return > 0 and
            self.profit_factor > 1.0 and
            self.sharpe_ratio > 0.5
        )
    
    def summary(self) -> str:
        status = "✅ PROFITABLE" if self.is_profitable() else "❌ NOT PROFITABLE"
        
        return f"""
{'=' * 60}
            BACKTEST RESULTS - {status}
{'=' * 60}

  RETURNS:
    Strategy:    {self.total_return:+.2f}%
    Benchmark:   {self.benchmark_return:+.2f}%
    Excess:      {self.excess_return:+.2f}%

  TRADING:
    Trades:      {self.trades}
    Win Rate:    {self.win_rate:.1%}
    Profit Factor: {self.profit_factor:.2f}

  RISK:
    Sharpe Ratio:  {self.sharpe_ratio:.2f}
    Max Drawdown:  {self.max_drawdown:.1%}
    Calmar Ratio:  {self.calmar_ratio:.2f}

  WALK-FORWARD:
    Folds:       {self.num_folds}
    Avg Accuracy: {self.avg_fold_accuracy:.1%}

{'=' * 60}
"""


class Backtester:
    """
    Walk-Forward Backtesting System
    
    Properly handles:
    - Temporal data splits (no leakage)
    - Scaler fitting per fold
    - Realistic transaction costs
    - Multiple stocks
    
    Usage:
        bt = Backtester()
        result = bt.run(stock_codes=['600519', '000858'])
        print(result.summary())
    """
    
    def __init__(self):
        self.fetcher = DataFetcher()
        self.feature_engine = FeatureEngine()
    
    def run(self,
            stock_codes: List[str] = None,
            train_months: int = 12,
            test_months: int = 1,
            min_data_days: int = 500) -> BacktestResult:
        """
        Run walk-forward backtest
        
        Args:
            stock_codes: Stocks to backtest (default: first 5 from pool)
            train_months: Training period in months
            test_months: Testing period in months
            min_data_days: Minimum days of data required
            
        Returns:
            BacktestResult with all metrics
        """
        stocks = stock_codes or CONFIG.STOCK_POOL[:5]
        
        log.info(f"Running walk-forward backtest")
        log.info(f"  Stocks: {len(stocks)}")
        log.info(f"  Train: {train_months} months, Test: {test_months} months")
        
        # Collect and validate data
        all_data = self._collect_data(stocks, min_data_days)
        
        if not all_data:
            raise ValueError("No valid data available for backtesting")
        
        # Find common date range
        min_date = max(df.index.min() for df in all_data.values())
        max_date = min(df.index.max() for df in all_data.values())
        
        log.info(f"  Date range: {min_date.date()} to {max_date.date()}")
        
        # Generate walk-forward folds
        folds = self._generate_folds(min_date, max_date, train_months, test_months)
        
        if not folds:
            raise ValueError("Insufficient data for walk-forward testing")
        
        log.info(f"  Folds: {len(folds)}")
        
        # Run backtest for each fold
        all_preds = []
        all_labels = []
        all_returns = []
        all_confs = []
        fold_accuracies = []
        
        for fold_idx, (train_start, train_end, test_start, test_end) in enumerate(folds):
            log.info(f"\nFold {fold_idx + 1}/{len(folds)}")
            log.info(f"  Train: {train_start.date()} - {train_end.date()}")
            log.info(f"  Test:  {test_start.date()} - {test_end.date()}")
            
            fold_result = self._run_fold(
                all_data, train_start, train_end, test_start, test_end
            )
            
            if fold_result is not None:
                preds, labels, returns, confs, accuracy = fold_result
                all_preds.extend(preds)
                all_labels.extend(labels)
                all_returns.extend(returns)
                all_confs.extend(confs)
                fold_accuracies.append(accuracy)
        
        if not all_preds:
            raise ValueError("No predictions generated")
        
        # Compute final metrics
        result = self._compute_metrics(
            np.array(all_preds),
            np.array(all_labels),
            np.array(all_returns),
            np.array(all_confs),
            len(folds),
            np.mean(fold_accuracies) if fold_accuracies else 0
        )
        
        return result
    
    def _collect_data(self, stocks: List[str], 
                      min_days: int) -> Dict[str, pd.DataFrame]:
        """Collect and process data for all stocks"""
        all_data = {}
        
        for code in stocks:
            try:
                df = self.fetcher.get_history(code, days=2000)
                
                if len(df) < min_days:
                    log.warning(f"Insufficient data for {code}: {len(df)} days")
                    continue
                
                # Create features
                df = self.feature_engine.create_features(df)
                
                if len(df) >= CONFIG.SEQUENCE_LENGTH + 50:
                    all_data[code] = df
                    log.info(f"  {code}: {len(df)} samples")
                    
            except Exception as e:
                log.warning(f"Failed to load {code}: {e}")
        
        return all_data
    
    def _generate_folds(self,
                        min_date: pd.Timestamp,
                        max_date: pd.Timestamp,
                        train_months: int,
                        test_months: int) -> List[Tuple]:
        """Generate walk-forward folds"""
        folds = []
        train_start = min_date
        
        while True:
            train_end = train_start + pd.DateOffset(months=train_months)
            test_start = train_end
            test_end = test_start + pd.DateOffset(months=test_months)
            
            if test_end > max_date:
                break
            
            folds.append((train_start, train_end, test_start, test_end))
            
            # Move forward by test period
            train_start = train_start + pd.DateOffset(months=test_months)
        
        return folds
    
    def _run_fold(self,
                  all_data: Dict[str, pd.DataFrame],
                  train_start: pd.Timestamp,
                  train_end: pd.Timestamp,
                  test_start: pd.Timestamp,
                  test_end: pd.Timestamp) -> Optional[Tuple]:
        """
        Run a single fold of walk-forward backtest
        
        CRITICAL: Scaler is fitted ONLY on training data
        """
        # Create fresh processor for this fold
        processor = DataProcessor()
        feature_cols = self.feature_engine.get_feature_columns()
        
        # Phase 1: Collect training data and fit scaler
        train_features = []
        
        for code, df in all_data.items():
            mask = (df.index >= train_start) & (df.index < train_end)
            fold_df = df[mask]
            
            if len(fold_df) >= CONFIG.SEQUENCE_LENGTH:
                train_features.append(fold_df[feature_cols].values)
        
        if not train_features:
            log.warning("No training data for this fold")
            return None
        
        # Fit scaler on training data ONLY
        combined_train = np.concatenate(train_features)
        processor.fit_scaler(combined_train)
        
        # Phase 2: Prepare training sequences
        X_train, y_train = [], []
        
        for code, df in all_data.items():
            df_with_labels = processor.create_labels(df.copy())
            
            mask = (df_with_labels.index >= train_start) & (df_with_labels.index < train_end)
            fold_df = df_with_labels[mask]
            
            if len(fold_df) >= CONFIG.SEQUENCE_LENGTH + 10:
                X, y, _ = processor.prepare_sequences(
                    fold_df, feature_cols, fit_scaler=False
                )
                if len(X) > 0:
                    X_train.append(X)
                    y_train.append(y)
        
        if not X_train:
            log.warning("No training sequences")
            return None
        
        X_train = np.concatenate(X_train)
        y_train = np.concatenate(y_train)
        
        # Phase 3: Train model
        input_size = X_train.shape[2]
        model = EnsembleModel(input_size, model_names=['lstm', 'gru', 'tcn'])
        
        # Split training for validation
        split = int(len(X_train) * 0.85)
        
        model.train(
            X_train[:split], y_train[:split],
            X_train[split:], y_train[split:],
            epochs=30  # Shorter for backtest
        )
        
        # Phase 4: Test on out-of-sample data
        X_test, y_test, r_test = [], [], []
        
        for code, df in all_data.items():
            df_with_labels = processor.create_labels(df.copy())
            
            mask = (df_with_labels.index >= test_start) & (df_with_labels.index < test_end)
            fold_df = df_with_labels[mask]
            
            if len(fold_df) >= CONFIG.SEQUENCE_LENGTH + 5:
                X, y, r = processor.prepare_sequences(
                    fold_df, feature_cols, fit_scaler=False
                )
                if len(X) > 0:
                    X_test.append(X)
                    y_test.append(y)
                    r_test.append(r)
        
        if not X_test:
            log.warning("No test data")
            return None
        
        X_test = np.concatenate(X_test)
        y_test = np.concatenate(y_test)
        r_test = np.concatenate(r_test)
        
        # Phase 5: Generate predictions
        preds = []
        confs = []
        
        for i in range(len(X_test)):
            pred = model.predict(X_test[i:i+1])
            preds.append(pred.predicted_class)
            confs.append(pred.confidence)
        
        # Calculate fold accuracy
        accuracy = np.mean(np.array(preds) == y_test)
        log.info(f"  Fold accuracy: {accuracy:.2%}")
        
        return preds, y_test.tolist(), r_test.tolist(), confs, accuracy
    
    def _compute_metrics(self,
                         preds: np.ndarray,
                         labels: np.ndarray,
                         returns: np.ndarray,
                         confs: np.ndarray,
                         num_folds: int,
                         avg_accuracy: float) -> BacktestResult:
        """Compute backtest metrics"""
        # Position based on prediction
        position = np.zeros(len(preds))
        position[preds == 2] = 1   # UP -> Long
        position[preds == 0] = -1  # DOWN -> Short/Exit
        
        # Only trade when confident
        position = position * (confs >= CONFIG.MIN_CONFIDENCE)
        
        # Calculate returns with costs
        costs = CONFIG.COMMISSION * 2 + CONFIG.SLIPPAGE * 2 + CONFIG.STAMP_TAX
        
        strategy_returns = position * returns / 100
        trade_costs = np.abs(np.diff(position, prepend=0)) * costs
        net_returns = strategy_returns - trade_costs
        
        # Benchmark
        buy_hold_returns = returns / 100
        
        # Cumulative
        cum_strategy = (1 + net_returns).cumprod()
        cum_buyhold = (1 + buy_hold_returns).cumprod()
        
        total_return = (cum_strategy[-1] - 1) * 100 if len(cum_strategy) > 0 else 0
        buy_hold_return = (cum_buyhold[-1] - 1) * 100 if len(cum_buyhold) > 0 else 0
        
        # Trade stats
        trades = int(np.sum(position != 0))
        
        if trades > 0:
            trade_returns = net_returns[position != 0]
            wins = (trade_returns > 0).sum()
            win_rate = wins / trades
            
            gross_profit = trade_returns[trade_returns > 0].sum()
            gross_loss = abs(trade_returns[trade_returns < 0].sum())
            profit_factor = gross_profit / (gross_loss + 1e-8)
        else:
            win_rate = 0
            profit_factor = 0
        
        # Sharpe
        if len(net_returns) > 1 and net_returns.std() > 0:
            sharpe = net_returns.mean() / net_returns.std() * np.sqrt(252)
        else:
            sharpe = 0
        
        # Max drawdown
        if len(cum_strategy) > 0:
            running_max = np.maximum.accumulate(cum_strategy)
            drawdown = (cum_strategy - running_max) / (running_max + 1e-8)
            max_drawdown = abs(drawdown.min())
        else:
            max_drawdown = 0
        
        # Calmar
        calmar = total_return / (max_drawdown * 100 + 1e-8)
        
        return BacktestResult(
            total_return=total_return,
            benchmark_return=buy_hold_return,
            excess_return=total_return - buy_hold_return,
            trades=trades,
            win_rate=win_rate,
            profit_factor=profit_factor,
            sharpe_ratio=sharpe,
            max_drawdown=max_drawdown,
            calmar_ratio=calmar,
            num_folds=num_folds,
            avg_fold_accuracy=avg_accuracy
        )