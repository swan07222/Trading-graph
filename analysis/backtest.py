"""
Backtesting System
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
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

{'=' * 60}
"""


class Backtester:
    """
    Walk-forward backtesting system
    """
    
    def __init__(self):
        self.fetcher = DataFetcher()
        self.feature_engine = FeatureEngine()
        self.processor = DataProcessor()
    
    def run(self,
            stock_codes: List[str] = None,
            train_months: int = 12,
            test_months: int = 1) -> BacktestResult:
        """
        Run walk-forward backtest
        """
        stocks = stock_codes or CONFIG.STOCK_POOL[:5]
        
        log.info(f"Running backtest on {len(stocks)} stocks...")
        log.info(f"Train: {train_months} months, Test: {test_months} month(s)")
        
        # Collect data
        all_data = {}
        for code in stocks:
            try:
                df = self.fetcher.get_history(code, days=1500)
                if len(df) >= CONFIG.SEQUENCE_LENGTH + 50:
                    df = self.feature_engine.create_features(df)
                    df = self.processor.create_labels(df)
                    all_data[code] = df
            except Exception as e:
                log.warning(f"Failed to load {code}: {e}")
        
        if not all_data:
            raise ValueError("No data available")
        
        # Find common date range
        min_date = max(df.index.min() for df in all_data.values())
        max_date = min(df.index.max() for df in all_data.values())
        
        log.info(f"Date range: {min_date.date()} to {max_date.date()}")
        
        # Generate folds
        folds = self._generate_folds(min_date, max_date, train_months, test_months)
        log.info(f"Folds: {len(folds)}")
        
        # Run backtest
        all_preds = []
        all_labels = []
        all_returns = []
        all_confs = []
        
        for fold_idx, (train_start, train_end, test_start, test_end) in enumerate(folds):
            log.info(f"Fold {fold_idx + 1}/{len(folds)}")
            
            # Prepare training data
            X_train, y_train = [], []
            for code, df in all_data.items():
                mask = (df.index >= train_start) & (df.index < train_end)
                fold_df = df[mask]
                
                if len(fold_df) >= CONFIG.SEQUENCE_LENGTH + 10:
                    feature_cols = self.feature_engine.get_feature_columns()
                    X, y, _ = self.processor.prepare_sequences(fold_df, feature_cols)
                    if len(X) > 0:
                        X_train.append(X)
                        y_train.append(y)
            
            if not X_train:
                continue
            
            X_train = np.concatenate(X_train)
            y_train = np.concatenate(y_train)
            
            # Train model
            input_size = X_train.shape[2]
            model = EnsembleModel(input_size, model_names=['lstm', 'gru', 'tcn'])
            
            split = int(len(X_train) * 0.85)
            model.train(
                X_train[:split], y_train[:split],
                X_train[split:], y_train[split:],
                epochs=30
            )
            
            # Test data
            X_test, y_test, r_test = [], [], []
            for code, df in all_data.items():
                mask = (df.index >= test_start) & (df.index < test_end)
                fold_df = df[mask]
                
                if len(fold_df) >= CONFIG.SEQUENCE_LENGTH + 5:
                    feature_cols = self.feature_engine.get_feature_columns()
                    X, y, r = self.processor.prepare_sequences(fold_df, feature_cols)
                    if len(X) > 0:
                        X_test.append(X)
                        y_test.append(y)
                        r_test.append(r)
            
            if not X_test:
                continue
            
            X_test = np.concatenate(X_test)
            y_test = np.concatenate(y_test)
            r_test = np.concatenate(r_test)
            
            # Predict
            for i in range(len(X_test)):
                pred = model.predict(X_test[i:i+1])
                all_preds.append(pred.predicted_class)
                all_confs.append(pred.confidence)
            
            all_labels.extend(y_test)
            all_returns.extend(r_test)
        
        # Compute metrics
        return self._compute_metrics(
            np.array(all_preds),
            np.array(all_labels),
            np.array(all_returns),
            np.array(all_confs)
        )
    
    def _generate_folds(self, min_date, max_date, train_months: int, test_months: int) -> List[Tuple]:
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
            train_start = train_start + pd.DateOffset(months=test_months)
        
        return folds
    
    def _compute_metrics(self,
                         preds: np.ndarray,
                         labels: np.ndarray,
                         returns: np.ndarray,
                         confs: np.ndarray) -> BacktestResult:
        """Compute backtest metrics"""
        # Position based on prediction
        position = np.zeros(len(preds))
        position[preds == 2] = 1   # UP -> Long
        position[preds == 0] = -1  # DOWN -> Short/Exit
        
        # Only trade when confident
        position = position * (confs >= CONFIG.MIN_CONFIDENCE)
        
        # Calculate returns
        costs = CONFIG.COMMISSION * 2 + CONFIG.SLIPPAGE * 2 + CONFIG.STAMP_TAX
        
        strategy_returns = position * returns / 100
        trade_costs = np.abs(np.diff(position, prepend=0)) * costs
        net_returns = strategy_returns - trade_costs
        
        # Benchmark
        buy_hold_returns = returns / 100
        
        # Cumulative
        cum_strategy = (1 + net_returns).cumprod()
        cum_buyhold = (1 + buy_hold_returns).cumprod()
        
        total_return = (cum_strategy[-1] - 1) * 100
        buy_hold_return = (cum_buyhold[-1] - 1) * 100
        
        # Trade stats
        trades = np.sum(position != 0)
        
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
        if net_returns.std() > 0:
            sharpe = net_returns.mean() / net_returns.std() * np.sqrt(252)
        else:
            sharpe = 0
        
        # Max drawdown
        running_max = np.maximum.accumulate(cum_strategy)
        drawdown = (cum_strategy - running_max) / running_max
        max_drawdown = abs(drawdown.min())
        
        # Calmar
        calmar = total_return / (max_drawdown * 100 + 1e-8)
        
        return BacktestResult(
            total_return=total_return,
            benchmark_return=buy_hold_return,
            excess_return=total_return - buy_hold_return,
            trades=int(trades),
            win_rate=win_rate,
            profit_factor=profit_factor,
            sharpe_ratio=sharpe,
            max_drawdown=max_drawdown,
            calmar_ratio=calmar
        )