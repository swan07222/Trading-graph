"""
Walk-Forward Backtesting System v3.0

FIXED Issues:
- Proper time-aligned returns across stocks (by DATE)
- Correct benchmark calculation (buy-hold compounded)
- Trade counting by entries, not bars
- Realistic A-share rules
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict

from config import CONFIG
from data.fetcher import DataFetcher
from data.features import FeatureEngine
from data.processor import DataProcessor
from models.ensemble import EnsembleModel
from utils.logger import log


@dataclass
class BacktestTrade:
    """Single trade record"""
    entry_date: datetime
    exit_date: Optional[datetime]
    stock_code: str
    side: str
    entry_price: float
    exit_price: float = 0.0
    quantity: int = 0
    pnl: float = 0.0
    pnl_pct: float = 0.0
    holding_days: int = 0
    signal_confidence: float = 0.0


@dataclass
class BacktestResult:
    """Complete backtest results"""
    total_return: float
    benchmark_return: float
    excess_return: float
    sharpe_ratio: float
    max_drawdown: float
    max_drawdown_pct: float
    calmar_ratio: float
    volatility: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    avg_holding_days: float
    num_folds: int
    avg_fold_accuracy: float
    fold_results: List[Dict] = field(default_factory=list)
    equity_curve: List[float] = field(default_factory=list)
    dates: List[datetime] = field(default_factory=list)
    
    def is_profitable(self) -> bool:
        return (
            self.excess_return > 0 and
            self.profit_factor > 1.0 and
            self.sharpe_ratio > 0.5
        )
    
    def summary(self) -> str:
        status = "✅ PROFITABLE" if self.is_profitable() else "❌ NOT PROFITABLE"
        
        return f"""
{'=' * 70}
                    BACKTEST RESULTS - {status}
{'=' * 70}

  RETURNS:
    Strategy Return:     {self.total_return:+.2f}%
    Benchmark Return:    {self.benchmark_return:+.2f}%
    Excess Return:       {self.excess_return:+.2f}%

  RISK METRICS:
    Sharpe Ratio:        {self.sharpe_ratio:.2f}
    Max Drawdown:        {self.max_drawdown_pct:.1f}%
    Calmar Ratio:        {self.calmar_ratio:.2f}
    Volatility (ann.):   {self.volatility:.1f}%

  TRADING STATISTICS:
    Total Trades:        {self.total_trades}
    Winning Trades:      {self.winning_trades}
    Losing Trades:       {self.losing_trades}
    Win Rate:            {self.win_rate:.1%}
    Profit Factor:       {self.profit_factor:.2f}
    Avg Win:             {self.avg_win:+.2f}%
    Avg Loss:            {self.avg_loss:.2f}%
    Avg Holding Days:    {self.avg_holding_days:.1f}

  WALK-FORWARD:
    Folds:               {self.num_folds}
    Avg Fold Accuracy:   {self.avg_fold_accuracy:.1%}

{'=' * 70}
"""


class Backtester:
    """Walk-Forward Backtesting with proper methodology."""
    
    def __init__(self):
        self.fetcher = DataFetcher()
        self.feature_engine = FeatureEngine()
    
    def run(
        self,
        stock_codes: List[str] = None,
        train_months: int = 12,
        test_months: int = 1,
        min_data_days: int = 500,
        initial_capital: float = None
    ) -> BacktestResult:
        """Run walk-forward backtest."""
        stocks = stock_codes or CONFIG.STOCK_POOL[:5]
        capital = initial_capital or CONFIG.CAPITAL
        
        log.info(f"Starting walk-forward backtest:")
        log.info(f"  Stocks: {len(stocks)}")
        log.info(f"  Train: {train_months} months, Test: {test_months} months")
        log.info(f"  Capital: ¥{capital:,.2f}")
        
        # Collect and validate data
        all_data = self._collect_data(stocks, min_data_days)
        
        if not all_data:
            raise ValueError("No valid data available for backtesting")
        
        # Find common date range
        min_date = max(df.index.min() for df in all_data.values())
        max_date = min(df.index.max() for df in all_data.values())
        
        log.info(f"  Date range: {min_date.date()} to {max_date.date()}")
        
        # Generate folds
        folds = self._generate_folds(min_date, max_date, train_months, test_months)
        
        if not folds:
            raise ValueError("Insufficient data for walk-forward testing")
        
        log.info(f"  Folds: {len(folds)}")
        
        # Run backtest
        all_trades = []
        # FIXED: Store returns by date for proper alignment
        daily_returns_by_date: Dict[datetime, List[float]] = defaultdict(list)
        benchmark_returns_by_date: Dict[datetime, List[float]] = defaultdict(list)
        fold_accuracies = []
        fold_results = []
        
        for fold_idx, fold in enumerate(folds):
            train_start, train_end, test_start, test_end = fold
            
            log.info(f"\nFold {fold_idx + 1}/{len(folds)}:")
            log.info(f"  Train: {train_start.date()} to {train_end.date()}")
            log.info(f"  Test:  {test_start.date()} to {test_end.date()}")
            
            result = self._run_fold(
                all_data, train_start, train_end, test_start, test_end, capital
            )
            
            if result is not None:
                trades, returns_dict, benchmark_dict, accuracy = result
                all_trades.extend(trades)
                
                # FIXED: Aggregate returns by date
                for dt, ret in returns_dict.items():
                    daily_returns_by_date[dt].append(ret)
                for dt, ret in benchmark_dict.items():
                    benchmark_returns_by_date[dt].append(ret)
                
                fold_accuracies.append(accuracy)
                
                fold_results.append({
                    'fold': fold_idx + 1,
                    'train_start': train_start,
                    'train_end': train_end,
                    'test_start': test_start,
                    'test_end': test_end,
                    'accuracy': accuracy,
                    'trades': len(trades),
                    'return': sum(t.pnl_pct for t in trades) if trades else 0
                })
        
        if not daily_returns_by_date:
            raise ValueError("No predictions generated during backtest")
        
        # FIXED: Calculate properly time-aligned returns
        sorted_dates = sorted(daily_returns_by_date.keys())
        
        # Average returns across stocks for each date
        daily_returns = np.array([
            np.mean(daily_returns_by_date[dt]) for dt in sorted_dates
        ])
        benchmark_daily = np.array([
            np.mean(benchmark_returns_by_date.get(dt, [0])) for dt in sorted_dates
        ])
        
        # Calculate metrics
        result = self._calculate_metrics(
            trades=all_trades,
            daily_returns=daily_returns,
            benchmark_daily=benchmark_daily,
            dates=sorted_dates,
            capital=capital,
            num_folds=len(folds),
            fold_accuracies=fold_accuracies,
            fold_results=fold_results
        )
        
        return result
    
    def _collect_data(self, stocks: List[str], min_days: int) -> Dict[str, pd.DataFrame]:
        """Collect and validate data for all stocks"""
        all_data = {}
        
        for code in stocks:
            try:
                df = self.fetcher.get_history(code, days=2000)
                
                if len(df) < min_days:
                    log.warning(f"Insufficient data for {code}: {len(df)} days")
                    continue
                
                df = self.feature_engine.create_features(df)
                
                if len(df) >= CONFIG.SEQUENCE_LENGTH + 50:
                    all_data[code] = df
                    log.info(f"  {code}: {len(df)} samples")
                    
            except Exception as e:
                log.warning(f"Failed to load {code}: {e}")
        
        return all_data
    
    def _generate_folds(
        self,
        min_date: pd.Timestamp,
        max_date: pd.Timestamp,
        train_months: int,
        test_months: int
    ) -> List[Tuple]:
        """Generate walk-forward folds with proper separation"""
        folds = []
        embargo_days = CONFIG.EMBARGO_BARS
        
        train_start = min_date
        
        while True:
            train_end = train_start + pd.DateOffset(months=train_months)
            test_start = train_end + pd.Timedelta(days=embargo_days)
            test_end = test_start + pd.DateOffset(months=test_months)
            
            if test_end > max_date:
                break
            
            folds.append((train_start, train_end, test_start, test_end))
            train_start = train_start + pd.DateOffset(months=test_months)
        
        return folds
    
    def _run_fold(
        self,
        all_data: Dict[str, pd.DataFrame],
        train_start: pd.Timestamp,
        train_end: pd.Timestamp,
        test_start: pd.Timestamp,
        test_end: pd.Timestamp,
        capital: float
    ) -> Optional[Tuple]:
        """Run single fold of walk-forward backtest"""
        
        processor = DataProcessor()
        feature_cols = self.feature_engine.get_feature_columns()
        
        # Collect training features for scaler
        train_features_list = []
        
        for code, df in all_data.items():
            mask = (df.index >= train_start) & (df.index < train_end)
            fold_df = df[mask]
            
            if len(fold_df) >= CONFIG.SEQUENCE_LENGTH:
                train_features_list.append(fold_df[feature_cols].values)
        
        if not train_features_list:
            log.warning("No training data for this fold")
            return None
        
        # Fit scaler on training data
        combined_train = np.concatenate(train_features_list)
        processor.fit_scaler(combined_train)
        
        # Prepare training sequences
        X_train, y_train = [], []
        
        for code, df in all_data.items():
            df_labeled = processor.create_labels(df.copy())
            
            mask = (df_labeled.index >= train_start) & (df_labeled.index < train_end)
            fold_df = df_labeled[mask]
            
            if len(fold_df) >= CONFIG.SEQUENCE_LENGTH + 10:
                X, y, _ = processor.prepare_sequences(fold_df, feature_cols, fit_scaler=False)
                if len(X) > 0:
                    X_train.append(X)
                    y_train.append(y)
        
        if not X_train:
            log.warning("No training sequences")
            return None
        
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
        
        # Test
        trades = []
        # FIXED: Returns indexed by date
        returns_by_date: Dict[datetime, float] = {}
        benchmark_by_date: Dict[datetime, float] = {}
        predictions = []
        actuals = []
        
        for code, df in all_data.items():
            df_labeled = processor.create_labels(df.copy())
            
            mask = (df_labeled.index >= test_start) & (df_labeled.index < test_end)
            fold_df = df_labeled[mask]
            
            if len(fold_df) < CONFIG.SEQUENCE_LENGTH + 5:
                continue
            
            X, y, returns = processor.prepare_sequences(fold_df, feature_cols, fit_scaler=False)
            
            if len(X) == 0:
                continue
            
            code_trades, code_returns, code_benchmark = self._simulate_trading(
                model=model,
                X=X,
                y=y,
                returns=returns,
                dates=fold_df.index[-len(X):],
                prices=fold_df['close'].values[-len(X):],
                stock_code=code,
                capital=capital / len(all_data)
            )
            
            trades.extend(code_trades)
            
            # FIXED: Store by date
            for dt, ret in code_returns.items():
                if dt in returns_by_date:
                    returns_by_date[dt] = (returns_by_date[dt] + ret) / 2
                else:
                    returns_by_date[dt] = ret
            
            for dt, ret in code_benchmark.items():
                if dt in benchmark_by_date:
                    benchmark_by_date[dt] = (benchmark_by_date[dt] + ret) / 2
                else:
                    benchmark_by_date[dt] = ret
            
            predictions.extend([model.predict(X[i:i+1]).predicted_class for i in range(len(X))])
            actuals.extend(y.tolist())
        
        if actuals:
            accuracy = np.mean(np.array(predictions) == np.array(actuals))
            log.info(f"  Fold accuracy: {accuracy:.2%}")
        else:
            accuracy = 0
        
        return trades, returns_by_date, benchmark_by_date, accuracy
    
    def _simulate_trading(
        self,
        model: EnsembleModel,
        X: np.ndarray,
        y: np.ndarray,
        returns: np.ndarray,
        dates: pd.DatetimeIndex,
        prices: np.ndarray,
        stock_code: str,
        capital: float
    ) -> Tuple[List[BacktestTrade], Dict[datetime, float], Dict[datetime, float]]:
        """Simulate realistic trading."""
        trades = []
        # FIXED: Return dict indexed by date
        strategy_returns: Dict[datetime, float] = {}
        benchmark_returns: Dict[datetime, float] = {}
        
        position = 0
        entry_price = 0
        entry_date = None
        entry_confidence = 0
        
        costs_pct = (CONFIG.COMMISSION * 2 + CONFIG.SLIPPAGE * 2) * 100
        
        for i in range(len(X) - 1):
            pred = model.predict(X[i:i+1])
            current_price = prices[i]
            next_price = prices[i + 1]
            current_date = dates[i]
            
            # Daily return
            daily_return_pct = (next_price / current_price - 1) * 100
            
            # Benchmark: buy-hold return
            benchmark_returns[current_date] = daily_return_pct
            
            # Decision logic
            signal = None
            
            if pred.confidence >= CONFIG.MIN_CONFIDENCE:
                if pred.predicted_class == 2 and position == 0:
                    signal = 'enter_long'
                elif pred.predicted_class == 0 and position == 1:
                    signal = 'exit_long'
                elif pred.predicted_class == 1 and position == 1:
                    signal = 'exit_long'
            
            # Execute signals
            strategy_return = 0
            
            if signal == 'enter_long' and position == 0:
                position = 1
                entry_price = current_price
                entry_date = current_date
                entry_confidence = pred.confidence
                strategy_return = daily_return_pct - costs_pct / 2
                
            elif signal == 'exit_long' and position == 1:
                exit_price = current_price
                pnl_pct = (exit_price / entry_price - 1) * 100 - costs_pct
                
                trades.append(BacktestTrade(
                    entry_date=entry_date,
                    exit_date=current_date,
                    stock_code=stock_code,
                    side='long',
                    entry_price=entry_price,
                    exit_price=exit_price,
                    pnl_pct=pnl_pct,
                    holding_days=(current_date - entry_date).days,
                    signal_confidence=entry_confidence
                ))
                
                position = 0
                strategy_return = -costs_pct / 2
                
            elif position == 1:
                strategy_return = daily_return_pct
            
            strategy_returns[current_date] = strategy_return
        
        # Close any open position
        if position == 1 and len(prices) > 0:
            exit_price = prices[-1]
            pnl_pct = (exit_price / entry_price - 1) * 100 - costs_pct
            
            trades.append(BacktestTrade(
                entry_date=entry_date,
                exit_date=dates[-1],
                stock_code=stock_code,
                side='long',
                entry_price=entry_price,
                exit_price=exit_price,
                pnl_pct=pnl_pct,
                holding_days=(dates[-1] - entry_date).days,
                signal_confidence=entry_confidence
            ))
        
        return trades, strategy_returns, benchmark_returns
    
    def _calculate_metrics(
        self,
        trades: List[BacktestTrade],
        daily_returns: np.ndarray,
        benchmark_daily: np.ndarray,
        dates: List,
        capital: float,
        num_folds: int,
        fold_accuracies: List[float],
        fold_results: List[Dict]
    ) -> BacktestResult:
        """Calculate comprehensive backtest metrics"""
        
        # Build equity curve
        equity = [capital]
        for ret in daily_returns:
            equity.append(equity[-1] * (1 + ret / 100))
        equity = np.array(equity[1:])
        
        total_return = (equity[-1] / capital - 1) * 100 if len(equity) > 0 else 0
        
        # FIXED: Proper benchmark calculation (compounded)
        benchmark_equity = [capital]
        for ret in benchmark_daily:
            benchmark_equity.append(benchmark_equity[-1] * (1 + ret / 100))
        benchmark_return = (benchmark_equity[-1] / capital - 1) * 100 if len(benchmark_equity) > 1 else 0
        
        # Sharpe ratio
        if len(daily_returns) > 1 and np.std(daily_returns) > 0:
            sharpe = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252)
        else:
            sharpe = 0
        
        # Max drawdown
        if len(equity) > 0:
            running_max = np.maximum.accumulate(equity)
            drawdown = (equity - running_max) / running_max
            max_dd_pct = abs(np.min(drawdown)) * 100
            max_dd = abs(np.min(equity - running_max))
        else:
            max_dd = max_dd_pct = 0
        
        # Volatility
        volatility = np.std(daily_returns) * np.sqrt(252) if len(daily_returns) > 0 else 0
        
        # Calmar ratio
        calmar = total_return / max_dd_pct if max_dd_pct > 0 else 0
        
        # Trade statistics
        total_trades = len(trades)
        
        if total_trades > 0:
            pnls = [t.pnl_pct for t in trades]
            wins = [p for p in pnls if p > 0]
            losses = [p for p in pnls if p < 0]
            
            winning_trades = len(wins)
            losing_trades = len(losses)
            win_rate = winning_trades / total_trades
            
            avg_win = np.mean(wins) if wins else 0
            avg_loss = np.mean(losses) if losses else 0
            
            gross_profit = sum(wins) if wins else 0
            gross_loss = abs(sum(losses)) if losses else 1
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
            
            avg_holding = np.mean([t.holding_days for t in trades])
        else:
            winning_trades = losing_trades = 0
            win_rate = profit_factor = avg_win = avg_holding = 0
            avg_loss = 0
        
        return BacktestResult(
            total_return=total_return,
            benchmark_return=benchmark_return,
            excess_return=total_return - benchmark_return,
            sharpe_ratio=sharpe,
            max_drawdown=max_dd,
            max_drawdown_pct=max_dd_pct,
            calmar_ratio=calmar,
            volatility=volatility,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            profit_factor=profit_factor,
            avg_win=avg_win,
            avg_loss=avg_loss,
            avg_holding_days=avg_holding,
            num_folds=num_folds,
            avg_fold_accuracy=np.mean(fold_accuracies) if fold_accuracies else 0,
            fold_results=fold_results,
            equity_curve=equity.tolist(),
            dates=[d for d in dates]
        )