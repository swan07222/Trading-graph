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
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional
from collections import defaultdict
from core.constants import get_price_limit
from dataclasses import dataclass, field


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
class SlippageModel:
    """Realistic slippage based on order size and liquidity"""
    base_slippage: float = 0.001  # 0.1%
    volume_impact: float = 0.1    # Additional slippage per 1% of daily volume
    
    def calculate(self, order_value: float, daily_volume: float, daily_avg_price: float) -> float:
        """Calculate slippage for an order"""
        if daily_volume <= 0 or daily_avg_price <= 0:
            return self.base_slippage
        
        daily_value = daily_volume * daily_avg_price
        order_pct = order_value / daily_value if daily_value > 0 else 0
        
        # Slippage increases with order size relative to liquidity
        slippage = self.base_slippage + self.volume_impact * order_pct
        
        return min(slippage, 0.05)  # Cap at 5%
@dataclass
class BacktestResult:
    """Complete backtest results"""
    total_return: float
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

    benchmark_return: float = 0.0
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
        status = "PROFITABLE" if self.is_profitable() else "NOT PROFITABLE"
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
                for dt, ret_list in returns_dict.items():
                    daily_returns_by_date[dt].extend(ret_list)

                for dt, ret_list in benchmark_dict.items():
                    benchmark_returns_by_date[dt].extend(ret_list)
                
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

            mask = (df.index >= train_start) & (df.index < train_end)
            fold_raw = df.loc[mask].copy()
            fold_df = processor.create_labels(fold_raw)
            
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
        returns_by_date = defaultdict(list)
        benchmark_by_date = defaultdict(list)
        predictions = []
        actuals = []
        
        for code, df in all_data.items():
            mask = (df.index >= test_start) & (df.index < test_end)
            fold_raw = df.loc[mask].copy()
            fold_df = processor.create_labels(fold_raw)  
            
            if len(fold_df) < CONFIG.SEQUENCE_LENGTH + 5:
                continue
            
            X, y, returns, idx = processor.prepare_sequences(
                fold_df, feature_cols, fit_scaler=False, return_index=True
            )
            if len(X) == 0:
                continue

            aligned = fold_df.loc[idx]
            
            code_trades, code_returns, code_benchmark = self._simulate_trading(
                model=model,
                X=X,
                y=y,
                returns=returns,
                dates=idx,
                prices=aligned["close"].values,
                volumes=aligned["volume"].values,
                stock_code=code,
                capital=capital / len(all_data)
            )
            
            trades.extend(code_trades)
            
            # FIXED: Store by date
            for dt, ret in code_returns.items():
                returns_by_date[dt].append(ret)

            for dt, ret in code_benchmark.items():
                benchmark_by_date[dt].append(ret)
            
            preds = model.predict_batch(X)
            predictions.extend([p.predicted_class for p in preds])
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
        volumes: np.ndarray,
        stock_code: str,
        capital: float
    ):
        """
        Simulate trading with NEXT-BAR execution.
        
        CRITICAL: Signal at bar i -> execute at bar i+1 open/close
        """
        slippage_model = SlippageModel()
        trades = []
        
        cash = capital
        shares = 0
        entry_price = 0.0
        entry_date = None
        pending_signal = None  # Signal waiting for next bar execution
        
        daily_portfolio_values = {}
        daily_benchmark_values = {}
        
        horizon = CONFIG.PREDICTION_HORIZON
        limit = get_price_limit(stock_code) * 100.0

        def is_limit_up(prev_close, close):
            return prev_close > 0 and (close / prev_close - 1) * 100 >= (limit - 0.01)

        def is_limit_down(prev_close, close):
            return prev_close > 0 and (close / prev_close - 1) * 100 <= (-limit + 0.01)

        benchmark_shares = capital / float(prices[0]) if prices[0] > 0 else 0

        for i in range(len(X) - 1):
            current_price = float(prices[i])
            next_price = float(prices[i + 1])  # Execution price
            dt = dates[i]
            prev_close = float(prices[i - 1]) if i > 0 else current_price

            # Portfolio value at current bar (before any execution)
            portfolio_value = cash + shares * current_price
            benchmark_value = benchmark_shares * current_price
            
            daily_portfolio_values[dt] = portfolio_value
            daily_benchmark_values[dt] = benchmark_value

            # Execute pending signal from PREVIOUS bar
            if pending_signal is not None:
                action, signal_conf, signal_dt = pending_signal
                pending_signal = None
                
                if action == 'BUY' and shares == 0:
                    if not is_limit_up(prev_close, current_price):
                        # Execute at current bar's price (which was next bar when signal generated)
                        vol = float(volumes[i]) if i < len(volumes) else 1e6
                        slip = slippage_model.calculate(capital * 0.1, vol, current_price)
                        buy_price = current_price * (1 + slip)
                        cost = capital * 0.95
                        shares_to_buy = int(cost / buy_price / 100) * 100
                        
                        if shares_to_buy > 0:
                            actual_cost = shares_to_buy * buy_price * (1 + CONFIG.COMMISSION)
                            cash -= actual_cost
                            shares = shares_to_buy
                            entry_price = buy_price
                            entry_date = dt
                            
                            trades.append(BacktestTrade(
                                entry_date=signal_dt,  # Signal date
                                exit_date=None,
                                stock_code=stock_code,
                                side="buy",
                                entry_price=entry_price,
                                quantity=shares,
                                signal_confidence=signal_conf
                            ))
                            
                elif action == 'SELL' and shares > 0:
                    if not is_limit_down(prev_close, current_price):
                        vol = float(volumes[i]) if i < len(volumes) else 1e6
                        slip = slippage_model.calculate(shares * current_price, vol, current_price)
                        sell_price = current_price * (1 - slip)
                        proceeds = shares * sell_price * (1 - CONFIG.COMMISSION - CONFIG.STAMP_TAX)
                        
                        holding_days = (dt - entry_date).days if entry_date else 0
                        gross_pnl = (sell_price - entry_price) * shares
                        costs = shares * entry_price * CONFIG.COMMISSION + shares * sell_price * (CONFIG.COMMISSION + CONFIG.STAMP_TAX)
                        net_pnl = gross_pnl - costs
                        pnl_pct = (sell_price / entry_price - 1) * 100 - (CONFIG.COMMISSION * 2 + CONFIG.STAMP_TAX) * 100
                        
                        cash += proceeds
                        
                        if trades:
                            trades[-1].exit_date = dt
                            trades[-1].exit_price = sell_price
                            trades[-1].pnl = net_pnl
                            trades[-1].pnl_pct = pnl_pct
                            trades[-1].holding_days = holding_days
                        
                        shares = 0
                        entry_price = 0.0
                        entry_date = None

            # Generate signal for NEXT bar (if we have prediction data)
            if i < len(X) - 2:  # Need at least one more bar for execution
                pred = preds[i]
                
                if shares == 0 and pred.predicted_class == 2 and pred.confidence >= CONFIG.MIN_CONFIDENCE:
                    pending_signal = ('BUY', pred.confidence, dt)
                elif shares > 0:
                    holding_days = (dt - entry_date).days if entry_date else 0
                    should_exit = False
                    
                    if holding_days >= horizon:
                        should_exit = True
                    elif pred.predicted_class == 0 and pred.confidence >= CONFIG.MIN_CONFIDENCE:
                        should_exit = True
                    
                    if should_exit:
                        pending_signal = ('SELL', pred.confidence, dt)

        # Convert to daily returns (rest of method unchanged)
        sorted_dates = sorted(daily_portfolio_values.keys())
        strategy_returns = {}
        benchmark_returns = {}
        
        for i, dt in enumerate(sorted_dates):
            if i == 0:
                strategy_returns[dt] = 0.0
                benchmark_returns[dt] = 0.0
            else:
                prev_dt = sorted_dates[i-1]
                prev_val = daily_portfolio_values[prev_dt]
                curr_val = daily_portfolio_values[dt]
                
                if prev_val > 0:
                    strategy_returns[dt] = (curr_val / prev_val - 1) * 100
                else:
                    strategy_returns[dt] = 0.0
                
                prev_bench = daily_benchmark_values[prev_dt]
                curr_bench = daily_benchmark_values[dt]
                if prev_bench > 0:
                    benchmark_returns[dt] = (curr_bench / prev_bench - 1) * 100
                else:
                    benchmark_returns[dt] = 0.0

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