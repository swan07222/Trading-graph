# analysis/backtest.py
"""
Walk-Forward Backtesting System v3.1

FIXES:
- Correct config import
- Proper index alignment
- Import cleanup
- Better error handling
"""
import os
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict

from config.settings import CONFIG  # FIXED: correct import
from data.fetcher import DataFetcher
from data.features import FeatureEngine
from data.processor import DataProcessor
from models.ensemble import EnsembleModel
from utils.logger import get_logger
from core.constants import get_price_limit, get_lot_size, get_exchange  # FIXED: top-level import

log = get_logger(__name__)


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
        if daily_volume <= 0 or daily_avg_price <= 0 or np.isnan(daily_volume) or np.isnan(daily_avg_price):
            return self.base_slippage
        
        daily_value = daily_volume * daily_avg_price
        if daily_value <= 0:
            return self.base_slippage
            
        order_pct = order_value / daily_value
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
        stocks = stock_codes or CONFIG.stock_pool[:5]
        capital = initial_capital or CONFIG.capital
        
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
            raise ValueError("No predictions generated during backtest. Check model and data.")
        
        # Calculate properly time-aligned returns
        sorted_dates = sorted(daily_returns_by_date.keys())
        
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
                
                if df is None or df.empty:
                    log.warning(f"No data for {code}")
                    continue
                
                if len(df) < min_days:
                    log.warning(f"Insufficient data for {code}: {len(df)} days")
                    continue
                
                df = self.feature_engine.create_features(df)
                
                if len(df) >= CONFIG.model.sequence_length + 50:
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
        embargo_days = CONFIG.model.embargo_bars
        
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
            
            if len(fold_df) >= CONFIG.model.sequence_length:
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
            
            if len(fold_df) >= CONFIG.model.sequence_length + 10:
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
        returns_by_date = defaultdict(list)
        benchmark_by_date = defaultdict(list)
        predictions = []
        actuals = []
        
        for code, df in all_data.items():
            mask = (df.index >= test_start) & (df.index < test_end)
            fold_raw = df.loc[mask].copy()
            fold_df = processor.create_labels(fold_raw)
            
            if len(fold_df) < CONFIG.model.sequence_length + 5:
                continue
            
            X, y, returns, idx = processor.prepare_sequences(
                fold_df, feature_cols, fit_scaler=False, return_index=True
            )
            if len(X) == 0:
                continue

            # FIXED: Safe index alignment
            common_idx = fold_df.index.intersection(idx)
            if len(common_idx) == 0:
                continue
            aligned = fold_df.loc[common_idx]
            
            # Re-filter X, y, returns to match common_idx
            idx_mask = idx.isin(common_idx)
            X = X[idx_mask]
            y = y[idx_mask]
            returns = returns[idx_mask]
            idx = idx[idx_mask]

            if len(X) == 0:
                continue

            code_trades, code_returns, code_benchmark = self._simulate_trading(
                model=model,
                X=X,
                y=y,
                returns=returns,
                dates=idx,
                open_prices=aligned["open"].values,
                close_prices=aligned["close"].values,
                volumes=aligned["volume"].values,
                stock_code=code,
                capital=capital / len(all_data)
            )
            
            trades.extend(code_trades)
            
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
        open_prices: np.ndarray,
        close_prices: np.ndarray,
        volumes: np.ndarray,
        stock_code: str,
        capital: float
    ) -> Tuple[List[BacktestTrade], Dict, Dict]:
        """
        NEXT-BAR execution without lookahead + realistic CN costs.
        """
        slippage_model = SlippageModel()
        lot = int(get_lot_size(stock_code))

        trades: List[BacktestTrade] = []

        cash = float(capital)
        shares = 0
        entry_price = 0.0
        entry_exec_i: Optional[int] = None
        pending_signal: Optional[Tuple[str, float, pd.Timestamp]] = None

        daily_portfolio_values: Dict[pd.Timestamp, float] = {}
        daily_benchmark_values: Dict[pd.Timestamp, float] = {}

        horizon = int(CONFIG.model.prediction_horizon)
        limit_pct = float(get_price_limit(stock_code))

        # Costs
        commission_rate = float(CONFIG.trading.commission)
        stamp_tax_rate = float(CONFIG.trading.stamp_tax)
        commission_min = 5.0
        is_sse = (get_exchange(str(stock_code).zfill(6)) == "SSE")
        transfer_fee_rate = 0.00002 if is_sse else 0.0

        def commission(notional: float) -> float:
            return max(commission_min, notional * commission_rate) if notional > 0 else 0.0

        def transfer_fee(notional: float) -> float:
            return notional * transfer_fee_rate if notional > 0 else 0.0

        def is_limit_up(prev_close: float, px: float) -> bool:
            if prev_close <= 0:
                return False
            return px >= prev_close * (1.0 + limit_pct - 1e-4)

        def is_limit_down(prev_close: float, px: float) -> bool:
            if prev_close <= 0:
                return False
            return px <= prev_close * (1.0 - limit_pct + 1e-4)

        # FIXED: Benchmark buys at first OPEN, not close
        first_open = float(open_prices[0]) if len(open_prices) > 0 else 0.0
        benchmark_shares = (capital / first_open) if first_open > 0 else 0.0

        preds = model.predict_batch(X)

        n = min(len(dates), len(open_prices), len(close_prices), len(volumes), len(preds))
        if n == 0:
            return [], {}, {}

        for t in range(n):
            dt = dates[t]
            open_t = float(open_prices[t])
            close_t = float(close_prices[t])
            prev_close = float(close_prices[t - 1]) if t > 0 else close_t
            
            # Handle NaN/invalid prices
            if np.isnan(open_t) or np.isnan(close_t) or open_t <= 0 or close_t <= 0:
                continue

            # 1) Execute pending at OPEN
            if pending_signal is not None:
                action, signal_conf, signal_dt = pending_signal
                pending_signal = None

                if action == "BUY" and shares == 0:
                    if not is_limit_up(prev_close, open_t):
                        invest = cash * 0.95
                        vol = float(volumes[t]) if not np.isnan(volumes[t]) and volumes[t] > 0 else 1e6
                        slip = slippage_model.calculate(invest, vol, open_t)
                        buy_px = open_t * (1.0 + slip)

                        qty = int(invest / buy_px / lot) * lot
                        if qty > 0:
                            notional = qty * buy_px
                            fee = commission(notional) + transfer_fee(notional)
                            total = notional + fee

                            if total <= cash:
                                cash -= total
                                shares = qty
                                entry_price = buy_px
                                entry_exec_i = t

                                trades.append(BacktestTrade(
                                    entry_date=signal_dt,
                                    exit_date=None,
                                    stock_code=stock_code,
                                    side="buy",
                                    entry_price=entry_price,
                                    quantity=shares,
                                    signal_confidence=float(signal_conf)
                                ))

                elif action == "SELL" and shares > 0:
                    # T+1 check
                    if entry_exec_i is not None and t == entry_exec_i:
                        pass  # Can't sell same day
                    else:
                        if not is_limit_down(prev_close, open_t):
                            notional = shares * open_t
                            vol = float(volumes[t]) if not np.isnan(volumes[t]) and volumes[t] > 0 else 1e6
                            slip = slippage_model.calculate(notional, vol, open_t)
                            sell_px = open_t * (1.0 - slip)

                            proceeds = shares * sell_px
                            fee = commission(proceeds) + transfer_fee(proceeds)
                            tax = proceeds * stamp_tax_rate
                            net = proceeds - fee - tax
                            cash += net

                            if trades:
                                holding_bars = (t - entry_exec_i) if entry_exec_i is not None else 0
                                gross_pnl = (sell_px - entry_price) * shares

                                buy_notional = entry_price * shares
                                sell_notional = sell_px * shares
                                costs = (
                                    commission(buy_notional) + transfer_fee(buy_notional) +
                                    commission(sell_notional) + transfer_fee(sell_notional) +
                                    sell_notional * stamp_tax_rate
                                )
                                net_pnl = gross_pnl - costs
                                pnl_pct = (sell_px / entry_price - 1.0) * 100.0 - (costs / max(1e-8, buy_notional)) * 100.0

                                trades[-1].exit_date = dt
                                trades[-1].exit_price = sell_px
                                trades[-1].pnl = float(net_pnl)
                                trades[-1].pnl_pct = float(pnl_pct)
                                trades[-1].holding_days = int(holding_bars)

                            shares = 0
                            entry_price = 0.0
                            entry_exec_i = None

            # 2) Mark-to-market at CLOSE
            portfolio_value = cash + shares * close_t
            benchmark_value = benchmark_shares * close_t
            daily_portfolio_values[dt] = float(portfolio_value)
            daily_benchmark_values[dt] = float(benchmark_value)

            # 3) Signal at CLOSE for next OPEN
            if t < n - 1:
                pred = preds[t]
                if shares == 0 and pred.predicted_class == 2 and pred.confidence >= CONFIG.model.min_confidence:
                    pending_signal = ("BUY", float(pred.confidence), dt)
                elif shares > 0:
                    holding_bars = (t - entry_exec_i) if entry_exec_i is not None else 0
                    should_exit = (holding_bars >= horizon) or (pred.predicted_class == 0 and pred.confidence >= CONFIG.model.min_confidence)
                    if should_exit:
                        pending_signal = ("SELL", float(pred.confidence), dt)

        # Force close at last bar if still holding
        if shares > 0 and trades:
            dt = dates[n - 1]
            close_t = float(close_prices[n - 1])

            if not np.isnan(close_t) and close_t > 0:
                proceeds = shares * close_t
                fee = commission(proceeds) + transfer_fee(proceeds)
                tax = proceeds * stamp_tax_rate
                net = proceeds - fee - tax
                cash += net

                holding_bars = ((n - 1) - entry_exec_i) if entry_exec_i is not None else 0
                gross_pnl = (close_t - entry_price) * shares

                buy_notional = entry_price * shares
                sell_notional = close_t * shares
                costs = (
                    commission(buy_notional) + transfer_fee(buy_notional) +
                    commission(sell_notional) + transfer_fee(sell_notional) +
                    sell_notional * stamp_tax_rate
                )
                net_pnl = gross_pnl - costs
                pnl_pct = (close_t / entry_price - 1.0) * 100.0 - (costs / max(1e-8, buy_notional)) * 100.0

                trades[-1].exit_date = dt
                trades[-1].exit_price = close_t
                trades[-1].pnl = float(net_pnl)
                trades[-1].pnl_pct = float(pnl_pct)
                trades[-1].holding_days = int(holding_bars)

        # Daily returns (%)
        sorted_dates = sorted(daily_portfolio_values.keys())
        strategy_returns: Dict[pd.Timestamp, float] = {}
        benchmark_returns: Dict[pd.Timestamp, float] = {}

        for i, dt in enumerate(sorted_dates):
            if i == 0:
                strategy_returns[dt] = 0.0
                benchmark_returns[dt] = 0.0
                continue

            prev_dt = sorted_dates[i - 1]
            prev_val = daily_portfolio_values[prev_dt]
            curr_val = daily_portfolio_values[dt]
            prev_b = daily_benchmark_values[prev_dt]
            curr_b = daily_benchmark_values[dt]

            strategy_returns[dt] = ((curr_val / prev_val) - 1.0) * 100.0 if prev_val > 0 else 0.0
            benchmark_returns[dt] = ((curr_b / prev_b) - 1.0) * 100.0 if prev_b > 0 else 0.0

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
        
        # Benchmark calculation (compounded)
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
        
        # Calmar ratio (capped)
        if max_dd_pct > 0.01:  # Minimum 0.01% to avoid huge ratios
            calmar = total_return / max_dd_pct
        else:
            calmar = 0
        
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