# analysis/backtest.py
import os
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict

from config.settings import CONFIG
from data.fetcher import DataFetcher
from data.features import FeatureEngine
from data.processor import DataProcessor
from models.ensemble import EnsembleModel
from utils.logger import get_logger
from core.constants import get_price_limit, get_lot_size, get_exchange

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
    base_slippage: float = 0.001
    volume_impact: float = 0.1

    def calculate(self, order_value: float, daily_volume: float, daily_avg_price: float) -> float:
        if daily_volume <= 0 or daily_avg_price <= 0 or np.isnan(daily_volume) or np.isnan(daily_avg_price):
            return self.base_slippage

        daily_value = daily_volume * daily_avg_price
        if daily_value <= 0:
            return self.base_slippage

        order_pct = order_value / daily_value
        slippage = self.base_slippage + self.volume_impact * order_pct

        return min(slippage, 0.05)

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
        stocks = self._get_stock_list(stock_codes)
        capital = initial_capital or getattr(CONFIG, 'capital', 1000000)

        log.info(f"Starting walk-forward backtest:")
        log.info(f"  Stocks to test: {stocks}")
        log.info(f"  Train: {train_months} months, Test: {test_months} months")
        log.info(f"  Capital: ¥{capital:,.2f}")

        all_data = self._collect_data(stocks, min_data_days)

        if not all_data:
            error_msg = self._diagnose_data_issue(stocks)
            raise ValueError(error_msg)

        log.info(f"  Successfully loaded {len(all_data)} stocks")

        min_date = max(df.index.min() for df in all_data.values())
        max_date = min(df.index.max() for df in all_data.values())

        log.info(f"  Date range: {min_date.date()} to {max_date.date()}")

        folds = self._generate_folds(min_date, max_date, train_months, test_months)

        if not folds:
            raise ValueError(
                f"Insufficient data for walk-forward testing.\n"
                f"  Available date range: {min_date.date()} to {max_date.date()}\n"
                f"  Required: {train_months + test_months} months minimum\n"
                f"  Try reducing train_months or test_months, or use more historical data."
            )

        log.info(f"  Folds: {len(folds)}")

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

                for dt, ret in returns_dict.items():
                    daily_returns_by_date[dt].append(float(ret))

                for dt, ret in benchmark_dict.items():
                    benchmark_returns_by_date[dt].append(float(ret))

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

    def _get_stock_list(self, stock_codes: Optional[List[str]]) -> List[str]:
        """Get stock list with fallbacks"""
        if stock_codes:
            return stock_codes

        # Try CONFIG.stock_pool
        if hasattr(CONFIG, 'stock_pool') and CONFIG.stock_pool:
            return CONFIG.stock_pool[:5]

        # Try CONFIG.STOCK_POOL
        if hasattr(CONFIG, 'STOCK_POOL') and CONFIG.STOCK_POOL:
            return CONFIG.STOCK_POOL[:5]

        # Default test stocks (major Chinese stocks)
        default_stocks = [
            "600519",  # 贵州茅台
            "000858",  # 五粮液
            "601318",  # 中国平安
            "600036",  # 招商银行
            "000333",  # 美的集团
        ]
        log.warning(f"No stock pool configured, using default stocks: {default_stocks}")
        return default_stocks

    def _collect_data(self, stocks: List[str], min_days: int) -> Dict[str, pd.DataFrame]:
        """Collect RAW OHLCV only (NO features here to avoid fold leakage)."""
        all_data: Dict[str, pd.DataFrame] = {}
        errors = []

        for code in stocks:
            try:
                log.info(f"  Fetching data for {code}...")
                df = self.fetcher.get_history(code, days=2000, interval="1d")

                if df is None:
                    errors.append(f"{code}: No data returned (None)")
                    continue
                if df.empty:
                    errors.append(f"{code}: Empty dataframe returned")
                    continue

                required_cols = ['open', 'high', 'low', 'close', 'volume']
                missing_cols = [c for c in required_cols if c not in df.columns]
                if missing_cols:
                    errors.append(f"{code}: Missing columns {missing_cols}")
                    continue

                if len(df) < min_days:
                    errors.append(f"{code}: Only {len(df)} days (need {min_days})")
                    continue

                # Keep RAW; fold will compute features within split
                all_data[code] = df.sort_index()
                log.info(f"    ✓ {code}: {len(df)} raw rows ready")

            except Exception as e:
                errors.append(f"{code}: Exception - {str(e)}")
                log.warning(f"  Failed to load {code}: {e}")

        if errors:
            log.warning("Data collection issues:")
            for err in errors:
                log.warning(f"  - {err}")

        return all_data

    def _diagnose_data_issue(self, stocks: List[str]) -> str:
        """Provide detailed diagnosis of data issues"""
        issues = []

        issues.append("No valid data available for backtesting.\n")
        issues.append("Diagnosis:\n")

        for code in stocks[:3]:  # Check first 3 stocks
            try:
                df = self.fetcher.get_history(code, days=100)
                if df is None:
                    issues.append(f"  - {code}: DataFetcher returned None")
                elif df.empty:
                    issues.append(f"  - {code}: DataFetcher returned empty DataFrame")
                else:
                    issues.append(f"  - {code}: Got {len(df)} rows")
            except Exception as e:
                issues.append(f"  - {code}: Error - {e}")

        issues.append("\nPossible solutions:")
        issues.append("  1. Check internet connection")
        issues.append("  2. Verify stock codes are valid (e.g., 600519, 000858)")
        issues.append("  3. Check if data source (akshare/tushare) is working")
        issues.append("  4. Try running with --predict 600519 first to test data fetch")

        return "\n".join(issues)

    def _generate_folds(
        self,
        min_date: pd.Timestamp,
        max_date: pd.Timestamp,
        train_months: int,
        test_months: int
    ) -> List[Tuple]:
        """Generate walk-forward folds with proper separation"""
        folds = []

        embargo_days = 5  # default
        if hasattr(CONFIG, 'model') and hasattr(CONFIG.model, 'embargo_bars'):
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
        all_data: Dict[str, pd.DataFrame],   # RAW OHLCV now
        train_start: pd.Timestamp,
        train_end: pd.Timestamp,
        test_start: pd.Timestamp,
        test_end: pd.Timestamp,
        capital: float
    ) -> Optional[Tuple]:
        processor = DataProcessor()
        feature_cols = self.feature_engine.get_feature_columns()

        seq_length = int(getattr(getattr(CONFIG, "model", None), "sequence_length", 60))
        feature_lookback = int(getattr(getattr(CONFIG, "data", None), "feature_lookback", 60))

        # -------------------------
        # 1) Build TRAIN features (per-stock) and fit scaler on TRAIN only
        # -------------------------
        train_features_list = []
        train_split_data = {}  # code -> featured+labels df

        for code, raw in all_data.items():
            raw_train = raw.loc[(raw.index >= train_start) & (raw.index < train_end)].copy()
            if len(raw_train) < (seq_length + feature_lookback + 10):
                continue

            feat_train = self.feature_engine.create_features(raw_train)
            feat_train = processor.create_labels(feat_train)  # uses CONFIG horizon/thresholds

            valid = feat_train["label"].notna()
            if valid.sum() <= 10:
                continue

            train_features_list.append(feat_train.loc[valid, feature_cols].values)
            train_split_data[code] = feat_train

        if not train_features_list:
            log.warning("No training data for this fold")
            return None

        combined_train = np.concatenate(train_features_list, axis=0)
        processor.fit_scaler(combined_train)

        # -------------------------
        # 2) Build training sequences
        # -------------------------
        X_train_list, y_train_list = [], []

        for code, feat_train in train_split_data.items():
            if len(feat_train) >= seq_length + 10:
                X, y, _ = processor.prepare_sequences(feat_train, feature_cols, fit_scaler=False)
                if len(X) > 0:
                    X_train_list.append(X)
                    y_train_list.append(y)

        if not X_train_list:
            log.warning("No training sequences for this fold")
            return None

        X_train = np.concatenate(X_train_list, axis=0)
        y_train = np.concatenate(y_train_list, axis=0)

        # -------------------------
        # 3) Train model
        # -------------------------
        input_size = int(X_train.shape[2])
        model = EnsembleModel(input_size, model_names=["lstm", "gru", "tcn"])

        split = int(len(X_train) * 0.85)
        model.train(
            X_train[:split], y_train[:split],
            X_train[split:], y_train[split:],
            epochs=30
        )

        # -------------------------
        # 4) Prepare TEST (compute features inside fold, with lookback context)
        # -------------------------
        trades: List[BacktestTrade] = []
        predictions, actuals = [], []

        portfolio_values_by_date: Dict[pd.Timestamp, float] = defaultdict(float)
        benchmark_values_by_date: Dict[pd.Timestamp, float] = defaultdict(float)

        valid_test_codes = []
        prepared = {}

        # Need enough context so rolling features can warm up BEFORE test_start
        ctx_days = int((seq_length + feature_lookback + 10) * 2)

        for code, raw in all_data.items():
            ctx_start = test_start - pd.Timedelta(days=ctx_days)
            raw_ctx = raw.loc[(raw.index >= ctx_start) & (raw.index < test_end)].copy()
            if len(raw_ctx) < seq_length + feature_lookback + 10:
                continue

            feat = self.feature_engine.create_features(raw_ctx)
            feat = processor.create_labels(feat)

            X, y, returns, idx = processor.prepare_sequences(
                feat, feature_cols, fit_scaler=False, return_index=True
            )
            if len(X) == 0:
                continue

            test_mask = (idx >= test_start) & (idx < test_end)
            X = X[test_mask]
            y = y[test_mask]
            returns = returns[test_mask]
            idx = idx[test_mask]

            if len(X) == 0:
                continue

            aligned = feat.loc[idx]
            if aligned.empty:
                continue

            valid_test_codes.append(code)
            prepared[code] = (X, y, returns, idx, aligned)

        if not valid_test_codes:
            log.warning("No valid test stocks for this fold")
            return None

        cap_slice = float(capital) / float(len(valid_test_codes))

        for code in valid_test_codes:
            X, y, returns, idx, aligned = prepared[code]

            preds = model.predict_batch(X)
            predictions.extend([p.predicted_class for p in preds])
            actuals.extend(y.tolist())

            code_trades, code_vals, code_bench_vals = self._simulate_trading(
                model=model,
                X=X,
                y=y,
                returns=returns,
                dates=idx,
                open_prices=aligned["open"].values,
                close_prices=aligned["close"].values,
                volumes=aligned["volume"].values,
                stock_code=code,
                capital=cap_slice,
                preds=preds,
            )

            trades.extend(code_trades)
            for dt, v in code_vals.items():
                portfolio_values_by_date[dt] += float(v)
            for dt, v in code_bench_vals.items():
                benchmark_values_by_date[dt] += float(v)

        accuracy = float(np.mean(np.array(predictions) == np.array(actuals))) if actuals else 0.0
        log.info(f"  Fold accuracy: {accuracy:.2%}")

        sorted_dates = sorted(portfolio_values_by_date.keys())
        returns_by_date: Dict[pd.Timestamp, float] = {}
        bench_by_date: Dict[pd.Timestamp, float] = {}

        for i, dt in enumerate(sorted_dates):
            if i == 0:
                returns_by_date[dt] = 0.0
                bench_by_date[dt] = 0.0
                continue

            prev_dt = sorted_dates[i - 1]
            pv0 = float(portfolio_values_by_date.get(prev_dt, 0.0))
            pv1 = float(portfolio_values_by_date.get(dt, 0.0))
            bv0 = float(benchmark_values_by_date.get(prev_dt, 0.0))
            bv1 = float(benchmark_values_by_date.get(dt, 0.0))

            returns_by_date[dt] = ((pv1 / pv0) - 1.0) * 100.0 if pv0 > 0 else 0.0
            bench_by_date[dt] = ((bv1 / bv0) - 1.0) * 100.0 if bv0 > 0 else 0.0

        return trades, returns_by_date, bench_by_date, accuracy

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
        capital: float,
        preds: Optional[List] = None,
    ) -> Tuple[List[BacktestTrade], Dict, Dict]:
        """Simulate trading with realistic costs"""
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

        horizon = 5
        min_confidence = 0.6
        commission_rate = 0.0003
        stamp_tax_rate = 0.001

        if hasattr(CONFIG, 'model'):
            if hasattr(CONFIG.model, 'prediction_horizon'):
                horizon = CONFIG.model.prediction_horizon
            if hasattr(CONFIG.model, 'min_confidence'):
                min_confidence = CONFIG.model.min_confidence

        if hasattr(CONFIG, 'trading'):
            if hasattr(CONFIG.trading, 'commission'):
                commission_rate = CONFIG.trading.commission
            if hasattr(CONFIG.trading, 'stamp_tax'):
                stamp_tax_rate = CONFIG.trading.stamp_tax

        limit_pct = float(get_price_limit(stock_code))
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

        first_open = float(open_prices[0]) if len(open_prices) > 0 else 0.0
        benchmark_shares = (capital / first_open) if first_open > 0 else 0.0

        if preds is None:
            preds = model.predict_batch(X)

        n = min(len(dates), len(open_prices), len(close_prices), len(volumes), len(preds))
        if n == 0:
            return [], {}, {}

        for t in range(n):
            dt = dates[t]
            open_t = float(open_prices[t])
            close_t = float(close_prices[t])
            prev_close = float(close_prices[t - 1]) if t > 0 else close_t

            if np.isnan(open_t) or np.isnan(close_t) or open_t <= 0 or close_t <= 0:
                continue

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
                    if entry_exec_i is not None and t == entry_exec_i:
                        pass
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

            # Mark-to-market
            daily_portfolio_values[dt] = float(cash + shares * close_t)
            daily_benchmark_values[dt] = float(benchmark_shares * close_t)

            if t < n - 1:
                pred = preds[t]
                if shares == 0 and pred.predicted_class == 2 and pred.confidence >= min_confidence:
                    pending_signal = ("BUY", float(pred.confidence), dt)
                elif shares > 0:
                    holding_bars = (t - entry_exec_i) if entry_exec_i is not None else 0
                    should_exit = (
                        holding_bars >= horizon or
                        (pred.predicted_class == 0 and pred.confidence >= min_confidence)
                    )
                    if should_exit:
                        pending_signal = ("SELL", float(pred.confidence), dt)

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

                daily_portfolio_values[dt] = float(cash)

        return trades, daily_portfolio_values, daily_benchmark_values

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

        equity = [capital]
        for ret in daily_returns:
            equity.append(equity[-1] * (1 + ret / 100))
        equity = np.array(equity[1:])

        total_return = (equity[-1] / capital - 1) * 100 if len(equity) > 0 else 0

        benchmark_equity = [capital]
        for ret in benchmark_daily:
            benchmark_equity.append(benchmark_equity[-1] * (1 + ret / 100))
        benchmark_return = (benchmark_equity[-1] / capital - 1) * 100 if len(benchmark_equity) > 1 else 0

        if len(daily_returns) > 1 and np.std(daily_returns) > 0:
            sharpe = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252)
        else:
            sharpe = 0

        if len(equity) > 0:
            running_max = np.maximum.accumulate(equity)
            drawdown = (equity - running_max) / running_max
            max_dd_pct = abs(np.min(drawdown)) * 100
            max_dd = abs(np.min(equity - running_max))
        else:
            max_dd = max_dd_pct = 0

        volatility = np.std(daily_returns) * np.sqrt(252) if len(daily_returns) > 0 else 0

        if max_dd_pct > 0.01:
            calmar = total_return / max_dd_pct
        else:
            calmar = 0

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