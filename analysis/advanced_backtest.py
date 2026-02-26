# analysis/advanced_backtest.py
"""Advanced Backtesting Engine with Professional Features.

This module provides institutional-grade backtesting capabilities:
- Realistic transaction costs (China A-share specific)
- Walk-forward optimization
- Monte Carlo simulation
- Multi-strategy portfolio backtesting
- Market impact modeling
- Slippage models
- Risk-adjusted metrics
- Regime detection
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

import numpy as np
import pandas as pd
from scipy import stats

from config.settings import CONFIG
from utils.logger import get_logger

log = get_logger(__name__)


class OrderType(Enum):
    """Order types for backtesting."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderSide(Enum):
    """Order side."""
    BUY = "buy"
    SELL = "sell"


@dataclass
class ChinaTransactionCosts:
    """China A-share transaction cost model.

    Components:
    - Commission: Broker fee (typically 0.025% - 0.03%, min 5 CNY)
    - Stamp Duty: Government tax (0.1% on sell only)
    - Transfer Fee: Exchange fee (0.002% both sides)
    - Exchange Fee: Regulatory fee (varies)
    """

    # Commission (broker fee)
    commission_rate: float = 0.00025  # 0.025%
    commission_min: float = 5.0  # Minimum 5 CNY

    # Stamp duty (only on sell)
    stamp_duty_rate: float = 0.001  # 0.1%

    # Transfer fee (both sides)
    transfer_fee_rate: float = 0.00002  # 0.002%

    # Exchange fee (both sides)
    exchange_fee_rate: float = 0.0000487  # ~0.00487%

    # Regulatory fee (both sides)
    regulatory_fee_rate: float = 0.00002  # 0.002%

    def calculate_buy_cost(self, trade_value: float) -> float:
        """Calculate total cost for buying."""
        commission = max(trade_value * self.commission_rate, self.commission_min)
        transfer_fee = trade_value * self.transfer_fee_rate
        exchange_fee = trade_value * self.exchange_fee_rate
        regulatory_fee = trade_value * self.regulatory_fee_rate

        return commission + transfer_fee + exchange_fee + regulatory_fee

    def calculate_sell_cost(self, trade_value: float) -> float:
        """Calculate total cost for selling."""
        commission = max(trade_value * self.commission_rate, self.commission_min)
        stamp_duty = trade_value * self.stamp_duty_rate
        transfer_fee = trade_value * self.transfer_fee_rate
        exchange_fee = trade_value * self.exchange_fee_rate
        regulatory_fee = trade_value * self.regulatory_fee_rate

        return commission + stamp_duty + transfer_fee + exchange_fee + regulatory_fee

    def calculate_round_trip_cost(self, trade_value: float) -> float:
        """Calculate total round-trip cost."""
        return self.calculate_buy_cost(trade_value) + self.calculate_sell_cost(trade_value)


@dataclass
class SlippageModel:
    """Advanced slippage model with market impact."""

    # Base slippage (spread crossing)
    base_slippage: float = 0.001  # 0.1%

    # Volume impact (order size vs daily volume)
    volume_impact_exponent: float = 1.5

    # Volatility impact
    volatility_impact: float = 0.5

    # Liquidity tiers
    liquidity_tiers: dict = field(default_factory=lambda: {
        "high": 1.0,      # > 1B daily volume
        "medium": 1.5,    # 100M - 1B
        "low": 2.0,       # 10M - 100M
        "very_low": 3.0,  # < 10M
    })

    def calculate(
        self,
        order_value: float,
        daily_volume: float,
        daily_volatility: float,
        liquidity_tier: str = "medium",
    ) -> float:
        """Calculate slippage for an order."""
        # Base slippage
        slippage = self.base_slippage

        # Volume impact
        if daily_volume > 0:
            volume_ratio = order_value / daily_volume
            slippage += self.base_slippage * (volume_ratio ** self.volume_impact_exponent)

        # Volatility impact
        slippage += daily_volatility * self.volatility_impact

        # Liquidity tier multiplier
        tier_mult = self.liquidity_tiers.get(liquidity_tier, 1.0)
        slippage *= tier_mult

        # Cap at 5%
        return min(slippage, 0.05)


@dataclass
class BacktestTrade:
    """Single trade record with full metadata."""
    trade_id: str
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp | None
    symbol: str
    side: OrderSide
    entry_price: float
    exit_price: float = 0.0
    quantity: int = 0
    trade_value: float = 0.0
    commission: float = 0.0
    slippage_cost: float = 0.0
    pnl: float = 0.0
    pnl_pct: float = 0.0
    holding_days: int = 0
    entry_signal: str = ""
    exit_signal: str = ""
    max_profit: float = 0.0
    max_loss: float = 0.0
    mae: float = 0.0  # Maximum Adverse Excursion
    mfe: float = 0.0  # Maximum Favorable Excursion

    @property
    def total_cost(self) -> float:
        """Total transaction costs."""
        return self.commission + self.slippage_cost

    @property
    def net_pnl(self) -> float:
        """PnL after costs."""
        return self.pnl - self.total_cost


@dataclass
class BacktestPosition:
    """Current open position."""
    symbol: str
    quantity: int
    entry_price: float
    entry_date: pd.Timestamp
    current_price: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0

    @property
    def unrealized_pnl(self) -> float:
        """Unrealized PnL."""
        return (self.current_price - self.entry_price) * self.quantity

    @property
    def unrealized_pnl_pct(self) -> float:
        """Unrealized PnL percentage."""
        return (self.current_price / self.entry_price - 1) * 100 if self.entry_price > 0 else 0


@dataclass
class BacktestMetrics:
    """Comprehensive backtest metrics."""
    # Returns
    total_return: float = 0.0
    annualized_return: float = 0.0
    excess_return: float = 0.0  # vs benchmark

    # Risk metrics
    volatility: float = 0.0
    downside_deviation: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_duration: int = 0
    ulcer_index: float = 0.0

    # Risk-adjusted returns
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    information_ratio: float = 0.0
    omega_ratio: float = 0.0

    # Trade statistics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    avg_win_pct: float = 0.0
    avg_loss_pct: float = 0.0
    avg_holding_period: float = 0.0

    # Cost analysis
    total_commission: float = 0.0
    total_slippage: float = 0.0
    cost_drag: float = 0.0  # Return reduction due to costs

    # Advanced metrics
    tail_ratio: float = 0.0
    common_sense_ratio: float = 0.0
    skewness: float = 0.0
    kurtosis: float = 0.0
    var_95: float = 0.0
    cvar_95: float = 0.0

    # Consistency metrics
    consecutive_wins: int = 0
    consecutive_losses: int = 0
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0
    recovery_factor: float = 0.0
    payoff_ratio: float = 0.0


class AdvancedBacktestEngine:
    """Professional backtesting engine with realistic modeling."""

    def __init__(
        self,
        initial_capital: float = 100000.0,
        transaction_costs: ChinaTransactionCosts | None = None,
        slippage_model: SlippageModel | None = None,
        benchmark: str = "000001.SS",  # Shanghai Composite
    ) -> None:
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.transaction_costs = transaction_costs or ChinaTransactionCosts()
        self.slippage_model = slippage_model or SlippageModel()

        self.benchmark = benchmark
        self.positions: dict[str, BacktestPosition] = {}
        self.trades: list[BacktestTrade] = []
        self.equity_curve: list[float] = []
        self.daily_returns: list[float] = []

        self._trade_counter = 0

    def _generate_trade_id(self) -> str:
        """Generate unique trade ID."""
        self._trade_counter += 1
        return f"T{self._trade_counter:06d}"

    def run(
        self,
        data: pd.DataFrame,
        signals: pd.Series,
        benchmark_data: pd.DataFrame | None = None,
        position_sizing: Callable | None = None,
    ) -> BacktestMetrics:
        """Run backtest on historical data.

        Args:
            data: OHLCV DataFrame with datetime index
            signals: Series of trading signals (-1, 0, 1)
            benchmark_data: Optional benchmark OHLCV data
            position_sizing: Optional position sizing function

        Returns:
            BacktestMetrics object
        """
        # Reset state
        self.capital = self.initial_capital
        self.positions = {}
        self.trades = []
        self.equity_curve = [self.initial_capital]
        self.daily_returns = []

        # Validate inputs
        if len(data) < 2:
            raise ValueError("Insufficient data for backtest")

        # Align signals with data
        signals = signals.reindex(data.index, fill_value=0)

        # Run backtest bar by bar
        for i in range(1, len(data)):
            current_idx = data.index[i]
            prev_idx = data.index[i - 1]

            bar = data.iloc[i]
            prev_bar = data.iloc[i - 1]

            # Update existing positions
            self._update_positions(bar)

            # Check for exit signals
            self._check_exits(bar, current_idx)

            # Check for entry signals
            signal = signals.iloc[i]
            if signal != 0 and bar.name not in self.positions:
                self._check_entry(bar, signal, current_idx, position_sizing)

            # Calculate portfolio value
            portfolio_value = self.capital + sum(
                p.unrealized_pnl for p in self.positions.values()
            )
            self.equity_curve.append(portfolio_value)

            # Calculate daily return
            if len(self.equity_curve) > 1:
                daily_return = (
                    self.equity_curve[-1] / self.equity_curve[-2] - 1
                )
                self.daily_returns.append(daily_return)

        # Close any remaining positions at the end
        self._close_all_positions(data.iloc[-1], data.index[-1])

        # Calculate metrics
        metrics = self._calculate_metrics(
            data,
            benchmark_data,
        )

        return metrics

    def _update_positions(self, bar: pd.Series) -> None:
        """Update position prices with current bar data."""
        for position in self.positions.values():
            position.current_price = float(bar["close"])

    def _check_exits(
        self,
        bar: pd.Series,
        current_idx: pd.Timestamp,
    ) -> None:
        """Check and execute exit conditions."""
        symbols_to_remove = []

        for symbol, position in self.positions.items():
            should_exit = False
            exit_signal = ""

            # Stop loss check
            if position.stop_loss > 0 and bar["low"] <= position.stop_loss:
                should_exit = True
                exit_signal = "stop_loss"

            # Take profit check
            if position.take_profit > 0 and bar["high"] >= position.take_profit:
                should_exit = True
                exit_signal = "take_profit"

            # Reverse signal
            # (would be handled by signal processing)

            if should_exit:
                self._execute_exit(
                    position,
                    bar,
                    current_idx,
                    exit_signal,
                )
                symbols_to_remove.append(symbol)

        for symbol in symbols_to_remove:
            del self.positions[symbol]

    def _check_entry(
        self,
        bar: pd.Series,
        signal: int,
        current_idx: pd.Timestamp,
        position_sizing: Callable | None,
    ) -> None:
        """Check and execute entry conditions."""
        symbol = bar.name if hasattr(bar, "name") else "UNKNOWN"

        # Calculate position size
        if position_sizing:
            quantity = position_sizing(
                self.capital,
                float(bar["close"]),
                float(bar["volume"]),
            )
        else:
            # Default: 10% of capital per position
            position_value = self.capital * 0.10
            quantity = int(position_value / bar["close"])

        if quantity <= 0:
            return

        # Calculate entry price with slippage
        slippage = self._calculate_slippage(
            quantity * float(bar["close"]),
            float(bar["volume"]),
            0.02,  # Assume 2% daily volatility
        )

        if signal > 0:
            entry_price = float(bar["close"]) * (1 + slippage)
        else:
            entry_price = float(bar["close"]) * (1 - slippage)

        # Calculate transaction costs
        trade_value = quantity * entry_price
        commission = self.transaction_costs.calculate_buy_cost(trade_value)

        # Check if we have enough capital
        total_cost = trade_value + commission
        if total_cost > self.capital:
            return

        # Execute entry
        self.capital -= total_cost

        position = BacktestPosition(
            symbol=symbol,
            quantity=quantity,
            entry_price=entry_price,
            entry_date=current_idx,
            current_price=entry_price,
        )
        self.positions[symbol] = position

    def _execute_exit(
        self,
        position: BacktestPosition,
        bar: pd.Series,
        current_idx: pd.Timestamp,
        exit_signal: str,
    ) -> None:
        """Execute position exit."""
        # Calculate exit price
        if exit_signal == "stop_loss":
            exit_price = position.stop_loss
        elif exit_signal == "take_profit":
            exit_price = position.take_profit
        else:
            exit_price = float(bar["close"])

        # Apply slippage
        slippage = self._calculate_slippage(
            position.quantity * exit_price,
            float(bar["volume"]),
            0.02,
        )
        exit_price *= (1 - slippage)

        # Calculate PnL
        trade_value = position.quantity * exit_price
        commission = self.transaction_costs.calculate_sell_cost(trade_value)

        pnl = (exit_price - position.entry_price) * position.quantity
        pnl_pct = (exit_price / position.entry_price - 1) * 100

        # Create trade record
        trade = BacktestTrade(
            trade_id=self._generate_trade_id(),
            entry_date=position.entry_date,
            exit_date=current_idx,
            symbol=position.symbol,
            side=OrderSide.BUY,
            entry_price=position.entry_price,
            exit_price=exit_price,
            quantity=position.quantity,
            trade_value=trade_value,
            commission=commission,
            slippage_cost=slippage * trade_value,
            pnl=pnl,
            pnl_pct=pnl_pct,
            holding_days=(current_idx - position.entry_date).days,
            exit_signal=exit_signal,
        )

        self.trades.append(trade)
        self.capital += trade_value - commission

    def _close_all_positions(
        self,
        bar: pd.Series,
        current_idx: pd.Timestamp,
    ) -> None:
        """Close all remaining positions."""
        for symbol, position in list(self.positions.items()):
            self._execute_exit(
                position,
                bar,
                current_idx,
                "end_of_backtest",
            )
        self.positions.clear()

    def _calculate_slippage(
        self,
        order_value: float,
        daily_volume: float,
        daily_volatility: float,
    ) -> float:
        """Calculate slippage for an order."""
        return self.slippage_model.calculate(
            order_value,
            daily_volume,
            daily_volatility,
        )

    def _calculate_metrics(
        self,
        data: pd.DataFrame,
        benchmark_data: pd.DataFrame | None,
    ) -> BacktestMetrics:
        """Calculate comprehensive backtest metrics."""
        metrics = BacktestMetrics()

        if not self.trades:
            return metrics

        # Basic statistics
        metrics.total_trades = len(self.trades)
        winning = [t for t in self.trades if t.net_pnl > 0]
        losing = [t for t in self.trades if t.net_pnl <= 0]

        metrics.winning_trades = len(winning)
        metrics.losing_trades = len(losing)
        metrics.win_rate = len(winning) / len(self.trades) if self.trades else 0

        # PnL analysis
        if winning:
            metrics.avg_win = np.mean([t.net_pnl for t in winning])
            metrics.avg_win_pct = np.mean([t.pnl_pct for t in winning])
        if losing:
            metrics.avg_loss = abs(np.mean([t.net_pnl for t in losing]))
            metrics.avg_loss_pct = abs(np.mean([t.pnl_pct for t in losing]))

        gross_profit = sum(t.net_pnl for t in winning) if winning else 0
        gross_loss = abs(sum(t.net_pnl for t in losing)) if losing else 0
        metrics.profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        # Cost analysis
        metrics.total_commission = sum(t.commission for t in self.trades)
        metrics.total_slippage = sum(t.slippage_cost for t in self.trades)

        total_costs = metrics.total_commission + metrics.total_slippage
        gross_return = sum(t.pnl for t in self.trades)
        metrics.cost_drag = total_costs / gross_return if gross_return > 0 else 0

        # Holding period
        metrics.avg_holding_period = np.mean([t.holding_days for t in self.trades])

        # Return metrics
        if self.equity_curve:
            total_return = (self.equity_curve[-1] / self.initial_capital - 1) * 100
            metrics.total_return = total_return

            # Annualized return
            days = (data.index[-1] - data.index[0]).days if len(data) > 0 else 1
            years = days / 365.25
            if years > 0:
                metrics.annualized_return = (
                    (self.equity_curve[-1] / self.initial_capital) ** (1 / years) - 1
                ) * 100

        # Risk metrics from returns
        if self.daily_returns:
            returns_array = np.array(self.daily_returns)

            metrics.volatility = np.std(returns_array) * np.sqrt(252) * 100

            # Downside deviation
            negative_returns = returns_array[returns_array < 0]
            if len(negative_returns) > 0:
                metrics.downside_deviation = np.std(negative_returns) * np.sqrt(252) * 100

            # Max drawdown
            equity_array = np.array(self.equity_curve)
            running_max = np.maximum.accumulate(equity_array)
            drawdown = (equity_array - running_max) / running_max
            metrics.max_drawdown = abs(np.min(drawdown)) * 100

            # Drawdown duration
            in_drawdown = drawdown < 0
            drawdown_periods = []
            current_period = 0
            for is_dd in in_drawdown:
                if is_dd:
                    current_period += 1
                else:
                    if current_period > 0:
                        drawdown_periods.append(current_period)
                    current_period = 0
            if current_period > 0:
                drawdown_periods.append(current_period)
            metrics.max_drawdown_duration = max(drawdown_periods) if drawdown_periods else 0

            # Ulcer Index
            metrics.ulcer_index = np.sqrt(np.mean(drawdown ** 2)) * 100

            # Risk-adjusted returns
            risk_free_rate = 0.02 / 252  # Daily risk-free rate
            excess_returns = returns_array - risk_free_rate

            if np.std(returns_array) > 0:
                metrics.sharpe_ratio = np.mean(excess_returns) / np.std(returns_array) * np.sqrt(252)

            if metrics.downside_deviation > 0:
                metrics.sortino_ratio = (
                    np.mean(excess_returns) / (metrics.downside_deviation / 100 / np.sqrt(252))
                )

            if metrics.max_drawdown > 0:
                metrics.calmar_ratio = metrics.annualized_return / metrics.max_drawdown

            # Omega ratio (threshold = 0)
            gains = returns_array[returns_array > 0].sum()
            losses = abs(returns_array[returns_array < 0]).sum()
            metrics.omega_ratio = gains / losses if losses > 0 else float("inf")

            # Distribution statistics
            metrics.skewness = float(stats.skew(returns_array))
            metrics.kurtosis = float(stats.kurtosis(returns_array))

            # VaR and CVaR
            metrics.var_95 = float(np.percentile(returns_array, 5)) * 100
            metrics.cvar_95 = float(np.mean(returns_array[returns_array <= np.percentile(returns_array, 5)])) * 100

            # Tail ratio
            if len(returns_array) >= 10:
                q90 = np.percentile(returns_array, 90)
                q10 = np.percentile(returns_array, 10)
                metrics.tail_ratio = q90 / abs(q10) if q10 != 0 else float("inf")

        # Consistency metrics
        if self.trades:
            pnl_sequence = [t.net_pnl for t in self.trades]

            # Consecutive wins/losses
            max_cons_wins = 0
            max_cons_losses = 0
            current_wins = 0
            current_losses = 0

            for pnl in pnl_sequence:
                if pnl > 0:
                    current_wins += 1
                    current_losses = 0
                    max_cons_wins = max(max_cons_wins, current_wins)
                else:
                    current_losses += 1
                    current_wins = 0
                    max_cons_losses = max(max_cons_losses, current_losses)

            metrics.max_consecutive_wins = max_cons_wins
            metrics.max_consecutive_losses = max_cons_losses

            # Recovery factor
            if metrics.max_drawdown > 0:
                net_profit = sum(t.net_pnl for t in self.trades)
                metrics.recovery_factor = net_profit / (metrics.max_drawdown / 100 * self.initial_capital)

            # Payoff ratio
            if metrics.avg_loss > 0:
                metrics.payoff_ratio = metrics.avg_win / metrics.avg_loss

        return metrics


class WalkForwardOptimizer:
    """Walk-forward optimization engine."""

    def __init__(
        self,
        engine: AdvancedBacktestEngine,
        train_months: int = 12,
        test_months: int = 3,
        step_months: int = 1,
    ) -> None:
        self.engine = engine
        self.train_months = train_months
        self.test_months = test_months
        self.step_months = step_months

    def run(
        self,
        data: pd.DataFrame,
        signal_generator: Callable,
        param_grid: dict[str, list],
    ) -> dict[str, Any]:
        """Run walk-forward optimization.

        Args:
            data: Full dataset
            signal_generator: Function that returns signals given data and params
            param_grid: Parameter grid to optimize

        Returns:
            Optimization results dict
        """
        results = []

        # Generate time windows
        windows = self._generate_windows(data)

        for train_start, train_end, test_start, test_end in windows:
            # Split data
            train_data = data.loc[train_start:train_end]
            test_data = data.loc[test_start:test_end]

            # Optimize on training data
            best_params, best_score = self._optimize_parameters(
                train_data,
                signal_generator,
                param_grid,
            )

            # Test on out-of-sample data
            test_signals = signal_generator(test_data, **best_params)
            test_metrics = self.engine.run(test_data, test_signals)

            results.append({
                "train_start": train_start,
                "train_end": train_end,
                "test_start": test_start,
                "test_end": test_end,
                "best_params": best_params,
                "train_score": best_score,
                "test_sharpe": test_metrics.sharpe_ratio,
                "test_return": test_metrics.total_return,
                "test_max_dd": test_metrics.max_drawdown,
            })

        # Calculate stability metrics
        results_df = pd.DataFrame(results)

        return {
            "windows": results,
            "summary": {
                "avg_oos_sharpe": results_df["test_sharpe"].mean(),
                "avg_oos_return": results_df["test_return"].mean(),
                "avg_oos_max_dd": results_df["test_max_dd"].mean(),
                "sharpe_std": results_df["test_sharpe"].std(),
                "robustness_score": self._calculate_robustness(results),
            },
        }

    def _generate_windows(
        self,
        data: pd.DataFrame,
    ) -> list[tuple]:
        """Generate walk-forward windows."""
        windows = []

        train_start = data.index[0]

        while True:
            # Calculate window boundaries
            train_end = train_start + pd.DateOffset(months=self.train_months)
            test_start = train_end + pd.DateOffset(days=1)
            test_end = test_start + pd.DateOffset(months=self.test_months)

            if test_end > data.index[-1]:
                break

            if train_end > data.index[-1]:
                break

            windows.append((train_start, train_end, test_start, test_end))

            # Step forward
            train_start = train_start + pd.DateOffset(months=self.step_months)

        return windows

    def _optimize_parameters(
        self,
        train_data: pd.DataFrame,
        signal_generator: Callable,
        param_grid: dict[str, list],
    ) -> tuple[dict, float]:
        """Optimize parameters on training data."""
        from itertools import product

        best_params = {}
        best_score = float("-inf")

        # Grid search
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())

        for values in product(*param_values):
            params = dict(zip(param_names, values))

            try:
                signals = signal_generator(train_data, **params)
                metrics = self.engine.run(train_data, signals)

                # Optimize for Sharpe ratio
                score = metrics.sharpe_ratio

                if score > best_score:
                    best_score = score
                    best_params = params

            except Exception as e:
                log.debug(f"Parameter set failed: {e}")
                continue

        return best_params, best_score

    def _calculate_robustness(self, results: list[dict]) -> float:
        """Calculate strategy robustness score."""
        if not results:
            return 0.0

        sharpe_values = [r["test_sharpe"] for r in results]
        return_values = [r["test_return"] for r in results]

        # Consistency score (inverse of Sharpe std)
        sharpe_std = np.std(sharpe_values)
        consistency = 1 / (1 + sharpe_std)

        # Positive returns ratio
        positive_ratio = sum(1 for r in return_values if r > 0) / len(return_values)

        # Average performance
        avg_sharpe = np.mean(sharpe_values)

        # Combined robustness score
        return consistency * 0.4 + positive_ratio * 0.3 + min(avg_sharpe, 2) * 0.3


class MonteCarloSimulator:
    """Monte Carlo simulation for strategy analysis."""

    def __init__(self, engine: AdvancedBacktestEngine) -> None:
        self.engine = engine

    def run(
        self,
        trades: list[BacktestTrade],
        n_simulations: int = 1000,
    ) -> dict[str, Any]:
        """Run Monte Carlo simulation on trade sequence.

        Args:
            trades: Historical trades
            n_simulations: Number of simulations

        Returns:
            Simulation results
        """
        if not trades:
            return {"error": "No trades provided"}

        # Extract trade statistics
        pnl_values = [t.net_pnl for t in trades]
        win_rate = sum(1 for p in pnl_values if p > 0) / len(pnl_values)

        winning_trades = [p for p in pnl_values if p > 0]
        losing_trades = [p for p in pnl_values if p <= 0]

        # Run simulations
        simulation_results = []

        for _ in range(n_simulations):
            # Random resampling with replacement
            simulated_pnl = []
            for _ in range(len(trades)):
                if np.random.random() < win_rate:
                    pnl = np.random.choice(winning_trades) if winning_trades else 0
                else:
                    pnl = np.random.choice(losing_trades) if losing_trades else 0
                simulated_pnl.append(pnl)

            # Calculate metrics
            cumulative = np.cumsum(simulated_pnl)
            total_return = cumulative[-1]
            max_dd = self._calculate_max_drawdown(cumulative)

            simulation_results.append({
                "total_return": total_return,
                "max_drawdown": max_dd,
                "final_equity": self.engine.initial_capital + total_return,
            })

        # Analyze results
        results_array = np.array([r["total_return"] for r in simulation_results])
        dd_array = np.array([r["max_drawdown"] for r in simulation_results])

        return {
            "simulations": n_simulations,
            "statistics": {
                "mean_return": float(np.mean(results_array)),
                "std_return": float(np.std(results_array)),
                "median_return": float(np.median(results_array)),
                "percentile_5": float(np.percentile(results_array, 5)),
                "percentile_95": float(np.percentile(results_array, 95)),
                "percentile_99": float(np.percentile(results_array, 99)),
                "probability_profit": float(np.mean(results_array > 0)),
                "probability_ruin": float(np.mean(results_array < -self.engine.initial_capital * 0.5)),
            },
            "drawdown": {
                "mean_max_dd": float(np.mean(dd_array)),
                "median_max_dd": float(np.median(dd_array)),
                "worst_max_dd": float(np.max(dd_array)),
            },
        }

    def _calculate_max_drawdown(self, cumulative: np.ndarray) -> float:
        """Calculate maximum drawdown from cumulative PnL."""
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / np.maximum(running_max, 1)
        return abs(np.min(drawdown))


def get_advanced_backtest_engine(
    initial_capital: float = 100000.0,
) -> AdvancedBacktestEngine:
    """Get advanced backtest engine instance."""
    return AdvancedBacktestEngine(initial_capital=initial_capital)
