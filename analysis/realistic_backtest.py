"""
Enhanced Backtest with Realistic Market Impact Modeling

Addresses disadvantages:
- Backtest results don't guarantee live performance
- Simulation assumes perfect execution
- Real market impact not fully captured

Features:
- Realistic slippage modeling (volume-based, volatility-based)
- Market impact simulation
- Transaction cost analysis (commission, stamp duty, fees)
- Liquidity constraints
- Order book simulation
- Fill probability modeling
"""
from __future__ import annotations

import threading
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

import numpy as np

from utils.logger import get_logger

log = get_logger(__name__)


class OrderType(Enum):
    """Order type for backtest."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"


class FillType(Enum):
    """Fill type."""
    FULL = "full"
    PARTIAL = "partial"
    NONE = "none"


@dataclass
class MarketImpactModel:
    """
    Market impact model for realistic backtesting.

    Uses square-root law: impact = a * (qty / daily_volume)^b
    Where typically a ≈ 0.1-0.3, b ≈ 0.5-0.6
    """
    impact_coefficient: float = 0.15  # a parameter
    impact_exponent: float = 0.5  # b parameter (square-root law)
    base_spread_bps: float = 10.0  # Base bid-ask spread in bps

    def calculate_impact(
        self,
        order_qty: int,
        daily_volume: float,
        is_buy: bool,
    ) -> float:
        """
        Calculate market impact in price terms.

        Args:
            order_qty: Order quantity
            daily_volume: Average daily volume
            is_buy: True for buy, False for sell

        Returns:
            Impact as fraction of price (positive for buys, negative for sells)
        """
        if daily_volume <= 0:
            return 0.01  # Default 1% impact if no volume data

        # Participation rate
        participation = order_qty / daily_volume

        # Square-root law
        impact = self.impact_coefficient * (participation ** self.impact_exponent)

        # Cap impact at 10%
        return min(impact, 0.10)

    def calculate_effective_price(
        self,
        base_price: float,
        order_qty: int,
        daily_volume: float,
        is_buy: bool,
        bid_ask_spread: float = None,
    ) -> float:
        """
        Calculate effective fill price including impact and spread.

        Args:
            base_price: Mid price
            order_qty: Order quantity
            daily_volume: Average daily volume
            is_buy: True for buy, False for sell
            bid_ask_spread: Optional bid-ask spread (default: base_spread_bps)

        Returns:
            Effective fill price
        """
        if bid_ask_spread is None:
            bid_ask_spread = self.base_spread_bps / 10000

        # Market impact
        impact = self.calculate_impact(order_qty, daily_volume, is_buy)

        # Half spread (cost for market orders)
        half_spread = bid_ask_spread / 2

        if is_buy:
            # Buy at higher price
            effective_price = base_price * (1 + impact + half_spread)
        else:
            # Sell at lower price
            effective_price = base_price * (1 - impact - half_spread)

        return effective_price


@dataclass
class TransactionCostModel:
    """
    Transaction cost model for China A-shares.

    Costs include:
    - Commission (券商佣金): ~0.025% (min 5 CNY)
    - Stamp duty (印花税): 0.1% on sells only
    - Transfer fee (过户费): 0.002%
    - Exchange fee (经手费): 0.00487%
    """
    commission_rate: float = 0.00025  # 0.025%
    commission_minimum: float = 5.0  # 5 CNY minimum
    stamp_duty_rate: float = 0.001  # 0.1% on sells
    transfer_fee_rate: float = 0.00002  # 0.002%
    exchange_fee_rate: float = 0.0000487  # 0.00487%

    def calculate_total_cost(
        self,
        price: float,
        quantity: int,
        is_buy: bool,
    ) -> float:
        """
        Calculate total transaction cost.

        Args:
            price: Trade price
            quantity: Quantity
            is_buy: True for buy, False for sell

        Returns:
            Total cost in CNY
        """
        trade_value = price * quantity

        # Commission (both sides, with minimum)
        commission = max(trade_value * self.commission_rate, self.commission_minimum)

        # Stamp duty (sells only)
        stamp_duty = trade_value * self.stamp_duty_rate if not is_buy else 0.0

        # Transfer fee (both sides)
        transfer_fee = trade_value * self.transfer_fee_rate

        # Exchange fee (both sides)
        exchange_fee = trade_value * self.exchange_fee_rate

        total_cost = commission + stamp_duty + transfer_fee + exchange_fee

        return total_cost

    def calculate_cost_bps(self, is_buy: bool) -> float:
        """Calculate total cost in basis points."""
        # Approximate cost in bps
        buy_bps = (
            self.commission_rate * 10000 +
            self.transfer_fee_rate * 10000 +
            self.exchange_fee_rate * 10000
        )
        sell_bps = (
            self.commission_rate * 10000 +
            self.stamp_duty_rate * 10000 +
            self.transfer_fee_rate * 10000 +
            self.exchange_fee_rate * 10000
        )
        return buy_bps if is_buy else sell_bps


@dataclass
class FillResult:
    """Backtest fill result."""
    order_id: str
    symbol: str
    order_type: OrderType
    side: str  # buy/sell
    requested_qty: int
    filled_qty: int
    requested_price: float
    fill_price: float
    fill_type: FillType
    fill_ratio: float
    timestamp: datetime
    market_impact_bps: float
    slippage_bps: float
    transaction_cost: float
    notes: str = ""

    @property
    def total_cost_bps(self) -> float:
        """Total cost in basis points."""
        return self.market_impact_bps + self.slippage_bps + (
            self.transaction_cost / (self.fill_price * self.filled_qty) * 10000
            if self.fill_price > 0 and self.filled_qty > 0 else 0
        )


@dataclass
class LiquidityConstraint:
    """Liquidity constraints for realistic backtesting."""
    max_participation_rate: float = 0.05  # Max 5% of daily volume
    max_order_value: float = 1e7  # 10M CNY max order
    min_order_value: float = 100  # 100 CNY min order
    fill_probability_base: float = 0.95  # Base fill probability


class RealisticBacktestEngine:
    """
    Realistic backtest engine with market impact and transaction costs.

    Features:
    - Volume-based market impact
    - Realistic fill probabilities
    - Transaction cost modeling
    - Liquidity constraints
    - Partial fill simulation
    """

    def __init__(
        self,
        impact_model: MarketImpactModel = None,
        cost_model: TransactionCostModel = None,
        liquidity: LiquidityConstraint = None,
    ) -> None:
        self.impact_model = impact_model or MarketImpactModel()
        self.cost_model = cost_model or TransactionCostModel()
        self.liquidity = liquidity or LiquidityConstraint()

        self._lock = threading.RLock()
        self._fill_results: list[FillResult] = []
        self._daily_volumes: dict[str, float] = {}
        self._daily_prices: dict[str, list[float]] = {}

    def set_daily_volume(self, symbol: str, volume: float) -> None:
        """Set daily volume for symbol."""
        with self._lock:
            self._daily_volumes[symbol] = volume

    def get_daily_volume(self, symbol: str) -> float:
        """Get daily volume for symbol."""
        with self._lock:
            return self._daily_volumes.get(symbol, 1e6)  # Default 1M

    def simulate_fill(
        self,
        order_id: str,
        symbol: str,
        order_type: OrderType,
        side: str,
        quantity: int,
        price: float,
        daily_volume: float = None,
        bid_ask_spread: float = None,
        timestamp: datetime = None,
    ) -> FillResult:
        """
        Simulate realistic order fill.

        Args:
            order_id: Order ID
            symbol: Stock code
            order_type: Market, limit, or stop
            side: Buy or sell
            quantity: Order quantity
            price: Order price (limit price or expected market price)
            daily_volume: Daily volume (default: from set_daily_volume)
            bid_ask_spread: Bid-ask spread in decimal (default: model base)
            timestamp: Fill timestamp

        Returns:
            FillResult with realistic fill details
        """
        if timestamp is None:
            timestamp = datetime.now()

        if daily_volume is None:
            daily_volume = self.get_daily_volume(symbol)

        is_buy = side.lower() == "buy"

        # Check liquidity constraints
        order_value = quantity * price

        if order_value > self.liquidity.max_order_value:
            # Scale down order
            max_qty = int(self.liquidity.max_order_value / price)
            fill_ratio = max_qty / quantity
            filled_qty = max_qty
            fill_type = FillType.PARTIAL
            notes = f"Scaled to max order value {self.liquidity.max_order_value:.0f}"
        elif order_value < self.liquidity.min_order_value:
            # Reject order
            return FillResult(
                order_id=order_id,
                symbol=symbol,
                order_type=order_type,
                side=side,
                requested_qty=quantity,
                filled_qty=0,
                requested_price=price,
                fill_price=0.0,
                fill_type=FillType.NONE,
                fill_ratio=0.0,
                timestamp=timestamp,
                market_impact_bps=0.0,
                slippage_bps=0.0,
                transaction_cost=0.0,
                notes="Below minimum order value",
            )
        else:
            fill_ratio = 1.0
            filled_qty = quantity
            fill_type = FillType.FULL
            notes = ""

        # Check participation rate
        max_allowed_qty = int(daily_volume * self.liquidity.max_participation_rate)
        if filled_qty > max_allowed_qty:
            filled_qty = max_allowed_qty
            fill_ratio = filled_qty / quantity
            if fill_ratio < 1.0:
                fill_type = FillType.PARTIAL
                notes = f"Limited to {self.liquidity.max_participation_rate:.0%} of daily volume"

        # Calculate effective price with market impact
        effective_price = self.impact_model.calculate_effective_price(
            base_price=price,
            order_qty=filled_qty,
            daily_volume=daily_volume,
            is_buy=is_buy,
            bid_ask_spread=bid_ask_spread,
        )

        # Calculate market impact in bps
        market_impact_bps = abs(effective_price - price) / price * 10000

        # Calculate slippage (additional random component)
        slippage_bps = np.random.normal(0, 2.0)  # Random slippage ~2 bps std
        slippage_bps = max(0, slippage_bps)  # No negative slippage benefit
        effective_price *= (1 + slippage_bps / 10000 if is_buy else 1 - slippage_bps / 10000)

        # Fill probability for limit orders
        if order_type == OrderType.LIMIT:
            fill_prob = self._calculate_limit_fill_probability(
                price, effective_price, is_buy
            )
            if np.random.random() > fill_prob:
                fill_type = FillType.NONE
                filled_qty = 0
                fill_ratio = 0.0
                notes = "Limit order not filled"

        # Calculate transaction cost
        transaction_cost = self.cost_model.calculate_total_cost(
            price=effective_price,
            quantity=filled_qty,
            is_buy=is_buy,
        )

        result = FillResult(
            order_id=order_id,
            symbol=symbol,
            order_type=order_type,
            side=side,
            requested_qty=quantity,
            filled_qty=filled_qty,
            requested_price=price,
            fill_price=effective_price,
            fill_type=fill_type,
            fill_ratio=fill_ratio,
            timestamp=timestamp,
            market_impact_bps=round(market_impact_bps, 2),
            slippage_bps=round(slippage_bps, 2),
            transaction_cost=round(transaction_cost, 2),
            notes=notes,
        )

        with self._lock:
            self._fill_results.append(result)

        return result

    def _calculate_limit_fill_probability(
        self,
        limit_price: float,
        market_price: float,
        is_buy: bool,
    ) -> float:
        """Calculate probability of limit order fill."""
        if is_buy:
            # Buy limit: filled if limit >= market
            if limit_price >= market_price:
                return self.liquidity.fill_probability_base
            else:
                # Probability decreases as limit goes below market
                gap = (market_price - limit_price) / market_price
                return max(0, self.liquidity.fill_probability_base - gap * 10)
        else:
            # Sell limit: filled if limit <= market
            if limit_price <= market_price:
                return self.liquidity.fill_probability_base
            else:
                gap = (limit_price - market_price) / market_price
                return max(0, self.liquidity.fill_probability_base - gap * 10)

    def get_execution_statistics(self) -> dict:
        """Get execution quality statistics."""
        with self._lock:
            if not self._fill_results:
                return {"count": 0}

            fills = self._fill_results

        total_cost_bps = [f.total_cost_bps for f in fills if f.filled_qty > 0]
        market_impact_bps = [f.market_impact_bps for f in fills if f.filled_qty > 0]
        slippage_bps = [f.slippage_bps for f in fills if f.filled_qty > 0]

        full_fills = sum(1 for f in fills if f.fill_type == FillType.FULL)
        partial_fills = sum(1 for f in fills if f.fill_type == FillType.PARTIAL)
        no_fills = sum(1 for f in fills if f.fill_type == FillType.NONE)

        return {
            "total_orders": len(fills),
            "filled_orders": len([f for f in fills if f.filled_qty > 0]),
            "full_fills": full_fills,
            "partial_fills": partial_fills,
            "no_fills": no_fills,
            "fill_rate": len([f for f in fills if f.filled_qty > 0]) / len(fills),
            "avg_total_cost_bps": round(np.mean(total_cost_bps), 2) if total_cost_bps else 0,
            "avg_market_impact_bps": round(np.mean(market_impact_bps), 2) if market_impact_bps else 0,
            "avg_slippage_bps": round(np.mean(slippage_bps), 2) if slippage_bps else 0,
            "total_transaction_cost": round(sum(f.transaction_cost for f in fills), 2),
        }

    def reset(self) -> None:
        """Reset backtest engine."""
        with self._lock:
            self._fill_results.clear()
            self._daily_volumes.clear()
            self._daily_prices.clear()


@dataclass
class BacktestConfig:
    """Backtest configuration for realistic simulation."""
    initial_capital: float = 1e6
    impact_coefficient: float = 0.15
    base_spread_bps: float = 10.0
    commission_rate: float = 0.00025
    stamp_duty_rate: float = 0.001
    max_participation_rate: float = 0.05
    max_order_value: float = 1e7
    enable_partial_fills: bool = True
    enable_limit_fill_probability: bool = True


def create_realistic_backtest_engine(
    config: BacktestConfig = None,
) -> RealisticBacktestEngine:
    """Create realistic backtest engine with given configuration."""
    if config is None:
        config = BacktestConfig()

    impact_model = MarketImpactModel(
        impact_coefficient=config.impact_coefficient,
        base_spread_bps=config.base_spread_bps,
    )

    cost_model = TransactionCostModel(
        commission_rate=config.commission_rate,
        stamp_duty_rate=config.stamp_duty_rate,
    )

    liquidity = LiquidityConstraint(
        max_participation_rate=config.max_participation_rate,
        max_order_value=config.max_order_value,
    )

    return RealisticBacktestEngine(
        impact_model=impact_model,
        cost_model=cost_model,
        liquidity=liquidity,
    )
