"""Tests for trading strategies."""
import pytest

from core.types import OrderSide
from strategies import (
    BaseStrategy,
    BollingerBreakoutStrategy,
    EarningsMomentumStrategy,
    GapAndGoStrategy,
    GoldenCrossStrategy,
    MACDDivergenceStrategy,
    MeanReversionStrategy,
    MomentumBreakoutStrategy,
    RSIOversoldStrategy,
    Signal,
    SignalStrength,
    SupportResistanceStrategy,
    TrendFollowingStrategy,
    VolumeProfileStrategy,
    VWAPReversionStrategy,
    get_strategy,
    list_strategies,
    register_strategy,
)


def test_list_strategies() -> None:
    """Test that all strategies are registered."""
    strategies = list_strategies()
    assert len(strategies) >= 12
    assert "momentum_breakout" in strategies
    assert "mean_reversion" in strategies
    assert "trend_following" in strategies


def test_get_strategy() -> None:
    """Test getting strategy by name."""
    strategy = get_strategy("momentum_breakout")
    assert strategy is not None
    assert isinstance(strategy, MomentumBreakoutStrategy)


def test_get_strategy_not_found() -> None:
    """Test getting non-existent strategy."""
    strategy = get_strategy("nonexistent")
    assert strategy is None


def test_base_strategy_abstract() -> None:
    """Test that BaseStrategy requires generate_signal."""
    with pytest.raises(TypeError):
        BaseStrategy()


def test_signal_creation() -> None:
    """Test Signal creation."""
    signal = Signal(
        strategy_name="test",
        symbol="600519",
        side=OrderSide.BUY,
        strength=SignalStrength.STRONG,
        confidence=0.75,
        entry_price=100.0,
        target_price=110.0,
        stop_loss=95.0,
    )
    assert signal.is_valid
    assert signal.strategy_name == "test"
    assert signal.symbol == "600519"
    assert signal.side == OrderSide.BUY
    assert signal.confidence == 0.75


def test_signal_invalid() -> None:
    """Test invalid signal."""
    signal = Signal(
        strategy_name="test",
        symbol="600519",
        side=None,
        strength=SignalStrength.WEAK,
        confidence=0.0,
    )
    assert not signal.is_valid


def test_signal_to_dict() -> None:
    """Test Signal to_dict method."""
    signal = Signal(
        strategy_name="test",
        symbol="600519",
        side=OrderSide.BUY,
        strength=SignalStrength.MODERATE,
        confidence=0.6,
    )
    d = signal.to_dict()
    assert d["strategy_name"] == "test"
    assert d["symbol"] == "600519"
    assert d["side"] == "buy"  # OrderSide.BUY.value is lowercase
    assert d["confidence"] == 0.6


def test_momentum_breakout_no_data() -> None:
    """Test momentum breakout with insufficient data."""
    strategy = MomentumBreakoutStrategy()
    data = {"bars": [], "symbol": "600519"}
    signal = strategy.generate_signal(data)
    assert signal is None


def test_mean_reversion_no_data() -> None:
    """Test mean reversion with insufficient data."""
    strategy = MeanReversionStrategy()
    data = {"bars": [], "symbol": "600519"}
    signal = strategy.generate_signal(data)
    assert signal is None


def test_trend_following_no_data() -> None:
    """Test trend following with insufficient data."""
    strategy = TrendFollowingStrategy()
    data = {"bars": [], "symbol": "600519"}
    signal = strategy.generate_signal(data)
    assert signal is None


def test_volume_profile_no_data() -> None:
    """Test volume profile with insufficient data."""
    strategy = VolumeProfileStrategy()
    data = {"bars": [], "symbol": "600519"}
    signal = strategy.generate_signal(data)
    assert signal is None


def test_support_resistance_no_data() -> None:
    """Test support/resistance with insufficient data."""
    strategy = SupportResistanceStrategy()
    data = {"bars": [], "symbol": "600519"}
    signal = strategy.generate_signal(data)
    assert signal is None


def test_macd_divergence_no_data() -> None:
    """Test MACD divergence with insufficient data."""
    strategy = MACDDivergenceStrategy()
    data = {"bars": [], "symbol": "600519"}
    signal = strategy.generate_signal(data)
    assert signal is None


def test_bollinger_breakout_no_data() -> None:
    """Test Bollinger breakout with insufficient data."""
    strategy = BollingerBreakoutStrategy()
    data = {"bars": [], "symbol": "600519"}
    signal = strategy.generate_signal(data)
    assert signal is None


def test_rsi_oversold_no_data() -> None:
    """Test RSI oversold with insufficient data."""
    strategy = RSIOversoldStrategy()
    data = {"bars": [], "symbol": "600519"}
    signal = strategy.generate_signal(data)
    assert signal is None


def test_golden_cross_no_data() -> None:
    """Test golden cross with insufficient data."""
    strategy = GoldenCrossStrategy()
    data = {"bars": [], "symbol": "600519"}
    signal = strategy.generate_signal(data)
    assert signal is None


def test_gap_and_go_no_data() -> None:
    """Test gap and go with insufficient data."""
    strategy = GapAndGoStrategy()
    data = {"bars": [], "symbol": "600519"}
    signal = strategy.generate_signal(data)
    assert signal is None


def test_vwap_reversion_no_data() -> None:
    """Test VWAP reversion with insufficient data."""
    strategy = VWAPReversionStrategy()
    data = {"bars": [], "symbol": "600519"}
    signal = strategy.generate_signal(data)
    assert signal is None


def test_earnings_momentum_no_data() -> None:
    """Test earnings momentum with insufficient data."""
    strategy = EarningsMomentumStrategy()
    data = {"bars": [], "symbol": "600519"}
    signal = strategy.generate_signal(data)
    assert signal is None


def test_strategy_get_info() -> None:
    """Test strategy get_info method."""
    strategy = MomentumBreakoutStrategy()
    info = strategy.get_info()
    assert info["name"] == "momentum_breakout"
    assert "description" in info
    assert "version" in info
    assert "params" in info
    assert "min_confidence" in info


def test_register_strategy_decorator() -> None:
    """Test register_strategy decorator."""
    @register_strategy
    class TestStrategy(BaseStrategy):
        name = "test_strategy_xyz"
        description = "Test strategy"
        version = "1.0.0"
        
        def generate_signal(self, data) -> None:
            return None
    
    assert "test_strategy_xyz" in list_strategies()
    strategy = get_strategy("test_strategy_xyz")
    assert strategy is not None
    assert isinstance(strategy, TestStrategy)
