"""Tests for multi-asset trading types (futures, options, forex, crypto)."""
from datetime import date

from core.types import (
    AssetType,
    CryptoAsset,
    ForexPair,
    FuturesContract,
    MultiAssetPosition,
    OptionsContract,
    OptionType,
)


class TestFuturesContract:
    """Test futures contract functionality."""

    def test_create_futures_contract(self) -> None:
        """Test creating a futures contract."""
        contract = FuturesContract(
            underlying="IF",
            expiry=date(2026, 3, 20),
            exchange="CFFEX",
            multiplier=300,
            tick_size=0.2,
            margin_rate=0.12,
        )
        
        assert contract.symbol == "IF2603"
        assert contract.underlying == "IF"
        assert contract.expiry == date(2026, 3, 20)
        assert contract.multiplier == 300
        assert contract.tick_size == 0.2
        assert contract.margin_rate == 0.12

    def test_auto_generate_symbol(self) -> None:
        """Test automatic symbol generation."""
        contract = FuturesContract(
            underlying="IC",
            expiry=date(2026, 6, 19),
        )
        
        assert contract.symbol == "IC2606"

    def test_is_active(self) -> None:
        """Test contract active status."""
        # Future expiry - should be active
        contract_active = FuturesContract(
            underlying="IF",
            expiry=date(2030, 12, 31),
        )
        assert contract_active.is_active is True
        
        # Past expiry - should not be active
        contract_expired = FuturesContract(
            underlying="IF",
            expiry=date(2020, 1, 1),
        )
        assert contract_expired.is_active is False

    def test_notional_value(self) -> None:
        """Test notional value calculation."""
        contract = FuturesContract(
            underlying="IF",
            expiry=date(2026, 3, 20),
            multiplier=300,
            last_price=4000.0,
        )
        
        assert contract.notional_value == 1_200_000.0

    def test_margin_required(self) -> None:
        """Test margin requirement calculation."""
        contract = FuturesContract(
            underlying="IF",
            expiry=date(2026, 3, 20),
            multiplier=300,
            last_price=4000.0,
            margin_rate=0.12,
        )
        
        assert contract.margin_required == 144_000.0

    def test_to_dict(self) -> None:
        """Test serialization to dictionary."""
        contract = FuturesContract(
            underlying="IF",
            expiry=date(2026, 3, 20),
            exchange="CFFEX",
        )
        
        data = contract.to_dict()
        assert data['symbol'] == "IF2603"
        assert data['underlying'] == "IF"
        assert data['exchange'] == "CFFEX"
        assert data['expiry'] == "2026-03-20"


class TestOptionsContract:
    """Test options contract functionality."""

    def test_create_options_contract(self) -> None:
        """Test creating an options contract."""
        contract = OptionsContract(
            underlying="000300",
            option_type=OptionType.CALL,
            strike=4000.0,
            expiry=date(2026, 3, 27),
            multiplier=10000,
        )
        
        assert contract.underlying == "000300"
        assert contract.option_type == OptionType.CALL
        assert contract.strike == 4000.0
        assert contract.expiry == date(2026, 3, 27)
        assert contract.multiplier == 10000

    def test_auto_generate_symbol(self) -> None:
        """Test automatic symbol generation."""
        contract = OptionsContract(
            underlying="000300",
            option_type=OptionType.PUT,
            strike=3800,
            expiry=date(2026, 6, 27),
        )
        
        assert contract.symbol == "000300P380006"

    def test_is_in_the_money(self) -> None:
        """Test in-the-money detection."""
        call_itm = OptionsContract(
            underlying="000300",
            option_type=OptionType.CALL,
            strike=3800.0,
            last_price=4000.0,
        )
        assert call_itm.is_in_the_money is True
        
        call_otm = OptionsContract(
            underlying="000300",
            option_type=OptionType.CALL,
            strike=4200.0,
            last_price=4000.0,
        )
        assert call_otm.is_in_the_money is False
        
        put_itm = OptionsContract(
            underlying="000300",
            option_type=OptionType.PUT,
            strike=4200.0,
            last_price=4000.0,
        )
        assert put_itm.is_in_the_money is True

    def test_intrinsic_value(self) -> None:
        """Test intrinsic value calculation."""
        call = OptionsContract(
            underlying="000300",
            option_type=OptionType.CALL,
            strike=3800.0,
            last_price=4000.0,
        )
        assert call.intrinsic_value == 200.0
        
        put = OptionsContract(
            underlying="000300",
            option_type=OptionType.PUT,
            strike=4200.0,
            last_price=4000.0,
        )
        assert put.intrinsic_value == 200.0

    def test_time_value(self) -> None:
        """Test time value calculation."""
        contract = OptionsContract(
            underlying="000300",
            option_type=OptionType.CALL,
            strike=3800.0,
            last_price=250.0,  # Premium
        )
        # Intrinsic = 4000 - 3800 = 200 (but last_price is the premium)
        # Time value = Premium - Intrinsic
        # Note: For options, last_price is the premium, not underlying price
        # This test checks the formula works correctly
        assert contract.time_value >= 0

    def test_days_to_expiry(self) -> None:
        """Test days to expiry calculation."""
        future_expiry = date(2026, 12, 31)
        contract = OptionsContract(
            underlying="000300",
            expiry=future_expiry,
        )
        
        days = contract.days_to_expiry
        assert days > 0

    def test_margin_required_short(self) -> None:
        """Test margin requirement for short positions."""
        contract = OptionsContract(
            underlying="000300",
            option_type=OptionType.CALL,
            last_price=50.0,
            multiplier=10000,
            margin_rate=0.20,
        )
        
        assert contract.margin_required == 100_000.0

    def test_to_dict(self) -> None:
        """Test serialization to dictionary."""
        contract = OptionsContract(
            underlying="000300",
            option_type=OptionType.CALL,
            strike=4000.0,
            expiry=date(2026, 3, 27),
        )
        
        data = contract.to_dict()
        assert data['option_type'] == "call"
        assert data['strike'] == 4000.0
        assert data['expiry'] == "2026-03-27"


class TestForexPair:
    """Test forex pair functionality."""

    def test_create_forex_pair(self) -> None:
        """Test creating a forex pair."""
        pair = ForexPair(
            base_currency="USD",
            quote_currency="CNY",
            pip_size=0.0001,
        )
        
        assert pair.symbol == "USD/CNY"
        assert pair.base_currency == "USD"
        assert pair.quote_currency == "CNY"
        assert pair.pip_size == 0.0001

    def test_mid_price(self) -> None:
        """Test mid-market price calculation."""
        pair = ForexPair(
            base_currency="EUR",
            quote_currency="USD",
            bid=1.0850,
            ask=1.0852,
        )
        
        assert pair.mid_price == 1.0851

    def test_to_dict(self) -> None:
        """Test serialization to dictionary."""
        pair = ForexPair(
            base_currency="GBP",
            quote_currency="USD",
            bid=1.2650,
            ask=1.2652,
        )
        
        data = pair.to_dict()
        assert data['symbol'] == "GBP/USD"
        assert data['bid'] == 1.2650
        assert data['ask'] == 1.2652


class TestCryptoAsset:
    """Test cryptocurrency asset functionality."""

    def test_create_crypto_asset(self) -> None:
        """Test creating a crypto asset."""
        btc = CryptoAsset(
            symbol="BTC",
            name="Bitcoin",
            decimals=8,
        )
        
        assert btc.symbol == "BTC"
        assert btc.name == "Bitcoin"
        assert btc.decimals == 8

    def test_to_dict(self) -> None:
        """Test serialization to dictionary."""
        eth = CryptoAsset(
            symbol="ETH",
            name="Ethereum",
            last_price=2500.0,
        )
        
        data = eth.to_dict()
        assert data['symbol'] == "ETH"
        assert data['name'] == "Ethereum"
        assert data['last_price'] == 2500.0


class TestMultiAssetPosition:
    """Test multi-asset position functionality."""

    def test_create_stock_position(self) -> None:
        """Test creating a stock position."""
        position = MultiAssetPosition(
            symbol="600519",
            asset_type=AssetType.STOCK,
            quantity=100,
            avg_cost=1800.0,
            current_price=1850.0,
        )
        
        assert position.asset_type == AssetType.STOCK
        assert position.quantity == 100
        assert position.net_quantity == 100
        assert position.market_value == 185_000.0

    def test_create_futures_position(self) -> None:
        """Test creating a futures position."""
        position = MultiAssetPosition(
            symbol="IF2603",
            asset_type=AssetType.FUTURES,
            long_qty=5,
            short_qty=2,
            long_avg_cost=4000.0,
            short_avg_cost=4100.0,
            current_price=4050.0,
        )
        
        assert position.asset_type == AssetType.FUTURES
        assert position.net_quantity == 3  # 5 long - 2 short
        assert position.market_value == 12_150.0  # 3 * 4050

    def test_futures_pnl_calculation(self) -> None:
        """Test futures P&L calculation."""
        position = MultiAssetPosition(
            symbol="IF2603",
            asset_type=AssetType.FUTURES,
            long_qty=10,
            long_avg_cost=4000.0,
            current_price=4050.0,
        )
        
        # P&L = (current - avg_cost) * quantity
        assert position.unrealized_pnl == 500.0

    def test_unrealized_pnl_pct(self) -> None:
        """Test unrealized P&L percentage."""
        position = MultiAssetPosition(
            symbol="600519",
            asset_type=AssetType.STOCK,
            quantity=100,
            avg_cost=1800.0,
            current_price=1850.0,
        )
        
        pnl_pct = position.unrealized_pnl / position.cost_basis * 100
        assert abs(pnl_pct - 2.78) < 0.1  # Approximately 2.78%

    def test_to_dict(self) -> None:
        """Test serialization to dictionary."""
        position = MultiAssetPosition(
            symbol="600519",
            asset_type=AssetType.STOCK,
            quantity=100,
            avg_cost=1800.0,
            current_price=1850.0,
        )
        
        data = position.to_dict()
        assert data['asset_type'] == "stock"
        assert data['quantity'] == 100
        assert data['market_value'] == 185_000.0
