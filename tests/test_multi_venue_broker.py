from datetime import datetime

from config.settings import CONFIG
from core.types import Account, Fill, Order, OrderSide, OrderStatus, OrderType
from trading.broker import BrokerInterface, MultiVenueBroker, create_broker


class _DummyVenue(BrokerInterface):
    def __init__(self, name: str, fail_submit: bool = False):
        super().__init__()
        self._name = name
        self._connected = True
        self._fail_submit = fail_submit
        self._reject_submit = False
        self._reject_message = "Not connected to broker"
        self.submit_calls = 0
        self.cancel_calls = 0
        self.fills: list[Fill] = []

    @property
    def name(self) -> str:
        return self._name

    @property
    def is_connected(self) -> bool:
        return self._connected

    def connect(self, **kwargs) -> bool:
        self._connected = True
        return True

    def disconnect(self):
        self._connected = False

    def get_account(self) -> Account:
        return Account(broker_name=self._name)

    def get_positions(self):
        return {}

    def get_position(self, symbol: str):
        return None

    def submit_order(self, order: Order) -> Order:
        self.submit_calls += 1
        if self._fail_submit:
            raise RuntimeError(f"{self._name} down")
        if self._reject_submit:
            order.status = OrderStatus.REJECTED
            order.message = self._reject_message
            order.updated_at = datetime.now()
            return order
        order.status = OrderStatus.SUBMITTED
        order.broker_id = f"{self._name}-ok"
        order.updated_at = datetime.now()
        return order

    def cancel_order(self, order_id: str) -> bool:
        self.cancel_calls += 1
        return True

    def get_orders(self, active_only: bool = True):
        return []

    def get_quote(self, symbol: str):
        return 10.0

    def get_fills(self, since: datetime = None):
        return list(self.fills)

    def get_order_status(self, order_id: str):
        return OrderStatus.SUBMITTED

    def sync_order(self, order: Order) -> Order:
        return order


def test_multi_venue_failover_on_submit():
    primary = _DummyVenue("primary", fail_submit=True)
    secondary = _DummyVenue("secondary", fail_submit=False)
    router = MultiVenueBroker([primary, secondary], failover_cooldown_seconds=60)

    order = Order(symbol="600519", side=OrderSide.BUY, order_type=OrderType.LIMIT, quantity=100, price=10.0)
    out = router.submit_order(order)
    assert out.status == OrderStatus.SUBMITTED
    assert out.broker_id == "secondary-ok"
    assert primary.submit_calls == 1
    assert secondary.submit_calls == 1

    snap = router.get_health_snapshot()
    assert snap["active_venue"] == "secondary"


def test_create_broker_live_multi_venue_from_config():
    old_enable = getattr(CONFIG.trading, "enable_multi_venue", False)
    old_priority = list(getattr(CONFIG.trading, "venue_priority", []))
    old_cooldown = getattr(CONFIG.trading, "venue_failover_cooldown_seconds", 30)
    try:
        CONFIG.trading.enable_multi_venue = True
        CONFIG.trading.venue_priority = ["ths", "zszq"]
        CONFIG.trading.venue_failover_cooldown_seconds = 15
        broker = create_broker("live")
        assert isinstance(broker, MultiVenueBroker)
        assert broker.get_health_snapshot()["cooldown_seconds"] == 15
    finally:
        CONFIG.trading.enable_multi_venue = old_enable
        CONFIG.trading.venue_priority = old_priority
        CONFIG.trading.venue_failover_cooldown_seconds = old_cooldown


def test_multi_venue_failover_on_transient_reject():
    primary = _DummyVenue("primary", fail_submit=False)
    primary._reject_submit = True
    primary._reject_message = "Not connected to broker"
    secondary = _DummyVenue("secondary", fail_submit=False)
    router = MultiVenueBroker([primary, secondary], failover_cooldown_seconds=60)

    order = Order(symbol="600519", side=OrderSide.BUY, order_type=OrderType.LIMIT, quantity=100, price=10.0)
    out = router.submit_order(order)
    assert out.status == OrderStatus.SUBMITTED
    assert out.broker_id == "secondary-ok"
    assert primary.submit_calls == 1
    assert secondary.submit_calls == 1


def test_multi_venue_business_reject_no_failover():
    primary = _DummyVenue("primary", fail_submit=False)
    primary._reject_submit = True
    primary._reject_message = "Insufficient funds"
    secondary = _DummyVenue("secondary", fail_submit=False)
    router = MultiVenueBroker([primary, secondary], failover_cooldown_seconds=60)

    order = Order(symbol="600519", side=OrderSide.BUY, order_type=OrderType.LIMIT, quantity=100, price=10.0)
    out = router.submit_order(order)
    assert out.status == OrderStatus.REJECTED
    assert "Insufficient funds" in (out.message or "")
    assert primary.submit_calls == 1
    assert secondary.submit_calls == 0


def test_multi_venue_health_snapshot_cooldown_until_zero_when_no_failure():
    primary = _DummyVenue("primary", fail_submit=False)
    secondary = _DummyVenue("secondary", fail_submit=False)
    router = MultiVenueBroker([primary, secondary], failover_cooldown_seconds=60)
    snap = router.get_health_snapshot()
    assert snap["venues"][0]["cooldown_until"] == 0.0
    assert snap["venues"][1]["cooldown_until"] == 0.0


def test_multi_venue_aggregates_fills_from_all_connected_venues():
    primary = _DummyVenue("primary", fail_submit=False)
    secondary = _DummyVenue("secondary", fail_submit=False)

    f1 = Fill(order_id="O1", symbol="600519", side=OrderSide.BUY, quantity=100, price=10.0)
    f1.id = "F1"
    f2 = Fill(order_id="O2", symbol="000001", side=OrderSide.SELL, quantity=100, price=9.9)
    f2.id = "F2"
    primary.fills = [f1]
    secondary.fills = [f2]

    router = MultiVenueBroker([primary, secondary], failover_cooldown_seconds=60)
    out = router.get_fills()
    out_ids = {x.id for x in out}
    assert out_ids == {"F1", "F2"}


def test_multi_venue_cancel_prefers_order_affinity_venue():
    primary = _DummyVenue("primary", fail_submit=True)
    secondary = _DummyVenue("secondary", fail_submit=False)
    router = MultiVenueBroker([primary, secondary], failover_cooldown_seconds=60)

    order = Order(
        symbol="600519",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        quantity=100,
        price=10.0,
    )
    out = router.submit_order(order)
    assert out.broker_id == "secondary-ok"

    assert router.cancel_order(order.id) is True
    assert secondary.cancel_calls == 1
    assert primary.cancel_calls == 0
