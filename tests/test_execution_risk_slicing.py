import pytest

from core.types import Order, OrderSide, OrderType

try:
    from trading.execution_risk import SmartOrderRouter

    _EXECUTION_STACK_AVAILABLE = True
except ImportError:
    _EXECUTION_STACK_AVAILABLE = False
    SmartOrderRouter = None  # type: ignore[assignment]

pytestmark = pytest.mark.skipif(
    not _EXECUTION_STACK_AVAILABLE,
    reason="Execution stack modules are removed in analysis-only build.",
)


def test_slice_order_sets_parent_id_on_child_orders() -> None:
    router = SmartOrderRouter(
        order_split_threshold=1_000.0,
        max_participation_rate=0.05,
    )
    parent = Order(
        id="ORD_PARENT_001",
        symbol="600519",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        quantity=1_000,
        price=20.0,
    )

    children = router.slice_order(parent, daily_volume=10_000, max_participation=0.05)

    assert len(children) > 1
    assert sum(int(c.quantity) for c in children) == int(parent.quantity)
    assert all(c.parent_id == parent.id for c in children)
    assert all(c.symbol == parent.symbol for c in children)
    assert all(c.side == parent.side for c in children)
