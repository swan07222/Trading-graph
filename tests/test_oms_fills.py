def test_fill_dedup_with_broker_fill_id_allows_same_second_multi_fills(tmp_path):
    from trading.oms import get_oms, reset_oms
    from core.types import Order, OrderSide, OrderType, Fill
    from datetime import datetime

    db_path = tmp_path / "orders.db"
    reset_oms()
    oms = get_oms(initial_capital=100000, db_path=db_path)

    order = Order(symbol="600519", side=OrderSide.BUY, order_type=OrderType.LIMIT, quantity=200, price=100.0)
    oms.submit_order(order)

    ts = datetime.now().replace(microsecond=0)  # same-second timestamp

    f1 = Fill(
        id="F1", broker_fill_id="B1",
        order_id=order.id, symbol=order.symbol, side=OrderSide.BUY,
        quantity=100, price=100.0, commission=5.0, timestamp=ts
    )
    f2 = Fill(
        id="F2", broker_fill_id="B2",
        order_id=order.id, symbol=order.symbol, side=OrderSide.BUY,
        quantity=100, price=100.0, commission=5.0, timestamp=ts
    )

    oms.process_fill(order, f1)
    oms.process_fill(order, f2)

    fills = oms.get_fills(order.id)
    assert len(fills) == 2, "Should keep both fills even if same qty/price/timestamp (broker_fill_id differs)"


def test_order_timeline_records_submit_and_fill(tmp_path):
    from trading.oms import get_oms, reset_oms
    from core.types import Order, OrderSide, OrderType, Fill
    from datetime import datetime

    db_path = tmp_path / "orders.db"
    reset_oms()
    oms = get_oms(initial_capital=100000, db_path=db_path)

    order = Order(
        symbol="600519",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        quantity=100,
        price=100.0,
    )
    oms.submit_order(order)

    fill = Fill(
        id="F100",
        broker_fill_id="B100",
        order_id=order.id,
        symbol=order.symbol,
        side=OrderSide.BUY,
        quantity=100,
        price=100.0,
        commission=5.0,
        timestamp=datetime.now(),
    )
    oms.process_fill(order, fill)

    timeline = oms.get_order_timeline(order.id)
    event_types = [e.get("event_type") for e in timeline]
    assert "submitted" in event_types
    assert "fill" in event_types
