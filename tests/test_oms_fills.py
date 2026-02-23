def test_fill_dedup_with_broker_fill_id_allows_same_second_multi_fills(tmp_path) -> None:
    from datetime import datetime

    from core.types import Fill, Order, OrderSide, OrderType
    from trading.oms import get_oms, reset_oms

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


def test_order_timeline_records_submit_and_fill(tmp_path) -> None:
    from datetime import datetime

    from core.types import Fill, Order, OrderSide, OrderType
    from trading.oms import get_oms, reset_oms

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


def test_get_order_roundtrip_with_sqlite_row_factory(tmp_path) -> None:
    from core.types import Order, OrderSide, OrderType
    from trading.oms import get_oms, reset_oms

    db_path = tmp_path / "orders.db"
    reset_oms()
    oms = get_oms(initial_capital=100000, db_path=db_path)

    submitted = oms.submit_order(
        Order(
            symbol="600519",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=100,
            price=100.0,
        )
    )

    loaded = oms.get_order(submitted.id)
    assert loaded is not None
    assert loaded.id == submitted.id


def test_order_parent_id_roundtrip_persists(tmp_path) -> None:
    from core.types import Order, OrderSide, OrderType
    from trading.oms import get_oms, reset_oms

    db_path = tmp_path / "orders.db"
    reset_oms()
    oms = get_oms(initial_capital=100000, db_path=db_path)

    submitted = oms.submit_order(
        Order(
            symbol="600519",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=100,
            price=100.0,
            parent_id="PARENT-001",
        )
    )

    loaded = oms.get_order(submitted.id)
    assert loaded is not None
    assert loaded.parent_id == "PARENT-001"


def test_legacy_orders_table_is_migrated_for_parent_id(tmp_path) -> None:
    import sqlite3

    from core.types import Order, OrderSide, OrderType
    from trading.oms import get_oms, reset_oms

    db_path = tmp_path / "orders.db"

    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE orders (
            id TEXT PRIMARY KEY,
            broker_id TEXT,
            symbol TEXT NOT NULL,
            name TEXT,
            side TEXT NOT NULL,
            order_type TEXT NOT NULL,
            quantity INTEGER NOT NULL,
            price REAL,
            stop_price REAL,
            status TEXT NOT NULL,
            filled_qty INTEGER DEFAULT 0,
            avg_price REAL DEFAULT 0,
            commission REAL DEFAULT 0,
            message TEXT,
            strategy TEXT,
            signal_id TEXT,
            stop_loss REAL,
            take_profit REAL,
            created_at TEXT,
            submitted_at TEXT,
            filled_at TEXT,
            cancelled_at TEXT,
            updated_at TEXT,
            tags TEXT
        )
        """
    )
    conn.commit()
    conn.close()

    reset_oms()
    oms = get_oms(initial_capital=100000, db_path=db_path)
    submitted = oms.submit_order(
        Order(
            symbol="600519",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=100,
            price=100.0,
            parent_id="LEGACY-PARENT",
        )
    )

    loaded = oms.get_order(submitted.id)
    assert loaded is not None
    assert loaded.parent_id == "LEGACY-PARENT"


def test_get_order_tolerates_malformed_tags_json(tmp_path) -> None:
    import sqlite3

    from core.types import Order, OrderSide, OrderType
    from trading.oms import get_oms, reset_oms

    db_path = tmp_path / "orders.db"
    reset_oms()
    oms = get_oms(initial_capital=100000, db_path=db_path)

    submitted = oms.submit_order(
        Order(
            symbol="600519",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=100,
            price=100.0,
        )
    )

    conn = sqlite3.connect(db_path)
    conn.execute(
        "UPDATE orders SET tags = ? WHERE id = ?",
        ("{not-valid-json", submitted.id),
    )
    conn.commit()
    conn.close()

    loaded = oms.get_order(submitted.id)
    assert loaded is not None
    assert loaded.tags == {}


def test_process_fill_rejects_side_mismatch_without_mutation(tmp_path) -> None:
    from datetime import datetime

    import pytest

    from core.exceptions import OrderValidationError
    from core.types import Fill, Order, OrderSide, OrderType
    from trading.oms import get_oms, reset_oms

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
    cash_before = oms.get_account().cash

    bad_fill = Fill(
        id="F_BAD_SIDE",
        order_id=order.id,
        symbol=order.symbol,
        side=OrderSide.SELL,
        quantity=100,
        price=100.0,
        commission=5.0,
        timestamp=datetime.now(),
    )

    with pytest.raises(OrderValidationError, match="side mismatch"):
        oms.process_fill(order, bad_fill)

    loaded = oms.get_order(order.id)
    assert loaded is not None
    assert loaded.filled_qty == 0
    assert oms.get_account().cash == cash_before
    assert len(oms.get_fills(order.id)) == 0


def test_process_fill_rejects_overfill(tmp_path) -> None:
    from datetime import datetime

    import pytest

    from core.exceptions import OrderValidationError
    from core.types import Fill, Order, OrderSide, OrderStatus, OrderType
    from trading.oms import get_oms, reset_oms

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

    too_large = Fill(
        id="F_OVERFILL",
        order_id=order.id,
        symbol=order.symbol,
        side=OrderSide.BUY,
        quantity=150,
        price=100.0,
        commission=5.0,
        timestamp=datetime.now(),
    )

    with pytest.raises(OrderValidationError, match="exceeds remaining"):
        oms.process_fill(order, too_large)

    loaded = oms.get_order(order.id)
    assert loaded is not None
    assert loaded.status == OrderStatus.SUBMITTED
    assert loaded.filled_qty == 0
    assert len(oms.get_fills(order.id)) == 0


def test_partial_fill_publishes_partial_event_type(tmp_path) -> None:
    from datetime import datetime

    from core.events import EVENT_BUS, EventType
    from core.types import Fill, Order, OrderSide, OrderType
    from trading.oms import get_oms, reset_oms

    db_path = tmp_path / "orders.db"
    EVENT_BUS.clear_history()
    reset_oms()
    oms = get_oms(initial_capital=100000, db_path=db_path)

    order = Order(
        symbol="600519",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        quantity=200,
        price=100.0,
    )
    oms.submit_order(order)

    fill = Fill(
        id="F_PARTIAL",
        order_id=order.id,
        symbol=order.symbol,
        side=OrderSide.BUY,
        quantity=100,
        price=100.0,
        commission=5.0,
        timestamp=datetime.now(),
    )
    oms.process_fill(order, fill)

    history = EVENT_BUS.get_history(limit=10)
    assert history
    assert history[-1].type == EventType.ORDER_PARTIALLY_FILLED


def test_buy_fill_overrun_preserves_negative_cash(tmp_path) -> None:
    from datetime import datetime

    from core.types import Fill, Order, OrderSide, OrderType
    from trading.oms import get_oms, reset_oms

    db_path = tmp_path / "orders.db"
    reset_oms()
    oms = get_oms(initial_capital=11000, db_path=db_path)

    order = Order(
        symbol="600519",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        quantity=100,
        price=100.0,
    )
    oms.submit_order(order)

    expensive_fill = Fill(
        id="F_OVERRUN",
        order_id=order.id,
        symbol=order.symbol,
        side=OrderSide.BUY,
        quantity=100,
        price=200.0,
        commission=5.0,
        timestamp=datetime.now(),
    )
    oms.process_fill(order, expensive_fill)

    account = oms.get_account()
    assert account.cash < 0.0
    assert account.available == 0.0
    assert account.frozen == 0.0


def test_sell_fill_increases_available_cash(tmp_path) -> None:
    from datetime import datetime

    from core.types import Fill, Order, OrderSide, OrderType
    from trading.oms import get_oms, reset_oms

    db_path = tmp_path / "orders.db"
    reset_oms()
    oms = get_oms(initial_capital=100000, db_path=db_path)

    buy = Order(
        symbol="600519",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        quantity=100,
        price=100.0,
    )
    oms.submit_order(buy)
    oms.process_fill(
        buy,
        Fill(
            id="F_BUY_SEED",
            order_id=buy.id,
            symbol=buy.symbol,
            side=OrderSide.BUY,
            quantity=100,
            price=100.0,
            commission=5.0,
            timestamp=datetime.now(),
        ),
    )

    pos = oms.get_position("600519")
    assert pos is not None
    pos.available_qty = pos.quantity
    oms._db.save_position(pos)

    sell = Order(
        symbol="600519",
        side=OrderSide.SELL,
        order_type=OrderType.LIMIT,
        quantity=100,
        price=110.0,
    )
    oms.submit_order(sell)
    before_cash = float(oms.get_account().cash)
    before_available = float(oms.get_account().available)

    commission = 5.0
    proceeds = (100 * 110.0) - commission
    oms.process_fill(
        sell,
        Fill(
            id="F_SELL",
            order_id=sell.id,
            symbol=sell.symbol,
            side=OrderSide.SELL,
            quantity=100,
            price=110.0,
            commission=commission,
            timestamp=datetime.now(),
        ),
    )

    account = oms.get_account()
    assert round(float(account.cash) - before_cash, 6) == round(proceeds, 6)
    assert (
        round(float(account.available) - before_available, 6)
        == round(proceeds, 6)
    )


def test_buy_overrun_preserves_other_active_buy_reservations(tmp_path) -> None:
    from datetime import datetime

    from core.types import Fill, Order, OrderSide, OrderType
    from trading.oms import get_oms, reset_oms

    db_path = tmp_path / "orders.db"
    reset_oms()
    oms = get_oms(initial_capital=11000, db_path=db_path)

    o1 = Order(
        symbol="600519",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        quantity=100,
        price=50.0,
    )
    o2 = Order(
        symbol="600519",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        quantity=100,
        price=50.0,
    )
    oms.submit_order(o1)
    oms.submit_order(o2)

    oms.process_fill(
        o1,
        Fill(
            id="F_OVERRUN_MULTI",
            order_id=o1.id,
            symbol=o1.symbol,
            side=OrderSide.BUY,
            quantity=100,
            price=150.0,
            commission=5.0,
            timestamp=datetime.now(),
        ),
    )

    account = oms.get_account()
    reloaded_o2 = oms.get_order(o2.id)
    assert reloaded_o2 is not None
    assert float(account.cash) < 0.0
    assert float(account.available) == 0.0
    assert float(account.frozen) > 0.0
    assert float(reloaded_o2.tags.get("reserved_cash_remaining", 0.0)) > 0.0


def test_recovery_clears_stale_frozen_qty_when_no_active_sell_orders(tmp_path) -> None:
    from core.types import Position
    from trading.oms import get_oms, reset_oms

    db_path = tmp_path / "orders.db"
    reset_oms()
    oms = get_oms(initial_capital=100000, db_path=db_path)

    pos = Position(
        symbol="600519",
        quantity=100,
        available_qty=0,
        frozen_qty=100,
    )
    oms.get_account().positions["600519"] = pos
    oms._db.save_position(pos)
    oms._db.save_account_state(oms.get_account())

    reset_oms()
    recovered = get_oms(initial_capital=100000, db_path=db_path)
    recovered_pos = recovered.get_position("600519")
    assert recovered_pos is not None
    assert int(recovered_pos.frozen_qty) == 0
    assert int(recovered_pos.available_qty) == 100
