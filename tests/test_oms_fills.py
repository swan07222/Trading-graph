def test_fill_dedup_with_broker_fill_id_allows_same_second_multi_fills(tmp_path):
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


def test_order_timeline_records_submit_and_fill(tmp_path):
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


def test_get_order_roundtrip_with_sqlite_row_factory(tmp_path):
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


def test_order_parent_id_roundtrip_persists(tmp_path):
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


def test_legacy_orders_table_is_migrated_for_parent_id(tmp_path):
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


def test_get_order_tolerates_malformed_tags_json(tmp_path):
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
