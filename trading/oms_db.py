# trading/oms_db.py
import json
import sqlite3
import threading
from contextlib import contextmanager
from datetime import date, datetime, timedelta
from pathlib import Path

from config.settings import CONFIG
from core.types import Account, Fill, Order, OrderSide, OrderStatus, OrderType, Position
from utils.logger import get_logger

log = get_logger(__name__)

class OrderDatabase:
    """SQLite-backed order persistence.
    Uses WAL mode for better concurrency.
    """

    def __init__(self, db_path: Path = None):
        self._db_path = (
            Path(db_path) if db_path
            else (CONFIG.data_dir / "orders.db")
        )
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._init_lock = threading.Lock()
        self._init_db()

    @property
    def _conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, 'conn') or self._local.conn is None:
            self._local.conn = sqlite3.connect(
                str(self._db_path),
                check_same_thread=False,
                timeout=30
            )
            self._local.conn.row_factory = sqlite3.Row
            self._local.conn.execute("PRAGMA journal_mode=WAL")
            self._local.conn.execute("PRAGMA synchronous=NORMAL")
            self._local.conn.execute("PRAGMA foreign_keys=ON")
        return self._local.conn

    def close_connection(self):
        """Close thread-local connection."""
        if hasattr(self._local, 'conn') and self._local.conn:
            try:
                self._local.conn.close()
            except Exception as e:
                log.debug("OMS DB close_connection failed: %s", e)
            self._local.conn = None

    @contextmanager
    def transaction(self):
        """Context manager for atomic transactions."""
        conn = self._conn
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    def _init_db(self):
        with self._init_lock:
            with self.transaction() as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS orders (
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
                        parent_id TEXT,
                        stop_loss REAL,
                        take_profit REAL,
                        created_at TEXT,
                        submitted_at TEXT,
                        filled_at TEXT,
                        cancelled_at TEXT,
                        updated_at TEXT,
                        tags TEXT
                    )
                """)
                self._ensure_orders_schema(conn)

                conn.execute("""
                    CREATE TABLE IF NOT EXISTS fills (
                        id TEXT PRIMARY KEY,
                        order_id TEXT NOT NULL,
                        broker_fill_id TEXT,
                        symbol TEXT NOT NULL,
                        side TEXT NOT NULL,
                        quantity INTEGER NOT NULL,
                        price REAL NOT NULL,
                        commission REAL DEFAULT 0,
                        stamp_tax REAL DEFAULT 0,
                        timestamp TEXT,
                        FOREIGN KEY (order_id) REFERENCES orders(id)
                    )
                """)
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS order_events (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        order_id TEXT NOT NULL,
                        event_type TEXT NOT NULL,
                        old_status TEXT,
                        new_status TEXT,
                        filled_qty INTEGER,
                        avg_price REAL,
                        message TEXT,
                        payload TEXT,
                        created_at TEXT NOT NULL,
                        FOREIGN KEY (order_id) REFERENCES orders(id)
                    )
                """)

                # Dedup index: prefer broker_fill_id when present
                conn.execute("""
                    CREATE UNIQUE INDEX IF NOT EXISTS idx_fills_broker_dedup
                    ON fills(order_id, broker_fill_id)
                    WHERE broker_fill_id IS NOT NULL AND broker_fill_id <> ''
                """)

                # Fallback dedup when broker_fill_id missing
                conn.execute("""
                    CREATE UNIQUE INDEX IF NOT EXISTS idx_fills_composite_dedup
                    ON fills(order_id, quantity, price, timestamp)
                    WHERE broker_fill_id IS NULL OR broker_fill_id = ''
                """)

                conn.execute("""
                    CREATE TABLE IF NOT EXISTS positions (
                        symbol TEXT PRIMARY KEY,
                        name TEXT,
                        quantity INTEGER DEFAULT 0,
                        available_qty INTEGER DEFAULT 0,
                        frozen_qty INTEGER DEFAULT 0,
                        pending_buy INTEGER DEFAULT 0,
                        pending_sell INTEGER DEFAULT 0,
                        avg_cost REAL DEFAULT 0,
                        current_price REAL DEFAULT 0,
                        realized_pnl REAL DEFAULT 0,
                        commission_paid REAL DEFAULT 0,
                        opened_at TEXT,
                        last_updated TEXT
                    )
                """)

                conn.execute("""
                    CREATE TABLE IF NOT EXISTS account_state (
                        id INTEGER PRIMARY KEY CHECK (id = 1),
                        cash REAL DEFAULT 0,
                        available REAL DEFAULT 0,
                        frozen REAL DEFAULT 0,
                        initial_capital REAL DEFAULT 0,
                        realized_pnl REAL DEFAULT 0,
                        commission_paid REAL DEFAULT 0,
                        peak_equity REAL DEFAULT 0,
                        daily_start_equity REAL DEFAULT 0,
                        daily_start_date TEXT,
                        updated_at TEXT
                    )
                """)

                conn.execute("""
                    CREATE TABLE IF NOT EXISTS t1_pending (
                        symbol TEXT,
                        purchase_date TEXT,
                        quantity INTEGER DEFAULT 0,
                        PRIMARY KEY (symbol, purchase_date)
                    )
                """)

                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_orders_status "
                    "ON orders(status)"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_orders_symbol "
                    "ON orders(symbol)"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_orders_broker_id "
                    "ON orders(broker_id)"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_fills_order "
                    "ON fills(order_id)"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_order_events_order_ts "
                    "ON order_events(order_id, created_at)"
                )

    def _ensure_orders_schema(self, conn: sqlite3.Connection) -> None:
        """Apply non-destructive schema upgrades for legacy OMS databases.
        """
        cols = {
            str(row["name"])
            for row in conn.execute("PRAGMA table_info(orders)").fetchall()
        }
        if "parent_id" not in cols:
            conn.execute(
                "ALTER TABLE orders ADD COLUMN parent_id TEXT DEFAULT ''"
            )

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------

    def save_order(self, order: Order, conn: sqlite3.Connection = None):
        """Save or update order. Uses provided conn for transactions."""
        target = conn or self._conn
        auto_commit = conn is None

        try:
            target.execute("""
                INSERT INTO orders (
                    id, broker_id, symbol, name, side, order_type,
                    quantity, price, stop_price, status, filled_qty,
                    avg_price, commission, message, strategy, signal_id, parent_id,
                    stop_loss, take_profit, created_at, submitted_at,
                    filled_at, cancelled_at, updated_at, tags
                )
                VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?, ?, ?, ?
                )
                ON CONFLICT(id) DO UPDATE SET
                    broker_id=excluded.broker_id,
                    symbol=excluded.symbol,
                    name=excluded.name,
                    side=excluded.side,
                    order_type=excluded.order_type,
                    quantity=excluded.quantity,
                    price=excluded.price,
                    stop_price=excluded.stop_price,
                    status=excluded.status,
                    filled_qty=excluded.filled_qty,
                    avg_price=excluded.avg_price,
                    commission=excluded.commission,
                    message=excluded.message,
                    strategy=excluded.strategy,
                    signal_id=excluded.signal_id,
                    parent_id=excluded.parent_id,
                    stop_loss=excluded.stop_loss,
                    take_profit=excluded.take_profit,
                    created_at=excluded.created_at,
                    submitted_at=excluded.submitted_at,
                    filled_at=excluded.filled_at,
                    cancelled_at=excluded.cancelled_at,
                    updated_at=excluded.updated_at,
                    tags=excluded.tags
            """, (
                order.id, order.broker_id, order.symbol, order.name,
                order.side.value, order.order_type.value,
                order.quantity, order.price, order.stop_price,
                order.status.value, order.filled_qty, order.avg_price,
                order.commission, order.message, order.strategy,
                order.signal_id, order.parent_id,
                order.stop_loss, order.take_profit,
                order.created_at.isoformat() if order.created_at else None,
                order.submitted_at.isoformat() if order.submitted_at else None,
                order.filled_at.isoformat() if order.filled_at else None,
                order.cancelled_at.isoformat() if order.cancelled_at else None,
                datetime.now().isoformat(),
                json.dumps(order.tags or {})
            ))

            if auto_commit:
                target.commit()
        except Exception:
            if auto_commit:
                target.rollback()
            raise

    def load_order(self, order_id: str) -> Order | None:
        row = self._conn.execute(
            "SELECT * FROM orders WHERE id = ?", (order_id,)
        ).fetchone()
        return self._row_to_order(row) if row else None

    def load_order_by_broker_id(self, broker_id: str) -> Order | None:
        row = self._conn.execute(
            "SELECT * FROM orders WHERE broker_id = ?", (broker_id,)
        ).fetchone()
        return self._row_to_order(row) if row else None

    def load_active_orders(self) -> list[Order]:
        rows = self._conn.execute("""
            SELECT * FROM orders
            WHERE status IN ('pending', 'submitted', 'accepted', 'partial')
        """).fetchall()
        return [self._row_to_order(r) for r in rows]

    def load_orders_by_symbol(self, symbol: str) -> list[Order]:
        rows = self._conn.execute(
            "SELECT * FROM orders WHERE symbol = ? ORDER BY created_at DESC",
            (symbol,)
        ).fetchall()
        return [self._row_to_order(r) for r in rows]

    def _row_to_order(self, row) -> Order:
        order = Order.__new__(Order)
        # Bypass __post_init__ to avoid generating new id/timestamps
        order.id = row['id']
        order.client_id = ''
        order.broker_id = row['broker_id'] or ''
        order.symbol = row['symbol']
        order.name = row['name'] or ''
        order.side = OrderSide(row['side'])
        order.order_type = OrderType(row['order_type'])
        order.quantity = row['quantity']
        order.price = row['price'] or 0.0
        order.stop_price = row['stop_price'] or 0.0
        order.status = OrderStatus(row['status'])
        order.filled_qty = row['filled_qty'] or 0
        order.filled_price = 0.0
        order.avg_price = row['avg_price'] or 0.0
        order.commission = row['commission'] or 0.0
        order.slippage = 0.0
        order.message = row['message'] or ''
        order.strategy = row['strategy'] or ''
        order.signal_id = row['signal_id'] or ''
        order.parent_id = (
            row['parent_id']
            if 'parent_id' in row.keys() and row['parent_id'] is not None
            else ''
        )
        order.stop_loss = row['stop_loss'] or 0.0
        order.take_profit = row['take_profit'] or 0.0

        order.created_at = (
            datetime.fromisoformat(row['created_at'])
            if row['created_at'] else None
        )
        order.submitted_at = (
            datetime.fromisoformat(row['submitted_at'])
            if row['submitted_at'] else None
        )
        order.filled_at = (
            datetime.fromisoformat(row['filled_at'])
            if row['filled_at'] else None
        )
        order.cancelled_at = (
            datetime.fromisoformat(row['cancelled_at'])
            if row['cancelled_at'] else None
        )
        order.updated_at = (
            datetime.fromisoformat(row['updated_at'])
            if row['updated_at'] else None
        )

        if row['tags']:
            try:
                loaded = json.loads(row['tags'])
                order.tags = loaded if isinstance(loaded, dict) else {}
            except (json.JSONDecodeError, TypeError, ValueError):
                order.tags = {}
        else:
            order.tags = {}

        return order

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------

    def save_fill(self, fill: Fill, conn: sqlite3.Connection = None) -> bool:
        """Save fill with deduplication.
        Returns True if new, False if duplicate.
        """
        target = conn or self._conn
        auto_commit = conn is None
        ts_str = fill.timestamp.isoformat() if fill.timestamp else None
        broker_fill_id = (fill.broker_fill_id or "").strip() or None

        try:
            target.execute("""
                INSERT INTO fills
                (id, order_id, broker_fill_id, symbol, side, quantity,
                 price, commission, stamp_tax, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                fill.id, fill.order_id, broker_fill_id,
                fill.symbol, fill.side.value,
                int(fill.quantity), float(fill.price),
                float(fill.commission), float(fill.stamp_tax),
                ts_str
            ))

            if auto_commit:
                target.commit()
            return True

        except sqlite3.IntegrityError:
            if auto_commit:
                target.rollback()
            return False
        except Exception:
            if auto_commit:
                target.rollback()
            raise

    def fill_exists(self, fill: Fill) -> bool:
        """Check if fill already exists using same types as save_fill."""
        broker_fid = (fill.broker_fill_id or '').strip()

        if broker_fid:
            cur = self._conn.execute(
                "SELECT 1 FROM fills WHERE order_id = ? AND broker_fill_id = ?",
                (fill.order_id, broker_fid),
            )
            return cur.fetchone() is not None

        ts = fill.timestamp.isoformat() if fill.timestamp else None
        cur = self._conn.execute(
            """
            SELECT 1 FROM fills
            WHERE order_id = ?
              AND quantity = ?
              AND price = ?
              AND timestamp = ?
              AND (broker_fill_id IS NULL OR broker_fill_id = '')
            """,
            (fill.order_id, int(fill.quantity), float(fill.price), ts),
        )
        return cur.fetchone() is not None

    def load_fills(self, order_id: str = None) -> list[Fill]:
        if order_id:
            rows = self._conn.execute(
                "SELECT * FROM fills WHERE order_id = ? ORDER BY timestamp",
                (order_id,)
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM fills ORDER BY timestamp DESC"
            ).fetchall()

        fills = []
        for row in rows:
            fill = Fill.__new__(Fill)
            fill.id = row['id']
            fill.order_id = row['order_id']
            fill.broker_fill_id = row['broker_fill_id'] or ''
            fill.symbol = row['symbol']
            fill.side = OrderSide(row['side'])
            fill.quantity = row['quantity']
            fill.price = row['price']
            fill.commission = row['commission'] or 0.0
            fill.stamp_tax = row['stamp_tax'] or 0.0
            fill.timestamp = (
                datetime.fromisoformat(row['timestamp'])
                if row['timestamp'] else None
            )
            fills.append(fill)

        return fills

    def save_order_event(
        self,
        order_id: str,
        event_type: str,
        old_status: str | None = None,
        new_status: str | None = None,
        filled_qty: int | None = None,
        avg_price: float | None = None,
        message: str = "",
        payload: dict | None = None,
        conn: sqlite3.Connection = None,
    ) -> None:
        target = conn or self._conn
        auto_commit = conn is None
        try:
            target.execute(
                """
                INSERT INTO order_events
                (order_id, event_type, old_status, new_status, filled_qty,
                 avg_price, message, payload, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(order_id),
                    str(event_type or "event"),
                    str(old_status) if old_status else None,
                    str(new_status) if new_status else None,
                    int(filled_qty) if filled_qty is not None else None,
                    float(avg_price) if avg_price is not None else None,
                    str(message or ""),
                    json.dumps(payload or {}, ensure_ascii=False),
                    datetime.now().isoformat(),
                ),
            )
            if auto_commit:
                target.commit()
        except Exception:
            if auto_commit:
                target.rollback()
            raise

    def load_order_events(self, order_id: str, limit: int = 200) -> list[dict]:
        rows = self._conn.execute(
            """
            SELECT order_id, event_type, old_status, new_status, filled_qty,
                   avg_price, message, payload, created_at
            FROM order_events
            WHERE order_id = ?
            ORDER BY id ASC
            LIMIT ?
            """,
            (str(order_id), int(max(1, limit))),
        ).fetchall()
        out: list[dict] = []
        for r in rows:
            payload_raw = r["payload"] if "payload" in r.keys() else "{}"
            try:
                payload = json.loads(payload_raw) if payload_raw else {}
            except (json.JSONDecodeError, TypeError, ValueError):
                payload = {}
            out.append(
                {
                    "order_id": r["order_id"],
                    "event_type": r["event_type"],
                    "old_status": r["old_status"],
                    "new_status": r["new_status"],
                    "filled_qty": int(r["filled_qty"]) if r["filled_qty"] is not None else None,
                    "avg_price": float(r["avg_price"]) if r["avg_price"] is not None else None,
                    "message": r["message"] or "",
                    "payload": payload,
                    "created_at": r["created_at"],
                }
            )
        return out

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------

    def save_position(self, pos: Position, conn: sqlite3.Connection = None):
        target = conn or self._conn
        auto_commit = conn is None

        try:
            target.execute("""
                INSERT OR REPLACE INTO positions
                (symbol, name, quantity, available_qty, frozen_qty,
                 pending_buy, pending_sell, avg_cost, current_price,
                 realized_pnl, commission_paid, opened_at, last_updated)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                pos.symbol, pos.name, pos.quantity, pos.available_qty,
                pos.frozen_qty, pos.pending_buy, pos.pending_sell,
                pos.avg_cost, pos.current_price, pos.realized_pnl,
                pos.commission_paid,
                pos.opened_at.isoformat() if pos.opened_at else None,
                datetime.now().isoformat()
            ))

            if auto_commit:
                target.commit()
        except Exception:
            if auto_commit:
                target.rollback()
            raise

    def load_positions(self) -> dict[str, Position]:
        rows = self._conn.execute(
            "SELECT * FROM positions WHERE quantity > 0"
        ).fetchall()
        positions = {}

        for row in rows:
            pos = Position.__new__(Position)
            pos.symbol = row['symbol']
            pos.name = row['name'] or ''
            pos.quantity = row['quantity'] or 0
            pos.available_qty = row['available_qty'] or 0
            pos.frozen_qty = row['frozen_qty'] or 0
            pos.pending_buy = row['pending_buy'] or 0
            pos.pending_sell = row['pending_sell'] or 0
            pos.avg_cost = row['avg_cost'] or 0.0
            pos.current_price = row['current_price'] or 0.0
            pos.realized_pnl = row['realized_pnl'] or 0.0
            pos.commission_paid = row['commission_paid'] or 0.0
            pos.opened_at = (
                datetime.fromisoformat(row['opened_at'])
                if row['opened_at'] else None
            )
            pos.last_updated = (
                datetime.fromisoformat(row['last_updated'])
                if row['last_updated'] else datetime.now()
            )
            positions[pos.symbol] = pos

        return positions

    def delete_position(self, symbol: str, conn: sqlite3.Connection = None):
        target = conn or self._conn
        auto_commit = conn is None

        try:
            target.execute(
                "DELETE FROM positions WHERE symbol = ?", (symbol,)
            )
            if auto_commit:
                target.commit()
        except Exception:
            if auto_commit:
                target.rollback()
            raise

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------

    def save_account_state(
        self, account: Account, conn: sqlite3.Connection = None
    ):
        target = conn or self._conn
        auto_commit = conn is None

        try:
            target.execute("""
                INSERT OR REPLACE INTO account_state
                (id, cash, available, frozen, initial_capital,
                 realized_pnl, commission_paid, peak_equity,
                 daily_start_equity, daily_start_date, updated_at)
                VALUES (1, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                account.cash, account.available, account.frozen,
                account.initial_capital, account.realized_pnl,
                account.commission_paid, account.peak_equity,
                account.daily_start_equity,
                (account.daily_start_date.isoformat()
                 if account.daily_start_date else None),
                datetime.now().isoformat()
            ))

            if auto_commit:
                target.commit()
        except Exception:
            if auto_commit:
                target.rollback()
            raise

    def load_account_state(self) -> Account | None:
        row = self._conn.execute(
            "SELECT * FROM account_state WHERE id = 1"
        ).fetchone()
        if not row:
            return None

        account = Account.__new__(Account)
        account.broker_name = ''
        account.account_id = ''
        account.cash = row['cash'] or 0.0
        account.available = row['available'] or 0.0
        account.frozen = row['frozen'] or 0.0
        account.initial_capital = row['initial_capital'] or 0.0
        account.realized_pnl = row['realized_pnl'] or 0.0
        account.commission_paid = row['commission_paid'] or 0.0
        account.peak_equity = row['peak_equity'] or 0.0
        account.daily_start_equity = row['daily_start_equity'] or 0.0
        account.last_updated = datetime.now()

        if row['daily_start_date']:
            account.daily_start_date = date.fromisoformat(
                row['daily_start_date']
            )
        else:
            account.daily_start_date = date.today()

        account.positions = self.load_positions()
        return account

    # ------------------------------------------------------------------
    # T+1 pending
    # ------------------------------------------------------------------

    def save_t1_pending(
        self, symbol: str, quantity: int, purchase_date: date,
        conn: sqlite3.Connection = None
    ):
        """Save T+1 pending — accumulates if same symbol+date."""
        target = conn or self._conn
        auto_commit = conn is None

        try:
            target.execute("""
                INSERT INTO t1_pending (symbol, purchase_date, quantity)
                VALUES (?, ?, ?)
                ON CONFLICT(symbol, purchase_date) DO UPDATE SET
                    quantity = quantity + excluded.quantity
            """, (symbol, purchase_date.isoformat(), quantity))

            if auto_commit:
                target.commit()
        except Exception:
            if auto_commit:
                target.rollback()
            raise

    def get_t1_pending(self) -> dict[str, list[tuple[int, date]]]:
        rows = self._conn.execute(
            "SELECT symbol, purchase_date, quantity FROM t1_pending"
        ).fetchall()

        result: dict[str, list[tuple[int, date]]] = {}
        for row in rows:
            symbol = row['symbol']
            if symbol not in result:
                result[symbol] = []
            result[symbol].append((
                row['quantity'],
                date.fromisoformat(row['purchase_date'])
            ))
        return result

    def clear_t1_pending(
        self, symbol: str, purchase_date: date = None,
        conn: sqlite3.Connection = None
    ):
        target = conn or self._conn
        auto_commit = conn is None

        try:
            if purchase_date:
                target.execute(
                    "DELETE FROM t1_pending "
                    "WHERE symbol = ? AND purchase_date = ?",
                    (symbol, purchase_date.isoformat())
                )
            else:
                target.execute(
                    "DELETE FROM t1_pending WHERE symbol = ?",
                    (symbol,)
                )

            if auto_commit:
                target.commit()
        except Exception:
            if auto_commit:
                target.rollback()
            raise

    def process_t1_settlement(self) -> list[str]:
        """Process T+1 settlement — only on trading days."""
        from core.constants import is_trading_day

        today = date.today()
        if not is_trading_day(today):
            return []

        settled = []

        with self.transaction() as conn:
            rows = conn.execute(
                "SELECT symbol, purchase_date, quantity FROM t1_pending"
            ).fetchall()

            for row in rows:
                symbol = row['symbol']
                purchase_date = date.fromisoformat(row['purchase_date'])
                quantity = row['quantity']

                trading_days_passed = 0
                check_date = purchase_date + timedelta(days=1)
                while check_date <= today:
                    if is_trading_day(check_date):
                        trading_days_passed += 1
                        break
                    check_date += timedelta(days=1)

                if trading_days_passed >= 1:
                    conn.execute("""
                        UPDATE positions
                        SET available_qty = available_qty + ?
                        WHERE symbol = ?
                    """, (quantity, symbol))

                    conn.execute(
                        "DELETE FROM t1_pending "
                        "WHERE symbol = ? AND purchase_date = ?",
                        (symbol, row['purchase_date'])
                    )
                    settled.append(symbol)

        return settled

