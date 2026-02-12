# data/database.py
import sqlite3
import atexit
from pathlib import Path
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Tuple
from contextlib import contextmanager
import threading
import pandas as pd
import numpy as np

from config.settings import CONFIG
from utils.logger import get_logger
from utils.helpers import to_float, to_int

log = get_logger(__name__)

# Current schema version — bump when adding/altering tables
_SCHEMA_VERSION = 2

class MarketDatabase:
    """
    Local market data database with proper lifecycle management.

    Tables:
    - _meta: Schema version tracking
    - stocks: Stock metadata
    - daily_bars: Daily OHLCV data
    - intraday_bars: Intraday OHLCV data
    - features: Computed features
    - predictions: Model predictions

    FIX: Replaced weakref.WeakSet with regular set for connection tracking.
    sqlite3.Connection does not support weak references.
    """

    def __init__(self, db_path: Path = None):
        self._db_path = db_path or CONFIG.data_dir / "market_data.db"
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        # FIX: Use regular set instead of WeakSet
        # sqlite3.Connection doesn't support weak references
        self._connections: set = set()
        self._connections_lock = threading.Lock()
        self._schema_lock = threading.Lock()
        self._schema_ready = threading.Event()
        self._init_db()
        atexit.register(self.close_all)

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------

    @property
    def _conn(self) -> sqlite3.Connection:
        """Thread-local connection with automatic registration."""
        if not hasattr(self._local, "conn") or self._local.conn is None:
            conn = sqlite3.connect(
                str(self._db_path),
                check_same_thread=False,
                timeout=30,
            )
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA foreign_keys=ON")
            self._local.conn = conn
            with self._connections_lock:
                self._connections.add(conn)
        # Wait for schema to be ready (blocks only on first access
        # if another thread is still running _init_db)
        self._schema_ready.wait(timeout=30)
        return self._local.conn

    @contextmanager
    def _transaction(self):
        """Transaction context manager with proper rollback."""
        conn = self._conn
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------

    def _init_db(self):
        """Initialize or migrate database schema."""
        with self._schema_lock:
            try:
                self._create_or_migrate()
            finally:
                self._schema_ready.set()

    def _get_schema_version(self, conn: sqlite3.Connection) -> int:
        """Read current schema version (0 if fresh db)."""
        try:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS _meta "
                "(key TEXT PRIMARY KEY, value TEXT)"
            )
            conn.commit()
            cur = conn.execute(
                "SELECT value FROM _meta WHERE key = 'schema_version'"
            )
            row = cur.fetchone()
            return int(row[0]) if row else 0
        except Exception:
            return 0

    def _set_schema_version(self, conn: sqlite3.Connection, version: int):
        conn.execute(
            "INSERT INTO _meta (key, value) VALUES ('schema_version', ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (str(version),),
        )
        conn.commit()

    def _create_or_migrate(self):
        """Create tables or migrate from older schema versions."""
        # Use a dedicated connection for DDL so thread-local isn't
        # exposed before the schema is ready.
        conn = sqlite3.connect(str(self._db_path), timeout=30)
        conn.execute("PRAGMA journal_mode=WAL")
        try:
            current = self._get_schema_version(conn)

            if current < 1:
                self._apply_v1(conn)
            if current < 2:
                self._apply_v2(conn)

            self._set_schema_version(conn, _SCHEMA_VERSION)
        finally:
            conn.close()

    @staticmethod
    def _apply_v1(conn: sqlite3.Connection):
        """V1: core tables."""
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS stocks (
                code TEXT PRIMARY KEY,
                name TEXT,
                exchange TEXT,
                sector TEXT,
                market_cap REAL,
                list_date TEXT,
                is_st INTEGER DEFAULT 0,
                updated_at TEXT
            );

            CREATE TABLE IF NOT EXISTS daily_bars (
                code TEXT NOT NULL,
                date TEXT NOT NULL,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume INTEGER,
                amount REAL,
                turnover REAL,
                PRIMARY KEY (code, date)
            );
            CREATE INDEX IF NOT EXISTS idx_bars_date
                ON daily_bars(date);

            CREATE TABLE IF NOT EXISTS features (
                code TEXT NOT NULL,
                date TEXT NOT NULL,
                features BLOB,
                PRIMARY KEY (code, date)
            );

            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                code TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                signal TEXT,
                prob_up REAL,
                prob_down REAL,
                confidence REAL,
                price REAL,
                UNIQUE(code, timestamp)
            );
        """)
        conn.commit()

    @staticmethod
    def _apply_v2(conn: sqlite3.Connection):
        """V2: intraday bars table."""
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS intraday_bars (
                code TEXT NOT NULL,
                ts TEXT NOT NULL,
                interval TEXT NOT NULL,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume INTEGER,
                amount REAL,
                PRIMARY KEY (code, ts, interval)
            );
            CREATE INDEX IF NOT EXISTS idx_intraday_code_interval_ts
                ON intraday_bars(code, interval, ts);
        """)
        conn.commit()

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------

    _REQUIRED_BAR_COLUMNS = {"open", "high", "low", "close"}
    _OPTIONAL_BAR_COLUMNS = {"volume", "amount", "turnover"}

    @classmethod
    def _validate_bar_columns(cls, df: pd.DataFrame, context: str = "bars") -> None:
        """Raise ValueError if required columns are missing."""
        if df is None or df.empty:
            return
        present = set(c.lower() for c in df.columns)
        missing = cls._REQUIRED_BAR_COLUMNS - present
        if missing:
            raise ValueError(
                f"{context}: missing required columns {missing}. "
                f"Present: {sorted(present)}"
            )

    # ------------------------------------------------------------------
    # NaN-safe type converters
    # ------------------------------------------------------------------

    @staticmethod
    def _to_float(x, default: float = 0.0) -> float:
        return to_float(x, default)

    @staticmethod
    def _to_int(x, default: int = 0) -> int:
        return to_int(x, default)

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------

    def get_intraday_bars(
        self,
        code: str,
        interval: str = "1m",
        limit: int = 1000,
    ) -> pd.DataFrame:
        """Get last *limit* intraday bars for (code, interval)."""
        code = str(code).zfill(6)
        interval = str(interval).lower()
        limit = max(1, int(limit or 1000))

        query = """
            SELECT ts, open, high, low, close, volume, amount
            FROM intraday_bars
            WHERE code = ? AND interval = ?
            ORDER BY ts DESC
            LIMIT ?
        """
        try:
            df = pd.read_sql_query(
                query, self._conn, params=(code, interval, limit)
            )
        except Exception as e:
            log.warning(f"Intraday query failed for {code}/{interval}: {e}")
            return pd.DataFrame()

        if df.empty:
            return pd.DataFrame()

        df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
        df = df.dropna(subset=["ts"]).set_index("ts").sort_index()
        return df

    def upsert_intraday_bars(
        self, code: str, interval: str, df: pd.DataFrame
    ):
        """Insert/update intraday bars."""
        if df is None or df.empty:
            return

        self._validate_bar_columns(df, context="intraday_bars")

        code = str(code).zfill(6)
        interval = str(interval).lower()

        work = df.copy()
        if not isinstance(work.index, pd.DatetimeIndex):
            work.index = pd.to_datetime(work.index, errors="coerce")
        work = work[~work.index.isna()]
        work = work.sort_index()
        work = work[~work.index.duplicated(keep="last")]

        if work.empty:
            return

        rows = self._build_intraday_rows(code, interval, work)

        with self._transaction() as conn:
            conn.executemany(
                """
                INSERT INTO intraday_bars
                    (code, ts, interval, open, high, low, close, volume, amount)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(code, ts, interval) DO UPDATE SET
                    open=excluded.open, high=excluded.high,
                    low=excluded.low, close=excluded.close,
                    volume=excluded.volume, amount=excluded.amount
                """,
                rows,
            )

    def _build_intraday_rows(
        self, code: str, interval: str, work: pd.DataFrame
    ) -> List[tuple]:
        """Build rows for intraday bar insert."""
        tf = self._to_float
        ti = self._to_int

        timestamps = work.index.to_series().apply(
            lambda t: t.isoformat()
        ).values

        rows = []
        for i, (ts_iso, (_, row)) in enumerate(
            zip(timestamps, work.iterrows())
        ):
            rows.append((
                code,
                ts_iso,
                interval,
                tf(row.get("open", 0)),
                tf(row.get("high", 0)),
                tf(row.get("low", 0)),
                tf(row.get("close", 0)),
                ti(row.get("volume", 0)),
                tf(row.get("amount", 0)),
            ))
        return rows

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------

    def upsert_stock(
        self,
        code: str,
        name: str = None,
        exchange: str = None,
        sector: str = None,
        market_cap: float = None,
        is_st: bool = False,
    ):
        """Insert or update stock metadata."""
        with self._transaction() as conn:
            conn.execute(
                """
                INSERT INTO stocks
                    (code, name, exchange, sector, market_cap, is_st, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(code) DO UPDATE SET
                    name = COALESCE(excluded.name, stocks.name),
                    exchange = COALESCE(excluded.exchange, stocks.exchange),
                    sector = COALESCE(excluded.sector, stocks.sector),
                    market_cap = COALESCE(excluded.market_cap, stocks.market_cap),
                    is_st = excluded.is_st,
                    updated_at = excluded.updated_at
                """,
                (
                    code,
                    name,
                    exchange,
                    sector,
                    market_cap,
                    int(is_st),
                    datetime.now().isoformat(),
                ),
            )

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------

    def upsert_bars(self, code: str, df: pd.DataFrame):
        """Insert/update daily bars with column validation."""
        if df is None or df.empty:
            return

        self._validate_bar_columns(df, context="daily_bars")

        work = df.copy()
        if not isinstance(work.index, pd.DatetimeIndex):
            work.index = pd.to_datetime(work.index, errors="coerce")
        work = work[~work.index.isna()]
        work = work.sort_index()
        work = work[~work.index.duplicated(keep="last")]

        if work.empty:
            return

        code = str(code).zfill(6)
        rows = self._build_daily_rows(code, work)

        with self._transaction() as conn:
            conn.executemany(
                """
                INSERT INTO daily_bars
                    (code, date, open, high, low, close, volume, amount, turnover)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(code, date) DO UPDATE SET
                    open=excluded.open, high=excluded.high,
                    low=excluded.low, close=excluded.close,
                    volume=excluded.volume, amount=excluded.amount,
                    turnover=excluded.turnover
                """,
                rows,
            )

    def _build_daily_rows(
        self, code: str, work: pd.DataFrame
    ) -> List[tuple]:
        """Build rows for daily bar insert."""
        tf = self._to_float
        ti = self._to_int

        rows = []
        for idx, row in work.iterrows():
            rows.append((
                code,
                idx.strftime("%Y-%m-%d"),
                tf(row.get("open", 0)),
                tf(row.get("high", 0)),
                tf(row.get("low", 0)),
                tf(row.get("close", 0)),
                ti(row.get("volume", 0)),
                tf(row.get("amount", 0)),
                tf(row.get("turnover", 0)),
            ))
        return rows

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------

    def get_bars(
        self,
        code: str,
        start_date: date = None,
        end_date: date = None,
        limit: int = None,
    ) -> pd.DataFrame:
        """
        Get daily bars for a stock.

        Returns:
            DataFrame with OHLCV data indexed by date (empty on error)
        """
        code = str(code).zfill(6)

        conditions = ["code = ?"]
        params: list = [code]

        if start_date is not None:
            conditions.append("date >= ?")
            params.append(
                start_date.isoformat()
                if isinstance(start_date, date)
                else str(start_date)
            )

        if end_date is not None:
            conditions.append("date <= ?")
            params.append(
                end_date.isoformat()
                if isinstance(end_date, date)
                else str(end_date)
            )

        where_clause = " AND ".join(conditions)

        if limit is not None:
            inner_query = (
                f"SELECT * FROM daily_bars "
                f"WHERE {where_clause} "
                f"ORDER BY date DESC LIMIT ?"
            )
            params.append(int(limit))
            query = f"SELECT * FROM ({inner_query}) ORDER BY date ASC"
        else:
            query = (
                f"SELECT * FROM daily_bars "
                f"WHERE {where_clause} ORDER BY date ASC"
            )

        try:
            df = pd.read_sql_query(query, self._conn, params=params)
        except Exception as e:
            log.warning(f"Database query failed for {code}: {e}")
            return pd.DataFrame()

        if df.empty:
            return pd.DataFrame()

        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()

        if "code" in df.columns:
            df = df.drop(columns=["code"])

        return df

    def get_last_date(self, code: str) -> Optional[date]:
        """Get last available date for a stock."""
        code = str(code).zfill(6)
        try:
            cursor = self._conn.execute(
                "SELECT MAX(date) FROM daily_bars WHERE code = ?", (code,)
            )
            row = cursor.fetchone()
            if row and row[0]:
                return datetime.fromisoformat(row[0]).date()
        except Exception as e:
            log.debug(f"get_last_date failed for {code}: {e}")
        return None

    def get_all_stocks(self) -> List[Dict]:
        """Get all stock metadata."""
        try:
            cursor = self._conn.execute("SELECT * FROM stocks")
            return [dict(row) for row in cursor.fetchall()]
        except Exception as e:
            log.warning(f"get_all_stocks failed: {e}")
            return []

    def get_stocks_with_data(self, min_days: int = 100) -> List[str]:
        """Get stock codes with at least *min_days* of bar data."""
        try:
            cursor = self._conn.execute(
                """
                SELECT code, COUNT(*) as cnt
                FROM daily_bars
                GROUP BY code
                HAVING cnt >= ?
                """,
                (min_days,),
            )
            return [row["code"] for row in cursor.fetchall()]
        except Exception as e:
            log.warning(f"get_stocks_with_data failed: {e}")
            return []

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------

    def save_prediction(
        self,
        code: str,
        signal: str,
        prob_up: float,
        prob_down: float,
        confidence: float,
        price: float,
    ):
        """Save model prediction."""
        with self._transaction() as conn:
            conn.execute(
                """
                INSERT INTO predictions
                    (code, timestamp, signal, prob_up, prob_down,
                     confidence, price)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(code, timestamp) DO UPDATE SET
                    signal=excluded.signal,
                    prob_up=excluded.prob_up,
                    prob_down=excluded.prob_down,
                    confidence=excluded.confidence,
                    price=excluded.price
                """,
                (
                    code,
                    datetime.now().isoformat(),
                    signal,
                    prob_up,
                    prob_down,
                    confidence,
                    price,
                ),
            )

    # ------------------------------------------------------------------
    # Stats / maintenance
    # ------------------------------------------------------------------

    def get_data_stats(self) -> Dict:
        """Get database statistics."""
        stats: Dict = {}
        try:
            cursor = self._conn.execute(
                "SELECT COUNT(DISTINCT code) FROM stocks"
            )
            stats["total_stocks"] = cursor.fetchone()[0] or 0

            cursor = self._conn.execute("SELECT COUNT(*) FROM daily_bars")
            stats["total_bars"] = cursor.fetchone()[0] or 0

            cursor = self._conn.execute(
                "SELECT MIN(date), MAX(date) FROM daily_bars"
            )
            row = cursor.fetchone()
            stats["date_range"] = (row[0], row[1]) if row else (None, None)
        except Exception as e:
            log.warning(f"get_data_stats failed: {e}")
            stats.setdefault("total_stocks", 0)
            stats.setdefault("total_bars", 0)
            stats.setdefault("date_range", (None, None))

        return stats

    def vacuum(self):
        """Optimize database."""
        conn = self._conn
        conn.commit()
        old_isolation = conn.isolation_level
        try:
            conn.isolation_level = None
            conn.execute("VACUUM")
        finally:
            conn.isolation_level = old_isolation

    def close(self):
        """Close this thread's connection."""
        if hasattr(self._local, "conn") and self._local.conn is None:
            return
        if hasattr(self._local, "conn"):
            conn = self._local.conn
            try:
                conn.close()
            except Exception:
                pass
            with self._connections_lock:
                self._connections.discard(conn)
            self._local.conn = None

    def close_all(self):
        """Close all tracked connections (call on shutdown)."""
        with self._connections_lock:
            for conn in list(self._connections):
                try:
                    conn.close()
                except Exception:
                    pass
            self._connections.clear()
        # Also close this thread's connection
        if hasattr(self._local, "conn") and self._local.conn is not None:
            try:
                self._local.conn.close()
            except Exception:
                pass
            self._local.conn = None

# Global database instance (thread-safe)

_db: Optional[MarketDatabase] = None
_db_lock = threading.Lock()

def get_database() -> MarketDatabase:
    """Get global database instance (thread-safe)."""
    global _db
    if _db is None:
        with _db_lock:
            if _db is None:
                _db = MarketDatabase()
    return _db
