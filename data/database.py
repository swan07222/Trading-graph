# data/database.py
import atexit
import sqlite3
import threading
from contextlib import contextmanager
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from config.settings import CONFIG
from utils.helpers import to_float, to_int
from utils.logger import get_logger

log = get_logger(__name__)
_DB_QUERY_EXCEPTIONS = (sqlite3.Error, pd.errors.DatabaseError, ValueError, TypeError)
_DB_SOFT_EXCEPTIONS = (sqlite3.Error, OSError, RuntimeError, TypeError, ValueError)
# Current schema version — bump when adding/altering tables
_SCHEMA_VERSION = 2

class MarketDatabase:
    """Local market data database with proper lifecycle management.

    Tables:
    - _meta: Schema version tracking
    - stocks: Stock metadata
    - daily_bars: Daily OHLCV data
    - intraday_bars: Intraday OHLCV data
    - features: Computed features
    - predictions: Model predictions

    FIX #3: Tracks connections by thread ID so dead-thread connections
    can be cleaned up, preventing connection leaks.
    FIX #4: Schema readiness wait checks return value and raises on timeout.
    FIX #9: Migration connection is registered and always cleaned up.
    """

    def __init__(self, db_path: Path = None) -> None:
        self._db_path = db_path or CONFIG.data_dir / "market_data.db"
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        # FIX #3: Track by thread ID for dead-thread cleanup
        self._connections: dict[int, sqlite3.Connection] = {}
        self._connections_lock = threading.Lock()
        self._schema_lock = threading.Lock()
        self._schema_ready = threading.Event()
        self._init_db()
        atexit.register(self.close_all)

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    @property
    def _conn(self) -> sqlite3.Connection:
        """Thread-local connection with automatic registration."""
        if (
            not hasattr(self._local, "conn")
            or self._local.conn is None
        ):
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
            tid = threading.get_ident()
            with self._connections_lock:
                # FIX #3: Clean up dead threads while registering
                self._cleanup_dead_threads()
                self._connections[tid] = conn
        else:
            # FIX #3 & #8: Periodic cleanup every 50 accesses with overflow protection
            # to prevent connection accumulation and counter overflow.
            if not hasattr(self._local, "access_count"):
                self._local.access_count = 0
            self._local.access_count += 1
            # Reset counter periodically to prevent integer overflow
            if self._local.access_count >= 10000:
                self._local.access_count = 0
            elif self._local.access_count % 50 == 0:
                with self._connections_lock:
                    self._cleanup_dead_threads()

        # FIX #4: Check wait return value — raise on timeout
        if not self._schema_ready.wait(timeout=30):
            raise RuntimeError(
                "Database schema initialization timed out after 30s. "
                "This may indicate a deadlock or very slow disk."
            )
        return self._local.conn

    def _cleanup_dead_threads(self) -> None:
        """Remove and close connections for threads that no longer exist.

        FIX #3: Called under _connections_lock by the caller.
        FIX #10: Register atexit handler to ensure all connections are closed on shutdown.
        """
        alive_ids = {
            t.ident for t in threading.enumerate() if t.ident is not None
        }
        dead_ids = [
            tid
            for tid in self._connections
            if tid not in alive_ids
        ]
        for tid in dead_ids:  # noqa: B007 (tid used as dict key)
            conn = self._connections.pop(tid, None)
            if conn is not None:
                try:
                    conn.close()
                except (sqlite3.Error, OSError):
                    pass

    @contextmanager
    def _transaction(self):
        """Transaction context manager with proper rollback.
        
        FIX #9: Catch ALL exceptions to ensure rollback on any failure,
        not just soft exceptions. This prevents partial writes and data
        corruption.
        """
        conn = self._conn
        try:
            yield conn
            conn.commit()
        except Exception:
            # Catch ALL exceptions to ensure atomic transactions
            try:
                conn.rollback()
            except Exception as rb_error:
                # Log rollback failure but re-raise original exception
                log.error(f"Transaction rollback failed: {rb_error}")
            raise

    # ------------------------------------------------------------------
    # Schema initialization
    # ------------------------------------------------------------------

    def _init_db(self) -> None:
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
        except _DB_SOFT_EXCEPTIONS:
            return 0

    def _set_schema_version(
        self, conn: sqlite3.Connection, version: int
    ) -> None:
        conn.execute(
            "INSERT INTO _meta (key, value) VALUES ('schema_version', ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (str(version),),
        )
        conn.commit()

    def _create_or_migrate(self) -> None:
        """Create tables or migrate from older schema versions.

        FIX #9: Migration connection is tracked and always closed.
        """
        conn = None
        try:
            conn = sqlite3.connect(str(self._db_path), timeout=30)
            conn.execute("PRAGMA journal_mode=WAL")

            current = self._get_schema_version(conn)

            if current < 1:
                self._apply_v1(conn)
            if current < 2:
                self._apply_v2(conn)

            self._set_schema_version(conn, _SCHEMA_VERSION)
        finally:
            if conn is not None:
                try:
                    conn.close()
                except (sqlite3.Error, OSError):
                    pass

    @staticmethod
    def _apply_v1(conn: sqlite3.Connection) -> None:
        """V1: core tables."""
        conn.executescript(
            """
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
        """
        )
        conn.commit()

    @staticmethod
    def _apply_v2(conn: sqlite3.Connection) -> None:
        """V2: intraday bars table."""
        conn.executescript(
            """
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
        """
        )
        conn.commit()

    # ------------------------------------------------------------------
    # Column validation
    # ------------------------------------------------------------------

    _REQUIRED_BAR_COLUMNS = {"open", "high", "low", "close"}
    _OPTIONAL_BAR_COLUMNS = {"volume", "amount", "turnover"}

    @classmethod
    def _validate_bar_columns(
        cls, df: pd.DataFrame, context: str = "bars"
    ) -> None:
        """Raise ValueError if required columns are missing."""
        if df is None or df.empty:
            return
        present = {c.lower() for c in df.columns}
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

    @staticmethod
    def _intraday_quality_caps(
        interval: str,
    ) -> tuple[float, float, float, float]:
        """Return (body_cap, span_cap, wick_cap, jump_cap) for intraday writes."""
        iv = str(interval or "1m").strip().lower()
        if iv == "1m":
            return 0.0100, 0.0160, 0.0120, 0.10
        if iv == "2m":
            return 0.0120, 0.0180, 0.0130, 0.10
        if iv == "5m":
            return 0.0160, 0.0240, 0.0180, 0.12
        if iv == "15m":
            return 0.0240, 0.0360, 0.0240, 0.14
        if iv == "30m":
            return 0.0320, 0.0500, 0.0300, 0.16
        if iv in ("60m", "1h"):
            return 0.0450, 0.0700, 0.0400, 0.20
        return 0.0600, 0.1000, 0.0600, 0.25

    @classmethod
    def _validate_ohlcv_consistency(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and repair basic OHLCV consistency.

        Ensures high >= max(open, close), low <= min(open, close).
        """
        out = df.copy()
        oc_top = out[["open", "close"]].max(axis=1)
        oc_bot = out[["open", "close"]].min(axis=1)
        out["high"] = pd.concat([out["high"], oc_top], axis=1).max(axis=1)
        out["low"] = pd.concat([out["low"], oc_bot], axis=1).min(axis=1)
        return out

    @classmethod
    def _repair_price_anomalies(
        cls,
        df: pd.DataFrame,
        body_cap: float,
        span_cap: float,
        wick_cap: float,
        aggressive: bool = False,
        preserve_truth: bool = True,
    ) -> pd.DataFrame:
        """Repair anomalous price bars that exceed caps.

        FIX: Extracted from _sanitize_intraday_frame for clarity.
        """
        out = df.copy()
        close_safe = out["close"].clip(lower=1e-8)

        # Repair bad open prices
        body = (out["open"] - out["close"]).abs() / close_safe
        bad_open = body > float(body_cap)
        if bool(bad_open.any()):
            out.loc[bad_open, "open"] = out.loc[bad_open, "close"]
            out = cls._validate_ohlcv_consistency(out)

        # Repair bad shape (span/wick)
        span = (out["high"] - out["low"]).abs() / close_safe
        oc_top = out[["open", "close"]].max(axis=1)
        oc_bot = out[["open", "close"]].min(axis=1)
        upper_wick = (out["high"] - oc_top).clip(lower=0.0) / close_safe
        lower_wick = (oc_bot - out["low"]).clip(lower=0.0) / close_safe
        bad_shape = (
            (span > float(span_cap))
            | (upper_wick > float(wick_cap))
            | (lower_wick > float(wick_cap))
        )
        if bool(bad_shape.any()):
            if aggressive and not preserve_truth:
                wick_allow = close_safe * float(wick_cap)
                out.loc[bad_shape, "high"] = (oc_top + wick_allow)[bad_shape]
                out.loc[bad_shape, "low"] = (oc_bot - wick_allow)[bad_shape]
            else:
                out.loc[bad_shape, "high"] = oc_top[bad_shape]
                out.loc[bad_shape, "low"] = oc_bot[bad_shape]
            out = cls._validate_ohlcv_consistency(out)

        return out

    @classmethod
    def _handle_stale_bars(
        cls,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Drop stale/flat bars with no price movement or volume.

        FIX: Extracted from _sanitize_intraday_frame for clarity.
        """
        out = df.copy()
        close_safe = out["close"].clip(lower=1e-8)
        same_close = out["close"].diff().abs() <= (close_safe * 1e-6)
        flat_body = (out["open"] - out["close"]).abs() <= (close_safe * 1e-6)
        flat_span = (out["high"] - out["low"]).abs() <= (close_safe * 2e-6)
        vol = out["volume"].fillna(0)
        if "volume" not in out.columns:
            vol = pd.Series(0.0, index=out.index)
        else:
            vol = out["volume"].fillna(0)
        stale_flat = same_close & flat_body & flat_span & (vol <= 0)

        if bool(stale_flat.any()):
            group_key = stale_flat.ne(stale_flat.shift(fill_value=False)).cumsum()
            stale_pos = stale_flat.groupby(group_key).cumcount()
            drop_mask = stale_flat & (stale_pos % 12 != 0)
            if bool(drop_mask.any()):
                out = out.loc[~drop_mask]

        return out

    @classmethod
    def _handle_jump_anomalies(
        cls,
        df: pd.DataFrame,
        jump_cap: float,
        body_cap: float,
        aggressive: bool = False,
        preserve_truth: bool = True,
        wick_cap: float = 0.012,  # Default fallback
    ) -> pd.DataFrame:
        """Handle impossible inter-bar price jumps.

        FIX: Extracted from _sanitize_intraday_frame for clarity.
        """
        out = df.copy()
        prev_close = out["close"].shift(1)
        prev_safe = prev_close.where(prev_close > 0, np.nan)
        jump = (out["close"] / prev_safe - 1.0).abs()
        jump_cap_eff = float(max(jump_cap, body_cap * 4.0))
        bad_jump = jump > jump_cap_eff

        # Exclude day boundaries (jumps expected)
        if isinstance(out.index, pd.DatetimeIndex):
            day_change = (
                pd.Series(out.index.normalize(), index=out.index)
                .diff()
                .ne(pd.Timedelta(0))
            )
            day_change = day_change.fillna(False)
            bad_jump = bad_jump & (~day_change)
        bad_jump = bad_jump.fillna(False)

        if bool(bad_jump.any()):
            if aggressive and not preserve_truth:
                # Clip to max allowed jump
                prev_vals = prev_close[bad_jump].astype(float)
                curr_vals = out.loc[bad_jump, "close"].astype(float)
                signs = np.where(curr_vals >= prev_vals, 1.0, -1.0)
                clipped = prev_vals * (1.0 + (signs * jump_cap_eff))
                out.loc[bad_jump, "close"] = clipped.values
                out.loc[bad_jump, "open"] = prev_vals.values
                close_safe = out["close"].clip(lower=1e-8)
                oc_top = out[["open", "close"]].max(axis=1)
                oc_bot = out[["open", "close"]].min(axis=1)
                wick_allow = close_safe * float(wick_cap)
                out.loc[bad_jump, "high"] = np.minimum(
                    out.loc[bad_jump, "high"],
                    (oc_top + wick_allow)[bad_jump],
                )
                out.loc[bad_jump, "low"] = np.maximum(
                    out.loc[bad_jump, "low"],
                    (oc_bot - wick_allow)[bad_jump],
                )
                out = cls._validate_ohlcv_consistency(out)
            else:
                # Drop bars with impossible jumps (truth-preserving)
                out = out.loc[~bad_jump]
                if out.empty:
                    return pd.DataFrame()

        return out

    @classmethod
    def _sanitize_intraday_frame(
        cls,
        df: pd.DataFrame,
        interval: str,
    ) -> pd.DataFrame:
        """Sanitize intraday OHLCV rows before read/write usage."""
        if df is None or df.empty:
            return pd.DataFrame()

        iv = str(interval or "1m").strip().lower()
        out = df.copy()

        if not isinstance(out.index, pd.DatetimeIndex):
            out.index = pd.to_datetime(out.index, errors="coerce")
        out = out[~out.index.isna()]
        out = out[~out.index.duplicated(keep="last")].sort_index()
        if out.empty:
            return pd.DataFrame()

        for col in (
            "open",
            "high",
            "low",
            "close",
            "volume",
            "amount",
        ):
            if col in out.columns:
                out[col] = pd.to_numeric(out[col], errors="coerce")

        out = out.replace([np.inf, -np.inf], np.nan)
        out = out.dropna(subset=["close"])
        out = out[out["close"] > 0]
        if out.empty:
            return pd.DataFrame()

        if "open" not in out.columns:
            out["open"] = out["close"]
        out["open"] = pd.to_numeric(
            out["open"], errors="coerce"
        ).fillna(0.0)
        out["open"] = out["open"].where(
            out["open"] > 0, out["close"]
        )

        if "high" not in out.columns:
            out["high"] = out[["open", "close"]].max(axis=1)
        else:
            out["high"] = pd.to_numeric(
                out["high"], errors="coerce"
            )

        if "low" not in out.columns:
            out["low"] = out[["open", "close"]].min(axis=1)
        else:
            out["low"] = pd.to_numeric(
                out["low"], errors="coerce"
            )

        oc_top = out[["open", "close"]].max(axis=1)
        oc_bot = out[["open", "close"]].min(axis=1)
        out["high"] = pd.concat(
            [out["high"], oc_top], axis=1
        ).max(axis=1)
        out["low"] = pd.concat(
            [out["low"], oc_bot], axis=1
        ).min(axis=1)

        is_intraday = iv not in ("1d", "1wk", "1mo")
        if is_intraday:
            preserve_truth = bool(
                getattr(getattr(CONFIG, "data", None), "truth_preserving_cleaning", True)
            )
            aggressive_repairs = bool(
                getattr(
                    getattr(CONFIG, "data", None),
                    "aggressive_intraday_repair",
                    False,
                )
            )
            body_cap, span_cap, wick_cap, jump_cap = (
                cls._intraday_quality_caps(iv)
            )

            # FIX: Use extracted helper methods for clarity
            out = cls._repair_price_anomalies(
                out,
                body_cap=body_cap,
                span_cap=span_cap,
                wick_cap=wick_cap,
                aggressive=aggressive_repairs,
                preserve_truth=preserve_truth,
            )

            out = cls._handle_jump_anomalies(
                out,
                jump_cap=jump_cap,
                body_cap=body_cap,
                wick_cap=wick_cap,
                aggressive=aggressive_repairs,
                preserve_truth=preserve_truth,
            )

            out = cls._handle_stale_bars(out)

        if "volume" in out.columns:
            out = out[out["volume"].fillna(0) >= 0]
        else:
            out["volume"] = 0

        out = out[out["high"].fillna(0) >= out["low"].fillna(0)]
        if "amount" not in out.columns:
            out["amount"] = (
                out["close"] * out["volume"].fillna(0)
            )
        out = out.fillna(0)
        out = out[~out.index.duplicated(keep="last")].sort_index()
        return out

    # ------------------------------------------------------------------
    # Daily bar sanitization (FIX #23: symmetry with intraday)
    # ------------------------------------------------------------------

    @classmethod
    def _sanitize_daily_frame(
        cls, df: pd.DataFrame
    ) -> pd.DataFrame:
        """Sanitize daily OHLCV rows before write.

        FIX #23: Applies basic quality checks analogous to intraday
        sanitization — ensures high >= low, positive close, non-negative
        volume, and fills missing columns.
        """
        if df is None or df.empty:
            return pd.DataFrame()

        out = df.copy()

        if not isinstance(out.index, pd.DatetimeIndex):
            out.index = pd.to_datetime(out.index, errors="coerce")
        out = out[~out.index.isna()]
        out = out[~out.index.duplicated(keep="last")].sort_index()
        if out.empty:
            return pd.DataFrame()

        for col in (
            "open",
            "high",
            "low",
            "close",
            "volume",
            "amount",
            "turnover",
        ):
            if col in out.columns:
                out[col] = pd.to_numeric(out[col], errors="coerce")

        out = out.replace([np.inf, -np.inf], np.nan)
        out = out.dropna(subset=["close"])
        out = out[out["close"] > 0]
        if out.empty:
            return pd.DataFrame()

        # Ensure open is positive
        if "open" not in out.columns:
            out["open"] = out["close"]
        out["open"] = pd.to_numeric(
            out["open"], errors="coerce"
        ).fillna(0.0)
        out["open"] = out["open"].where(
            out["open"] > 0, out["close"]
        )

        # Ensure high/low are consistent
        if "high" not in out.columns:
            out["high"] = out[["open", "close"]].max(axis=1)
        else:
            out["high"] = pd.to_numeric(
                out["high"], errors="coerce"
            )

        if "low" not in out.columns:
            out["low"] = out[["open", "close"]].min(axis=1)
        else:
            out["low"] = pd.to_numeric(
                out["low"], errors="coerce"
            )

        oc_top = out[["open", "close"]].max(axis=1)
        oc_bot = out[["open", "close"]].min(axis=1)
        out["high"] = pd.concat(
            [out["high"], oc_top], axis=1
        ).max(axis=1)
        out["low"] = pd.concat(
            [out["low"], oc_bot], axis=1
        ).min(axis=1)

        # Ensure high >= low
        out = out[out["high"].fillna(0) >= out["low"].fillna(0)]

        # Non-negative volume
        if "volume" in out.columns:
            out = out[out["volume"].fillna(0) >= 0]
        else:
            out["volume"] = 0

        if "amount" not in out.columns:
            out["amount"] = (
                out["close"] * out["volume"].fillna(0)
            )

        if "turnover" not in out.columns:
            out["turnover"] = 0.0

        out = out.fillna(0)
        out = out[~out.index.duplicated(keep="last")].sort_index()
        return out

    # ------------------------------------------------------------------
    # Intraday bars
    # ------------------------------------------------------------------

    @staticmethod
    def _median_close_tail(frame: pd.DataFrame, tail_rows: int = 240) -> float:
        if frame is None or frame.empty or "close" not in frame.columns:
            return 0.0
        closes = pd.to_numeric(frame["close"], errors="coerce").dropna()
        closes = closes[closes > 0]
        if closes.empty:
            return 0.0
        return float(closes.tail(max(1, int(tail_rows))).median())

    def _latest_daily_close(
        self, code: str
    ) -> tuple[float, date | None]:
        try:
            cur = self._conn.execute(
                """
                SELECT date, close
                FROM daily_bars
                WHERE code = ?
                ORDER BY date DESC
                LIMIT 1
                """,
                (str(code).zfill(6),),
            )
            row = cur.fetchone()
        except _DB_SOFT_EXCEPTIONS:
            return 0.0, None
        if row is None:
            return 0.0, None
        close_px = self._to_float(row[1], 0.0)
        if close_px <= 0:
            return 0.0, None
        day_val = None
        try:
            day_val = datetime.strptime(str(row[0]), "%Y-%m-%d").date()
        except (TypeError, ValueError):
            day_val = None
        return float(close_px), day_val

    def _latest_intraday_close(
        self, code: str, interval: str
    ) -> float:
        try:
            cur = self._conn.execute(
                """
                SELECT close
                FROM intraday_bars
                WHERE code = ? AND interval = ?
                ORDER BY ts DESC
                LIMIT 1
                """,
                (str(code).zfill(6), str(interval or "1m").lower()),
            )
            row = cur.fetchone()
        except _DB_SOFT_EXCEPTIONS:
            return 0.0
        if row is None:
            return 0.0
        close_px = self._to_float(row[0], 0.0)
        return float(close_px) if close_px > 0 else 0.0

    def _intraday_scale_guard_allows(
        self,
        *,
        code: str,
        interval: str,
        work: pd.DataFrame,
    ) -> bool:
        iv = str(interval or "1m").strip().lower()
        if iv in ("1d", "1wk", "1mo"):
            return True

        med_close = self._median_close_tail(work, tail_rows=240)
        if med_close <= 0:
            return False

        daily_ref, daily_dt = self._latest_daily_close(code)
        if daily_ref > 0:
            min_ratio, max_ratio = 0.45, 2.2
            if daily_dt is not None:
                try:
                    age_days = max(0, (date.today() - daily_dt).days)
                except Exception:
                    age_days = 0
                if age_days > 45:
                    min_ratio, max_ratio = 0.30, 3.5
            ratio = med_close / max(daily_ref, 1e-8)
            if ratio < min_ratio or ratio > max_ratio:
                log.warning(
                    "Skipped intraday DB upsert for %s (%s): median %.6f "
                    "mismatches daily close %.6f (ratio=%.3f allowed %.3f..%.3f)",
                    str(code).zfill(6),
                    iv,
                    med_close,
                    daily_ref,
                    ratio,
                    min_ratio,
                    max_ratio,
                )
                return False
            return True

        intra_ref = self._latest_intraday_close(code, iv)
        if intra_ref > 0:
            ratio = med_close / max(intra_ref, 1e-8)
            min_ratio, max_ratio = 0.40, 3.5
            if ratio < min_ratio or ratio > max_ratio:
                log.warning(
                    "Skipped intraday DB upsert for %s (%s): median %.6f "
                    "mismatches prior intraday close %.6f (ratio=%.3f "
                    "allowed %.3f..%.3f)",
                    str(code).zfill(6),
                    iv,
                    med_close,
                    intra_ref,
                    ratio,
                    min_ratio,
                    max_ratio,
                )
                return False

        return True

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
                query,
                self._conn,
                params=(code, interval, limit),
            )
        except _DB_QUERY_EXCEPTIONS as e:
            log.warning(
                f"Intraday query failed for {code}/{interval}: {e}"
            )
            return pd.DataFrame()

        if df.empty:
            return pd.DataFrame()

        df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
        df = df.dropna(subset=["ts"]).set_index("ts").sort_index()
        return self._sanitize_intraday_frame(
            df, interval=interval
        )

    def upsert_intraday_bars(
        self, code: str, interval: str, df: pd.DataFrame
    ) -> None:
        """Insert/update intraday bars."""
        if df is None or df.empty:
            return

        self._validate_bar_columns(df, context="intraday_bars")

        code = str(code).zfill(6)
        interval = str(interval).lower()

        work = df.copy()
        if not isinstance(work.index, pd.DatetimeIndex):
            work.index = pd.to_datetime(
                work.index, errors="coerce"
            )
        work = work[~work.index.isna()]
        work = work.sort_index()
        work = work[~work.index.duplicated(keep="last")]

        # Defensive guard: keep CN intraday rows only within regular session
        if (
            interval not in ("1d", "1wk", "1mo")
            and code.isdigit()
            and len(code) == 6
        ):
            try:
                idx = work.index
                idx_local = idx
                if getattr(idx, "tz", None) is not None:
                    try:
                        from zoneinfo import ZoneInfo

                        idx_local = idx.tz_convert(
                            ZoneInfo("Asia/Shanghai")
                        ).tz_localize(None)
                    except (ImportError, ModuleNotFoundError, TypeError, ValueError):
                        idx_local = idx.tz_convert(None)
                hhmm = (idx_local.hour * 100) + idx_local.minute
                in_morning = (hhmm >= 930) & (hhmm <= 1130)
                in_afternoon = (hhmm >= 1300) & (hhmm <= 1500)
                weekday = idx_local.dayofweek < 5
                mask = weekday & (in_morning | in_afternoon)
                work = work.loc[mask]
            except (AttributeError, TypeError, ValueError):
                pass

        work = self._sanitize_intraday_frame(
            work, interval=interval
        )
        if work.empty:
            return
        if not self._intraday_scale_guard_allows(
            code=code,
            interval=interval,
            work=work,
        ):
            return

        rows = self._build_intraday_rows(code, interval, work)

        with self._transaction() as conn:
            conn.executemany(
                """
                INSERT INTO intraday_bars
                    (code, ts, interval, open, high, low, close,
                     volume, amount)
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
    ) -> list[tuple]:
        """Build rows for intraday bar insert.

        FIX #19: Uses itertuples() instead of iterrows() for performance.
        """
        tf = self._to_float
        ti = self._to_int

        rows = []
        for tup in work.itertuples():
            ts_iso = tup.Index.isoformat()
            rows.append(
                (
                    code,
                    ts_iso,
                    interval,
                    tf(getattr(tup, "open", 0)),
                    tf(getattr(tup, "high", 0)),
                    tf(getattr(tup, "low", 0)),
                    tf(getattr(tup, "close", 0)),
                    ti(getattr(tup, "volume", 0)),
                    tf(getattr(tup, "amount", 0)),
                )
            )
        return rows

    # ------------------------------------------------------------------
    # Stock metadata
    # ------------------------------------------------------------------

    def upsert_stock(
        self,
        code: str,
        name: str = None,
        exchange: str = None,
        sector: str = None,
        market_cap: float = None,
        is_st: bool = False,
    ) -> None:
        """Insert or update stock metadata."""
        with self._transaction() as conn:
            conn.execute(
                """
                INSERT INTO stocks
                    (code, name, exchange, sector, market_cap,
                     is_st, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(code) DO UPDATE SET
                    name = COALESCE(excluded.name, stocks.name),
                    exchange = COALESCE(
                        excluded.exchange, stocks.exchange
                    ),
                    sector = COALESCE(
                        excluded.sector, stocks.sector
                    ),
                    market_cap = COALESCE(
                        excluded.market_cap, stocks.market_cap
                    ),
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
    # Daily bars
    # ------------------------------------------------------------------

    def upsert_bars(self, code: str, df: pd.DataFrame) -> None:
        """Insert/update daily bars with column validation and sanitization.

        FIX #23: Now sanitizes daily bars (high >= low, positive close, etc.)
        for symmetry with intraday sanitization.
        """
        if df is None or df.empty:
            return

        self._validate_bar_columns(df, context="daily_bars")

        work = df.copy()
        if not isinstance(work.index, pd.DatetimeIndex):
            work.index = pd.to_datetime(
                work.index, errors="coerce"
            )
        work = work[~work.index.isna()]
        work = work.sort_index()
        work = work[~work.index.duplicated(keep="last")]

        if work.empty:
            return

        # FIX #23: Sanitize daily bars
        work = self._sanitize_daily_frame(work)
        if work.empty:
            return

        code = str(code).zfill(6)
        rows = self._build_daily_rows(code, work)

        with self._transaction() as conn:
            conn.executemany(
                """
                INSERT INTO daily_bars
                    (code, date, open, high, low, close, volume,
                     amount, turnover)
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
    ) -> list[tuple]:
        """Build rows for daily bar insert.

        FIX #19: Uses itertuples() instead of iterrows() for performance.
        """
        tf = self._to_float
        ti = self._to_int

        rows = []
        for tup in work.itertuples():
            rows.append(
                (
                    code,
                    tup.Index.strftime("%Y-%m-%d"),
                    tf(getattr(tup, "open", 0)),
                    tf(getattr(tup, "high", 0)),
                    tf(getattr(tup, "low", 0)),
                    tf(getattr(tup, "close", 0)),
                    ti(getattr(tup, "volume", 0)),
                    tf(getattr(tup, "amount", 0)),
                    tf(getattr(tup, "turnover", 0)),
                )
            )
        return rows

    # ------------------------------------------------------------------
    # Query methods
    # ------------------------------------------------------------------

    def get_bars(
        self,
        code: str,
        start_date: date = None,
        end_date: date = None,
        limit: int = None,
    ) -> pd.DataFrame:
        """Get daily bars for a stock.

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
            query = (
                f"SELECT * FROM ({inner_query}) ORDER BY date ASC"
            )
        else:
            query = (
                f"SELECT * FROM daily_bars "
                f"WHERE {where_clause} ORDER BY date ASC"
            )

        try:
            df = pd.read_sql_query(
                query, self._conn, params=params
            )
        except _DB_QUERY_EXCEPTIONS as e:
            log.warning(f"Database query failed for {code}: {e}")
            return pd.DataFrame()

        if df.empty:
            return pd.DataFrame()

        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()

        if "code" in df.columns:
            df = df.drop(columns=["code"])

        return df

    def get_last_date(self, code: str) -> date | None:
        """Get last available date for a stock."""
        code = str(code).zfill(6)
        try:
            cursor = self._conn.execute(
                "SELECT MAX(date) FROM daily_bars WHERE code = ?",
                (code,),
            )
            row = cursor.fetchone()
            if row and row[0]:
                return datetime.fromisoformat(row[0]).date()
        except _DB_SOFT_EXCEPTIONS as e:
            log.debug(f"get_last_date failed for {code}: {e}")
        return None

    def get_all_stocks(self) -> list[dict]:
        """Get all stock metadata."""
        try:
            cursor = self._conn.execute("SELECT * FROM stocks")
            return [dict(row) for row in cursor.fetchall()]
        except _DB_SOFT_EXCEPTIONS as e:
            log.warning(f"get_all_stocks failed: {e}")
            return []

    def get_stocks_with_data(
        self, min_days: int = 100
    ) -> list[str]:
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
        except _DB_SOFT_EXCEPTIONS as e:
            log.warning(f"get_stocks_with_data failed: {e}")
            return []

    # ------------------------------------------------------------------
    # Predictions
    # ------------------------------------------------------------------

    def save_prediction(
        self,
        code: str,
        signal: str,
        prob_up: float,
        prob_down: float,
        confidence: float,
        price: float,
    ) -> None:
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

    def get_data_stats(self) -> dict:
        """Get database statistics."""
        stats: dict = {}
        try:
            cursor = self._conn.execute(
                "SELECT COUNT(DISTINCT code) FROM stocks"
            )
            stats["total_stocks"] = cursor.fetchone()[0] or 0

            cursor = self._conn.execute(
                "SELECT COUNT(*) FROM daily_bars"
            )
            stats["total_bars"] = cursor.fetchone()[0] or 0

            cursor = self._conn.execute(
                "SELECT MIN(date), MAX(date) FROM daily_bars"
            )
            row = cursor.fetchone()
            stats["date_range"] = (
                (row[0], row[1]) if row else (None, None)
            )
        except _DB_SOFT_EXCEPTIONS as e:
            log.warning(f"get_data_stats failed: {e}")
            stats.setdefault("total_stocks", 0)
            stats.setdefault("total_bars", 0)
            stats.setdefault("date_range", (None, None))

        return stats

    def vacuum(self) -> None:
        """Optimize database.

        FIX #15: Uses a dedicated connection for VACUUM to avoid
        mutating the shared thread-local connection's isolation_level.
        """
        conn = None
        try:
            conn = sqlite3.connect(
                str(self._db_path), timeout=60
            )
            conn.execute("VACUUM")
        except _DB_SOFT_EXCEPTIONS as e:
            log.warning(f"VACUUM failed: {e}")
        finally:
            if conn is not None:
                try:
                    conn.close()
                except (sqlite3.Error, OSError):
                    pass

    def close(self) -> None:
        """Close this thread's connection."""
        if (
            hasattr(self._local, "conn")
            and self._local.conn is None
        ):
            return
        if hasattr(self._local, "conn"):
            conn = self._local.conn
            tid = threading.get_ident()
            try:
                conn.close()
            except (sqlite3.Error, OSError):
                pass
            with self._connections_lock:
                self._connections.pop(tid, None)
            self._local.conn = None

    def close_all(self) -> None:
        """Close all tracked connections (call on shutdown)."""
        with self._connections_lock:
            for _tid, conn in list(self._connections.items()):
                try:
                    conn.close()
                except (sqlite3.Error, OSError):
                    pass
            self._connections.clear()
        # Also close this thread's connection
        if (
            hasattr(self._local, "conn")
            and self._local.conn is not None
        ):
            try:
                self._local.conn.close()
            except (sqlite3.Error, OSError):
                pass
            self._local.conn = None

# Global database instance (thread-safe)

_db: MarketDatabase | None = None
_db_lock = threading.Lock()

def get_database() -> MarketDatabase:
    """Get global database instance (thread-safe)."""
    global _db
    if _db is None:
        with _db_lock:
            if _db is None:
                _db = MarketDatabase()
    return _db

def reset_database() -> None:
    """Reset global database instance (for testing).

    FIX #21: Provides a way to cleanly shut down and reset
    the global database singleton.
    """
    global _db
    with _db_lock:
        if _db is not None:
            try:
                _db.close_all()
            except _DB_SOFT_EXCEPTIONS:
                pass
            _db = None
