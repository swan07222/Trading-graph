# utils/security.py
from __future__ import annotations

import atexit
import gzip
import hashlib
import json
import os
import secrets
import threading
import weakref
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import IO, Any

from config.settings import CONFIG
from utils.logger import get_logger

log = get_logger(__name__)
_SECURITY_SOFT_EXCEPTIONS = (
    AttributeError,
    OSError,
    RuntimeError,
    TypeError,
    ValueError,
)
_ENV_SECURE_STORAGE_PATH = "TRADING_SECURE_STORAGE_PATH"
_ENV_SECURE_KEY_PATH = "TRADING_SECURE_KEY_PATH"
_ENV_SECURE_MASTER_KEY = "TRADING_SECURE_MASTER_KEY"
_ENV_LOCK_ACCESS_IDENTITY = "TRADING_LOCK_ACCESS_IDENTITY"

# Import cryptography once; SecureStorage is fail-closed when unavailable.
_CRYPTO_IMPORT_ERROR: ImportError | None
try:
    from cryptography.fernet import Fernet

    CRYPTO_AVAILABLE = True
except ImportError as exc:
    Fernet = None
    _CRYPTO_IMPORT_ERROR = exc
    CRYPTO_AVAILABLE = False
else:
    _CRYPTO_IMPORT_ERROR = None

class SecureStorage:
    """Secure storage for sensitive credentials.

    Fail-closed design:
    - requires cryptography/Fernet
    - never falls back to plain/base64 storage
    """

    def __init__(self) -> None:
        if not CRYPTO_AVAILABLE:
            raise RuntimeError(
                "cryptography is required for SecureStorage. "
                "Install with: pip install cryptography"
            ) from _CRYPTO_IMPORT_ERROR
        self._storage_path = self._resolve_storage_path()
        self._key_path = self._resolve_key_path()
        self._lock = threading.RLock()
        self._cipher = self._init_cipher()
        self._cache: dict[str, str] = {}
        self._closed = False
        self._load()

    def _resolve_storage_path(self) -> Path:
        raw = str(os.getenv(_ENV_SECURE_STORAGE_PATH, "") or "").strip()
        if raw:
            return Path(raw).expanduser()
        return CONFIG.data_dir / ".secure_storage.enc"

    def _resolve_key_path(self) -> Path:
        raw = str(os.getenv(_ENV_SECURE_KEY_PATH, "") or "").strip()
        if raw:
            return Path(raw).expanduser()
        # Keep key outside runtime data directory by default.
        return CONFIG.data_dir.parent / "secrets" / "trading_graph.key"

    @staticmethod
    def _set_private_perms(path: Path) -> None:
        try:
            os.chmod(path, 0o600)
        except OSError as e:
            log.warning("Cannot set file permissions for %s: %s", path, e)

    def _load_key_from_env(self) -> bytes | None:
        raw = str(os.getenv(_ENV_SECURE_MASTER_KEY, "") or "").strip()
        if not raw:
            return None
        return raw.encode("utf-8")

    def _read_key_file(self, path: Path) -> bytes:
        with open(path, "rb") as f:
            return f.read()

    def _write_key_file(self, path: Path, key: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        # FIX: Write key file with restricted permissions from the start
        fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        try:
            os.write(fd, key)
        finally:
            os.close(fd)
        self._set_private_perms(path)

    def _load_or_create_key(self) -> bytes:
        env_key = self._load_key_from_env()
        if env_key is not None:
            return env_key

        if self._key_path.exists():
            return self._read_key_file(self._key_path)

        legacy_key_path = CONFIG.data_dir / ".key"
        if legacy_key_path.exists():
            key = self._read_key_file(legacy_key_path)
            try:
                self._write_key_file(self._key_path, key)
            except OSError as e:
                log.warning("Failed to migrate secure key to %s: %s", self._key_path, e)
            return key

        if Fernet is None:
            raise RuntimeError("cryptography.Fernet unavailable")
        key = Fernet.generate_key()
        self._write_key_file(self._key_path, key)
        return key

    def _init_cipher(self) -> Any:
        """Initialize encryption cipher."""
        try:
            key = self._load_or_create_key()
            if Fernet is None:
                raise RuntimeError("cryptography.Fernet unavailable")
            return Fernet(key)
        except _SECURITY_SOFT_EXCEPTIONS as e:
            raise RuntimeError(f"Failed to initialize secure cipher: {e}") from e

    def _encrypt(self, data: str) -> bytes:
        """Encrypt data."""
        encrypted = self._cipher.encrypt(data.encode("utf-8"))
        return bytes(encrypted)

    def _decrypt(self, data: bytes) -> str:
        """Decrypt data."""
        decrypted = self._cipher.decrypt(data)
        return bytes(decrypted).decode("utf-8")

    @staticmethod
    def _normalize_cache(payload: Any) -> dict[str, str]:
        if not isinstance(payload, dict):
            return {}
        out: dict[str, str] = {}
        dropped = 0
        for raw_key, raw_val in payload.items():
            key = str(raw_key or "").strip()
            if not key:
                dropped += 1
                continue
            if isinstance(raw_val, str):
                out[key] = raw_val
            elif raw_val is None:
                out[key] = ""
            else:
                out[key] = str(raw_val)
        if dropped > 0:
            log.warning("Secure storage dropped %d invalid cache key(s)", dropped)
        return out

    def _load(self) -> None:
        """Load from storage."""
        if not self._storage_path.exists():
            return

        try:
            with open(self._storage_path, "rb") as f:
                encrypted = f.read()
            decrypted = self._decrypt(encrypted)
            decoded = json.loads(decrypted)
            if not isinstance(decoded, dict):
                log.warning("Secure storage payload must be a JSON object; resetting cache")
                self._cache = {}
                return
            self._cache = self._normalize_cache(decoded)
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as e:
            log.warning("Failed to load secure storage: %s", e)
            self._cache = {}

    def _save(self) -> None:
        """Save to storage atomically.

        Uses temp file + rename pattern with proper file descriptor cleanup.
        FIX: Ensures file descriptors are always closed, even on error.
        """
        tmp_path: Path | None = None
        try:
            data = json.dumps(self._cache)
            encrypted = self._encrypt(data)
            tmp_path = self._storage_path.with_suffix(".tmp")
            # FIX: Write temp file with restricted permissions from the start
            # to avoid race window where file exists with wrong permissions
            fd = os.open(
                str(tmp_path),
                os.O_WRONLY | os.O_CREAT | os.O_TRUNC,
                0o600  # Owner read/write only
            )
            try:
                os.write(fd, encrypted)
            finally:
                os.close(fd)  # FIX: Always close fd, even on error
            tmp_path.replace(self._storage_path)
            # Re-apply permissions after rename for extra safety
            try:
                self._set_private_perms(self._storage_path)
            except OSError as perm_err:
                log.debug("Secure storage permission update skipped: %s", perm_err)
        except (OSError, TypeError, ValueError) as e:
            log.error("Failed to save secure storage: %s", e)
            raise  # FIX: Re-raise to caller knows state wasn't saved
        finally:
            if tmp_path is not None and tmp_path.exists():
                try:
                    tmp_path.unlink()
                except OSError:
                    pass

    def set(self, key: str, value: str) -> None:
        """Store encrypted value."""
        # FIX #13: Input validation
        if not isinstance(key, str) or not key:
            raise ValueError("key must be a non-empty string")
        if not isinstance(value, str):
            raise TypeError("value must be a string")

        with self._lock:
            if self._closed:
                raise RuntimeError("SecureStorage is closed")
            self._cache[key] = value
            self._save()

    def get(self, key: str, default: str | None = None) -> str | None:
        """Retrieve decrypted value."""
        if not isinstance(key, str) or not key:
            raise ValueError("key must be a non-empty string")

        with self._lock:
            return self._cache.get(key, default)

    def delete(self, key: str) -> None:
        """Delete value."""
        if not isinstance(key, str) or not key:
            raise ValueError("key must be a non-empty string")

        with self._lock:
            if key in self._cache:
                del self._cache[key]
                self._save()

    def has(self, key: str) -> bool:
        """Check if key exists (thread-safe)."""
        # FIX #1: Acquire lock for consistency with set/delete
        with self._lock:
            return key in self._cache

    def close(self) -> None:
        """FIX #1: Explicitly close storage.
        Flushes any pending state. Idempotent.
        """
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._save()

    def __repr__(self) -> str:
        return f"SecureStorage(keys={len(self._cache)}, mode=encrypted)"

@dataclass
class AuditRecord:
    """Single audit record."""

    timestamp: datetime
    event_type: str
    user: str
    action: str
    details: dict[str, Any]
    ip_address: str = ""
    session_id: str = ""
    prev_hash: str = ""
    record_hash: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "event_type": self.event_type,
            "user": self.user,
            "action": self.action,
            "details": self.details,
            "ip_address": self.ip_address,
            "session_id": self.session_id,
            "prev_hash": self.prev_hash,
            "record_hash": self.record_hash,
        }

def _atexit_close_audit(ref: weakref.ReferenceType[AuditLog]) -> None:
    """FIX #4: atexit callback using weakref.
    If the AuditLog was already garbage collected, this is a no-op.
    """
    obj = ref()
    if obj is not None:
        obj.close()

class AuditLog:
    """Comprehensive audit logging for compliance.

    FIXES:
    - _flush only clears buffer on success (#3)
    - atexit uses weakref to avoid preventing GC (#4)
    - query flushes current buffer first (#6)
    - close is idempotent
    """

    def __init__(self) -> None:
        self._log_dir = CONFIG.audit_dir
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._current_file: IO[str] | None = None
        self._current_date: date | None = None
        self._lock = threading.RLock()
        self._buffer: list[AuditRecord] = []
        self._buffer_size = 100
        self._max_buffer_size = 10000  # Hard cap to prevent memory issues
        self._user = "system"
        self._session_id = secrets.token_hex(16)
        self._closed = False
        self._prev_hash = ""
        self._last_prune_date: date | None = None

        # FIX #4: Use weakref so GC can collect this if all references drop
        self._atexit_ref = weakref.ref(self)
        atexit.register(_atexit_close_audit, self._atexit_ref)

    def _get_file(self) -> IO[str] | None:
        """Get current log file, rotating daily."""
        today = date.today()

        if self._current_date != today:
            if self._current_file:
                try:
                    self._current_file.close()
                except _SECURITY_SOFT_EXCEPTIONS as e:
                    log.debug("Audit log file close during rotation failed: %s", e)
                self._current_file = None

            sid = (self._session_id or "nosession")[:8]
            path = self._log_dir / f"audit_{today.isoformat()}_{sid}.jsonl.gz"
            try:
                self._current_file = gzip.open(path, "at", encoding="utf-8")
                self._current_date = today
                self._maybe_auto_prune(today)
            except _SECURITY_SOFT_EXCEPTIONS as e:
                log.error("Failed to open audit log file: %s", e)
                self._current_file = None

        return self._current_file

    def _maybe_auto_prune(self, today: date) -> None:
        """Run daily retention prune based on security config."""
        try:
            sec = getattr(CONFIG, "security", None)
            if not bool(getattr(sec, "audit_auto_prune", True)):
                return
            if self._last_prune_date == today:
                return
            self._last_prune_date = today
            retention_days = int(getattr(sec, "audit_retention_days", 365))
            self.prune_old_files(retention_days=retention_days)
        except _SECURITY_SOFT_EXCEPTIONS as e:
            log.debug("Audit auto-prune skipped: %s", e)

    def _write(self, record: AuditRecord) -> None:
        """Write record to buffer."""
        if self._closed:
            return

        with self._lock:
            record.prev_hash = self._prev_hash
            record.record_hash = self._compute_record_hash(record)
            self._prev_hash = record.record_hash
            self._buffer.append(record)
            if len(self._buffer) >= self._buffer_size:
                self._flush()

    @staticmethod
    def _canonical_details(details: dict[str, Any]) -> str:
        try:
            return json.dumps(details, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        except (TypeError, ValueError):
            return "{}"

    def _compute_record_hash(self, record: AuditRecord) -> str:
        payload = "|".join([
            record.timestamp.isoformat(),
            record.event_type,
            record.user,
            record.action,
            self._canonical_details(record.details),
            record.ip_address or "",
            record.session_id or "",
            record.prev_hash or "",
        ])
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _flush(self) -> None:
        """FIX #3: Flush buffer to file. Only clears buffer on SUCCESS.
        """
        if not self._buffer:
            return

        f = self._get_file()
        if f is None:
            # Cannot write: keep buffer for retry, but cap size
            if len(self._buffer) > self._max_buffer_size:
                dropped = len(self._buffer) - self._max_buffer_size
                self._buffer = self._buffer[-self._max_buffer_size:]
                log.warning("Audit buffer overflow: dropped %d records", dropped)
            return

        try:
            for record in self._buffer:
                f.write(json.dumps(record.to_dict()) + "\n")
            f.flush()
            self._buffer.clear()
        except _SECURITY_SOFT_EXCEPTIONS as e:
            log.error("Audit log flush failed: %s", e)

    def log(
        self,
        event_type: str,
        action: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Log audit event."""
        record = AuditRecord(
            timestamp=datetime.now(),
            event_type=event_type,
            user=self._user,
            action=action,
            details=dict(details) if details else {},
            session_id=self._session_id,
        )
        self._write(record)

    def log_signal(
        self,
        code: str,
        signal: str,
        confidence: float,
        price: float,
        reasons: list[str] | None = None,
    ) -> None:
        """Log trading signal."""
        self.log(
            "signal",
            "generated",
            {
                "code": code,
                "signal": signal,
                "confidence": confidence,
                "price": price,
                "reasons": reasons or [],
            },
        )

    def log_order(
        self,
        order_id: str,
        code: str,
        side: str,
        quantity: int,
        price: float,
        status: str,
    ) -> None:
        """Log order event."""
        self.log(
            "order",
            status,
            {
                "order_id": order_id,
                "code": code,
                "side": side,
                "quantity": quantity,
                "price": price,
            },
        )

    def log_trade(
        self,
        order_id: str,
        code: str,
        side: str,
        quantity: int,
        price: float,
        commission: float,
        pnl: float | None = None,
    ) -> None:
        """Log trade execution."""
        self.log(
            "trade",
            "executed",
            {
                "order_id": order_id,
                "code": code,
                "side": side,
                "quantity": quantity,
                "price": price,
                "commission": commission,
                "pnl": pnl,
            },
        )

    def log_risk_event(self, event_type: str, details: dict[str, Any]) -> None:
        """Log risk management event."""
        self.log("risk", event_type, details)

    def log_access(self, action: str, resource: str, allowed: bool) -> None:
        """Log access control event."""
        self.log("access", action, {"resource": resource, "allowed": allowed})

    def set_user(self, user: str) -> None:
        """Set current user."""
        if not isinstance(user, str) or not user.strip():
            raise ValueError("user must be a non-empty string")
        self._user = user.strip()

    def new_session(self) -> None:
        """Start new session."""
        self._session_id = secrets.token_hex(16)
        self.log("session", "started", {})

    def close(self) -> None:
        """Close audit log (idempotent)."""
        if self._closed:
            return
        self._closed = True

        with self._lock:
            self._flush()
            if self._current_file:
                try:
                    self._current_file.close()
                except _SECURITY_SOFT_EXCEPTIONS as e:
                    log.debug("Audit log file close during shutdown failed: %s", e)
                self._current_file = None

    def mark_legal_hold(self, audit_file: Path) -> bool:
        """Mark an audit file as legal-hold protected via sidecar marker."""
        try:
            p = Path(audit_file)
            sidecar = p.with_name(p.name + ".hold")
            sidecar.write_text(
                json.dumps({"marked_at": datetime.now().isoformat()}),
                encoding="utf-8",
            )
            return True
        except _SECURITY_SOFT_EXCEPTIONS as e:
            log.debug("Failed to mark legal hold for %s: %s", audit_file, e)
            return False

    def unmark_legal_hold(self, audit_file: Path) -> bool:
        """Remove legal-hold sidecar marker."""
        try:
            p = Path(audit_file)
            sidecar = p.with_name(p.name + ".hold")
            if sidecar.exists():
                sidecar.unlink()
            return True
        except _SECURITY_SOFT_EXCEPTIONS as e:
            log.debug("Failed to unmark legal hold for %s: %s", audit_file, e)
            return False

    def prune_old_files(self, retention_days: int) -> dict[str, int]:
        """Delete audit files older than retention_days, except legal-hold files.
        """
        stats = {"deleted": 0, "held": 0, "kept": 0}
        try:
            keep_since = date.today() - timedelta(days=max(1, int(retention_days)))
            for path in self._log_dir.glob("audit_*.jsonl.gz"):
                try:
                    parts = path.name.split("_")
                    if len(parts) < 2:
                        stats["kept"] += 1
                        continue
                    file_day = date.fromisoformat(parts[1])
                except (TypeError, ValueError):
                    stats["kept"] += 1
                    continue

                if file_day >= keep_since:
                    stats["kept"] += 1
                    continue

                hold = path.with_name(path.name + ".hold")
                if hold.exists():
                    stats["held"] += 1
                    continue

                try:
                    path.unlink()
                    stats["deleted"] += 1
                except OSError:
                    stats["kept"] += 1
            if stats["deleted"] > 0:
                log.info(
                    "Audit retention pruned: deleted=%d held=%d kept=%d",
                    stats["deleted"], stats["held"], stats["kept"],
                )
        except _SECURITY_SOFT_EXCEPTIONS as e:
            log.debug("Audit prune failed: %s", e)
        return stats

    def query(
        self,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        event_type: str | None = None,
        limit: int = 1000,
    ) -> list[dict[str, Any]]:
        """Query audit records.

        FIX #6: Flushes current buffer first so recent records are included.
        """
        # Flush so in-memory records are queryable
        with self._lock:
            self._flush()

        results: list[dict[str, Any]] = []

        start = (
            start_date.date()
            if start_date
            else date.today() - timedelta(days=30)
        )
        end = end_date.date() if end_date else date.today()

        current = start
        while current <= end:
            pattern = f"audit_{current.isoformat()}_*.jsonl.gz"
            for path in sorted(self._log_dir.glob(pattern)):
                try:
                    with gzip.open(path, "rt", encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            record = json.loads(line)

                            if (
                                event_type
                                and record.get("event_type") != event_type
                            ):
                                continue

                            ts = datetime.fromisoformat(record["timestamp"])
                            if start_date and ts < start_date:
                                continue
                            if end_date and ts > end_date:
                                continue

                            results.append(record)
                            if len(results) >= limit:
                                return results
                except _SECURITY_SOFT_EXCEPTIONS as e:
                    log.debug("Failed to read audit file %s: %s", path, e)
                    continue

            current += timedelta(days=1)

        return results

    def verify_integrity(
        self,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        limit: int = 100000,
    ) -> dict[str, Any]:
        """Verify tamper-evident hash chain across queried audit records.
        """
        records = self.query(start_date=start_date, end_date=end_date, limit=limit)
        checked = 0
        prev_hash = ""
        for rec in records:
            checked += 1
            expected_payload = "|".join([
                rec.get("timestamp", ""),
                rec.get("event_type", ""),
                rec.get("user", ""),
                rec.get("action", ""),
                self._canonical_details(rec.get("details") or {}),
                rec.get("ip_address", "") or "",
                rec.get("session_id", "") or "",
                rec.get("prev_hash", "") or "",
            ])
            expected_hash = hashlib.sha256(expected_payload.encode("utf-8")).hexdigest()
            if rec.get("record_hash", "") != expected_hash:
                return {
                    "ok": False,
                    "checked": checked,
                    "reason": "record_hash_mismatch",
                    "record": rec,
                }
            if rec.get("prev_hash", "") != prev_hash:
                return {
                    "ok": False,
                    "checked": checked,
                    "reason": "chain_break",
                    "record": rec,
                }
            prev_hash = rec.get("record_hash", "")
        return {"ok": True, "checked": checked, "last_hash": prev_hash}

class RateLimiter:
    """Rate limiter for API calls and trading actions.

    FIX #7: check() and wait_if_needed() don't double-count.
    FIX #8: set_limit validates value.
    FIX #9: Bounded window memory.
    """

    _MAX_WINDOW_SIZE = 10000  # Hard cap per window

    def __init__(self) -> None:
        self._limits: dict[str, int] = {
            "orders_per_minute": 10,
            "orders_per_hour": 100,
            "api_calls_per_second": 10,
            "predictions_per_minute": 60,
        }
        self._windows: dict[str, list[datetime]] = {}
        self._lock = threading.RLock()

    def _get_window_duration(self, limit_type: str) -> timedelta:
        """Determine window duration from limit type name."""
        if "minute" in limit_type:
            return timedelta(minutes=1)
        elif "hour" in limit_type:
            return timedelta(hours=1)
        else:
            return timedelta(seconds=1)

    def check(self, limit_type: str, consume: bool = True) -> bool:
        """Check if action is allowed.

        FIX #7: Added `consume` parameter. When False, does a read-only check
        (used by wait_if_needed to avoid double-counting).
        """
        with self._lock:
            limit = self._limits.get(limit_type, 100)
            window = self._get_window_duration(limit_type)
            now = datetime.now()
            cutoff = now - window

            if limit_type not in self._windows:
                self._windows[limit_type] = []

            entries = self._windows[limit_type]
            self._windows[limit_type] = [t for t in entries if t > cutoff]

            if len(self._windows[limit_type]) >= limit:
                return False

            if consume:
                self._windows[limit_type].append(now)

                # FIX #9: Hard cap to prevent unbounded memory
                if len(self._windows[limit_type]) > self._MAX_WINDOW_SIZE:
                    self._windows[limit_type] = self._windows[limit_type][
                        -self._MAX_WINDOW_SIZE:
                    ]

            return True

    def wait_if_needed(self, limit_type: str, timeout: float = 60.0) -> bool:
        """Wait until action is allowed, then consume a slot.

        FIX #7: Uses consume=False for polling, then consume=True once allowed.
        """
        import time as _time

        start = _time.monotonic()

        while True:
            if self.check(limit_type, consume=False):
                if self.check(limit_type, consume=True):
                    return True
                # Race: another thread consumed between peek and consume: retry.
                continue

            if _time.monotonic() - start > timeout:
                return False
            _time.sleep(0.1)

    def set_limit(self, limit_type: str, value: int) -> None:
        """Set rate limit.

        FIX #8: Validates the value.
        """
        if not isinstance(limit_type, str) or not limit_type:
            raise ValueError("limit_type must be a non-empty string")
        if not isinstance(value, int) or value <= 0:
            raise ValueError(f"limit value must be a positive integer, got {value}")
        self._limits[limit_type] = value

    def get_usage(self, limit_type: str) -> dict[str, Any]:
        """Get current usage for a limit type."""
        with self._lock:
            limit = self._limits.get(limit_type, 100)
            window = self._get_window_duration(limit_type)
            cutoff = datetime.now() - window

            entries = self._windows.get(limit_type, [])
            current = sum(1 for t in entries if t > cutoff)

            return {
                "limit_type": limit_type,
                "limit": limit,
                "current": current,
                "remaining": max(0, limit - current),
                "window_seconds": window.total_seconds(),
            }

class AccessControl:
    """Access control for trading operations.

    FIX #10: Lazy audit with explicit recursion guard.
    FIX #11: Input validation on all public methods.
    """

    VALID_PERMISSIONS = frozenset({
        "view",
        "analyze",
        "trade_paper",
        "trade_live",
        "admin",
        "configure",
    })

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._permissions: dict[str, list[str]] = {
            "admin": ["*"],
            "trader": ["view", "trade_paper", "analyze"],
            "viewer": ["view", "analyze"],
            "live_trader": ["view", "trade_paper", "trade_live", "analyze"],
        }
        self._current_role = "trader"
        self._current_user = "system"
        self._session_started_at = datetime.now()
        self._session_ip: str = ""
        self._two_factor_verified = False
        self._two_factor_verified_at: datetime | None = None
        self._audit: AuditLog | None = None
        self._audit_logging_guard = threading.local()
        self._identity_locked = str(
            os.getenv(_ENV_LOCK_ACCESS_IDENTITY, "0")
        ).strip().lower() in {"1", "true", "yes", "on"}
        self._identity_lock_reason = "env_lock" if self._identity_locked else ""

    def _get_audit(self) -> AuditLog | None:
        """Lazy audit log access to break circular init."""
        if self._audit is None:
            try:
                self._audit = get_audit_log()
            except _SECURITY_SOFT_EXCEPTIONS:
                self._audit = None
        return self._audit

    def _ensure_identity_mutable(self) -> None:
        if self._identity_locked:
            raise RuntimeError(
                f"Access identity is locked ({self._identity_lock_reason or 'locked'})"
            )

    def lock_identity(self, reason: str = "manual") -> None:
        with self._lock:
            self._identity_locked = True
            self._identity_lock_reason = str(reason or "manual")

    def unlock_identity(self) -> None:
        with self._lock:
            self._identity_locked = False
            self._identity_lock_reason = ""

    def set_role(self, role: str) -> None:
        """Set current role."""
        if not isinstance(role, str) or not role:
            raise ValueError("role must be a non-empty string")
        with self._lock:
            self._ensure_identity_mutable()
            if role not in self._permissions:
                raise ValueError(
                    f"Unknown role {role!r}. Valid: {list(self._permissions.keys())}"
                )
            self._current_role = role
            audit = self._get_audit()
            if audit is not None:
                audit.log_access("set_role", role, True)

    def set_user(self, user: str) -> None:
        if not isinstance(user, str) or not user.strip():
            raise ValueError("user must be a non-empty string")
        with self._lock:
            self._ensure_identity_mutable()
            self._current_user = user.strip()
            audit = self._get_audit()
            if audit is not None:
                audit.set_user(self._current_user)

    def set_session_ip(self, ip: str) -> None:
        if not isinstance(ip, str):
            raise ValueError("ip must be a string")
        with self._lock:
            self._session_ip = ip.strip()

    def mark_2fa_verified(self, verified: bool = True) -> None:
        with self._lock:
            self._two_factor_verified = bool(verified)
            if self._two_factor_verified:
                self._two_factor_verified_at = datetime.now()
            else:
                self._two_factor_verified_at = None

    def _is_2fa_valid(self) -> bool:
        with self._lock:
            if not self._two_factor_verified:
                return False
            verified_at = self._two_factor_verified_at
            if verified_at is None:
                return False

            sec = getattr(CONFIG, "security", None)
            ttl_minutes = int(getattr(sec, "two_factor_ttl_minutes", 30) or 30)
            ttl_minutes = max(1, ttl_minutes)
            age = datetime.now() - verified_at
            if age.total_seconds() > float(ttl_minutes * 60):
                self._two_factor_verified = False
                self._two_factor_verified_at = None
                return False
            return True

    def create_role(self, role: str, permissions: list[str]) -> None:
        if not isinstance(role, str) or not role.strip():
            raise ValueError("role must be a non-empty string")
        if not isinstance(permissions, list):
            raise ValueError("permissions must be a list")
        clean: list[str] = []
        for p in permissions:
            p = str(p).strip()
            if not p:
                continue
            if p != "*" and p not in self.VALID_PERMISSIONS:
                raise ValueError(f"Unknown permission: {p}")
            clean.append(p)
        with self._lock:
            self._permissions[role.strip()] = sorted(set(clean))

    def grant_permission(self, role: str, permission: str) -> None:
        if not isinstance(role, str) or not role.strip():
            raise ValueError("role must be a non-empty string")
        if not isinstance(permission, str) or not permission.strip():
            raise ValueError("permission must be a non-empty string")
        p = permission.strip()
        if p != "*" and p not in self.VALID_PERMISSIONS:
            raise ValueError(f"Unknown permission: {p}")
        with self._lock:
            if role not in self._permissions:
                self._permissions[role] = []
            if p not in self._permissions[role]:
                self._permissions[role].append(p)

    def revoke_permission(self, role: str, permission: str) -> None:
        with self._lock:
            perms = self._permissions.get(role, [])
            p = str(permission or "").strip()
            if p in perms:
                perms.remove(p)

    def validate_session_policy(self) -> tuple[bool, str]:
        sec = getattr(CONFIG, "security", None)
        max_hours = int(getattr(sec, "max_session_hours", 8))
        ip_whitelist = list(getattr(sec, "ip_whitelist", []) or [])

        with self._lock:
            elapsed = datetime.now() - self._session_started_at
            if elapsed.total_seconds() > max(1, max_hours) * 3600:
                return False, f"session expired ({elapsed.total_seconds() / 3600.0:.1f}h)"

            if ip_whitelist:
                current_ip = self._session_ip.strip()
                if current_ip and current_ip not in ip_whitelist:
                    return False, f"ip not allowed ({current_ip})"

        return True, "ok"

    def check(self, permission: str) -> bool:
        """Check if current role has permission."""
        if not isinstance(permission, str) or not permission:
            raise ValueError("permission must be a non-empty string")

        allowed = False
        role = ""
        with self._lock:
            role = self._current_role
            perms = self._permissions.get(role, [])
            allowed = "*" in perms or permission in perms

            # Hard policy gates for privileged actions.
            if allowed and permission == "trade_live":
                ok, _reason = self.validate_session_policy()
                if not ok:
                    allowed = False
                elif bool(getattr(CONFIG.security, "require_2fa_for_live", True)):
                    allowed = allowed and self._is_2fa_valid()

        # Thread-local recursion guard: audit.log_access might trigger check()
        if not bool(getattr(self._audit_logging_guard, "active", False)):
            self._audit_logging_guard.active = True
            try:
                audit = self._get_audit()
                if audit is not None:
                    audit.log_access(permission, role, allowed)
            except _SECURITY_SOFT_EXCEPTIONS as e:
                log.debug("Access audit logging skipped: %s", e)
            finally:
                self._audit_logging_guard.active = False

        return allowed

    def require(self, permission: str) -> None:
        """Require permission, raise if not allowed."""
        if not self.check(permission):
            from core.exceptions import AuthorizationError

            raise AuthorizationError(
                f"Permission denied: {permission!r} "
                f"(role: {self._current_role!r})"
            )

    @contextmanager
    def elevate(self, role: str) -> Any:
        """Temporarily elevate to role."""
        with self._lock:
            if role not in self._permissions:
                raise ValueError(f"Unknown role {role!r}")
            old_role = self._current_role
            self._current_role = role
        try:
            yield
        finally:
            with self._lock:
                self._current_role = old_role

    @property
    def current_role(self) -> str:
        """Current role (read-only)."""
        with self._lock:
            return self._current_role

    @property
    def available_roles(self) -> list[str]:
        """List of available roles."""
        with self._lock:
            return list(self._permissions.keys())

    @property
    def identity_locked(self) -> bool:
        with self._lock:
            return bool(self._identity_locked)

# Module-level singletons with proper locks

_secure_storage: SecureStorage | None = None
_secure_storage_lock = threading.Lock()

_audit_log: AuditLog | None = None
_audit_log_lock = threading.Lock()

_rate_limiter: RateLimiter | None = None
_rate_limiter_lock = threading.Lock()

_access_control: AccessControl | None = None
_access_control_lock = threading.Lock()

def get_secure_storage() -> SecureStorage:
    """Get or create the global SecureStorage instance."""
    global _secure_storage
    if _secure_storage is None:
        with _secure_storage_lock:
            if _secure_storage is None:
                _secure_storage = SecureStorage()
    return _secure_storage

def get_audit_log() -> AuditLog:
    """Get or create the global AuditLog instance."""
    global _audit_log
    if _audit_log is None:
        with _audit_log_lock:
            if _audit_log is None:
                _audit_log = AuditLog()
    return _audit_log

def get_rate_limiter() -> RateLimiter:
    """Get or create the global RateLimiter instance."""
    global _rate_limiter
    if _rate_limiter is None:
        with _rate_limiter_lock:
            if _rate_limiter is None:
                _rate_limiter = RateLimiter()
    return _rate_limiter

def get_access_control() -> AccessControl:
    """Get or create the global AccessControl instance."""
    global _access_control
    if _access_control is None:
        with _access_control_lock:
            if _access_control is None:
                _access_control = AccessControl()
    return _access_control

def reset_security_singletons() -> None:
    """Reset all singletons for testing only.
    Closes resources cleanly before discarding.
    """
    global _secure_storage, _audit_log, _rate_limiter, _access_control

    with _secure_storage_lock:
        if _secure_storage is not None:
            _secure_storage.close()
            _secure_storage = None

    with _audit_log_lock:
        if _audit_log is not None:
            _audit_log.close()
            _audit_log = None

    with _rate_limiter_lock:
        _rate_limiter = None

    with _access_control_lock:
        _access_control = None
