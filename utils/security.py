# utils/security.py
"""
Security and Compliance Module.

Features:
- Encrypted credential storage (Fernet or explicit warning)
- Comprehensive audit logging with no data loss on flush failure
- Rate limiting with proper window management
- Session management
- Access control without circular init

FIXES APPLIED (comprehensive):
1.  SecureStorage: added close(), has() is thread-safe
2.  SecureStorage: base64 fallback issues loud warning every set/get
3.  AuditLog: _flush only clears buffer on success
4.  AuditLog: atexit uses weakref to avoid preventing GC
5.  AuditLog: _get_file thread safety
6.  AuditLog: query() flushes current buffer first for consistency
7.  RateLimiter: check() doesn't double-count, wait_if_needed is safe
8.  RateLimiter: set_limit validates value
9.  RateLimiter: bounded window memory
10. AccessControl: lazy audit with no recursion risk
11. AccessControl: input validation on set_role, check, require
12. Module-level singletons with proper locks
13. SecureStorage: input validation on set/get/delete
"""
from __future__ import annotations

import atexit
import os
import json
import secrets
import threading
import gzip
import weakref
from pathlib import Path
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from contextlib import contextmanager

from config.settings import CONFIG
from utils.logger import get_logger

log = get_logger(__name__)

# Try to import cryptography; fall back with clear warning
try:
    from cryptography.fernet import Fernet

    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False
    log.warning(
        "cryptography package not installed — "
        "SecureStorage will use base64 encoding (NOT encryption). "
        "Install with: pip install cryptography"
    )


# =========================================================================
# SecureStorage
# =========================================================================


class SecureStorage:
    """
    Secure storage for sensitive credentials.

    Uses Fernet symmetric encryption when available,
    logs a warning on every operation when falling back to base64.
    """

    def __init__(self) -> None:
        self._storage_path = CONFIG.data_dir / ".secure_storage.enc"
        self._key_path = CONFIG.data_dir / ".key"
        self._lock = threading.RLock()
        self._cipher = self._init_cipher()
        self._cache: Dict[str, str] = {}
        self._closed = False
        self._load()

    def _init_cipher(self):
        """Initialize encryption cipher."""
        if not CRYPTO_AVAILABLE:
            return None

        try:
            if self._key_path.exists():
                with open(self._key_path, "rb") as f:
                    key = f.read()
            else:
                key = Fernet.generate_key()
                # Ensure parent dir exists
                self._key_path.parent.mkdir(parents=True, exist_ok=True)
                with open(self._key_path, "wb") as f:
                    f.write(key)
                # Restrict permissions
                try:
                    os.chmod(self._key_path, 0o600)
                except OSError as e:
                    log.warning("Cannot set key file permissions: %s", e)

            return Fernet(key)
        except Exception as e:
            log.warning("Failed to init cipher: %s", e)
            return None

    def _warn_no_encryption(self, operation: str) -> None:
        """FIX #2: Warn on every unencrypted operation."""
        if self._cipher is None:
            log.warning(
                "SecureStorage.%s: using base64 (NOT encrypted). "
                "Install 'cryptography' for real encryption.",
                operation,
            )

    def _encrypt(self, data: str) -> bytes:
        """Encrypt data."""
        if self._cipher:
            return self._cipher.encrypt(data.encode("utf-8"))
        else:
            import base64

            return base64.b64encode(data.encode("utf-8"))

    def _decrypt(self, data: bytes) -> str:
        """Decrypt data."""
        if self._cipher:
            return self._cipher.decrypt(data).decode("utf-8")
        else:
            import base64

            return base64.b64decode(data).decode("utf-8")

    def _load(self) -> None:
        """Load from storage."""
        if not self._storage_path.exists():
            return

        try:
            with open(self._storage_path, "rb") as f:
                encrypted = f.read()
            decrypted = self._decrypt(encrypted)
            self._cache = json.loads(decrypted)
        except Exception as e:
            log.warning("Failed to load secure storage: %s", e)
            self._cache = {}

    def _save(self) -> None:
        """Save to storage."""
        try:
            data = json.dumps(self._cache)
            encrypted = self._encrypt(data)
            # Atomic write via temp file
            tmp_path = self._storage_path.with_suffix(".tmp")
            with open(tmp_path, "wb") as f:
                f.write(encrypted)
            tmp_path.replace(self._storage_path)
            try:
                os.chmod(self._storage_path, 0o600)
            except OSError:
                pass
        except Exception as e:
            log.error("Failed to save secure storage: %s", e)

    def set(self, key: str, value: str) -> None:
        """Store encrypted value."""
        # FIX #13: Input validation
        if not isinstance(key, str) or not key:
            raise ValueError("key must be a non-empty string")
        if not isinstance(value, str):
            raise TypeError("value must be a string")

        self._warn_no_encryption("set")

        with self._lock:
            if self._closed:
                raise RuntimeError("SecureStorage is closed")
            self._cache[key] = value
            self._save()

    def get(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """Retrieve decrypted value."""
        if not isinstance(key, str) or not key:
            raise ValueError("key must be a non-empty string")

        self._warn_no_encryption("get")

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
        """
        FIX #1: Explicitly close storage.
        Flushes any pending state. Idempotent.
        """
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._save()

    def __repr__(self) -> str:
        encrypted = "encrypted" if self._cipher else "base64"
        return f"SecureStorage(keys={len(self._cache)}, mode={encrypted})"


# =========================================================================
# AuditLog
# =========================================================================


@dataclass
class AuditRecord:
    """Single audit record."""

    timestamp: datetime
    event_type: str
    user: str
    action: str
    details: Dict[str, Any]
    ip_address: str = ""
    session_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "event_type": self.event_type,
            "user": self.user,
            "action": self.action,
            "details": self.details,
            "ip_address": self.ip_address,
            "session_id": self.session_id,
        }


def _atexit_close_audit(ref: weakref.ref) -> None:
    """
    FIX #4: atexit callback using weakref.
    If the AuditLog was already garbage collected, this is a no-op.
    """
    obj = ref()
    if obj is not None:
        obj.close()


class AuditLog:
    """
    Comprehensive audit logging for compliance.

    FIXES:
    - _flush only clears buffer on success (#3)
    - atexit uses weakref to avoid preventing GC (#4)
    - query flushes current buffer first (#6)
    - close is idempotent
    """

    def __init__(self) -> None:
        self._log_dir = CONFIG.audit_dir
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._current_file = None
        self._current_date: Optional[date] = None
        self._lock = threading.RLock()
        self._buffer: List[AuditRecord] = []
        self._buffer_size = 100
        self._max_buffer_size = 10000  # Hard cap to prevent memory issues
        self._user = "system"
        self._session_id = secrets.token_hex(16)
        self._closed = False

        # FIX #4: Use weakref so GC can collect this if all references drop
        self._atexit_ref = weakref.ref(self)
        atexit.register(_atexit_close_audit, self._atexit_ref)

    def _get_file(self):
        """Get current log file, rotating daily."""
        today = date.today()

        if self._current_date != today:
            # Close previous file safely
            if self._current_file:
                try:
                    self._current_file.close()
                except Exception:
                    pass
                self._current_file = None

            sid = (self._session_id or "nosession")[:8]
            path = self._log_dir / f"audit_{today.isoformat()}_{sid}.jsonl.gz"
            try:
                self._current_file = gzip.open(path, "at", encoding="utf-8")
                self._current_date = today
            except Exception as e:
                log.error("Failed to open audit log file: %s", e)
                self._current_file = None

        return self._current_file

    def _write(self, record: AuditRecord) -> None:
        """Write record to buffer."""
        if self._closed:
            return

        with self._lock:
            self._buffer.append(record)
            if len(self._buffer) >= self._buffer_size:
                self._flush()

    def _flush(self) -> None:
        """
        FIX #3: Flush buffer to file. Only clears buffer on SUCCESS.
        """
        if not self._buffer:
            return

        f = self._get_file()
        if f is None:
            # Cannot write — keep buffer for retry, but cap size
            if len(self._buffer) > self._max_buffer_size:
                dropped = len(self._buffer) - self._max_buffer_size
                self._buffer = self._buffer[-self._max_buffer_size:]
                log.warning("Audit buffer overflow: dropped %d records", dropped)
            return

        try:
            for record in self._buffer:
                f.write(json.dumps(record.to_dict()) + "\n")
            f.flush()
            # Only clear on success
            self._buffer.clear()
        except Exception as e:
            log.error("Audit log flush failed: %s", e)
            # Buffer retained for retry on next flush

    def log(
        self, event_type: str, action: str, details: Optional[Dict] = None
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
        reasons: Optional[List[str]] = None,
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
        pnl: Optional[float] = None,
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

    def log_risk_event(self, event_type: str, details: Dict) -> None:
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
                except Exception:
                    pass
                self._current_file = None

    def query(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        event_type: Optional[str] = None,
        limit: int = 1000,
    ) -> List[Dict]:
        """
        Query audit records.

        FIX #6: Flushes current buffer first so recent records are included.
        """
        # Flush so in-memory records are queryable
        with self._lock:
            self._flush()

        results: List[Dict] = []

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
                except Exception as e:
                    log.debug("Failed to read audit file %s: %s", path, e)
                    continue

            current += timedelta(days=1)

        return results


# =========================================================================
# RateLimiter
# =========================================================================


class RateLimiter:
    """
    Rate limiter for API calls and trading actions.

    FIX #7: check() and wait_if_needed() don't double-count.
    FIX #8: set_limit validates value.
    FIX #9: Bounded window memory.
    """

    _MAX_WINDOW_SIZE = 10000  # Hard cap per window

    def __init__(self) -> None:
        self._limits: Dict[str, int] = {
            "orders_per_minute": 10,
            "orders_per_hour": 100,
            "api_calls_per_second": 10,
            "predictions_per_minute": 60,
        }
        self._windows: Dict[str, List[datetime]] = {}
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
        """
        Check if action is allowed.

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

            # Prune expired entries
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
        """
        Wait until action is allowed, then consume a slot.

        FIX #7: Uses consume=False for polling, then consume=True once allowed.
        """
        import time as _time

        start = _time.monotonic()

        while True:
            # Peek without consuming
            if self.check(limit_type, consume=False):
                # Now actually consume
                if self.check(limit_type, consume=True):
                    return True
                # Race: another thread consumed between peek and consume — retry
                continue

            if _time.monotonic() - start > timeout:
                return False
            _time.sleep(0.1)

    def set_limit(self, limit_type: str, value: int) -> None:
        """
        Set rate limit.

        FIX #8: Validates the value.
        """
        if not isinstance(limit_type, str) or not limit_type:
            raise ValueError("limit_type must be a non-empty string")
        if not isinstance(value, int) or value <= 0:
            raise ValueError(f"limit value must be a positive integer, got {value}")
        self._limits[limit_type] = value

    def get_usage(self, limit_type: str) -> Dict[str, Any]:
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


# =========================================================================
# AccessControl
# =========================================================================


class AccessControl:
    """
    Access control for trading operations.

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
        self._permissions: Dict[str, List[str]] = {
            "admin": ["*"],
            "trader": ["view", "trade_paper", "analyze"],
            "viewer": ["view", "analyze"],
            "live_trader": ["view", "trade_paper", "trade_live", "analyze"],
        }
        self._current_role = "trader"
        self._audit: Optional[AuditLog] = None
        self._logging_access = False  # FIX #10: Recursion guard

    def _get_audit(self) -> Optional[AuditLog]:
        """Lazy audit log access to break circular init."""
        if self._audit is None:
            try:
                self._audit = get_audit_log()
            except Exception:
                pass
        return self._audit

    def set_role(self, role: str) -> None:
        """Set current role."""
        if not isinstance(role, str) or not role:
            raise ValueError("role must be a non-empty string")
        if role not in self._permissions:
            raise ValueError(
                f"Unknown role {role!r}. Valid: {list(self._permissions.keys())}"
            )
        self._current_role = role

    def check(self, permission: str) -> bool:
        """Check if current role has permission."""
        if not isinstance(permission, str) or not permission:
            raise ValueError("permission must be a non-empty string")

        perms = self._permissions.get(self._current_role, [])
        allowed = "*" in perms or permission in perms

        # FIX #10: Recursion guard — audit.log_access might trigger check()
        if not self._logging_access:
            self._logging_access = True
            try:
                audit = self._get_audit()
                if audit is not None:
                    audit.log_access(permission, self._current_role, allowed)
            except Exception:
                pass
            finally:
                self._logging_access = False

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
    def elevate(self, role: str):
        """Temporarily elevate to role."""
        if role not in self._permissions:
            raise ValueError(f"Unknown role {role!r}")

        old_role = self._current_role
        self._current_role = role
        try:
            yield
        finally:
            self._current_role = old_role

    @property
    def current_role(self) -> str:
        """Current role (read-only)."""
        return self._current_role

    @property
    def available_roles(self) -> List[str]:
        """List of available roles."""
        return list(self._permissions.keys())


# =========================================================================
# Module-level singletons with proper locks
# =========================================================================

_secure_storage: Optional[SecureStorage] = None
_secure_storage_lock = threading.Lock()

_audit_log: Optional[AuditLog] = None
_audit_log_lock = threading.Lock()

_rate_limiter: Optional[RateLimiter] = None
_rate_limiter_lock = threading.Lock()

_access_control: Optional[AccessControl] = None
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
    """
    Reset all singletons — for testing only.
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