from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from utils.atomic_io import atomic_write_json, read_json
from utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class LeaseResult:
    ok: bool
    record: dict[str, Any] | None = None


class RuntimeLeaseClient:
    """Lease client interface for single-writer execution ownership."""

    def acquire(
        self,
        owner_id: str,
        ttl_seconds: float,
        metadata: dict[str, Any] | None = None,
    ) -> LeaseResult:
        raise NotImplementedError

    def refresh(
        self,
        owner_id: str,
        ttl_seconds: float,
        metadata: dict[str, Any] | None = None,
    ) -> LeaseResult:
        raise NotImplementedError

    def release(
        self,
        owner_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        raise NotImplementedError

    def read(self) -> dict[str, Any] | None:
        raise NotImplementedError


class FileRuntimeLeaseClient(RuntimeLeaseClient):
    """
    JSON file lease backend.

    Works for single-host/mounted-volume scenarios.
    """

    def __init__(self, path: Path, cluster: str):
        self._path = Path(path)
        self._cluster = str(cluster or "execution_engine")
        self._lock_path = self._path.with_suffix(self._path.suffix + ".lock")
        self._lock_timeout_s = 5.0
        self._lock_stale_s = 30.0

    def _read(self) -> dict[str, Any]:
        if not self._path.exists():
            return {}
        raw = read_json(self._path)
        return raw if isinstance(raw, dict) else {}

    @contextmanager
    def _file_guard(self):
        """
        Best-effort inter-process lock for file-backed lease updates.

        Uses a sidecar lock file created with O_EXCL so read-modify-write
        cycles are serialized across processes.
        """
        self._path.parent.mkdir(parents=True, exist_ok=True)
        deadline = time.monotonic() + float(self._lock_timeout_s)

        while True:
            try:
                fd = os.open(
                    str(self._lock_path),
                    os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                )
                try:
                    payload = f"{os.getpid()} {time.time():.6f}".encode("ascii")
                    os.write(fd, payload)
                finally:
                    os.close(fd)
                break
            except FileExistsError as err:
                try:
                    age = time.time() - float(self._lock_path.stat().st_mtime)
                    if age > float(self._lock_stale_s):
                        self._lock_path.unlink(missing_ok=True)
                        continue
                except FileNotFoundError:
                    continue
                except Exception:
                    pass

                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"lease lock timeout for {self._lock_path.name}"
                    ) from err
                time.sleep(0.01)

        try:
            yield
        finally:
            try:
                self._lock_path.unlink(missing_ok=True)
            except Exception as e:
                log.debug("File lease lock cleanup failed (%s): %s", self._lock_path, e)

    def _is_stale(self, row: dict[str, Any], now_ts: float) -> bool:
        exp = float(row.get("lease_expires_ts", 0.0) or 0.0)
        return exp <= 0.0 or now_ts > exp

    def _row(
        self,
        owner_id: str,
        generation: int,
        ttl_seconds: float,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        now_ts = float(time.time())
        return {
            "cluster": self._cluster,
            "owner_id": str(owner_id),
            "generation": int(generation),
            "heartbeat_ts": now_ts,
            "lease_expires_ts": now_ts + max(5.0, float(ttl_seconds)),
            "acquired_ts": now_ts,
            "backend": "file",
            "metadata": dict(metadata or {}),
        }

    def acquire(
        self,
        owner_id: str,
        ttl_seconds: float,
        metadata: dict[str, Any] | None = None,
    ) -> LeaseResult:
        try:
            with self._file_guard():
                cur = self._read()
                now_ts = float(time.time())
                cur_owner = str(cur.get("owner_id", "") or "")
                if cur_owner and cur_owner != owner_id and not self._is_stale(cur, now_ts):
                    return LeaseResult(ok=False, record=cur)

                gen = int(cur.get("generation", 0) or 0) + 1
                row = self._row(
                    owner_id=owner_id,
                    generation=gen,
                    ttl_seconds=ttl_seconds,
                    metadata=metadata,
                )
                atomic_write_json(self._path, row, indent=2)
                out = self._read()
                ok = (
                    str(out.get("owner_id", "") or "") == str(owner_id)
                    and int(out.get("generation", 0) or 0) == gen
                )
                return LeaseResult(ok=ok, record=out)
        except Exception as e:
            log.warning("File lease acquire failed: %s", e)
            return LeaseResult(ok=False, record=None)

    def refresh(
        self,
        owner_id: str,
        ttl_seconds: float,
        metadata: dict[str, Any] | None = None,
    ) -> LeaseResult:
        try:
            with self._file_guard():
                cur = self._read()
                cur_owner = str(cur.get("owner_id", "") or "")
                if cur_owner != str(owner_id):
                    return LeaseResult(ok=False, record=cur)

                gen = int(cur.get("generation", 0) or 0)
                row = self._row(
                    owner_id=owner_id,
                    generation=gen,
                    ttl_seconds=ttl_seconds,
                    metadata=metadata,
                )
                # preserve original acquire ts on heartbeat updates
                acq = float(cur.get("acquired_ts", 0.0) or 0.0)
                if acq > 0.0:
                    row["acquired_ts"] = acq
                atomic_write_json(self._path, row, indent=2)
                return LeaseResult(ok=True, record=self._read())
        except Exception as e:
            log.warning("File lease refresh failed: %s", e)
            return LeaseResult(ok=False, record=None)

    def release(
        self,
        owner_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        try:
            with self._file_guard():
                cur = self._read()
                if str(cur.get("owner_id", "") or "") != str(owner_id):
                    return
                cur["owner_id"] = ""
                cur["heartbeat_ts"] = float(time.time())
                cur["lease_expires_ts"] = 0.0
                cur["released_ts"] = float(time.time())
                cur["released_by"] = str(owner_id)
                if metadata:
                    cur["metadata"] = dict(metadata)
                atomic_write_json(self._path, cur, indent=2)
        except Exception as e:
            log.debug("File lease release failed for owner=%s: %s", owner_id, e)

    def read(self) -> dict[str, Any] | None:
        return self._read()


class SqliteRuntimeLeaseClient(RuntimeLeaseClient):
    """
    SQLite lease backend with transactional updates + fencing token.

    This is safer for active/standby failover than plain file leases.
    """

    def __init__(self, db_path: Path, cluster: str):
        self._db_path = Path(db_path)
        self._cluster = str(cluster or "execution_engine")
        self._lock = threading.RLock()
        self._schema_ready = False

    def _conn(self) -> sqlite3.Connection:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        c = sqlite3.connect(str(self._db_path), timeout=30, isolation_level=None)
        c.row_factory = sqlite3.Row
        c.execute("PRAGMA journal_mode=WAL")
        c.execute("PRAGMA synchronous=NORMAL")
        return c

    def _ensure_schema(self) -> None:
        if self._schema_ready:
            return
        with self._lock:
            if self._schema_ready:
                return
            conn = self._conn()
            try:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS runtime_leases (
                        cluster TEXT PRIMARY KEY,
                        owner_id TEXT NOT NULL,
                        generation INTEGER NOT NULL DEFAULT 0,
                        heartbeat_ts REAL NOT NULL DEFAULT 0,
                        lease_expires_ts REAL NOT NULL DEFAULT 0,
                        acquired_ts REAL NOT NULL DEFAULT 0,
                        metadata_json TEXT NOT NULL DEFAULT '{}'
                    )
                    """
                )
                self._schema_ready = True
            finally:
                conn.close()

    @staticmethod
    def _row_to_dict(row: sqlite3.Row | None, backend: str) -> dict[str, Any]:
        if row is None:
            return {}
        out = dict(row)
        out["backend"] = backend
        meta_raw = str(out.get("metadata_json", "{}") or "{}")
        try:
            out["metadata"] = json.loads(meta_raw)
        except Exception as e:
            log.debug("Lease metadata decode failed (backend=%s): %s", backend, e)
            out["metadata"] = {}
        return out

    def acquire(
        self,
        owner_id: str,
        ttl_seconds: float,
        metadata: dict[str, Any] | None = None,
    ) -> LeaseResult:
        self._ensure_schema()
        conn = self._conn()
        now_ts = float(time.time())
        ttl = max(5.0, float(ttl_seconds))
        owner = str(owner_id)
        meta = json.dumps(dict(metadata or {}), ensure_ascii=True)

        try:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT * FROM runtime_leases WHERE cluster = ?",
                (self._cluster,),
            ).fetchone()

            if row is None:
                generation = 1
                conn.execute(
                    """
                    INSERT INTO runtime_leases
                    (cluster, owner_id, generation, heartbeat_ts, lease_expires_ts, acquired_ts, metadata_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (self._cluster, owner, generation, now_ts, now_ts + ttl, now_ts, meta),
                )
                conn.execute("COMMIT")
                out = self.read() or {}
                return LeaseResult(ok=True, record=out)

            current = self._row_to_dict(row, backend="sqlite")
            cur_owner = str(current.get("owner_id", "") or "")
            cur_exp = float(current.get("lease_expires_ts", 0.0) or 0.0)
            stale = cur_exp <= 0.0 or now_ts > cur_exp

            if cur_owner and cur_owner != owner and not stale:
                conn.execute("ROLLBACK")
                return LeaseResult(ok=False, record=current)

            generation = int(current.get("generation", 0) or 0) + 1
            conn.execute(
                """
                UPDATE runtime_leases
                SET owner_id = ?, generation = ?, heartbeat_ts = ?, lease_expires_ts = ?, acquired_ts = ?, metadata_json = ?
                WHERE cluster = ?
                """,
                (owner, generation, now_ts, now_ts + ttl, now_ts, meta, self._cluster),
            )
            conn.execute("COMMIT")
            out = self.read() or {}
            ok = str(out.get("owner_id", "") or "") == owner and int(out.get("generation", 0) or 0) == generation
            return LeaseResult(ok=ok, record=out)
        except Exception as e:
            try:
                conn.execute("ROLLBACK")
            except Exception as rollback_e:
                log.debug("SQLite lease acquire rollback failed: %s", rollback_e)
            log.warning("SQLite lease acquire failed: %s", e)
            return LeaseResult(ok=False, record=None)
        finally:
            conn.close()

    def refresh(
        self,
        owner_id: str,
        ttl_seconds: float,
        metadata: dict[str, Any] | None = None,
    ) -> LeaseResult:
        self._ensure_schema()
        conn = self._conn()
        now_ts = float(time.time())
        ttl = max(5.0, float(ttl_seconds))
        owner = str(owner_id)
        meta = json.dumps(dict(metadata or {}), ensure_ascii=True)

        try:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT * FROM runtime_leases WHERE cluster = ?",
                (self._cluster,),
            ).fetchone()
            current = self._row_to_dict(row, backend="sqlite")
            cur_owner = str(current.get("owner_id", "") or "")
            if cur_owner != owner:
                conn.execute("ROLLBACK")
                return LeaseResult(ok=False, record=current)

            conn.execute(
                """
                UPDATE runtime_leases
                SET heartbeat_ts = ?, lease_expires_ts = ?, metadata_json = ?
                WHERE cluster = ? AND owner_id = ?
                """,
                (now_ts, now_ts + ttl, meta, self._cluster, owner),
            )
            conn.execute("COMMIT")
            out = self.read() or {}
            return LeaseResult(ok=True, record=out)
        except Exception as e:
            try:
                conn.execute("ROLLBACK")
            except Exception as rollback_e:
                log.debug("SQLite lease refresh rollback failed: %s", rollback_e)
            log.warning("SQLite lease refresh failed: %s", e)
            return LeaseResult(ok=False, record=None)
        finally:
            conn.close()

    def release(
        self,
        owner_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self._ensure_schema()
        conn = self._conn()
        now_ts = float(time.time())
        owner = str(owner_id)
        release_meta = dict(metadata or {})
        release_meta["released_by"] = owner
        release_meta["released_ts"] = now_ts
        meta = json.dumps(release_meta, ensure_ascii=True)
        try:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                """
                UPDATE runtime_leases
                SET owner_id = '', heartbeat_ts = ?, lease_expires_ts = 0, metadata_json = ?
                WHERE cluster = ? AND owner_id = ?
                """,
                (now_ts, meta, self._cluster, owner),
            )
            conn.execute("COMMIT")
        except Exception as e:
            try:
                conn.execute("ROLLBACK")
            except Exception as rollback_e:
                log.debug("SQLite lease release rollback failed: %s", rollback_e)
            log.warning("SQLite lease release failed for owner=%s: %s", owner, e)
        finally:
            conn.close()

    def read(self) -> dict[str, Any] | None:
        self._ensure_schema()
        conn = self._conn()
        try:
            row = conn.execute(
                "SELECT * FROM runtime_leases WHERE cluster = ?",
                (self._cluster,),
            ).fetchone()
            return self._row_to_dict(row, backend="sqlite")
        except Exception as e:
            log.debug("SQLite lease read failed: %s", e)
            return None
        finally:
            conn.close()


def create_runtime_lease_client(
    backend: str,
    cluster: str,
    path: Path,
) -> RuntimeLeaseClient:
    mode = str(backend or "file").strip().lower()
    if mode == "sqlite":
        return SqliteRuntimeLeaseClient(db_path=Path(path), cluster=cluster)
    return FileRuntimeLeaseClient(path=Path(path), cluster=cluster)
