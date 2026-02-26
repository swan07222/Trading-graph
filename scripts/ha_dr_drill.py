from __future__ import annotations

import argparse
import json
import os
import sqlite3
import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _atomic_write_text(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(payload, encoding="utf-8")
    os.replace(str(tmp), str(path))


@dataclass
class LeaseResult:
    ok: bool
    record: dict[str, Any] | None
    reason: str = ""


class _LeaseClient:
    def acquire(self, owner_id: str, ttl_seconds: float, metadata: dict[str, Any] | None) -> LeaseResult:
        raise NotImplementedError

    def refresh(self, owner_id: str, ttl_seconds: float, metadata: dict[str, Any] | None) -> LeaseResult:
        raise NotImplementedError

    def release(self, owner_id: str, metadata: dict[str, Any] | None) -> LeaseResult:
        raise NotImplementedError


class _FileLeaseClient(_LeaseClient):
    def __init__(self, *, cluster: str, path: Path) -> None:
        self._cluster = str(cluster).strip() or "analysis_cluster"
        self._path = Path(path)

    def _read_record(self) -> dict[str, Any] | None:
        if not self._path.exists():
            return None
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
        except Exception:
            return None
        if not isinstance(raw, dict):
            return None
        if str(raw.get("cluster", "")) != self._cluster:
            return None
        return raw

    def _write_record(self, row: dict[str, Any]) -> None:
        _atomic_write_text(
            self._path,
            json.dumps(row, ensure_ascii=True, separators=(",", ":")) + "\n",
        )

    @staticmethod
    def _now() -> float:
        return float(time.time())

    def acquire(self, owner_id: str, ttl_seconds: float, metadata: dict[str, Any] | None) -> LeaseResult:
        now = self._now()
        ttl = max(1.0, float(ttl_seconds))
        row = self._read_record()
        generation = 0
        if isinstance(row, dict):
            generation = int(row.get("generation", 0) or 0)
            expires_at = float(row.get("expires_at", 0.0) or 0.0)
            active_owner = str(row.get("owner_id", "") or "")
            if expires_at > now and active_owner and active_owner != owner_id:
                return LeaseResult(ok=False, record=row, reason="held_by_other")
        generation += 1
        out = {
            "cluster": self._cluster,
            "owner_id": str(owner_id),
            "generation": int(generation),
            "expires_at": float(now + ttl),
            "metadata": dict(metadata or {}),
            "updated_at": float(now),
            "backend": "file",
        }
        self._write_record(out)
        return LeaseResult(ok=True, record=out)

    def refresh(self, owner_id: str, ttl_seconds: float, metadata: dict[str, Any] | None) -> LeaseResult:
        now = self._now()
        ttl = max(1.0, float(ttl_seconds))
        row = self._read_record()
        if not isinstance(row, dict):
            return LeaseResult(ok=False, record=None, reason="lease_missing")
        if str(row.get("owner_id", "") or "") != str(owner_id):
            return LeaseResult(ok=False, record=row, reason="not_owner")
        expires_at = float(row.get("expires_at", 0.0) or 0.0)
        if expires_at <= now:
            return LeaseResult(ok=False, record=row, reason="expired")
        row["expires_at"] = float(now + ttl)
        row["updated_at"] = float(now)
        row["metadata"] = dict(metadata or {})
        self._write_record(row)
        return LeaseResult(ok=True, record=row)

    def release(self, owner_id: str, metadata: dict[str, Any] | None) -> LeaseResult:
        _ = metadata
        row = self._read_record()
        if not isinstance(row, dict):
            return LeaseResult(ok=False, record=None, reason="lease_missing")
        if str(row.get("owner_id", "") or "") != str(owner_id):
            return LeaseResult(ok=False, record=row, reason="not_owner")
        try:
            self._path.unlink(missing_ok=True)
        except OSError:
            return LeaseResult(ok=False, record=row, reason="release_failed")
        return LeaseResult(ok=True, record=row)


class _SqliteLeaseClient(_LeaseClient):
    def __init__(self, *, cluster: str, path: Path) -> None:
        self._cluster = str(cluster).strip() or "analysis_cluster"
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._path), timeout=5.0)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS runtime_lease (
                    cluster TEXT PRIMARY KEY,
                    owner_id TEXT NOT NULL,
                    generation INTEGER NOT NULL,
                    expires_at REAL NOT NULL,
                    metadata TEXT NOT NULL,
                    updated_at REAL NOT NULL
                )
                """
            )
            conn.commit()

    @staticmethod
    def _now() -> float:
        return float(time.time())

    def _get_row(self, conn: sqlite3.Connection) -> dict[str, Any] | None:
        cur = conn.execute(
            "SELECT cluster, owner_id, generation, expires_at, metadata, updated_at "
            "FROM runtime_lease WHERE cluster = ?",
            (self._cluster,),
        )
        row = cur.fetchone()
        if row is None:
            return None
        metadata: dict[str, Any] = {}
        try:
            raw_meta = json.loads(str(row[4] or "{}"))
            if isinstance(raw_meta, dict):
                metadata = raw_meta
        except Exception:
            metadata = {}
        return {
            "cluster": str(row[0]),
            "owner_id": str(row[1]),
            "generation": int(row[2] or 0),
            "expires_at": float(row[3] or 0.0),
            "metadata": metadata,
            "updated_at": float(row[5] or 0.0),
            "backend": "sqlite",
        }

    def acquire(self, owner_id: str, ttl_seconds: float, metadata: dict[str, Any] | None) -> LeaseResult:
        now = self._now()
        ttl = max(1.0, float(ttl_seconds))
        owner = str(owner_id)
        meta_payload = json.dumps(dict(metadata or {}), ensure_ascii=True, separators=(",", ":"))
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = self._get_row(conn)
            generation = 0
            if isinstance(row, dict):
                generation = int(row.get("generation", 0) or 0)
                expires_at = float(row.get("expires_at", 0.0) or 0.0)
                active_owner = str(row.get("owner_id", "") or "")
                if expires_at > now and active_owner and active_owner != owner:
                    conn.rollback()
                    return LeaseResult(ok=False, record=row, reason="held_by_other")
            generation += 1
            conn.execute(
                """
                INSERT INTO runtime_lease(cluster, owner_id, generation, expires_at, metadata, updated_at)
                VALUES(?, ?, ?, ?, ?, ?)
                ON CONFLICT(cluster) DO UPDATE SET
                    owner_id = excluded.owner_id,
                    generation = excluded.generation,
                    expires_at = excluded.expires_at,
                    metadata = excluded.metadata,
                    updated_at = excluded.updated_at
                """,
                (
                    self._cluster,
                    owner,
                    int(generation),
                    float(now + ttl),
                    meta_payload,
                    float(now),
                ),
            )
            conn.commit()
        record = {
            "cluster": self._cluster,
            "owner_id": owner,
            "generation": int(generation),
            "expires_at": float(now + ttl),
            "metadata": dict(metadata or {}),
            "updated_at": float(now),
            "backend": "sqlite",
        }
        return LeaseResult(ok=True, record=record)

    def refresh(self, owner_id: str, ttl_seconds: float, metadata: dict[str, Any] | None) -> LeaseResult:
        now = self._now()
        ttl = max(1.0, float(ttl_seconds))
        owner = str(owner_id)
        meta_payload = json.dumps(dict(metadata or {}), ensure_ascii=True, separators=(",", ":"))
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = self._get_row(conn)
            if not isinstance(row, dict):
                conn.rollback()
                return LeaseResult(ok=False, record=None, reason="lease_missing")
            if str(row.get("owner_id", "") or "") != owner:
                conn.rollback()
                return LeaseResult(ok=False, record=row, reason="not_owner")
            if float(row.get("expires_at", 0.0) or 0.0) <= now:
                conn.rollback()
                return LeaseResult(ok=False, record=row, reason="expired")
            conn.execute(
                "UPDATE runtime_lease SET expires_at = ?, metadata = ?, updated_at = ? WHERE cluster = ?",
                (float(now + ttl), meta_payload, float(now), self._cluster),
            )
            conn.commit()
        row["expires_at"] = float(now + ttl)
        row["metadata"] = dict(metadata or {})
        row["updated_at"] = float(now)
        return LeaseResult(ok=True, record=row)

    def release(self, owner_id: str, metadata: dict[str, Any] | None) -> LeaseResult:
        _ = metadata
        owner = str(owner_id)
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = self._get_row(conn)
            if not isinstance(row, dict):
                conn.rollback()
                return LeaseResult(ok=False, record=None, reason="lease_missing")
            if str(row.get("owner_id", "") or "") != owner:
                conn.rollback()
                return LeaseResult(ok=False, record=row, reason="not_owner")
            conn.execute(
                "DELETE FROM runtime_lease WHERE cluster = ?",
                (self._cluster,),
            )
            conn.commit()
        return LeaseResult(ok=True, record=row)


def _create_lease_client(*, backend: str, cluster: str, path: Path) -> _LeaseClient:
    backend_name = str(backend or "sqlite").strip().lower()
    if backend_name == "sqlite":
        return _SqliteLeaseClient(cluster=cluster, path=path)
    if backend_name == "file":
        return _FileLeaseClient(cluster=cluster, path=path)
    raise ValueError(f"Unsupported lease backend: {backend}")


def run_ha_dr_drill(
    *,
    backend: str,
    cluster: str,
    lease_path: Path,
    ttl_seconds: float,
    stale_takeover: bool,
) -> dict[str, Any]:
    backend_name = str(backend or "sqlite").strip().lower()
    ttl = max(1.0, float(ttl_seconds))
    client = _create_lease_client(
        backend=backend_name,
        cluster=str(cluster or "analysis_cluster"),
        path=Path(lease_path),
    )

    owner_a = f"drill-a-{uuid.uuid4().hex[:10]}"
    owner_b = f"drill-b-{uuid.uuid4().hex[:10]}"
    steps: list[dict[str, Any]] = []
    failed: list[str] = []

    def _step(name: str, ok: bool, detail: dict[str, Any]) -> None:
        steps.append(
            {
                "name": str(name),
                "ok": bool(ok),
                "detail": detail,
                "ts": _utc_now_iso(),
            }
        )
        if not ok:
            failed.append(str(name))

    a1 = client.acquire(owner_id=owner_a, ttl_seconds=ttl, metadata={"phase": "initial_acquire"})
    _step("owner_a_acquire", bool(a1.ok), {"record": a1.record, "owner": owner_a})

    b1 = client.acquire(owner_id=owner_b, ttl_seconds=ttl, metadata={"phase": "contended_acquire"})
    _step("owner_b_blocked_while_a_active", not bool(b1.ok), {"record": b1.record, "owner": owner_b})

    a_refresh = client.refresh(owner_id=owner_a, ttl_seconds=ttl, metadata={"phase": "heartbeat"})
    _step("owner_a_refresh", bool(a_refresh.ok), {"record": a_refresh.record, "owner": owner_a})

    if stale_takeover:
        time.sleep(ttl + 0.2)
    else:
        client.release(owner_id=owner_a, metadata={"phase": "handoff"})

    b2 = client.acquire(owner_id=owner_b, ttl_seconds=ttl, metadata={"phase": "takeover"})
    _step(
        "owner_b_takeover",
        bool(b2.ok),
        {"record": b2.record, "owner": owner_b, "stale_takeover": bool(stale_takeover)},
    )

    a_refresh_after = client.refresh(
        owner_id=owner_a,
        ttl_seconds=ttl,
        metadata={"phase": "stale_owner_refresh"},
    )
    _step("owner_a_fenced_after_takeover", not bool(a_refresh_after.ok), {"record": a_refresh_after.record, "owner": owner_a})

    a_acquire_after = client.acquire(
        owner_id=owner_a,
        ttl_seconds=ttl,
        metadata={"phase": "reacquire_while_b_active"},
    )
    _step("owner_a_blocked_while_b_active", not bool(a_acquire_after.ok), {"record": a_acquire_after.record, "owner": owner_a})

    gen_a = int((a1.record or {}).get("generation", 0) or 0)
    gen_b = int((b2.record or {}).get("generation", 0) or 0)
    if stale_takeover:
        _step(
            "fencing_generation_increments",
            gen_b > gen_a,
            {"generation_initial": gen_a, "generation_takeover": gen_b},
        )
    else:
        _step(
            "handoff_generation_recorded",
            gen_b >= 1,
            {"generation_initial": gen_a, "generation_takeover": gen_b},
        )

    try:
        client.release(owner_id=owner_b, metadata={"phase": "cleanup"})
    except Exception:
        pass

    status = "pass" if not failed else "fail"
    return {
        "status": status,
        "backend": backend_name,
        "cluster": str(cluster),
        "lease_path": str(lease_path),
        "ttl_seconds": ttl,
        "stale_takeover": bool(stale_takeover),
        "failed_steps": failed,
        "steps": steps,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="HA/DR failover drill using runtime lease ownership fencing"
    )
    parser.add_argument(
        "--backend",
        default="sqlite",
        choices=["sqlite", "file"],
        help="Lease backend",
    )
    parser.add_argument(
        "--cluster",
        default="analysis_cluster",
        help="Lease cluster name",
    )
    parser.add_argument(
        "--lease-path",
        default="data_storage/ha_dr_drill_lease.db",
        help="Path to lease file/db",
    )
    parser.add_argument(
        "--ttl-seconds",
        type=float,
        default=5.0,
        help="Lease TTL for drill",
    )
    parser.add_argument(
        "--stale-takeover",
        action="store_true",
        help="Use stale-expiry takeover instead of explicit release handoff",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Optional output JSON path",
    )
    args = parser.parse_args()

    report = run_ha_dr_drill(
        backend=args.backend,
        cluster=args.cluster,
        lease_path=Path(args.lease_path),
        ttl_seconds=float(args.ttl_seconds),
        stale_takeover=bool(args.stale_takeover),
    )
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    print(rendered)

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(rendered + "\n", encoding="utf-8")
        print(f"ha/dr report written: {out}")

    return 0 if report.get("status") == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
