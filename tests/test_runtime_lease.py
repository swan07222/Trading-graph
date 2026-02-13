import sqlite3
import time
from types import SimpleNamespace

from trading.executor import ExecutionEngine
from trading.runtime_lease import create_runtime_lease_client


def _mk_engine(tmp_path, lease_id: str) -> ExecutionEngine:
    eng = ExecutionEngine.__new__(ExecutionEngine)
    eng.mode = SimpleNamespace(value="simulation")
    eng._runtime_lease_enabled = True
    eng._runtime_lease_ttl_seconds = 30.0
    eng._runtime_lease_id = lease_id
    eng._runtime_lease_path = tmp_path / "execution_runtime_lease.json"
    eng._runtime_lease_owner_hint = None
    return eng


def test_runtime_lease_exclusive_owner(tmp_path):
    eng1 = _mk_engine(tmp_path, "owner-1")
    eng2 = _mk_engine(tmp_path, "owner-2")

    assert eng1._acquire_runtime_lease() is True
    assert eng2._acquire_runtime_lease() is False
    assert isinstance(eng2._runtime_lease_owner_hint, dict)


def test_runtime_lease_refresh_and_release(tmp_path):
    eng = _mk_engine(tmp_path, "owner-1")
    assert eng._acquire_runtime_lease() is True
    assert eng._refresh_runtime_lease() is True
    assert eng._runtime_lease_path.exists()

    eng._release_runtime_lease()
    assert (not eng._runtime_lease_path.exists()) or eng._runtime_lease_path.read_text(encoding="utf-8")


def test_sqlite_runtime_lease_exclusive_owner(tmp_path):
    db_path = tmp_path / "lease.db"
    c1 = create_runtime_lease_client("sqlite", "cluster-a", db_path)
    c2 = create_runtime_lease_client("sqlite", "cluster-a", db_path)

    r1 = c1.acquire(owner_id="node-a", ttl_seconds=30.0, metadata={"n": 1})
    assert r1.ok is True
    assert int((r1.record or {}).get("generation", 0)) == 1

    r2 = c2.acquire(owner_id="node-b", ttl_seconds=30.0, metadata={"n": 2})
    assert r2.ok is False
    assert str((r2.record or {}).get("owner_id", "")) == "node-a"


def test_sqlite_runtime_lease_stale_takeover_increments_fencing_token(tmp_path):
    db_path = tmp_path / "lease.db"
    c1 = create_runtime_lease_client("sqlite", "cluster-b", db_path)
    c2 = create_runtime_lease_client("sqlite", "cluster-b", db_path)

    r1 = c1.acquire(owner_id="node-a", ttl_seconds=30.0, metadata={"site": "a"})
    assert r1.ok is True
    assert int((r1.record or {}).get("generation", 0)) == 1

    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute(
            "UPDATE runtime_leases SET lease_expires_ts = ? WHERE cluster = ?",
            (time.time() - 10.0, "cluster-b"),
        )
        conn.commit()
    finally:
        conn.close()

    r2 = c2.acquire(owner_id="node-b", ttl_seconds=30.0, metadata={"site": "b"})
    assert r2.ok is True
    assert str((r2.record or {}).get("owner_id", "")) == "node-b"
    assert int((r2.record or {}).get("generation", 0)) == 2
