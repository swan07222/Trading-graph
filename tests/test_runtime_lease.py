import sqlite3
import threading
import time
from types import SimpleNamespace

from trading.executor import ExecutionEngine
from trading.runtime_lease import FileRuntimeLeaseClient, create_runtime_lease_client


def _mk_engine(tmp_path, lease_id: str) -> ExecutionEngine:
    eng = ExecutionEngine.__new__(ExecutionEngine)
    eng.mode = SimpleNamespace(value="simulation")
    eng._runtime_lease_enabled = True
    eng._runtime_lease_ttl_seconds = 30.0
    eng._runtime_lease_id = lease_id
    eng._runtime_lease_path = tmp_path / "execution_runtime_lease.json"
    eng._runtime_lease_owner_hint = None
    return eng


def test_runtime_lease_exclusive_owner(tmp_path) -> None:
    eng1 = _mk_engine(tmp_path, "owner-1")
    eng2 = _mk_engine(tmp_path, "owner-2")

    assert eng1._acquire_runtime_lease() is True
    assert eng2._acquire_runtime_lease() is False
    assert isinstance(eng2._runtime_lease_owner_hint, dict)


def test_runtime_lease_refresh_and_release(tmp_path) -> None:
    eng = _mk_engine(tmp_path, "owner-1")
    assert eng._acquire_runtime_lease() is True
    assert eng._refresh_runtime_lease() is True
    assert eng._runtime_lease_path.exists()

    eng._release_runtime_lease()
    assert (not eng._runtime_lease_path.exists()) or eng._runtime_lease_path.read_text(encoding="utf-8")


def test_sqlite_runtime_lease_exclusive_owner(tmp_path) -> None:
    db_path = tmp_path / "lease.db"
    c1 = create_runtime_lease_client("sqlite", "cluster-a", db_path)
    c2 = create_runtime_lease_client("sqlite", "cluster-a", db_path)

    r1 = c1.acquire(owner_id="node-a", ttl_seconds=30.0, metadata={"n": 1})
    assert r1.ok is True
    assert int((r1.record or {}).get("generation", 0)) == 1

    r2 = c2.acquire(owner_id="node-b", ttl_seconds=30.0, metadata={"n": 2})
    assert r2.ok is False
    assert str((r2.record or {}).get("owner_id", "")) == "node-a"


def test_sqlite_runtime_lease_stale_takeover_increments_fencing_token(tmp_path) -> None:
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


def test_file_runtime_lease_contention_has_single_winner(tmp_path, monkeypatch) -> None:
    lease_path = tmp_path / "lease.json"
    c1 = FileRuntimeLeaseClient(path=lease_path, cluster="cluster-c")
    c2 = FileRuntimeLeaseClient(path=lease_path, cluster="cluster-c")

    start_barrier = threading.Barrier(3)
    race_barrier = threading.Barrier(2)
    original_row = FileRuntimeLeaseClient._row

    def _patched_row(self, *args, **kwargs):
        row = original_row(self, *args, **kwargs)
        # Force overlap in the critical section to verify acquire is serialized.
        try:
            race_barrier.wait(timeout=0.2)
        except Exception:
            pass
        return row

    monkeypatch.setattr(FileRuntimeLeaseClient, "_row", _patched_row, raising=True)
    out: dict[str, bool] = {}

    def _run(owner_id: str, client: FileRuntimeLeaseClient) -> None:
        start_barrier.wait(timeout=2.0)
        out[owner_id] = bool(client.acquire(owner_id=owner_id, ttl_seconds=30.0).ok)

    t1 = threading.Thread(target=_run, args=("node-a", c1))
    t2 = threading.Thread(target=_run, args=("node-b", c2))
    t1.start()
    t2.start()
    start_barrier.wait(timeout=2.0)
    t1.join(timeout=2.0)
    t2.join(timeout=2.0)

    assert t1.is_alive() is False
    assert t2.is_alive() is False
    assert sum(1 for ok in out.values() if ok) == 1

    row = c1.read() or {}
    assert str(row.get("owner_id", "")) in {"node-a", "node-b"}
