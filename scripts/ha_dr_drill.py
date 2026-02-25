from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from trading.runtime_lease import create_runtime_lease_client


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def run_ha_dr_drill(
    *,
    backend: str,
    cluster: str,
    lease_path: Path,
    ttl_seconds: float,
    stale_takeover: bool,
) -> dict[str, Any]:
    backend_name = str(backend or "sqlite").strip().lower()
    ttl = max(5.0, float(ttl_seconds))
    client = create_runtime_lease_client(
        backend=backend_name,
        cluster=str(cluster or "execution_engine"),
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
    _step(
        "owner_a_acquire",
        bool(a1.ok),
        {"record": a1.record, "owner": owner_a},
    )

    b1 = client.acquire(owner_id=owner_b, ttl_seconds=ttl, metadata={"phase": "contended_acquire"})
    _step(
        "owner_b_blocked_while_a_active",
        not bool(b1.ok),
        {"record": b1.record, "owner": owner_b},
    )

    a_refresh = client.refresh(owner_id=owner_a, ttl_seconds=ttl, metadata={"phase": "heartbeat"})
    _step(
        "owner_a_refresh",
        bool(a_refresh.ok),
        {"record": a_refresh.record, "owner": owner_a},
    )

    if stale_takeover:
        time.sleep(ttl + 0.5)
    else:
        client.release(owner_id=owner_a, metadata={"phase": "handoff"})

    b2 = client.acquire(owner_id=owner_b, ttl_seconds=ttl, metadata={"phase": "takeover"})
    _step(
        "owner_b_takeover",
        bool(b2.ok),
        {"record": b2.record, "owner": owner_b, "stale_takeover": bool(stale_takeover)},
    )

    a_refresh_after = client.refresh(owner_id=owner_a, ttl_seconds=ttl, metadata={"phase": "stale_owner_refresh"})
    _step(
        "owner_a_fenced_after_takeover",
        not bool(a_refresh_after.ok),
        {"record": a_refresh_after.record, "owner": owner_a},
    )

    a_acquire_after = client.acquire(owner_id=owner_a, ttl_seconds=ttl, metadata={"phase": "reacquire_while_b_active"})
    _step(
        "owner_a_blocked_while_b_active",
        not bool(a_acquire_after.ok),
        {"record": a_acquire_after.record, "owner": owner_a},
    )

    gen_a = int((a1.record or {}).get("generation", 0) or 0)
    gen_b = int((b2.record or {}).get("generation", 0) or 0)
    _step(
        "fencing_generation_increments",
        gen_b > gen_a,
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
        default="execution_engine",
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
