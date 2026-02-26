from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _run_step(name: str, cmd: list[str]) -> dict[str, Any]:
    started = time.monotonic()
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        elapsed = time.monotonic() - started
        return {
            "name": name,
            "command": cmd,
            "exit_code": int(proc.returncode),
            "duration_seconds": round(float(elapsed), 3),
            "ok": proc.returncode == 0,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
        }
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        elapsed = time.monotonic() - started
        return {
            "name": name,
            "command": cmd,
            "exit_code": 2,
            "duration_seconds": round(float(elapsed), 3),
            "ok": False,
            "stdout": "",
            "stderr": str(exc),
        }


def _is_runtime_artifact_path(path: str) -> bool:
    normalized = str(path or "").strip().replace("\\", "/")
    if not normalized:
        return False
    if normalized.startswith("venv/"):
        return True
    if normalized.startswith("__pycache__/") or "/__pycache__/" in normalized:
        return True
    if normalized.endswith((".pyc", ".pyo", ".pyd")):
        return True
    if normalized.endswith((".db", ".db-shm", ".db-wal")):
        return True
    return False


def _tracked_runtime_artifacts_step(repo_root: Path) -> dict[str, Any]:
    started = time.monotonic()
    cmd = ["git", "ls-files"]
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        elapsed = time.monotonic() - started
        if proc.returncode != 0:
            return {
                "name": "tracked_runtime_artifacts",
                "command": cmd,
                "exit_code": int(proc.returncode),
                "duration_seconds": round(float(elapsed), 3),
                "ok": False,
                "stdout": proc.stdout,
                "stderr": proc.stderr,
            }

        tracked = [
            line.strip()
            for line in proc.stdout.splitlines()
            if _is_runtime_artifact_path(line)
        ]
        preview_limit = 200
        preview = tracked[:preview_limit]
        stdout = "\n".join(preview)
        if len(tracked) > preview_limit:
            stdout += (
                ("\n" if stdout else "")
                + f"... and {len(tracked) - preview_limit} more"
            )
        stderr = ""
        if tracked:
            stderr = (
                "Tracked runtime artifacts found. Remove from index with "
                "`git rm -r --cached -- <paths>` and commit."
            )
        return {
            "name": "tracked_runtime_artifacts",
            "command": cmd,
            "exit_code": 0 if not tracked else 1,
            "duration_seconds": round(float(elapsed), 3),
            "ok": not tracked,
            "stdout": stdout,
            "stderr": stderr,
        }
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        elapsed = time.monotonic() - started
        return {
            "name": "tracked_runtime_artifacts",
            "command": cmd,
            "exit_code": 2,
            "duration_seconds": round(float(elapsed), 3),
            "ok": False,
            "stdout": "",
            "stderr": str(exc),
        }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run deployment preflight checks (health/doctor/typecheck/observability)"
    )
    parser.add_argument(
        "--profile",
        default="full",
        choices=["full", "quick"],
        help=(
            "Preflight profile. "
            "'full' runs all default gates; 'quick' keeps core runtime gates "
            "(artifact, health, doctor, typecheck, observability) and skips long checks."
        ),
    )
    parser.add_argument("--skip-lint", action="store_true", help="Skip Ruff lint gate")
    parser.add_argument("--skip-tests", action="store_true", help="Skip pytest strict gate")
    parser.add_argument("--skip-health", action="store_true", help="Skip health strict check")
    parser.add_argument("--skip-doctor", action="store_true", help="Skip doctor strict check")
    parser.add_argument("--skip-typecheck", action="store_true", help="Skip typecheck gate")
    parser.add_argument(
        "--skip-artifact-guard",
        action="store_true",
        help="Skip tracked runtime artifact gate",
    )
    parser.add_argument(
        "--skip-regulatory",
        action="store_true",
        help="Skip regulated institutional-readiness gate",
    )
    parser.add_argument(
        "--skip-ha-dr",
        action="store_true",
        help="Skip HA/DR failover drill",
    )
    parser.add_argument(
        "--ha-dr-backend",
        default="sqlite",
        choices=["sqlite", "file"],
        help="Lease backend used by HA/DR drill",
    )
    parser.add_argument(
        "--ha-dr-ttl-seconds",
        type=float,
        default=5.0,
        help="Lease TTL used by HA/DR drill",
    )
    parser.add_argument(
        "--ha-dr-stale-takeover",
        action="store_true",
        help="Use stale-expiry takeover path in HA/DR drill",
    )
    parser.add_argument(
        "--allow-test-warnings",
        action="store_true",
        help="Allow pytest warnings (default is warnings-as-errors)",
    )
    parser.add_argument(
        "--pytest-maxfail",
        type=int,
        default=1,
        help="Max failures before pytest exits",
    )
    parser.add_argument(
        "--observability-url",
        default="",
        help="Optional base URL to run observability endpoint probes",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Optional output JSON path",
    )
    args = parser.parse_args()

    if str(args.profile).strip().lower() == "quick":
        # Keep core runtime gates, skip the slower/operationally heavy checks.
        args.skip_lint = True
        args.skip_tests = True
        args.skip_regulatory = True
        args.skip_ha_dr = True

    repo_root = Path(__file__).resolve().parent.parent
    py = sys.executable
    steps: list[dict[str, Any]] = []

    if not args.skip_artifact_guard:
        steps.append(_tracked_runtime_artifacts_step(repo_root))
    if not args.skip_lint:
        steps.append(
            _run_step(
                "ruff_lint",
                [py, "-m", "ruff", "check", "."],
            )
        )
    if not args.skip_tests:
        pytest_cmd = [py, "-m", "pytest", "-q", f"--maxfail={max(1, int(args.pytest_maxfail))}"]
        if not args.allow_test_warnings:
            pytest_cmd.extend(["-W", "error"])
        steps.append(_run_step("pytest_strict", pytest_cmd))
    if not args.skip_health:
        steps.append(
            _run_step(
                "health_strict",
                [py, str(repo_root / "main.py"), "--health", "--health-strict"],
            )
        )
    if not args.skip_doctor:
        steps.append(
            _run_step(
                "doctor_strict",
                [py, str(repo_root / "main.py"), "--doctor", "--doctor-strict"],
            )
        )
    if not args.skip_typecheck:
        steps.append(
            _run_step(
                "typecheck_gate",
                [py, str(repo_root / "scripts/typecheck_gate.py")],
            )
        )
        steps.append(
            _run_step(
                "typecheck_strict_gate",
                [py, str(repo_root / "scripts/typecheck_strict_gate.py")],
            )
        )
        steps.append(
            _run_step(
                "exception_policy_gate",
                [py, str(repo_root / "scripts/exception_policy_gate.py")],
            )
        )
        steps.append(
            _run_step(
                "module_size_gate",
                [py, str(repo_root / "scripts/module_size_gate.py")],
            )
        )
        steps.append(
            _run_step(
                "feature_score_gate",
                [
                    py,
                    str(repo_root / "scripts/feature_score_gate.py"),
                    "--min-overall",
                    "9.5",
                    "--min-feature",
                    "9.5",
                ],
            )
        )
    if not args.skip_regulatory:
        steps.append(
            _run_step(
                "regulatory_readiness",
                [py, str(repo_root / "scripts/regulatory_readiness.py")],
            )
        )
    if not args.skip_ha_dr:
        steps.append(
            _run_step(
                "ha_dr_drill",
                [
                    py,
                    str(repo_root / "scripts/ha_dr_drill.py"),
                    "--backend",
                    str(args.ha_dr_backend),
                    "--ttl-seconds",
                    str(float(args.ha_dr_ttl_seconds)),
                    "--lease-path",
                    str(repo_root / "data_storage" / "ha_dr_preflight_lease.db"),
                ]
                + (["--stale-takeover"] if bool(args.ha_dr_stale_takeover) else []),
            )
        )
    if str(args.observability_url or "").strip():
        steps.append(
            _run_step(
                "observability_probe",
                [
                    py,
                    str(repo_root / "scripts/observability_probe.py"),
                    "--base-url",
                    str(args.observability_url).strip(),
                ],
            )
        )

    failed = [step for step in steps if not step.get("ok", False)]
    report: dict[str, Any] = {
        "status": "pass" if not failed else "fail",
        "generated_at": _utc_now_iso(),
        "profile": str(args.profile),
        "steps": steps,
        "failed_steps": [step["name"] for step in failed],
    }

    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    print(rendered)

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(rendered + "\n", encoding="utf-8")
        print(f"preflight report written: {out}")

    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
