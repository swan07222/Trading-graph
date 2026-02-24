from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

from scripts.gate_common import normalize_path
from scripts.typecheck_common import (
    ERROR_RE,
    load_baseline_entries,
    parse_mypy_errors,
    run_mypy,
    save_baseline_entries,
)

DEFAULT_TARGETS: tuple[str, ...] = (
    "main.py",
    "data/feeds.py",
    "data/fetcher.py",
    "data/fetcher_registry.py",
    "models/predictor.py",
    "trading/executor.py",
    "ui/app.py",
    "ui/app_bar_ops.py",
    "ui/app_auto_trade_ops.py",
    "ui/app_lifecycle_ops.py",
    "ui/app_monitoring_ops.py",
    "ui/app_chart_pipeline.py",
    "ui/app_panels.py",
    "ui/background_tasks.py",
    "utils/atomic_io.py",
    "utils/security.py",
    "scripts/release_preflight.py",
    "scripts/typecheck_gate.py",
    "scripts/module_size_gate.py",
    "scripts/exception_policy_gate.py",
)
DEFAULT_FLAGS: tuple[str, ...] = (
    "--strict",
    "--follow-imports=skip",
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Strict typecheck gate with baseline support."
    )
    parser.add_argument(
        "--baseline",
        default=".ci/mypy-strict-baseline.txt",
        help="Path to baseline issue list",
    )
    parser.add_argument(
        "--target",
        action="append",
        default=[],
        help="Additional mypy target path (repeatable)",
    )
    parser.add_argument(
        "--write-baseline",
        action="store_true",
        help="Overwrite baseline with current strict-mypy output",
    )
    args = parser.parse_args()

    if importlib.util.find_spec("mypy") is None:
        print(
            "mypy module is unavailable. Install dev deps with:\n"
            "  pip install -r requirements-dev.txt"
        )
        return 2

    baseline_path = Path(args.baseline)
    targets = tuple(DEFAULT_TARGETS) + tuple(str(t).strip() for t in args.target if str(t).strip())
    return_code, output, issues_now = run_mypy(targets, DEFAULT_FLAGS)

    if return_code not in (0, 1):
        print("strict mypy failed to execute successfully.")
        if output:
            print(output)
        return int(return_code or 2)

    if return_code == 1 and not issues_now:
        print("strict mypy returned errors but no parseable issue lines were found.")
        if output:
            print(output)
        return 2

    if args.write_baseline:
        save_baseline_entries(
            baseline_path, issues_now,
            header="# strict mypy baseline for scripts/typecheck_strict_gate.py",
        )
        print(f"Baseline written: {baseline_path} ({len(issues_now)} issues)")
        return 0

    baseline = load_baseline_entries(baseline_path)
    if not baseline and not baseline_path.exists():
        print(
            f"Baseline file is missing: {baseline_path}\n"
            "Create/update it with:\n"
            "  python scripts/typecheck_strict_gate.py --write-baseline"
        )
        return 2

    new_issues = sorted(issues_now - baseline)
    resolved_issues = sorted(baseline - issues_now)

    print(
        f"strict-mypy issues now={len(issues_now)} baseline={len(baseline)} "
        f"new={len(new_issues)} resolved={len(resolved_issues)}"
    )

    if resolved_issues:
        print("Resolved strict baseline issues detected; consider refreshing baseline.")
        for row in resolved_issues[:20]:
            print(f"  RESOLVED {row}")
        if len(resolved_issues) > 20:
            print(f"  ... and {len(resolved_issues) - 20} more")

    if new_issues:
        print("New strict mypy issues introduced:")
        for row in new_issues:
            print(f"  NEW {row}")
        if output:
            print("\nRaw strict mypy output:")
            print(output)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
