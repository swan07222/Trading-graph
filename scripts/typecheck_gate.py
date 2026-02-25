from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
from pathlib import Path

from scripts.typecheck_common import (
    load_baseline_entries,
    parse_mypy_errors,
    save_baseline_entries,
)

DEFAULT_TARGETS: tuple[str, ...] = (
    "main.py",
    "config/settings.py",
    "core/constants.py",
    "core/instruments.py",
    "core/network.py",
    "data/fetcher.py",
    "data/fetcher_registry.py",
    "data/feeds.py",
    "data/news.py",
    "data/processor.py",
    "data/session_cache.py",
    "data/discovery.py",
    "analysis/strategy_marketplace.py",
    "analysis/technical.py",
    "models/ensemble.py",
    "models/predictor.py",
    "models/trainer.py",
    "trading/broker.py",
    "trading/broker_live.py",
    "trading/executor.py",
    "trading/kill_switch.py",
    "trading/portfolio.py",
    "trading/alerts.py",
    "ui/app.py",
    "ui/app_bar_ops.py",
    "ui/app_lifecycle_ops.py",
    "ui/app_monitoring_ops.py",
    "ui/app_panels.py",
    "ui/app_chart_pipeline.py",
    "utils/logger.py",
    "utils/metrics.py",
    "utils/metrics_http.py",
    "utils/security.py",
    "scripts/deployment_snapshot.py",
    "scripts/release_preflight.py",
    "scripts/soak_broker_e2e.py",
    "scripts/exception_policy_gate.py",
    "scripts/module_size_gate.py",
    "scripts/typecheck_strict_gate.py",
)
DEFAULT_FLAGS: tuple[str, ...] = (
    "--follow-imports=silent",
    "--check-untyped-defs",
    "--warn-return-any",
    "--warn-redundant-casts",
)
DEFAULT_BATCH_SIZE = 8


def _run_mypy_once(
    targets: tuple[str, ...],
    flags: tuple[str, ...],
) -> tuple[int, str, set[str]]:
    cmd = [sys.executable, "-m", "mypy", *flags, *targets]
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    combined = "\n".join(part for part in (proc.stdout, proc.stderr) if part).strip()
    issues = parse_mypy_errors(combined)
    return proc.returncode, combined, issues


def _iter_target_batches(
    targets: tuple[str, ...],
    batch_size: int,
) -> list[tuple[str, ...]]:
    size = max(1, int(batch_size))
    out: list[tuple[str, ...]] = []
    for i in range(0, len(targets), size):
        out.append(tuple(targets[i:i + size]))
    return out


def run_mypy(
    targets: tuple[str, ...],
    flags: tuple[str, ...],
) -> tuple[int, str, set[str]]:
    """Run mypy in batches to avoid memory/CPU spikes on large target sets.
    
    Returns aggregate (return_code, output, issues).
    """
    batches = _iter_target_batches(targets, DEFAULT_BATCH_SIZE)
    all_issues: set[str] = set()
    output_lines: list[str] = []
    worst_code = 0
    
    for idx, batch in enumerate(batches, start=1):
        output_lines.append(f"batch {idx}/{len(batches)}")
        code, out, issues = _run_mypy_once(batch, flags)
        output_lines.append(out)
        all_issues.update(issues)
        if code > worst_code:
            worst_code = code
            if code >= 2:
                break
    
    return worst_code, "\n".join(output_lines), all_issues


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Typecheck gate with baseline support for gradual hardening"
    )
    parser.add_argument(
        "--baseline",
        default=".ci/mypy-baseline.txt",
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
        help="Overwrite baseline with current mypy output",
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
        print("mypy failed to execute successfully.")
        if output:
            print(output)
        return int(return_code or 2)

    if return_code == 1 and not issues_now:
        print("mypy returned errors but no parseable issue lines were found.")
        if output:
            print(output)
        return 2

    if args.write_baseline:
        save_baseline_entries(
            baseline_path, issues_now,
            header="# mypy baseline for scripts/typecheck_gate.py",
        )
        print(f"Baseline written: {baseline_path} ({len(issues_now)} issues)")
        return 0

    baseline = load_baseline_entries(baseline_path)
    if not baseline and not baseline_path.exists():
        print(
            f"Baseline file is missing: {baseline_path}\n"
            "Create/update it with:\n"
            "  python scripts/typecheck_gate.py --write-baseline"
        )
        return 2

    new_issues = sorted(issues_now - baseline)
    resolved_issues = sorted(baseline - issues_now)

    print(
        f"mypy issues now={len(issues_now)} baseline={len(baseline)} "
        f"new={len(new_issues)} resolved={len(resolved_issues)}"
    )

    if resolved_issues:
        print("Resolved baseline issues detected; consider refreshing baseline.")
        for row in resolved_issues[:20]:
            print(f"  RESOLVED {row}")
        if len(resolved_issues) > 20:
            print(f"  ... and {len(resolved_issues) - 20} more")

    if new_issues:
        print("New mypy issues introduced:")
        for row in new_issues:
            print(f"  NEW {row}")
        if output:
            print("\nRaw mypy output:")
            print(output)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
