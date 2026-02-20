from __future__ import annotations

import argparse
import importlib.util
import re
import subprocess
import sys
from pathlib import Path

DEFAULT_TARGETS: tuple[str, ...] = (
    "main.py",
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
ERROR_RE = re.compile(
    r"^(?P<path>.+?):(?P<line>\d+)(?::(?P<column>\d+))?: error: "
    r"(?P<message>.+?)\s+\[(?P<code>[^\]]+)\]$"
)


def _normalize_path(path: str) -> str:
    normalized = str(path).strip().replace("\\", "/")
    return re.sub(r"/+", "/", normalized)


def parse_mypy_errors(raw_output: str) -> set[str]:
    issues: set[str] = set()
    for line in str(raw_output or "").splitlines():
        matched = ERROR_RE.match(line.strip())
        if not matched:
            continue
        norm_path = _normalize_path(matched.group("path"))
        issue_key = (
            f"{norm_path}:{matched.group('line')}:"
            f"{matched.group('code')}:{matched.group('message')}"
        )
        issues.add(issue_key)
    return issues


def load_baseline_entries(path: Path) -> set[str]:
    if not path.exists():
        return set()
    rows: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        item = line.strip()
        if not item or item.startswith("#"):
            continue
        rows.add(item)
    return rows


def save_baseline_entries(path: Path, issues: set[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = [
        "# strict mypy baseline for scripts/typecheck_strict_gate.py",
        "# Format: path:line:code:message",
        "",
    ]
    body.extend(sorted(issues))
    path.write_text("\n".join(body) + "\n", encoding="utf-8")


def run_mypy(targets: tuple[str, ...], flags: tuple[str, ...]) -> tuple[int, str, set[str]]:
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
        save_baseline_entries(baseline_path, issues_now)
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
