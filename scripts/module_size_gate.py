from __future__ import annotations

import argparse
import json
from pathlib import Path

DEFAULT_TARGETS: tuple[str, ...] = (
    "main.py",
    "analysis",
    "config",
    "core",
    "data",
    "models",
    "scripts",
    "trading",
    "ui",
    "utils",
)
DEFAULT_MAX_LINES = 1200
BASELINE_HEADER = [
    "# module-size baseline",
    "# Format: path:line_count",
    "",
]


def _normalize_path(path: str | Path) -> str:
    return str(path).strip().replace("\\", "/")


def _iter_python_files(targets: tuple[str, ...]) -> list[Path]:
    out: list[Path] = []
    seen: set[str] = set()
    for target in targets:
        candidate = Path(str(target).strip())
        if not candidate.exists():
            continue
        if candidate.is_file():
            if candidate.suffix.lower() == ".py":
                norm = _normalize_path(candidate)
                if norm not in seen:
                    out.append(candidate)
                    seen.add(norm)
            continue
        for py_file in sorted(candidate.rglob("*.py")):
            norm = _normalize_path(py_file)
            if norm in seen:
                continue
            out.append(py_file)
            seen.add(norm)
    return out


def collect_oversized_modules(targets: tuple[str, ...], max_lines: int) -> dict[str, int]:
    oversized: dict[str, int] = {}
    for module_path, line_count in collect_module_line_counts(targets).items():
        if line_count > max_lines:
            oversized[module_path] = line_count
    return oversized


def collect_module_line_counts(targets: tuple[str, ...]) -> dict[str, int]:
    line_counts: dict[str, int] = {}
    for py_file in _iter_python_files(targets):
        try:
            line_count = int(
                len(py_file.read_text(encoding="utf-8", errors="replace").splitlines())
            )
        except OSError:
            continue
        line_counts[_normalize_path(py_file)] = line_count
    return line_counts


def load_baseline(path: Path) -> dict[str, int]:
    if not path.exists():
        return {}
    out: dict[str, int] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        item = line.strip()
        if not item or item.startswith("#"):
            continue
        if ":" not in item:
            continue
        module_path, _, raw_count = item.rpartition(":")
        try:
            out[_normalize_path(module_path)] = int(raw_count)
        except ValueError:
            continue
    return out


def save_baseline(path: Path, oversized: dict[str, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(BASELINE_HEADER)
    for module_path in sorted(oversized.keys()):
        rows.append(f"{module_path}:{int(oversized[module_path])}")
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def load_budget(path: Path) -> dict[str, int]:
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(raw, dict):
        return {}
    out: dict[str, int] = {}
    for module_path, raw_limit in raw.items():
        norm = _normalize_path(module_path)
        if not norm:
            continue
        try:
            out[norm] = max(1, int(raw_limit))
        except (TypeError, ValueError):
            continue
    return out


def save_budget(path: Path, budget: dict[str, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    normalized: dict[str, int] = {}
    for module_path in sorted(budget.keys()):
        try:
            normalized[_normalize_path(module_path)] = max(1, int(budget[module_path]))
        except (TypeError, ValueError):
            continue
    path.write_text(
        json.dumps(normalized, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Module-size ratchet gate to prevent new oversized files."
    )
    parser.add_argument(
        "--baseline",
        default=".ci/module-size-baseline.txt",
        help="Path to baseline size map",
    )
    parser.add_argument(
        "--max-lines",
        type=int,
        default=DEFAULT_MAX_LINES,
        help="Maximum allowed lines before module is considered oversized",
    )
    parser.add_argument(
        "--target",
        action="append",
        default=[],
        help="Additional target file/directory (repeatable)",
    )
    parser.add_argument(
        "--write-baseline",
        action="store_true",
        help="Overwrite baseline with current oversized module map",
    )
    parser.add_argument(
        "--budget",
        default=".ci/module-size-budget.json",
        help="Optional module size cap JSON (path -> max lines)",
    )
    parser.add_argument(
        "--write-budget",
        action="store_true",
        help="Write current oversized modules to --budget JSON",
    )
    args = parser.parse_args()

    max_lines = max(1, int(args.max_lines))
    baseline_path = Path(args.baseline)
    targets = tuple(DEFAULT_TARGETS) + tuple(
        str(t).strip() for t in list(args.target or []) if str(t).strip()
    )
    oversized_now = collect_oversized_modules(targets, max_lines=max_lines)

    if args.write_baseline:
        save_baseline(baseline_path, oversized_now)
        print(f"Baseline written: {baseline_path} ({len(oversized_now)} modules)")
        return 0

    budget_path = Path(args.budget)
    if args.write_budget:
        save_budget(budget_path, oversized_now)
        print(f"Budget written: {budget_path} ({len(oversized_now)} modules)")
        return 0

    baseline = load_baseline(baseline_path)
    if not baseline and not baseline_path.exists():
        print(
            f"Baseline file is missing: {baseline_path}\n"
            "Create/update it with:\n"
            "  python scripts/module_size_gate.py --write-baseline"
        )
        return 2

    new_modules = sorted(path for path in oversized_now if path not in baseline)
    regressions = sorted(
        path
        for path in oversized_now
        if path in baseline and int(oversized_now[path]) > int(baseline[path])
    )
    improvements = sorted(
        path
        for path in baseline
        if path not in oversized_now or int(oversized_now[path]) < int(baseline[path])
    )

    print(
        f"module-size oversized_now={len(oversized_now)} baseline={len(baseline)} "
        f"new={len(new_modules)} regressions={len(regressions)} "
        f"improvements={len(improvements)} max_lines={max_lines}"
    )

    if improvements:
        print("Module size improvements detected; consider refreshing baseline.")
        for path in improvements[:20]:
            old = int(baseline.get(path, 0))
            new = int(oversized_now.get(path, 0))
            print(f"  IMPROVED {path}: {old} -> {new}")
        if len(improvements) > 20:
            print(f"  ... and {len(improvements) - 20} more")

    if new_modules or regressions:
        if new_modules:
            print("New oversized modules introduced:")
            for path in new_modules:
                print(f"  NEW {path}:{int(oversized_now[path])}")
        if regressions:
            print("Oversized modules grew beyond baseline:")
            for path in regressions:
                print(
                    "  REGRESSION "
                    f"{path}: {int(baseline[path])} -> {int(oversized_now[path])}"
                )
        return 1

    budget = load_budget(budget_path)
    if budget_path.exists() and budget:
        line_counts = collect_module_line_counts(targets)
        violations: list[tuple[str, int, int]] = []
        for module_path in sorted(budget.keys()):
            limit = int(budget[module_path])
            actual = int(line_counts.get(module_path, 0))
            if actual > limit:
                violations.append((module_path, actual, limit))
        if violations:
            print("Module-size budget violations:")
            for module_path, actual, limit in violations:
                print(
                    f"  VIOLATION {module_path}: actual={actual} budget={limit}"
                )
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
