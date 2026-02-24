from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Iterable

from scripts.gate_common import normalize_path, iter_python_files

DEFAULT_TARGETS: tuple[str, ...] = (
    "trading/executor.py",
    "data/fetcher.py",
    "data/feeds.py",
    "models/predictor.py",
    "models/auto_learner.py",
    "ui/app.py",
    "ui/app_bar_ops.py",
    "ui/app_panels.py",
    "ui/app_chart_pipeline.py",
)

BROAD_EXCEPTION_NAMES = {"Exception", "BaseException"}
BASELINE_HEADER = [
    "# broad-exception policy baseline",
    "# Format: path:line:code",
    "",
]


def _normalize_path(path: Path | str) -> str:
    """Normalize path for consistent comparison."""
    return normalize_path(str(path))


def _iter_python_files(targets: Iterable[str]) -> Iterable[Path]:
    """Iterate over Python files from targets."""
    return iter_python_files(targets)


def _is_broad_exception_type(node: ast.expr | None) -> bool:
    if node is None:
        return True
    if isinstance(node, ast.Name):
        return node.id in BROAD_EXCEPTION_NAMES
    if isinstance(node, ast.Attribute):
        return str(node.attr) in BROAD_EXCEPTION_NAMES
    if isinstance(node, ast.Tuple):
        return any(_is_broad_exception_type(elt) for elt in node.elts)
    return False


def _is_silent_pass_body(body: list[ast.stmt]) -> bool:
    filtered: list[ast.stmt] = []
    for stmt in body:
        if isinstance(stmt, ast.Pass):
            continue
        if (
            isinstance(stmt, ast.Expr)
            and isinstance(stmt.value, ast.Constant)
            and isinstance(stmt.value.value, str)
        ):
            continue
        filtered.append(stmt)
    return len(filtered) == 0


class _ExceptionPolicyVisitor(ast.NodeVisitor):
    def __init__(self, normalized_path: str) -> None:
        self.path = normalized_path
        self.issues: set[str] = set()

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        code: str | None = None
        if node.type is None:
            code = "bare-except"
        elif _is_broad_exception_type(node.type):
            code = "broad-except"

        if code is not None:
            line = int(getattr(node, "lineno", 1) or 1)
            self.issues.add(f"{self.path}:{line}:{code}")
            if _is_silent_pass_body(list(node.body or [])):
                self.issues.add(f"{self.path}:{line}:silent-pass")

        self.generic_visit(node)


def collect_exception_policy_issues(targets: tuple[str, ...]) -> tuple[set[str], list[str]]:
    issues: set[str] = set()
    parse_errors: list[str] = []
    for py_file in _iter_python_files(targets):
        try:
            # Accept UTF-8 BOM files so gates don't fail on encoding artifacts.
            source = py_file.read_text(encoding="utf-8-sig")
        except UnicodeDecodeError:
            source = py_file.read_text(encoding="utf-8-sig", errors="replace")
        except OSError as exc:
            parse_errors.append(f"{_normalize_path(py_file)}: read error: {exc}")
            continue

        try:
            tree = ast.parse(source, filename=str(py_file))
        except SyntaxError as exc:
            parse_errors.append(
                f"{_normalize_path(py_file)}:{int(exc.lineno or 1)}: syntax error: {exc.msg}"
            )
            continue

        visitor = _ExceptionPolicyVisitor(_normalize_path(py_file))
        visitor.visit(tree)
        issues.update(visitor.issues)

    return issues, parse_errors


def load_baseline_entries(path: Path) -> set[str]:
    if not path.exists():
        return set()
    rows: set[str] = set()
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        item = line.strip()
        if not item or item.startswith("#"):
            continue
        rows.add(item)
    return rows


def save_baseline_entries(path: Path, issues: set[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = list(BASELINE_HEADER)
    payload.extend(sorted(issues))
    path.write_text("\n".join(payload) + "\n", encoding="utf-8")


def summarize_issue_counts(issues: set[str]) -> dict[str, dict[str, int]]:
    """Summarize issue keys into per-file, per-code counts.

    Input key format:
      path:line:code
    """
    out: dict[str, dict[str, int]] = {}
    for row in sorted(issues):
        try:
            path_part, _line_part, code = row.rsplit(":", 2)
        except ValueError:
            continue
        p = normalize_path(path_part)
        code_key = str(code).strip().lower()
        if not p or not code_key:
            continue
        bucket = out.setdefault(p, {})
        bucket[code_key] = int(bucket.get(code_key, 0)) + 1
    return out


def load_budget(path: Path) -> dict[str, dict[str, int]]:
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(raw, dict):
        return {}

    out: dict[str, dict[str, int]] = {}
    for path_key, value in raw.items():
        norm = _normalize_path(path_key)
        if not norm or not isinstance(value, dict):
            continue
        row: dict[str, int] = {}
        for code, limit in value.items():
            code_key = str(code).strip().lower()
            if not code_key:
                continue
            try:
                row[code_key] = max(0, int(limit))
            except (TypeError, ValueError):
                continue
        if row:
            out[norm] = row
    return out


def save_budget(path: Path, summary: dict[str, dict[str, int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    normalized: dict[str, dict[str, int]] = {}
    for path_key in sorted(summary.keys()):
        row = summary.get(path_key, {})
        clean: dict[str, int] = {}
        for code_key in sorted(row.keys()):
            try:
                clean[str(code_key).strip().lower()] = max(0, int(row[code_key]))
            except (TypeError, ValueError):
                continue
        if clean:
            normalized[_normalize_path(path_key)] = clean
    path.write_text(
        json.dumps(normalized, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Exception policy gate with baseline support."
    )
    parser.add_argument(
        "--baseline",
        default=".ci/exception-policy-baseline.txt",
        help="Path to baseline issue list",
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
        help="Overwrite baseline with current issue set",
    )
    parser.add_argument(
        "--budget",
        default=".ci/exception-policy-budget.json",
        help="Optional per-file issue budget JSON (path -> code -> max count)",
    )
    parser.add_argument(
        "--write-budget",
        action="store_true",
        help="Write current per-file issue counts to --budget JSON",
    )
    args = parser.parse_args()

    baseline_path = Path(args.baseline)
    targets = tuple(DEFAULT_TARGETS) + tuple(
        str(t).strip() for t in list(args.target or []) if str(t).strip()
    )

    issues_now, parse_errors = collect_exception_policy_issues(targets)
    if parse_errors:
        print("Failed to parse all targets:")
        for row in parse_errors[:20]:
            print(f"  {row}")
        if len(parse_errors) > 20:
            print(f"  ... and {len(parse_errors) - 20} more")
        return 2

    if args.write_baseline:
        save_baseline_entries(baseline_path, issues_now)
        print(f"Baseline written: {baseline_path} ({len(issues_now)} issues)")
        return 0

    summary_now = summarize_issue_counts(issues_now)
    budget_path = Path(args.budget)
    if args.write_budget:
        save_budget(budget_path, summary_now)
        print(f"Budget written: {budget_path} ({len(summary_now)} files)")
        return 0
    budget = load_budget(budget_path)
    budget_enabled = bool(budget_path.exists() and budget)

    baseline = load_baseline_entries(baseline_path)
    if (not baseline) and (not baseline_path.exists()) and (not budget_enabled):
        print(
            f"Baseline file is missing: {baseline_path}\n"
            "Create/update it with:\n"
            "  python scripts/exception_policy_gate.py --write-baseline"
        )
        return 2

    new_issues = sorted(issues_now - baseline)
    resolved_issues = sorted(baseline - issues_now)

    print(
        f"exception-policy issues now={len(issues_now)} baseline={len(baseline)} "
        f"new={len(new_issues)} resolved={len(resolved_issues)}"
    )

    if resolved_issues:
        print("Resolved baseline issues detected; consider refreshing baseline.")
        for row in resolved_issues[:20]:
            print(f"  RESOLVED {row}")
        if len(resolved_issues) > 20:
            print(f"  ... and {len(resolved_issues) - 20} more")

    if new_issues:
        print("New exception-policy issues introduced:")
        for row in new_issues:
            print(f"  NEW {row}")
        if not budget_enabled:
            return 1
        print(
            "Line-based baseline drift tolerated because "
            f"budget mode is enabled ({budget_path})."
        )

    if budget_enabled:
        violations: list[tuple[str, str, int, int]] = []
        for path_key in sorted(budget.keys()):
            limits = budget.get(path_key, {})
            current = summary_now.get(path_key, {})
            for code_key in sorted(limits.keys()):
                limit = int(limits[code_key])
                actual = int(current.get(code_key, 0))
                if actual > limit:
                    violations.append((path_key, code_key, actual, limit))
        if violations:
            print("Exception-policy budget violations:")
            for path_key, code_key, actual, limit in violations:
                print(
                    f"  VIOLATION {path_key}:{code_key} "
                    f"actual={actual} budget={limit}"
                )
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
