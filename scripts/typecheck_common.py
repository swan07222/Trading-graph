from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

from scripts.gate_common import normalize_path

ERROR_RE = re.compile(
    r"^(?P<path>.+?):(?P<line>\d+)(?::(?P<column>\d+))?: error: "
    r"(?P<message>.+?)\s+\[(?P<code>[^\]]+)\]$"
)


def parse_mypy_errors(raw_output: str) -> set[str]:
    issues: set[str] = set()
    for line in str(raw_output or "").splitlines():
        matched = ERROR_RE.match(line.strip())
        if not matched:
            continue
        norm_path = normalize_path(matched.group("path"))
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


def save_baseline_entries(path: Path, issues: set[str], header: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = [
        header,
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
