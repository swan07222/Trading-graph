from __future__ import annotations

import re
from pathlib import Path


def normalize_path(path: str | Path) -> str:
    normalized = str(path).strip().replace("\\", "/")
    return re.sub(r"/+", "/", normalized)


def iter_python_files(targets: tuple[str, ...]) -> list[Path]:
    out: list[Path] = []
    seen: set[str] = set()
    for target in targets:
        candidate = Path(str(target).strip())
        if not candidate.exists():
            continue
        if candidate.is_file():
            if candidate.suffix.lower() == ".py":
                norm = normalize_path(candidate)
                if norm not in seen:
                    out.append(candidate)
                    seen.add(norm)
            continue
        for py_file in sorted(candidate.rglob("*.py")):
            norm = normalize_path(py_file)
            if norm in seen:
                continue
            out.append(py_file)
            seen.add(norm)
    return out
