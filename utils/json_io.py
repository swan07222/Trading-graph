"""JSON I/O utilities for consistent file operations.

This module provides centralized JSON file read/write functions to replace
duplicated implementations across the codebase.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from utils.logger import get_logger

_log = get_logger()


def read_json_safe(path: Path | str, default: Any = None) -> Any:
    """Read JSON file safely with error handling.

    Args:
        path: Path to JSON file.
        default: Default value if file doesn't exist or is invalid.

    Returns:
        Parsed JSON data or default.
    """
    try:
        file_path = Path(path)
        if not file_path.exists():
            return default
        with open(file_path, encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        _log.warning(f"Invalid JSON in {path}: {e}")
        return default
    except OSError as e:
        _log.warning(f"Failed to read {path}: {e}")
        return default
    except Exception as e:
        _log.warning(f"Unexpected error reading {path}: {e}")
        return default


def write_json_safe(path: Path | str, data: Any, *, indent: int = 2) -> bool:
    """Write JSON file safely with error handling.

    Args:
        path: Path to JSON file.
        data: Data to serialize.
        indent: JSON indentation level.

    Returns:
        True if write succeeded, False otherwise.
    """
    try:
        file_path = Path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=indent, ensure_ascii=False)
        return True
    except (TypeError, ValueError) as e:
        _log.warning(f"Invalid data for JSON {path}: {e}")
        return False
    except OSError as e:
        _log.warning(f"Failed to write {path}: {e}")
        return False
    except Exception as e:
        _log.warning(f"Unexpected error writing {path}: {e}")
        return False


def read_jsonl(path: Path | str) -> list[dict[str, Any]]:
    """Read JSONL file (one JSON object per line).

    Args:
        path: Path to JSONL file.

    Returns:
        List of parsed JSON objects. Empty list if file doesn't exist.
    """
    results: list[dict[str, Any]] = []
    try:
        file_path = Path(path)
        if not file_path.exists():
            return results
        with open(file_path, encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if isinstance(obj, dict):
                        results.append(obj)
                except json.JSONDecodeError as e:
                    _log.warning(f"Invalid JSON on line {line_num} in {path}: {e}")
    except OSError as e:
        _log.warning(f"Failed to read {path}: {e}")
    except Exception as e:
        _log.warning(f"Unexpected error reading {path}: {e}")
    return results


def write_jsonl(path: Path | str, data: list[dict[str, Any]]) -> bool:
    """Write JSONL file (one JSON object per line).

    Args:
        path: Path to JSONL file.
        data: List of dictionaries to serialize.

    Returns:
        True if write succeeded, False otherwise.
    """
    try:
        file_path = Path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            for obj in data:
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        return True
    except (TypeError, ValueError) as e:
        _log.warning(f"Invalid data for JSONL {path}: {e}")
        return False
    except OSError as e:
        _log.warning(f"Failed to write {path}: {e}")
        return False
    except Exception as e:
        _log.warning(f"Unexpected error writing {path}: {e}")
        return False
