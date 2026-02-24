"""Serialization utilities for dataclass and object (de)serialization.

This module provides centralized serialization functions to replace
duplicated implementations across the codebase.
"""
from __future__ import annotations

from dataclasses import fields, is_dataclass
from datetime import date, datetime, time
from enum import Enum
from pathlib import Path
from typing import Any


def _coerce_bool(value: Any) -> tuple[bool, bool]:
    """Coerce common bool representations.

    Returns (ok, parsed_value). When ok is False, parsed_value is undefined.
    """
    if isinstance(value, bool):
        return True, value

    if isinstance(value, int):
        if value in (0, 1):
            return True, bool(value)
        return False, False

    if isinstance(value, float):
        if value in (0.0, 1.0):
            return True, bool(int(value))
        return False, False

    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in ("1", "true", "yes", "on", "y"):
            return True, True
        if normalized in ("0", "false", "no", "off", "n"):
            return True, False
        return False, False

    return False, False


def safe_dataclass_from_dict(dc_instance: Any, data: dict[str, Any]) -> list[str]:
    """Apply dict values to a dataclass instance with type checking.

    Args:
        dc_instance: Dataclass instance to update.
        data: Dictionary with new values.

    Returns:
        List of warnings for bad/ignored values.
    """
    warnings_list: list[str] = []
    if not isinstance(data, dict):
        return [f"Expected dict, got {type(data).__name__}"]

    if not is_dataclass(dc_instance):
        return [f"Not a dataclass: {type(dc_instance).__name__}"]

    dc_fields = {f.name: f for f in fields(dc_instance)}

    for key, value in data.items():
        if key not in dc_fields:
            warnings_list.append(f"Unknown field '{key}' - ignored")
            continue

        current_value = getattr(dc_instance, key)

        try:
            # Bool check must happen before int check since bool is an int subtype.
            if isinstance(current_value, bool):
                ok, parsed = _coerce_bool(value)
                if ok:
                    setattr(dc_instance, key, parsed)
                else:
                    warnings_list.append(
                        f"Bad value for bool field '{key}': {value!r}"
                    )
            elif isinstance(current_value, int) and isinstance(value, (int, float)):
                setattr(dc_instance, key, int(value))
            elif isinstance(current_value, float) and isinstance(value, (int, float)):
                setattr(dc_instance, key, float(value))
            elif isinstance(current_value, str) and isinstance(value, str):
                setattr(dc_instance, key, value)
            elif isinstance(current_value, list) and isinstance(value, list):
                setattr(dc_instance, key, value)
            elif isinstance(current_value, dict) and isinstance(value, dict):
                setattr(dc_instance, key, value)
            elif isinstance(current_value, time) and isinstance(value, str):
                parts = [int(p) for p in value.split(":")]
                # time() requires at least hour and minute
                if len(parts) >= 2:
                    setattr(dc_instance, key, time(parts[0], parts[1], parts[2] if len(parts) > 2 else 0, parts[3] if len(parts) > 3 else 0))
            elif isinstance(current_value, date) and isinstance(value, str):
                setattr(dc_instance, key, datetime.fromisoformat(value).date())
            elif isinstance(current_value, datetime) and isinstance(value, str):
                setattr(dc_instance, key, datetime.fromisoformat(value))
            else:
                warnings_list.append(
                    f"Type mismatch for '{key}': "
                    f"expected {type(current_value).__name__}, "
                    f"got {type(value).__name__}"
                )
        except (TypeError, ValueError) as e:
            warnings_list.append(f"Bad value for '{key}': {value!r} - {e}")

    return warnings_list


def dataclass_to_dict(dc_instance: Any) -> dict[str, Any]:
    """Serialize a dataclass to dict, handling special types.

    Args:
        dc_instance: Dataclass instance to serialize.

    Returns:
        Dictionary representation.
    """
    if not is_dataclass(dc_instance):
        return {}

    result: dict[str, Any] = {}
    for f in fields(dc_instance):
        value = getattr(dc_instance, f.name)
        if isinstance(value, time):
            result[f.name] = value.strftime("%H:%M:%S")
        elif isinstance(value, date):
            result[f.name] = value.isoformat()
        elif isinstance(value, datetime):
            result[f.name] = value.isoformat()
        elif isinstance(value, Enum):
            result[f.name] = value.value
        elif isinstance(value, Path):
            result[f.name] = str(value)
        elif is_dataclass(value):
            result[f.name] = dataclass_to_dict(value)
        elif isinstance(value, list):
            result[f.name] = [
                dataclass_to_dict(item) if is_dataclass(item) else item
                for item in value
            ]
        elif isinstance(value, dict):
            result[f.name] = {
                str(k): (dataclass_to_dict(v) if is_dataclass(v) else v)
                for k, v in value.items()
            }
        else:
            result[f.name] = value
    return result


def to_serializable(obj: Any) -> Any:
    """Convert an object to a JSON-serializable representation.

    Handles common non-serializable types like datetime, date, Enum, Path.

    Args:
        obj: Object to convert.

    Returns:
        JSON-serializable representation.
    """
    if obj is None:
        return None
    if isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, (date, datetime)):
        return obj.isoformat()
    if isinstance(obj, time):
        return obj.strftime("%H:%M:%S")
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, Path):
        return str(obj)
    if is_dataclass(obj):
        return dataclass_to_dict(obj)
    if isinstance(obj, (list, tuple)):
        return [to_serializable(item) for item in obj]
    if isinstance(obj, dict):
        return {str(k): to_serializable(v) for k, v in obj.items()}
    return str(obj)
