from __future__ import annotations

from dataclasses import fields
from datetime import time
from enum import Enum
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


def _safe_dataclass_from_dict(dc_instance: Any, data: dict[str, Any]) -> list[str]:
    """Apply dict values to a dataclass instance with type checking.
    Returns list of warnings for bad values.
    """
    warnings_list: list[str] = []
    if not isinstance(data, dict):
        return [f"Expected dict, got {type(data).__name__}"]

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
            else:
                warnings_list.append(
                    f"Type mismatch for '{key}': "
                    f"expected {type(current_value).__name__}, "
                    f"got {type(value).__name__}"
                )
        except (TypeError, ValueError) as e:
            warnings_list.append(f"Bad value for '{key}': {value!r} - {e}")

    return warnings_list


def _dataclass_to_dict(dc_instance: Any) -> dict[str, Any]:
    """Serialize a dataclass to dict, handling special types."""
    result: dict[str, Any] = {}
    for f in fields(dc_instance):
        value = getattr(dc_instance, f.name)
        if isinstance(value, time):
            result[f.name] = value.strftime("%H:%M:%S")
        elif isinstance(value, Enum):
            result[f.name] = value.value
        else:
            result[f.name] = value
    return result
