"""Type conversion utilities for safe and consistent type handling.

This module provides centralized safe type conversion functions to replace
duplicated implementations across the codebase.
"""
from __future__ import annotations

import math
from typing import Any


def safe_float(value: Any, default: float = 0.0) -> float:
    """Safely convert a value to float with fallback.

    Handles None, NaN, Inf, and conversion errors gracefully.

    Args:
        value: Value to convert.
        default: Default value if conversion fails.

    Returns:
        Converted float or default.
    """
    try:
        if value is None:
            return float(default)
        result = float(value)
        if math.isnan(result) or math.isinf(result):
            return float(default)
        return result
    except (TypeError, ValueError, OverflowError):
        return float(default)


def safe_int(value: Any, default: int = 0) -> int:
    """Safely convert a value to int with fallback.

    Handles None and conversion errors gracefully.

    Args:
        value: Value to convert.
        default: Default value if conversion fails.

    Returns:
        Converted int or default.
    """
    try:
        if value is None:
            return int(default)
        return int(float(value))
    except (TypeError, ValueError, OverflowError):
        return int(default)


def safe_str(value: Any, default: str = "") -> str:
    """Safely convert a value to string with fallback.

    Handles None and conversion errors gracefully.

    Args:
        value: Value to convert.
        default: Default value if conversion fails.

    Returns:
        Converted string or default.
    """
    try:
        if value is None:
            return str(default)
        return str(value)
    except Exception:
        return str(default)


def safe_attr(obj: Any, attr: str, default: Any = None) -> Any:
    """Safely extract an attribute from an object with fallback.

    Args:
        obj: Object to extract attribute from.
        attr: Attribute name.
        default: Default value if attribute doesn't exist or extraction fails.

    Returns:
        Attribute value or default.
    """
    try:
        return getattr(obj, attr, default)
    except Exception:
        return default


def safe_float_attr(obj: Any, attr: str, default: float = 0.0) -> float:
    """Safely extract a float attribute from an object.

    Combines safe_attr and safe_float for convenient attribute access.

    Args:
        obj: Object to extract attribute from.
        attr: Attribute name.
        default: Default value if extraction fails.

    Returns:
        Float attribute value or default.
    """
    return safe_float(safe_attr(obj, attr, default), default)


def safe_int_attr(obj: Any, attr: str, default: int = 0) -> int:
    """Safely extract an int attribute from an object.

    Combines safe_attr and safe_int for convenient attribute access.

    Args:
        obj: Object to extract attribute from.
        attr: Attribute name.
        default: Default value if extraction fails.

    Returns:
        Int attribute value or default.
    """
    return safe_int(safe_attr(obj, attr, default), default)


def safe_str_attr(obj: Any, attr: str, default: str = "") -> str:
    """Safely extract a string attribute from an object.

    Combines safe_attr and safe_str for convenient attribute access.

    Args:
        obj: Object to extract attribute from.
        attr: Attribute name.
        default: Default value if extraction fails.

    Returns:
        String attribute value or default.
    """
    return safe_str(safe_attr(obj, attr, default), default)


def clamp(value: float, min_value: float, max_value: float) -> float:
    """Clamp a value between min and max bounds.

    Args:
        value: Value to clamp.
        min_value: Minimum bound.
        max_value: Maximum bound.

    Returns:
        Clamped value.
    """
    return max(min_value, min(max_value, value))
