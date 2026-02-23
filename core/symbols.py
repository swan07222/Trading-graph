"""
Stock symbol utilities for consistent code handling.

This module provides centralized stock code cleaning, validation, and normalization
functions to replace duplicated implementations across the codebase.
"""
from __future__ import annotations

import re
from typing import Final

from core.instruments import parse_instrument


# CN stock prefixes
CN_PREFIXES: Final[tuple[str, ...]] = (
    "sh.", "sz.", "bj.", "SH.", "SZ.", "BJ.",
    "sh", "sz", "bj", "SH", "SZ", "BJ",
    "CN:", "cn:",
)

# CN stock suffixes
CN_SUFFIXES: Final[tuple[str, ...]] = (".SS", ".SZ", ".BJ", ".ss", ".sz", ".bj")

# HK stock prefixes and suffixes
HK_PREFIXES: Final[tuple[str, ...]] = ("hk.", "HK.", "hk", "HK", "HK:", "hk:")
HK_SUFFIXES: Final[tuple[str, ...]] = (".HK", ".hk")

# US stock prefixes
US_PREFIXES: Final[tuple[str, ...]] = ("us.", "US.", "US:", "us:")

# Valid CN stock code patterns
CN_STOCK_PATTERN: Final[re.Pattern[str]] = re.compile(r"^\d{6}$")

# ST stock pattern (Special Treatment - financially troubled companies)
ST_STOCK_PATTERN: Final[re.Pattern[str]] = re.compile(r"^ST|^[*]\s*ST", re.IGNORECASE)


def _strip_prefixes(s: str, prefixes: tuple[str, ...]) -> str:
    """Strip any of the given prefixes from a string."""
    for p in prefixes:
        if s.startswith(p):
            return s[len(p):]
    return s


def _strip_suffixes(s: str, suffixes: tuple[str, ...]) -> str:
    """Strip any of the given suffixes from a string."""
    for suf in suffixes:
        if s.endswith(suf):
            return s[: -len(suf)]
    return s


def _digits_only(s: str) -> str:
    """Extract only digits from a string."""
    return "".join(ch for ch in s if ch.isdigit())


def clean_code(code: str) -> str:
    """
    Normalize a stock code to bare 6-digit form for CN stocks.

    Handles:
    - Prefixes: sh., sz., bj., SH, SZ, BJ, etc.
    - Suffixes: .SS, .SZ, .BJ, etc.
    - Separators: spaces, dashes, underscores
    - Zero-padding to 6 digits

    Args:
        code: Stock code to clean.

    Returns:
        Cleaned 6-digit code or empty string if invalid.
    """
    if code is None:
        return ""
    s = str(code).strip()
    if not s:
        return ""
    s = s.replace(" ", "").replace("-", "").replace("_", "")

    # Strip prefixes
    s = _strip_prefixes(s, CN_PREFIXES)

    # Strip suffixes
    s = _strip_suffixes(s, CN_SUFFIXES)

    # Extract digits only
    digits = _digits_only(s)
    return digits.zfill(6) if digits else ""


def normalize_cn_code(code: str) -> str:
    """
    Canonical CN stock code normalization for UI/feeds/execution.

    Uses parse_instrument for comprehensive market detection.

    Args:
        code: Stock code to normalize.

    Returns:
        6-digit code or empty string.
    """
    inst = parse_instrument(code)
    if inst.get("market") == "CN" and inst.get("asset") == "EQUITY":
        return str(inst.get("symbol") or "").zfill(6)
    return ""


def normalize_stock_code(code: str) -> str:
    """
    Normalize stock code preserving market-specific formatting.

    For CN stocks: returns 6-digit code.
    For HK stocks: returns 5-digit code.
    For US stocks: returns ticker symbol.

    Args:
        code: Stock code to normalize.

    Returns:
        Normalized code or empty string.
    """
    if code is None:
        return ""
    s = str(code).strip()
    if not s:
        return ""
    s = s.replace(" ", "").replace("-", "").replace("_", "")

    # Check for HK stocks
    s_upper = s.upper()
    if s_upper.startswith(tuple(p.upper() for p in HK_PREFIXES)) or s_upper.endswith(".HK"):
        t = _strip_prefixes(s, HK_PREFIXES)
        t = _strip_suffixes(t, HK_SUFFIXES)
        digits = _digits_only(t)
        if digits:
            return digits.zfill(5)

    # Check for US stocks
    if s_upper.startswith(tuple(p.upper() for p in US_PREFIXES)):
        t = _strip_prefixes(s, US_PREFIXES).upper()
        return t

    # Default to CN normalization
    return clean_code(code)


def validate_stock_code(code: str) -> tuple[bool, str]:
    """
    Validate a stock code format.

    Checks:
    - Not empty after cleaning
    - Numeric only (for CN stocks)
    - Exactly 6 digits (for CN stocks)
    - Valid CN stock code ranges

    Args:
        code: Stock code to validate.

    Returns:
        (is_valid, error_message) tuple.
        If valid, error_message is empty string.
    """
    if code is None:
        return False, "Invalid stock code: None"
    
    cleaned = clean_code(code)
    if not cleaned:
        return False, f"Invalid stock code format: '{code}' (empty after cleaning)"
    if not cleaned.isdigit():
        return False, f"Invalid stock code format: '{code}' (must be numeric)"
    if len(cleaned) != 6:
        return False, f"Invalid stock code format: '{code}' (must be 6 digits, got {len(cleaned)})"

    # Basic range checks for CN stocks
    if cleaned.startswith(("000", "001")):  # SZSE main board
        return True, ""
    if cleaned.startswith("002"):  # SME board
        return True, ""
    if cleaned.startswith(("300", "301")):  # ChiNext
        return True, ""
    if cleaned.startswith(("600", "601", "603", "605")):  # SSE main board
        return True, ""
    if cleaned.startswith("688"):  # STAR Market
        return True, ""
    if cleaned.startswith(("4", "8", "9")):  # BSE/other
        return True, ""

    # Unknown but valid format - allow with warning (don't reject)
    # This prevents false negatives for new/unknown stock prefixes
    return True, ""


def is_st_stock(name: str) -> bool:
    """
    Check if a stock name indicates ST (Special Treatment) status.

    ST stocks are financially troubled companies under special supervision.

    Args:
        name: Stock name to check.

    Returns:
        True if ST stock, False otherwise.
    """
    if not name:
        return False
    return bool(ST_STOCK_PATTERN.search(name))


def format_yahoo_symbol(code: str, market: str = "CN") -> str:
    """
    Format a stock code for Yahoo Finance API.

    Args:
        code: Stock code (cleaned).
        market: Market identifier (CN, HK, US).

    Returns:
        Yahoo Finance symbol.
    """
    if not code:
        return ""

    if market == "CN":
        cleaned = clean_code(code)
        if cleaned.startswith(("600", "601", "603", "605", "688")):
            return f"{cleaned}.SS"
        if cleaned.startswith(("000", "001", "002", "003", "300", "301")):
            return f"{cleaned}.SZ"
        if cleaned.startswith(("83", "87", "43")):
            return f"{cleaned}.BJ"
        return cleaned

    if market == "HK":
        digits = _digits_only(code)
        if digits:
            return f"{digits.zfill(5)}.HK"
        return code

    if market == "US":
        return code.upper()

    return code
