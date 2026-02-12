# core/instruments.py
from __future__ import annotations

import re
from typing import Dict, Any

try:
    # Safe import: constants does not import instruments -> no circular
    from core.constants import get_exchange
except Exception:
    get_exchange = None

CN_PREFIXES = (
    "sh.", "sz.", "bj.", "SH.", "SZ.", "BJ.",
    "sh", "sz", "bj", "SH", "SZ", "BJ",
    "CN:", "cn:",
)
HK_PREFIXES = ("hk.", "HK.", "hk", "HK", "HK:", "hk:")
US_PREFIXES = ("us.", "US.", "us", "US", "US:", "us:")

CN_SUFFIXES = (".SS", ".SZ", ".BJ", ".ss", ".sz", ".bj")
HK_SUFFIXES = (".HK", ".hk")
US_SUFFIXES = ()  # keep empty; US tickers rarely have suffix in user input

def _strip_prefixes(s: str, prefixes) -> str:
    for p in prefixes:
        if s.startswith(p):
            return s[len(p):]
    return s

def _strip_suffixes(s: str, suffixes) -> str:
    for suf in suffixes:
        if s.endswith(suf):
            return s[:-len(suf)]
    return s

def _digits_only(s: str) -> str:
    return "".join(ch for ch in s if ch.isdigit())

def _letters_numbers(s: str) -> str:
    return "".join(ch for ch in s if ch.isalnum())

def _is_us_ticker(s: str) -> bool:
    # Basic ticker check: 1-6 letters/numbers/dot (e.g., BRK.B)
    # Keep it permissive; broker may accept more.
    return bool(re.fullmatch(r"[A-Z0-9]{1,6}(\.[A-Z])?", s))

def _cn_yahoo_suffix(code6: str) -> str:
    """
    Best-effort mapping of CN code -> Yahoo suffix.
    If get_exchange is available, use it; else fall back on prefix heuristics.
    """
    if get_exchange:
        ex = get_exchange(code6)
        if ex == "SSE":
            return ".SS"
        if ex == "SZSE":
            return ".SZ"
        if ex == "BSE":
            return ".BJ"

    if code6.startswith(("600", "601", "603", "605", "688")):
        return ".SS"
    if code6.startswith(("000", "001", "002", "003", "300", "301")):
        return ".SZ"
    if code6.startswith(("83", "87", "43")):
        return ".BJ"
    return ""

def instrument_key(inst: Dict[str, Any]) -> str:
    """Stable unique key for caching/storage."""
    market = str(inst.get("market") or "UNKNOWN").upper()
    asset = str(inst.get("asset") or "UNKNOWN").upper()
    sym = str(inst.get("symbol") or "")
    return f"{market}:{asset}:{sym}"

def parse_instrument(code: str) -> Dict[str, Any]:
    """
    Parse a user-provided symbol/code into a normalized instrument dict.

    Examples:
      "600519", "sh600519", "600519.SS"         -> CN EQUITY symbol=600519 yahoo=600519.SS
      "000001"                                  -> CN EQUITY
      "bj430047"                                -> CN EQUITY (BSE) yahoo=430047.BJ
      "0700.HK", "HK:0700", "hk0700"           -> HK EQUITY symbol=00700 yahoo=00700.HK
      "AAPL", "US:AAPL"                         -> US EQUITY yahoo=AAPL
      "BTC-USD", "BTC/USDT", "ETHUSDT"          -> CRYPTO CRYPTO
    """
    raw = "" if code is None else str(code).strip()
    s = raw.strip()
    s = s.replace(" ", "").replace("-", "").replace("_", "")

    if not s:
        return {
            "market": "UNKNOWN",
            "asset": "EQUITY",
            "symbol": "",
            "currency": "",
            "yahoo": "",
            "raw": raw,
            "vendor": {},
        }

    # Keep a case-preserving copy for suffix detection
    s_upper = s.upper()

    # -------------------------
    # -------------------------
    is_explicit_hk = s_upper.startswith(tuple(p.upper() for p in HK_PREFIXES)) or s_upper.endswith(".HK")
    is_explicit_us = s_upper.startswith(tuple(p.upper() for p in US_PREFIXES))
    is_explicit_cn = s_upper.startswith(tuple(p.upper() for p in CN_PREFIXES)) or any(s_upper.endswith(x) for x in (".SS", ".SZ", ".BJ"))

    # -------------------------
    # Crypto detection (best effort)
    # -------------------------
    # If it contains common crypto quote tokens, treat as crypto.
    # Accept formats: BTCUSD, BTCUSDT, BTC/USD, BTC-USDT, BTC-USD
    crypto_hint = any(tok in s_upper for tok in ("USDT", "BTC", "ETH", "USD")) and (
        "/" in raw or "-" in raw or s_upper.endswith(("USDT", "USD"))
    )
    if crypto_hint:
        sym = raw.upper().replace("/", "").replace("-", "")
        sym = _letters_numbers(sym)
        return {
            "market": "CRYPTO",
            "asset": "CRYPTO",
            "symbol": sym,
            "currency": "USD",
            "yahoo": "",   # yfinance crypto varies; leave blank unless you implement it
            "raw": raw,
            "vendor": {},
        }

    # -------------------------
    # HK equities (explicit only)
    # -------------------------
    if is_explicit_hk:
        t = _strip_prefixes(s, HK_PREFIXES)
        t = _strip_suffixes(t, HK_SUFFIXES)
        digits = _digits_only(t)
        # HK code should be 1-5 digits; standard is 5 with leading zeros
        if digits:
            sym5 = digits.zfill(5)
            return {
                "market": "HK",
                "asset": "EQUITY",
                "symbol": sym5,
                "currency": "HKD",
                "yahoo": f"{sym5}.HK",
                "raw": raw,
                "vendor": {},
            }

    # -------------------------
    # CN equities (default for digits)
    # -------------------------
    # Strip CN prefixes/suffixes first
    t = _strip_prefixes(s, CN_PREFIXES)
    t = _strip_suffixes(t, CN_SUFFIXES)
    digits = _digits_only(t)

    # If it is digits (<=6), treat as CN by default (maintains your existing behavior)
    if digits and len(digits) <= 6:
        code6 = digits.zfill(6)
        suf = _cn_yahoo_suffix(code6)
        yahoo = f"{code6}{suf}" if suf else ""
        return {
            "market": "CN",
            "asset": "EQUITY",
            "symbol": code6,
            "currency": "CNY",
            "yahoo": yahoo,
            "raw": raw,
            "vendor": {},
        }

    # -------------------------
    # US equities (ticker)
    # -------------------------
    # Strip optional US: prefix, keep dot class (BRK.B)
    if is_explicit_us:
        t = _strip_prefixes(s, US_PREFIXES)
        t = t.upper()
        t = _letters_numbers(t.replace(".", "."))  # keep dot through next step
        # restore dot if removed by letters_numbers
        # (letters_numbers removes dot; handle separately)
        t = raw.upper().replace("US:", "").replace("us:", "").replace("US", "").replace("us", "")
        t = t.strip()
        if _is_us_ticker(t):
            return {
                "market": "US",
                "asset": "EQUITY",
                "symbol": t,
                "currency": "USD",
                "yahoo": t,
                "raw": raw,
                "vendor": {},
            }

    sym = raw.upper().strip()
    if _is_us_ticker(sym):
        return {
            "market": "US",
            "asset": "EQUITY",
            "symbol": sym,
            "currency": "USD",
            "yahoo": sym,
            "raw": raw,
            "vendor": {},
        }

    return {
        "market": "UNKNOWN",
        "asset": "EQUITY",
        "symbol": _letters_numbers(raw.upper()),
        "currency": "",
        "yahoo": "",
        "raw": raw,
        "vendor": {},
    }