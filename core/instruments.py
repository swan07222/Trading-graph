# core/instruments.py
from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

_get_exchange: Callable[[str], str] | None
try:
    # Safe import: constants does not import instruments -> no circular
    from core.constants import get_exchange as _get_exchange
except Exception:
    _get_exchange = None

get_exchange: Callable[[str], str] | None = _get_exchange

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

_OCC_OPTION_RE = re.compile(
    r"^([A-Z]{1,6})(\d{2})(\d{2})(\d{2})([CP])(\d{8})$"
)
_CN_FUTURE_RE = re.compile(r"^([A-Z]{1,3})(\d{3,4})$")
_US_FUTURE_RE = re.compile(r"^/?([A-Z]{1,3})([FGHJKMNQUVXZ])(\d{1,2})$")
_FX_PAIR_RE = re.compile(r"^([A-Z]{3})[/-]?([A-Z]{3})$")
_KNOWN_CCY = {
    "USD", "EUR", "GBP", "JPY", "CHF", "AUD", "NZD", "CAD",
    "CNH", "CNY", "HKD", "SGD",
}
_CN_FUTURE_PREFIXES = {
    "IF", "IH", "IC", "IM", "TF", "T", "TS", "TL",
    "CU", "AL", "ZN", "AU", "AG", "RB", "HC", "I", "J", "JM",
    "M", "Y", "P", "A", "C", "SR", "CF", "TA", "MA", "RU",
}

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

def _parse_occ_option(s_upper: str) -> dict[str, Any] | None:
    """
    Parse OCC-style US option symbols.

    Format:
      <ROOT><YY><MM><DD><C|P><STRIKE*1000:8d>
    Example:
      AAPL250117C00150000
    """
    m = _OCC_OPTION_RE.fullmatch(s_upper.strip())
    if not m:
        return None
    root, yy, mm, dd, cp, strike_raw = m.groups()
    expiry = f"20{yy}-{mm}-{dd}"
    strike = int(strike_raw) / 1000.0
    return {
        "market": "US",
        "asset": "OPTION",
        "symbol": s_upper.strip(),
        "currency": "USD",
        "yahoo": "",
        "raw": s_upper,
        "vendor": {
            "underlying": root,
            "expiry": expiry,
            "option_type": "call" if cp == "C" else "put",
            "strike": strike,
        },
    }

def _parse_cn_future(s_upper: str, raw: str) -> dict[str, Any] | None:
    m = _CN_FUTURE_RE.fullmatch(s_upper.strip())
    if not m:
        return None
    root, ym = m.groups()
    if root not in _CN_FUTURE_PREFIXES:
        return None
    if len(ym) not in (3, 4):
        return None
    norm = f"{root}{ym}"
    return {
        "market": "CN",
        "asset": "FUTURE",
        "symbol": norm,
        "currency": "CNY",
        "yahoo": "",
        "raw": raw,
        "vendor": {
            "root": root,
            "contract_ym": ym,
        },
    }

def _parse_us_future(s_upper: str, raw: str) -> dict[str, Any] | None:
    m = _US_FUTURE_RE.fullmatch(s_upper.strip())
    if not m:
        return None
    root, month_code, year = m.groups()
    norm = f"{root}{month_code}{year}"
    return {
        "market": "US",
        "asset": "FUTURE",
        "symbol": norm,
        "currency": "USD",
        "yahoo": "",
        "raw": raw,
        "vendor": {
            "root": root,
            "month_code": month_code,
            "year": year,
        },
    }

def _parse_fx_pair(raw_upper: str, raw: str) -> dict[str, Any] | None:
    m = _FX_PAIR_RE.fullmatch(raw_upper.strip())
    if not m:
        return None
    base, quote = m.groups()
    if base not in _KNOWN_CCY or quote not in _KNOWN_CCY:
        return None
    return {
        "market": "FX",
        "asset": "FOREX",
        "symbol": f"{base}{quote}",
        "currency": quote,
        "yahoo": f"{base}{quote}=X",
        "raw": raw,
        "vendor": {
            "base": base,
            "quote": quote,
        },
    }

def _cn_yahoo_suffix(code6: str) -> str:
    """
    Best-effort mapping of CN code -> Yahoo suffix.
    If get_exchange is available, use it; else fall back on prefix heuristics.
    """
    if get_exchange is not None:
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

def instrument_key(inst: dict[str, Any]) -> str:
    """Stable unique key for caching/storage."""
    market = str(inst.get("market") or "UNKNOWN").upper()
    asset = str(inst.get("asset") or "UNKNOWN").upper()
    sym = str(inst.get("symbol") or "")
    return f"{market}:{asset}:{sym}"

def parse_instrument(code: str) -> dict[str, Any]:
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

    # Preserve an uppercase canonical string for non-equity pattern parsing.
    raw_upper = raw.upper().strip().replace(" ", "")
    # OCC options are unambiguous and should be detected early.
    occ = _parse_occ_option(raw_upper)
    if occ is not None:
        return occ
    # Futures-like roots are non-numeric so they won't clash with CN equity digits.
    cn_future = _parse_cn_future(raw_upper, raw)
    if cn_future is not None:
        return cn_future
    us_future = _parse_us_future(raw_upper, raw)
    if us_future is not None:
        return us_future

    # Keep a case-preserving copy for suffix detection
    s_upper = s.upper()

    # -------------------------
    # -------------------------
    is_explicit_hk = s_upper.startswith(tuple(p.upper() for p in HK_PREFIXES)) or s_upper.endswith(".HK")
    is_explicit_us = s_upper.startswith(tuple(p.upper() for p in US_PREFIXES))
    # -------------------------
    # Crypto detection (best effort)
    # -------------------------
    # If it contains common crypto quote tokens, treat as crypto.
    # Accept formats: BTCUSD, BTCUSDT, BTC/USD, BTC-USDT, BTC-USD
    crypto_norm = s_upper.replace("/", "")
    crypto_bases = (
        "BTC", "ETH", "SOL", "XRP", "DOGE", "ADA", "BNB",
        "LTC", "TRX", "AVAX",
    )
    has_crypto_base = any(crypto_norm.startswith(base) for base in crypto_bases)
    has_crypto_quote = crypto_norm.endswith(("USDT", "USDC", "BUSD", "USD", "BTC", "ETH"))
    crypto_hint = has_crypto_base and (
        "/" in raw or "-" in raw or has_crypto_quote
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

    fx = _parse_fx_pair(raw_upper, raw)
    if fx is not None:
        return fx

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
