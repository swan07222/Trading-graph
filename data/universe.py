# data/universe.py

from __future__ import annotations

import json
import socket
import threading
import time
from datetime import date, datetime, timedelta
from pathlib import Path

from config.settings import CONFIG
from core.constants import get_exchange
from core.network import get_network_env
from utils.logger import get_logger

log = get_logger(__name__)

try:
    import pandas as pd
    _HAS_PANDAS = True
except ImportError:
    pd = None
    _HAS_PANDAS = False

_universe_lock = threading.Lock()

def _universe_path() -> Path:
    return Path(CONFIG.data_dir) / "stock_universe.json"

def load_universe() -> dict:
    """Load universe from disk."""
    path = _universe_path()
    if not path.exists():
        return {}
    try:
        text = path.read_text(encoding="utf-8")
        data = json.loads(text)
        if not isinstance(data, dict):
            return {}
        return data
    except (json.JSONDecodeError, OSError) as exc:
        log.warning(f"Failed to load universe file: {exc}")
        return {}

def save_universe(data: dict) -> None:
    """Atomically save universe to disk."""
    path = _universe_path()
    with _universe_lock:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            import os
            tmp = path.with_suffix(".json.tmp")
            json_str = json.dumps(data, ensure_ascii=False, indent=2)
            with open(tmp, "w", encoding="utf-8") as f:
                f.write(json_str)
                f.flush()
                try:
                    os.fsync(f.fileno())
                except OSError:
                    pass
            tmp.replace(path)
        except OSError as exc:
            log.warning(f"Failed to save universe: {exc}")

def _parse_updated_ts(data: dict) -> float:
    raw = data.get("updated_ts")
    if raw is None:
        return 0.0
    try:
        return float(raw)
    except (ValueError, TypeError):
        return 0.0

def _validate_codes(codes: list) -> list[str]:
    """Validate stock codes."""
    validated: list[str] = []
    for c in codes:
        s = str(c).strip()
        if not s.isdigit():
            continue
        code6 = s.zfill(6)
        if get_exchange(code6) == "UNKNOWN":
            continue
        validated.append(code6)
    return sorted(set(validated))

def _fallback_codes() -> list[str]:
    """Build robust offline fallback universe."""
    base = _validate_codes(getattr(CONFIG, "stock_pool", []))

    # Reuse curated discovery fallback list when available.
    try:
        from data.discovery import UniversalStockDiscovery

        extra = [
            s.code
            for s in UniversalStockDiscovery._get_fallback_stocks()  # static helper
            if getattr(s, "code", None)
        ]
        base.extend(extra)
    except Exception:
        pass

    return _validate_codes(base)

def _can_use_akshare() -> bool:
    """
    Decide if AkShare/Eastmoney is worth trying.

    AkShare is backed by EastMoney; if EastMoney probe fails, skip it
    to avoid repeated connection-aborted warnings.
    """
    try:
        env = get_network_env()
    except Exception:
        return True

    if not bool(getattr(env, "eastmoney_ok", False)):
        mode = "VPN" if bool(getattr(env, "is_vpn_active", False)) else "DIRECT"
        log.info(
            "Skipping AkShare universe fetch: Eastmoney unreachable "
            f"(mode={mode})"
        )
        return False

    return bool(env.is_china_direct)

def _find_first_column(candidates: list[str], columns: list[str]) -> str | None:
    colset = set(columns)
    for c in candidates:
        if c in colset:
            return c
    return None

def _rank_codes_by_liquidity(df) -> list[str]:
    """
    Rank universe candidates by liquidity/size when columns are available.
    Falls back to the original code order if ranking fields are missing.
    """
    cols = list(getattr(df, "columns", []))
    code_col = _find_first_column(
        ["代码", "证券代码", "浠ｇ爜", "璇佸埜浠ｇ爜", "code"],
        cols,
    )
    if code_col is None:
        return []
    if not _HAS_PANDAS:
        raw_codes = (
            df[code_col].astype(str).str.extract(r"(\d+)")[0].dropna().tolist()
        )
        return _validate_codes(raw_codes)

    ranked = df.copy()
    ranked["__code__"] = (
        ranked[code_col]
        .astype(str)
        .str.extract(r"(\d+)")[0]
        .fillna("")
        .astype(str)
        .str.zfill(6)
    )
    ranked = ranked[ranked["__code__"].str.fullmatch(r"\d{6}")]

    amount_col = _find_first_column(
        ["成交额", "amount", "turnover", "amount_zh"],
        cols,
    )
    volume_col = _find_first_column(
        ["成交量", "volume", "vol"],
        cols,
    )
    cap_col = _find_first_column(
        ["总市值", "market_cap", "mktcap"],
        cols,
    )

    if amount_col:
        ranked["__amount__"] = pd.to_numeric(ranked[amount_col], errors="coerce").fillna(0.0)
    else:
        ranked["__amount__"] = 0.0
    if volume_col:
        ranked["__volume__"] = pd.to_numeric(ranked[volume_col], errors="coerce").fillna(0.0)
    else:
        ranked["__volume__"] = 0.0
    if cap_col:
        ranked["__cap__"] = pd.to_numeric(ranked[cap_col], errors="coerce").fillna(0.0)
    else:
        ranked["__cap__"] = 0.0

    if (ranked["__amount__"].sum() + ranked["__volume__"].sum() + ranked["__cap__"].sum()) <= 0:
        return _validate_codes(ranked["__code__"].tolist())

    ranked = ranked.sort_values(
        by=["__amount__", "__volume__", "__cap__", "__code__"],
        ascending=[False, False, False, True],
    )
    return _validate_codes(ranked["__code__"].tolist())

def _try_akshare_fetch(timeout: int = 20) -> list[str] | None:
    """
    Try to fetch stock universe from AkShare.

    FIX: Always tries regardless of network detection.
    Returns None on failure, valid codes on success.
    """
    try:
        import akshare as ak
    except ImportError:
        log.warning("AkShare not installed")
        return None

    old_timeout = socket.getdefaulttimeout()
    socket.setdefaulttimeout(timeout)

    try:
        log.info("Fetching universe from AkShare (stock_zh_a_spot_em)...")
        df = ak.stock_zh_a_spot_em()

        if df is None or df.empty:
            log.warning("AkShare returned empty DataFrame")
            return None

        code_col = None
        for c in ("代码", "code", "证券代码"):
            if c in df.columns:
                code_col = c
                break

        if code_col is None:
            log.warning(f"No code column found. Columns: {list(df.columns)[:10]}")
            return None

        codes = _rank_codes_by_liquidity(df)
        if not codes:
            raw_codes = df[code_col].astype(str).str.extract(r"(\d+)")[0].dropna().tolist()
            codes = _validate_codes(raw_codes)

        if codes:
            log.info(f"AkShare returned {len(codes)} valid codes")
            return codes
        else:
            log.warning("No valid codes after filtering")
            return None

    except Exception as exc:
        log.warning(f"AkShare fetch failed: {type(exc).__name__}: {exc}")
        return None
    finally:
        socket.setdefaulttimeout(old_timeout)

def get_universe_codes(
    force_refresh: bool = False,
    max_age_hours: float = 12.0,
) -> list[str]:
    """
    Return all known A-share stock codes.

    FIX: Always tries AkShare first, regardless of network detection.
    Falls back to cache, then CONFIG.stock_pool.
    """
    data = load_universe()
    now = time.time()
    last_ts = _parse_updated_ts(data)
    cached_codes = _validate_codes(data.get("codes") or [])
    stale = (now - last_ts) > max_age_hours * 3600.0

    if not force_refresh and cached_codes and not stale:
        log.debug(f"Using cached universe: {len(cached_codes)} codes")
        return cached_codes

    # Network-aware source choice: avoid AkShare timeouts when Eastmoney is blocked.
    fresh_codes = _try_akshare_fetch() if _can_use_akshare() else None

    if fresh_codes:
        out = {
            "codes": fresh_codes,
            "updated_ts": time.time(),
            "updated_at": datetime.now().isoformat(),
            "source": "akshare_spot_em",
        }
        save_universe(out)
        return fresh_codes

    # AkShare failed - use cache if available
    if cached_codes:
        age_hours = (now - last_ts) / 3600.0
        log.warning(
            f"AkShare failed, using cached universe "
            f"({len(cached_codes)} codes, {age_hours:.1f}h old)"
        )
        return cached_codes

    # Last resort: robust static fallback universe
    fallback = _fallback_codes()
    if fallback:
        log.warning(f"Using fallback universe ({len(fallback)} codes)")
        return fallback

    log.error("No stock codes available from any source!")
    return []

def refresh_universe() -> dict:
    """
    Refresh universe from AkShare.

    FIX: Removed network detection check - just try to fetch.
    """
    existing = load_universe()
    existing_codes = existing.get("codes") or []

    fresh_codes = _try_akshare_fetch() if _can_use_akshare() else None

    if fresh_codes:
        out = {
            "codes": fresh_codes,
            "updated_ts": time.time(),
            "updated_at": datetime.now().isoformat(),
            "source": "akshare_spot_em",
        }
        save_universe(out)
        log.info(f"Universe refreshed: {len(fresh_codes)} codes")
        return out

    if existing_codes:
        log.info(f"Refresh failed, keeping existing {len(existing_codes)} codes")
        return existing

    fallback = _fallback_codes()
    if fallback:
        out = {
            "codes": fallback,
            "updated_ts": time.time(),
            "updated_at": datetime.now().isoformat(),
            "source": "fallback",
        }
        save_universe(out)
        return out

    return {"codes": [], "updated_ts": time.time(), "source": "empty"}

def get_new_listings(days: int = 60, force_refresh: bool = False) -> list[str]:
    """Return codes of stocks listed within the last N days."""
    if not _HAS_PANDAS:
        return []

    try:
        import akshare as ak
    except ImportError:
        return []

    cutoff = date.today() - timedelta(days=days)
    old_timeout = socket.getdefaulttimeout()
    socket.setdefaulttimeout(10)

    try:
        for fn_name in ("stock_zh_a_new_em", "stock_zh_a_new"):
            fn = getattr(ak, fn_name, None)
            if not callable(fn):
                continue

            try:
                df = fn()
            except Exception:
                continue

            if df is None or df.empty:
                continue

            code_col = None
            for c in ("代码", "证券代码", "stock_code", "code"):
                if c in df.columns:
                    code_col = c
                    break
            if code_col is None:
                continue

            date_col = None
            for c in ("上市日期", "listing_date", "list_date"):
                if c in df.columns:
                    date_col = c
                    break

            raw_codes = df[code_col].astype(str).str.extract(r"(\d+)")[0].dropna().tolist()
            codes = _validate_codes(raw_codes)

            if date_col and codes:
                try:
                    dates = pd.to_datetime(df[date_col], errors="coerce")
                    mask = dates >= pd.Timestamp(cutoff)
                    filtered = _validate_codes(
                        df.loc[mask, code_col].astype(str).str.extract(r"(\d+)")[0].dropna().tolist()
                    )
                    if filtered:
                        codes = filtered
                except Exception:
                    pass

            log.info(f"New listings (last {days} days): {len(codes)} codes")
            return codes

    except Exception as exc:
        log.debug(f"get_new_listings failed: {exc}")
    finally:
        socket.setdefaulttimeout(old_timeout)

    return []
