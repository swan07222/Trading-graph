# data/universe.py

from __future__ import annotations

import json
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
_new_listings_cache: dict[str, object] = {
    "ts": 0.0,
    "days": 0,
    "codes": [],
}


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
    """
    Build robust offline fallback universe.

    FIX Bug 9: Uses shared fallback_stocks module instead of importing
    from data.discovery, which would create a circular import.
    """
    base = _validate_codes(getattr(CONFIG, "stock_pool", []))

    try:
        from data.fallback_stocks import FALLBACK_STOCK_LIST
        extra = [code for code, _name in FALLBACK_STOCK_LIST]
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

    eastmoney_ok = bool(getattr(env, "eastmoney_ok", False))
    is_china_direct = bool(getattr(env, "is_china_direct", False))
    if eastmoney_ok:
        return True
    if is_china_direct:
        # Probe can be stale/noisy; direct CN path is still worth trying.
        log.info("Eastmoney probe unavailable, retrying AkShare on direct CN network")
        return True

    mode = "VPN" if bool(getattr(env, "is_vpn_active", False)) else "DIRECT"
    log.info(
        "Skipping AkShare universe fetch: Eastmoney unreachable "
        f"(mode={mode})"
    )
    return False


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
        ["code", "symbol", "secid", "stock_code", "ticker"],
        cols,
    )
    if code_col is None and cols:
        code_col = cols[0]
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
        ["amount", "turnover", "amount_zh"],
        cols,
    )
    volume_col = _find_first_column(
        ["volume", "vol"],
        cols,
    )
    cap_col = _find_first_column(
        ["market_cap", "mktcap", "total_mv"],
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


def _extract_codes_from_df(df) -> list[str]:
    """Extract valid 6-digit A-share codes from a generic DataFrame."""
    if df is None or getattr(df, "empty", True):
        return []

    cols = list(getattr(df, "columns", []))
    preferred_cols = (
        "code",
        "symbol",
        "secid",
        "stock_code",
        "ticker",
    )

    for col in preferred_cols:
        if col not in cols:
            continue
        try:
            raw = (
                df[col]
                .astype(str)
                .str.extract(r"(\d{6})")[0]
                .dropna()
                .tolist()
            )
            codes = _validate_codes(raw)
            if codes:
                return codes
        except Exception:
            continue

    # Fallback: scan first few columns for 6-digit values.
    raw_codes: list[str] = []
    for col in cols[:8]:
        try:
            part = (
                df[col]
                .astype(str)
                .str.extract(r"(\d{6})")[0]
                .dropna()
                .tolist()
            )
            if part:
                raw_codes.extend(part)
        except Exception:
            continue
    return _validate_codes(raw_codes)


def _try_akshare_fetch(timeout: int = 20) -> list[str] | None:
    """
    Try to fetch stock universe from AkShare.

    FIX Bug 10: Uses ThreadPoolExecutor for timeout instead of
    socket.setdefaulttimeout which is process-global and racy.
    """
    try:
        import akshare as ak
    except ImportError:
        log.warning("AkShare not installed")
        return None

    from concurrent.futures import ThreadPoolExecutor
    from concurrent.futures import TimeoutError as FuturesTimeout

    def _do_spot_fetch():
        log.info("Fetching universe from AkShare (stock_zh_a_spot_em)...")
        df_spot = ak.stock_zh_a_spot_em()
        if df_spot is not None and not df_spot.empty:
            codes = _rank_codes_by_liquidity(df_spot)
            if not codes:
                codes = _extract_codes_from_df(df_spot)
            if codes:
                log.info(f"AkShare spot returned {len(codes)} valid codes")
                return codes
        else:
            log.warning("AkShare spot endpoint returned empty data")
        return None

    def _do_fallback_fetch():
        fallback_calls = [
            ("stock_info_a_code_name", getattr(ak, "stock_info_a_code_name", None)),
            ("stock_zh_a_name_code", getattr(ak, "stock_zh_a_name_code", None)),
        ]
        for api_name, api_fn in fallback_calls:
            if not callable(api_fn):
                continue
            try:
                log.info(f"Fetching universe from AkShare fallback ({api_name})...")
                df_codes = api_fn()
                codes = _extract_codes_from_df(df_codes)
                if codes:
                    log.info(
                        "AkShare fallback %s returned %s valid codes",
                        api_name,
                        len(codes),
                    )
                    return codes
            except Exception as exc:
                log.warning(
                    "AkShare fallback %s failed: %s: %s",
                    api_name,
                    type(exc).__name__,
                    exc,
                )
        return None

    # Try primary endpoint with timeout
    try:
        with ThreadPoolExecutor(max_workers=1) as ex:
            fut = ex.submit(_do_spot_fetch)
            result = fut.result(timeout=timeout)
            if result:
                return result
    except FuturesTimeout:
        log.warning(f"AkShare spot fetch timed out after {timeout}s")
    except Exception as exc:
        log.warning(f"AkShare spot fetch failed: {type(exc).__name__}: {exc}")

    # Try fallback endpoints with timeout
    try:
        with ThreadPoolExecutor(max_workers=1) as ex:
            fut = ex.submit(_do_fallback_fetch)
            result = fut.result(timeout=timeout)
            if result:
                return result
    except FuturesTimeout:
        log.warning(f"AkShare fallback fetch timed out after {timeout}s")
    except Exception as exc:
        log.warning(f"AkShare fallback fetch failed: {type(exc).__name__}: {exc}")

    log.warning("No valid stock codes from any AkShare universe endpoint")
    return None


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

    # Try AkShare regardless of probe result; use shorter timeout when probe
    # says Eastmoney is likely blocked.
    can_try_direct = _can_use_akshare()
    fresh_codes = _try_akshare_fetch(timeout=20 if can_try_direct else 8)

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

    can_try_direct = _can_use_akshare()
    fresh_codes = _try_akshare_fetch(timeout=20 if can_try_direct else 8)

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


def get_new_listings(
    days: int = 60,
    force_refresh: bool = False,
    max_age_seconds: float = 5.0,
) -> list[str]:
    """
    Return codes of stocks listed within the last N days.

    FIX Bug 14: Atomic snapshot reads and writes under lock.
    """
    now = time.time()

    # Atomic snapshot read
    with _universe_lock:
        try:
            cached_ts = float(_new_listings_cache.get("ts", 0.0) or 0.0)
            cached_days = int(_new_listings_cache.get("days", 0) or 0)
            cached_codes = _validate_codes(
                list(_new_listings_cache.get("codes", []) or [])
            )
        except Exception:
            cached_ts = 0.0
            cached_days = 0
            cached_codes = []

    if (
        not force_refresh
        and cached_codes
        and cached_days == int(days)
        and (now - cached_ts) <= max(0.0, float(max_age_seconds))
    ):
        return cached_codes

    if not _HAS_PANDAS:
        return []

    try:
        import akshare as ak
    except ImportError:
        return []

    cutoff = date.today() - timedelta(days=days)

    # FIX Bug 10: Use ThreadPoolExecutor for timeout
    from concurrent.futures import ThreadPoolExecutor
    from concurrent.futures import TimeoutError as FuturesTimeout

    def _fetch_new_listings_inner() -> list[str] | None:
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

            code_col = _find_first_column(
                ["code", "symbol", "stock_code", "secid", "ticker"],
                list(df.columns),
            )
            if code_col is None:
                for col in list(df.columns)[:8]:
                    try:
                        probe = (
                            df[col].astype(str).str.extract(r"(\d{6})")[0].dropna()
                        )
                    except Exception:
                        continue
                    if not probe.empty:
                        code_col = col
                        break
            if code_col is None:
                continue

            date_col = _find_first_column(
                ["listing_date", "list_date", "ipo_date", "date"],
                list(df.columns),
            )
            if date_col is None and _HAS_PANDAS:
                for col in list(df.columns)[:10]:
                    try:
                        parsed = pd.to_datetime(df[col], errors="coerce")
                    except Exception:
                        continue
                    valid = int(parsed.notna().sum())
                    if valid >= max(3, int(len(parsed) * 0.2)):
                        date_col = col
                        break

            raw_codes = (
                df[code_col].astype(str).str.extract(r"(\d+)")[0].dropna().tolist()
            )
            codes = _validate_codes(raw_codes)

            if date_col and codes:
                try:
                    dates = pd.to_datetime(df[date_col], errors="coerce")
                    mask = dates >= pd.Timestamp(cutoff)
                    filtered = _validate_codes(
                        df.loc[mask, code_col]
                        .astype(str)
                        .str.extract(r"(\d+)")[0]
                        .dropna()
                        .tolist()
                    )
                    if filtered:
                        codes = filtered
                except Exception:
                    pass

            log.info(f"New listings (last {days} days): {len(codes)} codes")
            return codes

        return None

    result_codes: list[str] | None = None
    try:
        with ThreadPoolExecutor(max_workers=1) as ex:
            fut = ex.submit(_fetch_new_listings_inner)
            result_codes = fut.result(timeout=15)
    except FuturesTimeout:
        log.warning("New listings fetch timed out")
    except Exception as exc:
        log.debug(f"get_new_listings failed: {exc}")

    if result_codes:
        # Atomic cache write
        with _universe_lock:
            _new_listings_cache["ts"] = time.time()
            _new_listings_cache["days"] = int(days)
            _new_listings_cache["codes"] = list(result_codes)
        return result_codes

    if cached_codes:
        return cached_codes
    return []
