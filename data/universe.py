# data/universe.py

from __future__ import annotations

import json
import threading
import time
import warnings
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
_MIN_REASONABLE_UNIVERSE_SIZE = 1200
_TARGET_FULL_UNIVERSE_SIZE = 4500
_FALLBACK_PER_CALL_TIMEOUT_S = 15.0
_MIN_MIX_BASELINE = 80
_MIN_EXCHANGE_RETENTION_RATIO = 0.20
_MIN_TOTAL_RETENTION_RATIO = 0.60
_MIN_MULTI_EXCHANGE_SIZE = 600


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


def persist_runtime_universe_codes(
    codes: list[str] | tuple[str, ...] | set[str],
    *,
    source: str = "runtime_refresh",
) -> dict:
    """Persist a continuously updated universe snapshot to disk.

    Merges incoming codes with the existing file so transient partial
    refreshes do not erase previously discovered symbols.
    """
    incoming = _validate_codes(list(codes or []))
    existing = load_universe()
    existing_codes = _validate_codes(list(existing.get("codes") or []))
    merged = _validate_codes(existing_codes + incoming)
    payload = {
        "codes": merged,
        "updated_ts": time.time(),
        "updated_at": datetime.now().isoformat(),
        "source": str(source or existing.get("source") or "runtime_refresh"),
    }
    if merged:
        save_universe(payload)
    return payload


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


def _exchange_counts(codes: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {"SSE": 0, "SZSE": 0, "BSE": 0}
    for code in codes:
        ex = str(get_exchange(code)).upper()
        if ex in counts:
            counts[ex] += 1
    return counts


def _should_merge_partial_refresh(
    fresh_codes: list[str],
    cached_codes: list[str],
) -> tuple[bool, str]:
    fresh = _validate_codes(list(fresh_codes or []))
    cached = _validate_codes(list(cached_codes or []))

    if not fresh:
        return True, "fresh universe empty"
    if not cached:
        return False, ""

    fresh_n = len(fresh)
    cached_n = len(cached)

    if fresh_n < _MIN_REASONABLE_UNIVERSE_SIZE:
        return True, (
            f"fresh universe too small ({fresh_n} < {_MIN_REASONABLE_UNIVERSE_SIZE})"
        )

    if cached_n >= _MIN_REASONABLE_UNIVERSE_SIZE:
        min_keep = int(max(
            _MIN_REASONABLE_UNIVERSE_SIZE,
            float(cached_n) * float(_MIN_TOTAL_RETENTION_RATIO),
        ))
        if fresh_n < min_keep:
            return True, (
                f"fresh universe dropped sharply ({fresh_n} < {min_keep})"
            )

    fresh_mix = _exchange_counts(fresh)
    cached_mix = _exchange_counts(cached)
    if fresh_n >= _MIN_MULTI_EXCHANGE_SIZE and cached_n >= _MIN_MULTI_EXCHANGE_SIZE:
        for ex, cached_count in cached_mix.items():
            if cached_count < _MIN_MIX_BASELINE:
                continue
            fresh_count = int(fresh_mix.get(ex, 0))
            min_expected = int(max(10, cached_count * _MIN_EXCHANGE_RETENTION_RATIO))
            if fresh_count < min_expected:
                return True, (
                    f"exchange coverage collapsed for {ex} ({fresh_count} < {min_expected})"
                )

    return False, ""


def _stabilize_universe_refresh(
    fresh_codes: list[str],
    cached_codes: list[str],
) -> tuple[list[str], str]:
    fresh = _validate_codes(list(fresh_codes or []))
    cached = _validate_codes(list(cached_codes or []))
    should_merge, reason = _should_merge_partial_refresh(fresh, cached)
    if not should_merge:
        return fresh, ""
    merged = _validate_codes(cached + fresh)
    if not merged:
        return fresh, reason
    return merged, reason


def _fallback_codes() -> list[str]:
    """Build robust offline fallback universe.

    Uses shared fallback_stocks module to avoid circular import
    with data.discovery.
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
    """Decide if AkShare/Eastmoney is worth trying.

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

    log.info(
        "Skipping AkShare universe fetch: Eastmoney unreachable "
        f"(mode=DIRECT)"
    )
    return False


def _find_first_column(candidates: list[str], columns: list[str]) -> str | None:
    colset = set(columns)
    for c in candidates:
        if c in colset:
            return c
    return None


def _rank_codes_by_liquidity(df) -> list[str]:
    """Rank universe candidates by liquidity/size when columns are available.
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


def _extract_codes_from_any(payload: object) -> list[str]:
    """Extract valid codes from DataFrame/list/dict style payloads."""
    if payload is None:
        return []

    # DataFrame-like
    if hasattr(payload, "columns") and hasattr(payload, "empty"):
        return _extract_codes_from_df(payload)

    # List/Tuple/Set of rows or strings.
    if isinstance(payload, (list, tuple, set)):
        raw_codes: list[str] = []
        for row in payload:
            if isinstance(row, dict):
                for key in ("code", "symbol", "stock_code", "ticker", "secid"):
                    if key in row:
                        raw_codes.append(str(row.get(key)))
            else:
                raw_codes.append(str(row))
        return _validate_codes(raw_codes)

    # Dict payload with common list fields.
    if isinstance(payload, dict):
        for key in ("codes", "data", "items", "list"):
            if key in payload:
                nested = _extract_codes_from_any(payload.get(key))
                if nested:
                    return nested
        return []

    return []


def _to_datetime_coerce(values):
    """Datetime parsing helper that suppresses noisy infer-format warnings."""
    if not _HAS_PANDAS:
        return values
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"^Could not infer format, so each element will be parsed individually.*",
            category=UserWarning,
        )
        try:
            # pandas>=2.0: mixed parser avoids infer-format warning path.
            return pd.to_datetime(values, errors="coerce", format="mixed")
        except TypeError:
            return pd.to_datetime(values, errors="coerce")


def _try_akshare_fetch(timeout: int = 20) -> list[str] | None:
    """Try to fetch stock universe from AkShare.

    Uses ThreadPoolExecutor for timeout-safe concurrent fetching.
    """
    try:
        import akshare as ak
    except ImportError:
        log.warning("AkShare not installed")
        return None

    from concurrent.futures import ThreadPoolExecutor
    from concurrent.futures import TimeoutError as FuturesTimeout

    log.info(
        "AkShare fetch: base_timeout=%.1fs",
        float(timeout)
    )

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

    def _call_with_timeout(api_name: str, api_fn, call_timeout: float):
        if not callable(api_fn):
            return None
        try:
            log.info("Fetching universe from AkShare fallback (%s)...", api_name)
            with ThreadPoolExecutor(max_workers=1) as ex:
                fut = ex.submit(api_fn)
                return fut.result(timeout=max(1.0, float(call_timeout)))
        except FuturesTimeout:
            log.warning(
                "AkShare fallback %s timed out after %.1fs",
                api_name,
                float(call_timeout),
            )
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
            result = fut.result(timeout=float(timeout))
            if result:
                return result
    except FuturesTimeout:
        log.warning(f"AkShare spot fetch timed out after {float(timeout):.1f}s")
    except Exception as exc:
        log.warning(f"AkShare spot fetch failed: {type(exc).__name__}: {exc}")

    # Try fallback endpoints with per-call timeout and early stop once we
    # already have a reasonably full market universe.
    fallback_calls = [
        ("stock_info_a_code_name", getattr(ak, "stock_info_a_code_name", None)),
        ("stock_zh_a_name_code", getattr(ak, "stock_zh_a_name_code", None)),
        ("stock_info_sh_name_code", getattr(ak, "stock_info_sh_name_code", None)),
        ("stock_info_sz_name_code", getattr(ak, "stock_info_sz_name_code", None)),
        ("stock_info_bj_name_code", getattr(ak, "stock_info_bj_name_code", None)),
        # Slow endpoint; keep it last and only try if needed.
        ("stock_zh_a_spot", getattr(ak, "stock_zh_a_spot", None)),
    ]

    merged_codes: list[str] = []
    deadline = time.monotonic() + float(timeout)
    for api_name, api_fn in fallback_calls:
        merged = _validate_codes(merged_codes)
        if len(merged) >= _MIN_REASONABLE_UNIVERSE_SIZE:
            log.info(
                "Universe fallback reached %s valid codes; stopping early at %s",
                len(merged),
                api_name,
            )
            return merged

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break

        call_timeout = min(float(_FALLBACK_PER_CALL_TIMEOUT_S), float(remaining))
        payload = _call_with_timeout(api_name, api_fn, call_timeout)
        if payload is None:
            continue
        codes = _extract_codes_from_any(payload)
        if codes:
            log.info(
                "AkShare fallback %s returned %s valid codes",
                api_name,
                len(codes),
            )
            merged_codes.extend(codes)

    merged = _validate_codes(merged_codes)
    if merged:
        return merged

    log.warning("No valid stock codes from any AkShare universe endpoint")
    return None


def get_universe_codes(
    force_refresh: bool = False,
    max_age_hours: float = 12.0,
) -> list[str]:
    """Return all known A-share stock codes.

    FIX: Always tries AkShare first, regardless of network detection.
    Falls back to cache, then CONFIG.stock_pool.
    """
    data = load_universe()
    now = time.time()
    last_ts = _parse_updated_ts(data)
    cached_codes = _validate_codes(data.get("codes") or [])
    cached_source = str(data.get("source", "") or "").strip().lower()
    stale = (now - last_ts) > max_age_hours * 3600.0
    cache_is_thin_fallback = (
        cached_source == "fallback"
        and len(cached_codes) < _MIN_REASONABLE_UNIVERSE_SIZE
    )

    if not force_refresh and cached_codes and not stale and not cache_is_thin_fallback:
        log.info("Using cached universe: %d codes", len(cached_codes))
        return cached_codes
    if cache_is_thin_fallback and not force_refresh:
        log.info(
            "Cached universe is fallback/thin (%s codes); attempting refresh",
            len(cached_codes),
        )

    # Try AkShare
    can_try_direct = _can_use_akshare()
    timeout = 25 if can_try_direct else 12  # Standard timeout

    log.info(
        "Universe refresh: can_try_direct=%s, timeout=%.1fs",
        can_try_direct, timeout
    )
    
    fresh_codes = _try_akshare_fetch(timeout=int(timeout))

    if fresh_codes:
        stabilized_codes, reason = _stabilize_universe_refresh(
            fresh_codes,
            cached_codes,
        )
        if reason:
            log.warning(
                "Universe refresh stabilized: %s (fresh=%s, stabilized=%s)",
                reason,
                len(_validate_codes(fresh_codes)),
                len(stabilized_codes),
            )
        fresh_codes = stabilized_codes

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
    """Refresh universe from AkShare.

    FIX: Removed network detection check - just try to fetch.
    """
    existing = load_universe()
    existing_codes = existing.get("codes") or []

    can_try_direct = _can_use_akshare()
    timeout = 25 if can_try_direct else 12  # Standard timeout

    log.info(
        "Universe refresh: can_try_direct=%s, timeout=%.1fs",
        can_try_direct, timeout
    )
    
    fresh_codes = _try_akshare_fetch(timeout=int(timeout))

    if fresh_codes:
        existing_codes = _validate_codes(existing_codes)
        stabilized_codes, reason = _stabilize_universe_refresh(
            fresh_codes,
            existing_codes,
        )
        if reason:
            log.warning(
                "Universe refresh stabilized: %s (fresh=%s, stabilized=%s)",
                reason,
                len(_validate_codes(fresh_codes)),
                len(stabilized_codes),
            )
        fresh_codes = stabilized_codes

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
    """Return codes of stocks listed within the last N days.

    Uses atomic snapshot reads and writes under lock for thread safety.
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

    # Use ThreadPoolExecutor for timeout-safe fetching
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
                        parsed = _to_datetime_coerce(df[col])
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
                    dates = _to_datetime_coerce(df[date_col])
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
        log.warning("get_new_listings failed: %s", exc)

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
