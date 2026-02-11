# data/universe.py
"""
Stock Universe Management

Maintains a cached JSON file of all known A-share stock codes.
Network-aware: refreshes from AkShare on China direct IP,
uses cached data on VPN/foreign connections.
"""
from __future__ import annotations

import json
import time
import threading
from dataclasses import dataclass
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Dict, List, Optional

from config.settings import CONFIG
from core.constants import get_exchange
from utils.logger import get_logger

log = get_logger(__name__)

# Module-level lock for file writes (protects within this process;
# cross-process safety would require fcntl/msvcrt file locking).
_universe_lock = threading.Lock()


def _universe_path() -> Path:
    return Path(CONFIG.data_dir) / "stock_universe.json"


def load_universe() -> Dict:
    """Load universe from disk. Returns empty dict on any error."""
    path = _universe_path()
    if not path.exists():
        return {}
    try:
        text = path.read_text(encoding="utf-8")
        data = json.loads(text)
        if not isinstance(data, dict):
            log.warning("Universe file is not a JSON object — ignoring")
            return {}
        return data
    except (json.JSONDecodeError, OSError) as exc:
        log.warning(f"Failed to load universe file: {exc}")
        return {}


def save_universe(data: Dict) -> None:
    """
    Atomically save universe to disk.

    Uses write-to-temp-then-rename for crash safety.
    Thread-safe within this process via ``_universe_lock``.
    """
    path = _universe_path()
    with _universe_lock:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".json.tmp")
            tmp.write_text(
                json.dumps(data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            tmp.replace(path)
        except OSError as exc:
            log.warning(f"Failed to save universe: {exc}")
            # Clean up temp file if rename failed
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass


def _parse_updated_ts(data: Dict) -> float:
    """
    Safely extract ``updated_ts`` as a float.

    Handles missing keys, None, empty strings, and non-numeric values.
    """
    raw = data.get("updated_ts")
    if raw is None:
        return 0.0
    try:
        return float(raw)
    except (ValueError, TypeError):
        return 0.0


def _validate_codes(codes: List) -> List[str]:
    """
    Normalise and validate a list of stock codes.

    - Keeps only digit-only strings.
    - Zero-pads to 6 digits.
    - Verifies each code maps to a known exchange.
    """
    validated: List[str] = []
    for c in codes:
        s = str(c).strip()
        if not s.isdigit():
            continue
        code6 = s.zfill(6)
        if get_exchange(code6) == "UNKNOWN":
            continue
        validated.append(code6)
    return sorted(set(validated))


def get_universe_codes(
    force_refresh: bool = False,
    max_age_hours: float = 12.0,
) -> List[str]:
    """
    Return all known marketed stock codes.

    Refresh logic:
    - China direct: refresh from AkShare if data is stale or forced.
    - VPN/foreign: use cached universe; warn if stale; fall back to
      CONFIG.stock_pool if cache is empty.

    Args:
        force_refresh:  Force a network refresh regardless of cache age.
        max_age_hours:  Maximum cache age before considering stale.

    Returns:
        List of 6-digit stock code strings.
    """
    data = load_universe()
    now = time.time()
    last_ts = _parse_updated_ts(data)
    stale = (now - last_ts) > max_age_hours * 3600.0

    if force_refresh or stale or not data.get("codes"):
        refreshed = refresh_universe()
        if refreshed.get("codes"):
            data = refreshed
        elif stale and data.get("codes"):
            log.warning(
                f"Universe is stale ({(now - last_ts) / 3600:.1f}h old) "
                f"but refresh failed — using cached data"
            )

    codes = _validate_codes(data.get("codes") or [])

    if not codes:
        log.info("No universe codes available — falling back to CONFIG.stock_pool")
        codes = _validate_codes(getattr(CONFIG, "stock_pool", []))

    return codes


def refresh_universe() -> Dict:
    """
    Refresh the universe from AkShare (China direct only).

    On VPN/foreign connections, returns the existing cached data
    without destroying it.

    Returns:
        Dict with keys: ``codes``, ``updated_ts``, ``updated_at``, ``source``.
    """
    from core.network import get_network_env

    env = get_network_env()
    existing = load_universe()
    existing_codes = existing.get("codes") or []

    if not env.is_china_direct:
        log.info(
            "Universe refresh skipped (VPN/foreign). "
            f"Cached universe has {len(existing_codes)} codes."
        )
        return existing if existing_codes else {
            "codes": [],
            "updated_ts": time.time(),
            "source": "empty",
        }

    # ---- China direct: refresh from AkShare -------------------------
    try:
        import akshare as ak

        df = ak.stock_zh_a_spot_em()
        if df is None or df.empty:
            log.warning("AkShare returned empty spot data — keeping existing universe")
            return existing

        # Find the code column
        code_col = None
        for c in ("代码", "code", "证券代码"):
            if c in df.columns:
                code_col = c
                break

        if code_col is None:
            log.warning(
                f"AkShare spot data has no recognised code column. "
                f"Available columns: {list(df.columns)}"
            )
            return existing

        raw_codes = df[code_col].astype(str).str.extract(r"(\d+)")[0].dropna().tolist()
        codes = _validate_codes(raw_codes)

        if not codes:
            log.warning("AkShare returned data but no valid codes — keeping existing")
            return existing

        out = {
            "codes": codes,
            "updated_ts": time.time(),
            "updated_at": datetime.now().isoformat(),
            "source": "akshare_spot_em",
        }
        save_universe(out)
        log.info(f"Universe refreshed: {len(codes)} codes")
        return out

    except ImportError:
        log.warning("AkShare not installed — cannot refresh universe")
        return existing
    except Exception as exc:
        log.warning(f"Universe refresh failed: {exc}")
        return existing


def get_new_listings(
    days: int = 60,
    force_refresh: bool = False,
) -> List[str]:
    """
    Return codes of stocks listed within the last *days* days.

    Only works on China direct connections with AkShare.
    Returns an empty list on failure (never breaks the pipeline).

    Args:
        days: Look-back window in calendar days.
        force_refresh: Currently unused but reserved for cache control.

    Returns:
        List of 6-digit stock code strings.
    """
    from core.network import get_network_env

    env = get_network_env()
    if not env.is_china_direct:
        log.debug("get_new_listings: skipped (not China direct)")
        return []

    cutoff = date.today() - timedelta(days=days)

    try:
        import akshare as ak

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

            # Find code column
            code_col = None
            for c in ("代码", "证券代码", "stock_code", "code"):
                if c in df.columns:
                    code_col = c
                    break
            if code_col is None:
                continue

            # Find listing date column
            date_col = None
            for c in ("上市日期", "listing_date", "list_date"):
                if c in df.columns:
                    date_col = c
                    break

            raw_codes = df[code_col].astype(str).str.extract(r"(\d+)")[0].dropna().tolist()
            codes = _validate_codes(raw_codes)

            # Filter by listing date if column is available
            if date_col is not None and codes:
                try:
                    dates = pd.to_datetime(df[date_col], errors="coerce")
                    mask = dates >= pd.Timestamp(cutoff)
                    filtered_raw = (
                        df.loc[mask, code_col]
                        .astype(str)
                        .str.extract(r"(\d+)")[0]
                        .dropna()
                        .tolist()
                    )
                    filtered = _validate_codes(filtered_raw)
                    if filtered:
                        codes = filtered
                    else:
                        log.debug(
                            f"Date filtering removed all codes; "
                            f"returning unfiltered {len(codes)} codes"
                        )
                except Exception as exc:
                    log.debug(f"Date filtering failed: {exc}")

            log.info(f"New listings (last {days} days): {len(codes)} codes")
            return codes

    except ImportError:
        log.debug("AkShare not installed — cannot fetch new listings")
    except Exception as exc:
        log.warning(f"get_new_listings failed: {exc}")

    return []


# Need pandas for date filtering in get_new_listings
try:
    import pandas as pd
except ImportError:
    pd = None  # type: ignore