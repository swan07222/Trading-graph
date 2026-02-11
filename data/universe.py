# data/universe.py

from __future__ import annotations

import json
import time
import threading
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Dict, List, Optional

from config.settings import CONFIG
from core.constants import get_exchange
from utils.logger import get_logger

log = get_logger(__name__)

# Import pandas at module level with proper fallback
try:
    import pandas as pd
    _HAS_PANDAS = True
except ImportError:
    pd = None  # type: ignore[assignment]
    _HAS_PANDAS = False

# Module-level lock for file writes
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

    FIX FSYNC: Uses atomic write pattern (write-to-temp, fsync, rename)
    for crash safety instead of direct write without fsync.
    """
    path = _universe_path()
    with _universe_lock:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)

            # Try atomic_io first
            try:
                from utils.atomic_io import atomic_write_json
                atomic_write_json(path, data, use_lock=False)  # We already hold _universe_lock
                return
            except ImportError:
                pass

            # Fallback: manual atomic write with fsync
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
            try:
                tmp = path.with_suffix(".json.tmp")
                tmp.unlink(missing_ok=True)
            except OSError:
                pass


def _parse_updated_ts(data: Dict) -> float:
    """Safely extract ``updated_ts`` as a float."""
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

    try:
        import akshare as ak

        df = ak.stock_zh_a_spot_em()
        if df is None or df.empty:
            log.warning("AkShare returned empty spot data — keeping existing universe")
            return existing

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

    FIX PANDAS: Guards against pd being None if pandas import failed.
    """
    # FIX PANDAS: Need pandas for date filtering
    if not _HAS_PANDAS:
        log.debug("get_new_listings: pandas not available")
        return []

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