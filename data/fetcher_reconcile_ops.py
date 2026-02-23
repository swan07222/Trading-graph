# data/fetcher_reconcile_ops.py
import json
import threading
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from config.settings import CONFIG
from data.session_cache import get_session_bar_cache
from utils.logger import get_logger

log = get_logger(__name__)
_RECOVERABLE_FETCH_EXCEPTIONS = (
    AttributeError,
    ImportError,
    IndexError,
    KeyError,
    OSError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
    json.JSONDecodeError,
)

def _now_shanghai_naive() -> datetime:
    """Return current Asia/Shanghai wall time as a naive datetime."""
    try:
        from zoneinfo import ZoneInfo
        return datetime.now(tz=ZoneInfo("Asia/Shanghai")).replace(tzinfo=None)
    except _RECOVERABLE_FETCH_EXCEPTIONS:
        # zoneinfo may be unavailable; keep Shanghai wall-clock fallback.
        return datetime.now(
            tz=timezone(timedelta(hours=8))
        ).replace(tzinfo=None)

def _get_refresh_reconcile_lock(self) -> threading.RLock:
    lock = getattr(self, "_refresh_reconcile_lock", None)
    if hasattr(lock, "acquire") and hasattr(lock, "release"):
        return lock
    lock = threading.RLock()
    self._refresh_reconcile_lock = lock
    return lock

def _get_refresh_reconcile_path(self) -> Path:
    path = getattr(self, "_refresh_reconcile_path", None)
    if isinstance(path, Path):
        return path
    path = Path(CONFIG.data_dir) / "refresh_reconcile_queue.json"
    self._refresh_reconcile_path = path
    return path

def _refresh_reconcile_key(self, code: str, interval: str) -> str:
    code6 = self.clean_code(code)
    iv = self._normalize_interval_token(interval)
    return f"{code6}:{iv}" if code6 else ""

def _load_refresh_reconcile_queue(self) -> dict[str, dict[str, object]]:
    """Load pending refresh reconcile tasks from disk."""
    path = self._get_refresh_reconcile_path()
    lock = self._get_refresh_reconcile_lock()
    with lock:
        if not path.exists():
            return {}
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except _RECOVERABLE_FETCH_EXCEPTIONS:
            return {}
    payload = raw.get("pending", raw) if isinstance(raw, dict) else {}
    if not isinstance(payload, dict):
        return {}

    out: dict[str, dict[str, object]] = {}
    for key, value in payload.items():
        if not isinstance(value, dict):
            continue
        code_hint = value.get("code")
        iv_hint = value.get("interval")
        if not code_hint and isinstance(key, str) and ":" in key:
            code_hint = key.split(":", 1)[0]
        if not iv_hint and isinstance(key, str) and ":" in key:
            iv_hint = key.split(":", 1)[1]
        code6 = self.clean_code(str(code_hint or ""))
        iv = self._normalize_interval_token(str(iv_hint or "1m"))
        if not code6:
            continue
        qkey = self._refresh_reconcile_key(code6, iv)
        if not qkey:
            continue
        out[qkey] = {
            "code": code6,
            "interval": iv,
            "pending_since": str(value.get("pending_since") or ""),
            "attempts": int(value.get("attempts", 0) or 0),
            "last_attempt_at": str(value.get("last_attempt_at") or ""),
            "last_error": str(value.get("last_error") or ""),
        }
    return out

def _save_refresh_reconcile_queue(self, queue: dict[str, dict[str, object]]) -> None:
    """Persist pending refresh reconcile tasks to disk."""
    path = self._get_refresh_reconcile_path()
    lock = self._get_refresh_reconcile_lock()
    payload = {
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "pending": dict(queue or {}),
    }
    with lock:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(
                json.dumps(payload, ensure_ascii=True, indent=2),
                encoding="utf-8",
            )
            tmp.replace(path)
        except Exception as exc:
            log.debug("Failed saving refresh reconcile queue: %s", exc)

def _mark_refresh_reconcile_pending(
    self,
    queue: dict[str, dict[str, object]],
    code: str,
    interval: str,
    *,
    error_text: str,
) -> bool:
    key = self._refresh_reconcile_key(code, interval)
    code6 = self.clean_code(code)
    iv = self._normalize_interval_token(interval)
    if not key or not code6:
        return False
    prev = dict(queue.get(key) or {})
    attempts = int(prev.get("attempts", 0) or 0) + 1
    pending_since = str(
        prev.get("pending_since") or datetime.now().isoformat(timespec="seconds")
    )
    entry = {
        "code": code6,
        "interval": iv,
        "pending_since": pending_since,
        "attempts": attempts,
        "last_attempt_at": datetime.now().isoformat(timespec="seconds"),
        "last_error": str(error_text or ""),
    }
    if prev == entry:
        return False
    queue[key] = entry
    return True

def _clear_refresh_reconcile_pending(
    self,
    queue: dict[str, dict[str, object]],
    code: str,
    interval: str,
) -> bool:
    key = self._refresh_reconcile_key(code, interval)
    if not key:
        return False
    return queue.pop(key, None) is not None

def get_pending_reconcile_entries(
    self,
    interval: str | None = None,
) -> dict[str, dict[str, object]]:
    """Return pending reconcile entries, optionally filtered by interval."""
    queue = self._load_refresh_reconcile_queue()
    iv_filter = self._normalize_interval_token(interval) if interval else ""
    if not iv_filter:
        return dict(queue)

    out: dict[str, dict[str, object]] = {}
    for key, entry in queue.items():
        iv = self._normalize_interval_token(entry.get("interval") if isinstance(entry, dict) else "")
        if iv != iv_filter:
            continue
        out[str(key)] = dict(entry)
    return out

def get_pending_reconcile_codes(
    self,
    interval: str | None = None,
) -> list[str]:
    """Return sorted unique stock codes that still need reconcile."""
    entries = self.get_pending_reconcile_entries(interval=interval)
    seen: set[str] = set()
    out: list[str] = []
    for entry in entries.values():
        code6 = self.clean_code(str(entry.get("code") if isinstance(entry, dict) else ""))
        if not code6 or code6 in seen:
            continue
        seen.add(code6)
        out.append(code6)
    return sorted(out)

def reconcile_pending_cache_sync(
    self,
    *,
    codes: list[str] | None = None,
    interval: str = "1m",
    db_limit: int | None = None,
    get_session_bar_cache_fn: Callable[[], Any] | None = None,
) -> dict[str, object]:
    """Attempt to heal pending DB->session-cache sync debt without network fetches.

    Reads pending queue entries, writes existing DB bars into session cache, and
    clears successfully reconciled entries.
    """
    iv = self._normalize_interval_token(interval)
    intraday = iv not in {"1d", "1wk", "1mo"}
    queue = self._load_refresh_reconcile_queue()

    target_codes = {
        self.clean_code(x)
        for x in list(codes or [])
        if self.clean_code(x)
    }
    pending_items: list[tuple[str, dict[str, object]]] = []
    for key, entry in queue.items():
        if not isinstance(entry, dict):
            continue
        code6 = self.clean_code(str(entry.get("code") or ""))
        if not code6:
            continue
        entry_iv = self._normalize_interval_token(str(entry.get("interval") or "1m"))
        if entry_iv != iv:
            continue
        if target_codes and code6 not in target_codes:
            continue
        pending_items.append((str(key), dict(entry)))

    report: dict[str, object] = {
        "interval": iv,
        "queued_before": int(len(queue)),
        "targeted": int(len(pending_items)),
        "reconciled": 0,
        "failed": 0,
        "errors": {},
        "remaining": int(len(queue)),
    }
    if not pending_items:
        return report

    session_cache_getter = (
        get_session_bar_cache
        if get_session_bar_cache_fn is None
        else get_session_bar_cache_fn
    )
    try:
        session_cache = session_cache_getter()
    except Exception as exc:
        session_cache = None
        report["errors"] = {"_session_cache": str(exc)}
        report["failed"] = int(len(pending_items))
        return report

    if session_cache is None:
        report["errors"] = {"_session_cache": "unavailable"}
        report["failed"] = int(len(pending_items))
        return report

    rows_limit = int(max(1, db_limit or (12000 if intraday else 2400)))
    changed = False
    errors: dict[str, str] = {}
    now_iso = datetime.now().isoformat(timespec="seconds")
    market_open = bool(CONFIG.is_market_open())

    for key, entry in pending_items:
        code6 = self.clean_code(str(entry.get("code") or ""))
        if not code6:
            continue
        try:
            if intraday:
                db_frame = self._clean_dataframe(
                    self._db.get_intraday_bars(code6, interval=iv, limit=rows_limit),
                    interval=iv,
                )
                db_frame = self._filter_cn_intraday_session(db_frame, iv)
            else:
                db_frame = self._clean_dataframe(
                    self._db.get_bars(code6, limit=rows_limit),
                    interval="1d",
                )
                db_frame = self._resample_daily_to_interval(db_frame, iv)

            if db_frame.empty:
                raise RuntimeError("no_db_rows_for_reconcile")

            session_cache.upsert_history_frame(
                code6,
                iv,
                db_frame,
                source="official_history",
                is_final=True,
            )

            if intraday and (not market_open):
                try:
                    markers = session_cache.describe_symbol_interval(code6, iv)
                    rt_anchor = markers.get("first_realtime_after_akshare_ts")
                    if rt_anchor is not None:
                        session_cache.purge_realtime_rows(
                            code6,
                            iv,
                            since_ts=rt_anchor,
                        )
                except Exception as exc:
                    log.debug("Suppressed exception in data/fetcher.py", exc_info=exc)

            if queue.pop(key, None) is not None:
                changed = True
            report["reconciled"] = int(report.get("reconciled", 0)) + 1
        except Exception as exc:
            msg = str(exc)
            errors[code6] = msg
            report["failed"] = int(report.get("failed", 0)) + 1
            self._mark_refresh_reconcile_pending(
                queue,
                code6,
                iv,
                error_text=msg,
            )
            current_key = self._refresh_reconcile_key(code6, iv)
            if current_key and current_key in queue:
                queue[current_key]["last_attempt_at"] = now_iso
            changed = True

    if changed:
        self._save_refresh_reconcile_queue(queue)

    report["errors"] = errors
    report["remaining"] = int(len(queue))
    return report


# Backward-compatible names expected by data.fetcher import wiring.
_get_pending_reconcile_entries = get_pending_reconcile_entries
_get_pending_reconcile_codes = get_pending_reconcile_codes
_reconcile_pending_cache_sync = reconcile_pending_cache_sync
