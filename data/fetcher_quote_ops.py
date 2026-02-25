# data/fetcher_quote_ops.py
import json
import time
from collections.abc import Callable
from dataclasses import replace
from datetime import datetime
from typing import Any

from data.fetcher_realtime_ops import drop_stale_quotes as _drop_stale_quotes_impl
from data.fetcher_realtime_ops import fill_from_spot_cache as _fill_from_spot_cache_impl
from data.fetcher_sources import (
    _LAST_GOOD_MAX_AGE,
    _MICRO_CACHE_TTL,
    DataSource,
    Quote,
    _is_offline,
    get_spot_cache,
)
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

def _is_tencent_source(source: object) -> bool:
    """Return True when source name resolves to Tencent."""
    return str(getattr(source, "name", "")).strip().lower() == "tencent"
def _fill_from_batch_sources(
    self,
    cleaned: list[str],
    result: dict[str, Quote],
    sources: list[DataSource],
) -> None:
    """Fill quotes from any batch-capable source list in order."""
    if not cleaned:
        return
    for source in sources:
        fn = getattr(source, "get_realtime_batch", None)
        if not callable(fn):
            continue
        remaining = [c for c in cleaned if c not in result]
        if not remaining:
            break
        try:
            out = fn(remaining)
            if not isinstance(out, dict):
                continue
            for code, q in out.items():
                code6 = self.clean_code(code)
                if not code6 or code6 not in remaining:
                    continue
                if q and float(getattr(q, "price", 0.0) or 0.0) > 0:
                    result[code6] = q
        except _RECOVERABLE_FETCH_EXCEPTIONS as exc:
            log.debug(
                "Batch quote source %s failed: %s",
                getattr(source, "name", "?"),
                exc,
            )
            continue
def get_realtime_batch(self, codes: list[str]) -> dict[str, Quote]:
    """Fetch real-time quotes for multiple codes in one batch."""
    cleaned = list(dict.fromkeys(
        c for c in (self.clean_code(c) for c in codes) if c
    ))
    if not cleaned:
        return {}
    if _is_offline():
        return {}
    now = time.time()
    key = ",".join(sorted(cleaned))
    # Micro-cache read
    with self._rt_cache_lock:
        mc = self._rt_batch_microcache
        if (
            mc["key"] == key
            and (now - float(mc["ts"])) < _MICRO_CACHE_TTL
        ):
            data = mc["data"]
            if isinstance(data, dict) and data:
                return dict(data)
    result: dict[str, Quote] = {}
    strict_realtime = bool(
        getattr(self, "_strict_realtime_quotes", False)
    )
    active_sources = list(self._get_active_sources())
    tencent_sources = [
        s for s in active_sources if self._is_tencent_source(s)
    ]
    fallback_sources = [
        s for s in active_sources if not self._is_tencent_source(s)
    ]
    # Prefer Tencent for CN quotes but keep multi-source realtime fallback.
    self._fill_from_batch_sources(cleaned, result, tencent_sources)
    missing = [c for c in cleaned if c not in result]
    if missing:
        self._fill_from_batch_sources(missing, result, fallback_sources)

    # Per-symbol APIs for providers without batch endpoints.
    missing = [c for c in cleaned if c not in result]
    if missing:
        self._fill_from_single_source_quotes(
            missing,
            result,
            fallback_sources,
        )

    if not strict_realtime:
        # Fill from spot-cache snapshot before forcing network refresh.
        missing = [c for c in cleaned if c not in result]
        if missing:
            self._fill_from_spot_cache(missing, result)

    # Force network refresh and retry once if still missing.
    missing = [c for c in cleaned if c not in result]
    if missing and self._maybe_force_network_refresh():
        retry_active = list(self._get_active_sources())
        retry_tencent = [
            s for s in retry_active if self._is_tencent_source(s)
        ]
        retry_fallback = [
            s for s in retry_active if not self._is_tencent_source(s)
        ]

        self._fill_from_batch_sources(missing, result, retry_tencent)
        missing = [c for c in cleaned if c not in result]
        if missing:
            self._fill_from_batch_sources(missing, result, retry_fallback)
        missing = [c for c in cleaned if c not in result]
        if missing:
            self._fill_from_single_source_quotes(
                missing,
                result,
                retry_fallback,
            )

    if not strict_realtime:
        # Last-good fallback
        missing = [c for c in cleaned if c not in result]
        if missing:
            last_good = self._fallback_last_good(missing)
            for code, quote in last_good.items():
                if code not in result:
                    result[code] = quote

    if not strict_realtime and bool(getattr(self, "_allow_last_close_fallback", True)):
        # DB last-close fallback
        missing = [c for c in cleaned if c not in result]
        if missing:
            last_close = self._fallback_last_close_from_db(missing)
            for code, quote in last_close.items():
                if code not in result:
                    result[code] = quote

    result = self._drop_stale_quotes(result, context="get_realtime_batch")

    # Update last-good store
    if result:
        with self._last_good_lock:
            for c, q in result.items():
                src = str(getattr(q, "source", "") or "")
                if q and q.price > 0 and src != "localdb_last_close":
                    self._last_good_quotes[c] = q

    # Micro-cache write
    with self._rt_cache_lock:
        self._rt_batch_microcache["ts"] = now
        self._rt_batch_microcache["key"] = key
        self._rt_batch_microcache["data"] = dict(result)

    return result

def _fill_from_spot_cache(
    self,
    missing: list[str],
    result: dict[str, Quote],
    *,
    get_spot_cache_fn: Callable[[], Any] | None = None,
) -> None:
    """Attempt to fill missing quotes from EastMoney spot cache."""
    cache_getter = get_spot_cache if get_spot_cache_fn is None else get_spot_cache_fn
    try:
        _fill_from_spot_cache_impl(
            cache=cache_getter(),
            missing=missing,
            result=result,
        )
    except _RECOVERABLE_FETCH_EXCEPTIONS as exc:
        log.debug(
            "Spot-cache quote fill failed (symbols=%d): %s",
            len(missing), exc
        )
        if bool(getattr(self, "_strict_errors", False)):
            raise

def _fill_from_single_source_quotes(
    self,
    missing: list[str],
    result: dict[str, Quote],
    sources: list[DataSource],
) -> None:
    """Fill missing symbols using per-symbol source APIs.
    Only uses sources that do NOT have a batch method (to avoid double-calling).
    """
    if not missing:
        return
    remaining = list(dict.fromkeys(
        self.clean_code(c) for c in missing if c
    ))
    if not remaining:
        return

    for source in sources:
        if not remaining:
            break
        # FIXED: skip sources that HAVE batch (already tried above)
        # Only use sources that only have per-symbol get_realtime
        fn = getattr(source, "get_realtime_batch", None)
        if callable(fn):
            continue  # already tried via batch path

        next_remaining: list[str] = []
        for code6 in remaining:
            if code6 in result:
                continue
            try:
                q = source.get_realtime(code6)
                if q and float(getattr(q, "price", 0.0) or 0.0) > 0:
                    result[code6] = q
                else:
                    next_remaining.append(code6)
            except _RECOVERABLE_FETCH_EXCEPTIONS:
                next_remaining.append(code6)
        remaining = next_remaining

def _fallback_last_good(self, codes: list[str]) -> dict[str, Quote]:
    """Return last-good quotes if they are recent enough."""
    result: dict[str, Quote] = {}
    max_age = float(getattr(self, "_last_good_max_age_s", _LAST_GOOD_MAX_AGE))
    with self._last_good_lock:
        for c in codes:
            q = self._last_good_quotes.get(c)
            if q and q.price > 0:
                age = self._quote_age_seconds(q)
                if age <= max_age:
                    result[c] = self._mark_quote_as_delayed(q)
    return result

def _mark_quote_as_delayed(q: Quote) -> Quote:
    """Clone quote for fallback use and mark as delayed."""
    try:
        src = str(getattr(q, "source", "") or "")
        return replace(
            q,
            source=src if src else "last_good",
            is_delayed=True,
            latency_ms=max(float(getattr(q, "latency_ms", 0.0) or 0.0), 1.0),
        )
    except (TypeError, ValueError, AttributeError):
        return q

def _quote_age_seconds(q: Quote | None) -> float:
    """Compute quote age robustly for naive and timezone-aware timestamps."""
    if q is None:
        return float("inf")
    ts = getattr(q, "timestamp", None)
    if ts is None:
        return float("inf")
    try:
        if getattr(ts, "tzinfo", None) is not None:
            now = datetime.now(tz=ts.tzinfo)
        else:
            now = datetime.now()
        return max(0.0, float((now - ts).total_seconds()))
    except (AttributeError, OverflowError, TypeError, ValueError):
        return float("inf")

def _drop_stale_quotes(
    self,
    quotes: dict[str, Quote],
    *,
    context: str,
) -> dict[str, Quote]:
    return _drop_stale_quotes_impl(
        quotes=quotes,
        max_age_s=float(getattr(self, "_realtime_quote_max_age_s", 8.0)),
        allow_stale=bool(
            getattr(self, "_allow_stale_realtime_fallback", False)
        ),
        quote_age_seconds=self._quote_age_seconds,
        mark_quote_as_delayed=self._mark_quote_as_delayed,
        context=context,
        logger=log,
    )

def _fallback_last_close_from_db(
    self, codes: list[str]
) -> dict[str, Quote]:
    """Fallback quote from local DB (last close)."""
    out: dict[str, Quote] = {}
    for code in codes:
        code6 = self.clean_code(code)
        if not code6:
            continue
        try:
            df = self._db.get_bars(code6, limit=1)
            if df is None or df.empty:
                continue
            row = df.iloc[-1]
            if "close" not in df.columns:
                log.warning("DB last-close fallback: missing 'close' column for %s (columns=%s)", code6, list(df.columns))
                continue
            px = float(row.get("close", 0.0) or 0.0)
            if px <= 0:
                continue
            ts = None
            try:
                ts = df.index[-1].to_pydatetime()
            except (AttributeError, TypeError, ValueError):
                ts = datetime.now()
            out[code6] = Quote(
                code=code6, name="",
                price=px,
                open=float(row.get("open", px) or px),
                high=float(row.get("high", px) or px),
                low=float(row.get("low", px) or px),
                close=px,
                volume=int(row.get("volume", 0) or 0),
                amount=float(row.get("amount", 0.0) or 0.0),
                change=0.0, change_pct=0.0,
                source="localdb_last_close",
                is_delayed=True, latency_ms=0.0,
                timestamp=ts,
            )
        except _RECOVERABLE_FETCH_EXCEPTIONS as exc:
            log.debug("DB last-close fallback failed for %s: %s", code6, exc)
            continue
    return out
