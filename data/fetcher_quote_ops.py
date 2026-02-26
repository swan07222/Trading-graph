# data/fetcher_quote_ops.py
import json
import math
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

_SOURCE_RELIABILITY_BONUS = {
    "tencent": 7.0,
    "akshare": 5.0,
    "sina": 4.0,
    "yahoo": 2.5,
    "spot_cache": -5.0,
    "last_good": -6.0,
    "localdb_last_close": -9.0,
}

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
    ConnectionError,
    ConnectionResetError,
    ConnectionAbortedError,
    ConnectionRefusedError,
)


def _quote_sanity_ok(q: Quote | None) -> bool:
    """Validate basic OHLC/price sanity before accepting a quote candidate."""
    if q is None:
        return False
    try:
        px = float(getattr(q, "price", 0.0) or 0.0)
    except (TypeError, ValueError):
        return False
    if px <= 0.0:
        return False

    try:
        high = float(getattr(q, "high", px) or px)
        low = float(getattr(q, "low", px) or px)
    except (TypeError, ValueError):
        return False
    if high > 0.0 and low > 0.0 and high < low:
        return False

    try:
        open_px = float(getattr(q, "open", px) or px)
        close_px = float(getattr(q, "close", px) or px)
    except (TypeError, ValueError):
        return False
    if open_px <= 0.0 or close_px <= 0.0:
        return False
    return True


def _source_reliability_bonus(name: object) -> float:
    src = str(name or "").strip().lower()
    if not src:
        return 0.0
    for key, bonus in _SOURCE_RELIABILITY_BONUS.items():
        if key in src:
            return float(bonus)
    return 0.0


def _quote_quality_score(
    self,
    quote: Quote | None,
    *,
    source_rank: float = 0.0,
) -> float:
    """Compute quote quality score; higher means more reliable.
    
    FIX 2026-02-25: Added comprehensive validation for edge cases
    including NaN, Inf, and type conversion errors.
    """
    if not _quote_sanity_ok(quote):
        return float("-inf")
    if quote is None:
        return float("-inf")

    score = 30.0 + float(max(0.0, source_rank))
    score += _source_reliability_bonus(getattr(quote, "source", ""))

    # Freshness is the strongest factor.
    try:
        age_s = float(self._quote_age_seconds(quote))
        if not math.isfinite(age_s):
            score -= 20.0
        else:
            score += max(-35.0, 18.0 - min(35.0, age_s * 1.8))
    except (TypeError, ValueError, AttributeError):
        # If we can't determine age, penalize but don't reject
        score -= 15.0

    # Prefer realtime/non-delayed quotes.
    try:
        is_delayed = bool(getattr(quote, "is_delayed", False))
        if is_delayed:
            score -= 8.0
        else:
            score += 4.0
    except (TypeError, ValueError):
        # Default to non-delayed if attribute is malformed
        score += 4.0

    # Lower reported latency is better.
    try:
        latency_ms = float(getattr(quote, "latency_ms", 0.0) or 0.0)
        if not math.isfinite(latency_ms):
            latency_ms = 0.0
    except (TypeError, ValueError):
        latency_ms = 0.0
    score += max(-10.0, 4.0 - min(10.0, latency_ms / 180.0))

    # Higher volume snapshots are usually more reliable.
    try:
        volume = float(getattr(quote, "volume", 0.0) or 0.0)
        if not math.isfinite(volume):
            volume = 0.0
    except (TypeError, ValueError):
        volume = 0.0
    if volume > 0:
        score += min(6.0, math.log1p(volume) / 3.5)

    # Penalize suspicious OHLC relationships if present.
    try:
        px = float(getattr(quote, "price", 0.0) or 0.0)
        if not math.isfinite(px) or px <= 0:
            px = 1e-9  # Use small positive value for ratio calculation
        high = float(getattr(quote, "high", px) or px)
        low = float(getattr(quote, "low", px) or px)
        if not math.isfinite(high):
            high = px
        if not math.isfinite(low):
            low = px
        if high > 0.0 and low > 0.0:
            span = abs(high - low) / max(px, 1e-9)
            if span > 0.25:
                score -= 10.0
    except (TypeError, ValueError):
        # Can't validate OHLC - don't penalize but don't reward
        pass

    return float(score)


def _upsert_best_quote(
    self,
    result: dict[str, Quote],
    code6: str,
    candidate: Quote | None,
    *,
    source_rank: float,
    best_scores: dict[str, float] | None = None,
) -> None:
    """Insert candidate quote if it is better than any existing one."""
    if not _quote_sanity_ok(candidate):
        return
    if candidate is None:
        return

    current = result.get(code6)
    cand_score = _quote_quality_score(
        self,
        candidate,
        source_rank=float(source_rank),
    )

    curr_score = float("-inf")
    if best_scores is not None and code6 in best_scores:
        try:
            curr_score = float(best_scores.get(code6, float("-inf")))
        except (TypeError, ValueError):
            curr_score = float("-inf")
    elif current is not None:
        curr_score = _quote_quality_score(
            self,
            current,
            source_rank=0.0,
        )

    replace_existing = cand_score > curr_score
    if not replace_existing and cand_score == curr_score and current is not None:
        cand_age = float(self._quote_age_seconds(candidate))
        curr_age = float(self._quote_age_seconds(current))
        replace_existing = cand_age < curr_age

    if replace_existing:
        result[code6] = candidate
        if best_scores is not None:
            best_scores[code6] = float(cand_score)


def _is_tencent_source(source: object) -> bool:
    """Return True when source name resolves to Tencent."""
    return str(getattr(source, "name", "")).strip().lower() == "tencent"


def _fill_from_batch_sources(
    self,
    cleaned: list[str],
    result: dict[str, Quote],
    sources: list[DataSource],
    *,
    best_scores: dict[str, float] | None = None,
) -> None:
    """Fill quotes from any batch-capable source list in order."""
    if not cleaned:
        return

    cleaned_set = set(cleaned)
    for idx, source in enumerate(sources):
        fn = getattr(source, "get_realtime_batch", None)
        if not callable(fn):
            continue

        try:
            # Query full request-set to allow lower-priority sources to replace
            # stale/delayed quotes with fresher snapshots.
            out = fn(cleaned)
            if not isinstance(out, dict):
                continue

            for code, q in out.items():
                code6 = self.clean_code(code)
                if not code6 or code6 not in cleaned_set:
                    continue
                source_rank = max(0.0, 40.0 - (float(idx) * 6.0))
                _upsert_best_quote(
                    self,
                    result,
                    code6,
                    q,
                    source_rank=source_rank,
                    best_scores=best_scores,
                )
        except _RECOVERABLE_FETCH_EXCEPTIONS as exc:
            log.debug(
                "Batch quote source %s failed: %s",
                getattr(source, "name", "?"),
                exc,
            )
            continue


def get_realtime_batch(self, codes: list[str]) -> dict[str, Quote]:
    """Fetch real-time quotes for multiple codes in one batch."""
    cleaned = list(dict.fromkeys(c for c in (self.clean_code(c) for c in codes) if c))
    if not cleaned:
        return {}
    if _is_offline():
        return {}

    now = time.time()
    key = ",".join(sorted(cleaned))

    # Micro-cache read
    with self._rt_cache_lock:
        mc = self._rt_batch_microcache
        if mc["key"] == key and (now - float(mc["ts"])) < _MICRO_CACHE_TTL:
            data = mc["data"]
            if isinstance(data, dict) and data:
                return dict(data)

    result: dict[str, Quote] = {}
    best_scores: dict[str, float] = {}
    strict_realtime = bool(getattr(self, "_strict_realtime_quotes", False))

    active_sources = list(self._get_active_sources())
    tencent_sources = [s for s in active_sources if self._is_tencent_source(s)]
    fallback_sources = [s for s in active_sources if not self._is_tencent_source(s)]

    # Prefer Tencent for CN quotes but still run fallback providers to improve
    # freshness and reliability where possible.
    _fill_from_batch_sources(
        self,
        cleaned,
        result,
        tencent_sources,
        best_scores=best_scores,
    )
    _fill_from_batch_sources(
        self,
        cleaned,
        result,
        fallback_sources,
        best_scores=best_scores,
    )

    # Per-symbol APIs for providers without batch endpoints.
    missing = [c for c in cleaned if c not in result]
    if missing:
        _fill_from_single_source_quotes(
            self,
            missing,
            result,
            fallback_sources,
            best_scores=best_scores,
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
        retry_tencent = [s for s in retry_active if self._is_tencent_source(s)]
        retry_fallback = [s for s in retry_active if not self._is_tencent_source(s)]

        _fill_from_batch_sources(
            self,
            missing,
            result,
            retry_tencent,
            best_scores=best_scores,
        )
        missing = [c for c in cleaned if c not in result]
        if missing:
            _fill_from_batch_sources(
                self,
                missing,
                result,
                retry_fallback,
                best_scores=best_scores,
            )
        missing = [c for c in cleaned if c not in result]
        if missing:
            _fill_from_single_source_quotes(
                self,
                missing,
                result,
                retry_fallback,
                best_scores=best_scores,
            )

    if not strict_realtime:
        # Last-good fallback
        missing = [c for c in cleaned if c not in result]
        if missing:
            last_good = self._fallback_last_good(missing)
            for code, quote in last_good.items():
                _upsert_best_quote(
                    self,
                    result,
                    code,
                    quote,
                    source_rank=-3.0,
                    best_scores=best_scores,
                )

    if not strict_realtime and bool(getattr(self, "_allow_last_close_fallback", True)):
        # DB last-close fallback
        missing = [c for c in cleaned if c not in result]
        if missing:
            last_close = self._fallback_last_close_from_db(missing)
            for code, quote in last_close.items():
                _upsert_best_quote(
                    self,
                    result,
                    code,
                    quote,
                    source_rank=-8.0,
                    best_scores=best_scores,
                )

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
        log.warning("Spot-cache quote fill failed (symbols=%d): %s", len(missing), exc)
        if bool(getattr(self, "_strict_errors", False)):
            raise


def _fill_from_single_source_quotes(
    self,
    missing: list[str],
    result: dict[str, Quote],
    sources: list[DataSource],
    *,
    best_scores: dict[str, float] | None = None,
) -> None:
    """Fill missing symbols using per-symbol source APIs.

    Only uses sources that do NOT have a batch method (to avoid double-calling).
    """
    if not missing:
        return

    remaining = list(dict.fromkeys(self.clean_code(c) for c in missing if c))
    if not remaining:
        return

    for idx, source in enumerate(sources):
        if not remaining:
            break

        # Skip sources that have batch support (already covered above).
        fn = getattr(source, "get_realtime_batch", None)
        if callable(fn):
            continue

        next_remaining: list[str] = []
        for code6 in remaining:
            if code6 in result:
                continue
            try:
                q = source.get_realtime(code6)
                if q:
                    source_rank = max(0.0, 28.0 - (float(idx) * 5.0))
                    _upsert_best_quote(
                        self,
                        result,
                        code6,
                        q,
                        source_rank=source_rank,
                        best_scores=best_scores,
                    )
                    if code6 in result:
                        continue
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
        allow_stale=bool(getattr(self, "_allow_stale_realtime_fallback", False)),
        quote_age_seconds=self._quote_age_seconds,
        mark_quote_as_delayed=self._mark_quote_as_delayed,
        context=context,
        logger=log,
    )


def _fallback_last_close_from_db(
    self,
    codes: list[str],
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
                log.warning(
                    "DB last-close fallback: missing 'close' column for %s (columns=%s)",
                    code6,
                    list(df.columns),
                )
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
                code=code6,
                name="",
                price=px,
                open=float(row.get("open", px) or px),
                high=float(row.get("high", px) or px),
                low=float(row.get("low", px) or px),
                close=px,
                volume=int(row.get("volume", 0) or 0),
                amount=float(row.get("amount", 0.0) or 0.0),
                change=0.0,
                change_pct=0.0,
                source="localdb_last_close",
                is_delayed=True,
                latency_ms=0.0,
                timestamp=ts,
            )
        except _RECOVERABLE_FETCH_EXCEPTIONS as exc:
            log.warning("DB last-close fallback failed for %s: %s", code6, exc)
            continue
    return out
