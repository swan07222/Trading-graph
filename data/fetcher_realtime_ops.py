import math
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

from data.fetcher_sources import Quote


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _to_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return int(default)


def fill_from_spot_cache(
    cache: Any,
    missing: list[str],
    result: dict[str, Quote],
) -> None:
    snapshot_ts: datetime | None = None
    cache_time = getattr(cache, "_cache_time", None)
    if isinstance(cache_time, (int, float)) and float(cache_time) > 0.0:
        try:
            import time as _time
            ct = float(cache_time)
            # Sanity-check: reject timestamps more than 60s in the future
            # (bogus values would cause quotes to never appear stale)
            if ct <= (_time.time() + 60.0):
                snapshot_ts = datetime.fromtimestamp(ct, tz=timezone.utc)
        except (OverflowError, OSError, ValueError):
            snapshot_ts = None

    for code in missing:
        if code in result:
            continue

        try:
            cached = cache.get_quote(code)
        except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
            continue

        if not isinstance(cached, dict):
            continue

        price = _to_float(cached.get("price"), 0.0)
        if price <= 0.0:
            continue

        result[code] = Quote(
            code=code,
            name=str(cached.get("name") or ""),
            price=price,
            open=_to_float(cached.get("open"), 0.0),
            high=_to_float(cached.get("high"), 0.0),
            low=_to_float(cached.get("low"), 0.0),
            close=_to_float(cached.get("close"), 0.0),
            volume=_to_int(cached.get("volume"), 0),
            amount=_to_float(cached.get("amount"), 0.0),
            change=_to_float(cached.get("change"), 0.0),
            change_pct=_to_float(cached.get("change_pct"), 0.0),
            source="spot_cache",
            is_delayed=True,
            latency_ms=0.0,
            timestamp=snapshot_ts,
        )


def drop_stale_quotes(
    quotes: dict[str, Quote],
    *,
    max_age_s: float,
    allow_stale: bool,
    quote_age_seconds: Callable[[Quote | None], float],
    mark_quote_as_delayed: Callable[[Quote], Quote],
    context: str,
    logger: Any,
) -> dict[str, Quote]:
    if not quotes:
        return {}

    kept: dict[str, Quote] = {}
    dropped: list[str] = []
    missing_ts: list[str] = []
    for code, quote in quotes.items():
        age_s = quote_age_seconds(quote)
        if not math.isfinite(age_s):
            # Infinite age means timestamp is None or unparseable
            if getattr(quote, "timestamp", None) is None:
                missing_ts.append(str(code))
            if allow_stale:
                kept[code] = mark_quote_as_delayed(quote)
                continue
            dropped.append(str(code))
            continue
        if age_s <= float(max_age_s):
            kept[code] = quote
            continue
        if allow_stale:
            kept[code] = mark_quote_as_delayed(quote)
            continue
        dropped.append(str(code))

    if missing_ts:
        logger.debug(
            "drop_stale_quotes [%s]: %d quote(s) have no timestamp: %s",
            context,
            len(missing_ts),
            ",".join(missing_ts[:8]),
        )

    if dropped:
        logger.debug(
            (
                "Dropped %d stale realtime quote(s) in %s "
                "(max_age=%.1fs): %s"
            ),
            len(dropped),
            context,
            float(max_age_s),
            ",".join(dropped[:8]),
        )

    return kept
