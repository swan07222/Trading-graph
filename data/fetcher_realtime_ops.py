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
    for code in missing:
        if code in result:
            continue

        try:
            cached = cache.get_quote(code)
        except Exception:
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
            is_delayed=False,
            latency_ms=0.0,
        )
