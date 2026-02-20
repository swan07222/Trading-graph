from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd

from data.fetcher_sources import DataSource, Quote


def normalize_source_name(name: str) -> str:
    return str(name or "").strip().lower()


def resolve_source_order(
    *,
    raw_value: str,
    source_classes: dict[str, type[DataSource]],
    default_order: tuple[str, ...],
    logger: Any,
) -> list[str]:
    raw = str(raw_value or "").strip()
    if not raw:
        return list(default_order)

    parsed = [
        normalize_source_name(part)
        for part in str(raw).replace(";", ",").split(",")
    ]
    out = [
        name
        for name in list(dict.fromkeys(parsed))
        if name in source_classes
    ]
    if out:
        return out

    logger.warning(
        "Invalid TRADING_ENABLED_SOURCES=%r; using default order",
        raw,
    )
    return list(default_order)


def create_local_database_source(db_ref: Any) -> DataSource:
    class LocalDatabaseSource(DataSource):
        name = "localdb"
        priority = -1
        needs_china_direct = False
        needs_vpn = False

        def __init__(self, db_source: Any):
            super().__init__()
            self._db = db_source

        def get_history(self, code: str, days: int) -> pd.DataFrame:
            return self._db.get_bars(str(code).zfill(6), limit=int(days))

        def get_history_instrument(
            self, inst: dict, days: int, interval: str = "1d"
        ) -> pd.DataFrame:
            sym = str(inst.get("symbol") or "").zfill(6)
            if not sym:
                return pd.DataFrame()
            if interval == "1d":
                return self._db.get_bars(sym, limit=int(days))
            return self._db.get_intraday_bars(
                sym, interval=interval, limit=int(days)
            )

        def get_realtime(self, code: str) -> Quote | None:
            del code
            return None

    return LocalDatabaseSource(db_ref)


def source_health_score(
    source: DataSource,
    env: Any,
    *,
    now: datetime | None = None,
) -> float:
    """Score a source by network suitability + recent health."""
    score = 0.0

    if source.name == "localdb":
        score += 120.0
    elif env.is_china_direct:
        eastmoney_ok = bool(getattr(env, "eastmoney_ok", False))
        if source.name == "tencent":
            score += 92.0
        elif source.name == "akshare":
            score += 88.0 if eastmoney_ok else 24.0
        elif source.name == "sina":
            score += 82.0
        elif source.name == "yahoo":
            score += 6.0
    else:
        if source.name == "yahoo":
            score += 90.0
        elif source.name == "tencent":
            score += 68.0
        elif source.name == "akshare":
            score += 8.0
        elif source.name == "sina":
            score += 6.0

    try:
        if source.is_suitable_for_network():
            score += 15.0
        else:
            score -= 40.0
    except (AttributeError, RuntimeError, TypeError, ValueError):
        score -= 5.0

    st = source.status
    attempts = max(1, int(st.success_count + st.error_count))
    success_rate = float(st.success_count) / attempts
    score += 30.0 * success_rate

    if st.avg_latency_ms > 0:
        score -= min(25.0, st.avg_latency_ms / 200.0)

    score -= min(20.0, float(st.consecutive_errors) * 1.5)
    now_dt = now or datetime.now()
    if st.disabled_until and now_dt < st.disabled_until:
        score -= 50.0

    return score
