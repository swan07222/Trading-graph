# data/fetcher.py
import json
import math
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from config.settings import CONFIG
from data.cache import get_cache
from data.database import get_database
from data.fetcher_sources import (
    _INTRADAY_CAPS,
    _INTRADAY_INTERVALS,
    _LAST_GOOD_MAX_AGE,
    _MICRO_CACHE_TTL,
    BARS_PER_DAY,
    INTERVAL_MAX_DAYS,
    AkShareSource,
    DataSource,
    DataSourceStatus,
    Quote,
    SinaHistorySource,
    TencentQuoteSource,
    YahooSource,
    _is_offline,
    bars_to_days,
    get_spot_cache,
)
from data.session_cache import get_session_bar_cache
from utils.logger import get_logger

log = get_logger(__name__)

__all__ = [
    "AkShareSource",
    "DataFetcher",
    "DataSource",
    "DataSourceStatus",
    "Quote",
    "SinaHistorySource",
    "TencentQuoteSource",
    "YahooSource",
    "bars_to_days",
    "get_fetcher",
    "get_spot_cache",
]

class DataFetcher:
    """
    High-performance data fetcher with automatic network-aware source
    selection, local DB caching, and multi-source fallback.
    """

    def __init__(self):
        self._all_sources: list[DataSource] = []
        self._cache = get_cache()
        self._db = get_database()
        self._rate_limiter = threading.Semaphore(CONFIG.data.parallel_downloads)
        self._request_times: dict[str, float] = {}
        self._min_interval: float = 0.5
        self._intraday_interval: float = 1.2

        self._last_good_quotes: dict[str, Quote] = {}
        self._last_good_lock = threading.RLock()

        # Micro-caches
        self._rt_cache_lock = threading.RLock()
        self._rt_batch_microcache: dict[str, object] = {
            "ts": 0.0, "key": None, "data": {},
        }
        self._rt_single_microcache: dict[str, dict[str, object]] = {}

        self._rate_lock = threading.Lock()
        self._last_source_fail_warn_ts: dict[str, float] = {}
        self._source_fail_warn_cooldown_s: float = 45.0
        self._last_source_fail_warn_global_ts: float = 0.0
        self._source_fail_warn_global_cooldown_s: float = 8.0
        self._last_network_mode: tuple[bool, bool, bool] | None = None
        self._last_network_force_refresh_ts: float = 0.0
        self._network_force_refresh_cooldown_s: float = 20.0
        self._refresh_reconcile_lock = threading.RLock()
        self._refresh_reconcile_path = Path(CONFIG.data_dir) / "refresh_reconcile_queue.json"
        self._init_sources()

    def _init_sources(self) -> None:
        self._all_sources = []
        self._init_local_db_source()

        # Runtime policy:
        # - historical data: Tencent/AkShare/Sina (China IP), Yahoo (foreign)
        # - realtime data: Tencent primary
        for source_cls in (
            TencentQuoteSource,
            AkShareSource,
            SinaHistorySource,
            YahooSource,
        ):
            try:
                source = source_cls()
                if source.status.available:
                    self._all_sources.append(source)
                    log.info(
                        "Data source %s initialized "
                        "(china_direct=%s, vpn=%s)",
                        source.name,
                        source.needs_china_direct,
                        source.needs_vpn,
                    )
            except Exception as exc:
                log.warning("Failed to init %s: %s", source_cls.__name__, exc)

        if not self._all_sources:
            log.error("No data sources available!")

    def _init_local_db_source(self) -> None:
        """Create and register the local database source."""
        try:
            db = self._db

            class LocalDatabaseSource(DataSource):
                name = "localdb"
                priority = -1
                needs_china_direct = False
                needs_vpn = False

                def __init__(self, db_ref):
                    super().__init__()
                    self._db = db_ref

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
                    return None

            self._all_sources.append(LocalDatabaseSource(db))
            log.info("Data source localdb initialized")

        except Exception as exc:
            log.warning("Failed to init localdb source: %s", exc)

    @property
    def _sources(self) -> list[DataSource]:
        """Backward-compatible alias."""
        return self._all_sources

    def _get_active_sources(self) -> list[DataSource]:
        """Get sources prioritized by current network environment."""
        from core.network import get_network_env
        env = get_network_env()

        net_sig = (
            bool(env.is_china_direct),
            bool(getattr(env, "eastmoney_ok", False)),
            bool(getattr(env, "yahoo_ok", False)),
        )
        if self._last_network_mode is None:
            self._last_network_mode = net_sig
        elif net_sig != self._last_network_mode:
            self._last_network_mode = net_sig
            for s in self._all_sources:
                with s._lock:
                    s.status.consecutive_errors = 0
                    s.status.disabled_until = None
                    s.status.available = True
            with self._rate_lock:
                self._request_times.clear()
            log.info(
                "Network mode changed -> cooldowns reset "
                "(%s)",
                "CHINA_DIRECT" if env.is_china_direct else "VPN_FOREIGN",
            )

        active: list[DataSource] = []
        for s in self._all_sources:
            try:
                if not s.is_available():
                    continue
                if not s.is_suitable_for_network():
                    continue
            except Exception as exc:
                log.debug(
                    "Source suitability check failed for %s: %s",
                    getattr(s, "name", "?"),
                    exc,
                )
                continue
            active.append(s)
        ranked = sorted(
            active,
            key=lambda s: (-self._source_health_score(s, env), s.priority),
        )
        return ranked

    def _source_health_score(self, source: DataSource, env) -> float:
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
        except Exception:
            score -= 5.0

        st = source.status
        attempts = max(1, int(st.success_count + st.error_count))
        success_rate = float(st.success_count) / attempts
        score += 30.0 * success_rate

        if st.avg_latency_ms > 0:
            score -= min(25.0, st.avg_latency_ms / 200.0)

        score -= min(20.0, float(st.consecutive_errors) * 1.5)
        if st.disabled_until and datetime.now() < st.disabled_until:
            score -= 50.0

        return score

    def _rate_limit(self, source: str, interval: str = "1d") -> None:
        with self._rate_lock:
            now = time.time()
            last = self._request_times.get(source, 0.0)
            if source == "yahoo":
                min_wait = 2.2 if interval in _INTRADAY_INTERVALS else 1.4
            else:
                min_wait = (
                    self._intraday_interval
                    if interval in _INTRADAY_INTERVALS
                    else self._min_interval
                )
            wait = min_wait - (now - last)
            if wait > 0:
                time.sleep(wait)
            self._request_times[source] = time.time()

    def _db_is_fresh_enough(
        self, code6: str, max_lag_days: int = 3
    ) -> bool:
        """Check whether local DB data is recent enough to skip online fetch."""
        try:
            last = self._db.get_last_date(code6)
            if not last:
                return False
            from core.constants import is_trading_day
            today = datetime.now().date()
            lag = 0
            d = last
            while d < today and lag <= max_lag_days:
                d += timedelta(days=1)
                if is_trading_day(d):
                    lag += 1
            return lag <= max_lag_days
        except Exception:
            return False

    @staticmethod
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
            except Exception as exc:
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
        key = ",".join(cleaned)

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
            self._fill_from_batch_sources(cleaned, result, fallback_sources)

        # Per-symbol APIs for providers without batch endpoints.
        missing = [c for c in cleaned if c not in result]
        if missing:
            self._fill_from_single_source_quotes(
                missing,
                result,
                fallback_sources,
            )

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

            self._fill_from_batch_sources(cleaned, result, retry_tencent)
            missing = [c for c in cleaned if c not in result]
            if missing:
                self._fill_from_batch_sources(cleaned, result, retry_fallback)
            missing = [c for c in cleaned if c not in result]
            if missing:
                self._fill_from_single_source_quotes(
                    missing,
                    result,
                    retry_fallback,
                )

        # Last-good fallback
        missing = [c for c in cleaned if c not in result]
        if missing:
            last_good = self._fallback_last_good(missing)
            for code, quote in last_good.items():
                if code not in result:
                    result[code] = quote

        # DB last-close fallback
        missing = [c for c in cleaned if c not in result]
        if missing:
            last_close = self._fallback_last_close_from_db(missing)
            for code, quote in last_close.items():
                if code not in result:
                    result[code] = quote

        # Update last-good store
        if result:
            with self._last_good_lock:
                for c, q in result.items():
                    if q and q.price > 0:
                        self._last_good_quotes[c] = q

        # Micro-cache write
        with self._rt_cache_lock:
            self._rt_batch_microcache["ts"] = now
            self._rt_batch_microcache["key"] = key
            self._rt_batch_microcache["data"] = dict(result)

        return result

    def _fill_from_spot_cache(
        self, missing: list[str], result: dict[str, Quote]
    ) -> None:
        """Attempt to fill missing quotes from EastMoney spot cache."""
        try:
            cache = get_spot_cache()
            for c in missing:
                if c in result:
                    continue
                q = cache.get_quote(c)
                if q and q.get("price", 0) and q["price"] > 0:
                    result[c] = Quote(
                        code=c,
                        name=q.get("name", ""),
                        price=float(q["price"]),
                        open=float(q.get("open") or 0),
                        high=float(q.get("high") or 0),
                        low=float(q.get("low") or 0),
                        close=float(q.get("close") or 0),
                        volume=int(q.get("volume") or 0),
                        amount=float(q.get("amount") or 0),
                        change=float(q.get("change") or 0),
                        change_pct=float(q.get("change_pct") or 0),
                        source="spot_cache",
                        is_delayed=False,
                        latency_ms=0.0,
                    )
        except Exception as exc:
            log.debug(
                "Spot-cache quote fill failed (symbols=%d): %s",
                len(missing), exc
            )

    def _fill_from_single_source_quotes(
        self,
        missing: list[str],
        result: dict[str, Quote],
        sources: list[DataSource],
    ) -> None:
        """
        Fill missing symbols using per-symbol source APIs.
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
                except Exception:
                    next_remaining.append(code6)
            remaining = next_remaining

    def _fallback_last_good(self, codes: list[str]) -> dict[str, Quote]:
        """Return last-good quotes if they are recent enough."""
        result: dict[str, Quote] = {}
        with self._last_good_lock:
            for c in codes:
                q = self._last_good_quotes.get(c)
                if q and q.price > 0:
                    age = self._quote_age_seconds(q)
                    if age <= _LAST_GOOD_MAX_AGE:
                        result[c] = self._mark_quote_as_delayed(q)
        return result

    @staticmethod
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
        except Exception:
            return q

    @staticmethod
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
        except Exception:
            return float("inf")

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
                px = float(row.get("close", 0.0) or 0.0)
                if px <= 0:
                    continue
                ts = None
                try:
                    ts = df.index[-1].to_pydatetime()
                except Exception:
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
            except Exception:
                continue
        return out

    def _maybe_force_network_refresh(self) -> bool:
        """Force network redetection at most once per cooldown window."""
        now = time.time()
        if (
            now - float(self._last_network_force_refresh_ts)
            < self._network_force_refresh_cooldown_s
        ):
            return False
        self._last_network_force_refresh_ts = now
        try:
            from core.network import get_network_env
            _ = get_network_env(force_refresh=True)
            return True
        except Exception:
            return False

    def _fetch_from_sources_instrument(
        self,
        inst: dict,
        days: int,
        interval: str = "1d",
        include_localdb: bool = True,
        return_meta: bool = False,
    ) -> pd.DataFrame | tuple[pd.DataFrame, dict[str, object]]:
        """Fetch from active sources with smart fallback."""
        sources = self._get_active_sources()
        if not include_localdb:
            sources = [
                s for s in sources if str(getattr(s, "name", "")) != "localdb"
            ]

        if not sources:
            log.warning(
                "No active sources for %s (%s), trying all as fallback",
                inst.get("symbol"), interval,
            )
            sources = [s for s in self._all_sources if s.name != "localdb"]

        if not sources:
            log.warning("No sources at all for %s (%s)", inst.get("symbol"), interval)
            if return_meta:
                return pd.DataFrame(), {}
            return pd.DataFrame()

        # Keep local database as fallback; online providers should be tried first.
        if include_localdb and len(sources) > 1:
            online_sources = [
                s for s in sources if str(getattr(s, "name", "")).strip().lower() != "localdb"
            ]
            local_sources = [
                s for s in sources if str(getattr(s, "name", "")).strip().lower() == "localdb"
            ]
            if online_sources:
                sources = online_sources + local_sources

        # Helpful signal: expected CN history providers when on China IP.
        source_names_now = [
            str(getattr(s, "name", "")).strip().lower() for s in sources
        ]
        is_china_direct = False
        try:
            from core.network import get_network_env

            is_china_direct = bool(getattr(get_network_env(), "is_china_direct", False))
        except Exception:
            is_china_direct = False
        if (
            inst.get("market") == "CN"
            and inst.get("asset") == "EQUITY"
            and is_china_direct
        ):
            expected = {"tencent", "akshare", "sina"}
            if not (expected & set(source_names_now)):
                log.debug(
                    "No expected CN online providers for %s (%s); active=%s",
                    inst.get("symbol"),
                    interval,
                    source_names_now,
                )

        log.debug(
            "Sources for %s (%s): %s",
            inst.get("symbol"), interval,
            [s.name for s in sources],
        )

        with self._rate_limiter:
            errors: list[str] = []
            iv_norm = self._normalize_interval_token(interval)
            is_intraday = iv_norm not in {"1d", "1wk", "1mo"}
            collected: list[dict] = []

            for src_rank, source in enumerate(sources):
                try:
                    self._rate_limit(source.name, interval)
                    df = self._try_source_instrument(
                        source, inst, days, interval
                    )
                    if df is None or df.empty:
                        log.debug(
                            "%s returned empty for %s (%s)",
                            source.name, inst.get("symbol"), interval,
                        )
                        continue

                    df = self._clean_dataframe(df, interval=interval)
                    if df.empty:
                        log.debug(
                            "%s returned unusable rows for %s (%s)",
                            source.name, inst.get("symbol"), interval,
                        )
                        continue

                    quality = (
                        self._intraday_frame_quality(df, interval)
                        if is_intraday
                        else self._daily_frame_quality(df)
                    )

                    min_required = max(5, min(days // 8, 20))
                    row_count = len(df)
                    log.debug(
                        "%s: %d bars for %s (%s) [score=%.3f]",
                        source.name, row_count, inst.get("symbol"),
                        interval, float(quality.get("score", 0.0)),
                    )

                    collected.append({
                        "source":  source.name,
                        "rank":    int(src_rank),
                        "df":      df,
                        "quality": quality,
                        "rows":    row_count,
                    })

                    # For intraday, stop early when we have enough strong bars.
                    # Daily bars should keep collecting so multi-source consensus
                    # can compare provider rows.
                    if (
                        is_intraday
                        and row_count >= min_required
                        and float(quality.get("score", 0.0)) >= 0.65
                    ):
                        break

                except Exception as exc:
                    errors.append(f"{source.name}: {exc}")
                    log.debug(
                        "%s failed for %s (%s): %s",
                        source.name, inst.get("symbol"), interval, exc,
                    )
                    continue

        if not collected:
            if errors:
                severe = [e for e in errors if not self._is_expected_no_data_error(e)]
                symbol = str(inst.get("symbol") or "")
                if severe and self._should_emit_source_fail_warning(symbol, interval):
                    log.warning(
                        "All sources failed for %s (%s): %s",
                        symbol, interval, "; ".join(severe[:3]),
                    )
                else:
                    log.debug(
                        "No usable history for %s (%s)",
                        symbol, interval,
                    )
            if return_meta:
                return pd.DataFrame(), {}
            return pd.DataFrame()

        # Pick best result
        try:
            if is_intraday:
                best = max(
                    collected,
                    key=lambda item: (
                        float(dict(item.get("quality") or {}).get("score", 0.0)),
                        int(item.get("rows", 0)),
                        -int(item.get("rank", 0)),
                    ),
                )
                best_df = best["df"]
                best_q = dict(best.get("quality") or {})
                best_score = float(best_q.get("score", 0.0))
                best_source = str(best.get("source", "unknown"))

                # Cross-validate: replace stale bars using alternative sources
                if (
                    float(best_q.get("stale_ratio", 0.0)) > 0.15
                    and len(collected) > 1
                ):
                    alt_dfs = [
                        c["df"] for c in collected
                        if c is not best and not c["df"].empty
                    ]
                    best_df = self._cross_validate_bars(
                        best_df, alt_dfs, iv_norm,
                    )

                # Opportunistically extend from alternatives
                bpd = float(BARS_PER_DAY.get(iv_norm, 1.0))
                target_rows = int(max(120, min(float(days) * bpd, 2200.0)))
                if len(best_df) < target_rows and len(collected) > 1:
                    candidates = sorted(
                        [c for c in collected if c is not best],
                        key=lambda item: (
                            float(dict(item.get("quality") or {}).get("score", 0.0)),
                            int(item.get("rows", 0)),
                            -int(item.get("rank", 0)),
                        ),
                        reverse=True,
                    )
                    out = best_df.copy()
                    for item in candidates:
                        q = dict(item.get("quality") or {})
                        if float(q.get("score", 0.0)) < 0.25:
                            continue
                        if bool(q.get("suspect", False)) and float(q.get("score", 0.0)) < best_score:
                            continue
                        df_alt = item["df"]
                        if df_alt.empty:
                            continue
                        extra = df_alt.loc[~df_alt.index.isin(out.index)]
                        if extra.empty:
                            continue
                        combined = pd.concat([extra, out], axis=0)
                        out = self._clean_dataframe(combined, interval=interval)
                        if len(out) >= target_rows:
                            break
                    best_df = out

                log.debug(
                    "Selected %s for %s (%s): score=%.3f rows=%d",
                    best_source, inst.get("symbol"), interval,
                    best_score, len(best_df),
                )
                if return_meta:
                    return best_df, {
                        "selection": "intraday",
                        "source": best_source,
                        "score": float(best_score),
                        "source_count": int(len(collected)),
                    }
                return best_df

            # Daily: compare overlapping bars across internet providers first.
            # Keep localdb mainly as fallback if online sources are missing.
            daily_candidates = list(collected)
            online_candidates = [
                item
                for item in collected
                if str(item.get("source", "")).strip().lower() != "localdb"
            ]
            if online_candidates:
                daily_candidates = online_candidates

            quorum_meta = self._daily_consensus_quorum_meta(daily_candidates)

            # Compare overlapping bars across providers and keep the row
            # closest to consensus for each timestamp.
            merged = self._merge_daily_by_consensus(
                daily_candidates,
                interval=interval,
            )
            if not merged.empty:
                if return_meta:
                    meta = dict(quorum_meta)
                    meta["source_count"] = int(len(daily_candidates))
                    meta["selected_rows"] = int(len(merged))
                    return merged, meta
                return merged

            # Safe fallback if consensus merge produced nothing.
            collected_by_score = sorted(
                daily_candidates,
                key=lambda item: float(dict(item.get("quality") or {}).get("score", 0.0)),
                reverse=True,
            )
            merged_parts = [
                item["df"] for item in collected_by_score if not item["df"].empty
            ]
            if not merged_parts:
                if return_meta:
                    return pd.DataFrame(), dict(quorum_meta)
                return pd.DataFrame()
            fallback = self._clean_dataframe(
                pd.concat(merged_parts, axis=0),
                interval=interval,
            )
            if return_meta:
                meta = dict(quorum_meta)
                meta["fallback_used"] = True
                meta["source_count"] = int(len(daily_candidates))
                meta["selected_rows"] = int(len(fallback))
                return fallback, meta
            return fallback

        except Exception as exc:
            log.debug(
                "Failed to select/merge history for %s: %s",
                inst.get("symbol"), exc,
            )
            # Return the first collected result as safe fallback
            if collected:
                if return_meta:
                    return collected[0]["df"], {}
                return collected[0]["df"]
            if return_meta:
                return pd.DataFrame(), {}
            return pd.DataFrame()

    @staticmethod
    def _is_expected_no_data_error(err_msg: str) -> bool:
        msg = str(err_msg or "").lower()
        expected = (
            "no data", "returned empty", "empty dataframe",
            "not found", "404", "no history", "symbol not found",
        )
        return any(k in msg for k in expected)

    def _should_emit_source_fail_warning(
        self, symbol: str, interval: str
    ) -> bool:
        key = f"{symbol}:{interval}"
        now = time.time()
        if (
            now - float(self._last_source_fail_warn_global_ts)
            < self._source_fail_warn_global_cooldown_s
        ):
            return False
        last = float(self._last_source_fail_warn_ts.get(key, 0.0))
        if (now - last) < self._source_fail_warn_cooldown_s:
            return False
        self._last_source_fail_warn_ts[key] = now
        self._last_source_fail_warn_global_ts = now
        return True

    @staticmethod
    def _try_source_instrument(
        source: DataSource, inst: dict, days: int, interval: str
    ) -> pd.DataFrame:
        """Try get_history_instrument, fall back to get_history for CN equity."""
        fn = getattr(source, "get_history_instrument", None)
        if callable(fn):
            return fn(inst, days=days, interval=interval)
        if inst.get("market") == "CN" and inst.get("asset") == "EQUITY":
            return source.get_history(inst["symbol"], days)
        return pd.DataFrame()

    @staticmethod
    def clean_code(code: str) -> str:
        """Normalize a stock code to bare 6-digit form."""
        if code is None:
            return ""
        s = str(code).strip()
        if not s:
            return ""
        s = s.replace(" ", "").replace("-", "").replace("_", "")

        prefixes = (
            "sh.", "sz.", "bj.", "SH.", "SZ.", "BJ.",
            "sh", "sz", "bj", "SH", "SZ", "BJ",
        )
        for p in prefixes:
            if s.startswith(p) and len(s) > len(p):
                candidate = s[len(p):]
                if candidate.replace(".", "").isdigit():
                    s = candidate
                    break

        suffixes = (".SS", ".SZ", ".BJ", ".ss", ".sz", ".bj")
        for suf in suffixes:
            if s.endswith(suf):
                s = s[: -len(suf)]
                break

        digits = "".join(ch for ch in s if ch.isdigit())
        return digits.zfill(6) if digits else ""

    @staticmethod
    def _normalize_interval_token(interval: str | None) -> str:
        iv = str(interval or "1d").strip().lower()
        aliases = {
            "1h":    "60m",
            "60min": "60m",
            "daily": "1d",
            "day":   "1d",
            "1day":  "1d",
            "1440m": "1d",
        }
        return aliases.get(iv, iv)

    @staticmethod
    def _interval_seconds(interval: str | None) -> int:
        iv = str(interval or "1d").strip().lower()
        aliases = {
            "1h":    "60m",
            "60min": "60m",
            "daily": "1d",
            "day":   "1d",
            "1day":  "1d",
            "1440m": "1d",
        }
        iv = aliases.get(iv, iv)
        mapping = {
            "1m":  60,
            "2m":  120,
            "5m":  300,
            "15m": 900,
            "30m": 1800,
            "60m": 3600,
            "1d":  86400,
            "1wk": 86400 * 7,
            "1mo": 86400 * 30,
        }
        if iv in mapping:
            return int(mapping[iv])
        try:
            if iv.endswith("m"):
                return max(1, int(float(iv[:-1]) * 60))
            if iv.endswith("s"):
                return max(1, int(float(iv[:-1])))
            if iv.endswith("h"):
                return max(1, int(float(iv[:-1]) * 3600))
        except Exception as exc:
            log.debug("Invalid interval token (%s): %s", iv, exc)
        return 60

    @staticmethod
    def _now_shanghai_naive() -> datetime:
        """Return current Asia/Shanghai wall time as a naive datetime."""
        try:
            from zoneinfo import ZoneInfo
            return datetime.now(tz=ZoneInfo("Asia/Shanghai")).replace(tzinfo=None)
        except Exception:
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
            except Exception:
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
    ) -> dict[str, object]:
        """
        Attempt to heal pending DB->session-cache sync debt without network fetches.

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

        try:
            session_cache = get_session_bar_cache()
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
                    except Exception:
                        pass

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

    @staticmethod
    def _intraday_quality_caps(
        interval: str | None,
    ) -> tuple[float, float, float, float]:
        """
        Return (body_cap, span_cap, wick_cap, jump_cap) for intraday cleanup.

        Values are deliberately generous to avoid corrupting legitimate price
        moves (China A-shares can move +/-10% intraday; ST stocks +/-5%).
        Only truly malformed bars are removed.
        """
        iv = DataFetcher._normalize_interval_token(interval)
        caps = _INTRADAY_CAPS.get(iv)
        if caps:
            return caps
        # Default: conservative daily-like caps
        return 0.15, 0.22, 0.16, 0.22

    @classmethod
    def _intraday_frame_quality(
        cls,
        df: pd.DataFrame,
        interval: str,
    ) -> dict[str, float | bool]:
        """
        Score intraday frame quality.
        Higher score = cleaner and more usable bars.
        """
        if df is None or df.empty:
            return {
                "score": 0.0, "rows": 0.0,
                "stale_ratio": 1.0, "doji_ratio": 1.0,
                "zero_vol_ratio": 1.0, "extreme_ratio": 1.0,
                "suspect": True,
            }

        out = cls._clean_dataframe(
            df,
            interval=interval,
            preserve_truth=True,
            aggressive_repairs=False,
            allow_synthetic_index=False,
        )
        if out.empty:
            return {
                "score": 0.0, "rows": 0.0,
                "stale_ratio": 1.0, "doji_ratio": 1.0,
                "zero_vol_ratio": 1.0, "extreme_ratio": 1.0,
                "suspect": True,
            }

        body_cap, span_cap, wick_cap, _ = cls._intraday_quality_caps(interval)
        close_safe = out["close"].clip(lower=1e-8)
        rows_n = float(len(out))

        body = (out["open"] - out["close"]).abs() / close_safe
        span = (out["high"] - out["low"]).abs() / close_safe
        oc_top = out[["open", "close"]].max(axis=1)
        oc_bot = out[["open", "close"]].min(axis=1)
        upper_wick = (out["high"] - oc_top).clip(lower=0.0) / close_safe
        lower_wick = (oc_bot - out["low"]).clip(lower=0.0) / close_safe

        vol = (
            out["volume"] if "volume" in out.columns
            else pd.Series(0.0, index=out.index)
        ).fillna(0)

        zero_vol = vol <= 0

        # Stale detection: same close, flat OHLC, zero volume
        same_close = out["close"].diff().abs() <= (close_safe * 1e-6)
        flat_body  = body <= 1e-6
        flat_span  = span <= 2e-6
        stale_flat = same_close & flat_body & flat_span & zero_vol

        # Doji: near-zero body relative to span
        doji_ratio      = float((body <= (span.clip(lower=1e-8) * 0.05)).mean())
        stale_ratio     = float(stale_flat.mean())
        zero_vol_ratio  = float(zero_vol.mean())

        # Extreme: bars with body/span/wick far above expected cap
        extreme_mask = (
            (body > float(body_cap) * 2.0)
            | (span > float(span_cap) * 2.0)
            | (upper_wick > float(wick_cap) * 2.0)
            | (lower_wick > float(wick_cap) * 2.0)
        )
        extreme_ratio = float(extreme_mask.mean())

        # Score: penalize stale/flat bars more aggressively so frames
        # with many O=H=L=C zero-volume bars rank below cleaner sources.
        depth_score = min(1.0, rows_n / 600.0)
        stale_penalty = min(1.0, stale_ratio * 3.0)
        score = (
            (0.40 * depth_score)
            + (0.35 * (1.0 - stale_penalty))
            + (0.15 * (1.0 - min(1.0, zero_vol_ratio)))
            + (0.10 * (1.0 - min(1.0, extreme_ratio * 3.0)))
        )
        if doji_ratio > 0.95:
            score -= float((doji_ratio - 0.95) * 1.5)
        score = float(max(0.0, min(1.0, score)))

        suspect = bool(
            (rows_n < 40)
            or (stale_ratio >= 0.40)
            or (extreme_ratio >= 0.15)
            or (doji_ratio >= 0.98 and zero_vol_ratio >= 0.85)
        )
        return {
            "score":          score,
            "rows":           rows_n,
            "stale_ratio":    float(stale_ratio),
            "doji_ratio":     float(doji_ratio),
            "zero_vol_ratio": float(zero_vol_ratio),
            "extreme_ratio":  float(extreme_ratio),
            "suspect":        suspect,
        }

    @classmethod
    def _daily_frame_quality(cls, df: pd.DataFrame) -> dict[str, float | bool]:
        """Score daily/weekly/monthly history quality for source comparison."""
        if df is None or df.empty:
            return {
                "score": 0.0,
                "rows": 0.0,
                "stale_ratio": 1.0,
                "doji_ratio": 1.0,
                "zero_vol_ratio": 1.0,
                "extreme_ratio": 1.0,
                "suspect": True,
            }

        out = cls._clean_dataframe(
            df,
            interval="1d",
            preserve_truth=True,
            aggressive_repairs=False,
            allow_synthetic_index=False,
        )
        if out.empty:
            return {
                "score": 0.0,
                "rows": 0.0,
                "stale_ratio": 1.0,
                "doji_ratio": 1.0,
                "zero_vol_ratio": 1.0,
                "extreme_ratio": 1.0,
                "suspect": True,
            }

        rows_n = float(len(out))
        close = pd.to_numeric(out.get("close"), errors="coerce")
        close = close[close > 0] if isinstance(close, pd.Series) else pd.Series(dtype=float)
        if close.empty:
            return {
                "score": 0.0,
                "rows": 0.0,
                "stale_ratio": 1.0,
                "doji_ratio": 1.0,
                "zero_vol_ratio": 1.0,
                "extreme_ratio": 1.0,
                "suspect": True,
            }

        ret = close.pct_change().abs().fillna(0.0)
        extreme_ratio = float((ret > 0.22).mean())
        stale_ratio = float((close.diff().abs() <= 1e-8).mean())
        vol = (
            pd.to_numeric(out.get("volume"), errors="coerce").fillna(0.0)
            if "volume" in out.columns
            else pd.Series(0.0, index=out.index)
        )
        zero_vol_ratio = float((vol <= 0).mean())
        doji_ratio = 0.0

        depth_score = min(1.0, rows_n / 700.0)
        score = (
            (0.50 * depth_score)
            + (0.25 * (1.0 - min(1.0, extreme_ratio * 3.0)))
            + (0.15 * (1.0 - min(1.0, zero_vol_ratio)))
            + (0.10 * (1.0 - min(1.0, stale_ratio * 2.0)))
        )
        score = float(max(0.0, min(1.0, score)))

        suspect = bool(
            (rows_n < 25)
            or (extreme_ratio >= 0.10)
            or (zero_vol_ratio >= 0.65)
        )
        return {
            "score": score,
            "rows": rows_n,
            "stale_ratio": stale_ratio,
            "doji_ratio": doji_ratio,
            "zero_vol_ratio": zero_vol_ratio,
            "extreme_ratio": extreme_ratio,
            "suspect": suspect,
        }

    @staticmethod
    def _max_close_cluster_size(
        closes: list[float],
        tolerance_ratio: float,
    ) -> int:
        """Largest in-tolerance close-price cluster size."""
        vals: list[float] = []
        for v in closes:
            try:
                fv = float(v)
            except Exception:
                continue
            if np.isfinite(fv) and fv > 0:
                vals.append(fv)
        if not vals:
            return 0
        tol = float(max(0.0, tolerance_ratio))
        if tol <= 0:
            return 1
        best = 1
        for base in vals:
            denom = max(abs(float(base)), 1e-8)
            support = sum(
                1
                for px in vals
                if abs(float(px) - float(base)) / denom <= tol
            )
            if support > best:
                best = int(support)
        return int(best)

    @classmethod
    def _daily_consensus_quorum_meta(
        cls,
        collected: list[dict],
    ) -> dict[str, object]:
        """
        Compute daily provider quorum metadata.

        Quorum passes when at least ``required_sources`` providers align on a
        bar for a sufficient fraction of overlapping bars.
        """
        required_sources = int(
            max(
                2,
                int(
                    getattr(
                        getattr(CONFIG, "data", None),
                        "history_quorum_required_sources",
                        2,
                    )
                    or 2
                ),
            )
        )
        tolerance_bps = float(
            getattr(
                getattr(CONFIG, "data", None),
                "history_quorum_tolerance_bps",
                80.0,
            )
            or 80.0
        )
        tolerance_ratio = max(0.0, tolerance_bps / 10000.0)
        min_ratio = float(
            getattr(
                getattr(CONFIG, "data", None),
                "history_quorum_min_ratio",
                0.55,
            )
            or 0.55
        )
        min_ratio = float(min(1.0, max(0.0, min_ratio)))

        valid = [
            c
            for c in list(collected or [])
            if isinstance(c.get("df"), pd.DataFrame)
            and (not c["df"].empty)
            and str(c.get("source", "")).strip().lower() != "localdb"
        ]
        source_names = sorted(
            {
                str(c.get("source", "")).strip().lower()
                for c in valid
                if str(c.get("source", "")).strip()
            }
        )
        meta: dict[str, object] = {
            "required_sources": int(required_sources),
            "sources": list(source_names),
            "source_count": int(len(source_names)),
            "tolerance_bps": float(tolerance_bps),
            "min_ratio": float(min_ratio),
            "compared_points": 0,
            "agreeing_points": 0,
            "agreeing_ratio": 0.0,
            "quorum_passed": False,
            "reason": "",
        }
        if len(valid) < required_sources:
            meta["reason"] = "insufficient_sources"
            return meta

        all_idx = pd.Index([])
        for item in valid:
            all_idx = all_idx.union(item["df"].index)
        if all_idx.empty:
            meta["reason"] = "empty_index"
            return meta

        compared_points = 0
        agreeing_points = 0
        for ts in all_idx.sort_values():
            closes: list[float] = []
            for item in valid:
                frame = item["df"]
                if ts not in frame.index:
                    continue
                row_obj = frame.loc[ts]
                row = (
                    row_obj.iloc[-1]
                    if isinstance(row_obj, pd.DataFrame)
                    else row_obj
                )
                if not isinstance(row, pd.Series):
                    continue
                try:
                    close_px = float(row.get("close", 0.0) or 0.0)
                except Exception:
                    close_px = 0.0
                if close_px > 0 and np.isfinite(close_px):
                    closes.append(close_px)
            if len(closes) < required_sources:
                continue
            compared_points += 1
            cluster_size = cls._max_close_cluster_size(
                closes,
                tolerance_ratio=tolerance_ratio,
            )
            if int(cluster_size) >= required_sources:
                agreeing_points += 1

        agreeing_ratio = (
            float(agreeing_points) / float(compared_points)
            if compared_points > 0
            else 0.0
        )
        quorum_passed = bool(
            compared_points > 0
            and agreeing_ratio >= min_ratio
        )

        meta.update(
            {
                "compared_points": int(compared_points),
                "agreeing_points": int(agreeing_points),
                "agreeing_ratio": float(agreeing_ratio),
                "quorum_passed": bool(quorum_passed),
                "reason": "" if quorum_passed else "insufficient_consensus",
            }
        )
        return meta

    def _history_quorum_allows_persist(
        self,
        *,
        interval: str,
        symbol: str,
        meta: dict[str, object] | None,
    ) -> bool:
        """Strict daily quorum gate before DB persistence."""
        iv = self._normalize_interval_token(interval)
        if iv != "1d":
            return True

        if not isinstance(meta, dict) or not meta:
            log.debug(
                "History quorum metadata unavailable for %s (%s); allowing persist",
                str(symbol or ""),
                iv,
            )
            return True

        if bool(meta.get("quorum_passed", False)):
            return True

        log.warning(
            "Skipped DB persist for %s (%s): quorum failed "
            "(agree=%s/%s, ratio=%.2f, required=%s, sources=%s, reason=%s)",
            str(symbol or ""),
            iv,
            int(meta.get("agreeing_points", 0) or 0),
            int(meta.get("compared_points", 0) or 0),
            float(meta.get("agreeing_ratio", 0.0) or 0.0),
            int(meta.get("required_sources", 2) or 2),
            ",".join(str(x) for x in list(meta.get("sources", []) or [])),
            str(meta.get("reason", "quorum_failed")),
        )
        return False

    @classmethod
    def _merge_daily_by_consensus(
        cls,
        collected: list[dict],
        *,
        interval: str = "1d",
    ) -> pd.DataFrame:
        """
        Compare overlapping daily bars across sources and keep the row closest
        to per-timestamp consensus close price.
        """
        valid = [c for c in collected if isinstance(c.get("df"), pd.DataFrame) and not c["df"].empty]
        if not valid:
            return pd.DataFrame()
        if len(valid) == 1:
            return cls._clean_dataframe(valid[0]["df"], interval=interval)

        all_idx = pd.Index([])
        for item in valid:
            all_idx = all_idx.union(item["df"].index)
        if all_idx.empty:
            return pd.DataFrame()

        def _to_float(row: pd.Series, col: str, default: float = 0.0) -> float:
            try:
                val = row.get(col, default)
            except Exception:
                val = default
            try:
                return float(val)
            except Exception:
                return float(default)

        out_rows: list[dict[str, float]] = []
        out_index: list[pd.Timestamp] = []

        for ts in all_idx.sort_values():
            candidates: list[tuple[pd.Series, float, float, int]] = []
            for item in valid:
                frame = item["df"]
                if ts not in frame.index:
                    continue
                row_obj = frame.loc[ts]
                row = row_obj.iloc[-1] if isinstance(row_obj, pd.DataFrame) else row_obj
                if not isinstance(row, pd.Series):
                    continue
                close_px = _to_float(row, "close", 0.0)
                if close_px <= 0:
                    continue
                quality_score = float(dict(item.get("quality") or {}).get("score", 0.0))
                rank = int(item.get("rank", 0))
                candidates.append((row, close_px, quality_score, rank))

            if not candidates:
                continue

            chosen: pd.Series
            if len(candidates) == 1:
                chosen = candidates[0][0]
            else:
                med = float(np.median(np.array([c[1] for c in candidates], dtype=float)))

                def _cost(
                    candidate: tuple[pd.Series, float, float, int],
                    median: float = med,
                ) -> tuple[float, float, int]:
                    row, close_px, quality_score, rank = candidate
                    open_px = _to_float(row, "open", close_px)
                    high_px = _to_float(row, "high", max(open_px, close_px))
                    low_px = _to_float(row, "low", min(open_px, close_px))
                    vol = _to_float(row, "volume", 0.0)

                    dev = abs(close_px - median) / max(abs(median), 1e-8)
                    ohlc_penalty = 0.0
                    if high_px < max(open_px, close_px):
                        ohlc_penalty += 0.03
                    if low_px > min(open_px, close_px):
                        ohlc_penalty += 0.03
                    vol_penalty = 0.02 if vol < 0 else 0.0
                    return (float(dev + ohlc_penalty + vol_penalty), -quality_score, rank)

                chosen = min(candidates, key=_cost)[0]

            close_px = _to_float(chosen, "close", 0.0)
            if close_px <= 0:
                continue
            open_px = _to_float(chosen, "open", close_px)
            high_px = max(_to_float(chosen, "high", close_px), open_px, close_px)
            low_px = min(_to_float(chosen, "low", close_px), open_px, close_px)
            vol = max(0.0, _to_float(chosen, "volume", 0.0))
            amount = _to_float(chosen, "amount", 0.0)
            if amount <= 0:
                amount = close_px * vol

            out_rows.append(
                {
                    "open": open_px,
                    "high": high_px,
                    "low": low_px,
                    "close": close_px,
                    "volume": vol,
                    "amount": amount,
                }
            )
            out_index.append(pd.Timestamp(ts))

        if not out_rows:
            return pd.DataFrame()

        out = pd.DataFrame(out_rows, index=pd.DatetimeIndex(out_index, name="date"))
        return cls._clean_dataframe(out, interval=interval)

    @staticmethod
    def _drop_stale_flat_bars(df: pd.DataFrame) -> pd.DataFrame:
        """Remove completely flat bars (O=H=L=C, volume=0) before DB upsert.

        These bars carry no meaningful price information and contaminate
        the local database baseline, making future quality comparisons
        unreliable.
        """
        if df is None or df.empty:
            return df
        try:
            close_safe = df["close"].clip(lower=1e-8)
            body = (df["open"] - df["close"]).abs() / close_safe
            span = (df["high"] - df["low"]).abs() / close_safe
            vol = (
                df["volume"].fillna(0)
                if "volume" in df.columns
                else pd.Series(0.0, index=df.index)
            )
            stale = (body <= 1e-6) & (span <= 2e-6) & (vol <= 0)
            n_stale = int(stale.sum())
            if n_stale > 0:
                log.debug(
                    "Dropping %d flat stale bars before DB upsert (%d remaining)",
                    n_stale, len(df) - n_stale,
                )
                return df[~stale]
        except Exception as exc:
            log.debug("_drop_stale_flat_bars skipped: %s", exc)
        return df

    @classmethod
    def _cross_validate_bars(
        cls,
        best_df: pd.DataFrame,
        alternatives: list[pd.DataFrame],
        interval: str,
    ) -> pd.DataFrame:
        """Replace stale bars in *best_df* with non-stale bars from *alternatives*.

        A bar is considered stale when its close is unchanged from the
        prior bar, body and span are near-zero, and volume is zero.
        For each such bar we look for a matching timestamp in the
        alternative DataFrames and substitute the first non-stale hit.
        """
        if best_df.empty or not alternatives:
            return best_df

        close_safe = best_df["close"].clip(lower=1e-8)
        body = (best_df["open"] - best_df["close"]).abs() / close_safe
        span = (best_df["high"] - best_df["low"]).abs() / close_safe
        vol = (
            best_df["volume"].fillna(0)
            if "volume" in best_df.columns
            else pd.Series(0.0, index=best_df.index)
        )
        same_close = best_df["close"].diff().abs() <= (close_safe * 1e-6)
        stale_mask = same_close & (body <= 1e-6) & (span <= 2e-6) & (vol <= 0)

        stale_indices = best_df.index[stale_mask]
        if len(stale_indices) == 0:
            return best_df

        out = best_df.copy()
        remaining = set(stale_indices)
        for alt_df in alternatives:
            if alt_df.empty or not remaining:
                break
            overlap = alt_df.index.intersection(pd.Index(list(remaining)))
            if overlap.empty:
                continue
            for idx in overlap:
                try:
                    alt_row = alt_df.loc[idx]
                    alt_body = abs(
                        float(alt_row.get("open", 0) if isinstance(alt_row, dict) else alt_row["open"])
                        - float(alt_row.get("close", 0) if isinstance(alt_row, dict) else alt_row["close"])
                    )
                    alt_vol = float(
                        (alt_row.get("volume", 0) if isinstance(alt_row, dict) else alt_row["volume"]) or 0
                    )
                except Exception:
                    continue
                if alt_body > 1e-6 or alt_vol > 0:
                    for col in ("open", "high", "low", "close", "volume", "amount"):
                        if col in out.columns and col in alt_df.columns:
                            out.at[idx, col] = alt_df.at[idx, col]
                    remaining.discard(idx)
            if not remaining:
                break

        replaced = len(stale_indices) - len(remaining)
        if replaced > 0:
            log.debug(
                "Cross-validated %d/%d stale bars from alternative sources",
                replaced, len(stale_indices),
            )
        return out

    @classmethod
    def _to_shanghai_naive_ts(cls, value: object) -> pd.Timestamp:
        """
        Parse one timestamp-like value -> Asia/Shanghai naive time.
        Returns NaT on failure.
        """
        if value is None:
            return pd.NaT

        try:
            if isinstance(value, (int, float, np.integer, np.floating)):
                v = float(value)
                if not np.isfinite(v) or abs(v) < 1e9:
                    return pd.NaT
                if abs(v) >= 1e11:
                    v /= 1000.0
                ts = pd.to_datetime(v, unit="s", errors="coerce", utc=True)
            else:
                text = str(value).strip()
                if not text:
                    return pd.NaT
                if text.isdigit():
                    num = float(text)
                    if abs(num) < 1e9:
                        return pd.NaT
                    if abs(num) >= 1e11:
                        num /= 1000.0
                    ts = pd.to_datetime(num, unit="s", errors="coerce", utc=True)
                else:
                    ts = pd.to_datetime(value, errors="coerce")
        except Exception:
            return pd.NaT

        if pd.isna(ts):
            return pd.NaT

        try:
            ts_obj = pd.Timestamp(ts)
        except Exception:
            return pd.NaT

        try:
            if ts_obj.tzinfo is not None:
                ts_obj = ts_obj.tz_convert("Asia/Shanghai").tz_localize(None)
        except Exception:
            try:
                ts_obj = ts_obj.tz_localize(None)
            except Exception:
                return pd.NaT
        return ts_obj

    @classmethod
    def _normalize_datetime_index(
        cls,
        idx: object,
    ) -> pd.DatetimeIndex | None:
        """
        Convert an index-like object to DatetimeIndex in Asia/Shanghai naive time.
        Returns None when conversion is unreliable.
        """
        if isinstance(idx, pd.DatetimeIndex):
            out = idx
            try:
                if out.tz is not None:
                    out = out.tz_convert("Asia/Shanghai").tz_localize(None)
            except Exception:
                try:
                    out = out.tz_localize(None)
                except Exception as exc:
                    log.debug("DatetimeIndex tz normalization failed: %s", exc)
            return pd.DatetimeIndex(out)

        values = list(idx) if idx is not None else []
        if not values:
            return None

        parsed = [cls._to_shanghai_naive_ts(v) for v in values]
        dt = pd.DatetimeIndex(parsed)
        valid_ratio = float(dt.notna().sum()) / float(max(1, len(dt)))
        if valid_ratio < 0.80:
            return None
        return dt
    @classmethod
    def _clean_dataframe(
        cls,
        df: pd.DataFrame,
        interval: str | None = None,
        *,
        preserve_truth: bool | None = None,
        aggressive_repairs: bool | None = None,
        allow_synthetic_index: bool | None = None,
    ) -> pd.DataFrame:
        """
        Standardize and validate an OHLCV dataframe.

        Defaults are truth-preserving:
        - no synthetic intraday timestamps unless explicitly enabled
        - no aggressive intraday mutation unless explicitly enabled
        - duplicate timestamps keep first occurrence by default
        """
        if df is None or df.empty:
            return pd.DataFrame()

        if preserve_truth is None:
            preserve_truth = bool(
                getattr(CONFIG.data, "truth_preserving_cleaning", True)
            )
        if aggressive_repairs is None:
            aggressive_repairs = bool(
                getattr(CONFIG.data, "aggressive_intraday_repair", False)
            )
        if allow_synthetic_index is None:
            allow_synthetic_index = bool(
                getattr(CONFIG.data, "synthesize_intraday_index", False)
            )

        preserve_truth = bool(preserve_truth)
        aggressive_repairs = bool(aggressive_repairs)
        allow_synthetic_index = bool(allow_synthetic_index)

        out = df.copy()
        iv = cls._normalize_interval_token(interval)
        is_intraday = iv not in {"1d", "1wk", "1mo"}

        # 1) Normalize index to DatetimeIndex when reliable.
        norm_idx = cls._normalize_datetime_index(out.index)
        has_dt_index = norm_idx is not None
        if norm_idx is not None:
            out.index = norm_idx

        if not has_dt_index:
            parsed_dt = None
            for col in ("datetime", "timestamp", "date", "time"):
                if col not in out.columns:
                    continue
                dt = cls._normalize_datetime_index(out[col])
                if dt is None or len(dt) == 0:
                    continue
                if float(dt.notna().sum()) / float(len(dt)) >= 0.80:
                    parsed_dt = dt
                    break

            if parsed_dt is None:
                try:
                    idx_num = pd.to_numeric(
                        pd.Series(out.index, dtype=object), errors="coerce"
                    )
                    numeric_ratio = (
                        float(idx_num.notna().sum()) / float(len(idx_num))
                        if len(idx_num) > 0 else 0.0
                    )
                except Exception:
                    numeric_ratio = 0.0

                if numeric_ratio < 0.60:
                    dt = cls._normalize_datetime_index(out.index)
                    if dt is not None and len(dt) > 0:
                        if float(dt.notna().sum()) / float(len(dt)) >= 0.80:
                            parsed_dt = dt

            if parsed_dt is not None:
                out.index = parsed_dt
                has_dt_index = isinstance(out.index, pd.DatetimeIndex)
            else:
                if is_intraday and preserve_truth:
                    return pd.DataFrame()
                out = out.reset_index(drop=True)

        # 2) Deduplicate and order.
        if has_dt_index:
            out = out[~out.index.isna()]
            keep_mode = "first" if preserve_truth else "last"
            out = out[~out.index.duplicated(keep=keep_mode)].sort_index()

        # 3) Numeric coercion.
        for c in ("open", "high", "low", "close", "volume", "amount"):
            if c in out.columns:
                out[c] = pd.to_numeric(out[c], errors="coerce")

        # 4) Basic close validity.
        if "close" not in out.columns:
            return pd.DataFrame()
        out = out.dropna(subset=["close"])
        out = out[out["close"] > 0]
        if out.empty:
            return pd.DataFrame()

        # 5) Minimal open repair.
        if "open" not in out.columns:
            out["open"] = out["close"]
        out["open"] = pd.to_numeric(out["open"], errors="coerce").fillna(0.0)
        out["open"] = out["open"].where(out["open"] > 0, out["close"])

        # 6) Minimal high/low repair.
        if "high" not in out.columns:
            out["high"] = out[["open", "close"]].max(axis=1)
        else:
            out["high"] = pd.to_numeric(out["high"], errors="coerce")

        if "low" not in out.columns:
            out["low"] = out[["open", "close"]].min(axis=1)
        else:
            out["low"] = pd.to_numeric(out["low"], errors="coerce")

        out["high"] = pd.concat(
            [out["high"], out["open"], out["close"]], axis=1
        ).max(axis=1)
        out["low"] = pd.concat(
            [out["low"], out["open"], out["close"]], axis=1
        ).min(axis=1)

        # 7) Aggressive intraday mutation is disabled to preserve raw source truth.
        _ = aggressive_repairs

        # 8) Volume >= 0.
        if "volume" in out.columns:
            out = out[out["volume"].fillna(0) >= 0]
        else:
            out["volume"] = 0.0

        # 9) high >= low.
        if "high" in out.columns and "low" in out.columns:
            out = out[out["high"].fillna(0) >= out["low"].fillna(0)]

        # 10) Derive amount if missing.
        if (
            "amount" not in out.columns
            and "close" in out.columns
            and "volume" in out.columns
        ):
            out["amount"] = out["close"] * out["volume"]

        # 11) Final cleanup.
        out = out.replace([np.inf, -np.inf], np.nan)
        ohlc_cols = [c for c in ("open", "high", "low", "close") if c in out.columns]
        if not ohlc_cols:
            return pd.DataFrame()

        if preserve_truth:
            out = out.dropna(subset=ohlc_cols)
            if out.empty:
                return pd.DataFrame()
            if "volume" in out.columns:
                out["volume"] = pd.to_numeric(out["volume"], errors="coerce").fillna(0.0)
                out = out[out["volume"] >= 0]
            if "amount" in out.columns:
                out["amount"] = pd.to_numeric(out["amount"], errors="coerce").fillna(0.0)
        else:
            out[ohlc_cols] = out[ohlc_cols].ffill().bfill()
            out = out.fillna(0)

        if has_dt_index:
            keep_mode = "first" if preserve_truth else "last"
            out = out[~out.index.duplicated(keep=keep_mode)].sort_index()

        return out

    # ------------------------------------------------------------------
    # History orchestration (unchanged logic, minor robustness improvements)
    # ------------------------------------------------------------------

    def get_history(
        self,
        code: str,
        days: int = 500,
        bars: int | None = None,
        use_cache: bool = True,
        update_db: bool = True,
        instrument: dict | None = None,
        interval: str = "1d",
        max_age_hours: float | None = None,
        allow_online: bool = True,
        refresh_intraday_after_close: bool = False,
    ) -> pd.DataFrame:
        """Unified history fetcher. Priority: cache -> local DB -> online."""
        from core.instruments import instrument_key, parse_instrument

        inst = instrument or parse_instrument(code)
        key = instrument_key(inst)
        interval = self._normalize_interval_token(interval)
        offline = _is_offline() or (not bool(allow_online))
        force_exact_intraday = bool(
            refresh_intraday_after_close
            and self._should_refresh_intraday_exact(
                interval=interval,
                update_db=bool(update_db),
                allow_online=bool(allow_online),
            )
        )
        is_cn_equity = (
            inst.get("market") == "CN" and inst.get("asset") == "EQUITY"
        )

        count = self._resolve_requested_bar_count(
            days=days, bars=bars, interval=interval
        )
        max_days = INTERVAL_MAX_DAYS.get(interval, 10_000)
        fetch_days = min(bars_to_days(count, interval), max_days)

        if max_age_hours is not None:
            ttl = float(max_age_hours)
        elif interval == "1d":
            ttl = float(CONFIG.data.cache_ttl_hours)
        else:
            ttl = min(float(CONFIG.data.cache_ttl_hours), 1.0 / 120.0)

        cache_key = f"history:{key}:{interval}"
        stale_cached_df = pd.DataFrame()

        if use_cache and (not force_exact_intraday):
            cached_df = self._cache.get(cache_key, ttl)
            if isinstance(cached_df, pd.DataFrame) and not cached_df.empty:
                cached_df = self._clean_dataframe(cached_df, interval=interval)
                if (
                    is_cn_equity
                    and self._normalize_interval_token(interval)
                    not in {"1d", "1wk", "1mo"}
                ):
                    cached_df = self._filter_cn_intraday_session(
                        cached_df, interval
                    )
                stale_cached_df = cached_df
                if len(cached_df) >= min(count, 100):
                    return cached_df.tail(count)
                if offline and len(cached_df) >= max(20, min(count, 80)):
                    return cached_df.tail(count)

        session_df = pd.DataFrame()
        use_session_history = bool((not force_exact_intraday) and (not is_cn_equity))
        if use_session_history:
            session_df = self._get_session_history(
                symbol=str(inst.get("symbol", code)),
                interval=interval,
                bars=count,
            )
        if (
            use_session_history
            and interval in _INTRADAY_INTERVALS
            and not session_df.empty
            and count <= 500
            and len(session_df) >= count
        ):
            return self._cache_tail(
                cache_key,
                session_df,
                count,
                interval=interval,
            )

        if is_cn_equity and interval in _INTRADAY_INTERVALS:
            if force_exact_intraday:
                return self._get_history_cn_intraday_exact(
                    inst, count, fetch_days, interval, cache_key, offline,
                )
            persist_intraday_db = bool(update_db) and (
                not bool(CONFIG.is_market_open())
            )
            try:
                return self._get_history_cn_intraday(
                    inst, count, fetch_days, interval,
                    cache_key, offline, session_df,
                    persist_intraday_db=persist_intraday_db,
                )
            except TypeError:
                return self._get_history_cn_intraday(
                    inst, count, fetch_days, interval,
                    cache_key, offline, session_df,
                )

        if is_cn_equity and interval in {"1d", "1wk", "1mo"}:
            return self._get_history_cn_daily(
                inst, count, fetch_days, cache_key,
                offline, update_db, session_df, interval=interval,
            )

        # Non-CN instrument
        if offline:
            return (
                stale_cached_df.tail(count) if not stale_cached_df.empty else pd.DataFrame()
            )
        df = self._fetch_history_with_depth_retry(
            inst=inst,
            interval=interval,
            requested_count=count,
            base_fetch_days=fetch_days,
        )
        if df.empty:
            return pd.DataFrame()
        merged = self._merge_parts(df, session_df, interval=interval)
        if merged.empty:
            return pd.DataFrame()
        return self._cache_tail(
            cache_key,
            merged,
            count,
            interval=interval,
        )

    def _should_refresh_intraday_exact(
        self,
        *,
        interval: str,
        update_db: bool,
        allow_online: bool,
    ) -> bool:
        iv = self._normalize_interval_token(interval)
        if iv in {"1d", "1wk", "1mo"}:
            return False
        if (not bool(update_db)) or (not bool(allow_online)):
            return False
        if _is_offline():
            return False
        return bool(self._is_post_close_or_preopen_window())

    @staticmethod
    def _is_post_close_or_preopen_window() -> bool:
        """True when outside regular A-share trading session."""
        try:
            from zoneinfo import ZoneInfo
            now = datetime.now(tz=ZoneInfo("Asia/Shanghai"))
        except Exception:
            now = datetime.now()

        if now.weekday() >= 5:
            return True

        try:
            from core.constants import is_trading_day
            if not is_trading_day(now.date()):
                return True
        except Exception as exc:
            log.debug("Trading-day calendar lookup failed: %s", exc)

        t = CONFIG.trading
        cur = now.time()
        morning   = t.market_open_am <= cur <= t.market_close_am
        afternoon = t.market_open_pm <= cur <= t.market_close_pm
        lunch     = t.market_close_am < cur < t.market_open_pm
        if morning or afternoon or lunch:
            return False
        return True

    @staticmethod
    def _resolve_requested_bar_count(
        days: int,
        bars: int | None,
        interval: str,
    ) -> int:
        if bars is not None:
            return max(1, int(bars))
        iv = str(interval or "1d").lower()
        if iv == "1d":
            return max(1, int(days))
        day_count = max(1, int(days))
        bpd = BARS_PER_DAY.get(iv, 1.0)
        if bpd <= 0:
            bpd = 1.0
        approx = int(math.ceil(day_count * bpd))
        max_bars = max(1, int(INTERVAL_MAX_DAYS.get(iv, 365) * bpd))
        return max(1, min(approx, max_bars))

    def _fetch_history_with_depth_retry(
        self,
        inst: dict,
        interval: str,
        requested_count: int,
        base_fetch_days: int,
        return_meta: bool = False,
    ) -> pd.DataFrame | tuple[pd.DataFrame, dict[str, object]]:
        """Fetch history with adaptive depth retries."""
        iv = str(interval or "1d").lower()
        max_days = int(INTERVAL_MAX_DAYS.get(iv, 10_000))
        base = max(1, int(base_fetch_days))
        candidates = [base, int(base * 2.0), int(base * 3.0)]

        tried: set[int] = set()
        best = pd.DataFrame()
        best_meta: dict[str, object] = {}
        best_score = -1.0
        target = max(60, int(min(requested_count, 1200)))
        is_intraday = iv not in {"1d", "1wk", "1mo"}

        for days in candidates:
            d = max(1, min(int(days), max_days))
            if d in tried:
                continue
            tried.add(d)
            try:
                raw_out = self._fetch_from_sources_instrument(
                    inst,
                    days=d,
                    interval=iv,
                    include_localdb=not is_intraday,
                    return_meta=True,
                )
            except TypeError:
                raw_out = self._fetch_from_sources_instrument(
                    inst, days=d, interval=iv,
                )
            if (
                isinstance(raw_out, tuple)
                and len(raw_out) == 2
                and isinstance(raw_out[0], pd.DataFrame)
            ):
                raw_df = raw_out[0]
                meta = (
                    dict(raw_out[1])
                    if isinstance(raw_out[1], dict)
                    else {}
                )
            else:
                raw_df = (
                    raw_out
                    if isinstance(raw_out, pd.DataFrame)
                    else pd.DataFrame()
                )
                meta = {}
            df = self._clean_dataframe(raw_df, interval=iv)
            if df.empty:
                continue

            if is_intraday:
                q = self._intraday_frame_quality(df, iv)
                score = float(q.get("score", 0.0))
                if (
                    score > best_score + 0.02
                    or (abs(score - best_score) <= 0.02 and len(df) > len(best))
                ):
                    best = df
                    best_meta = dict(meta)
                    best_score = score
            else:
                if len(df) > len(best):
                    best = df
                    best_meta = dict(meta)

            if len(best) >= target:
                if (not is_intraday) or (best_score >= 0.28):
                    break

        if return_meta:
            return best, best_meta
        return best

    def _accept_online_intraday_snapshot(
        self,
        *,
        symbol: str,
        interval: str,
        online_df: pd.DataFrame,
        baseline_df: pd.DataFrame | None = None,
    ) -> bool:
        """Decide whether to trust an online intraday snapshot over baseline."""
        if online_df is None or online_df.empty:
            return False
        if baseline_df is None or baseline_df.empty:
            return True

        iv = self._normalize_interval_token(interval)
        oq = self._intraday_frame_quality(online_df, iv)
        bq = self._intraday_frame_quality(baseline_df, iv)
        online_score   = float(oq.get("score", 0.0))
        base_score     = float(bq.get("score", 0.0))
        online_suspect = bool(oq.get("suspect", False))

        online_fresher = False
        try:
            if (
                isinstance(online_df.index, pd.DatetimeIndex)
                and isinstance(baseline_df.index, pd.DatetimeIndex)
                and len(online_df.index) > 0
                and len(baseline_df.index) > 0
            ):
                step = int(max(1, self._interval_seconds(iv)))
                online_last = pd.Timestamp(online_df.index.max())
                base_last   = pd.Timestamp(baseline_df.index.max())
                online_fresher = bool(
                    online_last >= (base_last + pd.Timedelta(seconds=step))
                )
        except Exception:
            online_fresher = False

        reject = bool(
            (
                online_suspect
                and base_score >= (online_score + 0.08)
                and not online_fresher
            )
            or (
                float(oq.get("stale_ratio", 0.0)) >= 0.35
                and float(bq.get("stale_ratio", 0.0))
                    <= (float(oq.get("stale_ratio", 0.0)) - 0.20)
            )
            or (
                float(oq.get("rows", 0.0)) < max(40.0, float(bq.get("rows", 0.0)) * 0.20)
                and online_score < base_score
                and not online_fresher
            )
        )
        if reject:
            log.warning(
                "Rejected weak online snapshot for %s (%s): "
                "online score=%.3f stale=%.1f%% rows=%d; "
                "baseline score=%.3f rows=%d",
                str(symbol or ""), iv,
                online_score,
                float(oq.get("stale_ratio", 0.0)) * 100.0,
                int(oq.get("rows", 0.0)),
                base_score,
                int(bq.get("rows", 0.0)),
            )
            return False
        return True

    def _get_history_cn_intraday(
        self,
        inst: dict,
        count: int,
        fetch_days: int,
        interval: str,
        cache_key: str,
        offline: bool,
        session_df: pd.DataFrame | None = None,
        *,
        persist_intraday_db: bool = True,
    ) -> pd.DataFrame:
        """Handle CN equity intraday intervals using online multi-source fetch."""
        code6 = str(inst["symbol"]).zfill(6)
        db_df = pd.DataFrame()
        db_limit = int(max(count * 3, count + 600))
        try:
            db_df = self._clean_dataframe(
                self._db.get_intraday_bars(
                    code6, interval=interval, limit=db_limit
                ),
                interval=interval,
            )
            db_df = self._filter_cn_intraday_session(db_df, interval)
        except Exception as exc:
            log.warning(
                "Intraday DB read failed for %s (%s): %s",
                code6, interval, exc,
            )

        online_df = pd.DataFrame()
        if not offline:
            online_df = self._fetch_history_with_depth_retry(
                inst=inst, interval=interval,
                requested_count=count, base_fetch_days=fetch_days,
            )
            online_df = self._filter_cn_intraday_session(online_df, interval)

        if online_df is not None and (not online_df.empty):
            if not self._accept_online_intraday_snapshot(
                symbol=code6,
                interval=interval,
                online_df=online_df,
                baseline_df=db_df,
            ):
                online_df = pd.DataFrame()

        if offline or online_df.empty:
            if db_df.empty:
                return pd.DataFrame()
            return self._cache_tail(
                cache_key,
                db_df,
                count,
                interval=interval,
            )

        # Prefer fresh online rows when timestamps overlap with local DB rows.
        merged = self._merge_parts(online_df, db_df, interval=interval)
        merged = self._filter_cn_intraday_session(merged, interval)
        if merged.empty:
            return pd.DataFrame()

        out = self._cache_tail(
            cache_key,
            merged,
            count,
            interval=interval,
        )
        if bool(persist_intraday_db):
            try:
                self._db.upsert_intraday_bars(code6, interval, online_df)
            except Exception as exc:
                log.warning(
                    "Intraday DB upsert failed for %s (%s): %s",
                    code6, interval, exc,
                )
        return out

    def _get_history_cn_intraday_exact(
        self,
        inst: dict,
        count: int,
        fetch_days: int,
        interval: str,
        cache_key: str,
        offline: bool,
    ) -> pd.DataFrame:
        """Post-close exact mode: refresh online bars, then update DB."""
        code6 = str(inst["symbol"]).zfill(6)
        online_df = pd.DataFrame()
        if not offline:
            online_df = self._fetch_history_with_depth_retry(
                inst=inst, interval=interval,
                requested_count=count, base_fetch_days=fetch_days,
            )
            online_df = self._filter_cn_intraday_session(online_df, interval)

        db_df = pd.DataFrame()
        db_limit = int(max(count * 3, count + 600))
        try:
            db_df = self._clean_dataframe(
                self._db.get_intraday_bars(
                    code6, interval=interval, limit=db_limit
                ),
                interval=interval,
            )
            db_df = self._filter_cn_intraday_session(db_df, interval)
        except Exception as exc:
            log.warning(
                "Intraday exact DB read failed for %s (%s): %s",
                code6, interval, exc,
            )

        if online_df is None or online_df.empty:
            if db_df is None or db_df.empty:
                return pd.DataFrame()
            return self._cache_tail(
                cache_key,
                db_df,
                count,
                interval=interval,
            )

        # Prefer fresh online rows when timestamps overlap with local DB rows.
        merged = self._merge_parts(online_df, db_df, interval=interval)
        merged = self._filter_cn_intraday_session(merged, interval)
        if merged.empty:
            return pd.DataFrame()

        out = self._cache_tail(
            cache_key,
            merged,
            count,
            interval=interval,
        )
        try:
            self._db.upsert_intraday_bars(code6, interval, online_df)
        except Exception as exc:
            log.warning(
                "Intraday exact DB upsert failed for %s (%s): %s",
                code6, interval, exc,
            )
        return out

    def _get_history_cn_daily(
        self,
        inst: dict,
        count: int,
        fetch_days: int,
        cache_key: str,
        offline: bool,
        update_db: bool,
        session_df: pd.DataFrame | None = None,
        interval: str = "1d",
    ) -> pd.DataFrame:
        """Handle CN equity daily/weekly/monthly intervals via online consensus."""
        iv = self._normalize_interval_token(interval)
        db_limit = (
            int(max(count, fetch_days))
            if iv == "1d"
            else int(max(count * 8, fetch_days))
        )
        db_df = self._clean_dataframe(
            self._db.get_bars(inst["symbol"], limit=db_limit),
            interval="1d",
        )
        base_df = self._resample_daily_to_interval(
            db_df,
            iv,
        )

        if offline:
            return base_df.tail(count) if not base_df.empty else pd.DataFrame()

        online_meta: dict[str, object] = {}
        try:
            online_out = self._fetch_history_with_depth_retry(
                inst=inst,
                interval=iv,
                requested_count=count,
                base_fetch_days=fetch_days,
                return_meta=True,
            )
        except TypeError:
            online_out = self._fetch_history_with_depth_retry(
                inst=inst, interval=iv,
                requested_count=count, base_fetch_days=fetch_days,
            )
        if (
            isinstance(online_out, tuple)
            and len(online_out) == 2
            and isinstance(online_out[0], pd.DataFrame)
        ):
            online_df = online_out[0]
            if isinstance(online_out[1], dict):
                online_meta = dict(online_out[1])
        else:
            online_df = (
                online_out if isinstance(online_out, pd.DataFrame) else pd.DataFrame()
            )
        if online_df is None or online_df.empty:
            return base_df.tail(count) if not base_df.empty else pd.DataFrame()

        # Prefer fresh online rows when timestamps overlap with local DB rows.
        merged = self._merge_parts(online_df, base_df, interval=iv)
        if merged.empty:
            return pd.DataFrame()

        out = self._cache_tail(
            cache_key,
            merged,
            count,
            interval=iv,
        )
        if update_db and iv == "1d":
            if not self._history_quorum_allows_persist(
                interval=iv,
                symbol=str(inst.get("symbol", "")),
                meta=online_meta,
            ):
                return out
            try:
                self._db.upsert_bars(inst["symbol"], online_df)
            except Exception as exc:
                log.warning(
                    "Daily DB upsert failed for %s: %s",
                    str(inst.get("symbol", "")), exc,
                )
        return out

    def refresh_trained_stock_history(
        self,
        codes: list[str],
        *,
        interval: str = "1m",
        window_days: int = 29,
        allow_online: bool = True,
        sync_session_cache: bool = True,
        replace_realtime_after_close: bool = True,
    ) -> dict[str, object]:
        """
        Refresh recent history for trained stocks from online providers and persist to DB.

        Incremental behavior:
        - Default anchor is the last saved official-history cache timestamp.
        - If no official-history cache anchor exists, use DB/cache latest timestamp.
        - After market close, if realtime rows exist and replacement is enabled,
          fetch from the first realtime timestamp and replace those realtime rows
          with official-history rows in session cache.
        """
        iv = self._normalize_interval_token(interval)
        wd = max(1, int(window_days or 29))
        intraday = iv not in {"1d", "1wk", "1mo"}
        bpd = float(BARS_PER_DAY.get(iv, 1.0) or 1.0)
        if bpd <= 0:
            bpd = 1.0

        target_bars = int(max(1, math.ceil(float(wd) * bpd)))
        db_limit = int(max(target_bars * 2, target_bars + 800))
        max_api_days = int(max(1, INTERVAL_MAX_DAYS.get(iv, wd)))
        now = self._now_shanghai_naive()
        window_start = pd.Timestamp(now - timedelta(days=wd))
        market_open = bool(CONFIG.is_market_open())
        do_sync_cache = bool(sync_session_cache)

        session_cache = None
        if do_sync_cache:
            try:
                session_cache = get_session_bar_cache()
            except Exception as exc:
                log.debug("Session cache unavailable for refresh: %s", exc)
                session_cache = None

        codes6 = list(
            dict.fromkeys(
                c for c in (self.clean_code(x) for x in list(codes or [])) if c
            )
        )
        report: dict[str, object] = {
            "interval": iv,
            "window_days": int(wd),
            "window_bars": int(target_bars),
            "total": int(len(codes6)),
            "completed": 0,
            "updated": 0,
            "cached": 0,
            "rows": {},
            "fetched_days": {},
            "purged_realtime_rows": {},
            "cache_markers": {},
            "replacement_anchor_used": {},
            "cache_sync_errors": {},
            "quorum_blocked": {},
            "status": {},
            "errors": {},
        }
        reconcile_queue = self._load_refresh_reconcile_queue()
        reconcile_dirty = False
        report["pending_reconcile_before"] = int(len(reconcile_queue))
        report["pending_reconcile_after"] = int(len(reconcile_queue))
        report["pending_reconcile_codes"] = sorted(list(reconcile_queue.keys()))

        for idx, code6 in enumerate(codes6, start=1):
            fetched = pd.DataFrame()
            fetched_meta: dict[str, object] = {}
            quorum_blocked = False
            fetched_days = int(wd)
            purged_rows = 0
            cache_sync_attempted = False
            cache_sync_errors: list[str] = []
            pending_key = self._refresh_reconcile_key(code6, iv)
            had_pending = bool(pending_key and pending_key in reconcile_queue)
            try:
                if intraday:
                    db_df = self._clean_dataframe(
                        self._db.get_intraday_bars(
                            code6, interval=iv, limit=db_limit
                        ),
                        interval=iv,
                    )
                    db_df = self._filter_cn_intraday_session(db_df, iv)
                else:
                    db_df = self._clean_dataframe(
                        self._db.get_bars(code6, limit=db_limit),
                        interval="1d",
                    )
                    db_df = self._resample_daily_to_interval(db_df, iv)

                if (
                    not db_df.empty
                    and isinstance(db_df.index, pd.DatetimeIndex)
                ):
                    db_recent = db_df.loc[db_df.index >= window_start]
                else:
                    db_recent = db_df

                first_rt_ts = None
                first_rt_after_ak_ts = None
                last_ak_ts = None
                last_cache_ts = None
                purge_required = False
                purge_attempted = False
                if intraday and session_cache is not None:
                    try:
                        markers = session_cache.describe_symbol_interval(code6, iv)
                    except Exception as exc:
                        log.debug(
                            "Session cache marker read failed for %s (%s): %s",
                            code6, iv, exc,
                        )
                        markers = {}
                    first_rt_ts = markers.get("first_realtime_ts")
                    first_rt_after_ak_ts = markers.get("first_realtime_after_akshare_ts")
                    last_ak_ts = markers.get("last_akshare_ts")
                    last_cache_ts = markers.get("last_ts")
                    report_markers = dict(report.get("cache_markers") or {})
                    report_markers[code6] = {
                        "first_realtime_ts": (
                            pd.Timestamp(first_rt_ts).isoformat()
                            if first_rt_ts is not None
                            else None
                        ),
                        "last_akshare_ts": (
                            pd.Timestamp(last_ak_ts).isoformat()
                            if last_ak_ts is not None
                            else None
                        ),
                        "last_cache_ts": (
                            pd.Timestamp(last_cache_ts).isoformat()
                            if last_cache_ts is not None
                            else None
                        ),
                        "first_realtime_after_akshare_ts": (
                            pd.Timestamp(first_rt_after_ak_ts).isoformat()
                            if first_rt_after_ak_ts is not None
                            else None
                        ),
                    }
                    report["cache_markers"] = report_markers

                purge_required = bool(
                    intraday
                    and bool(replace_realtime_after_close)
                    and (first_rt_after_ak_ts is not None)
                )

                replace_realtime = bool(
                    intraday
                    and (session_cache is not None)
                    and (not market_open)
                    and bool(replace_realtime_after_close)
                    and (first_rt_after_ak_ts is not None)
                )

                anchor_ts = None
                if replace_realtime:
                    anchor_ts = pd.Timestamp(first_rt_after_ak_ts)
                elif last_ak_ts is not None:
                    anchor_ts = pd.Timestamp(last_ak_ts)
                elif last_cache_ts is not None:
                    anchor_ts = pd.Timestamp(last_cache_ts)
                elif (
                    not db_recent.empty
                    and isinstance(db_recent.index, pd.DatetimeIndex)
                ):
                    anchor_ts = pd.Timestamp(db_recent.index.max())
                else:
                    anchor_ts = pd.Timestamp(window_start)

                if anchor_ts is not None:
                    try:
                        if anchor_ts.tzinfo is not None:
                            anchor_ts = anchor_ts.tz_localize(None)
                    except Exception:
                        pass

                if (
                    anchor_ts is not None
                    and (not replace_realtime)
                    and (anchor_ts < window_start)
                ):
                    anchor_ts = pd.Timestamp(window_start)

                fetched_days = int(wd)
                if anchor_ts is not None:
                    try:
                        gap_seconds = float(
                            max(
                                0.0,
                                (
                                    now - pd.Timestamp(anchor_ts).to_pydatetime()
                                ).total_seconds(),
                            )
                        )
                        step_seconds = float(max(60, self._interval_seconds(iv)))
                        if gap_seconds <= (step_seconds * 1.1):
                            fetched_days = 1
                        else:
                            fetched_days = int(
                                max(1, math.ceil(gap_seconds / 86400.0) + 1)
                            )
                    except Exception:
                        fetched_days = int(wd)
                fetched_days = int(min(max(1, fetched_days), max_api_days))

                report_days = dict(report.get("fetched_days") or {})
                report_days[code6] = int(fetched_days)
                report["fetched_days"] = report_days

                if bool(allow_online) and (not _is_offline()) and fetched_days > 0:
                    inst = {"market": "CN", "asset": "EQUITY", "symbol": code6}
                    try:
                        fetched_out = self._fetch_from_sources_instrument(
                            inst=inst,
                            days=fetched_days,
                            interval=iv,
                            include_localdb=False,
                            return_meta=True,
                        )
                    except TypeError:
                        fetched_out = self._fetch_from_sources_instrument(
                            inst=inst,
                            days=fetched_days,
                            interval=iv,
                            include_localdb=False,
                        )
                    if (
                        isinstance(fetched_out, tuple)
                        and len(fetched_out) == 2
                        and isinstance(fetched_out[0], pd.DataFrame)
                    ):
                        fetched = fetched_out[0]
                        if isinstance(fetched_out[1], dict):
                            fetched_meta = dict(fetched_out[1])
                    else:
                        fetched = (
                            fetched_out
                            if isinstance(fetched_out, pd.DataFrame)
                            else pd.DataFrame()
                        )
                    fetched = self._clean_dataframe(fetched, interval=iv)
                    if intraday:
                        fetched = self._filter_cn_intraday_session(fetched, iv)

                    if (
                        not fetched.empty
                        and isinstance(fetched.index, pd.DatetimeIndex)
                    ):
                        lower_bound = window_start - pd.Timedelta(days=2)
                        if anchor_ts is not None:
                            lower_bound = min(
                                lower_bound,
                                pd.Timestamp(anchor_ts) - pd.Timedelta(days=1),
                            )
                        fetched = fetched.loc[fetched.index >= lower_bound]
                        fetched_meta["selected_rows"] = int(len(fetched))

                    if not fetched.empty:
                        if intraday:
                            self._db.upsert_intraday_bars(code6, iv, fetched)
                        else:
                            if self._history_quorum_allows_persist(
                                interval=iv,
                                symbol=code6,
                                meta=fetched_meta,
                            ):
                                self._db.upsert_bars(code6, fetched)
                            else:
                                quorum_blocked = True
                                fetched = pd.DataFrame()

                cache_sync_frame = fetched
                if (
                    intraday
                    and cache_sync_frame.empty
                    and had_pending
                    and isinstance(db_recent, pd.DataFrame)
                    and (not db_recent.empty)
                    and isinstance(db_recent.index, pd.DatetimeIndex)
                ):
                    cache_sync_frame = db_recent.copy()

                if bool(do_sync_cache):
                    if session_cache is None:
                        if intraday and (not cache_sync_frame.empty):
                            cache_sync_errors.append("session_cache_unavailable")
                    else:
                        if not cache_sync_frame.empty:
                            cache_sync_attempted = True
                            try:
                                session_cache.upsert_history_frame(
                                    code6,
                                    iv,
                                    cache_sync_frame,
                                    source="official_history",
                                    is_final=True,
                                )
                            except Exception as exc:
                                msg = str(exc)
                                cache_sync_errors.append(f"upsert_failed:{msg}")
                                log.warning(
                                    "Session cache upsert failed for %s (%s): %s",
                                    code6,
                                    iv,
                                    msg,
                                )

                        if (
                            replace_realtime
                            and (anchor_ts is not None)
                            and (not cache_sync_frame.empty)
                        ):
                            cache_sync_attempted = True
                            purge_attempted = True
                            purge_anchor = pd.Timestamp(anchor_ts)
                            if isinstance(cache_sync_frame.index, pd.DatetimeIndex):
                                try:
                                    fetched_min_ts = pd.Timestamp(cache_sync_frame.index.min())
                                    if fetched_min_ts.tzinfo is not None:
                                        fetched_min_ts = fetched_min_ts.tz_localize(None)
                                    if fetched_min_ts > purge_anchor:
                                        log.warning(
                                            (
                                                "Realtime replacement window limited for %s (%s): "
                                                "requested_anchor=%s, fetched_start=%s; "
                                                "preserving older realtime cache rows."
                                            ),
                                            code6,
                                            iv,
                                            purge_anchor.isoformat(),
                                            fetched_min_ts.isoformat(),
                                        )
                                        purge_anchor = fetched_min_ts
                                except Exception:
                                    pass

                            report_anchor = dict(report.get("replacement_anchor_used") or {})
                            report_anchor[code6] = str(purge_anchor.isoformat())
                            report["replacement_anchor_used"] = report_anchor

                            try:
                                purged_rows = int(
                                    session_cache.purge_realtime_rows(
                                        code6,
                                        iv,
                                        since_ts=purge_anchor,
                                    )
                                )
                            except Exception as exc:
                                msg = str(exc)
                                cache_sync_errors.append(f"purge_failed:{msg}")
                                log.debug(
                                    "Session realtime purge failed for %s (%s): %s",
                                    code6,
                                    iv,
                                    msg,
                                )
                                purged_rows = 0

                if cache_sync_errors:
                    sync_msg = "; ".join(cache_sync_errors)
                    report_sync = dict(report.get("cache_sync_errors") or {})
                    report_sync[code6] = sync_msg
                    report["cache_sync_errors"] = report_sync
                    if intraday and bool(do_sync_cache) and (not cache_sync_frame.empty):
                        if self._mark_refresh_reconcile_pending(
                            reconcile_queue,
                            code6,
                            iv,
                            error_text=sync_msg,
                        ):
                            reconcile_dirty = True
                elif had_pending and cache_sync_attempted and (
                    (not purge_required) or purge_attempted
                ):
                    if self._clear_refresh_reconcile_pending(
                        reconcile_queue,
                        code6,
                        iv,
                    ):
                        reconcile_dirty = True

                report_purged = dict(report.get("purged_realtime_rows") or {})
                report_purged[code6] = int(max(0, purged_rows))
                report["purged_realtime_rows"] = report_purged
                report_quorum = dict(report.get("quorum_blocked") or {})
                report_quorum[code6] = bool(quorum_blocked)
                report["quorum_blocked"] = report_quorum

                if intraday:
                    db_after = self._clean_dataframe(
                        self._db.get_intraday_bars(
                            code6, interval=iv, limit=db_limit
                        ),
                        interval=iv,
                    )
                    db_after = self._filter_cn_intraday_session(db_after, iv)
                else:
                    db_after = self._clean_dataframe(
                        self._db.get_bars(code6, limit=db_limit),
                        interval="1d",
                    )
                    db_after = self._resample_daily_to_interval(db_after, iv)

                if (
                    not db_after.empty
                    and isinstance(db_after.index, pd.DatetimeIndex)
                ):
                    db_after = db_after.loc[db_after.index >= window_start]

                if (
                    not db_after.empty
                    and isinstance(db_after, pd.DataFrame)
                ):
                    try:
                        from core.instruments import instrument_key

                        hist_key = instrument_key(
                            {
                                "market": "CN",
                                "asset": "EQUITY",
                                "symbol": code6,
                            }
                        )
                        cache_key = f"history:{hist_key}:{iv}"
                        keep_rows = min(
                            len(db_after),
                            self._history_cache_store_rows(iv, target_bars),
                        )
                        self._cache.set(
                            cache_key,
                            db_after.tail(max(1, int(keep_rows))).copy(),
                        )
                    except Exception as exc:
                        log.debug(
                            "History cache refresh sync failed for %s (%s): %s",
                            code6,
                            iv,
                            exc,
                        )

                rows = int(len(db_after.tail(target_bars)))
                report_rows = dict(report.get("rows") or {})
                report_rows[code6] = rows
                report["rows"] = report_rows

                report_status = dict(report.get("status") or {})
                if not fetched.empty:
                    report_status[code6] = "updated"
                    report["updated"] = int(report.get("updated", 0)) + 1
                elif quorum_blocked:
                    report_status[code6] = "quorum_blocked"
                elif rows > 0:
                    report_status[code6] = "cached"
                    report["cached"] = int(report.get("cached", 0)) + 1
                else:
                    report_status[code6] = "empty"
                report["status"] = report_status

            except Exception as exc:
                report_status = dict(report.get("status") or {})
                report_status[code6] = "error"
                report["status"] = report_status
                report_errors = dict(report.get("errors") or {})
                report_errors[code6] = str(exc)
                report["errors"] = report_errors
                log.warning(
                    "Trained-stock history refresh failed for %s (%s): %s",
                    code6, iv, exc,
                )
            finally:
                report["completed"] = int(idx)

        if reconcile_dirty:
            self._save_refresh_reconcile_queue(reconcile_queue)
        report["pending_reconcile_after"] = int(len(reconcile_queue))
        report["pending_reconcile_codes"] = sorted(list(reconcile_queue.keys()))

        return report

    @classmethod
    def _resample_daily_to_interval(
        cls,
        df: pd.DataFrame,
        interval: str,
    ) -> pd.DataFrame:
        """Resample daily OHLCV bars to weekly/monthly bars when requested."""
        iv = cls._normalize_interval_token(interval)
        if iv == "1d":
            return cls._clean_dataframe(df, interval="1d")
        if iv not in {"1wk", "1mo"}:
            return cls._clean_dataframe(df, interval=iv)

        daily = cls._clean_dataframe(df, interval="1d")
        if daily.empty or not isinstance(daily.index, pd.DatetimeIndex):
            return pd.DataFrame()

        rule = "W-FRI" if iv == "1wk" else "ME"
        agg: dict[str, str] = {}
        if "open" in daily.columns:
            agg["open"] = "first"
        if "high" in daily.columns:
            agg["high"] = "max"
        if "low" in daily.columns:
            agg["low"] = "min"
        if "close" in daily.columns:
            agg["close"] = "last"
        if "volume" in daily.columns:
            agg["volume"] = "sum"
        if "amount" in daily.columns:
            agg["amount"] = "sum"
        if not agg:
            return pd.DataFrame()

        resampled = daily.resample(rule).agg(agg)
        return cls._clean_dataframe(resampled, interval=iv)

    def _merge_parts(
        self,
        *dfs: pd.DataFrame,
        interval: str | None = None,
    ) -> pd.DataFrame:
        """Merge and deduplicate non-empty dataframes."""
        parts = [
            p for p in dfs
            if isinstance(p, pd.DataFrame) and not p.empty
        ]
        if not parts:
            return pd.DataFrame()
        if len(parts) == 1:
            return self._clean_dataframe(parts[0], interval=interval)
        return self._clean_dataframe(
            pd.concat(parts, axis=0),
            interval=interval,
        )

    @classmethod
    def _filter_cn_intraday_session(
        cls,
        df: pd.DataFrame,
        interval: str,
    ) -> pd.DataFrame:
        """Keep only regular CN A-share intraday session rows."""
        iv = cls._normalize_interval_token(interval)
        if iv in {"1d", "1wk", "1mo"}:
            return df if isinstance(df, pd.DataFrame) else pd.DataFrame()
        if df is None or df.empty:
            return pd.DataFrame()

        out = cls._clean_dataframe(df, interval=iv)
        if out.empty or not isinstance(out.index, pd.DatetimeIndex):
            return out

        idx  = out.index
        hhmm = (idx.hour * 100) + idx.minute
        in_morning   = (hhmm >= 930)  & (hhmm <= 1130)
        in_afternoon = (hhmm >= 1300) & (hhmm <= 1500)
        weekday      = idx.dayofweek < 5
        mask = weekday & (in_morning | in_afternoon)
        return out.loc[mask]

    @classmethod
    def _history_cache_store_rows(
        cls,
        interval: str | None,
        requested_rows: int,
    ) -> int:
        """
        Compute how many rows to keep in the shared history cache key.

        A larger shared window prevents cache-key fragmentation while still
        bounding memory and disk usage.
        """
        iv = cls._normalize_interval_token(interval)
        req = max(1, int(requested_rows or 1))
        if iv in _INTRADAY_INTERVALS:
            floor = max(200, min(2400, req * 3))
        else:
            floor = max(400, min(5000, req * 2))
        return int(max(req, floor))

    def _cache_tail(
        self,
        cache_key: str,
        df: pd.DataFrame,
        count: int,
        *,
        interval: str | None = None,
    ) -> pd.DataFrame:
        out = df.tail(count).copy()
        keep_rows = min(
            len(df),
            self._history_cache_store_rows(interval, count),
        )
        cache_df = df.tail(max(1, int(keep_rows))).copy()
        self._cache.set(cache_key, cache_df)
        return out

    def _get_session_history(
        self, symbol: str, interval: str, bars: int
    ) -> pd.DataFrame:
        try:
            cache = get_session_bar_cache()
            return self._clean_dataframe(
                cache.read_history(
                    symbol=symbol, interval=interval, bars=bars
                ),
                interval=interval,
            )
        except Exception as exc:
            log.debug("Session cache lookup failed: %s", exc)
            return pd.DataFrame()

    def get_realtime(
        self, code: str, instrument: dict | None = None
    ) -> Quote | None:
        """Get real-time quote for a single instrument."""
        from core.instruments import parse_instrument
        inst = instrument or parse_instrument(code)

        if _is_offline():
            return None

        if inst.get("market") == "CN" and inst.get("asset") == "EQUITY":
            code6 = str(inst.get("symbol") or "").zfill(6)
            if code6:
                return self._get_realtime_cn(code6)

        return self._get_realtime_generic(inst)

    def _get_realtime_cn(self, code6: str) -> Quote | None:
        """Optimized CN equity real-time via batch + micro-cache."""
        now = time.time()
        with self._rt_cache_lock:
            rec = self._rt_single_microcache.get(code6)
            if rec and (now - float(rec["ts"])) < _MICRO_CACHE_TTL:
                return rec["q"]  # type: ignore[return-value]

        try:
            out = self.get_realtime_batch([code6])
            q = out.get(code6)
            if q and q.price > 0:
                with self._rt_cache_lock:
                    self._rt_single_microcache[code6] = {"ts": now, "q": q}
                return q
        except Exception as exc:
            log.debug("CN realtime batch fetch failed for %s: %s", code6, exc)

        with self._last_good_lock:
            q = self._last_good_quotes.get(code6)
            if q and q.price > 0:
                age = self._quote_age_seconds(q)
                if age <= _LAST_GOOD_MAX_AGE:
                    return self._mark_quote_as_delayed(q)
        return None

    def _get_realtime_generic(self, inst: dict) -> Quote | None:
        """Fetch real-time quote from all sources, pick best."""
        candidates: list[Quote] = []
        with self._rate_limiter:
            for source in self._get_active_sources():
                try:
                    fn = getattr(source, "get_realtime_instrument", None)
                    if callable(fn):
                        q = fn(inst)
                    else:
                        q = source.get_realtime(inst.get("symbol", ""))
                    if q and q.price and q.price > 0:
                        candidates.append(q)
                except Exception:
                    continue

        if not candidates:
            return None

        prices = np.array([c.price for c in candidates], dtype=float)
        med = float(np.median(prices))
        good = [
            c for c in candidates
            if abs(c.price - med) / max(med, 1e-8) < 0.02
        ]
        pool = good if good else candidates
        pool.sort(key=lambda q: (q.is_delayed, q.latency_ms))
        best = pool[0]

        if inst.get("market") == "CN" and inst.get("asset") == "EQUITY":
            code6 = str(inst.get("symbol") or "").zfill(6)
            with self._last_good_lock:
                self._last_good_quotes[code6] = best

        return best

    def get_multiple_parallel(
        self,
        codes: list[str],
        days: int = 500,
        callback: Callable[[str, int, int], None] | None = None,
        max_workers: int | None = None,
        interval: str = "1d",
    ) -> dict[str, pd.DataFrame]:
        """Fetch history for multiple codes in parallel."""
        results: dict[str, pd.DataFrame] = {}
        total = len(codes)
        completed = 0
        lock = threading.Lock()

        for source in self._all_sources:
            with source._lock:
                source.status.consecutive_errors = 0
                source.status.disabled_until = None

        def fetch_one(code: str) -> tuple[str, pd.DataFrame]:
            try:
                df = self.get_history(code, days, interval=interval)
                return code, df
            except Exception as exc:
                log.debug("Failed to fetch %s: %s", code, exc)
                return code, pd.DataFrame()

        default_workers = 2 if interval in _INTRADAY_INTERVALS else 5
        cap_workers = 2 if interval in _INTRADAY_INTERVALS else 5
        requested_workers = default_workers if max_workers is None else int(max_workers)
        workers = max(1, min(requested_workers, cap_workers))
        requested_bars = self._resolve_requested_bar_count(
            days=days, bars=None, interval=interval
        )
        min_required_rows = max(1, min(int(CONFIG.data.min_history_days), int(requested_bars)))

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(fetch_one, c): c for c in codes}
            for future in as_completed(futures):
                code = futures[future]
                try:
                    code, df = future.result(timeout=120)
                    if (
                        not df.empty
                        and len(df) >= min_required_rows
                    ):
                        results[code] = df
                except Exception as exc:
                    log.warning("Failed to fetch %s: %s", code, exc)
                with lock:
                    completed += 1
                    if callback:
                        callback(code, completed, total)

        log.info("Parallel fetch: %d/%d successful", len(results), total)
        return results

    def get_all_stocks(self) -> pd.DataFrame:
        for source in self._get_active_sources():
            if source.name == "akshare":
                try:
                    df = source.get_all_stocks()
                    if not df.empty:
                        return df
                except Exception as exc:
                    log.warning("Failed to get stock list: %s", exc)
        return pd.DataFrame()

    def get_source_status(self) -> list[DataSourceStatus]:
        return [s.status for s in self._all_sources]

    def reset_sources(self) -> None:
        from core.network import invalidate_network_cache
        invalidate_network_cache()
        for source in self._all_sources:
            with source._lock:
                source.status.consecutive_errors = 0
                source.status.disabled_until = None
                source.status.available = True
        log.info("All data sources reset, network cache invalidated")


_fetcher: DataFetcher | None = None
_fetcher_lock = threading.Lock()


def get_fetcher() -> DataFetcher:
    """Double-checked locking singleton for DataFetcher."""
    global _fetcher
    if _fetcher is None:
        with _fetcher_lock:
            if _fetcher is None:
                _fetcher = DataFetcher()
    return _fetcher
