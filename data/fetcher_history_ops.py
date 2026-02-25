# data/fetcher_history_ops.py
from typing import Any

import pandas as pd

from data.fetcher_sources import BARS_PER_DAY
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
)

def _fetch_from_sources_instrument(
    self,
    inst: dict[str, Any],
    days: int,
    interval: str = "1d",
    include_localdb: bool = True,
    return_meta: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, dict[str, Any]]:
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
    except _RECOVERABLE_FETCH_EXCEPTIONS:
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

    errors: list[str] = []
    iv_norm = self._normalize_interval_token(interval)
    is_intraday = iv_norm not in {"1d", "1wk", "1mo"}
    collected: list[dict[str, Any]] = []

    for src_rank, source in enumerate(sources):
        with self._rate_limiter:
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

            except _RECOVERABLE_FETCH_EXCEPTIONS as exc:
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

    except _RECOVERABLE_FETCH_EXCEPTIONS as exc:
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
