from __future__ import annotations

import math
import time
from statistics import median
from typing import Any

from config.settings import CONFIG
from utils.logger import get_logger
from utils.recoverable import COMMON_RECOVERABLE_EXCEPTIONS

log = get_logger(__name__)

_APP_CHART_RECOVERABLE_EXCEPTIONS = COMMON_RECOVERABLE_EXCEPTIONS

def _load_chart_history_bars(
    self: Any,
    symbol: str,
    interval: str,
    lookback_bars: int,
    _recursion_depth: int = 0,
) -> list[dict[str, Any]]:
    """Load historical OHLC bars for chart rendering."""
    if not self.predictor:
        return []
    try:
        fetcher = getattr(self.predictor, "fetcher", None)
        if fetcher is None:
            return []
        requested_iv = self._normalize_interval_token(interval)
        norm_iv = requested_iv or "1m"
        # Intraday charts are sourced from canonical 1m and resampled in UI.
        # Daily/weekly/monthly charts should fetch native intervals directly
        # to avoid 1m lookback/API-cap truncation.
        source_iv = "1m" if norm_iv not in {"1d", "1wk", "1mo"} else norm_iv
        is_trained = self._is_trained_stock(symbol)
        if is_trained:
            target_floor = int(self._trained_stock_window_bars(norm_iv))
            if norm_iv in {"1d", "1wk", "1mo"}:
                lookback = max(target_floor, int(lookback_bars))
            else:
                # Strictly keep trained intraday charts on the latest 2-day window.
                lookback = int(target_floor)
            refresh_requested = bool(
                self._consume_history_refresh(symbol, norm_iv)
            )
            force_refresh = bool(refresh_requested)
            use_cache = not force_refresh
            update_db = bool(force_refresh)
            allow_online = bool(force_refresh)
            fallback_allow_online = bool(force_refresh)
        else:
            if norm_iv in {"1d", "1wk", "1mo"}:
                lookback = max(7, int(min(max(lookback_bars, 7), 120)))
            else:
                lookback = max(120, int(self._seven_day_lookback(norm_iv)))
            refresh_requested = bool(
                self._consume_history_refresh(symbol, norm_iv)
            )
            force_refresh = bool(refresh_requested)
            use_cache = not force_refresh
            update_db = bool(force_refresh)
            allow_online = bool(force_refresh)
            fallback_allow_online = bool(force_refresh)

        if source_iv == norm_iv:
            source_lookback = int(
                max(
                    int(lookback),
                    int(self._recommended_lookback(source_iv)),
                )
            )
        else:
            source_lookback = int(
                max(
                    self._recommended_lookback(source_iv),
                    self._bars_needed_from_base_interval(
                        norm_iv,
                        int(lookback),
                        base_interval=source_iv,
                    ),
                )
            )
        source_min_floor = int(self._recommended_lookback(source_iv))
        is_intraday = norm_iv not in ("1d", "1wk", "1mo")
        market_open = bool(CONFIG.is_market_open())
        now_bucket = self._bar_bucket_epoch(time.time(), source_iv)
        try:
            df = fetcher.get_history(
                symbol,
                interval=source_iv,
                bars=source_lookback,
                use_cache=bool(use_cache),
                update_db=bool(update_db),
                allow_online=bool(allow_online),
                refresh_intraday_after_close=bool(force_refresh),
            )
        except TypeError:
            df = fetcher.get_history(
                symbol,
                interval=source_iv,
                bars=source_lookback,
                use_cache=bool(use_cache),
                update_db=bool(update_db),
            )
        if source_iv == "1d":
            min_required = int(max(5, min(source_lookback, 90)))
        elif source_iv == "1wk":
            min_required = int(max(4, min(source_lookback, 52)))
        elif source_iv == "1mo":
            min_required = int(max(3, min(source_lookback, 24)))
        else:
            min_required = int(max(20, source_min_floor if is_trained else 20))
            if source_lookback >= 200:
                depth_ratio = 0.55 if is_trained else 0.40
                min_required = max(
                    min_required,
                    int(max(120, float(source_lookback) * float(depth_ratio))),
                )
        if df is None or df.empty or len(df) < min_required:
            try:
                df_online = fetcher.get_history(
                    symbol,
                    interval=source_iv,
                    bars=source_lookback,
                    # Bypass in-memory short windows when depth is too thin.
                    use_cache=False,
                    update_db=True,
                    allow_online=True,
                    refresh_intraday_after_close=bool(force_refresh),
                )
            except TypeError:
                df_online = fetcher.get_history(
                    symbol,
                    interval=source_iv,
                    bars=source_lookback,
                    use_cache=False,
                    update_db=True,
                )
            if df_online is not None and not df_online.empty:
                df = df_online
        if df is None or df.empty:
            # Fallback query path when primary history window is empty.
            try:
                df = fetcher.get_history(
                    symbol,
                    interval=source_iv,
                    bars=source_lookback,
                    use_cache=True,
                    update_db=False,
                    allow_online=bool(allow_online),
                    refresh_intraday_after_close=bool(force_refresh),
                )
            except TypeError:
                df = fetcher.get_history(
                    symbol,
                    interval=source_iv,
                    bars=source_lookback,
                    use_cache=True,
                    update_db=False,
                )
        out: list[dict[str, Any]] = []
        prev_close: float | None = None
        prev_epoch: float | None = None

        if df is not None and not df.empty:
            for idx, row in df.tail(source_lookback).iterrows():
                c = float(row.get("close", 0) or 0)
                if c <= 0:
                    continue
                ts_obj = row.get("datetime", idx)
                epoch = self._bar_bucket_epoch(ts_obj, source_iv)
                ref_close = prev_close
                if (
                    prev_epoch is not None
                    and self._is_intraday_day_boundary(prev_epoch, epoch, source_iv)
                ):
                    ref_close = None
                o_raw = row.get("open", None)
                try:
                    o = float(o_raw or 0)
                except _APP_CHART_RECOVERABLE_EXCEPTIONS:
                    o = 0.0
                if o <= 0:
                    if source_iv in {"1d", "1wk", "1mo"}:
                        o = float(c)
                    else:
                        o = float(ref_close if ref_close and ref_close > 0 else c)
                h = float(row.get("high", max(o, c)) or max(o, c))
                low = float(row.get("low", min(o, c)) or min(o, c))
                sanitized = self._sanitize_ohlc(
                    o,
                    h,
                    low,
                    c,
                    interval=source_iv,
                    ref_close=ref_close,
                )
                if sanitized is None:
                    continue
                o, h, low, c = sanitized
                try:
                    vol = float(row.get("volume", 0) or 0.0)
                except _APP_CHART_RECOVERABLE_EXCEPTIONS:
                    vol = 0.0
                if (not math.isfinite(vol)) or vol < 0:
                    vol = 0.0
                try:
                    amount = float(row.get("amount", 0) or 0.0)
                except _APP_CHART_RECOVERABLE_EXCEPTIONS:
                    amount = 0.0
                if not math.isfinite(amount):
                    amount = 0.0
                if amount <= 0 and vol > 0 and c > 0:
                    amount = float(c) * float(vol)
                out.append(
                    {
                        "open": o,
                        "high": h,
                        "low": low,
                        "close": c,
                        "volume": float(vol),
                        "amount": float(max(0.0, amount)),
                        "timestamp": self._epoch_to_iso(epoch),
                        "_ts_epoch": float(epoch),
                        "final": True,
                        "interval": source_iv,
                    }
                )
                prev_close = c
                prev_epoch = float(epoch)

        # Include session-persisted bars so refresh/restart keeps data continuity.
        if self._session_bar_cache is not None and not force_refresh and market_open:
            sdf = self._session_bar_cache.read_history(
                symbol, source_iv, bars=source_lookback, final_only=False
            )
            if sdf is not None and not sdf.empty:
                for idx, row in sdf.tail(source_lookback).iterrows():
                    c = float(row.get("close", 0) or 0)
                    if c <= 0:
                        continue
                    epoch = self._bar_bucket_epoch(idx, source_iv)
                    ref_close = prev_close
                    if (
                        prev_epoch is not None
                        and self._is_intraday_day_boundary(prev_epoch, epoch, source_iv)
                    ):
                        ref_close = None
                    o_raw = row.get("open", None)
                    try:
                        o = float(o_raw or 0)
                    except _APP_CHART_RECOVERABLE_EXCEPTIONS:
                        o = 0.0
                    if o <= 0:
                        if source_iv in {"1d", "1wk", "1mo"}:
                            o = float(c)
                        else:
                            o = float(ref_close if ref_close and ref_close > 0 else c)
                    h = float(row.get("high", max(o, c)) or max(o, c))
                    low = float(row.get("low", min(o, c)) or min(o, c))
                    sanitized = self._sanitize_ohlc(
                        o,
                        h,
                        low,
                        c,
                        interval=source_iv,
                        ref_close=ref_close,
                    )
                    if sanitized is None:
                        continue
                    o, h, low, c = sanitized
                    try:
                        vol = float(row.get("volume", 0) or 0.0)
                    except _APP_CHART_RECOVERABLE_EXCEPTIONS:
                        vol = 0.0
                    if (not math.isfinite(vol)) or vol < 0:
                        vol = 0.0
                    try:
                        amount = float(row.get("amount", 0) or 0.0)
                    except _APP_CHART_RECOVERABLE_EXCEPTIONS:
                        amount = 0.0
                    if not math.isfinite(amount):
                        amount = 0.0
                    if amount <= 0 and vol > 0 and c > 0:
                        amount = float(c) * float(vol)
                    is_final = bool(row.get("is_final", True))
                    if (
                        is_intraday
                        and not is_final
                        and (
                            (not market_open)
                            or int(epoch) != int(now_bucket)
                        )
                    ):
                        # Keep only the current bucket partial bar while market is open.
                        continue
                    out.append(
                        {
                            "open": o,
                            "high": h,
                            "low": low,
                            "close": c,
                            "volume": float(vol),
                            "amount": float(max(0.0, amount)),
                            "timestamp": self._epoch_to_iso(epoch),
                            "_ts_epoch": float(epoch),
                            "final": is_final,
                            "interval": source_iv,
                        }
                    )
                    prev_close = c
                    prev_epoch = float(epoch)

        out = self._filter_bars_to_market_session(out, source_iv)

        # Deduplicate by normalized epoch and keep latest.
        merged: dict[int, dict[str, Any]] = {}
        for b in out:
            epoch = self._bar_bucket_epoch(
                b.get("_ts_epoch", b.get("timestamp", "")),
                source_iv,
            )
            row = dict(b)
            row["_ts_epoch"] = float(epoch)
            row["timestamp"] = self._epoch_to_iso(epoch)
            key = int(epoch)
            existing = merged.get(key)
            if existing is None:
                merged[key] = row
                continue

            existing_final = bool(existing.get("final", True))
            row_final = bool(row.get("final", True))
            if existing_final and not row_final:
                continue
            if row_final and not existing_final:
                merged[key] = row
                continue

            # Same finality: prefer richer bar (volume) then later row.
            try:
                e_vol = float(existing.get("volume", 0) or 0)
            except _APP_CHART_RECOVERABLE_EXCEPTIONS:
                e_vol = 0.0
            try:
                r_vol = float(row.get("volume", 0) or 0)
            except _APP_CHART_RECOVERABLE_EXCEPTIONS:
                r_vol = 0.0
            # Only replace DB bar when session cache has strictly higher volume.
            if r_vol > e_vol:
                merged[key] = row
        out = list(merged.values())
        out.sort(
            key=lambda x: float(
                x.get("_ts_epoch", self._ts_to_epoch(x.get("timestamp", "")))
            )
        )
        # One more unified scrub pass to drop residual malformed bars.
        out = self._merge_bars([], out, source_iv)
        out = out[-source_lookback:]

        # Chart intervals are display-only; source stream remains 1m.
        if norm_iv != source_iv:
            out = self._resample_chart_bars(
                out,
                source_interval=source_iv,
                target_interval=norm_iv,
            )
        out = out[-lookback:]

        if out and is_intraday and not force_refresh:
            sample = out[-min(520, len(out)):]
            total_q = 0
            degenerate_q = 0
            epochs: list[float] = []
            for row in sample:
                try:
                    c_q = float(row.get("close", 0) or 0)
                    o_q = float(row.get("open", c_q) or c_q)
                    h_q = float(row.get("high", c_q) or c_q)
                    l_q = float(row.get("low", c_q) or c_q)
                except _APP_CHART_RECOVERABLE_EXCEPTIONS:
                    continue
                if c_q <= 0 or (not all(math.isfinite(v) for v in (o_q, h_q, l_q, c_q))):
                    continue
                ref_q = max(c_q, 1e-8)
                body_q = abs(o_q - c_q) / ref_q
                span_q = abs(h_q - l_q) / ref_q
                if body_q <= 0.00012 and span_q <= 0.00120:
                    degenerate_q += 1
                total_q += 1
                try:
                    ep_q = float(
                        self._bar_bucket_epoch(
                            row.get("_ts_epoch", row.get("timestamp")),
                            norm_iv,
                        )
                    )
                    if math.isfinite(ep_q):
                        epochs.append(ep_q)
                except _APP_CHART_RECOVERABLE_EXCEPTIONS as exc:
                    log.debug("Suppressed exception in app_chart_history_load_ops", exc_info=exc)

            deg_ratio = (
                float(degenerate_q) / float(max(1, total_q))
                if total_q > 0
                else 0.0
            )
            med_step = 0.0
            if len(epochs) >= 3:
                epochs = sorted(epochs)
                diffs = [
                    float(epochs[i] - epochs[i - 1])
                    for i in range(1, len(epochs))
                    if float(epochs[i] - epochs[i - 1]) > 0
                ]
                if diffs:
                    med_step = float(median(diffs))

            expected_step = float(max(1, self._interval_seconds(norm_iv)))
            bad_degenerate = total_q >= 180 and deg_ratio >= 0.50
            bad_cadence = med_step > (expected_step * 3.5)
            if bad_degenerate or bad_cadence:
                self._debug_console(
                    f"chart_history_refresh:{self._ui_norm(symbol)}:{norm_iv}",
                    (
                        f"forcing one-shot online history refresh for {self._ui_norm(symbol)} {norm_iv}: "
                        f"degenerate={deg_ratio:.1%} cadence={med_step:.0f}s expected={expected_step:.0f}s "
                        f"bars={len(out)}"
                    ),
                    min_gap_seconds=1.0,
                    level="warning",
                )
                self._queue_history_refresh(symbol, norm_iv)
                if _recursion_depth < 1:
                    return self._load_chart_history_bars(symbol, norm_iv, lookback, _recursion_depth=_recursion_depth + 1)
        return out
    except _APP_CHART_RECOVERABLE_EXCEPTIONS as e:
        log.debug(f"Historical chart load failed for {symbol}: {e}")
        return []
