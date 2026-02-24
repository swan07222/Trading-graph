from __future__ import annotations

import math
import time
from statistics import median
from typing import Any

from PyQt6.QtWidgets import QTableWidgetItem

from config.settings import CONFIG
from ui.app_chart_history_load_ops import _load_chart_history_bars as _load_chart_history_bars_impl
from ui.background_tasks import WorkerThread
from utils.logger import get_logger
from utils.recoverable import COMMON_RECOVERABLE_EXCEPTIONS

log = get_logger(__name__)

_APP_CHART_RECOVERABLE_EXCEPTIONS = COMMON_RECOVERABLE_EXCEPTIONS
def _on_price_updated(self: Any, code: str, price: float) -> None:
    """Handle price update from monitor.

    FIXED: No longer calls update_data() which was overwriting candles.
    Instead, updates the current bar's close price so the candle
    reflects the live price.
    
    FIX: Added better throttling to reduce UI flicker in watchlist.
    """
    if not CONFIG.is_market_open():
        return

    code = self._ui_norm(code)
    try:
        price = float(price)
    except _APP_CHART_RECOVERABLE_EXCEPTIONS:
        return
    if not code or price <= 0:
        return

    row = self._watchlist_row_by_code.get(code)
    if row is None:
        # Lazy rebuild if row map was invalidated by table reset.
        for r in range(self.watchlist.rowCount()):
            item = self.watchlist.item(r, 0)
            if not item:
                continue
            mapped = self._ui_norm(item.text())
            if not mapped:
                continue
            self._watchlist_row_by_code[mapped] = int(r)
        row = self._watchlist_row_by_code.get(code)

    if row is not None:
        now_ui = time.monotonic()
        prev_ui = self._last_watchlist_price_ui.get(code)
        refresh_price = True
        if prev_ui is not None:
            prev_ts, prev_px = float(prev_ui[0]), float(prev_ui[1])
            elapsed_ms = (now_ui - prev_ts) * 1000
            
            # FIX: Improved throttling - 200ms minimum between updates
            # with 0.01% minimum change threshold to reduce flicker
            if elapsed_ms < 200.0:
                pct_change = abs(price - prev_px) / max(prev_px, 0.0001)
                if pct_change < 0.0001:  # 0.01% minimum change
                    refresh_price = False
            elif elapsed_ms < 500.0:
                # Medium throttle window - require 0.05% change
                pct_change = abs(price - prev_px) / max(prev_px, 0.0001)
                if pct_change < 0.0005:
                    refresh_price = False

        if refresh_price:
            text = f"CNY {price:.2f}"
            cell = self.watchlist.item(int(row), 1)
            if cell is None:
                self.watchlist.setItem(int(row), 1, QTableWidgetItem(text))
            elif cell.text() != text:
                cell.setText(text)
            self._last_watchlist_price_ui[code] = (now_ui, price)

    self._refresh_guess_rows_for_symbol(code, price)

    current_code = self._ui_norm(self.stock_input.text())
    ui_interval = self._normalize_interval_token(
        self.interval_combo.currentText()
    )
    inferred_interval = ui_interval

    # Update the last bar's close price for live candle display.
    arr = self._bars_by_symbol.get(code)
    if arr and code == current_code:
        same_interval = [
            b for b in arr
            if self._normalize_interval_token(
                b.get("interval", ui_interval), fallback=ui_interval
            ) == ui_interval
        ]
        if same_interval and len(same_interval) != len(arr):
            arr = same_interval
            self._bars_by_symbol[code] = arr
        elif not same_interval:
            # Rebuild on UI interval instead of mutating stale bars from a
            # different timeframe (can create giant synthetic candles).
            inferred_interval = ui_interval
            self._queue_history_refresh(code, ui_interval)
            self._debug_console(
                f"tick_iv_rebuild:{code}:{ui_interval}",
                (
                    f"rebuild bars on ui interval for {code}: "
                    f"existing interval mismatch -> using {ui_interval}"
                ),
                min_gap_seconds=1.0,
                level="info",
            )
    if arr and len(arr) > 1:
        try:
            arr.sort(
                key=lambda x: float(
                    x.get("_ts_epoch", self._ts_to_epoch(x.get("timestamp", "")))
                )
            )
        except _APP_CHART_RECOVERABLE_EXCEPTIONS as exc:
            log.debug(f"Bar sort failed for {code}: {exc}")
    if code == current_code:
        if arr and inferred_interval != ui_interval:
            interval = inferred_interval
        else:
            interval = ui_interval
    else:
        interval = self._normalize_interval_token(
            (arr[-1].get("interval") if arr else None),
            fallback=ui_interval,
        )
    interval_s = self._interval_seconds(interval)
    now_ts = time.time()
    feed_age = now_ts - float(self._last_bar_feed_ts.get(code, 0.0))
    has_recent_feed_bar = (
        bool(arr)
        and feed_age <= max(2.0, float(interval_s) * 1.2)
    )
    if has_recent_feed_bar:
        if arr:
            try:
                last = arr[-1]
                last_bucket = self._bar_bucket_epoch(
                    last.get("_ts_epoch", last.get("timestamp", now_ts)),
                    interval,
                )
                prev_ref: float | None = None
                if len(arr) >= 2:
                    try:
                        prev_bar = arr[-2]
                        prev_bucket = self._bar_bucket_epoch(
                            prev_bar.get("_ts_epoch", prev_bar.get("timestamp", last_bucket)),
                            interval,
                        )
                        if not self._is_intraday_day_boundary(
                            prev_bucket,
                            last_bucket,
                            interval,
                        ):
                            prev_ref = float(prev_bar.get("close", price) or price)
                    except _APP_CHART_RECOVERABLE_EXCEPTIONS:
                        prev_ref = None
                # FIX: Store previous close BEFORE any modifications to prevent race condition
                last_close_before_update = float(last.get("close", price) or price)
                if prev_ref is None:
                    prev_ref = last_close_before_update
                if last_close_before_update <= 0:
                    last_close_before_update = float(price)
                bar_price = float(price)
                if prev_ref and prev_ref > 0:
                    clamp_cap = float(self._synthetic_tick_jump_cap(interval))
                    raw_jump = abs(bar_price / float(prev_ref) - 1.0)
                    if raw_jump > clamp_cap:
                        sign = 1.0 if bar_price >= float(prev_ref) else -1.0
                        clamped = float(prev_ref) * (1.0 + (sign * clamp_cap))
                        self._debug_console(
                            f"tick_clamp_recent:{code}:{interval}",
                            (
                                f"clamped synthetic tick {code} {interval}: "
                                f"raw={bar_price:.4f} prev={float(prev_ref):.4f} "
                                f"jump={raw_jump:.2%} -> {clamped:.4f}"
                            ),
                            min_gap_seconds=1.0,
                            level="warning",
                        )
                        bar_price = float(clamped)
                if (not prev_ref) or (not self._is_outlier_tick(
                    float(prev_ref), bar_price, interval=interval
                )):
                    now_bucket = self._bar_bucket_epoch(now_ts, interval)
                    if int(last_bucket) == int(now_bucket):
                        # FIX: Use last_close_before_update for OHLC sanitization reference
                        s = self._sanitize_ohlc(
                            float(last.get("open", last_close_before_update) or last_close_before_update),
                            max(float(last.get("high", bar_price) or bar_price), bar_price),
                            min(float(last.get("low", bar_price) or bar_price), bar_price),
                            bar_price,
                            interval=interval,
                            ref_close=last_close_before_update if last_close_before_update > 0 else None,
                        )
                        if s is not None:
                            o, h, low, c = s
                            last["open"] = o
                            last["high"] = h
                            last["low"] = low
                            last["close"] = c
                            last["final"] = False
                            last["_ts_epoch"] = float(last_bucket)
                            last["timestamp"] = self._epoch_to_iso(last_bucket)
                    else:
                        # Feed bars can arrive slightly late around bucket
                        # boundaries. Roll to a synthetic new bucket so the
                        # live candle and guessed graph do not freeze.
                        last_close = float(last.get("close", bar_price) or bar_price)
                        iv_norm = self._normalize_interval_token(interval)
                        is_wide_interval = iv_norm in ("1d", "1wk", "1mo")
                        day_boundary = self._is_intraday_day_boundary(
                            last_bucket,
                            now_bucket,
                            interval,
                        )
                        ref_close_new = (
                            float(last_close)
                            if (
                                last_close > 0
                                and (not day_boundary)
                                and (not is_wide_interval)
                            )
                            else None
                        )

                        if not bool(last.get("final", False)):
                            last["final"] = True
                            finalized_bar = dict(last)
                            finalized_bar["interval"] = interval
                            finalized_bar["_ts_epoch"] = float(last_bucket)
                            finalized_bar["timestamp"] = self._epoch_to_iso(last_bucket)
                            self._persist_session_bar(
                                code,
                                interval,
                                finalized_bar,
                                channel="tick_final",
                                min_gap_seconds=0.0,
                            )

                        new_price = float(bar_price)
                        if ref_close_new and ref_close_new > 0:
                            clamp_cap_new = float(self._synthetic_tick_jump_cap(interval))
                            raw_jump_new = abs(new_price / float(ref_close_new) - 1.0)
                            if raw_jump_new > clamp_cap_new:
                                sign_new = (
                                    1.0
                                    if new_price >= float(ref_close_new)
                                    else -1.0
                                )
                                new_price = float(ref_close_new) * (
                                    1.0 + (sign_new * clamp_cap_new)
                                )

                        if (
                            (not ref_close_new)
                            or (
                                not self._is_outlier_tick(
                                    float(ref_close_new),
                                    new_price,
                                    interval=interval,
                                )
                            )
                        ):
                            bucket_open = float(
                                ref_close_new
                                if ref_close_new and ref_close_new > 0
                                else new_price
                            )
                            s_new = self._sanitize_ohlc(
                                bucket_open,
                                max(bucket_open, new_price),
                                min(bucket_open, new_price),
                                new_price,
                                interval=interval,
                                ref_close=ref_close_new,
                            )
                            if s_new is None:
                                s_new = (
                                    bucket_open,
                                    bucket_open,
                                    bucket_open,
                                    bucket_open,
                                )
                            o_new, h_new, low_new, c_new = s_new
                            arr.append(
                                {
                                    "open": o_new,
                                    "high": h_new,
                                    "low": low_new,
                                    "close": c_new,
                                    "timestamp": self._epoch_to_iso(now_bucket),
                                    "final": False,
                                    "interval": interval,
                                    "_ts_epoch": float(now_bucket),
                                }
                            )
                            arr.sort(
                                key=lambda x: float(
                                    x.get(
                                        "_ts_epoch",
                                        self._ts_to_epoch(x.get("timestamp", "")),
                                    )
                                )
                            )
                            keep = self._history_window_bars(interval)
                            if len(arr) > keep:
                                del arr[:-keep]
            except _APP_CHART_RECOVERABLE_EXCEPTIONS as exc:
                log.debug("Suppressed exception in app_chart_pipeline", exc_info=exc)

        if current_code == code and arr:
            try:
                arr = self._render_chart_state(
                    symbol=code,
                    interval=interval,
                    bars=arr,
                    context="tick_recent",
                    current_price=price,
                    update_latest_label=True,
                )
                # Sync back so persistence uses the canonically scrubbed list.
                if arr:
                    self._bars_by_symbol[code] = arr
            except _APP_CHART_RECOVERABLE_EXCEPTIONS as e:
                log.debug(f"Chart price refresh failed: {e}")

        if arr:
            self._persist_session_bar(
                code,
                interval,
                arr[-1],
                channel="tick",
                min_gap_seconds=0.9,
            )

    if not has_recent_feed_bar:
        bucket_s = float(max(interval_s, 1))
        bucket_epoch = float(int(now_ts // bucket_s) * int(bucket_s))
        bucket_iso = self._epoch_to_iso(bucket_epoch)

        # Check for outlier BEFORE creating a synthetic bar to avoid
        # polluting _bars_by_symbol with a phantom bad-price entry.
        if arr:
            _last_for_outlier = arr[-1]
            _ref_for_outlier = float(_last_for_outlier.get("close", price) or price)
            if (
                _ref_for_outlier > 0
                and self._is_outlier_tick(_ref_for_outlier, float(price), interval=interval)
            ):
                log.debug(
                    f"Skip outlier tick (pre-synthetic) for {code}: "
                    f"prev={_ref_for_outlier:.2f} new={float(price):.2f}"
                )
                return

        if not arr:
            arr = [{
                "open": price,
                "high": price,
                "low": price,
                "close": price,
                "timestamp": bucket_iso,
                "final": False,
                "interval": interval,
                "_ts_epoch": bucket_epoch,
            }]
            self._bars_by_symbol[code] = arr
        if arr and len(arr) > 0:
            last = arr[-1]
            prev_close = float(last.get("close", price) or price)
            last_epoch = self._ts_to_epoch(
                last.get("_ts_epoch", last.get("timestamp", bucket_iso))
            )
            last_bucket = float(int(last_epoch // bucket_s) * int(bucket_s))
            iv_norm = self._normalize_interval_token(interval)
            is_wide_interval = iv_norm in ("1d", "1wk", "1mo")
            day_boundary = self._is_intraday_day_boundary(
                last_bucket,
                bucket_epoch,
                interval,
            )
            ref_close = (
                float(prev_close)
                if (
                    prev_close > 0
                    and (not day_boundary)
                    and (not is_wide_interval)
                )
                else None
            )
            price_for_bar = float(price)
            if ref_close and ref_close > 0:
                clamp_cap = float(self._synthetic_tick_jump_cap(interval))
                raw_jump = abs(price_for_bar / float(ref_close) - 1.0)
                if raw_jump > clamp_cap:
                    sign = 1.0 if price_for_bar >= float(ref_close) else -1.0
                    clamped = float(ref_close) * (1.0 + (sign * clamp_cap))
                    self._debug_console(
                        f"tick_clamp_bucket:{code}:{interval}",
                        (
                            f"clamped synthetic bucket tick {code} {interval}: "
                            f"raw={price_for_bar:.4f} prev={float(ref_close):.4f} "
                            f"jump={raw_jump:.2%} -> {clamped:.4f}"
                        ),
                        min_gap_seconds=1.0,
                        level="warning",
                    )
                    price_for_bar = float(clamped)
            if (
                ref_close
                and ref_close > 0
                and self._is_outlier_tick(ref_close, price_for_bar, interval=interval)
            ):
                log.debug(
                    f"Skip outlier tick for {code}: prev={float(ref_close):.2f} new={price_for_bar:.2f}"
                )
                return

            if int(last_bucket) != int(bucket_epoch):
                if not bool(last.get("final", False)):
                    last["final"] = True
                finalized_bar = dict(last)
                finalized_bar["interval"] = interval
                finalized_bar["_ts_epoch"] = float(last_bucket)
                finalized_bar["timestamp"] = self._epoch_to_iso(last_bucket)
                self._persist_session_bar(
                    code,
                    interval,
                    finalized_bar,
                    channel="tick_final",
                    min_gap_seconds=0.0,
                )
                bucket_open = float(ref_close if ref_close and ref_close > 0 else price)
                s_new = self._sanitize_ohlc(
                    bucket_open,
                    max(bucket_open, price_for_bar),
                    min(bucket_open, price_for_bar),
                    price_for_bar,
                    interval=interval,
                    ref_close=ref_close,
                )
                if s_new is None:
                    # Keep continuity when tick is still unusable after clamp.
                    s_new = (bucket_open, bucket_open, bucket_open, bucket_open)
                o_new, h_new, l_new, c_new = s_new
                last = {
                    "open": o_new,
                    "high": h_new,
                    "low": l_new,
                    "close": c_new,
                    "timestamp": bucket_iso,
                    "final": False,
                    "interval": interval,
                    "_ts_epoch": bucket_epoch,
                }
                arr.append(last)
                keep = self._history_window_bars(interval)
                if len(arr) > keep:
                    del arr[:-keep]
            else:
                if float(last.get("open", 0) or 0) <= 0:
                    last["open"] = price_for_bar
                s = self._sanitize_ohlc(
                    float(last.get("open", price_for_bar) or price_for_bar),
                    max(float(last.get("high", price_for_bar) or price_for_bar), price_for_bar),
                    min(float(last.get("low", price_for_bar) or price_for_bar), price_for_bar),
                    price_for_bar,
                    interval=interval,
                    ref_close=ref_close,
                )
                if s is None:
                    return
                o, h, low, c = s
                last["open"] = o
                last["close"] = c
                last["high"] = h
                last["low"] = low
                last["final"] = False
                last["timestamp"] = bucket_iso
                last["_ts_epoch"] = bucket_epoch

            arr.sort(
                key=lambda x: float(
                    x.get("_ts_epoch", self._ts_to_epoch(x.get("timestamp", "")))
                )
            )
            keep = self._history_window_bars(interval)
            if len(arr) > keep:
                del arr[:-keep]
            last = arr[-1]

            if current_code == code:
                try:
                    arr = self._render_chart_state(
                        symbol=code,
                        interval=interval,
                        bars=arr,
                        context="tick_bucket",
                        current_price=price,
                        update_latest_label=True,
                    )
                    # Sync back so persistence uses the canonically scrubbed list.
                    if arr:
                        self._bars_by_symbol[code] = arr
                except _APP_CHART_RECOVERABLE_EXCEPTIONS as e:
                    log.debug(f"Chart price update failed: {e}")

            self._persist_session_bar(
                code,
                interval,
                last,
                channel="tick",
                min_gap_seconds=0.9,
            )

    # Only refresh guessed graph for the currently selected symbol.
    if current_code != code:
        return

    # =====================================================================
    # THROTTLED FORECAST REFRESH (keep existing logic but simplified)
    # =====================================================================

    if not self.predictor:
        return

    models_ready = False
    try:
        ready_fn = getattr(self.predictor, "_models_ready_for_runtime", None)
        if callable(ready_fn):
            models_ready = bool(ready_fn())
        else:
            models_ready = bool(
                getattr(self.predictor, "ensemble", None) is not None
                or getattr(self.predictor, "forecaster", None) is not None
            )
    except _APP_CHART_RECOVERABLE_EXCEPTIONS:
        models_ready = bool(
            getattr(self.predictor, "ensemble", None) is not None
            or getattr(self.predictor, "forecaster", None) is not None
        )

    if not models_ready:
        stale_cleared = False
        try:
            if (
                self.current_prediction
                and self.current_prediction.stock_code == code
            ):
                had_pred = bool(
                    list(getattr(self.current_prediction, "predicted_prices", []) or [])
                    or list(getattr(self.current_prediction, "predicted_prices_low", []) or [])
                    or list(getattr(self.current_prediction, "predicted_prices_high", []) or [])
                )
                self.current_prediction.predicted_prices = []
                self.current_prediction.predicted_prices_low = []
                self.current_prediction.predicted_prices_high = []
                stale_cleared = bool(had_pred)
        except _APP_CHART_RECOVERABLE_EXCEPTIONS:
            stale_cleared = False

        if stale_cleared:
            arr = self._bars_by_symbol.get(code) or []
            if arr:
                try:
                    self._render_chart_state(
                        symbol=code,
                        interval=self._normalize_interval_token(
                            self.interval_combo.currentText()
                        ),
                        bars=arr,
                        context="forecast_model_unavailable",
                        current_price=price if price > 0 else None,
                        predicted_prices=[],
                        source_interval=self._normalize_interval_token(
                            self.interval_combo.currentText()
                        ),
                        target_steps=int(self.forecast_spin.value()),
                        predicted_prepared=True,
                    )
                except _APP_CHART_RECOVERABLE_EXCEPTIONS as exc:
                    log.debug("Suppressed exception in app_chart_pipeline", exc_info=exc)

        self._debug_console(
            f"forecast_model_unavailable:{code}",
            f"skipped guessed-curve refresh for {code}: model artifacts unavailable",
            min_gap_seconds=3.0,
            level="warning",
        )
        return

    ui_interval = self._normalize_interval_token(
        self.interval_combo.currentText()
    )
    ui_horizon = int(self.forecast_spin.value())
    exact_artifacts = self._has_exact_model_artifacts(ui_interval, ui_horizon)
    refresh_gap = 1.0 if exact_artifacts else 2.2
    now = time.time()
    if (now - self._last_forecast_refresh_ts) < float(refresh_gap):
        return
    self._last_forecast_refresh_ts = now

    interval = ui_interval
    horizon = int(ui_horizon)
    lookback = max(
        120,
        int(self.lookback_spin.value()),
        int(self._recommended_lookback(interval)),
    )
    use_realtime = bool(CONFIG.is_market_open())
    infer_interval = "1m"
    infer_horizon = int(horizon)
    infer_lookback = int(
        max(
            self._recommended_lookback("1m"),
            self._bars_needed_from_base_interval(
                interval,
                int(lookback),
                base_interval="1m",
            ),
        )
    )
    history_allow_online = True
    if not self._has_exact_model_artifacts(infer_interval, infer_horizon):
        self._debug_console(
            f"forecast_model_fallback:{code}:{interval}:{horizon}",
            (
                f"forecast inference locked to 1m for {code}: "
                f"ui={interval}/{horizon} infer={infer_interval}/{infer_horizon} "
                f"lookback={infer_lookback} online=1"
            ),
            min_gap_seconds=2.0,
            level="info",
        )

    def do_forecast() -> Any:
        if hasattr(self.predictor, "get_realtime_forecast_curve"):
            return self.predictor.get_realtime_forecast_curve(
                stock_code=code,
                interval=infer_interval,
                horizon_steps=infer_horizon,
                lookback_bars=infer_lookback,
                use_realtime_price=use_realtime,
                history_allow_online=history_allow_online,
            )
        return None

    w_old = self.workers.get("forecast_refresh")
    if w_old and w_old.isRunning():
        if (
            self._forecast_refresh_symbol
            and self._forecast_refresh_symbol != code
        ):
            w_old.cancel()
        else:
            return

    worker = WorkerThread(do_forecast, timeout_seconds=30)
    self._track_worker(worker)
    worker._forecast_token = f"{code}:{time.monotonic():.6f}"
    self.workers["forecast_refresh"] = worker
    self._forecast_refresh_symbol = code

    def on_done(res: Any) -> None:
        try:
            if self.workers.get("forecast_refresh") is not worker:
                # Stale worker callback after a newer refresh started.
                return
            if not res:
                try:
                    if (
                        self.current_prediction
                        and self.current_prediction.stock_code == code
                    ):
                        self.current_prediction.predicted_prices = []
                        self.current_prediction.predicted_prices_low = []
                        self.current_prediction.predicted_prices_high = []
                except _APP_CHART_RECOVERABLE_EXCEPTIONS:
                    pass
                self._debug_console(
                    f"forecast_empty:{code}:{interval}",
                    f"forecast worker returned empty for {code} {interval}",
                    min_gap_seconds=1.0,
                )
                return
            actual_prices, predicted_prices = res
            selected = self._ui_norm(self.stock_input.text())
            if selected != code:
                return
            _ = actual_prices  # chart bars are maintained by feed/history path.

            stable_predicted: list[float] = []
            for v in self._safe_list(predicted_prices):
                try:
                    fv = float(v)
                except _APP_CHART_RECOVERABLE_EXCEPTIONS:
                    continue
                if fv > 0 and math.isfinite(fv):
                    stable_predicted.append(float(fv))
            predicted_prices = stable_predicted

            try:
                pvals = [
                    float(v) for v in (predicted_prices or [])
                    if float(v) > 0
                ]
            except _APP_CHART_RECOVERABLE_EXCEPTIONS:
                pvals = []
            if pvals:
                diffs = []
                for i in range(1, len(pvals)):
                    prev = float(pvals[i - 1])
                    cur = float(pvals[i])
                    if prev > 0:
                        diffs.append(abs(cur / prev - 1.0))
                max_step = max(diffs) if diffs else 0.0
                flip_ratio = 0.0
                if len(diffs) >= 3:
                    try:
                        s = []
                        for i in range(1, len(pvals)):
                            s.append(1 if pvals[i] >= pvals[i - 1] else -1)
                        flips = 0
                        for i in range(1, len(s)):
                            if s[i] != s[i - 1]:
                                flips += 1
                        flip_ratio = float(flips) / float(max(1, len(s) - 1))
                    except _APP_CHART_RECOVERABLE_EXCEPTIONS:
                        flip_ratio = 0.0

                if max_step > 0.02 or flip_ratio > 0.80:
                    self._debug_console(
                        f"forecast_shape:{code}:{infer_interval}",
                        (
                            f"forecast anomaly {code} {infer_interval}: len={len(pvals)} "
                            f"max_step={max_step:.2%} flip_ratio={flip_ratio:.2f} "
                            f"first={pvals[0]:.4f} last={pvals[-1]:.4f}"
                        ),
                        min_gap_seconds=1.0,
                    )

            display_current = 0.0
            try:
                if (
                    self.current_prediction
                    and self.current_prediction.stock_code == code
                ):
                    display_current = float(
                        getattr(self.current_prediction, "current_price", 0.0) or 0.0
                    )
            except _APP_CHART_RECOVERABLE_EXCEPTIONS:
                display_current = 0.0
            if display_current <= 0:
                try:
                    arr_tmp = self._bars_by_symbol.get(code) or []
                    if arr_tmp:
                        display_current = float(arr_tmp[-1].get("close", 0.0) or 0.0)
                except _APP_CHART_RECOVERABLE_EXCEPTIONS:
                    display_current = 0.0
            display_predicted = self._prepare_chart_predicted_prices(
                symbol=code,
                chart_interval=interval,
                predicted_prices=predicted_prices,
                source_interval=infer_interval,
                current_price=display_current if display_current > 0 else None,
                target_steps=int(self.forecast_spin.value()),
            )

            # Keep prediction state aligned with latest forecast payload.
            if (
                self.current_prediction
                and self.current_prediction.stock_code == code
            ):
                self.current_prediction.predicted_prices = list(display_predicted or [])
                if display_predicted:
                    low_band, high_band = self._build_chart_prediction_bands(
                        symbol=code,
                        predicted_prices=display_predicted,
                        anchor_price=display_current if display_current > 0 else None,
                        chart_interval=interval,
                    )
                else:
                    low_band, high_band = [], []
                self.current_prediction.predicted_prices_low = low_band
                self.current_prediction.predicted_prices_high = high_band

            arr = self._bars_by_symbol.get(code)
            if arr:
                iv = self._normalize_interval_token(
                    self.interval_combo.currentText()
                )
                anchor_px = 0.0
                try:
                    if (
                        self.current_prediction
                        and self.current_prediction.stock_code == code
                    ):
                        anchor_px = float(
                            getattr(
                                self.current_prediction,
                                "current_price",
                                0.0,
                            ) or 0.0
                        )
                except _APP_CHART_RECOVERABLE_EXCEPTIONS:
                    anchor_px = 0.0
                arr = self._render_chart_state(
                    symbol=code,
                    interval=iv,
                    bars=arr,
                    context="forecast_refresh",
                    current_price=anchor_px if anchor_px > 0 else None,
                    predicted_prices=display_predicted,
                    source_interval=iv,
                    target_steps=int(self.forecast_spin.value()),
                    predicted_prepared=True,
                )
        finally:
            if self.workers.get("forecast_refresh") is worker:
                self.workers.pop("forecast_refresh", None)
                if self._forecast_refresh_symbol == code:
                    self._forecast_refresh_symbol = ""

    worker.result.connect(on_done)
    def on_error(_e: Any) -> None:
        if self.workers.get("forecast_refresh") is worker:
            self.workers.pop("forecast_refresh", None)
            if self._forecast_refresh_symbol == code:
                self._forecast_refresh_symbol = ""
    worker.error.connect(on_error)
    worker.start()



def _prepare_chart_bars_for_interval(
    self: Any,
    bars: list[dict[str, Any]] | None,
    interval: str,
    *,
    symbol: str = "",
) -> list[dict[str, Any]]:
    """Final chart-only scrub to enforce one interval and normalized buckets.
    Prevents malformed/mixed bars from rendering giant candle bodies.
    """
    iv = self._normalize_interval_token(interval)
    sym = self._ui_norm(symbol)
    source_rows = list(bars or [])
    raw_count = len(source_rows)
    mixed_count = 0
    aligned_rows: list[dict[str, Any]] = []
    for row in source_rows:
        if not isinstance(row, dict):
            continue
        row_iv = self._normalize_interval_token(
            row.get("interval", iv),
            fallback=iv,
        )
        if row_iv != iv:
            mixed_count += 1
            continue
        aligned_rows.append(row)
    if mixed_count > 0:
        self._debug_console(
            f"chart_mixed_iv:{sym or 'active'}:{iv}",
            (
                f"chart scrub dropped mixed interval rows: "
                f"symbol={sym or '--'} target_iv={iv} "
                f"dropped={mixed_count}/{raw_count}"
            ),
            min_gap_seconds=1.0,
        )

    merged = self._merge_bars([], aligned_rows, iv)
    merged = self._filter_bars_to_market_session(merged, iv)
    out: list[dict[str, Any]] = []
    for row in merged:
        epoch = self._bar_bucket_epoch(
            row.get("_ts_epoch", row.get("timestamp")),
            iv,
        )
        item = dict(row)
        item["_ts_epoch"] = float(epoch)
        item["timestamp"] = self._epoch_to_iso(epoch)
        item["interval"] = iv
        out.append(item)
    keep = self._history_window_bars(iv)
    out = out[-keep:]

    # Final intraday pass: drop malformed rows that can still slip through
    # bootstrap sanitation (for example open=0 vendor rows around interval switches).
    if out and iv not in ("1d", "1wk", "1mo"):
        jump_cap, range_cap = self._bar_safety_caps(iv)
        # Tighter intraday caps to suppress giant block candles.
        body_cap = float(max(0.004, min(0.014, (range_cap * 0.92))))
        span_cap = float(max(0.006, min(0.022, (range_cap * 1.10))))
        wick_cap = float(max(0.004, min(0.013, (range_cap * 0.82))))
        ref_jump_cap = float(max(0.012, min(0.035, jump_cap * 0.45)))
        iv_s = float(max(1, self._interval_seconds(iv)))

        filtered: list[dict[str, Any]] = []
        recent_closes: list[float] = []
        recent_body: list[float] = []
        recent_span: list[float] = []
        dropped_shape = 0
        dropped_extreme_body = 0
        prev_close: float | None = None
        prev_epoch: float | None = None

        for row in out:
            try:
                c_raw = float(row.get("close", 0) or 0)
                o_raw = float(row.get("open", c_raw) or c_raw)
                h_raw = float(row.get("high", c_raw) or c_raw)
                l_raw = float(row.get("low", c_raw) or c_raw)
            except _APP_CHART_RECOVERABLE_EXCEPTIONS:
                dropped_shape += 1
                continue

            row_epoch = float(
                self._bar_bucket_epoch(
                    row.get("_ts_epoch", row.get("timestamp")),
                    iv,
                )
            )
            day_boundary = bool(
                prev_epoch is not None
                and self._is_intraday_day_boundary(prev_epoch, row_epoch, iv)
            )
            ref_close = prev_close
            if day_boundary:
                # Do not force previous-day close as intraday reference.
                # Corporate actions/ex-rights can reset price levels at open;
                # carrying old ref_close over day boundary causes fake spikes/drops.
                ref_close = None
                recent_closes.clear()
                recent_body.clear()
                recent_span.clear()

            sanitized = self._sanitize_ohlc(
                o_raw,
                h_raw,
                l_raw,
                c_raw,
                interval=iv,
                ref_close=ref_close,
            )
            if sanitized is None:
                dropped_shape += 1
                continue
            o, h, low, c = sanitized

            ref_values = [
                float(v) for v in recent_closes[-32:]
                if float(v) > 0 and math.isfinite(float(v))
            ]
            if ref_values:
                ref = float(median(ref_values))
            elif ref_close and float(ref_close) > 0:
                ref = float(ref_close)
            else:
                ref = float(c)

            if not math.isfinite(ref) or ref <= 0:
                ref = float(c)
            if not math.isfinite(ref) or ref <= 0:
                dropped_shape += 1
                continue

            body = abs(o - c) / ref
            span = abs(h - low) / ref
            top = max(o, c)
            bot = min(o, c)
            upper_wick = max(0.0, h - top) / ref
            lower_wick = max(0.0, bot - low) / ref
            ref_jump = abs(c / ref - 1.0)
            eff_body_cap = float(body_cap)
            eff_span_cap = float(span_cap)
            eff_wick_cap = float(wick_cap)
            if recent_body:
                med_body = float(median(recent_body[-48:]))
                if med_body > 0 and math.isfinite(med_body):
                    eff_body_cap = min(
                        eff_body_cap,
                        float(max(0.006, med_body * 8.0)),
                    )
            if recent_span:
                med_span = float(median(recent_span[-48:]))
                if med_span > 0 and math.isfinite(med_span):
                    eff_span_cap = min(
                        eff_span_cap,
                        float(max(0.009, med_span * 7.0)),
                    )
                    eff_wick_cap = min(
                        eff_wick_cap,
                        float(max(0.006, med_span * 5.0)),
                    )

            shape_outlier = bool(
                ref_jump > ref_jump_cap
                or body > eff_body_cap
                or span > eff_span_cap
                or upper_wick > eff_wick_cap
                or lower_wick > eff_wick_cap
            )
            if shape_outlier:
                if body > eff_body_cap:
                    dropped_extreme_body += 1
                dropped_shape += 1
                continue

            if (
                ref_close
                and float(ref_close) > 0
                and self._is_outlier_tick(ref_close, c, interval=iv)
            ):
                dropped_shape += 1
                continue

            if prev_epoch is not None:
                gap = max(0.0, row_epoch - float(prev_epoch))
                if (not day_boundary) and gap > (iv_s * 3.0):
                    boundary_body_cap = float(max(0.004, min(eff_body_cap, 0.008)))
                    boundary_span_cap = float(max(0.006, min(eff_span_cap, 0.012)))
                    boundary_wick_cap = float(max(0.004, min(eff_wick_cap, 0.008)))
                    boundary_jump_cap = float(max(0.008, min(ref_jump_cap, 0.018)))
                    if (
                        span > boundary_span_cap
                        or body > boundary_body_cap
                        or upper_wick > boundary_wick_cap
                        or lower_wick > boundary_wick_cap
                        or ref_jump > boundary_jump_cap
                    ):
                        dropped_shape += 1
                        continue

            row_out = dict(row)
            row_out["open"] = o
            row_out["high"] = h
            row_out["low"] = low
            row_out["close"] = c
            filtered.append(row_out)
            recent_closes.append(float(c))
            recent_body.append(float(body))
            recent_span.append(float(span))
            prev_close = float(c)
            prev_epoch = float(row_epoch)

        if dropped_shape > 0:
            self._debug_console(
                f"chart_shape_drop:{sym or 'active'}:{iv}",
                (
                    f"chart shape filter dropped {dropped_shape} bars: "
                    f"symbol={sym or '--'} iv={iv} kept={len(filtered)} raw={len(out)}"
                ),
                min_gap_seconds=1.0,
            )
        if dropped_extreme_body > 0:
            self._debug_console(
                f"chart_shape_body_drop:{sym or 'active'}:{iv}",
                (
                    f"chart body outlier drop symbol={sym or '--'} iv={iv} "
                    f"count={dropped_extreme_body} caps(body={body_cap:.2%},span={span_cap:.2%},wick={wick_cap:.2%})"
                ),
                min_gap_seconds=1.0,
                level="warning",
            )
        out = filtered[-keep:]

    if out and iv in ("1d", "1wk", "1mo"):
        if iv == "1d":
            daily_jump_cap = 0.22
            daily_body_cap = 0.22
            daily_span_cap = 0.28
            daily_wick_cap = 0.14
        elif iv == "1wk":
            daily_jump_cap = 0.26
            daily_body_cap = 0.20
            daily_span_cap = 0.34
            daily_wick_cap = 0.18
        else:
            daily_jump_cap = 0.35
            daily_body_cap = 0.28
            daily_span_cap = 0.45
            daily_wick_cap = 0.24

        daily_filtered: list[dict[str, Any]] = []
        recent_daily: list[float] = []
        prev_close_daily: float | None = None
        dropped_daily = 0

        for row in out:
            try:
                c_raw = float(row.get("close", 0) or 0)
                o_raw = float(row.get("open", c_raw) or c_raw)
                h_raw = float(row.get("high", c_raw) or c_raw)
                l_raw = float(row.get("low", c_raw) or c_raw)
            except _APP_CHART_RECOVERABLE_EXCEPTIONS:
                dropped_daily += 1
                continue

            ref_close = (
                float(prev_close_daily)
                if (prev_close_daily and float(prev_close_daily) > 0)
                else None
            )
            sanitized = self._sanitize_ohlc(
                o_raw,
                h_raw,
                l_raw,
                c_raw,
                interval=iv,
                ref_close=ref_close,
            )
            if sanitized is None:
                dropped_daily += 1
                continue
            o, h, low, c = sanitized

            if recent_daily:
                ref = float(median(recent_daily[-24:]))
            elif ref_close and float(ref_close) > 0:
                ref = float(ref_close)
            else:
                ref = float(c)
            if not math.isfinite(ref) or ref <= 0:
                ref = float(c)
            if not math.isfinite(ref) or ref <= 0:
                dropped_daily += 1
                continue

            if ref_close and float(ref_close) > 0:
                jump_prev = abs(c / float(ref_close) - 1.0)
                if jump_prev > daily_jump_cap:
                    dropped_daily += 1
                    continue

            top = max(o, c)
            bot = min(o, c)
            h = max(h, top)
            low = min(low, bot)
            body = abs(o - c) / max(ref, 1e-8)
            span = abs(h - low) / max(ref, 1e-8)
            upper_wick = max(0.0, h - top) / max(ref, 1e-8)
            lower_wick = max(0.0, bot - low) / max(ref, 1e-8)
            if (
                body > daily_body_cap
                or span > daily_span_cap
                or upper_wick > daily_wick_cap
                or lower_wick > daily_wick_cap
            ):
                dropped_daily += 1
                continue

            row_out = dict(row)
            row_out["open"] = float(o)
            row_out["high"] = float(h)
            row_out["low"] = float(low)
            row_out["close"] = float(c)
            daily_filtered.append(row_out)
            recent_daily.append(float(c))
            prev_close_daily = float(c)

        if dropped_daily > 0:
            self._debug_console(
                f"chart_daily_filter:{sym or 'active'}:{iv}",
                (
                    f"daily filter symbol={sym or '--'} iv={iv} "
                    f"dropped={dropped_daily} kept={len(daily_filtered)} raw={len(out)}"
                ),
                min_gap_seconds=1.0,
                level="info",
            )
        out = daily_filtered[-keep:]

    if out:
        max_body = 0.0
        max_range = 0.0
        for row in out:
            try:
                c = float(row.get("close", 0) or 0)
                if c <= 0:
                    continue
                o = float(row.get("open", c) or c)
                h = float(row.get("high", c) or c)
                low = float(row.get("low", c) or c)
                body = abs(o - c) / c
                span = abs(h - low) / c
                if body > max_body:
                    max_body = body
                if span > max_range:
                    max_range = span
            except _APP_CHART_RECOVERABLE_EXCEPTIONS:
                continue
        if iv not in ("1d", "1wk", "1mo") and (
            max_body > 0.08 or max_range > 0.12
        ):
            self._debug_console(
                f"chart_shape_anomaly:{sym or 'active'}:{iv}",
                (
                    f"chart shape anomaly symbol={sym or '--'} iv={iv} "
                    f"bars={len(out)} max_body={max_body:.2%} max_range={max_range:.2%}"
                ),
                min_gap_seconds=1.0,
            )

    drop_count = max(0, raw_count - len(out))
    if raw_count > 0 and drop_count >= max(5, int(raw_count * 0.20)):
        self._debug_console(
            f"chart_drop_ratio:{sym or 'active'}:{iv}",
            (
                f"chart scrub high drop ratio symbol={sym or '--'} "
                f"iv={iv} kept={len(out)} raw={raw_count}"
            ),
            min_gap_seconds=1.0,
        )

    return out



def _load_chart_history_bars(*args: Any, **kwargs: Any) -> Any:
    return _load_chart_history_bars_impl(*args, **kwargs)
