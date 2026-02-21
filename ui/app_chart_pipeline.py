from __future__ import annotations

import math
import time
from statistics import median
from typing import Any

from PyQt6.QtWidgets import QTableWidgetItem
from config.settings import CONFIG
from ui.background_tasks import WorkerThread
from utils.logger import get_logger
from utils.recoverable import COMMON_RECOVERABLE_EXCEPTIONS

log = get_logger(__name__)

_APP_CHART_RECOVERABLE_EXCEPTIONS = COMMON_RECOVERABLE_EXCEPTIONS
def _on_price_updated(self: Any, code: str, price: float) -> None:
    """
    Handle price update from monitor.

    FIXED: No longer calls update_data() which was overwriting candles.
    Instead, updates the current bar's close price so the candle
    reflects the live price.
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
            if (
                (now_ui - prev_ts) < 0.12
                and abs(price - prev_px)
                <= max(0.001, abs(prev_px) * 0.00004)
            ):
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
        arr.sort(
            key=lambda x: float(
                x.get("_ts_epoch", self._ts_to_epoch(x.get("timestamp", "")))
            )
        )
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
                if prev_ref is None:
                    prev_ref = float(last.get("close", price) or price)
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
                        s = self._sanitize_ohlc(
                            float(last.get("open", price) or price),
                            max(float(last.get("high", bar_price) or bar_price), bar_price),
                            min(float(last.get("low", bar_price) or bar_price), bar_price),
                            bar_price,
                            interval=interval,
                            ref_close=float(prev_ref) if prev_ref and prev_ref > 0 else None,
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
                        day_boundary = self._is_intraday_day_boundary(
                            last_bucket,
                            now_bucket,
                            interval,
                        )
                        ref_close_new = (
                            float(last_close)
                            if (last_close > 0 and not day_boundary)
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
                log.debug("Suppressed exception in ui/app.py", exc_info=exc)

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
            day_boundary = self._is_intraday_day_boundary(
                last_bucket,
                bucket_epoch,
                interval,
            )
            ref_close = float(prev_close) if (prev_close > 0 and not day_boundary) else None
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
    self.workers["forecast_refresh"] = worker
    self._forecast_refresh_symbol = code

    def on_done(res: Any) -> None:
        try:
            if not res:
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

            stable_predicted = [
                float(v)
                for v in self._safe_list(predicted_prices)
                if float(v) > 0 and math.isfinite(float(v))
            ]
            if not stable_predicted:
                if (
                    self.current_prediction
                    and self.current_prediction.stock_code == code
                ):
                    stable_predicted = [
                        float(v)
                        for v in self._safe_list(
                            getattr(
                                self.current_prediction,
                                "predicted_prices",
                                [],
                            )
                        )
                        if float(v) > 0 and math.isfinite(float(v))
                    ]
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

            # Update current_prediction with new forecast
            if (
                self.current_prediction
                and self.current_prediction.stock_code == code
            ):
                self.current_prediction.predicted_prices = display_predicted
                low_band, high_band = self._build_chart_prediction_bands(
                    symbol=code,
                    predicted_prices=display_predicted,
                    anchor_price=display_current if display_current > 0 else None,
                )
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
            self.workers.pop("forecast_refresh", None)
            if self._forecast_refresh_symbol == code:
                self._forecast_refresh_symbol = ""

    worker.result.connect(on_done)
    def on_error(_e: Any) -> None:
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
    """
    Final chart-only scrub to enforce one interval and normalized buckets.
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
        repaired_shape = 0
        repaired_gap = 0
        processed_count = 0
        allow_shape_rebuild = True
        rebuild_disabled = False
        rebuild_streak = 0
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
                # Keep prev_close as ref for the first bar of a new day
                # so overnight gaps are still clamped by _sanitize_ohlc
                # (jump_cap of 8% for 1m already covers A-share 10% limit).
                # Only clear rolling stats for adaptive caps.
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
            processed_count += 1
            rebuilt_now = False
            if (
                allow_shape_rebuild
                and ref_close
                and float(ref_close) > 0
            ):
                ref_prev = max(float(ref_close), float(c), 1e-8)
                body_prev = abs(o - c) / ref_prev
                span_prev = abs(h - low) / ref_prev
                jump_prev = abs(c / float(ref_close) - 1.0)
                # Many fallback sources emit close-only intraday bars
                # (open ~= close). Rebuild open from previous close so
                # candles are readable without inventing large moves.
                if (
                    body_prev <= 0.00008
                    and span_prev <= 0.0018
                    and jump_prev <= 0.0025
                ):
                    o = float(ref_close)
                    top0 = max(o, c)
                    bot0 = min(o, c)
                    h = max(h, top0)
                    low = min(low, bot0)
                    repaired_shape += 1
                    rebuilt_now = True
                    rebuild_streak += 1
                    if (
                        allow_shape_rebuild
                        and (
                            rebuild_streak >= 8
                            or (
                                processed_count >= 60
                                and (
                                    float(repaired_shape)
                                    / float(max(1, processed_count))
                                ) > 0.22
                            )
                        )
                    ):
                        allow_shape_rebuild = False
                        rebuild_disabled = True
            if not rebuilt_now:
                rebuild_streak = 0

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
                        float(max(0.0035, med_body * 6.0)),
                    )
            if recent_span:
                med_span = float(median(recent_span[-48:]))
                if med_span > 0 and math.isfinite(med_span):
                    eff_span_cap = min(
                        eff_span_cap,
                        float(max(0.0050, med_span * 5.5)),
                    )
                    eff_wick_cap = min(
                        eff_wick_cap,
                        float(max(0.0035, med_span * 3.8)),
                    )

            if (
                ref_jump > ref_jump_cap
                or body > eff_body_cap
                or span > eff_span_cap
                or upper_wick > eff_wick_cap
                or lower_wick > eff_wick_cap
            ):
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
                    # First bar after lunch/day gaps: repair extreme boundary bars
                    # before deciding to drop them.
                    boundary_body_cap = float(max(0.004, min(eff_body_cap, 0.008)))
                    boundary_span_cap = float(max(0.006, min(eff_span_cap, 0.012)))
                    boundary_wick_cap = float(max(0.004, min(eff_wick_cap, 0.008)))
                    boundary_jump_cap = float(max(0.008, min(ref_jump_cap, 0.018)))
                    if (
                        span > boundary_span_cap
                        or body > boundary_body_cap
                        or ref_jump > boundary_jump_cap
                    ):
                        if ref_close and float(ref_close) > 0:
                            o = float(ref_close)
                            jump_now = abs(c / max(float(ref_close), 1e-8) - 1.0)
                            if jump_now > boundary_jump_cap:
                                sign = 1.0 if c >= float(ref_close) else -1.0
                                c = float(ref_close) * (
                                    1.0 + (sign * boundary_jump_cap)
                                )
                            top = max(o, c)
                            bot = min(o, c)
                            ref_local = max(ref, c, o)
                            wick_allow = float(ref_local) * float(boundary_wick_cap)
                            h = min(max(h, top), top + wick_allow)
                            low = max(min(low, bot), bot - wick_allow)
                            if h < low:
                                h, low = low, h
                            span = abs(h - low) / max(ref_local, 1e-8)
                            upper_wick = max(0.0, h - top) / max(ref_local, 1e-8)
                            lower_wick = max(0.0, bot - low) / max(ref_local, 1e-8)
                            body = abs(o - c) / max(ref_local, 1e-8)
                            ref_jump = abs(c / max(ref_local, 1e-8) - 1.0)
                            repaired_gap += 1
                        if (
                            body > (boundary_body_cap * 1.45)
                            or span > (boundary_span_cap * 1.45)
                            or upper_wick > (boundary_wick_cap * 1.60)
                            or lower_wick > (boundary_wick_cap * 1.60)
                            or ref_jump > (boundary_jump_cap * 1.80)
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
        if repaired_shape > 0 or repaired_gap > 0:
            self._debug_console(
                f"chart_shape_repair:{sym or 'active'}:{iv}",
                (
                    f"chart shape repair symbol={sym or '--'} iv={iv} "
                    f"repaired={repaired_shape} gap_repaired={repaired_gap} "
                    f"kept={len(filtered)}"
                ),
                min_gap_seconds=1.0,
                level="info",
            )
        if rebuild_disabled:
            self._debug_console(
                f"chart_shape_repair_disable:{sym or 'active'}:{iv}",
                (
                    f"disabled close-only candle rebuild for {sym or '--'} {iv}: "
                    f"repaired={repaired_shape} processed={processed_count}"
                ),
                min_gap_seconds=1.0,
                level="warning",
            )
        out = filtered[-keep:]

    if out and iv in ("1d", "1wk", "1mo"):
        if iv == "1d":
            daily_jump_cap = 0.18
            daily_body_cap = 0.12
            daily_span_cap = 0.20
            daily_wick_cap = 0.10
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
        repaired_daily = 0

        for row in out:
            try:
                c_raw = float(row.get("close", 0) or 0)
                o_raw = float(row.get("open", c_raw) or c_raw)
                h_raw = float(row.get("high", c_raw) or c_raw)
                l_raw = float(row.get("low", c_raw) or c_raw)
            except _APP_CHART_RECOVERABLE_EXCEPTIONS:
                dropped_daily += 1
                continue

            sanitized = self._sanitize_ohlc(
                o_raw,
                h_raw,
                l_raw,
                c_raw,
                interval=iv,
                ref_close=prev_close_daily,
            )
            if sanitized is None:
                dropped_daily += 1
                continue
            o, h, low, c = sanitized

            if recent_daily:
                ref = float(median(recent_daily[-24:]))
            elif prev_close_daily and float(prev_close_daily) > 0:
                ref = float(prev_close_daily)
            else:
                ref = float(c)
            if not math.isfinite(ref) or ref <= 0:
                ref = float(c)
            if not math.isfinite(ref) or ref <= 0:
                dropped_daily += 1
                continue

            if prev_close_daily and float(prev_close_daily) > 0:
                jump_prev = abs(c / float(prev_close_daily) - 1.0)
                if jump_prev > daily_jump_cap:
                    sign = 1.0 if c >= float(prev_close_daily) else -1.0
                    c = float(prev_close_daily) * (1.0 + (sign * daily_jump_cap))
                    o = float(prev_close_daily)
                    repaired_daily += 1

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
                max_span_px = float(ref) * float(daily_span_cap)
                max_body_px = float(ref) * float(daily_body_cap)
                body_px = float(max(0.0, top - bot))
                if body_px > max_body_px:
                    if c >= o:
                        o = c - max_body_px
                    else:
                        o = c + max_body_px
                    top = max(o, c)
                    bot = min(o, c)
                    body_px = float(max(0.0, top - bot))
                if body_px > max_span_px:
                    o = c
                    top = c
                    bot = c
                    body_px = 0.0
                wick_allow = max(0.0, max_span_px - body_px)
                h = min(h, top + (wick_allow * 0.60))
                low = max(low, bot - (wick_allow * 0.60))
                if h < low:
                    h, low = low, h
                span = abs(h - low) / max(ref, 1e-8)
                body = abs(o - c) / max(ref, 1e-8)
                upper_wick = max(0.0, h - max(o, c)) / max(ref, 1e-8)
                lower_wick = max(0.0, min(o, c) - low) / max(ref, 1e-8)
                repaired_daily += 1
                if (
                    body > (daily_body_cap * 1.35)
                    or span > (daily_span_cap * 1.35)
                    or upper_wick > (daily_wick_cap * 1.40)
                    or lower_wick > (daily_wick_cap * 1.40)
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

        if dropped_daily > 0 or repaired_daily > 0:
            self._debug_console(
                f"chart_daily_filter:{sym or 'active'}:{iv}",
                (
                    f"daily filter symbol={sym or '--'} iv={iv} "
                    f"repaired={repaired_daily} dropped={dropped_daily} "
                    f"kept={len(daily_filtered)} raw={len(out)}"
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



def _load_chart_history_bars(
    self: Any,
    symbol: str,
    interval: str,
    lookback_bars: int,
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
            lookback = max(target_floor, int(lookback_bars))
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
        if (
            (df is None or df.empty or len(df) < min_required)
            and bool(fallback_allow_online)
        ):
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
        if self._session_bar_cache is not None and not force_refresh:
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
            if r_vol >= e_vol:
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
                    log.debug("Suppressed exception in ui/app.py", exc_info=exc)

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
                return self._load_chart_history_bars(symbol, norm_iv, lookback)
        return out
    except _APP_CHART_RECOVERABLE_EXCEPTIONS as e:
        log.debug(f"Historical chart load failed for {symbol}: {e}")
        return []
