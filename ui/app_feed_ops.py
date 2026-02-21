from __future__ import annotations

import math
import time
from typing import Any

from config.settings import CONFIG
from utils.logger import get_logger
from utils.recoverable import COMMON_RECOVERABLE_EXCEPTIONS

log = get_logger(__name__)
_UI_RECOVERABLE_EXCEPTIONS = COMMON_RECOVERABLE_EXCEPTIONS

def _ensure_feed_subscription(self, code: str) -> None:
    """Subscribe symbol to realtime feed using 1m source bars."""
    if not CONFIG.is_market_open():
        return
    try:
        from data.feeds import get_feed_manager
        fm = get_feed_manager(auto_init=True, async_init=True)

        # Keep data acquisition fixed at 1m. UI interval only controls
        # display/aggregation, not upstream fetch cadence.
        fm.set_bar_interval_seconds(60)
        fm.subscribe(code)

        if not getattr(self, "_bar_callback_attached", False):
            self._bar_callback_attached = True
            fm.add_bar_callback(self._on_bar_from_feed)
        if not getattr(self, "_tick_callback_attached", False):
            self._tick_callback_attached = True
            fm.add_tick_callback(self._on_tick_from_feed)

    except _UI_RECOVERABLE_EXCEPTIONS as e:
        log.debug(f"Feed subscription failed: {e}")

def _on_bar_from_feed(self, symbol: str, bar: dict[str, Any]) -> None:
    """
    Called from feed thread (NOT UI thread).
    Emit signal to update UI safely.
    """
    if not CONFIG.is_market_open():
        return
    try:
        payload = dict(bar or {})
        if not payload.get("interval"):
            iv = self._interval_token_from_seconds(
                payload.get("interval_seconds")
            )
            if iv:
                payload["interval"] = iv
        self.bar_received.emit(str(symbol), payload)
    except _UI_RECOVERABLE_EXCEPTIONS:
        log.exception("Failed to forward feed bar to UI (symbol=%s)", symbol)

def _on_tick_from_feed(self, quote: Any) -> None:
    """Forward feed quote updates to UI thread safely."""
    if not CONFIG.is_market_open():
        return
    try:
        symbol = self._ui_norm(getattr(quote, "code", ""))
        price = float(getattr(quote, "price", 0) or 0)
        if symbol and price > 0:
            now = time.monotonic()
            prev = self._last_quote_ui_emit.get(symbol)
            if prev is not None:
                prev_ts, prev_px = float(prev[0]), float(prev[1])
                if (
                    (now - prev_ts) < 0.08
                    and abs(price - prev_px)
                    <= max(0.001, abs(prev_px) * 0.00005)
                ):
                    return
            self._last_quote_ui_emit[symbol] = (now, price)
            self.quote_received.emit(symbol, price)
    except _UI_RECOVERABLE_EXCEPTIONS:
        log.exception("Failed to forward feed quote to UI")

def _on_bar_ui(self, symbol: str, bar: dict[str, Any]) -> None:
    """
    Handle bar data on UI thread.

    FIXED: Now properly updates chart with all three layers.
    """
    symbol = self._ui_norm(symbol)
    if not symbol:
        return

    arr = self._bars_by_symbol.get(symbol)
    if arr is None:
        arr = []
        self._bars_by_symbol[symbol] = arr

    ui_interval = self._normalize_interval_token(
        self.interval_combo.currentText()
    )
    bar_interval_raw = bar.get("interval")
    if (bar_interval_raw is None or str(bar_interval_raw).strip() == "") and (
        "interval_seconds" in bar
    ):
        bar_interval_raw = self._interval_token_from_seconds(
            bar.get("interval_seconds")
        )
    source_interval = self._normalize_interval_token(
        bar_interval_raw,
        fallback=ui_interval,
    )
    interval = source_interval
    aggregate_to_ui = False
    if source_interval != ui_interval:
        try:
            source_s = int(max(1, self._interval_seconds(source_interval)))
            ui_s = int(max(1, self._interval_seconds(ui_interval)))
            aggregate_to_ui = source_s < ui_s
        except _UI_RECOVERABLE_EXCEPTIONS:
            aggregate_to_ui = False
    # Drop stale/coarser bars from previous interval after interval switch.
    if (
        bar_interval_raw is not None
        and str(bar_interval_raw).strip() != ""
        and source_interval != ui_interval
        and not aggregate_to_ui
    ):
        self._debug_console(
            f"bar_iv_mismatch:{symbol}:{ui_interval}",
            (
                f"drop feed bar {symbol}: feed_iv={source_interval} ui_iv={ui_interval} "
                f"raw_iv={bar_interval_raw} ts={bar.get('timestamp', bar.get('time', '--'))}"
            ),
            min_gap_seconds=1.0,
        )
        return
    if aggregate_to_ui:
        interval = ui_interval
    if arr:
        same_interval = [
            b for b in arr
            if self._normalize_interval_token(
                b.get("interval", interval), fallback=interval
            ) == interval
        ]
        if len(same_interval) != len(arr):
            arr[:] = same_interval

    ts_raw = bar.get("timestamp", bar.get("time"))
    if not self._is_market_session_timestamp(ts_raw, interval):
        return
    if ts_raw is None:
        ts_raw = self._now_iso()
    ts_epoch = self._ts_to_epoch(ts_raw)
    ts_bucket = self._bar_bucket_epoch(ts_epoch, interval)
    ts = self._epoch_to_iso(ts_bucket)
    ts_key = int(ts_bucket)

    try:
        c = float(bar.get("close", 0) or 0)
        o = float(bar.get("open", c) or c)
        h = float(bar.get("high", c) or c)
        low = float(bar.get("low", c) or c)
    except _UI_RECOVERABLE_EXCEPTIONS:
        return

    ref_close = None
    prev_epoch = None
    if arr:
        try:
            ref_close = float(arr[-1].get("close", 0) or 0)
            prev_epoch = self._bar_bucket_epoch(
                arr[-1].get("_ts_epoch", arr[-1].get("timestamp", ts_bucket)),
                interval,
            )
        except _UI_RECOVERABLE_EXCEPTIONS:
            ref_close = None
            prev_epoch = None
    if (
        ref_close
        and float(ref_close) > 0
        and prev_epoch is not None
        and self._is_intraday_day_boundary(prev_epoch, ts_bucket, interval)
    ):
        ref_close = None
    sanitized = self._sanitize_ohlc(
        o,
        h,
        low,
        c,
        interval=interval,
        ref_close=ref_close,
    )
    if sanitized is None:
        self._debug_console(
            f"bar_sanitize_drop:{symbol}:{interval}",
            (
                f"sanitize drop {symbol} {interval}: "
                f"o={o:.4f} h={h:.4f} l={low:.4f} c={c:.4f} "
                f"ref={float(ref_close or 0.0):.4f}"
            ),
            min_gap_seconds=0.8,
        )
        return
    o, h, low, c = sanitized

    is_final = bool(bar.get("final", True))
    if aggregate_to_ui:
        try:
            source_bucket = self._bar_bucket_epoch(ts_epoch, source_interval)
            source_step = int(max(1, self._interval_seconds(source_interval)))
            next_target_bucket = self._bar_bucket_epoch(
                float(source_bucket) + float(source_step),
                interval,
            )
            # Finer source bar is final for target bucket only at boundary.
            is_final = bool(is_final and int(next_target_bucket) != int(ts_bucket))
        except _UI_RECOVERABLE_EXCEPTIONS:
            is_final = False
    norm_bar: dict[str, Any] = {
        "open": o,
        "high": h,
        "low": low,
        "close": c,
        "timestamp": ts,
        "_ts_epoch": float(ts_bucket),
        "final": is_final,
        "interval": interval,
    }
    try:
        vol_val = float(bar.get("volume", 0) or 0.0)
    except _UI_RECOVERABLE_EXCEPTIONS:
        vol_val = 0.0
    if (not math.isfinite(vol_val)) or vol_val < 0:
        vol_val = 0.0

    try:
        amt_val = float(bar.get("amount", 0) or 0.0)
    except _UI_RECOVERABLE_EXCEPTIONS:
        amt_val = 0.0
    if not math.isfinite(amt_val):
        amt_val = 0.0
    if amt_val <= 0 and vol_val > 0 and c > 0:
        amt_val = float(c) * float(vol_val)

    norm_bar["volume"] = float(vol_val)
    norm_bar["amount"] = float(max(0.0, amt_val))

    # Guard against bad feed bars causing endpoint jumps/spikes.
    if arr:
        try:
            ref = float(arr[-1].get("close", c) or c)
            ref_epoch = self._bar_bucket_epoch(
                arr[-1].get("_ts_epoch", arr[-1].get("timestamp", ts_bucket)),
                interval,
            )
            if self._is_intraday_day_boundary(ref_epoch, ts_bucket, interval):
                ref = 0.0
            if ref > 0 and self._is_outlier_tick(ref, c, interval=interval):
                self._debug_console(
                    f"bar_outlier:{symbol}:{interval}",
                    (
                        f"outlier drop {symbol} {interval}: "
                        f"prev={ref:.4f} new={c:.4f} "
                        f"jump={abs(c / ref - 1.0):.2%}"
                    ),
                    min_gap_seconds=0.6,
                )
                return
        except _UI_RECOVERABLE_EXCEPTIONS as exc:
            log.debug("Suppressed exception in ui/app.py", exc_info=exc)

    replaced = False
    if ts:
        for i in range(len(arr) - 1, max(-1, len(arr) - 8), -1):
            arr_epoch = self._bar_bucket_epoch(
                arr[i].get("_ts_epoch", arr[i].get("timestamp", ""))
                if isinstance(arr[i], dict)
                else time.time(),
                interval,
            )
            if int(arr_epoch) != ts_key:
                continue
            # Keep completed bars immutable; ignore stale partial rewrites.
            existing_final = bool(arr[i].get("final", False))
            if existing_final and not is_final:
                replaced = True
                break
            if aggregate_to_ui:
                existing = arr[i] if isinstance(arr[i], dict) else {}
                merged = dict(existing)
                try:
                    e_open = float(existing.get("open", 0) or 0)
                except _UI_RECOVERABLE_EXCEPTIONS:
                    e_open = 0.0
                try:
                    e_high = float(existing.get("high", c) or c)
                except _UI_RECOVERABLE_EXCEPTIONS:
                    e_high = c
                try:
                    e_low = float(existing.get("low", c) or c)
                except _UI_RECOVERABLE_EXCEPTIONS:
                    e_low = c
                merged["open"] = e_open if e_open > 0 else o
                merged["high"] = float(max(e_high, h, o, c))
                merged["low"] = float(min(e_low, low, o, c))
                merged["close"] = float(c)
                merged["timestamp"] = ts
                merged["_ts_epoch"] = float(ts_bucket)
                merged["interval"] = interval
                merged["final"] = bool(existing_final or is_final)
                if ("volume" in norm_bar) or ("volume" in existing):
                    try:
                        e_vol = float(existing.get("volume", 0) or 0.0)
                    except _UI_RECOVERABLE_EXCEPTIONS:
                        e_vol = 0.0
                    try:
                        n_vol = float(norm_bar.get("volume", 0) or 0.0)
                    except _UI_RECOVERABLE_EXCEPTIONS:
                        n_vol = 0.0
                    merged["volume"] = float(max(0.0, e_vol) + max(0.0, n_vol))
                if ("amount" in norm_bar) or ("amount" in existing):
                    try:
                        e_amt = float(existing.get("amount", 0) or 0.0)
                    except _UI_RECOVERABLE_EXCEPTIONS:
                        e_amt = 0.0
                    try:
                        n_amt = float(norm_bar.get("amount", 0) or 0.0)
                    except _UI_RECOVERABLE_EXCEPTIONS:
                        n_amt = 0.0
                    merged["amount"] = float(max(0.0, e_amt) + max(0.0, n_amt))
                arr[i] = merged
            else:
                arr[i] = norm_bar
            replaced = True
            break
    if not replaced:
        if arr:
            try:
                prev_bucket = self._bar_bucket_epoch(
                    arr[-1].get("_ts_epoch", arr[-1].get("timestamp", ts_bucket)),
                    interval,
                )
                if int(prev_bucket) != ts_key and not bool(arr[-1].get("final", True)):
                    arr[-1]["final"] = True
            except _UI_RECOVERABLE_EXCEPTIONS as exc:
                log.debug("Suppressed exception in ui/app.py", exc_info=exc)
        arr.append(norm_bar)

    arr.sort(
        key=lambda x: float(
            x.get("_ts_epoch", self._ts_to_epoch(x.get("timestamp", "")))
        )
    )
    keep = self._history_window_bars(interval)
    if len(arr) > keep:
        del arr[:-keep]
    self._last_bar_feed_ts[symbol] = time.time()

    # Avoid excessive disk writes for partial updates.
    min_gap = 0.9 if not is_final else 0.0
    persist_fn = getattr(self, "_persist_session_bar", None)
    if callable(persist_fn):
        persist_fn(
            symbol,
            interval,
            norm_bar,
            channel="bar_ui",
            min_gap_seconds=min_gap,
        )

    current_code = self._ui_norm(self.stock_input.text())
    if current_code != symbol:
        return

    render_fn = getattr(self, "_render_live_bar_update", None)
    if callable(render_fn):
        render_fn(
            symbol=symbol,
            interval=interval,
            bars=arr,
            norm_bar=norm_bar,
        )

def _render_live_bar_update(
    self,
    *,
    symbol: str,
    interval: str,
    bars: list[dict[str, Any]],
    norm_bar: dict[str, Any],
) -> None:
    predicted, pred_source_interval = self._resolve_chart_prediction_series(
        symbol=symbol,
        fallback_interval=interval,
    )
    try:
        current_price = float(norm_bar.get("close", 0) or 0)
        self._render_chart_state(
            symbol=symbol,
            interval=interval,
            bars=bars,
            context="bar_ui",
            current_price=current_price if current_price > 0 else None,
            predicted_prices=predicted,
            source_interval=pred_source_interval,
            target_steps=int(self.forecast_spin.value()),
            update_latest_label=True,
            allow_legacy_candles=True,
        )
    except _UI_RECOVERABLE_EXCEPTIONS as e:
        log.debug(f"Chart update failed: {e}")

def _update_chart_latest_label(
    self,
    symbol: str,
    *,
    bar: dict[str, Any] | None = None,
    price: float | None = None,
) -> None:
    """Show latest quote/bar summary below chart."""
    label = getattr(self, "chart_latest_label", None)
    if label is None:
        return
    try:
        if bar:
            o = float(bar.get("open", 0) or 0)
            h = float(bar.get("high", 0) or 0)
            low = float(bar.get("low", 0) or 0)
            c = float(bar.get("close", 0) or 0)
            ts = bar.get("timestamp") or bar.get("time") or "--"
            label.setText(
                f"Latest {symbol} | O {o:.2f}  H {h:.2f}  L {low:.2f}  C {c:.2f} | {ts}"
            )
        elif price is not None and float(price) > 0:
            label.setText(
                f"Latest {symbol} | Price {float(price):.2f} | waiting for OHLC bar"
            )
    except _UI_RECOVERABLE_EXCEPTIONS as exc:
        log.debug("Suppressed exception in ui/app.py", exc_info=exc)

