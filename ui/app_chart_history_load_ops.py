from __future__ import annotations

import math
import time
from datetime import date, datetime
from typing import Any

from config.settings import CONFIG
from utils.logger import get_logger
from utils.recoverable import COMMON_RECOVERABLE_EXCEPTIONS

log = get_logger(__name__)

_APP_CHART_RECOVERABLE_EXCEPTIONS = COMMON_RECOVERABLE_EXCEPTIONS


def _fetch_chart_history_frame(
    fetcher: Any,
    symbol: str,
    *,
    source_iv: str,
    bars: int,
    use_cache: bool,
    update_db: bool,
    allow_online: bool,
    refresh_intraday_after_close: bool,
) -> Any:
    """Fetch chart history while remaining backward-compatible with older signatures."""
    try:
        return fetcher.get_history(
            symbol,
            interval=source_iv,
            bars=bars,
            use_cache=bool(use_cache),
            update_db=bool(update_db),
            allow_online=bool(allow_online),
            refresh_intraday_after_close=bool(refresh_intraday_after_close),
        )
    except TypeError:
        return fetcher.get_history(
            symbol,
            interval=source_iv,
            bars=bars,
            use_cache=bool(use_cache),
            update_db=bool(update_db),
        )


def _append_normalized_chart_rows(
    self: Any,
    *,
    frame: Any,
    source_iv: str,
    source_lookback: int,
    out: list[dict[str, Any]],
    prev_close: float | None,
    prev_epoch: float | None,
    is_intraday: bool,
    market_open: bool,
    now_bucket: float,
    default_final: bool,
    ts_field: str | None = "datetime",
    final_field: str | None = None,
) -> tuple[float | None, float | None]:
    """Normalize history rows into chart bars and append to output list."""
    if frame is None or frame.empty:
        return prev_close, prev_epoch
    for idx, row in frame.tail(source_lookback).iterrows():
        try:
            c = float(row.get("close", 0) or 0)
        except _APP_CHART_RECOVERABLE_EXCEPTIONS:
            continue
        if c <= 0 or not math.isfinite(c):
            continue

        ts_obj = idx if ts_field is None else row.get(ts_field, idx)
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

        try:
            h = float(row.get("high", max(o, c)) or max(o, c))
            low = float(row.get("low", min(o, c)) or min(o, c))
        except _APP_CHART_RECOVERABLE_EXCEPTIONS:
            h = float(max(o, c))
            low = float(min(o, c))

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
        if not math.isfinite(vol) or vol < 0:
            vol = 0.0

        try:
            amount = float(row.get("amount", 0) or 0.0)
        except _APP_CHART_RECOVERABLE_EXCEPTIONS:
            amount = 0.0
        if not math.isfinite(amount):
            amount = 0.0
        if amount <= 0 and vol > 0 and c > 0:
            amount = float(c) * float(vol)

        is_final = bool(default_final)
        if final_field is not None:
            is_final = bool(row.get(final_field, default_final))
        if (
            is_intraday
            and not is_final
            and ((not market_open) or int(epoch) != int(now_bucket))
        ):
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
        prev_close = float(c)
        prev_epoch = float(epoch)
    return prev_close, prev_epoch


def _load_chart_history_bars(
    self: Any,
    symbol: str,
    interval: str,
    lookback_bars: int,
    _recursion_depth: int = 0,
) -> list[dict[str, Any]]:
    """Load historical OHLC bars for chart rendering with cache fallback."""
    _ = _recursion_depth
    if not self.predictor:
        return []
    try:
        fetcher = getattr(self.predictor, "fetcher", None)
        if fetcher is None:
            # Fallback to _bars_by_symbol cache if fetcher unavailable
            arr = list(self._bars_by_symbol.get(symbol) or [])
            if arr:
                return arr[-max(1, int(lookback_bars)):]
            return []

        requested_iv = self._normalize_interval_token(interval)
        norm_iv = requested_iv or "1m"
        source_iv = "1m" if norm_iv not in {"1d", "1wk", "1mo"} else norm_iv
        lookback = max(
            int(max(1, lookback_bars)),
            int(self._recommended_lookback(norm_iv)),
        )

        if source_iv == norm_iv:
            source_lookback = int(
                max(int(lookback), int(self._recommended_lookback(source_iv)))
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

        selected_date_text = str(getattr(self, "_selected_chart_date", "") or "").strip()
        selected_date: date | None = None
        if selected_date_text:
            try:
                selected_date = date.fromisoformat(selected_date_text)
            except ValueError:
                selected_date = None
        if selected_date is not None:
            try:
                from data.fetcher import BARS_PER_DAY

                bpd = float(BARS_PER_DAY.get(source_iv, 1.0) or 1.0)
            except _APP_CHART_RECOVERABLE_EXCEPTIONS:
                step = float(max(1, self._interval_seconds(source_iv)))
                bpd = float(max(1.0, round(86400.0 / step)))
            days_back = max(0, (datetime.now().date() - selected_date).days)
            source_lookback = max(
                source_lookback,
                int((days_back + 2) * max(1.0, bpd)),
            )

        # FIX: Try cache first for faster response and reliability
        df = _fetch_chart_history_frame(
            fetcher,
            symbol,
            source_iv=source_iv,
            bars=source_lookback,
            use_cache=True,  # Enable cache for fallback reliability
            update_db=True,
            allow_online=True,
            refresh_intraday_after_close=True,
        )
        
        # FIX: If online fetch fails, try multiple fallbacks
        if df is None or df.empty:
            # Fallback 1: Try session cache directly
            try:
                from data.session_cache import get_session_bar_cache
                cache = get_session_bar_cache()
                session_bars = cache.read_history(
                    symbol=symbol,
                    interval=source_iv,
                    bars=source_lookback,
                )
                if session_bars is not None and not session_bars.empty:
                    df = session_bars
                    log.debug(
                        "Chart history fallback to session cache for %s (%s): %d bars",
                        symbol, source_iv, len(df)
                    )
            except _APP_CHART_RECOVERABLE_EXCEPTIONS as exc:
                log.debug("Session cache fallback failed for %s: %s", symbol, exc)
            
            # Fallback 2: Use _bars_by_symbol if available
            if df is None or df.empty:
                arr = list(self._bars_by_symbol.get(symbol) or [])
                if arr:
                    log.debug(
                        "Chart history fallback to _bars_by_symbol for %s (%s): %d bars",
                        symbol, source_iv, len(arr)
                    )
                    return arr[-max(1, int(lookback_bars)):]
            
            # Fallback 3: Return empty if all sources failed
            if df is None or df.empty:
                log.debug(
                    "Chart history unavailable for %s (%s): all sources failed",
                    symbol, source_iv
                )
                return []

        out: list[dict[str, Any]] = []
        _append_normalized_chart_rows(
            self,
            frame=df,
            source_iv=source_iv,
            source_lookback=source_lookback,
            out=out,
            prev_close=None,
            prev_epoch=None,
            is_intraday=norm_iv not in {"1d", "1wk", "1mo"},
            market_open=bool(CONFIG.is_market_open()),
            now_bucket=self._bar_bucket_epoch(time.time(), source_iv),
            default_final=True,
            ts_field="datetime",
            final_field=None,
        )

        out = self._filter_bars_to_market_session(out, source_iv)
        out = self._merge_bars([], out, source_iv)
        if norm_iv != source_iv:
            out = self._resample_chart_bars(
                out,
                source_interval=source_iv,
                target_interval=norm_iv,
            )

        if selected_date is not None:
            selected_out: list[dict[str, Any]] = []
            for row in out:
                ts_raw = row.get("_ts_epoch", row.get("timestamp"))
                try:
                    epoch = float(self._ts_to_epoch(ts_raw))
                except _APP_CHART_RECOVERABLE_EXCEPTIONS:
                    continue
                try:
                    from zoneinfo import ZoneInfo

                    row_date = datetime.fromtimestamp(
                        epoch, tz=ZoneInfo("Asia/Shanghai")
                    ).date()
                except _APP_CHART_RECOVERABLE_EXCEPTIONS:
                    row_date = datetime.fromtimestamp(epoch).date()
                if row_date == selected_date:
                    selected_out.append(row)
            out = selected_out

        return out[-lookback:]
    except _APP_CHART_RECOVERABLE_EXCEPTIONS as e:
        log.debug("Historical chart load failed for %s: %s", symbol, e)
        return []
