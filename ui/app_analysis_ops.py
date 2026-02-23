from __future__ import annotations

import time
from datetime import datetime
from importlib import import_module
from typing import Any

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import QMessageBox, QTableWidgetItem

from config.settings import CONFIG
from core.types import AutoTradeMode
from ui.background_tasks import WorkerThread
from ui.background_tasks import sanitize_watch_list as _sanitize_watch_list
from ui.background_tasks import validate_stock_code as _validate_stock_code
from ui.modern_theme import ModernColors, ModernFonts
from utils.logger import get_logger
from utils.recoverable import COMMON_RECOVERABLE_EXCEPTIONS

log = get_logger(__name__)
_UI_RECOVERABLE_EXCEPTIONS = COMMON_RECOVERABLE_EXCEPTIONS

def _lazy_get(module: str, name: str) -> Any:
    return getattr(import_module(module), name)

def _quick_trade(self, pred: Any) -> None:
    """Quick trade from signal"""
    self.stock_input.setText(pred.stock_code)
    self._analyze_stock()

def _update_watchlist(self) -> None:
    """Update watchlist display"""
    sanitized = _sanitize_watch_list(
        self.watch_list,
        max_size=self.MAX_WATCHLIST_SIZE,
    )
    if sanitized != self.watch_list:
        self.watch_list = sanitized

    current_count = self.watchlist.rowCount()

    if current_count != len(self.watch_list):
        self.watchlist.setRowCount(len(self.watch_list))

    row_map: dict[str, int] = {}
    for row, code in enumerate(self.watch_list):
        norm_code = self._ui_norm(str(code))
        if norm_code:
            row_map[norm_code] = int(row)
        current_code = self.watchlist.item(row, 0)
        if current_code is None or current_code.text() != code:
            self.watchlist.setItem(row, 0, QTableWidgetItem(code))

        for col in range(1, 4):
            cell = self.watchlist.item(row, col)
            if cell is None:
                cell = QTableWidgetItem("--")
                self.watchlist.setItem(row, col, cell)
            elif not cell.text():
                cell.setText("--")

    self._watchlist_row_by_code = row_map
    if self._last_watchlist_price_ui:
        active = set(row_map.keys())
        stale = [
            k for k in self._last_watchlist_price_ui.keys()
            if k not in active
        ]
        for k in stale:
            self._last_watchlist_price_ui.pop(k, None)
    if self._last_quote_ui_emit:
        active = set(row_map.keys())
        stale_quotes = [
            k for k in self._last_quote_ui_emit.keys()
            if k not in active
        ]
        for k in stale_quotes:
            self._last_quote_ui_emit.pop(k, None)

    # Evict bar caches for symbols no longer in watchlist or active chart.
    if self._bars_by_symbol:
        active_syms = set(row_map.keys())
        selected = self._ui_norm(self.stock_input.text())
        if selected:
            active_syms.add(selected)
        max_inactive = 10
        stale_bars = [
            k for k in self._bars_by_symbol.keys()
            if k not in active_syms
        ]
        if len(stale_bars) > max_inactive:
            for k in stale_bars[max_inactive:]:
                self._bars_by_symbol.pop(k, None)

def _on_watchlist_click(
    self, row: int, col: int, code_override: str | None = None
) -> None:
    """Handle watchlist click and load selected stock reliably."""
    _ = col
    code = self._ui_norm(code_override or "")
    if not code:
        if row < 0 or row >= self.watchlist.rowCount():
            return
        item = self.watchlist.item(row, 0)
        if not item:
            return
        code = self._ui_norm(item.text())
    if not code:
        return

    self.stock_input.setText(code)

    old_worker = self.workers.get("analyze")
    if old_worker and old_worker.isRunning():
        old_worker.cancel()
    forecast_worker = self.workers.get("forecast_refresh")
    if forecast_worker and forecast_worker.isRunning():
        forecast_worker.cancel()
    self._forecast_refresh_symbol = ""
    self._last_forecast_refresh_ts = 0.0

    if self._chart_symbol and self._chart_symbol != code:
        try:
            if hasattr(self.chart, "reset_view"):
                self.chart.reset_view()
        except _UI_RECOVERABLE_EXCEPTIONS as exc:
            log.debug("Suppressed exception in ui/app.py", exc_info=exc)

    try:
        self._queue_history_refresh(
            code,
            self._normalize_interval_token(self.interval_combo.currentText()),
        )
    except _UI_RECOVERABLE_EXCEPTIONS as exc:
        log.debug("Suppressed exception in ui/app.py", exc_info=exc)

    self._analyze_stock()

def _add_to_watchlist(self) -> None:
    """Add stock to watchlist with validation"""
    code = self.stock_input.text().strip()
    normalized = self._ui_norm(code)

    if not normalized:
        self.log("Please enter a stock code", "warning")
        return

    if not _validate_stock_code(normalized):
        self.log(f"Invalid stock code: {code}", "warning")
        return

    if len(self.watch_list) >= self.MAX_WATCHLIST_SIZE:
        self.log(
            f"Watchlist full (max {self.MAX_WATCHLIST_SIZE})", "warning"
        )
        return

    if normalized not in self.watch_list:
        self.watch_list.append(normalized)
        self.watch_list = _sanitize_watch_list(
            self.watch_list,
            max_size=self.MAX_WATCHLIST_SIZE,
        )
        self._update_watchlist()
        self.log(f"Added {normalized} to watchlist", "info")
        self._ensure_feed_subscription(normalized)

        # Sync with auto-trader
        if self.executor and self.executor.auto_trader:
            self.executor.auto_trader.update_watchlist(self.watch_list)
    else:
        self.log(f"{normalized} already in watchlist", "info")

def _remove_from_watchlist(self) -> None:
    """Remove selected stock from watchlist"""
    row = self.watchlist.currentRow()
    if row >= 0 and row < self.watchlist.rowCount():
        item = self.watchlist.item(row, 0)
        if item:
            code = item.text()
            if code in self.watch_list:
                self.watch_list.remove(code)
                self._update_watchlist()
                self.log(f"Removed {code} from watchlist", "info")

                # Sync with auto-trader
                if self.executor and self.executor.auto_trader:
                    self.executor.auto_trader.update_watchlist(
                        self.watch_list
                    )

def _analyze_stock(self) -> None:
    """Analyze stock with validation"""
    code = self.stock_input.text().strip()
    if not code:
        self.log("Please enter a stock code", "warning")
        return

    normalized = self._ui_norm(code)
    if not normalized:
        self.log("Invalid stock code format", "warning")
        return

    if self.predictor is None or self.predictor.ensemble is None:
        self.log(
            "No model loaded. Please train a model first.", "error"
        )
        return

    interval = self._normalize_interval_token(
        self.interval_combo.currentText()
    )
    self._log_model_alignment_debug(
        context="analyze",
        requested_interval=interval,
        requested_horizon=int(self.forecast_spin.value()),
    )
    is_trained = self._is_trained_stock(normalized)
    if is_trained:
        existing = list(self._bars_by_symbol.get(normalized) or [])
        same_interval = [
            b
            for b in existing
            if self._normalize_interval_token(
                b.get("interval", interval),
                fallback=interval,
            ) == interval
        ]
        if not same_interval:
            self._queue_history_refresh(normalized, interval)
    if not is_trained:
        # Preserve user-selected interval even for non-trained symbols.
        if interval in {"1d", "1wk", "1mo"}:
            target_lookback = max(
                60,
                int(self._recommended_lookback(interval)),
            )
        else:
            target_lookback = max(
                120,
                int(self._seven_day_lookback(interval)),
            )
        self.lookback_spin.setValue(target_lookback)
        self._queue_history_refresh(normalized, interval)
        self._debug_console(
            f"non_trained_policy:{normalized}",
            (
                f"non-trained symbol policy (preserve interval): "
                f"symbol={normalized} iv={interval} lookback={target_lookback}"
            ),
            min_gap_seconds=1.0,
            level="info",
        )
    forecast_bars = int(self.forecast_spin.value())
    ui_lookback = max(
        int(self.lookback_spin.value()),
        int(self._recommended_lookback(interval)),
    )
    forecast_lookback = int(ui_lookback)
    use_realtime = bool(CONFIG.is_market_open())
    infer_interval = "1m"
    infer_horizon = int(forecast_bars)
    infer_lookback = int(
        max(
            self._recommended_lookback("1m"),
            self._bars_needed_from_base_interval(
                interval,
                int(forecast_lookback),
                base_interval="1m",
            ),
        )
    )
    history_allow_online = True
    skip_cache = True
    if not self._has_exact_model_artifacts(infer_interval, infer_horizon):
        self._debug_console(
            f"analyze_model_fallback:{normalized}:{interval}",
            (
                f"analyze inference locked to 1m for {normalized}: "
                f"ui={interval}/{forecast_bars} infer={infer_interval}/{infer_horizon} "
                f"lookback={infer_lookback} online=1"
            ),
            min_gap_seconds=1.0,
            level="warning",
        )

    # FIX: Improved request deduplication with sequence numbers and adaptive throttling
    request_key = (
        f"{normalized}:{interval}:{forecast_bars}:"
        f"{int(infer_lookback)}:{int(use_realtime)}:"
        f"{infer_interval}:{int(infer_horizon)}:{int(history_allow_online)}"
    )
    req_now = time.monotonic()
    
    # Check for duplicate request with adaptive throttle window
    last_req = dict(self._last_analyze_request or {})
    throttle_window = float(_ANALYSIS_THROTTLE_SECONDS)
    
    if (
        last_req.get("key") == request_key
        and (req_now - float(last_req.get("ts", 0.0) or 0.0)) < throttle_window
    ):
        self._debug_console(
            f"analyze_dedup:{normalized}:{interval}",
            f"Skipped duplicate analyze request for {normalized} ({interval}) within {throttle_window}s",
            min_gap_seconds=8.0,
            level="info",
        )
        return
    
    # Store request with sequence number for stale result detection
    self._analyze_request_seq = getattr(self, '_analyze_request_seq', 0) + 1
    self._last_analyze_request = {
        "key": request_key,
        "ts": req_now,
        "seq": self._analyze_request_seq,
    }
    current_seq = self._analyze_request_seq

    self.analyze_action.setEnabled(False)

    if hasattr(self.signal_panel, 'reset'):
        self.signal_panel.reset()

    self.status_label.setText(f"Analyzing {normalized}...")
    self.progress.setRange(0, 0)
    self.progress.show()

    # Cancel existing worker
    old_worker = self.workers.get("analyze")
    if old_worker and old_worker.isRunning():
        old_worker.cancel()

    def analyze():
        return self.predictor.predict(
            normalized,
            use_realtime_price=use_realtime,
            interval=infer_interval,
            forecast_minutes=infer_horizon,
            lookback_bars=infer_lookback,
            skip_cache=bool(skip_cache),
            history_allow_online=history_allow_online,
        )

    worker = WorkerThread(analyze, timeout_seconds=120)
    self._track_worker(worker)
    
    # FIX: Tag worker with sequence number for stale detection
    worker._request_seq = current_seq
    
    worker.result.connect(self._on_analysis_done)
    worker.error.connect(self._on_analysis_error)
    self.workers["analyze"] = worker
    worker.start()

def _on_analysis_done(self, pred: Any) -> None:
    """
    Handle analysis completion; also triggers news fetch.

    FIX: Added stale result detection using sequence numbers.
    """
    # FIX: Check for stale result using sequence number
    worker = self.workers.get("analyze")
    if worker is not None:
        worker_seq = getattr(worker, '_request_seq', None)
        current_seq = getattr(self, '_analyze_request_seq', None)
        if worker_seq is not None and current_seq is not None and worker_seq != current_seq:
            # Stale result from older request - discard
            self._debug_console(
                f"analyze_stale:{self._ui_norm(self.stock_input.text())}",
                f"Discarded stale analysis result (seq {worker_seq} != {current_seq})",
                min_gap_seconds=2.0,
                level="debug",
            )
            self.workers.pop('analyze', None)
            return

    self.analyze_action.setEnabled(True)
    self.progress.hide()
    self.status_label.setText("Ready")

    symbol = self._ui_norm(getattr(pred, "stock_code", ""))
    selected = self._ui_norm(self.stock_input.text())
    if selected and symbol and selected != symbol:
        # User switched symbol while worker was running; ignore stale result.
        self.workers.pop('analyze', None)
        return

    self.current_prediction = pred

    if hasattr(self.signal_panel, 'update_prediction'):
        self.signal_panel.update_prediction(pred)

    current_price = float(getattr(pred, "current_price", 0) or 0)
    interval = self._normalize_interval_token(
        self.interval_combo.currentText()
    )
    lookback = max(
        int(self.lookback_spin.value()),
        self._seven_day_lookback(interval),
    )

    if symbol:
        arr = self._load_chart_history_bars(symbol, interval, lookback)
        existing = self._bars_by_symbol.get(symbol) or []
        if existing:
            existing_same_interval = [
                b for b in existing
                if self._normalize_interval_token(
                    b.get("interval", interval),
                    fallback=interval,
                ) == interval
            ]
            # Avoid re-injecting stale malformed bars from in-memory cache.
            # Only merge the current live partial bucket while market is open.
            if existing_same_interval and CONFIG.is_market_open():
                now_bucket = self._bar_bucket_epoch(time.time(), interval)
                live_partial: list[dict[str, Any]] = []
                for b in existing_same_interval:
                    if bool(b.get("final", True)):
                        continue
                    b_bucket = self._bar_bucket_epoch(
                        b.get("_ts_epoch", b.get("timestamp")),
                        interval,
                    )
                    if int(b_bucket) == int(now_bucket):
                        live_partial.append(b)
                if live_partial:
                    arr = self._merge_bars(arr, live_partial, interval)

            # If newly loaded chart depth is far smaller than the existing
            # cached window, keep the deeper existing history to prevent
            # oscillation between full chart and tiny placeholder blocks.
            if existing_same_interval:
                old_len = len(existing_same_interval)
                new_len = len(arr or [])
                if new_len <= 0:
                    arr = list(existing_same_interval)
                elif old_len >= 12 and new_len < max(6, int(old_len * 0.45)):
                    merged_depth = self._merge_bars(
                        existing_same_interval,
                        arr,
                        interval,
                    )
                    if len(merged_depth) >= max(new_len, int(old_len * 0.62)):
                        arr = merged_depth
                    else:
                        arr = list(existing_same_interval)
                    self._debug_console(
                        f"chart_depth_preserve:{symbol}:{interval}",
                        (
                            f"preserved deeper chart window for {symbol} {interval}: "
                            f"new={new_len} old={old_len} final={len(arr)}"
                        ),
                        min_gap_seconds=1.0,
                        level="info",
                    )
        arr = self._filter_bars_to_market_session(arr, interval)

        if not arr and current_price > 0:
            arr = [{
                "open": current_price,
                "high": current_price,
                "low": current_price,
                "close": current_price,
                "timestamp": self._now_iso(),
                "final": False,
                "interval": interval,
                "_ts_epoch": time.time(),
            }]

        if arr and current_price > 0:
            update_last = True
            prev_ref: float | None = None
            if len(arr) >= 2:
                try:
                    prev_epoch = self._bar_bucket_epoch(
                        arr[-2].get("_ts_epoch", arr[-2].get("timestamp")),
                        interval,
                    )
                    last_epoch = self._bar_bucket_epoch(
                        arr[-1].get("_ts_epoch", arr[-1].get("timestamp")),
                        interval,
                    )
                    if not self._is_intraday_day_boundary(
                        prev_epoch,
                        last_epoch,
                        interval,
                    ):
                        prev_ref = float(
                            arr[-2].get("close", current_price) or current_price
                        )
                    if (
                        prev_ref
                        and prev_ref > 0
                        and self._is_outlier_tick(
                            prev_ref,
                            current_price,
                            interval=interval,
                        )
                    ):
                        update_last = False
                except _UI_RECOVERABLE_EXCEPTIONS:
                    update_last = True
            if update_last:
                s = self._sanitize_ohlc(
                    float(arr[-1].get("open", current_price) or current_price),
                    max(
                        float(arr[-1].get("high", current_price) or current_price),
                        current_price,
                    ),
                    min(
                        float(arr[-1].get("low", current_price) or current_price),
                        current_price,
                    ),
                    current_price,
                    interval=interval,
                    ref_close=prev_ref if (prev_ref and prev_ref > 0) else None,
                )
                if s is not None:
                    o, h, low, c = s
                    arr[-1]["open"] = o
                    arr[-1]["high"] = h
                    arr[-1]["low"] = low
                    arr[-1]["close"] = c
                arr[-1]["final"] = False
                if "_ts_epoch" not in arr[-1]:
                    arr[-1]["_ts_epoch"] = self._bar_bucket_epoch(
                        arr[-1].get("timestamp"),
                        interval,
                    )
                arr[-1]["timestamp"] = self._epoch_to_iso(arr[-1]["_ts_epoch"])

        if arr:
            try:
                arr = self._render_chart_state(
                    symbol=symbol,
                    interval=interval,
                    bars=arr,
                    context="analysis_done",
                    current_price=current_price if current_price > 0 else None,
                    predicted_prices=getattr(pred, "predicted_prices", []) or [],
                    source_interval=self._normalize_interval_token(
                        getattr(pred, "interval", interval),
                        fallback=interval,
                    ),
                    target_steps=int(self.forecast_spin.value()),
                    update_latest_label=True,
                    reset_view_on_symbol_switch=True,
                )
            except _UI_RECOVERABLE_EXCEPTIONS as e:
                log.debug(f"Chart update failed: {e}")

    # Update details (with news sentiment)
    self._update_details(pred)

    if (
        hasattr(self, 'news_panel')
        and hasattr(self.news_panel, 'set_stock')
    ):
        try:
            self.news_panel.set_stock(pred.stock_code)
        except _UI_RECOVERABLE_EXCEPTIONS as e:
            log.debug(f"News fetch for {pred.stock_code}: {e}")

    self._add_to_history(pred)

    try:
        self._ensure_feed_subscription(pred.stock_code)
    except _UI_RECOVERABLE_EXCEPTIONS as exc:
        log.debug("Suppressed exception in ui/app.py", exc_info=exc)
    if not (self.monitor and self.monitor.isRunning()):
        try:
            self.monitor_action.setChecked(True)
            self._start_monitoring()
        except _UI_RECOVERABLE_EXCEPTIONS as exc:
            log.debug("Suppressed exception in ui/app.py", exc_info=exc)

    # FIX: Enable trading buttons based on mode and stock selection
    # In MANUAL mode, always enable buttons when a valid stock is selected
    # In AUTO mode, disable buttons (AI handles trading)
    # In SEMI-AUTO mode, enable buttons for manual override
    Signal = _lazy_get("models.predictor", "Signal")
    is_manual = (self._auto_trade_mode == AutoTradeMode.MANUAL)
    is_semi_auto = (self._auto_trade_mode == AutoTradeMode.SEMI_AUTO)
    has_valid_stock = bool(self._ui_norm(self.stock_input.text()))
    is_connected = bool(self.executor is not None)
    
    # Enable buttons when: manual/semi-auto mode + valid stock + connected
    buttons_enabled = is_connected and has_valid_stock and (is_manual or is_semi_auto)
    
    if hasattr(self, "buy_btn"):
        self.buy_btn.setEnabled(buttons_enabled)
    if hasattr(self, "sell_btn"):
        self.sell_btn.setEnabled(buttons_enabled)

    signal_text = (
        pred.signal.value
        if hasattr(pred.signal, 'value')
        else str(pred.signal)
    )
    conf = getattr(pred, 'confidence', 0)
    warnings = list(getattr(pred, "warnings", []) or [])
    pred_interval = self._normalize_interval_token(
        getattr(pred, "interval", interval),
        fallback=interval,
    )
    insufficient_data = (
        current_price <= 0
        or any(
            ("insufficient data" in str(w).lower())
            or ("prediction error" in str(w).lower())
            for w in warnings
        )
    )
    log_key = f"{pred.stock_code}:{signal_text}:{float(conf):.4f}:{int(insufficient_data)}"
    last_log = dict(self._last_analysis_log or {})
    now_log_ts = time.monotonic()
    should_log = True
    if (
        last_log.get("key") == log_key
        and (now_log_ts - float(last_log.get("ts", 0.0) or 0.0)) < 2.5
    ):
        should_log = False
    if should_log:
        if insufficient_data:
            self.log(
                f"Analysis partial: {pred.stock_code} - "
                f"{signal_text} ({conf:.0%}) | data not ready",
                "warning",
            )
            self._schedule_analysis_recovery(
                symbol=pred.stock_code,
                interval=pred_interval,
                warnings=warnings,
            )
        else:
            self.log(
                f"Analysis complete: {pred.stock_code} - "
                f"{signal_text} ({conf:.0%})",
                "success",
            )
        self._last_analysis_log = {"key": log_key, "ts": now_log_ts}

    self.workers.pop('analyze', None)

def _on_analysis_error(self, error: str) -> None:
    """Handle analysis error"""
    self.analyze_action.setEnabled(True)
    self.progress.hide()
    self.status_label.setText("Ready")

    self.log(f"Analysis failed: {error}", "error")
    QMessageBox.warning(self, "Error", f"Analysis failed:\n{error}")

    self.workers.pop('analyze', None)

def _update_details(self, pred: Any) -> None:
    """Update analysis details with news sentiment"""
    Signal = _lazy_get("models.predictor", "Signal")

    signal_colors = {
        Signal.STRONG_BUY: ModernColors.ACCENT_SUCCESS,
        Signal.BUY: ModernColors.ACCENT_SUCCESS,
        Signal.HOLD: ModernColors.ACCENT_WARNING,
        Signal.SELL: ModernColors.ACCENT_DANGER,
        Signal.STRONG_SELL: ModernColors.ACCENT_DANGER,
    }

    signal = getattr(pred, 'signal', Signal.HOLD)
    color = signal_colors.get(signal, ModernColors.TEXT_PRIMARY)
    signal_text = (
        signal.value if hasattr(signal, 'value') else str(signal)
    )

    def safe_get(obj: Any, attr: str, default: Any = 0) -> Any:
        return (
            getattr(obj, attr, default)
            if hasattr(obj, attr) else default
        )

    prob_up = safe_get(pred, 'prob_up', 0.33)
    prob_neutral = safe_get(pred, 'prob_neutral', 0.34)
    prob_down = safe_get(pred, 'prob_down', 0.33)
    signal_strength = safe_get(pred, 'signal_strength', 0)
    confidence = safe_get(pred, 'confidence', 0)
    agreement = safe_get(pred, 'model_agreement', 1.0)
    entropy = safe_get(pred, 'entropy', 0.0)
    margin = safe_get(pred, 'model_margin', 0.0)
    uncertainty_score = safe_get(pred, 'uncertainty_score', 0.5)
    tail_risk_score = safe_get(pred, 'tail_risk_score', 0.5)
    rsi = safe_get(pred, 'rsi', 50)
    macd_signal = safe_get(pred, 'macd_signal', 'N/A')
    trend = safe_get(pred, 'trend', 'N/A')
    levels = getattr(pred, 'levels', None)
    position = getattr(pred, 'position', None)
    reasons = getattr(pred, 'reasons', [])
    warnings = getattr(pred, 'warnings', [])

    news_html = ""
    try:
        from core.network import get_network_env
        from data.news import get_news_aggregator

        env = get_network_env()
        if env.is_china_direct or env.tencent_ok:
            agg = get_news_aggregator()
            sentiment = agg.get_sentiment_summary(pred.stock_code)
            snapshot = agg.get_institutional_snapshot(
                stock_code=pred.stock_code, hours_lookback=24
            )

            if sentiment and sentiment.get('total', 0) > 0:
                sent_score = sentiment['overall_sentiment']
                sent_label = sentiment['label']

                if sent_label == "positive":
                    sent_color = ModernColors.ACCENT_SUCCESS
                    sent_emoji = "UP"
                elif sent_label == "negative":
                    sent_color = ModernColors.ACCENT_DANGER
                    sent_emoji = "DOWN"
                else:
                    sent_color = ModernColors.ACCENT_WARNING
                    sent_emoji = "NEUTRAL"

                news_html = f"""
                <div class="section">
                    <span class="label">News Sentiment: </span>
                    <span style="color: {sent_color}; font-weight: bold;">
                        {sent_emoji} {sent_score:+.2f} ({sent_label})
                    </span>
                    <span class="label"> |
                        {sentiment['positive_count']} positive,
                        {sentiment['negative_count']} negative,
                        {sentiment['total']} total
                    </span>
                </div>
                """
                source_mix = snapshot.get("source_mix", {}) if isinstance(snapshot, dict) else {}
                top_sources = list(source_mix.items())[:3]
                mix_txt = ", ".join(
                    f"{src}:{ratio:.0%}" for src, ratio in top_sources
                ) if top_sources else "n/a"
                latest_age = (
                    snapshot.get("freshness", {}).get("latest_age_seconds")
                    if isinstance(snapshot, dict) else None
                )
                latest_txt = (
                    f"{float(latest_age):.0f}s ago"
                    if isinstance(latest_age, (int, float))
                    else "n/a"
                )
                news_html += f"""
                <div class="section">
                    <span class="label">News Coverage:</span>
                    sources {mix_txt} | latest {latest_txt}
                </div>
                """

                top_pos = sentiment.get('top_positive', [])
                top_neg = sentiment.get('top_negative', [])

                if top_pos or top_neg:
                    news_html += (
                        '<div class="section">'
                        '<span class="label">Key Headlines:</span><br/>'
                    )
                    for n in top_pos[:2]:
                        news_html += (
                            f'<span class="positive">'
                            f'UP {n["title"]}</span><br/>'
                        )
                    for n in top_neg[:2]:
                        news_html += (
                            f'<span class="negative">'
                            f'DOWN {n["title"]}</span><br/>'
                        )
                    news_html += '</div>'
    except _UI_RECOVERABLE_EXCEPTIONS as e:
        log.debug(f"News sentiment fetch: {e}")

    html = f"""
    <style>
        body {{
            color: {ModernColors.TEXT_PRIMARY};
            font-family: Consolas;
            background-color: transparent;
        }}
        .signal {{
            color: {color}; font-size: 18px; font-weight: bold;
        }}
        .section {{
            margin: 10px 0;
            background-color: transparent;
        }}
        .label {{ color: {ModernColors.TEXT_SECONDARY}; }}
        .positive {{ color: {ModernColors.ACCENT_SUCCESS}; }}
        .negative {{ color: {ModernColors.ACCENT_DANGER}; }}
        .neutral {{ color: {ModernColors.ACCENT_WARNING}; }}
    </style>

    <div class="section">
        <span class="label">Signal: </span>
        <span class="signal">{signal_text}</span>
        <span class="label">
            | Strength: {signal_strength:.0%}
        </span>
    </div>

    <div class="section">
        <span class="label">AI Prediction: </span>
        <span class="positive">UP {prob_up:.0%}</span> |
        <span class="neutral">NEUTRAL {prob_neutral:.0%}</span> |
        <span class="negative">DOWN {prob_down:.0%}</span>
    </div>

    <div class="section">
        <span class="label">Model Quality: </span>
        Confidence={confidence:.0%} |
        Agreement={agreement:.0%} |
        Entropy={entropy:.2f} |
        Margin={margin:.2f}
    </div>

    <div class="section">
        <span class="label">Uncertainty: </span>
        Score={uncertainty_score:.2f} |
        Tail Risk={tail_risk_score:.2f}
    </div>

    {news_html}

    <div class="section">
        <span class="label">Technical: </span>
        RSI={rsi:.0f} | MACD={macd_signal} | Trend={trend}
    </div>
    """

    if levels:
        entry = safe_get(levels, 'entry', 0)
        stop_loss = safe_get(levels, 'stop_loss', 0)
        stop_loss_pct = safe_get(levels, 'stop_loss_pct', 0)
        target_1 = safe_get(levels, 'target_1', 0)
        target_1_pct = safe_get(levels, 'target_1_pct', 0)
        target_2 = safe_get(levels, 'target_2', 0)
        target_2_pct = safe_get(levels, 'target_2_pct', 0)

        html += f"""
        <div class="section">
            <span class="label">Trading Plan:</span><br/>
            Entry: CNY {entry:.2f} |
            Stop: CNY {stop_loss:.2f} ({stop_loss_pct:+.1f}%)<br/>
            Target 1: CNY {target_1:.2f} ({target_1_pct:+.1f}%) |
            Target 2: CNY {target_2:.2f} ({target_2_pct:+.1f}%)
        </div>
        """

    low_band = list(getattr(pred, "predicted_prices_low", []) or [])
    high_band = list(getattr(pred, "predicted_prices_high", []) or [])
    if low_band and high_band and len(low_band) == len(high_band):
        try:
            lo_last = float(low_band[-1])
            hi_last = float(high_band[-1])
            spread_pct = (
                ((hi_last - lo_last) / max(float(getattr(pred, "current_price", 0.0) or 0.0), 1e-8))
                * 100.0
            )
            html += f"""
            <div class="section">
                <span class="label">Forecast Interval:</span>
                CNY {lo_last:.2f} to CNY {hi_last:.2f}
                ({spread_pct:.1f}% width at horizon)
            </div>
            """
        except _UI_RECOVERABLE_EXCEPTIONS as exc:
            log.debug("Suppressed exception in ui/app.py", exc_info=exc)

    if position:
        shares = safe_get(position, 'shares', 0)
        value = safe_get(position, 'value', 0)
        risk_amount = safe_get(position, 'risk_amount', 0)
        html += f"""
        <div class="section">
            <span class="label">Position:</span>
            {shares:,} shares | CNY {value:,.2f} |
            Risk: CNY {risk_amount:,.2f}
        </div>
        """

    if reasons:
        html += (
            '<div class="section">'
            '<span class="label">Analysis:</span><br/>'
        )
        for reason in reasons[:5]:
            html += f"- {reason}<br/>"
        html += "</div>"

    if warnings:
        html += (
            '<div class="section">'
            '<span class="negative">Warnings:</span><br/>'
        )
        for warning in warnings:
            html += f"- {warning}<br/>"
        html += "</div>"

    self.details_text.setHtml(html)

def _add_to_history(self, pred: Any) -> None:
    """Add prediction to history"""
    row = 0
    self.history_table.insertRow(row)

    timestamp = getattr(pred, 'timestamp', datetime.now())
    self.history_table.setItem(row, 0, QTableWidgetItem(
        timestamp.strftime("%H:%M:%S")
        if hasattr(timestamp, 'strftime') else "--"
    ))
    self.history_table.setItem(
        row, 1, QTableWidgetItem(getattr(pred, 'stock_code', '--'))
    )

    signal = getattr(pred, 'signal', None)
    signal_text = (
        signal.value if hasattr(signal, 'value') else str(signal)
    )
    signal_item = QTableWidgetItem(signal_text)
    signal_item.setForeground(QColor(ModernColors.ACCENT_INFO))
    self.history_table.setItem(row, 2, signal_item)

    prob_up = getattr(pred, 'prob_up', 0)
    self.history_table.setItem(
        row, 3, QTableWidgetItem(f"{prob_up:.0%}")
    )

    confidence = getattr(pred, 'confidence', 0)
    self.history_table.setItem(
        row, 4, QTableWidgetItem(f"{confidence:.0%}")
    )
    entry_price = float(getattr(pred, "current_price", 0.0) or 0.0)
    result_item = QTableWidgetItem("--")
    result_item.setData(
        Qt.ItemDataRole.UserRole,
        {
            "symbol": self._ui_norm(getattr(pred, "stock_code", "")),
            "entry_price": entry_price,
            "direction": self._signal_to_direction(signal_text),
            "mark_price": entry_price,
            "shares": self._guess_profit_notional_shares,
        },
    )
    self.history_table.setItem(row, 5, result_item)

    while self.history_table.rowCount() > 100:
        self.history_table.removeRow(
            self.history_table.rowCount() - 1
        )

def _signal_to_direction(self, signal_text: str) -> str:
    """Map prediction signal text to directional guess."""
    text = str(signal_text or "").upper()
    if "BUY" in text:
        return "UP"
    if "SELL" in text:
        return "DOWN"
    return "NONE"

# Performance and quality constants
_ANALYSIS_THROTTLE_SECONDS = 2.5  # Minimum gap between analysis requests
_SESSION_CACHE_MIN_GAP_SECONDS = 5.0  # Reduced frequency for session cache writes
_QUOTE_UPDATE_THROTTLE_MS = 150  # Throttle quote UI updates
_GUESS_PROFIT_NOTIONAL_VALUE = 10000.0  # CNY notional value per guess

# Transaction cost parameters for realistic P&L estimation
_TRANSACTION_COSTS = {
    "commission_rate": 0.0003,  # 0.03% broker commission
    "commission_min": 5.0,  # Minimum CNY 5 per trade
    "stamp_duty": 0.001,  # 0.1% stamp duty on sells (CN market)
    "transfer_fee": 0.00002,  # 0.002% transfer fee
    "slippage_bps": 2,  # 2 basis points slippage assumption
}

def _compute_guess_profit(
    self,
    direction: str,
    entry_price: float,
    mark_price: float,
    shares: int | None = None,
) -> float:
    """
    Compute virtual directional P&L with transaction costs.

    Args:
        direction: "UP", "DOWN", or "NONE"
        entry_price: Entry price for the guess
        mark_price: Current mark price
        shares: Number of shares (if None, calculated from notional value)

    Returns:
        Net P&L after transaction costs (positive = correct guess)

    FIX: Added transaction cost modeling for realistic estimates.
    """
    entry = float(entry_price or 0.0)
    mark = float(mark_price or 0.0)

    if entry <= 0 or mark <= 0:
        return 0.0

    # Calculate shares based on notional value if not provided
    if shares is None or shares <= 0:
        lot_size = int(getattr(CONFIG, "LOT_SIZE", 100) or 100)
        # Calculate shares to match notional value
        raw_shares = _GUESS_PROFIT_NOTIONAL_VALUE / entry
        shares = int(raw_shares / lot_size) * lot_size  # Round to lot size
        shares = max(lot_size, shares)  # Minimum 1 lot

    qty = max(1, shares)

    # Calculate gross P&L
    if direction == "UP":
        gross_pnl = (mark - entry) * qty
    elif direction == "DOWN":
        gross_pnl = (entry - mark) * qty
    else:
        return 0.0

    # Calculate transaction costs
    notional_entry = entry * qty
    notional_exit = mark * qty
    notional_total = notional_entry + notional_exit

    commission = max(
        notional_total * _TRANSACTION_COSTS["commission_rate"],
        _TRANSACTION_COSTS["commission_min"] * 2  # Entry + exit
    )
    stamp_duty = notional_exit * _TRANSACTION_COSTS["stamp_duty"]
    transfer_fee = notional_total * _TRANSACTION_COSTS["transfer_fee"]
    slippage = notional_total * (_TRANSACTION_COSTS["slippage_bps"] / 10000)

    total_costs = commission + stamp_duty + transfer_fee + slippage

    # Net P&L after costs
    net_pnl = gross_pnl - total_costs
    return net_pnl

def _refresh_guess_rows_for_symbol(self, code: str, price: float) -> None:
    """
    Update history result for this symbol using latest real-time price.

    FIX: Uses improved profit calculation with transaction costs.
    """
    symbol = self._ui_norm(code)
    mark_price = float(price or 0.0)
    if not symbol or mark_price <= 0:
        return

    # FIX: Batch update for performance
    self.history_table.setUpdatesEnabled(False)
    try:
        for row in range(self.history_table.rowCount()):
            code_item = self.history_table.item(row, 1)
            result_item = self.history_table.item(row, 5)
            if not code_item or not result_item:
                continue
            if self._ui_norm(code_item.text()) != symbol:
                continue

            meta = result_item.data(Qt.ItemDataRole.UserRole) or {}
            direction = str(meta.get("direction", "NONE"))
            entry = float(meta.get("entry_price", 0.0) or 0.0)
            shares = int(meta.get("shares", 0) or 0)  # Will use notional if 0
            pnl = self._compute_guess_profit(direction, entry, mark_price, shares)
            raw_ret_pct = ((mark_price / entry - 1.0) * 100.0) if entry > 0 else 0.0
            signed_ret_pct = (
                raw_ret_pct
                if direction == "UP"
                else (-raw_ret_pct if direction == "DOWN" else 0.0)
            )

            if direction == "NONE":
                result_item.setText("--")
                result_item.setForeground(QColor(ModernColors.TEXT_SECONDARY))
            elif pnl > 0:
                result_item.setText(
                    f"CORRECT CNY {pnl:+,.2f} ({signed_ret_pct:+.2f}%)"
                )
                result_item.setForeground(QColor(ModernColors.ACCENT_SUCCESS))
            elif pnl < 0:
                result_item.setText(
                    f"WRONG CNY {pnl:,.2f} ({signed_ret_pct:+.2f}%)"
                )
                result_item.setForeground(QColor(ModernColors.ACCENT_DANGER))
            else:
                result_item.setText("FLAT CNY 0.00 (+0.00%)")
                result_item.setForeground(QColor(ModernColors.TEXT_SECONDARY))

            meta["mark_price"] = mark_price
            result_item.setData(Qt.ItemDataRole.UserRole, meta)
    finally:
        self.history_table.setUpdatesEnabled(True)

    self._update_correct_guess_profit_ui()

def _calculate_realtime_correct_guess_profit(self) -> dict[str, float]:
    """
    Aggregate real-time guess quality across history rows.
    Reports both net and gross-correct directional P&L.
    """
    total = 0
    correct = 0
    wrong = 0
    correct_profit = 0.0
    wrong_loss = 0.0
    net_profit = 0.0

    for row in range(self.history_table.rowCount()):
        result_item = self.history_table.item(row, 5)
        if not result_item:
            continue
        meta = result_item.data(Qt.ItemDataRole.UserRole) or {}
        direction = str(meta.get("direction", "NONE"))
        if direction not in ("UP", "DOWN"):
            continue

        entry = float(meta.get("entry_price", 0.0) or 0.0)
        mark = float(meta.get("mark_price", 0.0) or 0.0)
        shares = int(
            meta.get("shares", self._guess_profit_notional_shares) or 1
        )
        pnl = self._compute_guess_profit(direction, entry, mark, shares)

        total += 1
        if pnl > 0:
            correct += 1
            correct_profit += pnl
        elif pnl < 0:
            wrong += 1
            wrong_loss += abs(pnl)
        net_profit += pnl

    return {
        "total": float(total),
        "correct": float(correct),
        "wrong": float(wrong),
        "correct_profit": float(correct_profit),
        "wrong_loss": float(wrong_loss),
        "net_profit": float(net_profit),
        "hit_rate": (float(correct) / float(total)) if total > 0 else 0.0,
    }

def _update_correct_guess_profit_ui(self) -> None:
    """Display real-time directional-guess P&L and hit rate in UI."""
    if not hasattr(self, "auto_trade_labels"):
        return

    stats = self._calculate_realtime_correct_guess_profit()

    label_profit = self.auto_trade_labels.get("guess_profit")
    if label_profit:
        net_val = float(stats.get("net_profit", 0.0) or 0.0)
        gross_correct = float(stats.get("correct_profit", 0.0) or 0.0)
        gross_wrong = float(stats.get("wrong_loss", 0.0) or 0.0)
        label_profit.setText(f"CNY {net_val:+,.2f}")
        color = (
            ModernColors.ACCENT_SUCCESS
            if net_val >= 0
            else ModernColors.ACCENT_DANGER
        )
        label_profit.setStyleSheet(
            (
                f"color: {color}; "
                f"font-size: {ModernFonts.SIZE_XL}px; "
                f"font-weight: {ModernFonts.WEIGHT_BOLD};"
            )
        )
        label_profit.setToolTip(
            "Directional guess P&L\n"
            f"Net: CNY {net_val:+,.2f}\n"
            f"Gross Correct: CNY {gross_correct:,.2f}\n"
            f"Gross Wrong: CNY {gross_wrong:,.2f}"
        )

    label_rate = self.auto_trade_labels.get("guess_rate")
    if label_rate:
        total = int(stats.get("total", 0.0) or 0)
        correct = int(stats.get("correct", 0.0) or 0)
        rate = float(stats.get("hit_rate", 0.0) or 0.0)
        label_rate.setText(f"{rate:.1%} ({correct}/{total})")
        label_rate.setStyleSheet(
            (
                f"color: {ModernColors.ACCENT_INFO}; "
                f"font-size: {ModernFonts.SIZE_XL}px; "
                f"font-weight: {ModernFonts.WEIGHT_BOLD};"
            )
        )

def _scan_stocks(self) -> None:
    """Scan all stocks for signals"""
    if self.predictor is None or self.predictor.ensemble is None:
        self.log("No model loaded", "error")
        return

    self.log("Scanning stocks for trading signals...", "info")
    self.progress.setRange(0, 0)
    self.progress.show()

    def scan():
        if hasattr(self.predictor, 'get_top_picks'):
            return self.predictor.get_top_picks(
                CONFIG.STOCK_POOL, n=10, signal_type="buy"
            )
        return []

    worker = WorkerThread(scan, timeout_seconds=180)
    self._track_worker(worker)
    worker.result.connect(self._on_scan_done)
    def _on_scan_error(e: str) -> None:
        self.log(f"Scan failed: {e}", "error")
        self.progress.hide()
        self.workers.pop("scan", None)

    worker.error.connect(_on_scan_error)
    self.workers['scan'] = worker
    worker.start()

def _on_scan_done(self, picks: list[Any]) -> None:
    """Handle scan completion"""
    self.progress.hide()

    if not picks:
        self.log("No strong buy signals found", "info")
        return

    self.log(f"Found {len(picks)} buy signals:", "success")

    for pred in picks:
        signal_text = (
            pred.signal.value
            if hasattr(pred.signal, 'value')
            else str(pred.signal)
        )
        conf = getattr(pred, 'confidence', 0)
        name = getattr(pred, 'stock_name', '')
        self.log(
            f"  {pred.stock_code} {name}: "
            f"{signal_text} (confidence: {conf:.0%})",
            "info"
        )

    if picks:
        self.stock_input.setText(picks[0].stock_code)
        self._analyze_stock()

    self.workers.pop('scan', None)
