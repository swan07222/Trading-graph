from __future__ import annotations

import math
import time
from datetime import datetime
from importlib import import_module
from typing import Any

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import QMessageBox, QTableWidgetItem

from config.settings import CONFIG
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
    """Quick trade from signal."""
    self.stock_input.setText(pred.stock_code)
    self._analyze_stock()

def _update_watchlist(self) -> None:
    """Update watchlist display."""
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

    # [DBG] Stock selection diagnostic
    self._debug_console(
        f"stock_selected:{code}",
        f"Stock selected: {code} (row={row}, col={col})",
        min_gap_seconds=0.5,
        level="info",
    )

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

    # [DBG] Clear old bars for fresh load
    existing_bars = list(self._bars_by_symbol.get(code) or [])
    self._debug_console(
        f"stock_selection_bars_cleared:{code}",
        f"Clearing {len(existing_bars)} existing bars for {code} before refresh",
        min_gap_seconds=0.5,
        level="info",
    )

    try:
        self._queue_history_refresh(
            code,
            self._normalize_interval_token(self.interval_combo.currentText()),
        )
    except _UI_RECOVERABLE_EXCEPTIONS as exc:
        log.debug("Suppressed exception in ui/app.py", exc_info=exc)

    # [DBG] History refresh queued diagnostic
    self._debug_console(
        f"stock_selection_history_queued:{code}",
        f"History refresh queued for {code}, calling _analyze_stock",
        min_gap_seconds=0.5,
        level="info",
    )

    self._analyze_stock()

def _add_to_watchlist(self) -> None:
    """Add stock to watchlist with validation."""
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
    else:
        self.log(f"{normalized} already in watchlist", "info")

def _remove_from_watchlist(self) -> None:
    """Remove selected stock from watchlist."""
    row = self.watchlist.currentRow()
    if row >= 0 and row < self.watchlist.rowCount():
        item = self.watchlist.item(row, 0)
        if item:
            code = item.text()
            if code in self.watch_list:
                self.watch_list.remove(code)
                self._update_watchlist()
                self.log(f"Removed {code} from watchlist", "info")

def _analyze_stock(self) -> None:
    """Analyze stock with validation."""
    code = self.stock_input.text().strip()
    if not code:
        self.log("Please enter a stock code", "warning")
        return

    normalized = self._ui_norm(code)
    if not normalized:
        self.log("Invalid stock code format", "warning")
        return

    # [DBG] Analyze start diagnostic
    self._debug_console(
        f"analyze_start:{normalized}",
        f"Starting analysis for {normalized}",
        min_gap_seconds=0.5,
        level="info",
    )

    forecast_ready = False
    try:
        forecast_ready_fn = getattr(self, "_predictor_forecast_ready", None)
        if callable(forecast_ready_fn):
            forecast_ready = bool(forecast_ready_fn())
        else:
            forecast_ready = bool(self._predictor_runtime_ready())
    except _UI_RECOVERABLE_EXCEPTIONS:
        forecast_ready = bool(self._predictor_runtime_ready())

    if not forecast_ready:
        if hasattr(self.signal_panel, "reset"):
            self.signal_panel.reset()
        selected = self._ui_norm(code)
        try:
            if (
                self.current_prediction
                and getattr(self.current_prediction, "stock_code", "") == selected
            ):
                self.current_prediction.predicted_prices = []
                self.current_prediction.predicted_prices_low = []
                self.current_prediction.predicted_prices_high = []
        except _UI_RECOVERABLE_EXCEPTIONS:
            pass
        try:
            interval_no_model = self._normalize_interval_token(
                self.interval_combo.currentText()
            )
            lookback_no_model = int(
                max(
                    int(self.lookback_spin.value()),
                    int(self._recommended_lookback(interval_no_model)),
                )
            )
            arr = list(
                self._load_chart_history_bars(
                    selected,
                    interval_no_model,
                    lookback_no_model,
                )
                or []
            )
            if arr:
                arr = self._filter_bars_to_market_session(arr, interval_no_model)
            if not arr:
                arr = list(self._bars_by_symbol.get(selected) or [])
            # [DBG] No model - chart state diagnostic
            self._debug_console(
                f"analyze_no_model:{selected}",
                (
                    "No model loaded, rendering chart with "
                    f"{len(arr)} bars for {selected}"
                ),
                min_gap_seconds=0.5,
                level="warning",
            )
            if arr:
                self._render_chart_state(
                    symbol=selected,
                    interval=interval_no_model,
                    bars=arr,
                    context="analyze_no_model",
                    predicted_prices=[],
                    source_interval=interval_no_model,
                    target_steps=int(self.forecast_spin.value()),
                    predicted_prepared=True,
                )
        except _UI_RECOVERABLE_EXCEPTIONS as exc:
            log.debug("Suppressed exception in ui/app.py", exc_info=exc)
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
    target_lookback = int(
        max(
            int(self.lookback_spin.value()),
            int(self._recommended_lookback(interval)),
        )
    )
    self.lookback_spin.setValue(target_lookback)
    self._queue_history_refresh(normalized, interval)
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
            skip_cache=skip_cache,  # FIX: Bypass prediction cache for fresh data
            history_allow_online=history_allow_online,  # FIX: Force online fetch
        )

    worker = WorkerThread(analyze, timeout_seconds=120)
    self._track_worker(worker)
    
    # FIX: Tag worker with sequence number for stale detection
    worker._request_seq = current_seq
    
    worker.result.connect(
        lambda pred, seq=current_seq: self._on_analysis_done(
            pred,
            request_seq=seq,
        )
    )
    worker.error.connect(
        lambda error, seq=current_seq: self._on_analysis_error(
            error,
            request_seq=seq,
        )
    )
    self.workers["analyze"] = worker
    worker.start()

def _on_analysis_done(self, pred: Any, request_seq: int | None = None) -> None:
    """Handle analysis completion; also triggers news fetch.

    FIX: Added stale result detection using sequence numbers.
    """
    current_seq = getattr(self, "_analyze_request_seq", None)
    if request_seq is None:
        worker = self.workers.get("analyze")
        if worker is not None:
            request_seq = getattr(worker, "_request_seq", None)
    if (
        request_seq is not None
        and current_seq is not None
        and int(request_seq) != int(current_seq)
    ):
        # Stale result from older request - discard without touching current worker.
        self._debug_console(
            f"analyze_stale:{self._ui_norm(self.stock_input.text())}",
            f"Discarded stale analysis result (seq {request_seq} != {current_seq})",
            min_gap_seconds=2.0,
            level="debug",
        )
        return

    self.analyze_action.setEnabled(True)
    if not bool(getattr(self, "_startup_loading_active", False)):
        self.progress.hide()
        self.status_label.setText("Ready")

    symbol = self._ui_norm(getattr(pred, "stock_code", ""))
    selected = self._ui_norm(self.stock_input.text())
    if selected and symbol and selected != symbol:
        # User switched symbol while worker was running; ignore stale result.
        worker = self.workers.get("analyze")
        if worker is not None:
            worker_seq = getattr(worker, "_request_seq", None)
            if (
                request_seq is not None
                and worker_seq is not None
                and int(worker_seq) != int(request_seq)
            ):
                return
        self.workers.pop("analyze", None)
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

    predicted_prices = list(getattr(pred, "predicted_prices", []) or [])

    chart_rendered = False
    if symbol:
        arr = self._load_chart_history_bars(symbol, interval, lookback)
        arr = self._filter_bars_to_market_session(arr, interval)

        if arr:
            try:
                arr = self._render_chart_state(
                    symbol=symbol,
                    interval=interval,
                    bars=arr,
                    context="analysis_done",
                    current_price=current_price if current_price > 0 else None,
                    predicted_prices=predicted_prices,
                    source_interval=self._normalize_interval_token(
                        getattr(pred, "interval", interval),
                        fallback=interval,
                    ),
                    target_steps=int(self.forecast_spin.value()),
                    update_latest_label=True,
                    reset_view_on_symbol_switch=True,
                )
                chart_rendered = True
            except _UI_RECOVERABLE_EXCEPTIONS as e:
                log.debug(f"Chart update failed: {e}")
        if not chart_rendered:
            fallback_prices: list[float] = []
            for px_raw in list(getattr(pred, "price_history", []) or []):
                try:
                    px = float(px_raw)
                except _UI_RECOVERABLE_EXCEPTIONS:
                    continue
                if px > 0 and math.isfinite(px):
                    fallback_prices.append(px)
            if (
                not fallback_prices
                and current_price > 0
                and math.isfinite(current_price)
            ):
                fallback_prices = [float(current_price)]

            if fallback_prices and hasattr(self, "chart") and hasattr(self.chart, "update_data"):
                max_points = int(
                    max(
                        40,
                        min(
                            600,
                            int(self.lookback_spin.value()) if hasattr(self, "lookback_spin") else 180,
                        ),
                    )
                )
                fallback_prices = fallback_prices[-max_points:]
                anchor_price = float(fallback_prices[-1]) if fallback_prices else 0.0
                display_predicted = list(predicted_prices)
                try:
                    display_predicted = self._prepare_chart_predicted_prices(
                        symbol=symbol,
                        chart_interval=interval,
                        predicted_prices=predicted_prices,
                        source_interval=self._normalize_interval_token(
                            getattr(pred, "interval", interval),
                            fallback=interval,
                        ),
                        current_price=anchor_price if anchor_price > 0 else None,
                        target_steps=int(self.forecast_spin.value()),
                    )
                except _UI_RECOVERABLE_EXCEPTIONS as exc:
                    log.debug("Fallback chart prediction shaping failed: %s", exc)

                low_band: list[float] = []
                high_band: list[float] = []
                try:
                    low_band, high_band = self._build_chart_prediction_bands(
                        symbol=symbol,
                        predicted_prices=display_predicted,
                        anchor_price=anchor_price if anchor_price > 0 else None,
                        chart_interval=interval,
                    )
                except _UI_RECOVERABLE_EXCEPTIONS as exc:
                    log.debug("Fallback chart uncertainty bands failed: %s", exc)

                try:
                    self.chart.update_data(
                        fallback_prices,
                        predicted_prices=display_predicted,
                        predicted_prices_low=low_band,
                        predicted_prices_high=high_band,
                        levels=self._get_levels_dict(),
                    )
                    fallback_iv = self._normalize_interval_token(interval)
                    self._bars_by_symbol[symbol] = [
                        {
                            "open": float(px),
                            "high": float(px),
                            "low": float(px),
                            "close": float(px),
                            "interval": fallback_iv,
                        }
                        for px in fallback_prices
                    ]
                    self._update_chart_latest_label(
                        symbol,
                        bar=None,
                        price=(current_price if current_price > 0 else anchor_price),
                    )
                    chart_rendered = True
                    self._debug_console(
                        f"analysis_chart_fallback:{symbol}:{interval}",
                        (
                            f"analysis chart fallback for {symbol}: "
                            f"history unavailable, using {len(fallback_prices)} close points"
                        ),
                        min_gap_seconds=1.0,
                        level="warning",
                    )
                except _UI_RECOVERABLE_EXCEPTIONS as exc:
                    log.debug("Fallback chart render failed for %s: %s", symbol, exc)

    # Update details (with news sentiment)
    self._update_details(pred)
    if symbol and hasattr(self, "_refresh_news_policy_signal"):
        try:
            self._refresh_news_policy_signal(symbol, force=False)
        except _UI_RECOVERABLE_EXCEPTIONS:
            pass

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

    signal_text = (
        pred.signal.value
        if hasattr(pred.signal, 'value')
        else str(pred.signal)
    )
    conf = getattr(pred, 'confidence', 0)
    warnings = list(getattr(pred, "warnings", []) or [])
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
        else:
            self.log(
                f"Analysis complete: {pred.stock_code} - "
                f"{signal_text} ({conf:.0%})",
                "success",
            )
        self._last_analysis_log = {"key": log_key, "ts": now_log_ts}

    worker = self.workers.get("analyze")
    if worker is not None:
        worker_seq = getattr(worker, "_request_seq", None)
        if (
            request_seq is not None
            and worker_seq is not None
            and int(worker_seq) != int(request_seq)
        ):
            return
    self.workers.pop("analyze", None)

def _on_analysis_error(self, error: str, request_seq: int | None = None) -> None:
    """Handle analysis error."""
    current_seq = getattr(self, "_analyze_request_seq", None)
    if (
        request_seq is not None
        and current_seq is not None
        and int(request_seq) != int(current_seq)
    ):
        self._debug_console(
            f"analyze_stale_err:{self._ui_norm(self.stock_input.text())}",
            f"Ignored stale analysis error (seq {request_seq} != {current_seq})",
            min_gap_seconds=2.0,
            level="debug",
        )
        return

    self.analyze_action.setEnabled(True)
    if not bool(getattr(self, "_startup_loading_active", False)):
        self.progress.hide()
        self.status_label.setText("Ready")

    self.log(f"Analysis failed: {error}", "error")
    QMessageBox.warning(self, "Error", f"Analysis failed:\n{error}")

    worker = self.workers.get("analyze")
    if worker is not None:
        worker_seq = getattr(worker, "_request_seq", None)
        if (
            request_seq is not None
            and worker_seq is not None
            and int(worker_seq) != int(request_seq)
        ):
            return
    self.workers.pop("analyze", None)

def _update_details(self, pred: Any) -> None:
    """Update analysis details with news sentiment."""
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
    if signal == Signal.HOLD:
        try:
            signal_strength = max(0.0, min(1.0, abs(float(prob_up) - float(prob_down))))
        except _UI_RECOVERABLE_EXCEPTIONS as exc:
            log.debug("Suppressed exception in ui/app.py", exc_info=exc)
            signal_strength = 0.0
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
    forecast_vals = list(getattr(pred, "predicted_prices", []) or [])
    data_not_ready = bool(
        (
            safe_get(pred, "current_price", 0) <= 0
            or any(
                ("insufficient data" in str(w).lower())
                or ("prediction error" in str(w).lower())
                for w in list(warnings or [])
            )
        )
        and not forecast_vals
    )

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

    prediction_html = (
        '<span class="label">AI Prediction: warming up (not enough valid bars yet)</span>'
        if data_not_ready
        else (
            f'<span class="label">AI Prediction: </span>'
            f'<span class="positive">UP {prob_up:.0%}</span> | '
            f'<span class="neutral">NEUTRAL {prob_neutral:.0%}</span> | '
            f'<span class="negative">DOWN {prob_down:.0%}</span>'
        )
    )
    quality_html = (
        '<span class="label">Model Quality: Confidence=N/A | Agreement=N/A | Entropy=N/A | Margin=N/A</span>'
        if data_not_ready
        else (
            f'<span class="label">Model Quality: </span>'
            f'Confidence={confidence:.0%} | '
            f'Agreement={agreement:.0%} | '
            f'Entropy={entropy:.2f} | '
            f'Margin={margin:.2f}'
        )
    )
    uncertainty_html = (
        '<span class="label">Uncertainty: Score=N/A | Tail Risk=N/A</span>'
        if data_not_ready
        else (
            f'<span class="label">Uncertainty: </span>'
            f'Score={uncertainty_score:.2f} | '
            f'Tail Risk={tail_risk_score:.2f}'
        )
    )

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
        {prediction_html}
    </div>

    <div class="section">
        {quality_html}
    </div>

    <div class="section">
        {uncertainty_html}
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
    """Add prediction to history.
    
    FIX: Validate entry_price before storing to prevent invalid guess calculations.
    FIX: Store shares as None instead of 0 for clearer semantics.
    FIX: Add null checks for table operations.
    """
    # Validate table exists
    if not hasattr(self, "history_table") or self.history_table is None:
        return
    
    row = 0
    self.history_table.insertRow(row)

    timestamp = getattr(pred, 'timestamp', datetime.now())
    self.history_table.setItem(row, 0, QTableWidgetItem(
        timestamp.strftime("%H:%M:%S")
        if hasattr(timestamp, 'strftime') else "--"
    ))
    
    stock_code = getattr(pred, 'stock_code', '--')
    self.history_table.setItem(
        row, 1, QTableWidgetItem(stock_code if stock_code else '--')
    )

    signal = getattr(pred, 'signal', None)
    signal_text = (
        signal.value if hasattr(signal, 'value') else str(signal)
    ) if signal is not None else "NONE"
    
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
    
    # FIX: Validate entry_price before storing
    entry_price_raw = getattr(pred, "current_price", 0.0)
    try:
        entry_price = float(entry_price_raw or 0.0)
        if not math.isfinite(entry_price) or entry_price <= 0:
            entry_price = 0.0
    except (TypeError, ValueError):
        entry_price = 0.0
    
    result_item = QTableWidgetItem("--")
    
    # FIX: Ensure direction is properly computed with validated signal_text
    direction = self._signal_to_direction(signal_text if signal_text else "NONE")
    
    # FIX: Store shares as None for clearer semantics (triggers notional calculation)
    result_item.setData(
        Qt.ItemDataRole.UserRole,
        {
            "symbol": self._ui_norm(getattr(pred, "stock_code", "") or ""),
            "entry_price": entry_price,
            "direction": direction,
            "mark_price": entry_price,
            "shares": None,  # None means auto-size by notional value
        },
    )
    self.history_table.setItem(row, 5, result_item)

    # Clean up old rows with proper validation
    try:
        while self.history_table.rowCount() > 100:
            self.history_table.removeRow(
                self.history_table.rowCount() - 1
            )
    except (AttributeError, RuntimeError):
        pass

def _signal_to_direction(self, signal_text: str) -> str:
    """Map prediction signal text to directional guess.
    
    FIX: Handle None and empty string explicitly.
    FIX: Improve case-insensitive matching for various signal formats.
    FIX: Add support for additional signal variations (e.g., 'Strong Buy', 'Sell').
    """
    # Handle None or empty input
    if signal_text is None:
        return "NONE"
    
    text = str(signal_text).strip()
    if not text:
        return "NONE"
    
    # Case-insensitive matching
    text_upper = text.upper()
    
    # Check for BUY signals (includes "BUY", "STRONG_BUY", "STRONG BUY", etc.)
    if "BUY" in text_upper:
        return "UP"
    
    # Check for SELL signals (includes "SELL", "STRONG_SELL", "STRONG SELL", etc.)
    if "SELL" in text_upper:
        return "DOWN"
    
    # Default to NONE for HOLD or unknown signals
    return "NONE"

# Performance and quality constants
_ANALYSIS_THROTTLE_SECONDS = 2.5  # Minimum gap between analysis requests
_SESSION_CACHE_MIN_GAP_SECONDS = 5.0  # Reduced frequency for session cache writes
_QUOTE_UPDATE_THROTTLE_MS = 150  # Throttle quote UI updates
_GUESS_PROFIT_NOTIONAL_VALUE = 10000.0  # CNY notional value per guess (10,000 CNY)

# Transaction cost parameters for realistic P&L estimation (China A-share market)
_TRANSACTION_COSTS = {
    "commission_rate": 0.00025,  # 0.025% broker commission (typical CN rate)
    "commission_min": 5.0,  # Minimum CNY 5 per trade
    "stamp_duty": 0.0005,  # 0.05% stamp duty on sells only (reduced in 2023)
    "transfer_fee": 0.00002,  # 0.002% transfer fee
    "slippage_bps": 3,  # 3 basis points slippage assumption (conservative)
}

def _compute_guess_profit(
    self,
    direction: str,
    entry_price: float,
    mark_price: float,
    shares: int | None = None,
) -> float:
    """Compute virtual directional P&L with transaction costs.

    Args:
        direction: "UP", "DOWN", or "NONE"
        entry_price: Entry price for the guess
        mark_price: Current mark price
        shares: Number of shares (if None, calculated from notional value)

    Returns:
        Net P&L after transaction costs (positive = correct guess)

    FIX: Realistic China A-share transaction cost modeling.
    FIX Bug #6: Handle edge cases for shares calculation and invalid prices.
    FIX: Add validation for extreme prices and overflow protection.
    """
    # Validate and convert inputs
    try:
        entry = float(entry_price or 0.0)
        mark = float(mark_price or 0.0)
    except (TypeError, ValueError):
        return 0.0

    # Validate prices are positive and finite
    if entry <= 0 or mark <= 0:
        return 0.0
    if not (math.isfinite(entry) and math.isfinite(mark)):
        return 0.0

    # Validate price ratio to prevent overflow in shares calculation
    # If entry is too small, shares would be astronomically large
    _MIN_ENTRY_PRICE = 0.01  # Minimum valid entry price (CNY)
    if entry < _MIN_ENTRY_PRICE:
        return 0.0

    # Calculate shares based on notional value if not provided.
    if shares is None or int(shares) <= 0:
        try:
            lot_size = int(getattr(CONFIG, "LOT_SIZE", 100) or 100)
            lot_size = max(1, lot_size)  # Ensure lot_size >= 1
            
            # Calculate shares to match notional value (10,000 CNY)
            raw_shares = _GUESS_PROFIT_NOTIONAL_VALUE / entry
            
            # Validate raw_shares is finite and reasonable
            if not math.isfinite(raw_shares) or raw_shares <= 0:
                return 0.0
            
            # Cap raw_shares to prevent integer overflow
            _MAX_SHARES = 10_000_000  # Maximum 10 million shares
            if raw_shares > _MAX_SHARES:
                raw_shares = _MAX_SHARES
            
            shares = int(raw_shares / lot_size) * lot_size  # Round to lot size
            shares = max(lot_size, shares)  # Minimum 1 lot
        except (TypeError, ValueError, ZeroDivisionError, OverflowError):
            return 0.0

    # Validate and convert shares to quantity
    try:
        qty = int(shares)
        if qty <= 0 or qty > _MAX_SHARES:
            return 0.0
        if not math.isfinite(qty):
            return 0.0
    except (TypeError, ValueError, OverflowError):
        return 0.0

    # Calculate gross P&L
    if direction == "UP":
        gross_pnl = (mark - entry) * qty
        notional_buy = entry * qty
        notional_sell = mark * qty
    elif direction == "DOWN":
        gross_pnl = (entry - mark) * qty
        # Synthetic short guess: sell at entry, buy back at mark.
        notional_sell = entry * qty
        notional_buy = mark * qty
    else:  # NONE or invalid
        return 0.0

    # Validate notionals to prevent overflow in cost calculation
    if not (math.isfinite(notional_buy) and math.isfinite(notional_sell)):
        return 0.0
    if notional_buy <= 0 or notional_sell <= 0:
        return 0.0

    # Calculate transaction costs (China A-share market)
    # Commission: charged on both entry and exit (min CNY 5 each)
    try:
        commission_entry = max(
            notional_buy * _TRANSACTION_COSTS["commission_rate"],
            _TRANSACTION_COSTS["commission_min"]
        )
        commission_exit = max(
            notional_sell * _TRANSACTION_COSTS["commission_rate"],
            _TRANSACTION_COSTS["commission_min"]
        )
        commission = commission_entry + commission_exit

        # Stamp duty: charged on sell side only.
        stamp_duty = notional_sell * _TRANSACTION_COSTS["stamp_duty"]

        # Transfer fee: charged on both sides
        turnover = notional_buy + notional_sell
        transfer_fee = turnover * _TRANSACTION_COSTS["transfer_fee"]

        # Slippage: estimated market impact
        slippage = turnover * (_TRANSACTION_COSTS["slippage_bps"] / 10000)

        total_costs = commission + stamp_duty + transfer_fee + slippage

        # Validate total_costs is finite
        if not math.isfinite(total_costs):
            return 0.0

        # Net P&L after costs
        net_pnl = gross_pnl - total_costs

        # Final validation: return 0 if result is not finite
        if not math.isfinite(net_pnl):
            return 0.0

        return net_pnl
    except (TypeError, ValueError, KeyError, ZeroDivisionError, OverflowError):
        return 0.0

def _refresh_guess_rows_for_symbol(self, code: str, price: float) -> None:
    """Update history result for this symbol using latest real-time price.

    FIX: Uses improved profit calculation with transaction costs.
    FIX Bug #5: Add proper error handling to prevent race conditions and UI crashes.
    FIX: Add proper null checks and validate table state before batch updates.
    FIX: Prevent division by zero in return percentage calculation.
    """
    try:
        symbol = self._ui_norm(code)
        mark_price = float(price or 0.0)
        if not symbol or mark_price <= 0:
            return

        # Validate table exists and is valid before batch update
        if not hasattr(self, "history_table") or self.history_table is None:
            return
        if not hasattr(self.history_table, "setUpdatesEnabled"):
            return

        # FIX: Batch update for performance with proper state validation
        updates_enabled = self.history_table.updatesEnabled()
        self.history_table.setUpdatesEnabled(False)
        try:
            row_count = self.history_table.rowCount()
            for row in range(row_count):
                try:
                    code_item = self.history_table.item(row, 1)
                    result_item = self.history_table.item(row, 5)
                    if code_item is None or result_item is None:
                        continue
                    
                    code_text = code_item.text()
                    if code_text is None:
                        continue
                    
                    if self._ui_norm(code_text) != symbol:
                        continue

                    # Safely get metadata with null checks
                    meta = result_item.data(Qt.ItemDataRole.UserRole)
                    if meta is None:
                        continue
                    if not isinstance(meta, dict):
                        continue
                    
                    direction = str(meta.get("direction", "NONE") or "NONE")
                    entry_raw = meta.get("entry_price", 0.0)
                    entry = float(entry_raw or 0.0)
                    shares_raw = meta.get("shares", 0)
                    shares = int(shares_raw or 0)  # Will use notional if 0
                    
                    pnl = self._compute_guess_profit(direction, entry, mark_price, shares)
                    
                    # FIX: Safe return percentage calculation with division by zero protection
                    if entry > 0:
                        raw_ret_pct = (mark_price / entry - 1.0) * 100.0
                    else:
                        raw_ret_pct = 0.0
                    
                    if direction == "UP":
                        signed_ret_pct = raw_ret_pct
                    elif direction == "DOWN":
                        signed_ret_pct = -raw_ret_pct
                    else:
                        signed_ret_pct = 0.0

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
                except (TypeError, ValueError, AttributeError, RuntimeError):
                    # Skip problematic rows without crashing
                    continue
        finally:
            # Always restore update state even if error occurs
            try:
                self.history_table.setUpdatesEnabled(updates_enabled)
            except (AttributeError, RuntimeError):
                pass

        self._update_correct_guess_profit_ui()
    except (TypeError, ValueError, AttributeError, RuntimeError) as e:
        # FIX: Catch UI thread errors and log without crashing
        log.debug(f"_refresh_guess_rows_for_symbol failed: {e}")

def _calculate_realtime_correct_guess_profit(self) -> dict[str, float]:
    """Aggregate real-time guess quality across history rows.
    Reports both net and gross-correct directional P&L.
    
    FIX: Add proper null checks and validate table state.
    FIX: Ensure consistent shares handling with _compute_guess_profit.
    FIX: Validate mark_price to prevent incorrect P&L calculations.
    """
    # Validate table exists
    if not hasattr(self, "history_table") or self.history_table is None:
        return {
            "total": 0.0,
            "correct": 0.0,
            "wrong": 0.0,
            "correct_profit": 0.0,
            "wrong_loss": 0.0,
            "net_profit": 0.0,
            "hit_rate": 0.0,
        }
    
    total = 0
    correct = 0
    wrong = 0
    correct_profit = 0.0
    wrong_loss = 0.0
    net_profit = 0.0

    row_count = self.history_table.rowCount()
    for row in range(row_count):
        try:
            result_item = self.history_table.item(row, 5)
            if result_item is None:
                continue
            
            meta = result_item.data(Qt.ItemDataRole.UserRole)
            if meta is None or not isinstance(meta, dict):
                continue
            
            direction = str(meta.get("direction", "NONE") or "NONE")
            if direction not in ("UP", "DOWN"):
                continue

            entry_raw = meta.get("entry_price", 0.0)
            entry = float(entry_raw or 0.0)
            
            mark_raw = meta.get("mark_price", 0.0)
            mark = float(mark_raw or 0.0)
            
            # FIX: Skip if mark_price is invalid (not yet updated)
            if mark <= 0:
                continue
            
            # FIX: Consistent shares handling - pass 0 to trigger notional calculation
            shares_raw = meta.get("shares", 0)
            if shares_raw is None or (isinstance(shares_raw, int) and shares_raw <= 0):
                shares = None  # Let _compute_guess_profit calculate from notional
            else:
                try:
                    shares = int(shares_raw)
                    if shares <= 0:
                        shares = None
                except (TypeError, ValueError):
                    shares = None
            
            pnl = self._compute_guess_profit(direction, entry, mark, shares)

            total += 1
            if pnl > 0:
                correct += 1
                correct_profit += pnl
            elif pnl < 0:
                wrong += 1
                wrong_loss += abs(pnl)
            net_profit += pnl
        except (TypeError, ValueError, AttributeError, RuntimeError):
            # Skip problematic rows
            continue

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
        total = int(stats.get("total", 0) or 0)
        correct = int(stats.get("correct", 0) or 0)
        wrong = int(stats.get("wrong", 0) or 0)
        hit_rate = float(stats.get("hit_rate", 0.0) or 0.0)
        
        # Display net P&L
        label_profit.setText(f"CNY {net_val:+,.2f}")
        color = (
            ModernColors.ACCENT_SUCCESS
            if net_val >= 0
            else ModernColors.ACCENT_DANGER
        )
        label_profit.setStyleSheet(
            f"color: {color}; "
            f"font-size: {ModernFonts.SIZE_XL}px; "
            f"font-weight: {ModernFonts.WEIGHT_BOLD};"
        )
        
        # Enhanced tooltip with detailed statistics
        avg_win = gross_correct / correct if correct > 0 else 0.0
        avg_loss = gross_wrong / wrong if wrong > 0 else 0.0
        profit_factor = gross_correct / gross_wrong if gross_wrong > 0 else 999.99
        
        tooltip = (
            f"{'='*40}\n"
            f"DIRECTIONAL GUESS PERFORMANCE\n"
            f"{'='*40}\n\n"
            f"Net P&L:        CNY {net_val:+,.2f}\n"
            f"Gross Correct:  CNY {gross_correct:,.2f}\n"
            f"Gross Wrong:    CNY {gross_wrong:,.2f}\n\n"
            f"{'='*40}\n"
            f"STATISTICS\n"
            f"{'='*40}\n\n"
            f"Total Guesses:  {total}\n"
            f"Correct:        {correct} ({hit_rate:.1%})\n"
            f"Wrong:          {wrong} ({1-hit_rate:.1%})\n\n"
            f"Avg Win:        CNY {avg_win:,.2f}\n"
            f"Avg Loss:       CNY {avg_loss:,.2f}\n"
            f"Win/Loss Ratio: {profit_factor:.2f}"
        )
        label_profit.setToolTip(tooltip)

    label_rate = self.auto_trade_labels.get("guess_rate")
    if label_rate:
        total = int(stats.get("total", 0.0) or 0)
        correct = int(stats.get("correct", 0.0) or 0)
        rate = float(stats.get("hit_rate", 0.0) or 0.0)
        label_rate.setText(f"{rate:.1%} ({correct}/{total})")
        label_rate.setStyleSheet(
            
                f"color: {ModernColors.ACCENT_INFO}; "
                f"font-size: {ModernFonts.SIZE_XL}px; "
                f"font-weight: {ModernFonts.WEIGHT_BOLD};"
            
        )

def _init_screener_profile_ui(self) -> None:
    """Initialize screener profile selector in toolbar."""
    combo = getattr(self, "screener_profile_combo", None)
    if combo is None:
        return

    profiles: list[str] = ["balanced"]
    active = "balanced"
    try:
        from analysis.screener import (
            get_active_screener_profile_name,
            list_screener_profiles,
        )

        names = [str(x).strip().lower() for x in list_screener_profiles()]
        names = [x for x in names if x]
        if names:
            profiles = sorted(dict.fromkeys(names))
        active_raw = str(get_active_screener_profile_name()).strip().lower()
        if active_raw:
            active = active_raw
    except Exception as e:
        log.debug("Screener profile list unavailable: %s", e)

    if active not in profiles:
        profiles = sorted(dict.fromkeys(profiles + [active]))

    self._syncing_screener_profile_ui = True
    try:
        combo.blockSignals(True)
        combo.clear()
        combo.addItems(profiles)
        combo.setCurrentText(active)
        combo.setToolTip("Scan profile used by 'Scan Market'")
    finally:
        combo.blockSignals(False)
        self._syncing_screener_profile_ui = False

    self._active_screener_profile = active

def _on_screener_profile_changed(self, profile_name: str) -> None:
    """Handle toolbar profile selector change."""
    if bool(getattr(self, "_syncing_screener_profile_ui", False)):
        return

    name = str(profile_name or "").strip().lower()
    if not name:
        return

    try:
        from analysis.screener import (
            build_default_screener,
            set_active_screener_profile,
        )

        ok = bool(set_active_screener_profile(name))
        # Force refresh singleton so subsequent scans use new thresholds.
        build_default_screener(name, force_reload=True)
        self._active_screener_profile = name
        if ok:
            self.log(f"Screener profile set to {name}", "info")
        else:
            self.log(f"Failed to persist screener profile {name}", "warning")
    except Exception as e:
        self.log(f"Failed to switch screener profile: {e}", "warning")
        log.debug("Screener profile change failed: %s", e)

def _show_screener_profile_dialog(self) -> None:
    """Open screener profile editor dialog."""
    try:
        ScreenerProfileDialog = _lazy_get(
            "ui.dialogs",
            "ScreenerProfileDialog",
        )
        dialog = ScreenerProfileDialog(self)
        accepted = int(dialog.exec())
        if accepted:
            self._init_screener_profile_ui()
            selected = str(
                getattr(dialog, "selected_profile_name", "")
            ).strip().lower()
            if selected and getattr(self, "screener_profile_combo", None) is not None:
                self._syncing_screener_profile_ui = True
                try:
                    self.screener_profile_combo.setCurrentText(selected)
                finally:
                    self._syncing_screener_profile_ui = False
                self._on_screener_profile_changed(selected)
    except Exception as e:
        QMessageBox.warning(
            self,
            "Profile Manager",
            f"Failed to open screener profile manager:\n{e}",
        )

def _scan_stocks(self) -> None:
    """Scan all stocks for signals."""
    if not self._predictor_runtime_ready():
        self.log("No model loaded", "error")
        return

    profile_name = "balanced"
    try:
        from analysis.screener import get_active_screener_profile_name

        profile_name = str(get_active_screener_profile_name() or profile_name)
    except (ImportError, AttributeError, TypeError, ValueError):
        profile_name = "balanced"

    self.log(
        f"Scanning stocks for trading signals (profile: {profile_name})...",
        "info",
    )
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
    """Handle scan completion."""
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
        rank_score = float(getattr(pred, "rank_score", conf) or conf)
        fscore = float(getattr(pred, "fundamental_score", 0.5) or 0.5)
        name = getattr(pred, 'stock_name', '')
        self.log(
            f"  {pred.stock_code} {name}: "
            f"{signal_text} (confidence: {conf:.0%}, "
            f"rank: {rank_score:.0%}, fundamentals: {fscore:.0%})",
            "info"
        )

    if picks:
        self.stock_input.setText(picks[0].stock_code)
        self._analyze_stock()

    self.workers.pop('scan', None)
