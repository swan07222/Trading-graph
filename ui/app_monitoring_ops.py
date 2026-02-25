from __future__ import annotations

from importlib import import_module
from typing import Any

from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import QApplication, QPushButton, QTableWidgetItem

from config.settings import CONFIG
from ui.background_tasks import RealTimeMonitor
from ui.modern_theme import ModernColors, ModernFonts
from utils.logger import get_logger

log = get_logger(__name__)


def _lazy_get(module: str, name: str) -> Any:
    return getattr(import_module(module), name)


def _toggle_monitoring(self: Any, checked: bool) -> None:
    """Toggle real-time monitoring."""
    if checked:
        self._start_monitoring()
    else:
        self._stop_monitoring()


def _start_monitoring(self: Any) -> None:
    """Start real-time monitoring safely (no orphan threads)."""
    if self.monitor and self.monitor.isRunning():
        self._stop_monitoring()

    if not self._predictor_runtime_ready():
        self.log("Cannot start monitoring: No model loaded", "error")
        self.monitor_action.setChecked(False)
        return

    requested_interval = self._normalize_interval_token(self.interval_combo.currentText())
    requested_horizon = int(self.forecast_spin.value())
    lookback = max(
        int(self.lookback_spin.value()),
        int(self._recommended_lookback(requested_interval)),
    )
    monitor_interval = "1m"
    monitor_horizon = int(requested_horizon)
    monitor_lookback = int(
        max(
            self._recommended_lookback("1m"),
            self._bars_needed_from_base_interval(
                requested_interval,
                int(lookback),
                base_interval="1m",
            ),
        )
    )
    monitor_history_allow_online = True
    if not self._has_exact_model_artifacts(monitor_interval, requested_horizon):
        self._debug_console(
            f"monitor_model_fallback:{requested_interval}:{requested_horizon}",
            (
                "monitor inference locked to 1m source stream: "
                f"ui={requested_interval}/{requested_horizon} "
                f"infer={monitor_interval}/{requested_horizon} "
                f"lookback={monitor_lookback} online=1"
            ),
            min_gap_seconds=2.0,
            level="info",
        )

    try:
        from data.feeds import get_feed_manager

        fm = get_feed_manager(auto_init=True, async_init=True)
        fm.subscribe_many(self.watch_list)
        try:
            code = self.stock_input.text().strip()
            if code:
                normalized = self._ui_norm(code)
                if normalized:
                    self._ensure_feed_subscription(normalized)
        except Exception as exc:
            log.debug("Suppressed exception in ui/app.py", exc_info=exc)
        self.log(
            f"Subscribed to feeds for {len(self.watch_list)} stocks",
            "info",
        )
    except Exception as exc:
        self.log(f"Feed subscription warning: {exc}", "warning")

    self.monitor = RealTimeMonitor(
        self.predictor,
        self.watch_list,
        interval=monitor_interval,
        forecast_minutes=monitor_horizon,
        lookback_bars=monitor_lookback,
        history_allow_online=monitor_history_allow_online,
    )
    self.monitor.signal_detected.connect(self._on_signal_detected)
    self.monitor.price_updated.connect(self._on_price_updated)
    self.monitor.error_occurred.connect(
        lambda err: self.log(f"Monitor: {err}", "warning")
    )
    self.monitor.status_changed.connect(
        lambda status: self.monitor_label.setText(f"Monitoring: {status}")
    )
    self.monitor.start()

    self.monitor_label.setText("Monitoring: ACTIVE")
    self.monitor_label.setStyleSheet(
        
            f"color: {ModernColors.ACCENT_SUCCESS}; "
            f"font-weight: {ModernFonts.WEIGHT_BOLD};"
        
    )
    self.monitor_action.setText("Stop Monitoring")

    if monitor_interval != requested_interval or int(monitor_horizon) != int(
        requested_horizon
    ):
        self.log(
            (
                f"Monitoring started: {requested_interval} interval, "
                f"{requested_horizon} bar forecast "
                f"(compute={monitor_interval}/{monitor_horizon}, cache-first)"
            ),
            "success",
        )
    else:
        self.log(
            f"Monitoring started: {requested_interval} interval, "
            f"{requested_horizon} bar forecast",
            "success",
        )


def _stop_monitoring(self: Any) -> None:
    """Stop real-time monitoring."""
    if self.monitor:
        try:
            self.monitor.stop()
            self.monitor.wait(3000)
        except Exception as exc:
            log.debug("Suppressed exception in ui/app.py", exc_info=exc)
        finally:
            self.monitor = None

    self.monitor_label.setText("Monitoring: OFF")
    self.monitor_label.setStyleSheet(
        f"color: {ModernColors.TEXT_MUTED};"
    )
    self.monitor_action.setText("Start Monitoring")
    self.monitor_action.setChecked(False)

    self.log("Real-time monitoring stopped", "info")


def _on_signal_detected(self: Any, pred: Any) -> None:
    """Handle detected trading signal."""
    signal_type = _lazy_get("models.predictor", "Signal")

    row = 0
    self.signals_table.insertRow(row)

    self.signals_table.setItem(
        row,
        0,
        QTableWidgetItem(
            pred.timestamp.strftime("%H:%M:%S")
            if hasattr(pred, "timestamp")
            else "--"
        ),
    )

    stock_text = f"{pred.stock_code}"
    if hasattr(pred, "stock_name") and pred.stock_name:
        stock_text += f" - {pred.stock_name}"
    self.signals_table.setItem(row, 1, QTableWidgetItem(stock_text))

    signal_text = (
        pred.signal.value
        if hasattr(pred.signal, "value")
        else str(pred.signal)
    )
    signal_item = QTableWidgetItem(signal_text)

    if hasattr(pred, "signal") and pred.signal in [signal_type.STRONG_BUY, signal_type.BUY]:
        signal_item.setForeground(QColor(ModernColors.ACCENT_SUCCESS))
    else:
        signal_item.setForeground(QColor(ModernColors.ACCENT_DANGER))
    self.signals_table.setItem(row, 2, signal_item)

    conf = pred.confidence if hasattr(pred, "confidence") else 0
    self.signals_table.setItem(row, 3, QTableWidgetItem(f"{conf:.0%}"))

    price = pred.current_price if hasattr(pred, "current_price") else 0
    self.signals_table.setItem(row, 4, QTableWidgetItem(f"CNY {price:.2f}"))

    action_btn = QPushButton("Trade")
    action_btn.clicked.connect(lambda: self._quick_trade(pred))
    self.signals_table.setCellWidget(row, 5, action_btn)

    while self.signals_table.rowCount() > 50:
        self.signals_table.removeRow(self.signals_table.rowCount() - 1)

    self.log(
        f"SIGNAL: {signal_text} - {pred.stock_code} @ CNY {price:.2f}",
        "success",
    )
    QApplication.alert(self)


def _refresh_live_chart_forecast(self: Any) -> None:
    """Periodic chart refresh for selected symbol.
    Ensures guessed graph updates in real time even with sparse feed ticks.
    """
    analyze_worker = self.workers.get("analyze")
    if analyze_worker and analyze_worker.isRunning():
        # Avoid creating transient single-bar placeholders while a
        # full history analysis for the selected symbol is in-flight.
        return
    if not CONFIG.is_market_open():
        return
    if not self.predictor:
        return
    code = self._ui_norm(self.stock_input.text())
    if not code:
        return

    try:
        from data.feeds import get_feed_manager

        fm = get_feed_manager(auto_init=True, async_init=True)
        quote = fm.get_quote(code)
        if quote and float(getattr(quote, "price", 0) or 0) > 0:
            self._on_price_updated(code, float(quote.price))
            return
    except Exception as exc:
        log.debug("Suppressed exception in ui/app.py", exc_info=exc)

    try:
        from data.fetcher import get_fetcher

        quote2 = get_fetcher().get_realtime(code)
        if quote2 and float(getattr(quote2, "price", 0) or 0) > 0:
            self._on_price_updated(code, float(quote2.price))
    except Exception as exc:
        log.debug("Suppressed exception in ui/app.py", exc_info=exc)
