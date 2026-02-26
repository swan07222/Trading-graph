from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from PyQt6.QtWidgets import QMessageBox

from config.settings import CONFIG
from core.types import AutoTradeMode
from ui.background_tasks import sanitize_watch_list as _sanitize_watch_list
from utils.logger import get_logger

_LOGGER = get_logger(__name__)


def _start_training(self: Any) -> None:
    """Start model training (UI dialog)."""
    interval = self.interval_combo.currentText().strip()
    horizon = self.forecast_spin.value()

    reply = QMessageBox.question(
        self,
        "Train GM",
        f"Start training with the following settings?\n\n"
        f"Interval: {interval}\n"
        f"Horizon: {horizon} bars\n\n"
        f"This may take time.\n\nContinue?",
        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        QMessageBox.StandardButton.No,
    )
    if reply != QMessageBox.StandardButton.Yes:
        return

    try:
        from .dialogs import TrainingDialog
    except ImportError as exc:
        self.log(f"Training dialog unavailable: {exc}", "error")
        return

    try:
        dialog = TrainingDialog(self)
        dialog.exec()
        result = getattr(dialog, "training_result", None)
        if isinstance(result, dict):
            if str(result.get("status", "")).strip().lower() == "complete":
                self.log("GM training completed successfully.", "success")
    except Exception as exc:
        self.log(f"Training dialog failed: {exc}", "error")
        return

    self._init_components()


def _show_auto_learn(self: Any, auto_start: bool = False) -> Any | None:
    """Show non-modal auto-learning (Auto Train GM) dialog."""
    try:
        from .auto_learn_dialog import AutoLearnDialog
    except ImportError:
        self.log("Auto-learn dialog not available", "error")
        return None

    dialog = getattr(self, "_auto_learn_dialog", None)
    if dialog is None:
        dialog = AutoLearnDialog(self, seed_stock_codes=[])
        self._auto_learn_dialog = dialog

        def _on_session_finished(results: dict[str, Any]) -> None:
            status = str((results or {}).get("status", "")).strip().lower()
            if status in {"ok", "complete", "trained"}:
                self._init_components()
            elif hasattr(self, "_refresh_model_training_statuses"):
                self._refresh_model_training_statuses()

        def _on_destroyed(*_args: object) -> None:
            self._auto_learn_dialog = None

        if hasattr(dialog, "session_finished"):
            dialog.session_finished.connect(_on_session_finished)
        dialog.destroyed.connect(_on_destroyed)

    dialog.setModal(False)
    dialog.show()
    dialog.raise_()
    dialog.activateWindow()

    if auto_start and hasattr(dialog, "start_or_resume_auto_learning"):
        dialog.start_or_resume_auto_learning()
    return dialog


def _show_strategy_marketplace(self: Any) -> None:
    """Show strategy marketplace manager."""
    try:
        from .strategy_marketplace_dialog import StrategyMarketplaceDialog
    except ImportError as exc:
        self.log(f"Strategy marketplace unavailable: {exc}", "error")
        return

    try:
        dialog = StrategyMarketplaceDialog(self)
        dialog.exec()
    except Exception as exc:
        self.log(f"Strategy marketplace unavailable: {exc}", "error")


def _show_backtest(self: Any) -> None:
    """Show backtest dialog."""
    try:
        from .dialogs import BacktestDialog
    except ImportError:
        self.log("Backtest dialog not available", "error")
        return

    dialog = BacktestDialog(self)
    dialog.exec()


def _show_about(self: Any) -> None:
    """Show about dialog."""
    QMessageBox.about(
        self,
        "About AI Stock Analysis System",
        "<h2>AI Stock Analysis System v2.0</h2>"
        "<p>Professional AI-powered stock analysis application</p>"
        "<h3>Features:</h3>"
        "<ul>"
        "<li>Custom AI model with ensemble neural networks</li>"
        "<li>Real-time signal monitoring (1m, 5m, 1d intervals)</li>"
        "<li>Automatic stock discovery from internet</li>"
        "<li>AI-generated price forecast curves</li>"
        "<li>Stock analysis dashboard and watchlist</li>"
        "</ul>"
        "<p><b>Notice:</b></p>"
        "<p>This build focuses on analysis and model learning workflows.</p>",
    )


def _log(self: Any, message: str, level: str = "info") -> None:
    """Log message to UI."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    colors = {
        "info": "#dbe4f3",
        "success": "#35b57c",
        "warning": "#d8a03a",
        "error": "#e5534b",
    }
    color = colors.get(level, "#dbe4f3")

    formatted = (
        f'<span style="color: #888;">[{timestamp}]</span> '
        f'<span style="color: {color};">{message}</span>'
    )

    if hasattr(self, "_append_ai_chat_message"):
        self._append_ai_chat_message(
            "System",
            str(message),
            role="system",
            level=level,
        )
    else:
        widget = getattr(self, "log_widget", None)
        if widget is not None and hasattr(widget, "log"):
            widget.log(message, level)
        elif widget is not None and hasattr(widget, "append"):
            widget.append(formatted)

    _LOGGER.info(message)


def _close_event(self: Any, event: Any) -> None:
    """Handle window close safely."""
    if self.monitor:
        try:
            self._stop_monitoring()
        except Exception as exc:
            _LOGGER.debug("Suppressed exception in ui/app.py", exc_info=exc)

    all_workers = set(self._active_workers) | set(self.workers.values())
    for worker in list(all_workers):
        try:
            worker.cancel()
            worker.quit()
            if not worker.wait(3000):
                worker.terminate()
                worker.wait(1000)
        except Exception as exc:
            _LOGGER.debug("Suppressed exception in ui/app.py", exc_info=exc)
    self._active_workers.clear()
    self.workers.clear()

    for timer_name in (
        "clock_timer",
        "market_timer",
        "sentiment_timer",
        "watchlist_timer",
        "auto_trade_timer",
        "chart_live_timer",
        "universe_refresh_timer",
        "cache_prune_timer",
    ):
        timer = getattr(self, timer_name, None)
        try:
            if timer:
                timer.stop()
        except Exception as exc:
            _LOGGER.debug("Suppressed exception in ui/app.py", exc_info=exc)

    try:
        self._shutdown_session_cache_writer()
    except Exception as exc:
        _LOGGER.debug("Suppressed exception in ui/app.py", exc_info=exc)

    try:
        self._save_state()
    except Exception as exc:
        _LOGGER.debug("Suppressed exception in ui/app.py", exc_info=exc)

    event.accept()


def _save_state(self: Any) -> None:
    """Save application state for next session."""
    try:
        safe_watch_list = _sanitize_watch_list(
            self.watch_list,
            max_size=self.MAX_WATCHLIST_SIZE,
        )
        self.watch_list = safe_watch_list
        state = {
            "watch_list": safe_watch_list,
            "interval": self.interval_combo.currentText(),
            "forecast": self.forecast_spin.value(),
            "lookback": self.lookback_spin.value(),
            "capital": self.capital_spin.value(),
            "last_stock": self.stock_input.text(),
            "chart_date": str(
                getattr(self, "_selected_chart_date", "") or ""
            ),
            "auto_trade_mode": self._auto_trade_mode.value,
        }

        state_path = CONFIG.DATA_DIR / "app_state.json"
        state_path.parent.mkdir(parents=True, exist_ok=True)
        with state_path.open("w", encoding="utf-8") as handle:
            json.dump(state, handle, indent=2)
    except Exception as exc:
        _LOGGER.debug("Failed to save state: %s", exc)


def _load_state(self: Any) -> None:
    """Load application state from previous session."""
    try:
        state_path = CONFIG.DATA_DIR / "app_state.json"
        if not state_path.exists():
            return

        with state_path.open(encoding="utf-8") as handle:
            state = json.load(handle)

        if "watch_list" in state:
            loaded = state["watch_list"]
            self.watch_list = _sanitize_watch_list(
                loaded,
                max_size=self.MAX_WATCHLIST_SIZE,
            )
        if "forecast" in state:
            self.forecast_spin.setValue(state["forecast"])

        # Startup should always begin on 1m with latest 2-day window.
        self.interval_combo.blockSignals(True)
        try:
            self.interval_combo.setCurrentText(self.STARTUP_INTERVAL)
        finally:
            self.interval_combo.blockSignals(False)
        self.lookback_spin.setValue(
            self._recommended_lookback(self.STARTUP_INTERVAL)
        )

        if "capital" in state:
            self.capital_spin.setValue(state["capital"])
        if "last_stock" in state:
            self.stock_input.setText(state["last_stock"])
        chart_date = str(state.get("chart_date", "") or "").strip()
        if chart_date and hasattr(self, "chart_date_edit"):
            try:
                from PyQt6.QtCore import QDate

                parsed = QDate.fromString(chart_date, "yyyy-MM-dd")
                if parsed.isValid():
                    self.chart_date_edit.setDate(parsed)
                    self._selected_chart_date = str(
                        parsed.toString("yyyy-MM-dd")
                    )
            except Exception:
                pass
        if "auto_trade_mode" in state:
            try:
                self._auto_trade_mode = AutoTradeMode(state["auto_trade_mode"])
                if self._auto_trade_mode == AutoTradeMode.AUTO:
                    self._auto_trade_mode = AutoTradeMode.MANUAL
            except (ValueError, KeyError):
                self._auto_trade_mode = AutoTradeMode.MANUAL

        _LOGGER.debug("Application state restored")
    except Exception as exc:
        _LOGGER.debug("Failed to load state: %s", exc)
