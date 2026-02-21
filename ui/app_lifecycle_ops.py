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
        "Train AI Model",
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
                self._handle_training_drift_alarm(
                    result,
                    context="training_dialog",
                )
                trained_codes = list(
                    dict.fromkeys(
                        self._ui_norm(x)
                        for x in list(result.get("trained_stock_codes", []) or [])
                        if self._ui_norm(x)
                    )
                )
                if trained_codes:
                    self._record_trained_stock_last_train(
                        trained_codes,
                        trained_at=datetime.now().isoformat(timespec="seconds"),
                    )
                    self._update_trained_stocks_ui()
    except Exception as exc:
        self.log(f"Training dialog failed: {exc}", "error")
        return

    self._init_components()


def _show_auto_learn(self: Any) -> None:
    """Show auto-learning dialog."""
    try:
        from .auto_learn_dialog import show_auto_learn_dialog
    except ImportError:
        self.log("Auto-learn dialog not available", "error")
        return

    seed_codes: list[str] = []
    try:
        if self._session_bar_cache is not None:
            interval = self._normalize_interval_token(self.interval_combo.currentText())
            seed_codes = self._session_bar_cache.get_recent_symbols(
                interval=interval,
                min_rows=10,
            )
    except Exception:
        seed_codes = []

    show_auto_learn_dialog(self, seed_stock_codes=seed_codes)
    self._init_components()


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
        "About AI Stock Trading System",
        "<h2>AI Stock Trading System v2.0</h2>"
        "<p>Professional AI-powered stock trading application</p>"
        "<h3>Features:</h3>"
        "<ul>"
        "<li>Custom AI model with ensemble neural networks</li>"
        "<li>Real-time signal monitoring (1m, 5m, 1d intervals)</li>"
        "<li>Automatic stock discovery from internet</li>"
        "<li>AI-generated price forecast curves</li>"
        "<li>Paper and live trading support</li>"
        "<li>Comprehensive risk management</li>"
        "</ul>"
        "<p><b>Risk Warning:</b></p>"
        "<p>Stock trading involves risk. Past performance does not "
        "guarantee future results. Only trade with money you can "
        "afford to lose.</p>",
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

    if hasattr(self.log_widget, "log"):
        self.log_widget.log(message, level)
    elif hasattr(self.log_widget, "append"):
        self.log_widget.append(formatted)

    _LOGGER.info(message)


def _close_event(self: Any, event: Any) -> None:
    """Handle window close safely."""
    if self.monitor:
        try:
            self._stop_monitoring()
        except Exception as exc:
            _LOGGER.debug("Suppressed exception in ui/app.py", exc_info=exc)

    if self.executor and self.executor.auto_trader:
        try:
            self.executor.auto_trader.stop()
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

    if self.executor:
        try:
            self.executor.stop()
        except Exception as exc:
            _LOGGER.debug("Suppressed exception in ui/app.py", exc_info=exc)
        self.executor = None

    for timer_name in (
        "clock_timer",
        "market_timer",
        "portfolio_timer",
        "watchlist_timer",
        "auto_trade_timer",
        "chart_live_timer",
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

        # Startup should always begin on 1m with latest 7-day window.
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
