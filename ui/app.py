# ui/app.py
import math
import os
import signal
import sys
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeout
from datetime import datetime
from importlib import import_module
from statistics import median
from typing import Any

import numpy as np
from PyQt6.QtCore import QSize, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QAction, QActionGroup, QColor, QFont
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QStatusBar,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from config.settings import CONFIG, TradingMode
from core.types import (
    AutoTradeAction,
    AutoTradeMode,
    OrderSide,
)
from ui import app_bar_ops as _app_bar_ops
from ui.app_auto_trade_ops import (
    _apply_auto_trade_mode as _apply_auto_trade_mode_impl,
)
from ui.app_auto_trade_ops import (
    _approve_all_pending as _approve_all_pending_impl,
)
from ui.app_auto_trade_ops import (
    _init_auto_trader as _init_auto_trader_impl,
)
from ui.app_auto_trade_ops import (
    _on_auto_trade_action as _on_auto_trade_action_impl,
)
from ui.app_auto_trade_ops import (
    _on_auto_trade_action_safe as _on_auto_trade_action_safe_impl,
)
from ui.app_auto_trade_ops import (
    _on_pending_approval as _on_pending_approval_impl,
)
from ui.app_auto_trade_ops import (
    _on_pending_approval_safe as _on_pending_approval_safe_impl,
)
from ui.app_auto_trade_ops import (
    _on_trade_mode_changed as _on_trade_mode_changed_impl,
)
from ui.app_auto_trade_ops import (
    _refresh_auto_trade_ui as _refresh_auto_trade_ui_impl,
)
from ui.app_auto_trade_ops import (
    _refresh_pending_table as _refresh_pending_table_impl,
)
from ui.app_auto_trade_ops import (
    _reject_all_pending as _reject_all_pending_impl,
)
from ui.app_auto_trade_ops import (
    _show_auto_trade_settings as _show_auto_trade_settings_impl,
)
from ui.app_auto_trade_ops import (
    _toggle_auto_pause as _toggle_auto_pause_impl,
)
from ui.app_auto_trade_ops import (
    _update_auto_trade_status_label as _update_auto_trade_status_label_impl,
)
from ui.app_chart_pipeline import (
    _load_chart_history_bars as _load_chart_history_bars_impl,
)
from ui.app_chart_pipeline import (
    _on_price_updated as _on_price_updated_impl,
)
from ui.app_chart_pipeline import (
    _prepare_chart_bars_for_interval as _prepare_chart_bars_for_interval_impl,
)
from ui.app_common import MainAppCommonMixin
from ui.app_lifecycle_ops import (
    _close_event as _close_event_impl,
)
from ui.app_lifecycle_ops import (
    _load_state as _load_state_impl,
)
from ui.app_lifecycle_ops import (
    _log as _log_impl,
)
from ui.app_lifecycle_ops import (
    _save_state as _save_state_impl,
)
from ui.app_lifecycle_ops import (
    _show_about as _show_about_impl,
)
from ui.app_lifecycle_ops import (
    _show_auto_learn as _show_auto_learn_impl,
)
from ui.app_lifecycle_ops import (
    _show_backtest as _show_backtest_impl,
)
from ui.app_lifecycle_ops import (
    _show_strategy_marketplace as _show_strategy_marketplace_impl,
)
from ui.app_lifecycle_ops import (
    _start_training as _start_training_impl,
)
from ui.app_monitoring_ops import (
    _on_signal_detected as _on_signal_detected_impl,
)
from ui.app_monitoring_ops import (
    _refresh_live_chart_forecast as _refresh_live_chart_forecast_impl,
)
from ui.app_monitoring_ops import (
    _start_monitoring as _start_monitoring_impl,
)
from ui.app_monitoring_ops import (
    _stop_monitoring as _stop_monitoring_impl,
)
from ui.app_monitoring_ops import (
    _toggle_monitoring as _toggle_monitoring_impl,
)
from ui.app_panels import (
    _apply_professional_style as _apply_professional_style_impl,
)
from ui.app_panels import (
    _create_left_panel as _create_left_panel_impl,
)
from ui.app_panels import (
    _create_right_panel as _create_right_panel_impl,
)
from ui.app_trading_ops import (
    _connect_trading as _connect_trading_impl,
)
from ui.app_trading_ops import (
    _disconnect_trading as _disconnect_trading_impl,
)
from ui.app_trading_ops import (
    _execute_buy as _execute_buy_impl,
)
from ui.app_trading_ops import (
    _execute_sell as _execute_sell_impl,
)
from ui.app_trading_ops import (
    _on_chart_trade_requested as _on_chart_trade_requested_impl,
)
from ui.app_trading_ops import (
    _on_mode_combo_changed as _on_mode_combo_changed_impl,
)
from ui.app_trading_ops import (
    _on_order_filled as _on_order_filled_impl,
)
from ui.app_trading_ops import (
    _on_order_rejected as _on_order_rejected_impl,
)
from ui.app_trading_ops import (
    _refresh_all as _refresh_all_impl,
)
from ui.app_trading_ops import (
    _refresh_portfolio as _refresh_portfolio_impl,
)
from ui.app_trading_ops import (
    _set_trading_mode as _set_trading_mode_impl,
)
from ui.app_trading_ops import (
    _show_chart_trade_dialog as _show_chart_trade_dialog_impl,
)
from ui.app_trading_ops import (
    _submit_chart_order as _submit_chart_order_impl,
)
from ui.app_trading_ops import (
    _toggle_trading as _toggle_trading_impl,
)
from ui.background_tasks import (
    RealTimeMonitor,
    WorkerThread,
)
from ui.background_tasks import (
    sanitize_watch_list as _sanitize_watch_list,
)
from ui.background_tasks import (
    validate_stock_code as _validate_stock_code,
)
from utils.logger import get_logger

log = get_logger(__name__)

def _lazy_get(module: str, name: str) -> Any:
    return getattr(import_module(module), name)

class MainApp(MainAppCommonMixin, QMainWindow):
    """
    Professional AI Stock Trading Application

    Features:
    - Real-time signal monitoring with multiple intervals
    - Custom AI model with ensemble neural networks
    - Professional dark theme
    - Live/Paper trading support
    - Comprehensive risk management
    - AI-generated price forecast curves
    """
    MAX_WATCHLIST_SIZE = 50
    GUESS_FORECAST_BARS = 30
    STARTUP_INTERVAL = "1m"

    bar_received = pyqtSignal(str, dict)
    quote_received = pyqtSignal(str, float)

    def __init__(self) -> None:
        super().__init__()

        self.setWindowTitle("AI Stock Trading System v2.0")
        self.setMinimumSize(980, 640)
        self._set_initial_window_geometry()

        self.predictor = None
        self.executor = None
        self.current_prediction = None
        self.workers: dict[str, WorkerThread] = {}
        self._active_workers: set[WorkerThread] = set()
        self.monitor: RealTimeMonitor | None = None
        self.watch_list: list[str] = _sanitize_watch_list(
            list(getattr(CONFIG, "STOCK_POOL", [])[:10]),
            max_size=self.MAX_WATCHLIST_SIZE,
        )

        # Real-time state with thread safety
        self._last_forecast_refresh_ts: float = 0.0
        self._forecast_refresh_symbol: str = ""
        self._live_price_series: dict[str, list[float]] = {}
        self._price_series_lock = threading.Lock()
        self._session_cache_write_lock = threading.Lock()
        self._last_session_cache_write_ts: dict[str, float] = {}
        self._session_cache_io_pool = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="session-cache-io",
        )
        self._session_cache_io_lock = threading.Lock()
        self._session_cache_io_futures: set[Future[object]] = set()
        self._last_analyze_request: dict[str, Any] = {}
        self._last_analysis_log: dict[str, Any] = {}
        self._analysis_recovery_attempt_ts: dict[str, float] = {}
        self._watchlist_row_by_code: dict[str, int] = {}
        self._last_watchlist_price_ui: dict[str, tuple[float, float]] = {}
        self._last_quote_ui_emit: dict[str, tuple[float, float]] = {}
        self._guess_profit_notional_shares: int = max(
            1, int(getattr(CONFIG, "LOT_SIZE", 100) or 100)
        )

        self._bars_by_symbol: dict[str, list[dict[str, Any]]] = {}
        self._trained_stock_codes_cache: list[str] = []
        self._trained_stock_last_train: dict[str, str] = {}
        self._last_bar_feed_ts: dict[str, float] = {}
        self._chart_symbol: str = ""
        self._history_refresh_once: set[tuple[str, str]] = set()
        self._strict_startup = str(
            os.environ.get("TRADING_STRICT_STARTUP", "0")
        ).strip().lower() in ("1", "true", "yes", "on")
        self._debug_console_enabled: bool = str(
            os.environ.get("TRADING_DEBUG_CONSOLE", "1")
        ).strip().lower() in ("1", "true", "yes", "on")
        self._debug_console_last_emit: dict[str, float] = {}
        self._syncing_mode_ui = False
        self._session_bar_cache = None
        try:
            from data.session_cache import get_session_bar_cache
            self._session_bar_cache = get_session_bar_cache()
        except Exception as exc:
            log.warning("Session cache unavailable at startup: %s", exc)
            self._session_bar_cache = None
            if self._strict_startup:
                raise
        self._load_trained_stock_last_train_meta()

        # Auto-trade state
        self._auto_trade_mode: AutoTradeMode = AutoTradeMode.MANUAL

        self._setup_menubar()
        self._setup_toolbar()
        self._setup_ui()
        init_mode = (
            TradingMode.LIVE
            if getattr(CONFIG.trading_mode, "value", "simulation") == "live"
            else TradingMode.SIMULATION
        )
        self._set_trading_mode(init_mode, prompt_reconnect=False)
        self._setup_statusbar()
        self._setup_timers()
        self._apply_professional_style()
        self.bar_received.connect(self._on_bar_ui)
        self.quote_received.connect(self._on_price_updated)

        try:
            self._load_state()
            self._update_watchlist()
        except Exception:
            log.exception("Startup state restore failed")
            if self._strict_startup:
                raise

        QTimer.singleShot(0, self._init_components)

    def _sync_ui_to_loaded_model(
        self,
        requested_interval: str | None = None,
        requested_horizon: int | None = None,
        preserve_requested_interval: bool = False,
    ) -> tuple[str, int]:
        """
        Align UI controls to the actual loaded model metadata.
        Prevents 'UI says 1m/120 while model runs 1d/5' mismatches.
        """
        if not self.predictor:
            iv = str(requested_interval or self.interval_combo.currentText()).strip().lower()
            h = int(requested_horizon or self.forecast_spin.value())
            return iv, h

        model_iv_raw = str(
            getattr(
                self.predictor,
                "_loaded_model_interval",
                getattr(
                    self.predictor,
                    "interval",
                    requested_interval or self.interval_combo.currentText(),
                ),
            )
        ).strip().lower()
        model_iv = self._model_interval_to_ui_token(model_iv_raw)

        items = [
            str(self.interval_combo.itemText(i)).strip().lower()
            for i in range(self.interval_combo.count())
        ]
        ui_iv = model_iv
        if preserve_requested_interval and requested_interval is not None:
            ui_iv = self._model_interval_to_ui_token(str(requested_interval).strip().lower())
        if ui_iv not in items:
            ui_iv = str(requested_interval or self.interval_combo.currentText()).strip().lower()

        try:
            model_h_raw = int(
                getattr(
                    self.predictor,
                    "_loaded_model_horizon",
                    getattr(
                        self.predictor,
                        "horizon",
                        requested_horizon
                        if requested_horizon is not None
                        else self.forecast_spin.value(),
                    ),
                )
            )
        except Exception:
            model_h_raw = int(self.forecast_spin.value())

        model_h = max(
            int(self.forecast_spin.minimum()),
            min(int(self.forecast_spin.maximum()), int(model_h_raw)),
        )
        ui_h = model_h
        if preserve_requested_interval and requested_horizon is not None:
            ui_h = max(
                int(self.forecast_spin.minimum()),
                min(int(self.forecast_spin.maximum()), int(requested_horizon)),
            )

        self.interval_combo.blockSignals(True)
        try:
            self.interval_combo.setCurrentText(ui_iv)
        finally:
            self.interval_combo.blockSignals(False)
        self.forecast_spin.setValue(ui_h)
        if ui_iv != model_iv or ui_h != model_h:
            self.model_info.setText(
                f"Interval: {ui_iv}, Horizon: {ui_h} (model: {model_iv}/{model_h})"
            )
        else:
            self.model_info.setText(f"Interval: {ui_iv}, Horizon: {ui_h}")

        if (
            (not preserve_requested_interval)
            and requested_interval is not None
            and requested_horizon is not None
        ):
            req_iv = str(requested_interval).strip().lower()
            req_h = int(requested_horizon)
            if req_iv != ui_iv or req_h != ui_h:
                self.log(
                    f"Loaded model metadata applied: requested {req_iv}/{req_h} -> active {ui_iv}/{ui_h}",
                    "warning",
                )

        return ui_iv, ui_h

    def _loaded_model_ui_meta(self) -> tuple[str, int]:
        """Return loaded model metadata as (ui_interval_token, horizon)."""
        if not self.predictor:
            iv = self._normalize_interval_token(self.interval_combo.currentText())
            hz = int(self.forecast_spin.value())
            return iv, hz
        raw_iv = str(
            getattr(
                self.predictor,
                "_loaded_model_interval",
                getattr(self.predictor, "interval", self.interval_combo.currentText()),
            )
        ).strip().lower()
        iv = self._model_interval_to_ui_token(raw_iv)
        try:
            hz = int(
                getattr(
                    self.predictor,
                    "_loaded_model_horizon",
                    getattr(self.predictor, "horizon", self.forecast_spin.value()),
                )
            )
        except Exception:
            hz = int(self.forecast_spin.value())
        return iv, max(1, hz)

    def _has_exact_model_artifacts(self, interval: str, horizon: int) -> bool:
        """Whether exact ensemble+scaler artifacts exist for interval+horizon."""
        iv = self._normalize_interval_token(interval)
        try:
            hz = max(1, int(horizon))
        except Exception:
            hz = int(self.forecast_spin.value())
        model_dir = CONFIG.MODEL_DIR
        ens = model_dir / f"ensemble_{iv}_{hz}.pt"
        scl = model_dir / f"scaler_{iv}_{hz}.pkl"
        return bool(ens.exists() and scl.exists())

    def _log_model_alignment_debug(
        self,
        *,
        context: str,
        requested_interval: str | None = None,
        requested_horizon: int | None = None,
    ) -> None:
        """Verbose model/UI alignment diagnostics for bug hunting."""
        ui_iv = self._normalize_interval_token(
            requested_interval if requested_interval is not None else self.interval_combo.currentText()
        )
        try:
            ui_h = int(requested_horizon if requested_horizon is not None else self.forecast_spin.value())
        except Exception:
            ui_h = int(self.forecast_spin.value())
        model_iv, model_h = self._loaded_model_ui_meta()
        exact = self._has_exact_model_artifacts(ui_iv, ui_h)
        min_gap = 8.0 if str(context).strip().lower() == "analyze" else 0.8
        self._debug_console(
            f"model_align:{context}:{ui_iv}:{ui_h}",
            (
                f"model alignment [{context}] ui={ui_iv}/{ui_h} "
                f"loaded={model_iv}/{model_h} exact_artifacts={int(exact)}"
            ),
            min_gap_seconds=min_gap,
            level="info",
        )

    def _debug_chart_state(
        self,
        *,
        symbol: str,
        interval: str,
        bars: list[dict[str, Any]] | None,
        predicted_prices: list[float] | None = None,
        context: str = "chart",
    ) -> None:
        """Emit compact chart diagnostics for rapid bug triage."""
        arr = self._safe_list(bars)
        preds = self._safe_list(predicted_prices)
        if not arr:
            self._debug_console(
                f"chart_state:{context}:{self._ui_norm(symbol)}:{self._normalize_interval_token(interval)}",
                (
                    f"chart state [{context}] symbol={self._ui_norm(symbol)} "
                    f"iv={self._normalize_interval_token(interval)} bars=0 preds={len(preds)}"
                ),
                min_gap_seconds=1.0,
                level="info",
            )
            return

        closes: list[float] = []
        max_span_pct = 0.0
        for row in arr[-min(220, len(arr)):]:
            try:
                c = float(row.get("close", 0) or 0)
                h = float(row.get("high", c) or c)
                low = float(row.get("low", c) or c)
            except Exception:
                continue
            if c > 0 and math.isfinite(c):
                closes.append(c)
                span = abs(h - low) / max(c, 1e-8)
                if math.isfinite(span):
                    max_span_pct = max(max_span_pct, float(span))

        last_close = float(closes[-1]) if closes else 0.0
        med_close = float(median(closes)) if closes else 0.0
        model_iv, model_h = self._loaded_model_ui_meta()
        iv = self._normalize_interval_token(interval)
        sym = self._ui_norm(symbol)
        self._debug_console(
            f"chart_state:{context}:{sym}:{iv}",
            (
                f"chart state [{context}] symbol={sym} iv={iv} bars={len(arr)} "
                f"preds={len(preds)} last={last_close:.4f} med={med_close:.4f} "
                f"max_span={max_span_pct:.2%} model={model_iv}/{model_h}"
            ),
            min_gap_seconds=1.0,
            level="info",
        )
        if max_span_pct > 0.035:
            self._debug_console(
                f"chart_state_anom:{context}:{sym}:{iv}",
                (
                    f"chart anomaly [{context}] symbol={sym} iv={iv} "
                    f"max_span={max_span_pct:.2%} bars={len(arr)} preds={len(preds)}"
                ),
                min_gap_seconds=0.6,
                level="warning",
            )

    def _debug_candle_quality(
        self,
        *,
        symbol: str,
        interval: str,
        bars: list[dict[str, Any]] | None,
        context: str,
    ) -> None:
        """Detailed candle-shape diagnostics for malformed chart bars."""
        if not bool(getattr(self, "_debug_console_enabled", False)):
            return

        arr = self._safe_list(bars)
        if not arr:
            return

        iv = self._normalize_interval_token(interval)
        jump_cap, range_cap = self._bar_safety_caps(iv)
        intraday = iv not in ("1d", "1wk", "1mo")
        if intraday:
            body_cap = float(max(range_cap * 1.35, 0.007))
            span_cap = float(max(range_cap * 2.10, 0.012))
            wick_cap = float(max(range_cap * 1.55, 0.008))
        else:
            body_cap = float(max(range_cap * 0.85, 0.045))
            span_cap = float(max(range_cap * 1.75, 0.45))
            wick_cap = float(max(range_cap * 1.25, 0.25))

        parsed = 0
        invalid = 0
        duplicates = 0
        doji_like = 0
        extreme = 0
        max_body = 0.0
        max_span = 0.0
        max_wick = 0.0
        max_jump = 0.0
        seen_ts: set[int] = set()
        prev_close: float | None = None

        for row in arr[-min(520, len(arr)):]:
            try:
                o = float(row.get("open", 0) or 0)
                h = float(row.get("high", 0) or 0)
                low = float(row.get("low", 0) or 0)
                c = float(row.get("close", 0) or 0)
            except Exception:
                invalid += 1
                continue
            if (
                c <= 0
                or not all(math.isfinite(v) for v in (o, h, low, c))
            ):
                invalid += 1
                continue
            parsed += 1

            ts_epoch = int(
                self._bar_bucket_epoch(
                    row.get("_ts_epoch", row.get("timestamp")),
                    iv,
                )
            )
            if ts_epoch in seen_ts:
                duplicates += 1
            else:
                seen_ts.add(ts_epoch)

            ref = float(prev_close if prev_close and prev_close > 0 else c)
            if ref <= 0:
                ref = float(c)
            body = abs(o - c) / max(ref, 1e-8)
            span = abs(h - low) / max(ref, 1e-8)
            uw = max(0.0, h - max(o, c)) / max(ref, 1e-8)
            lw = max(0.0, min(o, c) - low) / max(ref, 1e-8)
            max_body = max(max_body, float(body))
            max_span = max(max_span, float(span))
            max_wick = max(max_wick, float(max(uw, lw)))
            if prev_close and prev_close > 0:
                jump = abs(c / max(prev_close, 1e-8) - 1.0)
                max_jump = max(max_jump, float(jump))

            if body <= 0.00012 and span <= 0.00120:
                doji_like += 1
            if (
                body > body_cap
                or span > span_cap
                or uw > wick_cap
                or lw > wick_cap
            ):
                extreme += 1

            prev_close = float(c)

        if parsed <= 0:
            self._debug_console(
                f"candle_q:{context}:{self._ui_norm(symbol)}:{iv}",
                (
                    f"candle quality [{context}] symbol={self._ui_norm(symbol)} "
                    f"iv={iv} parsed=0 invalid={invalid}"
                ),
                min_gap_seconds=1.5,
                level="warning",
            )
            return

        doji_ratio = float(doji_like) / float(max(1, parsed))
        extreme_ratio = float(extreme) / float(max(1, parsed))
        self._debug_console(
            f"candle_q:{context}:{self._ui_norm(symbol)}:{iv}",
            (
                f"candle quality [{context}] symbol={self._ui_norm(symbol)} "
                f"iv={iv} bars={parsed} invalid={invalid} dup={duplicates} "
                f"doji={doji_ratio:.1%} extreme={extreme_ratio:.1%} "
                f"max_body={max_body:.2%} max_span={max_span:.2%} "
                f"max_wick={max_wick:.2%} max_jump={max_jump:.2%} "
                f"caps(body={body_cap:.2%},span={span_cap:.2%},wick={wick_cap:.2%},jump={jump_cap:.2%})"
            ),
            min_gap_seconds=1.5,
            level="info",
        )
        if duplicates > 0 or extreme > 0 or doji_ratio >= 0.86:
            self._debug_console(
                f"candle_q_warn:{context}:{self._ui_norm(symbol)}:{iv}",
                (
                    f"candle anomaly [{context}] symbol={self._ui_norm(symbol)} "
                    f"iv={iv} dup={duplicates} extreme={extreme}/{parsed} "
                    f"doji={doji_like}/{parsed}"
                ),
                min_gap_seconds=0.8,
                level="warning",
            )

    def _debug_forecast_quality(
        self,
        *,
        symbol: str,
        chart_interval: str,
        source_interval: str,
        predicted_prices: list[float] | None,
        anchor_price: float | None,
        context: str,
    ) -> None:
        """Detailed forecast-shape diagnostics for flat/erratic guessed graph."""
        if not bool(getattr(self, "_debug_console_enabled", False)):
            return

        iv_chart = self._normalize_interval_token(chart_interval)
        iv_src = self._normalize_interval_token(source_interval, fallback=iv_chart)
        vals: list[float] = []
        for v in self._safe_list(predicted_prices):
            try:
                fv = float(v)
            except Exception:
                continue
            if fv > 0 and math.isfinite(fv):
                vals.append(fv)

        sym = self._ui_norm(symbol)
        if not vals:
            self._debug_console(
                f"forecast_q:{context}:{sym}:{iv_chart}",
                (
                    f"forecast quality [{context}] symbol={sym} chart={iv_chart} "
                    f"source={iv_src} points=0"
                ),
                min_gap_seconds=1.5,
                level="warning",
            )
            return

        try:
            anchor = float(anchor_price or 0.0)
        except Exception:
            anchor = 0.0
        if not math.isfinite(anchor) or anchor <= 0:
            anchor = float(vals[0])

        max_step = 0.0
        flips = 0
        dirs: list[int] = []
        for i in range(1, len(vals)):
            prev = float(vals[i - 1])
            cur = float(vals[i])
            if prev > 0:
                step = abs(cur / max(prev, 1e-8) - 1.0)
                max_step = max(max_step, float(step))
            dirs.append(1 if cur >= prev else -1)
        for i in range(1, len(dirs)):
            if dirs[i] != dirs[i - 1]:
                flips += 1
        flip_ratio = float(flips) / float(max(1, len(dirs) - 1)) if len(dirs) >= 2 else 0.0

        vmin = min(vals)
        vmax = max(vals)
        span_pct = abs(vmax - vmin) / max(anchor, 1e-8)
        net_pct = (float(vals[-1]) / max(anchor, 1e-8)) - 1.0
        mean_v = float(sum(vals) / max(1, len(vals)))
        var = 0.0
        for v in vals:
            var += (float(v) - mean_v) ** 2
        std_pct = (math.sqrt(var / float(max(1, len(vals)))) / max(anchor, 1e-8))

        _, cap_step = self._chart_prediction_caps(iv_chart)
        quiet_span_pct = 0.0
        quiet_std_pct = 0.0
        try:
            recent_rows = list(self._bars_by_symbol.get(self._ui_norm(symbol), []) or [])
            recent_closes: list[float] = []
            for row in recent_rows[-96:]:
                try:
                    row_iv = self._normalize_interval_token(
                        row.get("interval", iv_chart),
                        fallback=iv_chart,
                    )
                    if row_iv != iv_chart:
                        continue
                    px = float(row.get("close", 0.0) or 0.0)
                    if px > 0 and math.isfinite(px):
                        recent_closes.append(px)
                except Exception:
                    continue
            if len(recent_closes) >= 6:
                q_anchor = float(np.median(np.asarray(recent_closes[-24:], dtype=float)))
                q_anchor = max(q_anchor, 1e-8)
                quiet_span_pct = (
                    abs(max(recent_closes) - min(recent_closes)) / q_anchor
                )
                q_std = float(np.std(np.asarray(recent_closes, dtype=float)))
                quiet_std_pct = q_std / q_anchor
        except Exception:
            quiet_span_pct = 0.0
            quiet_std_pct = 0.0

        quiet_market = bool(
            (quiet_span_pct > 0 and quiet_span_pct <= max(0.0030, cap_step * 0.85))
            or (quiet_std_pct > 0 and quiet_std_pct <= max(0.0010, cap_step * 0.22))
        )

        flat_line_raw = len(vals) >= 5 and (span_pct <= 0.0012 or std_pct <= 0.0005)
        flat_line = bool(flat_line_raw and not quiet_market)
        jagged = max_step > (cap_step * 1.80) or flip_ratio > 0.82

        self._debug_console(
            f"forecast_q:{context}:{sym}:{iv_chart}",
            (
                f"forecast quality [{context}] symbol={sym} chart={iv_chart} "
                f"source={iv_src} points={len(vals)} span={span_pct:.2%} "
                f"net={net_pct:+.2%} max_step={max_step:.2%} flips={flip_ratio:.2f} "
                f"std={std_pct:.2%} step_cap={cap_step:.2%}"
            ),
            min_gap_seconds=1.5,
            level="info",
        )
        if flat_line or jagged:
            why = []
            if flat_line:
                why.append("flat_line")
            if jagged:
                why.append("jagged")
            self._debug_console(
                f"forecast_q_warn:{context}:{sym}:{iv_chart}",
                (
                    f"forecast anomaly [{context}] symbol={sym} chart={iv_chart} "
                    f"source={iv_src} reason={','.join(why)} "
                    f"span={span_pct:.2%} max_step={max_step:.2%} flips={flip_ratio:.2f}"
                ),
                min_gap_seconds=0.8,
                level="warning",
            )
        elif flat_line_raw and quiet_market:
            self._debug_console(
                f"forecast_q_quiet:{context}:{sym}:{iv_chart}",
                (
                    f"forecast quiet-shape [{context}] symbol={sym} chart={iv_chart} "
                    f"source={iv_src} accepted_flat=1 "
                    f"market_span={quiet_span_pct:.2%} market_std={quiet_std_pct:.2%}"
                ),
                min_gap_seconds=2.0,
                level="info",
            )

    def _chart_prediction_caps(self, interval: str) -> tuple[float, float]:
        """Return (max_total_move, max_step_move) for display-only forecast shaping."""
        iv = self._normalize_interval_token(interval)
        if iv == "1m":
            return 0.018, 0.0045
        if iv == "5m":
            return 0.030, 0.008
        if iv in ("15m", "30m"):
            return 0.050, 0.012
        if iv in ("60m", "1h"):
            return 0.090, 0.022
        if iv in ("1d", "1wk", "1mo"):
            return 0.140, 0.035
        return 0.060, 0.020

    def _prepare_chart_predicted_prices(
        self,
        *,
        symbol: str,
        chart_interval: str,
        predicted_prices: list[float] | None,
        source_interval: str | None = None,
        current_price: float | None = None,
        target_steps: int | None = None,
    ) -> list[float]:
        """
        Shape forecast for chart display stability.
        - Clamp implausible per-step spikes.
        - When model interval != chart interval, project a smooth path to avoid
          abrupt vertical zig-zags on intraday charts.
        """
        raw_vals = self._safe_list(predicted_prices)
        cleaned: list[float] = []
        for v in raw_vals:
            try:
                fv = float(v)
            except Exception:
                continue
            if fv > 0 and math.isfinite(fv):
                cleaned.append(fv)
        if not cleaned:
            return []

        iv_chart = self._normalize_interval_token(chart_interval)
        iv_src = self._normalize_interval_token(source_interval, fallback=iv_chart)
        try:
            steps = int(target_steps if target_steps is not None else self.forecast_spin.value())
        except Exception:
            steps = len(cleaned)
        steps = max(1, steps)

        try:
            anchor = float(current_price or 0.0)
        except Exception:
            anchor = 0.0
        if not math.isfinite(anchor) or anchor <= 0:
            anchor = float(cleaned[0])
        if not math.isfinite(anchor) or anchor <= 0:
            return []

        max_total_move, max_step_move = self._chart_prediction_caps(iv_chart)

        # Mismatch mode: preserve source-shape, then resample to chart steps.
        if iv_src != iv_chart:
            chart_sec = float(max(1, self._interval_seconds(iv_chart)))
            src_sec = float(max(1, self._interval_seconds(iv_src)))
            tf_ratio = float(src_sec / max(chart_sec, 1.0))
            proj_total_cap = float(max_total_move)
            proj_step_cap = float(max_step_move)
            conservative_projection = False
            if tf_ratio >= 8.0:
                conservative_projection = True
                if iv_chart == "1m":
                    proj_total_cap = min(proj_total_cap, 0.008)
                    proj_step_cap = min(proj_step_cap, 0.0025)
                elif iv_chart == "5m":
                    proj_total_cap = min(proj_total_cap, 0.012)
                    proj_step_cap = min(proj_step_cap, 0.0040)
                else:
                    proj_total_cap = min(proj_total_cap, max_total_move * 0.72)
                    proj_step_cap = min(proj_step_cap, max_step_move * 0.70)

            src_curve: list[float] = [float(anchor)] + [float(v) for v in cleaned]
            # Clamp total move while preserving path curvature.
            raw_net = float(src_curve[-1] / max(anchor, 1e-8) - 1.0)
            net_ret = float(max(-proj_total_cap, min(proj_total_cap, raw_net)))
            if abs(raw_net) > 1e-8 and raw_net != net_ret:
                scale = float(net_ret / raw_net)
                for i in range(1, len(src_curve)):
                    src_ret = float(src_curve[i] / max(anchor, 1e-8) - 1.0)
                    src_curve[i] = float(anchor) * (1.0 + (src_ret * scale))
            # Hard-clip every source point into chart-safe movement band.
            lo_anchor = float(anchor) * (1.0 - proj_total_cap)
            hi_anchor = float(anchor) * (1.0 + proj_total_cap)
            for i in range(1, len(src_curve)):
                src_curve[i] = float(max(lo_anchor, min(hi_anchor, src_curve[i])))
            # Mild smoothing to suppress jagged daily->intraday interpolation artifacts.
            if len(src_curve) >= 4:
                smoothed = list(src_curve)
                for i in range(1, len(src_curve) - 1):
                    y0 = float(src_curve[i - 1])
                    y1 = float(src_curve[i])
                    y2 = float(src_curve[i + 1])
                    val = (0.18 * y0) + (0.64 * y1) + (0.18 * y2)
                    smoothed[i] = float(max(lo_anchor, min(hi_anchor, val)))
                src_curve = smoothed

            n_src = len(src_curve)
            src_x: list[float] = [0.0]
            if n_src <= 2:
                src_x.append(float(steps))
            else:
                step_span = float(steps) / float(max(1, n_src - 1))
                for i in range(1, n_src):
                    src_x.append(float(i) * step_span)

            projected_out: list[float] = []
            prev = float(anchor)
            seg = 0
            for i in range(1, steps + 1):
                x = float(i)
                while seg + 1 < len(src_x) and x > src_x[seg + 1]:
                    seg += 1
                if seg + 1 >= len(src_x):
                    target_px = float(src_curve[-1])
                else:
                    x0 = float(src_x[seg])
                    x1 = float(src_x[seg + 1])
                    y0 = float(src_curve[seg])
                    y1 = float(src_curve[seg + 1])
                    if x1 <= x0:
                        target_px = y1
                    else:
                        frac = (x - x0) / (x1 - x0)
                        target_px = y0 + ((y1 - y0) * frac)
                target_px = float(max(lo_anchor, min(hi_anchor, target_px)))

                step_ret = float(target_px / max(prev, 1e-8) - 1.0)
                if step_ret > proj_step_cap:
                    target_px = float(prev) * (1.0 + proj_step_cap)
                elif step_ret < -proj_step_cap:
                    target_px = float(prev) * (1.0 - proj_step_cap)
                target_px = float(max(lo_anchor, min(hi_anchor, target_px)))
                projected_out.append(float(target_px))
                prev = float(target_px)
            self._debug_console(
                f"forecast_display_project:{self._ui_norm(symbol)}:{iv_chart}",
                (
                    f"forecast display projection for {self._ui_norm(symbol)}: "
                    f"source={iv_src} chart={iv_chart} steps={steps} "
                    f"src_points={len(cleaned)} net={net_ret:+.2%} "
                    f"tf_ratio={tf_ratio:.1f} conservative={int(conservative_projection)}"
                ),
                min_gap_seconds=3.0,
                level="info",
            )
            return projected_out

        # Same-interval mode: clamp step spikes, keep model shape.
        out: list[float] = []
        prev = float(anchor)
        lo_anchor = float(anchor) * (1.0 - max_total_move)
        hi_anchor = float(anchor) * (1.0 + max_total_move)
        for p in cleaned[:steps]:
            px = float(max(lo_anchor, min(hi_anchor, float(p))))
            step_ret = float(px / max(prev, 1e-8) - 1.0)
            if step_ret > max_step_move:
                px = float(prev) * (1.0 + max_step_move)
            elif step_ret < -max_step_move:
                px = float(prev) * (1.0 - max_step_move)
            px = float(max(lo_anchor, min(hi_anchor, px)))
            out.append(float(px))
            prev = float(px)

        if len(out) >= 4:
            dirs = [1 if out[i] >= out[i - 1] else -1 for i in range(1, len(out))]
            flips = sum(1 for i in range(1, len(dirs)) if dirs[i] != dirs[i - 1])
            flip_ratio = float(flips) / float(max(1, len(dirs) - 1))
            raw_steps = [
                abs(float(out[i]) / max(float(out[i - 1]), 1e-8) - 1.0)
                for i in range(1, len(out))
            ]
            max_raw_step = max(raw_steps) if raw_steps else 0.0
            if flip_ratio > 0.65 or max_raw_step > (max_step_move * 1.35):
                smooth = list(out)
                for i in range(1, len(out) - 1):
                    val = (
                        (0.22 * float(out[i - 1]))
                        + (0.56 * float(out[i]))
                        + (0.22 * float(out[i + 1]))
                    )
                    smooth[i] = float(max(lo_anchor, min(hi_anchor, val)))

                out2: list[float] = []
                prev2 = float(anchor)
                for px in smooth:
                    px2 = float(max(lo_anchor, min(hi_anchor, float(px))))
                    step_ret2 = float(px2 / max(prev2, 1e-8) - 1.0)
                    if step_ret2 > max_step_move:
                        px2 = float(prev2) * (1.0 + max_step_move)
                    elif step_ret2 < -max_step_move:
                        px2 = float(prev2) * (1.0 - max_step_move)
                    px2 = float(max(lo_anchor, min(hi_anchor, px2)))
                    out2.append(float(px2))
                    prev2 = float(px2)
                out = out2
        return out

    def _chart_prediction_uncertainty_profile(
        self,
        symbol: str,
    ) -> tuple[float, float, float]:
        """
        Resolve (uncertainty, tail_risk, confidence) for chart forecast bands.
        """
        uncertainty = 0.55
        tail_risk = 0.55
        confidence = 0.45

        pred = getattr(self, "current_prediction", None)
        if pred and self._ui_norm(getattr(pred, "stock_code", "")) == self._ui_norm(symbol):
            try:
                uncertainty = float(
                    np.clip(getattr(pred, "uncertainty_score", uncertainty), 0.0, 1.0)
                )
            except Exception as exc:
                log.debug("Suppressed exception in ui/app.py", exc_info=exc)
            try:
                tail_risk = float(
                    np.clip(getattr(pred, "tail_risk_score", tail_risk), 0.0, 1.0)
                )
            except Exception as exc:
                log.debug("Suppressed exception in ui/app.py", exc_info=exc)
            try:
                confidence = float(
                    np.clip(getattr(pred, "confidence", confidence), 0.0, 1.0)
                )
            except Exception as exc:
                log.debug("Suppressed exception in ui/app.py", exc_info=exc)

        return uncertainty, tail_risk, confidence

    def _build_chart_prediction_bands(
        self,
        *,
        symbol: str,
        predicted_prices: list[float] | None,
        anchor_price: float | None,
    ) -> tuple[list[float], list[float]]:
        """Build chart uncertainty envelope around predicted prices."""
        vals = []
        for v in self._safe_list(predicted_prices):
            try:
                fv = float(v)
            except Exception:
                continue
            if fv > 0 and math.isfinite(fv):
                vals.append(float(fv))
        if not vals:
            return [], []

        try:
            anchor = float(anchor_price or 0.0)
        except Exception:
            anchor = 0.0
        if anchor <= 0:
            anchor = float(vals[0])

        uncertainty, tail_risk, confidence = self._chart_prediction_uncertainty_profile(
            symbol
        )
        base_width = float(
            np.clip(
                0.004
                + (0.020 * uncertainty)
                + (0.015 * tail_risk)
                + (0.012 * (1.0 - confidence)),
                0.004,
                0.24,
            )
        )

        n = max(1, len(vals))
        lows: list[float] = []
        highs: list[float] = []
        for i, px in enumerate(vals, start=1):
            growth = 1.0 + (float(i) / float(n)) * (0.90 + (0.70 * uncertainty))
            width = float(np.clip(base_width * growth, 0.004, 0.32))

            lo = max(0.01, float(px) * (1.0 - width))
            hi = max(lo + 1e-6, float(px) * (1.0 + width))

            # Keep envelope centered on plausible anchor neighborhood.
            if anchor > 0:
                lo = max(lo, anchor * 0.50)
                hi = min(hi, anchor * 1.50)
                if hi <= lo:
                    hi = lo + max(1e-6, abs(lo) * 0.002)

            lows.append(float(lo))
            highs.append(float(hi))

        return lows, highs

    def _resolve_chart_prediction_series(
        self,
        *,
        symbol: str,
        fallback_interval: str,
        predicted_prices: list[float] | None = None,
        source_interval: str | None = None,
    ) -> tuple[list[float], str]:
        """Resolve prediction series/source interval for chart rendering."""
        iv_fallback = self._normalize_interval_token(fallback_interval)
        iv_source = self._normalize_interval_token(
            source_interval,
            fallback=iv_fallback,
        )
        if predicted_prices is not None:
            return self._safe_list(predicted_prices), iv_source

        if (
            self.current_prediction
            and getattr(self.current_prediction, "stock_code", "") == symbol
        ):
            vals = (
                getattr(self.current_prediction, "predicted_prices", [])
                or []
            )
            iv_source = self._normalize_interval_token(
                getattr(self.current_prediction, "interval", iv_source),
                fallback=iv_source,
            )
            return list(vals), iv_source
        return [], iv_source

    def _render_chart_state(
        self,
        *,
        symbol: str,
        interval: str,
        bars: list[dict[str, Any]] | None,
        context: str,
        current_price: float | None = None,
        predicted_prices: list[float] | None = None,
        source_interval: str | None = None,
        target_steps: int | None = None,
        predicted_prepared: bool = False,
        update_latest_label: bool = False,
        allow_legacy_candles: bool = False,
        reset_view_on_symbol_switch: bool = False,
    ) -> list[dict[str, Any]]:
        """
        Unified chart rendering path used by bar/tick/analysis updates.
        """
        iv = self._normalize_interval_token(interval)
        arr = self._safe_list(bars)

        anchor_input: float | None = None
        if current_price is not None:
            try:
                px = float(current_price)
                if px > 0 and math.isfinite(px):
                    anchor_input = px
            except Exception:
                anchor_input = None

        chart_anchor = self._effective_anchor_price(symbol, anchor_input)
        arr = self._scrub_chart_bars(
            arr,
            iv,
            symbol=symbol,
            anchor_price=chart_anchor if chart_anchor > 0 else None,
        )
        arr = self._stabilize_chart_depth(symbol, iv, arr)
        self._bars_by_symbol[symbol] = arr
        self._debug_candle_quality(
            symbol=symbol,
            interval=iv,
            bars=arr,
            context=context,
        )

        pred_vals, pred_source_iv = self._resolve_chart_prediction_series(
            symbol=symbol,
            fallback_interval=iv,
            predicted_prices=predicted_prices,
            source_interval=source_interval,
        )
        try:
            steps = int(
                target_steps if target_steps is not None else self.forecast_spin.value()
            )
        except Exception:
            steps = int(self.forecast_spin.value())

        anchor_for_pred: float | None = None
        if arr:
            try:
                last_close = float(arr[-1].get("close", 0) or 0)
                if last_close > 0 and math.isfinite(last_close):
                    anchor_for_pred = last_close
            except Exception:
                anchor_for_pred = None
        if anchor_for_pred is None:
            anchor_for_pred = anchor_input

        source_iv_for_prepare = iv if predicted_prepared else pred_source_iv
        chart_predicted = self._prepare_chart_predicted_prices(
            symbol=symbol,
            chart_interval=iv,
            predicted_prices=pred_vals,
            source_interval=source_iv_for_prepare,
            current_price=anchor_for_pred,
            target_steps=steps,
        )
        chart_predicted_low, chart_predicted_high = self._build_chart_prediction_bands(
            symbol=symbol,
            predicted_prices=chart_predicted,
            anchor_price=anchor_for_pred,
        )
        self._debug_forecast_quality(
            symbol=symbol,
            chart_interval=iv,
            source_interval=pred_source_iv,
            predicted_prices=chart_predicted,
            anchor_price=anchor_for_pred,
            context=context,
        )

        if (
            reset_view_on_symbol_switch
            and self._chart_symbol
            and self._chart_symbol != symbol
        ):
            try:
                self.chart.reset_view()
            except Exception as exc:
                log.debug("Suppressed exception in ui/app.py", exc_info=exc)

        rendered = False
        if hasattr(self.chart, "update_chart"):
            self.chart.update_chart(
                arr,
                predicted_prices=chart_predicted,
                predicted_prices_low=chart_predicted_low,
                predicted_prices_high=chart_predicted_high,
                levels=self._get_levels_dict(),
            )
            self._debug_chart_state(
                symbol=symbol,
                interval=iv,
                bars=arr,
                predicted_prices=chart_predicted,
                context=context,
            )
            self._chart_symbol = symbol
            rendered = True
        elif allow_legacy_candles and hasattr(self.chart, "update_candles"):
            self.chart.update_candles(
                arr,
                predicted_prices=chart_predicted,
                predicted_prices_low=chart_predicted_low,
                predicted_prices_high=chart_predicted_high,
                levels=self._get_levels_dict(),
            )
            self._chart_symbol = symbol
            rendered = True

        if update_latest_label:
            label_price: float | None = None
            if anchor_input is not None:
                label_price = anchor_input
            self._update_chart_latest_label(
                symbol,
                bar=arr[-1] if arr else None,
                price=label_price,
            )
        if not rendered and not update_latest_label:
            # Keep the return side-effect free when no renderer is available.
            return arr
        return arr

    def _setup_menubar(self) -> None:
        """Setup professional menu bar"""
        menubar = self.menuBar()

        file_menu = menubar.addMenu("&File")

        new_action = QAction("&New Workspace", self)
        new_action.setShortcut("Ctrl+N")
        file_menu.addAction(new_action)

        file_menu.addSeparator()

        exit_action = QAction("E&xit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        trading_menu = menubar.addMenu("&Trading")

        connect_action = QAction("&Connect Broker", self)
        connect_action.triggered.connect(self._toggle_trading)
        trading_menu.addAction(connect_action)

        trading_menu.addSeparator()

        self.paper_action = QAction("&Paper Trading Mode", self)
        self.paper_action.setCheckable(True)
        self.live_action = QAction("&Live Trading Mode", self)
        self.live_action.setCheckable(True)

        mode_group = QActionGroup(self)
        mode_group.setExclusive(True)
        mode_group.addAction(self.paper_action)
        mode_group.addAction(self.live_action)

        self.paper_action.triggered.connect(
            lambda checked: self._set_trading_mode(TradingMode.SIMULATION) if checked else None
        )
        self.live_action.triggered.connect(
            lambda checked: self._set_trading_mode(TradingMode.LIVE) if checked else None
        )
        trading_menu.addAction(self.paper_action)
        trading_menu.addAction(self.live_action)

        ai_menu = menubar.addMenu("&AI Model")

        train_action = QAction("&Train Model", self)
        train_action.setShortcut("Ctrl+T")
        train_action.triggered.connect(self._start_training)
        ai_menu.addAction(train_action)

        auto_learn_action = QAction("&Auto Learn", self)
        auto_learn_action.triggered.connect(self._show_auto_learn)
        ai_menu.addAction(auto_learn_action)

        strategy_market_action = QAction("&Strategy Marketplace", self)
        strategy_market_action.triggered.connect(self._show_strategy_marketplace)
        ai_menu.addAction(strategy_market_action)

        ai_menu.addSeparator()

        backtest_action = QAction("&Backtest", self)
        backtest_action.triggered.connect(self._show_backtest)
        ai_menu.addAction(backtest_action)

        view_menu = menubar.addMenu("&View")

        refresh_action = QAction("&Refresh", self)
        refresh_action.setShortcut("F5")
        refresh_action.triggered.connect(self._refresh_all)
        view_menu.addAction(refresh_action)

        help_menu = menubar.addMenu("&Help")

        about_action = QAction("&About", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)

    def _setup_toolbar(self) -> None:
        """Setup professional toolbar with auto-trade controls"""
        toolbar = QToolBar("Main Toolbar")
        toolbar.setIconSize(QSize(24, 24))
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        self.analyze_action = QAction("Analyze", self)
        self.analyze_action.triggered.connect(self._analyze_stock)
        toolbar.addAction(self.analyze_action)

        toolbar.addSeparator()

        # Real-time monitoring toggle
        self.monitor_action = QAction("Start Monitoring", self)
        self.monitor_action.setCheckable(True)
        self.monitor_action.triggered.connect(self._toggle_monitoring)
        toolbar.addAction(self.monitor_action)

        toolbar.addSeparator()

        scan_action = QAction("Scan Market", self)
        scan_action.triggered.connect(self._scan_stocks)
        toolbar.addAction(scan_action)

        toolbar.addSeparator()

        # === AUTO-TRADE CONTROLS ===
        toolbar.addWidget(QLabel("  Mode: "))
        self.trade_mode_combo = QComboBox()
        self.trade_mode_combo.addItems(["Manual", "Auto", "Semi-Auto"])
        self.trade_mode_combo.setCurrentIndex(0)
        self.trade_mode_combo.setFixedWidth(110)
        self.trade_mode_combo.setToolTip(
            "Manual: Click to trade\n"
            "Auto: AI trades automatically\n"
            "Semi-Auto: AI suggests, you approve"
        )
        self.trade_mode_combo.currentIndexChanged.connect(
            self._on_trade_mode_changed
        )
        toolbar.addWidget(self.trade_mode_combo)

        # Auto-trade status indicator
        self.auto_trade_status_label = QLabel("  MANUAL  ")
        self.auto_trade_status_label.setStyleSheet(
            "color: #aac3ec; font-weight: bold; padding: 0 8px;"
        )
        toolbar.addWidget(self.auto_trade_status_label)

        # Auto-trade settings button
        auto_settings_action = QAction("Auto Settings", self)
        auto_settings_action.triggered.connect(self._show_auto_trade_settings)
        toolbar.addAction(auto_settings_action)

        toolbar.addSeparator()

        spacer = QWidget()
        spacer.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred
        )
        toolbar.addWidget(spacer)

        toolbar.addWidget(QLabel("  Stock: "))
        self.stock_input = QLineEdit()
        self.stock_input.setPlaceholderText("Enter code (e.g., 600519)")
        self.stock_input.setFixedWidth(150)
        self.stock_input.returnPressed.connect(self._analyze_stock)
        toolbar.addWidget(self.stock_input)

        # =========================================================================
    # =========================================================================

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

        except Exception as e:
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
        except Exception:
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
        except Exception:
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
            except Exception:
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
        except Exception:
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
            except Exception:
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
            except Exception:
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
        except Exception:
            vol_val = 0.0
        if (not math.isfinite(vol_val)) or vol_val < 0:
            vol_val = 0.0

        try:
            amt_val = float(bar.get("amount", 0) or 0.0)
        except Exception:
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
            except Exception as exc:
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
                    except Exception:
                        e_open = 0.0
                    try:
                        e_high = float(existing.get("high", c) or c)
                    except Exception:
                        e_high = c
                    try:
                        e_low = float(existing.get("low", c) or c)
                    except Exception:
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
                        except Exception:
                            e_vol = 0.0
                        try:
                            n_vol = float(norm_bar.get("volume", 0) or 0.0)
                        except Exception:
                            n_vol = 0.0
                        merged["volume"] = float(max(0.0, e_vol) + max(0.0, n_vol))
                    if ("amount" in norm_bar) or ("amount" in existing):
                        try:
                            e_amt = float(existing.get("amount", 0) or 0.0)
                        except Exception:
                            e_amt = 0.0
                        try:
                            n_amt = float(norm_bar.get("amount", 0) or 0.0)
                        except Exception:
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
                except Exception as exc:
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
        self._persist_session_bar(
            symbol,
            interval,
            norm_bar,
            channel="bar_ui",
            min_gap_seconds=min_gap,
        )

        current_code = self._ui_norm(self.stock_input.text())
        if current_code != symbol:
            return

        self._render_live_bar_update(
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
        except Exception as e:
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
        except Exception as exc:
            log.debug("Suppressed exception in ui/app.py", exc_info=exc)

    def _setup_ui(self) -> None:
        """Setup main UI with professional layout"""
        central = QWidget()
        self.setCentralWidget(central)

        layout = QHBoxLayout(central)
        layout.setSpacing(10)
        layout.setContentsMargins(10, 10, 10, 10)

        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        main_splitter.setChildrenCollapsible(False)

        # Left Panel - Control & Watchlist
        left_panel = self._create_left_panel()
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setFrameShape(QFrame.Shape.NoFrame)
        left_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        left_scroll.setWidget(left_panel)
        left_scroll.setMinimumWidth(240)
        left_scroll.setMaximumWidth(340)

        # Center Panel - Charts & Signals
        center_panel = self._create_center_panel()

        # Right Panel - Portfolio & Orders
        right_panel = self._create_right_panel()

        main_splitter.addWidget(left_scroll)
        main_splitter.addWidget(center_panel)
        main_splitter.addWidget(right_panel)
        main_splitter.setStretchFactor(0, 0)
        main_splitter.setStretchFactor(1, 1)
        main_splitter.setStretchFactor(2, 0)
        main_splitter.setSizes([280, 760, 360])

        layout.addWidget(main_splitter)

    def _create_left_panel(self) -> QWidget:
        return _create_left_panel_impl(self)

    def _make_table(
        self, headers: list[str], max_height: int | None = None
    ) -> QTableWidget:
        table = QTableWidget()
        table.setColumnCount(len(headers))
        table.setHorizontalHeaderLabels(headers)
        table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        table.verticalHeader().setVisible(False)
        table.setAlternatingRowColors(True)
        table.setShowGrid(True)
        table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        if max_height is not None:
            table.setMaximumHeight(int(max_height))
        return table

    def _add_labeled(
        self,
        layout: QGridLayout,
        row: int,
        text: str,
        widget: QWidget,
    ) -> None:
        layout.addWidget(QLabel(text), row, 0)
        layout.addWidget(widget, row, 1)

    def _build_stat_frame(
        self,
        labels: list[tuple[str, str, int, int]],
        value_style: str,
        padding: int = 15,
    ) -> tuple[QFrame, dict[str, QLabel]]:
        frame = QFrame()
        frame.setStyleSheet(
            "QFrame {"
            "background: #111c31;"
            "border: 1px solid #243454;"
            f"border-radius: 10px; padding: {int(padding)}px;"
            "}"
        )
        grid = QGridLayout(frame)
        out = {}
        for key, text, row, col in labels:
            container = QWidget()
            cont_layout = QVBoxLayout(container)
            cont_layout.setContentsMargins(5, 5, 5, 5)
            title = QLabel(text)
            title.setStyleSheet("color: #888; font-size: 11px;")
            value = QLabel("--")
            value.setStyleSheet(value_style)
            cont_layout.addWidget(title)
            cont_layout.addWidget(value)
            grid.addWidget(container, row, col)
            out[key] = value
        return frame, out

    def _create_center_panel(self) -> QWidget:
        """Create center panel with charts and signals"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setSpacing(10)

        # Signal Display - lazy import
        try:
            from .widgets import SignalPanel
            self.signal_panel = SignalPanel()
        except ImportError:
            self.signal_panel = QLabel("Signal Panel")
            self.signal_panel.setMinimumHeight(72)
        self.signal_panel.setMinimumHeight(120)
        self.signal_panel.setMaximumHeight(170)
        self.signal_panel.setSizePolicy(
            QSizePolicy.Policy.Preferred,
            QSizePolicy.Policy.Fixed,
        )
        layout.addWidget(self.signal_panel)

        chart_group = QGroupBox("Price Chart and AI Prediction")
        chart_layout = QVBoxLayout()

        try:
            from .charts import StockChart
            self.chart = StockChart()
            self.chart.setMinimumHeight(260)
            if hasattr(self.chart, "trade_requested"):
                self.chart.trade_requested.connect(self._on_chart_trade_requested)
        except ImportError:
            self.chart = QLabel("Chart (charts module not found)")
            self.chart.setMinimumHeight(260)
            self.chart.setAlignment(Qt.AlignmentFlag.AlignCenter)
        chart_layout.addWidget(self.chart)

        chart_actions = QHBoxLayout()
        self.zoom_in_btn = QPushButton("Zoom In")
        self.zoom_out_btn = QPushButton("Zoom Out")
        self.zoom_reset_btn = QPushButton("Reset View")
        self.zoom_in_btn.setMaximumWidth(110)
        self.zoom_out_btn.setMaximumWidth(110)
        self.zoom_reset_btn.setMaximumWidth(120)
        self.zoom_in_btn.clicked.connect(self._zoom_chart_in)
        self.zoom_out_btn.clicked.connect(self._zoom_chart_out)
        self.zoom_reset_btn.clicked.connect(self._zoom_chart_reset)
        chart_actions.addWidget(self.zoom_in_btn)
        chart_actions.addWidget(self.zoom_out_btn)
        chart_actions.addWidget(self.zoom_reset_btn)

        overlay_specs = [
            ("SMA20", "sma20", True),
            ("SMA50", "sma50", True),
            ("SMA200", "sma200", False),
            ("EMA21", "ema21", True),
            ("EMA55", "ema55", False),
            ("BBands", "bbands", True),
            ("VWAP20", "vwap20", True),
        ]
        self._chart_overlay_checks: dict[str, QCheckBox] = {}
        for label, key, default_enabled in overlay_specs:
            chk = QCheckBox(label)
            chk.setChecked(bool(default_enabled))
            chk.toggled.connect(
                lambda v, overlay_key=key: self._set_chart_overlay(
                    overlay_key,
                    v,
                )
            )
            self._chart_overlay_checks[key] = chk
            chart_actions.addWidget(chk)

        chart_actions.addStretch(1)
        chart_layout.addLayout(chart_actions)

        self.chart_latest_label = QLabel("Latest --")
        self.chart_latest_label.setStyleSheet("color: #9aa4b2; font-size: 11px;")
        chart_layout.addWidget(self.chart_latest_label)

        chart_group.setLayout(chart_layout)
        layout.addWidget(chart_group)

        details_group = QGroupBox("Analysis Details")
        details_layout = QVBoxLayout()

        self.details_text = QTextEdit()
        self.details_text.setReadOnly(True)
        self.details_text.setFont(QFont("Consolas", 10))
        self.details_text.setMaximumHeight(120)
        details_layout.addWidget(self.details_text)

        details_group.setLayout(details_layout)
        layout.addWidget(details_group)

        return panel

    def _zoom_chart_in(self) -> None:
        if hasattr(self.chart, "zoom_in"):
            try:
                self.chart.zoom_in()
            except Exception as exc:
                log.debug("Suppressed exception in ui/app.py", exc_info=exc)

    def _zoom_chart_out(self) -> None:
        if hasattr(self.chart, "zoom_out"):
            try:
                self.chart.zoom_out()
            except Exception as exc:
                log.debug("Suppressed exception in ui/app.py", exc_info=exc)

    def _zoom_chart_reset(self) -> None:
        if hasattr(self.chart, "reset_view"):
            try:
                self.chart.reset_view()
            except Exception as exc:
                log.debug("Suppressed exception in ui/app.py", exc_info=exc)

    def _set_chart_overlay(self, key: str, enabled: bool) -> None:
        if hasattr(self.chart, "set_overlay_enabled"):
            try:
                self.chart.set_overlay_enabled(str(key), bool(enabled))
            except Exception as exc:
                log.debug("Suppressed exception in ui/app.py", exc_info=exc)

    def _create_right_panel(self) -> QWidget:
        return _create_right_panel_impl(self)

        # =========================================================================
        # STATUS BAR & TIMERS
    # =========================================================================

    def _setup_statusbar(self) -> None:
        """Setup status bar"""
        self._status_bar = QStatusBar()
        self.setStatusBar(self._status_bar)

        self.progress = QProgressBar()
        self.progress.setMaximumWidth(200)
        self.progress.setMaximumHeight(15)
        self.progress.hide()
        self._status_bar.addWidget(self.progress)

        self.status_label = QLabel("Ready")
        self._status_bar.addWidget(self.status_label)

        self.market_label = QLabel("")
        self._status_bar.addPermanentWidget(self.market_label)

        self.monitor_label = QLabel("Monitoring: OFF")
        self.monitor_label.setStyleSheet("color: #888;")
        self._status_bar.addWidget(self.monitor_label)

        self.time_label = QLabel("")
        self._status_bar.addWidget(self.time_label)

    def _setup_timers(self) -> None:
        """Setup update timers"""
        self.clock_timer = QTimer()
        self.clock_timer.timeout.connect(self._update_clock)
        self.clock_timer.start(1000)

        self.market_timer = QTimer()
        self.market_timer.timeout.connect(self._update_market_status)
        self.market_timer.start(60000)

        self.portfolio_timer = QTimer()
        self.portfolio_timer.timeout.connect(self._refresh_portfolio)
        self.portfolio_timer.start(5000)

        self.watchlist_timer = QTimer()
        self.watchlist_timer.timeout.connect(self._update_watchlist)
        self.watchlist_timer.start(30000)

        # Auto-trade UI refresh
        self.auto_trade_timer = QTimer()
        self.auto_trade_timer.timeout.connect(self._refresh_auto_trade_ui)
        self.auto_trade_timer.start(2000)

        # Live chart refresh: keep real + guessed lines moving.
        self.chart_live_timer = QTimer()
        self.chart_live_timer.timeout.connect(self._refresh_live_chart_forecast)
        self.chart_live_timer.start(1500)

        self._update_market_status()

        # =========================================================================
    # =========================================================================

    def _apply_professional_style(self) -> None:
        _apply_professional_style_impl(self)

    # =========================================================================
    # =========================================================================

    def _init_components(self) -> None:
        """Initialize trading components"""
        try:
            Predictor = _lazy_get("models.predictor", "Predictor")

            interval = str(self.STARTUP_INTERVAL).strip().lower()
            self.interval_combo.blockSignals(True)
            try:
                self.interval_combo.setCurrentText(interval)
            finally:
                self.interval_combo.blockSignals(False)
            # Always start with 30-step guess horizon for live chart forecasting.
            self.forecast_spin.setValue(int(self.GUESS_FORECAST_BARS))
            self.lookback_spin.setValue(self._recommended_lookback(interval))
            horizon = int(self.forecast_spin.value())

            self.predictor = Predictor(
                capital=self.capital_spin.value(),
                interval=interval,
                prediction_horizon=horizon
            )

            if self.predictor.ensemble:
                num_models = len(self.predictor.ensemble.models)
                self.model_status.setText(
                    f"Model: Loaded ({num_models} networks)"
                )
                self.model_status.setStyleSheet("color: #4CAF50;")
                self._sync_ui_to_loaded_model(
                    interval,
                    horizon,
                    preserve_requested_interval=True,
                )
                self._log_model_alignment_debug(
                    context="startup",
                    requested_interval=interval,
                    requested_horizon=horizon,
                )
                self._update_trained_stocks_ui()
                self.log("AI model loaded successfully", "success")
            else:
                self.model_status.setText("Model: Not trained")
                self.model_status.setStyleSheet("color: #FFD54F;")
                self.model_info.setText(
                    "Train a model to enable predictions"
                )
                self._update_trained_stocks_ui([])
                self.log(
                    "No trained model found. Please train a model.", "warning"
                )

        except Exception as e:
            log.error(f"Failed to load model: {e}")
            self.log(f"Failed to load model: {e}", "error")
            self.predictor = None
            self.model_status.setText("Model: Error")
            self.model_status.setStyleSheet("color: #F44336;")
            self._update_trained_stocks_ui([])

        # Initialize auto-trader on executor if available
        self._init_auto_trader()

        # Auto-start live monitor when model is available.
        if self.predictor is not None and self.predictor.ensemble is not None:
            try:
                self.monitor_action.setChecked(True)
                self._start_monitoring()
            except Exception as e:
                log.debug(f"Auto-start monitoring failed: {e}")

        if self._debug_console_enabled:
            self.log(
                "Debug console enabled (set TRADING_DEBUG_CONSOLE=0 to disable)",
                "warning",
            )
        self.log("System initialized - Ready for trading", "info")

    def _get_trained_stock_codes(self) -> list[str]:
        """Read trained stock list from loaded predictor metadata."""
        if self.predictor is None:
            return []
        try:
            fn = getattr(self.predictor, "get_trained_stock_codes", None)
            if callable(fn):
                out = fn()
                if isinstance(out, list):
                    return [
                        str(x).strip()
                        for x in out
                        if str(x).strip()
                    ]
        except Exception as exc:
            log.debug("Suppressed exception in ui/app.py", exc_info=exc)
        return []

    def _sync_trained_stock_last_train_from_model(self) -> None:
        """Use loaded model artifacts as source-of-truth for last-train metadata."""
        if self.predictor is None:
            return
        fn = getattr(self.predictor, "get_trained_stock_last_train", None)
        if not callable(fn):
            return
        try:
            raw = fn()
        except Exception:
            return
        if not isinstance(raw, dict):
            return
        out: dict[str, str] = {}
        for k, v in raw.items():
            code = self._ui_norm(str(k or ""))
            if not code:
                continue
            ts = str(v or "").strip()
            if not ts:
                continue
            out[code] = ts
        if out == dict(self._trained_stock_last_train or {}):
            return
        self._trained_stock_last_train = out
        self._save_trained_stock_last_train_meta()

    def _get_trained_stock_set(self) -> set[str]:
        """Normalized trained stock set from metadata cache/predictor."""
        raw = list(getattr(self, "_trained_stock_codes_cache", []) or [])
        if not raw and self.predictor is not None:
            raw = self._get_trained_stock_codes()
            if raw:
                self._trained_stock_codes_cache = list(raw)
        out: set[str] = set()
        for item in raw:
            code = self._ui_norm(item)
            if code:
                out.add(code)
        return out

    def _is_trained_stock(self, symbol: str) -> bool:
        """Whether symbol is part of the currently loaded trained stock set."""
        code = self._ui_norm(symbol)
        if not code:
            return False
        return code in self._get_trained_stock_set()

    def _persist_session_bar(
        self,
        symbol: str,
        interval: str,
        bar: dict[str, Any] | None,
        *,
        channel: str = "tick",
        min_gap_seconds: float = 0.9,
    ) -> None:
        """Persist latest live bar snapshot to session cache."""
        if self._session_bar_cache is None or not isinstance(bar, dict):
            return
        try:
            iv = self._normalize_interval_token(interval)
            # Persist session bars only in canonical 1m stream.
            # Coarser intervals are display-only and must be derived from 1m.
            if iv != "1m":
                return
            now_ts = time.time()
            key = f"{symbol}:{iv}:{channel}"
            min_gap = float(max(0.0, min_gap_seconds))
            lock = getattr(self, "_session_cache_write_lock", None)
            if lock is None:
                prev_ts = float(self._last_session_cache_write_ts.get(key, 0.0))
                if (now_ts - prev_ts) < min_gap:
                    return
                self._last_session_cache_write_ts[key] = now_ts
            else:
                with lock:
                    prev_ts = float(self._last_session_cache_write_ts.get(key, 0.0))
                    if (now_ts - prev_ts) < min_gap:
                        return
                    self._last_session_cache_write_ts[key] = now_ts
            payload = dict(bar)
            payload["interval"] = iv
            payload["source"] = str(payload.get("source", "") or "tencent_rt")
            submit = getattr(self, "_submit_session_cache_write", None)
            if callable(submit):
                submit(symbol, iv, payload)
            else:
                self._session_bar_cache.append_bar(symbol, iv, payload)
        except Exception as e:
            log.debug(f"Session cache persist failed for {symbol}: {e}")

    def _submit_session_cache_write(
        self,
        symbol: str,
        interval: str,
        payload: dict[str, Any],
    ) -> None:
        cache = self._session_bar_cache
        pool = getattr(self, "_session_cache_io_pool", None)
        if cache is None:
            return
        if pool is None:
            try:
                cache.append_bar(symbol, interval, dict(payload))
            except Exception as exc:
                log.debug("Session cache write failed for %s: %s", symbol, exc)
            return
        lock = getattr(self, "_session_cache_io_lock", None)
        futures = getattr(self, "_session_cache_io_futures", None)
        if lock is None or futures is None:
            try:
                pool.submit(cache.append_bar, symbol, interval, dict(payload))
            except Exception as exc:
                log.debug("Failed to enqueue session cache write for %s: %s", symbol, exc)
            return
        with lock:
            if len(futures) >= 256:
                log.debug(
                    "Session cache write queue full; dropping %s (%s)",
                    symbol,
                    interval,
                )
                return
        try:
            future = pool.submit(cache.append_bar, symbol, interval, dict(payload))
        except Exception as exc:
            log.debug("Failed to enqueue session cache write for %s: %s", symbol, exc)
            return
        with lock:
            futures.add(future)
        future.add_done_callback(self._on_session_cache_write_done)

    def _on_session_cache_write_done(self, future: Future[object]) -> None:
        lock = getattr(self, "_session_cache_io_lock", None)
        futures = getattr(self, "_session_cache_io_futures", None)
        if lock is not None and futures is not None:
            with lock:
                futures.discard(future)
        try:
            future.result()
        except Exception as exc:
            log.debug("Async session cache write failed: %s", exc)

    def _shutdown_session_cache_writer(self) -> None:
        pool = getattr(self, "_session_cache_io_pool", None)
        if pool is None:
            return
        lock = getattr(self, "_session_cache_io_lock", None)
        futures = getattr(self, "_session_cache_io_futures", None)
        if lock is not None and futures is not None:
            with lock:
                pending = list(futures)
        else:
            pending = []
        for fut in pending:
            try:
                fut.result(timeout=0.3)
            except FuturesTimeout:
                break
            except Exception as exc:
                log.debug("Session cache writer flush failed: %s", exc)
        try:
            pool.shutdown(wait=False, cancel_futures=True)
        except Exception as exc:
            log.debug("Session cache writer shutdown failed: %s", exc)
        self._session_cache_io_pool = None

    def _filter_trained_stocks_ui(self, text: str) -> None:
        """Filter right-panel trained stock list by search query."""
        self._refresh_trained_stock_list(
            list(getattr(self, "_trained_stock_codes_cache", [])),
            str(text or ""),
        )

    def _pin_watchlist_symbol(self, code: str) -> None:
        """Ensure symbol is present and visible in watchlist (move to top)."""
        normalized = self._ui_norm(code)
        if not normalized:
            return

        changed = False
        if normalized in self.watch_list:
            if self.watch_list and self.watch_list[0] != normalized:
                self.watch_list.remove(normalized)
                self.watch_list.insert(0, normalized)
                changed = True
        else:
            if len(self.watch_list) >= self.MAX_WATCHLIST_SIZE:
                self.watch_list = self.watch_list[: self.MAX_WATCHLIST_SIZE - 1]
            self.watch_list.insert(0, normalized)
            changed = True

        if changed:
            self._update_watchlist()
            if self.executor and self.executor.auto_trader:
                try:
                    self.executor.auto_trader.update_watchlist(self.watch_list)
                except Exception as exc:
                    log.debug("Suppressed exception in ui/app.py", exc_info=exc)

        # Keep selection aligned with the active symbol.
        try:
            for row in range(self.watchlist.rowCount()):
                item = self.watchlist.item(row, 0)
                if item and self._ui_norm(item.text()) == normalized:
                    self.watchlist.setCurrentCell(row, 0)
                    break
        except Exception as exc:
            log.debug("Suppressed exception in ui/app.py", exc_info=exc)

    def _on_trained_stock_activated(self, item: QListWidgetItem) -> None:
        """Load selected trained stock from right-panel list."""
        if item is None:
            return
        code_hint = ""
        try:
            code_hint = str(
                item.data(Qt.ItemDataRole.UserRole) or ""
            ).strip()
        except Exception:
            code_hint = ""
        code = self._ui_norm(code_hint or item.text())
        if not code:
            return
        interval = self._normalize_interval_token(
            self.interval_combo.currentText()
        )
        try:
            self._queue_history_refresh(code, interval)
        except Exception as exc:
            log.debug("Suppressed exception in ui/app.py", exc_info=exc)
        try:
            target_lookback = int(
                max(
                    self._recommended_lookback(interval),
                    self._trained_stock_window_bars(interval),
                )
            )
            if int(self.lookback_spin.value()) < target_lookback:
                self.lookback_spin.setValue(target_lookback)
        except Exception as exc:
            log.debug("Suppressed exception in ui/app.py", exc_info=exc)
        self._pin_watchlist_symbol(code)
        self.stock_input.setText(code)
        try:
            self._ensure_feed_subscription(code)
        except Exception as exc:
            log.debug("Suppressed exception in ui/app.py", exc_info=exc)
        self._on_watchlist_click(-1, -1, code_override=code)

    def _refresh_trained_stock_list(
        self, stocks: list[str], query: str = ""
    ) -> None:
        """Render searchable trained-stock list in the right panel."""
        if not hasattr(self, "trained_stock_list"):
            return
        all_codes = [
            str(x).strip()
            for x in list(stocks or [])
            if str(x).strip()
        ]
        q = str(query or "").strip().lower()
        if q:
            view_codes = [c for c in all_codes if q in c.lower()]
        else:
            view_codes = all_codes

        self.trained_stock_list.clear()
        if view_codes:
            for code in view_codes:
                last_train = str(
                    self._trained_stock_last_train.get(code, "")
                ).strip()
                last_text = self._format_last_train_text(last_train)
                item = QListWidgetItem(
                    f"{code}  | last train: {last_text}"
                )
                item.setData(Qt.ItemDataRole.UserRole, code)
                item.setToolTip(
                    f"Stock: {code}\n"
                    f"Last Train: {last_text}"
                )
                self.trained_stock_list.addItem(item)
        elif all_codes and q:
            self.trained_stock_list.addItem(
                "No matching trained stocks for current search."
            )
        else:
            self.trained_stock_list.addItem(
                "No trained stock metadata found in the loaded model."
            )

        if hasattr(self, "trained_stock_count_label"):
            self.trained_stock_count_label.setText(
                f"Trained: {len(view_codes)} / {len(all_codes)}"
            )

        tabs = getattr(self, "right_tabs", None)
        idx = int(getattr(self, "_trained_tab_index", -1))
        if tabs is not None and idx >= 0:
            tabs.setTabText(idx, f"Trained Stocks ({len(all_codes)})")

    def _update_trained_stocks_ui(self, codes: list[str] | None = None) -> None:
        """Refresh trained-stock metadata section in AI panel."""
        self._sync_trained_stock_last_train_from_model()
        stocks = list(codes) if isinstance(codes, list) else self._get_trained_stock_codes()
        self._trained_stock_codes_cache = list(stocks)

        if hasattr(self, "trained_stocks_label"):
            if not stocks:
                self.trained_stocks_label.setText("Trained Stocks: --")
            else:
                self.trained_stocks_label.setText(
                    f"Trained Stocks: {len(stocks)}"
                )

        query = ""
        if hasattr(self, "trained_stock_search"):
            try:
                query = self.trained_stock_search.text()
            except Exception:
                query = ""
        self._refresh_trained_stock_list(stocks, query=query)

        if not stocks:
            return

    def _focus_trained_stocks_tab(self) -> None:
        """Focus the right-panel trained-stocks tab."""
        tabs = getattr(self, "right_tabs", None)
        idx = int(getattr(self, "_trained_tab_index", -1))
        if tabs is not None and idx >= 0:
            tabs.setCurrentIndex(idx)

    def _get_infor_trained_stocks(self) -> None:
        """
        Refresh 29-day AKShare history for all trained stocks.

        If data already exists in the target window, only fetches from the
        last saved timestamp forward.
        """
        raw_codes = self._get_trained_stock_codes()
        codes = list(
            dict.fromkeys(
                self._ui_norm(x) for x in list(raw_codes or []) if self._ui_norm(x)
            )
        )
        if not codes:
            self.log("No trained stocks found. Load/train a model first.", "warning")
            return

        old_worker = self.workers.get("get_infor")
        if old_worker and old_worker.isRunning():
            self.log("Get Infor is already running.", "info")
            return

        if hasattr(self, "get_infor_btn"):
            self.get_infor_btn.setEnabled(False)
        self.progress.setRange(0, 0)
        self.progress.show()
        self.status_label.setText(
            f"Get Infor: syncing {len(codes)} trained stocks..."
        )
        self.log(
            (
                "Get Infor started: AKShare sync for "
                f"{len(codes)} trained stocks (last 29 days, incremental)."
            ),
            "info",
        )

        def _task() -> Any:
            from data.fetcher import get_fetcher

            fetcher = get_fetcher()
            return fetcher.refresh_trained_stock_history(
                codes,
                interval="1m",
                window_days=29,
                allow_online=True,
                sync_session_cache=True,
                replace_realtime_after_close=True,
            )

        worker = WorkerThread(
            _task,
            timeout_seconds=float(max(180, int(len(codes)) * 18)),
        )
        self._track_worker(worker)
        self.workers["get_infor"] = worker

        def _finalize() -> None:
            self.progress.hide()
            if hasattr(self, "get_infor_btn"):
                self.get_infor_btn.setEnabled(True)
            self.workers.pop("get_infor", None)

        def _on_done(res: object) -> None:
            _finalize()
            report = dict(res or {})
            total = int(report.get("total", 0) or 0)
            updated = int(report.get("updated", 0) or 0)
            cached = int(report.get("cached", 0) or 0)
            purged_map = dict(report.get("purged_realtime_rows", {}) or {})
            purged = int(
                sum(int(v or 0) for v in purged_map.values())
            )
            errors = dict(report.get("errors", {}) or {})
            if errors:
                self.log(
                    (
                        "Get Infor completed with warnings: "
                        f"updated={updated}, cached={cached}, purged_rt={purged}, "
                        f"errors={len(errors)}, total={total}."
                    ),
                    "warning",
                )
                bad_codes = ", ".join(list(errors.keys())[:8])
                if bad_codes:
                    self.log(f"Get Infor error codes: {bad_codes}", "warning")
            else:
                self.log(
                    (
                        "Get Infor completed: "
                        f"updated={updated}, cached={cached}, purged_rt={purged}, "
                        f"total={total}."
                    ),
                    "success",
                )
            self.status_label.setText("Get Infor completed")

            try:
                sym = self._ui_norm(self.stock_input.text())
                iv = self._normalize_interval_token(self.interval_combo.currentText())
                if sym:
                    self._queue_history_refresh(sym, iv)
            except Exception as exc:
                log.debug("Suppressed exception in ui/app.py", exc_info=exc)

        def _on_error(err: str) -> None:
            _finalize()
            self.status_label.setText("Get Infor failed")
            self.log(f"Get Infor failed: {err}", "error")

        worker.result.connect(_on_done)
        worker.error.connect(_on_error)
        worker.start()

    def _train_trained_stocks(self) -> None:
        """
        Train only already-trained stocks using latest cached data.

        A dialog asks for stock count (N). The model is retrained on the
        N stocks with the oldest last-train timestamps.
        """
        trained = list(
            dict.fromkeys(
                self._ui_norm(x) for x in self._get_trained_stock_codes()
                if self._ui_norm(x)
            )
        )
        self._sync_trained_stock_last_train_from_model()

        pending_codes: set[str] = set()
        try:
            fetcher = getattr(self.predictor, "fetcher", None)
            if fetcher is None:
                from data.fetcher import get_fetcher
                fetcher = get_fetcher()
            reconcile_fn = getattr(fetcher, "reconcile_pending_cache_sync", None)
            if callable(reconcile_fn):
                try:
                    reconcile_fn(codes=list(trained), interval="1m")
                except TypeError:
                    reconcile_fn()
            pending_fn = getattr(fetcher, "get_pending_reconcile_codes", None)
            if callable(pending_fn):
                pending_codes = {
                    self._ui_norm(x)
                    for x in list(pending_fn(interval="1m") or [])
                    if self._ui_norm(x)
                }
        except Exception:
            pending_codes = set()

        if pending_codes:
            before = int(len(trained))
            trained = [c for c in trained if c not in pending_codes]
            removed = int(max(0, before - len(trained)))
            if removed > 0:
                self.log(
                    (
                        f"Skipped {removed} stock(s) with pending cache reconcile. "
                        "Press Get Infor to finish sync before training."
                    ),
                    "warning",
                )

        if not trained:
            if pending_codes:
                self.log(
                    "All trained stocks are waiting for cache reconcile. Run Get Infor first.",
                    "warning",
                )
            else:
                self.log("No trained stocks found. Load/train a model first.", "warning")
            return

        try:
            from .dialogs import TrainTrainedStocksDialog
        except Exception as exc:
            self.log(f"Train trained stocks dialog unavailable: {exc}", "error")
            return

        dialog = TrainTrainedStocksDialog(
            trained_codes=trained,
            last_train_map=dict(self._trained_stock_last_train or {}),
            parent=self,
        )
        dialog.exec()

        result = getattr(dialog, "training_result", None)
        if not isinstance(result, dict):
            return

        if str(result.get("status", "")).strip().lower() != "complete":
            status = str(result.get("status", "cancelled")).strip().lower()
            if status == "cancelled":
                self.log("Train trained stocks cancelled.", "info")
            return
        self._handle_training_drift_alarm(
            result,
            context="train_trained_stocks",
        )

        trained_codes = list(
            dict.fromkeys(
                self._ui_norm(x)
                for x in list(
                    result.get("trained_stock_codes")
                    or result.get("selected_codes")
                    or []
                )
                if self._ui_norm(x)
            )
        )
        if trained_codes:
            trained_at = str(
                result.get("trained_at") or datetime.now().isoformat(timespec="seconds")
            )
            self._record_trained_stock_last_train(
                trained_codes,
                trained_at=trained_at,
            )
            self._update_trained_stocks_ui()
            self.log(
                (
                    "Train trained stocks completed: "
                    f"{len(trained_codes)} stock(s)."
                ),
                "success",
            )

        self._init_components()

    def _handle_training_drift_alarm(
        self,
        result: dict[str, object] | None,
        *,
        context: str,
    ) -> None:
        """Escalate trainer drift alarms and force auto-trade to MANUAL."""
        payload = result if isinstance(result, dict) else {}
        drift_guard = payload.get("drift_guard", {})
        quality_gate = payload.get("quality_gate", {})
        if not isinstance(drift_guard, dict):
            drift_guard = {}
        if not isinstance(quality_gate, dict):
            quality_gate = {}

        action = str(drift_guard.get("action", "") or "").strip().lower()
        failed_reasons = {
            str(x).strip().lower()
            for x in list(quality_gate.get("failed_reasons", []) or [])
            if str(x).strip()
        }
        if (
            action != "rollback_recommended"
            and "drift_guard_block" not in failed_reasons
        ):
            return

        try:
            score_drop = float(drift_guard.get("score_drop", 0.0) or 0.0)
        except (TypeError, ValueError):
            score_drop = 0.0
        try:
            acc_drop = float(drift_guard.get("accuracy_drop", 0.0) or 0.0)
        except (TypeError, ValueError):
            acc_drop = 0.0

        reason = (
            f"{context}: model drift guard triggered "
            f"(action={action or 'unknown'}, "
            f"score_drop={score_drop:.3f}, accuracy_drop={acc_drop:.3f})"
        )
        self.log(reason, "warning")

        try:
            ExecutionEngine = _lazy_get("trading.executor", "ExecutionEngine")
            handled = int(
                ExecutionEngine.trigger_model_drift_alarm(
                    reason=reason,
                    severity="critical",
                    metadata={
                        "context": str(context),
                        "action": str(action),
                        "score_drop": float(score_drop),
                        "accuracy_drop": float(acc_drop),
                    },
                )
            )
            if handled > 0:
                self._auto_trade_mode = AutoTradeMode.MANUAL
                self._apply_auto_trade_mode(AutoTradeMode.MANUAL)
        except Exception as exc:
            self.log(f"Drift alarm escalation failed: {exc}", "warning")

    def _init_auto_trader(self) -> None:
        _init_auto_trader_impl(self)

    def _on_interval_changed(self, interval: str) -> None:
        """Handle interval change - reload model and restart monitor."""
        interval = self._normalize_interval_token(interval)
        horizon = self.forecast_spin.value()
        self.model_info.setText(f"Interval: {interval}, Horizon: {horizon}")
        self._update_trained_stocks_ui([])

        self.lookback_spin.setValue(self._recommended_lookback(interval))
        self._bars_by_symbol.clear()
        self._last_bar_feed_ts.clear()
        self._chart_symbol = ""
        self._queue_history_refresh(self.stock_input.text(), interval)
        try:
            if hasattr(self.chart, "clear"):
                self.chart.clear()
            if hasattr(self, "chart_latest_label"):
                self.chart_latest_label.setText("Latest --")
        except Exception as exc:
            log.debug("Suppressed exception in ui/app.py", exc_info=exc)

        was_monitoring = bool(self.monitor and self.monitor.isRunning())
        if was_monitoring:
            self._stop_monitoring()

        if self.predictor:
            try:
                if self._has_exact_model_artifacts(interval, horizon):
                    Predictor = _lazy_get("models.predictor", "Predictor")
                    self.predictor = Predictor(
                        capital=self.capital_spin.value(),
                        interval=interval,
                        prediction_horizon=horizon
                    )
                    if self.predictor.ensemble:
                        active_iv, active_h = self._sync_ui_to_loaded_model(
                            interval,
                            horizon,
                            preserve_requested_interval=True,
                        )
                        self.log(
                            f"Model reloaded for {active_iv} interval, horizon {active_h}",
                            "info",
                        )
                        self._update_trained_stocks_ui()
                        self._log_model_alignment_debug(
                            context="interval_reload",
                            requested_interval=interval,
                            requested_horizon=horizon,
                        )
                else:
                    # Keep current model loaded but preserve user's selected
                    # chart/analysis interval and show explicit mismatch status.
                    self._sync_ui_to_loaded_model(
                        interval,
                        horizon,
                        preserve_requested_interval=True,
                    )
                    model_iv, model_h = self._loaded_model_ui_meta()
                    self.log(
                        (
                            f"No exact model artifacts for {interval}/{horizon}; "
                            f"using loaded model {model_iv}/{model_h} for signals"
                        ),
                        "warning",
                    )
                    self._update_trained_stocks_ui()
                    self._log_model_alignment_debug(
                        context="interval_keep_loaded",
                        requested_interval=interval,
                        requested_horizon=horizon,
                    )
            except Exception as e:
                self.log(f"Model reload failed: {e}", "warning")

        if was_monitoring:
            self._start_monitoring()

        # Update auto-trader predictor
        if (
            self.executor
            and self.executor.auto_trader
            and self.predictor
        ):
            self.executor.auto_trader.update_predictor(self.predictor)

        selected = self._ui_norm(self.stock_input.text())
        if (
            selected
            and self.predictor is not None
            and self.predictor.ensemble is not None
        ):
            self.stock_input.setText(selected)
            self._analyze_stock()

        # =========================================================================
        # REAL-TIME MONITORING
    # =========================================================================

    def _toggle_monitoring(self, checked: bool) -> None:
        _toggle_monitoring_impl(self, checked)

    def _start_monitoring(self) -> None:
        _start_monitoring_impl(self)

    def _stop_monitoring(self) -> None:
        _stop_monitoring_impl(self)

    def _on_signal_detected(self, pred: Any) -> None:
        _on_signal_detected_impl(self, pred)

    def _on_price_updated(self, code: str, price: float) -> None:
        _on_price_updated_impl(self, code, price)

    def _refresh_live_chart_forecast(self) -> None:
        _refresh_live_chart_forecast_impl(self)

    def _prepare_chart_bars_for_interval(
        self,
        bars: list[dict[str, Any]] | None,
        interval: str,
        *,
        symbol: str = "",
    ) -> list[dict[str, Any]]:
        return _prepare_chart_bars_for_interval_impl(
            self,
            bars,
            interval,
            symbol=symbol,
        )

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
            except Exception as exc:
                log.debug("Suppressed exception in ui/app.py", exc_info=exc)

        try:
            self._queue_history_refresh(
                code,
                self._normalize_interval_token(self.interval_combo.currentText()),
            )
        except Exception as exc:
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

    # =========================================================================
    # =========================================================================

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

        request_key = (
            f"{normalized}:{interval}:{forecast_bars}:"
            f"{int(infer_lookback)}:{int(use_realtime)}:"
            f"{infer_interval}:{int(infer_horizon)}:{int(history_allow_online)}"
        )
        req_now = time.monotonic()
        last_req = dict(self._last_analyze_request or {})
        if (
            last_req.get("key") == request_key
            and (req_now - float(last_req.get("ts", 0.0) or 0.0)) < 1.2
        ):
            self._debug_console(
                f"analyze_dedup:{normalized}:{interval}",
                f"Skipped duplicate analyze request for {normalized} ({interval})",
                min_gap_seconds=8.0,
                level="info",
            )
            return
        self._last_analyze_request = {"key": request_key, "ts": req_now}

        self.analyze_action.setEnabled(False)

        if hasattr(self.signal_panel, 'reset'):
            self.signal_panel.reset()

        self.status_label.setText(f"Analyzing {normalized}...")
        self.progress.setRange(0, 0)
        self.progress.show()

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
        worker.result.connect(self._on_analysis_done)
        worker.error.connect(self._on_analysis_error)
        self.workers["analyze"] = worker
        worker.start()

    def _load_chart_history_bars(
        self,
        symbol: str,
        interval: str,
        lookback_bars: int,
    ) -> list[dict[str, Any]]:
        return _load_chart_history_bars_impl(
            self,
            symbol,
            interval,
            lookback_bars,
        )

    def _on_analysis_done(self, pred: Any) -> None:
        """Handle analysis completion; also triggers news fetch."""
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
                    except Exception:
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
                except Exception as e:
                    log.debug(f"Chart update failed: {e}")

        # Update details (with news sentiment)
        self._update_details(pred)

        if (
            hasattr(self, 'news_panel')
            and hasattr(self.news_panel, 'set_stock')
        ):
            try:
                self.news_panel.set_stock(pred.stock_code)
            except Exception as e:
                log.debug(f"News fetch for {pred.stock_code}: {e}")

        self._add_to_history(pred)

        try:
            self._ensure_feed_subscription(pred.stock_code)
        except Exception as exc:
            log.debug("Suppressed exception in ui/app.py", exc_info=exc)
        if not (self.monitor and self.monitor.isRunning()):
            try:
                self.monitor_action.setChecked(True)
                self._start_monitoring()
            except Exception as exc:
                log.debug("Suppressed exception in ui/app.py", exc_info=exc)

        Signal = _lazy_get("models.predictor", "Signal")
        if hasattr(pred, 'signal'):
            is_manual = (self._auto_trade_mode == AutoTradeMode.MANUAL)
            self.buy_btn.setEnabled(
                is_manual
                and pred.signal in [Signal.STRONG_BUY, Signal.BUY]
            )
            self.sell_btn.setEnabled(
                is_manual
                and pred.signal in [Signal.STRONG_SELL, Signal.SELL]
            )

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
            Signal.STRONG_BUY: "#2ea043",
            Signal.BUY: "#35b57c",
            Signal.HOLD: "#d8a03a",
            Signal.SELL: "#e5534b",
            Signal.STRONG_SELL: "#da3633",
        }

        signal = getattr(pred, 'signal', Signal.HOLD)
        color = signal_colors.get(signal, "#dbe4f3")
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
                        sent_color = "#35b57c"
                        sent_emoji = "UP"
                    elif sent_label == "negative":
                        sent_color = "#e5534b"
                        sent_emoji = "DOWN"
                    else:
                        sent_color = "#d8a03a"
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
        except Exception as e:
            log.debug(f"News sentiment fetch: {e}")

        html = f"""
        <style>
            body {{
                color: #dbe4f3;
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
            .label {{ color: #aac3ec; }}
            .positive {{ color: #35b57c; }}
            .negative {{ color: #e5534b; }}
            .neutral {{ color: #d8a03a; }}
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
            except Exception as exc:
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
        signal_item.setForeground(QColor("#79a6ff"))
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

    def _compute_guess_profit(
        self,
        direction: str,
        entry_price: float,
        mark_price: float,
        shares: int,
    ) -> float:
        """Compute virtual directional P&L (positive => currently correct)."""
        entry = float(entry_price or 0.0)
        mark = float(mark_price or 0.0)
        qty = max(1, int(shares or 1))

        if entry <= 0 or mark <= 0:
            return 0.0
        if direction == "UP":
            return (mark - entry) * qty
        if direction == "DOWN":
            return (entry - mark) * qty
        return 0.0

    def _refresh_guess_rows_for_symbol(self, code: str, price: float) -> None:
        """Update history result for this symbol using latest real-time price."""
        symbol = self._ui_norm(code)
        mark_price = float(price or 0.0)
        if not symbol or mark_price <= 0:
            return

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
            shares = int(meta.get("shares", self._guess_profit_notional_shares) or 1)
            pnl = self._compute_guess_profit(direction, entry, mark_price, shares)
            raw_ret_pct = ((mark_price / entry - 1.0) * 100.0) if entry > 0 else 0.0
            signed_ret_pct = (
                raw_ret_pct
                if direction == "UP"
                else (-raw_ret_pct if direction == "DOWN" else 0.0)
            )

            if direction == "NONE":
                result_item.setText("--")
                result_item.setForeground(QColor("#aac3ec"))
            elif pnl > 0:
                result_item.setText(
                    f"CORRECT CNY {pnl:+,.2f} ({signed_ret_pct:+.2f}%)"
                )
                result_item.setForeground(QColor("#35b57c"))
            elif pnl < 0:
                result_item.setText(
                    f"WRONG CNY {pnl:,.2f} ({signed_ret_pct:+.2f}%)"
                )
                result_item.setForeground(QColor("#e5534b"))
            else:
                result_item.setText("FLAT CNY 0.00 (+0.00%)")
                result_item.setForeground(QColor("#aac3ec"))

            meta["mark_price"] = mark_price
            result_item.setData(Qt.ItemDataRole.UserRole, meta)

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
            color = "#35b57c" if net_val >= 0 else "#e5534b"
            label_profit.setStyleSheet(
                f"color: {color}; font-size: 16px; font-weight: bold;"
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
                "color: #79a6ff; font-size: 16px; font-weight: bold;"
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

    def _refresh_all(self) -> None:
        _refresh_all_impl(self)

    # =========================================================================
    # =========================================================================

    def _toggle_trading(self) -> None:
        _toggle_trading_impl(self)

    def _on_mode_combo_changed(self, index: int) -> None:
        _on_mode_combo_changed_impl(self, index)

    def _set_trading_mode(
        self,
        mode: TradingMode,
        prompt_reconnect: bool = False,
    ) -> None:
        _set_trading_mode_impl(
            self,
            mode,
            prompt_reconnect=prompt_reconnect,
        )

    def _connect_trading(self) -> None:
        _connect_trading_impl(self)

    def _disconnect_trading(self) -> None:
        _disconnect_trading_impl(self)

    def _on_chart_trade_requested(self, side: str, price: float) -> None:
        _on_chart_trade_requested_impl(self, side, price)

    def _show_chart_trade_dialog(
        self,
        symbol: str,
        side: str,
        clicked_price: float,
        lot: int,
    ) -> dict[str, float | int | str | bool] | None:
        return _show_chart_trade_dialog_impl(
            self,
            symbol,
            side,
            clicked_price,
            lot,
        )

    def _submit_chart_order(
        self,
        symbol: str,
        side: OrderSide,
        qty: int,
        price: float,
        order_type: str = "limit",
        time_in_force: str = "day",
        trigger_price: float = 0.0,
        trailing_stop_pct: float = 0.0,
        trail_limit_offset_pct: float = 0.0,
        strict_time_in_force: bool = False,
        stop_loss: float = 0.0,
        take_profit: float = 0.0,
        bracket: bool = False,
    ) -> None:
        _submit_chart_order_impl(
            self,
            symbol,
            side,
            qty,
            price,
            order_type=order_type,
            time_in_force=time_in_force,
            trigger_price=trigger_price,
            trailing_stop_pct=trailing_stop_pct,
            trail_limit_offset_pct=trail_limit_offset_pct,
            strict_time_in_force=strict_time_in_force,
            stop_loss=stop_loss,
            take_profit=take_profit,
            bracket=bracket,
        )

    def _execute_buy(self) -> None:
        _execute_buy_impl(self)

    def _execute_sell(self) -> None:
        _execute_sell_impl(self)

    def _on_order_filled(self, order: Any, fill: Any) -> None:
        _on_order_filled_impl(self, order, fill)

    def _on_order_rejected(self, order: Any, reason: Any) -> None:
        _on_order_rejected_impl(self, order, reason)

    def _refresh_portfolio(self) -> None:
        _refresh_portfolio_impl(self)

    # =========================================================================
    # =========================================================================

    def _start_training(self) -> None:
        _start_training_impl(self)

    def _show_auto_learn(self) -> None:
        _show_auto_learn_impl(self)

    def _show_strategy_marketplace(self) -> None:
        _show_strategy_marketplace_impl(self)

    def _show_backtest(self) -> None:
        _show_backtest_impl(self)

    def _show_about(self) -> None:
        _show_about_impl(self)

    # =========================================================================
    # AUTO-TRADE CONTROLS
    # =========================================================================

    def _on_trade_mode_changed(self, index: int) -> None:
        _on_trade_mode_changed_impl(self, index)

    def _apply_auto_trade_mode(self, mode: AutoTradeMode) -> None:
        _apply_auto_trade_mode_impl(self, mode)

    def _update_auto_trade_status_label(self, mode: AutoTradeMode) -> None:
        _update_auto_trade_status_label_impl(self, mode)

    def _toggle_auto_pause(self) -> None:
        _toggle_auto_pause_impl(self)

    def _approve_all_pending(self) -> None:
        _approve_all_pending_impl(self)

    def _reject_all_pending(self) -> None:
        _reject_all_pending_impl(self)

    def _show_auto_trade_settings(self) -> None:
        _show_auto_trade_settings_impl(self)

    def _on_auto_trade_action_safe(self, action: AutoTradeAction) -> None:
        _on_auto_trade_action_safe_impl(self, action)

    def _on_auto_trade_action(self, action: AutoTradeAction) -> None:
        _on_auto_trade_action_impl(self, action)

    def _on_pending_approval_safe(self, action: AutoTradeAction) -> None:
        _on_pending_approval_safe_impl(self, action)

    def _on_pending_approval(self, action: AutoTradeAction) -> None:
        _on_pending_approval_impl(self, action)

    def _refresh_pending_table(self) -> None:
        _refresh_pending_table_impl(self)

    def _refresh_auto_trade_ui(self) -> None:
        _refresh_auto_trade_ui_impl(self)

    # =========================================================================
    # =========================================================================

    def _update_clock(self) -> None:
        """Update clock"""
        self.time_label.setText(
            datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )

    def _update_market_status(self) -> None:
        """Update market status"""
        is_open = CONFIG.is_market_open()
        hours_text = self._market_hours_text()
        now_sh = self._shanghai_now()

        if is_open:
            self.market_label.setText(
                f"Market Open | Trading Hours: {hours_text}"
            )
            self.market_label.setStyleSheet(
                "color: #35b57c; font-weight: bold;"
            )
        else:
            next_open = self._next_market_open(now_sh)
            if next_open is not None:
                next_open_text = next_open.strftime("%Y-%m-%d %H:%M CST")
            else:
                next_open_text = "--"
            self.market_label.setText(
                f"Market Closed | Trading Hours: {hours_text} | Next Open: {next_open_text}"
            )
            self.market_label.setStyleSheet("color: #e5534b;")

    def log(self, message: str, level: str = "info") -> None:
        _log_impl(self, message, level)

    # =========================================================================
    # CLOSE EVENT (FIX #3 - calls super)
    # =========================================================================

    def closeEvent(self, event: Any) -> None:
        _close_event_impl(self, event)
        super().closeEvent(event)

    # =========================================================================
    # =========================================================================

    def _save_state(self) -> None:
        _save_state_impl(self)

    def _load_state(self) -> None:
        _load_state_impl(self)

# Bind extracted bar/interval operations to MainApp.
MainApp._seven_day_lookback = _app_bar_ops._seven_day_lookback
MainApp._trained_stock_window_bars = _app_bar_ops._trained_stock_window_bars
MainApp._recommended_lookback = _app_bar_ops._recommended_lookback
MainApp._queue_history_refresh = _app_bar_ops._queue_history_refresh
MainApp._consume_history_refresh = _app_bar_ops._consume_history_refresh
MainApp._schedule_analysis_recovery = _app_bar_ops._schedule_analysis_recovery
MainApp._history_window_bars = _app_bar_ops._history_window_bars
MainApp._ts_to_epoch = _app_bar_ops._ts_to_epoch
MainApp._epoch_to_iso = _app_bar_ops._epoch_to_iso
MainApp._now_iso = _app_bar_ops._now_iso
MainApp._merge_bars = _app_bar_ops._merge_bars
MainApp._interval_seconds = _app_bar_ops._interval_seconds
MainApp._interval_token_from_seconds = _app_bar_ops._interval_token_from_seconds
MainApp._bars_needed_from_base_interval = _app_bar_ops._bars_needed_from_base_interval
MainApp._resample_chart_bars = _app_bar_ops._resample_chart_bars
MainApp._dominant_bar_interval = _app_bar_ops._dominant_bar_interval
MainApp._effective_anchor_price = _app_bar_ops._effective_anchor_price
MainApp._stabilize_chart_depth = _app_bar_ops._stabilize_chart_depth
MainApp._bar_bucket_epoch = _app_bar_ops._bar_bucket_epoch
MainApp._bar_trading_date = _app_bar_ops._bar_trading_date
MainApp._is_intraday_day_boundary = _app_bar_ops._is_intraday_day_boundary
MainApp._shanghai_now = _app_bar_ops._shanghai_now
MainApp._is_cn_trading_day = _app_bar_ops._is_cn_trading_day
MainApp._market_hours_text = _app_bar_ops._market_hours_text
MainApp._next_market_open = _app_bar_ops._next_market_open
MainApp._is_market_session_timestamp = _app_bar_ops._is_market_session_timestamp
MainApp._filter_bars_to_market_session = _app_bar_ops._filter_bars_to_market_session
MainApp._bar_safety_caps = _app_bar_ops._bar_safety_caps
MainApp._synthetic_tick_jump_cap = _app_bar_ops._synthetic_tick_jump_cap
MainApp._sanitize_ohlc = _app_bar_ops._sanitize_ohlc
MainApp._is_outlier_tick = _app_bar_ops._is_outlier_tick
MainApp._get_levels_dict = _app_bar_ops._get_levels_dict
MainApp._scrub_chart_bars = _app_bar_ops._scrub_chart_bars
MainApp._rescale_chart_bars_to_anchor = _app_bar_ops._rescale_chart_bars_to_anchor
MainApp._recover_chart_bars_from_close = _app_bar_ops._recover_chart_bars_from_close

def run_app() -> None:
    """Run the application"""
    os.environ.setdefault('QT_AUTO_SCREEN_SCALE_FACTOR', '1')

    app = QApplication(sys.argv)
    app.setStyle('Fusion')

    app.setApplicationName("AI Stock Trading System")
    app.setApplicationVersion("2.0")
    app.setOrganizationName("AI Trading")

    font = QFont("Segoe UI", 10)
    app.setFont(font)

    window = MainApp()
    window.show()

    # Keep Python signal handling responsive while Qt loop is running.
    heartbeat = QTimer()
    heartbeat.setInterval(200)
    heartbeat.timeout.connect(lambda: None)
    heartbeat.start()

    previous_sigint = _install_sigint_handler(app)
    exit_code = 0
    try:
        exit_code = int(app.exec())
    except KeyboardInterrupt:
        log.info("UI interrupted by user")
    finally:
        try:
            heartbeat.stop()
        except Exception as exc:
            log.debug("Suppressed exception in ui/app.py", exc_info=exc)
        _restore_sigint_handler(previous_sigint)

    sys.exit(exit_code)

def _install_sigint_handler(app: QApplication) -> Any | None:
    """Route Ctrl+C to Qt quit for graceful terminal shutdown."""
    try:
        previous = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, lambda *_: app.quit())
        return previous
    except Exception as e:
        log.debug(f"SIGINT handler install failed: {e}")
        return None

def _restore_sigint_handler(previous_handler: Any | None) -> None:
    """Restore previous SIGINT handler after app loop exits."""
    if previous_handler is None:
        return
    try:
        signal.signal(signal.SIGINT, previous_handler)
    except Exception as exc:
        log.debug("Suppressed exception in ui/app.py", exc_info=exc)

if __name__ == "__main__":
    run_app()

