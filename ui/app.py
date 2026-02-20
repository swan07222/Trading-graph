# ui/app.py
import json
import math
import os
import signal
import sys
import threading
import time
from datetime import datetime, timedelta, timezone
from importlib import import_module
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np
from PyQt6.QtCore import QSize, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QAction, QActionGroup, QColor, QFont
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QStatusBar,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextEdit,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from config.settings import CONFIG, TradingMode
from core.constants import get_lot_size
from core.types import (
    AutoTradeAction,
    AutoTradeMode,
    OrderSide,
    OrderType,
    TradeSignal,
)
from ui.background_tasks import (
    RealTimeMonitor,
    WorkerThread,
)
from ui.background_tasks import (
    collect_live_readiness_failures as _collect_live_readiness_failures,
)
from ui.background_tasks import (
    normalize_stock_code as _normalize_stock_code,
)
from ui.background_tasks import (
    sanitize_watch_list as _sanitize_watch_list,
)
from ui.background_tasks import (
    validate_stock_code as _validate_stock_code,
)
from utils.logger import get_logger

log = get_logger(__name__)

def _lazy_get(module: str, name: str):
    return getattr(import_module(module), name)

class MainApp(QMainWindow):
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

    def __init__(self):
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
        self._last_session_cache_write_ts: dict[str, float] = {}
        self._last_analyze_request: dict[str, Any] = {}
        self._last_analysis_log: dict[str, Any] = {}
        self._analysis_recovery_attempt_ts: dict[str, float] = {}
        self._watchlist_row_by_code: dict[str, int] = {}
        self._last_watchlist_price_ui: dict[str, tuple[float, float]] = {}
        self._last_quote_ui_emit: dict[str, tuple[float, float]] = {}
        self._guess_profit_notional_shares: int = max(
            1, int(getattr(CONFIG, "LOT_SIZE", 100) or 100)
        )

        self._bars_by_symbol: dict[str, list[dict]] = {}
        self._trained_stock_codes_cache: list[str] = []
        self._trained_stock_last_train: dict[str, str] = {}
        self._last_bar_feed_ts: dict[str, float] = {}
        self._chart_symbol: str = ""
        self._history_refresh_once: set[tuple[str, str]] = set()
        self._debug_console_enabled: bool = str(
            os.environ.get("TRADING_DEBUG_CONSOLE", "1")
        ).strip().lower() in ("1", "true", "yes", "on")
        self._debug_console_last_emit: dict[str, float] = {}
        self._syncing_mode_ui = False
        self._session_bar_cache = None
        try:
            from data.session_cache import get_session_bar_cache
            self._session_bar_cache = get_session_bar_cache()
        except Exception:
            self._session_bar_cache = None
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
            pass

        QTimer.singleShot(0, self._init_components)

    # =========================================================================
    # UI NORMALIZATION (FIX #1 - was missing entirely)
    # =========================================================================

    def _ui_norm(self, text: str) -> str:
        """Normalize stock code for UI comparison."""
        return _normalize_stock_code(text)

    @staticmethod
    def _safe_list(values: Any) -> list[Any]:
        """Convert optional iterables to list without truthiness checks."""
        if values is None:
            return []
        try:
            return list(values)
        except Exception:
            return []

    def _trained_stock_last_train_meta_path(self) -> Path:
        return Path(CONFIG.data_dir) / "trained_stock_last_train.json"

    def _load_trained_stock_last_train_meta(self) -> None:
        """Load trained-stock last-train timestamps from disk."""
        self._trained_stock_last_train = {}
        path = self._trained_stock_last_train_meta_path()
        if not path.exists():
            return
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            payload = raw.get("last_train", raw) if isinstance(raw, dict) else {}
            if not isinstance(payload, dict):
                return
            out: dict[str, str] = {}
            for k, v in payload.items():
                code = self._ui_norm(str(k or ""))
                if not code:
                    continue
                ts = str(v or "").strip()
                if not ts:
                    continue
                out[code] = ts
            self._trained_stock_last_train = out
        except Exception as exc:
            log.debug("Failed to load trained-stock last-train metadata: %s", exc)
            self._trained_stock_last_train = {}

    def _save_trained_stock_last_train_meta(self) -> None:
        """Persist trained-stock last-train timestamps to disk."""
        path = self._trained_stock_last_train_meta_path()
        payload = {
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "last_train": dict(self._trained_stock_last_train or {}),
        }
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(payload, ensure_ascii=True, indent=2),
                encoding="utf-8",
            )
        except Exception as exc:
            log.debug("Failed to save trained-stock last-train metadata: %s", exc)

    def _record_trained_stock_last_train(
        self,
        codes: list[str],
        *,
        trained_at: str | None = None,
    ) -> None:
        """Update last-train timestamps for the provided stock codes."""
        when = str(trained_at or datetime.now().isoformat(timespec="seconds"))
        changed = False
        for raw in list(codes or []):
            code = self._ui_norm(raw)
            if not code:
                continue
            if self._trained_stock_last_train.get(code) != when:
                self._trained_stock_last_train[code] = when
                changed = True
        if changed:
            try:
                if self.predictor is not None:
                    if hasattr(self.predictor, "_trained_stock_last_train"):
                        self.predictor._trained_stock_last_train = dict(  # type: ignore[attr-defined]
                            self._trained_stock_last_train
                        )
                    ens = getattr(self.predictor, "ensemble", None)
                    if ens is not None and hasattr(ens, "trained_stock_last_train"):
                        ens.trained_stock_last_train = dict(
                            self._trained_stock_last_train
                        )
            except Exception:
                pass
            self._save_trained_stock_last_train_meta()

    @staticmethod
    def _format_last_train_text(ts_text: str | None) -> str:
        """Format ISO timestamp into short display text."""
        if not ts_text:
            return "--"
        text = str(ts_text).strip()
        if not text:
            return "--"
        try:
            dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
            if dt.tzinfo is not None:
                dt = dt.astimezone().replace(tzinfo=None)
            return dt.strftime("%Y-%m-%d %H:%M")
        except Exception:
            return text[:16]

    def _set_initial_window_geometry(self) -> None:
        """Fit initial window to available screen so bottom controls remain visible."""
        try:
            screen = QApplication.primaryScreen()
            if screen is None:
                self.setGeometry(80, 60, 1360, 780)
                return
            avail = screen.availableGeometry()
            width = min(max(1120, int(avail.width() * 0.90)), int(avail.width()))
            height = min(max(700, int(avail.height() * 0.90)), int(avail.height()))
            x = int(avail.left() + ((avail.width() - width) / 2))
            y = int(avail.top() + ((avail.height() - height) / 2))
            self.setGeometry(x, y, width, height)
        except Exception:
            self.setGeometry(80, 60, 1360, 780)

    def _track_worker(self, worker: WorkerThread) -> None:
        """Track worker lifecycle so threads are never orphaned."""
        self._active_workers.add(worker)

        def _drop(*_args):
            self._active_workers.discard(worker)

        worker.finished.connect(_drop)

    def _model_interval_to_ui_token(self, interval: str) -> str:
        """Normalize model interval metadata to available UI tokens."""
        iv = str(interval or "").strip().lower()
        if iv == "1h":
            return "60m"
        return iv

    def _normalize_interval_token(
        self,
        interval: str | None,
        *,
        fallback: str = "1m",
    ) -> str:
        """Normalize interval aliases used across feed/cache/UI paths."""
        iv = str(interval or "").strip().lower()
        if not iv:
            return str(fallback or "1m").strip().lower()

        aliases = {
            "1h": "60m",
            "60min": "60m",
            "60mins": "60m",
            "1day": "1d",
            "day": "1d",
            "daily": "1d",
            "1440m": "1d",
        }
        return aliases.get(iv, iv)

    def _debug_console(
        self,
        key: str,
        message: str,
        *,
        min_gap_seconds: float = 2.0,
        level: str = "warning",
    ) -> None:
        """Throttled debug message to UI system-log + backend logger."""
        if not bool(getattr(self, "_debug_console_enabled", False)):
            return
        try:
            now = time.monotonic()
            prev = float(self._debug_console_last_emit.get(key, 0.0))
            if (now - prev) < float(max(0.0, min_gap_seconds)):
                return
            self._debug_console_last_emit[key] = now
            txt = f"[DBG] {message}"
            if hasattr(self, "log"):
                self.log(txt, level=level)
            else:
                log.warning(txt)
        except Exception:
            pass

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

            out: list[float] = []
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
                out.append(float(target_px))
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
            return out

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
            except Exception:
                pass
            try:
                tail_risk = float(
                    np.clip(getattr(pred, "tail_risk_score", tail_risk), 0.0, 1.0)
                )
            except Exception:
                pass
            try:
                confidence = float(
                    np.clip(getattr(pred, "confidence", confidence), 0.0, 1.0)
                )
            except Exception:
                pass

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
            except Exception:
                pass

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

    # =========================================================================
    # =========================================================================

    def _setup_menubar(self):
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
            lambda checked: checked and self._set_trading_mode(TradingMode.SIMULATION)
        )
        self.live_action.triggered.connect(
            lambda checked: checked and self._set_trading_mode(TradingMode.LIVE)
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

    # =========================================================================
    # =========================================================================

    def _setup_toolbar(self):
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

    def _ensure_feed_subscription(self, code: str):
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

    def _on_bar_from_feed(self, symbol: str, bar: dict):
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
            pass

    def _on_tick_from_feed(self, quote):
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
            pass

    def _on_bar_ui(self, symbol: str, bar: dict):
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
            except Exception:
                pass

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
                except Exception:
                    pass
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

        try:
            if self._session_bar_cache is not None:
                key = f"{symbol}:{interval}"
                now_ts = time.time()
                # Avoid excessive disk writes for partial updates.
                min_gap = 0.9 if not is_final else 0.0
                last_ts = float(self._last_session_cache_write_ts.get(key, 0.0))
                if interval == "1m" and (now_ts - last_ts) >= min_gap:
                    self._session_bar_cache.append_bar(symbol, interval, norm_bar)
                    self._last_session_cache_write_ts[key] = now_ts
        except Exception as e:
            log.debug(f"Session cache write failed: {e}")

        current_code = self._ui_norm(self.stock_input.text())
        if current_code != symbol:
            return

        predicted, pred_source_interval = self._resolve_chart_prediction_series(
            symbol=symbol,
            fallback_interval=interval,
        )

        # UNIFIED chart update - draws candles + line + prediction
        try:
            current_price = float(norm_bar.get("close", 0) or 0)
            self._render_chart_state(
                symbol=symbol,
                interval=interval,
                bars=arr,
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
        except Exception:
            pass

    # =========================================================================
    # =========================================================================

    def _setup_ui(self):
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
        """Create left control panel with interval/forecast settings"""
        panel = QWidget()
        panel.setMinimumWidth(250)
        panel.setMaximumWidth(320)
        layout = QVBoxLayout(panel)
        layout.setSpacing(10)

        watchlist_group = QGroupBox("Watchlist")
        watchlist_layout = QVBoxLayout()

        self.watchlist = self._make_table(
            ["Code", "Price", "Change", "Signal"], max_height=250
        )
        self.watchlist.cellClicked.connect(self._on_watchlist_click)

        self._update_watchlist()
        watchlist_layout.addWidget(self.watchlist)

        btn_layout = QHBoxLayout()
        add_btn = QPushButton("+ Add")
        add_btn.clicked.connect(self._add_to_watchlist)
        remove_btn = QPushButton("- Remove")
        remove_btn.clicked.connect(self._remove_from_watchlist)
        btn_layout.addWidget(add_btn)
        btn_layout.addWidget(remove_btn)
        watchlist_layout.addLayout(btn_layout)

        watchlist_group.setLayout(watchlist_layout)
        layout.addWidget(watchlist_group)

        settings_group = QGroupBox("Trading Settings")
        settings_layout = QGridLayout()

        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Paper Trading", "Live Trading"])
        self.mode_combo.currentIndexChanged.connect(self._on_mode_combo_changed)
        self._add_labeled(settings_layout, 0, "Mode:", self.mode_combo)

        self.capital_spin = QDoubleSpinBox()
        self.capital_spin.setRange(10000, 100000000)
        self.capital_spin.setValue(CONFIG.CAPITAL)
        self.capital_spin.setPrefix("CNY ")
        self._add_labeled(settings_layout, 1, "Capital:", self.capital_spin)

        self.risk_spin = QDoubleSpinBox()
        self.risk_spin.setRange(0.5, 5.0)
        self.risk_spin.setValue(CONFIG.RISK_PER_TRADE)
        self.risk_spin.setSuffix(" %")
        self._add_labeled(settings_layout, 2, "Risk/Trade:", self.risk_spin)

        self.interval_combo = QComboBox()
        self.interval_combo.addItems(["1m", "5m", "15m", "30m", "60m", "1d"])
        self.interval_combo.setCurrentText("1m")
        self.interval_combo.currentTextChanged.connect(
            self._on_interval_changed
        )
        self._add_labeled(settings_layout, 3, "Interval:", self.interval_combo)

        self.forecast_spin = QSpinBox()
        self.forecast_spin.setRange(5, 120)
        self.forecast_spin.setValue(self.GUESS_FORECAST_BARS)
        self.forecast_spin.setSuffix(" min")
        self.forecast_spin.setToolTip("Minutes to forecast ahead")
        self._add_labeled(settings_layout, 4, "Forecast:", self.forecast_spin)

        self.lookback_spin = QSpinBox()
        self.lookback_spin.setRange(7, 5000)
        self.lookback_spin.setValue(self._recommended_lookback("1m"))
        self.lookback_spin.setSuffix(" bars")
        self.lookback_spin.setToolTip("Historical bars to use for analysis")
        self._add_labeled(settings_layout, 5, "Lookback:", self.lookback_spin)

        settings_group.setLayout(settings_layout)
        layout.addWidget(settings_group)

        connection_group = QGroupBox("Connection")
        connection_layout = QVBoxLayout()

        self.connection_status = QLabel("Disconnected")
        self.connection_status.setStyleSheet(
            "color: #FF5252; font-weight: bold;"
        )
        connection_layout.addWidget(self.connection_status)

        self.connect_btn = QPushButton("Connect to Broker")
        self.connect_btn.clicked.connect(self._toggle_trading)
        self.connect_btn.setStyleSheet("""
            QPushButton {
                background: #4CAF50;
                color: white;
                border: none;
                padding: 12px;
                border-radius: 6px;
                font-weight: bold;
            }
            QPushButton:hover { background: #388E3C; }
        """)
        connection_layout.addWidget(self.connect_btn)

        connection_group.setLayout(connection_layout)
        layout.addWidget(connection_group)

        ai_group = QGroupBox("AI Model")
        ai_layout = QVBoxLayout()

        self.model_status = QLabel("Model: Loading...")
        ai_layout.addWidget(self.model_status)

        self.model_info = QLabel("")
        self.model_info.setStyleSheet("color: #888; font-size: 10px;")
        ai_layout.addWidget(self.model_info)

        self.trained_stocks_label = QLabel("Trained Stocks: --")
        self.trained_stocks_label.setStyleSheet("color: #9aa4b8; font-size: 10px;")
        ai_layout.addWidget(self.trained_stocks_label)

        self.trained_stocks_hint = QLabel(
            "Full trained stock list is in the right panel tab:\n"
            "Trained Stocks"
        )
        self.trained_stocks_hint.setStyleSheet("color: #6e7681; font-size: 10px;")
        ai_layout.addWidget(self.trained_stocks_hint)

        self.open_trained_tab_btn = QPushButton("Open Trained Stocks")
        self.open_trained_tab_btn.clicked.connect(
            self._focus_trained_stocks_tab
        )
        ai_layout.addWidget(self.open_trained_tab_btn)

        self.get_infor_btn = QPushButton("Get Infor (29d)")
        self.get_infor_btn.setToolTip(
            "Fetch 29-day history for all trained stocks from AKShare.\n"
            "If market is closed, replaces saved realtime rows with AKShare rows.\n"
            "Otherwise fetches incrementally from the last saved AKShare point."
        )
        self.get_infor_btn.clicked.connect(self._get_infor_trained_stocks)
        ai_layout.addWidget(self.get_infor_btn)

        self.train_btn = QPushButton("Train Model")
        self.train_btn.clicked.connect(self._start_training)
        ai_layout.addWidget(self.train_btn)

        self.train_trained_btn = QPushButton("Train Trained Stocks")
        self.train_trained_btn.setToolTip(
            "Train only already-trained stocks using newly synced cache data."
        )
        self.train_trained_btn.clicked.connect(self._train_trained_stocks)
        ai_layout.addWidget(self.train_trained_btn)

        self.auto_learn_btn = QPushButton("Auto Learn")
        self.auto_learn_btn.clicked.connect(self._show_auto_learn)
        ai_layout.addWidget(self.auto_learn_btn)

        self.train_progress = QProgressBar()
        self.train_progress.setVisible(False)
        ai_layout.addWidget(self.train_progress)

        ai_group.setLayout(ai_layout)
        layout.addWidget(ai_group)

        layout.addStretch()
        return panel

    def _make_table(self, headers: list[str], max_height: int | None = None):
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

    def _add_labeled(self, layout: QGridLayout, row: int, text: str, widget: QWidget):
        layout.addWidget(QLabel(text), row, 0)
        layout.addWidget(widget, row, 1)

    def _build_stat_frame(self, labels, value_style: str, padding: int = 15):
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

    def _zoom_chart_in(self):
        if hasattr(self.chart, "zoom_in"):
            try:
                self.chart.zoom_in()
            except Exception:
                pass

    def _zoom_chart_out(self):
        if hasattr(self.chart, "zoom_out"):
            try:
                self.chart.zoom_out()
            except Exception:
                pass

    def _zoom_chart_reset(self):
        if hasattr(self.chart, "reset_view"):
            try:
                self.chart.reset_view()
            except Exception:
                pass

    def _set_chart_overlay(self, key: str, enabled: bool):
        if hasattr(self.chart, "set_overlay_enabled"):
            try:
                self.chart.set_overlay_enabled(str(key), bool(enabled))
            except Exception:
                pass

    def _create_right_panel(self) -> QWidget:
        """Create right panel with portfolio, news, orders, and auto-trade"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setSpacing(10)

        self.right_tabs = QTabWidget()
        tabs = self.right_tabs

        portfolio_tab = QWidget()
        portfolio_layout = QVBoxLayout(portfolio_tab)

        self.account_labels = {}
        labels = [
            ('equity', 'Total Equity', 0, 0),
            ('cash', 'Available Cash', 0, 1),
            ('positions', 'Positions Value', 1, 0),
            ('pnl', 'Total P&L', 1, 1),
        ]
        account_frame, self.account_labels = self._build_stat_frame(
            labels, "color: #00E5FF; font-size: 18px; font-weight: bold;", 15
        )

        portfolio_layout.addWidget(account_frame)

        try:
            from .widgets import PositionTable
            self.positions_table = PositionTable()
        except ImportError:
            self.positions_table = self._make_table(
                ["Code", "Qty", "Price", "Value", "P&L"]
            )
        portfolio_layout.addWidget(self.positions_table)

        tabs.addTab(portfolio_tab, "Portfolio")

        news_tab = QWidget()
        news_layout = QVBoxLayout(news_tab)
        try:
            NewsPanel = _lazy_get("ui.news_widget", "NewsPanel")
            self.news_panel = NewsPanel()
            news_layout.addWidget(self.news_panel)
        except Exception as e:
            log.warning(f"News panel not available: {e}")
            self.news_panel = QLabel("News panel unavailable")
            news_layout.addWidget(self.news_panel)
        tabs.addTab(news_tab, "News and Policy")

        signals_tab = QWidget()
        signals_layout = QVBoxLayout(signals_tab)
        self.signals_table = self._make_table([
            "Time", "Code", "Signal", "Confidence", "Price", "Action"
        ])
        signals_layout.addWidget(self.signals_table)
        tabs.addTab(signals_tab, "Live Signals")

        history_tab = QWidget()
        history_layout = QVBoxLayout(history_tab)
        self.history_table = self._make_table([
            "Time", "Code", "Signal", "Prob UP", "Confidence", "Result"
        ])
        history_layout.addWidget(self.history_table)
        tabs.addTab(history_tab, "History")

        trained_tab = QWidget()
        trained_layout = QVBoxLayout(trained_tab)

        trained_top = QHBoxLayout()
        self.trained_stock_count_label = QLabel("Trained: --")
        self.trained_stock_count_label.setStyleSheet(
            "color: #9aa4b8; font-size: 10px;"
        )
        trained_top.addWidget(self.trained_stock_count_label)
        trained_top.addStretch(1)
        trained_layout.addLayout(trained_top)

        self.trained_stock_search = QLineEdit()
        self.trained_stock_search.setPlaceholderText(
            "Search trained stock code..."
        )
        self.trained_stock_search.textChanged.connect(
            self._filter_trained_stocks_ui
        )
        trained_layout.addWidget(self.trained_stock_search)

        self.trained_stock_list = QListWidget()
        self.trained_stock_list.itemClicked.connect(
            self._on_trained_stock_activated
        )
        self.trained_stock_list.itemDoubleClicked.connect(
            self._on_trained_stock_activated
        )
        self.trained_stock_list.setToolTip(
            "Click a stock to load and analyze it"
        )
        trained_layout.addWidget(self.trained_stock_list, 1)
        self._trained_tab_index = tabs.addTab(trained_tab, "Trained Stocks")

        # ==================== AUTO-TRADE TAB ====================
        auto_trade_tab = QWidget()
        auto_trade_layout = QVBoxLayout(auto_trade_tab)

        # Auto-trade status frame
        self.auto_trade_labels = {}
        auto_labels = [
            ('mode', 'Mode', 0, 0),
            ('trades', 'Trades Today', 0, 1),
            ('pnl', 'Auto P&L', 1, 0),
            ('status', 'Status', 1, 1),
            ('guess_profit', 'Correct Guess P&L', 2, 0),
            ('guess_rate', 'Guess Hit Rate', 2, 1),
        ]
        auto_status_frame, self.auto_trade_labels = self._build_stat_frame(
            auto_labels, "color: #00E5FF; font-size: 16px; font-weight: bold;", 10
        )

        auto_trade_layout.addWidget(auto_status_frame)

        # Pending approvals section (for semi-auto)
        pending_group = QGroupBox("Pending Approvals")
        pending_layout = QVBoxLayout()
        self.pending_table = self._make_table([
            "Time", "Code", "Signal", "Confidence", "Price", "Action"
        ], max_height=150)
        pending_layout.addWidget(self.pending_table)
        pending_group.setLayout(pending_layout)
        auto_trade_layout.addWidget(pending_group)

        # Auto-trade action history
        actions_group = QGroupBox("Auto-Trade Actions")
        actions_layout = QVBoxLayout()
        self.auto_actions_table = self._make_table([
            "Time", "Code", "Signal", "Confidence",
            "Decision", "Qty", "Reason"
        ])
        actions_layout.addWidget(self.auto_actions_table)
        actions_group.setLayout(actions_layout)
        auto_trade_layout.addWidget(actions_group)

        # Auto-trade control buttons
        auto_btn_frame = QFrame()
        auto_btn_layout = QHBoxLayout(auto_btn_frame)

        self.auto_pause_btn = QPushButton("Pause Auto")
        self.auto_pause_btn.clicked.connect(self._toggle_auto_pause)
        self.auto_pause_btn.setEnabled(False)
        auto_btn_layout.addWidget(self.auto_pause_btn)

        self.auto_approve_all_btn = QPushButton("Approve All")
        self.auto_approve_all_btn.clicked.connect(self._approve_all_pending)
        self.auto_approve_all_btn.setEnabled(False)
        auto_btn_layout.addWidget(self.auto_approve_all_btn)

        self.auto_reject_all_btn = QPushButton("Reject All")
        self.auto_reject_all_btn.clicked.connect(self._reject_all_pending)
        self.auto_reject_all_btn.setEnabled(False)
        auto_btn_layout.addWidget(self.auto_reject_all_btn)

        auto_trade_layout.addWidget(auto_btn_frame)

        tabs.addTab(auto_trade_tab, "Auto-Trade")

        layout.addWidget(tabs)

        log_group = QGroupBox("System Log")
        log_layout = QVBoxLayout()
        try:
            from .widgets import LogWidget
            self.log_widget = LogWidget()
        except ImportError:
            self.log_widget = QTextEdit()
            self.log_widget.setReadOnly(True)
            self.log_widget.setMaximumHeight(150)
        log_layout.addWidget(self.log_widget)
        log_group.setLayout(log_layout)
        layout.addWidget(log_group)

        action_frame = QFrame()
        action_frame.setStyleSheet("""
            QFrame {
                background: #1a1a3e;
                border-radius: 8px;
                padding: 10px;
            }
        """)
        action_layout = QHBoxLayout(action_frame)

        self.buy_btn = QPushButton("BUY")
        self.buy_btn.setStyleSheet("""
            QPushButton {
                background: #4CAF50; color: white; border: none;
                padding: 15px 40px; border-radius: 6px;
                font-weight: bold; font-size: 16px;
            }
            QPushButton:hover { background: #388E3C; }
            QPushButton:disabled { background: #333; color: #666; }
        """)
        self.buy_btn.clicked.connect(self._execute_buy)
        self.buy_btn.setEnabled(False)

        self.sell_btn = QPushButton("SELL")
        self.sell_btn.setStyleSheet("""
            QPushButton {
                background: #F44336; color: white; border: none;
                padding: 15px 40px; border-radius: 6px;
                font-weight: bold; font-size: 16px;
            }
            QPushButton:hover { background: #D32F2F; }
            QPushButton:disabled { background: #333; color: #666; }
        """)
        self.sell_btn.clicked.connect(self._execute_sell)
        self.sell_btn.setEnabled(False)

        action_layout.addWidget(self.buy_btn)
        action_layout.addWidget(self.sell_btn)
        layout.addWidget(action_frame)

        return panel

        # =========================================================================
        # STATUS BAR & TIMERS
    # =========================================================================

    def _setup_statusbar(self):
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

    def _setup_timers(self):
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

    def _apply_professional_style(self):
        """Apply a modern, clean desktop trading theme without changing behavior."""
        self.setFont(QFont("Segoe UI", 10))
        self.setStyleSheet("""
            QMainWindow, QWidget {
                background: #0b1422;
                color: #dbe4f3;
            }

            QMenuBar {
                background: #0f1b2e;
                color: #dbe4f3;
                border-bottom: 1px solid #253754;
                padding: 3px 6px;
            }
            QMenuBar::item {
                padding: 6px 11px;
                border-radius: 7px;
                margin: 2px 2px;
            }
            QMenuBar::item:selected { background: #172742; }
            QMenu {
                background: #0f1b2e;
                color: #dbe4f3;
                border: 1px solid #2d4263;
                padding: 6px;
            }
            QMenu::item {
                padding: 7px 16px;
                border-radius: 6px;
            }
            QMenu::item:selected { background: #1a2c49; }

            QToolBar {
                background: #0f1b2e;
                border: none;
                border-bottom: 1px solid #253754;
                spacing: 8px;
                padding: 6px 8px;
            }
            QToolButton {
                background: #15243d;
                color: #dbe4f3;
                border: 1px solid #2f4466;
                border-radius: 8px;
                padding: 6px 11px;
                font-weight: 600;
            }
            QToolButton:hover {
                background: #1b2f50;
                border-color: #4a7bff;
            }
            QToolButton:pressed { background: #233a61; }

            QGroupBox {
                font-weight: 700;
                font-size: 12px;
                border: 1px solid #253754;
                border-radius: 11px;
                margin-top: 12px;
                padding-top: 12px;
                color: #9ab8ea;
                background: #0f1b2e;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 6px;
            }

            QLabel {
                color: #dbe4f3;
                font-size: 12px;
                background: transparent;
            }

            QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QListWidget {
                min-height: 31px;
                padding: 4px 8px;
                border: 1px solid #324968;
                border-radius: 8px;
                background: #13223a;
                color: #dbe4f3;
                selection-background-color: #2f5fda;
                selection-color: #f8fbff;
            }
            QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus, QListWidget:focus {
                border-color: #4a7bff;
                background: #182b47;
            }

            QComboBox::drop-down {
                border: none;
                width: 22px;
            }

            QTableWidget, QTableView, QTreeView {
                background: #0e1a2d;
                color: #dbe4f3;
                border: 1px solid #253754;
                border-radius: 9px;
                gridline-color: #23334e;
                selection-background-color: #22406d;
                selection-color: #f7fbff;
                alternate-background-color: #101f34;
                outline: none;
            }
            QTableWidget::item, QTableView::item {
                padding: 6px;
                border: none;
            }
            QHeaderView::section {
                background: #172840;
                color: #aac3ec;
                padding: 8px 10px;
                border: none;
                border-right: 1px solid #253754;
                border-bottom: 1px solid #253754;
                font-weight: 700;
            }

            QTabWidget::pane {
                border: 1px solid #253754;
                background: #0e1a2d;
                border-radius: 9px;
                top: -1px;
            }
            QTabBar::tab {
                background: #13223a;
                color: #9db1d6;
                padding: 9px 16px;
                border-top-left-radius: 7px;
                border-top-right-radius: 7px;
                margin-right: 3px;
                min-width: 72px;
            }
            QTabBar::tab:selected {
                background: #1b3150;
                color: #e8f0ff;
                border: 1px solid #38537a;
                border-bottom: 1px solid #1b3150;
            }
            QTabBar::tab:hover:!selected {
                color: #c9daf7;
                background: #172a45;
            }

            QPushButton {
                background: #1c3253;
                color: #eaf1ff;
                border: 1px solid #3d5f8f;
                border-radius: 8px;
                padding: 8px 14px;
                font-weight: 700;
            }
            QPushButton:hover {
                background: #24416b;
                border-color: #4a7bff;
            }
            QPushButton:pressed { background: #2a4977; }
            QPushButton:disabled {
                background: #12223a;
                color: #6b7d9c;
                border-color: #253754;
            }

            QCheckBox, QRadioButton {
                spacing: 7px;
                color: #dbe4f3;
                background: transparent;
            }
            QCheckBox::indicator, QRadioButton::indicator {
                width: 16px;
                height: 16px;
            }
            QCheckBox::indicator {
                border: 1px solid #3b5479;
                border-radius: 4px;
                background: #13223a;
            }
            QCheckBox::indicator:checked {
                background: #2f5fda;
                border-color: #2f5fda;
            }

            QTextEdit, QPlainTextEdit {
                background: #0c1728;
                color: #cde8d7;
                border: 1px solid #253754;
                border-radius: 9px;
                font-family: 'Consolas', 'Cascadia Mono', monospace;
                padding: 6px;
                selection-background-color: #2f5fda;
                selection-color: #f8fbff;
            }

            QProgressBar {
                border: 1px solid #304968;
                background: #101f34;
                border-radius: 7px;
                text-align: center;
                color: #dbe4f3;
                min-height: 18px;
            }
            QProgressBar::chunk {
                border-radius: 6px;
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:0,
                    stop:0 #2f6be0, stop:1 #39b982
                );
            }

            QStatusBar {
                background: #0f1b2e;
                color: #9db1d6;
                border-top: 1px solid #253754;
            }

            QScrollBar:vertical {
                background: #0f1b2e;
                width: 11px;
                margin: 2px;
                border-radius: 5px;
            }
            QScrollBar::handle:vertical {
                background: #34507a;
                border-radius: 5px;
                min-height: 24px;
            }
            QScrollBar::handle:vertical:hover { background: #45669c; }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                border: none;
                background: none;
                height: 0;
            }
            QScrollBar:horizontal {
                background: #0f1b2e;
                height: 11px;
                margin: 2px;
                border-radius: 5px;
            }
            QScrollBar::handle:horizontal {
                background: #34507a;
                border-radius: 5px;
                min-width: 24px;
            }
            QScrollBar::handle:horizontal:hover { background: #45669c; }
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
                border: none;
                background: none;
                width: 0;
            }

            QSplitter::handle {
                background: #1a2c47;
            }
            QSplitter::handle:hover {
                background: #2b456b;
            }

            QToolTip {
                background: #1a2c49;
                color: #e9f0ff;
                border: 1px solid #3b5479;
                padding: 6px 8px;
            }
        """)

    # =========================================================================
    # =========================================================================

    def _init_components(self):
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
        except Exception:
            pass
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
            prev_ts = float(self._last_session_cache_write_ts.get(key, 0.0))
            if (now_ts - prev_ts) < float(max(0.0, min_gap_seconds)):
                return
            payload = dict(bar)
            payload["interval"] = iv
            payload["source"] = str(payload.get("source", "") or "tencent_rt")
            wrote = self._session_bar_cache.append_bar(symbol, iv, payload)
            if wrote:
                self._last_session_cache_write_ts[key] = now_ts
        except Exception as e:
            log.debug(f"Session cache persist failed for {symbol}: {e}")

    def _filter_trained_stocks_ui(self, text: str):
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
                except Exception:
                    pass

        # Keep selection aligned with the active symbol.
        try:
            for row in range(self.watchlist.rowCount()):
                item = self.watchlist.item(row, 0)
                if item and self._ui_norm(item.text()) == normalized:
                    self.watchlist.setCurrentCell(row, 0)
                    break
        except Exception:
            pass

    def _on_trained_stock_activated(self, item):
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
        except Exception:
            pass
        try:
            target_lookback = int(
                max(
                    self._recommended_lookback(interval),
                    self._trained_stock_window_bars(interval),
                )
            )
            if int(self.lookback_spin.value()) < target_lookback:
                self.lookback_spin.setValue(target_lookback)
        except Exception:
            pass
        self._pin_watchlist_symbol(code)
        self.stock_input.setText(code)
        try:
            self._ensure_feed_subscription(code)
        except Exception:
            pass
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

    def _update_trained_stocks_ui(self, codes: list[str] | None = None):
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

    def _focus_trained_stocks_tab(self):
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

        def _task():
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
            except Exception:
                pass

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

    def _init_auto_trader(self):
        """Initialize auto-trader on the execution engine."""
        if self.executor and self.predictor:
            try:
                self.executor.init_auto_trader(
                    self.predictor, self.watch_list
                )

                if self.executor.auto_trader:
                    self.executor.auto_trader.on_action = (
                        self._on_auto_trade_action_safe
                    )
                    self.executor.auto_trader.on_pending_approval = (
                        self._on_pending_approval_safe
                    )

                self.log("Auto-trader initialized", "info")
            except Exception as e:
                log.warning(f"Auto-trader init failed: {e}")
        elif self.predictor and not self.executor:
            # Executor not connected yet; will init when connected.
            pass

    def _on_interval_changed(self, interval: str):
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
        except Exception:
            pass

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

    def _seven_day_lookback(self, interval: str) -> int:
        """Return lookback bars representing ~7 trading days for interval."""
        iv = self._normalize_interval_token(interval)
        try:
            from data.fetcher import BARS_PER_DAY
            bpd = float(BARS_PER_DAY.get(iv, 1.0))
        except Exception:
            fallback = {"1m": 240.0, "5m": 48.0, "15m": 16.0, "30m": 8.0, "60m": 4.0, "1h": 4.0, "1d": 1.0}
            bpd = float(fallback.get(iv, 1.0))
        bars = int(max(7, round(7.0 * bpd)))
        return max(50, bars) if iv != "1d" else 7

    def _trained_stock_window_bars(
        self, interval: str, window_days: int = 29
    ) -> int:
        """Return lookback bars representing the trained-stock refresh window."""
        iv = self._normalize_interval_token(interval)
        wd = max(1, int(window_days or 29))
        try:
            from data.fetcher import BARS_PER_DAY
            bpd = float(BARS_PER_DAY.get(iv, 1.0))
        except Exception:
            fallback = {
                "1m": 240.0,
                "2m": 120.0,
                "5m": 48.0,
                "15m": 16.0,
                "30m": 8.0,
                "60m": 4.0,
                "1h": 4.0,
                "1d": 1.0,
                "1wk": 0.2,
                "1mo": 0.05,
            }
            bpd = float(fallback.get(iv, 1.0))
        return int(max(1, round(float(wd) * max(0.01, bpd))))

    def _recommended_lookback(self, interval: str) -> int:
        """
        Recommended lookback for analysis/forecast per interval.
        Startup 1m uses a true 7-day 1m window; higher intervals keep a
        minimum depth for feature generation stability.
        """
        iv = self._normalize_interval_token(interval)
        base = int(self._seven_day_lookback(iv))
        if iv in ("1d", "1wk", "1mo"):
            return max(60, base)
        return max(120, base)

    def _queue_history_refresh(self, symbol: str, interval: str) -> None:
        """Force next history load to bypass memory/session cache once."""
        iv = self._normalize_interval_token(interval)
        sym = self._ui_norm(symbol)
        key = (sym if sym else "*", iv)
        self._history_refresh_once.add(key)

    def _consume_history_refresh(self, symbol: str, interval: str) -> bool:
        """Consume one queued history refresh request for symbol/interval."""
        iv = self._normalize_interval_token(interval)
        sym = self._ui_norm(symbol)
        direct = (sym, iv)
        wildcard = ("*", iv)
        if direct in self._history_refresh_once:
            self._history_refresh_once.discard(direct)
            return True
        if wildcard in self._history_refresh_once:
            self._history_refresh_once.discard(wildcard)
            return True
        return False

    def _schedule_analysis_recovery(
        self,
        symbol: str,
        interval: str,
        warnings: list[str] | None = None,
    ) -> None:
        """
        Retry analysis once with a forced history refresh when output is partial.
        Throttled per symbol/interval to avoid retry loops.
        """
        sym = self._ui_norm(symbol)
        if not sym:
            return
        iv = self._normalize_interval_token(interval)
        key = f"{sym}:{iv}"
        now_ts = time.monotonic()
        last_ts = float(self._analysis_recovery_attempt_ts.get(key, 0.0) or 0.0)
        if (now_ts - last_ts) < 25.0:
            return
        self._analysis_recovery_attempt_ts[key] = now_ts

        self._queue_history_refresh(sym, iv)

        reason = ""
        warn_list = list(warnings or [])
        if warn_list:
            for item in warn_list:
                txt = str(item).strip()
                if txt:
                    reason = txt
                    break
        if reason:
            self.log(
                f"Data warm-up retry for {sym}: {reason}",
                "info",
            )
        else:
            self.log(
                f"Data warm-up retry for {sym}: refreshing history",
                "info",
            )

        def _retry_once():
            selected = self._ui_norm(self.stock_input.text())
            if selected != sym:
                return
            self._analyze_stock()

        QTimer.singleShot(1800, _retry_once)

    def _history_window_bars(self, interval: str) -> int:
        """Rolling chart/session window size (7-day equivalent)."""
        iv = self._normalize_interval_token(interval)
        bars = int(self._seven_day_lookback(iv))
        if iv == "1d":
            return max(7, bars)
        return max(120, bars)

    def _ts_to_epoch(self, ts_raw: Any) -> float:
        """Normalize timestamp-like values to epoch seconds."""
        if ts_raw is None:
            return float(time.time())

        try:
            if isinstance(ts_raw, (int, float)):
                v = float(ts_raw)
                # Treat large numeric timestamps as milliseconds.
                if abs(v) >= 1e11:
                    v = v / 1000.0
                return v
        except Exception:
            pass

        try:
            if isinstance(ts_raw, datetime):
                dt = ts_raw
            else:
                txt = str(ts_raw).strip()
                if not txt:
                    return float(time.time())
                dt = datetime.fromisoformat(txt.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                # Most provider timestamps without tz are China local time.
                try:
                    from zoneinfo import ZoneInfo
                    dt = dt.replace(tzinfo=ZoneInfo("Asia/Shanghai"))
                except Exception:
                    dt = dt.replace(tzinfo=timezone.utc)
            return float(dt.timestamp())
        except Exception:
            return float(time.time())

    def _epoch_to_iso(self, epoch: float) -> str:
        """Canonical ISO timestamp for chart bars."""
        try:
            return datetime.fromtimestamp(
                float(epoch), tz=timezone.utc
            ).isoformat(timespec="seconds")
        except Exception:
            return datetime.now(timezone.utc).isoformat(timespec="seconds")

    def _now_iso(self) -> str:
        """Consistent sortable timestamp for live bars."""
        return datetime.now(timezone.utc).isoformat(timespec="seconds")

    def _merge_bars(
        self,
        base: list[dict[str, Any]],
        extra: list[dict[str, Any]],
        interval: str,
    ) -> list[dict[str, Any]]:
        """Merge+deduplicate bars by timestamp and keep a 7-day rolling window."""
        merged: dict[int, dict[str, Any]] = {}
        iv = self._normalize_interval_token(interval)

        def _upsert(row_in: dict[str, Any]) -> None:
            epoch = self._bar_bucket_epoch(
                row_in.get("_ts_epoch", row_in.get("timestamp", "")),
                iv,
            )
            row = dict(row_in)
            row["_ts_epoch"] = float(epoch)
            row["timestamp"] = self._epoch_to_iso(epoch)

            # Never merge out-of-session intraday rows.
            if not self._is_market_session_timestamp(row["_ts_epoch"], iv):
                return

            key = int(epoch)
            existing = merged.get(key)
            if existing is None:
                merged[key] = row
                return

            existing_final = bool(existing.get("final", True))
            row_final = bool(row.get("final", True))
            if existing_final and not row_final:
                return
            if row_final and not existing_final:
                merged[key] = row
                return

            # Same finality: keep richer bar by volume, otherwise prefer newer row.
            try:
                e_vol = float(existing.get("volume", 0) or 0)
            except Exception:
                e_vol = 0.0
            try:
                r_vol = float(row.get("volume", 0) or 0)
            except Exception:
                r_vol = 0.0
            if r_vol >= e_vol:
                merged[key] = row

        for b in (base or []):
            _upsert(b)
        for b in (extra or []):
            _upsert(b)
        out = list(merged.values())
        out.sort(
            key=lambda x: float(
                x.get(
                    "_ts_epoch",
                    self._ts_to_epoch(x.get("timestamp", "")),
                )
            )
        )

        # Final pass: sanitize OHLC and drop abrupt jumps.
        cleaned: list[dict[str, Any]] = []
        prev_close: float | None = None
        prev_epoch: float | None = None
        for row in out:
            try:
                c = float(row.get("close", 0) or 0)
                o = float(row.get("open", c) or c)
                h = float(row.get("high", c) or c)
                low = float(row.get("low", c) or c)
            except Exception:
                continue
            row_epoch = float(
                self._bar_bucket_epoch(
                    row.get("_ts_epoch", row.get("timestamp")),
                    iv,
                )
            )
            ref_close = prev_close
            if (
                prev_epoch is not None
                and self._is_intraday_day_boundary(prev_epoch, row_epoch, iv)
            ):
                ref_close = None

            sanitized = self._sanitize_ohlc(
                o,
                h,
                low,
                c,
                interval=iv,
                ref_close=ref_close,
            )
            if sanitized is None:
                continue

            o, h, low, c = sanitized
            if ref_close and ref_close > 0 and self._is_outlier_tick(
                ref_close, c, interval=iv
            ):
                continue

            row_out = dict(row)
            row_out["open"] = o
            row_out["high"] = h
            row_out["low"] = low
            row_out["close"] = c
            cleaned.append(row_out)
            prev_close = c
            prev_epoch = row_epoch

        keep = self._history_window_bars(interval)
        return cleaned[-keep:]

    def _interval_seconds(self, interval: str) -> int:
        """Map UI interval token to candle duration in seconds."""
        iv = self._normalize_interval_token(interval)
        mapping = {
            "1m": 60,
            "5m": 300,
            "15m": 900,
            "30m": 1800,
            "60m": 3600,
            "1d": 86400,
        }
        if iv in mapping:
            return int(mapping[iv])
        # Generic support for provider labels like "90m" / "30s".
        try:
            if iv.endswith("m"):
                return max(1, int(float(iv[:-1])) * 60)
            if iv.endswith("s"):
                return max(1, int(float(iv[:-1])))
        except Exception:
            pass
        return 60

    def _interval_token_from_seconds(self, seconds: Any) -> str | None:
        """Best-effort inverse mapping from seconds to interval token."""
        try:
            sec = max(1, int(float(seconds)))
        except Exception:
            return None
        known = {
            60: "1m",
            300: "5m",
            900: "15m",
            1800: "30m",
            3600: "60m",
            86400: "1d",
        }
        if sec in known:
            return known[sec]
        if sec % 60 == 0:
            return f"{int(sec // 60)}m"
        return f"{sec}s"

    def _bars_needed_from_base_interval(
        self,
        target_interval: str,
        target_bars: int,
        base_interval: str = "1m",
    ) -> int:
        """
        Estimate how many base-interval bars are needed to render
        `target_bars` in `target_interval`.
        """
        tgt = self._normalize_interval_token(target_interval)
        base = self._normalize_interval_token(base_interval)
        tgt_n = max(1, int(target_bars))

        try:
            src_sec = float(max(1, self._interval_seconds(base)))
            tgt_sec = float(max(1, self._interval_seconds(tgt)))
            factor = int(max(1, math.ceil(tgt_sec / src_sec)))
        except Exception:
            factor = 1

        # CN market has about 240 one-minute bars per full session day.
        if tgt == "1d":
            factor = max(factor, 240)
        elif tgt == "1wk":
            factor = max(factor, 240 * 5)
        elif tgt == "1mo":
            factor = max(factor, 240 * 20)

        return int(max(tgt_n, (tgt_n * factor) + factor))

    def _resample_chart_bars(
        self,
        bars: list[dict[str, Any]],
        source_interval: str,
        target_interval: str,
    ) -> list[dict[str, Any]]:
        """
        Aggregate OHLC bars from source interval to target interval.
        Keeps candle integrity (open/close ordering, high/low envelope).
        """
        src = self._normalize_interval_token(source_interval)
        tgt = self._normalize_interval_token(target_interval)
        if src == tgt:
            return list(bars or [])
        if not bars:
            return []

        src_sec = int(max(1, self._interval_seconds(src)))
        tgt_sec = int(max(1, self._interval_seconds(tgt)))
        if tgt_sec <= src_sec:
            return list(bars or [])

        ranked = sorted(
            list(bars or []),
            key=lambda row: float(
                self._ts_to_epoch(
                    row.get("_ts_epoch", row.get("timestamp", row.get("time")))
                )
            ),
        )

        buckets: dict[str, dict[str, Any]] = {}
        for row in ranked:
            try:
                ep = float(
                    self._ts_to_epoch(
                        row.get("_ts_epoch", row.get("timestamp", row.get("time")))
                    )
                )
            except Exception:
                continue
            if not math.isfinite(ep):
                continue

            day_key = self._bar_trading_date(ep)
            if tgt == "1d":
                key = str(day_key) if day_key is not None else str(int(ep // 86400))
            elif tgt == "1wk":
                if day_key is None:
                    key = f"week:{int(ep // (86400 * 7))}"
                else:
                    iso = day_key.isocalendar()
                    key = f"week:{int(iso.year)}-{int(iso.week):02d}"
            elif tgt == "1mo":
                if day_key is None:
                    dt = datetime.fromtimestamp(ep)
                    key = f"month:{dt.year}-{dt.month:02d}"
                else:
                    key = f"month:{day_key.year}-{day_key.month:02d}"
            else:
                key = f"slot:{int(self._bar_bucket_epoch(ep, tgt))}"

            try:
                o = float(row.get("open", 0) or 0)
                h = float(row.get("high", 0) or 0)
                low = float(row.get("low", 0) or 0)
                c = float(row.get("close", 0) or 0)
            except Exception:
                continue
            if c <= 0 or not all(math.isfinite(v) for v in (o, h, low, c)):
                continue

            if key not in buckets:
                try:
                    vol = float(row.get("volume", 0) or 0.0)
                except Exception:
                    vol = 0.0
                buckets[key] = {
                    "open": o if o > 0 else c,
                    "high": max(h, o, c),
                    "low": min(low, o, c),
                    "close": c,
                    "volume": max(0.0, vol),
                    "_ts_epoch": float(ep),
                    "final": bool(row.get("final", True)),
                    "interval": tgt,
                }
                continue

            cur = buckets[key]
            cur["high"] = float(max(float(cur["high"]), h, o, c))
            cur["low"] = float(min(float(cur["low"]), low, o, c))
            cur["close"] = float(c)
            cur["_ts_epoch"] = float(max(float(cur["_ts_epoch"]), ep))
            cur["final"] = bool(cur.get("final", True) and bool(row.get("final", True)))
            try:
                cur["volume"] = float(cur.get("volume", 0.0)) + max(
                    0.0,
                    float(row.get("volume", 0) or 0.0),
                )
            except Exception:
                pass

        out: list[dict[str, Any]] = []
        for val in buckets.values():
            row_out = dict(val)
            row_out["timestamp"] = self._epoch_to_iso(float(row_out["_ts_epoch"]))
            out.append(row_out)

        out.sort(key=lambda row: float(row.get("_ts_epoch", 0.0)))
        return self._merge_bars([], out, tgt)

    def _dominant_bar_interval(
        self,
        bars: list[dict[str, Any]] | None,
        fallback: str = "1m",
    ) -> str:
        """Most frequent interval token in bar list (best effort)."""
        counts: dict[str, int] = {}
        for row in (bars or []):
            if not isinstance(row, dict):
                continue
            iv = self._normalize_interval_token(
                row.get("interval"),
                fallback="",
            )
            if not iv:
                continue
            counts[iv] = int(counts.get(iv, 0)) + 1
        if not counts:
            return self._normalize_interval_token(fallback)
        best_iv = max(counts.items(), key=lambda kv: kv[1])[0]
        return self._normalize_interval_token(best_iv, fallback=fallback)

    def _effective_anchor_price(
        self,
        symbol: str,
        candidate: float | None = None,
    ) -> float:
        """
        Resolve a robust anchor price for chart scale repair.
        Prefers live/watchlist quote when candidate is obviously off-scale.
        """
        sym = self._ui_norm(symbol)
        try:
            base = float(candidate or 0.0)
        except Exception:
            base = 0.0
        if not math.isfinite(base) or base <= 0:
            base = 0.0

        alt = 0.0
        try:
            rec = self._last_watchlist_price_ui.get(sym)
            if rec is not None:
                alt = float(rec[1] or 0.0)
        except Exception:
            alt = 0.0
        if not math.isfinite(alt) or alt <= 0:
            alt = 0.0

        # Try live quote only when needed to avoid excess calls.
        if alt <= 0 and (base <= 0 or base < 5.0):
            fetcher = None
            try:
                if self.predictor is not None:
                    fetcher = getattr(self.predictor, "fetcher", None)
            except Exception:
                fetcher = None
            if fetcher is not None and sym:
                try:
                    q = fetcher.get_realtime(sym)
                    alt = float(getattr(q, "price", 0) or 0.0) if q is not None else 0.0
                except Exception:
                    alt = 0.0
                if not math.isfinite(alt) or alt <= 0:
                    alt = 0.0

        if base > 0 and alt > 0:
            ratio = max(base, alt) / max(min(base, alt), 1e-8)
            # If candidate differs by 30x+, trust live/watchlist anchor.
            if ratio >= 30.0:
                return float(alt)
            return float(base)
        if alt > 0:
            return float(alt)
        return float(base)

    def _stabilize_chart_depth(
        self,
        symbol: str,
        interval: str,
        candidate: list[dict[str, Any]] | None,
    ) -> list[dict[str, Any]]:
        """
        Avoid replacing a healthy deep window with a transient tiny window.
        """
        cand = list(candidate or [])
        if not cand:
            return cand

        sym = self._ui_norm(symbol)
        iv = self._normalize_interval_token(interval)
        existing_all = list(self._bars_by_symbol.get(sym) or [])
        if not existing_all:
            return cand
        existing = [
            b for b in existing_all
            if self._normalize_interval_token(
                b.get("interval", iv),
                fallback=iv,
            ) == iv
        ]
        if not existing:
            return cand

        old_len = len(existing)
        new_len = len(cand)
        # Protect even medium-depth windows (for example 20-40 bars) from
        # being replaced by transient 1-5 bar snapshots.
        if old_len < 12 or new_len >= max(6, int(old_len * 0.45)):
            return cand

        merged = self._merge_bars(existing, cand, iv)
        if len(merged) >= max(new_len, int(old_len * 0.62)):
            out = merged
        else:
            out = existing

        self._debug_console(
            f"chart_depth_stabilize:{sym}:{iv}",
            (
                f"depth stabilization for {sym} {iv}: "
                f"new={new_len} old={old_len} final={len(out)}"
            ),
            min_gap_seconds=1.0,
            level="info",
        )
        return out

    def _bar_bucket_epoch(self, ts_raw: Any, interval: str) -> float:
        """Floor any timestamp to the interval bucket start (epoch seconds)."""
        epoch = self._ts_to_epoch(ts_raw)
        step = float(max(1, self._interval_seconds(interval)))
        return float(int(epoch // step) * int(step))

    def _bar_trading_date(self, ts_raw: Any) -> object | None:
        """Best-effort Shanghai trading date for a timestamp-like value."""
        try:
            epoch = float(self._ts_to_epoch(ts_raw))
        except Exception:
            return None
        try:
            from zoneinfo import ZoneInfo
            dt_val = datetime.fromtimestamp(epoch, tz=ZoneInfo("Asia/Shanghai"))
        except Exception:
            try:
                dt_val = datetime.fromtimestamp(epoch)
            except Exception:
                return None
        try:
            return dt_val.date()
        except Exception:
            return None

    def _is_intraday_day_boundary(
        self,
        prev_ts_raw: Any,
        cur_ts_raw: Any,
        interval: str,
    ) -> bool:
        """True when two intraday bars fall on different Shanghai trading dates."""
        iv = self._normalize_interval_token(interval)
        if iv in ("1d", "1wk", "1mo"):
            return False
        prev_day = self._bar_trading_date(prev_ts_raw)
        cur_day = self._bar_trading_date(cur_ts_raw)
        if prev_day is None or cur_day is None:
            return False
        return bool(cur_day != prev_day)

    def _shanghai_now(self) -> datetime:
        """Current time in Asia/Shanghai when zoneinfo is available."""
        try:
            from zoneinfo import ZoneInfo
            return datetime.now(tz=ZoneInfo("Asia/Shanghai"))
        except Exception:
            return datetime.now()

    def _is_cn_trading_day(self, day_obj) -> bool:
        """Best-effort CN trading-day check (weekday + optional holiday calendar)."""
        try:
            if day_obj.weekday() >= 5:
                return False
        except Exception:
            return False

        try:
            from core.constants import is_trading_day
            return bool(is_trading_day(day_obj))
        except Exception:
            return True

    def _market_hours_text(self) -> str:
        """Human-readable CN session hours."""
        t = CONFIG.trading
        return (
            f"{t.market_open_am.strftime('%H:%M')}-{t.market_close_am.strftime('%H:%M')}, "
            f"{t.market_open_pm.strftime('%H:%M')}-{t.market_close_pm.strftime('%H:%M')} CST"
        )

    def _next_market_open(self, now_sh: datetime | None = None) -> datetime | None:
        """Next CN market open timestamp in Shanghai time."""
        now_val = now_sh or self._shanghai_now()
        t = CONFIG.trading

        for days_ahead in range(0, 15):
            day_val = (now_val + timedelta(days=days_ahead)).date()
            if not self._is_cn_trading_day(day_val):
                continue

            open_am = now_val.replace(
                year=day_val.year,
                month=day_val.month,
                day=day_val.day,
                hour=t.market_open_am.hour,
                minute=t.market_open_am.minute,
                second=0,
                microsecond=0,
            )
            open_pm = now_val.replace(
                year=day_val.year,
                month=day_val.month,
                day=day_val.day,
                hour=t.market_open_pm.hour,
                minute=t.market_open_pm.minute,
                second=0,
                microsecond=0,
            )

            if days_ahead > 0:
                return open_am

            cur_time = now_val.time()
            if cur_time < t.market_open_am:
                return open_am
            if t.market_close_am < cur_time < t.market_open_pm:
                return open_pm
            if cur_time > t.market_close_pm:
                continue

        return None

    def _is_market_session_timestamp(self, ts_raw: Any, interval: str) -> bool:
        """True when timestamp falls inside CN trading session for intraday intervals."""
        iv = self._normalize_interval_token(interval)
        if iv in ("1d", "1wk", "1mo"):
            return True

        epoch = self._ts_to_epoch(ts_raw)
        try:
            from zoneinfo import ZoneInfo
            dt_val = datetime.fromtimestamp(float(epoch), tz=ZoneInfo("Asia/Shanghai"))
        except Exception:
            dt_val = datetime.fromtimestamp(float(epoch))

        if not self._is_cn_trading_day(dt_val.date()):
            return False

        cur_time = dt_val.time()
        t = CONFIG.trading
        morning = t.market_open_am <= cur_time <= t.market_close_am
        afternoon = t.market_open_pm <= cur_time <= t.market_close_pm
        return bool(morning or afternoon)

    def _filter_bars_to_market_session(
        self,
        bars: list[dict[str, Any]],
        interval: str,
    ) -> list[dict[str, Any]]:
        """Drop out-of-session intraday bars before chart rendering."""
        iv = self._normalize_interval_token(interval)
        if iv in ("1d", "1wk", "1mo"):
            return list(bars or [])

        out: list[dict[str, Any]] = []
        for b in (bars or []):
            ts_raw = b.get("_ts_epoch", b.get("timestamp", b.get("time")))
            if self._is_market_session_timestamp(ts_raw, iv):
                out.append(b)
        return out

    def _bar_safety_caps(self, interval: str) -> tuple[float, float]:
        """
        Return (max_jump_pct, max_range_pct) for bar sanitization.
        Values are intentionally conservative for intraday feeds.
        """
        iv = self._normalize_interval_token(interval)
        if iv == "1m":
            return 0.08, 0.006
        if iv == "5m":
            return 0.10, 0.012
        if iv in ("15m", "30m"):
            return 0.14, 0.020
        if iv in ("60m", "1h"):
            return 0.18, 0.040
        if iv in ("1d", "1wk", "1mo"):
            return 0.24, 0.22
        return 0.20, 0.15

    def _synthetic_tick_jump_cap(self, interval: str) -> float:
        """
        Stricter jump cap for tick-driven synthetic bar updates.
        Prevents stale or spiky quotes from creating giant intraday bodies.
        """
        iv = self._normalize_interval_token(interval)
        if iv == "1m":
            return 0.012
        if iv == "5m":
            return 0.018
        if iv in ("15m", "30m"):
            return 0.028
        if iv in ("60m", "1h"):
            return 0.045
        if iv in ("1d", "1wk", "1mo"):
            return 0.12
        return 0.03

    def _sanitize_ohlc(
        self,
        o: float,
        h: float,
        low: float,
        c: float,
        interval: str,
        ref_close: float | None = None,
    ) -> tuple[float, float, float, float] | None:
        """
        Normalize and clamp OHLC values to avoid malformed long candles
        from bad ticks/partial bars.
        """
        try:
            o = float(o or 0.0)
            h = float(h or 0.0)
            low = float(low or 0.0)
            c = float(c or 0.0)
        except Exception:
            return None
        if not all(math.isfinite(v) for v in (o, h, low, c)):
            return None
        if c <= 0:
            return None

        ref = float(ref_close or 0.0)
        if not math.isfinite(ref) or ref <= 0:
            ref = 0.0

        if o <= 0:
            o = c
        if h <= 0:
            h = max(o, c)
        if low <= 0:
            low = min(o, c)
        if h < low:
            h, low = low, h

        jump_cap, range_cap = self._bar_safety_caps(interval)
        iv = self._normalize_interval_token(interval)
        if ref > 0:
            effective_range_cap = float(range_cap)
        else:
            bootstrap_cap = (
                0.30
                if iv in ("1d", "1wk", "1mo")
                else float(max(0.008, min(0.020, range_cap * 2.0)))
            )
            effective_range_cap = float(max(range_cap, bootstrap_cap))
        if ref > 0:
            jump = abs(c / ref - 1.0)
            if jump > jump_cap:
                return None

        # Keep malformed opens from inflating body/range caps.
        anchor = ref if ref > 0 else c
        if anchor <= 0:
            anchor = c
        max_body = float(anchor) * float(max(jump_cap * 1.25, effective_range_cap * 0.9))
        if max_body > 0 and abs(o - c) > max_body:
            if ref > 0 and abs(c / ref - 1.0) <= jump_cap:
                o = ref
            else:
                o = c

        top = max(o, c)
        bot = min(o, c)
        if h < top:
            h = top
        if low > bot:
            low = bot
        if h < low:
            h, low = low, h

        max_range = float(anchor) * float(effective_range_cap)
        curr_range = max(0.0, h - low)
        if max_range > 0 and curr_range > max_range:
            body = max(0.0, top - bot)
            if body > max_range:
                # Body this large is likely a corrupt open/close pair.
                o = c
                top = c
                bot = c
                body = 0.0
            wick_allow = max(0.0, max_range - body)
            h = min(h, top + (wick_allow * 0.5))
            low = max(low, bot - (wick_allow * 0.5))
            if h < low:
                h, low = low, h

        o = min(max(o, low), h)
        c = min(max(c, low), h)

        # Final hard-stop: drop anything still outside allowed envelope.
        if anchor > 0 and (h - low) > (float(anchor) * float(effective_range_cap) * 1.05):
            return None

        return o, h, low, c

    def _is_outlier_tick(
        self, prev_price: float, new_price: float, interval: str = "1m"
    ) -> bool:
        """
        Guard against bad ticks creating abnormal long candles.
        Uses interval-aware thresholds to avoid rejecting valid fast moves.
        """
        prev = float(prev_price or 0.0)
        new = float(new_price or 0.0)
        if prev <= 0 or new <= 0:
            return False
        jump_cap, _ = self._bar_safety_caps(interval)
        jump_pct = abs(new / prev - 1.0)
        return jump_pct > float(jump_cap)

        # =========================================================================
        # REAL-TIME MONITORING
    # =========================================================================

    def _toggle_monitoring(self, checked):
        """Toggle real-time monitoring"""
        if checked:
            self._start_monitoring()
        else:
            self._stop_monitoring()

    def _start_monitoring(self):
        """Start real-time monitoring safely (no orphan threads)."""
        if self.monitor and self.monitor.isRunning():
            self._stop_monitoring()

        if self.predictor is None or self.predictor.ensemble is None:
            self.log("Cannot start monitoring: No model loaded", "error")
            self.monitor_action.setChecked(False)
            return

        requested_interval = self._normalize_interval_token(
            self.interval_combo.currentText()
        )
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
            except Exception:
                pass
            self.log(
                f"Subscribed to feeds for {len(self.watch_list)} stocks",
                "info"
            )
        except Exception as e:
            self.log(f"Feed subscription warning: {e}", "warning")

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
            lambda e: self.log(f"Monitor: {e}", "warning")
        )
        self.monitor.status_changed.connect(
            lambda s: self.monitor_label.setText(f"Monitoring: {s}")
        )
        self.monitor.start()

        self.monitor_label.setText("Monitoring: ACTIVE")
        self.monitor_label.setStyleSheet(
            "color: #4CAF50; font-weight: bold;"
        )
        self.monitor_action.setText("Stop Monitoring")

        if (
            monitor_interval != requested_interval
            or int(monitor_horizon) != int(requested_horizon)
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
                "success"
            )

    def _stop_monitoring(self):
        """Stop real-time monitoring"""
        if self.monitor:
            self.monitor.stop()
            self.monitor.wait(3000)
            self.monitor = None

        self.monitor_label.setText("Monitoring: OFF")
        self.monitor_label.setStyleSheet("color: #888;")
        self.monitor_action.setText("Start Monitoring")
        self.monitor_action.setChecked(False)

        self.log("Real-time monitoring stopped", "info")

    def _on_signal_detected(self, pred):
        """Handle detected trading signal"""
        Signal = _lazy_get("models.predictor", "Signal")

        row = 0
        self.signals_table.insertRow(row)

        self.signals_table.setItem(row, 0, QTableWidgetItem(
            pred.timestamp.strftime("%H:%M:%S")
            if hasattr(pred, 'timestamp') else "--"
        ))

        stock_text = f"{pred.stock_code}"
        if hasattr(pred, 'stock_name') and pred.stock_name:
            stock_text += f" - {pred.stock_name}"
        self.signals_table.setItem(row, 1, QTableWidgetItem(stock_text))

        signal_text = (
            pred.signal.value
            if hasattr(pred.signal, 'value')
            else str(pred.signal)
        )
        signal_item = QTableWidgetItem(signal_text)

        if hasattr(pred, 'signal') and pred.signal in [
            Signal.STRONG_BUY, Signal.BUY
        ]:
            signal_item.setForeground(QColor("#4CAF50"))
        else:
            signal_item.setForeground(QColor("#F44336"))
        self.signals_table.setItem(row, 2, signal_item)

        conf = pred.confidence if hasattr(pred, 'confidence') else 0
        self.signals_table.setItem(
            row, 3, QTableWidgetItem(f"{conf:.0%}")
        )

        price = pred.current_price if hasattr(pred, 'current_price') else 0
        self.signals_table.setItem(
            row, 4, QTableWidgetItem(f"CNY {price:.2f}")
        )

        action_btn = QPushButton("Trade")
        action_btn.clicked.connect(lambda: self._quick_trade(pred))
        self.signals_table.setCellWidget(row, 5, action_btn)

        # Keep only last 50 signals
        while self.signals_table.rowCount() > 50:
            self.signals_table.removeRow(
                self.signals_table.rowCount() - 1
            )

        self.log(
            f"SIGNAL: {signal_text} - {pred.stock_code} @ CNY {price:.2f}",
            "success"
        )

        QApplication.alert(self)

    def _on_price_updated(self, code: str, price: float):
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
        except Exception:
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
                        except Exception:
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
                except Exception:
                    pass

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
                except Exception as e:
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
                    except Exception as e:
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

        def do_forecast():
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

        def on_done(res):
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
                except Exception:
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
                        except Exception:
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
                except Exception:
                    display_current = 0.0
                if display_current <= 0:
                    try:
                        arr_tmp = self._bars_by_symbol.get(code) or []
                        if arr_tmp:
                            display_current = float(arr_tmp[-1].get("close", 0.0) or 0.0)
                    except Exception:
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
                    except Exception:
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
        def on_error(_e):
            self.workers.pop("forecast_refresh", None)
            if self._forecast_refresh_symbol == code:
                self._forecast_refresh_symbol = ""
        worker.error.connect(on_error)
        worker.start()

    def _refresh_live_chart_forecast(self):
        """
        Periodic chart refresh for selected symbol.
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
            q = fm.get_quote(code)
            if q and float(getattr(q, "price", 0) or 0) > 0:
                self._on_price_updated(code, float(q.price))
                return
        except Exception:
            pass
        try:
            from data.fetcher import get_fetcher
            q2 = get_fetcher().get_realtime(code)
            if q2 and float(getattr(q2, "price", 0) or 0) > 0:
                self._on_price_updated(code, float(q2.price))
        except Exception:
            pass

    def _get_levels_dict(self) -> dict[str, float] | None:
        """Get trading levels as dict"""
        if (
            not self.current_prediction
            or not hasattr(self.current_prediction, 'levels')
        ):
            return None

        levels = self.current_prediction.levels
        return {
            "stop_loss": getattr(levels, 'stop_loss', 0),
            "target_1": getattr(levels, 'target_1', 0),
            "target_2": getattr(levels, 'target_2', 0),
            "target_3": getattr(levels, 'target_3', 0),
        }

    def _scrub_chart_bars(
        self,
        bars: list[dict[str, Any]] | None,
        interval: str,
        *,
        symbol: str = "",
        anchor_price: float | None = None,
    ) -> list[dict[str, Any]]:
        """
        Prepare bars for charting and never fall back to unsanitized rows.
        """
        arr_in = list(bars or [])
        arr_out = self._prepare_chart_bars_for_interval(
            arr_in,
            interval,
            symbol=symbol,
        )
        if arr_in and len(arr_in) >= 25 and len(arr_out) <= 2:
            recovered = self._recover_chart_bars_from_close(
                arr_in,
                interval=interval,
                symbol=symbol,
                anchor_price=anchor_price,
            )
            if len(recovered) > len(arr_out):
                arr_out = recovered
        if arr_out:
            arr_out = self._rescale_chart_bars_to_anchor(
                arr_out,
                anchor_price=anchor_price,
                interval=interval,
                symbol=symbol,
            )
        if arr_in and not arr_out:
            recovered = self._recover_chart_bars_from_close(
                arr_in,
                interval=interval,
                symbol=symbol,
                anchor_price=anchor_price,
            )
            if recovered:
                return recovered
            iv = self._normalize_interval_token(interval)
            sym = self._ui_norm(symbol)
            self._debug_console(
                f"chart_scrub_empty:{sym or 'active'}:{iv}",
                (
                    f"chart scrub removed all rows: symbol={sym or '--'} "
                    f"iv={iv} raw={len(arr_in)}"
                ),
                min_gap_seconds=1.0,
            )
        return arr_out

    def _rescale_chart_bars_to_anchor(
        self,
        bars: list[dict[str, Any]],
        *,
        anchor_price: float | None,
        interval: str,
        symbol: str = "",
    ) -> list[dict[str, Any]]:
        """
        Repair obvious price-scale mismatches (e.g., 1.5 vs 1500) so bars
        are not fully dropped by jump filters.
        """
        arr = list(bars or [])
        if not arr:
            return []
        try:
            anchor = float(anchor_price or 0.0)
        except Exception:
            anchor = 0.0
        if not math.isfinite(anchor) or anchor <= 0:
            return arr

        closes: list[float] = []
        for row in arr:
            try:
                c = float(row.get("close", 0) or 0)
            except Exception:
                c = 0.0
            if c > 0 and math.isfinite(c):
                closes.append(c)
        if len(closes) < 5:
            return arr

        med = float(median(closes[-min(80, len(closes)):]))
        if med <= 0 or not math.isfinite(med):
            return arr

        raw_ratio = anchor / med
        if 0.2 <= raw_ratio <= 5.0:
            return arr

        candidates = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
        best_scale = 1.0
        best_err = float("inf")
        for s in candidates:
            try:
                ratio = (med * float(s)) / anchor
                if ratio <= 0 or not math.isfinite(ratio):
                    continue
                err = abs(math.log(ratio))
                if err < best_err:
                    best_err = err
                    best_scale = float(s)
            except Exception:
                continue

        scaled_ratio = (med * best_scale) / anchor if anchor > 0 else 1.0
        if not (0.2 <= scaled_ratio <= 5.0):
            return arr
        if abs(best_scale - 1.0) < 1e-9:
            return arr

        out: list[dict[str, Any]] = []
        for row in arr:
            item = dict(row)
            for key in ("open", "high", "low", "close"):
                try:
                    v = float(item.get(key, 0) or 0)
                except Exception:
                    v = 0.0
                if v > 0 and math.isfinite(v):
                    item[key] = float(v * best_scale)
            out.append(item)

        iv = self._normalize_interval_token(interval)
        sym = self._ui_norm(symbol)
        self._debug_console(
            f"chart_scale_fix:{sym or 'active'}:{iv}",
            (
                f"applied scale fix x{best_scale:g} for {sym or '--'} {iv}: "
                f"median={med:.6f} anchor={anchor:.6f}"
            ),
            min_gap_seconds=1.0,
            level="info",
        )
        return out

    def _recover_chart_bars_from_close(
        self,
        bars: list[dict[str, Any]],
        *,
        interval: str,
        symbol: str = "",
        anchor_price: float | None = None,
    ) -> list[dict[str, Any]]:
        """
        Minimal recovery path when strict scrub drops all bars.
        Builds stable OHLC from close/prev-close so chart remains usable.
        """
        iv = self._normalize_interval_token(interval)
        def _build(enforce_session: bool) -> list[dict[str, Any]]:
            merged: dict[int, dict[str, Any]] = {}
            prev_close: float | None = None
            prev_epoch: float | None = None
            for row in list(bars or []):
                if not isinstance(row, dict):
                    continue
                row_iv = self._normalize_interval_token(
                    row.get("interval", iv),
                    fallback=iv,
                )
                if row_iv != iv:
                    continue
                epoch = self._bar_bucket_epoch(
                    row.get("_ts_epoch", row.get("timestamp")),
                    iv,
                )
                if enforce_session and (not self._is_market_session_timestamp(epoch, iv)):
                    continue
                ref_close = prev_close
                if (
                    prev_epoch is not None
                    and self._is_intraday_day_boundary(prev_epoch, epoch, iv)
                ):
                    ref_close = None
                try:
                    c = float(
                        row.get("close", row.get("price", 0)) or 0
                    )
                except Exception:
                    c = 0.0
                if c <= 0 or not math.isfinite(c):
                    continue

                try:
                    o = float(row.get("open", 0) or 0)
                except Exception:
                    o = 0.0
                if o <= 0 and ref_close and ref_close > 0:
                    o = float(ref_close)
                if o <= 0:
                    o = c

                try:
                    h = float(row.get("high", max(o, c)) or max(o, c))
                except Exception:
                    h = max(o, c)
                try:
                    low = float(row.get("low", min(o, c)) or min(o, c))
                except Exception:
                    low = min(o, c)
                if iv not in ("1d", "1wk", "1mo"):
                    # Recovery mode should avoid carrying forward vendor day-range
                    # highs/lows into minute bars.
                    h = max(o, c)
                    low = min(o, c)

                s = self._sanitize_ohlc(
                    o,
                    h,
                    low,
                    c,
                    interval=iv,
                    ref_close=ref_close,
                )
                if s is None:
                    continue
                o, h, low, c = s

                key = int(epoch)
                item = {
                    "open": o,
                    "high": h,
                    "low": low,
                    "close": c,
                    "_ts_epoch": float(epoch),
                    "timestamp": self._epoch_to_iso(epoch),
                    "final": bool(row.get("final", True)),
                    "interval": iv,
                }
                existing = merged.get(key)
                if existing is None:
                    merged[key] = item
                else:
                    if bool(item.get("final", True)) and not bool(existing.get("final", True)):
                        merged[key] = item
                prev_close = c
                prev_epoch = float(epoch)

            out_local = list(merged.values())
            out_local.sort(key=lambda x: float(x.get("_ts_epoch", 0.0)))
            return out_local[-self._history_window_bars(iv):]

        out = _build(enforce_session=True)
        if not out:
            out = _build(enforce_session=False)
            if out:
                sym = self._ui_norm(symbol)
                self._debug_console(
                    f"chart_recover_lenient:{sym or 'active'}:{iv}",
                    (
                        f"lenient timestamp recovery enabled for {sym or '--'} {iv}: "
                        f"bars={len(out)}"
                    ),
                    min_gap_seconds=1.0,
                    level="warning",
                )

        out = self._rescale_chart_bars_to_anchor(
            out,
            anchor_price=anchor_price,
            interval=iv,
            symbol=symbol,
        )
        if out:
            sym = self._ui_norm(symbol)
            self._debug_console(
                f"chart_recover:{sym or 'active'}:{iv}",
                (
                    f"recovered chart bars from close-only path: "
                    f"symbol={sym or '--'} iv={iv} bars={len(out)}"
                ),
                min_gap_seconds=1.0,
                level="warning",
            )
        return out

    def _prepare_chart_bars_for_interval(
        self,
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
                except Exception:
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
                except Exception:
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
                except Exception:
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

    def _quick_trade(self, pred):
        """Quick trade from signal"""
        self.stock_input.setText(pred.stock_code)
        self._analyze_stock()

    # =========================================================================
    # =========================================================================

    def _update_watchlist(self):
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
    ):
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
            except Exception:
                pass

        try:
            self._queue_history_refresh(
                code,
                self._normalize_interval_token(self.interval_combo.currentText()),
            )
        except Exception:
            pass

        self._analyze_stock()

    def _add_to_watchlist(self):
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

    def _remove_from_watchlist(self):
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

    def _analyze_stock(self):
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
                    except Exception:
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
                    except Exception:
                        vol = 0.0
                    if (not math.isfinite(vol)) or vol < 0:
                        vol = 0.0
                    try:
                        amount = float(row.get("amount", 0) or 0.0)
                    except Exception:
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
                        except Exception:
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
                        except Exception:
                            vol = 0.0
                        if (not math.isfinite(vol)) or vol < 0:
                            vol = 0.0
                        try:
                            amount = float(row.get("amount", 0) or 0.0)
                        except Exception:
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
                except Exception:
                    e_vol = 0.0
                try:
                    r_vol = float(row.get("volume", 0) or 0)
                except Exception:
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
                    except Exception:
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
                    except Exception:
                        pass

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
        except Exception as e:
            log.debug(f"Historical chart load failed for {symbol}: {e}")
            return []

    def _on_analysis_done(self, pred):
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
        except Exception:
            pass
        if not (self.monitor and self.monitor.isRunning()):
            try:
                self.monitor_action.setChecked(True)
                self._start_monitoring()
            except Exception:
                pass

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

    def _on_analysis_error(self, error: str):
        """Handle analysis error"""
        self.analyze_action.setEnabled(True)
        self.progress.hide()
        self.status_label.setText("Ready")

        self.log(f"Analysis failed: {error}", "error")
        QMessageBox.warning(self, "Error", f"Analysis failed:\n{error}")

        self.workers.pop('analyze', None)

    def _update_details(self, pred):
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

        def safe_get(obj, attr, default=0):
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
            except Exception:
                pass

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

    def _add_to_history(self, pred):
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

    def _refresh_guess_rows_for_symbol(self, code: str, price: float):
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

    def _update_correct_guess_profit_ui(self):
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

    def _scan_stocks(self):
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
        worker.error.connect(
            lambda e: (
                self.log(f"Scan failed: {e}", "error"),
                self.progress.hide(),
                self.workers.pop('scan', None),
            )
        )
        self.workers['scan'] = worker
        worker.start()

    def _on_scan_done(self, picks):
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

    def _refresh_all(self):
        """Refresh all data"""
        self._update_watchlist()
        self._refresh_portfolio()
        self.log("Refreshed all data", "info")

    # =========================================================================
    # =========================================================================

    def _toggle_trading(self):
        """Toggle trading connection"""
        if self.executor is None:
            self._connect_trading()
        else:
            self._disconnect_trading()

    def _on_mode_combo_changed(self, index: int):
        if self._syncing_mode_ui:
            return
        mode = TradingMode.SIMULATION if int(index) == 0 else TradingMode.LIVE
        self._set_trading_mode(mode, prompt_reconnect=True)

    def _set_trading_mode(
        self,
        mode: TradingMode,
        prompt_reconnect: bool = False,
    ) -> None:
        mode = TradingMode.LIVE if mode == TradingMode.LIVE else TradingMode.SIMULATION
        try:
            CONFIG.trading_mode = mode
        except Exception as e:
            log.warning(f"Failed to set trading mode config: {e}")

        self._syncing_mode_ui = True
        try:
            self.mode_combo.setCurrentIndex(0 if mode != TradingMode.LIVE else 1)
            if hasattr(self, "paper_action"):
                self.paper_action.setChecked(mode != TradingMode.LIVE)
            if hasattr(self, "live_action"):
                self.live_action.setChecked(mode == TradingMode.LIVE)
        finally:
            self._syncing_mode_ui = False

        self.log(f"Trading mode set: {mode.value}", "info")

        if not prompt_reconnect or self.executor is None:
            return

        current = getattr(self.executor, "mode", TradingMode.SIMULATION)
        if current == mode:
            return

        reply = QMessageBox.question(
            self,
            "Reconnect Required",
            "Trading mode changed. Reconnect now to apply new mode?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self._disconnect_trading()
            self._connect_trading()

    def _connect_trading(self):
        """Connect to trading system"""
        mode = (
            TradingMode.SIMULATION
            if self.mode_combo.currentIndex() == 0
            else TradingMode.LIVE
        )

        if mode == TradingMode.LIVE:
            try:
                from core.network import get_network_env
                env = get_network_env()
                if not env.is_vpn_active:
                    reply = QMessageBox.warning(
                        self, "VPN Not Detected",
                        "LIVE trading in China typically requires VPN routing.\n\n"
                        "No VPN was detected by the network probe.\n"
                        "If you are on VPN, set TRADING_VPN=1 and retry.\n\n"
                        "Continue anyway?",
                        QMessageBox.StandardButton.Yes
                        | QMessageBox.StandardButton.No,
                        QMessageBox.StandardButton.No
                    )
                    if reply != QMessageBox.StandardButton.Yes:
                        self.mode_combo.setCurrentIndex(0)
                        return
            except Exception:
                pass

            failed_controls = _collect_live_readiness_failures()
            if failed_controls:
                strict_live = bool(
                    getattr(
                        getattr(CONFIG, "security", None),
                        "strict_live_governance",
                        False,
                    )
                )
                preview = "\n".join(f"- {x}" for x in failed_controls[:10])
                more = ""
                if len(failed_controls) > 10:
                    more = f"\n... and {len(failed_controls) - 10} more"
                msg = (
                    "Institutional live-readiness checks failed.\n\n"
                    f"{preview}{more}\n\n"
                    "Run `python scripts/regulatory_readiness.py` for details."
                )
                if strict_live:
                    QMessageBox.critical(
                        self,
                        "Live Readiness Failed",
                        msg,
                    )
                    self.mode_combo.setCurrentIndex(0)
                    return
                reply = QMessageBox.warning(
                    self,
                    "Live Readiness Warning",
                    msg + "\n\nContinue anyway?",
                    QMessageBox.StandardButton.Yes
                    | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.No,
                )
                if reply != QMessageBox.StandardButton.Yes:
                    self.mode_combo.setCurrentIndex(0)
                    return

            reply = QMessageBox.warning(
                self, "Live Trading Warning",
                "You are switching to LIVE TRADING mode!\n\n"
                "This will use REAL MONEY.\n\n"
                "Are you absolutely sure?",
                QMessageBox.StandardButton.Yes
                | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )
            if reply != QMessageBox.StandardButton.Yes:
                self.mode_combo.setCurrentIndex(0)
                return

        try:
            ExecutionEngine = _lazy_get("trading.executor", "ExecutionEngine")
            self.executor = ExecutionEngine(mode)
            self.executor.on_fill = self._on_order_filled
            self.executor.on_reject = self._on_order_rejected

            if self.executor.start():
                self.connection_status.setText("Connected")
                self.connection_status.setStyleSheet(
                    "color: #4CAF50; font-weight: bold;"
                )
                self.connect_btn.setText("Disconnect")
                self.connect_btn.setStyleSheet("""
                    QPushButton {
                        background: #F44336;
                        color: white;
                        border: none;
                        padding: 12px;
                        border-radius: 6px;
                        font-weight: bold;
                    }
                    QPushButton:hover { background: #D32F2F; }
                """)

                self.log(
                    f"Connected to {mode.value} trading", "success"
                )
                self._refresh_portfolio()

                # Initialize auto-trader after broker connection
                self._init_auto_trader()
                if self._auto_trade_mode != AutoTradeMode.MANUAL:
                    self._apply_auto_trade_mode(self._auto_trade_mode)
            else:
                self.executor = None
                self.log("Failed to connect to broker", "error")
        except Exception as e:
            self.log(f"Connection error: {e}", "error")
            self.executor = None

    def _disconnect_trading(self):
        """Disconnect from trading"""
        if self.executor:
            try:
                self.executor.stop()
            except Exception:
                pass
            self.executor = None

        self.connection_status.setText("Disconnected")
        self.connection_status.setStyleSheet(
            "color: #FF5252; font-weight: bold;"
        )
        self.connect_btn.setText("Connect to Broker")
        self.connect_btn.setStyleSheet("""
            QPushButton {
                background: #4CAF50;
                color: white;
                border: none;
                padding: 12px;
                border-radius: 6px;
                font-weight: bold;
            }
            QPushButton:hover { background: #388E3C; }
        """)

        self.log("Disconnected from broker", "info")

    def _on_chart_trade_requested(self, side: str, price: float) -> None:
        """Handle right-click chart quick trade request."""
        if self.executor is None:
            self.log("Connect broker before trading from chart", "warning")
            return
        symbol = _normalize_stock_code(self.stock_input.text())
        if not symbol and self.current_prediction is not None:
            symbol = _normalize_stock_code(
                getattr(self.current_prediction, "stock_code", "")
            )
        if not symbol:
            self.log("No active symbol for chart trade", "warning")
            return
        if price <= 0:
            self.log("Invalid chart price", "warning")
            return

        try:
            lot = max(1, int(get_lot_size(symbol)))
        except Exception:
            lot = 1

        order_params = self._show_chart_trade_dialog(
            symbol=symbol,
            side=side,
            clicked_price=float(price),
            lot=lot,
        )
        if not order_params:
            return

        order_side = OrderSide.BUY if str(side).lower() == "buy" else OrderSide.SELL
        self._submit_chart_order(
            symbol=symbol,
            side=order_side,
            qty=int(order_params["qty"]),
            price=float(order_params["price"]),
            order_type=str(order_params["order_type"]),
            time_in_force=str(order_params["time_in_force"]),
            trigger_price=float(order_params["trigger_price"]),
            trailing_stop_pct=float(order_params["trailing_stop_pct"]),
            trail_limit_offset_pct=float(order_params["trail_limit_offset_pct"]),
            strict_time_in_force=bool(order_params["strict_time_in_force"]),
            stop_loss=float(order_params["stop_loss"]),
            take_profit=float(order_params["take_profit"]),
            bracket=bool(order_params["bracket"]),
        )

    def _show_chart_trade_dialog(
        self,
        symbol: str,
        side: str,
        clicked_price: float,
        lot: int,
    ) -> dict[str, float | int | str | bool] | None:
        """Collect advanced chart trade parameters from user."""
        from PyQt6.QtWidgets import QDialog, QDialogButtonBox, QFormLayout

        dialog = QDialog(self)
        dialog.setWindowTitle("Chart Quick Trade")
        dialog.setMinimumWidth(420)

        layout = QVBoxLayout(dialog)
        heading = QLabel(
            f"{str(side).upper()} {symbol} | Chart Price: {clicked_price:.2f}"
        )
        heading.setStyleSheet("font-weight: bold;")
        layout.addWidget(heading)

        form = QFormLayout()
        layout.addLayout(form)

        qty_spin = QSpinBox()
        qty_spin.setRange(max(1, lot), 5_000_000)
        qty_spin.setSingleStep(max(1, lot))
        qty_spin.setValue(max(1, lot))
        qty_spin.setSuffix(f" (lot {lot})")
        form.addRow("Quantity:", qty_spin)

        order_type_combo = QComboBox()
        order_types = [
            ("Limit", OrderType.LIMIT.value),
            ("Market", OrderType.MARKET.value),
            ("Stop", OrderType.STOP.value),
            ("Stop Limit", OrderType.STOP_LIMIT.value),
            ("IOC", OrderType.IOC.value),
            ("FOK", OrderType.FOK.value),
            ("Trailing Market", OrderType.TRAIL_MARKET.value),
            ("Trailing Limit", OrderType.TRAIL_LIMIT.value),
        ]
        for label, value in order_types:
            order_type_combo.addItem(label, value)
        form.addRow("Order Type:", order_type_combo)

        tif_combo = QComboBox()
        for label, value in (
            ("DAY", "day"),
            ("GTC", "gtc"),
            ("IOC", "ioc"),
            ("FOK", "fok"),
        ):
            tif_combo.addItem(label, value)
        form.addRow("Time In Force:", tif_combo)

        strict_tif = QCheckBox("Strict TIF (cancel if unsupported)")
        form.addRow("", strict_tif)

        price_spin = QDoubleSpinBox()
        price_spin.setRange(0.01, 1_000_000.0)
        price_spin.setDecimals(3)
        price_spin.setValue(max(0.01, float(clicked_price)))
        price_spin.setSingleStep(max(0.01, float(clicked_price) * 0.002))
        form.addRow("Order Price:", price_spin)

        trigger_spin = QDoubleSpinBox()
        trigger_spin.setRange(0.0, 1_000_000.0)
        trigger_spin.setDecimals(3)
        trigger_spin.setValue(max(0.0, float(clicked_price)))
        trigger_spin.setSingleStep(max(0.01, float(clicked_price) * 0.002))
        form.addRow("Trigger Price:", trigger_spin)

        trailing_spin = QDoubleSpinBox()
        trailing_spin.setRange(0.0, 20.0)
        trailing_spin.setDecimals(2)
        trailing_spin.setSingleStep(0.1)
        trailing_spin.setSuffix(" %")
        trailing_spin.setValue(0.8)
        form.addRow("Trailing Stop:", trailing_spin)

        trail_limit_offset_spin = QDoubleSpinBox()
        trail_limit_offset_spin.setRange(0.0, 10.0)
        trail_limit_offset_spin.setDecimals(2)
        trail_limit_offset_spin.setSingleStep(0.05)
        trail_limit_offset_spin.setSuffix(" %")
        trail_limit_offset_spin.setValue(0.15)
        form.addRow("Trail Limit Offset:", trail_limit_offset_spin)

        bracket_check = QCheckBox("Attach stop-loss / take-profit")
        form.addRow("", bracket_check)

        stop_loss_spin = QDoubleSpinBox()
        stop_loss_spin.setRange(0.0, 1_000_000.0)
        stop_loss_spin.setDecimals(3)
        stop_loss_spin.setValue(0.0)
        form.addRow("Stop-Loss:", stop_loss_spin)

        take_profit_spin = QDoubleSpinBox()
        take_profit_spin.setRange(0.0, 1_000_000.0)
        take_profit_spin.setDecimals(3)
        take_profit_spin.setValue(0.0)
        form.addRow("Take-Profit:", take_profit_spin)

        def _sync_widgets():
            ot = str(order_type_combo.currentData() or "limit")
            is_market_like = ot in {
                OrderType.MARKET.value,
                OrderType.IOC.value,
                OrderType.FOK.value,
                OrderType.TRAIL_MARKET.value,
            }
            needs_trigger = ot in {
                OrderType.STOP.value,
                OrderType.STOP_LIMIT.value,
                OrderType.TRAIL_MARKET.value,
                OrderType.TRAIL_LIMIT.value,
            }
            needs_trailing = ot in {
                OrderType.TRAIL_MARKET.value,
                OrderType.TRAIL_LIMIT.value,
            }
            needs_trail_limit_offset = ot == OrderType.TRAIL_LIMIT.value

            price_spin.setEnabled(not is_market_like or ot == OrderType.TRAIL_LIMIT.value)
            trigger_spin.setEnabled(needs_trigger)
            trailing_spin.setEnabled(needs_trailing)
            trail_limit_offset_spin.setEnabled(needs_trail_limit_offset)

            if ot in (OrderType.IOC.value, OrderType.FOK.value):
                forced = "ioc" if ot == OrderType.IOC.value else "fok"
                idx = tif_combo.findData(forced)
                if idx >= 0:
                    tif_combo.setCurrentIndex(idx)

        order_type_combo.currentIndexChanged.connect(_sync_widgets)
        _sync_widgets()

        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
        btns.accepted.connect(dialog.accept)
        btns.rejected.connect(dialog.reject)
        layout.addWidget(btns)

        if dialog.exec() != QDialog.DialogCode.Accepted:
            return None

        return {
            "qty": int(qty_spin.value()),
            "price": float(price_spin.value()),
            "order_type": str(order_type_combo.currentData() or "limit"),
            "time_in_force": str(tif_combo.currentData() or "day"),
            "trigger_price": float(trigger_spin.value()),
            # Percent units (e.g., 0.8 means 0.8%).
            "trailing_stop_pct": float(trailing_spin.value()),
            "trail_limit_offset_pct": float(trail_limit_offset_spin.value()),
            "strict_time_in_force": bool(strict_tif.isChecked()),
            "bracket": bool(bracket_check.isChecked()),
            "stop_loss": float(stop_loss_spin.value()),
            "take_profit": float(take_profit_spin.value()),
        }

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
        if self.executor is None:
            return
        try:
            lot = max(1, int(get_lot_size(symbol)))
        except Exception:
            lot = 1

        requested_qty = max(1, int(qty))
        normalized_qty = max(lot, requested_qty)
        if normalized_qty % lot != 0:
            normalized_qty = (normalized_qty // lot) * lot
            if normalized_qty <= 0:
                normalized_qty = lot
        if normalized_qty != requested_qty:
            self.log(
                f"Adjusted quantity {requested_qty} -> {normalized_qty} (lot {lot})",
                "info",
            )

        normalized_order_type = str(order_type or "limit").strip().lower().replace("-", "_")
        valid_order_types = {
            OrderType.LIMIT.value,
            OrderType.MARKET.value,
            OrderType.STOP.value,
            OrderType.STOP_LIMIT.value,
            OrderType.IOC.value,
            OrderType.FOK.value,
            OrderType.TRAIL_MARKET.value,
            OrderType.TRAIL_LIMIT.value,
        }
        if normalized_order_type not in valid_order_types:
            normalized_order_type = OrderType.LIMIT.value

        normalized_tif = str(time_in_force or "day").strip().lower()
        if normalized_tif not in {"day", "gtc", "ioc", "fok"}:
            normalized_tif = "day"
        if normalized_order_type in {OrderType.IOC.value, OrderType.FOK.value}:
            normalized_tif = normalized_order_type

        normalized_price = max(0.0, float(price))
        if normalized_order_type in {
            OrderType.MARKET.value,
            OrderType.IOC.value,
            OrderType.FOK.value,
            OrderType.TRAIL_MARKET.value,
        } and normalized_price <= 0:
            normalized_price = 0.01

        normalized_trigger = max(0.0, float(trigger_price))
        if normalized_order_type in {
            OrderType.STOP.value,
            OrderType.STOP_LIMIT.value,
            OrderType.TRAIL_MARKET.value,
            OrderType.TRAIL_LIMIT.value,
        } and normalized_trigger <= 0:
            normalized_trigger = normalized_price
        if normalized_order_type not in {
            OrderType.STOP.value,
            OrderType.STOP_LIMIT.value,
            OrderType.TRAIL_MARKET.value,
            OrderType.TRAIL_LIMIT.value,
        }:
            normalized_trigger = 0.0

        normalized_trailing_stop = max(0.0, float(trailing_stop_pct))
        # Backward compatibility: older UI path sent fractional units (0.008).
        if 0.0 < normalized_trailing_stop < 0.05:
            normalized_trailing_stop *= 100.0
        normalized_trailing_stop = min(20.0, normalized_trailing_stop)
        if normalized_order_type not in {
            OrderType.TRAIL_MARKET.value,
            OrderType.TRAIL_LIMIT.value,
        }:
            normalized_trailing_stop = 0.0

        normalized_trail_limit_offset = max(0.0, float(trail_limit_offset_pct))
        if 0.0 < normalized_trail_limit_offset < 0.05:
            normalized_trail_limit_offset *= 100.0
        normalized_trail_limit_offset = min(10.0, normalized_trail_limit_offset)
        if normalized_order_type != OrderType.TRAIL_LIMIT.value:
            normalized_trail_limit_offset = 0.0

        normalized_stop_loss = max(0.0, float(stop_loss))
        normalized_take_profit = max(0.0, float(take_profit))
        use_bracket = bool(bracket) and (
            normalized_stop_loss > 0 or normalized_take_profit > 0
        )

        signal = TradeSignal(
            symbol=symbol,
            side=side,
            quantity=normalized_qty,
            price=normalized_price,
            strategy="chart_manual",
            reasons=[
                "Manual chart quick-trade",
                f"order_type={normalized_order_type}",
                f"tif={normalized_tif}",
            ],
            confidence=1.0,
            order_type=normalized_order_type,
            time_in_force=normalized_tif,
            trigger_price=normalized_trigger,
            trailing_stop_pct=normalized_trailing_stop,
            trail_limit_offset_pct=normalized_trail_limit_offset,
            stop_loss=normalized_stop_loss if use_bracket else 0.0,
            take_profit=normalized_take_profit if use_bracket else 0.0,
            bracket=use_bracket,
        )
        signal.strict_time_in_force = bool(strict_time_in_force)
        try:
            ok = self.executor.submit(signal)
            if ok:
                price_text = (
                    f"{normalized_price:.2f}"
                    if normalized_price > 0
                    else "MKT"
                )
                self.log(
                    "Chart trade submitted: "
                    f"{side.value.upper()} {normalized_qty} {symbol} "
                    f"@ {price_text} ({normalized_order_type}, {normalized_tif})",
                    "success",
                )
            else:
                self.log("Chart trade rejected by risk/permissions", "warning")
        except Exception as e:
            self.log(f"Chart trade failed: {e}", "error")

    def _execute_buy(self):
        """Execute buy order"""
        if not self.current_prediction or not self.executor:
            return

        pred = self.current_prediction

        levels = getattr(pred, 'levels', None)
        position = getattr(pred, 'position', None)

        if not levels or not position:
            self.log("Missing trading levels or position info", "error")
            return

        shares = getattr(position, 'shares', 0)
        entry = getattr(levels, 'entry', 0)
        value = getattr(position, 'value', 0)
        stop_loss = getattr(levels, 'stop_loss', 0)
        target_2 = getattr(levels, 'target_2', 0)
        stock_name = getattr(pred, 'stock_name', '')

        try:
            if not CONFIG.is_market_open():
                QMessageBox.warning(
                    self, "Market Closed",
                    "Market is currently closed. Live orders are blocked."
                )
                return
        except Exception:
            pass

        try:
            ok, msg, fresh_px = self.executor.check_quote_freshness(pred.stock_code)
            if not ok:
                QMessageBox.warning(
                    self, "Stale Quote",
                    f"Order blocked: {msg}"
                )
                return
            if fresh_px > 0:
                entry = float(fresh_px)
        except Exception:
            pass

        reply = QMessageBox.question(
            self, "Confirm Buy Order",
            f"<b>Buy {pred.stock_code} - {stock_name}</b><br><br>"
            f"Quantity: {shares:,} shares<br>"
            f"Price: CNY {entry:.2f}<br>"
            f"Value: CNY {value:,.2f}<br>"
            f"Stop Loss: CNY {stop_loss:.2f}<br>"
            f"Target: CNY {target_2:.2f}",
            QMessageBox.StandardButton.Yes
            | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            try:
                if hasattr(self.executor, 'submit_from_prediction'):
                    success = self.executor.submit_from_prediction(pred)
                else:
                    success = False

                if success:
                    self.log(
                        f"Buy order submitted: {pred.stock_code}", "info"
                    )
                else:
                    self.log("Buy order failed risk checks", "error")
            except Exception as e:
                self.log(f"Buy order error: {e}", "error")

    def _execute_sell(self):
        """Execute sell order"""
        if not self.current_prediction or not self.executor:
            return

        pred = self.current_prediction

        try:
            if not CONFIG.is_market_open():
                QMessageBox.warning(
                    self, "Market Closed",
                    "Market is currently closed. Live orders are blocked."
                )
                return
        except Exception:
            pass

        try:
            ok, msg, fresh_px = self.executor.check_quote_freshness(pred.stock_code)
            if not ok:
                QMessageBox.warning(
                    self, "Stale Quote",
                    f"Order blocked: {msg}"
                )
                return
        except Exception:
            fresh_px = 0.0

        try:
            positions = self.executor.get_positions()
            position = positions.get(pred.stock_code)

            if not position:
                self.log("No position to sell", "warning")
                return

            available_qty = getattr(position, 'available_qty', 0)
            current_price = getattr(position, 'current_price', 0) or fresh_px
            stock_name = getattr(pred, 'stock_name', '')

            reply = QMessageBox.question(
                self, "Confirm Sell Order",
                f"<b>Sell {pred.stock_code} - {stock_name}</b><br><br>"
                f"Available: {available_qty:,} shares<br>"
                f"Current Price: CNY {current_price:.2f}",
                QMessageBox.StandardButton.Yes
                | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )

            if reply == QMessageBox.StandardButton.Yes:
                from core.types import OrderSide, TradeSignal

                signal = TradeSignal(
                    symbol=pred.stock_code,
                    name=stock_name,
                    side=OrderSide.SELL,
                    quantity=available_qty,
                    price=current_price
                )

                success = self.executor.submit(signal)
                if success:
                    self.log(
                        f"Sell order submitted: {pred.stock_code}", "info"
                    )
                else:
                    self.log("Sell order failed", "error")
        except Exception as e:
            self.log(f"Sell order error: {e}", "error")

    def _on_order_filled(self, order, fill):
        """Handle order fill"""
        side = (
            order.side.value.upper()
            if hasattr(order.side, 'value')
            else str(order.side)
        )
        qty = getattr(fill, 'quantity', 0)
        price = getattr(fill, 'price', 0)

        self.log(
            f"Order filled: {side} {qty} {order.symbol} @ CNY {price:.2f}",
            "success"
        )
        self._refresh_portfolio()

    def _on_order_rejected(self, order, reason):
        """Handle order rejection"""
        self.log(
            f"Order rejected: {order.symbol} - {reason}", "error"
        )

    def _refresh_portfolio(self):
        """Refresh portfolio display with visible error handling"""
        if not self.executor:
            return

        try:
            account = self.executor.get_account()

            equity = getattr(account, 'equity', 0)
            available = getattr(account, 'available', 0)
            market_value = getattr(account, 'market_value', 0)
            total_pnl = getattr(account, 'total_pnl', 0)
            positions = getattr(account, 'positions', {})

            self.account_labels['equity'].setText(f"CNY {equity:,.2f}")
            self.account_labels['cash'].setText(f"CNY {available:,.2f}")
            self.account_labels['positions'].setText(
                f"CNY {market_value:,.2f}"
            )

            pnl_color = "#35b57c" if total_pnl >= 0 else "#e5534b"
            self.account_labels['pnl'].setText(f"CNY {total_pnl:,.2f}")
            self.account_labels['pnl'].setStyleSheet(
                f"color: {pnl_color}; font-size: 18px; font-weight: bold;"
            )

            if hasattr(self.positions_table, 'update_positions'):
                self.positions_table.update_positions(positions)

        except Exception as e:
            # FIX: Make portfolio errors visible instead of silent
            log.warning(f"Portfolio refresh error: {e}")
            self.log(f"Portfolio refresh failed: {e}", "warning")

    # =========================================================================
    # =========================================================================

    def _start_training(self):
        """Start model training (UI dialog)."""
        interval = self.interval_combo.currentText().strip()
        horizon = self.forecast_spin.value()

        reply = QMessageBox.question(
            self, "Train AI Model",
            f"Start training with the following settings?\n\n"
            f"Interval: {interval}\n"
            f"Horizon: {horizon} bars\n\n"
            f"This may take time.\n\nContinue?",
            QMessageBox.StandardButton.Yes
            | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        try:
            from .dialogs import TrainingDialog
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
        except Exception as e:
            self.log(f"Training dialog failed: {e}", "error")
            return

        self._init_components()

    def _show_auto_learn(self):
        """Show auto-learning dialog"""
        try:
            from .auto_learn_dialog import show_auto_learn_dialog
            seed_codes: list[str] = []
            try:
                if self._session_bar_cache is not None:
                    interval = self._normalize_interval_token(
                        self.interval_combo.currentText()
                    )
                    seed_codes = self._session_bar_cache.get_recent_symbols(
                        interval=interval, min_rows=10
                    )
            except Exception:
                seed_codes = []
            show_auto_learn_dialog(self, seed_stock_codes=seed_codes)
        except ImportError:
            self.log("Auto-learn dialog not available", "error")
            return

        self._init_components()

    def _show_strategy_marketplace(self):
        """Show strategy marketplace manager."""
        try:
            from .strategy_marketplace_dialog import StrategyMarketplaceDialog
            dialog = StrategyMarketplaceDialog(self)
            dialog.exec()
        except Exception as e:
            self.log(f"Strategy marketplace unavailable: {e}", "error")

    def _show_backtest(self):
        """Show backtest dialog"""
        try:
            from .dialogs import BacktestDialog
            dialog = BacktestDialog(self)
            dialog.exec()
        except ImportError:
            self.log("Backtest dialog not available", "error")

    def _show_about(self):
        """Show about dialog"""
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
            "afford to lose.</p>"
        )

    # =========================================================================
    # AUTO-TRADE CONTROLS
    # =========================================================================

    def _on_trade_mode_changed(self, index: int):
        """Handle trade mode combo box change"""
        mode_map = {
            0: AutoTradeMode.MANUAL,
            1: AutoTradeMode.AUTO,
            2: AutoTradeMode.SEMI_AUTO,
        }
        new_mode = mode_map.get(index, AutoTradeMode.MANUAL)

        if new_mode == AutoTradeMode.AUTO:
            if self.predictor is None or (
                self.predictor and self.predictor.ensemble is None
            ):
                QMessageBox.warning(
                    self, "Cannot Enable Auto-Trade",
                    "No AI model loaded. Train a model first."
                )
                self.trade_mode_combo.setCurrentIndex(0)
                return

            if self.executor is None:
                QMessageBox.warning(
                    self, "Cannot Enable Auto-Trade",
                    "Not connected to broker. Connect first."
                )
                self.trade_mode_combo.setCurrentIndex(0)
                return

            if (
                self.executor
                and self.executor.mode == TradingMode.LIVE
                and CONFIG.auto_trade.confirm_live_auto_trade
            ):
                reply = QMessageBox.warning(
                    self, "LIVE Auto-Trading",
                    "You are enabling AUTOMATIC trading with REAL MONEY!\n\n"
                    "The AI will execute trades WITHOUT your confirmation.\n\n"
                    "Risk limits still apply, but trades happen automatically.\n\n"
                    "Are you absolutely sure?",
                    QMessageBox.StandardButton.Yes
                    | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.No
                )
                if reply != QMessageBox.StandardButton.Yes:
                    self.trade_mode_combo.setCurrentIndex(0)
                    return

            reply = QMessageBox.question(
                self, "Enable Auto-Trading",
                "Enable fully automatic trading?\n\n"
                f"- Min confidence: {CONFIG.auto_trade.min_confidence:.0%}\n"
                f"- Max trades/day: {CONFIG.auto_trade.max_trades_per_day}\n"
                f"- Max order value: CNY {CONFIG.auto_trade.max_auto_order_value:,.0f}\n"
                f"- Max auto positions: {CONFIG.auto_trade.max_auto_positions}\n\n"
                "You can pause or switch to Manual at any time.",
                QMessageBox.StandardButton.Yes
                | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )
            if reply != QMessageBox.StandardButton.Yes:
                self.trade_mode_combo.setCurrentIndex(0)
                return

        self._auto_trade_mode = new_mode
        self._apply_auto_trade_mode(new_mode)

    def _apply_auto_trade_mode(self, mode: AutoTradeMode):
        """Apply the auto-trade mode to the system."""
        CONFIG.auto_trade.enabled = (mode != AutoTradeMode.MANUAL)

        # Update executor auto-trader
        if self.executor and self.executor.auto_trader:
            self.executor.set_auto_mode(mode)

            # Update watchlist on auto-trader
            self.executor.auto_trader.update_watchlist(self.watch_list)

            if self.predictor:
                self.executor.auto_trader.update_predictor(self.predictor)
        elif mode != AutoTradeMode.MANUAL:
            # Need to initialize auto-trader first
            self._init_auto_trader()
            if self.executor and self.executor.auto_trader:
                self.executor.set_auto_mode(mode)

        self._update_auto_trade_status_label(mode)

        # Enable/disable manual trade buttons based on mode
        if mode == AutoTradeMode.AUTO:
            self.buy_btn.setEnabled(False)
            self.sell_btn.setEnabled(False)
            self.auto_pause_btn.setEnabled(True)
            self.log("AUTO mode enabled: AI executes trades automatically", "success")
        elif mode == AutoTradeMode.SEMI_AUTO:
            self.auto_pause_btn.setEnabled(True)
            self.auto_approve_all_btn.setEnabled(True)
            self.auto_reject_all_btn.setEnabled(True)
            self.log(
                "SEMI-AUTO mode enabled: AI suggests and you approve",
                "success"
            )
        else:
            self.auto_pause_btn.setEnabled(False)
            self.auto_approve_all_btn.setEnabled(False)
            self.auto_reject_all_btn.setEnabled(False)
            self.log("MANUAL mode enabled: you control all trades", "info")

    def _update_auto_trade_status_label(self, mode: AutoTradeMode):
        """Update the toolbar status label."""
        if mode == AutoTradeMode.AUTO:
            self.auto_trade_status_label.setText("  AUTO  ")
            self.auto_trade_status_label.setStyleSheet(
                "color: #4CAF50; font-weight: bold; padding: 0 8px;"
            )
        elif mode == AutoTradeMode.SEMI_AUTO:
            self.auto_trade_status_label.setText("  SEMI-AUTO  ")
            self.auto_trade_status_label.setStyleSheet(
                "color: #FFD54F; font-weight: bold; padding: 0 8px;"
            )
        else:
            self.auto_trade_status_label.setText("  MANUAL  ")
            self.auto_trade_status_label.setStyleSheet(
                "color: #aac3ec; font-weight: bold; padding: 0 8px;"
            )

    def _toggle_auto_pause(self):
        """Pause/resume auto-trading."""
        if not self.executor or not self.executor.auto_trader:
            return

        state = self.executor.auto_trader.get_state()
        if state.is_safety_paused or state.is_paused:
            self.executor.auto_trader.resume()
            self.auto_pause_btn.setText("Pause Auto")
            self.log("Auto-trading resumed", "info")
        else:
            self.executor.auto_trader.pause("Manually paused by user")
            self.auto_pause_btn.setText("Resume Auto")
            self.log("Auto-trading paused", "warning")

    def _approve_all_pending(self):
        """Approve all pending auto-trade actions."""
        if not self.executor or not self.executor.auto_trader:
            return

        pending = self.executor.auto_trader.get_pending_approvals()
        if not pending:
            self.log("No pending approvals", "info")
            return

        reply = QMessageBox.question(
            self, "Approve All",
            f"Approve all {len(pending)} pending trades?",
            QMessageBox.StandardButton.Yes
            | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        approved = 0
        for action in pending:
            if self.executor.auto_trader.approve_pending(action.id):
                approved += 1

        self.log(f"Approved {approved}/{len(pending)} pending trades", "success")

    def _reject_all_pending(self):
        """Reject all pending auto-trade actions."""
        if not self.executor or not self.executor.auto_trader:
            return

        pending = self.executor.auto_trader.get_pending_approvals()
        for action in pending:
            self.executor.auto_trader.reject_pending(action.id)

        if pending:
            self.log(f"Rejected {len(pending)} pending trades", "warning")

    def _show_auto_trade_settings(self):
        """Show auto-trade settings dialog."""
        from PyQt6.QtWidgets import QDialog, QDialogButtonBox, QFormLayout

        dialog = QDialog(self)
        dialog.setWindowTitle("Auto-Trade Settings")
        dialog.setMinimumWidth(500)

        layout = QVBoxLayout(dialog)

        group = QGroupBox("Auto-Trade Parameters")
        form = QFormLayout(group)

        cfg = CONFIG.auto_trade

        min_conf_spin = QDoubleSpinBox()
        min_conf_spin.setRange(0.50, 0.99)
        min_conf_spin.setValue(cfg.min_confidence)
        min_conf_spin.setSingleStep(0.05)
        min_conf_spin.setSuffix(" ")
        form.addRow("Min Confidence:", min_conf_spin)

        min_strength_spin = QDoubleSpinBox()
        min_strength_spin.setRange(0.30, 0.99)
        min_strength_spin.setValue(cfg.min_signal_strength)
        min_strength_spin.setSingleStep(0.05)
        form.addRow("Min Signal Strength:", min_strength_spin)

        min_agreement_spin = QDoubleSpinBox()
        min_agreement_spin.setRange(0.30, 0.99)
        min_agreement_spin.setValue(cfg.min_model_agreement)
        min_agreement_spin.setSingleStep(0.05)
        form.addRow("Min Model Agreement:", min_agreement_spin)

        max_positions_spin = QSpinBox()
        max_positions_spin.setRange(1, 20)
        max_positions_spin.setValue(cfg.max_auto_positions)
        form.addRow("Max Auto Positions:", max_positions_spin)

        max_order_spin = QDoubleSpinBox()
        max_order_spin.setRange(1000, 1000000)
        max_order_spin.setValue(cfg.max_auto_order_value)
        max_order_spin.setPrefix("CNY ")
        max_order_spin.setSingleStep(5000)
        form.addRow("Max Order Value:", max_order_spin)

        max_trades_spin = QSpinBox()
        max_trades_spin.setRange(1, 50)
        max_trades_spin.setValue(cfg.max_trades_per_day)
        form.addRow("Max Trades/Day:", max_trades_spin)

        max_per_stock_spin = QSpinBox()
        max_per_stock_spin.setRange(1, 10)
        max_per_stock_spin.setValue(cfg.max_trades_per_stock_per_day)
        form.addRow("Max Trades/Stock/Day:", max_per_stock_spin)

        cooldown_spin = QSpinBox()
        cooldown_spin.setRange(30, 3600)
        cooldown_spin.setValue(cfg.cooldown_after_trade_seconds)
        cooldown_spin.setSuffix(" sec")
        form.addRow("Cooldown After Trade:", cooldown_spin)

        scan_interval_spin = QSpinBox()
        scan_interval_spin.setRange(10, 600)
        scan_interval_spin.setValue(cfg.scan_interval_seconds)
        scan_interval_spin.setSuffix(" sec")
        form.addRow("Scan Interval:", scan_interval_spin)

        max_pos_pct_spin = QDoubleSpinBox()
        max_pos_pct_spin.setRange(1.0, 30.0)
        max_pos_pct_spin.setValue(cfg.max_auto_position_pct)
        max_pos_pct_spin.setSuffix(" %")
        form.addRow("Max Auto Position %:", max_pos_pct_spin)

        vol_pause_check = QCheckBox("Pause on high volatility")
        vol_pause_check.setChecked(cfg.pause_on_high_volatility)
        form.addRow("", vol_pause_check)

        auto_stop_check = QCheckBox("Auto stop-loss")
        auto_stop_check.setChecked(cfg.auto_stop_loss)
        form.addRow("", auto_stop_check)

        layout.addWidget(group)

        signals_group = QGroupBox("Allowed Signals")
        signals_layout = QGridLayout(signals_group)

        strong_buy_check = QCheckBox("STRONG_BUY")
        strong_buy_check.setChecked(cfg.allow_strong_buy)
        signals_layout.addWidget(strong_buy_check, 0, 0)

        buy_check = QCheckBox("BUY")
        buy_check.setChecked(cfg.allow_buy)
        signals_layout.addWidget(buy_check, 0, 1)

        sell_check = QCheckBox("SELL")
        sell_check.setChecked(cfg.allow_sell)
        signals_layout.addWidget(sell_check, 1, 0)

        strong_sell_check = QCheckBox("STRONG_SELL")
        strong_sell_check.setChecked(cfg.allow_strong_sell)
        signals_layout.addWidget(strong_sell_check, 1, 1)

        layout.addWidget(signals_group)

        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save
            | QDialogButtonBox.StandardButton.Cancel
        )

        def save_settings():
            cfg.min_confidence = min_conf_spin.value()
            cfg.min_signal_strength = min_strength_spin.value()
            cfg.min_model_agreement = min_agreement_spin.value()
            cfg.max_auto_positions = max_positions_spin.value()
            cfg.max_auto_order_value = max_order_spin.value()
            cfg.max_trades_per_day = max_trades_spin.value()
            cfg.max_trades_per_stock_per_day = max_per_stock_spin.value()
            cfg.cooldown_after_trade_seconds = cooldown_spin.value()
            cfg.scan_interval_seconds = scan_interval_spin.value()
            cfg.max_auto_position_pct = max_pos_pct_spin.value()
            cfg.pause_on_high_volatility = vol_pause_check.isChecked()
            cfg.auto_stop_loss = auto_stop_check.isChecked()
            cfg.allow_strong_buy = strong_buy_check.isChecked()
            cfg.allow_buy = buy_check.isChecked()
            cfg.allow_sell = sell_check.isChecked()
            cfg.allow_strong_sell = strong_sell_check.isChecked()

            try:
                CONFIG.save()
            except Exception:
                pass

            self.log("Auto-trade settings saved", "success")
            dialog.accept()

        btns.accepted.connect(save_settings)
        btns.rejected.connect(dialog.reject)
        layout.addWidget(btns)

        dialog.exec()

    def _on_auto_trade_action_safe(self, action: AutoTradeAction):
        """Thread-safe callback from auto-trader action."""
        QTimer.singleShot(0, lambda: self._on_auto_trade_action(action))

    def _on_auto_trade_action(self, action: AutoTradeAction):
        """Handle auto-trade action on UI thread."""
        row = 0
        self.auto_actions_table.insertRow(row)

        self.auto_actions_table.setItem(row, 0, QTableWidgetItem(
            action.timestamp.strftime("%H:%M:%S")
            if action.timestamp else "--"
        ))

        code_text = action.stock_code
        if action.stock_name:
            code_text += f" {action.stock_name}"
        self.auto_actions_table.setItem(row, 1, QTableWidgetItem(code_text))

        signal_item = QTableWidgetItem(action.signal_type)
        if action.signal_type in ("STRONG_BUY", "BUY"):
            signal_item.setForeground(QColor("#4CAF50"))
        elif action.signal_type in ("STRONG_SELL", "SELL"):
            signal_item.setForeground(QColor("#F44336"))
        self.auto_actions_table.setItem(row, 2, signal_item)

        self.auto_actions_table.setItem(
            row, 3, QTableWidgetItem(f"{action.confidence:.0%}")
        )

        decision_item = QTableWidgetItem(action.decision)
        if action.decision == "EXECUTED":
            decision_item.setForeground(QColor("#4CAF50"))
        elif action.decision == "SKIPPED":
            decision_item.setForeground(QColor("#FFD54F"))
        elif action.decision == "REJECTED":
            decision_item.setForeground(QColor("#F44336"))
        self.auto_actions_table.setItem(row, 4, decision_item)

        self.auto_actions_table.setItem(
            row, 5, QTableWidgetItem(
                f"{action.quantity:,}" if action.quantity else "--"
            )
        )

        self.auto_actions_table.setItem(
            row, 6, QTableWidgetItem(
                action.skip_reason if action.skip_reason else "--"
            )
        )

        while self.auto_actions_table.rowCount() > 100:
            self.auto_actions_table.removeRow(
                self.auto_actions_table.rowCount() - 1
            )

        if action.decision == "EXECUTED":
            self.log(
                f"AUTO-TRADE: {action.side.upper()} "
                f"{action.quantity} {action.stock_code} "
                f"@ CNY {action.price:.2f} ({action.confidence:.0%})",
                "success"
            )
        elif action.decision == "SKIPPED":
            self.log(
                f"Auto-trade skipped {action.stock_code}: {action.skip_reason}",
                "info"
            )

        if action.decision == "EXECUTED":
            QApplication.alert(self)

    def _on_pending_approval_safe(self, action: AutoTradeAction):
        """Thread-safe callback for pending approval."""
        QTimer.singleShot(0, lambda: self._on_pending_approval(action))

    def _on_pending_approval(self, action: AutoTradeAction):
        """Handle pending approval on UI thread."""
        row = self.pending_table.rowCount()
        self.pending_table.insertRow(row)

        self.pending_table.setItem(row, 0, QTableWidgetItem(
            action.timestamp.strftime("%H:%M:%S")
            if action.timestamp else "--"
        ))
        self.pending_table.setItem(
            row, 1, QTableWidgetItem(action.stock_code)
        )

        signal_item = QTableWidgetItem(action.signal_type)
        if action.signal_type in ("STRONG_BUY", "BUY"):
            signal_item.setForeground(QColor("#4CAF50"))
        else:
            signal_item.setForeground(QColor("#F44336"))
        self.pending_table.setItem(row, 2, signal_item)

        self.pending_table.setItem(
            row, 3, QTableWidgetItem(f"{action.confidence:.0%}")
        )
        self.pending_table.setItem(
            row, 4, QTableWidgetItem(f"CNY {action.price:.2f}")
        )

        # Approve/Reject buttons
        btn_widget = QWidget()
        btn_layout = QHBoxLayout(btn_widget)
        btn_layout.setContentsMargins(2, 2, 2, 2)

        approve_btn = QPushButton("Approve")
        approve_btn.setFixedWidth(84)
        approve_btn.setToolTip("Approve this trade")
        action_id = action.id

        def do_approve():
            if self.executor and self.executor.auto_trader:
                self.executor.auto_trader.approve_pending(action_id)
                self._refresh_pending_table()

        approve_btn.clicked.connect(do_approve)

        reject_btn = QPushButton("Reject")
        reject_btn.setFixedWidth(84)
        reject_btn.setToolTip("Reject this trade")

        def do_reject():
            if self.executor and self.executor.auto_trader:
                self.executor.auto_trader.reject_pending(action_id)
                self._refresh_pending_table()

        reject_btn.clicked.connect(do_reject)

        btn_layout.addWidget(approve_btn)
        btn_layout.addWidget(reject_btn)
        self.pending_table.setCellWidget(row, 5, btn_widget)

        self.log(
            f"PENDING: {action.signal_type} {action.stock_code} "
            f"@ CNY {action.price:.2f} - approve or reject",
            "warning"
        )
        QApplication.alert(self)

    def _refresh_pending_table(self):
        """Rebuild pending table from auto-trader state."""
        self.pending_table.setRowCount(0)

        if not self.executor or not self.executor.auto_trader:
            return

        pending = self.executor.auto_trader.get_pending_approvals()
        for action in pending:
            self._on_pending_approval(action)

    def _refresh_auto_trade_ui(self):
        """Periodic refresh of auto-trade status display."""
        self._update_correct_guess_profit_ui()

        if not self.executor or not self.executor.auto_trader:
            self.auto_trade_labels.get('mode', QLabel()).setText(
                self._auto_trade_mode.value.upper()
            )
            self.auto_trade_labels.get('trades', QLabel()).setText("0")
            self.auto_trade_labels.get('pnl', QLabel()).setText("--")
            self.auto_trade_labels.get('status', QLabel()).setText("--")
            self.auto_pause_btn.setText("Pause Auto")
            self.auto_pause_btn.setEnabled(False)
            self.auto_approve_all_btn.setText("Approve All")
            self.auto_approve_all_btn.setEnabled(False)
            self.auto_reject_all_btn.setEnabled(False)
            return

        state = self.executor.auto_trader.get_state()

        mode_label = self.auto_trade_labels.get('mode')
        if mode_label:
            mode_text = state.mode.value.upper()
            if state.is_safety_paused:
                mode_text += " (PAUSED)"
            mode_label.setText(mode_text)

            if state.mode == AutoTradeMode.AUTO:
                color = "#F44336" if state.is_safety_paused else "#4CAF50"
            elif state.mode == AutoTradeMode.SEMI_AUTO:
                color = "#FFD54F"
            else:
                color = "#aac3ec"
            mode_label.setStyleSheet(
                f"color: {color}; font-size: 16px; font-weight: bold;"
            )

        trades_label = self.auto_trade_labels.get('trades')
        if trades_label:
            trades_label.setText(
                f"{state.trades_today} "
                f"(B:{state.buys_today} S:{state.sells_today})"
            )

        # P&L
        pnl_label = self.auto_trade_labels.get('pnl')
        if pnl_label:
            pnl = state.auto_trade_pnl
            pnl_color = "#35b57c" if pnl >= 0 else "#e5534b"
            pnl_label.setText(f"CNY {pnl:+,.2f}")
            pnl_label.setStyleSheet(
                f"color: {pnl_color}; font-size: 16px; font-weight: bold;"
            )

        status_label = self.auto_trade_labels.get('status')
        if status_label:
            if state.is_safety_paused:
                status_label.setText(f"Paused: {state.pause_reason}")
                status_label.setStyleSheet(
                    "color: #F44336; font-size: 14px; font-weight: bold;"
                )
            elif state.is_running:
                last_scan = ""
                if state.last_scan_time:
                    elapsed = (
                        datetime.now() - state.last_scan_time
                    ).total_seconds()
                    last_scan = f" ({elapsed:.0f}s ago)"
                status_label.setText(f"Running{last_scan}")
                status_label.setStyleSheet(
                    "color: #4CAF50; font-size: 14px; font-weight: bold;"
                )
            else:
                status_label.setText("Idle")
                status_label.setStyleSheet(
                    "color: #aac3ec; font-size: 14px;"
                )

        if state.is_safety_paused or state.is_paused:
            self.auto_pause_btn.setText("Resume Auto")
        else:
            self.auto_pause_btn.setText("Pause Auto")
        self.auto_pause_btn.setEnabled(state.mode != AutoTradeMode.MANUAL)

        pending_count = len(state.pending_approvals)
        can_bulk_decide = (
            state.mode == AutoTradeMode.SEMI_AUTO and pending_count > 0
        )
        if can_bulk_decide:
            self.auto_approve_all_btn.setText(
                f"Approve All ({pending_count})"
            )
            self.auto_approve_all_btn.setEnabled(True)
            self.auto_reject_all_btn.setEnabled(True)
        else:
            self.auto_approve_all_btn.setText("Approve All")
            self.auto_approve_all_btn.setEnabled(False)
            self.auto_reject_all_btn.setEnabled(False)

    # =========================================================================
    # =========================================================================

    def _update_clock(self):
        """Update clock"""
        self.time_label.setText(
            datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )

    def _update_market_status(self):
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

    def log(self, message: str, level: str = "info"):
        """Log message to UI"""
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

        if hasattr(self.log_widget, 'log'):
            self.log_widget.log(message, level)
        elif hasattr(self.log_widget, 'append'):
            self.log_widget.append(formatted)

        log.info(message)

    # =========================================================================
    # CLOSE EVENT (FIX #3 - calls super)
    # =========================================================================

    def closeEvent(self, event):
        """Handle window close safely."""
        if self.monitor:
            try:
                self.monitor.stop()
                self.monitor.wait(3000)
            except Exception:
                pass
            self.monitor = None

        # Stop auto-trader
        if self.executor and self.executor.auto_trader:
            try:
                self.executor.auto_trader.stop()
            except Exception:
                pass

        all_workers = set(self._active_workers) | set(self.workers.values())
        for worker in list(all_workers):
            try:
                worker.cancel()
                worker.quit()
                if not worker.wait(3000):
                    worker.terminate()
                    worker.wait(1000)
            except Exception:
                pass
        self._active_workers.clear()
        self.workers.clear()

        if self.executor:
            try:
                self.executor.stop()
            except Exception:
                pass
            self.executor = None

        for timer_name in (
            "clock_timer", "market_timer",
            "portfolio_timer", "watchlist_timer",
            "auto_trade_timer", "chart_live_timer"
        ):
            timer = getattr(self, timer_name, None)
            try:
                if timer:
                    timer.stop()
            except Exception:
                pass

        try:
            self._save_state()
        except Exception:
            pass

        event.accept()
        super().closeEvent(event)

    # =========================================================================
    # =========================================================================

    def _save_state(self):
        """Save application state for next session"""
        try:
            import json
            safe_watch_list = _sanitize_watch_list(
                self.watch_list,
                max_size=self.MAX_WATCHLIST_SIZE,
            )
            self.watch_list = safe_watch_list
            state = {
                'watch_list': safe_watch_list,
                'interval': self.interval_combo.currentText(),
                'forecast': self.forecast_spin.value(),
                'lookback': self.lookback_spin.value(),
                'capital': self.capital_spin.value(),
                'last_stock': self.stock_input.text(),
                'auto_trade_mode': self._auto_trade_mode.value,
            }

            state_path = CONFIG.DATA_DIR / "app_state.json"
            state_path.parent.mkdir(parents=True, exist_ok=True)

            with open(state_path, 'w') as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            log.debug(f"Failed to save state: {e}")

    def _load_state(self):
        """Load application state from previous session"""
        try:
            import json
            state_path = CONFIG.DATA_DIR / "app_state.json"

            if state_path.exists():
                with open(state_path) as f:
                    state = json.load(f)

                if 'watch_list' in state:
                    loaded = state['watch_list']
                    self.watch_list = _sanitize_watch_list(
                        loaded,
                        max_size=self.MAX_WATCHLIST_SIZE,
                    )
                if 'forecast' in state:
                    self.forecast_spin.setValue(state['forecast'])
                # Startup should always begin on 1m with latest 7-day window.
                self.interval_combo.blockSignals(True)
                try:
                    self.interval_combo.setCurrentText(self.STARTUP_INTERVAL)
                finally:
                    self.interval_combo.blockSignals(False)
                self.lookback_spin.setValue(
                    self._recommended_lookback(self.STARTUP_INTERVAL)
                )
                if 'capital' in state:
                    self.capital_spin.setValue(state['capital'])
                if 'last_stock' in state:
                    self.stock_input.setText(state['last_stock'])
                if 'auto_trade_mode' in state:
                    try:
                        self._auto_trade_mode = AutoTradeMode(
                            state['auto_trade_mode']
                        )
                        # Don't auto-start AUTO mode on load for safety
                        # User must explicitly re-enable
                        if self._auto_trade_mode == AutoTradeMode.AUTO:
                            self._auto_trade_mode = AutoTradeMode.MANUAL
                    except (ValueError, KeyError):
                        self._auto_trade_mode = AutoTradeMode.MANUAL

                log.debug("Application state restored")
        except Exception as e:
            log.debug(f"Failed to load state: {e}")

def run_app():
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
        except Exception:
            pass
        _restore_sigint_handler(previous_sigint)

    sys.exit(exit_code)


def _install_sigint_handler(app: QApplication):
    """Route Ctrl+C to Qt quit for graceful terminal shutdown."""
    try:
        previous = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, lambda *_: app.quit())
        return previous
    except Exception as e:
        log.debug(f"SIGINT handler install failed: {e}")
        return None


def _restore_sigint_handler(previous_handler) -> None:
    """Restore previous SIGINT handler after app loop exits."""
    if previous_handler is None:
        return
    try:
        signal.signal(signal.SIGINT, previous_handler)
    except Exception:
        pass

if __name__ == "__main__":
    run_app()
