# ui/app.py
import os
import signal
import sys
import threading
import time
from concurrent.futures import Future
from datetime import datetime
from importlib import import_module
from typing import Any, cast

from PyQt6.QtCore import QDate, QSize, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QAction, QFont
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDateEdit,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
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

from config.runtime_env import env_flag
from config.settings import CONFIG, TradingMode
from core.types import (
    AutoTradeAction,
    AutoTradeMode,
    OrderSide,
)
# Use enhanced AI ops with improved NLU, persistence, and async processing
try:
    from ui import app_ai_ops_enhanced as _app_ai_ops
except ImportError:
    from ui import app_ai_ops as _app_ai_ops
from ui import app_analysis_ops as _app_analysis_ops
from ui import app_bar_ops as _app_bar_ops
from ui import app_feed_ops as _app_feed_ops
from ui import app_model_chart_ops as _app_model_chart_ops
from ui import app_training_ops as _app_training_ops
from ui import app_universe_ops as _app_universe_ops
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
from ui.background_tasks import (
    RealTimeMonitor,
    WorkerThread,
)
from ui.background_tasks import (
    sanitize_watch_list as _sanitize_watch_list,
)
from ui.modern_theme import (
    ModernColors,
    ModernFonts,
    get_monospace_font_family,
    get_primary_font_family,
)
from utils.logger import get_logger
from utils.method_binding import bind_methods
from utils.recoverable import COMMON_RECOVERABLE_EXCEPTIONS

log = get_logger(__name__)

_UI_RECOVERABLE_EXCEPTIONS = COMMON_RECOVERABLE_EXCEPTIONS

def _lazy_get(module: str, name: str) -> Any:
    return getattr(import_module(module), name)

class MainApp(MainAppCommonMixin, QMainWindow):
    """Professional AI Stock Analysis Application.

    Features:
    - Real-time signal monitoring with multiple intervals
    - Custom AI model with ensemble neural networks
    - Professional modern theme
    - AI-generated price forecast curves with uncertainty bands
    """
    MAX_WATCHLIST_SIZE = 50

    # Dynamic forecast horizon based on interval (configurable via env)
    # Format: interval=steps pairs, e.g., "1m=60,5m=40,15m=30,1h=24,1d=30"
    GUESS_FORECAST_BARS_CONFIG = {
        "1m": 60,   # 1-hour forecast for 1-minute bars
        "2m": 50,
        "3m": 45,
        "5m": 40,   # ~3.3 hours forecast
        "15m": 32,  # ~8 hours forecast
        "30m": 28,  # ~14 hours forecast
        "60m": 24,  # ~24 hours forecast
        "1h": 24,
        "1d": 30,   # ~30 trading days forecast
    }
    
    # FIX #7: Adaptive forecast horizon configuration
    # Volatility-based adjustment factors for forecast length
    ADAPTIVE_FORECAST_CONFIG = {
        "min_horizon_factor": 0.5,    # Minimum 50% of base horizon
        "max_horizon_factor": 1.5,    # Maximum 150% of base horizon
        "volatility_lookback_bars": 20,  # Bars to use for volatility calculation
        "confidence_adjustment": 0.2,   # Horizon adjustment per confidence point
    }

    STARTUP_INTERVAL = "1m"

    bar_received = pyqtSignal(str, dict)
    quote_received = pyqtSignal(str, float)

    def __init__(self) -> None:
        super().__init__()

        self.setWindowTitle("Trading Graph Analysis")
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
        self._universe_catalog: list[dict[str, Any]] = []
        self._universe_last_refresh_ts: float = 0.0

        # Real-time state with thread safety
        self._last_forecast_refresh_ts: float = 0.0
        self._forecast_refresh_symbol: str = ""
        self._live_price_series: dict[str, list[float]] = {}
        self._price_series_lock = threading.Lock()
        self._session_cache_write_lock = threading.Lock()
        self._last_session_cache_write_ts: dict[str, float] = {}
        self._session_cache_io_pool = None
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

        # FIX: Bounded cache dicts with max size to prevent memory leaks
        self._MAX_CACHED_BARS = 500  # Max symbols with cached bars
        self._MAX_CACHED_QUOTES = 1000  # Max symbols with cached quotes
        self._bars_by_symbol: dict[str, list[dict[str, Any]]] = {}
        self._last_bar_feed_ts: dict[str, float] = {}
        self._chart_symbol: str = ""
        self._history_refresh_once: set[tuple[str, str]] = set()
        self._strict_startup = env_flag("TRADING_STRICT_STARTUP", "0")
        self._debug_console_enabled = env_flag("TRADING_DEBUG_CONSOLE", "0")
        self._debug_console_last_emit: dict[str, float] = {}
        self._syncing_mode_ui = False
        self._selected_chart_date = datetime.now().date().isoformat()
        self._session_bar_cache = None
        self._startup_loading_active = False
        self._ai_chat_history: list[dict[str, str]] = []
        self._news_policy_signal_cache: dict[str, dict[str, Any]] = {}

        # FIX #4: Prediction cache for graceful degradation when models unavailable
        self._prediction_cache: dict[str, dict[str, Any]] = {}
        self._prediction_cache_ttl: int = 300  # 5 minutes cache TTL
        self._prediction_cache_max_size: int = 50  # Max cached predictions

        # Auto-trade state
        self._auto_trade_mode: AutoTradeMode = AutoTradeMode.MANUAL

        self._setup_menubar()
        self._setup_toolbar()
        self._setup_ui()
        self._setup_statusbar()
        self._set_startup_loading(
            "Starting system...",
            value=8,
        )
        self._setup_timers()
        self._apply_professional_style()
        from PyQt6.QtCore import Qt as _Qt

        # PyQt signal stubs are narrower than runtime support for the
        # connection-type overload; cast to keep queued cross-thread delivery.
        cast(Any, self.bar_received).connect(
            self._on_bar_ui,
            _Qt.ConnectionType.QueuedConnection,
        )
        cast(Any, self.quote_received).connect(
            self._on_price_updated,
            _Qt.ConnectionType.QueuedConnection,
        )

        try:
            self._set_startup_loading(
                "Restoring workspace...",
                value=18,
            )
            self._load_state()
            self._update_watchlist()
        except _UI_RECOVERABLE_EXCEPTIONS:
            log.exception("Startup state restore failed")
            if self._strict_startup:
                raise

        self._set_startup_loading(
            "Initializing components...",
            value=32,
        )
        QTimer.singleShot(0, self._init_components)














    def _setup_menubar(self) -> None:
        """Setup professional menu bar."""
        menubar = self.menuBar()

        file_menu = menubar.addMenu("&File")

        new_action = QAction("&New Workspace", self)
        new_action.setShortcut("Ctrl+N")
        new_action.triggered.connect(self._new_workspace)
        file_menu.addAction(new_action)

        file_menu.addSeparator()

        exit_action = QAction("E&xit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        ai_menu = menubar.addMenu("&AI Model")

        train_action = QAction("&Train GM", self)
        train_action.setShortcut("Ctrl+T")
        train_action.triggered.connect(self._start_training)
        ai_menu.addAction(train_action)

        auto_learn_action = QAction("&Auto Train GM", self)
        auto_learn_action.triggered.connect(self._show_auto_learn)
        ai_menu.addAction(auto_learn_action)

        train_llm_action = QAction("Train &LLM", self)
        train_llm_action.triggered.connect(self._auto_train_llm)
        ai_menu.addAction(train_llm_action)

        auto_train_llm_action = QAction("&Auto Train LLM", self)
        auto_train_llm_action.triggered.connect(self._auto_train_llm)
        ai_menu.addAction(auto_train_llm_action)

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
        """Setup professional toolbar."""
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

        toolbar.addWidget(QLabel("  Profile: "))
        self.screener_profile_combo = QComboBox()
        self.screener_profile_combo.setObjectName("scanProfileCombo")
        self.screener_profile_combo.setFixedWidth(130)
        self.screener_profile_combo.currentTextChanged.connect(
            self._on_screener_profile_changed
        )
        toolbar.addWidget(self.screener_profile_combo)

        self.screener_profiles_action = QAction("Profiles...", self)
        self.screener_profiles_action.triggered.connect(
            self._show_screener_profile_dialog
        )
        toolbar.addAction(self.screener_profiles_action)

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

        toolbar.addWidget(QLabel("  Date: "))
        self.chart_date_edit = QDateEdit()
        self.chart_date_edit.setCalendarPopup(True)
        self.chart_date_edit.setDisplayFormat("yyyy-MM-dd")
        today = QDate.currentDate()
        self.chart_date_edit.setDate(today)
        self.chart_date_edit.setMaximumDate(today)
        self.chart_date_edit.setFixedWidth(126)
        self._selected_chart_date = str(today.toString("yyyy-MM-dd"))
        self.chart_date_edit.dateChanged.connect(self._on_chart_date_changed)
        try:
            cal = self.chart_date_edit.calendarWidget()
            if cal is not None:
                cal.setMinimumSize(320, 240)
        except _UI_RECOVERABLE_EXCEPTIONS as exc:
            log.debug("Suppressed exception in ui/app.py", exc_info=exc)
        toolbar.addWidget(self.chart_date_edit)

        self._init_screener_profile_ui()

    # =========================================================================
    # =========================================================================

    def _setup_ui(self) -> None:
        """Setup main UI with professional layout."""
        central = QWidget()
        central.setObjectName("AppRoot")
        self.setCentralWidget(central)

        layout = QHBoxLayout(central)
        layout.setSpacing(14)
        layout.setContentsMargins(14, 14, 14, 14)

        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        main_splitter.setChildrenCollapsible(False)
        main_splitter.setHandleWidth(8)

        # Left Panel - Control & Watchlist
        left_panel = self._create_left_panel()
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setFrameShape(QFrame.Shape.NoFrame)
        left_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        left_scroll.setWidget(left_panel)
        left_scroll.setMinimumWidth(250)
        left_scroll.setMaximumWidth(360)

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
        # Favor center workspace while keeping right-side logs/details readable.
        main_splitter.setSizes([280, 920, 340])

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

    def _make_item(self, text: str) -> QTableWidgetItem:
        """Create a non-editable table item with consistent alignment."""
        item = QTableWidgetItem(str(text or ""))
        item.setFlags(
            item.flags() & ~Qt.ItemFlag.ItemIsEditable
        )
        item.setTextAlignment(
            Qt.AlignmentFlag.AlignCenter
            | Qt.AlignmentFlag.AlignVCenter
        )
        return item

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
        frame.setObjectName("statFrame")
        grid = QGridLayout(frame)
        grid.setContentsMargins(
            int(padding),
            int(padding),
            int(padding),
            int(padding),
        )
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(8)
        out = {}
        for key, text, row, col in labels:
            container = QWidget()
            cont_layout = QVBoxLayout(container)
            cont_layout.setContentsMargins(4, 4, 4, 4)
            title = QLabel(text)
            title.setObjectName("metaLabel")
            value = QLabel("--")
            value.setStyleSheet(value_style)
            cont_layout.addWidget(title)
            cont_layout.addWidget(value)
            grid.addWidget(container, row, col)
            out[key] = value
        return frame, out

    def _create_center_panel(self) -> QWidget:
        """Create center panel with charts and signals."""
        panel = QWidget()
        panel.setObjectName("centerPanel")
        layout = QVBoxLayout(panel)
        layout.setSpacing(10)
        layout.setContentsMargins(0, 0, 0, 0)

        # Signal Display - lazy import
        try:
            from .widgets import SignalPanel
            self.signal_panel = SignalPanel()
        except ImportError:
            self.signal_panel = QLabel("Signal Panel")
            self.signal_panel.setMinimumHeight(68)
        # Keep signal card compact so chart/details get more vertical room.
        # FIX #23: Add hasattr guards for QLabel fallback
        if hasattr(self.signal_panel, 'setMinimumHeight'):
            self.signal_panel.setMinimumHeight(160)
        if hasattr(self.signal_panel, 'setMaximumHeight'):
            self.signal_panel.setMaximumHeight(210)
        if hasattr(self.signal_panel, 'setSizePolicy'):
            self.signal_panel.setSizePolicy(
                QSizePolicy.Policy.Preferred,
                QSizePolicy.Policy.Fixed,
            )
        layout.addWidget(self.signal_panel, 0)

        chart_group = QGroupBox("Price Chart and AI Prediction")
        chart_group.setObjectName("chartPrimaryGroup")
        chart_layout = QVBoxLayout()
        chart_layout.setContentsMargins(8, 8, 8, 8)
        chart_layout.setSpacing(8)

        try:
            from .charts import StockChart
            self.chart = StockChart()
            self.chart.setMinimumHeight(360)
        except ImportError:
            self.chart = QLabel("Chart (charts module not found)")
            self.chart.setMinimumHeight(360)
            self.chart.setAlignment(Qt.AlignmentFlag.AlignCenter)
        chart_layout.addWidget(self.chart)

        chart_action_frame = QFrame()
        chart_action_frame.setObjectName("chartActionStrip")
        chart_actions = QHBoxLayout(chart_action_frame)
        chart_actions.setContentsMargins(8, 6, 8, 6)
        chart_actions.setSpacing(8)

        self.zoom_in_btn = QPushButton("Zoom In")
        self.zoom_out_btn = QPushButton("Zoom Out")
        self.zoom_reset_btn = QPushButton("Reset View")
        self.zoom_in_btn.setObjectName("chartToolButton")
        self.zoom_out_btn.setObjectName("chartToolButton")
        self.zoom_reset_btn.setObjectName("chartToolButton")
        self.zoom_in_btn.setMaximumWidth(100)
        self.zoom_out_btn.setMaximumWidth(100)
        self.zoom_reset_btn.setMaximumWidth(110)
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
            chk.setObjectName("overlayToggle")
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
        chart_layout.addWidget(chart_action_frame)

        self.chart_latest_label = QLabel("Latest --")
        self.chart_latest_label.setObjectName("chartLatestLabel")
        chart_layout.addWidget(self.chart_latest_label)

        chart_group.setLayout(chart_layout)
        layout.addWidget(chart_group, 5)

        # Analysis details panel intentionally hidden from layout; details HTML
        # is still maintained for internal/debug usage.
        self.details_text = QTextEdit()
        self.details_text.setReadOnly(True)
        self.details_text.setFont(
            QFont(get_monospace_font_family(), ModernFonts.SIZE_SM)
        )
        self.details_text.setMinimumHeight(240)
        self.details_text.setMaximumHeight(16777215)
        self.details_text.hide()

        return panel

    def _zoom_chart_in(self) -> None:
        if hasattr(self.chart, "zoom_in"):
            try:
                self.chart.zoom_in()
            except _UI_RECOVERABLE_EXCEPTIONS as exc:
                log.debug("Suppressed exception in ui/app.py", exc_info=exc)

    def _zoom_chart_out(self) -> None:
        if hasattr(self.chart, "zoom_out"):
            try:
                self.chart.zoom_out()
            except _UI_RECOVERABLE_EXCEPTIONS as exc:
                log.debug("Suppressed exception in ui/app.py", exc_info=exc)

    def _zoom_chart_reset(self) -> None:
        if hasattr(self.chart, "reset_view"):
            try:
                self.chart.reset_view()
            except _UI_RECOVERABLE_EXCEPTIONS as exc:
                log.debug("Suppressed exception in ui/app.py", exc_info=exc)

    def _set_chart_overlay(self, key: str, enabled: bool) -> None:
        if hasattr(self.chart, "set_overlay_enabled"):
            try:
                self.chart.set_overlay_enabled(str(key), bool(enabled))
            except _UI_RECOVERABLE_EXCEPTIONS as exc:
                log.debug("Suppressed exception in ui/app.py", exc_info=exc)

    def _create_right_panel(self) -> QWidget:
        return _create_right_panel_impl(self)

        # =========================================================================
        # STATUS BAR & TIMERS
    # =========================================================================

    def _setup_statusbar(self) -> None:
        """Setup status bar."""
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
        self.market_label.setObjectName("subtleLabel")
        self._status_bar.addPermanentWidget(self.market_label)

        self.monitor_label = QLabel("Monitoring: OFF")
        self.monitor_label.setObjectName("monitorLabel")
        self._status_bar.addWidget(self.monitor_label)

        self.time_label = QLabel("")
        self._status_bar.addWidget(self.time_label)

    def _set_startup_loading(
        self,
        message: str,
        *,
        value: int | None = None,
        indeterminate: bool = False,
    ) -> None:
        """Show startup progress in status bar."""
        self._startup_loading_active = True
        self.status_label.setText(str(message or "Loading..."))
        if indeterminate:
            self.progress.setRange(0, 0)
        else:
            self.progress.setRange(0, 100)
            if value is not None:
                safe_val = max(0, min(100, int(value)))
                self.progress.setValue(safe_val)
        self.progress.show()

    def _complete_startup_loading(self, message: str = "Ready") -> None:
        """Clear startup progress once initial boot tasks complete."""
        if not bool(getattr(self, "_startup_loading_active", False)):
            return
        self._startup_loading_active = False
        for worker_name in ("analyze", "scan"):
            worker = self.workers.get(worker_name)
            if worker and worker.isRunning():
                return
        self.status_label.setText(str(message or "Ready"))
        self.progress.hide()

    def _setup_timers(self) -> None:
        """Setup update timers."""
        self.clock_timer = QTimer()
        self.clock_timer.timeout.connect(self._update_clock)
        self.clock_timer.start(1000)

        self.market_timer = QTimer()
        self.market_timer.timeout.connect(self._update_market_status)
        self.market_timer.start(60000)

        self.sentiment_timer = QTimer()
        self.sentiment_timer.timeout.connect(self._refresh_sentiment)
        self.sentiment_timer.start(30000)  # Refresh sentiment every 30 seconds

        self.watchlist_timer = QTimer()
        self.watchlist_timer.timeout.connect(self._update_watchlist)
        self.watchlist_timer.start(30000)

        self.universe_refresh_timer = QTimer()
        self.universe_refresh_timer.timeout.connect(self._refresh_universe_catalog)
        self.universe_refresh_timer.start(15000)

        # Live chart refresh: keep real + guessed lines moving.
        self.chart_live_timer = QTimer()
        self.chart_live_timer.timeout.connect(self._refresh_live_chart_forecast)
        self.chart_live_timer.start(1500)

        # FIX: Cache pruning to prevent memory leaks
        self.cache_prune_timer = QTimer()
        self.cache_prune_timer.timeout.connect(self._prune_caches)
        self.cache_prune_timer.timeout.connect(self._prune_session_futures)
        self.cache_prune_timer.start(60000)  # Prune every 60 seconds

        QTimer.singleShot(120, self._refresh_universe_catalog)
        self._update_market_status()

        # =========================================================================
    # =========================================================================

    def _apply_professional_style(self) -> None:
        _apply_professional_style_impl(self)

    # =========================================================================
    # =========================================================================

    def _init_components(self) -> None:
        """Initialize analysis components."""
        self._set_startup_loading(
            "Loading model runtime...",
            value=52,
        )
        try:
            Predictor = _lazy_get("models.predictor", "Predictor")

            interval = str(self.STARTUP_INTERVAL).strip().lower()
            self.interval_combo.blockSignals(True)
            try:
                self.interval_combo.setCurrentText(interval)
            finally:
                self.interval_combo.blockSignals(False)
            # Dynamic forecast horizon based on interval for better predictions
            forecast_bars = self._get_forecast_horizon_for_interval(interval)
            self.forecast_spin.setValue(int(forecast_bars))
            self.lookback_spin.setValue(self._recommended_lookback(interval))
            horizon = int(self.forecast_spin.value())

            self.predictor = Predictor(
                capital=self.capital_spin.value(),
                interval=interval,
                prediction_horizon=horizon
            )

            summary = self._predictor_model_summary()
            if bool(summary.get("runtime_ready", False)):
                num_models = int(summary.get("ensemble_models", 0) or 0)
                has_forecaster = bool(summary.get("has_forecaster", False))
                has_ensemble = bool(summary.get("has_ensemble", False))
                capability_bits: list[str] = []
                if has_ensemble:
                    capability_bits.append(f"ensemble x{max(1, num_models)}")
                if has_forecaster:
                    capability_bits.append("forecaster")
                capability = ", ".join(capability_bits) if capability_bits else "runtime models"
                self.model_status.setText(
                    f"GM Model: Loaded ({capability})"
                )
                self.model_status.setStyleSheet(
                    
                        f"color: {ModernColors.ACCENT_SUCCESS}; "
                        f"font-weight: {ModernFonts.WEIGHT_BOLD};"
                    
                )
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
                self.log("GM model loaded successfully", "success")
            elif bool(summary.get("forecast_ready", False)):
                self.model_status.setText("GM Model: Forecast-only")
                self.model_status.setStyleSheet(

                        f"color: {ModernColors.ACCENT_WARNING}; "
                        f"font-weight: {ModernFonts.WEIGHT_BOLD};"

                )
                self.model_info.setText(
                    "Guessed graph enabled (fallback mode). Train GM for stronger signals."
                )
                self.log(
                    "Fallback forecast runtime loaded (no ensemble/forecaster checkpoint).",
                    "warning",
                )
            else:
                self.model_status.setText("GM Model: Not trained")
                self.model_status.setStyleSheet(
                    
                        f"color: {ModernColors.ACCENT_WARNING}; "
                        f"font-weight: {ModernFonts.WEIGHT_BOLD};"
                    
                )
                self.model_info.setText(
                    "Train GM to enable predictions"
                )
                self.log(
                    "No trained GM model found. Please train GM.", "warning"
                )

        except _UI_RECOVERABLE_EXCEPTIONS as e:
            log.error(f"Failed to load model: {e}")
            self.log(f"Failed to load model: {e}", "error")
            self.predictor = None
            self.model_status.setText("GM Model: Error")
            self.model_status.setStyleSheet(
                
                    f"color: {ModernColors.ACCENT_DANGER}; "
                    f"font-weight: {ModernFonts.WEIGHT_BOLD};"
                
            )
            lower_msg = str(e).lower()
            if "c10.dll" in lower_msg or "dll initialization routine failed" in lower_msg:
                self.model_info.setText(
                    "PyTorch DLL load failed. Install VC++ Redistributable and use a matching torch build."
                )

        self._set_startup_loading(
            "Preparing live services...",
            value=72,
        )

        # Auto-start live monitor when model is available.
        if self._predictor_runtime_ready():
            try:
                self.monitor_action.setChecked(True)
                self._start_monitoring()
            except _UI_RECOVERABLE_EXCEPTIONS as e:
                log.debug(f"Auto-start monitoring failed: {e}")

        if hasattr(self, "_refresh_model_training_statuses"):
            try:
                self._refresh_model_training_statuses()
            except _UI_RECOVERABLE_EXCEPTIONS as exc:
                log.debug("Suppressed exception in ui/app.py", exc_info=exc)

        self._set_startup_loading(
            "Loading market universe...",
            indeterminate=True,
        )
        try:
            self._refresh_universe_catalog(force=False)
        except _UI_RECOVERABLE_EXCEPTIONS:
            self._complete_startup_loading("Ready")
        else:
            universe_worker = self.workers.get("universe_catalog")
            if not (universe_worker and universe_worker.isRunning()):
                self._complete_startup_loading("Ready")

        self.log("System initialized - Ready for analysis", "info")

    def _prune_caches(self) -> None:
        """Prune internal caches to prevent memory leaks.

        FIX #7: Use lock to prevent race condition with background worker threads.
        Called periodically to bound cache sizes.
        """
        # FIX #7: Acquire lock to prevent race condition with background workers
        # that may be modifying these dicts concurrently
        with self._session_cache_io_lock:
            # Prune _last_bar_feed_ts - keep only recent entries (last 10 minutes)
            now = time.time()
            max_age_seconds = 600  # 10 minutes
            stale_feed_ts = [
                key for key, ts in self._last_bar_feed_ts.items()
                if (now - ts) > max_age_seconds
            ]
            for key in stale_feed_ts:
                self._last_bar_feed_ts.pop(key, None)

            # Also prune by count if still too large
            if len(self._last_bar_feed_ts) > self._MAX_CACHED_QUOTES:
                sorted_items = sorted(
                    self._last_bar_feed_ts.items(),
                    key=lambda x: x[1]
                )
                cutoff = len(sorted_items) // 4
                for key, _ in sorted_items[:cutoff]:
                    self._last_bar_feed_ts.pop(key, None)

            # Prune _bars_by_symbol - keep only watchlist and active chart symbol
            active_syms = set(self._watchlist_row_by_code.keys())
            # FIX #13: Add hasattr guard for stock_input
            if hasattr(self, 'stock_input') and self.stock_input is not None:
                try:
                    selected = self._ui_norm(self.stock_input.text())
                    if selected:
                        active_syms.add(selected)
                except _UI_RECOVERABLE_EXCEPTIONS:
                    pass

            # Keep max 10 inactive symbols as buffer
            inactive_syms = [
                k for k in self._bars_by_symbol.keys()
                if k not in active_syms
            ]
            if len(inactive_syms) > 10:
                for key in inactive_syms[10:]:
                    self._bars_by_symbol.pop(key, None)

            # Prune stale quote UI emit tracking
            stale_quotes = [
                k for k in self._last_quote_ui_emit.keys()
                if k not in active_syms
            ]
            for k in stale_quotes:
                self._last_quote_ui_emit.pop(k, None)

            # Prune stale watchlist price UI tracking
            stale_watchlist = [
                k for k in self._last_watchlist_price_ui.keys()
                if k not in active_syms
            ]
            for k in stale_watchlist:
                self._last_watchlist_price_ui.pop(k, None)

    def _prune_session_futures(self) -> None:
        """FIX #14: Prune completed futures from _session_cache_io_futures set.
        
        This prevents memory leak for long-running sessions by removing
        completed futures from the tracking set.
        """
        with self._session_cache_io_lock:
            completed = {f for f in self._session_cache_io_futures if f.done()}
            self._session_cache_io_futures -= completed

    def _new_workspace(self) -> None:
        """Create a new workspace (reset current state)."""
        from PyQt6.QtWidgets import QMessageBox
        
        reply = QMessageBox.question(
            self,
            "New Workspace",
            "This will reset your current workspace. Are you sure?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self._reset_workspace()

    def _reset_workspace(self) -> None:
        """Reset the workspace to default state."""
        # Clear watchlist
        self.watch_list = []
        self._update_watchlist()
        
        # Clear charts and data
        self._bars_by_symbol.clear()
        self._last_bar_feed_ts.clear()
        
        # Reset to default stock
        self.stock_input.setText("")
        
        # Clear prediction cache if predictor exists
        if self.predictor:
            with self.predictor._cache_lock:
                self.predictor._pred_cache.clear()
        
        self.log("Workspace reset", "info")

    def _init_auto_trader(self) -> None:
        return

    def _on_interval_changed(self, interval: str) -> None:
        """Handle interval change - reload model and restart monitor."""
        interval = self._normalize_interval_token(interval)
        horizon = self.forecast_spin.value()
        self.model_info.setText(f"Interval: {interval}, Horizon: {horizon}")

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
        except _UI_RECOVERABLE_EXCEPTIONS as exc:
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
                    if self._predictor_runtime_ready():
                        active_iv, active_h = self._sync_ui_to_loaded_model(
                            interval,
                            horizon,
                            preserve_requested_interval=True,
                        )
                        self.log(
                            f"Model reloaded for {active_iv} interval, horizon {active_h}",
                            "info",
                        )
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
                    self._log_model_alignment_debug(
                        context="interval_keep_loaded",
                        requested_interval=interval,
                        requested_horizon=horizon,
                    )
            except _UI_RECOVERABLE_EXCEPTIONS as e:
                self.log(f"Model reload failed: {e}", "warning")

        if was_monitoring:
            self._start_monitoring()

        selected = self._ui_norm(self.stock_input.text())
        if (
            selected
            and self._predictor_runtime_ready()
        ):
            self.stock_input.setText(selected)
            self._analyze_stock()

    def _on_chart_date_changed(self, qdate: QDate) -> None:
        """Refresh chart data for selected trading date."""
        selected_date = str(qdate.toString("yyyy-MM-dd"))
        self._selected_chart_date = selected_date
        symbol = self._ui_norm(self.stock_input.text())
        if not symbol:
            return
        interval = self._normalize_interval_token(self.interval_combo.currentText())
        self._queue_history_refresh(symbol, interval)
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






    # =========================================================================
    # =========================================================================


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












    def _get_forecast_horizon_for_interval(self, interval: str) -> int:
        """Get dynamic forecast horizon based on interval.

        Args:
            interval: Time interval string (e.g., '1m', '5m', '1h', '1d')

        Returns:
            Number of forecast bars appropriate for the interval
            
        FIX #7: Adaptive forecast horizon based on market volatility
        and prediction confidence for optimal forecast length.
        """
        interval = str(interval).strip().lower()

        # Normalize interval aliases
        interval_aliases = {
            "1h": "60m",
            "60min": "60m",
            "60mins": "60m",
            "daily": "1d",
            "1day": "1d",
            "day": "1d",
        }
        interval = interval_aliases.get(interval, interval)

        # Get base horizon from config
        base_horizon = int(self.GUESS_FORECAST_BARS_CONFIG.get(interval, 30))

        # FIX #7: Apply adaptive horizon adjustment if enabled
        if getattr(CONFIG.model, "adaptive_horizon_enabled", True):
            # Calculate volatility-based adjustment
            current_symbol = self._ui_norm(self.stock_input.text())
            bars = self._bars_by_symbol.get(current_symbol, [])

            # Get prediction confidence if available
            confidence = 0.5  # Default neutral confidence
            try:
                if self.current_prediction and self.current_prediction.stock_code == current_symbol:
                    confidence = float(getattr(self.current_prediction, "confidence", 0.5))
            except Exception:
                pass

            if bars and len(bars) >= 20:
                closes = [b.get("close", 0.0) for b in bars[-30:] if b.get("close", 0) > 0]
                if len(closes) >= 10:
                    # Calculate recent volatility
                    returns = [(closes[i] - closes[i-1]) / max(closes[i-1], 1e-8)
                               for i in range(1, len(closes))]
                    volatility = float(np.std(returns)) if returns else 0.02

                    # High volatility = shorter horizon (less predictable)
                    # Low volatility = longer horizon (more predictable)
                    vol_scale = getattr(CONFIG.model, "horizon_volatility_scale", 0.5)
                    base_vol = 0.02  # Baseline volatility

                    vol_factor = 1.0 - vol_scale * (volatility - base_vol) / base_vol
                    vol_factor = float(np.clip(vol_factor, 0.5, 1.5))

                    # FIX #7: Add confidence-based adjustment
                    # Higher confidence = can extend horizon
                    # Lower confidence = shorten horizon for reliability
                    conf_adjustment = getattr(self, "ADAPTIVE_FORECAST_CONFIG", {}).get(
                        "confidence_adjustment", 0.2
                    )
                    conf_factor = 1.0 + conf_adjustment * (confidence - 0.5) * 2
                    conf_factor = float(np.clip(conf_factor, 0.7, 1.3))

                    # Combine volatility and confidence factors
                    combined_factor = vol_factor * conf_factor
                    
                    # Apply min/max bounds from config
                    min_factor = getattr(self, "ADAPTIVE_FORECAST_CONFIG", {}).get(
                        "min_horizon_factor", 0.5
                    )
                    max_factor = getattr(self, "ADAPTIVE_FORECAST_CONFIG", {}).get(
                        "max_horizon_factor", 1.5
                    )
                    combined_factor = float(np.clip(combined_factor, min_factor, max_factor))

                    # Adjust horizon
                    min_horizon = getattr(CONFIG.model, "min_prediction_horizon", 3)
                    max_horizon = getattr(CONFIG.model, "max_prediction_horizon", 60)

                    adjusted_horizon = int(base_horizon * combined_factor)
                    adjusted_horizon = max(min_horizon, min(max_horizon, adjusted_horizon))

                    # Log adjustment for debugging
                    if hasattr(self, "_debug_console_enabled") and self._debug_console_enabled:
                        self._debug_console(
                            f"adaptive_horizon:{current_symbol}:{interval}",
                            f"Adaptive horizon: {base_horizon} -> {adjusted_horizon} (vol={volatility:.3f}, conf={confidence:.2f})",
                            min_gap_seconds=5.0,
                            level="info",
                        )

                    return adjusted_horizon

        return base_horizon

    def _refresh_all(self) -> None:
        self._update_watchlist()
        self._refresh_sentiment()
        self._refresh_universe_catalog(force=True)
        self.log("Refreshed all data", "info")

    # =========================================================================
    # =========================================================================

    def _toggle_trading(self) -> None:
        self.log("Broker connection is disabled in this build.", "info")

    def _on_mode_combo_changed(self, index: int) -> None:
        _ = index

    def _set_trading_mode(
        self,
        mode: TradingMode,
        prompt_reconnect: bool = False,
    ) -> None:
        _ = prompt_reconnect
        mode = TradingMode.SIMULATION if mode != TradingMode.LIVE else TradingMode.LIVE
        try:
            CONFIG.trading_mode = TradingMode.SIMULATION
        except _UI_RECOVERABLE_EXCEPTIONS:
            pass
        if mode == TradingMode.LIVE:
            self.log("Live broker mode is disabled. Using simulation mode.", "warning")

    def _connect_trading(self) -> None:
        self.log("Broker connection is disabled in this build.", "info")

    def _disconnect_trading(self) -> None:
        self.log("Broker connection is disabled in this build.", "info")

    def _on_chart_trade_requested(self, side: str, price: float) -> None:
        _ = (side, price)
        self.log("Order execution is disabled in this build.", "info")

    def _show_chart_trade_dialog(
        self,
        symbol: str,
        side: str,
        clicked_price: float,
        lot: int,
    ) -> dict[str, float | int | str | bool] | None:
        _ = (symbol, side, clicked_price, lot)
        return None

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
        _ = (
            symbol,
            side,
            qty,
            price,
            order_type,
            time_in_force,
            trigger_price,
            trailing_stop_pct,
            trail_limit_offset_pct,
            strict_time_in_force,
            stop_loss,
            take_profit,
            bracket,
        )
        self.log("Order execution is disabled in this build.", "info")

    def _execute_buy(self) -> None:
        self.log("Buy action is disabled in this build.", "info")

    def _execute_sell(self) -> None:
        self.log("Sell action is disabled in this build.", "info")

    def _on_order_filled(self, order: Any, fill: Any) -> None:
        _ = (order, fill)
        self.log("Order updates are disabled in this build.", "info")

    def _on_order_rejected(self, order: Any, reason: Any) -> None:
        _ = (order, reason)
        self.log("Order updates are disabled in this build.", "info")

    def _refresh_sentiment(self) -> None:
        """Refresh sentiment analysis display asynchronously."""
        if not hasattr(self, "sentiment_labels"):
            return
        existing = self.workers.get("sentiment_refresh")
        if existing and existing.isRunning():
            return

        def _set_na() -> None:
            self.sentiment_labels["overall"].setText("N/A")
            self.sentiment_labels["policy"].setText("N/A")
            self.sentiment_labels["market"].setText("N/A")
            self.sentiment_labels["confidence"].setText("N/A")

        def _set_zero() -> None:
            self.sentiment_labels["overall"].setText("0.00")
            self.sentiment_labels["policy"].setText("0.00")
            self.sentiment_labels["market"].setText("0.00")
            self.sentiment_labels["confidence"].setText("0%")

        def _work() -> dict[str, Any]:
            from data.news_collector import get_collector
            from data.sentiment_analyzer import get_analyzer

            collector = get_collector()
            analyzer = get_analyzer()
            articles = collector.collect_news(limit=50, hours_back=24)
            if not articles:
                return {"empty": True, "entities": []}

            sentiment = analyzer.analyze_articles(articles, hours_back=24)
            entities = analyzer.extract_entities(articles)
            top_entities = []
            for entity in sorted(
                entities,
                key=lambda x: getattr(x, "mention_count", 0),
                reverse=True,
            )[:10]:
                top_entities.append(
                    {
                        "entity": str(getattr(entity, "entity", "")),
                        "entity_type": str(getattr(entity, "entity_type", "")),
                        "sentiment": float(getattr(entity, "sentiment", 0.0) or 0.0),
                        "mention_count": int(getattr(entity, "mention_count", 0) or 0),
                    }
                )

            return {
                "empty": False,
                "overall": float(getattr(sentiment, "overall", 0.0) or 0.0),
                "policy": float(getattr(sentiment, "policy_impact", 0.0) or 0.0),
                "market": float(getattr(sentiment, "market_sentiment", 0.0) or 0.0),
                "confidence": float(getattr(sentiment, "confidence", 0.0) or 0.0),
                "entities": top_entities,
            }

        worker = WorkerThread(_work, timeout_seconds=45)
        self._track_worker(worker)

        def _on_done(payload: Any) -> None:
            self.workers.pop("sentiment_refresh", None)
            if not isinstance(payload, dict):
                _set_na()
                return
            if bool(payload.get("empty", False)):
                _set_zero()
                if hasattr(self, "entities_table") and self.entities_table:
                    self.entities_table.setRowCount(0)
                return

            overall = float(payload.get("overall", 0.0) or 0.0)
            policy = float(payload.get("policy", 0.0) or 0.0)
            market = float(payload.get("market", 0.0) or 0.0)
            confidence = float(payload.get("confidence", 0.0) or 0.0)
            if hasattr(self, "_set_news_policy_signal"):
                try:
                    self._set_news_policy_signal(
                        "__market__",
                        {
                            "symbol": "__market__",
                            "overall": overall,
                            "policy": policy,
                            "market": market,
                            "confidence": confidence,
                            "news_count": int(len(list(payload.get("entities", []) or []))),
                            "ts": float(time.time()),
                        },
                    )
                except _UI_RECOVERABLE_EXCEPTIONS:
                    pass

            self.sentiment_labels["overall"].setText(f"{overall:+.2f}")
            self.sentiment_labels["policy"].setText(f"{policy:+.2f}")
            self.sentiment_labels["market"].setText(f"{market:+.2f}")
            self.sentiment_labels["confidence"].setText(f"{confidence:.0%}")

            if overall > 0.3:
                color = ModernColors.ACCENT_SUCCESS
            elif overall < -0.3:
                color = ModernColors.ACCENT_DANGER
            else:
                color = ModernColors.TEXT_PRIMARY
            self.sentiment_labels["overall"].setStyleSheet(
                f"color: {color}; "
                f"font-size: {ModernFonts.SIZE_XXL}px; "
                f"font-weight: {ModernFonts.WEIGHT_BOLD};"
            )

            if hasattr(self, "entities_table") and self.entities_table:
                self.entities_table.setRowCount(0)
                for entity in list(payload.get("entities", []) or []):
                    row = self.entities_table.rowCount()
                    self.entities_table.insertRow(row)
                    self.entities_table.setItem(
                        row, 0, self._make_item(str(entity.get("entity", "")))
                    )
                    self.entities_table.setItem(
                        row, 1, self._make_item(str(entity.get("entity_type", "")))
                    )
                    self.entities_table.setItem(
                        row, 2, self._make_item(f"{float(entity.get('sentiment', 0.0) or 0.0):+.2f}")
                    )
                    self.entities_table.setItem(
                        row, 3, self._make_item(str(int(entity.get("mention_count", 0) or 0)))
                    )

        def _on_error(err: str) -> None:
            self.workers.pop("sentiment_refresh", None)
            log.debug("Sentiment refresh failed: %s", err)
            _set_na()

        worker.result.connect(_on_done)
        worker.error.connect(_on_error)
        self.workers["sentiment_refresh"] = worker
        worker.start()

    # =========================================================================
    # =========================================================================

    def _start_training(self) -> None:
        _start_training_impl(self)

    def _show_auto_learn(self, auto_start: bool = False) -> Any | None:
        return _show_auto_learn_impl(self, auto_start=auto_start)

    def _show_backtest(self) -> None:
        _show_backtest_impl(self)

    def _show_about(self) -> None:
        _show_about_impl(self)

    # =========================================================================
    # AUTO-TRADE CONTROLS
    # =========================================================================

    def _on_trade_mode_changed(self, index: int) -> None:
        _ = index

    def _apply_auto_trade_mode(self, mode: AutoTradeMode) -> None:
        self._auto_trade_mode = AutoTradeMode.MANUAL

    def _update_auto_trade_status_label(self, mode: AutoTradeMode) -> None:
        _ = mode

    def _toggle_auto_pause(self) -> None:
        return

    def _approve_all_pending(self) -> None:
        return

    def _reject_all_pending(self) -> None:
        return

    def _show_auto_trade_settings(self) -> None:
        self.log("Auto-trade settings are disabled in this build.", "info")

    def _on_auto_trade_action_safe(self, action: AutoTradeAction) -> None:
        _ = action

    def _on_auto_trade_action(self, action: AutoTradeAction) -> None:
        _ = action

    def _on_pending_approval_safe(self, action: AutoTradeAction) -> None:
        _ = action

    def _on_pending_approval(self, action: AutoTradeAction) -> None:
        _ = action

    def _refresh_pending_table(self) -> None:
        return

    def _refresh_auto_trade_ui(self) -> None:
        return

    # =========================================================================
    # =========================================================================

    def _update_clock(self) -> None:
        """Update clock."""
        self.time_label.setText(
            datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )

    def _update_market_status(self) -> None:
        """Update market status."""
        is_open = CONFIG.is_market_open()
        hours_text = self._market_hours_text()
        now_sh = self._shanghai_now()

        if is_open:
            self.market_label.setText(
                f"Market Open | Trading Hours: {hours_text}"
            )
            self.market_label.setStyleSheet(
                
                    f"color: {ModernColors.ACCENT_SUCCESS};"
                    f" font-weight: {ModernFonts.WEIGHT_BOLD};"
                
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
            self.market_label.setStyleSheet(
                f"color: {ModernColors.ACCENT_DANGER};"
            )

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

def _bind_mainapp_extracted_ops() -> None:
    bindings: dict[str, Any] = {
        "_seven_day_lookback": _app_bar_ops._seven_day_lookback,
        "_trained_stock_window_bars": _app_bar_ops._trained_stock_window_bars,
        "_recommended_lookback": _app_bar_ops._recommended_lookback,
        "_queue_history_refresh": _app_bar_ops._queue_history_refresh,
        "_consume_history_refresh": _app_bar_ops._consume_history_refresh,
        "_schedule_analysis_recovery": _app_bar_ops._schedule_analysis_recovery,
        "_history_window_bars": _app_bar_ops._history_window_bars,
        "_ts_to_epoch": _app_bar_ops._ts_to_epoch,
        "_epoch_to_iso": _app_bar_ops._epoch_to_iso,
        "_now_iso": _app_bar_ops._now_iso,
        "_merge_bars": _app_bar_ops._merge_bars,
        "_interval_seconds": _app_bar_ops._interval_seconds,
        "_interval_token_from_seconds": _app_bar_ops._interval_token_from_seconds,
        "_bars_needed_from_base_interval": _app_bar_ops._bars_needed_from_base_interval,
        "_resample_chart_bars": _app_bar_ops._resample_chart_bars,
        "_dominant_bar_interval": _app_bar_ops._dominant_bar_interval,
        "_effective_anchor_price": _app_bar_ops._effective_anchor_price,
        "_stabilize_chart_depth": _app_bar_ops._stabilize_chart_depth,
        "_bar_bucket_epoch": _app_bar_ops._bar_bucket_epoch,
        "_bar_trading_date": _app_bar_ops._bar_trading_date,
        "_is_intraday_day_boundary": _app_bar_ops._is_intraday_day_boundary,
        "_shanghai_now": _app_bar_ops._shanghai_now,
        "_is_cn_trading_day": _app_bar_ops._is_cn_trading_day,
        "_market_hours_text": _app_bar_ops._market_hours_text,
        "_next_market_open": _app_bar_ops._next_market_open,
        "_is_market_session_timestamp": _app_bar_ops._is_market_session_timestamp,
        "_filter_bars_to_market_session": _app_bar_ops._filter_bars_to_market_session,
        "_bar_safety_caps": _app_bar_ops._bar_safety_caps,
        "_synthetic_tick_jump_cap": _app_bar_ops._synthetic_tick_jump_cap,
        "_sanitize_ohlc": _app_bar_ops._sanitize_ohlc,
        "_is_outlier_tick": _app_bar_ops._is_outlier_tick,
        "_get_levels_dict": _app_bar_ops._get_levels_dict,
        "_scrub_chart_bars": _app_bar_ops._scrub_chart_bars,
        "_rescale_chart_bars_to_anchor": _app_bar_ops._rescale_chart_bars_to_anchor,
        "_recover_chart_bars_from_close": _app_bar_ops._recover_chart_bars_from_close,
        "_sync_ui_to_loaded_model": _app_model_chart_ops._sync_ui_to_loaded_model,
        "_loaded_model_ui_meta": _app_model_chart_ops._loaded_model_ui_meta,
        "_has_exact_model_artifacts": _app_model_chart_ops._has_exact_model_artifacts,
        "_log_model_alignment_debug": _app_model_chart_ops._log_model_alignment_debug,
        "_debug_chart_state": _app_model_chart_ops._debug_chart_state,
        "_debug_candle_quality": _app_model_chart_ops._debug_candle_quality,
        "_debug_forecast_quality": _app_model_chart_ops._debug_forecast_quality,
        "_chart_prediction_caps": _app_model_chart_ops._chart_prediction_caps,
        "_apply_news_policy_bias_to_forecast": _app_model_chart_ops._apply_news_policy_bias_to_forecast,
        "_prepare_chart_predicted_prices": _app_model_chart_ops._prepare_chart_predicted_prices,
        "_chart_prediction_uncertainty_profile": _app_model_chart_ops._chart_prediction_uncertainty_profile,
        "_build_chart_prediction_bands": _app_model_chart_ops._build_chart_prediction_bands,
        "_resolve_chart_prediction_series": _app_model_chart_ops._resolve_chart_prediction_series,
        "_render_chart_state": _app_model_chart_ops._render_chart_state,
        "_ensure_feed_subscription": _app_feed_ops._ensure_feed_subscription,
        "_on_bar_from_feed": _app_feed_ops._on_bar_from_feed,
        "_on_tick_from_feed": _app_feed_ops._on_tick_from_feed,
        "_on_bar_ui": _app_feed_ops._on_bar_ui,
        "_render_live_bar_update": _app_feed_ops._render_live_bar_update,
        "_update_chart_latest_label": _app_feed_ops._update_chart_latest_label,
        "_get_trained_stock_codes": _app_training_ops._get_trained_stock_codes,
        "_invalidate_trained_stock_cache": _app_training_ops._invalidate_trained_stock_cache,
        "_sync_trained_stock_last_train_from_model": _app_training_ops._sync_trained_stock_last_train_from_model,
        "_get_trained_stock_set": _app_training_ops._get_trained_stock_set,
        "_is_trained_stock": _app_training_ops._is_trained_stock,
        "_persist_session_bar": _app_training_ops._persist_session_bar,
        "_submit_session_cache_write": _app_training_ops._submit_session_cache_write,
        "_on_session_cache_write_done": _app_training_ops._on_session_cache_write_done,
        "_shutdown_session_cache_writer": _app_training_ops._shutdown_session_cache_writer,
        "_filter_trained_stocks_ui": _app_training_ops._filter_trained_stocks_ui,
        "_pin_watchlist_symbol": _app_training_ops._pin_watchlist_symbol,
        "_on_trained_stock_activated": _app_training_ops._on_trained_stock_activated,
        "_refresh_trained_stock_list": _app_training_ops._refresh_trained_stock_list,
        "_update_trained_stocks_ui": _app_training_ops._update_trained_stocks_ui,
        "_focus_trained_stocks_tab": _app_training_ops._focus_trained_stocks_tab,
        "_get_infor_trained_stocks": _app_training_ops._get_infor_trained_stocks,
        "_train_trained_stocks": _app_training_ops._train_trained_stocks,
        "_handle_training_drift_alarm": _app_training_ops._handle_training_drift_alarm,
        "_append_ai_chat_message": _app_ai_ops._append_ai_chat_message,
        "_on_ai_chat_send": _app_ai_ops._on_ai_chat_send,
        "_handle_ai_chat_prompt": _app_ai_ops._handle_ai_chat_prompt,
        "_execute_ai_chat_command": _app_ai_ops._execute_ai_chat_command,
        "_build_ai_chat_response": _app_ai_ops._build_ai_chat_response,
        "_generate_ai_chat_reply": _app_ai_ops._generate_ai_chat_reply,
        "_start_llm_training": _app_ai_ops._start_llm_training,
        "_auto_train_llm": _app_ai_ops._auto_train_llm,
        "_show_llm_train_dialog": _app_ai_ops._show_llm_train_dialog,
        "_on_llm_training_session_finished": _app_ai_ops._on_llm_training_session_finished,
        "_refresh_model_training_statuses": _app_ai_ops._refresh_model_training_statuses,
        "_set_news_policy_signal": _app_ai_ops._set_news_policy_signal,
        "_news_policy_signal_for": _app_ai_ops._news_policy_signal_for,
        "_refresh_news_policy_signal": _app_ai_ops._refresh_news_policy_signal,
        "_quick_trade": _app_analysis_ops._quick_trade,
        "_update_watchlist": _app_analysis_ops._update_watchlist,
        "_on_watchlist_click": _app_analysis_ops._on_watchlist_click,
        "_add_to_watchlist": _app_analysis_ops._add_to_watchlist,
        "_remove_from_watchlist": _app_analysis_ops._remove_from_watchlist,
        "_analyze_stock": _app_analysis_ops._analyze_stock,
        "_on_analysis_done": _app_analysis_ops._on_analysis_done,
        "_on_analysis_error": _app_analysis_ops._on_analysis_error,
        "_update_details": _app_analysis_ops._update_details,
        "_add_to_history": _app_analysis_ops._add_to_history,
        "_signal_to_direction": _app_analysis_ops._signal_to_direction,
        "_compute_guess_profit": _app_analysis_ops._compute_guess_profit,
        "_refresh_guess_rows_for_symbol": _app_analysis_ops._refresh_guess_rows_for_symbol,
        "_calculate_realtime_correct_guess_profit": _app_analysis_ops._calculate_realtime_correct_guess_profit,
        "_update_correct_guess_profit_ui": _app_analysis_ops._update_correct_guess_profit_ui,
        "_scan_stocks": _app_analysis_ops._scan_stocks,
        "_on_scan_done": _app_analysis_ops._on_scan_done,
        "_init_screener_profile_ui": _app_analysis_ops._init_screener_profile_ui,
        "_on_screener_profile_changed": _app_analysis_ops._on_screener_profile_changed,
        "_show_screener_profile_dialog": _app_analysis_ops._show_screener_profile_dialog,
        "_refresh_universe_catalog": _app_universe_ops._refresh_universe_catalog,
        "_on_universe_catalog_loaded": _app_universe_ops._on_universe_catalog_loaded,
        "_on_universe_catalog_error": _app_universe_ops._on_universe_catalog_error,
        "_filter_universe_list": _app_universe_ops._filter_universe_list,
        "_on_universe_item_activated": _app_universe_ops._on_universe_item_activated,
    }
    bind_methods(
        MainApp,
        bindings,
        context="ui.app.MainApp",
    )


_bind_mainapp_extracted_ops()

def run_app() -> None:
    """Run the application with modern professional theme."""
    os.environ.setdefault('QT_AUTO_SCREEN_SCALE_FACTOR', '1')

    app = QApplication(sys.argv)
    app.setStyle('Fusion')

    app.setApplicationName("AI Stock Analysis System")
    app.setApplicationVersion("2.0")
    app.setOrganizationName("AI Trading")

    # Set modern font
    font = QFont(get_primary_font_family(), ModernFonts.SIZE_BASE)
    app.setFont(font)
    
    # Apply modern theme
    from ui.modern_theme import apply_modern_theme
    apply_modern_theme(app)

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
        except _UI_RECOVERABLE_EXCEPTIONS as exc:
            log.debug("Suppressed exception in ui/app.py", exc_info=exc)
        _restore_sigint_handler(previous_sigint)

    sys.exit(exit_code)

def _install_sigint_handler(app: QApplication) -> Any | None:
    """Route Ctrl+C to Qt quit for graceful terminal shutdown."""
    try:
        previous = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, lambda *_: app.quit())
        return previous
    except _UI_RECOVERABLE_EXCEPTIONS as e:
        log.debug(f"SIGINT handler install failed: {e}")
        return None

def _restore_sigint_handler(previous_handler: Any | None) -> None:
    """Restore previous SIGINT handler after app loop exits."""
    if previous_handler is None:
        return
    try:
        signal.signal(signal.SIGINT, previous_handler)
    except _UI_RECOVERABLE_EXCEPTIONS as exc:
        log.debug("Suppressed exception in ui/app.py", exc_info=exc)

if __name__ == "__main__":
    run_app()
