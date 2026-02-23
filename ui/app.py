# ui/app.py
import os
import signal
import sys
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime
from importlib import import_module
from typing import Any

from PyQt6.QtCore import QSize, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QAction, QActionGroup, QFont
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
    QMainWindow,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QStatusBar,
    QTableWidget,
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
from ui.modern_theme import ModernFonts
from ui import app_analysis_ops as _app_analysis_ops
from ui import app_bar_ops as _app_bar_ops
from ui import app_feed_ops as _app_feed_ops
from ui import app_model_chart_ops as _app_model_chart_ops
from ui import app_training_ops as _app_training_ops
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
from utils.logger import get_logger
from utils.recoverable import COMMON_RECOVERABLE_EXCEPTIONS

log = get_logger(__name__)

_UI_RECOVERABLE_EXCEPTIONS = COMMON_RECOVERABLE_EXCEPTIONS

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

        # FIX: Bounded cache dicts with max size to prevent memory leaks
        self._MAX_CACHED_BARS = 500  # Max symbols with cached bars
        self._MAX_CACHED_QUOTES = 1000  # Max symbols with cached quotes
        self._bars_by_symbol: dict[str, list[dict[str, Any]]] = {}
        self._trained_stock_codes_cache: list[str] = []
        self._trained_stock_last_train: dict[str, str] = {}
        self._last_bar_feed_ts: dict[str, float] = {}
        self._chart_symbol: str = ""
        self._history_refresh_once: set[tuple[str, str]] = set()
        self._strict_startup = env_flag("TRADING_STRICT_STARTUP", "0")
        self._debug_console_enabled = env_flag("TRADING_DEBUG_CONSOLE", "1")
        self._debug_console_last_emit: dict[str, float] = {}
        self._syncing_mode_ui = False
        self._session_bar_cache = None
        try:
            from data.session_cache import get_session_bar_cache
            self._session_bar_cache = get_session_bar_cache()
        except _UI_RECOVERABLE_EXCEPTIONS as exc:
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
        except _UI_RECOVERABLE_EXCEPTIONS:
            log.exception("Startup state restore failed")
            if self._strict_startup:
                raise

        QTimer.singleShot(0, self._init_components)














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

        # FIX: Cache pruning to prevent memory leaks
        self.cache_prune_timer = QTimer()
        self.cache_prune_timer.timeout.connect(self._prune_caches)
        self.cache_prune_timer.start(60000)  # Prune every 60 seconds

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

        except _UI_RECOVERABLE_EXCEPTIONS as e:
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
            except _UI_RECOVERABLE_EXCEPTIONS as e:
                log.debug(f"Auto-start monitoring failed: {e}")

        if self._debug_console_enabled:
            self.log(
                "Debug console enabled (set TRADING_DEBUG_CONSOLE=0 to disable)",
                "warning",
            )
        self.log("System initialized - Ready for trading", "info")

    def _prune_caches(self) -> None:
        """
        Prune internal caches to prevent memory leaks.

        FIX: Called periodically to bound cache sizes.
        """
        # Prune _last_bar_feed_ts - keep only recent entries
        if len(self._last_bar_feed_ts) > self._MAX_CACHED_QUOTES:
            # Remove oldest 25%
            sorted_items = sorted(
                self._last_bar_feed_ts.items(),
                key=lambda x: x[1]
            )
            cutoff = len(sorted_items) // 4
            for key, _ in sorted_items[:cutoff]:
                self._last_bar_feed_ts.pop(key, None)

        # Prune _bars_by_symbol - keep only most recent
        if len(self._bars_by_symbol) > self._MAX_CACHED_BARS:
            # Remove oldest entries (by last access time if available)
            # For simplicity, remove first 25%
            keys_to_remove = list(self._bars_by_symbol.keys())[:len(self._bars_by_symbol) // 4]
            for key in keys_to_remove:
                self._bars_by_symbol.pop(key, None)

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
            except _UI_RECOVERABLE_EXCEPTIONS as e:
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
    }
    for name, fn in bindings.items():
        setattr(MainApp, name, fn)


_bind_mainapp_extracted_ops()

def run_app() -> None:
    """Run the application with modern professional theme"""
    os.environ.setdefault('QT_AUTO_SCREEN_SCALE_FACTOR', '1')

    app = QApplication(sys.argv)
    app.setStyle('Fusion')

    app.setApplicationName("AI Stock Trading System")
    app.setApplicationVersion("2.0")
    app.setOrganizationName("AI Trading")

    # Set modern font
    font = QFont(ModernFonts.FAMILY_PRIMARY, ModernFonts.SIZE_BASE)
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
