# ui/app.py
import sys
import os
from datetime import datetime
from typing import Optional, Dict, List, Any
import threading
import time

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QLineEdit, QGroupBox, QProgressBar,
    QTabWidget, QStatusBar, QTextEdit, QDoubleSpinBox, QSpinBox,
    QSplitter, QComboBox, QMessageBox, QListWidget, QGridLayout,
    QFrame, QTableWidget, QTableWidgetItem, QHeaderView, QToolBar,
    QDockWidget, QSystemTrayIcon, QMenu, QSizePolicy, QCheckBox
)
from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal, QSize
from PyQt6.QtGui import QFont, QColor, QIcon, QAction, QPalette

from config.settings import CONFIG, TradingMode
from core.types import AutoTradeMode, AutoTradeState, AutoTradeAction
from utils.logger import get_logger

log = get_logger(__name__)

# =============================================================================
# LAZY IMPORTS - Avoid circular imports and improve startup time
# =============================================================================

def get_predictor_class():
    """Lazy import Predictor"""
    from models.predictor import Predictor
    return Predictor


def get_prediction_class():
    """Lazy import Prediction"""
    from models.predictor import Prediction
    return Prediction


def get_signal_class():
    """Lazy import Signal"""
    from models.predictor import Signal
    return Signal


def get_trainer_class():
    """Lazy import Trainer"""
    from models.trainer import Trainer
    return Trainer


def get_execution_engine():
    """Lazy import ExecutionEngine"""
    from trading.executor import ExecutionEngine
    return ExecutionEngine


def get_news_panel():
    """Lazy import NewsPanel"""
    from ui.news_widget import NewsPanel
    return NewsPanel


def get_auto_trader_class():
    """Lazy import AutoTrader"""
    from trading.executor import AutoTrader
    return AutoTrader


# =============================================================================
# STOCK CODE VALIDATION
# =============================================================================

def _validate_stock_code(code: str) -> bool:
    """Validate that a stock code is a valid 6-digit Chinese stock code."""
    if not code:
        return False
    digits = "".join(c for c in str(code).strip() if c.isdigit())
    if len(digits) != 6:
        return False
    # Valid prefixes for Chinese stocks
    valid_prefixes = (
        "000", "001", "002", "003",  # SZSE main/SME
        "300", "301",                 # ChiNext
        "600", "601", "603", "605",  # SSE main
        "688",                        # STAR Market
        "83", "87", "43",            # BSE
    )
    return digits.startswith(valid_prefixes)


def _normalize_stock_code(text: str) -> str:
    """Normalize stock code: strip prefixes/suffixes, keep digits, zero-pad."""
    if not text:
        return ""
    text = str(text).strip()
    for prefix in ("sh", "sz", "SH", "SZ", "bj", "BJ"):
        if text.startswith(prefix):
            text = text[len(prefix):]
    for suffix in (".SS", ".SZ", ".BJ"):
        if text.endswith(suffix):
            text = text[:-len(suffix)]
    text = "".join(c for c in text if c.isdigit())
    return text.zfill(6) if text else ""


# =============================================================================
# REAL-TIME MONITORING THREAD
# =============================================================================

class RealTimeMonitor(QThread):
    """
    Real-time market monitoring thread.
    Continuously checks for trading signals using the predictor.

    Features:
    - Uses Predictor for fast inference
    - Supports multiple intervals (1m, 5m, 1d, etc.)
    - Graceful error handling with exponential backoff
    - Thread-safe signal emission
    - Watchlist size guard
    """
    MAX_WATCHLIST_SIZE = 50

    signal_detected = pyqtSignal(object)   # Prediction
    price_updated = pyqtSignal(str, float)  # code, price
    error_occurred = pyqtSignal(str)
    status_changed = pyqtSignal(str)        # status message

    def __init__(
        self,
        predictor: Any,
        watch_list: List[str],
        interval: str = "1m",
        forecast_minutes: int = 30,
        lookback_bars: int = 1400
    ):
        super().__init__()
        self.predictor = predictor
        self.watch_list = list(watch_list)[:self.MAX_WATCHLIST_SIZE]
        self.running = False
        self._stop_event = threading.Event()

        if len(watch_list) > self.MAX_WATCHLIST_SIZE:
            log.warning(
                f"Watchlist truncated from {len(watch_list)} "
                f"to {self.MAX_WATCHLIST_SIZE}"
            )

        # Configuration
        self.scan_interval = 30  # seconds between scan cycles
        self.data_interval = str(interval).lower()
        self.forecast_minutes = int(forecast_minutes)
        self.lookback_bars = int(lookback_bars)

        # Backoff settings
        self._backoff = 1
        self._max_backoff = 60

    def run(self):
        """
        Monitoring loop:
        - Uses quick batch for speed
        - Runs full prediction (with future graph) only for strongest signals
        - Handles errors gracefully with exponential backoff
        """
        self.running = True
        self._stop_event.clear()
        self._backoff = 1

        self.status_changed.emit("Monitoring started")

        while self.running and not self._stop_event.is_set():
            loop_start = time.time()

            try:
                # Quick batch prediction for all watchlist
                preds = self.predictor.predict_quick_batch(
                    self.watch_list,
                    use_realtime_price=True,
                    interval=self.data_interval,
                    lookback_bars=self.lookback_bars
                )

                # Emit price updates
                for p in preds:
                    if hasattr(p, 'current_price') and p.current_price > 0:
                        self.price_updated.emit(p.stock_code, p.current_price)

                # Get Signal class
                Signal = get_signal_class()

                # Filter for strong signals
                strong = [
                    p for p in preds
                    if hasattr(p, 'signal') and p.signal in [
                        Signal.STRONG_BUY, Signal.STRONG_SELL,
                        Signal.BUY, Signal.SELL
                    ]
                    and hasattr(p, 'confidence')
                    and p.confidence >= CONFIG.MIN_CONFIDENCE
                ]

                # Sort by confidence and cap at 2
                strong.sort(key=lambda x: x.confidence, reverse=True)
                strong = strong[:2]

                # Full prediction for strong signals (includes forecast)
                for p in strong:
                    if self._stop_event.is_set():
                        break

                    try:
                        full = self.predictor.predict(
                            p.stock_code,
                            use_realtime_price=True,
                            interval=self.data_interval,
                            forecast_minutes=self.forecast_minutes,
                            lookback_bars=self.lookback_bars,
                            skip_cache=True
                        )
                        self.signal_detected.emit(full)
                    except Exception as e:
                        log.warning(
                            f"Full prediction failed for {p.stock_code}: {e}"
                        )

                # Reset backoff on success
                self._backoff = 1
                self.status_changed.emit(
                    f"Scanned {len(preds)} stocks, {len(strong)} signals"
                )

            except Exception as e:
                error_msg = str(e)
                self.error_occurred.emit(error_msg)
                log.warning(f"Monitor error: {error_msg}")

                # Exponential backoff on error
                sleep_s = min(self._max_backoff, self._backoff)
                self._backoff = min(self._max_backoff, self._backoff * 2)

                self.status_changed.emit(f"Error, retrying in {sleep_s}s")

                for _ in range(sleep_s):
                    if self._stop_event.is_set():
                        break
                    time.sleep(1)
                continue

            # Wait for next scan cycle
            elapsed = time.time() - loop_start
            remaining = max(0.0, self.scan_interval - elapsed)

            for _ in range(int(remaining)):
                if self._stop_event.is_set():
                    break
                time.sleep(1)

        self.status_changed.emit("Monitoring stopped")

    def stop(self):
        """Stop monitoring gracefully"""
        self.running = False
        self._stop_event.set()

    def update_config(
        self,
        interval: str = None,
        forecast_minutes: int = None,
        lookback_bars: int = None
    ):
        """Update monitoring configuration"""
        if interval:
            self.data_interval = str(interval).lower()
        if forecast_minutes:
            self.forecast_minutes = int(forecast_minutes)
        if lookback_bars:
            self.lookback_bars = int(lookback_bars)


class WorkerThread(QThread):
    """Generic worker thread for background tasks with timeout support"""
    finished = pyqtSignal(object)
    error = pyqtSignal(str)
    progress = pyqtSignal(int, str)

    def __init__(self, func, *args, timeout_seconds: float = 300, **kwargs):
        super().__init__()
        self.func = func
        self.args = args
        self.kwargs = kwargs
        self._cancelled = False
        self._timeout = timeout_seconds

    def run(self):
        try:
            if self._cancelled:
                return

            # Run function with timeout via threading.Timer watchdog
            result_holder = [None]
            error_holder = [None]
            done_event = threading.Event()

            def target():
                try:
                    result_holder[0] = self.func(*self.args, **self.kwargs)
                except Exception as e:
                    error_holder[0] = e
                finally:
                    done_event.set()

            worker = threading.Thread(target=target, daemon=True)
            worker.start()

            # Wait with timeout
            done_event.wait(timeout=self._timeout)

            if not done_event.is_set():
                self._cancelled = True
                if not self._cancelled:
                    self.error.emit(
                        f"Operation timed out after {self._timeout}s"
                    )
                return

            if self._cancelled:
                return

            if error_holder[0] is not None:
                self.error.emit(str(error_holder[0]))
            else:
                self.finished.emit(result_holder[0])

        except Exception as e:
            if not self._cancelled:
                self.error.emit(str(e))

    def cancel(self):
        """Cancel the worker"""
        self._cancelled = True


# =============================================================================
# MAIN APPLICATION
# =============================================================================

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

    bar_received = pyqtSignal(str, dict)

    def __init__(self):
        super().__init__()

        self.setWindowTitle("AI Stock Trading System v2.0")
        self.setGeometry(50, 50, 1800, 1000)

        # State
        self.predictor = None
        self.executor = None
        self.current_prediction = None
        self.workers: Dict[str, WorkerThread] = {}
        self.monitor: Optional[RealTimeMonitor] = None
        self.watch_list: List[str] = CONFIG.STOCK_POOL[:10]

        # Real-time state with thread safety
        self._last_forecast_refresh_ts: float = 0.0
        self._live_price_series: Dict[str, List[float]] = {}
        self._price_series_lock = threading.Lock()

        # Bar data
        self._bars_by_symbol: Dict[str, List[dict]] = {}

        # Auto-trade state
        self._auto_trade_mode: AutoTradeMode = AutoTradeMode.MANUAL

        # Setup UI
        self._setup_menubar()
        self._setup_toolbar()
        self._setup_ui()
        self._setup_statusbar()
        self._setup_timers()
        self._apply_professional_style()
        self.bar_received.connect(self._on_bar_ui)

        # Load state BEFORE initializing model
        try:
            self._load_state()
            self._update_watchlist()
        except Exception:
            pass

        # Initialize components AFTER state load
        QTimer.singleShot(0, self._init_components)

    # =========================================================================
    # UI NORMALIZATION (FIX #1 - was missing entirely)
    # =========================================================================

    def _ui_norm(self, text: str) -> str:
        """Normalize stock code for UI comparison."""
        return _normalize_stock_code(text)

    # =========================================================================
    # MENU BAR
    # =========================================================================

    def _setup_menubar(self):
        """Setup professional menu bar"""
        menubar = self.menuBar()

        # File Menu
        file_menu = menubar.addMenu("&File")

        new_action = QAction("&New Workspace", self)
        new_action.setShortcut("Ctrl+N")
        file_menu.addAction(new_action)

        file_menu.addSeparator()

        exit_action = QAction("E&xit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # Trading Menu
        trading_menu = menubar.addMenu("&Trading")

        connect_action = QAction("&Connect Broker", self)
        connect_action.triggered.connect(self._toggle_trading)
        trading_menu.addAction(connect_action)

        trading_menu.addSeparator()

        paper_action = QAction("&Paper Trading Mode", self)
        paper_action.setCheckable(True)
        paper_action.setChecked(True)
        trading_menu.addAction(paper_action)

        live_action = QAction("&Live Trading Mode", self)
        live_action.setCheckable(True)
        trading_menu.addAction(live_action)

        # AI Menu
        ai_menu = menubar.addMenu("&AI Model")

        train_action = QAction("&Train Model", self)
        train_action.setShortcut("Ctrl+T")
        train_action.triggered.connect(self._start_training)
        ai_menu.addAction(train_action)

        auto_learn_action = QAction("&Auto Learn", self)
        auto_learn_action.triggered.connect(self._show_auto_learn)
        ai_menu.addAction(auto_learn_action)

        ai_menu.addSeparator()

        backtest_action = QAction("&Backtest", self)
        backtest_action.triggered.connect(self._show_backtest)
        ai_menu.addAction(backtest_action)

        # View Menu
        view_menu = menubar.addMenu("&View")

        refresh_action = QAction("&Refresh", self)
        refresh_action.setShortcut("F5")
        refresh_action.triggered.connect(self._refresh_all)
        view_menu.addAction(refresh_action)

        # Help Menu
        help_menu = menubar.addMenu("&Help")

        about_action = QAction("&About", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)

    # =========================================================================
    # TOOLBAR
    # =========================================================================

    def _setup_toolbar(self):
        """Setup professional toolbar with auto-trade controls"""
        toolbar = QToolBar("Main Toolbar")
        toolbar.setIconSize(QSize(24, 24))
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        # Analyze button
        self.analyze_action = QAction("🔍 Analyze", self)
        self.analyze_action.triggered.connect(self._analyze_stock)
        toolbar.addAction(self.analyze_action)

        toolbar.addSeparator()

        # Real-time monitoring toggle
        self.monitor_action = QAction("📡 Start Monitoring", self)
        self.monitor_action.setCheckable(True)
        self.monitor_action.triggered.connect(self._toggle_monitoring)
        toolbar.addAction(self.monitor_action)

        toolbar.addSeparator()

        # Quick scan
        scan_action = QAction("🔎 Scan All", self)
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
        self.auto_trade_status_label = QLabel("  ⚪ Manual  ")
        self.auto_trade_status_label.setStyleSheet(
            "color: #8b949e; font-weight: bold; padding: 0 8px;"
        )
        toolbar.addWidget(self.auto_trade_status_label)

        # Auto-trade settings button
        auto_settings_action = QAction("⚙️ Auto Settings", self)
        auto_settings_action.triggered.connect(self._show_auto_trade_settings)
        toolbar.addAction(auto_settings_action)

        toolbar.addSeparator()

        # Spacer
        spacer = QWidget()
        spacer.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred
        )
        toolbar.addWidget(spacer)

        # Stock input in toolbar
        toolbar.addWidget(QLabel("  Stock: "))
        self.stock_input = QLineEdit()
        self.stock_input.setPlaceholderText("Enter code (e.g., 600519)")
        self.stock_input.setFixedWidth(150)
        self.stock_input.returnPressed.connect(self._analyze_stock)
        toolbar.addWidget(self.stock_input)

        # =========================================================================
        # FEED SUBSCRIPTION
    # =========================================================================

    def _ensure_feed_subscription(self, code: str):
        """Subscribe symbol to realtime feed + ensure bar interval matches UI."""
        try:
            from data.feeds import get_feed_manager
            fm = get_feed_manager(auto_init=True, async_init=True)

            # set bar interval from UI interval
            interval = self.interval_combo.currentText().strip().lower()
            bar_seconds_map = {
                "1m": 60, "5m": 300, "15m": 900,
                "30m": 1800, "60m": 3600, "1h": 3600,
            }
            bar_seconds = bar_seconds_map.get(interval, 60)

            fm.set_bar_interval_seconds(bar_seconds)
            fm.subscribe(code)

            # attach bar callback ONCE
            if not getattr(self, "_bar_callback_attached", False):
                self._bar_callback_attached = True
                fm.add_bar_callback(self._on_bar_from_feed)

        except Exception as e:
            log.debug(f"Feed subscription failed: {e}")

    def _on_bar_from_feed(self, symbol: str, bar: dict):
        """
        Called from feed thread (NOT UI thread).
        Emit signal to update UI safely.
        """
        try:
            self.bar_received.emit(str(symbol), dict(bar))
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

        # Get or create bar array for this symbol
        arr = self._bars_by_symbol.get(symbol)
        if arr is None:
            arr = []
            self._bars_by_symbol[symbol] = arr

        # Check if this is a partial (live) bar or final bar
        is_final = bar.get("final", True)

        if is_final:
            # Final bar - append to history
            arr.append(bar)
            if len(arr) > 400:
                del arr[:-400]
        else:
            # Partial bar - update the last bar in place
            if arr:
                # Update existing last bar
                arr[-1] = bar
            else:
                # First bar - just append
                arr.append(bar)

        # Only update chart if this is the currently displayed stock
        current_code = self._ui_norm(self.stock_input.text())
        if current_code != symbol:
            return

        # Get prediction if available
        predicted = []
        if (
            self.current_prediction
            and getattr(self.current_prediction, "stock_code", "") == symbol
        ):
            predicted = (
                getattr(self.current_prediction, "predicted_prices", [])
                or []
            )

        # UNIFIED chart update - draws candles + line + prediction
        try:
            if hasattr(self.chart, 'update_chart'):
                self.chart.update_chart(
                    arr,
                    predicted_prices=predicted,
                    levels=self._get_levels_dict()
                )
            elif hasattr(self.chart, 'update_candles'):
                self.chart.update_candles(
                    arr,
                    predicted_prices=predicted,
                    levels=self._get_levels_dict()
                )
        except Exception as e:
            log.debug(f"Chart update failed: {e}")

    # =========================================================================
    # MAIN UI LAYOUT
    # =========================================================================

    def _setup_ui(self):
        """Setup main UI with professional layout"""
        central = QWidget()
        self.setCentralWidget(central)

        layout = QHBoxLayout(central)
        layout.setSpacing(10)
        layout.setContentsMargins(10, 10, 10, 10)

        # Main splitter
        main_splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left Panel - Control & Watchlist
        left_panel = self._create_left_panel()

        # Center Panel - Charts & Signals
        center_panel = self._create_center_panel()

        # Right Panel - Portfolio & Orders
        right_panel = self._create_right_panel()

        main_splitter.addWidget(left_panel)
        main_splitter.addWidget(center_panel)
        main_splitter.addWidget(right_panel)
        main_splitter.setSizes([300, 900, 500])

        layout.addWidget(main_splitter)

    def _create_left_panel(self) -> QWidget:
        """Create left control panel with interval/forecast settings"""
        panel = QWidget()
        panel.setMaximumWidth(350)
        layout = QVBoxLayout(panel)
        layout.setSpacing(10)

        # Watchlist
        watchlist_group = QGroupBox("📋 Watchlist")
        watchlist_layout = QVBoxLayout()

        self.watchlist = QTableWidget()
        self.watchlist.setColumnCount(4)
        self.watchlist.setHorizontalHeaderLabels(
            ["Code", "Price", "Change", "Signal"]
        )
        self.watchlist.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        self.watchlist.setMaximumHeight(250)
        self.watchlist.cellDoubleClicked.connect(self._on_watchlist_click)

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

        # Trading Settings
        settings_group = QGroupBox("⚙️ Trading Settings")
        settings_layout = QGridLayout()

        settings_layout.addWidget(QLabel("Mode:"), 0, 0)
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Paper Trading", "Live Trading"])
        settings_layout.addWidget(self.mode_combo, 0, 1)

        settings_layout.addWidget(QLabel("Capital:"), 1, 0)
        self.capital_spin = QDoubleSpinBox()
        self.capital_spin.setRange(10000, 100000000)
        self.capital_spin.setValue(CONFIG.CAPITAL)
        self.capital_spin.setPrefix("¥ ")
        settings_layout.addWidget(self.capital_spin, 1, 1)

        settings_layout.addWidget(QLabel("Risk/Trade:"), 2, 0)
        self.risk_spin = QDoubleSpinBox()
        self.risk_spin.setRange(0.5, 5.0)
        self.risk_spin.setValue(CONFIG.RISK_PER_TRADE)
        self.risk_spin.setSuffix(" %")
        settings_layout.addWidget(self.risk_spin, 2, 1)

        # Interval selector
        settings_layout.addWidget(QLabel("Interval:"), 3, 0)
        self.interval_combo = QComboBox()
        self.interval_combo.addItems(["1m", "5m", "15m", "30m", "60m", "1d"])
        self.interval_combo.setCurrentText("1m")
        self.interval_combo.currentTextChanged.connect(
            self._on_interval_changed
        )
        settings_layout.addWidget(self.interval_combo, 3, 1)

        # Forecast horizon
        settings_layout.addWidget(QLabel("Forecast:"), 4, 0)
        self.forecast_spin = QSpinBox()
        self.forecast_spin.setRange(5, 120)
        self.forecast_spin.setValue(30)
        self.forecast_spin.setSuffix(" bars")
        self.forecast_spin.setToolTip("Number of bars to forecast ahead")
        settings_layout.addWidget(self.forecast_spin, 4, 1)

        # Lookback bars
        settings_layout.addWidget(QLabel("Lookback:"), 5, 0)
        self.lookback_spin = QSpinBox()
        self.lookback_spin.setRange(100, 5000)
        self.lookback_spin.setValue(1400)
        self.lookback_spin.setSuffix(" bars")
        self.lookback_spin.setToolTip("Historical bars to use for analysis")
        settings_layout.addWidget(self.lookback_spin, 5, 1)

        settings_group.setLayout(settings_layout)
        layout.addWidget(settings_group)

        # Connection Status
        connection_group = QGroupBox("🔌 Connection")
        connection_layout = QVBoxLayout()

        self.connection_status = QLabel("● Disconnected")
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

        # AI Model Status
        ai_group = QGroupBox("🧠 AI Model")
        ai_layout = QVBoxLayout()

        self.model_status = QLabel("Model: Loading...")
        ai_layout.addWidget(self.model_status)

        self.model_info = QLabel("")
        self.model_info.setStyleSheet("color: #888; font-size: 10px;")
        ai_layout.addWidget(self.model_info)

        self.train_btn = QPushButton("🎓 Train Model")
        self.train_btn.clicked.connect(self._start_training)
        ai_layout.addWidget(self.train_btn)

        self.auto_learn_btn = QPushButton("🤖 Auto Learn")
        self.auto_learn_btn.clicked.connect(self._show_auto_learn)
        ai_layout.addWidget(self.auto_learn_btn)

        self.train_progress = QProgressBar()
        self.train_progress.setVisible(False)
        ai_layout.addWidget(self.train_progress)

        ai_group.setLayout(ai_layout)
        layout.addWidget(ai_group)

        layout.addStretch()
        return panel

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
            self.signal_panel.setMinimumHeight(100)
        layout.addWidget(self.signal_panel)

        # Chart
        chart_group = QGroupBox("📈 Price Chart & AI Prediction")
        chart_layout = QVBoxLayout()

        try:
            from .charts import StockChart
            self.chart = StockChart()
            self.chart.setMinimumHeight(400)
        except ImportError:
            self.chart = QLabel("Chart (charts module not found)")
            self.chart.setMinimumHeight(400)
            self.chart.setAlignment(Qt.AlignmentFlag.AlignCenter)
        chart_layout.addWidget(self.chart)

        chart_group.setLayout(chart_layout)
        layout.addWidget(chart_group)

        # Analysis Details
        details_group = QGroupBox("📊 Analysis Details")
        details_layout = QVBoxLayout()

        self.details_text = QTextEdit()
        self.details_text.setReadOnly(True)
        self.details_text.setFont(QFont("Consolas", 10))
        self.details_text.setMaximumHeight(200)
        details_layout.addWidget(self.details_text)

        details_group.setLayout(details_layout)
        layout.addWidget(details_group)

        return panel

    def _create_right_panel(self) -> QWidget:
        """Create right panel with portfolio, news, orders, and auto-trade"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setSpacing(10)

        tabs = QTabWidget()

        # Portfolio Tab
        portfolio_tab = QWidget()
        portfolio_layout = QVBoxLayout(portfolio_tab)

        account_frame = QFrame()
        account_frame.setStyleSheet("""
            QFrame {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #1a1a3e, stop:1 #2a2a5a);
                border-radius: 10px; padding: 15px;
            }
        """)
        account_layout = QGridLayout(account_frame)

        self.account_labels = {}
        labels = [
            ('equity', 'Total Equity', 0, 0),
            ('cash', 'Available Cash', 0, 1),
            ('positions', 'Positions Value', 1, 0),
            ('pnl', 'Total P&L', 1, 1),
        ]

        for key, text, row, col in labels:
            container = QWidget()
            cont_layout = QVBoxLayout(container)
            cont_layout.setContentsMargins(5, 5, 5, 5)
            title = QLabel(text)
            title.setStyleSheet("color: #888; font-size: 11px;")
            value = QLabel("--")
            value.setStyleSheet(
                "color: #00E5FF; font-size: 18px; font-weight: bold;"
            )
            cont_layout.addWidget(title)
            cont_layout.addWidget(value)
            account_layout.addWidget(container, row, col)
            self.account_labels[key] = value

        portfolio_layout.addWidget(account_frame)

        try:
            from .widgets import PositionTable
            self.positions_table = PositionTable()
        except ImportError:
            self.positions_table = QTableWidget()
            self.positions_table.setColumnCount(5)
            self.positions_table.setHorizontalHeaderLabels(
                ["Code", "Qty", "Price", "Value", "P&L"]
            )
        portfolio_layout.addWidget(self.positions_table)

        tabs.addTab(portfolio_tab, "💼 Portfolio")

        # News Tab
        news_tab = QWidget()
        news_layout = QVBoxLayout(news_tab)
        try:
            NewsPanel = get_news_panel()
            self.news_panel = NewsPanel()
            news_layout.addWidget(self.news_panel)
        except Exception as e:
            log.warning(f"News panel not available: {e}")
            self.news_panel = QLabel("News panel unavailable")
            news_layout.addWidget(self.news_panel)
        tabs.addTab(news_tab, "📰 News & Policy")

        # Live Signals Tab
        signals_tab = QWidget()
        signals_layout = QVBoxLayout(signals_tab)
        self.signals_table = QTableWidget()
        self.signals_table.setColumnCount(6)
        self.signals_table.setHorizontalHeaderLabels([
            "Time", "Code", "Signal", "Confidence", "Price", "Action"
        ])
        self.signals_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        signals_layout.addWidget(self.signals_table)
        tabs.addTab(signals_tab, "📡 Live Signals")

        # History Tab
        history_tab = QWidget()
        history_layout = QVBoxLayout(history_tab)
        self.history_table = QTableWidget()
        self.history_table.setColumnCount(6)
        self.history_table.setHorizontalHeaderLabels([
            "Time", "Code", "Signal", "Prob UP", "Confidence", "Result"
        ])
        self.history_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        history_layout.addWidget(self.history_table)
        tabs.addTab(history_tab, "📜 History")

        # ==================== AUTO-TRADE TAB ====================
        auto_trade_tab = QWidget()
        auto_trade_layout = QVBoxLayout(auto_trade_tab)

        # Auto-trade status frame
        auto_status_frame = QFrame()
        auto_status_frame.setStyleSheet("""
            QFrame {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #1a1a3e, stop:1 #2a2a5a);
                border-radius: 10px; padding: 10px;
            }
        """)
        auto_status_layout = QGridLayout(auto_status_frame)

        self.auto_trade_labels = {}
        auto_labels = [
            ('mode', 'Mode', 0, 0),
            ('trades', 'Trades Today', 0, 1),
            ('pnl', 'Auto P&L', 1, 0),
            ('status', 'Status', 1, 1),
        ]

        for key, text, row, col in auto_labels:
            container = QWidget()
            cont_layout = QVBoxLayout(container)
            cont_layout.setContentsMargins(5, 5, 5, 5)
            title = QLabel(text)
            title.setStyleSheet("color: #888; font-size: 11px;")
            value = QLabel("--")
            value.setStyleSheet(
                "color: #00E5FF; font-size: 16px; font-weight: bold;"
            )
            cont_layout.addWidget(title)
            cont_layout.addWidget(value)
            auto_status_layout.addWidget(container, row, col)
            self.auto_trade_labels[key] = value

        auto_trade_layout.addWidget(auto_status_frame)

        # Pending approvals section (for semi-auto)
        pending_group = QGroupBox("⏳ Pending Approvals")
        pending_layout = QVBoxLayout()
        self.pending_table = QTableWidget()
        self.pending_table.setColumnCount(6)
        self.pending_table.setHorizontalHeaderLabels([
            "Time", "Code", "Signal", "Confidence", "Price", "Action"
        ])
        self.pending_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        self.pending_table.setMaximumHeight(150)
        pending_layout.addWidget(self.pending_table)
        pending_group.setLayout(pending_layout)
        auto_trade_layout.addWidget(pending_group)

        # Auto-trade action history
        actions_group = QGroupBox("📋 Auto-Trade Actions")
        actions_layout = QVBoxLayout()
        self.auto_actions_table = QTableWidget()
        self.auto_actions_table.setColumnCount(7)
        self.auto_actions_table.setHorizontalHeaderLabels([
            "Time", "Code", "Signal", "Confidence",
            "Decision", "Qty", "Reason"
        ])
        self.auto_actions_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        actions_layout.addWidget(self.auto_actions_table)
        actions_group.setLayout(actions_layout)
        auto_trade_layout.addWidget(actions_group)

        # Auto-trade control buttons
        auto_btn_frame = QFrame()
        auto_btn_layout = QHBoxLayout(auto_btn_frame)

        self.auto_pause_btn = QPushButton("⏸ Pause Auto")
        self.auto_pause_btn.clicked.connect(self._toggle_auto_pause)
        self.auto_pause_btn.setEnabled(False)
        auto_btn_layout.addWidget(self.auto_pause_btn)

        self.auto_approve_all_btn = QPushButton("✅ Approve All")
        self.auto_approve_all_btn.clicked.connect(self._approve_all_pending)
        self.auto_approve_all_btn.setEnabled(False)
        auto_btn_layout.addWidget(self.auto_approve_all_btn)

        self.auto_reject_all_btn = QPushButton("❌ Reject All")
        self.auto_reject_all_btn.clicked.connect(self._reject_all_pending)
        self.auto_reject_all_btn.setEnabled(False)
        auto_btn_layout.addWidget(self.auto_reject_all_btn)

        auto_trade_layout.addWidget(auto_btn_frame)

        tabs.addTab(auto_trade_tab, "🤖 Auto-Trade")

        layout.addWidget(tabs)

        # Log
        log_group = QGroupBox("📋 System Log")
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

        # Action Buttons
        action_frame = QFrame()
        action_frame.setStyleSheet("""
            QFrame {
                background: #1a1a3e;
                border-radius: 8px;
                padding: 10px;
            }
        """)
        action_layout = QHBoxLayout(action_frame)

        self.buy_btn = QPushButton("📈 BUY")
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

        self.sell_btn = QPushButton("📉 SELL")
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

        # Progress bar
        self.progress = QProgressBar()
        self.progress.setMaximumWidth(200)
        self.progress.setMaximumHeight(15)
        self.progress.hide()
        self._status_bar.addWidget(self.progress)

        # Status
        self.status_label = QLabel("Ready")
        self._status_bar.addWidget(self.status_label)

        # Market status
        self.market_label = QLabel("")
        self._status_bar.addPermanentWidget(self.market_label)

        # Monitoring status
        self.monitor_label = QLabel("Monitoring: OFF")
        self.monitor_label.setStyleSheet("color: #888;")
        self._status_bar.addWidget(self.monitor_label)

        # Clock
        self.time_label = QLabel("")
        self._status_bar.addWidget(self.time_label)

    def _setup_timers(self):
        """Setup update timers"""
        # Clock
        self.clock_timer = QTimer()
        self.clock_timer.timeout.connect(self._update_clock)
        self.clock_timer.start(1000)

        # Market status
        self.market_timer = QTimer()
        self.market_timer.timeout.connect(self._update_market_status)
        self.market_timer.start(60000)

        # Portfolio refresh
        self.portfolio_timer = QTimer()
        self.portfolio_timer.timeout.connect(self._refresh_portfolio)
        self.portfolio_timer.start(5000)

        # Watchlist refresh
        self.watchlist_timer = QTimer()
        self.watchlist_timer.timeout.connect(self._update_watchlist)
        self.watchlist_timer.start(30000)

        # Auto-trade UI refresh
        self.auto_trade_timer = QTimer()
        self.auto_trade_timer.timeout.connect(self._refresh_auto_trade_ui)
        self.auto_trade_timer.start(2000)

        self._update_market_status()

        # =========================================================================
        # PROFESSIONAL DARK THEME
    # =========================================================================

    def _apply_professional_style(self):
        """Apply professional dark theme"""
        self.setStyleSheet("""
            QMainWindow {
                background: #0d1117;
            }

            QMenuBar {
                background: #161b22;
                color: #c9d1d9;
                border-bottom: 1px solid #30363d;
            }
            QMenuBar::item:selected {
                background: #21262d;
            }
            QMenu {
                background: #161b22;
                color: #c9d1d9;
                border: 1px solid #30363d;
            }
            QMenu::item:selected {
                background: #21262d;
            }

            QToolBar {
                background: #161b22;
                border-bottom: 1px solid #30363d;
                spacing: 10px;
                padding: 5px;
            }

            QGroupBox {
                font-weight: bold;
                font-size: 12px;
                border: 1px solid #30363d;
                border-radius: 8px;
                margin-top: 12px;
                padding-top: 12px;
                color: #58a6ff;
                background: #0d1117;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 5px;
            }

            QLabel {
                color: #c9d1d9;
                font-size: 12px;
            }

            QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
                padding: 8px;
                border: 1px solid #30363d;
                border-radius: 6px;
                background: #21262d;
                color: #c9d1d9;
                font-size: 12px;
            }
            QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus {
                border-color: #58a6ff;
            }

            QTableWidget {
                background: #0d1117;
                color: #c9d1d9;
                border: none;
                gridline-color: #30363d;
                selection-background-color: #21262d;
            }
            QTableWidget::item {
                padding: 8px;
            }
            QHeaderView::section {
                background: #21262d;
                color: #58a6ff;
                padding: 10px;
                border: none;
                font-weight: bold;
            }

            QTabWidget::pane {
                border: 1px solid #30363d;
                background: #0d1117;
                border-radius: 8px;
            }
            QTabBar::tab {
                background: #161b22;
                color: #8b949e;
                padding: 10px 20px;
                border-top-left-radius: 6px;
                border-top-right-radius: 6px;
                margin-right: 2px;
            }
            QTabBar::tab:selected {
                background: #21262d;
                color: #58a6ff;
            }

            QPushButton {
                background: #21262d;
                color: #c9d1d9;
                border: 1px solid #30363d;
                padding: 8px 16px;
                border-radius: 6px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: #30363d;
                border-color: #58a6ff;
            }
            QPushButton:disabled {
                background: #161b22;
                color: #484f58;
                border-color: #21262d;
            }

            QProgressBar {
                border: none;
                background: #21262d;
                border-radius: 4px;
                text-align: center;
                color: #c9d1d9;
            }
            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #238636, stop:1 #2ea043);
                border-radius: 4px;
            }

            QStatusBar {
                background: #161b22;
                color: #8b949e;
                border-top: 1px solid #30363d;
            }

            QTextEdit {
                background: #0d1117;
                color: #7ee787;
                border: 1px solid #30363d;
                border-radius: 6px;
                font-family: 'Consolas', monospace;
            }

            QScrollBar:vertical {
                background: #161b22;
                width: 10px;
                border-radius: 5px;
            }
            QScrollBar::handle:vertical {
                background: #30363d;
                border-radius: 5px;
                min-height: 30px;
            }
            QScrollBar::handle:vertical:hover {
                background: #484f58;
            }
        """)

    # =========================================================================
    # COMPONENT INITIALIZATION
    # =========================================================================

    def _init_components(self):
        """Initialize trading components"""
        try:
            Predictor = get_predictor_class()

            # Get current interval and horizon from UI
            interval = self.interval_combo.currentText().strip()
            horizon = self.forecast_spin.value()

            self.predictor = Predictor(
                capital=self.capital_spin.value(),
                interval=interval,
                prediction_horizon=horizon
            )

            if self.predictor.ensemble:
                num_models = len(self.predictor.ensemble.models)
                self.model_status.setText(
                    f"✅ Model: Loaded ({num_models} networks)"
                )
                self.model_status.setStyleSheet("color: #4CAF50;")
                self.model_info.setText(
                    f"Interval: {interval}, Horizon: {horizon}"
                )
                self.log("AI model loaded successfully", "success")
            else:
                self.model_status.setText("⚠️ Model: Not trained")
                self.model_status.setStyleSheet("color: #FFD54F;")
                self.model_info.setText(
                    "Train a model to enable predictions"
                )
                self.log(
                    "No trained model found. Please train a model.", "warning"
                )

        except Exception as e:
            log.error(f"Failed to load model: {e}")
            self.log(f"Failed to load model: {e}", "error")
            self.predictor = None
            self.model_status.setText("❌ Model: Error")
            self.model_status.setStyleSheet("color: #F44336;")

        # Initialize auto-trader on executor if available
        self._init_auto_trader()

        self.log("System initialized - Ready for trading", "info")

    def _init_auto_trader(self):
        """Initialize auto-trader on the execution engine."""
        if self.executor and self.predictor:
            try:
                self.executor.init_auto_trader(
                    self.predictor, self.watch_list
                )

                # Wire up callbacks for UI updates
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
            # Executor not connected yet — will init when connected
            pass

    def _on_interval_changed(self, interval: str):
        """Handle interval change - reload model and restart monitor."""
        horizon = self.forecast_spin.value()
        self.model_info.setText(f"Interval: {interval}, Horizon: {horizon}")

        # suggest lookback
        lookback_map = {"1m": 1400, "5m": 600, "15m": 600}
        self.lookback_spin.setValue(lookback_map.get(interval, 300))

        was_monitoring = bool(self.monitor and self.monitor.isRunning())
        if was_monitoring:
            self._stop_monitoring()

        # reload predictor
        if self.predictor:
            try:
                Predictor = get_predictor_class()
                self.predictor = Predictor(
                    capital=self.capital_spin.value(),
                    interval=interval,
                    prediction_horizon=horizon
                )
                if self.predictor.ensemble:
                    self.log(
                        f"Model reloaded for {interval} interval", "info"
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

        interval = self.interval_combo.currentText().strip()
        forecast_bars = self.forecast_spin.value()
        lookback = self.lookback_spin.value()

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
            interval=interval,
            forecast_minutes=forecast_bars,
            lookback_bars=lookback
        )
        self.monitor.signal_detected.connect(self._on_signal_detected)
        self.monitor.price_updated.connect(self._on_price_updated)
        self.monitor.error_occurred.connect(
            lambda e: self.log(f"Monitor: {e}", "warning")
        )
        self.monitor.status_changed.connect(
            lambda s: self.monitor_label.setText(f"📡 {s}")
        )
        self.monitor.start()

        self.monitor_label.setText("📡 Monitoring: ACTIVE")
        self.monitor_label.setStyleSheet(
            "color: #4CAF50; font-weight: bold;"
        )
        self.monitor_action.setText("⏹ Stop Monitoring")

        self.log(
            f"Monitoring started: {interval} interval, "
            f"{forecast_bars} bar forecast",
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
        self.monitor_action.setText("📡 Start Monitoring")
        self.monitor_action.setChecked(False)

        self.log("Real-time monitoring stopped", "info")

    def _on_signal_detected(self, pred):
        """Handle detected trading signal"""
        Signal = get_signal_class()

        # Add to signals table
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
            row, 4, QTableWidgetItem(f"¥{price:.2f}")
        )

        # Action button
        action_btn = QPushButton("Trade")
        action_btn.clicked.connect(lambda: self._quick_trade(pred))
        self.signals_table.setCellWidget(row, 5, action_btn)

        # Keep only last 50 signals
        while self.signals_table.rowCount() > 50:
            self.signals_table.removeRow(
                self.signals_table.rowCount() - 1
            )

        # Notification
        self.log(
            f"🔔 SIGNAL: {signal_text} - {pred.stock_code} @ ¥{price:.2f}",
            "success"
        )

        # Flash the window if minimized
        QApplication.alert(self)

    def _on_price_updated(self, code: str, price: float):
        """
        Handle price update from monitor.
        
        FIXED: No longer calls update_data() which was overwriting candles.
        Instead, updates the current bar's close price so the candle
        reflects the live price.
        """
        code = self._ui_norm(code)
        if not code:
            return

        # Update watchlist table
        for row in range(self.watchlist.rowCount()):
            item = self.watchlist.item(row, 0)
            if item and self._ui_norm(item.text()) == code:
                self.watchlist.setItem(
                    row, 1, QTableWidgetItem(f"¥{price:.2f}")
                )
                break

        # Only process further if this is the current stock
        current_code = self._ui_norm(self.stock_input.text())
        if current_code != code:
            return

        # Update the last bar's close price for live candle display
        arr = self._bars_by_symbol.get(code)
        if arr and len(arr) > 0:
            # Update OHLC of current bar
            arr[-1]["close"] = price
            arr[-1]["high"] = max(arr[-1].get("high", price), price)
            arr[-1]["low"] = min(arr[-1].get("low", price), price)

            # Get prediction
            predicted = []
            if (
                self.current_prediction
                and getattr(self.current_prediction, "stock_code", "") == code
            ):
                predicted = (
                    getattr(self.current_prediction, "predicted_prices", [])
                    or []
                )

            # Update chart with modified bar data
            try:
                if hasattr(self.chart, 'update_chart'):
                    self.chart.update_chart(
                        arr,
                        predicted_prices=predicted,
                        levels=self._get_levels_dict()
                    )
            except Exception as e:
                log.debug(f"Chart price update failed: {e}")

        # =====================================================================
        # THROTTLED FORECAST REFRESH (keep existing logic but simplified)
        # =====================================================================
        
        if not self.predictor:
            return

        now = time.time()
        if (now - self._last_forecast_refresh_ts) < 2.0:
            return
        self._last_forecast_refresh_ts = now

        interval = self.interval_combo.currentText().strip()
        horizon = self.forecast_spin.value()
        lookback = self.lookback_spin.value()

        def do_forecast():
            if hasattr(self.predictor, "get_realtime_forecast_curve"):
                return self.predictor.get_realtime_forecast_curve(
                    stock_code=code,
                    interval=interval,
                    horizon_steps=horizon,
                    lookback_bars=lookback,
                    use_realtime_price=True,
                )
            return None

        # Avoid piling up forecast workers
        w_old = self.workers.get("forecast_refresh")
        if w_old and w_old.isRunning():
            return

        worker = WorkerThread(do_forecast, timeout_seconds=30)
        self.workers["forecast_refresh"] = worker

        def on_done(res):
            try:
                if not res:
                    return
                actual_prices, predicted_prices = res
                
                # Update current_prediction with new forecast
                if (
                    self.current_prediction
                    and self.current_prediction.stock_code == code
                ):
                    self.current_prediction.predicted_prices = predicted_prices
                
                # Refresh chart with new prediction
                arr = self._bars_by_symbol.get(code)
                if arr and hasattr(self.chart, 'update_chart'):
                    self.chart.update_chart(
                        arr,
                        predicted_prices=predicted_prices,
                        levels=self._get_levels_dict()
                    )
            finally:
                self.workers.pop("forecast_refresh", None)

        worker.finished.connect(on_done)
        worker.error.connect(
            lambda e: self.workers.pop("forecast_refresh", None)
        )
        worker.start()

    def _get_levels_dict(self) -> Optional[Dict[str, float]]:
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

    def _quick_trade(self, pred):
        """Quick trade from signal"""
        self.stock_input.setText(pred.stock_code)
        self._analyze_stock()

    # =========================================================================
    # WATCHLIST
    # =========================================================================

    def _update_watchlist(self):
        """Update watchlist display"""
        current_count = self.watchlist.rowCount()

        if current_count != len(self.watch_list):
            self.watchlist.setRowCount(len(self.watch_list))

        for row, code in enumerate(self.watch_list):
            current_code = self.watchlist.item(row, 0)
            if current_code is None or current_code.text() != code:
                self.watchlist.setItem(row, 0, QTableWidgetItem(code))

            for col in range(1, 4):
                if self.watchlist.item(row, col) is None:
                    self.watchlist.setItem(
                        row, col, QTableWidgetItem("--")
                    )

    def _on_watchlist_click(self, row, col):
        """Handle watchlist double-click with bounds check"""
        if row < 0 or row >= self.watchlist.rowCount():
            return
        item = self.watchlist.item(row, 0)
        if item:
            self.stock_input.setText(item.text())
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
            self._update_watchlist()
            self.log(f"Added {normalized} to watchlist", "info")

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
    # ANALYSIS
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

        interval = self.interval_combo.currentText().strip()
        forecast_bars = self.forecast_spin.value()
        lookback = self.lookback_spin.value()

        self.analyze_action.setEnabled(False)

        if hasattr(self.signal_panel, 'reset'):
            self.signal_panel.reset()

        self.status_label.setText(f"Analyzing {normalized}...")
        self.progress.setRange(0, 0)
        self.progress.show()

        def analyze():
            return self.predictor.predict(
                normalized,
                use_realtime_price=True,
                interval=interval,
                forecast_minutes=forecast_bars,
                lookback_bars=lookback,
                skip_cache=True
            )

        worker = WorkerThread(analyze, timeout_seconds=120)
        worker.finished.connect(self._on_analysis_done)
        worker.error.connect(self._on_analysis_error)
        self.workers["analyze"] = worker
        worker.start()

    def _on_analysis_done(self, pred):
        """Handle analysis completion — also triggers news fetch"""
        self.analyze_action.setEnabled(True)
        self.progress.hide()
        self.status_label.setText("Ready")

        self.current_prediction = pred

        # Update signal panel
        if hasattr(self.signal_panel, 'update_prediction'):
            self.signal_panel.update_prediction(pred)

        # Update chart
        if hasattr(self.chart, 'update_data'):
            levels = self._get_levels_dict()
            price_history = getattr(pred, 'price_history', [])
            predicted_prices = getattr(pred, 'predicted_prices', [])
            try:
                self.chart.update_data(
                    price_history, predicted_prices, levels
                )
            except Exception as e:
                log.debug(f"Chart update failed: {e}")

        # Update details (with news sentiment)
        self._update_details(pred)

        # Fetch news for this stock
        if (
            hasattr(self, 'news_panel')
            and hasattr(self.news_panel, 'set_stock')
        ):
            try:
                self.news_panel.set_stock(pred.stock_code)
            except Exception as e:
                log.debug(f"News fetch for {pred.stock_code}: {e}")

        # Add to history
        self._add_to_history(pred)

        try:
            self._ensure_feed_subscription(pred.stock_code)
        except Exception:
            pass

        # Enable buttons ONLY in manual mode
        Signal = get_signal_class()
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
        self.log(
            f"Analysis complete: {pred.stock_code} - "
            f"{signal_text} ({conf:.0%})",
            "success"
        )

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
        Signal = get_signal_class()

        signal_colors = {
            Signal.STRONG_BUY: "#2ea043",
            Signal.BUY: "#3fb950",
            Signal.HOLD: "#d29922",
            Signal.SELL: "#f85149",
            Signal.STRONG_SELL: "#da3633",
        }

        signal = getattr(pred, 'signal', Signal.HOLD)
        color = signal_colors.get(signal, "#c9d1d9")
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
        rsi = safe_get(pred, 'rsi', 50)
        macd_signal = safe_get(pred, 'macd_signal', 'N/A')
        trend = safe_get(pred, 'trend', 'N/A')
        levels = getattr(pred, 'levels', None)
        position = getattr(pred, 'position', None)
        reasons = getattr(pred, 'reasons', [])
        warnings = getattr(pred, 'warnings', [])

        # Get news sentiment
        news_html = ""
        try:
            from data.news import get_news_aggregator
            from core.network import get_network_env

            env = get_network_env()
            if env.is_china_direct or env.tencent_ok:
                agg = get_news_aggregator()
                sentiment = agg.get_sentiment_summary(pred.stock_code)

                if sentiment and sentiment.get('total', 0) > 0:
                    sent_score = sentiment['overall_sentiment']
                    sent_label = sentiment['label']

                    if sent_label == "positive":
                        sent_color = "#3fb950"
                        sent_emoji = "📈"
                    elif sent_label == "negative":
                        sent_color = "#f85149"
                        sent_emoji = "📉"
                    else:
                        sent_color = "#d29922"
                        sent_emoji = "➡️"

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

                    # Show top headlines
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
                                f'📈 {n["title"]}</span><br/>'
                            )
                        for n in top_neg[:2]:
                            news_html += (
                                f'<span class="negative">'
                                f'📉 {n["title"]}</span><br/>'
                            )
                        news_html += '</div>'
        except Exception as e:
            log.debug(f"News sentiment fetch: {e}")

        html = f"""
        <style>
            body {{ color: #c9d1d9; font-family: Consolas; }}
            .signal {{
                color: {color}; font-size: 18px; font-weight: bold;
            }}
            .section {{ margin: 10px 0; }}
            .label {{ color: #8b949e; }}
            .positive {{ color: #3fb950; }}
            .negative {{ color: #f85149; }}
            .neutral {{ color: #d29922; }}
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
                Entry: ¥{entry:.2f} |
                Stop: ¥{stop_loss:.2f} ({stop_loss_pct:+.1f}%)<br/>
                Target 1: ¥{target_1:.2f} ({target_1_pct:+.1f}%) |
                Target 2: ¥{target_2:.2f} ({target_2_pct:+.1f}%)
            </div>
            """

        if position:
            shares = safe_get(position, 'shares', 0)
            value = safe_get(position, 'value', 0)
            risk_amount = safe_get(position, 'risk_amount', 0)
            html += f"""
            <div class="section">
                <span class="label">Position:</span>
                {shares:,} shares | ¥{value:,.2f} |
                Risk: ¥{risk_amount:,.2f}
            </div>
            """

        if reasons:
            html += (
                '<div class="section">'
                '<span class="label">Analysis:</span><br/>'
            )
            for reason in reasons[:5]:
                html += f"• {reason}<br/>"
            html += "</div>"

        if warnings:
            html += (
                '<div class="section">'
                '<span class="negative">⚠️ Warnings:</span><br/>'
            )
            for warning in warnings:
                html += f"• {warning}<br/>"
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
        signal_item.setForeground(QColor("#58a6ff"))
        self.history_table.setItem(row, 2, signal_item)

        prob_up = getattr(pred, 'prob_up', 0)
        self.history_table.setItem(
            row, 3, QTableWidgetItem(f"{prob_up:.0%}")
        )

        confidence = getattr(pred, 'confidence', 0)
        self.history_table.setItem(
            row, 4, QTableWidgetItem(f"{confidence:.0%}")
        )
        self.history_table.setItem(row, 5, QTableWidgetItem("--"))

        while self.history_table.rowCount() > 100:
            self.history_table.removeRow(
                self.history_table.rowCount() - 1
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
        worker.finished.connect(self._on_scan_done)
        worker.error.connect(
            lambda e: (
                self.log(f"Scan failed: {e}", "error"),
                self.progress.hide()
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
                f"  📈 {pred.stock_code} {name}: "
                f"{signal_text} (confidence: {conf:.0%})",
                "info"
            )

        # Analyze top pick
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
    # TRADING
    # =========================================================================

    def _toggle_trading(self):
        """Toggle trading connection"""
        if self.executor is None:
            self._connect_trading()
        else:
            self._disconnect_trading()

    def _connect_trading(self):
        """Connect to trading system"""
        mode = (
            TradingMode.SIMULATION
            if self.mode_combo.currentIndex() == 0
            else TradingMode.LIVE
        )

        if mode == TradingMode.LIVE:
            reply = QMessageBox.warning(
                self, "⚠️ Live Trading Warning",
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
            ExecutionEngine = get_execution_engine()
            self.executor = ExecutionEngine(mode)
            self.executor.on_fill = self._on_order_filled
            self.executor.on_reject = self._on_order_rejected

            if self.executor.start():
                self.connection_status.setText("● Connected")
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

        self.connection_status.setText("● Disconnected")
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

        reply = QMessageBox.question(
            self, "Confirm Buy Order",
            f"<b>Buy {pred.stock_code} - {stock_name}</b><br><br>"
            f"Quantity: {shares:,} shares<br>"
            f"Price: ¥{entry:.2f}<br>"
            f"Value: ¥{value:,.2f}<br>"
            f"Stop Loss: ¥{stop_loss:.2f}<br>"
            f"Target: ¥{target_2:.2f}",
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
            positions = self.executor.get_positions()
            position = positions.get(pred.stock_code)

            if not position:
                self.log("No position to sell", "warning")
                return

            available_qty = getattr(position, 'available_qty', 0)
            current_price = getattr(position, 'current_price', 0)
            stock_name = getattr(pred, 'stock_name', '')

            reply = QMessageBox.question(
                self, "Confirm Sell Order",
                f"<b>Sell {pred.stock_code} - {stock_name}</b><br><br>"
                f"Available: {available_qty:,} shares<br>"
                f"Current Price: ¥{current_price:.2f}",
                QMessageBox.StandardButton.Yes
                | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )

            if reply == QMessageBox.StandardButton.Yes:
                from core.types import TradeSignal, OrderSide

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
            f"✅ Order Filled: {side} {qty} {order.symbol} @ ¥{price:.2f}",
            "success"
        )
        self._refresh_portfolio()

    def _on_order_rejected(self, order, reason):
        """Handle order rejection"""
        self.log(
            f"❌ Order Rejected: {order.symbol} - {reason}", "error"
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

            self.account_labels['equity'].setText(f"¥{equity:,.2f}")
            self.account_labels['cash'].setText(f"¥{available:,.2f}")
            self.account_labels['positions'].setText(
                f"¥{market_value:,.2f}"
            )

            pnl_color = "#3fb950" if total_pnl >= 0 else "#f85149"
            self.account_labels['pnl'].setText(f"¥{total_pnl:,.2f}")
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
    # TRAINING
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
        except Exception as e:
            self.log(f"Training dialog failed: {e}", "error")
            return

        # Reload model after training
        self._init_components()

    def _show_auto_learn(self):
        """Show auto-learning dialog"""
        try:
            from .auto_learn_dialog import show_auto_learn_dialog
            show_auto_learn_dialog(self)
        except ImportError:
            self.log("Auto-learn dialog not available", "error")
            return

        # Reload model after learning
        self._init_components()

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
            "<p><b>⚠️ Risk Warning:</b></p>"
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

        # Safety check for AUTO mode
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

            # Live trading confirmation
            if (
                self.executor
                and self.executor.mode == TradingMode.LIVE
                and CONFIG.auto_trade.confirm_live_auto_trade
            ):
                reply = QMessageBox.warning(
                    self, "⚠️ LIVE Auto-Trading",
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

            # Standard confirmation for auto mode
            reply = QMessageBox.question(
                self, "Enable Auto-Trading",
                "Enable fully automatic trading?\n\n"
                f"• Min confidence: {CONFIG.auto_trade.min_confidence:.0%}\n"
                f"• Max trades/day: {CONFIG.auto_trade.max_trades_per_day}\n"
                f"• Max order value: ¥{CONFIG.auto_trade.max_auto_order_value:,.0f}\n"
                f"• Max auto positions: {CONFIG.auto_trade.max_auto_positions}\n\n"
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
        # Update CONFIG
        CONFIG.auto_trade.enabled = (mode != AutoTradeMode.MANUAL)

        # Update executor auto-trader
        if self.executor and self.executor.auto_trader:
            self.executor.set_auto_mode(mode)

            # Update watchlist on auto-trader
            self.executor.auto_trader.update_watchlist(self.watch_list)

            # Update predictor reference
            if self.predictor:
                self.executor.auto_trader.update_predictor(self.predictor)
        elif mode != AutoTradeMode.MANUAL:
            # Need to initialize auto-trader first
            self._init_auto_trader()
            if self.executor and self.executor.auto_trader:
                self.executor.set_auto_mode(mode)

        # Update UI
        self._update_auto_trade_status_label(mode)

        # Enable/disable manual trade buttons based on mode
        if mode == AutoTradeMode.AUTO:
            self.buy_btn.setEnabled(False)
            self.sell_btn.setEnabled(False)
            self.auto_pause_btn.setEnabled(True)
            self.log("🤖 AUTO-TRADE enabled — AI will trade automatically", "success")
        elif mode == AutoTradeMode.SEMI_AUTO:
            self.auto_pause_btn.setEnabled(True)
            self.auto_approve_all_btn.setEnabled(True)
            self.auto_reject_all_btn.setEnabled(True)
            self.log(
                "🤖 SEMI-AUTO enabled — AI will suggest, you approve",
                "success"
            )
        else:
            self.auto_pause_btn.setEnabled(False)
            self.auto_approve_all_btn.setEnabled(False)
            self.auto_reject_all_btn.setEnabled(False)
            self.log("✋ MANUAL mode — you control all trades", "info")

    def _update_auto_trade_status_label(self, mode: AutoTradeMode):
        """Update the toolbar status label."""
        if mode == AutoTradeMode.AUTO:
            self.auto_trade_status_label.setText("  🟢 AUTO  ")
            self.auto_trade_status_label.setStyleSheet(
                "color: #4CAF50; font-weight: bold; padding: 0 8px;"
            )
        elif mode == AutoTradeMode.SEMI_AUTO:
            self.auto_trade_status_label.setText("  🟡 SEMI  ")
            self.auto_trade_status_label.setStyleSheet(
                "color: #FFD54F; font-weight: bold; padding: 0 8px;"
            )
        else:
            self.auto_trade_status_label.setText("  ⚪ Manual  ")
            self.auto_trade_status_label.setStyleSheet(
                "color: #8b949e; font-weight: bold; padding: 0 8px;"
            )

    def _toggle_auto_pause(self):
        """Pause/resume auto-trading."""
        if not self.executor or not self.executor.auto_trader:
            return

        state = self.executor.auto_trader.get_state()
        if state.is_safety_paused or state.is_paused:
            self.executor.auto_trader.resume()
            self.auto_pause_btn.setText("⏸ Pause Auto")
            self.log("Auto-trading resumed", "info")
        else:
            self.executor.auto_trader.pause("Manually paused by user")
            self.auto_pause_btn.setText("▶ Resume Auto")
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
        from PyQt6.QtWidgets import (
            QDialog, QFormLayout, QDialogButtonBox
        )

        dialog = QDialog(self)
        dialog.setWindowTitle("Auto-Trade Settings")
        dialog.setMinimumWidth(500)

        layout = QVBoxLayout(dialog)

        # Settings group
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
        max_order_spin.setPrefix("¥ ")
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

        # Signal type checkboxes
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

        # Buttons
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
        # Use QTimer to marshal to UI thread
        QTimer.singleShot(0, lambda: self._on_auto_trade_action(action))

    def _on_auto_trade_action(self, action: AutoTradeAction):
        """Handle auto-trade action on UI thread."""
        # Add to actions table
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

        # Cap table rows
        while self.auto_actions_table.rowCount() > 100:
            self.auto_actions_table.removeRow(
                self.auto_actions_table.rowCount() - 1
            )

        # Log it
        if action.decision == "EXECUTED":
            self.log(
                f"🤖 AUTO-TRADE: {action.side.upper()} "
                f"{action.quantity} {action.stock_code} "
                f"@ ¥{action.price:.2f} ({action.confidence:.0%})",
                "success"
            )
        elif action.decision == "SKIPPED":
            self.log(
                f"🤖 Skipped {action.stock_code}: {action.skip_reason}",
                "info"
            )

        # Flash window for executed trades
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
            row, 4, QTableWidgetItem(f"¥{action.price:.2f}")
        )

        # Approve/Reject buttons
        btn_widget = QWidget()
        btn_layout = QHBoxLayout(btn_widget)
        btn_layout.setContentsMargins(2, 2, 2, 2)

        approve_btn = QPushButton("✅")
        approve_btn.setFixedWidth(30)
        approve_btn.setToolTip("Approve this trade")
        action_id = action.id

        def do_approve():
            if self.executor and self.executor.auto_trader:
                self.executor.auto_trader.approve_pending(action_id)
                self._refresh_pending_table()

        approve_btn.clicked.connect(do_approve)

        reject_btn = QPushButton("❌")
        reject_btn.setFixedWidth(30)
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
            f"🔔 PENDING: {action.signal_type} {action.stock_code} "
            f"@ ¥{action.price:.2f} — approve or reject",
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
        if not self.executor or not self.executor.auto_trader:
            # Show defaults
            self.auto_trade_labels.get('mode', QLabel()).setText(
                self._auto_trade_mode.value.upper()
            )
            self.auto_trade_labels.get('trades', QLabel()).setText("0")
            self.auto_trade_labels.get('pnl', QLabel()).setText("--")
            self.auto_trade_labels.get('status', QLabel()).setText("--")
            return

        state = self.executor.auto_trader.get_state()

        # Mode
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
                color = "#8b949e"
            mode_label.setStyleSheet(
                f"color: {color}; font-size: 16px; font-weight: bold;"
            )

        # Trades
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
            pnl_color = "#3fb950" if pnl >= 0 else "#f85149"
            pnl_label.setText(f"¥{pnl:+,.2f}")
            pnl_label.setStyleSheet(
                f"color: {pnl_color}; font-size: 16px; font-weight: bold;"
            )

        # Status
        status_label = self.auto_trade_labels.get('status')
        if status_label:
            if state.is_safety_paused:
                status_label.setText(f"⏸ {state.pause_reason}")
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
                status_label.setText(f"🟢 Running{last_scan}")
                status_label.setStyleSheet(
                    "color: #4CAF50; font-size: 14px; font-weight: bold;"
                )
            else:
                status_label.setText("⚪ Idle")
                status_label.setStyleSheet(
                    "color: #8b949e; font-size: 14px;"
                )

        # Update pause button text
        if state.is_safety_paused or state.is_paused:
            self.auto_pause_btn.setText("▶ Resume Auto")
        else:
            self.auto_pause_btn.setText("⏸ Pause Auto")

        # Pending approvals count
        pending_count = len(state.pending_approvals)
        if pending_count > 0:
            self.auto_approve_all_btn.setText(
                f"✅ Approve All ({pending_count})"
            )
            self.auto_approve_all_btn.setEnabled(True)
            self.auto_reject_all_btn.setEnabled(True)
        else:
            self.auto_approve_all_btn.setText("✅ Approve All")
            if self._auto_trade_mode != AutoTradeMode.SEMI_AUTO:
                self.auto_approve_all_btn.setEnabled(False)
                self.auto_reject_all_btn.setEnabled(False)

    # =========================================================================
    # UTILITIES
    # =========================================================================

    def _update_clock(self):
        """Update clock"""
        self.time_label.setText(
            datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )

    def _update_market_status(self):
        """Update market status"""
        is_open = CONFIG.is_market_open()

        if is_open:
            self.market_label.setText("🟢 Market Open")
            self.market_label.setStyleSheet(
                "color: #3fb950; font-weight: bold;"
            )
        else:
            self.market_label.setText("🔴 Market Closed")
            self.market_label.setStyleSheet("color: #f85149;")

    def log(self, message: str, level: str = "info"):
        """Log message to UI"""
        timestamp = datetime.now().strftime("%H:%M:%S")

        colors = {
            "info": "#c9d1d9",
            "success": "#3fb950",
            "warning": "#d29922",
            "error": "#f85149",
        }
        color = colors.get(level, "#c9d1d9")

        formatted = (
            f'<span style="color: #888;">[{timestamp}]</span> '
            f'<span style="color: {color};">{message}</span>'
        )

        if hasattr(self.log_widget, 'log'):
            self.log_widget.log(message, level)
        elif hasattr(self.log_widget, 'append'):
            self.log_widget.append(formatted)

        # Also log to file
        log.info(message)

    # =========================================================================
    # CLOSE EVENT (FIX #3 - calls super)
    # =========================================================================

    def closeEvent(self, event):
        """Handle window close safely."""
        # Stop monitoring
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

        # Stop workers
        for name, worker in list(self.workers.items()):
            try:
                worker.cancel()
                worker.quit()
                worker.wait(2000)
            except Exception:
                pass
        self.workers.clear()

        # Disconnect trading
        if self.executor:
            try:
                self.executor.stop()
            except Exception:
                pass
            self.executor = None

        # Stop timers
        for timer_name in (
            "clock_timer", "market_timer",
            "portfolio_timer", "watchlist_timer",
            "auto_trade_timer"
        ):
            timer = getattr(self, timer_name, None)
            try:
                if timer:
                    timer.stop()
            except Exception:
                pass

        # Save state
        try:
            self._save_state()
        except Exception:
            pass

        event.accept()
        super().closeEvent(event)

    # =========================================================================
    # STATE PERSISTENCE
    # =========================================================================

    def _save_state(self):
        """Save application state for next session"""
        try:
            import json
            state = {
                'watch_list': self.watch_list,
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
                with open(state_path, 'r') as f:
                    state = json.load(f)

                if 'watch_list' in state:
                    loaded = state['watch_list']
                    # Validate and cap watchlist
                    validated = [
                        c for c in loaded
                        if _validate_stock_code(c)
                    ][:self.MAX_WATCHLIST_SIZE]
                    if validated:
                        self.watch_list = validated
                if 'interval' in state:
                    self.interval_combo.setCurrentText(state['interval'])
                if 'forecast' in state:
                    self.forecast_spin.setValue(state['forecast'])
                if 'lookback' in state:
                    self.lookback_spin.setValue(state['lookback'])
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


# =============================================================================
# APPLICATION ENTRY POINT
# =============================================================================

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

    sys.exit(app.exec())


if __name__ == "__main__":
    run_app()