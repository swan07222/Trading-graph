# ui/app.py
"""
AI Stock Trading System - Professional Desktop Application
Real-time trading signals with custom AI model

FIXED:
- Proper integration with updated processor.py (RealtimePredictor)
- Correct interval/horizon parameter passing
- Real-time feed subscription for dealing bars
- Robust error handling and graceful degradation
"""
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
    QDockWidget, QSystemTrayIcon, QMenu
)
from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal, QSize
from PyQt6.QtGui import QFont, QColor, QIcon, QAction, QPalette

from config.settings import CONFIG, TradingMode
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

def get_realtime_predictor():
    """Lazy import RealtimePredictor from processor"""
    from data.processor import RealtimePredictor
    return RealtimePredictor


def get_news_panel():
    """Lazy import NewsPanel"""
    from ui.news_widget import NewsPanel
    return NewsPanel


# =============================================================================
# REAL-TIME MONITORING THREAD
# =============================================================================

class RealTimeMonitor(QThread):
    """
    Real-time market monitoring thread.
    Continuously checks for trading signals using the updated processor.
    
    Features:
    - Uses RealtimePredictor for fast inference
    - Supports multiple intervals (1m, 5m, 1d, etc.)
    - Graceful error handling with exponential backoff
    - Thread-safe signal emission
    """
    signal_detected = pyqtSignal(object)  # Prediction
    price_updated = pyqtSignal(str, float)  # code, price
    error_occurred = pyqtSignal(str)
    status_changed = pyqtSignal(str)  # status message
    
    def __init__(
        self, 
        predictor: Any,  # Predictor instance
        watch_list: List[str], 
        interval: str = "1m", 
        forecast_minutes: int = 30, 
        lookback_bars: int = 1400
    ):
        super().__init__()
        self.predictor = predictor
        self.watch_list = list(watch_list)
        self.running = False
        self._stop_event = threading.Event()
        
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
                    and hasattr(p, 'confidence') and p.confidence >= CONFIG.MIN_CONFIDENCE
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
                            lookback_bars=self.lookback_bars
                        )
                        self.signal_detected.emit(full)
                    except Exception as e:
                        log.warning(f"Full prediction failed for {p.stock_code}: {e}")
                
                # Reset backoff on success
                self._backoff = 1
                self.status_changed.emit(f"Scanned {len(preds)} stocks, {len(strong)} signals")
                
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
    """Generic worker thread for background tasks"""
    finished = pyqtSignal(object)
    error = pyqtSignal(str)
    progress = pyqtSignal(int, str)
    
    def __init__(self, func, *args, **kwargs):
        super().__init__()
        self.func = func
        self.args = args
        self.kwargs = kwargs
        self._cancelled = False
    
    def run(self):
        try:
            if self._cancelled:
                return
            result = self.func(*self.args, **self.kwargs)
            if not self._cancelled:
                self.finished.emit(result)
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

        # Real-time state
        self._last_forecast_refresh_ts: float = 0.0
        self._live_price_series: Dict[str, List[float]] = {}

        # Setup UI
        self._setup_menubar()
        self._setup_toolbar()
        self._setup_ui()
        self._setup_statusbar()
        self._setup_timers()
        self._apply_professional_style()

        # Load state BEFORE initializing model
        try:
            self._load_state()
            self._update_watchlist()
        except Exception:
            pass

        # Initialize components AFTER state load
        QTimer.singleShot(0, self._init_components)
    
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
    
    def _setup_toolbar(self):
        """Setup professional toolbar"""
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
        
        # Spacer
        spacer = QWidget()
        from PyQt6.QtWidgets import QSizePolicy
        spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        toolbar.addWidget(spacer)
        
        # Stock input in toolbar
        toolbar.addWidget(QLabel("  Stock: "))
        self.stock_input = QLineEdit()
        self.stock_input.setPlaceholderText("Enter code (e.g., 600519)")
        self.stock_input.setFixedWidth(150)
        self.stock_input.returnPressed.connect(self._analyze_stock)
        toolbar.addWidget(self.stock_input)
    
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
        self.watchlist.setHorizontalHeaderLabels(["Code", "Price", "Change", "Signal"])
        self.watchlist.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
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
        self.interval_combo.currentTextChanged.connect(self._on_interval_changed)
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
        self.connection_status.setStyleSheet("color: #FF5252; font-weight: bold;")
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
    
    # ui/app.py — Replace _create_right_panel method

    def _create_right_panel(self) -> QWidget:
        """Create right panel with portfolio, news, and orders"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setSpacing(10)

        tabs = QTabWidget()

        # Portfolio Tab (unchanged)
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
            value.setStyleSheet("color: #00E5FF; font-size: 18px; font-weight: bold;")
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
                ["Code", "Qty", "Price", "Value", "P&L"])
        portfolio_layout.addWidget(self.positions_table)

        tabs.addTab(portfolio_tab, "💼 Portfolio")

        # ===== NEW: News Tab =====
        news_tab = QWidget()
        news_layout = QVBoxLayout(news_tab)
        try:
            NewsPanel = get_news_panel()  # Now calls module-level function (Fix 1)
            self.news_panel = NewsPanel()
            news_layout.addWidget(self.news_panel)
        except Exception as e:
            log.warning(f"News panel not available: {e}")
            self.news_panel = QLabel("News panel unavailable")
            news_layout.addWidget(self.news_panel)
        tabs.addTab(news_tab, "📰 News & Policy")

        # Live Signals Tab (unchanged)
        signals_tab = QWidget()
        signals_layout = QVBoxLayout(signals_tab)
        self.signals_table = QTableWidget()
        self.signals_table.setColumnCount(6)
        self.signals_table.setHorizontalHeaderLabels([
            "Time", "Code", "Signal", "Confidence", "Price", "Action"
        ])
        self.signals_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch)
        signals_layout.addWidget(self.signals_table)
        tabs.addTab(signals_tab, "📡 Live Signals")

        # History Tab (unchanged)
        history_tab = QWidget()
        history_layout = QVBoxLayout(history_tab)
        self.history_table = QTableWidget()
        self.history_table.setColumnCount(6)
        self.history_table.setHorizontalHeaderLabels([
            "Time", "Code", "Signal", "Prob UP", "Confidence", "Result"
        ])
        self.history_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch)
        history_layout.addWidget(self.history_table)
        tabs.addTab(history_tab, "📜 History")

        layout.addWidget(tabs)

        # Log (unchanged)
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

        # Action Buttons (unchanged)
        action_frame = QFrame()
        action_frame.setStyleSheet("""
            QFrame { background: #1a1a3e; border-radius: 8px; padding: 10px; }
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
        
        self._update_market_status()
    
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
                self.model_status.setText(f"✅ Model: Loaded ({num_models} networks)")
                self.model_status.setStyleSheet("color: #4CAF50;")
                self.model_info.setText(f"Interval: {interval}, Horizon: {horizon}")
                self.log("AI model loaded successfully", "success")
            else:
                self.model_status.setText("⚠️ Model: Not trained")
                self.model_status.setStyleSheet("color: #FFD54F;")
                self.model_info.setText("Train a model to enable predictions")
                self.log("No trained model found. Please train a model.", "warning")
                
        except Exception as e:
            log.error(f"Failed to load model: {e}")
            self.log(f"Failed to load model: {e}", "error")
            self.predictor = None
            self.model_status.setText("❌ Model: Error")
            self.model_status.setStyleSheet("color: #F44336;")
        
        self.log("System initialized - Ready for trading", "info")
    
    def _on_interval_changed(self, interval: str):
        """Handle interval change - reload model and update monitor if active"""
        horizon = self.forecast_spin.value()
        self.model_info.setText(f"Interval: {interval}, Horizon: {horizon}")

        # Update lookback bars suggestion
        if interval == "1m":
            self.lookback_spin.setValue(1400)
        elif interval in ("5m", "15m"):
            self.lookback_spin.setValue(600)
        else:
            self.lookback_spin.setValue(300)

        # Reload model for new interval
        if self.predictor:
            try:
                Predictor = get_predictor_class()
                self.predictor = Predictor(
                    capital=self.capital_spin.value(),
                    interval=interval,
                    prediction_horizon=horizon
                )
                if self.predictor.ensemble:
                    self.log(f"Model reloaded for {interval} interval", "info")
            except Exception as e:
                self.log(f"Model reload failed: {e}", "warning")

        # Update running monitor with new settings
        if self.monitor and self.monitor.isRunning():
            lookback = self.lookback_spin.value()
            self.monitor.update_config(
                interval=interval,
                forecast_minutes=horizon,
                lookback_bars=lookback
            )
            self.log(f"Monitor updated: {interval}, {horizon} bars, lookback={lookback}", "info")
        
    # ==================== Real-time Monitoring ====================
    
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
            self.log(f"Subscribed to feeds for {len(self.watch_list)} stocks", "info")
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
        self.monitor.error_occurred.connect(lambda e: self.log(f"Monitor: {e}", "warning"))
        self.monitor.status_changed.connect(lambda s: self.monitor_label.setText(f"📡 {s}"))
        self.monitor.start()

        self.monitor_label.setText("📡 Monitoring: ACTIVE")
        self.monitor_label.setStyleSheet("color: #4CAF50; font-weight: bold;")
        self.monitor_action.setText("⏹ Stop Monitoring")

        self.log(f"Monitoring started: {interval} interval, {forecast_bars} bar forecast", "success")
    
    def _stop_monitoring(self):
        """Stop real-time monitoring"""
        if self.monitor:
            self.monitor.stop()
            self.monitor.wait(3000)
            self.monitor = None
        
        self.monitor_label.setText("Monitoring: OFF")
        self.monitor_label.setStyleSheet("color: #888;")
        self.monitor_action.setText("📡 Start Monitoring")
        
        self.log("Real-time monitoring stopped", "info")
    
    def _on_signal_detected(self, pred):
        """Handle detected trading signal"""
        Signal = get_signal_class()
        
        # Add to signals table
        row = 0
        self.signals_table.insertRow(row)
        
        self.signals_table.setItem(row, 0, QTableWidgetItem(
            pred.timestamp.strftime("%H:%M:%S") if hasattr(pred, 'timestamp') else "--"
        ))
        
        stock_text = f"{pred.stock_code}"
        if hasattr(pred, 'stock_name') and pred.stock_name:
            stock_text += f" - {pred.stock_name}"
        self.signals_table.setItem(row, 1, QTableWidgetItem(stock_text))
        
        signal_text = pred.signal.value if hasattr(pred.signal, 'value') else str(pred.signal)
        signal_item = QTableWidgetItem(signal_text)
        
        if hasattr(pred, 'signal') and pred.signal in [Signal.STRONG_BUY, Signal.BUY]:
            signal_item.setForeground(QColor("#4CAF50"))
        else:
            signal_item.setForeground(QColor("#F44336"))
        self.signals_table.setItem(row, 2, signal_item)
        
        conf = pred.confidence if hasattr(pred, 'confidence') else 0
        self.signals_table.setItem(row, 3, QTableWidgetItem(f"{conf:.0%}"))
        
        price = pred.current_price if hasattr(pred, 'current_price') else 0
        self.signals_table.setItem(row, 4, QTableWidgetItem(f"¥{price:.2f}"))
        
        # Action button
        action_btn = QPushButton("Trade")
        action_btn.clicked.connect(lambda: self._quick_trade(pred))
        self.signals_table.setCellWidget(row, 5, action_btn)
        
        # Keep only last 50 signals
        while self.signals_table.rowCount() > 50:
            self.signals_table.removeRow(self.signals_table.rowCount() - 1)
        
        # Notification
        self.log(
            f"🔔 SIGNAL: {signal_text} - {pred.stock_code} @ ¥{price:.2f}",
            "success"
        )
        
        # Flash the window if minimized
        QApplication.alert(self)
    
    def _on_price_updated(self, code: str, price: float):
        """Handle real-time price update (UI-safe; forecast refresh in background)."""
        # Watchlist cell update
        for row in range(self.watchlist.rowCount()):
            item = self.watchlist.item(row, 0)
            if item and item.text() == code:
                self.watchlist.setItem(row, 1, QTableWidgetItem(f"¥{price:.2f}"))
                break

        if not hasattr(self.chart, "update_data"):
            return

        current_code = self.stock_input.text().strip()
        if not current_code or current_code != code:
            return
        if not self.predictor:
            return

        # Fast chart update (no inference)
        if self.current_prediction and hasattr(self.current_prediction, "price_history"):
            series = self._live_price_series.get(code) or list(self.current_prediction.price_history or [])
            series = (series[-180:] + [float(price)])[-180:]
            self._live_price_series[code] = series
            predicted = getattr(self.current_prediction, "predicted_prices", [])
            self.chart.update_data(series, predicted, self._get_levels_dict())

        # Throttle expensive refresh
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

        worker = WorkerThread(do_forecast)
        self.workers["forecast_refresh"] = worker

        def on_done(res):
            try:
                if not res:
                    return
                actual_prices, predicted_prices = res
                self.chart.update_data(actual_prices, predicted_prices, self._get_levels_dict())
                if self.current_prediction and self.current_prediction.stock_code == code:
                    self.current_prediction.predicted_prices = predicted_prices
            finally:
                self.workers.pop("forecast_refresh", None)

        worker.finished.connect(on_done)
        worker.error.connect(lambda e: self.workers.pop("forecast_refresh", None))
        worker.start()
    
    def _get_levels_dict(self) -> Optional[Dict[str, float]]:
        """Get trading levels as dict"""
        if not self.current_prediction or not hasattr(self.current_prediction, 'levels'):
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
    
    # ==================== Watchlist ====================
    
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
                    self.watchlist.setItem(row, col, QTableWidgetItem("--"))
    
    def _on_watchlist_click(self, row, col):
        """Handle watchlist double-click"""
        item = self.watchlist.item(row, 0)
        if item:
            self.stock_input.setText(item.text())
            self._analyze_stock()
    
    def _add_to_watchlist(self):
        """Add stock to watchlist"""
        code = self.stock_input.text().strip()
        if code and code not in self.watch_list:
            self.watch_list.append(code)
            self._update_watchlist()
            self.log(f"Added {code} to watchlist", "info")
    
    def _remove_from_watchlist(self):
        """Remove selected stock from watchlist"""
        row = self.watchlist.currentRow()
        if row >= 0:
            item = self.watchlist.item(row, 0)
            if item:
                code = item.text()
                if code in self.watch_list:
                    self.watch_list.remove(code)
                    self._update_watchlist()
                    self.log(f"Removed {code} from watchlist", "info")
    
    # ==================== Analysis ====================
    
    def _analyze_stock(self):
        """Analyze stock"""
        code = self.stock_input.text().strip()
        if not code:
            self.log("Please enter a stock code", "warning")
            return

        if self.predictor is None or self.predictor.ensemble is None:
            self.log("No model loaded. Please train a model first.", "error")
            return

        interval = self.interval_combo.currentText().strip()
        forecast_bars = self.forecast_spin.value()
        lookback = self.lookback_spin.value()

        self.analyze_action.setEnabled(False)
        
        if hasattr(self.signal_panel, 'reset'):
            self.signal_panel.reset()
        
        self.status_label.setText(f"Analyzing {code}...")
        self.progress.setRange(0, 0)
        self.progress.show()

        def analyze():
            return self.predictor.predict(
                code,
                use_realtime_price=True,
                interval=interval,
                forecast_minutes=forecast_bars,
                lookback_bars=lookback
            )

        worker = WorkerThread(analyze)
        worker.finished.connect(self._on_analysis_done)
        worker.error.connect(self._on_analysis_error)
        self.workers["analyze"] = worker
        worker.start()
    
    # ui/app.py — Update _analyze_stock to also fetch news

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
            self.chart.update_data(price_history, predicted_prices, levels)

        # Update details (with news sentiment)
        self._update_details(pred)

        # Fetch news for this stock
        if hasattr(self, 'news_panel') and hasattr(self.news_panel, 'set_stock'):
            try:
                self.news_panel.set_stock(pred.stock_code)
            except Exception as e:
                log.debug(f"News fetch for {pred.stock_code}: {e}")

        # Add to history
        self._add_to_history(pred)

        # Enable buttons
        Signal = get_signal_class()
        if hasattr(pred, 'signal'):
            self.buy_btn.setEnabled(pred.signal in [Signal.STRONG_BUY, Signal.BUY])
            self.sell_btn.setEnabled(pred.signal in [Signal.STRONG_SELL, Signal.SELL])

        signal_text = pred.signal.value if hasattr(pred.signal, 'value') else str(pred.signal)
        conf = getattr(pred, 'confidence', 0)
        self.log(f"Analysis complete: {pred.stock_code} - {signal_text} ({conf:.0%})", "success")

        if 'analyze' in self.workers:
            del self.workers['analyze']
    
    def _on_analysis_error(self, error: str):
        """Handle analysis error"""
        self.analyze_action.setEnabled(True)
        self.progress.hide()
        self.status_label.setText("Ready")
        
        self.log(f"Analysis failed: {error}", "error")
        QMessageBox.warning(self, "Error", f"Analysis failed:\n{error}")
        
        if 'analyze' in self.workers:
            del self.workers['analyze']
    
    # ui/app.py — Update _update_details to include news sentiment

    def _update_details(self, pred):
        """Update analysis details with news sentiment"""
        Signal = get_signal_class()

        signal_colors = {
            Signal.STRONG_BUY: "#2ea043", Signal.BUY: "#3fb950",
            Signal.HOLD: "#d29922", Signal.SELL: "#f85149",
            Signal.STRONG_SELL: "#da3633",
        }

        signal = getattr(pred, 'signal', Signal.HOLD)
        color = signal_colors.get(signal, "#c9d1d9")
        signal_text = signal.value if hasattr(signal, 'value') else str(signal)

        def safe_get(obj, attr, default=0):
            return getattr(obj, attr, default) if hasattr(obj, attr) else default

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
                        news_html += '<div class="section"><span class="label">Key Headlines:</span><br/>'
                        for n in top_pos[:2]:
                            news_html += f'<span class="positive">📈 {n["title"]}</span><br/>'
                        for n in top_neg[:2]:
                            news_html += f'<span class="negative">📉 {n["title"]}</span><br/>'
                        news_html += '</div>'
        except Exception as e:
            log.debug(f"News sentiment fetch: {e}")

        html = f"""
        <style>
            body {{ color: #c9d1d9; font-family: Consolas; }}
            .signal {{ color: {color}; font-size: 18px; font-weight: bold; }}
            .section {{ margin: 10px 0; }}
            .label {{ color: #8b949e; }}
            .positive {{ color: #3fb950; }}
            .negative {{ color: #f85149; }}
            .neutral {{ color: #d29922; }}
        </style>

        <div class="section">
            <span class="label">Signal: </span>
            <span class="signal">{signal_text}</span>
            <span class="label"> | Strength: {signal_strength:.0%}</span>
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
                {shares:,} shares | ¥{value:,.2f} | Risk: ¥{risk_amount:,.2f}
            </div>
            """

        if reasons:
            html += '<div class="section"><span class="label">Analysis:</span><br/>'
            for reason in reasons[:5]:
                html += f"• {reason}<br/>"
            html += "</div>"

        if warnings:
            html += '<div class="section"><span class="negative">⚠️ Warnings:</span><br/>'
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
            timestamp.strftime("%H:%M:%S") if hasattr(timestamp, 'strftime') else "--"
        ))
        self.history_table.setItem(row, 1, QTableWidgetItem(getattr(pred, 'stock_code', '--')))
        
        signal = getattr(pred, 'signal', None)
        signal_text = signal.value if hasattr(signal, 'value') else str(signal)
        signal_item = QTableWidgetItem(signal_text)
        signal_item.setForeground(QColor("#58a6ff"))
        self.history_table.setItem(row, 2, signal_item)
        
        prob_up = getattr(pred, 'prob_up', 0)
        self.history_table.setItem(row, 3, QTableWidgetItem(f"{prob_up:.0%}"))
        
        confidence = getattr(pred, 'confidence', 0)
        self.history_table.setItem(row, 4, QTableWidgetItem(f"{confidence:.0%}"))
        self.history_table.setItem(row, 5, QTableWidgetItem("--"))
        
        while self.history_table.rowCount() > 100:
            self.history_table.removeRow(self.history_table.rowCount() - 1)
    
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
                return self.predictor.get_top_picks(CONFIG.STOCK_POOL, n=10, signal_type="buy")
            return []
        
        worker = WorkerThread(scan)
        worker.finished.connect(self._on_scan_done)
        worker.error.connect(lambda e: (self.log(f"Scan failed: {e}", "error"), self.progress.hide()))
        self.workers['scan'] = worker
        worker.start()
    
    def _on_scan_done(self, picks):
        """Handle scan completion"""
        self.progress.hide()
        
        if not picks:
            self.log("No strong buy signals found", "info")
            return
        
        self.log(f"Found {len(picks)} buy signals:", "success")
        
        Signal = get_signal_class()
        for pred in picks:
            signal_text = pred.signal.value if hasattr(pred.signal, 'value') else str(pred.signal)
            conf = getattr(pred, 'confidence', 0)
            name = getattr(pred, 'stock_name', '')
            self.log(f"  📈 {pred.stock_code} {name}: {signal_text} (confidence: {conf:.0%})", "info")
        
        # Analyze top pick
        if picks:
            self.stock_input.setText(picks[0].stock_code)
            self._analyze_stock()
        
        if 'scan' in self.workers:
            del self.workers['scan']
    
    def _refresh_all(self):
        """Refresh all data"""
        self._update_watchlist()
        self._refresh_portfolio()
        self.log("Refreshed all data", "info")
    
    # ==================== Trading ====================
    
    def _toggle_trading(self):
        """Toggle trading connection"""
        if self.executor is None:
            self._connect_trading()
        else:
            self._disconnect_trading()
    
    def _connect_trading(self):
        """Connect to trading system"""
        mode = TradingMode.SIMULATION if self.mode_combo.currentIndex() == 0 else TradingMode.LIVE
        
        if mode == TradingMode.LIVE:
            reply = QMessageBox.warning(
                self, "⚠️ Live Trading Warning",
                "You are switching to LIVE TRADING mode!\n\n"
                "This will use REAL MONEY.\n\n"
                "Are you absolutely sure?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
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
                self.connection_status.setStyleSheet("color: #4CAF50; font-weight: bold;")
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
                
                self.log(f"Connected to {mode.value} trading", "success")
                self._refresh_portfolio()
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
        self.connection_status.setStyleSheet("color: #FF5252; font-weight: bold;")
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
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            try:
                if hasattr(self.executor, 'submit_from_prediction'):
                    success = self.executor.submit_from_prediction(pred)
                else:
                    success = False
                    
                if success:
                    self.log(f"Buy order submitted: {pred.stock_code}", "info")
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
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
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
                    self.log(f"Sell order submitted: {pred.stock_code}", "info")
                else:
                    self.log("Sell order failed", "error")
        except Exception as e:
            self.log(f"Sell order error: {e}", "error")
    
    def _on_order_filled(self, order, fill):
        """Handle order fill"""
        side = order.side.value.upper() if hasattr(order.side, 'value') else str(order.side)
        qty = getattr(fill, 'quantity', 0)
        price = getattr(fill, 'price', 0)
        
        self.log(
            f"✅ Order Filled: {side} {qty} {order.symbol} @ ¥{price:.2f}",
            "success"
        )
        self._refresh_portfolio()
    
    def _on_order_rejected(self, order, reason):
        """Handle order rejection"""
        self.log(f"❌ Order Rejected: {order.symbol} - {reason}", "error")

    def _refresh_portfolio(self):
        """Refresh portfolio display"""
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
            self.account_labels['positions'].setText(f"¥{market_value:,.2f}")
            
            pnl_color = "#3fb950" if total_pnl >= 0 else "#f85149"
            self.account_labels['pnl'].setText(f"¥{total_pnl:,.2f}")
            self.account_labels['pnl'].setStyleSheet(
                f"color: {pnl_color}; font-size: 18px; font-weight: bold;"
            )
            
            if hasattr(self.positions_table, 'update_positions'):
                self.positions_table.update_positions(positions)
                
        except Exception as e:
            log.debug(f"Portfolio refresh error: {e}")
    
    # ==================== Training ====================
    
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
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        try:
            # Your TrainingDialog currently accepts only (parent=None)
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
            "<p>Stock trading involves risk. Past performance does not guarantee future results. "
            "Only trade with money you can afford to lose.</p>"
        )
    
    # ==================== Utilities ====================
    
    def _update_clock(self):
        """Update clock"""
        self.time_label.setText(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    
    def _update_market_status(self):
        """Update market status"""
        is_open = CONFIG.is_market_open()
        
        if is_open:
            self.market_label.setText("🟢 Market Open")
            self.market_label.setStyleSheet("color: #3fb950; font-weight: bold;")
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
        
        formatted = f'<span style="color: #888;">[{timestamp}]</span> <span style="color: {color};">{message}</span>'
        
        if hasattr(self.log_widget, 'log'):
            self.log_widget.log(message, level)
        elif hasattr(self.log_widget, 'append'):
            self.log_widget.append(formatted)
        
        # Also log to file
        log.info(message)

    def closeEvent(self, event):
        """Handle window close safely (stop threads, stop trading, persist state)."""
        # Stop monitoring
        if self.monitor:
            try:
                self.monitor.stop()
                self.monitor.wait(3000)
            except Exception:
                pass
            self.monitor = None

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
        for timer in [getattr(self, "clock_timer", None),
                    getattr(self, "market_timer", None),
                    getattr(self, "portfolio_timer", None),
                    getattr(self, "watchlist_timer", None)]:
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
                    self.watch_list = state['watch_list']
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

    # Load state BEFORE show so UI reflects saved settings
    window._load_state()
    window._update_watchlist()

    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    run_app()