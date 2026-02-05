"""
AI Stock Trading System - Professional Desktop Application
Real-time trading signals with custom AI model
"""
import sys
from datetime import datetime
from typing import Optional, Dict, List
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

from config import CONFIG, TradingMode
from models.predictor import Predictor, Prediction, Signal
from models.trainer import Trainer
from trading.executor import ExecutionEngine
from .widgets import SignalPanel, PositionTable, LogWidget
from .charts import StockChart
from utils.logger import log
from core.types import Order, OrderSide, OrderStatus, TradeSignal, Account, Position, Fill
from trading.alerts import AlertPriority

class RealTimeMonitor(QThread):
    """
    Real-time market monitoring thread
    Continuously checks for trading signals
    """
    signal_detected = pyqtSignal(object)  # Prediction
    price_updated = pyqtSignal(str, float)  # code, price
    error_occurred = pyqtSignal(str)
    
    def __init__(self, predictor: Predictor, watch_list: List[str]):
        super().__init__()
        self.predictor = predictor
        self.watch_list = watch_list
        self.running = False
        self.interval = 30  # seconds
    
    def run(self):
        """Main monitoring loop"""
        self.running = True
        
        while self.running:
            for code in self.watch_list:
                if not self.running:
                    break
                
                try:
                    pred = self.predictor.predict(code)
                    self.price_updated.emit(code, pred.current_price)
                    
                    # Emit signal if strong enough
                    if pred.signal in [Signal.STRONG_BUY, Signal.STRONG_SELL]:
                        if pred.confidence >= CONFIG.MIN_CONFIDENCE:
                            self.signal_detected.emit(pred)
                    
                except Exception as e:
                    self.error_occurred.emit(f"{code}: {str(e)}")
                
                time.sleep(2)  # Rate limit
            
            # Wait before next cycle
            for _ in range(self.interval):
                if not self.running:
                    break
                time.sleep(1)
    
    def stop(self):
        self.running = False


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
    
    def run(self):
        try:
            result = self.func(*self.args, **self.kwargs)
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))


class MainApp(QMainWindow):
    """
    Professional AI Stock Trading Application
    
    Features:
    - Real-time signal monitoring
    - Custom AI model with 6 neural networks
    - Professional dark theme
    - Live/Paper trading support
    - Comprehensive risk management
    """
    
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("AI Stock Trading System v2.0")
        self.setGeometry(50, 50, 1800, 1000)
        
        # State
        self.predictor: Optional[Predictor] = None
        self.executor: Optional[ExecutionEngine] = None
        self.current_prediction: Optional[Prediction] = None
        self.workers: Dict = {}
        self.monitor: Optional[RealTimeMonitor] = None
        self.watch_list: List[str] = CONFIG.STOCK_POOL[:10]
        
        # Setup
        self._setup_menubar()
        self._setup_toolbar()
        self._setup_ui()
        self._setup_statusbar()
        self._setup_timers()
        self._apply_professional_style()
        self._init_components()
    
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
        spacer.setSizePolicy(spacer.sizePolicy().horizontalPolicy(), 
                            spacer.sizePolicy().verticalPolicy())
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
        """Create left control panel"""
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
        
        # Populate initial watchlist
        self._update_watchlist()
        
        watchlist_layout.addWidget(self.watchlist)
        
        # Add/Remove buttons
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
        
        self.model_status = QLabel("Model: Not Loaded")
        ai_layout.addWidget(self.model_status)
        
        self.train_btn = QPushButton("🎓 Train Model")
        self.train_btn.clicked.connect(self._start_training)
        ai_layout.addWidget(self.train_btn)
        
        self.auto_learn_btn = QPushButton("🤖 Auto Learn")
        self.auto_learn_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #667eea, stop:1 #764ba2);
                color: white;
                border: none;
                padding: 12px;
                border-radius: 6px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #764ba2, stop:1 #667eea);
            }
        """)
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
        
        # Signal Display
        self.signal_panel = SignalPanel()
        layout.addWidget(self.signal_panel)
        
        # Chart
        chart_group = QGroupBox("📈 Price Chart & AI Prediction")
        chart_layout = QVBoxLayout()
        
        self.chart = StockChart()
        self.chart.setMinimumHeight(400)
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
        """Create right panel with portfolio and orders"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setSpacing(10)
        
        # Tabs
        tabs = QTabWidget()
        
        # Portfolio Tab
        portfolio_tab = QWidget()
        portfolio_layout = QVBoxLayout(portfolio_tab)
        
        # Account Summary
        account_frame = QFrame()
        account_frame.setStyleSheet("""
            QFrame {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #1a1a3e, stop:1 #2a2a5a);
                border-radius: 10px;
                padding: 15px;
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
        
        # Positions Table
        self.positions_table = PositionTable()
        portfolio_layout.addWidget(self.positions_table)
        
        tabs.addTab(portfolio_tab, "💼 Portfolio")
        
        # Real-time Signals Tab
        signals_tab = QWidget()
        signals_layout = QVBoxLayout(signals_tab)
        
        self.signals_table = QTableWidget()
        self.signals_table.setColumnCount(6)
        self.signals_table.setHorizontalHeaderLabels([
            "Time", "Code", "Signal", "Confidence", "Price", "Action"
        ])
        self.signals_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
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
        self.history_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        history_layout.addWidget(self.history_table)
        
        tabs.addTab(history_tab, "📜 History")
        
        layout.addWidget(tabs)
        
        # Log
        log_group = QGroupBox("📋 System Log")
        log_layout = QVBoxLayout()
        
        self.log_widget = LogWidget()
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
                background: #4CAF50;
                color: white;
                border: none;
                padding: 15px 40px;
                border-radius: 6px;
                font-weight: bold;
                font-size: 16px;
            }
            QPushButton:hover { background: #388E3C; }
            QPushButton:disabled { background: #333; color: #666; }
        """)
        self.buy_btn.clicked.connect(self._execute_buy)
        self.buy_btn.setEnabled(False)
        
        self.sell_btn = QPushButton("📉 SELL")
        self.sell_btn.setStyleSheet("""
            QPushButton {
                background: #F44336;
                color: white;
                border: none;
                padding: 15px 40px;
                border-radius: 6px;
                font-weight: bold;
                font-size: 16px;
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
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        
        # Progress bar
        self.progress = QProgressBar()
        self.progress.setMaximumWidth(200)
        self.progress.setMaximumHeight(15)
        self.progress.hide()
        self.statusBar.addPermanentWidget(self.progress)
        
        # Status
        self.status_label = QLabel("Ready")
        self.statusBar.addWidget(self.status_label)
        
        # Market status
        self.market_label = QLabel("")
        self.statusBar.addPermanentWidget(self.market_label)
        
        # Monitoring status
        self.monitor_label = QLabel("Monitoring: OFF")
        self.monitor_label.setStyleSheet("color: #888;")
        self.statusBar.addPermanentWidget(self.monitor_label)
        
        # Clock
        self.time_label = QLabel("")
        self.statusBar.addPermanentWidget(self.time_label)
    
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
            self.predictor = Predictor(self.capital_spin.value())
            if self.predictor.ensemble:
                self.model_status.setText("✅ Model: Loaded (6 networks)")
                self.model_status.setStyleSheet("color: #4CAF50;")
                self.log("AI model loaded successfully", "success")
            else:
                self.model_status.setText("⚠️ Model: Not trained")
                self.model_status.setStyleSheet("color: #FFD54F;")
                self.log("No trained model found. Please train a model.", "warning")
        except Exception as e:
            self.log(f"Failed to load model: {e}", "error")
            self.predictor = None
        
        self.log("System initialized - Ready for trading", "info")
    
    # ==================== Real-time Monitoring ====================
    
    def _toggle_monitoring(self, checked):
        """Toggle real-time monitoring"""
        if checked:
            self._start_monitoring()
        else:
            self._stop_monitoring()
    
    def _start_monitoring(self):
        """Start real-time signal monitoring"""
        if self.predictor is None or self.predictor.ensemble is None:
            self.log("Cannot start monitoring: No model loaded", "error")
            self.monitor_action.setChecked(False)
            return
        
        self.monitor = RealTimeMonitor(self.predictor, self.watch_list)
        self.monitor.signal_detected.connect(self._on_signal_detected)
        self.monitor.price_updated.connect(self._on_price_updated)
        self.monitor.error_occurred.connect(lambda e: self.log(f"Monitor: {e}", "warning"))
        self.monitor.start()
        
        self.monitor_label.setText("📡 Monitoring: ACTIVE")
        self.monitor_label.setStyleSheet("color: #4CAF50; font-weight: bold;")
        self.monitor_action.setText("⏹️ Stop Monitoring")
        
        self.log(f"Real-time monitoring started for {len(self.watch_list)} stocks", "success")
    
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
    
    def _on_signal_detected(self, pred: Prediction):
        """Handle detected trading signal"""
        # Add to signals table
        row = 0
        self.signals_table.insertRow(row)
        
        self.signals_table.setItem(row, 0, QTableWidgetItem(
            pred.timestamp.strftime("%H:%M:%S")
        ))
        self.signals_table.setItem(row, 1, QTableWidgetItem(
            f"{pred.stock_code} - {pred.stock_name}"
        ))
        
        signal_item = QTableWidgetItem(pred.signal.value)
        if pred.signal in [Signal.STRONG_BUY, Signal.BUY]:
            signal_item.setForeground(QColor("#4CAF50"))
        else:
            signal_item.setForeground(QColor("#F44336"))
        self.signals_table.setItem(row, 2, signal_item)
        
        self.signals_table.setItem(row, 3, QTableWidgetItem(f"{pred.confidence:.0%}"))
        self.signals_table.setItem(row, 4, QTableWidgetItem(f"¥{pred.current_price:.2f}"))
        
        # Action button
        action_btn = QPushButton("Trade")
        action_btn.clicked.connect(lambda: self._quick_trade(pred))
        self.signals_table.setCellWidget(row, 5, action_btn)
        
        # Keep only last 50 signals
        while self.signals_table.rowCount() > 50:
            self.signals_table.removeRow(self.signals_table.rowCount() - 1)
        
        # Notification
        self.log(
            f"🔔 SIGNAL: {pred.signal.value} - {pred.stock_code} @ ¥{pred.current_price:.2f}",
            "success"
        )
        
        # Flash the window if minimized
        QApplication.alert(self)
    
    def _on_price_updated(self, code: str, price: float):
        """Update price in watchlist"""
        for row in range(self.watchlist.rowCount()):
            if self.watchlist.item(row, 0).text() == code:
                self.watchlist.setItem(row, 1, QTableWidgetItem(f"¥{price:.2f}"))
                break
    
    def _quick_trade(self, pred: Prediction):
        """Quick trade from signal"""
        self.stock_input.setText(pred.stock_code)
        self._analyze_stock()
    
    # ==================== Watchlist ====================
    
    def _update_watchlist(self):
        """Update watchlist - preserve existing prices"""
        current_count = self.watchlist.rowCount()
        
        # Only add/remove rows if list changed
        if current_count != len(self.watch_list):
            self.watchlist.setRowCount(len(self.watch_list))
        
        for row, code in enumerate(self.watch_list):
            # Only set code if different
            current_code = self.watchlist.item(row, 0)
            if current_code is None or current_code.text() != code:
                self.watchlist.setItem(row, 0, QTableWidgetItem(code))
            
            # Initialize other columns ONLY if empty
            for col in range(1, 4):
                if self.watchlist.item(row, col) is None:
                    self.watchlist.setItem(row, col, QTableWidgetItem("--"))
    
    def _on_watchlist_click(self, row, col):
        """Handle watchlist double-click"""
        code = self.watchlist.item(row, 0).text()
        self.stock_input.setText(code)
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
            code = self.watchlist.item(row, 0).text()
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
        
        self.analyze_action.setEnabled(False)
        self.signal_panel.reset()
        self.status_label.setText(f"Analyzing {code}...")
        self.progress.setRange(0, 0)
        self.progress.show()
        
        def analyze():
            return self.predictor.predict(code)
        
        worker = WorkerThread(analyze)
        worker.finished.connect(self._on_analysis_done)
        worker.error.connect(self._on_analysis_error)
        self.workers['analyze'] = worker
        worker.start()
    
    def _on_analysis_done(self, pred: Prediction):
        """Handle analysis completion"""
        self.analyze_action.setEnabled(True)
        self.progress.hide()
        self.status_label.setText("Ready")
        
        self.current_prediction = pred
        
        # Update signal panel
        self.signal_panel.update_prediction(pred)
        
        # Update chart
        levels = {
            'stop_loss': pred.levels.stop_loss,
            'target_1': pred.levels.target_1,
            'target_2': pred.levels.target_2,
            'target_3': pred.levels.target_3,
        }
        self.chart.update_data(
            pred.price_history,
            pred.predicted_prices,
            levels
        )
        
        # Update details
        self._update_details(pred)
        
        # Add to history
        self._add_to_history(pred)
        
        # Enable buttons based on signal
        self.buy_btn.setEnabled(pred.signal in [Signal.STRONG_BUY, Signal.BUY])
        self.sell_btn.setEnabled(pred.signal in [Signal.STRONG_SELL, Signal.SELL])
        
        self.log(
            f"Analysis complete: {pred.stock_code} - {pred.signal.value} "
            f"(confidence: {pred.confidence:.0%})",
            "success"
        )
        
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
    
    def _update_details(self, pred: Prediction):
        """Update analysis details"""
        signal_colors = {
            Signal.STRONG_BUY: "#2ea043",
            Signal.BUY: "#3fb950",
            Signal.HOLD: "#d29922",
            Signal.SELL: "#f85149",
            Signal.STRONG_SELL: "#da3633",
        }
        
        color = signal_colors.get(pred.signal, "#c9d1d9")
        
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
            <span class="signal">{pred.signal.value}</span>
            <span class="label"> | Strength: {pred.signal_strength:.0%}</span>
        </div>
        
        <div class="section">
            <span class="label">AI Prediction: </span>
            <span class="positive">UP {pred.prob_up:.0%}</span> | 
            <span class="neutral">NEUTRAL {pred.prob_neutral:.0%}</span> | 
            <span class="negative">DOWN {pred.prob_down:.0%}</span>
        </div>
        
        <div class="section">
            <span class="label">Technical: </span>
            RSI={pred.rsi:.0f} | MACD={pred.macd_signal} | Trend={pred.trend}
        </div>
        
        <div class="section">
            <span class="label">Trading Plan:</span><br/>
            Entry: ¥{pred.levels.entry:.2f} | 
            Stop: ¥{pred.levels.stop_loss:.2f} ({pred.levels.stop_loss_pct:+.1f}%)<br/>
            Target 1: ¥{pred.levels.target_1:.2f} ({pred.levels.target_1_pct:+.1f}%) |
            Target 2: ¥{pred.levels.target_2:.2f} ({pred.levels.target_2_pct:+.1f}%)
        </div>
        
        <div class="section">
            <span class="label">Position:</span>
            {pred.position.shares:,} shares | ¥{pred.position.value:,.2f} | 
            Risk: ¥{pred.position.risk_amount:,.2f}
        </div>
        
        <div class="section">
            <span class="label">Analysis:</span><br/>
        """
        
        for reason in pred.reasons[:5]:
            html += f"• {reason}<br/>"
        
        if pred.warnings:
            html += "<br/><span class='negative'>⚠️ Warnings:</span><br/>"
            for warning in pred.warnings:
                html += f"• {warning}<br/>"
        
        html += "</div>"
        
        self.details_text.setHtml(html)
    
    def _add_to_history(self, pred: Prediction):
        """Add prediction to history"""
        row = 0
        self.history_table.insertRow(row)
        
        self.history_table.setItem(row, 0, QTableWidgetItem(
            pred.timestamp.strftime("%H:%M:%S")
        ))
        self.history_table.setItem(row, 1, QTableWidgetItem(pred.stock_code))
        
        signal_item = QTableWidgetItem(pred.signal.value)
        signal_item.setForeground(QColor("#58a6ff"))
        self.history_table.setItem(row, 2, signal_item)
        
        self.history_table.setItem(row, 3, QTableWidgetItem(f"{pred.prob_up:.0%}"))
        self.history_table.setItem(row, 4, QTableWidgetItem(f"{pred.confidence:.0%}"))
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
            return self.predictor.get_top_picks(CONFIG.STOCK_POOL, n=10, signal_type="buy")
        
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
        
        for pred in picks:
            self.log(
                f"  📈 {pred.stock_code} {pred.stock_name}: "
                f"{pred.signal.value} (confidence: {pred.confidence:.0%})",
                "info"
            )
        
        # Analyze top pick
        if picks:
            self.stock_input.setText(picks[0].stock_code)
            self._analyze_stock()
        
        if 'scan' in self.workers:
            del self.workers['scan']
    
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
    
    def _disconnect_trading(self):
        """Disconnect from trading"""
        if self.executor:
            self.executor.stop()
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
        
        reply = QMessageBox.question(
            self, "Confirm Buy Order",
            f"<b>Buy {pred.stock_code} - {pred.stock_name}</b><br><br>"
            f"Quantity: {pred.position.shares:,} shares<br>"
            f"Price: ¥{pred.levels.entry:.2f}<br>"
            f"Value: ¥{pred.position.value:,.2f}<br>"
            f"Stop Loss: ¥{pred.levels.stop_loss:.2f}<br>"
            f"Target: ¥{pred.levels.target_2:.2f}",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            success = self.executor.submit_from_prediction(pred)
            if success:
                self.log(f"Buy order submitted: {pred.stock_code}", "info")
            else:
                self.log("Buy order failed risk checks", "error")
    
    def _execute_sell(self):
        """Execute sell order"""
        if not self.current_prediction or not self.executor:
            return
        
        pred = self.current_prediction
        positions = self.executor.get_positions()
        position = positions.get(pred.stock_code)
        
        if not position:
            self.log("No position to sell", "warning")
            return
        
        reply = QMessageBox.question(
            self, "Confirm Sell Order",
            f"<b>Sell {pred.stock_code} - {pred.stock_name}</b><br><br>"
            f"Available: {position.available_qty:,} shares<br>"
            f"Current Price: ¥{position.current_price:.2f}",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            # FIXED: Use core.types
            signal = TradeSignal(
                symbol=pred.stock_code,
                name=pred.stock_name,
                side=OrderSide.SELL,
                quantity=position.available_qty,
                price=position.current_price
            )
            
            success = self.executor.submit(signal)
            if success:
                self.log(f"Sell order submitted: {pred.stock_code}", "info")
            else:
                self.log("Sell order failed", "error")
        
    def _on_order_filled(self, order: Order, fill: Fill):
        """Handle order fill - now receives both order and fill"""
        self.log(
            f"✅ Order Filled: {order.side.value.upper()} {fill.quantity} "
            f"{order.symbol} @ ¥{fill.price:.2f}",
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
            
            self.account_labels['equity'].setText(f"¥{account.equity:,.2f}")
            self.account_labels['cash'].setText(f"¥{account.available:,.2f}")
            self.account_labels['positions'].setText(f"¥{account.market_value:,.2f}")
            
            pnl_color = "#3fb950" if account.total_pnl >= 0 else "#f85149"
            self.account_labels['pnl'].setText(f"¥{account.total_pnl:,.2f}")
            self.account_labels['pnl'].setStyleSheet(
                f"color: {pnl_color}; font-size: 18px; font-weight: bold;"
            )
            
            self.positions_table.update_positions(account.positions)
            
        except Exception as e:
            log.warning(f"Failed to refresh portfolio: {e}")
    
    # ==================== Training ====================
    
    def _start_training(self):
        """Start model training"""
        reply = QMessageBox.question(
            self, "Train AI Model",
            f"Start training with the following settings?\n\n"
            f"This will train 6 neural networks (LSTM, Transformer, GRU, TCN, Hybrid)\n"
            f"and may take 30-60 minutes.\n\n"
            f"Continue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        
        if reply != QMessageBox.StandardButton.Yes:
            return
        
        from .dialogs import TrainingDialog
        dialog = TrainingDialog(self)
        dialog.exec()
        
        # Reload model after training
        self._init_components()
    
    def _show_auto_learn(self):
        """Show auto-learning dialog"""
        from .auto_learn_dialog import show_auto_learn_dialog
        show_auto_learn_dialog(self)
        
        # Reload model after learning
        self._init_components()
    
    def _show_backtest(self):
        """Show backtest dialog"""
        from .dialogs import BacktestDialog
        dialog = BacktestDialog(self)
        dialog.exec()
    
    def _show_about(self):
        """Show about dialog"""
        QMessageBox.about(
            self,
            "About AI Stock Trading System",
            "<h2>AI Stock Trading System v2.0</h2>"
            "<p>Professional AI-powered stock trading application</p>"
            "<h3>Features:</h3>"
            "<ul>"
            "<li>Custom AI model with 6 neural networks</li>"
            "<li>Real-time signal monitoring</li>"
            "<li>Automatic stock discovery from internet</li>"
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
        """Log message"""
        self.log_widget.log(message, level)
    
    def closeEvent(self, event):
        """Handle window close"""
        # Stop monitoring
        if self.monitor:
            self.monitor.stop()
            self.monitor.wait(3000)
        
        # Stop workers
        for worker in self.workers.values():
            worker.quit()
            worker.wait(2000)
        
        # Disconnect trading
        if self.executor:
            self.executor.stop()
        
        # Stop timers
        self.clock_timer.stop()
        self.market_timer.stop()
        self.portfolio_timer.stop()
        self.watchlist_timer.stop()
        
        event.accept()


def run_app():
    """Run the application"""
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    # Set application-wide font
    font = QFont("Segoe UI", 10)
    app.setFont(font)
    
    window = MainApp()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    run_app()