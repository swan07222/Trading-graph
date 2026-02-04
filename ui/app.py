"""
Main Application Window
"""
import sys
from datetime import datetime
from typing import Optional, Dict

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QLineEdit, QGroupBox, QProgressBar,
    QTabWidget, QStatusBar, QTextEdit, QDoubleSpinBox, QSpinBox,
    QSplitter, QComboBox, QMessageBox, QListWidget, QGridLayout,
    QFrame
)
from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt6.QtGui import QFont

from config import CONFIG, TradingMode
from models.predictor import Predictor, Prediction, Signal
from models.trainer import Trainer
from trading.executor import ExecutionEngine
from trading.broker_base import OrderSide
from .widgets import SignalPanel, PositionTable, LogWidget
from .charts import StockChart
from utils.logger import log


class WorkerThread(QThread):
    """Generic worker thread"""
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
    Main Application Window
    
    Features:
    - Stock analysis with AI predictions
    - Price charts with predictions
    - Portfolio management
    - Order execution
    """
    
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("AI Stock Trading Advisor")
        self.setGeometry(50, 50, 1600, 900)
        
        # State
        self.predictor: Optional[Predictor] = None
        self.executor: Optional[ExecutionEngine] = None
        self.current_prediction: Optional[Prediction] = None
        self.workers: Dict = {}
        
        # Setup
        self._setup_ui()
        self._setup_timers()
        self._apply_style()
        self._init_components()
    
    def _setup_ui(self):
        """Setup UI components"""
        central = QWidget()
        self.setCentralWidget(central)
        
        layout = QHBoxLayout(central)
        layout.setSpacing(10)
        layout.setContentsMargins(10, 10, 10, 10)
        
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Left panel
        left = self._create_left_panel()
        
        # Center panel
        center = self._create_center_panel()
        
        # Right panel
        right = self._create_right_panel()
        
        splitter.addWidget(left)
        splitter.addWidget(center)
        splitter.addWidget(right)
        splitter.setSizes([350, 700, 450])
        
        layout.addWidget(splitter)
        
        # Status bar
        self._setup_status_bar()
    
    def _create_left_panel(self) -> QWidget:
        """Create left control panel"""
        panel = QWidget()
        panel.setMaximumWidth(400)
        layout = QVBoxLayout(panel)
        layout.setSpacing(10)
        
        # Stock input
        input_group = QGroupBox("Stock Analysis")
        input_layout = QVBoxLayout()
        
        code_layout = QHBoxLayout()
        self.stock_input = QLineEdit()
        self.stock_input.setPlaceholderText("Enter stock code (e.g., 600519)")
        self.stock_input.returnPressed.connect(self._analyze_stock)
        code_layout.addWidget(self.stock_input)
        
        self.analyze_btn = QPushButton("Analyze")
        self.analyze_btn.clicked.connect(self._analyze_stock)
        self.analyze_btn.setStyleSheet("""
            QPushButton {
                background: #2196F3;
                color: white;
                border: none;
                padding: 8px 20px;
                border-radius: 5px;
                font-weight: bold;
            }
            QPushButton:hover { background: #1976D2; }
        """)
        code_layout.addWidget(self.analyze_btn)
        
        input_layout.addLayout(code_layout)
        input_group.setLayout(input_layout)
        layout.addWidget(input_group)
        
        # Stock list
        list_group = QGroupBox("Stock List")
        list_layout = QVBoxLayout()
        
        self.stock_list = QListWidget()
        for code in CONFIG.STOCK_POOL[:10]:
            self.stock_list.addItem(code)
        self.stock_list.setMaximumHeight(150)
        self.stock_list.itemDoubleClicked.connect(
            lambda item: (self.stock_input.setText(item.text()), self._analyze_stock())
        )
        list_layout.addWidget(self.stock_list)
        
        scan_btn = QPushButton("🔍 Scan All")
        scan_btn.clicked.connect(self._scan_stocks)
        list_layout.addWidget(scan_btn)
        
        list_group.setLayout(list_layout)
        layout.addWidget(list_group)
        
        # Trading mode
        mode_group = QGroupBox("Trading Mode")
        mode_layout = QVBoxLayout()
        
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Simulation", "Live Trading"])
        mode_layout.addWidget(self.mode_combo)
        
        capital_layout = QHBoxLayout()
        capital_layout.addWidget(QLabel("Capital:"))
        self.capital_spin = QDoubleSpinBox()
        self.capital_spin.setRange(10000, 100000000)
        self.capital_spin.setValue(CONFIG.CAPITAL)
        self.capital_spin.setPrefix("¥ ")
        capital_layout.addWidget(self.capital_spin)
        mode_layout.addLayout(capital_layout)
        
        self.connect_btn = QPushButton("Connect Trading")
        self.connect_btn.clicked.connect(self._toggle_trading)
        self.connect_btn.setStyleSheet("""
            QPushButton {
                background: #4CAF50;
                color: white;
                border: none;
                padding: 10px;
                border-radius: 5px;
                font-weight: bold;
            }
            QPushButton:hover { background: #388E3C; }
        """)
        mode_layout.addWidget(self.connect_btn)
        
        mode_group.setLayout(mode_layout)
        layout.addWidget(mode_group)
        
        # Training
        train_group = QGroupBox("Model Training")
        train_layout = QVBoxLayout()
        
        epochs_layout = QHBoxLayout()
        epochs_layout.addWidget(QLabel("Epochs:"))
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(10, 500)
        self.epochs_spin.setValue(CONFIG.EPOCHS)
        epochs_layout.addWidget(self.epochs_spin)
        train_layout.addLayout(epochs_layout)
        
        self.train_btn = QPushButton("🎓 Train Model")
        self.train_btn.clicked.connect(self._start_training)
        train_layout.addWidget(self.train_btn)
        
        self.train_progress = QProgressBar()
        self.train_progress.setVisible(False)
        train_layout.addWidget(self.train_progress)
        
        train_group.setLayout(train_layout)
        layout.addWidget(train_group)
        
        layout.addStretch()
        
        return panel
    
    def _create_center_panel(self) -> QWidget:
        """Create center panel with chart and signal"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setSpacing(10)
        
        # Signal panel
        self.signal_panel = SignalPanel()
        layout.addWidget(self.signal_panel)
        
        # Chart
        chart_group = QGroupBox("Price Chart & Prediction")
        chart_layout = QVBoxLayout()
        
        self.chart = StockChart()
        self.chart.setMinimumHeight(350)
        chart_layout.addWidget(self.chart)
        
        chart_group.setLayout(chart_layout)
        layout.addWidget(chart_group)
        
        # Analysis details
        details_group = QGroupBox("Analysis Details")
        details_layout = QVBoxLayout()
        
        self.details_text = QTextEdit()
        self.details_text.setReadOnly(True)
        self.details_text.setFont(QFont("Consolas", 10))
        self.details_text.setMaximumHeight(180)
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
        
        # Portfolio tab
        portfolio_tab = QWidget()
        portfolio_layout = QVBoxLayout(portfolio_tab)
        
        # Account summary
        account_frame = QFrame()
        account_frame.setStyleSheet("""
            QFrame {
                background: #1a1a3e;
                border-radius: 8px;
                padding: 10px;
            }
        """)
        account_layout = QGridLayout(account_frame)
        
        self.account_labels = {}
        labels = [
            ('equity', 'Total Equity', 0, 0),
            ('available', 'Available', 0, 1),
            ('market_value', 'Market Value', 1, 0),
            ('total_pnl', 'Total P&L', 1, 1),
        ]
        
        for key, text, row, col in labels:
            container = QWidget()
            cont_layout = QVBoxLayout(container)
            cont_layout.setContentsMargins(5, 5, 5, 5)
            
            title = QLabel(text)
            title.setStyleSheet("color: #888; font-size: 11px;")
            value = QLabel("--")
            value.setStyleSheet("color: #00E5FF; font-size: 16px; font-weight: bold;")
            
            cont_layout.addWidget(title)
            cont_layout.addWidget(value)
            
            account_layout.addWidget(container, row, col)
            self.account_labels[key] = value
        
        portfolio_layout.addWidget(account_frame)
        
        # Positions table
        self.positions_table = PositionTable()
        portfolio_layout.addWidget(self.positions_table)
        
        tabs.addTab(portfolio_tab, "Portfolio")
        
        # History tab
        history_tab = QWidget()
        history_layout = QVBoxLayout(history_tab)
        
        from PyQt6.QtWidgets import QTableWidget, QHeaderView
        self.history_table = QTableWidget()
        self.history_table.setColumnCount(6)
        self.history_table.setHorizontalHeaderLabels([
            'Time', 'Code', 'Signal', 'Prob UP', 'Confidence', 'Result'
        ])
        self.history_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        history_layout.addWidget(self.history_table)
        
        tabs.addTab(history_tab, "History")
        
        layout.addWidget(tabs)
        
        # Log
        log_group = QGroupBox("System Log")
        log_layout = QVBoxLayout()
        
        self.log_widget = LogWidget()
        log_layout.addWidget(self.log_widget)
        
        log_group.setLayout(log_layout)
        layout.addWidget(log_group)
        
        # Action buttons
        action_layout = QHBoxLayout()
        
        self.buy_btn = QPushButton("BUY")
        self.buy_btn.setStyleSheet("""
            QPushButton {
                background: #4CAF50;
                color: white;
                border: none;
                padding: 12px 30px;
                border-radius: 5px;
                font-weight: bold;
                font-size: 14px;
            }
            QPushButton:hover { background: #388E3C; }
            QPushButton:disabled { background: #333; color: #666; }
        """)
        self.buy_btn.clicked.connect(self._execute_buy)
        self.buy_btn.setEnabled(False)
        
        self.sell_btn = QPushButton("SELL")
        self.sell_btn.setStyleSheet("""
            QPushButton {
                background: #F44336;
                color: white;
                border: none;
                padding: 12px 30px;
                border-radius: 5px;
                font-weight: bold;
                font-size: 14px;
            }
            QPushButton:hover { background: #D32F2F; }
            QPushButton:disabled { background: #333; color: #666; }
        """)
        self.sell_btn.clicked.connect(self._execute_sell)
        self.sell_btn.setEnabled(False)
        
        action_layout.addWidget(self.buy_btn)
        action_layout.addWidget(self.sell_btn)
        
        layout.addLayout(action_layout)
        
        return panel
    
    def _setup_status_bar(self):
        """Setup status bar"""
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        
        self.progress = QProgressBar()
        self.progress.setMaximumWidth(200)
        self.progress.setMaximumHeight(15)
        self.progress.hide()
        self.statusBar.addPermanentWidget(self.progress)
        
        self.status_label = QLabel("Ready")
        self.statusBar.addWidget(self.status_label)
        
        self.market_label = QLabel("")
        self.statusBar.addPermanentWidget(self.market_label)
        
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
        
        self._update_market_status()
    
    def _apply_style(self):
        """Apply application style"""
        self.setStyleSheet("""
            QMainWindow {
                background: #0a0a1a;
            }
            
            QGroupBox {
                font-weight: bold;
                font-size: 12px;
                border: 2px solid #2a2a5a;
                border-radius: 10px;
                margin-top: 12px;
                padding-top: 12px;
                color: #00E5FF;
                background: #0f0f2a;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 5px;
            }
            
            QLabel {
                color: #ddd;
                font-size: 12px;
            }
            
            QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
                padding: 8px;
                border: 2px solid #2a2a5a;
                border-radius: 6px;
                background: #1a1a3e;
                color: #fff;
                font-size: 12px;
            }
            QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus {
                border-color: #00E5FF;
            }
            
            QListWidget {
                background: #1a1a3e;
                color: #fff;
                border: 1px solid #2a2a5a;
                border-radius: 6px;
            }
            QListWidget::item {
                padding: 6px;
            }
            QListWidget::item:selected {
                background: #3a3a7a;
            }
            QListWidget::item:hover {
                background: #2a2a5a;
            }
            
            QTableWidget {
                background: #1a1a3e;
                color: #fff;
                border: none;
                gridline-color: #2a2a5a;
            }
            QTableWidget::item {
                padding: 5px;
            }
            QTableWidget::item:selected {
                background: #3a3a7a;
            }
            
            QHeaderView::section {
                background: #2a2a5a;
                color: #00E5FF;
                padding: 8px;
                border: none;
                font-weight: bold;
            }
            
            QTabWidget::pane {
                border: 2px solid #2a2a5a;
                background: #0a0a1a;
                border-radius: 8px;
            }
            QTabBar::tab {
                background: #1a1a3e;
                color: #888;
                padding: 10px 20px;
                border-top-left-radius: 6px;
                border-top-right-radius: 6px;
                margin-right: 2px;
            }
            QTabBar::tab:selected {
                background: #2a2a5a;
                color: #00E5FF;
            }
            
            QPushButton {
                background: #3a3a7a;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 5px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: #4a4a9a;
            }
            QPushButton:disabled {
                background: #222;
                color: #555;
            }
            
            QProgressBar {
                border: none;
                background: #1a1a3e;
                border-radius: 5px;
                text-align: center;
                color: #fff;
            }
            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #00E5FF, stop:1 #00BCD4);
                border-radius: 5px;
            }
            
            QStatusBar {
                background: #0f0f2a;
                color: #888;
                border-top: 1px solid #2a2a5a;
            }
            
            QTextEdit {
                background: #0a0a1a;
                color: #0f0;
                border: 1px solid #2a2a5a;
                border-radius: 5px;
            }
            
            QScrollBar:vertical {
                background: #1a1a3e;
                width: 10px;
                border-radius: 5px;
            }
            QScrollBar::handle:vertical {
                background: #3a3a7a;
                border-radius: 5px;
                min-height: 30px;
            }
        """)
    
    def _init_components(self):
        """Initialize trading components"""
        try:
            self.predictor = Predictor(self.capital_spin.value())
            if self.predictor.ensemble:
                self.log("AI model loaded successfully", "success")
            else:
                self.log("No trained model found. Please train a model.", "warning")
        except Exception as e:
            self.log(f"Failed to load model: {e}", "error")
            self.predictor = None
        
        self.log("System initialized", "info")
    
    # ==================== Event Handlers ====================
    
    def _analyze_stock(self):
        """Analyze stock"""
        code = self.stock_input.text().strip()
        if not code:
            self.log("Please enter a stock code", "warning")
            return
        
        if self.predictor is None or self.predictor.ensemble is None:
            self.log("No model loaded. Please train a model first.", "error")
            return
        
        self.analyze_btn.setEnabled(False)
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
        self.analyze_btn.setEnabled(True)
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
        
        # Enable buttons
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
        self.analyze_btn.setEnabled(True)
        self.progress.hide()
        self.status_label.setText("Ready")
        
        self.log(f"Analysis failed: {error}", "error")
        QMessageBox.warning(self, "Error", f"Analysis failed:\n{error}")
        
        if 'analyze' in self.workers:
            del self.workers['analyze']
    
    def _update_details(self, pred: Prediction):
        """Update analysis details text"""
        signal_colors = {
            Signal.STRONG_BUY: "#00E676",
            Signal.BUY: "#69F0AE",
            Signal.HOLD: "#FFD54F",
            Signal.SELL: "#FF8A80",
            Signal.STRONG_SELL: "#FF1744",
        }
        
        color = signal_colors.get(pred.signal, "#fff")
        
        html = f"""
        <style>
            body {{ color: #ddd; font-family: Consolas; }}
            .signal {{ color: {color}; font-size: 16px; font-weight: bold; }}
            .section {{ margin: 8px 0; }}
            .label {{ color: #888; }}
            .positive {{ color: #4CAF50; }}
            .negative {{ color: #FF5252; }}
        </style>
        
        <div class="section">
            <span class="label">Signal: </span>
            <span class="signal">{pred.signal.value}</span>
            <span class="label"> | Strength: {pred.signal_strength:.0%}</span>
        </div>
        
        <div class="section">
            <span class="label">AI Probabilities: </span>
            <span class="positive">UP {pred.prob_up:.0%}</span> | 
            <span>NEUTRAL {pred.prob_neutral:.0%}</span> | 
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
        
        from PyQt6.QtWidgets import QTableWidgetItem
        from PyQt6.QtGui import QColor
        
        self.history_table.setItem(row, 0, QTableWidgetItem(
            pred.timestamp.strftime("%H:%M:%S")
        ))
        self.history_table.setItem(row, 1, QTableWidgetItem(pred.stock_code))
        
        signal_item = QTableWidgetItem(pred.signal.value)
        signal_item.setForeground(QColor("#00E5FF"))
        self.history_table.setItem(row, 2, signal_item)
        
        self.history_table.setItem(row, 3, QTableWidgetItem(f"{pred.prob_up:.0%}"))
        self.history_table.setItem(row, 4, QTableWidgetItem(f"{pred.confidence:.0%}"))
        self.history_table.setItem(row, 5, QTableWidgetItem("--"))
        
        while self.history_table.rowCount() > 50:
            self.history_table.removeRow(self.history_table.rowCount() - 1)
    
    def _scan_stocks(self):
        """Scan all stocks for signals"""
        if self.predictor is None or self.predictor.ensemble is None:
            self.log("No model loaded", "error")
            return
        
        self.log("Scanning stocks...", "info")
        self.progress.setRange(0, 0)
        self.progress.show()
        
        def scan():
            return self.predictor.get_top_picks(CONFIG.STOCK_POOL, n=5, signal_type="buy")
        
        worker = WorkerThread(scan)
        worker.finished.connect(self._on_scan_done)
        worker.error.connect(lambda e: (self.log(f"Scan failed: {e}", "error"), self.progress.hide()))
        self.workers['scan'] = worker
        worker.start()
    
    def _on_scan_done(self, picks):
        """Handle scan completion"""
        self.progress.hide()
        
        if not picks:
            self.log("No buy signals found", "info")
            return
        
        self.log(f"Found {len(picks)} buy signals:", "success")
        
        for pred in picks:
            self.log(
                f"  {pred.stock_code} {pred.stock_name}: "
                f"{pred.signal.value} (conf: {pred.confidence:.0%})",
                "info"
            )
        
        # Analyze top pick
        if picks:
            self.stock_input.setText(picks[0].stock_code)
            self._analyze_stock()
        
        if 'scan' in self.workers:
            del self.workers['scan']
    
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
                self, "Warning",
                "You are switching to LIVE trading mode!\n\n"
                "This will use REAL money. Are you sure?",
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
            self.connect_btn.setText("Disconnect")
            self.connect_btn.setStyleSheet("""
                QPushButton {
                    background: #F44336;
                    color: white;
                    border: none;
                    padding: 10px;
                    border-radius: 5px;
                    font-weight: bold;
                }
                QPushButton:hover { background: #D32F2F; }
            """)
            
            self.log(f"Connected to {mode.value} trading", "success")
            self._refresh_portfolio()
        else:
            self.executor = None
            self.log("Failed to connect", "error")
    
    def _disconnect_trading(self):
        """Disconnect from trading"""
        if self.executor:
            self.executor.stop()
            self.executor = None
        
        self.connect_btn.setText("Connect Trading")
        self.connect_btn.setStyleSheet("""
            QPushButton {
                background: #4CAF50;
                color: white;
                border: none;
                padding: 10px;
                border-radius: 5px;
                font-weight: bold;
            }
            QPushButton:hover { background: #388E3C; }
        """)
        
        self.log("Disconnected from trading", "info")
    
    def _execute_buy(self):
        """Execute buy order"""
        if not self.current_prediction or not self.executor:
            return
        
        pred = self.current_prediction
        
        reply = QMessageBox.question(
            self, "Confirm Buy",
            f"Buy {pred.stock_code} - {pred.stock_name}?\n\n"
            f"Quantity: {pred.position.shares:,} shares\n"
            f"Price: ¥{pred.levels.entry:.2f}\n"
            f"Value: ¥{pred.position.value:,.2f}\n"
            f"Stop Loss: ¥{pred.levels.stop_loss:.2f}",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            success = self.executor.submit_from_prediction(pred)
            if success:
                self.log(f"Buy order submitted: {pred.stock_code}", "info")
            else:
                self.log("Buy order failed", "error")
    
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
            self, "Confirm Sell",
            f"Sell {pred.stock_code} - {pred.stock_name}?\n\n"
            f"Available: {position.available_qty:,} shares\n"
            f"Current Price: ¥{position.current_price:.2f}",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            from trading.executor import TradeSignal
            
            signal = TradeSignal(
                stock_code=pred.stock_code,
                side=OrderSide.SELL,
                quantity=position.available_qty,
                price=position.current_price
            )
            
            success = self.executor.submit(signal)
            if success:
                self.log(f"Sell order submitted: {pred.stock_code}", "info")
            else:
                self.log("Sell order failed", "error")
    
    def _on_order_filled(self, order):
        """Handle order fill"""
        self.log(
            f"Filled: {order.side.value.upper()} {order.filled_qty} "
            f"{order.stock_code} @ ¥{order.filled_price:.2f}",
            "success"
        )
        self._refresh_portfolio()
    
    def _on_order_rejected(self, order, reason):
        """Handle order rejection"""
        self.log(f"Rejected: {order.stock_code} - {reason}", "error")
    
    def _refresh_portfolio(self):
        """Refresh portfolio display"""
        if not self.executor:
            return
        
        try:
            account = self.executor.get_account()
            
            self.account_labels['equity'].setText(f"¥{account.equity:,.2f}")
            self.account_labels['available'].setText(f"¥{account.available:,.2f}")
            self.account_labels['market_value'].setText(f"¥{account.market_value:,.2f}")
            
            pnl_color = "#4CAF50" if account.total_pnl >= 0 else "#FF5252"
            self.account_labels['total_pnl'].setText(f"¥{account.total_pnl:,.2f}")
            self.account_labels['total_pnl'].setStyleSheet(
                f"color: {pnl_color}; font-size: 16px; font-weight: bold;"
            )
            
            self.positions_table.update_positions(account.positions)
            
        except Exception as e:
            log.warning(f"Failed to refresh portfolio: {e}")
    
    def _start_training(self):
        """Start model training"""
        reply = QMessageBox.question(
            self, "Confirm Training",
            f"Start training with {self.epochs_spin.value()} epochs?\n\n"
            "This may take a while.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        
        if reply != QMessageBox.StandardButton.Yes:
            return
        
        self.train_btn.setEnabled(False)
        self.train_progress.setVisible(True)
        self.train_progress.setRange(0, 0)
        self.log("Starting model training...", "info")
        
        def train():
            trainer = Trainer()
            return trainer.train(epochs=self.epochs_spin.value())
        
        worker = WorkerThread(train)
        worker.finished.connect(self._on_training_done)
        worker.error.connect(self._on_training_error)
        self.workers['train'] = worker
        worker.start()
    
    def _on_training_done(self, results):
        """Handle training completion"""
        self.train_btn.setEnabled(True)
        self.train_progress.setVisible(False)
        
        accuracy = results.get('best_accuracy', 0)
        self.log(f"Training complete! Accuracy: {accuracy:.2%}", "success")
        
        # Show trading metrics
        if 'test_metrics' in results and 'trading' in results['test_metrics']:
            tm = results['test_metrics']['trading']
            self.log(
                f"Strategy return: {tm.get('total_return', 0):.1f}%, "
                f"Win rate: {tm.get('win_rate', 0):.1%}, "
                f"Sharpe: {tm.get('sharpe_ratio', 0):.2f}",
                "info"
            )
        
        # Reload model
        self._init_components()
        
        if 'train' in self.workers:
            del self.workers['train']
    
    def _on_training_error(self, error):
        """Handle training error"""
        self.train_btn.setEnabled(True)
        self.train_progress.setVisible(False)
        self.log(f"Training failed: {error}", "error")
        
        if 'train' in self.workers:
            del self.workers['train']
    
    def _update_clock(self):
        """Update clock"""
        self.time_label.setText(datetime.now().strftime("%H:%M:%S"))
    
    def _update_market_status(self):
        """Update market status"""
        is_open = CONFIG.is_market_open()
        
        if is_open:
            self.market_label.setText("🟢 Market Open")
            self.market_label.setStyleSheet("color: #4CAF50;")
        else:
            self.market_label.setText("🔴 Market Closed")
            self.market_label.setStyleSheet("color: #FF5252;")
    
    def log(self, message: str, level: str = "info"):
        """Log message"""
        self.log_widget.log(message, level)
    
    def closeEvent(self, event):
        """Handle window close"""
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
        
        event.accept()


def run_app():
    """Run the application"""
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    font = QFont("Arial", 10)
    app.setFont(font)
    
    window = MainApp()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    run_app()