from __future__ import annotations

from importlib import import_module
from typing import Any

from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from config.settings import CONFIG
from utils.logger import get_logger
from utils.recoverable import COMMON_RECOVERABLE_EXCEPTIONS

log = get_logger(__name__)

_APP_PANELS_RECOVERABLE_EXCEPTIONS = COMMON_RECOVERABLE_EXCEPTIONS


def _lazy_get(module: str, name: str) -> Any:
    return getattr(import_module(module), name)


def _create_left_panel(self: Any) -> QWidget:
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



def _create_right_panel(self: Any) -> QWidget:
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
    except _APP_PANELS_RECOVERABLE_EXCEPTIONS as e:
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
            background: transparent;
            border-radius: 12px;
            padding: 12px;
        }
    """)
    action_layout = QHBoxLayout(action_frame)
    action_layout.setSpacing(16)

    self.buy_btn = QPushButton("BUY")
    self.buy_btn.setObjectName("buyButton")
    self.buy_btn.clicked.connect(self._execute_buy)
    self.buy_btn.setEnabled(False)

    self.sell_btn = QPushButton("SELL")
    self.sell_btn.setObjectName("sellButton")
    self.sell_btn.clicked.connect(self._execute_sell)
    self.sell_btn.setEnabled(False)

    action_layout.addWidget(self.buy_btn)
    action_layout.addWidget(self.sell_btn)
    layout.addWidget(action_frame)

    return panel



def _apply_professional_style(self: Any) -> None:
    """Apply a modern, professional trading platform theme with enhanced UX."""
    self.setFont(QFont("Segoe UI", 10))
    self.setStyleSheet("""
        /* ===== MAIN WINDOW ===== */
        QMainWindow, QWidget {
            background: #0a0e1a;
            color: #e6e9f0;
            font-family: 'Segoe UI', 'Inter', sans-serif;
        }

        /* ===== MENU BAR ===== */
        QMenuBar {
            background: #111827;
            color: #e6e9f0;
            border-bottom: 1px solid #1f2937;
            padding: 4px 8px;
            font-weight: 500;
        }
        QMenuBar::item {
            padding: 8px 14px;
            border-radius: 6px;
            margin: 2px 4px;
            background: transparent;
        }
        QMenuBar::item:selected {
            background: #1f2937;
        }
        QMenuBar::item:pressed {
            background: #374151;
        }
        QMenu {
            background: #1f2937;
            color: #e6e9f0;
            border: 1px solid #374151;
            border-radius: 8px;
            padding: 8px;
        }
        QMenu::item {
            padding: 10px 20px;
            border-radius: 6px;
            margin: 2px 0;
        }
        QMenu::item:selected {
            background: #374151;
        }
        QMenu::separator {
            height: 1px;
            background: #374151;
            margin: 6px 0;
        }

        /* ===== TOOLBAR ===== */
        QToolBar {
            background: #111827;
            border: none;
            border-bottom: 1px solid #1f2937;
            spacing: 10px;
            padding: 8px 12px;
        }
        QToolButton {
            background: #1f2937;
            color: #e6e9f0;
            border: 1px solid #374151;
            border-radius: 8px;
            padding: 8px 16px;
            font-weight: 600;
            transition: all 0.15s ease;
        }
        QToolButton:hover {
            background: #374151;
            border-color: #60a5fa;
        }
        QToolButton:pressed {
            background: #4b5563;
            transform: translateY(1px);
        }

        /* ===== GROUP BOX ===== */
        QGroupBox {
            font-weight: 600;
            font-size: 12px;
            border: 1px solid #1f2937;
            border-radius: 12px;
            margin-top: 14px;
            padding-top: 14px;
            color: #93c5fd;
            background: #111827;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 14px;
            padding: 0 8px;
            color: #60a5fa;
        }

        /* ===== LABELS ===== */
        QLabel {
            color: #e6e9f0;
            font-size: 12px;
            background: transparent;
        }

        /* ===== INPUT FIELDS ===== */
        QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QListWidget, QTextEdit {
            min-height: 36px;
            padding: 6px 12px;
            border: 1px solid #374151;
            border-radius: 8px;
            background: #1f2937;
            color: #e6e9f0;
            selection-background-color: #3b82f6;
            selection-color: #ffffff;
            font-size: 13px;
        }
        QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus, QListWidget:focus, QTextEdit:focus {
            border-color: #60a5fa;
            background: #1f2937;
            outline: none;
        }
        QLineEdit:hover, QSpinBox:hover, QDoubleSpinBox:hover, QComboBox:hover, QListWidget:hover {
            border-color: #4b5563;
        }

        /* ===== COMBO BOX SPECIFIC ===== */
        QComboBox::drop-down {
            border: none;
            width: 28px;
        }
        QComboBox::down-arrow {
            width: 12px;
            height: 12px;
        }

        /* ===== TABLES & LISTS ===== */
        QTableWidget, QTableView, QTreeView, QListWidget {
            background: #111827;
            color: #e6e9f0;
            border: 1px solid #1f2937;
            border-radius: 10px;
            gridline-color: #1f2937;
            selection-background-color: #1e3a5f;
            selection-color: #ffffff;
            alternate-background-color: #0f1724;
            outline: none;
            font-size: 12px;
        }
        QTableWidget::item, QTableView::item, QListWidget::item {
            padding: 8px 10px;
            border: none;
        }
        QTableWidget::item:hover, QTableView::item:hover, QListWidget::item:hover {
            background: #1f2937;
        }
        QHeaderView::section {
            background: #1f2937;
            color: #93c5fd;
            padding: 10px 12px;
            border: none;
            border-right: 1px solid #1f2937;
            border-bottom: 1px solid #1f2937;
            font-weight: 600;
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        /* ===== TABS ===== */
        QTabWidget::pane {
            border: 1px solid #1f2937;
            background: #111827;
            border-radius: 10px;
            top: -1px;
        }
        QTabBar::tab {
            background: #1f2937;
            color: #9ca3af;
            padding: 10px 18px;
            border-top-left-radius: 8px;
            border-top-right-radius: 8px;
            margin-right: 4px;
            min-width: 80px;
            font-weight: 500;
        }
        QTabBar::tab:selected {
            background: #374151;
            color: #e6e9f0;
            border: 1px solid #4b5563;
            border-bottom: 1px solid #374151;
        }
        QTabBar::tab:hover:!selected {
            color: #e6e9f0;
            background: #2d3748;
        }

        /* ===== BUTTONS ===== */
        QPushButton {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 #3b82f6, stop:1 #2563eb);
            color: #ffffff;
            border: 1px solid #1d4ed8;
            border-radius: 10px;
            padding: 10px 20px;
            font-weight: 600;
            font-size: 13px;
            min-height: 38px;
        }
        QPushButton:hover {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 #60a5fa, stop:1 #3b82f6);
            border-color: #3b82f6;
        }
        QPushButton:pressed {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 #2563eb, stop:1 #1d4ed8);
            transform: translateY(1px);
        }
        QPushButton:disabled {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 #374151, stop:1 #1f2937);
            color: #6b7280;
            border-color: #374151;
        }

        /* Special button colors */
        QPushButton#buyButton, QPushButton[cssClass="buy"] {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 #10b981, stop:1 #059669);
            border-color: #047857;
        }
        QPushButton#buyButton:hover, QPushButton[cssClass="buy"]:hover {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 #34d399, stop:1 #10b981);
            border-color: #10b981;
        }

        QPushButton#sellButton, QPushButton[cssClass="sell"] {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 #ef4444, stop:1 #dc2626);
            border-color: #b91c1c;
        }
        QPushButton#sellButton:hover, QPushButton[cssClass="sell"]:hover {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 #f87171, stop:1 #ef4444);
            border-color: #ef4444;
        }

        /* ===== CHECKBOXES & RADIOS ===== */
        QCheckBox, QRadioButton {
            spacing: 8px;
            color: #e6e9f0;
            background: transparent;
            font-size: 12px;
        }
        QCheckBox::indicator, QRadioButton::indicator {
            width: 18px;
            height: 18px;
        }
        QCheckBox::indicator {
            border: 2px solid #4b5563;
            border-radius: 4px;
            background: #1f2937;
        }
        QCheckBox::indicator:hover {
            border-color: #60a5fa;
        }
        QCheckBox::indicator:checked {
            background: #3b82f6;
            border-color: #3b82f6;
        }
        QRadioButton::indicator {
            border: 2px solid #4b5563;
            border-radius: 9px;
            background: #1f2937;
        }
        QRadioButton::indicator:hover {
            border-color: #60a5fa;
        }
        QRadioButton::indicator:checked {
            background: #3b82f6;
            border-color: #3b82f6;
        }

        /* ===== TEXT EDITORS ===== */
        QTextEdit, QPlainTextEdit {
            background: #0f1724;
            color: #d1fae5;
            border: 1px solid #1f2937;
            border-radius: 10px;
            font-family: 'Consolas', 'Cascadia Code', 'JetBrains Mono', monospace;
            padding: 8px;
            selection-background-color: #3b82f6;
            selection-color: #ffffff;
            font-size: 12px;
        }

        /* ===== PROGRESS BAR ===== */
        QProgressBar {
            border: 1px solid #374151;
            background: #1f2937;
            border-radius: 8px;
            text-align: center;
            color: #e6e9f0;
            min-height: 20px;
            font-weight: 600;
            font-size: 11px;
        }
        QProgressBar::chunk {
            border-radius: 7px;
            background: qlineargradient(
                x1:0, y1:0, x2:1, y2:0,
                stop:0 #3b82f6, stop:1 #10b981
            );
        }

        /* ===== STATUS BAR ===== */
        QStatusBar {
            background: #111827;
            color: #9ca3af;
            border-top: 1px solid #1f2937;
            font-size: 11px;
        }
        QStatusBar::item {
            border: none;
        }

        /* ===== SCROLLBARS ===== */
        QScrollBar:vertical {
            background: #111827;
            width: 12px;
            margin: 3px;
            border-radius: 6px;
        }
        QScrollBar::handle:vertical {
            background: #4b5563;
            border-radius: 6px;
            min-height: 30px;
        }
        QScrollBar::handle:vertical:hover {
            background: #6b7280;
        }
        QScrollBar::handle:vertical:pressed {
            background: #9ca3af;
        }
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
            border: none;
            background: none;
            height: 0;
        }
        QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
            background: none;
        }
        QScrollBar:horizontal {
            background: #111827;
            height: 12px;
            margin: 3px;
            border-radius: 6px;
        }
        QScrollBar::handle:horizontal {
            background: #4b5563;
            border-radius: 6px;
            min-width: 30px;
        }
        QScrollBar::handle:horizontal:hover {
            background: #6b7280;
        }
        QScrollBar::handle:horizontal:pressed {
            background: #9ca3af;
        }
        QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
            border: none;
            background: none;
            width: 0;
        }
        QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {
            background: none;
        }

        /* ===== SPLITTER ===== */
        QSplitter::handle {
            background: #1f2937;
        }
        QSplitter::handle:hover {
            background: #374151;
        }
        QSplitter::handle:horizontal {
            width: 2px;
        }
        QSplitter::handle:vertical {
            height: 2px;
        }

        /* ===== TOOLTIPS ===== */
        QToolTip {
            background: #1f2937;
            color: #e6e9f0;
            border: 1px solid #374151;
            border-radius: 6px;
            padding: 8px 12px;
            font-size: 12px;
            font-weight: 500;
        }

        /* ===== SLIDERS ===== */
        QSlider::groove:horizontal {
            border: 1px solid #374151;
            height: 6px;
            background: #1f2937;
            border-radius: 3px;
        }
        QSlider::handle:horizontal {
            background: #3b82f6;
            border: 1px solid #1d4ed8;
            width: 18px;
            margin: -7px 0;
            border-radius: 9px;
        }
        QSlider::handle:horizontal:hover {
            background: #60a5fa;
        }
        QSlider::add-page:horizontal {
            background: #1f2937;
            border-radius: 3px;
        }
        QSlider::sub-page:horizontal {
            background: #3b82f6;
            border-radius: 3px;
        }

        /* ===== SPIN BOX ARROWS ===== */
        QSpinBox::up-button, QDoubleSpinBox::up-button {
            subcontrol-origin: border;
            subcontrol-position: top right;
            width: 20px;
            border: none;
            border-top-right-radius: 8px;
            background: #374151;
        }
        QSpinBox::down-button, QDoubleSpinBox::down-button {
            subcontrol-origin: border;
            subcontrol-position: bottom right;
            width: 20px;
            border: none;
            border-bottom-right-radius: 8px;
            background: #374151;
        }
        QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover,
        QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {
            background: #4b5563;
        }
        QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {
            width: 10px;
            height: 10px;
        }
        QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {
            width: 10px;
            height: 10px;
        }

        /* ===== LIST WIDGET ITEMS ===== */
        QListWidget::item:selected {
            background: #1e3a5f;
            color: #ffffff;
        }
        QListWidget::item:hover {
            background: #1f2937;
        }
    """)
