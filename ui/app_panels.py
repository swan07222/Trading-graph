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



def _apply_professional_style(self: Any) -> None:
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
