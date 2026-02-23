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
from ui.modern_theme import (
    ModernColors,
    ModernFonts,
    get_app_stylesheet,
    get_connection_button_style,
    get_connection_status_style,
    get_primary_font_family,
)
from utils.logger import get_logger
from utils.recoverable import COMMON_RECOVERABLE_EXCEPTIONS

log = get_logger(__name__)

_APP_PANELS_RECOVERABLE_EXCEPTIONS = COMMON_RECOVERABLE_EXCEPTIONS


def _lazy_get(module: str, name: str) -> Any:
    return getattr(import_module(module), name)


def _create_left_panel(self: Any) -> QWidget:
    """Create left control panel with interval/forecast settings."""
    panel = QWidget()
    panel.setObjectName("leftPanel")
    panel.setMinimumWidth(260)
    panel.setMaximumWidth(340)
    layout = QVBoxLayout(panel)
    layout.setSpacing(14)
    layout.setContentsMargins(6, 10, 6, 8)

    watchlist_group = QGroupBox("Watchlist")
    watchlist_layout = QVBoxLayout()
    watchlist_layout.setSpacing(10)

    self.watchlist = self._make_table(
        ["Code", "Price", "Change", "Signal"], max_height=250
    )
    self.watchlist.cellClicked.connect(self._on_watchlist_click)

    self._update_watchlist()
    watchlist_layout.addWidget(self.watchlist)

    btn_layout = QHBoxLayout()
    btn_layout.setSpacing(10)
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
    settings_layout.setHorizontalSpacing(10)
    settings_layout.setVerticalSpacing(9)

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
    connection_layout.setSpacing(10)

    self.connection_status = QLabel("Disconnected")
    self.connection_status.setObjectName("connectionStatus")
    self.connection_status.setStyleSheet(
        get_connection_status_style(False)
    )
    connection_layout.addWidget(self.connection_status)

    self.connect_btn = QPushButton("Connect to Broker")
    self.connect_btn.setObjectName("connectButton")
    self.connect_btn.clicked.connect(self._toggle_trading)
    self.connect_btn.setStyleSheet(
        get_connection_button_style(False)
    )
    connection_layout.addWidget(self.connect_btn)

    connection_group.setLayout(connection_layout)
    layout.addWidget(connection_group)

    ai_group = QGroupBox("AI Model")
    ai_layout = QVBoxLayout()

    self.model_status = QLabel("Model: Loading...")
    ai_layout.addWidget(self.model_status)

    self.model_info = QLabel("")
    self.model_info.setObjectName("metaLabel")
    ai_layout.addWidget(self.model_info)

    self.trained_stocks_label = QLabel("Trained Stocks: --")
    self.trained_stocks_label.setObjectName("metaLabel")
    ai_layout.addWidget(self.trained_stocks_label)

    self.trained_stocks_hint = QLabel(
        "Full trained stock list is in the right panel tab:\n"
        "Trained Stocks"
    )
    self.trained_stocks_hint.setObjectName("metaLabel")
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
    """Create right panel with portfolio, news, orders, and auto-trade."""
    panel = QWidget()
    panel.setObjectName("rightPanel")
    layout = QVBoxLayout(panel)
    layout.setSpacing(14)
    layout.setContentsMargins(4, 4, 4, 4)

    self.right_tabs = QTabWidget()
    self.right_tabs.setDocumentMode(True)
    tabs = self.right_tabs

    portfolio_tab = QWidget()
    portfolio_layout = QVBoxLayout(portfolio_tab)
    portfolio_layout.setSpacing(10)
    portfolio_layout.setContentsMargins(8, 8, 8, 8)

    self.account_labels = {}
    labels = [
        ('equity', 'Total Equity', 0, 0),
        ('cash', 'Available Cash', 0, 1),
        ('positions', 'Positions Value', 1, 0),
        ('pnl', 'Total P&L', 1, 1),
    ]
    account_frame, self.account_labels = self._build_stat_frame(
        labels,
        (
            f"color: {ModernColors.ACCENT_INFO}; "
            f"font-size: {ModernFonts.SIZE_XXL}px; "
            f"font-weight: {ModernFonts.WEIGHT_BOLD};"
        ),
        15,
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
    news_layout.setContentsMargins(8, 8, 8, 8)
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
    signals_layout.setContentsMargins(8, 8, 8, 8)
    self.signals_table = self._make_table([
        "Time", "Code", "Signal", "Confidence", "Price", "Action"
    ])
    signals_layout.addWidget(self.signals_table)
    tabs.addTab(signals_tab, "Live Signals")

    history_tab = QWidget()
    history_layout = QVBoxLayout(history_tab)
    history_layout.setContentsMargins(8, 8, 8, 8)
    self.history_table = self._make_table([
        "Time", "Code", "Signal", "Prob UP", "Confidence", "Result"
    ])
    history_layout.addWidget(self.history_table)
    tabs.addTab(history_tab, "History")

    trained_tab = QWidget()
    trained_layout = QVBoxLayout(trained_tab)
    trained_layout.setContentsMargins(8, 8, 8, 8)

    trained_top = QHBoxLayout()
    self.trained_stock_count_label = QLabel("Trained: --")
    self.trained_stock_count_label.setObjectName("metaLabel")
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
    auto_trade_layout.setSpacing(10)
    auto_trade_layout.setContentsMargins(8, 8, 8, 8)

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
        auto_labels,
        (
            f"color: {ModernColors.ACCENT_INFO}; "
            f"font-size: {ModernFonts.SIZE_XL}px; "
            f"font-weight: {ModernFonts.WEIGHT_BOLD};"
        ),
        10,
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
    auto_btn_frame.setObjectName("actionStrip")
    auto_btn_layout = QHBoxLayout(auto_btn_frame)
    auto_btn_layout.setContentsMargins(8, 8, 8, 8)
    auto_btn_layout.setSpacing(10)

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
    action_frame.setObjectName("actionStrip")
    action_layout = QHBoxLayout(action_frame)
    action_layout.setContentsMargins(10, 10, 10, 10)
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
    self.setFont(QFont(get_primary_font_family(), ModernFonts.SIZE_BASE))
    self.setStyleSheet(get_app_stylesheet())
