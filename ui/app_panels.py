from __future__ import annotations

from importlib import import_module
from typing import Any

from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
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
    add_btn.setObjectName("smallGhostButton")
    add_btn.clicked.connect(self._add_to_watchlist)
    remove_btn = QPushButton("- Remove")
    remove_btn.setObjectName("smallGhostButton")
    remove_btn.clicked.connect(self._remove_from_watchlist)
    btn_layout.addWidget(add_btn)
    btn_layout.addWidget(remove_btn)
    watchlist_layout.addLayout(btn_layout)

    watchlist_group.setLayout(watchlist_layout)
    layout.addWidget(watchlist_group)

    universe_group = QGroupBox("Market Universe")
    universe_layout = QVBoxLayout()
    universe_layout.setSpacing(8)

    universe_controls = QHBoxLayout()
    universe_controls.setSpacing(8)

    self.universe_search_input = QLineEdit()
    self.universe_search_input.setPlaceholderText("Search all stocks by code or name...")
    self.universe_search_input.textChanged.connect(self._filter_universe_list)
    universe_controls.addWidget(self.universe_search_input, 1)

    universe_refresh_btn = QPushButton("Refresh")
    universe_refresh_btn.setObjectName("smallGhostButton")
    universe_refresh_btn.clicked.connect(
        lambda _checked=False: self._refresh_universe_catalog(force=True)
    )
    universe_controls.addWidget(universe_refresh_btn)
    universe_layout.addLayout(universe_controls)

    self.universe_status_label = QLabel("Universe: loading...")
    self.universe_status_label.setObjectName("metaLabel")
    universe_layout.addWidget(self.universe_status_label)

    self.universe_list = QListWidget()
    self.universe_list.setMinimumHeight(210)
    self.universe_list.itemClicked.connect(self._on_universe_item_activated)
    self.universe_list.itemActivated.connect(self._on_universe_item_activated)
    universe_layout.addWidget(self.universe_list)

    universe_hint = QLabel(
        "Click a stock to load chart data and AI guess for the selected date."
    )
    universe_hint.setObjectName("metaLabel")
    universe_hint.setWordWrap(True)
    universe_layout.addWidget(universe_hint)

    universe_group.setLayout(universe_layout)
    layout.addWidget(universe_group)

    settings_group = QGroupBox("Analysis Settings")
    settings_layout = QGridLayout()
    settings_layout.setHorizontalSpacing(10)
    settings_layout.setVerticalSpacing(9)

    self.capital_spin = QDoubleSpinBox()
    self.capital_spin.setRange(10000, 100000000)
    self.capital_spin.setValue(CONFIG.CAPITAL)
    self.capital_spin.setPrefix("CNY ")
    self._add_labeled(settings_layout, 0, "Capital:", self.capital_spin)

    self.interval_combo = QComboBox()
    self.interval_combo.addItems(["1m", "5m", "15m", "30m", "60m", "1d"])
    self.interval_combo.setCurrentText("1m")
    self.interval_combo.currentTextChanged.connect(
        self._on_interval_changed
    )
    self._add_labeled(settings_layout, 1, "Interval:", self.interval_combo)

    self.forecast_spin = QSpinBox()
    self.forecast_spin.setRange(5, 120)
    self.forecast_spin.setValue(self.GUESS_FORECAST_BARS)
    self.forecast_spin.setSuffix(" bars")
    self.forecast_spin.setToolTip("Number of bars to forecast ahead (actual time depends on interval)")
    self._add_labeled(settings_layout, 2, "Forecast:", self.forecast_spin)

    self.lookback_spin = QSpinBox()
    self.lookback_spin.setRange(7, 5000)
    self.lookback_spin.setValue(self._recommended_lookback("1m"))
    self.lookback_spin.setSuffix(" bars")
    self.lookback_spin.setToolTip("Historical bars to use for analysis")
    self._add_labeled(settings_layout, 3, "Lookback:", self.lookback_spin)

    settings_group.setLayout(settings_layout)
    layout.addWidget(settings_group)

    ai_group = QGroupBox("AI Model")
    ai_layout = QVBoxLayout()

    self.model_status = QLabel("Model: Loading...")
    ai_layout.addWidget(self.model_status)

    self.model_info = QLabel("")
    self.model_info.setObjectName("metaLabel")
    ai_layout.addWidget(self.model_info)

    self.train_btn = QPushButton("Train Model")
    self.train_btn.clicked.connect(self._start_training)
    ai_layout.addWidget(self.train_btn)

    self.auto_learn_btn = QPushButton("Continue Learning")
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
    """Create right panel with sentiment, news, and logs."""
    panel = QWidget()
    panel.setObjectName("rightPanel")
    layout = QVBoxLayout(panel)
    layout.setSpacing(12)
    layout.setContentsMargins(4, 4, 4, 4)

    self.right_tabs = QTabWidget()
    self.right_tabs.setDocumentMode(True)
    tabs = self.right_tabs

    # Sentiment Analysis Tab (replaces Portfolio)
    sentiment_tab = QWidget()
    sentiment_layout = QVBoxLayout(sentiment_tab)
    sentiment_layout.setSpacing(10)
    sentiment_layout.setContentsMargins(8, 8, 8, 8)

    # Sentiment summary labels
    self.sentiment_labels = {}
    labels = [
        ('overall', 'Overall Sentiment', 0, 0),
        ('policy', 'Policy Impact', 0, 1),
        ('market', 'Market Sentiment', 1, 0),
        ('confidence', 'Confidence', 1, 1),
    ]
    sentiment_frame, self.sentiment_labels = self._build_stat_frame(
        labels,
        (
            f"color: {ModernColors.ACCENT_INFO}; "
            f"font-size: {ModernFonts.SIZE_XXL}px; "
            f"font-weight: {ModernFonts.WEIGHT_BOLD};"
        ),
        15,
    )
    sentiment_layout.addWidget(sentiment_frame)

    # Sentiment chart placeholder
    self.sentiment_chart_label = QLabel("Sentiment Trend (Last 7 Days)")
    self.sentiment_chart_label.setObjectName("metaLabel")
    sentiment_layout.addWidget(self.sentiment_chart_label)

    # Entity mentions table
    self.entities_table = self._make_table(
        ["Entity", "Type", "Sentiment", "Mentions"]
    )
    sentiment_layout.addWidget(self.entities_table)

    # Refresh button
    refresh_btn = QPushButton("Refresh Sentiment")
    refresh_btn.clicked.connect(self._refresh_sentiment)
    sentiment_layout.addWidget(refresh_btn)

    tabs.addTab(sentiment_tab, "Sentiment Analysis")

    # News and Policy Tab
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

    # Live Signals Tab
    signals_tab = QWidget()
    signals_layout = QVBoxLayout(signals_tab)
    signals_layout.setContentsMargins(8, 8, 8, 8)
    self.signals_table = self._make_table([
        "Time", "Code", "Signal", "Confidence", "Price", "Action"
    ])
    signals_layout.addWidget(self.signals_table)
    tabs.addTab(signals_tab, "Live Signals")

    # History Tab
    history_tab = QWidget()
    history_layout = QVBoxLayout(history_tab)
    history_layout.setContentsMargins(8, 8, 8, 8)
    self.history_table = self._make_table([
        "Time", "Code", "Signal", "Prob UP", "Confidence", "Result"
    ])
    history_layout.addWidget(self.history_table)
    tabs.addTab(history_tab, "History")

    layout.addWidget(tabs, 4)

    # Log Group
    log_group = QGroupBox("System Log")
    log_group.setObjectName("systemLogGroup")
    log_layout = QVBoxLayout()
    try:
        from .widgets import LogWidget
        self.log_widget = LogWidget()
    except ImportError:
        self.log_widget = QTextEdit()
        self.log_widget.setReadOnly(True)
        self.log_widget.setMinimumHeight(220)
        self.log_widget.setMaximumHeight(380)
    if hasattr(self.log_widget, "setMinimumHeight"):
        self.log_widget.setMinimumHeight(220)
    if hasattr(self.log_widget, "setMaximumHeight"):
        self.log_widget.setMaximumHeight(380)
    log_layout.addWidget(self.log_widget)
    log_group.setLayout(log_layout)
    layout.addWidget(log_group, 2)

    return panel



def _apply_professional_style(self: Any) -> None:
    """Apply a modern, professional trading platform theme with enhanced UX."""
    self.setFont(QFont(get_primary_font_family(), ModernFonts.SIZE_BASE))
    self.setStyleSheet(get_app_stylesheet())
