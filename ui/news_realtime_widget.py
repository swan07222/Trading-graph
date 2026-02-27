# ui/news_realtime_widget.py
"""Real-time news widget for the PyQt6 desktop UI."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from PyQt6.QtCore import QTimer, QUrl, pyqtSignal
from PyQt6.QtGui import QDesktopServices
from PyQt6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ui.modern_theme import (
    ModernColors,
    ModernFonts,
    get_display_font_family,
    get_primary_font_family,
)
from utils.logger import get_logger

log = get_logger(__name__)


class NewsListItemWidget(QWidget):
    """Single news list item with sentiment indicator."""

    def __init__(self, article_data: dict[str, Any], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.article_data = article_data
        self._setup_ui()

    def _setup_ui(self) -> None:
        """Build the item content."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(8)

        title = str(self.article_data.get("title", "No title"))
        self.title_label = QLabel(title)
        self.title_label.setObjectName("newsTitle")
        self.title_label.setWordWrap(True)
        layout.addWidget(self.title_label)

        sentiment = float(self.article_data.get("sentiment_score", 0.0))
        self._apply_sentiment_style(sentiment)

        meta_layout = QHBoxLayout()
        meta_layout.setSpacing(10)

        source = str(self.article_data.get("source", "Unknown"))
        self.source_label = QLabel(f"Source: {source}")
        self.source_label.setObjectName("newsMeta")
        meta_layout.addWidget(self.source_label)

        time_text = "Time: --:--"
        published_at = self.article_data.get("published_at", "")
        if published_at:
            try:
                dt = datetime.fromisoformat(str(published_at).replace("Z", "+00:00"))
                time_text = f"Time: {dt.strftime('%H:%M')}"
            except Exception:
                pass
        self.time_label = QLabel(time_text)
        self.time_label.setObjectName("newsMeta")
        meta_layout.addWidget(self.time_label)

        category = str(self.article_data.get("category", "market")).strip().lower()
        self.category_label = QLabel(f"Category: {category.capitalize()}")
        self.category_label.setObjectName("newsMeta")
        meta_layout.addWidget(self.category_label)

        meta_layout.addStretch()
        layout.addLayout(meta_layout)

        summary = str(self.article_data.get("summary", "")).strip()
        if summary and len(summary) > 140:
            summary = f"{summary[:137]}..."
        self.summary_label = QLabel(summary)
        self.summary_label.setObjectName("newsSummary")
        self.summary_label.setWordWrap(True)
        if summary:
            layout.addWidget(self.summary_label)

        tags = self.article_data.get("tags", [])
        if isinstance(tags, list) and tags:
            tags_layout = QHBoxLayout()
            tags_layout.setSpacing(6)
            for tag in tags[:3]:
                tag_text = str(tag).strip()
                if not tag_text:
                    continue
                tag_label = QLabel(f"#{tag_text}")
                tag_label.setObjectName("newsTag")
                tags_layout.addWidget(tag_label)
            tags_layout.addStretch()
            layout.addLayout(tags_layout)

    def _apply_sentiment_style(self, sentiment: float) -> None:
        """Apply sentiment colors to card borders/background."""
        primary_font = get_primary_font_family()
        display_font = get_display_font_family()

        if sentiment > 0.3:
            bg_color = ModernColors.SIGNAL_BUY_BG
            accent_color = ModernColors.SIGNAL_BUY
            hover_color = "rgba(54, 211, 164, 0.24)"
        elif sentiment < -0.3:
            bg_color = ModernColors.SIGNAL_SELL_BG
            accent_color = ModernColors.SIGNAL_SELL
            hover_color = "rgba(255, 108, 128, 0.24)"
        else:
            bg_color = "rgba(136, 160, 196, 0.14)"
            accent_color = ModernColors.BORDER_DEFAULT
            hover_color = "rgba(136, 160, 196, 0.22)"

        self.setStyleSheet(
            f"""
            NewsListItemWidget {{
                background: {bg_color};
                border: 1px solid {ModernColors.BORDER_SUBTLE};
                border-left: 4px solid {accent_color};
                border-radius: 12px;
                margin: 2px 0px;
            }}
            NewsListItemWidget:hover {{
                background: {hover_color};
                border-color: {ModernColors.BORDER_DEFAULT};
            }}
            QLabel#newsTitle {{
                color: {ModernColors.TEXT_STRONG};
                font-family: "{display_font}";
                font-size: {ModernFonts.SIZE_BASE}px;
                font-weight: {ModernFonts.WEIGHT_SEMIBOLD};
            }}
            QLabel#newsMeta {{
                color: {ModernColors.TEXT_SECONDARY};
                font-family: "{primary_font}";
                font-size: {ModernFonts.SIZE_XS}px;
                font-weight: {ModernFonts.WEIGHT_MEDIUM};
            }}
            QLabel#newsSummary {{
                color: {ModernColors.TEXT_MUTED};
                font-family: "{primary_font}";
                font-size: {ModernFonts.SIZE_SM}px;
            }}
            QLabel#newsTag {{
                color: {ModernColors.TEXT_PRIMARY};
                background: rgba(52, 209, 255, 0.16);
                border: 1px solid {ModernColors.BORDER_SUBTLE};
                border-radius: 8px;
                padding: 2px 7px;
                font-family: "{primary_font}";
                font-size: {ModernFonts.SIZE_XS}px;
                font-weight: {ModernFonts.WEIGHT_MEDIUM};
            }}
            """
        )


class RealTimeNewsWidget(QWidget):
    """Real-time news feed widget with channel filtering and live updates."""

    article_clicked = pyqtSignal(object)
    sentiment_updated = pyqtSignal(str, float)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._current_channel = "all"
        self._articles: list[dict[str, Any]] = []
        self._max_articles = 200
        self._auto_scroll = True
        self._websocket_client = None

        self._setup_ui()
        self._setup_timers()

    def _setup_ui(self) -> None:
        """Setup widget UI."""
        self.setObjectName("realTimeNewsRoot")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        toolbar = self._create_toolbar()
        layout.addWidget(toolbar)

        self.news_list = QListWidget()
        self.news_list.setObjectName("newsList")
        self.news_list.setAlternatingRowColors(False)
        self.news_list.setSpacing(6)
        self.news_list.itemClicked.connect(self._on_item_clicked)
        layout.addWidget(self.news_list, 1)

        self.status_bar = QWidget()
        self.status_bar.setObjectName("newsStatusBar")
        status_layout = QHBoxLayout(self.status_bar)
        status_layout.setContentsMargins(10, 6, 10, 6)
        status_layout.setSpacing(8)

        self.status_label = QLabel("Disconnected")
        self.status_label.setObjectName("statusLabel")
        status_layout.addWidget(self.status_label)

        status_layout.addStretch()

        self.count_label = QLabel("0/0 articles")
        self.count_label.setObjectName("countLabel")
        status_layout.addWidget(self.count_label)

        layout.addWidget(self.status_bar)
        self._apply_widget_styles()
        self._set_connection_status(False)

    def _create_toolbar(self) -> QWidget:
        """Create toolbar with filters and quick actions."""
        toolbar_widget = QWidget()
        toolbar_widget.setObjectName("newsToolbar")
        toolbar_layout = QHBoxLayout(toolbar_widget)
        toolbar_layout.setContentsMargins(10, 8, 10, 8)
        toolbar_layout.setSpacing(8)

        channel_label = QLabel("Channel")
        channel_label.setObjectName("toolbarLabel")
        toolbar_layout.addWidget(channel_label)

        self.channel_combo = QComboBox()
        self.channel_combo.setObjectName("channelCombo")
        self.channel_combo.addItems(["all", "policy", "market", "company", "regulatory"])
        self.channel_combo.setCurrentText("all")
        self.channel_combo.currentTextChanged.connect(self._on_channel_changed)
        toolbar_layout.addWidget(self.channel_combo)

        self.search_box = QLineEdit()
        self.search_box.setObjectName("searchBox")
        self.search_box.setPlaceholderText("Search news by title or summary")
        self.search_box.textChanged.connect(self._on_search_changed)
        toolbar_layout.addWidget(self.search_box, 1)

        self.clear_btn = QPushButton("Clear")
        self.clear_btn.setObjectName("secondaryButton")
        self.clear_btn.clicked.connect(self.clear)
        toolbar_layout.addWidget(self.clear_btn)

        self.auto_scroll_check = QPushButton("Auto-scroll: ON")
        self.auto_scroll_check.setObjectName("toggleButton")
        self.auto_scroll_check.setCheckable(True)
        self.auto_scroll_check.setChecked(True)
        self.auto_scroll_check.clicked.connect(self._toggle_auto_scroll)
        toolbar_layout.addWidget(self.auto_scroll_check)

        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.setObjectName("accentButton")
        self.refresh_btn.clicked.connect(self._manual_refresh)
        toolbar_layout.addWidget(self.refresh_btn)

        return toolbar_widget

    def _apply_widget_styles(self) -> None:
        """Apply modern visual styling for the container and controls."""
        primary_font = get_primary_font_family()
        display_font = get_display_font_family()

        self.setStyleSheet(
            f"""
            QWidget#newsToolbar {{
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:0,
                    stop:0 #1a3454,
                    stop:1 #12253f
                );
                border: 1px solid {ModernColors.BORDER_SUBTLE};
                border-radius: 12px;
            }}
            QLabel#toolbarLabel {{
                color: {ModernColors.TEXT_SECONDARY};
                font-family: "{primary_font}";
                font-size: {ModernFonts.SIZE_XS}px;
                font-weight: {ModernFonts.WEIGHT_MEDIUM};
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }}
            QComboBox#channelCombo,
            QLineEdit#searchBox {{
                color: {ModernColors.TEXT_PRIMARY};
                background: #0d1e35;
                border: 1px solid {ModernColors.BORDER_SUBTLE};
                border-radius: 8px;
                padding: 6px 10px;
                font-family: "{primary_font}";
                font-size: {ModernFonts.SIZE_SM}px;
            }}
            QComboBox#channelCombo:hover,
            QLineEdit#searchBox:hover {{
                border-color: {ModernColors.BORDER_DEFAULT};
            }}
            QComboBox#channelCombo:focus,
            QLineEdit#searchBox:focus {{
                border: 1px solid {ModernColors.BORDER_FOCUS};
            }}
            QPushButton {{
                color: {ModernColors.TEXT_PRIMARY};
                background: #17304f;
                border: 1px solid {ModernColors.BORDER_SUBTLE};
                border-radius: 8px;
                padding: 6px 10px;
                font-family: "{primary_font}";
                font-size: {ModernFonts.SIZE_SM}px;
                font-weight: {ModernFonts.WEIGHT_MEDIUM};
            }}
            QPushButton:hover {{
                background: #1e3b61;
                border-color: {ModernColors.BORDER_DEFAULT};
            }}
            QPushButton:pressed {{
                background: #152d49;
            }}
            QPushButton#accentButton {{
                background: {ModernColors.GRADIENT_PRIMARY};
                color: #042337;
                border: 1px solid #5de4ff;
                font-weight: {ModernFonts.WEIGHT_SEMIBOLD};
            }}
            QPushButton#accentButton:hover {{
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:0,
                    stop:0 #48d7ff,
                    stop:1 #7be8ff
                );
            }}
            QPushButton#toggleButton:checked {{
                background: rgba(54, 211, 164, 0.24);
                border-color: {ModernColors.SIGNAL_BUY};
            }}
            QListWidget#newsList {{
                background: rgba(9, 21, 38, 0.86);
                border: 1px solid {ModernColors.BORDER_SUBTLE};
                border-radius: 12px;
                padding: 8px;
            }}
            QListWidget#newsList::item {{
                border: none;
                margin: 0px;
                padding: 2px;
            }}
            QListWidget#newsList::item:selected {{
                background: rgba(52, 209, 255, 0.17);
                border: 1px solid {ModernColors.BORDER_FOCUS};
                border-radius: 10px;
            }}
            QWidget#newsStatusBar {{
                background: rgba(20, 38, 62, 0.9);
                border: 1px solid {ModernColors.BORDER_SUBTLE};
                border-radius: 10px;
            }}
            QLabel#statusLabel {{
                color: {ModernColors.TEXT_SECONDARY};
                font-family: "{display_font}";
                font-size: {ModernFonts.SIZE_SM}px;
                font-weight: {ModernFonts.WEIGHT_SEMIBOLD};
            }}
            QLabel#countLabel {{
                color: {ModernColors.TEXT_MUTED};
                font-family: "{primary_font}";
                font-size: {ModernFonts.SIZE_XS}px;
                font-weight: {ModernFonts.WEIGHT_MEDIUM};
            }}
            """
        )

    def _setup_timers(self) -> None:
        """Setup update timers."""
        self.poll_timer = QTimer()
        self.poll_timer.timeout.connect(self._poll_for_updates)
        self.poll_timer.setInterval(30000)

        self.status_timer = QTimer()
        self.status_timer.timeout.connect(self._update_status)
        self.status_timer.setInterval(5000)

    def set_websocket_client(self, client: Any) -> None:
        """Set WebSocket client for real-time updates."""
        self._websocket_client = client

        if client:
            client.message_received.connect(self._on_websocket_message)
            client.connected.connect(self._on_websocket_connected)
            client.disconnected.connect(self._on_websocket_disconnected)

        self.poll_timer.start()
        self.status_timer.start()

    def subscribe_to_channel(self, channel: str) -> None:
        """Subscribe to a news channel."""
        self._current_channel = channel
        self.channel_combo.setCurrentText(channel)

        if self._websocket_client and self._websocket_client.is_connected:
            self._websocket_client.send(
                {
                    "type": "subscribe",
                    "channel": channel,
                }
            )

        log.info(f"Subscribed to news channel: {channel}")

    def clear(self) -> None:
        """Clear all loaded articles and list widgets."""
        self._articles.clear()
        self.news_list.clear()
        self._update_count()

    def _add_article(self, article_data: dict[str, Any]) -> None:
        """Add one article to memory and list, deduplicating by non-empty id."""
        article_id = str(article_data.get("id", "")).strip()
        if article_id:
            for existing in self._articles:
                if str(existing.get("id", "")).strip() == article_id:
                    return

        self._articles.append(article_data)

        list_was_trimmed = False
        if len(self._articles) > self._max_articles:
            self._articles = self._articles[-self._max_articles:]
            list_was_trimmed = True

        if list_was_trimmed:
            self._refresh_list()
        else:
            self._add_article_to_list(article_data)

        self.article_clicked.emit(article_data)

        sentiment = float(article_data.get("sentiment_score", 0.0))
        entities = article_data.get("entities", [])
        for entity in entities:
            self.sentiment_updated.emit(str(entity), sentiment)

    def _insert_article_widget(self, article_data: dict[str, Any]) -> None:
        """Insert a prepared article card into the list widget."""
        item = QListWidgetItem()
        widget = NewsListItemWidget(article_data)
        item.setSizeHint(widget.sizeHint())
        self.news_list.addItem(item)
        self.news_list.setItemWidget(item, widget)

    def _add_article_to_list(self, article_data: dict[str, Any]) -> None:
        """Add article to QListWidget with channel/search filters."""
        channel = article_data.get("category", "market")
        if self._current_channel != "all" and channel != self._current_channel:
            return

        if "_title_lower" not in article_data:
            article_data["_title_lower"] = str(article_data.get("title", "")).lower()
        if "_summary_lower" not in article_data:
            article_data["_summary_lower"] = str(article_data.get("summary", "")).lower()

        search_text = self.search_box.text().lower().strip()
        if search_text:
            title = article_data.get("_title_lower", "")
            summary = article_data.get("_summary_lower", "")
            if search_text not in title and search_text not in summary:
                return

        self._insert_article_widget(article_data)

        if self._auto_scroll:
            self.news_list.scrollToBottom()

        self._update_count()

    def _on_item_clicked(self, item: QListWidgetItem) -> None:
        """Open article URL in browser when card is clicked."""
        widget = self.news_list.itemWidget(item)
        if widget and hasattr(widget, "article_data"):
            article_data = widget.article_data
            url = article_data.get("url", "")
            if url:
                QDesktopServices.openUrl(QUrl(url))

    def _on_channel_changed(self, channel: str) -> None:
        """Handle channel filter changes."""
        self._current_channel = channel
        self._refresh_list()

        if self._websocket_client and self._websocket_client.is_connected:
            self._websocket_client.send(
                {
                    "type": "subscribe",
                    "channel": channel,
                }
            )

    def _on_search_changed(self, text: str) -> None:
        """Debounced search to avoid UI lag during typing."""
        _ = text

        if hasattr(self, "_search_debounce_timer"):
            self._search_debounce_timer.stop()

        if not hasattr(self, "_search_debounce_timer"):
            self._search_debounce_timer = QTimer()
            self._search_debounce_timer.setSingleShot(True)
            self._search_debounce_timer.timeout.connect(self._execute_search)

        self._search_debounce_timer.start(300)

    def _execute_search(self) -> None:
        """Execute the actual search after debounce."""
        self._refresh_list()

    def _refresh_list(self) -> None:
        """Refresh list with current filters."""
        self.news_list.clear()

        search_text = self.search_box.text().lower().strip() if self.search_box else ""
        channel_filter = self._current_channel if self._current_channel != "all" else None

        for article_data in self._articles:
            if channel_filter:
                article_channel = article_data.get("category", "market")
                if article_channel != channel_filter:
                    continue

            if search_text:
                title_lower = article_data.get("_title_lower", "")
                summary_lower = article_data.get("_summary_lower", "")

                if not title_lower:
                    title_lower = str(article_data.get("title", "")).lower()
                    article_data["_title_lower"] = title_lower
                if not summary_lower:
                    summary_lower = str(article_data.get("summary", "")).lower()
                    article_data["_summary_lower"] = summary_lower

                if search_text not in title_lower and search_text not in summary_lower:
                    continue

            self._insert_article_widget(article_data)

        if self._auto_scroll and self.news_list.count() > 0:
            self.news_list.scrollToBottom()

        self._update_count()

    def _toggle_auto_scroll(self, checked: bool) -> None:
        """Toggle auto-scroll."""
        self._auto_scroll = checked
        self.auto_scroll_check.setText(f"Auto-scroll: {'ON' if checked else 'OFF'}")

    def _on_websocket_message(self, message: Any) -> None:
        """Handle incoming websocket messages."""
        if message.type == "news":
            article_data = message.data
            self._add_article(article_data)

    def _set_connection_status(self, connected: bool) -> None:
        """Update status label text/color."""
        if connected:
            self.status_label.setText("Connected")
            self.status_label.setStyleSheet(f"color: {ModernColors.ACCENT_SUCCESS};")
        else:
            self.status_label.setText("Disconnected")
            self.status_label.setStyleSheet(f"color: {ModernColors.ACCENT_DANGER};")

    def _on_websocket_connected(self) -> None:
        """Handle websocket connection."""
        self._set_connection_status(True)

        if self._websocket_client:
            self._websocket_client.send(
                {
                    "type": "subscribe",
                    "channel": self._current_channel,
                }
            )
            self._websocket_client.send(
                {
                    "type": "get_backlog",
                    "limit": 50,
                    "channel": self._current_channel,
                }
            )

    def _on_websocket_disconnected(self) -> None:
        """Handle websocket disconnection."""
        self._set_connection_status(False)

    def _poll_for_updates(self) -> None:
        """Poll for news updates if websocket is unavailable."""
        pass

    def _update_status(self) -> None:
        """Update periodic status diagnostics."""
        if self._websocket_client and self._websocket_client.is_connected:
            stats = self._websocket_client.stats
            log.debug(f"WebSocket stats: {stats}")

    def _update_count(self) -> None:
        """Update visible and total article counter."""
        visible_count = self.news_list.count()
        total_count = len(self._articles)
        self.count_label.setText(f"{visible_count}/{total_count} articles")

    def _manual_refresh(self) -> None:
        """Manual refresh request from UI."""
        if self._websocket_client and self._websocket_client.is_connected:
            self._websocket_client.send(
                {
                    "type": "get_backlog",
                    "limit": 50,
                    "channel": self._current_channel,
                }
            )
        else:
            self._poll_for_updates()

    def get_current_articles(self) -> list[dict[str, Any]]:
        """Get a copy of all loaded articles."""
        return self._articles.copy()

    def closeEvent(self, event) -> None:
        """Handle widget close and stop timers."""
        self.poll_timer.stop()
        self.status_timer.stop()
        super().closeEvent(event)
