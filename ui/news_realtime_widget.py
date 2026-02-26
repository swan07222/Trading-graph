# ui/news_realtime_widget.py
"""Real-Time News Widget for PyQt6 UI.

This module provides:
- Real-time news feed widget with auto-updates
- Category filtering and search
- Sentiment color coding
- Click-to-open article URLs
- WebSocket-based live updates

Usage:
    from ui.news_realtime_widget import RealTimeNewsWidget
    
    widget = RealTimeNewsWidget(parent)
    widget.subscribe_to_channel("market")
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

from PyQt6.QtCore import (
    QUrl,
    Qt,
    QTimer,
    pyqtSignal,
)
from PyQt6.QtGui import (
    QAction,
    QDesktopServices,
    QFont,
    QPalette,
)
from PyQt6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSizePolicy,
    QToolBar,
    QVBoxLayout,
    QWidget,
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
        """Setup the widget UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(4)
        
        # Title
        title = self.article_data.get("title", "No title")
        self.title_label = QLabel(title)
        self.title_label.setWordWrap(True)
        self.title_label.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        
        # Apply sentiment color
        sentiment = float(self.article_data.get("sentiment_score", 0))
        self._apply_sentiment_style(sentiment)
        
        # Metadata row
        meta_layout = QHBoxLayout()
        meta_layout.setSpacing(12)
        
        # Source
        source = self.article_data.get("source", "Unknown")
        self.source_label = QLabel(f"📰 {source}")
        self.source_label.setFont(QFont("Arial", 8))
        meta_layout.addWidget(self.source_label)
        
        # Time
        published_at = self.article_data.get("published_at", "")
        if published_at:
            try:
                dt = datetime.fromisoformat(published_at.replace("Z", "+00:00"))
                time_str = dt.strftime("%H:%M")
                self.time_label = QLabel(f"🕐 {time_str}")
                self.time_label.setFont(QFont("Arial", 8))
                meta_layout.addWidget(self.time_label)
            except Exception:
                pass
        
        # Category
        category = self.article_data.get("category", "market")
        category_icon = {
            "policy": "📋",
            "market": "📊",
            "company": "🏢",
            "regulatory": "⚖️",
        }.get(category, "📄")
        self.category_label = QLabel(f"{category_icon} {category.capitalize()}")
        self.category_label.setFont(QFont("Arial", 8))
        meta_layout.addWidget(self.category_label)
        
        meta_layout.addStretch()
        layout.addLayout(meta_layout)
        
        # Summary
        summary = self.article_data.get("summary", "")
        if summary and len(summary) > 100:
            summary = summary[:100] + "..."
        self.summary_label = QLabel(summary)
        self.summary_label.setWordWrap(True)
        self.summary_label.setFont(QFont("Arial", 9))
        self.summary_label.setStyleSheet("color: #666;")
        layout.addWidget(self.summary_label)
        
        # Tags
        tags = self.article_data.get("tags", [])
        if tags:
            tags_layout = QHBoxLayout()
            tags_layout.setSpacing(4)
            for tag in tags[:3]:
                tag_label = QLabel(f"#{tag}")
                tag_label.setFont(QFont("Arial", 7))
                tag_label.setStyleSheet(
                    "background: #e0e0e0; padding: 2px 6px; border-radius: 3px;"
                )
                tags_layout.addWidget(tag_label)
            tags_layout.addStretch()
            layout.addLayout(tags_layout)
        
        self.setLayout(layout)
    
    def _apply_sentiment_style(self, sentiment: float) -> None:
        """Apply color based on sentiment score."""
        # Normalize sentiment to 0-1 range
        normalized = (sentiment + 1) / 2  # -1..1 -> 0..1
        
        if sentiment > 0.3:
            # Positive - green
            intensity = min(255, int(155 + normalized * 100))
            bg_color = f"rgb(230, {255 - int(normalized * 50)}, 230)"
            border_color = f"rgb(0, {int(normalized * 200)}, 0)"
        elif sentiment < -0.3:
            # Negative - red
            intensity = min(255, int(155 + (1 - normalized) * 100))
            bg_color = f"rgb({255 - int(normalized * 50)}, 230, 230)"
            border_color = f"rgb({int((1 - normalized) * 200)}, 0, 0)"
        else:
            # Neutral - gray
            bg_color = "rgb(245, 245, 245)"
            border_color = "rgb(180, 180, 180)"
        
        self.setStyleSheet(f"""
            NewsListItemWidget {{
                background: {bg_color};
                border-left: 4px solid {border_color};
                border-radius: 4px;
                margin: 2px;
            }}
            NewsListItemWidget:hover {{
                background: rgb(230, 240, 255);
            }}
        """)


class RealTimeNewsWidget(QWidget):
    """Real-time news feed widget.
    
    Features:
    - Live news updates via WebSocket
    - Category filtering
    - Search functionality
    - Sentiment color coding
    - Auto-scroll option
    
    Signals:
        article_clicked: Emitted when article is clicked (article_data)
        sentiment_updated: Emitted when sentiment changes (stock_code, sentiment)
    """
    
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
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Toolbar
        toolbar = self._create_toolbar()
        layout.addWidget(toolbar)
        
        # News list
        self.news_list = QListWidget()
        self.news_list.setAlternatingRowColors(True)
        self.news_list.setStyleSheet("""
            QListWidget {
                border: 1px solid #ccc;
                background: white;
            }
            QListWidget::item {
                padding: 4px;
                border-bottom: 1px solid #eee;
            }
            QListWidget::item:selected {
                background: #0078d4;
                color: white;
            }
        """)
        self.news_list.itemClicked.connect(self._on_item_clicked)
        layout.addWidget(self.news_list)
        
        # Status bar
        status_layout = QHBoxLayout()
        
        self.status_label = QLabel("⚪ Disconnected")
        self.status_label.setFont(QFont("Arial", 8))
        status_layout.addWidget(self.status_label)
        
        status_layout.addStretch()
        
        self.count_label = QLabel("0 articles")
        self.count_label.setFont(QFont("Arial", 8))
        status_layout.addWidget(self.count_label)
        
        layout.addLayout(status_layout)
        
        self.setLayout(layout)
    
    def _create_toolbar(self) -> QWidget:
        """Create toolbar with filters."""
        toolbar_widget = QWidget()
        toolbar_layout = QHBoxLayout(toolbar_widget)
        toolbar_layout.setContentsMargins(4, 4, 4, 4)
        
        # Channel selector
        toolbar_layout.addWidget(QLabel("Channel:"))
        self.channel_combo = QComboBox()
        self.channel_combo.addItems(["all", "policy", "market", "company", "regulatory"])
        self.channel_combo.setCurrentText("all")
        self.channel_combo.currentTextChanged.connect(self._on_channel_changed)
        toolbar_layout.addWidget(self.channel_combo)
        
        # Search box
        self.search_box = QLineEdit()
        self.search_box.setPlaceholderText("Search news...")
        self.search_box.textChanged.connect(self._on_search_changed)
        toolbar_layout.addWidget(self.search_box)
        
        # Clear button
        clear_btn = QPushButton("🗑 Clear")
        clear_btn.clicked.connect(self.clear)
        toolbar_layout.addWidget(clear_btn)
        
        # Auto-scroll toggle
        self.auto_scroll_check = QPushButton("📜 Auto-scroll: ON")
        self.auto_scroll_check.setCheckable(True)
        self.auto_scroll_check.setChecked(True)
        self.auto_scroll_check.clicked.connect(self._toggle_auto_scroll)
        toolbar_layout.addWidget(self.auto_scroll_check)
        
        # Refresh button
        refresh_btn = QPushButton("🔄 Refresh")
        refresh_btn.clicked.connect(self._manual_refresh)
        toolbar_layout.addWidget(refresh_btn)
        
        return toolbar_widget
    
    def _setup_timers(self) -> None:
        """Setup update timers."""
        # Poll for updates (fallback if WebSocket not available)
        self.poll_timer = QTimer()
        self.poll_timer.timeout.connect(self._poll_for_updates)
        self.poll_timer.setInterval(30000)  # 30 seconds
        
        # Status update timer
        self.status_timer = QTimer()
        self.status_timer.timeout.connect(self._update_status)
        self.status_timer.setInterval(5000)
        
    def set_websocket_client(self, client: Any) -> None:
        """Set WebSocket client for real-time updates.
        
        Args:
            client: WebSocketClient instance
        """
        self._websocket_client = client
        
        # Connect signals
        if client:
            client.message_received.connect(self._on_websocket_message)
            client.connected.connect(self._on_websocket_connected)
            client.disconnected.connect(self._on_websocket_disconnected)
        
        # Start polling as fallback
        self.poll_timer.start()
        self.status_timer.start()
    
    def subscribe_to_channel(self, channel: str) -> None:
        """Subscribe to news channel.
        
        Args:
            channel: Channel name (all, policy, market, company, regulatory)
        """
        self._current_channel = channel
        self.channel_combo.setCurrentText(channel)
        
        # Send subscription via WebSocket
        if self._websocket_client and self._websocket_client.is_connected:
            self._websocket_client.send({
                "type": "subscribe",
                "channel": channel,
            })
        
        log.info(f"Subscribed to news channel: {channel}")
    
    def clear(self) -> None:
        """Clear all articles."""
        self._articles.clear()
        self.news_list.clear()
        self._update_count()
    
    def _add_article(self, article_data: dict[str, Any]) -> None:
        """Add article to list.
        
        Args:
            article_data: Article dictionary
        """
        # Check for duplicates
        article_id = article_data.get("id", "")
        for existing in self._articles:
            if existing.get("id") == article_id:
                return  # Duplicate
        
        # Add to list
        self._articles.append(article_data)
        
        # Trim if needed
        if len(self._articles) > self._max_articles:
            self._articles = self._articles[-self._max_articles:]
        
        # Add to UI
        self._add_article_to_list(article_data)
        
        # Emit signal
        self.article_clicked.emit(article_data)
        
        # Update sentiment
        sentiment = float(article_data.get("sentiment_score", 0))
        entities = article_data.get("entities", [])
        for entity in entities:
            self.sentiment_updated.emit(str(entity), sentiment)
    
    def _add_article_to_list(self, article_data: dict[str, Any]) -> None:
        """Add article to QListWidget.
        
        China-optimized: Pre-caches lowercase fields for faster searching.

        Args:
            article_data: Article dictionary
        """
        # Filter by channel
        channel = article_data.get("category", "market")
        if self._current_channel != "all" and channel != self._current_channel:
            return

        # Pre-cache lowercase fields for faster search (China-optimized)
        if "_title_lower" not in article_data:
            article_data["_title_lower"] = article_data.get("title", "").lower()
        if "_summary_lower" not in article_data:
            article_data["_summary_lower"] = article_data.get("summary", "").lower()

        # Filter by search (using cached lowercase fields)
        search_text = self.search_box.text().lower()
        if search_text:
            title = article_data.get("_title_lower", "")
            summary = article_data.get("_summary_lower", "")
            if search_text not in title and search_text not in summary:
                return

        # Create widget
        item = QListWidgetItem(self.news_list)
        widget = NewsListItemWidget(article_data)
        item.setSizeHint(widget.sizeHint())
        self.news_list.addItem(item)
        self.news_list.setItemWidget(item, widget)

        # Auto-scroll to bottom
        if self._auto_scroll:
            self.news_list.scrollToBottom()

        self._update_count()
    
    def _on_item_clicked(self, item: QListWidgetItem) -> None:
        """Handle item click.
        
        Args:
            item: Clicked list item
        """
        widget = self.news_list.itemWidget(item)
        if widget and hasattr(widget, 'article_data'):
            article_data = widget.article_data
            url = article_data.get("url", "")
            
            if url:
                # Open URL in browser
                QDesktopServices.openUrl(QUrl(url))
    
    def _on_channel_changed(self, channel: str) -> None:
        """Handle channel change.
        
        Args:
            channel: New channel
        """
        self._current_channel = channel
        self._refresh_list()
        
        # Update WebSocket subscription
        if self._websocket_client and self._websocket_client.is_connected:
            self._websocket_client.send({
                "type": "subscribe",
                "channel": channel,
            })
    
    def _on_search_changed(self, text: str) -> None:
        """Handle search text change.
        
        China-optimized: Debounced search to reduce UI lag with large datasets.

        Args:
            text: Search text
        """
        # Debounce search to avoid lag during typing
        if hasattr(self, '_search_debounce_timer'):
            self._search_debounce_timer.stop()
        
        # Use QTimer for debouncing (300ms delay)
        if not hasattr(self, '_search_debounce_timer'):
            from PyQt6.QtCore import QTimer
            self._search_debounce_timer = QTimer()
            self._search_debounce_timer.setSingleShot(True)
            self._search_debounce_timer.timeout.connect(self._execute_search)
        
        self._search_debounce_timer.start(300)
    
    def _execute_search(self) -> None:
        """Execute the actual search after debounce."""
        self._refresh_list()

    def _refresh_list(self) -> None:
        """Refresh list with current filters.
        
        China-optimized: Improved search performance with pre-computed lowercase cache.
        """
        self.news_list.clear()

        # Pre-compute search text for efficiency
        search_text = self.search_box.text().lower().strip() if self.search_box else ""
        
        # Channel filter
        channel_filter = self._current_channel if self._current_channel != "all" else None
        
        for article_data in self._articles:
            # Fast channel filtering
            if channel_filter:
                article_channel = article_data.get("category", "market")
                if article_channel != channel_filter:
                    continue
            
            # Fast search filtering with early exit
            if search_text:
                # Check cached lowercase fields first if available
                title_lower = article_data.get("_title_lower", "")
                summary_lower = article_data.get("_summary_lower", "")
                
                # Cache lowercase versions if not already cached
                if not title_lower:
                    title_lower = article_data.get("title", "").lower()
                    article_data["_title_lower"] = title_lower
                if not summary_lower:
                    summary_lower = article_data.get("summary", "").lower()
                    article_data["_summary_lower"] = summary_lower
                
                if search_text not in title_lower and search_text not in summary_lower:
                    continue
            
            self._add_article_to_list(article_data)
    
    def _toggle_auto_scroll(self, checked: bool) -> None:
        """Toggle auto-scroll.
        
        Args:
            checked: Auto-scroll enabled
        """
        self._auto_scroll = checked
        self.auto_scroll_check.setText(
            f"📜 Auto-scroll: {'ON' if checked else 'OFF'}"
        )
    
    def _on_websocket_message(self, message: Any) -> None:
        """Handle WebSocket message.
        
        Args:
            message: WebSocketMessage object
        """
        if message.type == "news":
            article_data = message.data
            self._add_article(article_data)
    
    def _on_websocket_connected(self) -> None:
        """Handle WebSocket connection."""
        self.status_label.setText("🟢 Connected")
        self.status_label.setStyleSheet("color: green;")
        
        # Subscribe to channel
        if self._websocket_client:
            self._websocket_client.send({
                "type": "subscribe",
                "channel": self._current_channel,
            })
            
            # Request backlog
            self._websocket_client.send({
                "type": "get_backlog",
                "limit": 50,
                "channel": self._current_channel,
            })
    
    def _on_websocket_disconnected(self) -> None:
        """Handle WebSocket disconnection."""
        self.status_label.setText("🔴 Disconnected")
        self.status_label.setStyleSheet("color: red;")
    
    def _poll_for_updates(self) -> None:
        """Poll for news updates (fallback)."""
        # This is a fallback if WebSocket is not available
        # In production, you would fetch from an API endpoint
        pass
    
    def _update_status(self) -> None:
        """Update status display."""
        if self._websocket_client and self._websocket_client.is_connected:
            stats = self._websocket_client.stats
            log.debug(f"WebSocket stats: {stats}")
    
    def _update_count(self) -> None:
        """Update article count label."""
        visible_count = self.news_list.count()
        total_count = len(self._articles)
        self.count_label.setText(f"{visible_count}/{total_count} articles")
    
    def _manual_refresh(self) -> None:
        """Manual refresh request."""
        if self._websocket_client and self._websocket_client.is_connected:
            self._websocket_client.send({
                "type": "get_backlog",
                "limit": 50,
                "channel": self._current_channel,
            })
        else:
            # Trigger poll
            self._poll_for_updates()
    
    def get_current_articles(self) -> list[dict[str, Any]]:
        """Get current articles.
        
        Returns:
            List of article dictionaries
        """
        return self._articles.copy()
    
    def closeEvent(self, event) -> None:
        """Handle widget close."""
        self.poll_timer.stop()
        self.status_timer.stop()
        super().closeEvent(event)
