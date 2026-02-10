# ui/news_widget.py
"""
News Panel Widget for the Trading UI.
Shows real-time news with sentiment coloring.
"""
from datetime import datetime
from typing import List, Optional

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QTableWidget, QTableWidgetItem, QHeaderView,
    QPushButton, QGroupBox, QFrame, QProgressBar,
)
from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt6.QtGui import QColor, QFont

from utils.logger import get_logger

log = get_logger(__name__)


class NewsFetchThread(QThread):
    """Background thread to fetch news"""
    finished = pyqtSignal(list)  # List[NewsItem]
    sentiment_updated = pyqtSignal(dict)  # sentiment summary

    def __init__(self, stock_code: str = None, fetch_type: str = "market"):
        super().__init__()
        self.stock_code = stock_code
        self.fetch_type = fetch_type

    def run(self):
        try:
            from data.news import get_news_aggregator
            agg = get_news_aggregator()

            if self.fetch_type == "stock" and self.stock_code:
                news = agg.get_stock_news(self.stock_code, count=20)
                sentiment = agg.get_sentiment_summary(self.stock_code)
            else:
                news = agg.get_market_news(count=30)
                sentiment = agg.get_sentiment_summary()

            self.finished.emit(news)
            self.sentiment_updated.emit(sentiment)

        except Exception as e:
            log.warning(f"News fetch error: {e}")
            self.finished.emit([])
            self.sentiment_updated.emit({})


class SentimentGauge(QFrame):
    """Visual sentiment gauge"""

    def __init__(self):
        super().__init__()
        self.setFixedHeight(60)
        self.setStyleSheet("""
            QFrame {
                background: #161b22;
                border-radius: 8px;
                padding: 5px;
            }
        """)

        layout = QHBoxLayout(self)

        self.label = QLabel("Market Sentiment:")
        self.label.setStyleSheet("color: #8b949e; font-size: 11px;")
        layout.addWidget(self.label)

        self.score_label = QLabel("--")
        self.score_label.setStyleSheet("font-size: 16px; font-weight: bold;")
        layout.addWidget(self.score_label)

        self.bar = QProgressBar()
        self.bar.setRange(0, 100)
        self.bar.setValue(50)
        self.bar.setFixedWidth(120)
        self.bar.setFixedHeight(12)
        self.bar.setTextVisible(False)
        layout.addWidget(self.bar)

        self.counts_label = QLabel("")
        self.counts_label.setStyleSheet("color: #8b949e; font-size: 10px;")
        layout.addWidget(self.counts_label)

        layout.addStretch()

    def update_sentiment(self, summary: dict):
        if not summary:
            return

        score = summary.get('overall_sentiment', 0.0)
        label = summary.get('label', 'neutral')
        pos = summary.get('positive_count', 0)
        neg = summary.get('negative_count', 0)
        total = summary.get('total', 0)

        if label == "positive":
            color = "#3fb950"
            emoji = "📈"
        elif label == "negative":
            color = "#f85149"
            emoji = "📉"
        else:
            color = "#d29922"
            emoji = "➡️"

        self.score_label.setText(f"{emoji} {score:+.2f}")
        self.score_label.setStyleSheet(f"color: {color}; font-size: 16px; font-weight: bold;")

        bar_value = int((score + 1.0) / 2.0 * 100)
        self.bar.setValue(max(0, min(100, bar_value)))

        if score > 0.1:
            self.bar.setStyleSheet("""
                QProgressBar { background: #21262d; border-radius: 6px; }
                QProgressBar::chunk { background: #3fb950; border-radius: 6px; }
            """)
        elif score < -0.1:
            self.bar.setStyleSheet("""
                QProgressBar { background: #21262d; border-radius: 6px; }
                QProgressBar::chunk { background: #f85149; border-radius: 6px; }
            """)
        else:
            self.bar.setStyleSheet("""
                QProgressBar { background: #21262d; border-radius: 6px; }
                QProgressBar::chunk { background: #d29922; border-radius: 6px; }
            """)

        self.counts_label.setText(f"📰 {total} | ✅{pos} | ❌{neg}")


class NewsPanel(QWidget):
    """
    Complete news panel with:
    - Sentiment gauge
    - News table with color-coded sentiment
    - Auto-refresh every 5 minutes
    - Stock-specific or market-wide view
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._current_stock: Optional[str] = None
        self._fetch_thread: Optional[NewsFetchThread] = None

        self._setup_ui()
        self._setup_timer()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(5)
        layout.setContentsMargins(0, 0, 0, 0)

        # Sentiment gauge
        self.sentiment_gauge = SentimentGauge()
        layout.addWidget(self.sentiment_gauge)

        # News table
        self.table = QTableWidget()
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(["Time", "Sentiment", "Title", "Source"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Fixed)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.Fixed)
        self.table.setColumnWidth(0, 80)
        self.table.setColumnWidth(1, 70)
        self.table.setColumnWidth(3, 70)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        layout.addWidget(self.table)

        # Controls
        btn_layout = QHBoxLayout()
        self.refresh_btn = QPushButton("🔄 Refresh")
        self.refresh_btn.clicked.connect(lambda: self.refresh(force=True))
        btn_layout.addWidget(self.refresh_btn)

        self.mode_label = QLabel("📰 Market News")
        self.mode_label.setStyleSheet("color: #58a6ff; font-size: 11px;")
        btn_layout.addWidget(self.mode_label)

        btn_layout.addStretch()
        layout.addLayout(btn_layout)

    def _setup_timer(self):
        self._timer = QTimer()
        self._timer.timeout.connect(self.refresh)
        self._timer.start(300000)  # 5 minutes

    def set_stock(self, stock_code: str):
        """Switch to stock-specific news"""
        self._current_stock = stock_code
        self.mode_label.setText(f"📰 News for {stock_code}")
        self.refresh(force=True)

    def set_market_mode(self):
        """Switch to market-wide news"""
        self._current_stock = None
        self.mode_label.setText("📰 Market News")
        self.refresh(force=True)

    def refresh(self, force: bool = False):
        """Fetch news in background"""
        # Check if on China network (news only available there)
        try:
            from core.network import get_network_env
            env = get_network_env()
            if not env.is_china_direct and not env.tencent_ok:
                self.mode_label.setText("📰 News unavailable (VPN active)")
                return
        except Exception:
            pass

        if self._fetch_thread and self._fetch_thread.isRunning():
            return

        fetch_type = "stock" if self._current_stock else "market"
        self._fetch_thread = NewsFetchThread(self._current_stock, fetch_type)
        self._fetch_thread.finished.connect(self._on_news_received)
        self._fetch_thread.sentiment_updated.connect(self.sentiment_gauge.update_sentiment)
        self._fetch_thread.start()

    def _on_news_received(self, news_items: list):
        """Update table with received news + cleanup finished thread."""
        self.table.setRowCount(len(news_items))

        for row, item in enumerate(news_items):
            # Time
            time_str = item.publish_time.strftime("%H:%M") if hasattr(item, 'publish_time') else "--"
            time_item = QTableWidgetItem(time_str)
            time_item.setForeground(QColor("#8b949e"))
            self.table.setItem(row, 0, time_item)

            # Sentiment
            score = float(getattr(item, "sentiment_score", 0.0) or 0.0)
            if score > 0.2:
                sent_text = f"📈 +{score:.1f}"
                sent_color = "#3fb950"
            elif score < -0.2:
                sent_text = f"📉 {score:.1f}"
                sent_color = "#f85149"
            else:
                sent_text = f"➡️ {score:.1f}"
                sent_color = "#d29922"

            sent_item = QTableWidgetItem(sent_text)
            sent_item.setForeground(QColor(sent_color))
            sent_item.setFont(QFont("Consolas", 9))
            self.table.setItem(row, 1, sent_item)

            # Title
            title = str(getattr(item, "title", "") or "")
            title_item = QTableWidgetItem(title)
            if score > 0.3:
                title_item.setForeground(QColor("#3fb950"))
            elif score < -0.3:
                title_item.setForeground(QColor("#f85149"))
            else:
                title_item.setForeground(QColor("#c9d1d9"))
            self.table.setItem(row, 2, title_item)

            # Source
            source = str(getattr(item, "source", "") or "")
            source_item = QTableWidgetItem(source)
            source_item.setForeground(QColor("#8b949e"))
            self.table.setItem(row, 3, source_item)

        # Cleanup finished thread to avoid leaks
        try:
            if self._fetch_thread:
                self._fetch_thread.deleteLater()
        except Exception:
            pass