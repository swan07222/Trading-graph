# ui/news_widget.py

from PyQt6.QtCore import QThread, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QFont
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QProgressBar,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from utils.logger import get_logger

log = get_logger(__name__)

class NewsFetchThread(QThread):
    """
    Background thread to fetch news.

    Uses custom signals for data delivery. Does NOT redefine 'finished'
    to avoid collision with QThread.finished.
    """
    news_ready = pyqtSignal(list)           # List[NewsItem]
    sentiment_updated = pyqtSignal(dict)    # sentiment summary
    fetch_complete = pyqtSignal()           # emitted when done (for cleanup)
    fetch_error = pyqtSignal(str)           # error message

    def __init__(self, stock_code: str = None, fetch_type: str = "market"):
        super().__init__()
        self.stock_code = stock_code
        self.fetch_type = str(fetch_type)
        self._is_running = False

    def run(self):
        self._is_running = True
        try:
            from data.news import get_news_aggregator
            agg = get_news_aggregator()

            if self.fetch_type == "stock" and self.stock_code:
                news = agg.get_stock_news(self.stock_code, count=20)
                sentiment = agg.get_sentiment_summary(self.stock_code)
            else:
                news = agg.get_market_news(count=30)
                sentiment = agg.get_sentiment_summary()

            if news is None:
                news = []
            if sentiment is None:
                sentiment = {}

            self.news_ready.emit(news)
            self.sentiment_updated.emit(sentiment)

        except Exception as e:
            log.warning(f"News fetch error: {e}")
            self.news_ready.emit([])
            self.sentiment_updated.emit({})
            self.fetch_error.emit(str(e))

        finally:
            self._is_running = False
            try:
                self.fetch_complete.emit()
            except RuntimeError:
                pass

    @property
    def is_active(self) -> bool:
        """Check if thread is still running (safe against deleted object)."""
        try:
            return self._is_running or self.isRunning()
        except RuntimeError:
            return False

class SentimentGauge(QFrame):
    """Visual sentiment gauge with robust data handling."""

    def __init__(self):
        super().__init__()
        self.setFixedHeight(60)
        self.setStyleSheet("""
            QFrame {
                background: #0f1b2e;
                border-radius: 8px;
                padding: 5px;
            }
        """)

        layout = QHBoxLayout(self)

        self.label = QLabel("Market Sentiment:")
        self.label.setStyleSheet("color: #aac3ec; font-size: 11px;")
        layout.addWidget(self.label)

        self.score_label = QLabel("--")
        self.score_label.setStyleSheet(
            "font-size: 16px; font-weight: bold;"
        )
        layout.addWidget(self.score_label)

        self.bar = QProgressBar()
        self.bar.setRange(0, 100)
        self.bar.setValue(50)
        self.bar.setFixedWidth(120)
        self.bar.setFixedHeight(12)
        self.bar.setTextVisible(False)
        layout.addWidget(self.bar)

        self.counts_label = QLabel("")
        self.counts_label.setStyleSheet(
            "color: #aac3ec; font-size: 10px;"
        )
        layout.addWidget(self.counts_label)

        layout.addStretch()

    @staticmethod
    def _safe_float(data: dict, key: str, default: float = 0.0) -> float:
        """Safely extract float from dict."""
        try:
            val = data.get(key, default)
            if val is None:
                return float(default)
            return float(val)
        except (TypeError, ValueError):
            return float(default)

    @staticmethod
    def _safe_int(data: dict, key: str, default: int = 0) -> int:
        """Safely extract int from dict."""
        try:
            val = data.get(key, default)
            if val is None:
                return int(default)
            return int(val)
        except (TypeError, ValueError):
            return int(default)

    def update_sentiment(self, summary: dict):
        """Update gauge with sentiment summary data."""
        if not summary or not isinstance(summary, dict):
            self.reset()
            return

        score = self._safe_float(summary, 'overall_sentiment')
        label = str(summary.get('label', 'neutral') or 'neutral')
        pos = self._safe_int(summary, 'positive_count')
        neg = self._safe_int(summary, 'negative_count')
        total = self._safe_int(summary, 'total')

        if label == "positive":
            color = "#35b57c"
            sentiment_tag = "POS"
        elif label == "negative":
            color = "#e5534b"
            sentiment_tag = "NEG"
        else:
            color = "#d8a03a"
            sentiment_tag = "NEU"

        self.score_label.setText(f"{sentiment_tag} {score:+.2f}")
        self.score_label.setStyleSheet(
            f"color: {color}; font-size: 16px; font-weight: bold;"
        )

        # Map score (-1.0 to 1.0) to bar (0 to 100)
        bar_value = int((score + 1.0) / 2.0 * 100)
        self.bar.setValue(max(0, min(100, bar_value)))

        if score > 0.1:
            bar_style = """
                QProgressBar {
                    background: #14243d; border-radius: 6px;
                }
                QProgressBar::chunk {
                    background: #35b57c; border-radius: 6px;
                }
            """
        elif score < -0.1:
            bar_style = """
                QProgressBar {
                    background: #14243d; border-radius: 6px;
                }
                QProgressBar::chunk {
                    background: #e5534b; border-radius: 6px;
                }
            """
        else:
            bar_style = """
                QProgressBar {
                    background: #14243d; border-radius: 6px;
                }
                QProgressBar::chunk {
                    background: #d8a03a; border-radius: 6px;
                }
            """
        self.bar.setStyleSheet(bar_style)

        self.counts_label.setText(f"NEWS {total} | POS {pos} | NEG {neg}")

    def reset(self):
        """Reset gauge to default state."""
        self.score_label.setText("--")
        self.score_label.setStyleSheet(
            "font-size: 16px; font-weight: bold; color: #aac3ec;"
        )
        self.bar.setValue(50)
        self.counts_label.setText("")

class NewsPanel(QWidget):
    """
    Complete news panel with:
    - Sentiment gauge
    - News table with color-coded sentiment
    - Auto-refresh every 5 minutes
    - Stock-specific or market-wide view
    - Robust thread management
    """

    REFRESH_INTERVAL_MS = 300000  # 5 minutes

    def __init__(self, parent=None):
        super().__init__(parent)
        self._current_stock: str | None = None
        self._fetch_thread: NewsFetchThread | None = None
        self._is_fetching = False
        self._pending_refresh = False

        self._setup_ui()
        self._setup_timer()

    @staticmethod
    def _normalize_stock_code(stock_code: str | None) -> str | None:
        """Normalize stock code values for comparisons."""
        text = str(stock_code or "").strip()
        return text if text else None

    def _mode_text(self) -> str:
        """Current base mode label text."""
        if self._current_stock:
            return f"News for {self._current_stock}"
        return "Market News"

    def _clear_news_view(self) -> None:
        """Clear table and reset gauge to avoid stale UI state."""
        self.table.setRowCount(0)
        self.sentiment_gauge.reset()

    def _is_stale_sender(self) -> bool:
        """Return True when a signal comes from a no-longer-active request."""
        sender = self.sender()
        if not isinstance(sender, NewsFetchThread):
            return False
        sender_stock = self._normalize_stock_code(sender.stock_code)
        current_stock = self._normalize_stock_code(self._current_stock)
        return sender_stock != current_stock

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(5)
        layout.setContentsMargins(0, 0, 0, 0)

        self.sentiment_gauge = SentimentGauge()
        layout.addWidget(self.sentiment_gauge)

        self.table = QTableWidget()
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(
            ["Time", "Sentiment", "Title", "Source"]
        )
        self.table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.Fixed
        )
        self.table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.Fixed
        )
        self.table.horizontalHeader().setSectionResizeMode(
            2, QHeaderView.ResizeMode.Stretch
        )
        self.table.horizontalHeader().setSectionResizeMode(
            3, QHeaderView.ResizeMode.Fixed
        )
        self.table.setColumnWidth(0, 80)
        self.table.setColumnWidth(1, 70)
        self.table.setColumnWidth(3, 70)
        self.table.setSelectionBehavior(
            QTableWidget.SelectionBehavior.SelectRows
        )
        self.table.setEditTriggers(
            QTableWidget.EditTrigger.NoEditTriggers
        )
        self.table.verticalHeader().setVisible(False)
        layout.addWidget(self.table)

        btn_layout = QHBoxLayout()
        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.clicked.connect(lambda: self.refresh(force=True))
        btn_layout.addWidget(self.refresh_btn)

        self.mode_label = QLabel(self._mode_text())
        self.mode_label.setStyleSheet(
            "color: #79a6ff; font-size: 11px;"
        )
        btn_layout.addWidget(self.mode_label)

        btn_layout.addStretch()
        layout.addLayout(btn_layout)

    def _setup_timer(self):
        """Setup auto-refresh timer."""
        self._timer = QTimer()
        self._timer.timeout.connect(self.refresh)
        self._timer.start(self.REFRESH_INTERVAL_MS)

    def set_stock(self, stock_code: str):
        """Switch to stock-specific news."""
        if not stock_code:
            return
        self._current_stock = str(stock_code).strip()
        self.mode_label.setText(self._mode_text())
        self.refresh(force=True)

    def set_market_mode(self):
        """Switch to market-wide news."""
        self._current_stock = None
        self.mode_label.setText(self._mode_text())
        self.refresh(force=True)

    def refresh(self, force: bool = False):
        """
        Fetch news in background safely.

        Guards:
        - Won't start if already fetching
        - Handles deleted thread references
        - Checks network availability
        """
        if self._is_fetching:
            if force:
                self._pending_refresh = True
            return

        # Check if previous thread is still running (safely)
        if self._fetch_thread is not None:
            try:
                if self._fetch_thread.is_active:
                    return
            except (RuntimeError, AttributeError):
                pass
            self._cleanup_thread()

        # Check network availability (non-blocking)
        try:
            from core.network import get_network_env
            env = get_network_env()
            if not env.is_china_direct and not env.tencent_ok:
                self.mode_label.setText(
                    "News unavailable (VPN active)"
                )
                self._clear_news_view()
                return
        except Exception:
            pass

        fetch_type = "stock" if self._current_stock else "market"
        thread = NewsFetchThread(self._current_stock, fetch_type)

        self._fetch_thread = thread
        self._is_fetching = True
        self.refresh_btn.setEnabled(False)
        self.mode_label.setText(self._mode_text())

        thread.news_ready.connect(self._on_news_received)
        thread.sentiment_updated.connect(self._on_sentiment_received)
        thread.fetch_complete.connect(self._on_fetch_complete)
        thread.fetch_error.connect(self._on_fetch_error)

        thread.start()

    def _on_fetch_complete(self):
        """Handle fetch thread completion."""
        self._is_fetching = False
        self.refresh_btn.setEnabled(True)

        should_refresh = self._pending_refresh
        self._pending_refresh = False

        # Schedule cleanup (don't delete thread from its own signal)
        QTimer.singleShot(100, self._cleanup_thread)
        if should_refresh:
            QTimer.singleShot(0, lambda: self.refresh(force=False))

    def _on_fetch_error(self, error: str):
        """Handle fetch error."""
        if self._is_stale_sender():
            return
        self._is_fetching = False
        self.refresh_btn.setEnabled(True)
        self.mode_label.setText(f"{self._mode_text()} (fetch error)")
        self._clear_news_view()
        log.debug(f"News fetch error: {error}")

    def _on_sentiment_received(self, summary: dict):
        """Apply sentiment updates for the currently active request only."""
        if self._is_stale_sender():
            return
        self.sentiment_gauge.update_sentiment(summary)

    def _cleanup_thread(self):
        """Safely clean up fetch thread reference."""
        thread = self._fetch_thread
        self._fetch_thread = None
        self._is_fetching = False
        self.refresh_btn.setEnabled(True)

        if thread is not None:
            try:
                if thread.isRunning():
                    thread.wait(2000)
            except RuntimeError:
                pass

            try:
                thread.deleteLater()
            except (RuntimeError, AttributeError):
                pass

    @staticmethod
    def _safe_item_attr(item, attr: str, default=""):
        """Safely get attribute from a news item."""
        try:
            val = getattr(item, attr, None)
            if val is None:
                return default
            return val
        except Exception:
            return default

    @staticmethod
    def _safe_float_attr(item, attr: str, default: float = 0.0) -> float:
        """Safely get float attribute from a news item."""
        try:
            val = getattr(item, attr, None)
            if val is None:
                return float(default)
            return float(val)
        except (TypeError, ValueError):
            return float(default)

    def _on_news_received(self, news_items: list):
        """Update table with received news and handle missing attributes."""
        if self._is_stale_sender():
            return

        self.mode_label.setText(self._mode_text())

        if news_items is None:
            news_items = []

        self.table.setRowCount(len(news_items))

        for row, item in enumerate(news_items):
            time_str = "--"
            publish_time = self._safe_item_attr(
                item, 'publish_time', None
            )
            if publish_time is not None:
                try:
                    if hasattr(publish_time, 'strftime'):
                        time_str = publish_time.strftime("%H:%M")
                    else:
                        time_str = str(publish_time)[:5]
                except Exception:
                    pass

            time_item = QTableWidgetItem(time_str)
            time_item.setForeground(QColor("#aac3ec"))
            self.table.setItem(row, 0, time_item)

            score = self._safe_float_attr(item, "sentiment_score", 0.0)

            if score > 0.2:
                sent_text = f"POS {score:+.1f}"
                sent_color = "#35b57c"
            elif score < -0.2:
                sent_text = f"NEG {score:+.1f}"
                sent_color = "#e5534b"
            else:
                sent_text = f"NEU {score:+.1f}"
                sent_color = "#d8a03a"

            sent_item = QTableWidgetItem(sent_text)
            sent_item.setForeground(QColor(sent_color))
            sent_item.setFont(QFont("Consolas", 9))
            self.table.setItem(row, 1, sent_item)

            title = str(self._safe_item_attr(item, "title", ""))
            title_item = QTableWidgetItem(title)

            if score > 0.3:
                title_item.setForeground(QColor("#35b57c"))
            elif score < -0.3:
                title_item.setForeground(QColor("#e5534b"))
            else:
                title_item.setForeground(QColor("#dbe4f3"))
            self.table.setItem(row, 2, title_item)

            source = str(self._safe_item_attr(item, "source", ""))
            source_item = QTableWidgetItem(source)
            source_item.setForeground(QColor("#aac3ec"))
            self.table.setItem(row, 3, source_item)

    def stop(self):
        """Stop all background activity."""
        self._pending_refresh = False
        if hasattr(self, '_timer') and self._timer:
            try:
                self._timer.stop()
            except Exception:
                pass

        self._cleanup_thread()

    def __del__(self):
        """Destructor: stop timer and threads."""
        try:
            self.stop()
        except Exception:
            pass
