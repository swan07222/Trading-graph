# ui/realtime_worker.py
"""Real-Time Data Streaming Worker for PyQt6.

This module provides:
- Background worker for real-time market data
- Real-time sentiment updates
- Live prediction streaming
- Integration with WebSocket client
- Thread-safe data updates

Usage:
    from ui.realtime_worker import RealtimeWorker

    worker = RealtimeWorker(symbols=["600519", "000001"])
    worker.data_updated.connect(update_ui)
    worker.start()
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from PyQt6.QtCore import QObject, QThread, QTimer, pyqtSignal, pyqtSlot

from config.runtime_env import env_int
from utils.logger import get_logger

from .websocket_client import WebSocketMessage, get_websocket_client

log = get_logger(__name__)


@dataclass
class MarketData:
    """Real-time market data."""
    symbol: str
    price: float
    change: float
    change_percent: float
    volume: int
    high: float
    low: float
    open: float
    close: float
    bid: float = 0.0
    ask: float = 0.0
    bid_size: int = 0
    ask_size: int = 0
    timestamp: datetime = field(default_factory=datetime.now)
    source: str = "websocket"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "symbol": self.symbol,
            "price": self.price,
            "change": self.change,
            "change_percent": self.change_percent,
            "volume": self.volume,
            "high": self.high,
            "low": self.low,
            "open": self.open,
            "close": self.close,
            "bid": self.bid,
            "ask": self.ask,
            "bid_size": self.bid_size,
            "ask_size": self.ask_size,
            "timestamp": self.timestamp.isoformat(),
            "source": self.source,
        }


@dataclass
class SentimentUpdate:
    """Real-time sentiment update."""
    symbol: str
    overall_sentiment: float
    policy_impact: float
    market_sentiment: float
    trader_sentiment: float
    retail_sentiment: float
    institutional_sentiment: float
    confidence: float
    article_count: int
    discussion_topics: list[str] = field(default_factory=list)
    shared_experiences: list[dict] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "symbol": self.symbol,
            "overall_sentiment": self.overall_sentiment,
            "policy_impact": self.policy_impact,
            "market_sentiment": self.market_sentiment,
            "trader_sentiment": self.trader_sentiment,
            "retail_sentiment": self.retail_sentiment,
            "institutional_sentiment": self.institutional_sentiment,
            "confidence": self.confidence,
            "article_count": self.article_count,
            "discussion_topics": self.discussion_topics,
            "shared_experiences": self.shared_experiences,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class PredictionUpdate:
    """Real-time prediction update."""
    symbol: str
    signal: str  # BUY, HOLD, SELL
    confidence: float
    target_price: float | None = None
    horizon_days: int | None = None
    model_version: str = ""
    trader_signal: str = ""  # From trader discussion
    retail_signal: str = ""  # From retail sentiment
    institutional_signal: str = ""  # From institutional sentiment
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "symbol": self.symbol,
            "signal": self.signal,
            "confidence": self.confidence,
            "target_price": self.target_price,
            "horizon_days": self.horizon_days,
            "model_version": self.model_version,
            "trader_signal": self.trader_signal,
            "retail_signal": self.retail_signal,
            "institutional_signal": self.institutional_signal,
            "timestamp": self.timestamp.isoformat(),
        }


class RealtimeWorker(QObject):
    """Background worker for real-time data streaming."""

    # Signals for UI updates
    market_data_updated = pyqtSignal(object)  # MarketData
    sentiment_updated = pyqtSignal(object)  # SentimentUpdate
    prediction_updated = pyqtSignal(object)  # PredictionUpdate
    error_occurred = pyqtSignal(str)
    status_message = pyqtSignal(str)

    def __init__(
        self,
        symbols: list[str] | None = None,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)

        self.symbols = symbols or []
        self._running = False
        self._thread: QThread | None = None

        # WebSocket client
        self._ws_client = get_websocket_client()

        # Timers
        self._refresh_timer: QTimer | None = None
        self._refresh_interval = env_int("REALTIME_REFRESH_MS", 1000)  # 1 second

        # Cache for latest data
        self._latest_market_data: dict[str, MarketData] = {}
        self._latest_sentiment: dict[str, SentimentUpdate] = {}
        self._latest_predictions: dict[str, PredictionUpdate] = {}

        # Statistics
        self._stats = {
            "messages_processed": 0,
            "last_update": None,
            "update_count": 0,
        }

        # Setup
        self._setup_websocket()

    def _setup_websocket(self) -> None:
        """Setup WebSocket client connections."""
        self._ws_client.message_received.connect(self._handle_websocket_message)
        self._ws_client.connected.connect(self._on_websocket_connected)
        self._ws_client.disconnected.connect(self._on_websocket_disconnected)
        self._ws_client.error_occurred.connect(self._on_websocket_error)

    def _on_websocket_connected(self) -> None:
        """Handle WebSocket connection."""
        self.status_message.emit("Real-time data connected")

        # Subscribe to channels
        self._ws_client.subscribe("market")
        self._ws_client.subscribe("sentiment")
        self._ws_client.subscribe("predictions")

        # Request initial data
        for symbol in self.symbols:
            self._ws_client.send({
                "type": "subscribe_symbol",
                "symbol": symbol,
            })

    def _on_websocket_disconnected(self) -> None:
        """Handle WebSocket disconnection."""
        self.status_message.emit("Real-time data disconnected")

    def _on_websocket_error(self, error: str) -> None:
        """Handle WebSocket error."""
        self.error_occurred.emit(f"WebSocket error: {error}")

    @pyqtSlot(object)
    def _handle_websocket_message(self, msg: WebSocketMessage) -> None:
        """Handle incoming WebSocket message.

        Args:
            msg: WebSocket message
        """
        try:
            self._stats["messages_processed"] += 1
            self._stats["last_update"] = datetime.now()

            if msg.channel == "market":
                self._process_market_data(msg)
            elif msg.channel == "sentiment":
                self._process_sentiment_data(msg)
            elif msg.channel == "predictions":
                self._process_prediction_data(msg)

            self._stats["update_count"] += 1

        except Exception as e:
            log.error(f"Error processing WebSocket message: {e}")
            self.error_occurred.emit(str(e))

    def _process_market_data(self, msg: WebSocketMessage) -> None:
        """Process market data message.

        Args:
            msg: WebSocket message
        """
        data = msg.data
        symbol = data.get("symbol", "")

        if not symbol:
            return

        market_data = MarketData(
            symbol=symbol,
            price=data.get("price", 0.0),
            change=data.get("change", 0.0),
            change_percent=data.get("change_percent", 0.0),
            volume=data.get("volume", 0),
            high=data.get("high", 0.0),
            low=data.get("low", 0.0),
            open=data.get("open", 0.0),
            close=data.get("close", 0.0),
            bid=data.get("bid", 0.0),
            ask=data.get("ask", 0.0),
            timestamp=datetime.fromisoformat(data.get("timestamp", datetime.now().isoformat())),
            source="websocket",
        )

        self._latest_market_data[symbol] = market_data
        self.market_data_updated.emit(market_data)

    def _process_sentiment_data(self, msg: WebSocketMessage) -> None:
        """Process sentiment data message.

        Args:
            msg: WebSocket message
        """
        data = msg.data
        symbol = data.get("symbol", "")

        if not symbol:
            return

        sentiment = SentimentUpdate(
            symbol=symbol,
            overall_sentiment=data.get("overall_sentiment", 0.0),
            policy_impact=data.get("policy_impact", 0.0),
            market_sentiment=data.get("market_sentiment", 0.0),
            trader_sentiment=data.get("trader_sentiment", 0.0),
            retail_sentiment=data.get("retail_sentiment", 0.0),
            institutional_sentiment=data.get("institutional_sentiment", 0.0),
            confidence=data.get("confidence", 0.0),
            article_count=data.get("article_count", 0),
            discussion_topics=data.get("discussion_topics", []),
            shared_experiences=data.get("shared_experiences", []),
            timestamp=datetime.fromisoformat(data.get("timestamp", datetime.now().isoformat())),
        )

        self._latest_sentiment[symbol] = sentiment
        self.sentiment_updated.emit(sentiment)

    def _process_prediction_data(self, msg: WebSocketMessage) -> None:
        """Process prediction data message.

        Args:
            msg: WebSocket message
        """
        data = msg.data
        symbol = data.get("symbol", "")

        if not symbol:
            return

        prediction = PredictionUpdate(
            symbol=symbol,
            signal=data.get("signal", "HOLD"),
            confidence=data.get("confidence", 0.0),
            target_price=data.get("target_price"),
            horizon_days=data.get("horizon_days"),
            model_version=data.get("model_version", ""),
            trader_signal=self._sentiment_to_signal(data.get("trader_sentiment", 0.0)),
            retail_signal=self._sentiment_to_signal(data.get("retail_sentiment", 0.0)),
            institutional_signal=self._sentiment_to_signal(data.get("institutional_sentiment", 0.0)),
            timestamp=datetime.fromisoformat(data.get("timestamp", datetime.now().isoformat())),
        )

        self._latest_predictions[symbol] = prediction
        self.prediction_updated.emit(prediction)

    @staticmethod
    def _sentiment_to_signal(sentiment: float) -> str:
        """Convert sentiment score to trading signal.

        Args:
            sentiment: Sentiment score (-1.0 to 1.0)

        Returns:
            Trading signal
        """
        if sentiment > 0.5:
            return "BUY"
        elif sentiment > 0.2:
            return "WEAK_BUY"
        elif sentiment < -0.5:
            return "SELL"
        elif sentiment < -0.2:
            return "WEAK_SELL"
        else:
            return "HOLD"

    def start(self) -> None:
        """Start the real-time worker."""
        if self._running:
            return

        self._running = True
        log.info("Starting real-time worker")

        # Move to thread
        self._thread = QThread()
        self.moveToThread(self._thread)

        # Start thread
        self._thread.start()

        # Connect WebSocket
        self._ws_client.connect()

        # Start refresh timer
        self._refresh_timer = QTimer()
        self._refresh_timer.timeout.connect(self._on_refresh)
        self._refresh_timer.start(self._refresh_interval)

        self.status_message.emit("Real-time data streaming started")

    def stop(self) -> None:
        """Stop the real-time worker."""
        if not self._running:
            return

        self._running = False
        log.info("Stopping real-time worker")

        # Stop timer
        if self._refresh_timer:
            self._refresh_timer.stop()
            self._refresh_timer = None

        # Disconnect WebSocket
        self._ws_client.disconnect()

        # Stop thread
        if self._thread:
            self._thread.quit()
            self._thread.wait(3000)
            self._thread = None

        self.status_message.emit("Real-time data streaming stopped")

    @pyqtSlot()
    def _on_refresh(self) -> None:
        """Handle refresh timer."""
        # Send heartbeat
        if self._ws_client.is_connected:
            self._ws_client.send({
                "type": "heartbeat",
                "timestamp": datetime.now().isoformat(),
            })

        # Emit stats periodically
        self.status_message.emit(
            f"RT: {self._stats['update_count']} updates, "
            f"{self._stats['messages_processed']} msgs"
        )

    def get_latest_data(self, symbol: str) -> dict[str, Any]:
        """Get latest data for a symbol.

        Args:
            symbol: Stock symbol

        Returns:
            Dictionary with market data, sentiment, and predictions
        """
        return {
            "market": self._latest_market_data.get(symbol),
            "sentiment": self._latest_sentiment.get(symbol),
            "prediction": self._latest_predictions.get(symbol),
        }

    @property
    def stats(self) -> dict[str, Any]:
        """Get worker statistics."""
        return self._stats.copy()

    def cleanup(self) -> None:
        """Cleanup resources."""
        self.stop()


# Worker factory
_workers: dict[str, RealtimeWorker] = {}


def get_realtime_worker(symbols: list[str]) -> RealtimeWorker:
    """Get or create real-time worker.

    Args:
        symbols: List of symbols to track

    Returns:
        Real-time worker instance
    """
    key = ",".join(sorted(symbols))

    if key not in _workers:
        _workers[key] = RealtimeWorker(symbols)

    return _workers[key]


def cleanup_workers() -> None:
    """Cleanup all workers."""
    for worker in _workers.values():
        worker.cleanup()
    _workers.clear()
