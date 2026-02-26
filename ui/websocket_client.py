# ui/websocket_client.py
"""WebSocket Client for Real-Time Data Streaming in PyQt6.

This module provides:
- WebSocket client for real-time market data
- Real-time sentiment updates
- Live prediction streaming
- Auto-reconnection with exponential backoff
- Thread-safe signal emission to PyQt6 UI

Usage:
    from ui.websocket_client import WebSocketClient

    client = WebSocketClient("ws://localhost:8765")
    client.message_received.connect(handle_message)
    client.connect()
"""
from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any

from PyQt6.QtCore import (
    QObject,
    QUrl,
    pyqtSignal,
    pyqtSlot,
)
from PyQt6.QtNetwork import (
    QAbstractSocket,
    QNetworkProxy,
    QSslConfiguration,
)
from PyQt6.QtWebSockets import QWebSocket

from config.runtime_env import env_int, env_text
from utils.logger import get_logger

log = get_logger(__name__)


class ConnectionState(Enum):
    """WebSocket connection state."""
    DISCONNECTED = auto()
    CONNECTING = auto()
    CONNECTED = auto()
    RECONNECTING = auto()
    ERROR = auto()


@dataclass
class WebSocketMessage:
    """WebSocket message structure."""
    type: str
    data: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    channel: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "type": self.type,
            "data": self.data,
            "timestamp": self.timestamp.isoformat(),
            "channel": self.channel,
        }


class WebSocketClient(QObject):
    """WebSocket client for real-time data streaming in PyQt6.

    Features:
    - Auto-reconnection with exponential backoff
    - Heartbeat/ping-pong support
    - Channel subscription
    - Thread-safe signal emission
    """

    # Signals for UI updates
    message_received = pyqtSignal(object)  # WebSocketMessage
    connected = pyqtSignal()
    disconnected = pyqtSignal()
    state_changed = pyqtSignal(ConnectionState)
    error_occurred = pyqtSignal(str)
    stats_updated = pyqtSignal(dict)

    def __init__(
        self,
        url: str = "",
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)

        # Configuration
        self.url = url or env_text("WEBSOCKET_URL", "ws://localhost:8765")
        self.reconnect_enabled = env_text("WEBSOCKET_RECONNECT", "true").lower() == "true"
        self.max_reconnect_attempts = env_int("WEBSOCKET_MAX_RECONNECT", 10)
        self.reconnect_delay_base = env_int("WEBSOCKET_RECONNECT_DELAY", 5)
        self.heartbeat_interval = env_int("WEBSOCKET_HEARTBEAT", 30)

        # State
        self._state = ConnectionState.DISCONNECTED
        self._socket: QWebSocket | None = None
        self._reconnect_attempts = 0
        self._last_message_time: datetime | None = None
        self._channels: set[str] = set()
        self._subscriptions: dict[str, list[Callable]] = {}

        # Statistics
        self._stats = {
            "messages_received": 0,
            "messages_sent": 0,
            "reconnect_count": 0,
            "last_error": "",
            "uptime_seconds": 0.0,
            "connect_time": None,
        }

    @property
    def state(self) -> ConnectionState:
        """Get current connection state."""
        return self._state

    @property
    def is_connected(self) -> bool:
        """Check if connected."""
        return self._state == ConnectionState.CONNECTED

    @property
    def stats(self) -> dict[str, Any]:
        """Get connection statistics."""
        if self._stats["connect_time"]:
            self._stats["uptime_seconds"] = (
                datetime.now() - self._stats["connect_time"]
            ).total_seconds()
        return self._stats.copy()

    def connect(self) -> None:
        """Connect to WebSocket server."""
        if self._state in (ConnectionState.CONNECTING, ConnectionState.CONNECTED):
            log.debug("Already connecting or connected")
            return

        self._set_state(ConnectionState.CONNECTING)
        log.info(f"Connecting to WebSocket: {self.url}")

        try:
            self._socket = QWebSocket()

            # Connect signals
            self._socket.connected.connect(self._on_connected)
            self._socket.disconnected.connect(self._on_disconnected)
            self._socket.textMessageReceived.connect(self._on_message)
            self._socket.errorOccurred.connect(self._on_error)

            # SSL configuration for wss://
            if self.url.startswith("wss://"):
                ssl_config = QSslConfiguration.defaultConfiguration()
                self._socket.setSslConfiguration(ssl_config)

            # Connect
            self._socket.open(QUrl(self.url))

        except Exception as e:
            log.error(f"Failed to create WebSocket: {e}")
            self._set_state(ConnectionState.ERROR)
            self.error_occurred.emit(str(e))
            self._schedule_reconnect()

    def disconnect(self) -> None:
        """Disconnect from WebSocket server."""
        self.reconnect_enabled = False

        if self._socket:
            self._socket.close()
            self._socket = None

        self._set_state(ConnectionState.DISCONNECTED)
        log.info("WebSocket disconnected")

    def subscribe(self, channel: str, callback: Callable | None = None) -> None:
        """Subscribe to a channel.

        Args:
            channel: Channel name (e.g., "market", "sentiment", "predictions")
            callback: Optional callback function
        """
        self._channels.add(channel)

        if callback:
            if channel not in self._subscriptions:
                self._subscriptions[channel] = []
            self._subscriptions[channel].append(callback)

        # Send subscription message if connected
        if self.is_connected:
            self._send_message({
                "type": "subscribe",
                "channel": channel,
            })

        log.info(f"Subscribed to channel: {channel}")

    def unsubscribe(self, channel: str) -> None:
        """Unsubscribe from a channel."""
        self._channels.discard(channel)
        self._subscriptions.pop(channel, None)

        if self.is_connected:
            self._send_message({
                "type": "unsubscribe",
                "channel": channel,
            })

        log.info(f"Unsubscribed from channel: {channel}")

    def send(self, message: dict[str, Any]) -> bool:
        """Send message to WebSocket server.

        Args:
            message: Message dictionary

        Returns:
            True if sent successfully
        """
        if not self.is_connected or not self._socket:
            log.warning("Cannot send: not connected")
            return False

        try:
            self._socket.sendTextMessage(json.dumps(message))
            self._stats["messages_sent"] += 1
            return True
        except Exception as e:
            log.error(f"Failed to send message: {e}")
            return False

    def _send_message(self, message: dict[str, Any]) -> bool:
        """Internal send with logging."""
        return self.send(message)

    @pyqtSlot()
    def _on_connected(self) -> None:
        """Handle connection established."""
        self._set_state(ConnectionState.CONNECTED)
        self._reconnect_attempts = 0
        self._stats["connect_time"] = datetime.now()
        self._stats["last_error"] = ""

        log.info("WebSocket connected")
        self.connected.emit()

        # Resubscribe to channels
        for channel in self._channels:
            self._send_message({
                "type": "subscribe",
                "channel": channel,
            })

        # Send heartbeat
        self._send_heartbeat()

    @pyqtSlot()
    def _on_disconnected(self) -> None:
        """Handle disconnection."""
        was_connected = self._state == ConnectionState.CONNECTED
        self._set_state(ConnectionState.DISCONNECTED)

        log.info("WebSocket disconnected")
        self.disconnected.emit()

        # Attempt reconnection
        if was_connected and self.reconnect_enabled:
            self._schedule_reconnect()

    @pyqtSlot(str)
    def _on_message(self, text: str) -> None:
        """Handle incoming message.

        Args:
            text: Message text
        """
        try:
            data = json.loads(text)
            msg = WebSocketMessage(
                type=data.get("type", "unknown"),
                data=data.get("data", {}),
                channel=data.get("channel", ""),
            )

            self._stats["messages_received"] += 1
            self._last_message_time = datetime.now()

            # Emit signal for UI
            self.message_received.emit(msg)

            # Call channel-specific callbacks
            if msg.channel in self._subscriptions:
                for callback in self._subscriptions[msg.channel]:
                    try:
                        callback(msg)
                    except Exception as e:
                        log.error(f"Callback error for channel {msg.channel}: {e}")

            # Handle server messages
            self._handle_server_message(msg)

        except json.JSONDecodeError as e:
            log.warning(f"Failed to parse WebSocket message: {e}")
        except Exception as e:
            log.error(f"Error processing WebSocket message: {e}")

    @pyqtSlot(QAbstractSocket.SocketError)
    def _on_error(self, error: QAbstractSocket.SocketError) -> None:
        """Handle WebSocket error.

        Args:
            error: Socket error
        """
        if self._socket:
            error_string = self._socket.errorString()
        else:
            error_string = f"Unknown error: {error}"

        self._stats["last_error"] = error_string
        self._set_state(ConnectionState.ERROR)

        log.error(f"WebSocket error: {error_string}")
        self.error_occurred.emit(error_string)

        # Attempt reconnection
        if self.reconnect_enabled:
            self._schedule_reconnect()

    @pyqtSlot()
    def _on_reconnect(self) -> None:
        """Handle reconnection attempt."""
        if self._reconnect_attempts < self.max_reconnect_attempts:
            self._reconnect_attempts += 1
            self._stats["reconnect_count"] += 1

            delay = self.reconnect_delay_base * (2 ** (self._reconnect_attempts - 1))
            log.info(f"Reconnecting (attempt {self._reconnect_attempts}/{self.max_reconnect_attempts}) in {delay}s")

            self._set_state(ConnectionState.RECONNECTING)

            # Schedule reconnection
            from PyQt6.QtCore import QTimer
            QTimer.singleShot(delay * 1000, self.connect)
        else:
            log.error("Max reconnection attempts reached")
            self.error_occurred.emit("Max reconnection attempts reached")

    def _schedule_reconnect(self) -> None:
        """Schedule reconnection."""
        from PyQt6.QtCore import QTimer
        QTimer.singleShot(1000, self._on_reconnect)

    def _send_heartbeat(self) -> None:
        """Send heartbeat/ping to server."""
        if self.is_connected:
            self._send_message({
                "type": "ping",
                "timestamp": datetime.now().isoformat(),
            })

    def _handle_server_message(self, msg: WebSocketMessage) -> None:
        """Handle server-initiated messages.

        Args:
            msg: WebSocket message
        """
        if msg.type == "pong":
            # Heartbeat response
            pass
        elif msg.type == "subscribe_ack":
            log.info(f"Subscription acknowledged: {msg.channel}")
        elif msg.type == "error":
            log.error(f"Server error: {msg.data}")
            self.error_occurred.emit(str(msg.data))

    def _set_state(self, state: ConnectionState) -> None:
        """Set connection state and emit signal.

        Args:
            state: New state
        """
        old_state = self._state
        self._state = state
        self.state_changed.emit(state)

        if old_state != state:
            log.debug(f"WebSocket state: {old_state.name} -> {state.name}")

    def close(self) -> None:
        """Close connection and cleanup."""
        self.disconnect()
        if self._socket:
            self._socket.deleteLater()
            self._socket = None


# Singleton instance
_client: WebSocketClient | None = None


def get_websocket_client(url: str | None = None) -> WebSocketClient:
    """Get or create WebSocket client instance.

    Args:
        url: Optional WebSocket URL

    Returns:
        WebSocket client instance
    """
    global _client
    if _client is None:
        _client = WebSocketClient(url or "")
    return _client


def reset_websocket_client() -> None:
    """Reset WebSocket client instance."""
    global _client
    if _client:
        _client.close()
    _client = None
