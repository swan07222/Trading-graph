# data/news_websocket_server.py
"""WebSocket Server for Real-Time News Broadcasting."""
from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from utils.logger import get_logger

log = get_logger(__name__)

import websockets
from websockets.exceptions import ConnectionClosed

try:
    # websockets>=15
    from websockets.asyncio.server import ServerConnection as WebSocketConnection
except ImportError:  # pragma: no cover - compatibility with older websockets
    from websockets.server import WebSocketServerProtocol as WebSocketConnection


@dataclass
class ClientInfo:
    """Connected client information."""
    websocket: WebSocketConnection
    connected_at: datetime = field(default_factory=datetime.now)
    channels: set[str] = field(default_factory=set)
    messages_sent: int = 0
    last_activity: datetime = field(default_factory=datetime.now)


class NewsWebSocketServer:
    """WebSocket server for real-time news broadcasting."""

    def __init__(self, host: str = "0.0.0.0", port: int = 8765, max_clients: int = 100) -> None:
        self.host = host
        self.port = port
        self.max_clients = max_clients
        self._server: Any = None
        self._running = False
        self._clients: dict[str, ClientInfo] = {}
        self._clients_lock = asyncio.Lock()
        self._streamer: Any = None
        self._total_connections = 0
        self._total_messages = 0
        self._start_time: datetime | None = None

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def client_count(self) -> int:
        return len(self._clients)

    @property
    def metrics(self) -> dict[str, Any]:
        return {
            "total_connections": self._total_connections,
            "total_messages": self._total_messages,
            "current_clients": len(self._clients),
            "max_clients": self.max_clients,
        }

    def set_streamer(self, streamer: Any) -> None:
        self._streamer = streamer

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._start_time = datetime.now()
        self._server = await websockets.serve(
            self._handle_client, self.host, self.port,
            ping_interval=30, ping_timeout=10, max_size=10 * 1024 * 1024,
        )
        log.info(f"WebSocket server started on ws://{self.host}:{self.port}")
        await self._server.wait_closed()

    async def stop(self) -> None:
        if not self._running:
            return
        self._running = False
        async with self._clients_lock:
            for client_info in list(self._clients.values()):
                await client_info.websocket.close(1001, "Server shutting down")
            self._clients.clear()
        if self._server:
            self._server.close()
            await self._server.wait_closed()
            self._server = None
        log.info("WebSocket server stopped")

    def _get_client_id(self, websocket: WebSocketConnection) -> str:
        """Best-effort client identifier from websocket remote address."""
        remote = websocket.remote_address
        if isinstance(remote, tuple) and len(remote) >= 2:
            return f"{remote[0]}:{remote[1]}"
        return "unknown:0"

    async def _get_client_websocket(self, client_id: str) -> WebSocketConnection | None:
        """Safely fetch client websocket from the registry."""
        async with self._clients_lock:
            client_info = self._clients.get(client_id)
            return client_info.websocket if client_info else None

    async def _handle_client(self, websocket: WebSocketConnection) -> None:
        client_id = self._get_client_id(websocket)
        log.info(f"Client connected: {client_id}")
        async with self._clients_lock:
            if len(self._clients) >= self.max_clients:
                await websocket.close(1013, "Server at capacity")
                return
            self._clients[client_id] = ClientInfo(websocket=websocket)
            self._total_connections += 1
        try:
            async for message in websocket:
                await self._handle_message(client_id, message)
        except ConnectionClosed:
            pass
        finally:
            async with self._clients_lock:
                self._clients.pop(client_id, None)
            log.info(f"Client removed: {client_id}")

    async def _handle_message(self, client_id: str, message: str) -> None:
        try:
            data = json.loads(message)
        except json.JSONDecodeError:
            websocket = await self._get_client_websocket(client_id)
            if websocket:
                await self._send_message(websocket, {
                    "type": "error", "data": {"message": "invalid_json"},
                })
            return

        msg_type = data.get("type", "")
        if msg_type == "subscribe":
            await self._handle_subscribe(client_id, data)
        elif msg_type == "unsubscribe":
            await self._handle_unsubscribe(client_id, data)
        elif msg_type == "ping":
            await self._handle_ping(client_id)
        elif msg_type == "get_backlog":
            await self._handle_get_backlog(client_id, data)
        elif msg_type == "get_stats":
            await self._handle_get_stats(client_id)

    async def _handle_subscribe(self, client_id: str, data: dict[str, Any]) -> None:
        channel = data.get("channel", "")
        websocket: WebSocketConnection | None = None
        async with self._clients_lock:
            if client_id in self._clients:
                self._clients[client_id].channels.add(channel)
                websocket = self._clients[client_id].websocket
        if self._streamer:
            self._streamer.subscribe(channel, lambda x: None)
        if websocket:
            await self._send_message(websocket, {
                "type": "subscribe_ack", "data": {"channel": channel},
            })
            log.info(f"Client {client_id} subscribed to {channel}")

    async def _handle_unsubscribe(self, client_id: str, data: dict[str, Any]) -> None:
        channel = data.get("channel", "")
        websocket: WebSocketConnection | None = None
        async with self._clients_lock:
            if client_id in self._clients:
                self._clients[client_id].channels.discard(channel)
                websocket = self._clients[client_id].websocket
        if websocket:
            await self._send_message(websocket, {
                "type": "unsubscribe_ack", "data": {"channel": channel},
            })
            log.info(f"Client {client_id} unsubscribed from {channel}")

    async def _handle_ping(self, client_id: str) -> None:
        websocket = await self._get_client_websocket(client_id)
        if websocket:
            await self._send_message(websocket, {
                "type": "pong", "data": {"timestamp": datetime.now().isoformat()},
            })

    async def _handle_get_backlog(self, client_id: str, data: dict[str, Any]) -> None:
        limit = min(int(data.get("limit", 50)), 200)
        channel = data.get("channel")
        if not self._streamer:
            return
        articles = self._streamer.get_recent(category=channel, limit=limit)
        websocket = await self._get_client_websocket(client_id)
        if websocket:
            await self._send_message(websocket, {
                "type": "backlog",
                "data": {"articles": [a.to_dict() for a in articles], "count": len(articles)},
            })

    async def _handle_get_stats(self, client_id: str) -> None:
        stats = self.metrics
        if self._streamer:
            stats["streamer"] = self._streamer.stats
        websocket = await self._get_client_websocket(client_id)
        if websocket:
            await self._send_message(websocket, {
                "type": "stats", "data": stats,
            })

    async def _send_message(self, websocket: WebSocketConnection, message: dict[str, Any]) -> bool:
        try:
            await websocket.send(json.dumps(message))
        except ConnectionClosed:
            return False
        self._total_messages += 1
        return True

    async def broadcast(self, message: dict[str, Any], channel: str | None = None) -> None:
        async with self._clients_lock:
            disconnected_clients: list[str] = []
            for cid, client_info in list(self._clients.items()):
                if channel and channel not in client_info.channels and "all" not in client_info.channels:
                    continue
                sent = await self._send_message(client_info.websocket, message)
                if sent:
                    client_info.messages_sent += 1
                    client_info.last_activity = datetime.now()
                else:
                    disconnected_clients.append(cid)

            for cid in disconnected_clients:
                self._clients.pop(cid, None)

    async def broadcast_news(self, article: Any) -> None:
        message = {
            "type": "news", "channel": article.category,
            "data": article.to_dict(), "timestamp": datetime.now().isoformat(),
        }
        await self.broadcast(message, channel=article.category)


class NewsWebSocketServerPooled:
    """Alternative pooled WebSocket server."""

    def __init__(self, host: str = "0.0.0.0", port: int = 8766, max_clients: int = 100) -> None:
        self.host = host
        self.port = port
        self.max_clients = max_clients

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass


def create_news_websocket_server(host: str = "0.0.0.0", port: int = 8765, max_clients: int = 100, use_pooled: bool = False):
    if use_pooled:
        return NewsWebSocketServerPooled(host=host, port=port, max_clients=max_clients)
    return NewsWebSocketServer(host=host, port=port, max_clients=max_clients)
