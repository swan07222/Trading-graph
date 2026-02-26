# data/news_streamer.py
"""Real-Time News Streaming Service with WebSocket Support.

This module provides:
- Real-time news streaming via WebSocket
- Async news collection with configurable intervals
- Event-driven architecture for UI integration
- Channel-based subscriptions (policy, market, company)
- Automatic reconnection and health monitoring

Usage:
    from data.news_streamer import NewsStreamer
    
    async def on_news(article):
        print(f"New article: {article.title}")
    
    streamer = NewsStreamer()
    streamer.subscribe("market", on_news)
    await streamer.start()
"""
from __future__ import annotations

import asyncio
import json
import threading
import time
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class StreamStats:
    """Streaming statistics."""
    articles_received: int = 0
    articles_broadcast: int = 0
    errors: int = 0
    last_article_time: datetime | None = None
    start_time: datetime | None = None
    reconnect_count: int = 0
    avg_latency_ms: float = 0.0
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "articles_received": self.articles_received,
            "articles_broadcast": self.articles_broadcast,
            "errors": self.errors,
            "last_article_time": self.last_article_time.isoformat() if self.last_article_time else None,
            "uptime_seconds": (datetime.now() - self.start_time).total_seconds() if self.start_time else 0,
            "reconnect_count": self.reconnect_count,
            "avg_latency_ms": self.avg_latency_ms,
        }


class NewsStreamer:
    """Real-time news streaming service.
    
    Features:
    - Async news collection with configurable poll intervals
    - WebSocket broadcasting to connected clients
    - Channel-based subscriptions (policy, market, company, regulatory)
    - Deduplication using article IDs
    - Automatic reconnection with exponential backoff
    - Health monitoring and metrics
    
    Args:
        poll_interval: Seconds between news collection cycles
        categories: Categories to stream
        keywords: Optional keywords to filter
        max_backlog: Maximum articles to buffer
    """
    
    def __init__(
        self,
        poll_interval: float = 30.0,
        categories: list[str] | None = None,
        keywords: list[str] | None = None,
        max_backlog: int = 1000,
    ) -> None:
        from data.news_collector import NewsCollector
        
        self.poll_interval = float(poll_interval)
        self.categories = categories or ["policy", "market", "company", "regulatory"]
        self.keywords = keywords
        self.max_backlog = int(max_backlog)
        
        # Components
        self._collector = NewsCollector()
        
        # State
        self._running = False
        self._task: asyncio.Task | None = None
        self._seen_ids: set[str] = set()
        self._backlog: list[Any] = []  # List of NewsArticle
        self._backlog_lock = threading.RLock()
        
        # Subscriptions: channel -> list of callbacks
        self._subscriptions: dict[str, list[Callable[[Any], Coroutine[Any, Any, None]]]] = {}
        self._sub_lock = threading.RLock()
        
        # WebSocket clients
        self._ws_clients: set[Any] = set()  # Set of WebSocket connections
        self._ws_lock = asyncio.Lock()
        
        # Statistics
        self._stats = StreamStats()
        
        # Reconnection
        self._reconnect_delay = 5.0
        self._max_reconnect_delay = 300.0
        self._consecutive_failures = 0
        
        # Health
        self._last_successful_fetch: datetime | None = None
        self._health_score = 1.0
        
    @property
    def is_running(self) -> bool:
        """Check if streamer is running."""
        return self._running
    
    @property
    def stats(self) -> dict[str, Any]:
        """Get streaming statistics."""
        return self._stats.to_dict()
    
    @property
    def health(self) -> float:
        """Get health score (0-1)."""
        return self._health_score
    
    def subscribe(
        self,
        channel: str,
        callback: Callable[[Any], Coroutine[Any, Any, None]],
    ) -> None:
        """Subscribe to news channel.
        
        Args:
            channel: Channel name (policy, market, company, regulatory, all)
            callback: Async callback function receiving NewsArticle
        """
        with self._sub_lock:
            if channel not in self._subscriptions:
                self._subscriptions[channel] = []
            if callback not in self._subscriptions[channel]:
                self._subscriptions[channel].append(callback)
        log.info(f"Registered subscription for channel: {channel}")
    
    def unsubscribe(
        self,
        channel: str,
        callback: Callable[[Any], Coroutine[Any, Any, None]] | None = None,
    ) -> None:
        """Unsubscribe from channel.
        
        Args:
            channel: Channel name
            callback: Optional specific callback to remove
        """
        with self._sub_lock:
            if channel in self._subscriptions:
                if callback:
                    if callback in self._subscriptions[channel]:
                        self._subscriptions[channel].remove(callback)
                else:
                    self._subscriptions[channel].clear()
        log.info(f"Unsubscribed from channel: {channel}")
    
    async def start(self) -> None:
        """Start news streaming."""
        if self._running:
            log.warning("News streamer already running")
            return
        
        log.info(f"Starting news streamer (poll_interval={self.poll_interval}s)")
        self._running = True
        self._stats.start_time = datetime.now()
        self._stats.reconnect_count = 0
        
        self._task = asyncio.create_task(self._stream_loop())
        log.info("News streamer started")
    
    async def stop(self) -> None:
        """Stop news streaming."""
        if not self._running:
            return
        
        log.info("Stopping news streamer...")
        self._running = False
        
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

        log.info("News streamer stopped")
    
    async def _stream_loop(self) -> None:
        """Main streaming loop."""
        while self._running:
            try:
                start_time = time.time()
                
                # Collect news
                articles = await self._collect_news_async()
                
                # Process and broadcast
                for article in articles:
                    await self._process_article(article)
                
                # Update health
                self._last_successful_fetch = datetime.now()
                self._consecutive_failures = 0
                self._health_score = min(1.0, self._health_score + 0.1)
                
                # Calculate latency
                latency_ms = (time.time() - start_time) * 1000
                self._stats.avg_latency_ms = (
                    0.9 * self._stats.avg_latency_ms + 0.1 * latency_ms
                )
                
                # Wait for next poll
                elapsed = time.time() - start_time
                sleep_time = max(0, self.poll_interval - elapsed)
                await asyncio.sleep(sleep_time)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                log.exception(f"News stream error: {e}")
                self._stats.errors += 1
                self._consecutive_failures += 1
                self._health_score = max(0.0, self._health_score - 0.2)
                
                # Exponential backoff
                delay = min(
                    self._max_reconnect_delay,
                    self._reconnect_delay * (2 ** self._consecutive_failures)
                )
                await asyncio.sleep(delay)
    
    async def _collect_news_async(self) -> list[Any]:
        """Collect news asynchronously."""
        from data.news_collector import NewsArticle
        
        # Run blocking collector in executor
        loop = asyncio.get_event_loop()
        
        def fetch():
            return self._collector.collect_news(
                keywords=self.keywords,
                categories=self.categories,
                limit=50,
                hours_back=1,  # Last hour for real-time
            )
        
        articles = await loop.run_in_executor(None, fetch)
        self._stats.articles_received += len(articles)
        return articles
    
    async def _process_article(self, article: Any) -> None:
        """Process and broadcast article.
        
        Args:
            article: NewsArticle object
        """
        from data.news_collector import NewsArticle
        
        # Deduplicate
        if article.id in self._seen_ids:
            return
        self._seen_ids.add(article.id)
        
        # Add to backlog
        with self._backlog_lock:
            self._backlog.append(article)
            # Trim backlog
            if len(self._backlog) > self.max_backlog:
                self._backlog = self._backlog[-self.max_backlog:]
        
        # Update stats
        self._stats.articles_broadcast += 1
        self._stats.last_article_time = datetime.now()
        
        # Broadcast to subscribers
        await self._broadcast(article)
        
        # Broadcast to WebSocket clients
        await self._broadcast_ws(article)
        
        log.debug(f"Broadcast news: {article.category} - {article.title[:50]}...")
    
    async def _broadcast(self, article: Any) -> None:
        """Broadcast to channel subscribers.
        
        Args:
            article: NewsArticle object
        """
        with self._sub_lock:
            # Snapshot callbacks before awaiting to avoid lock contention.
            subscribers = list(self._subscriptions.get(article.category, []))
            all_subscribers = list(self._subscriptions.get("all", []))

        for callback in subscribers + all_subscribers:
            try:
                await callback(article)
            except Exception as e:
                log.error(f"News callback error: {e}")
    
    async def _broadcast_ws(self, article: Any) -> None:
        """Broadcast to WebSocket clients.
        
        Args:
            article: NewsArticle object
        """
        if not self._ws_clients:
            return
        
        message = {
            "type": "news",
            "channel": article.category,
            "data": article.to_dict(),
            "timestamp": datetime.now().isoformat(),
        }
        
        async with self._ws_lock:
            disconnected = set()
            for client in self._ws_clients:
                try:
                    await client.send_json(message)
                except Exception:
                    disconnected.add(client)
            
            # Remove disconnected clients
            for client in disconnected:
                self._ws_clients.discard(client)
    
    async def add_ws_client(self, websocket: Any) -> None:
        """Add WebSocket client.
        
        Args:
            websocket: WebSocket connection object
        """
        async with self._ws_lock:
            self._ws_clients.add(websocket)
            log.info(f"Added WebSocket client (total: {len(self._ws_clients)})")
    
    async def remove_ws_client(self, websocket: Any) -> None:
        """Remove WebSocket client.
        
        Args:
            websocket: WebSocket connection object
        """
        async with self._ws_lock:
            self._ws_clients.discard(websocket)
            log.info(f"Removed WebSocket client (total: {len(self._ws_clients)})")
    
    def get_backlog(self, limit: int = 100) -> list[Any]:
        """Get recent articles from backlog.
        
        Args:
            limit: Maximum articles to return
            
        Returns:
            List of NewsArticle objects
        """
        with self._backlog_lock:
            return list(self._backlog[-limit:])
    
    def get_recent(self, category: str | None = None, limit: int = 50) -> list[Any]:
        """Get recent articles, optionally filtered by category.
        
        Args:
            category: Optional category filter
            limit: Maximum articles to return
            
        Returns:
            List of NewsArticle objects
        """
        with self._backlog_lock:
            articles = self._backlog.copy()
        if category:
            articles = [a for a in articles if a.category == category]
        return list(articles[-limit:])


# Singleton instance
_streamer: NewsStreamer | None = None


def get_streamer() -> NewsStreamer:
    """Get or create news streamer instance."""
    global _streamer
    if _streamer is None:
        _streamer = NewsStreamer()
    return _streamer


def reset_streamer() -> None:
    """Reset streamer instance."""
    global _streamer
    if _streamer:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            asyncio.run(_streamer.stop())
        else:
            loop.create_task(_streamer.stop())
    _streamer = None
