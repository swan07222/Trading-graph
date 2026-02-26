# data/news_sentiment_stream.py
"""Real-Time Sentiment Analysis Streaming Pipeline.

This module provides:
- Streaming sentiment analysis for news articles
- Rolling sentiment windows with exponential decay
- Real-time sentiment score updates via events
- Policy impact detection and alerting
- Sentiment trend analysis

Usage:
    from data.news_sentiment_stream import SentimentStreamProcessor
    
    processor = SentimentStreamProcessor()
    await processor.start()
    
    # Get current sentiment
    sentiment = processor.get_current_sentiment("000001")
"""
from __future__ import annotations

import asyncio
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class SentimentWindow:
    """Rolling sentiment window with decay."""
    values: deque = field(default_factory=lambda: deque(maxlen=100))
    timestamps: deque = field(default_factory=lambda: deque(maxlen=100))
    policy_values: deque = field(default_factory=lambda: deque(maxlen=50))
    
    def add(self, value: float, timestamp: datetime, is_policy: bool = False) -> None:
        """Add sentiment value to window."""
        self.values.append(value)
        self.timestamps.append(timestamp)
        if is_policy:
            self.policy_values.append(value)
    
    def get_weighted_average(self, decay_half_life: timedelta = timedelta(hours=2)) -> float:
        """Calculate time-decayed weighted average.
        
        Args:
            decay_half_life: Half-life for exponential decay
            
        Returns:
            Weighted sentiment score
        """
        if not self.values:
            return 0.0
        
        now = datetime.now()
        weights = []
        values = []
        
        for value, ts in zip(self.values, self.timestamps):
            age = now - ts
            # Exponential decay: weight = 0.5^(age / half_life)
            weight = 0.5 ** (age.total_seconds() / decay_half_life.total_seconds())
            weights.append(weight)
            values.append(value)
        
        total_weight = sum(weights)
        if total_weight == 0:
            return 0.0
        
        return sum(w * v for w, v in zip(weights, values)) / total_weight
    
    def get_trend(self, window_minutes: int = 30) -> float:
        """Calculate sentiment trend (slope).
        
        Args:
            window_minutes: Minutes to look back
            
        Returns:
            Trend value (positive = improving, negative = worsening)
        """
        if len(self.values) < 2:
            return 0.0
        
        now = datetime.now()
        cutoff = now - timedelta(minutes=window_minutes)
        
        # Get recent values
        recent_values = []
        recent_times = []
        for value, ts in zip(self.values, self.timestamps):
            if ts >= cutoff:
                recent_values.append(value)
                recent_times.append(ts.timestamp())
        
        if len(recent_values) < 2:
            return 0.0
        
        # Simple linear regression slope
        n = len(recent_values)
        sum_x = sum(recent_times)
        sum_y = sum(recent_values)
        sum_xy = sum(x * y for x, y in zip(recent_times, recent_values))
        sum_x2 = sum(x * x for x in recent_times)
        
        denominator = n * sum_x2 - sum_x * sum_x
        if abs(denominator) < 1e-10:
            return 0.0
        
        slope = (n * sum_xy - sum_x * sum_y) / denominator
        
        # Normalize to -1 to 1 range (assuming ~1 minute between samples)
        return max(-1.0, min(1.0, slope * 60))


@dataclass
class StockSentiment:
    """Sentiment data for a single stock."""
    stock_code: str
    window: SentimentWindow = field(default_factory=SentimentWindow)
    article_count: int = 0
    last_update: datetime = field(default_factory=datetime.now)
    last_sentiment: float = 0.0
    policy_impact_count: int = 0
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "stock_code": self.stock_code,
            "current_sentiment": self.window.get_weighted_average(),
            "trend": self.window.get_trend(),
            "article_count": self.article_count,
            "policy_impact_count": self.policy_impact_count,
            "last_update": self.last_update.isoformat(),
            "last_sentiment": self.last_sentiment,
        }


class SentimentStreamProcessor:
    """Real-time sentiment stream processor.
    
    Features:
    - Per-stock sentiment tracking
    - Rolling windows with exponential decay
    - Trend detection
    - Policy impact alerts
    - Market-wide sentiment aggregation
    
    Args:
        decay_half_life: Half-life for sentiment decay
        alert_threshold: Sentiment change threshold for alerts
    """
    
    def __init__(
        self,
        decay_half_life: timedelta = timedelta(hours=2),
        alert_threshold: float = 0.3,
    ) -> None:
        self.decay_half_life = decay_half_life
        self.alert_threshold = alert_threshold
        
        # State
        self._running = False
        self._task: asyncio.Task | None = None
        
        # Sentiment storage
        self._stock_sentiments: dict[str, StockSentiment] = {}
        self._sentiments_lock = asyncio.Lock()
        
        # Market-wide sentiment
        self._market_window = SentimentWindow()
        self._policy_window = SentimentWindow()
        
        # Subscribers
        self._subscribers: list[callable] = []
        
        # Alert callbacks
        self._alert_callbacks: list[callable] = []
        
        # Statistics
        self._total_processed = 0
        self._alerts_triggered = 0
        self._start_time: datetime | None = None
        
    @property
    def is_running(self) -> bool:
        """Check if processor is running."""
        return self._running
    
    @property
    def stats(self) -> dict[str, Any]:
        """Get processing statistics."""
        return {
            "total_processed": self._total_processed,
            "alerts_triggered": self._alerts_triggered,
            "tracked_stocks": len(self._stock_sentiments),
            "uptime_seconds": (datetime.now() - self._start_time).total_seconds() if self._start_time else 0,
        }
    
    async def start(self) -> None:
        """Start sentiment processor."""
        if self._running:
            return
        
        log.info("Starting sentiment stream processor")
        self._running = True
        self._start_time = datetime.now()
        
        # Start cleanup task
        self._task = asyncio.create_task(self._cleanup_loop())
        log.info("Sentiment stream processor started")
    
    async def stop(self) -> None:
        """Stop sentiment processor."""
        if not self._running:
            return
        
        self._running = False
        
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        
        log.info("Sentiment stream processor stopped")
    
    def subscribe(self, callback: callable) -> None:
        """Subscribe to sentiment updates.
        
        Args:
            callback: Function(stock_code, sentiment_data)
        """
        self._subscribers.append(callback)
        log.info(f"Added sentiment subscriber (total: {len(self._subscribers)})")
    
    def on_alert(self, callback: callable) -> None:
        """Register alert callback.
        
        Args:
            callback: Function(stock_code, alert_type, data)
        """
        self._alert_callbacks.append(callback)
    
    async def process_article(self, article: Any) -> None:
        """Process news article and update sentiment.
        
        Args:
            article: NewsArticle object with sentiment_score
        """
        from data.news_collector import NewsArticle
        
        start_time = time.time()
        
        # Extract sentiment
        sentiment_score = float(getattr(article, 'sentiment_score', 0.0))
        is_policy = getattr(article, 'category', '') == 'policy'
        
        # Get mentioned entities (stocks)
        entities = getattr(article, 'entities', [])
        
        if not entities:
            # Apply to market-wide sentiment
            await self._update_market_sentiment(sentiment_score, is_policy)
            return
        
        # Update per-stock sentiment
        for entity in entities:
            stock_code = self._normalize_stock_code(entity)
            if stock_code:
                await self._update_stock_sentiment(
                    stock_code,
                    sentiment_score,
                    is_policy,
                    article.published_at,
                )
        
        # Notify subscribers
        await self._notify_subscribers()
        
        # Check for alerts
        await self._check_alerts(entities, sentiment_score)
        
        self._total_processed += 1
        
        latency_ms = (time.time() - start_time) * 1000
        if latency_ms > 100:
            log.debug(f"Sentiment processing latency: {latency_ms:.1f}ms")
    
    async def _update_stock_sentiment(
        self,
        stock_code: str,
        sentiment: float,
        is_policy: bool,
        timestamp: datetime,
    ) -> None:
        """Update sentiment for a stock.
        
        Args:
            stock_code: Stock code
            sentiment: Sentiment score (-1 to 1)
            is_policy: Whether article is policy-related
            timestamp: Article timestamp
        """
        async with self._sentiments_lock:
            if stock_code not in self._stock_sentiments:
                self._stock_sentiments[stock_code] = StockSentiment(stock_code=stock_code)
            
            stock_sentiment = self._stock_sentiments[stock_code]
            stock_sentiment.window.add(sentiment, timestamp, is_policy)
            stock_sentiment.article_count += 1
            stock_sentiment.last_update = datetime.now()
            stock_sentiment.last_sentiment = sentiment
            
            if is_policy:
                stock_sentiment.policy_impact_count += 1
    
    async def _update_market_sentiment(self, sentiment: float, is_policy: bool) -> None:
        """Update market-wide sentiment.
        
        Args:
            sentiment: Sentiment score
            is_policy: Whether policy-related
        """
        async with self._sentiments_lock:
            self._market_window.add(sentiment, datetime.now(), is_policy)
            if is_policy:
                self._policy_window.add(sentiment, datetime.now())
    
    async def _check_alerts(self, entities: list[str], sentiment: float) -> None:
        """Check for sentiment alerts.
        
        Args:
            entities: List of stock codes
            sentiment: Current sentiment
        """
        if abs(sentiment) < self.alert_threshold:
            return
        
        for entity in entities:
            stock_code = self._normalize_stock_code(entity)
            if not stock_code:
                continue
            
            async with self._sentiments_lock:
                if stock_code not in self._stock_sentiments:
                    continue
                
                stock_sentiment = self._stock_sentiments[stock_code]
                trend = stock_sentiment.window.get_trend()
                
                # Check for significant sentiment change
                if abs(trend) > 0.5:  # Strong trend
                    alert_type = "SURGE" if trend > 0 else "PLUNGE"
                    self._alerts_triggered += 1
                    
                    # Notify alert callbacks
                    for callback in self._alert_callbacks:
                        try:
                            callback(stock_code, alert_type, {
                                "sentiment": sentiment,
                                "trend": trend,
                                "article_count": stock_sentiment.article_count,
                            })
                        except Exception as e:
                            log.error(f"Alert callback error: {e}")
    
    async def _notify_subscribers(self) -> None:
        """Notify all subscribers of updates."""
        for subscriber in self._subscribers:
            try:
                subscriber()
            except Exception as e:
                log.error(f"Sentiment subscriber error: {e}")
    
    async def _cleanup_loop(self) -> None:
        """Periodic cleanup of stale data."""
        while self._running:
            try:
                await asyncio.sleep(300)  # Every 5 minutes
                
                async with self._sentiments_lock:
                    # Remove stocks with no recent updates
                    stale_threshold = datetime.now() - timedelta(hours=24)
                    stale_stocks = [
                        code for code, data in self._stock_sentiments.items()
                        if data.last_update < stale_threshold
                    ]
                    
                    for code in stale_stocks:
                        del self._stock_sentiments[code]
                    
                    if stale_stocks:
                        log.info(f"Cleaned up {len(stale_stocks)} stale stock sentiments")
                        
            except asyncio.CancelledError:
                break
            except Exception as e:
                log.exception(f"Sentiment cleanup error: {e}")
    
    @staticmethod
    def _normalize_stock_code(entity: str) -> str | None:
        """Normalize stock code from entity name.
        
        Args:
            entity: Entity name or stock code
            
        Returns:
            Normalized 6-digit stock code or None
        """
        import re
        
        entity = str(entity).strip()
        
        # Match 6-digit codes
        if re.match(r'^\d{6}$', entity):
            return entity
        
        # Match common formats: 000001.SZ, 600000.SH, etc.
        match = re.match(r'^(\d{6})\.(SZ|SH|SS)$', entity, re.IGNORECASE)
        if match:
            return match.group(1)

        # Match alternate exchange-prefix forms: SH600000, SZ000001.
        match = re.match(r'^(SH|SZ|SS)(\d{6})$', entity, re.IGNORECASE)
        if match:
            return match.group(2)

        # Minimal company/entity fallback map for common aliases.
        # This is intentionally small and deterministic to avoid false mapping.
        alias_map = {
            "PINGAN BANK": "000001",
            "PING AN BANK": "000001",
            "平安银行": "000001",
            "贵州茅台": "600519",
            "KWEICHOW MOUTAI": "600519",
            "浦发银行": "600000",
            "SHANGHAI PUDONG DEVELOPMENT BANK": "600000",
        }
        mapped = alias_map.get(entity.upper())
        if mapped:
            return mapped
        
        return None
    
    def get_sentiment(self, stock_code: str) -> dict[str, Any] | None:
        """Get current sentiment for a stock.
        
        Args:
            stock_code: Stock code
            
        Returns:
            Sentiment data dictionary or None
        """
        rec = self._stock_sentiments.get(stock_code)
        if rec is None:
            return None
        return rec.to_dict()
    
    def get_market_sentiment(self) -> dict[str, Any]:
        """Get current market-wide sentiment.
        
        Returns:
            Market sentiment data
        """
        return {
            "market_sentiment": self._market_window.get_weighted_average(self.decay_half_life),
            "market_trend": self._market_window.get_trend(),
            "policy_sentiment": self._policy_window.get_weighted_average(self.decay_half_life),
            "policy_trend": self._policy_window.get_trend(),
        }
    
    def get_top_sentiment(self, limit: int = 10, descending: bool = True) -> list[dict[str, Any]]:
        """Get stocks with top sentiment scores.
        
        Args:
            limit: Number of stocks to return
            descending: Sort by highest sentiment first
            
        Returns:
            List of (stock_code, sentiment_data) tuples
        """
        stocks = list(self._stock_sentiments.values())
        stocks.sort(
            key=lambda s: s.window.get_weighted_average(self.decay_half_life),
            reverse=descending,
        )
        return [s.to_dict() for s in stocks[:limit]]


# Singleton instance
_processor: SentimentStreamProcessor | None = None


def get_sentiment_processor() -> SentimentStreamProcessor:
    """Get or create sentiment processor instance."""
    global _processor
    if _processor is None:
        _processor = SentimentStreamProcessor()
    return _processor


def reset_sentiment_processor() -> None:
    """Reset processor instance."""
    global _processor
    if _processor:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            asyncio.run(_processor.stop())
        else:
            loop.create_task(_processor.stop())
    _processor = None
