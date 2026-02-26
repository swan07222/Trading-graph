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
import re
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable

from utils.logger import get_logger

log = get_logger(__name__)


def _resolve_publish_time(article: Any) -> datetime:
    """Extract publish timestamp from either NewsArticle or NewsItem.

    NewsArticle uses ``published_at``; NewsItem uses ``publish_time``.
    Returns ``datetime.now()`` when neither is available.
    """
    for attr in ("published_at", "publish_time"):
        val = getattr(article, attr, None)
        if isinstance(val, datetime):
            return val
    return datetime.now()


def _resolve_entities(article: Any) -> list[str]:
    """Extract entity/stock-code list from either NewsArticle or NewsItem.

    - ``NewsArticle.entities`` may be ``list[str]`` or ``list[dict]``
      (the LLM analyzer creates dicts with a ``"text"`` key).
    - ``NewsItem.stock_codes`` is ``list[str]``.

    Returns a flat ``list[str]`` of entity names or stock codes.
    Entities are deduplicated using a seen set; stock_codes are processed
    first, then entities, with earlier entries taking precedence.
    """
    codes: list[str] = []
    seen: set[str] = set()

    # Try stock_codes first (NewsItem)
    raw_codes = getattr(article, "stock_codes", None)
    if isinstance(raw_codes, list):
        for c in raw_codes:
            s = str(c or "").strip()
            if s and s not in seen:
                seen.add(s)
                codes.append(s)

    # Then try entities (NewsArticle)
    raw_entities = getattr(article, "entities", None)
    if isinstance(raw_entities, list):
        for e in raw_entities:
            if isinstance(e, dict):
                s = str(e.get("text", "") or "").strip()
            else:
                s = str(e or "").strip()
            if s and s not in seen:
                seen.add(s)
                codes.append(s)

    return codes


@dataclass
class SentimentWindow:
    """Rolling sentiment window with decay."""

    values: deque = field(default_factory=lambda: deque(maxlen=100))
    timestamps: deque = field(default_factory=lambda: deque(maxlen=100))
    policy_values: deque = field(default_factory=lambda: deque(maxlen=50))

    def add(
        self, value: float, timestamp: datetime, is_policy: bool = False
    ) -> None:
        """Add sentiment value to window."""
        self.values.append(value)
        self.timestamps.append(timestamp)
        if is_policy:
            self.policy_values.append(value)

    def get_weighted_average(
        self, decay_half_life: timedelta = timedelta(hours=2)
    ) -> float:
        """Calculate time-decayed weighted average."""
        if not self.values:
            return 0.0

        now = datetime.now()
        weights: list[float] = []
        values: list[float] = []
        half_life_seconds = max(1.0, decay_half_life.total_seconds())

        for value, ts in zip(self.values, self.timestamps):
            age = now - ts
            weight = 0.5 ** (age.total_seconds() / half_life_seconds)
            weights.append(weight)
            values.append(value)

        total_weight = sum(weights)
        if total_weight <= 0:
            return 0.0

        return sum(w * v for w, v in zip(weights, values)) / total_weight

    def get_trend(self, window_minutes: int = 30) -> float:
        """Calculate sentiment trend (slope) over recent time window.
        
        Uses linear regression on sentiment values within the time window.
        The slope is multiplied by 60 to convert from per-second to per-minute
        rate of change, then clamped to [-1.0, 1.0] range.
        
        Positive values indicate improving sentiment; negative values indicate
        deteriorating sentiment.
        """
        if len(self.values) < 2:
            return 0.0

        now = datetime.now()
        cutoff = now - timedelta(minutes=window_minutes)

        recent_values: list[float] = []
        recent_times: list[float] = []
        for value, ts in zip(self.values, self.timestamps):
            if ts >= cutoff:
                recent_values.append(value)
                recent_times.append(ts.timestamp())

        if len(recent_values) < 2:
            return 0.0

        n = len(recent_values)
        sum_x = sum(recent_times)
        sum_y = sum(recent_values)
        sum_xy = sum(x * y for x, y in zip(recent_times, recent_values))
        sum_x2 = sum(x * x for x in recent_times)

        denominator = n * sum_x2 - sum_x * sum_x
        if abs(denominator) < 1e-10:
            return 0.0

        # slope is in units of sentiment-change per second; multiply by 60
        # to express as sentiment-change per minute for better readability
        slope = (n * sum_xy - sum_x * sum_y) / denominator
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

    Accepts both ``NewsArticle`` and ``NewsItem`` objects transparently
    via helper resolvers (_resolve_publish_time, _resolve_entities).
    """

    def __init__(
        self,
        decay_half_life: timedelta = timedelta(hours=2),
        alert_threshold: float = 0.3,
    ) -> None:
        self.decay_half_life = decay_half_life
        self.alert_threshold = alert_threshold

        self._running = False
        self._task: asyncio.Task | None = None

        self._stock_sentiments: dict[str, StockSentiment] = {}
        # Thread-safe lock for cross-thread/event-loop safety
        self._sentiments_lock = threading.Lock()

        self._market_window = SentimentWindow()
        self._policy_window = SentimentWindow()

        self._subscribers: list[Callable] = []
        self._alert_callbacks: list[Callable] = []

        self._total_processed = 0
        self._alerts_triggered = 0
        self._start_time: datetime | None = None

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "total_processed": self._total_processed,
            "alerts_triggered": self._alerts_triggered,
            "tracked_stocks": len(self._stock_sentiments),
            "uptime_seconds": (
                (datetime.now() - self._start_time).total_seconds()
                if self._start_time
                else 0
            ),
        }

    async def start(self) -> None:
        if self._running:
            return
        log.info("Starting sentiment stream processor")
        self._running = True
        self._start_time = datetime.now()
        self._task = asyncio.create_task(self._cleanup_loop())
        log.info("Sentiment stream processor started")

    async def stop(self) -> None:
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

    def subscribe(self, callback: Callable) -> None:
        self._subscribers.append(callback)
        log.info(
            f"Added sentiment subscriber (total: {len(self._subscribers)})"
        )

    def on_alert(self, callback: Callable) -> None:
        self._alert_callbacks.append(callback)

    async def process_article(self, article: Any) -> None:
        """Process news article and update sentiment.

        Accepts both ``NewsArticle`` (from news_collector) and
        ``NewsItem`` (from news_aggregator) transparently.
        """
        start_time = time.time()

        sentiment_score = float(
            getattr(article, "sentiment_score", 0.0) or 0.0
        )
        # Validate and clamp sentiment score to valid range [-1, 1]
        if sentiment_score < -1.0 or sentiment_score > 1.0:
            log.warning(
                f"Sentiment score {sentiment_score} out of range [-1, 1], "
                f"clamping to valid range"
            )
            sentiment_score = max(-1.0, min(1.0, sentiment_score))

        is_policy = (
            str(getattr(article, "category", "") or "").lower() == "policy"
        )

        # Extract entities and publish time using resolvers that handle
        # both NewsArticle and NewsItem formats
        entities = _resolve_entities(article)
        published_at = _resolve_publish_time(article)

        if not entities:
            await self._update_market_sentiment(sentiment_score, is_policy)
            self._total_processed += 1
            return

        for entity in entities:
            stock_code = self._normalize_stock_code(entity)
            if stock_code:
                await self._update_stock_sentiment(
                    stock_code,
                    sentiment_score,
                    is_policy,
                    published_at,
                )

        await self._notify_subscribers()
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
        with self._sentiments_lock:
            if stock_code not in self._stock_sentiments:
                self._stock_sentiments[stock_code] = StockSentiment(
                    stock_code=stock_code
                )

            stock_sentiment = self._stock_sentiments[stock_code]
            stock_sentiment.window.add(sentiment, timestamp, is_policy)
            stock_sentiment.article_count += 1
            stock_sentiment.last_update = datetime.now()
            stock_sentiment.last_sentiment = sentiment

            if is_policy:
                stock_sentiment.policy_impact_count += 1

    async def _update_market_sentiment(
        self, sentiment: float, is_policy: bool
    ) -> None:
        with self._sentiments_lock:
            self._market_window.add(
                sentiment, datetime.now(), is_policy
            )
            if is_policy:
                self._policy_window.add(sentiment, datetime.now())

    async def _check_alerts(
        self, entities: list[str], sentiment: float
    ) -> None:
        if abs(sentiment) < self.alert_threshold:
            return

        # Track alerted stock codes to prevent duplicate alerts
        alerted_codes: set[str] = set()

        for entity in entities:
            stock_code = self._normalize_stock_code(entity)
            if not stock_code or stock_code in alerted_codes:
                continue

            with self._sentiments_lock:
                if stock_code not in self._stock_sentiments:
                    continue

                stock_sentiment = self._stock_sentiments[stock_code]
                trend = stock_sentiment.window.get_trend()

                if abs(trend) > 0.5:
                    alerted_codes.add(stock_code)
                    alert_type = "SURGE" if trend > 0 else "PLUNGE"
                    self._alerts_triggered += 1

                    for callback in self._alert_callbacks:
                        try:
                            callback(
                                stock_code,
                                alert_type,
                                {
                                    "sentiment": sentiment,
                                    "trend": trend,
                                    "article_count": stock_sentiment.article_count,
                                },
                            )
                        except Exception as e:
                            log.error(f"Alert callback error: {e}")

    async def _notify_subscribers(self) -> None:
        for subscriber in self._subscribers:
            try:
                subscriber()
            except Exception as e:
                log.error(f"Sentiment subscriber error: {e}")

    async def _cleanup_loop(self) -> None:
        while self._running:
            try:
                await asyncio.sleep(300)

                with self._sentiments_lock:
                    stale_threshold = datetime.now() - timedelta(hours=24)
                    stale_stocks = [
                        code
                        for code, data in self._stock_sentiments.items()
                        if data.last_update < stale_threshold
                    ]

                    for code in stale_stocks:
                        del self._stock_sentiments[code]

                    if stale_stocks:
                        log.info(
                            f"Cleaned up {len(stale_stocks)} stale stock sentiments"
                        )

            except asyncio.CancelledError:
                break
            except Exception as e:
                log.exception(f"Sentiment cleanup error: {e}")

    @staticmethod
    def _normalize_stock_code(entity: str) -> str | None:
        entity = str(entity).strip()

        if re.match(r"^\d{6}$", entity):
            return entity

        match = re.match(
            r"^(\d{6})\.(SZ|SH|SS)$", entity, re.IGNORECASE
        )
        if match:
            return match.group(1)

        match = re.match(
            r"^(SH|SZ|SS)(\d{6})$", entity, re.IGNORECASE
        )
        if match:
            return match.group(2)

        alias_map = {
            "PINGAN BANK": "000001",
            "PING AN BANK": "000001",
            "平安银行": "000001",
            "贵州茅台": "600519",
            "KWEICHOW MOUTAI": "600519",
            "浦发银行": "600000",
            "SHANGHAI PUDONG DEVELOPMENT BANK": "600000",
        }
        # Case-insensitive lookup: uppercase for English, original for Chinese
        entity_upper = entity.upper()
        mapped = alias_map.get(entity_upper)
        if mapped:
            return mapped
        # Also try original entity for Chinese character matching
        if entity != entity_upper:
            mapped = alias_map.get(entity)
            if mapped:
                return mapped

        return None

    def get_sentiment(self, stock_code: str) -> dict[str, Any] | None:
        rec = self._stock_sentiments.get(stock_code)
        if rec is None:
            return None
        return rec.to_dict()

    def get_market_sentiment(self) -> dict[str, Any]:
        return {
            "market_sentiment": self._market_window.get_weighted_average(
                self.decay_half_life
            ),
            "market_trend": self._market_window.get_trend(),
            "policy_sentiment": self._policy_window.get_weighted_average(
                self.decay_half_life
            ),
            "policy_trend": self._policy_window.get_trend(),
        }

    def get_top_sentiment(
        self, limit: int = 10, descending: bool = True
    ) -> list[dict[str, Any]]:
        stocks = list(self._stock_sentiments.values())
        stocks.sort(
            key=lambda s: s.window.get_weighted_average(
                self.decay_half_life
            ),
            reverse=descending,
        )
        return [s.to_dict() for s in stocks[:limit]]


# Thread-safe singleton pattern for global sentiment processor
_processor: SentimentStreamProcessor | None = None
_processor_lock = threading.Lock()


def get_sentiment_processor() -> SentimentStreamProcessor:
    global _processor
    if _processor is None:
        with _processor_lock:
            if _processor is None:
                _processor = SentimentStreamProcessor()
    return _processor


def reset_sentiment_processor() -> None:
    """Reset the global sentiment processor instance.
    
    Safely stops the current processor (if running) and clears
    the global reference. Must be called from the same event loop
    context where the processor was started.
    """
    global _processor
    with _processor_lock:
        if _processor is None:
            return
        processor_to_stop = _processor
        _processor = None
    
    # Stop the processor outside the lock to avoid deadlocks
    try:
        loop = asyncio.get_running_loop()
        # Schedule stop on the running loop
        task = loop.create_task(processor_to_stop.stop())
        # Don't await here - let it run in background
    except RuntimeError:
        # No running loop, create a new one to stop
        try:
            asyncio.run(processor_to_stop.stop())
        except Exception as e:
            log.warning(f"Error stopping sentiment processor: {e}")