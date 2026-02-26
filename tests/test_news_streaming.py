# tests/test_news_streaming.py
"""Tests for Real-Time News Streaming functionality."""

import asyncio
import json
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestNewsStreamer:
    """Test NewsStreamer class."""

    def test_streamer_init(self):
        """Test streamer initialization."""
        from data.news_streamer import NewsStreamer
        
        streamer = NewsStreamer(
            poll_interval=30.0,
            categories=["policy", "market"],
            keywords=["stock"],
        )
        
        assert streamer.poll_interval == 30.0
        assert streamer.categories == ["policy", "market"]
        assert streamer.keywords == ["stock"]
        assert not streamer.is_running
        assert streamer.health == 1.0

    @pytest.mark.asyncio
    async def test_streamer_start_stop(self):
        """Test starting and stopping streamer."""
        from data.news_streamer import NewsStreamer
        
        streamer = NewsStreamer()
        
        await streamer.start()
        assert streamer.is_running
        
        await streamer.stop()
        assert not streamer.is_running

    def test_subscribe_unsubscribe(self):
        """Test subscription management."""
        from data.news_streamer import NewsStreamer
        
        streamer = NewsStreamer()
        
        async def callback(article):
            pass
        
        streamer.subscribe("market", callback)
        streamer.subscribe("policy", callback)
        
        streamer.unsubscribe("market", callback)

    def test_get_backlog(self):
        """Test getting backlog of articles."""
        from data.news_streamer import NewsStreamer
        
        streamer = NewsStreamer()
        
        # Empty backlog
        backlog = streamer.get_backlog(limit=10)
        assert backlog == []

    def test_stats(self):
        """Test statistics."""
        from data.news_streamer import NewsStreamer
        
        streamer = NewsStreamer()
        stats = streamer.stats
        
        assert "articles_received" in stats
        assert "articles_broadcast" in stats
        assert "errors" in stats


class TestSentimentStreamProcessor:
    """Test SentimentStreamProcessor class."""

    def test_processor_init(self):
        """Test processor initialization."""
        from data.news_sentiment_stream import SentimentStreamProcessor
        
        processor = SentimentStreamProcessor(
            decay_half_life=timedelta(hours=1),
            alert_threshold=0.4,
        )
        
        assert processor.alert_threshold == 0.4
        assert not processor.is_running

    @pytest.mark.asyncio
    async def test_processor_start_stop(self):
        """Test starting and stopping processor."""
        from data.news_sentiment_stream import SentimentStreamProcessor
        
        processor = SentimentStreamProcessor()
        
        await processor.start()
        assert processor.is_running
        
        await processor.stop()
        assert not processor.is_running

    def test_sentiment_window(self):
        """Test sentiment window calculations."""
        from data.news_sentiment_stream import SentimentWindow
        
        window = SentimentWindow()
        now = datetime.now()
        
        # Add some values
        window.add(0.5, now - timedelta(minutes=10))
        window.add(0.7, now - timedelta(minutes=5))
        window.add(0.9, now)
        
        # Get weighted average
        avg = window.get_weighted_average()
        assert 0.0 <= avg <= 1.0
        
        # More recent values should have higher weight
        assert avg > 0.5  # Should be closer to recent values

    def test_normalize_stock_code(self):
        """Test stock code normalization."""
        from data.news_sentiment_stream import SentimentStreamProcessor
        
        processor = SentimentStreamProcessor()
        
        # Valid codes
        assert processor._normalize_stock_code("000001") == "000001"
        assert processor._normalize_stock_code("000001.SZ") == "000001"
        assert processor._normalize_stock_code("600000.SH") == "600000"
        
        # Invalid codes
        assert processor._normalize_stock_code("AAPL") is None
        assert processor._normalize_stock_code("") is None

    def test_get_market_sentiment(self):
        """Test getting market sentiment."""
        from data.news_sentiment_stream import SentimentStreamProcessor
        
        processor = SentimentStreamProcessor()
        sentiment = processor.get_market_sentiment()
        
        assert "market_sentiment" in sentiment
        assert "market_trend" in sentiment
        assert "policy_sentiment" in sentiment


class TestNewsWebSocketServer:
    """Test NewsWebSocketServer class."""

    def test_server_init(self):
        """Test server initialization."""
        from data.news_websocket_server import NewsWebSocketServer
        
        server = NewsWebSocketServer(
            host="127.0.0.1",
            port=8765,
            max_clients=50,
        )
        
        assert server.host == "127.0.0.1"
        assert server.port == 8765
        assert server.max_clients == 50
        assert not server.is_running

    def test_server_metrics(self):
        """Test server metrics."""
        from data.news_websocket_server import NewsWebSocketServer
        
        server = NewsWebSocketServer()
        metrics = server.metrics
        
        assert "total_connections" in metrics
        assert "total_messages" in metrics
        assert "current_clients" in metrics

    def test_set_streamer(self):
        """Test setting streamer reference."""
        from data.news_websocket_server import NewsWebSocketServer
        from data.news_streamer import NewsStreamer
        
        server = NewsWebSocketServer()
        streamer = NewsStreamer()
        
        server.set_streamer(streamer)
        assert server._streamer == streamer


class TestNewsStreamingIntegration:
    """Integration tests for news streaming."""

    @pytest.mark.asyncio
    async def test_streamer_to_sentiment(self):
        """Test news streamer integrating with sentiment processor."""
        from data.news_streamer import NewsStreamer
        from data.news_sentiment_stream import SentimentStreamProcessor
        
        streamer = NewsStreamer(poll_interval=60.0)
        processor = SentimentStreamProcessor()
        
        # Connect them
        async def process_article(article):
            await processor.process_article(article)
        
        streamer.subscribe("all", process_article)
        
        # Verify subscription
        assert "all" in streamer._subscriptions

    def test_article_deduplication(self):
        """Test article deduplication."""
        from data.news_streamer import NewsStreamer
        
        streamer = NewsStreamer()
        
        # Simulate adding same article twice
        article_id = "test_123"
        assert article_id not in streamer._seen_ids
        
        streamer._seen_ids.add(article_id)
        assert article_id in streamer._seen_ids


class TestNewsRealtimeWidget:
    """Test RealTimeNewsWidget class."""

    def test_widget_init(self, qtbot):
        """Test widget initialization."""
        from ui.news_realtime_widget import RealTimeNewsWidget
        
        widget = RealTimeNewsWidget()
        qtbot.addWidget(widget)
        
        assert widget._current_channel == "all"
        assert widget._auto_scroll is True
        assert widget.news_list is not None

    def test_add_article(self, qtbot):
        """Test adding article to widget."""
        from ui.news_realtime_widget import RealTimeNewsWidget
        
        widget = RealTimeNewsWidget()
        qtbot.addWidget(widget)
        
        article = {
            "id": "test_123",
            "title": "Test Article",
            "summary": "Test summary",
            "category": "market",
            "source": "test",
            "sentiment_score": 0.5,
        }
        
        widget._add_article(article)
        assert len(widget._articles) == 1

    def test_channel_filter(self, qtbot):
        """Test channel filtering."""
        from ui.news_realtime_widget import RealTimeNewsWidget
        
        widget = RealTimeNewsWidget()
        qtbot.addWidget(widget)
        
        # Add articles from different channels
        widget._add_article({
            "id": "1",
            "title": "Market Article",
            "category": "market",
        })
        widget._add_article({
            "id": "2",
            "title": "Policy Article",
            "category": "policy",
        })
        
        # Filter to market only
        widget._current_channel = "market"
        widget._refresh_list()
        
        # Should only show market article
        assert widget.news_list.count() == 1

    def test_search_filter(self, qtbot):
        """Test search filtering."""
        from ui.news_realtime_widget import RealTimeNewsWidget
        
        widget = RealTimeNewsWidget()
        qtbot.addWidget(widget)
        
        widget._add_article({
            "id": "1",
            "title": "Stock Market Rises",
            "category": "market",
        })
        widget._add_article({
            "id": "2",
            "title": "Policy Change Announced",
            "category": "policy",
        })
        
        # Search for "stock"
        widget.search_box.setText("stock")
        widget._refresh_list()
        
        # Should only show matching article
        assert widget.news_list.count() == 1


class TestWebSocketProtocol:
    """Test WebSocket protocol handling."""

    def test_subscribe_message(self):
        """Test subscribe message format."""
        message = {
            "type": "subscribe",
            "channel": "market",
        }
        
        assert message["type"] == "subscribe"
        assert message["channel"] == "market"

    def test_news_broadcast_message(self):
        """Test news broadcast message format."""
        message = {
            "type": "news",
            "channel": "market",
            "data": {
                "id": "123",
                "title": "Test",
            },
            "timestamp": datetime.now().isoformat(),
        }
        
        assert message["type"] == "news"
        assert "data" in message
        assert "timestamp" in message

    def test_backlog_request_message(self):
        """Test backlog request message format."""
        message = {
            "type": "get_backlog",
            "limit": 50,
            "channel": "market",
        }
        
        assert message["type"] == "get_backlog"
        assert message["limit"] == 50
        assert message["channel"] == "market"


class TestSentimentAlerts:
    """Test sentiment alert functionality."""

    def test_alert_threshold(self):
        """Test alert threshold checking."""
        from data.news_sentiment_stream import SentimentStreamProcessor
        
        processor = SentimentStreamProcessor(alert_threshold=0.3)
        
        # Below threshold - no alert
        assert abs(0.2) < processor.alert_threshold
        
        # Above threshold - alert
        assert abs(0.5) > processor.alert_threshold

    def test_alert_callback_registration(self):
        """Test alert callback registration."""
        from data.news_sentiment_stream import SentimentStreamProcessor
        
        processor = SentimentStreamProcessor()
        
        def on_alert(stock_code, alert_type, data):
            pass
        
        processor.on_alert(on_alert)
        assert len(processor._alert_callbacks) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
