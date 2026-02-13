from datetime import datetime, timedelta

from data.news import NewsAggregator, NewsItem


def test_sentiment_summary_uses_weighted_mode(monkeypatch):
    agg = NewsAggregator()
    now = datetime.now()
    sample = [
        NewsItem(
            title="fresh positive",
            source="sina",
            publish_time=now - timedelta(minutes=20),
            sentiment_score=0.8,
            sentiment_label="positive",
            importance=0.9,
        ),
        NewsItem(
            title="old negative",
            source="tencent",
            publish_time=now - timedelta(hours=30),
            sentiment_score=-0.9,
            sentiment_label="negative",
            importance=0.3,
        ),
    ]

    monkeypatch.setattr(agg, "get_market_news", lambda *a, **k: sample)
    summary = agg.get_sentiment_summary()

    assert summary["weighted"] is True
    assert summary["total"] == 2
    assert "simple_sentiment" in summary
    assert "confidence" in summary
    assert summary["overall_sentiment"] > summary["simple_sentiment"]
