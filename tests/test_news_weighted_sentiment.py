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


def test_news_fusion_diagnostics_and_features(monkeypatch):
    agg = NewsAggregator()
    now = datetime.now()
    sample = [
        NewsItem(
            title="same headline",
            source="sina",
            publish_time=now - timedelta(minutes=8),
            sentiment_score=0.7,
            sentiment_label="positive",
            importance=0.9,
            category="market",
        ),
        NewsItem(
            title="same headline",
            source="sina",
            publish_time=now - timedelta(minutes=25),
            sentiment_score=0.5,
            sentiment_label="positive",
            importance=0.6,
            category="market",
        ),
        NewsItem(
            title="policy pressure",
            source="tencent",
            publish_time=now - timedelta(hours=2),
            sentiment_score=-0.4,
            sentiment_label="negative",
            importance=0.8,
            category="policy",
        ),
    ]

    monkeypatch.setattr(agg, "get_market_news", lambda *a, **k: sample)
    summary = agg.get_sentiment_summary()
    features = agg.get_news_features(hours_lookback=24)

    assert "source_entropy" in summary
    assert "source_concentration_hhi" in summary
    assert "novelty_score" in summary
    assert "sentiment_momentum_6h" in summary
    assert summary["fusion_version"] == "2.1"
    assert 0.0 <= summary["source_entropy"] <= 1.0
    assert 0.0 <= summary["novelty_score"] <= 1.0

    assert "news_sentiment_confidence" in features
    assert "news_source_entropy" in features
    assert "news_source_concentration_hhi" in features
    assert "news_novelty_score" in features
    assert "news_recent_momentum" in features
    assert "news_importance_weighted_sentiment" in features
    assert "news_weighted_vs_simple_gap" in features
    assert "news_average_age_hours" in features
    assert 0.0 <= features["news_disagreement_penalty"] <= 1.0


def test_sentiment_entropy_zero_for_single_source(monkeypatch):
    agg = NewsAggregator()
    now = datetime.now()
    sample = [
        NewsItem(
            title="single source alpha",
            source="sina",
            publish_time=now - timedelta(minutes=5),
            sentiment_score=0.6,
            sentiment_label="positive",
            importance=0.8,
        ),
        NewsItem(
            title="single source beta",
            source="sina",
            publish_time=now - timedelta(minutes=12),
            sentiment_score=0.2,
            sentiment_label="positive",
            importance=0.6,
        ),
    ]

    monkeypatch.setattr(agg, "get_market_news", lambda *a, **k: sample)
    summary = agg.get_sentiment_summary()

    assert summary["source_entropy"] == 0.0
