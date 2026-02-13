import time
from datetime import datetime

from analysis.sentiment import KeywordSentimentAnalyzer, NewsItem, NewsScraper


def test_keyword_sentiment_positive():
    analyzer = KeywordSentimentAnalyzer()
    r = analyzer.analyze("Company posts breakout and beats estimate with raised guidance")
    assert r.label == "positive"
    assert r.score > 0
    assert 0 <= r.confidence <= 1


def test_keyword_sentiment_negative_with_negation():
    analyzer = KeywordSentimentAnalyzer()
    r = analyzer.analyze("Company not in recovery and faces downgrade and lawsuit risk")
    assert r.label in {"negative", "neutral"}
    assert r.score <= 0


def test_news_scraper_custom_provider_registration():
    scraper = NewsScraper()
    scraper.register_provider(
        "custom",
        lambda: [
            NewsItem(
                title="custom positive",
                content="breakout and upgrade",
                source="custom",
                url="",
                timestamp=datetime.now(),
                stock_codes=["600519"],
                sentiment_score=0.8,
                sentiment_label="positive",
            )
        ],
        weight=1.5,
    )
    s, c = scraper.get_stock_sentiment("600519")
    assert c >= 0
    assert s >= 0
    assert scraper.unregister_provider("custom") is True


def test_news_scraper_offline_skips_builtin_providers():
    scraper = NewsScraper()
    scraper._providers["sina"] = lambda: (_ for _ in ()).throw(RuntimeError("should_skip_sina"))
    scraper._providers["eastmoney"] = lambda: (_ for _ in ()).throw(RuntimeError("should_skip_east"))
    scraper._network_available_cache = False
    scraper._last_network_check_ts = time.time()
    items = scraper.scrape_all()
    assert isinstance(items, list)
