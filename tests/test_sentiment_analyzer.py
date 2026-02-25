from __future__ import annotations

from datetime import datetime, timedelta

from data.news_collector import NewsArticle
from data.sentiment_analyzer import SentimentAnalyzer


def _article(
    *,
    article_id: str,
    title: str,
    content: str,
    language: str = "en",
    category: str = "market",
    relevance: float = 1.0,
    entities: list[str] | None = None,
    hours_ago: int = 1,
) -> NewsArticle:
    published_at = datetime.now() - timedelta(hours=hours_ago)
    return NewsArticle(
        id=article_id,
        title=title,
        content=content,
        summary="",
        source="unit",
        url="https://example.com",
        published_at=published_at,
        collected_at=published_at,
        language=language,
        category=category,
        relevance_score=float(relevance),
        entities=list(entities or []),
        tags=[],
    )


def test_analyze_articles_uses_weight_denominator() -> None:
    analyzer = SentimentAnalyzer()
    a1 = _article(
        article_id="a1",
        title="profit rise support bull breakout",
        content="market rebound and recovery",
        relevance=1.0,
    )
    a2 = _article(
        article_id="a2",
        title="loss fall risk bear breakdown",
        content="negative warning",
        relevance=0.1,
    )

    s1 = analyzer.analyze_article(a1)
    s2 = analyzer.analyze_article(a2)
    out = analyzer.analyze_articles([a1, a2], hours_back=24)

    expected = ((s1.overall * 1.0) + (s2.overall * 0.1)) / 1.1
    assert abs(out.overall - expected) < 1e-6
    assert 0.0 <= out.confidence <= 1.0


def test_extract_entities_includes_explicit_and_company_like_matches() -> None:
    analyzer = SentimentAnalyzer()
    art = _article(
        article_id="e1",
        title="AAPL and 600519 surge after Climate Policy support",
        content="Alpha Tech Inc reports strong profit growth",
        language="en",
        category="policy",
        entities=["AAPL", "600519"],
    )

    entities = analyzer.extract_entities([art])
    by_name = {e.entity: e for e in entities}

    assert "600519" in by_name
    assert by_name["600519"].entity_type in {"company", "entity"}
    assert "AAPL" in by_name
    assert by_name["AAPL"].mention_count >= 1
    assert any(e.entity_type == "policy" for e in entities)
