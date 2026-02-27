"""Tests for LLM training data accumulation feature.

FIX: These tests verify that training data is properly accumulated
across cycles instead of being discarded after each cycle.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

from data import llm_sentiment as llm_module
from data.news_collector import NewsArticle


def _sample_article(
    article_id: str,
    hours_ago: int = 0,
    relevance: float = 0.5,
    language: str = "en",
    category: str = "market",
) -> NewsArticle:
    """Create a sample article for testing."""
    now = datetime.now()
    published_at = now - timedelta(hours=hours_ago)
    return NewsArticle(
        id=article_id,
        title=f"Test article {article_id}",
        content=f"Content for article {article_id}. Market update.",
        summary=f"Summary {article_id}",
        source="test_source",
        url=f"https://example.test/news/{article_id}",
        published_at=published_at,
        collected_at=now,
        language=language,
        category=category,
        sentiment_score=0.3,
        relevance_score=relevance,
    )


def test_save_to_training_corpus_persists_data(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Test that articles are saved to persistent corpus file."""
    analyzer = llm_module.LLM_sentimentAnalyzer(cache_dir=tmp_path)

    articles = [
        _sample_article("article-1", hours_ago=24, relevance=0.7),
        _sample_article("article-2", hours_ago=48, relevance=0.8),
        _sample_article("article-3", hours_ago=12, relevance=0.6),
    ]

    # Save to corpus
    stats = analyzer._save_to_training_corpus(articles, max_corpus_size=5000)

    # Verify save stats
    assert stats["saved"] == 3
    assert stats["skipped_duplicates"] == 0
    assert stats["corpus_size"] == 3

    # Verify file exists
    assert analyzer._training_corpus_path.exists()

    # Verify file contents
    with open(analyzer._training_corpus_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        assert len(lines) == 3

        # Verify each article was saved correctly
        saved_ids = set()
        for line in lines:
            data = json.loads(line.strip())
            saved_ids.add(data["id"])
            assert "title" in data
            assert "content" in data
            assert "language" in data
            assert "category" in data

        assert saved_ids == {"article-1", "article-2", "article-3"}


def test_load_training_corpus_returns_articles(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Test that articles can be loaded from persistent corpus."""
    analyzer = llm_module.LLM_sentimentAnalyzer(cache_dir=tmp_path)

    # First save some articles
    articles = [
        _sample_article("article-1", hours_ago=24, relevance=0.7),
        _sample_article("article-2", hours_ago=48, relevance=0.8),
        _sample_article("article-3", hours_ago=12, relevance=0.6),
    ]
    analyzer._save_to_training_corpus(articles, max_corpus_size=5000)

    # Load them back
    loaded = analyzer._load_training_corpus(max_samples=10, min_relevance=0.3)

    assert len(loaded) == 3
    loaded_ids = {getattr(a, "id", "") for a in loaded}
    assert loaded_ids == {"article-1", "article-2", "article-3"}


def test_load_training_corpus_filters_by_relevance(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Test that low-relevance articles are filtered out."""
    analyzer = llm_module.LLM_sentimentAnalyzer(cache_dir=tmp_path)

    # Save articles with different relevance scores
    articles = [
        _sample_article("high-rel", hours_ago=24, relevance=0.9),
        _sample_article("med-rel", hours_ago=24, relevance=0.5),
        _sample_article("low-rel", hours_ago=24, relevance=0.2),
    ]
    analyzer._save_to_training_corpus(articles, max_corpus_size=5000)

    # Load with high relevance threshold
    loaded = analyzer._load_training_corpus(max_samples=10, min_relevance=0.6)

    assert len(loaded) == 1
    assert getattr(loaded[0], "id", "") == "high-rel"


def test_load_training_corpus_prefers_recent(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Test that recent articles are preferred when loading."""
    analyzer = llm_module.LLM_sentimentAnalyzer(cache_dir=tmp_path)

    # Save articles with different ages but same relevance
    articles = [
        _sample_article("old", hours_ago=168, relevance=0.7),  # 1 week old
        _sample_article("recent", hours_ago=1, relevance=0.7),  # 1 hour old
        _sample_article("medium", hours_ago=24, relevance=0.7),  # 1 day old
    ]
    analyzer._save_to_training_corpus(articles, max_corpus_size=5000)

    # Load with preference for recent
    loaded = analyzer._load_training_corpus(
        max_samples=2,
        min_relevance=0.3,
        prefer_recent=True,
    )

    assert len(loaded) == 2
    # Most recent should be first
    assert getattr(loaded[0], "id", "") == "recent"


def test_prune_training_corpus_removes_oldest(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Test that pruning removes oldest entries."""
    analyzer = llm_module.LLM_sentimentAnalyzer(cache_dir=tmp_path)

    # Save more articles than max_size
    # Note: articles with smaller hours_ago values are "newer"
    articles = [
        _sample_article(f"article-{i}", hours_ago=i * 24, relevance=0.7)
        for i in range(1, 11)  # 10 articles, 1-10 days old
    ]
    analyzer._save_to_training_corpus(articles, max_corpus_size=5000)

    # Prune to 5 entries (should keep newest = smallest hours_ago values)
    prune_stats = analyzer._prune_training_corpus(max_size=5, min_age_hours=0)

    assert prune_stats["pruned"] == 5
    assert prune_stats["corpus_size"] == 5

    # Verify only newest articles remain (article-1 through article-5 are newest)
    with open(analyzer._training_corpus_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        assert len(lines) == 5

        saved_ids = {json.loads(line)["id"] for line in lines}
        # Should keep the 5 most recent (article-1 through article-5)
        # Note: article-1 is 1 day old, article-5 is 5 days old
        # article-6 through article-10 are older (6-10 days) and should be pruned
        assert saved_ids == {"article-1", "article-2", "article-3", "article-4", "article-5"}


def test_save_to_corpus_skips_duplicates(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Test that duplicate articles are not saved."""
    analyzer = llm_module.LLM_sentimentAnalyzer(cache_dir=tmp_path)

    # Save initial articles
    articles1 = [
        _sample_article("article-1", hours_ago=24),
        _sample_article("article-2", hours_ago=48),
    ]
    stats1 = analyzer._save_to_training_corpus(articles1, max_corpus_size=5000)
    assert stats1["saved"] == 2

    # Try to save same articles again plus a new one
    articles2 = [
        _sample_article("article-1", hours_ago=24),  # Duplicate
        _sample_article("article-3", hours_ago=12),  # New
    ]
    stats2 = analyzer._save_to_training_corpus(articles2, max_corpus_size=5000)

    assert stats2["saved"] == 1
    assert stats2["skipped_duplicates"] == 1
    assert stats2["corpus_size"] == 3


def test_get_corpus_stats_returns_statistics(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Test that corpus statistics are computed correctly."""
    analyzer = llm_module.LLM_sentimentAnalyzer(cache_dir=tmp_path)

    # Empty corpus
    stats = analyzer.get_corpus_stats()
    assert stats["total_articles"] == 0

    # Save mixed articles
    articles = [
        _sample_article("zh-1", language="zh", category="policy"),
        _sample_article("zh-2", language="zh", category="market"),
        _sample_article("en-1", language="en", category="company"),
    ]
    analyzer._save_to_training_corpus(articles, max_corpus_size=5000)

    # Check stats
    stats = analyzer.get_corpus_stats()
    assert stats["total_articles"] == 3
    assert stats["zh_articles"] == 2
    assert stats["en_articles"] == 1
    assert "policy" in stats["categories"]
    assert "market" in stats["categories"]
    assert "company" in stats["categories"]


def test_auto_train_accumulates_data_across_cycles(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """FIX: Main test - verify data accumulation across training cycles."""
    analyzer = llm_module.LLM_sentimentAnalyzer(cache_dir=tmp_path)

    # Mock the collector to return controlled batches
    cycle_batches = {
        1: [_sample_article(f"cycle1-{i}", hours_ago=24) for i in range(5)],
        2: [_sample_article(f"cycle2-{i}", hours_ago=12) for i in range(5)],
        3: [_sample_article(f"cycle3-{i}", hours_ago=6) for i in range(5)],
    }
    current_cycle = {"count": 0}

    def mock_collect(*args, **kwargs):
        current_cycle["count"] += 1
        cycle_num = current_cycle["count"]
        return cycle_batches.get(cycle_num, [])

    monkeypatch.setattr(
        analyzer,
        "collect_llm_corpus",
        lambda **kwargs: (mock_collect(), {"collected": 5}),
    )

    # Mock train to just return success
    trained_article_counts = []

    def mock_train(articles, **kwargs):
        trained_article_counts.append(len(articles))
        return {
            "status": "trained",
            "trained_samples": len(articles),
            "calibrator_ready": True,
            "hybrid_nn_ready": True,
        }

    monkeypatch.setattr(analyzer, "train", mock_train)

    # Run 3 training cycles with accumulation enabled
    for _ in range(3):
        report = analyzer.auto_train_from_internet(
            hours_back=24,
            limit_per_query=20,
            max_samples=100,
            min_new_articles=1,
            accumulate_training_data=True,
            corpus_boost_ratio=0.5,
            max_corpus_size=5000,
            only_new=True,
        )
        assert report["status"] == "trained"

    # Verify accumulation: each cycle should train on MORE data
    # Cycle 1: 5 new articles
    # Cycle 2: 5 new + ~2 historical = ~7
    # Cycle 3: 5 new + ~5 historical = ~10
    assert len(trained_article_counts) == 3

    # First cycle has no historical data
    assert trained_article_counts[0] == 5

    # Subsequent cycles should have accumulated data
    # (exact count depends on corpus_boost_ratio and sampling)
    assert trained_article_counts[1] >= 5  # At least new articles
    assert trained_article_counts[2] >= 5  # At least new articles

    # Total corpus should have all 15 unique articles
    corpus_stats = analyzer.get_corpus_stats()
    assert corpus_stats["total_articles"] == 15


def test_auto_train_without_accumulation(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Test that accumulation can be disabled."""
    analyzer = llm_module.LLM_sentimentAnalyzer(cache_dir=tmp_path)

    # Mock collector
    def mock_collect(*args, **kwargs):
        return [_sample_article(f"test-{i}", hours_ago=24) for i in range(5)]

    monkeypatch.setattr(
        analyzer,
        "collect_llm_corpus",
        lambda **kwargs: (mock_collect(), {"collected": 5}),
    )

    # Mock train
    def mock_train(articles, **kwargs):
        return {
            "status": "trained",
            "trained_samples": len(articles),
            "calibrator_ready": True,
            "hybrid_nn_ready": True,
        }

    monkeypatch.setattr(analyzer, "train", mock_train)

    # Run with accumulation disabled
    report = analyzer.auto_train_from_internet(
        hours_back=24,
        max_samples=100,
        min_new_articles=1,
        accumulate_training_data=False,  # Disabled
        only_new=True,
    )

    assert report["status"] == "trained"
    assert report["accumulation_enabled"] is False
    assert report["collected_articles"] == 5

    # Corpus should be empty (no accumulation)
    corpus_stats = analyzer.get_corpus_stats()
    assert corpus_stats["total_articles"] == 0


def test_auto_train_historical_fallback(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Test that historical data is used as fallback when no new data."""
    analyzer = llm_module.LLM_sentimentAnalyzer(cache_dir=tmp_path)

    # Pre-populate corpus
    historical = [
        _sample_article(f"hist-{i}", hours_ago=24 * (i + 1), relevance=0.8)
        for i in range(10)
    ]
    analyzer._save_to_training_corpus(historical, max_corpus_size=5000)

    # Mock collector to return very few articles (less than min_new_articles)
    # This triggers the historical fallback
    monkeypatch.setattr(
        analyzer,
        "collect_llm_corpus",
        lambda **kwargs: (
            [_sample_article("only-one-new", hours_ago=1)],  # Only 1 article
            {"collected": 1},
        ),
    )

    # Mock train
    def mock_train(articles, **kwargs):
        return {
            "status": "trained",
            "trained_samples": len(articles),
            "calibrator_ready": True,
            "hybrid_nn_ready": True,
        }

    monkeypatch.setattr(analyzer, "train", mock_train)

    # Run with accumulation - should use historical as fallback
    report = analyzer.auto_train_from_internet(
        hours_back=24,
        max_samples=100,
        min_new_articles=5,  # Require 5 articles (more than the 1 returned by mock)
        accumulate_training_data=True,
        only_new=True,
    )

    # Should have used historical fallback
    assert report["status"] == "trained"
    # Check that collection_stats indicates historical fallback was used
    assert report.get("collection_stats", {}).get("used_historical_fallback", False) is True
    assert int(report.get("collected_articles", 0)) == 1
    assert int(report.get("new_articles", 0)) == 1
    assert int(report.get("fallback_articles", 0)) >= 5
    assert int(report.get("historical_articles", 0)) == 0
    assert report["trained_samples"] >= 5


def test_auto_train_reports_query_count_from_collection_stats(
    monkeypatch,
    tmp_path: Path,
) -> None:
    analyzer = llm_module.LLM_sentimentAnalyzer(cache_dir=tmp_path)

    monkeypatch.setattr(
        analyzer,
        "collect_llm_corpus",
        lambda **kwargs: (
            [_sample_article("new-1", hours_ago=1)],
            {"collected": 1, "queries_used": 7},
        ),
    )
    monkeypatch.setattr(
        analyzer,
        "train",
        lambda rows, **kwargs: {
            "status": "trained",
            "trained_samples": len(rows),
            "calibrator_ready": True,
            "hybrid_nn_ready": True,
        },
    )

    report = analyzer.auto_train_from_internet(
        hours_back=24,
        max_samples=40,
        min_new_articles=1,
        accumulate_training_data=False,
        only_new=True,
    )

    assert report["status"] == "trained"
    assert int(report.get("query_count", 0)) == 7
