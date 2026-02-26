from __future__ import annotations

from datetime import datetime
from pathlib import Path

from data import llm_sentiment as llm_module
from data.news_collector import NewsArticle


def _sample_article(article_id: str) -> NewsArticle:
    now = datetime.now()
    return NewsArticle(
        id=article_id,
        title=f"A-share policy support update {article_id}",
        content=f"Policy and market update for A-share stocks ({article_id}).",
        summary=f"Policy support update {article_id}.",
        source="unit_test",
        url="https://example.test/news",
        published_at=now,
        collected_at=now,
        language="en",
        category="market",
    )


def test_auto_train_from_internet_recovers_from_strict_empty_batch(
    monkeypatch,
    tmp_path: Path,
) -> None:
    analyzer = llm_module.LLM_sentimentAnalyzer(cache_dir=tmp_path)

    class _Collector:
        def __init__(self) -> None:
            self.calls: list[bool] = []

        def collect_news(
            self,
            *,
            keywords: list[str] | None = None,
            limit: int = 100,
            hours_back: int = 24,
            strict: bool = False,
        ) -> list[NewsArticle]:
            _ = (keywords, limit, hours_back)
            self.calls.append(bool(strict))
            if strict:
                raise RuntimeError("Strict news collection returned no articles")
            return [_sample_article("strict-fallback-1")]

    collector = _Collector()

    monkeypatch.setattr(llm_module, "get_collector", lambda: collector)
    monkeypatch.setattr(
        analyzer,
        "_build_auto_search_queries",
        lambda **_: [["ashare", "policy", "regulation"]],
    )
    monkeypatch.setattr(
        analyzer,
        "_collect_china_corpus_segments",
        lambda **_: (
            {
                "general_text": [],
                "policy_news": [],
                "stock_specific": [],
                "instruction_conversation": [],
            },
            [],
        ),
    )
    monkeypatch.setattr(
        analyzer,
        "train",
        lambda rows, *, max_samples=1200: {
            "status": "trained",
            "trained_samples": int(len(rows)),
            "training_architecture": "hybrid_neural_network",
            "max_samples_used": int(max_samples),
        },
    )

    report = analyzer.auto_train_from_internet(
        hours_back=24,
        limit_per_query=20,
        max_samples=80,
        min_new_articles=1,
        auto_related_search=False,
        force_china_direct=False,
        only_new=True,
    )

    assert report["status"] == "trained"
    assert int(report.get("collected_articles", 0)) == 1
    assert int(report.get("strict_batch_failures", 0)) == 1
    assert int(report.get("strict_batch_recoveries", 0)) == 1
    assert collector.calls == [True, False]


def test_auto_train_reports_china_corpus_breakdown(
    monkeypatch,
    tmp_path: Path,
) -> None:
    analyzer = llm_module.LLM_sentimentAnalyzer(cache_dir=tmp_path)

    class _Collector:
        @staticmethod
        def collect_news(
            *,
            keywords: list[str] | None = None,
            limit: int = 100,
            hours_back: int = 24,
            strict: bool = False,
        ) -> list[NewsArticle]:
            _ = (keywords, limit, hours_back, strict)
            return [_sample_article("search-1")]

    monkeypatch.setattr(llm_module, "get_collector", lambda: _Collector())
    monkeypatch.setattr(
        analyzer,
        "_build_auto_search_queries",
        lambda **_: [["ashare", "policy"]],
    )
    monkeypatch.setattr(
        analyzer,
        "_collect_china_corpus_segments",
        lambda **_: (
            {
                "general_text": [_sample_article("general-1")],
                "policy_news": [_sample_article("policy-1")],
                "stock_specific": [_sample_article("stock-1")],
                "instruction_conversation": [_sample_article("instr-1")],
            },
            ["600519"],
        ),
    )
    monkeypatch.setattr(
        analyzer,
        "train",
        lambda rows, *, max_samples=1200: {
            "status": "trained",
            "trained_samples": int(len(rows)),
            "training_architecture": "hybrid_neural_network",
            "max_samples_used": int(max_samples),
        },
    )

    report = analyzer.auto_train_from_internet(
        hours_back=24,
        limit_per_query=20,
        max_samples=80,
        min_new_articles=1,
        auto_related_search=False,
        force_china_direct=False,
        only_new=True,
    )

    assert report["status"] == "trained"
    assert int(report.get("collected_articles", 0)) == 5
    breakdown = dict(report.get("corpus_breakdown") or {})
    assert int(breakdown.get("search_news", 0)) == 1
    assert int(breakdown.get("general_text", 0)) == 1
    assert int(breakdown.get("policy_news", 0)) == 1
    assert int(breakdown.get("stock_specific", 0)) == 1
    assert int(breakdown.get("instruction_conversation", 0)) == 1


def test_auto_train_default_skips_gm_bootstrap(
    monkeypatch,
    tmp_path: Path,
) -> None:
    analyzer = llm_module.LLM_sentimentAnalyzer(cache_dir=tmp_path)
    gm_bootstrap_called = {"value": False}

    class _Collector:
        @staticmethod
        def collect_news(
            *,
            keywords: list[str] | None = None,
            limit: int = 100,
            hours_back: int = 24,
            strict: bool = False,
        ) -> list[NewsArticle]:
            _ = (keywords, limit, hours_back, strict)
            return []

    monkeypatch.setattr(llm_module, "get_collector", lambda: _Collector())
    monkeypatch.setattr(
        analyzer,
        "_load_related_codes_from_china_news",
        lambda **_: [],
    )
    monkeypatch.setattr(
        analyzer,
        "_load_recent_cycle_stock_codes",
        lambda **_: gm_bootstrap_called.update(value=True) or ["600519"],
    )
    monkeypatch.setattr(
        analyzer,
        "_collect_china_corpus_segments",
        lambda **_: (
            {
                "general_text": [],
                "policy_news": [],
                "stock_specific": [],
                "instruction_conversation": [],
            },
            [],
        ),
    )
    monkeypatch.setattr(
        analyzer,
        "_build_auto_search_queries",
        lambda **_: [["ashare", "policy"]],
    )

    report = analyzer.auto_train_from_internet(
        hours_back=24,
        limit_per_query=20,
        max_samples=80,
        min_new_articles=1,
        auto_related_search=True,
        allow_gm_bootstrap=False,
        force_china_direct=False,
        only_new=True,
    )

    assert report["status"] in {"no_new_data", "trained", "partial"}
    assert gm_bootstrap_called["value"] is False
