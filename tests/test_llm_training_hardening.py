from __future__ import annotations

from datetime import datetime
from pathlib import Path

from data import llm_sentiment as llm_module
from data.news_collector import NewsArticle


def _article(article_id: str, text: str) -> NewsArticle:
    now = datetime.now()
    return NewsArticle(
        id=article_id,
        title=text,
        content=text,
        summary=text[:120],
        source="unit_test",
        url="https://example.test/article",
        published_at=now,
        collected_at=now,
        language="en",
        category="market",
    )


def test_train_bootstraps_transformer_labels_and_feature_signal(
    monkeypatch,
    tmp_path: Path,
) -> None:
    analyzer = llm_module.LLM_sentimentAnalyzer(cache_dir=tmp_path)
    load_calls = {"count": 0}
    feature_rows: list[tuple[float, float]] = []

    def _fake_load_pipeline() -> None:
        load_calls["count"] += 1

        def _fake_pipe(text: str) -> list[list[dict[str, float | str]]]:
            if "POSMARK" in text:
                return [[
                    {"label": "POSITIVE", "score": 0.95},
                    {"label": "NEGATIVE", "score": 0.03},
                    {"label": "NEUTRAL", "score": 0.02},
                ]]
            if "NEGMARK" in text:
                return [[
                    {"label": "POSITIVE", "score": 0.02},
                    {"label": "NEGATIVE", "score": 0.95},
                    {"label": "NEUTRAL", "score": 0.03},
                ]]
            return [[
                {"label": "POSITIVE", "score": 0.33},
                {"label": "NEGATIVE", "score": 0.33},
                {"label": "NEUTRAL", "score": 0.34},
            ]]

        analyzer._pipe = _fake_pipe
        analyzer._pipe_name = "fake_pipe"

    def _fake_build_features(
        article: NewsArticle,
        tf_overall: float,
        tf_conf: float,
        language: str,
    ) -> list[float]:
        _ = article
        _ = language
        feature_rows.append((float(tf_overall), float(tf_conf)))
        return [
            float(tf_overall),
            0.0,
            0.0,
            0.0,
            float(tf_conf),
            0.0,
            1.0,
            0.5,
        ]

    monkeypatch.setattr(analyzer, "_load_pipeline", _fake_load_pipeline)
    monkeypatch.setattr(analyzer, "_build_features", _fake_build_features)
    monkeypatch.setattr(llm_module, "_SKLEARN_AVAILABLE", False)
    monkeypatch.setattr(llm_module, "_SKLEARN_MLP_AVAILABLE", False)

    rows = [
        _article("p1", "POSMARK neutral training sample one"),
        _article("n1", "NEGMARK neutral training sample two"),
        _article("p2", "POSMARK neutral training sample three"),
        _article("n2", "NEGMARK neutral training sample four"),
    ]
    report = analyzer.train(
        rows,
        max_samples=12,
        use_transformer_labels=True,
        feature_scaling=False,
        validation_split=0.2,
    )

    assert load_calls["count"] == 1
    assert bool(report.get("transformer_labels_requested", False)) is True
    assert bool(report.get("transformer_labels_used", False)) is True
    assert feature_rows
    assert any(abs(score) >= 0.8 for score, _conf in feature_rows)
    assert any(conf >= 0.9 for _score, conf in feature_rows)
    dist = dict(report.get("class_distribution") or {})
    assert int(dist.get("positive", 0)) >= 1
    assert int(dist.get("negative", 0)) >= 1


def test_train_handles_extreme_validation_split_without_empty_train(
    monkeypatch,
    tmp_path: Path,
) -> None:
    analyzer = llm_module.LLM_sentimentAnalyzer(cache_dir=tmp_path)
    monkeypatch.setattr(llm_module, "_SKLEARN_AVAILABLE", False)
    monkeypatch.setattr(llm_module, "_SKLEARN_MLP_AVAILABLE", False)

    rows = [
        _article("p1", "bullish support growth alpha"),
        _article("p2", "bullish upside recovery beta"),
        _article("p3", "bullish support growth gamma"),
        _article("n1", "bearish downside risk delta"),
        _article("n2", "bearish downside risk epsilon"),
        _article("n3", "bearish downside risk zeta"),
    ]
    report = analyzer.train(
        rows,
        max_samples=10,
        validation_split=0.99,
        use_transformer_labels=False,
        feature_scaling=False,
    )

    assert int(report.get("train_samples", 0)) >= 2
    assert int(report.get("val_samples", 0)) >= 1


def test_auto_train_cache_fallback_deduplicates_article_ids(
    monkeypatch,
    tmp_path: Path,
) -> None:
    analyzer = llm_module.LLM_sentimentAnalyzer(cache_dir=tmp_path)
    trained_ids: list[str] = []

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
        "_build_auto_search_queries",
        lambda **_: [["ashare", "policy"]],
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
        "_filter_high_quality_articles",
        lambda rows, **_: (list(rows), {"input": len(rows), "kept": len(rows)}),
    )
    monkeypatch.setattr(
        analyzer,
        "_load_recent_cached_articles_for_training",
        lambda **_: [
            _article("dup-1", "cached sample one"),
            _article("dup-1", "cached sample duplicate"),
            _article("dup-2", "cached sample two"),
        ],
    )

    def _fake_train(rows: list[NewsArticle], *, max_samples: int = 1200) -> dict[str, object]:
        _ = max_samples
        trained_ids.extend(str(getattr(a, "id", "") or "") for a in rows)
        return {
            "status": "trained",
            "trained_samples": int(len(rows)),
            "training_architecture": "hybrid_neural_network",
        }

    monkeypatch.setattr(analyzer, "train", _fake_train)

    report = analyzer.auto_train_from_internet(
        hours_back=24,
        limit_per_query=20,
        max_samples=20,
        min_new_articles=2,
        auto_related_search=False,
        force_china_direct=False,
        only_new=False,
    )

    assert report["status"] == "trained"
    assert int(report.get("collected_articles", 0)) == 2
    assert len(trained_ids) == 2
    assert sorted(trained_ids) == ["dup-1", "dup-2"]
