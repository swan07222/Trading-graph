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


def test_train_uses_moe_without_transformer_labels(
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
        _article("p1", "bullish upside recovery sample one"),
        _article("n1", "bearish downside risk sample two"),
        _article("p2", "bullish growth support sample three"),
        _article("n2", "bearish selloff tightening sample four"),
    ]
    report = analyzer.train(
        rows,
        max_samples=12,
        use_transformer_labels=True,
        feature_scaling=False,
        validation_split=0.2,
    )

    assert load_calls["count"] == 0
    assert str(report.get("training_architecture", "")) == "mixture_of_experts"
    assert bool(report.get("transformer_labels_requested", False)) is True
    assert bool(report.get("transformer_labels_used", False)) is False
    assert bool(report.get("moe_ready", False)) is True
    assert feature_rows
    assert any(abs(score) > 0.05 for score, _conf in feature_rows)
    assert any(conf > 0.3 for _score, conf in feature_rows)
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


def test_train_bootstraps_rank_labels_when_signal_is_flat(
    monkeypatch,
    tmp_path: Path,
) -> None:
    analyzer = llm_module.LLM_sentimentAnalyzer(cache_dir=tmp_path)
    monkeypatch.setattr(llm_module, "_SKLEARN_AVAILABLE", False)
    monkeypatch.setattr(llm_module, "_SKLEARN_MLP_AVAILABLE", False)

    rows = [_article(f"flat-{i}", "headline only neutral text") for i in range(8)]
    report = analyzer.train(
        rows,
        max_samples=20,
        use_transformer_labels=False,
        feature_scaling=False,
    )

    dist = dict(report.get("class_distribution") or {})
    assert int(dist.get("positive", 0)) >= 1
    assert int(dist.get("negative", 0)) >= 1
    assert "Rank-based bootstrap labels applied" in str(report.get("notes", ""))


def test_filter_force_relaxed_keeps_short_headlines(
    tmp_path: Path,
) -> None:
    analyzer = llm_module.LLM_sentimentAnalyzer(cache_dir=tmp_path)
    rows = [_article("short-1", "A")]
    out, stats = analyzer._filter_high_quality_articles(
        rows,
        hours_back=24,
        min_samples=50,
        force_relaxed=True,
    )
    assert len(out) == 1
    assert int(stats.get("drop_length", 0)) == 0


def test_train_logistic_fallback_without_multi_class_kwarg(
    monkeypatch,
    tmp_path: Path,
) -> None:
    analyzer = llm_module.LLM_sentimentAnalyzer(cache_dir=tmp_path)

    class _CompatLogReg:
        def __init__(self, **kwargs):
            if "multi_class" in kwargs:
                raise TypeError("unexpected keyword argument 'multi_class'")

        def fit(self, x, y):
            _ = (x, y)
            self.classes_ = [-1, 0, 1]
            return self

        def predict(self, x):
            return [0 for _ in range(len(x))]

    monkeypatch.setattr(llm_module, "LogisticRegression", _CompatLogReg)
    monkeypatch.setattr(llm_module, "_SKLEARN_AVAILABLE", True)
    monkeypatch.setattr(llm_module, "_SKLEARN_MLP_AVAILABLE", False)

    rows = [
        _article("p1", "bullish support growth alpha"),
        _article("p2", "bullish upside recovery beta"),
        _article("n1", "bearish downside risk gamma"),
        _article("n2", "bearish selloff risk delta"),
    ]
    report = analyzer.train(
        rows,
        max_samples=20,
        use_transformer_labels=False,
        feature_scaling=False,
    )

    assert bool(report.get("calibrator_ready", False)) is True
    assert report.get("status") in {"trained", "partial"}


def test_analyze_prefers_moe_artifact_when_available(
    tmp_path: Path,
) -> None:
    analyzer = llm_module.LLM_sentimentAnalyzer(cache_dir=tmp_path)
    analyzer._moe_model = {
        "version": 1,
        "expert_names": ["rule"],
        "expert_models": {},
        "gate_model": None,
        "static_weights": [1.0],
    }
    article = _article("m1", "bullish support growth")
    out = analyzer.analyze(article, use_cache=False)
    assert str(out.model_used).startswith("moe(")


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


def test_generate_response_uses_local_chat_backend(
    monkeypatch,
    tmp_path: Path,
) -> None:
    analyzer = llm_module.LLM_sentimentAnalyzer(cache_dir=tmp_path)

    class _Resp:
        content = "Professional market response."

    class _LLM:
        async def initialize(self) -> None:
            return None

        async def generate(self, *, prompt: str, system_prompt: str, history):
            _ = (prompt, system_prompt, history)
            return _Resp()

    monkeypatch.setattr(
        analyzer,
        "_ensure_chat_llm",
        lambda: _LLM(),
    )
    out = analyzer.generate_response(
        prompt="give me risk analysis",
        symbol="600519",
        app_state={"interval": "15m"},
        history=[],
    )
    assert bool(out.get("local_model_ready", False)) is True
    assert "Professional market response." in str(out.get("answer", ""))


def test_generate_response_fallback_when_chat_backend_fails(
    monkeypatch,
    tmp_path: Path,
) -> None:
    analyzer = llm_module.LLM_sentimentAnalyzer(cache_dir=tmp_path)
    monkeypatch.setattr(
        analyzer,
        "_ensure_chat_llm",
        lambda: (_ for _ in ()).throw(RuntimeError("chat backend missing")),
    )
    out = analyzer.generate_response(
        prompt="hello",
        symbol="600519",
        app_state={},
        history=[],
    )
    assert bool(out.get("local_model_ready", True)) is False
    assert "not ready" in str(out.get("answer", "")).lower()


def test_train_chat_model_delegates_to_self_trainer(
    monkeypatch,
    tmp_path: Path,
) -> None:
    analyzer = llm_module.LLM_sentimentAnalyzer(cache_dir=tmp_path)

    captured: dict[str, object] = {}

    def _fake_train(cfg) -> dict[str, object]:
        captured["cfg"] = cfg
        return {
            "status": "trained",
            "model_dir": str(tmp_path / "chat_transformer"),
            "steps": 12,
        }

    monkeypatch.setattr("ai.self_chat_trainer.train_self_chat_model", _fake_train)
    out = analyzer.train_chat_model(
        chat_history_path=tmp_path / "history.json",
        max_steps=300,
        epochs=1,
    )
    assert str(out.get("status", "")) == "trained"
    assert int(out.get("steps", 0)) == 12
    cfg_obj = captured.get("cfg")
    assert cfg_obj is not None
