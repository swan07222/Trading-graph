from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from data.news import NewsAggregator, NewsItem, _BaseNewsFetcher


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


def test_base_news_fetcher_uses_existing_ca_bundle(tmp_path, monkeypatch):
    ca_path = tmp_path / "ca.pem"
    ca_path.write_text("dummy", encoding="utf-8")

    monkeypatch.setattr("certifi.where", lambda: str(ca_path), raising=False)
    monkeypatch.setattr(
        "requests.utils.DEFAULT_CA_BUNDLE_PATH",
        "Z:/missing/default.pem",
        raising=False,
    )
    monkeypatch.setattr(
        "ssl.get_default_verify_paths",
        lambda: SimpleNamespace(cafile="Z:/missing/sys.pem", capath="Z:/missing/capath"),
    )

    fetcher = _BaseNewsFetcher()
    assert str(fetcher._session.verify) == str(ca_path)


def test_base_news_fetcher_keeps_verify_enabled_when_all_ca_paths_invalid(monkeypatch):
    monkeypatch.setattr("certifi.where", lambda: "Z:/missing/certifi.pem", raising=False)
    monkeypatch.setattr(
        "requests.utils.DEFAULT_CA_BUNDLE_PATH",
        "Z:/missing/default.pem",
        raising=False,
    )
    monkeypatch.setattr(
        "ssl.get_default_verify_paths",
        lambda: SimpleNamespace(cafile="Z:/missing/sys.pem", capath="Z:/missing/capath"),
    )
    monkeypatch.delenv("TRADING_NEWS_ALLOW_INSECURE_TLS", raising=False)

    fetcher = _BaseNewsFetcher()
    assert fetcher._session.verify is True


def test_base_news_fetcher_allows_insecure_tls_only_with_env(monkeypatch):
    monkeypatch.setattr("certifi.where", lambda: "Z:/missing/certifi.pem", raising=False)
    monkeypatch.setattr(
        "requests.utils.DEFAULT_CA_BUNDLE_PATH",
        "Z:/missing/default.pem",
        raising=False,
    )
    monkeypatch.setattr(
        "ssl.get_default_verify_paths",
        lambda: SimpleNamespace(cafile="Z:/missing/sys.pem", capath="Z:/missing/capath"),
    )
    monkeypatch.setenv("TRADING_NEWS_ALLOW_INSECURE_TLS", "1")

    fetcher = _BaseNewsFetcher()
    assert fetcher._session.verify is False


def test_sentiment_summary_handles_mixed_timezone_publish_times(monkeypatch):
    agg = NewsAggregator()
    now_naive = datetime.now()
    now_aware = datetime.now(timezone.utc)
    sample = [
        NewsItem(
            title="aware source",
            source="sina",
            publish_time=now_aware - timedelta(minutes=10),
            sentiment_score=0.4,
            sentiment_label="positive",
            importance=0.8,
        ),
        NewsItem(
            title="naive source",
            source="tencent",
            publish_time=now_naive - timedelta(minutes=40),
            sentiment_score=-0.2,
            sentiment_label="negative",
            importance=0.6,
        ),
    ]

    monkeypatch.setattr(agg, "get_market_news", lambda *a, **k: sample)
    out = agg.get_sentiment_summary()

    assert out["total"] == 2
    assert "overall_sentiment" in out
    assert "average_age_hours" in out


def test_news_features_supports_zero_lookback_and_mixed_timezone(monkeypatch):
    agg = NewsAggregator()
    now_naive = datetime.now()
    now_aware = datetime.now(timezone.utc)
    sample = [
        NewsItem(
            title="aware fresh",
            source="sina",
            publish_time=now_aware - timedelta(minutes=5),
            sentiment_score=0.7,
            sentiment_label="positive",
            importance=0.9,
        ),
        NewsItem(
            title="naive fresh",
            source="tencent",
            publish_time=now_naive - timedelta(minutes=15),
            sentiment_score=-0.1,
            sentiment_label="neutral",
            importance=0.4,
        ),
    ]

    monkeypatch.setattr(agg, "get_market_news", lambda *a, **k: sample)
    out = agg.get_news_features(hours_lookback=0)

    assert "news_weighted_sentiment" in out
    assert "news_recency_score" in out
    assert 0.0 <= out["news_volume"] <= 1.0


def test_market_news_sort_accepts_mixed_timezone_publish_times(monkeypatch):
    agg = NewsAggregator()
    now_naive = datetime.now()
    now_aware = datetime.now(timezone.utc)
    mixed = [
        NewsItem(
            title="older naive",
            source="tencent",
            publish_time=now_naive - timedelta(minutes=9),
            sentiment_score=0.1,
            sentiment_label="neutral",
            importance=0.5,
        ),
        NewsItem(
            title="newer aware",
            source="tencent",
            publish_time=now_aware - timedelta(minutes=2),
            sentiment_score=0.2,
            sentiment_label="positive",
            importance=0.7,
        ),
    ]

    monkeypatch.setattr(
        "core.network.get_network_env",
        lambda: SimpleNamespace(
            tencent_ok=True, is_china_direct=False, eastmoney_ok=False
        ),
    )
    monkeypatch.setattr(agg._tencent, "fetch_market_news", lambda count=20: mixed[:count])
    monkeypatch.setattr(agg._sina, "fetch_market_news", lambda count=20: [])
    monkeypatch.setattr(agg._eastmoney, "fetch_policy_news", lambda count=20: [])

    out = agg.get_market_news(count=2, force_refresh=True)

    assert len(out) == 2
    assert out[0].title == "newer aware"


def test_news_features_invalid_lookback_value_does_not_crash(monkeypatch):
    agg = NewsAggregator()
    sample = [
        NewsItem(
            title="stable sample",
            source="sina",
            publish_time=datetime.now() - timedelta(minutes=5),
            sentiment_score=0.3,
            sentiment_label="positive",
            importance=0.8,
        )
    ]
    monkeypatch.setattr(agg, "get_market_news", lambda *a, **k: sample)

    out = agg.get_news_features(hours_lookback="bad")

    assert "news_weighted_sentiment" in out


def test_sentiment_summary_tolerates_non_numeric_fields(monkeypatch):
    agg = NewsAggregator()
    item = NewsItem(
        title="malformed payload",
        source="sina",
        publish_time=datetime.now() - timedelta(minutes=3),
        sentiment_score=0.1,
        sentiment_label="positive",
        importance=0.5,
    )
    item.sentiment_score = "bad-score"  # type: ignore[assignment]
    item.importance = "bad-importance"  # type: ignore[assignment]

    monkeypatch.setattr(agg, "get_market_news", lambda *a, **k: [item])
    summary = agg.get_sentiment_summary()
    features = agg.get_news_features(hours_lookback=24)

    assert summary["total"] == 1
    assert "overall_sentiment" in summary
    assert "news_weighted_sentiment" in features
