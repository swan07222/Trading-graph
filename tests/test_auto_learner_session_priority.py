from datetime import datetime, timedelta
from types import SimpleNamespace

import pandas as pd

from models.auto_learner import ContinuousLearner, LearningProgress


def test_session_continuous_window_seconds_counts_contiguous_1m_bars(monkeypatch) -> None:
    start = datetime(2026, 1, 5, 9, 30, 0)
    idx = pd.date_range(start=start, periods=61, freq="1min")
    df = pd.DataFrame(
        {
            "open": [10.0] * 61,
            "high": [10.1] * 61,
            "low": [9.9] * 61,
            "close": [10.0] * 61,
            "volume": [1000.0] * 61,
        },
        index=idx,
    )

    class _FakeCache:
        def read_history(self, symbol, interval, bars=5000, final_only=False):  # noqa: ARG002
            return df.copy()

    monkeypatch.setattr(
        "data.session_cache.get_session_bar_cache",
        lambda: _FakeCache(),
        raising=True,
    )

    learner = ContinuousLearner.__new__(ContinuousLearner)
    span = learner._session_continuous_window_seconds("600519", "1m")
    assert span >= 3600.0


def test_filter_priority_session_codes_requires_one_hour_intraday() -> None:
    learner = ContinuousLearner.__new__(ContinuousLearner)
    learner.progress = LearningProgress()

    spans = {"600519": 4000.0, "000001": 1500.0}
    learner._session_continuous_window_seconds = (  # type: ignore[method-assign]
        lambda code, interval, max_bars=5000: spans.get(str(code), 0.0)
    )

    out = learner._filter_priority_session_codes(
        ["600519", "000001", "600519"],
        "1m",
        min_seconds=3600.0,
    )

    assert out == ["600519"]
    assert any(
        "Skipped 1 session-priority stocks" in msg
        for msg in learner.progress.warnings
    )


def test_filter_priority_session_codes_keeps_dedup_for_non_intraday() -> None:
    learner = ContinuousLearner.__new__(ContinuousLearner)
    learner.progress = LearningProgress()
    out = learner._filter_priority_session_codes(
        ["600519", "600519", "000001"],
        "1d",
        min_seconds=3600.0,
    )
    assert out == ["600519", "000001"]


def test_prioritize_codes_by_news_promotes_symbols_with_recent_mentions(monkeypatch) -> None:
    now = datetime(2026, 1, 6, 10, 0, 0)

    class _FakeAgg:
        def get_market_news(self, count=80, force_refresh=False):  # noqa: ARG002
            return [
                SimpleNamespace(
                    stock_codes=["000001"],
                    sentiment_score=0.8,
                    importance=1.2,
                    publish_time=now - timedelta(minutes=20),
                ),
                SimpleNamespace(
                    stock_codes=["600519"],
                    sentiment_score=0.1,
                    importance=0.6,
                    publish_time=now - timedelta(hours=6),
                ),
            ]

        def get_sentiment_summary(self, stock_code):  # noqa: ARG002
            return {"total": 0, "confidence": 0.0, "overall_sentiment": 0.0}

    monkeypatch.setattr(
        "data.news.get_news_aggregator",
        lambda: _FakeAgg(),
        raising=True,
    )

    learner = ContinuousLearner.__new__(ContinuousLearner)
    learner.progress = LearningProgress()
    learner._should_stop = lambda: False  # type: ignore[method-assign]
    learner._update = lambda **_kw: None  # type: ignore[method-assign]

    out = learner._prioritize_codes_by_news(
        ["600519", "000001", "300750"],
        "1m",
        max_probe=0,
    )

    assert out[0] == "000001"
    assert sorted(out) == ["000001", "300750", "600519"]


def test_prioritize_codes_by_news_keeps_order_when_no_news(monkeypatch) -> None:
    class _FakeAgg:
        def get_market_news(self, count=80, force_refresh=False):  # noqa: ARG002
            return []

        def get_sentiment_summary(self, stock_code):  # noqa: ARG002
            return {"total": 0, "confidence": 0.0, "overall_sentiment": 0.0}

    monkeypatch.setattr(
        "data.news.get_news_aggregator",
        lambda: _FakeAgg(),
        raising=True,
    )

    learner = ContinuousLearner.__new__(ContinuousLearner)
    learner.progress = LearningProgress()
    learner._should_stop = lambda: False  # type: ignore[method-assign]
    learner._update = lambda **_kw: None  # type: ignore[method-assign]

    original = ["600519", "000001", "300750"]
    out = learner._prioritize_codes_by_news(original, "1m", max_probe=0)
    assert out == original
