from __future__ import annotations

from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
import requests

from data.news_collector import NewsArticle, NewsCollector


class _Resp:
    def __init__(self, status_code: int, text: str) -> None:
        self.status_code = int(status_code)
        self.text = str(text)

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(
                f"HTTP {self.status_code}",
                response=SimpleNamespace(status_code=self.status_code),
            )


class _JsonResp(_Resp):
    def __init__(self, status_code: int, payload: dict) -> None:
        super().__init__(status_code=status_code, text="")
        self._payload = dict(payload or {})

    def json(self) -> dict:
        return dict(self._payload)


def _article(source: str, title: str) -> NewsArticle:
    now = datetime.now()
    return NewsArticle(
        id=f"{source}-{title}",
        title=title,
        content=title,
        summary=title,
        source=source,
        url="https://example.test/article",
        published_at=now,
        collected_at=now,
        language="en",
        category="market",
    )


def test_fetch_eastmoney_falls_back_to_html_titles(tmp_path: Path) -> None:
    collector = NewsCollector(cache_dir=tmp_path)

    class _Session:
        headers: dict[str, str] = {}
        proxies: dict[str, str] = {}

        @staticmethod
        def get(url: str, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003
            _ = (args, kwargs)
            if "search-api-web.eastmoney.com" in url:
                return _Resp(200, "callback(not-json)")
            if "finance.eastmoney.com" in url:
                return _Resp(
                    200,
                    '<a href="/a/20260226.html">Eastmoney Market Headline</a>',
                )
            return _Resp(404, "")

    collector._session = _Session()
    out = collector._fetch_eastmoney(
        keywords=["stock"],
        start_time=datetime(2020, 1, 1),
        limit=5,
    )

    assert out
    assert out[0].source == "eastmoney"
    assert "Eastmoney Market Headline" in out[0].title


def test_fetch_caixin_falls_back_to_homepage_links(tmp_path: Path) -> None:
    collector = NewsCollector(cache_dir=tmp_path)

    class _Session:
        headers: dict[str, str] = {}
        proxies: dict[str, str] = {}

        @staticmethod
        def get(url: str, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003
            _ = (args, kwargs)
            if "api.caixin.com" in url:
                return _Resp(404, "")
            if "www.caixinglobal.com" in url:
                return _Resp(
                    200,
                    (
                        '<a href="https://www.caixinglobal.com/2026-02-25/'
                        'china-markets-open-higher-102416925.html">'
                        "China Markets Open Higher"
                        "</a>"
                    ),
                )
            return _Resp(404, "")

    collector._session = _Session()
    out = collector._fetch_caixin(
        keywords=["market"],
        start_time=datetime(2020, 1, 1),
        limit=5,
    )

    assert out
    assert out[0].source == "caixin"
    assert "China Markets Open Higher" in out[0].title


def test_collect_news_retries_alternative_sources_when_primary_empty(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Test that collector tries alternative China-accessible sources when primary is empty."""
    collector = NewsCollector(cache_dir=tmp_path)
    calls: list[str] = []

    def _fake_fetch(  # noqa: ANN001
        source: str,
        keywords: list[str] | None,
        start_time: datetime,
        limit: int,
    ) -> list[NewsArticle]:
        _ = (keywords, start_time, limit)
        calls.append(source)
        if source == "sina_finance":
            return [_article("sina_finance", "China market recovery")]
        return []

    collector._fetch_from_source = _fake_fetch  # type: ignore[method-assign]
    rows = collector.collect_news(limit=10, hours_back=24, strict=False)

    assert rows
    assert rows[0].source == "sina_finance"
    assert "sina_finance" in calls


def test_collect_news_strict_raises_on_empty_results(tmp_path: Path) -> None:
    """Test that strict mode raises error when no articles found."""
    collector = NewsCollector(cache_dir=tmp_path)
    calls: list[str] = []

    def _fake_fetch(  # noqa: ANN001
        source: str,
        keywords: list[str] | None,
        start_time: datetime,
        limit: int,
    ) -> list[NewsArticle]:
        _ = (keywords, start_time, limit)
        calls.append(source)
        return []

    collector._fetch_from_source = _fake_fetch  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="Strict news collection returned no articles"):
        collector.collect_news(limit=10, hours_back=24, strict=True)

    assert calls


def test_collect_news_china_direct_uses_china_sources(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Test that China-direct mode uses only China-accessible sources."""
    collector = NewsCollector(cache_dir=tmp_path)
    calls: list[str] = []
    monkeypatch.setenv("TRADING_CHINA_DIRECT", "1")

    def _fake_fetch(  # noqa: ANN001
        source: str,
        keywords: list[str] | None,
        start_time: datetime,
        limit: int,
    ) -> list[NewsArticle]:
        _ = (keywords, start_time, limit)
        calls.append(source)
        return []

    collector._fetch_from_source = _fake_fetch  # type: ignore[method-assign]
    rows = collector.collect_news(limit=10, hours_back=24, strict=False)

    assert rows == []
    assert calls
    # China-only mode should only use China-accessible sources
    assert "reuters" not in calls
    assert "yahoo_finance" not in calls
    assert "marketwatch" not in calls
