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


def test_collect_news_retries_opposite_pool_when_primary_empty(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    collector = NewsCollector(cache_dir=tmp_path)
    calls: list[str] = []
    monkeypatch.setenv("TRADING_CHINA_DIRECT", "0")
    monkeypatch.setenv("TRADING_VPN", "1")

    collector.is_vpn_mode = lambda: True  # type: ignore[method-assign]

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
    assert "reuters" in calls
    assert "sina_finance" in calls


def test_collect_news_strict_does_not_retry_opposite_pool(tmp_path: Path) -> None:
    collector = NewsCollector(cache_dir=tmp_path)
    calls: list[str] = []

    collector.is_vpn_mode = lambda: True  # type: ignore[method-assign]

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

    assert "sina_finance" not in calls


def test_collect_news_china_direct_disables_cross_region_fallback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    collector = NewsCollector(cache_dir=tmp_path)
    calls: list[str] = []
    monkeypatch.setenv("TRADING_CHINA_DIRECT", "1")
    monkeypatch.setenv("TRADING_VPN", "0")

    collector.is_vpn_mode = lambda: False  # type: ignore[method-assign]

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
    assert "reuters" not in calls
    assert "yahoo_finance" not in calls
    assert "marketwatch" not in calls


def test_fetch_reuters_handles_timezone_aware_pubdate(tmp_path: Path) -> None:
    collector = NewsCollector(cache_dir=tmp_path)

    class _Session:
        headers: dict[str, str] = {}
        proxies: dict[str, str] = {}

        @staticmethod
        def get(url: str, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003
            _ = (args, kwargs)
            if "reuters.com/api/search" in url:
                return _JsonResp(
                    200,
                    {
                        "results": [
                            {
                                "title": "China stocks climb on policy support",
                                "description": "A-shares moved higher.",
                                "url": "/world/china/china-stocks",
                                "date": "2026-02-25T10:00:00Z",
                            }
                        ]
                    },
                )
            return _JsonResp(404, {})

    collector._session = _Session()
    rows = collector._fetch_reuters(
        keywords=["China stock"],
        start_time=datetime(2020, 1, 1),
        limit=5,
    )

    assert rows
    assert rows[0].source == "reuters"
    assert rows[0].published_at.tzinfo is None


def test_fetch_marketwatch_rss_feed_parses_articles(tmp_path: Path) -> None:
    collector = NewsCollector(cache_dir=tmp_path)
    rss_payload = (
        "<?xml version='1.0' encoding='UTF-8'?>"
        "<rss><channel>"
        "<item>"
        "<title>Global stock policy update</title>"
        "<description>Market response to new regulation.</description>"
        "<link>https://example.test/marketwatch/article</link>"
        "<pubDate>Wed, 25 Feb 2026 10:00:00 GMT</pubDate>"
        "</item>"
        "</channel></rss>"
    )

    class _Session:
        headers: dict[str, str] = {}
        proxies: dict[str, str] = {}

        @staticmethod
        def get(url: str, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003
            _ = (url, args, kwargs)
            return _Resp(200, rss_payload)

    collector._session = _Session()
    rows = collector._fetch_marketwatch(
        keywords=["policy"],
        start_time=datetime(2020, 1, 1),
        limit=5,
    )

    assert rows
    assert rows[0].source == "marketwatch"
    assert "policy" in rows[0].title.lower()
