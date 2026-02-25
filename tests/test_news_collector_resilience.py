from __future__ import annotations

from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import requests

from data.news_collector import NewsCollector


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
