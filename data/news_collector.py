# data/news_collector.py
"""News and Policy Data Collector for China-only mode.

This module collects news, policy, and regulatory data from China-accessible sources:
- EastMoney (eastmoney.com)
- Sina Finance (finance.sina.com.cn)
- Caixin (caixin.com)

All sources are accessible from mainland China without VPN.
"""

import hashlib
import html
import json
import re
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import requests

from config.settings import CONFIG
from utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class NewsArticle:
    """Normalized news article structure."""
    id: str
    title: str
    content: str
    summary: str
    source: str
    url: str
    published_at: datetime
    collected_at: datetime
    language: str  # 'zh'
    category: str  # 'policy', 'market', 'company', 'regulatory'
    sentiment_score: float = 0.0
    relevance_score: float = 0.0
    entities: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "content": self.content,
            "summary": self.summary,
            "source": self.source,
            "url": self.url,
            "published_at": self.published_at.isoformat(),
            "collected_at": self.collected_at.isoformat(),
            "language": self.language,
            "category": self.category,
            "sentiment_score": self.sentiment_score,
            "relevance_score": self.relevance_score,
            "entities": self.entities,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "NewsArticle":
        try:
            published_at = datetime.fromisoformat(data["published_at"])
        except (KeyError, ValueError, TypeError):
            published_at = datetime.now()
        try:
            collected_at = datetime.fromisoformat(data["collected_at"])
        except (KeyError, ValueError, TypeError):
            collected_at = datetime.now()
        return cls(
            id=data["id"],
            title=data["title"],
            content=data["content"],
            summary=data["summary"],
            source=data["source"],
            url=data["url"],
            published_at=published_at,
            collected_at=collected_at,
            language=data["language"],
            category=data["category"],
            sentiment_score=float(data.get("sentiment_score", 0.0)),
            relevance_score=float(data.get("relevance_score", 0.0)),
            entities=data.get("entities", []),
            tags=data.get("tags", []),
        )


class NewsCollector:
    """China-only news collector."""

    # China-accessible sources
    CHINA_SOURCES = [
        "eastmoney",
        "sina_finance",
        "caixin",
    ]

    _MIN_HEALTH_FOR_IMMEDIATE_RETRY = 0.20
    _MAX_SOURCE_COOLDOWN_SECONDS = 1800.0

    POLICY_KEYWORDS_ZH = [
        "政策", "规定", "监管", "证监会", "央行", "财政部",
        "货币政策", "财政政策", "产业政策", "法规", "条例",
        "新股", "IPO", "退市", "交易规则", "印花税",
    ]

    MARKET_KEYWORDS_ZH = [
        "股票", "股市", "A 股", "上证", "深证", "创业板",
        "成交量", "涨停", "跌停", "牛市", "熊市",
    ]

    def __init__(self, cache_dir: Path | None = None) -> None:
        self.cache_dir = cache_dir or CONFIG.cache_dir / "news"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._session = requests.Session()
        self._session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        })

        self._source_health: dict[str, float] = {}
        self._last_health_check: dict[str, float] = {}
        self._source_failures: dict[str, int] = {}
        self._strict_mode: bool = False

    def get_active_sources(self) -> list[str]:
        """Get list of active China sources."""
        return self.CHINA_SOURCES

    def collect_news(
        self,
        keywords: list[str] | None = None,
        categories: list[str] | None = None,
        limit: int = 100,
        hours_back: int = 24,
        strict: bool = False,
    ) -> list[NewsArticle]:
        """Collect news from China sources.

        Args:
            keywords: Optional keywords to filter by
            categories: Categories to collect
            limit: Maximum number of articles to return
            hours_back: How many hours back to collect
            strict: Fail-fast mode

        Returns:
            List of collected news articles
        """
        start_time = self._normalize_datetime(datetime.now() - timedelta(hours=hours_back))
        articles: list[NewsArticle] = []
        seen_ids: set[str] = set()
        previous_strict_mode = bool(self._strict_mode)
        self._strict_mode = bool(strict)
        strict_mode = bool(strict)

        active_sources = list(self.CHINA_SOURCES)
        log.debug("Collecting news from %s China sources", len(active_sources))

        for source in active_sources:
            if len(articles) >= limit:
                break
            if (not strict_mode) and self._is_source_temporarily_disabled(source):
                continue

            try:
                remaining = max(10, limit - len(articles))
                source_articles = self._fetch_from_source(
                    source=source,
                    keywords=keywords,
                    start_time=start_time,
                    limit=remaining,
                )

                for article in source_articles:
                    if article.id not in seen_ids:
                        seen_ids.add(article.id)
                        articles.append(article)

            except Exception as e:
                self._update_source_health(source, success=False)
                if strict_mode:
                    raise RuntimeError(
                        f"Strict news collection failed for source={source}: {e}"
                    ) from e
                log.warning(f"Failed to fetch from {source}: {e}")

        self._strict_mode = previous_strict_mode

        if not articles:
            cached = self._load_recent_cached_articles(
                start_time=start_time,
                limit=max(1, int(limit)),
                keywords=keywords,
            )
            if cached:
                for article in cached:
                    if article.id not in seen_ids:
                        seen_ids.add(article.id)
                        articles.append(article)
                log.info("Recovered %s articles from cache fallback.", len(articles))

        articles.sort(key=lambda x: x.published_at, reverse=True)

        for article in articles:
            self._categorize_article(article)
            self._calculate_relevance(article, keywords)

        if strict_mode and not articles:
            raise RuntimeError("Strict news collection returned no articles")
        if articles:
            source_count = len({str(getattr(a, "source", "") or "").strip() for a in articles})
            log.debug(
                "Collected %d unique articles from %d sources",
                len(articles),
                source_count,
            )
        else:
            log.debug("Collected 0 unique articles")
        return articles

    def _source_cooldown_seconds(self, source: str) -> float:
        """Dynamic source cooldown based on consecutive failures."""
        failures = int(self._source_failures.get(source, 0) or 0)
        if failures <= 0:
            return 0.0
        return float(
            min(
                self._MAX_SOURCE_COOLDOWN_SECONDS,
                30.0 * float(2 ** min(6, failures - 1)),
            )
        )

    def _is_source_temporarily_disabled(self, source: str) -> bool:
        """Skip repeatedly failing sources for a short cooldown window."""
        if bool(self._strict_mode):
            return False
        health = float(self._source_health.get(source, 0.5) or 0.5)
        if health >= self._MIN_HEALTH_FOR_IMMEDIATE_RETRY:
            return False
        last_check = float(self._last_health_check.get(source, 0.0) or 0.0)
        if last_check <= 0:
            return False
        cooldown = self._source_cooldown_seconds(source)
        if cooldown <= 0:
            return False
        age = float(time.time()) - last_check
        if age >= cooldown:
            return False
        log.debug(
            "Skipping source %s during cooldown (health=%.2f, retry_in=%.0fs)",
            source,
            health,
            max(0.0, cooldown - age),
        )
        return True

    @staticmethod
    def _normalize_datetime(value: datetime) -> datetime:
        """Normalize datetimes to naive UTC for safe comparisons."""
        if not isinstance(value, datetime):
            return datetime.now()
        if value.tzinfo is None:
            return value
        return value.astimezone(timezone.utc).replace(tzinfo=None)

    @classmethod
    def _is_recent_enough(cls, published_at: datetime, start_time: datetime) -> bool:
        """Timezone-safe comparison helper."""
        try:
            return cls._normalize_datetime(published_at) >= cls._normalize_datetime(start_time)
        except Exception:
            return True

    @staticmethod
    def _repair_mojibake(value: object) -> str:
        """Repair common UTF-8/Latin-1 mojibake from mixed news endpoints."""
        text = str(value or "")
        if not text:
            return ""
        suspicious = any(ch in text for ch in ("Ã", "Â", "æ", "ç", "è", "é", "å", "ä"))
        if not suspicious:
            return text
        try:
            repaired = text.encode("latin-1").decode("utf-8")
        except Exception:
            return text
        if repaired and repaired != text:
            return repaired
        return text

    @staticmethod
    def _clean_text(value: object) -> str:
        """Normalize text payload from mixed HTML/JSON content."""
        text = NewsCollector._repair_mojibake(value)
        if not text:
            return ""
        text = re.sub(r"<[^>]+>", " ", text)
        text = html.unescape(text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    @staticmethod
    def _parse_datetime(value: object) -> datetime | None:
        """Best-effort parser for timestamp-like values."""
        if value is None:
            return None
        if isinstance(value, datetime):
            return NewsCollector._normalize_datetime(value)
        raw = str(value).strip()
        if not raw:
            return None
        if raw.isdigit():
            try:
                iv = int(raw)
                if iv > 2_000_000_000_000:
                    return NewsCollector._normalize_datetime(datetime.fromtimestamp(iv / 1000.0))
                if iv > 0:
                    return NewsCollector._normalize_datetime(datetime.fromtimestamp(iv))
            except Exception:
                pass
        for fmt in (
            "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M",
            "%Y/%m/%d %H:%M:%S", "%Y/%m/%d %H:%M", "%Y-%m-%d",
        ):
            try:
                parsed = datetime.strptime(raw[:19], fmt)
                return NewsCollector._normalize_datetime(parsed)
            except ValueError:
                continue
        try:
            parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
            return NewsCollector._normalize_datetime(parsed)
        except Exception:
            return None

    @staticmethod
    def _xml_text(node: ET.Element, tag_name: str) -> str:
        """Extract first matching XML node text ignoring namespaces."""
        wanted = str(tag_name or "").strip().lower()
        if not wanted:
            return ""
        for child in node.iter():
            raw_tag = str(getattr(child, "tag", "") or "")
            if raw_tag.split("}")[-1].lower() != wanted:
                continue
            text = str(getattr(child, "text", "") or "").strip()
            if text:
                return text
        return ""

    @staticmethod
    def _xml_link(node: ET.Element) -> str:
        """Extract link text/href from RSS/Atom item entry."""
        for child in node.iter():
            raw_tag = str(getattr(child, "tag", "") or "")
            if raw_tag.split("}")[-1].lower() != "link":
                continue
            href = str(getattr(child, "attrib", {}).get("href", "") or "").strip()
            if href:
                return href
            text = str(getattr(child, "text", "") or "").strip()
            if text:
                return text
        return ""

    @staticmethod
    def _decode_json_payload(payload: object) -> object | None:
        """Parse direct JSON and JSONP wrappers into Python objects."""
        text = str(payload or "").strip().lstrip("\ufeff")
        if not text:
            return None

        candidates: list[str] = [text]
        import re as regex_module
        m_jsonp = regex_module.match(
            r"^\s*[\w\.$]+\s*\(\s*(?P<body>[\s\S]+)\s*\)\s*;?\s*$",
            text,
        )
        if m_jsonp:
            body = str(m_jsonp.group("body") or "").strip()
            if body:
                candidates.append(body)

        for left, right in (("{", "}"), ("[", "]")):
            li = text.find(left)
            ri = text.rfind(right)
            if li >= 0 and ri > li:
                candidates.append(text[li : ri + 1])

        seen: set[str] = set()
        for candidate in candidates:
            chunk = str(candidate or "").strip()
            if not chunk or chunk in seen:
                continue
            seen.add(chunk)
            try:
                return json.loads(chunk)
            except Exception:
                continue
        return None

    def _fetch_from_source(
        self,
        source: str,
        keywords: list[str] | None,
        start_time: datetime,
        limit: int,
    ) -> list[NewsArticle]:
        """Fetch articles from a specific source."""
        if source == "eastmoney":
            return self._fetch_eastmoney(keywords, start_time, limit)
        elif source == "sina_finance":
            return self._fetch_sina_finance(keywords, start_time, limit)
        elif source == "caixin":
            return self._fetch_caixin(keywords, start_time, limit)
        else:
            return []

    def _fetch_eastmoney(
        self,
        keywords: list[str] | None,
        start_time: datetime,
        limit: int,
    ) -> list[NewsArticle]:
        """Fetch from EastMoney."""
        articles: list[NewsArticle] = []
        url = "https://search-api-web.eastmoney.com/search/jsonp"
        primary_error: Exception | None = None

        try:
            params = {
                "cb": "jQuery",
                "keyword": " ".join(keywords or []),
                "pageNumber": 1,
                "pageSize": max(1, int(limit)),
            }
            resp = self._session.get(url, params=params, timeout=10)
            resp.raise_for_status()
            data = self._decode_json_payload(resp.text)

            if isinstance(data, dict):
                rows = data.get("result", {}).get("data")
                if not isinstance(rows, list):
                    rows = data.get("data", [])
                items = list(rows) if isinstance(rows, list) else []
            else:
                items = []

            for item in items[:limit]:
                title = self._clean_text(
                    item.get("title", "")
                    or item.get("Title", "")
                    or item.get("newsTitle", "")
                )
                content = self._clean_text(
                    item.get("content", "")
                    or item.get("Content", "")
                    or item.get("digest", "")
                )
                if not title:
                    continue

                pub_time = (
                    item.get("showTime", "")
                    or item.get("ShowTime", "")
                    or item.get("publishTime", "")
                )
                published_at = self._parse_datetime(pub_time) or datetime.now()

                if not self._is_recent_enough(published_at, start_time):
                    continue

                article_id = self._generate_id(f"eastmoney_{title}_{pub_time}")
                article_url = (
                    item.get("url", "")
                    or item.get("Url", "")
                    or item.get("articleUrl", "")
                )

                article = NewsArticle(
                    id=article_id,
                    title=title,
                    content=content or title,
                    summary=title[:200],
                    source="eastmoney",
                    url=str(article_url or ""),
                    published_at=published_at,
                    collected_at=datetime.now(),
                    language="zh",
                    category="company",
                )
                articles.append(article)
        except Exception as e:
            primary_error = e

        if not articles:
            articles = self._fetch_eastmoney_html_fallback(keywords, start_time, limit)

        if articles:
            self._update_source_health("eastmoney", success=True)
            return articles[:limit]

        if primary_error is not None:
            log.warning(f"EastMoney fetch failed: {primary_error}")
        self._update_source_health("eastmoney", success=False)
        return []

    def _fetch_sina_finance(
        self,
        keywords: list[str] | None,
        start_time: datetime,
        limit: int,
    ) -> list[NewsArticle]:
        """Fetch from Sina Finance with endpoint rotation and HTML fallback."""
        _ = keywords
        endpoint_specs = [
            (
                "https://feed.mix.sina.com.cn/api/roll/feed",
                {"page": 1, "page_size": max(20, int(limit)), "cid": "1"},
            ),
            (
                "https://feed.mix.sina.com.cn/api/roll/get",
                {"pageid": "153", "lid": "2510", "num": max(20, int(limit)), "page": 1},
            ),
            (
                "https://feed.mix.sina.com.cn/api/roll/get",
                {"pageid": "153", "lid": "2509", "num": max(20, int(limit)), "page": 1},
            ),
        ]

        endpoint_errors: list[str] = []
        for url, params in endpoint_specs:
            try:
                resp = self._session.get(url, params=params, timeout=10)
                if int(getattr(resp, "status_code", 0) or 0) == 404:
                    endpoint_errors.append(f"{url} HTTP 404")
                    continue
                resp.raise_for_status()
                data = self._decode_json_payload(getattr(resp, "text", ""))
                items = self._extract_sina_feed_items(data)
                if not items:
                    endpoint_errors.append(f"{url} empty_feed")
                    continue
                articles = self._parse_sina_articles(
                    items=items,
                    start_time=start_time,
                    limit=limit,
                )
                if articles:
                    self._update_source_health("sina_finance", success=True)
                    return articles[:limit]
                endpoint_errors.append(f"{url} no_recent_rows")
            except Exception as exc:
                endpoint_errors.append(f"{url} {exc}")

        html_fallback = self._fetch_sina_finance_html_fallback(
            keywords=keywords,
            start_time=start_time,
            limit=limit,
        )
        if html_fallback:
            self._update_source_health("sina_finance", success=True)
            return html_fallback[:limit]

        if endpoint_errors:
            log.warning(
                "Sina Finance fetch failed: %s",
                "; ".join(endpoint_errors[:2]),
            )
        self._update_source_health("sina_finance", success=False)
        return []

    def _extract_sina_feed_items(self, payload: object) -> list[dict[str, Any]]:
        """Extract Sina news rows from variant JSON response shapes."""
        if isinstance(payload, list):
            return [row for row in payload if isinstance(row, dict)]
        if not isinstance(payload, dict):
            return []

        def _coerce_rows(value: object) -> list[dict[str, Any]]:
            if isinstance(value, list):
                return [row for row in value if isinstance(row, dict)]
            return []

        containers: list[dict[str, Any]] = [payload]
        for key in ("result", "data"):
            value = payload.get(key)
            if isinstance(value, dict):
                containers.append(value)

        for container in containers:
            for key in ("data", "list", "items", "feed", "roll_data", "result"):
                rows = _coerce_rows(container.get(key))
                if rows:
                    return rows

        return []

    def _parse_sina_articles(
        self,
        *,
        items: list[dict[str, Any]],
        start_time: datetime,
        limit: int,
    ) -> list[NewsArticle]:
        """Normalize Sina feed rows into NewsArticle objects."""
        out: list[NewsArticle] = []
        seen_ids: set[str] = set()
        generic_titles = {
            "home",
            "homepage",
            "login",
            "logout",
            "register",
            "more",
            "more...",
            "首页",
            "新浪首页",
            "登录",
            "注册",
            "更多",
        }
        generic_urls = {
            "http://www.sina.com.cn/",
            "https://www.sina.com.cn/",
            "http://sina.com.cn/",
            "https://sina.com.cn/",
        }

        for item in list(items or []):
            title = self._clean_text(
                item.get("title", "")
                or item.get("wap_title", "")
                or item.get("title1", "")
            )
            content = self._clean_text(
                item.get("content", "")
                or item.get("intro", "")
                or item.get("summary", "")
                or item.get("k", "")
            )
            if not title:
                continue
            if len(title) < 4:
                continue
            if title.strip().lower() in generic_titles:
                continue

            published_at = None
            for pub_key in (
                "ctime",
                "intime",
                "pubtime",
                "pubDate",
                "time",
                "datetime",
                "created_at",
                "create_time",
            ):
                published_at = self._parse_datetime(item.get(pub_key))
                if published_at is not None:
                    break
            if published_at is None:
                published_at = datetime.now()

            if not self._is_recent_enough(published_at, start_time):
                continue

            raw_url = str(
                item.get("url", "")
                or item.get("wapurl", "")
                or item.get("docurl", "")
                or item.get("link", "")
                or ""
            ).strip()
            if raw_url.startswith("//"):
                raw_url = f"https:{raw_url}"
            if not raw_url:
                oid = str(item.get("oid", "") or "").strip().lstrip("/")
                if oid:
                    raw_url = f"https://finance.sina.com.cn/{oid}"
            if raw_url in generic_urls:
                continue
            if raw_url:
                raw_url_lower = raw_url.lower()
                if "finance.sina.com.cn" not in raw_url_lower:
                    continue
                if (
                    "doc-" not in raw_url_lower
                    and "/roll/" not in raw_url_lower
                    and "/20" not in raw_url_lower
                ):
                    continue
                if "index.d.html" in raw_url_lower:
                    continue

            article_id = self._generate_id(
                f"sina_{title}_{published_at.isoformat()}_{raw_url}"
            )
            if article_id in seen_ids:
                continue
            seen_ids.add(article_id)
            out.append(
                NewsArticle(
                    id=article_id,
                    title=title,
                    content=content or title,
                    summary=title[:200],
                    source="sina_finance",
                    url=raw_url,
                    published_at=published_at,
                    collected_at=datetime.now(),
                    language="zh",
                    category="market",
                )
            )
            if len(out) >= max(1, int(limit)):
                break
        return out

    def _fetch_sina_finance_html_fallback(
        self,
        keywords: list[str] | None,
        start_time: datetime,
        limit: int,
    ) -> list[NewsArticle]:
        """Fallback HTML extraction for Sina Finance when feed endpoints drift."""
        _ = keywords
        for url in (
            "https://finance.sina.com.cn/china/",
            "https://finance.sina.com.cn/stock/",
            "https://finance.sina.com.cn/",
        ):
            try:
                resp = self._session.get(url, timeout=10)
                resp.raise_for_status()
                rows = self._extract_html_anchor_articles(
                    html_text=getattr(resp, "text", ""),
                    source="sina_finance",
                    base_url=url,
                    start_time=start_time,
                    limit=limit,
                    keywords=keywords,
                    category="market",
                )
                if rows:
                    return rows[:limit]
            except Exception:
                continue
        return []

    def _fetch_caixin(
        self,
        keywords: list[str] | None,
        start_time: datetime,
        limit: int,
    ) -> list[NewsArticle]:
        """Fetch from Caixin."""
        articles: list[NewsArticle] = []
        url = "https://api.caixin.com/api/content/list"
        primary_error: Exception | None = None

        try:
            params = {
                "limit": limit,
                "page": 1,
            }
            resp = self._session.get(url, params=params, timeout=10)
            resp.raise_for_status()
            data = self._decode_json_payload(resp.text)

            if not isinstance(data, dict):
                data = {}
            items = data.get("data", [])
            if not isinstance(items, list):
                items = []

            for item in items[:limit]:
                title = self._clean_text(item.get("title", ""))
                content = self._clean_text(item.get("content", ""))
                if not title:
                    continue

                pub_time = item.get("pubtime", "")
                published_at = self._parse_datetime(pub_time) or datetime.now()
                
                if not self._is_recent_enough(published_at, start_time):
                    continue

                article_id = self._generate_id(f"caixin_{title}_{pub_time}")
                
                article = NewsArticle(
                    id=article_id,
                    title=title,
                    content=content or title,
                    summary=title[:200],
                    source="caixin",
                    url=item.get("url", ""),
                    published_at=published_at,
                    collected_at=datetime.now(),
                    language="zh",
                    category="policy",
                )
                articles.append(article)
        except Exception as e:
            primary_error = e

        if not articles:
            articles = self._fetch_caixin_html_fallback(keywords, start_time, limit)

        if articles:
            self._update_source_health("caixin", success=True)
            return articles[:limit]

        if primary_error is not None:
            log.warning(f"Caixin fetch failed: {primary_error}")
        self._update_source_health("caixin", success=False)
        return []

    def _fetch_eastmoney_html_fallback(
        self,
        keywords: list[str] | None,
        start_time: datetime,
        limit: int,
    ) -> list[NewsArticle]:
        """Fallback HTML title extraction for EastMoney."""
        try:
            url = "https://finance.eastmoney.com/"
            resp = self._session.get(url, timeout=10)
            resp.raise_for_status()
            return self._extract_html_anchor_articles(
                html_text=resp.text,
                source="eastmoney",
                base_url=url,
                start_time=start_time,
                limit=limit,
                keywords=keywords,
                category="market",
            )
        except Exception:
            return []

    def _fetch_caixin_html_fallback(
        self,
        keywords: list[str] | None,
        start_time: datetime,
        limit: int,
    ) -> list[NewsArticle]:
        """Fallback HTML title extraction for Caixin."""
        try:
            url = "https://www.caixinglobal.com/"
            resp = self._session.get(url, timeout=10)
            resp.raise_for_status()
            return self._extract_html_anchor_articles(
                html_text=resp.text,
                source="caixin",
                base_url=url,
                start_time=start_time,
                limit=limit,
                keywords=keywords,
                category="policy",
            )
        except Exception:
            return []

    def _extract_html_anchor_articles(
        self,
        *,
        html_text: str,
        source: str,
        base_url: str,
        start_time: datetime,
        limit: int,
        keywords: list[str] | None,
        category: str,
    ) -> list[NewsArticle]:
        """Extract lightweight article objects from anchor tags."""
        rows: list[NewsArticle] = []
        pattern = re.compile(
            r"<a[^>]+href=[\"'](?P<href>[^\"']+)[\"'][^>]*>(?P<title>.*?)</a>",
            re.IGNORECASE | re.DOTALL,
        )
        _ = keywords
        generic_titles = {
            "home",
            "homepage",
            "login",
            "logout",
            "register",
            "more",
            "more...",
            "nav",
            "首页",
            "新浪首页",
            "登录",
            "注册",
            "更多",
            "返回顶部",
        }
        for match in pattern.finditer(str(html_text or "")):
            if len(rows) >= max(1, int(limit)):
                break
            href = str(match.group("href") or "").strip()
            title = self._clean_text(match.group("title") or "")
            if not href or not title:
                continue
            href_lower = href.lower()
            if (
                href_lower.startswith("#")
                or href_lower.startswith("javascript:")
                or href_lower.startswith("mailto:")
            ):
                continue
            if title.strip().lower() in generic_titles:
                continue
            if len(title) < 4:
                continue

            published_at = datetime.now()
            if not self._is_recent_enough(published_at, start_time):
                continue

            article_url = urljoin(base_url, href)
            article_url_lower = article_url.lower()
            looks_like_article = (
                ".shtml" in article_url_lower
                or ".html" in article_url_lower
                or "/20" in article_url_lower
                or "doc-" in article_url_lower
                or "/article" in article_url_lower
            )
            if source == "sina_finance":
                if "finance.sina.com.cn" not in article_url_lower:
                    continue
                if (
                    "doc-" not in article_url_lower
                    and "/roll/" not in article_url_lower
                    and "/20" not in article_url_lower
                ):
                    continue
                if "index.d.html" in article_url_lower:
                    continue
            if not looks_like_article and len(title) < 12:
                continue

            article_id = self._generate_id(f"{source}_{title}_{article_url}")
            rows.append(
                NewsArticle(
                    id=article_id,
                    title=title,
                    content=title,
                    summary=title[:200],
                    source=source,
                    url=article_url,
                    published_at=published_at,
                    collected_at=datetime.now(),
                    language="zh",
                    category=category,
                )
            )
        return rows

    def _generate_id(self, text: str) -> str:
        """Generate unique article ID."""
        return hashlib.md5(text.encode("utf-8")).hexdigest()

    def _update_source_health(self, source: str, success: bool) -> None:
        """Update source health score."""
        now = time.time()
        self._last_health_check[source] = now
        
        if success:
            self._source_failures[source] = 0
            current = self._source_health.get(source, 0.5)
            self._source_health[source] = min(1.0, current + 0.1)
        else:
            self._source_failures[source] = self._source_failures.get(source, 0) + 1
            current = self._source_health.get(source, 0.5)
            self._source_health[source] = max(0.0, current - 0.2)

    def _categorize_article(self, article: NewsArticle) -> None:
        """Categorize article based on content."""
        text = (article.title + " " + article.content).lower()
        
        policy_score = sum(1 for kw in self.POLICY_KEYWORDS_ZH if kw in text)
        market_score = sum(1 for kw in self.MARKET_KEYWORDS_ZH if kw in text)
        
        if policy_score > market_score:
            article.category = "policy"
        elif market_score > 0:
            article.category = "market"
        else:
            article.category = "company"

    def _calculate_relevance(self, article: NewsArticle, keywords: list[str] | None) -> None:
        """Calculate relevance score."""
        if not keywords:
            article.relevance_score = 0.5
            return
        
        text = (article.title + " " + article.content).lower()
        matches = sum(1 for kw in keywords if kw.lower() in text)
        article.relevance_score = min(1.0, matches / max(1, len(keywords)))

    def _load_recent_cached_articles(
        self,
        start_time: datetime,
        limit: int,
        keywords: list[str] | None = None,
    ) -> list[NewsArticle]:
        """Load recent articles from cache."""
        cache_file = self.cache_dir / "recent_news.json"
        if not cache_file.exists():
            return []

        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            articles = []
            for item in data.get("articles", [])[:limit]:
                try:
                    article = NewsArticle.from_dict(item)
                    if self._is_recent_enough(article.published_at, start_time):
                        articles.append(article)
                except Exception:
                    continue
            
            return articles
        except Exception:
            return []

    def save_articles(self, articles: list[NewsArticle], filename: str | None = None) -> Path:
        """Save articles to cache file."""
        if filename is None:
            filename = f"news_{int(time.time())}.json"

        output_file = self.cache_dir / filename
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump({
                "collected_at": datetime.now().isoformat(),
                "count": len(articles),
                "articles": [a.to_dict() for a in articles],
            }, f, ensure_ascii=False, indent=2)

        return output_file


# Module-level singleton
_collector: NewsCollector | None = None


def get_collector() -> NewsCollector:
    """Get the news collector singleton."""
    global _collector
    if _collector is None:
        _collector = NewsCollector()
    return _collector


def reset_collector() -> None:
    """Reset the news collector singleton (for testing)."""
    global _collector
    _collector = None
