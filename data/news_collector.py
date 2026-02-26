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
        log.info("Collecting news from %s China sources", len(active_sources))

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
        log.info(f"Collected {len(articles)} unique articles")
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
    def _clean_text(value: object) -> str:
        """Normalize text payload from mixed HTML/JSON content."""
        text = str(value or "")
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
        articles = []
        url = "https://np-anotice-stock.eastmoney.com/api/security/notice"
        
        try:
            params = {
                "pageIndex": 0,
                "pageSize": limit,
            }
            resp = self._session.get(url, params=params, timeout=10)
            resp.raise_for_status()
            data = self._decode_json_payload(resp.text)
            
            if not data or not isinstance(data, dict):
                return []

            items = data.get("data", [])
            if not isinstance(items, list):
                return []

            for item in items[:limit]:
                title = self._clean_text(item.get("Title", ""))
                content = self._clean_text(item.get("Content", ""))
                if not title:
                    continue

                pub_time = item.get("ShowTime", "")
                published_at = self._parse_datetime(pub_time) or datetime.now()
                
                if not self._is_recent_enough(published_at, start_time):
                    continue

                article_id = self._generate_id(f"eastmoney_{title}_{pub_time}")
                
                article = NewsArticle(
                    id=article_id,
                    title=title,
                    content=content or title,
                    summary=title[:200],
                    source="eastmoney",
                    url=item.get("Url", ""),
                    published_at=published_at,
                    collected_at=datetime.now(),
                    language="zh",
                    category="company",
                )
                articles.append(article)

            self._update_source_health("eastmoney", success=True)
            return articles[:limit]

        except requests.RequestException as e:
            log.warning(f"EastMoney fetch failed: {e}")
            self._update_source_health("eastmoney", success=False)
            return []

    def _fetch_sina_finance(
        self,
        keywords: list[str] | None,
        start_time: datetime,
        limit: int,
    ) -> list[NewsArticle]:
        """Fetch from Sina Finance."""
        articles = []
        url = "https://feed.mix.sina.com.cn/api/roll/feed"
        
        try:
            params = {
                "page": 1,
                "page_size": limit,
                "cid": "1",
            }
            resp = self._session.get(url, params=params, timeout=10)
            resp.raise_for_status()
            data = self._decode_json_payload(resp.text)
            
            if not data or not isinstance(data, dict):
                return []

            result = data.get("result", {})
            items = result.get("data", []) if isinstance(result, dict) else []
            
            for item in items[:limit]:
                title = self._clean_text(item.get("title", ""))
                content = self._clean_text(item.get("content", ""))
                if not title:
                    continue

                pub_time = item.get("ctime", "")
                published_at = self._parse_datetime(pub_time) or datetime.now()
                
                if not self._is_recent_enough(published_at, start_time):
                    continue

                article_id = self._generate_id(f"sina_{title}_{pub_time}")
                
                article = NewsArticle(
                    id=article_id,
                    title=title,
                    content=content or title,
                    summary=title[:200],
                    source="sina_finance",
                    url=item.get("url", ""),
                    published_at=published_at,
                    collected_at=datetime.now(),
                    language="zh",
                    category="market",
                )
                articles.append(article)

            self._update_source_health("sina_finance", success=True)
            return articles[:limit]

        except requests.RequestException as e:
            log.warning(f"Sina Finance fetch failed: {e}")
            self._update_source_health("sina_finance", success=False)
            return []

    def _fetch_caixin(
        self,
        keywords: list[str] | None,
        start_time: datetime,
        limit: int,
    ) -> list[NewsArticle]:
        """Fetch from Caixin."""
        articles = []
        url = "https://api.caixin.com/api/content/list"
        
        try:
            params = {
                "limit": limit,
                "page": 1,
            }
            resp = self._session.get(url, params=params, timeout=10)
            resp.raise_for_status()
            data = self._decode_json_payload(resp.text)
            
            if not data or not isinstance(data, dict):
                return []

            items = data.get("data", [])
            if not isinstance(items, list):
                return []

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

            self._update_source_health("caixin", success=True)
            return articles[:limit]

        except requests.RequestException as e:
            log.warning(f"Caixin fetch failed: {e}")
            self._update_source_health("caixin", success=False)
            return []

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
