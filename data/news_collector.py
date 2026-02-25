# data/news_collector.py
"""News and Policy Data Collector with VPN-aware routing.

This module automatically collects news, policy, and regulatory data from:
- Chinese sources (when VPN is off): Jin10, EastMoney, Sina Finance, Xueqiu
- International sources (when VPN is on): Reuters, Bloomberg, Yahoo Finance

Features:
- Auto-detect VPN status
- Multi-source aggregation
- Content parsing and normalization
- Deduplication and relevance scoring
"""

import hashlib
import json
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional
from urllib.parse import quote_plus

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
    language: str  # 'zh' or 'en'
    category: str  # 'policy', 'market', 'company', 'economic', 'regulatory'
    sentiment_score: float = 0.0  # -1.0 to 1.0
    relevance_score: float = 0.0  # 0.0 to 1.0
    entities: list[str] = field(default_factory=list)  # Companies, people, policies
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
        return cls(
            id=data["id"],
            title=data["title"],
            content=data["content"],
            summary=data["summary"],
            source=data["source"],
            url=data["url"],
            published_at=datetime.fromisoformat(data["published_at"]),
            collected_at=datetime.fromisoformat(data["collected_at"]),
            language=data["language"],
            category=data["category"],
            sentiment_score=float(data.get("sentiment_score", 0.0)),
            relevance_score=float(data.get("relevance_score", 0.0)),
            entities=data.get("entities", []),
            tags=data.get("tags", []),
        )


class NewsCollector:
    """Multi-source news collector with VPN-aware routing."""

    # Chinese sources (VPN off)
    CHINESE_SOURCES = [
        "jin10",
        "eastmoney",
        "sina_finance",
        "xueqiu",
        "caixin",
        "csrc",  # China Securities Regulatory Commission
    ]

    # International sources (VPN on)
    INTERNATIONAL_SOURCES = [
        "reuters",
        "bloomberg",
        "yahoo_finance",
        "marketwatch",
        "cnbc",
    ]

    # Search keywords for policy/regulatory news
    POLICY_KEYWORDS_ZH = [
        "政策", "规定", "监管", "证监会", "央行", "财政部",
        "货币政策", "财政政策", "产业政策", "法规", "条例",
        "新股", "IPO", "退市", "交易规则", "印花税",
    ]

    POLICY_KEYWORDS_EN = [
        "policy", "regulation", "regulatory", "SEC", "Federal Reserve",
        "Treasury", "monetary policy", "fiscal policy", "industrial policy",
        "IPO", "delisting", "trading rules", "stamp duty",
    ]

    MARKET_KEYWORDS_ZH = [
        "股票", "股市", "A 股", "上证", "深证", "创业板",
        "成交量", "涨停", "跌停", "牛市", "熊市",
    ]

    MARKET_KEYWORDS_EN = [
        "stock", "market", "A-share", "SSE", "SZSE", "GEM",
        "volume", "limit up", "limit down", "bull market", "bear market",
    ]

    def __init__(self, cache_dir: Optional[Path] = None) -> None:
        self.cache_dir = cache_dir or CONFIG.cache_dir / "news"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._session = requests.Session()
        self._session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        })

        # Proxy configuration
        self._setup_proxy()

        # Source availability cache
        self._source_health: dict[str, float] = {}  # source -> health score (0-1)
        self._last_health_check: dict[str, float] = {}  # source -> last check timestamp

    def _setup_proxy(self) -> None:
        """Configure proxy based on VPN status."""
        from config.runtime_env import env_text

        proxy_url = env_text("TRADING_PROXY_URL", "")
        vpn_enabled = env_text("TRADING_VPN", "0") == "1"

        if vpn_enabled and proxy_url:
            self._session.proxies = {
                "http": proxy_url,
                "https": proxy_url,
            }
            log.info(f"News collector configured with proxy: {proxy_url}")
        elif vpn_enabled:
            log.warning("VPN enabled but no proxy configured")
        else:
            log.info("News collector running in direct mode (China sources)")

    def is_vpn_mode(self) -> bool:
        """Detect if VPN mode is active."""
        from config.runtime_env import env_text, env_flag

        # Explicit configuration
        if env_flag("TRADING_VPN", False):
            return True
        if env_flag("TRADING_CHINA_DIRECT", False):
            return False

        # Auto-detect by testing access to Chinese vs international sites
        return self._detect_network_environment()

    def _detect_network_environment(self) -> bool:
        """Auto-detect network environment by testing connectivity."""
        # Test access to Chinese site
        chinese_test = "https://www.baidu.com"
        international_test = "https://www.google.com"

        try:
            # Try Chinese site first
            resp = self._session.get(chinese_test, timeout=5)
            if resp.status_code == 200:
                # Can access Chinese sites, now test international
                try:
                    resp_int = self._session.get(international_test, timeout=5)
                    if resp_int.status_code == 200:
                        # Can access both - likely VPN or good international connection
                        return True
                except Exception:
                    # Can't access international - China direct mode
                    return False
        except Exception:
            pass

        # Default to VPN mode (international sources)
        return True

    def get_active_sources(self) -> list[str]:
        """Get list of active sources based on VPN mode."""
        if self.is_vpn_mode():
            return self.INTERNATIONAL_SOURCES
        else:
            return self.CHINESE_SOURCES

    def collect_news(
        self,
        keywords: Optional[list[str]] = None,
        categories: Optional[list[str]] = None,
        limit: int = 100,
        hours_back: int = 24,
    ) -> list[NewsArticle]:
        """Collect news from active sources.

        Args:
            keywords: Optional keywords to filter by
            categories: Categories to collect ('policy', 'market', 'company', etc.)
            limit: Maximum number of articles to return
            hours_back: How many hours back to collect

        Returns:
            List of collected news articles
        """
        start_time = datetime.now() - timedelta(hours=hours_back)
        articles: list[NewsArticle] = []
        seen_ids: set[str] = set()

        active_sources = self.get_active_sources()
        log.info(f"Collecting news from {len(active_sources)} sources (VPN mode: {self.is_vpn_mode()})")

        for source in active_sources:
            if len(articles) >= limit:
                break

            try:
                source_articles = self._fetch_from_source(
                    source=source,
                    keywords=keywords,
                    start_time=start_time,
                    limit=max(10, limit // len(active_sources)),
                )

                for article in source_articles:
                    if article.id not in seen_ids:
                        seen_ids.add(article.id)
                        articles.append(article)

            except Exception as e:
                log.warning(f"Failed to fetch from {source}: {e}")
                self._update_source_health(source, success=False)

        # Sort by published date (newest first)
        articles.sort(key=lambda x: x.published_at, reverse=True)

        # Categorize and score
        for article in articles:
            self._categorize_article(article)
            self._calculate_relevance(article, keywords)

        log.info(f"Collected {len(articles)} unique articles")
        return articles

    def _fetch_from_source(
        self,
        source: str,
        keywords: Optional[list[str]],
        start_time: datetime,
        limit: int,
    ) -> list[NewsArticle]:
        """Fetch articles from a specific source."""
        if source == "jin10":
            return self._fetch_jin10(keywords, start_time, limit)
        elif source == "eastmoney":
            return self._fetch_eastmoney(keywords, start_time, limit)
        elif source == "sina_finance":
            return self._fetch_sina_finance(keywords, start_time, limit)
        elif source == "xueqiu":
            return self._fetch_xueqiu(keywords, start_time, limit)
        elif source == "caixin":
            return self._fetch_caixin(keywords, start_time, limit)
        elif source == "csrc":
            return self._fetch_csrc(keywords, start_time, limit)
        elif source == "reuters":
            return self._fetch_reuters(keywords, start_time, limit)
        elif source == "bloomberg":
            return self._fetch_bloomberg(keywords, start_time, limit)
        elif source == "yahoo_finance":
            return self._fetch_yahoo_finance(keywords, start_time, limit)
        else:
            return []

    def _fetch_jin10(
        self,
        keywords: Optional[list[str]],
        start_time: datetime,
        limit: int,
    ) -> list[NewsArticle]:
        """Fetch from Jin10 (财经快讯)."""
        articles = []
        url = "https://api.jin10.com/v1/flash"

        try:
            resp = self._session.get(url, timeout=10)
            resp.raise_for_status()
            data = resp.json()

            for item in data.get("data", [])[:limit]:
                content = item.get("data", {}).get("content", "")
                if not content:
                    continue

                title = content[:100]
                pub_time = item.get("time", "")

                try:
                    published_at = datetime.fromisoformat(pub_time.replace("Z", "+00:00"))
                except Exception:
                    published_at = datetime.now()

                if published_at < start_time:
                    continue

                article_id = self._generate_id(f"jin10_{item.get('id', title)}")
                articles.append(NewsArticle(
                    id=article_id,
                    title=title,
                    content=content,
                    summary=content[:200],
                    source="jin10",
                    url=item.get("url", f"https://flash.jin10.com/detail/{item.get('id', '')}"),
                    published_at=published_at,
                    collected_at=datetime.now(),
                    language="zh",
                    category="market",
                    tags=["jin10", "flash"],
                ))

            self._update_source_health("jin10", success=True)

        except Exception as e:
            log.error(f"Jin10 fetch failed: {e}")
            self._update_source_health("jin10", success=False)

        return articles

    def _fetch_eastmoney(
        self,
        keywords: Optional[list[str]],
        start_time: datetime,
        limit: int,
    ) -> list[NewsArticle]:
        """Fetch from EastMoney (东方财富网)."""
        articles = []

        # EastMoney API endpoint
        search_term = keywords[0] if keywords else "股票"
        url = f"https://search-api-web.eastmoney.com/search/json"
        params = {
            "keyword": quote_plus(search_term),
            "type": "cmsArticle",
            "page": 1,
            "pagesize": limit,
        }

        try:
            resp = self._session.get(url, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()

            for item in data.get("Result", [])[:limit]:
                title = item.get("Title", "")
                content = item.get("Content", "")
                url = item.get("Url", "")
                pub_time = item.get("ShowTime", "")

                if not title:
                    continue

                try:
                    published_at = datetime.fromisoformat(pub_time.replace("Z", "+00:00"))
                except Exception:
                    published_at = datetime.now()

                if published_at < start_time:
                    continue

                article_id = self._generate_id(f"eastmoney_{title}")
                articles.append(NewsArticle(
                    id=article_id,
                    title=title,
                    content=content or title,
                    summary=title[:200],
                    source="eastmoney",
                    url=url,
                    published_at=published_at,
                    collected_at=datetime.now(),
                    language="zh",
                    category="market",
                    tags=["eastmoney"],
                ))

            self._update_source_health("eastmoney", success=True)

        except Exception as e:
            log.error(f"EastMoney fetch failed: {e}")
            self._update_source_health("eastmoney", success=False)

        return articles

    def _fetch_sina_finance(
        self,
        keywords: Optional[list[str]],
        start_time: datetime,
        limit: int,
    ) -> list[NewsArticle]:
        """Fetch from Sina Finance."""
        articles = []
        url = "https://feed.mix.sina.com.cn/api/roll/get"
        params = {
            "lid": "2509",  # Finance channel
            "num": limit,
            "page": 1,
        }

        try:
            resp = self._session.get(url, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()

            for item in data.get("result", {}).get("data", [])[:limit]:
                title = item.get("title", "")
                intro = item.get("intro", "")
                url = item.get("url", "")
                pub_time = item.get("ctime", "")

                if not title:
                    continue

                try:
                    published_at = datetime.fromtimestamp(int(pub_time))
                except Exception:
                    published_at = datetime.now()

                if published_at < start_time:
                    continue

                article_id = self._generate_id(f"sina_{title}")
                articles.append(NewsArticle(
                    id=article_id,
                    title=title,
                    content=intro or title,
                    summary=title[:200],
                    source="sina_finance",
                    url=url,
                    published_at=published_at,
                    collected_at=datetime.now(),
                    language="zh",
                    category="market",
                    tags=["sina"],
                ))

            self._update_source_health("sina_finance", success=True)

        except Exception as e:
            log.error(f"Sina Finance fetch failed: {e}")
            self._update_source_health("sina_finance", success=False)

        return articles

    def _fetch_xueqiu(
        self,
        keywords: Optional[list[str]],
        start_time: datetime,
        limit: int,
    ) -> list[NewsArticle]:
        """Fetch from Xueqiu (雪球)."""
        # Xueqiu requires authentication for API access
        # This is a simplified implementation
        articles = []
        log.debug("Xueqiu fetch skipped (requires authentication)")
        return articles

    def _fetch_caixin(
        self,
        keywords: Optional[list[str]],
        start_time: datetime,
        limit: int,
    ) -> list[NewsArticle]:
        """Fetch from Caixin."""
        articles = []
        url = "https://api.caixin.com/api/content/list"
        params = {
            "columnid": "20",  # Finance
            "num": limit,
        }

        try:
            resp = self._session.get(url, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()

            for item in data.get("data", {}).get("list", [])[:limit]:
                title = item.get("title", "")
                content = item.get("content", "")
                url = item.get("url", "")
                pub_time = item.get("time", "")

                if not title:
                    continue

                try:
                    published_at = datetime.fromisoformat(pub_time)
                except Exception:
                    published_at = datetime.now()

                if published_at < start_time:
                    continue

                article_id = self._generate_id(f"caixin_{title}")
                articles.append(NewsArticle(
                    id=article_id,
                    title=title,
                    content=content or title,
                    summary=title[:200],
                    source="caixin",
                    url=url,
                    published_at=published_at,
                    collected_at=datetime.now(),
                    language="zh",
                    category="market",
                    tags=["caixin"],
                ))

            self._update_source_health("caixin", success=True)

        except Exception as e:
            log.error(f"Caixin fetch failed: {e}")
            self._update_source_health("caixin", success=False)

        return articles

    def _fetch_csrc(
        self,
        keywords: Optional[list[str]],
        start_time: datetime,
        limit: int,
    ) -> list[NewsArticle]:
        """Fetch from CSRC (中国证监会)."""
        articles = []
        url = "http://www.csrc.gov.cn/csrc/web/xxgklist.shtml"

        # CSRC website scraping would require more complex handling
        # This is a placeholder
        log.debug("CSRC fetch requires specialized scraping")
        return articles

    def _fetch_reuters(
        self,
        keywords: Optional[list[str]],
        start_time: datetime,
        limit: int,
    ) -> list[NewsArticle]:
        """Fetch from Reuters."""
        articles = []
        search_term = keywords[0] if keywords else "China stock"
        url = f"https://www.reuters.com/api/search"
        params = {
            "query": search_term,
            "sort": "newest",
            "size": limit,
        }

        try:
            resp = self._session.get(url, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()

            for item in data.get("results", [])[:limit]:
                title = item.get("title", "")
                content = item.get("description", "")
                url = item.get("url", "")
                pub_time = item.get("date", "")

                if not title:
                    continue

                try:
                    published_at = datetime.fromisoformat(pub_time.replace("Z", "+00:00"))
                except Exception:
                    published_at = datetime.now()

                if published_at < start_time:
                    continue

                article_id = self._generate_id(f"reuters_{title}")
                articles.append(NewsArticle(
                    id=article_id,
                    title=title,
                    content=content or title,
                    summary=title[:200],
                    source="reuters",
                    url=f"https://www.reuters.com{url}" if url and not url.startswith("http") else url,
                    published_at=published_at,
                    collected_at=datetime.now(),
                    language="en",
                    category="market",
                    tags=["reuters"],
                ))

            self._update_source_health("reuters", success=True)

        except Exception as e:
            log.error(f"Reuters fetch failed: {e}")
            self._update_source_health("reuters", success=False)

        return articles

    def _fetch_bloomberg(
        self,
        keywords: Optional[list[str]],
        start_time: datetime,
        limit: int,
    ) -> list[NewsArticle]:
        """Fetch from Bloomberg."""
        # Bloomberg requires API subscription
        log.debug("Bloomberg fetch requires API subscription")
        return []

    def _fetch_yahoo_finance(
        self,
        keywords: Optional[list[str]],
        start_time: datetime,
        limit: int,
    ) -> list[NewsArticle]:
        """Fetch from Yahoo Finance."""
        articles = []
        search_term = keywords[0] if keywords else "China stock market"
        url = "https://query1.finance.yahoo.com/v1/finance/search"
        params = {
            "q": search_term,
            "quotesCount": 0,
            "newsCount": limit,
        }

        try:
            resp = self._session.get(url, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()

            for item in data.get("news", [])[:limit]:
                title = item.get("title", "")
                content = item.get("summary", "")
                url = item.get("link", "")
                pub_time = item.get("providerPublishTime", 0)

                if not title:
                    continue

                try:
                    published_at = datetime.fromtimestamp(pub_time)
                except Exception:
                    published_at = datetime.now()

                if published_at < start_time:
                    continue

                article_id = self._generate_id(f"yahoo_{title}")
                articles.append(NewsArticle(
                    id=article_id,
                    title=title,
                    content=content or title,
                    summary=title[:200],
                    source="yahoo_finance",
                    url=url,
                    published_at=published_at,
                    collected_at=datetime.now(),
                    language="en",
                    category="market",
                    tags=["yahoo"],
                ))

            self._update_source_health("yahoo_finance", success=True)

        except Exception as e:
            log.error(f"Yahoo Finance fetch failed: {e}")
            self._update_source_health("yahoo_finance", success=False)

        return articles

    def _generate_id(self, text: str) -> str:
        """Generate unique ID from text."""
        return hashlib.md5(text.encode()).hexdigest()[:16]

    def _update_source_health(self, source: str, success: bool) -> None:
        """Update source health score."""
        now = time.time()
        current_health = self._source_health.get(source, 0.5)

        if success:
            new_health = min(1.0, current_health + 0.1)
        else:
            new_health = max(0.0, current_health - 0.2)

        self._source_health[source] = new_health
        self._last_health_check[source] = now

    def _categorize_article(self, article: NewsArticle) -> None:
        """Categorize article based on content."""
        text = (article.title + " " + article.content).lower()

        # Policy/regulatory keywords
        policy_zh = ["政策", "规定", "监管", "证监会", "法规", "条例"]
        policy_en = ["policy", "regulation", "regulatory", "SEC", "rule"]

        # Market keywords
        market_zh = ["股票", "股市", "交易", "股价", "市值"]
        market_en = ["stock", "market", "trading", "share price"]

        # Company keywords
        company_zh = ["公司", "企业", "财报", "业绩", "盈利"]
        company_en = ["company", "earnings", "revenue", "profit"]

        # Count matches
        policy_count = sum(1 for kw in policy_zh + policy_en if kw in text)
        market_count = sum(1 for kw in market_zh + market_en if kw in text)
        company_count = sum(1 for kw in company_zh + company_en if kw in text)

        # Assign category
        scores = {
            "policy": policy_count,
            "market": market_count,
            "company": company_count,
        }

        if max(scores.values()) > 0:
            article.category = max(scores, key=scores.get)
        else:
            article.category = "market"  # Default

    def _calculate_relevance(
        self,
        article: NewsArticle,
        keywords: Optional[list[str]],
    ) -> None:
        """Calculate relevance score based on keywords and category."""
        score = 0.5  # Base score

        if keywords:
            text = (article.title + " " + article.content).lower()
            keyword_matches = sum(1 for kw in keywords if kw.lower() in text)
            score += min(0.5, keyword_matches * 0.1)

        # Policy news is highly relevant for trading
        if article.category == "policy":
            score += 0.2

        article.relevance_score = min(1.0, score)

    def save_articles(self, articles: list[NewsArticle], filename: Optional[str] = None) -> Path:
        """Save articles to JSON file."""
        if filename is None:
            filename = f"news_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        filepath = self.cache_dir / filename
        data = [article.to_dict() for article in articles]

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        log.info(f"Saved {len(articles)} articles to {filepath}")
        return filepath

    def load_articles(self, filepath: Path) -> list[NewsArticle]:
        """Load articles from JSON file."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        return [NewsArticle.from_dict(item) for item in data]


# Singleton instance
_collector: Optional[NewsCollector] = None


def get_collector() -> NewsCollector:
    """Get or create news collector instance."""
    global _collector
    if _collector is None:
        _collector = NewsCollector()
    return _collector


def reset_collector() -> None:
    """Reset collector instance."""
    global _collector
    _collector = None
