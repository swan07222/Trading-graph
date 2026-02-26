"""Enhanced Web Search Module for China Network.

This module provides China-compatible web search capabilities for collecting
LLM training data and market intelligence from various search engines.

Features:
    - China-compatible search engines (Bing CN, Baidu, Sogou)
    - International search engines (Google, DuckDuckGo)
    - Async search with retry logic optimized for China
    - Result deduplication and quality scoring
    - Caching and rate limiting
    - LLM training data collection

Example:
    >>> from data.web_search import WebSearchEngine
    >>> search_engine = WebSearchEngine()
    >>> results = await search_engine.search("A 股政策利好", limit=20)
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from pathlib import Path
from typing import Any

import aiohttp
from aiohttp import ClientTimeout

from config.runtime_env import env_text, env_int
from config.settings import CONFIG
from utils.async_http import AsyncHttpClient, HttpClientConfig
from utils.logger import get_logger

log = get_logger(__name__)


class SearchEngine(Enum):
    """Supported search engines."""
    BING_CN = auto()  # Bing China (cn.bing.com)
    BAIDU = auto()  # Baidu (www.baidu.com)
    SOGOU = auto()  # Sogou (www.sogou.com)
    GOOGLE = auto()  # Google (www.google.com)
    DUCKDUCKGO = auto()  # DuckDuckGo (duckduckgo.com)
    EASTMONEY_SEARCH = auto()  # EastMoney Search
    SINA_SEARCH = auto()  # Sina Finance Search


@dataclass
class SearchResult:
    """Normalized search result."""
    id: str
    title: str
    snippet: str
    url: str
    source: str
    engine: SearchEngine
    rank: int
    published_at: datetime | None = None
    language: str = "zh"
    quality_score: float = 0.0
    content: str = ""  # Full content if fetched

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "snippet": self.snippet,
            "url": self.url,
            "source": self.source,
            "engine": self.engine.name,
            "rank": self.rank,
            "published_at": self.published_at.isoformat() if self.published_at else None,
            "language": self.language,
            "quality_score": self.quality_score,
            "content": self.content,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SearchResult":
        published = None
        if data.get("published_at"):
            try:
                published = datetime.fromisoformat(data["published_at"])
            except (ValueError, TypeError):
                pass
        return cls(
            id=data["id"],
            title=data["title"],
            snippet=data["snippet"],
            url=data["url"],
            source=data["source"],
            engine=SearchEngine[data["engine"]],
            rank=data["rank"],
            published_at=published,
            language=data.get("language", "zh"),
            quality_score=float(data.get("quality_score", 0.0)),
            content=data.get("content", ""),
        )


@dataclass
class SearchQuery:
    """Search query configuration."""
    query: str
    engines: list[SearchEngine] = field(default_factory=list)
    limit: int = 20
    hours_back: int = 168  # 7 days
    language: str = "zh"
    fetch_content: bool = False  # Fetch full page content
    cache_results: bool = True


class SearchEngineManager:
    """Manages search engine availability and health."""

    # China-compatible engines (no VPN required)
    CHINA_ENGINES = [
        SearchEngine.BING_CN,
        SearchEngine.BAIDU,
        SearchEngine.SOGOU,
        SearchEngine.EASTMONEY_SEARCH,
        SearchEngine.SINA_SEARCH,
    ]

    # International engines (VPN required)
    INTERNATIONAL_ENGINES = [
        SearchEngine.GOOGLE,
        SearchEngine.DUCKDUCKGO,
    ]

    # Engine priorities for China mode
    CHINA_PRIORITY = {
        SearchEngine.BING_CN: 1,
        SearchEngine.BAIDU: 2,
        SearchEngine.SOGOU: 3,
        SearchEngine.EASTMONEY_SEARCH: 4,
        SearchEngine.SINA_SEARCH: 5,
    }

    # Engine priorities for VPN mode
    VPN_PRIORITY = {
        SearchEngine.GOOGLE: 1,
        SearchEngine.BING_CN: 2,
        SearchEngine.DUCKDUCKGO: 3,
        SearchEngine.BAIDU: 4,
    }

    def __init__(self) -> None:
        self._health_scores: dict[SearchEngine, float] = {
            engine: 1.0 for engine in list(self.CHINA_ENGINES) + list(self.INTERNATIONAL_ENGINES)
        }
        self._last_check: dict[SearchEngine, float] = {}
        self._cooldown: dict[SearchEngine, float] = {}
        self._lock = asyncio.Lock()

    def is_vpn_mode(self) -> bool:
        """Check if VPN mode is enabled."""
        return env_text("TRADING_VPN", "0") == "1"

    def get_available_engines(self) -> list[SearchEngine]:
        """Get available engines based on network mode."""
        if self.is_vpn_mode():
            return self.INTERNATIONAL_ENGINES + self.CHINA_ENGINES
        return self.CHINA_ENGINES

    def get_priority_order(self) -> list[SearchEngine]:
        """Get engines in priority order."""
        if self.is_vpn_mode():
            priority = self.VPN_PRIORITY
        else:
            priority = self.CHINA_PRIORITY

        available = self.get_available_engines()
        return sorted(available, key=lambda e: priority.get(e, 999))

    async def record_success(self, engine: SearchEngine) -> None:
        """Record successful search operation."""
        async with self._lock:
            self._health_scores[engine] = min(1.0, self._health_scores.get(engine, 0.5) + 0.1)
            self._last_check[engine] = time.time()
            self._cooldown.pop(engine, None)

    async def record_failure(self, engine: SearchEngine, error: str) -> None:
        """Record failed search operation."""
        async with self._lock:
            self._health_scores[engine] = max(0.0, self._health_scores.get(engine, 0.5) - 0.2)
            self._last_check[engine] = time.time()
            # Add cooldown for repeated failures
            if self._health_scores[engine] < 0.3:
                self._cooldown[engine] = time.time() + 300  # 5 min cooldown
            log.warning(f"Search engine {engine.name} failed: {error}")

    async def is_available(self, engine: SearchEngine) -> bool:
        """Check if engine is available (not in cooldown)."""
        async with self._lock:
            cooldown_end = self._cooldown.get(engine)
            if cooldown_end and time.time() < cooldown_end:
                return False
            return True

    def get_health(self, engine: SearchEngine) -> float:
        """Get health score for engine."""
        return self._health_scores.get(engine, 0.5)


class SearchResultCache:
    """Cache for search results with TTL support."""

    def __init__(self, cache_dir: Path | None = None, ttl_hours: int = 24) -> None:
        self.cache_dir = cache_dir or CONFIG.cache_dir / "web_search"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl_hours = ttl_hours
        self._memory_cache: dict[str, tuple[list[dict], float]] = {}
        self._memory_ttl = 300  # 5 minutes for memory cache

    def _get_cache_key(self, query: str, engines: list[SearchEngine], limit: int) -> str:
        """Generate cache key for query."""
        key_data = f"{query}:{','.join(e.name for e in engines)}:{limit}"
        return hashlib.md5(key_data.encode()).hexdigest()

    def _get_cache_file(self, key: str) -> Path:
        """Get cache file path."""
        return self.cache_dir / f"{key}.json"

    async def get(self, query: str, engines: list[SearchEngine], limit: int) -> list[SearchResult] | None:
        """Get cached results if available and not expired."""
        key = self._get_cache_key(query, engines, limit)

        # Check memory cache first
        if key in self._memory_cache:
            data, timestamp = self._memory_cache[key]
            if time.time() - timestamp < self._memory_ttl:
                log.debug(f"Cache hit (memory) for query: {query[:50]}")
                return [SearchResult.from_dict(item) for item in data]
            else:
                del self._memory_cache[key]

        # Check disk cache
        cache_file = self._get_cache_file(key)
        if cache_file.exists():
            try:
                mtime = datetime.fromtimestamp(cache_file.stat().st_mtime)
                age = datetime.now() - mtime
                if age.total_seconds() < self.ttl_hours * 3600:
                    content = cache_file.read_text(encoding="utf-8")
                    data = json.loads(content)
                    # Store in memory cache
                    self._memory_cache[key] = (data, time.time())
                    log.debug(f"Cache hit (disk) for query: {query[:50]}")
                    return [SearchResult.from_dict(item) for item in data]
                else:
                    cache_file.unlink()
            except (json.JSONDecodeError, OSError) as e:
                log.warning(f"Cache file corrupted: {e}")
                cache_file.unlink(missing_ok=True)

        return None

    async def set(self, query: str, engines: list[SearchEngine], limit: int,
                  results: list[SearchResult]) -> None:
        """Cache search results."""
        key = self._get_cache_key(query, engines, limit)
        data = [r.to_dict() for r in results]

        # Store in memory cache
        self._memory_cache[key] = (data, time.time())

        # Store in disk cache
        cache_file = self._get_cache_file(key)
        try:
            content = json.dumps(data, ensure_ascii=False, indent=2)
            cache_file.write_text(content, encoding="utf-8")
        except OSError as e:
            log.warning(f"Failed to write cache: {e}")

    async def clear(self) -> None:
        """Clear all cached results."""
        self._memory_cache.clear()
        for cache_file in self.cache_dir.glob("*.json"):
            try:
                cache_file.unlink()
            except OSError:
                pass


class WebSearchEngine:
    """Main web search engine with China network optimization.

    This class provides unified interface for searching multiple search engines
    with automatic failover, caching, and China network optimization.

    Example:
        >>> search_engine = WebSearchEngine()
        >>> results = await search_engine.search("A 股政策", limit=20)
        >>> for result in results:
        ...     print(f"{result.title}: {result.url}")
    """

    # LLM training data keywords (Chinese)
    LLM_TRAINING_KEYWORDS_ZH = [
        "A 股 政策",
        "股市 监管",
        "货币政策 央行",
        "财政政策 财政部",
        "产业政策 支持",
        "新股 IPO",
        "退市规定",
        "交易规则",
        "印花税",
        "股市 分析",
        "股票 估值",
        "量化交易",
        "技术分析",
        "基本面分析",
        "市场情绪",
        "资金流向",
        "北向资金",
        "南向资金",
        "融资融券",
        "股指期货",
    ]

    # LLM training data keywords (English)
    LLM_TRAINING_KEYWORDS_EN = [
        "China A-share policy",
        "stock market regulation",
        "monetary policy PBOC",
        "fiscal policy China",
        "industrial policy support",
        "IPO new listing",
        "delisting rules",
        "trading regulations",
        "stamp duty tax",
        "stock market analysis",
        "stock valuation",
        "quantitative trading",
        "technical analysis",
        "fundamental analysis",
        "market sentiment",
        "capital flow",
        "northbound capital",
        "southbound capital",
        "margin financing",
        "stock index futures",
    ]

    # Quality scoring weights
    TITLE_WEIGHT = 0.3
    SNIPPET_WEIGHT = 0.2
    SOURCE_WEIGHT = 0.2
    RECENCY_WEIGHT = 0.15
    RANK_WEIGHT = 0.15

    # High-quality source domains
    HIGH_QUALITY_DOMAINS = [
        # Chinese government/regulatory
        "gov.cn", "csrc.gov.cn", "pbc.gov.cn", "mof.gov.cn",
        # Financial news
        "eastmoney.com", "sina.com.cn", "caixin.com", "10jqka.com.cn",
        "jrj.com.cn", "cnstock.com", "stcn.com",
        # Stock exchanges
        "sse.com.cn", "szse.cn", "bse.cn",
        # International (VPN mode)
        "reuters.com", "bloomberg.com", "wsj.com", "ft.com",
    ]

    def __init__(
        self,
        cache_dir: Path | None = None,
        cache_ttl_hours: int = 24,
        max_concurrent: int = 5,
    ) -> None:
        """Initialize web search engine.

        Args:
            cache_dir: Directory for caching search results
            cache_ttl_hours: Cache TTL in hours
            max_concurrent: Maximum concurrent search requests
        """
        self.cache = SearchResultCache(cache_dir, cache_ttl_hours)
        self.engine_manager = SearchEngineManager()
        self.max_concurrent = max_concurrent
        self._semaphore = asyncio.Semaphore(max_concurrent)

        # Rate limiting per engine
        self._rate_limits: dict[SearchEngine, float] = {
            SearchEngine.BING_CN: 2.0,  # 2 seconds between requests
            SearchEngine.BAIDU: 3.0,
            SearchEngine.SOGOU: 3.0,
            SearchEngine.GOOGLE: 1.0,
            SearchEngine.DUCKDUCKGO: 1.0,
            SearchEngine.EASTMONEY_SEARCH: 2.0,
            SearchEngine.SINA_SEARCH: 2.0,
        }
        self._last_request: dict[SearchEngine, float] = {}
        self._rate_limit_lock = asyncio.Lock()

    async def _wait_rate_limit(self, engine: SearchEngine) -> None:
        """Wait for rate limit to be satisfied."""
        async with self._rate_limit_lock:
            last = self._last_request.get(engine, 0)
            min_interval = self._rate_limits.get(engine, 2.0)
            elapsed = time.time() - last
            if elapsed < min_interval:
                await asyncio.sleep(min_interval - elapsed)
            self._last_request[engine] = time.time()

    def _calculate_quality_score(
        self,
        result: SearchResult,
        query: str,
    ) -> float:
        """Calculate quality score for search result."""
        score = 0.0

        # Title relevance (keyword match)
        query_terms = query.lower().split()
        title_lower = result.title.lower()
        title_matches = sum(1 for term in query_terms if term in title_lower)
        title_score = title_matches / max(len(query_terms), 1)

        # Snippet relevance
        snippet_lower = result.snippet.lower()
        snippet_matches = sum(1 for term in query_terms if term in snippet_lower)
        snippet_score = snippet_matches / max(len(query_terms), 1)

        # Source quality
        source_score = 0.5
        for domain in self.HIGH_QUALITY_DOMAINS:
            if domain in result.url.lower():
                source_score = 1.0
                break

        # Recency score
        recency_score = 0.5
        if result.published_at:
            age_hours = (datetime.now() - result.published_at).total_seconds() / 3600
            if age_hours < 24:
                recency_score = 1.0
            elif age_hours < 72:
                recency_score = 0.8
            elif age_hours < 168:  # 7 days
                recency_score = 0.6
            elif age_hours > 720:  # 30 days
                recency_score = 0.2

        # Rank score (higher rank = better)
        rank_score = 1.0 / (1.0 + result.rank * 0.1)

        # Weighted sum
        score = (
            title_score * self.TITLE_WEIGHT +
            snippet_score * self.SNIPPET_WEIGHT +
            source_score * self.SOURCE_WEIGHT +
            recency_score * self.RECENCY_WEIGHT +
            rank_score * self.RANK_WEIGHT
        )

        return min(1.0, max(0.0, score))

    async def _search_bing_cn(self, query: str, limit: int) -> list[SearchResult]:
        """Search Bing China."""
        await self._wait_rate_limit(SearchEngine.BING_CN)

        url = "https://cn.bing.com/search"
        params = {"q": query, "count": limit}

        async with self._semaphore:
            client = AsyncHttpClient()
            try:
                response = await client.get(url, params=params)
                html = response.text()
                return self._parse_bing_results(html, query, limit)
            except Exception as e:
                log.warning(f"Bing CN search failed: {e}")
                return []

    def _parse_bing_results(self, html: str, query: str, limit: int) -> list[SearchResult]:
        """Parse Bing search results from HTML."""
        results = []
        # Simple regex-based parsing (production should use BeautifulSoup)
        pattern = r'<li class="b_algo"(.*?)</li>'
        matches = re.findall(pattern, html, re.DOTALL | re.IGNORECASE)

        for i, match in enumerate(matches[:limit]):
            try:
                # Extract title
                title_match = re.search(r'<h2.*?>.*?<a.*?>(.*?)</a>', match, re.DOTALL)
                title = re.sub(r'<.*?>', '', title_match.group(1) or "") if title_match else ""

                # Extract URL
                url_match = re.search(r'<a href="(.*?)"', match)
                url = url_match.group(1) if url_match else ""

                # Extract snippet
                snippet_match = re.search(r'<div class="b_caption".*?>(.*?)</div>', match, re.DOTALL)
                snippet = re.sub(r'<.*?>', '', snippet_match.group(1) or "") if snippet_match else ""

                if title and url:
                    result = SearchResult(
                        id=hashlib.md5(url.encode()).hexdigest(),
                        title=title.strip(),
                        snippet=snippet.strip()[:500],
                        url=url,
                        source="Bing CN",
                        engine=SearchEngine.BING_CN,
                        rank=i,
                        language="zh",
                    )
                    result.quality_score = self._calculate_quality_score(result, query)
                    results.append(result)
            except Exception as e:
                log.warning(f"Failed to parse Bing result: {e}")

        return results

    async def _search_baidu(self, query: str, limit: int) -> list[SearchResult]:
        """Search Baidu."""
        await self._wait_rate_limit(SearchEngine.BAIDU)

        url = "https://www.baidu.com/s"
        params = {"wd": query, "rn": limit}

        async with self._semaphore:
            client = AsyncHttpClient()
            try:
                response = await client.get(url, params=params)
                html = response.text()
                return self._parse_baidu_results(html, query, limit)
            except Exception as e:
                log.warning(f"Baidu search failed: {e}")
                return []

    def _parse_baidu_results(self, html: str, query: str, limit: int) -> list[SearchResult]:
        """Parse Baidu search results from HTML."""
        results = []
        # Baidu result containers
        pattern = r'<div class="result c-container"(.*?)</div>'
        matches = re.findall(pattern, html, re.DOTALL)

        for i, match in enumerate(matches[:limit]):
            try:
                # Extract title
                title_match = re.search(r'<h3 class="t".*?>.*?<a.*?>(.*?)</a>', match, re.DOTALL)
                title = re.sub(r'<.*?>', '', title_match.group(1) or "") if title_match else ""

                # Extract URL
                url_match = re.search(r'<a href="(.*?)"', match)
                url = url_match.group(1) if url_match else ""

                # Extract snippet
                snippet_match = re.search(r'<div class="c-abstract">(.*?)</div>', match, re.DOTALL)
                snippet = re.sub(r'<.*?>', '', snippet_match.group(1) or "") if snippet_match else ""

                if title and url:
                    result = SearchResult(
                        id=hashlib.md5(url.encode()).hexdigest(),
                        title=title.strip(),
                        snippet=snippet.strip()[:500],
                        url=url,
                        source="Baidu",
                        engine=SearchEngine.BAIDU,
                        rank=i,
                        language="zh",
                    )
                    result.quality_score = self._calculate_quality_score(result, query)
                    results.append(result)
            except Exception as e:
                log.warning(f"Failed to parse Baidu result: {e}")

        return results

    async def _search_sogou(self, query: str, limit: int) -> list[SearchResult]:
        """Search Sogou."""
        await self._wait_rate_limit(SearchEngine.SOGOU)

        url = "https://www.sogou.com/web"
        params = {"query": query, "num": limit}

        async with self._semaphore:
            client = AsyncHttpClient()
            try:
                response = await client.get(url, params=params)
                html = response.text()
                return self._parse_sogou_results(html, query, limit)
            except Exception as e:
                log.warning(f"Sogou search failed: {e}")
                return []

    def _parse_sogou_results(self, html: str, query: str, limit: int) -> list[SearchResult]:
        """Parse Sogou search results from HTML."""
        results = []
        pattern = r'<div class="fb-hint"(.*?)</div>'
        matches = re.findall(pattern, html, re.DOTALL)

        for i, match in enumerate(matches[:limit]):
            try:
                title_match = re.search(r'<a.*?>(.*?)</a>', match, re.DOTALL)
                title = re.sub(r'<.*?>', '', title_match.group(1) or "") if title_match else ""

                url_match = re.search(r'<a href="(.*?)"', match)
                url = url_match.group(1) if url_match else ""

                snippet_match = re.search(r'<div class="attribute".*?>(.*?)</div>', match, re.DOTALL)
                snippet = re.sub(r'<.*?>', '', snippet_match.group(1) or "") if snippet_match else ""

                if title and url:
                    result = SearchResult(
                        id=hashlib.md5(url.encode()).hexdigest(),
                        title=title.strip(),
                        snippet=snippet.strip()[:500],
                        url=url,
                        source="Sogou",
                        engine=SearchEngine.SOGOU,
                        rank=i,
                        language="zh",
                    )
                    result.quality_score = self._calculate_quality_score(result, query)
                    results.append(result)
            except Exception as e:
                log.warning(f"Failed to parse Sogou result: {e}")

        return results

    async def _search_eastmoney(self, query: str, limit: int) -> list[SearchResult]:
        """Search EastMoney finance news."""
        await self._wait_rate_limit(SearchEngine.EASTMONEY_SEARCH)

        url = "https://search-api-web.eastmoney.com/search/jsonp"
        params = {
            "type": "news",
            "keyword": query,
            "pageIndex": 1,
            "pageSize": limit,
            "sort": "recommend",
        }

        async with self._semaphore:
            client = AsyncHttpClient()
            try:
                response = await client.get(url, params=params)
                text = response.text()
                # EastMoney returns JSONP, extract JSON
                json_match = re.search(r'\((.*)\)', text)
                if json_match:
                    data = json.loads(json_match.group(1))
                    return self._parse_eastmoney_results(data, query, limit)
                return []
            except Exception as e:
                log.warning(f"EastMoney search failed: {e}")
                return []

    def _parse_eastmoney_results(
        self,
        data: dict[str, Any],
        query: str,
        limit: int,
    ) -> list[SearchResult]:
        """Parse EastMoney search results."""
        results = []
        items = data.get("Data", []) if isinstance(data, dict) else []

        for i, item in enumerate(items[:limit]):
            try:
                title = item.get("Title", "")
                url = item.get("Url", "")
                snippet = item.get("Content", "")
                pub_time = item.get("ShowTime", "")

                published_at = None
                if pub_time:
                    try:
                        published_at = datetime.fromisoformat(pub_time.replace(" ", "T"))
                    except (ValueError, TypeError):
                        pass

                if title and url:
                    result = SearchResult(
                        id=hashlib.md5(url.encode()).hexdigest(),
                        title=title.strip(),
                        snippet=snippet.strip()[:500],
                        url=url,
                        source="EastMoney",
                        engine=SearchEngine.EASTMONEY_SEARCH,
                        rank=i,
                        published_at=published_at,
                        language="zh",
                    )
                    result.quality_score = self._calculate_quality_score(result, query)
                    results.append(result)
            except Exception as e:
                log.warning(f"Failed to parse EastMoney result: {e}")

        return results

    async def _search_sina_finance(self, query: str, limit: int) -> list[SearchResult]:
        """Search Sina Finance."""
        await self._wait_rate_limit(SearchEngine.SINA_SEARCH)

        # Sina finance search uses a different API
        url = "https://search.sina.com.cn/"
        params = {"q": query, "c": "finance", "num": limit}

        async with self._semaphore:
            client = AsyncHttpClient()
            try:
                response = await client.get(url, params=params)
                html = response.text()
                return self._parse_sina_results(html, query, limit)
            except Exception as e:
                log.warning(f"Sina Finance search failed: {e}")
                return []

    def _parse_sina_results(self, html: str, query: str, limit: int) -> list[SearchResult]:
        """Parse Sina Finance search results."""
        results = []
        pattern = r'<div class="result-block"(.*?)</div>'
        matches = re.findall(pattern, html, re.DOTALL)

        for i, match in enumerate(matches[:limit]):
            try:
                title_match = re.search(r'<h2.*?>.*?<a.*?>(.*?)</a>', match, re.DOTALL)
                title = re.sub(r'<.*?>', '', title_match.group(1) or "") if title_match else ""

                url_match = re.search(r'<a href="(.*?)"', match)
                url = url_match.group(1) if url_match else ""

                snippet_match = re.search(r'<p class="summary">(.*?)</p>', match, re.DOTALL)
                snippet = re.sub(r'<.*?>', '', snippet_match.group(1) or "") if snippet_match else ""

                if title and url:
                    result = SearchResult(
                        id=hashlib.md5(url.encode()).hexdigest(),
                        title=title.strip(),
                        snippet=snippet.strip()[:500],
                        url=url,
                        source="Sina Finance",
                        engine=SearchEngine.SINA_SEARCH,
                        rank=i,
                        language="zh",
                    )
                    result.quality_score = self._calculate_quality_score(result, query)
                    results.append(result)
            except Exception as e:
                log.warning(f"Failed to parse Sina result: {e}")

        return results

    async def search(
        self,
        query: str,
        engines: list[SearchEngine] | None = None,
        limit: int = 20,
        use_cache: bool = True,
        deduplicate: bool = True,
        min_quality: float = 0.3,
    ) -> list[SearchResult]:
        """Search web with multiple engines.

        Args:
            query: Search query string
            engines: List of search engines to use (None = auto-select)
            limit: Maximum number of results
            use_cache: Whether to use cached results
            deduplicate: Remove duplicate results
            min_quality: Minimum quality score threshold

        Returns:
            List of search results sorted by quality score
        """
        # Check cache first
        if use_cache:
            cached = await self.cache.get(query, engines or [], limit)
            if cached:
                return cached

        # Auto-select engines if not specified
        if engines is None:
            engines = self.engine_manager.get_priority_order()[:3]

        # Filter available engines
        available_engines = []
        for engine in engines:
            if await self.engine_manager.is_available(engine):
                available_engines.append(engine)
            else:
                log.debug(f"Engine {engine.name} unavailable (cooldown)")

        if not available_engines:
            log.warning("No search engines available")
            return []

        # Search concurrently
        search_tasks = []
        for engine in available_engines:
            if engine == SearchEngine.BING_CN:
                search_tasks.append(self._search_bing_cn(query, limit))
            elif engine == SearchEngine.BAIDU:
                search_tasks.append(self._search_baidu(query, limit))
            elif engine == SearchEngine.SOGOU:
                search_tasks.append(self._search_sogou(query, limit))
            elif engine == SearchEngine.EASTMONEY_SEARCH:
                search_tasks.append(self._search_eastmoney(query, limit))
            elif engine == SearchEngine.SINA_SEARCH:
                search_tasks.append(self._search_sina_finance(query, limit))

        all_results = []
        try:
            results_list = await asyncio.gather(*search_tasks, return_exceptions=True)
            for results in results_list:
                if isinstance(results, list):
                    all_results.extend(results)
                elif isinstance(results, Exception):
                    log.warning(f"Search task failed: {results}")
        except Exception as e:
            log.error(f"Search failed: {e}")

        # Deduplicate results
        if deduplicate:
            seen_urls = set()
            unique_results = []
            for result in all_results:
                if result.url not in seen_urls:
                    seen_urls.add(result.url)
                    unique_results.append(result)
            all_results = unique_results

        # Filter by quality
        all_results = [r for r in all_results if r.quality_score >= min_quality]

        # Sort by quality score
        all_results.sort(key=lambda r: r.quality_score, reverse=True)

        # Limit results
        final_results = all_results[:limit]

        # Update engine health
        if final_results:
            for engine in available_engines:
                await self.engine_manager.record_success(engine)
        else:
            for engine in available_engines:
                await self.engine_manager.record_failure(engine, "No results")

        # Cache results
        if use_cache and final_results:
            await self.cache.set(query, engines or [], limit, final_results)

        return final_results

    async def search_for_llm_training(
        self,
        language: str = "zh",
        limit_per_query: int = 20,
        max_queries: int = 10,
    ) -> list[SearchResult]:
        """Search for LLM training data.

        This method searches for content specifically useful for LLM training,
        including policy documents, market analysis, and financial news.

        Args:
            language: Language preference ("zh" or "en")
            limit_per_query: Results per query
            max_queries: Maximum number of queries to search

        Returns:
            List of search results for LLM training
        """
        keywords = (
            self.LLM_TRAINING_KEYWORDS_ZH if language == "zh"
            else self.LLM_TRAINING_KEYWORDS_EN
        )

        all_results = []
        queries_to_run = min(max_queries, len(keywords))

        log.info(f"Searching for LLM training data: {queries_to_run} queries")

        for i, keyword in enumerate(keywords[:queries_to_run]):
            try:
                results = await self.search(
                    query=keyword,
                    limit=limit_per_query,
                    use_cache=True,
                )
                all_results.extend(results)
                log.debug(f"Query {i+1}/{queries_to_run}: '{keyword}' -> {len(results)} results")

                # Small delay between queries to avoid rate limiting
                if i < queries_to_run - 1:
                    await asyncio.sleep(1.0)

            except Exception as e:
                log.warning(f"Query '{keyword}' failed: {e}")

        # Deduplicate and sort
        seen_urls = set()
        unique_results = []
        for result in all_results:
            if result.url not in seen_urls:
                seen_urls.add(result.url)
                unique_results.append(result)

        unique_results.sort(key=lambda r: r.quality_score, reverse=True)

        log.info(f"Collected {len(unique_results)} unique results for LLM training")

        return unique_results

    async def fetch_content(self, url: str) -> str | None:
        """Fetch full content from URL.

        Args:
            url: URL to fetch

        Returns:
            Page content or None if failed
        """
        async with self._semaphore:
            client = AsyncHttpClient()
            try:
                response = await client.get(url)
                return response.text()
            except Exception as e:
                log.warning(f"Failed to fetch content from {url}: {e}")
                return None

    def clear_cache(self) -> None:
        """Clear search result cache."""
        asyncio.create_task(self.cache.clear())
        log.info("Search result cache cleared")


# Global instance
_web_search: WebSearchEngine | None = None


def get_search_engine() -> WebSearchEngine:
    """Get or create global search engine instance."""
    global _web_search
    if _web_search is None:
        _web_search = WebSearchEngine()
    return _web_search


def reset_search_engine() -> None:
    """Reset global search engine instance (for testing)."""
    global _web_search
    _web_search = None
