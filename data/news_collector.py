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
        # FIX: Guard datetime parsing so a malformed timestamp in cache doesn't discard
        # the entire cache file — fall back to datetime.now() on parse failure.
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
    """Multi-source news collector with VPN-aware routing."""

    # Chinese sources (VPN off)
    # FIX #10: Removed dead/unavailable sources that poisoned health scores every cycle:
    #   - jin10: requires paid API subscription (always fails → health death-spiral)
    #   - xueqiu: requires authentication (always returns [] → unnecessary health penalty)
    #   - csrc: placeholder, never implemented (always returns [] → unnecessary health penalty)
    CHINESE_SOURCES = [
        "eastmoney",
        "sina_finance",
        "caixin",
    ]

    # Sources excluded from active rotation because they require credentials or paid access
    _UNAVAILABLE_SOURCES = ["jin10", "xueqiu", "csrc", "bloomberg"]

    # International sources (VPN on)
    INTERNATIONAL_SOURCES = [
        "reuters",
        "bloomberg",
        "yahoo_finance",
        "marketwatch",
        "cnbc",
    ]
    _MIN_HEALTH_FOR_IMMEDIATE_RETRY = 0.20
    _MAX_SOURCE_COOLDOWN_SECONDS = 1800.0

    # Search keywords for policy/regulatory news
    POLICY_KEYWORDS_ZH = [
        "政策",
        "规定",
        "监管",
        "证监会",
        "央行",
        "财政部",
        "货币政策",
        "财政政策",
        "产业政策",
        "法规",
        "条例",
        "新股",
        "IPO",
        "退市",
        "交易规则",
        "印花税",
    ]

    POLICY_KEYWORDS_EN = [
        "policy", "regulation", "regulatory", "SEC", "Federal Reserve",
        "Treasury", "monetary policy", "fiscal policy", "industrial policy",
        "IPO", "delisting", "trading rules", "stamp duty",
    ]

    MARKET_KEYWORDS_ZH = [
        "股票",
        "股市",
        "A股",
        "上证",
        "深证",
        "创业板",
        "成交量",
        "涨停",
        "跌停",
        "牛市",
        "熊市",
    ]

    MARKET_KEYWORDS_EN = [
        "stock", "market", "A-share", "SSE", "SZSE", "GEM",
        "volume", "limit up", "limit down", "bull market", "bear market",
    ]

    # FIX #8: VPN detection cache TTL (5 minutes)
    _VPN_DETECTION_CACHE_TTL = 300.0

    def __init__(self, cache_dir: Path | None = None) -> None:
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
        self._source_failures: dict[str, int] = {}  # source -> consecutive failure count
        self._strict_mode: bool = False
        # FIX #8: Cache VPN detection result to avoid 3 HTTP calls per collect_news() call
        self._vpn_mode_cache: bool | None = None
        self._vpn_mode_cache_time: float = 0.0

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
        from config.runtime_env import env_flag

        # Explicit configuration overrides cache
        if env_flag("TRADING_VPN", False):
            return True
        if env_flag("TRADING_CHINA_DIRECT", False):
            return False

        # FIX #8: Return cached result if still fresh (avoids up to 12s of HTTP overhead per call)
        now = time.time()
        if (
            self._vpn_mode_cache is not None
            and (now - self._vpn_mode_cache_time) < self._VPN_DETECTION_CACHE_TTL
        ):
            return self._vpn_mode_cache

        # Auto-detect by testing access to Chinese vs international sites
        result = self._detect_network_environment()
        self._vpn_mode_cache = result
        self._vpn_mode_cache_time = now
        return result

    def _detect_network_environment(self) -> bool:
        """Auto-detect network environment by testing connectivity.

        FIX 2026-02-26 China Network:
        - Replaced Google test with Baidu (Google blocked in China)
        - Added Sina Finance as secondary China endpoint
        - Improved error handling for GFW scenarios
        - Reduced timeouts for faster detection
        
        FIX 2026-02-25: Replaced bare except Exception with specific
        requests.RequestException to avoid swallowing KeyboardInterrupt
        and other critical exceptions.
        FIX 2026-02-25 #2: Added proper exception handling for all network
        operations to prevent unexpected crashes.
        """
        # FIX China: Test access to Chinese sites (always accessible in China)
        chinese_test = "https://www.baidu.com"
        chinese_test_2 = "https://finance.sina.com.cn"
        # FIX China: Google is blocked, use Bing international as foreign test
        international_test = "https://www.bing.com"

        try:
            # Try Chinese site first (Baidu)
            resp = self._session.get(chinese_test, timeout=4)
            if resp.status_code == 200:
                # Can access Chinese sites, now test second Chinese endpoint
                try:
                    resp2 = self._session.get(chinese_test_2, timeout=4)
                    if resp2.status_code == 200:
                        # Both Chinese sites accessible - China direct mode
                        # Now test if international sites are also accessible (VPN mode)
                        try:
                            resp_int = self._session.get(international_test, timeout=4)
                            if resp_int.status_code == 200:
                                # Can access both Chinese and international - VPN mode
                                return True
                        except requests.RequestException:
                            # Can't access international - China direct mode
                            return False
                        except Exception as e:
                            # Log unexpected errors but don't crash
                            log.debug("International site test failed: %s", e)
                            return False
                except requests.RequestException:
                    # Second Chinese site failed, but first succeeded - likely China direct
                    return False
                except Exception as e:
                    log.debug("Second Chinese site test failed: %s", e)
                    return False
        except requests.RequestException:
            # Network error on Chinese sites - unusual, default to VPN mode
            log.warning("Chinese site test failed, defaulting to VPN mode")
            pass
        except KeyboardInterrupt:
            # Don't swallow keyboard interrupts
            raise
        except Exception as e:
            # Other unexpected errors - log and default to VPN mode
            log.warning("Unexpected error during network detection: %s", e, exc_info=True)
            pass

        # Default to VPN mode (international sources) for safety
        return True

    def get_active_sources(self) -> list[str]:
        """Get list of active sources based on VPN mode."""
        if self.is_vpn_mode():
            return self.INTERNATIONAL_SOURCES
        else:
            return self.CHINESE_SOURCES

    def collect_news(
        self,
        keywords: list[str] | None = None,
        categories: list[str] | None = None,
        limit: int = 100,
        hours_back: int = 24,
        strict: bool = False,
    ) -> list[NewsArticle]:
        """Collect news from active sources.

        Args:
            keywords: Optional keywords to filter by
            categories: Categories to collect ('policy', 'market', 'company', etc.)
            limit: Maximum number of articles to return
            hours_back: How many hours back to collect
            strict: Fail-fast mode (no cooldown skip, no degraded recovery)

        Returns:
            List of collected news articles
        """
        start_time = self._normalize_datetime(datetime.now() - timedelta(hours=hours_back))
        articles: list[NewsArticle] = []
        seen_ids: set[str] = set()
        previous_strict_mode = bool(self._strict_mode)
        self._strict_mode = bool(strict)
        strict_mode = bool(strict)
        from config.runtime_env import env_flag
        force_china_direct = bool(env_flag("TRADING_CHINA_DIRECT", "0"))
        allow_cross_region_fallback = not force_china_direct

        vpn_mode = bool(self.is_vpn_mode())
        active_sources = list(
            self.INTERNATIONAL_SOURCES if vpn_mode else self.CHINESE_SOURCES
        )
        fallback_sources = list(
            self.CHINESE_SOURCES if vpn_mode else self.INTERNATIONAL_SOURCES
        )
        log.info(
            "Collecting news from %s sources (VPN mode: %s)",
            len(active_sources),
            vpn_mode,
        )

        def _collect_from_sources(sources: list[str]) -> None:
            for source in sources:
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

        try:
            _collect_from_sources(active_sources)

            if (
                (not strict_mode)
                and (not articles)
                and allow_cross_region_fallback
            ):
                fallback_pool = [
                    source
                    for source in fallback_sources
                    if source not in active_sources
                ]
                if fallback_pool:
                    log.info(
                        "Primary source pool returned 0 articles; retrying fallback pool (%s mode).",
                        "VPN" if not vpn_mode else "China-direct",
                    )
                    _collect_from_sources(fallback_pool)
            elif (not strict_mode) and (not articles) and (not allow_cross_region_fallback):
                log.info(
                    "Cross-region fallback disabled (TRADING_CHINA_DIRECT=1); "
                    "staying on China-direct source pool.",
                )

            if (not strict_mode) and (not articles):
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
                    log.info(
                        "Recovered %s articles from cache fallback.",
                        len(articles),
                    )
        finally:
            self._strict_mode = previous_strict_mode

        # Sort by published date (newest first)
        articles.sort(key=lambda x: x.published_at, reverse=True)

        # Categorize and score
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
        # 30s, 60s, 120s ... up to 30 minutes.
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
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d %H:%M",
            "%Y/%m/%d %H:%M:%S",
            "%Y/%m/%d %H:%M",
            "%Y-%m-%d",
        ):
            try:
                parsed = datetime.strptime(raw[:19], fmt)
                return NewsCollector._normalize_datetime(parsed)
            except ValueError:
                continue
        for fmt in (
            "%a, %d %b %Y %H:%M:%S %z",
            "%a, %d %b %Y %H:%M:%S GMT",
        ):
            try:
                parsed = datetime.strptime(raw, fmt)
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
        m_jsonp = re.match(
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
        elif source == "marketwatch":
            return self._fetch_marketwatch(keywords, start_time, limit)
        elif source == "cnbc":
            return self._fetch_cnbc(keywords, start_time, limit)
        else:
            return []

    def _fetch_jin10(
        self,
        keywords: list[str] | None,
        start_time: datetime,
        limit: int,
    ) -> list[NewsArticle]:
        """Fetch from Jin10 (璐㈢粡蹇)."""
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

                if not self._is_recent_enough(published_at, start_time):
                    continue

                article_id = self._generate_id(f"jin10_{item.get('id', title)}")
                articles.append(NewsArticle(
                    id=article_id,
                    title=title,
                    content=content,
                    summary=content[:200],
                    source="jin10",
                    url=item.get("url", f"https://flash.jin10.com/detail/{item.get('id', '')}"),
                    published_at=self._normalize_datetime(published_at),
                    collected_at=datetime.now(),
                    language="zh",
                    category="market",
                    tags=["jin10", "flash"],
                ))

            self._update_source_health("jin10", success=True)

        except Exception as e:
            if self._strict_mode:
                raise
            log.debug(f"Jin10 fetch failed (paid API, may not be accessible): {e}")
            self._update_source_health("jin10", success=False)

        return articles

    def _fetch_eastmoney(
        self,
        keywords: list[str] | None,
        start_time: datetime,
        limit: int,
    ) -> list[NewsArticle]:
        """Fetch from EastMoney with JSONP and HTML fallback."""
        articles: list[NewsArticle] = []
        seen_titles: set[str] = set()

        try:
            search_term = str((keywords or ["stock"])[0] or "stock")
            url = "https://search-api-web.eastmoney.com/search/jsonp"
            params = {
                "cb": "jQuery",
                "param": json.dumps(
                    {
                        "uid": "",
                        "keyword": search_term,
                        "type": ["cmsArticleWebOld", "cmsArticleWeb"],
                        "client": "web",
                        "clientType": "web",
                        "clientVersion": "curr",
                        "param": {
                            "cmsArticleWebOld": {
                                "searchScope": "default",
                                "sort": "default",
                                "pageIndex": 1,
                                "pageSize": int(max(1, limit)),
                            },
                            "cmsArticleWeb": {
                                "searchScope": "default",
                                "sort": "default",
                                "pageIndex": 1,
                                "pageSize": int(max(1, limit)),
                            },
                        },
                    },
                    ensure_ascii=False,
                ),
            }

            resp = self._session.get(
                url,
                params=params,
                timeout=8,
                headers={"Referer": "https://www.eastmoney.com/"},
            )
            resp.raise_for_status()
            data = self._decode_json_payload(resp.text)

            rows: list[dict[str, Any]] = []
            if isinstance(data, dict):
                candidates = [
                    data.get("result", {}).get("cmsArticleWebOld", {}).get("list", []),
                    data.get("result", {}).get("cmsArticleWeb", {}).get("list", []),
                    data.get("result", {}).get("news", {}).get("list", []),
                    data.get("data", {}).get("list", []),
                    data.get("list", []),
                ]
                for candidate in candidates:
                    if isinstance(candidate, list) and candidate:
                        rows = [row for row in candidate if isinstance(row, dict)]
                        if rows:
                            break

            for item in rows[: int(max(1, limit * 2))]:
                title = self._clean_text(
                    item.get("title")
                    or item.get("Title")
                    or item.get("name")
                    or item.get("headline")
                    or ""
                )
                if not title:
                    continue
                title_key = title.lower()
                if title_key in seen_titles:
                    continue
                seen_titles.add(title_key)

                published_at = (
                    self._parse_datetime(
                        item.get("date")
                        or item.get("publish_time")
                        or item.get("showTime")
                        or item.get("ShowTime")
                        or item.get("ctime")
                    )
                    or datetime.now()
                )
                if not self._is_recent_enough(published_at, start_time):
                    continue

                content = self._clean_text(
                    item.get("content")
                    or item.get("Content")
                    or item.get("summary")
                    or item.get("mediaName")
                    or title
                )
                article_url = str(
                    item.get("url")
                    or item.get("Url")
                    or item.get("link")
                    or ""
                ).strip()

                article_id = self._generate_id(f"eastmoney_{title}")
                articles.append(
                    NewsArticle(
                        id=article_id,
                        title=title,
                        content=content or title,
                        summary=(content or title)[:200],
                        source="eastmoney",
                        url=article_url,
                        published_at=self._normalize_datetime(published_at),
                        collected_at=datetime.now(),
                        language="zh",
                        category="market",
                        tags=["eastmoney"],
                    )
                )
                if len(articles) >= int(limit):
                    break

            if (not self._strict_mode) and len(articles) < int(limit):
                html_resp = self._session.get(
                    "https://finance.eastmoney.com/",
                    timeout=8,
                )
                html_resp.raise_for_status()
                for match in re.finditer(
                    r"<a[^>]+href=[\"'](?P<url>[^\"']+)[\"'][^>]*>(?P<title>.*?)</a>",
                    str(html_resp.text or ""),
                    flags=re.IGNORECASE | re.DOTALL,
                ):
                    title = self._clean_text(match.group("title"))
                    if len(title) < 8:
                        continue
                    title_key = title.lower()
                    if title_key in seen_titles:
                        continue
                    seen_titles.add(title_key)
                    article_url = str(match.group("url") or "").strip()
                    if article_url.startswith("/"):
                        article_url = f"https://finance.eastmoney.com{article_url}"
                    article_id = self._generate_id(f"eastmoney_html_{title}")
                    # FIX #7: Use start_time as published_at for scraped links (no known publish date).
                    # This prevents injecting arbitrarily old content by assigning datetime.now()
                    # to every scraped link, which would always pass the recency check.
                    articles.append(
                        NewsArticle(
                            id=article_id,
                            title=title,
                            content=title,
                            summary=title[:200],
                            source="eastmoney",
                            url=article_url,
                            published_at=start_time,
                            collected_at=datetime.now(),
                            language="zh",
                            category="market",
                            tags=["eastmoney", "html_fallback"],
                        )
                    )
                    if len(articles) >= int(limit):
                        break

            self._update_source_health("eastmoney", success=bool(articles))

        except Exception as e:
            if self._strict_mode:
                raise
            log.debug("EastMoney fetch degraded: %s", e)
            self._update_source_health("eastmoney", success=False)

        return articles[: int(limit)]

    def _fetch_sina_finance(
        self,
        keywords: list[str] | None,
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

                if not self._is_recent_enough(published_at, start_time):
                    continue

                article_id = self._generate_id(f"sina_{title}")
                articles.append(NewsArticle(
                    id=article_id,
                    title=title,
                    content=intro or title,
                    summary=title[:200],
                    source="sina_finance",
                    url=url,
                    published_at=self._normalize_datetime(published_at),
                    collected_at=datetime.now(),
                    language="zh",
                    category="market",
                    tags=["sina"],
                ))

            self._update_source_health("sina_finance", success=True)

        except Exception as e:
            if self._strict_mode:
                raise
            log.error(f"Sina Finance fetch failed: {e}")
            self._update_source_health("sina_finance", success=False)

        return articles

    def _fetch_xueqiu(
        self,
        keywords: list[str] | None,
        start_time: datetime,
        limit: int,
    ) -> list[NewsArticle]:
        """Fetch from Xueqiu (闆悆)."""
        # Xueqiu requires authentication for API access
        # This is a simplified implementation
        articles = []
        log.debug("Xueqiu fetch skipped (requires authentication)")
        return articles

    def _fetch_caixin(
        self,
        keywords: list[str] | None,
        start_time: datetime,
        limit: int,
    ) -> list[NewsArticle]:
        """Fetch from Caixin Global homepage links (API endpoint is unstable)."""
        articles: list[NewsArticle] = []
        seen_titles: set[str] = set()
        keyword_set = [str(k).strip().lower() for k in (keywords or []) if str(k).strip()]

        try:
            url = "https://api.caixin.com/api/content/list"
            params = {
                "columnid": "20",
                "num": int(max(1, limit)),
            }
            try:
                resp = self._session.get(url, params=params, timeout=6)
                if resp.status_code < 400:
                    payload = self._decode_json_payload(resp.text)
                else:
                    payload = None
            except Exception:
                payload = None

            if isinstance(payload, dict):
                rows = payload.get("data", {}).get("list", [])
                if isinstance(rows, list):
                    for item in rows:
                        if not isinstance(item, dict):
                            continue
                        title = self._clean_text(item.get("title", ""))
                        if not title:
                            continue
                        title_key = title.lower()
                        if title_key in seen_titles:
                            continue
                        seen_titles.add(title_key)
                        published_at = self._parse_datetime(item.get("time")) or datetime.now()
                        if not self._is_recent_enough(published_at, start_time):
                            continue
                        content = self._clean_text(item.get("content") or title)
                        article_url = str(item.get("url", "") or "").strip()
                        articles.append(
                            NewsArticle(
                                id=self._generate_id(f"caixin_api_{title}"),
                                title=title,
                                content=content,
                                summary=content[:200],
                                source="caixin",
                                url=article_url,
                                published_at=self._normalize_datetime(published_at),
                                collected_at=datetime.now(),
                                language="zh",
                                category="market",
                                tags=["caixin", "api"],
                            )
                        )
                        if len(articles) >= int(limit):
                            break

            if len(articles) < int(limit):
                home = self._session.get(
                    "https://www.caixinglobal.com/",
                    timeout=8,
                    headers={"Referer": "https://www.caixinglobal.com/"},
                )
                home.raise_for_status()
                text = str(home.text or "")
                link_pattern = re.compile(
                    r"<a[^>]+href=[\"'](?P<url>https?://www\.caixinglobal\.com/(?P<date>\d{4}-\d{2}-\d{2})/[^\"']+?\.html)[\"'][^>]*>(?P<title>.*?)</a>",
                    flags=re.IGNORECASE | re.DOTALL,
                )
                for match in link_pattern.finditer(text):
                    title = self._clean_text(match.group("title"))
                    if len(title) < 8:
                        continue
                    title_key = title.lower()
                    if title_key in seen_titles:
                        continue
                    if keyword_set:
                        t_low = title.lower()
                        if not any(k in t_low for k in keyword_set):
                            continue
                    seen_titles.add(title_key)

                    raw_date = str(match.group("date") or "")
                    try:
                        published_at = datetime.strptime(raw_date, "%Y-%m-%d")
                    except Exception:
                        published_at = datetime.now()
                    if not self._is_recent_enough(published_at, start_time):
                        continue

                    article_url = str(match.group("url") or "").strip()
                    articles.append(
                        NewsArticle(
                            id=self._generate_id(f"caixin_home_{title}"),
                            title=title,
                            content=title,
                            summary=title[:200],
                            source="caixin",
                            url=article_url,
                            published_at=self._normalize_datetime(published_at),
                            collected_at=datetime.now(),
                            language="en",
                            category="market",
                            tags=["caixin", "homepage"],
                        )
                    )
                    if len(articles) >= int(limit):
                        break

            self._update_source_health("caixin", success=bool(articles))

        except Exception as e:
            if self._strict_mode:
                raise
            log.debug("Caixin fetch degraded: %s", e)
            self._update_source_health("caixin", success=False)

        return articles[: int(limit)]

    def _fetch_csrc(
        self,
        keywords: list[str] | None,
        start_time: datetime,
        limit: int,
    ) -> list[NewsArticle]:
        """Fetch from CSRC (涓浗璇佺洃浼?."""
        articles = []

        # CSRC website scraping would require more complex handling
        # This is a placeholder
        log.debug("CSRC fetch requires specialized scraping")
        return articles

    def _fetch_reuters(
        self,
        keywords: list[str] | None,
        start_time: datetime,
        limit: int,
    ) -> list[NewsArticle]:
        """Fetch from Reuters."""
        articles = []
        search_term = keywords[0] if keywords else "China stock"
        url = "https://www.reuters.com/api/search"
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

                if not self._is_recent_enough(published_at, start_time):
                    continue

                article_id = self._generate_id(f"reuters_{title}")
                articles.append(NewsArticle(
                    id=article_id,
                    title=title,
                    content=content or title,
                    summary=title[:200],
                    source="reuters",
                    url=f"https://www.reuters.com{url}" if url and not url.startswith("http") else url,
                    published_at=self._normalize_datetime(published_at),
                    collected_at=datetime.now(),
                    language="en",
                    category="market",
                    tags=["reuters"],
                ))

            self._update_source_health("reuters", success=True)

        except Exception as e:
            if self._strict_mode:
                raise
            log.error(f"Reuters fetch failed: {e}")
            self._update_source_health("reuters", success=False)

        return articles

    def _fetch_bloomberg(
        self,
        keywords: list[str] | None,
        start_time: datetime,
        limit: int,
    ) -> list[NewsArticle]:
        """Fetch from Bloomberg."""
        # Bloomberg requires API subscription
        log.debug("Bloomberg fetch requires API subscription")
        return []

    def _fetch_yahoo_finance(
        self,
        keywords: list[str] | None,
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

                if not self._is_recent_enough(published_at, start_time):
                    continue

                article_id = self._generate_id(f"yahoo_{title}")
                articles.append(NewsArticle(
                    id=article_id,
                    title=title,
                    content=content or title,
                    summary=title[:200],
                    source="yahoo_finance",
                    url=url,
                    published_at=self._normalize_datetime(published_at),
                    collected_at=datetime.now(),
                    language="en",
                    category="market",
                    tags=["yahoo"],
                ))

            self._update_source_health("yahoo_finance", success=True)

        except Exception as e:
            if self._strict_mode:
                raise
            log.error(f"Yahoo Finance fetch failed: {e}")
            self._update_source_health("yahoo_finance", success=False)

        return articles

    def _fetch_rss_feed(
        self,
        *,
        source: str,
        feed_urls: list[str],
        keywords: list[str] | None,
        start_time: datetime,
        limit: int,
    ) -> list[NewsArticle]:
        """Fetch RSS/Atom feed entries and normalize into NewsArticle rows."""
        articles: list[NewsArticle] = []
        seen_ids: set[str] = set()
        keyword_set = [
            str(token).strip().lower()
            for token in (keywords or [])
            if str(token).strip()
        ]

        try:
            for feed_url in list(feed_urls):
                if len(articles) >= int(limit):
                    break
                resp = self._session.get(
                    feed_url,
                    timeout=10,
                    headers={"Accept": "application/rss+xml, application/xml;q=0.9, */*;q=0.8"},
                )
                resp.raise_for_status()
                payload = str(resp.text or "").strip()
                if not payload:
                    continue

                root = ET.fromstring(payload)
                entries = list(root.findall(".//item"))
                if not entries:
                    entries = list(root.findall(".//{http://www.w3.org/2005/Atom}entry"))

                for entry in entries:
                    title = self._clean_text(self._xml_text(entry, "title"))
                    if not title:
                        continue
                    summary = self._clean_text(
                        self._xml_text(entry, "description")
                        or self._xml_text(entry, "summary")
                        or self._xml_text(entry, "content")
                    )
                    article_url = self._xml_link(entry)
                    published_at = (
                        self._parse_datetime(
                            self._xml_text(entry, "pubDate")
                            or self._xml_text(entry, "published")
                            or self._xml_text(entry, "updated")
                        )
                        or datetime.now()
                    )
                    if not self._is_recent_enough(published_at, start_time):
                        continue
                    if keyword_set:
                        lookup_text = f"{title} {summary}".lower()
                        if not any(token in lookup_text for token in keyword_set):
                            continue

                    article_id = self._generate_id(f"{source}_{title}_{article_url}")
                    if article_id in seen_ids:
                        continue
                    seen_ids.add(article_id)
                    articles.append(
                        NewsArticle(
                            id=article_id,
                            title=title,
                            content=summary or title,
                            summary=(summary or title)[:200],
                            source=source,
                            url=article_url,
                            published_at=self._normalize_datetime(published_at),
                            collected_at=datetime.now(),
                            language="en",
                            category="market",
                            tags=[source, "rss"],
                        )
                    )
                    if len(articles) >= int(limit):
                        break

            self._update_source_health(source, success=bool(articles))
        except Exception as e:
            if self._strict_mode:
                raise
            log.error("%s fetch failed: %s", source, e)
            self._update_source_health(source, success=False)

        return articles[: int(limit)]

    def _fetch_marketwatch(
        self,
        keywords: list[str] | None,
        start_time: datetime,
        limit: int,
    ) -> list[NewsArticle]:
        """Fetch from MarketWatch RSS feeds."""
        return self._fetch_rss_feed(
            source="marketwatch",
            feed_urls=[
                "https://feeds.content.dowjones.io/public/rss/mw_topstories",
                "https://feeds.content.dowjones.io/public/rss/mw_markets",
            ],
            keywords=keywords,
            start_time=start_time,
            limit=limit,
        )

    def _fetch_cnbc(
        self,
        keywords: list[str] | None,
        start_time: datetime,
        limit: int,
    ) -> list[NewsArticle]:
        """Fetch from CNBC RSS feeds."""
        return self._fetch_rss_feed(
            source="cnbc",
            feed_urls=[
                "https://www.cnbc.com/id/100003114/device/rss/rss.html",
                "https://www.cnbc.com/id/10000664/device/rss/rss.html",
            ],
            keywords=keywords,
            start_time=start_time,
            limit=limit,
        )

    def _generate_id(self, text: str) -> str:
        """Generate unique ID from text."""
        return hashlib.md5(text.encode()).hexdigest()[:16]

    def _update_source_health(self, source: str, success: bool) -> None:
        """Update source health score."""
        now = time.time()
        current_health = float(self._source_health.get(source, 0.5) or 0.5)

        if success:
            new_health = min(1.0, current_health + 0.1)
            self._source_failures[source] = 0
        else:
            new_health = max(0.0, current_health - 0.2)
            self._source_failures[source] = int(self._source_failures.get(source, 0) or 0) + 1

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
        keywords: list[str] | None,
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

    def save_articles(self, articles: list[NewsArticle], filename: str | None = None) -> Path:
        """Save articles to JSON file."""
        if filename is None:
            filename = f"news_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        filepath = self.cache_dir / filename
        data = [article.to_dict() for article in articles]

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        log.info(f"Saved {len(articles)} articles to {filepath}")
        return filepath

    def _load_recent_cached_articles(
        self,
        *,
        start_time: datetime,
        limit: int,
        keywords: list[str] | None,
    ) -> list[NewsArticle]:
        """Load recent cached news files when online fetchers return no data."""
        keyword_set = [
            str(token).strip().lower()
            for token in (keywords or [])
            if str(token).strip()
        ]
        seen_ids: set[str] = set()
        out: list[NewsArticle] = []

        def _mtime(path: Path) -> float:
            try:
                return float(path.stat().st_mtime)
            except OSError:
                return 0.0

        cache_files = sorted(
            self.cache_dir.glob("news_*.json"),
            key=_mtime,
            reverse=True,
        )
        for path in cache_files[:12]:
            if len(out) >= int(limit):
                break
            try:
                cached_rows = self.load_articles(path)
            except Exception as exc:
                log.debug("Ignoring unreadable cache file %s: %s", path, exc)
                continue
            for article in cached_rows:
                if len(out) >= int(limit):
                    break
                published_at = getattr(article, "published_at", None)
                if not isinstance(published_at, datetime):
                    continue
                if not self._is_recent_enough(published_at, start_time):
                    continue
                title = str(getattr(article, "title", "") or "")
                content = str(getattr(article, "content", "") or "")
                if keyword_set:
                    lookup = f"{title} {content}".lower()
                    if not any(token in lookup for token in keyword_set):
                        continue
                aid = str(getattr(article, "id", "") or "").strip()
                if not aid:
                    aid = self._generate_id(
                        f"cache_{getattr(article, 'source', '')}_{title}_{getattr(article, 'url', '')}"
                    )
                    article.id = aid
                if aid in seen_ids:
                    continue
                seen_ids.add(aid)
                article.published_at = self._normalize_datetime(published_at)
                article.collected_at = self._normalize_datetime(
                    getattr(article, "collected_at", datetime.now())
                )
                out.append(article)
        return out[: int(limit)]

    def load_articles(self, filepath: Path) -> list[NewsArticle]:
        """Load articles from JSON file."""
        with open(filepath, encoding="utf-8") as f:
            data = json.load(f)

        return [NewsArticle.from_dict(item) for item in data]

    def close(self) -> None:
        """Close the HTTP session and release resources.
        
        FIX 2026-02-25: Added explicit session cleanup to prevent
        resource leaks. Call this method when the collector is no longer needed.
        """
        try:
            self._session.close()
        except Exception as e:
            log.debug(f"Error closing news collector session: {e}")

    def __del__(self) -> None:
        """Cleanup on garbage collection."""
        try:
            self.close()
        except Exception:
            pass


# Singleton instance
_collector: NewsCollector | None = None
# FIX #4: Lock for thread-safe singleton creation
import threading as _threading
_collector_lock = _threading.Lock()


def get_collector() -> NewsCollector:
    """Get or create news collector instance."""
    global _collector
    if _collector is None:
        with _collector_lock:
            if _collector is None:  # double-checked locking
                _collector = NewsCollector()
    return _collector


def reset_collector() -> None:
    """Reset collector instance."""
    global _collector
    with _collector_lock:
        _collector = None
