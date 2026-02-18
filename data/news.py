# data/news.py
import copy
import json
import math
import os
import re
import ssl
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import requests

from utils.logger import get_logger

log = get_logger(__name__)

_NEWS_CACHE_TTL: int = 300          # 5 minutes
_NEWS_BUFFER_SIZE: int = 200        # Rolling buffer max items
_DEDUP_PREFIX_LEN: int = 40         # Title prefix length for dedup
_FETCH_TIMEOUT: int = 5             # HTTP timeout for news fetchers
_SENTIMENT_NEUTRAL_BAND: float = 0.2  # |score| below this is neutral

# Positive keywords (Chinese financial, Unicode-escaped to avoid encoding drift)
POSITIVE_WORDS: dict[str, float] = {
    "\u6da8\u505c": 2.0,
    "\u5927\u6da8": 1.8,
    "\u66b4\u6da8": 1.8,
    "\u521b\u65b0\u9ad8": 1.5,
    "\u7a81\u7834": 1.3,
    "\u5229\u597d": 1.5,
    "\u91cd\u5927\u5229\u597d": 2.0,
    "\u8d85\u9884\u671f": 1.5,
    "\u4e1a\u7ee9\u5927\u589e": 1.8,
    "\u51c0\u5229\u6da6\u589e\u957f": 1.3,
    "\u8425\u6536\u589e\u957f": 1.2,
    "\u76c8\u5229": 1.0,
    "\u626d\u4e8f": 1.5,
    "\u56de\u8d2d": 1.2,
    "\u589e\u6301": 1.3,
    "\u673a\u6784\u4e70\u5165": 1.3,
    "\u5317\u5411\u8d44\u91d1\u6d41\u5165": 1.2,
    "\u4e0a\u6da8": 0.8,
    "\u53cd\u5f39": 0.7,
    "\u964d\u606f": 1.0,
    "\u652f\u6301": 0.6,
    "\u4e2d\u6807": 1.0,
    "\u5408\u4f5c": 0.6,
    "\u6280\u672f\u7a81\u7834": 1.0,
    "\u8ba2\u5355\u589e\u957f": 1.0,
    "\u653f\u7b56\u652f\u6301": 1.0,
    "\u65b0\u80fd\u6e90": 0.6,
}

NEGATIVE_WORDS: dict[str, float] = {
    "\u8dcc\u505c": -2.0,
    "\u5927\u8dcc": -1.8,
    "\u66b4\u8dcc": -1.8,
    "\u5d29\u76d8": -2.0,
    "\u5229\u7a7a": -1.5,
    "\u91cd\u5927\u5229\u7a7a": -2.0,
    "\u7206\u96f7": -2.0,
    "\u8fdd\u89c4": -1.5,
    "\u5904\u7f5a": -1.5,
    "\u9000\u5e02": -2.0,
    "\u4e8f\u635f": -1.3,
    "\u4e1a\u7ee9\u4e0b\u6ed1": -1.5,
    "\u51c0\u5229\u6da6\u4e0b\u964d": -1.3,
    "\u51cf\u6301": -1.3,
    "\u8d28\u62bc": -0.8,
    "\u7206\u4ed3": -1.8,
    "\u8fdd\u7ea6": -1.5,
    "\u4e0b\u8dcc": -0.8,
    "\u56de\u8c03": -0.5,
    "\u52a0\u606f": -0.8,
    "\u76d1\u7ba1": -0.6,
    "\u9650\u5236": -0.6,
    "\u5236\u88c1": -1.0,
    "\u8d38\u6613\u6218": -1.0,
    "\u505c\u4ea7": -1.0,
    "\u8bc9\u8bbc": -0.7,
    "\u8c03\u67e5": -0.7,
    "\u98ce\u9669": -0.5,
    "\u8b66\u544a": -0.6,
    "\u6ce1\u6cab": -0.8,
    "\u901a\u80c0": -0.5,
    "\u5317\u5411\u8d44\u91d1\u6d41\u51fa": -1.0,
}

# Max times a single keyword is counted (prevents spam amplification)
_MAX_KEYWORD_COUNT: int = 3


def _safe_age_seconds_from_now(
    publish_time: datetime | None,
) -> float | None:
    """Age in seconds vs now for naive/aware datetimes without type errors."""
    if not isinstance(publish_time, datetime):
        return None
    try:
        if publish_time.tzinfo is not None:
            now_dt = datetime.now(tz=publish_time.tzinfo)
        else:
            now_dt = datetime.now()
        return float((now_dt - publish_time).total_seconds())
    except Exception:
        return None


def _safe_age_hours_from_now(
    publish_time: datetime | None,
) -> float | None:
    age_s = _safe_age_seconds_from_now(publish_time)
    if age_s is None:
        return None
    return float(age_s / 3600.0)


def _safe_float(value: object, default: float = 0.0) -> float:
    """Best-effort float conversion with finite guard."""
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not math.isfinite(out):
        return float(default)
    return out


def analyze_sentiment(text: str) -> tuple[float, str]:
    """
    Weighted keyword sentiment scoring.

    - Counts each keyword up to _MAX_KEYWORD_COUNT times
    - Normalizes by total absolute weight contribution
    - Applies tanh to produce smooth [-1, 1] output
    - Returns (score, label) where label in {positive, negative, neutral}
    """
    if not text:
        return 0.0, "neutral"

    raw_score = 0.0
    abs_contrib = 0.0

    for word, weight in POSITIVE_WORDS.items():
        c = min(text.count(word), _MAX_KEYWORD_COUNT)
        if c > 0:
            raw_score += weight * c
            abs_contrib += abs(weight) * c

    for word, weight in NEGATIVE_WORDS.items():
        c = min(text.count(word), _MAX_KEYWORD_COUNT)
        if c > 0:
            raw_score += weight * c
            abs_contrib += abs(weight) * c

    if abs_contrib < 1e-9:
        return 0.0, "neutral"

    base = raw_score / abs_contrib
    normalized = math.tanh(1.25 * base)

    if normalized >= _SENTIMENT_NEUTRAL_BAND:
        label = "positive"
    elif normalized <= -_SENTIMENT_NEUTRAL_BAND:
        label = "negative"
    else:
        label = "neutral"

    return round(float(normalized), 3), label


@dataclass
class NewsItem:
    """Single news article with auto-computed sentiment."""

    title: str
    content: str = ""
    source: str = ""
    url: str = ""
    publish_time: datetime = field(default_factory=datetime.now)
    stock_codes: list[str] = field(default_factory=list)
    category: str = ""  # policy, earnings, market, industry, company
    sentiment_score: float = 0.0  # -1.0 .. +1.0
    sentiment_label: str = "neutral"  # positive, negative, neutral
    importance: float = 0.5  # 0.0 .. 1.0
    keywords: list[str] = field(default_factory=list)

    def __post_init__(self):
        # Only auto-compute sentiment if not already set and text exists
        if self.sentiment_score == 0.0 and (self.title or self.content):
            combined = (self.title or "") + " " + (self.content or "")
            self.sentiment_score, self.sentiment_label = analyze_sentiment(
                combined.strip()
            )

    def age_minutes(self) -> float:
        age_s = _safe_age_seconds_from_now(self.publish_time)
        if age_s is None:
            return 0.0
        return max(0.0, float(age_s) / 60.0)

    def is_relevant_to(self, stock_code: str) -> bool:
        return stock_code in self.stock_codes

    def to_dict(self) -> dict:
        ts = self.publish_time if isinstance(self.publish_time, datetime) else None
        return {
            "title": self.title,
            "source": self.source,
            "time": ts.strftime("%Y-%m-%d %H:%M") if ts is not None else "",
            "sentiment": self.sentiment_score,
            "sentiment_label": self.sentiment_label,
            "importance": self.importance,
            "category": self.category,
            "codes": self.stock_codes,
        }


class _BaseNewsFetcher:
    """Shared session setup for news fetchers."""

    @staticmethod
    def _allow_insecure_tls() -> bool:
        raw = str(os.environ.get("TRADING_NEWS_ALLOW_INSECURE_TLS", "0"))
        return raw.strip().lower() in ("1", "true", "yes", "on")

    @staticmethod
    def _resolve_tls_verify() -> bool | str:
        """
        Resolve a usable CA bundle path for requests.

        Some environments can have a stale certifi path; in that case we first
        try system/default bundles. If no valid bundle path can be found,
        remain secure-by-default (verify=True) unless explicitly overridden.
        """
        candidates: list[str] = []
        try:
            import certifi

            p = str(certifi.where() or "").strip()
            if p:
                candidates.append(p)
        except Exception:
            pass

        try:
            from requests.utils import DEFAULT_CA_BUNDLE_PATH

            p = str(DEFAULT_CA_BUNDLE_PATH or "").strip()
            if p:
                candidates.append(p)
        except Exception:
            pass

        try:
            dflt = ssl.get_default_verify_paths()
            for p in (dflt.cafile, dflt.capath):
                s = str(p or "").strip()
                if s:
                    candidates.append(s)
        except Exception:
            pass

        seen: set[str] = set()
        for raw in candidates:
            if raw in seen:
                continue
            seen.add(raw)
            try:
                if Path(raw).exists():
                    return raw
            except OSError:
                continue

        if _BaseNewsFetcher._allow_insecure_tls():
            return False
        # Keep TLS verification enabled by default.
        return True

    def __init__(self, referer: str = ""):
        self._session = requests.Session()
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36"
            ),
        }
        if referer:
            headers["Referer"] = referer
        self._session.headers.update(headers)
        verify_cfg = self._resolve_tls_verify()
        self._session.verify = verify_cfg
        if verify_cfg is False:
            log.warning(
                "News TLS verification disabled by env "
                "(TRADING_NEWS_ALLOW_INSECURE_TLS=1)"
            )


class SinaNewsFetcher(_BaseNewsFetcher):
    """Fetch news from Sina Finance (works on China IP)."""

    def __init__(self):
        super().__init__(referer="https://finance.sina.com.cn/")

    def fetch_market_news(self, count: int = 20) -> list[NewsItem]:
        """Fetch general market news."""
        try:
            url = "https://feed.mix.sina.com.cn/api/roll/get"
            params = {
                "pageid": "153",
                "lid": "2516",
                "k": "",
                "num": str(min(count, 50)),
                "page": "1",
            }
            r = self._session.get(url, params=params, timeout=_FETCH_TIMEOUT)
            data = r.json()

            items: list[NewsItem] = []
            for article in data.get("result", {}).get("data", []):
                title = article.get("title", "").strip()
                if not title:
                    continue

                pub_time = datetime.now()
                ctime = article.get("ctime", "")
                if ctime:
                    try:
                        pub_time = datetime.fromtimestamp(int(ctime))
                    except (ValueError, OSError):
                        pass

                items.append(NewsItem(
                    title=title,
                    content=(
                        article.get("summary", "")
                        or article.get("intro", "")
                        or ""
                    ),
                    source="sina",
                    url=article.get("url", ""),
                    publish_time=pub_time,
                    category="market",
                ))

            log.debug(f"Sina: fetched {len(items)} market news")
            return items

        except Exception as exc:
            log.warning(f"Sina market news failed: {exc}")
            return []

    def fetch_stock_news(
        self, stock_code: str, count: int = 10
    ) -> list[NewsItem]:
        """Fetch news for a specific stock via Sina search."""
        try:
            code6 = str(stock_code).zfill(6)
            url = "https://search.sina.com.cn/news"
            params = {
                "q": code6,
                "c": "news",
                "from": "channel",
                "ie": "utf-8",
                "num": str(min(count, 20)),
            }
            r = self._session.get(url, params=params, timeout=_FETCH_TIMEOUT)

            items: list[NewsItem] = []
            titles = re.findall(r"<h2><a[^>]*>(.+?)</a></h2>", r.text)
            for raw_title in titles[:count]:
                title = re.sub(r"<[^>]+>", "", raw_title).strip()
                if title:
                    items.append(NewsItem(
                        title=title,
                        source="sina",
                        stock_codes=[code6],
                        category="company",
                    ))

            return items

        except Exception as exc:
            log.debug(f"Sina stock news failed for {stock_code}: {exc}")
            return []


class EastmoneyNewsFetcher(_BaseNewsFetcher):
    """Fetch news from Eastmoney (works on China IP only)."""

    def __init__(self):
        super().__init__(referer="https://www.eastmoney.com/")

    def fetch_stock_news(
        self, stock_code: str, count: int = 10
    ) -> list[NewsItem]:
        """Fetch stock-specific news and announcements."""
        try:
            code6 = str(stock_code).zfill(6)
            url = "https://search-api-web.eastmoney.com/search/jsonp"
            params = {
                "cb": "jQuery_callback",
                "param": json.dumps({
                    "uid": "",
                    "keyword": code6,
                    "type": ["cmsArticleWebOld"],
                    "client": "web",
                    "clientType": "web",
                    "clientVersion": "curr",
                    "param": {
                        "cmsArticleWebOld": {
                            "searchScope": "default",
                            "sort": "default",
                            "pageIndex": 1,
                            "pageSize": count,
                        }
                    },
                }),
            }

            r = self._session.get(url, params=params, timeout=_FETCH_TIMEOUT)
            text = r.text

            lparen = text.index("(")
            rparen = text.rindex(")")
            json_str = text[lparen + 1 : rparen]
            data = json.loads(json_str)

            items: list[NewsItem] = []
            articles = (
                data.get("result", {})
                .get("cmsArticleWebOld", {})
                .get("list", [])
            )

            for article in articles:
                title = re.sub(
                    r"<[^>]+>", "", article.get("title", "")
                ).strip()
                if not title:
                    continue

                pub_time = datetime.now()
                date_str = article.get("date", "")
                if date_str:
                    try:
                        pub_time = datetime.strptime(
                            date_str[:19], "%Y-%m-%d %H:%M:%S"
                        )
                    except ValueError:
                        pass

                content = re.sub(
                    r"<[^>]+>",
                    "",
                    article.get("content", "")
                    or article.get("mediaName", "")
                    or "",
                )

                items.append(NewsItem(
                    title=title,
                    content=content[:500],
                    source="eastmoney",
                    url=article.get("url", ""),
                    publish_time=pub_time,
                    stock_codes=[code6],
                    category="company",
                ))

            log.debug(f"Eastmoney: fetched {len(items)} news for {code6}")
            return items

        except Exception as exc:
            log.debug(f"Eastmoney news failed for {stock_code}: {exc}")
            return []

    def fetch_policy_news(self, count: int = 15) -> list[NewsItem]:
        """Fetch policy and regulatory news."""
        try:
            url = "https://np-listapi.eastmoney.com/comm/web/getNewsByColumns"
            params = {
                "columns": "CSRC,PBOC,MOF",
                "pageSize": str(min(count, 30)),
                "pageIndex": "1",
            }
            r = self._session.get(url, params=params, timeout=_FETCH_TIMEOUT)
            data = r.json()

            items: list[NewsItem] = []
            for article in data.get("data", {}).get("list", []):
                title = article.get("title", "").strip()
                if not title:
                    continue
                items.append(NewsItem(
                    title=title,
                    source="eastmoney_policy",
                    publish_time=datetime.now(),
                    category="policy",
                    importance=0.8,
                ))

            return items

        except Exception as exc:
            log.debug(f"Eastmoney policy news failed: {exc}")
            return []


class TencentNewsFetcher(_BaseNewsFetcher):
    """Fetch news from Tencent Finance (works from ANY IP)."""

    def __init__(self):
        super().__init__()

    def fetch_market_news(self, count: int = 20) -> list[NewsItem]:
        """Fetch general financial news via Tencent."""
        try:
            url = "https://r.inews.qq.com/getSimpleNews"
            params = {
                "ids": "finance_hot",
                "num": str(min(count, 30)),
            }
            r = self._session.get(url, params=params, timeout=_FETCH_TIMEOUT)
            data = r.json()

            items: list[NewsItem] = []
            for article in data.get("newslist", []):
                title = article.get("title", "").strip()
                if not title:
                    continue

                pub_time = datetime.now()
                ts = article.get("timestamp", "")
                if ts:
                    try:
                        pub_time = datetime.fromtimestamp(int(ts))
                    except (ValueError, OSError):
                        pass

                items.append(NewsItem(
                    title=title,
                    content=article.get("abstract", "") or "",
                    source="tencent",
                    url=article.get("url", ""),
                    publish_time=pub_time,
                    category="market",
                ))

            log.debug(f"Tencent: fetched {len(items)} market news")
            return items

        except Exception as exc:
            log.warning(f"Tencent market news failed: {exc}")
            return []


_POLICY_KEYWORDS: tuple[str, ...] = (
    "\u592e\u884c",
    "\u8bc1\u76d1\u4f1a",
    "\u8d22\u653f\u90e8",
    "\u56fd\u52a1\u9662",
    "\u653f\u7b56",
    "\u76d1\u7ba1",
    "\u6539\u9769",
    "\u6cd5\u89c4",
)


def _make_dedup_key(item: NewsItem) -> tuple[str, str]:
    """
    Create a stable dedup key from title prefix and publish_time.

    FIX Bug 1: Original code referenced item.published_at which does not
    exist on NewsItem. The correct attribute is item.publish_time.
    """
    title_part = str(item.title or "").strip()[:_DEDUP_PREFIX_LEN]
    if isinstance(item.publish_time, datetime):
        time_part = item.publish_time.strftime("%Y-%m-%d %H:%M")
    else:
        time_part = ""
    return (title_part, time_part)


class NewsAggregator:
    """
    Aggregates news from multiple sources with caching.
    Network-aware: uses different sources based on China/VPN.
    """

    def __init__(self):
        self._sina = SinaNewsFetcher()
        self._eastmoney = EastmoneyNewsFetcher()
        self._tencent = TencentNewsFetcher()

        self._cache: dict[str, list[NewsItem]] = {}
        self._cache_time: dict[str, float] = {}
        self._cache_ttl: int = _NEWS_CACHE_TTL
        self._lock = threading.RLock()

        # Rolling news buffer (last N items)
        self._all_news: deque[NewsItem] = deque(maxlen=_NEWS_BUFFER_SIZE)
        self._source_health: dict[str, dict[str, object]] = {
            "tencent": {
                "ok_calls": 0,
                "failed_calls": 0,
                "last_success_ts": 0.0,
                "last_error": "",
                "last_items": 0,
            },
            "sina": {
                "ok_calls": 0,
                "failed_calls": 0,
                "last_success_ts": 0.0,
                "last_error": "",
                "last_items": 0,
            },
            "eastmoney_policy": {
                "ok_calls": 0,
                "failed_calls": 0,
                "last_success_ts": 0.0,
                "last_error": "",
                "last_items": 0,
            },
            "eastmoney_stock": {
                "ok_calls": 0,
                "failed_calls": 0,
                "last_success_ts": 0.0,
                "last_error": "",
                "last_items": 0,
            },
        }

    # -- cache helpers -------------------------------------------------------

    def _is_cache_valid(self, key: str) -> bool:
        """Check if cached entry exists and is within TTL."""
        return (
            key in self._cache
            and (time.time() - self._cache_time.get(key, 0)) < self._cache_ttl
        )

    def _record_source_result(
        self, source: str, ok: bool, item_count: int = 0, error: str = ""
    ) -> None:
        with self._lock:
            state = self._source_health.get(source)
            if state is None:
                state = {
                    "ok_calls": 0,
                    "failed_calls": 0,
                    "last_success_ts": 0.0,
                    "last_error": "",
                    "last_items": 0,
                }
                self._source_health[source] = state

            if ok:
                state["ok_calls"] = int(state.get("ok_calls", 0)) + 1
                state["last_success_ts"] = float(time.time())
                state["last_error"] = ""
                state["last_items"] = int(item_count)
            else:
                state["failed_calls"] = int(state.get("failed_calls", 0)) + 1
                state["last_error"] = str(error)[:240]

    def _source_reliability_weight(self, source: str) -> float:
        """
        Reliability prior from rolling source health.
        Returns [0.5, 1.3] so weak sources are down-weighted, not removed.
        """
        with self._lock:
            state = self._source_health.get(source, {})
            ok_calls = int(state.get("ok_calls", 0))
            failed_calls = int(state.get("failed_calls", 0))
        total = ok_calls + failed_calls
        if total <= 0:
            return 1.0
        rate = ok_calls / float(total)
        return float(min(1.3, max(0.5, 0.6 + 0.9 * rate)))

    @staticmethod
    def _norm_code(value: str) -> str:
        digits = "".join(ch for ch in str(value or "") if ch.isdigit())
        return digits.zfill(6) if digits else ""

    def _item_mentions_code(self, item: NewsItem, code6: str) -> bool:
        code6 = self._norm_code(code6)
        if not code6:
            return False
        title = str(getattr(item, "title", "") or "")
        content = str(getattr(item, "content", "") or "")
        if code6 in title or code6 in content:
            return True
        for v in list(getattr(item, "stock_codes", []) or []):
            if self._norm_code(str(v)) == code6:
                return True
        return False

    @staticmethod
    def _publish_recency_key(item: NewsItem) -> float:
        """
        Stable recency sort key for mixed naive/aware publish_time values.
        Smaller means newer. Missing timestamps are pushed to the end.
        """
        age_s = _safe_age_seconds_from_now(getattr(item, "publish_time", None))
        if age_s is None:
            return float("inf")
        return max(0.0, float(age_s))

    # -- market news ---------------------------------------------------------

    def get_market_news(
        self, count: int = 30, force_refresh: bool = False
    ) -> list[NewsItem]:
        """Get aggregated market news from all available sources."""
        cache_key = f"market_{count}"

        with self._lock:
            if not force_refresh and self._is_cache_valid(cache_key):
                return list(self._cache[cache_key])

        from core.network import get_network_env
        env = get_network_env()

        all_items: list[NewsItem] = []
        should_try_tencent = bool(env.tencent_ok) or (not env.is_china_direct)
        should_try_sina = bool(env.is_china_direct) or bool(env.eastmoney_ok)
        should_try_eastmoney_policy = bool(env.eastmoney_ok) or bool(env.is_china_direct)

        # If detector is stale/incorrect, still do one best-effort Tencent fetch.
        if not (should_try_tencent or should_try_sina or should_try_eastmoney_policy):
            should_try_tencent = True

        if should_try_tencent:
            try:
                fetched = self._tencent.fetch_market_news(count)
                all_items.extend(fetched)
                self._record_source_result("tencent", True, len(fetched))
            except Exception as exc:
                self._record_source_result("tencent", False, error=str(exc))

        if should_try_sina:
            try:
                fetched = self._sina.fetch_market_news(count)
                all_items.extend(fetched)
                self._record_source_result("sina", True, len(fetched))
            except Exception as exc:
                self._record_source_result("sina", False, error=str(exc))

        if should_try_eastmoney_policy:
            try:
                fetched = self._eastmoney.fetch_policy_news(count)
                all_items.extend(fetched)
                self._record_source_result("eastmoney_policy", True, len(fetched))
            except Exception as exc:
                self._record_source_result("eastmoney_policy", False, error=str(exc))

        unique = self._deduplicate(all_items)
        unique.sort(key=self._publish_recency_key)
        unique = unique[:count]

        # Institutional fail-safe: stale cache fallback if all providers fail.
        if not unique:
            with self._lock:
                stale = list(self._cache.get(cache_key, []))
            if stale:
                log.warning(
                    "News providers returned no items; serving stale cache "
                    f"({len(stale)} items)"
                )
                return stale[:count]

        with self._lock:
            self._cache[cache_key] = unique
            self._cache_time[cache_key] = time.time()
            for item in unique:
                self._all_news.appendleft(item)

        log.info(f"Aggregated {len(unique)} market news items")
        return unique

    # -- stock news ----------------------------------------------------------

    def get_stock_news(
        self,
        stock_code: str,
        count: int = 15,
        force_refresh: bool = False,
    ) -> list[NewsItem]:
        """Get news for a specific stock."""
        code6 = self._norm_code(stock_code)
        if not code6:
            return []
        cache_key = f"stock_{code6}_{count}"

        with self._lock:
            if not force_refresh and self._is_cache_valid(cache_key):
                return list(self._cache[cache_key])

        from core.network import get_network_env
        env = get_network_env()

        all_items: list[NewsItem] = []
        context_items: list[NewsItem] = []

        should_try_sina = bool(env.is_china_direct) or bool(env.tencent_ok) or bool(env.eastmoney_ok)
        should_try_eastmoney = bool(env.eastmoney_ok) or bool(env.is_china_direct)

        if should_try_sina:
            try:
                fetched = self._sina.fetch_stock_news(code6, count)
                all_items.extend(fetched)
                self._record_source_result("sina", True, len(fetched))
            except Exception as exc:
                self._record_source_result("sina", False, error=str(exc))

        if should_try_eastmoney:
            try:
                fetched = self._eastmoney.fetch_stock_news(code6, count)
                all_items.extend(fetched)
                self._record_source_result("eastmoney_stock", True, len(fetched))
            except Exception as exc:
                self._record_source_result("eastmoney_stock", False, error=str(exc))

        # FIX Bug 4: Access _all_news under lock for thread safety
        with self._lock:
            for item in self._all_news:
                if self._item_mentions_code(item, code6):
                    all_items.append(item)

        # Blend in market-wide context so downstream views are not limited
        # to symbol-tagged headlines only.
        # FIX Bug 3: Don't propagate force_refresh to avoid unnecessary
        # cache invalidation of unrelated market cache.
        market_pool = self.get_market_news(
            count=max(60, int(count) * 4),
            force_refresh=False,
        )
        for item in market_pool:
            if self._item_mentions_code(item, code6):
                all_items.append(item)
            else:
                context_items.append(item)

        unique_all = self._deduplicate(all_items + context_items)
        unique_all.sort(key=self._publish_recency_key)

        direct = [
            item for item in unique_all
            if self._item_mentions_code(item, code6)
        ]
        context = [
            item for item in unique_all
            if not self._item_mentions_code(item, code6)
        ]

        target_count = max(1, int(count))

        # Keep stock-relevant items first; only fallback to context if none exist.
        selected: list[NewsItem] = []
        candidate_pool = direct if direct else context
        for item in candidate_pool:
            selected.append(item)
            if len(selected) >= target_count:
                break

        selected.sort(key=self._publish_recency_key)
        selected = selected[:target_count]

        # Same stale-cache guard as market path.
        if not selected:
            with self._lock:
                stale = list(self._cache.get(cache_key, []))
            if stale:
                log.warning(
                    "Stock/news blend returned no items; serving stale cache "
                    f"for {code6} ({len(stale)} items)"
                )
                return stale[:target_count]

        # Avoid mutating shared market/news objects in-place.
        unique: list[NewsItem] = []
        for item in selected:
            cloned = copy.deepcopy(item)
            if code6 not in cloned.stock_codes:
                cloned.stock_codes.append(code6)
            unique.append(cloned)

        with self._lock:
            self._cache[cache_key] = unique
            self._cache_time[cache_key] = time.time()

        return unique

    # -- policy news ---------------------------------------------------------

    def get_policy_news(self, count: int = 10) -> list[NewsItem]:
        """Get policy/regulatory news only."""
        all_news = self.get_market_news(count=50)
        policy = [
            n for n in all_news
            if n.category == "policy"
            or any(kw in n.title for kw in _POLICY_KEYWORDS)
        ]
        return policy[:count]

    # -- sentiment summary ---------------------------------------------------

    def get_sentiment_summary(
        self,
        stock_code: str | None = None,
        _news: list[NewsItem] | None = None,
    ) -> dict:
        """
        Get aggregated sentiment for stock or market.

        FIX Bug 2: Accept optional _news parameter to break recursive call
        chain. When called from get_news_features(), the already-fetched
        news list is passed directly instead of re-fetching.
        """
        if _news is not None:
            news = _news
        elif stock_code:
            news = self.get_stock_news(stock_code)
        else:
            news = self.get_market_news()

        if not news:
            return {
                "overall_sentiment": 0.0,
                "simple_sentiment": 0.0,
                "importance_weighted_sentiment": 0.0,
                "label": "neutral",
                "confidence": 0.0,
                "weighted": True,
                "fusion_version": "2.1",
                "recency_half_life_hours": 18.0,
                "source_diversity": 0.0,
                "source_entropy": 0.0,
                "source_concentration_hhi": 0.0,
                "disagreement_index": 0.0,
                "novelty_score": 0.0,
                "average_age_hours": 0.0,
                "sentiment_momentum_6h": 0.0,
                "source_mix": {},
                "source_weight_mix": {},
                "source_contributions": {},
                "positive_count": 0,
                "negative_count": 0,
                "neutral_count": 0,
                "total": 0,
                "top_positive": [],
                "top_negative": [],
            }

        scores = [_safe_float(getattr(n, "sentiment_score", 0.0), 0.0) for n in news]
        simple_avg = (sum(scores) / len(scores)) if scores else 0.0

        weighted_total = 0.0
        weight_sum = 0.0
        source_weighted: dict[str, float] = {}
        source_weight_sum: dict[str, float] = {}
        source_scores: dict[str, list[float]] = {}
        source_counts: dict[str, int] = {}
        source_weight_mass: dict[str, float] = {}
        headline_seen: dict[tuple[str, str], int] = {}
        recency_half_life_hours = 18.0
        novelty_values: list[float] = []
        weighted_age_hours = 0.0
        importance_weighted_total = 0.0
        importance_weight_sum = 0.0
        scores_recent_6h: list[float] = []
        scores_older_6h: list[float] = []

        for n in news:
            score = _safe_float(getattr(n, "sentiment_score", 0.0), 0.0)
            importance = _safe_float(getattr(n, "importance", 0.5), 0.5)
            age_h_raw = _safe_age_hours_from_now(getattr(n, "publish_time", None))
            age_h = max(0.0, float(age_h_raw)) if age_h_raw is not None else 0.0
            recency_w = float(0.5 ** (age_h / recency_half_life_hours))
            src = str(getattr(n, "source", "") or "").strip().lower()
            src_w = self._source_reliability_weight(src)
            title_norm = str(getattr(n, "title", "") or "").strip().lower()
            novelty_key = (src or "unknown", title_norm[:_DEDUP_PREFIX_LEN])
            repeat = int(headline_seen.get(novelty_key, 0))
            headline_seen[novelty_key] = repeat + 1
            novelty_w = 1.0 / (1.0 + 0.35 * repeat)
            novelty_values.append(float(novelty_w))

            w = max(
                0.03,
                recency_w * src_w * max(0.1, importance) * novelty_w,
            )
            weighted_age_hours += age_h * w
            importance_w = max(0.1, importance)
            importance_weighted_total += score * importance_w
            importance_weight_sum += importance_w
            if age_h <= 6.0:
                scores_recent_6h.append(score)
            else:
                scores_older_6h.append(score)

            weighted_total += score * w
            weight_sum += w
            source_weighted[src] = float(source_weighted.get(src, 0.0) + (score * w))
            source_weight_sum[src] = float(source_weight_sum.get(src, 0.0) + w)
            source_weight_mass[src] = float(source_weight_mass.get(src, 0.0) + w)
            source_scores.setdefault(src, []).append(score)
            source_counts[src] = int(source_counts.get(src, 0) + 1)

        overall = (weighted_total / weight_sum) if weight_sum > 0 else simple_avg
        importance_weighted = (
            importance_weighted_total / importance_weight_sum
            if importance_weight_sum > 0
            else overall
        )

        positive = [n for n in news if n.sentiment_label == "positive"]
        negative = [n for n in news if n.sentiment_label == "negative"]
        neutral = [n for n in news if n.sentiment_label == "neutral"]

        source_diversity = float(len(source_scores) / max(1, len(news)))
        source_probs = [
            float(cnt) / max(1.0, float(len(news)))
            for cnt in source_counts.values()
            if cnt > 0
        ]
        entropy_raw = -sum((p * math.log(p)) for p in source_probs if p > 0.0)
        entropy_norm = (
            entropy_raw / math.log(len(source_probs))
            if len(source_probs) > 1
            else 0.0
        )
        source_concentration_hhi = sum((p * p) for p in source_probs)
        centroid_scores = [
            float(np.mean(vals))
            for vals in source_scores.values()
            if vals
        ]
        disagreement = (
            float(np.std(np.asarray(centroid_scores, dtype=float)))
            if len(centroid_scores) > 1
            else 0.0
        )
        novelty_score = (
            float(np.mean(novelty_values))
            if novelty_values
            else 0.0
        )
        avg_age_hours = (
            weighted_age_hours / weight_sum
            if weight_sum > 0
            else 0.0
        )
        if scores_recent_6h and scores_older_6h:
            momentum_6h = float(np.mean(scores_recent_6h) - np.mean(scores_older_6h))
        elif scores_recent_6h:
            momentum_6h = float(np.mean(scores_recent_6h) - overall)
        else:
            momentum_6h = 0.0

        coverage = min(1.0, len(news) / 30.0)
        strength = min(1.0, abs(float(overall)))
        source_coverage = min(1.0, len(source_scores) / 4.0)
        diversity_quality = min(1.0, (0.6 * source_coverage) + (0.4 * entropy_norm))

        confidence = (
            (0.40 * coverage)
            + (0.30 * strength)
            + (0.20 * diversity_quality)
            + (0.10 * novelty_score)
        )
        confidence *= max(0.0, 1.0 - (0.35 * min(1.0, disagreement * 2.5)))
        confidence = min(
            1.0,
            max(
                0.0,
                confidence,
            ),
        )

        source_contributions = {
            src: round(
                float(source_weighted.get(src, 0.0))
                / max(1e-9, float(source_weight_sum.get(src, 0.0))),
                4,
            )
            for src in sorted(source_weighted.keys())
        }
        source_mix = {
            src: round(float(cnt) / max(1.0, float(len(news))), 4)
            for src, cnt in sorted(
                source_counts.items(), key=lambda kv: kv[1], reverse=True
            )
        }
        source_weight_mix = {
            src: round(float(w) / max(1e-9, float(weight_sum)), 4)
            for src, w in sorted(
                source_weight_mass.items(), key=lambda kv: kv[1], reverse=True
            )
        }

        return {
            "overall_sentiment": round(overall, 3),
            "simple_sentiment": round(simple_avg, 3),
            "importance_weighted_sentiment": round(float(importance_weighted), 3),
            "label": (
                "positive" if overall > 0.1
                else ("negative" if overall < -0.1 else "neutral")
            ),
            "confidence": round(float(confidence), 3),
            "weighted": True,
            "fusion_version": "2.1",
            "recency_half_life_hours": recency_half_life_hours,
            "source_diversity": round(float(source_diversity), 3),
            "source_entropy": round(float(entropy_norm), 3),
            "source_concentration_hhi": round(float(source_concentration_hhi), 3),
            "disagreement_index": round(float(disagreement), 3),
            "novelty_score": round(float(novelty_score), 3),
            "average_age_hours": round(float(avg_age_hours), 3),
            "sentiment_momentum_6h": round(float(momentum_6h), 3),
            "source_mix": source_mix,
            "source_weight_mix": source_weight_mix,
            "source_contributions": source_contributions,
            "positive_count": len(positive),
            "negative_count": len(negative),
            "neutral_count": len(neutral),
            "total": len(news),
            "top_positive": [
                n.to_dict()
                for n in sorted(
                    positive, key=lambda x: x.sentiment_score, reverse=True
                )[:3]
            ],
            "top_negative": [
                n.to_dict()
                for n in sorted(negative, key=lambda x: x.sentiment_score)[:3]
            ],
        }

    # -- numerical features for AI model -------------------------------------

    def get_news_features(
        self,
        stock_code: str | None = None,
        hours_lookback: int = 24,
    ) -> dict[str, float]:
        """
        Get numerical features from news for AI model input.
        These can be appended to the technical feature vector.

        FIX Bug 2: Fetch news once and pass to get_sentiment_summary
        to avoid recursive re-fetching.
        """
        news = (
            self.get_stock_news(stock_code, count=50)
            if stock_code
            else self.get_market_news(count=50)
        )

        try:
            lookback_h_raw = float(hours_lookback)
        except (TypeError, ValueError):
            lookback_h_raw = 24.0
        lookback_h = lookback_h_raw if lookback_h_raw > 0.0 else 1.0

        # FIX Bug 7: Store age per index instead of id() which can be
        # unreliable after list slicing/copying
        recent: list[NewsItem] = []
        recent_age_hours: list[float] = []
        for n in news:
            age_h = _safe_age_hours_from_now(getattr(n, "publish_time", None))
            if age_h is None:
                continue
            age_h = max(0.0, float(age_h))
            if age_h <= lookback_h:
                recent.append(n)
                recent_age_hours.append(age_h)

        if not recent:
            return {
                "news_sentiment_avg": 0.0,
                "news_sentiment_std": 0.0,
                "news_weighted_sentiment": 0.0,
                "news_sentiment_disagreement": 0.0,
                "news_positive_ratio": 0.0,
                "news_negative_ratio": 0.0,
                "news_volume": 0.0,
                "news_importance_avg": 0.5,
                "news_recency_score": 0.0,
                "news_source_diversity": 0.0,
                "news_sentiment_confidence": 0.0,
                "news_source_entropy": 0.0,
                "news_source_concentration_hhi": 0.0,
                "news_novelty_score": 0.0,
                "news_recent_momentum": 0.0,
                "news_importance_weighted_sentiment": 0.0,
                "news_weighted_vs_simple_gap": 0.0,
                "news_average_age_hours": 0.0,
                "news_disagreement_penalty": 1.0,
                "policy_sentiment": 0.0,
            }

        scores = [_safe_float(getattr(n, "sentiment_score", 0.0), 0.0) for n in recent]
        total = len(scores)
        positive = sum(1 for s in scores if s > 0.1)
        negative = sum(1 for s in scores if s < -0.1)

        # Recency-weighted sentiment (newer news matters more)
        recency_weights: list[float] = []
        for age_hours in recent_age_hours:
            weight = max(0.1, 1.0 - (age_hours / lookback_h))
            recency_weights.append(weight)

        weight_sum = sum(recency_weights)
        weighted_sentiment = (
            sum(s * w for s, w in zip(scores, recency_weights, strict=False))
            / weight_sum
            if weight_sum > 0
            else 0.0
        )

        source_groups: dict[str, list[float]] = {}
        fused_w_sum = 0.0
        fused_s_sum = 0.0
        for idx, n in enumerate(recent):
            src = str(getattr(n, "source", "") or "").strip().lower()
            source_groups.setdefault(src, []).append(
                _safe_float(getattr(n, "sentiment_score", 0.0), 0.0)
            )
            src_w = self._source_reliability_weight(src)
            age_h = recent_age_hours[idx]
            rec_w = float(0.5 ** (age_h / 18.0))
            imp_w = max(0.1, _safe_float(getattr(n, "importance", 0.5), 0.5))
            w = max(0.03, src_w * rec_w * imp_w)
            fused_w_sum += w
            fused_s_sum += _safe_float(getattr(n, "sentiment_score", 0.0), 0.0) * w

        fused_sentiment = fused_s_sum / fused_w_sum if fused_w_sum > 0 else weighted_sentiment
        source_disagreement = (
            float(np.std([float(np.mean(v)) for v in source_groups.values() if v]))
            if len(source_groups) > 1
            else 0.0
        )
        source_diversity = float(len(source_groups) / max(1, len(recent)))

        # Policy-specific sentiment
        policy_items = [n for n in recent if n.category == "policy"]
        policy_sentiment = (
            sum(
                _safe_float(getattr(n, "sentiment_score", 0.0), 0.0)
                for n in policy_items
            )
            / len(policy_items)
            if policy_items
            else 0.0
        )

        importances = [_safe_float(getattr(n, "importance", 0.5), 0.5) for n in recent]

        # FIX Bug 2: Pass already-fetched news to avoid recursive calls
        summary = self.get_sentiment_summary(stock_code=stock_code, _news=news)
        summary_overall = float(summary.get("overall_sentiment", 0.0) or 0.0)
        summary_simple = float(summary.get("simple_sentiment", 0.0) or 0.0)
        summary_disagreement = float(summary.get("disagreement_index", 0.0) or 0.0)
        disagreement_penalty = max(0.0, 1.0 - min(1.0, summary_disagreement * 2.0))

        return {
            "news_sentiment_avg": round(float(np.mean(scores)), 4),
            "news_sentiment_std": (
                round(float(np.std(scores)), 4) if len(scores) > 1 else 0.0
            ),
            "news_weighted_sentiment": round(float(fused_sentiment), 4),
            "news_sentiment_disagreement": round(float(source_disagreement), 4),
            "news_positive_ratio": round(positive / total, 4),
            "news_negative_ratio": round(negative / total, 4),
            "news_volume": min(total / 20.0, 1.0),  # Normalized 0-1
            "news_importance_avg": round(float(np.mean(importances)), 4),
            "news_recency_score": round(weighted_sentiment, 4),
            "news_source_diversity": round(float(source_diversity), 4),
            "news_sentiment_confidence": round(float(summary.get("confidence", 0.0) or 0.0), 4),
            "news_source_entropy": round(float(summary.get("source_entropy", 0.0) or 0.0), 4),
            "news_source_concentration_hhi": round(float(summary.get("source_concentration_hhi", 0.0) or 0.0), 4),
            "news_novelty_score": round(float(summary.get("novelty_score", 0.0) or 0.0), 4),
            "news_recent_momentum": round(float(summary.get("sentiment_momentum_6h", 0.0) or 0.0), 4),
            "news_importance_weighted_sentiment": round(float(summary.get("importance_weighted_sentiment", 0.0) or 0.0), 4),
            "news_weighted_vs_simple_gap": round(float(summary_overall - summary_simple), 4),
            "news_average_age_hours": round(float(summary.get("average_age_hours", 0.0) or 0.0), 4),
            "news_disagreement_penalty": round(float(disagreement_penalty), 4),
            "policy_sentiment": round(policy_sentiment, 4),
        }

    # -- deduplication -------------------------------------------------------

    @staticmethod
    def _deduplicate(items: list[NewsItem]) -> list[NewsItem]:
        """Remove duplicate news by title prefix similarity."""
        seen_titles: set[str] = set()
        unique: list[NewsItem] = []
        for item in items:
            # FIX Bug 5: Skip items with empty/whitespace-only titles
            # instead of adding empty string to seen set
            title = (item.title or "").strip()
            if not title:
                continue
            key = title[:_DEDUP_PREFIX_LEN]
            if key not in seen_titles:
                seen_titles.add(key)
                unique.append(item)
        return unique

    # -- cache management ----------------------------------------------------

    def clear_cache(self) -> None:
        with self._lock:
            self._cache.clear()
            self._cache_time.clear()

    def get_source_health(self) -> dict[str, dict[str, object]]:
        """Institutional telemetry: fetch-source reliability and freshness."""
        with self._lock:
            out: dict[str, dict[str, object]] = {}
            now_ts = float(time.time())
            for src, state in self._source_health.items():
                ok_calls = int(state.get("ok_calls", 0))
                failed_calls = int(state.get("failed_calls", 0))
                total = ok_calls + failed_calls
                success_rate = (ok_calls / total) if total > 0 else 1.0
                last_success_ts = float(state.get("last_success_ts", 0.0) or 0.0)
                age_s = (now_ts - last_success_ts) if last_success_ts > 0 else float("inf")
                out[src] = {
                    "ok_calls": ok_calls,
                    "failed_calls": failed_calls,
                    "success_rate": round(float(success_rate), 4),
                    "last_success_age_seconds": (
                        round(float(age_s), 1) if math.isfinite(age_s) else None
                    ),
                    "last_items": int(state.get("last_items", 0)),
                    "last_error": str(state.get("last_error", "")),
                }
            return out

    def get_institutional_snapshot(
        self, stock_code: str | None = None, hours_lookback: int = 24
    ) -> dict[str, object]:
        """
        Institutional-grade unified news snapshot.
        Includes sentiment, model features, source health, and freshness stats.
        """
        news = (
            self.get_stock_news(stock_code, count=60)
            if stock_code
            else self.get_market_news(count=60)
        )
        # FIX Bug 2: Pass fetched news to avoid redundant calls
        summary = self.get_sentiment_summary(stock_code=stock_code, _news=news)
        features = self.get_news_features(
            stock_code=stock_code, hours_lookback=hours_lookback
        )
        source_health = self.get_source_health()

        now = datetime.now()
        ages = [
            max(0.0, float(age_s))
            for age_s in (
                _safe_age_seconds_from_now(getattr(n, "publish_time", None))
                for n in news
            )
            if age_s is not None
        ]
        freshness = {
            "latest_age_seconds": round(float(min(ages)), 1) if ages else None,
            "median_age_seconds": round(float(np.median(ages)), 1) if ages else None,
            "items_with_timestamp": int(len(ages)),
        }

        source_counts: dict[str, int] = {}
        for n in news:
            src = str(getattr(n, "source", "") or "unknown").strip().lower()
            source_counts[src] = source_counts.get(src, 0) + 1
        total = max(len(news), 1)
        source_mix = {
            src: round(cnt / total, 4) for src, cnt in sorted(
                source_counts.items(), key=lambda kv: kv[1], reverse=True
            )
        }

        return {
            "scope": str(stock_code or "market"),
            "timestamp": now.isoformat(),
            "news_count": len(news),
            "source_mix": source_mix,
            "source_health": source_health,
            "freshness": freshness,
            "sentiment": summary,
            "features": features,
        }


# Thread-safe singleton

_aggregator: NewsAggregator | None = None
_aggregator_lock = threading.Lock()


def get_news_aggregator() -> NewsAggregator:
    """Double-checked locking singleton for NewsAggregator."""
    global _aggregator
    if _aggregator is None:
        with _aggregator_lock:
            if _aggregator is None:
                _aggregator = NewsAggregator()
    return _aggregator
