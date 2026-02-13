# data/news.py
import math
import re
import time
import json
import threading
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque

import numpy as np
import requests
import pandas as pd

from config.settings import CONFIG
from utils.logger import get_logger

log = get_logger(__name__)

_NEWS_CACHE_TTL: int = 300          # 5 minutes
_NEWS_BUFFER_SIZE: int = 200        # Rolling buffer max items
_DEDUP_PREFIX_LEN: int = 40         # Title prefix length for dedup
_FETCH_TIMEOUT: int = 5             # HTTP timeout for news fetchers
_SENTIMENT_NEUTRAL_BAND: float = 0.2  # |score| < this → neutral

# Positive keywords (Chinese financial)
POSITIVE_WORDS: Dict[str, float] = {
    "涨停": 2.0, "大涨": 1.8, "暴涨": 1.8, "创新高": 1.5, "突破": 1.3,
    "利好": 1.5, "重大利好": 2.0, "超预期": 1.5, "业绩大增": 1.8,
    "净利润增长": 1.3, "营收增长": 1.2, "盈利": 1.0, "扭亏": 1.5,
    "分红": 1.0, "回购": 1.2, "增持": 1.3, "大股东增持": 1.5,
    "机构买入": 1.3, "北向资金流入": 1.2, "外资增持": 1.2,
    "上涨": 0.8, "走高": 0.8, "反弹": 0.7, "回升": 0.7,
    "利率下调": 0.8, "降准": 1.0, "降息": 1.0, "宽松": 0.8,
    "刺激": 0.7, "支持": 0.6, "鼓励": 0.6, "扶持": 0.7,
    "中标": 1.0, "签约": 0.8, "合作": 0.6, "战略合作": 0.8,
    "新产品": 0.7, "技术突破": 1.0, "专利": 0.7, "创新": 0.6,
    "产能扩张": 0.8, "订单增长": 1.0, "市场份额提升": 0.9,
    "减税": 1.0, "补贴": 0.8, "政策支持": 1.0, "国家战略": 0.9,
    "改革": 0.5, "开放": 0.5, "自贸区": 0.7, "新基建": 0.8,
    "数字经济": 0.7, "碳中和": 0.6, "新能源": 0.6,
}

NEGATIVE_WORDS: Dict[str, float] = {
    "跌停": -2.0, "大跌": -1.8, "暴跌": -1.8, "崩盘": -2.0,
    "利空": -1.5, "重大利空": -2.0, "爆雷": -2.0, "违规": -1.5,
    "处罚": -1.5, "罚款": -1.3, "退市": -2.0, "ST": -1.5,
    "亏损": -1.3, "业绩下滑": -1.5, "净利润下降": -1.3,
    "减持": -1.3, "大股东减持": -1.5, "高管减持": -1.2,
    "质押": -0.8, "爆仓": -1.8, "违约": -1.5,
    "下跌": -0.8, "走低": -0.8, "回调": -0.5, "下探": -0.7,
    "加息": -0.8, "收紧": -0.8, "监管": -0.6, "审查": -0.7,
    "限制": -0.6, "禁止": -0.8, "制裁": -1.0, "贸易战": -1.0,
    "疫情": -0.7, "停产": -1.0, "停工": -0.8, "召回": -0.8,
    "诉讼": -0.7, "仲裁": -0.6, "调查": -0.7,
    "风险": -0.5, "警告": -0.6, "预警": -0.6, "泡沫": -0.8,
    "过热": -0.6, "通胀": -0.5, "滞涨": -0.7,
    "北向资金流出": -1.0, "外资减持": -1.0,
}

# Max times a single keyword is counted (prevents spam amplification)
_MAX_KEYWORD_COUNT: int = 3

def analyze_sentiment(text: str) -> Tuple[float, str]:
    """
    Weighted keyword sentiment scoring.

    - Counts each keyword up to _MAX_KEYWORD_COUNT times
    - Normalizes by total absolute weight contribution
    - Applies tanh to produce smooth [-1, 1] output
    - Returns (score, label) where label ∈ {positive, negative, neutral}
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
    stock_codes: List[str] = field(default_factory=list)
    category: str = ""  # policy, earnings, market, industry, company
    sentiment_score: float = 0.0  # -1.0 … +1.0
    sentiment_label: str = "neutral"  # positive, negative, neutral
    importance: float = 0.5  # 0.0 … 1.0
    keywords: List[str] = field(default_factory=list)

    def __post_init__(self):
        # Only auto-compute sentiment if not already set and text exists
        if self.sentiment_score == 0.0 and (self.title or self.content):
            combined = (self.title or "") + " " + (self.content or "")
            self.sentiment_score, self.sentiment_label = analyze_sentiment(
                combined.strip()
            )

    def age_minutes(self) -> float:
        return (datetime.now() - self.publish_time).total_seconds() / 60.0

    def is_relevant_to(self, stock_code: str) -> bool:
        return stock_code in self.stock_codes

    def to_dict(self) -> Dict:
        return {
            "title": self.title,
            "source": self.source,
            "time": self.publish_time.strftime("%Y-%m-%d %H:%M"),
            "sentiment": self.sentiment_score,
            "sentiment_label": self.sentiment_label,
            "importance": self.importance,
            "category": self.category,
            "codes": self.stock_codes,
        }

class _BaseNewsFetcher:
    """Shared session setup for news fetchers."""

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

class SinaNewsFetcher(_BaseNewsFetcher):
    """Fetch news from Sina Finance (works on China IP)."""

    def __init__(self):
        super().__init__(referer="https://finance.sina.com.cn/")

    def fetch_market_news(self, count: int = 20) -> List[NewsItem]:
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

            items: List[NewsItem] = []
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
    ) -> List[NewsItem]:
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

            items: List[NewsItem] = []
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
    ) -> List[NewsItem]:
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

            items: List[NewsItem] = []
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

    def fetch_policy_news(self, count: int = 15) -> List[NewsItem]:
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

            items: List[NewsItem] = []
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

    def fetch_market_news(self, count: int = 20) -> List[NewsItem]:
        """Fetch general financial news via Tencent."""
        try:
            url = "https://r.inews.qq.com/getSimpleNews"
            params = {
                "ids": "finance_hot",
                "num": str(min(count, 30)),
            }
            r = self._session.get(url, params=params, timeout=_FETCH_TIMEOUT)
            data = r.json()

            items: List[NewsItem] = []
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

_POLICY_KEYWORDS: Tuple[str, ...] = (
    "央行", "证监会", "财政部", "国务院",
    "政策", "监管", "改革", "法规",
)

class NewsAggregator:
    """
    Aggregates news from multiple sources with caching.
    Network-aware: uses different sources based on China/VPN.
    """

    def __init__(self):
        self._sina = SinaNewsFetcher()
        self._eastmoney = EastmoneyNewsFetcher()
        self._tencent = TencentNewsFetcher()

        self._cache: Dict[str, List[NewsItem]] = {}
        self._cache_time: Dict[str, float] = {}
        self._cache_ttl: int = _NEWS_CACHE_TTL
        self._lock = threading.RLock()

        # Rolling news buffer (last N items)
        self._all_news: deque = deque(maxlen=_NEWS_BUFFER_SIZE)
        self._source_health: Dict[str, Dict[str, object]] = {
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

    # -- market news ---------------------------------------------------------

    def get_market_news(
        self, count: int = 30, force_refresh: bool = False
    ) -> List[NewsItem]:
        """Get aggregated market news from all available sources."""
        cache_key = f"market_{count}"

        with self._lock:
            if not force_refresh and self._is_cache_valid(cache_key):
                return self._cache[cache_key]

        from core.network import get_network_env
        env = get_network_env()

        all_items: List[NewsItem] = []

        # Tencent always works (China or VPN)
        if env.tencent_ok:
            try:
                fetched = self._tencent.fetch_market_news(count)
                all_items.extend(fetched)
                self._record_source_result("tencent", True, len(fetched))
            except Exception as exc:
                self._record_source_result("tencent", False, error=str(exc))

        # China direct: use Sina + Eastmoney
        if env.is_china_direct:
            try:
                fetched = self._sina.fetch_market_news(count)
                all_items.extend(fetched)
                self._record_source_result("sina", True, len(fetched))
            except Exception as exc:
                self._record_source_result("sina", False, error=str(exc))

            if env.eastmoney_ok:
                try:
                    fetched = self._eastmoney.fetch_policy_news(count)
                    all_items.extend(fetched)
                    self._record_source_result(
                        "eastmoney_policy", True, len(fetched)
                    )
                except Exception as exc:
                    self._record_source_result(
                        "eastmoney_policy", False, error=str(exc)
                    )

        unique = self._deduplicate(all_items)
        unique.sort(key=lambda x: x.publish_time, reverse=True)
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
    ) -> List[NewsItem]:
        """Get news for a specific stock."""
        code6 = str(stock_code).zfill(6)
        cache_key = f"stock_{code6}_{count}"

        with self._lock:
            if not force_refresh and self._is_cache_valid(cache_key):
                return self._cache[cache_key]

        from core.network import get_network_env
        env = get_network_env()

        all_items: List[NewsItem] = []

        # China direct: full access
        if env.is_china_direct:
            try:
                fetched = self._sina.fetch_stock_news(code6, count)
                all_items.extend(fetched)
                self._record_source_result("sina", True, len(fetched))
            except Exception as exc:
                self._record_source_result("sina", False, error=str(exc))

            if env.eastmoney_ok:
                try:
                    fetched = self._eastmoney.fetch_stock_news(code6, count)
                    all_items.extend(fetched)
                    self._record_source_result(
                        "eastmoney_stock", True, len(fetched)
                    )
                except Exception as exc:
                    self._record_source_result(
                        "eastmoney_stock", False, error=str(exc)
                    )

        with self._lock:
            for item in self._all_news:
                if code6 in item.title or code6 in str(item.stock_codes):
                    all_items.append(item)

        unique = self._deduplicate(all_items)
        unique.sort(key=lambda x: x.publish_time, reverse=True)
        unique = unique[:count]

        for item in unique:
            if code6 not in item.stock_codes:
                item.stock_codes.append(code6)

        with self._lock:
            self._cache[cache_key] = unique
            self._cache_time[cache_key] = time.time()

        return unique

    # -- policy news ---------------------------------------------------------

    def get_policy_news(self, count: int = 10) -> List[NewsItem]:
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
        self, stock_code: Optional[str] = None
    ) -> Dict:
        """Get aggregated sentiment for stock or market."""
        news = (
            self.get_stock_news(stock_code)
            if stock_code
            else self.get_market_news()
        )

        if not news:
            return {
                "overall_sentiment": 0.0,
                "label": "neutral",
                "positive_count": 0,
                "negative_count": 0,
                "neutral_count": 0,
                "total": 0,
                "top_positive": [],
                "top_negative": [],
            }

        scores = [n.sentiment_score for n in news]
        overall = sum(scores) / len(scores)

        positive = [n for n in news if n.sentiment_label == "positive"]
        negative = [n for n in news if n.sentiment_label == "negative"]
        neutral = [n for n in news if n.sentiment_label == "neutral"]

        return {
            "overall_sentiment": round(overall, 3),
            "label": (
                "positive" if overall > 0.1
                else ("negative" if overall < -0.1 else "neutral")
            ),
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
        stock_code: Optional[str] = None,
        hours_lookback: int = 24,
    ) -> Dict[str, float]:
        """
        Get numerical features from news for AI model input.
        These can be appended to the technical feature vector.
        """
        news = (
            self.get_stock_news(stock_code, count=50)
            if stock_code
            else self.get_market_news(count=50)
        )

        cutoff = datetime.now() - timedelta(hours=hours_lookback)
        recent = [n for n in news if n.publish_time >= cutoff]

        if not recent:
            return {
                "news_sentiment_avg": 0.0,
                "news_sentiment_std": 0.0,
                "news_positive_ratio": 0.0,
                "news_negative_ratio": 0.0,
                "news_volume": 0.0,
                "news_importance_avg": 0.5,
                "news_recency_score": 0.0,
                "policy_sentiment": 0.0,
            }

        scores = [n.sentiment_score for n in recent]
        total = len(scores)
        positive = sum(1 for s in scores if s > 0.1)
        negative = sum(1 for s in scores if s < -0.1)

        # Recency-weighted sentiment (newer news matters more)
        recency_weights: List[float] = []
        for n in recent:
            age_hours = (
                (datetime.now() - n.publish_time).total_seconds() / 3600.0
            )
            weight = max(0.1, 1.0 - (age_hours / hours_lookback))
            recency_weights.append(weight)

        weight_sum = sum(recency_weights)
        weighted_sentiment = (
            sum(s * w for s, w in zip(scores, recency_weights)) / weight_sum
            if weight_sum > 0
            else 0.0
        )

        # Policy-specific sentiment
        policy_items = [n for n in recent if n.category == "policy"]
        policy_sentiment = (
            sum(n.sentiment_score for n in policy_items) / len(policy_items)
            if policy_items
            else 0.0
        )

        importances = [n.importance for n in recent]

        return {
            "news_sentiment_avg": round(float(np.mean(scores)), 4),
            "news_sentiment_std": (
                round(float(np.std(scores)), 4) if len(scores) > 1 else 0.0
            ),
            "news_positive_ratio": round(positive / total, 4),
            "news_negative_ratio": round(negative / total, 4),
            "news_volume": min(total / 20.0, 1.0),  # Normalized 0–1
            "news_importance_avg": round(float(np.mean(importances)), 4),
            "news_recency_score": round(weighted_sentiment, 4),
            "policy_sentiment": round(policy_sentiment, 4),
        }

    # -- deduplication -------------------------------------------------------

    @staticmethod
    def _deduplicate(items: List[NewsItem]) -> List[NewsItem]:
        """Remove duplicate news by title prefix similarity."""
        seen_titles: set = set()
        unique: List[NewsItem] = []
        for item in items:
            key = item.title[:_DEDUP_PREFIX_LEN] if item.title else ""
            if key and key not in seen_titles:
                seen_titles.add(key)
                unique.append(item)
        return unique

    # -- cache management ----------------------------------------------------

    def clear_cache(self) -> None:
        with self._lock:
            self._cache.clear()
            self._cache_time.clear()

    def get_source_health(self) -> Dict[str, Dict[str, object]]:
        """Institutional telemetry: fetch-source reliability and freshness."""
        with self._lock:
            out: Dict[str, Dict[str, object]] = {}
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
        self, stock_code: Optional[str] = None, hours_lookback: int = 24
    ) -> Dict[str, object]:
        """
        Institutional-grade unified news snapshot.
        Includes sentiment, model features, source health, and freshness stats.
        """
        news = (
            self.get_stock_news(stock_code, count=60)
            if stock_code
            else self.get_market_news(count=60)
        )
        summary = self.get_sentiment_summary(stock_code=stock_code)
        features = self.get_news_features(
            stock_code=stock_code, hours_lookback=hours_lookback
        )
        source_health = self.get_source_health()

        now = datetime.now()
        ages = [
            (now - n.publish_time).total_seconds()
            for n in news
            if getattr(n, "publish_time", None) is not None
        ]
        freshness = {
            "latest_age_seconds": round(float(min(ages)), 1) if ages else None,
            "median_age_seconds": round(float(np.median(ages)), 1) if ages else None,
            "items_with_timestamp": int(len(ages)),
        }

        source_counts: Dict[str, int] = {}
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

_aggregator: Optional[NewsAggregator] = None
_aggregator_lock = threading.Lock()

def get_news_aggregator() -> NewsAggregator:
    """Double-checked locking singleton for NewsAggregator."""
    global _aggregator
    if _aggregator is None:
        with _aggregator_lock:
            if _aggregator is None:
                _aggregator = NewsAggregator()
    return _aggregator
