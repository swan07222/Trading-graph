# data/news.py
"""
Chinese Financial News & Policy Fetcher

Sources (all domestic, work WITHOUT VPN):
1. Sina Finance - Real-time news feed
2. Eastmoney - Stock-specific news & announcements  
3. Tencent Finance - Market news
4. CCTV/Xinhua - Policy announcements

Features:
- Auto-detects network (China direct vs VPN)
- Sentiment analysis using keyword scoring
- Caches news to avoid repeated fetches
- Thread-safe for real-time updates
"""
import re
import time
import json
import threading
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque

import requests
import pandas as pd

from config.settings import CONFIG
from utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class NewsItem:
    """Single news article"""
    title: str
    content: str = ""
    source: str = ""
    url: str = ""
    publish_time: datetime = field(default_factory=datetime.now)
    stock_codes: List[str] = field(default_factory=list)
    category: str = ""  # policy, earnings, market, industry, company
    sentiment_score: float = 0.0  # -1.0 (very negative) to +1.0 (very positive)
    sentiment_label: str = "neutral"  # positive, negative, neutral
    importance: float = 0.5  # 0.0 to 1.0
    keywords: List[str] = field(default_factory=list)

    def __post_init__(self):
        if self.sentiment_score == 0.0 and self.title:
            self.sentiment_score, self.sentiment_label = analyze_sentiment(self.title + " " + self.content)

    def age_minutes(self) -> float:
        return (datetime.now() - self.publish_time).total_seconds() / 60.0

    def is_relevant_to(self, stock_code: str) -> bool:
        if stock_code in self.stock_codes:
            return True
        return False

    def to_dict(self) -> Dict:
        return {
            'title': self.title,
            'source': self.source,
            'time': self.publish_time.strftime("%Y-%m-%d %H:%M"),
            'sentiment': self.sentiment_score,
            'sentiment_label': self.sentiment_label,
            'importance': self.importance,
            'category': self.category,
            'codes': self.stock_codes,
        }


# =============================================================================
# SENTIMENT ANALYSIS (Keyword-based, no external dependencies)
# =============================================================================

# Positive keywords (Chinese financial)
POSITIVE_WORDS = {
    # Strong positive
    "涨停": 2.0, "大涨": 1.8, "暴涨": 1.8, "创新高": 1.5, "突破": 1.3,
    "利好": 1.5, "重大利好": 2.0, "超预期": 1.5, "业绩大增": 1.8,
    "净利润增长": 1.3, "营收增长": 1.2, "盈利": 1.0, "扭亏": 1.5,
    "分红": 1.0, "回购": 1.2, "增持": 1.3, "大股东增持": 1.5,
    "机构买入": 1.3, "北向资金流入": 1.2, "外资增持": 1.2,
    
    # Moderate positive
    "上涨": 0.8, "走高": 0.8, "反弹": 0.7, "回升": 0.7,
    "利率下调": 0.8, "降准": 1.0, "降息": 1.0, "宽松": 0.8,
    "刺激": 0.7, "支持": 0.6, "鼓励": 0.6, "扶持": 0.7,
    "中标": 1.0, "签约": 0.8, "合作": 0.6, "战略合作": 0.8,
    "新产品": 0.7, "技术突破": 1.0, "专利": 0.7, "创新": 0.6,
    "产能扩张": 0.8, "订单增长": 1.0, "市场份额提升": 0.9,
    
    # Policy positive
    "减税": 1.0, "补贴": 0.8, "政策支持": 1.0, "国家战略": 0.9,
    "改革": 0.5, "开放": 0.5, "自贸区": 0.7, "新基建": 0.8,
    "数字经济": 0.7, "碳中和": 0.6, "新能源": 0.6,
}

# Negative keywords
NEGATIVE_WORDS = {
    # Strong negative
    "跌停": -2.0, "大跌": -1.8, "暴跌": -1.8, "崩盘": -2.0,
    "利空": -1.5, "重大利空": -2.0, "爆雷": -2.0, "违规": -1.5,
    "处罚": -1.5, "罚款": -1.3, "退市": -2.0, "ST": -1.5,
    "亏损": -1.3, "业绩下滑": -1.5, "净利润下降": -1.3,
    "减持": -1.3, "大股东减持": -1.5, "高管减持": -1.2,
    "质押": -0.8, "爆仓": -1.8, "违约": -1.5,
    
    # Moderate negative
    "下跌": -0.8, "走低": -0.8, "回调": -0.5, "下探": -0.7,
    "加息": -0.8, "收紧": -0.8, "监管": -0.6, "审查": -0.7,
    "限制": -0.6, "禁止": -0.8, "制裁": -1.0, "贸易战": -1.0,
    "疫情": -0.7, "停产": -1.0, "停工": -0.8, "召回": -0.8,
    "诉讼": -0.7, "仲裁": -0.6, "调查": -0.7,
    
    # Risk keywords
    "风险": -0.5, "警告": -0.6, "预警": -0.6, "泡沫": -0.8,
    "过热": -0.6, "通胀": -0.5, "滞涨": -0.7,
    "北向资金流出": -1.0, "外资减持": -1.0,
}


def analyze_sentiment(text: str) -> Tuple[float, str]:
    """
    Improved sentiment scoring:
    - Uses weighted sum
    - Normalizes by total absolute weight contribution
    - Applies tanh to avoid easy saturation at ±1.0
    """
    if not text:
        return 0.0, "neutral"

    raw_score = 0.0
    abs_contrib = 0.0

    # cap each keyword count to avoid spam
    for word, weight in POSITIVE_WORDS.items():
        c = text.count(word)
        if c > 0:
            c = min(c, 3)
            raw_score += float(weight) * c
            abs_contrib += abs(float(weight)) * c

    for word, weight in NEGATIVE_WORDS.items():
        c = text.count(word)
        if c > 0:
            c = min(c, 3)
            raw_score += float(weight) * c
            abs_contrib += abs(float(weight)) * c

    if abs_contrib <= 1e-9:
        return 0.0, "neutral"

    # normalized base in [-inf, inf]
    base = raw_score / abs_contrib

    # squash to [-1, 1] smoothly
    import math
    normalized = math.tanh(1.25 * base)

    if normalized >= 0.2:
        label = "positive"
    elif normalized <= -0.2:
        label = "negative"
    else:
        label = "neutral"

    return round(float(normalized), 3), label


# =============================================================================
# NEWS FETCHERS (Work on China direct IP)
# =============================================================================

class SinaNewsFetcher:
    """Fetch news from Sina Finance (works on China IP)"""

    def __init__(self):
        self._session = requests.Session()
        self._session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Referer': 'https://finance.sina.com.cn/',
        })

    def fetch_market_news(self, count: int = 20) -> List[NewsItem]:
        """Fetch general market news"""
        try:
            url = "https://feed.mix.sina.com.cn/api/roll/get"
            params = {
                'pageid': '153',
                'lid': '2516',
                'k': '',
                'num': str(min(count, 50)),
                'page': '1',
            }
            r = self._session.get(url, params=params, timeout=5)
            data = r.json()

            items = []
            for article in data.get('result', {}).get('data', []):
                title = article.get('title', '').strip()
                if not title:
                    continue

                pub_time = datetime.now()
                try:
                    ctime = article.get('ctime', '')
                    if ctime:
                        pub_time = datetime.fromtimestamp(int(ctime))
                except Exception:
                    pass

                items.append(NewsItem(
                    title=title,
                    content=article.get('summary', '') or article.get('intro', '') or '',
                    source="sina",
                    url=article.get('url', ''),
                    publish_time=pub_time,
                    category="market",
                ))

            log.debug(f"Sina: fetched {len(items)} market news")
            return items

        except Exception as e:
            log.warning(f"Sina market news failed: {e}")
            return []

    def fetch_stock_news(self, stock_code: str, count: int = 10) -> List[NewsItem]:
        """Fetch news for specific stock"""
        try:
            code6 = str(stock_code).zfill(6)

            if code6.startswith(('6', '5')):
                symbol = f"sh{code6}"
            else:
                symbol = f"sz{code6}"

            url = f"https://search.sina.com.cn/news"
            params = {
                'q': code6,
                'c': 'news',
                'from': 'channel',
                'ie': 'utf-8',
                'num': str(min(count, 20)),
            }
            r = self._session.get(url, params=params, timeout=5)

            # Parse results (simplified)
            items = []
            # Sina search returns HTML, extract titles
            import re
            titles = re.findall(r'<h2><a[^>]*>(.+?)</a></h2>', r.text)
            for title in titles[:count]:
                title = re.sub(r'<[^>]+>', '', title).strip()
                if title:
                    items.append(NewsItem(
                        title=title,
                        source="sina",
                        stock_codes=[code6],
                        category="company",
                    ))

            return items

        except Exception as e:
            log.debug(f"Sina stock news failed for {stock_code}: {e}")
            return []


class EastmoneyNewsFetcher:
    """Fetch news from Eastmoney (works on China IP only)"""

    def __init__(self):
        self._session = requests.Session()
        self._session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Referer': 'https://www.eastmoney.com/',
        })

    def fetch_stock_news(self, stock_code: str, count: int = 10) -> List[NewsItem]:
        """Fetch stock-specific news and announcements"""
        try:
            code6 = str(stock_code).zfill(6)

            url = "https://search-api-web.eastmoney.com/search/jsonp"
            params = {
                'cb': 'jQuery_callback',
                'param': json.dumps({
                    "uid": "",
                    "keyword": code6,
                    "type": ["cmsArticleWebOld"],
                    "client": "web",
                    "clientType": "web",
                    "clientVersion": "curr",
                    "param": {"cmsArticleWebOld": {"searchScope": "default", "sort": "default", "pageIndex": 1, "pageSize": count}}
                })
            }

            r = self._session.get(url, params=params, timeout=5)
            text = r.text

            # Extract JSON from JSONP
            json_str = text[text.index('(') + 1:text.rindex(')')]
            data = json.loads(json_str)

            items = []
            articles = data.get('result', {}).get('cmsArticleWebOld', {}).get('list', [])

            for article in articles:
                title = article.get('title', '').strip()
                title = re.sub(r'<[^>]+>', '', title)  # Remove HTML tags
                if not title:
                    continue

                pub_time = datetime.now()
                try:
                    date_str = article.get('date', '')
                    if date_str:
                        pub_time = datetime.strptime(date_str[:19], "%Y-%m-%d %H:%M:%S")
                except Exception:
                    pass

                content = article.get('content', '') or article.get('mediaName', '') or ''
                content = re.sub(r'<[^>]+>', '', content)

                items.append(NewsItem(
                    title=title,
                    content=content[:500],
                    source="eastmoney",
                    url=article.get('url', ''),
                    publish_time=pub_time,
                    stock_codes=[code6],
                    category="company",
                ))

            log.debug(f"Eastmoney: fetched {len(items)} news for {code6}")
            return items

        except Exception as e:
            log.debug(f"Eastmoney news failed for {stock_code}: {e}")
            return []

    def fetch_policy_news(self, count: int = 15) -> List[NewsItem]:
        """Fetch policy and regulatory news"""
        try:
            url = "https://np-listapi.eastmoney.com/comm/web/getNewsByColumns"
            params = {
                'columns': 'CSRC,PBOC,MOF',  # 证监会, 央行, 财政部
                'pageSize': str(min(count, 30)),
                'pageIndex': '1',
            }
            r = self._session.get(url, params=params, timeout=5)
            data = r.json()

            items = []
            for article in data.get('data', {}).get('list', []):
                title = article.get('title', '').strip()
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

        except Exception as e:
            log.debug(f"Eastmoney policy news failed: {e}")
            return []


class TencentNewsFetcher:
    """Fetch news from Tencent Finance (works from ANY IP)"""

    def __init__(self):
        self._session = requests.Session()
        self._session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        })

    def fetch_market_news(self, count: int = 20) -> List[NewsItem]:
        """Fetch general financial news via Tencent"""
        try:
            url = "https://r.inews.qq.com/getSimpleNews"
            params = {
                'ids': 'finance_hot',
                'num': str(min(count, 30)),
            }
            r = self._session.get(url, params=params, timeout=5)
            data = r.json()

            items = []
            for article in data.get('newslist', []):
                title = article.get('title', '').strip()
                if not title:
                    continue

                pub_time = datetime.now()
                try:
                    ts = article.get('timestamp', '')
                    if ts:
                        pub_time = datetime.fromtimestamp(int(ts))
                except Exception:
                    pass

                items.append(NewsItem(
                    title=title,
                    content=article.get('abstract', '') or '',
                    source="tencent",
                    url=article.get('url', ''),
                    publish_time=pub_time,
                    category="market",
                ))

            log.debug(f"Tencent: fetched {len(items)} market news")
            return items

        except Exception as e:
            log.warning(f"Tencent market news failed: {e}")
            return []


# =============================================================================
# NEWS AGGREGATOR
# =============================================================================

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
        self._cache_ttl = 300  # 5 minutes
        self._lock = threading.RLock()

        # Rolling news buffer (last 200 items)
        self._all_news: deque = deque(maxlen=200)

    def _is_cache_valid(self, key: str) -> bool:
        return (key in self._cache and
                (time.time() - self._cache_time.get(key, 0)) < self._cache_ttl)

    def get_market_news(self, count: int = 30, force_refresh: bool = False) -> List[NewsItem]:
        """Get aggregated market news from all available sources."""
        cache_key = f"market_{count}"

        with self._lock:
            if not force_refresh and self._is_cache_valid(cache_key):
                return self._cache[cache_key]

        from core.network import get_network_env
        env = get_network_env()

        all_items: List[NewsItem] = []

        # Tencent always works
        if env.tencent_ok:
            try:
                all_items.extend(self._tencent.fetch_market_news(count))
            except Exception:
                pass

        # China direct: use Sina + Eastmoney
        if env.is_china_direct:
            try:
                all_items.extend(self._sina.fetch_market_news(count))
            except Exception:
                pass

            if env.eastmoney_ok:
                try:
                    all_items.extend(self._eastmoney.fetch_policy_news(count))
                except Exception:
                    pass

        # Deduplicate by title similarity
        unique = self._deduplicate(all_items)

        # Sort by time (newest first)
        unique.sort(key=lambda x: x.publish_time, reverse=True)
        unique = unique[:count]

        with self._lock:
            self._cache[cache_key] = unique
            self._cache_time[cache_key] = time.time()
            for item in unique:
                self._all_news.appendleft(item)

        log.info(f"Aggregated {len(unique)} market news items")
        return unique

    def get_stock_news(self, stock_code: str, count: int = 15,
                       force_refresh: bool = False) -> List[NewsItem]:
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
                all_items.extend(self._sina.fetch_stock_news(code6, count))
            except Exception:
                pass

            if env.eastmoney_ok:
                try:
                    all_items.extend(self._eastmoney.fetch_stock_news(code6, count))
                except Exception:
                    pass

        # Filter market news for this stock code
        with self._lock:
            for item in self._all_news:
                if code6 in item.title or code6 in str(item.stock_codes):
                    all_items.append(item)

        unique = self._deduplicate(all_items)
        unique.sort(key=lambda x: x.publish_time, reverse=True)
        unique = unique[:count]

        # Tag all with stock code
        for item in unique:
            if code6 not in item.stock_codes:
                item.stock_codes.append(code6)

        with self._lock:
            self._cache[cache_key] = unique
            self._cache_time[cache_key] = time.time()

        return unique

    def get_policy_news(self, count: int = 10) -> List[NewsItem]:
        """Get policy/regulatory news only."""
        all_news = self.get_market_news(count=50)
        policy = [n for n in all_news if n.category == "policy" or
                  any(kw in n.title for kw in ["央行", "证监会", "财政部", "国务院",
                                                 "政策", "监管", "改革", "法规"])]
        return policy[:count]

    def get_sentiment_summary(self, stock_code: str = None) -> Dict:
        """Get aggregated sentiment for stock or market."""
        if stock_code:
            news = self.get_stock_news(stock_code)
        else:
            news = self.get_market_news()

        if not news:
            return {
                'overall_sentiment': 0.0,
                'label': 'neutral',
                'positive_count': 0,
                'negative_count': 0,
                'neutral_count': 0,
                'total': 0,
                'top_positive': [],
                'top_negative': [],
            }

        scores = [n.sentiment_score for n in news]
        overall = sum(scores) / len(scores) if scores else 0.0

        positive = [n for n in news if n.sentiment_label == "positive"]
        negative = [n for n in news if n.sentiment_label == "negative"]
        neutral = [n for n in news if n.sentiment_label == "neutral"]

        return {
            'overall_sentiment': round(overall, 3),
            'label': "positive" if overall > 0.1 else ("negative" if overall < -0.1 else "neutral"),
            'positive_count': len(positive),
            'negative_count': len(negative),
            'neutral_count': len(neutral),
            'total': len(news),
            'top_positive': [n.to_dict() for n in sorted(positive, key=lambda x: x.sentiment_score, reverse=True)[:3]],
            'top_negative': [n.to_dict() for n in sorted(negative, key=lambda x: x.sentiment_score)[:3]],
        }

    def get_news_features(self, stock_code: str = None,
                          hours_lookback: int = 24) -> Dict[str, float]:
        """
        Get numerical features from news for AI model input.
        These can be appended to the technical features.
        """
        if stock_code:
            news = self.get_stock_news(stock_code, count=50)
        else:
            news = self.get_market_news(count=50)

        cutoff = datetime.now() - timedelta(hours=hours_lookback)
        recent = [n for n in news if n.publish_time >= cutoff]

        if not recent:
            return {
                'news_sentiment_avg': 0.0,
                'news_sentiment_std': 0.0,
                'news_positive_ratio': 0.0,
                'news_negative_ratio': 0.0,
                'news_volume': 0.0,
                'news_importance_avg': 0.5,
                'news_recency_score': 0.0,
                'policy_sentiment': 0.0,
            }

        scores = [n.sentiment_score for n in recent]
        positive = sum(1 for s in scores if s > 0.1)
        negative = sum(1 for s in scores if s < -0.1)
        total = len(scores)

        # Recency-weighted sentiment (newer news matters more)
        recency_weights = []
        for n in recent:
            age_hours = (datetime.now() - n.publish_time).total_seconds() / 3600.0
            weight = max(0.1, 1.0 - (age_hours / hours_lookback))
            recency_weights.append(weight)

        weighted_sentiment = sum(s * w for s, w in zip(scores, recency_weights)) / sum(recency_weights) if recency_weights else 0.0

        # Policy sentiment
        policy_items = [n for n in recent if n.category == "policy"]
        policy_sentiment = sum(n.sentiment_score for n in policy_items) / max(len(policy_items), 1)

        import numpy as np

        return {
            'news_sentiment_avg': round(float(np.mean(scores)), 4),
            'news_sentiment_std': round(float(np.std(scores)), 4) if len(scores) > 1 else 0.0,
            'news_positive_ratio': round(positive / total, 4),
            'news_negative_ratio': round(negative / total, 4),
            'news_volume': min(total / 20.0, 1.0),  # Normalized 0-1
            'news_importance_avg': round(float(np.mean([n.importance for n in recent])), 4),
            'news_recency_score': round(weighted_sentiment, 4),
            'policy_sentiment': round(policy_sentiment, 4),
        }

    def _deduplicate(self, items: List[NewsItem]) -> List[NewsItem]:
        """Remove duplicate news by title similarity."""
        seen_titles = set()
        unique = []
        for item in items:
            # Simple dedup: first 20 chars of title
            key = item.title[:20] if item.title else ""
            if key and key not in seen_titles:
                seen_titles.add(key)
                unique.append(item)
        return unique

    def clear_cache(self):
        with self._lock:
            self._cache.clear()
            self._cache_time.clear()


# Singleton
_aggregator: Optional[NewsAggregator] = None


def get_news_aggregator() -> NewsAggregator:
    global _aggregator
    try:
        lock = globals().get("_aggregator_lock")
    except Exception:
        lock = None

    if lock is None:
        import threading
        globals()["_aggregator_lock"] = threading.Lock()
        lock = globals()["_aggregator_lock"]

    if _aggregator is None:
        with lock:
            if _aggregator is None:
                _aggregator = NewsAggregator()
    return _aggregator