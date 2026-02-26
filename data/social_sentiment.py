# data/social_sentiment.py
"""Social Media Sentiment Analysis for China Markets.

This module collects and analyzes sentiment from Chinese social media platforms:
- Weibo (微博) - Twitter-like microblogging platform
- Xueqiu (雪球) - Social investment platform
- EastMoney Guba (东方财富股吧) - Stock discussion forum
- Sina Finance Comments - Financial news comments

Features:
- Real-time social media monitoring
- Influencer/trader sentiment tracking
- Retail sentiment index
- Social volume spikes detection
- Sentiment divergence analysis
"""

import hashlib
import json
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

import requests

from config.settings import CONFIG
from utils.logger import get_logger

from .news_collector import NewsArticle

log = get_logger(__name__)


@dataclass
class SocialPost:
    """Social media post structure."""
    id: str
    platform: str  # weibo, xueqiu, guba, sina
    content: str
    author: str
    author_type: str  # retail, influencer, institution, verified
    published_at: datetime
    collected_at: datetime
    likes: int = 0
    comments: int = 0
    shares: int = 0
    sentiment_score: float = 0.0
    relevance_score: float = 0.0
    stock_mentions: list[str] = field(default_factory=list)
    topics: list[str] = field(default_factory=list)
    is_original: bool = True
    language: str = "zh"

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "platform": self.platform,
            "content": self.content,
            "author": self.author,
            "author_type": self.author_type,
            "published_at": self.published_at.isoformat(),
            "collected_at": self.collected_at.isoformat(),
            "likes": self.likes,
            "comments": self.comments,
            "shares": self.shares,
            "sentiment_score": self.sentiment_score,
            "relevance_score": self.relevance_score,
            "stock_mentions": self.stock_mentions,
            "topics": self.topics,
            "is_original": self.is_original,
            "language": self.language,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SocialPost":
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
            platform=data["platform"],
            content=data["content"],
            author=data["author"],
            author_type=data.get("author_type", "retail"),
            published_at=published_at,
            collected_at=collected_at,
            likes=data.get("likes", 0),
            comments=data.get("comments", 0),
            shares=data.get("shares", 0),
            sentiment_score=float(data.get("sentiment_score", 0.0)),
            relevance_score=float(data.get("relevance_score", 0.0)),
            stock_mentions=data.get("stock_mentions", []),
            topics=data.get("topics", []),
            is_original=data.get("is_original", True),
            language=data.get("language", "zh"),
        )


@dataclass
class SocialSentimentIndex:
    """Aggregated social sentiment index."""
    platform: str
    overall_sentiment: float  # -1.0 to 1.0
    bullish_percent: float  # 0-100
    bearish_percent: float  # 0-100
    neutral_percent: float  # 0-100
    post_count: int
    engagement_score: float  # 0-100
    influencer_sentiment: float  # -1.0 to 1.0
    retail_sentiment: float  # -1.0 to 1.0
    sentiment_change_24h: float
    volume_change_24h: float
    top_topics: list[str] = field(default_factory=list)
    top_stocks: list[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        return {
            "platform": self.platform,
            "overall_sentiment": self.overall_sentiment,
            "bullish_percent": self.bullish_percent,
            "bearish_percent": self.bearish_percent,
            "neutral_percent": self.neutral_percent,
            "post_count": self.post_count,
            "engagement_score": self.engagement_score,
            "influencer_sentiment": self.influencer_sentiment,
            "retail_sentiment": self.retail_sentiment,
            "sentiment_change_24h": self.sentiment_change_24h,
            "volume_change_24h": self.volume_change_24h,
            "top_topics": self.top_topics,
            "top_stocks": self.top_stocks,
            "timestamp": self.timestamp.isoformat(),
        }


class SocialMediaCollector:
    """Social media data collector for China platforms."""

    # Platform endpoints (public APIs where available)
    PLATFORMS = {
        "weibo": {
            "base_url": "https://weibo.com",
            "search_url": "https://m.weibo.cn/api/container/getIndex",
            "trending_url": "https://s.weibo.com/top/summary",
        },
        "xueqiu": {
            "base_url": "https://xueqiu.com",
            "hot_url": "https://xueqiu.com/hq",
            "search_url": "https://xueqiu.com/query/posts",
        },
        "guba": {
            "base_url": "https://guba.eastmoney.com",
            "hot_url": "https://guba.eastmoney.com/rank/",
        },
        "sina": {
            "base_url": "https://finance.sina.com.cn",
            "comments_url": "https://comment5.news.sina.com.cn",
        },
    }

    # Stock code patterns for mention detection
    STOCK_PATTERNS = [
        r'(?:股票代码 | 代码 | 股票)[:：\s]*([0-9]{6})',  # Explicit stock code
        r'(?:SZ|SH|sz|sh)([0-9]{6})',  # With exchange prefix
        r'\b([0-9]{6})\b',  # Plain 6-digit code
        r'([A-Z][A-Z0-9]{5,})',  # Stock abbreviations
    ]

    def __init__(self) -> None:
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        })
        self._rate_limits: dict[str, float] = {}

    def _check_rate_limit(self, platform: str) -> bool:
        """Check if platform request is rate limited."""
        last_request = self._rate_limits.get(platform, 0)
        min_interval = 2.0  # Minimum 2 seconds between requests
        return (time.time() - last_request) >= min_interval

    def _update_rate_limit(self, platform: str) -> None:
        """Update rate limit timestamp for platform."""
        self._rate_limits[platform] = time.time()

    def _extract_stock_mentions(self, text: str) -> list[str]:
        """Extract stock codes and names from text."""
        mentions = []
        for pattern in self.STOCK_PATTERNS:
            matches = re.findall(pattern, text)
            mentions.extend(matches)

        # Deduplicate and validate
        unique_mentions = list(set(mentions))
        return [m for m in unique_mentions if len(m) >= 4]

    def _calculate_engagement_score(
        self,
        likes: int,
        comments: int,
        shares: int,
    ) -> float:
        """Calculate engagement score (0-100)."""
        # Weighted engagement
        score = likes * 1 + comments * 2 + shares * 3
        # Normalize to 0-100 (log scale to handle viral posts)
        import math
        normalized = min(100, math.log1p(score) * 15)
        return round(normalized, 2)

    def fetch_weibo_posts(
        self,
        keywords: list[str] | None = None,
        limit: int = 50,
        hours_back: int = 24,
    ) -> list[SocialPost]:
        """Fetch posts from Weibo."""
        posts = []

        if not self._check_rate_limit("weibo"):
            log.warning("Weibo rate limited, skipping")
            return posts

        try:
            # Search for stock-related posts
            if keywords:
                query = " ".join(keywords)
                params = {
                    "containerid": f"100103type=1&q={query}",
                    "page_type": "search_all",
                }
                # Note: Actual API call would need proper authentication
                # This is a placeholder for the implementation
                log.info(f"Weibo search for: {query}")

            # Fetch trending topics
            trending = self._fetch_weibo_trending()
            for topic in trending[:10]:
                post = SocialPost(
                    id=f"weibo_trend_{topic['name']}",
                    platform="weibo",
                    content=topic["name"],
                    author="trending",
                    author_type="verified",
                    published_at=datetime.now(),
                    collected_at=datetime.now(),
                    likes=topic.get("hot_value", 0),
                    comments=0,
                    shares=0,
                    stock_mentions=self._extract_stock_mentions(topic["name"]),
                    topics=[topic["name"]],
                )
                posts.append(post)

            self._update_rate_limit("weibo")

        except Exception as e:
            log.error(f"Weibo fetch error: {e}")

        return posts

    def _fetch_weibo_trending(self) -> list[dict]:
        """Fetch Weibo trending topics."""
        # Placeholder - would need actual API integration
        return [
            {"name": "A 股", "hot_value": 1000000},
            {"name": "股票", "hot_value": 500000},
            {"name": "基金", "hot_value": 300000},
        ]

    def fetch_xueqiu_posts(
        self,
        symbols: list[str] | None = None,
        limit: int = 50,
        hours_back: int = 24,
    ) -> list[SocialPost]:
        """Fetch posts from Xueqiu (雪球)."""
        posts = []

        if not self._check_rate_limit("xueqiu"):
            log.warning("Xueqiu rate limited, skipping")
            return posts

        try:
            # Fetch hot stocks discussion
            hot_posts = self._fetch_xueqiu_hot()
            for post_data in hot_posts[:limit]:
                post = SocialPost(
                    id=f"xueqiu_{post_data.get('id', '')}",
                    platform="xueqiu",
                    content=post_data.get("content", ""),
                    author=post_data.get("user", {}).get("screen_name", "unknown"),
                    author_type="influencer" if post_data.get("user", {}).get("verified") else "retail",
                    published_at=datetime.fromtimestamp(post_data.get("created_at", 0) / 1000),
                    collected_at=datetime.now(),
                    likes=post_data.get("likes_count", 0),
                    comments=post_data.get("comments_count", 0),
                    shares=post_data.get("shares_count", 0),
                    stock_mentions=self._extract_stock_mentions(post_data.get("content", "")),
                )
                posts.append(post)

            self._update_rate_limit("xueqiu")

        except Exception as e:
            log.error(f"Xueqiu fetch error: {e}")

        return posts

    def _fetch_xueqiu_hot(self) -> list[dict]:
        """Fetch Xueqiu hot posts."""
        # Placeholder - would need actual API integration
        return [
            {
                "id": "123456",
                "content": "今日 A 股市场表现强劲，建议关注科技股",
                "user": {"screen_name": "投资达人", "verified": True},
                "created_at": int(time.time() * 1000),
                "likes_count": 1500,
                "comments_count": 200,
                "shares_count": 100,
            },
        ]

    def fetch_guba_posts(
        self,
        symbols: list[str] | None = None,
        limit: int = 50,
        hours_back: int = 24,
    ) -> list[SocialPost]:
        """Fetch posts from EastMoney Guba (股吧)."""
        posts = []

        if not self._check_rate_limit("guba"):
            log.warning("Guba rate limited, skipping")
            return posts

        try:
            # Fetch hot stock discussions
            if symbols:
                for symbol in symbols[:5]:  # Limit to 5 symbols per request
                    symbol_posts = self._fetch_guba_by_symbol(symbol)
                    posts.extend(symbol_posts)
            else:
                # Fetch general hot posts
                posts = self._fetch_guba_hot()

            self._update_rate_limit("guba")

        except Exception as e:
            log.error(f"Guba fetch error: {e}")

        return posts[:limit]

    def _fetch_guba_by_symbol(self, symbol: str) -> list[SocialPost]:
        """Fetch Guba posts for specific stock symbol."""
        # Placeholder - would need actual API integration
        return [
            SocialPost(
                id=f"guba_{symbol}_1",
                platform="guba",
                content=f"{symbol} 今日走势强劲，建议持有",
                author="股民小王",
                author_type="retail",
                published_at=datetime.now(),
                collected_at=datetime.now(),
                likes=50,
                comments=10,
                shares=5,
                stock_mentions=[symbol],
            ),
        ]

    def _fetch_guba_hot(self) -> list[SocialPost]:
        """Fetch hot posts from Guba."""
        # Placeholder
        return []

    def collect_all(
        self,
        symbols: list[str] | None = None,
        keywords: list[str] | None = None,
        limit: int = 100,
        hours_back: int = 24,
    ) -> list[SocialPost]:
        """Collect posts from all platforms."""
        all_posts = []

        # Weibo - keyword based
        weibo_keywords = keywords or ["股票", "A 股", "基金"]
        all_posts.extend(self.fetch_weibo_posts(weibo_keywords, limit // 3, hours_back))

        # Xueqiu - symbol based
        all_posts.extend(self.fetch_xueqiu_posts(symbols, limit // 3, hours_back))

        # Guba - symbol based
        all_posts.extend(self.fetch_guba_posts(symbols, limit // 3, hours_back))

        # Deduplicate by content hash
        seen = set()
        unique_posts = []
        for post in all_posts:
            content_hash = hashlib.md5(post.content.encode()).hexdigest()
            if content_hash not in seen:
                seen.add(content_hash)
                unique_posts.append(post)

        # Sort by engagement
        unique_posts.sort(
            key=lambda p: p.likes + p.comments * 2 + p.shares * 3,
            reverse=True,
        )

        return unique_posts[:limit]


class SocialSentimentAnalyzer:
    """Analyzer for social media sentiment."""

    # Sentiment keywords
    BULLISH_ZH = [
        "涨", "升", "突破", "利好", "看好", "买入", "强势", "创新高",
        "牛市", "反弹", "反转", "机会", "价值", "低估", "潜力",
    ]
    BEARISH_ZH = [
        "跌", "降", "跌破", "利空", "看空", "卖出", "弱势", "新低",
        "熊市", "回调", "风险", "高估", "泡沫", "警惕",
    ]

    def __init__(self) -> None:
        self._historical_data: dict[str, list[SocialSentimentIndex]] = {}

    def analyze_sentiment(self, text: str) -> float:
        """Analyze sentiment of text (-1.0 to 1.0)."""
        score = 0.0
        total = 0

        # Count bullish terms
        for term in self.BULLISH_ZH:
            if term in text:
                score += 1
                total += 1

        # Count bearish terms
        for term in self.BEARISH_ZH:
            if term in text:
                score -= 1
                total += 1

        if total == 0:
            return 0.0

        return score / total

    def calculate_platform_index(
        self,
        posts: list[SocialPost],
        platform: str,
    ) -> SocialSentimentIndex:
        """Calculate sentiment index for a platform."""
        if not posts:
            return SocialSentimentIndex(
                platform=platform,
                overall_sentiment=0.0,
                bullish_percent=33.3,
                bearish_percent=33.3,
                neutral_percent=33.4,
                post_count=0,
                engagement_score=0.0,
                influencer_sentiment=0.0,
                retail_sentiment=0.0,
                sentiment_change_24h=0.0,
                volume_change_24h=0.0,
            )

        # Calculate sentiment distribution
        bullish = 0
        bearish = 0
        neutral = 0
        influencer_scores = []
        retail_scores = []
        total_engagement = 0
        topics_count: dict[str, int] = {}
        stocks_count: dict[str, int] = {}

        for post in posts:
            # Sentiment
            if post.sentiment_score > 0.1:
                bullish += 1
            elif post.sentiment_score < -0.1:
                bearish += 1
            else:
                neutral += 1

            # By author type
            if post.author_type in ("influencer", "verified", "institution"):
                influencer_scores.append(post.sentiment_score)
            else:
                retail_scores.append(post.sentiment_score)

            # Engagement
            total_engagement += post.likes + post.comments + post.shares

            # Topics and stocks
            for topic in post.topics:
                topics_count[topic] = topics_count.get(topic, 0) + 1
            for stock in post.stock_mentions:
                stocks_count[stock] = stocks_count.get(stock, 0) + 1

        total = len(posts)
        overall_sentiment = sum(p.sentiment_score for p in posts) / total

        # Calculate index
        index = SocialSentimentIndex(
            platform=platform,
            overall_sentiment=round(overall_sentiment, 3),
            bullish_percent=round(bullish / total * 100, 1),
            bearish_percent=round(bearish / total * 100, 1),
            neutral_percent=round(neutral / total * 100, 1),
            post_count=total,
            engagement_score=round(total_engagement / total, 2),
            influencer_sentiment=round(
                sum(influencer_scores) / len(influencer_scores) if influencer_scores else 0.0, 3
            ),
            retail_sentiment=round(
                sum(retail_scores) / len(retail_scores) if retail_scores else 0.0, 3
            ),
            sentiment_change_24h=0.0,  # Would compare with historical data
            volume_change_24h=0.0,
            top_topics=sorted(topics_count.keys(), key=lambda x: topics_count[x], reverse=True)[:5],
            top_stocks=sorted(stocks_count.keys(), key=lambda x: stocks_count[x], reverse=True)[:5],
        )

        return index

    def detect_volume_spike(
        self,
        current_posts: list[SocialPost],
        historical_avg: float,
        threshold: float = 2.0,
    ) -> bool:
        """Detect unusual volume spike in social mentions."""
        current_volume = len(current_posts)
        if historical_avg == 0:
            return False
        return current_volume > (historical_avg * threshold)

    def detect_sentiment_divergence(
        self,
        influencer_sentiment: float,
        retail_sentiment: float,
        threshold: float = 0.3,
    ) -> str:
        """Detect divergence between influencer and retail sentiment."""
        diff = influencer_sentiment - retail_sentiment
        if abs(diff) > threshold:
            if diff > 0:
                return "influencers_more_bullish"
            else:
                return "influencers_more_bearish"
        return "aligned"


def get_social_collector() -> SocialMediaCollector:
    """Get social media collector instance."""
    return SocialMediaCollector()


def get_social_analyzer() -> SocialSentimentAnalyzer:
    """Get social sentiment analyzer instance."""
    return SocialSentimentAnalyzer()
