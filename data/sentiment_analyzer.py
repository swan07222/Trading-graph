# data/sentiment_analyzer.py
"""Sentiment Analysis for News and Policy Data.

This module provides sentiment analysis specifically tuned for:
- Financial news sentiment
- Policy/regulatory impact assessment
- Market sentiment aggregation
- Chinese and English language support

The analyzer uses a hybrid approach:
1. Rule-based sentiment scoring (financial lexicons)
2. Keyword-based policy impact detection
3. Entity extraction for company-specific news
4. Temporal sentiment tracking
"""

import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from config.settings import CONFIG
from utils.logger import get_logger

from .news_collector import NewsArticle, get_collector

log = get_logger(__name__)


@dataclass
class SentimentScore:
    """Sentiment score with breakdown."""
    overall: float = 0.0  # -1.0 (very negative) to 1.0 (very positive)
    policy_impact: float = 0.0  # -1.0 (negative policy) to 1.0 (positive policy)
    market_sentiment: float = 0.0  # -1.0 (bearish) to 1.0 (bullish)
    confidence: float = 0.0  # 0.0 to 1.0
    article_count: int = 0
    time_range_hours: int = 24

    def to_dict(self) -> dict:
        return {
            "overall": self.overall,
            "policy_impact": self.policy_impact,
            "market_sentiment": self.market_sentiment,
            "confidence": self.confidence,
            "article_count": self.article_count,
            "time_range_hours": self.time_range_hours,
        }


@dataclass
class EntitySentiment:
    """Sentiment for a specific entity (company, policy, person)."""
    entity: str
    entity_type: str  # 'company', 'policy', 'person', 'sector'
    sentiment: float
    mention_count: int
    articles: list[str] = field(default_factory=list)  # Article IDs


class SentimentAnalyzer:
    """Multi-factor sentiment analyzer for financial news."""

    # Chinese positive sentiment words (financial context)
    POSITIVE_ZH = [
        "上涨", "增长", "盈利", "利好", "突破", "牛市", "反弹",
        "超预期", "创新高", "强势", "看好", "买入", "推荐",
        "政策支持", "放宽", "刺激", "复苏", "繁荣",
    ]

    # Chinese negative sentiment words
    NEGATIVE_ZH = [
        "下跌", "下降", "亏损", "利空", "跌破", "熊市", "回调",
        "低于预期", "新低", "弱势", "看空", "卖出", "警告",
        "监管收紧", "处罚", "调查", "风险", "衰退",
    ]

    # English positive sentiment words
    POSITIVE_EN = [
        "rise", "grow", "profit", "positive", "breakthrough", "bull",
        "rebound", "beat", "high", "strong", "optimistic", "buy",
        "recommend", "support", "stimulus", "recovery", "boom",
    ]

    # English negative sentiment words
    NEGATIVE_EN = [
        "fall", "decline", "loss", "negative", "breakdown", "bear",
        "pullback", "miss", "low", "weak", "pessimistic", "sell",
        "warn", "tighten", "penalty", "investigation", "risk", "recession",
    ]

    # Policy impact keywords (positive)
    POSITIVE_POLICY_ZH = [
        "支持", "鼓励", "放宽", "刺激", "优惠", "补贴", "减税",
        "放宽监管", "政策利好", "扶持", "促进", "发展",
    ]

    POSITIVE_POLICY_EN = [
        "support", "encourage", "relax", "stimulus", "preferential",
        "subsidy", "tax cut", "deregulation", "promote", "develop",
    ]

    # Policy impact keywords (negative)
    NEGATIVE_POLICY_ZH = [
        "收紧", "限制", "禁止", "处罚", "调查", "监管加强",
        "政策收紧", "打压", "风险", "警告",
    ]

    NEGATIVE_POLICY_EN = [
        "tighten", "restrict", "ban", "penalty", "investigation",
        "regulation", "crackdown", "suppress", "risk", "warn",
    ]

    # Market sentiment indicators
    BULLISH_ZH = ["牛市", "做多", "买入", "看涨", "突破", "创新高"]
    BEARISH_ZH = ["熊市", "做空", "卖出", "看跌", "跌破", "新低"]

    BULLISH_EN = ["bull", "long", "buy", "breakout", "new high"]
    BEARISH_EN = ["bear", "short", "sell", "breakdown", "new low"]

    def __init__(self, lexicon_path: Optional[Path] = None) -> None:
        self.lexicon_path = lexicon_path or CONFIG.cache_dir / "sentiment_lexicon.json"
        self._custom_lexicon: dict[str, float] = {}
        self._load_lexicon()

    def _load_lexicon(self) -> None:
        """Load custom sentiment lexicon if exists."""
        if self.lexicon_path.exists():
            try:
                with open(self.lexicon_path, "r", encoding="utf-8") as f:
                    self._custom_lexicon = json.load(f)
                log.info(f"Loaded custom sentiment lexicon: {len(self._custom_lexicon)} entries")
            except Exception as e:
                log.warning(f"Failed to load lexicon: {e}")

    def _save_lexicon(self) -> None:
        """Save custom lexicon."""
        try:
            with open(self.lexicon_path, "w", encoding="utf-8") as f:
                json.dump(self._custom_lexicon, f, ensure_ascii=False, indent=2)
        except Exception as e:
            log.warning(f"Failed to save lexicon: {e}")

    def analyze_article(self, article: NewsArticle) -> SentimentScore:
        """Analyze sentiment of a single article."""
        text = article.title + " " + article.content

        # Determine language
        is_chinese = article.language == "zh" or self._detect_chinese(text)

        # Calculate sentiment scores
        sentiment = self._calculate_sentiment(text, is_chinese)
        policy_impact = self._calculate_policy_impact(text, is_chinese)
        market_sentiment = self._calculate_market_sentiment(text, is_chinese)

        # Combine scores
        overall = (sentiment * 0.4 + policy_impact * 0.4 + market_sentiment * 0.2)

        # Adjust based on article category
        if article.category == "policy":
            overall = overall * 0.3 + policy_impact * 0.7
        elif article.category == "market":
            overall = overall * 0.3 + market_sentiment * 0.7

        # Calculate confidence based on text length and keyword matches
        confidence = min(1.0, len(text) / 500.0)

        return SentimentScore(
            overall=overall,
            policy_impact=policy_impact,
            market_sentiment=market_sentiment,
            confidence=confidence,
            article_count=1,
        )

    def analyze_articles(
        self,
        articles: list[NewsArticle],
        hours_back: int = 24,
    ) -> SentimentScore:
        """Analyze sentiment across multiple articles."""
        if not articles:
            return SentimentScore()

        # Filter by time
        cutoff = datetime.now() - timedelta(hours=hours_back)
        recent = [a for a in articles if a.published_at >= cutoff]

        if not recent:
            return SentimentScore()

        # Calculate weighted average
        total_sentiment = 0.0
        total_policy = 0.0
        total_market = 0.0
        total_confidence = 0.0

        for article in recent:
            score = self.analyze_article(article)
            weight = article.relevance_score

            total_sentiment += score.overall * weight
            total_policy += score.policy_impact * weight
            total_market += score.market_sentiment * weight
            total_confidence += score.confidence * weight

        count = len(recent)
        return SentimentScore(
            overall=total_sentiment / count if count > 0 else 0.0,
            policy_impact=total_policy / count if count > 0 else 0.0,
            market_sentiment=total_market / count if count > 0 else 0.0,
            confidence=total_confidence / count if count > 0 else 0.0,
            article_count=count,
            time_range_hours=hours_back,
        )

    def analyze_for_symbol(
        self,
        symbol: str,
        articles: list[NewsArticle],
        hours_back: int = 24,
    ) -> SentimentScore:
        """Analyze sentiment specific to a stock symbol."""
        # Filter articles mentioning the symbol
        symbol_articles = [
            a for a in articles
            if symbol in a.title or symbol in a.content or symbol in a.entities
        ]

        if not symbol_articles:
            return SentimentScore()

        return self.analyze_articles(symbol_articles, hours_back)

    def extract_entities(self, articles: list[NewsArticle]) -> list[EntitySentiment]:
        """Extract entities and their sentiment from articles."""
        entity_mentions: dict[str, dict] = defaultdict(lambda: {
            "type": "unknown",
            "sentiment_sum": 0.0,
            "count": 0,
            "articles": [],
        })

        for article in articles:
            score = self.analyze_article(article)

            # Extract company names (simplified - would use NER in production)
            # Look for stock codes and company names
            text = article.title + " " + article.content

            # Match Chinese stock codes (6 digits)
            stock_codes = re.findall(r'\b[0-9]{6}\b', text)
            for code in stock_codes:
                entity_mentions[code]["type"] = "company"
                entity_mentions[code]["sentiment_sum"] += score.overall
                entity_mentions[code]["count"] += 1
                entity_mentions[code]["articles"].append(article.id)

            # Match policy names (simplified)
            if article.category == "policy":
                policy_keywords = self._extract_policy_keywords(text, article.language == "zh")
                for policy in policy_keywords:
                    entity_mentions[policy]["type"] = "policy"
                    entity_mentions[policy]["sentiment_sum"] += score.policy_impact
                    entity_mentions[policy]["count"] += 1
                    entity_mentions[policy]["articles"].append(article.id)

        # Convert to EntitySentiment objects
        result = []
        for entity, data in entity_mentions.items():
            if data["count"] > 0:
                result.append(EntitySentiment(
                    entity=entity,
                    entity_type=data["type"],
                    sentiment=data["sentiment_sum"] / data["count"],
                    mention_count=data["count"],
                    articles=data["articles"],
                ))

        return result

    def get_trading_signal(self, sentiment: SentimentScore) -> str:
        """Convert sentiment score to trading signal."""
        if sentiment.confidence < 0.3:
            return "HOLD"  # Low confidence

        if sentiment.overall > 0.5:
            return "STRONG_BUY"
        elif sentiment.overall > 0.2:
            return "BUY"
        elif sentiment.overall < -0.5:
            return "STRONG_SELL"
        elif sentiment.overall < -0.2:
            return "SELL"
        else:
            return "HOLD"

    def _detect_chinese(self, text: str) -> bool:
        """Detect if text is primarily Chinese."""
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        return chinese_chars > len(text) * 0.3

    def _calculate_sentiment(self, text: str, is_chinese: bool) -> float:
        """Calculate general sentiment score."""
        text_lower = text.lower()

        if is_chinese:
            positive = self.POSITIVE_ZH
            negative = self.NEGATIVE_ZH
        else:
            positive = self.POSITIVE_EN
            negative = self.NEGATIVE_EN

        positive_count = sum(1 for word in positive if word in text_lower)
        negative_count = sum(1 for word in negative if word in text_lower)

        total = positive_count + negative_count
        if total == 0:
            return 0.0

        # Score from -1 to 1
        return (positive_count - negative_count) / total

    def _calculate_policy_impact(self, text: str, is_chinese: bool) -> float:
        """Calculate policy impact score."""
        text_lower = text.lower()

        if is_chinese:
            positive = self.POSITIVE_POLICY_ZH
            negative = self.NEGATIVE_POLICY_ZH
        else:
            positive = self.POSITIVE_POLICY_EN
            negative = self.NEGATIVE_POLICY_EN

        positive_count = sum(1 for word in positive if word in text_lower)
        negative_count = sum(1 for word in negative if word in text_lower)

        total = positive_count + negative_count
        if total == 0:
            return 0.0

        return (positive_count - negative_count) / total

    def _calculate_market_sentiment(self, text: str, is_chinese: bool) -> float:
        """Calculate market sentiment score."""
        text_lower = text.lower()

        if is_chinese:
            bullish = self.BULLISH_ZH
            bearish = self.BEARISH_ZH
        else:
            bullish = self.BULLISH_EN
            bearish = self.BEARISH_EN

        bullish_count = sum(1 for word in bullish if word in text_lower)
        bearish_count = sum(1 for word in bearish if word in text_lower)

        total = bullish_count + bearish_count
        if total == 0:
            return 0.0

        return (bullish_count - bearish_count) / total

    def _extract_policy_keywords(self, text: str, is_chinese: bool) -> list[str]:
        """Extract policy-related keywords from text."""
        keywords = []

        # Look for policy names (simplified pattern matching)
        if is_chinese:
            # Match patterns like "XXX 政策", "XXX 规定"
            patterns = [
                r'([\u4e00-\u9fff]{2,10}政策)',
                r'([\u4e00-\u9fff]{2,10}规定)',
                r'([\u4e00-\u9fff]{2,10}条例)',
            ]
        else:
            patterns = [
                r'([A-Z][a-z]+ Policy)',
                r'([A-Z][a-z]+ Regulation)',
                r'([A-Z][a-z]+ Act)',
            ]

        for pattern in patterns:
            matches = re.findall(pattern, text)
            keywords.extend(matches)

        return list(set(keywords))

    def add_custom_sentiment(self, word: str, score: float) -> None:
        """Add custom word to sentiment lexicon."""
        self._custom_lexicon[word.lower()] = max(-1.0, min(1.0, score))
        self._save_lexicon()

    def get_sentiment_history(
        self,
        articles: list[NewsArticle],
        days: int = 7,
    ) -> list[dict]:
        """Get sentiment trend over time."""
        if not articles:
            return []

        # Group by date
        by_date: dict[str, list[NewsArticle]] = defaultdict(list)
        for article in articles:
            date_str = article.published_at.strftime("%Y-%m-%d")
            by_date[date_str].append(article)

        # Calculate daily sentiment
        result = []
        for date_str in sorted(by_date.keys())[-days:]:
            daily_articles = by_date[date_str]
            score = self.analyze_articles(daily_articles, hours_back=24)
            result.append({
                "date": date_str,
                "sentiment": score.overall,
                "policy_impact": score.policy_impact,
                "market_sentiment": score.market_sentiment,
                "article_count": score.article_count,
            })

        return result


# Singleton instance
_analyzer: Optional[SentimentAnalyzer] = None


def get_analyzer() -> SentimentAnalyzer:
    """Get or create sentiment analyzer instance."""
    global _analyzer
    if _analyzer is None:
        _analyzer = SentimentAnalyzer()
    return _analyzer


def reset_analyzer() -> None:
    """Reset analyzer instance."""
    global _analyzer
    _analyzer = None
