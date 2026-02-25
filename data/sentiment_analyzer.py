# data/sentiment_analyzer.py
"""Sentiment Analysis for News and Policy Data.

This module provides sentiment analysis specifically tuned for:
- Financial news sentiment
- Policy/regulatory impact assessment
- Market sentiment aggregation
- Chinese and English language support

The analyzer uses a hybrid approach:
1. LLM-based sentiment scoring (transformer models)
2. Deep learning policy impact detection
3. Entity extraction for company-specific news
4. Temporal sentiment tracking
5. Uncertainty estimation with Monte Carlo dropout
"""

import json
import math
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

from config.settings import CONFIG
from utils.logger import get_logger

from .llm_sentiment import LLM_sentimentAnalyzer, get_llm_analyzer
from .news_collector import NewsArticle

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
    """Multi-factor sentiment analyzer for financial news with LLM support."""

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

    def __init__(
        self,
        lexicon_path: Path | None = None,
        use_llm: bool = True,
        llm_confidence_threshold: float = 0.6,
    ) -> None:
        self.lexicon_path = lexicon_path or CONFIG.cache_dir / "sentiment_lexicon.json"
        self._custom_lexicon: dict[str, float] = {}
        self._load_lexicon()

        # LLM-based analysis
        self.use_llm = use_llm
        self.llm_confidence_threshold = llm_confidence_threshold
        self._llm_analyzer: LLM_sentimentAnalyzer | None = None

        if use_llm:
            try:
                self._llm_analyzer = get_llm_analyzer()
                log.info("LLM sentiment analyzer initialized")
            except Exception as e:
                log.warning(f"Failed to initialize LLM analyzer: {e}. Using fallback.")
                self.use_llm = False

    def _load_lexicon(self) -> None:
        """Load custom sentiment lexicon if exists."""
        if self.lexicon_path.exists():
            try:
                with open(self.lexicon_path, encoding="utf-8") as f:
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
        """Analyze sentiment of a single article using LLM + hybrid approach."""
        start_time = time.time()

        # Try LLM-based analysis first
        if self.use_llm and self._llm_analyzer is not None:
            try:
                llm_result = self._llm_analyzer.analyze(article)

                # Use LLM result if confidence is high enough
                if llm_result.confidence >= self.llm_confidence_threshold:
                    processing_time = (time.time() - start_time) * 1000
                    log.debug(
                        f"LLM analysis for {article.id}: "
                        f"overall={llm_result.overall:.3f}, "
                        f"confidence={llm_result.confidence:.3f}, "
                        f"time={processing_time:.1f}ms"
                    )

                    return SentimentScore(
                        overall=llm_result.overall,
                        policy_impact=llm_result.policy_impact,
                        market_sentiment=llm_result.market_sentiment,
                        confidence=llm_result.confidence,
                        article_count=1,
                    )
                else:
                    log.debug(
                        f"LLM confidence too low ({llm_result.confidence:.3f}), "
                        f"using fallback"
                    )
            except Exception as e:
                log.warning(f"LLM analysis failed for {article.id}: {e}. Using fallback.")

        # Fallback to rule-based analysis
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

        processing_time = (time.time() - start_time) * 1000
        log.debug(
            f"Fallback analysis for {article.id}: "
            f"overall={overall:.3f}, confidence={confidence:.3f}, "
            f"time={processing_time:.1f}ms"
        )

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
        total_weight = 0.0

        for article in recent:
            score = self.analyze_article(article)
            weight = self._article_weight(article)
            total_weight += weight

            total_sentiment += score.overall * weight
            total_policy += score.policy_impact * weight
            total_market += score.market_sentiment * weight
            total_confidence += score.confidence * weight

        count = len(recent)
        denom = total_weight if total_weight > 0 else float(count)
        confidence = (total_confidence / denom) if denom > 0 else 0.0
        if count > 1:
            # Increase confidence as article sample-size grows.
            confidence *= (0.75 + min(0.35, math.log1p(float(count)) / 4.0))
        return SentimentScore(
            overall=total_sentiment / denom if denom > 0 else 0.0,
            policy_impact=total_policy / denom if denom > 0 else 0.0,
            market_sentiment=total_market / denom if denom > 0 else 0.0,
            confidence=max(0.0, min(1.0, confidence)),
            article_count=count,
            time_range_hours=hours_back,
        )

    @staticmethod
    def _article_weight(article: NewsArticle) -> float:
        try:
            weight = float(getattr(article, "relevance_score", 0.0) or 0.0)
        except Exception:
            weight = 0.0
        if not math.isfinite(weight) or weight <= 0:
            return 1.0
        return max(0.1, min(2.0, weight))

    @staticmethod
    def _normalize_entity_name(raw: object) -> str:
        text = str(raw or "").strip()
        if not text:
            return ""
        text = re.sub(r"\s+", " ", text)
        if len(text) > 64:
            return ""
        if re.fullmatch(r"\d{1,6}", text):
            return text.zfill(6)
        return text

    def _record_entity(
        self,
        entity_mentions: dict[str, dict],
        *,
        name: str,
        entity_type: str,
        sentiment: float,
        article_id: str,
    ) -> None:
        entity = self._normalize_entity_name(name)
        if not entity:
            return
        normalized_type = self._normalize_entity_type(entity_type)
        slot = entity_mentions[entity]
        if slot["type"] == "unknown":
            slot["type"] = normalized_type
        elif normalized_type == "policy":
            # Policy labeling wins for policy matches.
            slot["type"] = normalized_type
        elif normalized_type == "company" and slot["type"] == "entity":
            # Upgrade generic labels when we later confirm it is a company.
            slot["type"] = normalized_type
        slot["sentiment_sum"] += float(sentiment)
        slot["count"] += 1
        slot["articles"].add(str(article_id))

    @staticmethod
    def _normalize_entity_type(entity_type: object) -> str:
        """Canonicalize entity type labels from mixed extraction sources."""
        kind = str(entity_type or "").strip().lower()
        if kind in {"stock", "stock_code", "ticker", "symbol", "equity", "org", "organization"}:
            return "company"
        if kind in {"company", "policy", "person", "sector", "keyword"}:
            return kind
        if kind in {"regulation", "law"}:
            return "policy"
        if kind in {"people", "executive"}:
            return "person"
        if kind in {"industry"}:
            return "sector"
        return kind or "entity"

    @staticmethod
    def _extract_company_like_entities(text: str, is_chinese: bool) -> list[str]:
        entities: list[str] = []
        entities.extend(re.findall(r"\b[0-9]{6}\b", text))
        entities.extend(
            t.lstrip("$")
            for t in re.findall(r"\$?[A-Z]{2,5}\b", text)
            if t.lstrip("$") not in {"USD", "CNY", "GDP", "CPI", "PPI", "FOMC"}
        )

        if is_chinese:
            entities.extend(
                re.findall(
                    r"([\u4e00-\u9fff]{2,18}(?:股份|集团|银行|科技|能源|证券|医药|汽车|地产))",
                    text,
                )
            )
        else:
            entities.extend(
                re.findall(
                    r"\b([A-Z][A-Za-z&\-]{1,30}\s(?:Inc|Corp|Corporation|Ltd|Limited|Group|Bank|Energy|Pharma|Technology|Tech))\b",
                    text,
                )
            )
        return list(dict.fromkeys(entities))

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
        """Extract entities and their sentiment from articles using LLM NER."""
        entity_mentions: dict[str, dict] = defaultdict(lambda: {
            "type": "unknown",
            "sentiment_sum": 0.0,
            "count": 0,
            "articles": set(),
        })

        for article in articles:
            score = self.analyze_article(article)

            text = article.title + " " + article.content
            is_chinese = article.language == "zh" or self._detect_chinese(text)

            # Use LLM-based entity extraction if available
            if self.use_llm and self._llm_analyzer is not None:
                try:
                    llm_result = self._llm_analyzer.analyze(article)
                    
                    # Use LLM-extracted entities
                    for entity_data in llm_result.entities:
                        entity = self._normalize_entity_name(entity_data.get("text", ""))
                        if not entity:
                            continue
                        entity_type = entity_data.get("type", "entity")
                        self._record_entity(
                            entity_mentions,
                            name=entity,
                            entity_type=entity_type,
                            sentiment=score.overall,
                            article_id=article.id,
                        )
                    
                    # Also use keywords from LLM
                    for keyword in llm_result.keywords:
                        if keyword not in entity_mentions:
                            entity_mentions[keyword]["type"] = "keyword"
                except Exception as e:
                    log.warning("LLM entity extraction failed: %s", e)

            # Fallback to rule-based extraction
            for raw_entity in list(getattr(article, "entities", []) or []):
                entity = self._normalize_entity_name(raw_entity)
                if not entity:
                    continue
                inferred_type = "company" if re.fullmatch(r"\d{6}", entity) else "entity"
                self._record_entity(
                    entity_mentions,
                    name=entity,
                    entity_type=inferred_type,
                    sentiment=score.overall,
                    article_id=article.id,
                )

            for entity in self._extract_company_like_entities(text, is_chinese):
                self._record_entity(
                    entity_mentions,
                    name=entity,
                    entity_type="company",
                    sentiment=score.overall,
                    article_id=article.id,
                )

            if article.category == "policy" or ("policy" in text.lower()):
                policy_keywords = self._extract_policy_keywords(text, is_chinese)
                for policy in policy_keywords:
                    self._record_entity(
                        entity_mentions,
                        name=policy,
                        entity_type="policy",
                        sentiment=score.policy_impact,
                        article_id=article.id,
                    )

        # Convert to EntitySentiment objects
        result: list[EntitySentiment] = []
        for entity, data in entity_mentions.items():
            if data["count"] > 0:
                result.append(EntitySentiment(
                    entity=entity,
                    entity_type=data["type"],
                    sentiment=data["sentiment_sum"] / data["count"],
                    mention_count=data["count"],
                    articles=sorted(data["articles"]),
                ))

        result.sort(key=lambda x: (x.mention_count, abs(x.sentiment)), reverse=True)
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
_analyzer: SentimentAnalyzer | None = None


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
