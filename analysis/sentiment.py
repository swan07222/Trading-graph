"""
Sentiment Analysis - News and Social Media Analysis
"""
import re
import time
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import json
import pickle
import requests
from bs4 import BeautifulSoup

from config import CONFIG
from utils.logger import log


@dataclass
class NewsItem:
    """News article"""
    title: str
    content: str
    source: str
    url: str
    timestamp: datetime
    stock_codes: List[str]
    sentiment_score: float = 0.0
    sentiment_label: str = "neutral"


@dataclass 
class SentimentResult:
    """Sentiment analysis result"""
    score: float
    confidence: float
    label: str
    method: str


class KeywordSentimentAnalyzer:
    """Keyword-based sentiment analysis for Chinese financial text"""
    
    POSITIVE = {
        '暴涨': 3, '涨停': 3, '大涨': 3, '飙升': 3,
        '突破新高': 3, '业绩暴增': 3, '超预期': 3,
        '上涨': 2, '涨': 2, '利好': 2, '增长': 2,
        '强势': 2, '看好': 2, '推荐': 2, '买入': 2,
        '稳定': 1, '向好': 1, '乐观': 1, '反弹': 1,
    }
    
    NEGATIVE = {
        '暴跌': 3, '跌停': 3, '大跌': 3, '崩盘': 3,
        '业绩暴雷': 3, '退市': 3, '重大利空': 3,
        '下跌': 2, '跌': 2, '利空': 2, '下降': 2,
        '减持': 2, '卖出': 2, '亏损': 2, '风险': 2,
        '弱势': 1, '回调': 1, '下行': 1, '谨慎': 1,
    }
    
    def analyze(self, text: str) -> SentimentResult:
        if not text:
            return SentimentResult(0.0, 0.0, "neutral", "keyword")
        
        pos_score = sum(weight for word, weight in self.POSITIVE.items() if word in text)
        neg_score = sum(weight for word, weight in self.NEGATIVE.items() if word in text)
        
        total = pos_score + neg_score
        
        if total == 0:
            return SentimentResult(0.0, 0.2, "neutral", "keyword")
        
        score = (pos_score - neg_score) / total
        confidence = min(total / 10, 1.0)
        
        if score > 0.2:
            label = "positive"
        elif score < -0.2:
            label = "negative"
        else:
            label = "neutral"
        
        return SentimentResult(score, confidence, label, "keyword")


class SentimentAnalyzer:
    """Ensemble sentiment analyzer"""
    
    def __init__(self, use_bert: bool = False):
        self.keyword_analyzer = KeywordSentimentAnalyzer()
    
    def analyze(self, text: str) -> SentimentResult:
        return self.keyword_analyzer.analyze(text)
    
    def analyze_batch(self, texts: List[str]) -> List[SentimentResult]:
        return [self.analyze(text) for text in texts]


class NewsScraper:
    """News scraper for Chinese financial news sources"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
        self.analyzer = SentimentAnalyzer()
        
        self._cache_dir = CONFIG.DATA_DIR / "news_cache"
        self._cache_dir.mkdir(exist_ok=True)
        self._seen_hashes = set()
        
        self._last_request = {}
        self._rate_limits = {'sina': 0.5, 'eastmoney': 0.5, 'cls': 0.5}
    
    def _hash(self, text: str) -> str:
        return hashlib.md5(text.encode()).hexdigest()
    
    def _rate_limit(self, source: str):
        limit = self._rate_limits.get(source, 0.5)
        last = self._last_request.get(source, 0)
        elapsed = time.time() - last
        if elapsed < limit:
            time.sleep(limit - elapsed)
        self._last_request[source] = time.time()
    
    def _extract_stock_codes(self, text: str) -> List[str]:
        patterns = [
            r'[（(]([036]\d{5})[)）]',
            r'(?<!\d)([036]\d{5})(?!\d)',
        ]
        codes = set()
        for p in patterns:
            codes.update(re.findall(p, text))
        return list(codes)
    
    def scrape_sina(self, max_items: int = 50) -> List[NewsItem]:
        """Scrape Sina Finance news"""
        items = []
        self._rate_limit('sina')
        
        try:
            url = "https://feed.mix.sina.com.cn/api/roll/get"
            params = {'pageid': 153, 'lid': 2516, 'num': max_items}
            
            resp = self.session.get(url, params=params, timeout=10)
            data = resp.json()
            
            for item in data.get('result', {}).get('data', []):
                title = item.get('title', '')
                content = item.get('intro', '')[:500]
                
                h = self._hash(title + content)
                if h in self._seen_hashes:
                    continue
                self._seen_hashes.add(h)
                
                text = title + " " + content
                sentiment = self.analyzer.analyze(text)
                
                try:
                    ts = datetime.strptime(item.get('ctime', ''), "%Y-%m-%d %H:%M:%S")
                except:
                    ts = datetime.now()
                
                items.append(NewsItem(
                    title=title,
                    content=content,
                    source='sina',
                    url=item.get('url', ''),
                    timestamp=ts,
                    stock_codes=self._extract_stock_codes(text),
                    sentiment_score=sentiment.score,
                    sentiment_label=sentiment.label
                ))
            
            log.info(f"Scraped {len(items)} items from Sina")
            
        except Exception as e:
            log.error(f"Sina scraping error: {e}")
        
        return items
    
    def scrape_all(self) -> List[NewsItem]:
        """Scrape from all sources"""
        items = []
        items.extend(self.scrape_sina())
        items.sort(key=lambda x: x.timestamp, reverse=True)
        return items
    
    def get_stock_sentiment(self, stock_code: str) -> Tuple[float, float]:
        """Get aggregated sentiment for a specific stock"""
        all_news = self.scrape_all()
        relevant = [n for n in all_news if stock_code in n.stock_codes]
        
        if not relevant:
            return 0.0, 0.0
        
        now = datetime.now()
        weighted_score = 0
        total_weight = 0
        
        for news in relevant:
            hours_ago = (now - news.timestamp).total_seconds() / 3600
            weight = 1 / (1 + hours_ago / 24)
            weighted_score += news.sentiment_score * weight
            total_weight += weight
        
        if total_weight == 0:
            return 0.0, 0.0
        
        avg_score = weighted_score / total_weight
        confidence = min(len(relevant) / 5, 1.0)
        
        return avg_score, confidence
    
    def get_market_sentiment(self) -> Dict:
        """Get overall market sentiment"""
        all_news = self.scrape_all()
        
        if not all_news:
            return {'score': 0, 'label': 'neutral', 'news_count': 0}
        
        scores = [n.sentiment_score for n in all_news]
        avg_score = sum(scores) / len(scores)
        
        pos_count = sum(1 for n in all_news if n.sentiment_label == 'positive')
        neg_count = sum(1 for n in all_news if n.sentiment_label == 'negative')
        
        if avg_score > 0.15:
            label = 'bullish'
        elif avg_score < -0.15:
            label = 'bearish'
        else:
            label = 'neutral'
        
        return {
            'score': avg_score,
            'label': label,
            'news_count': len(all_news),
            'positive_count': pos_count,
            'negative_count': neg_count,
            'neutral_count': len(all_news) - pos_count - neg_count
        }