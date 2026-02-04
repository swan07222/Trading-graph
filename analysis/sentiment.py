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

# Optional: BERT-based sentiment
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


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
    score: float  # -1 (very negative) to +1 (very positive)
    confidence: float  # 0 to 1
    label: str  # "positive", "negative", "neutral"
    method: str  # "bert", "keyword", "ensemble"


class KeywordSentimentAnalyzer:
    """
    Keyword-based sentiment analysis for Chinese financial text
    """
    
    # Positive keywords with weights
    POSITIVE = {
        # Strong positive (weight 3)
        '暴涨': 3, '涨停': 3, '大涨': 3, '飙升': 3, '井喷': 3,
        '突破新高': 3, '创历史新高': 3, '强势涨停': 3, '连续涨停': 3,
        '业绩暴增': 3, '超预期': 3, '重大利好': 3,
        
        # Medium positive (weight 2)
        '上涨': 2, '涨': 2, '利好': 2, '增长': 2, '突破': 2,
        '强势': 2, '看好': 2, '推荐': 2, '买入': 2, '增持': 2,
        '盈利': 2, '景气': 2, '回暖': 2, '复苏': 2, '扩张': 2,
        '新高': 2, '上调': 2, '获批': 2, '中标': 2,
        
        # Weak positive (weight 1)
        '稳定': 1, '向好': 1, '乐观': 1, '反弹': 1, '回升': 1,
        '改善': 1, '提升': 1, '增加': 1, '扩大': 1, '签约': 1,
    }
    
    # Negative keywords with weights
    NEGATIVE = {
        # Strong negative (weight 3)
        '暴跌': 3, '跌停': 3, '大跌': 3, '崩盘': 3, '闪崩': 3,
        '业绩暴雷': 3, '爆雷': 3, '退市': 3, '重大利空': 3,
        '清仓': 3, '踩雷': 3, 'ST': 3, '*ST': 3,
        
        # Medium negative (weight 2)
        '下跌': 2, '跌': 2, '利空': 2, '下降': 2, '风险': 2,
        '减持': 2, '卖出': 2, '亏损': 2, '下调': 2, '萎缩': 2,
        '警告': 2, '处罚': 2, '违规': 2, '调查': 2, '诉讼': 2,
        '债务': 2, '担忧': 2, '承压': 2,
        
        # Weak negative (weight 1)
        '弱势': 1, '回调': 1, '下行': 1, '放缓': 1, '收窄': 1,
        '谨慎': 1, '观望': 1, '不确定': 1, '波动': 1,
    }
    
    def analyze(self, text: str) -> SentimentResult:
        """Analyze sentiment of text"""
        if not text:
            return SentimentResult(0.0, 0.0, "neutral", "keyword")
        
        pos_score = sum(weight for word, weight in self.POSITIVE.items() if word in text)
        neg_score = sum(weight for word, weight in self.NEGATIVE.items() if word in text)
        
        total = pos_score + neg_score
        
        if total == 0:
            return SentimentResult(0.0, 0.2, "neutral", "keyword")
        
        # Calculate normalized score
        score = (pos_score - neg_score) / total
        confidence = min(total / 10, 1.0)  # More keywords = more confident
        
        if score > 0.2:
            label = "positive"
        elif score < -0.2:
            label = "negative"
        else:
            label = "neutral"
        
        return SentimentResult(score, confidence, label, "keyword")


class BertSentimentAnalyzer:
    """
    BERT-based sentiment analysis for Chinese text
    Uses pre-trained Chinese financial BERT model
    """
    
    def __init__(self, model_name: str = "hfl/chinese-roberta-wwm-ext"):
        self.model = None
        self.tokenizer = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self._initialized = False
        
        if TRANSFORMERS_AVAILABLE:
            self._load_model(model_name)
    
    def _load_model(self, model_name: str):
        """Load BERT model"""
        try:
            log.info(f"Loading BERT model: {model_name}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                num_labels=3  # negative, neutral, positive
            )
            self.model.to(self.device)
            self.model.eval()
            
            self._initialized = True
            log.info("BERT model loaded successfully")
            
        except Exception as e:
            log.warning(f"Failed to load BERT model: {e}")
            self._initialized = False
    
    def is_available(self) -> bool:
        return self._initialized
    
    def analyze(self, text: str) -> Optional[SentimentResult]:
        """Analyze sentiment using BERT"""
        if not self._initialized:
            return None
        
        try:
            # Tokenize
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                max_length=512,
                truncation=True,
                padding=True
            ).to(self.device)
            
            # Predict
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()[0]
            
            # probs: [negative, neutral, positive]
            score = probs[2] - probs[0]  # positive - negative
            confidence = max(probs)
            
            if probs[2] > probs[0] and probs[2] > probs[1]:
                label = "positive"
            elif probs[0] > probs[2] and probs[0] > probs[1]:
                label = "negative"
            else:
                label = "neutral"
            
            return SentimentResult(score, confidence, label, "bert")
            
        except Exception as e:
            log.warning(f"BERT analysis failed: {e}")
            return None


class SentimentAnalyzer:
    """
    Ensemble sentiment analyzer combining multiple methods
    """
    
    def __init__(self, use_bert: bool = True):
        self.keyword_analyzer = KeywordSentimentAnalyzer()
        self.bert_analyzer = BertSentimentAnalyzer() if use_bert else None
        
        # Weights for ensemble
        self.bert_weight = 0.6
        self.keyword_weight = 0.4
    
    def analyze(self, text: str) -> SentimentResult:
        """Analyze sentiment using ensemble of methods"""
        results = []
        
        # Keyword analysis (always available)
        keyword_result = self.keyword_analyzer.analyze(text)
        results.append(('keyword', keyword_result, self.keyword_weight))
        
        # BERT analysis (if available)
        if self.bert_analyzer and self.bert_analyzer.is_available():
            bert_result = self.bert_analyzer.analyze(text)
            if bert_result:
                results.append(('bert', bert_result, self.bert_weight))
        
        # Weighted ensemble
        if len(results) == 1:
            return results[0][1]
        
        total_weight = sum(w for _, _, w in results)
        
        ensemble_score = sum(r.score * w for _, r, w in results) / total_weight
        ensemble_confidence = sum(r.confidence * w for _, r, w in results) / total_weight
        
        if ensemble_score > 0.2:
            label = "positive"
        elif ensemble_score < -0.2:
            label = "negative"
        else:
            label = "neutral"
        
        return SentimentResult(
            score=ensemble_score,
            confidence=ensemble_confidence,
            label=label,
            method="ensemble"
        )
    
    def analyze_batch(self, texts: List[str]) -> List[SentimentResult]:
        """Analyze multiple texts"""
        return [self.analyze(text) for text in texts]


class NewsScraper:
    """
    News scraper for multiple Chinese financial news sources
    """
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
        self.analyzer = SentimentAnalyzer()
        
        # Cache
        self._cache_dir = CONFIG.DATA_DIR / "news_cache"
        self._cache_dir.mkdir(exist_ok=True)
        self._seen_hashes = self._load_seen_hashes()
        
        # Rate limiting
        self._last_request = {}
        self._rate_limits = {
            'sina': 0.5,
            'eastmoney': 0.5,
            'cls': 0.5,
            '10jqka': 1.0,
        }
    
    def _load_seen_hashes(self) -> set:
        """Load previously seen news hashes"""
        path = self._cache_dir / "seen_hashes.pkl"
        if path.exists():
            try:
                with open(path, 'rb') as f:
                    return pickle.load(f)
            except:
                pass
        return set()
    
    def _save_seen_hashes(self):
        """Save seen hashes"""
        path = self._cache_dir / "seen_hashes.pkl"
        # Keep only recent hashes
        if len(self._seen_hashes) > 10000:
            self._seen_hashes = set(list(self._seen_hashes)[-5000:])
        with open(path, 'wb') as f:
            pickle.dump(self._seen_hashes, f)
    
    def _hash(self, text: str) -> str:
        return hashlib.md5(text.encode()).hexdigest()
    
    def _rate_limit(self, source: str):
        """Apply rate limiting"""
        limit = self._rate_limits.get(source, 0.5)
        last = self._last_request.get(source, 0)
        elapsed = time.time() - last
        if elapsed < limit:
            time.sleep(limit - elapsed)
        self._last_request[source] = time.time()
    
    def _extract_stock_codes(self, text: str) -> List[str]:
        """Extract stock codes from text"""
        patterns = [
            r'[（(]([036]\d{5})[)）]',
            r'(?<!\d)([036]\d{5})(?!\d)',
            r'SH([0-9]{6})',
            r'SZ([0-9]{6})',
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
    
    def scrape_eastmoney(self, max_items: int = 50) -> List[NewsItem]:
        """Scrape East Money news"""
        items = []
        self._rate_limit('eastmoney')
        
        try:
            url = "https://np-listapi.eastmoney.com/comm/wap/getListInfo"
            params = {'type': 1, 'pageSize': max_items, 'pageNo': 1}
            
            resp = self.session.get(url, params=params, timeout=10)
            data = resp.json()
            
            for item in data.get('data', {}).get('list', []):
                title = item.get('title', '')
                content = item.get('digest', '')[:500]
                
                h = self._hash(title + content)
                if h in self._seen_hashes:
                    continue
                self._seen_hashes.add(h)
                
                text = title + " " + content
                sentiment = self.analyzer.analyze(text)
                
                items.append(NewsItem(
                    title=title,
                    content=content,
                    source='eastmoney',
                    url=item.get('url', ''),
                    timestamp=datetime.now(),
                    stock_codes=self._extract_stock_codes(text),
                    sentiment_score=sentiment.score,
                    sentiment_label=sentiment.label
                ))
            
            log.info(f"Scraped {len(items)} items from EastMoney")
            
        except Exception as e:
            log.error(f"EastMoney scraping error: {e}")
        
        return items
    
    def scrape_cls(self, max_items: int = 30) -> List[NewsItem]:
        """Scrape CLS (财联社) news - high quality source"""
        items = []
        self._rate_limit('cls')
        
        try:
            url = "https://www.cls.cn/api/sw"
            params = {'app': 'CailianpressWeb', 'os': 'web', 'sv': '7.7.5'}
            
            resp = self.session.get(
                "https://www.cls.cn/nodeapi/telegraphList",
                params={'page': 1, 'rn': max_items},
                timeout=10
            )
            data = resp.json()
            
            for item in data.get('data', {}).get('roll_data', []):
                title = item.get('title', '') or item.get('brief', '')
                content = item.get('content', '')[:500]
                
                h = self._hash(title + content)
                if h in self._seen_hashes:
                    continue
                self._seen_hashes.add(h)
                
                text = title + " " + content
                sentiment = self.analyzer.analyze(text)
                
                items.append(NewsItem(
                    title=title,
                    content=content,
                    source='cls',
                    url=f"https://www.cls.cn/detail/{item.get('id')}",
                    timestamp=datetime.now(),
                    stock_codes=self._extract_stock_codes(text),
                    sentiment_score=sentiment.score,
                    sentiment_label=sentiment.label
                ))
            
            log.info(f"Scraped {len(items)} items from CLS")
            
        except Exception as e:
            log.error(f"CLS scraping error: {e}")
        
        return items
    
    def scrape_all(self) -> List[NewsItem]:
        """Scrape from all sources"""
        items = []
        
        items.extend(self.scrape_sina())
        items.extend(self.scrape_eastmoney())
        items.extend(self.scrape_cls())
        
        self._save_seen_hashes()
        
        # Sort by timestamp (newest first)
        items.sort(key=lambda x: x.timestamp, reverse=True)
        
        return items
    
    def get_stock_sentiment(self, stock_code: str) -> Tuple[float, float]:
        """
        Get aggregated sentiment for a specific stock
        
        Returns: (score, confidence)
        """
        # Load cached news
        all_news = self.scrape_all()
        
        # Filter for this stock
        relevant = [n for n in all_news if stock_code in n.stock_codes]
        
        if not relevant:
            return 0.0, 0.0
        
        # Weighted average by recency
        now = datetime.now()
        weighted_score = 0
        total_weight = 0
        
        for news in relevant:
            hours_ago = (now - news.timestamp).total_seconds() / 3600
            weight = 1 / (1 + hours_ago / 24)  # Decay over 24 hours
            
            weighted_score += news.sentiment_score * weight
            total_weight += weight
        
        if total_weight == 0:
            return 0.0, 0.0
        
        avg_score = weighted_score / total_weight
        confidence = min(len(relevant) / 5, 1.0)  # More news = more confident
        
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