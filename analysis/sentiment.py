# analysis/sentiment.py
from __future__ import annotations

import hashlib
import json
import pickle
import re
import socket
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime

import requests

from config.settings import CONFIG
from utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class NewsItem:
    """News article"""

    title: str
    content: str
    source: str
    url: str
    timestamp: datetime
    stock_codes: list[str]
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
    """Keyword-based sentiment analysis for financial text."""

    POSITIVE = {
        "surge": 3,
        "limit up": 3,
        "breakout": 3,
        "beats estimate": 3,
        "record high": 3,
        "uptrend": 2,
        "upgrade": 2,
        "buy rating": 2,
        "outperform": 2,
        "growth": 1,
        "recovery": 1,
        "guidance raised": 2,
        "profit rise": 2,
    }
    NEGATIVE = {
        "crash": 3,
        "limit down": 3,
        "misses estimate": 3,
        "fraud": 3,
        "default": 3,
        "downgrade": 2,
        "sell rating": 2,
        "warning": 2,
        "decline": 2,
        "lawsuit": 2,
        "loss": 2,
        "weak guidance": 2,
        "risk": 1,
        "volatility spike": 1,
    }
    NEGATIONS = ("not", "no", "never", "without")

    @staticmethod
    def _normalize(text: str) -> str:
        return re.sub(r"\s+", " ", str(text or "").lower()).strip()

    def analyze(self, text: str) -> SentimentResult:
        norm = self._normalize(text)
        if not norm:
            return SentimentResult(0.0, 0.0, "neutral", "keyword")

        pos_score = 0.0
        neg_score = 0.0
        tokens = norm.split(" ")

        for word, weight in self.POSITIVE.items():
            if word in norm:
                pos_score += weight
        for word, weight in self.NEGATIVE.items():
            if word in norm:
                neg_score += weight

        # Simple negation flip in a small token window.
        for idx, tok in enumerate(tokens):
            if tok not in self.NEGATIONS:
                continue
            window = " ".join(tokens[idx : idx + 4])
            for word, weight in self.POSITIVE.items():
                if word in window:
                    pos_score -= 0.6 * weight
                    neg_score += 0.6 * weight
            for word, weight in self.NEGATIVE.items():
                if word in window:
                    neg_score -= 0.6 * weight
                    pos_score += 0.6 * weight

        total = pos_score + neg_score
        if total <= 0:
            return SentimentResult(0.0, 0.2, "neutral", "keyword")

        score = (pos_score - neg_score) / total
        confidence = min(total / 10.0, 1.0)

        if score > 0.2:
            label = "positive"
        elif score < -0.2:
            label = "negative"
        else:
            label = "neutral"
        return SentimentResult(float(score), float(confidence), label, "keyword")


class SentimentAnalyzer:
    """Ensemble sentiment analyzer."""

    def __init__(
        self,
        use_bert: bool = False,
        hf_model: str = "IDEA-CCNL/Erlangshen-Roberta-110M-Sentiment",
    ):
        self.keyword_analyzer = KeywordSentimentAnalyzer()
        self._use_bert = bool(use_bert)
        self._hf = None
        self._hf_model = hf_model

        if self._use_bert:
            try:
                from transformers import pipeline

                self._hf = pipeline("sentiment-analysis", model=hf_model)
            except Exception as e:
                log.warning(f"Failed to load HuggingFace model: {e}")
                self._hf = None
                self._use_bert = False

    def analyze(self, text: str) -> SentimentResult:
        kw = self.keyword_analyzer.analyze(text)
        if not self._use_bert or self._hf is None or not text:
            return kw

        try:
            out = self._hf(str(text)[:512])[0]
            label = str(out.get("label", "")).lower()
            score = float(out.get("score", 0.5))

            if "neg" in label:
                hf_score = -score
            elif "pos" in label:
                hf_score = score
            else:
                hf_score = 0.0

            blended = 0.55 * kw.score + 0.45 * hf_score
            confidence = min(1.0, 0.6 * kw.confidence + 0.4 * score)

            if blended > 0.2:
                lbl = "positive"
            elif blended < -0.2:
                lbl = "negative"
            else:
                lbl = "neutral"

            return SentimentResult(float(blended), float(confidence), lbl, "keyword+hf")
        except Exception:
            return kw

    def analyze_batch(self, texts: list[str]) -> list[SentimentResult]:
        return [self.analyze(text) for text in texts]


class NewsScraper:
    """Enhanced news scraper with China network optimization.
    
    Features:
    - Multiple Chinese news providers (Sina, EastMoney, Jin10, Xueqiu, Yahoo China)
    - China network quality scoring and automatic failover
    - Proxy support for restricted content
    - Optimized timeouts for China ISP conditions
    - Social sentiment from Chinese platforms
    """

    def __init__(self):
        # Use optimized session from China network module
        try:
            from core.china_network import get_best_endpoint, get_optimized_session
            self._china_optimized = True
            self._get_provider_session = lambda name: get_optimized_session(name)
            self._get_best_endpoint = get_best_endpoint
        except ImportError:
            self._china_optimized = False
            self._get_provider_session = lambda name: requests.Session()
            self._get_best_endpoint = lambda name: ""

        self.session = requests.Session()
        self.session.headers.update(
            {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        )
        self.analyzer = SentimentAnalyzer()

        self._cache_dir = CONFIG.data_dir / "news_cache"
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._seen_hashes: set[str] = set()
        self._seen_path = self._cache_dir / "seen_hashes.json"
        self._load_seen_hashes()

        self._last_request: dict[str, float] = {}
        self._rate_limits = {
            "sina": 0.5,
            "eastmoney": 0.7,
            "jin10": 0.3,
            "xueqiu": 1.0,
            "yahoo_cn": 0.5,
        }
        self._provider_weights: dict[str, float] = {
            "sina": 1.0,
            "eastmoney": 1.1,
            "jin10": 1.2,      # Fastest financial news
            "xueqiu": 1.0,     # Social sentiment
            "yahoo_cn": 0.9,
        }
        self._providers: dict[str, Callable[[], list[NewsItem]]] = {
            "sina": lambda: self.scrape_sina(),
            "eastmoney": lambda: self.scrape_eastmoney(),
            "jin10": lambda: self.scrape_jin10(),
            "xueqiu": lambda: self.scrape_xueqiu(),
        }
        self._network_check_ttl_s = 30.0
        self._last_network_check_ts = 0.0
        self._network_available_cache = True
        
        # China-specific timeout optimization
        base_timeout = float(getattr(CONFIG.data, "request_timeout", 3))
        self._request_timeout = max(2.0, min(5.0, base_timeout))  # Longer timeout for China
        
        # Initialize China network optimization
        if self._china_optimized:
            self._init_china_network()

    def register_provider(
        self,
        name: str,
        fetcher: Callable[[], list[NewsItem]],
        weight: float = 1.0,
    ) -> None:
        norm = str(name or "").strip().lower()
        if not norm:
            raise ValueError("provider name cannot be empty")
        if not callable(fetcher):
            raise TypeError("fetcher must be callable")
        self._providers[norm] = fetcher
        self._provider_weights[norm] = float(max(0.1, weight))

    def unregister_provider(self, name: str) -> bool:
        norm = str(name or "").strip().lower()
        if norm in ("sina", "eastmoney"):
            return False
        removed = self._providers.pop(norm, None)
        self._provider_weights.pop(norm, None)
        return removed is not None

    def get_provider_weights(self) -> dict[str, float]:
        return dict(self._provider_weights)

    def _load_seen_hashes(self) -> None:
        try:
            if not self._seen_path.exists():
                return
            data = json.loads(self._seen_path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                self._seen_hashes = {str(x) for x in data if x}
        except Exception:
            self._seen_hashes = set()

    def _save_seen_hashes(self) -> None:
        try:
            data = list(self._seen_hashes)[-5000:]
            self._seen_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        except Exception:
            pass

    def _hash(self, text: str) -> str:
        return hashlib.md5(text.encode("utf-8")).hexdigest()

    def _rate_limit(self, source: str) -> None:
        limit = self._rate_limits.get(source, 0.5)
        last = self._last_request.get(source, 0.0)
        elapsed = time.time() - last
        if elapsed < limit:
            time.sleep(limit - elapsed)
        self._last_request[source] = time.time()

    def _extract_stock_codes(self, text: str) -> list[str]:
        patterns = [
            r"[\(\[]([036]\d{5})[\)\]]",
            r"(?<!\d)([036]\d{5})(?!\d)",
        ]
        codes = set()
        for p in patterns:
            codes.update(re.findall(p, str(text)))
        return list(codes)

    def _network_available(self) -> bool:
        """Fast connectivity probe for external news providers.

        Cached for a short TTL so offline environments do not repeatedly pay
        provider-level HTTP timeout costs.
        """
        now = time.time()
        if (now - self._last_network_check_ts) < self._network_check_ttl_s:
            return bool(self._network_available_cache)

        hosts = ("feed.mix.sina.com.cn", "np-anotice-stock.eastmoney.com")
        ok = False
        for host in hosts:
            try:
                with socket.create_connection((host, 443), timeout=0.8):
                    ok = True
                    break
            except OSError:
                continue

        self._last_network_check_ts = now
        self._network_available_cache = ok
        return ok

    def _init_china_network(self) -> None:
        """Initialize China network optimization."""
        try:
            from core.china_network import get_optimizer
            optimizer = get_optimizer()
            
            # Run initial probe for news providers
            for provider in ["sina", "eastmoney", "jin10", "xueqiu"]:
                optimizer.run_endpoint_probe(provider)
        except Exception as e:
            log.debug(f"China network init skipped: {e}")

    def scrape_jin10(self, max_items: int = 40) -> list[NewsItem]:
        """Scrape Jin10 financial news (fastest China financial news service).
        
        Jin10 provides real-time financial news and announcements.
        """
        items: list[NewsItem] = []
        self._rate_limit("jin10")
        
        try:
            # Use China-optimized endpoint
            if self._china_optimized:
                base_url = self._get_best_endpoint("jin10")
            else:
                base_url = "https://api.jin10.com"
            
            if not base_url:
                base_url = "https://api.jin10.com"
            
            url = f"{base_url}/news_service/v1/news"
            params = {
                "category": "stock",
                "limit": max_items,
                "channel": "cn",
            }
            
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                "Accept": "application/json",
                "Referer": "https://www.jin10.com",
            }
            
            resp = self.session.get(url, params=params, headers=headers, timeout=self._request_timeout)
            
            if resp.status_code != 200:
                return items
            
            data = resp.json()
            news_list = data.get("data", []) if isinstance(data, dict) else []
            
            for item in news_list[:max_items]:
                title = str(item.get("title", "") or item.get("content", ""))
                content = str(item.get("content", "") or "")[:500]
                
                if not title:
                    continue
                
                h = self._hash(f"jin10::{title}::{content}")
                if h in self._seen_hashes:
                    continue
                self._seen_hashes.add(h)
                
                text = f"{title} {content}"
                sentiment = self.analyzer.analyze(text)
                
                # Parse timestamp
                ts_str = item.get("time", "")
                try:
                    # Jin10 uses ISO format or Unix timestamp
                    if ts_str.isdigit():
                        ts = datetime.fromtimestamp(int(ts_str) / 1000.0)
                    else:
                        ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                except Exception:
                    ts = datetime.now()
                
                items.append(
                    NewsItem(
                        title=title,
                        content=content,
                        source="jin10",
                        url=str(item.get("url", "") or f"https://www.jin10.com/news/{item.get('id', '')}"),
                        timestamp=ts,
                        stock_codes=self._extract_stock_codes(text),
                        sentiment_score=sentiment.score,
                        sentiment_label=sentiment.label,
                    )
                )
                
        except Exception as e:
            log.debug(f"Jin10 scraping error: {e}")
        
        return items

    def scrape_xueqiu(self, max_items: int = 30) -> list[NewsItem]:
        """Scrape Xueqiu (Snowball) social sentiment.
        
        Xueqiu is China's leading social investment platform.
        """
        items: list[NewsItem] = []
        self._rate_limit("xueqiu")
        
        try:
            # Use China-optimized endpoint
            if self._china_optimized:
                base_url = self._get_best_endpoint("xueqiu")
            else:
                base_url = "https://xueqiu.com"
            
            if not base_url:
                base_url = "https://xueqiu.com"
            
            # First get cookies
            self.session.get(f"{base_url}/", timeout=self._request_timeout)
            
            # Get hot posts
            url = f"{base_url}/v4/statuses/hot_list.json"
            params = {
                "size": max_items,
                "type": "0",
            }
            
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                "Accept": "application/json",
                "X-Requested-With": "XMLHttpRequest",
            }
            
            resp = self.session.get(url, params=params, headers=headers, timeout=self._request_timeout)
            
            if resp.status_code != 200:
                return items
            
            data = resp.json()
            posts = data.get("list", []) if isinstance(data, dict) else []
            
            for post in posts[:max_items]:
                title = str(post.get("title", "") or post.get("text", "")[:100])
                content = str(post.get("text", "") or "")[:500]
                
                if not title:
                    continue
                
                h = self._hash(f"xueqiu::{title}::{content}")
                if h in self._seen_hashes:
                    continue
                self._seen_hashes.add(h)
                
                text = f"{title} {content}"
                sentiment = self.analyzer.analyze(text)
                
                # Parse timestamp
                ts_ms = post.get("created_at")
                try:
                    if ts_ms and str(ts_ms).isdigit():
                        ts = datetime.fromtimestamp(int(ts_ms) / 1000.0)
                    else:
                        ts = datetime.now()
                except Exception:
                    ts = datetime.now()
                
                # Extract stock symbols mentioned
                stock_codes = self._extract_stock_codes(text)
                
                # Also get from mentioned stocks in post
                symbols = post.get("symbols", [])
                if isinstance(symbols, list):
                    for sym in symbols:
                        if isinstance(sym, dict):
                            symbol_code = str(sym.get("symbol", "")).strip("SHSZ")
                            if symbol_code and symbol_code not in stock_codes:
                                stock_codes.append(symbol_code)
                
                items.append(
                    NewsItem(
                        title=title,
                        content=content,
                        source="xueqiu",
                        url=f"{base_url}/status/{post.get('id', '')}",
                        timestamp=ts,
                        stock_codes=stock_codes,
                        sentiment_score=sentiment.score,
                        sentiment_label=sentiment.label,
                    )
                )
                
        except Exception as e:
            log.debug(f"Xueqiu scraping error: {e}")
        
        return items

    def scrape_sina(self, max_items: int = 50) -> list[NewsItem]:
        items: list[NewsItem] = []
        self._rate_limit("sina")
        try:
            url = "https://feed.mix.sina.com.cn/api/roll/get"
            params = {"pageid": 153, "lid": 2516, "num": max_items}
            resp = self.session.get(url, params=params, timeout=self._request_timeout)
            data = resp.json()
            for item in data.get("result", {}).get("data", []):
                title = str(item.get("title") or "")
                content = str(item.get("intro") or "")[:500]
                if not title:
                    continue
                h = self._hash(f"sina::{title}::{content}")
                if h in self._seen_hashes:
                    continue
                self._seen_hashes.add(h)
                text = f"{title} {content}"
                sentiment = self.analyzer.analyze(text)

                raw_ts = str(item.get("ctime") or "")
                try:
                    ts = datetime.strptime(raw_ts, "%Y-%m-%d %H:%M:%S")
                except Exception:
                    ts = datetime.now()

                items.append(
                    NewsItem(
                        title=title,
                        content=content,
                        source="sina",
                        url=str(item.get("url") or ""),
                        timestamp=ts,
                        stock_codes=self._extract_stock_codes(text),
                        sentiment_score=sentiment.score,
                        sentiment_label=sentiment.label,
                    )
                )
        except Exception as e:
            log.error(f"Sina scraping error: {e}")
        return items

    def scrape_eastmoney(self, max_items: int = 30) -> list[NewsItem]:
        items: list[NewsItem] = []
        self._rate_limit("eastmoney")
        try:
            url = "https://np-anotice-stock.eastmoney.com/api/security/ann"
            params = {
                "sr": -1,
                "page_size": max(10, min(100, int(max_items))),
                "page_index": 1,
                "ann_type": "A",
                "client_source": "web",
            }
            resp = self.session.get(url, params=params, timeout=self._request_timeout)
            data = resp.json()
            rows = data.get("data", {}).get("list", []) if isinstance(data, dict) else []
            for item in rows:
                title = str(item.get("title") or "")
                content = str(item.get("columns") or "")[:500]
                if not title:
                    continue
                h = self._hash(f"eastmoney::{title}::{content}")
                if h in self._seen_hashes:
                    continue
                self._seen_hashes.add(h)
                text = f"{title} {content}"
                sentiment = self.analyzer.analyze(text)

                ts_ms = item.get("notice_date")
                try:
                    ts = datetime.fromtimestamp(float(ts_ms) / 1000.0) if ts_ms else datetime.now()
                except Exception:
                    ts = datetime.now()

                items.append(
                    NewsItem(
                        title=title,
                        content=content,
                        source="eastmoney",
                        url=str(item.get("art_code") or ""),
                        timestamp=ts,
                        stock_codes=self._extract_stock_codes(text),
                        sentiment_score=sentiment.score,
                        sentiment_label=sentiment.label,
                    )
                )
        except Exception as e:
            log.debug(f"Eastmoney scraping skipped: {e}")
        return items

    def scrape_all(self) -> list[NewsItem]:
        cache_path = self._cache_dir / "scrape_all.pkl"
        ttl_seconds = 180
        try:
            if cache_path.exists():
                mtime = cache_path.stat().st_mtime
                if (time.time() - mtime) < ttl_seconds:
                    with open(cache_path, "rb") as f:
                        items = pickle.load(f)
                    if isinstance(items, list):
                        return items
        except Exception:
            pass

        items: list[NewsItem] = []
        for name, fetcher in list(self._providers.items()):
            if name in ("sina", "eastmoney") and not self._network_available():
                continue
            try:
                out = fetcher()
                if isinstance(out, list):
                    for row in out:
                        if isinstance(row, NewsItem) and not row.source:
                            row.source = name
                    items.extend([x for x in out if isinstance(x, NewsItem)])
            except Exception as e:
                log.debug(f"Provider {name} failed: {e}")
        items.sort(key=lambda x: x.timestamp, reverse=True)
        self._save_seen_hashes()

        try:
            with open(cache_path, "wb") as f:
                pickle.dump(items, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception:
            pass
        return items

    def get_stock_sentiment(self, stock_code: str) -> tuple[float, float]:
        code = str(stock_code or "").strip()
        custom_sources = [
            name for name in self._providers.keys()
            if name not in ("sina", "eastmoney")
        ]

        # Fast path: if custom providers are registered and they provide
        # relevant records, avoid external provider latency.
        if custom_sources:
            custom_news: list[NewsItem] = []
            for name in custom_sources:
                fetcher = self._providers.get(name)
                if fetcher is None:
                    continue
                try:
                    out = fetcher()
                    if isinstance(out, list):
                        for row in out:
                            if isinstance(row, NewsItem):
                                if not row.source:
                                    row.source = name
                                custom_news.append(row)
                except Exception as e:
                    log.debug(f"Custom provider {name} failed: {e}")
            relevant_custom = [n for n in custom_news if code in n.stock_codes]
            if relevant_custom:
                return self._aggregate_stock_sentiment(relevant_custom)

        all_news = self.scrape_all()
        relevant = [n for n in all_news if code in n.stock_codes]
        if not relevant:
            return 0.0, 0.0
        return self._aggregate_stock_sentiment(relevant)

    def _aggregate_stock_sentiment(self, relevant: list[NewsItem]) -> tuple[float, float]:
        now = datetime.now()
        weighted_score = 0.0
        total_weight = 0.0
        for news in relevant:
            hours_ago = max(0.0, (now - news.timestamp).total_seconds() / 3600.0)
            src_w = float(self._provider_weights.get(news.source, 1.0))
            weight = src_w * (1.0 / (1.0 + hours_ago / 24.0))
            weighted_score += float(news.sentiment_score) * weight
            total_weight += weight

        if total_weight <= 0:
            return 0.0, 0.0

        avg_score = weighted_score / total_weight
        source_count = len({n.source for n in relevant})
        breadth_boost = min(source_count / 3.0, 1.0)
        confidence = min(1.0, 0.7 * min(len(relevant) / 8.0, 1.0) + 0.3 * breadth_boost)
        return float(avg_score), float(confidence)

    def get_market_sentiment(self) -> dict:
        all_news = self.scrape_all()
        if not all_news:
            return {"score": 0.0, "label": "neutral", "news_count": 0}

        now = datetime.now()
        weighted = []
        for news in all_news:
            age_h = max(0.0, (now - news.timestamp).total_seconds() / 3600.0)
            recency_w = float(self._provider_weights.get(news.source, 1.0)) * (1.0 / (1.0 + age_h / 24.0))
            weighted.append((float(news.sentiment_score), recency_w))

        total_w = sum(w for _, w in weighted)
        avg_score = (sum(s * w for s, w in weighted) / total_w) if total_w > 0 else 0.0

        pos_count = sum(1 for n in all_news if n.sentiment_label == "positive")
        neg_count = sum(1 for n in all_news if n.sentiment_label == "negative")

        if avg_score > 0.15:
            label = "bullish"
        elif avg_score < -0.15:
            label = "bearish"
        else:
            label = "neutral"

        return {
            "score": float(avg_score),
            "label": label,
            "news_count": len(all_news),
            "positive_count": pos_count,
            "negative_count": neg_count,
            "neutral_count": len(all_news) - pos_count - neg_count,
            "source_count": len({n.source for n in all_news}),
        }
