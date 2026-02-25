"""Bilingual LLM sentiment analyzer with auto-training."""

from __future__ import annotations

import math
import pickle
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

from config.settings import CONFIG
from utils.logger import get_logger

from .news_collector import NewsArticle, get_collector

log = get_logger(__name__)

_TRANSFORMERS_AVAILABLE = False
_SKLEARN_AVAILABLE = False
_SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    from transformers import pipeline

    _TRANSFORMERS_AVAILABLE = True
except Exception:
    _TRANSFORMERS_AVAILABLE = False

try:
    from sklearn.linear_model import LogisticRegression

    _SKLEARN_AVAILABLE = True
except Exception:
    _SKLEARN_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer

    _SENTENCE_TRANSFORMERS_AVAILABLE = True
except Exception:
    _SENTENCE_TRANSFORMERS_AVAILABLE = False


class SentimentLabel(Enum):
    VERY_NEGATIVE = -2
    NEGATIVE = -1
    NEUTRAL = 0
    POSITIVE = 1
    VERY_POSITIVE = 2


@dataclass
class LLMSentimentResult:
    overall: float
    label: SentimentLabel
    confidence: float
    positive_score: float
    negative_score: float
    neutral_score: float
    policy_impact: float
    market_sentiment: float
    entities: list[dict[str, Any]] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    uncertainty: float = 0.0
    model_used: str = ""
    processing_time_ms: float = 0.0
    trader_sentiment: float = 0.0
    discussion_topics: list[str] = field(default_factory=list)
    shared_experiences: list[dict[str, Any]] = field(default_factory=list)
    social_mentions: int = 0
    retail_sentiment: float = 0.0
    institutional_sentiment: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "overall": float(self.overall),
            "label": self.label.name,
            "label_value": int(self.label.value),
            "confidence": float(self.confidence),
            "positive_score": float(self.positive_score),
            "negative_score": float(self.negative_score),
            "neutral_score": float(self.neutral_score),
            "policy_impact": float(self.policy_impact),
            "market_sentiment": float(self.market_sentiment),
            "trader_sentiment": float(self.trader_sentiment),
            "retail_sentiment": float(self.retail_sentiment),
            "institutional_sentiment": float(self.institutional_sentiment),
            "entities": list(self.entities),
            "keywords": list(self.keywords),
            "discussion_topics": list(self.discussion_topics),
            "shared_experiences": list(self.shared_experiences),
            "social_mentions": int(self.social_mentions),
            "uncertainty": float(self.uncertainty),
            "model_used": str(self.model_used),
            "processing_time_ms": float(self.processing_time_ms),
        }


class LLM_sentimentAnalyzer:
    """Backward-compatible class name used by the app."""

    DEFAULT_MODEL = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
    FALLBACK_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"

    ZH_POS = ["利好", "上涨", "增持", "突破", "提振", "支持", "刺激", "复苏"]
    ZH_NEG = ["利空", "下跌", "减持", "处罚", "调查", "风险", "收紧", "下滑"]
    EN_POS = ["bullish", "upside", "support", "stimulus", "growth", "recovery", "positive"]
    EN_NEG = ["bearish", "downside", "tightening", "restriction", "selloff", "risk", "negative"]
    ZH_POLICY_POS = ["政策支持", "减税", "补贴", "降准", "降息", "稳增长"]
    ZH_POLICY_NEG = ["监管收紧", "处罚", "限制", "禁令", "整顿", "审查"]
    EN_POLICY_POS = ["policy support", "tax cut", "subsidy", "easing"]
    EN_POLICY_NEG = ["crackdown", "ban", "sanction", "tightening"]
    ZH_MARKET_BULL = ["牛市", "看多", "做多", "反弹", "强势"]
    ZH_MARKET_BEAR = ["熊市", "看空", "做空", "回调", "弱势"]
    EN_MARKET_BULL = ["bull market", "long", "breakout", "risk-on"]
    EN_MARKET_BEAR = ["bear market", "short", "breakdown", "risk-off"]

    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
        cache_dir: Path | None = None,
        use_gpu: bool = True,
    ) -> None:
        _ = (device, use_gpu)
        self.model_name = str(model_name or self.DEFAULT_MODEL)
        self.cache_dir = cache_dir or (CONFIG.cache_dir / "llm")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._pipe: Any = None
        self._pipe_name = ""
        self._emb: Any = None
        self._calibrator: Any = None
        self._calibrator_path = self.cache_dir / "llm_calibrator.pkl"
        self._cache: dict[str, tuple[LLMSentimentResult, float]] = {}
        self._cache_ttl = 300.0
        self._load_calibrator()

    @staticmethod
    def _clip(x: float, lo: float, hi: float) -> float:
        return float(max(lo, min(hi, x)))

    @staticmethod
    def _safe_float(x: object, default: float = 0.0) -> float:
        try:
            v = float(x)
            if math.isfinite(v):
                return v
        except Exception:
            pass
        return float(default)

    def _detect_language(self, text: str) -> str:
        zh = len(re.findall(r"[\u4e00-\u9fff]", str(text or "")))
        total = len(re.sub(r"\s+", "", str(text or "")))
        if total <= 0:
            return "en"
        return "zh" if (zh / float(total)) >= 0.22 else "en"

    def _kw_score(self, text: str, pos: list[str], neg: list[str]) -> float:
        t = str(text or "").lower()
        p = sum(1 for w in pos if str(w).lower() in t)
        n = sum(1 for w in neg if str(w).lower() in t)
        d = p + n
        if d <= 0:
            return 0.0
        return self._clip((p - n) / float(d), -1.0, 1.0)

    def _load_pipeline(self) -> None:
        if self._pipe is not None and self._pipe_name == self.model_name:
            return
        if not _TRANSFORMERS_AVAILABLE:
            return
        try:
            self._pipe = pipeline(
                "sentiment-analysis",
                model=self.model_name,
                tokenizer=self.model_name,
                return_all_scores=True,
                device=-1,
            )
            self._pipe_name = self.model_name
        except Exception:
            try:
                self._pipe = pipeline(
                    "sentiment-analysis",
                    model=self.FALLBACK_MODEL,
                    tokenizer=self.FALLBACK_MODEL,
                    return_all_scores=True,
                    device=-1,
                )
                self._pipe_name = self.FALLBACK_MODEL
            except Exception:
                self._pipe = None
                self._pipe_name = ""

    def _parse_scores(self, raw: object) -> tuple[float, float, float, float, float]:
        rows: list[dict[str, Any]] = []
        if isinstance(raw, list):
            if raw and isinstance(raw[0], list):
                rows = [r for r in raw[0] if isinstance(r, dict)]
            else:
                rows = [r for r in raw if isinstance(r, dict)]
        pos = 0.0
        neg = 0.0
        neu = 0.0
        for r in rows:
            label = str(r.get("label", "")).lower()
            score = self._safe_float(r.get("score", 0.0), 0.0)
            if "pos" in label or label.endswith("2"):
                pos = max(pos, score)
            elif "neg" in label or label.endswith("0"):
                neg = max(neg, score)
            else:
                neu = max(neu, score)
        rem = max(0.0, 1.0 - pos - neg - neu)
        neu += rem
        total = pos + neg + neu
        if total <= 0:
            return 0.0, 0.0, 0.0, 1.0, 0.0
        pos /= total
        neg /= total
        neu /= total
        overall = self._clip(pos - neg, -1.0, 1.0)
        conf = self._clip(max(pos, neg, neu), 0.0, 1.0)
        return overall, pos, neg, neu, conf

    def _label(self, score: float) -> SentimentLabel:
        if score >= 0.6:
            return SentimentLabel.VERY_POSITIVE
        if score >= 0.2:
            return SentimentLabel.POSITIVE
        if score <= -0.6:
            return SentimentLabel.VERY_NEGATIVE
        if score <= -0.2:
            return SentimentLabel.NEGATIVE
        return SentimentLabel.NEUTRAL

    def _cache_key(self, article: NewsArticle) -> str:
        ts = getattr(article, "published_at", None)
        if isinstance(ts, datetime):
            iso = ts.isoformat()
        else:
            iso = ""
        return f"{getattr(article, 'id', '')}:{iso}"

    def _build_features(self, article: NewsArticle, tf_overall: float, tf_conf: float, language: str) -> list[float]:
        text = f"{getattr(article, 'title', '')} {getattr(article, 'content', '')}"
        if language == "zh":
            rule = self._kw_score(text, self.ZH_POS, self.ZH_NEG)
            policy = self._kw_score(text, self.ZH_POLICY_POS, self.ZH_POLICY_NEG)
            market = self._kw_score(text, self.ZH_MARKET_BULL, self.ZH_MARKET_BEAR)
        else:
            rule = self._kw_score(text, self.EN_POS, self.EN_NEG)
            policy = self._kw_score(text, self.EN_POLICY_POS, self.EN_POLICY_NEG)
            market = self._kw_score(text, self.EN_MARKET_BULL, self.EN_MARKET_BEAR)
        return [
            float(tf_overall),
            float(rule),
            float(policy),
            float(market),
            float(tf_conf),
            1.0 if language == "zh" else 0.0,
            1.0 if language == "en" else 0.0,
            self._safe_float(getattr(article, "relevance_score", 0.5), 0.5),
        ]

    def analyze(self, article: NewsArticle, use_cache: bool = True) -> LLMSentimentResult:
        started = time.time()
        ck = self._cache_key(article)
        now = time.time()
        if use_cache and ck in self._cache:
            old, ts = self._cache[ck]
            if (now - float(ts)) <= self._cache_ttl:
                return old

        text = f"{getattr(article, 'title', '')}. {getattr(article, 'summary', '')}. {getattr(article, 'content', '')}"[:3200]
        language = self._detect_language(text)

        self._load_pipeline()
        tf_overall, tf_pos, tf_neg, tf_neu, tf_conf = 0.0, 0.0, 0.0, 1.0, 0.38
        model_used = "rule"
        if self._pipe is not None:
            try:
                tf_overall, tf_pos, tf_neg, tf_neu, tf_conf = self._parse_scores(self._pipe(text[:512]))
                model_used = self._pipe_name or self.model_name
            except Exception:
                pass

        feat = self._build_features(article, tf_overall=tf_overall, tf_conf=tf_conf, language=language)
        rule = float(feat[1])
        policy = float(feat[2])
        market = float(feat[3])
        overall = self._clip((0.72 * tf_overall) + (0.28 * rule), -1.0, 1.0)
        if str(getattr(article, "category", "")).lower() == "policy":
            overall = self._clip((0.55 * overall) + (0.45 * policy), -1.0, 1.0)

        if self._calibrator is not None:
            try:
                probs = self._calibrator.predict_proba(np.asarray([feat], dtype=float))[0]
                classes = list(self._calibrator.classes_)
                p_pos = sum(float(probs[i]) for i, c in enumerate(classes) if int(c) > 0)
                p_neg = sum(float(probs[i]) for i, c in enumerate(classes) if int(c) < 0)
                cal = self._clip(p_pos - p_neg, -1.0, 1.0)
                overall = self._clip((0.75 * overall) + (0.25 * cal), -1.0, 1.0)
                tf_conf = self._clip((0.8 * tf_conf) + (0.2 * max(float(x) for x in probs)), 0.0, 1.0)
            except Exception:
                pass

        entities = [{"text": c, "type": "stock_code", "confidence": 0.95} for c in re.findall(r"\b(\d{6})\b", text)]
        kw_pool = (self.ZH_POS + self.ZH_NEG + self.ZH_POLICY_POS + self.ZH_POLICY_NEG) if language == "zh" else (self.EN_POS + self.EN_NEG + self.EN_POLICY_POS + self.EN_POLICY_NEG)
        keywords = [w for w in kw_pool if str(w).lower() in str(text).lower()][:40]

        signal_strength = max(abs(float(overall)), abs(float(policy)), abs(float(market)))
        conf = self._clip(
            (0.55 * tf_conf)
            + (0.25 * signal_strength)
            + (0.10 * (1.0 - min(1.0, abs(policy - market))))
            + 0.10,
            0.0,
            1.0,
        )
        result = LLMSentimentResult(
            overall=float(overall),
            label=self._label(float(overall)),
            confidence=float(conf),
            positive_score=float(max(0.0, overall)),
            negative_score=float(max(0.0, -overall)),
            neutral_score=float(max(0.0, 1.0 - abs(overall))),
            policy_impact=float(policy),
            market_sentiment=float(market),
            entities=entities,
            keywords=keywords,
            uncertainty=float(1.0 - conf),
            model_used=model_used,
            processing_time_ms=float((time.time() - started) * 1000.0),
            trader_sentiment=float(overall),
            discussion_topics=["policy"] if ("policy" in text.lower() or "政策" in text) else [],
            social_mentions=sum(1 for tok in ("xueqiu", "雪球", "reddit", "微博") if tok in text.lower()),
            retail_sentiment=float(overall * 0.85),
            institutional_sentiment=float(overall * 0.8),
        )
        self._cache[ck] = (result, now)
        return result

    def analyze_batch(self, articles: list[NewsArticle], batch_size: int = 8) -> list[LLMSentimentResult]:
        _ = batch_size
        return [self.analyze(a) for a in list(articles or [])]

    def _load_calibrator(self) -> None:
        self._calibrator = None
        if not self._calibrator_path.exists():
            return
        try:
            with self._calibrator_path.open("rb") as f:
                self._calibrator = pickle.load(f)
        except Exception:
            self._calibrator = None

    def _save_calibrator(self) -> None:
        if self._calibrator is None:
            return
        try:
            with self._calibrator_path.open("wb") as f:
                pickle.dump(self._calibrator, f)
        except Exception:
            pass

    def train(self, articles: list[NewsArticle], *, epochs: int = 3, max_samples: int = 1000, learning_rate: float = 2e-5) -> dict[str, Any]:
        _ = (epochs, learning_rate)
        t0 = time.time()
        start = datetime.now().isoformat()
        rows = list(articles or [])[: max(50, int(max_samples))]
        zh = 0
        en = 0
        if not rows:
            return {
                "status": "skipped",
                "model_name": self.model_name,
                "trained_samples": 0,
                "zh_samples": 0,
                "en_samples": 0,
                "started_at": start,
                "finished_at": datetime.now().isoformat(),
                "duration_seconds": float(time.time() - t0),
                "notes": "No articles provided.",
            }

        x: list[list[float]] = []
        y: list[int] = []
        for a in rows:
            lang = str(getattr(a, "language", "") or "")
            if not lang:
                lang = self._detect_language(f"{getattr(a, 'title', '')} {getattr(a, 'content', '')}")
            if lang == "zh":
                zh += 1
            else:
                en += 1
            text = f"{getattr(a, 'title', '')} {getattr(a, 'content', '')}"
            base = self._kw_score(text, self.ZH_POS if lang == 'zh' else self.EN_POS, self.ZH_NEG if lang == 'zh' else self.EN_NEG)
            lbl = 1 if base >= 0.08 else (-1 if base <= -0.08 else 0)
            x.append(self._build_features(a, tf_overall=base, tf_conf=0.4, language=lang))
            y.append(lbl)

        status = "trained"
        notes = "Calibration updated."
        if not _SKLEARN_AVAILABLE:
            status = "partial"
            notes = "scikit-learn unavailable; using base model only."
        elif len(set(y)) < 2:
            status = "partial"
            notes = "Not enough class diversity for calibration fit."
        else:
            try:
                clf = LogisticRegression(max_iter=320, multi_class="auto", class_weight="balanced")
                clf.fit(np.asarray(x, dtype=float), np.asarray(y, dtype=int))
                self._calibrator = clf
                self._save_calibrator()
            except Exception as exc:
                status = "partial"
                notes = f"Calibration fit failed: {exc}"

        return {
            "status": status,
            "model_name": self.model_name,
            "trained_samples": int(len(rows)),
            "zh_samples": int(zh),
            "en_samples": int(en),
            "started_at": start,
            "finished_at": datetime.now().isoformat(),
            "duration_seconds": float(time.time() - t0),
            "notes": notes,
        }

    def auto_train_from_internet(self, *, hours_back: int = 96, limit_per_query: int = 180, max_samples: int = 1200) -> dict[str, Any]:
        collector = get_collector()
        queries = [
            ["政策", "监管", "A股", "央行"],
            ["上市公司", "业绩", "产业政策"],
            ["China stock policy", "regulation", "market"],
            ["Federal Reserve", "SEC", "China ADR"],
        ]
        seen: set[str] = set()
        rows: list[NewsArticle] = []
        for kw in queries:
            try:
                batch = collector.collect_news(
                    keywords=list(kw),
                    limit=max(20, int(limit_per_query)),
                    hours_back=max(12, int(hours_back)),
                )
            except Exception:
                batch = []
            for a in batch:
                aid = str(getattr(a, "id", "") or "")
                if not aid or aid in seen:
                    continue
                seen.add(aid)
                rows.append(a)
        rows.sort(key=lambda a: getattr(a, "published_at", datetime.min), reverse=True)
        rows = rows[: max(80, int(max_samples))]
        report = self.train(rows, max_samples=max_samples)
        report["collected_articles"] = int(len(rows))
        report["hours_back"] = int(hours_back)
        report["limit_per_query"] = int(limit_per_query)
        return report

    def summarize_articles(self, articles: list[NewsArticle], *, hours_back: int = 48) -> dict[str, float]:
        if not articles:
            return {"overall": 0.0, "policy": 0.0, "market": 0.0, "confidence": 0.0, "article_count": 0.0, "zh_ratio": 0.0, "en_ratio": 0.0}
        cutoff = datetime.now() - timedelta(hours=max(1, int(hours_back)))
        recent = [a for a in articles if isinstance(getattr(a, "published_at", None), datetime) and a.published_at >= cutoff]
        if not recent:
            recent = list(articles)
        scores = self.analyze_batch(recent)
        if not scores:
            return {"overall": 0.0, "policy": 0.0, "market": 0.0, "confidence": 0.0, "article_count": 0.0, "zh_ratio": 0.0, "en_ratio": 0.0}
        zh = 0
        for a in recent:
            lang = str(getattr(a, "language", "") or "").lower()
            if lang == "zh":
                zh += 1
        total = float(len(recent))
        return {
            "overall": float(np.mean([s.overall for s in scores])),
            "policy": float(np.mean([s.policy_impact for s in scores])),
            "market": float(np.mean([s.market_sentiment for s in scores])),
            "confidence": float(np.mean([s.confidence for s in scores])),
            "article_count": float(len(recent)),
            "zh_ratio": float(zh / max(1.0, total)),
            "en_ratio": float((len(recent) - zh) / max(1.0, total)),
        }

    def get_embedding(self, text: str) -> list[float]:
        if _SENTENCE_TRANSFORMERS_AVAILABLE:
            if self._emb is None:
                try:
                    self._emb = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
                except Exception:
                    self._emb = None
            if self._emb is not None:
                try:
                    out = self._emb.encode(str(text or ""))
                    return [float(v) for v in np.asarray(out, dtype=float).tolist()]
                except Exception:
                    pass
        vec = np.zeros(96, dtype=float)
        for tok in re.findall(r"[a-zA-Z0-9\u4e00-\u9fff]+", str(text or "").lower()):
            vec[hash(tok) % vec.size] += 1.0
        n = float(np.linalg.norm(vec))
        if n > 0:
            vec /= n
        return [float(v) for v in vec.tolist()]

    def find_similar_articles(self, query: str, articles: list[NewsArticle], top_k: int = 5) -> list[tuple[NewsArticle, float]]:
        q = np.asarray(self.get_embedding(query), dtype=float)
        qn = float(np.linalg.norm(q))
        if qn <= 0:
            return []
        out: list[tuple[NewsArticle, float]] = []
        for a in list(articles or []):
            txt = f"{getattr(a, 'title', '')}. {getattr(a, 'content', '')[:500]}"
            v = np.asarray(self.get_embedding(txt), dtype=float)
            vn = float(np.linalg.norm(v))
            if vn <= 0 or v.shape != q.shape:
                continue
            out.append((a, float(np.dot(q, v) / max(1e-12, qn * vn))))
        out.sort(key=lambda x: x[1], reverse=True)
        return out[: max(1, int(top_k))]


_analyzer: LLM_sentimentAnalyzer | None = None


def get_llm_analyzer() -> LLM_sentimentAnalyzer:
    global _analyzer
    if _analyzer is None:
        _analyzer = LLM_sentimentAnalyzer()
    return _analyzer


def reset_llm_analyzer() -> None:
    global _analyzer
    _analyzer = None
