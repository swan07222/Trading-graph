"""Bilingual LLM sentiment analyzer with auto-training."""

from __future__ import annotations

import json
import math
import os
import pickle
import re
import hashlib
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

from config.settings import CONFIG
from config.runtime_env import env_flag
from utils.logger import get_logger

from .news_collector import NewsArticle, get_collector, reset_collector

log = get_logger(__name__)

# FIX: Define ConvergenceWarning for sklearn compatibility
try:
    from sklearn.exceptions import ConvergenceWarning
except ImportError:
    # Fallback: define a dummy exception class
    class ConvergenceWarning(Exception):
        """Dummy ConvergenceWarning for sklearn compatibility."""
        pass

_TRANSFORMERS_AVAILABLE = False
_SKLEARN_AVAILABLE = False
_SKLEARN_MLP_AVAILABLE = False
_SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    import warnings
    warnings.filterwarnings('ignore', category=ConvergenceWarning)
except Exception:
    pass

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
    from sklearn.neural_network import MLPClassifier

    _SKLEARN_MLP_AVAILABLE = True
except Exception:
    _SKLEARN_MLP_AVAILABLE = False

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
    _SEEN_ARTICLE_TTL_SECONDS = 7 * 24 * 3600
    _SEEN_ARTICLE_MAX = 100000

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
        cache_ttl_seconds: float | None = None,
        max_seen_articles: int | None = None,
    ) -> None:
        self.model_name = str(model_name or self.DEFAULT_MODEL)
        default_dir = getattr(
            CONFIG,
            "llm_model_dir",
            (CONFIG.model_dir.parent / "LLM"),
        )
        self.cache_dir = Path(cache_dir or default_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._pipe: Any = None
        self._pipe_name = ""
        self._emb: Any = None
        self._calibrator: Any = None
        self._hybrid_calibrator: Any = None
        self._calibrator_path = self.cache_dir / "llm_calibrator.pkl"
        self._hybrid_calibrator_path = self.cache_dir / "llm_hybrid_nn.pkl"
        self._status_path = self.cache_dir / "llm_training_status.json"
        self._seen_article_path = self.cache_dir / "llm_seen_articles.json"
        self._seen_articles: dict[str, float] = {}
        # FIX 9: Configurable cache TTL with longer default (1 hour instead of 5 minutes)
        self._cache_ttl = float(cache_ttl_seconds if cache_ttl_seconds is not None else 3600.0)
        # FIX 10: Configurable max seen articles with memory limit
        self._max_seen_articles = int(max_seen_articles if max_seen_articles is not None else self._SEEN_ARTICLE_MAX)
        self._seen_article_ttl_seconds = float(self._SEEN_ARTICLE_TTL_SECONDS)
        self._cache: dict[str, tuple[LLMSentimentResult, float]] = {}
        self._lock = threading.RLock()
        # FIX 8: Proper GPU device handling
        self._device = self._init_device(device, use_gpu)
        self._load_calibrator()
        self._load_hybrid_calibrator()
        self._load_seen_articles()

    @staticmethod
    def _init_device(device: str | None, use_gpu: bool) -> str:
        """Initialize device for model inference.

        Args:
            device: Explicit device specification (e.g., 'cuda:0', 'cpu')
            use_gpu: Whether to use GPU if available

        Returns:
            Device string for transformers pipeline
        """
        if device is not None:
            return device

        if use_gpu:
            try:
                import torch
                if torch.cuda.is_available():
                    return "cuda:0"
                elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    return "mps"  # Apple Silicon
            except ImportError:
                pass

        return "cpu"

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
        """Load transformer pipeline with proper device handling."""
        if self._pipe is not None and self._pipe_name == self.model_name:
            return
        if not _TRANSFORMERS_AVAILABLE:
            return
        try:
            # Convert device string to device ID for transformers
            device_id = -1  # CPU default
            if self._device.startswith("cuda"):
                try:
                    device_id = int(self._device.split(":")[1]) if ":" in self._device else 0
                except (ValueError, IndexError):
                    device_id = 0
            elif self._device == "mps":
                device_id = "mps"

            self._pipe = pipeline(
                "sentiment-analysis",
                model=self.model_name,
                tokenizer=self.model_name,
                return_all_scores=True,
                device=device_id,
            )
            self._pipe_name = self.model_name
        except Exception as e:
            log.debug("Primary model load failed: %s", e)
            try:
                self._pipe = pipeline(
                    "sentiment-analysis",
                    model=self.FALLBACK_MODEL,
                    tokenizer=self.FALLBACK_MODEL,
                    return_all_scores=True,
                    device=-1,  # CPU for fallback
                )
                self._pipe_name = self.FALLBACK_MODEL
            except Exception as e2:
                log.debug("Fallback model load failed: %s", e2)
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
        hybrid_used = False
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
                hybrid_used = True
            except Exception:
                pass

        if self._hybrid_calibrator is not None:
            try:
                probs2 = self._hybrid_calibrator.predict_proba(np.asarray([feat], dtype=float))[0]
                classes2 = list(self._hybrid_calibrator.classes_)
                p_pos2 = sum(float(probs2[i]) for i, c in enumerate(classes2) if int(c) > 0)
                p_neg2 = sum(float(probs2[i]) for i, c in enumerate(classes2) if int(c) < 0)
                nn_score = self._clip(p_pos2 - p_neg2, -1.0, 1.0)
                overall = self._clip((0.68 * overall) + (0.32 * nn_score), -1.0, 1.0)
                tf_conf = self._clip((0.82 * tf_conf) + (0.18 * max(float(x) for x in probs2)), 0.0, 1.0)
                hybrid_used = True
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
            model_used=(
                f"{model_used} + hybrid_neural_network"
                if hybrid_used
                else model_used
            ),
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
            from utils.safe_pickle import safe_pickle_load
            with self._calibrator_path.open("rb") as f:
                self._calibrator = safe_pickle_load(f)
        except Exception as e:
            log.warning("Failed to load calibrator: %s", e)
            self._calibrator = None

    def _save_calibrator(self) -> None:
        if self._calibrator is None:
            return
        try:
            with self._calibrator_path.open("wb") as f:
                pickle.dump(self._calibrator, f)
        except Exception:
            pass

    def _load_hybrid_calibrator(self) -> None:
        self._hybrid_calibrator = None
        if not self._hybrid_calibrator_path.exists():
            return
        try:
            from utils.safe_pickle import safe_pickle_load
            with self._hybrid_calibrator_path.open("rb") as f:
                self._hybrid_calibrator = safe_pickle_load(f)
        except Exception as e:
            log.warning("Failed to load hybrid calibrator: %s", e)
            self._hybrid_calibrator = None

    def _save_hybrid_calibrator(self) -> None:
        if self._hybrid_calibrator is None:
            return
        try:
            with self._hybrid_calibrator_path.open("wb") as f:
                pickle.dump(self._hybrid_calibrator, f)
        except Exception:
            pass

    def _write_training_status(self, payload: dict[str, Any]) -> None:
        report = dict(payload or {})
        report["saved_at"] = datetime.now().isoformat()
        report["artifact_dir"] = str(self.cache_dir)
        try:
            tmp = self._status_path.with_suffix(".tmp")
            with tmp.open("w", encoding="utf-8") as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            tmp.replace(self._status_path)
        except Exception:
            pass

    def _load_seen_articles(self) -> None:
        self._seen_articles = {}
        if not self._seen_article_path.exists():
            return
        try:
            with self._seen_article_path.open("r", encoding="utf-8") as f:
                payload = json.load(f)
            raw = payload.get("seen_articles", {}) if isinstance(payload, dict) else {}
            if isinstance(raw, dict):
                for aid, ts in raw.items():
                    key = str(aid or "").strip()
                    if not key:
                        continue
                    try:
                        tsf = float(ts)
                    except (TypeError, ValueError):
                        continue
                    if math.isfinite(tsf):
                        self._seen_articles[key] = tsf
        except Exception as exc:
            log.debug("Failed to load seen-article registry: %s", exc)
            self._seen_articles = {}
        self._prune_seen_articles()

    def _save_seen_articles(self) -> None:
        payload = {
            "saved_at": datetime.now().isoformat(),
            "ttl_seconds": float(self._seen_article_ttl_seconds),
            "seen_articles": dict(self._seen_articles),
        }
        try:
            tmp = self._seen_article_path.with_suffix(".tmp")
            with tmp.open("w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            tmp.replace(self._seen_article_path)
        except Exception as exc:
            log.debug("Failed to save seen-article registry: %s", exc)

    def _prune_seen_articles(self) -> None:
        """Prune expired and excess seen articles to manage memory."""
        now = float(time.time())
        cutoff = now - max(60.0, float(self._seen_article_ttl_seconds))
        kept: dict[str, float] = {}

        # First pass: keep only non-expired articles
        for aid, ts in list(self._seen_articles.items()):
            key = str(aid or "").strip()
            if not key:
                continue
            try:
                tsf = float(ts)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(tsf):
                continue
            if tsf >= cutoff:
                kept[key] = tsf

        # FIX 10: Enforce memory limit by keeping only most recent articles
        max_articles = int(self._max_seen_articles)
        if len(kept) > max_articles:
            # Sort by timestamp descending and keep most recent
            sorted_items = sorted(
                kept.items(),
                key=lambda kv: float(kv[1]),
                reverse=True,
            )[:max_articles]
            kept = {k: float(v) for k, v in sorted_items}

        self._seen_articles = kept

    def _remember_seen_articles(self, article_ids: list[str]) -> None:
        now = float(time.time())
        with self._lock:
            for aid in list(article_ids or []):
                key = str(aid or "").strip()
                if key:
                    self._seen_articles[key] = now
            self._prune_seen_articles()
            self._save_seen_articles()

    @staticmethod
    def _normalize_stock_code(raw: object) -> str:
        digits = "".join(ch for ch in str(raw or "").strip() if ch.isdigit())
        return digits if len(digits) == 6 else ""

    def _load_recent_cycle_stock_codes(
        self,
        *,
        max_codes: int = 12,
        max_files: int = 40,
    ) -> list[str]:
        out: list[str] = []
        seen: set[str] = set()
        try:
            history_dir = Path(CONFIG.DATA_DIR) / "cycle_history"
            files = sorted(
                history_dir.glob("cycle_*.json"),
                key=lambda p: p.name,
                reverse=True,
            )[: max(1, int(max_files))]
        except Exception as exc:
            log.debug("Cycle-history lookup failed: %s", exc)
            files = []

        for path in files:
            try:
                with path.open("r", encoding="utf-8") as f:
                    payload = json.load(f)
            except Exception as exc:
                log.debug("Cycle-history read failed for %s: %s", path, exc)
                continue
            if not isinstance(payload, dict):
                continue
            for key in ("ok_stocks", "new_stocks", "replay_stocks"):
                for raw in list(payload.get(key, []) or []):
                    code = self._normalize_stock_code(raw)
                    if not code or code in seen:
                        continue
                    seen.add(code)
                    out.append(code)
                    if len(out) >= int(max(1, max_codes)):
                        return out
        return out

    def _build_auto_search_queries(
        self,
        *,
        date_token: str,
        related_codes: list[str] | None = None,
        related_keywords: list[str] | None = None,
    ) -> list[list[str]]:
        queries: list[list[str]] = [
            ["A股", "政策", "监管", "证监会", date_token],
            ["沪深", "板块", "资金", "产业链", date_token],
            ["上市公司", "业绩", "增持", "回购", date_token],
            ["央行", "人民币", "降准", "降息", date_token],
            ["北向资金", "行业政策", "中国经济", date_token],
        ]

        code_list = list(related_codes or [])[:12]
        for code in code_list:
            queries.append([code, "A股", "公告", "业绩", date_token])
            queries.append([code, "主力资金", "板块", "研报", date_token])

        kw_list = [
            str(k).strip()
            for k in list(related_keywords or [])
            if str(k).strip()
        ][:12]
        for kw in kw_list:
            queries.append([kw, "A股", "政策", date_token])

        deduped: list[list[str]] = []
        seen_keys: set[tuple[str, ...]] = set()
        for row in queries:
            normalized = [str(x).strip() for x in row if str(x).strip()]
            if not normalized:
                continue
            key = tuple(normalized)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            deduped.append(normalized)
        return deduped

    @staticmethod
    def _stable_article_id(*parts: object) -> str:
        raw = "|".join(str(p or "").strip() for p in parts if str(p or "").strip())
        if not raw:
            raw = f"fallback:{time.time():.6f}"
        return hashlib.md5(raw.encode("utf-8", errors="ignore")).hexdigest()[:20]

    @staticmethod
    def _extract_stock_codes_from_text(
        text: str,
        *,
        max_codes: int = 12,
    ) -> list[str]:
        out: list[str] = []
        seen: set[str] = set()
        for raw in re.findall(r"\b(\d{6})\b", str(text or "")):
            code = str(raw).strip()
            if len(code) != 6 or code in seen:
                continue
            seen.add(code)
            out.append(code)
            if len(out) >= int(max(1, max_codes)):
                break
        return out

    def _load_related_codes_from_china_news(
        self,
        *,
        max_codes: int = 12,
    ) -> list[str]:
        """Extract candidate stock codes from China market/news context."""
        try:
            from data.news_aggregator import get_news_aggregator

            agg = get_news_aggregator()
            news = agg.get_market_news(
                count=max(36, int(max_codes) * 8),
                force_refresh=False,
            )
        except Exception as exc:
            log.debug("Related-code discovery via China news failed: %s", exc)
            return []

        out: list[str] = []
        seen: set[str] = set()
        for item in list(news or []):
            code_candidates: list[str] = []
            for raw in list(getattr(item, "stock_codes", []) or []):
                code = self._normalize_stock_code(raw)
                if code:
                    code_candidates.append(code)
            if not code_candidates:
                text = f"{getattr(item, 'title', '')} {getattr(item, 'content', '')}"
                code_candidates = self._extract_stock_codes_from_text(
                    text,
                    max_codes=max_codes,
                )
            for code in code_candidates:
                if code in seen:
                    continue
                seen.add(code)
                out.append(code)
                if len(out) >= int(max(1, max_codes)):
                    return out
        return out

    def _news_item_to_article(
        self,
        item: Any,
        *,
        corpus_tag: str,
        default_category: str = "market",
        bound_code: str | None = None,
    ) -> NewsArticle:
        """Convert NewsItem-like payload into normalized NewsArticle."""
        now_dt = datetime.now()
        title = str(getattr(item, "title", "") or "").strip()
        content = str(getattr(item, "content", "") or "").strip()
        if not content:
            content = title
        summary = str(content[:220] or title[:220]).strip()
        source = str(getattr(item, "source", "") or "china_news").strip()
        url = str(getattr(item, "url", "") or "").strip()

        published_at = getattr(item, "publish_time", None)
        if not isinstance(published_at, datetime):
            published_at = now_dt

        text = f"{title} {content}".strip()
        language = self._detect_language(text)
        category = str(getattr(item, "category", "") or default_category).strip().lower()
        if category not in {"policy", "market", "company", "economic", "regulatory", "instruction"}:
            category = str(default_category or "market")

        entities: list[str] = []
        if bound_code:
            ncode = self._normalize_stock_code(bound_code)
            if ncode:
                entities.append(ncode)
        for raw in list(getattr(item, "stock_codes", []) or []):
            code = self._normalize_stock_code(raw)
            if code and code not in entities:
                entities.append(code)

        tags = [
            str(corpus_tag or "").strip().lower(),
            "china_network",
            str(source or "").strip().lower(),
        ]
        tags = [t for t in tags if t]
        if category:
            tags.append(str(category).strip().lower())

        article_id = self._stable_article_id(
            corpus_tag,
            source,
            title,
            url,
            published_at.isoformat() if isinstance(published_at, datetime) else "",
            ",".join(entities),
        )
        return NewsArticle(
            id=article_id,
            title=title,
            content=content,
            summary=summary,
            source=source,
            url=url,
            published_at=published_at,
            collected_at=now_dt,
            language=language,
            category=category,
            sentiment_score=self._safe_float(getattr(item, "sentiment_score", 0.0), 0.0),
            relevance_score=self._clip(
                self._safe_float(getattr(item, "importance", 0.6), 0.6),
                0.0,
                1.0,
            ),
            entities=entities,
            tags=tags,
        )

    def _build_instruction_conversation_corpus(
        self,
        base_articles: list[NewsArticle],
        *,
        max_items: int,
    ) -> list[NewsArticle]:
        """Create instruction/chat-style corpus from real China news text."""
        out: list[NewsArticle] = []
        now_dt = datetime.now()
        for idx, article in enumerate(list(base_articles or [])[: int(max(1, max_items))]):
            title = str(getattr(article, "title", "") or "").strip()
            content = str(getattr(article, "content", "") or "").strip()
            if len(title) < 6:
                continue
            lang = str(getattr(article, "language", "") or "").strip().lower()
            if lang not in {"zh", "en"}:
                lang = self._detect_language(f"{title} {content}")

            snippet = str(content[:280] or title)
            if lang == "zh":
                prompt = f"请根据以下A股资讯判断市场情绪并给出风险提示：{title}"
                answer = (
                    "结论：请结合政策方向、行业景气与资金面进行综合判断。"
                    f"要点：{snippet}"
                )
                convo = f"用户：{prompt}\n助手：{answer}"
            else:
                prompt = f"Assess sentiment and risk for this China market update: {title}"
                answer = (
                    "Conclusion: evaluate policy tone, sector momentum, and capital flows together. "
                    f"Key points: {snippet}"
                )
                convo = f"User: {prompt}\nAssistant: {answer}"

            article_id = self._stable_article_id(
                "instruction_conversation",
                article.id,
                idx,
            )
            tags = list(getattr(article, "tags", []) or [])
            tags.extend(["instruction_conversation", "china_network"])

            out.append(
                NewsArticle(
                    id=article_id,
                    title=f"Instruction sample: {title}"[:220],
                    content=convo,
                    summary=answer[:220],
                    source="china_instruction_corpus",
                    url=str(getattr(article, "url", "") or ""),
                    published_at=getattr(article, "published_at", now_dt)
                    if isinstance(getattr(article, "published_at", None), datetime)
                    else now_dt,
                    collected_at=now_dt,
                    language=lang,
                    category="instruction",
                    sentiment_score=self._safe_float(getattr(article, "sentiment_score", 0.0), 0.0),
                    relevance_score=max(0.70, self._safe_float(getattr(article, "relevance_score", 0.65), 0.65)),
                    entities=list(getattr(article, "entities", []) or []),
                    tags=tags,
                )
            )
        return out

    def _collect_china_corpus_segments(
        self,
        *,
        related_codes: list[str],
        limit_per_query: int,
        max_related_codes: int,
        stop_flag: Callable[[], bool] | None = None,
    ) -> tuple[dict[str, list[NewsArticle]], list[str]]:
        """Collect segmented China corpus for LLM training."""
        buckets: dict[str, list[NewsArticle]] = {
            "general_text": [],
            "policy_news": [],
            "stock_specific": [],
            "instruction_conversation": [],
        }
        discovered_codes: list[str] = []

        def _stopped() -> bool:
            if stop_flag is None:
                return False
            try:
                return bool(stop_flag())
            except Exception:
                return False

        try:
            from data.news_aggregator import get_news_aggregator

            agg = get_news_aggregator()
        except Exception as exc:
            log.warning("China corpus aggregation unavailable: %s", exc)
            return buckets, discovered_codes

        market_count = int(max(30, min(220, int(limit_per_query))))
        policy_count = int(max(16, min(120, int(limit_per_query // 2))))
        stock_count = int(max(12, min(80, int(limit_per_query // 3))))

        try:
            market_items = agg.get_market_news(
                count=market_count,
                force_refresh=True,
            )
        except Exception as exc:
            log.debug("China general-text corpus fetch failed: %s", exc)
            market_items = []
        for item in list(market_items or []):
            if _stopped():
                break
            article = self._news_item_to_article(
                item,
                corpus_tag="general_text",
                default_category="market",
            )
            buckets["general_text"].append(article)
            if len(discovered_codes) < int(max(1, max_related_codes)):
                text = f"{article.title} {article.content}"
                for code in self._extract_stock_codes_from_text(
                    text,
                    max_codes=max_related_codes,
                ):
                    if code not in discovered_codes:
                        discovered_codes.append(code)
                        if len(discovered_codes) >= int(max(1, max_related_codes)):
                            break

        try:
            policy_items = agg.get_policy_news(count=policy_count)
        except Exception as exc:
            log.debug("China policy corpus fetch failed: %s", exc)
            policy_items = []
        for item in list(policy_items or []):
            if _stopped():
                break
            buckets["policy_news"].append(
                self._news_item_to_article(
                    item,
                    corpus_tag="policy_news",
                    default_category="policy",
                )
            )

        code_candidates: list[str] = []
        seen_codes: set[str] = set()
        for code in list(related_codes or []) + list(discovered_codes or []):
            norm = self._normalize_stock_code(code)
            if not norm or norm in seen_codes:
                continue
            seen_codes.add(norm)
            code_candidates.append(norm)
            if len(code_candidates) >= int(max(1, max_related_codes)):
                break

        for code in code_candidates:
            if _stopped():
                break
            try:
                stock_items = agg.get_stock_news(
                    code,
                    count=stock_count,
                    force_refresh=True,
                )
            except Exception as exc:
                log.debug("China stock-specific corpus fetch failed for %s: %s", code, exc)
                stock_items = []
            for item in list(stock_items or []):
                buckets["stock_specific"].append(
                    self._news_item_to_article(
                        item,
                        corpus_tag="stock_specific",
                        default_category="company",
                        bound_code=code,
                    )
                )

        base_for_instruction = (
            list(buckets["stock_specific"])
            + list(buckets["policy_news"])
            + list(buckets["general_text"])
        )
        buckets["instruction_conversation"] = self._build_instruction_conversation_corpus(
            base_for_instruction,
            max_items=max(18, int(limit_per_query // 2)),
        )
        return buckets, code_candidates

    def _filter_high_quality_articles(
        self,
        rows: list[NewsArticle],
        *,
        hours_back: int,
    ) -> tuple[list[NewsArticle], dict[str, int]]:
        """High-quality filter for LLM training corpus."""
        now_dt = datetime.now()
        cutoff = now_dt - timedelta(hours=max(12, int(hours_back)))
        stats = {
            "input": int(len(rows or [])),
            "kept": 0,
            "drop_missing": 0,
            "drop_length": 0,
            "drop_time": 0,
            "drop_garbled": 0,
            "drop_duplicate": 0,
        }
        out: list[NewsArticle] = []
        seen_keys: set[tuple[str, str, str]] = set()
        seen_ids: set[str] = set()

        for article in list(rows or []):
            title = str(getattr(article, "title", "") or "").strip()
            content = str(getattr(article, "content", "") or "").strip()
            if not content:
                content = title
                article.content = content
            if not title:
                title = content[:160]
                article.title = title
            if not title and not content:
                stats["drop_missing"] += 1
                continue

            text = f"{title} {content}".strip()
            token_chars = len(re.findall(r"[A-Za-z0-9\u4e00-\u9fff]", text))
            if len(title) < 6 or token_chars < 18:
                stats["drop_length"] += 1
                continue

            replacement = text.count("\ufffd")
            if replacement > 0:
                stats["drop_garbled"] += 1
                continue

            published_at = getattr(article, "published_at", None)
            if not isinstance(published_at, datetime):
                published_at = now_dt
                article.published_at = published_at
            if published_at < cutoff or published_at > (now_dt + timedelta(minutes=30)):
                stats["drop_time"] += 1
                continue

            aid = str(getattr(article, "id", "") or "").strip()
            if not aid:
                aid = self._stable_article_id(
                    getattr(article, "source", ""),
                    title,
                    published_at.isoformat(),
                    getattr(article, "url", ""),
                )
                article.id = aid
            if aid in seen_ids:
                stats["drop_duplicate"] += 1
                continue

            dedup_key = (
                str(getattr(article, "source", "") or "").strip().lower(),
                title.lower()[:120],
                content.lower()[:240],
            )
            if dedup_key in seen_keys:
                stats["drop_duplicate"] += 1
                continue

            lang = str(getattr(article, "language", "") or "").strip().lower()
            if lang not in {"zh", "en"}:
                article.language = self._detect_language(text)
            category = str(getattr(article, "category", "") or "").strip().lower()
            if category not in {"policy", "market", "company", "economic", "regulatory", "instruction"}:
                article.category = "market"
            summary = str(getattr(article, "summary", "") or "").strip()
            if not summary:
                article.summary = content[:220]

            seen_ids.add(aid)
            seen_keys.add(dedup_key)
            out.append(article)

        stats["kept"] = int(len(out))
        return out, stats

    def get_training_status(self) -> dict[str, Any]:
        calibrator_ready = bool(self._calibrator is not None)
        hybrid_nn_ready = bool(self._hybrid_calibrator is not None)
        models_loaded = calibrator_ready or hybrid_nn_ready

        if self._status_path.exists():
            try:
                with self._status_path.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    file_status = str(data.get("status", "") or "").strip().lower()
                    # Only upgrade "not_trained" or empty — not "stopped" (which is a real user action).
                    # This handles the case where the status file was never written or was corrupted.
                    if models_loaded and file_status in {"not_trained", ""}:
                        data = dict(data)
                        data["status"] = "trained"
                        data["calibrator_ready"] = calibrator_ready
                        data["hybrid_nn_ready"] = hybrid_nn_ready
                        self._write_training_status(data)
                    return data
            except Exception:
                pass
        if models_loaded:
            return {
                "status": "trained",
                "training_architecture": "hybrid_neural_network",
                "artifact_dir": str(self.cache_dir),
                "calibrator_ready": calibrator_ready,
                "hybrid_nn_ready": hybrid_nn_ready,
            }
        return {
            "status": "not_trained",
            "training_architecture": "hybrid_neural_network",
            "artifact_dir": str(self.cache_dir),
            "calibrator_ready": False,
            "hybrid_nn_ready": False,
        }

    def train(
        self,
        articles: list[NewsArticle],
        *,
        epochs: int = 3,
        max_samples: int = 1000,
        learning_rate: float = 2e-5,
        validation_split: float = 0.15,
        use_transformer_labels: bool = True,
        feature_scaling: bool = True,
    ) -> dict[str, Any]:
        """Train hybrid sentiment models with proper validation and metrics.

        Args:
            articles: List of news articles for training
            epochs: Number of training epochs for MLP (used if sklearn available)
            max_samples: Maximum number of samples to use
            learning_rate: Learning rate (currently unused, reserved for future transformer fine-tuning)
            validation_split: Fraction of data to hold out for validation
            use_transformer_labels: Use transformer model for label generation if available
            feature_scaling: Apply StandardScaler to features before training

        Returns:
            Training report with metrics and status
        """
        t0 = time.time()
        start = datetime.now().isoformat()
        rows = list(articles or [])[: max(50, int(max_samples))]
        zh = 0
        en = 0
        if not rows:
            out = {
                "status": "skipped",
                "model_name": self.model_name,
                "trained_samples": 0,
                "zh_samples": 0,
                "en_samples": 0,
                "started_at": start,
                "finished_at": datetime.now().isoformat(),
                "duration_seconds": float(time.time() - t0),
                "notes": "No articles provided.",
                "training_architecture": "hybrid_neural_network",
                "calibrator_ready": bool(self._calibrator is not None),
                "hybrid_nn_ready": bool(self._hybrid_calibrator is not None),
            }
            self._write_training_status(out)
            return out

        # Build features and generate labels
        x: list[list[float]] = []
        y: list[int] = []
        transformer_labels: list[float] = []

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

            # Use transformer for label generation if available and requested
            if use_transformer_labels and self._pipe is not None:
                try:
                    self._load_pipeline()
                    if self._pipe is not None:
                        tf_overall, tf_pos, tf_neg, tf_neu, tf_conf = self._parse_scores(self._pipe(text[:512]))
                        # Use transformer score as primary label signal
                        lbl = 1 if tf_overall >= 0.15 else (-1 if tf_overall <= -0.15 else 0)
                        transformer_labels.append(tf_overall)
                    else:
                        lbl = 1 if base >= 0.08 else (-1 if base <= -0.08 else 0)
                        transformer_labels.append(base)
                except Exception:
                    lbl = 1 if base >= 0.08 else (-1 if base <= -0.08 else 0)
                    transformer_labels.append(base)
            else:
                lbl = 1 if base >= 0.08 else (-1 if base <= -0.08 else 0)
                transformer_labels.append(base)

            x.append(self._build_features(a, tf_overall=base, tf_conf=0.4, language=lang))
            y.append(lbl)

        notes: list[str] = []
        metrics: dict[str, float] = {}
        x_arr = np.asarray(x, dtype=float)
        y_arr = np.asarray(y, dtype=int)

        # Apply feature scaling if requested
        scaler = None
        if feature_scaling and _SKLEARN_AVAILABLE:
            try:
                from sklearn.preprocessing import StandardScaler
                scaler = StandardScaler()
                x_arr = scaler.fit_transform(x_arr)
                notes.append("Feature scaling applied (StandardScaler)")
            except Exception as exc:
                notes.append(f"Feature scaling failed: {exc}")

        # Calculate class distribution
        class_counts = np.bincount(y_arr + 1)  # Shift -1,0,1 to 0,1,2
        class_distribution = {
            "negative": int(class_counts[0]) if len(class_counts) > 0 else 0,
            "neutral": int(class_counts[1]) if len(class_counts) > 1 else 0,
            "positive": int(class_counts[2]) if len(class_counts) > 2 else 0,
        }
        metrics["class_distribution"] = class_distribution  # type: ignore

        # Check class diversity and balance
        unique_classes = len(set(y_arr))
        class_imbalance_ratio = float(max(class_counts) / max(1, min(class_counts))) if len(class_counts) > 1 else 1.0
        metrics["class_imbalance_ratio"] = class_imbalance_ratio  # type: ignore

        logreg_ready = False
        hybrid_nn_ready = False
        val_metrics: dict[str, float] = {}

        # Split data for validation
        n_samples = len(x_arr)
        n_val = int(n_samples * validation_split)
        indices = np.random.permutation(n_samples)
        train_idx, val_idx = indices[n_val:], indices[:n_val]

        x_train, x_val = x_arr[train_idx], x_arr[val_idx]
        y_train, y_val = y_arr[train_idx], y_arr[val_idx]

        metrics["train_samples"] = float(len(x_train))  # type: ignore
        metrics["val_samples"] = float(len(x_val))  # type: ignore

        if unique_classes < 2:
            notes.append("Not enough class diversity for calibration fit.")
        elif _SKLEARN_AVAILABLE:
            # Train Logistic Regression with improved hyperparameters
            try:
                # Adjust class weights for imbalanced data
                if class_imbalance_ratio > 2.0:
                    class_weight = "balanced"
                    notes.append(f"Using balanced class weights (imbalance ratio={class_imbalance_ratio:.2f})")
                else:
                    class_weight = "balanced"

                clf = LogisticRegression(
                    max_iter=max(320, epochs * 100),  # Use epochs parameter
                    multi_class="auto",
                    class_weight=class_weight,
                    solver="lbfgs",
                    C=1.0,
                    random_state=42,
                )
                clf.fit(x_train, y_train)
                self._calibrator = clf
                self._save_calibrator()
                logreg_ready = True

                # Calculate validation metrics
                if len(x_val) > 0 and len(set(y_val)) > 1:
                    val_pred = clf.predict(x_val)
                    val_proba = clf.predict_proba(x_val)

                    from sklearn.metrics import accuracy_score, f1_score
                    val_metrics["accuracy"] = float(accuracy_score(y_val, val_pred))
                    val_metrics["f1_macro"] = float(f1_score(y_val, val_pred, average="macro", zero_division=0))
                    val_metrics["f1_weighted"] = float(f1_score(y_val, val_pred, average="weighted", zero_division=0))
                    metrics["validation_accuracy"] = val_metrics["accuracy"]  # type: ignore

                    notes.append(f"Validation accuracy={val_metrics['accuracy']:.3f}, F1={val_metrics['f1_weighted']:.3f}")
            except Exception as exc:
                notes.append(f"Logistic calibration fit failed: {exc}")
        else:
            notes.append("scikit-learn unavailable; logistic calibrator skipped.")

        # Train MLP with improved configuration
        if unique_classes >= 2 and _SKLEARN_MLP_AVAILABLE and len(x_train) >= 30:
            try:
                # Adaptive hidden layer sizes based on sample count
                n_features = x_train.shape[1]
                if len(x_train) < 100:
                    hidden_sizes = (16, 8)
                elif len(x_train) < 500:
                    hidden_sizes = (32, 16)
                else:
                    hidden_sizes = (64, 32, 16)

                nn = MLPClassifier(
                    hidden_layer_sizes=hidden_sizes,
                    activation="relu",
                    solver="adam",
                    alpha=1e-4,  # Increased regularization
                    learning_rate="adaptive",
                    learning_rate_init=max(1e-4, learning_rate),  # Use learning_rate parameter
                    batch_size=min(64, max(16, len(x_train) // 16)),
                    max_iter=max(200, epochs * 50),  # Use epochs parameter
                    random_state=42,
                    early_stopping=True,
                    validation_fraction=min(0.2, max(0.1, validation_split)),
                    n_iter_no_change=20,
                    tol=1e-4,
                )
                nn.fit(x_train, y_train)
                self._hybrid_calibrator = nn
                self._save_hybrid_calibrator()
                hybrid_nn_ready = True

                # Calculate MLP validation metrics
                if len(x_val) > 0 and len(set(y_val)) > 1:
                    nn_pred = nn.predict(x_val)
                    nn_proba = nn.predict_proba(x_val)

                    from sklearn.metrics import accuracy_score, f1_score
                    val_metrics["mlp_accuracy"] = float(accuracy_score(y_val, nn_pred))
                    val_metrics["mlp_f1_macro"] = float(f1_score(y_val, nn_pred, average="macro", zero_division=0))
                    val_metrics["mlp_f1_weighted"] = float(f1_score(y_val, nn_pred, average="weighted", zero_division=0))

                    notes.append(f"MLP validation accuracy={val_metrics['mlp_accuracy']:.3f}")
            except (ConvergenceWarning, UserWarning) as exc:
                log.debug("MLP convergence warning: %s", exc)
                if hasattr(nn, 'classes_') and nn.classes_ is not None:
                    self._hybrid_calibrator = nn
                    self._save_hybrid_calibrator()
                    hybrid_nn_ready = True
                    notes.append(f"Hybrid NN fit with convergence warning: {exc}")
                else:
                    notes.append(f"Hybrid NN fit failed due to convergence: {exc}")
            except Exception as exc:
                notes.append(f"Hybrid NN fit failed: {exc}")
        elif not _SKLEARN_MLP_AVAILABLE:
            notes.append("MLP classifier unavailable; hybrid NN head skipped.")
        elif len(x_train) < 30:
            notes.append(f"Insufficient samples ({len(x_train)}) for stable hybrid NN fit (need >=30).")

        # Determine overall status
        if logreg_ready or hybrid_nn_ready:
            status = "trained"
        else:
            status = "partial" if len(rows) > 0 else "skipped"

        architecture = "hybrid_neural_network"
        if not (logreg_ready or hybrid_nn_ready):
            architecture = "rule_based_fallback"

        # Build comprehensive output report
        out = {
            "status": status,
            "model_name": self.model_name,
            "trained_samples": int(len(rows)),
            "train_samples": int(len(x_train)),
            "val_samples": int(len(x_val)),
            "zh_samples": int(zh),
            "en_samples": int(en),
            "started_at": start,
            "finished_at": datetime.now().isoformat(),
            "duration_seconds": float(time.time() - t0),
            "notes": "; ".join(notes) if notes else "Hybrid neural training updated.",
            "training_architecture": architecture,
            "calibrator_ready": bool(logreg_ready),
            "hybrid_nn_ready": bool(hybrid_nn_ready),
            "artifact_dir": str(self.cache_dir),
            "validation_split": float(validation_split),
            "feature_scaling_applied": bool(scaler is not None),
            "class_distribution": class_distribution,
            "class_imbalance_ratio": float(class_imbalance_ratio),
            "validation_metrics": val_metrics if val_metrics else None,
            "epochs_used": int(epochs),
            "learning_rate_used": float(learning_rate),
        }
        self._write_training_status(out)
        return out

    def auto_train_from_internet(
        self,
        *,
        hours_back: int = 96,
        limit_per_query: int = 180,
        max_samples: int = 1200,
        stop_flag: Callable[[], bool] | None = None,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
        force_china_direct: bool = False,
        only_new: bool = True,
        min_new_articles: int = 24,
        seen_ttl_hours: int = 168,
        auto_related_search: bool = True,
        related_keywords: list[str] | None = None,
        max_related_codes: int = 12,
        allow_gm_bootstrap: bool = False,
    ) -> dict[str, Any]:
        """Auto-train from internet news with robust error handling.

        Args:
            hours_back: Hours of news to collect
            limit_per_query: Max articles per query
            max_samples: Max samples for training
            stop_flag: Callback to check if training should stop
            progress_callback: Callback for progress updates
            force_china_direct: Force China direct mode
            only_new: Only train on new unseen articles
            min_new_articles: Minimum new articles required for training
            seen_ttl_hours: TTL for seen articles cache
            auto_related_search: Enable auto-related search
            related_keywords: Optional related keywords
            max_related_codes: Max related stock codes to include
            allow_gm_bootstrap: Allow fallback to GM cycle-history stocks

        Returns:
            Training report with status and metrics
        """
        if force_china_direct:
            os.environ["TRADING_CHINA_DIRECT"] = "1"
            os.environ["TRADING_VPN"] = "0"
            from core.network import invalidate_network_cache

            invalidate_network_cache()
            reset_collector()
        if int(seen_ttl_hours) > 0:
            self._seen_article_ttl_seconds = float(
                max(3600, int(seen_ttl_hours) * 3600)
            )
        self._prune_seen_articles()

        def _is_stopped() -> bool:
            if stop_flag is None:
                return False
            try:
                return bool(stop_flag())
            except Exception:
                return False

        def _emit(percent: int, message: str, stage: str) -> None:
            if callable(progress_callback):
                try:
                    progress_callback(
                        {
                            "percent": int(max(0, min(100, int(percent)))),
                            "message": str(message or ""),
                            "stage": str(stage or ""),
                        }
                    )
                except Exception:
                    pass

        # FIX 7: Robust collector initialization with error handling
        collector = None
        try:
            collector = get_collector()
            if collector is None:
                raise RuntimeError("News collector initialization returned None")
        except Exception as exc:
            log.error("Failed to initialize news collector: %s", exc)
            error_report = {
                "status": "error",
                "error_type": "collector_initialization_failed",
                "error_message": str(exc),
                "collected_articles": 0,
                "trained_samples": 0,
                "notes": f"News collector initialization failed: {exc}",
            }
            self._write_training_status(error_report)
            _emit(100, "Collector initialization failed", "error")
            return error_report

        date_token = datetime.now().strftime("%Y-%m-%d")
        related_codes: list[str] = []
        related_codes_source = "none"
        if bool(auto_related_search):
            try:
                related_codes = self._load_related_codes_from_china_news(
                    max_codes=max_related_codes,
                )
                if related_codes:
                    related_codes_source = "china_news"
            except Exception as exc:
                log.debug("Failed to load related codes from China news: %s", exc)
                related_codes = []

            allow_gm_bootstrap_effective = bool(allow_gm_bootstrap) or bool(
                env_flag("TRADING_LLM_ALLOW_GM_BOOTSTRAP", "0")
            )
            if (not related_codes) and allow_gm_bootstrap_effective:
                try:
                    related_codes = self._load_recent_cycle_stock_codes(
                        max_codes=max_related_codes,
                        max_files=40,
                    )
                    if related_codes:
                        related_codes_source = "gm_cycle_history"
                except Exception as exc:
                    log.debug("GM bootstrap related-code load failed: %s", exc)
                    related_codes = []

        queries = self._build_auto_search_queries(
            date_token=date_token,
            related_codes=related_codes,
            related_keywords=related_keywords,
        )
        _emit(
            4,
            (
                f"Auto search prepared {len(queries)} query groups "
                f"(related_codes={len(related_codes)}, source={related_codes_source})"
            ),
            "query_build",
        )
        seen: set[str] = set()
        rows: list[NewsArticle] = []
        skipped_seen = 0
        strict_batch_failures = 0
        strict_batch_recoveries = 0
        corpus_breakdown: dict[str, int] = {
            "search_news": 0,
            "general_text": 0,
            "policy_news": 0,
            "stock_specific": 0,
            "instruction_conversation": 0,
        }
        _emit(2, "Starting internet collection...", "start")

        def _build_stopped_payload() -> dict[str, Any]:
            return {
                "status": "stopped",
                "collected_articles": int(len(rows)),
                "new_articles": int(len(rows)),
                "reused_articles_skipped": int(skipped_seen),
                "hours_back": int(hours_back),
                "limit_per_query": int(limit_per_query),
                "query_count": int(len(queries)),
                "related_stock_codes": list(related_codes),
                "related_codes_source": str(related_codes_source),
                "china_direct_mode": bool(env_flag("TRADING_CHINA_DIRECT", "0")),
                "corpus_breakdown": dict(corpus_breakdown),
                "training_architecture": "hybrid_neural_network",
            }

        def _append_batch(
            batch: list[NewsArticle],
            *,
            bucket: str,
        ) -> tuple[int, bool]:
            nonlocal skipped_seen
            added = 0
            for article in list(batch or []):
                if _is_stopped():
                    return added, True
                aid = str(getattr(article, "id", "") or "").strip()
                if not aid:
                    aid = self._stable_article_id(
                        bucket,
                        getattr(article, "source", ""),
                        getattr(article, "title", ""),
                        getattr(article, "url", ""),
                        getattr(article, "published_at", ""),
                    )
                    article.id = aid
                if aid in seen:
                    continue
                if bool(only_new) and aid in self._seen_articles:
                    skipped_seen += 1
                    continue
                seen.add(aid)
                rows.append(article)
                added += 1
            corpus_breakdown[bucket] = int(corpus_breakdown.get(bucket, 0) + added)
            return added, False

        for i, kw in enumerate(queries):
            if _is_stopped():
                stopped = _build_stopped_payload()
                self._write_training_status(stopped)
                return stopped

            query_keywords = list(kw)
            batch: list[NewsArticle]

            try:
                batch = collector.collect_news(
                    keywords=query_keywords,
                    limit=max(20, int(limit_per_query)),
                    hours_back=max(12, int(hours_back)),
                    strict=True,
                )
            except Exception as exc:
                strict_batch_failures += 1
                log.warning(
                    "Strict news batch collection failed for query=%s: %s",
                    ",".join(query_keywords),
                    exc,
                )
                _emit(
                    8 + int(((i + 1) / max(1, len(queries))) * 62),
                    (
                        f"Strict batch {i + 1}/{len(queries)} failed; "
                        "retrying in non-strict mode."
                    ),
                    "collect",
                )
                try:
                    batch = collector.collect_news(
                        keywords=query_keywords,
                        limit=max(20, int(limit_per_query)),
                        hours_back=max(12, int(hours_back)),
                        strict=False,
                    )
                    if batch:
                        strict_batch_recoveries += 1
                except Exception as fallback_exc:
                    log.warning(
                        "Non-strict news fallback failed for query=%s: %s",
                        ",".join(query_keywords),
                        fallback_exc,
                    )
                    batch = []

            added_rows, stopped = _append_batch(batch, bucket="search_news")
            _emit(
                8 + int(((i + 1) / max(1, len(queries))) * 62),
                (
                    f"Collected batch {i + 1}/{len(queries)} "
                    f"(raw={len(batch)} new={added_rows})"
                ),
                "collect",
            )
            if stopped:
                stopped_payload = _build_stopped_payload()
                self._write_training_status(stopped_payload)
                return stopped_payload

        _emit(70, "Collecting China corpus segments (general/policy/stock/instruction)...", "collect")
        china_buckets, discovered_codes = self._collect_china_corpus_segments(
            related_codes=related_codes,
            limit_per_query=max(20, int(limit_per_query)),
            max_related_codes=max_related_codes,
            stop_flag=stop_flag,
        )
        if not related_codes and discovered_codes:
            related_codes = list(discovered_codes)
            related_codes_source = "china_corpus_discovery"

        extra_keys = [
            "general_text",
            "policy_news",
            "stock_specific",
            "instruction_conversation",
        ]
        for idx, key in enumerate(extra_keys, start=1):
            batch_rows = list(china_buckets.get(key, []) or [])
            added_rows, stopped = _append_batch(batch_rows, bucket=key)
            _emit(
                70 + int((idx / max(1, len(extra_keys))) * 6),
                f"China corpus {key}: raw={len(batch_rows)} new={added_rows}",
                "collect",
            )
            if stopped:
                stopped_payload = _build_stopped_payload()
                self._write_training_status(stopped_payload)
                return stopped_payload

        rows, quality_stats = self._filter_high_quality_articles(
            rows,
            hours_back=max(12, int(hours_back)),
        )
        rows.sort(key=lambda a: getattr(a, "published_at", datetime.min), reverse=True)
        rows = rows[: max(80, int(max_samples))]

        if not rows:
            report = {
                "status": "no_new_data",
                "model_name": self.model_name,
                "trained_samples": 0,
                "collected_articles": 0,
                "new_articles": 0,
                "reused_articles_skipped": int(skipped_seen),
                "hours_back": int(hours_back),
                "limit_per_query": int(limit_per_query),
                "query_count": int(len(queries)),
                "related_stock_codes": list(related_codes),
                "related_codes_source": str(related_codes_source),
                "china_direct_mode": bool(env_flag("TRADING_CHINA_DIRECT", "0")),
                "corpus_breakdown": dict(corpus_breakdown),
                "quality_filter": dict(quality_stats),
                "training_architecture": "hybrid_neural_network",
                "strict_batch_failures": int(strict_batch_failures),
                "strict_batch_recoveries": int(strict_batch_recoveries),
                "notes": (
                    "No unseen language data collected in this cycle."
                    if strict_batch_failures <= 0
                    else (
                        "No unseen language data collected in this cycle. "
                        f"Strict batch failures={strict_batch_failures}, "
                        f"recoveries={strict_batch_recoveries}."
                    )
                ),
            }
            _emit(100, "No new language data this cycle.", "complete")
            self._write_training_status(report)
            return report

        min_new = max(1, int(min_new_articles))
        if len(rows) < min_new:
            report = {
                "status": "no_new_data",
                "model_name": self.model_name,
                "trained_samples": 0,
                "collected_articles": int(len(rows)),
                "new_articles": int(len(rows)),
                "reused_articles_skipped": int(skipped_seen),
                "hours_back": int(hours_back),
                "limit_per_query": int(limit_per_query),
                "query_count": int(len(queries)),
                "related_stock_codes": list(related_codes),
                "related_codes_source": str(related_codes_source),
                "china_direct_mode": bool(env_flag("TRADING_CHINA_DIRECT", "0")),
                "corpus_breakdown": dict(corpus_breakdown),
                "quality_filter": dict(quality_stats),
                "training_architecture": "hybrid_neural_network",
                "strict_batch_failures": int(strict_batch_failures),
                "strict_batch_recoveries": int(strict_batch_recoveries),
                "notes": f"Only {len(rows)} new articles collected (min={min_new}); skipping training.",
            }
            _emit(100, f"Too few new articles ({len(rows)}<{min_new}); skipping training.", "complete")
            self._write_training_status(report)
            return report

        _emit(
            78,
            (
                f"Collected {len(rows)} high-quality rows "
                f"(dropped={max(0, int(quality_stats.get('input', 0)) - int(quality_stats.get('kept', 0)))}); "
                "starting hybrid training."
            ),
            "train",
        )
        report = self.train(rows, max_samples=max_samples)
        # Don't downgrade a successful "trained" status to "stopped".
        # Only mark stopped if training itself didn't complete successfully.
        train_status = str(report.get("status", "")).lower()
        if _is_stopped() and train_status not in {"trained", "complete", "ok", "error", "failed"}:
            report["status"] = "stopped"
        status_lower = str(report.get("status", "")).strip().lower()
        if status_lower not in {"error", "failed", "stopped"}:
            # Only remember articles that train() actually used (capped by train's own max_samples).
            trained_cap = max(50, int(max_samples))
            self._remember_seen_articles(
                [
                    str(getattr(a, "id", "") or "")
                    for a in rows[:trained_cap]
                ]
            )
        report["collected_articles"] = int(len(rows))
        report["new_articles"] = int(len(rows))
        report["reused_articles_skipped"] = int(skipped_seen)
        report["only_new_mode"] = bool(only_new)
        report["query_count"] = int(len(queries))
        report["related_stock_codes"] = list(related_codes)
        report["related_codes_source"] = str(related_codes_source)
        report["china_direct_mode"] = bool(env_flag("TRADING_CHINA_DIRECT", "0"))
        report["corpus_breakdown"] = dict(corpus_breakdown)
        report["quality_filter"] = dict(quality_stats)
        report["hours_back"] = int(hours_back)
        report["limit_per_query"] = int(limit_per_query)
        report["strict_batch_failures"] = int(strict_batch_failures)
        report["strict_batch_recoveries"] = int(strict_batch_recoveries)
        report["training_architecture"] = str(
            report.get("training_architecture") or "hybrid_neural_network"
        )
        _emit(
            100,
            "Auto internet hybrid training completed."
            if str(report.get("status", "")).lower() not in {"stopped", "error", "failed"}
            else f"Auto training status: {report.get('status', 'unknown')}",
            "complete",
        )
        self._write_training_status(report)
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
