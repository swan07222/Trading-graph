"""Bilingual LLM sentiment analyzer with auto-training."""

from __future__ import annotations

import json
import math
import os
import pickle
import re
import time
from collections.abc import Callable
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

# FIX: Define recoverable exceptions for LLM operations
_LLM_CHAT_RECOVERABLE_EXCEPTIONS = (
    ImportError,
    OSError,
    RuntimeError,
    ValueError,
    TypeError,
    KeyError,
    IndexError,
    AttributeError,
    json.JSONDecodeError,
    TimeoutError,
    ConnectionError,
    ConnectionResetError,
    ConnectionAbortedError,
    ConnectionRefusedError,
)

_TRANSFORMERS_AVAILABLE = False
_SKLEARN_AVAILABLE = False
_SKLEARN_MLP_AVAILABLE = False
_SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    from sklearn.exceptions import ConvergenceWarning
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
        self._cache: dict[str, tuple[LLMSentimentResult, float]] = {}
        self._cache_ttl = 300.0
        self._load_calibrator()
        self._load_hybrid_calibrator()

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
            with self._status_path.open("w", encoding="utf-8") as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def get_training_status(self) -> dict[str, Any]:
        if self._status_path.exists():
            try:
                with self._status_path.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    return data
            except Exception:
                pass
        return {
            "status": "not_trained",
            "training_architecture": "hybrid_neural_network",
            "artifact_dir": str(self.cache_dir),
            "calibrator_ready": bool(self._calibrator is not None),
            "hybrid_nn_ready": bool(self._hybrid_calibrator is not None),
        }

    def train(self, articles: list[NewsArticle], *, epochs: int = 3, max_samples: int = 1000, learning_rate: float = 2e-5) -> dict[str, Any]:
        _ = (epochs, learning_rate)
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

        notes: list[str] = []
        class_diverse = len(set(y)) >= 2
        x_arr = np.asarray(x, dtype=float)
        y_arr = np.asarray(y, dtype=int)

        logreg_ready = False
        hybrid_nn_ready = False

        if not class_diverse:
            notes.append("Not enough class diversity for calibration fit.")
        elif _SKLEARN_AVAILABLE:
            try:
                clf = LogisticRegression(
                    max_iter=320,
                    multi_class="auto",
                    class_weight="balanced",
                )
                clf.fit(x_arr, y_arr)
                self._calibrator = clf
                self._save_calibrator()
                logreg_ready = True
            except Exception as exc:
                notes.append(f"Logistic calibration fit failed: {exc}")
        else:
            notes.append("scikit-learn unavailable; logistic calibrator skipped.")

        if class_diverse and _SKLEARN_MLP_AVAILABLE and len(rows) >= 30:
            try:
                nn = MLPClassifier(
                    hidden_layer_sizes=(32, 16),
                    activation="relu",
                    solver="adam",
                    alpha=1e-3,
                    learning_rate_init=3e-3,
                    batch_size=min(128, max(16, len(rows) // 8)),
                    max_iter=320,
                    random_state=42,
                    early_stopping=True,
                    validation_fraction=0.1,
                    n_iter_no_change=15,
                )
                nn.fit(x_arr, y_arr)
                self._hybrid_calibrator = nn
                self._save_hybrid_calibrator()
                hybrid_nn_ready = True
            except (ConvergenceWarning, UserWarning) as exc:
                # Handle convergence warnings gracefully - model may still be usable
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
        elif len(rows) < 30:
            notes.append("Insufficient samples for stable hybrid NN fit.")

        if logreg_ready or hybrid_nn_ready:
            status = "trained"
        else:
            status = "partial" if rows else "skipped"

        architecture = "hybrid_neural_network"
        if not (logreg_ready or hybrid_nn_ready):
            architecture = "rule_based_fallback"

        out = {
            "status": status,
            "model_name": self.model_name,
            "trained_samples": int(len(rows)),
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
    ) -> dict[str, Any]:
        if force_china_direct:
            os.environ["TRADING_CHINA_DIRECT"] = "1"
            os.environ["TRADING_VPN"] = "0"
            try:
                from core.network import invalidate_network_cache

                invalidate_network_cache()
            except Exception:
                pass

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

        collector = get_collector()
        queries = [
            ["政策", "监管", "A股", "央行"],
            ["上市公司", "业绩", "产业政策"],
            ["沪深", "板块", "宏观政策", "产业链"],
            ["China stock policy", "regulation", "market"],
            ["Federal Reserve", "SEC", "China ADR"],
        ]
        seen: set[str] = set()
        rows: list[NewsArticle] = []
        _emit(2, "Starting internet collection...", "start")
        for i, kw in enumerate(queries):
            if _is_stopped():
                stopped = {
                    "status": "stopped",
                    "collected_articles": int(len(rows)),
                    "hours_back": int(hours_back),
                    "limit_per_query": int(limit_per_query),
                    "training_architecture": "hybrid_neural_network",
                }
                self._write_training_status(stopped)
                return stopped
            try:
                batch = collector.collect_news(
                    keywords=list(kw),
                    limit=max(20, int(limit_per_query)),
                    hours_back=max(12, int(hours_back)),
                )
            except _LLM_CHAT_RECOVERABLE_EXCEPTIONS as exc:
                log.warning("LLM sentiment collection failed for query %d: %s", i + 1, exc)
                batch = []
            except Exception as exc:
                log.error("LLM sentiment collection unexpected error for query %d: %s", i + 1, exc)
                batch = []
            _emit(
                8 + int(((i + 1) / max(1, len(queries))) * 62),
                f"Collected batch {i + 1}/{len(queries)} ({len(batch)} rows)",
                "collect",
            )
            for a in batch:
                aid = str(getattr(a, "id", "") or "")
                if not aid or aid in seen:
                    continue
                seen.add(aid)
                rows.append(a)
                if _is_stopped():
                    stopped = {
                        "status": "stopped",
                        "collected_articles": int(len(rows)),
                        "hours_back": int(hours_back),
                        "limit_per_query": int(limit_per_query),
                        "training_architecture": "hybrid_neural_network",
                    }
                    self._write_training_status(stopped)
                    return stopped
        rows.sort(key=lambda a: getattr(a, "published_at", datetime.min), reverse=True)
        rows = rows[: max(80, int(max_samples))]
        _emit(78, f"Collected {len(rows)} deduplicated rows; starting hybrid training.", "train")
        report = self.train(rows, max_samples=max_samples)
        if _is_stopped() and str(report.get("status", "")).lower() not in {"error", "failed"}:
            report["status"] = "stopped"
        report["collected_articles"] = int(len(rows))
        report["hours_back"] = int(hours_back)
        report["limit_per_query"] = int(limit_per_query)
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
