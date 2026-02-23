from __future__ import annotations

import copy
import math
import threading
import time
from collections import deque
from datetime import datetime

import numpy as np

from data.news import (
    _DEDUP_PREFIX_LEN,
    _NEWS_BUFFER_SIZE,
    _NEWS_CACHE_TTL,
    _POLICY_KEYWORDS,
    EastmoneyNewsFetcher,
    NewsItem,
    SinaNewsFetcher,
    TencentNewsFetcher,
    _safe_age_hours_from_now,
    _safe_age_seconds_from_now,
)
from utils.logger import get_logger
from utils.type_utils import safe_float

log = get_logger(__name__)

class NewsAggregator:
    """Aggregates news from multiple sources with caching.
    Network-aware: uses different sources based on China/VPN.
    """

    def __init__(self):
        self._sina = SinaNewsFetcher()
        self._eastmoney = EastmoneyNewsFetcher()
        self._tencent = TencentNewsFetcher()

        self._cache: dict[str, list[NewsItem]] = {}
        self._cache_time: dict[str, float] = {}
        self._cache_ttl: int = _NEWS_CACHE_TTL
        self._lock = threading.RLock()

        # Rolling news buffer (last N items)
        self._all_news: deque[NewsItem] = deque(maxlen=_NEWS_BUFFER_SIZE)
        self._source_health: dict[str, dict[str, object]] = {
            "tencent": {
                "ok_calls": 0,
                "failed_calls": 0,
                "last_success_ts": 0.0,
                "last_error": "",
                "last_items": 0,
            },
            "sina": {
                "ok_calls": 0,
                "failed_calls": 0,
                "last_success_ts": 0.0,
                "last_error": "",
                "last_items": 0,
            },
            "eastmoney_policy": {
                "ok_calls": 0,
                "failed_calls": 0,
                "last_success_ts": 0.0,
                "last_error": "",
                "last_items": 0,
            },
            "eastmoney_stock": {
                "ok_calls": 0,
                "failed_calls": 0,
                "last_success_ts": 0.0,
                "last_error": "",
                "last_items": 0,
            },
        }

    # -- cache helpers -------------------------------------------------------

    def _is_cache_valid(self, key: str) -> bool:
        """Check if cached entry exists and is within TTL."""
        return (
            key in self._cache
            and (time.time() - self._cache_time.get(key, 0)) < self._cache_ttl
        )

    def _record_source_result(
        self, source: str, ok: bool, item_count: int = 0, error: str = ""
    ) -> None:
        with self._lock:
            state = self._source_health.get(source)
            if state is None:
                state = {
                    "ok_calls": 0,
                    "failed_calls": 0,
                    "last_success_ts": 0.0,
                    "last_error": "",
                    "last_items": 0,
                }
                self._source_health[source] = state

            if ok:
                state["ok_calls"] = int(state.get("ok_calls", 0)) + 1
                state["last_success_ts"] = float(time.time())
                state["last_error"] = ""
                state["last_items"] = int(item_count)
            else:
                state["failed_calls"] = int(state.get("failed_calls", 0)) + 1
                state["last_error"] = str(error)[:240]

    def _source_reliability_weight(self, source: str) -> float:
        """Reliability prior from rolling source health.
        Returns [0.5, 1.3] so weak sources are down-weighted, not removed.
        """
        with self._lock:
            state = self._source_health.get(source, {})
            ok_calls = int(state.get("ok_calls", 0))
            failed_calls = int(state.get("failed_calls", 0))
        total = ok_calls + failed_calls
        if total <= 0:
            return 1.0
        rate = ok_calls / float(total)
        return float(min(1.3, max(0.5, 0.6 + 0.9 * rate)))

    @staticmethod
    def _norm_code(value: str) -> str:
        digits = "".join(ch for ch in str(value or "") if ch.isdigit())
        return digits.zfill(6) if digits else ""

    def _item_mentions_code(self, item: NewsItem, code6: str) -> bool:
        code6 = self._norm_code(code6)
        if not code6:
            return False
        title = str(getattr(item, "title", "") or "")
        content = str(getattr(item, "content", "") or "")
        if code6 in title or code6 in content:
            return True
        for v in list(getattr(item, "stock_codes", []) or []):
            if self._norm_code(str(v)) == code6:
                return True
        return False

    @staticmethod
    def _publish_recency_key(item: NewsItem) -> float:
        """Stable recency sort key for mixed naive/aware publish_time values.
        Smaller means newer. Missing timestamps are pushed to the end.
        """
        age_s = _safe_age_seconds_from_now(getattr(item, "publish_time", None))
        if age_s is None:
            return float("inf")
        return max(0.0, float(age_s))

    # -- market news ---------------------------------------------------------

    def get_market_news(
        self, count: int = 30, force_refresh: bool = False
    ) -> list[NewsItem]:
        """Get aggregated market news from all available sources."""
        cache_key = f"market_{count}"

        with self._lock:
            if not force_refresh and self._is_cache_valid(cache_key):
                return list(self._cache[cache_key])

        from core.network import get_network_env
        env = get_network_env()

        all_items: list[NewsItem] = []
        should_try_tencent = bool(env.tencent_ok) or (not env.is_china_direct)
        should_try_sina = bool(env.is_china_direct) or bool(env.eastmoney_ok)
        should_try_eastmoney_policy = bool(env.eastmoney_ok) or bool(env.is_china_direct)

        # If detector is stale/incorrect, still do one best-effort Tencent fetch.
        if not (should_try_tencent or should_try_sina or should_try_eastmoney_policy):
            should_try_tencent = True

        if should_try_tencent:
            try:
                fetched = self._tencent.fetch_market_news(count)
                all_items.extend(fetched)
                self._record_source_result("tencent", True, len(fetched))
            except Exception as exc:
                self._record_source_result("tencent", False, error=str(exc))

        if should_try_sina:
            try:
                fetched = self._sina.fetch_market_news(count)
                all_items.extend(fetched)
                self._record_source_result("sina", True, len(fetched))
            except Exception as exc:
                self._record_source_result("sina", False, error=str(exc))

        if should_try_eastmoney_policy:
            try:
                fetched = self._eastmoney.fetch_policy_news(count)
                all_items.extend(fetched)
                self._record_source_result("eastmoney_policy", True, len(fetched))
            except Exception as exc:
                self._record_source_result("eastmoney_policy", False, error=str(exc))

        unique = self._deduplicate(all_items)
        unique.sort(key=self._publish_recency_key)
        unique = unique[:count]

        # Institutional fail-safe: stale cache fallback if all providers fail.
        if not unique:
            with self._lock:
                stale = list(self._cache.get(cache_key, []))
            if stale:
                log.warning(
                    "News providers returned no items; serving stale cache "
                    f"({len(stale)} items)"
                )
                return stale[:count]

        with self._lock:
            self._cache[cache_key] = unique
            self._cache_time[cache_key] = time.time()
            for item in unique:
                self._all_news.appendleft(item)

        log.info(f"Aggregated {len(unique)} market news items")
        return unique

    # -- stock news ----------------------------------------------------------

    def get_stock_news(
        self,
        stock_code: str,
        count: int = 15,
        force_refresh: bool = False,
    ) -> list[NewsItem]:
        """Get news for a specific stock."""
        code6 = self._norm_code(stock_code)
        if not code6:
            return []
        cache_key = f"stock_{code6}_{count}"

        with self._lock:
            if not force_refresh and self._is_cache_valid(cache_key):
                return list(self._cache[cache_key])

        from core.network import get_network_env
        env = get_network_env()

        all_items: list[NewsItem] = []
        context_items: list[NewsItem] = []

        should_try_sina = bool(env.is_china_direct) or bool(env.tencent_ok) or bool(env.eastmoney_ok)
        should_try_eastmoney = bool(env.eastmoney_ok) or bool(env.is_china_direct)

        if should_try_sina:
            try:
                fetched = self._sina.fetch_stock_news(code6, count)
                all_items.extend(fetched)
                self._record_source_result("sina", True, len(fetched))
            except Exception as exc:
                self._record_source_result("sina", False, error=str(exc))

        if should_try_eastmoney:
            try:
                fetched = self._eastmoney.fetch_stock_news(code6, count)
                all_items.extend(fetched)
                self._record_source_result("eastmoney_stock", True, len(fetched))
            except Exception as exc:
                self._record_source_result("eastmoney_stock", False, error=str(exc))

        # FIX Bug 4: Access _all_news under lock for thread safety
        with self._lock:
            for item in self._all_news:
                if self._item_mentions_code(item, code6):
                    all_items.append(item)

        # Blend in market-wide context so downstream views are not limited
        # to symbol-tagged headlines only.
        # FIX Bug 3: Don't propagate force_refresh to avoid unnecessary
        # cache invalidation of unrelated market cache.
        market_pool = self.get_market_news(
            count=max(60, int(count) * 4),
            force_refresh=False,
        )
        for item in market_pool:
            if self._item_mentions_code(item, code6):
                all_items.append(item)
            else:
                context_items.append(item)

        unique_all = self._deduplicate(all_items + context_items)
        unique_all.sort(key=self._publish_recency_key)

        direct = [
            item for item in unique_all
            if self._item_mentions_code(item, code6)
        ]
        context = [
            item for item in unique_all
            if not self._item_mentions_code(item, code6)
        ]

        target_count = max(1, int(count))

        # Keep stock-relevant items first; only fallback to context if none exist.
        selected: list[NewsItem] = []
        candidate_pool = direct if direct else context
        for item in candidate_pool:
            selected.append(item)
            if len(selected) >= target_count:
                break

        selected.sort(key=self._publish_recency_key)
        selected = selected[:target_count]

        # Same stale-cache guard as market path.
        if not selected:
            with self._lock:
                stale = list(self._cache.get(cache_key, []))
            if stale:
                log.warning(
                    "Stock/news blend returned no items; serving stale cache "
                    f"for {code6} ({len(stale)} items)"
                )
                return stale[:target_count]

        # Avoid mutating shared market/news objects in-place.
        unique: list[NewsItem] = []
        for item in selected:
            cloned = copy.deepcopy(item)
            if code6 not in cloned.stock_codes:
                cloned.stock_codes.append(code6)
            unique.append(cloned)

        with self._lock:
            self._cache[cache_key] = unique
            self._cache_time[cache_key] = time.time()

        return unique

    # -- policy news ---------------------------------------------------------

    def get_policy_news(self, count: int = 10) -> list[NewsItem]:
        """Get policy/regulatory news only."""
        all_news = self.get_market_news(count=50)
        policy = [
            n for n in all_news
            if n.category == "policy"
            or any(kw in n.title for kw in _POLICY_KEYWORDS)
        ]
        return policy[:count]

    # -- sentiment summary ---------------------------------------------------

    def get_sentiment_summary(
        self,
        stock_code: str | None = None,
        _news: list[NewsItem] | None = None,
    ) -> dict:
        """Get aggregated sentiment for stock or market.

        FIX Bug 2: Accept optional _news parameter to break recursive call
        chain. When called from get_news_features(), the already-fetched
        news list is passed directly instead of re-fetching.
        """
        if _news is not None:
            news = _news
        elif stock_code:
            news = self.get_stock_news(stock_code)
        else:
            news = self.get_market_news()

        if not news:
            return {
                "overall_sentiment": 0.0,
                "simple_sentiment": 0.0,
                "importance_weighted_sentiment": 0.0,
                "label": "neutral",
                "confidence": 0.0,
                "weighted": True,
                "fusion_version": "2.1",
                "recency_half_life_hours": 18.0,
                "source_diversity": 0.0,
                "source_entropy": 0.0,
                "source_concentration_hhi": 0.0,
                "disagreement_index": 0.0,
                "novelty_score": 0.0,
                "average_age_hours": 0.0,
                "sentiment_momentum_6h": 0.0,
                "source_mix": {},
                "source_weight_mix": {},
                "source_contributions": {},
                "positive_count": 0,
                "negative_count": 0,
                "neutral_count": 0,
                "total": 0,
                "top_positive": [],
                "top_negative": [],
            }

        scores = [safe_float(getattr(n, "sentiment_score", 0.0), 0.0) for n in news]
        simple_avg = (sum(scores) / len(scores)) if scores else 0.0

        weighted_total = 0.0
        weight_sum = 0.0
        source_weighted: dict[str, float] = {}
        source_weight_sum: dict[str, float] = {}
        source_scores: dict[str, list[float]] = {}
        source_counts: dict[str, int] = {}
        source_weight_mass: dict[str, float] = {}
        headline_seen: dict[tuple[str, str], int] = {}
        recency_half_life_hours = 18.0
        novelty_values: list[float] = []
        weighted_age_hours = 0.0
        importance_weighted_total = 0.0
        importance_weight_sum = 0.0
        scores_recent_6h: list[float] = []
        scores_older_6h: list[float] = []

        for n in news:
            score = safe_float(getattr(n, "sentiment_score", 0.0), 0.0)
            importance = safe_float(getattr(n, "importance", 0.5), 0.5)
            age_h_raw = _safe_age_hours_from_now(getattr(n, "publish_time", None))
            age_h = max(0.0, float(age_h_raw)) if age_h_raw is not None else 0.0
            recency_w = float(0.5 ** (age_h / recency_half_life_hours))
            src = str(getattr(n, "source", "") or "").strip().lower()
            src_w = self._source_reliability_weight(src)
            title_norm = str(getattr(n, "title", "") or "").strip().lower()
            novelty_key = (src or "unknown", title_norm[:_DEDUP_PREFIX_LEN])
            repeat = int(headline_seen.get(novelty_key, 0))
            headline_seen[novelty_key] = repeat + 1
            novelty_w = 1.0 / (1.0 + 0.35 * repeat)
            novelty_values.append(float(novelty_w))

            w = max(
                0.03,
                recency_w * src_w * max(0.1, importance) * novelty_w,
            )
            weighted_age_hours += age_h * w
            importance_w = max(0.1, importance)
            importance_weighted_total += score * importance_w
            importance_weight_sum += importance_w
            if age_h <= 6.0:
                scores_recent_6h.append(score)
            else:
                scores_older_6h.append(score)

            weighted_total += score * w
            weight_sum += w
            source_weighted[src] = float(source_weighted.get(src, 0.0) + (score * w))
            source_weight_sum[src] = float(source_weight_sum.get(src, 0.0) + w)
            source_weight_mass[src] = float(source_weight_mass.get(src, 0.0) + w)
            source_scores.setdefault(src, []).append(score)
            source_counts[src] = int(source_counts.get(src, 0) + 1)

        overall = (weighted_total / weight_sum) if weight_sum > 0 else simple_avg
        importance_weighted = (
            importance_weighted_total / importance_weight_sum
            if importance_weight_sum > 0
            else overall
        )

        positive = [n for n in news if n.sentiment_label == "positive"]
        negative = [n for n in news if n.sentiment_label == "negative"]
        neutral = [n for n in news if n.sentiment_label == "neutral"]

        source_diversity = float(len(source_scores) / max(1, len(news)))
        source_probs = [
            float(cnt) / max(1.0, float(len(news)))
            for cnt in source_counts.values()
            if cnt > 0
        ]
        entropy_raw = -sum((p * math.log(p)) for p in source_probs if p > 0.0)
        entropy_norm = (
            entropy_raw / math.log(len(source_probs))
            if len(source_probs) > 1
            else 0.0
        )
        source_concentration_hhi = sum((p * p) for p in source_probs)
        centroid_scores = [
            float(np.mean(vals))
            for vals in source_scores.values()
            if vals
        ]
        disagreement = (
            float(np.std(np.asarray(centroid_scores, dtype=float)))
            if len(centroid_scores) > 1
            else 0.0
        )
        novelty_score = (
            float(np.mean(novelty_values))
            if novelty_values
            else 0.0
        )
        avg_age_hours = (
            weighted_age_hours / weight_sum
            if weight_sum > 0
            else 0.0
        )
        if scores_recent_6h and scores_older_6h:
            momentum_6h = float(np.mean(scores_recent_6h) - np.mean(scores_older_6h))
        elif scores_recent_6h:
            momentum_6h = float(np.mean(scores_recent_6h) - overall)
        else:
            momentum_6h = 0.0

        coverage = min(1.0, len(news) / 30.0)
        strength = min(1.0, abs(float(overall)))
        source_coverage = min(1.0, len(source_scores) / 4.0)
        diversity_quality = min(1.0, (0.6 * source_coverage) + (0.4 * entropy_norm))

        confidence = (
            (0.40 * coverage)
            + (0.30 * strength)
            + (0.20 * diversity_quality)
            + (0.10 * novelty_score)
        )
        confidence *= max(0.0, 1.0 - (0.35 * min(1.0, disagreement * 2.5)))
        confidence = min(
            1.0,
            max(
                0.0,
                confidence,
            ),
        )

        source_contributions = {
            src: round(
                float(source_weighted.get(src, 0.0))
                / max(1e-9, float(source_weight_sum.get(src, 0.0))),
                4,
            )
            for src in sorted(source_weighted.keys())
        }
        source_mix = {
            src: round(float(cnt) / max(1.0, float(len(news))), 4)
            for src, cnt in sorted(
                source_counts.items(), key=lambda kv: kv[1], reverse=True
            )
        }
        source_weight_mix = {
            src: round(float(w) / max(1e-9, float(weight_sum)), 4)
            for src, w in sorted(
                source_weight_mass.items(), key=lambda kv: kv[1], reverse=True
            )
        }

        return {
            "overall_sentiment": round(overall, 3),
            "simple_sentiment": round(simple_avg, 3),
            "importance_weighted_sentiment": round(float(importance_weighted), 3),
            "label": (
                "positive" if overall > 0.1
                else ("negative" if overall < -0.1 else "neutral")
            ),
            "confidence": round(float(confidence), 3),
            "weighted": True,
            "fusion_version": "2.1",
            "recency_half_life_hours": recency_half_life_hours,
            "source_diversity": round(float(source_diversity), 3),
            "source_entropy": round(float(entropy_norm), 3),
            "source_concentration_hhi": round(float(source_concentration_hhi), 3),
            "disagreement_index": round(float(disagreement), 3),
            "novelty_score": round(float(novelty_score), 3),
            "average_age_hours": round(float(avg_age_hours), 3),
            "sentiment_momentum_6h": round(float(momentum_6h), 3),
            "source_mix": source_mix,
            "source_weight_mix": source_weight_mix,
            "source_contributions": source_contributions,
            "positive_count": len(positive),
            "negative_count": len(negative),
            "neutral_count": len(neutral),
            "total": len(news),
            "top_positive": [
                n.to_dict()
                for n in sorted(
                    positive, key=lambda x: x.sentiment_score, reverse=True
                )[:3]
            ],
            "top_negative": [
                n.to_dict()
                for n in sorted(negative, key=lambda x: x.sentiment_score)[:3]
            ],
        }

    # -- numerical features for AI model -------------------------------------

    def get_news_features(
        self,
        stock_code: str | None = None,
        hours_lookback: int = 24,
    ) -> dict[str, float]:
        """Get numerical features from news for AI model input.
        These can be appended to the technical feature vector.

        FIX Bug 2: Fetch news once and pass to get_sentiment_summary
        to avoid recursive re-fetching.
        """
        news = (
            self.get_stock_news(stock_code, count=50)
            if stock_code
            else self.get_market_news(count=50)
        )

        try:
            lookback_h_raw = float(hours_lookback)
        except (TypeError, ValueError):
            lookback_h_raw = 24.0
        lookback_h = lookback_h_raw if lookback_h_raw > 0.0 else 1.0

        # FIX Bug 7: Store age per index instead of id() which can be
        # unreliable after list slicing/copying
        recent: list[NewsItem] = []
        recent_age_hours: list[float] = []
        for n in news:
            age_h = _safe_age_hours_from_now(getattr(n, "publish_time", None))
            if age_h is None:
                continue
            age_h = max(0.0, float(age_h))
            if age_h <= lookback_h:
                recent.append(n)
                recent_age_hours.append(age_h)

        if not recent:
            return {
                "news_sentiment_avg": 0.0,
                "news_sentiment_std": 0.0,
                "news_weighted_sentiment": 0.0,
                "news_sentiment_disagreement": 0.0,
                "news_positive_ratio": 0.0,
                "news_negative_ratio": 0.0,
                "news_volume": 0.0,
                "news_importance_avg": 0.5,
                "news_recency_score": 0.0,
                "news_source_diversity": 0.0,
                "news_sentiment_confidence": 0.0,
                "news_source_entropy": 0.0,
                "news_source_concentration_hhi": 0.0,
                "news_novelty_score": 0.0,
                "news_recent_momentum": 0.0,
                "news_importance_weighted_sentiment": 0.0,
                "news_weighted_vs_simple_gap": 0.0,
                "news_average_age_hours": 0.0,
                "news_disagreement_penalty": 1.0,
                "policy_sentiment": 0.0,
            }

        scores = [safe_float(getattr(n, "sentiment_score", 0.0), 0.0) for n in recent]
        total = len(scores)
        positive = sum(1 for s in scores if s > 0.1)
        negative = sum(1 for s in scores if s < -0.1)

        # Recency-weighted sentiment (newer news matters more)
        recency_weights: list[float] = []
        for age_hours in recent_age_hours:
            weight = max(0.1, 1.0 - (age_hours / lookback_h))
            recency_weights.append(weight)

        weight_sum = sum(recency_weights)
        weighted_sentiment = (
            sum(s * w for s, w in zip(scores, recency_weights, strict=False))
            / weight_sum
            if weight_sum > 0
            else 0.0
        )

        source_groups: dict[str, list[float]] = {}
        fused_w_sum = 0.0
        fused_s_sum = 0.0
        for idx, n in enumerate(recent):
            src = str(getattr(n, "source", "") or "").strip().lower()
            source_groups.setdefault(src, []).append(
                safe_float(getattr(n, "sentiment_score", 0.0), 0.0)
            )
            src_w = self._source_reliability_weight(src)
            age_h = recent_age_hours[idx]
            rec_w = float(0.5 ** (age_h / 18.0))
            imp_w = max(0.1, safe_float(getattr(n, "importance", 0.5), 0.5))
            w = max(0.03, src_w * rec_w * imp_w)
            fused_w_sum += w
            fused_s_sum += safe_float(getattr(n, "sentiment_score", 0.0), 0.0) * w

        fused_sentiment = fused_s_sum / fused_w_sum if fused_w_sum > 0 else weighted_sentiment
        source_disagreement = (
            float(np.std([float(np.mean(v)) for v in source_groups.values() if v]))
            if len(source_groups) > 1
            else 0.0
        )
        source_diversity = float(len(source_groups) / max(1, len(recent)))

        # Policy-specific sentiment
        policy_items = [n for n in recent if n.category == "policy"]
        policy_sentiment = (
            sum(
                safe_float(getattr(n, "sentiment_score", 0.0), 0.0)
                for n in policy_items
            )
            / len(policy_items)
            if policy_items
            else 0.0
        )

        importances = [safe_float(getattr(n, "importance", 0.5), 0.5) for n in recent]

        # FIX Bug 2: Pass already-fetched news to avoid recursive calls
        summary = self.get_sentiment_summary(stock_code=stock_code, _news=news)
        summary_overall = float(summary.get("overall_sentiment", 0.0) or 0.0)
        summary_simple = float(summary.get("simple_sentiment", 0.0) or 0.0)
        summary_disagreement = float(summary.get("disagreement_index", 0.0) or 0.0)
        disagreement_penalty = max(0.0, 1.0 - min(1.0, summary_disagreement * 2.0))

        return {
            "news_sentiment_avg": round(float(np.mean(scores)), 4),
            "news_sentiment_std": (
                round(float(np.std(scores)), 4) if len(scores) > 1 else 0.0
            ),
            "news_weighted_sentiment": round(float(fused_sentiment), 4),
            "news_sentiment_disagreement": round(float(source_disagreement), 4),
            "news_positive_ratio": round(positive / total, 4),
            "news_negative_ratio": round(negative / total, 4),
            "news_volume": min(total / 20.0, 1.0),  # Normalized 0-1
            "news_importance_avg": round(float(np.mean(importances)), 4),
            "news_recency_score": round(weighted_sentiment, 4),
            "news_source_diversity": round(float(source_diversity), 4),
            "news_sentiment_confidence": round(float(summary.get("confidence", 0.0) or 0.0), 4),
            "news_source_entropy": round(float(summary.get("source_entropy", 0.0) or 0.0), 4),
            "news_source_concentration_hhi": round(float(summary.get("source_concentration_hhi", 0.0) or 0.0), 4),
            "news_novelty_score": round(float(summary.get("novelty_score", 0.0) or 0.0), 4),
            "news_recent_momentum": round(float(summary.get("sentiment_momentum_6h", 0.0) or 0.0), 4),
            "news_importance_weighted_sentiment": round(float(summary.get("importance_weighted_sentiment", 0.0) or 0.0), 4),
            "news_weighted_vs_simple_gap": round(float(summary_overall - summary_simple), 4),
            "news_average_age_hours": round(float(summary.get("average_age_hours", 0.0) or 0.0), 4),
            "news_disagreement_penalty": round(float(disagreement_penalty), 4),
            "policy_sentiment": round(policy_sentiment, 4),
        }

    # -- deduplication -------------------------------------------------------

    @staticmethod
    def _deduplicate(items: list[NewsItem]) -> list[NewsItem]:
        """Remove duplicate news by title prefix similarity."""
        seen_titles: set[str] = set()
        unique: list[NewsItem] = []
        for item in items:
            # FIX Bug 5: Skip items with empty/whitespace-only titles
            # instead of adding empty string to seen set
            title = (item.title or "").strip()
            if not title:
                continue
            key = title[:_DEDUP_PREFIX_LEN]
            if key not in seen_titles:
                seen_titles.add(key)
                unique.append(item)
        return unique

    # -- cache management ----------------------------------------------------

    def clear_cache(self) -> None:
        with self._lock:
            self._cache.clear()
            self._cache_time.clear()

    def get_source_health(self) -> dict[str, dict[str, object]]:
        """Institutional telemetry: fetch-source reliability and freshness."""
        with self._lock:
            out: dict[str, dict[str, object]] = {}
            now_ts = float(time.time())
            for src, state in self._source_health.items():
                ok_calls = int(state.get("ok_calls", 0))
                failed_calls = int(state.get("failed_calls", 0))
                total = ok_calls + failed_calls
                success_rate = (ok_calls / total) if total > 0 else 1.0
                last_success_ts = float(state.get("last_success_ts", 0.0) or 0.0)
                age_s = (now_ts - last_success_ts) if last_success_ts > 0 else float("inf")
                out[src] = {
                    "ok_calls": ok_calls,
                    "failed_calls": failed_calls,
                    "success_rate": round(float(success_rate), 4),
                    "last_success_age_seconds": (
                        round(float(age_s), 1) if math.isfinite(age_s) else None
                    ),
                    "last_items": int(state.get("last_items", 0)),
                    "last_error": str(state.get("last_error", "")),
                }
            return out

    def get_institutional_snapshot(
        self, stock_code: str | None = None, hours_lookback: int = 24
    ) -> dict[str, object]:
        """Institutional-grade unified news snapshot.
        Includes sentiment, model features, source health, and freshness stats.
        """
        news = (
            self.get_stock_news(stock_code, count=60)
            if stock_code
            else self.get_market_news(count=60)
        )
        # FIX Bug 2: Pass fetched news to avoid redundant calls
        summary = self.get_sentiment_summary(stock_code=stock_code, _news=news)
        features = self.get_news_features(
            stock_code=stock_code, hours_lookback=hours_lookback
        )
        source_health = self.get_source_health()

        now = datetime.now()
        ages = [
            max(0.0, float(age_s))
            for age_s in (
                _safe_age_seconds_from_now(getattr(n, "publish_time", None))
                for n in news
            )
            if age_s is not None
        ]
        freshness = {
            "latest_age_seconds": round(float(min(ages)), 1) if ages else None,
            "median_age_seconds": round(float(np.median(ages)), 1) if ages else None,
            "items_with_timestamp": int(len(ages)),
        }

        source_counts: dict[str, int] = {}
        for n in news:
            src = str(getattr(n, "source", "") or "unknown").strip().lower()
            source_counts[src] = source_counts.get(src, 0) + 1
        total = max(len(news), 1)
        source_mix = {
            src: round(cnt / total, 4) for src, cnt in sorted(
                source_counts.items(), key=lambda kv: kv[1], reverse=True
            )
        }

        return {
            "scope": str(stock_code or "market"),
            "timestamp": now.isoformat(),
            "news_count": len(news),
            "source_mix": source_mix,
            "source_health": source_health,
            "freshness": freshness,
            "sentiment": summary,
            "features": features,
        }


# Thread-safe singleton

_aggregator: NewsAggregator | None = None
_aggregator_lock = threading.Lock()


def get_news_aggregator() -> NewsAggregator:
    """Double-checked locking singleton for NewsAggregator."""
    global _aggregator
    if _aggregator is None:
        with _aggregator_lock:
            if _aggregator is None:
                _aggregator = NewsAggregator()
    return _aggregator
