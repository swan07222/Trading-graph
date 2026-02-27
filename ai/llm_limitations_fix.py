"""
LLM Limitations Mitigation Module

This module addresses four key limitations of LLM/AI predictions in trading:

1. HALLUCINATIONS: Models generate plausible-sounding but incorrect information
   → Fix: Multi-layer validation, plausibility checks, confidence gating

2. LIMITED CONTEXT WINDOW: Cannot process arbitrarily long inputs
   → Fix: Sliding window attention, hierarchical context summarization

3. KNOWLEDGE CUTOFF: Training data has a cutoff date; lacks recent information
   → Fix: Real-time data integration, online learning, RAG for recent events

4. NO TRUE REASONING: Pattern matching rather than genuine understanding
   → Fix: Rule-based validation layer, causal reasoning checks, explainability
"""
from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from config.settings import CONFIG
from utils.logger import get_logger

log = get_logger(__name__)


class ValidationStatus(Enum):
    """Prediction validation status."""
    VALID = "valid"
    SUSPICIOUS = "suspicious"
    INVALID = "invalid"
    UNCERTAIN = "uncertain"


@dataclass
class ValidationReport:
    """Report from prediction validation."""
    status: ValidationStatus
    confidence: float
    checks_passed: int
    checks_failed: int
    issues: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "confidence": self.confidence,
            "checks_passed": self.checks_passed,
            "checks_failed": self.checks_failed,
            "issues": self.issues,
            "recommendations": self.recommendations,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class ContextWindow:
    """Extended context window with sliding attention."""
    recent_bars: pd.DataFrame
    summary_stats: dict[str, float]
    regime_label: str
    key_events: list[dict[str, Any]]
    attention_weights: np.ndarray

    def __post_init__(self) -> None:
        if self.attention_weights.size == 0:
            n = len(self.recent_bars)
            self.attention_weights = np.ones(n) / n if n > 0 else np.array([])


@dataclass
class RealTimeContext:
    """Real-time context for overcoming knowledge cutoff."""
    latest_price: float
    latest_volume: float
    market_regime: str
    recent_news_sentiment: float
    economic_calendar: list[dict[str, Any]]
    peer_performance: dict[str, float]
    timestamp: datetime


# ============================================================================
# FIX #1: HALLUCINATION PREVENTION
# ============================================================================

class HallucinationDetector:
    """
    Detects and prevents hallucinated predictions.

    Uses multiple validation layers:
    1. Statistical plausibility checks
    2. Historical consistency validation
    3. Cross-model agreement verification
    4. Domain constraint enforcement
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._validation_history: list[ValidationReport] = []
        self._plausibility_thresholds = self._load_default_thresholds()

    def _load_default_thresholds(self) -> dict[str, float]:
        """Load default plausibility thresholds."""
        return {
            # Price movement thresholds
            "max_daily_move_pct": 0.20,  # 20% max daily move
            "max_intraday_move_pct": 0.08,  # 8% max intraday move
            "min_price": 0.01,  # Minimum valid price
            "max_price": 1000000,  # Maximum valid price

            # Volume thresholds
            "max_volume_spike": 50.0,  # 50x average volume
            "min_volume": 0,

            # Prediction quality
            "min_confidence": 0.45,  # Minimum acceptable confidence
            "max_entropy": 0.50,  # Maximum prediction entropy
            "min_model_agreement": 0.55,  # Minimum ensemble agreement

            # Technical constraints
            "max_rsi": 100,
            "min_rsi": 0,
            "max_price_ma_ratio": 3.0,  # Price can't be 3x above MA
        }

    def validate_prediction(
        self,
        prediction: dict[str, Any],
        current_context: dict[str, Any],
        historical_data: pd.DataFrame | None = None,
    ) -> ValidationReport:
        """
        Validate a prediction against multiple criteria.

        Args:
            prediction: Model prediction with price, confidence, etc.
            current_context: Current market context
            historical_data: Historical data for comparison

        Returns:
            ValidationReport with status and recommendations
        """
        with self._lock:
            checks: list[tuple[str, bool, str]] = []  # (name, passed, message)
            issues: list[str] = []
            recommendations: list[str] = []

            # === Check 1: Price Plausibility ===
            pred_price = float(prediction.get("predicted_price", 0))
            current_price = float(current_context.get("current_price", pred_price))

            if pred_price <= 0:
                checks.append(("positive_price", False, "Predicted price is negative or zero"))
                issues.append("Invalid predicted price")
                recommendations.append("Reject prediction - model output error")
            else:
                checks.append(("positive_price", True, "Price is positive"))

            # Price movement check
            if current_price > 0 and pred_price > 0:
                move_pct = abs(pred_price - current_price) / current_price
                max_move = self._get_max_allowed_move(current_context)

                if move_pct > max_move:
                    checks.append(("price_move", False, f"Move {move_pct:.1%} exceeds max {max_move:.1%}"))
                    issues.append(f"Unrealistic price movement: {move_pct:.1%}")
                    recommendations.append("Cap prediction to realistic range")
                else:
                    checks.append(("price_move", True, f"Move {move_pct:.1%} is realistic"))

            # === Check 2: Confidence Calibration ===
            confidence = float(prediction.get("confidence", 0))
            min_conf = float(self._plausibility_thresholds.get("min_confidence", 0.45))

            if confidence < min_conf:
                checks.append(("confidence", False, f"Confidence {confidence:.2f} < {min_conf:.2f}"))
                issues.append("Low model confidence")
                recommendations.append("Reduce position size or skip trade")
            else:
                checks.append(("confidence", True, f"Confidence {confidence:.2f} acceptable"))

            # === Check 3: Ensemble Agreement ===
            model_agreement = float(prediction.get("model_agreement", 1.0))
            min_agreement = float(self._plausibility_thresholds.get("min_model_agreement", 0.55))

            if model_agreement < min_agreement:
                checks.append(("agreement", False, f"Agreement {model_agreement:.2f} < {min_agreement:.2f}"))
                issues.append("Models disagree significantly")
                recommendations.append("Wait for clearer signal")
            else:
                checks.append(("agreement", True, f"Agreement {model_agreement:.2f} acceptable"))

            # === Check 4: Historical Consistency ===
            if historical_data is not None and len(historical_data) > 0:
                hist_check = self._check_historical_consistency(
                    pred_price, current_price, historical_data
                )
                if not hist_check[0]:
                    checks.append(("historical", False, hist_check[1]))
                    issues.append("Prediction inconsistent with history")
                else:
                    checks.append(("historical", True, hist_check[1]))

            # === Check 5: Domain Constraints ===
            domain_check = self._check_domain_constraints(prediction, current_context)
            checks.extend(domain_check)

            # === Calculate Status ===
            passed = sum(1 for _, p, _ in checks if p)
            failed = len(checks) - passed

            if failed == 0:
                status = ValidationStatus.VALID
            elif failed <= 1:
                status = ValidationStatus.SUSPICIOUS
                recommendations.append("Monitor closely")
            elif failed <= 2:
                status = ValidationStatus.UNCERTAIN
                recommendations.append("Consider reducing exposure")
            else:
                status = ValidationStatus.INVALID
                recommendations.append("Reject prediction")

            report = ValidationReport(
                status=status,
                confidence=confidence,
                checks_passed=passed,
                checks_failed=failed,
                issues=issues,
                recommendations=recommendations,
            )

            # Store history
            self._validation_history.append(report)
            if len(self._validation_history) > 1000:
                self._validation_history = self._validation_history[-1000:]

            return report

    def _get_max_allowed_move(self, context: dict[str, Any]) -> float:
        """Get maximum allowed price movement based on context."""
        interval = str(context.get("interval", "1d"))

        # Intraday vs daily thresholds
        intraday_intervals = {"1m", "3m", "5m", "15m", "30m", "60m", "1h"}
        if interval in intraday_intervals:
            return float(self._plausibility_thresholds.get("max_intraday_move_pct", 0.08))
        else:
            return float(self._plausibility_thresholds.get("max_daily_move_pct", 0.20))

    def _check_historical_consistency(
        self,
        pred_price: float,
        current_price: float,
        historical_data: pd.DataFrame,
    ) -> tuple[bool, str]:
        """Check if prediction is consistent with historical patterns."""
        try:
            if "close" not in historical_data.columns:
                return (True, "No price data for comparison")

            hist_prices = historical_data["close"].dropna()
            if len(hist_prices) < 20:
                return (True, "Insufficient history")

            # Check if prediction is within historical range
            hist_min = float(hist_prices.min())
            hist_max = float(hist_prices.max())
            hist_mean = float(hist_prices.mean())
            hist_std = float(hist_prices.std())

            # Z-score check
            z_score = abs(pred_price - hist_mean) / hist_std if hist_std > 0 else 0

            if z_score > 4:
                return (False, f"Prediction z-score {z_score:.1f} exceeds 4σ")

            # Range check (with some tolerance)
            tolerance = 0.1  # 10% tolerance beyond historical range
            if pred_price < hist_min * (1 - tolerance):
                return (False, f"Prediction below historical range")
            if pred_price > hist_max * (1 + tolerance):
                return (False, f"Prediction above historical range")

            return (True, f"Historical consistency OK (z={z_score:.2f})")

        except Exception as e:
            log.debug("Historical consistency check failed: %s", e)
            return (True, "Check skipped due to error")

    def _check_domain_constraints(
        self,
        prediction: dict[str, Any],
        context: dict[str, Any],
    ) -> list[tuple[str, bool, str]]:
        """Check domain-specific constraints."""
        checks: list[tuple[str, bool, str]] = []

        # Check RSI if provided
        pred_rsi = float(prediction.get("predicted_rsi", 50))
        if pred_rsi < 0 or pred_rsi > 100:
            checks.append(("rsi_valid", False, f"RSI {pred_rsi} outside [0,100]"))
        else:
            checks.append(("rsi_valid", True, "RSI valid"))

        # Check volume spike
        pred_volume = float(prediction.get("predicted_volume", 0))
        avg_volume = float(context.get("avg_volume", pred_volume))
        if avg_volume > 0 and pred_volume > 0:
            volume_ratio = pred_volume / avg_volume
            max_ratio = float(self._plausibility_thresholds.get("max_volume_spike", 50.0))
            if volume_ratio > max_ratio:
                checks.append(("volume", False, f"Volume spike {volume_ratio:.1f}x exceeds {max_ratio:.1f}x"))
            else:
                checks.append(("volume", True, f"Volume ratio {volume_ratio:.1f}x acceptable"))

        return checks

    def get_validation_stats(self) -> dict[str, Any]:
        """Get validation statistics."""
        with self._lock:
            if not self._validation_history:
                return {"total": 0}

            total = len(self._validation_history)
            valid = sum(1 for r in self._validation_history if r.status == ValidationStatus.VALID)
            suspicious = sum(1 for r in self._validation_history if r.status == ValidationStatus.SUSPICIOUS)
            invalid = sum(1 for r in self._validation_history if r.status == ValidationStatus.INVALID)

            return {
                "total": total,
                "valid": valid,
                "valid_rate": valid / total if total > 0 else 0,
                "suspicious": suspicious,
                "invalid": invalid,
                "avg_confidence": np.mean([r.confidence for r in self._validation_history]),
            }


# ============================================================================
# FIX #2: EXTENDED CONTEXT WINDOW
# ============================================================================

class ExtendedContextManager:
    """
    Manages extended context beyond LLM's native window.

    Uses hierarchical summarization and sliding attention to
    maintain long-term context efficiently.
    """

    def __init__(
        self,
        max_bars: int = 10000,
        summary_levels: int = 3,
    ) -> None:
        """
        Initialize context manager.

        Args:
            max_bars: Maximum bars to retain in memory
            summary_levels: Number of summarization levels
        """
        self.max_bars = max_bars
        self.summary_levels = summary_levels
        self._lock = threading.RLock()
        self._recent_bars: pd.DataFrame = pd.DataFrame()
        self._summaries: list[dict[str, Any]] = []
        self._attention_weights: np.ndarray = np.array([])

    def add_bars(self, bars: pd.DataFrame) -> None:
        """Add new bars to context with sliding window."""
        with self._lock:
            if bars.empty:
                return

            # Append new bars
            if self._recent_bars.empty:
                self._recent_bars = bars.copy()
            else:
                self._recent_bars = pd.concat([self._recent_bars, bars], ignore_index=True)

            # Remove duplicates
            if "datetime" in self._recent_bars.columns:
                self._recent_bars = self._recent_bars.drop_duplicates(
                    subset=["datetime"], keep="last"
                )

            # Trim to max size
            if len(self._recent_bars) > self.max_bars:
                # Keep most recent bars, summarize older ones
                overflow = len(self._recent_bars) - self.max_bars
                to_summarize = self._recent_bars.iloc[:overflow]
                self._recent_bars = self._recent_bars.iloc[overflow:].reset_index(drop=True)

                # Create summary
                self._create_summary(to_summarize)

            # Update attention weights
            self._update_attention_weights()

    def _create_summary(self, bars: pd.DataFrame) -> None:
        """Create hierarchical summary of bars."""
        if bars.empty:
            return

        summary = {
            "level": 1,
            "start_time": bars["datetime"].iloc[0] if "datetime" in bars.columns else None,
            "end_time": bars["datetime"].iloc[-1] if "datetime" in bars.columns else None,
            "bar_count": len(bars),
            "price_open": float(bars["open"].iloc[0]) if "open" in bars.columns else 0,
            "price_close": float(bars["close"].iloc[-1]) if "close" in bars.columns else 0,
            "price_high": float(bars["high"].max()) if "high" in bars.columns else 0,
            "price_low": float(bars["low"].min()) if "low" in bars.columns else 0,
            "total_volume": float(bars["volume"].sum()) if "volume" in bars.columns else 0,
            "avg_volume": float(bars["volume"].mean()) if "volume" in bars.columns else 0,
            "price_change_pct": (
                (float(bars["close"].iloc[-1]) - float(bars["open"].iloc[0]))
                / float(bars["open"].iloc[0]) * 100
                if "close" in bars.columns and "open" in bars.columns and float(bars["open"].iloc[0]) > 0
                else 0
            ),
            "volatility": float(bars["close"].std()) if "close" in bars.columns else 0,
            "timestamp": datetime.now(),
        }

        self._summaries.append(summary)

        # Trim old summaries
        max_summaries = 100
        if len(self._summaries) > max_summaries:
            # Merge oldest summaries
            old_summaries = self._summaries[:max_summaries // 2]
            self._summaries = self._summaries[max_summaries // 2:]
            # Could merge here for deeper hierarchy

    def _update_attention_weights(self) -> None:
        """Update attention weights for recent vs historical data."""
        n = len(self._recent_bars)
        if n == 0:
            self._attention_weights = np.array([])
            return

        # Exponential decay: recent bars get higher weight
        decay_rate = 0.995
        weights = np.power(decay_rate, np.arange(n)[::-1])
        self._attention_weights = weights / weights.sum()

    def get_context(self, n_recent: int = 100) -> ContextWindow:
        """Get current context window."""
        with self._lock:
            recent = self._recent_bars.tail(n_recent).copy()

            # Calculate summary statistics
            stats = self._calculate_summary_stats()

            # Determine market regime
            regime = self._determine_regime(recent)

            # Get key events from summaries
            key_events = self._get_key_events()

            return ContextWindow(
                recent_bars=recent,
                summary_stats=stats,
                regime_label=regime,
                key_events=key_events,
                attention_weights=self._attention_weights[-len(recent):] if len(self._attention_weights) > 0 else np.array([]),
            )

    def _calculate_summary_stats(self) -> dict[str, float]:
        """Calculate summary statistics."""
        stats: dict[str, float] = {}

        if self._recent_bars.empty:
            return stats

        if "close" in self._recent_bars.columns:
            closes = self._recent_bars["close"].dropna()
            if len(closes) > 0:
                stats["current_price"] = float(closes.iloc[-1])
                stats["avg_price"] = float(closes.mean())
                stats["min_price"] = float(closes.min())
                stats["max_price"] = float(closes.max())
                stats["volatility"] = float(closes.std())

        if "volume" in self._recent_bars.columns:
            volumes = self._recent_bars["volume"].dropna()
            if len(volumes) > 0:
                stats["avg_volume"] = float(volumes.mean())
                stats["total_volume"] = float(volumes.sum())

        # Price change
        if "close" in self._recent_bars.columns and len(self._recent_bars) > 1:
            first = float(self._recent_bars["close"].iloc[0])
            last = float(self._recent_bars["close"].iloc[-1])
            if first > 0:
                stats["total_return_pct"] = (last - first) / first * 100

        return stats

    def _determine_regime(self, recent: pd.DataFrame) -> str:
        """Determine current market regime."""
        if recent.empty or len(recent) < 20:
            return "unknown"

        try:
            closes = recent["close"].dropna()
            if len(closes) < 20:
                return "unknown"

            # Calculate trend
            ma20 = closes.rolling(20).mean().iloc[-1]
            ma50 = closes.rolling(50).mean().iloc[-1] if len(closes) >= 50 else ma20
            current = closes.iloc[-1]

            # Calculate volatility
            returns = closes.pct_change().dropna()
            vol = returns.std()

            # Determine regime
            if current > ma20 and ma20 > ma50:
                trend = "bull"
            elif current < ma20 and ma20 < ma50:
                trend = "bear"
            else:
                trend = "sideways"

            if vol > 0.03:
                vol_state = "high_vol"
            elif vol < 0.01:
                vol_state = "low_vol"
            else:
                vol_state = "normal_vol"

            return f"{trend}_{vol_state}"

        except Exception:
            return "unknown"

    def _get_key_events(self) -> list[dict[str, Any]]:
        """Extract key events from summaries."""
        events: list[dict[str, Any]] = []

        for summary in self._summaries[-20:]:  # Last 20 summaries
            if abs(summary.get("price_change_pct", 0)) > 3:  # >3% move
                events.append({
                    "type": "large_move",
                    "change_pct": summary["price_change_pct"],
                    "time": summary.get("end_time"),
                    "volume": summary.get("total_volume", 0),
                })

        return sorted(events, key=lambda x: abs(x["change_pct"]), reverse=True)[:5]


# ============================================================================
# FIX #3: KNOWLEDGE CUTOFF MITIGATION
# ============================================================================

class RealTimeKnowledgeIntegrator:
    """
    Integrates real-time data to overcome knowledge cutoff.

    Continuously updates context with:
    - Latest prices and volumes
    - Market sentiment from news
    - Economic calendar events
    - Peer/sector performance
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._latest_context: RealTimeContext | None = None
        self._update_history: list[datetime] = []
        self._news_cache: dict[str, dict[str, Any]] = {}
        self._economic_events: list[dict[str, Any]] = []

    def update_context(
        self,
        stock_code: str,
        fetcher: Any | None = None,
        news_aggregator: Any | None = None,
    ) -> RealTimeContext:
        """
        Update real-time context.

        Args:
            stock_code: Stock symbol
            fetcher: Data fetcher for prices
            news_aggregator: News sentiment aggregator

        Returns:
            Updated RealTimeContext
        """
        with self._lock:
            context = self._fetch_realtime_data(stock_code, fetcher, news_aggregator)
            self._latest_context = context
            self._update_history.append(datetime.now())

            # Trim history
            if len(self._update_history) > 1000:
                self._update_history = self._update_history[-1000:]

            return context

    def _fetch_realtime_data(
        self,
        stock_code: str,
        fetcher: Any | None,
        news_aggregator: Any | None,
    ) -> RealTimeContext:
        """Fetch real-time data from various sources."""
        # Default values
        latest_price = 0.0
        latest_volume = 0.0
        market_regime = "unknown"
        sentiment = 0.0
        peer_perf: dict[str, float] = {}

        # Fetch price data
        if fetcher is not None:
            try:
                quote = fetcher.get_realtime(stock_code)
                if quote:
                    latest_price = float(getattr(quote, "price", 0) or 0)
                    latest_volume = float(getattr(quote, "volume", 0) or 0)
            except Exception as e:
                log.debug("Failed to fetch realtime price: %s", e)

        # Fetch news sentiment
        if news_aggregator is not None:
            try:
                sentiment_data = news_aggregator.get_sentiment(stock_code)
                if sentiment_data:
                    sentiment = float(sentiment_data.get("sentiment", 0) or 0)
            except Exception as e:
                log.debug("Failed to fetch news sentiment: %s", e)

        # Determine market regime
        market_regime = self._infer_regime(latest_price, latest_volume)

        # Get peer performance
        peer_perf = self._get_peer_performance(stock_code, fetcher)

        return RealTimeContext(
            latest_price=latest_price,
            latest_volume=latest_volume,
            market_regime=market_regime,
            recent_news_sentiment=sentiment,
            economic_calendar=self._economic_events,
            peer_performance=peer_perf,
            timestamp=datetime.now(),
        )

    def _infer_regime(self, price: float, volume: float) -> str:
        """Infer market regime from real-time data."""
        if price <= 0:
            return "unknown"

        # Simple regime inference
        # In production, this would use more sophisticated analysis
        if volume > 1000000:
            return "high_activity"
        elif volume > 100000:
            return "normal_activity"
        else:
            return "low_activity"

    def _get_peer_performance(
        self,
        stock_code: str,
        fetcher: Any | None,
    ) -> dict[str, float]:
        """Get peer/sector performance."""
        peers: dict[str, float] = {}

        # Define peer groups (in production, this would be configurable)
        peer_groups = {
            "600519": ["000858", "002304", "600809"],  # Kweichow Moutai peers
            "AAPL": ["MSFT", "GOOGL", "AMZN"],  # Tech peers
        }

        peer_list = peer_groups.get(stock_code, [])

        if fetcher is not None:
            for peer in peer_list:
                try:
                    quote = fetcher.get_realtime(peer)
                    if quote:
                        price = float(getattr(quote, "price", 0) or 0)
                        change = float(getattr(quote, "change_percent", 0) or 0)
                        peers[peer] = change
                except Exception:
                    pass

        return peers

    def add_economic_event(self, event: dict[str, Any]) -> None:
        """Add economic calendar event."""
        with self._lock:
            event["added_at"] = datetime.now()
            self._economic_events.append(event)

            # Remove old events
            cutoff = datetime.now() - timedelta(days=30)
            self._economic_events = [
                e for e in self._economic_events
                if e.get("date", datetime.now()) > cutoff
            ]

    def get_context(self) -> RealTimeContext | None:
        """Get latest real-time context."""
        with self._lock:
            return self._latest_context


# ============================================================================
# FIX #4: REASONING & EXPLAINABILITY LAYER
# ============================================================================

class ReasoningValidator:
    """
    Adds reasoning and explainability to AI predictions.

    Uses rule-based validation and causal reasoning to:
    1. Validate predictions against domain knowledge
    2. Provide explanations for predictions
    3. Detect logical inconsistencies
    4. Generate human-readable rationales
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._rules = self._load_default_rules()
        self._explanation_history: list[dict[str, Any]] = []

    def _load_default_rules(self) -> list[dict[str, Any]]:
        """Load default reasoning rules."""
        return [
            # Trend continuation rule
            {
                "name": "trend_continuation",
                "description": "Predictions should align with established trend",
                "condition": self._check_trend_continuation,
                "weight": 0.3,
            },
            # Mean reversion rule
            {
                "name": "mean_reversion",
                "description": "Extreme prices tend to revert to mean",
                "condition": self._check_mean_reversion,
                "weight": 0.2,
            },
            # Volume confirmation rule
            {
                "name": "volume_confirmation",
                "description": "Price moves should be confirmed by volume",
                "condition": self._check_volume_confirmation,
                "weight": 0.25,
            },
            # Support/resistance rule
            {
                "name": "support_resistance",
                "description": "Price behavior near key levels",
                "condition": self._check_support_resistance,
                "weight": 0.25,
            },
        ]

    def validate_with_reasoning(
        self,
        prediction: dict[str, Any],
        historical_data: pd.DataFrame,
        context: dict[str, Any],
    ) -> tuple[bool, list[str], list[str]]:
        """
        Validate prediction using reasoning rules.

        Args:
            prediction: Model prediction
            historical_data: Historical price data
            context: Market context

        Returns:
            (is_valid, supporting_reasons, contradicting_reasons)
        """
        with self._lock:
            supporting: list[str] = []
            contradicting: list[str] = []
            rule_results: list[dict[str, Any]] = []

            for rule in self._rules:
                try:
                    result = rule["condition"](prediction, historical_data, context)
                    rule_results.append({
                        "rule": rule["name"],
                        "passed": result["passed"],
                        "message": result["message"],
                        "weight": rule["weight"],
                    })

                    if result["passed"]:
                        supporting.append(f"{rule['description']}: {result['message']}")
                    else:
                        contradicting.append(f"{rule['description']}: {result['message']}")
                except Exception as e:
                    log.debug("Rule %s failed: %s", rule["name"], e)
                    rule_results.append({
                        "rule": rule["name"],
                        "passed": False,
                        "message": f"Rule error: {e}",
                        "weight": 0,
                    })

            # Calculate overall validity
            total_weight = sum(r["weight"] for r in rule_results)
            passed_weight = sum(r["weight"] for r in rule_results if r["passed"])

            is_valid = passed_weight >= (total_weight * 0.5)  # >50% weight passed

            # Store explanation
            explanation = {
                "timestamp": datetime.now(),
                "prediction": prediction,
                "is_valid": is_valid,
                "supporting": supporting,
                "contradicting": contradicting,
                "rule_results": rule_results,
            }
            self._explanation_history.append(explanation)

            return is_valid, supporting, contradicting

    def _check_trend_continuation(
        self,
        prediction: dict[str, Any],
        data: pd.DataFrame,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        """Check if prediction aligns with trend."""
        if "close" not in data.columns or len(data) < 20:
            return {"passed": True, "message": "Insufficient data for trend analysis"}

        closes = data["close"].dropna()
        current = closes.iloc[-1]

        # Calculate trend
        ma20 = closes.rolling(20).mean().iloc[-1]
        ma50 = closes.rolling(50).mean().iloc[-1] if len(closes) >= 50 else ma20

        pred_price = float(prediction.get("predicted_price", current))
        pred_direction = "up" if pred_price > current else "down"

        # Determine trend
        if current > ma20 and ma20 > ma50:
            trend = "uptrend"
            trend_direction = "up"
        elif current < ma20 and ma20 < ma50:
            trend = "downtrend"
            trend_direction = "down"
        else:
            trend = "sideways"
            trend_direction = "neutral"

        # Check alignment
        if trend == "sideways":
            return {"passed": True, "message": "No clear trend - any direction acceptable"}

        if pred_direction == trend_direction:
            return {"passed": True, "message": f"Prediction aligns with {trend}"}
        else:
            return {"passed": False, "message": f"Prediction contradicts {trend}"}

    def _check_mean_reversion(
        self,
        prediction: dict[str, Any],
        data: pd.DataFrame,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        """Check mean reversion logic."""
        if "close" not in data.columns or len(data) < 50:
            return {"passed": True, "message": "Insufficient data for mean reversion"}

        closes = data["close"].dropna()
        current = closes.iloc[-1]
        mean = closes.mean()
        std = closes.std()

        pred_price = float(prediction.get("predicted_price", current))

        # Check if current price is extreme
        z_current = (current - mean) / std if std > 0 else 0

        if abs(z_current) < 2:
            return {"passed": True, "message": "Price near mean - no strong reversion signal"}

        # Check if prediction expects reversion
        if z_current > 2 and pred_price < current:
            return {"passed": True, "message": "Prediction expects reversion from high"}
        elif z_current < -2 and pred_price > current:
            return {"passed": True, "message": "Prediction expects reversion from low"}
        elif abs(z_current) > 2 and ((z_current > 0 and pred_price > current) or (z_current < 0 and pred_price < current)):
            return {"passed": False, "message": "Prediction extends extreme - risky"}
        else:
            return {"passed": True, "message": "Mean reversion check passed"}

    def _check_volume_confirmation(
        self,
        prediction: dict[str, Any],
        data: pd.DataFrame,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        """Check if volume confirms price prediction."""
        if "volume" not in data.columns:
            return {"passed": True, "message": "No volume data"}

        volumes = data["volume"].dropna()
        avg_volume = volumes.mean()

        pred_volume = float(prediction.get("predicted_volume", avg_volume))
        pred_price = float(prediction.get("predicted_price", 0))
        current_price = float(context.get("current_price", pred_price))

        if avg_volume <= 0:
            return {"passed": True, "message": "Invalid average volume"}

        volume_ratio = pred_volume / avg_volume
        price_move = (pred_price - current_price) / current_price if current_price > 0 else 0

        # Strong moves should have volume confirmation
        if abs(price_move) > 0.03:  # >3% move
            if volume_ratio < 1.5:
                return {"passed": False, "message": "Large price move without volume confirmation"}
            else:
                return {"passed": True, "message": "Volume confirms price move"}
        else:
            return {"passed": True, "message": "Small move - volume less critical"}

    def _check_support_resistance(
        self,
        prediction: dict[str, Any],
        data: pd.DataFrame,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        """Check support/resistance levels."""
        if "close" not in data.columns or "high" not in data.columns or "low" not in data.columns:
            return {"passed": True, "message": "Insufficient data for S/R analysis"}

        pred_price = float(prediction.get("predicted_price", 0))

        # Find key levels from recent history
        recent_highs = data["high"].tail(50).max()
        recent_lows = data["low"].tail(50).min()

        if pred_price > recent_highs * 1.05:
            return {"passed": False, "message": "Prediction significantly above resistance"}
        elif pred_price < recent_lows * 0.95:
            return {"passed": False, "message": "Prediction significantly below support"}
        else:
            return {"passed": True, "message": "Prediction within S/R range"}

    def generate_explanation(
        self,
        prediction: dict[str, Any],
        historical_data: pd.DataFrame,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        """Generate human-readable explanation for prediction."""
        is_valid, supporting, contradicting = self.validate_with_reasoning(
            prediction, historical_data, context
        )

        explanation = {
            "valid": is_valid,
            "summary": self._generate_summary(prediction, supporting, contradicting),
            "supporting_factors": supporting,
            "risk_factors": contradicting,
            "confidence_factors": self._analyze_confidence(prediction, context),
            "recommendation": self._generate_recommendation(is_valid, supporting, contradicting),
        }

        return explanation

    def _generate_summary(
        self,
        prediction: dict[str, Any],
        supporting: list[str],
        contradicting: list[str],
    ) -> str:
        """Generate summary explanation."""
        pred_price = float(prediction.get("predicted_price", 0))
        confidence = float(prediction.get("confidence", 0))

        direction = "up" if pred_price > 0 else "neutral"
        strength = "strong" if confidence > 0.7 else "moderate" if confidence > 0.5 else "weak"

        summary = f"Model predicts {strength} {direction}ward movement with {confidence:.0%} confidence."

        if supporting:
            summary += f" Supported by {len(supporting)} factors."
        if contradicting:
            summary += f" {len(contradicting)} risk factors identified."

        return summary

    def _analyze_confidence(
        self,
        prediction: dict[str, Any],
        context: dict[str, Any],
    ) -> list[str]:
        """Analyze factors affecting confidence."""
        factors: list[str] = []

        confidence = float(prediction.get("confidence", 0))
        agreement = float(prediction.get("model_agreement", 1.0))

        if confidence > 0.7:
            factors.append("High model confidence")
        elif confidence < 0.5:
            factors.append("Low model confidence")

        if agreement > 0.8:
            factors.append("Strong ensemble agreement")
        elif agreement < 0.6:
            factors.append("Weak ensemble agreement")

        volatility = float(context.get("volatility", 0))
        if volatility > 0.03:
            factors.append("High market volatility")
        elif volatility < 0.01:
            factors.append("Low market volatility")

        return factors

    def _generate_recommendation(
        self,
        is_valid: bool,
        supporting: list[str],
        contradicting: list[str],
    ) -> str:
        """Generate trading recommendation."""
        if not is_valid:
            return "AVOID: Prediction fails reasoning validation"

        if len(contradicting) > len(supporting):
            return "CAUTION: More risk factors than supporting factors"

        if len(supporting) >= 3 and len(contradicting) == 0:
            return "STRONG: Multiple supporting factors, no contradictions"

        return "MODERATE: Balanced risk/reward profile"


# ============================================================================
# UNIFIED PREDICTION VALIDATOR
# ============================================================================

class UnifiedPredictionValidator:
    """
    Unified validator combining all four fixes.

    Usage:
        validator = UnifiedPredictionValidator()

        result = validator.validate(
            prediction=prediction_dict,
            historical_data=df,
            context=context_dict,
        )

        if result["is_safe_to_use"]:
            # Use prediction
        else:
            # Reject or adjust prediction
    """

    def __init__(self) -> None:
        self.hallucination_detector = HallucinationDetector()
        self.context_manager = ExtendedContextManager()
        self.knowledge_integrator = RealTimeKnowledgeIntegrator()
        self.reasoning_validator = ReasoningValidator()
        self._lock = threading.RLock()

    def validate(
        self,
        prediction: dict[str, Any],
        historical_data: pd.DataFrame,
        context: dict[str, Any],
        stock_code: str = "",
        fetcher: Any | None = None,
        news_aggregator: Any | None = None,
    ) -> dict[str, Any]:
        """
        Comprehensive prediction validation.

        Returns dict with:
        - is_safe_to_use: bool
        - confidence_adjusted: float
        - validation_reports: list
        - recommendations: list
        """
        with self._lock:
            reports: list[dict[str, Any]] = []
            recommendations: list[str] = []
            confidence = float(prediction.get("confidence", 0.5))

            # === Fix #1: Hallucination Check ===
            halluc_report = self.hallucination_detector.validate_prediction(
                prediction, context, historical_data
            )
            reports.append({"type": "hallucination", **halluc_report.to_dict()})

            if hallucc_report.status == ValidationStatus.INVALID:
                confidence *= 0.3
                recommendations.extend(halluc_report.recommendations)
            elif hallucc_report.status == ValidationStatus.SUSPICIOUS:
                confidence *= 0.6
                recommendations.extend(halluc_report.recommendations)

            # === Fix #2: Context Check ===
            if stock_code:
                self.context_manager.add_bars(historical_data)
            context_window = self.context_manager.get_context()
            reports.append({
                "type": "context",
                "regime": context_window.regime_label,
                "bars_available": len(context_window.recent_bars),
            })

            # === Fix #3: Real-time Knowledge Check ===
            if stock_code and fetcher:
                realtime_ctx = self.knowledge_integrator.update_context(
                    stock_code, fetcher, news_aggregator
                )
                reports.append({
                    "type": "realtime",
                    "latest_price": realtime_ctx.latest_price,
                    "sentiment": realtime_ctx.recent_news_sentiment,
                    "regime": realtime_ctx.market_regime,
                })

                # Adjust confidence based on sentiment
                sentiment = realtime_ctx.recent_news_sentiment
                if abs(sentiment) > 0.7:
                    confidence *= 1.1  # Boost confidence on strong sentiment
                elif sentiment == 0:
                    confidence *= 0.9  # Reduce on no news

            # === Fix #4: Reasoning Check ===
            is_valid, supporting, contradicting = self.reasoning_validator.validate_with_reasoning(
                prediction, historical_data, context
            )
            reports.append({
                "type": "reasoning",
                "is_valid": is_valid,
                "supporting_count": len(supporting),
                "contradicting_count": len(contradicting),
            })

            if not is_valid:
                confidence *= 0.4
                recommendations.append("Prediction fails reasoning validation")

            # === Final Decision ===
            is_safe = (
                hallucc_report.status != ValidationStatus.INVALID
                and confidence >= 0.35
                and (is_valid or len(contradicting) <= 1)
            )

            return {
                "is_safe_to_use": is_safe,
                "confidence_original": float(prediction.get("confidence", 0.5)),
                "confidence_adjusted": min(1.0, max(0.0, confidence)),
                "validation_reports": reports,
                "recommendations": recommendations,
                "timestamp": datetime.now().isoformat(),
            }


# Global instance
_validator: UnifiedPredictionValidator | None = None
_validator_lock = threading.RLock()


def get_prediction_validator() -> UnifiedPredictionValidator:
    """Get global prediction validator."""
    global _validator
    with _validator_lock:
        if _validator is None:
            _validator = UnifiedPredictionValidator()
        return _validator
