# trading/signals.py
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np

from config.settings import CONFIG
from utils.logger import get_logger

# Type-checking imports (no runtime cost)
if TYPE_CHECKING:
    import pandas as pd

    from models.predictor import Prediction

from models.predictor import Signal

log = get_logger(__name__)

class SignalConfidence(Enum):
    """Signal confidence level"""
    VERY_HIGH = "very_high"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    VERY_LOW = "very_low"

@dataclass
class TradingSignal:
    """
    Complete trading signal with all analysis components.

    Attributes:
        stock_code: Stock ticker symbol
        stock_name: Human-readable stock name
        current_price: Current market price
        signal: Final trading signal (BUY, SELL, HOLD, etc.)
        signal_strength: Signal strength from 0.0 to 1.0
        confidence: Overall confidence level
        ai_signal: Raw AI model signal
        ai_confidence: AI model confidence score
        ai_prob_up: AI probability of price increase
        ai_prob_down: AI probability of price decrease
        tech_signal: Technical analysis signal
        tech_score: Technical analysis score (-100 to +100)
        trend: Current market trend
        sentiment_score: News sentiment score (-100 to +100)
        sentiment_label: Sentiment classification
        news_count: Number of news articles analyzed
        combined_score: Weighted combined score (-100 to +100)
        entry_price: Suggested entry price
        stop_loss: Suggested stop loss price
        take_profit_1: First profit target
        take_profit_2: Second profit target
        position_size: Suggested position size in shares
        position_value: Total position value
        risk_amount: Amount at risk
        reasons: List of reasons supporting the signal
        warnings: List of warnings/risks
    """
    stock_code: str
    stock_name: str
    current_price: float

    signal: str  # Signal enum value as string
    signal_strength: float
    confidence: SignalConfidence

    ai_signal: str
    ai_confidence: float
    ai_prob_up: float
    ai_prob_down: float

    tech_signal: str
    tech_score: float
    trend: str

    sentiment_score: float
    sentiment_label: str
    news_count: int

    # Combined score (-100 to +100)
    combined_score: float

    entry_price: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    position_size: int
    position_value: float
    risk_amount: float

    # Reasons and warnings
    reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def __post_init__(self):
        """Validate fields after initialization."""
        if self.reasons is None:
            self.reasons = []
        if self.warnings is None:
            self.warnings = []

        self.signal_strength = float(np.clip(self.signal_strength, 0.0, 1.0))
        self.combined_score = float(np.clip(self.combined_score, -100.0, 100.0))
        self.tech_score = float(np.clip(self.tech_score, -100.0, 100.0))
        self.sentiment_score = float(np.clip(self.sentiment_score, -100.0, 100.0))

    @property
    def is_actionable(self) -> bool:
        """Check if signal is actionable (not HOLD with sufficient strength)."""
        return (
            self.signal not in (Signal.HOLD.value, "HOLD") 
            and self.signal_strength >= 0.5
            and self.confidence in (SignalConfidence.HIGH, SignalConfidence.VERY_HIGH)
        )

    @property
    def risk_reward_ratio(self) -> float:
        """Calculate risk/reward ratio for first target."""
        if self.entry_price <= 0 or self.stop_loss <= 0 or self.take_profit_1 <= 0:
            return 0.0

        risk = abs(self.entry_price - self.stop_loss)
        reward = abs(self.take_profit_1 - self.entry_price)

        if risk <= 0:
            return 0.0

        return reward / risk

class SignalGenerator:
    """
    Generates trading signals by combining:
    - AI predictions (50% weight)
    - Technical analysis (30% weight)
    - Sentiment analysis (20% weight)

    Uses weighted scoring to produce final signal with confidence levels.

    Example:
        generator = SignalGenerator()
        prediction = predictor.predict("600519")
        signal = generator.generate(prediction, df)

        if signal.is_actionable:
            print(f"Signal: {signal.signal} @ {signal.entry_price}")
    """

    # Component weights (must sum to 1.0)
    WEIGHT_AI: float = 0.50
    WEIGHT_TECHNICAL: float = 0.30
    WEIGHT_SENTIMENT: float = 0.20

    THRESHOLD_STRONG_BUY: float = 50.0
    THRESHOLD_BUY: float = 25.0
    THRESHOLD_SELL: float = -25.0
    THRESHOLD_STRONG_SELL: float = -50.0

    CONFIDENCE_VERY_HIGH: float = 0.80
    CONFIDENCE_HIGH: float = 0.65
    CONFIDENCE_MEDIUM: float = 0.50
    CONFIDENCE_LOW: float = 0.35

    # Minimum thresholds for warnings
    MIN_TECH_SCORE: float = 20.0
    MIN_MODEL_AGREEMENT: float = 0.60
    MIN_SENTIMENT_SIGNIFICANT: float = 0.30

    def __init__(self):
        """Initialize signal generator with lazy-loaded analyzers."""
        self.tech_analyzer = None
        self.sentiment_analyzer = None
        self.news_scraper = None
        self.strategy_engine = None
        self._init_analyzers()

    def _init_analyzers(self) -> None:
        """Initialize analyzers lazily with proper error handling."""
        try:
            from analysis.technical import TechnicalAnalyzer
            self.tech_analyzer = TechnicalAnalyzer()
            log.debug("Technical analyzer initialized")
        except ImportError as e:
            log.warning(f"Could not import technical analyzer: {e}")
        except Exception as e:
            log.warning(f"Could not init technical analyzer: {e}")

        try:
            from analysis.sentiment import NewsScraper, SentimentAnalyzer
            self.sentiment_analyzer = SentimentAnalyzer()
            self.news_scraper = NewsScraper()
            log.debug("Sentiment analyzer initialized")
        except ImportError as e:
            log.warning(f"Could not import sentiment analyzer: {e}")
        except Exception as e:
            log.warning(f"Could not init sentiment analyzer: {e}")

        try:
            from analysis.strategy_engine import StrategyScriptEngine
            self.strategy_engine = StrategyScriptEngine()
            log.debug("Strategy script engine initialized")
        except Exception as e:
            log.warning(f"Could not init strategy script engine: {e}")

    @staticmethod
    def _safe_get(obj: object, attr: str, default=None):
        """
        Safely get attribute from object with default value.

        Args:
            obj: Object to get attribute from
            attr: Attribute name
            default: Default value if attribute missing or None

        Returns:
            Attribute value or default
        """
        try:
            value = getattr(obj, attr, default)
            return value if value is not None else default
        except Exception as e:
            log.debug("Safe attribute access failed (%s): %s", attr, e)
            return default

    @staticmethod
    def _safe_float(value, default: float = 0.0) -> float:
        """Safely convert value to float."""
        try:
            if value is None:
                return default
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _safe_int(value, default: int = 0) -> int:
        """Safely convert value to int."""
        try:
            if value is None:
                return default
            return int(value)
        except (TypeError, ValueError):
            return default

    def generate(
        self, 
        prediction: Prediction, 
        df: pd.DataFrame | None = None, 
        include_sentiment: bool = True
    ) -> TradingSignal:
        """
        Generate comprehensive trading signal from prediction and market data.

        Args:
            prediction: AI model prediction result
            df: Optional DataFrame with OHLCV data for technical analysis
            include_sentiment: Whether to include sentiment analysis

        Returns:
            Complete TradingSignal with all analysis components

        Raises:
            ValueError: If prediction is None or invalid
        """
        if prediction is None:
            raise ValueError("Prediction cannot be None")

        reasons: list[str] = list(self._safe_get(prediction, "reasons", []) or [])
        warnings: list[str] = list(self._safe_get(prediction, "warnings", []) or [])

        stock_code = str(self._safe_get(prediction, "stock_code", "") or "")
        stock_name = str(self._safe_get(prediction, "stock_name", "") or "")
        current_price = self._safe_float(self._safe_get(prediction, "current_price"), 0.0)

        # === AI Score ===
        ai_score = self._calculate_ai_score(prediction)
        ai_conf = self._safe_float(self._safe_get(prediction, "confidence"), 0.0)
        model_agreement = self._safe_float(
            self._safe_get(prediction, "model_agreement", 
                          self._safe_get(prediction, "agreement", 1.0)),
            1.0
        )

        # === Technical Score ===
        tech_score, tech_signal, trend, indicators = self._analyze_technical(
            df, reasons, warnings
        )

        # === Sentiment Score ===
        sentiment_score, sentiment_label, news_count = self._analyze_sentiment(
            stock_code, include_sentiment, reasons
        )

        # === Combined Score ===
        combined_score = (
            ai_score * self.WEIGHT_AI +
            tech_score * self.WEIGHT_TECHNICAL +
            sentiment_score * self.WEIGHT_SENTIMENT
        )

        # Optional external strategy scripts can bias the score slightly.
        if self.strategy_engine is not None:
            try:
                script_bias, script_reasons = self.strategy_engine.evaluate(
                    df=df, indicators=indicators, symbol=stock_code
                )
                if script_bias:
                    combined_score += float(script_bias)
                reasons.extend(script_reasons[:3])
            except Exception as e:
                log.warning(f"Strategy script evaluation failed: {e}")

        combined_score = float(np.clip(combined_score, -100.0, 100.0))

        # === Final Signal ===
        final_signal, signal_strength = self._determine_signal(
            combined_score, ai_conf
        )

        # === Confidence Level ===
        confidence = self._determine_confidence(
            ai_conf, model_agreement, abs(combined_score) / 100.0
        )

        # === Generate Warnings ===
        self._generate_warnings(
            warnings, ai_conf, model_agreement, tech_score, 
            trend, final_signal
        )

        # === Extract Trading Levels ===
        levels = self._safe_get(prediction, "levels", None)
        position = self._safe_get(prediction, "position", None)

        # === Build AI Signal String ===
        pred_signal = self._safe_get(prediction, "signal", Signal.HOLD)
        if hasattr(pred_signal, "value"):
            ai_signal_str = pred_signal.value
        else:
            ai_signal_str = str(pred_signal) if pred_signal else "HOLD"

        return TradingSignal(
            stock_code=stock_code,
            stock_name=stock_name,
            current_price=current_price,

            signal=final_signal.value,
            signal_strength=signal_strength,
            confidence=confidence,

            ai_signal=ai_signal_str,
            ai_confidence=ai_conf,
            ai_prob_up=self._safe_float(self._safe_get(prediction, "prob_up"), 0.33),
            ai_prob_down=self._safe_float(self._safe_get(prediction, "prob_down"), 0.33),

            tech_signal=tech_signal,
            tech_score=tech_score,
            trend=trend,

            sentiment_score=sentiment_score,
            sentiment_label=sentiment_label,
            news_count=news_count,

            combined_score=combined_score,

            entry_price=self._safe_float(self._safe_get(levels, "entry"), 0.0),
            stop_loss=self._safe_float(self._safe_get(levels, "stop_loss"), 0.0),
            take_profit_1=self._safe_float(self._safe_get(levels, "target_1"), 0.0),
            take_profit_2=self._safe_float(self._safe_get(levels, "target_2"), 0.0),
            position_size=self._safe_int(self._safe_get(position, "shares"), 0),
            position_value=self._safe_float(self._safe_get(position, "value"), 0.0),
            risk_amount=self._safe_float(self._safe_get(position, "risk_amount"), 0.0),

            reasons=reasons,
            warnings=warnings,
        )

    def _calculate_ai_score(self, prediction) -> float:
        """
        Calculate AI component score from prediction probabilities.

        The score is weighted by confidence and model agreement.

        Args:
            prediction: AI prediction object

        Returns:
            AI score in range [-100, +100]
        """
        prob_up = self._safe_float(self._safe_get(prediction, "prob_up"), 0.33)
        prob_down = self._safe_float(self._safe_get(prediction, "prob_down"), 0.33)
        conf = self._safe_float(self._safe_get(prediction, "confidence"), 0.0)

        agreement = self._safe_get(prediction, "model_agreement", None)
        if agreement is None:
            agreement = self._safe_get(prediction, "agreement", 1.0)
        agreement = self._safe_float(agreement, 1.0)

        score = (prob_up - prob_down) * 100.0

        # Weight by confidence (0.5 to 1.0 range)
        score *= (0.5 + 0.5 * conf)

        # Weight by model agreement (0.5 to 1.0 range)
        score *= (0.5 + 0.5 * agreement)

        return float(np.clip(score, -100.0, 100.0))

    def _analyze_technical(
        self, 
        df: pd.DataFrame | None, 
        reasons: list[str], 
        warnings: list[str]
    ) -> tuple[float, str, str, dict[str, float]]:
        """
        Perform technical analysis on price data.

        Args:
            df: DataFrame with OHLCV data
            reasons: List to append reasons to
            warnings: List to append warnings to

        Returns:
            Tuple of (score, signal, trend)
        """
        tech_score = 0.0
        tech_signal = "neutral"
        trend = "sideways"

        indicators: dict[str, float] = {}
        if df is None or len(df) < 60 or self.tech_analyzer is None:
            return tech_score, tech_signal, trend, indicators

        try:
            tech_summary = self.tech_analyzer.analyze(df)

            tech_score = self._safe_float(
                self._safe_get(tech_summary, "overall_score"), 0.0
            )

            raw_signal = self._safe_get(tech_summary, "overall_signal", "neutral")
            tech_signal = str(raw_signal) if raw_signal else "neutral"
            indicators = dict(self._safe_get(tech_summary, "indicators", {}) or {})

            # Extract trend (handle enum)
            raw_trend = self._safe_get(tech_summary, "trend", None)
            if raw_trend is not None:
                if hasattr(raw_trend, "value"):
                    trend = raw_trend.value
                else:
                    trend = str(raw_trend)

            signals = self._safe_get(tech_summary, "signals", []) or []
            for sig in signals[:3]:
                try:
                    strength = self._safe_get(sig, "strength", None)
                    if strength is not None:
                        strength_val = strength.value if hasattr(strength, "value") else 0
                        if strength_val >= 2:
                            description = self._safe_get(sig, "description", "")
                            if description:
                                reasons.append(f"Technical: {description}")
                except Exception as e:
                    log.debug("Technical signal detail parse skipped: %s", e)

        except Exception as e:
            log.warning(f"Technical analysis failed: {e}")
            warnings.append("Technical analysis unavailable")

        return tech_score, tech_signal, trend, indicators

    def _analyze_sentiment(
        self,
        stock_code: str,
        include_sentiment: bool,
        reasons: list[str]
    ) -> tuple[float, str, int]:
        """
        Analyze stock sentiment using both legacy and institutional sources.

        Blending strategy:
        - Legacy scraper score (if available)
        - Weighted institutional sentiment from the aggregated news engine
        """
        sentiment_score = 0.0
        sentiment_label = "neutral"
        news_count = 0

        if not include_sentiment:
            return sentiment_score, sentiment_label, news_count

        components: list[tuple[float, float]] = []
        news_count_candidates: list[int] = []

        # Legacy source score
        if self.news_scraper is not None:
            try:
                sent_score, sent_conf = self.news_scraper.get_stock_sentiment(stock_code)
                market_sent = self.news_scraper.get_market_sentiment()
                sent_score = float(np.clip(float(sent_score), -1.0, 1.0))
                sent_conf = float(np.clip(float(sent_conf), 0.0, 1.0))
                legacy_weight = 0.45 + (0.65 * sent_conf)
                components.append((sent_score, legacy_weight))
                news_count_candidates.append(
                    self._safe_int(market_sent.get("news_count", 0), 0)
                )
            except Exception as e:
                log.debug(f"Legacy sentiment unavailable for {stock_code}: {e}")

        # Institutional weighted score
        try:
            from data.news import get_news_aggregator

            agg = get_news_aggregator()
            summary = agg.get_sentiment_summary(stock_code)
            if isinstance(summary, dict):
                inst_score = self._safe_float(summary.get("overall_sentiment"), 0.0)
                inst_score = float(np.clip(inst_score, -1.0, 1.0))
                inst_conf = float(
                    np.clip(self._safe_float(summary.get("confidence"), 0.0), 0.0, 1.0)
                )
                source_div = float(
                    np.clip(
                        self._safe_float(summary.get("source_diversity"), 0.0),
                        0.0,
                        1.0,
                    )
                )
                total_news = self._safe_int(summary.get("total", 0), 0)
                inst_weight = 0.55 + (0.45 * inst_conf) + (0.20 * source_div)
                components.append((inst_score, inst_weight))
                news_count_candidates.append(total_news)

                if total_news > 0:
                    reasons.append(
                        "News flow: "
                        f"{str(summary.get('label', 'neutral')).lower()} "
                        f"({inst_score:+.2f}, confidence {inst_conf:.0%})"
                    )
        except Exception as e:
            log.debug(f"Institutional sentiment unavailable for {stock_code}: {e}")

        if components:
            weighted_sum = sum(score * weight for score, weight in components)
            weight_sum = sum(weight for _, weight in components)
            blended = weighted_sum / max(weight_sum, 1e-9)
            sentiment_score = float(np.clip(blended * 100.0, -100.0, 100.0))
        else:
            sentiment_score = 0.0

        news_count = max(news_count_candidates) if news_count_candidates else 0

        if sentiment_score > 15.0:
            sentiment_label = "positive"
        elif sentiment_score < -15.0:
            sentiment_label = "negative"
        else:
            sentiment_label = "neutral"

        if abs(sentiment_score) >= (self.MIN_SENTIMENT_SIGNIFICANT * 100.0):
            reasons.append(f"Sentiment edge: {sentiment_score:+.1f}")

        return sentiment_score, sentiment_label, news_count

    def _determine_signal(
        self, 
        combined_score: float, 
        ai_confidence: float
    ) -> tuple[Signal, float]:
        """
        Determine final trading signal from combined score.

        Args:
            combined_score: Weighted combined score (-100 to +100)
            ai_confidence: AI model confidence (0 to 1)

        Returns:
            Tuple of (Signal enum, signal_strength)
        """
        # Calculate signal strength (0 to 1)
        strength = min(abs(combined_score) / 80.0, 1.0)

        if combined_score >= self.THRESHOLD_STRONG_BUY:
            signal = Signal.STRONG_BUY
        elif combined_score >= self.THRESHOLD_BUY:
            signal = Signal.BUY
        elif combined_score <= self.THRESHOLD_STRONG_SELL:
            signal = Signal.STRONG_SELL
        elif combined_score <= self.THRESHOLD_SELL:
            signal = Signal.SELL
        else:
            signal = Signal.HOLD

        min_confidence = self._safe_float(CONFIG.MIN_CONFIDENCE, 0.55)
        if ai_confidence < min_confidence:
            if signal in (Signal.BUY, Signal.SELL):
                signal = Signal.HOLD
                strength *= 0.5

        return signal, float(strength)

    def _determine_confidence(
        self,
        ai_confidence: float,
        model_agreement: float,
        score_strength: float
    ) -> SignalConfidence:
        """
        Determine overall signal confidence level.

        Args:
            ai_confidence: AI model confidence (0 to 1)
            model_agreement: Agreement between ensemble models (0 to 1)
            score_strength: Normalized score strength (0 to 1)

        Returns:
            SignalConfidence enum
        """
        # Combined confidence score (average of all components)
        conf_score = (ai_confidence + model_agreement + score_strength) / 3.0

        if conf_score >= self.CONFIDENCE_VERY_HIGH:
            return SignalConfidence.VERY_HIGH
        elif conf_score >= self.CONFIDENCE_HIGH:
            return SignalConfidence.HIGH
        elif conf_score >= self.CONFIDENCE_MEDIUM:
            return SignalConfidence.MEDIUM
        elif conf_score >= self.CONFIDENCE_LOW:
            return SignalConfidence.LOW
        else:
            return SignalConfidence.VERY_LOW

    def _generate_warnings(
        self,
        warnings: list[str],
        ai_conf: float,
        model_agreement: float,
        tech_score: float,
        trend: str,
        signal: Signal
    ) -> None:
        """
        Generate warning messages based on analysis.

        Args:
            warnings: List to append warnings to (modified in place)
            ai_conf: AI confidence score
            model_agreement: Model agreement score
            tech_score: Technical analysis score
            trend: Current trend string
            signal: Final signal
        """
        min_confidence = self._safe_float(CONFIG.MIN_CONFIDENCE, 0.55)

        if ai_conf < min_confidence:
            warnings.append(f"Low AI model confidence ({ai_conf:.0%})")

        if model_agreement < self.MIN_MODEL_AGREEMENT:
            warnings.append(f"AI models disagree ({model_agreement:.0%} agreement)")

        if abs(tech_score) < self.MIN_TECH_SCORE:
            warnings.append("Weak technical signals")

        trend_lower = trend.lower() if trend else ""

        if "strong_down" in trend_lower or trend_lower == "strong_downtrend":
            if signal in (Signal.BUY, Signal.STRONG_BUY):
                warnings.append("Buying against strong downtrend")

        if "strong_up" in trend_lower or trend_lower == "strong_uptrend":
            if signal in (Signal.SELL, Signal.STRONG_SELL):
                warnings.append("Selling in strong uptrend")

    def scan_stocks(
        self,
        predictions: list[Prediction],
        min_signal_strength: float = 0.5,
        signal_type: str = "all",
        min_ai_confidence: float | None = None,
        max_warnings: int | None = None,
    ) -> list[TradingSignal]:
        """
        Scan multiple stocks and filter by signal quality criteria.

        Args:
            predictions: List of AI predictions
            min_signal_strength: Minimum signal strength (0 to 1)
            signal_type: Filter type - "all", "buy", or "sell"
            min_ai_confidence: Optional confidence floor override
            max_warnings: Optional warning-count cap

        Returns:
            Ranked list of TradingSignals matching criteria
        """
        signals: list[TradingSignal] = []

        conf_floor = (
            self._safe_float(min_ai_confidence, self._safe_float(CONFIG.MIN_CONFIDENCE, 0.55))
            if min_ai_confidence is not None
            else self._safe_float(CONFIG.MIN_CONFIDENCE, 0.55)
        )
        warning_cap = int(max_warnings) if max_warnings is not None else None

        for pred in predictions:
            try:
                signal = self.generate(pred, include_sentiment=False)

                if signal.signal_strength < min_signal_strength:
                    continue
                if float(signal.ai_confidence) < float(conf_floor):
                    continue
                if warning_cap is not None and len(signal.warnings) > warning_cap:
                    continue

                if signal_type == "buy":
                    if signal.signal not in (Signal.BUY.value, Signal.STRONG_BUY.value):
                        continue
                elif signal_type == "sell":
                    if signal.signal not in (Signal.SELL.value, Signal.STRONG_SELL.value):
                        continue

                signals.append(signal)

            except Exception as e:
                stock_code = self._safe_get(pred, "stock_code", "unknown")
                log.warning(f"Signal generation failed for {stock_code}: {e}")

        def _rank(s: TradingSignal) -> float:
            rr = float(max(0.0, min(4.0, s.risk_reward_ratio)))
            warning_penalty = float(min(5, len(s.warnings))) * 4.0
            return float(
                (abs(s.combined_score) * 1.15)
                + (float(s.signal_strength) * 55.0)
                + (float(s.ai_confidence) * 45.0)
                + (rr * 6.0)
                - warning_penalty
            )

        signals.sort(key=_rank, reverse=True)

        return signals

    def get_top_opportunities(
        self,
        predictions: list[Prediction],
        n: int = 5
    ) -> dict[str, list[TradingSignal]]:
        """
        Get top N buy and sell opportunities.

        Args:
            predictions: List of AI predictions
            n: Number of top picks per category

        Returns:
            Dict with 'buy' and 'sell' lists of TradingSignals
        """
        all_signals = self.scan_stocks(predictions)

        buy_signals = [
            s for s in all_signals 
            if s.signal in (Signal.BUY.value, Signal.STRONG_BUY.value)
        ]
        sell_signals = [
            s for s in all_signals 
            if s.signal in (Signal.SELL.value, Signal.STRONG_SELL.value)
        ]

        return {
            'buy': buy_signals[:n],
            'sell': sell_signals[:n]
        }

    def get_signal_summary(self, signal: TradingSignal) -> str:
        """
        Generate human-readable summary of trading signal.

        Args:
            signal: TradingSignal to summarize

        Returns:
            Formatted summary string
        """
        lines = [
            f"{'='*50}",
            f"{signal.stock_name} ({signal.stock_code})",
            f"{'='*50}",
            "",
            f"Signal: {signal.signal} (Strength: {signal.signal_strength:.0%})",
            f"Confidence: {signal.confidence.value}",
            f"Price: CNY {signal.current_price:.2f}",
            "",
            "--- Scores ---",
            f"Combined: {signal.combined_score:+.1f}",
            f"AI: {signal.ai_confidence:.0%} confidence",
            f"Technical: {signal.tech_score:+.1f} ({signal.tech_signal})",
            f"Sentiment: {signal.sentiment_score:+.1f} ({signal.sentiment_label})",
            f"Trend: {signal.trend}",
            "",
        ]

        if signal.entry_price > 0:
            lines.extend([
                "--- Trading Plan ---",
                f"Entry: CNY {signal.entry_price:.2f}",
                f"Stop Loss: CNY {signal.stop_loss:.2f}",
                f"Target 1: CNY {signal.take_profit_1:.2f}",
                f"Target 2: CNY {signal.take_profit_2:.2f}",
                f"Position: {signal.position_size} shares (CNY {signal.position_value:,.0f})",
                f"Risk: CNY {signal.risk_amount:,.0f}",
                f"R/R Ratio: {signal.risk_reward_ratio:.2f}",
                "",
            ])

        if signal.reasons:
            lines.append("--- Reasons ---")
            for reason in signal.reasons:
                lines.append(f"  - {reason}")
            lines.append("")

        if signal.warnings:
            lines.append("--- Warnings ---")
            for warning in signal.warnings:
                lines.append(f"  - {warning}")
            lines.append("")

        return "\n".join(lines)
