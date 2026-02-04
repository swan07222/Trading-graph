"""
Signal Generation - Combine AI predictions with technical analysis
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from enum import Enum
import numpy as np

from config import CONFIG
from models.predictor import Prediction, Signal
from analysis.technical import TechnicalAnalyzer, TechnicalSummary, TrendDirection
from analysis.sentiment import SentimentAnalyzer, NewsScraper
from utils.logger import log


class SignalConfidence(Enum):
    """Signal confidence level"""
    VERY_HIGH = "very_high"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    VERY_LOW = "very_low"


@dataclass
class TradingSignal:
    """Complete trading signal with all analysis"""
    # Stock info
    stock_code: str
    stock_name: str
    current_price: float
    
    # Signal
    signal: Signal
    signal_strength: float
    confidence: SignalConfidence
    
    # AI prediction
    ai_signal: Signal
    ai_confidence: float
    ai_prob_up: float
    ai_prob_down: float
    
    # Technical analysis
    tech_signal: str
    tech_score: float
    trend: TrendDirection
    
    # Sentiment
    sentiment_score: float
    sentiment_label: str
    news_count: int
    
    # Combined score (-100 to +100)
    combined_score: float
    
    # Trading plan
    entry_price: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    position_size: int
    position_value: float
    risk_amount: float
    
    # Reasons and warnings
    reasons: List[str]
    warnings: List[str]


class SignalGenerator:
    """
    Generates trading signals by combining:
    - AI predictions
    - Technical analysis
    - Sentiment analysis
    
    Uses weighted scoring to produce final signal
    """
    
    # Component weights
    WEIGHTS = {
        'ai': 0.50,         # AI model prediction
        'technical': 0.30,  # Technical analysis
        'sentiment': 0.20,  # News sentiment
    }
    
    def __init__(self):
        self.tech_analyzer = TechnicalAnalyzer()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.news_scraper = NewsScraper()
    
    def generate(self, 
                 prediction: Prediction,
                 df=None,
                 include_sentiment: bool = True) -> TradingSignal:
        """
        Generate comprehensive trading signal
        
        Args:
            prediction: AI prediction result
            df: Optional DataFrame for technical analysis
            include_sentiment: Whether to include sentiment analysis
            
        Returns:
            TradingSignal with complete analysis
        """
        reasons = prediction.reasons.copy()
        warnings = prediction.warnings.copy()
        
        # === AI Score ===
        ai_score = self._calculate_ai_score(prediction)
        
        # === Technical Score ===
        if df is not None and len(df) >= 60:
            try:
                tech_summary = self.tech_analyzer.analyze(df)
                tech_score = tech_summary.overall_score
                tech_signal = tech_summary.overall_signal
                trend = tech_summary.trend
                
                # Add technical reasons
                for sig in tech_summary.signals[:3]:
                    if sig.strength.value >= 2:
                        reasons.append(f"📊 {sig.description}")
                
            except Exception as e:
                log.warning(f"Technical analysis failed: {e}")
                tech_score = 0
                tech_signal = "neutral"
                trend = TrendDirection.SIDEWAYS
        else:
            tech_score = 0
            tech_signal = "neutral"
            trend = TrendDirection.SIDEWAYS
        
        # === Sentiment Score ===
        if include_sentiment:
            try:
                sent_score, sent_conf = self.news_scraper.get_stock_sentiment(
                    prediction.stock_code
                )
                market_sent = self.news_scraper.get_market_sentiment()
                
                sentiment_score = sent_score * 100  # Convert to -100 to +100
                sentiment_label = market_sent.get('label', 'neutral')
                news_count = market_sent.get('news_count', 0)
                
                if abs(sent_score) > 0.3:
                    direction = "positive" if sent_score > 0 else "negative"
                    reasons.append(f"📰 News sentiment: {direction} ({sent_score:+.2f})")
                
            except Exception as e:
                log.warning(f"Sentiment analysis failed: {e}")
                sentiment_score = 0
                sentiment_label = "neutral"
                news_count = 0
        else:
            sentiment_score = 0
            sentiment_label = "neutral"
            news_count = 0
        
        # === Combined Score ===
        combined_score = (
            ai_score * self.WEIGHTS['ai'] +
            tech_score * self.WEIGHTS['technical'] +
            sentiment_score * self.WEIGHTS['sentiment']
        )
        
        # === Determine Final Signal ===
        final_signal, signal_strength = self._determine_signal(combined_score, prediction)
        
        # === Determine Confidence ===
        confidence = self._determine_confidence(
            prediction.confidence,
            prediction.model_agreement,
            abs(combined_score) / 100
        )
        
        # === Add Warnings ===
        if prediction.confidence < CONFIG.MIN_CONFIDENCE:
            warnings.append("Low AI model confidence")
        
        if prediction.model_agreement < 0.6:
            warnings.append("AI models disagree")
        
        if abs(tech_score) < 20:
            warnings.append("Weak technical signals")
        
        if trend == TrendDirection.STRONG_DOWN and final_signal in [Signal.BUY, Signal.STRONG_BUY]:
            warnings.append("Buying against strong downtrend")
        
        if trend == TrendDirection.STRONG_UP and final_signal in [Signal.SELL, Signal.STRONG_SELL]:
            warnings.append("Selling in strong uptrend")
        
        return TradingSignal(
            stock_code=prediction.stock_code,
            stock_name=prediction.stock_name,
            current_price=prediction.current_price,
            signal=final_signal,
            signal_strength=signal_strength,
            confidence=confidence,
            ai_signal=prediction.signal,
            ai_confidence=prediction.confidence,
            ai_prob_up=prediction.prob_up,
            ai_prob_down=prediction.prob_down,
            tech_signal=tech_signal,
            tech_score=tech_score,
            trend=trend,
            sentiment_score=sentiment_score,
            sentiment_label=sentiment_label,
            news_count=news_count,
            combined_score=combined_score,
            entry_price=prediction.levels.entry,
            stop_loss=prediction.levels.stop_loss,
            take_profit_1=prediction.levels.target_1,
            take_profit_2=prediction.levels.target_2,
            position_size=prediction.position.shares,
            position_value=prediction.position.value,
            risk_amount=prediction.position.risk_amount,
            reasons=reasons,
            warnings=warnings
        )
    
    def _calculate_ai_score(self, prediction: Prediction) -> float:
        """Calculate AI component score"""
        # Base score from probabilities
        score = (prediction.prob_up - prediction.prob_down) * 100
        
        # Adjust by confidence
        score *= (0.5 + 0.5 * prediction.confidence)
        
        # Adjust by model agreement
        score *= (0.5 + 0.5 * prediction.model_agreement)
        
        return score
    
    def _determine_signal(self, 
                          combined_score: float,
                          prediction: Prediction) -> Tuple[Signal, float]:
        """Determine final signal from combined score"""
        # Signal strength (0-1)
        strength = min(abs(combined_score) / 80, 1.0)
        
        # Determine signal
        if combined_score >= 50:
            signal = Signal.STRONG_BUY
        elif combined_score >= 25:
            signal = Signal.BUY
        elif combined_score <= -50:
            signal = Signal.STRONG_SELL
        elif combined_score <= -25:
            signal = Signal.SELL
        else:
            signal = Signal.HOLD
        
        # Override with HOLD if confidence too low
        if prediction.confidence < CONFIG.MIN_CONFIDENCE:
            if signal in [Signal.BUY, Signal.SELL]:
                signal = Signal.HOLD
                strength *= 0.5
        
        return signal, strength
    
    def _determine_confidence(self,
                              ai_confidence: float,
                              model_agreement: float,
                              score_strength: float) -> SignalConfidence:
        """Determine signal confidence level"""
        # Combined confidence score
        conf_score = (ai_confidence + model_agreement + score_strength) / 3
        
        if conf_score >= 0.8:
            return SignalConfidence.VERY_HIGH
        elif conf_score >= 0.65:
            return SignalConfidence.HIGH
        elif conf_score >= 0.5:
            return SignalConfidence.MEDIUM
        elif conf_score >= 0.35:
            return SignalConfidence.LOW
        else:
            return SignalConfidence.VERY_LOW
    
    def scan_stocks(self, 
                    predictions: List[Prediction],
                    min_signal_strength: float = 0.5,
                    signal_type: str = "all") -> List[TradingSignal]:
        """
        Scan multiple stocks and filter by signal criteria
        
        Args:
            predictions: List of AI predictions
            min_signal_strength: Minimum signal strength to include
            signal_type: "buy", "sell", or "all"
            
        Returns:
            Filtered and sorted list of trading signals
        """
        signals = []
        
        for pred in predictions:
            try:
                signal = self.generate(pred, include_sentiment=False)
                
                # Filter by strength
                if signal.signal_strength < min_signal_strength:
                    continue
                
                # Filter by type
                if signal_type == "buy":
                    if signal.signal not in [Signal.BUY, Signal.STRONG_BUY]:
                        continue
                elif signal_type == "sell":
                    if signal.signal not in [Signal.SELL, Signal.STRONG_SELL]:
                        continue
                
                signals.append(signal)
                
            except Exception as e:
                log.warning(f"Signal generation failed for {pred.stock_code}: {e}")
        
        # Sort by combined score (absolute value)
        signals.sort(key=lambda s: (
            s.signal in [Signal.STRONG_BUY, Signal.STRONG_SELL],
            abs(s.combined_score),
            s.signal_strength
        ), reverse=True)
        
        return signals
    
    def get_top_opportunities(self,
                              predictions: List[Prediction],
                              n: int = 5) -> Dict[str, List[TradingSignal]]:
        """
        Get top buy and sell opportunities
        
        Returns:
            Dict with 'buy' and 'sell' lists
        """
        all_signals = self.scan_stocks(predictions)
        
        buy_signals = [s for s in all_signals if s.signal in [Signal.BUY, Signal.STRONG_BUY]]
        sell_signals = [s for s in all_signals if s.signal in [Signal.SELL, Signal.STRONG_SELL]]
        
        return {
            'buy': buy_signals[:n],
            'sell': sell_signals[:n]
        }