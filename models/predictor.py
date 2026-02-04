"""
Stock Predictor - Generate trading signals and predictions
"""
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path

import torch

from config import CONFIG
from data.fetcher import DataFetcher
from data.features import FeatureEngine
from models.ensemble import EnsembleModel, EnsemblePrediction
from utils.logger import log


class Signal(Enum):
    """Trading signal types"""
    STRONG_BUY = "STRONG BUY"
    BUY = "BUY"
    HOLD = "HOLD"
    SELL = "SELL"
    STRONG_SELL = "STRONG SELL"


@dataclass
class TradeLevels:
    """Trading price levels"""
    entry: float
    stop_loss: float
    stop_loss_pct: float
    target_1: float
    target_1_pct: float
    target_2: float
    target_2_pct: float
    target_3: float
    target_3_pct: float
    risk_reward: float


@dataclass
class PositionSize:
    """Position sizing recommendation"""
    shares: int
    value: float
    portfolio_pct: float
    risk_amount: float


@dataclass
class Prediction:
    """Complete prediction result"""
    # Stock info
    stock_code: str
    stock_name: str
    current_price: float
    timestamp: datetime
    
    # AI predictions
    prob_up: float
    prob_neutral: float
    prob_down: float
    predicted_class: int
    confidence: float
    model_agreement: float
    
    # Signal
    signal: Signal
    signal_strength: float
    
    # Trading levels
    levels: TradeLevels
    
    # Position sizing
    position: PositionSize
    
    # Technical indicators
    rsi: float = 50.0
    macd_signal: str = "neutral"
    trend: str = "sideways"
    
    # Analysis
    reasons: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    # Chart data
    price_history: List[float] = field(default_factory=list)
    predicted_prices: List[float] = field(default_factory=list)


class Predictor:
    """
    Stock Predictor - Generates trading signals and recommendations
    
    Features:
    - AI-based price direction prediction
    - Signal strength calculation
    - Risk-adjusted position sizing
    - Price target calculation
    - Future price visualization
    """
    
    def __init__(self, capital: float = None):
        self.capital = capital or CONFIG.CAPITAL
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.fetcher = DataFetcher()
        self.feature_engine = FeatureEngine()
        
        self.ensemble: Optional[EnsembleModel] = None
        self._load_model()
    
    def _load_model(self):
        """Load the trained ensemble model"""
        model_path = CONFIG.MODEL_DIR / "ensemble.pt"
        
        if not model_path.exists():
            log.warning("No trained model found. Train a model first.")
            return
        
        try:
            # Load to get input size
            state = torch.load(model_path, map_location='cpu')
            input_size = state.get('input_size', len(FeatureEngine.FEATURE_NAMES))
            
            self.ensemble = EnsembleModel(input_size)
            if self.ensemble.load(str(model_path)):
                log.info("Prediction model loaded successfully")
            else:
                self.ensemble = None
        except Exception as e:
            log.error(f"Failed to load model: {e}")
            self.ensemble = None
    
    def predict(self, stock_code: str) -> Prediction:
        """
        Generate complete prediction for a stock
        
        Args:
            stock_code: Stock code (e.g., "600519")
            
        Returns:
            Prediction object with all analysis
        """
        if self.ensemble is None:
            raise RuntimeError("No model loaded. Train a model first.")
        
        # Get data
        df = self.fetcher.get_history(stock_code)
        
        if len(df) < CONFIG.SEQUENCE_LENGTH + 10:
            raise ValueError(f"Insufficient data for {stock_code}: {len(df)} bars")
        
        # Get real-time quote
        quote = self.fetcher.get_realtime(stock_code)
        current_price = quote.price if quote else float(df['close'].iloc[-1])
        stock_name = quote.name if quote else stock_code
        
        # Create features
        df = self.feature_engine.create_features(df)
        
        # Prepare sequence for prediction
        feature_cols = self.feature_engine.get_feature_columns()
        features = df[feature_cols].values
        
        # Normalize (use simple standardization for prediction)
        mean = np.mean(features, axis=0)
        std = np.std(features, axis=0) + 1e-8
        features = (features - mean) / std
        features = np.clip(features, -5, 5)
        
        # Get last sequence
        X = features[-CONFIG.SEQUENCE_LENGTH:]
        
        # Get ensemble prediction
        ensemble_pred = self.ensemble.predict(X)
        
        # Extract technical indicators for display
        rsi = (df['rsi_14'].iloc[-1] + 0.5) * 100 if 'rsi_14' in df.columns else 50
        macd = df['macd_hist'].iloc[-1] if 'macd_hist' in df.columns else 0
        ma_ratio = df['ma_ratio_5_20'].iloc[-1] if 'ma_ratio_5_20' in df.columns else 0
        
        # Determine signal
        signal, strength, reasons = self._calculate_signal(
            ensemble_pred, rsi, macd, ma_ratio
        )
        
        # Calculate trading levels
        atr_pct = df['atr_pct'].iloc[-1] if 'atr_pct' in df.columns else 2.0
        atr = current_price * atr_pct / 100
        levels = self._calculate_levels(current_price, atr, signal)
        
        # Calculate position size
        position = self._calculate_position(
            signal, strength, current_price, levels, ensemble_pred.confidence
        )
        
        # Generate price predictions for chart
        price_history = df['close'].tail(60).tolist()
        predicted_prices = self._generate_price_forecast(
            current_price, ensemble_pred, atr_pct
        )
        
        # Generate warnings
        warnings = self._generate_warnings(ensemble_pred, rsi)
        
        return Prediction(
            stock_code=stock_code,
            stock_name=stock_name,
            current_price=current_price,
            timestamp=datetime.now(),
            prob_up=ensemble_pred.prob_up,
            prob_neutral=ensemble_pred.prob_neutral,
            prob_down=ensemble_pred.prob_down,
            predicted_class=ensemble_pred.predicted_class,
            confidence=ensemble_pred.confidence,
            model_agreement=ensemble_pred.agreement,
            signal=signal,
            signal_strength=strength,
            levels=levels,
            position=position,
            rsi=rsi,
            macd_signal="bullish" if macd > 0 else "bearish",
            trend="up" if ma_ratio > 2 else ("down" if ma_ratio < -2 else "sideways"),
            reasons=reasons,
            warnings=warnings,
            price_history=price_history,
            predicted_prices=predicted_prices
        )
    
    def _calculate_signal(
        self,
        pred: EnsemblePrediction,
        rsi: float,
        macd: float,
        ma_ratio: float
    ) -> Tuple[Signal, float, List[str]]:
        """Calculate trading signal with scoring system"""
        reasons = []
        score = 0
        
        # AI Prediction (50% weight)
        if pred.prob_up >= CONFIG.STRONG_BUY_THRESHOLD:
            score += 50
            reasons.append(f"🤖 AI: Strong UP signal ({pred.prob_up:.0%})")
        elif pred.prob_up >= CONFIG.BUY_THRESHOLD:
            score += 30
            reasons.append(f"🤖 AI: UP signal ({pred.prob_up:.0%})")
        elif pred.prob_down >= CONFIG.STRONG_SELL_THRESHOLD:
            score -= 50
            reasons.append(f"🤖 AI: Strong DOWN signal ({pred.prob_down:.0%})")
        elif pred.prob_down >= CONFIG.SELL_THRESHOLD:
            score -= 30
            reasons.append(f"🤖 AI: DOWN signal ({pred.prob_down:.0%})")
        else:
            reasons.append(f"🤖 AI: Neutral (UP:{pred.prob_up:.0%}, DOWN:{pred.prob_down:.0%})")
        
        # Confidence adjustment
        if pred.confidence < CONFIG.MIN_CONFIDENCE:
            score *= 0.5
            reasons.append(f"⚠️ Low confidence ({pred.confidence:.0%})")
        elif pred.confidence > 0.7:
            score *= 1.1
            reasons.append(f"✅ High confidence ({pred.confidence:.0%})")
        
        # Model agreement
        if pred.agreement < 0.6:
            score *= 0.7
            reasons.append(f"⚠️ Low model agreement ({pred.agreement:.0%})")
        elif pred.agreement > 0.8:
            score *= 1.1
            reasons.append(f"✅ High model agreement ({pred.agreement:.0%})")
        
        # RSI (20% weight)
        if rsi < 30:
            score += 15
            reasons.append(f"📈 RSI oversold ({rsi:.0f})")
        elif rsi < 40:
            score += 8
            reasons.append(f"📈 RSI low ({rsi:.0f})")
        elif rsi > 70:
            score -= 15
            reasons.append(f"📉 RSI overbought ({rsi:.0f})")
        elif rsi > 60:
            score -= 8
            reasons.append(f"📉 RSI high ({rsi:.0f})")
        
        # MACD (15% weight)
        if macd > 0.5:
            score += 12
            reasons.append("📈 MACD bullish")
        elif macd < -0.5:
            score -= 12
            reasons.append("📉 MACD bearish")
        
        # Trend (15% weight)
        if ma_ratio > 3:
            score += 12
            reasons.append(f"📈 Strong uptrend (+{ma_ratio:.1f}%)")
        elif ma_ratio > 1:
            score += 6
            reasons.append(f"📈 Uptrend (+{ma_ratio:.1f}%)")
        elif ma_ratio < -3:
            score -= 12
            reasons.append(f"📉 Strong downtrend ({ma_ratio:.1f}%)")
        elif ma_ratio < -1:
            score -= 6
            reasons.append(f"📉 Downtrend ({ma_ratio:.1f}%)")
        
        # Calculate signal strength (0-1)
        strength = min(abs(score) / 80, 1.0)
        
        # Determine signal
        if score >= 50:
            signal = Signal.STRONG_BUY
        elif score >= 25:
            signal = Signal.BUY
        elif score <= -50:
            signal = Signal.STRONG_SELL
        elif score <= -25:
            signal = Signal.SELL
        else:
            signal = Signal.HOLD
        
        return signal, strength, reasons
    
    def _calculate_levels(
        self, 
        price: float, 
        atr: float, 
        signal: Signal
    ) -> TradeLevels:
        """Calculate entry, stop loss, and take profit levels"""
        is_buy = signal in [Signal.STRONG_BUY, Signal.BUY]
        
        # Tighter stops for strong signals
        multiplier = 2.0 if signal in [Signal.STRONG_BUY, Signal.STRONG_SELL] else 2.5
        
        if is_buy:
            stop_loss = price - atr * multiplier
            target_1 = price + atr * 1.5
            target_2 = price + atr * 3.0
            target_3 = price + atr * 5.0
        else:
            stop_loss = price + atr * multiplier
            target_1 = price - atr * 1.5
            target_2 = price - atr * 3.0
            target_3 = price - atr * 5.0
        
        risk = abs(price - stop_loss)
        reward = abs(target_2 - price)
        rr = reward / risk if risk > 0 else 0
        
        return TradeLevels(
            entry=round(price, 2),
            stop_loss=round(stop_loss, 2),
            stop_loss_pct=round((stop_loss - price) / price * 100, 2),
            target_1=round(target_1, 2),
            target_1_pct=round((target_1 - price) / price * 100, 2),
            target_2=round(target_2, 2),
            target_2_pct=round((target_2 - price) / price * 100, 2),
            target_3=round(target_3, 2),
            target_3_pct=round((target_3 - price) / price * 100, 2),
            risk_reward=round(rr, 2)
        )
    
    def _calculate_position(
        self,
        signal: Signal,
        strength: float,
        price: float,
        levels: TradeLevels,
        confidence: float
    ) -> PositionSize:
        """Calculate position size using risk-based sizing"""
        if signal == Signal.HOLD or confidence < CONFIG.MIN_CONFIDENCE:
            return PositionSize(0, 0, 0, 0)
        
        # Risk per share
        risk_per_share = abs(price - levels.stop_loss)
        
        if risk_per_share <= 0:
            return PositionSize(0, 0, 0, 0)
        
        # Base risk amount (% of capital)
        base_risk = self.capital * (CONFIG.RISK_PER_TRADE / 100)
        
        # Adjust by signal strength and confidence
        adjusted_risk = base_risk * strength * confidence
        
        # Calculate shares
        shares = int(adjusted_risk / risk_per_share)
        
        # Round to lot size
        shares = (shares // CONFIG.LOT_SIZE) * CONFIG.LOT_SIZE
        
        # Apply position limit
        max_position = self.capital * (CONFIG.MAX_POSITION_PCT / 100)
        max_shares = int(max_position / price / CONFIG.LOT_SIZE) * CONFIG.LOT_SIZE
        shares = min(shares, max_shares)
        
        # Check if we can afford it
        available = self.capital * 0.95  # Keep 5% buffer
        affordable_shares = int(available / price / CONFIG.LOT_SIZE) * CONFIG.LOT_SIZE
        shares = min(shares, affordable_shares)
        
        if shares < CONFIG.LOT_SIZE:
            return PositionSize(0, 0, 0, 0)
        
        value = shares * price
        pct = value / self.capital * 100
        risk = shares * risk_per_share
        
        return PositionSize(
            shares=shares,
            value=round(value, 2),
            portfolio_pct=round(pct, 2),
            risk_amount=round(risk, 2)
        )
    
    def _generate_price_forecast(
        self,
        current_price: float,
        pred: EnsemblePrediction,
        atr_pct: float
    ) -> List[float]:
        """Generate future price predictions for visualization"""
        horizon = CONFIG.PREDICTION_HORIZON
        volatility = atr_pct / 100
        
        # Expected direction based on probabilities
        expected_return = (pred.prob_up - pred.prob_down) * 2  # Scale to ~-2% to +2%
        
        prices = [current_price]
        
        for i in range(horizon):
            # Add daily expected return plus some noise
            daily_return = expected_return / horizon
            noise = np.random.normal(0, volatility * 0.3)
            
            new_price = prices[-1] * (1 + (daily_return + noise) / 100)
            prices.append(round(new_price, 2))
        
        return prices
    
    def _generate_warnings(
        self, 
        pred: EnsemblePrediction, 
        rsi: float
    ) -> List[str]:
        """Generate warning messages"""
        warnings = []
        
        if pred.confidence < 0.5:
            warnings.append("Low model confidence - trade with caution")
        
        if pred.agreement < 0.5:
            warnings.append("Models disagree significantly")
        
        if rsi > 80:
            warnings.append("Extremely overbought - potential reversal")
        elif rsi < 20:
            warnings.append("Extremely oversold - may continue falling")
        
        return warnings
    
    def batch_predict(self, codes: List[str]) -> List[Prediction]:
        """Predict multiple stocks"""
        predictions = []
        
        for code in codes:
            try:
                pred = self.predict(code)
                predictions.append(pred)
            except Exception as e:
                log.warning(f"Failed to predict {code}: {e}")
        
        # Sort by signal strength
        predictions.sort(key=lambda p: (
            p.signal in [Signal.STRONG_BUY, Signal.STRONG_SELL],
            p.signal_strength,
            p.confidence
        ), reverse=True)
        
        return predictions
    
    def get_top_picks(
        self, 
        codes: List[str] = None, 
        n: int = 5,
        signal_type: str = "buy"
    ) -> List[Prediction]:
        """Get top N stock picks"""
        codes = codes or CONFIG.STOCK_POOL
        predictions = self.batch_predict(codes)
        
        if signal_type == "buy":
            filtered = [p for p in predictions if p.signal in [Signal.STRONG_BUY, Signal.BUY]]
        elif signal_type == "sell":
            filtered = [p for p in predictions if p.signal in [Signal.STRONG_SELL, Signal.SELL]]
        else:
            filtered = predictions
        
        return filtered[:n]