# models/predictor.py
"""
Stock Predictor - Generate Trading Signals
"""
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path

import pandas as pd
import torch
import threading

from config.settings import CONFIG
from data.fetcher import DataFetcher
from data.features import FeatureEngine
from data.processor import DataProcessor
from models.ensemble import EnsembleModel, EnsemblePrediction
from utils.logger import get_logger

log = get_logger(__name__)


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
class QuickPrediction:
    stock_code: str
    current_price: float
    prob_up: float
    prob_neutral: float
    prob_down: float
    confidence: float
    signal: Signal


@dataclass
class Prediction:
    """Complete prediction result"""
    stock_code: str
    stock_name: str
    current_price: float
    timestamp: datetime
    
    prob_up: float
    prob_neutral: float
    prob_down: float
    predicted_class: int
    confidence: float
    model_agreement: float
    
    signal: Signal
    signal_strength: float
    
    levels: TradeLevels
    position: PositionSize
    
    rsi: float = 50.0
    macd_signal: str = "neutral"
    trend: str = "sideways"
    
    reasons: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    price_history: List[float] = field(default_factory=list)
    predicted_prices: List[float] = field(default_factory=list)


class Predictor:
    """Stock Predictor - Generate trading signals and recommendations"""
    
    def __init__(self, capital: float = None):
        self.capital = capital or CONFIG.CAPITAL
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.fetcher = DataFetcher()
        self.feature_engine = FeatureEngine()
        self.processor = DataProcessor()
        self._lock = threading.RLock()
        
        self.ensemble: Optional[EnsembleModel] = None
        self._model_loaded = False
        
        self._feature_cache: Dict[str, Tuple[pd.DataFrame, datetime]] = {}
        self._cache_ttl_seconds = 60
        
        self._manifest_path = CONFIG.MODEL_DIR / "model_manifest.json"
        self._manifest_mtime = None

        self._load_model()
    
    def predict_quick_batch(
        self,
        codes: List[str],
        use_realtime_price: bool = True,
        interval: str = "1m",
        lookback_bars: int = 900
    ) -> List[QuickPrediction]:
        """Fast batch prediction for UI. Supports 1m mode."""
        interval = str(interval).lower()

        # For quick signals we assume horizon is your UI-selected horizon,
        # but the model loaded here is only used for probabilities anyway.
        # We'll load the "closest default" (30) so the UI behaves predictably.
        horizon = 30 if interval != "1d" else int(CONFIG.PREDICTION_HORIZON)
        self._ensure_model(interval=interval, horizon=horizon)

        if not self.is_ready():
            raise RuntimeError("Model not loaded")

        feature_cols = self.feature_engine.get_feature_columns()
        now = datetime.now()

        price_map = {}
        if use_realtime_price:
            try:
                quotes = self.fetcher.get_realtime_batch(codes)
                price_map = {k: float(v.price) for k, v in quotes.items() if v and v.price > 0}
            except Exception:
                price_map = {}

        seqs: List[np.ndarray] = []
        metas: List[Tuple[str, float, pd.DataFrame]] = []

        for code in codes:
            code = DataFetcher.clean_code(code)
            if not code:
                continue

            with self._lock:
                cached = self._feature_cache.get(code)
                if cached and (now - cached[1]).total_seconds() < self._cache_ttl_seconds:
                    df = cached[0]
                else:
                    df = self._fetch_and_process(code, interval=interval, lookback_bars=lookback_bars)
                    self._feature_cache[code] = (df, now)

            if len(df) < CONFIG.SEQUENCE_LENGTH + 5:
                continue

            last_close = float(df["close"].iloc[-1])
            price = float(price_map.get(code, last_close))

            X = self.processor.prepare_inference_sequence(df, feature_cols)
            seqs.append(X[0])
            metas.append((code, price, df))

        if not seqs:
            return []

        X_batch = np.stack(seqs, axis=0)
        preds = self.ensemble.predict_batch(X_batch)

        out: List[QuickPrediction] = []
        for (code, price, df), ep in zip(metas, preds):
            sig = Signal.HOLD
            if ep.predicted_class == 2 and ep.confidence >= CONFIG.MIN_CONFIDENCE:
                sig = Signal.BUY
            elif ep.predicted_class == 0 and ep.confidence >= CONFIG.MIN_CONFIDENCE:
                sig = Signal.SELL

            out.append(QuickPrediction(
                stock_code=code,
                current_price=price,
                prob_up=ep.prob_up,
                prob_neutral=ep.prob_neutral,
                prob_down=ep.prob_down,
                confidence=ep.confidence,
                signal=sig,
            ))

        return out

    def _ensure_model(self, interval: str, horizon: int):
        """
        Ensure the correct model/scaler (interval + horizon) is loaded.
        For 1m forecasting 30 minutes, horizon=30 bars.

        If not found, fall back to existing "ensemble.pt"/"scaler.pkl" behavior.
        """
        interval = str(interval).lower()
        horizon = int(horizon)

        key = f"{interval}_{horizon}"
        loaded_key = getattr(self, "_loaded_model_key", None)
        if loaded_key == key and self.is_ready():
            return

        # Preferred per-model files:
        model_path = CONFIG.MODEL_DIR / f"ensemble_{interval}_{horizon}.pt"
        scaler_path = CONFIG.MODEL_DIR / f"scaler_{interval}_{horizon}.pkl"

        # Fallback legacy files:
        if not model_path.exists():
            model_path = CONFIG.MODEL_DIR / "ensemble.pt"
        if not scaler_path.exists():
            scaler_path = CONFIG.MODEL_DIR / "scaler.pkl"

        # Load scaler (best effort)
        try:
            self.processor.load_scaler(str(scaler_path))
        except Exception:
            pass

        # Load ensemble
        if not model_path.exists():
            self.ensemble = None
            self._model_loaded = False
            self._loaded_model_key = None
            return

        try:
            state = torch.load(model_path, map_location="cpu", weights_only=False)
            input_size = int(state.get("input_size", len(FeatureEngine.FEATURE_NAMES)))

            self.ensemble = EnsembleModel(input_size)
            ok = self.ensemble.load(str(model_path))
            self._model_loaded = bool(ok)

            # store metadata for forecasting scaling
            meta = state.get("meta", {})
            self._trained_interval = str(meta.get("interval", interval))
            self._trained_horizon = int(meta.get("prediction_horizon", horizon))

            self._loaded_model_key = key if ok else None
        except Exception:
            self.ensemble = None
            self._model_loaded = False
            self._loaded_model_key = None

    def _load_model(self):
        """Load the trained ensemble model and scaler"""
        model_path = CONFIG.MODEL_DIR / "ensemble.pt"
        scaler_path = CONFIG.MODEL_DIR / "scaler.pkl"
        
        if not self.processor.load_scaler(str(scaler_path)):
            log.warning("No scaler found. Predictions may be inaccurate.")
            log.warning("Train a model first with: python main.py --train")
        
        if not model_path.exists():
            log.warning("No trained model found. Train a model first.")
            log.warning("Run: python main.py --train")
            return
        
        try:
            state = torch.load(model_path, map_location='cpu')
            input_size = state.get('input_size', len(FeatureEngine.FEATURE_NAMES))
            
            self.ensemble = EnsembleModel(input_size)
            
            if self.ensemble.load(str(model_path)):
                self._model_loaded = True
                log.info(f"Model loaded successfully ({len(self.ensemble.models)} networks)")
            else:
                self.ensemble = None
                log.error("Failed to load model")
                
        except Exception as e:
            log.error(f"Failed to load model: {e}")
            self.ensemble = None
    
    def is_ready(self) -> bool:
        """Check if predictor is ready"""
        return self._model_loaded and self.ensemble is not None
    
    def predict(
        self,
        stock_code: str,
        use_realtime_price: bool = False,
        interval: str = "1m",
        forecast_minutes: int = 30,
        lookback_bars: int = 1200
    ) -> Prediction:
        """
        Generate complete prediction for a stock.

        FIX:
        - interval="1m"
        - forecast_minutes controls future graph length (20~30)
        """
        interval = str(interval).lower()
        forecast_minutes = int(forecast_minutes)

        horizon_steps = forecast_minutes if interval != "1d" else int(CONFIG.PREDICTION_HORIZON)
        self._ensure_model(interval=interval, horizon=horizon_steps)

        if not self.is_ready():
            raise RuntimeError("Model not loaded. Train a model first.")

        stock_code = DataFetcher.clean_code(stock_code)
        now = datetime.now()

        with self._lock:
            cached = self._feature_cache.get(stock_code)
            if cached and (now - cached[1]).total_seconds() < self._cache_ttl_seconds:
                df = cached[0]
            else:
                df = self._fetch_and_process(stock_code, interval=interval, lookback_bars=lookback_bars)
                self._feature_cache[stock_code] = (df, now)

        if len(df) < CONFIG.SEQUENCE_LENGTH + 10:
            raise ValueError(f"Insufficient data for {stock_code}: {len(df)} bars")

        last_close = float(df["close"].iloc[-1])
        quote = self.fetcher.get_realtime(stock_code)
        stock_name = quote.name if quote else stock_code

        current_price = float(quote.price) if (use_realtime_price and quote and quote.price > 0) else last_close

        feature_cols = self.feature_engine.get_feature_columns()
        X = self.processor.prepare_inference_sequence(df, feature_cols)

        ensemble_pred = self.ensemble.predict(X)

        rsi = self._get_indicator(df, "rsi_14", default=50)
        macd = self._get_indicator(df, "macd_hist", default=0)
        ma_ratio = self._get_indicator(df, "ma_ratio_5_20", default=0)
        atr_pct = self._get_indicator(df, "atr_pct", default=2.0)

        signal, strength, reasons = self._calculate_signal(ensemble_pred, rsi, macd, ma_ratio)

        atr = current_price * atr_pct / 100.0
        levels = self._calculate_levels(current_price, atr, signal)

        position = self._calculate_position(signal, strength, current_price, levels, ensemble_pred.confidence)

        warnings = self._generate_warnings(ensemble_pred, rsi)

        # show recent history: for 1m show 120 points; for 1d show 60
        hist_len = 120 if interval != "1d" else 60
        price_history = df["close"].tail(hist_len).tolist()

        model_horizon = int(getattr(self, "_trained_horizon", horizon_steps))
        predicted_prices = self._generate_price_forecast(
            current_price=current_price,
            pred=ensemble_pred,
            atr_pct=atr_pct,
            horizon_steps=horizon_steps,
            model_horizon=model_horizon
        )

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
    
    def _maybe_reload_model(self):
        try:
            if self._manifest_path.exists():
                mtime = self._manifest_path.stat().st_mtime
                if self._manifest_mtime is None:
                    self._manifest_mtime = mtime
                elif mtime != self._manifest_mtime:
                    self._manifest_mtime = mtime
                    self._load_model()
        except Exception:
            pass

    def _get_indicator(self, df: pd.DataFrame, col: str, default: float) -> float:
        """Safely get an indicator from df, handling normalization."""
        try:
            if col not in df.columns or len(df) == 0:
                return float(default)

            v = df[col].iloc[-1]
            if v is None or (isinstance(v, float) and np.isnan(v)):
                return float(default)

            v = float(v)

            # RSI: features.py normalizes as: rsi / 100 - 0.5, giving range [-0.5, 0.5]
            # We need to convert back to [0, 100] for signal calculations
            if col.lower() in ("rsi", "rsi_14", "rsi_7"):
                # Check if value is in normalized range [-0.6, 0.6]
                if -0.6 <= v <= 0.6:
                    # Reverse normalization: (v + 0.5) * 100
                    return (v + 0.5) * 100.0
                # Already in 0-100 range (shouldn't happen with current features.py)
                elif 0.0 <= v <= 100.0:
                    return v
                # Fallback
                return float(default)

            return v
        except Exception:
            return float(default)
    
    def _fetch_and_process(self, stock_code: str, interval: str = "1d", lookback_bars: int = None) -> pd.DataFrame:
        """
        Fetch and process data for a stock.

        FIX:
        - supports 1m interval using bars
        """
        interval = str(interval).lower()

        if interval == "1d":
            df = self.fetcher.get_history(stock_code, days=500, interval="1d")
        else:
            lb = int(lookback_bars or max(800, CONFIG.SEQUENCE_LENGTH + 300))
            df = self.fetcher.get_history(stock_code, bars=lb, days=lb, interval=interval, use_cache=True)

        if df is None or df.empty:
            raise ValueError(f"No data for {stock_code} ({interval})")

        if len(df) < CONFIG.SEQUENCE_LENGTH + 20:
            raise ValueError(f"Insufficient data for {stock_code} ({interval}): {len(df)} bars")

        return self.feature_engine.create_features(df)

    def _calculate_signal(
        self,
        pred: EnsemblePrediction,
        rsi: float,
        macd: float,
        ma_ratio: float
    ) -> Tuple[Signal, float, List[str]]:
        """Calculate trading signal with scoring"""
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
            reasons.append(f"🤖 AI: Neutral (UP:{pred.prob_up:.0%})")
        
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
            reasons.append(f"⚠️ Models disagree ({pred.agreement:.0%})")
        elif pred.agreement > 0.8:
            score *= 1.1
            reasons.append(f"✅ Models agree ({pred.agreement:.0%})")
        
        # RSI (20% weight)
        if rsi < 30:
            score += 15
            reasons.append(f"📈 RSI oversold ({rsi:.0f})")
        elif rsi < 40:
            score += 8
        elif rsi > 70:
            score -= 15
            reasons.append(f"📉 RSI overbought ({rsi:.0f})")
        elif rsi > 60:
            score -= 8
        
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
            reasons.append(f"📈 Strong uptrend")
        elif ma_ratio > 1:
            score += 6
        elif ma_ratio < -3:
            score -= 12
            reasons.append(f"📉 Strong downtrend")
        elif ma_ratio < -1:
            score -= 6
        
        # Calculate signal strength
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
        """Calculate trading levels"""
        is_buy = signal in [Signal.STRONG_BUY, Signal.BUY]
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
        """Calculate position size"""
        if signal == Signal.HOLD or confidence < CONFIG.MIN_CONFIDENCE:
            return PositionSize(0, 0, 0, 0)
        
        risk_per_share = abs(price - levels.stop_loss)
        
        if risk_per_share <= 0:
            return PositionSize(0, 0, 0, 0)
        
        # Base risk amount
        base_risk = self.capital * (CONFIG.RISK_PER_TRADE / 100)
        adjusted_risk = base_risk * strength * confidence
        
        # Calculate shares
        shares = int(adjusted_risk / risk_per_share)
        shares = (shares // CONFIG.LOT_SIZE) * CONFIG.LOT_SIZE
        
        # Apply position limit
        max_position = self.capital * (CONFIG.MAX_POSITION_PCT / 100)
        max_shares = int(max_position / price / CONFIG.LOT_SIZE) * CONFIG.LOT_SIZE
        shares = min(shares, max_shares)
        
        # Check affordability
        available = self.capital * 0.95
        affordable = int(available / price / CONFIG.LOT_SIZE) * CONFIG.LOT_SIZE
        shares = min(shares, affordable)
        
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
        atr_pct: float,
        horizon_steps: int,
        model_horizon: int
    ) -> List[float]:
        """
        Deterministic future price path for charting.

        - horizon_steps: how many minutes to draw into the future (20~30)
        - model_horizon: the horizon the model was trained on (bars)

        If you forecast 30 but model was trained for 20, we scale drift approximately.
        """
        horizon_steps = int(horizon_steps)
        if horizon_steps <= 0:
            return [round(float(current_price), 2)]

        px0 = float(current_price)
        if px0 <= 0:
            return [0.0]

        up_thr = float(CONFIG.UP_THRESHOLD)
        dn_thr = float(CONFIG.DOWN_THRESHOLD)

        # Expected TOTAL return (%) over model horizon
        exp_total_model = (pred.prob_up * up_thr) + (pred.prob_down * dn_thr)

        # Confidence shrinkage
        exp_total_model *= float(0.35 + 0.65 * pred.confidence)

        # Scale drift from model horizon to requested horizon
        mh = max(1, int(model_horizon))
        exp_total = exp_total_model * (float(horizon_steps) / float(mh))

        # Small deterministic curvature based on ATR%
        vol = max(0.0001, float(atr_pct) / 100.0)

        prices = [px0]
        for step in range(1, horizon_steps + 1):
            step_ret = exp_total / horizon_steps  # % per step
            curve = (vol * 0.10) * (step / horizon_steps) * (1 if exp_total >= 0 else -1)
            px1 = prices[-1] * (1.0 + (step_ret / 100.0) + curve)
            prices.append(px1)

        return [round(float(p), 2) for p in prices]
    
    def _generate_warnings(
        self,
        pred: EnsemblePrediction,
        rsi: float
    ) -> List[str]:
        """Generate warnings"""
        warnings = []
        
        if pred.confidence < 0.5:
            warnings.append("Low model confidence - trade with caution")
        
        if pred.agreement < 0.5:
            warnings.append("Models disagree significantly")
        
        if rsi > 80:
            warnings.append("Extremely overbought")
        elif rsi < 20:
            warnings.append("Extremely oversold")
        
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
            filtered = [p for p in predictions 
                       if p.signal in [Signal.STRONG_BUY, Signal.BUY]]
        elif signal_type == "sell":
            filtered = [p for p in predictions 
                       if p.signal in [Signal.STRONG_SELL, Signal.SELL]]
        else:
            filtered = predictions
        
        return filtered[:n]