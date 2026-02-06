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
import time

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
        
        # Forecaster
        self._forecaster = None
        self._forecaster_device = self.device
        self._last_inference_X: Optional[np.ndarray] = None
        
        # Model metadata
        self._trained_interval = "1d"
        self._trained_horizon = CONFIG.PREDICTION_HORIZON
        self._loaded_model_key: Optional[str] = None

        self._load_model()
    
    def predict_quick_batch(
        self,
        codes: List[str],
        use_realtime_price: bool = True,
        interval: str = "1m",
        lookback_bars: int = 900
    ) -> List[QuickPrediction]:
        """Fast batch prediction for UI."""
        interval = str(interval).lower()
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
                    try:
                        df = self._fetch_and_process(code, interval=interval, lookback_bars=lookback_bars)
                        self._feature_cache[code] = (df, now)
                    except Exception as e:
                        log.warning(f"Failed to fetch {code}: {e}")
                        continue

            if len(df) < CONFIG.SEQUENCE_LENGTH + 5:
                continue

            last_close = float(df["close"].iloc[-1])
            price = float(price_map.get(code, last_close))

            try:
                X = self.processor.prepare_inference_sequence(df, feature_cols)
                seqs.append(X[0])
                metas.append((code, price, df))
            except Exception as e:
                log.warning(f"Failed to prepare sequence for {code}: {e}")
                continue

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

    def get_realtime_forecast_curve(
        self,
        stock_code: str,
        interval: str = "1m",
        horizon_steps: int = 30,
        lookback_bars: int = 1400,
        use_realtime_price: bool = True,
    ) -> Tuple[List[float], List[float]]:
        """
        Real-time AI graph: fetch latest intraday bars and produce forecast curve.
        """
        interval = str(interval).lower()
        horizon_steps = int(horizon_steps)
        stock_code = DataFetcher.clean_code(stock_code)

        self._ensure_model(interval=interval, horizon=horizon_steps)
        if not self.is_ready():
            raise RuntimeError("Model not loaded")

        feature_cols = self.feature_engine.get_feature_columns()

        df = self.fetcher.get_history(
            stock_code,
            interval=interval,
            bars=int(lookback_bars),
            days=int(lookback_bars),
            use_cache=True,
            update_db=True,
        )
        if df is None or df.empty:
            raise ValueError(f"No data for {stock_code} ({interval})")

        df = self.feature_engine.create_features(df)
        if len(df) < CONFIG.SEQUENCE_LENGTH + 20:
            raise ValueError(f"Insufficient data for forecast: {stock_code} ({interval}) len={len(df)}")

        last_close = float(df["close"].iloc[-1])
        current_price = last_close
        if use_realtime_price:
            try:
                q = self.fetcher.get_realtime(stock_code)
                if q and q.price and float(q.price) > 0:
                    current_price = float(q.price)
            except Exception:
                pass

        X = self.processor.prepare_inference_sequence(df, feature_cols)
        self._last_inference_X = X

        atr_pct = float(df["atr_pct"].iloc[-1]) if "atr_pct" in df.columns else 2.0

        model_horizon = int(getattr(self, "_trained_horizon", horizon_steps))
        predicted_prices = self._generate_price_forecast(
            current_price=current_price,
            pred=self.ensemble.predict(X),
            atr_pct=atr_pct,
            horizon_steps=horizon_steps,
            model_horizon=model_horizon,
        )

        hist_len = 180 if interval != "1d" else 60
        actual_prices = df["close"].tail(hist_len).tolist()
        if actual_prices:
            actual_prices[-1] = float(current_price)

        return actual_prices, predicted_prices

    def _ensure_model(self, interval: str, horizon: int):
        """Ensure the correct classifier ensemble + forecaster are loaded."""
        interval = str(interval).lower()
        horizon = int(horizon)

        key = f"{interval}_{horizon}"
        loaded_key = getattr(self, "_loaded_model_key", None)
        if loaded_key == key and self.is_ready():
            return

        model_path = CONFIG.MODEL_DIR / f"ensemble_{interval}_{horizon}.pt"
        scaler_path = CONFIG.MODEL_DIR / f"scaler_{interval}_{horizon}.pkl"
        forecast_path = CONFIG.MODEL_DIR / f"forecast_{interval}_{horizon}.pt"

        # Legacy fallbacks
        if not model_path.exists():
            model_path = CONFIG.MODEL_DIR / "ensemble.pt"
        if not scaler_path.exists():
            scaler_path = CONFIG.MODEL_DIR / "scaler.pkl"

        # Load scaler
        try:
            self.processor.load_scaler(str(scaler_path))
        except Exception as e:
            log.warning(f"Failed to load scaler: {e}")

        # Load classifier ensemble
        if not model_path.exists():
            self.ensemble = None
            self._model_loaded = False
            self._loaded_model_key = None
            self._forecaster = None
            log.warning(f"Model not found: {model_path}")
            return

        try:
            state = torch.load(model_path, map_location="cpu", weights_only=False)
            input_size = int(state.get("input_size", len(FeatureEngine.FEATURE_NAMES)))

            self.ensemble = EnsembleModel(input_size)
            ok = self.ensemble.load(str(model_path))
            self._model_loaded = bool(ok)

            meta = state.get("meta", {})
            self._trained_interval = str(meta.get("interval", interval))
            self._trained_horizon = int(meta.get("prediction_horizon", horizon))
            self._loaded_model_key = key if ok else None
            
        except Exception as e:
            log.error(f"Failed to load model: {e}")
            self.ensemble = None
            self._model_loaded = False
            self._loaded_model_key = None
            self._forecaster = None
            return

        # Load forecaster (optional)
        self._forecaster = None
        try:
            if forecast_path.exists():
                from models.networks import TCNModel
                
                fstate = torch.load(forecast_path, map_location="cpu", weights_only=False)
                finput = int(fstate.get("input_size", input_size))
                fh = int(fstate.get("horizon", horizon))
                arch = fstate.get("arch", {})
                hidden = int(arch.get("hidden_size", CONFIG.model.hidden_size))
                drop = float(arch.get("dropout", CONFIG.model.dropout))

                device = "cuda" if torch.cuda.is_available() else "cpu"
                fore = TCNModel(
                    input_size=finput,
                    hidden_size=hidden,
                    num_classes=fh,
                    dropout=drop
                ).to(device)
                fore.load_state_dict(fstate["state_dict"])
                fore.eval()
                self._forecaster = fore
                self._forecaster_device = device
                log.info(f"Forecaster loaded: {forecast_path}")
        except Exception as e:
            log.warning(f"Failed to load forecaster: {e}")
            self._forecaster = None

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
            state = torch.load(model_path, map_location='cpu', weights_only=False)
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
        """Generate complete prediction for a stock."""
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
        self._last_inference_X = X

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
            # Convert back to [0, 100] for signal calculations
            if col.lower() in ("rsi", "rsi_14", "rsi_7"):
                if -0.6 <= v <= 0.6:
                    return (v + 0.5) * 100.0
                elif 0.0 <= v <= 100.0:
                    return v
                return float(default)

            return v
        except Exception:
            return float(default)
    
    def _fetch_and_process(self, stock_code: str, interval: str = "1d", lookback_bars: int = None) -> pd.DataFrame:
        """Fetch and process data for a stock."""
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
            reasons.append("📈 Strong uptrend")
        elif ma_ratio > 1:
            score += 6
        elif ma_ratio < -3:
            score -= 12
            reasons.append("📉 Strong downtrend")
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
        
        base_risk = self.capital * (CONFIG.RISK_PER_TRADE / 100)
        adjusted_risk = base_risk * strength * confidence
        
        shares = int(adjusted_risk / risk_per_share)
        shares = (shares // CONFIG.LOT_SIZE) * CONFIG.LOT_SIZE
        
        max_position = self.capital * (CONFIG.MAX_POSITION_PCT / 100)
        max_shares = int(max_position / price / CONFIG.LOT_SIZE) * CONFIG.LOT_SIZE
        shares = min(shares, max_shares)
        
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
        Generate price forecast curve using AI forecaster if available,
        otherwise use heuristic approach.
        """
        horizon_steps = int(horizon_steps)
        if horizon_steps <= 0:
            return [round(float(current_price), 2)]

        px0 = float(current_price)
        if px0 <= 0:
            return [0.0]

        # Use AI forecaster if available
        fore = getattr(self, "_forecaster", None)
        if fore is not None and self._last_inference_X is not None:
            try:
                X = self._last_inference_X
                device = getattr(self, "_forecaster_device", "cpu")
                xb = torch.FloatTensor(X).to(device)

                with torch.inference_mode():
                    out, _ = fore(xb)
                    rets = out.detach().cpu().numpy()[0].astype(float)

                rets = rets[:horizon_steps]

                # Safety clamp
                cap = max(1.0, float(atr_pct) * 3.0)
                rets = np.clip(rets, -cap, cap)

                prices = [px0]
                for k in range(len(rets)):
                    prices.append(px0 * (1.0 + rets[k] / 100.0))

                return [round(float(p), 2) for p in prices]
            except Exception as e:
                log.debug(f"Forecaster inference failed: {e}")

        # Fallback heuristic
        up_thr = float(CONFIG.UP_THRESHOLD)
        dn_thr = float(CONFIG.DOWN_THRESHOLD)

        exp_total_model = (pred.prob_up * up_thr) + (pred.prob_down * dn_thr)
        exp_total_model *= float(0.35 + 0.65 * pred.confidence)

        mh = max(1, int(model_horizon))
        exp_total = exp_total_model * (float(horizon_steps) / float(mh))

        vol = max(0.0001, float(atr_pct) / 100.0)

        prices = [px0]
        for step in range(1, horizon_steps + 1):
            step_ret = exp_total / horizon_steps
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
    
    def clear_cache(self):
        """Clear feature cache"""
        with self._lock:
            self._feature_cache.clear()