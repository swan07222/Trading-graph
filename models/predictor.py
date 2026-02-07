# models/predictor.py
"""
AI Stock Predictor - Real-time prediction with ensemble models

FIXED:
- Integration with updated DataProcessor and RealtimePredictor
- Proper interval/horizon handling
- Real-time forecast curve generation
- Robust error handling
"""
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Optional, List, Dict, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import threading

from config.settings import CONFIG
from utils.logger import get_logger

log = get_logger(__name__)


class Signal(Enum):
    """Trading signal types"""
    STRONG_BUY = "STRONG_BUY"
    BUY = "BUY"
    HOLD = "HOLD"
    SELL = "SELL"
    STRONG_SELL = "STRONG_SELL"


@dataclass
class TradingLevels:
    """Trading price levels"""
    entry: float = 0.0
    stop_loss: float = 0.0
    stop_loss_pct: float = 0.0
    target_1: float = 0.0
    target_1_pct: float = 0.0
    target_2: float = 0.0
    target_2_pct: float = 0.0
    target_3: float = 0.0
    target_3_pct: float = 0.0


@dataclass
class PositionSize:
    """Position sizing information"""
    shares: int = 0
    value: float = 0.0
    risk_amount: float = 0.0
    risk_pct: float = 0.0


@dataclass
class Prediction:
    """Complete prediction result"""
    stock_code: str
    stock_name: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Signal
    signal: Signal = Signal.HOLD
    signal_strength: float = 0.0
    confidence: float = 0.0
    
    # Probabilities
    prob_up: float = 0.33
    prob_neutral: float = 0.34
    prob_down: float = 0.33
    
    # Price data
    current_price: float = 0.0
    price_history: List[float] = field(default_factory=list)
    predicted_prices: List[float] = field(default_factory=list)
    
    # Technical indicators
    rsi: float = 50.0
    macd_signal: str = "NEUTRAL"
    trend: str = "NEUTRAL"
    
    # Trading levels
    levels: TradingLevels = field(default_factory=TradingLevels)
    
    # Position sizing
    position: PositionSize = field(default_factory=PositionSize)
    
    # Analysis
    reasons: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    # Metadata
    interval: str = "1d"
    horizon: int = 5


class Predictor:
    """
    AI Stock Predictor with real-time capabilities.
    
    Features:
    - Ensemble model predictions
    - Multi-step price forecasting
    - Real-time chart updates
    - Multiple interval support (1m, 5m, 1d, etc.)
    """
    
    def __init__(
        self, 
        capital: float = None,
        interval: str = "1d",
        prediction_horizon: int = None
    ):
        """
        Initialize predictor.
        
        Args:
            capital: Trading capital
            interval: Default data interval
            prediction_horizon: Default prediction horizon
        """
        self.capital = float(capital or CONFIG.CAPITAL)
        self.interval = str(interval).lower()
        self.horizon = int(prediction_horizon or CONFIG.PREDICTION_HORIZON)
        
        self._lock = threading.RLock()
        
        # Components (lazy loaded)
        self.ensemble = None
        self.forecaster = None
        self.processor = None
        self.feature_engine = None
        self.fetcher = None
        
        self._feature_cols: List[str] = []
        self._loaded = False
        
        # Load models
        self._load_models()
    
    def _load_models(self) -> bool:
        """Load all required models"""
        try:
            from data.processor import DataProcessor
            from data.features import FeatureEngine
            from data.fetcher import get_fetcher
            from models.ensemble import EnsembleModel
            
            # Initialize components
            self.processor = DataProcessor()
            self.feature_engine = FeatureEngine()
            self.fetcher = get_fetcher()
            self._feature_cols = self.feature_engine.get_feature_columns()
            
            # Load scaler
            scaler_path = CONFIG.MODEL_DIR / f"scaler_{self.interval}_{self.horizon}.pkl"
            if not scaler_path.exists():
                # Try default
                scaler_path = CONFIG.MODEL_DIR / "scaler_1d_5.pkl"
            
            if scaler_path.exists():
                self.processor.load_scaler(str(scaler_path))
            else:
                log.warning("No scaler found")
            
            # Load ensemble
            ensemble_path = CONFIG.MODEL_DIR / f"ensemble_{self.interval}_{self.horizon}.pt"
            if not ensemble_path.exists():
                # Try default
                ensemble_path = CONFIG.MODEL_DIR / "ensemble_1d_5.pt"
            
            if ensemble_path.exists():
                input_size = self.processor.n_features or len(self._feature_cols)
                self.ensemble = EnsembleModel(input_size=input_size)
                self.ensemble.load(str(ensemble_path))
                log.info(f"Ensemble loaded from {ensemble_path}")
            else:
                log.warning("No ensemble model found")
            
            # Load forecaster (optional)
            self._load_forecaster()
            
            self._loaded = True
            return True
            
        except Exception as e:
            log.error(f"Failed to load models: {e}")
            return False
    
    def _load_forecaster(self):
        """Load TCN forecaster for price curve prediction"""
        try:
            import torch
            from models.networks import TCNModel
            
            forecast_path = CONFIG.MODEL_DIR / f"forecast_{self.interval}_{self.horizon}.pt"
            if not forecast_path.exists():
                forecast_path = CONFIG.MODEL_DIR / "forecast_1d_5.pt"
            
            if not forecast_path.exists():
                log.debug("No forecaster model found")
                return
            
            data = torch.load(forecast_path, map_location='cpu', weights_only=False)
            
            self.forecaster = TCNModel(
                input_size=data['input_size'],
                hidden_size=data['arch']['hidden_size'],
                num_classes=data['horizon'],
                dropout=data['arch']['dropout']
            )
            self.forecaster.load_state_dict(data['state_dict'])
            self.forecaster.eval()
            
            log.info(f"Forecaster loaded: horizon={data['horizon']}")
            
        except Exception as e:
            log.debug(f"Forecaster not loaded: {e}")
            self.forecaster = None
    
    def predict(
        self,
        stock_code: str,
        use_realtime_price: bool = True,
        interval: str = None,
        forecast_minutes: int = None,
        lookback_bars: int = None
    ) -> Prediction:
        """
        Make full prediction with price forecast.
        
        Args:
            stock_code: Stock code to predict
            use_realtime_price: Use real-time price data
            interval: Data interval (overrides default)
            forecast_minutes: Forecast horizon in minutes/bars
            lookback_bars: Historical bars to fetch
        
        Returns:
            Complete Prediction object
        """
        interval = str(interval or self.interval).lower()
        horizon = int(forecast_minutes or self.horizon)
        lookback = int(lookback_bars or (1400 if interval == "1m" else 600))
        
        # Clean stock code
        code = self._clean_code(stock_code)
        
        # Create base prediction
        pred = Prediction(
            stock_code=code,
            timestamp=datetime.now(),
            interval=interval,
            horizon=horizon
        )
        
        try:
            # Fetch data
            df = self._fetch_data(code, interval, lookback, use_realtime_price)
            
            if df is None or df.empty or len(df) < CONFIG.SEQUENCE_LENGTH:
                pred.warnings.append("Insufficient data")
                return pred
            
            # Get stock name
            pred.stock_name = self._get_stock_name(code, df)
            
            # Current price
            pred.current_price = float(df['close'].iloc[-1])
            
            # Price history
            pred.price_history = df['close'].tail(180).tolist()
            
            # Create features
            df = self.feature_engine.create_features(df)
            
            # Get technical indicators
            self._extract_technicals(df, pred)
            
            # Prepare sequence for prediction
            X = self.processor.prepare_inference_sequence(df, self._feature_cols)
            
            # Ensemble prediction
            if self.ensemble:
                ensemble_pred = self.ensemble.predict(X)
                
                pred.prob_up = float(ensemble_pred.probabilities[2])
                pred.prob_neutral = float(ensemble_pred.probabilities[1])
                pred.prob_down = float(ensemble_pred.probabilities[0])
                pred.confidence = float(ensemble_pred.confidence)
                
                # Determine signal
                pred.signal = self._determine_signal(ensemble_pred, pred)
                pred.signal_strength = self._calculate_signal_strength(ensemble_pred, pred)
            
            # Generate price forecast
            pred.predicted_prices = self._generate_forecast(X, pred.current_price, horizon)
            
            # Calculate trading levels
            pred.levels = self._calculate_levels(pred)
            
            # Calculate position size
            pred.position = self._calculate_position(pred)
            
            # Generate analysis reasons
            self._generate_reasons(pred)
            
        except Exception as e:
            log.error(f"Prediction failed for {code}: {e}")
            pred.warnings.append(f"Prediction error: {str(e)}")
        
        return pred
    
    def predict_quick_batch(
        self,
        stock_codes: List[str],
        use_realtime_price: bool = True,
        interval: str = None,
        lookback_bars: int = None
    ) -> List[Prediction]:
        """
        Quick batch prediction without full forecasting.
        Faster for scanning multiple stocks.
        """
        interval = str(interval or self.interval).lower()
        lookback = int(lookback_bars or (1400 if interval == "1m" else 300))
        
        predictions = []
        
        for code in stock_codes:
            try:
                code = self._clean_code(code)
                
                pred = Prediction(
                    stock_code=code,
                    timestamp=datetime.now(),
                    interval=interval
                )
                
                # Fetch data
                df = self._fetch_data(code, interval, lookback, use_realtime_price)
                
                if df is None or df.empty or len(df) < CONFIG.SEQUENCE_LENGTH:
                    continue
                
                # Current price
                pred.current_price = float(df['close'].iloc[-1])
                
                # Create features
                df = self.feature_engine.create_features(df)
                
                # Prepare sequence
                X = self.processor.prepare_inference_sequence(df, self._feature_cols)
                
                # Quick ensemble prediction
                if self.ensemble:
                    ensemble_pred = self.ensemble.predict(X)
                    
                    pred.prob_up = float(ensemble_pred.probabilities[2])
                    pred.prob_neutral = float(ensemble_pred.probabilities[1])
                    pred.prob_down = float(ensemble_pred.probabilities[0])
                    pred.confidence = float(ensemble_pred.confidence)
                    pred.signal = self._determine_signal(ensemble_pred, pred)
                
                predictions.append(pred)
                
            except Exception as e:
                log.debug(f"Quick prediction failed for {code}: {e}")
        
        return predictions
    
    def get_realtime_forecast_curve(
        self,
        stock_code: str,
        interval: str = None,
        horizon_steps: int = None,
        lookback_bars: int = None,
        use_realtime_price: bool = True
    ) -> Tuple[List[float], List[float]]:
        """
        Get real-time forecast curve for charting.
        
        Returns:
            (actual_prices, predicted_prices)
        """
        interval = str(interval or self.interval).lower()
        horizon = int(horizon_steps or self.horizon)
        lookback = int(lookback_bars or (1400 if interval == "1m" else 600))
        
        code = self._clean_code(stock_code)
        
        try:
            # Fetch data
            df = self._fetch_data(code, interval, lookback, use_realtime_price)
            
            if df is None or df.empty or len(df) < CONFIG.SEQUENCE_LENGTH:
                return [], []
            
            # Actual prices
            actual = df['close'].tail(180).tolist()
            current_price = float(df['close'].iloc[-1])
            
            # Create features
            df = self.feature_engine.create_features(df)
            
            # Prepare sequence
            X = self.processor.prepare_inference_sequence(df, self._feature_cols)
            
            # Generate forecast
            predicted = self._generate_forecast(X, current_price, horizon)
            
            return actual, predicted
            
        except Exception as e:
            log.warning(f"Forecast curve failed for {code}: {e}")
            return [], []
    
    def get_top_picks(
        self,
        stock_codes: List[str],
        n: int = 10,
        signal_type: str = "buy"
    ) -> List[Prediction]:
        """
        Get top N stock picks based on signal type.
        
        Args:
            stock_codes: List of codes to scan
            n: Number of top picks to return
            signal_type: "buy" or "sell"
        """
        predictions = self.predict_quick_batch(stock_codes)
        
        # Filter by signal type
        if signal_type.lower() == "buy":
            filtered = [
                p for p in predictions
                if p.signal in [Signal.STRONG_BUY, Signal.BUY]
                and p.confidence >= CONFIG.MIN_CONFIDENCE
            ]
        else:
            filtered = [
                p for p in predictions
                if p.signal in [Signal.STRONG_SELL, Signal.SELL]
                and p.confidence >= CONFIG.MIN_CONFIDENCE
            ]
        
        # Sort by confidence
        filtered.sort(key=lambda x: x.confidence, reverse=True)
        
        return filtered[:n]
    
    def _fetch_data(
        self,
        code: str,
        interval: str,
        lookback: int,
        use_realtime: bool
    ) -> Optional[pd.DataFrame]:
        """Fetch stock data"""
        try:
            df = self.fetcher.get_history(
                code,
                interval=interval,
                bars=lookback,
                days=lookback,
                use_cache=True,
                update_db=True
            )
            
            if df is None or df.empty:
                return None
            
            # Update with realtime price if requested
            if use_realtime:
                try:
                    quote = self.fetcher.get_realtime(code)
                    if quote and quote.price > 0:
                        # Update last row
                        df.loc[df.index[-1], 'close'] = quote.price
                        df.loc[df.index[-1], 'high'] = max(df['high'].iloc[-1], quote.price)
                        df.loc[df.index[-1], 'low'] = min(df['low'].iloc[-1], quote.price)
                except Exception:
                    pass
            
            return df
            
        except Exception as e:
            log.warning(f"Failed to fetch data for {code}: {e}")
            return None
    
    def _generate_forecast(
        self,
        X: np.ndarray,
        current_price: float,
        horizon: int
    ) -> List[float]:
        """Generate price forecast using forecaster or ensemble"""
        if self.forecaster is not None:
            try:
                import torch
                
                self.forecaster.eval()
                with torch.inference_mode():
                    X_tensor = torch.FloatTensor(X)
                    returns, _ = self.forecaster(X_tensor)
                    returns = returns[0].cpu().numpy()
                
                # Convert returns to prices
                prices = [current_price]
                for r in returns[:horizon]:
                    next_price = prices[-1] * (1 + r / 100)
                    prices.append(float(next_price))
                
                return prices[1:]  # Exclude current price
                
            except Exception as e:
                log.debug(f"Forecaster failed: {e}")
        
        # Fallback: use ensemble probabilities
        if self.ensemble:
            try:
                pred = self.ensemble.predict(X)
                
                # Simple directional forecast
                direction = pred.probabilities[2] - pred.probabilities[0]
                volatility = 0.02  # 2% base volatility
                
                prices = []
                price = current_price
                
                for i in range(horizon):
                    change = direction * volatility * (1 - i / horizon)  # Decay
                    price = price * (1 + change)
                    prices.append(float(price))
                
                return prices
                
            except Exception:
                pass
        
        # Ultimate fallback: flat forecast
        return [current_price] * horizon
    
    def _determine_signal(self, ensemble_pred, pred: Prediction) -> Signal:
        """Determine trading signal from prediction"""
        prob_up = ensemble_pred.probabilities[2]
        prob_down = ensemble_pred.probabilities[0]
        confidence = ensemble_pred.confidence
        predicted_class = ensemble_pred.predicted_class
        
        if predicted_class == 2:  # UP
            if confidence >= CONFIG.STRONG_BUY_THRESHOLD:
                return Signal.STRONG_BUY
            elif confidence >= CONFIG.BUY_THRESHOLD:
                return Signal.BUY
        elif predicted_class == 0:  # DOWN
            if confidence >= CONFIG.STRONG_SELL_THRESHOLD:
                return Signal.STRONG_SELL
            elif confidence >= CONFIG.SELL_THRESHOLD:
                return Signal.SELL
        
        return Signal.HOLD
    
    def _calculate_signal_strength(self, ensemble_pred, pred: Prediction) -> float:
        """Calculate signal strength 0-1"""
        confidence = ensemble_pred.confidence
        agreement = ensemble_pred.agreement
        entropy = 1 - ensemble_pred.entropy
        
        return float((confidence + agreement + entropy) / 3)
    
    def _calculate_levels(self, pred: Prediction) -> TradingLevels:
        """Calculate trading levels"""
        price = pred.current_price
        
        if price <= 0:
            return TradingLevels()
        
        # ATR-based stops (simplified)
        atr_pct = 0.02  # 2% default
        
        levels = TradingLevels(entry=price)
        
        if pred.signal in [Signal.STRONG_BUY, Signal.BUY]:
            levels.stop_loss = price * (1 - atr_pct * 1.5)
            levels.target_1 = price * (1 + atr_pct * 1.5)
            levels.target_2 = price * (1 + atr_pct * 3.0)
            levels.target_3 = price * (1 + atr_pct * 5.0)
        elif pred.signal in [Signal.STRONG_SELL, Signal.SELL]:
            levels.stop_loss = price * (1 + atr_pct * 1.5)
            levels.target_1 = price * (1 - atr_pct * 1.5)
            levels.target_2 = price * (1 - atr_pct * 3.0)
            levels.target_3 = price * (1 - atr_pct * 5.0)
        else:
            levels.stop_loss = price * 0.97
            levels.target_1 = price * 1.03
            levels.target_2 = price * 1.05
            levels.target_3 = price * 1.08
        
        # Calculate percentages
        levels.stop_loss_pct = (levels.stop_loss / price - 1) * 100
        levels.target_1_pct = (levels.target_1 / price - 1) * 100
        levels.target_2_pct = (levels.target_2 / price - 1) * 100
        levels.target_3_pct = (levels.target_3 / price - 1) * 100
        
        return levels
    
    def _calculate_position(self, pred: Prediction) -> PositionSize:
        """Calculate position size"""
        price = pred.current_price
        
        if price <= 0:
            return PositionSize()
        
        # Risk-based position sizing
        risk_pct = CONFIG.RISK_PER_TRADE / 100
        risk_amount = self.capital * risk_pct
        
        # Stop distance
        stop_distance = abs(price - pred.levels.stop_loss)
        if stop_distance <= 0:
            stop_distance = price * 0.02
        
        # Position size
        shares = int(risk_amount / stop_distance)
        shares = (shares // CONFIG.LOT_SIZE) * CONFIG.LOT_SIZE
        shares = max(CONFIG.LOT_SIZE, shares)
        
        # Cap at max position
        max_value = self.capital * (CONFIG.MAX_POSITION_PCT / 100)
        if shares * price > max_value:
            shares = int(max_value / price)
            shares = (shares // CONFIG.LOT_SIZE) * CONFIG.LOT_SIZE
        
        return PositionSize(
            shares=shares,
            value=shares * price,
            risk_amount=shares * stop_distance,
            risk_pct=(shares * stop_distance / self.capital) * 100
        )
    
    def _extract_technicals(self, df: pd.DataFrame, pred: Prediction):
        """Extract technical indicators from dataframe"""
        try:
            # RSI
            if 'rsi_14' in df.columns:
                pred.rsi = float(df['rsi_14'].iloc[-1]) * 100 + 50
            
            # MACD
            if 'macd_hist' in df.columns:
                macd = float(df['macd_hist'].iloc[-1])
                if macd > 0.001:
                    pred.macd_signal = "BULLISH"
                elif macd < -0.001:
                    pred.macd_signal = "BEARISH"
                else:
                    pred.macd_signal = "NEUTRAL"
            
            # Trend
            if 'ma_ratio_5_20' in df.columns:
                trend = float(df['ma_ratio_5_20'].iloc[-1])
                if trend > 1:
                    pred.trend = "UPTREND"
                elif trend < -1:
                    pred.trend = "DOWNTREND"
                else:
                    pred.trend = "SIDEWAYS"
                    
        except Exception as e:
            log.debug(f"Technical extraction error: {e}")
    
    def _generate_reasons(self, pred: Prediction):
        """Generate analysis reasons"""
        reasons = []
        warnings = []
        
        # AI confidence
        if pred.confidence >= 0.7:
            reasons.append(f"High AI confidence: {pred.confidence:.0%}")
        elif pred.confidence >= 0.6:
            reasons.append(f"Moderate AI confidence: {pred.confidence:.0%}")
        else:
            warnings.append(f"Low AI confidence: {pred.confidence:.0%}")
        
        # Probabilities
        if pred.prob_up > 0.5:
            reasons.append(f"AI predicts UP with {pred.prob_up:.0%} probability")
        elif pred.prob_down > 0.5:
            reasons.append(f"AI predicts DOWN with {pred.prob_down:.0%} probability")
        
        # RSI
        if pred.rsi > 70:
            warnings.append(f"RSI overbought: {pred.rsi:.0f}")
        elif pred.rsi < 30:
            warnings.append(f"RSI oversold: {pred.rsi:.0f}")
        else:
            reasons.append(f"RSI neutral: {pred.rsi:.0f}")
        
        # Trend alignment
        if pred.signal in [Signal.STRONG_BUY, Signal.BUY] and pred.trend == "UPTREND":
            reasons.append("Signal aligned with uptrend")
        elif pred.signal in [Signal.STRONG_SELL, Signal.SELL] and pred.trend == "DOWNTREND":
            reasons.append("Signal aligned with downtrend")
        elif pred.trend != "SIDEWAYS":
            warnings.append(f"Signal against trend ({pred.trend})")
        
        # MACD
        if pred.macd_signal != "NEUTRAL":
            reasons.append(f"MACD: {pred.macd_signal}")
        
        pred.reasons = reasons
        pred.warnings = warnings
    
    def _get_stock_name(self, code: str, df: pd.DataFrame) -> str:
        """Get stock name"""
        try:
            quote = self.fetcher.get_realtime(code)
            if quote and quote.name:
                return quote.name
        except Exception:
            pass
        return ""
    
    def _clean_code(self, code: str) -> str:
        """Clean and normalize stock code"""
        if not code:
            return ""
        
        code = str(code).strip()
        
        # Remove prefixes
        for prefix in ['sh', 'sz', 'SH', 'SZ', 'bj', 'BJ']:
            if code.startswith(prefix):
                code = code[len(prefix):]
        
        # Remove suffixes
        for suffix in ['.SS', '.SZ', '.BJ']:
            if code.endswith(suffix):
                code = code[:-len(suffix)]
        
        # Keep only digits
        code = ''.join(c for c in code if c.isdigit())
        
        return code.zfill(6) if code else ""