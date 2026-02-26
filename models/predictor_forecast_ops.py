from __future__ import annotations

import copy
import threading
import time
from typing import Any

import numpy as np
import pandas as pd

from config.settings import CONFIG
from models.predictor import (
    _PREDICTOR_RECOVERABLE_EXCEPTIONS,
    FloatArray,
    Prediction,
    Signal,
)
from utils.logger import get_logger

log = get_logger(__name__)

def get_realtime_forecast_curve(
    self,
    stock_code: str,
    interval: str = None,
    horizon_steps: int = None,
    lookback_bars: int = None,
    use_realtime_price: bool = True,
    history_allow_online: bool = True,
) -> tuple[list[float], list[float]]:
    """Get real-time forecast curve for charting.

    Returns:
        (actual_prices, predicted_prices)
    """
    with self._predict_lock:
        self._maybe_reload_models(reason="forecast_curve")
        interval = self._normalize_interval_token(interval)
        horizon = max(
            1,
            int(horizon_steps if horizon_steps is not None else 30),
        )
        default_lookback = int(self._default_lookback_bars(interval))
        if lookback_bars is None and self._is_intraday_interval(interval):
            try:
                from data.fetcher_sources import BARS_PER_DAY

                bars_per_day = float(BARS_PER_DAY.get(interval, 1.0) or 1.0)
            except _PREDICTOR_RECOVERABLE_EXCEPTIONS:
                fallback_bars_per_day = {
                    "1m": 240.0,
                    "2m": 120.0,
                    "3m": 80.0,
                    "5m": 48.0,
                    "15m": 16.0,
                    "30m": 8.0,
                    "60m": 4.0,
                    "1h": 4.0,
                }
                bars_per_day = float(fallback_bars_per_day.get(interval, 1.0))
            # Forecast-curve charts are more stable when seeded from
            # roughly one trading week of intraday context.
            default_lookback = max(
                default_lookback,
                int(round(max(1.0, bars_per_day) * 7.0)),
            )
        lookback = max(
            120,
            int(
                lookback_bars
                if lookback_bars is not None
                else default_lookback
            ),
        )

        code = self._clean_code(stock_code)

        try:
            min_rows = getattr(
                self.feature_engine, 'MIN_ROWS',
                CONFIG.SEQUENCE_LENGTH
            )
            window = int(
                max(lookback, int(min_rows), int(CONFIG.SEQUENCE_LENGTH))
            )

            try:
                df = self.fetcher.get_history(
                    code,
                    interval=interval,
                    bars=window,
                    use_cache=False,
                    update_db=True,
                    allow_online=bool(history_allow_online),
                )
            except TypeError:
                df = self.fetcher.get_history(
                    code,
                    interval=interval,
                    bars=window,
                    use_cache=False,
                    update_db=True,
                )

            # Merge latest session bars (including partial intraday bars)
            # so realtime guessed curve follows the current live candle.
            if interval in {"1m", "3m", "5m", "15m", "30m", "60m", "1h"}:
                try:
                    from data.session_cache import get_session_bar_cache

                    s_df = get_session_bar_cache().read_history(
                        symbol=code,
                        interval=interval,
                        bars=window,
                        final_only=False,
                    )
                    if s_df is not None and not s_df.empty:
                        parts: list[pd.DataFrame] = []
                        for part in (df, s_df):
                            if part is None or part.empty:
                                continue
                            p = part.copy()
                            if not isinstance(p.index, pd.DatetimeIndex):
                                if "datetime" in p.columns:
                                    p["datetime"] = pd.to_datetime(
                                        p["datetime"],
                                        errors="coerce",
                                    )
                                    p = p.dropna(subset=["datetime"]).set_index("datetime")
                                elif "timestamp" in p.columns:
                                    p["datetime"] = pd.to_datetime(
                                        p["timestamp"],
                                        errors="coerce",
                                    )
                                    p = p.dropna(subset=["datetime"]).set_index("datetime")
                            parts.append(p)
                        if parts:
                            merged = pd.concat(parts, axis=0)
                            if isinstance(merged.index, pd.DatetimeIndex):
                                merged = merged[~merged.index.duplicated(keep="last")]
                                merged = merged.sort_index()
                            df = merged
                except _PREDICTOR_RECOVERABLE_EXCEPTIONS as e:
                    log.debug("Realtime session-merge skipped for %s: %s", code, e)

            if (
                df is None
                or df.empty
                or len(df) < CONFIG.SEQUENCE_LENGTH
                or len(df) < min_rows
            ):
                return [], []

            # Real-time guess should follow the latest candles window.
            df = df.tail(window).copy()

            if use_realtime_price:
                try:
                    quote = self.fetcher.get_realtime(code)
                    if quote and float(getattr(quote, "price", 0) or 0) > 0:
                        px = float(quote.price)
                        df.loc[df.index[-1], "close"] = px
                        df.loc[df.index[-1], "high"] = max(
                            float(df["high"].iloc[-1]),
                            px,
                        )
                        df.loc[df.index[-1], "low"] = min(
                            float(df["low"].iloc[-1]),
                            px,
                        )
                except _PREDICTOR_RECOVERABLE_EXCEPTIONS as e:
                    log.debug(
                        "Realtime tail merge skipped while building forecast curve for %s: %s",
                        code,
                        e,
                    )

            df = self._sanitize_history_df(df, interval)
            if (
                df is None
                or df.empty
                or len(df) < CONFIG.SEQUENCE_LENGTH
                or len(df) < min_rows
            ):
                return [], []

            actual = df["close"].tail(min(lookback, len(df))).tolist()
            current_price = float(df["close"].iloc[-1])

            scaler_ready = bool(
                self.processor is not None
                and getattr(self.processor, "is_fitted", True)
            )
            if not scaler_ready:
                self._maybe_reload_models(reason="forecast_curve_scaler_missing")
                scaler_ready = bool(
                    self.processor is not None
                    and getattr(self.processor, "is_fitted", True)
                )
            if not scaler_ready:
                log.debug(
                    "Realtime forecast skipped for %s/%s: scaler unavailable",
                    code,
                    interval,
                )
                return actual, []

            df = self.feature_engine.create_features(df)
            X = self.processor.prepare_inference_sequence(
                df, self._feature_cols
            )

            atr_pct = self._get_atr_pct(df)
            news_score, news_conf, news_count = self._get_news_sentiment(
                code,
                interval,
            )
            news_bias = self._compute_news_bias(
                news_score,
                news_conf,
                news_count,
                interval,
            )

            try:
                predicted = self._generate_forecast(
                    X,
                    current_price,
                    horizon,
                    atr_pct,
                    sequence_signature=self._sequence_signature(X),
                    seed_context=f"{code}:{interval}",
                    recent_prices=actual,
                    news_bias=news_bias,
                )
            except TypeError:
                predicted = self._generate_forecast(
                    X,
                    current_price,
                    horizon,
                    atr_pct,
                    sequence_signature=self._sequence_signature(X),
                    seed_context=f"{code}:{interval}",
                    recent_prices=actual,
                )
            
            # FIX Bug #8: Handle None/empty predictions properly
            if not predicted or not isinstance(predicted, (list, tuple)):
                log.debug(
                    "Forecast generated empty prediction for %s/%s, using actuals only",
                    code,
                    interval,
                )
                return actual, []
            
            # Filter out invalid values from prediction
            try:
                predicted = [
                    float(p) for p in predicted
                    if p is not None and np.isfinite(float(p)) and float(p) > 0
                ]
            except (TypeError, ValueError):
                return actual, []
            
            if not predicted:
                return actual, []
            
            predicted = self._stabilize_forecast_curve(
                predicted,
                current_price=current_price,
                atr_pct=atr_pct,
            )

            return actual, predicted

        except _PREDICTOR_RECOVERABLE_EXCEPTIONS as e:
            log.warning(f"Forecast curve failed for {code}: {e}")
            return [], []

def _stabilize_forecast_curve(
    values: list[float],
    *,
    current_price: float,
    atr_pct: float,
) -> list[float]:
    """Clamp/smooth forecast curve so one noisy step cannot create
    unrealistic V-shapes in real-time chart updates.
    """
    if not values:
        return []
    px0 = float(current_price or 0.0)
    if px0 <= 0:
        return [float(v) for v in values if float(v) > 0]

    vol = float(np.nan_to_num(atr_pct, nan=0.02, posinf=0.02, neginf=0.02))
    if vol <= 0:
        vol = 0.02

    # Per-step clamp for intraday visualization stability.
    max_step = float(np.clip(vol * 0.75, 0.003, 0.03))
    prev = px0
    out: list[float] = []
    for raw in values:
        try:
            p = float(raw)
        except (TypeError, ValueError):
            p = prev
        if not np.isfinite(p) or p <= 0:
            p = prev

        lo = prev * (1.0 - max_step)
        hi = prev * (1.0 + max_step)
        p = float(np.clip(p, lo, hi))

        # Mild EMA smoothing to reduce sawtooth artifacts.
        p = float((0.82 * p) + (0.18 * prev))
        out.append(p)
        prev = p
    return out

def get_top_picks(
    self,
    stock_codes: list[str],
    n: int = 10,
    signal_type: str = "buy",
) -> list[Prediction]:
    """Get top N stock picks based on signal type."""
    with self._predict_lock:
        predictions = self.predict_quick_batch(stock_codes)

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

        if not filtered:
            return []

        # Candidate cap keeps screener overlays fast on large universes.
        candidate_cap = max(int(n) * 3, int(n), 10)
        filtered.sort(key=lambda x: x.confidence, reverse=True)
        candidates = filtered[:candidate_cap]

        try:
            from analysis.screener import build_default_screener

            screener = build_default_screener()
            ranked = screener.rank_predictions(
                candidates,
                top_n=max(1, int(n)),
                include_fundamentals=True,
            )
            if ranked:
                return ranked[:n]
        except Exception as e:
            log.debug("Screener overlay unavailable; confidence ranking fallback: %s", e)

        return candidates[:n]

def _apply_ensemble_prediction(self, X: FloatArray, pred: Prediction) -> None:
    """Apply ensemble prediction with bounds checking."""
    ensemble_pred = self.ensemble.predict(X)
    self._apply_ensemble_result(ensemble_pred, pred)

def _apply_ensemble_result(self, ensemble_pred: Any, pred: Prediction) -> None:
    """Apply a precomputed ensemble result to a prediction object."""
    probs = np.asarray(
        getattr(ensemble_pred, "probabilities", [0.33, 0.34, 0.33]),
        dtype=float,
    ).reshape(-1)
    n_classes = len(probs)

    pred.prob_down = float(probs[0]) if n_classes > 0 else 0.33
    pred.prob_neutral = float(probs[1]) if n_classes > 1 else 0.34
    pred.prob_up = float(probs[2]) if n_classes > 2 else 0.33

    pred.prob_down = max(0.0, min(1.0, pred.prob_down))
    pred.prob_neutral = max(0.0, min(1.0, pred.prob_neutral))
    pred.prob_up = max(0.0, min(1.0, pred.prob_up))
    p_sum = pred.prob_down + pred.prob_neutral + pred.prob_up
    if p_sum > 0:
        pred.prob_down /= p_sum
        pred.prob_neutral /= p_sum
        pred.prob_up /= p_sum
    else:
        pred.prob_down, pred.prob_neutral, pred.prob_up = (
            0.33,
            0.34,
            0.33,
        )

    # Guard against collapsed one-hot neutral outputs that can happen when
    # model logits saturate under noisy intraday inputs.
    if (
        float(pred.prob_neutral) >= 0.985
        and float(max(pred.prob_up, pred.prob_down)) <= 0.015
    ):
        trend = str(getattr(pred, "trend", "") or "").upper()
        macd = str(getattr(pred, "macd_signal", "") or "").upper()
        try:
            rsi = float(getattr(pred, "rsi", 50.0) or 50.0)
        except (TypeError, ValueError):
            rsi = 50.0
        if not np.isfinite(rsi):
            rsi = 50.0

        bias = 0.0
        if trend == "UPTREND":
            bias += 0.10
        elif trend == "DOWNTREND":
            bias -= 0.10
        if "BULL" in macd:
            bias += 0.06
        elif "BEAR" in macd:
            bias -= 0.06
        bias += float(np.clip((50.0 - rsi) / 500.0, -0.06, 0.06))
        bias = float(np.clip(bias, -0.20, 0.20))

        neutral_mass = 0.72
        side_mass = 1.0 - neutral_mass
        up = float(np.clip((side_mass * 0.5) + (side_mass * bias), 0.08, side_mass - 0.08))
        down = float(max(0.08, side_mass - up))
        norm = float(max(1e-8, up + down + neutral_mass))
        pred.prob_up = float(up / norm)
        pred.prob_down = float(down / norm)
        pred.prob_neutral = float(neutral_mass / norm)
        _append_warning_once(
            pred,
            "Extreme-neutral probability output was rebalanced for stability",
        )

    pred.confidence = float(
        max(0.0, min(1.0, getattr(ensemble_pred, "confidence", 0.0)))
    )
    pred.raw_confidence = float(
        max(
            0.0,
            min(
                1.0,
                getattr(
                    ensemble_pred,
                    "raw_confidence",
                    getattr(ensemble_pred, "confidence", 0.0),
                ),
            ),
        )
    )

    pred.model_agreement = float(
        getattr(ensemble_pred, "agreement", 1.0)
    )
    pred.entropy = float(
        getattr(ensemble_pred, "entropy", 0.0)
    )
    pred.model_margin = float(
        getattr(ensemble_pred, "margin", 0.10)
    )
    pred.brier_score = float(
        max(0.0, getattr(ensemble_pred, "brier_score", 0.0))
    )

    pred.signal = self._determine_signal(ensemble_pred, pred)
    pred.signal_strength = self._calculate_signal_strength(
        ensemble_pred, pred
    )
    self._refresh_prediction_uncertainty(pred)
    self._apply_high_precision_gate(pred)
    self._apply_runtime_signal_quality_gate(pred)
    self._apply_tail_risk_guard(pred)
    
    # FIX: Ensure probabilities are always valid even if ensemble failed
    if not hasattr(pred, 'prob_up') or pred.prob_up is None:
        pred.prob_up = 0.33
    if not hasattr(pred, 'prob_down') or pred.prob_down is None:
        pred.prob_down = 0.33
    if not hasattr(pred, 'prob_neutral') or pred.prob_neutral is None:
        pred.prob_neutral = 0.34

def _append_warning_once(pred: Prediction, message: str) -> None:
    """Append warning only once to avoid noisy duplicates."""
    text = str(message).strip()
    if not text:
        return
    existing = [str(x) for x in list(pred.warnings or [])]
    if text in existing:
        return
    pred.warnings.append(text)

def _refresh_prediction_uncertainty(self, pred: Prediction) -> None:
    """Derive uncertainty and tail-risk from signal quality metrics.

    Also moderates confidence when entropy/adverse-risk is high to avoid
    over-confident chart narratives.
    """
    conf = float(np.clip(getattr(pred, "confidence", 0.0), 0.0, 1.0))
    incoming_conf = conf
    raw_conf = float(np.clip(getattr(pred, "raw_confidence", conf), 0.0, 1.0))
    agreement = float(np.clip(getattr(pred, "model_agreement", 1.0), 0.0, 1.0))
    entropy = float(np.clip(getattr(pred, "entropy", 0.0), 0.0, 1.0))
    margin = float(np.clip(getattr(pred, "model_margin", 0.10), 0.0, 1.0))
    edge = float(abs(float(pred.prob_up) - float(pred.prob_down)))

    atr = float(np.nan_to_num(getattr(pred, "atr_pct_value", 0.02), nan=0.02))
    atr = float(np.clip(atr, 0.003, 0.12))
    vol_scale = float(np.clip(atr / 0.030, 0.0, 2.0))

    if pred.signal in (Signal.BUY, Signal.STRONG_BUY):
        adverse_prob = float(np.clip(pred.prob_down, 0.0, 1.0))
    elif pred.signal in (Signal.SELL, Signal.STRONG_SELL):
        adverse_prob = float(np.clip(pred.prob_up, 0.0, 1.0))
    else:
        adverse_prob = float(
            np.clip(max(pred.prob_up, pred.prob_down), 0.0, 1.0)
        )

    quality = (
        (0.46 * conf)
        + (0.22 * agreement)
        + (0.20 * (1.0 - entropy))
        + (0.12 * margin)
    )
    uncertainty = float(
        np.clip(
            (1.0 - quality)
            + (0.18 * (1.0 - edge))
            + (0.14 * vol_scale),
            0.0,
            1.0,
        )
    )
    tail_risk = float(
        np.clip(
            (0.58 * adverse_prob)
            + (0.24 * entropy)
            + (0.18 * max(0.0, vol_scale - 1.0)),
            0.0,
            1.0,
        )
    )

    # Confidence moderation only when risk is materially elevated.
    penalty = (
        (max(0.0, entropy - 0.62) * 0.35)
        + (max(0.0, tail_risk - 0.58) * 0.30)
        + (max(0.0, 0.55 - agreement) * 0.28)
    )
    if margin < 0.04:
        penalty += 0.05
    if penalty > 0.0:
        # Cap this stage's reduction so confidence is not over-penalized
        # after upstream calibration/regime adjustments.
        stage_cap = float(np.clip(incoming_conf * 0.18, 0.04, 0.10))
        reduction = float(min(penalty, stage_cap))
        old_conf = conf
        conf = float(np.clip(incoming_conf - reduction, 0.0, 1.0))
        pred.confidence = conf
        if (old_conf - conf) >= 0.08:
            self._append_warning_once(
                pred,
                "Confidence moderated due to uncertainty/tail-risk conditions",
            )

    if raw_conf > 0 and conf > raw_conf:
        pred.confidence = raw_conf

    pred.uncertainty_score = float(uncertainty)
    pred.tail_risk_score = float(tail_risk)

def _apply_tail_risk_guard(self, pred: Prediction) -> None:
    """Block actionable signals when adverse-tail probability is too high."""
    if pred.signal == Signal.HOLD:
        return
    conf = float(np.clip(getattr(pred, "confidence", 0.0), 0.0, 1.0))
    tail_risk = float(np.clip(getattr(pred, "tail_risk_score", 0.0), 0.0, 1.0))
    uncertainty = float(
        np.clip(getattr(pred, "uncertainty_score", 0.0), 0.0, 1.0)
    )

    reasons: list[str] = []
    if tail_risk >= 0.72 and conf < 0.88:
        reasons.append(f"tail_risk {tail_risk:.2f}")
    if uncertainty >= 0.82 and conf < 0.86:
        reasons.append(f"uncertainty {uncertainty:.2f}")

    if not reasons:
        return

    old_signal = pred.signal.value
    pred.signal = Signal.HOLD
    pred.signal_strength = min(float(pred.signal_strength), 0.49)
    self._append_warning_once(
        pred,
        "Tail-risk guard filtered signal "
        f"{old_signal} -> HOLD ({'; '.join(reasons[:2])})",
    )

def _build_prediction_bands(self, pred: Prediction) -> None:
    """Build per-step prediction intervals to visualize uncertainty."""
    values = [
        float(v)
        for v in list(getattr(pred, "predicted_prices", []) or [])
        if float(v) > 0 and np.isfinite(float(v))
    ]
    if not values:
        pred.predicted_prices_low = []
        pred.predicted_prices_high = []
        return

    uncertainty = float(
        np.clip(getattr(pred, "uncertainty_score", 0.5), 0.0, 1.0)
    )
    tail_risk = float(np.clip(getattr(pred, "tail_risk_score", 0.5), 0.0, 1.0))
    conf = float(np.clip(getattr(pred, "confidence", 0.0), 0.0, 1.0))
    atr = float(
        np.clip(
            np.nan_to_num(getattr(pred, "atr_pct_value", 0.02), nan=0.02),
            0.003,
            0.12,
        )
    )
    n = max(1, len(values))

    base_width = float(
        np.clip(
            atr
            * (0.60 + (1.10 * uncertainty) + (0.75 * tail_risk))
            * (1.05 + (0.35 * (1.0 - conf))),
            0.004,
            0.30,
        )
    )

    lows: list[float] = []
    highs: list[float] = []
    for i, px in enumerate(values, start=1):
        growth = 1.0 + (float(i) / float(n)) * (0.85 + (0.65 * uncertainty))
        width = float(np.clip(base_width * growth, 0.004, 0.35))
        lo = max(0.01, float(px) * (1.0 - width))
        hi = max(lo + 1e-6, float(px) * (1.0 + width))
        lows.append(float(lo))
        highs.append(float(hi))

    pred.predicted_prices_low = lows
    pred.predicted_prices_high = highs

def _apply_high_precision_gate(self, pred: Prediction) -> None:
    """Optionally downgrade weak actionable predictions to HOLD."""
    cfg = self._high_precision
    if not cfg or cfg.get("enabled", 0.0) <= 0:
        return
    if pred.signal == Signal.HOLD:
        return

    reasons: list[str] = []

    # Regime-aware confidence floor: range/high-vol require stronger evidence.
    required_conf = float(cfg["min_confidence"])
    if cfg.get("regime_routing", 0.0) > 0:
        if str(pred.trend).upper() == "SIDEWAYS":
            required_conf += float(cfg.get("range_conf_boost", 0.0))
        if float(pred.atr_pct_value) >= float(cfg.get("high_vol_atr_pct", 0.035)):
            required_conf += float(cfg.get("high_vol_conf_boost", 0.0))

    if pred.confidence < required_conf:
        reasons.append(
            f"confidence {pred.confidence:.2f} < {required_conf:.2f}"
        )
    if pred.model_agreement < cfg["min_agreement"]:
        reasons.append(
            f"agreement {pred.model_agreement:.2f} < {cfg['min_agreement']:.2f}"
        )
    if pred.entropy > cfg["max_entropy"]:
        reasons.append(
            f"entropy {pred.entropy:.2f} > {cfg['max_entropy']:.2f}"
        )
    edge = abs(float(pred.prob_up) - float(pred.prob_down))
    if edge < cfg["min_edge"]:
        reasons.append(f"edge {edge:.2f} < {cfg['min_edge']:.2f}")

    if not reasons:
        return

    old_signal = pred.signal.value
    pred.signal = Signal.HOLD
    pred.signal_strength = min(float(pred.signal_strength), 0.49)
    pred.warnings.append(
        "High Precision Mode filtered signal "
        f"{old_signal} -> HOLD ({'; '.join(reasons[:3])})"
    )

def _apply_runtime_signal_quality_gate(self, pred: Prediction) -> None:
    """Always-on runtime guard to reduce low-quality actionable signals.
    This improves precision by preferring HOLD when edge quality is weak.
    """
    if pred.signal == Signal.HOLD:
        return

    reasons: list[str] = []
    conf = float(np.clip(pred.confidence, 0.0, 1.0))
    agreement = float(np.clip(pred.model_agreement, 0.0, 1.0))
    entropy = float(np.clip(pred.entropy, 0.0, 1.0))
    edge = float(pred.prob_up) - float(pred.prob_down)
    trend = str(pred.trend).upper()

    if pred.signal in (Signal.BUY, Signal.STRONG_BUY) and edge < 0.03:
        reasons.append(f"edge {edge:.2f} too weak for long")
    if pred.signal in (Signal.SELL, Signal.STRONG_SELL) and edge > -0.03:
        reasons.append(f"edge {edge:.2f} too weak for short")
    if agreement < 0.50 and conf < 0.78:
        reasons.append(
            f"agreement/conf weak ({agreement:.2f}/{conf:.2f})"
        )
    if entropy > 0.78 and conf < 0.80:
        reasons.append(f"high entropy {entropy:.2f}")
    if trend == "SIDEWAYS" and conf < 0.72:
        reasons.append("sideways regime with low confidence")
    if (
        trend == "UPTREND"
        and pred.signal in (Signal.SELL, Signal.STRONG_SELL)
        and conf < 0.86
    ):
        reasons.append("counter-trend short lacks conviction")
    if (
        trend == "DOWNTREND"
        and pred.signal in (Signal.BUY, Signal.STRONG_BUY)
        and conf < 0.86
    ):
        reasons.append("counter-trend long lacks conviction")
    if pred.atr_pct_value >= 0.04 and conf < 0.76:
        reasons.append("high volatility requires stronger confidence")

    if not reasons:
        return

    old_signal = pred.signal.value
    pred.signal = Signal.HOLD
    pred.signal_strength = min(float(pred.signal_strength), 0.49)
    pred.warnings.append(
        "Runtime quality gate filtered signal "
        f"{old_signal} -> HOLD ({'; '.join(reasons[:3])})"
    )

def _get_cache_ttl(self, use_realtime: bool, interval: str, volatility: float = None) -> float:
    """Adaptive cache TTL.
    
    FIX #CACHE-001: TTL now adapts to market volatility - higher volatility
    means shorter cache life to reduce stale predictions.
    
    FIX #CACHE-002: Added cache invalidation hook for model reload events.
    
    Args:
        use_realtime: Whether using real-time price
        interval: Data interval
        volatility: Optional ATR-based volatility measure
        
    Returns:
        Cache TTL in seconds
    """
    base = float(self._CACHE_TTL)
    
    if not use_realtime:
        # Non-realtime: use base TTL with small volatility adjustment
        if volatility is not None and volatility > 0.03:  # High volatility
            return float(max(1.0, base * 0.7))  # Reduce TTL by 30%
        return base
    
    # Real-time paths: more aggressive TTL reduction
    intraday = str(interval).lower() in {"1m", "3m", "5m", "15m", "30m", "60m"}
    
    # Base realtime TTL
    if intraday:
        ttl = float(max(0.2, min(base, self._CACHE_TTL_REALTIME)))
    else:
        ttl = float(max(0.5, min(base, 2.0)))
    
    # FIX #CACHE-001: Volatility adjustment
    if volatility is not None and volatility > 0:
        if volatility > 0.05:  # Very high volatility
            ttl = max(0.15, ttl * 0.4)  # 60% reduction
        elif volatility > 0.03:  # High volatility
            ttl = max(0.2, ttl * 0.6)  # 40% reduction
        elif volatility > 0.02:  # Moderate volatility
            ttl = max(0.3, ttl * 0.8)  # 20% reduction
    
    return float(ttl)

def _get_cache_version(self) -> str:
    """Get current model cache version.
    
    FIX CACHE INVALIDATION: Version changes when models are reloaded,
    ensuring stale predictions aren't served after model updates.
    """
    # Use model artifact signature as version.
    lock = getattr(self, "_cache_lock", None)
    if lock is None:
        return str(getattr(self, "_model_artifact_sig", "v1"))
    with lock:
        return str(getattr(self, "_model_artifact_sig", "v1"))

def _get_cached_prediction(
    self,
    cache_key: str,
    ttl: float | None = None,
) -> Prediction | None:
    """Get cached prediction if still valid.

    FIX CACHE INVALIDATION: Includes version check to invalidate
    all cached predictions when models are reloaded.
    
    FIX #CACHE-001: Supports per-entry TTL stored in cache entry.
    
    Args:
        cache_key: Cache key
        ttl: Default TTL if entry doesn't have its own TTL
        
    Returns:
        Cached prediction or None if expired/missing
    """
    default_ttl = float(self._CACHE_TTL if ttl is None else ttl)
    cache_version = self._get_cache_version()
    pred_to_copy: Prediction | None = None

    with self._cache_lock:
        entry = self._pred_cache.get(cache_key)
        if entry is not None:
            # New format: (timestamp, prediction, cache_version, ttl)
            if len(entry) >= 4:
                ts, pred, version, entry_ttl = entry
                use_ttl = entry_ttl if entry_ttl is not None else default_ttl
                # Invalidate if version mismatch (model was reloaded)
                if version != cache_version:
                    del self._pred_cache[cache_key]
                    return None
                # Check TTL
                if (time.time() - ts) < use_ttl:
                    pred_to_copy = pred
                else:
                    del self._pred_cache[cache_key]
            elif len(entry) >= 3:
                # Old format without per-entry TTL: (timestamp, prediction, version)
                ts, pred, version = entry
                if version != cache_version:
                    del self._pred_cache[cache_key]
                    return None
                if (time.time() - ts) < default_ttl:
                    pred_to_copy = pred
                else:
                    del self._pred_cache[cache_key]
            else:
                # Very old format without version - invalidate
                del self._pred_cache[cache_key]
                
    if pred_to_copy is None:
        return None
    return copy.deepcopy(pred_to_copy)

def _set_cached_prediction(
    self,
    cache_key: str,
    pred: Prediction,
    ttl: float | None = None,
) -> None:
    """Cache a prediction result with bounded size.

    FIX CACHE INVALIDATION: Stores model version with each prediction
    to enable bulk invalidation on model reload.
    
    FIX #CACHE-001: Added optional TTL parameter for per-entry TTL override.
    
    Args:
        cache_key: Cache key
        pred: Prediction to cache
        ttl: Optional TTL override (uses default if None)
    """
    cache_version = self._get_cache_version()
    pred_copy = copy.deepcopy(pred)

    with self._cache_lock:
        # Store version with prediction for invalidation checking.
        # Keep legacy 3-tuple format when no per-entry TTL override is used.
        if ttl is None:
            self._pred_cache[cache_key] = (
                time.time(),
                pred_copy,
                cache_version,
            )
        else:
            self._pred_cache[cache_key] = (
                time.time(),
                pred_copy,
                cache_version,
                ttl,  # Store per-entry TTL
            )

        if len(self._pred_cache) > self._MAX_CACHE_SIZE:
            now = time.time()
            expired: list[str] = []
            for k, entry in self._pred_cache.items():
                if len(entry) >= 4:
                    ts, _, _, entry_ttl = entry
                    use_ttl = entry_ttl if entry_ttl is not None else self._CACHE_TTL
                elif len(entry) >= 1:
                    ts = entry[0]
                    use_ttl = self._CACHE_TTL
                else:
                    expired.append(k)
                    continue
                    
                if (now - ts) > use_ttl:
                    expired.append(k)
            
            for k in expired:
                del self._pred_cache[k]

            # If still too large, evict oldest
            if len(self._pred_cache) > self._MAX_CACHE_SIZE:
                sorted_keys = sorted(
                    self._pred_cache.keys(),
                    key=lambda k: self._pred_cache[k][0] if len(self._pred_cache[k]) >= 1 else 0
                )
                for k in sorted_keys[:len(sorted_keys) // 2]:
                    del self._pred_cache[k]

def _news_cache_ttl(self, interval: str) -> float:
    """News sentiment cache TTL by interval profile."""
    if self._is_intraday_interval(interval):
        return float(self._NEWS_CACHE_TTL_INTRADAY)
    return float(self._NEWS_CACHE_TTL_SWING)

def _ensure_news_cache_state(self) -> None:
    """Lazy-init news cache fields for tests using Predictor.__new__()."""
    if not hasattr(self, "_news_cache") or self._news_cache is None:
        self._news_cache = {}
    if not hasattr(self, "_news_cache_lock") or self._news_cache_lock is None:
        self._news_cache_lock = threading.Lock()

def _get_news_sentiment(
    self,
    stock_code: str,
    interval: str,
) -> tuple[float, float, int]:
    """Return (sentiment, confidence, count) for stock news.
    Sentiment is in [-1, 1], confidence in [0, 1].
    """
    self._ensure_news_cache_state()
    code = self._clean_code(stock_code)
    if not code:
        return 0.0, 0.0, 0

    ttl = self._news_cache_ttl(interval)
    now = time.time()
    with self._news_cache_lock:
        rec = self._news_cache.get(code)
        if rec is not None:
            ts, s, conf, cnt = rec
            if (now - float(ts)) < ttl:
                return float(s), float(conf), int(cnt)

    try:
        from data.news import get_news_aggregator

        agg = get_news_aggregator()
        summary = agg.get_sentiment_summary(code)
        score = float(summary.get("overall_sentiment", 0.0) or 0.0)
        conf = float(summary.get("confidence", 0.0) or 0.0)
        cnt = int(summary.get("total", 0) or 0)

        score = float(np.clip(score, -1.0, 1.0))
        conf = float(np.clip(conf, 0.0, 1.0))
        cnt = max(0, int(cnt))

    except _PREDICTOR_RECOVERABLE_EXCEPTIONS as e:
        log.debug("News sentiment lookup failed for %s: %s", code, e)
        score, conf, cnt = 0.0, 0.0, 0

    with self._news_cache_lock:
        self._news_cache[code] = (now, float(score), float(conf), int(cnt))
        if len(self._news_cache) > 500:
            oldest = sorted(
                self._news_cache.items(),
                key=lambda kv: float(kv[1][0]),
            )[:180]
            for k, _ in oldest:
                self._news_cache.pop(k, None)

    return float(score), float(conf), int(cnt)

def _compute_news_bias(
    self,
    sentiment: float,
    confidence: float,
    count: int,
    interval: str,
) -> float:
    """Convert news metrics into a bounded directional bias.
    Positive => bullish tilt, negative => bearish tilt.
    """
    s = float(np.clip(np.nan_to_num(sentiment, nan=0.0), -1.0, 1.0))
    conf = float(np.clip(np.nan_to_num(confidence, nan=0.0), 0.0, 1.0))
    cnt = max(0, int(count))

    coverage = float(np.clip(cnt / 24.0, 0.0, 1.0))
    eff_conf = conf * (0.30 + (0.70 * coverage))
    raw = s * eff_conf
    cap = 0.14 if self._is_intraday_interval(interval) else 0.20
    return float(np.clip(raw, -cap, cap))

def _apply_news_influence(
    self,
    pred: Prediction,
    stock_code: str,
    interval: str,
) -> float:
    """Blend news sentiment into class probabilities and confidence.
    Returns the directional bias used for forecast shaping.
    """
    sentiment, conf, count = self._get_news_sentiment(stock_code, interval)
    pred.news_sentiment = float(sentiment)
    pred.news_confidence = float(conf)
    pred.news_count = int(count)

    news_bias = self._compute_news_bias(sentiment, conf, count, interval)
    if abs(news_bias) <= 1e-8:
        return 0.0

    shift = float(min(0.18, abs(news_bias) * 0.55))
    if news_bias > 0:
        moved = min(float(pred.prob_down), shift)
        pred.prob_down = float(pred.prob_down - moved)
        pred.prob_up = float(pred.prob_up + moved)
    else:
        moved = min(float(pred.prob_up), shift)
        pred.prob_up = float(pred.prob_up - moved)
        pred.prob_down = float(pred.prob_down + moved)

    # Keep probabilities normalized.
    pred.prob_down = float(np.clip(pred.prob_down, 0.0, 1.0))
    pred.prob_neutral = float(np.clip(pred.prob_neutral, 0.0, 1.0))
    pred.prob_up = float(np.clip(pred.prob_up, 0.0, 1.0))
    p_sum = float(pred.prob_down + pred.prob_neutral + pred.prob_up)
    # FIX: Use epsilon threshold to avoid division by very small numbers
    if p_sum <= 1e-10:
        pred.prob_down, pred.prob_neutral, pred.prob_up = 0.33, 0.34, 0.33
    else:
        pred.prob_down /= p_sum
        pred.prob_neutral /= p_sum
        pred.prob_up /= p_sum

    edge = float(pred.prob_up - pred.prob_down)
    aligned = (edge == 0.0) or ((edge > 0) == (news_bias > 0))
    conf_delta = min(0.10, abs(news_bias) * (0.35 if aligned else 0.18))
    if aligned:
        pred.confidence = float(np.clip(pred.confidence + conf_delta, 0.0, 1.0))
    else:
        pred.confidence = float(np.clip(pred.confidence - (conf_delta * 0.6), 0.0, 1.0))

    # News can upgrade HOLD when the post-blend edge is meaningful.
    if pred.signal == Signal.HOLD and pred.confidence >= 0.56:
        if edge >= 0.08:
            pred.signal = Signal.BUY
        elif edge <= -0.08:
            pred.signal = Signal.SELL

    # News can also dampen contradictory directional signals.
    if pred.signal in (Signal.BUY, Signal.STRONG_BUY) and edge < 0:
        pred.signal = Signal.HOLD
    elif pred.signal in (Signal.SELL, Signal.STRONG_SELL) and edge > 0:
        pred.signal = Signal.HOLD

    if count > 0:
        direction = "bullish" if news_bias > 0 else "bearish"
        msg = (
            f"News sentiment tilt: {direction} "
            f"({sentiment:+.2f}, conf {conf:.2f}, n={count})"
        )
        if msg not in pred.reasons:
            pred.reasons.append(msg)

    return float(news_bias)
