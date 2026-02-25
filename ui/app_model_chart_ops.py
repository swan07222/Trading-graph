from __future__ import annotations

import math
import time
from statistics import median
from typing import Any

import numpy as np

from config.settings import CONFIG
from utils.logger import get_logger
from utils.recoverable import COMMON_RECOVERABLE_EXCEPTIONS

log = get_logger(__name__)
_UI_RECOVERABLE_EXCEPTIONS = COMMON_RECOVERABLE_EXCEPTIONS

def _sync_ui_to_loaded_model(
    self,
    requested_interval: str | None = None,
    requested_horizon: int | None = None,
    preserve_requested_interval: bool = False,
) -> tuple[str, int]:
    """Align UI controls to the actual loaded model metadata.
    Prevents 'UI says 1m/120 while model runs 1d/5' mismatches.
    """
    if not self.predictor:
        iv = str(requested_interval or self.interval_combo.currentText()).strip().lower()
        h = int(requested_horizon or self.forecast_spin.value())
        return iv, h

    model_iv_raw = str(
        getattr(
            self.predictor,
            "_loaded_model_interval",
            getattr(
                self.predictor,
                "interval",
                requested_interval or self.interval_combo.currentText(),
            ),
        )
    ).strip().lower()
    model_iv = self._model_interval_to_ui_token(model_iv_raw)

    items = [
        str(self.interval_combo.itemText(i)).strip().lower()
        for i in range(self.interval_combo.count())
    ]
    ui_iv = model_iv
    if preserve_requested_interval and requested_interval is not None:
        ui_iv = self._model_interval_to_ui_token(str(requested_interval).strip().lower())
    if ui_iv not in items:
        ui_iv = str(requested_interval or self.interval_combo.currentText()).strip().lower()

    try:
        model_h_raw = int(
            getattr(
                self.predictor,
                "_loaded_model_horizon",
                getattr(
                    self.predictor,
                    "horizon",
                    requested_horizon
                    if requested_horizon is not None
                    else self.forecast_spin.value(),
                ),
            )
        )
    except _UI_RECOVERABLE_EXCEPTIONS:
        model_h_raw = int(self.forecast_spin.value())

    model_h = max(
        int(self.forecast_spin.minimum()),
        min(int(self.forecast_spin.maximum()), int(model_h_raw)),
    )
    ui_h = model_h
    if preserve_requested_interval and requested_horizon is not None:
        ui_h = max(
            int(self.forecast_spin.minimum()),
            min(int(self.forecast_spin.maximum()), int(requested_horizon)),
        )

    self.interval_combo.blockSignals(True)
    try:
        self.interval_combo.setCurrentText(ui_iv)
    finally:
        self.interval_combo.blockSignals(False)
    self.forecast_spin.setValue(ui_h)
    if ui_iv != model_iv or ui_h != model_h:
        self.model_info.setText(
            f"Interval: {ui_iv}, Horizon: {ui_h} (model: {model_iv}/{model_h})"
        )
    else:
        self.model_info.setText(f"Interval: {ui_iv}, Horizon: {ui_h}")

    if (
        (not preserve_requested_interval)
        and requested_interval is not None
        and requested_horizon is not None
    ):
        req_iv = str(requested_interval).strip().lower()
        req_h = int(requested_horizon)
        if req_iv != ui_iv or req_h != ui_h:
            self.log(
                f"Loaded model metadata applied: requested {req_iv}/{req_h} -> active {ui_iv}/{ui_h}",
                "warning",
            )

    return ui_iv, ui_h

def _loaded_model_ui_meta(self) -> tuple[str, int]:
    """Return loaded model metadata as (ui_interval_token, horizon)."""
    if not self.predictor:
        iv = self._normalize_interval_token(self.interval_combo.currentText())
        hz = int(self.forecast_spin.value())
        return iv, hz
    raw_iv = str(
        getattr(
            self.predictor,
            "_loaded_model_interval",
            getattr(self.predictor, "interval", self.interval_combo.currentText()),
        )
    ).strip().lower()
    iv = self._model_interval_to_ui_token(raw_iv)
    try:
        hz = int(
            getattr(
                self.predictor,
                "_loaded_model_horizon",
                getattr(self.predictor, "horizon", self.forecast_spin.value()),
            )
        )
    except _UI_RECOVERABLE_EXCEPTIONS:
        hz = int(self.forecast_spin.value())
    return iv, max(1, hz)

def _has_exact_model_artifacts(self, interval: str, horizon: int) -> bool:
    """Whether exact ensemble+scaler artifacts exist for interval+horizon."""
    iv = self._normalize_interval_token(interval)
    try:
        hz = max(1, int(horizon))
    except _UI_RECOVERABLE_EXCEPTIONS:
        hz = int(self.forecast_spin.value())
    model_dirs = [CONFIG.MODEL_DIR]
    legacy_dir = CONFIG.BASE_DIR / "models_saved"
    if legacy_dir not in model_dirs:
        model_dirs.append(legacy_dir)
    for model_dir in model_dirs:
        ens = model_dir / f"ensemble_{iv}_{hz}.pt"
        scl = model_dir / f"scaler_{iv}_{hz}.pkl"
        if ens.exists() and scl.exists():
            return True
    return False

def _log_model_alignment_debug(
    self,
    *,
    context: str,
    requested_interval: str | None = None,
    requested_horizon: int | None = None,
) -> None:
    """Verbose model/UI alignment diagnostics for bug hunting."""
    ui_iv = self._normalize_interval_token(
        requested_interval if requested_interval is not None else self.interval_combo.currentText()
    )
    try:
        ui_h = int(requested_horizon if requested_horizon is not None else self.forecast_spin.value())
    except _UI_RECOVERABLE_EXCEPTIONS:
        ui_h = int(self.forecast_spin.value())
    model_iv, model_h = self._loaded_model_ui_meta()
    exact = self._has_exact_model_artifacts(ui_iv, ui_h)
    min_gap = 8.0 if str(context).strip().lower() == "analyze" else 0.8
    self._debug_console(
        f"model_align:{context}:{ui_iv}:{ui_h}",
        (
            f"model alignment [{context}] ui={ui_iv}/{ui_h} "
            f"loaded={model_iv}/{model_h} exact_artifacts={int(exact)}"
        ),
        min_gap_seconds=min_gap,
        level="info",
    )

def _debug_chart_state(
    self,
    *,
    symbol: str,
    interval: str,
    bars: list[dict[str, Any]] | None,
    predicted_prices: list[float] | None = None,
    context: str = "chart",
) -> None:
    """Emit compact chart diagnostics for rapid bug triage."""
    arr = self._safe_list(bars)
    preds = self._safe_list(predicted_prices)
    if not arr:
        self._debug_console(
            f"chart_state:{context}:{self._ui_norm(symbol)}:{self._normalize_interval_token(interval)}",
            (
                f"chart state [{context}] symbol={self._ui_norm(symbol)} "
                f"iv={self._normalize_interval_token(interval)} bars=0 preds={len(preds)}"
            ),
            min_gap_seconds=1.0,
            level="info",
        )
        return

    closes: list[float] = []
    max_span_pct = 0.0
    for row in arr[-min(220, len(arr)):]:
        try:
            c = float(row.get("close", 0) or 0)
            h = float(row.get("high", c) or c)
            low = float(row.get("low", c) or c)
        except _UI_RECOVERABLE_EXCEPTIONS:
            continue
        if c > 0 and math.isfinite(c):
            closes.append(c)
            span = abs(h - low) / max(c, 1e-8)
            if math.isfinite(span):
                max_span_pct = max(max_span_pct, float(span))

    last_close = float(closes[-1]) if closes else 0.0
    med_close = float(median(closes)) if closes else 0.0
    model_iv, model_h = self._loaded_model_ui_meta()
    iv = self._normalize_interval_token(interval)
    sym = self._ui_norm(symbol)
    self._debug_console(
        f"chart_state:{context}:{sym}:{iv}",
        (
            f"chart state [{context}] symbol={sym} iv={iv} bars={len(arr)} "
            f"preds={len(preds)} last={last_close:.4f} med={med_close:.4f} "
            f"max_span={max_span_pct:.2%} model={model_iv}/{model_h}"
        ),
        min_gap_seconds=1.0,
        level="info",
    )
    if max_span_pct > 0.035:
        self._debug_console(
            f"chart_state_anom:{context}:{sym}:{iv}",
            (
                f"chart anomaly [{context}] symbol={sym} iv={iv} "
                f"max_span={max_span_pct:.2%} bars={len(arr)} preds={len(preds)}"
            ),
            min_gap_seconds=0.6,
            level="warning",
        )

def _debug_candle_quality(
    self,
    *,
    symbol: str,
    interval: str,
    bars: list[dict[str, Any]] | None,
    context: str,
) -> None:
    """Detailed candle-shape diagnostics for malformed chart bars."""
    if not bool(getattr(self, "_debug_console_enabled", False)):
        return

    arr = self._safe_list(bars)
    if not arr:
        return

    iv = self._normalize_interval_token(interval)
    jump_cap, range_cap = self._bar_safety_caps(iv)
    intraday = iv not in ("1d", "1wk", "1mo")
    if intraday:
        body_cap = float(max(range_cap * 1.35, 0.007))
        span_cap = float(max(range_cap * 2.10, 0.012))
        wick_cap = float(max(range_cap * 1.55, 0.008))
    else:
        body_cap = float(max(range_cap * 0.85, 0.045))
        span_cap = float(max(range_cap * 1.75, 0.45))
        wick_cap = float(max(range_cap * 1.25, 0.25))

    parsed = 0
    invalid = 0
    duplicates = 0
    doji_like = 0
    extreme = 0
    max_body = 0.0
    max_span = 0.0
    max_wick = 0.0
    max_jump = 0.0
    seen_ts: set[int] = set()
    prev_close: float | None = None

    for row in arr[-min(520, len(arr)):]:
        try:
            o = float(row.get("open", 0) or 0)
            h = float(row.get("high", 0) or 0)
            low = float(row.get("low", 0) or 0)
            c = float(row.get("close", 0) or 0)
        except _UI_RECOVERABLE_EXCEPTIONS:
            invalid += 1
            continue
        if (
            c <= 0
            or not all(math.isfinite(v) for v in (o, h, low, c))
        ):
            invalid += 1
            continue
        parsed += 1

        ts_epoch = int(
            self._bar_bucket_epoch(
                row.get("_ts_epoch", row.get("timestamp")),
                iv,
            )
        )
        if ts_epoch in seen_ts:
            duplicates += 1
        else:
            seen_ts.add(ts_epoch)

        ref = float(prev_close if prev_close and prev_close > 0 else c)
        if ref <= 0:
            ref = float(c)
        body = abs(o - c) / max(ref, 1e-8)
        span = abs(h - low) / max(ref, 1e-8)
        uw = max(0.0, h - max(o, c)) / max(ref, 1e-8)
        lw = max(0.0, min(o, c) - low) / max(ref, 1e-8)
        max_body = max(max_body, float(body))
        max_span = max(max_span, float(span))
        max_wick = max(max_wick, float(max(uw, lw)))
        if prev_close and prev_close > 0:
            jump = abs(c / max(prev_close, 1e-8) - 1.0)
            max_jump = max(max_jump, float(jump))

        if body <= 0.00012 and span <= 0.00120:
            doji_like += 1
        if (
            body > body_cap
            or span > span_cap
            or uw > wick_cap
            or lw > wick_cap
        ):
            extreme += 1

        prev_close = float(c)

    if parsed <= 0:
        self._debug_console(
            f"candle_q:{context}:{self._ui_norm(symbol)}:{iv}",
            (
                f"candle quality [{context}] symbol={self._ui_norm(symbol)} "
                f"iv={iv} parsed=0 invalid={invalid}"
            ),
            min_gap_seconds=1.5,
            level="warning",
        )
        return

    doji_ratio = float(doji_like) / float(max(1, parsed))
    extreme_ratio = float(extreme) / float(max(1, parsed))
    self._debug_console(
        f"candle_q:{context}:{self._ui_norm(symbol)}:{iv}",
        (
            f"candle quality [{context}] symbol={self._ui_norm(symbol)} "
            f"iv={iv} bars={parsed} invalid={invalid} dup={duplicates} "
            f"doji={doji_ratio:.1%} extreme={extreme_ratio:.1%} "
            f"max_body={max_body:.2%} max_span={max_span:.2%} "
            f"max_wick={max_wick:.2%} max_jump={max_jump:.2%} "
            f"caps(body={body_cap:.2%},span={span_cap:.2%},wick={wick_cap:.2%},jump={jump_cap:.2%})"
        ),
        min_gap_seconds=1.5,
        level="info",
    )
    if duplicates > 0 or extreme > 0 or doji_ratio >= 0.86:
        self._debug_console(
            f"candle_q_warn:{context}:{self._ui_norm(symbol)}:{iv}",
            (
                f"candle anomaly [{context}] symbol={self._ui_norm(symbol)} "
                f"iv={iv} dup={duplicates} extreme={extreme}/{parsed} "
                f"doji={doji_like}/{parsed}"
            ),
            min_gap_seconds=0.8,
            level="warning",
        )

def _debug_forecast_quality(
    self,
    *,
    symbol: str,
    chart_interval: str,
    source_interval: str,
    predicted_prices: list[float] | None,
    anchor_price: float | None,
    context: str,
) -> None:
    """Detailed forecast-shape diagnostics for flat/erratic guessed graph."""
    if not bool(getattr(self, "_debug_console_enabled", False)):
        return

    iv_chart = self._normalize_interval_token(chart_interval)
    iv_src = self._normalize_interval_token(source_interval, fallback=iv_chart)
    vals: list[float] = []
    for v in self._safe_list(predicted_prices):
        try:
            fv = float(v)
        except _UI_RECOVERABLE_EXCEPTIONS:
            continue
        if fv > 0 and math.isfinite(fv):
            vals.append(fv)

    sym = self._ui_norm(symbol)
    if not vals:
        self._debug_console(
            f"forecast_q:{context}:{sym}:{iv_chart}",
            (
                f"forecast quality [{context}] symbol={sym} chart={iv_chart} "
                f"source={iv_src} points=0"
            ),
            min_gap_seconds=1.5,
            level="warning",
        )
        return

    try:
        anchor = float(anchor_price or 0.0)
    except _UI_RECOVERABLE_EXCEPTIONS:
        anchor = 0.0
    if not math.isfinite(anchor) or anchor <= 0:
        anchor = float(vals[0])

    max_step = 0.0
    flips = 0
    dirs: list[int] = []
    for i in range(1, len(vals)):
        prev = float(vals[i - 1])
        cur = float(vals[i])
        if prev > 0:
            step = abs(cur / max(prev, 1e-8) - 1.0)
            max_step = max(max_step, float(step))
        dirs.append(1 if cur >= prev else -1)
    for i in range(1, len(dirs)):
        if dirs[i] != dirs[i - 1]:
            flips += 1
    flip_ratio = float(flips) / float(max(1, len(dirs) - 1)) if len(dirs) >= 2 else 0.0

    vmin = min(vals)
    vmax = max(vals)
    span_pct = abs(vmax - vmin) / max(anchor, 1e-8)
    net_pct = (float(vals[-1]) / max(anchor, 1e-8)) - 1.0
    mean_v = float(sum(vals) / max(1, len(vals)))
    var = 0.0
    for v in vals:
        var += (float(v) - mean_v) ** 2
    std_pct = (math.sqrt(var / float(max(1, len(vals)))) / max(anchor, 1e-8))

    _, cap_step = self._chart_prediction_caps(iv_chart)
    quiet_span_pct = 0.0
    quiet_std_pct = 0.0
    try:
        recent_rows = list(self._bars_by_symbol.get(self._ui_norm(symbol), []) or [])
        recent_closes: list[float] = []
        for row in recent_rows[-96:]:
            try:
                row_iv = self._normalize_interval_token(
                    row.get("interval", iv_chart),
                    fallback=iv_chart,
                )
                if row_iv != iv_chart:
                    continue
                px = float(row.get("close", 0.0) or 0.0)
                if px > 0 and math.isfinite(px):
                    recent_closes.append(px)
            except _UI_RECOVERABLE_EXCEPTIONS:
                continue
        if len(recent_closes) >= 6:
            q_anchor = float(np.median(np.asarray(recent_closes[-24:], dtype=float)))
            q_anchor = max(q_anchor, 1e-8)
            quiet_span_pct = (
                abs(max(recent_closes) - min(recent_closes)) / q_anchor
            )
            q_std = float(np.std(np.asarray(recent_closes, dtype=float)))
            quiet_std_pct = q_std / q_anchor
    except _UI_RECOVERABLE_EXCEPTIONS:
        quiet_span_pct = 0.0
        quiet_std_pct = 0.0

    quiet_market = bool(
        (quiet_span_pct > 0 and quiet_span_pct <= max(0.0018, cap_step * 0.55))
        and (quiet_std_pct > 0 and quiet_std_pct <= max(0.0006, cap_step * 0.14))
    )

    flat_line_raw = len(vals) >= 5 and (span_pct <= 0.0012 or std_pct <= 0.0005)
    flat_line = bool(flat_line_raw and not quiet_market)
    jagged = max_step > (cap_step * 1.80) or flip_ratio > 0.82

    self._debug_console(
        f"forecast_q:{context}:{sym}:{iv_chart}",
        (
            f"forecast quality [{context}] symbol={sym} chart={iv_chart} "
            f"source={iv_src} points={len(vals)} span={span_pct:.2%} "
            f"net={net_pct:+.2%} max_step={max_step:.2%} flips={flip_ratio:.2f} "
            f"std={std_pct:.2%} step_cap={cap_step:.2%}"
        ),
        min_gap_seconds=1.5,
        level="info",
    )
    if flat_line or jagged:
        why = []
        if flat_line:
            why.append("flat_line")
        if jagged:
            why.append("jagged")
        self._debug_console(
            f"forecast_q_warn:{context}:{sym}:{iv_chart}",
            (
                f"forecast anomaly [{context}] symbol={sym} chart={iv_chart} "
                f"source={iv_src} reason={','.join(why)} "
                f"span={span_pct:.2%} max_step={max_step:.2%} flips={flip_ratio:.2f}"
            ),
            min_gap_seconds=0.8,
            level="warning",
        )
    elif flat_line_raw and quiet_market:
        self._debug_console(
            f"forecast_q_quiet:{context}:{sym}:{iv_chart}",
            (
                f"forecast quiet-shape [{context}] symbol={sym} chart={iv_chart} "
                f"source={iv_src} accepted_flat=1 "
                f"market_span={quiet_span_pct:.2%} market_std={quiet_std_pct:.2%}"
            ),
            min_gap_seconds=2.0,
            level="info",
        )

def _chart_prediction_caps(self, interval: str) -> tuple[float, float]:
    """Return (max_total_move, max_step_move) for display-only forecast shaping."""
    iv = self._normalize_interval_token(interval)
    if iv == "1m":
        return 0.028, 0.007
    if iv == "5m":
        return 0.045, 0.012
    if iv in ("15m", "30m"):
        return 0.065, 0.016
    if iv in ("60m", "1h"):
        return 0.100, 0.025
    if iv == "1d":
        return 0.120, 0.020
    if iv == "1wk":
        return 0.160, 0.028
    if iv == "1mo":
        return 0.220, 0.040
    return 0.070, 0.022

def _apply_news_policy_bias_to_forecast(
    self,
    *,
    symbol: str,
    chart_interval: str,
    anchor: float,
    predicted_prices: list[float],
) -> list[float]:
    """Blend cached news/policy signal into chart forecast (display-only).

    The adjustment is intentionally bounded so it nudges regime direction
    without introducing visual spikes.
    """
    vals = [float(v) for v in self._safe_list(predicted_prices) if float(v) > 0]
    if not vals:
        return []
    if anchor <= 0 or not math.isfinite(anchor):
        return vals

    signal = {}
    if hasattr(self, "_news_policy_signal_for"):
        try:
            signal = dict(self._news_policy_signal_for(symbol) or {})
        except _UI_RECOVERABLE_EXCEPTIONS:
            signal = {}
    if not signal:
        return vals

    try:
        age_s = float(time.time() - float(signal.get("ts", 0.0) or 0.0))
    except _UI_RECOVERABLE_EXCEPTIONS:
        age_s = float("inf")
    if age_s > 900.0:
        return vals

    overall = float(signal.get("overall", 0.0) or 0.0)
    policy = float(signal.get("policy", 0.0) or 0.0)
    market = float(signal.get("market", 0.0) or 0.0)
    confidence = float(signal.get("confidence", 0.0) or 0.0)

    directional_bias = float(
        np.clip(
            (0.48 * overall) + (0.34 * policy) + (0.18 * market),
            -1.0,
            1.0,
        )
    )
    if abs(directional_bias) < 0.02:
        return vals

    iv = self._normalize_interval_token(chart_interval)
    max_total_move, _max_step_move = self._chart_prediction_caps(iv)
    bias_cap = float(np.clip(max_total_move * 0.30, 0.004, 0.03))
    bias_total = float(np.clip(directional_bias * (0.25 + (0.55 * confidence)), -bias_cap, bias_cap))
    if abs(bias_total) < 1e-6:
        return vals

    out: list[float] = []
    n = max(1, len(vals))
    for i, px in enumerate(vals, start=1):
        frac = float(i) / float(n)
        drift = float(bias_total * frac)
        adj = float(px) * (1.0 + drift)
        out.append(float(max(0.01, adj)))
    return out

def _prepare_chart_predicted_prices(
    self,
    *,
    symbol: str,
    chart_interval: str,
    predicted_prices: list[float] | None,
    source_interval: str | None = None,
    current_price: float | None = None,
    target_steps: int | None = None,
) -> list[float]:
    """Shape forecast for chart display stability.
    - Clamp implausible per-step spikes.
    - When model interval != chart interval, project a smooth path to avoid
      abrupt vertical zig-zags on intraday charts.
    """
    raw_vals = self._safe_list(predicted_prices)
    cleaned: list[float] = []
    for v in raw_vals:
        try:
            fv = float(v)
        except _UI_RECOVERABLE_EXCEPTIONS:
            continue
        if fv > 0 and math.isfinite(fv):
            cleaned.append(fv)
    if not cleaned:
        return []

    iv_chart = self._normalize_interval_token(chart_interval)
    iv_src = self._normalize_interval_token(source_interval, fallback=iv_chart)
    try:
        steps = int(target_steps if target_steps is not None else self.forecast_spin.value())
    except _UI_RECOVERABLE_EXCEPTIONS:
        steps = len(cleaned)
    steps = max(1, steps)

    try:
        anchor = float(current_price or 0.0)
    except _UI_RECOVERABLE_EXCEPTIONS:
        anchor = 0.0
    if not math.isfinite(anchor) or anchor <= 0:
        anchor = float(cleaned[0])
    if not math.isfinite(anchor) or anchor <= 0:
        return []

    cleaned = self._apply_news_policy_bias_to_forecast(
        symbol=symbol,
        chart_interval=iv_chart,
        anchor=anchor,
        predicted_prices=cleaned,
    )
    if not cleaned:
        return []

    max_total_move, max_step_move = self._chart_prediction_caps(iv_chart)

    # Mismatch mode: preserve source-shape, then resample to chart steps.
    if iv_src != iv_chart:
        chart_sec = float(max(1, self._interval_seconds(iv_chart)))
        src_sec = float(max(1, self._interval_seconds(iv_src)))
        tf_ratio = float(src_sec / max(chart_sec, 1.0))
        proj_total_cap = float(max_total_move)
        proj_step_cap = float(max_step_move)
        conservative_projection = False
        if tf_ratio <= 0.20:
            # Source is much finer than chart interval (for example 1m -> 1d).
            # Compress micro-noise before projection to avoid sawtooth guessed
            # curves on higher-timeframe charts.
            conservative_projection = True
            if iv_chart == "1d":
                proj_total_cap = min(proj_total_cap, 0.10)
                proj_step_cap = min(proj_step_cap, 0.012)
            elif iv_chart in ("1wk", "1mo"):
                proj_total_cap = min(proj_total_cap, 0.14)
                proj_step_cap = min(proj_step_cap, 0.020)
            else:
                proj_total_cap = min(proj_total_cap, max_total_move * 0.80)
                proj_step_cap = min(proj_step_cap, max_step_move * 0.75)
        if tf_ratio >= 8.0:
            conservative_projection = True
            if iv_chart == "1m":
                proj_total_cap = min(proj_total_cap, 0.016)
                proj_step_cap = min(proj_step_cap, 0.004)
            elif iv_chart == "5m":
                proj_total_cap = min(proj_total_cap, 0.025)
                proj_step_cap = min(proj_step_cap, 0.007)
            else:
                proj_total_cap = min(proj_total_cap, max_total_move * 0.85)
                proj_step_cap = min(proj_step_cap, max_step_move * 0.85)

        src_curve: list[float] = [float(anchor)] + [float(v) for v in cleaned]
        if tf_ratio <= 0.20 and len(src_curve) >= 6:
            chunk = int(
                max(
                    3,
                    min(
                        len(src_curve) - 1,
                        round(1.0 / max(tf_ratio, 1e-4)),
                    ),
                )
            )
            compressed = [float(src_curve[0])]
            for start in range(1, len(src_curve), chunk):
                seg = src_curve[start:start + chunk]
                if not seg:
                    continue
                compressed.append(float(np.median(np.asarray(seg, dtype=float))))
            if len(compressed) >= 2:
                src_curve = compressed
        # Clamp total move while preserving path curvature.
        raw_net = float(src_curve[-1] / max(anchor, 1e-8) - 1.0)
        net_ret = float(max(-proj_total_cap, min(proj_total_cap, raw_net)))
        if abs(raw_net) > 1e-8 and raw_net != net_ret:
            scale = float(net_ret / raw_net)
            for i in range(1, len(src_curve)):
                src_ret = float(src_curve[i] / max(anchor, 1e-8) - 1.0)
                src_curve[i] = float(anchor) * (1.0 + (src_ret * scale))
        # Hard-clip every source point into chart-safe movement band.
        lo_anchor = float(anchor) * (1.0 - proj_total_cap)
        hi_anchor = float(anchor) * (1.0 + proj_total_cap)
        for i in range(1, len(src_curve)):
            src_curve[i] = float(max(lo_anchor, min(hi_anchor, src_curve[i])))
        # Mild smoothing to suppress jagged daily->intraday interpolation artifacts.
        if len(src_curve) >= 4:
            smoothed = list(src_curve)
            for i in range(1, len(src_curve) - 1):
                y0 = float(src_curve[i - 1])
                y1 = float(src_curve[i])
                y2 = float(src_curve[i + 1])
                val = (0.18 * y0) + (0.64 * y1) + (0.18 * y2)
                smoothed[i] = float(max(lo_anchor, min(hi_anchor, val)))
            src_curve = smoothed

        n_src = len(src_curve)
        src_x: list[float] = [0.0]
        if n_src <= 2:
            src_x.append(float(steps))
        else:
            step_span = float(steps) / float(max(1, n_src - 1))
            for i in range(1, n_src):
                src_x.append(float(i) * step_span)

        projected_out: list[float] = []
        prev = float(anchor)
        seg = 0
        for i in range(1, steps + 1):
            x = float(i)
            while seg + 1 < len(src_x) and x > src_x[seg + 1]:
                seg += 1
            if seg + 1 >= len(src_x):
                target_px = float(src_curve[-1])
            else:
                x0 = float(src_x[seg])
                x1 = float(src_x[seg + 1])
                y0 = float(src_curve[seg])
                y1 = float(src_curve[seg + 1])
                if x1 <= x0:
                    target_px = y1
                else:
                    frac = (x - x0) / (x1 - x0)
                    target_px = y0 + ((y1 - y0) * frac)
            target_px = float(max(lo_anchor, min(hi_anchor, target_px)))

            step_ret = float(target_px / max(prev, 1e-8) - 1.0)
            if step_ret > proj_step_cap:
                target_px = float(prev) * (1.0 + proj_step_cap)
            elif step_ret < -proj_step_cap:
                target_px = float(prev) * (1.0 - proj_step_cap)
            target_px = float(max(lo_anchor, min(hi_anchor, target_px)))
            projected_out.append(float(target_px))
            prev = float(target_px)
        self._debug_console(
            f"forecast_display_project:{self._ui_norm(symbol)}:{iv_chart}",
            (
                f"forecast display projection for {self._ui_norm(symbol)}: "
                f"source={iv_src} chart={iv_chart} steps={steps} "
                f"src_points={len(cleaned)} net={net_ret:+.2%} "
                f"tf_ratio={tf_ratio:.1f} conservative={int(conservative_projection)}"
            ),
            min_gap_seconds=3.0,
            level="info",
        )
        return projected_out

    # Same-interval mode: clamp step spikes, keep model shape.
    out: list[float] = []
    prev = float(anchor)
    lo_anchor = float(anchor) * (1.0 - max_total_move)
    hi_anchor = float(anchor) * (1.0 + max_total_move)
    for p in cleaned[:steps]:
        px = float(max(lo_anchor, min(hi_anchor, float(p))))
        step_ret = float(px / max(prev, 1e-8) - 1.0)
        if step_ret > max_step_move:
            px = float(prev) * (1.0 + max_step_move)
        elif step_ret < -max_step_move:
            px = float(prev) * (1.0 - max_step_move)
        px = float(max(lo_anchor, min(hi_anchor, px)))
        out.append(float(px))
        prev = float(px)

    if len(out) >= 4:
        dirs = [1 if out[i] >= out[i - 1] else -1 for i in range(1, len(out))]
        flips = sum(1 for i in range(1, len(dirs)) if dirs[i] != dirs[i - 1])
        flip_ratio = float(flips) / float(max(1, len(dirs) - 1))
        raw_steps = [
            abs(float(out[i]) / max(float(out[i - 1]), 1e-8) - 1.0)
            for i in range(1, len(out))
        ]
        max_raw_step = max(raw_steps) if raw_steps else 0.0
        if flip_ratio > 0.65 or max_raw_step > (max_step_move * 1.35):
            smooth = list(out)
            for i in range(1, len(out) - 1):
                val = (
                    (0.22 * float(out[i - 1]))
                    + (0.56 * float(out[i]))
                    + (0.22 * float(out[i + 1]))
                )
                smooth[i] = float(max(lo_anchor, min(hi_anchor, val)))

            out2: list[float] = []
            prev2 = float(anchor)
            for px in smooth:
                px2 = float(max(lo_anchor, min(hi_anchor, float(px))))
                step_ret2 = float(px2 / max(prev2, 1e-8) - 1.0)
                if step_ret2 > max_step_move:
                    px2 = float(prev2) * (1.0 + max_step_move)
                elif step_ret2 < -max_step_move:
                    px2 = float(prev2) * (1.0 - max_step_move)
                px2 = float(max(lo_anchor, min(hi_anchor, px2)))
                out2.append(float(px2))
                prev2 = float(px2)
            out = out2
    return out

def _chart_prediction_uncertainty_profile(
    self,
    symbol: str,
) -> tuple[float, float, float]:
    """Resolve (uncertainty, tail_risk, confidence) for chart forecast bands."""
    uncertainty = 0.55
    tail_risk = 0.55
    confidence = 0.45

    pred = getattr(self, "current_prediction", None)
    if pred and self._ui_norm(getattr(pred, "stock_code", "")) == self._ui_norm(symbol):
        try:
            uncertainty = float(
                np.clip(getattr(pred, "uncertainty_score", uncertainty), 0.0, 1.0)
            )
        except _UI_RECOVERABLE_EXCEPTIONS as exc:
            log.debug("Suppressed exception in ui/app.py", exc_info=exc)
        try:
            tail_risk = float(
                np.clip(getattr(pred, "tail_risk_score", tail_risk), 0.0, 1.0)
            )
        except _UI_RECOVERABLE_EXCEPTIONS as exc:
            log.debug("Suppressed exception in ui/app.py", exc_info=exc)
        try:
            confidence = float(
                np.clip(getattr(pred, "confidence", confidence), 0.0, 1.0)
            )
        except _UI_RECOVERABLE_EXCEPTIONS as exc:
            log.debug("Suppressed exception in ui/app.py", exc_info=exc)

    return uncertainty, tail_risk, confidence

def _build_chart_prediction_bands(
    self,
    *,
    symbol: str,
    predicted_prices: list[float] | None,
    anchor_price: float | None,
    chart_interval: str | None = None,
) -> tuple[list[float], list[float]]:
    """Build chart uncertainty envelope around predicted prices."""
    vals = []
    for v in self._safe_list(predicted_prices):
        try:
            fv = float(v)
        except _UI_RECOVERABLE_EXCEPTIONS:
            continue
        if fv > 0 and math.isfinite(fv):
            vals.append(float(fv))
    if not vals:
        return [], []

    try:
        anchor = float(anchor_price or 0.0)
    except _UI_RECOVERABLE_EXCEPTIONS:
        anchor = 0.0
    if anchor <= 0:
        anchor = float(vals[0])

    fallback_iv = "1m"
    try:
        fallback_iv = self._normalize_interval_token(
            self.interval_combo.currentText(),
            fallback="1m",
        )
    except _UI_RECOVERABLE_EXCEPTIONS:
        fallback_iv = "1m"
    iv = self._normalize_interval_token(
        chart_interval,
        fallback=fallback_iv,
    )
    max_total_move, max_step_move = self._chart_prediction_caps(iv)

    uncertainty, tail_risk, confidence = self._chart_prediction_uncertainty_profile(
        symbol
    )
    max_width = float(
        np.clip(max_total_move * 0.60, max(max_step_move * 1.2, 0.006), 0.18)
    )
    base_width = float(
        np.clip(
            0.004
            + (0.010 * uncertainty)
            + (0.008 * tail_risk)
            + (0.006 * (1.0 - confidence)),
            0.004,
            max(0.008, max_width * 0.55),
        )
    )

    n = max(1, len(vals))
    lows: list[float] = []
    highs: list[float] = []
    envelope_cap = float(
        np.clip(
            max_total_move * (0.82 + (0.18 * uncertainty)),
            max(max_step_move * 2.0, 0.01),
            max(max_total_move * 1.05, 0.02),
        )
    )
    for i, px in enumerate(vals, start=1):
        growth = 1.0 + (float(i) / float(n)) * (0.70 + (0.45 * uncertainty))
        width = float(np.clip(base_width * growth, 0.004, max_width))

        lo = max(0.01, float(px) * (1.0 - width))
        hi = max(lo + 1e-6, float(px) * (1.0 + width))

        # Keep envelope centered on plausible anchor neighborhood.
        if anchor > 0:
            lo = max(lo, anchor * (1.0 - envelope_cap))
            hi = min(hi, anchor * (1.0 + envelope_cap))
            if hi <= lo:
                hi = lo + max(1e-6, abs(lo) * 0.002)

        lows.append(float(lo))
        highs.append(float(hi))

    return lows, highs

def _resolve_chart_prediction_series(
    self,
    *,
    symbol: str,
    fallback_interval: str,
    predicted_prices: list[float] | None = None,
    source_interval: str | None = None,
) -> tuple[list[float], str]:
    """Resolve prediction series/source interval for chart rendering."""
    iv_fallback = self._normalize_interval_token(fallback_interval)
    iv_source = self._normalize_interval_token(
        source_interval,
        fallback=iv_fallback,
    )
    if predicted_prices is not None:
        return self._safe_list(predicted_prices), iv_source

    if (
        self.current_prediction
        and getattr(self.current_prediction, "stock_code", "") == symbol
    ):
        vals = (
            getattr(self.current_prediction, "predicted_prices", [])
            or []
        )
        iv_source = self._normalize_interval_token(
            getattr(self.current_prediction, "interval", iv_source),
            fallback=iv_source,
        )
        return list(vals), iv_source
    return [], iv_source

def _render_chart_state(
    self,
    *,
    symbol: str,
    interval: str,
    bars: list[dict[str, Any]] | None,
    context: str,
    current_price: float | None = None,
    predicted_prices: list[float] | None = None,
    source_interval: str | None = None,
    target_steps: int | None = None,
    predicted_prepared: bool = False,
    update_latest_label: bool = False,
    allow_legacy_candles: bool = False,
    reset_view_on_symbol_switch: bool = False,
) -> list[dict[str, Any]]:
    """Unified chart rendering path used by bar/tick/analysis updates."""
    iv = self._normalize_interval_token(interval)
    arr = self._safe_list(bars)

    anchor_input: float | None = None
    if current_price is not None:
        try:
            px = float(current_price)
            if px > 0 and math.isfinite(px):
                anchor_input = px
        except _UI_RECOVERABLE_EXCEPTIONS:
            anchor_input = None

    chart_anchor = self._effective_anchor_price(symbol, anchor_input)
    arr = self._scrub_chart_bars(
        arr,
        iv,
        symbol=symbol,
        anchor_price=chart_anchor if chart_anchor > 0 else None,
    )
    arr = self._stabilize_chart_depth(symbol, iv, arr)
    self._bars_by_symbol[symbol] = arr
    self._debug_candle_quality(
        symbol=symbol,
        interval=iv,
        bars=arr,
        context=context,
    )

    pred_vals, pred_source_iv = self._resolve_chart_prediction_series(
        symbol=symbol,
        fallback_interval=iv,
        predicted_prices=predicted_prices,
        source_interval=source_interval,
    )
    try:
        steps = int(
            target_steps if target_steps is not None else self.forecast_spin.value()
        )
    except _UI_RECOVERABLE_EXCEPTIONS:
        steps = int(self.forecast_spin.value())

    anchor_for_pred: float | None = None
    if arr:
        try:
            last_close = float(arr[-1].get("close", 0) or 0)
            if last_close > 0 and math.isfinite(last_close):
                anchor_for_pred = last_close
        except _UI_RECOVERABLE_EXCEPTIONS:
            anchor_for_pred = None
    if anchor_for_pred is None:
        anchor_for_pred = anchor_input

    if predicted_prepared and pred_vals:
        # Already shaped by _prepare_chart_predicted_prices upstream;
        # re-processing with a different anchor can distort or empty them.
        chart_predicted = list(pred_vals)
    else:
        chart_predicted = self._prepare_chart_predicted_prices(
            symbol=symbol,
            chart_interval=iv,
            predicted_prices=pred_vals,
            source_interval=pred_source_iv,
            current_price=anchor_for_pred,
            target_steps=steps,
        )
    chart_predicted_low, chart_predicted_high = self._build_chart_prediction_bands(
        symbol=symbol,
        predicted_prices=chart_predicted,
        anchor_price=anchor_for_pred,
        chart_interval=iv,
    )
    self._debug_forecast_quality(
        symbol=symbol,
        chart_interval=iv,
        source_interval=pred_source_iv,
        predicted_prices=chart_predicted,
        anchor_price=anchor_for_pred,
        context=context,
    )

    if (
        reset_view_on_symbol_switch
        and self._chart_symbol
        and self._chart_symbol != symbol
    ):
        try:
            self.chart.reset_view()
        except _UI_RECOVERABLE_EXCEPTIONS as exc:
            log.debug("Suppressed exception in ui/app.py", exc_info=exc)

    rendered = False
    if hasattr(self.chart, "update_chart"):
        self.chart.update_chart(
            arr,
            predicted_prices=chart_predicted,
            predicted_prices_low=chart_predicted_low,
            predicted_prices_high=chart_predicted_high,
            levels=self._get_levels_dict(),
        )
        self._debug_chart_state(
            symbol=symbol,
            interval=iv,
            bars=arr,
            predicted_prices=chart_predicted,
            context=context,
        )
        self._chart_symbol = symbol
        rendered = True
    elif allow_legacy_candles and hasattr(self.chart, "update_candles"):
        self.chart.update_candles(
            arr,
            predicted_prices=chart_predicted,
            predicted_prices_low=chart_predicted_low,
            predicted_prices_high=chart_predicted_high,
            levels=self._get_levels_dict(),
        )
        self._chart_symbol = symbol
        rendered = True

    if update_latest_label:
        label_price: float | None = None
        if anchor_input is not None:
            label_price = anchor_input
        self._update_chart_latest_label(
            symbol,
            bar=arr[-1] if arr else None,
            price=label_price,
        )
    if not rendered and not update_latest_label:
        # Keep the return side-effect free when no renderer is available.
        return arr
    return arr

