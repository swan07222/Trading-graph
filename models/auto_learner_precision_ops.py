from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from config.settings import CONFIG
from utils.logger import get_logger
from utils.recoverable import JSON_RECOVERABLE_EXCEPTIONS

log = get_logger(__name__)
_AUTO_LEARNER_RECOVERABLE_EXCEPTIONS = JSON_RECOVERABLE_EXCEPTIONS


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    if math.isnan(out) or math.isinf(out):
        return float(default)
    return float(out)


def _safe_int(value: object, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(float(lo), min(float(hi), float(value)))


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    clean: list[float] = []
    for v in values:
        val = _safe_float(v, default=math.nan)
        if math.isnan(val):
            continue
        clean.append(float(val))
    clean.sort()
    if not clean:
        return 0.0
    q = _clamp(float(q), 0.0, 1.0)
    pos = (len(clean) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(clean[lo])
    frac = pos - lo
    return float(clean[lo] * (1.0 - frac) + clean[hi] * frac)


def _wilson_lower_bound(rate: float, n: int, z: float) -> float:
    if n <= 0:
        return 0.0
    p = _clamp(rate, 0.0, 1.0)
    n_f = float(max(1, n))
    z2 = float(z * z)
    denom = 1.0 + z2 / n_f
    center = p + z2 / (2.0 * n_f)
    margin = z * math.sqrt((p * (1.0 - p) + z2 / (4.0 * n_f)) / n_f)
    return _clamp((center - margin) / denom, 0.0, 1.0)


def _extract_trade_samples(samples: list[dict[str, Any]]) -> list[dict[str, float]]:
    out: list[dict[str, float]] = []
    for row in list(samples or []):
        if not isinstance(row, dict):
            continue
        pred_cls = _safe_int(row.get("predicted", 1), default=1)
        if pred_cls not in (0, 2):
            continue
        prob_up = _clamp(_safe_float(row.get("prob_up", 0.33), default=0.33), 0.0, 1.0)
        prob_dn = _clamp(
            _safe_float(row.get("prob_down", 0.33), default=0.33), 0.0, 1.0
        )
        out.append(
            {
                "confidence": _clamp(
                    _safe_float(row.get("confidence", 0.0), default=0.0), 0.0, 1.0
                ),
                "agreement": _clamp(
                    _safe_float(row.get("agreement", 0.0), default=0.0), 0.0, 1.0
                ),
                "entropy": _clamp(
                    _safe_float(row.get("entropy", 1.0), default=1.0), 0.0, 1.0
                ),
                "edge": _clamp(abs(prob_up - prob_dn), 0.0, 1.0),
                "future_return": _safe_float(row.get("future_return", 0.0), default=0.0),
            }
        )
    return out


def _infer_regime(samples: list[dict[str, float]]) -> str:
    if not samples:
        return "unknown"
    cfg = getattr(CONFIG, "precision", None)
    high_vol_ret = _safe_float(
        getattr(cfg, "validation_high_vol_return_pct", 1.2), default=1.2
    )
    low_signal_edge = _safe_float(
        getattr(cfg, "validation_low_signal_edge", 0.10), default=0.10
    )

    abs_ret = [abs(_safe_float(s.get("future_return", 0.0), default=0.0)) for s in samples]
    edges = [_safe_float(s.get("edge", 0.0), default=0.0) for s in samples]
    entropy = [_safe_float(s.get("entropy", 1.0), default=1.0) for s in samples]

    q75_abs_ret = _quantile(abs_ret, 0.75)
    med_edge = _quantile(edges, 0.50)
    med_entropy = _quantile(entropy, 0.50)

    if q75_abs_ret >= max(0.1, high_vol_ret):
        return "high_vol"
    if med_edge <= max(0.01, low_signal_edge) or med_entropy >= 0.55:
        return "low_signal"
    return "trend"


def _load_previous_profile_thresholds() -> dict[str, float]:
    cfg = getattr(CONFIG, "precision", None)
    filename = str(getattr(cfg, "profile_filename", "precision_thresholds.json"))
    path = Path(CONFIG.data_dir) / filename
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except _AUTO_LEARNER_RECOVERABLE_EXCEPTIONS:
        return {}
    if not isinstance(payload, dict):
        return {}
    thresholds = payload.get("thresholds", {})
    if not isinstance(thresholds, dict):
        return {}
    out: dict[str, float] = {}
    for key in ("min_confidence", "min_agreement", "max_entropy", "min_edge"):
        if key in thresholds:
            out[key] = _safe_float(thresholds[key], default=0.0)
    return out


def _compress_candidates(values: list[float], *, max_size: int) -> list[float]:
    uniq = sorted(set(round(float(v), 4) for v in values))
    if len(uniq) <= int(max_size):
        return [float(v) for v in uniq]
    if max_size <= 1:
        return [float(uniq[len(uniq) // 2])]
    idxs = [
        int(round(i * (len(uniq) - 1) / float(max_size - 1)))
        for i in range(max_size)
    ]
    return [float(uniq[i]) for i in sorted(set(idxs))]


def _build_threshold_candidates(
    samples: list[dict[str, float]],
) -> tuple[list[float], list[float], list[float], list[float], str]:
    cfg = getattr(CONFIG, "precision", None)
    prev = _load_previous_profile_thresholds()
    regime = _infer_regime(samples)

    max_per_axis = max(
        4,
        _safe_int(getattr(cfg, "tuning_max_candidates_per_axis", 7), default=7),
    )

    conf_values = [_safe_float(s.get("confidence", 0.0), default=0.0) for s in samples]
    agree_values = [_safe_float(s.get("agreement", 0.0), default=0.0) for s in samples]
    entropy_values = [_safe_float(s.get("entropy", 1.0), default=1.0) for s in samples]
    edge_values = [_safe_float(s.get("edge", 0.0), default=0.0) for s in samples]

    base_conf = _clamp(
        _safe_float(getattr(cfg, "min_confidence", 0.78), default=0.78), 0.0, 1.0
    )
    base_agree = _clamp(
        _safe_float(getattr(cfg, "min_agreement", 0.72), default=0.72), 0.0, 1.0
    )
    base_entropy = _clamp(
        _safe_float(getattr(cfg, "max_entropy", 0.35), default=0.35), 0.0, 1.0
    )
    base_edge = _clamp(
        _safe_float(getattr(cfg, "min_edge", 0.14), default=0.14), 0.0, 1.0
    )

    conf_grid = [
        0.60,
        0.65,
        0.70,
        0.75,
        0.80,
        _quantile(conf_values, 0.40),
        _quantile(conf_values, 0.55),
        _quantile(conf_values, 0.70),
        _quantile(conf_values, 0.82),
        base_conf,
        prev.get("min_confidence", base_conf),
        prev.get("min_confidence", base_conf) - 0.03,
        prev.get("min_confidence", base_conf) + 0.03,
    ]
    agree_grid = [
        0.55,
        0.60,
        0.65,
        0.70,
        0.75,
        _quantile(agree_values, 0.40),
        _quantile(agree_values, 0.55),
        _quantile(agree_values, 0.70),
        _quantile(agree_values, 0.82),
        base_agree,
        prev.get("min_agreement", base_agree),
        prev.get("min_agreement", base_agree) - 0.03,
        prev.get("min_agreement", base_agree) + 0.03,
    ]
    entropy_grid = [
        0.20,
        0.30,
        0.40,
        0.50,
        0.60,
        _quantile(entropy_values, 0.20),
        _quantile(entropy_values, 0.35),
        _quantile(entropy_values, 0.50),
        _quantile(entropy_values, 0.65),
        base_entropy,
        prev.get("max_entropy", base_entropy),
        prev.get("max_entropy", base_entropy) - 0.05,
        prev.get("max_entropy", base_entropy) + 0.05,
    ]
    edge_grid = [
        0.06,
        0.10,
        0.14,
        0.18,
        0.22,
        _quantile(edge_values, 0.40),
        _quantile(edge_values, 0.55),
        _quantile(edge_values, 0.70),
        _quantile(edge_values, 0.82),
        base_edge,
        prev.get("min_edge", base_edge),
        prev.get("min_edge", base_edge) - 0.03,
        prev.get("min_edge", base_edge) + 0.03,
    ]

    if regime == "high_vol":
        conf_grid.extend([base_conf - 0.04, base_conf - 0.02])
        agree_grid.extend([base_agree - 0.04, base_agree - 0.02])
        edge_grid.extend([base_edge + 0.02, base_edge + 0.04])
    elif regime == "low_signal":
        conf_grid.extend([base_conf + 0.03, base_conf + 0.06])
        agree_grid.extend([base_agree + 0.03, base_agree + 0.06])
        entropy_grid.extend([base_entropy - 0.06, base_entropy - 0.03])
        edge_grid.extend([base_edge + 0.03, base_edge + 0.06])

    conf_candidates = _compress_candidates(
        [_clamp(v, 0.45, 0.98) for v in conf_grid], max_size=max_per_axis
    )
    agree_candidates = _compress_candidates(
        [_clamp(v, 0.45, 0.98) for v in agree_grid], max_size=max_per_axis
    )
    entropy_candidates = _compress_candidates(
        [_clamp(v, 0.10, 0.95) for v in entropy_grid], max_size=max_per_axis
    )
    edge_candidates = _compress_candidates(
        [_clamp(v, 0.01, 0.90) for v in edge_grid], max_size=max_per_axis
    )
    return (
        conf_candidates,
        agree_candidates,
        entropy_candidates,
        edge_candidates,
        regime,
    )


def _score_candidate(metrics: dict[str, float], regime: str, min_trades: int) -> float:
    pf = _safe_float(metrics.get("profit_factor", 0.0), default=0.0)
    precision = _safe_float(metrics.get("precision", 0.0), default=0.0)
    expectancy = _safe_float(metrics.get("expectancy", 0.0), default=0.0)
    trade_rate = _safe_float(metrics.get("trade_rate", 0.0), default=0.0)
    trades = _safe_float(metrics.get("trades", 0.0), default=0.0)

    if regime == "high_vol":
        score = (pf * 2.25) + (precision * 1.00) + (expectancy * 0.30) - (trade_rate * 0.12)
    elif regime == "low_signal":
        score = (pf * 1.70) + (precision * 1.60) + (expectancy * 0.20) - (trade_rate * 0.08)
    else:
        score = (pf * 2.00) + (precision * 1.20) + (expectancy * 0.25) - (trade_rate * 0.05)

    support = min(1.0, trades / float(max(1, min_trades)))
    score += support * 0.20
    if precision < 0.45:
        score -= (0.45 - precision) * 1.2
    return float(score)


def tune_precision_thresholds(self, samples: list[dict[str, Any]]) -> dict[str, Any] | None:
    filtered = _extract_trade_samples(samples)
    if not filtered:
        return None

    (
        conf_candidates,
        agree_candidates,
        entropy_candidates,
        edge_candidates,
        regime,
    ) = _build_threshold_candidates(filtered)

    cfg = getattr(CONFIG, "precision", None)
    min_trade_rate = _clamp(
        _safe_float(getattr(cfg, "tuning_min_trade_rate", 0.03), default=0.03),
        0.005,
        0.50,
    )
    min_required = max(
        int(getattr(self, "_MIN_TUNED_TRADES", 3)),
        int(math.ceil(float(len(filtered)) * min_trade_rate)),
    )

    best_score = -1e18
    best: dict[str, float] | None = None
    search_space_size = (
        len(conf_candidates)
        * len(agree_candidates)
        * len(entropy_candidates)
        * len(edge_candidates)
    )

    for c in conf_candidates:
        for a in agree_candidates:
            for e in entropy_candidates:
                for edge in edge_candidates:
                    metrics = self._score_thresholds(
                        samples,
                        min_conf=float(c),
                        min_agree=float(a),
                        max_entropy=float(e),
                        min_edge=float(edge),
                    )
                    if int(metrics.get("trades", 0.0) or 0) < min_required:
                        continue
                    score = _score_candidate(
                        metrics, regime=regime, min_trades=min_required
                    )
                    if score > best_score:
                        best_score = score
                        best = {
                            "min_confidence": float(c),
                            "min_agreement": float(a),
                            "max_entropy": float(e),
                            "min_edge": float(edge),
                            "precision": _safe_float(
                                metrics.get("precision", 0.0), default=0.0
                            ),
                            "profit_factor": _safe_float(
                                metrics.get("profit_factor", 0.0), default=0.0
                            ),
                            "expectancy": _safe_float(
                                metrics.get("expectancy", 0.0), default=0.0
                            ),
                            "trades": _safe_float(metrics.get("trades", 0.0), default=0.0),
                            "trade_rate": _safe_float(
                                metrics.get("trade_rate", 0.0), default=0.0
                            ),
                            "regime": str(regime),
                            "search_space_size": float(search_space_size),
                            "min_required_trades": float(min_required),
                            "objective": float(score),
                        }

    if best is not None:
        return best

    # Fallback: keep the current precision profile defaults if no candidate passes.
    base_cfg = getattr(CONFIG, "precision", None)
    fallback_conf = _clamp(
        _safe_float(getattr(base_cfg, "min_confidence", 0.78), default=0.78), 0.45, 0.98
    )
    fallback_agree = _clamp(
        _safe_float(getattr(base_cfg, "min_agreement", 0.72), default=0.72), 0.45, 0.98
    )
    fallback_entropy = _clamp(
        _safe_float(getattr(base_cfg, "max_entropy", 0.35), default=0.35), 0.10, 0.95
    )
    fallback_edge = _clamp(
        _safe_float(getattr(base_cfg, "min_edge", 0.14), default=0.14), 0.01, 0.90
    )
    metrics = self._score_thresholds(
        samples,
        min_conf=float(fallback_conf),
        min_agree=float(fallback_agree),
        max_entropy=float(fallback_entropy),
        min_edge=float(fallback_edge),
    )
    if int(metrics.get("trades", 0.0) or 0) < int(getattr(self, "_MIN_TUNED_TRADES", 3)):
        return None
    return {
        "min_confidence": float(fallback_conf),
        "min_agreement": float(fallback_agree),
        "max_entropy": float(fallback_entropy),
        "min_edge": float(fallback_edge),
        "precision": _safe_float(metrics.get("precision", 0.0), default=0.0),
        "profit_factor": _safe_float(metrics.get("profit_factor", 0.0), default=0.0),
        "expectancy": _safe_float(metrics.get("expectancy", 0.0), default=0.0),
        "trades": _safe_float(metrics.get("trades", 0.0), default=0.0),
        "trade_rate": _safe_float(metrics.get("trade_rate", 0.0), default=0.0),
        "regime": str(regime),
        "search_space_size": float(search_space_size),
        "min_required_trades": float(min_required),
        "objective": float(_score_candidate(metrics, regime=regime, min_trades=min_required)),
    }


def validate_and_decide(
    self,
    interval: str,
    horizon: int,
    lookback: int,
    pre_val: dict[str, Any] | None,
    new_acc: float,
) -> bool:
    holdout_snapshot = list(self._get_holdout_set())
    if not holdout_snapshot:
        log.info("No holdout validation - accepting")
        return True

    post_val = self._guardian.validate_model(
        interval, horizon, holdout_snapshot, lookback, collect_samples=True
    )
    post_acc = _clamp(_safe_float(post_val.get("accuracy", 0.0), default=0.0), 0.0, 1.0)
    post_conf = _clamp(
        _safe_float(post_val.get("avg_confidence", 0.0), default=0.0), 0.0, 1.0
    )
    post_preds = max(0, _safe_int(post_val.get("predictions_made", 0), default=0))
    samples = _extract_trade_samples(list(post_val.get("samples", []) or []))

    self.progress.old_stock_accuracy = post_acc
    self.progress.old_stock_confidence = post_conf

    cfg = getattr(CONFIG, "precision", None)
    min_preds = max(
        int(getattr(self, "_MIN_HOLDOUT_PREDICTIONS", 3)),
        _safe_int(getattr(cfg, "validation_min_predictions", 5), default=5),
    )
    z = max(0.5, _safe_float(getattr(cfg, "validation_confidence_z", 1.64), default=1.64))
    post_lb = _wilson_lower_bound(post_acc, post_preds, z=z)

    base_min_lb = _clamp(
        _safe_float(getattr(cfg, "validation_min_accept_lb", 0.30), default=0.30),
        0.05,
        0.95,
    )
    base_acc_deg = _clamp(
        _safe_float(
            getattr(cfg, "validation_max_accuracy_degradation", 0.15), default=0.15
        ),
        0.01,
        0.90,
    )
    base_conf_deg = _clamp(
        _safe_float(
            getattr(cfg, "validation_max_confidence_degradation", 0.18), default=0.18
        ),
        0.01,
        0.90,
    )
    max_train_holdout_gap = _clamp(
        _safe_float(getattr(cfg, "validation_max_train_holdout_gap", 0.40), default=0.40),
        0.05,
        1.00,
    )
    conf_margin = _clamp(
        _safe_float(getattr(cfg, "validation_confidence_margin", 0.03), default=0.03),
        0.00,
        0.30,
    )
    high_vol_relax = _clamp(
        _safe_float(getattr(cfg, "validation_high_vol_relax", 0.05), default=0.05),
        0.0,
        0.40,
    )
    low_signal_tighten = _clamp(
        _safe_float(getattr(cfg, "validation_low_signal_tighten", 0.04), default=0.04),
        0.0,
        0.40,
    )

    regime = _infer_regime(samples)
    required_lb = base_min_lb
    max_acc_deg = base_acc_deg
    max_conf_deg = base_conf_deg
    if regime == "high_vol":
        required_lb = max(0.20, base_min_lb - high_vol_relax)
        max_acc_deg = min(0.60, base_acc_deg + high_vol_relax)
        max_conf_deg = min(0.60, base_conf_deg + high_vol_relax)
    elif regime == "low_signal":
        required_lb = min(0.65, base_min_lb + low_signal_tighten)
        max_acc_deg = max(0.05, base_acc_deg - low_signal_tighten)
        max_conf_deg = max(0.05, base_conf_deg - low_signal_tighten)

    def _reject(reason: str) -> bool:
        log.warning("REJECTED: %s", reason)
        self.progress.add_warning(f"Rejected: {reason}")
        self._guardian.restore_backup(interval, horizon)
        return False

    if post_preds < min_preds:
        return _reject(
            f"holdout insufficient ({post_preds}/{min_preds} predictions)"
        )

    if post_lb < required_lb:
        return _reject(
            f"holdout lower-bound {post_lb:.1%} below required {required_lb:.1%} (regime={regime})"
        )

    if (float(new_acc) - post_acc) > max_train_holdout_gap:
        return _reject(
            f"train/holdout gap too large ({float(new_acc):.1%} vs {post_acc:.1%})"
        )

    pre_preds = max(
        0,
        _safe_int((pre_val or {}).get("predictions_made", 0), default=0),
    )
    if not pre_val or pre_preds < min_preds:
        log.info(
            "No reliable pre-validation baseline (preds=%s). "
            "post_acc=%.1f%% post_lb=%.1f%% regime=%s",
            pre_preds,
            post_acc * 100.0,
            post_lb * 100.0,
            regime,
        )
        self._maybe_tune_precision_thresholds(
            interval, horizon, list(post_val.get("samples", []) or [])
        )
        return True

    pre_acc = _clamp(
        _safe_float(pre_val.get("accuracy", 0.0), default=0.0), 0.0, 1.0
    )
    pre_conf = _clamp(
        _safe_float(pre_val.get("avg_confidence", 0.0), default=0.0), 0.0, 1.0
    )
    pre_lb = _wilson_lower_bound(pre_acc, pre_preds, z=z)
    acc_deg = (
        max(0.0, (pre_lb - post_lb) / max(pre_lb, 1e-9))
        if pre_lb > 0.0
        else 0.0
    )
    if acc_deg > max_acc_deg:
        return _reject(
            f"holdout lower-bound degraded {acc_deg:.1%} "
            f"(limit {max_acc_deg:.1%}, regime={regime})"
        )

    conf_deg = (
        max(0.0, (pre_conf - post_conf) / max(pre_conf, 1e-9))
        if pre_conf > 0.0
        else 0.0
    )
    conf_se = math.sqrt(
        max(post_conf * (1.0 - post_conf), 1e-4) / float(max(1, post_preds))
    )
    conf_limit = max_conf_deg + conf_margin + conf_se
    if conf_deg > conf_limit:
        return _reject(
            f"holdout confidence degraded {conf_deg:.1%} "
            f"(limit {conf_limit:.1%}, regime={regime})"
        )

    log.info(
        "ACCEPTED: regime=%s pre_lb=%.1f%% post_lb=%.1f%% "
        "pre_conf=%.3f post_conf=%.3f train_acc=%.1f%%",
        regime,
        pre_lb * 100.0,
        post_lb * 100.0,
        pre_conf,
        post_conf,
        float(new_acc) * 100.0,
    )
    self._maybe_tune_precision_thresholds(
        interval, horizon, list(post_val.get("samples", []) or [])
    )
    return True
