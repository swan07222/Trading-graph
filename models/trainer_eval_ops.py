from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np

from config.settings import CONFIG
from utils.logger import get_logger

log = get_logger(__name__)

_EPS = 1e-8

# Stop check interval for batch loops (check every N batches)
_STOP_CHECK_INTERVAL = 10
_TRAINING_INTERVAL_LOCK = "1m"
# FIX 1M: Reduced from 10080 to 480 bars - free sources provide 1-2 days of 1m data
_MIN_1M_LOOKBACK_BARS = 480
_DEFAULT_ENSEMBLE_MODELS = ["lstm", "gru", "tcn", "transformer", "hybrid"]
_WALK_FORWARD_FOLDS = 3
_MIN_WALK_FORWARD_SAMPLES = 180
_OVERFIT_VAL_ACC_DROP_WARN = 0.06
_OVERFIT_LOSS_GAP_WARN = 0.35
_DRIFT_WARN_SCORE_DROP = 0.08
_DRIFT_BLOCK_SCORE_DROP = 0.16
_DRIFT_WARN_ACC_DROP = 0.05
_DRIFT_BLOCK_ACC_DROP = 0.10
_MIN_BASELINE_RISK_SCORE = 0.52
_MIN_BASELINE_PROFIT_FACTOR = 1.05
_MAX_BASELINE_DRAWDOWN = 0.25
_MIN_BASELINE_TRADES = 5
_DATA_QUALITY_MAX_NAN_RATIO = 0.04
_DATA_QUALITY_MAX_NONPOS_PRICE_RATIO = 0.0
_DATA_QUALITY_MAX_BROKEN_OHLC_RATIO = 0.001
_DATA_QUALITY_MIN_VALID_SYMBOL_RATIO = 0.55
_INCREMENTAL_REGIME_BLOCK_LEVELS = {"high"}
_STRESS_COST_MULTIPLIERS = (1.0, 1.5, 2.0)
_TAIL_STRESS_QUANTILE = 0.90
_MIN_TAIL_STRESS_SAMPLES = 24
_TAIL_EVENT_SHOCK_MIN_PCT = 1.0
_TAIL_EVENT_SHOCK_MAX_PCT = 6.0

def _walk_forward_validate(
    self,
    X_val: np.ndarray | None,
    y_val: np.ndarray | None,
    r_val: np.ndarray | None,
    X_test: np.ndarray | None,
    y_test: np.ndarray | None,
    r_test: np.ndarray | None,
    regime_profile: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate stability on contiguous forward windows.

    This is a post-train diagnostic (not re-training folds) used to
    detect unstable model behavior across recent slices.
    """
    X_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    r_parts: list[np.ndarray] = []

    for X_arr, y_arr, r_arr in [
        (X_val, y_val, r_val),
        (X_test, y_test, r_test),
    ]:
        if (
            X_arr is None
            or y_arr is None
            or r_arr is None
            or len(X_arr) == 0
            or len(y_arr) == 0
            or len(r_arr) == 0
        ):
            continue
        n = int(min(len(X_arr), len(y_arr), len(r_arr)))
        if n <= 0:
            continue
        X_parts.append(X_arr[:n])
        y_parts.append(y_arr[:n])
        r_parts.append(r_arr[:n])

    if not X_parts:
        return {
            "enabled": False,
            "reason": "no_eval_data",
        }

    X_eval = np.concatenate(X_parts, axis=0)
    y_eval = np.concatenate(y_parts, axis=0)
    r_eval = np.concatenate(r_parts, axis=0)

    max_samples = 3000
    if len(X_eval) > max_samples:
        X_eval = X_eval[-max_samples:]
        y_eval = y_eval[-max_samples:]
        r_eval = r_eval[-max_samples:]

    fold_count = int(_WALK_FORWARD_FOLDS)
    min_fold_size = max(int(CONFIG.SEQUENCE_LENGTH), 64)
    min_required = max(
        _MIN_WALK_FORWARD_SAMPLES,
        fold_count * min_fold_size,
    )
    if len(X_eval) < min_required:
        return {
            "enabled": False,
            "reason": (
                f"insufficient_samples (need>={min_required}, "
                f"got={len(X_eval)})"
            ),
        }

    fold_size = len(X_eval) // fold_count
    fold_results: list[dict[str, Any]] = []

    for fold in range(fold_count):
        start = int(fold * fold_size)
        end = (
            int(len(X_eval))
            if fold == fold_count - 1
            else int((fold + 1) * fold_size)
        )
        if end - start < min_fold_size:
            continue

        metrics = self._evaluate(
            X_eval[start:end],
            y_eval[start:end],
            r_eval[start:end],
            regime_profile=regime_profile,
        )
        fold_results.append(
            {
                "fold": int(fold + 1),
                "start": int(start),
                "end": int(end),
                "accuracy": float(metrics.get("accuracy", 0.0)),
                "risk_adjusted_score": float(
                    metrics.get("risk_adjusted_score", 0.0)
                ),
                "sharpe_ratio": float(
                    (metrics.get("trading", {}) or {}).get(
                        "sharpe_ratio", 0.0
                    )
                ),
            }
        )

    if not fold_results:
        return {
            "enabled": False,
            "reason": "fold_construction_failed",
        }

    accs = np.array(
        [x["accuracy"] for x in fold_results], dtype=np.float64
    )
    scores = np.array(
        [x["risk_adjusted_score"] for x in fold_results],
        dtype=np.float64,
    )

    acc_mean = float(np.mean(accs))
    acc_std = float(np.std(accs))
    score_mean = float(np.mean(scores))
    score_std = float(np.std(scores))

    stability = 1.0 - (
        0.6 * (acc_std / (acc_mean + _EPS))
        + 0.4 * (score_std / (abs(score_mean) + 0.1))
    )
    stability = float(np.clip(stability, 0.0, 1.0))

    return {
        "enabled": True,
        "folds": fold_results,
        "mean_accuracy": acc_mean,
        "std_accuracy": acc_std,
        "mean_risk_adjusted_score": score_mean,
        "std_risk_adjusted_score": score_std,
        "stability_score": stability,
    }


def _build_quality_gate(
    self,
    test_metrics: dict[str, Any],
    walk_forward: dict[str, Any],
    overfit_report: dict[str, Any],
    drift_guard: dict[str, Any],
    data_quality: dict[str, Any] | None = None,
    incremental_guard: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Combine diagnostics into a single deployment recommendation."""
    trading = test_metrics.get("trading", {}) or {}
    stress_tests = test_metrics.get("stress_tests", {}) or {}
    risk_score = self._risk_adjusted_score(test_metrics)
    walk_enabled = bool(walk_forward.get("enabled", False))
    walk_stability = float(
        walk_forward.get("stability_score", 0.0) if walk_enabled else 0.0
    )

    dq = data_quality if isinstance(data_quality, dict) else {}
    symbols_checked = int(dq.get("symbols_checked", 0))
    valid_symbol_ratio = float(
        np.clip(dq.get("valid_symbol_ratio", 1.0), 0.0, 1.0)
    )
    data_quality_ok = bool(
        symbols_checked <= 0
        or valid_symbol_ratio >= _DATA_QUALITY_MIN_VALID_SYMBOL_RATIO
    )

    ig = incremental_guard if isinstance(incremental_guard, dict) else {}
    incremental_ok = not bool(ig.get("blocked", False))
    tail_stress_ok = bool(stress_tests.get("tail_guard_passed", True))
    cost_resilience_ok = bool(
        stress_tests.get("cost_resilience_passed", True)
    )
    trade_count = int(max(0, trading.get("trades", 0)))

    checks = {
        "risk_score": bool(risk_score >= 0.52),
        "profit_factor": bool(float(trading.get("profit_factor", 0.0)) >= 1.05),
        "drawdown": bool(float(trading.get("max_drawdown", 1.0)) <= 0.25),
        "trade_count": bool(trade_count >= _MIN_BASELINE_TRADES),
        "overfit": not bool(overfit_report.get("detected", False)),
        "drift": str(drift_guard.get("action", "")) != "rollback_recommended",
        "walk_forward": (not walk_enabled) or bool(walk_stability >= 0.35),
        "data_quality": bool(data_quality_ok),
        "tail_stress": bool(tail_stress_ok),
        "cost_resilience": bool(cost_resilience_ok),
        "incremental_guard": bool(incremental_ok),
    }

    reasons: list[str] = []
    if not checks["risk_score"]:
        reasons.append("risk_adjusted_score_below_threshold")
    if not checks["profit_factor"]:
        reasons.append("profit_factor_below_1.05")
    if not checks["drawdown"]:
        reasons.append("max_drawdown_above_25pct")
    if not checks["trade_count"]:
        reasons.append("insufficient_trade_count")
    if not checks["overfit"]:
        reasons.append("overfitting_detected")
    if not checks["drift"]:
        reasons.append("drift_guard_block")
    if not checks["walk_forward"]:
        reasons.append("walk_forward_instability")
    if not checks["data_quality"]:
        reasons.append("data_quality_gate_failed")
    if not checks["tail_stress"]:
        reasons.append("tail_stress_failure")
    if not checks["cost_resilience"]:
        reasons.append("cost_sensitivity_failure")
    if not checks["incremental_guard"]:
        reasons.append("incremental_regime_guard_block")

    if str(drift_guard.get("action", "")) == "rollback_recommended":
        action = "rollback_recommended"
        passed = False
    elif all(checks.values()):
        action = "deploy_ok"
        passed = True
    else:
        action = "shadow_mode_recommended"
        passed = False

    return {
        "passed": bool(passed),
        "recommended_action": action,
        "checks": checks,
        "failed_reasons": reasons,
        "risk_adjusted_score": float(risk_score),
        "walk_forward_stability": float(walk_stability),
        "data_quality_ratio": float(valid_symbol_ratio),
        "stress_tests": stress_tests if isinstance(stress_tests, dict) else {},
        "incremental_guard": ig,
    }


def _run_drift_guard(
    self,
    interval: str,
    horizon: int,
    test_metrics: dict[str, Any],
    risk_adjusted_score: float,
) -> dict[str, Any]:
    """Compare current run with previous baseline and recommend rollout mode."""
    path = self._drift_baseline_path(interval, horizon)
    baseline = self._read_json_safely(path) if path.exists() else None

    trading = test_metrics.get("trading", {}) or {}
    current = {
        "accuracy": float(test_metrics.get("accuracy", 0.0)),
        "risk_adjusted_score": float(risk_adjusted_score),
        "sharpe_ratio": float(trading.get("sharpe_ratio", 0.0)),
        "profit_factor": float(trading.get("profit_factor", 0.0)),
        "max_drawdown": float(trading.get("max_drawdown", 0.0)),
        "excess_return": float(trading.get("excess_return", 0.0)),
        "trades": int(trading.get("trades", 0)),
    }

    baseline_metrics: dict[str, Any] = {}
    score_drop = 0.0
    accuracy_drop = 0.0
    action = "no_baseline"

    if baseline:
        baseline_metrics = dict(baseline.get("metrics", {}) or {})
        prev_score = float(baseline_metrics.get("risk_adjusted_score", 0.0))
        prev_acc = float(baseline_metrics.get("accuracy", 0.0))
        score_drop = max(0.0, prev_score - current["risk_adjusted_score"])
        accuracy_drop = max(0.0, prev_acc - current["accuracy"])

        if (
            score_drop >= _DRIFT_BLOCK_SCORE_DROP
            or accuracy_drop >= _DRIFT_BLOCK_ACC_DROP
        ):
            action = "rollback_recommended"
        elif (
            score_drop >= _DRIFT_WARN_SCORE_DROP
            or accuracy_drop >= _DRIFT_WARN_ACC_DROP
        ):
            action = "shadow_mode_recommended"
        else:
            action = "deploy_ok"

    meets_floor = self._meets_baseline_quality_floor(current)
    baseline_update_block_reason = ""
    should_update_baseline = baseline is None and meets_floor

    if baseline and meets_floor:
        prev_score = float(
            (baseline.get("metrics", {}) or {}).get(
                "risk_adjusted_score", -1.0
            )
        )
        should_update_baseline = bool(
            current["risk_adjusted_score"] >= prev_score
        )
    elif not meets_floor:
        baseline_update_block_reason = "quality_floor_not_met"

    baseline_updated = False
    if should_update_baseline:
        payload = {
            "updated_at": datetime.now().isoformat(),
            "interval": str(interval),
            "horizon": int(horizon),
            "metrics": current,
        }
        baseline_updated = self._write_json_safely(path, payload)

    return {
        "baseline_path": str(path),
        "baseline_found": bool(baseline is not None),
        "baseline_metrics": baseline_metrics,
        "current_metrics": current,
        "score_drop": float(score_drop),
        "accuracy_drop": float(accuracy_drop),
        "action": action,
        "meets_quality_floor": bool(meets_floor),
        "baseline_update_block_reason": str(baseline_update_block_reason),
        "baseline_updated": bool(baseline_updated),
    }


def _trade_quality_thresholds(
    self, confidence_floor: float | None = None
) -> dict[str, float]:
    """Thresholds for confidence-first no-trade filtering."""
    precision_cfg = getattr(CONFIG, "precision", None)

    min_confidence = max(
        float(CONFIG.MIN_CONFIDENCE),
        float(
            confidence_floor
            if confidence_floor is not None
            else CONFIG.MIN_CONFIDENCE
        ),
    )
    min_agreement = 0.55
    max_entropy = 0.80
    min_edge = 0.03

    if precision_cfg is not None:
        min_agreement = max(
            min_agreement,
            float(getattr(precision_cfg, "min_agreement", min_agreement))
            - 0.10,
        )
        max_entropy = min(
            max_entropy,
            float(getattr(precision_cfg, "max_entropy", max_entropy))
            + 0.15,
        )
        min_edge = max(
            min_edge,
            float(getattr(precision_cfg, "min_edge", min_edge)) * 0.5,
        )

    min_margin = max(0.04, min(0.20, (min_edge * 0.8) + 0.02))

    return {
        "min_confidence": float(min_confidence),
        "min_agreement": float(min_agreement),
        "max_entropy": float(max_entropy),
        "min_margin": float(min_margin),
        "min_edge": float(min_edge),
    }


def _trade_masks(
    preds: np.ndarray,
    confs: np.ndarray,
    agreements: np.ndarray | None,
    entropies: np.ndarray | None,
    margins: np.ndarray | None,
    edges: np.ndarray | None,
    thresholds: dict[str, float],
) -> dict[str, np.ndarray]:
    """Build per-signal eligibility masks for no-trade filtering."""
    n = int(len(preds))
    is_up = np.asarray(preds).reshape(-1) == 2
    conf_ok = np.asarray(confs, dtype=np.float64).reshape(-1) >= float(
        thresholds["min_confidence"]
    )

    if agreements is not None and len(agreements) == n:
        agreement_ok = np.asarray(agreements, dtype=np.float64).reshape(-1) >= float(
            thresholds["min_agreement"]
        )
    else:
        agreement_ok = np.ones(n, dtype=bool)

    if entropies is not None and len(entropies) == n:
        entropy_ok = np.asarray(entropies, dtype=np.float64).reshape(-1) <= float(
            thresholds["max_entropy"]
        )
    else:
        entropy_ok = np.ones(n, dtype=bool)

    if margins is not None and len(margins) == n:
        margin_ok = np.asarray(margins, dtype=np.float64).reshape(-1) >= float(
            thresholds["min_margin"]
        )
    else:
        margin_ok = np.ones(n, dtype=bool)

    if edges is not None and len(edges) == n:
        edge_ok = np.asarray(edges, dtype=np.float64).reshape(-1) >= float(
            thresholds["min_edge"]
        )
    else:
        edge_ok = np.ones(n, dtype=bool)

    eligible = is_up & conf_ok & agreement_ok & entropy_ok & margin_ok & edge_ok

    return {
        "is_up": is_up,
        "conf": conf_ok,
        "agreement": agreement_ok,
        "entropy": entropy_ok,
        "margin": margin_ok,
        "edge": edge_ok,
        "eligible": eligible,
    }


def _build_explainability_samples(
    self,
    predictions: list[Any],
    sample_count: int,
    thresholds: dict[str, float],
    masks: dict[str, np.ndarray],
    limit: int = 8,
) -> list[dict[str, Any]]:
    """Create compact per-decision diagnostics for top-confidence samples."""
    if sample_count <= 0 or not predictions:
        return []

    n = int(min(sample_count, len(predictions)))
    rank = sorted(
        range(n),
        key=lambda i: float(getattr(predictions[i], "confidence", 0.0)),
        reverse=True,
    )
    out: list[dict[str, Any]] = []

    for idx in rank[: int(max(1, limit))]:
        pred = predictions[idx]
        pred_cls = int(getattr(pred, "predicted_class", 1))
        probs = np.asarray(
            getattr(pred, "probabilities", np.array([0.0, 0.0, 0.0])),
            dtype=np.float64,
        ).reshape(-1)
        probs3 = [float(x) for x in probs[:3]]
        while len(probs3) < 3:
            probs3.append(0.0)

        reasons: list[str] = []
        if pred_cls != 2:
            reasons.append("predicted_not_up")
        if not bool(masks["conf"][idx]):
            reasons.append("low_confidence")
        if not bool(masks["agreement"][idx]):
            reasons.append("low_agreement")
        if not bool(masks["entropy"][idx]):
            reasons.append("high_entropy")
        if not bool(masks["margin"][idx]):
            reasons.append("low_margin")
        if not bool(masks["edge"][idx]):
            reasons.append("low_edge")

        action = "TRADE_LONG" if bool(masks["eligible"][idx]) else "NO_TRADE"
        reason = reasons[0] if reasons else "passed_all_filters"

        out.append(
            {
                "index": int(idx),
                "predicted_class": int(pred_cls),
                "confidence": float(getattr(pred, "confidence", 0.0)),
                "agreement": float(getattr(pred, "agreement", 0.0)),
                "entropy": float(getattr(pred, "entropy", 1.0)),
                "margin": float(getattr(pred, "margin", 0.0)),
                "edge": float(abs(probs3[2] - probs3[0])),
                "probabilities": probs3,
                "action": action,
                "primary_reason": reason,
                "thresholds": {
                    "min_confidence": float(thresholds["min_confidence"]),
                    "min_agreement": float(thresholds["min_agreement"]),
                    "max_entropy": float(thresholds["max_entropy"]),
                    "min_margin": float(thresholds["min_margin"]),
                    "min_edge": float(thresholds["min_edge"]),
                },
            }
        )

    return out

# =========================================================================
# prepare_data (standalone, used by external callers)
# =========================================================================


def _build_trading_stress_tests(
    self,
    preds: np.ndarray,
    confs: np.ndarray,
    returns: np.ndarray,
    agreements: np.ndarray | None,
    entropies: np.ndarray | None,
    margins: np.ndarray | None,
    edges: np.ndarray | None,
    confidence_floor: float,
    masks: dict[str, np.ndarray],
) -> dict[str, Any]:
    """Build robustness diagnostics for tail events and higher trading costs.

    These checks are used to reduce live deployment of fragile models.
    """
    cost_scenarios: list[dict[str, Any]] = []
    for multiplier in _STRESS_COST_MULTIPLIERS:
        sim = self._simulate_trading(
            preds,
            confs,
            returns,
            agreements=agreements,
            entropies=entropies,
            margins=margins,
            edges=edges,
            confidence_floor=confidence_floor,
            masks=masks,
            cost_multiplier=float(multiplier),
        )
        cost_scenarios.append(
            {
                "cost_multiplier": float(multiplier),
                "profit_factor": float(sim.get("profit_factor", 0.0)),
                "max_drawdown": float(sim.get("max_drawdown", 0.0)),
                "excess_return": float(sim.get("excess_return", 0.0)),
                "trades": int(sim.get("trades", 0)),
            }
        )

    base_cost = cost_scenarios[0] if cost_scenarios else {}
    high_cost = cost_scenarios[-1] if cost_scenarios else {}
    cost_resilience_passed = True
    cost_resilience_reason = "insufficient_trades"
    if int(base_cost.get("trades", 0)) >= _MIN_BASELINE_TRADES:
        base_pf = max(float(base_cost.get("profit_factor", 0.0)), _EPS)
        high_pf = float(high_cost.get("profit_factor", 0.0))
        pf_drop = max(0.0, (base_pf - high_pf) / base_pf)
        high_dd = float(high_cost.get("max_drawdown", 1.0))
        cost_resilience_passed = bool(
            high_pf >= 0.90 and pf_drop <= 0.45 and high_dd <= 0.35
        )
        cost_resilience_reason = (
            "ok" if cost_resilience_passed else "high_cost_degradation"
        )

    returns_arr = np.asarray(returns, dtype=np.float64).reshape(-1)
    abs_returns = np.abs(returns_arr)
    tail_block: dict[str, Any] = {
        "enabled": False,
        "reason": "insufficient_samples",
        "tail_samples": 0,
        "quantile": float(_TAIL_STRESS_QUANTILE),
    }
    tail_guard_passed = True
    tail_guard_reason = "insufficient_samples"

    if len(abs_returns) >= _MIN_TAIL_STRESS_SAMPLES:
        threshold = float(
            np.percentile(abs_returns, float(_TAIL_STRESS_QUANTILE * 100.0))
        )
        tail_mask = abs_returns >= threshold
        tail_samples = int(np.sum(tail_mask))
        min_tail_samples = max(8, int(round(len(abs_returns) * 0.08)))

        if tail_samples >= min_tail_samples:
            tail_preds = np.asarray(preds)[tail_mask]
            tail_confs = np.asarray(confs)[tail_mask]
            tail_returns = returns_arr[tail_mask]

            tail_masks = {
                name: np.asarray(val, dtype=bool)[tail_mask]
                for name, val in masks.items()
                if isinstance(val, np.ndarray) and len(val) == len(tail_mask)
            }
            tail_sim = self._simulate_trading(
                tail_preds,
                tail_confs,
                tail_returns,
                agreements=(
                    np.asarray(agreements)[tail_mask]
                    if agreements is not None and len(agreements) == len(tail_mask)
                    else None
                ),
                entropies=(
                    np.asarray(entropies)[tail_mask]
                    if entropies is not None and len(entropies) == len(tail_mask)
                    else None
                ),
                margins=(
                    np.asarray(margins)[tail_mask]
                    if margins is not None and len(margins) == len(tail_mask)
                    else None
                ),
                edges=(
                    np.asarray(edges)[tail_mask]
                    if edges is not None and len(edges) == len(tail_mask)
                    else None
                ),
                confidence_floor=confidence_floor,
                masks=tail_masks if tail_masks else None,
            )

            shock_pct = float(
                np.clip(
                    np.percentile(abs_returns, 95) * 1.5,
                    _TAIL_EVENT_SHOCK_MIN_PCT,
                    _TAIL_EVENT_SHOCK_MAX_PCT,
                )
            )
            shock_sim = self._simulate_trading(
                preds,
                confs,
                returns_arr,
                agreements=agreements,
                entropies=entropies,
                margins=margins,
                edges=edges,
                confidence_floor=confidence_floor,
                masks=masks,
                cost_multiplier=1.5,
                stress_return_shock_pct=shock_pct,
            )

            tail_guard_passed = True
            tail_guard_reason = "insufficient_trades"
            if int(shock_sim.get("trades", 0)) >= _MIN_BASELINE_TRADES:
                shock_pf = float(shock_sim.get("profit_factor", 0.0))
                shock_dd = float(shock_sim.get("max_drawdown", 1.0))
                tail_guard_passed = bool(shock_pf >= 0.90 and shock_dd <= 0.35)
                tail_guard_reason = (
                    "ok" if tail_guard_passed else "tail_event_fragile"
                )

            tail_block = {
                "enabled": True,
                "reason": "ok",
                "tail_samples": int(tail_samples),
                "quantile": float(_TAIL_STRESS_QUANTILE),
                "threshold_abs_return_pct": float(threshold),
                "tail_metrics": {
                    "profit_factor": float(tail_sim.get("profit_factor", 0.0)),
                    "max_drawdown": float(tail_sim.get("max_drawdown", 0.0)),
                    "excess_return": float(tail_sim.get("excess_return", 0.0)),
                    "trades": int(tail_sim.get("trades", 0)),
                },
                "shock_pct": float(shock_pct),
                "shock_metrics": {
                    "profit_factor": float(shock_sim.get("profit_factor", 0.0)),
                    "max_drawdown": float(shock_sim.get("max_drawdown", 0.0)),
                    "excess_return": float(shock_sim.get("excess_return", 0.0)),
                    "trades": int(shock_sim.get("trades", 0)),
                },
            }
        else:
            tail_block = {
                "enabled": False,
                "reason": "insufficient_tail_samples",
                "tail_samples": int(tail_samples),
                "quantile": float(_TAIL_STRESS_QUANTILE),
            }

    return {
        "cost_scenarios": cost_scenarios,
        "cost_resilience_passed": bool(cost_resilience_passed),
        "cost_resilience_reason": str(cost_resilience_reason),
        "tail_event": tail_block,
        "tail_guard_passed": bool(tail_guard_passed),
        "tail_guard_reason": str(tail_guard_reason),
    }


def _evaluate(
    self,
    X: np.ndarray,
    y: np.ndarray,
    r: np.ndarray,
    regime_profile: dict[str, Any] | None = None,
) -> dict:
    """Evaluate model on test data."""
    try:
        from sklearn.metrics import (
            confusion_matrix,
            precision_recall_fscore_support,
        )
        metrics_backend = "sklearn"
    except Exception as e:
        log.warning(
            "scikit-learn metrics unavailable (%s); "
            "falling back to numpy metrics in evaluation",
            e,
        )
        metrics_backend = "numpy"

        def confusion_matrix(y_true, y_pred, labels):  # type: ignore[redef]
            labels_arr = np.asarray(labels, dtype=int).reshape(-1)
            out = np.zeros((len(labels_arr), len(labels_arr)), dtype=np.int64)
            pos = {int(v): i for i, v in enumerate(labels_arr.tolist())}
            for t, p in zip(y_true, y_pred, strict=False):
                ti = pos.get(int(t))
                pi = pos.get(int(p))
                if ti is None or pi is None:
                    continue
                out[ti, pi] += 1
            return out

        def precision_recall_fscore_support(  # type: ignore[redef]
            y_true,
            y_pred,
            labels=None,
            average=None,
            zero_division=0,
        ):
            del average  # current caller uses labels=[2], average=None
            labels_arr = np.asarray(labels if labels is not None else [2], dtype=int)
            p_list: list[float] = []
            r_list: list[float] = []
            f_list: list[float] = []
            s_list: list[int] = []
            for lbl in labels_arr.tolist():
                tp = int(np.sum((y_true == lbl) & (y_pred == lbl)))
                fp = int(np.sum((y_true != lbl) & (y_pred == lbl)))
                fn = int(np.sum((y_true == lbl) & (y_pred != lbl)))
                support = int(np.sum(y_true == lbl))
                precision = (
                    (tp / (tp + fp))
                    if (tp + fp) > 0
                    else float(zero_division)
                )
                recall = (
                    (tp / (tp + fn))
                    if (tp + fn) > 0
                    else float(zero_division)
                )
                denom = precision + recall
                f1 = (
                    (2.0 * precision * recall / denom)
                    if denom > 0.0
                    else float(zero_division)
                )
                p_list.append(float(precision))
                r_list.append(float(recall))
                f_list.append(float(f1))
                s_list.append(int(support))
            return (
                np.asarray(p_list, dtype=np.float64),
                np.asarray(r_list, dtype=np.float64),
                np.asarray(f_list, dtype=np.float64),
                np.asarray(s_list, dtype=np.int64),
            )

    empty_result = {
        "accuracy": 0.0,
        "trading": {},
        "stress_tests": {},
        "confusion_matrix": [],
        "up_precision": 0.0,
        "up_recall": 0.0,
        "up_f1": 0.0,
        "risk_adjusted_score": 0.0,
        "regime_profile": regime_profile or {},
        "explainability": {
            "samples": [],
            "filters": {},
        },
    }

    if len(X) == 0 or len(y) == 0:
        return empty_result

    predictions = self.ensemble.predict_batch(X)
    pred_classes = np.array(
        [p.predicted_class for p in predictions]
    )

    if len(pred_classes) == 0:
        return empty_result

    min_len = min(len(pred_classes), len(y), len(r))
    pred_classes = pred_classes[:min_len]
    y_eval = y[:min_len]
    r_eval = r[:min_len]

    cm = confusion_matrix(y_eval, pred_classes, labels=[0, 1, 2])

    # FIX EVAL: Use average='binary' style with safe extraction
    # labels=[2] with average=None returns arrays of length 1
    pr, rc, f1, _ = precision_recall_fscore_support(
        y_eval,
        pred_classes,
        labels=[2],
        average=None,
        zero_division=0,
    )

    up_precision = float(pr[0]) if len(pr) > 0 else 0.0
    up_recall = float(rc[0]) if len(rc) > 0 else 0.0
    up_f1 = float(f1[0]) if len(f1) > 0 else 0.0

    confidences = np.array(
        [p.confidence for p in predictions[:min_len]]
    )
    agreements = np.array(
        [float(getattr(p, "agreement", 0.0)) for p in predictions[:min_len]]
    )
    entropies = np.array(
        [float(getattr(p, "entropy", 1.0)) for p in predictions[:min_len]]
    )
    margins = np.array(
        [float(getattr(p, "margin", 0.0)) for p in predictions[:min_len]]
    )
    prob_up = np.array(
        [float(getattr(p, "prob_up", 0.0)) for p in predictions[:min_len]]
    )
    prob_down = np.array(
        [float(getattr(p, "prob_down", 0.0)) for p in predictions[:min_len]]
    )
    edges = np.abs(prob_up - prob_down)

    accuracy = float(np.mean(pred_classes == y_eval))

    class_acc = {}
    for c in range(CONFIG.NUM_CLASSES):
        mask = y_eval == c
        if mask.sum() > 0:
            class_acc[c] = float(np.mean(pred_classes[mask] == c))
        else:
            class_acc[c] = 0.0

    confidence_floor = self._effective_confidence_floor(regime_profile)
    thresholds = self._trade_quality_thresholds(confidence_floor)
    masks = self._trade_masks(
        pred_classes,
        confidences,
        agreements,
        entropies,
        margins,
        edges,
        thresholds,
    )
    trading_metrics = self._simulate_trading(
        pred_classes,
        confidences,
        r_eval,
        agreements=agreements,
        entropies=entropies,
        margins=margins,
        edges=edges,
        confidence_floor=confidence_floor,
        masks=masks,
    )
    stress_tests = self._build_trading_stress_tests(
        preds=pred_classes,
        confs=confidences,
        returns=r_eval,
        agreements=agreements,
        entropies=entropies,
        margins=margins,
        edges=edges,
        confidence_floor=confidence_floor,
        masks=masks,
    )

    explainability_samples = self._build_explainability_samples(
        predictions=predictions,
        sample_count=min_len,
        thresholds=thresholds,
        masks=masks,
        limit=8,
    )

    risk_adjusted_score = self._risk_adjusted_score(
        {
            "accuracy": accuracy,
            "trading": trading_metrics,
        }
    )

    return {
        "accuracy": accuracy,
        "class_accuracy": class_acc,
        "mean_confidence": (
            float(np.mean(confidences))
            if len(confidences) > 0
            else 0.0
        ),
        "trading": trading_metrics,
        "stress_tests": stress_tests,
        "confusion_matrix": cm.tolist(),
        "up_precision": up_precision,
        "up_recall": up_recall,
        "up_f1": up_f1,
        "risk_adjusted_score": float(risk_adjusted_score),
        "regime_profile": regime_profile or {},
        "explainability": {
            "samples": explainability_samples,
            "filters": thresholds,
        },
        "metrics_backend": metrics_backend,
    }


def _simulate_trading(
    self,
    preds: np.ndarray,
    confs: np.ndarray,
    returns: np.ndarray,
    agreements: np.ndarray | None = None,
    entropies: np.ndarray | None = None,
    margins: np.ndarray | None = None,
    edges: np.ndarray | None = None,
    confidence_floor: float | None = None,
    masks: dict[str, np.ndarray] | None = None,
    cost_multiplier: float = 1.0,
    stress_return_shock_pct: float = 0.0,
) -> dict:
    """Simulate trading with proper compounding and consistent units.

    FIX TRADE: Accumulates returns over the actual holding period
    instead of using a single returns[entry_idx] value.
    FIX COST: Stamp tax only on sells (China A-share rule).
    FIX m7: Use log-sum instead of np.prod to prevent overflow.
    PLUS: No-trade filtering for low-quality predictions.
    """
    thresholds = self._trade_quality_thresholds(confidence_floor)
    if masks is None:
        masks = self._trade_masks(
            preds=preds,
            confs=confs,
            agreements=agreements,
            entropies=entropies,
            margins=margins,
            edges=edges,
            thresholds=thresholds,
        )

    eligible_mask = np.asarray(masks.get("eligible", []), dtype=bool)
    up_mask = np.asarray(masks.get("is_up", []), dtype=bool)
    conf_mask = np.asarray(masks.get("conf", []), dtype=bool)
    agreement_mask = np.asarray(masks.get("agreement", []), dtype=bool)
    entropy_mask = np.asarray(masks.get("entropy", []), dtype=bool)
    margin_mask = np.asarray(masks.get("margin", []), dtype=bool)
    edge_mask = np.asarray(masks.get("edge", []), dtype=bool)

    position = eligible_mask.astype(float)

    horizon = self.prediction_horizon

    # FIX COST: Commission on both sides, stamp tax only on sell.
    cost_scale = float(max(0.0, cost_multiplier))
    entry_costs = cost_scale * (CONFIG.COMMISSION + CONFIG.SLIPPAGE)
    exit_costs = cost_scale * (
        CONFIG.COMMISSION + CONFIG.SLIPPAGE + CONFIG.STAMP_TAX
    )
    shock_decimal = float(max(0.0, stress_return_shock_pct)) / 100.0

    entries = np.diff(position, prepend=0) > 0
    exits = np.diff(position, prepend=0) < 0

    trades_decimal = []
    trade_confidences = []
    in_position = False
    entry_idx = 0

    for i in range(len(position)):
        if entries[i] and not in_position:
            in_position = True
            entry_idx = i
        elif (exits[i] or i == len(position) - 1) and in_position:
            # FIX TRADE: Accumulate returns over holding period
            # Each returns[j] is a percentage return for that period
            exit_idx = i
            holding_returns = returns[entry_idx:exit_idx + 1]

            if len(holding_returns) > 0:
                # Convert percentage returns to factors, compound them
                factors = 1.0 + holding_returns / 100.0
                safe_factors = np.maximum(factors, _EPS)
                cumulative = np.exp(np.sum(np.log(safe_factors)))
                trade_return = cumulative - 1.0 - shock_decimal
                trade_return -= entry_costs + exit_costs
                trades_decimal.append(trade_return)
                trade_confidences.append(
                    float(np.mean(confs[entry_idx:exit_idx + 1]))
                )

            in_position = False

    num_trades = len(trades_decimal)
    avg_trade_confidence = (
        float(np.mean(trade_confidences))
        if trade_confidences
        else 0.0
    )

    if num_trades > 0:
        trades = np.array(trades_decimal)

        # FIX m7: Use log-sum to prevent overflow/underflow
        safe_factors = np.maximum(1 + trades, _EPS)
        total_return_factor = np.exp(np.sum(np.log(safe_factors)))
        total_return_pct = (total_return_factor - 1) * 100

        wins = trades[trades > 0]
        losses = trades[trades < 0]

        win_rate = len(wins) / num_trades

        gross_profit = np.sum(wins) if len(wins) > 0 else 0.0
        gross_loss = (
            abs(np.sum(losses)) if len(losses) > 0 else _EPS
        )
        profit_factor = gross_profit / gross_loss

        if len(trades) > 1 and np.std(trades) > 0:
            avg_holding = max(horizon, 1)
            trades_per_year = 252 / avg_holding
            sharpe = (
                np.mean(trades)
                / np.std(trades)
                * np.sqrt(trades_per_year)
            )
        else:
            sharpe = 0.0

        # FIX m7: Use log-sum for cumulative returns too
        log_returns = np.log(safe_factors)
        cumulative_log = np.cumsum(log_returns)
        cumulative = np.exp(cumulative_log)

        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / (running_max + _EPS)
        max_drawdown = (
            abs(float(np.min(drawdown)))
            if len(drawdown) > 0
            else 0.0
        )

    else:
        total_return_pct = 0.0
        win_rate = 0.0
        profit_factor = 0.0
        sharpe = 0.0
        max_drawdown = 0.0

    if len(returns) > 0 and horizon > 0:
        period_returns = returns / 100.0
        indices = list(
            range(0, len(period_returns), max(horizon, 1))
        )
        bh_returns = period_returns[indices]

        # FIX m7: Use log-sum for buy-hold too
        safe_bh = np.maximum(1 + bh_returns, _EPS)
        buyhold_factor = np.exp(np.sum(np.log(safe_bh)))
        buyhold_return_pct = (buyhold_factor - 1) * 100
    else:
        buyhold_return_pct = 0.0

    signal_count = int(len(preds))
    up_signals = int(np.sum(up_mask)) if len(up_mask) == signal_count else 0
    eligible_signals = (
        int(np.sum(eligible_mask)) if len(eligible_mask) == signal_count else 0
    )
    rejected_low_conf = (
        int(np.sum(up_mask & ~conf_mask))
        if len(conf_mask) == signal_count and up_signals > 0
        else 0
    )
    rejected_low_agreement = (
        int(np.sum(up_mask & ~agreement_mask))
        if len(agreement_mask) == signal_count and up_signals > 0
        else 0
    )
    rejected_high_entropy = (
        int(np.sum(up_mask & ~entropy_mask))
        if len(entropy_mask) == signal_count and up_signals > 0
        else 0
    )
    rejected_low_margin = (
        int(np.sum(up_mask & ~margin_mask))
        if len(margin_mask) == signal_count and up_signals > 0
        else 0
    )
    rejected_low_edge = (
        int(np.sum(up_mask & ~edge_mask))
        if len(edge_mask) == signal_count and up_signals > 0
        else 0
    )

    trade_coverage = float(eligible_signals / max(up_signals, 1))
    no_trade_rate = float(1.0 - (eligible_signals / max(signal_count, 1)))

    return {
        "total_return": float(total_return_pct),
        "buyhold_return": float(buyhold_return_pct),
        "excess_return": float(
            total_return_pct - buyhold_return_pct
        ),
        "trades": num_trades,
        "win_rate": float(win_rate),
        "profit_factor": float(profit_factor),
        "sharpe_ratio": float(sharpe),
        "max_drawdown": float(max_drawdown),
        "avg_trade_confidence": float(avg_trade_confidence),
        "trade_coverage": float(trade_coverage),
        "no_trade_rate": float(no_trade_rate),
        "signals_total": int(signal_count),
        "signals_up": int(up_signals),
        "signals_eligible": int(eligible_signals),
        "rejected_low_confidence": int(rejected_low_conf),
        "rejected_low_agreement": int(rejected_low_agreement),
        "rejected_high_entropy": int(rejected_high_entropy),
        "rejected_low_margin": int(rejected_low_margin),
        "rejected_low_edge": int(rejected_low_edge),
        "cost_multiplier": float(cost_scale),
        "stress_return_shock_pct": float(max(0.0, stress_return_shock_pct)),
        "filters": {
            "min_confidence": float(thresholds["min_confidence"]),
            "min_agreement": float(thresholds["min_agreement"]),
            "max_entropy": float(thresholds["max_entropy"]),
            "min_margin": float(thresholds["min_margin"]),
            "min_edge": float(thresholds["min_edge"]),
        },
    }

# =========================================================================
# =========================================================================
