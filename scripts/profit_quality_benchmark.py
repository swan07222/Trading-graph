from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from config.settings import CONFIG
from models.trainer import Trainer


def _sigmoid(x: float) -> float:
    z = float(np.clip(x, -20.0, 20.0))
    return float(1.0 / (1.0 + np.exp(-z)))


def _make_synthetic_dataset(
    samples: int,
    *,
    seed: int,
    seq_len: int,
    features: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(int(seed))
    latent = rng.normal(0.0, 1.0, int(samples))
    trend = np.cumsum(rng.normal(0.0, 0.04, int(samples)))
    signal = latent + (0.4 * trend)

    X = rng.normal(0.0, 1.0, (int(samples), int(seq_len), int(features))).astype(
        np.float32
    )
    X[:, -1, 0] = signal.astype(np.float32)
    X[:, -1, 1] = (0.6 * latent + rng.normal(0.0, 0.6, int(samples))).astype(
        np.float32
    )
    X[:, -1, 2] = trend.astype(np.float32)

    y_score = signal + rng.normal(0.0, 0.65, int(samples))
    y = np.where(y_score > 0.35, 2, np.where(y_score < -0.35, 0, 1)).astype(np.int64)

    r = np.where(
        y == 2,
        rng.normal(0.42, 0.34, int(samples)),
        rng.normal(-0.28, 0.38, int(samples)),
    ).astype(np.float32)
    r = np.clip(r, -3.0, 3.0).astype(np.float32)
    return X, y, r


class _SyntheticEnsemble:
    def predict_batch(self, X: np.ndarray) -> list[SimpleNamespace]:
        out: list[SimpleNamespace] = []
        for row in np.asarray(X, dtype=np.float32):
            recent = row[-5:]
            feature_0 = float(np.mean(recent[:, 0]))
            feature_1 = float(np.mean(recent[:, 1]))
            feature_2 = float(np.mean(recent[:, 2]))

            decision = (0.95 * feature_0) + (0.35 * feature_2) - (0.25 * feature_1)
            p_up = _sigmoid(decision * 1.9)
            p_down = _sigmoid(-decision * 1.7)
            p_flat = float(max(0.02, 1.0 - max(p_up, p_down)))

            probs = np.asarray([p_down, p_flat, p_up], dtype=np.float64)
            probs = probs / max(1e-8, float(np.sum(probs)))
            pred_class = int(np.argmax(probs))

            ordered = np.sort(probs)[::-1]
            margin = float(max(0.0, ordered[0] - ordered[1]))
            uncertainty = 1.0 - float(abs(probs[2] - probs[0]))
            # Intentionally over-confident near uncertain regions.
            confidence = float(
                np.clip((0.62 + (0.32 * float(ordered[0]))) + (0.22 * uncertainty), 0.0, 0.99)
            )

            entropy = float(
                -np.sum(probs * np.log(np.maximum(1e-8, probs))) / np.log(3.0)
            )
            agreement = float(np.clip(confidence - (0.12 * uncertainty), 0.0, 1.0))

            out.append(
                SimpleNamespace(
                    predicted_class=int(pred_class),
                    confidence=float(confidence),
                    agreement=float(agreement),
                    entropy=float(entropy),
                    margin=float(margin),
                    prob_up=float(probs[2]),
                    prob_down=float(probs[0]),
                    probabilities=probs.astype(np.float32),
                )
            )
        return out


def _slice_metrics(metrics: dict[str, Any], walk: dict[str, Any]) -> dict[str, Any]:
    trading = metrics.get("trading", {}) or {}
    calibration = metrics.get("calibration", {}) or {}
    return {
        "accuracy": float(metrics.get("accuracy", 0.0)),
        "risk_adjusted_score": float(metrics.get("risk_adjusted_score", 0.0)),
        "profit_factor": float(trading.get("profit_factor", 0.0)),
        "sharpe_ratio": float(trading.get("sharpe_ratio", 0.0)),
        "max_drawdown": float(trading.get("max_drawdown", 0.0)),
        "excess_return": float(trading.get("excess_return", 0.0)),
        "trades": int(trading.get("trades", 0)),
        "trade_coverage": float(trading.get("trade_coverage", 0.0)),
        "ece": float(
            calibration.get(
                "ece_after",
                calibration.get("ece_before", 0.0),
            )
        ),
        "brier": float(
            calibration.get(
                "brier_after",
                calibration.get("brier_before", 0.0),
            )
        ),
        "walk_stability": float(walk.get("stability_score", 0.0)),
        "walk_selection_score": float(walk.get("selection_score", 0.0)),
        "walk_downside_stability": float(walk.get("downside_stability", 0.0)),
    }


def _build_deltas(before: dict[str, Any], after: dict[str, Any]) -> dict[str, float]:
    return {
        "risk_adjusted_score_delta": float(after["risk_adjusted_score"] - before["risk_adjusted_score"]),
        "accuracy_delta": float(after["accuracy"] - before["accuracy"]),
        "profit_factor_delta": float(after["profit_factor"] - before["profit_factor"]),
        "sharpe_ratio_delta": float(after["sharpe_ratio"] - before["sharpe_ratio"]),
        "excess_return_delta": float(after["excess_return"] - before["excess_return"]),
        "trade_coverage_delta": float(after["trade_coverage"] - before["trade_coverage"]),
        "walk_selection_score_delta": float(after["walk_selection_score"] - before["walk_selection_score"]),
        "walk_stability_delta": float(after["walk_stability"] - before["walk_stability"]),
        # Positive means improvement for risk/error metrics.
        "max_drawdown_improvement": float(before["max_drawdown"] - after["max_drawdown"]),
        "ece_improvement": float(before["ece"] - after["ece"]),
        "brier_improvement": float(before["brier"] - after["brier"]),
    }


def run_benchmark(samples: int, seed: int) -> dict[str, Any]:
    total_samples = int(max(900, samples))
    seq_len = int(CONFIG.SEQUENCE_LENGTH)
    features = 6
    X, y, r = _make_synthetic_dataset(
        total_samples,
        seed=int(seed),
        seq_len=seq_len,
        features=features,
    )

    split_val = int(total_samples * 0.60)
    split_test = int(total_samples * 0.80)
    X_val = X[split_val:split_test]
    y_val = y[split_val:split_test]
    r_val = r[split_val:split_test]
    X_test = X[split_test:]
    y_test = y[split_test:]
    r_test = r[split_test:]

    trainer = Trainer.__new__(Trainer)
    trainer.ensemble = _SyntheticEnsemble()
    trainer.prediction_horizon = int(max(1, CONFIG.PREDICTION_HORIZON))
    trainer.interval = "1m"

    baseline_eval = trainer._evaluate(
        X_test,
        y_test,
        r_test,
        regime_profile={},
    )
    baseline_walk = trainer._walk_forward_validate(
        X_val,
        y_val,
        r_val,
        X_test,
        y_test,
        r_test,
        regime_profile={},
        calibration_map=None,
    )

    calibration_report = trainer._build_confidence_calibration(X_val, y_val)
    improved_eval = trainer._evaluate(
        X_test,
        y_test,
        r_test,
        regime_profile={},
        calibration_map=calibration_report,
    )
    improved_walk = trainer._walk_forward_validate(
        X_val,
        y_val,
        r_val,
        X_test,
        y_test,
        r_test,
        regime_profile={},
        calibration_map=calibration_report,
    )

    before = _slice_metrics(baseline_eval, baseline_walk)
    after = _slice_metrics(improved_eval, improved_walk)
    deltas = _build_deltas(before, after)
    return {
        "meta": {
            "samples": int(total_samples),
            "seed": int(seed),
            "sequence_length": int(seq_len),
            "prediction_horizon": int(trainer.prediction_horizon),
            "note": "Synthetic benchmark for calibration + walk-forward quality pass.",
        },
        "calibration_report": calibration_report,
        "before": before,
        "after": after,
        "deltas": deltas,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run profit-quality benchmark before/after calibration and walk-forward selection upgrades."
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=1800,
        help="Number of synthetic samples for the benchmark (min 900).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for synthetic dataset generation.",
    )
    parser.add_argument(
        "--output",
        default="analysis/profit_quality_benchmark.json",
        help="Output JSON path.",
    )
    args = parser.parse_args(argv)

    report = run_benchmark(samples=int(args.samples), seed=int(args.seed))
    rendered = json.dumps(report, indent=2)
    print(rendered)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(rendered + "\n", encoding="utf-8")
    print(f"benchmark report written: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
