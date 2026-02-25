from __future__ import annotations

from types import SimpleNamespace

from models.auto_learner import ContinuousLearner


def test_emit_model_drift_alarm_records_warning_only() -> None:
    learner = ContinuousLearner.__new__(ContinuousLearner)
    warnings: list[str] = []
    learner.progress = SimpleNamespace(add_warning=lambda msg: warnings.append(str(msg)))

    out = learner._emit_model_drift_alarm_if_needed(
        {
            "drift_guard": {
                "action": "rollback_recommended",
                "score_drop": 0.25,
                "accuracy_drop": 0.12,
            },
            "quality_gate": {
                "failed_reasons": ["drift_guard_block"],
            },
        },
        context="unit_cycle",
    )

    assert out is True
    assert warnings
    assert "unit_cycle" in warnings[0]
