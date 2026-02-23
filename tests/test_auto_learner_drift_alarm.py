from __future__ import annotations

from types import SimpleNamespace

from models.auto_learner import ContinuousLearner


def test_emit_model_drift_alarm_calls_execution_trigger(monkeypatch) -> None:
    learner = ContinuousLearner.__new__(ContinuousLearner)
    warnings: list[str] = []
    learner.progress = SimpleNamespace(add_warning=lambda msg: warnings.append(str(msg)))

    import trading.executor as exec_mod

    calls = {"count": 0, "reason": "", "severity": ""}
    old = exec_mod.ExecutionEngine.trigger_model_drift_alarm

    @classmethod
    def _fake_trigger(cls, reason, *, severity="critical", metadata=None) -> int:  # noqa: ARG001
        calls["count"] += 1
        calls["reason"] = str(reason)
        calls["severity"] = str(severity)
        return 1

    monkeypatch.setattr(
        exec_mod.ExecutionEngine,
        "trigger_model_drift_alarm",
        _fake_trigger,
        raising=True,
    )
    try:
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
    finally:
        monkeypatch.setattr(
            exec_mod.ExecutionEngine,
            "trigger_model_drift_alarm",
            old,
            raising=True,
        )

    assert out is True
    assert calls["count"] == 1
    assert calls["severity"] == "critical"
    assert "unit_cycle" in calls["reason"]
    assert warnings

