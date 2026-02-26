import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


def _load_feature_score_gate_module():
    path = Path("scripts/feature_score_gate.py").resolve()
    module_name = "feature_score_gate"
    spec = spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load scripts/feature_score_gate.py")
    module = module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_feature_score_gate_build_report_has_30_plus_features() -> None:
    module = _load_feature_score_gate_module()
    report = module.build_feature_score_report(Path(".").resolve())
    assert int(report["feature_count"]) >= 30
    assert float(report["overall_score"]) >= 9.5
    names = {row["name"] for row in report["features"]}
    assert "Real-time quote robustness" in names
    assert "LLM assistant integration" in names
    assert "Deployment/rollback/DR readiness" in names


def test_feature_score_gate_main_fails_when_threshold_too_high(monkeypatch) -> None:
    module = _load_feature_score_gate_module()
    monkeypatch.setattr(
        sys,
        "argv",
        ["feature_score_gate.py", "--min-overall", "10.1", "--min-feature", "10.1"],
    )
    assert module.main() == 1
