from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


def _load_typecheck_gate_module():
    path = Path("scripts/typecheck_gate.py").resolve()
    spec = spec_from_file_location("typecheck_gate", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load scripts/typecheck_gate.py")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_parse_mypy_errors_extracts_stable_issue_keys():
    gate = _load_typecheck_gate_module()
    raw = (
        "trading\\\\alerts.py:321: error: Item \"None\" of \"datetime | None\" "
        "has no attribute \"isoformat\"  [union-attr]\n"
        "Found 1 error in 1 file (checked 1 source file)\n"
    )
    out = gate.parse_mypy_errors(raw)
    assert (
        "trading/alerts.py:321:union-attr:Item \"None\" of \"datetime | None\" "
        "has no attribute \"isoformat\""
    ) in out


def test_baseline_roundtrip(tmp_path: Path):
    gate = _load_typecheck_gate_module()
    baseline_path = tmp_path / "baseline.txt"
    sample = {
        "a.py:1:arg-type:bad arg",
        "b.py:2:return-value:bad return",
    }

    gate.save_baseline_entries(baseline_path, sample)
    loaded = gate.load_baseline_entries(baseline_path)
    assert loaded == sample


def test_main_accepts_existing_empty_baseline(tmp_path: Path, monkeypatch):
    gate = _load_typecheck_gate_module()
    baseline_path = tmp_path / "baseline.txt"
    gate.save_baseline_entries(baseline_path, set())

    monkeypatch.setattr(gate.importlib.util, "find_spec", lambda _: object())
    monkeypatch.setattr(gate, "run_mypy", lambda _targets, _flags: (0, "", set()))
    monkeypatch.setattr(
        gate.sys,
        "argv",
        ["typecheck_gate.py", "--baseline", str(baseline_path)],
    )

    assert gate.main() == 0
