from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


def _load_typecheck_strict_gate_module():
    path = Path("scripts/typecheck_strict_gate.py").resolve()
    spec = spec_from_file_location("typecheck_strict_gate", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load scripts/typecheck_strict_gate.py")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_parse_mypy_errors_extracts_stable_issue_keys():
    gate = _load_typecheck_strict_gate_module()
    raw = (
        "scripts\\\\module_size_gate.py:42: error: "
        "Incompatible return value type (got \"str\", expected \"int\") [return-value]\n"
    )
    out = gate.parse_mypy_errors(raw)
    assert (
        "scripts/module_size_gate.py:42:return-value:"
        "Incompatible return value type (got \"str\", expected \"int\")"
    ) in out


def test_baseline_roundtrip(tmp_path: Path):
    gate = _load_typecheck_strict_gate_module()
    baseline_path = tmp_path / "strict-baseline.txt"
    sample = {
        "a.py:1:arg-type:bad arg",
        "b.py:2:return-value:bad return",
    }

    gate.save_baseline_entries(baseline_path, sample)
    loaded = gate.load_baseline_entries(baseline_path)
    assert loaded == sample
