from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


def _load_module_size_gate_module():
    path = Path("scripts/module_size_gate.py").resolve()
    spec = spec_from_file_location("module_size_gate", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load scripts/module_size_gate.py")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_collect_oversized_modules_flags_large_file(tmp_path: Path) -> None:
    gate = _load_module_size_gate_module()
    py_file = tmp_path / "big.py"
    py_file.write_text(
        "a = 1\nb = 2\nc = 3\nd = 4\n",
        encoding="utf-8",
    )

    oversized = gate.collect_oversized_modules((str(py_file),), max_lines=3)
    normalized = str(py_file).replace("\\", "/")
    assert oversized == {normalized: 4}


def test_module_size_baseline_roundtrip(tmp_path: Path) -> None:
    gate = _load_module_size_gate_module()
    baseline = tmp_path / "baseline.txt"
    sample = {"a.py": 1200, "b.py": 2000}

    gate.save_baseline(baseline, sample)
    loaded = gate.load_baseline(baseline)
    assert loaded == sample


def test_module_size_collect_line_counts(tmp_path: Path) -> None:
    gate = _load_module_size_gate_module()
    py_file = tmp_path / "small.py"
    py_file.write_text("x = 1\ny = 2\n", encoding="utf-8")
    counts = gate.collect_module_line_counts((str(py_file),))
    normalized = str(py_file).replace("\\", "/")
    assert counts[normalized] == 2


def test_module_size_budget_roundtrip(tmp_path: Path) -> None:
    gate = _load_module_size_gate_module()
    budget = tmp_path / "budget.json"
    sample = {"a.py": 1000, "b.py": 1500}
    gate.save_budget(budget, sample)
    loaded = gate.load_budget(budget)
    assert loaded == sample
