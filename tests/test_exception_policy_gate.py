from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


def _load_exception_policy_gate_module():
    path = Path("scripts/exception_policy_gate.py").resolve()
    spec = spec_from_file_location("exception_policy_gate", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load scripts/exception_policy_gate.py")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_collect_exception_policy_issues_detects_broad_and_silent(tmp_path: Path) -> None:
    gate = _load_exception_policy_gate_module()
    sample = tmp_path / "sample.py"
    sample.write_text(
        "try:\n"
        "    raise RuntimeError('x')\n"
        "except Exception:\n"
        "    pass\n",
        encoding="utf-8",
    )

    issues, parse_errors = gate.collect_exception_policy_issues((str(sample),))
    assert parse_errors == []
    normalized = str(sample).replace("\\", "/")
    assert f"{normalized}:3:broad-except" in issues
    assert f"{normalized}:3:silent-pass" in issues


def test_collect_exception_policy_issues_handles_utf8_bom(tmp_path: Path) -> None:
    gate = _load_exception_policy_gate_module()
    sample = tmp_path / "sample_bom.py"
    sample.write_text(
        "try:\n"
        "    raise RuntimeError('x')\n"
        "except Exception:\n"
        "    pass\n",
        encoding="utf-8-sig",
    )

    issues, parse_errors = gate.collect_exception_policy_issues((str(sample),))
    assert parse_errors == []
    normalized = str(sample).replace("\\", "/")
    assert f"{normalized}:3:broad-except" in issues
    assert f"{normalized}:3:silent-pass" in issues


def test_exception_policy_baseline_roundtrip(tmp_path: Path) -> None:
    gate = _load_exception_policy_gate_module()
    baseline = tmp_path / "baseline.txt"
    sample = {
        "a.py:10:broad-except",
        "b.py:20:silent-pass",
    }
    gate.save_baseline_entries(baseline, sample)
    loaded = gate.load_baseline_entries(baseline)
    assert loaded == sample


def test_exception_policy_summarize_issue_counts() -> None:
    gate = _load_exception_policy_gate_module()
    issues = {
        "a.py:10:broad-except",
        "a.py:11:broad-except",
        "a.py:20:silent-pass",
        "b.py:1:bare-except",
    }
    summary = gate.summarize_issue_counts(issues)
    assert summary["a.py"]["broad-except"] == 2
    assert summary["a.py"]["silent-pass"] == 1
    assert summary["b.py"]["bare-except"] == 1


def test_exception_policy_budget_roundtrip(tmp_path: Path) -> None:
    gate = _load_exception_policy_gate_module()
    budget_path = tmp_path / "budget.json"
    sample = {
        "a.py": {"broad-except": 3, "silent-pass": 1},
        "b.py": {"bare-except": 0},
    }
    gate.save_budget(budget_path, sample)
    loaded = gate.load_budget(budget_path)
    assert loaded == sample
