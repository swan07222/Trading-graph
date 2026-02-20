from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


def _load_release_preflight_module():
    path = Path("scripts/release_preflight.py").resolve()
    spec = spec_from_file_location("release_preflight", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load scripts/release_preflight.py")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_release_preflight_default_includes_lint_and_pytest(monkeypatch):
    module = _load_release_preflight_module()
    captured_steps: list[str] = []

    def fake_run_step(name: str, cmd: list[str]):
        captured_steps.append(name)
        return {
            "name": name,
            "command": cmd,
            "exit_code": 0,
            "duration_seconds": 0.001,
            "ok": True,
            "stdout": "",
            "stderr": "",
        }

    monkeypatch.setattr(module, "_run_step", fake_run_step)
    monkeypatch.setattr(
        module.sys,
        "argv",
        [
            "release_preflight.py",
            "--skip-health",
            "--skip-doctor",
            "--skip-typecheck",
            "--skip-regulatory",
            "--skip-ha-dr",
        ],
    )

    assert module.main() == 0
    assert captured_steps[:2] == ["ruff_lint", "pytest_strict"]


def test_release_preflight_allow_test_warnings(monkeypatch):
    module = _load_release_preflight_module()
    captured_pytest_cmd: list[str] = []

    def fake_run_step(name: str, cmd: list[str]):
        if name == "pytest_strict":
            captured_pytest_cmd.extend(cmd)
        return {
            "name": name,
            "command": cmd,
            "exit_code": 0,
            "duration_seconds": 0.001,
            "ok": True,
            "stdout": "",
            "stderr": "",
        }

    monkeypatch.setattr(module, "_run_step", fake_run_step)
    monkeypatch.setattr(
        module.sys,
        "argv",
        [
            "release_preflight.py",
            "--allow-test-warnings",
            "--skip-health",
            "--skip-doctor",
            "--skip-typecheck",
            "--skip-regulatory",
            "--skip-ha-dr",
        ],
    )

    assert module.main() == 0
    assert "-W" not in captured_pytest_cmd
    assert "error" not in captured_pytest_cmd


def test_release_preflight_returns_failure_when_any_step_fails(monkeypatch):
    module = _load_release_preflight_module()

    def fake_run_step(name: str, cmd: list[str]):
        ok = name != "ruff_lint"
        return {
            "name": name,
            "command": cmd,
            "exit_code": 0 if ok else 1,
            "duration_seconds": 0.001,
            "ok": ok,
            "stdout": "",
            "stderr": "",
        }

    monkeypatch.setattr(module, "_run_step", fake_run_step)
    monkeypatch.setattr(
        module.sys,
        "argv",
        [
            "release_preflight.py",
            "--skip-health",
            "--skip-doctor",
            "--skip-typecheck",
            "--skip-regulatory",
            "--skip-ha-dr",
        ],
    )

    assert module.main() == 1


def test_release_preflight_quick_profile_skips_slow_steps(monkeypatch):
    module = _load_release_preflight_module()
    captured_steps: list[str] = []

    def fake_run_step(name: str, cmd: list[str]):
        captured_steps.append(name)
        return {
            "name": name,
            "command": cmd,
            "exit_code": 0,
            "duration_seconds": 0.001,
            "ok": True,
            "stdout": "",
            "stderr": "",
        }

    monkeypatch.setattr(module, "_run_step", fake_run_step)
    monkeypatch.setattr(
        module.sys,
        "argv",
        [
            "release_preflight.py",
            "--profile",
            "quick",
            "--skip-artifact-guard",
            "--skip-health",
            "--skip-doctor",
            "--skip-typecheck",
        ],
    )

    assert module.main() == 0
    assert captured_steps == []
