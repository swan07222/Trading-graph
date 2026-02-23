from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest


def _load_snapshot_module():
    path = Path("scripts/deployment_snapshot.py").resolve()
    spec = spec_from_file_location("deployment_snapshot", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load scripts/deployment_snapshot.py")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_snapshot_create_and_restore_roundtrip(tmp_path: Path) -> None:
    mod = _load_snapshot_module()
    root = tmp_path / "repo"
    root.mkdir(parents=True)

    cfg = root / "config.json"
    cfg.write_text("{\"mode\": \"paper\"}\n", encoding="utf-8")
    policy = root / "config" / "security_policy.json"
    policy.parent.mkdir(parents=True, exist_ok=True)
    policy.write_text("{\"enabled\": true}\n", encoding="utf-8")

    out = mod.create_snapshot(
        root=root,
        snapshot_dir=(tmp_path / "backups"),
        include_paths=["config.json", "config/security_policy.json"],
        include_models=False,
        tag="test",
    )
    archive = Path(out["archive"])
    assert archive.exists()

    cfg.write_text("{\"mode\": \"live\"}\n", encoding="utf-8")

    dry = mod.restore_snapshot(root=root, archive=archive, dry_run=True, confirm=False)
    assert dry["restored_count"] == 2

    applied = mod.restore_snapshot(root=root, archive=archive, dry_run=False, confirm=True)
    assert applied["restored_count"] == 2
    assert "\"paper\"" in cfg.read_text(encoding="utf-8")


def test_safe_member_path_rejects_traversal() -> None:
    mod = _load_snapshot_module()
    with pytest.raises(ValueError):
        mod._safe_member_path("../outside.txt")


def test_post_restore_verification_runs_preflight(monkeypatch, tmp_path: Path) -> None:
    mod = _load_snapshot_module()
    captured: list[tuple[str, list[str]]] = []

    def fake_run_step(name: str, cmd: list[str], cwd: Path):
        captured.append((name, cmd))
        return {
            "name": name,
            "command": cmd,
            "exit_code": 0,
            "duration_seconds": 0.001,
            "ok": True,
            "stdout": "",
            "stderr": "",
        }

    monkeypatch.setattr(mod, "_run_step", fake_run_step, raising=True)
    out = mod._run_post_restore_verification(
        root=tmp_path,
        profile="quick",
        observability_url="http://127.0.0.1:9090",
        soak_minutes=0.0,
    )

    assert out["status"] == "pass"
    assert captured
    assert captured[0][0] == "release_preflight"
    assert "--profile" in captured[0][1]
    assert "quick" in captured[0][1]


def test_post_restore_live_soak_requires_allow_live(tmp_path: Path) -> None:
    mod = _load_snapshot_module()
    out = mod._run_post_restore_verification(
        root=tmp_path,
        profile="quick",
        soak_minutes=1.0,
        soak_mode="live",
        allow_live=False,
    )

    assert out["status"] == "fail"
    names = [step["name"] for step in out["steps"]]
    assert "soak_smoke" in names
