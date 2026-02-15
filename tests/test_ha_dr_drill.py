from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


def _load_ha_dr_module():
    path = Path("scripts/ha_dr_drill.py").resolve()
    spec = spec_from_file_location("ha_dr_drill", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load scripts/ha_dr_drill.py")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_ha_dr_drill_passes_for_sqlite_backend(tmp_path: Path):
    mod = _load_ha_dr_module()
    report = mod.run_ha_dr_drill(
        backend="sqlite",
        cluster="test_cluster",
        lease_path=tmp_path / "lease.db",
        ttl_seconds=5.0,
        stale_takeover=False,
    )
    assert report["status"] == "pass"
    assert report["failed_steps"] == []
