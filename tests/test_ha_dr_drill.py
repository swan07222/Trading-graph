import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest


def _load_ha_dr_module():
    path = Path("scripts/ha_dr_drill.py").resolve()
    module_name = "ha_dr_drill"
    spec = spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load scripts/ha_dr_drill.py")
    module = module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("backend", "filename"),
    [
        ("file", "lease.json"),
        ("sqlite", "lease.db"),
    ],
)
def test_run_ha_dr_drill_passes_across_backends(
    tmp_path: Path,
    backend: str,
    filename: str,
) -> None:
    module = _load_ha_dr_module()
    report = module.run_ha_dr_drill(
        backend=backend,
        cluster="unit-test-cluster",
        lease_path=tmp_path / filename,
        ttl_seconds=1.0,
        stale_takeover=False,
    )
    assert report["status"] == "pass"
    assert report["backend"] == backend
    assert report["failed_steps"] == []
