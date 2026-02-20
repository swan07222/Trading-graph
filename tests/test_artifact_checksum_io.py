from pathlib import Path

import pytest

from utils.atomic_io import (
    artifact_checksum_path,
    atomic_pickle_dump,
    pickle_load,
    verify_checksum_sidecar,
)


def test_atomic_pickle_dump_writes_checksum_sidecar(tmp_path: Path):
    target = tmp_path / "obj.pkl"
    atomic_pickle_dump(target, {"x": 1})
    sidecar = artifact_checksum_path(target)
    assert sidecar.exists()
    assert verify_checksum_sidecar(target, require=True) is True


def test_pickle_load_with_checksum_verification_blocks_tamper(tmp_path: Path):
    target = tmp_path / "obj.pkl"
    atomic_pickle_dump(target, {"x": 1})
    target.write_bytes(b"tampered")

    with pytest.raises(ValueError):
        pickle_load(
            target,
            verify_checksum=True,
            require_checksum=True,
            allow_unsafe=True,
        )


def test_pickle_load_requires_explicit_unsafe_opt_in(tmp_path: Path):
    target = tmp_path / "obj.pkl"
    atomic_pickle_dump(target, {"x": 1})
    with pytest.raises(ValueError):
        pickle_load(target)
