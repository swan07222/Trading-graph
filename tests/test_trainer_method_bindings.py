from __future__ import annotations

from models.trainer import Trainer


def test_trainer_has_temporal_split_integrity_binding() -> None:
    assert hasattr(Trainer, "_validate_temporal_split_integrity")
    assert callable(Trainer._validate_temporal_split_integrity)
