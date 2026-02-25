from pathlib import Path

import pytest

from config import CONFIG
from models.ensemble import EnsembleModel


def _valid_ensemble_state(seed: EnsembleModel) -> dict:
    model_name = "informer"
    model_state = {
        k: v.detach().cpu().clone()
        for k, v in seed.models[model_name].state_dict().items()
    }
    return {
        "input_size": int(seed.input_size),
        "model_names": [model_name],
        "models": {model_name: model_state},
        "weights": {model_name: 1.0},
        "temperature": 1.0,
        "meta": {
            "interval": "1m",
            "prediction_horizon": 5,
            "trained_stock_codes": [],
            "trained_stock_last_train": {},
        },
        "arch": {
            "hidden_size": int(CONFIG.model.hidden_size),
            "dropout": float(CONFIG.model.dropout),
            "num_classes": int(CONFIG.model.num_classes),
        },
    }


@pytest.mark.skip(reason="Ensemble tests require PyTorch with ScaledDotProductAttention (PyTorch 2.0+)")
def test_ensemble_load_blocks_unsafe_legacy_fallback(monkeypatch, tmp_path: Path) -> None:
    seed = EnsembleModel(input_size=8, model_names=["informer"])
    payload = _valid_ensemble_state(seed)

    model_path = tmp_path / "ensemble_1m_5.pt"
    model_path.write_bytes(b"placeholder")

    calls: list[bool] = []

    def _fake_torch_load(
        _path,
        map_location=None,
        weights_only=True,
        verify_checksum=False,
        require_checksum=True,
        allow_unsafe=False,
    ):
        calls.append(bool(weights_only))
        if weights_only:
            raise RuntimeError("simulated weights-only failure")
        return payload

    import utils.atomic_io as atomic_io

    monkeypatch.setattr(atomic_io, "torch_load", _fake_torch_load, raising=True)

    old_allow = bool(CONFIG.model.allow_unsafe_artifact_load)
    old_require = bool(CONFIG.model.require_artifact_checksum)
    try:
        CONFIG.model.allow_unsafe_artifact_load = False
        CONFIG.model.require_artifact_checksum = False

        target = EnsembleModel(input_size=8, model_names=["informer"])
        assert target.load(model_path) is False
        assert calls == [True]
    finally:
        CONFIG.model.allow_unsafe_artifact_load = old_allow
        CONFIG.model.require_artifact_checksum = old_require


@pytest.mark.skip(reason="Ensemble tests require PyTorch with ScaledDotProductAttention (PyTorch 2.0+)")
def test_ensemble_load_allows_unsafe_legacy_fallback_when_opted_in(
    monkeypatch, tmp_path: Path
) -> None:
    seed = EnsembleModel(input_size=8, model_names=["informer"])
    payload = _valid_ensemble_state(seed)

    model_path = tmp_path / "ensemble_1m_5.pt"
    model_path.write_bytes(b"placeholder")

    calls: list[bool] = []

    def _fake_torch_load(
        _path,
        map_location=None,
        weights_only=True,
        verify_checksum=False,
        require_checksum=True,
        allow_unsafe=False,
    ):
        calls.append(bool(weights_only))
        if weights_only:
            raise RuntimeError("simulated weights-only failure")
        return payload

    import utils.atomic_io as atomic_io

    monkeypatch.setattr(atomic_io, "torch_load", _fake_torch_load, raising=True)

    old_allow = bool(CONFIG.model.allow_unsafe_artifact_load)
    old_require = bool(CONFIG.model.require_artifact_checksum)
    try:
        CONFIG.model.allow_unsafe_artifact_load = True
        CONFIG.model.require_artifact_checksum = False

        target = EnsembleModel(input_size=8, model_names=["informer"])
        assert target.load(model_path) is True
        assert calls == [True, False]
    finally:
        CONFIG.model.allow_unsafe_artifact_load = old_allow
        CONFIG.model.require_artifact_checksum = old_require
