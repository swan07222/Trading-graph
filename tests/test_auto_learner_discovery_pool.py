from __future__ import annotations

import time

from models.auto_learner import StockRotator


def test_rotator_keeps_full_universe_and_persists_without_truncation(monkeypatch):
    universe = [f"{600000 + i:06d}" for i in range(5200)]
    new_listing = ["001399"]

    monkeypatch.setattr(
        "data.universe.get_universe_codes",
        lambda force_refresh=False, max_age_hours=12.0: list(universe),  # noqa: ARG005
        raising=True,
    )
    monkeypatch.setattr(
        "data.universe.get_new_listings",
        lambda days=120, force_refresh=False, max_age_seconds=5.0: list(new_listing),  # noqa: ARG005
        raising=True,
    )

    rot = StockRotator()
    batch = rot.discover_new(
        max_stocks=40,
        min_market_cap=0.0,
        stop_check=lambda: False,
        progress_cb=lambda _msg, _count: None,
    )

    snapshot = rot.get_pool_snapshot()
    assert len(batch) == 40
    assert len(snapshot) >= 5201
    assert "001399" in snapshot

    state = rot.to_dict()
    assert len(list(state.get("pool", []) or [])) == len(snapshot)

    restored = StockRotator()
    restored.from_dict(state)
    assert len(restored.get_pool_snapshot()) == len(snapshot)


def test_rotator_injects_new_listing_between_full_refresh_windows(monkeypatch):
    monkeypatch.setattr(
        "data.universe.get_universe_codes",
        lambda force_refresh=False, max_age_hours=12.0: ["600000", "600001"],  # noqa: ARG005
        raising=True,
    )
    monkeypatch.setattr(
        "data.universe.get_new_listings",
        lambda days=120, force_refresh=False, max_age_seconds=5.0: ["001777"],  # noqa: ARG005
        raising=True,
    )

    rot = StockRotator()
    rot._pool = ["600000", "600001"]
    rot._last_discovery = time.time()
    rot._discovery_ttl = 99999.0
    rot._new_listing_probe_ttl = 0.0

    out = rot.discover_new(
        max_stocks=5,
        min_market_cap=0.0,
        stop_check=lambda: False,
        progress_cb=lambda _msg, _count: None,
    )

    assert "001777" in rot.get_pool_snapshot()
    assert out[0] == "001777"
