from __future__ import annotations

import sys
import warnings
from types import SimpleNamespace

import pytest

from data import universe as universe_mod


def test_try_akshare_fetch_stops_after_reasonable_universe(monkeypatch) -> None:
    calls = {
        "spot_em": 0,
        "fallback_fast": 0,
        "fallback_slow": 0,
    }

    def _spot_em():
        calls["spot_em"] += 1
        raise RuntimeError("primary endpoint unavailable")

    def _fallback_fast():
        calls["fallback_fast"] += 1
        return [{"code": str(600000 + i)} for i in range(1600)]

    def _fallback_slow():
        calls["fallback_slow"] += 1
        raise AssertionError("slow fallback should not run when coverage already sufficient")

    fake_ak = SimpleNamespace(
        stock_zh_a_spot_em=_spot_em,
        stock_info_a_code_name=_fallback_fast,
        stock_zh_a_name_code=None,
        stock_info_sh_name_code=None,
        stock_info_sz_name_code=None,
        stock_info_bj_name_code=None,
        stock_zh_a_spot=_fallback_slow,
    )
    monkeypatch.setitem(sys.modules, "akshare", fake_ak)

    out = universe_mod._try_akshare_fetch(timeout=5)
    assert out is not None
    assert len(out) >= 1200
    assert calls["spot_em"] == 1
    assert calls["fallback_fast"] == 1
    assert calls["fallback_slow"] == 0


def test_get_universe_codes_refreshes_when_cached_source_is_fallback(monkeypatch) -> None:
    cached = {
        "codes": [str(600000 + i) for i in range(97)],
        "updated_ts": float(10**10),
        "source": "fallback",
    }
    monkeypatch.setattr(universe_mod, "load_universe", lambda: cached)
    monkeypatch.setattr(universe_mod, "_can_use_akshare", lambda: True)
    fresh_codes = [str(600000 + i) for i in range(1400)]
    monkeypatch.setattr(universe_mod, "_try_akshare_fetch", lambda timeout=20: fresh_codes)
    saved: dict[str, object] = {}
    monkeypatch.setattr(universe_mod, "save_universe", lambda data: saved.update(dict(data)))

    out = universe_mod.get_universe_codes(force_refresh=False, max_age_hours=24.0)

    assert len(out) == len(fresh_codes)
    assert len(saved.get("codes", [])) == len(fresh_codes)
    assert str(saved.get("source", "")) == "akshare_spot_em"


def test_persist_runtime_universe_codes_merges_existing(monkeypatch) -> None:
    existing = {
        "codes": ["600001", "600002"],
        "source": "akshare_spot_em",
    }
    saved: dict[str, object] = {}
    monkeypatch.setattr(universe_mod, "load_universe", lambda: dict(existing))
    monkeypatch.setattr(universe_mod, "save_universe", lambda data: saved.update(dict(data)))

    out = universe_mod.persist_runtime_universe_codes(
        ["600002", "000001"],
        source="ui_universe_refresh",
    )

    assert sorted(out.get("codes", [])) == ["000001", "600001", "600002"]
    assert sorted(saved.get("codes", [])) == ["000001", "600001", "600002"]
    assert str(saved.get("source", "")) == "ui_universe_refresh"


@pytest.mark.skipif(not universe_mod._HAS_PANDAS, reason="pandas unavailable")
def test_get_new_listings_suppresses_datetime_infer_warning(monkeypatch) -> None:
    import pandas as pd

    df = pd.DataFrame(
        {
            "code": ["600001", "600002", "600003", "600004", "600005"],
            # Mixed text previously triggered infer-format warning path.
            "mixed_col": ["a", "b", "c", "d", "e"],
            # Parseable date-like column but non-standard name.
            "x": ["2026-01-03", "2026-01-04", "2026-01-05", "2026-01-06", "2026-01-07"],
        }
    )
    fake_ak = SimpleNamespace(
        stock_zh_a_new_em=lambda: df,
        stock_zh_a_new=None,
    )
    monkeypatch.setitem(sys.modules, "akshare", fake_ak)

    with universe_mod._universe_lock:
        universe_mod._new_listings_cache["ts"] = 0.0
        universe_mod._new_listings_cache["days"] = 0
        universe_mod._new_listings_cache["codes"] = []

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        codes = universe_mod.get_new_listings(
            days=365,
            force_refresh=True,
            max_age_seconds=0.0,
        )

    assert len(codes) >= 1
    assert not any(
        "Could not infer format, so each element will be parsed individually"
        in str(w.message)
        for w in captured
    )
