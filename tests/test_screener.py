from __future__ import annotations

import json

from analysis.screener import ScreenerEngine, ScreenerProfile, ScreenerProfileStore, ScreenerWeights
from core.types import Signal
from data.fundamentals import FundamentalSnapshot
from models.predictor_types import Prediction


class _FakeFundamentalService:
    def __init__(self, snapshots: dict[str, FundamentalSnapshot]) -> None:
        self._snapshots = dict(snapshots)

    def get_snapshots(self, symbols: list[str], *, force_refresh: bool = False) -> dict[str, FundamentalSnapshot]:
        out: dict[str, FundamentalSnapshot] = {}
        for code in symbols:
            digits = "".join(ch for ch in str(code) if ch.isdigit()).zfill(6)
            snap = self._snapshots.get(digits)
            if snap is None:
                snap = FundamentalSnapshot(symbol=digits, source="test", composite_score=0.5)
            out[digits] = snap
        return out


def _mk_snapshot(
    code: str,
    *,
    fscore: float,
    avg_notional: float = 1e8,
    ann_vol: float = 0.6,
    trend_60d: float = 0.10,
) -> FundamentalSnapshot:
    return FundamentalSnapshot(
        symbol=code,
        source="test",
        composite_score=fscore,
        value_score=fscore,
        quality_score=fscore,
        growth_score=fscore,
        avg_notional_20d_cny=avg_notional,
        annualized_volatility=ann_vol,
        trend_60d=trend_60d,
    )


def test_screener_ranks_using_fundamentals_overlay() -> None:
    p1 = Prediction(stock_code="600519", signal=Signal.BUY, confidence=0.80, signal_strength=0.80)
    p2 = Prediction(stock_code="000001", signal=Signal.BUY, confidence=0.80, signal_strength=0.80)

    profile = ScreenerProfile(
        name="balanced",
        min_confidence=0.50,
        min_signal_strength=0.50,
        min_fundamental_score=0.0,
        min_avg_notional_cny=0.0,
        max_annualized_volatility=2.0,
        weights=ScreenerWeights(confidence=0.65, signal_strength=0.05, fundamentals=0.30),
    )
    engine = ScreenerEngine(
        profile=profile,
        fundamentals=_FakeFundamentalService(
            {
                "600519": _mk_snapshot("600519", fscore=0.20),
                "000001": _mk_snapshot("000001", fscore=0.90),
            }
        ),
    )
    ranked = engine.rank_predictions([p1, p2], top_n=2, include_fundamentals=True)

    assert len(ranked) == 2
    assert ranked[0].stock_code == "000001"
    assert float(getattr(ranked[0], "fundamental_score", 0.0)) > float(
        getattr(ranked[1], "fundamental_score", 0.0)
    )
    assert any("Fundamental composite:" in str(x) for x in ranked[0].reasons)


def test_screener_can_run_without_fundamental_overlay() -> None:
    p1 = Prediction(stock_code="600519", signal=Signal.BUY, confidence=0.86, signal_strength=0.86)
    p2 = Prediction(stock_code="000001", signal=Signal.BUY, confidence=0.80, signal_strength=0.80)

    profile = ScreenerProfile(
        name="balanced",
        min_confidence=0.50,
        min_signal_strength=0.50,
        min_fundamental_score=0.0,
        min_avg_notional_cny=0.0,
        max_annualized_volatility=2.0,
        weights=ScreenerWeights(confidence=0.90, signal_strength=0.10, fundamentals=0.00),
    )
    engine = ScreenerEngine(
        profile=profile,
        fundamentals=_FakeFundamentalService(
            {
                "600519": _mk_snapshot("600519", fscore=0.10),
                "000001": _mk_snapshot("000001", fscore=0.95),
            }
        ),
    )
    ranked = engine.rank_predictions([p2, p1], top_n=2, include_fundamentals=False)

    assert [p.stock_code for p in ranked] == ["600519", "000001"]


def test_screener_gate_rejects_low_liquidity() -> None:
    p1 = Prediction(stock_code="600519", signal=Signal.BUY, confidence=0.82, signal_strength=0.70)
    p2 = Prediction(stock_code="000001", signal=Signal.BUY, confidence=0.81, signal_strength=0.70)
    profile = ScreenerProfile(
        name="quality",
        min_confidence=0.70,
        min_signal_strength=0.60,
        min_fundamental_score=0.50,
        min_avg_notional_cny=5e7,
        max_annualized_volatility=0.9,
        require_positive_trend_60d=True,
        allow_missing_fundamentals=False,
    )
    engine = ScreenerEngine(
        profile=profile,
        fundamentals=_FakeFundamentalService(
            {
                "600519": _mk_snapshot(
                    "600519",
                    fscore=0.80,
                    avg_notional=1e6,
                    ann_vol=0.6,
                    trend_60d=0.2,
                ),
                "000001": _mk_snapshot(
                    "000001",
                    fscore=0.80,
                    avg_notional=8e7,
                    ann_vol=0.6,
                    trend_60d=0.2,
                ),
            }
        ),
    )

    ranked = engine.rank_predictions([p1, p2], top_n=2, include_fundamentals=True)
    assert [p.stock_code for p in ranked] == ["000001"]


def test_profile_store_resolves_custom_active_profile(tmp_path) -> None:
    path = tmp_path / "screener_profiles.json"
    payload = {
        "active_profile": "quality_plus",
        "profiles": {
            "quality_plus": {
                "name": "quality_plus",
                "min_confidence": 0.78,
                "min_signal_strength": 0.65,
                "min_fundamental_score": 0.70,
                "min_avg_notional_cny": 8e7,
                "max_annualized_volatility": 0.70,
                "require_positive_trend_60d": True,
                "allow_missing_fundamentals": False,
                "weights": {
                    "confidence": 0.55,
                    "signal_strength": 0.10,
                    "fundamentals": 0.35,
                },
            }
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    store = ScreenerProfileStore(path=path)
    profile = store.resolve_profile()

    assert profile.name == "quality_plus"
    assert float(profile.min_fundamental_score) >= 0.70
    assert bool(profile.require_positive_trend_60d) is True


def test_profile_store_save_profile_and_delete(tmp_path) -> None:
    path = tmp_path / "screener_profiles.json"
    store = ScreenerProfileStore(path=path)
    profile = ScreenerProfile(
        name="intraday_fast",
        min_confidence=0.74,
        min_signal_strength=0.58,
        min_fundamental_score=0.25,
        min_avg_notional_cny=3.0e7,
        max_annualized_volatility=1.20,
        require_positive_trend_60d=True,
        allow_missing_fundamentals=True,
        weights=ScreenerWeights(confidence=0.64, signal_strength=0.26, fundamentals=0.10),
    )

    assert store.save_profile(profile, set_active=True) is True
    resolved = store.resolve_profile()
    assert resolved.name == "intraday_fast"
    assert bool(resolved.require_positive_trend_60d) is True

    assert store.delete_profile("intraday_fast") is True
    fallback = store.resolve_profile()
    assert fallback.name == "balanced"
