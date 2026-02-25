from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from config.settings import CONFIG
from data.fundamentals import FundamentalDataService, FundamentalSnapshot, get_fundamental_service
from models.predictor_types import Prediction
from utils.json_io import read_json_safe, write_json_safe
from utils.metrics import inc_counter, observe, set_gauge

_PROFILE_FILE_NAME = "screener_profiles.json"


@dataclass(frozen=True)
class ScreenerWeights:
    confidence: float = 0.72
    signal_strength: float = 0.08
    fundamentals: float = 0.20

    def normalized(self) -> ScreenerWeights:
        total = float(self.confidence + self.signal_strength + self.fundamentals)
        if total <= 0:
            return ScreenerWeights()
        return ScreenerWeights(
            confidence=float(self.confidence / total),
            signal_strength=float(self.signal_strength / total),
            fundamentals=float(self.fundamentals / total),
        )


@dataclass(frozen=True)
class ScreenerProfile:
    name: str = "balanced"
    min_confidence: float = 0.70
    min_signal_strength: float = 0.52
    min_fundamental_score: float = 0.35
    min_avg_notional_cny: float = 2.5e7
    max_annualized_volatility: float = 1.05
    require_positive_trend_60d: bool = False
    allow_missing_fundamentals: bool = True
    weights: ScreenerWeights = field(default_factory=ScreenerWeights)

    def normalized(self) -> ScreenerProfile:
        return ScreenerProfile(
            name=str(self.name or "balanced").strip().lower() or "balanced",
            min_confidence=float(np.clip(self.min_confidence, 0.30, 0.99)),
            min_signal_strength=float(np.clip(self.min_signal_strength, 0.10, 0.99)),
            min_fundamental_score=float(np.clip(self.min_fundamental_score, 0.0, 0.99)),
            min_avg_notional_cny=max(0.0, float(self.min_avg_notional_cny)),
            max_annualized_volatility=max(0.10, float(self.max_annualized_volatility)),
            require_positive_trend_60d=bool(self.require_positive_trend_60d),
            allow_missing_fundamentals=bool(self.allow_missing_fundamentals),
            weights=(self.weights or ScreenerWeights()).normalized(),
        )

    def to_dict(self) -> dict[str, Any]:
        weights = self.weights.normalized()
        return {
            "name": self.name,
            "min_confidence": float(self.min_confidence),
            "min_signal_strength": float(self.min_signal_strength),
            "min_fundamental_score": float(self.min_fundamental_score),
            "min_avg_notional_cny": float(self.min_avg_notional_cny),
            "max_annualized_volatility": float(self.max_annualized_volatility),
            "require_positive_trend_60d": bool(self.require_positive_trend_60d),
            "allow_missing_fundamentals": bool(self.allow_missing_fundamentals),
            "weights": {
                "confidence": float(weights.confidence),
                "signal_strength": float(weights.signal_strength),
                "fundamentals": float(weights.fundamentals),
            },
        }

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        *,
        default_name: str = "balanced",
        base: ScreenerProfile | None = None,
    ) -> ScreenerProfile:
        src = base or cls(name=default_name)
        row = dict(data or {})
        weights_raw = row.get("weights")
        if isinstance(weights_raw, dict):
            weights = ScreenerWeights(
                confidence=float(weights_raw.get("confidence", src.weights.confidence)),
                signal_strength=float(
                    weights_raw.get("signal_strength", src.weights.signal_strength)
                ),
                fundamentals=float(weights_raw.get("fundamentals", src.weights.fundamentals)),
            ).normalized()
        else:
            weights = src.weights
        profile = cls(
            name=str(row.get("name", src.name) or src.name),
            min_confidence=float(row.get("min_confidence", src.min_confidence)),
            min_signal_strength=float(row.get("min_signal_strength", src.min_signal_strength)),
            min_fundamental_score=float(row.get("min_fundamental_score", src.min_fundamental_score)),
            min_avg_notional_cny=float(row.get("min_avg_notional_cny", src.min_avg_notional_cny)),
            max_annualized_volatility=float(
                row.get("max_annualized_volatility", src.max_annualized_volatility)
            ),
            require_positive_trend_60d=bool(
                row.get("require_positive_trend_60d", src.require_positive_trend_60d)
            ),
            allow_missing_fundamentals=bool(
                row.get("allow_missing_fundamentals", src.allow_missing_fundamentals)
            ),
            weights=weights,
        )
        return profile.normalized()


_DEFAULT_PROFILE_PRESETS: dict[str, ScreenerProfile] = {
    "balanced": ScreenerProfile(
        name="balanced",
        min_confidence=0.70,
        min_signal_strength=0.52,
        min_fundamental_score=0.35,
        min_avg_notional_cny=2.5e7,
        max_annualized_volatility=1.05,
        require_positive_trend_60d=False,
        allow_missing_fundamentals=True,
        weights=ScreenerWeights(confidence=0.72, signal_strength=0.08, fundamentals=0.20),
    ).normalized(),
    "momentum": ScreenerProfile(
        name="momentum",
        min_confidence=0.72,
        min_signal_strength=0.60,
        min_fundamental_score=0.20,
        min_avg_notional_cny=3.5e7,
        max_annualized_volatility=1.30,
        require_positive_trend_60d=True,
        allow_missing_fundamentals=True,
        weights=ScreenerWeights(confidence=0.62, signal_strength=0.28, fundamentals=0.10),
    ).normalized(),
    "quality": ScreenerProfile(
        name="quality",
        min_confidence=0.72,
        min_signal_strength=0.52,
        min_fundamental_score=0.60,
        min_avg_notional_cny=4.0e7,
        max_annualized_volatility=0.85,
        require_positive_trend_60d=False,
        allow_missing_fundamentals=False,
        weights=ScreenerWeights(confidence=0.55, signal_strength=0.10, fundamentals=0.35),
    ).normalized(),
    "value": ScreenerProfile(
        name="value",
        min_confidence=0.66,
        min_signal_strength=0.45,
        min_fundamental_score=0.65,
        min_avg_notional_cny=2.0e7,
        max_annualized_volatility=0.95,
        require_positive_trend_60d=False,
        allow_missing_fundamentals=False,
        weights=ScreenerWeights(confidence=0.50, signal_strength=0.05, fundamentals=0.45),
    ).normalized(),
    "defensive": ScreenerProfile(
        name="defensive",
        min_confidence=0.76,
        min_signal_strength=0.60,
        min_fundamental_score=0.55,
        min_avg_notional_cny=6.0e7,
        max_annualized_volatility=0.70,
        require_positive_trend_60d=True,
        allow_missing_fundamentals=False,
        weights=ScreenerWeights(confidence=0.58, signal_strength=0.17, fundamentals=0.25),
    ).normalized(),
}


class ScreenerProfileStore:
    def __init__(self, path: Path | None = None) -> None:
        self._path = Path(path) if path else (Path(CONFIG.data_dir) / _PROFILE_FILE_NAME)
        self._lock = threading.RLock()

    @property
    def path(self) -> Path:
        return self._path

    def load_payload(self) -> dict[str, Any]:
        with self._lock:
            data = read_json_safe(self._path, default={})
        if not isinstance(data, dict):
            return {}
        return dict(data)

    def resolve_profile_name(self, preferred: str | None = None) -> str:
        env_name = str(os.environ.get("TRADING_SCREENER_PROFILE", "")).strip().lower()
        if env_name:
            return env_name
        if preferred:
            return str(preferred).strip().lower()
        payload = self.load_payload()
        active = str(payload.get("active_profile", "balanced")).strip().lower()
        return active or "balanced"

    def available_profiles(self) -> dict[str, ScreenerProfile]:
        payload = self.load_payload()
        custom = payload.get("profiles", {})
        profiles: dict[str, ScreenerProfile] = dict(_DEFAULT_PROFILE_PRESETS)
        if isinstance(custom, dict):
            for raw_name, raw_cfg in custom.items():
                if not isinstance(raw_cfg, dict):
                    continue
                name = str(raw_name or "").strip().lower()
                if not name:
                    continue
                base = profiles.get(name, ScreenerProfile(name=name))
                profiles[name] = ScreenerProfile.from_dict(raw_cfg, default_name=name, base=base)
        return profiles

    def resolve_profile(self, preferred: str | None = None) -> ScreenerProfile:
        name = self.resolve_profile_name(preferred)
        profiles = self.available_profiles()
        profile = profiles.get(name)
        if profile is not None:
            return profile.normalized()
        return profiles["balanced"].normalized()

    def save_profile(
        self,
        profile: ScreenerProfile,
        *,
        set_active: bool = False,
    ) -> bool:
        row = profile.normalized()
        with self._lock:
            payload = self.load_payload()
            custom = payload.get("profiles")
            if not isinstance(custom, dict):
                custom = {}
            custom[str(row.name)] = row.to_dict()
            payload["profiles"] = custom
            if set_active:
                payload["active_profile"] = str(row.name)
            return write_json_safe(self._path, payload)

    def delete_profile(self, name: str) -> bool:
        key = str(name or "").strip().lower()
        if not key:
            return False
        with self._lock:
            payload = self.load_payload()
            custom = payload.get("profiles")
            if not isinstance(custom, dict):
                return False
            if key not in custom:
                return False
            del custom[key]
            payload["profiles"] = custom
            active = str(payload.get("active_profile", "balanced")).strip().lower()
            if active == key:
                payload["active_profile"] = "balanced"
            return write_json_safe(self._path, payload)

    def save_active_profile(self, name: str) -> bool:
        profile_name = str(name or "").strip().lower() or "balanced"
        with self._lock:
            payload = self.load_payload()
            payload["active_profile"] = profile_name
            return write_json_safe(self._path, payload)


def _normalize_symbol(code: object) -> str:
    digits = "".join(ch for ch in str(code or "") if ch.isdigit())
    if not digits:
        return ""
    return digits[-6:].zfill(6)


class ScreenerEngine:
    """Ranks model predictions with fundamentals-aware hard gates."""

    def __init__(
        self,
        *,
        profile: ScreenerProfile | None = None,
        fundamentals: FundamentalDataService | None = None,
    ) -> None:
        self._profile = (profile or ScreenerProfile()).normalized()
        self._weights = self._profile.weights.normalized()
        self._fundamentals = fundamentals or get_fundamental_service()

    @property
    def profile(self) -> ScreenerProfile:
        return self._profile

    def rank_predictions(
        self,
        predictions: list[Prediction],
        *,
        top_n: int,
        include_fundamentals: bool = True,
    ) -> list[Prediction]:
        t0 = time.perf_counter()
        rows = list(predictions or [])
        if not rows:
            return []

        fundamental_map: dict[str, FundamentalSnapshot] = {}
        if include_fundamentals:
            codes = [str(getattr(p, "stock_code", "")) for p in rows]
            fundamental_map = self._fundamentals.get_snapshots(codes)
            set_gauge("screener_last_fundamental_count", float(len(fundamental_map)))

        ranked: list[Prediction] = []
        rejects = 0
        for pred in rows:
            code = _normalize_symbol(getattr(pred, "stock_code", ""))
            snap = fundamental_map.get(code) if include_fundamentals else None
            accepted, reason = self._passes_gate(pred, snap, include_fundamentals=include_fundamentals)
            if not accepted:
                rejects += 1
                inc_counter(
                    "screener_gate_reject_total",
                    labels={"profile": self._profile.name, "reason": reason},
                )
                continue

            conf = float(np.clip(getattr(pred, "confidence", 0.0), 0.0, 1.0))
            strength = float(np.clip(getattr(pred, "signal_strength", conf), 0.0, 1.0))
            fscore = float(np.clip(getattr(snap, "composite_score", 0.5), 0.0, 1.0))
            liq_score = self._liquidity_score(snap)
            vol_score = self._volatility_score(snap)

            rank_core = (
                (self._weights.confidence * conf)
                + (self._weights.signal_strength * strength)
                + (self._weights.fundamentals * fscore)
            )
            rank_score = float(np.clip((0.88 * rank_core) + (0.07 * liq_score) + (0.05 * vol_score), 0.0, 1.0))

            pred.rank_score = rank_score
            pred.fundamental_score = fscore
            pred.liquidity_score = liq_score
            pred.volatility_score = vol_score
            pred.screener_profile = self._profile.name
            self._append_reason(pred, f"Rank profile: {self._profile.name} ({rank_score:.2f})")
            if snap is not None:
                self._append_reason(pred, f"Fundamental composite: {fscore:.2f} ({snap.source})")
            ranked.append(pred)

        ranked.sort(
            key=lambda p: (
                float(getattr(p, "rank_score", 0.0)),
                float(getattr(p, "confidence", 0.0)),
            ),
            reverse=True,
        )
        selected = ranked[: max(1, int(top_n))]

        elapsed = max(0.0, time.perf_counter() - t0)
        observe("screener_rank_seconds", elapsed)
        set_gauge("screener_last_candidates", float(len(rows)))
        set_gauge("screener_last_rejects", float(rejects))
        set_gauge("screener_last_selected", float(len(selected)))
        pass_rate = (len(ranked) / len(rows)) if rows else 0.0
        set_gauge("screener_gate_pass_rate", float(pass_rate))
        inc_counter("screener_runs_total", labels={"profile": self._profile.name})
        if not selected:
            inc_counter("screener_empty_after_gates_total", labels={"profile": self._profile.name})
        return selected

    def _passes_gate(
        self,
        pred: Prediction,
        snap: FundamentalSnapshot | None,
        *,
        include_fundamentals: bool,
    ) -> tuple[bool, str]:
        conf = float(np.clip(getattr(pred, "confidence", 0.0), 0.0, 1.0))
        if conf < self._profile.min_confidence:
            return False, "confidence"

        strength = float(np.clip(getattr(pred, "signal_strength", conf), 0.0, 1.0))
        if strength < self._profile.min_signal_strength:
            return False, "signal_strength"

        if not include_fundamentals:
            return True, "ok"

        if snap is None:
            return (True, "ok") if self._profile.allow_missing_fundamentals else (False, "missing_fundamentals")

        fscore = float(np.clip(getattr(snap, "composite_score", 0.5), 0.0, 1.0))
        if fscore < self._profile.min_fundamental_score:
            return False, "fundamental_score"

        min_notional = float(self._profile.min_avg_notional_cny)
        avg_notional = getattr(snap, "avg_notional_20d_cny", None)
        if min_notional > 0 and avg_notional is not None:
            if float(avg_notional) < min_notional:
                return False, "liquidity"
        elif min_notional > 0 and (not self._profile.allow_missing_fundamentals):
            return False, "liquidity_missing"

        max_vol = float(self._profile.max_annualized_volatility)
        ann_vol = getattr(snap, "annualized_volatility", None)
        if ann_vol is not None and float(ann_vol) > max_vol:
            return False, "volatility"
        if ann_vol is None and (not self._profile.allow_missing_fundamentals):
            return False, "volatility_missing"

        if self._profile.require_positive_trend_60d:
            trend = getattr(snap, "trend_60d", None)
            if trend is None:
                if not self._profile.allow_missing_fundamentals:
                    return False, "trend_missing"
            elif float(trend) <= 0:
                return False, "trend"

        return True, "ok"

    def _liquidity_score(self, snap: FundamentalSnapshot | None) -> float:
        if snap is None:
            return 0.5
        avg_notional = getattr(snap, "avg_notional_20d_cny", None)
        min_notional = max(1.0, float(self._profile.min_avg_notional_cny))
        if avg_notional is None or avg_notional <= 0:
            return 0.5
        ratio = float(avg_notional) / min_notional
        # 1x threshold -> 0.5, 4x threshold -> ~1.0
        return float(np.clip(np.log1p(max(0.0, ratio)) / np.log1p(4.0), 0.0, 1.0))

    def _volatility_score(self, snap: FundamentalSnapshot | None) -> float:
        if snap is None:
            return 0.5
        vol = getattr(snap, "annualized_volatility", None)
        if vol is None or vol <= 0:
            return 0.5
        max_vol = max(0.10, float(self._profile.max_annualized_volatility))
        # lower is better; at limit => ~0.5, half limit => ~1.0
        return float(np.clip(1.0 - (float(vol) / (2.0 * max_vol)), 0.0, 1.0))

    @staticmethod
    def _append_reason(pred: Prediction, text: str) -> None:
        existing = list(getattr(pred, "reasons", []) or [])
        if any(str(text) == str(x) for x in existing):
            return
        existing.append(str(text))
        pred.reasons = existing


_SCREENER_LOCK = threading.Lock()
_SCREENER_SINGLETONS: dict[str, ScreenerEngine] = {}


def _profile_store() -> ScreenerProfileStore:
    return ScreenerProfileStore()


def list_screener_profiles() -> list[str]:
    profiles = _profile_store().available_profiles()
    return sorted(profiles.keys())


def get_active_screener_profile_name(preferred: str | None = None) -> str:
    return _profile_store().resolve_profile_name(preferred)


def build_default_screener(
    profile_name: str | None = None,
    *,
    force_reload: bool = False,
) -> ScreenerEngine:
    resolved = _profile_store().resolve_profile(profile_name)
    key = resolved.name
    if not force_reload and key in _SCREENER_SINGLETONS:
        return _SCREENER_SINGLETONS[key]
    with _SCREENER_LOCK:
        if (not force_reload) and key in _SCREENER_SINGLETONS:
            return _SCREENER_SINGLETONS[key]
        engine = ScreenerEngine(profile=resolved, fundamentals=get_fundamental_service())
        _SCREENER_SINGLETONS[key] = engine
        return engine


def set_active_screener_profile(name: str) -> bool:
    ok = _profile_store().save_active_profile(name)
    if ok:
        reset_default_screener()
    return ok


def reset_default_screener() -> None:
    with _SCREENER_LOCK:
        _SCREENER_SINGLETONS.clear()


__all__ = [
    "ScreenerWeights",
    "ScreenerProfile",
    "ScreenerProfileStore",
    "ScreenerEngine",
    "list_screener_profiles",
    "get_active_screener_profile_name",
    "build_default_screener",
    "set_active_screener_profile",
    "reset_default_screener",
]
