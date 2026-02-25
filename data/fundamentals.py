from __future__ import annotations

import math
import os
import threading
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd

from data.fetcher_registry import get_active_fetcher_registry
from utils.logger import get_logger
from utils.metrics import inc_counter, observe, set_gauge

log = get_logger(__name__)

_FUNDAMENTAL_SERVICE_KEY = "fundamental_service"


@dataclass
class FundamentalSnapshot:
    """Single-symbol fundamental snapshot used by the screener."""

    symbol: str
    as_of: datetime = field(default_factory=lambda: datetime.now(UTC))
    source: str = "proxy"

    pe_ttm: float | None = None
    pb: float | None = None
    roe_pct: float | None = None
    revenue_yoy_pct: float | None = None
    net_profit_yoy_pct: float | None = None
    debt_to_asset_pct: float | None = None
    market_cap_cny: float | None = None
    dividend_yield_pct: float | None = None
    avg_notional_20d_cny: float | None = None
    annualized_volatility: float | None = None
    trend_60d: float | None = None

    value_score: float = 0.50
    quality_score: float = 0.50
    growth_score: float = 0.50
    composite_score: float = 0.50

    stale: bool = False
    warnings: list[str] = field(default_factory=list)


def _clip01(value: float) -> float:
    return float(np.clip(float(value), 0.0, 1.0))


def _normalize_symbol(symbol: object) -> str:
    digits = "".join(ch for ch in str(symbol or "") if ch.isdigit())
    if not digits:
        return ""
    return digits[-6:].zfill(6)


def _safe_float(raw: object) -> float | None:
    """Parse numeric text from mixed Chinese/English provider formats."""
    if raw is None:
        return None

    if isinstance(raw, (int, float, np.number)):
        value = float(raw)
        if math.isfinite(value):
            return value
        return None

    text = str(raw).strip()
    if not text:
        return None
    lowered = text.lower()
    if lowered in {"nan", "none", "null", "--", "n/a", "na", "-"}:
        return None

    negative = False
    if text.startswith("(") and text.endswith(")"):
        negative = True
        text = text[1:-1]

    multiplier = 1.0
    text = text.replace(",", "").replace("\uff0c", "").replace(" ", "")
    if "\u4ebf" in text:
        multiplier = 1e8
        text = text.replace("\u4ebf", "")
    elif "\u4e07" in text:
        multiplier = 1e4
        text = text.replace("\u4e07", "")
    elif text.endswith("B") or text.endswith("b"):
        multiplier = 1e9
        text = text[:-1]
    elif text.endswith("M") or text.endswith("m"):
        multiplier = 1e6
        text = text[:-1]

    is_pct = text.endswith("%")
    if is_pct:
        text = text[:-1]

    try:
        value = float(text)
    except ValueError:
        return None

    if negative:
        value = -value
    value *= multiplier
    if is_pct:
        return value
    return value if math.isfinite(value) else None


def _score_value(pe_ttm: float | None, pb: float | None, dividend_yield_pct: float | None) -> float:
    parts: list[float] = []
    if pe_ttm is not None and pe_ttm > 0:
        # Best zone around 8-20x, heavily penalize >40x.
        parts.append(_clip01(1.0 - ((pe_ttm - 8.0) / 32.0)))
    if pb is not None and pb > 0:
        # Best zone around 0.8-2.5x, penalize >6x.
        parts.append(_clip01(1.0 - ((pb - 0.8) / 5.2)))
    if dividend_yield_pct is not None:
        parts.append(_clip01(dividend_yield_pct / 6.0))
    if not parts:
        return 0.50
    return float(np.mean(parts))


def _score_quality(roe_pct: float | None, debt_to_asset_pct: float | None, volatility: float | None) -> float:
    parts: list[float] = []
    if roe_pct is not None:
        parts.append(_clip01((roe_pct - 5.0) / 20.0))
    if debt_to_asset_pct is not None:
        parts.append(_clip01(1.0 - ((debt_to_asset_pct - 20.0) / 60.0)))
    if volatility is not None and volatility > 0:
        parts.append(_clip01(1.0 - ((volatility - 0.20) / 0.60)))
    if not parts:
        return 0.50
    return float(np.mean(parts))


def _score_growth(revenue_yoy_pct: float | None, net_profit_yoy_pct: float | None, trend_60d: float | None) -> float:
    parts: list[float] = []
    if revenue_yoy_pct is not None:
        parts.append(_clip01((revenue_yoy_pct + 10.0) / 40.0))
    if net_profit_yoy_pct is not None:
        parts.append(_clip01((net_profit_yoy_pct + 10.0) / 60.0))
    if trend_60d is not None:
        parts.append(_clip01((trend_60d + 0.10) / 0.40))
    if not parts:
        return 0.50
    return float(np.mean(parts))


class FundamentalDataService:
    """Fundamental integration with cached snapshots and proxy fallback."""

    def __init__(self, cache_ttl_seconds: float = 6 * 3600.0) -> None:
        self._cache_ttl_seconds = max(120.0, float(cache_ttl_seconds))
        self._lock = threading.RLock()
        self._cache: dict[str, tuple[float, FundamentalSnapshot]] = {}
        self._online_enabled = str(
            os.environ.get("TRADING_FUNDAMENTALS_ONLINE", "0")
        ).strip().lower() in {"1", "true", "yes", "on"}

    def reset_cache(self) -> None:
        with self._lock:
            self._cache.clear()

    def get_snapshot(self, symbol: str, *, force_refresh: bool = False) -> FundamentalSnapshot:
        code = _normalize_symbol(symbol)
        if not code:
            return FundamentalSnapshot(symbol="", source="invalid", warnings=["Invalid symbol"])

        now = time.monotonic()
        with self._lock:
            hit = self._cache.get(code)
            if (not force_refresh) and hit and (now - hit[0] <= self._cache_ttl_seconds):
                inc_counter("fundamentals_cache_hit_total", labels={"source": hit[1].source})
                return hit[1]

        t0 = time.perf_counter()
        snapshot = self._build_proxy_snapshot(code)

        if self._online_enabled:
            online = self._fetch_online_snapshot(code)
            if online is not None:
                snapshot = self._merge_snapshots(snapshot, online)

        self._recompute_scores(snapshot)

        with self._lock:
            self._cache[code] = (time.monotonic(), snapshot)

        elapsed = max(0.0, time.perf_counter() - t0)
        observe("fundamentals_fetch_seconds", elapsed)
        set_gauge("fundamentals_cache_size", float(len(self._cache)))
        inc_counter("fundamentals_requests_total", labels={"source": snapshot.source})
        return snapshot

    def get_snapshots(self, symbols: list[str], *, force_refresh: bool = False) -> dict[str, FundamentalSnapshot]:
        out: dict[str, FundamentalSnapshot] = {}
        for sym in list(symbols or []):
            snap = self.get_snapshot(sym, force_refresh=force_refresh)
            if snap.symbol:
                out[snap.symbol] = snap
        return out

    def _merge_snapshots(self, proxy: FundamentalSnapshot, online: FundamentalSnapshot) -> FundamentalSnapshot:
        merged = FundamentalSnapshot(
            symbol=proxy.symbol,
            as_of=online.as_of,
            source=online.source,
            pe_ttm=online.pe_ttm if online.pe_ttm is not None else proxy.pe_ttm,
            pb=online.pb if online.pb is not None else proxy.pb,
            roe_pct=online.roe_pct if online.roe_pct is not None else proxy.roe_pct,
            revenue_yoy_pct=(
                online.revenue_yoy_pct
                if online.revenue_yoy_pct is not None
                else proxy.revenue_yoy_pct
            ),
            net_profit_yoy_pct=(
                online.net_profit_yoy_pct
                if online.net_profit_yoy_pct is not None
                else proxy.net_profit_yoy_pct
            ),
            debt_to_asset_pct=(
                online.debt_to_asset_pct
                if online.debt_to_asset_pct is not None
                else proxy.debt_to_asset_pct
            ),
            market_cap_cny=(
                online.market_cap_cny
                if online.market_cap_cny is not None
                else proxy.market_cap_cny
            ),
            dividend_yield_pct=(
                online.dividend_yield_pct
                if online.dividend_yield_pct is not None
                else proxy.dividend_yield_pct
            ),
            avg_notional_20d_cny=(
                online.avg_notional_20d_cny
                if online.avg_notional_20d_cny is not None
                else proxy.avg_notional_20d_cny
            ),
            annualized_volatility=(
                online.annualized_volatility
                if online.annualized_volatility is not None
                else proxy.annualized_volatility
            ),
            trend_60d=(
                online.trend_60d
                if online.trend_60d is not None
                else proxy.trend_60d
            ),
            stale=False,
            warnings=list(dict.fromkeys(proxy.warnings + online.warnings)),
        )
        return merged

    def _build_proxy_snapshot(self, symbol: str) -> FundamentalSnapshot:
        snapshot = FundamentalSnapshot(
            symbol=symbol,
            source="proxy",
            warnings=["Using proxy fundamentals derived from local market history"],
        )

        volatility: float | None = None
        trend_60d: float | None = None
        avg_notional_20d_cny: float | None = None
        market_cap_proxy: float | None = None
        try:
            from data.fetcher import get_fetcher

            fetcher = get_fetcher()
            try:
                history = fetcher.get_history(
                    symbol,
                    interval="1d",
                    bars=150,
                    use_cache=True,
                    update_db=False,
                    allow_online=False,
                )
            except TypeError:
                history = fetcher.get_history(
                    symbol,
                    interval="1d",
                    bars=150,
                    use_cache=True,
                    update_db=False,
                )
            if isinstance(history, pd.DataFrame) and not history.empty:
                close = pd.to_numeric(history.get("close"), errors="coerce").dropna()
                volume = pd.to_numeric(history.get("volume"), errors="coerce").dropna()
                if len(close) >= 20:
                    ret = close.pct_change().dropna()
                    if not ret.empty:
                        volatility = float(ret.tail(60).std() * np.sqrt(252.0))
                    if len(close) >= 60:
                        base = float(close.iloc[-60])
                        if base > 0:
                            trend_60d = float((float(close.iloc[-1]) / base) - 1.0)
                if len(close) >= 5 and len(volume) >= 5:
                    q = min(len(close), len(volume))
                    notional = (
                        close.tail(q).to_numpy(dtype=float)
                        * volume.tail(q).to_numpy(dtype=float)
                    )
                    if np.isfinite(notional).any():
                        market_cap_proxy = float(np.nanmedian(notional) * 25.0)
                        tail = notional[-20:] if len(notional) > 20 else notional
                        avg_notional_20d_cny = float(np.nanmean(tail))
        except Exception as e:  # pragma: no cover - best effort fallback
            snapshot.warnings.append(f"Proxy history unavailable: {e}")

        snapshot.market_cap_cny = market_cap_proxy
        snapshot.avg_notional_20d_cny = avg_notional_20d_cny
        snapshot.annualized_volatility = volatility
        snapshot.trend_60d = trend_60d
        snapshot.value_score = _score_value(snapshot.pe_ttm, snapshot.pb, snapshot.dividend_yield_pct)
        snapshot.quality_score = _score_quality(snapshot.roe_pct, snapshot.debt_to_asset_pct, volatility)
        snapshot.growth_score = _score_growth(snapshot.revenue_yoy_pct, snapshot.net_profit_yoy_pct, trend_60d)
        snapshot.composite_score = _clip01(
            (0.35 * snapshot.value_score)
            + (0.35 * snapshot.quality_score)
            + (0.30 * snapshot.growth_score)
        )
        return snapshot

    def _fetch_online_snapshot(self, symbol: str) -> FundamentalSnapshot | None:
        try:
            import akshare as ak
        except ImportError:
            return None

        try:
            t0 = time.perf_counter()
            raw_df = ak.stock_individual_info_em(symbol=symbol)
            observe("fundamentals_online_fetch_seconds", max(0.0, time.perf_counter() - t0))
            if not isinstance(raw_df, pd.DataFrame) or raw_df.empty:
                return None
        except Exception as e:
            log.debug("Online fundamentals fetch failed for %s: %s", symbol, e)
            inc_counter("fundamentals_online_fetch_error_total")
            return None

        key_col = "item" if "item" in raw_df.columns else raw_df.columns[0]
        val_col = "value" if "value" in raw_df.columns else raw_df.columns[-1]

        values: dict[str, Any] = {}
        for _, row in raw_df.iterrows():
            k = str(row.get(key_col, "")).strip()
            if not k:
                continue
            values[k] = row.get(val_col)

        key_pe_dynamic = "\u5e02\u76c8\u7387-\u52a8\u6001"
        key_pe = "\u5e02\u76c8\u7387"
        key_pb = "\u5e02\u51c0\u7387"
        key_market_cap = "\u603b\u5e02\u503c"
        key_roe = "\u51c0\u8d44\u4ea7\u6536\u76ca\u7387"
        key_revenue_yoy = "\u8425\u4e1a\u603b\u6536\u5165\u540c\u6bd4\u589e\u957f"
        key_profit_yoy = "\u51c0\u5229\u6da6\u540c\u6bd4\u589e\u957f"
        key_debt_to_asset = "\u8d44\u4ea7\u8d1f\u503a\u7387"
        key_dividend_yield = "\u80a1\u606f\u7387"

        snapshot = FundamentalSnapshot(
            symbol=symbol,
            source="akshare",
            pe_ttm=_safe_float(values.get(key_pe_dynamic)) or _safe_float(values.get(key_pe)),
            pb=_safe_float(values.get(key_pb)),
            market_cap_cny=_safe_float(values.get(key_market_cap)),
            warnings=[],
        )

        # Optional extra fields if provider exposes them.
        snapshot.roe_pct = _safe_float(values.get(key_roe))
        snapshot.revenue_yoy_pct = _safe_float(values.get(key_revenue_yoy))
        snapshot.net_profit_yoy_pct = _safe_float(values.get(key_profit_yoy))
        snapshot.debt_to_asset_pct = _safe_float(values.get(key_debt_to_asset))
        snapshot.dividend_yield_pct = _safe_float(values.get(key_dividend_yield))
        return snapshot

    @staticmethod
    def _completeness_ratio(snapshot: FundamentalSnapshot) -> float:
        fields = (
            snapshot.pe_ttm,
            snapshot.pb,
            snapshot.roe_pct,
            snapshot.revenue_yoy_pct,
            snapshot.net_profit_yoy_pct,
            snapshot.debt_to_asset_pct,
            snapshot.market_cap_cny,
            snapshot.dividend_yield_pct,
            snapshot.avg_notional_20d_cny,
            snapshot.annualized_volatility,
            snapshot.trend_60d,
        )
        present = sum(1 for v in fields if v is not None)
        total = max(1, len(fields))
        return float(present) / float(total)

    @staticmethod
    def _freshness_factor(snapshot: FundamentalSnapshot) -> float:
        as_of = snapshot.as_of
        try:
            if as_of.tzinfo is None:
                as_of = as_of.replace(tzinfo=UTC)
            now = datetime.now(UTC)
            age_hours = max(0.0, (now - as_of).total_seconds() / 3600.0)
        except Exception:
            age_hours = 0.0

        if age_hours <= 24.0:
            snapshot.stale = False
            return 1.0
        if age_hours <= 72.0:
            snapshot.stale = False
            return 0.96
        if age_hours <= (7.0 * 24.0):
            snapshot.stale = False
            return 0.88
        snapshot.stale = True
        return 0.74

    def _recompute_scores(self, snapshot: FundamentalSnapshot) -> None:
        snapshot.value_score = _score_value(
            snapshot.pe_ttm,
            snapshot.pb,
            snapshot.dividend_yield_pct,
        )
        snapshot.quality_score = _score_quality(
            snapshot.roe_pct,
            snapshot.debt_to_asset_pct,
            snapshot.annualized_volatility,
        )
        snapshot.growth_score = _score_growth(
            snapshot.revenue_yoy_pct,
            snapshot.net_profit_yoy_pct,
            snapshot.trend_60d,
        )
        raw_composite = _clip01(
            (0.35 * snapshot.value_score)
            + (0.35 * snapshot.quality_score)
            + (0.30 * snapshot.growth_score)
        )
        completeness_ratio = self._completeness_ratio(snapshot)
        completeness_factor = 0.70 + (0.30 * completeness_ratio)
        freshness_factor = self._freshness_factor(snapshot)

        source_factor = 1.0
        if str(snapshot.source).strip().lower() == "proxy":
            source_factor = 0.95

        quality_factor = _clip01(completeness_factor * freshness_factor * source_factor)
        snapshot.composite_score = _clip01(raw_composite * quality_factor)

        if completeness_ratio < 0.35:
            snapshot.warnings.append(
                "Fundamental coverage is sparse; composite score quality-adjusted"
            )
        if freshness_factor < 0.90:
            snapshot.warnings.append(
                "Fundamental snapshot is aging; composite score quality-adjusted"
            )
        snapshot.warnings = list(dict.fromkeys(snapshot.warnings))


def get_fundamental_service() -> FundamentalDataService:
    registry = get_active_fetcher_registry()
    service = registry.get_or_create(
        _FUNDAMENTAL_SERVICE_KEY,
        lambda: FundamentalDataService(),
    )
    if isinstance(service, FundamentalDataService):
        return service
    # Should not happen, but fail safe if registry key was reused with wrong type.
    fallback = FundamentalDataService()
    return fallback


def reset_fundamental_service() -> None:
    registry = get_active_fetcher_registry()
    registry.reset(instance=_FUNDAMENTAL_SERVICE_KEY)


__all__ = [
    "FundamentalSnapshot",
    "FundamentalDataService",
    "get_fundamental_service",
    "reset_fundamental_service",
]
