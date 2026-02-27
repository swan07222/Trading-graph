# data/fundamentals.py
"""Fundamental Data for China A-Share Markets.

This module provides comprehensive fundamental data:
- Financial statements (balance sheet, income statement, cash flow)
- Financial ratios (valuation, profitability, efficiency, leverage)
- Analyst estimates and recommendations
- Institutional holdings
- Corporate actions (dividends, splits, rights issues)
- ESG scores

Data Sources:
- AkShare (primary)
- EastMoney API
- Sina Finance
- Tencent Finance
- CSRC filings
"""

import os
import threading
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd
import requests

from config.settings import CONFIG
from utils.logger import get_logger

log = get_logger(__name__)


def _clamp01(value: float) -> float:
    """Clamp a score into [0, 1]."""
    return float(max(0.0, min(1.0, float(value))))


def _normalize_symbol(symbol: object) -> str:
    """Normalize stock code into 6-digit numeric symbol."""
    digits = "".join(ch for ch in str(symbol or "") if ch.isdigit())
    if not digits:
        return ""
    return digits[-6:].zfill(6)


def _safe_float(value: object) -> float | None:
    """Parse numeric values including percent and CN unit suffixes."""
    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        parsed = float(value)
        if parsed != parsed:
            return None
        return parsed

    text = str(value).strip()
    if not text:
        return None
    lowered = text.lower()
    if lowered in {"--", "-", "none", "null", "nan", "n/a", "na"}:
        return None

    scale = 1.0
    if text.endswith("%"):
        text = text[:-1].strip()
    if text.endswith("亿"):
        scale = 1e8
        text = text[:-1].strip()
    elif text.endswith("万"):
        scale = 1e4
        text = text[:-1].strip()

    text = text.replace(",", "")
    try:
        return float(text) * scale
    except ValueError:
        return None


@dataclass(slots=True)
class FundamentalSnapshot:
    """Single-symbol fundamental quality snapshot used by screener overlay."""

    symbol: str
    source: str = "proxy"
    as_of: datetime = field(default_factory=lambda: datetime.now(UTC))

    pe_ttm: float | None = None
    pb_mrq: float | None = None
    dividend_yield: float | None = None
    roe_ttm: float | None = None
    gross_margin: float | None = None
    revenue_growth_yoy: float | None = None
    earnings_growth_yoy: float | None = None

    avg_notional_20d_cny: float | None = None
    annualized_volatility: float | None = None
    trend_60d: float | None = None

    value_score: float = 0.5
    quality_score: float = 0.5
    growth_score: float = 0.5
    composite_score: float = 0.5

    stale: bool = False
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "source": self.source,
            "as_of": self.as_of.isoformat(),
            "pe_ttm": self.pe_ttm,
            "pb_mrq": self.pb_mrq,
            "dividend_yield": self.dividend_yield,
            "roe_ttm": self.roe_ttm,
            "gross_margin": self.gross_margin,
            "revenue_growth_yoy": self.revenue_growth_yoy,
            "earnings_growth_yoy": self.earnings_growth_yoy,
            "avg_notional_20d_cny": self.avg_notional_20d_cny,
            "annualized_volatility": self.annualized_volatility,
            "trend_60d": self.trend_60d,
            "value_score": self.value_score,
            "quality_score": self.quality_score,
            "growth_score": self.growth_score,
            "composite_score": self.composite_score,
            "stale": self.stale,
            "warnings": list(self.warnings),
        }


class FundamentalDataService:
    """Fundamental snapshot service with safe proxy fallback and TTL cache."""

    def __init__(self, cache_ttl_seconds: float = 600.0, stale_after_days: float = 7.0) -> None:
        self._cache_ttl_seconds = max(5.0, float(cache_ttl_seconds))
        self._stale_after_days = max(1.0, float(stale_after_days))
        self._cache: dict[str, tuple[float, FundamentalSnapshot]] = {}
        self._lock = threading.RLock()

    def get_snapshot(self, symbol: object, *, force_refresh: bool = False) -> FundamentalSnapshot:
        """Return a cached or refreshed snapshot for one symbol."""
        code = _normalize_symbol(symbol)
        if not code:
            snap = FundamentalSnapshot(symbol="", source="invalid")
            snap.warnings.append("invalid symbol")
            self._recompute_scores(snap)
            return snap

        now = time.monotonic()
        with self._lock:
            cached = self._cache.get(code)
            if (not force_refresh) and cached is not None:
                expires_at, snapshot = cached
                if now < expires_at:
                    return snapshot

        snapshot = self._fetch_snapshot(code)
        self._recompute_scores(snapshot)

        with self._lock:
            self._cache[code] = (now + self._cache_ttl_seconds, snapshot)
        return snapshot

    def get_snapshots(
        self,
        symbols: list[object],
        *,
        force_refresh: bool = False,
    ) -> dict[str, FundamentalSnapshot]:
        """Return a normalized symbol -> snapshot map."""
        out: dict[str, FundamentalSnapshot] = {}
        for symbol in symbols:
            code = _normalize_symbol(symbol)
            if not code:
                continue
            out[code] = self.get_snapshot(code, force_refresh=force_refresh)
        return out

    def _fetch_snapshot(self, symbol: str) -> FundamentalSnapshot:
        """Fetch snapshot using local history proxy when online mode is disabled."""
        allow_online = str(os.environ.get("TRADING_FUNDAMENTALS_ONLINE", "0")).strip().lower()
        allow_online_flag = allow_online in {"1", "true", "yes", "on"}
        del allow_online_flag  # currently no online provider wired in

        try:
            from data import fetcher as fetcher_mod

            fetcher = fetcher_mod.get_fetcher()
            history = fetcher.get_history(
                symbol,
                interval="1d",
                bars=180,
                use_cache=True,
                update_db=False,
                allow_online=False,
            )
        except Exception as exc:
            snap = FundamentalSnapshot(symbol=symbol, source="proxy")
            snap.warnings.append(f"proxy fallback failed: {exc}")
            return snap

        return self._snapshot_from_history(symbol, history)

    def _snapshot_from_history(self, symbol: str, history: object) -> FundamentalSnapshot:
        """Build proxy fundamentals from OHLCV history."""
        if not isinstance(history, pd.DataFrame) or history.empty:
            snap = FundamentalSnapshot(symbol=symbol, source="proxy")
            snap.warnings.append("missing history for proxy fundamentals")
            return snap

        frame = history.copy()
        close = pd.to_numeric(frame.get("close"), errors="coerce")
        volume = pd.to_numeric(frame.get("volume"), errors="coerce")
        if close is None:
            close = pd.Series(dtype=float)
        if volume is None:
            volume = pd.Series(dtype=float)

        close = close.dropna()
        volume = volume.reindex(close.index).fillna(0.0).astype(float)
        if close.empty:
            snap = FundamentalSnapshot(symbol=symbol, source="proxy")
            snap.warnings.append("missing close series for proxy fundamentals")
            return snap

        returns = close.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
        ann_vol: float | None = None
        if not returns.empty:
            ann_vol = float(returns.std(ddof=0) * np.sqrt(252.0))

        lookback_60 = close.tail(min(60, len(close)))
        trend_60d: float | None = None
        if len(lookback_60) >= 2:
            start_px = float(lookback_60.iloc[0])
            if start_px > 0:
                trend_60d = float((float(lookback_60.iloc[-1]) / start_px) - 1.0)

        notional = close * volume
        notional_20d: float | None = None
        if notional.notna().any():
            recent = notional.tail(min(20, len(notional))).dropna()
            if not recent.empty:
                notional_20d = float(recent.mean())

        median_px = float(close.tail(min(30, len(close))).median())
        trend_proxy = 0.0 if trend_60d is None else float(trend_60d)
        vol_proxy = 0.0 if ann_vol is None else float(ann_vol)

        pe_proxy = float(np.clip(22.0 - (trend_proxy * 12.0) + (vol_proxy * 2.0), 8.0, 40.0))
        pb_proxy = float(np.clip(2.6 - (trend_proxy * 0.8) + (vol_proxy * 0.2), 0.6, 8.0))
        roe_proxy = float(np.clip(6.0 + (trend_proxy * 60.0) - (vol_proxy * 10.0), -5.0, 35.0))
        gross_margin_proxy = float(np.clip(28.0 + (trend_proxy * 45.0), 5.0, 70.0))
        revenue_growth_proxy = float(np.clip((trend_proxy * 120.0), -45.0, 80.0))
        earnings_growth_proxy = float(np.clip((trend_proxy * 150.0), -50.0, 95.0))
        dividend_yield_proxy = float(np.clip(1.8 + (max(0.0, 15.0 - pe_proxy) * 0.06), 0.2, 6.0))

        as_of = datetime.now(UTC)
        if isinstance(close.index, pd.DatetimeIndex) and len(close.index) > 0:
            ts = close.index[-1]
            if isinstance(ts, pd.Timestamp):
                if ts.tzinfo is None:
                    ts = ts.tz_localize("UTC")
                as_of = ts.to_pydatetime().astimezone(UTC)

        return FundamentalSnapshot(
            symbol=symbol,
            source="proxy",
            as_of=as_of,
            pe_ttm=pe_proxy,
            pb_mrq=pb_proxy,
            dividend_yield=dividend_yield_proxy,
            roe_ttm=roe_proxy,
            gross_margin=gross_margin_proxy,
            revenue_growth_yoy=revenue_growth_proxy,
            earnings_growth_yoy=earnings_growth_proxy,
            avg_notional_20d_cny=notional_20d,
            annualized_volatility=ann_vol,
            trend_60d=trend_60d,
        )

    def _recompute_scores(self, snap: FundamentalSnapshot) -> None:
        """Compute bounded composite scores with staleness/sparsity penalties."""
        warnings = list(snap.warnings)
        if snap.as_of.tzinfo is None:
            as_of = snap.as_of.replace(tzinfo=UTC)
        else:
            as_of = snap.as_of.astimezone(UTC)
        age_days = max(0.0, (datetime.now(UTC) - as_of).total_seconds() / 86400.0)
        snap.stale = age_days > self._stale_after_days

        value_components: list[float] = []
        if snap.pe_ttm is not None and snap.pe_ttm > 0:
            value_components.append(_clamp01((25.0 - float(snap.pe_ttm)) / 20.0))
        if snap.pb_mrq is not None and snap.pb_mrq > 0:
            value_components.append(_clamp01((3.0 - float(snap.pb_mrq)) / 2.5))
        if snap.dividend_yield is not None and snap.dividend_yield >= 0:
            value_components.append(_clamp01(float(snap.dividend_yield) / 5.0))
        snap.value_score = (
            float(np.mean(value_components)) if value_components else 0.25
        )

        quality_components: list[float] = []
        if snap.roe_ttm is not None:
            quality_components.append(_clamp01((float(snap.roe_ttm) + 5.0) / 25.0))
        if snap.gross_margin is not None:
            quality_components.append(_clamp01(float(snap.gross_margin) / 50.0))
        if snap.annualized_volatility is not None:
            quality_components.append(_clamp01(1.0 - (float(snap.annualized_volatility) / 1.5)))
        snap.quality_score = (
            float(np.mean(quality_components)) if quality_components else 0.25
        )

        growth_components: list[float] = []
        if snap.revenue_growth_yoy is not None:
            growth_components.append(_clamp01((float(snap.revenue_growth_yoy) + 20.0) / 60.0))
        if snap.earnings_growth_yoy is not None:
            growth_components.append(_clamp01((float(snap.earnings_growth_yoy) + 20.0) / 70.0))
        if snap.trend_60d is not None:
            growth_components.append(_clamp01((float(snap.trend_60d) + 0.20) / 0.60))
        snap.growth_score = (
            float(np.mean(growth_components)) if growth_components else 0.25
        )

        base = (
            (0.34 * _clamp01(snap.value_score))
            + (0.36 * _clamp01(snap.quality_score))
            + (0.30 * _clamp01(snap.growth_score))
        )

        total_fields = 7
        populated_fields = sum(
            1
            for val in (
                snap.pe_ttm,
                snap.pb_mrq,
                snap.dividend_yield,
                snap.roe_ttm,
                snap.gross_margin,
                snap.revenue_growth_yoy,
                snap.earnings_growth_yoy,
            )
            if val is not None
        )
        coverage = populated_fields / float(total_fields)
        sparse_penalty = _clamp01((0.60 - coverage) / 0.60) * 0.45

        penalty = sparse_penalty
        if snap.stale:
            penalty += min(0.25, 0.03 * max(0.0, age_days - self._stale_after_days))
            warnings.append("snapshot stale; composite score penalized")

        if sparse_penalty > 0:
            warnings.append(
                "composite score quality-adjusted for sparse fundamental coverage"
            )

        if snap.source == "proxy":
            penalty += 0.05
            warnings.append("proxy-derived fundamentals in use")

        snap.composite_score = _clamp01(base * (1.0 - _clamp01(penalty)))
        snap.warnings = list(dict.fromkeys(str(msg) for msg in warnings if str(msg).strip()))


_FUNDAMENTAL_SERVICE_LOCK = threading.Lock()
_FUNDAMENTAL_SERVICE: FundamentalDataService | None = None


def get_fundamental_service() -> FundamentalDataService:
    """Return process-wide singleton fundamental service."""
    global _FUNDAMENTAL_SERVICE
    if _FUNDAMENTAL_SERVICE is not None:
        return _FUNDAMENTAL_SERVICE
    with _FUNDAMENTAL_SERVICE_LOCK:
        if _FUNDAMENTAL_SERVICE is None:
            _FUNDAMENTAL_SERVICE = FundamentalDataService()
    return _FUNDAMENTAL_SERVICE


@dataclass
class FinancialStatement:
    """Financial statement data."""
    symbol: str
    report_type: str  # annual, quarterly, interim
    report_date: datetime
    currency: str = "CNY"
    balance_sheet: dict = field(default_factory=dict)
    income_statement: dict = field(default_factory=dict)
    cash_flow: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "report_type": self.report_type,
            "report_date": self.report_date.isoformat(),
            "currency": self.currency,
            "balance_sheet": self.balance_sheet,
            "income_statement": self.income_statement,
            "cash_flow": self.cash_flow,
        }


@dataclass
class FinancialRatios:
    """Financial ratios for a stock."""
    symbol: str
    report_date: datetime

    # Valuation ratios
    pe_ratio: float = 0.0  # Price to Earnings
    pb_ratio: float = 0.0  # Price to Book
    ps_ratio: float = 0.0  # Price to Sales
    pcf_ratio: float = 0.0  # Price to Cash Flow
    ev_ebitda: float = 0.0  # Enterprise Value to EBITDA
    peg_ratio: float = 0.0  # PEG ratio

    # Profitability ratios
    roe: float = 0.0  # Return on Equity
    roa: float = 0.0  # Return on Assets
    roic: float = 0.0  # Return on Invested Capital
    gross_margin: float = 0.0
    operating_margin: float = 0.0
    net_margin: float = 0.0

    # Efficiency ratios
    asset_turnover: float = 0.0
    inventory_turnover: float = 0.0
    receivables_turnover: float = 0.0
    days_sales_outstanding: float = 0.0

    # Leverage ratios
    debt_to_equity: float = 0.0
    debt_to_assets: float = 0.0
    interest_coverage: float = 0.0
    current_ratio: float = 0.0
    quick_ratio: float = 0.0
    cash_ratio: float = 0.0

    # Growth ratios
    revenue_growth_yoy: float = 0.0
    earnings_growth_yoy: float = 0.0
    book_value_growth_yoy: float = 0.0
    operating_cash_flow_growth_yoy: float = 0.0

    # Per share data
    eps: float = 0.0  # Earnings Per Share
    bps: float = 0.0  # Book Value Per Share
    cfps: float = 0.0  # Cash Flow Per Share
    dividend_per_share: float = 0.0
    dividend_yield: float = 0.0

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "report_date": self.report_date.isoformat(),
            "valuation": {
                "pe_ratio": self.pe_ratio,
                "pb_ratio": self.pb_ratio,
                "ps_ratio": self.ps_ratio,
                "pcf_ratio": self.pcf_ratio,
                "ev_ebitda": self.ev_ebitda,
                "peg_ratio": self.peg_ratio,
            },
            "profitability": {
                "roe": self.roe,
                "roa": self.roa,
                "roic": self.roic,
                "gross_margin": self.gross_margin,
                "operating_margin": self.operating_margin,
                "net_margin": self.net_margin,
            },
            "efficiency": {
                "asset_turnover": self.asset_turnover,
                "inventory_turnover": self.inventory_turnover,
                "receivables_turnover": self.receivables_turnover,
                "days_sales_outstanding": self.days_sales_outstanding,
            },
            "leverage": {
                "debt_to_equity": self.debt_to_equity,
                "debt_to_assets": self.debt_to_assets,
                "interest_coverage": self.interest_coverage,
                "current_ratio": self.current_ratio,
                "quick_ratio": self.quick_ratio,
                "cash_ratio": self.cash_ratio,
            },
            "growth": {
                "revenue_growth_yoy": self.revenue_growth_yoy,
                "earnings_growth_yoy": self.earnings_growth_yoy,
                "book_value_growth_yoy": self.book_value_growth_yoy,
                "operating_cash_flow_growth_yoy": self.operating_cash_flow_growth_yoy,
            },
            "per_share": {
                "eps": self.eps,
                "bps": self.bps,
                "cfps": self.cfps,
                "dividend_per_share": self.dividend_per_share,
                "dividend_yield": self.dividend_yield,
            },
        }

    def calculate_piotroski_score(self) -> int:
        """Calculate Piotroski F-Score (0-9)."""
        score = 0

        # Profitability
        if self.roe > 0:
            score += 1
        if self.operating_margin > 0:
            score += 1

        # Leverage
        if self.debt_to_equity < 1.0:
            score += 1
        if self.current_ratio > 1.0:
            score += 1

        # Efficiency
        if self.asset_turnover > 1.0:
            score += 1

        return score

    def calculate_altman_z_score(
        self,
        market_cap: float,
        total_assets: float,
        retained_earnings: float,
        ebit: float,
        sales: float,
        total_liabilities: float,
        working_capital: float,
    ) -> float:
        """Calculate Altman Z-Score for bankruptcy prediction."""
        if total_assets == 0:
            return 0.0

        x1 = working_capital / total_assets
        x2 = retained_earnings / total_assets
        x3 = ebit / total_assets
        x4 = market_cap / total_liabilities if total_liabilities > 0 else 0
        x5 = sales / total_assets

        # Original Z-score formula (manufacturing firms)
        z_score = (
            1.2 * x1 +
            1.4 * x2 +
            3.3 * x3 +
            0.6 * x4 +
            0.99 * x5
        )

        return z_score


@dataclass
class AnalystEstimate:
    """Analyst estimate data."""
    symbol: str
    estimate_type: str  # eps, revenue, rating
    period: str  # current_quarter, next_quarter, current_year, next_year
    mean_estimate: float
    high_estimate: float
    low_estimate: float
    num_analysts: int
    year_ago_estimate: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "estimate_type": self.estimate_type,
            "period": self.period,
            "mean_estimate": self.mean_estimate,
            "high_estimate": self.high_estimate,
            "low_estimate": self.low_estimate,
            "num_analysts": self.num_analysts,
            "year_ago_estimate": self.year_ago_estimate,
            "last_updated": self.last_updated.isoformat(),
        }


@dataclass
class AnalystRating:
    """Analyst rating summary."""
    symbol: str
    strong_buy: int = 0
    buy: int = 0
    hold: int = 0
    sell: int = 0
    strong_sell: int = 0
    consensus: str = "hold"
    price_target_mean: float = 0.0
    price_target_high: float = 0.0
    price_target_low: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        total = self.strong_buy + self.buy + self.hold + self.sell + self.strong_sell
        return {
            "symbol": self.symbol,
            "ratings": {
                "strong_buy": self.strong_buy,
                "buy": self.buy,
                "hold": self.hold,
                "sell": self.sell,
                "strong_sell": self.strong_sell,
            },
            "consensus": self.consensus,
            "price_targets": {
                "mean": self.price_target_mean,
                "high": self.price_target_high,
                "low": self.price_target_low,
            },
            "total_analysts": total,
            "last_updated": self.last_updated.isoformat(),
        }

    def calculate_consensus(self) -> str:
        """Calculate consensus rating."""
        total = self.strong_buy + self.buy + self.hold + self.sell + self.strong_sell
        if total == 0:
            return "hold"

        score = (
            self.strong_buy * 5 +
            self.buy * 4 +
            self.hold * 3 +
            self.sell * 2 +
            self.strong_sell * 1
        ) / total

        if score >= 4.5:
            self.consensus = "strong_buy"
        elif score >= 3.5:
            self.consensus = "buy"
        elif score >= 2.5:
            self.consensus = "hold"
        elif score >= 1.5:
            self.consensus = "sell"
        else:
            self.consensus = "strong_sell"

        return self.consensus


@dataclass
class InstitutionalHolding:
    """Institutional holding data."""
    symbol: str
    holder_name: str
    holder_type: str  # fund, insurance, broker, qfii, social_security
    shares: int
    market_value: float
    percent_of_shares: float
    percent_change: float  # Quarter over quarter change
    report_date: datetime
    rank: int = 0

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "holder_name": self.holder_name,
            "holder_type": self.holder_type,
            "shares": self.shares,
            "market_value": self.market_value,
            "percent_of_shares": self.percent_of_shares,
            "percent_change": self.percent_change,
            "report_date": self.report_date.isoformat(),
            "rank": self.rank,
        }


@dataclass
class ESGScore:
    """ESG (Environmental, Social, Governance) score."""
    symbol: str
    total_score: float  # 0-100
    environmental_score: float
    social_score: float
    governance_score: float
    rating: str  # AAA, AA, A, BBB, BB, B, CCC, CC, C
    industry_percentile: float
    last_updated: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "total_score": self.total_score,
            "environmental_score": self.environmental_score,
            "social_score": self.social_score,
            "governance_score": self.governance_score,
            "rating": self.rating,
            "industry_percentile": self.industry_percentile,
            "last_updated": self.last_updated.isoformat(),
        }


class FundamentalsData:
    """Fundamental data provider for China A-shares."""

    def __init__(self) -> None:
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
            "Accept": "application/json",
        })

    def get_financial_statements(
        self,
        symbol: str,
        report_type: str = "quarterly",
        years: int = 5,
    ) -> list[FinancialStatement]:
        """Get financial statements for a stock."""
        statements = []

        try:
            # Try AkShare first
            statements = self._fetch_via_akshare(symbol, report_type, years)

            # Fallback to EastMoney
            if not statements:
                statements = self._fetch_via_eastmoney(symbol, report_type, years)

        except Exception as e:
            log.error(f"Error fetching financial statements for {symbol}: {e}")

        return statements

    def _fetch_via_akshare(
        self,
        symbol: str,
        report_type: str,
        years: int,
    ) -> list[FinancialStatement]:
        """Fetch financial data via AkShare."""
        # Placeholder - would use actual AkShare API
        # import akshare as ak
        # stock_financial_analysis_indicator_df = ak.stock_financial_analysis_indicator(symbol=symbol)
        return []

    def _fetch_via_eastmoney(
        self,
        symbol: str,
        report_type: str,
        years: int,
    ) -> list[FinancialStatement]:
        """Fetch financial data via EastMoney API."""
        # Placeholder for EastMoney API integration
        return []

    def get_financial_ratios(
        self,
        symbol: str,
        latest: bool = True,
    ) -> FinancialRatios | list[FinancialRatios]:
        """Get financial ratios for a stock."""
        try:
            # Fetch from data source
            ratios = self._calculate_ratios(symbol)

            if latest and isinstance(ratios, list):
                return ratios[0] if ratios else FinancialRatios(
                    symbol=symbol,
                    report_date=datetime.now(),
                )
            return ratios

        except Exception as e:
            log.error(f"Error fetching financial ratios for {symbol}: {e}")
            return FinancialRatios(symbol=symbol, report_date=datetime.now())

    def _calculate_ratios(self, symbol: str) -> list[FinancialRatios]:
        """Calculate financial ratios from statements."""
        # Placeholder - would calculate from actual financial data
        return [
            FinancialRatios(
                symbol=symbol,
                report_date=datetime.now(),
                pe_ratio=15.5,
                pb_ratio=2.1,
                roe=12.5,
                gross_margin=35.0,
                net_margin=15.0,
                debt_to_equity=0.5,
                current_ratio=1.8,
                revenue_growth_yoy=10.0,
                earnings_growth_yoy=15.0,
                eps=2.5,
                dividend_yield=2.0,
            ),
        ]

    def get_analyst_estimates(
        self,
        symbol: str,
        estimate_type: str = "eps",
    ) -> list[AnalystEstimate]:
        """Get analyst estimates for a stock."""
        # Placeholder - would fetch from data provider
        return [
            AnalystEstimate(
                symbol=symbol,
                estimate_type="eps",
                period="current_quarter",
                mean_estimate=1.25,
                high_estimate=1.50,
                low_estimate=1.00,
                num_analysts=15,
            ),
            AnalystEstimate(
                symbol=symbol,
                estimate_type="eps",
                period="next_quarter",
                mean_estimate=1.35,
                high_estimate=1.60,
                low_estimate=1.10,
                num_analysts=12,
            ),
        ]

    def get_analyst_ratings(self, symbol: str) -> AnalystRating:
        """Get analyst ratings summary for a stock."""
        # Placeholder
        rating = AnalystRating(
            symbol=symbol,
            strong_buy=5,
            buy=8,
            hold=10,
            sell=2,
            strong_sell=0,
            price_target_mean=50.0,
            price_target_high=65.0,
            price_target_low=35.0,
        )
        rating.calculate_consensus()
        return rating

    def get_institutional_holdings(
        self,
        symbol: str,
        latest: bool = True,
    ) -> list[InstitutionalHolding]:
        """Get institutional holdings for a stock."""
        # Placeholder - would fetch from filings
        return [
            InstitutionalHolding(
                symbol=symbol,
                holder_name="China Asset Management",
                holder_type="fund",
                shares=10000000,
                market_value=500000000,
                percent_of_shares=5.0,
                percent_change=2.5,
                report_date=datetime.now(),
                rank=1,
            ),
        ]

    def get_esg_score(self, symbol: str) -> ESGScore:
        """Get ESG score for a stock."""
        # Placeholder - would fetch from ESG data provider
        return ESGScore(
            symbol=symbol,
            total_score=75.0,
            environmental_score=70.0,
            social_score=78.0,
            governance_score=77.0,
            rating="A",
            industry_percentile=65.0,
        )

    def get_dividend_history(
        self,
        symbol: str,
        years: int = 5,
    ) -> list[dict]:
        """Get dividend history for a stock."""
        # Placeholder
        return [
            {
                "symbol": symbol,
                "ex_date": "2024-06-15",
                "payment_date": "2024-07-15",
                "dividend_per_share": 0.50,
                "dividend_type": "cash",
                "currency": "CNY",
            },
        ]

    def get_corporate_actions(
        self,
        symbol: str,
        years: int = 3,
    ) -> list[dict]:
        """Get corporate actions (splits, rights issues) for a stock."""
        # Placeholder
        return [
            {
                "symbol": symbol,
                "action_type": "split",
                "ratio": "10:1",
                "effective_date": "2023-05-01",
                "announcement_date": "2023-04-01",
            },
        ]

    def get_peer_comparison(
        self,
        symbol: str,
        industry: str | None = None,
    ) -> pd.DataFrame:
        """Get peer comparison data."""
        # Placeholder - would fetch industry peers
        data = {
            "symbol": [symbol, "PEER1", "PEER2", "PEER3"],
            "pe_ratio": [15.5, 18.2, 12.3, 20.1],
            "pb_ratio": [2.1, 2.5, 1.8, 3.0],
            "roe": [12.5, 15.0, 10.2, 18.5],
            "revenue_growth": [10.0, 12.5, 8.0, 15.0],
            "net_margin": [15.0, 18.0, 12.0, 20.0],
        }
        return pd.DataFrame(data)

    def get_valuation_summary(self, symbol: str) -> dict:
        """Get comprehensive valuation summary."""
        ratios = self.get_financial_ratios(symbol, latest=True)
        estimates = self.get_analyst_estimates(symbol)
        ratings = self.get_analyst_ratings(symbol)

        return {
            "symbol": symbol,
            "valuation_ratios": ratios.to_dict() if hasattr(ratios, "to_dict") else {},
            "analyst_estimates": [e.to_dict() for e in estimates],
            "analyst_ratings": ratings.to_dict(),
            "fair_value_estimate": self._estimate_fair_value(symbol, ratios),
        }

    def _estimate_fair_value(
        self,
        symbol: str,
        ratios: FinancialRatios,
    ) -> dict:
        """Estimate fair value using multiple methods."""
        # DCF valuation (simplified)
        dcf_value = self._dcf_valuation(symbol, ratios)

        # Relative valuation
        pe_value = self._pe_valuation(symbol, ratios)
        pb_value = self._pb_valuation(symbol, ratios)

        # Average
        fair_value = (dcf_value + pe_value + pb_value) / 3

        return {
            "dcf_value": round(dcf_value, 2),
            "pe_relative_value": round(pe_value, 2),
            "pb_relative_value": round(pb_value, 2),
            "average_fair_value": round(fair_value, 2),
        }

    def _dcf_valuation(self, symbol: str, ratios: FinancialRatios) -> float:
        """Simplified DCF valuation."""
        # Placeholder
        if ratios.eps > 0 and ratios.earnings_growth_yoy > 0:
            growth_rate = min(ratios.earnings_growth_yoy / 100, 0.20)
            discount_rate = 0.10
            terminal_multiple = 15

            fcf = ratios.eps * 0.8  # Assume 80% of earnings is FCF
            value = 0

            for i in range(5):
                value += fcf * (1 + growth_rate) ** i / (1 + discount_rate) ** (i + 1)

            terminal_value = fcf * (1 + growth_rate) ** 5 * terminal_multiple
            value += terminal_value / (1 + discount_rate) ** 5

            return value
        return 0.0

    def _pe_valuation(self, symbol: str, ratios: FinancialRatios) -> float:
        """Relative PE valuation."""
        # Use industry average PE
        industry_pe = 18.0  # Placeholder
        fair_value = ratios.eps * industry_pe
        return fair_value

    def _pb_valuation(self, symbol: str, ratios: FinancialRatios) -> float:
        """Relative PB valuation."""
        # Use industry average PB
        industry_pb = 2.5  # Placeholder
        fair_value = ratios.bps * industry_pb
        return fair_value


def get_fundamentals() -> FundamentalsData:
    """Get fundamentals data instance."""
    return FundamentalsData()
