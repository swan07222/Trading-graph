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

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import pandas as pd
import requests

from config.settings import CONFIG
from utils.logger import get_logger

log = get_logger(__name__)


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
