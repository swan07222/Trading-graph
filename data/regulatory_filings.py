# data/regulatory_filings.py
"""Regulatory Filings Analysis for China Markets.

This module collects and analyzes regulatory filings from:
- CSRC (中国证监会) - China Securities Regulatory Commission
- SSE (上海证券交易所) - Shanghai Stock Exchange
- SZSE (深圳证券交易所) - Shenzhen Stock Exchange  
- BSE (北京证券交易所) - Beijing Stock Exchange
- CNINFO (巨潮资讯网) - Designated disclosure website

Filing Types:
- Annual reports (年报)
- Quarterly reports (季报)
- IPO prospectuses (招股说明书)
- Material asset restructuring (重大资产重组)
- Related party transactions (关联交易)
- Shareholder changes (股东变动)
- Pledge of shares (股权质押)
- Regulatory penalties (监管处罚)
- Trading hal announcements (停牌公告)
"""

import hashlib
import json
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

import pandas as pd
import requests

from config.settings import CONFIG
from utils.logger import get_logger

from .news_collector import NewsArticle

log = get_logger(__name__)


@dataclass
class RegulatoryFiling:
    """Regulatory filing document."""
    id: str
    filing_type: str
    title: str
    symbol: str
    company_name: str
    exchange: str  # SSE, SZSE, BSE
    published_at: datetime
    filed_at: datetime
    collected_at: datetime
    url: str
    content: str = ""
    summary: str = ""
    sentiment_score: float = 0.0
    impact_score: float = 0.0  # 0-100
    categories: list[str] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)
    attachments: list[dict] = field(default_factory=list)
    language: str = "zh"

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "filing_type": self.filing_type,
            "title": self.title,
            "symbol": self.symbol,
            "company_name": self.company_name,
            "exchange": self.exchange,
            "published_at": self.published_at.isoformat(),
            "filed_at": self.filed_at.isoformat(),
            "collected_at": self.collected_at.isoformat(),
            "url": self.url,
            "content": self.content,
            "summary": self.summary,
            "sentiment_score": self.sentiment_score,
            "impact_score": self.impact_score,
            "categories": self.categories,
            "entities": self.entities,
            "attachments": self.attachments,
            "language": self.language,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RegulatoryFiling":
        try:
            published_at = datetime.fromisoformat(data["published_at"])
        except (KeyError, ValueError, TypeError):
            published_at = datetime.now()
        try:
            filed_at = datetime.fromisoformat(data["filed_at"])
        except (KeyError, ValueError, TypeError):
            filed_at = datetime.now()
        try:
            collected_at = datetime.fromisoformat(data["collected_at"])
        except (KeyError, ValueError, TypeError):
            collected_at = datetime.now()
        return cls(
            id=data["id"],
            filing_type=data["filing_type"],
            title=data["title"],
            symbol=data["symbol"],
            company_name=data["company_name"],
            exchange=data.get("exchange", "SSE"),
            published_at=published_at,
            filed_at=filed_at,
            collected_at=collected_at,
            url=data["url"],
            content=data.get("content", ""),
            summary=data.get("summary", ""),
            sentiment_score=float(data.get("sentiment_score", 0.0)),
            impact_score=float(data.get("impact_score", 0.0)),
            categories=data.get("categories", []),
            entities=data.get("entities", []),
            attachments=data.get("attachments", []),
            language=data.get("language", "zh"),
        )


@dataclass
class FilingAlert:
    """Alert for significant regulatory filing."""
    alert_type: str
    severity: str  # critical, high, medium, low
    symbol: str
    filing: RegulatoryFiling
    message: str
    recommended_action: str
    created_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        return {
            "alert_type": self.alert_type,
            "severity": self.severity,
            "symbol": self.symbol,
            "filing_id": self.filing.id,
            "message": self.message,
            "recommended_action": self.recommended_action,
            "created_at": self.created_at.isoformat(),
        }


class RegulatoryFilingsCollector:
    """Collector for regulatory filings."""

    # Official sources
    SOURCES = {
        "cninfo": {
            "name": "巨潮资讯网",
            "base_url": "http://www.cninfo.com.cn",
            "search_url": "http://www.cninfo.com.cn/new/commonUrl/pageOfSearch",
            "enabled": True,
        },
        "sse": {
            "name": "上海证券交易所",
            "base_url": "http://www.sse.com.cn",
            "disclosure_url": "http://www.sse.com.cn/disclosure/",
            "enabled": True,
        },
        "szse": {
            "name": "深圳证券交易所",
            "base_url": "http://www.szse.cn",
            "disclosure_url": "http://www.szse.cn/disclosure/",
            "enabled": True,
        },
        "bse": {
            "name": "北京证券交易所",
            "base_url": "http://www.bse.cn",
            "disclosure_url": "http://www.bse.cn/disclosure/",
            "enabled": True,
        },
        "csrc": {
            "name": "中国证监会",
            "base_url": "http://www.csrc.gov.cn",
            "enabled": True,
        },
    }

    # Filing type mappings
    FILING_TYPES = {
        "annual_report": ["年报", "年度报告", "annual report"],
        "quarterly_report": ["季报", "季度报告", "quarterly report"],
        "ipo_prospectus": ["招股说明书", "prospectus"],
        "asset_restructuring": ["重大资产重组", "asset restructuring"],
        "related_transaction": ["关联交易", "related party transaction"],
        "shareholder_change": ["股东变动", "shareholder change"],
        "share_pledge": ["股权质押", "share pledge"],
        "regulatory_penalty": ["监管处罚", "regulatory penalty", "处罚"],
        "trading_halt": ["停牌公告", "trading halt"],
        "earnings_forecast": ["业绩预告", "earnings forecast"],
        "dividend_announcement": ["分红公告", "dividend announcement"],
        "insider_trading": ["董监高买卖", "insider trading"],
    }

    def __init__(self) -> None:
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
            "Accept": "application/json, text/html",
            "Accept-Language": "zh-CN,zh;q=0.9",
        })
        self._rate_limits: dict[str, float] = {}

    def _check_rate_limit(self, source: str) -> bool:
        """Check rate limit for source."""
        last_request = self._rate_limits.get(source, 0)
        min_interval = 1.0
        return (time.time() - last_request) >= min_interval

    def _update_rate_limit(self, source: str) -> None:
        """Update rate limit timestamp."""
        self._rate_limits[source] = time.time()

    def _classify_filing_type(self, title: str) -> str:
        """Classify filing type based on title."""
        title_lower = title.lower()
        for filing_type, keywords in self.FILING_TYPES.items():
            for keyword in keywords:
                if keyword.lower() in title_lower:
                    return filing_type
        return "other"

    def _calculate_impact_score(
        self,
        filing_type: str,
        content: str,
    ) -> float:
        """Calculate impact score (0-100) for filing."""
        base_scores = {
            "annual_report": 30,
            "quarterly_report": 20,
            "ipo_prospectus": 50,
            "asset_restructuring": 80,
            "related_transaction": 40,
            "shareholder_change": 35,
            "share_pledge": 45,
            "regulatory_penalty": 70,
            "trading_halt": 60,
            "earnings_forecast": 55,
            "dividend_announcement": 40,
            "insider_trading": 50,
            "other": 10,
        }

        score = base_scores.get(filing_type, 10)

        # Adjust based on content keywords
        high_impact_keywords = [
            "重大", "特别", "异常", "风险", "警告", "处罚",
            "重大风险", "特别处理", "退市", "调查",
        ]
        for keyword in high_impact_keywords:
            if keyword in content:
                score = min(100, score + 10)

        return score

    def fetch_cninfo_filings(
        self,
        symbols: list[str] | None = None,
        filing_types: list[str] | None = None,
        days_back: int = 30,
        limit: int = 100,
    ) -> list[RegulatoryFiling]:
        """Fetch filings from CNINFO (巨潮资讯网)."""
        filings = []

        if not self._check_rate_limit("cninfo"):
            return filings

        try:
            # Placeholder - would need actual API integration
            # CNINFO requires JavaScript rendering, so would need Selenium/Playwright
            log.info("Fetching from CNINFO...")

            # Simulated response
            if symbols:
                for symbol in symbols[:5]:
                    filing = RegulatoryFiling(
                        id=f"cninfo_{symbol}_{int(time.time())}",
                        filing_type="quarterly_report",
                        title=f"{symbol} 2024 年第三季度报告",
                        symbol=symbol,
                        company_name=f"公司{symbol}",
                        exchange="SZSE",
                        published_at=datetime.now() - timedelta(days=5),
                        filed_at=datetime.now() - timedelta(days=5),
                        collected_at=datetime.now(),
                        url=f"http://www.cninfo.com.cn/{symbol}",
                        summary="公司发布第三季度报告",
                        categories=["定期报告", "季度报告"],
                    )
                    filing.impact_score = self._calculate_impact_score(
                        filing.filing_type, filing.summary
                    )
                    filings.append(filing)

            self._update_rate_limit("cninfo")

        except Exception as e:
            log.error(f"CNINFO fetch error: {e}")

        return filings[:limit]

    def fetch_sse_filings(
        self,
        symbols: list[str] | None = None,
        filing_types: list[str] | None = None,
        days_back: int = 30,
        limit: int = 100,
    ) -> list[RegulatoryFiling]:
        """Fetch filings from Shanghai Stock Exchange."""
        filings = []

        if not self._check_rate_limit("sse"):
            return filings

        try:
            log.info("Fetching from SSE...")

            # Placeholder for actual API
            if symbols:
                for symbol in symbols[:5]:
                    if symbol.startswith("6"):  # SSE stocks start with 6
                        filing = RegulatoryFiling(
                            id=f"sse_{symbol}_{int(time.time())}",
                            filing_type="trading_halt",
                            title=f"{symbol} 停牌公告",
                            symbol=symbol,
                            company_name=f"上海公司{symbol}",
                            exchange="SSE",
                            published_at=datetime.now() - timedelta(days=1),
                            filed_at=datetime.now() - timedelta(days=1),
                            collected_at=datetime.now(),
                            url=f"http://www.sse.com.cn/{symbol}",
                            summary="公司股票临时停牌",
                            categories=["停牌公告"],
                        )
                        filing.impact_score = self._calculate_impact_score(
                            filing.filing_type, filing.summary
                        )
                        filings.append(filing)

            self._update_rate_limit("sse")

        except Exception as e:
            log.error(f"SSE fetch error: {e}")

        return filings[:limit]

    def fetch_szse_filings(
        self,
        symbols: list[str] | None = None,
        filing_types: list[str] | None = None,
        days_back: int = 30,
        limit: int = 100,
    ) -> list[RegulatoryFiling]:
        """Fetch filings from Shenzhen Stock Exchange."""
        filings = []

        if not self._check_rate_limit("szse"):
            return filings

        try:
            log.info("Fetching from SZSE...")

            # Placeholder
            if symbols:
                for symbol in symbols[:5]:
                    if symbol.startswith(("0", "3")):  # SZSE stocks
                        filing = RegulatoryFiling(
                            id=f"szse_{symbol}_{int(time.time())}",
                            filing_type="earnings_forecast",
                            title=f"{symbol} 业绩预告公告",
                            symbol=symbol,
                            company_name=f"深圳公司{symbol}",
                            exchange="SZSE",
                            published_at=datetime.now() - timedelta(days=2),
                            filed_at=datetime.now() - timedelta(days=2),
                            collected_at=datetime.now(),
                            url=f"http://www.szse.cn/{symbol}",
                            summary="公司发布年度业绩预告",
                            categories=["业绩预告"],
                        )
                        filing.impact_score = self._calculate_impact_score(
                            filing.filing_type, filing.summary
                        )
                        filings.append(filing)

            self._update_rate_limit("szse")

        except Exception as e:
            log.error(f"SZSE fetch error: {e}")

        return filings[:limit]

    def fetch_bse_filings(
        self,
        symbols: list[str] | None = None,
        filing_types: list[str] | None = None,
        days_back: int = 30,
        limit: int = 100,
    ) -> list[RegulatoryFiling]:
        """Fetch filings from Beijing Stock Exchange."""
        filings = []

        if not self._check_rate_limit("bse"):
            return filings

        try:
            log.info("Fetching from BSE...")

            # Placeholder
            if symbols:
                for symbol in symbols[:5]:
                    if symbol.startswith(("4", "8")):  # BSE stocks
                        filing = RegulatoryFiling(
                            id=f"bse_{symbol}_{int(time.time())}",
                            filing_type="annual_report",
                            title=f"{symbol} 2023 年年度报告",
                            symbol=symbol,
                            company_name=f"北京公司{symbol}",
                            exchange="BSE",
                            published_at=datetime.now() - timedelta(days=10),
                            filed_at=datetime.now() - timedelta(days=10),
                            collected_at=datetime.now(),
                            url=f"http://www.bse.cn/{symbol}",
                            summary="公司发布年度报告",
                            categories=["定期报告", "年度报告"],
                        )
                        filing.impact_score = self._calculate_impact_score(
                            filing.filing_type, filing.summary
                        )
                        filings.append(filing)

            self._update_rate_limit("bse")

        except Exception as e:
            log.error(f"BSE fetch error: {e}")

        return filings[:limit]

    def fetch_all_filings(
        self,
        symbols: list[str] | None = None,
        filing_types: list[str] | None = None,
        days_back: int = 30,
        limit: int = 200,
    ) -> list[RegulatoryFiling]:
        """Fetch filings from all sources."""
        all_filings = []

        # Fetch from each exchange
        all_filings.extend(self.fetch_cninfo_filings(symbols, filing_types, days_back, limit // 4))
        all_filings.extend(self.fetch_sse_filings(symbols, filing_types, days_back, limit // 4))
        all_filings.extend(self.fetch_szse_filings(symbols, filing_types, days_back, limit // 4))
        all_filings.extend(self.fetch_bse_filings(symbols, filing_types, days_back, limit // 4))

        # Deduplicate by ID
        seen = set()
        unique_filings = []
        for filing in all_filings:
            if filing.id not in seen:
                seen.add(filing.id)
                unique_filings.append(filing)

        # Sort by published date
        unique_filings.sort(key=lambda x: x.published_at, reverse=True)

        return unique_filings[:limit]


class RegulatoryFilingsAnalyzer:
    """Analyzer for regulatory filings."""

    # Alert patterns
    ALERT_PATTERNS = {
        "regulatory_investigation": [
            "立案调查", "调查", "稽查", "立案",
        ],
        "financial_irregularity": [
            "财务造假", "虚假记载", "误导性陈述", "重大遗漏",
        ],
        "delisting_risk": [
            "退市风险", "暂停上市", "终止上市", "*ST",
        ],
        "major_penalty": [
            "行政处罚", "罚款", "警示函", "监管措施",
        ],
        "share_pledge_risk": [
            "质押平仓", "强制平仓", "质押风险",
        ],
        "liquidity_crisis": [
            "资金链断裂", "债务违约", "无法偿还",
        ],
    }

    def __init__(self) -> None:
        self._filing_history: dict[str, list[RegulatoryFiling]] = {}

    def analyze_filing(self, filing: RegulatoryFiling) -> dict[str, Any]:
        """Analyze a single filing."""
        result = {
            "filing_id": filing.id,
            "symbol": filing.symbol,
            "filing_type": filing.filing_type,
            "sentiment": self._analyze_sentiment(filing),
            "impact": filing.impact_score,
            "alerts": self._check_alerts(filing),
            "entities": self._extract_entities(filing),
            "risk_level": self._assess_risk(filing),
        }
        return result

    def _analyze_sentiment(self, filing: RegulatoryFiling) -> float:
        """Analyze sentiment of filing (-1.0 to 1.0)."""
        content = (filing.title + " " + filing.summary + " " + filing.content).lower()

        negative_terms = [
            "亏损", "下降", "风险", "警告", "处罚", "调查",
            "违规", "诉讼", "仲裁", "减值", "下滑",
        ]
        positive_terms = [
            "增长", "盈利", "突破", "利好", "奖励", "表彰",
        ]

        score = 0.0
        total = 0

        for term in negative_terms:
            if term in content:
                score -= 1
                total += 1

        for term in positive_terms:
            if term in content:
                score += 1
                total += 1

        if total == 0:
            return 0.0

        return score / total

    def _check_alerts(self, filing: RegulatoryFiling) -> list[FilingAlert]:
        """Check if filing triggers any alerts."""
        alerts = []
        content = filing.title + " " + filing.summary + " " + filing.content

        for alert_type, patterns in self.ALERT_PATTERNS.items():
            for pattern in patterns:
                if pattern in content:
                    severity = self._determine_severity(alert_type, filing)
                    alert = FilingAlert(
                        alert_type=alert_type,
                        severity=severity,
                        symbol=filing.symbol,
                        filing=filing,
                        message=f"Detected {alert_type} in {filing.filing_type}",
                        recommended_action=self._get_recommendation(alert_type),
                    )
                    alerts.append(alert)
                    break

        return alerts

    def _determine_severity(
        self,
        alert_type: str,
        filing: RegulatoryFiling,
    ) -> str:
        """Determine alert severity."""
        critical_types = ["delisting_risk", "financial_irregularity"]
        high_types = ["regulatory_investigation", "major_penalty"]
        medium_types = ["share_pledge_risk", "liquidity_crisis"]

        if alert_type in critical_types:
            return "critical"
        elif alert_type in high_types:
            return "high"
        elif alert_type in medium_types:
            return "medium"
        return "low"

    def _get_recommendation(self, alert_type: str) -> str:
        """Get recommended action for alert type."""
        recommendations = {
            "regulatory_investigation": "Monitor closely, consider reducing position",
            "financial_irregularity": "Strong sell signal, exit position immediately",
            "delisting_risk": "Critical risk, liquidate position",
            "major_penalty": "High risk, consider reducing exposure",
            "share_pledge_risk": "Monitor for forced liquidation risk",
            "liquidity_crisis": "High default risk, reduce exposure",
        }
        return recommendations.get(alert_type, "Review filing details")

    def _extract_entities(self, filing: RegulatoryFiling) -> list[str]:
        """Extract entities from filing."""
        entities = []

        # Extract company names
        company_pattern = r'([\u4e00-\u9fa5]{2,}股份有限公司)'
        companies = re.findall(company_pattern, filing.content)
        entities.extend(companies)

        # Extract person names
        person_pattern = r'([A-Z][\u4e00-\u9fa5]{2,4})'
        persons = re.findall(person_pattern, filing.content)
        entities.extend(persons)

        # Extract amounts
        amount_pattern = r'([\d,]+\.?\d*[亿元万元])'
        amounts = re.findall(amount_pattern, filing.content)
        entities.extend(amounts)

        return list(set(entities))[:20]

    def _assess_risk(self, filing: RegulatoryFiling) -> str:
        """Assess overall risk level."""
        if filing.impact_score >= 70:
            return "critical"
        elif filing.impact_score >= 50:
            return "high"
        elif filing.impact_score >= 30:
            return "medium"
        return "low"

    def generate_summary_report(
        self,
        filings: list[RegulatoryFiling],
        symbols: list[str] | None = None,
    ) -> dict[str, Any]:
        """Generate summary report for filings."""
        if symbols:
            filings = [f for f in filings if f.symbol in symbols]

        # Group by symbol
        by_symbol: dict[str, list[RegulatoryFiling]] = {}
        for filing in filings:
            if filing.symbol not in by_symbol:
                by_symbol[filing.symbol] = []
            by_symbol[filing.symbol].append(filing)

        report = {
            "total_filings": len(filings),
            "symbols_covered": len(by_symbol),
            "by_filing_type": {},
            "by_exchange": {},
            "alerts": [],
            "high_impact_filings": [],
        }

        # Count by filing type
        for filing in filings:
            ft = filing.filing_type
            if ft not in report["by_filing_type"]:
                report["by_filing_type"][ft] = 0
            report["by_filing_type"][ft] += 1

        # Count by exchange
        for filing in filings:
            ex = filing.exchange
            if ex not in report["by_exchange"]:
                report["by_exchange"][ex] = 0
            report["by_exchange"][ex] += 1

        # Collect alerts
        for filing in filings:
            analyzer = RegulatoryFilingsAnalyzer()
            result = analyzer.analyze_filing(filing)
            report["alerts"].extend([a.to_dict() for a in result.get("alerts", [])])

            if filing.impact_score >= 60:
                report["high_impact_filings"].append(filing.to_dict())

        return report


def get_regulatory_collector() -> RegulatoryFilingsCollector:
    """Get regulatory filings collector instance."""
    return RegulatoryFilingsCollector()


def get_regulatory_analyzer() -> RegulatoryFilingsAnalyzer:
    """Get regulatory filings analyzer instance."""
    return RegulatoryFilingsAnalyzer()
