# data/discovery.py
"""
Universal Stock Discovery System
Searches ALL available sources for stocks to train on
"""
from __future__ import annotations  # MUST be first import (after docstring)

import time
import threading
from datetime import datetime
from typing import List, Dict, Optional, Callable
from dataclasses import dataclass, field

import pandas as pd

from config import CONFIG
from utils.logger import log


@dataclass
class DiscoveredStock:
    """Stock discovered from any source"""
    code: str
    name: str = ""
    source: str = ""
    score: float = 0.5
    market_cap: float = 0.0
    volume: float = 0.0
    change_pct: float = 0.0
    sector: str = ""
    discovered_at: datetime = field(default_factory=datetime.now)

    def __post_init__(self):
        self.code = self._clean_code(self.code)

    def _clean_code(self, code: str) -> str:
        if not code:
            return ""
        code = str(code).strip()
        for prefix in ["sh", "sz", "SH", "SZ", "bj", "BJ"]:
            code = code.replace(prefix, "")
        code = code.replace(".", "").replace("-", "")
        if code.isdigit():
            return code.zfill(6)
        return ""

    def is_valid(self) -> bool:
        if not self.code or len(self.code) != 6 or not self.code.isdigit():
            return False
        valid_prefixes = ["60", "00", "30", "68", "83", "43"]
        return any(self.code.startswith(p) for p in valid_prefixes)

    @property
    def market(self) -> str:
        if self.code.startswith(("60", "68")):
            return "SH"
        if self.code.startswith(("00", "30")):
            return "SZ"
        if self.code.startswith(("83", "43")):
            return "BJ"
        return "UNKNOWN"


class UniversalStockDiscovery:
    """
    Discovers stocks from multiple sources (AkShare based when available).
    """

    def __init__(self):
        self._ak = None
        self._rate_limit = 1.0
        self._last_request = 0.0
        self._lock = threading.Lock()

        try:
            import akshare as ak  # type: ignore
            self._ak = ak
        except ImportError:
            log.warning("AkShare not available - discovery will return empty list")

    def _wait(self):
        with self._lock:
            elapsed = time.time() - self._last_request
            if elapsed < self._rate_limit:
                time.sleep(self._rate_limit - elapsed)
            self._last_request = time.time()

    def discover_all(self, callback=None, max_stocks=None, min_market_cap=0, include_st=False):
        """Discover ALL available stocks from all sources"""
        if not self._ak:
            return []

        all_stocks: Dict[str, DiscoveredStock] = {}

        # Fetch spot data ONCE
        spot_df = None
        try:
            self._wait()
            spot_df = self._ak.stock_zh_a_spot_em()
        except Exception as e:
            log.warning(f"Failed to fetch spot data: {e}")
            spot_df = None

        sources = [
            ("All A-Shares", lambda: self._from_spot_df(spot_df, "all")),
            ("Top Gainers", lambda: self._from_spot_df(spot_df, "gainers")),
            ("Top Losers", lambda: self._from_spot_df(spot_df, "losers")),
            ("High Volume", lambda: self._from_spot_df(spot_df, "volume")),
            ("Large Cap", lambda: self._from_spot_df(spot_df, "large_cap")),
            ("CSI 300 Index", self._get_csi300),
            ("CSI 500 Index", self._get_csi500),
            ("CSI 1000 Index", self._get_csi1000),
            ("Top Gainers (fresh)", self._get_top_gainers),
            ("Top Losers (fresh)", self._get_top_losers),
            ("High Volume (fresh)", self._get_high_volume),
            ("Large Cap (fresh)", self._get_large_cap),
            ("Growth Stocks", self._get_growth_stocks),
            ("Value Stocks", self._get_value_stocks),
        ]

        for source_name, source_fn in sources:
            if callback:
                callback(f"Searching {source_name}...", len(all_stocks))

            try:
                self._wait()
                stocks = source_fn() or []

                for stock in stocks:
                    if not stock.is_valid():
                        continue

                    # Filter ST stocks
                    if not include_st and stock.name and "ST" in stock.name.upper():
                        continue

                    if stock.code in all_stocks:
                        existing = all_stocks[stock.code]
                        existing.score = (existing.score + stock.score) / 2.0
                        existing.market_cap = max(existing.market_cap, stock.market_cap)
                        existing.volume = max(existing.volume, stock.volume)
                        if not existing.name and stock.name:
                            existing.name = stock.name
                    else:
                        all_stocks[stock.code] = stock

                log.info(f"Found {len(stocks)} from {source_name}, total: {len(all_stocks)}")

            except Exception as e:
                log.warning(f"Failed to search {source_name}: {e}")

        result = list(all_stocks.values())

        # Filter by market cap (note: AkShare '总市值' already in RMB, often units = RMB)
        if min_market_cap > 0:
            # user passes billions? your old code used * 1e9; keep same behavior:
            result = [s for s in result if s.market_cap >= min_market_cap * 1e9]

        # Score by multiple factors
        for stock in result:
            stock.score = self._calculate_score(stock)

        result.sort(key=lambda x: x.score, reverse=True)

        if max_stocks:
            result = result[:max_stocks]

        log.info(f"Total discovered: {len(result)} stocks")
        return result

    def _from_spot_df(self, df: Optional[pd.DataFrame], filter_type: str) -> List[DiscoveredStock]:
        """Extract stocks from cached spot dataframe"""
        if df is None or df.empty:
            return []

        work = df.copy()

        if filter_type == "all":
            pass
        elif filter_type == "gainers":
            work = work.sort_values("涨跌幅", ascending=False).head(100)
        elif filter_type == "losers":
            work = work.sort_values("涨跌幅", ascending=True).head(100)
        elif filter_type == "volume":
            work = work.sort_values("成交额", ascending=False).head(100)
        elif filter_type == "large_cap":
            work = work.sort_values("总市值", ascending=False).head(200)

        items: List[DiscoveredStock] = []
        for _, row in work.iterrows():
            items.append(
                DiscoveredStock(
                    code=str(row.get("代码", "")),
                    name=str(row.get("名称", "")),
                    source=filter_type,
                    market_cap=float(row.get("总市值", 0) or 0),
                    volume=float(row.get("成交额", 0) or 0),
                    change_pct=float(row.get("涨跌幅", 0) or 0),
                )
            )
        return items

    def _calculate_score(self, stock: DiscoveredStock) -> float:
        """Calculate comprehensive stock score for training priority"""
        score = 0.5

        # Market cap bonus
        if stock.market_cap > 100e8:
            score += 0.20
        elif stock.market_cap > 50e8:
            score += 0.15
        elif stock.market_cap > 10e8:
            score += 0.10

        # Volume bonus
        if stock.volume > 1e9:
            score += 0.15
        elif stock.volume > 5e8:
            score += 0.10
        elif stock.volume > 1e8:
            score += 0.05

        # Volatility bonus
        abs_change = abs(stock.change_pct)
        if 2 < abs_change < 8:
            score += 0.10
        elif abs_change > 8:
            score += 0.05

        # Index bonus
        if stock.source in ["CSI300", "CSI500", "CSI1000"]:
            score += 0.10

        return min(score, 1.0)

    # ---------------- Index / factor lists ----------------

    def _get_csi300(self) -> List[DiscoveredStock]:
        if not self._ak:
            return []
        try:
            df = self._ak.index_stock_cons_csindex(symbol="000300")
            return [
                DiscoveredStock(
                    code=str(row.get("成分券代码", "")),
                    name=str(row.get("成分券名称", "")),
                    source="CSI300",
                    score=0.8,
                )
                for _, row in df.iterrows()
            ]
        except Exception:
            return []

    def _get_csi500(self) -> List[DiscoveredStock]:
        if not self._ak:
            return []
        try:
            df = self._ak.index_stock_cons_csindex(symbol="000905")
            return [
                DiscoveredStock(
                    code=str(row.get("成分券代码", "")),
                    name=str(row.get("成分券名称", "")),
                    source="CSI500",
                    score=0.7,
                )
                for _, row in df.iterrows()
            ]
        except Exception:
            return []

    def _get_top_losers(self) -> List[DiscoveredStock]:
        return []

    def _get_csi1000(self) -> List[DiscoveredStock]:
        if not self._ak:
            return []
        try:
            df = self._ak.index_stock_cons_csindex(symbol="000852")
            return [
                DiscoveredStock(
                    code=str(row.get("成分券代码", "")),
                    name=str(row.get("成分券名称", "")),
                    source="CSI1000",
                    score=0.6,
                )
                for _, row in df.iterrows()
            ]
        except Exception:
            return []

    def _get_top_gainers(self) -> List[DiscoveredStock]:
        if not self._ak:
            return []
        try:
            df = self._ak.stock_zh_a_spot_em()
            df["涨跌幅"] = pd.to_numeric(df["涨跌幅"], errors="coerce")
            df = df.dropna(subset=["涨跌幅"]).sort_values("涨跌幅", ascending=False).head(100)
            return [
                DiscoveredStock(
                    code=str(row["代码"]),
                    name=str(row.get("名称", "")),
                    source="Gainers",
                    change_pct=float(row["涨跌幅"]),
                    score=min(abs(float(row["涨跌幅"])) / 10.0, 0.8),
                )
                for _, row in df.iterrows()
            ]
        except Exception:
            return []

    def _get_high_volume(self) -> List[DiscoveredStock]:
        if not self._ak:
            return []
        try:
            df = self._ak.stock_zh_a_spot_em()
            df["成交额"] = pd.to_numeric(df["成交额"], errors="coerce")
            df = df.dropna(subset=["成交额"]).sort_values("成交额", ascending=False).head(100)
            return [
                DiscoveredStock(
                    code=str(row["代码"]),
                    name=str(row.get("名称", "")),
                    source="HighVolume",
                    volume=float(row["成交额"]),
                    score=0.7,
                )
                for _, row in df.iterrows()
            ]
        except Exception:
            return []

    def _get_large_cap(self) -> List[DiscoveredStock]:
        if not self._ak:
            return []
        try:
            df = self._ak.stock_zh_a_spot_em()
            df["总市值"] = pd.to_numeric(df["总市值"], errors="coerce")
            df = df.dropna(subset=["总市值"]).sort_values("总市值", ascending=False).head(200)
            return [
                DiscoveredStock(
                    code=str(row["代码"]),
                    name=str(row.get("名称", "")),
                    source="LargeCap",
                    market_cap=float(row["总市值"]),
                    score=0.75,
                )
                for _, row in df.iterrows()
            ]
        except Exception:
            return []

    def _get_growth_stocks(self) -> List[DiscoveredStock]:
        if not self._ak:
            return []
        try:
            df = self._ak.stock_zh_a_spot_em()
            if "60日涨跌幅" not in df.columns:
                return []
            df["60日涨跌幅"] = pd.to_numeric(df["60日涨跌幅"], errors="coerce")
            df = df.dropna(subset=["60日涨跌幅"]).sort_values("60日涨跌幅", ascending=False).head(100)
            return [
                DiscoveredStock(
                    code=str(row["代码"]),
                    name=str(row.get("名称", "")),
                    source="Growth",
                    score=0.65,
                )
                for _, row in df.iterrows()
            ]
        except Exception:
            return []

    def _get_value_stocks(self) -> List[DiscoveredStock]:
        if not self._ak:
            return []
        try:
            df = self._ak.stock_zh_a_spot_em()
            col = "市盈率-动态" if "市盈率-动态" in df.columns else "市盈率"
            if col not in df.columns:
                return []
            df["市盈率"] = pd.to_numeric(df[col], errors="coerce")
            df = df[(df["市盈率"] > 0) & (df["市盈率"] < 20)].sort_values("市盈率").head(100)
            return [
                DiscoveredStock(
                    code=str(row["代码"]),
                    name=str(row.get("名称", "")),
                    source="Value",
                    score=0.6,
                )
                for _, row in df.iterrows()
            ]
        except Exception:
            return []