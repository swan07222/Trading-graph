# data/discovery.py
"""
Universal Stock Discovery System
Searches ALL available sources for stocks to train on
"""
from __future__ import annotations

import time
import threading
from datetime import datetime
from typing import List, Dict, Optional
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
        """Clean and normalize stock code"""
        if not code:
            return ""
        
        code = str(code).strip()
        
        # Remove common prefixes/suffixes
        for prefix in ["sh", "sz", "SH", "SZ", "bj", "BJ", "sh.", "sz.", "SH.", "SZ."]:
            if code.startswith(prefix):
                code = code[len(prefix):]
        
        for suffix in [".SH", ".SZ", ".BJ", ".ss", ".sz"]:
            if code.endswith(suffix):
                code = code[:-len(suffix)]
        
        # Remove any remaining dots, dashes, spaces
        code = code.replace(".", "").replace("-", "").replace(" ", "")
        
        # Extract only digits
        digits = ''.join(c for c in code if c.isdigit())
        
        if digits:
            return digits.zfill(6)
        return ""

    def is_valid(self) -> bool:
        """Check if stock code is valid"""
        if not self.code:
            return False
        
        if len(self.code) != 6:
            return False
        
        if not self.code.isdigit():
            return False
        
        valid_prefixes = [
            "60",  # Shanghai main board
            "00",  # Shenzhen main board  
            "30",  # ChiNext
            "68",  # STAR Market
            "83", "43", "87",  # Beijing
        ]
        
        return any(self.code.startswith(p) for p in valid_prefixes)

    @property
    def market(self) -> str:
        if self.code.startswith(("60", "68")):
            return "SH"
        if self.code.startswith(("00", "30")):
            return "SZ"
        if self.code.startswith(("83", "43", "87")):
            return "BJ"
        return "UNKNOWN"


class UniversalStockDiscovery:
    """
    Discovers stocks from multiple sources.
    """

    def __init__(self):
        self._ak = None
        self._rate_limit = 2.0
        self._last_request = 0.0
        self._lock = threading.Lock()
        self._request_timeout = 60

        try:
            import akshare as ak
            self._ak = ak
            log.info("AkShare initialized for stock discovery")
        except ImportError:
            log.warning("AkShare not available - using fallback stocks")

    def _wait(self):
        """Rate limiting between requests"""
        with self._lock:
            elapsed = time.time() - self._last_request
            if elapsed < self._rate_limit:
                time.sleep(self._rate_limit - elapsed)
            self._last_request = time.time()

    def _safe_fetch(self, fetch_func, description: str = "data"):
        """Safely fetch data with timeout and retry"""
        import socket
        
        max_retries = 3
        base_timeout = 60
        
        for attempt in range(max_retries):
            try:
                old_timeout = socket.getdefaulttimeout()
                socket.setdefaulttimeout(base_timeout * (attempt + 1))
                
                try:
                    self._wait()
                    result = fetch_func()
                    return result
                finally:
                    socket.setdefaulttimeout(old_timeout)
                    
            except Exception as e:
                log.warning(f"Attempt {attempt + 1}/{max_retries} failed for {description}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                continue
        
        return None

    def discover_all(self, callback=None, max_stocks=None, min_market_cap=0, include_st=False):
        """Discover ALL available stocks from all sources"""
        all_stocks: Dict[str, DiscoveredStock] = {}
        
        # If AkShare not available, return default stock pool
        if not self._ak:
            log.warning("AkShare not available, using fallback stocks")
            return self._get_fallback_stocks(max_stocks)

        if callback:
            callback("Fetching market data (this may take a moment)...", 0)
        
        # Try spot data first
        spot_df = self._safe_fetch(
            lambda: self._ak.stock_zh_a_spot_em(),
            "spot data"
        )
        
        spot_available = spot_df is not None and not spot_df.empty
        
        if not spot_available:
            log.warning("Failed to fetch spot data, trying index constituents...")

        sources = []
        
        if spot_available:
            sources = [
                ("All A-Shares", lambda: self._from_spot_df(spot_df, "all")),
                ("Top Gainers", lambda: self._from_spot_df(spot_df, "gainers")),
                ("Top Losers", lambda: self._from_spot_df(spot_df, "losers")),
                ("High Volume", lambda: self._from_spot_df(spot_df, "volume")),
                ("Large Cap", lambda: self._from_spot_df(spot_df, "large_cap")),
            ]
        else:
            sources = [
                ("CSI 300 Index", self._get_csi300),
                ("CSI 500 Index", self._get_csi500),
                ("CSI 1000 Index", self._get_csi1000),
            ]

        for source_name, source_fn in sources:
            if callback:
                callback(f"Searching {source_name}...", len(all_stocks))

            try:
                stocks = source_fn() or []
                valid_count = 0

                for stock in stocks:
                    if not stock.is_valid():
                        continue

                    if not include_st and stock.name and "ST" in stock.name.upper():
                        continue

                    valid_count += 1
                    
                    if stock.code in all_stocks:
                        existing = all_stocks[stock.code]
                        existing.score = (existing.score + stock.score) / 2.0
                        existing.market_cap = max(existing.market_cap, stock.market_cap)
                        existing.volume = max(existing.volume, stock.volume)
                        if not existing.name and stock.name:
                            existing.name = stock.name
                    else:
                        all_stocks[stock.code] = stock

                log.info(f"Found {valid_count} valid from {source_name}, total: {len(all_stocks)}")

            except Exception as e:
                log.warning(f"Failed to search {source_name}: {e}")

        # Convert to list
        result = list(all_stocks.values())
        
        log.info(f"Before filtering: {len(result)} stocks")

        # Filter by market cap ONLY if we have market cap data
        if min_market_cap > 0:
            # Only filter stocks that have market cap data
            filtered = [s for s in result if s.market_cap <= 0 or s.market_cap >= min_market_cap * 1e9]
            if len(filtered) > 0:
                result = filtered
            # If all would be filtered, keep original

        # Calculate scores
        for stock in result:
            stock.score = self._calculate_score(stock)

        # Sort by score
        result.sort(key=lambda x: x.score, reverse=True)

        # Apply max_stocks limit
        if max_stocks and max_stocks > 0:
            result = result[:max_stocks]

        log.info(f"After filtering: {len(result)} stocks")

        # If still no stocks, use fallback
        if not result:
            log.warning("No stocks after filtering, using fallback")
            return self._get_fallback_stocks(max_stocks)

        return result

    def _get_fallback_stocks(self, max_stocks: Optional[int] = None) -> List[DiscoveredStock]:
        """Return fallback stock list when discovery fails"""
        fallback_codes = [
            ("600519", "贵州茅台"), ("601318", "中国平安"), ("600036", "招商银行"),
            ("000858", "五粮液"), ("600900", "长江电力"), ("000333", "美的集团"),
            ("000651", "格力电器"), ("002594", "比亚迪"), ("300750", "宁德时代"),
            ("002475", "立讯精密"), ("600887", "伊利股份"), ("603288", "海天味业"),
            ("600276", "恒瑞医药"), ("300760", "迈瑞医疗"), ("300015", "爱尔眼科"),
            ("601166", "兴业银行"), ("601398", "工商银行"), ("600030", "中信证券"),
            ("002230", "科大讯飞"), ("300059", "东方财富"), ("601857", "中国石油"),
            ("600028", "中国石化"), ("601088", "中国神华"), ("600309", "万华化学"),
            ("601012", "隆基绿能"), ("000568", "泸州老窖"), ("600000", "浦发银行"),
            ("601328", "交通银行"), ("000002", "万科A"), ("002714", "牧原股份"),
            ("600690", "海尔智家"), ("000725", "京东方A"), ("601899", "紫金矿业"),
            ("600585", "海螺水泥"), ("002352", "顺丰控股"), ("300124", "汇川技术"),
            ("002415", "海康威视"), ("600031", "三一重工"), ("000001", "平安银行"),
            ("002304", "洋河股份"),
        ]
        
        stocks = [
            DiscoveredStock(code=code, name=name, source="fallback", score=0.7)
            for code, name in fallback_codes
        ]
        
        if max_stocks and max_stocks > 0:
            stocks = stocks[:max_stocks]
        
        log.info(f"Using {len(stocks)} fallback stocks")
        return stocks

    def _from_spot_df(self, df: Optional[pd.DataFrame], filter_type: str) -> List[DiscoveredStock]:
        """Extract stocks from cached spot dataframe"""
        if df is None or df.empty:
            return []

        work = df.copy()

        if filter_type == "gainers":
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

        if stock.market_cap > 100e8:
            score += 0.20
        elif stock.market_cap > 50e8:
            score += 0.15
        elif stock.market_cap > 10e8:
            score += 0.10

        if stock.volume > 1e9:
            score += 0.15
        elif stock.volume > 5e8:
            score += 0.10
        elif stock.volume > 1e8:
            score += 0.05

        abs_change = abs(stock.change_pct)
        if 2 < abs_change < 8:
            score += 0.10
        elif abs_change > 8:
            score += 0.05

        if stock.source in ["CSI300", "CSI500", "CSI1000"]:
            score += 0.10

        return min(score, 1.0)

    def _get_csi300(self) -> List[DiscoveredStock]:
        """Get CSI 300 constituents"""
        if not self._ak:
            return []
        try:
            self._wait()
            df = self._ak.index_stock_cons_csindex(symbol="000300")
            return self._parse_index_df(df, "CSI300", 0.8)
        except Exception as e:
            log.warning(f"Failed to get CSI300: {e}")
            return []

    def _get_csi500(self) -> List[DiscoveredStock]:
        """Get CSI 500 constituents"""
        if not self._ak:
            return []
        try:
            self._wait()
            df = self._ak.index_stock_cons_csindex(symbol="000905")
            return self._parse_index_df(df, "CSI500", 0.7)
        except Exception as e:
            log.warning(f"Failed to get CSI500: {e}")
            return []

    def _get_csi1000(self) -> List[DiscoveredStock]:
        """Get CSI 1000 constituents"""
        if not self._ak:
            return []
        try:
            self._wait()
            df = self._ak.index_stock_cons_csindex(symbol="000852")
            return self._parse_index_df(df, "CSI1000", 0.6)
        except Exception as e:
            log.warning(f"Failed to get CSI1000: {e}")
            return []

    def _parse_index_df(self, df: pd.DataFrame, source: str, base_score: float) -> List[DiscoveredStock]:
        """Parse index constituent dataframe"""
        if df is None or df.empty:
            return []
        
        # Find code column
        code_col = None
        for col in ["成分券代码", "证券代码", "代码", "stock_code", "code", "symbol"]:
            if col in df.columns:
                code_col = col
                break
        
        # Find name column
        name_col = None
        for col in ["成分券名称", "证券简称", "名称", "stock_name", "name"]:
            if col in df.columns:
                name_col = col
                break
        
        if code_col is None:
            # Try first column as code
            code_col = df.columns[0]
            log.debug(f"Using first column '{code_col}' as code for {source}")
        
        stocks = []
        for _, row in df.iterrows():
            code = str(row.get(code_col, ""))
            name = str(row.get(name_col, "")) if name_col else ""
            
            stock = DiscoveredStock(
                code=code,
                name=name,
                source=source,
                score=base_score,
            )
            
            if stock.is_valid():
                stocks.append(stock)
        
        return stocks

    def _get_top_gainers(self) -> List[DiscoveredStock]:
        if not self._ak:
            return []
        try:
            self._wait()
            df = self._ak.stock_zh_a_spot_em()
            df["涨跌幅"] = pd.to_numeric(df["涨跌幅"], errors="coerce")
            df = df.dropna(subset=["涨跌幅"]).sort_values("涨跌幅", ascending=False).head(100)
            return [
                DiscoveredStock(
                    code=str(row["代码"]),
                    name=str(row.get("名称", "")),
                    source="Gainers",
                    change_pct=float(row["涨跌幅"]),
                    score=0.7,
                )
                for _, row in df.iterrows()
            ]
        except Exception:
            return []

    def _get_top_losers(self) -> List[DiscoveredStock]:
        if not self._ak:
            return []
        try:
            self._wait()
            df = self._ak.stock_zh_a_spot_em()
            df["涨跌幅"] = pd.to_numeric(df["涨跌幅"], errors="coerce")
            df = df.dropna(subset=["涨跌幅"]).sort_values("涨跌幅", ascending=True).head(100)
            return [
                DiscoveredStock(
                    code=str(row["代码"]),
                    name=str(row.get("名称", "")),
                    source="Losers",
                    change_pct=float(row["涨跌幅"]),
                    score=0.7,
                )
                for _, row in df.iterrows()
            ]
        except Exception:
            return []

    def _get_high_volume(self) -> List[DiscoveredStock]:
        if not self._ak:
            return []
        try:
            self._wait()
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
            self._wait()
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
            self._wait()
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
            self._wait()
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