# data/discovery.py
"""
Universal Stock Discovery System
Network-aware: uses AkShare on China IP, fallback+Tencent otherwise.
"""
from __future__ import annotations

import time
import threading
from datetime import datetime
from typing import List, Dict, Optional
from dataclasses import dataclass, field

import pandas as pd

from config.settings import CONFIG
from utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class DiscoveredStock:
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
        for prefix in ["sh", "sz", "SH", "SZ", "bj", "BJ", "sh.", "sz.", "SH.", "SZ."]:
            if code.startswith(prefix):
                code = code[len(prefix):]
        for suffix in [".SH", ".SZ", ".BJ", ".ss", ".sz"]:
            if code.endswith(suffix):
                code = code[:-len(suffix)]
        code = code.replace(".", "").replace("-", "").replace(" ", "")
        digits = ''.join(c for c in code if c.isdigit())
        return digits.zfill(6) if digits else ""

    def is_valid(self) -> bool:
        if not self.code or len(self.code) != 6 or not self.code.isdigit():
            return False
        valid_prefixes = ["60", "00", "30", "68", "83", "43", "87"]
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

    def __init__(self):
        self._ak = None
        self._rate_limit = 1.0
        self._last_request = 0.0
        self._lock = threading.Lock()
        self._request_timeout = 10

        try:
            import akshare as ak
            self._ak = ak
            log.info("AkShare initialized for stock discovery")
        except ImportError:
            log.warning("AkShare not available - using fallback stocks")

    def _wait(self):
        with self._lock:
            elapsed = time.time() - self._last_request
            if elapsed < self._rate_limit:
                time.sleep(self._rate_limit - elapsed)
            self._last_request = time.time()

    def _safe_fetch(self, fetch_func, description: str = "data"):
        import socket
        max_retries = 2
        base_timeout = 10
        for attempt in range(max_retries):
            try:
                old_timeout = socket.getdefaulttimeout()
                socket.setdefaulttimeout(base_timeout * (attempt + 1))
                try:
                    self._wait()
                    return fetch_func()
                finally:
                    socket.setdefaulttimeout(old_timeout)
            except Exception as e:
                log.warning(f"Attempt {attempt+1}/{max_retries} failed for {description}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
        return None

    def discover_all(self, callback=None, max_stocks=None, min_market_cap=0, include_st=False):
        all_stocks: Dict[str, DiscoveredStock] = {}

        # Check network environment
        from core.network import get_network_env
        env = get_network_env()

        if callback:
            net_mode = "China direct" if env.is_china_direct else "VPN (foreign IP)"
            callback(f"Network: {net_mode}. Discovering stocks...", 0)

        # Strategy based on network
        if env.is_china_direct and env.eastmoney_ok and self._ak:
            # China direct: use AkShare (eastmoney)
            log.info("Discovery via AkShare (China direct IP)")
            stocks = self._discover_via_akshare(callback, max_stocks, min_market_cap, include_st)
            if stocks:
                return stocks

        # Tencent works from any IP
        if env.tencent_ok:
            log.info("Discovery via Tencent (works from any IP)")
            stocks = self._discover_via_tencent(max_stocks)
            if stocks:
                for s in stocks:
                    if s.is_valid() and (include_st or "ST" not in (s.name or "").upper()):
                        all_stocks[s.code] = s

        # Index constituents (works sometimes even through VPN)
        if env.is_china_direct and self._ak:
            for source_name, source_fn in [
                ("CSI 300 Index", self._get_csi300),
                ("CSI 500 Index", self._get_csi500),
                ("CSI 1000 Index", self._get_csi1000),
            ]:
                if callback:
                    callback(f"Searching {source_name}...", len(all_stocks))
                try:
                    for stock in (source_fn() or []):
                        if stock.is_valid() and stock.code not in all_stocks:
                            if include_st or "ST" not in (stock.name or "").upper():
                                all_stocks[stock.code] = stock
                except Exception as e:
                    log.warning(f"Failed {source_name}: {e}")

        result = list(all_stocks.values())
        log.info(f"Before filtering: {len(result)} stocks")

        if min_market_cap > 0:
            filtered = [s for s in result if s.market_cap <= 0 or s.market_cap >= min_market_cap * 1e9]
            if filtered:
                result = filtered

        for stock in result:
            stock.score = self._calculate_score(stock)
        result.sort(key=lambda x: x.score, reverse=True)

        if max_stocks and max_stocks > 0:
            result = result[:max_stocks]

        log.info(f"After filtering: {len(result)} stocks")

        if not result:
            log.warning("No stocks after filtering, using fallback")
            return self._get_fallback_stocks(max_stocks)

        return result

    def _discover_via_akshare(self, callback, max_stocks, min_market_cap, include_st):
        """Full AkShare discovery (China direct only)."""
        all_stocks: Dict[str, DiscoveredStock] = {}

        if callback:
            callback("Fetching market data via AkShare...", 0)

        spot_df = self._safe_fetch(lambda: self._ak.stock_zh_a_spot_em(), "spot data")
        spot_available = spot_df is not None and not spot_df.empty

        if not spot_available:
            log.warning("Spot data unavailable, trying index constituents")

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
                for stock in (source_fn() or []):
                    if not stock.is_valid():
                        continue
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
                log.info(f"Found from {source_name}, total: {len(all_stocks)}")
            except Exception as e:
                log.warning(f"Failed {source_name}: {e}")

        result = list(all_stocks.values())
        if not result:
            return None

        if min_market_cap > 0:
            filtered = [s for s in result if s.market_cap <= 0 or s.market_cap >= min_market_cap * 1e9]
            if filtered:
                result = filtered

        for s in result:
            s.score = self._calculate_score(s)
        result.sort(key=lambda x: x.score, reverse=True)

        if max_stocks and max_stocks > 0:
            result = result[:max_stocks]

        return result if result else None

    def _discover_via_tencent(self, max_stocks: Optional[int] = None) -> List[DiscoveredStock]:
        """Verify fallback stocks via Tencent (works from any IP)."""
        import requests
        from core.constants import get_exchange

        fallback = self._get_fallback_stocks(max_stocks=200)
        codes = [s.code for s in fallback]

        vendor_symbols = []
        code_map = {}
        for c in codes:
            ex = get_exchange(c)
            if ex == "SSE":
                sym = f"sh{c}"
            elif ex == "SZSE":
                sym = f"sz{c}"
            else:
                continue
            vendor_symbols.append(sym)
            code_map[sym] = c

        verified: List[DiscoveredStock] = []
        CHUNK = 80
        for i in range(0, len(vendor_symbols), CHUNK):
            chunk = vendor_symbols[i:i + CHUNK]
            try:
                url = "https://qt.gtimg.cn/q=" + ",".join(chunk)
                r = requests.get(url, timeout=5)
                for line in r.text.splitlines():
                    if "~" not in line or "=" not in line:
                        continue
                    try:
                        left, right = line.split("=", 1)
                        sym = left.strip().replace("v_", "")
                        parts = right.strip().strip('";').split("~")
                        if len(parts) < 10:
                            continue
                        code6 = code_map.get(sym)
                        if not code6:
                            continue
                        name = parts[1]
                        price = float(parts[3] or 0)
                        if price <= 0:
                            continue
                        verified.append(DiscoveredStock(
                            code=code6, name=name, source="tencent", score=0.7
                        ))
                    except Exception:
                        continue
            except Exception as e:
                log.debug(f"Tencent discovery chunk failed: {e}")

        log.info(f"Tencent discovery verified {len(verified)} stocks")
        if max_stocks and max_stocks > 0:
            verified = verified[:max_stocks]
        return verified

    def _get_fallback_stocks(self, max_stocks: Optional[int] = None) -> List[DiscoveredStock]:
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
            ("002304", "洋河股份"), ("601688", "华泰证券"), ("600104", "上汽集团"),
            ("601888", "中国中免"), ("600809", "山西汾酒"), ("002371", "北方华创"),
            ("688041", "海光信息"), ("688256", "寒武纪"), ("300896", "爱美客"),
            ("688012", "中微公司"), ("002049", "紫光国微"), ("600050", "中国联通"),
            ("601728", "中国电信"), ("600941", "中国移动"), ("601669", "中国电建"),
            ("601668", "中国建筑"), ("601390", "中国中铁"), ("000063", "中兴通讯"),
            ("002460", "赣锋锂业"), ("300274", "阳光电源"), ("601816", "京沪高铁"),
            ("600438", "通威股份"), ("002466", "天齐锂业"), ("601225", "陕西煤业"),
            ("600048", "保利发展"), ("601633", "长城汽车"), ("002812", "恩捷股份"),
            ("300033", "同花顺"), ("601919", "中远海控"), ("603259", "药明康德"),
            ("600346", "恒力石化"), ("002241", "歌尔股份"), ("688981", "中芯国际"),
            ("300347", "泰格医药"), ("600763", "通策医疗"), ("601100", "恒立液压"),
            ("300782", "卓胜微"), ("603501", "韦尔股份"), ("300661", "圣邦股份"),
            ("688036", "传音控股"), ("002709", "天赐材料"), ("300014", "亿纬锂能"),
            ("600745", "闻泰科技"), ("601865", "福莱特"), ("300316", "晶盛机电"),
            ("688111", "金山办公"), ("300999", "金龙鱼"), ("603986", "兆易创新"),
            ("688561", "奇安信"), ("300308", "中际旭创"), ("002916", "深南电路"),
            ("300413", "芒果超媒"), ("601138", "工业富联"), ("600406", "国电南瑞"),
            ("601615", "明阳智能"), ("002382", "蓝思科技"), ("300122", "智飞生物"),
            ("600196", "复星医药"),
        ]

        seen = set()
        stocks = []
        for code, name in fallback_codes:
            s = DiscoveredStock(code=code, name=name, source="fallback", score=0.7)
            if s.is_valid() and s.code not in seen:
                seen.add(s.code)
                stocks.append(s)

        if max_stocks and max_stocks > 0:
            stocks = stocks[:max_stocks]
        log.info(f"Using {len(stocks)} fallback stocks")
        return stocks

    def _from_spot_df(self, df: Optional[pd.DataFrame], filter_type: str) -> List[DiscoveredStock]:
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

        items = []
        for _, row in work.iterrows():
            items.append(DiscoveredStock(
                code=str(row.get("代码", "")), name=str(row.get("名称", "")),
                source=filter_type, market_cap=float(row.get("总市值", 0) or 0),
                volume=float(row.get("成交额", 0) or 0),
                change_pct=float(row.get("涨跌幅", 0) or 0),
            ))
        return items

    def _calculate_score(self, stock: DiscoveredStock) -> float:
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
        if df is None or df.empty:
            return []
        code_col = None
        for col in ["成分券代码", "证券代码", "代码", "stock_code", "code", "symbol"]:
            if col in df.columns:
                code_col = col
                break
        name_col = None
        for col in ["成分券名称", "证券简称", "名称", "stock_name", "name"]:
            if col in df.columns:
                name_col = col
                break
        if code_col is None:
            code_col = df.columns[0]

        stocks = []
        for _, row in df.iterrows():
            stock = DiscoveredStock(
                code=str(row.get(code_col, "")),
                name=str(row.get(name_col, "")) if name_col else "",
                source=source, score=base_score,
            )
            if stock.is_valid():
                stocks.append(stock)
        return stocks