# data/discovery.py
from __future__ import annotations

import re
import socket
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from types import SimpleNamespace

import pandas as pd

try:
    import requests
except ImportError:
    requests = None  # type: ignore

from utils.logger import get_logger

log = get_logger(__name__)

# SCORING CONSTANTS (no more magic numbers)

# Market cap thresholds (in raw yuan)
_MCAP_LARGE: float = 100e8       # 100亿 (10B CNY)
_MCAP_MEDIUM: float = 50e8       # 50亿
_MCAP_SMALL: float = 10e8        # 10亿

_MCAP_LARGE_SCORE: float = 0.20
_MCAP_MEDIUM_SCORE: float = 0.15
_MCAP_SMALL_SCORE: float = 0.10

# Volume thresholds (in raw yuan)
_VOL_HIGH: float = 1e9           # 10亿
_VOL_MEDIUM: float = 5e8         # 5亿
_VOL_LOW: float = 1e8            # 1亿

_VOL_HIGH_SCORE: float = 0.15
_VOL_MEDIUM_SCORE: float = 0.10
_VOL_LOW_SCORE: float = 0.05

# Change-pct scoring
_CHANGE_ACTIVE_MIN: float = 2.0
_CHANGE_ACTIVE_MAX: float = 8.0
_CHANGE_ACTIVE_SCORE: float = 0.10
_CHANGE_EXTREME_SCORE: float = 0.05

_INDEX_SOURCES = frozenset({"CSI300", "CSI500", "CSI1000"})
_INDEX_SCORE: float = 0.10

_BASE_SCORE: float = 0.50

# Valid A-share prefixes (aligned with core/constants.py EXCHANGES)
_VALID_PREFIXES = (
    "600", "601", "603", "605",   # SSE main
    "688",                         # SSE STAR
    "000", "001", "002", "003",   # SZSE main
    "300", "301",                  # SZSE ChiNext
    "83", "87", "43",              # BSE
)

# Prefix → exchange mapping (aligned with core/constants.py naming)
_PREFIX_TO_EXCHANGE: dict[str, str] = {}
_SSE_PREFIXES = ("600", "601", "603", "605", "688")
_SZSE_PREFIXES = ("000", "001", "002", "003", "300", "301")
_BSE_PREFIXES = ("83", "87", "43")

_PREFIX_RE = re.compile(r'^(sh|sz|bj)\.?', re.IGNORECASE)
_SUFFIX_RE = re.compile(r'\.(sh|sz|bj|ss)$', re.IGNORECASE)

# Rate limit & network
_DEFAULT_RATE_LIMIT: float = 1.0
_DEFAULT_TIMEOUT: int = 10
_MAX_RETRIES: int = 2
_TENCENT_CHUNK_SIZE: int = 80

@dataclass
class DiscoveredStock:
    """A discovered stock with metadata."""

    code: str
    name: str = ""
    source: str = ""
    score: float = _BASE_SCORE
    market_cap: float = 0.0
    volume: float = 0.0
    change_pct: float = 0.0
    sector: str = ""
    discovered_at: datetime = field(default_factory=datetime.now)

    def __post_init__(self) -> None:
        self.code = self._clean_code(self.code)

    # ------------------------------------------------------------------ #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _clean_code(code: str) -> str:
        if not code:
            return ""
        code = str(code).strip()
        code = _PREFIX_RE.sub("", code)
        code = _SUFFIX_RE.sub("", code)
        code = code.replace(".", "").replace("-", "").replace(" ", "")
        digits = "".join(c for c in code if c.isdigit())
        return digits.zfill(6) if digits else ""

    # ------------------------------------------------------------------ #
    # Validation (uses exact prefixes from core/constants.py)
    # ------------------------------------------------------------------ #
    def is_valid(self) -> bool:
        if not self.code or len(self.code) != 6 or not self.code.isdigit():
            return False
        return any(self.code.startswith(p) for p in _VALID_PREFIXES)

    # ------------------------------------------------------------------ #
    # Exchange (naming aligned with core/constants.py: SSE / SZSE / BSE)
    # ------------------------------------------------------------------ #
    @property
    def market(self) -> str:
        """Return exchange string consistent with ``core.constants.get_exchange``."""
        if any(self.code.startswith(p) for p in _SSE_PREFIXES):
            return "SSE"
        if any(self.code.startswith(p) for p in _SZSE_PREFIXES):
            return "SZSE"
        if any(self.code.startswith(p) for p in _BSE_PREFIXES):
            return "BSE"
        return "UNKNOWN"

class UniversalStockDiscovery:
    """
    Discovers A-share stocks using the best available data source
    given the current network environment.

    Priority:
        1. AkShare full spot data   (China direct IP only)
        2. AkShare index constituents (China direct IP only)
        3. Tencent quote verification (works from any IP)
        4. Hardcoded fallback list    (always works)
    """

    def __init__(self) -> None:
        self._ak = None  # type: ignore
        self._rate_limit: float = _DEFAULT_RATE_LIMIT
        self._last_request: float = 0.0
        self._lock = threading.Lock()
        self._timeout: int = _DEFAULT_TIMEOUT

        # Network environment (cached once per instance)
        self._net_env: SimpleNamespace | object | None = None  # lazy

        try:
            import akshare as ak  # type: ignore
            self._ak = ak
            log.info("AkShare initialised for stock discovery")
        except ImportError:
            log.warning("AkShare not available — using fallback stocks")

    # ================================================================== #
    # Network environment (cached)
    # ================================================================== #
    def _get_net_env(self):
        """Return cached network environment probe."""
        if self._net_env is None:
            try:
                from core.network import get_network_env
                self._net_env = get_network_env()
            except Exception as exc:
                log.warning(f"Network probe failed, assuming foreign IP: {exc}")

                @dataclass
                class _FallbackEnv:
                    is_china_direct: bool = False
                    eastmoney_ok: bool = False
                    tencent_ok: bool = True

                self._net_env = _FallbackEnv()
        return self._net_env

    # ================================================================== #
    # ================================================================== #
    def _wait(self) -> None:
        with self._lock:
            elapsed = time.time() - self._last_request
            if elapsed < self._rate_limit:
                time.sleep(self._rate_limit - elapsed)
            self._last_request = time.time()

    # ================================================================== #
    # Safe fetch with retry + timeout
    # ================================================================== #
    def _safe_fetch(self, fetch_func, description: str = "data"):
        for attempt in range(_MAX_RETRIES):
            try:
                old_timeout = socket.getdefaulttimeout()
                socket.setdefaulttimeout(self._timeout * (attempt + 1))
                try:
                    self._wait()
                    return fetch_func()
                finally:
                    socket.setdefaulttimeout(old_timeout)
            except Exception as exc:
                log.warning(
                    f"Attempt {attempt + 1}/{_MAX_RETRIES} failed for "
                    f"{description}: {exc}"
                )
                if attempt < _MAX_RETRIES - 1:
                    time.sleep(2 ** attempt)
        return None

    # ================================================================== #
    # ================================================================== #
    def discover_all(
        self,
        callback: Callable[[str, int], None] | None = None,
        max_stocks: int | None = None,
        min_market_cap: float = 0,
        include_st: bool = False,
    ) -> list[DiscoveredStock]:
        """
        Discover stocks from the best available source.

        Args:
            callback:        ``(message, count_so_far) -> None`` progress hook.
            max_stocks:      Cap on returned list length.
            min_market_cap:  Minimum market cap **in billions of CNY** (e.g. 10 = 10B).
            include_st:      Whether to include ST / \\*ST stocks.

        Returns:
            Sorted list of ``DiscoveredStock`` (highest score first).
        """
        env = self._get_net_env()

        if callback:
            mode = "China direct" if env.is_china_direct else "VPN / foreign IP"
            callback(f"Network: {mode}. Discovering stocks…", 0)

        all_stocks: dict[str, DiscoveredStock] = {}

        # -------------------------------------------------------------- #
        # 1. AkShare full spot (China direct only)
        # -------------------------------------------------------------- #
        if env.is_china_direct and env.eastmoney_ok and self._ak:
            log.info("Discovery via AkShare (China direct IP)")
            ak_stocks = self._discover_via_akshare(callback, include_st)
            if ak_stocks:
                for s in ak_stocks:
                    all_stocks[s.code] = s

        # -------------------------------------------------------------- #
        # 2. Tencent verification (any IP)
        # -------------------------------------------------------------- #
        if not all_stocks and env.tencent_ok:
            log.info("Discovery via Tencent (works from any IP)")
            for s in self._discover_via_tencent():
                if s.is_valid() and (include_st or not _is_st(s.name)):
                    all_stocks.setdefault(s.code, s)

        # -------------------------------------------------------------- #
        # 3. Index constituents (China direct, AkShare)
        # -------------------------------------------------------------- #
        if env.is_china_direct and self._ak:
            for label, fn in (
                ("CSI 300", self._get_csi300),
                ("CSI 500", self._get_csi500),
                ("CSI 1000", self._get_csi1000),
            ):
                if callback:
                    callback(f"Searching {label}…", len(all_stocks))
                try:
                    for s in fn() or []:
                        if s.is_valid() and s.code not in all_stocks:
                            if include_st or not _is_st(s.name):
                                all_stocks[s.code] = s
                except Exception as exc:
                    log.warning(f"Failed {label}: {exc}")

        # -------------------------------------------------------------- #
        # 4. Fallback
        # -------------------------------------------------------------- #
        if not all_stocks:
            log.warning("No stocks discovered — using fallback list")
            return self._finalize(
                self._get_fallback_stocks(), max_stocks, min_market_cap
            )

        return self._finalize(
            list(all_stocks.values()), max_stocks, min_market_cap
        )

    # ================================================================== #
    # SINGLE filter / score / sort / truncate pipeline
    # ================================================================== #
    def _finalize(
        self,
        stocks: list[DiscoveredStock],
        max_stocks: int | None,
        min_market_cap: float,
    ) -> list[DiscoveredStock]:
        """
        Centralised post-processing applied exactly once to every
        discovery path. Eliminates the old duplicated logic.

        Args:
            stocks:          Raw discovered stocks.
            max_stocks:      Cap on output list.
            min_market_cap:  In **billions of CNY**.
        """
        log.info(f"Before filtering: {len(stocks)} stocks")

        # Market-cap filter (convert billions → raw yuan)
        if min_market_cap > 0:
            threshold = min_market_cap * 1e9
            filtered = [
                s for s in stocks
                if s.market_cap <= 0 or s.market_cap >= threshold
            ]
            if filtered:
                stocks = filtered

        # Score (pure arithmetic — no network calls)
        for s in stocks:
            s.score = self._calculate_score(s)

        stocks.sort(key=lambda x: x.score, reverse=True)

        if max_stocks and max_stocks > 0:
            stocks = stocks[:max_stocks]

        log.info(f"After filtering: {len(stocks)} stocks")
        return stocks

    # ================================================================== #
    # Scoring (pure, no I/O)
    # ================================================================== #
    @staticmethod
    def _calculate_score(stock: DiscoveredStock) -> float:
        """
        Deterministic score in [0, 1]. No network calls.

        Components
        ----------
        - Market capitalisation tier       (up to 0.20)
        - Trading volume tier              (up to 0.15)
        - Intraday change activity         (up to 0.10)
        - Index constituent bonus          (0.10)
        """
        score = _BASE_SCORE

        if stock.market_cap > _MCAP_LARGE:
            score += _MCAP_LARGE_SCORE
        elif stock.market_cap > _MCAP_MEDIUM:
            score += _MCAP_MEDIUM_SCORE
        elif stock.market_cap > _MCAP_SMALL:
            score += _MCAP_SMALL_SCORE

        if stock.volume > _VOL_HIGH:
            score += _VOL_HIGH_SCORE
        elif stock.volume > _VOL_MEDIUM:
            score += _VOL_MEDIUM_SCORE
        elif stock.volume > _VOL_LOW:
            score += _VOL_LOW_SCORE

        abs_change = abs(stock.change_pct)
        if _CHANGE_ACTIVE_MIN < abs_change < _CHANGE_ACTIVE_MAX:
            score += _CHANGE_ACTIVE_SCORE
        elif abs_change >= _CHANGE_ACTIVE_MAX:
            score += _CHANGE_EXTREME_SCORE

        if stock.source in _INDEX_SOURCES:
            score += _INDEX_SCORE

        return min(score, 1.0)

    # ================================================================== #
    # AkShare discovery (China direct)
    # ================================================================== #
    def _discover_via_akshare(
        self,
        callback: Callable | None = None,
        include_st: bool = False,
    ) -> list[DiscoveredStock] | None:
        """Full AkShare discovery — China direct IP only."""
        if not self._ak:
            return None

        if callback:
            callback("Fetching market data via AkShare…", 0)

        spot_df = self._safe_fetch(
            lambda: self._ak.stock_zh_a_spot_em(), "spot data"
        )
        spot_ok = spot_df is not None and not spot_df.empty

        if spot_ok:
            sources = [
                ("All A-Shares", lambda: self._from_spot_df(spot_df, "all")),
                ("Top Gainers", lambda: self._from_spot_df(spot_df, "gainers")),
                ("Top Losers", lambda: self._from_spot_df(spot_df, "losers")),
                ("High Volume", lambda: self._from_spot_df(spot_df, "volume")),
                ("Large Cap", lambda: self._from_spot_df(spot_df, "large_cap")),
            ]
        else:
            log.warning("Spot data unavailable, trying index constituents")
            sources = [
                ("CSI 300", self._get_csi300),
                ("CSI 500", self._get_csi500),
                ("CSI 1000", self._get_csi1000),
            ]

        all_stocks: dict[str, DiscoveredStock] = {}
        for label, fn in sources:
            if callback:
                callback(f"Searching {label}…", len(all_stocks))
            try:
                for s in fn() or []:
                    if not s.is_valid():
                        continue
                    if not include_st and _is_st(s.name):
                        continue
                    if s.code in all_stocks:
                        self._merge_stock(all_stocks[s.code], s)
                    else:
                        all_stocks[s.code] = s
                log.info(f"After {label}: {len(all_stocks)} stocks")
            except Exception as exc:
                log.warning(f"Failed {label}: {exc}")

        return list(all_stocks.values()) if all_stocks else None

    @staticmethod
    def _merge_stock(existing: DiscoveredStock, new: DiscoveredStock) -> None:
        """Merge metadata from *new* into *existing*."""
        existing.score = (existing.score + new.score) / 2.0
        existing.market_cap = max(existing.market_cap, new.market_cap)
        existing.volume = max(existing.volume, new.volume)
        if not existing.name and new.name:
            existing.name = new.name

    # ================================================================== #
    # ================================================================== #
    def _discover_via_tencent(self) -> list[DiscoveredStock]:
        """
        Verify fallback stocks via Tencent HTTP quotes.
        Works from any IP.  Returns only stocks with a valid live price.
        """
        if requests is None:
            log.warning("requests library not available for Tencent discovery")
            return []

        from core.constants import get_exchange

        candidates = self._get_fallback_stocks()
        vendor_symbols: list[str] = []
        code_map: dict[str, str] = {}

        for s in candidates:
            ex = get_exchange(s.code)
            if ex == "SSE":
                sym = f"sh{s.code}"
            elif ex == "SZSE":
                sym = f"sz{s.code}"
            else:
                continue
            vendor_symbols.append(sym)
            code_map[sym] = s.code

        verified: list[DiscoveredStock] = []
        for i in range(0, len(vendor_symbols), _TENCENT_CHUNK_SIZE):
            chunk = vendor_symbols[i : i + _TENCENT_CHUNK_SIZE]
            try:
                url = "https://qt.gtimg.cn/q=" + ",".join(chunk)
                resp = requests.get(url, timeout=self._timeout)
                # Tencent may return one long ';'-delimited line or multi-line text.
                chunks = []
                for line in resp.text.splitlines():
                    if not line:
                        continue
                    chunks.extend([seg for seg in line.split(";") if seg])

                for line in chunks:
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
                        verified.append(
                            DiscoveredStock(
                                code=code6,
                                name=name,
                                source="tencent",
                                score=0.7,
                            )
                        )
                    except Exception:
                        continue
            except Exception as exc:
                log.debug(f"Tencent chunk failed: {exc}")

        log.info(f"Tencent verified {len(verified)} stocks")
        return verified

    # ================================================================== #
    # Index constituents (AkShare)
    # ================================================================== #
    def _get_csi300(self) -> list[DiscoveredStock]:
        return self._fetch_index("000300", "CSI300", 0.8)

    def _get_csi500(self) -> list[DiscoveredStock]:
        return self._fetch_index("000905", "CSI500", 0.7)

    def _get_csi1000(self) -> list[DiscoveredStock]:
        return self._fetch_index("000852", "CSI1000", 0.6)

    def _fetch_index(
        self, symbol: str, source: str, base_score: float
    ) -> list[DiscoveredStock]:
        if not self._ak:
            return []
        try:
            self._wait()
            df = self._ak.index_stock_cons_csindex(symbol=symbol)
            return self._parse_index_df(df, source, base_score)
        except Exception as exc:
            log.warning(f"Failed to get {source} ({symbol}): {exc}")
            return []

    @staticmethod
    def _parse_index_df(
        df: pd.DataFrame | None, source: str, base_score: float
    ) -> list[DiscoveredStock]:
        if df is None or df.empty:
            return []

        code_col = _find_column(
            df, ["成分券代码", "证券代码", "代码", "stock_code", "code", "symbol"]
        )
        name_col = _find_column(
            df, ["成分券名称", "证券简称", "名称", "stock_name", "name"]
        )

        stocks: list[DiscoveredStock] = []
        for _, row in df.iterrows():
            s = DiscoveredStock(
                code=str(row.get(code_col, "")),
                name=str(row.get(name_col, "")) if name_col else "",
                source=source,
                score=base_score,
            )
            if s.is_valid():
                stocks.append(s)
        return stocks

    # ================================================================== #
    # Spot-DF slicing
    # ================================================================== #
    @staticmethod
    def _from_spot_df(
        df: pd.DataFrame | None, filter_type: str
    ) -> list[DiscoveredStock]:
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
        elif filter_type == "all":
            work = work.sort_values("成交额", ascending=False)

        items: list[DiscoveredStock] = []
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

    # ================================================================== #
    # Fallback (hardcoded blue-chips)
    # ================================================================== #
    @staticmethod
    def _get_fallback_stocks() -> list[DiscoveredStock]:
        """Return a curated list of liquid A-share blue-chips."""
        _FALLBACK = (
            ("600519", "贵州茅台"), ("601318", "中国平安"),
            ("600036", "招商银行"), ("000858", "五粮液"),
            ("600900", "长江电力"), ("000333", "美的集团"),
            ("000651", "格力电器"), ("002594", "比亚迪"),
            ("300750", "宁德时代"), ("002475", "立讯精密"),
            ("600887", "伊利股份"), ("603288", "海天味业"),
            ("600276", "恒瑞医药"), ("300760", "迈瑞医疗"),
            ("300015", "爱尔眼科"), ("601166", "兴业银行"),
            ("601398", "工商银行"), ("600030", "中信证券"),
            ("002230", "科大讯飞"), ("300059", "东方财富"),
            ("601857", "中国石油"), ("600028", "中国石化"),
            ("601088", "中国神华"), ("600309", "万华化学"),
            ("601012", "隆基绿能"), ("000568", "泸州老窖"),
            ("600000", "浦发银行"), ("601328", "交通银行"),
            ("000002", "万科A"),   ("002714", "牧原股份"),
            ("600690", "海尔智家"), ("000725", "京东方A"),
            ("601899", "紫金矿业"), ("600585", "海螺水泥"),
            ("002352", "顺丰控股"), ("300124", "汇川技术"),
            ("002415", "海康威视"), ("600031", "三一重工"),
            ("000001", "平安银行"), ("002304", "洋河股份"),
            ("601688", "华泰证券"), ("600104", "上汽集团"),
            ("601888", "中国中免"), ("600809", "山西汾酒"),
            ("002371", "北方华创"), ("688041", "海光信息"),
            ("688256", "寒武纪"),   ("300896", "爱美客"),
            ("688012", "中微公司"), ("002049", "紫光国微"),
            ("600050", "中国联通"), ("601728", "中国电信"),
            ("600941", "中国移动"), ("601669", "中国电建"),
            ("601668", "中国建筑"), ("601390", "中国中铁"),
            ("000063", "中兴通讯"), ("002460", "赣锋锂业"),
            ("300274", "阳光电源"), ("601816", "京沪高铁"),
            ("600438", "通威股份"), ("002466", "天齐锂业"),
            ("601225", "陕西煤业"), ("600048", "保利发展"),
            ("601633", "长城汽车"), ("002812", "恩捷股份"),
            ("300033", "同花顺"),   ("601919", "中远海控"),
            ("603259", "药明康德"), ("600346", "恒力石化"),
            ("002241", "歌尔股份"), ("688981", "中芯国际"),
            ("300347", "泰格医药"), ("600763", "通策医疗"),
            ("601100", "恒立液压"), ("300782", "卓胜微"),
            ("603501", "韦尔股份"), ("300661", "圣邦股份"),
            ("688036", "传音控股"), ("002709", "天赐材料"),
            ("300014", "亿纬锂能"), ("600745", "闻泰科技"),
            ("601865", "福莱特"),   ("300316", "晶盛机电"),
            ("688111", "金山办公"), ("300999", "金龙鱼"),
            ("603986", "兆易创新"), ("688561", "奇安信"),
            ("300308", "中际旭创"), ("002916", "深南电路"),
            ("300413", "芒果超媒"), ("601138", "工业富联"),
            ("600406", "国电南瑞"), ("601615", "明阳智能"),
            ("002382", "蓝思科技"), ("300122", "智飞生物"),
            ("600196", "复星医药"),
        )

        seen: set = set()
        stocks: list[DiscoveredStock] = []
        for code, name in _FALLBACK:
            s = DiscoveredStock(
                code=code, name=name, source="fallback", score=0.7
            )
            if s.is_valid() and s.code not in seen:
                seen.add(s.code)
                stocks.append(s)

        log.info(f"Fallback list: {len(stocks)} stocks")
        return stocks

# MODULE-LEVEL HELPERS

def _is_st(name: str | None) -> bool:
    """Check if a stock name indicates ST status."""
    if not name:
        return False
    return "ST" in name.upper()

def _find_column(
    df: pd.DataFrame, candidates: list[str]
) -> str | None:
    """Return the first column name from *candidates* that exists in *df*."""
    for col in candidates:
        if col in df.columns:
            return col
    # Last resort: first column
    return df.columns[0] if len(df.columns) > 0 else None
