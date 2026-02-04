"""
Universal Stock Discovery System
Searches ALL available sources for stocks to train on
"""
import time
import random
from datetime import datetime, timedelta
from typing import List, Set, Dict, Callable, Optional
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from config import CONFIG
from utils.logger import log


@dataclass
class DiscoveredStock:
    """Stock discovered from any source"""
    code: str
    name: str = ""
    source: str = ""
    score: float = 0.5
    market_cap: float = 0
    volume: float = 0
    change_pct: float = 0
    sector: str = ""
    discovered_at: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        self.code = self._clean_code(self.code)
    
    def _clean_code(self, code: str) -> str:
        if not code:
            return ""
        code = str(code).strip()
        for prefix in ['sh', 'sz', 'SH', 'SZ', 'bj', 'BJ']:
            code = code.replace(prefix, '')
        code = code.replace('.', '').replace('-', '')
        if code.isdigit():
            return code.zfill(6)
        return ""
    
    def is_valid(self) -> bool:
        if not self.code or len(self.code) != 6 or not self.code.isdigit():
            return False
        valid_prefixes = ['60', '00', '30', '68', '83', '43']  # Include all markets
        return any(self.code.startswith(p) for p in valid_prefixes)
    
    @property
    def market(self) -> str:
        if self.code.startswith('60') or self.code.startswith('68'):
            return 'SH'
        elif self.code.startswith('00') or self.code.startswith('30'):
            return 'SZ'
        elif self.code.startswith('83') or self.code.startswith('43'):
            return 'BJ'
        return 'UNKNOWN'


class UniversalStockDiscovery:
    """
    Discovers ALL available stocks from multiple sources:
    1. AkShare market data (real-time quotes for all stocks)
    2. Index constituents (CSI 300, CSI 500, CSI 1000, etc.)
    3. Industry/sector lists
    4. Top movers (gainers, losers, volume)
    5. Analyst recommendations
    6. News mentions
    """
    
    def __init__(self):
        self._ak = None
        self._rate_limit = 1.0
        self._last_request = 0
        self._lock = threading.Lock()
        
        try:
            import akshare as ak
            self._ak = ak
        except ImportError:
            log.warning("AkShare not available - limited discovery")
    
    def _wait(self):
        with self._lock:
            elapsed = time.time() - self._last_request
            if elapsed < self._rate_limit:
                time.sleep(self._rate_limit - elapsed)
            self._last_request = time.time()
    
    def discover_all(
        self,
        callback: Callable[[str, int], None] = None,
        max_stocks: int = None,
        min_market_cap: float = 0,
        include_st: bool = False
    ) -> List[DiscoveredStock]:
        """
        Discover ALL available stocks from all sources
        
        Args:
            callback: Progress callback (message, count)
            max_stocks: Maximum stocks to return (None = all)
            min_market_cap: Minimum market cap in billions
            include_st: Include ST/*ST stocks
        
        Returns:
            List of discovered stocks, deduplicated and scored
        """
        all_stocks: Dict[str, DiscoveredStock] = {}
        
        sources = [
            ("All A-Shares", self._get_all_a_shares),
            ("CSI 300 Index", self._get_csi300),
            ("CSI 500 Index", self._get_csi500),
            ("CSI 1000 Index", self._get_csi1000),
            ("Top Gainers", self._get_top_gainers),
            ("Top Losers", self._get_top_losers),
            ("High Volume", self._get_high_volume),
            ("Large Cap", self._get_large_cap),
            ("Growth Stocks", self._get_growth_stocks),
            ("Value Stocks", self._get_value_stocks),
        ]
        
        for source_name, source_fn in sources:
            if callback:
                callback(f"Searching {source_name}...", len(all_stocks))
            
            try:
                self._wait()
                stocks = source_fn()
                
                for stock in stocks:
                    if not stock.is_valid():
                        continue
                    
                    # Filter ST stocks
                    if not include_st and 'ST' in stock.name.upper():
                        continue
                    
                    # Merge with existing or add new
                    if stock.code in all_stocks:
                        existing = all_stocks[stock.code]
                        # Combine scores
                        existing.score = (existing.score + stock.score) / 2
                        # Keep better data
                        if stock.market_cap > existing.market_cap:
                            existing.market_cap = stock.market_cap
                        if stock.volume > existing.volume:
                            existing.volume = stock.volume
                    else:
                        all_stocks[stock.code] = stock
                
                log.info(f"Found {len(stocks)} from {source_name}, total: {len(all_stocks)}")
                
            except Exception as e:
                log.warning(f"Failed to search {source_name}: {e}")
        
        # Filter and sort
        result = list(all_stocks.values())
        
        # Filter by market cap
        if min_market_cap > 0:
            result = [s for s in result if s.market_cap >= min_market_cap * 1e8]
        
        # Score by multiple factors
        for stock in result:
            stock.score = self._calculate_score(stock)
        
        # Sort by score
        result.sort(key=lambda x: x.score, reverse=True)
        
        # Limit
        if max_stocks:
            result = result[:max_stocks]
        
        log.info(f"Total discovered: {len(result)} stocks")
        return result
    
    def _calculate_score(self, stock: DiscoveredStock) -> float:
        """Calculate comprehensive stock score for training priority"""
        score = 0.5
        
        # Market cap bonus (larger = more data, more stable)
        if stock.market_cap > 100e8:  # > 100B
            score += 0.2
        elif stock.market_cap > 50e8:
            score += 0.15
        elif stock.market_cap > 10e8:
            score += 0.1
        
        # Volume bonus (more liquid = better execution)
        if stock.volume > 1e9:  # > 1B daily
            score += 0.15
        elif stock.volume > 5e8:
            score += 0.1
        elif stock.volume > 1e8:
            score += 0.05
        
        # Volatility bonus (more movement = more trading opportunities)
        abs_change = abs(stock.change_pct)
        if 2 < abs_change < 8:
            score += 0.1  # Good volatility
        elif abs_change > 8:
            score += 0.05  # Too volatile
        
        # Index constituent bonus
        if stock.source in ['CSI300', 'CSI500', 'CSI1000']:
            score += 0.1
        
        return min(score, 1.0)
    
    def _get_all_a_shares(self) -> List[DiscoveredStock]:
        """Get ALL A-share stocks"""
        if not self._ak:
            return []
        
        try:
            df = self._ak.stock_zh_a_spot_em()
            stocks = []
            
            for _, row in df.iterrows():
                stocks.append(DiscoveredStock(
                    code=str(row.get('代码', '')),
                    name=str(row.get('名称', '')),
                    source='AllAShares',
                    market_cap=float(row.get('总市值', 0) or 0),
                    volume=float(row.get('成交额', 0) or 0),
                    change_pct=float(row.get('涨跌幅', 0) or 0),
                    score=0.5
                ))
            
            return stocks
        except Exception as e:
            log.warning(f"Failed to get all A-shares: {e}")
            return []
    
    def _get_csi300(self) -> List[DiscoveredStock]:
        """Get CSI 300 index constituents"""
        if not self._ak:
            return []
        
        try:
            df = self._ak.index_stock_cons_csindex(symbol="000300")
            return [
                DiscoveredStock(
                    code=str(row.get('成分券代码', '')),
                    name=str(row.get('成分券名称', '')),
                    source='CSI300',
                    score=0.8
                )
                for _, row in df.iterrows()
            ]
        except:
            return []
    
    def _get_csi500(self) -> List[DiscoveredStock]:
        """Get CSI 500 index constituents"""
        if not self._ak:
            return []
        
        try:
            df = self._ak.index_stock_cons_csindex(symbol="000905")
            return [
                DiscoveredStock(
                    code=str(row.get('成分券代码', '')),
                    name=str(row.get('成分券名称', '')),
                    source='CSI500',
                    score=0.7
                )
                for _, row in df.iterrows()
            ]
        except:
            return []
    
    def _get_csi1000(self) -> List[DiscoveredStock]:
        """Get CSI 1000 index constituents"""
        if not self._ak:
            return []
        
        try:
            df = self._ak.index_stock_cons_csindex(symbol="000852")
            return [
                DiscoveredStock(
                    code=str(row.get('成分券代码', '')),
                    name=str(row.get('成分券名称', '')),
                    source='CSI1000',
                    score=0.6
                )
                for _, row in df.iterrows()
            ]
        except:
            return []
    
    def _get_top_gainers(self) -> List[DiscoveredStock]:
        """Get top gainers"""
        if not self._ak:
            return []
        
        try:
            df = self._ak.stock_zh_a_spot_em()
            df['涨跌幅'] = pd.to_numeric(df['涨跌幅'], errors='coerce')
            df = df.dropna(subset=['涨跌幅'])
            df = df.sort_values('涨跌幅', ascending=False).head(100)
            
            return [
                DiscoveredStock(
                    code=str(row['代码']),
                    name=str(row.get('名称', '')),
                    source='Gainers',
                    change_pct=float(row['涨跌幅']),
                    score=min(abs(float(row['涨跌幅'])) / 10, 0.8)
                )
                for _, row in df.iterrows()
            ]
        except:
            return []
    
    def _get_top_losers(self) -> List[DiscoveredStock]:
        """Get top losers"""
        if not self._ak:
            return []
        
        try:
            df = self._ak.stock_zh_a_spot_em()
            df['涨跌幅'] = pd.to_numeric(df['涨跌幅'], errors='coerce')
            df = df.dropna(subset=['涨跌幅'])
            df = df.sort_values('涨跌幅', ascending=True).head(100)
            
            return [
                DiscoveredStock(
                    code=str(row['代码']),
                    name=str(row.get('名称', '')),
                    source='Losers',
                    change_pct=float(row['涨跌幅']),
                    score=min(abs(float(row['涨跌幅'])) / 10, 0.7)
                )
                for _, row in df.iterrows()
            ]
        except:
            return []
    
    def _get_high_volume(self) -> List[DiscoveredStock]:
        """Get highest volume stocks"""
        if not self._ak:
            return []
        
        try:
            df = self._ak.stock_zh_a_spot_em()
            df['成交额'] = pd.to_numeric(df['成交额'], errors='coerce')
            df = df.dropna(subset=['成交额'])
            df = df.sort_values('成交额', ascending=False).head(100)
            
            return [
                DiscoveredStock(
                    code=str(row['代码']),
                    name=str(row.get('名称', '')),
                    source='HighVolume',
                    volume=float(row['成交额']),
                    score=0.7
                )
                for _, row in df.iterrows()
            ]
        except:
            return []
    
    def _get_large_cap(self) -> List[DiscoveredStock]:
        """Get largest market cap stocks"""
        if not self._ak:
            return []
        
        try:
            df = self._ak.stock_zh_a_spot_em()
            df['总市值'] = pd.to_numeric(df['总市值'], errors='coerce')
            df = df.dropna(subset=['总市值'])
            df = df.sort_values('总市值', ascending=False).head(200)
            
            return [
                DiscoveredStock(
                    code=str(row['代码']),
                    name=str(row.get('名称', '')),
                    source='LargeCap',
                    market_cap=float(row['总市值']),
                    score=0.75
                )
                for _, row in df.iterrows()
            ]
        except:
            return []
    
    def _get_growth_stocks(self) -> List[DiscoveredStock]:
        """Get growth stocks (high momentum)"""
        if not self._ak:
            return []
        
        try:
            # Use 60-day momentum
            df = self._ak.stock_zh_a_spot_em()
            df['60日涨跌幅'] = pd.to_numeric(df.get('60日涨跌幅', 0), errors='coerce')
            df = df.dropna(subset=['60日涨跌幅'])
            df = df.sort_values('60日涨跌幅', ascending=False).head(100)
            
            return [
                DiscoveredStock(
                    code=str(row['代码']),
                    name=str(row.get('名称', '')),
                    source='Growth',
                    score=0.65
                )
                for _, row in df.iterrows()
            ]
        except:
            return []
    
    def _get_value_stocks(self) -> List[DiscoveredStock]:
        """Get value stocks (low P/E, P/B)"""
        if not self._ak:
            return []
        
        try:
            df = self._ak.stock_zh_a_spot_em()
            df['市盈率'] = pd.to_numeric(df.get('市盈率-动态', 0), errors='coerce')
            df = df[(df['市盈率'] > 0) & (df['市盈率'] < 20)]
            df = df.sort_values('市盈率').head(100)
            
            return [
                DiscoveredStock(
                    code=str(row['代码']),
                    name=str(row.get('名称', '')),
                    source='Value',
                    score=0.6
                )
                for _, row in df.iterrows()
            ]
        except:
            return []


# Need pandas import
import pandas as pd