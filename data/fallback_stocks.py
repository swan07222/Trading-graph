# data/fallback_stocks.py
"""Shared fallback stock list used by both discovery.py and universe.py.
Extracted to break the circular import: universe → discovery → universe.

FIX FALLBACK: Added update mechanism and validation for the fallback list.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

# Core fallback stocks - manually curated blue-chip A-share stocks
# Last updated: 2026-02-24
# To update: run update_fallback_stocks() or modify FALLBACK_STOCK_LIST directly
FALLBACK_STOCK_LIST: tuple[tuple[str, str], ...] = (
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
    ("000002", "万科 A"),   ("002714", "牧原股份"),
    ("600690", "海尔智家"), ("000725", "京东方 A"),
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


def get_fallback_codes() -> list[str]:
    """Get list of fallback stock codes."""
    return [code for code, _name in FALLBACK_STOCK_LIST]


def get_fallback_stock_count() -> int:
    """Get the number of fallback stocks."""
    return len(FALLBACK_STOCK_LIST)


def validate_fallback_codes() -> dict[str, list[str]]:
    """Validate fallback stock codes format.
    
    Returns:
        Dict with 'valid' and 'invalid' lists
    """
    valid = []
    invalid = []
    for code, name in FALLBACK_STOCK_LIST:
        if not code or not isinstance(code, str):
            invalid.append(f"{code}:{name}")
            continue
        cleaned = code.strip()
        if not cleaned.isdigit() or len(cleaned) != 6:
            invalid.append(f"{code}:{name}")
        else:
            valid.append(code)
    return {"valid": valid, "invalid": invalid}


def update_fallback_stocks(
    new_stocks: list[tuple[str, str]],
    save_to_file: bool = True,
    config_dir: str | None = None,
) -> dict[str, str]:
    """Update the fallback stock list.
    
    Args:
        new_stocks: List of (code, name) tuples
        save_to_file: Whether to save to a JSON file for persistence
        config_dir: Directory to save the update file (default: data/fallback_stocks_cache)
    
    Returns:
        Status dict with 'status', 'count', 'timestamp'
    """
    from config.settings import CONFIG
    
    # Validate new stocks
    validated = []
    for code, name in new_stocks:
        code_clean = str(code).strip()
        if code_clean.isdigit() and len(code_clean) == 6:
            validated.append((code_clean, str(name).strip()))
    
    if not validated:
        return {
            "status": "error",
            "message": "No valid stocks provided",
            "count": 0,
        }
    
    # Remove duplicates while preserving order
    seen = set()
    unique = []
    for code, name in validated:
        if code not in seen:
            seen.add(code)
            unique.append((code, name))
    
    # Save to cache file if requested
    if save_to_file:
        cache_dir = Path(config_dir) if config_dir else Path(CONFIG.data_dir) / "fallback_stocks_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        cache_file = cache_dir / "fallback_stocks_update.json"
        update_data = {
            "updated_at": datetime.now().isoformat(),
            "stock_count": len(unique),
            "stocks": [{"code": code, "name": name} for code, name in unique],
        }
        
        try:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(update_data, f, ensure_ascii=False, indent=2)
        except OSError as e:
            return {
                "status": "error",
                "message": f"Failed to save update file: {e}",
                "count": len(unique),
            }
    
    return {
        "status": "success",
        "message": f"Updated {len(unique)} fallback stocks",
        "count": len(unique),
        "timestamp": datetime.now().isoformat(),
    }


def load_fallback_stocks_from_cache() -> list[tuple[str, str]] | None:
    """Load fallback stocks from cache file if available.
    
    Returns:
        List of (code, name) tuples or None if cache doesn't exist
    """
    from config.settings import CONFIG
    
    cache_file = Path(CONFIG.data_dir) / "fallback_stocks_cache" / "fallback_stocks_update.json"
    if not cache_file.exists():
        return None
    
    try:
        with open(cache_file, encoding="utf-8") as f:
            data = json.load(f)
        
        stocks = data.get("stocks", [])
        return [(s["code"], s["name"]) for s in stocks if "code" in s and "name" in s]
    except (json.JSONDecodeError, OSError, KeyError):
        return None
