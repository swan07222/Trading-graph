def normalize_cn_code(code: str) -> str:
    """
    Canonical CN stock code normalization for UI/feeds/execution.
    Returns 6-digit code or "".
    """
    from core.instruments import parse_instrument
    inst = parse_instrument(code)
    if inst.get("market") == "CN" and inst.get("asset") == "EQUITY":
        return str(inst.get("symbol") or "").zfill(6)
    return ""