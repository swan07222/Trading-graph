from __future__ import annotations

import json
import time

import pandas as pd

from data.fetcher_sources import (
    _TENCENT_CHUNK_SIZE,
    DataSource,
    Quote,
    _build_tencent_batch_url,
    _build_tencent_daily_url,
    _endpoint_candidates,
)
from utils.logger import get_logger

log = get_logger(__name__)

class TencentQuoteSource(DataSource):
    """Tencent quotes -> works from ANY IP (China or foreign)."""

    name = "tencent"
    priority = 0
    needs_china_direct = False
    _CB_ERROR_THRESHOLD = 10
    _CB_MIN_COOLDOWN = 18
    _CB_MAX_COOLDOWN = 75
    _CB_COOLDOWN_INCREMENT = 2
    _CB_HALF_OPEN_PROBE_INTERVAL = 6.0
    _BATCH_ENDPOINTS = ("https://qt.gtimg.cn/q={symbols}",)
    _BATCH_ENDPOINTS_ENV = "TRADING_TENCENT_BATCH_ENDPOINTS"
    _DAILY_ENDPOINTS = (
        "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
        "?param={vendor_symbol},day,,,{fetch_count},qfq",
    )
    _DAILY_ENDPOINTS_ENV = "TRADING_TENCENT_DAILY_ENDPOINTS"

    def get_realtime_batch(self, codes: list[str]) -> dict[str, Quote]:
        if not self.is_available():
            return {}

        from core.constants import get_exchange

        vendor_symbols: list[str] = []
        vendor_to_code: dict[str, str] = {}
        vendor_to_exchange: dict[str, str] = {}
        for c in codes:
            code6 = str(c).zfill(6)
            ex = get_exchange(code6)
            prefix_map = {"SSE": "sh", "SZSE": "sz", "BSE": "bj"}
            prefix = prefix_map.get(ex)
            if prefix is None:
                continue
            sym = f"{prefix}{code6}"
            vendor_symbols.append(sym)
            vendor_to_code[sym] = code6
            vendor_to_exchange[sym] = str(ex or "").upper()

        if not vendor_symbols:
            return {}

        out: dict[str, Quote] = {}
        start_all = time.time()
        endpoint_errors: list[str] = []
        batch_endpoints = _endpoint_candidates(
            self._BATCH_ENDPOINTS_ENV,
            self._BATCH_ENDPOINTS,
        )

        try:
            for i in range(0, len(vendor_symbols), _TENCENT_CHUNK_SIZE):
                chunk = vendor_symbols[i: i + _TENCENT_CHUNK_SIZE]
                chunk_csv = ",".join(chunk)
                chunk_ok = False
                for endpoint in batch_endpoints:
                    url = _build_tencent_batch_url(endpoint, chunk_csv)
                    if not url:
                        continue
                    try:
                        resp = self._session.get(url, timeout=10)
                        if int(resp.status_code) != 200:
                            endpoint_errors.append(f"{url} HTTP {resp.status_code}")
                            continue
                        resp.encoding = "gbk"  # Tencent returns GBK encoded content

                        raw_lines: list[str] = []
                        for block in str(resp.text or "").splitlines():
                            block_txt = str(block or "").strip()
                            if not block_txt:
                                continue
                            if ";" in block_txt:
                                raw_lines.extend(
                                    seg.strip()
                                    for seg in block_txt.split(";")
                                    if str(seg or "").strip()
                                )
                            else:
                                raw_lines.append(block_txt)

                        parsed_before = len(out)
                        schema_mismatches = 0
                        for line in raw_lines:
                            if "~" not in line or "=" not in line:
                                continue
                            try:
                                left, right = line.split("=", 1)
                                vendor_sym = left.strip().replace("v_", "")
                                payload = right.strip().strip('";')
                                if not payload:
                                    continue
                                parts = payload.split("~")
                                if len(parts) < 32:
                                    schema_mismatches += 1
                                    continue

                                code6 = vendor_to_code.get(vendor_sym)
                                if not code6:
                                    continue

                                name = str(parts[1]) if parts[1] else ""
                                price_str = parts[3].strip()
                                if not price_str:
                                    continue
                                price = float(price_str)
                                if price <= 0:
                                    continue

                                prev_close = float(parts[4] or 0)
                                open_px = float(parts[5] or 0)
                                volume = int(float(parts[6] or 0) * 100)
                                amount = (
                                    float(parts[37] or 0)
                                    if len(parts) > 37 else 0.0
                                )
                                high_px = (
                                    float(parts[33] or price)
                                    if len(parts) > 33 else price
                                )
                                low_px = (
                                    float(parts[34] or price)
                                    if len(parts) > 34 else price
                                )
                                bid_px = (
                                    float(parts[9] or 0)
                                    if len(parts) > 9 else 0.0
                                )
                                ask_px = (
                                    float(parts[19] or 0)
                                    if len(parts) > 19 else 0.0
                                )

                                # Validate price bounds (sanity check)
                                if prev_close > 0:
                                    ratio = price / prev_close
                                    ex_name = str(
                                        vendor_to_exchange.get(vendor_sym, "")
                                    ).upper()
                                    max_move = 0.25
                                    if ex_name == "BSE":
                                        try:
                                            from core.constants import PRICE_LIMITS
                                            bse_limit = float(
                                                PRICE_LIMITS.get("bse", 0.30) or 0.30
                                            )
                                        except (ImportError, AttributeError, TypeError, ValueError):
                                            bse_limit = 0.30
                                        max_move = max(0.30, bse_limit) + 0.03
                                    if ratio > (1.0 + max_move) or ratio < (1.0 - max_move):
                                        log.debug(
                                            "Tencent: suspicious price for %s: "
                                            "price=%.2f prev_close=%.2f cap=%.1f%%",
                                            code6, price, prev_close, max_move * 100.0,
                                        )
                                        continue

                                # Fix OHLC bounds
                                open_px = open_px if open_px > 0 else price
                                high_px = max(high_px, open_px, price)
                                low_px = min(low_px, open_px, price)
                                if low_px <= 0:
                                    low_px = price

                                chg = price - prev_close if prev_close > 0 else 0.0
                                chg_pct = (
                                    (chg / prev_close * 100)
                                    if prev_close > 0 else 0.0
                                )

                                out[code6] = Quote(
                                    code=code6,
                                    name=name,
                                    price=price,
                                    open=open_px,
                                    high=high_px,
                                    low=low_px,
                                    close=price,
                                    volume=volume,
                                    amount=amount,
                                    change=chg,
                                    change_pct=chg_pct,
                                    bid=bid_px,
                                    ask=ask_px,
                                    source=self.name,
                                    is_delayed=False,
                                    latency_ms=0.0,
                                )
                            except Exception as exc:
                                log.debug("Tencent parse error line: %s", exc)
                                continue

                        parsed_now = len(out)
                        if parsed_now > parsed_before:
                            chunk_ok = True
                            break
                        if schema_mismatches > 0:
                            endpoint_errors.append(
                                f"{url} schema_mismatch={schema_mismatches}"
                            )
                        else:
                            endpoint_errors.append(f"{url} empty_payload")
                    except Exception as exc:
                        endpoint_errors.append(f"{url} {exc}")
                        continue
                if not chunk_ok:
                    log.debug(
                        "Tencent quote chunk unresolved after endpoint rotation "
                        "(chunk=%d)",
                        len(chunk),
                    )

            latency = (time.time() - start_all) * 1000
            if not out:
                err = "; ".join(endpoint_errors[:2]) if endpoint_errors else "no_quotes"
                self._record_error(err)
                return {}
            self._record_success(latency)
            for q in out.values():
                q.latency_ms = latency
            log.debug(
                "Tencent batch: %d/%d quotes fetched in %.0fms",
                len(out), len(codes), latency
            )
            return out

        except Exception as exc:
            self._record_error(str(exc))
            log.debug("Tencent batch failed: %s", exc)
            return {}

    def get_realtime(self, code: str) -> Quote | None:
        res = self.get_realtime_batch([code])
        return res.get(str(code).zfill(6))

    def get_history(self, code: str, days: int) -> pd.DataFrame:
        inst = {
            "market": "CN", "asset": "EQUITY",
            "symbol": str(code).zfill(6)
        }
        return self.get_history_instrument(inst, days=days, interval="1d")

    def get_history_instrument(
        self, inst: dict, days: int, interval: str = "1d"
    ) -> pd.DataFrame:
        if inst.get("market") != "CN" or inst.get("asset") != "EQUITY":
            return pd.DataFrame()
        if str(interval).lower() != "1d":
            return pd.DataFrame()

        code6 = str(inst.get("symbol") or "").zfill(6)
        if not code6.isdigit() or len(code6) != 6:
            return pd.DataFrame()

        from core.constants import get_exchange
        ex = get_exchange(code6)
        prefix = {"SSE": "sh", "SZSE": "sz", "BSE": "bj"}.get(ex)
        if not prefix:
            return pd.DataFrame()

        vendor_symbol = f"{prefix}{code6}"
        start_t = time.time()
        daily_endpoints = _endpoint_candidates(
            self._DAILY_ENDPOINTS_ENV,
            self._DAILY_ENDPOINTS,
        )
        errors: list[str] = []
        try:
            # Request more bars than needed to account for gaps
            fetch_count = max(100, int(days) + 60)
            for endpoint in daily_endpoints:
                url = _build_tencent_daily_url(
                    endpoint,
                    vendor_symbol=vendor_symbol,
                    fetch_count=fetch_count,
                )
                if not url:
                    continue
                try:
                    resp = self._session.get(url, timeout=12)
                    resp.encoding = "utf-8"
                    if int(resp.status_code) != 200:
                        errors.append(f"{url} HTTP {resp.status_code}")
                        continue
                    payload = str(resp.text or "")
                    if not payload:
                        errors.append(f"{url} empty_payload")
                        continue
                    data = self._parse_daily_kline(payload, vendor_symbol)
                    if data.empty:
                        errors.append(f"{url} empty_bars")
                        continue
                    latency = (time.time() - start_t) * 1000.0
                    self._record_success(latency)
                    log.debug(
                        "Tencent daily %s: %d bars in %.0fms",
                        code6, len(data), latency
                    )
                    return data.tail(max(1, int(days)))
                except Exception as exc:
                    errors.append(f"{url} {exc}")
                    continue
            if errors:
                self._record_error("; ".join(errors[:2]))
            return pd.DataFrame()
        except Exception as exc:
            self._record_error(str(exc))
            log.debug("Tencent history failed for %s: %s", code6, exc)
            return pd.DataFrame()

    @staticmethod
    def _parse_daily_kline(payload_text: str, vendor_symbol: str) -> pd.DataFrame:
        """Parse Tencent qfq daily K-line JSON response."""
        text = str(payload_text or "").strip()
        if not text:
            return pd.DataFrame()

        payload = None
        try:
            payload = json.loads(text)
        except (json.JSONDecodeError, ValueError, TypeError):
            # Handle JSONP wrapper
            left = text.find("{")
            right = text.rfind("}")
            if left < 0 or right <= left:
                return pd.DataFrame()
            try:
                payload = json.loads(text[left: right + 1])
            except (json.JSONDecodeError, ValueError, TypeError):
                return pd.DataFrame()

        if not isinstance(payload, dict):
            return pd.DataFrame()

        data_root = payload.get("data")
        if not isinstance(data_root, dict):
            return pd.DataFrame()

        item = data_root.get(vendor_symbol)
        if not isinstance(item, dict):
            return pd.DataFrame()

        # Try multiple key names Tencent uses
        rows = (
            item.get("qfqday")
            or item.get("day")
            or item.get("hfqday")
            or []
        )
        if not isinstance(rows, list) or not rows:
            return pd.DataFrame()

        out_rows = []
        for row in rows:
            if not isinstance(row, (list, tuple)) or len(row) < 6:
                continue
            try:
                # Tencent format: [date, open, close, high, low, volume, ...]
                date_str = str(row[0]).strip()
                date = pd.to_datetime(date_str, errors="coerce")
                if pd.isna(date):
                    continue

                open_px  = float(row[1] or 0)
                close_px = float(row[2] or 0)
                high_px  = float(row[3] or 0)
                low_px   = float(row[4] or 0)
                # Volume in Tencent daily is in lots (手), convert to shares
                vol      = float(row[5] or 0) * 100

                if close_px <= 0:
                    continue

                # Fix OHLC
                open_px = open_px if open_px > 0 else close_px
                high_px = max(high_px, open_px, close_px)
                low_px  = min(low_px,  open_px, close_px)
                if low_px <= 0:
                    low_px = close_px

                if high_px < low_px:
                    continue

                # Amount: Tencent sometimes provides index 6
                amount = 0.0
                if len(row) > 6:
                    try:
                        amount = float(row[6] or 0)
                    except (ValueError, TypeError, OverflowError):
                        amount = close_px * max(0.0, vol)
                else:
                    amount = close_px * max(0.0, vol)

                out_rows.append({
                    "date":   date,
                    "open":   open_px,
                    "high":   high_px,
                    "low":    low_px,
                    "close":  close_px,
                    "volume": max(0.0, vol),
                    "amount": max(0.0, amount),
                })
            except (ValueError, TypeError, OverflowError, pd.errors.ParserError):
                continue

        if not out_rows:
            return pd.DataFrame()

        df = (
            pd.DataFrame(out_rows)
            .dropna(subset=["date"])
            .sort_values("date")
            .set_index("date")
        )
        return df
