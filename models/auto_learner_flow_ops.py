from __future__ import annotations

import threading
from datetime import datetime
from typing import Any

import numpy as np

from config.settings import CONFIG
from data.fetcher import get_fetcher
from utils.logger import get_logger

log = get_logger(__name__)


def _compute_lookback_bars(self: Any, interval: str) -> int:
    """Compute default lookback bars for an interval."""
    try:
        from data.fetcher import BARS_PER_DAY, INTERVAL_MAX_DAYS

        bpd = BARS_PER_DAY.get(str(interval).lower(), 1)
        max_d = INTERVAL_MAX_DAYS.get(str(interval).lower(), 500)
    except ImportError:
        bpd = self._BARS_PER_DAY_FALLBACK.get(str(interval).lower(), 1)
        max_d = self._INTERVAL_MAX_DAYS_FALLBACK.get(str(interval).lower(), 500)
    iv = str(interval).lower()
    is_intraday = iv in ("1m", "2m", "5m", "15m", "30m", "60m", "1h")
    target_days = min(int(max_d), 7) if is_intraday else min(int(max_d), 365)
    bars = int(max(1, round(float(bpd) * float(target_days))))
    if str(iv) == "1m":
        return int(max(self._MIN_1M_LOOKBACK_BARS, bars))
    if is_intraday:
        return max(120, bars)
    return min(max(200, bars), 3000)


def _interval_seconds(interval: str) -> int:
    """Map interval token to seconds for continuity checks."""
    iv = str(interval or "1m").strip().lower()
    mapping = {
        "1m": 60,
        "2m": 120,
        "3m": 180,
        "5m": 300,
        "15m": 900,
        "30m": 1800,
        "60m": 3600,
        "1h": 3600,
        "1d": 86400,
        "1wk": 604800,
        "1mo": 2592000,
    }
    return int(mapping.get(iv, 60))


def _session_continuous_window_seconds(
    self: Any,
    code: str,
    interval: str,
    max_bars: int = 5000,
) -> float:
    """
    Longest continuous cached window for a symbol/interval in seconds.
    Uses session bars captured during trading, including partial bars.
    """
    try:
        from data.session_cache import get_session_bar_cache

        cache = get_session_bar_cache()
        df = cache.read_history(
            symbol=code,
            interval=interval,
            bars=max(10, int(max_bars)),
            final_only=False,
        )
    except Exception as e:
        log.debug("Coverage score history read failed for %s: %s", code, e)
        return 0.0

    if df is None or df.empty:
        return 0.0

    step = float(max(1, self._interval_seconds(interval)))
    buckets: list[int] = []
    try:
        for ts in df.index.tolist():
            try:
                ep = float(ts.timestamp())
            except (AttributeError, OSError, OverflowError, TypeError, ValueError):
                continue
            if not np.isfinite(ep):
                continue
            buckets.append(int(ep // step))
    except Exception as e:
        log.debug("Coverage score timestamp processing failed for %s: %s", code, e)
        return 0.0

    if not buckets:
        return 0.0

    uniq = sorted(set(buckets))
    run = 1
    longest = 1
    for i in range(1, len(uniq)):
        if (uniq[i] - uniq[i - 1]) <= 1:
            run += 1
        else:
            longest = max(longest, run)
            run = 1
    longest = max(longest, run)

    return float(max(0, longest - 1) * step)


def _filter_priority_session_codes(
    self: Any,
    codes: list[str],
    interval: str,
    min_seconds: float = 3600.0,
) -> list[str]:
    """
    Keep only session-priority symbols with enough continuous captured data.
    For intraday training this enforces >=1 hour of contiguous session bars.
    """
    iv = str(interval or "").strip().lower()
    intraday = {"1m", "2m", "3m", "5m", "15m", "30m", "60m", "1h"}

    dedup: list[str] = []
    seen: set[str] = set()
    for raw in (codes or []):
        c = str(raw).strip()
        if not c or c in seen:
            continue
        seen.add(c)
        dedup.append(c)

    if iv not in intraday:
        return dedup

    filtered: list[str] = []
    dropped = 0
    for c in dedup:
        span_s = self._session_continuous_window_seconds(c, iv)
        if span_s >= float(min_seconds):
            filtered.append(c)
        else:
            dropped += 1

    if dropped > 0:
        self.progress.add_warning(
            f"Skipped {dropped} session-priority stocks without >=1h continuous {iv} bars"
        )
    return filtered


def _norm_code(raw: str) -> str:
    code = "".join(c for c in str(raw or "").strip() if c.isdigit())
    return code.zfill(6) if code else ""


def _prioritize_codes_by_news(
    self: Any,
    codes: list[str],
    interval: str,
    max_probe: int = 16,
) -> list[str]:
    """
    Reorder candidate symbols by fresh market/stock news relevance.
    Keeps original order for ties and when news is unavailable.
    """
    ordered = [self._norm_code(c) for c in list(codes or [])]
    ordered = [c for c in ordered if c]
    if len(ordered) <= 1:
        return ordered

    try:
        from data.news import get_news_aggregator

        agg = get_news_aggregator()
    except Exception as e:
        log.debug("News prioritization disabled (aggregator unavailable): %s", e)
        return ordered

    candidate_set = set(ordered)
    scores: dict[str, float] = {c: 0.0 for c in ordered}
    now = datetime.now()

    try:
        market_news = agg.get_market_news(count=80, force_refresh=False)
    except Exception as e:
        log.debug("Market-news fetch failed during prioritization: %s", e)
        market_news = []

    for item in list(market_news or []):
        linked = {self._norm_code(x) for x in list(getattr(item, "stock_codes", []) or [])}
        linked = {c for c in linked if c and c in candidate_set}
        if not linked:
            continue
        try:
            age_h = max(
                0.0,
                (now - getattr(item, "publish_time", now)).total_seconds() / 3600.0,
            )
        except (AttributeError, TypeError, ValueError):
            age_h = 24.0
        recency = 1.0 / (1.0 + (age_h / 10.0))
        sentiment_mag = abs(float(getattr(item, "sentiment_score", 0.0) or 0.0))
        importance = float(getattr(item, "importance", 0.5) or 0.5)
        weight = recency * max(0.2, min(1.6, importance)) * (0.40 + sentiment_mag)
        for code in linked:
            scores[code] = float(scores.get(code, 0.0) + weight)

    # Optional light probe for top unseen candidates to capture stock-specific headlines.
    probed = 0
    for code in ordered:
        if self._should_stop():
            break
        if probed >= int(max(0, max_probe)):
            break
        if scores.get(code, 0.0) > 0.0:
            continue
        try:
            summary = agg.get_sentiment_summary(code)
        except Exception as e:
            log.debug("Sentiment summary probe failed for %s: %s", code, e)
            continue
        count = int(summary.get("total", 0) or 0)
        if count <= 0:
            continue
        conf = float(summary.get("confidence", 0.0) or 0.0)
        sent = abs(float(summary.get("overall_sentiment", 0.0) or 0.0))
        momentum = abs(float(summary.get("sentiment_momentum_6h", 0.0) or 0.0))
        score = (0.45 * sent + 0.25 * momentum + 0.30 * conf) * min(1.0, count / 12.0)
        scores[code] = float(score)
        probed += 1

    ranked = sorted(
        enumerate(ordered),
        key=lambda it: (-float(scores.get(it[1], 0.0)), it[0]),
    )
    out = [code for _, code in ranked]

    moved = sum(1 for i, code in enumerate(out) if i < len(ordered) and code != ordered[i])
    if moved > 0:
        self._update(
            message=(
                f"News-prioritized candidates: {sum(1 for v in scores.values() if v > 0):d} "
                f"stocks with signal"
            ),
            progress=max(3.0, float(self.progress.progress)),
        )
    return out


def run(self: Any, **kwargs: Any) -> Any:
    kwargs.setdefault("continuous", False)
    self.start(**kwargs)
    with self._thread_lock:
        thread = self._thread
    if thread:
        thread.join()
    return self.progress


def stop(self: Any, join_timeout: float = 30.0) -> None:
    log.info("Stopping learning...")
    self._cancel_token.cancel()
    with self._thread_lock:
        thread = self._thread
    if thread and thread is not threading.current_thread():
        timeout = max(0.5, float(join_timeout))
        thread.join(timeout=timeout)
        if thread.is_alive():
            log.info("Learning thread still finalizing after stop request")
        else:
            with self._thread_lock:
                if self._thread is thread:
                    self._thread = None
    self._save_state()
    self.progress.is_running = False
    self._notify()


def pause(self: Any) -> None:
    self.progress.is_paused = True
    self._notify()


def resume(self: Any) -> None:
    self.progress.is_paused = False
    self._notify()


def validate_stock_code(
    self: Any,
    code: str,
    interval: str = "1m",
    *,
    get_fetcher_fn: Any | None = None,
) -> dict[str, Any]:
    """Validate that a stock code exists and has sufficient data."""
    code = str(code).strip()
    if not code:
        return {
            "valid": False,
            "code": code,
            "name": "",
            "bars": 0,
            "message": "Empty stock code",
        }

    fetcher_getter = get_fetcher if get_fetcher_fn is None else get_fetcher_fn
    try:
        fetcher = fetcher_getter()
        bars_for_interval = max(
            300,
            int(self._compute_lookback_bars(interval)),
        )
        try:
            df = fetcher.get_history(
                code,
                interval=interval,
                bars=bars_for_interval,
                use_cache=True,
                update_db=True,
                allow_online=True,
                refresh_intraday_after_close=True,
            )
        except TypeError:
            df = fetcher.get_history(
                code, interval=interval, bars=bars_for_interval, use_cache=True
            )

        if df is None or df.empty:
            return {
                "valid": False,
                "code": code,
                "name": "",
                "bars": 0,
                "message": f"No data found for {code}",
            }

        bars = len(df)
        min_bars = CONFIG.SEQUENCE_LENGTH + 20

        if bars < min_bars:
            return {
                "valid": False,
                "code": code,
                "name": "",
                "bars": bars,
                "message": f"Insufficient data: {bars} bars (need at least {min_bars})",
            }

        name = ""
        try:
            from data.fetcher import get_spot_cache

            spot = get_spot_cache()
            quote = spot.get_quote(code)
            if quote and quote.get("name"):
                name = str(quote["name"])
        except Exception as e:
            log.debug("Spot-cache name lookup failed for %s: %s", code, e)

        return {
            "valid": True,
            "code": code,
            "name": name,
            "bars": bars,
            "message": f"OK - {bars} bars available",
        }

    except Exception as e:
        return {
            "valid": False,
            "code": code,
            "name": "",
            "bars": 0,
            "message": f"Validation error: {str(e)[:200]}",
        }
