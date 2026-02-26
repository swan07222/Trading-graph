from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from config import CONFIG
from data.fetcher import DataFetcher
from data.news_aggregator import get_news_aggregator
from data.sentiment_analyzer import get_analyzer


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _parse_symbols(raw: str) -> list[str]:
    values = [item.strip() for item in str(raw or "").split(",") if item.strip()]
    if values:
        return values
    return [str(x) for x in list(getattr(CONFIG, "stock_pool", []) or [])[:3]]


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _safe_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analysis-only soak test for data/feed/news/sentiment runtime"
    )
    parser.add_argument(
        "--mode",
        default="paper",
        choices=["simulation", "paper", "live"],
        help="Run mode label for reporting compatibility",
    )
    parser.add_argument(
        "--duration-minutes",
        type=float,
        default=30.0,
        help="Soak duration in minutes",
    )
    parser.add_argument(
        "--poll-seconds",
        type=float,
        default=5.0,
        help="Polling interval per soak tick",
    )
    parser.add_argument(
        "--symbols",
        default="",
        help="Comma separated symbols for quote probes",
    )
    parser.add_argument(
        "--allow-live",
        action="store_true",
        help="Required safety switch when --mode live is selected",
    )
    parser.add_argument(
        "--max-failure-ticks",
        type=int,
        default=3,
        help="Maximum tolerated failed ticks",
    )
    parser.add_argument(
        "--max-consecutive-failures",
        type=int,
        default=2,
        help="Maximum tolerated consecutive failed ticks",
    )
    parser.add_argument(
        "--min-quote-success-ratio",
        type=float,
        default=0.95,
        help="Minimum acceptable quote probe success ratio",
    )
    parser.add_argument(
        "--samples-limit",
        type=int,
        default=200,
        help="Max sample rows stored in JSON report",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Optional output JSON path",
    )
    # Compatibility flags kept for old automation scripts.
    parser.add_argument("--place-probe-order", action="store_true")
    parser.add_argument("--probe-every-ticks", type=int, default=60)
    parser.add_argument("--probe-symbol", default="")
    parser.add_argument("--probe-qty", type=int, default=100)
    parser.add_argument("--probe-price-factor", type=float, default=0.8)
    parser.add_argument("--probe-max-notional", type=float, default=10000.0)
    parser.add_argument("--allow-live-orders", action="store_true")
    args = parser.parse_args()

    if args.duration_minutes <= 0:
        raise SystemExit("--duration-minutes must be > 0")
    if args.poll_seconds <= 0:
        raise SystemExit("--poll-seconds must be > 0")
    if args.mode == "live" and not args.allow_live:
        raise SystemExit("--allow-live is required when --mode live")

    symbols = _parse_symbols(args.symbols)
    if not symbols:
        raise SystemExit("No symbols available for quote probes")

    fetcher = DataFetcher()
    aggregator = get_news_aggregator()
    analyzer = get_analyzer()

    started_at = _utc_now_iso()
    started_monotonic = time.monotonic()
    deadline = started_monotonic + float(args.duration_minutes) * 60.0

    samples: list[dict[str, Any]] = []
    probe_events: list[dict[str, Any]] = []
    failure_ticks = 0
    consecutive_failures = 0
    max_consecutive_failures = 0
    quote_probe_total = 0
    quote_probe_ok = 0
    status = "pass"
    fail_reasons: list[str] = []
    tick_count = 0
    news_probe_ok = 0
    news_probe_total = 0
    sentiment_probe_ok = 0
    sentiment_probe_total = 0

    try:
        while time.monotonic() < deadline:
            tick_started = time.monotonic()
            tick_count += 1
            tick_ok = True
            tick_errors: list[str] = []
            sample: dict[str, Any] = {
                "tick": tick_count,
                "ts": _utc_now_iso(),
                "mode": str(args.mode),
                "symbols": list(symbols),
            }

            try:
                quotes = fetcher.get_realtime_batch(symbols)
                good = 0
                for sym in symbols:
                    q = quotes.get(sym)
                    if q is None:
                        continue
                    px = _safe_float(getattr(q, "price", 0.0))
                    if px > 0:
                        good += 1
                sample["quote_probe_total"] = int(len(symbols))
                sample["quote_probe_ok"] = int(good)
                quote_probe_total += int(len(symbols))
                quote_probe_ok += int(good)
                if good < len(symbols):
                    tick_ok = False
                    tick_errors.append("quote_probe_partial_failure")
            except Exception as exc:
                tick_ok = False
                tick_errors.append(f"quote_probe_error:{exc}")

            # Probe news and sentiment every 6 ticks to avoid overload.
            if tick_count % 6 == 0:
                symbol = str(symbols[0])
                news_probe_total += 1
                sentiment_probe_total += 1
                try:
                    news = aggregator.get_stock_news(symbol, count=10)
                    sample["news_count"] = int(len(news))
                    if news:
                        news_probe_ok += 1
                    else:
                        tick_errors.append("news_empty")
                except Exception as exc:
                    tick_errors.append(f"news_error:{exc}")

                try:
                    recent = aggregator.get_stock_news(symbol, count=20)
                    sent = analyzer.analyze_articles(recent, hours_back=48)
                    conf = _safe_float(getattr(sent, "confidence", 0.0))
                    sample["sentiment_confidence"] = conf
                    sample["sentiment_overall"] = _safe_float(getattr(sent, "overall", 0.0))
                    if conf > 0.0:
                        sentiment_probe_ok += 1
                    else:
                        tick_errors.append("sentiment_confidence_zero")
                except Exception as exc:
                    tick_errors.append(f"sentiment_error:{exc}")

            if args.place_probe_order and (tick_count % max(1, int(args.probe_every_ticks)) == 0):
                probe_events.append(
                    {
                        "tick": tick_count,
                        "ts": _utc_now_iso(),
                        "status": "skipped",
                        "reason": "execution_removed_analysis_only_build",
                    }
                )

            sample["errors"] = tick_errors
            if tick_errors:
                tick_ok = False

            if not tick_ok:
                failure_ticks += 1
                consecutive_failures += 1
            else:
                consecutive_failures = 0
            max_consecutive_failures = max(max_consecutive_failures, consecutive_failures)

            samples.append(sample)
            if len(samples) > int(args.samples_limit):
                samples = samples[-int(args.samples_limit) :]

            sleep_left = float(args.poll_seconds) - (time.monotonic() - tick_started)
            if sleep_left > 0:
                time.sleep(sleep_left)

    except Exception as exc:
        status = "fail"
        fail_reasons.append(str(exc))

    ended_at = _utc_now_iso()
    quote_success_ratio = (
        float(quote_probe_ok) / float(quote_probe_total)
        if quote_probe_total > 0
        else 0.0
    )
    news_success_ratio = (
        float(news_probe_ok) / float(news_probe_total)
        if news_probe_total > 0
        else 0.0
    )
    sentiment_success_ratio = (
        float(sentiment_probe_ok) / float(sentiment_probe_total)
        if sentiment_probe_total > 0
        else 0.0
    )

    if tick_count == 0:
        status = "fail"
        fail_reasons.append("no_ticks_collected")
    if failure_ticks > int(args.max_failure_ticks):
        status = "fail"
        fail_reasons.append(
            f"failure_ticks={failure_ticks} > max_failure_ticks={int(args.max_failure_ticks)}"
        )
    if max_consecutive_failures > int(args.max_consecutive_failures):
        status = "fail"
        fail_reasons.append(
            "max_consecutive_failures="
            f"{max_consecutive_failures} > "
            f"max_consecutive_failures={int(args.max_consecutive_failures)}"
        )
    if quote_success_ratio < float(args.min_quote_success_ratio):
        status = "fail"
        fail_reasons.append(
            f"quote_success_ratio={quote_success_ratio:.4f} < "
            f"min_quote_success_ratio={float(args.min_quote_success_ratio):.4f}"
        )

    report: dict[str, Any] = {
        "status": status,
        "mode": str(args.mode),
        "started_at": started_at,
        "ended_at": ended_at,
        "duration_minutes_requested": float(args.duration_minutes),
        "tick_count": int(tick_count),
        "failure_ticks": int(failure_ticks),
        "max_consecutive_failures_observed": int(max_consecutive_failures),
        "quote_probe_total": int(quote_probe_total),
        "quote_probe_ok": int(quote_probe_ok),
        "quote_success_ratio": round(float(quote_success_ratio), 6),
        "news_probe_total": int(news_probe_total),
        "news_probe_ok": int(news_probe_ok),
        "news_success_ratio": round(float(news_success_ratio), 6),
        "sentiment_probe_total": int(sentiment_probe_total),
        "sentiment_probe_ok": int(sentiment_probe_ok),
        "sentiment_success_ratio": round(float(sentiment_success_ratio), 6),
        "probe_events": probe_events[-50:],
        "fail_reasons": fail_reasons,
        "samples": samples,
    }

    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.output:
        _safe_write_json(Path(args.output), report)
        print(f"soak report written: {args.output}")

    return 0 if status == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
