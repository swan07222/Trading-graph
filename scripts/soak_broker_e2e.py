from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import CONFIG, TradingMode
from core.types import Order, OrderSide, OrderType
from trading.executor import ExecutionEngine
from trading.health import HealthStatus, get_health_monitor


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _mode_from_text(raw: str) -> TradingMode:
    mode = str(raw or "").strip().lower()
    if mode == "simulation":
        return TradingMode.SIMULATION
    if mode == "paper":
        return TradingMode.PAPER
    if mode == "live":
        return TradingMode.LIVE
    raise ValueError(f"Unsupported mode: {raw}")


def _parse_symbols(raw: str) -> list[str]:
    values = [item.strip() for item in str(raw or "").split(",") if item.strip()]
    if values:
        return values
    return list(getattr(CONFIG, "stock_pool", [])[:3])


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _quote_probe(broker, symbols: list[str]) -> tuple[int, int]:
    checked = 0
    ok = 0
    for symbol in symbols:
        checked += 1
        try:
            px = broker.get_quote(symbol)
            if px is not None and float(px) > 0:
                ok += 1
        except Exception:
            continue
    return checked, ok


def _place_probe_order(
    engine: ExecutionEngine,
    symbol: str,
    qty: int,
    price_factor: float,
    max_notional: float,
) -> dict[str, Any]:
    broker = engine.broker
    quote = broker.get_quote(symbol)
    if quote is None or float(quote) <= 0:
        return {"status": "skipped", "reason": "quote_unavailable"}

    ref = float(quote)
    order_px = max(0.01, ref * float(price_factor))
    lot_size = max(1, int(getattr(CONFIG, "LOT_SIZE", 100)))
    final_qty = max(lot_size, (int(qty) // lot_size) * lot_size)
    notional = final_qty * order_px
    if notional > float(max_notional):
        capped_qty = int(float(max_notional) / max(order_px, 0.01))
        final_qty = max(lot_size, (capped_qty // lot_size) * lot_size)
        notional = final_qty * order_px

    if final_qty <= 0:
        return {"status": "skipped", "reason": "qty_non_positive_after_cap"}

    order = Order(
        symbol=symbol,
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        quantity=final_qty,
        price=order_px,
        strategy="soak_probe_order",
    )
    order.tags["soak_probe"] = True
    order.tags["probe_ref_quote"] = ref

    submitted = broker.submit_order(order)
    if submitted.is_active:
        try:
            broker.cancel_order(submitted.id)
        except Exception:
            pass

    return {
        "status": "ok",
        "order_id": submitted.id,
        "broker_id": submitted.broker_id,
        "order_status": submitted.status.value,
        "qty": int(final_qty),
        "price": float(order_px),
        "notional": float(notional),
        "ref_quote": float(ref),
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="End-to-end soak test against broker + execution runtime"
    )
    parser.add_argument(
        "--mode",
        default="paper",
        choices=["simulation", "paper", "live"],
        help="Trading mode for execution engine",
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
        help="Required safety switch for --mode live",
    )
    parser.add_argument(
        "--max-failure-ticks",
        type=int,
        default=3,
        help="Maximum tolerated failed ticks",
    )
    parser.add_argument(
        "--max-disconnect-ticks",
        type=int,
        default=0,
        help="Maximum tolerated broker disconnect ticks",
    )
    parser.add_argument(
        "--max-unhealthy-ticks",
        type=int,
        default=0,
        help="Maximum tolerated unhealthy/critical health ticks",
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
    parser.add_argument(
        "--place-probe-order",
        action="store_true",
        help="Submit a low-price buy + cancel probe order at intervals",
    )
    parser.add_argument(
        "--probe-every-ticks",
        type=int,
        default=60,
        help="Probe order interval in ticks",
    )
    parser.add_argument(
        "--probe-symbol",
        default="",
        help="Symbol used for probe order (defaults to first probe symbol)",
    )
    parser.add_argument(
        "--probe-qty",
        type=int,
        default=100,
        help="Probe order quantity before lot-size normalization",
    )
    parser.add_argument(
        "--probe-price-factor",
        type=float,
        default=0.8,
        help="Probe order limit price as quote * factor (must be <1.0 for live)",
    )
    parser.add_argument(
        "--probe-max-notional",
        type=float,
        default=10000.0,
        help="Maximum notional allowed for probe order",
    )
    parser.add_argument(
        "--allow-live-orders",
        action="store_true",
        help="Required safety switch for probe orders in live mode",
    )
    args = parser.parse_args()

    if args.duration_minutes <= 0:
        raise SystemExit("--duration-minutes must be > 0")
    if args.poll_seconds <= 0:
        raise SystemExit("--poll-seconds must be > 0")

    mode = _mode_from_text(args.mode)
    if mode == TradingMode.LIVE and not args.allow_live:
        raise SystemExit("--allow-live is required when --mode live")
    if (
        mode == TradingMode.LIVE
        and args.place_probe_order
        and not args.allow_live_orders
    ):
        raise SystemExit(
            "--allow-live-orders is required for --place-probe-order in live mode"
        )
    if (
        mode == TradingMode.LIVE
        and args.place_probe_order
        and float(args.probe_price_factor) >= 1.0
    ):
        raise SystemExit(
            "--probe-price-factor must be < 1.0 for live probe orders "
            "(to keep orders far from market)"
        )

    symbols = _parse_symbols(args.symbols)
    if not symbols:
        raise SystemExit("No symbols available for quote probes")

    CONFIG.ensure_dirs()
    engine = ExecutionEngine(mode=mode)

    started_at = _utc_now_iso()
    started_monotonic = time.monotonic()
    deadline = started_monotonic + float(args.duration_minutes) * 60.0

    samples: list[dict[str, Any]] = []
    probe_events: list[dict[str, Any]] = []
    failure_ticks = 0
    disconnect_ticks = 0
    unhealthy_ticks = 0
    consecutive_failures = 0
    max_consecutive_failures = 0
    quote_probe_total = 0
    quote_probe_ok = 0
    fills_seen = 0
    tick_count = 0
    status = "pass"
    fail_reasons: list[str] = []
    last_fill_since = datetime.now() - timedelta(minutes=5)

    try:
        if not engine.start():
            raise RuntimeError("Execution engine failed to start")

        while time.monotonic() < deadline:
            tick_started = time.monotonic()
            tick_count += 1
            tick_ok = True
            tick_errors: list[str] = []

            sample: dict[str, Any] = {
                "tick": tick_count,
                "ts": _utc_now_iso(),
                "broker_connected": bool(engine.broker.is_connected),
            }

            if not sample["broker_connected"]:
                disconnect_ticks += 1
                tick_ok = False
                tick_errors.append("broker_disconnected")

            try:
                account = engine.broker.get_account()
                sample["equity"] = _safe_float(getattr(account, "equity", 0.0))
                sample["cash"] = _safe_float(getattr(account, "cash", 0.0))
                sample["position_count"] = int(
                    len(getattr(account, "positions", {}) or {})
                )
            except Exception as exc:
                tick_ok = False
                tick_errors.append(f"account_error:{exc}")

            try:
                active_orders = engine.broker.get_orders(active_only=True)
                sample["active_orders"] = int(len(active_orders or []))
            except Exception as exc:
                tick_ok = False
                tick_errors.append(f"orders_error:{exc}")

            try:
                fills = engine.broker.get_fills(since=last_fill_since)
                fills_seen += int(len(fills or []))
                last_fill_since = datetime.now() - timedelta(seconds=2)
                sample["fills_seen_total"] = int(fills_seen)
            except Exception as exc:
                tick_ok = False
                tick_errors.append(f"fills_error:{exc}")

            try:
                health = get_health_monitor().get_health()
                health_status = str(getattr(health.status, "value", health.status))
                sample["health_status"] = health_status
                sample["health_can_trade"] = bool(getattr(health, "can_trade", False))
                sample["health_degraded_mode"] = bool(
                    getattr(health, "degraded_mode", False)
                )
                if getattr(health, "status", None) in (
                    HealthStatus.UNHEALTHY,
                    HealthStatus.CRITICAL,
                ):
                    unhealthy_ticks += 1
                    tick_ok = False
                    tick_errors.append(f"health_{health_status}")
            except Exception as exc:
                tick_ok = False
                tick_errors.append(f"health_error:{exc}")

            q_total, q_ok = _quote_probe(engine.broker, symbols)
            quote_probe_total += q_total
            quote_probe_ok += q_ok
            sample["quote_probe_total"] = int(q_total)
            sample["quote_probe_ok"] = int(q_ok)
            if q_total > 0 and q_ok < q_total:
                tick_ok = False
                tick_errors.append("quote_probe_partial_failure")

            if args.place_probe_order and (tick_count % max(1, args.probe_every_ticks) == 0):
                probe_symbol = str(args.probe_symbol or symbols[0]).strip()
                try:
                    probe_result = _place_probe_order(
                        engine=engine,
                        symbol=probe_symbol,
                        qty=int(args.probe_qty),
                        price_factor=float(args.probe_price_factor),
                        max_notional=float(args.probe_max_notional),
                    )
                except Exception as exc:
                    probe_result = {"status": "error", "reason": str(exc)}
                    tick_ok = False
                    tick_errors.append(f"probe_order_error:{exc}")
                probe_result["tick"] = tick_count
                probe_result["ts"] = _utc_now_iso()
                probe_events.append(probe_result)
                sample["probe_order"] = probe_result

            if hasattr(engine, "_build_execution_snapshot"):
                try:
                    snapshot = engine._build_execution_snapshot()  # noqa: SLF001
                    runtime = snapshot.get("runtime", {})
                    sample["runtime_queue_depth"] = int(
                        (runtime or {}).get("queue_depth", 0)
                    )
                except Exception:
                    pass

            sample["errors"] = tick_errors
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
    finally:
        try:
            engine.stop()
        except Exception:
            pass

    ended_at = _utc_now_iso()
    quote_success_ratio = (
        float(quote_probe_ok) / float(quote_probe_total)
        if quote_probe_total > 0
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
    if disconnect_ticks > int(args.max_disconnect_ticks):
        status = "fail"
        fail_reasons.append(
            "disconnect_ticks="
            f"{disconnect_ticks} > max_disconnect_ticks={int(args.max_disconnect_ticks)}"
        )
    if unhealthy_ticks > int(args.max_unhealthy_ticks):
        status = "fail"
        fail_reasons.append(
            "unhealthy_ticks="
            f"{unhealthy_ticks} > max_unhealthy_ticks={int(args.max_unhealthy_ticks)}"
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
        "mode": mode.value,
        "started_at": started_at,
        "ended_at": ended_at,
        "duration_minutes_requested": float(args.duration_minutes),
        "tick_count": int(tick_count),
        "failure_ticks": int(failure_ticks),
        "disconnect_ticks": int(disconnect_ticks),
        "unhealthy_ticks": int(unhealthy_ticks),
        "max_consecutive_failures_observed": int(max_consecutive_failures),
        "quote_probe_total": int(quote_probe_total),
        "quote_probe_ok": int(quote_probe_ok),
        "quote_success_ratio": round(float(quote_success_ratio), 6),
        "fills_seen_total": int(fills_seen),
        "probe_events": probe_events[-50:],
        "fail_reasons": fail_reasons,
        "samples": samples,
    }

    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.output:
        _write_json(Path(args.output), report)
        print(f"soak report written: {args.output}")

    return 0 if status == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
