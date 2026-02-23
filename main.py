import argparse
import json
from importlib.util import find_spec
from pathlib import Path
from typing import Any


def _module_exists(module: str) -> bool:
    """Fast dependency probe without importing heavy modules."""
    try:
        return find_spec(module) is not None
    except Exception:
        return False


def check_dependencies(
    require_gui: bool = False,
    require_ml: bool = False,
    require_security: bool = True,
    require_live: bool = False,
) -> bool:
    """Check required dependencies.

    Startup optimization:
    - Use find_spec() instead of importing heavy modules.
    - Only require ML stack for ML-heavy modes.
    """
    required = [("psutil", "psutil")]
    if require_security:
        required.append(("cryptography", "cryptography"))
    if require_ml:
        required.extend([
            ("torch", "torch"),
            ("numpy", "numpy"),
            ("pandas", "pandas"),
            ("sklearn", "scikit-learn"),
        ])
    if require_live:
        required.append(("easytrader", "easytrader"))

    optional = []
    if require_gui:
        required.append(("PyQt6", "PyQt6"))
        # Optional chart accel lib; app already has fallback.
        optional.append(("pyqtgraph", "pyqtgraph"))

    missing = []
    for module, package in required:
        if not _module_exists(module):
            missing.append(package)

    if missing:
        print(f"Missing required packages: {', '.join(missing)}")
        print(f"Install with: pip install {' '.join(missing)}")
        return False

    # Optional warnings
    for module, package in optional:
        if not _module_exists(module):
            print(f"Optional package missing: {package} (some UI charts may be simplified)")

    return True

def _parse_positive_int_csv(raw: str, arg_name: str) -> list[int]:
    """Parse comma-separated positive integers with strict validation."""
    values: list[int] = []
    invalid_tokens: list[str] = []
    for piece in str(raw or "").split(","):
        token = piece.strip()
        if not token:
            continue
        try:
            value = int(token)
        except ValueError:
            invalid_tokens.append(token)
            continue
        if value <= 0:
            invalid_tokens.append(token)
            continue
        values.append(value)

    deduped = sorted(set(values))
    if deduped:
        if invalid_tokens:
            raise ValueError(
                f"{arg_name} contains invalid values: {', '.join(invalid_tokens)}"
            )
        return deduped

    raise ValueError(
        f"{arg_name} must contain at least one positive integer"
    )


def _parse_probability_csv(raw: str, arg_name: str) -> list[float]:
    """Parse comma-separated probabilities (0, 1] with strict validation."""
    values: list[float] = []
    invalid_tokens: list[str] = []
    for piece in str(raw or "").split(","):
        token = piece.strip()
        if not token:
            continue
        try:
            value = float(token)
        except ValueError:
            invalid_tokens.append(token)
            continue
        if not (0.0 < value <= 1.0):
            invalid_tokens.append(token)
            continue
        values.append(value)

    deduped = sorted(set(values))
    if deduped:
        if invalid_tokens:
            raise ValueError(
                f"{arg_name} contains invalid values: {', '.join(invalid_tokens)}"
            )
        return deduped

    raise ValueError(
        f"{arg_name} must contain at least one value in the range (0, 1]"
    )


def _parse_float_csv(
    raw: str,
    arg_name: str,
    *,
    min_value: float = 0.0,
    allow_equal_min: bool = True,
) -> list[float]:
    """Parse comma-separated floats with lower-bound validation."""
    values: list[float] = []
    invalid_tokens: list[str] = []
    for piece in str(raw or "").split(","):
        token = piece.strip()
        if not token:
            continue
        try:
            value = float(token)
        except ValueError:
            invalid_tokens.append(token)
            continue
        if allow_equal_min:
            ok = value >= float(min_value)
        else:
            ok = value > float(min_value)
        if not ok:
            invalid_tokens.append(token)
            continue
        values.append(value)

    deduped = sorted(set(values))
    if deduped:
        if invalid_tokens:
            raise ValueError(
                f"{arg_name} contains invalid values: {', '.join(invalid_tokens)}"
            )
        return deduped

    bound = (
        f">={min_value}" if allow_equal_min else f">{min_value}"
    )
    raise ValueError(
        f"{arg_name} must contain at least one value {bound}"
    )


def _require_positive_int(value: int, arg_name: str) -> int:
    """Validate that an integer argument is positive."""
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{arg_name} must be a positive integer")
    return parsed


def _ensure_backtest_optimize_success(summary: dict[str, Any]) -> None:
    """Raise if optimization did not produce a successful result."""
    status = str(summary.get("status", "")).strip().lower()
    if status in {"ok", "success"}:
        return

    errors = summary.get("errors")
    if isinstance(errors, list) and errors:
        raise RuntimeError(f"Backtest optimization failed: {errors[0]}")
    raise RuntimeError("Backtest optimization failed")


def _health_gate_violations(report: dict[str, Any]) -> list[str]:
    """Return health gate violations for production readiness checks."""
    violations: list[str] = []
    status = str(report.get("status", "")).strip().lower()
    if status != "healthy":
        violations.append(f"status={status or 'unknown'}")
    if not bool(report.get("can_trade", False)):
        violations.append("can_trade=false")
    if bool(report.get("degraded_mode", False)):
        violations.append("degraded_mode=true")
    if report.get("slo_pass") is False:
        violations.append("slo_pass=false")
    return violations


def _ensure_health_gate_from_json(raw_health_json: str) -> None:
    """Raise if health JSON does not pass production gate checks."""
    try:
        payload = json.loads(raw_health_json)
    except Exception as exc:
        raise RuntimeError(f"Health output is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("Health output JSON must be an object")

    violations = _health_gate_violations(payload)
    if violations:
        raise RuntimeError(f"Health gate failed: {', '.join(violations)}")


def _doctor_gate_violations(report: dict[str, Any]) -> list[str]:
    """Return doctor gate violations for production readiness checks."""
    violations: list[str] = []

    deps = report.get("dependencies")
    if isinstance(deps, dict):
        required_modules = ("psutil", "numpy", "pandas", "sklearn", "requests", "cryptography")
        missing = [name for name in required_modules if not bool(deps.get(name, False))]
        if missing:
            violations.append(f"missing_dependencies={','.join(missing)}")
    else:
        violations.append("dependencies_report_missing")

    paths = report.get("paths")
    if isinstance(paths, dict):
        bad_paths: list[str] = []
        for name, info in paths.items():
            if not isinstance(info, dict):
                bad_paths.append(f"{name}:invalid")
                continue
            if not bool(info.get("exists", False)):
                bad_paths.append(f"{name}:missing")
            elif not bool(info.get("writable", False)):
                bad_paths.append(f"{name}:readonly")
        if bad_paths:
            violations.append(f"path_issues={';'.join(bad_paths)}")
    else:
        violations.append("paths_report_missing")

    config_warnings = report.get("config_validation_warnings")
    if isinstance(config_warnings, list) and config_warnings:
        violations.append(f"config_warnings={len(config_warnings)}")
    elif config_warnings is None:
        violations.append("config_validation_warnings_missing")

    institutional = report.get("institutional_readiness")
    if isinstance(institutional, dict):
        if not bool(institutional.get("pass", False)):
            failed = institutional.get("failed_required_controls", [])
            failed_count = len(failed) if isinstance(failed, list) else 0
            violations.append(f"institutional_readiness_failed={failed_count}")
    else:
        violations.append("institutional_readiness_missing")

    live_readiness = report.get("live_readiness")
    if isinstance(live_readiness, dict):
        enforced = bool(live_readiness.get("enforced", False))
        if enforced and not bool(live_readiness.get("pass", False)):
            missing = live_readiness.get("missing_dependencies", [])
            if isinstance(missing, list) and missing:
                violations.append(
                    f"live_missing_dependencies={','.join(str(x) for x in missing)}"
                )
            if live_readiness.get("broker_path_exists") is False:
                violations.append("live_broker_path_missing")
    elif report.get("doctor_live_enforced") is True:
        violations.append("live_readiness_missing")

    return violations


def _ensure_doctor_gate(report: dict[str, Any]) -> None:
    """Raise if doctor report fails production readiness gate checks."""
    violations = _doctor_gate_violations(report)
    if violations:
        raise RuntimeError(f"Doctor gate failed: {', '.join(violations)}")


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(description='AI Stock Trading System')

    parser.add_argument('--train', action='store_true', help='Train model')
    parser.add_argument('--train-stock', type=str, help='Train model on a specific stock (e.g., 600519)')
    parser.add_argument('--auto-learn', action='store_true', help='Auto-discover and train')
    parser.add_argument('--predict', type=str, help='Predict stock')
    parser.add_argument('--backtest', action='store_true', help='Run backtest')
    parser.add_argument('--backtest-optimize', action='store_true', help='Optimize backtest parameters')
    parser.add_argument('--replay-file', type=str, help='Replay market data file (csv/jsonl)')
    parser.add_argument('--replay-speed', type=float, default=20.0, help='Replay speed multiplier')
    parser.add_argument('--health', action='store_true', help='Show system health')
    parser.add_argument('--health-strict', action='store_true', help='Fail when health gate is not fully healthy (use with --health)')
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--max-stocks', type=int, default=200, help='Max stocks for training')
    parser.add_argument('--continuous', action='store_true', help='Continuous learning mode')
    parser.add_argument('--cli', action='store_true', help='CLI mode')
    parser.add_argument('--recovery-drill', action='store_true', help='Run crash recovery drill')
    parser.add_argument('--doctor', action='store_true', help='Run system diagnostics')
    parser.add_argument('--doctor-strict', action='store_true', help='Fail when doctor readiness gate is not met (use with --doctor)')
    parser.add_argument('--doctor-live', action='store_true', help='Enforce live-trading dependency/path checks (use with --doctor)')
    parser.add_argument('--opt-train-months', type=str, default='6,9,12,18', help='Backtest optimization train months list')
    parser.add_argument('--opt-test-months', type=str, default='1,2,3', help='Backtest optimization test months list')
    parser.add_argument('--opt-min-confidence', type=str, default='0.55,0.60,0.65,0.70', help='Backtest optimization confidence list')
    parser.add_argument('--opt-trade-horizon', type=str, default='3,5,8', help='Backtest optimization holding horizon (bars) list')
    parser.add_argument('--opt-max-participation', type=str, default='0.02,0.03,0.05', help='Backtest optimization max volume participation list')
    parser.add_argument('--opt-slippage-bps', type=str, default='8,12,18', help='Backtest optimization slippage assumptions in bps')
    parser.add_argument('--opt-commission-bps', type=str, default='2.0,2.5,3.0', help='Backtest optimization commission assumptions in bps')
    parser.add_argument('--opt-top-k', type=int, default=5, help='Backtest optimization top-k results')

    args = parser.parse_args()

    if args.health_strict and not args.health:
        parser.error("--health-strict requires --health")
    if args.doctor_strict and not args.doctor:
        parser.error("--doctor-strict requires --doctor")
    if args.doctor_live and not args.doctor:
        parser.error("--doctor-live requires --doctor")

    require_gui = not any([
        args.train, args.auto_learn, args.predict, args.backtest, args.backtest_optimize, args.replay_file,
        args.health, args.cli, args.recovery_drill, args.doctor,
    ])
    require_ml = any([
        args.train, args.auto_learn, args.predict, args.backtest, args.backtest_optimize, args.replay_file,
    ])

    from config.settings import CONFIG

    config_live_mode = str(getattr(getattr(CONFIG, "trading_mode", None), "value", "")).lower() == "live"
    require_live = bool((require_gui and config_live_mode) or args.doctor_live)

    if not check_dependencies(
        require_gui=require_gui,
        require_ml=require_ml,
        require_security=True,
        require_live=require_live,
    ):
        return 1

    from utils.logger import get_logger
    log = get_logger()

    log.info("=" * 60)
    log.info("AI STOCK TRADING SYSTEM - Production Grade")
    log.info("=" * 60)

    from core.events import EVENT_BUS
    EVENT_BUS.start()

    from utils.metrics import start_process_metrics
    start_process_metrics()

    metrics_server = None
    # Optional metrics + local API endpoint
    from config.runtime_env import env_text

    port = env_text("TRADING_METRICS_PORT", "").strip()
    if port:
        try:
            from utils.metrics_http import serve
            host = env_text("TRADING_METRICS_HOST", "127.0.0.1").strip() or "127.0.0.1"
            metrics_server = serve(port=int(port), host=host)
            log.info(
                "Metrics/API server started at %s "
                "(/metrics, /healthz, /api/v1/dashboard)",
                metrics_server.url,
            )
        except Exception as e:
            log.warning(f"Metrics server failed: {e}")

    from trading.kill_switch import get_kill_switch
    _ = get_kill_switch()

    exit_code = 0
    try:
        if args.health:
            from trading.health import get_health_monitor
            monitor = get_health_monitor()
            health_json = monitor.get_health_json()
            print(health_json)
            if args.health_strict:
                _ensure_health_gate_from_json(health_json)

        elif args.doctor:
            report = run_system_doctor(check_live=bool(args.doctor_live or config_live_mode))
            if args.doctor_strict:
                _ensure_doctor_gate(report)

        elif args.train:
            from models.trainer import Trainer
            trainer = Trainer()
            trainer.train(epochs=args.epochs)

        elif args.train_stock:
            from data.fetcher import DataFetcher
            from models.trainer import Trainer

            # Validate stock code
            stock_code = DataFetcher.clean_code(args.train_stock)
            if not stock_code:
                print(f"Error: Invalid stock code '{args.train_stock}'")
                return 1
            
            is_valid, error = DataFetcher.validate_stock_code(stock_code)
            if not is_valid:
                print(f"Error: {error}")
                return 1
            
            print(f"Training on specific stock: {stock_code}")
            trainer = Trainer()
            trainer.train(
                stock_codes=[stock_code],
                epochs=args.epochs,
                interval="1m",
            )
            print(f"Training completed for {stock_code}")

        elif args.auto_learn:
            from models.auto_learner import AutoLearner
            learner = AutoLearner()
            learner.run(
                max_stocks=args.max_stocks,
                epochs_per_cycle=args.epochs,
                continuous=args.continuous,
            )

        elif args.recovery_drill:
            run_recovery_drill()

        elif args.predict:
            from models.predictor import Predictor
            predictor = Predictor()
            prediction_result = predictor.predict(args.predict)
            print(f"\n{prediction_result.stock_code} - {prediction_result.stock_name}")
            print(f"Signal: {prediction_result.signal.value}")
            print(f"Confidence: {prediction_result.confidence:.0%}")
            print(f"Price: {prediction_result.current_price:.2f}")

        elif args.backtest_optimize:
            train_months_options = _parse_positive_int_csv(
                args.opt_train_months,
                "--opt-train-months",
            )
            test_months_options = _parse_positive_int_csv(
                args.opt_test_months,
                "--opt-test-months",
            )
            min_confidence_options = _parse_probability_csv(
                args.opt_min_confidence,
                "--opt-min-confidence",
            )
            trade_horizon_options = _parse_positive_int_csv(
                args.opt_trade_horizon,
                "--opt-trade-horizon",
            )
            max_participation_options = _parse_float_csv(
                args.opt_max_participation,
                "--opt-max-participation",
                min_value=0.0,
                allow_equal_min=False,
            )
            slippage_bps_options = _parse_float_csv(
                args.opt_slippage_bps,
                "--opt-slippage-bps",
                min_value=0.0,
                allow_equal_min=True,
            )
            commission_bps_options = _parse_float_csv(
                args.opt_commission_bps,
                "--opt-commission-bps",
                min_value=0.0,
                allow_equal_min=True,
            )
            top_k = _require_positive_int(args.opt_top_k, "--opt-top-k")

            from analysis.backtest import Backtester
            bt = Backtester()
            summary = bt.optimize(
                train_months_options=train_months_options,
                test_months_options=test_months_options,
                min_confidence_options=min_confidence_options,
                trade_horizon_options=trade_horizon_options,
                max_participation_options=max_participation_options,
                slippage_bps_options=slippage_bps_options,
                commission_bps_options=commission_bps_options,
                top_k=top_k,
            )
            print(json.dumps(summary, indent=2, ensure_ascii=False))
            _ensure_backtest_optimize_success(summary)

        elif args.backtest:
            from analysis.backtest import Backtester
            bt = Backtester()
            backtest_result = bt.run()
            print(backtest_result.summary())

        elif args.replay_file:
            from analysis.replay import MarketReplay
            replay = MarketReplay.from_file(Path(args.replay_file))
            print(f"[REPLAY] Loaded {len(replay)} bars from {args.replay_file}")
            seen = 0
            symbols = set()
            for bar in replay.play(speed=max(0.1, float(args.replay_speed))):
                seen += 1
                symbols.add(bar.symbol)
                if seen % 500 == 0:
                    print(
                        f"[REPLAY] {seen} bars | symbols={len(symbols)} "
                        f"| last={bar.symbol} {bar.ts.isoformat()} close={bar.close:.2f}"
                    )
            print(f"[REPLAY] Done. bars={seen}, symbols={len(symbols)}")

        else:
            from ui.app import run_app
            run_app()

    except KeyboardInterrupt:
        log.info("Interrupted by user")
        exit_code = 130
    except Exception as e:
        log.exception(f"Error: {e}")
        exit_code = 1
    finally:
        if metrics_server is not None:
            try:
                metrics_server.stop()
            except Exception as e:
                log.warning(f"Failed to stop metrics server cleanly: {e}")
        EVENT_BUS.stop()
        from utils.security import get_audit_log
        get_audit_log().close()
    return exit_code

def run_recovery_drill() -> None:
    """Recovery drill:
    1) Create isolated OMS DB in temp folder
    2) Submit an order, process a fill
    3) Simulate crash: drop OMS instance
    4) Re-open OMS from same DB and verify fills are not duplicated.
    """
    import tempfile
    from pathlib import Path

    from core.types import Fill, Order, OrderSide, OrderType
    from trading.oms import get_oms, reset_oms

    tmpdir = Path(tempfile.mkdtemp(prefix="recovery_drill_"))
    db_path = tmpdir / "orders_drill.db"

    print(f"[DRILL] Using temp OMS db: {db_path}")

    reset_oms()
    oms = get_oms(initial_capital=100000, db_path=db_path)

    order = Order(symbol="600519", side=OrderSide.BUY, order_type=OrderType.LIMIT, quantity=100, price=100.0)
    oms.submit_order(order)

    fill = Fill(order_id=order.id, symbol=order.symbol, side=OrderSide.BUY, quantity=100, price=100.0, commission=5.0)
    oms.process_fill(order, fill)

    fills_before = oms.get_fills(order.id)
    print(f"[DRILL] fills before restart: {len(fills_before)}")

    # simulate crash/restart
    reset_oms()
    oms2 = get_oms(initial_capital=100000, db_path=db_path)

    fills_after = oms2.get_fills(order.id)
    print(f"[DRILL] fills after restart: {len(fills_after)}")

    # Use explicit exception instead of assert (which can be optimized away)
    if len(fills_after) != len(fills_before):
        raise RuntimeError(
            f"Fill count changed after restart: before={len(fills_before)}, "
            f"after={len(fills_after)} (dedup broken)"
        )
    print("[DRILL] PASS: fill dedup + recovery OK")


def run_system_doctor(*, check_live: bool = False) -> dict[str, Any]:
    """One-shot system diagnostics for setup and runtime readiness."""
    import os
    from datetime import datetime

    from config.settings import CONFIG
    from utils.institutional import collect_institutional_readiness

    modules = [
        "psutil",
        "numpy",
        "pandas",
        "sklearn",
        "torch",
        "akshare",
        "yfinance",
        "PyQt6",
        "websocket",
        "requests",
        "cryptography",
        "easytrader",
    ]
    deps = {m: bool(_module_exists(m)) for m in modules}

    CONFIG.ensure_dirs()
    paths = {
        "data_dir": CONFIG.data_dir,
        "model_dir": CONFIG.model_dir,
        "log_dir": CONFIG.log_dir,
        "cache_dir": CONFIG.cache_dir,
        "audit_dir": CONFIG.audit_dir,
    }
    path_report = {}
    for name, p in paths.items():
        path_report[name] = {
            "path": str(p),
            "exists": p.exists(),
            "writable": os.access(str(p), os.W_OK) if p.exists() else False,
        }

    model_dir = CONFIG.model_dir

    broker_path = str(getattr(CONFIG, "broker_path", "") or "").strip()
    trading_mode = str(getattr(getattr(CONFIG, "trading_mode", None), "value", "")).lower()
    doctor_live_enforced = bool(check_live or trading_mode == "live")
    live_missing_deps = [
        name for name in ("easytrader", "cryptography") if not bool(deps.get(name, False))
    ]
    broker_path_exists = bool(broker_path and Path(broker_path).exists())
    live_readiness = {
        "enforced": doctor_live_enforced,
        "trading_mode": trading_mode or "unknown",
        "missing_dependencies": live_missing_deps,
        "broker_path": broker_path,
        "broker_path_exists": broker_path_exists,
        "pass": (len(live_missing_deps) == 0 and broker_path_exists),
    }

    report: dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "doctor_live_enforced": doctor_live_enforced,
        "dependencies": deps,
        "paths": path_report,
        "live_readiness": live_readiness,
        "models": {
            "ensembles": len(list(model_dir.glob("ensemble_*.pt"))),
            "forecasters": len(list(model_dir.glob("forecast_*.pt"))),
            "scalers": len(list(model_dir.glob("scaler_*.pkl"))),
        },
        "config_validation_warnings": list(CONFIG.validation_warnings),
        "institutional_readiness": collect_institutional_readiness(),
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return report

if __name__ == "__main__":
    raise SystemExit(main())
