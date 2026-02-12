
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

def check_dependencies(require_gui: bool = False) -> bool:
    """Check required dependencies. Only require GUI libs if launching GUI."""
    required = [
        ("torch", "pytorch"),
        ("numpy", "numpy"),
        ("pandas", "pandas"),
        ("sklearn", "scikit-learn"),
        ("psutil", "psutil"),
    ]

    # GUI: PyQt6 is required; pyqtgraph is OPTIONAL (you already have fallback).
    optional = []
    if require_gui:
        required.append(("PyQt6", "PyQt6"))
        optional.append(("pyqtgraph", "pyqtgraph"))

    missing = []
    for module, package in required:
        try:
            __import__(module)
        except ImportError:
            missing.append(package)

    if missing:
        print(f"Missing required packages: {', '.join(missing)}")
        print(f"Install with: pip install {' '.join(missing)}")
        return False

    # Optional warnings
    for module, package in optional:
        try:
            __import__(module)
        except ImportError:
            print(f"Optional package missing: {package} (some UI charts may be simplified)")

    return True

def main():
    """Main entry point"""
    import argparse
    import os
    import threading

    parser = argparse.ArgumentParser(description='AI Stock Trading System')

    parser.add_argument('--train', action='store_true', help='Train model')
    parser.add_argument('--auto-learn', action='store_true', help='Auto-discover and train')
    parser.add_argument('--predict', type=str, help='Predict stock')
    parser.add_argument('--backtest', action='store_true', help='Run backtest')
    parser.add_argument('--replay-file', type=str, help='Replay market data file (csv/jsonl)')
    parser.add_argument('--replay-speed', type=float, default=20.0, help='Replay speed multiplier')
    parser.add_argument('--health', action='store_true', help='Show system health')
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--max-stocks', type=int, default=200, help='Max stocks for training')
    parser.add_argument('--continuous', action='store_true', help='Continuous learning mode')
    parser.add_argument('--cli', action='store_true', help='CLI mode')
    parser.add_argument('--recovery-drill', action='store_true', help='Run crash recovery drill')

    args = parser.parse_args()

    require_gui = not any([
        args.train, args.auto_learn, args.predict, args.backtest, args.replay_file,
        args.health, args.cli, args.recovery_drill,
    ])

    if not check_dependencies(require_gui=require_gui):
        sys.exit(1)

    from utils.logger import get_logger
    log = get_logger()

    log.info("=" * 60)
    log.info("AI STOCK TRADING SYSTEM - Production Grade")
    log.info("=" * 60)

    from core.events import EVENT_BUS
    EVENT_BUS.start()

    from utils.metrics import start_process_metrics
    start_process_metrics()

    # ---- NEW: optional Prometheus endpoint ----
    port = os.environ.get("TRADING_METRICS_PORT")
    if port:
        try:
            from utils.metrics_http import serve
            t = threading.Thread(target=serve, args=(int(port),), daemon=True, name="metrics_http")
            t.start()
            log.info(f"Metrics server started on :{port} (/metrics, /healthz)")
        except Exception as e:
            log.warning(f"Metrics server failed: {e}")

    from trading.kill_switch import get_kill_switch
    _ = get_kill_switch()

    try:
        if args.health:
            from trading.health import get_health_monitor
            monitor = get_health_monitor()
            print(monitor.get_health_json())

        elif args.train:
            from models.trainer import Trainer
            trainer = Trainer()
            trainer.train(epochs=args.epochs)

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
            result = predictor.predict(args.predict)
            print(f"\n{result.stock_code} - {result.stock_name}")
            print(f"Signal: {result.signal.value}")
            print(f"Confidence: {result.confidence:.0%}")
            print(f"Price: {result.current_price:.2f}")

        elif args.backtest:
            from analysis.backtest import Backtester
            bt = Backtester()
            result = bt.run()
            print(result.summary())

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
    except Exception as e:
        log.exception(f"Error: {e}")
        sys.exit(1)
    finally:
        EVENT_BUS.stop()
        from utils.security import get_audit_log
        get_audit_log().close()

def run_recovery_drill():
    """
    Recovery drill:
    1) Create isolated OMS DB in temp folder
    2) Submit an order, process a fill
    3) Simulate crash: drop OMS instance
    4) Re-open OMS from same DB and verify fills are not duplicated
    """
    import tempfile
    from pathlib import Path
    from trading.oms import get_oms, reset_oms
    from core.types import Order, OrderSide, OrderType, Fill

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

    assert len(fills_after) == len(fills_before), "Fill count changed after restart (dedup broken)"
    print("[DRILL] PASS: fill dedup + recovery OK")

if __name__ == "__main__":
    main()
