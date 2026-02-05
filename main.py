#!/usr/bin/env python3
"""
AI Stock Trading System - Production Grade
"""
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def check_dependencies() -> bool:
    """Check required dependencies"""
    required = [
        ('torch', 'pytorch'),
        ('numpy', 'numpy'),
        ('pandas', 'pandas'),
        ('PyQt6', 'PyQt6'),
        ('sklearn', 'scikit-learn'),
        ('psutil', 'psutil'),
    ]
    
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
    
    return True


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='AI Stock Trading System')
    
    parser.add_argument('--train', action='store_true', help='Train model')
    parser.add_argument('--auto-learn', action='store_true', help='Auto-discover and train')
    parser.add_argument('--predict', type=str, help='Predict stock')
    parser.add_argument('--backtest', action='store_true', help='Run backtest')
    parser.add_argument('--health', action='store_true', help='Show system health')
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--max-stocks', type=int, default=200, help='Max stocks for training')
    parser.add_argument('--continuous', action='store_true', help='Continuous learning mode')
    parser.add_argument('--cli', action='store_true', help='CLI mode')
    
    args = parser.parse_args()
    
    if not check_dependencies():
        sys.exit(1)
    
    # Initialize logging
    from utils.logger import get_logger
    log = get_logger()
    
    log.info("=" * 60)
    log.info("AI STOCK TRADING SYSTEM - Production Grade")
    log.info("=" * 60)
    
    # Initialize event bus
    from core.events import EVENT_BUS
    EVENT_BUS.start()
    
    # Initialize kill switch
    from trading.kill_switch import get_kill_switch
    kill_switch = get_kill_switch()
    
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
                epochs_per_cycle=args.epochs,  # FIXED: correct parameter name
                continuous=args.continuous,
                search_all=True
            )
            
        elif args.predict:
            from models.predictor import Predictor
            predictor = Predictor()
            result = predictor.predict(args.predict)
            print(f"\n{result.stock_code} - {result.stock_name}")
            print(f"Signal: {result.signal.value}")
            print(f"Confidence: {result.confidence:.0%}")
            print(f"Price: {result.current_price:.2f}")  # FIXED: removed ¥ for console safety
            
        elif args.backtest:
            from analysis.backtest import Backtester
            bt = Backtester()
            result = bt.run()
            print(result.summary())
            
        else:
            # Start GUI
            from ui.app import run_app
            run_app()
            
    except KeyboardInterrupt:
        log.info("Interrupted by user")
    except Exception as e:
        log.exception(f"Error: {e}")
        sys.exit(1)
    finally:
        EVENT_BUS.stop()
        
        # Close audit log
        from utils.security import get_audit_log
        get_audit_log().close()


if __name__ == "__main__":
    main()