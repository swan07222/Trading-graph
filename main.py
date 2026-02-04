#!/usr/bin/env python3
"""
AI Stock Trading Advisor
Main Entry Point

Usage:
    python main.py              # Start GUI
    python main.py --train      # Train model
    python main.py --predict 600519  # Predict stock
    python main.py --check      # Check dependencies
"""
import sys
import argparse
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))


def check_dependencies() -> bool:
    """Check required packages"""
    required = [
        ('torch', 'torch'),
        ('numpy', 'numpy'),
        ('pandas', 'pandas'),
        ('PyQt6', 'PyQt6'),
        ('pyqtgraph', 'pyqtgraph'),
        ('akshare', 'akshare'),
        ('ta', 'ta'),
        ('sklearn', 'scikit-learn'),
        ('loguru', 'loguru'),
        ('tqdm', 'tqdm'),
    ]
    
    optional = [
        ('easytrader', 'easytrader'),
        ('transformers', 'transformers'),
    ]
    
    missing = []
    for module, package in required:
        try:
            __import__(module)
        except ImportError:
            missing.append(package)
    
    if missing:
        print("=" * 60)
        print("❌ Missing required dependencies:")
        print("=" * 60)
        for pkg in missing:
            print(f"   - {pkg}")
        print(f"\nInstall with: pip install {' '.join(missing)}")
        print("=" * 60)
        return False
    
    # Check optional
    opt_missing = []
    for module, package in optional:
        try:
            __import__(module)
        except ImportError:
            opt_missing.append(package)
    
    if opt_missing:
        print("⚠️  Optional dependencies (some features limited):")
        for pkg in opt_missing:
            print(f"   - {pkg}")
        print()
    
    return True


def print_banner():
    """Print application banner"""
    print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║                                                               ║
    ║           🤖 AI STOCK TRADING ADVISOR                        ║
    ║           ────────────────────────────                        ║
    ║                                                               ║
    ║           ✅ Custom AI Models (LSTM, Transformer, TCN)       ║
    ║           ✅ Ensemble Predictions                             ║
    ║           ✅ Real Broker Integration                          ║
    ║           ✅ Risk Management                                  ║
    ║           ✅ Professional GUI                                 ║
    ║                                                               ║
    ╚═══════════════════════════════════════════════════════════════╝
    """)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='AI Stock Trading Advisor',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python main.py              # Start GUI
    python main.py --train      # Train model
    python main.py --predict 600519
    python main.py --scan       # Scan for signals
        """
    )
    
    parser.add_argument('--check', action='store_true', help='Check dependencies')
    parser.add_argument('--train', action='store_true', help='Train model')
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--predict', type=str, help='Predict stock code')
    parser.add_argument('--scan', action='store_true', help='Scan for signals')
    parser.add_argument('--cli', action='store_true', help='CLI mode (no GUI)')
    
    args = parser.parse_args()
    
    print_banner()
    
    # Check dependencies
    print("Checking dependencies...")
    if not check_dependencies():
        sys.exit(1)
    print("✅ All dependencies OK!\n")
    
    if args.check:
        return
    
    # Train model
    if args.train:
        print("=" * 60)
        print("Starting Model Training")
        print("=" * 60)
        
        from models.trainer import Trainer
        
        trainer = Trainer()
        results = trainer.train(epochs=args.epochs)
        
        print("\n" + "=" * 60)
        print(f"Training Complete!")
        print(f"Best Accuracy: {results['best_accuracy']:.2%}")
        
        if 'test_metrics' in results and 'trading' in results['test_metrics']:
            tm = results['test_metrics']['trading']
            print(f"\nTrading Simulation:")
            print(f"  Return: {tm.get('total_return', 0):+.2f}%")
            print(f"  Win Rate: {tm.get('win_rate', 0):.1%}")
            print(f"  Sharpe Ratio: {tm.get('sharpe_ratio', 0):.2f}")
            print(f"  Max Drawdown: {tm.get('max_drawdown', 0):.1%}")
        
        print("=" * 60)
        return
    
    # Predict single stock
    if args.predict:
        from models.predictor import Predictor
        
        print(f"Analyzing {args.predict}...")
        
        try:
            predictor = Predictor()
            pred = predictor.predict(args.predict)
            
            print("\n" + "=" * 60)
            print(f"  {pred.stock_code} - {pred.stock_name}")
            print("=" * 60)
            print(f"  Signal: {pred.signal.value}")
            print(f"  Strength: {pred.signal_strength:.0%}")
            print(f"  Confidence: {pred.confidence:.0%}")
            print(f"  Model Agreement: {pred.model_agreement:.0%}")
            print()
            print(f"  Probabilities:")
            print(f"    UP:      {pred.prob_up:.1%}")
            print(f"    NEUTRAL: {pred.prob_neutral:.1%}")
            print(f"    DOWN:    {pred.prob_down:.1%}")
            print()
            print(f"  Price: ¥{pred.current_price:.2f}")
            print(f"  Stop Loss: ¥{pred.levels.stop_loss:.2f} ({pred.levels.stop_loss_pct:+.1f}%)")
            print(f"  Target 1: ¥{pred.levels.target_1:.2f} ({pred.levels.target_1_pct:+.1f}%)")
            print(f"  Target 2: ¥{pred.levels.target_2:.2f} ({pred.levels.target_2_pct:+.1f}%)")
            print()
            print(f"  Position: {pred.position.shares} shares (¥{pred.position.value:,.2f})")
            print()
            print("  Analysis:")
            for reason in pred.reasons:
                print(f"    • {reason}")
            
            if pred.warnings:
                print()
                print("  ⚠️ Warnings:")
                for warning in pred.warnings:
                    print(f"    • {warning}")
            
            print("=" * 60)
            
        except Exception as e:
            print(f"❌ Error: {e}")
            sys.exit(1)
        
        return
    
    # Scan stocks
    if args.scan:
        from models.predictor import Predictor
        from config import CONFIG
        
        print("Scanning stocks for signals...")
        
        try:
            predictor = Predictor()
            picks = predictor.get_top_picks(CONFIG.STOCK_POOL, n=10, signal_type="buy")
            
            print("\n" + "=" * 60)
            print("  TOP BUY SIGNALS")
            print("=" * 60)
            
            if picks:
                for i, pred in enumerate(picks, 1):
                    print(f"\n  {i}. {pred.stock_code} - {pred.stock_name}")
                    print(f"     Signal: {pred.signal.value} | Confidence: {pred.confidence:.0%}")
                    print(f"     Price: ¥{pred.current_price:.2f} | Target: ¥{pred.levels.target_2:.2f}")
            else:
                print("\n  No buy signals found.")
            
            print("\n" + "=" * 60)
            
        except Exception as e:
            print(f"❌ Error: {e}")
            sys.exit(1)
        
        return
    
    # Start GUI
    print("Starting GUI...")
    
    try:
        from ui.app import run_app
        run_app()
    except KeyboardInterrupt:
        print("\n\nExited.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()