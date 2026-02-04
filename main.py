#!/usr/bin/env python3
"""
AI Stock Trading System
Professional Trading Application with Custom AI Model

⚠️ WARNING: This system can trade with REAL MONEY.
Please read all documentation before using.

Usage:
    python main.py              # Start GUI
    python main.py --train      # Train AI model
    python main.py --auto-learn # Auto search and train
    python main.py --predict 600519
    python main.py --live       # Enable live trading
"""
import sys
import argparse
from pathlib import Path

# Add project root to path
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
        ('requests', 'requests'),
        ('bs4', 'beautifulsoup4'),
    ]
    
    optional = [
        ('easytrader', 'easytrader'),
        ('transformers', 'transformers'),
        ('playwright', 'playwright'),
    ]
    
    missing = []
    for module, package in required:
        try:
            __import__(module)
        except ImportError:
            missing.append(package)
    
    if missing:
        print("=" * 70)
        print("❌ MISSING REQUIRED DEPENDENCIES")
        print("=" * 70)
        for pkg in missing:
            print(f"   ✗ {pkg}")
        print(f"\nInstall with: pip install {' '.join(missing)}")
        print("=" * 70)
        return False
    
    # Check optional
    opt_missing = []
    for module, package in optional:
        try:
            __import__(module)
        except ImportError:
            opt_missing.append(package)
    
    if opt_missing:
        print(f"⚠️  Optional packages (some features limited): {', '.join(opt_missing)}")
    
    return True


def print_banner():
    """Print application banner"""
    print("""
    ╔══════════════════════════════════════════════════════════════════════╗
    ║                                                                      ║
    ║      █████╗ ██╗    ████████╗██████╗  █████╗ ██████╗ ███████╗        ║
    ║     ██╔══██╗██║    ╚══██╔══╝██╔══██╗██╔══██╗██╔══██╗██╔════╝        ║
    ║     ███████║██║       ██║   ██████╔╝███████║██║  ██║█████╗          ║
    ║     ██╔══██║██║       ██║   ██╔══██╗██╔══██║██║  ██║██╔══╝          ║
    ║     ██║  ██║██║       ██║   ██║  ██║██║  ██║██████╔╝███████╗        ║
    ║     ╚═╝  ╚═╝╚═╝       ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝╚═════╝ ╚══════╝        ║
    ║                                                                      ║
    ║              AI STOCK TRADING SYSTEM v2.0                            ║
    ║                                                                      ║
    ║    ✅ Custom AI Model (6 Neural Networks Ensemble)                   ║
    ║    ✅ Real-time Signal Monitoring                                    ║
    ║    ✅ Automatic Stock Discovery                                      ��
    ║    ✅ Professional Risk Management                                   ║
    ║    ✅ Paper & Live Trading Support                                   ║
    ║                                                                      ║
    ╚══════════════════════════════════════════════════════════════════════╝
    """)


def print_risk_warning():
    """Print risk warning"""
    print("""
    ╔══════════════════════════════════════════════════════════════════════╗
    ║                        ⚠️  RISK WARNING  ⚠️                          ║
    ╠══════════════════════════════════════════════════════════════════════╣
    ║                                                                      ║
    ║  1. Stock trading involves substantial risk of loss                  ║
    ║                                                                      ║
    ║  2. AI predictions are for reference only, not financial advice      ║
    ║                                                                      ║
    ║  3. Past performance does not guarantee future results               ║
    ║                                                                      ║
    ║  4. Never invest money you cannot afford to lose                     ║
    ║                                                                      ║
    ║  5. Practice with paper trading for at least 3 months first          ║
    ║                                                                      ║
    ║  By continuing, you acknowledge and accept these risks               ║
    ║                                                                      ║
    ╚══════════════════════════════════════════════════════════════════════╝
    """)


def print_model_info():
    """Print AI model information"""
    print("""
    ┌──────────────────────────────────────────────────────────────────────┐
    │                    🧠 AI MODEL ARCHITECTURE                          │
    ├──────────────────────────────────────────────────────────────────────┤
    │                                                                      │
    │  The system uses an ensemble of 6 neural networks:                   │
    │                                                                      │
    │  1. LSTM with Multi-Head Attention                                   │
    │     - Bidirectional LSTM for temporal patterns                       │
    │     - Self-attention for important features                          │
    │                                                                      │
    │  2. Transformer Encoder                                              │
    │     - Positional encoding for sequence order                         │
    │     - Multi-head self-attention mechanism                            │
    │                                                                      │
    │  3. GRU (Gated Recurrent Unit)                                       │
    │     - Lightweight recurrent network                                  │
    │     - Attention-based pooling                                        │
    │                                                                      │
    │  4. TCN (Temporal Convolutional Network)                             │
    │     - Dilated causal convolutions                                    │
    │     - Long-range dependency capture                                  │
    │                                                                      │
    │  5. Hybrid CNN-LSTM                                                  │
    │     - CNN for local pattern extraction                               │
    │     - LSTM for sequential modeling                                   │
    │                                                                      │
    │  6. Mamba State Space Model (Advanced)                               │
    │     - Linear time complexity                                         │
    │     - State-of-the-art for sequences                                 │
    │                                                                      │
    │  Ensemble combines predictions with learned weights                  │
    │  based on validation performance                                     │
    │                                                                      │
    └──────────────────────────────────────────────────────────────────────┘
    """)


def train_model(epochs: int, stocks: list = None):
    """Train the AI model"""
    from models.trainer import Trainer
    from config import CONFIG
    
    print("\n" + "=" * 70)
    print("                    TRAINING AI MODEL")
    print("=" * 70)
    
    print_model_info()
    
    print(f"\nConfiguration:")
    print(f"  • Epochs: {epochs}")
    print(f"  • Sequence Length: {CONFIG.SEQUENCE_LENGTH} days")
    print(f"  • Hidden Size: {CONFIG.HIDDEN_SIZE}")
    print(f"  • Models: LSTM, Transformer, GRU, TCN, Hybrid")
    print(f"  • Stocks: {len(stocks or CONFIG.STOCK_POOL)}")
    
    trainer = Trainer()
    
    def progress_callback(model_name, epoch, val_acc):
        print(f"\r  [{model_name}] Epoch {epoch+1}: accuracy = {val_acc:.2%}", end="", flush=True)
    
    results = trainer.train(
        stock_codes=stocks,
        epochs=epochs,
        callback=progress_callback
    )
    
    print("\n\n" + "=" * 70)
    print("                    TRAINING COMPLETE")
    print("=" * 70)
    print(f"\n  Best Validation Accuracy: {results['best_accuracy']:.2%}")
    
    if 'test_metrics' in results:
        tm = results['test_metrics']
        print(f"\n  Test Results:")
        print(f"    • Accuracy: {tm.get('accuracy', 0):.2%}")
        
        if 'trading' in tm:
            trading = tm['trading']
            print(f"\n  Simulated Trading Performance:")
            print(f"    • Total Return: {trading.get('total_return', 0):+.2f}%")
            print(f"    • Buy & Hold Return: {trading.get('buyhold_return', 0):+.2f}%")
            print(f"    • Excess Return: {trading.get('excess_return', 0):+.2f}%")
            print(f"    • Win Rate: {trading.get('win_rate', 0):.1%}")
            print(f"    • Profit Factor: {trading.get('profit_factor', 0):.2f}")
            print(f"    • Sharpe Ratio: {trading.get('sharpe_ratio', 0):.2f}")
            print(f"    • Max Drawdown: {trading.get('max_drawdown', 0):.1%}")
    
    print("\n" + "=" * 70)
    print("  Model saved to: saved_models/ensemble.pt")
    print("=" * 70 + "\n")


def auto_learn(epochs: int, max_stocks: int):
    """Auto-learn: search internet and train"""
    from models.auto_learner import AutoLearner
    
    print("\n" + "=" * 70)
    print("                    AUTO-LEARNING MODE")
    print("=" * 70)
    print("""
    The system will automatically:
    1. Search the internet for trending stocks
    2. Download historical data
    3. Create technical features
    4. Train the AI model
    5. Save the best model
    """)
    
    learner = AutoLearner()
    
    def progress_callback(progress):
        stage_names = {
            'idle': '⏸️  Idle',
            'searching': '🔍 Searching Internet',
            'downloading': '📥 Downloading Data',
            'preparing': '🔧 Preparing Features',
            'training': '🧠 Training AI Model',
            'evaluating': '📊 Evaluating',
            'complete': '✅ Complete',
            'error': '❌ Error'
        }
        stage = stage_names.get(progress.stage, progress.stage)
        print(f"\r  {stage}: {progress.message} ({progress.progress:.0f}%)     ", end="", flush=True)
    
    learner.add_callback(progress_callback)
    
    print("\n  Starting auto-learning process...")
    print("  This may take 30-60 minutes.\n")
    
    learner.start_learning(
        auto_search=True,
        max_stocks=max_stocks,
        epochs=epochs,
        incremental=True
    )
    
    # Wait for completion
    import time
    while learner.progress.is_running:
        time.sleep(1)
    
    print("\n\n" + "=" * 70)
    
    if learner.progress.stage == 'complete':
        print("                    AUTO-LEARNING COMPLETE")
        print("=" * 70)
        print(f"\n  Final Accuracy: {learner.progress.training_accuracy:.2%}")
        print(f"  Stocks Processed: {learner.progress.stocks_processed}")
        print(f"  Model saved successfully")
    else:
        print("                    AUTO-LEARNING FAILED")
        print("=" * 70)
        for error in learner.progress.errors:
            print(f"  ❌ {error}")
    
    print("\n" + "=" * 70 + "\n")


def predict_stock(code: str):
    """Predict single stock"""
    from models.predictor import Predictor
    
    print(f"\n  Analyzing {code}...")
    
    try:
        predictor = Predictor()
        
        if predictor.ensemble is None:
            print("\n  ❌ No trained model found.")
            print("     Run 'python main.py --train' first.")
            return
        
        pred = predictor.predict(code)
        
        # Signal colors for terminal
        signal_indicators = {
            'STRONG BUY': '🟢🟢',
            'BUY': '🟢',
            'HOLD': '🟡',
            'SELL': '🔴',
            'STRONG SELL': '🔴🔴',
        }
        
        indicator = signal_indicators.get(pred.signal.value, '⚪')
        
        print("\n" + "=" * 70)
        print(f"  {pred.stock_code} - {pred.stock_name}")
        print("=" * 70)
        
        print(f"\n  {indicator} Signal: {pred.signal.value}")
        print(f"     Confidence: {pred.confidence:.0%}")
        print(f"     Model Agreement: {pred.model_agreement:.0%}")
        
        print(f"\n  📊 AI Predictions:")
        print(f"     UP:      {pred.prob_up:.1%} {'█' * int(pred.prob_up * 20)}")
        print(f"     NEUTRAL: {pred.prob_neutral:.1%} {'█' * int(pred.prob_neutral * 20)}")
        print(f"     DOWN:    {pred.prob_down:.1%} {'█' * int(pred.prob_down * 20)}")
        
        print(f"\n  💰 Current Price: ¥{pred.current_price:.2f}")
        
        print(f"\n  📈 Trading Plan:")
        print(f"     Entry:     ¥{pred.levels.entry:.2f}")
        print(f"     Stop Loss: ¥{pred.levels.stop_loss:.2f} ({pred.levels.stop_loss_pct:+.1f}%)")
        print(f"     Target 1:  ¥{pred.levels.target_1:.2f} ({pred.levels.target_1_pct:+.1f}%)")
        print(f"     Target 2:  ¥{pred.levels.target_2:.2f} ({pred.levels.target_2_pct:+.1f}%)")
        print(f"     Risk/Reward: {pred.levels.risk_reward:.1f}x")
        
        if pred.position.shares > 0:
            print(f"\n  📦 Suggested Position:")
            print(f"     Shares: {pred.position.shares:,}")
            print(f"     Value: ¥{pred.position.value:,.2f}")
            print(f"     Risk: ¥{pred.position.risk_amount:,.2f}")
        
        print(f"\n  📋 Analysis:")
        for reason in pred.reasons[:5]:
            print(f"     • {reason}")
        
        if pred.warnings:
            print(f"\n  ⚠️  Warnings:")
            for warning in pred.warnings:
                print(f"     • {warning}")
        
        print("\n" + "=" * 70 + "\n")
        
    except Exception as e:
        print(f"\n  ❌ Error: {e}")
        import traceback
        traceback.print_exc()


def scan_stocks(signal_type: str = "buy", count: int = 10):
    """Scan stocks for signals"""
    from models.predictor import Predictor
    from config import CONFIG
    
    print(f"\n  Scanning {len(CONFIG.STOCK_POOL)} stocks for {signal_type} signals...")
    
    try:
        predictor = Predictor()
        
        if predictor.ensemble is None:
            print("\n  ❌ No trained model found.")
            print("     Run 'python main.py --train' first.")
            return
        
        picks = predictor.get_top_picks(
            CONFIG.STOCK_POOL, 
            n=count, 
            signal_type=signal_type
        )
        
        if not picks:
            print(f"\n  No {signal_type} signals found.")
            return
        
        print("\n" + "=" * 70)
        print(f"  TOP {signal_type.upper()} SIGNALS")
        print("=" * 70)
        
        for i, pred in enumerate(picks, 1):
            indicator = "🟢" if signal_type == "buy" else "🔴"
            print(f"\n  {i}. {indicator} {pred.stock_code} - {pred.stock_name}")
            print(f"     Signal: {pred.signal.value}")
            print(f"     Confidence: {pred.confidence:.0%}")
            print(f"     Price: ¥{pred.current_price:.2f}")
            print(f"     Prob UP: {pred.prob_up:.0%} | Prob DOWN: {pred.prob_down:.0%}")
        
        print("\n" + "=" * 70 + "\n")
        
    except Exception as e:
        print(f"\n  ❌ Error: {e}")


def run_backtest():
    """Run backtest"""
    from analysis.backtest import Backtester
    from config import CONFIG
    
    print("\n" + "=" * 70)
    print("                    WALK-FORWARD BACKTEST")
    print("=" * 70)
    
    print("\n  Configuration:")
    print("    • Training Period: 12 months")
    print("    • Testing Period: 1 month")
    print("    • Rolling forward until present")
    print(f"    • Stocks: {len(CONFIG.STOCK_POOL[:5])}")
    
    print("\n  Running backtest (this may take a while)...\n")
    
    try:
        bt = Backtester()
        result = bt.run(
            stock_codes=CONFIG.STOCK_POOL[:5],
            train_months=12,
            test_months=1
        )
        
        print(result.summary())
        
    except Exception as e:
        print(f"\n  ❌ Error: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='AI Stock Trading System - Professional Trading Application',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python main.py                    # Start GUI application
    python main.py --train            # Train AI model
    python main.py --train --epochs 200
    python main.py --auto-learn       # Auto search and train
    python main.py --predict 600519   # Predict single stock
    python main.py --scan             # Scan all stocks for buy signals
    python main.py --scan --type sell # Scan for sell signals
    python main.py --backtest         # Run backtest
    python main.py --live             # Enable live trading (REAL MONEY!)
        """
    )
    
    # Commands
    parser.add_argument('--check', action='store_true', 
                       help='Check dependencies only')
    parser.add_argument('--train', action='store_true', 
                       help='Train AI model')
    parser.add_argument('--auto-learn', action='store_true', 
                       help='Auto search internet and train')
    parser.add_argument('--predict', type=str, metavar='CODE',
                       help='Predict single stock (e.g., 600519)')
    parser.add_argument('--scan', action='store_true', 
                       help='Scan stocks for signals')
    parser.add_argument('--backtest', action='store_true',
                       help='Run walk-forward backtest')
    parser.add_argument('--model-info', action='store_true',
                       help='Show AI model architecture info')
    
    # Options
    parser.add_argument('--epochs', type=int, default=100, 
                       help='Training epochs (default: 100)')
    parser.add_argument('--max-stocks', type=int, default=80,
                       help='Max stocks for auto-learn (default: 80)')
    parser.add_argument('--type', type=str, choices=['buy', 'sell', 'all'],
                       default='buy', help='Signal type for scan')
    parser.add_argument('--count', type=int, default=10,
                       help='Number of results for scan')
    parser.add_argument('--live', action='store_true', 
                       help='Enable live trading (REAL MONEY!)')
    parser.add_argument('--broker', type=str, 
                       help='Broker executable path for live trading')
    parser.add_argument('--cli', action='store_true', 
                       help='CLI mode (no GUI)')
    parser.add_argument('--risk', type=str, 
                       choices=['conservative', 'moderate', 'aggressive'],
                       default='moderate', help='Risk profile')
    parser.add_argument('--quiet', action='store_true',
                       help='Minimal output')
    
    args = parser.parse_args()
    
    # Print banner
    if not args.quiet:
        print_banner()
    
    # Check dependencies
    if not args.quiet:
        print("  Checking dependencies...")
    
    if not check_dependencies():
        sys.exit(1)
    
    if not args.quiet:
        print("  ✅ All dependencies OK!\n")
    
    if args.check:
        return
    
    # Show model info
    if args.model_info:
        print_model_info()
        return
    
    # Risk warning
    if not args.quiet:
        print_risk_warning()
    
    # Set risk profile
    from config import CONFIG
    CONFIG.set_risk_profile(args.risk)
    
    if not args.quiet:
        print(f"  Risk Profile: {args.risk.upper()}")
        print(f"    • Max Position: {CONFIG.MAX_POSITION_PCT}%")
        print(f"    • Max Daily Loss: {CONFIG.MAX_DAILY_LOSS_PCT}%")
        print(f"    • Risk per Trade: {CONFIG.RISK_PER_TRADE}%")
    
    # Enable live trading
    if args.live:
        if not args.broker:
            print("\n  ❌ Error: --broker path required for live trading")
            print("     Example: python main.py --live --broker 'C:/ths/xiadan.exe'")
            sys.exit(1)
        
        print("\n" + "=" * 70)
        print("  ⚠️  LIVE TRADING MODE")
        print("=" * 70)
        print("\n  You are about to enable LIVE TRADING.")
        print("  This will trade with REAL MONEY!")
        print("\n  Are you sure you want to continue?")
        
        confirm = input("  Type 'YES' to confirm: ")
        if confirm.strip().upper() != 'YES':
            print("\n  Cancelled. Exiting.")
            return
        
        CONFIG.enable_live_trading(args.broker)
        print("\n  ✅ Live trading mode ENABLED")
        print("  ⚠️  Be extremely careful!")
    
    # Execute commands
    if args.train:
        train_model(args.epochs)
        return
    
    if args.auto_learn:
        auto_learn(args.epochs, args.max_stocks)
        return
    
    if args.predict:
        predict_stock(args.predict)
        return
    
    if args.scan:
        scan_stocks(args.type, args.count)
        return
    
    if args.backtest:
        run_backtest()
        return
    
    # Start GUI
    if not args.quiet:
        print("\n  Starting GUI application...")
        print("  Loading AI models and initializing...\n")
    
    try:
        from ui.app import run_app
        run_app()
    except KeyboardInterrupt:
        print("\n\n  Exited.")
    except Exception as e:
        print(f"\n  ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()