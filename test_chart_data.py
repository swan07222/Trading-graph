"""Quick diagnostic script to test chart data loading."""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from config.settings import CONFIG
from data.fetcher import DataFetcher

def test_data_fetch():
    """Test if we can fetch historical data for a stock."""
    print("=" * 60)
    print("CHART DATA DIAGNOSTIC TEST")
    print("=" * 60)
    
    test_stocks = ["600519", "000001", "601318"]
    fetcher = DataFetcher()
    
    for stock in test_stocks:
        print(f"\nTesting {stock}...")
        try:
            # Test daily data
            df_daily = fetcher.get_history(stock, interval="1d", bars=100)
            if df_daily is not None and not df_daily.empty:
                print(f"  [OK] Daily: {len(df_daily)} bars")
                print(f"    Date range: {df_daily.index[0]} to {df_daily.index[-1]}")
                print(f"    Last close: {df_daily['close'].iloc[-1]:.2f}")
            else:
                print(f"  [FAIL] Daily: EMPTY or None")
            
            # Test 1m data
            df_1m = fetcher.get_history(stock, interval="1m", bars=240)
            if df_1m is not None and not df_1m.empty:
                print(f"  [OK] 1m: {len(df_1m)} bars")
                print(f"    Time range: {df_1m.index[0]} to {df_1m.index[-1]}")
                print(f"    Last close: {df_1m['close'].iloc[-1]:.2f}")
            else:
                print(f"  [FAIL] 1m: EMPTY or None")
                
        except Exception as e:
            print(f"  [ERROR] {e}")
    
    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    test_data_fetch()
