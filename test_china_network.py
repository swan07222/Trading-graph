"""Test data sources from China network."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from data.fetcher import DataFetcher
from data.fetcher_sources import TencentQuoteSource, AkShareSource, SinaHistorySource

print("=" * 70)
print("CHINA NETWORK DATA SOURCE TEST")
print("=" * 70)

test_stock = "000001"  # Ping An Bank

# Test each source individually
sources = [
    ("Tencent (Real-time)", TencentQuoteSource()),
    ("AkShare (History)", AkShareSource()),
    ("Sina (History)", SinaHistorySource()),
]

for name, source in sources:
    print(f"\n{name}:")
    try:
        # Test quote
        if hasattr(source, 'get_quote'):
            quote = source.get_quote(test_stock)
            if quote and quote.price > 0:
                print(f"  [OK] Quote: {quote.price:.2f}")
            else:
                print(f"  [FAIL] Quote: None or zero")
        
        # Test history
        if hasattr(source, 'get_history'):
            df = source.get_history(test_stock, days=10)
            if df is not None and not df.empty:
                print(f"  [OK] History: {len(df)} bars, last close: {df['close'].iloc[-1]:.2f}")
            else:
                print(f"  [FAIL] History: Empty or None")
    except Exception as e:
        print(f"  [ERROR] {type(e).__name__}: {e}")

# Test full fetcher
print("\n" + "=" * 70)
print("FULL FETCHER TEST (000001):")
print("=" * 70)

fetcher = DataFetcher()

print("\n1m data (240 bars):")
try:
    df_1m = fetcher.get_history(test_stock, interval="1m", bars=240, 
                                 use_cache=False, update_db=False, allow_online=True)
    if df_1m is not None and not df_1m.empty:
        print(f"  [OK] {len(df_1m)} bars")
        print(f"  Range: {df_1m.index[0]} to {df_1m.index[-1]}")
        print(f"  Last close: {df_1m['close'].iloc[-1]:.2f}")
    else:
        print(f"  [FAIL] Empty/None")
except Exception as e:
    print(f"  [ERROR] {type(e).__name__}: {e}")

print("\n1d data (60 bars):")
try:
    df_1d = fetcher.get_history(test_stock, interval="1d", bars=60,
                                 use_cache=False, update_db=False, allow_online=True)
    if df_1d is not None and not df_1d.empty:
        print(f"  [OK] {len(df_1d)} bars")
        print(f"  Range: {df_1d.index[0]} to {df_1d.index[-1]}")
        print(f"  Last close: {df_1d['close'].iloc[-1]:.2f}")
    else:
        print(f"  [FAIL] Empty/None")
except Exception as e:
    print(f"  [ERROR] {type(e).__name__}: {e}")

print("\n" + "=" * 70)
print("TEST COMPLETE")
print("=" * 70)
