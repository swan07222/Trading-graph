# save as: test_akshare_1m.py
import akshare as ak
import socket
import time

print("Testing AkShare 1-minute data fetch...")
print(f"AkShare version: {ak.__version__}")
print()

test_codes = ["600519", "000001", "000063"]

for code in test_codes:
    print(f"--- {code} ---")
    
    old_timeout = socket.getdefaulttimeout()
    socket.setdefaulttimeout(15)
    
    try:
        start = time.time()
        df = ak.stock_zh_a_hist_min_em(symbol=code, period="1", adjust="qfq")
        elapsed = time.time() - start
        
        if df is None:
            print(f"  Result: None")
        elif df.empty:
            print(f"  Result: Empty DataFrame")
        else:
            print(f"  Rows: {len(df)}")
            print(f"  Columns: {list(df.columns)}")
            print(f"  Date range: {df.iloc[0, 0]} → {df.iloc[-1, 0]}")
            print(f"  First row: {df.iloc[0].to_dict()}")
            print(f"  Time: {elapsed:.1f}s")
    except Exception as e:
        print(f"  ERROR: {type(e).__name__}: {e}")
    finally:
        socket.setdefaulttimeout(old_timeout)
    
    print()
    time.sleep(1)

# Also test daily data
print("--- Daily data test (600519) ---")
try:
    df = ak.stock_zh_a_hist(symbol="600519", period="daily",
                            start_date="20250601", end_date="20250615", adjust="qfq")
    if df is not None and not df.empty:
        print(f"  Rows: {len(df)}")
        print(f"  Columns: {list(df.columns)}")
    else:
        print(f"  Result: Empty")
except Exception as e:
    print(f"  ERROR: {type(e).__name__}: {e}")

# Test 5-minute data
print("\n--- 5-minute data test (600519) ---")
try:
    df = ak.stock_zh_a_hist_min_em(symbol="600519", period="5", adjust="qfq")
    if df is not None and not df.empty:
        print(f"  Rows: {len(df)}")
        print(f"  Columns: {list(df.columns)}")
        print(f"  Date range: {df.iloc[0, 0]} → {df.iloc[-1, 0]}")
    else:
        print(f"  Result: Empty")
except Exception as e:
    print(f"  ERROR: {type(e).__name__}: {e}")

print("\nDone!")