# save as test_fetch.py in Trading-graph folder
from data.fetcher import get_fetcher, DataFetcher

fetcher = get_fetcher()

print("=" * 60)
print("Active sources:")
for s in fetcher._get_active_sources():
    print(f"  {s.name}: available={s.is_available()}, network_ok={s.is_suitable_for_network()}")

print("\nAll sources:")
for s in fetcher._all_sources:
    print(f"  {s.name}: available={s.status.available}, network_ok={s.is_suitable_for_network()}")

print("\n" + "=" * 60)
print("Test fetch 600519 (Moutai):")

# Try each source individually
for s in fetcher._all_sources:
    if s.name == "localdb":
        continue
    print(f"\n  Trying {s.name}...")
    print(f"    available: {s.is_available()}")
    print(f"    network_ok: {s.is_suitable_for_network()}")
    try:
        from core.instruments import parse_instrument
        inst = parse_instrument("600519")
        print(f"    instrument: {inst}")
        df = s.get_history_instrument(inst, days=100, interval="1d")
        if df is not None and not df.empty:
            print(f"    ✅ Got {len(df)} bars")
        else:
            print(f"    ❌ Empty result")
    except Exception as e:
        print(f"    ❌ Error: {e}")

print("\n" + "=" * 60)
print("Test via main fetcher:")
df = fetcher.get_history("600519", interval="1d", bars=100)
if df is not None and not df.empty:
    print(f"✅ Main fetcher OK: {len(df)} bars")
else:
    print("❌ Main fetcher returned empty")