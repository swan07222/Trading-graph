# save as debug_learn.py in Trading-graph folder
import time
import threading

# Patch the auto_learner to show what's happening
from config.settings import CONFIG
from data.universe import get_universe_codes
from data.fetcher import get_fetcher

print("=" * 60)
print("STEP 1: Universe codes")
print("=" * 60)

codes = get_universe_codes(force_refresh=True)
print(f"Universe: {len(codes)} codes")

print("\n" + "=" * 60)
print("STEP 2: CONFIG.stock_pool")
print("=" * 60)
print(f"Stock pool: {len(CONFIG.stock_pool)} codes")
print(f"First 5: {CONFIG.stock_pool[:5]}")

print("\n" + "=" * 60)
print("STEP 3: Load learner state")
print("=" * 60)

from models.auto_learner import ContinuousLearner
learner = ContinuousLearner()

print(f"Holdout codes: {len(learner._holdout_codes)}")
print(f"Holdout: {learner._holdout_codes}")
print(f"Replay size: {len(learner._replay)}")
print(f"Rotator processed: {learner._rotator.processed_count}")
print(f"Rotator pool: {learner._rotator.pool_size}")

print("\n" + "=" * 60)
print("STEP 4: Simulate discover_new()")
print("=" * 60)

new_codes = learner._rotator.discover_new(
    max_stocks=50,
    min_market_cap=10,
    stop_check=lambda: False,
    progress_cb=lambda msg, cnt: print(f"  {msg}"),
)
print(f"Discovered: {len(new_codes)} codes")
print(f"First 10: {new_codes[:10]}")

print("\n" + "=" * 60)
print("STEP 5: After holdout filter")
print("=" * 60)

holdout_set = set(learner._holdout_codes)
filtered = [c for c in new_codes if c not in holdout_set]
print(f"After removing holdout: {len(filtered)} codes")
print(f"Holdout set: {holdout_set}")
print(f"Overlap: {set(new_codes) & holdout_set}")

print("\n" + "=" * 60)
print("STEP 6: Mix calculation")
print("=" * 60)

total_learned = len(learner._replay)
if total_learned < 20:
    new_ratio = 0.9
elif total_learned < 100:
    new_ratio = 0.7
else:
    new_ratio = 0.3

max_stocks = 50
num_new = max(3, int(max_stocks * new_ratio))
num_replay = max_stocks - num_new

new_batch = filtered[:num_new]
replay_batch = learner._replay.sample(num_replay)
replay_batch = [c for c in replay_batch if c not in new_batch and c not in holdout_set]

codes_final = new_batch + replay_batch

print(f"new_ratio: {new_ratio}")
print(f"num_new: {num_new}, num_replay: {num_replay}")
print(f"new_batch: {len(new_batch)}")
print(f"replay_batch: {len(replay_batch)}")
print(f"FINAL codes: {len(codes_final)}")

if not codes_final:
    print("\n❌ PROBLEM: Final code list is EMPTY!")
    print("   This is why auto-learning shows 'No stocks available'")
else:
    print(f"\n✅ Would train on {len(codes_final)} stocks")

print("\n" + "=" * 60)
print("STEP 7: Test fetching one stock")
print("=" * 60)

if codes_final:
    test_code = codes_final[0]
else:
    test_code = "600519"

print(f"Fetching {test_code}...")
fetcher = get_fetcher()
try:
    df = fetcher.get_history(test_code, interval="1d", bars=200)
    if df is not None and not df.empty:
        print(f"OK: {len(df)} bars")
    else:
        print("EMPTY result")
except Exception as e:
    print(f"ERROR: {e}")

print("\n" + "=" * 60)
print("DONE")
print("=" * 60)