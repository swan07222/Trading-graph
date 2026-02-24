# Project Bug Report & Issues Analysis

## Critical Issues Found

### 1. **Division by Zero Risk** (`predictor_runtime_ops.py`)
**Location**: Line 477-478, `_forecast_seed()`
```python
vol_hash = int(abs(volatility_context) * 10000) & 0x7FFFFFFF
# If volatility_context is 0.0, vol_hash = 0, which is fine
# BUT in other places:
dir_hash = int((float(np.clip(direction_hint, -1.0, 1.0)) + 1.0) * 500000) & 0x7FFFFFFF
# If direction_hint = -1.0, this gives 0, potentially causing seed = 0
```

**Risk**: While there's a fallback (`seed = 42`), the seed calculation could produce identical seeds for different stocks if all hash components are zero.

**Fix**: Ensure minimum entropy in seed calculation.

---

### 2. **Index Out of Bounds Risk** (`predictor_runtime_ops.py`)
**Location**: Line 437-438, `_sequence_signature()`
```python
tail = arr[-min(64, arr.size):]
# If arr.size = 0, this becomes arr[0:] which is empty but safe
# However, callers might assume non-empty result
```

**Risk**: Silent degradation - returns 0.0 for empty arrays, which might mask data pipeline issues.

**Fix**: Add logging when empty arrays are received.

---

### 3. **Unsafe Type Conversions** (Multiple files)
**Location**: Throughout `predictor_runtime_ops.py`
```python
int(max(1, int(horizon)))  # Double int() conversion
int(abs(float(sequence_signature)) * 1000)  # Float->int can overflow
```

**Risk**: Large float values can overflow when converted to int.

**Fix**: Add bounds checking before conversion.

---

### 4. **Missing Error Handling** (`auto_learner_components.py`)
**Location**: Line 1130-1143, `fetch_one()`
```python
if df is None:
    log.debug(f"Fetch returned None for {code}")
elif df.empty:
    log.debug(f"Fetch returned empty DataFrame for {code}...")
else:
    log.debug(f"Fetch returned {len(df)} bars for {code}, need {min_bars}")
return code, False
```

**Issue**: All failures are logged at DEBUG level - production monitoring won't see systematic fetch failures.

**Fix**: Log at WARNING level when failure rate exceeds threshold.

---

### 5. **Thread Safety Issues** (`auto_learner.py`)
**Location**: Line 579-586, `_ensure_holdout()`
```python
with self._lock:
    current_holdout_set = set(self._holdout_codes)
    if current_holdout_set != old_holdout_set and self._holdout_codes:
        log.debug("Holdout already updated by another thread - skipping")
        return
    self._holdout_codes = new_holdout
```

**Issue**: Race condition check happens AFTER expensive fetch operation, wasting resources.

**Fix**: Check lock status before starting fetch.

---

### 6. **Inconsistent Exception Handling** (Throughout)
**Pattern Found**:
```python
except _AUTO_LEARNER_RECOVERABLE_EXCEPTIONS as e:
    log.debug("... skipped: %s", e)
```

**Issue**: Too many exceptions are silently swallowed at DEBUG level, making production debugging difficult.

**Fix**: 
- Log WARNING when exception rate exceeds threshold
- Add metrics for exception tracking

---

### 7. **Memory Leak Risk** (`auto_learner_components.py`)
**Location**: Line 79-80, `LearningProgress.add_warning()`
```python
self.warnings.append(msg)
if len(self.warnings) > _MAX_MESSAGES:
    self.warnings = self.warnings[-_MAX_MESSAGES:]
```

**Issue**: List slicing creates new list object each time - O(n) operation on every warning after limit.

**Fix**: Use `collections.deque(maxlen=_MAX_MESSAGES)` for O(1) append.

---

### 8. **Incorrect Transaction Cost Calculation** (FIXED)
**Location**: `ui/app_analysis_ops.py` - Already fixed in previous session
- Was using generic rates, now uses China A-share specific rates
- Commission now calculated separately for entry/exit (each has min CNY 5)

---

## Medium Priority Issues

### 9. **Hardcoded Constants Without Configuration**
```python
_GUESS_PROFIT_NOTIONAL_VALUE = 10000.0  # Not in CONFIG
_MAX_MESSAGES = 100  # Not configurable
_SEQUENCE_LENGTH = 60  # In CONFIG but duplicated in some places
```

**Recommendation**: Move to CONFIG for user customization.

---

### 10. **Magic Numbers in Safety Caps** (`predictor_runtime_ops.py`)
```python
hard_jump_cap = max(jump_cap * 1.7, 0.12 if intraday else jump_cap)
bootstrap_cap = 0.30 if not intraday else ...
```

**Issue**: No documentation for why these specific values.

**Fix**: Add comments explaining derivation or move to CONFIG.

---

### 11. **Inefficient Loop in Sanitization** (`predictor_runtime_ops.py`)
**Location**: Line 240-280, `_sanitize_history_df()`
```python
for idx, row in work.iterrows():  # Very slow for large DataFrames
    # ... processing ...
```

**Impact**: Performance bottleneck for stocks with long history.

**Fix**: Vectorize operations where possible.

---

### 12. **Missing Validation** (`auto_learner_cycle_ops.py`)
**Location**: Line 381-386
```python
batch_size = max(1, int(len(codes)))
min_ok = min(batch_size, max(3, int(batch_size * 0.05)))
```

**Issue**: If `codes` has 100 items, `min_ok = 5` (5%). If 10 items, `min_ok = 3` (30%). Inconsistent threshold.

**Fix**: Use consistent percentage with absolute minimum.

---

## Low Priority / Code Quality

### 13. **Duplicate Interval Aliases**
Found in multiple files:
- `auto_learner.py`
- `predictor.py`
- `predictor_runtime_ops.py`
- `fetcher.py`

**Recommendation**: Centralize in `core/constants.py`.

---

### 14. **Inconsistent Logging Levels**
```python
log.debug("...")  # 80% of logs
log.info("...")   # 15%
log.warning("...") # 5%
```

**Issue**: Production systems need INFO level for operational visibility.

**Fix**: Upgrade important operational logs to INFO.

---

### 15. **Missing Docstrings**
Several public methods lack documentation:
- `_sanitize_ohlc_row()` - parameters not explained
- `_bar_safety_caps()` - return values not documented
- `_intraday_session_mask()` - edge cases not documented

---

## Recommendations

### Immediate Actions (This Week)
1. ✅ Fix transaction cost calculation (DONE)
2. Add WARNING level logging for systematic failures
3. Fix memory leak in `LearningProgress.warnings`
4. Add bounds checking for int() conversions

### Short Term (This Month)
5. Centralize interval aliases
6. Move hardcoded constants to CONFIG
7. Add exception rate monitoring
8. Vectorize DataFrame operations

### Long Term (Next Quarter)
9. Comprehensive integration tests
10. Performance profiling and optimization
11. Documentation audit and completion

---

## Files Requiring Attention

| File | Issues | Priority |
|------|--------|----------|
| `models/predictor_runtime_ops.py` | 3, 4, 10, 11 | High |
| `models/auto_learner_components.py` | 4, 7 | High |
| `models/auto_learner.py` | 5, 6 | Medium |
| `models/auto_learner_cycle_ops.py` | 12 | Medium |
| `ui/app_analysis_ops.py` | 8 (FIXED) | Done |

---

## Test Coverage Gaps

The following areas lack adequate test coverage:
1. `_sanitize_history_df()` - edge cases with malformed data
2. `_forecast_seed()` - seed uniqueness across symbols
3. `_compute_guess_profit()` - boundary conditions
4. Thread safety in `ContinuousLearner`
5. Memory usage under sustained operation

---

*Generated: 2026-02-24*
*Analysis based on codebase review*
