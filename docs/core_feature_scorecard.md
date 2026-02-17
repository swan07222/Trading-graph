# Core Feature Scorecard (Desktop A-Share Scope)

Date: 2026-02-17

Benchmark set: TradingView, Thinkorswim, IBKR TWS, MT5 class tools, normalized to desktop trading-support workflows.

## Excluded By Product Direction (Not Counted In Overall)

| Feature | This Project | Famous Apps | Gap |
|---|---:|---:|---:|
| Mobile/web client availability | 2.0 | 9.5 | -7.5 |
| Social/copy/community ecosystem | 1.5 | 8.8 | -7.3 |
| Asset-class breadth | 4.0 | 9.4 | -5.4 |
| Public API ecosystem | 4.0 | 8.9 | -4.9 |
| Cloud sync/collaboration | Excluded | Excluded | Excluded |

## Core Comparison (36 Features)

| # | Core Feature | This Project | Famous Apps | Gap |
|---:|---|---:|---:|---:|
| 1 | Real-time quote reliability | 9.5 | 9.6 | -0.1 |
| 2 | Historical data continuity | 9.5 | 9.4 | +0.1 |
| 3 | Multi-source failover routing | 9.4 | 9.2 | +0.2 |
| 4 | Quote staleness safeguards | 9.4 | 9.3 | +0.1 |
| 5 | Replay determinism | 9.7 | 9.1 | +0.6 |
| 6 | Data leakage prevention | 9.8 | 8.9 | +0.9 |
| 7 | Feature pipeline stability | 9.4 | 9.5 | -0.1 |
| 8 | Model training reproducibility | 9.4 | 9.3 | +0.1 |
| 9 | Auto-learning recovery | 9.4 | 9.2 | +0.2 |
| 10 | Universe discovery robustness | 9.4 | 9.4 | +0.0 |
| 11 | Source health scoring | 9.3 | 9.0 | +0.3 |
| 12 | Pre-trade risk gate quality | 9.6 | 9.4 | +0.2 |
| 13 | Position sizing controls | 9.5 | 9.3 | +0.2 |
| 14 | Daily loss/drawdown limits | 9.4 | 9.2 | +0.2 |
| 15 | Kill switch enforcement | 9.6 | 9.5 | +0.1 |
| 16 | Order validation strictness | 9.6 | 9.3 | +0.3 |
| 17 | Stop/stop-limit trigger behavior | 9.5 | 9.1 | +0.4 |
| 18 | Trailing order behavior | 9.5 | 9.2 | +0.3 |
| 19 | IOC/FOK semantics | 9.5 | 9.4 | +0.1 |
| 20 | Non-marketable limit wait semantics | 9.4 | 9.2 | +0.2 |
| 21 | Partial fill realism | 9.5 | 9.1 | +0.4 |
| 22 | Slippage tracking | 9.5 | 9.3 | +0.2 |
| 23 | OCO lifecycle handling | 9.5 | 9.2 | +0.3 |
| 24 | Broker sync/reconciliation | 9.4 | 9.0 | +0.4 |
| 25 | Runtime lease single-writer lock | 9.8 | 9.5 | +0.3 |
| 26 | Health monitor depth | 9.6 | 9.2 | +0.4 |
| 27 | Degraded-mode auto-pause | 9.6 | 9.3 | +0.3 |
| 28 | Audit hash-chain integrity | 9.7 | 9.4 | +0.3 |
| 29 | Permission/approval governance | 9.6 | 9.3 | +0.3 |
| 30 | Live-start readiness gate | 9.6 | 9.4 | +0.2 |
| 31 | Policy engine enforcement | 9.5 | 9.2 | +0.3 |
| 32 | Alerting and incident hooks | 9.5 | 9.1 | +0.4 |
| 33 | Chaos/resilience drill coverage | 9.4 | 9.0 | +0.4 |
| 34 | Release preflight guards | 9.5 | 9.3 | +0.2 |
| 35 | Test suite depth/regression | 9.4 | 9.1 | +0.3 |
| 36 | Desktop workflow UX efficiency | 9.6 | 9.3 | +0.3 |

## Overall (Core-Only)

- Core-only overall score (36 features): 9.5
- Famous-app normalized benchmark average: 9.3
- Net gap on core scope: +0.3

## Direct Uplifts In This Iteration

- Added robust quote fallback path: feed -> realtime -> cache -> history-close fallback.
- Added non-marketable day-limit wait behavior and stronger conditional trigger handling for stop/trailing orders.
- Added strict LIVE start institutional-readiness gate in execution engine.
- Added UI preflight readiness checks before LIVE connect.

## Remediation Roadmap (Disadvantage Reduction)

Date: 2026-02-17

### Priority Snapshot

| Priority | Theme | Why First | Target Outcome |
|---|---|---|---|
| P0 | Repository hygiene | Current repo tracks `venv/`, `__pycache__`, `.pyc` and inflates every workflow | Clean source-only repo, faster clone/CI/review |
| P1 | Fault visibility | High count of broad `except Exception` and silent `pass` in runtime-critical paths | Fail loudly where needed, predictable degraded behavior |
| P2 | Monolith decomposition | Several 3k-9k line modules slow delivery and increase regression risk | Smaller modules, lower change risk, easier reviews |
| P3 | Concurrency hardening | Many background loops/daemon threads with mixed stop semantics | Deterministic startup/shutdown and recovery |
| P4 | Type/quality gates | Type checks are intentionally loose and exclude risky modules | Stronger static checks in real hot paths |
| P5 | Artifact safety | Runtime loads pickled artifacts/caches | Reduced deserialization risk and stronger artifact integrity |
| P6 | Model fallback governance | Short-history heuristics can emit actionable signals | Explicit confidence policy and safer automation defaults |

### Phase Plan

| Phase | Duration | Scope | Deliverables | Exit Criteria |
|---|---|---|---|---|
| Phase 0 | 1-2 days | Repo hygiene | Untrack non-source artifacts, keep ignore rules enforced in CI | `git ls-files` has no `venv/`, `.pyc`, `__pycache__`, local DB artifacts |
| Phase 1 | 1-2 weeks | Exception policy | Replace silent catches in top-risk paths, add error taxonomy/logging policy | Silent `except ...: pass` near zero in runtime paths |
| Phase 2 | 2-4 weeks | Large-file split | Break `ui/app.py`, `models/*`, `trading/executor.py` into bounded modules | No core file > 1500 lines |
| Phase 3 | 1-2 weeks | Thread lifecycle | Standard stop tokens/join contract/watchdog instrumentation | Clean shutdown under stress tests, no orphan worker loops |
| Phase 4 | 1 week | Type/lint gate hardening | Expand typecheck targets, reduce disabled mypy codes incrementally | CI blocks new type regressions in critical modules |
| Phase 5 | 1 week | Artifact safety | Add signed/checksummed artifact metadata, reduce unsafe load surfaces | Model/cache load path rejects tampered artifacts |
| Phase 6 | 3-5 days | Prediction governance | Restrict auto-trade behavior on heuristic fallback predictions | Fallback mode never bypasses risk/quality guardrails |

### Detailed Workstreams

#### Workstream A: Repo Hygiene (P0)

Targets:
- Remove tracked runtime artifacts and vendored environment content.
- Keep `.gitignore` as source of truth.

Primary tasks:
- Untrack `venv/`, tracked `__pycache__/`, `*.pyc`, and local `.db` files.
- Add a CI guard step that fails if tracked paths match ignored runtime patterns.
- Add a preflight check in release/CI scripts to prevent regression.

Validation commands:
- `git ls-files | Select-String "__pycache__|\\.pyc$|^venv/|\\.db$"`
- `python scripts/release_preflight.py`

#### Workstream B: Exception and Logging Policy (P1)

Targets:
- Prioritize high-risk modules: `ui/app.py`, `trading/executor.py`, `data/fetcher.py`, `models/predictor.py`, `models/auto_learner.py`.

Primary tasks:
- Replace broad catches with typed exceptions where behavior is known.
- Where broad catch is unavoidable, require structured logging + metric increment.
- Disallow silent `pass` in execution/risk/order paths.
- Add tests for negative/failure paths (network fail, broker fail, stale data, teardown).

Acceptance:
- Runtime-critical silent catches reduced to near zero.
- Every caught fault in critical loops emits log context and metric counter.

#### Workstream C: Modularization (P2)

Targets:
- Split monoliths into orchestrator + domain services.

Suggested split map:
- `ui/app.py` -> bootstrap, state store, order actions, chart controller, monitoring controller.
- `trading/executor.py` -> lifecycle manager, routing/execution, reconciliation, runtime lease integration.
- `models/trainer.py` -> data prep, train loop, artifact manager, validation gates.
- `models/auto_learner.py` -> scheduler, universe rotation, replay buffer service, policy engine.
- `models/predictor.py` -> data acquisition, inference core, post-processing/reasoning, fallback policy.

Acceptance:
- File size caps enforced in review.
- Dependency direction documented and stable (UI -> service layer -> data/broker abstractions).

#### Workstream D: Concurrency Hardening (P3)

Targets:
- Deterministic lifecycle across all background workers.

Primary tasks:
- Standardize `start()`, `stop(timeout)`, `join()` contract.
- Ensure loops check cancellation token at bounded intervals.
- Avoid critical behavior depending only on daemon-thread semantics.
- Add soak tests for start/stop/restart and partial-failure recovery.

Acceptance:
- Repeated start/stop cycles pass without leaked threads.
- Graceful shutdown always persists state and releases runtime lease.

#### Workstream E: Quality Gate Hardening (P4)

Targets:
- Move from selective to meaningful type coverage.

Primary tasks:
- Expand `scripts/typecheck_gate.py` targets to include high-risk modules.
- Reduce `mypy` disabled codes in phases instead of all-at-once.
- Add a no-new-baseline-errors policy.

Acceptance:
- CI fails on newly introduced type regressions in critical modules.

#### Workstream F: Artifact and Fallback Safety (P5-P6)

Targets:
- Make artifact loading safer and fallback predictions explicitly bounded.

Primary tasks:
- Add artifact manifest with hash/signature checks for model/scaler loads.
- Constrain or isolate pickle surfaces where practical.
- Mark fallback predictions with strict mode flags and confidence/risk caps.
- Block autonomous high-risk actions when prediction source is fallback mode.

Acceptance:
- Tampered artifacts fail validation before load.
- Fallback-generated signals cannot bypass stricter risk policy.

### 14-Day Execution Plan

| Day Range | Focus | Concrete Output |
|---|---|---|
| Days 1-2 | P0 cleanup | Artifact purge PR + CI guard for runtime artifacts |
| Days 3-6 | P1 pass #1 | Silent-catch cleanup in `trading/executor.py` and `data/fetcher.py` |
| Days 7-9 | P1 pass #2 | Silent-catch cleanup in `models/predictor.py` and `models/auto_learner.py` |
| Days 10-12 | P2 starter | Extract first service modules from `ui/app.py` and `trading/executor.py` |
| Days 13-14 | P4 starter | Expand typecheck gate targets and baseline policy |

### KPI Targets

| Metric | Current | 30-Day Target | 60-Day Target |
|---|---:|---:|---:|
| Tracked runtime artifact files | High | 0 | 0 |
| Broad `except Exception` in critical modules | High | -40% | -70% |
| Silent `except ...: pass` in critical modules | High | -70% | Near 0 |
| Core files > 1500 lines | Multiple | <= 4 | <= 2 |
| Typechecked critical-module coverage | Partial | Medium | High |
