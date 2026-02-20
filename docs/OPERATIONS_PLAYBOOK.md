# Operations Playbook

This playbook hardens deployment, rollback, and runtime observability for `Trading-graph`.

## 1) Pre-Deploy Gates

Run full preflight locally (or in staging) before every release:

```bash
python scripts/release_preflight.py --observability-url http://127.0.0.1:9090
```

Fast-path preflight for frequent deploy iterations:

```bash
python scripts/release_preflight.py --profile quick
```

`--profile quick` keeps core runtime gates (`artifact`, `health`, `doctor`, `typecheck`)
and skips longer checks (`lint`, `pytest`, `regulatory`, `ha_dr`).

What it checks:
- `main.py --health --health-strict`
- `main.py --doctor --doctor-strict`
- `scripts/typecheck_gate.py` (baseline-ratchet type gate)
- `scripts/regulatory_readiness.py` (institutional control gate)
- `scripts/ha_dr_drill.py` (lease-fencing failover drill)
- Optional HTTP probe of `/healthz`, `/metrics`, and `/api/v1/*`

Standalone governance/HA checks:

```bash
python scripts/regulatory_readiness.py
python scripts/ha_dr_drill.py --backend sqlite --ttl-seconds 5
```

## 2) Soak Testing (Including Live Broker Conditions)

Paper/simulation soak:

```bash
python scripts/soak_broker_e2e.py --mode paper --duration-minutes 120 --poll-seconds 5 --symbols 600519,000001
```

Live-condition soak (read-only broker/account/order/fill/health checks):

```bash
python scripts/soak_broker_e2e.py --mode live --allow-live --duration-minutes 120 --poll-seconds 5 --symbols 600519,000001
```

Optional submit+cancel probe order path (high risk in live, explicitly gated):

```bash
python scripts/soak_broker_e2e.py --mode live --allow-live --place-probe-order --allow-live-orders --probe-symbol 600519
```

Recommended acceptance thresholds:
- `disconnect_ticks == 0`
- `unhealthy_ticks == 0`
- `quote_success_ratio >= 0.95`
- `max_consecutive_failures <= 2`

## 3) Deployment Snapshot

Create snapshot right before deployment:

```bash
python scripts/deployment_snapshot.py create --snapshot-dir backups
```

This captures runtime state and governance files:
- `config.json`
- `config/security_policy.json`
- `strategies/enabled.json`
- runtime/lease/order DB files under `data_storage/`

Optional:
- include extra files: `--include-path path/to/file`
- include model directory: `--include-models`

## 4) Rollback

Dry-run rollback:

```bash
python scripts/deployment_snapshot.py restore --archive backups/snapshot_<tag>.tar.gz --dry-run
```

Execute rollback:

```bash
python scripts/deployment_snapshot.py restore --archive backups/snapshot_<tag>.tar.gz --confirm
```

Execute rollback + automated post-verify in one command:

```bash
python scripts/deployment_snapshot.py restore --archive backups/snapshot_<tag>.tar.gz --confirm --post-verify --post-verify-profile quick --post-observability-url http://127.0.0.1:9090 --post-soak-minutes 10 --post-soak-mode paper --post-soak-symbols 600519,000001
```

After rollback:
1. Re-run preflight (`scripts/release_preflight.py`)
2. Re-run soak smoke (`scripts/soak_broker_e2e.py --duration-minutes 10 ...`)
3. Verify observability endpoints (`scripts/observability_probe.py`)
4. Re-run HA/DR fencing drill (`scripts/ha_dr_drill.py`)

## 5) Type Gate Operations

Run local gate:

```bash
python scripts/typecheck_gate.py
```

If legacy issue set changes intentionally, update baseline:

```bash
python scripts/typecheck_gate.py --write-baseline
```

Baseline location: `.ci/mypy-baseline.txt`
