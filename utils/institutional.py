from __future__ import annotations

from pathlib import Path
from typing import Any

from config.settings import CONFIG
from utils.policy import get_trade_policy_engine


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value or "").strip().lower()
    if text in {"1", "true", "yes", "on", "y"}:
        return True
    if text in {"0", "false", "no", "off", "n", ""}:
        return False
    return bool(value)


def _to_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def collect_institutional_readiness(
    *,
    security_config: Any | None = None,
    policy_payload: dict[str, Any] | None = None,
    policy_path: Path | None = None,
) -> dict[str, Any]:
    """Evaluate operational controls required for HA/DR + regulated live trading.

    Returns a JSON-serializable report with required/recommended controls.
    """
    sec = security_config if security_config is not None else getattr(CONFIG, "security", None)
    if sec is None:
        sec = object()

    if policy_payload is None:
        engine = get_trade_policy_engine()
        try:
            engine._reload_if_needed(force=True)  # noqa: SLF001
        except Exception:
            pass
        payload = dict(getattr(engine, "_policy", {}) or {})  # noqa: SLF001
        p_path = policy_path if policy_path is not None else Path(getattr(engine, "path", ""))
    else:
        payload = dict(policy_payload or {})
        p_path = Path(policy_path) if policy_path is not None else Path("config/security_policy.json")

    live = dict(payload.get("live_trade") or {})
    controls: list[dict[str, Any]] = []

    def _add(
        control_id: str,
        *,
        ok: bool,
        required: bool,
        expected: str,
        actual: str,
        message: str,
    ) -> None:
        controls.append(
            {
                "id": str(control_id),
                "ok": bool(ok),
                "required": bool(required),
                "expected": str(expected),
                "actual": str(actual),
                "message": str(message),
            }
        )

    lease_enabled = _to_bool(getattr(sec, "enable_runtime_lease", False))
    _add(
        "runtime_lease_enabled",
        ok=lease_enabled,
        required=True,
        expected="true",
        actual=str(lease_enabled).lower(),
        message="Single-writer lease must be enabled to prevent split-brain execution.",
    )

    lease_backend = str(getattr(sec, "runtime_lease_backend", "") or "").strip().lower()
    _add(
        "runtime_lease_backend",
        ok=lease_backend == "sqlite",
        required=True,
        expected="sqlite",
        actual=lease_backend or "unset",
        message="SQLite lease backend provides transactional fencing for active/standby failover.",
    )

    lease_ttl = _to_float(getattr(sec, "runtime_lease_ttl_seconds", 0.0), 0.0)
    _add(
        "runtime_lease_ttl_bounds",
        ok=5.0 <= lease_ttl <= 120.0,
        required=True,
        expected="5..120 seconds",
        actual=f"{lease_ttl:.3f}",
        message="Lease TTL must be bounded to reduce stale-ownership and takeover latency risk.",
    )

    audit_logging = _to_bool(getattr(sec, "audit_logging", False))
    _add(
        "audit_logging_enabled",
        ok=audit_logging,
        required=True,
        expected="true",
        actual=str(audit_logging).lower(),
        message="Regulated operation requires comprehensive audit event capture.",
    )

    audit_chain = _to_bool(getattr(sec, "audit_hash_chain", False))
    _add(
        "audit_hash_chain_enabled",
        ok=audit_chain,
        required=True,
        expected="true",
        actual=str(audit_chain).lower(),
        message="Tamper-evident hash chain must be enabled for audit integrity.",
    )

    retention_days = _to_int(getattr(sec, "audit_retention_days", 0), 0)
    _add(
        "audit_retention_days",
        ok=retention_days >= 365,
        required=True,
        expected=">=365",
        actual=str(retention_days),
        message="Audit retention should be at least one year for baseline regulatory traceability.",
    )

    live_permission = _to_bool(getattr(sec, "require_live_trade_permission", False))
    _add(
        "live_permission_gate",
        ok=live_permission,
        required=True,
        expected="true",
        actual=str(live_permission).lower(),
        message="Live trading permission check must be enabled.",
    )

    require_2fa = _to_bool(getattr(sec, "require_2fa_for_live", False))
    _add(
        "live_2fa_gate",
        ok=require_2fa,
        required=True,
        expected="true",
        actual=str(require_2fa).lower(),
        message="2FA requirement must be enabled for live trade authorization.",
    )

    strict_live = _to_bool(getattr(sec, "strict_live_governance", False))
    _add(
        "strict_live_governance",
        ok=strict_live,
        required=True,
        expected="true",
        actual=str(strict_live).lower(),
        message="Strict live governance must be enabled for dual-control enforcement.",
    )

    min_live_approvals = _to_int(getattr(sec, "min_live_approvals", 0), 0)
    _add(
        "min_live_approvals",
        ok=min_live_approvals >= 2,
        required=True,
        expected=">=2",
        actual=str(min_live_approvals),
        message="At least two approvals are required for institutional dual-control.",
    )

    policy_exists = bool(p_path and Path(p_path).exists())
    _add(
        "policy_file_exists",
        ok=policy_exists,
        required=True,
        expected="true",
        actual=str(policy_exists).lower(),
        message="Live trade policy file must exist and be loadable.",
    )

    policy_enabled = _to_bool(payload.get("enabled", False))
    _add(
        "policy_enabled",
        ok=policy_enabled,
        required=True,
        expected="true",
        actual=str(policy_enabled).lower(),
        message="Policy engine must be enabled for live trade governance.",
    )

    policy_min_approvals = _to_int(live.get("min_approvals", 0), 0)
    _add(
        "policy_min_approvals",
        ok=policy_min_approvals >= 2,
        required=True,
        expected=">=2",
        actual=str(policy_min_approvals),
        message="Policy must enforce at least two live-trade approvals.",
    )

    distinct_approvers = _to_bool(live.get("require_distinct_approvers", False))
    _add(
        "policy_distinct_approvers",
        ok=distinct_approvers,
        required=True,
        expected="true",
        actual=str(distinct_approvers).lower(),
        message="Distinct approvers are required for separation-of-duties.",
    )

    change_ticket = _to_bool(live.get("require_change_ticket", False))
    _add(
        "policy_change_ticket",
        ok=change_ticket,
        required=True,
        expected="true",
        actual=str(change_ticket).lower(),
        message="Policy should require a change ticket for each live trade action.",
    )

    business_justification = _to_bool(live.get("require_business_justification", False))
    _add(
        "policy_business_justification",
        ok=business_justification,
        required=True,
        expected="true",
        actual=str(business_justification).lower(),
        message="Policy should require explicit business justification for live trading.",
    )

    max_notional = _to_float(live.get("max_order_notional", 0.0), 0.0)
    _add(
        "policy_max_notional",
        ok=max_notional > 0.0,
        required=True,
        expected=">0",
        actual=f"{max_notional:.4f}",
        message="Policy must cap max order notional.",
    )

    required_controls = [c for c in controls if bool(c.get("required", False))]
    failed_required = [c for c in required_controls if not bool(c.get("ok", False))]
    failed_ids = [str(c.get("id", "")) for c in failed_required]

    return {
        "pass": len(failed_required) == 0,
        "failed_required_controls": failed_ids,
        "required_total": len(required_controls),
        "required_passed": len(required_controls) - len(failed_required),
        "policy_path": str(p_path) if p_path else "",
        "policy_version": str(payload.get("version", "unknown")),
        "controls": controls,
    }
