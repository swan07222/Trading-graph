from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from config.settings import CONFIG
from utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class PolicyDecision:
    allowed: bool
    reason: str
    policy_version: str = "unknown"
    metadata: Optional[Dict[str, Any]] = None


class TradePolicyEngine:
    """
    File-backed policy evaluator for governance controls.

    Default policy file: config/security_policy.json
    """

    def __init__(self, policy_path: Optional[Path] = None):
        base = Path(getattr(CONFIG, "base_dir", Path(".")))
        self._path = Path(policy_path) if policy_path else (base / "config" / "security_policy.json")
        self._lock = threading.RLock()
        self._mtime_ns: Optional[int] = None
        self._policy: Dict[str, Any] = {}
        self._reload_if_needed(force=True)

    @property
    def path(self) -> Path:
        return self._path

    def _default_policy(self) -> Dict[str, Any]:
        return {
            "version": "1.0",
            "enabled": True,
            "live_trade": {
                "min_approvals": 2,
                "require_distinct_approvers": True,
                "max_order_notional": 250000.0,
                "blocked_symbols": [],
                "allowed_sides": ["buy", "sell"],
            },
        }

    def _reload_if_needed(self, force: bool = False) -> None:
        with self._lock:
            try:
                if not self._path.exists():
                    self._policy = self._default_policy()
                    return

                mtime_ns = self._path.stat().st_mtime_ns
                if not force and self._mtime_ns == mtime_ns:
                    return

                raw = json.loads(self._path.read_text(encoding="utf-8"))
                if isinstance(raw, dict):
                    self._policy = raw
                    self._mtime_ns = mtime_ns
                else:
                    self._policy = self._default_policy()
            except Exception as e:
                log.warning("Failed to load policy file %s: %s", self._path, e)
                self._policy = self._default_policy()

    def evaluate_live_trade(self, signal: Any) -> PolicyDecision:
        self._reload_if_needed(force=False)
        p = dict(self._policy or {})
        version = str(p.get("version", "unknown"))

        if not bool(p.get("enabled", True)):
            return PolicyDecision(True, "policy disabled", version)

        lp = dict(p.get("live_trade") or {})
        symbol = str(getattr(signal, "symbol", "") or "").strip()
        side = str(getattr(getattr(signal, "side", ""), "value", getattr(signal, "side", "")) or "").strip().lower()
        qty = int(getattr(signal, "quantity", 0) or 0)
        px = float(getattr(signal, "price", 0.0) or 0.0)
        notional = float(max(0.0, qty * px))

        blocked = {str(x).strip() for x in list(lp.get("blocked_symbols", []) or []) if str(x).strip()}
        if symbol and symbol in blocked:
            return PolicyDecision(False, f"policy blocked symbol: {symbol}", version, {"symbol": symbol})

        allowed_sides = {str(x).strip().lower() for x in list(lp.get("allowed_sides", ["buy", "sell"]))}
        if side and side not in allowed_sides:
            return PolicyDecision(False, f"side not allowed by policy: {side}", version, {"side": side})

        max_notional = float(lp.get("max_order_notional", 0.0) or 0.0)
        if max_notional > 0 and notional > max_notional:
            return PolicyDecision(
                False,
                f"order notional exceeds policy max ({notional:,.2f} > {max_notional:,.2f})",
                version,
                {"notional": notional, "max_order_notional": max_notional},
            )

        min_approvals = int(lp.get("min_approvals", 0) or 0)
        approvals_count = int(getattr(signal, "approvals_count", 0) or 0)
        approver_ids = getattr(signal, "approver_ids", None)
        if isinstance(approver_ids, list):
            distinct = {str(x).strip().lower() for x in approver_ids if str(x).strip()}
            approvals_count = max(approvals_count, len(distinct))
            if bool(lp.get("require_distinct_approvers", True)) and len(distinct) < min_approvals:
                return PolicyDecision(
                    False,
                    f"distinct approvals below policy minimum ({len(distinct)}/{min_approvals})",
                    version,
                    {"distinct_approvers": len(distinct), "min_approvals": min_approvals},
                )

        if min_approvals > 0 and approvals_count < min_approvals:
            return PolicyDecision(
                False,
                f"approvals below policy minimum ({approvals_count}/{min_approvals})",
                version,
                {"approvals": approvals_count, "min_approvals": min_approvals},
            )

        return PolicyDecision(True, "allowed", version, {"evaluated_at": datetime.now().isoformat()})


_engine: Optional[TradePolicyEngine] = None
_engine_lock = threading.Lock()


def get_trade_policy_engine() -> TradePolicyEngine:
    global _engine
    if _engine is None:
        with _engine_lock:
            if _engine is None:
                _engine = TradePolicyEngine()
    return _engine
