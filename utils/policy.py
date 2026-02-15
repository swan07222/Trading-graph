from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from config.settings import CONFIG
from utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class PolicyDecision:
    allowed: bool
    reason: str
    policy_version: str = "unknown"
    metadata: dict[str, Any] | None = None


class TradePolicyEngine:
    """
    File-backed policy evaluator for governance controls.

    Default policy file: config/security_policy.json
    """

    def __init__(self, policy_path: Path | None = None):
        base = Path(getattr(CONFIG, "base_dir", Path(".")))
        self._path = Path(policy_path) if policy_path else (base / "config" / "security_policy.json")
        self._lock = threading.RLock()
        self._mtime_ns: int | None = None
        self._policy: dict[str, Any] = {}
        self._reload_if_needed(force=True)

    @property
    def path(self) -> Path:
        return self._path

    @staticmethod
    def _normalize_symbol(symbol: str) -> str:
        digits = "".join(ch for ch in str(symbol or "").strip() if ch.isdigit())
        return digits.zfill(6) if digits else ""

    @staticmethod
    def _to_int(value: Any, default: int = 0) -> int:
        try:
            return int(value)
        except Exception:
            return int(default)

    @staticmethod
    def _to_float(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except Exception:
            return float(default)

    @staticmethod
    def _normalize_order_type(value: Any) -> str:
        raw = str(value or "limit").strip().lower().replace("-", "_")
        alias = {
            "trailing": "trail_market",
            "trailing_market": "trail_market",
            "trailing_stop": "trail_market",
            "trailing_limit": "trail_limit",
            "stoploss": "stop",
            "stop_loss": "stop",
            "market_ioc": "ioc",
            "market_fok": "fok",
        }
        return alias.get(raw, raw)

    def _default_policy(self) -> dict[str, Any]:
        return {
            "version": "1.0",
            "enabled": True,
            "live_trade": {
                "min_approvals": 2,
                "require_distinct_approvers": True,
                "max_order_notional": 250000.0,
                "blocked_symbols": [],
                "allowed_sides": ["buy", "sell"],
                "allowed_order_types": [
                    "limit",
                    "market",
                    "stop",
                    "stop_limit",
                    "ioc",
                    "fok",
                    "trail_market",
                    "trail_limit",
                ],
                "blocked_strategies": [],
                "require_manual_for_live": False,
                "require_change_ticket": True,
                "require_business_justification": True,
            },
        }

    def _reload_if_needed(self, force: bool = False) -> None:
        with self._lock:
            try:
                if not self._path.exists():
                    self._policy = self._default_policy()
                    self._mtime_ns = None
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
                self._mtime_ns = None

    def evaluate_live_trade(self, signal: Any) -> PolicyDecision:
        self._reload_if_needed(force=False)
        p = dict(self._policy or {})
        version = str(p.get("version", "unknown"))

        if not bool(p.get("enabled", True)):
            return PolicyDecision(True, "policy disabled", version)

        lp = dict(p.get("live_trade") or {})
        symbol = str(getattr(signal, "symbol", "") or "").strip()
        symbol_norm = self._normalize_symbol(symbol)
        side = str(getattr(getattr(signal, "side", ""), "value", getattr(signal, "side", "")) or "").strip().lower()
        order_type = self._normalize_order_type(
            getattr(signal, "order_type", "limit")
        )
        qty = self._to_int(getattr(signal, "quantity", 0) or 0)
        px = self._to_float(getattr(signal, "price", 0.0) or 0.0)
        notional = float(max(0.0, qty * px))
        strategy = str(getattr(signal, "strategy", "") or "").strip().lower()
        auto_generated = bool(getattr(signal, "auto_generated", False))

        blocked = {
            str(x).strip()
            for x in list(lp.get("blocked_symbols", []) or [])
            if str(x).strip()
        }
        blocked_norm = {self._normalize_symbol(x) for x in blocked}
        if (symbol and symbol in blocked) or (symbol_norm and symbol_norm in blocked_norm):
            return PolicyDecision(False, f"policy blocked symbol: {symbol}", version, {"symbol": symbol})

        if qty <= 0:
            return PolicyDecision(
                False,
                "policy rejected non-positive quantity",
                version,
                {"quantity": qty},
            )
        if order_type in {"limit", "stop_limit", "trail_limit"} and px <= 0:
            return PolicyDecision(
                False,
                "policy rejected non-positive limit price",
                version,
                {"price": px},
            )

        allowed_sides = {str(x).strip().lower() for x in list(lp.get("allowed_sides", ["buy", "sell"]))}
        if side and side not in allowed_sides:
            return PolicyDecision(False, f"side not allowed by policy: {side}", version, {"side": side})

        allowed_order_types = {
            self._normalize_order_type(x)
            for x in list(lp.get("allowed_order_types", ["limit", "market"]))
            if str(x).strip()
        }
        if order_type and allowed_order_types and order_type not in allowed_order_types:
            return PolicyDecision(
                False,
                f"order_type not allowed by policy: {order_type}",
                version,
                {"order_type": order_type},
            )

        blocked_strategies = {
            str(x).strip().lower()
            for x in list(lp.get("blocked_strategies", []) or [])
            if str(x).strip()
        }
        if strategy and strategy in blocked_strategies:
            return PolicyDecision(
                False,
                f"policy blocked strategy: {strategy}",
                version,
                {"strategy": strategy},
            )

        if bool(lp.get("require_manual_for_live", False)) and auto_generated:
            return PolicyDecision(
                False,
                "policy requires manual submission for live trades",
                version,
                {"auto_generated": True},
            )

        if bool(lp.get("require_change_ticket", False)):
            ticket = str(
                getattr(signal, "change_ticket", "")
                or getattr(signal, "ticket_id", "")
                or ""
            ).strip()
            if not ticket:
                return PolicyDecision(
                    False,
                    "policy requires change ticket for live trade",
                    version,
                    {},
                )

        if bool(lp.get("require_business_justification", False)):
            reasons = getattr(signal, "reasons", None)
            reason_count = len([x for x in reasons if str(x).strip()]) if isinstance(reasons, list) else 0
            justification = str(getattr(signal, "business_justification", "") or "").strip()
            if reason_count <= 0 and not justification:
                return PolicyDecision(
                    False,
                    "policy requires non-empty business justification",
                    version,
                    {},
                )

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
        if not isinstance(approver_ids, list):
            approver_ids = getattr(signal, "approved_by", None)
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


_engine: TradePolicyEngine | None = None
_engine_lock = threading.Lock()


def get_trade_policy_engine() -> TradePolicyEngine:
    global _engine
    if _engine is None:
        with _engine_lock:
            if _engine is None:
                _engine = TradePolicyEngine()
    return _engine
