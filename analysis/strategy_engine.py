from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import Any

from analysis.strategy_marketplace import StrategyMarketplace
from config.settings import CONFIG
from utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class StrategySignal:
    strategy: str
    action: str  # buy / sell / hold
    score: float  # 0..1 confidence
    weight: float = 1.0
    reason: str = ""


class StrategyScriptEngine:
    """
    Lightweight strategy script runner.

    Strategy script contract:
    - Optional: strategy_meta() -> dict(name, version, description)
    - Required: generate_signal(df, indicators, context) -> dict
      or generate_signals(df, indicators, context) -> list[dict]
      Dict keys: action(str), score(float), reason(str, optional), weight(float, optional)
    """

    def __init__(self, strategies_dir: Path | None = None) -> None:
        base = Path(getattr(CONFIG, "base_dir", Path(".")))
        self._dir = Path(strategies_dir) if strategies_dir else (base / "strategies")
        self._marketplace = StrategyMarketplace(self._dir)

    def list_strategy_files(self) -> list[Path]:
        files = self._marketplace.get_enabled_files()
        if files:
            return files
        # If marketplace metadata exists, treat enabled set as authoritative.
        # This avoids executing disabled scripts via legacy fallback.
        if self._marketplace.manifest_path.exists():
            return []
        if not self._dir.exists():
            return []
        # Fallback for repos without marketplace metadata.
        return sorted(
            p for p in self._dir.glob("*.py")
            if p.is_file() and not p.name.startswith("_")
        )

    def evaluate(
        self,
        df,
        indicators: dict[str, float] | None,
        symbol: str,
        context: dict[str, Any] | None = None,
    ) -> tuple[float, list[str]]:
        """
        Evaluate all strategy scripts and return:
        - bias score contribution in [-25, +25]
        - human-readable reasons list
        """
        total_bias = 0.0
        reasons: list[str] = []
        indicator_map = dict(indicators or {})
        script_context = {
            "symbol": str(symbol or ""),
            "timestamp": datetime.now().isoformat(),
            "market_open": bool(getattr(CONFIG, "is_market_open", lambda: False)()),
            "trading_mode": str(getattr(getattr(CONFIG, "trading_mode", None), "value", "simulation")),
        }
        if isinstance(context, dict):
            script_context.update(context)

        for path in self.list_strategy_files():
            if hasattr(df, "copy"):
                try:
                    df_for_script = df.copy(deep=True)
                except TypeError:
                    df_for_script = df.copy()
            else:
                df_for_script = df
            signals = self._run_one(
                path,
                df_for_script,
                dict(indicator_map),
                dict(script_context),
            )
            if not signals:
                continue

            for signal in signals:
                signed = 0.0
                if signal.action == "buy":
                    signed = signal.score
                elif signal.action == "sell":
                    signed = -signal.score

                total_bias += signed * (15.0 * max(0.1, min(3.0, float(signal.weight))))
                if signal.reason:
                    reasons.append(f"Strategy[{signal.strategy}]: {signal.reason}")

        total_bias = max(-25.0, min(25.0, total_bias))
        return total_bias, reasons

    def _run_one(
        self,
        path: Path,
        df,
        indicators: dict[str, float],
        context: dict[str, Any],
    ) -> list[StrategySignal]:
        out: list[StrategySignal] = []
        try:
            module_name = f"strategy_{path.stem}"
            spec = spec_from_file_location(module_name, str(path))
            if spec is None or spec.loader is None:
                return out
            module = module_from_spec(spec)
            spec.loader.exec_module(module)

            fn_batch = getattr(module, "generate_signals", None)
            fn_single = getattr(module, "generate_signal", None)
            if callable(fn_batch):
                raw_any = fn_batch(df, indicators, context)
            elif callable(fn_single):
                raw_any = fn_single(df, indicators, context)
            else:
                return out

            rows = raw_any if isinstance(raw_any, list) else [raw_any]
            action_alias = {
                "long": "buy",
                "short": "sell",
                "flat": "hold",
            }
            for raw in rows:
                if not isinstance(raw, dict):
                    continue

                action = str(raw.get("action", "hold")).strip().lower()
                action = action_alias.get(action, action)
                if action not in {"buy", "sell", "hold"}:
                    action = "hold"

                score_raw = raw.get("score", 0.0)
                try:
                    score = float(score_raw)
                except (TypeError, ValueError):
                    score = 0.0
                score = max(0.0, min(1.0, score))

                weight_raw = raw.get("weight", 1.0)
                try:
                    weight = float(weight_raw)
                except (TypeError, ValueError):
                    weight = 1.0
                if not (weight > 0):
                    weight = 1.0

                reason = str(raw.get("reason", "") or "").strip()
                if len(reason) > 320:
                    reason = reason[:317].rstrip() + "..."

                out.append(
                    StrategySignal(
                        strategy=path.stem,
                        action=action,
                        score=score,
                        weight=weight,
                        reason=reason,
                    )
                )
            return out
        except Exception as e:
            log.warning("Strategy script failed (%s): %s", path.name, e)
            return out
