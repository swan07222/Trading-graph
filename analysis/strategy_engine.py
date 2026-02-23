from __future__ import annotations

import inspect
from dataclasses import asdict, dataclass
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
    cooldown_seconds: int = 0
    tags: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None


class StrategyScriptEngine:
    """Lightweight strategy script runner.

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
        # If marketplace metadata exists with strategies, respect the enabled list.
        # For empty/fresh manifests (no strategies listed), use legacy fallback.
        if self._marketplace.manifest_path.exists():
            # Check if manifest has any strategies defined
            try:
                import json
                manifest_data = json.loads(self._marketplace.manifest_path.read_text(encoding="utf-8"))
                has_strategies = bool(manifest_data.get("strategies"))
            except Exception:
                has_strategies = False
            
            if has_strategies:
                # Manifest has strategies - check if enabled.json exists
                if not self._marketplace.enabled_path.exists():
                    # Use strategies marked enabled_by_default from manifest
                    default_enabled = self._marketplace.get_enabled_ids()
                    if default_enabled:
                        # Return files for default-enabled strategies
                        result = []
                        for item in self._marketplace.list_entries():
                            if item.id in default_enabled and item._resolved_file:
                                file_path = Path(item._resolved_file)
                                if file_path.exists() and file_path.is_file():
                                    result.append(file_path)
                        if result:
                            return result
                # If manifest exists with strategies but none are enabled, return empty
                return []
            # Empty manifest - fall through to legacy file discovery
        
        if not self._dir.exists():
            return []
        # Fallback for repos without marketplace metadata or with empty manifest.
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
        """Evaluate all strategy scripts and return:
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

        strategy_meta_by_path: dict[str, dict[str, Any]] = {}
        try:
            for row in self._marketplace.get_enabled_entries():
                resolved = Path(str(getattr(row, "_resolved_file", "")))
                if not resolved:
                    continue
                # Use dataclasses.asdict() to properly convert dataclass to dict
                strategy_meta_by_path[str(resolved.resolve())] = asdict(row)
        except Exception:
            strategy_meta_by_path = {}

        for path in self.list_strategy_files():
            entry = strategy_meta_by_path.get(str(path.resolve()), {})
            configured_params = entry.get("params")
            if not isinstance(configured_params, dict):
                configured_params = entry.get("config")
            if not isinstance(configured_params, dict):
                configured_params = {}
            try:
                base_weight = float(entry.get("weight", 1.0) or 1.0)
            except Exception:
                base_weight = 1.0
            base_weight = max(0.1, min(5.0, base_weight))

            strategy_context = dict(script_context)
            strategy_context["strategy_id"] = str(entry.get("id") or path.stem)
            strategy_context["strategy_name"] = str(entry.get("name") or path.stem)
            strategy_context["strategy_params"] = dict(configured_params)

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
                dict(strategy_context),
                params=dict(configured_params),
                base_weight=base_weight,
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
        params: dict[str, Any] | None = None,
        base_weight: float = 1.0,
    ) -> list[StrategySignal]:
        out: list[StrategySignal] = []
        try:
            module_name = f"strategy_{path.stem}"
            spec = spec_from_file_location(module_name, str(path))
            if spec is None or spec.loader is None:
                return out
            module = module_from_spec(spec)
            spec.loader.exec_module(module)

            effective_params: dict[str, Any] = {}
            if isinstance(params, dict):
                effective_params.update(params)

            declared_params = getattr(module, "STRATEGY_DEFAULT_PARAMS", None)
            if isinstance(declared_params, dict):
                for k, v in declared_params.items():
                    effective_params.setdefault(str(k), v)

            raw_strategy_params = getattr(module, "strategy_params", None)
            if callable(raw_strategy_params):
                try:
                    suggested = raw_strategy_params()
                    if isinstance(suggested, dict):
                        for k, v in suggested.items():
                            effective_params.setdefault(str(k), v)
                except Exception:
                    pass
            elif isinstance(raw_strategy_params, dict):
                for k, v in raw_strategy_params.items():
                    effective_params.setdefault(str(k), v)

            runtime_context = dict(context)
            runtime_context["strategy_params"] = dict(effective_params)

            meta = {}
            raw_meta = getattr(module, "strategy_meta", None)
            if callable(raw_meta):
                try:
                    m = raw_meta()
                    if isinstance(m, dict):
                        meta = dict(m)
                except Exception:
                    meta = {}

            meta_weight = float(meta.get("weight", 1.0) or 1.0) if isinstance(meta, dict) else 1.0
            meta_weight = max(0.1, min(3.0, meta_weight))
            combined_base_weight = max(0.1, min(8.0, float(base_weight) * meta_weight))

            fn_batch = getattr(module, "generate_signals", None)
            fn_single = getattr(module, "generate_signal", None)

            def _invoke(fn):
                try:
                    sig = inspect.signature(fn)
                    if len(sig.parameters) >= 4:
                        return fn(df, indicators, runtime_context, effective_params)
                except Exception:
                    pass
                return fn(df, indicators, runtime_context)

            if callable(fn_batch):
                raw_any = _invoke(fn_batch)
            elif callable(fn_single):
                raw_any = _invoke(fn_single)
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
                weight *= combined_base_weight

                reason = str(raw.get("reason", "") or "").strip()
                if len(reason) > 320:
                    reason = reason[:317].rstrip() + "..."

                cooldown_seconds = 0
                try:
                    cooldown_seconds = int(raw.get("cooldown_seconds", 0) or 0)
                except Exception:
                    cooldown_seconds = 0
                cooldown_seconds = max(0, min(86400, cooldown_seconds))

                tags = raw.get("tags")
                if not isinstance(tags, dict):
                    tags = {}
                metadata = raw.get("metadata")
                if not isinstance(metadata, dict):
                    metadata = {}
                if meta:
                    metadata.setdefault("strategy_meta", dict(meta))

                out.append(
                    StrategySignal(
                        strategy=path.stem,
                        action=action,
                        score=score,
                        weight=weight,
                        reason=reason,
                        cooldown_seconds=cooldown_seconds,
                        tags=tags,
                        metadata=metadata,
                    )
                )
            return out
        except Exception as e:
            log.warning("Strategy script failed (%s): %s", path.name, e)
            return out
