from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from config.settings import CONFIG
from analysis.strategy_marketplace import StrategyMarketplace
from utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class StrategySignal:
    strategy: str
    action: str  # buy / sell / hold
    score: float  # 0..1 confidence
    reason: str = ""


class StrategyScriptEngine:
    """
    Lightweight strategy script runner.

    Strategy script contract:
    - Optional: strategy_meta() -> dict(name, version, description)
    - Required: generate_signal(df, indicators, context) -> dict
      Dict keys: action(str), score(float), reason(str, optional)
    """

    def __init__(self, strategies_dir: Optional[Path] = None) -> None:
        base = Path(getattr(CONFIG, "base_dir", Path(".")))
        self._dir = Path(strategies_dir) if strategies_dir else (base / "strategies")
        self._marketplace = StrategyMarketplace(self._dir)

    def list_strategy_files(self) -> List[Path]:
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
        indicators: Optional[Dict[str, float]],
        symbol: str,
    ) -> Tuple[float, List[str]]:
        """
        Evaluate all strategy scripts and return:
        - bias score contribution in [-25, +25]
        - human-readable reasons list
        """
        total_bias = 0.0
        reasons: List[str] = []
        indicator_map = dict(indicators or {})
        context = {
            "symbol": str(symbol or ""),
            "timestamp": datetime.now().isoformat(),
        }

        for path in self.list_strategy_files():
            signal = self._run_one(path, df, indicator_map, context)
            if signal is None:
                continue

            signed = 0.0
            if signal.action == "buy":
                signed = signal.score
            elif signal.action == "sell":
                signed = -signal.score

            total_bias += signed * 15.0
            if signal.reason:
                reasons.append(f"Strategy[{signal.strategy}]: {signal.reason}")

        total_bias = max(-25.0, min(25.0, total_bias))
        return total_bias, reasons

    def _run_one(
        self,
        path: Path,
        df,
        indicators: Dict[str, float],
        context: Dict[str, Any],
    ) -> Optional[StrategySignal]:
        try:
            module_name = f"strategy_{path.stem}"
            spec = spec_from_file_location(module_name, str(path))
            if spec is None or spec.loader is None:
                return None
            module = module_from_spec(spec)
            spec.loader.exec_module(module)

            fn = getattr(module, "generate_signal", None)
            if not callable(fn):
                return None

            raw = fn(df, indicators, context)
            if not isinstance(raw, dict):
                return None

            action = str(raw.get("action", "hold")).strip().lower()
            if action not in {"buy", "sell", "hold"}:
                action = "hold"

            score_raw = raw.get("score", 0.0)
            try:
                score = float(score_raw)
            except (TypeError, ValueError):
                score = 0.0
            score = max(0.0, min(1.0, score))

            reason = str(raw.get("reason", "") or "").strip()
            return StrategySignal(
                strategy=path.stem,
                action=action,
                score=score,
                reason=reason,
            )
        except Exception as e:
            log.warning("Strategy script failed (%s): %s", path.name, e)
            return None
