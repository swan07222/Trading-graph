from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from config.settings import CONFIG
from utils.logger import get_logger

log = get_logger(__name__)


class StrategyMarketplace:
    """Local strategy catalog with enable/disable state and integrity checks."""

    def __init__(self, strategies_dir: Path | None = None) -> None:
        base = Path(getattr(CONFIG, "base_dir", Path(".")))
        self._dir = Path(strategies_dir) if strategies_dir else (base / "strategies")
        self._manifest_path = self._dir / "marketplace.json"
        self._enabled_path = self._dir / "enabled.json"
        self._dir.mkdir(parents=True, exist_ok=True)

    @property
    def manifest_path(self) -> Path:
        return self._manifest_path

    @property
    def enabled_path(self) -> Path:
        return self._enabled_path

    def _read_json(self, path: Path, default: Any) -> Any:
        try:
            if not path.exists():
                return default
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            log.warning("Failed reading %s: %s", path.name, e)
            return default

    def _write_json(self, path: Path, data: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )

    def _file_hash(self, path: Path) -> str:
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()

    def _resolve_strategy_file(self, file_name: str) -> Path | None:
        """Resolve a manifest file entry to a safe python script path."""
        raw = str(file_name or "").strip()
        if not raw:
            return None

        candidate = Path(raw)
        if candidate.is_absolute():
            return None
        if candidate.suffix.lower() != ".py":
            return None

        try:
            root = self._dir.resolve()
            resolved = (self._dir / candidate).resolve()
            resolved.relative_to(root)
        except Exception:
            return None
        return resolved

    def list_entries(self) -> list[dict]:
        raw = self._read_json(self._manifest_path, {"strategies": []})
        items = list(raw.get("strategies", []) or [])
        enabled_set = set(self.get_enabled_ids())

        out: list[dict] = []
        for entry in items:
            sid = str(entry.get("id", "")).strip()
            file_name = str(entry.get("file", "")).strip()
            if not sid or not file_name:
                continue
            file_path = self._resolve_strategy_file(file_name)
            installed = bool(file_path and file_path.exists() and file_path.is_file())
            expected_hash = str(entry.get("sha256", "")).strip().lower()
            integrity = "unknown"
            if file_path is None:
                integrity = "error"
            elif installed:
                if expected_hash:
                    try:
                        integrity = "ok" if self._file_hash(file_path) == expected_hash else "mismatch"
                    except Exception:
                        integrity = "error"
                else:
                    integrity = "unverified"
            else:
                integrity = "missing"

            item = dict(entry)
            item["id"] = sid
            item["file"] = file_name
            item["installed"] = installed
            item["integrity"] = integrity
            item["enabled"] = sid in enabled_set
            out.append(item)
        return out

    def get_enabled_ids(self) -> list[str]:
        data = self._read_json(self._enabled_path, {})
        enabled = data.get("enabled")
        if isinstance(enabled, list):
            out: list[str] = []
            seen: set[str] = set()
            for x in enabled:
                sid = str(x).strip()
                if not sid or sid in seen:
                    continue
                seen.add(sid)
                out.append(sid)
            return out

        # Fallback to defaults from manifest
        defaults: list[str] = []
        for item in self._read_json(self._manifest_path, {"strategies": []}).get("strategies", []):
            sid = str(item.get("id", "")).strip()
            if sid and bool(item.get("enabled_by_default", False)):
                defaults.append(sid)
        return defaults

    def save_enabled_ids(self, strategy_ids: list[str]) -> None:
        valid_entries = {str(item["id"]): item for item in self.list_entries()}
        clean = []
        seen = set()
        for sid in strategy_ids:
            s = str(sid).strip()
            if not s or s in seen:
                continue
            entry = valid_entries.get(s)
            if not entry:
                continue
            if not bool(entry.get("installed", False)):
                continue
            if str(entry.get("integrity", "")) in ("mismatch", "missing", "error"):
                continue
            seen.add(s)
            clean.append(s)
        self._write_json(self._enabled_path, {"enabled": clean})

    def get_enabled_files(self) -> list[Path]:
        return [
            p
            for p in (
                Path(str(item.get("_resolved_file")))
                for item in self.get_enabled_entries()
            )
            if p.exists() and p.is_file()
        ]

    def get_enabled_entries(self) -> list[dict]:
        """Enabled strategy entries with resolved file paths and metadata."""
        enabled = set(self.get_enabled_ids())
        out: list[dict] = []
        seen_paths: set[str] = set()
        for item in self.list_entries():
            if item["id"] not in enabled:
                continue
            integrity = str(item.get("integrity", "")).lower()
            if integrity not in ("ok", "unverified"):
                continue
            file_path = self._resolve_strategy_file(str(item.get("file", "")))
            if file_path is None or not file_path.exists() or not file_path.is_file():
                continue
            key = str(file_path)
            if key in seen_paths:
                continue
            seen_paths.add(key)
            row = dict(item)
            row["_resolved_file"] = str(file_path)
            out.append(row)
        return out

    def get_integrity_summary(self) -> dict[str, int]:
        summary: dict[str, int] = {
            "total": 0,
            "ok": 0,
            "unverified": 0,
            "mismatch": 0,
            "missing": 0,
            "error": 0,
            "unknown": 0,
        }
        for item in self.list_entries():
            summary["total"] += 1
            key = str(item.get("integrity", "unknown"))
            if key not in summary:
                key = "unknown"
            summary[key] += 1
        return summary
