"""
Strategy Marketplace - Enhanced local strategy catalog with ratings, performance tracking,
versioning, import/export, and community sharing capabilities.

Features:
- Strategy rating and review system
- Performance tracking and leaderboards
- Version control and auto-update checks
- Import/export with checksums
- Category-based organization
- Strategy performance metrics
- Community sharing metadata
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import time
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from config.settings import CONFIG
from utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class StrategyRating:
    """User rating for a strategy."""
    strategy_id: str
    rating: int  # 1-5 stars
    review: str = ""
    author: str = "anonymous"
    timestamp: str = ""
    helpful_count: int = 0
    verified_user: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy_id": self.strategy_id,
            "rating": self.rating,
            "review": self.review,
            "author": self.author,
            "timestamp": self.timestamp or datetime.now().isoformat(),
            "helpful_count": self.helpful_count,
            "verified_user": self.verified_user,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StrategyRating":
        return cls(
            strategy_id=data.get("strategy_id", ""),
            rating=max(1, min(5, int(data.get("rating", 3)))),
            review=data.get("review", ""),
            author=data.get("author", "anonymous"),
            timestamp=data.get("timestamp", ""),
            helpful_count=int(data.get("helpful_count", 0)),
            verified_user=bool(data.get("verified_user", False)),
        )


@dataclass
class StrategyPerformance:
    """Performance metrics for a strategy."""
    strategy_id: str
    total_trades: int = 0
    win_rate: float = 0.0
    avg_return: float = 0.0
    total_return: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    profit_factor: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    consecutive_wins: int = 0
    consecutive_losses: int = 0
    best_trade: float = 0.0
    worst_trade: float = 0.0
    avg_holding_period: float = 0.0  # in bars
    last_updated: str = ""
    backtest_period_days: int = 0
    sample_size: str = "small"  # small/medium/large

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy_id": self.strategy_id,
            "total_trades": self.total_trades,
            "win_rate": round(self.win_rate, 4),
            "avg_return": round(self.avg_return, 6),
            "total_return": round(self.total_return, 6),
            "sharpe_ratio": round(self.sharpe_ratio, 4),
            "max_drawdown": round(self.max_drawdown, 4),
            "profit_factor": round(self.profit_factor, 4),
            "avg_win": round(self.avg_win, 6),
            "avg_loss": round(self.avg_loss, 6),
            "consecutive_wins": self.consecutive_wins,
            "consecutive_losses": self.consecutive_losses,
            "best_trade": round(self.best_trade, 6),
            "worst_trade": round(self.worst_trade, 6),
            "avg_holding_period": round(self.avg_holding_period, 2),
            "last_updated": self.last_updated or datetime.now().isoformat(),
            "backtest_period_days": self.backtest_period_days,
            "sample_size": self.sample_size,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StrategyPerformance":
        return cls(
            strategy_id=data.get("strategy_id", ""),
            total_trades=int(data.get("total_trades", 0)),
            win_rate=float(data.get("win_rate", 0)),
            avg_return=float(data.get("avg_return", 0)),
            total_return=float(data.get("total_return", 0)),
            sharpe_ratio=float(data.get("sharpe_ratio", 0)),
            max_drawdown=float(data.get("max_drawdown", 0)),
            profit_factor=float(data.get("profit_factor", 0)),
            avg_win=float(data.get("avg_win", 0)),
            avg_loss=float(data.get("avg_loss", 0)),
            consecutive_wins=int(data.get("consecutive_wins", 0)),
            consecutive_losses=int(data.get("consecutive_losses", 0)),
            best_trade=float(data.get("best_trade", 0)),
            worst_trade=float(data.get("worst_trade", 0)),
            avg_holding_period=float(data.get("avg_holding_period", 0)),
            last_updated=data.get("last_updated", ""),
            backtest_period_days=int(data.get("backtest_period_days", 0)),
            sample_size=data.get("sample_size", "small"),
        )


@dataclass
class StrategyEntry:
    """Complete strategy entry with all metadata."""
    id: str
    name: str
    version: str
    author: str
    category: str
    risk_level: str
    intervals: list[str]
    min_bars: int
    description: str
    tags: list[str]
    file: str
    sha256: str
    enabled_by_default: bool = False
    installed: bool = False
    integrity: str = "unknown"
    enabled: bool = False
    rating_avg: float = 0.0
    rating_count: int = 0
    download_count: int = 0
    performance: StrategyPerformance | None = None
    last_updated: str = ""
    created_at: str = ""
    license: str = "MIT"
    website: str = ""
    documentation: str = ""
    params: dict[str, Any] = field(default_factory=dict)
    weight: float = 1.0
    _resolved_file: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "version": self.version,
            "author": self.author,
            "category": self.category,
            "risk_level": self.risk_level,
            "intervals": self.intervals,
            "min_bars": self.min_bars,
            "description": self.description,
            "tags": self.tags,
            "file": self.file,
            "sha256": self.sha256,
            "enabled_by_default": self.enabled_by_default,
            "installed": self.installed,
            "integrity": self.integrity,
            "enabled": self.enabled,
            "rating_avg": round(self.rating_avg, 2),
            "rating_count": self.rating_count,
            "download_count": self.download_count,
            "performance": self.performance.to_dict() if self.performance else None,
            "last_updated": self.last_updated,
            "created_at": self.created_at,
            "license": self.license,
            "website": self.website,
            "documentation": self.documentation,
            "params": self.params,
            "weight": self.weight,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StrategyEntry":
        perf_data = data.get("performance")
        return cls(
            id=data.get("id", ""),
            name=data.get("name", ""),
            version=data.get("version", "1.0.0"),
            author=data.get("author", "Unknown"),
            category=data.get("category", "General"),
            risk_level=data.get("risk_level", "medium"),
            intervals=list(data.get("intervals", [])),
            min_bars=int(data.get("min_bars", 60)),
            description=data.get("description", ""),
            tags=list(data.get("tags", [])),
            file=data.get("file", ""),
            sha256=data.get("sha256", ""),
            enabled_by_default=bool(data.get("enabled_by_default", False)),
            installed=bool(data.get("installed", False)),
            integrity=data.get("integrity", "unknown"),
            enabled=bool(data.get("enabled", False)),
            rating_avg=float(data.get("rating_avg", 0)),
            rating_count=int(data.get("rating_count", 0)),
            download_count=int(data.get("download_count", 0)),
            performance=StrategyPerformance.from_dict(perf_data) if perf_data else None,
            last_updated=data.get("last_updated", ""),
            created_at=data.get("created_at", ""),
            license=data.get("license", "MIT"),
            website=data.get("website", ""),
            documentation=data.get("documentation", ""),
            params=dict(data.get("params", {})),
            weight=float(data.get("weight", 1.0)),
            _resolved_file=data.get("_resolved_file", ""),
        )


class StrategyMarketplace:
    """
    Enhanced local strategy catalog with ratings, performance tracking,
    versioning, import/export, and sharing capabilities.
    """

    CATEGORIES = [
        "Trend Following",
        "Mean Reversion",
        "Momentum",
        "Breakout",
        "Scalping",
        "Swing Trading",
        "Arbitrage",
        "Market Making",
        "Machine Learning",
        "Sentiment-Based",
        "Multi-Factor",
        "Event-Driven",
        "Technical Analysis",
        "Statistical",
        "Pattern Recognition",
    ]

    RISK_LEVELS = ["very_low", "low", "medium", "high", "very_high"]

    def __init__(self, strategies_dir: Path | None = None) -> None:
        base = Path(getattr(CONFIG, "base_dir", Path(".")))
        self._dir = Path(strategies_dir) if strategies_dir else (base / "strategies")
        self._manifest_path = self._dir / "marketplace.json"
        self._enabled_path = self._dir / "enabled.json"
        self._ratings_path = self._dir / "ratings.json"
        self._performance_path = self._dir / "performance.json"
        self._imports_path = self._dir / "imports.json"
        self._dir.mkdir(parents=True, exist_ok=True)

        # Initialize default files
        self._init_manifest()
        self._init_ratings()
        self._init_performance()

    @property
    def manifest_path(self) -> Path:
        return self._manifest_path

    @property
    def enabled_path(self) -> Path:
        return self._enabled_path

    @property
    def ratings_path(self) -> Path:
        return self._ratings_path

    @property
    def performance_path(self) -> Path:
        return self._performance_path

    def _init_manifest(self) -> None:
        """Initialize manifest file if not exists."""
        if not self._manifest_path.exists():
            self._write_json(self._manifest_path, {
                "version": 2,
                "updated_at": datetime.now().isoformat(),
                "strategies": [],
            })

    def _init_ratings(self) -> None:
        """Initialize ratings file if not exists."""
        if not self._ratings_path.exists():
            self._write_json(self._ratings_path, {
                "ratings": [],
                "strategy_averages": {},
            })

    def _init_performance(self) -> None:
        """Initialize performance file if not exists."""
        if not self._performance_path.exists():
            self._write_json(self._performance_path, {
                "performance": {},
                "leaderboard": [],
                "last_updated": datetime.now().isoformat(),
            })

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

    # ==================== STRATEGY LISTING ====================

    def list_entries(self) -> list[StrategyEntry]:
        """List all strategies with full metadata."""
        raw = self._read_json(self._manifest_path, {"strategies": []})
        items = list(raw.get("strategies", []) or [])
        enabled_set = set(self.get_enabled_ids())
        ratings_data = self._read_json(self._ratings_path, {"ratings": [], "strategy_averages": {}})
        perf_data = self._read_json(self._performance_path, {"performance": {}})
        strategy_averages = ratings_data.get("strategy_averages", {})

        out: list[StrategyEntry] = []
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

            # Get rating average
            rating_avg = 0.0
            rating_count = 0
            if sid in strategy_averages:
                avg_data = strategy_averages[sid]
                rating_avg = float(avg_data.get("average", 0))
                rating_count = int(avg_data.get("count", 0))

            # Get performance data
            performance = None
            if sid in perf_data.get("performance", {}):
                perf_entry = perf_data["performance"][sid]
                performance = StrategyPerformance.from_dict(perf_entry)

            item = StrategyEntry.from_dict(entry)
            item.installed = installed
            item.integrity = integrity
            item.enabled = sid in enabled_set
            item.rating_avg = rating_avg
            item.rating_count = rating_count
            item.performance = performance
            if file_path:
                item._resolved_file = str(file_path)
            out.append(item)

        return out

    def get_enabled_ids(self) -> list[str]:
        """Get list of enabled strategy IDs."""
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
        """Save enabled strategy IDs."""
        valid_entries = {str(item.id): item for item in self.list_entries()}
        clean = []
        seen = set()
        for sid in strategy_ids:
            s = str(sid).strip()
            if not s or s in seen:
                continue
            entry = valid_entries.get(s)
            if not entry:
                continue
            if not entry.installed:
                continue
            if entry.integrity in ("mismatch", "missing", "error"):
                continue
            seen.add(s)
            clean.append(s)
        self._write_json(self._enabled_path, {"enabled": clean})

    def get_enabled_files(self) -> list[Path]:
        """Get list of enabled strategy file paths."""
        return [
            Path(item._resolved_file)
            for item in self.get_enabled_entries()
            if item._resolved_file
        ]

    def get_enabled_entries(self) -> list[StrategyEntry]:
        """Get enabled strategy entries with resolved file paths."""
        enabled = set(self.get_enabled_ids())
        out: list[StrategyEntry] = []
        seen_paths: set[str] = set()
        for item in self.list_entries():
            if item.id not in enabled:
                continue
            integrity = str(item.integrity).lower()
            if integrity not in ("ok", "unverified"):
                continue
            if not item._resolved_file:
                continue
            file_path = Path(item._resolved_file)
            if not file_path.exists() or not file_path.is_file():
                continue
            key = str(file_path)
            if key in seen_paths:
                continue
            seen_paths.add(key)
            out.append(item)
        return out

    def get_integrity_summary(self) -> dict[str, int]:
        """Get integrity check summary."""
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
            key = str(item.integrity)
            if key not in summary:
                key = "unknown"
            summary[key] += 1
        return summary

    # ==================== RATINGS SYSTEM ====================

    def add_rating(self, rating: StrategyRating) -> None:
        """Add or update a strategy rating."""
        data = self._read_json(self._ratings_path, {"ratings": [], "strategy_averages": {}})
        ratings_list = data.get("ratings", [])
        strategy_averages = data.get("strategy_averages", {})

        # Remove existing rating from same author
        ratings_list = [
            r for r in ratings_list
            if not (r.get("strategy_id") == rating.strategy_id and r.get("author") == rating.author)
        ]

        # Add new rating
        ratings_list.append(rating.to_dict())

        # Recalculate average
        strategy_ratings = [r for r in ratings_list if r.get("strategy_id") == rating.strategy_id]
        if strategy_ratings:
            avg = sum(r.get("rating", 3) for r in strategy_ratings) / len(strategy_ratings)
            strategy_averages[rating.strategy_id] = {
                "average": round(avg, 2),
                "count": len(strategy_ratings),
            }

        data["ratings"] = ratings_list
        data["strategy_averages"] = strategy_averages
        data["last_updated"] = datetime.now().isoformat()
        self._write_json(self._ratings_path, data)

    def get_strategy_ratings(self, strategy_id: str) -> list[StrategyRating]:
        """Get all ratings for a strategy."""
        data = self._read_json(self._ratings_path, {"ratings": []})
        ratings = [
            StrategyRating.from_dict(r)
            for r in data.get("ratings", [])
            if r.get("strategy_id") == strategy_id
        ]
        return sorted(ratings, key=lambda r: r.timestamp or "", reverse=True)

    def get_rating_summary(self, strategy_id: str) -> dict[str, Any]:
        """Get rating summary for a strategy."""
        data = self._read_json(self._ratings_path, {"strategy_averages": {}})
        averages = data.get("strategy_averages", {})
        if strategy_id in averages:
            return averages[strategy_id]
        return {"average": 0.0, "count": 0}

    def get_top_rated_strategies(self, min_ratings: int = 3, limit: int = 10) -> list[StrategyEntry]:
        """Get top-rated strategies."""
        entries = [e for e in self.list_entries() if e.rating_count >= min_ratings]
        entries.sort(key=lambda e: e.rating_avg, reverse=True)
        return entries[:limit]

    # ==================== PERFORMANCE TRACKING ====================

    def update_performance(self, performance: StrategyPerformance) -> None:
        """Update strategy performance metrics."""
        data = self._read_json(self._performance_path, {"performance": {}, "leaderboard": []})
        data["performance"][performance.strategy_id] = performance.to_dict()
        data["last_updated"] = datetime.now().isoformat()

        # Update leaderboard
        leaderboard = []
        for sid, perf in data["performance"].items():
            if perf.get("total_trades", 0) >= 10:  # Minimum sample size
                leaderboard.append({
                    "strategy_id": sid,
                    "total_return": perf.get("total_return", 0),
                    "sharpe_ratio": perf.get("sharpe_ratio", 0),
                    "win_rate": perf.get("win_rate", 0),
                })

        leaderboard.sort(key=lambda x: x["sharpe_ratio"], reverse=True)
        data["leaderboard"] = leaderboard[:20]  # Top 20
        self._write_json(self._performance_path, data)

    def get_performance(self, strategy_id: str) -> StrategyPerformance | None:
        """Get performance metrics for a strategy."""
        data = self._read_json(self._performance_path, {"performance": {}})
        perf = data.get("performance", {}).get(strategy_id)
        if perf:
            return StrategyPerformance.from_dict(perf)
        return None

    def get_leaderboard(self, limit: int = 10, sort_by: str = "sharpe_ratio") -> list[dict[str, Any]]:
        """Get strategy performance leaderboard."""
        data = self._read_json(self._performance_path, {"leaderboard": []})
        leaderboard = list(data.get("leaderboard", []))

        if sort_by in ("total_return", "sharpe_ratio", "win_rate"):
            leaderboard.sort(key=lambda x: x.get(sort_by, 0), reverse=True)

        return leaderboard[:limit]

    def get_performance_summary(self) -> dict[str, Any]:
        """Get overall performance summary."""
        data = self._read_json(self._performance_path, {"performance": {}})
        performances = data.get("performance", {})

        if not performances:
            return {
                "total_strategies": 0,
                "avg_sharpe": 0.0,
                "avg_win_rate": 0.0,
                "avg_return": 0.0,
                "best_strategy": None,
            }

        sharpe_values = [p.get("sharpe_ratio", 0) for p in performances.values() if p.get("sharpe_ratio", 0) != 0]
        win_rates = [p.get("win_rate", 0) for p in performances.values() if p.get("total_trades", 0) > 0]
        returns = [p.get("total_return", 0) for p in performances.values()]

        best = max(performances.items(), key=lambda x: x[1].get("sharpe_ratio", 0), default=(None, {}))

        return {
            "total_strategies": len(performances),
            "avg_sharpe": round(sum(sharpe_values) / len(sharpe_values), 4) if sharpe_values else 0,
            "avg_win_rate": round(sum(win_rates) / len(win_rates), 4) if win_rates else 0,
            "avg_return": round(sum(returns) / len(returns), 6) if returns else 0,
            "best_strategy": best[0],
            "last_updated": data.get("last_updated", ""),
        }

    # ==================== IMPORT/EXPORT ====================

    def export_strategy(self, strategy_id: str, output_path: Path) -> bool:
        """Export a strategy to a zip file for sharing."""
        entry = next((e for e in self.list_entries() if e.id == strategy_id), None)
        if not entry or not entry.installed:
            return False

        strategy_file = self._resolve_strategy_file(entry.file)
        if not strategy_file or not strategy_file.exists():
            return False

        try:
            with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zf:
                # Add strategy file
                zf.write(strategy_file, strategy_file.name)

                # Add metadata
                metadata = {
                    "strategy_id": entry.id,
                    "name": entry.name,
                    "version": entry.version,
                    "author": entry.author,
                    "category": entry.category,
                    "description": entry.description,
                    "tags": entry.tags,
                    "params": entry.params,
                    "sha256": self._file_hash(strategy_file),
                    "exported_at": datetime.now().isoformat(),
                }
                zf.writestr("metadata.json", json.dumps(metadata, indent=2))

                # Add performance if available
                if entry.performance:
                    zf.writestr("performance.json", json.dumps(entry.performance.to_dict(), indent=2))

                # Add documentation if exists
                doc_file = strategy_file.with_suffix(".md")
                if doc_file.exists():
                    zf.write(doc_file, f"docs/{doc_file.name}")

            return True
        except Exception as e:
            log.error(f"Failed to export strategy: {e}")
            return False

    def import_strategy(self, zip_path: Path, verify_hash: bool = True) -> tuple[bool, str]:
        """Import a strategy from a zip file."""
        try:
            with zipfile.ZipFile(zip_path, 'r') as zf:
                # Extract and validate metadata
                if "metadata.json" not in zf.namelist():
                    return False, "Missing metadata.json"

                metadata = json.loads(zf.read("metadata.json"))
                expected_hash = metadata.get("sha256", "")

                # Find strategy file
                py_files = [f for f in zf.namelist() if f.endswith(".py")]
                if not py_files:
                    return False, "No Python strategy file found"

                strategy_file = py_files[0]
                content = zf.read(strategy_file)

                # Verify hash
                actual_hash = hashlib.sha256(content).hexdigest()
                if verify_hash and expected_hash and actual_hash != expected_hash:
                    return False, "Hash verification failed - file may be corrupted"

                # Save strategy file
                dest_path = self._dir / Path(strategy_file).name
                dest_path.write_bytes(content)

                # Update manifest
                self._add_to_manifest(metadata, dest_path.name, actual_hash)

                return True, f"Successfully imported {metadata.get('name', strategy_file)}"

        except zipfile.BadZipFile:
            return False, "Invalid zip file"
        except Exception as e:
            log.error(f"Failed to import strategy: {e}")
            return False, str(e)

    def _add_to_manifest(self, metadata: dict[str, Any], file_name: str, sha256: str) -> None:
        """Add strategy entry to manifest."""
        data = self._read_json(self._manifest_path, {"strategies": []})

        # Check if strategy already exists
        existing_idx = None
        for idx, s in enumerate(data.get("strategies", [])):
            if s.get("id") == metadata.get("strategy_id"):
                existing_idx = idx
                break

        entry = {
            "id": metadata.get("strategy_id", Path(file_name).stem),
            "name": metadata.get("name", Path(file_name).stem),
            "version": metadata.get("version", "1.0.0"),
            "author": metadata.get("author", "Unknown"),
            "category": metadata.get("category", "General"),
            "risk_level": metadata.get("risk_level", "medium"),
            "intervals": metadata.get("intervals", ["1d"]),
            "min_bars": metadata.get("min_bars", 60),
            "description": metadata.get("description", ""),
            "tags": metadata.get("tags", []),
            "file": file_name,
            "sha256": sha256,
            "enabled_by_default": False,
            "params": metadata.get("params", {}),
            "license": metadata.get("license", "MIT"),
        }

        if existing_idx is not None:
            data["strategies"][existing_idx] = entry
        else:
            data["strategies"].append(entry)

        data["updated_at"] = datetime.now().isoformat()
        self._write_json(self._manifest_path, data)

    # ==================== STRATEGY DISCOVERY ====================

    def get_strategies_by_category(self, category: str) -> list[StrategyEntry]:
        """Get strategies filtered by category."""
        return [
            e for e in self.list_entries()
            if e.category.lower() == category.lower()
        ]

    def get_strategies_by_risk(self, risk_level: str) -> list[StrategyEntry]:
        """Get strategies filtered by risk level."""
        return [
            e for e in self.list_entries()
            if e.risk_level.lower() == risk_level.lower()
        ]

    def search_strategies(self, query: str) -> list[StrategyEntry]:
        """Search strategies by name, description, or tags."""
        query_lower = query.lower()
        results = []
        for entry in self.list_entries():
            if (query_lower in entry.name.lower() or
                query_lower in entry.description.lower() or
                any(query_lower in tag.lower() for tag in entry.tags)):
                results.append(entry)
        return results

    def get_categories_stats(self) -> dict[str, int]:
        """Get strategy count per category."""
        stats: dict[str, int] = {}
        for entry in self.list_entries():
            cat = entry.category
            stats[cat] = stats.get(cat, 0) + 1
        return stats

    # ==================== VERSION MANAGEMENT ====================

    def check_for_updates(self, local_version: str, remote_manifest_url: str | None = None) -> dict[str, Any]:
        """Check for strategy updates (placeholder for remote manifest)."""
        # In production, this would fetch from a remote manifest server
        # For now, return local version info
        return {
            "current_version": local_version,
            "update_available": False,
            "strategies_to_update": [],
            "message": "Update check requires remote manifest configuration",
        }

    def get_strategy_version(self, strategy_id: str) -> str | None:
        """Get version of a specific strategy."""
        entry = next((e for e in self.list_entries() if e.id == strategy_id), None)
        return entry.version if entry else None

    # ==================== UTILITIES ====================

    def get_strategy_details(self, strategy_id: str) -> dict[str, Any]:
        """Get complete details for a strategy."""
        entry = next((e for e in self.list_entries() if e.id == strategy_id), None)
        if not entry:
            return {"error": "Strategy not found"}

        return {
            "entry": entry.to_dict(),
            "ratings": [r.to_dict() for r in self.get_strategy_ratings(strategy_id)],
            "performance": entry.performance.to_dict() if entry.performance else None,
            "category_stats": self.get_categories_stats(),
        }

    def calculate_strategy_score(self, strategy_id: str) -> float:
        """Calculate overall strategy score (0-100) based on multiple factors."""
        entry = next((e for e in self.list_entries() if e.id == strategy_id), None)
        if not entry:
            return 0.0

        score = 50.0  # Base score

        # Rating component (up to 20 points)
        if entry.rating_count >= 3:
            score += (entry.rating_avg / 5.0) * 20

        # Performance component (up to 20 points)
        if entry.performance:
            perf = entry.performance
            if perf.total_trades >= 10:
                # Sharpe ratio contribution
                sharpe_score = min(10, perf.sharpe_ratio * 5)
                # Win rate contribution
                win_score = perf.win_rate * 10
                score += sharpe_score + win_score

        # Integrity bonus (up to 5 points)
        if entry.integrity == "ok":
            score += 5

        # Enabled bonus (up to 5 points)
        if entry.enabled:
            score += 5

        return min(100, max(0, score))

    def get_recommendations(self, risk_tolerance: str = "medium", preferred_categories: list[str] | None = None) -> list[StrategyEntry]:
        """Get strategy recommendations based on user preferences."""
        entries = self.list_entries()

        # Filter by risk
        risk_order = {"very_low": 0, "low": 1, "medium": 2, "high": 3, "very_high": 4}
        target_risk = risk_order.get(risk_tolerance.lower(), 2)

        filtered = []
        for entry in entries:
            entry_risk = risk_order.get(entry.risk_level.lower(), 2)
            if abs(entry_risk - target_risk) <= 1:  # Allow adjacent risk levels
                filtered.append(entry)

        # Filter by category if specified
        if preferred_categories:
            filtered = [
                e for e in filtered
                if e.category in preferred_categories
            ]

        # Sort by score
        filtered.sort(key=lambda e: self.calculate_strategy_score(e.id), reverse=True)

        return filtered[:10]
