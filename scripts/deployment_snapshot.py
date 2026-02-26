from __future__ import annotations

import argparse
import hashlib
import io
import json
import subprocess
import sys
import tarfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

DEFAULT_RUNTIME_PATHS: tuple[str, ...] = (
    "config.json",
    "config/security_policy.json",
    "strategies/enabled.json",
    "data_storage/orders.db",
    "data_storage/orders.db-wal",
    "data_storage/orders.db-shm",
)
MANIFEST_MEMBER_NAME = ".snapshot_manifest.json"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _safe_member_path(raw: str) -> Path:
    path = Path(str(raw))
    if path.is_absolute() or path.drive:
        raise ValueError(f"Unsafe absolute member path: {raw}")
    if ".." in path.parts:
        raise ValueError(f"Unsafe traversal member path: {raw}")
    return path


def _collect_files(root: Path, relative_paths: list[str]) -> list[Path]:
    files: list[Path] = []
    seen: set[str] = set()
    for raw in relative_paths:
        rel = str(raw or "").strip().replace("\\", "/")
        if not rel:
            continue
        candidate = root / rel
        if candidate.is_file():
            key = candidate.resolve().as_posix()
            if key not in seen:
                seen.add(key)
                files.append(candidate)
            continue
        if candidate.is_dir():
            for child in sorted(candidate.rglob("*")):
                if not child.is_file():
                    continue
                key = child.resolve().as_posix()
                if key in seen:
                    continue
                seen.add(key)
                files.append(child)
    return sorted(files, key=lambda p: p.as_posix().lower())


def create_snapshot(
    root: Path,
    snapshot_dir: Path,
    include_paths: list[str] | None = None,
    include_models: bool = False,
    tag: str | None = None,
) -> dict[str, Any]:
    include = list(DEFAULT_RUNTIME_PATHS)
    if include_paths:
        include.extend(include_paths)
    if include_models:
        include.append("models_saved")

    files = _collect_files(root=root, relative_paths=include)
    if not files:
        raise RuntimeError("No files found for snapshot")

    stamp = str(tag or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"))
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    archive_path = snapshot_dir / f"snapshot_{stamp}.tar.gz"
    sidecar_manifest_path = snapshot_dir / f"snapshot_{stamp}.manifest.json"

    manifest: dict[str, Any] = {
        "snapshot_id": f"snapshot_{stamp}",
        "created_at": datetime.now(UTC).isoformat(),
        "root": str(root),
        "archive": str(archive_path),
        "files": [],
    }

    for fpath in files:
        rel = fpath.relative_to(root).as_posix()
        manifest["files"].append(
            {
                "path": rel,
                "size": int(fpath.stat().st_size),
                "sha256": sha256_file(fpath),
            }
        )

    with tarfile.open(archive_path, "w:gz") as tf:
        for fpath in files:
            tf.add(fpath, arcname=fpath.relative_to(root).as_posix(), recursive=False)
        manifest_bytes = json.dumps(
            manifest,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        info = tarfile.TarInfo(name=MANIFEST_MEMBER_NAME)
        info.size = len(manifest_bytes)
        info.mtime = int(datetime.now(UTC).timestamp())
        tf.addfile(info, io.BytesIO(manifest_bytes))

    sidecar_manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return {
        "status": "ok",
        "archive": str(archive_path),
        "manifest": str(sidecar_manifest_path),
        "file_count": len(files),
        "snapshot_id": manifest["snapshot_id"],
    }


def _read_embedded_manifest(tf: tarfile.TarFile) -> dict[str, Any]:
    try:
        member = tf.getmember(MANIFEST_MEMBER_NAME)
    except KeyError:
        return {}
    fh = tf.extractfile(member)
    if fh is None:
        return {}
    try:
        payload = json.loads(fh.read().decode("utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def restore_snapshot(
    root: Path,
    archive: Path,
    *,
    dry_run: bool = False,
    confirm: bool = False,
) -> dict[str, Any]:
    if not archive.exists():
        raise FileNotFoundError(f"Archive not found: {archive}")
    if not dry_run and not confirm:
        raise RuntimeError("Refusing restore without --confirm")

    restored: list[str] = []
    with tarfile.open(archive, "r:gz") as tf:
        manifest = _read_embedded_manifest(tf)
        expected_hash = {
            str(item.get("path", "")): str(item.get("sha256", ""))
            for item in list(manifest.get("files", []))
            if isinstance(item, dict) and item.get("path")
        }

        for member in tf.getmembers():
            if not member.isfile():
                continue
            if member.name == MANIFEST_MEMBER_NAME:
                continue
            rel = _safe_member_path(member.name)
            restored.append(rel.as_posix())
            if dry_run:
                continue

            source = tf.extractfile(member)
            if source is None:
                raise RuntimeError(f"Failed to read archive member: {member.name}")
            payload = source.read()
            dest = root / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(payload)

            expected = expected_hash.get(rel.as_posix(), "").strip().lower()
            if expected:
                actual = hashlib.sha256(payload).hexdigest().lower()
                if actual != expected:
                    raise RuntimeError(
                        f"Checksum mismatch after restore for {rel}: "
                        f"expected {expected}, got {actual}"
                    )

    return {
        "status": "ok",
        "archive": str(archive),
        "dry_run": bool(dry_run),
        "restored_files": restored,
        "restored_count": len(restored),
    }


def _run_step(name: str, cmd: list[str], cwd: Path) -> dict[str, Any]:
    started = time.monotonic()
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        elapsed = time.monotonic() - started
        return {
            "name": name,
            "command": cmd,
            "exit_code": int(proc.returncode),
            "duration_seconds": round(float(elapsed), 3),
            "ok": proc.returncode == 0,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
        }
    except Exception as exc:
        elapsed = time.monotonic() - started
        return {
            "name": name,
            "command": cmd,
            "exit_code": 2,
            "duration_seconds": round(float(elapsed), 3),
            "ok": False,
            "stdout": "",
            "stderr": str(exc),
        }


def _run_post_restore_verification(
    root: Path,
    *,
    profile: str,
    observability_url: str = "",
    soak_minutes: float = 0.0,
    soak_mode: str = "paper",
    soak_symbols: str = "",
    allow_live: bool = False,
) -> dict[str, Any]:
    py = sys.executable
    steps: list[dict[str, Any]] = []

    preflight_cmd = [
        py,
        str(root / "scripts/release_preflight.py"),
        "--profile",
        str(profile),
    ]
    obs = str(observability_url or "").strip()
    if obs:
        preflight_cmd.extend(["--observability-url", obs])
    steps.append(_run_step("release_preflight", preflight_cmd, cwd=root))

    soak_minutes_val = float(soak_minutes or 0.0)
    if soak_minutes_val > 0.0:
        mode = str(soak_mode or "paper").strip().lower()
        if mode == "live" and not bool(allow_live):
            steps.append(
                {
                    "name": "soak_smoke",
                    "command": [],
                    "exit_code": 2,
                    "duration_seconds": 0.0,
                    "ok": False,
                    "stdout": "",
                    "stderr": (
                        "post-restore soak in live mode requires --post-allow-live"
                    ),
                }
            )
        else:
            soak_cmd = [
                py,
                str(root / "scripts/soak_broker_e2e.py"),
                "--mode",
                mode,
                "--duration-minutes",
                str(soak_minutes_val),
                "--poll-seconds",
                "5",
            ]
            symbols = str(soak_symbols or "").strip()
            if symbols:
                soak_cmd.extend(["--symbols", symbols])
            if mode == "live":
                soak_cmd.append("--allow-live")
            steps.append(_run_step("soak_smoke", soak_cmd, cwd=root))

    failed = [step for step in steps if not bool(step.get("ok", False))]
    return {
        "status": "pass" if not failed else "fail",
        "profile": str(profile),
        "steps": steps,
        "failed_steps": [str(step.get("name", "")) for step in failed],
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Deployment snapshot + rollback helper"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    create_parser = subparsers.add_parser("create", help="Create snapshot archive")
    create_parser.add_argument(
        "--snapshot-dir",
        default="backups",
        help="Output directory for snapshot archives",
    )
    create_parser.add_argument(
        "--tag",
        default="",
        help="Optional snapshot tag (default UTC timestamp)",
    )
    create_parser.add_argument(
        "--include-path",
        action="append",
        default=[],
        help="Extra file/dir path to include (repeatable)",
    )
    create_parser.add_argument(
        "--include-models",
        action="store_true",
        help="Include models_saved/ recursively (can be large)",
    )

    restore_parser = subparsers.add_parser("restore", help="Restore from snapshot archive")
    restore_parser.add_argument(
        "--archive",
        required=True,
        help="Path to snapshot tar.gz archive",
    )
    restore_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show files that would be restored without writing",
    )
    restore_parser.add_argument(
        "--confirm",
        action="store_true",
        help="Required for actual restore",
    )
    restore_parser.add_argument(
        "--post-verify",
        action="store_true",
        help=(
            "After restore, run release preflight and optional smoke soak "
            "as a single command."
        ),
    )
    restore_parser.add_argument(
        "--post-verify-profile",
        default="quick",
        choices=["full", "quick"],
        help="Preflight profile used by --post-verify",
    )
    restore_parser.add_argument(
        "--post-observability-url",
        default="",
        help="Optional observability base URL passed to post-verify preflight",
    )
    restore_parser.add_argument(
        "--post-soak-minutes",
        type=float,
        default=0.0,
        help=(
            "Optional smoke soak duration in minutes after restore "
            "(0 disables soak)."
        ),
    )
    restore_parser.add_argument(
        "--post-soak-mode",
        default="paper",
        choices=["simulation", "paper", "live"],
        help="Trading mode used by optional post-restore smoke soak",
    )
    restore_parser.add_argument(
        "--post-soak-symbols",
        default="",
        help="Optional comma-separated symbols for post-restore smoke soak",
    )
    restore_parser.add_argument(
        "--post-allow-live",
        action="store_true",
        help="Required when --post-soak-mode live is used",
    )

    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent

    if args.command == "create":
        result = create_snapshot(
            root=root,
            snapshot_dir=Path(args.snapshot_dir),
            include_paths=[str(p) for p in (args.include_path or [])],
            include_models=bool(args.include_models),
            tag=str(args.tag or "").strip() or None,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0

    if args.command == "restore":
        result = restore_snapshot(
            root=root,
            archive=Path(args.archive),
            dry_run=bool(args.dry_run),
            confirm=bool(args.confirm),
        )
        exit_code = 0
        if bool(args.post_verify) and not bool(args.dry_run):
            post = _run_post_restore_verification(
                root=root,
                profile=str(args.post_verify_profile),
                observability_url=str(args.post_observability_url or ""),
                soak_minutes=float(args.post_soak_minutes or 0.0),
                soak_mode=str(args.post_soak_mode or "paper"),
                soak_symbols=str(args.post_soak_symbols or ""),
                allow_live=bool(args.post_allow_live),
            )
            result["post_verify"] = post
            if str(post.get("status", "")).lower() != "pass":
                exit_code = 1
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return int(exit_code)

    raise RuntimeError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
