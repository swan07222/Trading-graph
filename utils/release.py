from __future__ import annotations

import hashlib
import hmac
import json
import os
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def build_release_manifest(artifacts: list[Path], version: str) -> dict:
    rows = []
    for p in sorted(artifacts, key=lambda x: x.name):
        rows.append(
            {
                "name": p.name,
                "size": int(p.stat().st_size),
                "sha256": sha256_file(p),
            }
        )
    return {
        "version": str(version),
        "generated_at": datetime.now(UTC).isoformat(),
        "algorithm": "sha256",
        "build": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "source_revision": str(
                os.environ.get("GITHUB_SHA")
                or os.environ.get("CI_COMMIT_SHA")
                or ""
            ),
        },
        "artifacts": rows,
    }


def _canonical_manifest_payload(manifest: dict) -> str:
    payload_manifest = {k: v for k, v in dict(manifest).items() if k != "signature"}
    return json.dumps(
        payload_manifest,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def sign_manifest_hmac(manifest: dict, secret: str) -> dict:
    payload_manifest = {k: v for k, v in dict(manifest).items() if k != "signature"}
    payload = _canonical_manifest_payload(payload_manifest)
    sig = hmac.new(secret.encode("utf-8"), payload.encode("utf-8"), hashlib.sha256).hexdigest()
    out = dict(payload_manifest)
    out["signature"] = {"type": "hmac-sha256", "value": sig}
    return out


def verify_manifest_hmac(manifest: dict, secret: str) -> bool:
    sig = dict(manifest.get("signature") or {})
    if str(sig.get("type", "")).strip().lower() != "hmac-sha256":
        return False
    given = str(sig.get("value", "") or "").strip().lower()
    if not given:
        return False
    expected = hmac.new(
        secret.encode("utf-8"),
        _canonical_manifest_payload(manifest).encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    return hmac.compare_digest(given, expected)
