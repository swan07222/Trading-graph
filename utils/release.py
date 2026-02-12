from __future__ import annotations

import hashlib
import hmac
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def build_release_manifest(artifacts: List[Path], version: str) -> Dict:
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
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "algorithm": "sha256",
        "artifacts": rows,
    }


def sign_manifest_hmac(manifest: Dict, secret: str) -> Dict:
    payload = json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    sig = hmac.new(secret.encode("utf-8"), payload.encode("utf-8"), hashlib.sha256).hexdigest()
    out = dict(manifest)
    out["signature"] = {"type": "hmac-sha256", "value": sig}
    return out
