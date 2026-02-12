from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from utils.release import build_release_manifest, sign_manifest_hmac


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate release manifest for dist artifacts")
    parser.add_argument("--dist-dir", default="dist", help="Directory containing wheel/sdist files")
    parser.add_argument("--version", required=True, help="Release version (e.g. v1.2.3)")
    parser.add_argument("--output", default="dist/release_manifest.json", help="Output manifest path")
    parser.add_argument(
        "--sign-secret-env",
        default="RELEASE_MANIFEST_SECRET",
        help="Env var name containing optional HMAC secret",
    )
    args = parser.parse_args()

    dist_dir = Path(args.dist_dir)
    files = [p for p in dist_dir.glob("*") if p.is_file()]
    if not files:
        raise SystemExit(f"No artifacts found in {dist_dir}")

    manifest = build_release_manifest(files, version=args.version)
    secret = os.environ.get(args.sign_secret_env, "")
    if secret:
        manifest = sign_manifest_hmac(manifest, secret)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Manifest written: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
