from __future__ import annotations

import argparse
import os
from pathlib import Path

from config.settings import CONFIG
from data.cache import get_cache


def _delete_pycache(root: Path) -> int:
    deleted = 0
    for p in root.rglob("__pycache__"):
        if not p.is_dir():
            continue
        for child in p.rglob("*"):
            if child.is_file():
                child.unlink(missing_ok=True)
        try:
            p.rmdir()
            deleted += 1
        except OSError:
            pass
    return deleted


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Manual cache deletion tool (guarded)."
    )
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Required flag to perform deletion.",
    )
    parser.add_argument(
        "--tier",
        choices=["l1", "l2", "l3", "all"],
        default="all",
        help="Cache tier to clear.",
    )
    parser.add_argument(
        "--older-than-hours",
        type=float,
        default=None,
        help="Only delete entries older than this age (disk tiers only).",
    )
    parser.add_argument(
        "--delete-pycache",
        action="store_true",
        help="Also delete __pycache__ directories in project root.",
    )
    args = parser.parse_args()

    if not args.confirm:
        print("Refusing to delete cache without --confirm")
        return 2

    os.environ["TRADING_MANUAL_CACHE_DELETE"] = "1"

    cache = get_cache()
    tier = None if args.tier == "all" else args.tier
    cache.clear(tier=tier, older_than_hours=args.older_than_hours)

    print(
        f"Cache clear completed: tier={args.tier}, "
        f"older_than_hours={args.older_than_hours}"
    )

    if args.delete_pycache:
        n = _delete_pycache(CONFIG.base_dir)
        print(f"Deleted __pycache__ dirs: {n}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
