from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.institutional import collect_institutional_readiness


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate regulated institutional-readiness controls"
    )
    parser.add_argument(
        "--output",
        default="",
        help="Optional output JSON path",
    )
    args = parser.parse_args()

    report = collect_institutional_readiness()
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    print(rendered)

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(rendered + "\n", encoding="utf-8")
        print(f"regulatory readiness report written: {out}")

    return 0 if bool(report.get("pass", False)) else 1


if __name__ == "__main__":
    raise SystemExit(main())
