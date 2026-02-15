from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from typing import Any

import requests


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _full_url(base_url: str, path: str) -> str:
    return f"{base_url.rstrip('/')}{path}"


def _probe(
    session: requests.Session,
    url: str,
    *,
    timeout_seconds: float,
    expect_json: bool,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "url": url,
        "ok": False,
        "status_code": 0,
        "error": "",
    }
    try:
        resp = session.get(url, timeout=timeout_seconds)
        result["status_code"] = int(resp.status_code)
        if resp.status_code != 200:
            result["error"] = f"http_{resp.status_code}"
            return result
        if expect_json:
            payload = resp.json()
            if not isinstance(payload, dict):
                result["error"] = "json_payload_not_object"
                return result
            result["json_keys"] = sorted(payload.keys())
        else:
            text = resp.text
            result["body_length"] = len(text)
            if not text.strip():
                result["error"] = "empty_body"
                return result
        result["ok"] = True
        return result
    except Exception as exc:
        result["error"] = str(exc)
        return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe runtime observability endpoints")
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:9090",
        help="Metrics/API base URL",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=3.0,
        help="HTTP timeout per endpoint",
    )
    parser.add_argument(
        "--api-key-env",
        default="TRADING_HTTP_API_KEY",
        help="Env var for optional API key",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Optional output JSON path",
    )
    args = parser.parse_args()

    session = requests.Session()
    session.headers.update({"User-Agent": "trading-graph-observability-probe/1.0"})
    api_key = str(os.environ.get(args.api_key_env, "")).strip()
    if api_key:
        session.headers.update({"X-API-Key": api_key})

    checks = [
        ("/healthz", False),
        ("/metrics", False),
        ("/api/v1/health", True),
        ("/api/v1/execution", True),
        ("/api/v1/dashboard/full?limit=5", True),
    ]

    results: list[dict[str, Any]] = []
    failures: list[str] = []
    for path, expect_json in checks:
        url = _full_url(args.base_url, path)
        row = _probe(
            session,
            url,
            timeout_seconds=float(args.timeout_seconds),
            expect_json=expect_json,
        )
        row["path"] = path
        results.append(row)
        if not row.get("ok"):
            failures.append(path)

    status = "pass" if not failures else "fail"
    report: dict[str, Any] = {
        "status": status,
        "started_at": _utc_now_iso(),
        "base_url": str(args.base_url),
        "api_key_env": str(args.api_key_env),
        "checks": results,
        "failed_paths": failures,
    }

    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    print(rendered)

    if args.output:
        from pathlib import Path

        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(rendered + "\n", encoding="utf-8")
        print(f"observability report written: {out}")

    return 0 if status == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
