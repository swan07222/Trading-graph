import gzip
import json

from config.settings import CONFIG
from utils.security import get_audit_log, reset_security_singletons


def test_audit_hash_chain_integrity_ok(tmp_path):
    old = CONFIG._audit_dir_cached
    CONFIG._audit_dir_cached = tmp_path
    try:
        reset_security_singletons()
        audit = get_audit_log()
        audit.log("test", "a1", {"x": 1})
        audit.log("test", "a2", {"x": 2})
        audit.close()

        reset_security_singletons()
        audit2 = get_audit_log()
        res = audit2.verify_integrity()
        assert res["ok"] is True
        assert res["checked"] >= 2
    finally:
        reset_security_singletons()
        CONFIG._audit_dir_cached = old


def test_audit_hash_chain_detects_tamper(tmp_path):
    old = CONFIG._audit_dir_cached
    CONFIG._audit_dir_cached = tmp_path
    try:
        reset_security_singletons()
        audit = get_audit_log()
        audit.log("test", "a1", {"x": 1})
        audit.log("test", "a2", {"x": 2})
        audit.close()

        files = list(tmp_path.glob("audit_*.jsonl.gz"))
        assert files, "audit file should exist"

        path = files[0]
        with gzip.open(path, "rt", encoding="utf-8") as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        assert len(lines) >= 2

        rec0 = json.loads(lines[0])
        rec0["record_hash"] = "tampered"
        lines[0] = json.dumps(rec0)

        with gzip.open(path, "wt", encoding="utf-8") as f:
            for ln in lines:
                f.write(ln + "\n")

        reset_security_singletons()
        audit2 = get_audit_log()
        res = audit2.verify_integrity()
        assert res["ok"] is False
        assert res["reason"] in ("record_hash_mismatch", "chain_break")
    finally:
        reset_security_singletons()
        CONFIG._audit_dir_cached = old
