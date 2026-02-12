import gzip
import json
from datetime import date, timedelta

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


def test_audit_prune_respects_legal_hold(tmp_path):
    old = CONFIG._audit_dir_cached
    old_ret = getattr(CONFIG.security, "audit_retention_days", 365)
    old_auto = getattr(CONFIG.security, "audit_auto_prune", True)
    CONFIG._audit_dir_cached = tmp_path
    CONFIG.security.audit_retention_days = 1
    CONFIG.security.audit_auto_prune = False
    try:
        reset_security_singletons()
        audit = get_audit_log()

        old_day = date.today() - timedelta(days=10)
        old_file = tmp_path / f"audit_{old_day.isoformat()}_deadbeef.jsonl.gz"
        with gzip.open(old_file, "wt", encoding="utf-8") as f:
            f.write('{"timestamp":"2026-01-01T00:00:00","event_type":"x","user":"u","action":"a","details":{},"session_id":"s","prev_hash":"","record_hash":"h"}\n')

        assert audit.mark_legal_hold(old_file) is True
        stats = audit.prune_old_files(retention_days=1)
        assert stats["held"] >= 1
        assert old_file.exists()

        assert audit.unmark_legal_hold(old_file) is True
        stats2 = audit.prune_old_files(retention_days=1)
        assert stats2["deleted"] >= 1
        assert not old_file.exists()
    finally:
        reset_security_singletons()
        CONFIG._audit_dir_cached = old
        CONFIG.security.audit_retention_days = old_ret
        CONFIG.security.audit_auto_prune = old_auto
