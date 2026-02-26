import types

import pytest

import main


def test_check_dependencies_gui_mode_skips_ml_stack(monkeypatch) -> None:
    seen = []

    def fake_find_spec(name):
        seen.append(name)
        if name in {"psutil", "PyQt6", "cryptography"}:
            return types.SimpleNamespace()
        return None

    monkeypatch.setattr(main, "find_spec", fake_find_spec)

    ok = main.check_dependencies(require_gui=True, require_ml=False)
    assert ok is True
    assert "torch" not in seen
    assert "numpy" not in seen
    assert "pandas" not in seen
    assert "sklearn" not in seen


def test_check_dependencies_ml_mode_requires_ml_stack(monkeypatch, capsys) -> None:
    def fake_find_spec(name):
        if name in {"psutil", "numpy", "pandas", "sklearn", "cryptography"}:
            return types.SimpleNamespace()
        return None

    monkeypatch.setattr(main, "find_spec", fake_find_spec)

    ok = main.check_dependencies(require_gui=False, require_ml=True)
    assert ok is False
    out = capsys.readouterr().out
    assert "torch" in out


def test_check_dependencies_no_longer_requires_live_broker_stack(
    monkeypatch, capsys
) -> None:
    def fake_find_spec(name):
        if name in {"psutil", "cryptography"}:
            return types.SimpleNamespace()
        return None

    monkeypatch.setattr(main, "find_spec", fake_find_spec)

    ok = main.check_dependencies(require_gui=False, require_ml=False)
    assert ok is True
    out = capsys.readouterr().out
    assert "easytrader" not in out


def test_parse_positive_int_csv_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="--opt-train-months"):
        main._parse_positive_int_csv("12,foo,0", "--opt-train-months")


def test_parse_probability_csv_rejects_out_of_range() -> None:
    with pytest.raises(ValueError, match="--opt-min-confidence"):
        main._parse_probability_csv("0.6,1.2", "--opt-min-confidence")


def test_ensure_backtest_optimize_success_raises_on_failed_status() -> None:
    with pytest.raises(RuntimeError, match="failed"):
        main._ensure_backtest_optimize_success(
            {"status": "failed", "errors": ["boom"]}
        )


def test_require_positive_int_rejects_non_positive_values() -> None:
    with pytest.raises(ValueError, match="--opt-top-k"):
        main._require_positive_int(0, "--opt-top-k")


def test_health_gate_violations_for_unhealthy_report() -> None:
    violations = main._health_gate_violations(
        {
            "status": "degraded",
            "execution_enabled": False,
            "can_trade": False,
            "degraded_mode": True,
            "slo_pass": False,
        }
    )
    assert "status=degraded" in violations
    assert "can_trade=false" not in violations
    assert "degraded_mode=true" in violations
    assert "slo_pass=false" in violations


def test_health_gate_violations_enforce_can_trade_when_execution_enabled() -> None:
    violations = main._health_gate_violations(
        {
            "status": "healthy",
            "execution_enabled": True,
            "can_trade": False,
        }
    )
    assert "can_trade=false" in violations


def test_ensure_health_gate_from_json_rejects_invalid_payload() -> None:
    with pytest.raises(RuntimeError, match="valid JSON"):
        main._ensure_health_gate_from_json("{not-json}")


def test_doctor_gate_violations_detect_missing_readiness() -> None:
    violations = main._doctor_gate_violations(
        {
            "dependencies": {
                "psutil": True,
                "numpy": True,
                "pandas": False,
                "sklearn": True,
                "requests": True,
                "cryptography": True,
            },
            "paths": {
                "data_dir": {"exists": True, "writable": True},
                "cache_dir": {"exists": True, "writable": False},
            },
            "config_validation_warnings": ["bad threshold"],
        }
    )
    assert any(v.startswith("missing_dependencies=") for v in violations)
    assert any(v.startswith("path_issues=") for v in violations)
    assert any(v.startswith("config_warnings=") for v in violations)


def test_doctor_gate_violations_ignores_removed_live_readiness_block() -> None:
    violations = main._doctor_gate_violations(
        {
            "dependencies": {
                "psutil": True,
                "numpy": True,
                "pandas": True,
                "sklearn": True,
                "requests": True,
                "cryptography": True,
            },
            "paths": {
                "data_dir": {"exists": True, "writable": True},
            },
            "config_validation_warnings": [],
            "institutional_readiness": {"pass": True},
            "doctor_live_enforced": True,
            "live_readiness": {
                "enforced": True,
                "pass": False,
                "missing_dependencies": ["easytrader"],
                "broker_path_exists": False,
            },
        }
    )
    assert violations == []
