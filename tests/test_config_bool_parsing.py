from config.settings import CONFIG, TradingConfig, _safe_dataclass_from_dict


def test_safe_dataclass_bool_parses_false_strings() -> None:
    cfg = TradingConfig()
    cfg.t_plus_1 = True

    warnings = _safe_dataclass_from_dict(cfg, {"t_plus_1": "false"})

    assert warnings == []
    assert cfg.t_plus_1 is False


def test_safe_dataclass_bool_parses_true_strings() -> None:
    cfg = TradingConfig()
    cfg.allow_short = False

    warnings = _safe_dataclass_from_dict(cfg, {"allow_short": "yes"})

    assert warnings == []
    assert cfg.allow_short is True


def test_safe_dataclass_bool_rejects_unknown_string() -> None:
    cfg = TradingConfig()
    cfg.allow_short = True

    warnings = _safe_dataclass_from_dict(cfg, {"allow_short": "maybe"})

    assert cfg.allow_short is True
    assert warnings
    assert "Bad value for bool field 'allow_short'" in warnings[0]


def test_load_from_env_runtime_lease_enabled_false(monkeypatch) -> None:
    monkeypatch.setenv("TRADING_RUNTIME_LEASE_ENABLED", "false")
    try:
        CONFIG.reload()
        assert CONFIG.security.enable_runtime_lease is False
    finally:
        monkeypatch.delenv("TRADING_RUNTIME_LEASE_ENABLED", raising=False)
        CONFIG.reload()


def test_load_from_env_runtime_lease_invalid_does_not_override(monkeypatch) -> None:
    monkeypatch.delenv("TRADING_RUNTIME_LEASE_ENABLED", raising=False)
    CONFIG.reload()
    baseline = bool(CONFIG.security.enable_runtime_lease)

    monkeypatch.setenv("TRADING_RUNTIME_LEASE_ENABLED", "maybe")
    try:
        CONFIG.reload()
        assert bool(CONFIG.security.enable_runtime_lease) == baseline
    finally:
        monkeypatch.delenv("TRADING_RUNTIME_LEASE_ENABLED", raising=False)
        CONFIG.reload()
