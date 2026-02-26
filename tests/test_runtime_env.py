from config.runtime_env import env_flag, env_int, env_text


def test_env_flag_truthy_values(monkeypatch) -> None:
    monkeypatch.setenv("FLAG_ENABLED", "YeS")
    assert env_flag("FLAG_ENABLED") is True


def test_env_text_uses_empty_string_for_none_default(monkeypatch) -> None:
    monkeypatch.delenv("TEXT_VALUE", raising=False)
    assert env_text("TEXT_VALUE", None) == ""


def test_env_int_reads_valid_integer(monkeypatch) -> None:
    monkeypatch.setenv("COUNT_VALUE", "42")
    assert env_int("COUNT_VALUE", 7) == 42


def test_env_int_uses_default_for_invalid_values(monkeypatch) -> None:
    monkeypatch.setenv("COUNT_VALUE", "not-a-number")
    assert env_int("COUNT_VALUE", 7) == 7

    monkeypatch.setenv("COUNT_VALUE", "")
    assert env_int("COUNT_VALUE", 9) == 9

    monkeypatch.delenv("COUNT_VALUE", raising=False)
    assert env_int("COUNT_VALUE", 5) == 5
