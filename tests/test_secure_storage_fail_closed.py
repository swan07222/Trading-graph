import pytest

from utils import security


def test_secure_storage_requires_cryptography(monkeypatch):
    monkeypatch.setattr(security, "CRYPTO_AVAILABLE", False)
    monkeypatch.setattr(
        security,
        "_CRYPTO_IMPORT_ERROR",
        ImportError("cryptography missing"),
    )

    with pytest.raises(RuntimeError, match="cryptography is required"):
        security.SecureStorage()


def test_get_secure_storage_propagates_fail_closed(monkeypatch):
    security.reset_security_singletons()
    monkeypatch.setattr(security, "CRYPTO_AVAILABLE", False)
    monkeypatch.setattr(
        security,
        "_CRYPTO_IMPORT_ERROR",
        ImportError("cryptography missing"),
    )

    with pytest.raises(RuntimeError, match="cryptography is required"):
        security.get_secure_storage()


def test_secure_storage_default_key_path_is_outside_storage_dir(monkeypatch, tmp_path):
    if security.Fernet is None:
        pytest.skip("cryptography unavailable")

    security.reset_security_singletons()
    monkeypatch.setenv("TRADING_SECURE_MASTER_KEY", "")
    monkeypatch.setenv("TRADING_SECURE_KEY_PATH", "")
    monkeypatch.setattr(
        security.CONFIG,
        "_data_dir_cached",
        tmp_path / "data_storage",
        raising=False,
    )

    store = security.SecureStorage()
    try:
        assert store._storage_path.parent != store._key_path.parent
    finally:
        store.close()


def test_secure_storage_env_master_key_avoids_local_key_file(monkeypatch, tmp_path):
    if security.Fernet is None:
        pytest.skip("cryptography unavailable")

    security.reset_security_singletons()
    monkeypatch.setattr(
        security.CONFIG,
        "_data_dir_cached",
        tmp_path / "data_storage",
        raising=False,
    )
    monkeypatch.setenv(
        "TRADING_SECURE_MASTER_KEY",
        security.Fernet.generate_key().decode("utf-8"),
    )
    monkeypatch.setenv("TRADING_SECURE_KEY_PATH", "")

    store = security.SecureStorage()
    try:
        assert store._key_path.exists() is False
    finally:
        store.close()
