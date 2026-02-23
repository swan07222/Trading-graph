import pytest

from utils import security


def test_secure_storage_requires_cryptography(monkeypatch) -> None:
    monkeypatch.setattr(security, "CRYPTO_AVAILABLE", False)
    monkeypatch.setattr(
        security,
        "_CRYPTO_IMPORT_ERROR",
        ImportError("cryptography missing"),
    )

    with pytest.raises(RuntimeError, match="cryptography is required"):
        security.SecureStorage()


def test_get_secure_storage_propagates_fail_closed(monkeypatch) -> None:
    security.reset_security_singletons()
    monkeypatch.setattr(security, "CRYPTO_AVAILABLE", False)
    monkeypatch.setattr(
        security,
        "_CRYPTO_IMPORT_ERROR",
        ImportError("cryptography missing"),
    )

    with pytest.raises(RuntimeError, match="cryptography is required"):
        security.get_secure_storage()


def test_secure_storage_default_key_path_is_outside_storage_dir(monkeypatch, tmp_path) -> None:
    if security.Fernet is None:
        pytest.skip("cryptography unavailable")

    security.reset_security_singletons()
    monkeypatch.setenv("TRADING_SECURE_MASTER_KEY", "")
    monkeypatch.setenv("TRADING_SECURE_KEY_PATH", "")
    
    # Create the data_storage directory structure
    data_dir = tmp_path / "data_storage"
    data_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(
        security.CONFIG,
        "_data_dir_cached",
        data_dir,
        raising=False,
    )

    store = security.SecureStorage()
    try:
        # Key path should be outside data_storage by default
        assert "data_storage" not in str(store._key_path.parent)
    finally:
        store.close()


def test_secure_storage_env_master_key_avoids_local_key_file(monkeypatch, tmp_path) -> None:
    if security.Fernet is None:
        pytest.skip("cryptography unavailable")

    security.reset_security_singletons()
    
    # Create the data_storage directory structure
    data_dir = tmp_path / "data_storage"
    data_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(
        security.CONFIG,
        "_data_dir_cached",
        data_dir,
        raising=False,
    )
    monkeypatch.setenv(
        "TRADING_SECURE_MASTER_KEY",
        security.Fernet.generate_key().decode("utf-8"),
    )
    monkeypatch.setenv("TRADING_SECURE_KEY_PATH", "")

    store = security.SecureStorage()
    try:
        # When using env master key, local key file should not be created
        assert store._key_path.exists() is False
    finally:
        store.close()


def test_secure_storage_non_object_payload_resets_cache(monkeypatch, tmp_path) -> None:
    if security.Fernet is None:
        pytest.skip("cryptography unavailable")

    security.reset_security_singletons()
    
    # Create the data_storage directory structure
    data_dir = tmp_path / "data_storage"
    data_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(
        security.CONFIG,
        "_data_dir_cached",
        data_dir,
        raising=False,
    )
    monkeypatch.setenv(
        "TRADING_SECURE_MASTER_KEY",
        security.Fernet.generate_key().decode("utf-8"),
    )
    monkeypatch.setenv("TRADING_SECURE_KEY_PATH", "")

    store = security.SecureStorage()
    storage_path = store._storage_path
    cipher = store._cipher
    store.close()

    # Ensure parent directory exists before writing
    storage_path.parent.mkdir(parents=True, exist_ok=True)
    storage_path.write_bytes(cipher.encrypt(b"[]"))

    reloaded = security.SecureStorage()
    try:
        # Non-object payload (array) should reset cache to empty dict
        assert reloaded._cache == {}
    finally:
        reloaded.close()
