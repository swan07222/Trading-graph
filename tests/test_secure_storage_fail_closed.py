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
