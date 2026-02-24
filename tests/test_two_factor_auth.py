"""Tests for two-factor authentication module."""
import tempfile
from pathlib import Path

import pytest

from utils.two_factor_auth import (
    TOTPAuthenticator,
    TwoFactorAuth,
)


class TestTOTPAuthenticator:
    """Test TOTP authenticator."""

    def test_generate_secret(self) -> None:
        """Test secret generation."""
        auth = TOTPAuthenticator()
        secret = auth.generate_secret()
        
        # Secret should be 32 characters (20 bytes base32 encoded)
        assert len(secret) == 32
        # Should only contain valid base32 characters
        assert all(c in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ234567' for c in secret)

    def test_generate_totp(self) -> None:
        """Test TOTP code generation."""
        auth = TOTPAuthenticator()
        secret = auth.generate_secret()
        
        code = auth.generate_totp(secret)
        
        # Code should be 6 digits
        assert len(code) == 6
        assert code.isdigit()

    def test_verify_totp_valid(self) -> None:
        """Test verifying a valid TOTP code."""
        auth = TOTPAuthenticator()
        secret = auth.generate_secret()
        
        code = auth.generate_totp(secret)
        assert auth.verify_totp(secret, code) is True

    def test_verify_totp_invalid(self) -> None:
        """Test verifying an invalid TOTP code."""
        auth = TOTPAuthenticator()
        secret = auth.generate_secret()
        
        # Wrong code
        assert auth.verify_totp(secret, "000000") is False

    def test_verify_totp_time_window(self) -> None:
        """Test TOTP verification with time window."""
        auth = TOTPAuthenticator()
        secret = auth.generate_secret()
        
        # Get current code
        code = auth.generate_totp(secret)
        
        # Should verify within window
        assert auth.verify_totp(secret, code, window=1) is True

    def test_get_provisioning_uri(self) -> None:
        """Test provisioning URI generation."""
        auth = TOTPAuthenticator()
        secret = "JBSWY3DPEHPK3PXP"
        
        uri = auth.get_provisioning_uri(
            secret,
            "trader_001",
            "Trading Graph",
        )
        
        assert uri.startswith("otpauth://totp/")
        assert "Trading%20Graph" in uri or "Trading+Graph" in uri or "Trading Graph" in uri
        assert "secret=JBSWY3DPEHPK3PXP" in uri
        assert "trader_001" in uri

    def test_different_time_steps(self) -> None:
        """Test different time step configurations."""
        auth_30 = TOTPAuthenticator(time_step=30)
        auth_60 = TOTPAuthenticator(time_step=60)
        
        secret = auth_30.generate_secret()
        
        code_30 = auth_30.generate_totp(secret)
        code_60 = auth_60.generate_totp(secret)
        
        assert len(code_30) == 6
        assert len(code_60) == 6


class TestTwoFactorAuth:
    """Test two-factor authentication manager."""

    @pytest.fixture
    def tfa(self) -> TwoFactorAuth:
        """Create TFA instance with temp storage."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield TwoFactorAuth(storage_path=Path(tmpdir))

    def test_setup_2fa(self, tfa: TwoFactorAuth) -> None:
        """Test setting up 2FA for a user."""
        user_id = "trader_001"
        setup = tfa.setup_2fa(user_id)
        
        assert 'secret' in setup
        assert 'uri' in setup
        assert 'qr_code' in setup
        assert 'backup_codes' in setup
        assert len(setup['backup_codes']) == 10
        
        # User should not be enabled yet
        status = tfa.get_2fa_status(user_id)
        assert status['enabled'] is False

    def test_verify_setup(self, tfa: TwoFactorAuth) -> None:
        """Test verifying 2FA setup."""
        user_id = "trader_001"
        setup = tfa.setup_2fa(user_id)
        
        # Generate valid code
        auth = TOTPAuthenticator()
        code = auth.generate_totp(setup['secret'])
        
        # Verify setup
        assert tfa.verify_setup(user_id, code) is True
        
        # User should now be enabled
        status = tfa.get_2fa_status(user_id)
        assert status['enabled'] is True

    def test_verify_code_valid(self, tfa: TwoFactorAuth) -> None:
        """Test verifying a valid code."""
        user_id = "trader_001"
        setup = tfa.setup_2fa(user_id)
        
        # Enable 2FA
        auth = TOTPAuthenticator()
        code = auth.generate_totp(setup['secret'])
        tfa.verify_setup(user_id, code)
        
        # Verify new code
        new_code = auth.generate_totp(setup['secret'])
        assert tfa.verify_code(user_id, new_code) is True

    def test_verify_code_not_enabled(self, tfa: TwoFactorAuth) -> None:
        """Test verification when 2FA not enabled."""
        user_id = "trader_001"
        tfa.setup_2fa(user_id)
        
        # Should allow if not enabled
        assert tfa.verify_code(user_id, "123456") is True

    def test_verify_backup_code(self, tfa: TwoFactorAuth) -> None:
        """Test verifying a backup code."""
        user_id = "trader_001"
        setup = tfa.setup_2fa(user_id)
        
        # Enable 2FA
        auth = TOTPAuthenticator()
        code = auth.generate_totp(setup['secret'])
        tfa.verify_setup(user_id, code)
        
        # Use backup code
        backup_code = setup['backup_codes'][0]
        assert tfa.verify_code(user_id, backup_code, use_backup_code=True) is True
        
        # Backup code should be consumed
        status = tfa.get_2fa_status(user_id)
        assert status['backup_codes_remaining'] == 9

    def test_verify_backup_code_invalid(self, tfa: TwoFactorAuth) -> None:
        """Test verifying an invalid backup code."""
        user_id = "trader_001"
        setup = tfa.setup_2fa(user_id)
        
        # Enable 2FA
        auth = TOTPAuthenticator()
        code = auth.generate_totp(setup['secret'])
        tfa.verify_setup(user_id, code)
        
        # Invalid backup code
        assert tfa.verify_code(user_id, "INVALID", use_backup_code=True) is False

    def test_rate_limiting(self, tfa: TwoFactorAuth) -> None:
        """Test rate limiting after failed attempts."""
        user_id = "trader_001"
        setup = tfa.setup_2fa(user_id)
        
        # Enable 2FA
        auth = TOTPAuthenticator()
        tfa.verify_setup(user_id, auth.generate_totp(setup['secret']))
        
        # Multiple failed attempts
        for _ in range(5):
            tfa.verify_code(user_id, "000000")
        
        # Should be rate limited
        status = tfa.get_2fa_status(user_id)
        assert status['locked'] is True

    def test_disable_2fa(self, tfa: TwoFactorAuth) -> None:
        """Test disabling 2FA."""
        user_id = "trader_001"
        setup = tfa.setup_2fa(user_id)
        
        # Enable 2FA
        auth = TOTPAuthenticator()
        code = auth.generate_totp(setup['secret'])
        tfa.verify_setup(user_id, code)
        
        # Disable with valid code
        new_code = auth.generate_totp(setup['secret'])
        assert tfa.disable_2fa(user_id, new_code) is True
        
        # Should be disabled
        status = tfa.get_2fa_status(user_id)
        assert status['enabled'] is False

    def test_regenerate_backup_codes(self, tfa: TwoFactorAuth) -> None:
        """Test regenerating backup codes."""
        user_id = "trader_001"
        setup = tfa.setup_2fa(user_id)
        original_codes = setup['backup_codes'].copy()
        
        # Enable 2FA
        auth = TOTPAuthenticator()
        code = auth.generate_totp(setup['secret'])
        tfa.verify_setup(user_id, code)
        
        # Regenerate codes
        new_code = auth.generate_totp(setup['secret'])
        new_codes = tfa.regenerate_backup_codes(user_id, new_code)
        
        assert new_codes is not None
        assert len(new_codes) == 10
        assert new_codes != original_codes

    def test_get_2fa_status(self, tfa: TwoFactorAuth) -> None:
        """Test getting 2FA status."""
        user_id = "trader_001"
        
        # Not configured
        status = tfa.get_2fa_status(user_id)
        assert status['configured'] is False
        assert status['enabled'] is False
        
        # Setup but not enabled
        tfa.setup_2fa(user_id)
        status = tfa.get_2fa_status(user_id)
        assert status['configured'] is True
        assert status['enabled'] is False

    def test_persistence(self, tfa: TwoFactorAuth) -> None:
        """Test 2FA config persistence."""
        user_id = "trader_001"
        setup = tfa.setup_2fa(user_id)
        
        # Enable 2FA
        auth = TOTPAuthenticator()
        code = auth.generate_totp(setup['secret'])
        tfa.verify_setup(user_id, code)
        
        # Create new instance (should load from storage)
        tfa2 = TwoFactorAuth(storage_path=tfa.storage_path)
        status = tfa2.get_2fa_status(user_id)
        
        assert status['enabled'] is True
