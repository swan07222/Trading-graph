"""Two-Factor Authentication (2FA) Module.

Provides TOTP-based two-factor authentication for:
- User login
- Trade confirmation
- Sensitive operations (withdrawal, config changes)

Features:
- TOTP (Time-based One-Time Password) generation
- QR code setup for authenticator apps
- Backup codes for recovery
- Rate limiting for brute-force protection
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import struct
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from utils.logger import get_logger
from utils.security import get_secure_storage

log = get_logger(__name__)


@dataclass
class User2FA:
    """User 2FA configuration."""
    user_id: str
    enabled: bool = False
    secret: str = ""
    backup_codes: list[str] = field(default_factory=list)
    created_at: datetime | None = None
    last_used: datetime | None = None
    
    # Settings
    issuer: str = "Trading Graph"
    account_name: str = ""
    
    # Security
    failed_attempts: int = 0
    locked_until: datetime | None = None
    
    def __post_init__(self) -> None:
        if self.created_at is None:
            self.created_at = datetime.now()
        if not self.account_name:
            self.account_name = self.user_id


@dataclass
class BackupCode:
    """Backup code for account recovery."""
    code: str
    used: bool = False
    used_at: datetime | None = None
    created_at: datetime | None = None
    
    def __post_init__(self) -> None:
        if self.created_at is None:
            self.created_at = datetime.now()


class TOTPAuthenticator:
    """TOTP (Time-based One-Time Password) implementation.
    
    Compatible with Google Authenticator, Authy, Microsoft Authenticator, etc.
    """
    
    def __init__(self, time_step: int = 30, digits: int = 6) -> None:
        """Initialize TOTP authenticator.
        
        Args:
            time_step: Time step in seconds (default: 30)
            digits: Number of digits in code (default: 6)
        """
        self.time_step = time_step
        self.digits = digits
    
    def generate_secret(self) -> str:
        """Generate random 20-byte secret (160 bits)."""
        return base64.b32encode(os.urandom(20)).decode('utf-8')
    
    def generate_totp(self, secret: str, timestamp: int | None = None) -> str:
        """Generate TOTP code.
        
        Args:
            secret: Base32-encoded secret
            timestamp: Unix timestamp (default: current time)
        
        Returns:
            TOTP code as string
        """
        if timestamp is None:
            timestamp = int(time.time())
        
        # Calculate time counter
        counter = timestamp // self.time_step
        
        # Decode secret
        key = base64.b32decode(secret.upper())
        
        # Pack counter as big-endian 8-byte integer
        counter_bytes = struct.pack('>Q', counter)
        
        # HMAC-SHA1
        hmac_hash = hmac.new(key, counter_bytes, hashlib.sha1).digest()
        
        # Dynamic truncation
        offset = hmac_hash[-1] & 0x0F
        code_int = struct.unpack('>I', hmac_hash[offset:offset + 4])[0]
        code_int &= 0x7FFFFFFF
        
        # Get last N digits
        code = code_int % (10 ** self.digits)
        return str(code).zfill(self.digits)
    
    def verify_totp(
        self,
        secret: str,
        code: str,
        window: int = 1,
        timestamp: int | None = None,
    ) -> bool:
        """Verify TOTP code.
        
        Args:
            secret: Base32-encoded secret
            code: Code to verify
            window: Acceptable time drift (in steps, default: 1)
            timestamp: Unix timestamp (default: current time)
        
        Returns:
            True if code is valid
        """
        if timestamp is None:
            timestamp = int(time.time())
        
        # Check current and adjacent time windows
        for offset in range(-window, window + 1):
            expected = self.generate_totp(secret, timestamp + (offset * self.time_step))
            if hmac.compare_digest(code, expected):
                return True
        
        return False
    
    def get_provisioning_uri(
        self,
        secret: str,
        account_name: str,
        issuer: str,
    ) -> str:
        """Generate provisioning URI for QR code.
        
        Args:
            secret: Base32-encoded secret
            account_name: User account name
            issuer: Service name
        
        Returns:
            otpauth:// URI
        """
        import urllib.parse
        
        params = {
            'secret': secret,
            'issuer': issuer,
            'algorithm': 'SHA1',
            'digits': str(self.digits),
            'period': str(self.time_step),
        }
        
        encoded_name = urllib.parse.quote(account_name, safe='')
        return f"otpauth://totp/{issuer}:{encoded_name}?{urllib.parse.urlencode(params)}"
    
    def generate_qr_code(self, uri: str) -> str:
        """Generate QR code as ASCII art or image data.
        
        Args:
            uri: otpauth:// URI
        
        Returns:
            QR code as base64-encoded PNG
        """
        try:
            import qrcode
            import io
            
            qr = qrcode.QRCode(
                version=1,
                error_correction=qrcode.constants.ERROR_CORRECT_L,
                box_size=10,
                border=2,
            )
            qr.add_data(uri)
            qr.make(fit=True)
            
            img = qr.make_image(fill_color="black", back_color="white")
            
            # Convert to base64
            buffer = io.BytesIO()
            img.save(buffer, format='PNG')
            buffer.seek(0)
            
            return base64.b64encode(buffer.read()).decode('utf-8')
            
        except ImportError:
            log.warning("qrcode not installed. Install with: pip install qrcode")
            return ""


class TwoFactorAuth:
    """Two-factor authentication manager.
    
    Usage:
        tfa = TwoFactorAuth()
        
        # Setup for new user
        user_id = "trader_001"
        setup = tfa.setup_2fa(user_id)
        # Show QR code to user
        
        # Verify setup
        tfa.verify_setup(user_id, code)
        
        # Login verification
        if tfa.verify_code(user_id, code):
            # Login successful
    """
    
    def __init__(self, storage_path: Path | str | None = None) -> None:
        """Initialize 2FA manager.

        Args:
            storage_path: Path to store 2FA configs
        """
        self.storage_path = storage_path
        self._cache: dict[str, User2FA] = {}
        self._authenticator = TOTPAuthenticator()
        self._storage: Any | None = None  # Lazy init to avoid env dependency in tests

        # Rate limiting
        self._failed_attempts: dict[str, list[datetime]] = {}
        self._lockout_duration = timedelta(minutes=15)
        self._max_attempts = 5

    def _get_storage(self) -> Any:
        """Get storage instance, using custom path if provided."""
        if self._storage is None:
            if self.storage_path is not None:
                # Create a dedicated SecureStorage for this instance
                from utils.security import SecureStorage
                # If storage_path is a directory, create a file path inside it
                import os
                storage_path = Path(self.storage_path)
                if storage_path.is_dir():
                    storage_file = storage_path / ".tfa_storage.enc"
                else:
                    storage_file = storage_path
                # Temporarily override the storage path env var
                old_path = os.getenv("TRADING_SECURE_STORAGE_PATH")
                os.environ["TRADING_SECURE_STORAGE_PATH"] = str(storage_file)
                try:
                    self._storage = SecureStorage()
                finally:
                    # Restore original env var
                    if old_path is None:
                        os.environ.pop("TRADING_SECURE_STORAGE_PATH", None)
                    else:
                        os.environ["TRADING_SECURE_STORAGE_PATH"] = old_path
            else:
                from utils.security import get_secure_storage
                self._storage = get_secure_storage()
        return self._storage

    def setup_2fa(self, user_id: str) -> dict[str, Any]:
        """Setup 2FA for a user.
        
        Args:
            user_id: User identifier
        
        Returns:
            Setup information including QR code
        """
        # Generate secret
        secret = self._authenticator.generate_secret()
        
        # Create user config
        user_config = User2FA(
            user_id=user_id,
            enabled=False,  # Not enabled until verified
            secret=secret,
        )
        
        # Generate provisioning URI
        uri = self._authenticator.get_provisioning_uri(
            secret,
            user_config.account_name,
            user_config.issuer,
        )
        
        # Generate QR code
        qr_code = self._authenticator.generate_qr_code(uri)
        
        # Generate backup codes
        backup_codes = self._generate_backup_codes()
        user_config.backup_codes = backup_codes
        
        # Store (not enabled yet)
        self._cache[user_id] = user_config
        self._save_user(user_id)
        
        return {
            'secret': secret,
            'uri': uri,
            'qr_code': qr_code,
            'backup_codes': backup_codes,
            'instructions': (
                "1. Scan the QR code with your authenticator app\n"
                "2. Enter the 6-digit code to verify\n"
                "3. Save your backup codes in a secure location"
            ),
        }
    
    def _generate_backup_codes(self, count: int = 10) -> list[str]:
        """Generate backup codes.
        
        Args:
            count: Number of codes to generate
        
        Returns:
            List of backup codes
        """
        codes = []
        for _ in range(count):
            # Generate 8-character alphanumeric code
            code = base64.b32encode(os.urandom(5)).decode('utf-8')[:8]
            codes.append(code)
        return codes
    
    def verify_setup(self, user_id: str, code: str) -> bool:
        """Verify 2FA setup with initial code.
        
        Args:
            user_id: User identifier
            code: TOTP code from authenticator app
        
        Returns:
            True if setup is verified and enabled
        """
        user_config = self._load_user(user_id)
        if not user_config:
            return False
        
        # Verify code
        if self._authenticator.verify_totp(user_config.secret, code):
            user_config.enabled = True
            user_config.last_used = datetime.now()
            self._save_user(user_id)
            log.info(f"2FA enabled for user {user_id}")
            return True
        
        return False
    
    def verify_code(
        self,
        user_id: str,
        code: str,
        use_backup_code: bool = False,
    ) -> bool:
        """Verify 2FA code for login or operation.
        
        Args:
            user_id: User identifier
            code: TOTP or backup code
            use_backup_code: Whether code is a backup code
        
        Returns:
            True if code is valid
        """
        # Check rate limiting
        if self._is_rate_limited(user_id):
            log.warning(f"2FA rate limited for user {user_id}")
            return False
        
        user_config = self._load_user(user_id)
        if not user_config:
            return False
        
        # Check if locked
        if user_config.locked_until and datetime.now() < user_config.locked_until:
            log.warning(f"User {user_id} is locked until {user_config.locked_until}")
            return False
        
        # Check 2FA is enabled
        if not user_config.enabled:
            log.warning(f"2FA not enabled for user {user_id}")
            return True  # Allow if not enabled
        
        # Try backup code
        if use_backup_code:
            return self._verify_backup_code(user_config, code)
        
        # Verify TOTP
        if self._authenticator.verify_totp(user_config.secret, code):
            user_config.last_used = datetime.now()
            user_config.failed_attempts = 0
            self._save_user(user_id)
            self._clear_failed_attempts(user_id)
            return True
        
        # Record failed attempt
        self._record_failed_attempt(user_id)
        return False
    
    def _verify_backup_code(
        self,
        user_config: User2FA,
        code: str,
    ) -> bool:
        """Verify backup code."""
        code = code.strip().upper().replace('-', '')
        
        for i, backup_code in enumerate(user_config.backup_codes):
            if backup_code.upper() == code:
                # Mark as used
                user_config.backup_codes.pop(i)
                user_config.last_used = datetime.now()
                self._save_user(user_id=user_config.user_id)
                log.info(f"Backup code used for user {user_config.user_id}")
                return True
        
        return False
    
    def _is_rate_limited(self, user_id: str) -> bool:
        """Check if user is rate limited."""
        attempts = self._failed_attempts.get(user_id, [])
        
        # Clean old attempts
        cutoff = datetime.now() - timedelta(minutes=5)
        attempts = [a for a in attempts if a > cutoff]
        self._failed_attempts[user_id] = attempts
        
        return len(attempts) >= self._max_attempts
    
    def _record_failed_attempt(self, user_id: str) -> None:
        """Record failed 2FA attempt."""
        now = datetime.now()
        
        if user_id not in self._failed_attempts:
            self._failed_attempts[user_id] = []
        
        self._failed_attempts[user_id].append(now)
        
        # Check if should lock
        if len(self._failed_attempts[user_id]) >= self._max_attempts:
            user_config = self._load_user(user_id)
            if user_config:
                user_config.locked_until = now + self._lockout_duration
                user_config.failed_attempts += 1
                self._save_user(user_id)
                log.warning(f"User {user_id} locked until {user_config.locked_until}")
    
    def _clear_failed_attempts(self, user_id: str) -> None:
        """Clear failed attempts for user."""
        self._failed_attempts.pop(user_id, None)
        
        user_config = self._load_user(user_id)
        if user_config:
            user_config.locked_until = None
            user_config.failed_attempts = 0
            self._save_user(user_id)
    
    def disable_2fa(self, user_id: str, code: str) -> bool:
        """Disable 2FA for a user.
        
        Args:
            user_id: User identifier
            code: Current TOTP code for verification
        
        Returns:
            True if disabled
        """
        if not self.verify_code(user_id, code):
            return False
        
        user_config = self._load_user(user_id)
        if user_config:
            user_config.enabled = False
            user_config.secret = ""
            self._save_user(user_id)
            log.info(f"2FA disabled for user {user_id}")
            return True
        
        return False
    
    def regenerate_backup_codes(self, user_id: str, code: str) -> list[str] | None:
        """Regenerate backup codes.
        
        Args:
            user_id: User identifier
            code: Current TOTP code for verification
        
        Returns:
            New backup codes, or None if verification failed
        """
        if not self.verify_code(user_id, code):
            return None
        
        user_config = self._load_user(user_id)
        if user_config:
            new_codes = self._generate_backup_codes()
            user_config.backup_codes = new_codes
            self._save_user(user_id)
            return new_codes
        
        return None
    
    def get_2fa_status(self, user_id: str) -> dict[str, Any]:
        """Get 2FA status for user.
        
        Args:
            user_id: User identifier
        
        Returns:
            Status information
        """
        user_config = self._load_user(user_id)
        
        if not user_config:
            return {
                'enabled': False,
                'configured': False,
            }
        
        return {
            'enabled': user_config.enabled,
            'configured': bool(user_config.secret),
            'backup_codes_remaining': len(user_config.backup_codes),
            'last_used': user_config.last_used.isoformat() if user_config.last_used else None,
            'locked': bool(
                user_config.locked_until and
                datetime.now() < user_config.locked_until
            ),
        }
    
    def _save_user(self, user_id: str) -> None:
        """Save user 2FA config."""
        # In production, use secure storage
        storage = self._get_storage()

        user_config = self._cache.get(user_id)
        if user_config:
            data = {
                'user_id': user_config.user_id,
                'enabled': user_config.enabled,
                'secret': user_config.secret,
                'backup_codes': user_config.backup_codes,
                'created_at': user_config.created_at.isoformat() if user_config.created_at else None,
                'last_used': user_config.last_used.isoformat() if user_config.last_used else None,
                'failed_attempts': user_config.failed_attempts,
                'locked_until': user_config.locked_until.isoformat() if user_config.locked_until else None,
            }

            # Encrypt and store
            storage.set(f"2fa_{user_id}", json.dumps(data))

    def _load_user(self, user_id: str) -> User2FA | None:
        """Load user 2FA config."""
        # Check cache first
        if user_id in self._cache:
            return self._cache[user_id]

        # Load from storage
        storage = self._get_storage()
        data_str = storage.get(f"2fa_{user_id}")
        
        if not data_str:
            return None
        
        try:
            data = json.loads(data_str)
            user_config = User2FA(
                user_id=data['user_id'],
                enabled=data.get('enabled', False),
                secret=data.get('secret', ''),
                backup_codes=data.get('backup_codes', []),
                created_at=datetime.fromisoformat(data['created_at']) if data.get('created_at') else None,
                last_used=datetime.fromisoformat(data['last_used']) if data.get('last_used') else None,
                failed_attempts=data.get('failed_attempts', 0),
                locked_until=datetime.fromisoformat(data['locked_until']) if data.get('locked_until') else None,
            )
            self._cache[user_id] = user_config
            return user_config
        except Exception as e:
            log.error(f"Failed to load 2FA config: {e}")
            return None


# Global instance
_tfa_instance: TwoFactorAuth | None = None


def get_2fa() -> TwoFactorAuth:
    """Get global 2FA manager."""
    global _tfa_instance
    if _tfa_instance is None:
        _tfa_instance = TwoFactorAuth()
    return _tfa_instance
