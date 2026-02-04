"""
Security and Compliance Module
"""
import os
import json
import hashlib
import secrets
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64

from config import CONFIG
from utils.logger import log


class SecureConfig:
    """
    Secure configuration storage for sensitive data
    (API keys, broker credentials, etc.)
    """
    
    def __init__(self):
        self._config_path = CONFIG.DATA_DIR / ".secure_config.enc"
        self._key_path = CONFIG.DATA_DIR / ".key"
        self._cipher = self._init_cipher()
    
    def _init_cipher(self) -> Fernet:
        """Initialize encryption cipher"""
        if self._key_path.exists():
            with open(self._key_path, 'rb') as f:
                key = f.read()
        else:
            key = Fernet.generate_key()
            with open(self._key_path, 'wb') as f:
                f.write(key)
            # Restrict permissions
            os.chmod(self._key_path, 0o600)
        
        return Fernet(key)
    
    def set(self, key: str, value: str):
        """Store encrypted value"""
        config = self._load()
        config[key] = value
        self._save(config)
    
    def get(self, key: str, default: str = None) -> Optional[str]:
        """Retrieve decrypted value"""
        config = self._load()
        return config.get(key, default)
    
    def _load(self) -> Dict:
        if not self._config_path.exists():
            return {}
        
        try:
            with open(self._config_path, 'rb') as f:
                encrypted = f.read()
            decrypted = self._cipher.decrypt(encrypted)
            return json.loads(decrypted.decode())
        except Exception as e:
            log.error(f"Failed to load secure config: {e}")
            return {}
    
    def _save(self, config: Dict):
        try:
            data = json.dumps(config).encode()
            encrypted = self._cipher.encrypt(data)
            with open(self._config_path, 'wb') as f:
                f.write(encrypted)
            os.chmod(self._config_path, 0o600)
        except Exception as e:
            log.error(f"Failed to save secure config: {e}")


class AuditLog:
    """
    Audit logging for compliance
    Records all trading decisions and executions
    """
    
    def __init__(self):
        self._log_dir = CONFIG.LOG_DIR / "audit"
        self._log_dir.mkdir(exist_ok=True)
        self._current_file = None
        self._current_date = None
    
    def _get_file(self):
        today = datetime.now().date()
        if self._current_date != today:
            if self._current_file:
                self._current_file.close()
            path = self._log_dir / f"audit_{today.isoformat()}.jsonl"
            self._current_file = open(path, 'a')
            self._current_date = today
        return self._current_file
    
    def log_signal(self, stock_code: str, signal: str, confidence: float, 
                   reasons: list, price: float):
        """Log trading signal generation"""
        self._write({
            'event': 'signal',
            'timestamp': datetime.now().isoformat(),
            'stock_code': stock_code,
            'signal': signal,
            'confidence': confidence,
            'reasons': reasons,
            'price': price
        })
    
    def log_order(self, order_id: str, stock_code: str, side: str, 
                  quantity: int, price: float, status: str):
        """Log order submission/update"""
        self._write({
            'event': 'order',
            'timestamp': datetime.now().isoformat(),
            'order_id': order_id,
            'stock_code': stock_code,
            'side': side,
            'quantity': quantity,
            'price': price,
            'status': status
        })
    
    def log_trade(self, order_id: str, stock_code: str, side: str,
                  quantity: int, price: float, commission: float, pnl: float = None):
        """Log trade execution"""
        self._write({
            'event': 'trade',
            'timestamp': datetime.now().isoformat(),
            'order_id': order_id,
            'stock_code': stock_code,
            'side': side,
            'quantity': quantity,
            'price': price,
            'commission': commission,
            'pnl': pnl
        })
    
    def log_risk_event(self, event_type: str, details: dict):
        """Log risk management events"""
        self._write({
            'event': 'risk',
            'timestamp': datetime.now().isoformat(),
            'type': event_type,
            'details': details
        })
    
    def _write(self, record: dict):
        try:
            f = self._get_file()
            f.write(json.dumps(record) + '\n')
            f.flush()
        except Exception as e:
            log.error(f"Audit log failed: {e}")
    
    def close(self):
        if self._current_file:
            self._current_file.close()


class RateLimiter:
    """
    Rate limiter for API calls and order submissions
    Prevents excessive trading and API abuse
    """
    
    def __init__(self):
        self._limits = {
            'orders_per_minute': 10,
            'orders_per_hour': 100,
            'api_calls_per_second': 5,
        }
        self._counters = {}
        self._windows = {}
    
    def check(self, limit_type: str) -> bool:
        """Check if action is allowed"""
        now = datetime.now()
        limit = self._limits.get(limit_type, 100)
        
        # Get time window
        if 'minute' in limit_type:
            window_seconds = 60
        elif 'hour' in limit_type:
            window_seconds = 3600
        else:
            window_seconds = 1
        
        # Clean old entries
        window_start = now.timestamp() - window_seconds
        key = limit_type
        
        if key not in self._counters:
            self._counters[key] = []
        
        self._counters[key] = [
            t for t in self._counters[key]
            if t > window_start
        ]
        
        if len(self._counters[key]) >= limit:
            return False
        
        self._counters[key].append(now.timestamp())
        return True
    
    def wait_if_needed(self, limit_type: str, timeout: float = 60):
        """Wait until action is allowed"""
        import time
        start = time.time()
        while not self.check(limit_type):
            if time.time() - start > timeout:
                raise TimeoutError(f"Rate limit timeout: {limit_type}")
            time.sleep(0.1)


# Global instances
_secure_config = SecureConfig()
_audit_log = AuditLog()
_rate_limiter = RateLimiter()


def get_secure_config() -> SecureConfig:
    return _secure_config


def get_audit_log() -> AuditLog:
    return _audit_log


def get_rate_limiter() -> RateLimiter:
    return _rate_limiter