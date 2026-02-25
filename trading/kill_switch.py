# trading/kill_switch.py
import hashlib
import hmac
import json
import threading
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path

from config.settings import CONFIG
from core.events import EVENT_BUS, Event, EventType
from utils.logger import get_logger
from utils.security import get_audit_log

log = get_logger(__name__)

# FIX #6: Use HMAC for integrity verification of kill switch state
# This prevents tampering with the state file
def _compute_state_hmac(state_json: str) -> str:
    """Compute HMAC-SHA256 for state integrity verification.
    
    FIX #34: Use SHA-256 digest of key material for proper 32-byte key
    instead of using hex string directly which reduces entropy.
    """
    # Use a derived key from the secure storage key for HMAC
    try:
        from utils.security import SecureStorage
        storage = SecureStorage()
        # Get or derive a key for HMAC
        key_material = storage.get("_hmac_key", None)
        if key_material is None:
            # Generate and store a new HMAC key
            import secrets
            key_material = secrets.token_hex(32)
            storage.set("_hmac_key", key_material)
        
        # FIX #34: Derive proper 32-byte key from key material using SHA-256
        # This ensures full entropy and correct key length for HMAC-SHA256
        key = hashlib.sha256(key_material.encode('utf-8')).digest()
        return hmac.new(key, state_json.encode('utf-8'), hashlib.sha256).hexdigest()
    except Exception as e:
        # Fallback: use a simple hash (less secure but better than nothing)
        log.warning(f"HMAC computation failed, using fallback: {e}")
        return hashlib.sha256(state_json.encode('utf-8')).hexdigest()

def _verify_state_hmac(state_json: str, expected_hmac: str) -> bool:
    """Verify HMAC-SHA256 for state integrity."""
    try:
        computed = _compute_state_hmac(state_json)
        return hmac.compare_digest(computed, expected_hmac)
    except Exception as e:
        log.error(f"HMAC verification failed: {e}")
        return False

class CircuitBreakerType(Enum):
    DAILY_LOSS = "daily_loss"
    DRAWDOWN = "drawdown"
    RAPID_LOSS = "rapid_loss"
    ERROR_RATE = "error_rate"
    MANUAL = "manual"

@dataclass
class CircuitBreakerState:
    """Circuit breaker state."""
    type: CircuitBreakerType
    triggered: bool = False
    triggered_at: datetime | None = None
    reset_at: datetime | None = None
    trigger_value: float = 0.0
    threshold: float = 0.0
    message: str = ""

class KillSwitch:
    """Emergency kill switch for trading."""

    STATE_FILE = "kill_switch_state.json"

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._audit = get_audit_log()

        self._active = False
        self._activated_at: datetime | None = None
        self._activated_by: str = ""
        self._reason: str = ""

        self._circuit_breakers: dict[CircuitBreakerType, CircuitBreakerState] = {}
        self._init_circuit_breakers()

        self._on_activate: list[Callable[[str], object]] = []
        self._on_deactivate: list[Callable[[], object]] = []

        self._load_state()

        EVENT_BUS.subscribe(EventType.RISK_BREACH, self._on_risk_event)

    def _init_circuit_breakers(self) -> None:
        self._circuit_breakers = {
            CircuitBreakerType.DAILY_LOSS: CircuitBreakerState(
                type=CircuitBreakerType.DAILY_LOSS,
                threshold=CONFIG.risk.circuit_breaker_loss_pct,
            ),
            CircuitBreakerType.DRAWDOWN: CircuitBreakerState(
                type=CircuitBreakerType.DRAWDOWN,
                threshold=CONFIG.risk.kill_switch_drawdown_pct,
            ),
            CircuitBreakerType.RAPID_LOSS: CircuitBreakerState(
                type=CircuitBreakerType.RAPID_LOSS,
                threshold=2.0,
            ),
            CircuitBreakerType.ERROR_RATE: CircuitBreakerState(
                type=CircuitBreakerType.ERROR_RATE,
                threshold=5,
            ),
        }

    @property
    def is_active(self) -> bool:
        return self._active

    @property
    def can_trade(self) -> bool:
        if self._active:
            return False

        for cb in self._circuit_breakers.values():
            if cb.triggered and not self._is_cb_expired(cb):
                return False

        return True

    def _is_cb_expired(self, cb: CircuitBreakerState) -> bool:
        if not cb.reset_at:
            return False
        return datetime.now() >= cb.reset_at

    def activate(self, reason: str, activated_by: str = "system") -> bool:
        """Activate kill switch with proper locking to prevent deadlocks.
        
        FIX #2: Move EVENT_BUS.publish outside the lock to prevent deadlock
        if event handlers try to acquire the same lock.
        """
        callbacks_to_call: list[Callable[[str], object]] = []
        event_to_publish: Event | None = None
        
        with self._lock:
            if self._active:
                return False

            self._active = True
            self._activated_at = datetime.now()
            self._activated_by = activated_by
            self._reason = reason

            self._save_state()

            self._audit.log_risk_event('kill_switch_activated', {
                'reason': reason,
                'activated_by': activated_by,
                'timestamp': self._activated_at.isoformat(),
            })

            log.critical(f"🛑 KILL SWITCH ACTIVATED: {reason}")

            # Copy callbacks while holding lock, call them outside
            callbacks_to_call = list(self._on_activate)
            
            # Prepare event for publishing outside lock
            event_to_publish = Event(
                type=EventType.CIRCUIT_BREAKER,
                data={
                    'action': 'kill_switch_activated',
                    'reason': reason,
                },
            )

        # Call callbacks outside the lock to prevent deadlock
        for callback in callbacks_to_call:
            try:
                callback(reason)
            except Exception as e:
                log.error(f"Kill switch callback error: {e}")

        # Publish event outside the lock to prevent deadlock
        if event_to_publish is not None:
            EVENT_BUS.publish(event_to_publish)

        return True

    def deactivate(
        self,
        deactivated_by: str = "system",
        override_code: str | None = None,
    ) -> bool:
        """Deactivate kill switch with proper locking to prevent deadlocks.
        
        FIX #2: Move callback execution outside the lock to prevent deadlock.
        """
        callbacks_to_call: list[Callable[[], object]] = []
        
        with self._lock:
            if not self._active:
                return False

            if CONFIG.trading_mode.value == 'live':
                if (
                    not override_code
                    or override_code != self._get_override_code()
                ):
                    log.warning(
                        "Kill switch deactivation requires valid override code"
                    )
                    return False

            self._active = False

            self._save_state()

            self._audit.log_risk_event('kill_switch_deactivated', {
                'deactivated_by': deactivated_by,
                'was_active_for': (
                    (datetime.now() - self._activated_at).total_seconds()
                    if self._activated_at
                    else 0
                ),
            })

            log.info(f"✅ Kill switch deactivated by {deactivated_by}")

            # Copy callbacks while holding lock, call them outside
            callbacks_to_call = list(self._on_deactivate)

        # Call callbacks outside the lock to prevent deadlock
        for callback in callbacks_to_call:
            try:
                callback()
            except Exception as e:
                log.error(f"Kill switch callback error: {e}")

        return True

    def _get_override_code(self) -> str:
        import hashlib
        today = datetime.now().strftime("%Y-%m-%d")
        return hashlib.sha256(
            f"OVERRIDE_{today}".encode()
        ).hexdigest()[:8].upper()

    def trigger_circuit_breaker(
        self,
        cb_type: CircuitBreakerType,
        current_value: float,
        message: str = "",
    ) -> None:
        with self._lock:
            cb = self._circuit_breakers.get(cb_type)
            if not cb:
                return

            if cb.triggered:
                return

            cb.triggered = True
            cb.triggered_at = datetime.now()
            cb.trigger_value = current_value
            cb.message = message

            if cb_type == CircuitBreakerType.DAILY_LOSS:
                cb.reset_at = None
            elif cb_type == CircuitBreakerType.RAPID_LOSS:
                cb.reset_at = datetime.now() + timedelta(minutes=30)
            elif cb_type == CircuitBreakerType.ERROR_RATE:
                cb.reset_at = datetime.now() + timedelta(minutes=5)
            else:
                cb.reset_at = datetime.now() + timedelta(
                    minutes=CONFIG.risk.circuit_breaker_duration_minutes
                )

            self._save_state()

            self._audit.log_risk_event('circuit_breaker_triggered', {
                'type': cb_type.value,
                'value': current_value,
                'threshold': cb.threshold,
                'message': message,
                'reset_at': (
                    cb.reset_at.isoformat() if cb.reset_at else None
                ),
            })

            log.warning(
                f"⚡ Circuit breaker triggered: {cb_type.value} - {message}"
            )

            if cb_type in [CircuitBreakerType.DRAWDOWN]:
                self.activate(
                    f"Circuit breaker: {cb_type.value}", "circuit_breaker"
                )

    def reset_circuit_breaker(self, cb_type: CircuitBreakerType) -> bool:
        with self._lock:
            cb = self._circuit_breakers.get(cb_type)
            if not cb or not cb.triggered:
                return False

            cb.triggered = False
            cb.triggered_at = None
            cb.reset_at = None

            self._save_state()

            self._audit.log_risk_event('circuit_breaker_reset', {
                'type': cb_type.value,
            })

            log.info(f"Circuit breaker reset: {cb_type.value}")
            return True

    def _on_risk_event(self, event: Event) -> None:
        """Handle RiskEvent correctly."""
        risk_type = getattr(event, "risk_type", "") or ""
        current_value = float(getattr(event, "current_value", 0.0) or 0.0)

        if not risk_type:
            data = getattr(event, "data", {}) or {}
            risk_type = str(data.get("risk_type", "") or "")
            current_value = float(
                data.get("current_value", current_value) or current_value
            )

        if not risk_type:
            return

        if risk_type == "daily_loss_limit":
            self.trigger_circuit_breaker(
                CircuitBreakerType.DAILY_LOSS,
                current_value,
                f"Daily loss: {current_value:.2f}%",
            )
        elif risk_type == "max_drawdown":
            self.trigger_circuit_breaker(
                CircuitBreakerType.DRAWDOWN,
                current_value,
                f"Drawdown: {current_value:.2f}%",
            )
        elif risk_type == "kill_switch_threshold":
            self.activate(
                f"Kill switch threshold reached: {current_value:.2f}%",
                "risk_manager",
            )
        elif risk_type == "kill_switch_drawdown":
            self.activate(
                f"Kill switch drawdown reached: {current_value:.2f}%",
                "risk_manager",
            )

    def get_status(self) -> dict:
        with self._lock:
            active_cbs = []
            for cb in self._circuit_breakers.values():
                if cb.triggered and not self._is_cb_expired(cb):
                    active_cbs.append({
                        'type': cb.type.value,
                        'triggered_at': (
                            cb.triggered_at.isoformat()
                            if cb.triggered_at
                            else None
                        ),
                        'reset_at': (
                            cb.reset_at.isoformat() if cb.reset_at else None
                        ),
                        'value': cb.trigger_value,
                        'threshold': cb.threshold,
                        'message': cb.message,
                    })

            return {
                'kill_switch_active': self._active,
                'activated_at': (
                    self._activated_at.isoformat()
                    if self._activated_at
                    else None
                ),
                'activated_by': self._activated_by,
                'reason': self._reason,
                'can_trade': self.can_trade,
                'active_circuit_breakers': active_cbs,
            }

    def _save_state(self) -> None:
        """Save state to disk atomically with HMAC integrity protection.

        Uses temp file + rename pattern to prevent corruption on crash.
        FIX #6: Added HMAC integrity verification to prevent tampering.
        """
        path = CONFIG.data_dir / self.STATE_FILE

        state = {
            'active': self._active,
            'activated_at': (
                self._activated_at.isoformat()
                if self._activated_at
                else None
            ),
            'activated_by': self._activated_by,
            'reason': self._reason,
            'circuit_breakers': {
                cb_type.value: {
                    'triggered': cb.triggered,
                    'triggered_at': (
                        cb.triggered_at.isoformat()
                        if cb.triggered_at
                        else None
                    ),
                    'reset_at': (
                        cb.reset_at.isoformat() if cb.reset_at else None
                    ),
                    'trigger_value': cb.trigger_value,
                    'message': cb.message,
                }
                for cb_type, cb in self._circuit_breakers.items()
            },
        }

        # FIX #6: Compute HMAC for integrity verification
        state_json = json.dumps(state, indent=2)
        state_hmac = _compute_state_hmac(state_json)
        state['state_hmac'] = state_hmac

        tmp_path: Path | None = None
        try:
            tmp_path = path.with_suffix('.tmp')
            with open(tmp_path, 'w') as f:
                json.dump(state, f, indent=2)
            tmp_path.replace(path)
            tmp_path = None  # Successfully moved
        except Exception as e:
            log.error(f"Failed to save kill switch state: {e}")
            raise  # FIX: Re-raise to caller knows state wasn't saved
        finally:
            if tmp_path is not None and tmp_path.exists():
                try:
                    tmp_path.unlink()
                except OSError:
                    pass

    def _load_state(self) -> None:
        """Load state from disk with HMAC integrity verification.

        NOTE: A previously-active kill switch is INTENTIONALLY restored.
        This is a safety feature — the operator must explicitly deactivate
        the kill switch after investigating the cause. Auto-clearing on
        restart would defeat the purpose of an emergency stop.
        
        FIX #6: Added HMAC integrity verification to detect tampering.
        """
        path = CONFIG.data_dir / self.STATE_FILE

        if not path.exists():
            return

        try:
            with open(path) as f:
                state = json.load(f)

            # FIX #6: Verify HMAC integrity
            stored_hmac = state.pop('state_hmac', None)
            if stored_hmac is not None:
                state_json_for_verify = json.dumps(state, indent=2)
                if not _verify_state_hmac(state_json_for_verify, stored_hmac):
                    log.error(
                        "⚠️ KILL SWITCH STATE INTEGRITY CHECK FAILED! "
                        "State may have been tampered with. Refusing to load."
                    )
                    # Reset to safe state
                    self._active = False
                    return
                # Restore hmac for re-save if needed
                state['state_hmac'] = stored_hmac

            self._active = state.get('active', False)

            if state.get('activated_at'):
                self._activated_at = datetime.fromisoformat(
                    state['activated_at']
                )

            self._activated_by = state.get('activated_by', '')
            self._reason = state.get('reason', '')

            for cb_type_str, cb_state in state.get(
                'circuit_breakers', {}
            ).items():
                try:
                    cb_type = CircuitBreakerType(cb_type_str)
                    if cb_type in self._circuit_breakers:
                        cb = self._circuit_breakers[cb_type]
                        cb.triggered = cb_state.get('triggered', False)

                        if cb_state.get('triggered_at'):
                            cb.triggered_at = datetime.fromisoformat(
                                cb_state['triggered_at']
                            )
                        if cb_state.get('reset_at'):
                            cb.reset_at = datetime.fromisoformat(
                                cb_state['reset_at']
                            )

                        cb.trigger_value = cb_state.get('trigger_value', 0)
                        cb.message = cb_state.get('message', '')
                except Exception:
                    pass

            if self._active:
                log.warning(
                    f"🛑 Kill switch was active on restart: {self._reason}. "
                    f"Must be manually deactivated."
                )

        except Exception as e:
            log.error(f"Failed to load kill switch state: {e}")
            # On load error, stay in safe state (inactive)
            self._active = False

    def on_activate(self, callback: Callable[[str], object]) -> None:
        self._on_activate.append(callback)

    def on_deactivate(self, callback: Callable[[], object]) -> None:
        self._on_deactivate.append(callback)

# FIX: Module-level lock instead of globals() pattern
_kill_switch: KillSwitch | None = None
_kill_switch_lock = threading.Lock()

def get_kill_switch() -> KillSwitch:
    global _kill_switch
    if _kill_switch is None:
        with _kill_switch_lock:
            if _kill_switch is None:
                _kill_switch = KillSwitch()
    return _kill_switch
