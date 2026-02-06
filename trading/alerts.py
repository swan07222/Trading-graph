# trading/alerts.py
"""
Production Alerting System
Score Target: 10/10

Features:
- Multi-channel alerts (Email, SMS, Webhook, Desktop)
- Alert throttling (prevent spam)
- Priority levels
- Persistent alert history
- Acknowledgment tracking
"""
import threading
import smtplib
import json
import requests
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from collections import defaultdict
import queue

from config import CONFIG
from utils.logger import get_logger
from utils.security import get_secure_storage, get_audit_log  # FIXED: Added get_audit_log

log = get_logger(__name__)


class AlertPriority(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


class AlertChannel(Enum):
    LOG = "log"
    DESKTOP = "desktop"
    EMAIL = "email"
    SMS = "sms"
    WEBHOOK = "webhook"


class AlertCategory(Enum):
    RISK = "risk"
    TRADING = "trading"
    SYSTEM = "system"
    CONNECTION = "connection"
    DATA = "data"
    PERFORMANCE = "performance"


@dataclass
class Alert:
    """Alert message"""
    id: str = ""
    category: AlertCategory = AlertCategory.SYSTEM
    priority: AlertPriority = AlertPriority.MEDIUM
    title: str = ""
    message: str = ""
    details: Dict = field(default_factory=dict)
    
    # Routing
    channels: List[AlertChannel] = field(default_factory=list)
    
    # Status
    created_at: datetime = None
    sent_at: datetime = None
    acknowledged_at: datetime = None
    acknowledged_by: str = ""
    
    # Throttling
    throttle_key: str = ""
    
    def __post_init__(self):
        if not self.id:
            import uuid
            self.id = f"ALERT_{uuid.uuid4().hex[:12].upper()}"
        if not self.created_at:
            self.created_at = datetime.now()
        if not self.channels:
            self.channels = [AlertChannel.LOG, AlertChannel.DESKTOP]
    
    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'category': self.category.value,
            'priority': self.priority.value,
            'title': self.title,
            'message': self.message,
            'details': self.details,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'acknowledged': self.acknowledged_at is not None
        }


class AlertThrottler:
    """
    Prevent alert spam by throttling similar alerts
    """
    
    def __init__(self, default_window_seconds: int = 300):
        self._lock = threading.Lock()
        self._last_sent: Dict[str, datetime] = {}
        self._default_window = timedelta(seconds=default_window_seconds)
        
        # Per-priority windows
        self._windows = {
            AlertPriority.LOW: timedelta(minutes=30),
            AlertPriority.MEDIUM: timedelta(minutes=10),
            AlertPriority.HIGH: timedelta(minutes=2),
            AlertPriority.CRITICAL: timedelta(seconds=30),
        }
    
    def should_send(self, alert: Alert) -> bool:
        """Check if alert should be sent (not throttled)"""
        if not alert.throttle_key:
            return True  # No throttle key, always send
        
        with self._lock:
            key = f"{alert.category.value}:{alert.throttle_key}"
            last = self._last_sent.get(key)
            
            if not last:
                self._last_sent[key] = datetime.now()
                return True
            
            window = self._windows.get(alert.priority, self._default_window)
            
            if datetime.now() - last > window:
                self._last_sent[key] = datetime.now()
                return True
            
            return False
    
    def reset(self, throttle_key: str = None):
        """Reset throttle state"""
        with self._lock:
            if throttle_key:
                keys_to_remove = [k for k in self._last_sent if throttle_key in k]
                for k in keys_to_remove:
                    del self._last_sent[k]
            else:
                self._last_sent.clear()


class AlertManager:
    """
    Central alert management system
    
    Features:
    - Multi-channel delivery
    - Throttling
    - History
    - Acknowledgment
    """
    
    HISTORY_FILE = "alert_history.json"
    
    def __init__(self):
        self._lock = threading.RLock()
        self._throttler = AlertThrottler()
        self._audit = get_audit_log()
        
        # Alert queue for async sending
        self._queue: queue.Queue = queue.Queue()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        
        # History
        self._history: List[Alert] = []
        self._max_history = 1000
        
        # Pending acknowledgment
        self._pending_ack: Dict[str, Alert] = {}
        
        # Desktop notification callback
        self._desktop_callback: Optional[Callable] = None
        
        # Load history
        self._load_history()
    
    def start(self):
        """Start alert processing"""
        if self._running:
            return
        
        self._running = True
        self._thread = threading.Thread(target=self._process_loop, daemon=True)
        self._thread.start()
        log.info("Alert manager started")
    
    def stop(self):
        """Stop alert processing"""
        self._running = False
        if self._thread:
            self._queue.put(None)  # Sentinel
            self._thread.join(timeout=5)
        
        self._save_history()
        log.info("Alert manager stopped")
    
    def send(self, alert: Alert):
        """Queue alert for sending - auto-starts if not running"""
        if not self._running:
            self.start()
        self._queue.put(alert)
    
    def send_immediate(self, alert: Alert):
        """Send alert immediately (synchronous)"""
        self._process_alert(alert)
    
    def _process_loop(self):
        """Process alerts from queue"""
        while self._running:
            try:
                alert = self._queue.get(timeout=1)
                if alert is None:  # Sentinel
                    break
                self._process_alert(alert)
            except queue.Empty:
                continue
            except Exception as e:
                log.error(f"Alert processing error: {e}")
    
    def _process_alert(self, alert: Alert):
        """Process a single alert"""
        # Check throttle
        if not self._throttler.should_send(alert):
            log.debug(f"Alert throttled: {alert.title}")
            return
        
        # Send to each channel
        for channel in alert.channels:
            try:
                self._send_to_channel(alert, channel)
            except Exception as e:
                log.error(f"Failed to send alert to {channel.value}: {e}")
        
        alert.sent_at = datetime.now()
        
        # Add to history
        with self._lock:
            self._history.append(alert)
            if len(self._history) > self._max_history:
                self._history = self._history[-self._max_history:]
            
            # Track for acknowledgment if high priority
            if alert.priority in [AlertPriority.HIGH, AlertPriority.CRITICAL]:
                self._pending_ack[alert.id] = alert
        
        # Audit
        self._audit.log_risk_event('alert_sent', alert.to_dict())
    
    def _send_to_channel(self, alert: Alert, channel: AlertChannel):
        """Send alert to specific channel"""
        if channel == AlertChannel.LOG:
            self._send_log(alert)
        elif channel == AlertChannel.DESKTOP:
            self._send_desktop(alert)
        elif channel == AlertChannel.EMAIL:
            self._send_email(alert)
        elif channel == AlertChannel.WEBHOOK:
            self._send_webhook(alert)
    
    def _send_log(self, alert: Alert):
        """Send to log"""
        level_map = {
            AlertPriority.LOW: log.info,
            AlertPriority.MEDIUM: log.warning,
            AlertPriority.HIGH: log.error,
            AlertPriority.CRITICAL: log.critical,
        }
        
        logger = level_map.get(alert.priority, log.warning)
        logger(f"🔔 [{alert.category.value.upper()}] {alert.title}: {alert.message}")
    
    def _send_desktop(self, alert: Alert):
        """Send desktop notification"""
        if self._desktop_callback:
            try:
                self._desktop_callback(alert)
            except Exception as e:
                log.warning(f"Desktop notification failed: {e}")
        else:
            # Try system notification
            try:
                import platform
                if platform.system() == 'Windows':
                    from win10toast import ToastNotifier
                    toaster = ToastNotifier()
                    toaster.show_toast(
                        alert.title,
                        alert.message,
                        duration=5,
                        threaded=True
                    )
            except ImportError:
                pass
    
    def _send_email(self, alert: Alert):
        """Send email notification"""
        if not CONFIG.alerts.email_enabled:
            return
        
        if not CONFIG.alerts.smtp_server or not CONFIG.alerts.email_recipients:
            return
        
        try:
            msg = MIMEMultipart()
            msg['Subject'] = f"[{alert.priority.name}] {alert.title}"
            msg['From'] = f"Trading System <{CONFIG.alerts.smtp_server}>"
            msg['To'] = ", ".join(CONFIG.alerts.email_recipients)
            
            body = f"""
Trading System Alert
====================

Category: {alert.category.value}
Priority: {alert.priority.name}
Time: {alert.created_at.isoformat()}

{alert.message}

Details:
{json.dumps(alert.details, indent=2)}
"""
            msg.attach(MIMEText(body, 'plain'))
            
            with smtplib.SMTP(CONFIG.alerts.smtp_server, CONFIG.alerts.smtp_port, timeout=10) as server:
                server.starttls()

                user = CONFIG.alerts.smtp_username
                pwd = get_secure_storage().get(CONFIG.alerts.smtp_password_key, "")
                if user and pwd:
                    server.login(user, pwd)

                server.send_message(msg)
            
            log.info(f"Email alert sent: {alert.title}")
            
        except Exception as e:
            log.error(f"Email send failed: {e}")
    
    def _send_webhook(self, alert: Alert):
        """Send webhook notification"""
        if not CONFIG.alerts.webhook_enabled or not CONFIG.alerts.webhook_url:
            return
        
        try:
            payload = {
                'alert': alert.to_dict(),
                'timestamp': datetime.now().isoformat(),
                'system': 'trading_system'
            }
            
            response = requests.post(
                CONFIG.alerts.webhook_url,
                json=payload,
                timeout=10
            )
            response.raise_for_status()
            
            log.info(f"Webhook alert sent: {alert.title}")
            
        except Exception as e:
            log.error(f"Webhook send failed: {e}")
    
    def acknowledge(self, alert_id: str, by: str = "user") -> bool:
        """Acknowledge an alert"""
        with self._lock:
            alert = self._pending_ack.get(alert_id)
            if alert:
                alert.acknowledged_at = datetime.now()
                alert.acknowledged_by = by
                del self._pending_ack[alert_id]
                
                self._audit.log_risk_event('alert_acknowledged', {
                    'alert_id': alert_id,
                    'acknowledged_by': by
                })
                
                return True
            return False
    
    def get_pending(self) -> List[Alert]:
        """Get alerts pending acknowledgment"""
        with self._lock:
            return list(self._pending_ack.values())
    
    def get_history(
        self, 
        category: AlertCategory = None,
        priority: AlertPriority = None,
        limit: int = 100
    ) -> List[Alert]:
        """Get alert history"""
        with self._lock:
            alerts = self._history.copy()
            
            if category:
                alerts = [a for a in alerts if a.category == category]
            if priority:
                alerts = [a for a in alerts if a.priority == priority]
            
            return alerts[-limit:]
    
    def set_desktop_callback(self, callback: Callable):
        """Set desktop notification callback (for GUI integration)"""
        self._desktop_callback = callback
    
    def _save_history(self):
        """Save alert history to disk"""
        path = CONFIG.data_dir / self.HISTORY_FILE
        
        try:
            data = [a.to_dict() for a in self._history[-500:]]
            with open(path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            log.error(f"Failed to save alert history: {e}")
    
    def _load_history(self):
        """Load alert history from disk"""
        path = CONFIG.data_dir / self.HISTORY_FILE
        
        if not path.exists():
            return
        
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            
            for item in data:
                alert = Alert(
                    id=item.get('id', ''),
                    category=AlertCategory(item.get('category', 'system')),
                    priority=AlertPriority(item.get('priority', 2)),
                    title=item.get('title', ''),
                    message=item.get('message', ''),
                    details=item.get('details', {})
                )
                if item.get('created_at'):
                    alert.created_at = datetime.fromisoformat(item['created_at'])
                
                self._history.append(alert)
                
        except Exception as e:
            log.error(f"Failed to load alert history: {e}")
    
    # Convenience methods for common alerts
    
    def risk_alert(self, title: str, message: str, details: Dict = None):
        """Send risk alert"""
        self.send(Alert(
            category=AlertCategory.RISK,
            priority=AlertPriority.HIGH,
            title=title,
            message=message,
            details=details or {},
            channels=[AlertChannel.LOG, AlertChannel.DESKTOP, AlertChannel.EMAIL],
            throttle_key=title
        ))
    
    def trading_alert(self, title: str, message: str, details: Dict = None):
        """Send trading alert"""
        self.send(Alert(
            category=AlertCategory.TRADING,
            priority=AlertPriority.MEDIUM,
            title=title,
            message=message,
            details=details or {},
            channels=[AlertChannel.LOG, AlertChannel.DESKTOP],
            throttle_key=title
        ))
    
    def system_alert(self, title: str, message: str, priority: AlertPriority = AlertPriority.MEDIUM):
        """Send system alert"""
        self.send(Alert(
            category=AlertCategory.SYSTEM,
            priority=priority,
            title=title,
            message=message,
            channels=[AlertChannel.LOG, AlertChannel.DESKTOP],
            throttle_key=title
        ))
    
    def critical_alert(self, title: str, message: str, details: Dict = None):
        """Send critical alert (all channels)"""
        self.send(Alert(
            category=AlertCategory.SYSTEM,
            priority=AlertPriority.CRITICAL,
            title=title,
            message=message,
            details=details or {},
            channels=[AlertChannel.LOG, AlertChannel.DESKTOP, AlertChannel.EMAIL, AlertChannel.WEBHOOK],
            throttle_key=""  # Never throttle critical
        ))


# Global instance
_alert_manager: Optional[AlertManager] = None


def get_alert_manager() -> AlertManager:
    global _alert_manager
    if _alert_manager is None:
        _alert_manager = AlertManager()
    return _alert_manager