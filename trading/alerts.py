# trading/alerts.py
from __future__ import annotations

import json
import queue
import smtplib
import threading
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from enum import Enum
from typing import Any

import requests

from config import CONFIG
from utils.logger import get_logger
from utils.security import get_audit_log, get_secure_storage

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
    """Alert message."""

    id: str = ""
    category: AlertCategory = AlertCategory.SYSTEM
    priority: AlertPriority = AlertPriority.MEDIUM
    title: str = ""
    message: str = ""
    details: dict = field(default_factory=dict)
    channels: list[AlertChannel] = field(default_factory=list)
    created_at: datetime | None = None
    sent_at: datetime | None = None
    acknowledged_at: datetime | None = None
    acknowledged_by: str = ""
    throttle_key: str = ""

    def __post_init__(self):
        if not self.id:
            import uuid

            self.id = f"ALERT_{uuid.uuid4().hex[:12].upper()}"
        if not self.created_at:
            self.created_at = datetime.now()
        if not self.channels:
            self.channels = [AlertChannel.LOG, AlertChannel.DESKTOP]

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "category": self.category.value,
            "priority": self.priority.value,
            "title": self.title,
            "message": self.message,
            "details": self.details,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "acknowledged": self.acknowledged_at is not None,
        }


class AlertThrottler:
    """Prevent alert spam by throttling similar alerts."""

    def __init__(self, default_window_seconds: int = 300):
        self._lock = threading.Lock()
        self._last_sent: dict[str, datetime] = {}
        self._default_window = timedelta(seconds=default_window_seconds)
        self._windows = {
            AlertPriority.LOW: timedelta(minutes=30),
            AlertPriority.MEDIUM: timedelta(minutes=10),
            AlertPriority.HIGH: timedelta(minutes=2),
            AlertPriority.CRITICAL: timedelta(seconds=30),
        }

    def should_send(self, alert: Alert) -> bool:
        if not alert.throttle_key:
            return True

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

    def reset(self, throttle_key: str | None = None):
        with self._lock:
            if throttle_key:
                keys_to_remove = [k for k in self._last_sent if throttle_key in k]
                for k in keys_to_remove:
                    del self._last_sent[k]
            else:
                self._last_sent.clear()


class AlertManager:
    """Central alert management system."""

    HISTORY_FILE = "alert_history.json"

    def __init__(self):
        self._lock = threading.RLock()
        self._throttler = AlertThrottler()
        self._audit = get_audit_log()
        self._queue: queue.Queue = queue.Queue()
        self._running = False
        self._thread: threading.Thread | None = None
        self._history: list[Alert] = []
        self._max_history = 1000
        self._pending_ack: dict[str, Alert] = {}
        self._repeat_counter: dict[str, int] = defaultdict(int)
        self._channel_stats: dict[str, dict[str, Any]] = defaultdict(
            lambda: {
                "sent": 0,
                "failed": 0,
                "last_sent_at": None,
                "last_error": "",
            }
        )
        self._desktop_callback: Callable | None = None
        self._load_history()

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._process_loop, daemon=True)
        self._thread.start()
        log.info("Alert manager started")

    def stop(self):
        self._running = False
        if self._thread:
            self._queue.put(None)
            self._thread.join(timeout=5)
        self._save_history()
        log.info("Alert manager stopped")

    def send(self, alert: Alert):
        if not self._running:
            self.start()
        self._queue.put(alert)

    def send_immediate(self, alert: Alert):
        self._process_alert(alert)

    def _process_loop(self):
        while self._running:
            try:
                alert = self._queue.get(timeout=1)
                if alert is None:
                    break
                self._process_alert(alert)
            except queue.Empty:
                continue
            except Exception as e:
                log.error(f"Alert processing error: {e}")

    def _process_alert(self, alert: Alert):
        key = f"{alert.category.value}:{alert.throttle_key or alert.title}"
        with self._lock:
            self._repeat_counter[key] += 1
            repeats = self._repeat_counter[key]

        if repeats >= 3 and alert.priority == AlertPriority.MEDIUM:
            alert.priority = AlertPriority.HIGH
        elif repeats >= 5 and alert.priority == AlertPriority.HIGH:
            alert.priority = AlertPriority.CRITICAL

        # Allow escalation alerts through even if their base key is throttled.
        throttled = not self._throttler.should_send(alert)
        if throttled and repeats < 3:
            log.debug(f"Alert throttled: {alert.title}")
            return

        sent_channels = 0
        failed_channels = 0
        for channel in alert.channels:
            try:
                self._send_to_channel(alert, channel)
                self._record_channel_result(channel, ok=True)
                sent_channels += 1
            except Exception as e:
                self._record_channel_result(channel, ok=False, error=str(e))
                log.error(f"Failed to send alert to {channel.value}: {e}")
                failed_channels += 1

        alert.sent_at = datetime.now()
        with self._lock:
            self._history.append(alert)
            if len(self._history) > self._max_history:
                self._history = self._history[-self._max_history :]
            if alert.priority in (AlertPriority.HIGH, AlertPriority.CRITICAL):
                self._pending_ack[alert.id] = alert
        if failed_channels > 0:
            self._audit.log_risk_event(
                "alert_delivery_partial_failure",
                {
                    "alert_id": alert.id,
                    "title": alert.title,
                    "failed_channels": int(failed_channels),
                    "sent_channels": int(sent_channels),
                },
            )
        self._audit.log_risk_event("alert_sent", alert.to_dict())

    def _record_channel_result(
        self,
        channel: AlertChannel,
        ok: bool,
        error: str = "",
    ) -> None:
        name = channel.value
        now_iso = datetime.now().isoformat()
        with self._lock:
            stats = self._channel_stats[name]
            if ok:
                stats["sent"] = int(stats.get("sent", 0)) + 1
                stats["last_sent_at"] = now_iso
            else:
                stats["failed"] = int(stats.get("failed", 0)) + 1
                stats["last_error"] = str(error or "")[:300]

    def _send_to_channel(self, alert: Alert, channel: AlertChannel):
        if channel == AlertChannel.LOG:
            self._send_log(alert)
        elif channel == AlertChannel.DESKTOP:
            self._send_desktop(alert)
        elif channel == AlertChannel.EMAIL:
            self._send_email(alert)
        elif channel == AlertChannel.WEBHOOK:
            self._send_webhook(alert)

    def _send_log(self, alert: Alert):
        level_map = {
            AlertPriority.LOW: log.info,
            AlertPriority.MEDIUM: log.warning,
            AlertPriority.HIGH: log.error,
            AlertPriority.CRITICAL: log.critical,
        }
        logger = level_map.get(alert.priority, log.warning)
        logger(f"[ALERT:{alert.category.value.upper()}] {alert.title}: {alert.message}")

    def _send_desktop(self, alert: Alert):
        if self._desktop_callback:
            try:
                self._desktop_callback(alert)
            except Exception as e:
                log.warning(f"Desktop notification failed: {e}")
            return

        try:
            import platform

            if platform.system() == "Windows":
                from win10toast import ToastNotifier

                toaster = ToastNotifier()
                toaster.show_toast(
                    alert.title,
                    alert.message,
                    duration=5,
                    threaded=True,
                )
        except ImportError:
            pass
        except Exception as e:
            log.debug(f"Desktop notification skipped: {e}")

    def _send_email(self, alert: Alert):
        if not CONFIG.alerts.email_enabled:
            return
        if not CONFIG.alerts.smtp_server or not CONFIG.alerts.email_recipients:
            return

        try:
            msg = MIMEMultipart()
            msg["Subject"] = f"[{alert.priority.name}] {alert.title}"
            from_addr = CONFIG.alerts.from_email or CONFIG.alerts.smtp_username or "trading-system@localhost"
            msg["From"] = f"Trading System <{from_addr}>"
            msg["To"] = ", ".join(CONFIG.alerts.email_recipients)
            created_at = alert.created_at or datetime.now()
            body = (
                "Trading System Alert\n"
                "====================\n\n"
                f"Category: {alert.category.value}\n"
                f"Priority: {alert.priority.name}\n"
                f"Time: {created_at.isoformat()}\n\n"
                f"{alert.message}\n\n"
                "Details:\n"
                f"{json.dumps(alert.details, indent=2)}\n"
            )
            msg.attach(MIMEText(body, "plain"))

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
        if not CONFIG.alerts.webhook_enabled or not CONFIG.alerts.webhook_url:
            return
        payload = {
            "alert": alert.to_dict(),
            "timestamp": datetime.now().isoformat(),
            "system": "trading_system",
        }
        last_error = None
        for _ in range(2):
            try:
                response = requests.post(
                    CONFIG.alerts.webhook_url,
                    json=payload,
                    timeout=10,
                )
                response.raise_for_status()
                log.info(f"Webhook alert sent: {alert.title}")
                return
            except Exception as e:
                last_error = e
        log.error(f"Webhook send failed: {last_error}")

    def acknowledge(self, alert_id: str, by: str = "user") -> bool:
        with self._lock:
            alert = self._pending_ack.get(alert_id)
            if alert is None:
                return False
            alert.acknowledged_at = datetime.now()
            alert.acknowledged_by = by
            del self._pending_ack[alert_id]
            self._audit.log_risk_event(
                "alert_acknowledged",
                {"alert_id": alert_id, "acknowledged_by": by},
            )
            return True

    def get_pending(self) -> list[Alert]:
        with self._lock:
            return list(self._pending_ack.values())

    def get_history(
        self,
        category: AlertCategory | None = None,
        priority: AlertPriority | None = None,
        limit: int = 100,
    ) -> list[Alert]:
        with self._lock:
            alerts = list(self._history)
        if category:
            alerts = [a for a in alerts if a.category == category]
        if priority:
            alerts = [a for a in alerts if a.priority == priority]
        return alerts[-limit:]

    def get_alert_stats(self) -> dict[str, Any]:
        with self._lock:
            total = len(self._history)
            acked = sum(1 for a in self._history if a.acknowledged_at is not None)
            by_priority: dict[str, int] = defaultdict(int)
            by_category: dict[str, int] = defaultdict(int)
            for alert in self._history:
                by_priority[alert.priority.name] += 1
                by_category[alert.category.value] += 1
            repeat_pairs = [
                (k, int(v))
                for k, v in self._repeat_counter.items()
                if int(v) > 1
            ]
            repeat_pairs.sort(key=lambda item: item[1], reverse=True)
            top_repeats = [
                {"key": key, "count": count}
                for key, count in repeat_pairs[:10]
            ]
            return {
                "total": total,
                "acknowledged": acked,
                "ack_rate": (acked / total) if total > 0 else 0.0,
                "pending_ack": len(self._pending_ack),
                "by_priority": dict(by_priority),
                "by_category": dict(by_category),
                "channel_delivery": dict(self._channel_stats),
                "top_repeats": top_repeats,
            }

    def set_desktop_callback(self, callback: Callable):
        self._desktop_callback = callback

    def _save_history(self):
        path = CONFIG.data_dir / self.HISTORY_FILE
        try:
            data = [a.to_dict() for a in self._history[-500:]]
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            log.error(f"Failed to save alert history: {e}")

    def _load_history(self):
        path = CONFIG.data_dir / self.HISTORY_FILE
        if not path.exists():
            return
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            for item in data:
                alert = Alert(
                    id=item.get("id", ""),
                    category=AlertCategory(item.get("category", "system")),
                    priority=AlertPriority(item.get("priority", 2)),
                    title=item.get("title", ""),
                    message=item.get("message", ""),
                    details=item.get("details", {}),
                )
                raw_ts = item.get("created_at")
                if raw_ts:
                    alert.created_at = datetime.fromisoformat(raw_ts)
                self._history.append(alert)
        except Exception as e:
            log.error(f"Failed to load alert history: {e}")

    def risk_alert(self, title: str, message: str, details: dict | None = None):
        self.send(
            Alert(
                category=AlertCategory.RISK,
                priority=AlertPriority.HIGH,
                title=title,
                message=message,
                details=details or {},
                channels=[AlertChannel.LOG, AlertChannel.DESKTOP, AlertChannel.EMAIL],
                throttle_key=title,
            )
        )

    def trading_alert(self, title: str, message: str, details: dict | None = None):
        self.send(
            Alert(
                category=AlertCategory.TRADING,
                priority=AlertPriority.MEDIUM,
                title=title,
                message=message,
                details=details or {},
                channels=[AlertChannel.LOG, AlertChannel.DESKTOP],
                throttle_key=title,
            )
        )

    def system_alert(
        self,
        title: str,
        message: str,
        priority: AlertPriority = AlertPriority.MEDIUM,
    ):
        self.send(
            Alert(
                category=AlertCategory.SYSTEM,
                priority=priority,
                title=title,
                message=message,
                channels=[AlertChannel.LOG, AlertChannel.DESKTOP],
                throttle_key=title,
            )
        )

    def critical_alert(self, title: str, message: str, details: dict | None = None):
        self.send(
            Alert(
                category=AlertCategory.SYSTEM,
                priority=AlertPriority.CRITICAL,
                title=title,
                message=message,
                details=details or {},
                channels=[
                    AlertChannel.LOG,
                    AlertChannel.DESKTOP,
                    AlertChannel.EMAIL,
                    AlertChannel.WEBHOOK,
                ],
                throttle_key="",  # Never throttle critical.
            )
        )


_alert_manager: AlertManager | None = None
_alert_lock = threading.Lock()


def get_alert_manager() -> AlertManager:
    global _alert_manager
    if _alert_manager is None:
        with _alert_lock:
            if _alert_manager is None:
                _alert_manager = AlertManager()
    return _alert_manager
