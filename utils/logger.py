# utils/logger.py
"""
Production Logging Configuration (Fixed for Windows GBK + no duplicate handlers)

Fixes:
- Configure handlers ONLY on base logger "trading"
- Child loggers propagate to "trading" (no duplicate console lines)
- Force UTF-8 console output with safe fallback
- File handlers always UTF-8
"""
from __future__ import annotations

import io
import sys
import logging
from pathlib import Path
from datetime import datetime
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler

from config.settings import CONFIG

_CONFIGURED = False
BASE_LOGGER_NAME = "trading"


class ColoredFormatter(logging.Formatter):
    """Colored console formatter - doesn't mutate record permanently"""

    COLORS = {
        "DEBUG": "\033[36m",
        "INFO": "\033[32m",
        "WARNING": "\033[33m",
        "ERROR": "\033[31m",
        "CRITICAL": "\033[41m",
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, "")
        original_levelname = record.levelname
        try:
            record.levelname = f"{color}{record.levelname:8}{self.RESET}"
            return super().format(record)
        finally:
            record.levelname = original_levelname


class SafeStreamHandler(logging.StreamHandler):
    """Stream handler that safely encodes unicode for Windows consoles"""
    
    def emit(self, record):
        try:
            msg = self.format(record)
            stream = self.stream
            # Replace problematic characters for Windows console
            try:
                stream.write(msg + self.terminator)
            except UnicodeEncodeError:
                # Fallback: encode with replacement
                safe_msg = msg.encode("utf-8", errors="replace").decode("utf-8", errors="replace")
                stream.write(safe_msg + self.terminator)
            self.flush()
        except Exception:
            self.handleError(record)


def _get_safe_stream():
    """
    Return a UTF-8 capable stream for logging on Windows consoles.
    Prevents UnicodeEncodeError for ¥ / emojis.
    """
    try:
        # Try to reconfigure stdout for UTF-8
        if hasattr(sys.stdout, 'reconfigure'):
            sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        return sys.stdout
    except Exception:
        pass
    
    # Fallback wrapper
    try:
        return io.TextIOWrapper(
            sys.stdout.buffer, 
            encoding='utf-8', 
            errors='replace',
            line_buffering=True
        )
    except Exception:
        return sys.stdout


def setup_logging() -> logging.Logger:
    """Configure base logger once."""
    global _CONFIGURED
    
    base_logger = logging.getLogger(BASE_LOGGER_NAME)
    
    if _CONFIGURED:
        return base_logger

    base_logger.setLevel(logging.DEBUG)
    base_logger.propagate = False  # base logger should NOT propagate to root

    # Clear any existing handlers to avoid duplicates
    base_logger.handlers.clear()

    # --- Console handler (UTF-8 safe) ---
    console = SafeStreamHandler(_get_safe_stream())
    console.setLevel(logging.INFO)
    console.setFormatter(
        ColoredFormatter(
            "%(asctime)s | %(levelname)s | %(name)s - %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    base_logger.addHandler(console)

    # --- File handler (all logs) ---
    try:
        CONFIG.log_dir.mkdir(parents=True, exist_ok=True)
        log_file = CONFIG.log_dir / f"app_{datetime.now().strftime('%Y-%m-%d')}.log"
        file_handler = TimedRotatingFileHandler(
            log_file,
            when="midnight",
            interval=1,
            backupCount=30,
            encoding="utf-8",
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d - %(message)s"
            )
        )
        base_logger.addHandler(file_handler)

        # --- Error handler ---
        error_file = CONFIG.log_dir / "errors.log"
        error_handler = RotatingFileHandler(
            error_file,
            maxBytes=10 * 1024 * 1024,
            backupCount=5,
            encoding="utf-8",
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d - %(message)s\n%(exc_info)s"
            )
        )
        base_logger.addHandler(error_handler)
    except Exception as e:
        # If file handlers fail, continue with console only
        base_logger.warning(f"Could not set up file logging: {e}")

    _CONFIGURED = True
    return base_logger


def get_logger(name: str = None) -> logging.Logger:
    """
    Get a logger instance.
    
    All loggers are children of the base 'trading' logger.
    They propagate to the base logger which has the handlers.
    """
    # Ensure base logger is configured
    setup_logging()
    
    if name is None:
        return logging.getLogger(BASE_LOGGER_NAME)
    
    # Create child logger under trading namespace
    if not name.startswith(BASE_LOGGER_NAME):
        name = f"{BASE_LOGGER_NAME}.{name}"
    
    logger = logging.getLogger(name)
    # Child loggers should NOT have their own handlers - they propagate to base
    # Do NOT add handlers here
    return logger


# Convenience: module-level logger
log = get_logger()