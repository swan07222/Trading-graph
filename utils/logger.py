# utils/logger.py
from __future__ import annotations

import copy
import logging
import logging.handlers
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict
import threading

_colorama_lock = threading.Lock()
_colorama_initialized = False
_colorama_available = False

def _ensure_colorama() -> bool:
    """
    Thread-safe, lazy colorama initialization.
    Returns True if colorama is available and initialized.
    """
    global _colorama_initialized, _colorama_available

    if _colorama_initialized:
        return _colorama_available

    with _colorama_lock:
        if _colorama_initialized:
            return _colorama_available
        try:
            import colorama

            colorama.init()
            _colorama_available = True
        except ImportError:
            _colorama_available = False
        _colorama_initialized = True
        return _colorama_available

class ColorFormatter(logging.Formatter):
    """
    Custom formatter with ANSI colors for terminal output.

    Uses a shallow copy of the LogRecord instead of mutating
    the original, which is shared across handlers in multi-handler setups.
    """

    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        record_copy = copy.copy(record)
        color = self.COLORS.get(record_copy.levelname)
        if color:
            record_copy.levelname = f"{color}{record_copy.levelname}{self.RESET}"
        return super().format(record_copy)

class SafeConsoleHandler(logging.StreamHandler):
    """
    Console handler resilient to terminal encoding limitations.

    On Windows GBK consoles, symbols like '¥' can raise UnicodeEncodeError.
    This handler falls back to ASCII-safe output instead of emitting
    logging internal tracebacks.
    """

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            stream = self.stream
            try:
                stream.write(msg + self.terminator)
            except UnicodeEncodeError:
                enc = getattr(stream, "encoding", None) or "utf-8"
                safe = msg.encode(enc, errors="replace").decode(
                    enc, errors="replace"
                )
                stream.write(safe + self.terminator)
            self.flush()
        except Exception:
            self.handleError(record)

class LoggerManager:
    """
    Thread-safe singleton logger manager.

    Creates and caches loggers with consistent formatting.
    Supports optional file logging with rotation.
    """

    _instance: Optional[LoggerManager] = None
    _instance_lock = threading.Lock()

    def __new__(cls) -> LoggerManager:
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    inst = super().__new__(cls)
                    inst._initialized = False
                    cls._instance = inst
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Destroy singleton for testing."""
        with cls._instance_lock:
            if cls._instance is not None:
                cls._instance.teardown()
            cls._instance = None

    def __init__(self) -> None:
        if self._initialized:
            return

        self._initialized = True
        self._loggers: Dict[str, logging.Logger] = {}
        self._logger_lock = threading.Lock()
        self._log_dir: Optional[Path] = None
        self._file_handler: Optional[logging.Handler] = None
        self._log_level: int = logging.INFO

    def setup(
        self,
        log_dir: Optional[Path] = None,
        level: int = logging.INFO,
        max_bytes: int = 10 * 1024 * 1024,  # 10 MB
        backup_count: int = 5,
    ) -> None:
        """
        Setup or reconfigure logging.

        Safe to call multiple times — closes previous file handler first.
        """
        with self._logger_lock:
            self._log_level = level

            self._close_file_handler()

            if log_dir is not None:
                self._log_dir = Path(log_dir)
                self._log_dir.mkdir(parents=True, exist_ok=True)

                log_file = (
                    self._log_dir
                    / f"trading_{datetime.now().strftime('%Y%m%d')}.log"
                )

                self._file_handler = logging.handlers.RotatingFileHandler(
                    log_file,
                    maxBytes=max_bytes,
                    backupCount=backup_count,
                    encoding="utf-8",
                )
                self._file_handler.setLevel(level)
                self._file_handler.setFormatter(
                    logging.Formatter(
                        "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
                    )
                )

            # Update all already-created loggers
            for logger in self._loggers.values():
                self._apply_level(logger, level)
                self._attach_file_handler(logger)

    def get_logger(self, name: str = "trading") -> logging.Logger:
        """Get or create a named logger. Thread-safe, cached."""
        with self._logger_lock:
            if name in self._loggers:
                return self._loggers[name]

            logger = logging.getLogger(name)
            logger.setLevel(self._log_level)
            logger.propagate = False

            # Clear any pre-existing handlers to prevent duplicates
            existing_handlers = logger.handlers[:]
            for h in existing_handlers:
                logger.removeHandler(h)

            console = self._create_console_handler()
            logger.addHandler(console)

            if self._file_handler is not None:
                logger.addHandler(self._file_handler)

            self._loggers[name] = logger
            return logger

    def teardown(self) -> None:
        """
        Remove all handlers and clear cached loggers.
        Call during shutdown or between tests.
        """
        with self._logger_lock:
            file_handler = self._file_handler

            for logger in self._loggers.values():
                for handler in logger.handlers[:]:
                    if handler is file_handler:
                        logger.removeHandler(handler)
                        continue
                    try:
                        handler.close()
                    except Exception:
                        pass
                    logger.removeHandler(handler)
            self._loggers.clear()

        self._close_file_handler()

    # -----------------------------------------------------------------
    # -----------------------------------------------------------------

    def _create_console_handler(self) -> logging.StreamHandler:
        """Create a console handler with optional color formatting."""
        handler = SafeConsoleHandler(sys.stdout)
        handler.setLevel(self._log_level)

        use_color = False
        try:
            if sys.stdout.isatty():
                use_color = True
                if sys.platform == "win32":
                    use_color = _ensure_colorama()
        except Exception:
            pass

        fmt_string = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
        datefmt = "%H:%M:%S"

        if use_color:
            handler.setFormatter(ColorFormatter(fmt_string, datefmt=datefmt))
        else:
            handler.setFormatter(
                logging.Formatter(fmt_string, datefmt=datefmt)
            )

        return handler

    def _close_file_handler(self) -> None:
        """Close and discard the current file handler (idempotent)."""
        handler = self._file_handler
        if handler is not None:
            self._file_handler = None
            try:
                handler.close()
            except Exception:
                pass

    def _apply_level(self, logger: logging.Logger, level: int) -> None:
        """Set level on logger and all its handlers."""
        logger.setLevel(level)
        for handler in logger.handlers:
            handler.setLevel(level)

    def _attach_file_handler(self, logger: logging.Logger) -> None:
        """Add file handler if not already attached."""
        if self._file_handler is None:
            return
        if not any(h is self._file_handler for h in logger.handlers):
            logger.addHandler(self._file_handler)

# Module-level convenience API

_manager = LoggerManager()

def get_logger(name: str = "trading") -> logging.Logger:
    """Get a logger instance."""
    return _manager.get_logger(name)

def setup_logging(
    log_dir: Optional[Path] = None,
    level: int = logging.INFO,
) -> None:
    """Setup global logging configuration."""
    _manager.setup(log_dir, level)

def teardown_logging() -> None:
    """Tear down logging — close handlers, clear caches."""
    _manager.teardown()

# This logger is created before setup_logging() runs.
# It will be updated when setup_logging() is called later.
log: logging.Logger = get_logger("trading")
