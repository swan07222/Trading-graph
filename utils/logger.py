# utils/logger.py
"""
Logging utility with colored output and file logging
"""
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional
import threading

# Try to import colorama for Windows color support
try:
    import colorama
    colorama.init()
    HAS_COLORAMA = True
except ImportError:
    HAS_COLORAMA = False


class ColorFormatter(logging.Formatter):
    """Custom formatter with colors"""
    
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
        'RESET': '\033[0m',
    }
    
    def format(self, record):
        # Preserve original to avoid affecting other handlers/formatters
        original_levelname = record.levelname
        try:
            levelname = record.levelname
            if levelname in self.COLORS:
                record.levelname = f"{self.COLORS[levelname]}{levelname}{self.COLORS['RESET']}"
            return super().format(record)
        finally:
            record.levelname = original_levelname


class LoggerManager:
    """Thread-safe logger manager"""
    
    _instance = None
    _lock = threading.Lock()
    _loggers: dict = {}
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._initialized = True
        self._log_dir: Optional[Path] = None
        self._file_handler: Optional[logging.FileHandler] = None
        self._log_level = logging.INFO
    
    def setup(self, log_dir: Path = None, level: int = logging.INFO):
        """Setup logging and update existing loggers too (safe hot setup)."""
        self._log_level = level

        if log_dir:
            self._log_dir = Path(log_dir)
            self._log_dir.mkdir(parents=True, exist_ok=True)

            log_file = self._log_dir / f"trading_{datetime.now().strftime('%Y%m%d')}.log"
            self._file_handler = logging.FileHandler(log_file, encoding="utf-8")
            self._file_handler.setLevel(level)
            self._file_handler.setFormatter(
                logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s")
            )

        # Update any already-created loggers
        for lg in self._loggers.values():
            try:
                lg.setLevel(level)
                for h in lg.handlers:
                    h.setLevel(level)

                if self._file_handler:
                    if not any(isinstance(h, logging.FileHandler) for h in lg.handlers):
                        lg.addHandler(self._file_handler)
            except Exception:
                pass
    
    def get_logger(self, name: str = "trading") -> logging.Logger:
        """Get or create a logger"""
        if name in self._loggers:
            return self._loggers[name]
        
        logger = logging.getLogger(name)
        logger.setLevel(self._log_level)
        
        # Prevent duplicate handlers
        if not logger.handlers:
            # Console handler with colors
            console = logging.StreamHandler(sys.stdout)
            console.setLevel(self._log_level)
            
            if sys.stdout.isatty() or HAS_COLORAMA:
                console.setFormatter(ColorFormatter(
                    '%(asctime)s | %(name)s | %(levelname)s | %(message)s',
                    datefmt='%H:%M:%S'
                ))
            else:
                console.setFormatter(logging.Formatter(
                    '%(asctime)s | %(name)s | %(levelname)s | %(message)s',
                    datefmt='%H:%M:%S'
                ))
            
            logger.addHandler(console)
            
            # File handler if configured
            if self._file_handler:
                logger.addHandler(self._file_handler)
        
        # Prevent propagation to root logger
        logger.propagate = False
        
        self._loggers[name] = logger
        return logger


# Global manager instance
_manager = LoggerManager()


def get_logger(name: str = "trading") -> logging.Logger:
    """Get a logger instance
    
    Args:
        name: Logger name (default: "trading")
        
    Returns:
        logging.Logger instance
    """
    return _manager.get_logger(name)


def setup_logging(log_dir: Path = None, level: int = logging.INFO):
    """Setup global logging configuration
    
    Args:
        log_dir: Directory for log files (optional)
        level: Logging level (default: INFO)
    """
    _manager.setup(log_dir, level)


# Default logger instance for convenience
log = get_logger("trading")