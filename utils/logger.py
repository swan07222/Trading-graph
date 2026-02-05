"""
Production Logging Configuration
"""
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional
import logging
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
import threading

from config.settings import CONFIG


class ColoredFormatter(logging.Formatter):
    """Colored console formatter - doesn't mutate log record"""
    
    COLORS = {
        'DEBUG': '\033[36m',
        'INFO': '\033[32m',
        'WARNING': '\033[33m',
        'ERROR': '\033[31m',
        'CRITICAL': '\033[41m',
    }
    RESET = '\033[0m'
    
    def format(self, record):
        # Don't mutate the original record
        color = self.COLORS.get(record.levelname, '')
        
        # Save original levelname
        original_levelname = record.levelname
        
        try:
            # Temporarily modify for colored output
            record.levelname = f"{color}{record.levelname:8}{self.RESET}"
            return super().format(record)
        finally:
            # Restore original
            record.levelname = original_levelname


def setup_logging(name: str = None) -> logging.Logger:
    """Setup logging for module"""
    logger = logging.getLogger(name or 'trading')
    
    if logger.handlers:
        return logger
    
    logger.setLevel(logging.DEBUG)
    
    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(ColoredFormatter(
        '%(asctime)s | %(levelname)s | %(name)s - %(message)s',
        datefmt='%H:%M:%S'
    ))
    logger.addHandler(console)
    
    # File handler - all logs
    log_file = CONFIG.log_dir / f"app_{datetime.now().strftime('%Y-%m-%d')}.log"
    file_handler = TimedRotatingFileHandler(
        log_file,
        when='midnight',
        interval=1,
        backupCount=30
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d - %(message)s'
    ))
    logger.addHandler(file_handler)
    
    # Error file handler
    error_file = CONFIG.log_dir / "errors.log"
    error_handler = RotatingFileHandler(
        error_file,
        maxBytes=10*1024*1024,
        backupCount=5
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(logging.Formatter(
        '%(asctime)s | %(levelname)s | %(name)s:%(funcName)s:%(lineno)d\n%(message)s\n'
    ))
    logger.addHandler(error_handler)
    
    return logger


def get_logger(name: str = None) -> logging.Logger:
    """Get logger for module"""
    return setup_logging(name)


# Default logger
log = get_logger()