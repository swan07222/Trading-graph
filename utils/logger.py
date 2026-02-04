"""
Logging Configuration
"""
import sys
from pathlib import Path
from loguru import logger

from config import CONFIG


def setup_logger():
    """Configure application logging"""
    logger.remove()
    
    # Console
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan> - <level>{message}</level>",
        level="INFO",
        colorize=True
    )
    
    # File - all logs
    logger.add(
        CONFIG.LOG_DIR / "app_{time:YYYY-MM-DD}.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level="DEBUG",
        rotation="00:00",
        retention="30 days",
        compression="zip"
    )
    
    # File - errors only
    logger.add(
        CONFIG.LOG_DIR / "errors.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} - {message}\n{exception}",
        level="ERROR",
        rotation="10 MB",
        retention="90 days"
    )
    
    return logger


log = setup_logger()