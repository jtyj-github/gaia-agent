"""
Logging utility for the GAIA multi-agent system.
Provides structured logging to both files and console.
"""

import logging
import os
from datetime import datetime
from pathlib import Path


def setup_logger(name: str, log_level: str = "INFO") -> logging.Logger:
    """
    Set up logger with file and console handlers.

    Args:
        name: Name of the logger (typically the module name)
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)

    Returns:
        Configured logger instance
    """
    # Create logs directory
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level.upper()))

    # Avoid duplicate handlers if logger already exists
    if logger.handlers:
        return logger

    # File handler - logs everything to daily log file
    log_file = log_dir / f"{name}_{datetime.now():%Y%m%d}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)

    # Console handler - logs INFO and above to console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger
