import logging
import sys
from pathlib import Path
from typing import Optional
import os
from rich.logging import RichHandler
import datetime

# Default log levels
DEFAULT_CONSOLE_LEVEL = logging.INFO
DEFAULT_FILE_LEVEL = logging.DEBUG

# Log format for file logging
FILE_LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s - %(message)s"

# Logger name for the application
LOGGER_NAME = "moghedien"


def setup_logger(
        console_level: int = DEFAULT_CONSOLE_LEVEL,
        file_level: int = DEFAULT_FILE_LEVEL,
        log_file: Optional[str] = None,
        log_dir: Optional[str] = None,
) -> logging.Logger:
    """
    Configure and return the application logger.

    Args:
        console_level: Logging level for console output
        file_level: Logging level for file output
        log_file: Specific log file path to use, or None for auto-generated name
        log_dir: Directory to store log files, or None for default

    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.DEBUG)  # Capture all logs, handlers will filter

    # Clear any existing handlers
    if logger.handlers:
        logger.handlers.clear()

    # Console handler with rich formatting
    console_handler = RichHandler(level=console_level, rich_tracebacks=True)
    console_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(console_handler)

    # File handler if enabled
    if file_level is not None:
        # Determine log file path
        if log_file is None:
            # Generate timestamped log filename
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = f"moghedien_{timestamp}.log"

        # Determine log directory
        if log_dir is None:
            # Use default log directory
            log_dir = Path.home() / ".moghedien" / "logs"
        else:
            log_dir = Path(log_dir)

        # Create log directory if it doesn't exist
        log_dir.mkdir(parents=True, exist_ok=True)

        # Full path to log file
        log_path = log_dir / log_file

        # Add file handler
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(file_level)
        file_handler.setFormatter(logging.Formatter(FILE_LOG_FORMAT))
        logger.addHandler(file_handler)

        logger.info(f"Logging to file: {log_path}")

    return logger


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Get a logger instance for a specific module.

    Args:
        name: Module name, or None to use the root logger

    Returns:
        Logger instance
    """
    if name is None:
        return logging.getLogger(LOGGER_NAME)
    else:
        return logging.getLogger(f"{LOGGER_NAME}.{name}")


def set_log_level(level: int) -> None:
    """
    Set the log level for the console handler.

    Args:
        level: New logging level (e.g., logging.DEBUG)
    """
    logger = logging.getLogger(LOGGER_NAME)

    for handler in logger.handlers:
        if isinstance(handler, RichHandler):
            handler.setLevel(level)
            logger.info(f"Console log level set to: {logging.getLevelName(level)}")
            break