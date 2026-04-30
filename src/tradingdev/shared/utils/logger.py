"""Logging configuration for the tradingdev package."""

import logging


def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Create and configure a logger with a StreamHandler.

    Args:
        name: Logger name, typically ``__name__`` of the calling module.
        level: Logging level (default: INFO).

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger
