from __future__ import annotations

import logging
from pathlib import Path

from .models import ConfigurationError


VALID_LOG_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}


def setup_logging(run_directory: Path, log_level: str = "INFO") -> logging.Logger:
    normalized_level = str(log_level).upper()
    if normalized_level not in VALID_LOG_LEVELS:
        raise ConfigurationError(f"Invalid log level: {log_level}")

    run_directory.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("pipeline")
    logger.setLevel(getattr(logging, normalized_level))
    logger.propagate = False

    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s", "%Y-%m-%d %H:%M:%S")

    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, normalized_level))
    console_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(run_directory / "pipeline.log", encoding="utf-8")
    file_handler.setLevel(getattr(logging, normalized_level))
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger
