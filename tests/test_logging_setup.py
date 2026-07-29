from __future__ import annotations

from pathlib import Path

import pytest

from src.logging_setup import setup_logging
from src.models import ConfigurationError


def test_logging_creates_pipeline_log(tmp_path: Path) -> None:
    logger = setup_logging(tmp_path, "INFO")
    logger.info("hello")
    assert (tmp_path / "pipeline.log").exists()


def test_logging_does_not_duplicate_handlers(tmp_path: Path) -> None:
    logger = setup_logging(tmp_path, "INFO")
    handler_count_1 = len(logger.handlers)
    logger = setup_logging(tmp_path, "INFO")
    handler_count_2 = len(logger.handlers)
    assert handler_count_1 == 2
    assert handler_count_2 == 2


def test_invalid_log_level_raises_configuration_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError):
        setup_logging(tmp_path, "NOPE")
