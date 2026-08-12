from __future__ import annotations

import warnings
from pathlib import Path

import pytest

from src.logging_setup import configure_warning_presentation, setup_logging
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


def test_known_external_warnings_are_shown_once() -> None:
    bytetrack_message = "The `ByteTrack` was deprecated since v0.28.0. It will be removed in v0.31.0."

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("default")
        configure_warning_presentation()
        warnings.warn(bytetrack_message, FutureWarning, stacklevel=1)
        warnings.warn(bytetrack_message, FutureWarning, stacklevel=1)

    messages = [str(item.message) for item in captured]
    assert messages.count(bytetrack_message) == 1
