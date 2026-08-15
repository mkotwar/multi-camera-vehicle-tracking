from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .env_loader import load_project_env_defaults
from .postgres_run_repository import PostgresRunRepository, PostgresRunRepositoryConfig
from .run_repository import RunRepository


DEFAULT_DATA_SOURCE = "files"
SUPPORTED_DATA_SOURCES = {"files", "postgres"}


class RepositoryConfigurationError(RuntimeError):
    pass


def configured_data_source() -> str:
    load_project_env_defaults()
    value = str(os.environ.get("DATA_SOURCE") or DEFAULT_DATA_SOURCE).strip().lower()
    if value not in SUPPORTED_DATA_SOURCES:
        raise RepositoryConfigurationError(
            f"Invalid DATA_SOURCE {value!r}. Supported values are: {', '.join(sorted(SUPPORTED_DATA_SOURCES))}."
        )
    return value


def get_run_repository(*, outputs_root: str | Path = "outputs/runs") -> Any:
    data_source = configured_data_source()
    if data_source == "files":
        return RunRepository(outputs_root)
    return PostgresRunRepository(PostgresRunRepositoryConfig.from_env(), outputs_root=outputs_root)
