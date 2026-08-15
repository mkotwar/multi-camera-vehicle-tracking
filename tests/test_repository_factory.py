from __future__ import annotations

from pathlib import Path

import pytest

from src.postgres_run_repository import PostgresRunRepository
from src.repository_factory import RepositoryConfigurationError, configured_data_source, get_run_repository
from src.run_repository import RunRepository


def test_default_data_source_is_files(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("DATA_SOURCE", raising=False)

    assert configured_data_source() == "files"
    assert isinstance(get_run_repository(outputs_root=tmp_path), RunRepository)


def test_data_source_files(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DATA_SOURCE", "files")

    assert isinstance(get_run_repository(outputs_root=tmp_path), RunRepository)


def test_data_source_postgres(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DATA_SOURCE", "postgres")
    monkeypatch.setenv("DATABASE_URL", "postgresql://user:pass@example.invalid/db")
    monkeypatch.setenv("DB_SCHEMA", "vehicle_analytics")

    repository = get_run_repository(outputs_root=tmp_path)

    assert isinstance(repository, PostgresRunRepository)
    assert repository.config.schema == "vehicle_analytics"


def test_invalid_data_source_fails_clearly(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATA_SOURCE", "sqlite")

    with pytest.raises(RepositoryConfigurationError, match="Invalid DATA_SOURCE"):
        configured_data_source()
