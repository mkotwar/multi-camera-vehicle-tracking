from __future__ import annotations

import os
from pathlib import Path

from src.importers.db_writer import (
    DEFAULT_DB_SCHEMA,
    DatabaseRunWriter,
    DatabaseWriteConfig,
    DatabaseWriteConfigurationError,
    build_payload,
)
from src.importers.run_db_import import main as db_import_main
from src.importers.run_file_importer import build_dry_run
from tests.test_run_file_importer import _base_run


MIGRATION_PATH = Path("database/migrations/202608150002_align_vehicle_analytics_schema.sql")


def test_migration_contains_required_tables_and_no_destructive_sql() -> None:
    sql = MIGRATION_PATH.read_text(encoding="utf-8").lower()
    for table in [
        "processing_runs",
        "run_cameras",
        "vehicle_tracks",
        "track_observations",
        "track_evidence",
        "media_assets",
        "colour_predictions",
        "vehicle_attribute_predictions",
        "plate_detections",
        "plate_readings",
        "pipeline_artifacts",
        "pipeline_errors",
        "physical_vehicles",
        "physical_vehicle_tracks",
        "identity_decisions",
    ]:
        assert f"vehicle_analytics.{table}" in sql
    executable_lines = [line.strip() for line in sql.splitlines() if line.strip() and not line.strip().startswith("--")]
    unexpected_drops = [
        line
        for line in executable_lines
        if line.startswith("drop ")
        and "drop constraint if exists vehicle_tracks_run_id_camera_id_tracker_namespace_native_tr_key" not in line
    ]
    assert unexpected_drops == []
    assert not any(line.startswith("truncate ") for line in executable_lines)
    assert not any(line.startswith("delete ") for line in executable_lines)


def test_payload_counts_match_importer_rows(tmp_path: Path) -> None:
    report = build_dry_run(_base_run(tmp_path))
    payload = build_payload(report)
    assert payload.counts["processing_runs"] == 1
    assert payload.counts["run_cameras"] == len(report.rows.run_cameras)
    assert payload.counts["vehicle_tracks"] == len(report.rows.vehicle_tracks)
    assert payload.counts["track_observations"] == len(report.rows.track_observations)
    assert payload.counts["track_evidence"] == len(report.rows.track_evidence)
    assert payload.counts["media_assets"] == len(report.rows.media_assets)
    assert payload.counts["colour_predictions"] == len(report.rows.colour_predictions)
    assert payload.counts["vehicle_attribute_predictions"] == len(report.rows.vehicle_attribute_predictions)
    assert payload.counts["plate_detections"] == 0
    assert payload.counts["plate_readings"] == 0


def test_payload_preserves_discarded_tracks_and_unknown_values(tmp_path: Path) -> None:
    payload = build_payload(build_dry_run(_base_run(tmp_path)))
    tracks = payload.tables["vehicle_tracks"]
    assert any(row["track_status"] == "DISCARDED" for row in tracks)
    assert any(row["body_type"] == "UNKNOWN" for row in tracks)
    assert any(row["plate_detected"] is False and row["plate_text"] is None for row in tracks)


def test_payload_uses_run_camera_track_logical_identity(tmp_path: Path) -> None:
    payload = build_payload(build_dry_run(_base_run(tmp_path)))
    run_id = payload.tables["processing_runs"][0]["id"]
    camera = payload.tables["run_cameras"][0]
    track = payload.tables["vehicle_tracks"][0]
    assert camera["run_id"] == run_id
    assert track["run_id"] == run_id
    assert track["camera_id"] == camera["id"]
    assert track["local_track_id"] == "CAM_001:TRACK_1"


def test_track_observation_unique_key_payload_is_clean(tmp_path: Path) -> None:
    payload = build_payload(build_dry_run(_base_run(tmp_path)))
    seen = set()
    for row in payload.tables["track_observations"]:
        key = (row["track_id"], row["frame_number"])
        assert key not in seen
        seen.add(key)


def test_evidence_roles_are_text_not_enum_limited() -> None:
    sql = MIGRATION_PATH.read_text(encoding="utf-8").lower()
    assert "create type" not in sql


def test_media_local_relative_paths_are_accepted(tmp_path: Path) -> None:
    payload = build_payload(build_dry_run(_base_run(tmp_path)))
    media = payload.tables["media_assets"]
    assert media
    assert all(row["storage_provider"] == "local" for row in media)
    assert all(row["relative_path"] and not Path(row["relative_path"]).is_absolute() for row in media)


def test_zero_plate_rows_cause_no_issue(tmp_path: Path) -> None:
    payload = build_payload(build_dry_run(_base_run(tmp_path)))
    assert payload.tables["plate_detections"] == []
    assert payload.tables["plate_readings"] == []


def test_chat_session_message_constraints_exist() -> None:
    sql = Path("supabase/migrations/202608140001_create_vehicle_analytics_v1.sql").read_text(encoding="utf-8").lower()
    assert "unique (session_id, message_index)" in sql
    assert "role text not null check" in sql
    assert "chat_session_runs" in sql


def test_database_config_requires_postgres_dsn(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("SUPABASE_DB_URL", raising=False)
    monkeypatch.delenv("POSTGRES_URL", raising=False)
    env_path = tmp_path / ".env"
    env_path.write_text("SUPABASE_URL=https://example.supabase.co\nSUPABASE_SERVICE_ROLE_KEY=secret\n", encoding="utf-8")
    try:
        DatabaseWriteConfig.from_env(env_path)
    except DatabaseWriteConfigurationError as exc:
        assert "no PostgreSQL connection string" in str(exc)
    else:
        raise AssertionError("Expected DatabaseWriteConfigurationError")


def test_database_config_uses_vehicle_analytics_schema_by_default(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("DB_SCHEMA", raising=False)
    env_path = tmp_path / ".env"
    env_path.write_text("DATABASE_URL=postgresql://user:pass@example:5432/vehicle_analytics\n", encoding="utf-8")
    config = DatabaseWriteConfig.from_env(env_path)
    assert config.schema == DEFAULT_DB_SCHEMA
    assert config.observation_batch_size == 1000


def test_database_config_reads_schema_and_batch_size_from_env_file(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("DB_SCHEMA", raising=False)
    monkeypatch.delenv("IMPORT_OBSERVATION_BATCH_SIZE", raising=False)
    env_path = tmp_path / ".env"
    env_path.write_text(
        "DATABASE_URL=postgresql://user:pass@example:5432/vehicle_analytics\nDB_SCHEMA=vehicle_analytics\nIMPORT_OBSERVATION_BATCH_SIZE=5000\n",
        encoding="utf-8",
    )
    config = DatabaseWriteConfig.from_env(env_path)
    assert config.schema == "vehicle_analytics"
    assert config.observation_batch_size == 5000


def test_db_import_cli_default_is_no_database_write(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(Path.cwd())
    run_dir = _base_run(tmp_path)
    assert db_import_main(["--run-dir", str(run_dir)]) == 0


def test_db_import_cli_write_fails_without_dsn(tmp_path: Path, monkeypatch) -> None:
    for key in ["DATABASE_URL", "SUPABASE_DB_URL", "POSTGRES_URL"]:
        monkeypatch.delenv(key, raising=False)
    env_backup = os.environ.get("SUPABASE_URL")
    monkeypatch.setenv("SUPABASE_URL", env_backup or "https://example.supabase.co")
    run_dir = _base_run(tmp_path)
    assert db_import_main(["--run-dir", str(run_dir), "--write-db"]) == 2


def test_db_import_cli_reports_database_connection_failure(tmp_path: Path, monkeypatch) -> None:
    run_dir = _base_run(tmp_path)

    class FailingWriter:
        def __init__(self, config) -> None:
            self.config = config

        def insert_reference_run(self, payload, replace=False):
            raise RuntimeError("connection failed")

    monkeypatch.setenv("DATABASE_URL", "postgresql://user:pass@example:5432/vehicle_analytics")
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.setattr("src.importers.run_db_import.DatabaseRunWriter", FailingWriter)
    assert db_import_main(["--run-dir", str(run_dir), "--write-db"]) == 2


class _FakeCursor:
    def __init__(self, conn: "_FakeConnection") -> None:
        self.conn = conn

    def __enter__(self) -> "_FakeCursor":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def execute(self, sql: str, params=()) -> None:
        self.conn.executed.append((sql, params))

    def executemany(self, sql: str, values: list[tuple]) -> None:
        self.conn.executemany_calls.append((sql, values))

    def fetchall(self) -> list[tuple[str, str]]:
        return []


class _FakeTransaction:
    def __init__(self, conn: "_FakeConnection") -> None:
        self.conn = conn

    def __enter__(self) -> "_FakeTransaction":
        self.conn.transaction_entered = True
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.conn.rolled_back = exc_type is not None
        self.conn.committed = exc_type is None
        return None


class _FakeConnection:
    def __init__(self) -> None:
        self.executed: list[tuple[str, tuple]] = []
        self.executemany_calls: list[tuple[str, list[tuple]]] = []
        self.transaction_entered = False
        self.committed = False
        self.rolled_back = False

    def __enter__(self) -> "_FakeConnection":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def transaction(self) -> _FakeTransaction:
        return _FakeTransaction(self)

    def cursor(self) -> _FakeCursor:
        return _FakeCursor(self)


class _FakeWriter(DatabaseRunWriter):
    def __init__(self, config: DatabaseWriteConfig, conn: _FakeConnection) -> None:
        super().__init__(config)
        self.conn = conn

    def _connect(self):
        return self.conn

    def _load_columns_by_table(self, conn):
        columns = {}
        for table, rows in self.payload.tables.items():
            table_columns = set(rows[0].keys()) if rows else {"id", "run_id", "track_id"}
            if table == "plate_readings":
                table_columns.discard("track_id")
                table_columns.add("plate_detection_id")
            columns[table] = table_columns
        return columns


def test_writer_uses_schema_qualified_upserts_and_batches_observations(tmp_path: Path) -> None:
    payload = build_payload(build_dry_run(_base_run(tmp_path)))
    base = dict(payload.tables["track_observations"][0])
    payload.tables["track_observations"] = [
        {**base, "frame_number": 1},
        {**base, "frame_number": 2},
        {**base, "frame_number": 3},
    ]
    conn = _FakeConnection()
    writer = _FakeWriter(DatabaseWriteConfig(dsn="postgresql://x", schema="vehicle_analytics", observation_batch_size=2), conn)
    writer.payload = payload
    result = writer.insert_reference_run(payload)

    assert conn.transaction_entered is True
    assert conn.committed is True
    observation_calls = [call for call in conn.executemany_calls if "track_observations" in call[0]]
    assert [len(values) for _, values in observation_calls] == [2, 1]
    assert all('"vehicle_analytics".' in sql for sql, _ in conn.executemany_calls)
    assert result.observation_batch_count == 2


class _FailingWriter(_FakeWriter):
    def _insert_rows(self, conn, table, rows, columns_by_table):
        if table == "colour_predictions":
            raise RuntimeError("boom")
        return super()._insert_rows(conn, table, rows, columns_by_table)


def test_writer_transaction_rolls_back_on_failure(tmp_path: Path) -> None:
    payload = build_payload(build_dry_run(_base_run(tmp_path)))
    conn = _FakeConnection()
    writer = _FailingWriter(DatabaseWriteConfig(dsn="postgresql://x"), conn)
    writer.payload = payload
    try:
        writer.insert_reference_run(payload)
    except RuntimeError:
        pass
    else:
        raise AssertionError("Expected import failure")
    assert conn.rolled_back is True
    assert conn.committed is False
