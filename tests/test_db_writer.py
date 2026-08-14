from __future__ import annotations

import os
from pathlib import Path

from src.importers.db_writer import DatabaseWriteConfig, DatabaseWriteConfigurationError, build_payload
from src.importers.run_db_import import main as db_import_main
from src.importers.run_file_importer import build_dry_run
from tests.test_run_file_importer import _base_run


MIGRATION_PATH = Path("supabase/migrations/202608140001_create_vehicle_analytics_v1.sql")


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
        "chat_sessions",
        "chat_session_runs",
        "chat_messages",
    ]:
        assert f"create table if not exists public.{table}" in sql
    executable_lines = [line.strip() for line in sql.splitlines() if line.strip() and not line.strip().startswith("--")]
    assert not any(line.startswith("drop ") for line in executable_lines)
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
    assert track["local_track_id"] == "TRACK_1"


def test_track_observation_unique_key_payload_is_clean(tmp_path: Path) -> None:
    payload = build_payload(build_dry_run(_base_run(tmp_path)))
    seen = set()
    for row in payload.tables["track_observations"]:
        key = (row["track_id"], row["frame_number"])
        assert key not in seen
        seen.add(key)


def test_evidence_roles_are_text_not_enum_limited() -> None:
    sql = MIGRATION_PATH.read_text(encoding="utf-8").lower()
    assert "evidence_role text" in sql
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
    sql = MIGRATION_PATH.read_text(encoding="utf-8").lower()
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
