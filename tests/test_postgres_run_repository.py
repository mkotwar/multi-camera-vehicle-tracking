from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from src.postgres_run_repository import (
    PostgresRepositoryError,
    PostgresRunRepository,
    PostgresRunRepositoryConfig,
)
from src.vehicle_analytics import count_by_class, count_by_colour


class FakePostgresRunRepository(PostgresRunRepository):
    def __init__(self, rows_by_kind: dict[str, list[dict[str, Any]]], tmp_path: Path) -> None:
        super().__init__(PostgresRunRepositoryConfig(dsn="postgresql://example/db"), outputs_root=tmp_path)
        self.rows_by_kind = rows_by_kind

    def _fetchall(self, sql: str, params: tuple[Any, ...]) -> list[dict[str, Any]]:
        normalized = " ".join(sql.split())
        if "from \"vehicle_analytics\".\"processing_runs\" r" in normalized and "where r.run_key" in normalized:
            return [{**self.rows_by_kind["runs"][0], "id": "run-id", "config_snapshot": {}, "metadata": {}}]
        if "from \"vehicle_analytics\".\"processing_runs\" r" in normalized:
            assert "coalesce(r.completed_at, r.started_at, r.created_at) desc" in normalized
            return self.rows_by_kind.get("runs", [])
        if "select run_key, output_directory from" in normalized:
            run_key = str(params[0])
            return [row for row in self.rows_by_kind.get("run_lookup", []) if row["run_key"] == run_key]
        if "from \"vehicle_analytics\".\"vehicle_tracks\" t" in normalized:
            rows = list(self.rows_by_kind.get("tracks", []))
            if params:
                values = {str(item).upper() for item in params}
                if "COMPLETED" in values:
                    rows = [row for row in rows if str(row.get("track_status")).upper() == "COMPLETED"]
                if "WHITE" in values:
                    rows = [row for row in rows if str(row.get("vehicle_colour")).upper() == "WHITE"]
                if "BLACK" in values:
                    rows = [row for row in rows if str(row.get("vehicle_colour")).upper() == "BLACK"]
                if "CAR" in values:
                    rows = [row for row in rows if str(row.get("vehicle_class")).upper() == "CAR"]
                if "CAM_001" in values:
                    rows = [row for row in rows if str(row.get("camera_key")) == "CAM_001"]
                for value in params:
                    if str(value).startswith("TRACK_") or ":" in str(value):
                        rows = [row for row in rows if str(row.get("local_track_id")) == value or str(row.get("local_track_id")).endswith(f":{value}")]
                if any(isinstance(item, float) for item in params):
                    if 5.0 in params:
                        rows = [row for row in rows if row.get("last_seen_seconds") is None or float(row["last_seen_seconds"]) >= 5.0]
                    if 10.0 in params:
                        rows = [row for row in rows if row.get("first_seen_seconds") is None or float(row["first_seen_seconds"]) <= 10.0]
            return rows
        if "from \"vehicle_analytics\".\"physical_vehicles\" v" in normalized:
            return self.rows_by_kind.get("physical_vehicles", [])
        if "from \"vehicle_analytics\".\"track_evidence\" e" in normalized:
            return [row for row in self.rows_by_kind.get("evidence", []) if row["track_id"] == params[0]]
        if "from \"vehicle_analytics\".\"colour_predictions\"" in normalized:
            return [row for row in self.rows_by_kind.get("colour_predictions", []) if row["track_id"] == params[0]]
        if "from \"vehicle_analytics\".\"run_cameras\" c" in normalized:
            return self.rows_by_kind.get("cameras", [])
        return []


def _rows(tmp_path: Path) -> dict[str, list[dict[str, Any]]]:
    run_dir = tmp_path / "20260814_181311"
    crop = run_dir / "05_florence_selected_crops" / "CAM_001" / "TRACK_5" / "frame_000006_MIDDLE.jpg"
    crop.parent.mkdir(parents=True)
    crop.write_bytes(b"crop")
    return {
        "runs": [
            {
                "run_key": "20260814_181311",
                "status": "COMPLETED",
                "started_at": "2026-08-14T18:13:11+05:30",
                "completed_at": "2026-08-14T18:20:00+05:30",
                "output_directory": str(run_dir),
                "summary": {"processed_frames": 100, "overall_pipeline_runtime_ms": 2000},
                "metrics": {},
                "camera_count": 1,
                "track_count": 3,
            }
        ],
        "run_lookup": [{"run_key": "20260814_181311", "output_directory": str(run_dir)}],
        "tracks": [
            {
                "id": "track-white-car",
                "run_key": "20260814_181311",
                "output_directory": str(run_dir),
                "camera_key": "CAM_001",
                "local_track_id": "CAM_001:TRACK_5",
                "track_status": "COMPLETED",
                "first_frame": 1,
                "last_frame": 9,
                "first_seen_seconds": 6.0,
                "last_seen_seconds": 8.0,
                "observation_count": 3,
                "completion_reason": "END_OF_STREAM",
                "vehicle_class": "CAR",
                "vehicle_colour": "WHITE",
                "vehicle_colour_status": "completed",
                "plate_text": "DL8CAF5030",
                "plate_detected": True,
                "plate_colour": "WHITE",
                "registration_category": "PRIVATE",
                "plate_detection_confidence": 0.91,
                "plate_text_confidence": 0.86,
                "plate_quality_status": "plate_quality_accepted",
                "plate_ocr_reason": "ocr_completed",
                "plate_crop_path": "05_florence_selected_crops/CAM_001/TRACK_5/plate/frame_000006_MIDDLE_plate.jpg",
                "enrichment_summary": {"status": "completed"},
                "raw_track": {},
            },
            {
                "id": "track-black-bike",
                "run_key": "20260814_181311",
                "output_directory": str(run_dir),
                "camera_key": "CAM_001",
                "local_track_id": "CAM_001:TRACK_6",
                "track_status": "COMPLETED",
                "first_frame": 10,
                "last_frame": 20,
                "first_seen_seconds": 20.0,
                "last_seen_seconds": 25.0,
                "observation_count": 4,
                "completion_reason": "END_OF_STREAM",
                "vehicle_class": "MOTORCYCLE",
                "vehicle_colour": "BLACK",
                "vehicle_colour_status": "completed",
                "enrichment_summary": {"status": "completed"},
                "raw_track": {},
            },
            {
                "id": "track-discarded",
                "run_key": "20260814_181311",
                "output_directory": str(run_dir),
                "camera_key": "CAM_001",
                "local_track_id": "CAM_001:TRACK_7",
                "track_status": "DISCARDED",
                "first_frame": 30,
                "last_frame": 32,
                "first_seen_seconds": 30.0,
                "last_seen_seconds": 31.0,
                "observation_count": 1,
                "completion_reason": "SHORT_TRACK",
                "vehicle_class": "CAR",
                "vehicle_colour": "WHITE",
                "vehicle_colour_status": "completed",
                "enrichment_summary": {"status": "completed"},
                "raw_track": {},
            },
        ],
        "evidence": [
            {
                "track_id": "track-white-car",
                "frame_number": 6,
                "timestamp_seconds": 6.0,
                "evidence_role": "MIDDLE",
                "detection_confidence": 0.91,
                "quality_score": 0.8,
                "sharpness_score": 0.7,
                "brightness_score": 0.6,
                "crop_width": 120,
                "crop_height": 80,
                "selected_for_colour": True,
                "evidence_source": "colour",
                "bbox": [1, 2, 3, 4],
                "crop_path": "05_florence_selected_crops/CAM_001/TRACK_5/frame_000006_MIDDLE.jpg",
            }
        ],
        "colour_predictions": [
            {
                "track_id": "track-white-car",
                "predicted_colour": "WHITE",
                "normalized_colour": "WHITE",
                "confidence": 0.9,
                "status": "completed",
                "evidence_frame_number": 6,
                "metadata": {"evidence_role": "MIDDLE", "reason": "valid"},
            }
        ],
        "cameras": [
            {
                "run_key": "20260814_181311",
                "camera_key": "CAM_001",
                "source": "video.mp4",
                "source_type": "file",
                "fps": 25,
                "total_frames": 100,
                "processed_frames": 100,
                "timestamp_seconds": 25.0,
                "frame_number": 20,
                "active_vehicle_count": 3,
            }
        ],
    }


def test_postgres_repository_runs_tracks_filters_and_evidence(tmp_path: Path) -> None:
    repository = FakePostgresRunRepository(_rows(tmp_path), tmp_path)

    assert repository.latest_run_id() == "20260814_181311"
    assert repository.resolve_run_id("latest") == "20260814_181311"
    assert repository.list_runs()[0]["track_count"] == 3
    assert repository.get_run("20260814_181311")["summary"]["processed_frames"] == 100

    tracks = repository.list_tracks(run_id="20260814_181311")
    assert len(tracks) == 3
    assert repository.list_tracks(run_id="20260814_181311", status="COMPLETED")
    assert [item["local_track_id"] for item in repository.list_tracks(run_id="20260814_181311", vehicle_class="CAR", colour="WHITE")] == [
        "CAM_001:TRACK_5",
        "CAM_001:TRACK_7",
    ]
    assert [item["local_track_id"] for item in repository.list_tracks(run_id="20260814_181311", from_time=5.0, to_time=10.0)] == [
        "CAM_001:TRACK_5",
    ]

    track = repository.get_track(camera_id="CAM_001", track_id="TRACK_5", run_id="20260814_181311")
    assert track is not None
    assert track["local_track_id"] == "CAM_001:TRACK_5"
    assert track["plate_text"] == "DL8CAF5030"
    assert track["plate_detected"] is True
    assert track["plate_text_confidence"] == 0.86
    assert track["best_crop_parts"]["category"] == "florence_selected_crops"
    assert len(track["evidence"]) == 1
    assert track["colour_resolution"][0]["label"] == "WHITE"


def test_postgres_repository_completed_analytics_semantics(tmp_path: Path) -> None:
    repository = FakePostgresRunRepository(_rows(tmp_path), tmp_path)

    records = repository.list_vehicle_records(run_id="20260814_181311")

    assert len(records) == 2
    assert count_by_class(records)["CAR"] == 1
    assert count_by_colour(records)["WHITE"] == 1
    assert count_by_colour(records)["BLACK"] == 1
    assert records[0].plate_text == "DL8CAF5030"


def test_postgres_repository_uses_physical_vehicle_counts_and_records_when_available(tmp_path: Path) -> None:
    rows = _rows(tmp_path)
    run_dir = tmp_path / "20260814_181311"
    rows["runs"][0] = {
        **rows["runs"][0],
        "track_count": 2,
        "raw_track_count": 3,
        "completed_track_count": 2,
        "physical_vehicle_count": 1,
    }
    rows["physical_vehicles"] = [
        {
            "run_key": "20260814_181311",
            "id": "vehicle-one",
            "vehicle_key": "VEHICLE_001",
            "vehicle_class": "CAR",
            "vehicle_colour": "WHITE",
            "first_timestamp_seconds": 6.0,
            "last_timestamp_seconds": 25.0,
            "identity_confidence": 0.8,
            "identity_method": "production",
            "identity_status": "completed",
            "consensus_plate_text": None,
            "plate_confidence": None,
            "metadata": {
                "representative_evidence": [
                    {
                        "local_track_id": "CAM_001:TRACK_5",
                        "vehicle_crop_path": str(run_dir / "evidence" / "CAM_001" / "CAM_001_TRACK_5" / "crops" / "frame_000006.jpg"),
                    }
                ]
            },
            "member_track_ids": ["CAM_001:TRACK_5", "CAM_001:TRACK_6"],
            "camera_ids": ["CAM_001"],
        }
    ]
    repository = FakePostgresRunRepository(rows, tmp_path)

    summary = repository.list_runs()[0]
    assert summary["track_count"] == 1
    assert summary["physical_vehicle_count"] == 1
    assert summary["raw_track_count"] == 3
    assert summary["completed_track_count"] == 2

    records = repository.list_vehicle_records(run_id="20260814_181311")
    assert len(records) == 1
    assert records[0].vehicle_id == "VEHICLE_001"
    assert records[0].member_track_ids == ("CAM_001:TRACK_5", "CAM_001:TRACK_6")
    assert records[0].plate_text is None


def test_postgres_repository_connection_failure_is_clear(tmp_path: Path) -> None:
    class FailingRepository(PostgresRunRepository):
        def _fetchall(self, sql: str, params: tuple[Any, ...]) -> list[dict[str, Any]]:
            raise PostgresRepositoryError("PostgreSQL read failed: OperationalError: unavailable")

    repository = FailingRepository(PostgresRunRepositoryConfig(dsn="postgresql://example/db"), outputs_root=tmp_path)

    with pytest.raises(PostgresRepositoryError, match="PostgreSQL read failed"):
        repository.list_runs()


def test_postgres_repository_physical_vehicle_plate_filter_uses_normalized_text(tmp_path: Path) -> None:
    class CapturingRepository(FakePostgresRunRepository):
        def __init__(self, rows_by_kind: dict[str, list[dict[str, Any]]], tmp_path: Path) -> None:
            super().__init__(rows_by_kind, tmp_path)
            self.last_sql = ""
            self.last_params: tuple[Any, ...] = ()

        def _fetchall(self, sql: str, params: tuple[Any, ...]) -> list[dict[str, Any]]:
            normalized = " ".join(sql.split())
            if "from \"vehicle_analytics\".\"physical_vehicles\" v" in normalized:
                self.last_sql = normalized
                self.last_params = params
                return self.rows_by_kind.get("physical_vehicles", [])
            return super()._fetchall(sql, params)

    rows = _rows(tmp_path)
    rows["physical_vehicles"] = [
        {
            "run_key": "20260814_181311",
            "vehicle_key": "VEHICLE_001",
            "vehicle_class": "CAR",
            "vehicle_colour": "WHITE",
            "consensus_plate_text": "DL6C Q1126",
            "member_track_ids": ["CAM_001:TRACK_5"],
            "camera_ids": ["CAM_001"],
        }
    ]
    repository = CapturingRepository(rows, tmp_path)

    repository.list_physical_vehicles(run_id="20260814_181311", plate_text="dl6cq-1126")

    assert "regexp_replace(upper(coalesce(v.consensus_plate_text, '')), '[^A-Z0-9]+', '', 'g') = %s" in repository.last_sql
    assert repository.last_params[-1] == "DL6CQ1126"
