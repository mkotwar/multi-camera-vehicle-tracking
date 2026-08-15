from __future__ import annotations

import json
from pathlib import Path

from src.run_repository import RunRepository
from src.vehicle_enrichment.taxonomy import SUPPORTED_VEHICLE_CLASSES, SUPPORTED_VEHICLE_COLOUR_LABELS


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def test_run_repository_lists_runs_and_tracks(tmp_path: Path) -> None:
    run_dir = tmp_path / "20260808_120000"
    _write_json(run_dir / "summary.json", {"run_id": "20260808_120000", "status": "COMPLETED", "configured_camera_count": 2, "processed_frames": 40})
    _write_json(run_dir / "run_metadata.json", {"started_at": "2026-08-08T10:00:00+00:00", "completed_at": "2026-08-08T10:10:00+00:00", "camera_count": 2})
    _write_json(
        run_dir / "tracks.json",
        [
            {
                "local_track_id": "CAM_001:TRACK_1",
                "camera_id": "CAM_001",
                "status": "COMPLETED",
                "first_timestamp_seconds": 1.0,
                "last_timestamp_seconds": 2.0,
                "first_frame": 10,
                "last_frame": 20,
                "observation_count": 5,
                "completion_reason": "END_OF_STREAM",
                "final_class": "car",
            }
        ],
    )
    _write_json(
        run_dir / "vehicle_enrichment.json",
        [
            {
                "local_track_id": "CAM_001:TRACK_1",
                "camera_id": "CAM_001",
                "vehicle_class": "CAR",
                "vehicle_colour": {"label": "WHITE", "status": "completed"},
                "plate_detected": True,
                "plate_text": "DL8CAF5030",
                "plate_text_confidence": 0.86,
                "plate_ocr_reason": "ocr_completed",
                "evidence_used": [{"vehicle_crop_path": "crop.jpg"}],
                "selected_crop_paths": ["crop.jpg"],
                "status": "completed",
            }
        ],
    )
    repository = RunRepository(tmp_path)
    runs = repository.list_runs()
    tracks = repository.list_tracks()
    track = repository.get_track(camera_id="CAM_001", track_id="TRACK_1")
    assert runs[0]["run_id"] == "20260808_120000"
    assert tracks[0]["track_id"] == "TRACK_1"
    assert tracks[0]["colour"] == "WHITE"
    assert tracks[0]["plate_text"] == "DL8CAF5030"
    assert track is not None
    assert track["local_track_id"] == "CAM_001:TRACK_1"
    assert track["plate_text_confidence"] == 0.86


def test_run_repository_resolve_media_path_blocks_traversal(tmp_path: Path) -> None:
    run_dir = tmp_path / "20260808_120001"
    target = run_dir / "evidence" / "CAM_001" / "TRACK_1" / "crop.jpg"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"x")
    repository = RunRepository(tmp_path)
    safe = repository.resolve_media_path(run_id="20260808_120001", category="evidence", relative_parts=["CAM_001", "TRACK_1", "crop.jpg"])
    blocked = repository.resolve_media_path(run_id="20260808_120001", category="evidence", relative_parts=["..", "crop.jpg"])
    assert safe == target.resolve()
    assert blocked is None


def test_run_repository_filter_options_return_supported_vocabularies(tmp_path: Path) -> None:
    run_dir = tmp_path / "20260808_120002"
    _write_json(
        run_dir / "tracks.json",
        [
            {
                "local_track_id": "CAM_001:TRACK_1",
                "camera_id": "CAM_001",
                "status": "COMPLETED",
                "first_timestamp_seconds": 1.0,
                "last_timestamp_seconds": 2.0,
                "first_frame": 10,
                "last_frame": 20,
                "observation_count": 5,
                "completion_reason": "END_OF_STREAM",
                "final_class": "car",
            }
        ],
    )
    _write_json(
        run_dir / "vehicle_enrichment.json",
        [
            {
                "local_track_id": "CAM_001:TRACK_1",
                "camera_id": "CAM_001",
                "vehicle_class": "CAR",
                "vehicle_colour": {"label": "WHITE", "status": "completed"},
                "evidence_used": [],
                "selected_crop_paths": [],
                "status": "completed",
            }
        ],
    )
    repository = RunRepository(tmp_path)

    options = repository.get_filter_options(run_id="20260808_120002")

    assert options["vehicle_classes"] == list(SUPPORTED_VEHICLE_CLASSES)
    assert options["colours"] == list(SUPPORTED_VEHICLE_COLOUR_LABELS)
    assert options["cameras"] == ["CAM_001"]
