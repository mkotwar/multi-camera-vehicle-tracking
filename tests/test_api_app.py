from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from src.api_app import create_app
from src.vehicle_enrichment.taxonomy import SUPPORTED_VEHICLE_CLASSES, SUPPORTED_VEHICLE_COLOUR_LABELS


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _build_run(tmp_path: Path) -> str:
    run_id = "20260808_182124"
    run_dir = tmp_path / run_id
    _write_json(
        run_dir / "summary.json",
        {
            "run_id": run_id,
            "status": "COMPLETED",
            "configured_camera_count": 2,
            "processed_frames": 40,
            "overall_pipeline_runtime_ms": 42000,
            "vehicle_enrichment_enabled": True,
            "colour_queue_size": 100,
        },
    )
    _write_json(run_dir / "run_metadata.json", {"started_at": "2026-08-08T10:00:00+00:00", "completed_at": "2026-08-08T10:10:00+00:00", "camera_count": 2})
    _write_json(
        run_dir / "detection_tracking_metrics.json",
        {
            "duration_seconds": 10.0,
            "detection_frames_total": 40,
            "image_size": 1024,
            "detection_batch_size_configured": 1,
            "frame_order_violations": 0,
        },
    )
    _write_json(run_dir / "vehicle_enrichment_metrics.json", {"average_colour_calls_per_track": 1.1, "colour_queue_peak_depth": 3})
    crop = run_dir / "05_florence_selected_crops" / "CAM_001" / "TRACK_1" / "frame_000005_MIDDLE.jpg"
    crop.parent.mkdir(parents=True, exist_ok=True)
    crop.write_bytes(b"crop")
    frame = run_dir / "evidence" / "CAM_001" / "CAM_001_TRACK_1" / "annotated_frames" / "frame_000005.jpg"
    frame.parent.mkdir(parents=True, exist_ok=True)
    frame.write_bytes(b"frame")
    tracked = run_dir / "tracked_frames" / "CAM_001" / "frame_000013.jpg"
    tracked.parent.mkdir(parents=True, exist_ok=True)
    tracked.write_bytes(b"tracked")
    _write_json(
        run_dir / "tracks.json",
        [
            {
                "local_track_id": "CAM_001:TRACK_1",
                "camera_id": "CAM_001",
                "status": "COMPLETED",
                "first_timestamp_seconds": 1.0,
                "last_timestamp_seconds": 2.5,
                "first_frame": 10,
                "last_frame": 13,
                "observation_count": 5,
                "completion_reason": "END_OF_STREAM",
                "final_class": "car",
            },
            {
                "local_track_id": "CAM_002:TRACK_2",
                "camera_id": "CAM_002",
                "status": "COMPLETED",
                "first_timestamp_seconds": 5.0,
                "last_timestamp_seconds": 9.0,
                "first_frame": 20,
                "last_frame": 30,
                "observation_count": 8,
                "completion_reason": "END_OF_STREAM",
                "final_class": "truck",
            },
        ],
    )
    _write_json(
        run_dir / "vehicle_enrichment.json",
        [
            {
                "local_track_id": "CAM_001:TRACK_1",
                "camera_id": "CAM_001",
                "vehicle_class": "CAR",
                "vehicle_colour": {
                    "label": "WHITE",
                    "status": "completed",
                    "predictions": [
                        {
                            "label": "WHITE",
                            "source_frame_number": 5,
                            "evidence_role": "MIDDLE",
                            "quality_weight": 0.9,
                            "status": "completed",
                            "reason": "valid",
                            "source_crop_path": str(crop),
                        }
                    ],
                },
                "evidence_used": [
                    {
                        "frame_number": 5,
                        "timestamp_seconds": 1.5,
                        "vehicle_crop_path": str(crop),
                        "annotated_frame_path": str(frame),
                        "evidence_role": "MIDDLE",
                        "colour_crop_result": "WHITE",
                    }
                ],
                "selected_crop_paths": [str(crop)],
                "status": "completed",
            },
            {
                "local_track_id": "CAM_002:TRACK_2",
                "camera_id": "CAM_002",
                "vehicle_class": "TRUCK",
                "vehicle_colour": {"label": "BLUE", "status": "completed", "predictions": []},
                "evidence_used": [],
                "selected_crop_paths": [],
                "status": "completed",
            },
        ],
    )
    return run_id


def test_api_app_filter_options_and_track_filters(tmp_path: Path) -> None:
    run_id = _build_run(tmp_path)
    client = TestClient(create_app(outputs_root=tmp_path))

    filter_response = client.get("/api/filter-options", params={"run_id": run_id})
    assert filter_response.status_code == 200
    filter_payload = filter_response.json()
    assert "CAM_001" in filter_payload["cameras"]
    assert filter_payload["vehicle_classes"] == list(SUPPORTED_VEHICLE_CLASSES)
    assert filter_payload["colours"] == list(SUPPORTED_VEHICLE_COLOUR_LABELS)
    assert "BUS" in filter_payload["vehicle_classes"]
    assert "TRUCK" in filter_payload["vehicle_classes"]

    tracks_response = client.get("/api/tracks", params={"run_id": run_id, "camera_id": "CAM_001", "vehicle_class": "CAR", "colour": "WHITE"})
    assert tracks_response.status_code == 200
    tracks = tracks_response.json()
    assert len(tracks) == 1
    assert tracks[0]["duration_seconds"] == 1.5
    assert tracks[0]["best_crop_url"].endswith("frame_000005_MIDDLE.jpg")

    time_response = client.get("/api/tracks", params={"run_id": run_id, "from_time": 2.0, "to_time": 10.0})
    assert time_response.status_code == 200
    time_tracks = time_response.json()
    assert len(time_tracks) == 2


def test_api_app_evidence_and_media_endpoints(tmp_path: Path) -> None:
    run_id = _build_run(tmp_path)
    client = TestClient(create_app(outputs_root=tmp_path))

    detail_response = client.get(f"/api/tracks/CAM_001/TRACK_1", params={"run_id": run_id})
    assert detail_response.status_code == 200
    detail_payload = detail_response.json()
    assert detail_payload["first_seen_seconds"] == 1.0
    assert detail_payload["last_seen_seconds"] == 2.5

    evidence_response = client.get(f"/api/tracks/CAM_001/TRACK_1/evidence", params={"run_id": run_id})
    assert evidence_response.status_code == 200
    evidence_payload = evidence_response.json()
    assert evidence_payload[0]["crop_url"].endswith("frame_000005_MIDDLE.jpg")
    assert evidence_payload[0]["full_frame_url"].endswith("frame_000005.jpg")

    media_response = client.get(evidence_payload[0]["full_frame_url"])
    assert media_response.status_code == 200
    assert media_response.content == b"frame"

    blocked_media = client.get(f"/api/media/evidence/{run_id}/../secret.txt")
    assert blocked_media.status_code == 404


def test_api_app_saved_run_camera_and_system_fallback(tmp_path: Path) -> None:
    run_id = _build_run(tmp_path)
    client = TestClient(create_app(outputs_root=tmp_path))

    cameras_response = client.get("/api/cameras", params={"run_id": "latest"})
    assert cameras_response.status_code == 200
    cameras = cameras_response.json()
    assert cameras[0]["camera_id"] == "CAM_001"
    assert cameras[0]["frame_url"].endswith("frame_000013.jpg")

    frame_response = client.get("/api/cameras/CAM_001/frame", params={"run_id": run_id})
    assert frame_response.status_code == 200
    assert frame_response.content == b"tracked"

    system_response = client.get("/api/system/status")
    assert system_response.status_code == 200
    system_payload = system_response.json()
    assert system_payload["pipeline_status"] == "completed"
    assert system_payload["yolo_image_size"] == 1024
