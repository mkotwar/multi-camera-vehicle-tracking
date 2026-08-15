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


def _build_physical_vehicle_run(tmp_path: Path) -> str:
    run_id = "20260815_170454"
    run_dir = tmp_path / run_id
    _write_json(run_dir / "summary.json", {"run_id": run_id, "status": "COMPLETED", "processed_frames": 100})
    _write_json(run_dir / "run_metadata.json", {"status": "COMPLETED", "camera_count": 1})
    representative_crop = run_dir / "evidence" / "CAM_001" / "CAM_001_TRACK_1" / "crops" / "frame_000001.jpg"
    representative_crop.parent.mkdir(parents=True, exist_ok=True)
    representative_crop.write_bytes(b"representative crop")
    fallback_crop = run_dir / "05_florence_selected_crops" / "CAM_001" / "TRACK_3" / "frame_000003_MIDDLE.jpg"
    fallback_crop.parent.mkdir(parents=True, exist_ok=True)
    fallback_crop.write_bytes(b"fallback crop")
    tracks = [
        {
            "local_track_id": "CAM_001:TRACK_1",
            "camera_id": "CAM_001",
            "status": "COMPLETED",
            "first_timestamp_seconds": 1.0,
            "last_timestamp_seconds": 2.0,
            "observation_count": 5,
            "final_class": "car",
            "vehicle_enrichment": {"vehicle_colour": {"label": "WHITE", "status": "completed"}},
        },
        {
            "local_track_id": "CAM_001:TRACK_2",
            "camera_id": "CAM_001",
            "status": "COMPLETED",
            "first_timestamp_seconds": 3.0,
            "last_timestamp_seconds": 4.0,
            "observation_count": 5,
            "final_class": "car",
            "vehicle_enrichment": {"vehicle_colour": {"label": "WHITE", "status": "completed"}},
        },
        {
            "local_track_id": "CAM_001:TRACK_3",
            "camera_id": "CAM_001",
            "status": "COMPLETED",
            "first_timestamp_seconds": 5.0,
            "last_timestamp_seconds": 6.0,
            "observation_count": 5,
            "final_class": "motorcycle",
            "vehicle_enrichment": {"vehicle_colour": {"label": "BLACK", "status": "completed"}},
        },
    ]
    _write_json(run_dir / "tracks.json", tracks)
    _write_json(
        run_dir / "vehicle_enrichment.json",
        [
            {
                "local_track_id": "CAM_001:TRACK_3",
                "camera_id": "CAM_001",
                "vehicle_class": "MOTORCYCLE",
                "vehicle_colour": {"label": "BLACK", "status": "completed"},
                "evidence_used": [{"vehicle_crop_path": str(fallback_crop), "selected_for_colour": True}],
                "selected_crop_paths": [str(fallback_crop)],
                "status": "completed",
            }
        ],
    )
    _write_json(
        run_dir / "physical_vehicles.json",
        {
            "physical_vehicles": [
                {
                    "vehicle_id": "VEHICLE_001",
                    "vehicle_key": "VEHICLE_001",
                    "vehicle_class": "CAR",
                    "vehicle_colour": "WHITE",
                    "first_seen_seconds": 1.0,
                    "last_seen_seconds": 4.0,
                    "member_track_ids": ["CAM_001:TRACK_1", "CAM_001:TRACK_2"],
                    "member_track_count": 2,
                    "primary_camera_id": "CAM_001",
                    "camera_ids": ["CAM_001"],
                    "representative_evidence": [{"local_track_id": "CAM_001:TRACK_1", "vehicle_crop_path": str(representative_crop)}],
                },
                {
                    "vehicle_id": "VEHICLE_002",
                    "vehicle_key": "VEHICLE_002",
                    "vehicle_class": "MOTORCYCLE",
                    "vehicle_colour": "BLACK",
                    "first_seen_seconds": 5.0,
                    "last_seen_seconds": 6.0,
                    "member_track_ids": ["CAM_001:TRACK_3"],
                    "member_track_count": 1,
                    "primary_camera_id": "CAM_001",
                    "camera_ids": ["CAM_001"],
                    "representative_evidence": [],
                },
            ]
        },
    )
    return run_id


def _build_vehicle_search_run(tmp_path: Path) -> str:
    run_id = "20260812_113742"
    run_dir = tmp_path / run_id
    _write_json(run_dir / "summary.json", {"run_id": run_id, "status": "COMPLETED", "processed_frames": 600})
    _write_json(run_dir / "run_metadata.json", {"status": "COMPLETED", "camera_count": 1})
    car_ids = {
        "TRACK_1": "BLACK",
        "TRACK_2": "SILVER",
        "TRACK_3": "BLACK",
        "TRACK_4": "BLACK",
        "TRACK_5": "BLACK",
        "TRACK_11": "BLACK",
        "TRACK_13": "WHITE",
        "TRACK_19": "WHITE",
        "TRACK_26": "WHITE",
        "TRACK_28": "WHITE",
        "TRACK_30": "BLACK",
        "TRACK_33": "WHITE",
        "TRACK_34": "BLACK",
        "TRACK_35": "BLACK",
        "TRACK_42": "WHITE",
        "TRACK_43": "BLACK",
        "TRACK_46": "BLACK",
    }
    motorcycle_ids = {
        "TRACK_7": "WHITE",
        "TRACK_6": "BLACK",
        "TRACK_10": "RED",
        "TRACK_9": "RED",
        "TRACK_14": "BLACK",
        "TRACK_18": "BLACK",
        "TRACK_17": "BLACK",
        "TRACK_24": "RED",
        "TRACK_25": "BLACK",
        "TRACK_23": "BLACK",
        "TRACK_27": "BLACK",
        "TRACK_31": "BLACK",
        "TRACK_32": "BLACK",
        "TRACK_38": "BLACK",
        "TRACK_39": "BLUE",
        "TRACK_40": "RED",
        "TRACK_36": "BLACK",
        "TRACK_45": "BLACK",
    }
    between_ids = {"TRACK_14", "TRACK_16", "TRACK_15", "TRACK_18", "TRACK_17", "TRACK_19", "TRACK_24", "TRACK_25", "TRACK_23"}
    tracks = []
    enrichments = []

    def add(track_id: str, vehicle_class: str, colour: str) -> None:
        in_window = track_id in between_ids
        crop = run_dir / "05_florence_selected_crops" / "CAM_001" / track_id / "frame_000006_MIDDLE.jpg"
        crop.parent.mkdir(parents=True, exist_ok=True)
        crop.write_bytes(f"{track_id} crop".encode("utf-8"))
        tracks.append(
            {
                "local_track_id": f"CAM_001:{track_id}",
                "camera_id": "CAM_001",
                "status": "COMPLETED",
                "final_class": vehicle_class,
                "first_timestamp_seconds": 6.0 if in_window else 20.0,
                "last_timestamp_seconds": 8.0 if in_window else 25.0,
                "observation_count": 10,
                "vehicle_enrichment": {
                    "vehicle_colour": {
                        "label": colour,
                        "status": "completed" if colour != "UNKNOWN" else "skipped",
                    }
                },
            }
        )
        enrichments.append(
            {
                "local_track_id": f"CAM_001:{track_id}",
                "camera_id": "CAM_001",
                "vehicle_class": vehicle_class.upper(),
                "vehicle_colour": {"label": colour, "status": "completed"},
                "evidence_used": [
                    {
                        "frame_number": 6,
                        "timestamp_seconds": 6.0 if in_window else 20.0,
                        "vehicle_crop_path": str(crop),
                        "evidence_role": "MIDDLE",
                        "selected_for_colour": True,
                    }
                ],
                "selected_crop_paths": [str(crop)],
                "status": "completed",
            }
        )

    for track_id, colour in car_ids.items():
        add(track_id, "car", colour)
    for track_id, colour in motorcycle_ids.items():
        add(track_id, "motorcycle", colour)
    add("TRACK_16", "truck", "WHITE")
    add("TRACK_15", "unknown", "UNKNOWN")
    for track_id in ["TRACK_29", "TRACK_37", "TRACK_41", "TRACK_44"]:
        add(track_id, "3wheeler", "GREEN")
    _write_json(run_dir / "tracks.json", tracks)
    _write_json(run_dir / "vehicle_enrichment.json", enrichments)
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


def test_api_app_physical_vehicle_counts_and_video_chat_evidence(tmp_path: Path) -> None:
    run_id = _build_physical_vehicle_run(tmp_path)
    client = TestClient(create_app(outputs_root=tmp_path))

    runs_response = client.get("/api/runs")
    assert runs_response.status_code == 200
    run_summary = runs_response.json()[0]
    assert run_summary["run_id"] == run_id
    assert run_summary["track_count"] == 2
    assert run_summary["physical_vehicle_count"] == 2
    assert run_summary["raw_track_count"] == 3
    assert run_summary["completed_track_count"] == 3

    tracks_response = client.get("/api/tracks", params={"run_id": run_id})
    assert tracks_response.status_code == 200
    assert len(tracks_response.json()) == 3

    vehicles_response = client.get("/api/vehicles", params={"run_id": run_id})
    assert vehicles_response.status_code == 200
    vehicles = vehicles_response.json()
    assert len(vehicles) == 2
    vehicles_by_id = {item["vehicle_id"]: item for item in vehicles}
    assert vehicles_by_id["VEHICLE_001"]["best_crop_url"].startswith(f"/api/media/evidence/{run_id}/")
    assert vehicles_by_id["VEHICLE_002"]["best_crop_url"].startswith(f"/api/media/florence_selected_crops/{run_id}/")

    chat_response = client.post("/api/video-chat", json={"message": "Show them", "run_id": run_id, "session_id": "physical-chat"})
    assert chat_response.status_code == 200
    payload = chat_response.json()
    assert payload["analytics_result"]["total"] == 2
    assert set(payload["matching_vehicle_ids"]) == {"VEHICLE_001", "VEHICLE_002"}
    assert payload["evidence_page"]["matching_total"] == 2
    assert payload["evidence_page"]["evidence_returned_count"] == 2
    assert payload["answer"] == "2 vehicles were observed. Showing 2 of 2."
    evidence_by_id = {item["vehicle_id"]: item for item in payload["evidence"]}
    assert evidence_by_id["VEHICLE_001"]["member_track_ids"] == ["CAM_001:TRACK_1", "CAM_001:TRACK_2"]
    assert evidence_by_id["VEHICLE_001"]["best_crop_url"].startswith(f"/api/media/evidence/{run_id}/")
    assert evidence_by_id["VEHICLE_002"]["best_crop_url"].startswith(f"/api/media/florence_selected_crops/{run_id}/")


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


def test_api_vehicle_search_known_run_queries(tmp_path: Path) -> None:
    run_id = _build_vehicle_search_run(tmp_path)
    client = TestClient(create_app(outputs_root=tmp_path))

    all_response = client.post("/api/vehicle-search", json={"query": "How many vehicles are there?", "run_id": run_id})
    assert all_response.status_code == 200
    assert all_response.json()["run_id"] == run_id
    assert all_response.json()["analytics_result"]["total"] == 41

    white_car_response = client.post("/api/vehicle-search", json={"query": "How many white cars are there?", "run_id": run_id})
    assert white_car_response.status_code == 200
    assert white_car_response.json()["parsed_query"]["vehicle_class"] == "CAR"
    assert white_car_response.json()["parsed_query"]["colour"] == "WHITE"
    assert white_car_response.json()["analytics_result"]["total"] == 6

    car_window_response = client.post("/api/vehicle-search", json={"query": "Show cars between 5 and 10 seconds", "run_id": run_id})
    assert car_window_response.status_code == 200
    assert car_window_response.json()["analytics_result"]["total"] == 1
    assert car_window_response.json()["analytics_result"]["vehicle_ids"] == ["CAM_001:TRACK_19"]

    black_motorcycle_response = client.post(
        "/api/vehicle-search",
        json={"query": "Show black motorcycles between 5 and 10 seconds", "run_id": run_id},
    )
    assert black_motorcycle_response.status_code == 200
    assert black_motorcycle_response.json()["analytics_result"]["total"] == 5
    assert black_motorcycle_response.json()["analytics_result"]["vehicle_ids"] == [
        "CAM_001:TRACK_14",
        "CAM_001:TRACK_18",
        "CAM_001:TRACK_17",
        "CAM_001:TRACK_25",
        "CAM_001:TRACK_23",
    ]


def test_api_vehicle_search_latest_invalid_and_unknown_run(tmp_path: Path) -> None:
    run_id = _build_vehicle_search_run(tmp_path)
    client = TestClient(create_app(outputs_root=tmp_path))

    latest_response = client.post("/api/vehicle-search", json={"query": "How many white cars are there?", "run_id": "latest"})
    assert latest_response.status_code == 200
    assert latest_response.json()["run_id"] == run_id
    assert latest_response.json()["analytics_result"]["total"] == 6

    invalid_response = client.post("/api/vehicle-search", json={"query": "show dark vehicles", "run_id": run_id})
    assert invalid_response.status_code == 400
    assert invalid_response.json()["detail"]["error"] == "query_not_understood"

    unknown_run_response = client.post("/api/vehicle-search", json={"query": "How many vehicles are there?", "run_id": "missing"})
    assert unknown_run_response.status_code == 404
    assert unknown_run_response.json()["detail"]["error"] == "run_not_found"


def test_api_vehicle_search_missing_tracks_json_is_data_error(tmp_path: Path) -> None:
    run_id = "20260812_120000"
    _write_json(tmp_path / run_id / "summary.json", {"run_id": run_id, "status": "COMPLETED"})
    client = TestClient(create_app(outputs_root=tmp_path))

    response = client.post("/api/vehicle-search", json={"query": "How many vehicles are there?", "run_id": run_id})

    assert response.status_code == 500
    assert response.json()["detail"]["error"] == "tracks_json_missing"


def test_api_video_chat_count_multi_class_colour_and_time_queries(tmp_path: Path) -> None:
    run_id = _build_vehicle_search_run(tmp_path)
    client = TestClient(create_app(outputs_root=tmp_path))

    car_response = client.post("/api/video-chat", json={"message": "How many cars were there?", "run_id": run_id, "session_id": "chat-a"})
    assert car_response.status_code == 200
    assert car_response.json()["parsed_query"]["intent"] == "COUNT"
    assert car_response.json()["analytics_result"]["total"] == 17
    assert car_response.json()["evidence"] == []

    multi_response = client.post("/api/video-chat", json={"message": "How many cars and motorcycles were there?", "run_id": run_id, "session_id": "chat-b"})
    assert multi_response.status_code == 200
    assert multi_response.json()["analytics_result"]["total"] == 35

    colour_response = client.post("/api/video-chat", json={"message": "How many black motorcycles were there?", "run_id": run_id, "session_id": "chat-c"})
    assert colour_response.status_code == 200
    assert colour_response.json()["analytics_result"]["total"] == 12

    time_response = client.post("/api/video-chat", json={"message": "Show black motorcycles between 5 and 10 seconds", "run_id": run_id, "session_id": "chat-d"})
    assert time_response.status_code == 200
    assert time_response.json()["analytics_result"]["total"] == 5
    assert len(time_response.json()["evidence"]) == 5


def test_api_video_chat_unknown_vehicle_filters_unknown_class(tmp_path: Path) -> None:
    run_id = _build_vehicle_search_run(tmp_path)
    client = TestClient(create_app(outputs_root=tmp_path))

    response = client.post("/api/video-chat", json={"message": "unknown vehicle", "run_id": run_id, "session_id": "unknown-class"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["parsed_query"]["intent"] == "LIST"
    assert payload["parsed_query"]["include_classes"] == ["UNKNOWN"]
    assert payload["analytics_result"]["total"] == 1
    assert payload["matching_vehicle_ids"] == ["CAM_001:TRACK_15"]
    assert [item["vehicle_id"] for item in payload["evidence"]] == ["CAM_001:TRACK_15"]
    assert "41" not in payload["answer"]


def test_api_video_chat_list_with_evidence_summary_compare_and_missing_evidence(tmp_path: Path) -> None:
    run_id = _build_vehicle_search_run(tmp_path)
    client = TestClient(create_app(outputs_root=tmp_path))

    list_response = client.post("/api/video-chat", json={"message": "Show me the white cars.", "run_id": run_id, "session_id": "chat-e"})
    assert list_response.status_code == 200
    payload = list_response.json()
    assert payload["analytics_result"]["total"] == 6
    assert [item["vehicle_id"] for item in payload["evidence"]] == [
        "CAM_001:TRACK_13",
        "CAM_001:TRACK_19",
        "CAM_001:TRACK_26",
        "CAM_001:TRACK_28",
        "CAM_001:TRACK_33",
        "CAM_001:TRACK_42",
    ]
    assert payload["evidence"][0]["image_url"].startswith("/api/media/florence_selected_crops/")
    assert payload["evidence"][0]["track_detail_url"] == f"/tracks/CAM_001/TRACK_13?run_id={run_id}"

    summary_response = client.post("/api/video-chat", json={"message": "Give me a traffic summary.", "run_id": run_id, "session_id": "chat-f"})
    assert summary_response.status_code == 200
    assert summary_response.json()["analytics_result"]["total_unique_vehicles"] == 41

    compare_response = client.post("/api/video-chat", json={"message": "Were there more cars than motorcycles?", "run_id": run_id, "session_id": "chat-g"})
    assert compare_response.status_code == 200
    assert compare_response.json()["analytics_result"]["left_total"] == 17
    assert compare_response.json()["analytics_result"]["right_total"] == 18
    assert compare_response.json()["answer"].startswith("No.")

    crop = tmp_path / run_id / "05_florence_selected_crops" / "CAM_001" / "TRACK_13" / "frame_000006_MIDDLE.jpg"
    crop.unlink()
    missing_response = client.post("/api/video-chat", json={"message": "Show me the white cars.", "run_id": run_id, "session_id": "chat-h"})
    assert missing_response.status_code == 200
    assert missing_response.json()["evidence"][0]["vehicle_id"] == "CAM_001:TRACK_13"
    assert missing_response.json()["evidence"][0]["image_url"] is None


def test_api_video_chat_conversational_follow_up_show_them_and_invalid_query(tmp_path: Path) -> None:
    run_id = _build_vehicle_search_run(tmp_path)
    client = TestClient(create_app(outputs_root=tmp_path))

    first = client.post("/api/video-chat", json={"message": "How many motorcycles were there?", "run_id": run_id, "session_id": "chat-context"})
    assert first.status_code == 200
    assert first.json()["analytics_result"]["total"] == 18

    follow_up = client.post("/api/video-chat", json={"message": "How many of those were black?", "run_id": run_id, "session_id": "chat-context"})
    assert follow_up.status_code == 200
    assert follow_up.json()["context_used"] is True
    assert follow_up.json()["parsed_query"]["include_classes"] == ["MOTORCYCLE"]
    assert follow_up.json()["parsed_query"]["include_colours"] == ["BLACK"]
    assert follow_up.json()["analytics_result"]["total"] == 12

    show_them = client.post("/api/video-chat", json={"message": "Show them", "run_id": run_id, "session_id": "chat-context"})
    assert show_them.status_code == 200
    assert show_them.json()["context_used"] is True
    assert show_them.json()["analytics_result"]["total"] == 12
    assert len(show_them.json()["evidence"]) == 6

    invalid = client.post("/api/video-chat", json={"message": "show dark vehicles", "run_id": run_id, "session_id": "chat-invalid"})
    assert invalid.status_code == 400
    assert invalid.json()["detail"]["error"] == "query_not_understood"


def test_api_video_chat_temporal_class_comparison_queries(tmp_path: Path) -> None:
    run_id = _build_vehicle_search_run(tmp_path)
    client = TestClient(create_app(outputs_root=tmp_path))

    cars_response = client.post("/api/video-chat", json={"message": "when are cars more than bikes?", "run_id": run_id, "session_id": "interval-a"})
    assert cars_response.status_code == 200
    cars_payload = cars_response.json()
    assert cars_payload["parsed_query"]["intent"] == "FIND_INTERVALS"
    assert cars_payload["analytics_result"]["left_class"] == "CAR"
    assert cars_payload["analytics_result"]["right_class"] == "MOTORCYCLE"
    assert cars_payload["analytics_result"]["operator"] == ">"
    assert cars_payload["analytics_result"]["intervals"]

    bikes_response = client.post("/api/video-chat", json={"message": "when are bikes more than cars?", "run_id": run_id, "session_id": "interval-b"})
    assert bikes_response.status_code == 200
    bikes_payload = bikes_response.json()
    assert bikes_payload["parsed_query"]["intent"] == "FIND_INTERVALS"
    assert bikes_payload["analytics_result"]["left_class"] == "MOTORCYCLE"
    assert bikes_payload["analytics_result"]["right_class"] == "CAR"
    assert bikes_payload["analytics_result"]["operator"] == ">"
    assert bikes_payload["analytics_result"]["intervals"]


def test_api_video_chat_group_colours_uses_natural_deterministic_answer(tmp_path: Path) -> None:
    run_id = _build_vehicle_search_run(tmp_path)
    client = TestClient(create_app(outputs_root=tmp_path))

    response = client.post("/api/video-chat", json={"message": "What colours were the motorcycles?", "run_id": run_id, "session_id": "group-colours"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["parsed_query"]["intent"] == "GROUP"
    assert payload["parsed_query"]["include_classes"] == ["MOTORCYCLE"]
    assert payload["parsed_query"]["group_by"] == "colour"
    assert payload["analytics_result"]["by_colour"]["BLACK"] == 12
    assert payload["matching_vehicle_ids"] == payload["analytics_result"]["vehicle_ids"]
    assert len(payload["matching_vehicle_ids"]) == 18
    assert payload["answer"] == "The 18 motorcycles were:\n\nBlack: 12\nRed: 4\nWhite: 1\nBlue: 1\n\nBlack was the most common motorcycle colour."


def test_api_video_chat_filtered_group_colours_show_them_paginates_previous_ids(tmp_path: Path) -> None:
    run_id = _build_vehicle_search_run(tmp_path)
    client = TestClient(create_app(outputs_root=tmp_path))
    session_id = "filtered-group-show-them"

    group = client.post("/api/video-chat", json={"message": "give me the colours of motorcycles", "run_id": run_id, "session_id": session_id})
    assert group.status_code == 200
    group_payload = group.json()
    motorcycle_ids = group_payload["matching_vehicle_ids"]
    assert group_payload["parsed_query"]["intent"] == "GROUP"
    assert group_payload["parsed_query"]["include_classes"] == ["MOTORCYCLE"]
    assert group_payload["parsed_query"]["group_by"] == "colour"
    assert group_payload["analytics_result"]["total"] == 18
    assert group_payload["analytics_result"]["by_colour"]["BLACK"] == 12
    assert group_payload["analytics_result"]["by_colour"]["RED"] == 4
    assert group_payload["analytics_result"]["by_colour"]["WHITE"] == 1
    assert group_payload["analytics_result"]["by_colour"]["BLUE"] == 1
    assert len(motorcycle_ids) == 18
    assert group_payload["matching_vehicle_ids_count"] == 18
    assert group_payload["context_saved_vehicle_ids_count"] == 18

    first_page = client.post("/api/video-chat", json={"message": "Show them", "run_id": run_id, "session_id": session_id})
    assert first_page.status_code == 200
    first_payload = first_page.json()
    first_ids = [item["vehicle_id"] for item in first_payload["evidence"]]
    assert first_payload["context_used"] is True
    assert first_payload["evidence_page"]["matching_total"] == 18
    assert first_payload["evidence_page"]["evidence_returned_count"] == 6
    assert all(item["vehicle_class"] == "MOTORCYCLE" for item in first_payload["evidence"])
    assert set(first_ids).issubset(motorcycle_ids)

    second_page = client.post("/api/video-chat", json={"message": "Show more", "run_id": run_id, "session_id": session_id})
    assert second_page.status_code == 200
    second_payload = second_page.json()
    second_ids = [item["vehicle_id"] for item in second_payload["evidence"]]
    assert len(second_ids) == 6
    assert second_payload["evidence_page"]["evidence_offset"] == 6
    assert all(item["vehicle_class"] == "MOTORCYCLE" for item in second_payload["evidence"])

    third_page = client.post("/api/video-chat", json={"message": "Show more", "run_id": run_id, "session_id": session_id})
    assert third_page.status_code == 200
    third_payload = third_page.json()
    third_ids = [item["vehicle_id"] for item in third_payload["evidence"]]
    assert len(third_ids) == 6
    assert third_payload["evidence_page"]["evidence_offset"] == 12
    assert third_payload["evidence_page"]["evidence_remaining_count"] == 0
    assert all(item["vehicle_class"] == "MOTORCYCLE" for item in third_payload["evidence"])
    combined = first_ids + second_ids + third_ids
    assert len(combined) == 18
    assert len(set(combined)) == 18
    assert combined == motorcycle_ids


def test_api_video_chat_group_colour_variants_preserve_explicit_class_filters(tmp_path: Path) -> None:
    run_id = _build_vehicle_search_run(tmp_path)
    client = TestClient(create_app(outputs_root=tmp_path))

    motorcycle = client.post("/api/video-chat", json={"message": "motorcycle colour breakdown", "run_id": run_id, "session_id": "group-motorcycle"})
    assert motorcycle.status_code == 200
    assert motorcycle.json()["parsed_query"]["include_classes"] == ["MOTORCYCLE"]
    assert motorcycle.json()["analytics_result"]["total"] == 18

    cars = client.post("/api/video-chat", json={"message": "what colours were the cars?", "run_id": run_id, "session_id": "group-cars"})
    assert cars.status_code == 200
    cars_payload = cars.json()
    assert cars_payload["parsed_query"]["include_classes"] == ["CAR"]
    assert cars_payload["parsed_query"]["group_by"] == "colour"
    assert cars_payload["analytics_result"]["total"] == 17
    assert cars_payload["analytics_result"]["by_colour"]["BLACK"] == 10
    assert cars_payload["analytics_result"]["by_colour"]["WHITE"] == 6
    assert cars_payload["analytics_result"]["by_colour"]["SILVER"] == 1

    three_wheelers = client.post("/api/video-chat", json={"message": "what colours were the three wheelers?", "run_id": run_id, "session_id": "group-3w"})
    assert three_wheelers.status_code == 200
    three_payload = three_wheelers.json()
    assert three_payload["parsed_query"]["include_classes"] == ["3WHEELER"]
    assert three_payload["parsed_query"]["group_by"] == "colour"
    assert three_payload["analytics_result"]["total"] == 4
    assert three_payload["analytics_result"]["by_colour"]["GREEN"] == 4

    global_colours = client.post("/api/video-chat", json={"message": "what colours are present?", "run_id": run_id, "session_id": "global-colours"})
    assert global_colours.status_code == 200
    assert global_colours.json()["parsed_query"]["intent"] == "UNIQUE_COLOURS"

    black_classes = client.post("/api/video-chat", json={"message": "what vehicle classes were black?", "run_id": run_id, "session_id": "black-classes"})
    assert black_classes.status_code == 200
    black_payload = black_classes.json()
    assert black_payload["parsed_query"]["intent"] == "GROUP"
    assert black_payload["parsed_query"]["group_by"] == "class"
    assert black_payload["parsed_query"]["include_colours"] == ["BLACK"]
    assert black_payload["analytics_result"]["total"] == 22


def test_api_video_chat_evidence_pagination_next_exhausted_reset_and_missing_evidence(tmp_path: Path) -> None:
    run_id = _build_vehicle_search_run(tmp_path)
    client = TestClient(create_app(outputs_root=tmp_path))
    session_id = "evidence-pages"

    first = client.post("/api/video-chat", json={"message": "How many motorcycles were there?", "run_id": run_id, "session_id": session_id})
    assert first.status_code == 200
    assert first.json()["analytics_result"]["total"] == 18
    black = client.post("/api/video-chat", json={"message": "Which of those were black?", "run_id": run_id, "session_id": session_id})
    assert black.status_code == 200
    assert black.json()["analytics_result"]["total"] == 12

    first_page = client.post("/api/video-chat", json={"message": "show evidence", "run_id": run_id, "session_id": session_id})
    assert first_page.status_code == 200
    first_payload = first_page.json()
    first_ids = [item["vehicle_id"] for item in first_payload["evidence"]]
    assert len(first_ids) == 6
    assert first_payload["evidence_page"]["matching_total"] == 12
    assert first_payload["evidence_page"]["evidence_returned_count"] == 6
    assert first_payload["evidence_page"]["evidence_offset"] == 0
    assert first_payload["evidence_page"]["evidence_remaining_count"] == 6

    second_page = client.post("/api/video-chat", json={"message": "show the other 6", "run_id": run_id, "session_id": session_id})
    assert second_page.status_code == 200
    second_payload = second_page.json()
    second_ids = [item["vehicle_id"] for item in second_payload["evidence"]]
    assert len(second_ids) == 6
    assert second_payload["evidence_page"]["evidence_offset"] == 6
    assert second_payload["evidence_page"]["evidence_remaining_count"] == 0
    assert set(first_ids).isdisjoint(second_ids)
    assert first_ids + second_ids == black.json()["matching_vehicle_ids"]

    exhausted = client.post("/api/video-chat", json={"message": "show more", "run_id": run_id, "session_id": session_id})
    assert exhausted.status_code == 200
    assert exhausted.json()["evidence"] == []
    assert "already been shown" in exhausted.json()["answer"]

    crop = tmp_path / run_id / "05_florence_selected_crops" / "CAM_001" / "TRACK_13" / "frame_000006_MIDDLE.jpg"
    crop.unlink()
    reset = client.post("/api/video-chat", json={"message": "show white cars", "run_id": run_id, "session_id": session_id})
    assert reset.status_code == 200
    reset_payload = reset.json()
    assert reset_payload["analytics_result"]["total"] == 6
    assert reset_payload["evidence_page"]["evidence_offset"] == 0
    assert reset_payload["evidence"][0]["vehicle_id"] == "CAM_001:TRACK_13"
    assert reset_payload["evidence"][0]["image_url"] is None


def test_api_video_chat_greeting_and_class_wise_queries_do_not_leak_context(tmp_path: Path) -> None:
    run_id = _build_vehicle_search_run(tmp_path)
    client = TestClient(create_app(outputs_root=tmp_path))
    session_id = "context-leak"

    hello = client.post("/api/video-chat", json={"message": "hello", "run_id": run_id, "session_id": session_id})
    assert hello.status_code == 200
    hello_payload = hello.json()
    assert hello_payload["parsed_query"]["intent"] == "GENERAL_CHAT"
    assert hello_payload["analytics_result"] == {}
    assert hello_payload["matching_vehicle_ids"] == []
    assert "41 vehicles were observed" not in hello_payload["answer"]

    class_wise = client.post("/api/video-chat", json={"message": "give the numbers class wise", "run_id": run_id, "session_id": session_id})
    assert class_wise.status_code == 200
    class_payload = class_wise.json()
    assert class_payload["parsed_query"]["intent"] == "GROUP"
    assert class_payload["parsed_query"]["group_by"] == "class"
    assert class_payload["parsed_query"]["include_colours"] == []
    assert class_payload["context_used"] is False
    assert class_payload["analytics_result"]["total"] == 41
    assert class_payload["analytics_result"]["by_class"]["MOTORCYCLE"] == 18
    assert class_payload["analytics_result"]["by_class"]["CAR"] == 17
    assert class_payload["analytics_result"]["by_class"]["3WHEELER"] == 4
    assert class_payload["analytics_result"]["by_class"]["TRUCK"] == 1
    assert class_payload["analytics_result"]["by_class"]["UNKNOWN"] == 1
    assert "Motorcycles: 18" in class_payload["answer"]
    assert "Cars: 17" in class_payload["answer"]
    assert "Three-wheelers: 4" in class_payload["answer"]
    assert "Truck: 1" in class_payload["answer"]
    assert "Unknown: 1" in class_payload["answer"]
    assert "Total: 41 vehicles." in class_payload["answer"]

    all_class_wise = client.post("/api/video-chat", json={"message": "i want all the vehicles class wise", "run_id": run_id, "session_id": session_id})
    assert all_class_wise.status_code == 200
    all_class_payload = all_class_wise.json()
    assert all_class_payload["analytics_result"]["total"] == 41
    assert all_class_payload["analytics_result"]["by_class"] == class_payload["analytics_result"]["by_class"]
    assert "22 black vehicles" not in all_class_payload["answer"].lower()


def test_api_video_chat_follow_up_uses_context_but_new_group_queries_reset_it(tmp_path: Path) -> None:
    run_id = _build_vehicle_search_run(tmp_path)
    client = TestClient(create_app(outputs_root=tmp_path))
    session_id = "context-reset"

    black = client.post("/api/video-chat", json={"message": "show black motorcycles", "run_id": run_id, "session_id": session_id})
    assert black.status_code == 200
    assert black.json()["analytics_result"]["total"] == 12

    red_follow_up = client.post("/api/video-chat", json={"message": "how many of those were red?", "run_id": run_id, "session_id": session_id})
    assert red_follow_up.status_code == 200
    red_payload = red_follow_up.json()
    assert red_payload["context_used"] is True
    assert red_payload["analytics_result"]["total"] == 0

    class_wise = client.post("/api/video-chat", json={"message": "give all vehicles class wise", "run_id": run_id, "session_id": session_id})
    assert class_wise.status_code == 200
    assert class_wise.json()["context_used"] is False
    assert class_wise.json()["analytics_result"]["total"] == 41
    assert class_wise.json()["parsed_query"]["include_colours"] == []

    colours = client.post("/api/video-chat", json={"message": "what colours are present?", "run_id": run_id, "session_id": session_id})
    assert colours.status_code == 200
    colours_payload = colours.json()
    assert colours_payload["context_used"] is False
    assert colours_payload["parsed_query"]["intent"] == "UNIQUE_COLOURS"
    assert set(colours_payload["analytics_result"]["colours_present"]) >= {"BLACK", "WHITE", "GREEN", "RED", "SILVER", "BLUE", "UNKNOWN"}


def test_api_video_chat_colour_wise_query_is_global_without_context_leak(tmp_path: Path) -> None:
    run_id = _build_vehicle_search_run(tmp_path)
    client = TestClient(create_app(outputs_root=tmp_path))
    session_id = "colour-wise"

    previous = client.post("/api/video-chat", json={"message": "show black motorcycles", "run_id": run_id, "session_id": session_id})
    assert previous.status_code == 200
    response = client.post("/api/video-chat", json={"message": "give vehicle numbers colour wise", "run_id": run_id, "session_id": session_id})
    assert response.status_code == 200
    payload = response.json()
    assert payload["context_used"] is False
    assert payload["parsed_query"]["intent"] == "GROUP"
    assert payload["parsed_query"]["group_by"] == "colour"
    assert payload["parsed_query"]["include_classes"] == []
    assert payload["analytics_result"]["total"] == 41
    assert payload["analytics_result"]["by_colour"]["BLACK"] == 22
    assert payload["analytics_result"]["by_colour"]["WHITE"] == 8
    assert payload["analytics_result"]["by_colour"]["GREEN"] == 4
    assert payload["analytics_result"]["by_colour"]["RED"] == 4
    assert payload["analytics_result"]["by_colour"]["SILVER"] == 1
    assert payload["analytics_result"]["by_colour"]["BLUE"] == 1
    assert payload["analytics_result"]["by_colour"]["UNKNOWN"] == 1


def test_api_health_tracks_and_latest_regression(tmp_path: Path) -> None:
    run_id = _build_run(tmp_path)
    client = TestClient(create_app(outputs_root=tmp_path))

    health_response = client.get("/api/health")
    assert health_response.status_code == 200
    assert health_response.json()["status"] == "ok"

    tracks_response = client.get("/api/tracks", params={"run_id": run_id})
    assert tracks_response.status_code == 200
    assert len(tracks_response.json()) == 2

    runs_response = client.get("/api/runs")
    assert runs_response.status_code == 200
    assert runs_response.json()[0]["run_id"] == run_id


def test_api_openapi_schema_builds(tmp_path: Path) -> None:
    _build_run(tmp_path)
    client = TestClient(create_app(outputs_root=tmp_path))

    response = client.get("/openapi.json")

    assert response.status_code == 200
    assert response.json()["info"]["title"] == "Multi-Camera Vehicle Tracking API"


def test_api_track_reconciliation_available_missing_and_tracks_json_untouched(tmp_path: Path) -> None:
    run_id = _build_run(tmp_path)
    run_dir = tmp_path / run_id
    tracks_before = (run_dir / "tracks.json").read_text(encoding="utf-8")
    reconciliation_dir = run_dir / "track_reconciliation_test"
    visual_dir = reconciliation_dir / "visual_evidence" / "accepted" / "CAM_001_TRACK_1__CAM_002_TRACK_2"
    visual_dir.mkdir(parents=True)
    (visual_dir / "before_after_contact_sheet.jpg").write_bytes(b"contact")
    _write_json(
        reconciliation_dir / "track_reconciliation_test.json",
        {
            "metrics": {
                "raw_bytetrack_unique_tracks": 2,
                "reconciled_vehicle_identities": 1,
                "track_fragments_merged": 1,
                "accepted_matches": 1,
                "ambiguous_matches": 0,
            },
            "config": {"enabled": True},
            "tracks": [
                {
                    "local_track_id": "CAM_001:TRACK_1",
                    "camera_id": "CAM_001",
                    "status": "COMPLETED",
                    "vehicle_id": "VEHICLE_001",
                    "reconciliation": {"matched": False, "result": "unmatched"},
                },
                {
                    "local_track_id": "CAM_002:TRACK_2",
                    "camera_id": "CAM_002",
                    "status": "COMPLETED",
                    "vehicle_id": "VEHICLE_001",
                    "reconciliation": {
                        "matched": True,
                        "previous_track_id": "CAM_001:TRACK_1",
                        "score": 0.7478,
                        "second_best_score": 0.12,
                        "time_gap_frames": 28,
                        "time_gap_seconds": 0.93,
                        "result": "accepted",
                    },
                },
            ],
            "accepted_associations": [
                {
                    "old_track": "CAM_001:TRACK_1",
                    "new_track": "CAM_002:TRACK_2",
                    "vehicle_id": "VEHICLE_001",
                    "gap_frames": 28,
                    "gap_seconds": 0.93,
                    "score": 0.7478,
                    "second_best_score": 0.12,
                    "colour": "WHITE",
                    "class": "CAR",
                    "result": "ACCEPTED",
                }
            ],
        },
    )
    (reconciliation_dir / "association_table.csv").write_text(
        "old_track,new_track,vehicle_id,gap_frames,gap_seconds,score,second_best_score,colour,class,result\n"
        "CAM_001:TRACK_1,CAM_002:TRACK_2,VEHICLE_001,28,0.93,0.7478,0.12,WHITE,CAR,ACCEPTED\n",
        encoding="utf-8",
    )
    (reconciliation_dir / "manual_validation.csv").write_text(
        "old_track,new_track,vehicle_id,score,manual_label,reviewer_notes\n"
        "CAM_001:TRACK_1,CAM_002:TRACK_2,VEHICLE_001,0.7478,UNCERTAIN,\n",
        encoding="utf-8",
    )
    client = TestClient(create_app(outputs_root=tmp_path))

    response = client.get(f"/api/runs/{run_id}/reconciliation")

    assert response.status_code == 200
    payload = response.json()
    assert payload["available"] is True
    assert payload["metrics"]["raw_bytetrack_unique_tracks"] == 2
    assert payload["metrics"]["reconciled_vehicle_identities"] == 1
    assert payload["tracks"][0]["vehicle_id"] == "VEHICLE_001"
    assert payload["tracks"][1]["vehicle_id"] == "VEHICLE_001"
    assert payload["accepted_associations"][0]["old_track"] == "CAM_001:TRACK_1"
    assert payload["manual_validation"][0]["manual_label"] == "UNCERTAIN"
    assert payload["visual_evidence"][0]["contact_sheet_url"].endswith(
        "/api/media/track_reconciliation_visual/20260808_182124/accepted/CAM_001_TRACK_1__CAM_002_TRACK_2/before_after_contact_sheet.jpg"
    )
    assert (run_dir / "tracks.json").read_text(encoding="utf-8") == tracks_before

    missing_run_id = "20260808_190000"
    _write_json(tmp_path / missing_run_id / "summary.json", {"run_id": missing_run_id, "status": "COMPLETED"})
    missing_response = client.get(f"/api/runs/{missing_run_id}/reconciliation")
    assert missing_response.status_code == 200
    assert missing_response.json()["available"] is False
    assert missing_response.json()["message"] == "Reconciliation test has not been run for this run."


def test_api_experimental_vehicle_identity_available_missing_and_tracks_json_untouched(tmp_path: Path) -> None:
    run_id = _build_run(tmp_path)
    run_dir = tmp_path / run_id
    tracks_before = (run_dir / "tracks.json").read_text(encoding="utf-8")
    identity_dir = run_dir / "vehicle_identity_test"
    visual_dir = identity_dir / "visual_evidence"
    visual_dir.mkdir(parents=True)
    (visual_dir / "VEHICLE_001.jpg").write_bytes(b"identity-contact")
    _write_json(
        identity_dir / "vehicles.json",
        {
            "vehicles": [
                {
                    "vehicle_id": "VEHICLE_001",
                    "camera_id": "CAM_001",
                    "member_tracks": ["CAM_001:TRACK_1", "CAM_002:TRACK_2"],
                    "final_class": "CAR",
                    "first_seen_seconds": 1.0,
                    "last_seen_seconds": 9.0,
                    "stationary": False,
                }
            ]
        },
    )
    _write_json(identity_dir / "vehicle_id_map.json", {"CAM_001:TRACK_1": "VEHICLE_001", "CAM_002:TRACK_2": "VEHICLE_001"})
    _write_json(
        identity_dir / "evaluation.json",
        {
            "metrics": {"precision": 1.0, "recall": 0.5, "suspicious_overmerge_count": 0},
            "analytics_simulation": {"raw_completed_tracks": 2, "reconciled_physical_vehicles": 1, "duplicates_removed": 1},
            "existing_reconciliation_baseline": {"reconciled_vehicle_identities": 1},
            "config": {"acceptance_threshold": 0.85},
            "calibration": {"selected_config": {"acceptance_threshold": 0.85}, "selected_row": {"f1": 0.67}},
        },
    )
    (identity_dir / "association_decisions.csv").write_text(
        "track_a,track_b,candidate_vehicle_id,decision,association_reason,best_member_score,vehicle_consistency_score,conflicting_member_count\n"
        "CAM_001:TRACK_1,CAM_002:TRACK_2,VEHICLE_001,MERGE,BEST_SEQUENTIAL_PAIR,0.91,0.88,0\n",
        encoding="utf-8",
    )
    client = TestClient(create_app(outputs_root=tmp_path))

    response = client.get("/api/experimental/vehicles", params={"run_id": run_id})

    assert response.status_code == 200
    payload = response.json()
    assert payload["experimental"] is True
    assert payload["available"] is True
    assert payload["metrics"]["suspicious_overmerge_count"] == 0
    assert payload["analytics_simulation"]["raw_completed_tracks"] == 2
    assert payload["vehicle_id_map"]["CAM_001:TRACK_1"] == "VEHICLE_001"
    assert payload["vehicles"][0]["contact_sheet_url"].endswith("/api/media/vehicle_identity_visual/20260808_182124/VEHICLE_001.jpg")
    assert payload["association_decisions"][0]["association_reason"] == "BEST_SEQUENTIAL_PAIR"
    assert (run_dir / "tracks.json").read_text(encoding="utf-8") == tracks_before

    summary_response = client.get("/api/experimental/vehicle-summary", params={"run_id": run_id})
    assert summary_response.status_code == 200
    assert summary_response.json()["experimental"] is True
    assert "vehicles" not in summary_response.json()

    missing_run_id = "20260808_190000"
    _write_json(tmp_path / missing_run_id / "summary.json", {"run_id": missing_run_id, "status": "COMPLETED"})
    missing_response = client.get("/api/experimental/vehicles", params={"run_id": missing_run_id})
    assert missing_response.status_code == 200
    assert missing_response.json()["available"] is False
    assert missing_response.json()["experimental"] is True


def test_api_stationary_recovery_endpoint_available_missing_and_tracks_json_untouched(tmp_path: Path) -> None:
    run_id = _build_run(tmp_path)
    run_dir = tmp_path / run_id
    tracks_before = (run_dir / "tracks.json").read_text(encoding="utf-8")
    recovery_dir = run_dir / "vehicle_identity_test" / "stationary_recovery"
    contact_dir = recovery_dir / "contact_sheets"
    contact_dir.mkdir(parents=True)
    (contact_dir / "PVEHICLE_001.jpg").write_bytes(b"stationary-contact")
    _write_json(
        recovery_dir / "persistent_vehicles.json",
        {
            "persistent_vehicles": [
                {
                    "persistent_vehicle_id": "PVEHICLE_001",
                    "source_vehicle_ids": ["VEHICLE_006", "VEHICLE_022", "VEHICLE_024"],
                    "member_tracks": ["CAM_001:TRACK_6", "CAM_001:TRACK_12", "CAM_001:TRACK_25"],
                    "camera_id": "CAM_001",
                    "final_class": "CAR",
                    "recovery_confidence": 0.81,
                }
            ]
        },
    )
    _write_json(recovery_dir / "persistent_vehicle_id_map.json", {"VEHICLE_006": "PVEHICLE_001"})
    _write_json(
        recovery_dir / "evaluation.json",
        {
            "metrics": {"yellow_car_fully_recovered": True, "confirmed_false_merges": 0, "suspicious_overmerge_count": 0},
            "analytics_simulation": {"conservative_vehicle_identities": 3, "stationary_recovered_vehicle_identities": 1},
            "config": {"recovery_threshold": 0.74},
            "calibration": {"selected_row": {"recovery_threshold": 0.74}},
        },
    )
    (recovery_dir / "recovery_decisions.csv").write_text(
        "source_vehicle_a,source_vehicle_b,decision,score,location_score,final_reason\n"
        "VEHICLE_006,VEHICLE_022,MERGE,0.80,0.90,\n",
        encoding="utf-8",
    )
    (recovery_dir / "recovery_scores.csv").write_text(
        "source_vehicle_a,source_vehicle_b,rejected,rejection_reason,raw_score,location_score\n"
        "VEHICLE_006,VEHICLE_029,True,different_parking_location,0.57,0.44\n",
        encoding="utf-8",
    )
    client = TestClient(create_app(outputs_root=tmp_path))

    response = client.get("/api/experimental/stationary-recovered-vehicles", params={"run_id": run_id})

    assert response.status_code == 200
    payload = response.json()
    assert payload["experimental"] is True
    assert payload["stage"] == "stationary_recovery"
    assert payload["available"] is True
    assert payload["persistent_vehicles"][0]["contact_sheet_url"].endswith("/api/media/stationary_recovery_contact_sheets/20260808_182124/PVEHICLE_001.jpg")
    assert payload["recovery_scores"][0]["rejection_reason"] == "different_parking_location"
    assert (run_dir / "tracks.json").read_text(encoding="utf-8") == tracks_before

    missing_run_id = "20260808_190000"
    _write_json(tmp_path / missing_run_id / "summary.json", {"run_id": missing_run_id, "status": "COMPLETED"})
    missing_response = client.get("/api/experimental/stationary-recovered-vehicles", params={"run_id": missing_run_id})
    assert missing_response.status_code == 200
    assert missing_response.json()["available"] is False


def test_api_plate_assisted_identity_endpoint_available_missing_and_raw_tracks_unchanged(tmp_path: Path) -> None:
    run_id = _build_run(tmp_path)
    run_dir = tmp_path / run_id
    tracks_before = (run_dir / "tracks.json").read_text(encoding="utf-8")
    output_dir = run_dir / "vehicle_identity_test" / "plate_assisted"
    contact_dir = output_dir / "contact_sheets"
    contact_dir.mkdir(parents=True)
    (contact_dir / "same__TRACK_1__TRACK_2.jpg").write_bytes(b"plate-contact")
    plate_crop = run_dir / "05_florence_selected_crops" / "CAM_001" / "TRACK_1" / "plate" / "frame_000005_plate.jpg"
    plate_crop.parent.mkdir(parents=True, exist_ok=True)
    plate_crop.write_bytes(b"plate")
    _write_json(
        output_dir / "vehicles.json",
        {
            "vehicles": [
                {
                    "vehicle_id": "VEHICLE_001",
                    "camera_id": "CAM_001",
                    "member_tracks": ["CAM_001:TRACK_1", "CAM_002:TRACK_2"],
                    "final_class": "CAR",
                    "first_seen_seconds": 1.0,
                    "last_seen_seconds": 9.0,
                }
            ]
        },
    )
    _write_json(output_dir / "vehicle_id_map.json", {"CAM_001:TRACK_1": "VEHICLE_001", "CAM_002:TRACK_2": "VEHICLE_001"})
    _write_json(
        output_dir / "track_plate_consensus.json",
        [
            {
                "local_track_id": "CAM_001:TRACK_1",
                "plate_detected": True,
                "ocr_attempted": True,
                "raw_plate_text": "HR38AD4296",
                "normalized_plate_text": "HR38AD4296",
                "plate_detection_confidence": 0.82,
                "plate_text_confidence": 0.81,
                "plate_crop_path": str(plate_crop),
                "vehicle_crop_path": str(run_dir / "05_florence_selected_crops" / "CAM_001" / "TRACK_1" / "frame_000005_MIDDLE.jpg"),
                "reliability_label": "HIGH",
            },
            {
                "local_track_id": "CAM_002:TRACK_2",
                "plate_detected": True,
                "ocr_attempted": True,
                "raw_plate_text": "HR38AD4296",
                "normalized_plate_text": "HR38AD4296",
                "plate_detection_confidence": 0.80,
                "plate_text_confidence": 0.79,
                "plate_crop_path": str(plate_crop),
                "vehicle_crop_path": str(run_dir / "05_florence_selected_crops" / "CAM_001" / "TRACK_1" / "frame_000005_MIDDLE.jpg"),
                "reliability_label": "HIGH",
            },
        ],
    )
    _write_json(
        output_dir / "evaluation.json",
        {
            "verification": {"plate_enabled": True, "ocr_enabled": True, "rectangle_roi_enabled": True},
            "plate_coverage": {"completed_tracks": 2, "readable_plate_count": 2, "high_quality_plate_count": 2, "exact_matching_plate_pairs": 1},
            "baseline_without_plate": {"reconciled_identities": 2},
            "plate_assisted": {"raw_completed_tracks": 2, "reconciled_identities": 1, "duplicates_removed": 1, "true_fragment_merges": 1, "false_merges": 0},
        },
    )
    (output_dir / "association_decisions.csv").write_text(
        "track_a,track_b,candidate_vehicle_id,decision,plate_reason_code,decision_reason_codes,best_member_score\n"
        "CAM_001:TRACK_1,CAM_002:TRACK_2,VEHICLE_001,MERGE,PLATE_EXACT_MATCH,PLATE_EXACT_MATCH | SPATIAL_MATCH,0.98\n",
        encoding="utf-8",
    )
    (output_dir / "plate_pair_scores.csv").write_text(
        "track_a,track_b,plate_evidence,plate_reason_code\nCAM_001:TRACK_1,CAM_002:TRACK_2,STRONG_POSITIVE,PLATE_EXACT_MATCH\n",
        encoding="utf-8",
    )
    (output_dir / "identity_scores.csv").write_text(
        "track_a,track_b,score,plate_reason_code\nCAM_001:TRACK_1,CAM_002:TRACK_2,0.98,PLATE_EXACT_MATCH\n",
        encoding="utf-8",
    )
    client = TestClient(create_app(outputs_root=tmp_path))

    response = client.get("/api/experimental/plate-assisted-vehicles", params={"run_id": run_id})

    assert response.status_code == 200
    payload = response.json()
    assert payload["experimental"] is True
    assert payload["stage"] == "plate_assisted_identity"
    assert payload["available"] is True
    assert payload["plate_coverage"]["readable_plate_count"] == 2
    assert payload["plate_assisted"]["reconciled_identities"] == 1
    assert payload["vehicles"][0]["member_track_ids"] == ["CAM_001:TRACK_1", "CAM_002:TRACK_2"]
    assert payload["vehicles"][0]["plate"]["consensus_text"] == "HR38AD4296"
    assert payload["vehicles"][0]["plate"]["quality"] == "HIGH"
    assert payload["vehicles"][0]["plate"]["member_plates"][0]["plate_crop_url"].endswith("frame_000005_plate.jpg")
    assert payload["vehicles"][0]["contact_sheet_url"].endswith("/api/media/plate_assisted_contact_sheets/20260808_182124/same__TRACK_1__TRACK_2.jpg")
    assert "PLATE_EXACT_MATCH" in payload["vehicles"][0]["association_reasons"][0]
    assert (run_dir / "tracks.json").read_text(encoding="utf-8") == tracks_before

    raw_response = client.get("/api/tracks", params={"run_id": run_id})
    assert raw_response.status_code == 200
    assert len(raw_response.json()) == 2

    missing_run_id = "20260808_190000"
    _write_json(tmp_path / missing_run_id / "summary.json", {"run_id": missing_run_id, "status": "COMPLETED"})
    missing_response = client.get("/api/experimental/plate-assisted-vehicles", params={"run_id": missing_run_id})
    assert missing_response.status_code == 200
    assert missing_response.json()["available"] is False
    assert missing_response.json()["message"] == "Plate-assisted identity experiment has not been run for this run."
