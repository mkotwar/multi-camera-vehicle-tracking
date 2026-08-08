from __future__ import annotations

import numpy as np

from src.runtime_state import RuntimeStateManager


def test_runtime_state_tracks_camera_frame_and_system_metrics() -> None:
    manager = RuntimeStateManager()
    manager.initialize_run(
        run_id="20260808_120000",
        run_directory="outputs/runs/20260808_120000",
        cameras=[{"camera_id": "CAM_001", "source_type": "video", "source": "sample.mp4"}],
    )
    frame = np.zeros((32, 32, 3), dtype=np.uint8)
    manager.update_camera_runtime(
        camera_id="CAM_001",
        frame_number=12,
        timestamp_seconds=1.2,
        input_fps=10.0,
        detections=[{"track_id": "TRACK_1", "vehicle_class": "car", "bbox": [1, 2, 3, 4], "confidence": 0.9, "colour": None, "colour_status": "pending"}],
        active_track_ids=["TRACK_1"],
        active_vehicle_count=1,
        frame_bgr=frame,
    )
    camera = manager.get_camera("CAM_001")
    assert camera is not None
    assert camera["frame_number"] == 12
    assert camera["active_vehicle_count"] == 1
    assert manager.get_frame_bytes("CAM_001") is not None
    system = manager.get_system_status()
    assert system["camera_count"] == 1
    assert system["online_camera_count"] == 1


def test_runtime_state_track_updates_and_colour_updates() -> None:
    manager = RuntimeStateManager()
    manager.update_track_runtime(
        camera_id="CAM_001",
        local_track_id="CAM_001:TRACK_1",
        short_track_id="TRACK_1",
        vehicle_class="car",
        bbox=[1, 2, 10, 20],
        confidence=0.8,
        timestamp_seconds=1.5,
        frame_number=15,
        colour=None,
        colour_status="pending",
        status="active",
    )
    manager.update_track_colour(
        camera_id="CAM_001",
        local_track_id="CAM_001:TRACK_1",
        short_track_id="TRACK_1",
        colour="WHITE",
        colour_status="completed",
    )
    track = manager.get_track("CAM_001:TRACK_1")
    assert track is not None
    assert track["track_id"] == "TRACK_1"
    assert track["colour"] == "WHITE"
    assert track["colour_status"] == "completed"
