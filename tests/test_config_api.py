from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient

from src.api_app import create_app


def _write_config(config_dir: Path) -> None:
    model_path = config_dir / "model.pt"
    model_path.write_bytes(b"model")
    config = {
        "project": {"name": "test"},
        "input": {"cameras": [{"camera_id": "CAM_001", "source_type": "video", "source": str(config_dir / "missing.mp4"), "enabled": True}], "max_frames_per_camera": None},
        "ingestion": {"worker_count": 1, "frame_queue_size": 2, "per_camera_buffer_size": 1, "scheduler_policy": "round_robin"},
        "detection": {"model_path": str(model_path), "confidence_threshold": 0.2, "iou_threshold": 0.45, "image_size": 640},
        "tracking": {"track_activation_threshold": 0.25, "lost_track_buffer": 150, "minimum_matching_threshold": 0.7, "minimum_consecutive_frames": 3},
        "tracking_roi": {"enabled": True, "mode": "rectangle", "rectangle": {"x_min_fraction": 0.0, "y_min_fraction": 0.4, "x_max_fraction": 1.0, "y_max_fraction": 0.75}, "anchor": "bottom_center"},
        "visualization": {},
        "output": {"root_directory": "outputs/runs", "save_run_config": True},
    }
    (config_dir / "validation_rectangle_roi.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def _create_test_video(path: Path, *, width: int = 64, height: int = 48) -> Path:
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 5.0, (width, height))
    writer.write(np.full((height, width, 3), 120, dtype=np.uint8))
    writer.release()
    return path


def test_config_api_load_validate_save_and_reject_invalid(tmp_path: Path) -> None:
    config_dir = tmp_path / "config"
    outputs_dir = tmp_path / "runs"
    config_dir.mkdir()
    _write_config(config_dir)
    client = TestClient(create_app(outputs_root=outputs_dir, config_dir=config_dir))

    list_response = client.get("/api/configs")
    assert list_response.status_code == 200
    assert list_response.json()["configs"][0]["config_name"] == "validation_rectangle_roi.yaml"

    detail_response = client.get("/api/configs/validation_rectangle_roi.yaml")
    assert detail_response.status_code == 200
    config = detail_response.json()["config"]
    config["tracking"]["lost_track_buffer"] = 155

    validate_response = client.post("/api/configs/validation_rectangle_roi.yaml/validate", json={"config": config})
    assert validate_response.status_code == 200
    assert validate_response.json()["valid"] is True

    save_response = client.put("/api/configs/validation_rectangle_roi.yaml", json={"config": config})
    assert save_response.status_code == 200
    assert yaml.safe_load((config_dir / "validation_rectangle_roi.yaml").read_text(encoding="utf-8"))["tracking"]["lost_track_buffer"] == 155

    config["tracking_roi"]["rectangle"]["x_min_fraction"] = 1.0
    invalid_response = client.post("/api/configs/validation_rectangle_roi.yaml/validate", json={"config": config})
    assert invalid_response.status_code == 200
    assert invalid_response.json()["valid"] is False
    assert invalid_response.json()["errors"][0]["path"] == "tracking_roi.rectangle.x_min_fraction"

    traversal_response = client.get("/api/configs/..production.yaml")
    assert traversal_response.status_code == 400


def test_config_api_round_trips_three_cameras_with_independent_roi(tmp_path: Path) -> None:
    config_dir = tmp_path / "config"
    outputs_dir = tmp_path / "runs"
    config_dir.mkdir()
    _write_config(config_dir)
    client = TestClient(create_app(outputs_root=outputs_dir, config_dir=config_dir))

    detail_response = client.get("/api/configs/validation_rectangle_roi.yaml")
    config = detail_response.json()["config"]
    config["input"]["cameras"] = [
        {
            "camera_id": "CAM_001",
            "source_type": "video",
            "source": str(config_dir / "cam1.mp4"),
            "enabled": True,
            "tracking_roi": {
                "enabled": True,
                "mode": "rectangle",
                "rectangle": {"x_min_fraction": 0.1, "y_min_fraction": 0.2, "x_max_fraction": 0.7, "y_max_fraction": 0.8},
                "anchor": "bottom_center",
            },
        },
        {
            "camera_id": "CAM_002",
            "source_type": "video",
            "source": str(config_dir / "cam2.mp4"),
            "enabled": True,
            "tracking_roi": {
                "enabled": False,
                "mode": "rectangle",
                "rectangle": {"x_min_fraction": 0.0, "y_min_fraction": 0.0, "x_max_fraction": 1.0, "y_max_fraction": 1.0},
                "anchor": "bottom_center",
            },
        },
        {"camera_id": "CAM_003", "source_type": "video", "source": str(config_dir / "cam3.mp4"), "enabled": False},
    ]

    save_response = client.put("/api/configs/validation_rectangle_roi.yaml", json={"config": config})
    get_response = client.get("/api/configs/validation_rectangle_roi.yaml")

    assert save_response.status_code == 200
    reloaded = get_response.json()["config"]
    assert [camera["camera_id"] for camera in reloaded["input"]["cameras"]] == ["CAM_001", "CAM_002", "CAM_003"]
    assert reloaded["input"]["cameras"][0]["tracking_roi"]["rectangle"]["x_min_fraction"] == 0.1
    assert reloaded["input"]["cameras"][1]["tracking_roi"]["enabled"] is False


def test_config_api_roi_preview_reads_existing_saved_camera(tmp_path: Path) -> None:
    config_dir = tmp_path / "config"
    outputs_dir = tmp_path / "runs"
    config_dir.mkdir()
    _write_config(config_dir)
    video_path = _create_test_video(config_dir / "saved.mp4", width=64, height=48)
    config_path = config_dir / "validation_rectangle_roi.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["input"]["cameras"][0]["source"] = str(video_path)
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    client = TestClient(create_app(outputs_root=outputs_dir, config_dir=config_dir))

    response = client.get("/api/configs/validation_rectangle_roi.yaml/roi-preview", params={"camera_id": "CAM_001"})

    assert response.status_code == 200
    assert response.content.startswith(b"\xff\xd8")
    assert response.headers["x-frame-width"] == "64"


def test_config_api_roi_preview_accepts_unsaved_dynamic_camera(tmp_path: Path) -> None:
    config_dir = tmp_path / "config"
    outputs_dir = tmp_path / "runs"
    config_dir.mkdir()
    _write_config(config_dir)
    video_path = _create_test_video(config_dir / "cam3.mp4", width=80, height=50)
    client = TestClient(create_app(outputs_root=outputs_dir, config_dir=config_dir))

    response = client.post(
        "/api/configs/validation_rectangle_roi.yaml/roi-preview",
        json={"camera": {"camera_id": "CAM_003", "source_type": "video", "source": str(video_path), "enabled": True}},
    )

    assert response.status_code == 200
    assert response.headers["x-frame-width"] == "80"


def test_config_api_roi_preview_uses_edited_source_before_save(tmp_path: Path) -> None:
    config_dir = tmp_path / "config"
    outputs_dir = tmp_path / "runs"
    config_dir.mkdir()
    _write_config(config_dir)
    saved_video = _create_test_video(config_dir / "saved.mp4", width=64, height=48)
    edited_video = _create_test_video(config_dir / "edited.mp4", width=96, height=54)
    config_path = config_dir / "validation_rectangle_roi.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["input"]["cameras"][0]["source"] = str(saved_video)
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    client = TestClient(create_app(outputs_root=outputs_dir, config_dir=config_dir))

    response = client.post(
        "/api/configs/validation_rectangle_roi.yaml/roi-preview",
        json={"camera": {"camera_id": "CAM_001", "source_type": "video", "source": str(edited_video), "enabled": True}},
    )

    assert response.status_code == 200
    assert response.headers["x-frame-width"] == "96"


def test_config_api_roi_preview_invalid_source_returns_clear_error(tmp_path: Path) -> None:
    config_dir = tmp_path / "config"
    outputs_dir = tmp_path / "runs"
    config_dir.mkdir()
    _write_config(config_dir)
    client = TestClient(create_app(outputs_root=outputs_dir, config_dir=config_dir))

    response = client.post(
        "/api/configs/validation_rectangle_roi.yaml/roi-preview",
        json={"camera": {"camera_id": "CAM_003", "source_type": "video", "source": str(config_dir / "missing.mp4"), "enabled": True}},
    )

    assert response.status_code == 404
    assert "Video source does not exist for CAM_003" in response.json()["detail"]["detail"]
