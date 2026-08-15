from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.config_service import ConfigService, ConfigServiceError


def _write_config(config_dir: Path, name: str = "validation_rectangle_roi.yaml") -> Path:
    model_path = config_dir / "model.pt"
    model_path.write_bytes(b"model")
    source_path = config_dir / "source.mp4"
    config = {
        "project": {"name": "test", "environment": "validation", "log_level": "INFO"},
        "input": {
            "cameras": [{"camera_id": "CAM_001", "source_type": "video", "source": str(source_path), "enabled": True}],
            "max_frames_per_camera": None,
        },
        "ingestion": {"worker_count": 1, "target_read_fps": 10.0, "frame_queue_size": 2, "per_camera_buffer_size": 1, "scheduler_policy": "round_robin"},
        "detection": {
            "backend": "ocr_mukul",
            "model_path": str(model_path),
            "device": "auto",
            "dtype": "auto",
            "confidence_threshold": 0.2,
            "iou_threshold": 0.45,
            "image_size": 640,
            "allowed_class_ids": [0],
        },
        "tracking": {
            "backend": "ocr_mukul_supervision_bytetrack",
            "isolation_mode": "per_camera",
            "track_activation_threshold": 0.25,
            "lost_track_buffer": 150,
            "minimum_matching_threshold": 0.70,
            "minimum_consecutive_frames": 3,
        },
        "tracking_roi": {
            "enabled": True,
            "mode": "rectangle",
            "rectangle": {
                "x_min_fraction": 0.0,
                "y_min_fraction": 0.4,
                "x_max_fraction": 1.0,
                "y_max_fraction": 0.75,
            },
            "anchor": "bottom_center",
        },
        "evidence": {"enabled": True, "maximum_candidates_per_track": 3},
        "visualization": {"detected_frames": {"enabled": False}, "tracked_frames": {"enabled": False}},
        "output": {"root_directory": "outputs/runs", "save_run_config": True},
        "vehicle_identity": {
            "enabled": True,
            "conservative": {"enabled": True, "acceptance_threshold": 0.7, "ambiguity_margin": 0.03, "vehicle_consistency_floor": 0.58},
            "plate_assistance": {"enabled": True, "require_high_quality_for_exact_override": True, "contradiction_veto": True},
            "stationary_recovery": {"enabled": False},
        },
        "vehicle_enrichment": {"enabled": False},
    }
    path = config_dir / name
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def test_config_service_lists_loads_and_validates_config(tmp_path: Path) -> None:
    _write_config(tmp_path)
    service = ConfigService(tmp_path)

    rows = service.list_configs()["configs"]
    assert rows[0]["config_name"] == "validation_rectangle_roi.yaml"

    detail = service.load_config("validation_rectangle_roi.yaml")
    assert detail["validation"]["valid"] is True
    assert detail["config"]["tracking_roi"]["rectangle"]["y_min_fraction"] == 0.4
    assert any(item["path"] == "tracking.minimum_matching_threshold" for item in detail["inventory"])


def test_config_service_rejects_path_traversal(tmp_path: Path) -> None:
    _write_config(tmp_path)
    service = ConfigService(tmp_path)

    with pytest.raises(ConfigServiceError):
      service.load_config("../production.yaml")


def test_config_service_rejects_invalid_roi_with_field_error(tmp_path: Path) -> None:
    _write_config(tmp_path)
    service = ConfigService(tmp_path)
    config = service.load_config("validation_rectangle_roi.yaml")["config"]
    config["tracking_roi"]["rectangle"]["y_min_fraction"] = 0.9
    config["tracking_roi"]["rectangle"]["y_max_fraction"] = 0.75

    result = service.validate_config("validation_rectangle_roi.yaml", config)

    assert result["valid"] is False
    assert result["errors"][0]["path"] == "tracking_roi.rectangle.y_min_fraction"


def test_config_service_rejects_plate_ocr_with_blank_detector_model_path(tmp_path: Path) -> None:
    _write_config(tmp_path)
    service = ConfigService(tmp_path)
    config = service.load_config("validation_rectangle_roi.yaml")["config"]
    config["vehicle_enrichment"] = {
        "enabled": True,
        "enrichment": {
            "plate": {
                "enabled": True,
                "detector": {"enabled": True, "model_path": ""},
                "ocr": {"enabled": True, "backend": "ocr_mukul_adapter"},
            }
        },
    }

    result = service.validate_config("validation_rectangle_roi.yaml", config)

    assert result["valid"] is False
    assert result["errors"][0]["path"] == "vehicle_enrichment.enrichment.plate.detector.model_path"


def test_config_service_saves_atomically_and_clones(tmp_path: Path) -> None:
    path = _write_config(tmp_path)
    service = ConfigService(tmp_path)
    config = service.load_config("validation_rectangle_roi.yaml")["config"]
    config["tracking"]["lost_track_buffer"] = 160

    save_result = service.save_config("validation_rectangle_roi.yaml", config)
    clone_result = service.clone_config("validation_rectangle_roi.yaml", "working_copy.yaml")

    assert save_result["valid"] is True
    assert yaml.safe_load(path.read_text(encoding="utf-8"))["tracking"]["lost_track_buffer"] == 160
    assert clone_result["config_name"] == "working_copy.yaml"
    assert (tmp_path / "working_copy.yaml").exists()


def test_config_service_roi_preview_reads_one_video_frame(tmp_path: Path) -> None:
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    _write_config(tmp_path)
    video_path = tmp_path / "source.mp4"
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 5.0, (64, 48))
    writer.write(np.full((48, 64, 3), 120, dtype=np.uint8))
    writer.release()
    service = ConfigService(tmp_path)

    frame_bytes, headers = service.read_roi_preview_frame("validation_rectangle_roi.yaml", "CAM_001")

    assert frame_bytes.startswith(b"\xff\xd8")
    assert headers["X-Frame-Width"] == "64"
    assert headers["X-Frame-Height"] == "48"
