from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import yaml

import src.pipeline as pipeline_module
from src.models import BBoxQualityDiagnostic, Detection, TrackedDetection
from src.pipeline import run_pipeline


def _create_test_video(path: Path, *, fps: float = 10.0, frame_count: int = 5, width: int = 32, height: int = 24) -> Path:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    for index in range(frame_count):
        frame = np.full((height, width, 3), index * 20, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return path


class _FakeDetectorTrackerResult:
    def __init__(self, packet):
        self.detections = [
            Detection(
                bbox_xyxy=(1.0, 2.0, 10.0, 12.0),
                confidence=0.84,
                class_id=0,
                class_name="car",
            )
        ]
        self.tracked_detections = [
            TrackedDetection(
                camera_id=packet.camera_id,
                frame_number=packet.frame_number,
                timestamp_seconds=packet.timestamp_seconds,
                tracker_id=1,
                bbox_xyxy=(1.0, 2.0, 10.0, 12.0),
                confidence=0.84,
                raw_class_id=0,
                raw_class_name="car",
            )
        ]
        self.bbox_quality_diagnostics = [
            BBoxQualityDiagnostic(
                camera_id=packet.camera_id,
                frame_number=packet.frame_number,
                timestamp_seconds=packet.timestamp_seconds,
                class_name="car",
                normalized_class_name="car",
                confidence=0.84,
                bbox_xyxy=(1.0, 2.0, 10.0, 12.0),
                bbox_width=9.0,
                bbox_height=10.0,
                bbox_area=90.0,
                frame_width=int(packet.frame.shape[1]),
                frame_height=int(packet.frame.shape[0]),
                width_ratio=9.0 / float(packet.frame.shape[1]),
                height_ratio=10.0 / float(packet.frame.shape[0]),
                area_ratio=0.1,
                aspect_ratio=0.9,
                touches_edge=False,
                touches_left_edge=False,
                touches_right_edge=False,
                touches_top_edge=False,
                touches_bottom_edge=False,
                accepted_by_bbox_quality=True,
                rejection_reason=None,
            )
        ]
        self.detected_frame = packet.frame.copy()
        self.tracked_frame = packet.frame.copy()
        self.inference_time_ms = 5.0


class FakeVehicleDetectorTracker:
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.device = "cpu"
        self.metrics = {
            "model_load_count": 1,
            "tracker_instance_count": 2,
            "tracker_camera_ids": ["CAM_001", "CAM_002"],
            "inference_times_ms": [],
            "inference_errors": [],
        }

    def process_frame(self, packet):
        self.metrics["inference_times_ms"].append(5.0)
        return _FakeDetectorTrackerResult(packet)

    def reset_all(self):
        return None


def _write_config(
    path: Path,
    *,
    cameras: list[dict[str, object]],
    output_root: str,
    model_path: str,
    max_frames_per_camera: int = 3,
    worker_count: int = 7,
    frame_queue_size: int = 50,
    save_every_n_frames: int = 1,
    max_saved_frames_per_camera: int = 5,
    stop_on_camera_error: bool = False,
) -> None:
    payload = {
        "project": {"name": "test_project", "environment": "test", "log_level": "INFO"},
        "input": {"cameras": cameras, "max_frames_per_camera": max_frames_per_camera},
        "ingestion": {
            "worker_count": worker_count,
            "target_read_fps": 10.0,
            "frame_queue_size": frame_queue_size,
            "queue_put_timeout_seconds": 0.1,
            "queue_get_timeout_seconds": 0.1,
            "stop_on_camera_error": stop_on_camera_error,
            "round_robin": True,
            "raw_frames": {
                "enabled": True,
                "save_every_n_frames": save_every_n_frames,
                "max_saved_frames_per_camera": max_saved_frames_per_camera,
                "image_format": "jpg",
                "jpeg_quality": 90,
            },
        },
        "detection": {
            "model_path": model_path,
            "device": "cpu",
            "confidence_threshold": 0.38,
            "iou_threshold": 0.45,
            "image_size": 640,
            "allowed_classes": ["car", "truck", "bus", "motorcycle", "3wheeler"],
            "bbox_quality": {
                "enabled": True,
                "minimum_width_pixels": 60,
                "minimum_height_pixels": 60,
                "minimum_area_ratio": 0.005,
                "maximum_area_ratio": 0.90,
                "minimum_aspect_ratio": 0.30,
                "maximum_aspect_ratio": 4.50,
                "reject_edge_truncated": True,
                "edge_margin_pixels": 8,
            },
        },
        "tracking": {
            "backend": "supervision_bytetrack",
            "track_activation_threshold": 0.15,
            "lost_track_buffer": 30,
            "minimum_matching_threshold": 0.80,
            "minimum_consecutive_frames": 1,
        },
        "visualization": {
            "show_rejected_boxes": False,
            "detected_frames": {"enabled": True, "save_every_n_frames": 1, "max_saved_frames_per_camera": 5},
            "tracked_frames": {"enabled": True, "save_every_n_frames": 1, "max_saved_frames_per_camera": 5},
        },
        "output": {"root_directory": output_root, "save_run_config": True},
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def test_pipeline_succeeds_with_one_configured_camera(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(pipeline_module, "VehicleDetectorTracker", FakeVehicleDetectorTracker)
    video_path = _create_test_video(tmp_path / "sample.mp4", frame_count=5)
    model_path = tmp_path / "model.pt"
    model_path.write_bytes(b"x")
    output_root = tmp_path / "runs"
    config_path = tmp_path / "config.yaml"
    _write_config(
        config_path,
        cameras=[{"camera_id": "CAM_001", "source_type": "video", "source": str(video_path), "enabled": True}],
        output_root=str(output_root),
        model_path=str(model_path),
        max_frames_per_camera=3,
    )
    exit_code, run_id, run_directory = run_pipeline(str(config_path))
    run_dir = Path(run_directory)
    metrics = json.loads((run_dir / "ingestion_metrics.json").read_text(encoding="utf-8"))
    dt_metrics = json.loads((run_dir / "detection_tracking_metrics.json").read_text(encoding="utf-8"))
    bbox_metrics = json.loads((run_dir / "bbox_quality_metrics.json").read_text(encoding="utf-8"))
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert exit_code == 0
    assert run_id == run_dir.name
    assert summary["processed_frames"] == 3
    assert metrics["worker_count"] == 7
    assert metrics["enabled_camera_count"] == 1
    assert dt_metrics["detections_by_camera"]["CAM_001"] == 3
    assert bbox_metrics["accepted_detections"] == 3
    assert bbox_metrics["rejected_detections"] == 0


def test_pipeline_succeeds_with_multiple_temporary_videos(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(pipeline_module, "VehicleDetectorTracker", FakeVehicleDetectorTracker)
    video_path_1 = _create_test_video(tmp_path / "cam1.mp4", frame_count=4)
    video_path_2 = _create_test_video(tmp_path / "cam2.mp4", frame_count=6)
    model_path = tmp_path / "model.pt"
    model_path.write_bytes(b"x")
    output_root = tmp_path / "runs"
    config_path = tmp_path / "config.yaml"
    _write_config(
        config_path,
        cameras=[
            {"camera_id": "CAM_001", "source_type": "video", "source": str(video_path_1), "enabled": True},
            {"camera_id": "CAM_002", "source_type": "video", "source": str(video_path_2), "enabled": True},
        ],
        output_root=str(output_root),
        model_path=str(model_path),
        max_frames_per_camera=10,
        save_every_n_frames=2,
        max_saved_frames_per_camera=2,
    )
    exit_code, _run_id, run_directory = run_pipeline(str(config_path))
    run_dir = Path(run_directory)
    metrics = json.loads((run_dir / "ingestion_metrics.json").read_text(encoding="utf-8"))
    dt_metrics = json.loads((run_dir / "detection_tracking_metrics.json").read_text(encoding="utf-8"))
    assert exit_code == 0
    assert metrics["frames_by_camera"]["CAM_001"] == 4
    assert metrics["frames_by_camera"]["CAM_002"] == 6
    assert dt_metrics["tracked_observations_by_camera"]["CAM_001"] == 4
    assert dt_metrics["tracked_observations_by_camera"]["CAM_002"] == 6
    assert (run_dir / "raw_frames" / "CAM_001").exists()
    assert (run_dir / "detected_frames" / "CAM_001").exists()
    assert (run_dir / "tracked_frames" / "CAM_001").exists()


def test_pipeline_fails_cleanly_on_invalid_configuration(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "project": {"name": "demo"},
                "input": {"cameras": [], "max_frames_per_camera": 0},
                "ingestion": {"worker_count": 0, "frame_queue_size": 0},
                "detection": {"model_path": ""},
                "tracking": {},
                "visualization": {},
                "output": {"root_directory": str(tmp_path / "runs")},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    exit_code, _run_id, run_directory = run_pipeline(str(config_path))
    run_dir = Path(run_directory)
    metadata = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
    assert exit_code != 0
    assert metadata["status"] == "FAILED"


def test_pipeline_continues_when_one_camera_fails_and_stop_on_camera_error_false(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(pipeline_module, "VehicleDetectorTracker", FakeVehicleDetectorTracker)
    valid_video = _create_test_video(tmp_path / "good.mp4", frame_count=3)
    model_path = tmp_path / "model.pt"
    model_path.write_bytes(b"x")
    output_root = tmp_path / "runs"
    config_path = tmp_path / "config.yaml"
    _write_config(
        config_path,
        cameras=[
            {"camera_id": "CAM_001", "source_type": "video", "source": str(valid_video), "enabled": True},
            {"camera_id": "CAM_002", "source_type": "video", "source": str(tmp_path / "missing.mp4"), "enabled": True},
        ],
        output_root=str(output_root),
        model_path=str(model_path),
        max_frames_per_camera=10,
        stop_on_camera_error=False,
    )
    exit_code, _run_id, run_directory = run_pipeline(str(config_path))
    run_dir = Path(run_directory)
    metrics = json.loads((run_dir / "ingestion_metrics.json").read_text(encoding="utf-8"))
    metadata = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
    assert exit_code == 0
    assert metrics["frames_by_camera"]["CAM_001"] == 3
    assert "CAM_002" in metrics["camera_errors"]
    assert metadata["status"] == "COMPLETED"
    assert metadata["error_count"] == 1


def test_outputs_are_written_only_under_configured_output_root(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(pipeline_module, "VehicleDetectorTracker", FakeVehicleDetectorTracker)
    video_path = _create_test_video(tmp_path / "sample.mp4", frame_count=2)
    model_path = tmp_path / "model.pt"
    model_path.write_bytes(b"x")
    output_root = tmp_path / "custom_runs"
    config_path = tmp_path / "config.yaml"
    _write_config(
        config_path,
        cameras=[{"camera_id": "CAM_001", "source_type": "video", "source": str(video_path), "enabled": True}],
        output_root=str(output_root),
        model_path=str(model_path),
        max_frames_per_camera=2,
    )
    exit_code, _run_id, run_directory = run_pipeline(str(config_path))
    assert exit_code == 0
    assert Path(run_directory).is_relative_to(output_root.resolve())
