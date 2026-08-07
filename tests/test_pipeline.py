from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import yaml

import pytest

import src.pipeline as pipeline_module
from src.models import BBoxQualityDiagnostic, Detection, TrackedDetection
from src.pipeline import _build_vehicle_colour_result_rows, _build_vehicle_colour_track_summary_rows, _load_raw_config, _validate_config, run_pipeline
from src.vehicle_enrichment.schemas import TrackEnrichmentResult, VehicleBodyTypeResult, VehicleColourResult, AttributePrediction


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
                tracker_namespace="camera",
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
        self.configured_device = "cpu"
        self.device = "cpu"
        self.runtime_device_info = type(
            "RuntimeDeviceInfoStub",
            (),
            {
                "configured_device": "cpu",
                "configured_dtype": "auto",
                "resolved_device": "cpu",
                "resolved_dtype": "float32",
                "cuda_available": False,
                "cuda_device_count": 0,
                "cuda_device_name": None,
                "torch_version": "test-torch",
                "torch_cuda_version": None,
                "reason": "CUDA unavailable because installed PyTorch build is CPU-only.",
            },
        )()
        self.metrics = {
            "model_load_count": 1,
            "tracker_instance_count": 2,
            "tracker_camera_ids": ["CAM_001", "CAM_002"],
            "tracker_instances_created_total": 2,
            "tracker_keys": ["CAM_001", "CAM_002"],
            "trackers_created_by_camera": {"CAM_001": 1, "CAM_002": 1},
            "trackers_created_by_camera_namespace": {"CAM_001:camera": 1, "CAM_002:camera": 1},
            "inference_times_ms": [],
            "inference_errors": [],
        }

    def process_frame(self, packet):
        self.metrics["inference_times_ms"].append(5.0)
        return _FakeDetectorTrackerResult(packet)

    def reset_camera(self, camera_id):
        return None

    def reset_all(self):
        return None


def _write_config(
    path: Path,
    *,
    cameras: list[dict[str, object]],
    output_root: str,
    model_path: str,
    max_frames_per_camera: int | None = 3,
    worker_count: int = 7,
    frame_queue_size: int = 50,
    save_every_n_frames: int = 1,
    max_saved_frames_per_camera: int = 5,
    stop_on_camera_error: bool = False,
    debug_outputs: dict | None = None,
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
            "agnostic_nms": False,
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
            "isolation_mode": "per_camera",
            "supported_isolation_modes": ["per_camera", "per_camera_class"],
            "track_activation_threshold": 0.15,
            "lost_track_buffer": 30,
            "minimum_matching_threshold": 0.80,
            "minimum_consecutive_frames": 1,
        },
        "lifecycle": {
            "minimum_observations": 3,
            "maximum_lost_frames": 30,
            "keep_discarded_tracks": True,
        },
        "track_class": {
            "minimum_observations": 3,
            "minimum_winner_ratio": 0.60,
            "strategy": "confidence_weighted_majority",
            "unknown_class_name": "UNKNOWN",
        },
        "evidence": {
            "enabled": True,
            "collect_first": True,
            "collect_middle": True,
            "collect_last": True,
            "collect_highest_confidence": True,
            "collect_largest": True,
            "collect_sharpest": True,
            "collect_best_overall": True,
            "maximum_candidates_per_track": 7,
            "minimum_crop_width_pixels": 5,
            "minimum_crop_height_pixels": 5,
            "crop_padding_ratio_x": 0.08,
            "crop_padding_ratio_y": 0.08,
            "minimum_padding_pixels": 2,
            "clamp_bbox_to_frame": True,
            "reject_invalid_bbox": True,
            "sharpness_enabled": True,
            "best_overall_weights": {
                "confidence": 0.35,
                "sharpness": 0.25,
                "bbox_area": 0.20,
                "centeredness": 0.10,
                "edge_visibility": 0.10,
            },
            "jpeg_quality": 90,
            "save_vehicle_crops": True,
            "save_annotated_full_frames": True,
            "save_all_candidates": False,
            "include_discarded_tracks": False,
            "fail_pipeline_on_error": False,
        },
        "visualization": {
            "show_rejected_boxes": False,
            "detected_frames": {"enabled": True, "save_every_n_frames": 1, "max_saved_frames_per_camera": 5},
            "tracked_frames": {"enabled": True, "save_every_n_frames": 1, "max_saved_frames_per_camera": 5},
        },
        "output": {"root_directory": output_root, "save_run_config": True},
        "debug_outputs": debug_outputs or {"enabled": False},
        "vehicle_enrichment": {
            "enabled": True,
            "trigger": "track_completed",
            "fail_open": True,
            "best_crops_per_track": 3,
            "write_separate_output": True,
            "extend_tracks_json": True,
            "evidence": {
                "source": "existing_track_evidence",
                "save_vehicle_crops": True,
                "minimum_crop_width": 5,
                "minimum_crop_height": 5,
                "minimum_sharpness": 1.0,
                "minimum_quality_score": 0.0,
                "border_margin_ratio": 0.02,
                "scoring": {
                    "area_weight": 0.25,
                    "sharpness_weight": 0.25,
                    "confidence_weight": 0.20,
                    "role_weight": 0.15,
                    "border_weight": 0.05,
                    "clipping_weight": 0.05,
                    "brightness_weight": 0.05,
                },
            },
            "shared_florence": {
                "enabled": False,
                "backend": "florence2",
                "base_model_id": "microsoft/Florence-2-base-ft",
                "adapter_path": "model_weights/florence/adaptor_florance_baseFT",
                "device": "auto",
                "trust_remote_code": True,
                "attention_implementation": "eager",
                "max_new_tokens": 1024,
                "num_beams": 3,
                "use_cache": False,
            },
            "body_type": {"enabled": False},
            "colour": {"enabled": False},
            "make_model": {"enabled": False},
            "plate": {"detection_enabled": False, "colour_enabled": False},
            "ocr": {"enabled": False, "run_only_when_plate_detected": True},
        },
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def test_validate_config_accepts_null_max_frames_per_camera(tmp_path: Path) -> None:
    video_path = _create_test_video(tmp_path / "sample.mp4", frame_count=4)
    model_path = tmp_path / "model.pt"
    model_path.write_bytes(b"x")
    config_path = tmp_path / "config.yaml"
    _write_config(
        config_path,
        cameras=[{"camera_id": "CAM_001", "source_type": "video", "source": str(video_path), "enabled": True}],
        output_root=str(tmp_path / "runs"),
        model_path=str(model_path),
        max_frames_per_camera=None,
    )

    validated = _validate_config(_load_raw_config(config_path), config_path)

    assert validated["input"]["max_frames_per_camera"] is None


def test_validate_config_rejects_invalid_max_frames_per_camera_values(tmp_path: Path) -> None:
    video_path = _create_test_video(tmp_path / "sample.mp4", frame_count=2)
    model_path = tmp_path / "model.pt"
    model_path.write_bytes(b"x")
    for bad_value in ("all", "", -10):
        config_path = tmp_path / f"config_{str(bad_value).replace('-', 'neg')}.yaml"
        _write_config(
            config_path,
            cameras=[{"camera_id": "CAM_001", "source_type": "video", "source": str(video_path), "enabled": True}],
            output_root=str(tmp_path / "runs"),
            model_path=str(model_path),
            max_frames_per_camera=bad_value,  # type: ignore[arg-type]
        )
        with pytest.raises(Exception, match="integer or null|positive integer or null"):
            _validate_config(_load_raw_config(config_path), config_path)


def test_validate_config_rejects_invalid_capture_zone_ratios(tmp_path: Path) -> None:
    video_path = _create_test_video(tmp_path / "sample.mp4", frame_count=2)
    model_path = tmp_path / "model.pt"
    model_path.write_bytes(b"x")
    config_path = tmp_path / "config_invalid_capture_zone.yaml"
    _write_config(
        config_path,
        cameras=[{"camera_id": "CAM_001", "source_type": "video", "source": str(video_path), "enabled": True}],
        output_root=str(tmp_path / "runs"),
        model_path=str(model_path),
        max_frames_per_camera=2,
    )
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    payload["evidence"]["capture_zone"] = {
        "enabled": True,
        "top_ratio": 0.80,
        "bottom_ratio": 0.55,
        "trigger_point": "bottom_center",
    }
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(Exception, match="capture_zone"):
        _validate_config(_load_raw_config(config_path), config_path)


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
    lifecycle_metrics = json.loads((run_dir / "track_lifecycle_metrics.json").read_text(encoding="utf-8"))
    evidence_index = json.loads((run_dir / "evidence_index.json").read_text(encoding="utf-8"))
    evidence_metrics = json.loads((run_dir / "evidence_metrics.json").read_text(encoding="utf-8"))
    enrichment_results = json.loads((run_dir / "vehicle_enrichment.json").read_text(encoding="utf-8"))
    enrichment_metrics = json.loads((run_dir / "vehicle_enrichment_metrics.json").read_text(encoding="utf-8"))
    validation_report = (run_dir / "vehicle_enrichment_validation_report.csv").read_text(encoding="utf-8")
    tracks = json.loads((run_dir / "tracks.json").read_text(encoding="utf-8"))
    observations = (run_dir / "observations.csv").read_text(encoding="utf-8")
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    metadata = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
    assert exit_code == 0
    assert run_id == run_dir.name
    assert summary["processed_frames"] == 3
    assert summary["configured_device"] == "cpu"
    assert summary["configured_dtype"] == "auto"
    assert summary["resolved_device"] == "cpu"
    assert summary["resolved_dtype"] == "float32"
    assert summary["cuda_available"] is False
    assert summary["cuda_device_name"] is None
    assert metrics["worker_count"] == 7
    assert metrics["enabled_camera_count"] == 1
    assert metadata["configured_device"] == "cpu"
    assert metadata["configured_dtype"] == "auto"
    assert metadata["resolved_device"] == "cpu"
    assert metadata["resolved_dtype"] == "float32"
    assert metadata["cuda_available"] is False
    assert metadata["cuda_device_count"] == 0
    assert metadata["cuda_device_name"] is None
    assert dt_metrics["detections_by_camera"]["CAM_001"] == 3
    assert dt_metrics["configured_device"] == "cpu"
    assert dt_metrics["configured_dtype"] == "auto"
    assert dt_metrics["resolved_device"] == "cpu"
    assert dt_metrics["resolved_dtype"] == "float32"
    assert dt_metrics["cuda_device_name"] is None
    assert bbox_metrics["accepted_detections"] == 3
    assert bbox_metrics["rejected_detections"] == 0
    assert lifecycle_metrics["active_tracks_at_shutdown"] == 0
    assert tracks[0]["local_track_id"] == "CAM_001:TRACK_1"
    assert tracks[0]["evidence_record_count"] >= 1
    assert "FIRST" in tracks[0]["evidence_roles"]
    assert evidence_index[0]["local_track_id"] == "CAM_001:TRACK_1"
    assert evidence_metrics["tracks_with_evidence"] >= 1
    assert enrichment_metrics["completed_tracks_received"] >= 1
    assert enrichment_metrics["florence_loaded"] is False
    assert enrichment_results[0]["vehicle_body_type"]["status"] == "disabled"
    assert "predicted_colour" in validation_report
    assert tracks[0]["vehicle_enrichment"]["status"] in {"evidence_ready", "disabled", "no_evidence"}
    assert "CAM_001:TRACK_1" in observations


def test_pipeline_supports_unlimited_frame_limit_and_logs_it(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(pipeline_module, "VehicleDetectorTracker", FakeVehicleDetectorTracker)
    video_path = _create_test_video(tmp_path / "sample.mp4", frame_count=4)
    model_path = tmp_path / "model.pt"
    model_path.write_bytes(b"x")
    output_root = tmp_path / "runs"
    config_path = tmp_path / "config.yaml"
    _write_config(
        config_path,
        cameras=[{"camera_id": "CAM_001", "source_type": "video", "source": str(video_path), "enabled": True}],
        output_root=str(output_root),
        model_path=str(model_path),
        max_frames_per_camera=None,
    )

    exit_code, _run_id, run_directory = run_pipeline(str(config_path))
    run_dir = Path(run_directory)
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    pipeline_log = (run_dir / "pipeline.log").read_text(encoding="utf-8")

    assert exit_code == 0
    assert summary["processed_frames"] == 4
    assert summary["max_frames_per_camera"] is None
    assert summary["frame_limit_mode"] == "unlimited"
    assert "Frame limit per camera: unlimited" in pipeline_log


def test_pipeline_writes_motorcycle_geometry_report_summary_entry(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(pipeline_module, "VehicleDetectorTracker", FakeVehicleDetectorTracker)
    video_path = _create_test_video(tmp_path / "sample.mp4", frame_count=3)
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

    exit_code, _run_id, run_directory = run_pipeline(str(config_path))
    run_dir = Path(run_directory)
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))

    assert exit_code == 0
    assert (run_dir / "motorcycle_geometry_report.csv").exists()
    assert summary["motorcycle_geometry"]["report_path"].endswith("motorcycle_geometry_report.csv")
    assert summary["motorcycle_geometry"]["motorcycle_tracks_total"] == 0


def test_pipeline_debug_outputs_write_numbered_directories(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(pipeline_module, "VehicleDetectorTracker", FakeVehicleDetectorTracker)
    video_path = _create_test_video(tmp_path / "sample.mp4", frame_count=3)
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
        debug_outputs={
            "enabled": True,
            "extracted_frames": {"enabled": True, "save_every_n_frames": 1, "max_saved_frames_per_camera": 10},
            "detected_frames": {"enabled": True, "save_every_n_frames": 1, "max_saved_frames_per_camera": 10},
            "tracked_frames": {"enabled": True, "save_every_n_frames": 1, "max_saved_frames_per_camera": 10},
            "track_crops": {"enabled": True, "save_every_n_frames": 1, "max_crops_per_track": 10},
            "florence_selected_crops": {"enabled": True},
        },
    )

    exit_code, _run_id, run_directory = run_pipeline(str(config_path))
    run_dir = Path(run_directory)

    assert exit_code == 0
    assert (run_dir / "01_extracted_frames" / "CAM_001").exists()
    assert (run_dir / "02_yolo_detected_frames" / "CAM_001").exists()
    assert (run_dir / "03_tracked_frames" / "CAM_001").exists()
    assert (run_dir / "04_track_crops" / "track_crop_manifest.csv").exists()


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
    lifecycle_metrics = json.loads((run_dir / "track_lifecycle_metrics.json").read_text(encoding="utf-8"))
    evidence_metrics = json.loads((run_dir / "evidence_metrics.json").read_text(encoding="utf-8"))
    enrichment_metrics = json.loads((run_dir / "vehicle_enrichment_metrics.json").read_text(encoding="utf-8"))
    assert exit_code == 0
    assert metrics["frames_by_camera"]["CAM_001"] == 4
    assert metrics["frames_by_camera"]["CAM_002"] == 6
    assert dt_metrics["tracked_observations_by_camera"]["CAM_001"] == 4
    assert dt_metrics["tracked_observations_by_camera"]["CAM_002"] == 6
    assert lifecycle_metrics["tracks_completed_by_camera"]["CAM_001"] >= 1
    assert evidence_metrics["tracks_received"] >= 2
    assert enrichment_metrics["completed_tracks_received"] >= 2
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


def test_enrichment_enabled_does_not_change_tracking_outputs(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(pipeline_module, "VehicleDetectorTracker", FakeVehicleDetectorTracker)
    video_path = _create_test_video(tmp_path / "sample.mp4", frame_count=4)
    model_path = tmp_path / "model.pt"
    model_path.write_bytes(b"x")
    enabled_config = tmp_path / "enabled.yaml"
    disabled_config = tmp_path / "disabled.yaml"
    _write_config(
        enabled_config,
        cameras=[{"camera_id": "CAM_001", "source_type": "video", "source": str(video_path), "enabled": True}],
        output_root=str(tmp_path / "runs_enabled"),
        model_path=str(model_path),
        max_frames_per_camera=4,
    )
    _write_config(
        disabled_config,
        cameras=[{"camera_id": "CAM_001", "source_type": "video", "source": str(video_path), "enabled": True}],
        output_root=str(tmp_path / "runs_disabled"),
        model_path=str(model_path),
        max_frames_per_camera=4,
    )
    disabled_payload = yaml.safe_load(disabled_config.read_text(encoding="utf-8"))
    disabled_payload["vehicle_enrichment"]["enabled"] = False
    disabled_config.write_text(yaml.safe_dump(disabled_payload, sort_keys=False), encoding="utf-8")

    enabled_exit_code, _enabled_run_id, enabled_run_dir = run_pipeline(str(enabled_config))
    disabled_exit_code, _disabled_run_id, disabled_run_dir = run_pipeline(str(disabled_config))

    assert enabled_exit_code == 0
    assert disabled_exit_code == 0

    enabled_tracks = json.loads((Path(enabled_run_dir) / "tracks.json").read_text(encoding="utf-8"))
    disabled_tracks = json.loads((Path(disabled_run_dir) / "tracks.json").read_text(encoding="utf-8"))
    enabled_observations = (Path(enabled_run_dir) / "observations.csv").read_text(encoding="utf-8")
    disabled_observations = (Path(disabled_run_dir) / "observations.csv").read_text(encoding="utf-8")

    def _normalize_track(track: dict[str, object]) -> dict[str, object]:
        payload = {key: value for key, value in track.items() if key != "vehicle_enrichment"}
        payload["evidence_directory"] = "<normalized>"
        return payload

    comparable_enabled = [_normalize_track(item) for item in enabled_tracks]
    comparable_disabled = [_normalize_track(item) for item in disabled_tracks]

    assert comparable_enabled == comparable_disabled
    assert enabled_observations == disabled_observations


def test_pipeline_builds_vehicle_colour_only_artifact_rows() -> None:
    prediction = AttributePrediction(
        attribute_name="vehicle_colour",
        label="BLACK",
        source_backend="base_florence",
        source_model="model",
        source_frame_number=12,
        source_crop_path="crop.jpg",
        raw_response="black",
        confidence=None,
        quality_weight=0.9,
        inference_duration_ms=45.0,
        status="completed",
        reason="valid",
    )
    result = TrackEnrichmentResult(
        local_track_id="TRACK_1",
        camera_id="CAM_001",
        vehicle_class="CAR",
        vehicle_class_confidence=0.9,
        vehicle_body_type=VehicleBodyTypeResult(label="UNKNOWN", status="disabled", source="base_florence", reason="disabled"),
        vehicle_colour=VehicleColourResult(
            label="BLACK",
            predictions=[prediction],
            status="completed",
            source="base_florence",
            reason=None,
            task_prompt="<VQA>",
            prompt_text="What colour is the vehicle?",
        ),
        vehicle_make=None,
        vehicle_model=None,
        plate_detected=False,
        plate_colour=None,
        registration_category=None,
        plate_text=None,
        status="completed",
        attribute_backend="base_florence",
        vehicle_attribute_inference_count=1,
        evidence_used=[],
        crop_level_captions=[{"crop_path": "crop.jpg", "frame_index": 12, "caption": "black"}],
        crop_level_colours=[{"crop_path": "crop.jpg", "frame_index": 12, "normalized_colour": "BLACK", "status": "completed", "reason": "valid", "crop_source": "saved_vehicle_crop", "crop_available": True, "crop_skip_reason": None}],
        crop_level_body_types=[{"crop_path": "crop.jpg", "frame_index": 12, "normalized_body_type": "UNKNOWN"}],
        final_colour_reason="weighted_agreement",
    )

    crop_rows = _build_vehicle_colour_result_rows([result])
    track_rows = _build_vehicle_colour_track_summary_rows([result])

    assert crop_rows[0]["task_token"] == "<VQA>"
    assert crop_rows[0]["prompt"] == "What colour is the vehicle?"
    assert crop_rows[0]["parsed_colour"] == "BLACK"
    assert crop_rows[0]["adapter_loaded"] is False
    assert track_rows[0]["final_vehicle_colour"] == "BLACK"
    assert track_rows[0]["colour_inference_count"] == 1
