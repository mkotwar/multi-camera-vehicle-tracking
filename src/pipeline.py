from __future__ import annotations

import queue
import traceback
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from .detector_tracker import VehicleDetectorTracker
from .evidence import EvidenceCollector
from .ingestion_manager import MultiCameraIngestionManager
from .logging_setup import setup_logging
from .models import (
    ConfigurationError,
    LocalTrack,
    RUN_STATUS_COMPLETED,
    RUN_STATUS_CREATED,
    RUN_STATUS_FAILED,
    RUN_STATUS_RUNNING,
    TrackObservation,
    RunMetadata,
)
from .output_writer import RunOutputManager
from .track_manager import TrackManager
from .vehicle_enrichment import VehicleEnrichmentManager, normalize_vehicle_enrichment_config


def _normalize_bbox_quality_section(raw_bbox_quality: Any) -> dict[str, Any]:
    bbox_quality = dict(raw_bbox_quality or {})
    has_profiles = "default" in bbox_quality or "classes" in bbox_quality
    if has_profiles:
        default_payload = dict(bbox_quality.get("default", {}) or {})
        classes_payload = dict(bbox_quality.get("classes", {}) or {})
    else:
        default_payload = dict(bbox_quality)
        classes_payload = {}

    def _normalize_profile(profile: dict[str, Any], fallback: dict[str, Any] | None = None) -> dict[str, Any]:
        base = fallback or {
            "minimum_width_pixels": 60.0,
            "minimum_height_pixels": 60.0,
            "minimum_area_ratio": 0.005,
            "maximum_area_ratio": 0.90,
            "minimum_aspect_ratio": 0.30,
            "maximum_aspect_ratio": 4.50,
            "edge_margin_pixels": 8.0,
            "edge_mode": "C",
        }
        edge_mode = profile.get("edge_mode")
        if edge_mode is None:
            if "reject_edge_truncated" in profile:
                if has_profiles:
                    edge_mode = "C" if bool(profile.get("reject_edge_truncated", True)) else "A"
                else:
                    edge_mode = "LEGACY" if bool(profile.get("reject_edge_truncated", True)) else "A"
            else:
                edge_mode = base["edge_mode"]
        normalized = {
            "minimum_width_pixels": float(profile.get("minimum_width_pixels", base["minimum_width_pixels"])),
            "minimum_height_pixels": float(profile.get("minimum_height_pixels", base["minimum_height_pixels"])),
            "minimum_area_ratio": float(profile.get("minimum_area_ratio", base["minimum_area_ratio"])),
            "maximum_area_ratio": float(profile.get("maximum_area_ratio", base["maximum_area_ratio"])),
            "minimum_aspect_ratio": float(profile.get("minimum_aspect_ratio", base["minimum_aspect_ratio"])),
            "maximum_aspect_ratio": float(profile.get("maximum_aspect_ratio", base["maximum_aspect_ratio"])),
            "edge_margin_pixels": float(profile.get("edge_margin_pixels", base["edge_margin_pixels"])),
            "edge_mode": str(edge_mode).strip().upper() or base["edge_mode"],
        }
        if normalized["minimum_width_pixels"] < 0.0:
            raise ConfigurationError("detection.bbox_quality minimum_width_pixels must be at least 0.")
        if normalized["minimum_height_pixels"] < 0.0:
            raise ConfigurationError("detection.bbox_quality minimum_height_pixels must be at least 0.")
        if normalized["minimum_area_ratio"] < 0.0 or normalized["maximum_area_ratio"] < 0.0:
            raise ConfigurationError("detection.bbox_quality area ratios must be at least 0.")
        if normalized["maximum_area_ratio"] < normalized["minimum_area_ratio"]:
            raise ConfigurationError(
                "detection.bbox_quality.maximum_area_ratio must be greater than or equal to minimum_area_ratio."
            )
        if normalized["minimum_aspect_ratio"] <= 0.0 or normalized["maximum_aspect_ratio"] <= 0.0:
            raise ConfigurationError("detection.bbox_quality aspect ratios must be positive.")
        if normalized["maximum_aspect_ratio"] < normalized["minimum_aspect_ratio"]:
            raise ConfigurationError(
                "detection.bbox_quality.maximum_aspect_ratio must be greater than or equal to minimum_aspect_ratio."
            )
        if normalized["edge_margin_pixels"] < 0.0:
            raise ConfigurationError("detection.bbox_quality.edge_margin_pixels must be at least 0.")
        if normalized["edge_mode"] not in {"A", "B", "C", "LEGACY"}:
            raise ConfigurationError("detection.bbox_quality.edge_mode must be one of: A, B, C, LEGACY.")
        return normalized

    normalized_default = _normalize_profile(default_payload)
    normalized_classes: dict[str, dict[str, Any]] = {}
    for class_name, class_profile in classes_payload.items():
        if not isinstance(class_profile, dict):
            raise ConfigurationError(f"detection.bbox_quality.classes.{class_name} must be a mapping.")
        normalized_classes[str(class_name).strip()] = _normalize_profile(dict(class_profile), fallback=normalized_default)
    return {
        "enabled": bool(bbox_quality.get("enabled", False)),
        "default": normalized_default,
        "classes": normalized_classes,
    }


def _load_raw_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path).expanduser().resolve()
    if not path.exists():
        raise ConfigurationError(f"Configuration file does not exist: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ConfigurationError("Configuration root must be a mapping.")
    return payload


def _validate_config(raw_config: dict[str, Any], config_path: Path) -> dict[str, Any]:
    project = raw_config.get("project")
    input_section = raw_config.get("input")
    ingestion = raw_config.get("ingestion")
    detection = raw_config.get("detection")
    tracking = raw_config.get("tracking")
    evidence = raw_config.get("evidence")
    visualization = raw_config.get("visualization")
    output_section = raw_config.get("output")
    vehicle_enrichment = raw_config.get("vehicle_enrichment")
    if not isinstance(project, dict):
        raise ConfigurationError("Missing or invalid 'project' section.")
    if not isinstance(input_section, dict):
        raise ConfigurationError("Missing or invalid 'input' section.")
    if not isinstance(ingestion, dict):
        raise ConfigurationError("Missing or invalid 'ingestion' section.")
    if not isinstance(detection, dict):
        raise ConfigurationError("Missing or invalid 'detection' section.")
    if not isinstance(tracking, dict):
        raise ConfigurationError("Missing or invalid 'tracking' section.")
    if evidence is not None and not isinstance(evidence, dict):
        raise ConfigurationError("Invalid 'evidence' section.")
    if not isinstance(visualization, dict):
        raise ConfigurationError("Missing or invalid 'visualization' section.")
    if not isinstance(output_section, dict):
        raise ConfigurationError("Missing or invalid 'output' section.")
    if vehicle_enrichment is not None and not isinstance(vehicle_enrichment, dict):
        raise ConfigurationError("Invalid 'vehicle_enrichment' section.")

    cameras = input_section.get("cameras")
    if not isinstance(cameras, list):
        raise ConfigurationError("input.cameras must be a list.")

    normalized_cameras: list[dict[str, Any]] = []
    seen_camera_ids: set[str] = set()
    enabled_count = 0
    for camera in cameras:
        if not isinstance(camera, dict):
            raise ConfigurationError("Each camera entry must be a mapping.")
        camera_id = str(camera.get("camera_id", "")).strip()
        source_type = str(camera.get("source_type", "")).strip().lower()
        source = camera.get("source")
        enabled = bool(camera.get("enabled", False))
        if not camera_id:
            raise ConfigurationError("camera_id is required for every camera.")
        if camera_id in seen_camera_ids:
            raise ConfigurationError(f"Duplicate camera_id found: {camera_id}")
        seen_camera_ids.add(camera_id)
        if source_type not in {"video", "rtsp", "webcam"}:
            raise ConfigurationError(f"Unsupported source type for camera '{camera_id}': {source_type or '<empty>'}")
        if source in (None, ""):
            raise ConfigurationError(f"source is required for camera '{camera_id}'.")
        if source_type == "webcam":
            try:
                normalized_source: str | int = int(source)
            except Exception as exc:
                raise ConfigurationError(f"Webcam source must be an integer for camera '{camera_id}'.") from exc
        else:
            normalized_source = str(source).strip()
            if not normalized_source:
                raise ConfigurationError(f"source is required for camera '{camera_id}'.")
            if source_type == "video":
                normalized_source = str(Path(normalized_source).expanduser().resolve())
        normalized_cameras.append(
            {
                "camera_id": camera_id,
                "source_type": source_type,
                "source": normalized_source,
                "enabled": enabled,
            }
        )
        if enabled:
            enabled_count += 1
    if enabled_count < 1:
        raise ConfigurationError("At least one enabled camera is required.")

    max_frames_per_camera = int(input_section.get("max_frames_per_camera", 0))
    if max_frames_per_camera <= 0:
        raise ConfigurationError("input.max_frames_per_camera must be a positive integer.")

    worker_count = int(ingestion.get("worker_count", 7))
    target_read_fps = ingestion.get("target_read_fps", 10.0)
    frame_queue_size = int(ingestion.get("frame_queue_size", 200))
    if worker_count < 1:
        raise ConfigurationError("ingestion.worker_count must be at least 1.")
    if frame_queue_size < 1:
        raise ConfigurationError("ingestion.frame_queue_size must be at least 1.")
    if target_read_fps is not None and float(target_read_fps) <= 0.0:
        raise ConfigurationError("ingestion.target_read_fps must be positive when provided.")

    raw_frames = dict(ingestion.get("raw_frames", {}) or {})
    raw_frames_enabled = bool(raw_frames.get("enabled", True))
    save_every_n_frames = int(raw_frames.get("save_every_n_frames", 10))
    max_saved_frames_per_camera = int(raw_frames.get("max_saved_frames_per_camera", 50))
    image_format = str(raw_frames.get("image_format", "jpg")).strip().lower() or "jpg"
    jpeg_quality = int(raw_frames.get("jpeg_quality", 90))
    if save_every_n_frames < 1:
        raise ConfigurationError("ingestion.raw_frames.save_every_n_frames must be at least 1.")
    if max_saved_frames_per_camera < 0:
        raise ConfigurationError("ingestion.raw_frames.max_saved_frames_per_camera must be at least 0.")
    if image_format not in {"jpg", "jpeg", "png"}:
        raise ConfigurationError("ingestion.raw_frames.image_format must be one of: jpg, jpeg, png.")
    if not 0 <= jpeg_quality <= 100:
        raise ConfigurationError("ingestion.raw_frames.jpeg_quality must be between 0 and 100.")

    detection_model_path = detection.get("model_path")
    if detection_model_path in (None, ""):
        raise ConfigurationError("detection.model_path is required.")
    resolved_model_path = Path(str(detection_model_path)).expanduser()
    if not resolved_model_path.is_absolute():
        resolved_model_path = (config_path.parent / resolved_model_path).resolve()
    else:
        resolved_model_path = resolved_model_path.resolve()
    if not resolved_model_path.exists():
        raise ConfigurationError(f"Resolved model path does not exist: {resolved_model_path}")
    detection_backend = str(detection.get("backend", "ocr_mukul")).strip().lower() or "ocr_mukul"
    allowed_classes = [str(item).strip() for item in detection.get("allowed_classes", []) if str(item).strip()]
    allowed_class_ids = [int(item) for item in detection.get("allowed_class_ids", list(range(8)))]
    if detection_backend == "legacy_clean" and not allowed_classes:
        raise ConfigurationError("detection.allowed_classes must not be empty for legacy_clean backend.")
    normalized_bbox_quality = _normalize_bbox_quality_section(detection.get("bbox_quality", {}))
    if detection_backend == "ocr_mukul":
        normalized_bbox_quality["enabled"] = False
    lifecycle = dict(raw_config.get("lifecycle", {}) or {})
    track_class = dict(raw_config.get("track_class", {}) or {})
    lifecycle_minimum_observations = int(lifecycle.get("minimum_observations", 3))
    lifecycle_maximum_lost_frames = int(lifecycle.get("maximum_lost_frames", 30))
    if lifecycle_minimum_observations < 1:
        raise ConfigurationError("lifecycle.minimum_observations must be at least 1.")
    if lifecycle_maximum_lost_frames < 0:
        raise ConfigurationError("lifecycle.maximum_lost_frames must be at least 0.")
    track_class_minimum_observations = int(track_class.get("minimum_observations", 3))
    track_class_minimum_winner_ratio = float(track_class.get("minimum_winner_ratio", 0.60))
    track_class_strategy = str(track_class.get("strategy", "confidence_weighted_majority")).strip() or "confidence_weighted_majority"
    unknown_class_name = str(track_class.get("unknown_class_name", "UNKNOWN")).strip() or "UNKNOWN"
    if track_class_minimum_observations < 1:
        raise ConfigurationError("track_class.minimum_observations must be at least 1.")
    if not 0.0 <= track_class_minimum_winner_ratio <= 1.0:
        raise ConfigurationError("track_class.minimum_winner_ratio must be between 0 and 1.")
    if track_class_strategy != "confidence_weighted_majority":
        raise ConfigurationError("track_class.strategy must be confidence_weighted_majority.")

    detected_frames = dict(visualization.get("detected_frames", {}) or {})
    tracked_frames = dict(visualization.get("tracked_frames", {}) or {})
    for name, payload in (("detected_frames", detected_frames), ("tracked_frames", tracked_frames)):
        if int(payload.get("save_every_n_frames", 10)) < 1:
            raise ConfigurationError(f"visualization.{name}.save_every_n_frames must be at least 1.")
        if int(payload.get("max_saved_frames_per_camera", 20)) < 0:
            raise ConfigurationError(f"visualization.{name}.max_saved_frames_per_camera must be at least 0.")

    output_root = Path(str(output_section.get("root_directory", "outputs/runs"))).expanduser()
    if not output_root.is_absolute():
        output_root = (config_path.parent / output_root).resolve()
    else:
        output_root = output_root.resolve()
    evidence_section = dict(evidence or {})
    evidence_weights = dict(evidence_section.get("best_overall_weights", {}) or {})
    normalized_vehicle_enrichment = normalize_vehicle_enrichment_config(vehicle_enrichment or {})

    return {
        "project": {
            "name": str(project.get("name", "multicamera_vehicle_tracking")).strip() or "multicamera_vehicle_tracking",
            "environment": str(project.get("environment", "development")).strip() or "development",
            "log_level": str(project.get("log_level", "INFO")).strip() or "INFO",
        },
        "input": {
            "cameras": normalized_cameras,
            "max_frames_per_camera": max_frames_per_camera,
        },
        "ingestion": {
            "worker_count": worker_count,
            "target_read_fps": None if target_read_fps is None else float(target_read_fps),
            "frame_queue_size": frame_queue_size,
            "queue_put_timeout_seconds": float(ingestion.get("queue_put_timeout_seconds", 2.0)),
            "queue_get_timeout_seconds": float(ingestion.get("queue_get_timeout_seconds", 1.0)),
            "stop_on_camera_error": bool(ingestion.get("stop_on_camera_error", False)),
            "round_robin": bool(ingestion.get("round_robin", True)),
            "raw_frames": {
                "enabled": raw_frames_enabled,
                "save_every_n_frames": save_every_n_frames,
                "max_saved_frames_per_camera": max_saved_frames_per_camera,
                "image_format": image_format,
                "jpeg_quality": jpeg_quality,
            },
        },
        "detection": {
            "backend": detection_backend,
            "model_path": str(resolved_model_path),
            "device": str(detection.get("device", "auto")),
            "confidence_threshold": float(detection.get("confidence_threshold", 0.2 if detection_backend == "ocr_mukul" else 0.38)),
            "iou_threshold": float(detection.get("iou_threshold", 0.45)),
            "image_size": int(detection.get("image_size", 1024 if detection_backend == "ocr_mukul" else 640)),
            "agnostic_nms": bool(detection.get("agnostic_nms", False)),
            "allowed_classes": allowed_classes,
            "allowed_class_ids": allowed_class_ids,
            "bbox_quality": normalized_bbox_quality,
        },
        "tracking": {
            "backend": str(tracking.get("backend", "ocr_mukul_supervision_bytetrack")),
            "track_activation_threshold": float(tracking.get("track_activation_threshold", 0.3 if detection_backend == "ocr_mukul" else 0.15)),
            "lost_track_buffer": int(tracking.get("lost_track_buffer", 40 if detection_backend == "ocr_mukul" else 30)),
            "minimum_matching_threshold": float(tracking.get("minimum_matching_threshold", 0.6 if detection_backend == "ocr_mukul" else 0.80)),
            "minimum_consecutive_frames": int(tracking.get("minimum_consecutive_frames", 3 if detection_backend == "ocr_mukul" else 1)),
            "isolation_mode": str(tracking.get("isolation_mode", "per_camera")).strip() or "per_camera",
            "supported_isolation_modes": [
                str(item).strip()
                for item in tracking.get("supported_isolation_modes", ["per_camera"] if detection_backend == "ocr_mukul" else ["per_camera", "per_camera_class"])
                if str(item).strip()
            ],
        },
        "lifecycle": {
            "minimum_observations": lifecycle_minimum_observations,
            "maximum_lost_frames": lifecycle_maximum_lost_frames,
            "keep_discarded_tracks": bool(lifecycle.get("keep_discarded_tracks", True)),
        },
        "track_class": {
            "minimum_observations": track_class_minimum_observations,
            "minimum_winner_ratio": track_class_minimum_winner_ratio,
            "strategy": track_class_strategy,
            "unknown_class_name": unknown_class_name,
        },
        "evidence": {
            "enabled": bool(evidence_section.get("enabled", True)),
            "collect_first": bool(evidence_section.get("collect_first", True)),
            "collect_middle": bool(evidence_section.get("collect_middle", True)),
            "collect_last": bool(evidence_section.get("collect_last", True)),
            "collect_highest_confidence": bool(evidence_section.get("collect_highest_confidence", True)),
            "collect_largest": bool(evidence_section.get("collect_largest", True)),
            "collect_sharpest": bool(evidence_section.get("collect_sharpest", True)),
            "collect_best_overall": bool(evidence_section.get("collect_best_overall", True)),
            "maximum_candidates_per_track": int(evidence_section.get("maximum_candidates_per_track", 7)),
            "minimum_crop_width_pixels": int(evidence_section.get("minimum_crop_width_pixels", 40)),
            "minimum_crop_height_pixels": int(evidence_section.get("minimum_crop_height_pixels", 40)),
            "crop_padding_ratio_x": float(evidence_section.get("crop_padding_ratio_x", 0.08)),
            "crop_padding_ratio_y": float(evidence_section.get("crop_padding_ratio_y", 0.08)),
            "minimum_padding_pixels": int(evidence_section.get("minimum_padding_pixels", 8)),
            "clamp_bbox_to_frame": bool(evidence_section.get("clamp_bbox_to_frame", True)),
            "reject_invalid_bbox": bool(evidence_section.get("reject_invalid_bbox", True)),
            "sharpness_enabled": bool(evidence_section.get("sharpness_enabled", True)),
            "jpeg_quality": int(evidence_section.get("jpeg_quality", 90)),
            "save_vehicle_crops": bool(evidence_section.get("save_vehicle_crops", True)),
            "save_annotated_full_frames": bool(evidence_section.get("save_annotated_full_frames", True)),
            "save_all_candidates": bool(evidence_section.get("save_all_candidates", False)),
            "include_discarded_tracks": bool(evidence_section.get("include_discarded_tracks", False)),
            "fail_pipeline_on_error": bool(evidence_section.get("fail_pipeline_on_error", False)),
            "best_overall_weights": {
                "confidence": float(evidence_weights.get("confidence", 0.35)),
                "sharpness": float(evidence_weights.get("sharpness", 0.25)),
                "bbox_area": float(evidence_weights.get("bbox_area", 0.20)),
                "centeredness": float(evidence_weights.get("centeredness", 0.10)),
                "edge_visibility": float(evidence_weights.get("edge_visibility", 0.10)),
            },
        },
        "visualization": {
            "show_rejected_boxes": bool(visualization.get("show_rejected_boxes", False)),
            "detected_frames": {
                "enabled": bool(detected_frames.get("enabled", True)),
                "save_every_n_frames": int(detected_frames.get("save_every_n_frames", 10)),
                "max_saved_frames_per_camera": int(detected_frames.get("max_saved_frames_per_camera", 20)),
            },
            "tracked_frames": {
                "enabled": bool(tracked_frames.get("enabled", True)),
                "save_every_n_frames": int(tracked_frames.get("save_every_n_frames", 10)),
                "max_saved_frames_per_camera": int(tracked_frames.get("max_saved_frames_per_camera", 20)),
            },
        },
        "output": {
            "root_directory": str(output_root),
            "save_run_config": bool(output_section.get("save_run_config", True)),
        },
        "vehicle_enrichment": normalized_vehicle_enrichment,
    }


def _should_save_raw_frame(frame_number: int, saved_so_far: int, raw_frames_config: dict[str, Any]) -> bool:
    if not bool(raw_frames_config.get("enabled", False)):
        return False
    if saved_so_far >= int(raw_frames_config.get("max_saved_frames_per_camera", 0)):
        return False
    every_n = int(raw_frames_config.get("save_every_n_frames", 1))
    return frame_number % every_n == 0


def _should_save_visualization_frame(frame_number: int, saved_so_far: int, visualization_config: dict[str, Any]) -> bool:
    if not bool(visualization_config.get("enabled", False)):
        return False
    if saved_so_far >= int(visualization_config.get("max_saved_frames_per_camera", 0)):
        return False
    every_n = int(visualization_config.get("save_every_n_frames", 1))
    return frame_number % every_n == 0


def run_pipeline(config_path: str) -> tuple[int, str, str]:
    config_file = Path(config_path).expanduser().resolve()
    raw_config = {}
    try:
        raw_config = _load_raw_config(config_file)
    except Exception:
        raw_config = {}
    output_root = raw_config.get("output", {}).get("root_directory", "outputs/runs") if isinstance(raw_config.get("output"), dict) else "outputs/runs"
    output_root_path = Path(output_root).expanduser()
    if not output_root_path.is_absolute():
        output_root_path = (config_file.parent / output_root_path).resolve()
    else:
        output_root_path = output_root_path.resolve()

    output_manager = RunOutputManager(output_root_path)
    requested_log_level = str(raw_config.get("project", {}).get("log_level", "INFO")) if isinstance(raw_config.get("project"), dict) else "INFO"
    deferred_setup_error: Exception | None = None
    try:
        logger = setup_logging(output_manager.run_directory, log_level=requested_log_level)
    except ConfigurationError as exc:
        logger = setup_logging(output_manager.run_directory, log_level="INFO")
        deferred_setup_error = exc

    metadata = RunMetadata(
        run_id=output_manager.run_id,
        project_name="multicamera_vehicle_tracking",
        started_at=datetime.now(timezone.utc).isoformat(),
        completed_at=None,
        status=RUN_STATUS_CREATED,
        camera_count=0,
        processed_frames=0,
        completed_tracks=0,
        error_count=0,
        config_path=str(config_file),
    )
    output_manager.save_metadata(metadata)

    ingestion_manager: MultiCameraIngestionManager | None = None
    track_manager: TrackManager | None = None
    evidence_collector: EvidenceCollector | None = None
    vehicle_enrichment_manager: VehicleEnrichmentManager | None = None
    try:
        logger.info("Pipeline started")
        if deferred_setup_error is not None:
            raise deferred_setup_error
        validated_config = _validate_config(raw_config or _load_raw_config(config_file), config_file)
        metadata.project_name = validated_config["project"]["name"]
        metadata.camera_count = len([camera for camera in validated_config["input"]["cameras"] if camera["enabled"]])
        if bool(validated_config["output"]["save_run_config"]):
            output_manager.save_effective_config(validated_config)
        logger.info("Config loaded")
        metadata.status = RUN_STATUS_RUNNING
        output_manager.save_metadata(metadata)

        ingestion_manager = MultiCameraIngestionManager(validated_config, logger)
        detector_tracker = VehicleDetectorTracker(validated_config, logger)
        device_info = detector_tracker.runtime_device_info
        metadata.configured_device = device_info.configured_device
        metadata.resolved_device = device_info.resolved_device
        metadata.cuda_available = device_info.cuda_available
        metadata.cuda_device_count = device_info.cuda_device_count
        metadata.cuda_device_name = device_info.cuda_device_name
        metadata.torch_version = device_info.torch_version
        metadata.torch_cuda_version = device_info.torch_cuda_version
        output_manager.save_metadata(metadata)
        if device_info.resolved_device.startswith("cuda:"):
            selected_cuda_index = int(device_info.resolved_device.split(":", 1)[1])
            logger.info(
                'Runtime device: configured=%s resolved=%s gpu="%s" cuda_available=%s cuda_device_count=%s selected_cuda_index=%s torch_version=%s torch_cuda_version=%s',
                device_info.configured_device,
                device_info.resolved_device,
                device_info.cuda_device_name,
                device_info.cuda_available,
                device_info.cuda_device_count,
                selected_cuda_index,
                device_info.torch_version,
                device_info.torch_cuda_version,
            )
        else:
            logger.info(
                "Runtime device: configured=%s resolved=%s cuda_available=%s reason=CUDA is not available through the installed PyTorch build torch_version=%s torch_cuda_version=%s",
                device_info.configured_device,
                device_info.resolved_device,
                device_info.cuda_available,
                device_info.torch_version,
                device_info.torch_cuda_version,
            )
        track_manager = TrackManager(validated_config, logger)
        evidence_collector = EvidenceCollector(validated_config, logger, output_manager)
        vehicle_enrichment_manager = VehicleEnrichmentManager(validated_config, logger, output_manager)
        ingestion_manager.start()

        raw_frames_config = dict(validated_config["ingestion"]["raw_frames"])
        detected_frames_config = dict(validated_config["visualization"]["detected_frames"])
        tracked_frames_config = dict(validated_config["visualization"]["tracked_frames"])
        frames_by_camera: dict[str, int] = {camera["camera_id"]: 0 for camera in validated_config["input"]["cameras"] if camera["enabled"]}
        saved_raw_frames_by_camera: dict[str, int] = {camera_id: 0 for camera_id in frames_by_camera}
        detections_by_camera: dict[str, int] = {camera_id: 0 for camera_id in frames_by_camera}
        tracked_observations_by_camera: dict[str, int] = {camera_id: 0 for camera_id in frames_by_camera}
        detections_by_class: dict[str, int] = {}
        bbox_quality_diagnostics: list[dict[str, Any]] = []
        raw_detections = 0
        accepted_detections = 0
        rejected_detections = 0
        rejected_by_reason: dict[str, int] = {}
        rejected_by_class: dict[str, int] = {}
        accepted_by_class: dict[str, int] = {}
        unique_native_track_ids_by_camera: dict[str, set[int]] = {camera_id: set() for camera_id in frames_by_camera}
        lifecycle_completed_tracks: list[LocalTrack] = []
        saved_detected_frames_by_camera: dict[str, int] = {camera_id: 0 for camera_id in frames_by_camera}
        saved_tracked_frames_by_camera: dict[str, int] = {camera_id: 0 for camera_id in frames_by_camera}
        enrichment_results = []

        while True:
            try:
                packet = ingestion_manager.get_packet()
            except queue.Empty:
                if ingestion_manager.is_finished() and ingestion_manager.frame_queue.empty():
                    break
                continue

            frames_by_camera[packet.camera_id] += 1
            metadata.processed_frames += 1
            result = detector_tracker.process_frame(packet)
            detections_by_camera[packet.camera_id] += len(result.detections)
            tracked_observations_by_camera[packet.camera_id] += len(result.tracked_detections)
            raw_detections += len(result.bbox_quality_diagnostics)
            accepted_detections += len(result.detections)
            bbox_quality_diagnostics.extend(asdict(item) for item in result.bbox_quality_diagnostics)
            for detection_item in result.detections:
                normalized_class = str(detection_item.class_name).strip().lower()
                detections_by_class[normalized_class] = detections_by_class.get(normalized_class, 0) + 1
                accepted_by_class[normalized_class] = accepted_by_class.get(normalized_class, 0) + 1
            for diagnostic in result.bbox_quality_diagnostics:
                if diagnostic.accepted_by_bbox_quality:
                    continue
                rejected_detections += 1
                normalized_class = str(diagnostic.class_name).strip().lower()
                rejected_by_class[normalized_class] = rejected_by_class.get(normalized_class, 0) + 1
                if diagnostic.rejection_reason is not None:
                    rejected_by_reason[diagnostic.rejection_reason] = rejected_by_reason.get(diagnostic.rejection_reason, 0) + 1
            for tracked_item in result.tracked_detections:
                unique_native_track_ids_by_camera[packet.camera_id].add(tracked_item.tracker_id)
            evidence_collector.register_frame(packet, result.tracked_detections)
            completed_now = track_manager.update_frame(packet.camera_id, packet.frame_number, result.tracked_detections)
            lifecycle_completed_tracks.extend(completed_now)
            finalized_evidence_now = evidence_collector.finalize_tracks(completed_now)
            enrichment_results.extend(vehicle_enrichment_manager.enrich_completed_tracks(completed_now, finalized_evidence_now))
            logger.debug(
                "camera=%s worker=%s frame=%s timestamp=%.3f queue_size=%s detections=%s tracked=%s",
                packet.camera_id,
                packet.worker_id,
                packet.frame_number,
                packet.timestamp_seconds,
                ingestion_manager.frame_queue.qsize(),
                len(result.detections),
                len(result.tracked_detections),
            )
            if frames_by_camera[packet.camera_id] % 10 == 0:
                logger.info(
                    "%s processed_frames=%s detections=%s tracked=%s",
                    packet.camera_id,
                    frames_by_camera[packet.camera_id],
                    detections_by_camera[packet.camera_id],
                    tracked_observations_by_camera[packet.camera_id],
                )
            if _should_save_raw_frame(packet.frame_number, saved_raw_frames_by_camera[packet.camera_id], raw_frames_config):
                output_manager.save_raw_frame(
                    packet,
                    image_format=str(raw_frames_config["image_format"]),
                    jpeg_quality=int(raw_frames_config["jpeg_quality"]),
                )
                saved_raw_frames_by_camera[packet.camera_id] += 1
            if _should_save_visualization_frame(packet.frame_number, saved_detected_frames_by_camera[packet.camera_id], detected_frames_config):
                output_manager.save_detected_frame(packet.camera_id, packet.frame_number, result.detected_frame)
                saved_detected_frames_by_camera[packet.camera_id] += 1
            if _should_save_visualization_frame(packet.frame_number, saved_tracked_frames_by_camera[packet.camera_id], tracked_frames_config):
                output_manager.save_tracked_frame(packet.camera_id, packet.frame_number, result.tracked_frame)
                saved_tracked_frames_by_camera[packet.camera_id] += 1
            ingestion_manager.mark_task_done()

        for camera_id in frames_by_camera:
            completed_now = track_manager.flush_camera(camera_id)
            lifecycle_completed_tracks.extend(completed_now)
            finalized_evidence_now = evidence_collector.finalize_tracks(completed_now)
            enrichment_results.extend(vehicle_enrichment_manager.enrich_completed_tracks(completed_now, finalized_evidence_now))
            detector_tracker.reset_camera(camera_id)
        ingestion_manager.set_saved_raw_frames_by_camera(saved_raw_frames_by_camera)
        ingestion_manager.stop()
        metrics = ingestion_manager.get_metrics()
        lifecycle_metrics = track_manager.get_metrics()
        metadata.completed_tracks = len([track for track in track_manager.get_all_output_tracks() if track.status == "COMPLETED"])
        metadata.error_count = len(metrics["camera_errors"])
        metadata.status = RUN_STATUS_COMPLETED
        metadata.completed_at = datetime.now(timezone.utc).isoformat()
        output_manager.save_metadata(metadata)
        output_manager.save_ingestion_metrics(metrics)
        tracker_metrics = detector_tracker.metrics
        inference_times = tracker_metrics["inference_times_ms"]
        detection_tracking_metrics = {
            "detection_backend": validated_config["detection"]["backend"],
            "tracking_backend": validated_config["tracking"]["backend"],
            "model_path": validated_config["detection"]["model_path"],
            "configured_device": device_info.configured_device,
            "resolved_device": device_info.resolved_device,
            "cuda_available": device_info.cuda_available,
            "cuda_device_count": device_info.cuda_device_count,
            "cuda_device_name": device_info.cuda_device_name,
            "torch_version": device_info.torch_version,
            "torch_cuda_version": device_info.torch_cuda_version,
            "device": detector_tracker.device,
            "confidence_threshold": validated_config["detection"]["confidence_threshold"],
            "iou_threshold": validated_config["detection"]["iou_threshold"],
            "image_size": validated_config["detection"]["image_size"],
            "agnostic_nms": validated_config["detection"]["agnostic_nms"],
            "isolation_mode": validated_config["tracking"]["isolation_mode"],
            "processed_frames_by_camera": frames_by_camera,
            "detections_by_camera": detections_by_camera,
            "tracked_observations_by_camera": tracked_observations_by_camera,
            "detections_by_class": detections_by_class,
            "unique_native_track_ids_by_camera": {camera_id: sorted(values) for camera_id, values in unique_native_track_ids_by_camera.items()},
            "tracker_instance_count": tracker_metrics.get("tracker_instance_count", 0),
            "tracker_instances_created_total": tracker_metrics.get(
                "tracker_instances_created_total",
                tracker_metrics.get("tracker_instance_count", 0),
            ),
            "tracker_keys": tracker_metrics.get("tracker_keys", []),
            "trackers_created_by_camera": tracker_metrics.get("trackers_created_by_camera", {}),
            "trackers_created_by_camera_namespace": tracker_metrics.get("trackers_created_by_camera_namespace", {}),
            "saved_detected_frames_by_camera": saved_detected_frames_by_camera,
            "saved_tracked_frames_by_camera": saved_tracked_frames_by_camera,
            "inference_errors": tracker_metrics.get("inference_errors", []),
            "total_inference_time_ms": float(sum(inference_times)) if inference_times else 0.0,
            "average_inference_time_ms": float(sum(inference_times) / len(inference_times)) if inference_times else 0.0,
            "minimum_inference_time_ms": float(min(inference_times)) if inference_times else 0.0,
            "maximum_inference_time_ms": float(max(inference_times)) if inference_times else 0.0,
            "duration_seconds": float(metrics.get("duration_seconds", 0.0)),
        }
        output_manager.save_detection_tracking_metrics(detection_tracking_metrics)
        output_tracks = track_manager.get_all_output_tracks()
        output_observations = track_manager.get_all_observations()
        evidence_index_records = evidence_collector.evidence_index
        evidence_summary_by_track = _build_evidence_summary_by_track(evidence_index_records)
        enrichment_by_track = {item.local_track_id: item.to_dict() for item in enrichment_results}
        tracks_path = output_manager.save_tracks(
            [
                _serialize_track(
                    track,
                    evidence_summary_by_track,
                    enrichment_by_track if validated_config["vehicle_enrichment"]["extend_tracks_json"] else None,
                )
                for track in output_tracks
            ]
        )
        observations_path = output_manager.save_observations([_serialize_observation(item) for item in output_observations])
        lifecycle_metrics_path = output_manager.save_track_lifecycle_metrics(lifecycle_metrics)
        evidence_index_path = output_manager.save_evidence_index(evidence_index_records)
        evidence_metrics_path = output_manager.save_evidence_metrics(evidence_collector.metrics)
        vehicle_enrichment_path = output_manager.save_vehicle_enrichment([item.to_dict() for item in enrichment_results])
        vehicle_enrichment_metrics_path = output_manager.save_vehicle_enrichment_metrics(vehicle_enrichment_manager.metrics)
        output_manager.save_bbox_quality_metrics(
            {
                "raw_detections": raw_detections,
                "accepted_detections": accepted_detections,
                "rejected_detections": rejected_detections,
                "rejected_by_reason": rejected_by_reason,
                "rejected_by_class": rejected_by_class,
                "accepted_by_class": accepted_by_class,
                "detections": bbox_quality_diagnostics,
            }
        )
        output_manager.save_summary(
            {
                "run_id": metadata.run_id,
                "status": metadata.status,
                "project_name": metadata.project_name,
                "detection_backend": validated_config["detection"]["backend"],
                "tracking_backend": validated_config["tracking"]["backend"],
                "configured_device": device_info.configured_device,
                "resolved_device": device_info.resolved_device,
                "cuda_available": device_info.cuda_available,
                "cuda_device_count": device_info.cuda_device_count,
                "cuda_device_name": device_info.cuda_device_name,
                "torch_version": device_info.torch_version,
                "torch_cuda_version": device_info.torch_cuda_version,
                "configured_camera_count": metrics["configured_camera_count"],
                "enabled_camera_count": metrics["enabled_camera_count"],
                "worker_count": metrics["worker_count"],
                "processed_frames": metadata.processed_frames,
                "frames_by_camera": metrics["frames_by_camera"],
                "frames_by_worker": metrics["frames_by_worker"],
                "detections_by_camera": detections_by_camera,
                "tracked_observations_by_camera": tracked_observations_by_camera,
                "saved_raw_frames_by_camera": metrics["saved_raw_frames_by_camera"],
                "saved_detected_frames_by_camera": saved_detected_frames_by_camera,
                "saved_tracked_frames_by_camera": saved_tracked_frames_by_camera,
                "camera_errors": metrics["camera_errors"],
                "queue_full_events": metrics["queue_full_events"],
                "tracks_completed_by_camera": lifecycle_metrics["tracks_completed_by_camera"],
                "tracks_discarded_by_camera": lifecycle_metrics["tracks_discarded_by_camera"],
                "observations_by_camera": lifecycle_metrics["observations_by_camera"],
                "selected_evidence_records": len(evidence_index_records),
                "vehicle_enrichment_enabled": validated_config["vehicle_enrichment"]["enabled"],
                "vehicle_enrichment_result_count": len(enrichment_results),
                "run_directory": str(output_manager.run_directory),
            }
        )
        logger.info(
            "track output paths tracks=%s observations=%s lifecycle_metrics=%s evidence_index=%s evidence_metrics=%s vehicle_enrichment=%s vehicle_enrichment_metrics=%s",
            tracks_path,
            observations_path,
            lifecycle_metrics_path,
            evidence_index_path,
            evidence_metrics_path,
            vehicle_enrichment_path,
            vehicle_enrichment_metrics_path,
        )
        detector_tracker.reset_all()
        logger.info("Pipeline completed")
        return 0, output_manager.run_id, str(output_manager.run_directory)
    except Exception as exc:
        if ingestion_manager is not None:
            try:
                ingestion_manager.stop()
            except Exception:
                logger.exception("Ingestion shutdown failed during error handling")
        if track_manager is not None:
            try:
                track_manager.flush_all()
            except Exception:
                logger.exception("TrackManager shutdown failed during error handling")
        metadata.status = RUN_STATUS_FAILED
        metadata.error_count += 1
        metadata.completed_at = datetime.now(timezone.utc).isoformat()
        output_manager.save_metadata(metadata)
        logger.exception("Pipeline failed")
        output_manager.save_error(
            "pipeline_error",
            {
                "run_id": output_manager.run_id,
                "error_type": exc.__class__.__name__,
                "error_message": str(exc),
                "traceback": traceback.format_exc(),
            },
        )
        output_manager.save_summary(
            {
                "run_id": metadata.run_id,
                "status": metadata.status,
                "processed_frames": metadata.processed_frames,
                "error_count": metadata.error_count,
                "error_type": exc.__class__.__name__,
                "error_message": str(exc),
                "run_directory": str(output_manager.run_directory),
            }
        )
        return 1, output_manager.run_id, str(output_manager.run_directory)


def _serialize_track(
    track: LocalTrack,
    evidence_summary_by_track: dict[str, dict[str, Any]] | None = None,
    enrichment_by_track: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    summary = (evidence_summary_by_track or {}).get(track.local_track_id, {})
    payload = {
        "local_track_id": track.local_track_id,
        "camera_id": track.camera_id,
        "tracker_namespace": track.tracker_namespace,
        "native_tracker_id": track.native_tracker_id,
        "status": track.status,
        "first_frame": track.first_frame,
        "last_frame": track.last_frame,
        "first_timestamp_seconds": track.first_timestamp_seconds,
        "last_timestamp_seconds": track.last_timestamp_seconds,
        "observation_count": track.observation_count,
        "lost_frames": track.lost_frames,
        "final_class": track.final_class,
        "final_class_reason": track.final_class_reason,
        "class_counts": dict(track.class_counts),
        "class_confidence_sums": dict(track.class_confidence_sums),
        "completion_reason": track.completion_reason,
    }
    if summary:
        payload["evidence_record_count"] = int(summary.get("evidence_record_count", 0))
        payload["evidence_roles"] = list(summary.get("evidence_roles", []))
        payload["evidence_directory"] = summary.get("evidence_directory")
    else:
        payload["evidence_record_count"] = 0
        payload["evidence_roles"] = []
        payload["evidence_directory"] = None
    if enrichment_by_track is not None:
        payload["vehicle_enrichment"] = enrichment_by_track.get(track.local_track_id)
    return payload


def _serialize_observation(observation: TrackObservation) -> dict[str, Any]:
    x1, y1, x2, y2 = observation.bbox_xyxy
    return {
        "local_track_id": observation.local_track_id,
        "camera_id": observation.camera_id,
        "tracker_namespace": observation.tracker_namespace,
        "native_tracker_id": observation.native_tracker_id,
        "frame_number": observation.frame_number,
        "timestamp_seconds": observation.timestamp_seconds,
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
        "confidence": observation.confidence,
        "raw_class_id": observation.raw_class_id,
        "raw_class_name": observation.raw_class_name,
    }


def _build_evidence_summary_by_track(evidence_index_records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    summary_by_track: dict[str, dict[str, Any]] = {}
    for record in evidence_index_records:
        local_track_id = str(record["local_track_id"])
        summary = summary_by_track.setdefault(
            local_track_id,
            {
                "evidence_record_count": 0,
                "evidence_roles": [],
                "evidence_directory": None,
            },
        )
        summary["evidence_record_count"] += 1
        summary["evidence_roles"].append(str(record["role"]))
        if summary["evidence_directory"] is None:
            crop_path = record.get("crop_path")
            annotated_path = record.get("annotated_frame_path")
            source_path = crop_path or annotated_path
            if source_path:
                summary["evidence_directory"] = str(Path(str(source_path)).parent.parent)
    for summary in summary_by_track.values():
        summary["evidence_roles"] = sorted(set(summary["evidence_roles"]))
    return summary_by_track
