from __future__ import annotations

import json
import queue
import time
import traceback
from dataclasses import asdict
from datetime import datetime, timezone
import csv
from pathlib import Path
import shutil
from typing import Any

import numpy as np
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
from .runtime_state import get_runtime_state_manager
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


def _build_distribution_stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "max": float(array.max()),
    }


def _build_runtime_local_track_id(camera_id: str, tracker_namespace: str, native_tracker_id: int) -> str:
    normalized_namespace = str(tracker_namespace).strip()
    if normalized_namespace == "camera":
        return f"{camera_id}:TRACK_{native_tracker_id}"
    return f"{camera_id}:{normalized_namespace.upper()}:TRACK_{native_tracker_id}"


def _short_track_id(local_track_id: str) -> str:
    parts = str(local_track_id).split(":")
    return parts[-1] if parts else local_track_id


def _validate_capture_zone_profile(profile: dict[str, Any], context: str) -> dict[str, Any]:
    normalized = {
        "top_ratio": float(profile.get("top_ratio", 0.55)),
        "bottom_ratio": float(profile.get("bottom_ratio", 0.72)),
        "trigger_point": str(profile.get("trigger_point", "bottom_center")).strip() or "bottom_center",
        "maximum_saved_candidates_per_track": int(profile.get("maximum_saved_candidates_per_track", 3)),
        "minimum_frame_gap": int(profile.get("minimum_frame_gap", 2)),
        "capture_policy": str(profile.get("capture_policy", "best_quality")).strip() or "best_quality",
        "save_immediately": bool(profile.get("save_immediately", True)),
        "require_confirmed_track": bool(profile.get("require_confirmed_track", True)),
        "minimum_bbox_width_pixels": int(profile.get("minimum_bbox_width_pixels", 40)),
        "minimum_bbox_height_pixels": int(profile.get("minimum_bbox_height_pixels", 40)),
        "direction_mode": str(profile.get("direction_mode", "any")).strip() or "any",
    }
    if normalized["trigger_point"] != "bottom_center":
        raise ConfigurationError(f"{context}.trigger_point must be bottom_center.")
    if not 0.0 <= normalized["top_ratio"] < normalized["bottom_ratio"] <= 1.0:
        raise ConfigurationError(f"{context} ratios must satisfy 0.0 <= top_ratio < bottom_ratio <= 1.0.")
    if normalized["maximum_saved_candidates_per_track"] < 1:
        raise ConfigurationError(f"{context}.maximum_saved_candidates_per_track must be at least 1.")
    if normalized["minimum_frame_gap"] < 0:
        raise ConfigurationError(f"{context}.minimum_frame_gap must be at least 0.")
    if normalized["minimum_bbox_width_pixels"] < 1 or normalized["minimum_bbox_height_pixels"] < 1:
        raise ConfigurationError(f"{context} minimum bbox dimensions must be at least 1.")
    return normalized


def _validate_config(raw_config: dict[str, Any], config_path: Path) -> dict[str, Any]:
    project = raw_config.get("project")
    input_section = raw_config.get("input")
    ingestion = raw_config.get("ingestion")
    detection = raw_config.get("detection")
    tracking = raw_config.get("tracking")
    evidence = raw_config.get("evidence")
    visualization = raw_config.get("visualization")
    output_section = raw_config.get("output")
    debug_outputs = raw_config.get("debug_outputs")
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
    if debug_outputs is not None and not isinstance(debug_outputs, dict):
        raise ConfigurationError("Invalid 'debug_outputs' section.")
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

    _missing = object()
    raw_max_frames_per_camera = input_section.get("max_frames_per_camera", _missing)
    if raw_max_frames_per_camera is None:
        max_frames_per_camera = None
    else:
        default_value = 0 if raw_max_frames_per_camera is _missing else raw_max_frames_per_camera
        try:
            max_frames_per_camera = int(default_value)
        except (TypeError, ValueError) as exc:
            raise ConfigurationError("input.max_frames_per_camera must be an integer or null.") from exc
        if max_frames_per_camera <= 0:
            raise ConfigurationError("input.max_frames_per_camera must be a positive integer or null.")

    worker_count = int(ingestion.get("worker_count", 7))
    target_read_fps = ingestion.get("target_read_fps", 10.0)
    frame_queue_size = int(ingestion.get("frame_queue_size", 200))
    per_camera_buffer_size = int(ingestion.get("per_camera_buffer_size", 2))
    scheduler_policy = str(ingestion.get("scheduler_policy", "round_robin")).strip().lower() or "round_robin"
    if worker_count < 1:
        raise ConfigurationError("ingestion.worker_count must be at least 1.")
    if frame_queue_size < 1:
        raise ConfigurationError("ingestion.frame_queue_size must be at least 1.")
    if per_camera_buffer_size < 1:
        raise ConfigurationError("ingestion.per_camera_buffer_size must be at least 1.")
    if scheduler_policy != "round_robin":
        raise ConfigurationError("ingestion.scheduler_policy must be round_robin.")
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
    detection_batch = dict(detection.get("batch", {}) or {})
    detection_batch_enabled = bool(detection_batch.get("enabled", False))
    detection_batch_max_size = int(detection_batch.get("max_size", 1) or 1)
    detection_batch_max_wait_ms = float(detection_batch.get("max_wait_ms", 0.0) or 0.0)
    if detection_batch_max_size < 1:
        raise ConfigurationError("detection.batch.max_size must be at least 1.")
    if detection_batch_max_wait_ms < 0.0:
        raise ConfigurationError("detection.batch.max_wait_ms must be at least 0.")
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
    capture_zone_section = dict(evidence_section.get("capture_zone", {}) or {})
    capture_zone_cameras = dict(capture_zone_section.get("cameras", {}) or {})
    capture_zone_enabled = bool(capture_zone_section.get("enabled", False))
    capture_zone_default = {
        **dict(capture_zone_section.get("default", {}) or {}),
        "trigger_point": capture_zone_section.get("trigger_point", dict(capture_zone_section.get("default", {}) or {}).get("trigger_point", "bottom_center")),
        "maximum_saved_candidates_per_track": capture_zone_section.get("maximum_saved_candidates_per_track", dict(capture_zone_section.get("default", {}) or {}).get("maximum_saved_candidates_per_track", 3)),
        "minimum_frame_gap": capture_zone_section.get("minimum_frame_gap", dict(capture_zone_section.get("default", {}) or {}).get("minimum_frame_gap", 2)),
        "capture_policy": capture_zone_section.get("capture_policy", dict(capture_zone_section.get("default", {}) or {}).get("capture_policy", "best_quality")),
        "save_immediately": capture_zone_section.get("save_immediately", dict(capture_zone_section.get("default", {}) or {}).get("save_immediately", True)),
        "require_confirmed_track": capture_zone_section.get("require_confirmed_track", dict(capture_zone_section.get("default", {}) or {}).get("require_confirmed_track", True)),
        "minimum_bbox_width_pixels": capture_zone_section.get("minimum_bbox_width_pixels", dict(capture_zone_section.get("default", {}) or {}).get("minimum_bbox_width_pixels", evidence_section.get("minimum_crop_width_pixels", 40))),
        "minimum_bbox_height_pixels": capture_zone_section.get("minimum_bbox_height_pixels", dict(capture_zone_section.get("default", {}) or {}).get("minimum_bbox_height_pixels", evidence_section.get("minimum_crop_height_pixels", 40))),
        "direction_mode": capture_zone_section.get("direction_mode", dict(capture_zone_section.get("default", {}) or {}).get("direction_mode", "any")),
    }
    if "top_ratio" in capture_zone_section or "bottom_ratio" in capture_zone_section:
        capture_zone_default = {
            **capture_zone_default,
            "top_ratio": capture_zone_section.get("top_ratio", capture_zone_default.get("top_ratio", 0.55)),
            "bottom_ratio": capture_zone_section.get("bottom_ratio", capture_zone_default.get("bottom_ratio", 0.72)),
        }
    capture_zone_default.setdefault("trigger_point", capture_zone_section.get("trigger_point", "bottom_center"))
    capture_zone_default.setdefault("maximum_saved_candidates_per_track", capture_zone_section.get("maximum_saved_candidates_per_track", 3))
    capture_zone_default.setdefault("minimum_frame_gap", capture_zone_section.get("minimum_frame_gap", 2))
    capture_zone_default.setdefault("capture_policy", capture_zone_section.get("capture_policy", "best_quality"))
    capture_zone_default.setdefault("save_immediately", capture_zone_section.get("save_immediately", True))
    capture_zone_default.setdefault("require_confirmed_track", capture_zone_section.get("require_confirmed_track", True))
    capture_zone_default.setdefault("minimum_bbox_width_pixels", capture_zone_section.get("minimum_bbox_width_pixels", evidence_section.get("minimum_crop_width_pixels", 40)))
    capture_zone_default.setdefault("minimum_bbox_height_pixels", capture_zone_section.get("minimum_bbox_height_pixels", evidence_section.get("minimum_crop_height_pixels", 40)))
    capture_zone_default.setdefault("direction_mode", capture_zone_section.get("direction_mode", "any"))
    normalized_capture_zone_default = _validate_capture_zone_profile(capture_zone_default, "evidence.capture_zone.default")
    normalized_capture_zone_class_specific: dict[str, dict[str, Any]] = {}
    for class_name, payload in dict(capture_zone_section.get("class_specific", {}) or {}).items():
        if not isinstance(payload, dict):
            raise ConfigurationError(f"evidence.capture_zone.class_specific.{class_name} must be a mapping.")
        normalized_capture_zone_class_specific[str(class_name).strip().lower()] = _validate_capture_zone_profile(
            {**normalized_capture_zone_default, **dict(payload)},
            f"evidence.capture_zone.class_specific.{class_name}",
        )
    normalized_capture_zone_cameras: dict[str, dict[str, Any]] = {}
    for camera_id, payload in capture_zone_cameras.items():
        if not isinstance(payload, dict):
            raise ConfigurationError(f"evidence.capture_zone.cameras.{camera_id} must be a mapping.")
        payload = dict(payload)
        camera_default = {
            **dict(payload.get("default", {}) or {}),
            "trigger_point": payload.get("trigger_point", dict(payload.get("default", {}) or {}).get("trigger_point", normalized_capture_zone_default["trigger_point"])),
            "maximum_saved_candidates_per_track": payload.get("maximum_saved_candidates_per_track", dict(payload.get("default", {}) or {}).get("maximum_saved_candidates_per_track", normalized_capture_zone_default["maximum_saved_candidates_per_track"])),
            "minimum_frame_gap": payload.get("minimum_frame_gap", dict(payload.get("default", {}) or {}).get("minimum_frame_gap", normalized_capture_zone_default["minimum_frame_gap"])),
            "capture_policy": payload.get("capture_policy", dict(payload.get("default", {}) or {}).get("capture_policy", normalized_capture_zone_default["capture_policy"])),
            "save_immediately": payload.get("save_immediately", dict(payload.get("default", {}) or {}).get("save_immediately", normalized_capture_zone_default["save_immediately"])),
            "require_confirmed_track": payload.get("require_confirmed_track", dict(payload.get("default", {}) or {}).get("require_confirmed_track", normalized_capture_zone_default["require_confirmed_track"])),
            "minimum_bbox_width_pixels": payload.get("minimum_bbox_width_pixels", dict(payload.get("default", {}) or {}).get("minimum_bbox_width_pixels", normalized_capture_zone_default["minimum_bbox_width_pixels"])),
            "minimum_bbox_height_pixels": payload.get("minimum_bbox_height_pixels", dict(payload.get("default", {}) or {}).get("minimum_bbox_height_pixels", normalized_capture_zone_default["minimum_bbox_height_pixels"])),
            "direction_mode": payload.get("direction_mode", dict(payload.get("default", {}) or {}).get("direction_mode", normalized_capture_zone_default["direction_mode"])),
        }
        if "top_ratio" in payload or "bottom_ratio" in payload:
            camera_default = {
                **camera_default,
                "top_ratio": payload.get("top_ratio", camera_default.get("top_ratio", normalized_capture_zone_default["top_ratio"])),
                "bottom_ratio": payload.get("bottom_ratio", camera_default.get("bottom_ratio", normalized_capture_zone_default["bottom_ratio"])),
            }
        normalized_camera_default = _validate_capture_zone_profile(
            {**normalized_capture_zone_default, **camera_default},
            f"evidence.capture_zone.cameras.{camera_id}.default",
        )
        normalized_camera_class_specific: dict[str, dict[str, Any]] = {}
        for class_name, class_payload in dict(payload.get("class_specific", {}) or {}).items():
            if not isinstance(class_payload, dict):
                raise ConfigurationError(f"evidence.capture_zone.cameras.{camera_id}.class_specific.{class_name} must be a mapping.")
            class_key = str(class_name).strip().lower()
            fallback_profile = normalized_capture_zone_class_specific.get(class_key, normalized_camera_default)
            normalized_camera_class_specific[class_key] = _validate_capture_zone_profile(
                {**fallback_profile, **dict(class_payload)},
                f"evidence.capture_zone.cameras.{camera_id}.class_specific.{class_name}",
            )
        normalized_capture_zone_cameras[str(camera_id).strip()] = {
            "enabled": bool(payload.get("enabled", capture_zone_enabled)),
            "default": normalized_camera_default,
            "class_specific": normalized_camera_class_specific,
        }
    visualization_capture_zone = dict(visualization.get("capture_zone", {}) or {})
    debug_outputs_section = dict(debug_outputs or {})
    def _normalize_debug_output_item(name: str, *, default_enabled: bool = False, default_every_n: int = 1, default_max_frames: int = 0, default_max_crops: int = 0) -> dict[str, Any]:
        payload = dict(debug_outputs_section.get(name, {}) or {})
        save_every_n_frames = int(payload.get("save_every_n_frames", default_every_n))
        max_saved_frames_per_camera = int(payload.get("max_saved_frames_per_camera", default_max_frames))
        max_crops_per_track = int(payload.get("max_crops_per_track", default_max_crops))
        if save_every_n_frames < 1:
            raise ConfigurationError(f"debug_outputs.{name}.save_every_n_frames must be at least 1.")
        if max_saved_frames_per_camera < 0:
            raise ConfigurationError(f"debug_outputs.{name}.max_saved_frames_per_camera must be at least 0.")
        if max_crops_per_track < 0:
            raise ConfigurationError(f"debug_outputs.{name}.max_crops_per_track must be at least 0.")
        return {
            "enabled": bool(payload.get("enabled", default_enabled)),
            "save_every_n_frames": save_every_n_frames,
            "max_saved_frames_per_camera": max_saved_frames_per_camera,
            "max_crops_per_track": max_crops_per_track,
        }
    normalized_debug_outputs = {
        "enabled": bool(debug_outputs_section.get("enabled", False)),
        "extracted_frames": _normalize_debug_output_item("extracted_frames"),
        "detected_frames": _normalize_debug_output_item("detected_frames"),
        "tracked_frames": _normalize_debug_output_item("tracked_frames"),
        "track_crops": _normalize_debug_output_item("track_crops", default_every_n=3, default_max_crops=100),
        "florence_selected_crops": {
            "enabled": bool(dict(debug_outputs_section.get("florence_selected_crops", {}) or {}).get("enabled", False)),
        },
    }
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
            "per_camera_buffer_size": per_camera_buffer_size,
            "scheduler_policy": scheduler_policy,
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
            "dtype": str(detection.get("dtype", "auto")),
            "confidence_threshold": float(detection.get("confidence_threshold", 0.2 if detection_backend == "ocr_mukul" else 0.38)),
            "iou_threshold": float(detection.get("iou_threshold", 0.45)),
            "image_size": int(detection.get("image_size", 1024 if detection_backend == "ocr_mukul" else 640)),
            "agnostic_nms": bool(detection.get("agnostic_nms", False)),
            "batch": {
                "enabled": detection_batch_enabled,
                "max_size": detection_batch_max_size,
                "max_wait_ms": detection_batch_max_wait_ms,
            },
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
            "capture_zone": {
                "enabled": capture_zone_enabled,
                "default": normalized_capture_zone_default,
                "class_specific": normalized_capture_zone_class_specific,
                "top_ratio": normalized_capture_zone_default["top_ratio"],
                "bottom_ratio": normalized_capture_zone_default["bottom_ratio"],
                "trigger_point": normalized_capture_zone_default["trigger_point"],
                "maximum_saved_candidates_per_track": normalized_capture_zone_default["maximum_saved_candidates_per_track"],
                "minimum_frame_gap": normalized_capture_zone_default["minimum_frame_gap"],
                "capture_policy": normalized_capture_zone_default["capture_policy"],
                "save_immediately": normalized_capture_zone_default["save_immediately"],
                "require_confirmed_track": normalized_capture_zone_default["require_confirmed_track"],
                "minimum_bbox_width_pixels": normalized_capture_zone_default["minimum_bbox_width_pixels"],
                "minimum_bbox_height_pixels": normalized_capture_zone_default["minimum_bbox_height_pixels"],
                "direction_mode": normalized_capture_zone_default["direction_mode"],
                "cameras": normalized_capture_zone_cameras,
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
            "capture_zone": {
                "enabled": bool(visualization_capture_zone.get("enabled", False)),
            },
        },
        "output": {
            "root_directory": str(output_root),
            "save_run_config": bool(output_section.get("save_run_config", True)),
        },
        "debug_outputs": normalized_debug_outputs,
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


def _capture_zone_enabled_for_camera(config: dict[str, Any], camera_id: str) -> bool:
    capture_zone = dict(dict(config.get("evidence", {}) or {}).get("capture_zone", {}) or {})
    enabled = bool(capture_zone.get("enabled", False))
    overrides = dict(capture_zone.get("cameras", {}) or {}).get(camera_id)
    if isinstance(overrides, dict) and "enabled" in overrides:
        enabled = bool(overrides.get("enabled"))
    return enabled


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
    runtime_state = get_runtime_state_manager()
    try:
        logger.info("Pipeline started")
        if deferred_setup_error is not None:
            raise deferred_setup_error
        validated_config = _validate_config(raw_config or _load_raw_config(config_file), config_file)
        output_manager.configure_debug_outputs(validated_config.get("debug_outputs", {}))
        for camera in validated_config["input"]["cameras"]:
            if camera["enabled"]:
                logger.info(
                    "Resolved input source camera_id=%s source_type=%s source=%s",
                    camera["camera_id"],
                    camera["source_type"],
                    camera["source"],
                )
                if _capture_zone_enabled_for_camera(validated_config, camera["camera_id"]):
                    camera_capture_zone = dict(validated_config["evidence"]["capture_zone"])
                    overrides = dict(camera_capture_zone.get("cameras", {}) or {}).get(camera["camera_id"])
                    camera_default_zone = dict(camera_capture_zone.get("default", {}) or {})
                    if isinstance(overrides, dict):
                        camera_default_zone = dict(overrides.get("default", camera_default_zone) or camera_default_zone)
                    logger.info(
                        "Evidence zone configured camera=%s top_ratio=%.2f bottom_ratio=%.2f",
                        camera["camera_id"],
                        float(camera_default_zone.get("top_ratio", camera_capture_zone.get("top_ratio", 0.0))),
                        float(camera_default_zone.get("bottom_ratio", camera_capture_zone.get("bottom_ratio", 0.0))),
                    )
        metadata.project_name = validated_config["project"]["name"]
        metadata.camera_count = len([camera for camera in validated_config["input"]["cameras"] if camera["enabled"]])
        if bool(validated_config["output"]["save_run_config"]):
            output_manager.save_effective_config(validated_config)
        logger.info("Config loaded")
        runtime_state.initialize_run(
            run_id=output_manager.run_id,
            run_directory=str(output_manager.run_directory),
            cameras=[camera for camera in validated_config["input"]["cameras"] if camera["enabled"]],
        )
        frame_limit = validated_config["input"]["max_frames_per_camera"]
        logger.info(
            "Frame limit per camera: %s",
            "unlimited" if frame_limit is None else frame_limit,
        )
        logger.info(
            "Vehicle enrichment startup: colour_enabled=%s body_type_enabled=%s adapter_enabled=%s plate_ocr_enabled=%s",
            bool(validated_config["vehicle_enrichment"].get("enrichment", {}).get("colour", {}).get("enabled", False)),
            bool(validated_config["vehicle_enrichment"].get("enrichment", {}).get("body_type", {}).get("enabled", True)),
            bool(validated_config["vehicle_enrichment"].get("florence", {}).get("adapter", {}).get("enabled", False)),
            bool(validated_config["vehicle_enrichment"].get("enrichment", {}).get("plate", {}).get("ocr", {}).get("enabled", False)),
        )
        logger.info(
            "Async colour enrichment: enabled=%s worker_count=%s queue_count=%s queue_size=%s",
            bool(validated_config["vehicle_enrichment"].get("enrichment", {}).get("colour", {}).get("async", {}).get("enabled", False)),
            int(validated_config["vehicle_enrichment"].get("enrichment", {}).get("colour", {}).get("async", {}).get("worker_count", 0)),
            1 if bool(validated_config["vehicle_enrichment"].get("enrichment", {}).get("colour", {}).get("async", {}).get("enabled", False)) else 0,
            int(validated_config["vehicle_enrichment"].get("enrichment", {}).get("colour", {}).get("async", {}).get("queue_size", 0)),
        )
        logger.info(
            "Detection batching: enabled=%s max_size=%s max_wait_ms=%s",
            bool(validated_config["detection"].get("batch", {}).get("enabled", False)),
            int(validated_config["detection"].get("batch", {}).get("max_size", 1) or 1),
            float(validated_config["detection"].get("batch", {}).get("max_wait_ms", 0.0) or 0.0),
        )
        runtime_state.update_system_status(
            yolo_status="loaded",
            colour_worker_status="running" if bool(validated_config["vehicle_enrichment"].get("enrichment", {}).get("colour", {}).get("async", {}).get("enabled", False)) else "disabled",
            colour_queue_capacity=int(validated_config["vehicle_enrichment"].get("enrichment", {}).get("colour", {}).get("async", {}).get("queue_size", 0) or 0),
            pending_colour_jobs=0,
            frame_loss=0,
            order_violations=0,
        )
        metadata.status = RUN_STATUS_RUNNING
        output_manager.save_metadata(metadata)

        ingestion_manager = MultiCameraIngestionManager(validated_config, logger)
        detector_tracker = VehicleDetectorTracker(validated_config, logger)
        device_info = detector_tracker.runtime_device_info
        metadata.configured_device = device_info.configured_device
        metadata.configured_dtype = device_info.configured_dtype
        metadata.resolved_device = device_info.resolved_device
        metadata.resolved_dtype = device_info.resolved_dtype
        metadata.cuda_available = device_info.cuda_available
        metadata.cuda_device_count = device_info.cuda_device_count
        metadata.cuda_device_name = device_info.cuda_device_name
        metadata.torch_version = device_info.torch_version
        metadata.torch_cuda_version = device_info.torch_cuda_version
        output_manager.save_metadata(metadata)
        logger.info(
            "Runtime device: configured_device=%s configured_dtype=%s resolved_device=%s resolved_dtype=%s cuda_available=%s cuda_device_name=%s torch_version=%s torch_cuda_version=%s reason=%s",
            device_info.configured_device,
            device_info.configured_dtype,
            device_info.resolved_device,
            device_info.resolved_dtype,
            device_info.cuda_available,
            device_info.cuda_device_name,
            device_info.torch_version,
            device_info.torch_cuda_version,
            device_info.reason,
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
        detection_batch_config = dict(validated_config["detection"].get("batch", {}) or {})
        detection_batch_enabled = bool(detection_batch_config.get("enabled", False))
        detection_batch_max_size = int(detection_batch_config.get("max_size", 1) or 1)
        detection_batch_max_wait_ms = float(detection_batch_config.get("max_wait_ms", 0.0) or 0.0)
        detection_batch_wait_times_ms: list[float] = []
        detection_latency_ms: list[float] = []
        frame_order_violations = 0
        last_processed_frame_by_camera: dict[str, int] = {}

        while True:
            try:
                first_packet = ingestion_manager.get_packet()
            except queue.Empty:
                if ingestion_manager.is_finished() and ingestion_manager.frame_queue.empty():
                    break
                continue

            batch_packets: list[Any] = [first_packet]
            batch_collected_at: list[float] = [time.perf_counter()]
            batch_started_at = batch_collected_at[0]
            if detection_batch_enabled and detection_batch_max_size > 1:
                batch_deadline = batch_started_at + (detection_batch_max_wait_ms / 1000.0)
                while len(batch_packets) < detection_batch_max_size:
                    remaining = batch_deadline - time.perf_counter()
                    if detection_batch_max_wait_ms <= 0.0:
                        remaining = 0.0
                    if remaining < 0.0:
                        break
                    try:
                        next_packet = ingestion_manager.get_packet(timeout=remaining)
                    except queue.Empty:
                        break
                    batch_packets.append(next_packet)
                    batch_collected_at.append(time.perf_counter())
            detection_batch_wait_times_ms.append((time.perf_counter() - batch_started_at) * 1000.0)
            batch_results = detector_tracker.process_frames(batch_packets)
            batch_completed_at = time.perf_counter()

            for packet, result, collected_at in zip(batch_packets, batch_results, batch_collected_at):
                if packet.frame_number <= last_processed_frame_by_camera.get(packet.camera_id, -1):
                    frame_order_violations += 1
                last_processed_frame_by_camera[packet.camera_id] = packet.frame_number
                detection_latency_ms.append((batch_completed_at - collected_at) * 1000.0)
                frames_by_camera[packet.camera_id] += 1
                metadata.processed_frames += 1
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
                runtime_detection_rows: list[dict[str, Any]] = []
                for tracked_item in result.tracked_detections:
                    local_track_id = _build_runtime_local_track_id(packet.camera_id, tracked_item.tracker_namespace, int(tracked_item.tracker_id))
                    short_track_id = _short_track_id(local_track_id)
                    colour_value = None
                    colour_status = "pending"
                    runtime_detection_rows.append(
                        {
                            "track_id": short_track_id,
                            "local_track_id": local_track_id,
                            "vehicle_class": str(tracked_item.raw_class_name),
                            "bbox": [float(value) for value in tracked_item.bbox_xyxy],
                            "confidence": float(tracked_item.confidence),
                            "colour": colour_value,
                            "colour_status": colour_status,
                        }
                    )
                    runtime_state.update_track_runtime(
                        camera_id=packet.camera_id,
                        local_track_id=local_track_id,
                        short_track_id=short_track_id,
                        vehicle_class=str(tracked_item.raw_class_name),
                        bbox=list(tracked_item.bbox_xyxy),
                        confidence=float(tracked_item.confidence),
                        timestamp_seconds=float(tracked_item.timestamp_seconds),
                        frame_number=int(tracked_item.frame_number),
                        colour=colour_value,
                        colour_status=colour_status,
                        status="active",
                    )
                runtime_state.update_camera_runtime(
                    camera_id=packet.camera_id,
                    frame_number=int(packet.frame_number),
                    timestamp_seconds=float(packet.timestamp_seconds),
                    input_fps=float(packet.source_fps),
                    detections=runtime_detection_rows,
                    active_track_ids=[str(item["track_id"]) for item in runtime_detection_rows],
                    active_vehicle_count=len(runtime_detection_rows),
                    frame_bgr=result.tracked_frame,
                    status="processing",
                )
                if completed_now:
                    latest_enrichment_by_track = {
                        str(item.local_track_id): item
                        for item in enrichment_results[-len(completed_now):]
                        if getattr(item, "local_track_id", None)
                    }
                    for completed_track in completed_now:
                        local_track_id = str(completed_track.local_track_id)
                        short_track_id = _short_track_id(local_track_id)
                        enrichment_item = latest_enrichment_by_track.get(local_track_id)
                        colour_label = None
                        colour_status = "pending"
                        evidence_rows = None
                        if enrichment_item is not None:
                            colour_payload = getattr(enrichment_item, "vehicle_colour", None)
                            colour_label = None if colour_payload is None else getattr(colour_payload, "label", None)
                            colour_status = "pending" if colour_payload is None else str(getattr(colour_payload, "status", "pending"))
                            evidence_rows = list(getattr(enrichment_item, "evidence_used", []) or [])
                        runtime_state.update_track_runtime(
                            camera_id=str(completed_track.camera_id),
                            local_track_id=local_track_id,
                            short_track_id=short_track_id,
                            vehicle_class=str(completed_track.final_class),
                            bbox=list(completed_track.observations[-1].bbox_xyxy) if completed_track.observations else None,
                            confidence=float(completed_track.observations[-1].confidence) if completed_track.observations else None,
                            timestamp_seconds=float(completed_track.last_timestamp_seconds),
                            frame_number=int(completed_track.last_frame),
                            colour=colour_label,
                            colour_status=colour_status,
                            status="completed",
                            evidence=evidence_rows,
                        )
                logger.debug(
                    "camera=%s worker=%s frame=%s timestamp=%.3f queue_size=%s detections=%s tracked=%s batch_size=%s",
                    packet.camera_id,
                    packet.worker_id,
                    packet.frame_number,
                    packet.timestamp_seconds,
                    ingestion_manager.frame_queue.qsize(),
                    len(result.detections),
                    len(result.tracked_detections),
                    len(batch_packets),
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
                runtime_state.update_system_status(
                    colour_queue_depth=int(vehicle_enrichment_manager.metrics.get("colour_queue_peak_depth", 0) or 0),
                    pending_colour_jobs=int(vehicle_enrichment_manager.metrics.get("track_evidence_pending_count", 0) or 0),
                    cache_misses=int(evidence_collector.metrics.get("evidence_cache_misses", 0) or 0),
                    frame_loss=0,
                    order_violations=int(frame_order_violations),
                )
                ingestion_manager.mark_task_done()

        for camera_id in frames_by_camera:
            completed_now = track_manager.flush_camera(camera_id)
            lifecycle_completed_tracks.extend(completed_now)
            finalized_evidence_now = evidence_collector.finalize_tracks(completed_now)
            enrichment_results.extend(vehicle_enrichment_manager.enrich_completed_tracks(completed_now, finalized_evidence_now))
            detector_tracker.reset_camera(camera_id)
        enrichment_results.extend(vehicle_enrichment_manager.finalize_async_colour())
        for enrichment_item in enrichment_results:
            local_track_id = str(getattr(enrichment_item, "local_track_id", ""))
            if not local_track_id:
                continue
            camera_id = str(getattr(enrichment_item, "camera_id", ""))
            short_track_id = _short_track_id(local_track_id)
            colour_payload = getattr(enrichment_item, "vehicle_colour", None)
            runtime_state.update_track_colour(
                camera_id=camera_id,
                local_track_id=local_track_id,
                short_track_id=short_track_id,
                colour=None if colour_payload is None else getattr(colour_payload, "label", None),
                colour_status="pending" if colour_payload is None else str(getattr(colour_payload, "status", "pending")),
            )
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
        preprocess_times = list(tracker_metrics.get("preprocess_times_ms", []) or [])
        model_inference_stage_times = list(tracker_metrics.get("model_inference_stage_times_ms", []) or [])
        postprocess_times = list(tracker_metrics.get("postprocess_times_ms", []) or [])
        result_conversion_times = list(tracker_metrics.get("result_conversion_times_ms", []) or [])
        result_routing_times = list(tracker_metrics.get("result_routing_times_ms", []) or [])
        tracker_update_times = list(tracker_metrics.get("tracker_update_times_ms", []) or [])
        total_detection_times = list(tracker_metrics.get("total_detection_times_ms", []) or [])
        total_detection_stage_time_ms = float(sum(total_detection_times)) if total_detection_times else 0.0
        preprocess_stage_time_ms = float(sum(preprocess_times)) if preprocess_times else 0.0
        model_inference_stage_time_ms = float(sum(model_inference_stage_times)) if model_inference_stage_times else 0.0
        postprocess_stage_time_ms = float(sum(postprocess_times)) if postprocess_times else 0.0
        result_conversion_stage_time_ms = float(sum(result_conversion_times)) if result_conversion_times else 0.0
        result_routing_stage_time_ms = float(sum(result_routing_times)) if result_routing_times else 0.0
        tracker_update_stage_time_ms = float(sum(tracker_update_times)) if tracker_update_times else 0.0
        detection_tracking_metrics = {
            "detection_backend": validated_config["detection"]["backend"],
            "tracking_backend": validated_config["tracking"]["backend"],
            "model_path": validated_config["detection"]["model_path"],
            "configured_device": device_info.configured_device,
            "configured_dtype": device_info.configured_dtype,
            "resolved_device": device_info.resolved_device,
            "resolved_dtype": device_info.resolved_dtype,
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
            "detection_batch_enabled": detection_batch_enabled,
            "detection_batch_size_configured": detection_batch_max_size if detection_batch_enabled else 1,
            "detection_batch_wait_ms_configured": detection_batch_max_wait_ms if detection_batch_enabled else 0.0,
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
            "preprocess_stage_profile_ms": _build_distribution_stats(preprocess_times),
            "model_inference_stage_profile_ms": _build_distribution_stats(model_inference_stage_times),
            "postprocess_stage_profile_ms": _build_distribution_stats(postprocess_times),
            "result_conversion_stage_profile_ms": _build_distribution_stats(result_conversion_times),
            "result_routing_stage_profile_ms": _build_distribution_stats(result_routing_times),
            "tracker_update_stage_profile_ms": _build_distribution_stats(tracker_update_times),
            "total_detection_stage_profile_ms": _build_distribution_stats(total_detection_times),
            "preprocess_stage_total_ms": preprocess_stage_time_ms,
            "model_inference_stage_total_ms": model_inference_stage_time_ms,
            "postprocess_stage_total_ms": postprocess_stage_time_ms,
            "result_conversion_stage_total_ms": result_conversion_stage_time_ms,
            "result_routing_stage_total_ms": result_routing_stage_time_ms,
            "tracker_update_stage_total_ms": tracker_update_stage_time_ms,
            "total_detection_stage_total_ms": total_detection_stage_time_ms,
            "preprocess_stage_share_of_total_percent": float((preprocess_stage_time_ms * 100.0) / total_detection_stage_time_ms) if total_detection_stage_time_ms > 0.0 else 0.0,
            "model_inference_stage_share_of_total_percent": float((model_inference_stage_time_ms * 100.0) / total_detection_stage_time_ms) if total_detection_stage_time_ms > 0.0 else 0.0,
            "postprocess_stage_share_of_total_percent": float((postprocess_stage_time_ms * 100.0) / total_detection_stage_time_ms) if total_detection_stage_time_ms > 0.0 else 0.0,
            "result_conversion_stage_share_of_total_percent": float((result_conversion_stage_time_ms * 100.0) / total_detection_stage_time_ms) if total_detection_stage_time_ms > 0.0 else 0.0,
            "result_routing_stage_share_of_total_percent": float((result_routing_stage_time_ms * 100.0) / total_detection_stage_time_ms) if total_detection_stage_time_ms > 0.0 else 0.0,
            "tracker_update_stage_share_of_total_percent": float((tracker_update_stage_time_ms * 100.0) / total_detection_stage_time_ms) if total_detection_stage_time_ms > 0.0 else 0.0,
            "detection_batches_total": int(tracker_metrics.get("detection_batches_total", 0) or 0),
            "detection_frames_total": int(tracker_metrics.get("detection_frames_total", 0) or 0),
            "average_detection_batch_size": float(
                (tracker_metrics.get("detection_batch_size_sum", 0) or 0) / float(tracker_metrics.get("detection_batches_total", 0) or 1)
            ) if int(tracker_metrics.get("detection_batches_total", 0) or 0) > 0 else 0.0,
            "max_detection_batch_size_observed": int(tracker_metrics.get("max_detection_batch_size_observed", 0) or 0),
            "partial_detection_batches": int(tracker_metrics.get("partial_detection_batches", 0) or 0),
            "detection_batch_wait_time_ms_avg": float(sum(detection_batch_wait_times_ms) / len(detection_batch_wait_times_ms)) if detection_batch_wait_times_ms else 0.0,
            "detection_batch_wait_time_ms_max": float(max(detection_batch_wait_times_ms)) if detection_batch_wait_times_ms else 0.0,
            "yolo_model_invocations": int(tracker_metrics.get("yolo_model_invocations", 0) or 0),
            "yolo_frames_processed": int(tracker_metrics.get("yolo_frames_processed", 0) or 0),
            "yolo_inference_time_total_ms": float(tracker_metrics.get("yolo_inference_time_total_ms", 0.0) or 0.0),
            "yolo_inference_time_per_batch_avg_ms": float(
                sum(tracker_metrics.get("yolo_inference_time_per_batch_ms", []) or [])
                / len(tracker_metrics.get("yolo_inference_time_per_batch_ms", []) or [1])
            ) if list(tracker_metrics.get("yolo_inference_time_per_batch_ms", []) or []) else 0.0,
            "yolo_inference_time_per_frame_avg_ms": float(
                (tracker_metrics.get("yolo_inference_time_total_ms", 0.0) or 0.0)
                / float(tracker_metrics.get("yolo_frames_processed", 0) or 1)
            ) if int(tracker_metrics.get("yolo_frames_processed", 0) or 0) > 0 else 0.0,
            "detection_latency_ms_avg": float(sum(detection_latency_ms) / len(detection_latency_ms)) if detection_latency_ms else 0.0,
            "detection_latency_ms_p50": float(np.percentile(np.asarray(detection_latency_ms, dtype=np.float64), 50)) if detection_latency_ms else 0.0,
            "detection_latency_ms_p95": float(np.percentile(np.asarray(detection_latency_ms, dtype=np.float64), 95)) if detection_latency_ms else 0.0,
            "detection_latency_ms_max": float(max(detection_latency_ms)) if detection_latency_ms else 0.0,
            "frame_order_violations": frame_order_violations,
            "gpu_memory_allocated_mb": tracker_metrics.get("gpu_memory_allocated_mb"),
            "gpu_memory_reserved_mb": tracker_metrics.get("gpu_memory_reserved_mb"),
            "gpu_peak_allocated_mb": tracker_metrics.get("gpu_peak_allocated_mb"),
            "gpu_peak_reserved_mb": tracker_metrics.get("gpu_peak_reserved_mb"),
            "duration_seconds": float(metrics.get("duration_seconds", 0.0)),
        }
        output_manager.save_detection_tracking_metrics(detection_tracking_metrics)
        output_tracks = track_manager.get_all_output_tracks()
        output_observations = track_manager.get_all_observations()
        evidence_index_records = evidence_collector.evidence_index
        evidence_metrics = evidence_collector.metrics
        if evidence_metrics.get("pending_evidence_tracks_at_shutdown", 0) != 0 or evidence_metrics.get("pending_frame_reference_count", 0) != 0:
            logger.error(
                "EvidenceCollector shutdown state pending_tracks=%s pending_frame_references=%s",
                evidence_metrics.get("pending_evidence_tracks_at_shutdown", 0),
                evidence_metrics.get("pending_frame_reference_count", 0),
            )
        evidence_summary_by_track = _build_evidence_summary_by_track(evidence_index_records)
        enrichment_by_track = {item.local_track_id: item.to_dict() for item in enrichment_results}
        track_crop_manifest_rows = _merge_track_crop_manifest_rows(
            evidence_collector.debug_track_crop_rows,
            enrichment_results,
        )
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
        evidence_metrics_path = output_manager.save_evidence_metrics(evidence_metrics)
        track_crop_manifest_path = output_manager.save_track_crop_manifest(track_crop_manifest_rows)
        capture_zone_index_path = output_manager.save_capture_zone_index(evidence_collector.capture_zone_index)
        capture_zone_metrics_path = output_manager.save_capture_zone_metrics(
            {
                key: value
                for key, value in evidence_metrics.items()
                if str(key).startswith("capture_zone_")
            }
        )
        vehicle_enrichment_path = output_manager.save_vehicle_enrichment([item.to_dict() for item in enrichment_results])
        vehicle_enrichment_metrics_path = output_manager.save_vehicle_enrichment_metrics(vehicle_enrichment_manager.metrics)
        vehicle_enrichment_validation_report_path = output_manager.save_vehicle_enrichment_validation_report(
            _build_vehicle_enrichment_validation_rows(enrichment_results)
        )
        vehicle_enrichment_crop_diagnostics_path = output_manager.save_vehicle_enrichment_crop_diagnostics(
            _build_vehicle_enrichment_crop_diagnostics_rows(enrichment_results)
        )
        vehicle_enrichment_track_evidence_summary_path = output_manager.save_vehicle_enrichment_track_evidence_summary(
            _build_vehicle_enrichment_track_evidence_summary_rows(enrichment_results, track_manager.get_all_output_tracks())
        )
        enrichment_by_local_track_id = {item.local_track_id: item for item in enrichment_results}
        motorcycle_geometry_rows: list[dict[str, Any]] = []
        for row in evidence_collector.motorcycle_geometry_records:
            merged = dict(row)
            enrichment = enrichment_by_local_track_id.get(str(row.get("local_track_id", "")))
            evidence_sources = [getattr(item, "evidence_source", None) for item in getattr(enrichment, "evidence_used", [])] if enrichment is not None else []
            used_capture_zone = "capture_zone" in {str(item or "") for item in evidence_sources}
            florence_called = bool(getattr(enrichment, "vehicle_attribute_inference_count", 0)) if enrichment is not None else False
            final_colour = str(getattr(getattr(enrichment, "vehicle_colour", None), "label", "UNKNOWN") or "UNKNOWN") if enrichment is not None else "UNKNOWN"
            final_colour_status = str(getattr(getattr(enrichment, "vehicle_colour", None), "status", "") or "") if enrichment is not None else ""
            if merged.get("geometry_status") == "CAPTURED_ELIGIBLE" and used_capture_zone and florence_called and final_colour.upper() == "UNKNOWN":
                merged["geometry_status"] = "CAPTURED_BUT_ENRICHMENT_FAILED"
                merged["geometry_reason"] = "capture_zone_crop_reached_enrichment_but_colour_unknown"
            merged["enrichment_source"] = "capture_zone" if used_capture_zone else ("existing_track_evidence" if enrichment is not None else None)
            merged["eligible_crop_count"] = int(getattr(enrichment, "eligible_crop_count", 0)) if enrichment is not None else 0
            merged["florence_called"] = florence_called
            merged["final_colour"] = final_colour
            merged["final_colour_status"] = final_colour_status
            motorcycle_geometry_rows.append(merged)
        motorcycle_geometry_report_path = output_manager.save_motorcycle_geometry_report(motorcycle_geometry_rows)
        status_counts: dict[str, int] = {}
        for row in motorcycle_geometry_rows:
            status = str(row.get("geometry_status", "UNKNOWN"))
            status_counts[status] = status_counts.get(status, 0) + 1
        trigger_buckets = {"<0.50": 0, "0.50-0.60": 0, "0.60-0.70": 0, "0.70-0.80": 0, ">0.80": 0}
        for row in motorcycle_geometry_rows:
            frame_height = float(row.get("source_frame_height", 0) or 0)
            max_trigger_y = float(row.get("max_trigger_y", 0.0) or 0.0)
            ratio = (max_trigger_y / frame_height) if frame_height > 0 else 0.0
            if ratio < 0.50:
                trigger_buckets["<0.50"] += 1
            elif ratio < 0.60:
                trigger_buckets["0.50-0.60"] += 1
            elif ratio < 0.70:
                trigger_buckets["0.60-0.70"] += 1
            elif ratio < 0.80:
                trigger_buckets["0.70-0.80"] += 1
            else:
                trigger_buckets[">0.80"] += 1
        motorcycle_metrics_summary = {
            "motorcycle_tracks_total": len(motorcycle_geometry_rows),
            "motorcycle_tracks_reached_zone": sum(1 for row in motorcycle_geometry_rows if bool(row.get("entered_zone"))),
            "motorcycle_tracks_never_reached_zone": status_counts.get("NEVER_REACHED_ZONE", 0),
            "motorcycle_tracks_lost_before_zone": status_counts.get("TRACK_ENDED_BEFORE_ZONE", 0),
            "motorcycle_tracks_reached_zone_no_capture": status_counts.get("REACHED_ZONE_NO_CAPTURE", 0),
            "motorcycle_tracks_captured": sum(1 for row in motorcycle_geometry_rows if int(row.get("retained_candidates", 0) or 0) > 0),
            "motorcycle_tracks_captured_too_small": status_counts.get("CAPTURED_TOO_SMALL", 0),
            "motorcycle_tracks_with_eligible_zone_crop": sum(1 for row in motorcycle_geometry_rows if bool(row.get("evidence_eligible_zone_crop")) or bool(row.get("florence_eligible_zone_crop"))),
            "motorcycle_tracks_sent_to_florence": sum(1 for row in motorcycle_geometry_rows if bool(row.get("florence_called")) and row.get("enrichment_source") == "capture_zone"),
            "motorcycle_tracks_with_valid_colour": sum(
                1
                for row in motorcycle_geometry_rows
                if row.get("enrichment_source") == "capture_zone" and str(row.get("final_colour", "UNKNOWN")).upper() != "UNKNOWN"
            ),
            "motorcycle_max_trigger_y_distribution": trigger_buckets,
            "motorcycle_geometry_status_counts": status_counts,
        }
        vehicle_pipeline_trace_rows = _build_vehicle_pipeline_trace_rows(
            track_manager.get_all_output_tracks(),
            track_crop_manifest_rows,
            motorcycle_geometry_rows,
            enrichment_results,
        )
        vehicle_pipeline_trace_path = output_manager.save_vehicle_pipeline_trace(vehicle_pipeline_trace_rows)
        ocr_mukul_result_rows = _build_ocr_mukul_result_rows(enrichment_results)
        if ocr_mukul_result_rows:
            with (output_manager.run_directory / "ocr_mukul_florence_results.csv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(ocr_mukul_result_rows[0].keys()))
                writer.writeheader()
                writer.writerows(ocr_mukul_result_rows)
            (output_manager.run_directory / "ocr_mukul_florence_results.json").write_text(
                json.dumps(ocr_mukul_result_rows, indent=2),
                encoding="utf-8",
            )
        if validated_config["vehicle_enrichment"].get("execution_mode") == "comparison":
            _write_current_vs_ocr_mukul_artifacts(output_manager.run_directory, enrichment_results)
        vehicle_attribute_result_rows = _build_vehicle_attribute_result_rows(enrichment_results)
        if vehicle_attribute_result_rows:
            with (output_manager.run_directory / "vehicle_attribute_results.csv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(vehicle_attribute_result_rows[0].keys()))
                writer.writeheader()
                writer.writerows(vehicle_attribute_result_rows)
            (output_manager.run_directory / "vehicle_attribute_results.json").write_text(
                json.dumps(vehicle_attribute_result_rows, indent=2),
                encoding="utf-8",
            )
        vehicle_colour_result_rows = _build_vehicle_colour_result_rows(enrichment_results)
        if vehicle_colour_result_rows:
            with (output_manager.run_directory / "vehicle_colour_results.csv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(vehicle_colour_result_rows[0].keys()))
                writer.writeheader()
                writer.writerows(vehicle_colour_result_rows)
            (output_manager.run_directory / "vehicle_colour_results.json").write_text(
                json.dumps(vehicle_colour_result_rows, indent=2),
                encoding="utf-8",
            )
            with (output_manager.florence_results_directory / "vehicle_colour_results.csv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(vehicle_colour_result_rows[0].keys()))
                writer.writeheader()
                writer.writerows(vehicle_colour_result_rows)
            (output_manager.florence_results_directory / "vehicle_colour_results.json").write_text(
                json.dumps(vehicle_colour_result_rows, indent=2),
                encoding="utf-8",
            )
        vehicle_body_type_result_rows = _build_vehicle_body_type_result_rows(enrichment_results)
        if vehicle_body_type_result_rows:
            with (output_manager.run_directory / "vehicle_body_type_results.csv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(vehicle_body_type_result_rows[0].keys()))
                writer.writeheader()
                writer.writerows(vehicle_body_type_result_rows)
            (output_manager.run_directory / "vehicle_body_type_results.json").write_text(
                json.dumps(vehicle_body_type_result_rows, indent=2),
                encoding="utf-8",
            )
            with (output_manager.body_type_results_directory / "vehicle_body_type_results.csv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(vehicle_body_type_result_rows[0].keys()))
                writer.writeheader()
                writer.writerows(vehicle_body_type_result_rows)
            (output_manager.body_type_results_directory / "vehicle_body_type_results.json").write_text(
                json.dumps(vehicle_body_type_result_rows, indent=2),
                encoding="utf-8",
            )
            for row in vehicle_body_type_result_rows:
                crop_path = Path(str(row.get("crop_path") or ""))
                if not crop_path.exists():
                    continue
                camera_id = str(row.get("camera_id") or "UNKNOWN")
                local_track_id = str(row.get("local_track_id") or "TRACK_UNKNOWN")
                track_name = local_track_id.split(":")[-1]
                target_dir = output_manager.body_type_selected_crops_directory / camera_id / track_name
                target_dir.mkdir(parents=True, exist_ok=True)
                target_path = target_dir / crop_path.name
                if crop_path.resolve() != target_path.resolve():
                    shutil.copyfile(str(crop_path), str(target_path))
        vehicle_colour_track_summary_rows = _build_vehicle_colour_track_summary_rows(enrichment_results)
        if vehicle_colour_track_summary_rows:
            with (output_manager.run_directory / "vehicle_colour_track_summary.csv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(vehicle_colour_track_summary_rows[0].keys()))
                writer.writeheader()
                writer.writerows(vehicle_colour_track_summary_rows)
            (output_manager.run_directory / "vehicle_colour_track_summary.json").write_text(
                json.dumps(vehicle_colour_track_summary_rows, indent=2),
                encoding="utf-8",
            )
        plate_ocr_result_rows = _build_plate_ocr_result_rows(enrichment_results)
        if plate_ocr_result_rows:
            with (output_manager.run_directory / "plate_ocr_results.csv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(plate_ocr_result_rows[0].keys()))
                writer.writeheader()
                writer.writerows(plate_ocr_result_rows)
            (output_manager.run_directory / "plate_ocr_results.json").write_text(
                json.dumps(plate_ocr_result_rows, indent=2),
                encoding="utf-8",
            )
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
                "configured_dtype": device_info.configured_dtype,
                "resolved_device": device_info.resolved_device,
                "resolved_dtype": device_info.resolved_dtype,
                "cuda_available": device_info.cuda_available,
                "cuda_device_count": device_info.cuda_device_count,
                "cuda_device_name": device_info.cuda_device_name,
                "torch_version": device_info.torch_version,
                "torch_cuda_version": device_info.torch_cuda_version,
                "configured_camera_count": metrics["configured_camera_count"],
                "enabled_camera_count": metrics["enabled_camera_count"],
                "max_frames_per_camera": validated_config["input"]["max_frames_per_camera"],
                "frame_limit_mode": "unlimited" if validated_config["input"]["max_frames_per_camera"] is None else "limited",
                "worker_count": metrics["worker_count"],
                "ingestion_worker_count": int(metrics.get("ingestion_worker_count", metrics["worker_count"])),
                "per_camera_buffer_count": int(metrics.get("per_camera_buffer_count", 0)),
                "per_camera_buffer_size": int(metrics.get("per_camera_buffer_size", 0)),
                "scheduler_policy": str(metrics.get("scheduler_policy", "round_robin")),
                "camera_read_jobs": int(metrics.get("camera_read_jobs", 0)),
                "camera_read_failures": int(metrics.get("camera_read_failures", 0)),
                "frames_scheduled_by_camera": dict(metrics.get("frames_scheduled_by_camera", {})),
                "frames_consumed_by_camera": dict(metrics.get("frames_consumed_by_camera", {})),
                "per_camera_buffer_peak": dict(metrics.get("per_camera_buffer_peak", {})),
                "buffer_full_count": int(metrics.get("buffer_full_count", 0)),
                "max_consecutive_frames_same_camera": int(metrics.get("max_consecutive_frames_same_camera", 0)),
                "scheduler_skipped_empty_camera": int(metrics.get("scheduler_skipped_empty_camera", 0)),
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
                "evidence_capture_zone": {
                    "enabled": any(_capture_zone_enabled_for_camera(validated_config, camera["camera_id"]) for camera in validated_config["input"]["cameras"] if camera["enabled"]),
                    "top_ratio": float(validated_config["evidence"].get("capture_zone", {}).get("top_ratio", 0.0)),
                    "bottom_ratio": float(validated_config["evidence"].get("capture_zone", {}).get("bottom_ratio", 0.0)),
                    "trigger_point": str(validated_config["evidence"].get("capture_zone", {}).get("trigger_point", "bottom_center")),
                    "tracks_entered": int(evidence_metrics.get("capture_zone_tracks_entered", 0)),
                    "tracks_with_saved_evidence": int(evidence_metrics.get("capture_zone_tracks_with_saved_evidence", 0)),
                    "candidates_saved": int(evidence_metrics.get("capture_zone_candidates_saved", 0)),
                    "crops_used_by_enrichment": int(vehicle_enrichment_manager.metrics.get("capture_zone_crops_used_by_enrichment", 0)),
                    "fallback_count": int(vehicle_enrichment_manager.metrics.get("capture_zone_fallback_to_existing_evidence", 0)),
                },
                "motorcycle_geometry": {
                    **motorcycle_metrics_summary,
                    "report_path": str(motorcycle_geometry_report_path),
                },
                "track_crop_manifest_path": str(track_crop_manifest_path),
                "vehicle_pipeline_trace_path": str(vehicle_pipeline_trace_path),
                "vehicle_enrichment_enabled": validated_config["vehicle_enrichment"]["enabled"],
                "vehicle_enrichment_result_count": len(enrichment_results),
                "colour_async_enabled": bool(vehicle_enrichment_manager.metrics.get("colour_async_enabled", False)),
                "colour_worker_count": int(vehicle_enrichment_manager.metrics.get("colour_worker_count", 0)),
                "colour_queue_count": int(vehicle_enrichment_manager.metrics.get("colour_queue_count", 0)),
                "colour_queue_size": int(vehicle_enrichment_manager.metrics.get("colour_queue_size", 0)),
                "colour_queue_peak_depth": int(vehicle_enrichment_manager.metrics.get("colour_queue_peak_depth", 0)),
                "colour_queue_block_count": int(vehicle_enrichment_manager.metrics.get("colour_queue_block_count", 0)),
                "colour_jobs_enqueued": int(vehicle_enrichment_manager.metrics.get("colour_jobs_enqueued", 0)),
                "colour_jobs_completed": int(vehicle_enrichment_manager.metrics.get("colour_jobs_completed", 0)),
                "colour_jobs_failed": int(vehicle_enrichment_manager.metrics.get("colour_jobs_failed", 0)),
                "colour_jobs_duplicate_attempts": int(vehicle_enrichment_manager.metrics.get("colour_jobs_duplicate_attempts", 0)),
                "colour_jobs_lost": int(vehicle_enrichment_manager.metrics.get("colour_jobs_lost", 0)),
                "pending_colour_jobs_at_shutdown": int(vehicle_enrichment_manager.metrics.get("track_evidence_pending_count", 0)),
                "colour_worker_busy_time_ms": float(vehicle_enrichment_manager.metrics.get("colour_worker_busy_time_ms", 0.0) or 0.0),
                "colour_inference_strategy": str(vehicle_enrichment_manager.metrics.get("colour_inference_strategy", "")),
                "colour_tracks_processed": int(vehicle_enrichment_manager.metrics.get("colour_tracks_processed", 0)),
                "colour_tracks_resolved_crop1": int(vehicle_enrichment_manager.metrics.get("colour_tracks_resolved_crop1", 0)),
                "colour_tracks_resolved_crop2": int(vehicle_enrichment_manager.metrics.get("colour_tracks_resolved_crop2", 0)),
                "colour_tracks_resolved_crop3": int(vehicle_enrichment_manager.metrics.get("colour_tracks_resolved_crop3", 0)),
                "colour_tracks_unresolved": int(vehicle_enrichment_manager.metrics.get("colour_tracks_unresolved", 0)),
                "colour_florence_calls_total": int(vehicle_enrichment_manager.metrics.get("colour_florence_calls_total", 0)),
                "average_colour_calls_per_track": float(vehicle_enrichment_manager.metrics.get("vehicle_attribute_average_colour_calls_per_track", 0.0) or 0.0),
                "fallback_to_crop2_count": int(vehicle_enrichment_manager.metrics.get("fallback_to_crop2_count", 0)),
                "fallback_to_crop3_count": int(vehicle_enrichment_manager.metrics.get("fallback_to_crop3_count", 0)),
                "detection_total_inference_time_ms": float(sum(inference_times)),
                "overall_pipeline_runtime_ms": float((datetime.now(timezone.utc) - datetime.fromisoformat(metadata.started_at)).total_seconds() * 1000.0),
                "car_tracks_total": int(vehicle_enrichment_manager.metrics.get("car_tracks_total", 0)),
                "car_tracks_with_body_type_crop": int(vehicle_enrichment_manager.metrics.get("car_tracks_with_body_type_crop", 0)),
                "car_tracks_sent_to_body_type_florence": int(vehicle_enrichment_manager.metrics.get("car_tracks_sent_to_body_type_florence", 0)),
                "car_tracks_with_valid_body_type": int(vehicle_enrichment_manager.metrics.get("car_tracks_with_valid_body_type", 0)),
                "car_tracks_body_type_unknown": int(vehicle_enrichment_manager.metrics.get("car_tracks_body_type_unknown", 0)),
                "body_type_distribution": dict(vehicle_enrichment_manager.metrics.get("body_type_label_distribution", {})),
                "body_type_florence_inference_count": int(vehicle_enrichment_manager.metrics.get("body_type_inference_calls", 0)),
                "body_type_average_inference_ms": float(vehicle_enrichment_manager.metrics.get("body_type_average_inference_ms", 0.0) or 0.0),
                "run_directory": str(output_manager.run_directory),
            }
        )
        runtime_state.update_system_status(
            colour_queue_depth=int(vehicle_enrichment_manager.metrics.get("colour_queue_peak_depth", 0) or 0),
            colour_queue_capacity=int(vehicle_enrichment_manager.metrics.get("colour_queue_size", 0) or 0),
            pending_colour_jobs=0,
            cache_misses=int(evidence_metrics.get("evidence_cache_misses", 0) or 0),
            frame_loss=0,
            order_violations=int(frame_order_violations),
            yolo_status="completed",
            colour_worker_status="completed",
        )
        runtime_state.mark_run_completed(
            status=metadata.status,
            summary={
                "run_id": metadata.run_id,
                "processed_frames": metadata.processed_frames,
                "run_directory": str(output_manager.run_directory),
            },
        )
        logger.info(
            "track output paths tracks=%s observations=%s lifecycle_metrics=%s evidence_index=%s evidence_metrics=%s track_crop_manifest=%s capture_zone_index=%s capture_zone_metrics=%s motorcycle_geometry_report=%s vehicle_pipeline_trace=%s vehicle_enrichment=%s vehicle_enrichment_metrics=%s vehicle_enrichment_validation_report=%s vehicle_enrichment_crop_diagnostics=%s vehicle_enrichment_track_evidence_summary=%s",
            tracks_path,
            observations_path,
            lifecycle_metrics_path,
            evidence_index_path,
            evidence_metrics_path,
            track_crop_manifest_path,
            capture_zone_index_path,
            capture_zone_metrics_path,
            motorcycle_geometry_report_path,
            vehicle_pipeline_trace_path,
            vehicle_enrichment_path,
            vehicle_enrichment_metrics_path,
            vehicle_enrichment_validation_report_path,
            vehicle_enrichment_crop_diagnostics_path,
            vehicle_enrichment_track_evidence_summary_path,
        )
        detector_tracker.reset_all()
        logger.info("Pipeline completed")
        return 0, output_manager.run_id, str(output_manager.run_directory)
    except Exception as exc:
        if vehicle_enrichment_manager is not None:
            try:
                vehicle_enrichment_manager.finalize_async_colour()
            except Exception:
                logger.exception("VehicleEnrichmentManager shutdown failed during error handling")
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
        runtime_state.set_pipeline_status("failed", run_id=output_manager.run_id)
        runtime_state.mark_run_completed(
            status=metadata.status,
            summary={
                "run_id": metadata.run_id,
                "processed_frames": metadata.processed_frames,
                "run_directory": str(output_manager.run_directory),
                "error_type": exc.__class__.__name__,
            },
        )
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


def _build_vehicle_enrichment_validation_rows(enrichment_results: list[Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result in enrichment_results:
        body_predictions = list(getattr(result.vehicle_body_type, "predictions", []) or [])
        colour_predictions = list(getattr(result.vehicle_colour, "predictions", []) or [])
        crop_paths = sorted(
            {
                str(item)
                for item in [
                    *[prediction.source_crop_path for prediction in body_predictions if prediction.source_crop_path],
                    *[prediction.source_crop_path for prediction in colour_predictions if prediction.source_crop_path],
                ]
                if item
            }
        )
        body_raw = [str(prediction.raw_response) for prediction in body_predictions if prediction.raw_response not in (None, "")]
        colour_raw = [str(prediction.raw_response) for prediction in colour_predictions if prediction.raw_response not in (None, "")]
        evidence_item = (list(getattr(result, "evidence_used", []) or []) or [None])[0]
        body_prediction = body_predictions[0] if body_predictions else None
        colour_prediction = colour_predictions[0] if colour_predictions else None
        rows.append(
            {
                "camera_id": result.camera_id,
                "local_track_id": result.local_track_id,
                "vehicle_class": result.vehicle_class,
                "crop_path": " | ".join(crop_paths),
                "candidate_crop_count": getattr(result, "candidate_crop_count", 0),
                "eligible_crop_count": getattr(result, "eligible_crop_count", 0),
                "preferred_crop_count": getattr(result, "preferred_crop_count", 0),
                "selected_body_type_crop_paths": " | ".join(getattr(result, "selected_body_type_crop_paths", []) or []),
                "selected_colour_crop_paths": " | ".join(getattr(result, "selected_colour_crop_paths", []) or []),
                "florence_mode": getattr(result, "florence_mode", None),
                "adapter_loaded": getattr(result, "adapter_loaded", None),
                "selected_crop_paths": " | ".join(getattr(result, "selected_crop_paths", []) or []),
                "caption_inference_count": getattr(result, "caption_inference_count", 0),
                "classification_trigger": getattr(result, "classification_trigger", None),
                "source_frame_width": getattr(evidence_item, "source_frame_width", None),
                "source_frame_height": getattr(evidence_item, "source_frame_height", None),
                "original_bbox": list(getattr(evidence_item, "original_bbox_xyxy", []) or []),
                "expanded_crop_bbox": list(getattr(evidence_item, "expanded_crop_bbox_xyxy", []) or []),
                "context_padding_ratio": getattr(evidence_item, "context_padding_ratio", None),
                "original_crop_width": getattr(evidence_item, "original_crop_width", None),
                "original_crop_height": getattr(evidence_item, "original_crop_height", None),
                "resolution_tier": getattr(evidence_item, "resolution_tier", None),
                "sharpness": getattr(evidence_item, "sharpness_score", None),
                "brightness": getattr(evidence_item, "brightness_score", None),
                "edge_truncated": getattr(evidence_item, "edge_truncated", None),
                "quality_score": getattr(evidence_item, "quality_score", None),
                "square_padding_applied": getattr(body_prediction or colour_prediction, "square_padding_applied", None),
                "padded_width": getattr(body_prediction or colour_prediction, "padded_width", None),
                "padded_height": getattr(body_prediction or colour_prediction, "padded_height", None),
                "florence_input_width": getattr(body_prediction or colour_prediction, "florence_input_width", None),
                "florence_input_height": getattr(body_prediction or colour_prediction, "florence_input_height", None),
                "predicted_body_type": result.vehicle_body_type.label,
                "body_type_raw_response": " | ".join(body_raw),
                "body_type_reason": result.vehicle_body_type.aggregation_reason or result.vehicle_body_type.reason,
                "final_body_type_reason": getattr(result, "final_body_type_reason", None),
                "predicted_colour": result.vehicle_colour.label,
                "colour_raw_response": " | ".join(colour_raw),
                "colour_reason": result.vehicle_colour.aggregation_reason or result.vehicle_colour.reason,
                "final_colour_reason": getattr(result, "final_colour_reason", None),
                "final_body_type": result.vehicle_body_type.label,
                "final_colour": result.vehicle_colour.label,
                "final_reason": getattr(result, "final_reason", None),
                "manual_body_type": "",
                "manual_colour": "",
                "body_type_correct": "",
                "colour_correct": "",
                "review_notes": "",
            }
        )
    return rows


def _build_vehicle_enrichment_crop_diagnostics_rows(enrichment_results: list[Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result in enrichment_results:
        for item in list(getattr(result, "evidence_used", []) or []):
            rows.append(
                {
                    "camera_id": result.camera_id,
                    "local_track_id": result.local_track_id,
                    "evidence_role": getattr(item, "evidence_role", None),
                    "frame_index": getattr(item, "frame_number", None),
                    "timestamp": getattr(item, "timestamp_seconds", None),
                    "candidate_rank": getattr(item, "candidate_rank", None),
                    "candidate_retained": getattr(item, "candidate_retained", None),
                    "candidate_rejection_reason": getattr(item, "candidate_rejection_reason", None),
                    "frame_gap_from_previous_selected": getattr(item, "frame_gap_from_previous_selected", None),
                    "duplicate_score": getattr(item, "duplicate_score", None),
                    "crop_path": getattr(item, "vehicle_crop_path", None),
                    "source_frame_width": getattr(item, "source_frame_width", None),
                    "source_frame_height": getattr(item, "source_frame_height", None),
                    "original_crop_width": getattr(item, "original_crop_width", None),
                    "original_crop_height": getattr(item, "original_crop_height", None),
                "resolution_tier": getattr(item, "resolution_tier", None),
                "selection_tier": getattr(item, "colour_selection_tier", None),
                "sharpness": getattr(item, "sharpness_score", None),
                    "brightness": getattr(item, "brightness_score", None),
                    "quality_score": getattr(item, "quality_score", None),
                    "eligible_for_body_type": getattr(item, "florence_eligible_for_body_type", None),
                    "eligible_for_colour": getattr(item, "florence_eligible_for_colour", None),
                    "body_type_skip_reason": getattr(item, "florence_body_type_skip_reason", None),
                    "colour_skip_reason": getattr(item, "florence_colour_skip_reason", None),
                    "selected_for_body_type": getattr(item, "selected_for_body_type", None),
                    "selected_for_colour": getattr(item, "selected_for_colour", None),
                    "body_type_crop_result": getattr(item, "body_type_crop_result", None),
                    "colour_crop_result": getattr(item, "colour_crop_result", None),
                    "florence_mode": getattr(result, "florence_mode", None),
                }
            )
    return rows


def _build_vehicle_enrichment_track_evidence_summary_rows(enrichment_results: list[Any], tracks: list[Any]) -> list[dict[str, Any]]:
    results_by_track = {str(result.local_track_id): result for result in enrichment_results}
    rows: list[dict[str, Any]] = []
    for track in tracks:
        result = results_by_track.get(str(track.local_track_id))
        if result is None:
            continue
        evidence_items = list(getattr(result, "evidence_used", []) or [])
        rows.append(
            {
                "camera_id": result.camera_id,
                "local_track_id": result.local_track_id,
                "vehicle_class": result.vehicle_class,
                "track_start_frame": int(track.first_frame),
                "track_end_frame": int(track.last_frame),
                "track_duration_frames": int(max(0, track.last_frame - track.first_frame + 1)),
                "candidate_crops_seen": int(getattr(result, "candidate_crop_count", 0)),
                "candidate_crops_retained": int(len(evidence_items)),
                "acceptable_crops": int(len([item for item in evidence_items if getattr(item, "resolution_tier", "") == "acceptable"])),
                "preferred_crops": int(len([item for item in evidence_items if getattr(item, "resolution_tier", "") == "preferred"])),
                "selected_body_type_crops": " | ".join(getattr(result, "selected_body_type_crop_paths", []) or []),
                "selected_colour_crops": " | ".join(getattr(result, "selected_colour_crop_paths", []) or []),
                "selected_crop_paths": " | ".join(getattr(result, "selected_crop_paths", []) or []),
                "caption_inference_count": int(getattr(result, "caption_inference_count", 0)),
                "largest_original_crop_width": max((int(getattr(item, "original_crop_width", 0)) for item in evidence_items), default=0),
                "largest_original_crop_height": max((int(getattr(item, "original_crop_height", 0)) for item in evidence_items), default=0),
                "best_quality_score": max((float(getattr(item, "quality_score", 0.0)) for item in evidence_items), default=0.0),
                "body_type_status": result.vehicle_body_type.status,
                "body_type_label": result.vehicle_body_type.label,
                "colour_status": result.vehicle_colour.status,
                "colour_label": result.vehicle_colour.label,
                "florence_mode": getattr(result, "florence_mode", None),
                "adapter_loaded": getattr(result, "adapter_loaded", None),
            }
        )
    return rows


def _build_ocr_mukul_result_rows(enrichment_results: list[Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result in enrichment_results:
        for caption_row in list(getattr(result, "crop_level_captions", []) or []):
            frame_index = caption_row.get("frame_index")
            crop_path = caption_row.get("crop_path")
            body_item = next((item for item in list(getattr(result, "crop_level_body_types", []) or []) if item.get("crop_path") == crop_path and item.get("frame_index") == frame_index), {})
            colour_item = next((item for item in list(getattr(result, "crop_level_colours", []) or []) if item.get("crop_path") == crop_path and item.get("frame_index") == frame_index), {})
            evidence = next((item for item in list(getattr(result, "evidence_used", []) or []) if str(getattr(item, "vehicle_crop_path", "")) == str(crop_path) and int(getattr(item, "frame_number", -1)) == int(frame_index)), None)
            rows.append(
                {
                    "camera_id": result.camera_id,
                    "local_track_id": result.local_track_id,
                    "frame_index": frame_index,
                    "crop_path": crop_path,
                    "original_crop_width": getattr(evidence, "original_crop_width", None),
                    "original_crop_height": getattr(evidence, "original_crop_height", None),
                    "resolution_tier": getattr(evidence, "resolution_tier", None),
                    "quality_score": getattr(evidence, "quality_score", None),
                    "caption": caption_row.get("caption"),
                    "raw_body_type_phrase": body_item.get("raw_body_type_phrase"),
                    "normalized_body_type": body_item.get("normalized_body_type"),
                    "raw_colour_phrase": colour_item.get("raw_colour_phrase"),
                    "normalized_colour": colour_item.get("normalized_colour"),
                    "inference_time_ms": "",
                }
            )
    return rows


def _build_vehicle_attribute_result_rows(enrichment_results: list[Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result in enrichment_results:
        if getattr(result, "attribute_backend", None) != "base_florence":
            continue
        for caption_row in list(getattr(result, "crop_level_captions", []) or []):
            frame_index = caption_row.get("frame_index")
            crop_path = caption_row.get("crop_path")
            body_item = next((item for item in list(getattr(result, "crop_level_body_types", []) or []) if item.get("crop_path") == crop_path and item.get("frame_index") == frame_index), {})
            colour_item = next((item for item in list(getattr(result, "crop_level_colours", []) or []) if item.get("crop_path") == crop_path and item.get("frame_index") == frame_index), {})
            rows.append(
                {
                    "camera_id": result.camera_id,
                    "local_track_id": result.local_track_id,
                    "frame_index": frame_index,
                    "vehicle_crop_path": crop_path,
                    "colour_task_token": colour_item.get("task_token"),
                    "colour_prompt": colour_item.get("prompt"),
                    "colour_effective_processor_text": colour_item.get("effective_processor_text"),
                    "colour_raw_response": colour_item.get("raw_response"),
                    "parsed_colour": colour_item.get("normalized_colour"),
                    "body_type_task_token": body_item.get("task_token"),
                    "body_type_prompt": body_item.get("prompt"),
                    "body_type_effective_processor_text": body_item.get("effective_processor_text"),
                    "body_type_raw_response": body_item.get("raw_response"),
                    "parsed_body_type": body_item.get("normalized_body_type"),
                    "colour_reason": result.final_colour_reason,
                    "body_type_reason": result.final_body_type_reason,
                    "colour_inference_time_ms": colour_item.get("inference_time_ms"),
                    "body_type_inference_time_ms": body_item.get("inference_time_ms"),
                    "adapter_loaded": False,
                }
            )
    return rows


def _build_vehicle_body_type_result_rows(enrichment_results: list[Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result in enrichment_results:
        if getattr(result, "attribute_backend", None) != "base_florence":
            continue
        body_predictions = list(getattr(result.vehicle_body_type, "predictions", []) or [])
        prediction_lookup = {
            (str(prediction.source_crop_path or ""), int(prediction.source_frame_number if prediction.source_frame_number is not None else -1)): prediction
            for prediction in body_predictions
        }
        for body_item in list(getattr(result, "crop_level_body_types", []) or []):
            crop_path = str(body_item.get("crop_path") or "")
            frame_index = int(body_item.get("frame_index", -1))
            prediction = prediction_lookup.get((crop_path, frame_index))
            evidence = next(
                (
                    item
                    for item in list(getattr(result, "evidence_used", []) or [])
                    if str(getattr(item, "vehicle_crop_path", "") or "") == crop_path
                    and int(getattr(item, "frame_number", -1)) == frame_index
                ),
                None,
            )
            rows.append(
                {
                    "camera_id": result.camera_id,
                    "local_track_id": result.local_track_id,
                    "vehicle_class": result.vehicle_class,
                    "frame_number": frame_index,
                    "crop_path": crop_path,
                    "crop_width": getattr(evidence, "original_crop_width", None),
                    "crop_height": getattr(evidence, "original_crop_height", None),
                    "quality_score": getattr(evidence, "quality_score", None),
                    "resolution_tier": getattr(evidence, "resolution_tier", None),
                    "raw_response": body_item.get("raw_response") or ("" if prediction is None or prediction.raw_response in (None, "") else str(prediction.raw_response)),
                    "parsed_body_type": body_item.get("normalized_body_type"),
                    "status": body_item.get("status") or ("completed" if prediction is not None else "skipped"),
                    "reason": body_item.get("reason") or getattr(result, "final_body_type_reason", None),
                    "inference_duration_ms": body_item.get("inference_time_ms") or getattr(prediction, "inference_duration_ms", None),
                }
            )
    return rows


def _build_vehicle_colour_result_rows(enrichment_results: list[Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result in enrichment_results:
        if getattr(result, "attribute_backend", None) != "base_florence":
            continue
        colour_predictions = list(getattr(result.vehicle_colour, "predictions", []) or [])
        prediction_lookup = {
            (str(prediction.source_crop_path or ""), int(prediction.source_frame_number if prediction.source_frame_number is not None else -1)): prediction
            for prediction in colour_predictions
        }
        caption_lookup = {
            (str(item.get("crop_path") or ""), int(item.get("frame_index", -1))): item
            for item in list(getattr(result, "crop_level_captions", []) or [])
        }
        for colour_item in list(getattr(result, "crop_level_colours", []) or []):
            crop_path = str(colour_item.get("crop_path") or "")
            frame_index = int(colour_item.get("frame_index", -1))
            prediction = prediction_lookup.get((crop_path, frame_index))
            caption_row = caption_lookup.get((crop_path, frame_index), {})
            evidence = next(
                (
                    item
                    for item in list(getattr(result, "evidence_used", []) or [])
                    if str(getattr(item, "vehicle_crop_path", "") or "") == crop_path
                    and int(getattr(item, "frame_number", -1)) == frame_index
                ),
                None,
            )
            rows.append(
                {
                    "camera_id": result.camera_id,
                    "local_track_id": result.local_track_id,
                    "frame_index": frame_index,
                    "vehicle_class": result.vehicle_class,
                    "vehicle_crop_path": crop_path,
                    "source_crop_path": crop_path,
                    "florence_selected_copy_path": crop_path,
                    "crop_quality_score": getattr(evidence, "quality_score", None),
                    "original_crop_width": getattr(evidence, "original_crop_width", None),
                    "original_crop_height": getattr(evidence, "original_crop_height", None),
                    "task_token": getattr(result.vehicle_colour, "task_prompt", None),
                    "prompt": getattr(result.vehicle_colour, "prompt_text", None),
                    "effective_processor_text": f"{getattr(result.vehicle_colour, 'task_prompt', '') or ''}{getattr(result.vehicle_colour, 'prompt_text', '') or ''}",
                    "raw_response": "" if prediction is None or prediction.raw_response in (None, "") else str(prediction.raw_response),
                    "post_processed_response": caption_row.get("caption"),
                    "parsed_colour": colour_item.get("normalized_colour"),
                    "selection_tier": colour_item.get("selection_tier") or getattr(evidence, "colour_selection_tier", None),
                    "colour_status": "completed" if prediction is not None else "skipped",
                    "colour_reason": colour_item.get("reason") or getattr(result, "final_colour_reason", None),
                    "inference_time_ms": None if prediction is None else prediction.inference_duration_ms,
                    "model": None if prediction is None else prediction.source_model,
                    "adapter_loaded": False if prediction is None or prediction.adapter_active is None else prediction.adapter_active,
                    "crop_source": colour_item.get("crop_source"),
                    "crop_available": colour_item.get("crop_available"),
                    "crop_skip_reason": colour_item.get("crop_skip_reason"),
                }
            )
    return rows


def _build_vehicle_colour_track_summary_rows(enrichment_results: list[Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result in enrichment_results:
        if getattr(result, "attribute_backend", None) != "base_florence":
            continue
        colour_predictions = list(getattr(result.vehicle_colour, "predictions", []) or [])
        crop_colour_predictions = [
            {
                "frame_index": prediction.source_frame_number,
                "label": prediction.label,
                "raw_response": prediction.raw_response,
            }
            for prediction in colour_predictions
        ]
        valid_prediction_count = sum(1 for prediction in colour_predictions if str(prediction.label) not in {"", "UNKNOWN", "None"})
        unknown_prediction_count = sum(1 for prediction in colour_predictions if str(prediction.label) in {"", "UNKNOWN", "None"})
        rows.append(
            {
                "camera_id": result.camera_id,
                "local_track_id": result.local_track_id,
                "vehicle_class": result.vehicle_class,
                "selected_crop_count": len(list(getattr(result, "evidence_used", []) or [])),
                "colour_inference_count": int(getattr(result, "vehicle_attribute_inference_count", 0) or len(colour_predictions)),
                "colour_selection_tier": getattr(result, "colour_selection_tier", None),
                "crop_colour_predictions": crop_colour_predictions,
                "final_vehicle_colour": result.vehicle_colour.label,
                "colour_consensus_status": result.vehicle_colour.status,
                "colour_consensus_reason": getattr(result, "final_colour_reason", None),
                "valid_prediction_count": valid_prediction_count,
                "unknown_prediction_count": unknown_prediction_count,
            }
        )
    return rows


def _merge_track_crop_manifest_rows(raw_rows: list[dict[str, Any]], enrichment_results: list[Any]) -> list[dict[str, Any]]:
    merged = [dict(row) for row in raw_rows]
    by_track_frame: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in merged:
        by_track_frame.setdefault((str(row.get("local_track_id", "")), int(row.get("frame_number", -1))), []).append(row)
    for result in enrichment_results:
        for item in list(getattr(result, "evidence_used", []) or []):
            key = (str(getattr(item, "local_track_id", "")), int(getattr(item, "frame_number", -1)))
            for row in by_track_frame.get(key, []):
                row["evidence_eligible"] = bool(getattr(item, "evidence_eligible", row.get("evidence_eligible", False)))
                evidence_reasons = list(getattr(item, "rejection_reasons", []) or [])
                row["evidence_rejection_reason"] = " | ".join(evidence_reasons) if evidence_reasons else row.get("evidence_rejection_reason")
                row["florence_eligible"] = bool(getattr(item, "florence_eligible_for_colour", row.get("florence_eligible", False)))
                row["florence_rejection_reason"] = getattr(item, "florence_colour_skip_reason", None) or row.get("florence_rejection_reason")
                row["selection_tier"] = getattr(item, "colour_selection_tier", row.get("selection_tier"))
    return merged


def _build_vehicle_pipeline_trace_rows(
    tracks: list[Any],
    track_crop_manifest_rows: list[dict[str, Any]],
    motorcycle_geometry_rows: list[dict[str, Any]],
    enrichment_results: list[Any],
) -> list[dict[str, Any]]:
    raw_crop_counts: dict[str, int] = {}
    evidence_eligible_counts: dict[str, int] = {}
    florence_eligible_counts: dict[str, int] = {}
    preferred_crop_counts: dict[str, int] = {}
    fallback_crop_counts: dict[str, int] = {}
    for row in track_crop_manifest_rows:
        local_track_id = str(row.get("local_track_id", ""))
        raw_crop_counts[local_track_id] = raw_crop_counts.get(local_track_id, 0) + 1
        if bool(row.get("evidence_eligible")):
            evidence_eligible_counts[local_track_id] = evidence_eligible_counts.get(local_track_id, 0) + 1
        if bool(row.get("florence_eligible")):
            florence_eligible_counts[local_track_id] = florence_eligible_counts.get(local_track_id, 0) + 1
        if str(row.get("selection_tier", "") or "") == "preferred":
            preferred_crop_counts[local_track_id] = preferred_crop_counts.get(local_track_id, 0) + 1
        if str(row.get("selection_tier", "") or "") == "low_resolution_fallback":
            fallback_crop_counts[local_track_id] = fallback_crop_counts.get(local_track_id, 0) + 1
    geometry_by_track = {str(row.get("local_track_id", "")): dict(row) for row in motorcycle_geometry_rows}
    enrichment_by_track = {str(result.local_track_id): result for result in enrichment_results}
    rows: list[dict[str, Any]] = []
    for track in tracks:
        local_track_id = str(track.local_track_id)
        enrichment = enrichment_by_track.get(local_track_id)
        geometry = geometry_by_track.get(local_track_id, {})
        colour_predictions = list(getattr(getattr(enrichment, "vehicle_colour", None), "predictions", []) or []) if enrichment is not None else []
        body_predictions = list(getattr(getattr(enrichment, "vehicle_body_type", None), "predictions", []) or []) if enrichment is not None else []
        valid_colour_prediction_count = sum(1 for prediction in colour_predictions if str(getattr(prediction, "label", "UNKNOWN")).upper() not in {"", "UNKNOWN", "NONE"})
        valid_body_prediction_count = sum(1 for prediction in body_predictions if str(getattr(prediction, "label", "UNKNOWN")).upper() not in {"", "UNKNOWN", "NONE"})
        selected_crop_paths = list(getattr(enrichment, "selected_crop_paths", []) or []) if enrichment is not None else []
        florence_call_count = int(getattr(enrichment, "vehicle_attribute_inference_count", 0) or len(colour_predictions)) if enrichment is not None else 0
        failure_stage = "SUCCESS"
        failure_reason = ""
        if raw_crop_counts.get(local_track_id, 0) == 0:
            failure_stage = "NO_TRACK_CROP"
            failure_reason = "no_saved_debug_track_crop"
        if geometry:
            status = str(geometry.get("geometry_status", ""))
            if status == "NEVER_REACHED_ZONE":
                failure_stage = "NEVER_REACHED_CAPTURE_ZONE"
                failure_reason = str(geometry.get("geometry_reason", ""))
            elif status == "TRACK_ENDED_BEFORE_ZONE":
                failure_stage = "TRACKING_FAILED"
                failure_reason = str(geometry.get("geometry_reason", ""))
            elif status == "REACHED_ZONE_NO_CAPTURE":
                failure_stage = "CAPTURE_RETENTION_FAILED"
                failure_reason = str(geometry.get("geometry_reason", ""))
            elif status == "CAPTURED_TOO_SMALL":
                failure_stage = "EVIDENCE_ELIGIBILITY_FAILED"
                failure_reason = str(geometry.get("geometry_reason", ""))
            elif status == "CAPTURED_ELIGIBLE" and not selected_crop_paths:
                failure_stage = "FLORENCE_HANDOFF_FAILED"
                failure_reason = "eligible_crop_not_selected_for_florence"
        if selected_crop_paths and florence_call_count == 0:
            failure_stage = "FLORENCE_HANDOFF_FAILED"
            failure_reason = "selected_crop_not_sent_to_florence"
        if florence_call_count > 0 and valid_colour_prediction_count == 0 and enrichment is not None:
            failure_stage = "AGGREGATION_FAILED" if str(getattr(enrichment.vehicle_colour, "status", "")).lower() == "completed" else "FLORENCE_INFERENCE_FAILED"
            failure_reason = str(getattr(enrichment, "final_colour_reason", "") or getattr(getattr(enrichment, "vehicle_colour", None), "reason", "") or "")
        if valid_colour_prediction_count > 0 and str(getattr(getattr(enrichment, "vehicle_colour", None), "label", "UNKNOWN")).upper() not in {"", "UNKNOWN"}:
            failure_stage = "SUCCESS"
            failure_reason = ""
        rows.append(
            {
                "camera_id": track.camera_id,
                "local_track_id": local_track_id,
                "vehicle_class": str(track.final_class or "UNKNOWN"),
                "detection_count": int(track.observation_count),
                "tracking_observation_count": int(track.observation_count),
                "raw_track_crop_count": int(raw_crop_counts.get(local_track_id, 0)),
                "preferred_crop_count": int(getattr(enrichment, "preferred_crop_count", preferred_crop_counts.get(local_track_id, 0)) if enrichment is not None else preferred_crop_counts.get(local_track_id, 0)),
                "fallback_crop_count": int(getattr(enrichment, "fallback_crop_count", fallback_crop_counts.get(local_track_id, 0)) if enrichment is not None else fallback_crop_counts.get(local_track_id, 0)),
                "selected_colour_crop_count": int(getattr(enrichment, "selected_colour_crop_count", len(selected_crop_paths)) if enrichment is not None else len(selected_crop_paths)),
                "colour_selection_tier": getattr(enrichment, "colour_selection_tier", None) if enrichment is not None else None,
                "capture_zone_entered": bool(geometry.get("entered_zone", False)),
                "capture_zone_candidate_count": int(geometry.get("capture_candidates", 0) or 0),
                "capture_zone_retained_count": int(geometry.get("retained_candidates", 0) or 0),
                "evidence_candidate_count": int(raw_crop_counts.get(local_track_id, 0)),
                "evidence_eligible_count": int(evidence_eligible_counts.get(local_track_id, 0)),
                "florence_eligible_count": int(florence_eligible_counts.get(local_track_id, 0)),
                "florence_selected_count": int(len(selected_crop_paths)),
                "florence_call_count": florence_call_count,
                "valid_colour_prediction_count": int(valid_colour_prediction_count),
                "final_colour": str(getattr(getattr(enrichment, "vehicle_colour", None), "label", "UNKNOWN") if enrichment is not None else "UNKNOWN"),
                "body_type_eligible": getattr(enrichment, "body_type_eligible", None) if enrichment is not None else None,
                "body_type_candidate_crop_count": int(getattr(enrichment, "body_type_candidate_crop_count", 0) if enrichment is not None else 0),
                "body_type_selected_crop_count": int(getattr(enrichment, "body_type_selected_crop_count", 0) if enrichment is not None else 0),
                "body_type_florence_call_count": int(getattr(enrichment, "body_type_florence_call_count", 0) if enrichment is not None else 0),
                "body_type_valid_prediction_count": int(getattr(enrichment, "body_type_valid_prediction_count", valid_body_prediction_count) if enrichment is not None else valid_body_prediction_count),
                "body_type_final": str(getattr(getattr(enrichment, "vehicle_body_type", None), "label", "UNKNOWN") if enrichment is not None else "UNKNOWN"),
                "body_type_failure_reason": getattr(enrichment, "body_type_failure_reason", None) if enrichment is not None else None,
                "failure_stage": failure_stage,
                "failure_reason": failure_reason,
            }
        )
    return rows


def _build_plate_ocr_result_rows(enrichment_results: list[Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result in enrichment_results:
        if not (getattr(result, "plate_detected", False) or getattr(result, "plate_ocr_attempted", False) or getattr(result, "plate_ocr_reason", None)):
            continue
        rows.append(
            {
                "camera_id": result.camera_id,
                "local_track_id": result.local_track_id,
                "frame_index": "",
                "vehicle_crop_path": " | ".join(getattr(result, "selected_crop_paths", []) or []),
                "plate_bbox": getattr(result, "plate_bbox", None),
                "plate_crop_path": getattr(result, "plate_crop_path", None),
                "plate_detection_confidence": getattr(result, "plate_detection_confidence", None),
                "plate_quality_status": getattr(result, "plate_quality_status", None),
                "raw_ocr_response": getattr(result, "plate_ocr_raw_response", None),
                "normalized_plate_text": getattr(result, "plate_text", None),
                "ocr_status": "completed" if getattr(result, "plate_text", None) else "skipped",
                "ocr_reason": getattr(result, "plate_ocr_reason", None),
                "inference_time_ms": "",
                "adapter_loaded": bool(getattr(result, "plate_ocr_attempted", False)),
            }
        )
    return rows


def _write_current_vs_ocr_mukul_artifacts(run_directory: Path, enrichment_results: list[Any]) -> None:
    comparison_rows: list[dict[str, Any]] = []
    manual_rows: list[dict[str, Any]] = []
    for result in enrichment_results:
        payload = getattr(result, "comparison_payload", None)
        if not payload:
            continue
        comparison_row = {
            "camera_id": result.camera_id,
            "local_track_id": result.local_track_id,
            "selected_crop_paths": " | ".join(getattr(result, "selected_crop_paths", []) or []),
            "current_body_type": payload["current"]["body_type_label"],
            "ocr_mukul_body_type": payload["ocr_mukul"]["body_type_label"],
            "current_colour": payload["current"]["colour_label"],
            "ocr_mukul_colour": payload["ocr_mukul"]["colour_label"],
            "current_body_type_raw": " | ".join(payload["current"]["body_type_raw_responses"]),
            "current_colour_raw": " | ".join(payload["current"]["colour_raw_responses"]),
            "ocr_mukul_captions": " | ".join(str(item.get("caption", "")) for item in payload["ocr_mukul"]["captions"]),
            "current_body_type_reason": payload["current"]["body_type_reason"],
            "ocr_mukul_body_type_reason": payload["ocr_mukul"]["body_type_reason"],
            "current_colour_reason": payload["current"]["colour_reason"],
            "ocr_mukul_colour_reason": payload["ocr_mukul"]["colour_reason"],
            "manual_body_type": "",
            "manual_colour": "",
            "review_notes": "",
        }
        comparison_rows.append(comparison_row)
        if payload["current"]["body_type_label"] != payload["ocr_mukul"]["body_type_label"] or payload["current"]["colour_label"] != payload["ocr_mukul"]["colour_label"]:
            manual_rows.append(dict(comparison_row))
    if not comparison_rows:
        return
    output_dir = run_directory.parent.parent / "current_vs_ocr_mukul"
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "comparison.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(comparison_rows[0].keys()))
        writer.writeheader()
        writer.writerows(comparison_rows)
    summary = {
        "tracks_compared": len(comparison_rows),
        "body_type_disagreements": sum(1 for row in comparison_rows if row["current_body_type"] != row["ocr_mukul_body_type"]),
        "colour_disagreements": sum(1 for row in comparison_rows if row["current_colour"] != row["ocr_mukul_colour"]),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_dir / "report.md").write_text(
        "\n".join(
            [
                "# Current vs OCR_MUKUL Report",
                "",
                f"- Tracks compared: `{summary['tracks_compared']}`",
                f"- Body type disagreements: `{summary['body_type_disagreements']}`",
                f"- Colour disagreements: `{summary['colour_disagreements']}`",
            ]
        ),
        encoding="utf-8",
    )
    with (output_dir / "manual_review.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manual_rows[0].keys()) if manual_rows else ["camera_id", "local_track_id", "review_notes"])
        writer.writeheader()
        writer.writerows(manual_rows)
