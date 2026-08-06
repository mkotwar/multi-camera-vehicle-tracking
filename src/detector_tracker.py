from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO

from .models import BBoxQualityDiagnostic, ConfigurationError, Detection, FramePacket, TrackedDetection
from .runtime_device import RuntimeDevice, resolve_runtime_device


EDGE_MODE_A = "A"
EDGE_MODE_B = "B"
EDGE_MODE_C = "C"
EDGE_MODE_LEGACY = "LEGACY"
SUPPORTED_EDGE_MODES = {EDGE_MODE_A, EDGE_MODE_B, EDGE_MODE_C, EDGE_MODE_LEGACY}
DETECTION_BACKEND_LEGACY_CLEAN = "legacy_clean"
DETECTION_BACKEND_OCR_MUKUL = "ocr_mukul"
SUPPORTED_DETECTION_BACKENDS = {DETECTION_BACKEND_LEGACY_CLEAN, DETECTION_BACKEND_OCR_MUKUL}
TRACKER_ISOLATION_MODE_PER_CAMERA = "per_camera"
TRACKER_ISOLATION_MODE_PER_CAMERA_CLASS = "per_camera_class"
SUPPORTED_TRACKER_ISOLATION_MODES = {TRACKER_ISOLATION_MODE_PER_CAMERA, TRACKER_ISOLATION_MODE_PER_CAMERA_CLASS}
TRACKING_BACKEND_SUPERVISION_BYTETRACK = "supervision_bytetrack"
TRACKING_BACKEND_OCR_MUKUL_SUPERVISION_BYTETRACK = "ocr_mukul_supervision_bytetrack"
SUPPORTED_TRACKING_BACKENDS = {TRACKING_BACKEND_SUPERVISION_BYTETRACK, TRACKING_BACKEND_OCR_MUKUL_SUPERVISION_BYTETRACK}
OCR_MUKUL_DEFAULT_ALLOWED_CLASS_IDS = tuple(range(8))

ALIAS_TO_CANONICAL_CLASS = {
    "3wheeler": "3wheeler",
    "3 wheeler": "3wheeler",
    "3-wheeler": "3wheeler",
    "three wheeler": "3wheeler",
    "three-wheeler": "3wheeler",
    "three_wheeler": "3wheeler",
    "auto": "3wheeler",
    "auto rickshaw": "3wheeler",
    "auto-rickshaw": "3wheeler",
    "auto_rickshaw": "3wheeler",
    "rickshaw": "3wheeler",
    "car": "car",
    "truck": "truck",
    "bus": "bus",
    "motorcycle": "motorcycle",
    "bike": "motorcycle",
    "motorbike": "motorcycle",
    "2wheeler": "motorcycle",
    "2 wheeler": "motorcycle",
    "2-wheeler": "motorcycle",
    "4wheeler": "car",
    "4 wheeler": "car",
    "4-wheeler": "car",
}


@dataclass(slots=True, frozen=True)
class BBoxQualityProfile:
    minimum_width_pixels: float
    minimum_height_pixels: float
    minimum_area_ratio: float
    maximum_area_ratio: float
    minimum_aspect_ratio: float
    maximum_aspect_ratio: float
    edge_margin_pixels: float
    edge_mode: str


@dataclass(slots=True)
class DetectorTrackerResult:
    detections: list[Detection]
    tracked_detections: list[TrackedDetection]
    bbox_quality_diagnostics: list[BBoxQualityDiagnostic]
    detected_frame: np.ndarray
    tracked_frame: np.ndarray
    inference_time_ms: float


class VehicleDetectorTracker:
    def __init__(
        self,
        config: dict[str, Any],
        logger: logging.Logger,
        *,
        model_loader: Callable[[str], Any] | None = None,
        tracker_factory: Callable[..., Any] | None = None,
    ) -> None:
        self.config = dict(config)
        self.logger = logger
        detection_config = dict(self.config.get("detection", {}) or {})
        tracking_config = dict(self.config.get("tracking", {}) or {})
        if not detection_config:
            raise ConfigurationError("Missing 'detection' configuration.")
        if not tracking_config:
            raise ConfigurationError("Missing 'tracking' configuration.")
        self.detection_backend = str(detection_config.get("backend", DETECTION_BACKEND_OCR_MUKUL)).strip().lower() or DETECTION_BACKEND_OCR_MUKUL
        if self.detection_backend not in SUPPORTED_DETECTION_BACKENDS:
            raise ConfigurationError(f"Unsupported detection.backend value: {self.detection_backend}")
        self.model_path = self._resolve_model_path(detection_config.get("model_path"))
        self.runtime_device_info: RuntimeDevice = resolve_runtime_device(
            detection_config.get("device", "auto"),
            detection_config.get("dtype", "auto"),
        )
        self.configured_device = self.runtime_device_info.configured_device
        self.configured_dtype = self.runtime_device_info.configured_dtype
        self.device = self.runtime_device_info.resolved_device
        self.dtype = self.runtime_device_info.resolved_dtype
        self.confidence_threshold = float(
            detection_config.get(
                "confidence_threshold",
                0.2 if self.detection_backend == DETECTION_BACKEND_OCR_MUKUL else 0.38,
            )
        )
        self.iou_threshold = float(detection_config.get("iou_threshold", 0.45))
        self.image_size = int(
            detection_config.get(
                "image_size",
                1024 if self.detection_backend == DETECTION_BACKEND_OCR_MUKUL else 640,
            )
        )
        self.agnostic_nms = bool(detection_config.get("agnostic_nms", False))
        self.allowed_classes = [self._normalize_class_name(name) for name in detection_config.get("allowed_classes", []) if str(name).strip()]
        raw_allowed_class_ids = detection_config.get("allowed_class_ids", OCR_MUKUL_DEFAULT_ALLOWED_CLASS_IDS)
        self.allowed_class_ids = tuple(int(item) for item in raw_allowed_class_ids)
        if self.detection_backend == DETECTION_BACKEND_LEGACY_CLEAN and not self.allowed_classes:
            raise ConfigurationError("detection.allowed_classes must not be empty for legacy_clean backend.")
        self.bbox_quality_enabled = bool(dict(detection_config.get("bbox_quality", {}) or {}).get("enabled", False))
        if self.detection_backend == DETECTION_BACKEND_OCR_MUKUL:
            self.bbox_quality_enabled = False
        self._default_bbox_quality_profile, self._class_bbox_quality_profiles = self._parse_bbox_quality_profiles(detection_config)
        self.tracking_backend = (
            str(tracking_config.get("backend", TRACKING_BACKEND_OCR_MUKUL_SUPERVISION_BYTETRACK)).strip().lower()
            or TRACKING_BACKEND_OCR_MUKUL_SUPERVISION_BYTETRACK
        )
        if self.tracking_backend not in SUPPORTED_TRACKING_BACKENDS:
            raise ConfigurationError(f"Unsupported tracking backend: {self.tracking_backend}")
        default_track_activation_threshold = 0.3 if self.tracking_backend == TRACKING_BACKEND_OCR_MUKUL_SUPERVISION_BYTETRACK else 0.15
        default_lost_track_buffer = 40 if self.tracking_backend == TRACKING_BACKEND_OCR_MUKUL_SUPERVISION_BYTETRACK else 30
        default_minimum_matching_threshold = 0.6 if self.tracking_backend == TRACKING_BACKEND_OCR_MUKUL_SUPERVISION_BYTETRACK else 0.80
        default_minimum_consecutive_frames = 3 if self.tracking_backend == TRACKING_BACKEND_OCR_MUKUL_SUPERVISION_BYTETRACK else 1
        self.track_activation_threshold = float(tracking_config.get("track_activation_threshold", default_track_activation_threshold))
        self.lost_track_buffer = int(tracking_config.get("lost_track_buffer", default_lost_track_buffer))
        self.minimum_matching_threshold = float(tracking_config.get("minimum_matching_threshold", default_minimum_matching_threshold))
        self.minimum_consecutive_frames = int(tracking_config.get("minimum_consecutive_frames", default_minimum_consecutive_frames))
        self.isolation_mode = str(tracking_config.get("isolation_mode", TRACKER_ISOLATION_MODE_PER_CAMERA)).strip()
        self.supported_isolation_modes = [str(item).strip() for item in tracking_config.get("supported_isolation_modes", []) if str(item).strip()]
        if not self.supported_isolation_modes:
            self.supported_isolation_modes = [TRACKER_ISOLATION_MODE_PER_CAMERA, TRACKER_ISOLATION_MODE_PER_CAMERA_CLASS]
        if self.isolation_mode not in self.supported_isolation_modes or self.isolation_mode not in SUPPORTED_TRACKER_ISOLATION_MODES:
            raise ConfigurationError(f"Unsupported tracking.isolation_mode value: {self.isolation_mode}")
        if self.tracking_backend == TRACKING_BACKEND_OCR_MUKUL_SUPERVISION_BYTETRACK and self.isolation_mode != TRACKER_ISOLATION_MODE_PER_CAMERA:
            raise ConfigurationError("ocr_mukul_supervision_bytetrack only supports tracking.isolation_mode=per_camera.")
        visualization_config = dict(self.config.get("visualization", {}) or {})
        self.show_rejected_boxes = bool(visualization_config.get("show_rejected_boxes", False))
        self._model_loader = model_loader or YOLO
        self._tracker_factory = tracker_factory or self._create_tracker
        self._model: Any | None = None
        self._model_class_names: dict[int, str] = {}
        self._allowed_model_class_ids: set[int] = set()
        self._trackers: dict[str | tuple[str, str], Any] = {}
        self._metrics: dict[str, Any] = {
            "model_load_count": 0,
            "tracker_instance_count": 0,
            "tracker_instances_created_total": 0,
            "tracker_camera_ids": [],
            "tracker_keys": [],
            "trackers_created_by_camera": {},
            "trackers_created_by_camera_namespace": {},
            "inference_times_ms": [],
            "inference_errors": [],
        }
        self._load_model_once()
        self.logger.info("Detector backend: %s", self.detection_backend)
        self.logger.info("Tracker backend: %s", self.tracking_backend)

    @property
    def metrics(self) -> dict[str, Any]:
        self._refresh_tracker_metrics()
        return {
            **self._metrics,
            "tracker_camera_ids": list(self._metrics["tracker_camera_ids"]),
            "tracker_keys": list(self._metrics["tracker_keys"]),
            "trackers_created_by_camera": dict(self._metrics["trackers_created_by_camera"]),
            "trackers_created_by_camera_namespace": dict(self._metrics["trackers_created_by_camera_namespace"]),
            "inference_times_ms": list(self._metrics["inference_times_ms"]),
            "inference_errors": list(self._metrics["inference_errors"]),
        }

    def get_bbox_quality_profile(self, class_name: str) -> BBoxQualityProfile:
        normalized = self._normalize_class_name(class_name)
        return self._class_bbox_quality_profiles.get(normalized, self._default_bbox_quality_profile)

    def infer_yolo_detections(self, packet: FramePacket) -> list[Detection]:
        if self.detection_backend == DETECTION_BACKEND_OCR_MUKUL:
            raw_result = self._infer_ocr_mukul_result(packet)
            return self._convert_ocr_mukul_result(raw_result)
        try:
            raw_result = self._model.predict(
                source=packet.frame,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                imgsz=self.image_size,
                device=self.runtime_device_info.yolo_device,
                half=self.runtime_device_info.yolo_half,
                agnostic_nms=self.agnostic_nms,
                verbose=False,
            )[0]
        except Exception as exc:
            self._metrics["inference_errors"].append(
                {
                    "camera_id": packet.camera_id,
                    "frame_number": packet.frame_number,
                    "model_path": str(self.model_path),
                    "error_type": exc.__class__.__name__,
                    "error_message": str(exc),
                }
            )
            self.logger.error(
                "Inference failed camera=%s frame=%s model=%s error_type=%s error=%s",
                packet.camera_id,
                packet.frame_number,
                self.model_path,
                exc.__class__.__name__,
                exc,
            )
            raise
        return self._convert_yolo_result(raw_result)

    def filter_detections(
        self,
        packet: FramePacket,
        detections: list[Detection],
    ) -> tuple[list[Detection], list[BBoxQualityDiagnostic]]:
        frame_height, frame_width = packet.frame.shape[:2]
        frame_area = float(frame_width * frame_height) if frame_width > 0 and frame_height > 0 else 0.0
        accepted_detections: list[Detection] = []
        diagnostics: list[BBoxQualityDiagnostic] = []
        for detection in detections:
            profile = self.get_bbox_quality_profile(detection.class_name)
            diagnostic = self._build_bbox_quality_diagnostic(
                packet=packet,
                detection=detection,
                frame_width=frame_width,
                frame_height=frame_height,
                frame_area=frame_area,
                profile=profile,
            )
            diagnostics.append(diagnostic)
            if diagnostic.accepted_by_bbox_quality:
                accepted_detections.append(detection)
        return accepted_detections, diagnostics

    def track_detections(self, packet: FramePacket, detections: list[Detection], raw_result: Any | None = None) -> list[TrackedDetection]:
        if self.tracking_backend == TRACKING_BACKEND_OCR_MUKUL_SUPERVISION_BYTETRACK:
            supervision_detections = self._to_ocr_mukul_supervision_detections(raw_result, detections)
            tracker = self._get_or_create_tracker(packet.camera_id, packet.source_fps, tracker_namespace="camera")
            tracked = tracker.update_with_detections(supervision_detections)
            return self._to_tracked_detections(packet, detections, tracked, tracker_namespace="camera")
        if self.isolation_mode == TRACKER_ISOLATION_MODE_PER_CAMERA:
            supervision_detections = self._to_supervision_detections(detections)
            tracker = self._get_or_create_tracker(packet.camera_id, packet.source_fps, tracker_namespace="camera")
            tracked = tracker.update_with_detections(supervision_detections)
            return self._to_tracked_detections(packet, detections, tracked, tracker_namespace="camera")

        grouped_detections = self._group_detections_by_normalized_class(detections)
        tracked_results: list[TrackedDetection] = []
        existing_namespaces = self._get_existing_class_tracker_namespaces(packet.camera_id)
        namespaces_to_update = sorted(existing_namespaces | set(grouped_detections))
        for tracker_namespace in namespaces_to_update:
            tracker = self._get_or_create_tracker(packet.camera_id, packet.source_fps, tracker_namespace=tracker_namespace)
            class_detections = grouped_detections.get(tracker_namespace, [])
            tracked = tracker.update_with_detections(self._to_supervision_detections(class_detections))
            tracked_results.extend(self._to_tracked_detections(packet, class_detections, tracked, tracker_namespace=tracker_namespace))
        return tracked_results

    def process_frame(self, packet: FramePacket) -> DetectorTrackerResult:
        started_at = time.perf_counter()
        raw_result: Any | None = None
        if self.detection_backend == DETECTION_BACKEND_OCR_MUKUL:
            raw_result = self._infer_ocr_mukul_result(packet)
            raw_detections = self._convert_ocr_mukul_result(raw_result)
        else:
            raw_detections = self.infer_yolo_detections(packet)
        accepted_detections, bbox_quality_diagnostics = self.filter_detections(packet, raw_detections)
        tracked_detections = self.track_detections(packet, accepted_detections, raw_result=raw_result)
        inference_time_ms = (time.perf_counter() - started_at) * 1000.0
        self._metrics["inference_times_ms"].append(inference_time_ms)
        rejected_detection_count = len([item for item in bbox_quality_diagnostics if not item.accepted_by_bbox_quality])
        self.logger.debug(
            "camera=%s frame=%s raw_detections=%s accepted_detections=%s rejected_detections=%s tracked_observations=%s tracker_ids=%s inference_ms=%.3f",
            packet.camera_id,
            packet.frame_number,
            len(raw_detections),
            len(accepted_detections),
            rejected_detection_count,
            len(tracked_detections),
            [item.tracker_id for item in tracked_detections],
            inference_time_ms,
        )
        return DetectorTrackerResult(
            detections=accepted_detections,
            tracked_detections=tracked_detections,
            bbox_quality_diagnostics=bbox_quality_diagnostics,
            detected_frame=self.annotate_detected_frame(packet.frame, accepted_detections, bbox_quality_diagnostics),
            tracked_frame=self.annotate_tracked_frame(packet.frame, packet.camera_id, tracked_detections),
            inference_time_ms=inference_time_ms,
        )

    def reset_camera(self, camera_id: str) -> None:
        for key in list(self._trackers):
            if self._tracker_key_camera_id(key) == camera_id:
                self._trackers.pop(key, None)
        self._metrics["tracker_instance_count"] = len(self._trackers)
        self._refresh_tracker_metrics()

    def reset_all(self) -> None:
        self._trackers.clear()
        self._metrics["tracker_instance_count"] = 0
        self._metrics["tracker_camera_ids"] = []
        self._metrics["tracker_keys"] = []

    def annotate_detected_frame(
        self,
        frame: np.ndarray,
        detections: list[Detection],
        bbox_quality_diagnostics: list[BBoxQualityDiagnostic],
    ) -> np.ndarray:
        annotated = frame.copy()
        for detection in detections:
            x1, y1, x2, y2 = [int(round(value)) for value in detection.bbox_xyxy]
            label = self.build_detected_label(detection)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 200, 255), 2)
            cv2.putText(annotated, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 2)
        if self.show_rejected_boxes:
            for diagnostic in bbox_quality_diagnostics:
                if diagnostic.accepted_by_bbox_quality:
                    continue
                x1, y1, x2, y2 = [int(round(value)) for value in diagnostic.bbox_xyxy]
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (40, 40, 255), 2)
                cv2.putText(
                    annotated,
                    f"REJECTED {diagnostic.rejection_reason}",
                    (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (40, 40, 255),
                    2,
                )
        return annotated

    def annotate_tracked_frame(self, frame: np.ndarray, camera_id: str, tracked_detections: list[TrackedDetection]) -> np.ndarray:
        annotated = frame.copy()
        for tracked in tracked_detections:
            x1, y1, x2, y2 = [int(round(value)) for value in tracked.bbox_xyxy]
            label = self.build_tracked_label(camera_id, tracked)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (80, 220, 80), 2)
            cv2.putText(annotated, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 220, 80), 2)
        return annotated

    def build_detected_label(self, detection: Detection) -> str:
        return f"{self._normalize_class_name(detection.class_name).upper()} {detection.confidence:.2f}"

    def build_tracked_label(self, camera_id: str, tracked: TrackedDetection) -> str:
        if tracked.tracker_namespace == "camera":
            namespace_text = f"TRACK_{tracked.tracker_id}"
        else:
            namespace_text = f"{tracked.tracker_namespace.upper()} | TRACK_{tracked.tracker_id}"
        return f"{camera_id} | {namespace_text} | {self._normalize_class_name(tracked.raw_class_name).upper()} | {tracked.confidence:.2f}"

    def _load_model_once(self) -> None:
        if self._model is not None:
            return
        if not self.model_path.exists():
            raise ConfigurationError(f"Model file not found: {self.model_path}")
        self._model = self._model_loader(str(self.model_path))
        self._metrics["model_load_count"] += 1
        self._model_class_names = self._extract_model_class_names(self._model)
        if self.detection_backend == DETECTION_BACKEND_OCR_MUKUL:
            self._model_class_names[2] = "4Wheeler"
            self._model_class_names[3] = "2Wheeler"
        self._allowed_model_class_ids = self._resolve_allowed_model_class_ids()
        if self.detection_backend == DETECTION_BACKEND_OCR_MUKUL and not self.allowed_classes:
            self.allowed_classes = sorted(
                {
                    self._normalize_class_name(self._model_class_names.get(class_id, str(class_id)))
                    for class_id in self._allowed_model_class_ids
                }
            )
        self.logger.info(
            "YOLO model loaded device=%s precision=%s",
            self.device,
            self.dtype,
        )
        self.logger.info(
            "Model loaded model_path=%s device=%s dtype=%s allowed_classes=%s bbox_quality_enabled=%s",
            self.model_path,
            self.device,
            self.dtype,
            self.allowed_classes,
            self.bbox_quality_enabled,
        )

    def _resolve_model_path(self, raw_path: Any) -> Path:
        if raw_path in (None, ""):
            raise ConfigurationError("detection.model_path is required.")
        candidate = Path(str(raw_path)).expanduser()
        if not candidate.is_absolute():
            candidate = (Path(__file__).resolve().parents[1] / candidate).resolve()
        else:
            candidate = candidate.resolve()
        if not candidate.exists():
            raise ConfigurationError(f"Resolved model path does not exist: {candidate}")
        return candidate

    def _extract_model_class_names(self, model: Any) -> dict[int, str]:
        names = getattr(model, "names", {})
        if isinstance(names, dict):
            return {int(key): str(value) for key, value in names.items()}
        if isinstance(names, (list, tuple)):
            return {index: str(value) for index, value in enumerate(names)}
        raise ConfigurationError("Unable to extract class names from YOLO model.")

    def _normalize_class_name(self, value: Any) -> str:
        normalized = " ".join(str(value).strip().lower().replace("_", " ").replace("-", " ").split())
        return ALIAS_TO_CANONICAL_CLASS.get(normalized, normalized)

    def _resolve_allowed_model_class_ids(self) -> set[int]:
        if self.detection_backend == DETECTION_BACKEND_OCR_MUKUL:
            available = set(self._model_class_names)
            missing = [class_id for class_id in self.allowed_class_ids if class_id not in available]
            if missing:
                self.logger.warning(
                    "OCR_MUKUL allowed_class_ids missing from model.names missing=%s available=%s using_intersection_only=True",
                    missing,
                    sorted(available),
                )
            resolved = {int(class_id) for class_id in self.allowed_class_ids if class_id in available}
            if not resolved:
                raise ConfigurationError("Configured detection.allowed_class_ids do not overlap with model.names.")
            return resolved
        normalized_to_ids: dict[str, list[int]] = {}
        for class_id, class_name in self._model_class_names.items():
            normalized = self._normalize_class_name(class_name)
            normalized_to_ids.setdefault(normalized, []).append(class_id)
        missing = [class_name for class_name in self.allowed_classes if class_name not in normalized_to_ids]
        if missing:
            available = sorted(normalized_to_ids)
            raise ConfigurationError(
                f"Configured allowed classes do not match YOLO model classes. Missing={missing}; available={available}"
            )
        allowed_ids: set[int] = set()
        for class_name in self.allowed_classes:
            allowed_ids.update(normalized_to_ids[class_name])
        return allowed_ids

    def _parse_bbox_quality_profiles(
        self,
        detection_config: dict[str, Any],
    ) -> tuple[BBoxQualityProfile, dict[str, BBoxQualityProfile]]:
        bbox_quality_config = dict(detection_config.get("bbox_quality", {}) or {})
        default_payload: dict[str, Any]
        classes_payload = bbox_quality_config.get("classes", {})
        has_profiles = "default" in bbox_quality_config or "classes" in bbox_quality_config
        if has_profiles:
            default_payload = dict(bbox_quality_config.get("default", {}) or {})
        else:
            default_payload = dict(bbox_quality_config)
            classes_payload = {}
        default_profile = self._coerce_profile(default_payload, fallback=None, legacy_flat=not has_profiles)
        class_profiles: dict[str, BBoxQualityProfile] = {}
        if classes_payload:
            if not isinstance(classes_payload, dict):
                raise ConfigurationError("detection.bbox_quality.classes must be a mapping when provided.")
            for raw_class_name, raw_profile in classes_payload.items():
                if not isinstance(raw_profile, dict):
                    raise ConfigurationError(f"detection.bbox_quality.classes.{raw_class_name} must be a mapping.")
                normalized_class_name = self._normalize_class_name(raw_class_name)
                class_profiles[normalized_class_name] = self._coerce_profile(raw_profile, fallback=default_profile, legacy_flat=False)
        return default_profile, class_profiles

    def _coerce_profile(
        self,
        payload: dict[str, Any],
        *,
        fallback: BBoxQualityProfile | None,
        legacy_flat: bool,
    ) -> BBoxQualityProfile:
        base = fallback or BBoxQualityProfile(
            minimum_width_pixels=60.0,
            minimum_height_pixels=60.0,
            minimum_area_ratio=0.005,
            maximum_area_ratio=0.90,
            minimum_aspect_ratio=0.30,
            maximum_aspect_ratio=4.50,
            edge_margin_pixels=8.0,
            edge_mode=EDGE_MODE_C,
        )
        raw_edge_mode = payload.get("edge_mode")
        if raw_edge_mode is None:
            if "reject_edge_truncated" in payload:
                raw_edge_mode = EDGE_MODE_LEGACY if legacy_flat and bool(payload.get("reject_edge_truncated", True)) else EDGE_MODE_A
                if not legacy_flat and bool(payload.get("reject_edge_truncated", True)):
                    raw_edge_mode = EDGE_MODE_C
            else:
                raw_edge_mode = base.edge_mode
        edge_mode = str(raw_edge_mode).strip().upper() or base.edge_mode
        if edge_mode not in SUPPORTED_EDGE_MODES:
            raise ConfigurationError(f"Unsupported bbox edge_mode: {raw_edge_mode}")
        profile = BBoxQualityProfile(
            minimum_width_pixels=float(payload.get("minimum_width_pixels", base.minimum_width_pixels)),
            minimum_height_pixels=float(payload.get("minimum_height_pixels", base.minimum_height_pixels)),
            minimum_area_ratio=float(payload.get("minimum_area_ratio", base.minimum_area_ratio)),
            maximum_area_ratio=float(payload.get("maximum_area_ratio", base.maximum_area_ratio)),
            minimum_aspect_ratio=float(payload.get("minimum_aspect_ratio", base.minimum_aspect_ratio)),
            maximum_aspect_ratio=float(payload.get("maximum_aspect_ratio", base.maximum_aspect_ratio)),
            edge_margin_pixels=float(payload.get("edge_margin_pixels", base.edge_margin_pixels)),
            edge_mode=edge_mode,
        )
        self._validate_profile(profile)
        return profile

    def _validate_profile(self, profile: BBoxQualityProfile) -> None:
        if profile.minimum_width_pixels < 0.0:
            raise ConfigurationError("bbox minimum_width_pixels must be at least 0.")
        if profile.minimum_height_pixels < 0.0:
            raise ConfigurationError("bbox minimum_height_pixels must be at least 0.")
        if profile.minimum_area_ratio < 0.0 or profile.maximum_area_ratio < 0.0:
            raise ConfigurationError("bbox area ratios must be at least 0.")
        if profile.maximum_area_ratio < profile.minimum_area_ratio:
            raise ConfigurationError("bbox maximum_area_ratio must be greater than or equal to minimum_area_ratio.")
        if profile.minimum_aspect_ratio <= 0.0 or profile.maximum_aspect_ratio <= 0.0:
            raise ConfigurationError("bbox aspect ratios must be positive.")
        if profile.maximum_aspect_ratio < profile.minimum_aspect_ratio:
            raise ConfigurationError("bbox maximum_aspect_ratio must be greater than or equal to minimum_aspect_ratio.")
        if profile.edge_margin_pixels < 0.0:
            raise ConfigurationError("bbox edge_margin_pixels must be at least 0.")

    def _convert_yolo_result(self, result: Any) -> list[Detection]:
        boxes = getattr(result, "boxes", None)
        if boxes is None or getattr(boxes, "xyxy", None) is None:
            return []
        xyxy_values = boxes.xyxy.tolist()
        class_values = boxes.cls.tolist() if getattr(boxes, "cls", None) is not None else []
        confidence_values = boxes.conf.tolist() if getattr(boxes, "conf", None) is not None else []
        detections: list[Detection] = []
        for index, raw_box in enumerate(xyxy_values):
            class_id = int(class_values[index]) if index < len(class_values) else -1
            if class_id not in self._allowed_model_class_ids:
                continue
            confidence = float(confidence_values[index]) if index < len(confidence_values) else 0.0
            x1, y1, x2, y2 = [float(value) for value in raw_box[:4]]
            if not np.isfinite([x1, y1, x2, y2]).all():
                continue
            if x2 <= x1 or y2 <= y1:
                continue
            detections.append(
                Detection(
                    bbox_xyxy=(x1, y1, x2, y2),
                    confidence=confidence,
                    class_id=class_id,
                    class_name=self._normalize_class_name(self._model_class_names.get(class_id, str(class_id))),
                )
            )
        return detections

    def _to_supervision_detections(self, detections: list[Detection]) -> sv.Detections:
        if not detections:
            empty = sv.Detections.empty()
            empty.tracker_id = np.array([], dtype=int)
            return empty
        xyxy = np.asarray([detection.bbox_xyxy for detection in detections], dtype=np.float32)
        confidence = np.asarray([detection.confidence for detection in detections], dtype=np.float32)
        class_id = np.asarray([detection.class_id for detection in detections], dtype=np.int32)
        return sv.Detections(xyxy=xyxy, confidence=confidence, class_id=class_id)

    def _group_detections_by_normalized_class(self, detections: list[Detection]) -> dict[str, list[Detection]]:
        grouped: dict[str, list[Detection]] = {}
        for detection in detections:
            normalized_class_name = self._normalize_class_name(detection.class_name)
            grouped.setdefault(normalized_class_name, []).append(detection)
        return grouped

    def _build_bbox_quality_diagnostic(
        self,
        *,
        packet: FramePacket,
        detection: Detection,
        frame_width: int,
        frame_height: int,
        frame_area: float,
        profile: BBoxQualityProfile,
    ) -> BBoxQualityDiagnostic:
        x1, y1, x2, y2 = detection.bbox_xyxy
        bbox_width = float(x2 - x1)
        bbox_height = float(y2 - y1)
        bbox_area = float(bbox_width * bbox_height)
        width_ratio = float(bbox_width / frame_width) if frame_width > 0 else 0.0
        height_ratio = float(bbox_height / frame_height) if frame_height > 0 else 0.0
        area_ratio = float(bbox_area / frame_area) if frame_area > 0.0 else 0.0
        aspect_ratio = float(bbox_width / bbox_height) if bbox_height > 0.0 else float("inf")
        touches_left_edge = bool(x1 <= profile.edge_margin_pixels)
        touches_right_edge = bool(x2 >= frame_width - profile.edge_margin_pixels)
        touches_top_edge = bool(y1 <= profile.edge_margin_pixels)
        touches_bottom_edge = bool(y2 >= frame_height - profile.edge_margin_pixels)
        touches_edge = bool(touches_left_edge or touches_right_edge or touches_top_edge or touches_bottom_edge)
        rejection_reason: str | None = None
        if self.bbox_quality_enabled:
            if bbox_width < profile.minimum_width_pixels:
                rejection_reason = "BBOX_TOO_NARROW"
            elif bbox_height < profile.minimum_height_pixels:
                rejection_reason = "BBOX_TOO_SHORT"
            elif area_ratio < profile.minimum_area_ratio:
                rejection_reason = "BBOX_TOO_SMALL"
            elif area_ratio > profile.maximum_area_ratio:
                rejection_reason = "BBOX_TOO_LARGE"
            elif aspect_ratio < profile.minimum_aspect_ratio:
                rejection_reason = "ASPECT_RATIO_TOO_LOW"
            elif aspect_ratio > profile.maximum_aspect_ratio:
                rejection_reason = "ASPECT_RATIO_TOO_HIGH"
            elif self._edge_mode_rejects(profile, touches_edge, area_ratio, bbox_width, bbox_height):
                rejection_reason = "EDGE_TRUNCATED"
        return BBoxQualityDiagnostic(
            camera_id=packet.camera_id,
            frame_number=packet.frame_number,
            timestamp_seconds=packet.timestamp_seconds,
            class_name=detection.class_name,
            normalized_class_name=self._normalize_class_name(detection.class_name),
            confidence=detection.confidence,
            bbox_xyxy=detection.bbox_xyxy,
            bbox_width=bbox_width,
            bbox_height=bbox_height,
            bbox_area=bbox_area,
            frame_width=frame_width,
            frame_height=frame_height,
            width_ratio=width_ratio,
            height_ratio=height_ratio,
            area_ratio=area_ratio,
            aspect_ratio=aspect_ratio,
            touches_edge=touches_edge,
            touches_left_edge=touches_left_edge,
            touches_right_edge=touches_right_edge,
            touches_top_edge=touches_top_edge,
            touches_bottom_edge=touches_bottom_edge,
            accepted_by_bbox_quality=rejection_reason is None,
            rejection_reason=rejection_reason,
        )

    def _edge_mode_rejects(
        self,
        profile: BBoxQualityProfile,
        touches_edge: bool,
        area_ratio: float,
        bbox_width: float,
        bbox_height: float,
    ) -> bool:
        if not touches_edge:
            return False
        if profile.edge_mode == EDGE_MODE_A:
            return False
        if profile.edge_mode == EDGE_MODE_B:
            return area_ratio < profile.minimum_area_ratio
        if profile.edge_mode == EDGE_MODE_C:
            return bbox_width < profile.minimum_width_pixels or bbox_height < profile.minimum_height_pixels
        if profile.edge_mode == EDGE_MODE_LEGACY:
            return True
        raise ConfigurationError(f"Unsupported bbox edge_mode: {profile.edge_mode}")

    def _get_or_create_tracker(self, camera_id: str, source_fps: float, *, tracker_namespace: str) -> Any:
        tracker_key = self._build_tracker_key(camera_id, tracker_namespace)
        tracker = self._trackers.get(tracker_key)
        if tracker is None:
            tracker = self._tracker_factory(frame_rate=float(source_fps or 30.0))
            self._trackers[tracker_key] = tracker
            self._metrics["tracker_instance_count"] = len(self._trackers)
            self._metrics["tracker_instances_created_total"] += 1
            self._metrics["trackers_created_by_camera"][camera_id] = self._metrics["trackers_created_by_camera"].get(camera_id, 0) + 1
            formatted_namespace_key = f"{camera_id}:{tracker_namespace}"
            self._metrics["trackers_created_by_camera_namespace"][formatted_namespace_key] = (
                self._metrics["trackers_created_by_camera_namespace"].get(formatted_namespace_key, 0) + 1
            )
            self._refresh_tracker_metrics()
            self.logger.info(
                "Tracker created for camera camera_id=%s tracker_namespace=%s frame_rate=%.3f isolation_mode=%s",
                camera_id,
                tracker_namespace,
                float(source_fps or 30.0),
                self.isolation_mode,
            )
        return tracker

    def _build_tracker_key(self, camera_id: str, tracker_namespace: str) -> str | tuple[str, str]:
        if self.isolation_mode == TRACKER_ISOLATION_MODE_PER_CAMERA:
            return camera_id
        return (camera_id, tracker_namespace)

    def _tracker_key_camera_id(self, key: str | tuple[str, str]) -> str:
        return key[0] if isinstance(key, tuple) else key

    def _tracker_key_namespace(self, key: str | tuple[str, str]) -> str:
        return key[1] if isinstance(key, tuple) else "camera"

    def _get_existing_class_tracker_namespaces(self, camera_id: str) -> set[str]:
        return {
            self._tracker_key_namespace(key)
            for key in self._trackers
            if isinstance(key, tuple) and self._tracker_key_camera_id(key) == camera_id
        }

    def _refresh_tracker_metrics(self) -> None:
        self._metrics["tracker_instance_count"] = len(self._trackers)
        self._metrics["tracker_camera_ids"] = sorted({self._tracker_key_camera_id(key) for key in self._trackers})
        self._metrics["tracker_keys"] = sorted(self._format_tracker_key(key) for key in self._trackers)

    def _format_tracker_key(self, key: str | tuple[str, str]) -> str:
        if isinstance(key, tuple):
            return f"{key[0]}:{key[1]}"
        return key

    def _create_tracker(self, *, frame_rate: float) -> Any:
        if self.tracking_backend == TRACKING_BACKEND_OCR_MUKUL_SUPERVISION_BYTETRACK:
            return sv.ByteTrack(
                lost_track_buffer=self.lost_track_buffer,
                track_activation_threshold=self.track_activation_threshold,
                minimum_matching_threshold=self.minimum_matching_threshold,
                minimum_consecutive_frames=self.minimum_consecutive_frames,
            )
        if self.tracking_backend != TRACKING_BACKEND_SUPERVISION_BYTETRACK:
            raise ConfigurationError(f"Unsupported tracking backend: {self.tracking_backend}")
        return sv.ByteTrack(
            track_activation_threshold=self.track_activation_threshold,
            lost_track_buffer=self.lost_track_buffer,
            minimum_matching_threshold=self.minimum_matching_threshold,
            frame_rate=frame_rate,
            minimum_consecutive_frames=self.minimum_consecutive_frames,
        )

    def _infer_ocr_mukul_result(self, packet: FramePacket) -> Any:
        try:
            if callable(self._model):
                return self._model(
                    packet.frame,
                    conf=self.confidence_threshold,
                    imgsz=self.image_size,
                    device=self.runtime_device_info.yolo_device,
                    half=self.runtime_device_info.yolo_half,
                    verbose=False,
                )[0]
            return self._model.predict(
                source=packet.frame,
                conf=self.confidence_threshold,
                imgsz=self.image_size,
                device=self.runtime_device_info.yolo_device,
                half=self.runtime_device_info.yolo_half,
                verbose=False,
            )[0]
        except Exception as exc:
            self._metrics["inference_errors"].append(
                {
                    "camera_id": packet.camera_id,
                    "frame_number": packet.frame_number,
                    "model_path": str(self.model_path),
                    "error_type": exc.__class__.__name__,
                    "error_message": str(exc),
                }
            )
            self.logger.error(
                "OCR_MUKUL inference failed camera=%s frame=%s model=%s error_type=%s error=%s",
                packet.camera_id,
                packet.frame_number,
                self.model_path,
                exc.__class__.__name__,
                exc,
            )
            raise

    def _convert_ocr_mukul_result(self, result: Any) -> list[Detection]:
        detections = self._convert_yolo_result(result)
        return [item for item in detections if item.class_id in self._allowed_model_class_ids]

    def _to_ocr_mukul_supervision_detections(self, raw_result: Any | None, detections: list[Detection]) -> sv.Detections:
        if raw_result is None:
            return self._to_supervision_detections(detections)
        try:
            supervision_detections = sv.Detections.from_ultralytics(raw_result)
        except Exception:
            return self._to_supervision_detections(detections)
        if len(supervision_detections.xyxy) == 0:
            return self._to_supervision_detections(detections)
        allowed_ids = np.asarray(sorted(self._allowed_model_class_ids), dtype=np.int32)
        class_ids = np.asarray(supervision_detections.class_id) if getattr(supervision_detections, "class_id", None) is not None else np.array([], dtype=np.int32)
        if len(class_ids) == 0:
            return self._to_supervision_detections(detections)
        mask = np.isin(class_ids.astype(np.int32), allowed_ids)
        filtered = supervision_detections[mask]
        if len(filtered.xyxy) != len(detections):
            return self._to_supervision_detections(detections)
        return filtered

    def _to_tracked_detections(
        self,
        packet: FramePacket,
        detections: list[Detection],
        tracked: sv.Detections,
        *,
        tracker_namespace: str,
    ) -> list[TrackedDetection]:
        tracker_ids = list(tracked.tracker_id) if getattr(tracked, "tracker_id", None) is not None else []
        boxes = list(tracked.xyxy) if getattr(tracked, "xyxy", None) is not None else []
        confidences = list(tracked.confidence) if getattr(tracked, "confidence", None) is not None else []
        class_ids = list(tracked.class_id) if getattr(tracked, "class_id", None) is not None else []
        results: list[TrackedDetection] = []
        for index, tracker_id in enumerate(tracker_ids):
            if tracker_id is None:
                continue
            class_id = int(class_ids[index]) if index < len(class_ids) else -1
            source_detection = next(
                (
                    item
                    for item in detections
                    if item.class_id == class_id and np.allclose(np.asarray(item.bbox_xyxy), np.asarray(boxes[index]), atol=2.0)
                ),
                None,
            )
            raw_class_name = source_detection.class_name if source_detection is not None else self._model_class_names.get(class_id, str(class_id))
            confidence = source_detection.confidence if source_detection is not None else float(confidences[index]) if index < len(confidences) else 0.0
            x1, y1, x2, y2 = [float(value) for value in boxes[index]]
            results.append(
                TrackedDetection(
                    camera_id=packet.camera_id,
                    tracker_namespace=tracker_namespace,
                    frame_number=packet.frame_number,
                    timestamp_seconds=packet.timestamp_seconds,
                    tracker_id=int(tracker_id),
                    bbox_xyxy=(x1, y1, x2, y2),
                    confidence=float(confidence),
                    raw_class_id=class_id,
                    raw_class_name=self._normalize_class_name(raw_class_name),
                )
            )
        return results
