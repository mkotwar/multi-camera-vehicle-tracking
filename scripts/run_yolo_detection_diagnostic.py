from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable

import cv2
import torch

from src.detector_tracker import VehicleDetectorTracker
from src.models import BBoxQualityDiagnostic, ConfigurationError, Detection, FramePacket
from src.pipeline import _load_raw_config, _validate_config

PROFILE_CLEAN = "clean"
PROFILE_REFERENCE = "reference"
PROFILE_OCR_MUKUL = "ocr_mukul"
SUPPORTED_PROFILES = (PROFILE_CLEAN, PROFILE_REFERENCE, PROFILE_OCR_MUKUL)
REFERENCE_CLASS_THRESHOLDS = {
    "3wheeler": 0.68,
    "car": 0.70,
    "bus": 0.84,
    "truck": 0.85,
    "motorcycle": 0.25,
}
OCR_MUKUL_ALLOWED_CLASS_IDS = tuple(range(8))


@dataclass(slots=True, frozen=True)
class RawYoloDetection:
    frame_number: int
    timestamp_seconds: float
    class_id: int
    raw_class_name: str
    normalized_class_name: str
    confidence: float
    bbox_xyxy: tuple[float, float, float, float] | None
    is_valid_bbox: bool
    rejection_reason: str | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a YOLO-only raw-vs-filtered diagnostic without tracking.")
    parser.add_argument("--video", required=True, help="Input video path.")
    parser.add_argument("--config", required=True, help="Clean project YAML config path.")
    parser.add_argument(
        "--profile",
        choices=SUPPORTED_PROFILES,
        default=PROFILE_CLEAN,
        help="Detection profile to emulate: clean, reference, or ocr_mukul.",
    )
    parser.add_argument("--all-frames", action="store_true", help="Process every frame in the selected video.")
    parser.add_argument("--start-frame", type=int, default=0, help="First frame index to consider.")
    parser.add_argument("--end-frame", type=int, default=None, help="Last frame index to consider, inclusive.")
    parser.add_argument("--frame-step", type=int, default=1, help="Sampling step between processed frames.")
    return parser.parse_args()


def compute_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_validated_config(config_path: str | Path) -> tuple[dict[str, Any], Path]:
    resolved = Path(config_path).expanduser().resolve()
    raw_config = _load_raw_config(resolved)
    validated = _validate_config(raw_config, resolved)
    return validated, resolved


def resolve_video_path(config_path: Path, raw_video_path: str | Path) -> Path:
    candidate = Path(raw_video_path).expanduser()
    if not candidate.is_absolute():
        candidate = (config_path.parent / candidate).resolve()
    else:
        candidate = candidate.resolve()
    if not candidate.exists():
        raise ConfigurationError(f"Diagnostic video path does not exist: {candidate}")
    return candidate


def select_frame_numbers(
    total_frames: int,
    *,
    all_frames: bool,
    start_frame: int,
    end_frame: int | None,
    frame_step: int,
) -> list[int]:
    if total_frames <= 0:
        return []
    if start_frame < 0:
        raise ConfigurationError("--start-frame must be at least 0.")
    if frame_step <= 0:
        raise ConfigurationError("--frame-step must be at least 1.")
    last_frame = total_frames - 1
    effective_end = last_frame if end_frame is None else min(end_frame, last_frame)
    if effective_end < start_frame:
        return []
    step = 1 if all_frames else frame_step
    return list(range(start_frame, effective_end + 1, step))


def create_output_directories(output_root: Path) -> dict[str, Path]:
    paths = {
        "raw_yolo_frames": output_root / "raw_yolo_frames",
        "accepted_detection_frames": output_root / "accepted_detection_frames",
        "rejected_detection_frames": output_root / "rejected_detection_frames",
        "side_by_side_frames": output_root / "side_by_side_frames",
    }
    output_root.mkdir(parents=True, exist_ok=True)
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def _json_clone(payload: dict[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(payload))


def apply_profile_overrides(validated_config: dict[str, Any], profile: str) -> tuple[dict[str, Any], dict[str, Any]]:
    adjusted = _json_clone(validated_config)
    detection = dict(adjusted["detection"])
    bbox_quality = dict(detection.get("bbox_quality", {}) or {})
    tracking = dict(adjusted["tracking"])
    metadata: dict[str, Any] = {"profile": profile}

    if profile == PROFILE_CLEAN:
        detection["backend"] = "legacy_clean"
        tracking["backend"] = "supervision_bytetrack"
        tracking["supported_isolation_modes"] = ["per_camera", "per_camera_class"]
        bbox_quality["enabled"] = True
        detection["bbox_quality"] = bbox_quality
        metadata["description"] = "Clean project YOLO config with clean bbox-quality filtering."
    elif profile == PROFILE_REFERENCE:
        detection["backend"] = "legacy_clean"
        tracking["backend"] = "supervision_bytetrack"
        tracking["supported_isolation_modes"] = ["per_camera", "per_camera_class"]
        detection["confidence_threshold"] = min(REFERENCE_CLASS_THRESHOLDS.values())
        detection["image_size"] = 640
        detection["agnostic_nms"] = False
        bbox_quality["enabled"] = False
        detection["bbox_quality"] = bbox_quality
        metadata["description"] = "Reference pipeline profile with per-class thresholds and no bbox-quality stage."
        metadata["class_confidence_thresholds"] = dict(REFERENCE_CLASS_THRESHOLDS)
    elif profile == PROFILE_OCR_MUKUL:
        detection["backend"] = "ocr_mukul"
        tracking["backend"] = "ocr_mukul_supervision_bytetrack"
        tracking["supported_isolation_modes"] = ["per_camera"]
        detection["confidence_threshold"] = 0.2
        detection["image_size"] = 1024
        detection["device"] = "cuda" if torch.cuda.is_available() else "cpu"
        bbox_quality["enabled"] = False
        detection["bbox_quality"] = bbox_quality
        metadata["description"] = "OCR_MUKUL profile using conf=0.2, imgsz=1024, and no clean bbox filtering."
        metadata["allowed_class_ids"] = list(OCR_MUKUL_ALLOWED_CLASS_IDS)
    else:
        raise ConfigurationError(f"Unsupported diagnostic profile: {profile}")

    adjusted["detection"] = detection
    adjusted["tracking"] = tracking
    return adjusted, metadata


def _tensor_to_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if hasattr(value, "tolist"):
        return list(value.tolist())
    return list(value)


def run_profile_inference(
    tracker: VehicleDetectorTracker,
    packet: FramePacket,
    *,
    profile: str,
) -> tuple[Any, dict[str, Any]]:
    if profile == PROFILE_CLEAN:
        predict_kwargs = {
            "source": packet.frame,
            "conf": tracker.confidence_threshold,
            "iou": tracker.iou_threshold,
            "imgsz": tracker.image_size,
            "device": tracker.device,
            "agnostic_nms": tracker.agnostic_nms,
            "verbose": False,
        }
        return tracker._model.predict(**predict_kwargs)[0], predict_kwargs  # noqa: SLF001

    if profile == PROFILE_REFERENCE:
        predict_kwargs = {
            "source": packet.frame,
            "conf": tracker.confidence_threshold,
            "iou": tracker.iou_threshold,
            "imgsz": tracker.image_size,
            "device": tracker.device,
            "verbose": False,
        }
        return tracker._model.predict(**predict_kwargs)[0], predict_kwargs  # noqa: SLF001

    if profile == PROFILE_OCR_MUKUL:
        call_kwargs = {
            "conf": tracker.confidence_threshold,
            "imgsz": tracker.image_size,
            "verbose": False,
        }
        model = tracker._model  # noqa: SLF001
        if callable(model):
            return model(packet.frame, **call_kwargs)[0], {"source": "frame ndarray", **call_kwargs}
        predict_kwargs = {
            "source": packet.frame,
            "conf": tracker.confidence_threshold,
            "imgsz": tracker.image_size,
            "verbose": False,
        }
        return model.predict(**predict_kwargs)[0], predict_kwargs

    raise ConfigurationError(f"Unsupported diagnostic profile: {profile}")


def extract_raw_yolo_detections_for_profile(
    tracker: VehicleDetectorTracker,
    packet: FramePacket,
    *,
    profile: str,
) -> tuple[list[RawYoloDetection], Any, dict[str, Any]]:
    raw_result, inference_kwargs = run_profile_inference(tracker, packet, profile=profile)
    boxes = getattr(raw_result, "boxes", None)
    if boxes is None or getattr(boxes, "xyxy", None) is None:
        return [], raw_result, inference_kwargs
    xyxy_values = _tensor_to_list(boxes.xyxy)
    class_values = _tensor_to_list(getattr(boxes, "cls", None))
    confidence_values = _tensor_to_list(getattr(boxes, "conf", None))
    raw_rows: list[RawYoloDetection] = []
    for index, raw_box in enumerate(xyxy_values):
        class_id = int(class_values[index]) if index < len(class_values) else -1
        confidence = float(confidence_values[index]) if index < len(confidence_values) else 0.0
        raw_class_name = str(tracker._model_class_names.get(class_id, str(class_id)))  # noqa: SLF001
        normalized_class_name = tracker._normalize_class_name(raw_class_name)  # noqa: SLF001
        bbox: tuple[float, float, float, float] | None
        is_valid_bbox = True
        rejection_reason: str | None = None
        try:
            x1, y1, x2, y2 = [float(value) for value in raw_box[:4]]
            bbox = (x1, y1, x2, y2)
            if not all(float("-inf") < value < float("inf") for value in bbox):
                is_valid_bbox = False
                rejection_reason = "INVALID_BBOX_NONFINITE"
            elif x2 <= x1 or y2 <= y1:
                is_valid_bbox = False
                rejection_reason = "INVALID_BBOX_NON_POSITIVE"
        except Exception:
            bbox = None
            is_valid_bbox = False
            rejection_reason = "INVALID_BBOX_PARSE_ERROR"
        raw_rows.append(
            RawYoloDetection(
                frame_number=packet.frame_number,
                timestamp_seconds=packet.timestamp_seconds,
                class_id=class_id,
                raw_class_name=raw_class_name,
                normalized_class_name=normalized_class_name,
                confidence=confidence,
                bbox_xyxy=bbox,
                is_valid_bbox=is_valid_bbox,
                rejection_reason=rejection_reason,
            )
        )
    return raw_rows, raw_result, inference_kwargs


def _build_detection_from_raw(row: RawYoloDetection) -> Detection:
    if row.bbox_xyxy is None:
        raise ConfigurationError("Cannot build Detection from a raw row without a bbox.")
    return Detection(
        bbox_xyxy=row.bbox_xyxy,
        confidence=row.confidence,
        class_id=row.class_id,
        class_name=row.normalized_class_name,
    )


def apply_detection_profile(
    *,
    tracker: VehicleDetectorTracker,
    packet: FramePacket,
    raw_rows: list[RawYoloDetection],
    profile: str,
) -> tuple[list[Detection], list[BBoxQualityDiagnostic], list[dict[str, Any]]]:
    profile_rejections: list[dict[str, Any]] = []

    def _reject(row: RawYoloDetection, reason: str, stage: str) -> None:
        profile_rejections.append(
            {
                "frame_number": row.frame_number,
                "timestamp_seconds": row.timestamp_seconds,
                "class_id": row.class_id,
                "raw_class_name": row.raw_class_name,
                "normalized_class_name": row.normalized_class_name,
                "confidence": row.confidence,
                "bbox_xyxy": list(row.bbox_xyxy) if row.bbox_xyxy is not None else None,
                "rejection_reason": reason,
                "rejection_stage": stage,
            }
        )

    if profile == PROFILE_CLEAN:
        accepted_candidates: list[Detection] = []
        allowed_class_names = set(tracker.allowed_classes)
        for row in raw_rows:
            if not row.is_valid_bbox or row.bbox_xyxy is None:
                continue
            if row.normalized_class_name not in allowed_class_names:
                _reject(row, "CLASS_NOT_ALLOWED", "allowed_class")
                continue
            accepted_candidates.append(_build_detection_from_raw(row))
        accepted_detections, bbox_quality_diagnostics = tracker.filter_detections(packet, accepted_candidates)
        return accepted_detections, bbox_quality_diagnostics, profile_rejections

    if profile == PROFILE_REFERENCE:
        accepted_detections: list[Detection] = []
        for row in raw_rows:
            if not row.is_valid_bbox or row.bbox_xyxy is None:
                continue
            threshold = REFERENCE_CLASS_THRESHOLDS.get(row.normalized_class_name)
            if threshold is None:
                _reject(row, "CLASS_NOT_ALLOWED", "allowed_class")
                continue
            if row.confidence < threshold:
                _reject(row, "BELOW_CLASS_CONFIDENCE_THRESHOLD", "class_confidence_threshold")
                continue
            accepted_detections.append(_build_detection_from_raw(row))
        return accepted_detections, [], profile_rejections

    if profile == PROFILE_OCR_MUKUL:
        accepted_detections = []
        allowed_class_ids = set(OCR_MUKUL_ALLOWED_CLASS_IDS)
        for row in raw_rows:
            if not row.is_valid_bbox or row.bbox_xyxy is None:
                continue
            if row.class_id not in allowed_class_ids:
                _reject(row, "CLASS_ID_NOT_TRACKED_BY_OCR_MUKUL", "class_id_allowlist")
                continue
            accepted_detections.append(_build_detection_from_raw(row))
        return accepted_detections, [], profile_rejections

    raise ConfigurationError(f"Unsupported diagnostic profile: {profile}")


def build_rejected_rows(
    raw_rows: Iterable[RawYoloDetection],
    bbox_quality_diagnostics: Iterable[BBoxQualityDiagnostic],
    profile_rejections: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    rejected_rows: list[dict[str, Any]] = []
    for row in raw_rows:
        if row.rejection_reason is None:
            continue
        rejected_rows.append(
            {
                "frame_number": row.frame_number,
                "timestamp_seconds": row.timestamp_seconds,
                "class_id": row.class_id,
                "raw_class_name": row.raw_class_name,
                "normalized_class_name": row.normalized_class_name,
                "confidence": row.confidence,
                "bbox_xyxy": list(row.bbox_xyxy) if row.bbox_xyxy is not None else None,
                "rejection_reason": row.rejection_reason,
                "rejection_stage": "raw_bbox_validation",
            }
        )
    rejected_rows.extend(dict(item) for item in profile_rejections)
    for diagnostic in bbox_quality_diagnostics:
        if diagnostic.accepted_by_bbox_quality:
            continue
        rejected_rows.append(
            {
                "frame_number": diagnostic.frame_number,
                "timestamp_seconds": diagnostic.timestamp_seconds,
                "class_id": None,
                "raw_class_name": diagnostic.class_name,
                "normalized_class_name": diagnostic.normalized_class_name,
                "confidence": diagnostic.confidence,
                "bbox_xyxy": list(diagnostic.bbox_xyxy),
                "rejection_reason": diagnostic.rejection_reason,
                "rejection_stage": "bbox_quality",
            }
        )
    return rejected_rows


def draw_boxes(
    frame: Any,
    rows: Iterable[dict[str, Any]],
    *,
    color: tuple[int, int, int],
    title_prefix: str,
) -> Any:
    annotated = frame.copy()
    for row in rows:
        bbox = row.get("bbox_xyxy")
        if bbox is None:
            continue
        x1, y1, x2, y2 = [int(round(float(value))) for value in bbox]
        label = f"{title_prefix} {row.get('normalized_class_name', row.get('raw_class_name', 'unknown')).upper()} {float(row.get('confidence', 0.0)):.2f}"
        reason = row.get("rejection_reason")
        if reason:
            label = f"{label} {reason}"
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        cv2.putText(annotated, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
    return annotated


def build_side_by_side(left_frame: Any, right_frame: Any) -> Any:
    if left_frame.shape[0] != right_frame.shape[0]:
        raise ConfigurationError("Side-by-side diagnostic frames must have matching heights.")
    return cv2.hconcat([left_frame, right_frame])


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            payload = dict(row)
            bbox = payload.get("bbox_xyxy")
            if isinstance(bbox, list):
                payload["bbox_xyxy"] = json.dumps(bbox)
            writer.writerow(payload)


def summarize_rejections(rejected_rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    by_reason: dict[str, int] = {}
    by_class: dict[str, int] = {}
    by_stage: dict[str, int] = {}
    total = 0
    for row in rejected_rows:
        total += 1
        reason = str(row.get("rejection_reason") or "UNKNOWN")
        class_name = str(row.get("normalized_class_name") or "UNKNOWN")
        stage = str(row.get("rejection_stage") or "UNKNOWN")
        by_reason[reason] = by_reason.get(reason, 0) + 1
        by_class[class_name] = by_class.get(class_name, 0) + 1
        by_stage[stage] = by_stage.get(stage, 0) + 1
    return {
        "total_rejected_detections": total,
        "rejected_by_reason": by_reason,
        "rejected_by_class": by_class,
        "rejected_by_stage": by_stage,
    }


def build_runtime_config(
    *,
    validated_config: dict[str, Any],
    config_path: Path,
    video_path: Path,
    output_root: Path,
    selected_frames: list[int],
    tracker: VehicleDetectorTracker,
    profile: str,
    profile_metadata: dict[str, Any],
    predict_call: dict[str, Any],
) -> dict[str, Any]:
    model_path = Path(str(validated_config["detection"]["model_path"])).resolve()
    serialized_predict_call = {
        key: ("frame ndarray" if key == "source" else value)
        for key, value in predict_call.items()
    }
    return {
        "profile": profile,
        "profile_metadata": profile_metadata,
        "config_path": str(config_path),
        "video_path": str(video_path),
        "output_directory": str(output_root),
        "selected_frames": selected_frames,
        "frame_count_selected": len(selected_frames),
        "detection": {
            "model_path": str(model_path),
            "model_size_bytes": model_path.stat().st_size,
            "model_sha256": compute_sha256(model_path),
            "confidence_threshold": tracker.confidence_threshold,
            "iou_threshold": tracker.iou_threshold,
            "image_size": tracker.image_size,
            "configured_device": tracker.configured_device,
            "resolved_device": tracker.device,
            "agnostic_nms": tracker.agnostic_nms,
            "allowed_classes": list(tracker.allowed_classes),
            "allowed_model_class_ids": sorted(tracker._allowed_model_class_ids),  # noqa: SLF001
            "bbox_quality": validated_config["detection"]["bbox_quality"],
            "predict_call": serialized_predict_call,
        },
    }


def write_markdown_report(
    path: Path,
    *,
    runtime_config: dict[str, Any],
    processed_frames: int,
    raw_rows: list[dict[str, Any]],
    accepted_rows: list[dict[str, Any]],
    rejected_rows: list[dict[str, Any]],
    rejection_metrics: dict[str, Any],
) -> None:
    lines = [
        "# YOLO Diagnostic Report",
        "",
        f"- Profile: `{runtime_config['profile']}`",
        f"- Config: `{runtime_config['config_path']}`",
        f"- Video: `{runtime_config['video_path']}`",
        f"- Output: `{runtime_config['output_directory']}`",
        f"- Processed frames: `{processed_frames}`",
        f"- Raw YOLO detections: `{len(raw_rows)}`",
        f"- Accepted detections: `{len(accepted_rows)}`",
        f"- Rejected detections: `{len(rejected_rows)}`",
        "",
        "## Settings",
        "",
        f"- Model: `{runtime_config['detection']['model_path']}`",
        f"- Confidence: `{runtime_config['detection']['confidence_threshold']}`",
        f"- IoU: `{runtime_config['detection']['iou_threshold']}`",
        f"- Image size: `{runtime_config['detection']['image_size']}`",
        f"- Device: `{runtime_config['detection']['resolved_device']}`",
        f"- Allowed classes: `{', '.join(runtime_config['detection']['allowed_classes'])}`",
        f"- Agnostic NMS: `{runtime_config['detection']['agnostic_nms']}`",
        "",
        "## Rejections",
        "",
        f"- By reason: `{json.dumps(rejection_metrics['rejected_by_reason'], sort_keys=True)}`",
        f"- By class: `{json.dumps(rejection_metrics['rejected_by_class'], sort_keys=True)}`",
        f"- By stage: `{json.dumps(rejection_metrics['rejected_by_stage'], sort_keys=True)}`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_diagnostic(
    *,
    video_path: str | Path,
    config_path: str | Path,
    profile: str = PROFILE_CLEAN,
    all_frames: bool = False,
    start_frame: int = 0,
    end_frame: int | None = None,
    frame_step: int = 1,
    tracker_factory: Callable[[dict[str, Any], logging.Logger], VehicleDetectorTracker] | None = None,
) -> Path:
    validated_config, resolved_config_path = load_validated_config(config_path)
    profiled_config, profile_metadata = apply_profile_overrides(validated_config, profile)
    resolved_video_path = resolve_video_path(resolved_config_path, video_path)
    logger = logging.getLogger("yolo-diagnostic")
    logger.handlers.clear()
    logger.addHandler(logging.NullHandler())
    tracker = tracker_factory(profiled_config, logger) if tracker_factory is not None else VehicleDetectorTracker(profiled_config, logger)
    diagnostic_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = (resolved_config_path.parent / "outputs" / "yolo_diagnostics" / f"{diagnostic_id}_{profile}").resolve()
    output_paths = create_output_directories(output_root)

    capture = cv2.VideoCapture(str(resolved_video_path))
    if not capture.isOpened():
        raise ConfigurationError(f"OpenCV could not open diagnostic video: {resolved_video_path}")
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0) or 1.0
    selected_frames = select_frame_numbers(
        total_frames,
        all_frames=all_frames,
        start_frame=start_frame,
        end_frame=end_frame,
        frame_step=frame_step,
    )
    raw_rows: list[dict[str, Any]] = []
    accepted_rows: list[dict[str, Any]] = []
    rejected_rows: list[dict[str, Any]] = []
    processed_frames = 0
    last_predict_call: dict[str, Any] = {}

    try:
        for frame_number in selected_frames:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ok, frame = capture.read()
            if not ok:
                continue
            timestamp_seconds = frame_number / fps
            packet = FramePacket(
                camera_id="CAM_001",
                frame_number=frame_number,
                timestamp_seconds=timestamp_seconds,
                source_fps=fps,
                frame=frame,
                worker_id=0,
                captured_at="diagnostic",
                source_type="video",
            )
            processed_frames += 1
            frame_raw_rows, _raw_result, last_predict_call = extract_raw_yolo_detections_for_profile(
                tracker,
                packet,
                profile=profile,
            )
            accepted_detections, bbox_quality_diagnostics, profile_rejections = apply_detection_profile(
                tracker=tracker,
                packet=packet,
                raw_rows=frame_raw_rows,
                profile=profile,
            )
            frame_rejected_rows = build_rejected_rows(frame_raw_rows, bbox_quality_diagnostics, profile_rejections)

            raw_frame_rows = [
                {
                    "frame_number": item.frame_number,
                    "timestamp_seconds": item.timestamp_seconds,
                    "class_id": item.class_id,
                    "raw_class_name": item.raw_class_name,
                    "normalized_class_name": item.normalized_class_name,
                    "confidence": item.confidence,
                    "bbox_xyxy": list(item.bbox_xyxy) if item.bbox_xyxy is not None else None,
                    "is_valid_bbox": item.is_valid_bbox,
                    "rejection_reason": item.rejection_reason,
                }
                for item in frame_raw_rows
            ]
            accepted_frame_rows = [
                {
                    "frame_number": packet.frame_number,
                    "timestamp_seconds": packet.timestamp_seconds,
                    "class_id": item.class_id,
                    "raw_class_name": item.class_name,
                    "normalized_class_name": item.class_name,
                    "confidence": item.confidence,
                    "bbox_xyxy": list(item.bbox_xyxy),
                    "rejection_reason": None,
                }
                for item in accepted_detections
            ]
            raw_rows.extend(raw_frame_rows)
            accepted_rows.extend(accepted_frame_rows)
            rejected_rows.extend(frame_rejected_rows)

            raw_annotated = draw_boxes(frame, raw_frame_rows, color=(255, 180, 40), title_prefix="RAW")
            accepted_annotated = draw_boxes(frame, accepted_frame_rows, color=(0, 200, 255), title_prefix="ACCEPT")
            rejected_annotated = draw_boxes(frame, frame_rejected_rows, color=(40, 40, 255), title_prefix="REJECT")
            side_by_side = build_side_by_side(raw_annotated, accepted_annotated)

            file_name = f"frame_{frame_number:06d}.jpg"
            cv2.imwrite(str(output_paths["raw_yolo_frames"] / file_name), raw_annotated)
            cv2.imwrite(str(output_paths["accepted_detection_frames"] / file_name), accepted_annotated)
            cv2.imwrite(str(output_paths["rejected_detection_frames"] / file_name), rejected_annotated)
            cv2.imwrite(str(output_paths["side_by_side_frames"] / file_name), side_by_side)
    finally:
        capture.release()

    rejection_metrics = summarize_rejections(rejected_rows)
    runtime_config = build_runtime_config(
        validated_config=profiled_config,
        config_path=resolved_config_path,
        video_path=resolved_video_path,
        output_root=output_root,
        selected_frames=selected_frames,
        tracker=tracker,
        profile=profile,
        profile_metadata=profile_metadata,
        predict_call=last_predict_call,
    )
    runtime_config["video_probe"] = {
        "fps": fps,
        "total_frames": total_frames,
        "processed_frames": processed_frames,
    }

    write_csv(
        output_root / "raw_detections.csv",
        raw_rows,
        [
            "frame_number",
            "timestamp_seconds",
            "class_id",
            "raw_class_name",
            "normalized_class_name",
            "confidence",
            "bbox_xyxy",
            "is_valid_bbox",
            "rejection_reason",
        ],
    )
    write_csv(
        output_root / "accepted_detections.csv",
        accepted_rows,
        [
            "frame_number",
            "timestamp_seconds",
            "class_id",
            "raw_class_name",
            "normalized_class_name",
            "confidence",
            "bbox_xyxy",
            "rejection_reason",
        ],
    )
    write_csv(
        output_root / "rejected_detections.csv",
        rejected_rows,
        [
            "frame_number",
            "timestamp_seconds",
            "class_id",
            "raw_class_name",
            "normalized_class_name",
            "confidence",
            "bbox_xyxy",
            "rejection_reason",
            "rejection_stage",
        ],
    )
    (output_root / "rejection_metrics.json").write_text(json.dumps(rejection_metrics, indent=2), encoding="utf-8")
    (output_root / "runtime_config.json").write_text(json.dumps(runtime_config, indent=2), encoding="utf-8")
    write_markdown_report(
        output_root / "diagnostic_report.md",
        runtime_config=runtime_config,
        processed_frames=processed_frames,
        raw_rows=raw_rows,
        accepted_rows=accepted_rows,
        rejected_rows=rejected_rows,
        rejection_metrics=rejection_metrics,
    )
    return output_root


def main() -> int:
    args = parse_args()
    output_root = run_diagnostic(
        video_path=args.video,
        config_path=args.config,
        profile=str(args.profile),
        all_frames=bool(args.all_frames),
        start_frame=int(args.start_frame),
        end_frame=args.end_frame,
        frame_step=int(args.frame_step),
    )
    print(f"YOLO diagnostic created: {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
