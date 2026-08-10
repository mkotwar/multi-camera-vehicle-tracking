from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_scalability_benchmark import (  # noqa: E402
    build_notes,
    classify_bottlenecks,
    classify_fairness,
    determine_primary_bottlenecks,
    run_pipeline_subprocess,
)
from src.detector_tracker import VehicleDetectorTracker  # noqa: E402
from src.models import ConfigurationError, FramePacket  # noqa: E402
from src.pipeline import _load_raw_config, _validate_config  # noqa: E402
from src.yolo_imgsz_benchmark import (  # noqa: E402
    DEFAULT_IOU_THRESHOLD,
    DEFAULT_MEDIUM_AREA_RATIO_THRESHOLD,
    DEFAULT_SMALL_AREA_RATIO_THRESHOLD,
    BenchmarkDetection,
    detection_key,
    summarize_parity,
)


DEFAULT_IMGSZ_VALUES = [1024, 896, 768]
DEFAULT_CAMERA_COUNTS = [4, 8, 12]
FRAME_NAME_PATTERN = re.compile(r"frame_(\d+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile YOLO detection stage and benchmark imgsz parity/performance.")
    parser.add_argument("--base-config", default="config/validation.yaml")
    parser.add_argument("--output-dir", default="diagnostics/yolo_imgsz_benchmark")
    parser.add_argument("--imgsz-values", nargs="+", type=int, default=DEFAULT_IMGSZ_VALUES)
    parser.add_argument("--camera-counts", nargs="+", type=int, default=DEFAULT_CAMERA_COUNTS)
    parser.add_argument("--frame-limit", type=int, default=30)
    parser.add_argument("--frame-set-target", type=int, default=100)
    parser.add_argument("--sample-interval-seconds", type=float, default=1.0)
    parser.add_argument("--iou-threshold", type=float, default=DEFAULT_IOU_THRESHOLD)
    parser.add_argument("--small-area-threshold", type=float, default=DEFAULT_SMALL_AREA_RATIO_THRESHOLD)
    parser.add_argument("--medium-area-threshold", type=float, default=DEFAULT_MEDIUM_AREA_RATIO_THRESHOLD)
    return parser.parse_args()


def _read_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return payload


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def _ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _json_clone(payload: dict[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(payload))


def build_performance_config(
    base_config: dict[str, Any],
    *,
    camera_count: int,
    frame_limit: int,
    imgsz: int,
    mode: str,
) -> dict[str, Any]:
    config = _json_clone(base_config)
    input_section = dict(config.get("input", {}) or {})
    cameras = list(input_section.get("cameras", []) or [])
    if not cameras:
        raise ValueError("Base config must contain at least one camera.")
    first_camera = dict(cameras[0])
    logical_cameras = [
        {
            "camera_id": f"CAM_{index + 1:03d}",
            "source_type": first_camera.get("source_type"),
            "source": first_camera.get("source"),
            "enabled": True,
        }
        for index in range(camera_count)
    ]
    input_section["cameras"] = logical_cameras
    input_section["max_frames_per_camera"] = frame_limit
    config["input"] = input_section

    ingestion = dict(config.get("ingestion", {}) or {})
    ingestion["worker_count"] = 3
    ingestion["per_camera_buffer_size"] = 2
    ingestion["scheduler_policy"] = "round_robin"
    config["ingestion"] = ingestion

    detection = dict(config.get("detection", {}) or {})
    detection["image_size"] = int(imgsz)
    detection["batch"] = {"enabled": False, "max_size": 1, "max_wait_ms": 0.0}
    config["detection"] = detection

    vehicle_enrichment = dict(config.get("vehicle_enrichment", {}) or {})
    async_colour = dict(vehicle_enrichment.get("async_colour", {}) or {})
    async_colour["worker_count"] = 1
    async_colour["queue_size"] = int(async_colour.get("queue_size", 100) or 100)
    if mode == "detector_only":
        vehicle_enrichment["enabled"] = False
        async_colour["enabled"] = False
    vehicle_enrichment["async_colour"] = async_colour
    config["vehicle_enrichment"] = vehicle_enrichment
    return config


def write_benchmark_config(output_dir: Path, *, camera_count: int, imgsz: int, mode: str, config: dict[str, Any]) -> Path:
    configs_dir = _ensure_directory(output_dir / "configs")
    path = configs_dir / f"config.{mode}.{camera_count}cam.imgsz{imgsz}.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def collect_run_metrics(
    run_directory: Path,
    *,
    camera_count: int,
    frames_per_camera: int,
    imgsz: int,
    mode: str,
    monitor: dict[str, Any],
) -> dict[str, Any]:
    summary = _read_json(run_directory / "summary.json")
    ingestion = _read_json(run_directory / "ingestion_metrics.json")
    detection = _read_json(run_directory / "detection_tracking_metrics.json")
    evidence = _read_json(run_directory / "evidence_metrics.json")
    enrichment = _read_json(run_directory / "vehicle_enrichment_metrics.json")
    run_config = _read_yaml(run_directory / "run_config.yaml")

    frames_processed = int(summary.get("processed_frames", 0) or 0)
    expected_total_frames = camera_count * frames_per_camera
    processing_runtime_sec = float(summary.get("overall_pipeline_runtime_ms", 0.0) or 0.0) / 1000.0
    fairness_classification, fairness_imbalance_percent = classify_fairness(
        dict(summary.get("frames_consumed_by_camera", {}) or {}),
        max_consecutive_frames_same_camera=int(summary.get("max_consecutive_frames_same_camera", 0) or 0),
    )
    yolo_frames_processed = int(detection.get("yolo_frames_processed", frames_processed) or 0)
    yolo_total_inference_time_ms = float(detection.get("yolo_inference_time_total_ms", detection.get("total_inference_time_ms", 0.0)) or 0.0)
    yolo_fps = (yolo_frames_processed / (yolo_total_inference_time_ms / 1000.0)) if yolo_total_inference_time_ms > 0.0 else 0.0
    pipeline_fps = (frames_processed / processing_runtime_sec) if processing_runtime_sec > 0.0 else 0.0
    metrics = {
        "benchmark_generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "camera_count": camera_count,
        "imgsz": imgsz,
        "frames_per_camera": frames_per_camera,
        "expected_total_frames": expected_total_frames,
        "processed_total_frames": frames_processed,
        "run_id": summary.get("run_id"),
        "run_directory": str(run_directory),
        "total_runtime_sec": float(monitor.get("total_runtime_seconds", 0.0) or 0.0),
        "processing_runtime_sec": processing_runtime_sec,
        "yolo_model_invocations": int(detection.get("yolo_model_invocations", frames_processed) or 0),
        "yolo_frames_processed": yolo_frames_processed,
        "yolo_frames_per_second": round(yolo_fps, 3) if yolo_fps > 0.0 else 0.0,
        "pipeline_frames_per_second": round(pipeline_fps, 3) if pipeline_fps > 0.0 else 0.0,
        "average_total_detection_latency_ms": float(detection.get("total_detection_stage_profile_ms", {}).get("mean", 0.0) or 0.0),
        "p50_total_detection_latency_ms": float(detection.get("total_detection_stage_profile_ms", {}).get("p50", 0.0) or 0.0),
        "p95_total_detection_latency_ms": float(detection.get("total_detection_stage_profile_ms", {}).get("p95", 0.0) or 0.0),
        "preprocess_mean_ms": float(detection.get("preprocess_stage_profile_ms", {}).get("mean", 0.0) or 0.0),
        "inference_mean_ms": float(detection.get("model_inference_stage_profile_ms", {}).get("mean", 0.0) or 0.0),
        "postprocess_mean_ms": float(detection.get("postprocess_stage_profile_ms", {}).get("mean", 0.0) or 0.0),
        "conversion_mean_ms": float(detection.get("result_conversion_stage_profile_ms", {}).get("mean", 0.0) or 0.0),
        "routing_mean_ms": float(detection.get("result_routing_stage_profile_ms", {}).get("mean", 0.0) or 0.0),
        "detection_latency_ms_avg": float(detection.get("detection_latency_ms_avg", 0.0) or 0.0),
        "detection_latency_ms_p50": float(detection.get("detection_latency_ms_p50", 0.0) or 0.0),
        "detection_latency_ms_p95": float(detection.get("detection_latency_ms_p95", 0.0) or 0.0),
        "detection_latency_ms_max": float(detection.get("detection_latency_ms_max", 0.0) or 0.0),
        "detection_queue_peak": int(ingestion.get("maximum_observed_queue_size", 0) or 0),
        "detection_queue_capacity": int(run_config.get("ingestion", {}).get("frame_queue_size", 0) or 0),
        "frame_order_violations": int(detection.get("frame_order_violations", 0) or 0),
        "frame_loss_count": max(0, expected_total_frames - frames_processed),
        "fairness_classification": fairness_classification,
        "fairness_imbalance_percent": fairness_imbalance_percent,
        "gpu_memory_peak_mb": monitor.get("gpu_memory_peak_mb"),
        "cuda_peak_allocated_mb": detection.get("gpu_peak_allocated_mb"),
        "cuda_peak_reserved_mb": detection.get("gpu_peak_reserved_mb"),
        "colour_queue_peak": int(summary.get("colour_queue_peak_depth", 0) or 0),
        "colour_blocking_events": int(enrichment.get("colour_queue_blocked_events", 0) or 0),
        "pending_jobs_shutdown": max(
            int(summary.get("pending_colour_jobs_at_shutdown", 0) or 0),
            int(enrichment.get("colour_worker_shutdown_pending_jobs", 0) or 0),
            int(evidence.get("pending_evidence_tracks_at_shutdown", 0) or 0),
        ),
        "subprocess_command": monitor.get("subprocess_command"),
    }
    statuses = classify_bottlenecks(
        {
            **metrics,
            "tracker_states": int(detection.get("tracker_instances_created_total", 0) or 0),
            "processed_total_frames": frames_processed,
            "ram_peak_mb": monitor.get("ram_peak_mb"),
            "colour_queue_peak_percent": float(summary.get("colour_queue_peak_depth", 0) or 0.0),
            "detection_queue_peak_percent": (
                float(metrics["detection_queue_peak"] * 100.0 / metrics["detection_queue_capacity"])
                if metrics["detection_queue_capacity"]
                else 0.0
            ),
            "buffer_full_count": int(summary.get("buffer_full_count", 0) or 0),
            "evidence_cache_misses": int(evidence.get("evidence_cache_misses", 0) or 0),
        }
    )
    primary, secondary = determine_primary_bottlenecks(
        {
            **metrics,
            "tracker_states": int(detection.get("tracker_instances_created_total", 0) or 0),
            "processed_total_frames": frames_processed,
            "ram_peak_mb": monitor.get("ram_peak_mb"),
            "colour_queue_peak_percent": float(summary.get("colour_queue_peak_depth", 0) or 0.0),
            "detection_queue_peak_percent": (
                float(metrics["detection_queue_peak"] * 100.0 / metrics["detection_queue_capacity"])
                if metrics["detection_queue_capacity"]
                else 0.0
            ),
            "buffer_full_count": int(summary.get("buffer_full_count", 0) or 0),
            "evidence_cache_misses": int(evidence.get("evidence_cache_misses", 0) or 0),
        },
        statuses,
    )
    metrics["bottleneck_status"] = statuses
    metrics["primary_bottleneck"] = primary
    metrics["secondary_bottleneck"] = secondary
    metrics["notes"] = build_notes(
        {
            **metrics,
            "colour_queue_peak_percent": float(summary.get("colour_queue_peak_depth", 0) or 0.0),
            "detection_queue_peak_percent": (
                float(metrics["detection_queue_peak"] * 100.0 / metrics["detection_queue_capacity"])
                if metrics["detection_queue_capacity"]
                else 0.0
            ),
        }
    )
    return metrics


def _extract_frame_number(path: Path) -> int:
    match = FRAME_NAME_PATTERN.search(path.stem)
    if match is None:
        raise ValueError(f"Could not parse frame number from {path}")
    return int(match.group(1))


def collect_frame_set(*, outputs_root: Path, target_count: int) -> list[dict[str, Any]]:
    frame_paths: list[Path] = []
    seen_paths: set[str] = set()
    for run_dir in sorted(outputs_root.glob("runs/*"), key=lambda item: item.stat().st_mtime, reverse=True):
        raw_root = run_dir / "raw_frames"
        if not raw_root.exists():
            continue
        for frame_path in sorted(raw_root.rglob("*.jpg")):
            normalized = str(frame_path.resolve())
            if normalized in seen_paths:
                continue
            seen_paths.add(normalized)
            frame_paths.append(frame_path)
            if len(frame_paths) >= target_count:
                break
        if len(frame_paths) >= target_count:
            break
    if len(frame_paths) < target_count:
        raise ConfigurationError(
            f"Need at least {target_count} stored raw frames for the parity benchmark; found {len(frame_paths)}."
        )
    items: list[dict[str, Any]] = []
    for frame_path in frame_paths[:target_count]:
        camera_id = frame_path.parent.name
        run_id = frame_path.parents[2].name
        items.append(
            {
                "camera_id": camera_id,
                "frame_number": _extract_frame_number(frame_path),
                "frame_path": str(frame_path.resolve()),
                "run_id": run_id,
            }
        )
    return items


def write_frame_set_manifest(output_dir: Path, frame_items: list[dict[str, Any]]) -> Path:
    path = output_dir / "frame_set_manifest.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["run_id", "camera_id", "frame_number", "frame_path"])
        writer.writeheader()
        for item in frame_items:
            writer.writerow(item)
    return path


def build_tracker_config(validated_config: dict[str, Any], *, imgsz: int) -> dict[str, Any]:
    config = _json_clone(validated_config)
    detection = dict(config.get("detection", {}) or {})
    detection["image_size"] = int(imgsz)
    detection["batch"] = {"enabled": False, "max_size": 1, "max_wait_ms": 0.0}
    config["detection"] = detection
    return config


def run_frame_benchmark(
    validated_config: dict[str, Any],
    *,
    imgsz: int,
    frame_items: list[dict[str, Any]],
    logger: logging.Logger,
) -> dict[str, Any]:
    tracker = VehicleDetectorTracker(build_tracker_config(validated_config, imgsz=imgsz), logger)
    detections: list[BenchmarkDetection] = []
    profile_rows: list[dict[str, Any]] = []
    for item in frame_items:
        frame_path = Path(item["frame_path"])
        frame = cv2.imread(str(frame_path))
        if frame is None:
            raise RuntimeError(f"Failed to read frame image: {frame_path}")
        packet = FramePacket(
            camera_id=str(item["camera_id"]),
            frame_number=int(item["frame_number"]),
            timestamp_seconds=float(item["frame_number"]),
            source_fps=10.0,
            frame=frame,
            source_frame_width=int(frame.shape[1]),
            source_frame_height=int(frame.shape[0]),
            worker_id=0,
            captured_at="2026-08-08T00:00:00+00:00",
            source_type="image",
        )
        result = tracker.process_frame(packet)
        profile_rows.append(
            {
                "camera_id": packet.camera_id,
                "frame_number": packet.frame_number,
                "frame_path": str(frame_path),
                "preprocess_ms": float(result.preprocess_ms),
                "model_inference_ms": float(result.model_inference_ms),
                "postprocess_ms": float(result.postprocess_ms),
                "result_conversion_ms": float(result.result_conversion_ms),
                "result_routing_ms": float(result.result_routing_ms),
                "tracker_update_ms": float(result.tracker_update_ms),
                "total_detection_ms": float(result.total_detection_ms),
            }
        )
        for detection in result.detections:
            detections.append(
                BenchmarkDetection(
                    camera_id=packet.camera_id,
                    frame_number=packet.frame_number,
                    frame_path=str(frame_path),
                    class_name=str(detection.class_name),
                    confidence=float(detection.confidence),
                    bbox_xyxy=tuple(float(value) for value in detection.bbox_xyxy),
                    frame_width=int(frame.shape[1]),
                    frame_height=int(frame.shape[0]),
                    imgsz=int(imgsz),
                )
            )
    return {"detections": detections, "profile_rows": profile_rows}


def write_detection_rows(output_dir: Path, *, imgsz: int, detections: list[BenchmarkDetection], profile_rows: list[dict[str, Any]]) -> None:
    detections_json = output_dir / f"detections_{imgsz}.json"
    detections_csv = output_dir / f"detections_{imgsz}.csv"
    profile_csv = output_dir / f"profile_{imgsz}.csv"
    detection_rows = [
        {
            "camera_id": item.camera_id,
            "frame_number": item.frame_number,
            "frame_path": item.frame_path,
            "class": item.class_name,
            "confidence": item.confidence,
            "bbox_x1": item.bbox_xyxy[0],
            "bbox_y1": item.bbox_xyxy[1],
            "bbox_x2": item.bbox_xyxy[2],
            "bbox_y2": item.bbox_xyxy[3],
            "frame_width": item.frame_width,
            "frame_height": item.frame_height,
            "imgsz": item.imgsz,
        }
        for item in detections
    ]
    detections_json.write_text(json.dumps(detection_rows, indent=2), encoding="utf-8")
    with detections_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(detection_rows[0].keys()) if detection_rows else [
            "camera_id", "frame_number", "frame_path", "class", "confidence", "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2", "frame_width", "frame_height", "imgsz"
        ])
        writer.writeheader()
        for row in detection_rows:
            writer.writerow(row)
    with profile_csv.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = ["camera_id", "frame_number", "frame_path", "preprocess_ms", "model_inference_ms", "postprocess_ms", "result_conversion_ms", "result_routing_ms", "tracker_update_ms", "total_detection_ms"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in profile_rows:
            writer.writerow(row)


def _stage_summary(profile_rows: list[dict[str, Any]], key: str) -> dict[str, float]:
    values = [float(row.get(key, 0.0) or 0.0) for row in profile_rows]
    if not values:
        return {"mean": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}
    values_sorted = sorted(values)
    def _percentile(percent: float) -> float:
        if len(values_sorted) == 1:
            return values_sorted[0]
        position = (len(values_sorted) - 1) * percent
        lower = int(position)
        upper = min(lower + 1, len(values_sorted) - 1)
        weight = position - lower
        return float(values_sorted[lower] * (1.0 - weight) + values_sorted[upper] * weight)
    return {
        "mean": float(sum(values) / len(values)),
        "p50": _percentile(0.50),
        "p95": _percentile(0.95),
        "max": float(max(values)),
    }


def write_parity_csv(output_dir: Path, *, label: str, parity: dict[str, Any]) -> Path:
    path = output_dir / f"parity_{label}.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "camera_id",
                "frame_number",
                "frame_path",
                "class_name",
                "baseline_confidence",
                "candidate_confidence",
                "iou",
                "status",
            ],
        )
        writer.writeheader()
        for match in parity["matches"]:
            writer.writerow(
                {
                    "camera_id": match["baseline"].camera_id,
                    "frame_number": match["baseline"].frame_number,
                    "frame_path": match["baseline"].frame_path,
                    "class_name": match["baseline"].class_name,
                    "baseline_confidence": match["baseline"].confidence,
                    "candidate_confidence": match["candidate"].confidence,
                    "iou": match["iou"],
                    "status": "matched",
                }
            )
        for item in parity["missing"]:
            writer.writerow(
                {
                    "camera_id": item.camera_id,
                    "frame_number": item.frame_number,
                    "frame_path": item.frame_path,
                    "class_name": item.class_name,
                    "baseline_confidence": item.confidence,
                    "candidate_confidence": "",
                    "iou": "",
                    "status": "missing",
                }
            )
        for item in parity["additional"]:
            writer.writerow(
                {
                    "camera_id": item.camera_id,
                    "frame_number": item.frame_number,
                    "frame_path": item.frame_path,
                    "class_name": item.class_name,
                    "baseline_confidence": "",
                    "candidate_confidence": item.confidence,
                    "iou": "",
                    "status": "additional",
                }
            )
    return path


def build_missing_detection_review(
    baseline_1024: list[BenchmarkDetection],
    parity_896: dict[str, Any],
    parity_768: dict[str, Any],
) -> list[dict[str, Any]]:
    by_key = {detection_key(item): item for item in baseline_1024}
    review_rows: dict[str, dict[str, Any]] = {}
    missing_896 = {detection_key(item): item for item in parity_896["missing"]}
    missing_768 = {detection_key(item): item for item in parity_768["missing"]}
    match_896 = {detection_key(item["baseline"]): item for item in parity_896["matches"]}
    match_768 = {detection_key(item["baseline"]): item for item in parity_768["matches"]}
    for key, baseline_item in by_key.items():
        if key not in missing_896 and key not in missing_768:
            continue
        review_rows[key] = {
            "camera_id": baseline_item.camera_id,
            "frame_number": baseline_item.frame_number,
            "frame_path": baseline_item.frame_path,
            "vehicle_class": baseline_item.class_name,
            "1024_bbox": list(baseline_item.bbox_xyxy),
            "1024_confidence": baseline_item.confidence,
            "896_match": key not in missing_896,
            "896_confidence": match_896.get(key, {}).get("candidate").confidence if key in match_896 else "",
            "896_iou": match_896.get(key, {}).get("iou", ""),
            "768_match": key not in missing_768,
            "768_confidence": match_768.get(key, {}).get("candidate").confidence if key in match_768 else "",
            "768_iou": match_768.get(key, {}).get("iou", ""),
            "object_size_group": "",
            "manual_ground_truth_present": "",
            "notes": "",
        }
    return list(review_rows.values())


def write_missing_detection_review(output_dir: Path, rows: list[dict[str, Any]]) -> Path:
    path = output_dir / "missing_detection_review.csv"
    fieldnames = [
        "camera_id",
        "frame_number",
        "frame_path",
        "vehicle_class",
        "1024_bbox",
        "1024_confidence",
        "896_match",
        "896_confidence",
        "896_iou",
        "768_match",
        "768_confidence",
        "768_iou",
        "object_size_group",
        "manual_ground_truth_present",
        "notes",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def write_review_images(output_dir: Path, *, parity: dict[str, Any], label: str, max_images: int = 12) -> list[str]:
    review_dir = _ensure_directory(output_dir / "review_images")
    saved: list[str] = []
    for item in parity["missing"]:
        if len(saved) >= max_images:
            break
        frame = cv2.imread(item.frame_path)
        if frame is None:
            continue
        x1, y1, x2, y2 = [int(round(value)) for value in item.bbox_xyxy]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(frame, f"missing_{label}_{item.class_name}", (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        path = review_dir / f"{label}_{item.camera_id}_{item.frame_number}_{len(saved)+1:02d}.jpg"
        cv2.imwrite(str(path), frame)
        saved.append(str(path))
    return saved


def choose_recommendations(detector_rows: list[dict[str, Any]], parity_by_size: dict[int, dict[str, Any]]) -> dict[str, Any]:
    rows_12cam = {int(row["imgsz"]): row for row in detector_rows if int(row["camera_count"]) == 12}
    baseline_row = rows_12cam.get(1024)
    smaller_sizes = [size for size in sorted(parity_by_size) if size != 1024]
    best_performance_size = 1024
    if rows_12cam:
        best_performance_size = max(rows_12cam.values(), key=lambda row: float(row.get("pipeline_frames_per_second", 0.0) or 0.0))["imgsz"]
    recommended_size = 1024
    best_quality_performance_size = 1024
    qualified: list[tuple[float, int]] = []
    for size in smaller_sizes:
        parity = parity_by_size[size]
        overall = float(parity.get("match_rate", 0.0) or 0.0)
        small = float(parity.get("size_groups", {}).get("small", {}).get("match_rate", 0.0) or 0.0)
        if overall >= 0.985 and small >= 0.97 and size in rows_12cam:
            qualified.append((float(rows_12cam[size].get("pipeline_frames_per_second", 0.0) or 0.0), size))
    if qualified:
        qualified.sort(reverse=True)
        best_quality_performance_size = qualified[0][1]
        recommended_size = best_quality_performance_size
    throughput_gain = 0.0
    p95_latency_change = 0.0
    gpu_memory_change = 0.0
    if baseline_row is not None and recommended_size in rows_12cam:
        recommended_row = rows_12cam[recommended_size]
        baseline_fps = float(baseline_row.get("pipeline_frames_per_second", 0.0) or 0.0)
        recommended_fps = float(recommended_row.get("pipeline_frames_per_second", 0.0) or 0.0)
        if baseline_fps > 0.0:
            throughput_gain = ((recommended_fps - baseline_fps) / baseline_fps) * 100.0
        p95_latency_change = float(recommended_row.get("p95_total_detection_latency_ms", 0.0) or 0.0) - float(
            baseline_row.get("p95_total_detection_latency_ms", 0.0) or 0.0
        )
        gpu_memory_change = float(recommended_row.get("cuda_peak_allocated_mb", 0.0) or 0.0) - float(
            baseline_row.get("cuda_peak_allocated_mb", 0.0) or 0.0
        )
    return {
        "best_performance_size": int(best_performance_size),
        "best_quality_performance_size": int(best_quality_performance_size),
        "recommended_production_size": int(recommended_size),
        "throughput_gain_vs_1024_percent": float(throughput_gain),
        "p95_latency_change_vs_1024_ms": float(p95_latency_change),
        "gpu_memory_change_vs_1024_mb": float(gpu_memory_change),
    }


def write_summary(output_dir: Path, detector_rows: list[dict[str, Any]], full_pipeline_rows: list[dict[str, Any]], payload: dict[str, Any]) -> None:
    summary_rows = []
    for row in detector_rows + full_pipeline_rows:
        summary_rows.append(row)
    summary_csv = output_dir / "summary.csv"
    fieldnames = [
        "mode",
        "camera_count",
        "imgsz",
        "processed_total_frames",
        "expected_total_frames",
        "total_runtime_sec",
        "processing_runtime_sec",
        "yolo_model_invocations",
        "yolo_frames_per_second",
        "pipeline_frames_per_second",
        "average_total_detection_latency_ms",
        "p95_total_detection_latency_ms",
        "preprocess_mean_ms",
        "inference_mean_ms",
        "postprocess_mean_ms",
        "conversion_mean_ms",
        "gpu_memory_peak_mb",
        "cuda_peak_allocated_mb",
        "cuda_peak_reserved_mb",
        "frame_loss_count",
        "frame_order_violations",
        "fairness_classification",
        "primary_bottleneck",
        "run_id",
    ]
    with summary_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow({field: row.get(field) for field in fieldnames})
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_report(
    output_dir: Path,
    *,
    frame_count: int,
    profile_1024: dict[str, Any],
    detector_rows: list[dict[str, Any]],
    full_pipeline_rows: list[dict[str, Any]],
    parity_by_size: dict[int, dict[str, Any]],
    recommendations: dict[str, Any],
) -> Path:
    total_mean = float(profile_1024["total_detection"]["mean"] or 0.0)
    def _share(stage_mean: float) -> float:
        return (stage_mean * 100.0 / total_mean) if total_mean > 0.0 else 0.0
    lines = [
        "# YOLO IMGSZ Benchmark",
        "",
        "CURRENT IMGSZ = 1024",
        "PRODUCTION BATCH SIZE = 1",
        "YOLO MODEL INSTANCES = 1",
        "",
        f"Frame set size = {frame_count}",
        f"IoU threshold = {DEFAULT_IOU_THRESHOLD}",
        f"Size thresholds = small<{DEFAULT_SMALL_AREA_RATIO_THRESHOLD:.3f}, medium<{DEFAULT_MEDIUM_AREA_RATIO_THRESHOLD:.3f}, else large",
        "",
        "PROFILE @1024",
        "",
        f"PREPROCESS: mean={profile_1024['preprocess']['mean']:.3f} p95={profile_1024['preprocess']['p95']:.3f} share={_share(profile_1024['preprocess']['mean']):.2f}%",
        f"INFERENCE: mean={profile_1024['inference']['mean']:.3f} p95={profile_1024['inference']['p95']:.3f} share={_share(profile_1024['inference']['mean']):.2f}%",
        f"POSTPROCESS: mean={profile_1024['postprocess']['mean']:.3f} p95={profile_1024['postprocess']['p95']:.3f} share={_share(profile_1024['postprocess']['mean']):.2f}%",
        f"RESULT CONVERSION: mean={profile_1024['conversion']['mean']:.3f} p95={profile_1024['conversion']['p95']:.3f} share={_share(profile_1024['conversion']['mean']):.2f}%",
        f"TOTAL DETECTION: mean={profile_1024['total_detection']['mean']:.3f} p95={profile_1024['total_detection']['p95']:.3f}",
        "",
    ]
    for size in sorted({int(row['imgsz']) for row in detector_rows}):
        lines.extend([f"==============================", f"{size}", "=============================="])
        for camera_count in sorted({int(row['camera_count']) for row in detector_rows if int(row['imgsz']) == size}):
            row = next(row for row in detector_rows if int(row["imgsz"]) == size and int(row["camera_count"]) == camera_count)
            lines.append(
                f"{camera_count} CAM: YOLO FPS={row['yolo_frames_per_second']:.3f} p95 latency={row['p95_total_detection_latency_ms']:.3f} GPU peak={row.get('cuda_peak_allocated_mb')}"
            )
        lines.append("")
    for size in sorted(parity_by_size):
        if size == 1024:
            continue
        parity = parity_by_size[size]
        lines.extend(
            [
                f"{size} vs 1024",
                "",
                f"MATCH RATE = {float(parity['match_rate']) * 100.0:.2f}%",
                f"MISSING DETECTIONS = {parity['missing_detections']}",
                f"ADDITIONAL DETECTIONS = {parity['additional_detections']}",
                f"MEAN IOU = {parity['mean_bbox_iou']:.4f}",
                f"SMALL VEHICLE MATCH RATE = {float(parity['size_groups']['small']['match_rate']) * 100.0:.2f}%",
                f"MEDIUM VEHICLE MATCH RATE = {float(parity['size_groups']['medium']['match_rate']) * 100.0:.2f}%",
                f"LARGE VEHICLE MATCH RATE = {float(parity['size_groups']['large']['match_rate']) * 100.0:.2f}%",
                "",
            ]
        )
    lines.extend(
        [
            f"BEST PERFORMANCE SIZE = {recommendations['best_performance_size']}",
            f"BEST QUALITY/PERFORMANCE SIZE = {recommendations['best_quality_performance_size']}",
            f"RECOMMENDED PRODUCTION SIZE = {recommendations['recommended_production_size']}",
            f"THROUGHPUT GAIN VS 1024 = {recommendations['throughput_gain_vs_1024_percent']:.2f}%",
            f"P95 LATENCY CHANGE = {recommendations['p95_latency_change_vs_1024_ms']:.3f} ms",
            f"GPU MEMORY CHANGE = {recommendations['gpu_memory_change_vs_1024_mb']:.3f} MB",
        ]
    )
    path = output_dir / "report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> None:
    args = parse_args()
    logger = logging.getLogger("yolo-imgsz-benchmark")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler(sys.stdout))
    output_dir = _ensure_directory(Path(args.output_dir))
    validated_config = _validate_config(_load_raw_config(args.base_config), Path(args.base_config).expanduser().resolve())
    frame_items = collect_frame_set(outputs_root=ROOT / "outputs", target_count=args.frame_set_target)
    write_frame_set_manifest(output_dir, frame_items)

    detections_by_size: dict[int, list[BenchmarkDetection]] = {}
    profile_rows_by_size: dict[int, list[dict[str, Any]]] = {}
    for imgsz in args.imgsz_values:
        print(f"\n=== Profiling frame set @ imgsz={imgsz} ===")
        benchmark_result = run_frame_benchmark(validated_config, imgsz=imgsz, frame_items=frame_items, logger=logger)
        detections_by_size[int(imgsz)] = list(benchmark_result["detections"])
        profile_rows_by_size[int(imgsz)] = list(benchmark_result["profile_rows"])
        write_detection_rows(output_dir, imgsz=int(imgsz), detections=detections_by_size[int(imgsz)], profile_rows=profile_rows_by_size[int(imgsz)])

    profile_1024 = {
        "preprocess": _stage_summary(profile_rows_by_size[1024], "preprocess_ms"),
        "inference": _stage_summary(profile_rows_by_size[1024], "model_inference_ms"),
        "postprocess": _stage_summary(profile_rows_by_size[1024], "postprocess_ms"),
        "conversion": _stage_summary(profile_rows_by_size[1024], "result_conversion_ms"),
        "routing": _stage_summary(profile_rows_by_size[1024], "result_routing_ms"),
        "tracker_update": _stage_summary(profile_rows_by_size[1024], "tracker_update_ms"),
        "total_detection": _stage_summary(profile_rows_by_size[1024], "total_detection_ms"),
    }

    parity_by_size: dict[int, dict[str, Any]] = {1024: {"match_rate": 1.0, "missing_detections": 0, "additional_detections": 0, "mean_bbox_iou": 1.0, "size_groups": {"small": {"match_rate": 1.0}, "medium": {"match_rate": 1.0}, "large": {"match_rate": 1.0}}, "per_class": {}}}
    for imgsz in args.imgsz_values:
        if int(imgsz) == 1024:
            continue
        parity = summarize_parity(
            detections_by_size[1024],
            detections_by_size[int(imgsz)],
            iou_threshold=float(args.iou_threshold),
            small_threshold=float(args.small_area_threshold),
            medium_threshold=float(args.medium_area_threshold),
        )
        parity_by_size[int(imgsz)] = parity
        write_parity_csv(output_dir, label=f"{imgsz}_vs_1024", parity=parity)
        write_review_images(output_dir, parity=parity, label=f"{imgsz}_vs_1024")

    missing_review_rows = build_missing_detection_review(
        detections_by_size[1024],
        parity_by_size.get(896, {"missing": [], "matches": []}),
        parity_by_size.get(768, {"missing": [], "matches": []}),
    )
    write_missing_detection_review(output_dir, missing_review_rows)

    base_config = _read_yaml(Path(args.base_config))
    detector_rows: list[dict[str, Any]] = []
    for camera_count in args.camera_counts:
        for imgsz in args.imgsz_values:
            print(f"\n=== Detector-only {camera_count} cameras @ imgsz={imgsz} ===")
            config = build_performance_config(base_config, camera_count=int(camera_count), frame_limit=int(args.frame_limit), imgsz=int(imgsz), mode="detector_only")
            config_path = write_benchmark_config(output_dir, camera_count=int(camera_count), imgsz=int(imgsz), mode="detector_only", config=config)
            monitor = run_pipeline_subprocess(config_path, sample_interval_seconds=float(args.sample_interval_seconds))
            metrics = collect_run_metrics(
                Path(str(monitor["run_directory"])),
                camera_count=int(camera_count),
                frames_per_camera=int(args.frame_limit),
                imgsz=int(imgsz),
                mode="detector_only",
                monitor=monitor,
            )
            detector_rows.append(metrics)
            (output_dir / f"detector_only_{camera_count}cam_imgsz{imgsz}.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    recommendations = choose_recommendations(detector_rows, parity_by_size)
    full_pipeline_candidate = int(recommendations["best_quality_performance_size"])
    if full_pipeline_candidate == 1024:
        smaller_sizes = [int(size) for size in args.imgsz_values if int(size) != 1024]
        if smaller_sizes:
            full_pipeline_candidate = max(
                smaller_sizes,
                key=lambda size: next(
                    (float(row.get("pipeline_frames_per_second", 0.0) or 0.0) for row in detector_rows if int(row["camera_count"]) == 12 and int(row["imgsz"]) == int(size)),
                    0.0,
                ),
            )

    full_pipeline_rows: list[dict[str, Any]] = []
    for camera_count in [count for count in args.camera_counts if int(count) in {8, 12}]:
        for imgsz in sorted({1024, full_pipeline_candidate}):
            print(f"\n=== Full-pipeline {camera_count} cameras @ imgsz={imgsz} ===")
            config = build_performance_config(base_config, camera_count=int(camera_count), frame_limit=int(args.frame_limit), imgsz=int(imgsz), mode="full_pipeline")
            config_path = write_benchmark_config(output_dir, camera_count=int(camera_count), imgsz=int(imgsz), mode="full_pipeline", config=config)
            monitor = run_pipeline_subprocess(config_path, sample_interval_seconds=float(args.sample_interval_seconds))
            metrics = collect_run_metrics(
                Path(str(monitor["run_directory"])),
                camera_count=int(camera_count),
                frames_per_camera=int(args.frame_limit),
                imgsz=int(imgsz),
                mode="full_pipeline",
                monitor=monitor,
            )
            full_pipeline_rows.append(metrics)
            (output_dir / f"full_pipeline_{camera_count}cam_imgsz{imgsz}.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "frame_set_count": len(frame_items),
        "imgsz_values": [int(item) for item in args.imgsz_values],
        "detector_only": detector_rows,
        "full_pipeline": full_pipeline_rows,
        "profile_1024": profile_1024,
        "parity": {
            str(size): {
                key: value
                for key, value in parity.items()
                if key not in {"matches", "missing", "additional"}
            }
            for size, parity in parity_by_size.items()
        },
        "recommendations": recommendations,
    }
    write_summary(output_dir, detector_rows, full_pipeline_rows, payload)
    write_report(
        output_dir,
        frame_count=len(frame_items),
        profile_1024=profile_1024,
        detector_rows=detector_rows,
        full_pipeline_rows=full_pipeline_rows,
        parity_by_size=parity_by_size,
        recommendations=recommendations,
    )


if __name__ == "__main__":
    main()
