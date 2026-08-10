from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_scalability_benchmark import (
    build_notes,
    classify_bottlenecks,
    classify_fairness,
    determine_primary_bottlenecks,
    run_pipeline_subprocess,
)


DEFAULT_CAMERA_COUNTS = [4, 8, 12]
DEFAULT_BATCH_SIZES = [1, 2, 4]
DEFAULT_BATCH_WAITS_MS = [0.0]
DEFAULT_MODES = ["detector_only", "full_pipeline"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark true shared-YOLO micro-batching for multi-camera frames.")
    parser.add_argument("--base-config", default="config/validation.yaml")
    parser.add_argument("--output-dir", default="diagnostics/yolo_batch_benchmark")
    parser.add_argument("--camera-counts", nargs="+", type=int, default=DEFAULT_CAMERA_COUNTS)
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=DEFAULT_BATCH_SIZES)
    parser.add_argument("--batch-waits-ms", nargs="+", type=float, default=DEFAULT_BATCH_WAITS_MS)
    parser.add_argument("--modes", nargs="+", choices=["detector_only", "full_pipeline"], default=DEFAULT_MODES)
    parser.add_argument("--frame-limit", type=int, default=30)
    parser.add_argument("--sample-interval-seconds", type=float, default=1.0)
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


def build_benchmark_config(
    base_config: dict[str, Any],
    *,
    camera_count: int,
    frame_limit: int,
    batch_size: int,
    batch_wait_ms: float,
    mode: str,
) -> dict[str, Any]:
    config = json.loads(json.dumps(base_config))
    input_section = dict(config.get("input", {}) or {})
    cameras = list(input_section.get("cameras", []) or [])
    if not cameras:
        raise ValueError("Base config must contain at least one camera.")
    first_camera = dict(cameras[0])
    source = first_camera.get("source")
    source_type = first_camera.get("source_type")
    logical_cameras = [
        {
            "camera_id": f"CAM_{index + 1:03d}",
            "source_type": source_type,
            "source": source,
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
    detection["batch"] = {
        "enabled": int(batch_size) > 1,
        "max_size": int(batch_size),
        "max_wait_ms": float(batch_wait_ms),
    }
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


def write_benchmark_config(
    output_dir: Path,
    *,
    camera_count: int,
    batch_size: int,
    batch_wait_ms: float,
    mode: str,
    config: dict[str, Any],
) -> Path:
    configs_dir = _ensure_directory(output_dir / "configs")
    wait_label = str(batch_wait_ms).replace(".", "p")
    config_path = configs_dir / f"config.{mode}.{camera_count}cam.batch{batch_size}.wait{wait_label}.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return config_path


def collect_run_metrics(
    run_directory: Path,
    *,
    camera_count: int,
    frames_per_camera: int,
    batch_size: int,
    batch_wait_ms: float,
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
        "batch_size": batch_size,
        "batch_wait_ms": batch_wait_ms,
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
        "average_detection_batch_size": float(detection.get("average_detection_batch_size", 1.0) or 1.0),
        "max_detection_batch_size_observed": int(detection.get("max_detection_batch_size_observed", 1) or 1),
        "partial_detection_batches": int(detection.get("partial_detection_batches", 0) or 0),
        "detection_batch_wait_time_ms_avg": float(detection.get("detection_batch_wait_time_ms_avg", 0.0) or 0.0),
        "detection_batch_wait_time_ms_max": float(detection.get("detection_batch_wait_time_ms_max", 0.0) or 0.0),
        "yolo_inference_time_per_batch_avg_ms": float(detection.get("yolo_inference_time_per_batch_avg_ms", 0.0) or 0.0),
        "yolo_inference_time_per_frame_avg_ms": float(detection.get("yolo_inference_time_per_frame_avg_ms", 0.0) or 0.0),
        "detection_latency_ms_avg": float(detection.get("detection_latency_ms_avg", 0.0) or 0.0),
        "detection_latency_ms_p50": float(detection.get("detection_latency_ms_p50", 0.0) or 0.0),
        "detection_latency_ms_p95": float(detection.get("detection_latency_ms_p95", 0.0) or 0.0),
        "detection_latency_ms_max": float(detection.get("detection_latency_ms_max", 0.0) or 0.0),
        "detection_queue_peak": int(ingestion.get("maximum_observed_queue_size", 0) or 0),
        "detection_queue_capacity": int(run_config.get("ingestion", {}).get("frame_queue_size", 0) or 0),
        "frame_order_violations": int(detection.get("frame_order_violations", 0) or 0),
        "frame_loss_count": max(0, expected_total_frames - frames_processed),
        "per_camera_frames_processed": dict(summary.get("frames_consumed_by_camera", {}) or {}),
        "per_camera_detection_count": dict(detection.get("detections_by_camera", {}) or {}),
        "fairness_classification": fairness_classification,
        "fairness_imbalance_percent": fairness_imbalance_percent,
        "gpu_memory_peak_mb": monitor.get("gpu_memory_peak_mb"),
        "cuda_peak_allocated_mb": detection.get("gpu_peak_allocated_mb"),
        "cuda_peak_reserved_mb": detection.get("gpu_peak_reserved_mb"),
        "colour_jobs_enqueued": int(summary.get("colour_jobs_enqueued", 0) or 0),
        "colour_queue_peak": int(summary.get("colour_queue_peak_depth", 0) or 0),
        "evidence_cache_misses": int(evidence.get("evidence_cache_misses", 0) or 0),
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
            "evidence_cache_misses": metrics["evidence_cache_misses"],
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
            "evidence_cache_misses": metrics["evidence_cache_misses"],
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


def write_run_json(output_dir: Path, metrics: dict[str, Any]) -> Path:
    wait_label = str(metrics["batch_wait_ms"]).replace(".", "p")
    path = output_dir / f"{metrics['mode']}_{int(metrics['camera_count'])}cam_batch{int(metrics['batch_size'])}_wait{wait_label}.json"
    path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return path


def write_summary_csv(output_dir: Path, rows: list[dict[str, Any]]) -> Path:
    path = output_dir / "summary.csv"
    fieldnames = [
        "mode",
        "camera_count",
        "batch_size",
        "batch_wait_ms",
        "processed_total_frames",
        "expected_total_frames",
        "total_runtime_sec",
        "processing_runtime_sec",
        "yolo_model_invocations",
        "yolo_frames_per_second",
        "pipeline_frames_per_second",
        "average_detection_batch_size",
        "max_detection_batch_size_observed",
        "detection_latency_ms_p95",
        "gpu_memory_peak_mb",
        "cuda_peak_allocated_mb",
        "cuda_peak_reserved_mb",
        "frame_loss_count",
        "frame_order_violations",
        "fairness_classification",
        "primary_bottleneck",
        "run_id",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})
    return path


def write_summary_json(output_dir: Path, rows: list[dict[str, Any]]) -> Path:
    path = output_dir / "summary.json"
    path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return path


def write_report(output_dir: Path, rows: list[dict[str, Any]]) -> Path:
    lines = [
        "# YOLO Batch Benchmark",
        "",
        "| Mode | Cameras | Batch | Wait ms | Runtime s | YOLO FPS | Pipeline FPS | Model Calls | Avg Batch | Latency P95 ms | GPU Peak MB | Order Violations | Bottleneck |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['mode']} | {row['camera_count']} | {row['batch_size']} | {row['batch_wait_ms']} | "
            f"{row['total_runtime_sec']:.3f} | {row['yolo_frames_per_second']:.3f} | {row['pipeline_frames_per_second']:.3f} | "
            f"{row['yolo_model_invocations']} | {row['average_detection_batch_size']:.3f} | {row['detection_latency_ms_p95']:.3f} | "
            f"{row.get('gpu_memory_peak_mb') or 0} | {row['frame_order_violations']} | {row['primary_bottleneck']} |"
        )
    path = output_dir / "report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> None:
    args = parse_args()
    base_config = _read_yaml(Path(args.base_config))
    output_dir = _ensure_directory(Path(args.output_dir))
    rows: list[dict[str, Any]] = []
    for mode in args.modes:
        for camera_count in args.camera_counts:
            for batch_size in args.batch_sizes:
                for batch_wait_ms in args.batch_waits_ms:
                    if batch_size == 1 and batch_wait_ms not in (0, 0.0):
                        continue
                    print(f"\n=== Running {mode} {camera_count}-camera batch={batch_size} wait_ms={batch_wait_ms} ===")
                    benchmark_config = build_benchmark_config(
                        base_config,
                        camera_count=camera_count,
                        frame_limit=args.frame_limit,
                        batch_size=batch_size,
                        batch_wait_ms=batch_wait_ms,
                        mode=mode,
                    )
                    config_path = write_benchmark_config(
                        output_dir,
                        camera_count=camera_count,
                        batch_size=batch_size,
                        batch_wait_ms=batch_wait_ms,
                        mode=mode,
                        config=benchmark_config,
                    )
                    try:
                        monitor = run_pipeline_subprocess(config_path, sample_interval_seconds=args.sample_interval_seconds)
                        metrics = collect_run_metrics(
                            Path(str(monitor["run_directory"])),
                            camera_count=camera_count,
                            frames_per_camera=args.frame_limit,
                            batch_size=batch_size,
                            batch_wait_ms=batch_wait_ms,
                            mode=mode,
                            monitor=monitor,
                        )
                    except Exception as exc:
                        metrics = {
                            "mode": mode,
                            "camera_count": camera_count,
                            "batch_size": batch_size,
                            "batch_wait_ms": batch_wait_ms,
                            "failed": True,
                            "error": str(exc),
                        }
                    rows.append(metrics)
                    write_run_json(output_dir, metrics)
    write_summary_csv(output_dir, rows)
    write_summary_json(output_dir, rows)
    write_report(output_dir, rows)
    print(f"\nBenchmark artifacts written to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
