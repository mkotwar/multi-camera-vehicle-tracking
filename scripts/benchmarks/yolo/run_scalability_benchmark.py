from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psutil
import yaml


DEFAULT_CAMERA_COUNTS = [2, 4, 8, 12]
DEFAULT_FRAME_LIMIT = 30


@dataclass
class ProcessSample:
    cpu_percent: float | None
    ram_mb: float | None
    thread_count: int | None
    gpu_util_percent: float | None
    gpu_memory_mb: float | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run fixed-worker scalability benchmarks for the multicamera pipeline.")
    parser.add_argument(
        "--base-config",
        default="config/validation.yaml",
        help="Base YAML config used as the template for all logical camera benchmarks.",
    )
    parser.add_argument(
        "--camera-counts",
        nargs="+",
        type=int,
        default=DEFAULT_CAMERA_COUNTS,
        help="Logical camera counts to benchmark.",
    )
    parser.add_argument(
        "--frame-limit",
        type=int,
        default=DEFAULT_FRAME_LIMIT,
        help="Per-camera frame limit used for every benchmark run.",
    )
    parser.add_argument(
        "--output-dir",
        default="diagnostics/scalability_benchmark",
        help="Directory where benchmark configs and reports are written.",
    )
    parser.add_argument(
        "--sample-interval-seconds",
        type=float,
        default=1.0,
        help="Monitoring sample interval while the pipeline process is running.",
    )
    parser.add_argument(
        "--repeat-camera-count",
        type=int,
        default=0,
        help="Optional camera count to rerun once more for stability checks. Use 0 to skip.",
    )
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


def build_benchmark_config(base_config: dict[str, Any], *, camera_count: int, frame_limit: int) -> dict[str, Any]:
    config = json.loads(json.dumps(base_config))
    input_section = dict(config.get("input", {}) or {})
    cameras = list(input_section.get("cameras", []) or [])
    if not cameras:
        raise ValueError("Base config must contain at least one camera.")
    first_camera = dict(cameras[0])
    source = first_camera.get("source")
    source_type = first_camera.get("source_type")
    if source in (None, "") or source_type in (None, ""):
        raise ValueError("Base config first camera must contain source and source_type.")
    logical_cameras: list[dict[str, Any]] = []
    for index in range(camera_count):
        logical_cameras.append(
            {
                "camera_id": f"CAM_{index + 1:03d}",
                "source_type": source_type,
                "source": source,
                "enabled": True,
            }
        )
    input_section["cameras"] = logical_cameras
    input_section["max_frames_per_camera"] = frame_limit
    config["input"] = input_section

    ingestion = dict(config.get("ingestion", {}) or {})
    ingestion["worker_count"] = 3
    ingestion["per_camera_buffer_size"] = 2
    ingestion["scheduler_policy"] = "round_robin"
    config["ingestion"] = ingestion

    vehicle_enrichment = dict(config.get("vehicle_enrichment", {}) or {})
    async_colour = dict(vehicle_enrichment.get("async_colour", {}) or {})
    async_colour["worker_count"] = 1
    async_colour["queue_size"] = int(async_colour.get("queue_size", 100) or 100)
    vehicle_enrichment["async_colour"] = async_colour
    config["vehicle_enrichment"] = vehicle_enrichment
    return config


def write_benchmark_config(output_dir: Path, *, camera_count: int, config: dict[str, Any]) -> Path:
    configs_dir = _ensure_directory(output_dir / "configs")
    config_path = configs_dir / f"config.benchmark_{camera_count}cam.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return config_path


def _query_nvidia_smi() -> tuple[float | None, float | None]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
    except Exception:
        return None, None
    line = next((item.strip() for item in result.stdout.splitlines() if item.strip()), "")
    if not line:
        return None, None
    parts = [part.strip() for part in line.split(",")]
    if len(parts) < 2:
        return None, None
    try:
        return float(parts[0]), float(parts[1])
    except ValueError:
        return None, None


def monitor_process(process: subprocess.Popen[str], *, sample_interval_seconds: float) -> dict[str, Any]:
    ps_process = psutil.Process(process.pid)
    ps_process.cpu_percent(interval=None)
    samples: list[ProcessSample] = []
    peak_ram_mb = 0.0
    peak_threads = 0
    while process.poll() is None:
        time.sleep(sample_interval_seconds)
        try:
            cpu_percent = float(ps_process.cpu_percent(interval=None))
            ram_mb = float(ps_process.memory_info().rss / (1024 * 1024))
            thread_count = int(ps_process.num_threads())
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            break
        gpu_util_percent, gpu_memory_mb = _query_nvidia_smi()
        peak_ram_mb = max(peak_ram_mb, ram_mb)
        peak_threads = max(peak_threads, thread_count)
        samples.append(
            ProcessSample(
                cpu_percent=cpu_percent,
                ram_mb=ram_mb,
                thread_count=thread_count,
                gpu_util_percent=gpu_util_percent,
                gpu_memory_mb=gpu_memory_mb,
            )
        )
    return summarize_process_samples(samples, fallback_peak_ram_mb=peak_ram_mb, fallback_peak_threads=peak_threads)


def summarize_process_samples(
    samples: list[ProcessSample], *, fallback_peak_ram_mb: float = 0.0, fallback_peak_threads: int = 0
) -> dict[str, Any]:
    cpu_values = [sample.cpu_percent for sample in samples if sample.cpu_percent is not None]
    ram_values = [sample.ram_mb for sample in samples if sample.ram_mb is not None]
    thread_values = [sample.thread_count for sample in samples if sample.thread_count is not None]
    gpu_util_values = [sample.gpu_util_percent for sample in samples if sample.gpu_util_percent is not None]
    gpu_memory_values = [sample.gpu_memory_mb for sample in samples if sample.gpu_memory_mb is not None]
    return {
        "cpu_avg_percent": round(statistics.fmean(cpu_values), 3) if cpu_values and max(cpu_values) > 0.0 else None,
        "cpu_peak_percent": round(max(cpu_values), 3) if cpu_values and max(cpu_values) > 0.0 else None,
        "ram_peak_mb": round(max(ram_values + ([fallback_peak_ram_mb] if fallback_peak_ram_mb else [])), 3) if (ram_values or fallback_peak_ram_mb) else None,
        "thread_count_peak": int(max(thread_values + ([fallback_peak_threads] if fallback_peak_threads else []))) if (thread_values or fallback_peak_threads) else None,
        "gpu_util_avg_percent": round(statistics.fmean(gpu_util_values), 3) if gpu_util_values else None,
        "gpu_util_peak_percent": round(max(gpu_util_values), 3) if gpu_util_values else None,
        "gpu_memory_peak_mb": round(max(gpu_memory_values), 3) if gpu_memory_values else None,
    }


def run_pipeline_subprocess(config_path: Path, *, sample_interval_seconds: float) -> dict[str, Any]:
    command = [sys.executable, "app.py", "--config", str(config_path)]
    started_at = time.perf_counter()
    process = subprocess.Popen(
        command,
        cwd=str(Path.cwd()),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    stdout_lines: list[str] = []
    if process.stdout is None:
        raise RuntimeError("Failed to capture subprocess stdout.")
    def _read_stdout() -> None:
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            stdout_lines.append(line.rstrip("\n"))

    reader_thread = threading.Thread(target=_read_stdout, name="benchmark-stdout-reader", daemon=True)
    reader_thread.start()
    metrics_payload = monitor_process(process, sample_interval_seconds=sample_interval_seconds)
    return_code = process.wait()
    reader_thread.join(timeout=5.0)
    total_runtime_seconds = time.perf_counter() - started_at
    run_id = ""
    run_directory = ""
    for line in stdout_lines:
        if line.startswith("Run completed:"):
            run_id = line.split(":", 1)[1].strip()
        if line.startswith("Output:"):
            run_directory = line.split(":", 1)[1].strip()
    if return_code != 0:
        raise RuntimeError(f"Benchmark run failed for {config_path}. Exit code={return_code}")
    if not run_id or not run_directory:
        raise RuntimeError(f"Could not parse run output for {config_path}")
    metrics_payload["subprocess_command"] = command
    metrics_payload["stdout_lines"] = stdout_lines
    metrics_payload["total_runtime_seconds"] = round(total_runtime_seconds, 6)
    metrics_payload["run_id"] = run_id
    metrics_payload["run_directory"] = run_directory
    return metrics_payload


def classify_fairness(frames_sent: dict[str, int], *, max_consecutive_frames_same_camera: int) -> tuple[str, float]:
    if not frames_sent:
        return "POOR", 100.0
    values = list(frames_sent.values())
    minimum = min(values)
    maximum = max(values)
    imbalance_percent = 0.0 if maximum <= 0 else round(((maximum - minimum) / maximum) * 100.0, 3)
    if imbalance_percent <= 2.0 and max_consecutive_frames_same_camera <= 2:
        return "GOOD", imbalance_percent
    if imbalance_percent <= 10.0 and max_consecutive_frames_same_camera <= 4:
        return "PARTIAL", imbalance_percent
    return "POOR", imbalance_percent


def _safe_divide(numerator: float, denominator: float) -> float | None:
    if denominator <= 0.0:
        return None
    return numerator / denominator


def _round_or_none(value: float | None, digits: int = 3) -> float | None:
    return None if value is None else round(value, digits)


def classify_bottlenecks(metrics: dict[str, Any]) -> dict[str, str]:
    colour_queue_peak_percent = float(metrics.get("colour_queue_peak_percent", 0.0) or 0.0)
    detection_queue_peak_percent = float(metrics.get("detection_queue_peak_percent", 0.0) or 0.0)
    fairness = str(metrics.get("fairness_classification", "POOR"))
    frame_loss_count = int(metrics.get("frame_loss_count", 0) or 0)
    evidence_cache_misses = int(metrics.get("evidence_cache_misses", 0) or 0)
    colour_shutdown_pending = int(metrics.get("pending_jobs_shutdown", 0) or 0)
    processed_total_frames = int(metrics.get("processed_total_frames", 0) or 0)
    yolo_fps = float(metrics.get("yolo_effective_fps", 0.0) or 0.0)
    pipeline_fps = float(metrics.get("effective_pipeline_fps", 0.0) or 0.0)

    statuses = {
        "INGESTION": "OK",
        "SCHEDULER": "OK",
        "YOLO": "OK",
        "TRACKING": "OK",
        "EVIDENCE": "OK",
        "FLORENCE": "OK",
        "OUTPUT": "OK",
    }
    if fairness != "GOOD":
        statuses["SCHEDULER"] = "BOTTLENECK"
    if frame_loss_count > 0 or int(metrics.get("buffer_full_count", 0) or 0) > 0:
        statuses["INGESTION"] = "BOTTLENECK"
    if colour_queue_peak_percent >= 85.0 or colour_shutdown_pending > 0:
        statuses["FLORENCE"] = "BOTTLENECK"
    if (
        detection_queue_peak_percent >= 85.0
        and colour_queue_peak_percent < 85.0
    ) or (
        processed_total_frames > 0
        and yolo_fps > 0.0
        and pipeline_fps < (yolo_fps * 0.45)
        and colour_queue_peak_percent < 85.0
    ):
        statuses["YOLO"] = "BOTTLENECK"
    if evidence_cache_misses > 0:
        statuses["EVIDENCE"] = "BOTTLENECK"
    if int(metrics.get("tracker_states", 0) or 0) != int(metrics.get("camera_count", 0) or 0):
        statuses["TRACKING"] = "BOTTLENECK"
    if float(metrics.get("ram_peak_mb", 0.0) or 0.0) >= 12000.0:
        statuses["OUTPUT"] = "BOTTLENECK"
    return statuses


def determine_primary_bottlenecks(metrics: dict[str, Any], statuses: dict[str, str]) -> tuple[str, str]:
    if float(metrics.get("colour_queue_peak_percent", 0.0) or 0.0) >= 85.0:
        return "FLORENCE", "YOLO"
    candidates: list[tuple[str, float]] = []
    if statuses["FLORENCE"] == "BOTTLENECK":
        candidates.append(("FLORENCE", float(metrics.get("colour_queue_peak_percent", 0.0) or 0.0)))
    if statuses["YOLO"] == "BOTTLENECK":
        candidates.append(("YOLO", float(metrics.get("detection_queue_peak_percent", 0.0) or 0.0)))
    if statuses["INGESTION"] == "BOTTLENECK":
        candidates.append(("INGESTION", float(metrics.get("buffer_full_count", 0.0) or 0.0)))
    if statuses["SCHEDULER"] == "BOTTLENECK":
        candidates.append(("SCHEDULER", float(metrics.get("fairness_imbalance_percent", 0.0) or 0.0)))
    if statuses["EVIDENCE"] == "BOTTLENECK":
        candidates.append(("EVIDENCE", float(metrics.get("evidence_cache_misses", 0.0) or 0.0)))
    if not candidates:
        if float(metrics.get("colour_queue_peak_percent", 0.0) or 0.0) >= float(metrics.get("detection_queue_peak_percent", 0.0) or 0.0):
            return "FLORENCE", "YOLO"
        return "YOLO", "FLORENCE"
    candidates.sort(key=lambda item: item[1], reverse=True)
    primary = candidates[0][0]
    secondary = candidates[1][0] if len(candidates) > 1 else ("YOLO" if primary != "YOLO" else "FLORENCE")
    return primary, secondary


def collect_run_metrics(run_directory: Path, *, camera_count: int, frames_per_camera: int, monitor: dict[str, Any]) -> dict[str, Any]:
    summary = _read_json(run_directory / "summary.json")
    ingestion = _read_json(run_directory / "ingestion_metrics.json")
    detection = _read_json(run_directory / "detection_tracking_metrics.json")
    evidence = _read_json(run_directory / "evidence_metrics.json")
    enrichment = _read_json(run_directory / "vehicle_enrichment_metrics.json")
    run_config = _read_yaml(run_directory / "run_config.yaml")

    expected_total_frames = camera_count * frames_per_camera
    frames_processed = int(summary.get("processed_frames", 0) or 0)
    processing_runtime_sec = float(summary.get("overall_pipeline_runtime_ms", 0.0) or 0.0) / 1000.0
    yolo_total_inference_sec = float(detection.get("total_inference_time_ms", 0.0) or 0.0) / 1000.0
    florence_total_inference_sec = float(enrichment.get("vehicle_attribute_total_colour_inference_ms", 0.0) or 0.0) / 1000.0
    ingestion_queue_capacity = int(run_config.get("ingestion", {}).get("frame_queue_size", 0) or 0)
    colour_queue_capacity = int(summary.get("colour_queue_size", 0) or 0)
    fairness_classification, fairness_imbalance_percent = classify_fairness(
        dict(summary.get("frames_consumed_by_camera", {}) or {}),
        max_consecutive_frames_same_camera=int(summary.get("max_consecutive_frames_same_camera", 0) or 0),
    )
    pipeline_fps = _safe_divide(float(frames_processed), processing_runtime_sec)
    yolo_effective_fps = _safe_divide(float(frames_processed), yolo_total_inference_sec)
    florence_effective_fps = _safe_divide(float(enrichment.get("colour_inference_calls", 0) or 0), florence_total_inference_sec)
    detection_queue_peak = int(ingestion.get("maximum_observed_queue_size", 0) or 0)
    colour_queue_peak = int(summary.get("colour_queue_peak_depth", 0) or 0)
    detection_queue_peak_percent = _round_or_none(_safe_divide(float(detection_queue_peak * 100), float(ingestion_queue_capacity or 0)))
    colour_queue_peak_percent = _round_or_none(_safe_divide(float(colour_queue_peak * 100), float(colour_queue_capacity or 0)))
    cpu_avg_percent = monitor.get("cpu_avg_percent")
    cpu_peak_percent = monitor.get("cpu_peak_percent")
    ram_peak_mb = monitor.get("ram_peak_mb")
    thread_count_peak = monitor.get("thread_count_peak")
    if cpu_peak_percent in (0, 0.0):
        cpu_avg_percent = None
        cpu_peak_percent = None
    if ram_peak_mb is not None and ram_peak_mb < 16.0:
        ram_peak_mb = None
    if thread_count_peak is not None and thread_count_peak <= 2:
        thread_count_peak = None

    logical_cameras = list(run_config.get("input", {}).get("cameras", []) or [])
    logical_source_path = str(logical_cameras[0].get("source")) if logical_cameras else None
    metrics = {
        "benchmark_generated_at": datetime.now(timezone.utc).isoformat(),
        "camera_count": camera_count,
        "frames_per_camera": frames_per_camera,
        "expected_total_frames": expected_total_frames,
        "processed_total_frames": frames_processed,
        "frame_limit_mode": summary.get("frame_limit_mode"),
        "run_id": summary.get("run_id"),
        "run_directory": str(run_directory),
        "total_runtime_sec": float(monitor.get("total_runtime_seconds", 0.0) or 0.0),
        "processing_runtime_sec": processing_runtime_sec,
        "effective_pipeline_fps": _round_or_none(pipeline_fps),
        "effective_per_camera_fps": _round_or_none(_safe_divide(float(frames_processed), float(camera_count) * processing_runtime_sec if camera_count > 0 else 0.0)),
        "ingestion_workers": int(summary.get("ingestion_worker_count", summary.get("worker_count", 0)) or 0),
        "detection_workers": 1,
        "yolo_model_instances": 1,
        "colour_workers": int(summary.get("colour_worker_count", 0) or 0),
        "florence_model_instances": int(enrichment.get("florence_model_instances", 0) or 0),
        "camera_read_jobs": int(summary.get("camera_read_jobs", 0) or 0),
        "camera_read_failures": int(summary.get("camera_read_failures", 0) or 0),
        "frames_read_by_camera": dict(ingestion.get("frames_by_camera", {}) or {}),
        "frames_buffered_by_camera": dict(ingestion.get("frames_by_camera", {}) or {}),
        "frames_scheduled_by_camera": dict(summary.get("frames_scheduled_by_camera", {}) or {}),
        "frames_sent_to_detection_by_camera": dict(summary.get("frames_consumed_by_camera", {}) or {}),
        "per_camera_buffer_peak": dict(summary.get("per_camera_buffer_peak", {}) or {}),
        "buffer_full_count": int(summary.get("buffer_full_count", 0) or 0),
        "buffer_full_count_by_camera": dict(ingestion.get("buffer_full_count_by_camera", {}) or {}),
        "scheduler_skipped_empty_camera": int(summary.get("scheduler_skipped_empty_camera", 0) or 0),
        "max_consecutive_frames_same_camera": int(summary.get("max_consecutive_frames_same_camera", 0) or 0),
        "fairness_classification": fairness_classification,
        "fairness_imbalance_percent": fairness_imbalance_percent,
        "fairness_min_frames_sent": min((summary.get("frames_consumed_by_camera", {}) or {"_": 0}).values()),
        "fairness_max_frames_sent": max((summary.get("frames_consumed_by_camera", {}) or {"_": 0}).values()),
        "yolo_calls": frames_processed,
        "yolo_total_inference_time_ms": float(detection.get("total_inference_time_ms", 0.0) or 0.0),
        "yolo_avg_latency_ms": float(detection.get("average_inference_time_ms", 0.0) or 0.0),
        "yolo_median_latency_ms": None,
        "yolo_p95_latency_ms": None,
        "yolo_effective_fps": _round_or_none(yolo_effective_fps),
        "detection_queue_peak": detection_queue_peak,
        "detection_queue_capacity": ingestion_queue_capacity,
        "detection_queue_peak_percent": detection_queue_peak_percent,
        "tracker_states": int(detection.get("tracker_instances_created_total", 0) or 0),
        "tracker_updates": int(sum((detection.get("tracked_observations_by_camera", {}) or {}).values())),
        "tracked_observations": int(sum((detection.get("tracked_observations_by_camera", {}) or {}).values())),
        "completed_tracks": int(sum((summary.get("tracks_completed_by_camera", {}) or {}).values())),
        "discarded_tracks": int(sum((summary.get("tracks_discarded_by_camera", {}) or {}).values())),
        "evidence_cache_hits": int(evidence.get("evidence_cache_hits", 0) or 0),
        "evidence_cache_misses": int(evidence.get("evidence_cache_misses", 0) or 0),
        "evidence_cache_evictions": int(evidence.get("evidence_cache_evictions", 0) or 0),
        "evidence_cache_peak_frames": int(evidence.get("cache_peak_frames", 0) or 0),
        "skipped_evidence_items": int(evidence.get("evidence_items_skipped_missing_frame", 0) or 0),
        "partial_evidence_tracks": int(evidence.get("tracks_with_partial_evidence", 0) or 0),
        "pending_frame_reference_count": int(evidence.get("pending_frame_reference_count", 0) or 0),
        "colour_jobs_enqueued": int(summary.get("colour_jobs_enqueued", 0) or 0),
        "colour_jobs_started": int(enrichment.get("colour_jobs_started", 0) or 0),
        "colour_jobs_completed": int(summary.get("colour_jobs_completed", 0) or 0),
        "colour_jobs_failed": int(summary.get("colour_jobs_failed", 0) or 0),
        "colour_jobs_duplicate_attempts": int(summary.get("colour_jobs_duplicate_attempts", 0) or 0),
        "colour_jobs_lost": int(summary.get("colour_jobs_lost", 0) or 0),
        "colour_queue_peak": colour_queue_peak,
        "colour_queue_capacity": colour_queue_capacity,
        "colour_queue_peak_percent": colour_queue_peak_percent,
        "colour_queue_block_count": int(summary.get("colour_queue_block_count", 0) or 0),
        "colour_queue_block_time_ms": float(enrichment.get("colour_queue_block_time_ms", 0.0) or 0.0),
        "pending_jobs_shutdown": max(
            int(summary.get("pending_colour_jobs_at_shutdown", 0) or 0),
            int(enrichment.get("colour_worker_shutdown_pending_jobs", 0) or 0),
            int(evidence.get("pending_evidence_tracks_at_shutdown", 0) or 0),
        ),
        "florence_calls": int(enrichment.get("colour_inference_calls", 0) or 0),
        "florence_total_inference_time_ms": float(enrichment.get("vehicle_attribute_total_colour_inference_ms", 0.0) or 0.0),
        "florence_avg_latency_ms": float(enrichment.get("average_colour_inference_time_ms", 0.0) or 0.0),
        "florence_effective_fps": _round_or_none(florence_effective_fps),
        "valid_colour_predictions": int(enrichment.get("vehicle_attribute_valid_colour", 0) or 0),
        "unknown_colours": int(enrichment.get("vehicle_attribute_unknown_colour", 0) or 0),
        "frame_loss_count": max(0, expected_total_frames - frames_processed),
        "unprocessed_frames_at_shutdown": max(0, expected_total_frames - frames_processed),
        "cpu_avg_percent": cpu_avg_percent,
        "cpu_peak_percent": cpu_peak_percent,
        "ram_peak_mb": ram_peak_mb,
        "thread_count_peak": thread_count_peak,
        "gpu_util_avg_percent": monitor.get("gpu_util_avg_percent"),
        "gpu_util_peak_percent": monitor.get("gpu_util_peak_percent"),
        "gpu_memory_peak_mb": monitor.get("gpu_memory_peak_mb"),
        "cuda_device_name": summary.get("cuda_device_name"),
        "cuda_peak_allocated_mb": enrichment.get("gpu_memory_allocated_mb"),
        "cuda_peak_reserved_mb": enrichment.get("gpu_memory_reserved_mb"),
        "queue_full_events": int(summary.get("queue_full_events", 0) or 0),
        "logical_source_path": logical_source_path,
        "subprocess_command": monitor.get("subprocess_command"),
    }
    statuses = classify_bottlenecks(metrics)
    metrics["bottleneck_status"] = statuses
    primary, secondary = determine_primary_bottlenecks(metrics, statuses)
    metrics["primary_bottleneck"] = primary
    metrics["secondary_bottleneck"] = secondary
    metrics["notes"] = build_notes(metrics)
    return metrics


def build_notes(metrics: dict[str, Any]) -> str:
    notes: list[str] = []
    if int(metrics.get("evidence_cache_misses", 0) or 0) == 0:
        notes.append("evidence cache healthy")
    if float(metrics.get("colour_queue_peak_percent", 0.0) or 0.0) >= 80.0:
        notes.append("colour queue near saturation")
    if float(metrics.get("detection_queue_peak_percent", 0.0) or 0.0) >= 80.0:
        notes.append("detection queue near saturation")
    if int(metrics.get("frame_loss_count", 0) or 0) == 0:
        notes.append("no frame loss")
    if metrics.get("fairness_classification") == "GOOD":
        notes.append("camera fairness good")
    return "; ".join(notes)


def write_benchmark_json(output_dir: Path, metrics: dict[str, Any]) -> Path:
    path = output_dir / f"benchmark_{int(metrics['camera_count'])}cam.json"
    path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return path


def write_summary_csv(output_dir: Path, rows: list[dict[str, Any]]) -> Path:
    path = output_dir / "scalability_summary.csv"
    fieldnames = [
        "camera_count",
        "ingestion_workers",
        "detection_workers",
        "colour_workers",
        "frames_per_camera",
        "expected_total_frames",
        "processed_total_frames",
        "total_runtime_sec",
        "processing_runtime_sec",
        "effective_pipeline_fps",
        "effective_per_camera_fps",
        "yolo_calls",
        "yolo_avg_latency_ms",
        "yolo_effective_fps",
        "detection_queue_peak",
        "detection_queue_capacity",
        "tracker_states",
        "completed_tracks",
        "evidence_cache_hits",
        "evidence_cache_misses",
        "evidence_cache_peak_frames",
        "colour_jobs_enqueued",
        "colour_jobs_completed",
        "colour_jobs_failed",
        "colour_queue_peak",
        "colour_queue_capacity",
        "colour_queue_peak_percent",
        "colour_queue_block_count",
        "florence_calls",
        "florence_avg_latency_ms",
        "cpu_avg_percent",
        "cpu_peak_percent",
        "ram_peak_mb",
        "thread_count_peak",
        "gpu_util_avg_percent",
        "gpu_util_peak_percent",
        "gpu_memory_peak_mb",
        "cuda_peak_allocated_mb",
        "cuda_peak_reserved_mb",
        "frame_loss_count",
        "pending_jobs_shutdown",
        "fairness_classification",
        "primary_bottleneck",
        "secondary_bottleneck",
        "notes",
        "run_id",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})
    return path


def build_scalability_report(rows: list[dict[str, Any]], *, frames_per_camera: int, repeated_run: dict[str, Any] | None) -> str:
    table_lines = [
        "| Cameras | Frames | Runtime (s) | Pipeline FPS | YOLO FPS | Colour Queue Peak | Cache Misses | Bottleneck |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        table_lines.append(
            f"| {row['camera_count']} | {row['processed_total_frames']}/{row['expected_total_frames']} | {row['total_runtime_sec']:.3f} | "
            f"{row.get('effective_pipeline_fps') or 0:.3f} | {row.get('yolo_effective_fps') or 0:.3f} | "
            f"{row.get('colour_queue_peak') or 0}/{row.get('colour_queue_capacity') or 0} ({row.get('colour_queue_peak_percent') or 0}%) | "
            f"{row.get('evidence_cache_misses') or 0} | {row.get('primary_bottleneck')} |"
        )
    saturation_row = next(
        (
            row
            for row in rows
            if row.get("colour_queue_peak_percent", 0.0) >= 80.0
            or row.get("colour_queue_block_count", 0) > 0
            or row.get("pending_jobs_shutdown", 0) > 0
            or row.get("frame_loss_count", 0) > 0
            or row.get("fairness_classification") != "GOOD"
        ),
        rows[-1] if rows else None,
    )
    saturation_starts_at = saturation_row["camera_count"] if saturation_row is not None else "unknown"
    primary_bottleneck = saturation_row["primary_bottleneck"] if saturation_row is not None else "unknown"
    secondary_bottleneck = saturation_row["secondary_bottleneck"] if saturation_row is not None else "unknown"
    repeated_section = ""
    if repeated_run is not None:
        repeated_section = (
            "\n## Repeatability\n\n"
            f"- Repeated camera count: `{repeated_run['camera_count']}`\n"
            f"- Repeat run ID: `{repeated_run['run_id']}`\n"
            f"- Runtime: `{repeated_run['total_runtime_sec']}` seconds\n"
            f"- Pipeline FPS: `{repeated_run.get('effective_pipeline_fps')}`\n"
            f"- Primary bottleneck: `{repeated_run.get('primary_bottleneck')}`\n"
        )
    return (
        "# Scalability Benchmark\n\n"
        "## Test Conditions\n\n"
        f"- Frames per camera: `{frames_per_camera}`\n"
        "- Ingestion workers: `3`\n"
        "- Detection workers: `1`\n"
        "- YOLO model instances: `1`\n"
        "- Colour workers: `1`\n"
        "- Florence model instances: `1`\n"
        "- Logical camera counts: `2, 4, 8, 12`\n"
        "- Source reuse mode: one independent reader handle per logical camera, same local video path reused safely\n\n"
        "## Comparison\n\n"
        + "\n".join(table_lines)
        + "\n\n## Findings\n\n"
        f"- Saturation starts at: `{saturation_starts_at}` cameras\n"
        f"- Primary bottleneck: `{primary_bottleneck}`\n"
        f"- Secondary bottleneck: `{secondary_bottleneck}`\n"
        f"- Recommended next optimization: `{recommend_next_step(rows)}`\n"
        + repeated_section
    )


def recommend_next_step(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "Measure first"
    heaviest = max(rows, key=lambda item: int(item.get("camera_count", 0) or 0))
    primary = str(heaviest.get("primary_bottleneck", ""))
    mapping = {
        "YOLO": "A. YOLO batching",
        "FLORENCE": "F. optimize Florence",
        "INGESTION": "C. increase ingestion worker count",
        "SCHEDULER": "D. improve scheduler",
        "OUTPUT": "G. output worker",
        "EVIDENCE": "G. output worker",
    }
    return mapping.get(primary, "B. target-FPS/frame sampling")


def save_report(output_dir: Path, report_text: str) -> Path:
    path = output_dir / "scalability_report.md"
    path.write_text(report_text, encoding="utf-8")
    return path


def run_single_benchmark(
    *,
    base_config: dict[str, Any],
    output_dir: Path,
    camera_count: int,
    frame_limit: int,
    sample_interval_seconds: float,
) -> dict[str, Any]:
    benchmark_config = build_benchmark_config(base_config, camera_count=camera_count, frame_limit=frame_limit)
    config_path = write_benchmark_config(output_dir, camera_count=camera_count, config=benchmark_config)
    monitor = run_pipeline_subprocess(config_path, sample_interval_seconds=sample_interval_seconds)
    metrics = collect_run_metrics(
        Path(str(monitor["run_directory"])),
        camera_count=camera_count,
        frames_per_camera=frame_limit,
        monitor=monitor,
    )
    write_benchmark_json(output_dir, metrics)
    return metrics


def main() -> int:
    args = parse_args()
    output_dir = _ensure_directory(Path(args.output_dir).expanduser().resolve())
    base_config_path = Path(args.base_config).expanduser().resolve()
    base_config = _read_yaml(base_config_path)
    rows: list[dict[str, Any]] = []
    for camera_count in args.camera_counts:
        print(f"\n=== Running {camera_count}-camera benchmark ===")
        rows.append(
            run_single_benchmark(
                base_config=base_config,
                output_dir=output_dir,
                camera_count=camera_count,
                frame_limit=args.frame_limit,
                sample_interval_seconds=args.sample_interval_seconds,
            )
        )
    repeated_run: dict[str, Any] | None = None
    if args.repeat_camera_count > 0:
        print(f"\n=== Repeating {args.repeat_camera_count}-camera benchmark ===")
        repeated_run = run_single_benchmark(
            base_config=base_config,
            output_dir=output_dir,
            camera_count=args.repeat_camera_count,
            frame_limit=args.frame_limit,
            sample_interval_seconds=args.sample_interval_seconds,
        )
        repeated_run["is_repeat"] = True
        repeated_run["notes"] = f"repeat run; {repeated_run.get('notes', '')}".strip()
        repeat_path = output_dir / f"benchmark_{args.repeat_camera_count}cam_repeat.json"
        repeat_path.write_text(json.dumps(repeated_run, indent=2), encoding="utf-8")
    write_summary_csv(output_dir, rows)
    report_text = build_scalability_report(rows, frames_per_camera=args.frame_limit, repeated_run=repeated_run)
    save_report(output_dir, report_text)
    print(f"\nBenchmark artifacts written to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
