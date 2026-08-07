from __future__ import annotations

from pathlib import Path

import yaml

from scripts.run_scalability_benchmark import (
    ProcessSample,
    build_benchmark_config,
    classify_bottlenecks,
    classify_fairness,
    determine_primary_bottlenecks,
    summarize_process_samples,
    write_benchmark_config,
)


def _base_config() -> dict:
    return {
        "project": {"name": "benchmark"},
        "input": {
            "cameras": [
                {
                    "camera_id": "CAM_001",
                    "source_type": "video",
                    "source": "D:/old_files/vinfocom/traffic_far_video/fa_traff_1min.mp4",
                    "enabled": True,
                }
            ],
            "max_frames_per_camera": 20,
        },
        "ingestion": {
            "worker_count": 7,
            "frame_queue_size": 50,
            "per_camera_buffer_size": 4,
            "scheduler_policy": "round_robin",
        },
        "vehicle_enrichment": {
            "async_colour": {
                "worker_count": 2,
                "queue_size": 100,
            }
        },
    }


def test_build_benchmark_config_keeps_workers_fixed_and_expands_logical_cameras() -> None:
    config = build_benchmark_config(_base_config(), camera_count=4, frame_limit=30)
    cameras = config["input"]["cameras"]
    assert len(cameras) == 4
    assert [camera["camera_id"] for camera in cameras] == ["CAM_001", "CAM_002", "CAM_003", "CAM_004"]
    assert all(camera["source"] == cameras[0]["source"] for camera in cameras)
    assert config["input"]["max_frames_per_camera"] == 30
    assert config["ingestion"]["worker_count"] == 3
    assert config["ingestion"]["per_camera_buffer_size"] == 2
    assert config["ingestion"]["scheduler_policy"] == "round_robin"
    assert config["vehicle_enrichment"]["async_colour"]["worker_count"] == 1


def test_write_benchmark_config_uses_expected_filename(tmp_path: Path) -> None:
    config_path = write_benchmark_config(tmp_path, camera_count=8, config=build_benchmark_config(_base_config(), camera_count=8, frame_limit=30))
    saved = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert config_path.name == "config.benchmark_8cam.yaml"
    assert len(saved["input"]["cameras"]) == 8


def test_fairness_and_bottleneck_classification_favor_florence_when_colour_queue_is_high() -> None:
    fairness_classification, imbalance_percent = classify_fairness(
        {"CAM_001": 30, "CAM_002": 30, "CAM_003": 30, "CAM_004": 30},
        max_consecutive_frames_same_camera=2,
    )
    assert fairness_classification == "GOOD"
    assert imbalance_percent == 0.0

    metrics = {
        "camera_count": 8,
        "fairness_classification": fairness_classification,
        "fairness_imbalance_percent": imbalance_percent,
        "frame_loss_count": 0,
        "buffer_full_count": 0,
        "evidence_cache_misses": 0,
        "pending_jobs_shutdown": 12,
        "processed_total_frames": 240,
        "yolo_effective_fps": 8.5,
        "effective_pipeline_fps": 3.2,
        "colour_queue_peak_percent": 91.0,
        "detection_queue_peak_percent": 48.0,
        "tracker_states": 8,
        "ram_peak_mb": 2048.0,
    }
    statuses = classify_bottlenecks(metrics)
    primary, secondary = determine_primary_bottlenecks(metrics, statuses)
    assert statuses["FLORENCE"] == "BOTTLENECK"
    assert primary == "FLORENCE"
    assert secondary in {"YOLO", "FLORENCE"}


def test_summarize_process_samples_computes_peaks_and_averages() -> None:
    summary = summarize_process_samples(
        [
            ProcessSample(cpu_percent=10.0, ram_mb=100.0, thread_count=12, gpu_util_percent=25.0, gpu_memory_mb=500.0),
            ProcessSample(cpu_percent=30.0, ram_mb=150.0, thread_count=16, gpu_util_percent=35.0, gpu_memory_mb=750.0),
        ]
    )
    assert summary["cpu_avg_percent"] == 20.0
    assert summary["cpu_peak_percent"] == 30.0
    assert summary["ram_peak_mb"] == 150.0
    assert summary["thread_count_peak"] == 16
    assert summary["gpu_util_peak_percent"] == 35.0
    assert summary["gpu_memory_peak_mb"] == 750.0
