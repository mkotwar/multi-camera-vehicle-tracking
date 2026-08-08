from __future__ import annotations

from pathlib import Path

import yaml

from scripts.run_yolo_batch_benchmark import build_benchmark_config, write_benchmark_config


def _base_config() -> dict:
    return {
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
            "per_camera_buffer_size": 4,
            "scheduler_policy": "round_robin",
        },
        "detection": {
            "model_path": "D:/project/models/best_old.pt",
            "confidence_threshold": 0.2,
            "image_size": 1024,
        },
        "vehicle_enrichment": {
            "enabled": True,
            "async_colour": {
                "enabled": True,
                "worker_count": 2,
                "queue_size": 100,
            },
        },
    }


def test_build_benchmark_config_sets_batch_and_disables_colour_for_detector_only() -> None:
    config = build_benchmark_config(
        _base_config(),
        camera_count=4,
        frame_limit=30,
        batch_size=4,
        batch_wait_ms=5.0,
        mode="detector_only",
    )
    assert len(config["input"]["cameras"]) == 4
    assert config["input"]["max_frames_per_camera"] == 30
    assert config["detection"]["batch"] == {"enabled": True, "max_size": 4, "max_wait_ms": 5.0}
    assert config["vehicle_enrichment"]["enabled"] is False
    assert config["vehicle_enrichment"]["async_colour"]["enabled"] is False


def test_write_benchmark_config_uses_mode_batch_and_wait_in_name(tmp_path: Path) -> None:
    config_path = write_benchmark_config(
        tmp_path,
        camera_count=8,
        batch_size=2,
        batch_wait_ms=10.0,
        mode="full_pipeline",
        config=build_benchmark_config(
            _base_config(),
            camera_count=8,
            frame_limit=30,
            batch_size=2,
            batch_wait_ms=10.0,
            mode="full_pipeline",
        ),
    )
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert config_path.name == "config.full_pipeline.8cam.batch2.wait10p0.yaml"
    assert payload["detection"]["batch"]["max_size"] == 2
