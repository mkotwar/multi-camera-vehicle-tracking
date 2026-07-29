from __future__ import annotations

import logging
import queue
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.ingestion_manager import MultiCameraIngestionManager
from src.models import ConfigurationError


def _create_test_video(path: Path, *, fps: float = 10.0, frame_count: int = 4, width: int = 32, height: int = 24) -> Path:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    for index in range(frame_count):
        frame = np.full((height, width, 3), index * 30, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return path


def _build_config(
    cameras: list[dict[str, object]],
    *,
    worker_count: int = 7,
    frame_queue_size: int = 20,
    max_frames_per_camera: int = 10,
    stop_on_camera_error: bool = False,
) -> dict[str, object]:
    return {
        "project": {"name": "test_project", "environment": "test", "log_level": "DEBUG"},
        "input": {"cameras": cameras, "max_frames_per_camera": max_frames_per_camera},
        "ingestion": {
            "worker_count": worker_count,
            "frame_queue_size": frame_queue_size,
            "queue_put_timeout_seconds": 0.05,
            "queue_get_timeout_seconds": 0.05,
            "stop_on_camera_error": stop_on_camera_error,
            "round_robin": True,
            "raw_frames": {
                "enabled": True,
                "save_every_n_frames": 2,
                "max_saved_frames_per_camera": 5,
                "image_format": "jpg",
                "jpeg_quality": 90,
            },
        },
        "output": {"root_directory": "outputs/runs", "save_run_config": True},
    }


def _logger() -> logging.Logger:
    logger = logging.getLogger("test-ingestion-manager")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    logger.propagate = False
    logger.addHandler(logging.NullHandler())
    return logger


def _collect_all_packets(manager: MultiCameraIngestionManager) -> list:
    packets = []
    while True:
        try:
            packet = manager.get_packet(timeout=0.05)
        except queue.Empty:
            if manager.is_finished() and manager.frame_queue.empty():
                break
            continue
        packets.append(packet)
        manager.mark_task_done()
    return packets


def test_seven_workers_are_created_by_default_and_worker_count_is_configurable(tmp_path: Path) -> None:
    video_path = _create_test_video(tmp_path / "sample.mp4")
    base_cameras = [{"camera_id": "CAM_001", "source_type": "video", "source": str(video_path), "enabled": True}]
    default_manager = MultiCameraIngestionManager(_build_config(base_cameras), _logger())
    configurable_manager = MultiCameraIngestionManager(_build_config(base_cameras, worker_count=3), _logger())
    assert default_manager.worker_count == 7
    assert len(default_manager.worker_threads) == 7
    assert configurable_manager.worker_count == 3
    assert len(configurable_manager.worker_threads) == 3


def test_camera_assignment_is_deterministic_for_ten_cameras(tmp_path: Path) -> None:
    cameras = []
    for index in range(10):
        path = _create_test_video(tmp_path / f"cam_{index:02d}.mp4")
        cameras.append({"camera_id": f"CAM_{index + 1:03d}", "source_type": "video", "source": str(path), "enabled": True})
    manager = MultiCameraIngestionManager(_build_config(cameras), _logger())
    assert manager.camera_assignments[0] == ["CAM_001", "CAM_008"]
    assert manager.camera_assignments[1] == ["CAM_002", "CAM_009"]
    assert manager.camera_assignments[2] == ["CAM_003", "CAM_010"]
    assert manager.camera_assignments[3] == ["CAM_004"]
    assert manager.camera_assignments[4] == ["CAM_005"]
    assert manager.camera_assignments[5] == ["CAM_006"]
    assert manager.camera_assignments[6] == ["CAM_007"]


def test_disabled_cameras_are_ignored_and_one_reader_is_created_per_enabled_camera(tmp_path: Path) -> None:
    video_a = _create_test_video(tmp_path / "a.mp4")
    video_b = _create_test_video(tmp_path / "b.mp4")
    config = _build_config(
        [
            {"camera_id": "CAM_001", "source_type": "video", "source": str(video_a), "enabled": True},
            {"camera_id": "CAM_002", "source_type": "video", "source": str(video_b), "enabled": False},
        ]
    )
    manager = MultiCameraIngestionManager(config, _logger())
    assert len(manager.enabled_cameras) == 1
    assert list(manager.readers_by_camera) == ["CAM_001"]


def test_duplicate_camera_ids_and_unsupported_source_type_raise_configuration_error(tmp_path: Path) -> None:
    video_a = _create_test_video(tmp_path / "a.mp4")
    duplicate_config = _build_config(
        [
            {"camera_id": "CAM_001", "source_type": "video", "source": str(video_a), "enabled": True},
            {"camera_id": "CAM_001", "source_type": "video", "source": str(video_a), "enabled": True},
        ]
    )
    bad_source_config = _build_config(
        [{"camera_id": "CAM_001", "source_type": "file", "source": str(video_a), "enabled": True}]
    )
    with pytest.raises(ConfigurationError):
        MultiCameraIngestionManager(duplicate_config, _logger())
    with pytest.raises(ConfigurationError):
        MultiCameraIngestionManager(bad_source_config, _logger())


def test_frame_order_is_preserved_and_counters_are_independent_per_camera(tmp_path: Path) -> None:
    video_a = _create_test_video(tmp_path / "a.mp4", frame_count=3)
    video_b = _create_test_video(tmp_path / "b.mp4", frame_count=4)
    manager = MultiCameraIngestionManager(
        _build_config(
            [
                {"camera_id": "CAM_001", "source_type": "video", "source": str(video_a), "enabled": True},
                {"camera_id": "CAM_002", "source_type": "video", "source": str(video_b), "enabled": True},
            ]
        ),
        _logger(),
    )
    manager.start()
    packets = _collect_all_packets(manager)
    manager.stop()
    by_camera: dict[str, list[int]] = {"CAM_001": [], "CAM_002": []}
    for packet in packets:
        by_camera[packet.camera_id].append(packet.frame_number)
    assert by_camera["CAM_001"] == [0, 1, 2]
    assert by_camera["CAM_002"] == [0, 1, 2, 3]


def test_one_camera_ending_or_failing_does_not_stop_others_when_configured(tmp_path: Path) -> None:
    video_a = _create_test_video(tmp_path / "a.mp4", frame_count=2)
    video_b = _create_test_video(tmp_path / "b.mp4", frame_count=5)
    manager = MultiCameraIngestionManager(
        _build_config(
            [
                {"camera_id": "CAM_001", "source_type": "video", "source": str(video_a), "enabled": True},
                {"camera_id": "CAM_002", "source_type": "video", "source": str(video_b), "enabled": True},
                {"camera_id": "CAM_003", "source_type": "video", "source": str(tmp_path / "missing.mp4"), "enabled": True},
            ],
            stop_on_camera_error=False,
        ),
        _logger(),
    )
    manager.start()
    packets = _collect_all_packets(manager)
    manager.stop()
    metrics = manager.get_metrics()
    counts = {"CAM_001": 0, "CAM_002": 0}
    for packet in packets:
        counts[packet.camera_id] += 1
    assert counts["CAM_001"] == 2
    assert counts["CAM_002"] == 5
    assert "CAM_003" in metrics["camera_errors"]


def test_bounded_queue_timeout_does_not_deadlock_and_all_workers_stop_and_release_resources(tmp_path: Path) -> None:
    video_a = _create_test_video(tmp_path / "a.mp4", frame_count=6)
    manager = MultiCameraIngestionManager(
        _build_config(
            [{"camera_id": "CAM_001", "source_type": "video", "source": str(video_a), "enabled": True}],
            frame_queue_size=1,
            max_frames_per_camera=6,
        ),
        _logger(),
    )
    manager.start()
    time.sleep(0.1)
    packets = _collect_all_packets(manager)
    manager.stop()
    assert manager.frame_queue.maxsize == 1
    assert len(packets) == 6
    assert manager.all_workers_stopped() is True
    assert manager.readers_by_camera["CAM_001"].released is True
    assert manager.get_metrics()["queue_full_events"] >= 0
