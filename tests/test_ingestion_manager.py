from __future__ import annotations

import logging
import queue
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.ingestion_manager import MultiCameraIngestionManager
from src.models import ConfigurationError, FramePacket


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
    max_frames_per_camera: int | None = 10,
    stop_on_camera_error: bool = False,
) -> dict[str, object]:
    return {
        "project": {"name": "test_project", "environment": "test", "log_level": "DEBUG"},
        "input": {"cameras": cameras, "max_frames_per_camera": max_frames_per_camera},
        "ingestion": {
            "worker_count": worker_count,
            "frame_queue_size": frame_queue_size,
            "per_camera_buffer_size": 2,
            "scheduler_policy": "round_robin",
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


class _FakeReader:
    def __init__(
        self,
        camera_id: str,
        *,
        frame_count: int,
        delay_seconds: float = 0.0,
        fail_on_read_number: int | None = None,
    ) -> None:
        self.camera_id = camera_id
        self.source_type = "video"
        self.source_display = f"fake://{camera_id}"
        self.released = False
        self._frame_count = frame_count
        self._delay_seconds = delay_seconds
        self._fail_on_read_number = fail_on_read_number
        self._frame_number = 0
        self._active_reads = 0
        self.max_active_reads = 0
        self.read_calls = 0

    def read_next_frame(self, *, worker_id: int) -> FramePacket | None:
        self._active_reads += 1
        self.max_active_reads = max(self.max_active_reads, self._active_reads)
        try:
            if self._active_reads > 1:
                raise RuntimeError(f"concurrent read detected for {self.camera_id}")
            if self._fail_on_read_number is not None and self._frame_number == self._fail_on_read_number:
                raise RuntimeError(f"synthetic failure for {self.camera_id}")
            if self._delay_seconds > 0.0:
                time.sleep(self._delay_seconds)
            if self._frame_number >= self._frame_count:
                return None
            frame_number = self._frame_number
            self._frame_number += 1
            self.read_calls += 1
            frame = np.full((16, 16, 3), frame_number, dtype=np.uint8)
            return FramePacket(
                camera_id=self.camera_id,
                frame_number=frame_number,
                timestamp_seconds=float(frame_number) / 30.0,
                source_fps=30.0,
                frame=frame,
                source_frame_width=16,
                source_frame_height=16,
                worker_id=worker_id,
                captured_at="2026-08-07T00:00:00+00:00",
                source_type="video",
            )
        finally:
            self._active_reads -= 1

    def close(self) -> None:
        self.released = True


def test_seven_workers_are_created_by_default_and_worker_count_is_configurable(tmp_path: Path) -> None:
    video_path = _create_test_video(tmp_path / "sample.mp4")
    base_cameras = [{"camera_id": "CAM_001", "source_type": "video", "source": str(video_path), "enabled": True}]
    default_manager = MultiCameraIngestionManager(_build_config(base_cameras), _logger())
    configurable_manager = MultiCameraIngestionManager(_build_config(base_cameras, worker_count=3), _logger())
    assert default_manager.worker_count == 7
    assert len(default_manager.worker_threads) == 7
    assert configurable_manager.worker_count == 3
    assert len(configurable_manager.worker_threads) == 3
    assert configurable_manager.scheduler_thread.name == "ingestion-scheduler"


def test_ten_cameras_with_three_workers_do_not_create_one_worker_per_camera(tmp_path: Path) -> None:
    cameras = []
    for index in range(10):
        path = _create_test_video(tmp_path / f"cam_{index:02d}.mp4")
        cameras.append({"camera_id": f"CAM_{index + 1:03d}", "source_type": "video", "source": str(path), "enabled": True})
    manager = MultiCameraIngestionManager(_build_config(cameras, worker_count=3), _logger())
    metrics = manager.get_metrics()
    assert len(manager.worker_threads) == 3
    assert metrics["ingestion_worker_count"] == 3
    assert metrics["camera_count"] == 10
    assert metrics["camera_assignment_mode"] == "dynamic_task_queue"


def test_disabled_cameras_are_ignored_and_camera_registry_contains_only_enabled_sources(tmp_path: Path) -> None:
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
    assert list(manager.camera_sources) == ["CAM_001"]


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
    metrics = manager.get_metrics()
    assert metrics["frames_scheduled_by_camera"]["CAM_001"] == 3
    assert metrics["frames_scheduled_by_camera"]["CAM_002"] == 4
    assert metrics["frames_consumed_by_camera"]["CAM_001"] == 3
    assert metrics["frames_consumed_by_camera"]["CAM_002"] == 4


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
    assert manager.get_metrics()["per_camera_buffer_size"] == 2


def test_null_max_frames_per_camera_allows_full_video_processing(tmp_path: Path) -> None:
    video_a = _create_test_video(tmp_path / "a.mp4", frame_count=6)
    manager = MultiCameraIngestionManager(
        _build_config(
            [{"camera_id": "CAM_001", "source_type": "video", "source": str(video_a), "enabled": True}],
            max_frames_per_camera=None,
        ),
        _logger(),
    )

    manager.start()
    packets = _collect_all_packets(manager)
    manager.stop()

    assert len(packets) == 6
    assert manager.max_frames_per_camera is None


def test_invalid_and_negative_max_frames_per_camera_fail_clearly(tmp_path: Path) -> None:
    video_a = _create_test_video(tmp_path / "a.mp4", frame_count=2)
    for bad_value in ("all", "", -10, 0):
        with pytest.raises(ConfigurationError, match="positive integer or null"):
            MultiCameraIngestionManager(
                _build_config(
                    [{"camera_id": "CAM_001", "source_type": "video", "source": str(video_a), "enabled": True}],
                    max_frames_per_camera=bad_value,  # type: ignore[arg-type]
                ),
                _logger(),
            )


def test_same_camera_is_never_read_concurrently_with_fixed_worker_pool(tmp_path: Path) -> None:
    video_a = _create_test_video(tmp_path / "a.mp4", frame_count=5)
    manager = MultiCameraIngestionManager(
        _build_config(
            [{"camera_id": "CAM_001", "source_type": "video", "source": str(video_a), "enabled": True}],
            worker_count=3,
        ),
        _logger(),
    )
    fake_reader = _FakeReader("CAM_001", frame_count=5, delay_seconds=0.02)
    manager.readers_by_camera["CAM_001"] = fake_reader
    manager.camera_sources["CAM_001"] = fake_reader

    manager.start()
    packets = _collect_all_packets(manager)
    manager.stop()

    assert len(packets) == 5
    assert fake_reader.max_active_reads == 1


def test_round_robin_fairness_with_twelve_cameras_and_three_workers(tmp_path: Path) -> None:
    cameras = []
    for index in range(12):
        path = _create_test_video(tmp_path / f"fair_{index:02d}.mp4", frame_count=2)
        cameras.append({"camera_id": f"CAM_{index + 1:03d}", "source_type": "video", "source": str(path), "enabled": True})
    manager = MultiCameraIngestionManager(_build_config(cameras, worker_count=3), _logger())
    for camera in cameras:
        camera_id = str(camera["camera_id"])
        fake_reader = _FakeReader(camera_id, frame_count=2)
        manager.readers_by_camera[camera_id] = fake_reader
        manager.camera_sources[camera_id] = fake_reader

    manager.start()
    packets = _collect_all_packets(manager)
    manager.stop()
    metrics = manager.get_metrics()

    assert len(manager.worker_threads) == 3
    assert len(packets) == 24
    assert all(metrics["frames_by_camera"][str(camera["camera_id"])] == 2 for camera in cameras)
    assert all(metrics["frames_scheduled_by_camera"][str(camera["camera_id"])] == 2 for camera in cameras)
    assert metrics["max_consecutive_frames_same_camera"] == 1


def test_empty_slow_camera_does_not_block_ready_cameras(tmp_path: Path) -> None:
    cameras = []
    for camera_id in ("CAM_001", "CAM_002", "CAM_003"):
        path = _create_test_video(tmp_path / f"{camera_id}.mp4", frame_count=3)
        cameras.append({"camera_id": camera_id, "source_type": "video", "source": str(path), "enabled": True})
    manager = MultiCameraIngestionManager(_build_config(cameras, worker_count=2), _logger())
    manager.readers_by_camera["CAM_001"] = _FakeReader("CAM_001", frame_count=3)
    manager.readers_by_camera["CAM_002"] = _FakeReader("CAM_002", frame_count=3, delay_seconds=0.05)
    manager.readers_by_camera["CAM_003"] = _FakeReader("CAM_003", frame_count=3)
    manager.camera_sources = dict(manager.readers_by_camera)

    manager.start()
    packets = _collect_all_packets(manager)
    manager.stop()
    metrics = manager.get_metrics()

    assert len(packets) == 9
    assert metrics["scheduler_skipped_empty_camera"] > 0
    assert metrics["frames_scheduled_by_camera"]["CAM_001"] == 3
    assert metrics["frames_scheduled_by_camera"]["CAM_003"] == 3


def test_camera_exception_does_not_kill_worker_pool(tmp_path: Path) -> None:
    cameras = []
    for camera_id in ("CAM_001", "CAM_002", "CAM_003"):
        path = _create_test_video(tmp_path / f"{camera_id}.mp4", frame_count=3)
        cameras.append({"camera_id": camera_id, "source_type": "video", "source": str(path), "enabled": True})
    manager = MultiCameraIngestionManager(_build_config(cameras, worker_count=3, stop_on_camera_error=False), _logger())
    manager.readers_by_camera["CAM_001"] = _FakeReader("CAM_001", frame_count=3)
    manager.readers_by_camera["CAM_002"] = _FakeReader("CAM_002", frame_count=3, fail_on_read_number=1)
    manager.readers_by_camera["CAM_003"] = _FakeReader("CAM_003", frame_count=3)
    manager.camera_sources = dict(manager.readers_by_camera)

    manager.start()
    packets = _collect_all_packets(manager)
    manager.stop()
    metrics = manager.get_metrics()

    assert sum(1 for packet in packets if packet.camera_id == "CAM_001") == 3
    assert sum(1 for packet in packets if packet.camera_id == "CAM_003") == 3
    assert metrics["camera_read_failures"] == 1
    assert "CAM_002" in metrics["camera_errors"]
    assert manager.all_workers_stopped() is True
