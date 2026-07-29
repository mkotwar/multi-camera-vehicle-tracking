from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from src.camera_reader import VideoCameraReader
from src.models import FramePacket, VideoOpenError


def _create_test_video(path: Path, *, fps: float = 10.0, frame_count: int = 4, width: int = 32, height: int = 24) -> Path:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    for index in range(frame_count):
        frame = np.full((height, width, 3), index * 40, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return path


def test_invalid_video_path_raises_video_open_error(tmp_path: Path) -> None:
    reader = VideoCameraReader("CAM_001", "video", tmp_path / "missing.mp4")
    with pytest.raises(VideoOpenError):
        reader.open()


def test_reader_frame_number_starts_at_zero_and_timestamp_uses_fps(tmp_path: Path) -> None:
    video_path = _create_test_video(tmp_path / "sample.mp4", fps=8.0, frame_count=3)
    reader = VideoCameraReader("CAM_001", "video", video_path)
    with reader:
        frames = []
        while True:
            packet = reader.read_next_frame(worker_id=0)
            if packet is None:
                break
            frames.append(packet)
    assert frames[0].frame_number == 0
    assert frames[0].timestamp_seconds == pytest.approx(0.0)
    assert frames[1].timestamp_seconds == pytest.approx(1.0 / 8.0)


def test_reader_yields_frame_packet_and_non_empty_frame(tmp_path: Path) -> None:
    video_path = _create_test_video(tmp_path / "sample.mp4")
    reader = VideoCameraReader("CAM_001", "video", video_path)
    with reader:
        packet = reader.read_next_frame(worker_id=3)
    assert isinstance(packet, FramePacket)
    assert packet is not None
    assert packet.camera_id == "CAM_001"
    assert packet.worker_id == 3
    assert packet.source_type == "video"
    assert packet.frame.size > 0


def test_reader_releases_video(tmp_path: Path) -> None:
    video_path = _create_test_video(tmp_path / "sample.mp4")
    reader = VideoCameraReader("CAM_001", "video", video_path)
    with reader:
        while reader.read_next_frame(worker_id=0) is not None:
            pass
    assert reader.released is True
