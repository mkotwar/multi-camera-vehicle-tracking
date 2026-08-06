from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2

from .models import FramePacket, SUPPORTED_SOURCE_TYPES, ConfigurationError, VideoOpenError


class VideoCameraReader:
    """Owns the capture state for one configured camera or video source."""

    def __init__(self, camera_id: str, source_type: str, source: str | int, *, target_read_fps: float | None = None) -> None:
        self.camera_id = str(camera_id).strip()
        self.source_type = str(source_type).strip().lower()
        if self.source_type not in SUPPORTED_SOURCE_TYPES:
            raise ConfigurationError(f"Unsupported source type: {source_type}")
        self.source = source
        self.target_read_fps = None if target_read_fps is None else float(target_read_fps)
        if self.target_read_fps is not None and self.target_read_fps <= 0.0:
            raise ConfigurationError("target_read_fps must be positive when provided.")
        self._capture: cv2.VideoCapture | None = None
        self._source_fps: float = 0.0
        self._frame_width = 0
        self._frame_height = 0
        self._released = False
        self._frame_number = 0
        self._ended = False
        self._opened_monotonic: float | None = None
        self._resolved_video_path: Path | None = None
        self._last_read_monotonic: float | None = None

    @property
    def source_fps(self) -> float:
        return self._source_fps

    @property
    def frame_width(self) -> int:
        return self._frame_width

    @property
    def frame_height(self) -> int:
        return self._frame_height

    @property
    def released(self) -> bool:
        return self._released

    @property
    def ended(self) -> bool:
        return self._ended

    @property
    def source_display(self) -> str:
        if self.source_type == "video" and self._resolved_video_path is not None:
            return str(self._resolved_video_path)
        return str(self.source)

    def open(self) -> None:
        if self._capture is not None:
            return
        capture_target = self._resolve_capture_target()
        capture = cv2.VideoCapture(capture_target)
        if not capture.isOpened():
            capture.release()
            raise VideoOpenError(f"Source could not be opened for camera '{self.camera_id}': {self.source_display}")
        source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        if source_fps <= 0.0:
            source_fps = 30.0
        self._frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        self._frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        self._capture = capture
        self._source_fps = source_fps
        self._released = False
        self._ended = False
        self._frame_number = 0
        self._opened_monotonic = time.monotonic()
        self._last_read_monotonic = None

    def read_next_frame(self, *, worker_id: int) -> FramePacket | None:
        self.open()
        assert self._capture is not None
        if self._ended:
            return None
        self._throttle_if_needed()
        ok, frame = self._capture.read()
        if not ok:
            self._ended = True
            return None
        frame_number = self._frame_number
        self._frame_number += 1
        self._last_read_monotonic = time.monotonic()
        captured_at = datetime.now(timezone.utc).isoformat()
        if self.source_type == "video":
            timestamp_seconds = frame_number / self._source_fps
        else:
            timestamp_seconds = max(0.0, time.monotonic() - (self._opened_monotonic or time.monotonic()))
        return FramePacket(
            camera_id=self.camera_id,
            frame_number=frame_number,
            timestamp_seconds=timestamp_seconds,
            source_fps=self._source_fps,
            frame=frame,
            source_frame_width=int(frame.shape[1]),
            source_frame_height=int(frame.shape[0]),
            worker_id=int(worker_id),
            captured_at=captured_at,
            source_type=self.source_type,
        )

    def close(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None
        self._released = True

    def _resolve_capture_target(self) -> str | int:
        if self.source_type == "video":
            video_path = Path(str(self.source)).expanduser().resolve()
            if not video_path.exists():
                raise VideoOpenError(f"Video path does not exist for camera '{self.camera_id}': {video_path}")
            self._resolved_video_path = video_path
            return str(video_path)
        if self.source_type == "webcam":
            try:
                return int(self.source)
            except Exception as exc:
                raise ConfigurationError(f"Webcam source must be an integer for camera '{self.camera_id}'.") from exc
        return str(self.source)

    def _throttle_if_needed(self) -> None:
        if self.target_read_fps is None:
            return
        if self._last_read_monotonic is None:
            return
        minimum_interval_seconds = 1.0 / self.target_read_fps
        elapsed_seconds = time.monotonic() - self._last_read_monotonic
        remaining_seconds = minimum_interval_seconds - elapsed_seconds
        if remaining_seconds > 0.0:
            time.sleep(remaining_seconds)

    def __enter__(self) -> "VideoCameraReader":
        self.open()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()
