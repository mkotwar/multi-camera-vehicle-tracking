from __future__ import annotations

from dataclasses import dataclass

import numpy as np


RUN_STATUS_CREATED = "CREATED"
RUN_STATUS_RUNNING = "RUNNING"
RUN_STATUS_COMPLETED = "COMPLETED"
RUN_STATUS_FAILED = "FAILED"
ALLOWED_RUN_STATUSES = {
    RUN_STATUS_CREATED,
    RUN_STATUS_RUNNING,
    RUN_STATUS_COMPLETED,
    RUN_STATUS_FAILED,
}
SUPPORTED_SOURCE_TYPES = {"video", "rtsp", "webcam"}


class ConfigurationError(Exception):
    pass


class VideoOpenError(Exception):
    pass


class PipelineRuntimeError(Exception):
    pass


@dataclass(slots=True)
class FramePacket:
    camera_id: str
    frame_number: int
    timestamp_seconds: float
    source_fps: float
    frame: np.ndarray
    worker_id: int
    captured_at: str
    source_type: str


@dataclass(slots=True, frozen=True)
class Detection:
    bbox_xyxy: tuple[float, float, float, float]
    confidence: float
    class_id: int
    class_name: str


@dataclass(slots=True, frozen=True)
class BBoxQualityDiagnostic:
    camera_id: str
    frame_number: int
    timestamp_seconds: float
    class_name: str
    normalized_class_name: str
    confidence: float
    bbox_xyxy: tuple[float, float, float, float]
    bbox_width: float
    bbox_height: float
    bbox_area: float
    frame_width: int
    frame_height: int
    width_ratio: float
    height_ratio: float
    area_ratio: float
    aspect_ratio: float
    touches_edge: bool
    touches_left_edge: bool
    touches_right_edge: bool
    touches_top_edge: bool
    touches_bottom_edge: bool
    accepted_by_bbox_quality: bool
    rejection_reason: str | None


@dataclass(slots=True, frozen=True)
class TrackedDetection:
    camera_id: str
    frame_number: int
    timestamp_seconds: float
    tracker_id: int
    bbox_xyxy: tuple[float, float, float, float]
    confidence: float
    raw_class_id: int
    raw_class_name: str


@dataclass(slots=True)
class RunMetadata:
    run_id: str
    project_name: str
    started_at: str
    completed_at: str | None
    status: str
    camera_count: int
    processed_frames: int
    completed_tracks: int
    error_count: int
    config_path: str

    def __post_init__(self) -> None:
        if self.status not in ALLOWED_RUN_STATUSES:
            raise ValueError(f"Unsupported run status: {self.status}")
