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
TRACK_STATUS_TENTATIVE = "TENTATIVE"
TRACK_STATUS_ACTIVE = "ACTIVE"
TRACK_STATUS_LOST = "LOST"
TRACK_STATUS_COMPLETED = "COMPLETED"
TRACK_STATUS_DISCARDED = "DISCARDED"
ALLOWED_TRACK_STATUSES = {
    TRACK_STATUS_TENTATIVE,
    TRACK_STATUS_ACTIVE,
    TRACK_STATUS_LOST,
    TRACK_STATUS_COMPLETED,
    TRACK_STATUS_DISCARDED,
}


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
    tracker_namespace: str
    frame_number: int
    timestamp_seconds: float
    tracker_id: int
    bbox_xyxy: tuple[float, float, float, float]
    confidence: float
    raw_class_id: int
    raw_class_name: str


@dataclass(slots=True, frozen=True)
class TrackObservation:
    camera_id: str
    tracker_namespace: str
    native_tracker_id: int
    local_track_id: str
    frame_number: int
    timestamp_seconds: float
    bbox_xyxy: tuple[float, float, float, float]
    confidence: float
    raw_class_id: int
    raw_class_name: str


@dataclass(slots=True)
class LocalTrack:
    local_track_id: str
    camera_id: str
    tracker_namespace: str
    native_tracker_id: int
    status: str
    first_frame: int
    last_frame: int
    first_timestamp_seconds: float
    last_timestamp_seconds: float
    observation_count: int
    lost_frames: int
    final_class: str | None
    final_class_reason: str | None
    class_counts: dict[str, int]
    class_confidence_sums: dict[str, float]
    observations: list[TrackObservation]
    completion_reason: str | None

    def __post_init__(self) -> None:
        if self.status not in ALLOWED_TRACK_STATUSES:
            raise ValueError(f"Unsupported track status: {self.status}")


@dataclass(slots=True)
class EvidenceCandidate:
    local_track_id: str
    camera_id: str
    native_tracker_id: int
    tracker_namespace: str
    frame_number: int
    timestamp_seconds: float
    bbox_xyxy: tuple[float, float, float, float]
    confidence: float
    raw_class_name: str
    final_class: str
    role: str
    bbox_area: float
    sharpness_score: float
    centeredness_score: float
    edge_visibility_score: float
    best_overall_score: float


@dataclass(slots=True)
class TrackEvidence:
    local_track_id: str
    camera_id: str
    native_tracker_id: int
    tracker_namespace: str
    role: str
    frame_number: int
    timestamp_seconds: float
    raw_class_name: str
    final_class: str
    confidence: float
    crop_path: str | None
    annotated_frame_path: str | None
    bbox_xyxy: tuple[float, float, float, float]
    sharpness_score: float
    best_overall_score: float


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
    configured_device: str | None = None
    resolved_device: str | None = None
    cuda_available: bool | None = None
    cuda_device_count: int | None = None
    cuda_device_name: str | None = None
    torch_version: str | None = None
    torch_cuda_version: str | None = None

    def __post_init__(self) -> None:
        if self.status not in ALLOWED_RUN_STATUSES:
            raise ValueError(f"Unsupported run status: {self.status}")
