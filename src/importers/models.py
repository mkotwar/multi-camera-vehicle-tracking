from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal


Severity = Literal["ERROR", "WARNING", "INFO"]


@dataclass(frozen=True, slots=True)
class LogicalTrackRef:
    run_key: str
    camera_key: str
    local_track_id: str

    @property
    def key(self) -> str:
        return f"{self.run_key}|{self.camera_key}|{self.local_track_id}"


@dataclass(slots=True)
class ValidationIssue:
    severity: Severity
    code: str
    message: str
    context: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ProcessingRunRow:
    run_key: str
    status: str | None
    project_name: str | None
    started_at: str | None
    completed_at: str | None
    output_directory: str | None
    config_path: str | None
    config_snapshot: dict[str, Any]
    summary: dict[str, Any]
    detection_backend: str | None
    tracking_backend: str | None
    enrichment_enabled: bool | None
    processed_frames: int | None
    raw_yolo_detections: int | None
    roi_filtered_detections: int | None
    completed_tracks: int | None
    discarded_tracks: int | None
    error_count: int | None
    metrics: dict[str, Any]
    metadata: dict[str, Any]


@dataclass(slots=True)
class RunCameraRow:
    run_key: str
    camera_key: str
    source: str | None
    source_type: str | None
    enabled: bool | None
    fps: float | None
    width: int | None
    height: int | None
    total_frames: int | None
    frames_processed: int | None
    detections_count: int | None
    metadata: dict[str, Any]


@dataclass(slots=True)
class VehicleTrackRow:
    ref: LogicalTrackRef
    tracker_namespace: str | None
    native_tracker_id: str | None
    track_status: str | None
    searchable_by_default: bool
    completion_reason: str | None
    first_frame: int | None
    last_frame: int | None
    first_seen_seconds: float | None
    last_seen_seconds: float | None
    observation_count: int | None
    lost_frames: int | None
    vehicle_class: str | None
    final_class_reason: str | None
    vehicle_class_confidence: float | None
    vehicle_colour: str | None
    vehicle_colour_status: str | None
    body_type: str | None
    body_type_status: str | None
    plate_text: str | None
    plate_detected: bool | None
    plate_colour: str | None
    registration_category: str | None
    class_counts: dict[str, Any]
    class_confidence_sums: dict[str, Any]
    evidence_record_count: int | None
    raw_track: dict[str, Any]
    enrichment_summary: dict[str, Any]


@dataclass(slots=True)
class TrackObservationRow:
    ref: LogicalTrackRef
    tracker_namespace: str | None
    native_tracker_id: str | None
    frame_number: int | None
    timestamp_seconds: float | None
    bbox_x1: float | None
    bbox_y1: float | None
    bbox_x2: float | None
    bbox_y2: float | None
    detection_confidence: float | None
    raw_class_id: int | None
    raw_class_name: str | None
    metadata: dict[str, Any]


@dataclass(slots=True)
class TrackEvidenceRow:
    ref: LogicalTrackRef
    evidence_role: str | None
    frame_number: int | None
    timestamp_seconds: float | None
    bbox_xyxy: list[Any] | None
    original_bbox_xyxy: list[Any] | None
    expanded_crop_bbox_xyxy: list[Any] | None
    detection_confidence: float | None
    quality_score: float | None
    sharpness_score: float | None
    centeredness_score: float | None
    edge_visibility_score: float | None
    brightness_score: float | None
    crop_width: int | None
    crop_height: int | None
    resolution_tier: str | None
    selected_for_colour: bool | None
    selected_for_body_type: bool | None
    evidence_source: str | None
    candidate_rank: int | None
    crop_relative_path: str | None
    source_frame_relative_path: str | None
    annotated_frame_relative_path: str | None
    metadata: dict[str, Any]


@dataclass(slots=True)
class MediaAssetRow:
    run_key: str
    camera_key: str | None
    track_local_id: str | None
    media_type: str
    relative_path: str | None
    original_path: str | None
    frame_number: int | None
    timestamp_seconds: float | None
    width: int | None
    height: int | None
    exists: bool
    invalid_path: bool
    outside_run_directory: bool
    metadata: dict[str, Any]


@dataclass(slots=True)
class ColourPredictionRow:
    ref: LogicalTrackRef
    evidence_relative_path: str | None
    predicted_colour: str | None
    normalized_colour: str | None
    confidence: float | None
    status: str | None
    source_model: str | None
    source_backend: str | None
    prompt: str | None
    raw_response: str | None
    inference_duration_ms: float | None
    evidence_frame_number: int | None
    evidence_timestamp_seconds: float | None
    metadata: dict[str, Any]


@dataclass(slots=True)
class VehicleAttributePredictionRow:
    ref: LogicalTrackRef
    attribute_type: str
    attribute_value: str | None
    status: str | None
    confidence: float | None
    source_backend: str | None
    source_model: str | None
    raw_response: str | None
    evidence_relative_path: str | None
    evidence_frame_number: int | None
    evidence_timestamp_seconds: float | None
    metadata: dict[str, Any]


@dataclass(slots=True)
class PlateDetectionRow:
    ref: LogicalTrackRef
    plate_bbox: list[Any] | None
    confidence: float | None
    crop_relative_path: str | None
    frame_number: int | None
    timestamp_seconds: float | None
    source_model: str | None
    quality_status: str | None
    metadata: dict[str, Any]


@dataclass(slots=True)
class PlateReadingRow:
    ref: LogicalTrackRef
    plate_text: str | None
    confidence: float | None
    plate_colour: str | None
    registration_category: str | None
    ocr_backend: str | None
    raw_response: str | None
    reason: str | None
    is_selected: bool
    metadata: dict[str, Any]


@dataclass(slots=True)
class PhysicalVehicleRow:
    run_key: str
    vehicle_key: str
    vehicle_class: str | None
    vehicle_colour: str | None
    first_timestamp_seconds: float | None
    last_timestamp_seconds: float | None
    identity_confidence: float | None
    identity_method: str | None
    identity_status: str | None
    consensus_plate_text: str | None
    plate_confidence: float | None
    metadata: dict[str, Any]


@dataclass(slots=True)
class PhysicalVehicleTrackRow:
    run_key: str
    vehicle_key: str
    ref: LogicalTrackRef
    association_score: float | None
    association_method: str | None
    association_reason: str | None
    metadata: dict[str, Any]


@dataclass(slots=True)
class IdentityDecisionRow:
    run_key: str
    source_ref: LogicalTrackRef | None
    target_ref: LogicalTrackRef | None
    decision: str
    final_score: float | None
    plate_score: float | None
    spatial_score: float | None
    temporal_score: float | None
    motion_score: float | None
    appearance_score: float | None
    colour_score: float | None
    reason: str | None
    metadata: dict[str, Any]


@dataclass(slots=True)
class DryRunRows:
    processing_runs: list[ProcessingRunRow] = field(default_factory=list)
    run_cameras: list[RunCameraRow] = field(default_factory=list)
    vehicle_tracks: list[VehicleTrackRow] = field(default_factory=list)
    track_observations: list[TrackObservationRow] = field(default_factory=list)
    track_evidence: list[TrackEvidenceRow] = field(default_factory=list)
    media_assets: list[MediaAssetRow] = field(default_factory=list)
    colour_predictions: list[ColourPredictionRow] = field(default_factory=list)
    vehicle_attribute_predictions: list[VehicleAttributePredictionRow] = field(default_factory=list)
    plate_detections: list[PlateDetectionRow] = field(default_factory=list)
    plate_readings: list[PlateReadingRow] = field(default_factory=list)
    physical_vehicles: list[PhysicalVehicleRow] = field(default_factory=list)
    physical_vehicle_tracks: list[PhysicalVehicleTrackRow] = field(default_factory=list)
    identity_decisions: list[IdentityDecisionRow] = field(default_factory=list)


@dataclass(slots=True)
class DryRunReport:
    run_dir: Path
    run_key: str
    rows: DryRunRows
    issues: list[ValidationIssue]
    counts: dict[str, Any]
    field_mapping: dict[str, Any]
    media_checks: dict[str, int]
    normalizations: list[dict[str, Any]]
    verdict: str
