from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime
from typing import Any


VEHICLE_BODY_TYPE_UNKNOWN = "UNKNOWN"
VEHICLE_COLOUR_UNKNOWN = "UNKNOWN"
ATTRIBUTE_STATUS_DISABLED = "disabled"
ATTRIBUTE_STATUS_NOT_RUN = "not_run"
ATTRIBUTE_STATUS_READY = "ready"
ATTRIBUTE_STATUS_ERROR = "error"
ENRICHMENT_STATUS_DISABLED = "disabled"
ENRICHMENT_STATUS_NO_EVIDENCE = "no_evidence"
ENRICHMENT_STATUS_EVIDENCE_READY = "evidence_ready"
ENRICHMENT_STATUS_COMPLETED = "completed"
ENRICHMENT_STATUS_ERROR = "error"


def dataclass_to_dict(value: Any) -> Any:
    if is_dataclass(value):
        return {key: dataclass_to_dict(item) for key, item in asdict(value).items()}
    if isinstance(value, list):
        return [dataclass_to_dict(item) for item in value]
    if isinstance(value, dict):
        return {str(key): dataclass_to_dict(item) for key, item in value.items()}
    return value


@dataclass(slots=True)
class EnrichmentEvidenceItem:
    local_track_id: str
    camera_id: str
    native_tracker_id: int
    frame_number: int
    timestamp_seconds: float
    source_image_path: str | None
    vehicle_crop_path: str | None
    annotated_frame_path: str | None
    bbox_xyxy: tuple[float, float, float, float]
    evidence_role: str
    detection_confidence: float
    crop_width: int
    crop_height: int
    crop_area: int
    sharpness_score: float
    brightness_score: float
    border_penalty: float
    clipping_ratio: float
    quality_score: float
    rejection_reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return dataclass_to_dict(self)


@dataclass(slots=True)
class TrackEnrichmentRequest:
    local_track_id: str
    camera_id: str
    native_tracker_id: int
    vehicle_class: str
    vehicle_class_confidence: float | None
    track_status: str
    completion_reason: str | None
    started_at_seconds: float
    ended_at_seconds: float
    evidence_items: list[EnrichmentEvidenceItem] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return dataclass_to_dict(self)


@dataclass(slots=True)
class AttributePrediction:
    attribute_name: str
    label: str | None
    source_backend: str | None
    source_model: str | None
    source_frame_number: int | None
    source_crop_path: str | None
    raw_response: Any | None
    confidence: float | None
    quality_weight: float | None
    evidence_role: str | None = None
    adapter_active: bool | None = None
    inference_duration_ms: float | None = None
    status: str = ATTRIBUTE_STATUS_NOT_RUN
    reason: str | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return dataclass_to_dict(self)


@dataclass(slots=True)
class VehicleBodyTypeResult:
    label: str
    predictions: list[AttributePrediction] = field(default_factory=list)
    status: str = ATTRIBUTE_STATUS_DISABLED
    source: str | None = None
    reason: str | None = None
    model: str | None = None
    adapter_active: bool | None = None
    aggregation_reason: str | None = None
    agreement_score: float | None = None
    accumulated_quality_weight: float | None = None
    task_prompt: str | None = None
    prompt_text: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return dataclass_to_dict(self)


@dataclass(slots=True)
class VehicleColourResult:
    label: str
    predictions: list[AttributePrediction] = field(default_factory=list)
    status: str = ATTRIBUTE_STATUS_DISABLED
    source: str | None = None
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return dataclass_to_dict(self)


@dataclass(slots=True)
class VehicleMakeModelResult:
    make: str | None
    model: str | None
    predictions: list[AttributePrediction] = field(default_factory=list)
    status: str = ATTRIBUTE_STATUS_DISABLED
    source: str | None = None
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return dataclass_to_dict(self)


@dataclass(slots=True)
class PlateDetectionResult:
    detected: bool
    predictions: list[AttributePrediction] = field(default_factory=list)
    status: str = ATTRIBUTE_STATUS_DISABLED
    source: str | None = None
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return dataclass_to_dict(self)


@dataclass(slots=True)
class PlateQualityResult:
    acceptable: bool | None
    predictions: list[AttributePrediction] = field(default_factory=list)
    status: str = ATTRIBUTE_STATUS_DISABLED
    source: str | None = None
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return dataclass_to_dict(self)


@dataclass(slots=True)
class PlateColourResult:
    label: str | None
    predictions: list[AttributePrediction] = field(default_factory=list)
    status: str = ATTRIBUTE_STATUS_DISABLED
    source: str | None = None
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return dataclass_to_dict(self)


@dataclass(slots=True)
class PlateOCRResult:
    text: str | None
    predictions: list[AttributePrediction] = field(default_factory=list)
    status: str = ATTRIBUTE_STATUS_DISABLED
    source: str | None = None
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return dataclass_to_dict(self)


@dataclass(slots=True)
class TrackEnrichmentResult:
    local_track_id: str
    camera_id: str
    vehicle_class: str
    vehicle_class_confidence: float | None
    vehicle_body_type: VehicleBodyTypeResult
    vehicle_colour: VehicleColourResult
    vehicle_make: str | None
    vehicle_model: str | None
    plate_detected: bool
    plate_colour: str | None
    registration_category: str | None
    plate_text: str | None
    status: str
    evidence_used: list[EnrichmentEvidenceItem] = field(default_factory=list)
    predictions: list[AttributePrediction] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    processing_started_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    processing_completed_at: str | None = None
    processing_duration_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return dataclass_to_dict(self)
