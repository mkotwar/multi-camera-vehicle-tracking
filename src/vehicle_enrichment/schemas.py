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
    vehicle_class: str = "UNKNOWN"
    evidence_source: str = "existing_track_evidence"
    original_bbox_xyxy: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    expanded_crop_bbox_xyxy: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    source_frame_width: int = 0
    source_frame_height: int = 0
    context_padding_ratio: float = 0.0
    original_crop_width: int = 0
    original_crop_height: int = 0
    candidate_rank: int | None = None
    candidate_retained: bool = True
    candidate_rejection_reason: str | None = None
    frame_gap_from_previous_selected: int | None = None
    duplicate_score: float | None = None
    resolution_tier: str = "below_minimum"
    florence_eligible_for_body_type: bool = False
    florence_eligible_for_colour: bool = False
    florence_body_type_skip_reason: str | None = None
    florence_colour_skip_reason: str | None = None
    edge_truncated: bool = False
    ranking_score: float = 0.0
    selected_for_body_type: bool = False
    selected_for_colour: bool = False
    body_type_crop_result: str | None = None
    colour_crop_result: str | None = None
    readable_crop: bool = False
    colour_selection_tier: str | None = None
    trigger_x: float | None = None
    trigger_y: float | None = None
    zone_top: int | None = None
    zone_bottom: int | None = None
    class_minimum_width: int | None = None
    class_minimum_height: int | None = None
    florence_minimum_width: int | None = None
    florence_minimum_height: int | None = None
    evidence_eligible: bool | None = None
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
    original_crop_width: int | None = None
    original_crop_height: int | None = None
    resolution_tier: str | None = None
    square_padding_applied: bool | None = None
    padded_width: int | None = None
    padded_height: int | None = None
    florence_input_width: int | None = None
    florence_input_height: int | None = None
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
    vehicle_attribute_raw_responses: list[str] = field(default_factory=list)
    vehicle_attribute_selected_crop_paths: list[str] = field(default_factory=list)
    vehicle_attribute_inference_count: int = 0
    plate_detection_confidence: float | None = None
    plate_bbox: list[float] | None = None
    plate_crop_path: str | None = None
    plate_ocr_attempted: bool = False
    plate_ocr_raw_response: str | None = None
    plate_text_confidence: float | None = None
    plate_ocr_reason: str | None = None
    attribute_backend: str | None = None
    plate_ocr_backend: str | None = None
    plate_quality_status: str | None = None
    evidence_used: list[EnrichmentEvidenceItem] = field(default_factory=list)
    candidate_crop_count: int = 0
    eligible_crop_count: int = 0
    preferred_crop_count: int = 0
    readable_crop_count: int = 0
    fallback_crop_count: int = 0
    selected_colour_crop_count: int = 0
    colour_selection_tier: str | None = None
    selected_body_type_crop_paths: list[str] = field(default_factory=list)
    selected_colour_crop_paths: list[str] = field(default_factory=list)
    body_type_eligible: bool | None = None
    body_type_candidate_crop_count: int = 0
    body_type_selected_crop_count: int = 0
    body_type_florence_call_count: int = 0
    body_type_valid_prediction_count: int = 0
    body_type_failure_reason: str | None = None
    florence_mode: str | None = None
    adapter_loaded: bool | None = None
    selected_crop_paths: list[str] = field(default_factory=list)
    crop_level_captions: list[dict[str, Any]] = field(default_factory=list)
    crop_level_body_types: list[dict[str, Any]] = field(default_factory=list)
    crop_level_colours: list[dict[str, Any]] = field(default_factory=list)
    final_body_type_reason: str | None = None
    final_colour_reason: str | None = None
    caption_inference_count: int = 0
    comparison_payload: dict[str, Any] | None = None
    classification_trigger: str | None = None
    final_reason: str | None = None
    predictions: list[AttributePrediction] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    processing_started_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    processing_completed_at: str | None = None
    processing_duration_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return dataclass_to_dict(self)
