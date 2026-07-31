from __future__ import annotations

from src.models import LocalTrack
from src.pipeline import _serialize_track
from src.vehicle_enrichment.schemas import (
    AttributePrediction,
    EnrichmentEvidenceItem,
    TrackEnrichmentResult,
    VehicleBodyTypeResult,
    VehicleColourResult,
    dataclass_to_dict,
)


def _track() -> LocalTrack:
    return LocalTrack(
        local_track_id="CAM_001:TRACK_7",
        camera_id="CAM_001",
        tracker_namespace="camera",
        native_tracker_id=7,
        status="COMPLETED",
        first_frame=10,
        last_frame=20,
        first_timestamp_seconds=1.0,
        last_timestamp_seconds=2.0,
        observation_count=3,
        lost_frames=0,
        final_class="car",
        final_class_reason="WEIGHTED_MAJORITY",
        class_counts={"car": 3},
        class_confidence_sums={"car": 2.7},
        observations=[],
        completion_reason="END_OF_STREAM",
    )


def test_schema_serialization_preserves_optional_fields() -> None:
    result = TrackEnrichmentResult(
        local_track_id="CAM_001:TRACK_7",
        camera_id="CAM_001",
        vehicle_class="CAR",
        vehicle_class_confidence=0.9,
        vehicle_body_type=VehicleBodyTypeResult(label="UNKNOWN", status="disabled", source="test", reason="disabled"),
        vehicle_colour=VehicleColourResult(label="UNKNOWN", status="disabled", source="test", reason="disabled"),
        vehicle_make=None,
        vehicle_model=None,
        plate_detected=False,
        plate_colour=None,
        registration_category=None,
        plate_text=None,
        status="evidence_ready",
        evidence_used=[
            EnrichmentEvidenceItem(
                local_track_id="CAM_001:TRACK_7",
                camera_id="CAM_001",
                native_tracker_id=7,
                frame_number=45,
                timestamp_seconds=4.5,
                source_image_path="source.jpg",
                vehicle_crop_path="crop.jpg",
                annotated_frame_path="annotated.jpg",
                bbox_xyxy=(1.0, 2.0, 30.0, 40.0),
                evidence_role="BEST_OVERALL",
                detection_confidence=0.89,
                crop_width=29,
                crop_height=38,
                crop_area=1102,
                sharpness_score=15.0,
                brightness_score=90.0,
                border_penalty=0.1,
                clipping_ratio=0.0,
                quality_score=0.86,
                rejection_reasons=[],
            )
        ],
        predictions=[
            AttributePrediction(
                attribute_name="vehicle_body_type",
                label=None,
                source_backend=None,
                source_model=None,
                source_frame_number=None,
                source_crop_path=None,
                raw_response=None,
                confidence=None,
                quality_weight=None,
                evidence_role=None,
                adapter_active=None,
                inference_duration_ms=None,
                status="disabled",
                reason="not enabled",
                error=None,
            )
        ],
        errors=[],
        processing_started_at="2026-07-31T00:00:00+00:00",
        processing_completed_at="2026-07-31T00:00:01+00:00",
        processing_duration_ms=12.5,
    )

    payload = dataclass_to_dict(result)

    assert payload["vehicle_make"] is None
    assert payload["plate_text"] is None
    assert payload["vehicle_body_type"]["label"] == "UNKNOWN"
    assert payload["vehicle_body_type"]["model"] is None
    assert payload["evidence_used"][0]["evidence_role"] == "BEST_OVERALL"
    assert payload["predictions"][0]["confidence"] is None


def test_tracks_json_extension_is_backward_compatible() -> None:
    track = _track()
    baseline = _serialize_track(track, {"CAM_001:TRACK_7": {"evidence_record_count": 1, "evidence_roles": ["FIRST"], "evidence_directory": "evidence"}})
    extended = _serialize_track(
        track,
        {"CAM_001:TRACK_7": {"evidence_record_count": 1, "evidence_roles": ["FIRST"], "evidence_directory": "evidence"}},
        {"CAM_001:TRACK_7": {"local_track_id": "CAM_001:TRACK_7", "status": "disabled"}},
    )

    assert baseline["local_track_id"] == extended["local_track_id"]
    assert baseline["camera_id"] == extended["camera_id"]
    assert baseline["observation_count"] == extended["observation_count"]
    assert "vehicle_enrichment" not in baseline
    assert extended["vehicle_enrichment"]["status"] == "disabled"
