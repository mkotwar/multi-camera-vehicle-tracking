from __future__ import annotations

from src.vehicle_enrichment.plate.result_aggregator import PlateResultAggregator
from src.vehicle_enrichment.schemas import AttributePrediction, PlateDetectionResult, PlateOCRResult, PlateQualityResult


def test_plate_result_aggregator_does_not_promote_invalid_ocr_text() -> None:
    aggregator = PlateResultAggregator({})
    detection = PlateDetectionResult(
        detected=True,
        status="completed",
        predictions=[
            AttributePrediction(
                attribute_name="plate_detection",
                label="PLATE",
                source_backend="yolo",
                source_model="plate.pt",
                source_frame_number=1,
                source_crop_path="plate.jpg",
                raw_response={"confidence": 0.91, "bbox_xyxy": [1, 2, 3, 4], "plate_crop_path": "plate.jpg"},
                confidence=0.91,
                quality_weight=0.91,
                status="completed",
            )
        ],
    )
    quality = PlateQualityResult(acceptable=True, status="completed", reason="quality_ok")
    ocr = PlateOCRResult(
        text=None,
        raw_text="LIGAJ7519",
        normalized_text="LIGAJ7519",
        validation_status="invalid",
        validation_reason="unsupported_indian_registration_structure",
        status="completed",
        source="plate.ocr_engine",
        reason="plate_validation_failed:unsupported_indian_registration_structure",
        predictions=[],
    )

    aggregated = aggregator.aggregate(detection, quality, ocr)

    assert aggregated["plate_detected"] is True
    assert aggregated["plate_readable"] is False
    assert aggregated["plate_text"] is None
    assert aggregated["plate_raw_text"] == "LIGAJ7519"


def test_plate_result_aggregator_prefers_valid_ocr_payload() -> None:
    aggregator = PlateResultAggregator({})
    detection = PlateDetectionResult(detected=True, status="completed", predictions=[])
    quality = PlateQualityResult(acceptable=True, status="completed", reason="quality_ok")
    ocr = PlateOCRResult(
        text="DL6CQ1126",
        raw_text="DL6CQI126",
        normalized_text="DL6CQI126",
        format_type="standard_private",
        validation_status="valid",
        validation_reason="validated_standard_state_registration",
        correction_applied=True,
        correction_count=1,
        status="completed",
        source="plate.ocr_engine",
        reason="validated_standard_state_registration",
        predictions=[],
    )

    aggregated = aggregator.aggregate(detection, quality, ocr)

    assert aggregated["plate_detected"] is True
    assert aggregated["plate_readable"] is True
    assert aggregated["plate_text"] == "DL6CQ1126"
    assert aggregated["plate_correction_applied"] is True
