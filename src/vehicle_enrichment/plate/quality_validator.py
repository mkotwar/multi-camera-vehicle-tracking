from __future__ import annotations

from typing import Any

from ..schemas import ATTRIBUTE_STATUS_DISABLED, AttributePrediction, PlateQualityResult


class PlateQualityValidator:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = dict(config)
        detector_config = dict(self.config.get("detector", {}) or {})
        self.enabled = bool(detector_config.get("enabled", self.config.get("detection_enabled", False)))
        self.minimum_crop_width = int(detector_config.get("minimum_crop_width", 40))
        self.minimum_crop_height = int(detector_config.get("minimum_crop_height", 16))
        self.minimum_detector_confidence = float(detector_config.get("confidence_threshold", 0.5))

    def validate(self, detection_payload: dict[str, Any] | None) -> PlateQualityResult:
        if not self.enabled:
            return PlateQualityResult(
                acceptable=None,
                predictions=[],
                status=ATTRIBUTE_STATUS_DISABLED,
                source="plate.quality_validator",
                reason="plate_quality_disabled",
            )
        if not detection_payload:
            return PlateQualityResult(
                acceptable=False,
                predictions=[],
                status="skipped",
                source="plate.quality_validator",
                reason="no_plate_detection",
            )
        confidence = float(detection_payload.get("confidence", 0.0) or 0.0)
        width = int(detection_payload.get("plate_crop_width", 0) or 0)
        height = int(detection_payload.get("plate_crop_height", 0) or 0)
        if confidence < self.minimum_detector_confidence:
            reason = "plate_confidence_below_threshold"
            acceptable = False
        elif width < self.minimum_crop_width or height < self.minimum_crop_height:
            reason = "plate_crop_below_minimum_size"
            acceptable = False
        else:
            reason = "plate_quality_accepted"
            acceptable = True
        return PlateQualityResult(
            acceptable=acceptable,
            predictions=[
                AttributePrediction(
                    attribute_name="plate_quality",
                    label="ACCEPTABLE" if acceptable else "REJECTED",
                    source_backend="deterministic_plate_quality",
                    source_model=None,
                    source_frame_number=None,
                    source_crop_path=str(detection_payload.get("plate_crop_path") or ""),
                    raw_response=dict(detection_payload),
                    confidence=confidence,
                    quality_weight=confidence,
                    status="completed",
                    reason=reason,
                )
            ],
            status="completed",
            source="plate.quality_validator",
            reason=reason,
        )
