from __future__ import annotations

from typing import Any

from ..schemas import ATTRIBUTE_STATUS_DISABLED, PlateQualityResult


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
        return PlateQualityResult(
            acceptable=False,
            predictions=[],
            status="skipped",
            source="plate.quality_validator",
            reason="plate_quality_not_implemented_for_runtime",
        )
