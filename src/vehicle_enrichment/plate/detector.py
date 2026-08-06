from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2

from ..schemas import ATTRIBUTE_STATUS_DISABLED, ATTRIBUTE_STATUS_ERROR, AttributePrediction, PlateDetectionResult


class PlateDetector:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = dict(config)
        detector_config = dict(self.config.get("detector", {}) or {})
        self.enabled = bool(detector_config.get("enabled", self.config.get("detection_enabled", False)))
        self.model_path = str(detector_config.get("model_path", "") or "")
        self.confidence_threshold = float(detector_config.get("confidence_threshold", 0.5))
        self.minimum_crop_width = int(detector_config.get("minimum_crop_width", 40))
        self.minimum_crop_height = int(detector_config.get("minimum_crop_height", 16))
        self._disabled_reason = "plate_detector_disabled"
        if self.enabled and not self.model_path:
            self.enabled = False
            self._disabled_reason = "plate_detector_model_path_missing"

    def detect(self, evidence_item: Any) -> PlateDetectionResult:
        if not self.enabled:
            return PlateDetectionResult(
                detected=False,
                predictions=[],
                status=ATTRIBUTE_STATUS_DISABLED,
                source="plate.detector",
                reason=self._disabled_reason,
            )
        crop_path = Path(str(getattr(evidence_item, "vehicle_crop_path", "") or ""))
        image = cv2.imread(str(crop_path))
        if image is None or image.size == 0:
            return PlateDetectionResult(
                detected=False,
                predictions=[],
                status=ATTRIBUTE_STATUS_ERROR,
                source="plate.detector",
                reason="vehicle_crop_unreadable",
            )
        return PlateDetectionResult(
            detected=False,
            predictions=[],
            status="skipped",
            source="plate.detector",
            reason="plate_detector_not_implemented_for_runtime",
        )
