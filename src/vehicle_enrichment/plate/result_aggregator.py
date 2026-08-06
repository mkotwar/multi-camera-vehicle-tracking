from __future__ import annotations

from typing import Any


class PlateResultAggregator:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = dict(config)

    def aggregate(self, detection_result: Any, quality_result: Any, ocr_result: Any) -> dict[str, Any]:
        return {
            "status": getattr(ocr_result, "status", "disabled"),
            "reason": getattr(ocr_result, "reason", None) or getattr(quality_result, "reason", None) or getattr(detection_result, "reason", None),
            "plate_detected": bool(getattr(detection_result, "detected", False)),
            "plate_text": getattr(ocr_result, "text", None),
        }
