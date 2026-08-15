from __future__ import annotations

from typing import Any


class PlateResultAggregator:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = dict(config)

    def aggregate(self, detection_result: Any, quality_result: Any, ocr_result: Any) -> dict[str, Any]:
        detection_prediction = next(iter(getattr(detection_result, "predictions", []) or []), None)
        detection_payload = dict(getattr(detection_prediction, "raw_response", {}) or {})
        ocr_prediction = next(iter(getattr(ocr_result, "predictions", []) or []), None)
        return {
            "status": getattr(ocr_result, "status", "disabled"),
            "reason": getattr(ocr_result, "reason", None) or getattr(quality_result, "reason", None) or getattr(detection_result, "reason", None),
            "plate_detected": bool(getattr(detection_result, "detected", False)),
            "plate_text": getattr(ocr_result, "text", None),
            "plate_detection_confidence": detection_payload.get("confidence"),
            "plate_bbox": detection_payload.get("bbox_xyxy"),
            "plate_crop_path": detection_payload.get("plate_crop_path"),
            "plate_text_confidence": getattr(ocr_prediction, "confidence", None) if ocr_prediction else None,
            "plate_ocr_raw_response": str(getattr(ocr_prediction, "raw_response", "")) if ocr_prediction else None,
            "quality_acceptable": getattr(quality_result, "acceptable", None),
        }
