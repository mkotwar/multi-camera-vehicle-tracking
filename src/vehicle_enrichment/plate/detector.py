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
        self.inference_size = detector_config.get("inference_size")
        self.device = str(detector_config.get("device", "") or "").strip() or None
        self.save_plate_crops = bool(detector_config.get("save_plate_crops", True))
        self._disabled_reason = "plate_detector_disabled"
        self._model: Any | None = None
        if self.enabled and not self.model_path:
            self.enabled = False
            self._disabled_reason = "plate_detector_model_path_missing"
        elif self.enabled and not Path(self.model_path).exists():
            self.enabled = False
            self._disabled_reason = "plate_detector_model_path_not_found"

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
        try:
            boxes = self._run_yolo(image)
        except Exception as exc:
            return PlateDetectionResult(
                detected=False,
                predictions=[],
                status=ATTRIBUTE_STATUS_ERROR,
                source="plate.detector",
                reason=f"plate_detector_failed:{exc}",
            )
        if not boxes:
            return PlateDetectionResult(
                detected=False,
                predictions=[],
                status="completed",
                source="plate.detector",
                reason="no_plate_detected",
            )

        best = max(boxes, key=lambda item: item["confidence"])
        x1, y1, x2, y2 = self._clamp_bbox(best["bbox"], width=int(image.shape[1]), height=int(image.shape[0]))
        plate_width = x2 - x1
        plate_height = y2 - y1
        if plate_width < self.minimum_crop_width or plate_height < self.minimum_crop_height:
            return PlateDetectionResult(
                detected=False,
                predictions=[],
                status="skipped",
                source="plate.detector",
                reason="plate_crop_below_minimum_size",
            )

        plate_crop_path = None
        if self.save_plate_crops:
            plate_crop = image[y1:y2, x1:x2]
            plate_crop_path = self._save_plate_crop(crop_path, plate_crop)

        prediction = AttributePrediction(
            attribute_name="plate_detection",
            label="PLATE",
            source_backend="ultralytics_yolo",
            source_model=self.model_path,
            source_frame_number=getattr(evidence_item, "frame_number", None),
            source_crop_path=str(plate_crop_path) if plate_crop_path else str(crop_path),
            raw_response={
                "bbox_xyxy": [float(x1), float(y1), float(x2), float(y2)],
                "confidence": float(best["confidence"]),
                "vehicle_crop_path": str(crop_path),
                "plate_crop_path": str(plate_crop_path) if plate_crop_path else None,
                "plate_crop_width": int(plate_width),
                "plate_crop_height": int(plate_height),
            },
            confidence=float(best["confidence"]),
            quality_weight=float(best["confidence"]),
            evidence_role=getattr(evidence_item, "evidence_role", None),
            original_crop_width=int(image.shape[1]),
            original_crop_height=int(image.shape[0]),
            status="completed",
            reason="plate_detected",
        )
        return PlateDetectionResult(
            detected=True,
            predictions=[prediction],
            status="completed",
            source="plate.detector",
            reason="plate_detected",
        )

    def _load_model(self) -> Any:
        if self._model is None:
            from ultralytics import YOLO

            self._model = YOLO(self.model_path)
        return self._model

    def _run_yolo(self, image: Any) -> list[dict[str, Any]]:
        model = self._load_model()
        kwargs: dict[str, Any] = {"conf": self.confidence_threshold, "verbose": False}
        if self.inference_size:
            kwargs["imgsz"] = int(self.inference_size)
        if self.device:
            kwargs["device"] = self.device
        results = model(image, **kwargs)
        detections: list[dict[str, Any]] = []
        if not results:
            return detections
        boxes = getattr(results[0], "boxes", None)
        if boxes is None:
            return detections
        for box in boxes:
            confidence = float(box.conf[0]) if getattr(box, "conf", None) is not None else 0.0
            if confidence < self.confidence_threshold:
                continue
            detections.append(
                {
                    "confidence": confidence,
                    "bbox": [float(value) for value in box.xyxy[0]],
                }
            )
        return detections

    @staticmethod
    def _clamp_bbox(bbox: list[float], *, width: int, height: int) -> tuple[int, int, int, int]:
        x1, y1, x2, y2 = bbox
        left = max(0, min(width - 1, int(round(x1))))
        top = max(0, min(height - 1, int(round(y1))))
        right = max(left + 1, min(width, int(round(x2))))
        bottom = max(top + 1, min(height, int(round(y2))))
        return left, top, right, bottom

    @staticmethod
    def _plate_directory(vehicle_crop_path: Path) -> Path:
        if vehicle_crop_path.parent.name.lower() in {"crops", "crop_candidates", "vehicle_crops"}:
            return vehicle_crop_path.parent.parent / "plate"
        return vehicle_crop_path.parent / "plate"

    def _save_plate_crop(self, vehicle_crop_path: Path, plate_crop: Any) -> Path | None:
        if plate_crop is None or plate_crop.size == 0:
            return None
        plate_dir = self._plate_directory(vehicle_crop_path)
        plate_dir.mkdir(parents=True, exist_ok=True)
        plate_path = plate_dir / f"{vehicle_crop_path.stem}_plate.jpg"
        cv2.imwrite(str(plate_path), plate_crop)
        return plate_path
