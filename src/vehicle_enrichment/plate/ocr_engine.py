from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import cv2

from src.indian_plate_validator import validate_indian_plate
from ..ocr_mukul.image_preprocessor import resize_proportionally_if_needed
from ..schemas import ATTRIBUTE_STATUS_DISABLED, ATTRIBUTE_STATUS_ERROR, AttributePrediction, PlateOCRResult
from ..shared.florence_backend import FlorenceBackend

LOGGER = logging.getLogger(__name__)


class PlateOCREngine:
    def __init__(self, config: dict[str, Any], *, backend: FlorenceBackend | None = None) -> None:
        self.config = dict(config)
        self.enabled = bool(self.config.get("enabled", False))
        self.backend = backend
        self.task_token = str(self.config.get("task_token", "<OCR>")).strip() or "<OCR>"
        self.prompt = str(self.config.get("prompt", "") or "")

    @staticmethod
    def normalize_plate_text(raw_text: str) -> str:
        return re.sub(r"[^A-Z0-9]", "", str(raw_text or "").upper())

    def recognize(self, plate_crop_path: str | Path | None, *, frame_number: int | None = None, confidence: float | None = None) -> PlateOCRResult:
        if not self.enabled:
            return PlateOCRResult(
                text=None,
                predictions=[],
                status=ATTRIBUTE_STATUS_DISABLED,
                source="plate.ocr_engine",
                reason="plate_ocr_disabled",
            )
        if self.backend is None:
            return PlateOCRResult(
                text=None,
                predictions=[],
                status=ATTRIBUTE_STATUS_ERROR,
                source="plate.ocr_engine",
                reason="plate_ocr_backend_missing",
            )
        self.backend.load()
        if not self.backend.adapter_active:
            raise RuntimeError("Plate OCR requires the OCR_MUKUL adapter, but it is not active.")
        crop_path = Path(str(plate_crop_path or ""))
        image = cv2.imread(str(crop_path))
        if image is None or image.size == 0:
            return PlateOCRResult(
                text=None,
                predictions=[],
                status=ATTRIBUTE_STATUS_ERROR,
                source="plate.ocr_engine",
                reason="plate_crop_unreadable",
            )
        prepared = resize_proportionally_if_needed(image)
        response = self.backend.run_task(prepared, self.task_token, self.prompt, adapter_active=True)
        if response["status"] != "completed":
            return PlateOCRResult(
                text=None,
                predictions=[],
                status=ATTRIBUTE_STATUS_ERROR,
                source="plate.ocr_engine",
                reason=str(response.get("reason") or "plate_ocr_failed"),
            )
        payload = dict(response.get("payload") or {})
        parsed_answer = payload.get("parsed_answer")
        if isinstance(parsed_answer, dict):
            raw_text = str(parsed_answer.get(self.task_token) or payload.get("generated_text") or "")
        else:
            raw_text = str(parsed_answer or payload.get("generated_text") or "")
        validation = validate_indian_plate(raw_text)
        normalized_text = validation.normalized_text
        canonical_text = validation.canonical_text
        reason = validation.reason if validation.valid else f"plate_validation_failed:{validation.reason}"
        if not validation.valid:
            LOGGER.debug(
                "plate OCR rejected raw_text=%r normalized=%r valid=%s reason=%s attempted_candidates=%s",
                raw_text,
                normalized_text,
                validation.valid,
                validation.reason,
                validation.attempted_candidates,
            )
        return PlateOCRResult(
            text=canonical_text,
            raw_text=raw_text or None,
            normalized_text=normalized_text,
            format_type=validation.format_type,
            validation_status="valid" if validation.valid else "invalid",
            validation_reason=validation.reason,
            correction_applied=validation.correction_applied,
            correction_count=validation.correction_count,
            attempted_candidates=list(validation.attempted_candidates),
            predictions=[
                AttributePrediction(
                    attribute_name="plate_ocr",
                    label=canonical_text,
                    source_backend="ocr_mukul_adapter",
                    source_model=self.backend.model_identifier,
                    source_frame_number=frame_number,
                    source_crop_path=str(crop_path),
                    raw_response=raw_text,
                    confidence=confidence,
                    quality_weight=confidence,
                    adapter_active=True,
                    inference_duration_ms=float(payload.get("inference_duration_ms", 0.0) or 0.0),
                    original_crop_width=int(image.shape[1]),
                    original_crop_height=int(image.shape[0]),
                    status="completed",
                    reason=reason if normalized_text else "empty_ocr_response",
                    error=None,
                )
            ],
            status="completed",
            source="plate.ocr_engine",
            reason=reason if normalized_text else "empty_ocr_response",
        )
