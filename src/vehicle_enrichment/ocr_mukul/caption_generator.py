from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2

from ..schemas import EnrichmentEvidenceItem
from ..shared.florence_backend import FlorenceBackend
from .image_preprocessor import OCRMukulImagePreprocessor, OCRMukulPreparedImage


@dataclass(slots=True, frozen=True)
class CaptionInferenceResult:
    crop_path: str
    task_token: str
    prompt: str
    raw_generated_text: str
    post_processed_caption: str
    inference_time_ms: float
    prepared: OCRMukulPreparedImage
    pixel_values_shape: list[int] | None
    model_identifier: str
    processor_identifier: str


class OCRMukulCaptionGenerator:
    """
    Adapted from:
    D:\\project\\models\\OCR_MUKUL\\OCR_MUKUL\\anpr_frog_speed.py
    function: run_florence_inference
    """

    def __init__(self, backend: FlorenceBackend, *, task_token: str = "<CAPTION>", prompt: str = "") -> None:
        self.backend = backend
        self.task_token = task_token
        self.prompt = prompt
        self.preprocessor = OCRMukulImagePreprocessor()

    def generate(self, evidence_item: EnrichmentEvidenceItem) -> CaptionInferenceResult:
        crop_path = Path(str(evidence_item.vehicle_crop_path))
        image = cv2.imread(str(crop_path))
        if image is None or image.size == 0:
            raise FileNotFoundError(f"OCR_MUKUL crop image could not be read: {crop_path}")
        prepared = self.preprocessor.prepare(image)
        response = self.backend.run_task(
            prepared.image_bgr,
            self.task_token,
            self.prompt,
            adapter_active=self.backend.adapter_active,
        )
        if response["status"] != "completed":
            raise RuntimeError(str(response.get("reason") or "OCR_MUKUL caption inference failed."))
        payload = dict(response.get("payload") or {})
        post_processed = payload.get("parsed_answer")
        if isinstance(post_processed, str):
            caption = post_processed
        elif isinstance(post_processed, dict):
            caption = str(post_processed.get(self.task_token) or payload.get("generated_text") or "")
        else:
            caption = str(payload.get("generated_text") or "")
        return CaptionInferenceResult(
            crop_path=str(crop_path),
            task_token=self.task_token,
            prompt=self.prompt,
            raw_generated_text=str(payload.get("generated_text") or ""),
            post_processed_caption=caption.strip(),
            inference_time_ms=float(payload.get("inference_duration_ms", 0.0) or 0.0),
            prepared=prepared,
            pixel_values_shape=payload.get("pixel_values_shape"),
            model_identifier=str(payload.get("model_identifier") or self.backend.model_identifier),
            processor_identifier=str(payload.get("processor_identifier") or self.backend.processor_identifier),
        )
