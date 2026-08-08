from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any

from ..image_size_policy import ImageSizePolicy
from ..schemas import (
    ATTRIBUTE_STATUS_ERROR,
    ATTRIBUTE_STATUS_NOT_RUN,
    AttributePrediction,
    TrackEnrichmentRequest,
    VEHICLE_BODY_TYPE_UNKNOWN,
    VEHICLE_COLOUR_UNKNOWN,
    VehicleBodyTypeResult,
    VehicleColourResult,
)
from ..shared.florence_backend import FlorenceBackend
from ..taxonomy import SUPPORTED_VEHICLE_CLASSES
from .aggregator import aggregate_predictions
from .attribute_parser import OCR_MUKUL_UNKNOWN, ParsedCaptionAttributes, parse_caption_attributes
from .caption_generator import CaptionInferenceResult, OCRMukulCaptionGenerator


@dataclass(slots=True, frozen=True)
class OCRMukulFlowResult:
    body_type: VehicleBodyTypeResult
    colour: VehicleColourResult
    crop_level_rows: list[dict[str, Any]]
    caption_inference_count: int
    adapter_loaded: bool


class OCRMukulFlorenceFlow:
    def __init__(
        self,
        config: dict[str, Any],
        *,
        backend: FlorenceBackend,
        image_size_policy: ImageSizePolicy,
        logger: logging.Logger,
    ) -> None:
        self.config = dict(config)
        self.backend = backend
        self.image_size_policy = image_size_policy
        self.logger = logger
        self.enabled = bool(self.config.get("enabled", False))
        self.maximum_crops_per_track = max(1, int(self.config.get("maximum_crops_per_track", 3)))
        self.reuse_caption_for_attributes = bool(self.config.get("reuse_caption_for_attributes", True))
        self.task_token = str(self.config.get("task_token", "<CAPTION>")).strip() or "<CAPTION>"
        self.prompt = str(self.config.get("prompt", "") or "")
        self.caption_generator = OCRMukulCaptionGenerator(backend, task_token=self.task_token, prompt=self.prompt)
        self.body_vehicle_classes = {
            str(item).strip().upper()
            for item in self.config.get("body_type_vehicle_classes", ["CAR"])
            if str(item).strip()
        }
        self.colour_vehicle_classes = {
            str(item).strip().upper()
            for item in self.config.get("colour_vehicle_classes", SUPPORTED_VEHICLE_CLASSES)
            if str(item).strip()
        }
        self._metrics: dict[str, Any] = {
            "ocr_mukul_enabled": self.enabled,
            "ocr_mukul_adapter_requested": True,
            "ocr_mukul_adapter_loaded": False,
            "ocr_mukul_caption_calls": 0,
            "ocr_mukul_captions_generated": 0,
            "ocr_mukul_caption_failures": 0,
            "ocr_mukul_caption_reused_for_body_type": 0,
            "ocr_mukul_caption_reused_for_colour": 0,
            "ocr_mukul_body_type_valid": 0,
            "ocr_mukul_body_type_unknown": 0,
            "ocr_mukul_body_type_conflicts": 0,
            "ocr_mukul_colour_valid": 0,
            "ocr_mukul_colour_unknown": 0,
            "ocr_mukul_colour_conflicts": 0,
            "ocr_mukul_generic_caption_count": 0,
            "ocr_mukul_prompt_echo_count": 0,
            "ocr_mukul_total_inference_time_ms": 0.0,
            "ocr_mukul_average_inference_time_ms": 0.0,
        }

    @property
    def metrics(self) -> dict[str, Any]:
        payload = dict(self._metrics)
        if self._metrics["ocr_mukul_captions_generated"] > 0:
            payload["ocr_mukul_average_inference_time_ms"] = float(
                self._metrics["ocr_mukul_total_inference_time_ms"] / self._metrics["ocr_mukul_captions_generated"]
            )
        return payload

    def classify(self, request: TrackEnrichmentRequest) -> OCRMukulFlowResult:
        if not self.enabled:
            return OCRMukulFlowResult(
                body_type=VehicleBodyTypeResult(label=VEHICLE_BODY_TYPE_UNKNOWN, status="disabled", source="ocr_mukul"),
                colour=VehicleColourResult(label=VEHICLE_COLOUR_UNKNOWN, status="disabled", source="ocr_mukul"),
                crop_level_rows=[],
                caption_inference_count=0,
                adapter_loaded=False,
            )
        self.backend.load()
        if not self.backend.adapter_active:
            raise RuntimeError("OCR_MUKUL mode requires the Florence adapter, but it is not active.")
        self._metrics["ocr_mukul_adapter_loaded"] = True
        selected_items = list(request.evidence_items[: self.maximum_crops_per_track])
        body_predictions: list[AttributePrediction] = []
        colour_predictions: list[AttributePrediction] = []
        crop_rows: list[dict[str, Any]] = []
        track_class = str(request.vehicle_class).strip().upper()
        for item in selected_items:
            original_width = int(getattr(item, "original_crop_width", 0) or 0)
            original_height = int(getattr(item, "original_crop_height", 0) or 0)
            if not self.image_size_policy.florence.is_eligible(original_width, original_height):
                continue
            self._metrics["ocr_mukul_caption_calls"] += 1
            try:
                caption_result = self.caption_generator.generate(item)
            except Exception as exc:
                self._metrics["ocr_mukul_caption_failures"] += 1
                body_predictions.append(self._error_prediction(item, "body", str(exc)))
                colour_predictions.append(self._error_prediction(item, "colour", str(exc)))
                continue
            self._metrics["ocr_mukul_captions_generated"] += 1
            self._metrics["ocr_mukul_total_inference_time_ms"] += caption_result.inference_time_ms
            parsed = parse_caption_attributes(caption_result.post_processed_caption)
            if parsed.normalized_body_type == OCR_MUKUL_UNKNOWN and parsed.normalized_colour == OCR_MUKUL_UNKNOWN:
                self._metrics["ocr_mukul_generic_caption_count"] += 1
            body_prediction = self._body_prediction(item, caption_result, parsed, enabled=track_class in self.body_vehicle_classes)
            colour_prediction = self._colour_prediction(item, caption_result, parsed, enabled=track_class in self.colour_vehicle_classes)
            body_predictions.append(body_prediction)
            colour_predictions.append(colour_prediction)
            if body_prediction.status == "completed":
                self._metrics["ocr_mukul_caption_reused_for_body_type"] += 1
            if colour_prediction.status == "completed":
                self._metrics["ocr_mukul_caption_reused_for_colour"] += 1
            crop_rows.append(
                {
                    "camera_id": request.camera_id,
                    "local_track_id": request.local_track_id,
                    "frame_index": item.frame_number,
                    "crop_path": str(item.vehicle_crop_path),
                    "original_crop_width": original_width,
                    "original_crop_height": original_height,
                    "resolution_tier": getattr(item, "resolution_tier", self.image_size_policy.florence.resolution_tier(original_width, original_height)),
                    "quality_score": float(getattr(item, "quality_score", 0.0) or 0.0),
                    "caption": caption_result.post_processed_caption,
                    "raw_body_type_phrase": parsed.raw_body_type_phrase,
                    "normalized_body_type": body_prediction.label,
                    "raw_colour_phrase": parsed.raw_colour_phrase,
                    "normalized_colour": colour_prediction.label,
                    "inference_time_ms": caption_result.inference_time_ms,
                }
            )

        body_label, body_reason, body_agreement, body_weight = aggregate_predictions(
            body_predictions,
            unknown_label=VEHICLE_BODY_TYPE_UNKNOWN,
            conflict_reason="conflicting_body_type_predictions",
        )
        colour_label, colour_reason, colour_agreement, colour_weight = aggregate_predictions(
            colour_predictions,
            unknown_label=VEHICLE_COLOUR_UNKNOWN,
            conflict_reason="conflicting_colour_predictions",
        )
        self._count_final_metrics(body_label, colour_label, body_reason, colour_reason)
        return OCRMukulFlowResult(
            body_type=VehicleBodyTypeResult(
                label=body_label,
                predictions=body_predictions,
                status="completed",
                source="ocr_mukul",
                model=self.backend.model_identifier,
                adapter_active=self.backend.adapter_active,
                aggregation_reason=body_reason,
                agreement_score=body_agreement,
                accumulated_quality_weight=body_weight,
                task_prompt=self.task_token,
                prompt_text=self.prompt,
            ),
            colour=VehicleColourResult(
                label=colour_label,
                predictions=colour_predictions,
                status="completed",
                source="ocr_mukul",
                model=self.backend.model_identifier,
                adapter_active=self.backend.adapter_active,
                aggregation_reason=colour_reason,
                agreement_score=colour_agreement,
                accumulated_quality_weight=colour_weight,
                task_prompt=self.task_token,
                prompt_text=self.prompt,
            ),
            crop_level_rows=crop_rows,
            caption_inference_count=len(crop_rows),
            adapter_loaded=self.backend.adapter_active,
        )

    def _body_prediction(self, item: Any, caption_result: CaptionInferenceResult, parsed: ParsedCaptionAttributes, *, enabled: bool) -> AttributePrediction:
        label = parsed.normalized_body_type if enabled else VEHICLE_BODY_TYPE_UNKNOWN
        status = "completed" if enabled else "skipped"
        reason = parsed.body_type_reason if enabled else "vehicle_class_not_eligible"
        return AttributePrediction(
            attribute_name="vehicle_body_type",
            label=label,
            source_backend="ocr_mukul",
            source_model=caption_result.model_identifier,
            source_frame_number=item.frame_number,
            source_crop_path=str(item.vehicle_crop_path),
            raw_response=caption_result.post_processed_caption,
            confidence=None,
            quality_weight=float(getattr(item, "quality_score", 0.0) or 0.0),
            evidence_role=getattr(item, "evidence_role", None),
            adapter_active=True,
            inference_duration_ms=caption_result.inference_time_ms,
            original_crop_width=caption_result.prepared.original_width,
            original_crop_height=caption_result.prepared.original_height,
            resolution_tier=str(getattr(item, "resolution_tier", "")),
            square_padding_applied=False,
            padded_width=caption_result.prepared.preprocessed_width,
            padded_height=caption_result.prepared.preprocessed_height,
            florence_input_width=caption_result.prepared.preprocessed_width,
            florence_input_height=caption_result.prepared.preprocessed_height,
            status=status,
            reason=reason,
            error=None,
        )

    def _colour_prediction(self, item: Any, caption_result: CaptionInferenceResult, parsed: ParsedCaptionAttributes, *, enabled: bool) -> AttributePrediction:
        label = parsed.normalized_colour if enabled else VEHICLE_COLOUR_UNKNOWN
        status = "completed" if enabled else "skipped"
        reason = parsed.colour_reason if enabled else "vehicle_class_not_eligible"
        return AttributePrediction(
            attribute_name="vehicle_colour",
            label=label,
            source_backend="ocr_mukul",
            source_model=caption_result.model_identifier,
            source_frame_number=item.frame_number,
            source_crop_path=str(item.vehicle_crop_path),
            raw_response=caption_result.post_processed_caption,
            confidence=None,
            quality_weight=float(getattr(item, "quality_score", 0.0) or 0.0),
            evidence_role=getattr(item, "evidence_role", None),
            adapter_active=True,
            inference_duration_ms=caption_result.inference_time_ms,
            original_crop_width=caption_result.prepared.original_width,
            original_crop_height=caption_result.prepared.original_height,
            resolution_tier=str(getattr(item, "resolution_tier", "")),
            square_padding_applied=False,
            padded_width=caption_result.prepared.preprocessed_width,
            padded_height=caption_result.prepared.preprocessed_height,
            florence_input_width=caption_result.prepared.preprocessed_width,
            florence_input_height=caption_result.prepared.preprocessed_height,
            status=status,
            reason=reason,
            error=None,
        )

    @staticmethod
    def _error_prediction(item: Any, attribute_name: str, error: str) -> AttributePrediction:
        return AttributePrediction(
            attribute_name=f"vehicle_{attribute_name}",
            label=OCR_MUKUL_UNKNOWN,
            source_backend="ocr_mukul",
            source_model=None,
            source_frame_number=getattr(item, "frame_number", None),
            source_crop_path=str(getattr(item, "vehicle_crop_path", "") or ""),
            raw_response=None,
            confidence=None,
            quality_weight=float(getattr(item, "quality_score", 0.0) or 0.0),
            evidence_role=getattr(item, "evidence_role", None),
            adapter_active=True,
            inference_duration_ms=None,
            original_crop_width=int(getattr(item, "original_crop_width", 0) or 0),
            original_crop_height=int(getattr(item, "original_crop_height", 0) or 0),
            resolution_tier=str(getattr(item, "resolution_tier", "")),
            square_padding_applied=False,
            padded_width=None,
            padded_height=None,
            florence_input_width=None,
            florence_input_height=None,
            status=ATTRIBUTE_STATUS_ERROR,
            reason="caption_inference_failed",
            error=error,
        )

    def _count_final_metrics(self, body_label: str, colour_label: str, body_reason: str, colour_reason: str) -> None:
        if body_label == VEHICLE_BODY_TYPE_UNKNOWN:
            self._metrics["ocr_mukul_body_type_unknown"] += 1
            if body_reason == "conflicting_body_type_predictions":
                self._metrics["ocr_mukul_body_type_conflicts"] += 1
        else:
            self._metrics["ocr_mukul_body_type_valid"] += 1
        if colour_label == VEHICLE_COLOUR_UNKNOWN:
            self._metrics["ocr_mukul_colour_unknown"] += 1
            if colour_reason == "conflicting_colour_predictions":
                self._metrics["ocr_mukul_colour_conflicts"] += 1
        else:
            self._metrics["ocr_mukul_colour_valid"] += 1
