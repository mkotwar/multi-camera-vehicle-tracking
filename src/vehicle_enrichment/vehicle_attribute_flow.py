from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from .body_type.labels import (
    BODY_TYPE_ALLOWED_LABELS,
    BODY_TYPE_PROMPT_TEXT,
    BODY_TYPE_TASK_PROMPT,
    normalize_body_type_label,
)
from .image_size_policy import ImageSizePolicy, pad_to_square
from .ocr_mukul.aggregator import aggregate_predictions
from .ocr_mukul.attribute_parser import ParsedCaptionAttributes, parse_caption_attributes
from .vehicle_attribute_prompts import assess_response_quality
from .schemas import (
    ATTRIBUTE_STATUS_DISABLED,
    ATTRIBUTE_STATUS_ERROR,
    AttributePrediction,
    TrackEnrichmentRequest,
    VEHICLE_BODY_TYPE_UNKNOWN,
    VEHICLE_COLOUR_UNKNOWN,
    VehicleBodyTypeResult,
    VehicleColourResult,
)
from .shared.florence_backend import FlorenceBackend


@dataclass(slots=True, frozen=True)
class VehicleAttributeFlowResult:
    body_type: VehicleBodyTypeResult
    colour: VehicleColourResult
    crop_level_rows: list[dict[str, Any]]
    inference_count: int
    adapter_loaded: bool
    raw_responses: list[str]
    body_type_eligible: bool
    body_type_candidate_crop_count: int
    body_type_selected_crop_count: int
    body_type_florence_call_count: int
    body_type_valid_prediction_count: int
    body_type_failure_reason: str | None


class BaseFlorenceVehicleAttributesFlow:
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
        self.task_token = str(self.config.get("task_token", "<VQA>")).strip() or "<VQA>"
        self.prompt = str(self.config.get("prompt", "") or "")
        self.reuse_single_response_for_attributes = bool(self.config.get("reuse_single_response_for_attributes", True))
        colour_section = dict(self.config.get("colour", {}) or {})
        body_type_section = dict(self.config.get("body_type", {}) or {})
        nested_attribute_sections_present = bool(colour_section or body_type_section)
        self.colour_enabled = bool(colour_section.get("enabled", True if not nested_attribute_sections_present else False))
        self.body_type_enabled = bool(body_type_section.get("enabled", True))
        self.colour_task_token = str(colour_section.get("task_token", self.task_token)).strip() or self.task_token
        self.colour_prompt = str(colour_section.get("prompt", self.prompt) or "")
        self.colour_generation = dict(colour_section.get("generation", {}) or {})
        self.body_type_task_token = str(body_type_section.get("task_token", self.task_token)).strip() or self.task_token
        self.body_type_prompt = str(body_type_section.get("prompt", BODY_TYPE_PROMPT_TEXT) or BODY_TYPE_PROMPT_TEXT)
        self.body_type_generation = dict(body_type_section.get("generation", {}) or {})
        self.body_type_car_only = bool(body_type_section.get("car_only", True))
        self.body_type_allowed_labels = [
            str(item).strip().upper() for item in body_type_section.get("allowed_labels", BODY_TYPE_ALLOWED_LABELS) if str(item).strip()
        ]
        self.body_type_vehicle_classes = {
            str(item).strip().upper()
            for item in body_type_section.get("run_only_when_vehicle_class", ["CAR"])
            if str(item).strip()
        }
        self.body_type_maximum_crops_per_track = max(1, int(body_type_section.get("maximum_crops_per_track", self.maximum_crops_per_track)))
        self.body_type_minimum_original_width = int(
            body_type_section.get("minimum_original_width", body_type_section.get("minimum_crop_width", 120))
        )
        self.body_type_minimum_original_height = int(
            body_type_section.get("minimum_original_height", body_type_section.get("minimum_crop_height", 100))
        )
        self.body_type_preferred_original_width = int(body_type_section.get("preferred_original_width", 240))
        self.body_type_preferred_original_height = int(body_type_section.get("preferred_original_height", 160))
        self._metrics: dict[str, Any] = {
            "vehicle_attribute_base_loads": 0,
            "vehicle_attribute_inference_calls": 0,
            "vehicle_attribute_colour_inference_calls": 0,
            "vehicle_attribute_body_inference_calls": 0,
            "vehicle_attribute_valid_colour": 0,
            "vehicle_attribute_unknown_colour": 0,
            "vehicle_attribute_valid_body_type": 0,
            "vehicle_attribute_unknown_body_type": 0,
            "vehicle_attribute_prompt_echo_count": 0,
            "vehicle_attribute_plate_like_response_count": 0,
            "vehicle_attribute_missing_colour_value_count": 0,
            "vehicle_attribute_unsupported_colour_count": 0,
            "vehicle_attribute_empty_response_count": 0,
            "vehicle_attribute_total_colour_inference_ms": 0.0,
            "vehicle_attribute_average_colour_inference_time_ms": 0.0,
            "vehicle_attribute_total_body_inference_ms": 0.0,
            "vehicle_attribute_average_body_inference_time_ms": 0.0,
            "vehicle_attribute_skipped_missing_crop": 0,
            "vehicle_attribute_tracks_with_fallback_selected": 0,
            "vehicle_attribute_tracks_with_zero_florence_calls": 0,
            "vehicle_attribute_non_car_body_type_skips": 0,
            "vehicle_attribute_body_type_no_usable_crop_skips": 0,
            "gpu_memory_before_attribute_load_mb": 0.0,
            "gpu_memory_after_attribute_load_mb": 0.0,
        }

    @property
    def metrics(self) -> dict[str, Any]:
        return dict(self._metrics)

    def classify(self, request: TrackEnrichmentRequest) -> VehicleAttributeFlowResult:
        if not self.enabled:
            return VehicleAttributeFlowResult(
                body_type=self._disabled_body_type_result(),
                colour=self._disabled_colour_result(),
                crop_level_rows=[],
                inference_count=0,
                adapter_loaded=False,
                raw_responses=[],
                body_type_eligible=False,
                body_type_candidate_crop_count=0,
                body_type_selected_crop_count=0,
                body_type_florence_call_count=0,
                body_type_valid_prediction_count=0,
                body_type_failure_reason="disabled",
            )
        self._metrics["gpu_memory_before_attribute_load_mb"] = float(self.backend.metrics.get("gpu_memory_allocated_mb") or 0.0)
        self.backend.load()
        self._metrics["vehicle_attribute_base_loads"] += 1
        self._metrics["gpu_memory_after_attribute_load_mb"] = float(self.backend.metrics.get("gpu_memory_allocated_mb") or 0.0)
        if self.backend.adapter_active:
            raise RuntimeError("Vehicle attribute flow must use base Florence without the OCR_MUKUL adapter.")

        selected_items = list(request.evidence_items[: self.maximum_crops_per_track])
        if any(str(getattr(item, "colour_selection_tier", "") or "") == "low_resolution_fallback" for item in selected_items):
            self._metrics["vehicle_attribute_tracks_with_fallback_selected"] += 1
        body_predictions: list[AttributePrediction] = []
        colour_predictions: list[AttributePrediction] = []
        crop_rows_by_key: dict[tuple[str, int], dict[str, Any]] = {}
        raw_responses: list[str] = []
        local_inference_count = 0
        for item in selected_items:
            self._ensure_crop_row(crop_rows_by_key, request, item)
            try:
                colour_payload = self._infer_single_crop(
                    item,
                    task_token=self.colour_task_token,
                    prompt=self.colour_prompt,
                    generation=self.colour_generation,
                )
            except Exception as exc:
                colour_predictions.append(self._error_prediction(item, "vehicle_colour", str(exc)))
                self._metrics["vehicle_attribute_skipped_missing_crop"] += int("could not be read" in str(exc).lower() or "does not exist" in str(exc).lower())
                row = crop_rows_by_key[(str(item.vehicle_crop_path), int(item.frame_number))]
                row["colour_status"] = "skipped"
                row["colour_reason"] = "missing_crop"
                row["crop_available"] = False
                row["crop_skip_reason"] = "missing_crop"
                continue
            parsed = parse_caption_attributes(str(colour_payload["post_processed_response"]))
            raw_responses.append(str(colour_payload["raw_generated_text"]))
            colour_status, colour_reason = assess_response_quality(
                str(colour_payload["raw_generated_text"]),
                parsed,
                attribute_task="colour",
                prompt=self.colour_prompt,
            )
            colour_prediction = self._prediction_from_parsed(item, parsed, colour_payload, "vehicle_colour", status=colour_status, reason=colour_reason)
            colour_predictions.append(colour_prediction)
            self._metrics["vehicle_attribute_inference_calls"] += 1
            self._metrics["vehicle_attribute_colour_inference_calls"] += 1
            self._metrics["vehicle_attribute_total_colour_inference_ms"] += float(colour_payload["inference_time_ms"])
            local_inference_count += 1
            self._count_response_reason(colour_reason)
            row = crop_rows_by_key[(str(item.vehicle_crop_path), int(item.frame_number))]
            row["colour_task_token"] = self.colour_task_token
            row["colour_prompt"] = self.colour_prompt
            row["colour_effective_processor_text"] = f"{self.colour_task_token}{self.colour_prompt}"
            row["colour_raw_response"] = colour_payload["raw_generated_text"]
            row["colour_post_processed_response"] = colour_payload["post_processed_response"]
            row["parsed_colour"] = colour_prediction.label
            row["colour_status"] = colour_status
            row["colour_reason"] = colour_prediction.reason
            row["colour_inference_time_ms"] = colour_payload["inference_time_ms"]

        body_type_eligible = self._body_type_eligible(request.vehicle_class)
        body_type_failure_reason: str | None = None
        body_type_candidate_items = [item for item in selected_items if self._is_body_type_candidate(item)]
        body_type_selected_items = self._select_body_type_items(body_type_candidate_items)
        if self.body_type_enabled and not body_type_eligible:
            self._metrics["vehicle_attribute_non_car_body_type_skips"] += 1
            body_type_failure_reason = "non_car_vehicle"
        elif self.body_type_enabled and not body_type_selected_items:
            self._metrics["vehicle_attribute_body_type_no_usable_crop_skips"] += 1
            body_type_failure_reason = "no_body_type_usable_crop"
        if self.body_type_enabled and body_type_eligible and body_type_selected_items:
            for item in body_type_selected_items:
                self._ensure_crop_row(crop_rows_by_key, request, item)
                try:
                    body_payload = self._infer_single_crop(
                        item,
                        task_token=self.body_type_task_token,
                        prompt=self.body_type_prompt,
                        generation=self.body_type_generation,
                    )
                except Exception as exc:
                    body_predictions.append(self._error_prediction(item, "vehicle_body_type", str(exc)))
                    row = crop_rows_by_key[(str(item.vehicle_crop_path), int(item.frame_number))]
                    row["body_type_status"] = ATTRIBUTE_STATUS_ERROR
                    row["body_type_reason"] = "inference_failed"
                    continue
                body_prediction = self._prediction_from_body_payload(item, body_payload)
                body_predictions.append(body_prediction)
                self._metrics["vehicle_attribute_inference_calls"] += 1
                self._metrics["vehicle_attribute_body_inference_calls"] += 1
                self._metrics["vehicle_attribute_total_body_inference_ms"] += float(body_payload["inference_time_ms"])
                local_inference_count += 1
                raw_responses.append(str(body_payload["raw_generated_text"]))
                row = crop_rows_by_key[(str(item.vehicle_crop_path), int(item.frame_number))]
                row["body_type_task_token"] = self.body_type_task_token
                row["body_type_prompt"] = self.body_type_prompt
                row["body_type_effective_processor_text"] = f"{self.body_type_task_token}{self.body_type_prompt}"
                row["body_type_raw_response"] = body_payload["raw_generated_text"]
                row["body_type_post_processed_response"] = body_payload["post_processed_response"]
                row["parsed_body_type"] = body_prediction.label
                row["body_type_status"] = body_prediction.status
                row["body_type_reason"] = body_prediction.reason
                row["body_type_inference_time_ms"] = body_payload["inference_time_ms"]

        if self.body_type_enabled:
            if not body_type_eligible:
                body_label, body_reason, body_agreement, body_weight = VEHICLE_BODY_TYPE_UNKNOWN, "non_car_vehicle", None, 0.0
            elif not body_type_selected_items:
                body_label, body_reason, body_agreement, body_weight = VEHICLE_BODY_TYPE_UNKNOWN, "no_body_type_usable_crop", None, 0.0
            else:
                body_label, body_reason, body_agreement, body_weight = aggregate_predictions(
                    body_predictions,
                    unknown_label=VEHICLE_BODY_TYPE_UNKNOWN,
                    conflict_reason="conflicting_body_type_predictions",
                )
        else:
            body_label, body_reason, body_agreement, body_weight = VEHICLE_BODY_TYPE_UNKNOWN, "disabled", None, 0.0
        colour_label, colour_reason, colour_agreement, colour_weight = aggregate_predictions(
            colour_predictions,
            unknown_label=VEHICLE_COLOUR_UNKNOWN,
            conflict_reason="conflicting_colour_predictions",
        )
        self._metrics["vehicle_attribute_average_colour_inference_time_ms"] = (
            float(self._metrics["vehicle_attribute_total_colour_inference_ms"] / self._metrics["vehicle_attribute_colour_inference_calls"])
            if self._metrics["vehicle_attribute_colour_inference_calls"] > 0
            else 0.0
        )
        self._metrics["vehicle_attribute_average_body_inference_time_ms"] = (
            float(self._metrics["vehicle_attribute_total_body_inference_ms"] / self._metrics["vehicle_attribute_body_inference_calls"])
            if self._metrics["vehicle_attribute_body_inference_calls"] > 0
            else 0.0
        )
        self._metrics["vehicle_attribute_valid_body_type"] += int(self.body_type_enabled and body_label != VEHICLE_BODY_TYPE_UNKNOWN)
        self._metrics["vehicle_attribute_unknown_body_type"] += int(self.body_type_enabled and body_label == VEHICLE_BODY_TYPE_UNKNOWN)
        self._metrics["vehicle_attribute_valid_colour"] += int(colour_label != VEHICLE_COLOUR_UNKNOWN)
        self._metrics["vehicle_attribute_unknown_colour"] += int(colour_label == VEHICLE_COLOUR_UNKNOWN)
        self._metrics["vehicle_attribute_tracks_with_zero_florence_calls"] += int(not colour_predictions)
        crop_rows = sorted(crop_rows_by_key.values(), key=lambda item: (int(item["frame_index"]), str(item["vehicle_crop_path"])))
        body_type_valid_prediction_count = sum(
            1 for prediction in body_predictions if prediction.status == "completed" and prediction.label not in {"", VEHICLE_BODY_TYPE_UNKNOWN, None}
        )
        return VehicleAttributeFlowResult(
            body_type=(
                VehicleBodyTypeResult(
                    label=body_label,
                    predictions=body_predictions,
                    status="completed" if body_type_eligible and body_type_selected_items else "skipped",
                    source="base_florence",
                    model=self.backend.model_identifier,
                    adapter_active=False,
                    reason=None if body_type_eligible and body_type_selected_items else body_reason,
                    aggregation_reason=body_reason,
                    agreement_score=body_agreement,
                    accumulated_quality_weight=body_weight,
                    task_prompt=self.body_type_task_token,
                    prompt_text=self.body_type_prompt,
                )
                if self.body_type_enabled
                else self._disabled_body_type_result()
            ),
            colour=VehicleColourResult(
                label=colour_label,
                predictions=colour_predictions,
                status="completed",
                source="base_florence",
                model=self.backend.model_identifier,
                adapter_active=False,
                aggregation_reason=colour_reason,
                agreement_score=colour_agreement,
                accumulated_quality_weight=colour_weight,
                task_prompt=self.colour_task_token,
                prompt_text=self.colour_prompt,
            ),
            crop_level_rows=crop_rows,
            inference_count=int(local_inference_count),
            adapter_loaded=False,
            raw_responses=raw_responses,
            body_type_eligible=body_type_eligible,
            body_type_candidate_crop_count=len(body_type_candidate_items),
            body_type_selected_crop_count=len(body_type_selected_items),
            body_type_florence_call_count=len(body_predictions),
            body_type_valid_prediction_count=body_type_valid_prediction_count,
            body_type_failure_reason=body_type_failure_reason or body_reason,
        )

    def _infer_single_crop(self, item: Any, *, task_token: str, prompt: str, generation: dict[str, Any]) -> dict[str, Any]:
        crop_path = Path(str(item.vehicle_crop_path))
        image = cv2.imread(str(crop_path))
        if image is None or image.size == 0:
            raise FileNotFoundError(f"Vehicle attribute crop image could not be read: {crop_path}")
        prepared_image, padding_metadata = self._prepare_image_for_florence(image)
        response = self.backend.run_task(
            prepared_image,
            task_token,
            prompt,
            adapter_active=False,
            generation_overrides=generation or None,
        )
        if response["status"] != "completed":
            raise RuntimeError(str(response.get("reason") or "Vehicle attribute Florence inference failed."))
        payload = dict(response.get("payload") or {})
        raw_generated_text = str(payload.get("generated_text") or "")
        post_processed = payload.get("parsed_answer")
        if isinstance(post_processed, dict):
            response_text = str(post_processed.get(task_token) or raw_generated_text)
        else:
            response_text = str(post_processed or raw_generated_text)
        return {
            "raw_generated_text": raw_generated_text.strip(),
            "post_processed_response": response_text.strip(),
            "inference_time_ms": float(payload.get("inference_duration_ms", 0.0) or 0.0),
            "padded_width": int(padding_metadata["padded_width"]),
            "padded_height": int(padding_metadata["padded_height"]),
        }

    def _prediction_from_body_payload(self, item: Any, response_payload: dict[str, Any]) -> AttributePrediction:
        normalized_label, normalization_reason = normalize_body_type_label(str(response_payload["post_processed_response"]))
        if normalized_label not in self.body_type_allowed_labels:
            normalized_label = VEHICLE_BODY_TYPE_UNKNOWN
            normalization_reason = "unsupported_body_type"
        original_width = int(getattr(item, "original_crop_width", 0) or getattr(item, "crop_width", 0) or 0)
        original_height = int(getattr(item, "original_crop_height", 0) or getattr(item, "crop_height", 0) or 0)
        return AttributePrediction(
            attribute_name="vehicle_body_type",
            label=normalized_label,
            source_backend="base_florence",
            source_model=self.backend.model_identifier,
            source_frame_number=item.frame_number,
            source_crop_path=str(item.vehicle_crop_path),
            raw_response=response_payload["raw_generated_text"],
            confidence=None,
            quality_weight=float(getattr(item, "quality_score", 0.0) or 0.0),
            evidence_role=getattr(item, "evidence_role", None),
            adapter_active=False,
            inference_duration_ms=float(response_payload["inference_time_ms"]),
            original_crop_width=original_width,
            original_crop_height=original_height,
            resolution_tier=str(getattr(item, "resolution_tier", "")),
            square_padding_applied=bool(response_payload["padded_width"] != original_width or response_payload["padded_height"] != original_height),
            padded_width=int(response_payload["padded_width"]),
            padded_height=int(response_payload["padded_height"]),
            florence_input_width=int(response_payload["padded_width"]),
            florence_input_height=int(response_payload["padded_height"]),
            status="completed",
            reason=normalization_reason,
            error=None,
        )

    def _prediction_from_parsed(self, item: Any, parsed: ParsedCaptionAttributes, response_payload: dict[str, Any], attribute_name: str, *, status: str = "valid", reason: str | None = None) -> AttributePrediction:
        if attribute_name == "vehicle_body_type":
            label = parsed.normalized_body_type
            resolved_reason = parsed.body_type_reason
        else:
            label = parsed.normalized_colour if status == "valid" else VEHICLE_COLOUR_UNKNOWN
            resolved_reason = reason or parsed.colour_reason
        original_width = int(getattr(item, "original_crop_width", 0) or getattr(item, "crop_width", 0) or 0)
        original_height = int(getattr(item, "original_crop_height", 0) or getattr(item, "crop_height", 0) or 0)
        return AttributePrediction(
            attribute_name=attribute_name,
            label=label,
            source_backend="base_florence",
            source_model=self.backend.model_identifier,
            source_frame_number=item.frame_number,
            source_crop_path=str(item.vehicle_crop_path),
            raw_response=response_payload["raw_generated_text"],
            confidence=None,
            quality_weight=float(getattr(item, "quality_score", 0.0) or 0.0),
            evidence_role=getattr(item, "evidence_role", None),
            adapter_active=False,
            inference_duration_ms=float(response_payload["inference_time_ms"]),
            original_crop_width=original_width,
            original_crop_height=original_height,
            resolution_tier=str(getattr(item, "resolution_tier", "")),
            square_padding_applied=bool(response_payload["padded_width"] != original_width or response_payload["padded_height"] != original_height),
            padded_width=int(response_payload["padded_width"]),
            padded_height=int(response_payload["padded_height"]),
            florence_input_width=int(response_payload["padded_width"]),
            florence_input_height=int(response_payload["padded_height"]),
            status="completed",
            reason=resolved_reason,
            error=None,
        )

    def _count_response_reason(self, reason: str) -> None:
        if reason == "prompt_echo":
            self._metrics["vehicle_attribute_prompt_echo_count"] += 1
        elif reason == "plate_like_response":
            self._metrics["vehicle_attribute_plate_like_response_count"] += 1
        elif reason == "missing_colour_value":
            self._metrics["vehicle_attribute_missing_colour_value_count"] += 1
        elif reason == "unsupported_colour":
            self._metrics["vehicle_attribute_unsupported_colour_count"] += 1
        elif reason == "empty_response":
            self._metrics["vehicle_attribute_empty_response_count"] += 1

    @staticmethod
    def _resolve_crop_source(item: Any) -> str:
        crop_path = str(getattr(item, "vehicle_crop_path", "") or "")
        source_image_path = str(getattr(item, "source_image_path", "") or "")
        if crop_path and "vehicle_enrichment\\crops" in crop_path.replace("/", "\\"):
            return "saved_vehicle_crop"
        if source_image_path and crop_path and source_image_path != crop_path:
            return "evidence_cache"
        if crop_path:
            return "fallback_source"
        return "missing"

    def _disabled_body_type_result(self) -> VehicleBodyTypeResult:
        backend_loaded = bool(getattr(self.backend, "is_loaded", False))
        return VehicleBodyTypeResult(
            label=VEHICLE_BODY_TYPE_UNKNOWN,
            predictions=[],
            status=ATTRIBUTE_STATUS_DISABLED,
            source="base_florence",
            reason="disabled",
            model=self.backend.model_identifier if backend_loaded else None,
            adapter_active=False,
            task_prompt=self.body_type_task_token or BODY_TYPE_TASK_PROMPT,
            prompt_text=self.body_type_prompt or BODY_TYPE_PROMPT_TEXT,
        )

    def _disabled_colour_result(self) -> VehicleColourResult:
        backend_loaded = bool(getattr(self.backend, "is_loaded", False))
        return VehicleColourResult(
            label=VEHICLE_COLOUR_UNKNOWN,
            predictions=[],
            status=ATTRIBUTE_STATUS_DISABLED,
            source="base_florence",
            reason="disabled",
            model=self.backend.model_identifier if backend_loaded else None,
            adapter_active=False,
            task_prompt=self.colour_task_token,
            prompt_text=self.colour_prompt,
        )

    @staticmethod
    def _error_prediction(item: Any, attribute_name: str, error: str) -> AttributePrediction:
        return AttributePrediction(
            attribute_name=attribute_name,
            label=VEHICLE_BODY_TYPE_UNKNOWN if attribute_name == "vehicle_body_type" else VEHICLE_COLOUR_UNKNOWN,
            source_backend="base_florence",
            source_model=None,
            source_frame_number=getattr(item, "frame_number", None),
            source_crop_path=str(getattr(item, "vehicle_crop_path", "") or ""),
            raw_response=None,
            confidence=None,
            quality_weight=float(getattr(item, "quality_score", 0.0) or 0.0),
            evidence_role=getattr(item, "evidence_role", None),
            adapter_active=False,
            inference_duration_ms=None,
            original_crop_width=int(getattr(item, "original_crop_width", 0) or 0),
            original_crop_height=int(getattr(item, "original_crop_height", 0) or 0),
            resolution_tier=str(getattr(item, "resolution_tier", "")),
            status=ATTRIBUTE_STATUS_ERROR,
            reason="inference_failed",
            error=error,
        )

    def _prepare_image_for_florence(self, image: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        padded = pil_image
        if self.image_size_policy.florence.pad_to_square and pil_image.width != pil_image.height:
            padded = pad_to_square(
                pil_image,
                fill=(
                    self.image_size_policy.florence.square_padding_value,
                    self.image_size_policy.florence.square_padding_value,
                    self.image_size_policy.florence.square_padding_value,
                ),
            )
        prepared_bgr = cv2.cvtColor(np.array(padded), cv2.COLOR_RGB2BGR)
        return prepared_bgr, {"padded_width": int(padded.width), "padded_height": int(padded.height)}

    def _body_type_eligible(self, vehicle_class: str) -> bool:
        normalized = str(vehicle_class or "UNKNOWN").strip().upper()
        if not self.body_type_enabled:
            return False
        if not self.body_type_car_only:
            return normalized in self.body_type_vehicle_classes
        return normalized == "CAR" and normalized in self.body_type_vehicle_classes

    def _is_body_type_candidate(self, item: Any) -> bool:
        width = int(getattr(item, "original_crop_width", 0) or getattr(item, "crop_width", 0) or 0)
        height = int(getattr(item, "original_crop_height", 0) or getattr(item, "crop_height", 0) or 0)
        if width < self.body_type_minimum_original_width or height < self.body_type_minimum_original_height:
            return False
        if not str(getattr(item, "vehicle_crop_path", "") or ""):
            return False
        if "missing_crop_image" in getattr(item, "rejection_reasons", []):
            return False
        if "empty_crop" in getattr(item, "rejection_reasons", []):
            return False
        if "invalid_bbox" in getattr(item, "rejection_reasons", []):
            return False
        if any(
            reason in getattr(item, "rejection_reasons", [])
            for reason in ("crop_rejected_quality", "brightness_below_minimum", "brightness_above_maximum", "edge_truncation_above_maximum", "sharpness_below_minimum")
        ):
            return False
        return True

    def _select_body_type_items(self, items: list[Any]) -> list[Any]:
        ordered = sorted(
            items,
            key=lambda item: (
                1 if int(getattr(item, "original_crop_width", 0) or 0) >= self.body_type_preferred_original_width and int(getattr(item, "original_crop_height", 0) or 0) >= self.body_type_preferred_original_height else 0,
                float(getattr(item, "quality_score", 0.0)),
                float(getattr(item, "sharpness_score", 0.0)),
                float((getattr(item, "original_crop_width", 0) or 0) * (getattr(item, "original_crop_height", 0) or 0)),
                -float(getattr(item, "clipping_ratio", 0.0)),
                1 if getattr(item, "evidence_source", "") == "capture_zone" else 0,
            ),
            reverse=True,
        )
        selected: list[Any] = []
        for item in ordered:
            if len(selected) >= self.body_type_maximum_crops_per_track:
                break
            if any(abs(int(existing.frame_number) - int(item.frame_number)) < 3 for existing in selected):
                continue
            selected.append(item)
        return selected if selected else ordered[: self.body_type_maximum_crops_per_track]

    def _ensure_crop_row(self, rows: dict[tuple[str, int], dict[str, Any]], request: TrackEnrichmentRequest, item: Any) -> None:
        key = (str(item.vehicle_crop_path), int(item.frame_number))
        if key in rows:
            return
        rows[key] = {
            "camera_id": request.camera_id,
            "local_track_id": request.local_track_id,
            "frame_index": item.frame_number,
            "vehicle_class": request.vehicle_class,
            "vehicle_crop_path": str(item.vehicle_crop_path),
            "crop_quality_score": float(getattr(item, "quality_score", 0.0) or 0.0),
            "colour_task_token": self.colour_task_token,
            "colour_prompt": self.colour_prompt,
            "colour_effective_processor_text": f"{self.colour_task_token}{self.colour_prompt}",
            "colour_raw_response": "",
            "colour_post_processed_response": "",
            "parsed_colour": VEHICLE_COLOUR_UNKNOWN,
            "colour_status": "skipped",
            "colour_reason": None,
            "colour_inference_time_ms": None,
            "body_type_task_token": self.body_type_task_token,
            "body_type_prompt": self.body_type_prompt,
            "body_type_effective_processor_text": f"{self.body_type_task_token}{self.body_type_prompt}",
            "body_type_raw_response": "",
            "body_type_post_processed_response": "",
            "parsed_body_type": VEHICLE_BODY_TYPE_UNKNOWN,
            "body_type_status": ATTRIBUTE_STATUS_DISABLED if not self.body_type_enabled else "skipped",
            "body_type_reason": "disabled" if not self.body_type_enabled else None,
            "body_type_inference_time_ms": None,
            "adapter_loaded": False,
            "crop_source": self._resolve_crop_source(item),
            "crop_available": True,
            "crop_skip_reason": None,
            "selection_tier": str(getattr(item, "colour_selection_tier", "") or ""),
        }
