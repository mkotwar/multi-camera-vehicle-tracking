from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from ..image_size_policy import ImageSizePolicy, normalize_image_size_policy, pad_to_square
from ..schemas import (
    ATTRIBUTE_STATUS_DISABLED,
    ATTRIBUTE_STATUS_ERROR,
    AttributePrediction,
    TrackEnrichmentRequest,
    VEHICLE_BODY_TYPE_UNKNOWN,
    VehicleBodyTypeResult,
)
from ..shared import FlorenceBackend
from .labels import (
    BODY_TYPE_ALLOWED_LABELS,
    BODY_TYPE_PROMPT_TEXT,
    BODY_TYPE_TASK_PROMPT,
    normalize_body_type_label,
)


class VehicleBodyTypeClassifier:
    def __init__(self, config: dict[str, Any], *, backend: FlorenceBackend, image_size_policy: ImageSizePolicy | None = None, logger: logging.Logger) -> None:
        self.config = dict(config)
        self.backend = backend
        self.image_size_policy = image_size_policy or normalize_image_size_policy(
            {},
            fallback_body_type=self.config,
            fallback_colour={"minimum_crop_width": 256, "minimum_crop_height": 192},
            detection={},
        )
        self.logger = logger
        self.enabled = bool(self.config.get("enabled", False))
        self.allowed_labels = [
            str(item).strip().upper() for item in self.config.get("allowed_labels", BODY_TYPE_ALLOWED_LABELS) if str(item).strip()
        ]
        self.eligible_vehicle_classes = {
            str(item).strip().upper()
            for item in self.config.get("run_only_when_vehicle_class", ["CAR"])
            if str(item).strip()
        }
        self.maximum_crops_per_track = max(1, int(self.config.get("maximum_crops_per_track", 2)))
        self.minimum_crop_width = int(self.image_size_policy.florence.minimum_original_width)
        self.minimum_crop_height = int(self.image_size_policy.florence.minimum_original_height)
        self._metrics: dict[str, Any] = {
            "body_type_eligible_tracks": 0,
            "body_type_ineligible_tracks": 0,
            "body_type_tracks_processed": 0,
            "body_type_tracks_skipped_small_crop": 0,
            "body_type_tracks_unknown": 0,
            "body_type_tracks_failed": 0,
            "body_type_crop_inference_count": 0,
            "body_type_total_inference_duration_ms": 0.0,
            "body_type_average_inference_duration_ms": 0.0,
            "body_type_labels": {},
            "body_type_crops_below_minimum": 0,
            "body_type_crops_acceptable": 0,
            "body_type_crops_preferred": 0,
            "body_type_crops_rejected_quality": 0,
        }

    @property
    def metrics(self) -> dict[str, Any]:
        payload = dict(self._metrics)
        payload["body_type_labels"] = dict(self._metrics["body_type_labels"])
        if self._metrics["body_type_crop_inference_count"] > 0:
            payload["body_type_average_inference_duration_ms"] = float(
                self._metrics["body_type_total_inference_duration_ms"] / self._metrics["body_type_crop_inference_count"]
            )
        return payload

    def classify(self, request: TrackEnrichmentRequest, *args: Any, **kwargs: Any) -> VehicleBodyTypeResult:
        if not self.enabled:
            return self._disabled_result("Vehicle body type inference is disabled.")
        if str(request.vehicle_class).strip().upper() not in self.eligible_vehicle_classes:
            self._metrics["body_type_ineligible_tracks"] += 1
            self.logger.info("Body type skipped: %s -> class %s", request.local_track_id, request.vehicle_class)
            return VehicleBodyTypeResult(
                label=VEHICLE_BODY_TYPE_UNKNOWN,
                predictions=[],
                status="skipped",
                source="florence2",
                reason="non_car_vehicle",
                model=self.backend.model_identifier,
                adapter_active=self.backend.adapter_active,
                aggregation_reason="non_car_vehicle",
                task_prompt=BODY_TYPE_TASK_PROMPT,
                prompt_text=BODY_TYPE_PROMPT_TEXT,
            )

        self._metrics["body_type_eligible_tracks"] += 1
        self.logger.info("Body type eligible track: %s", request.local_track_id)
        eligible_evidence = [
            item
            for item in request.evidence_items
            if self._is_evidence_item_eligible(item)
        ]
        if not eligible_evidence:
            self._metrics["body_type_tracks_skipped_small_crop"] += 1
            return VehicleBodyTypeResult(
                label=VEHICLE_BODY_TYPE_UNKNOWN,
                predictions=[],
                status="skipped",
                source="florence2",
                reason="no_body_type_usable_crop",
                model=self.backend.model_identifier,
                adapter_active=self.backend.adapter_active,
                aggregation_reason=self._no_evidence_reason(request.evidence_items),
                task_prompt=BODY_TYPE_TASK_PROMPT,
                prompt_text=BODY_TYPE_PROMPT_TEXT,
            )

        eligible_evidence = self._select_final_evidence_items(eligible_evidence)
        predictions = [self._infer_single_crop(item) for item in eligible_evidence]
        final_label, aggregation_reason, agreement_score, accumulated_weight = self._aggregate_predictions(predictions)
        self._metrics["body_type_tracks_processed"] += 1
        if final_label == VEHICLE_BODY_TYPE_UNKNOWN:
            self._metrics["body_type_tracks_unknown"] += 1
        else:
            self._metrics["body_type_labels"][final_label] = self._metrics["body_type_labels"].get(final_label, 0) + 1
        self.logger.info("Body type result: %s -> %s", request.local_track_id, final_label)
        return VehicleBodyTypeResult(
            label=final_label,
            predictions=predictions,
            status="completed",
            source="florence2",
            reason=None,
            model=self.backend.model_identifier,
            adapter_active=self.backend.adapter_active,
            aggregation_reason=aggregation_reason,
            agreement_score=agreement_score,
            accumulated_quality_weight=accumulated_weight,
            task_prompt=BODY_TYPE_TASK_PROMPT,
            prompt_text=BODY_TYPE_PROMPT_TEXT,
        )

    def _disabled_result(self, reason: str) -> VehicleBodyTypeResult:
        return VehicleBodyTypeResult(
            label=VEHICLE_BODY_TYPE_UNKNOWN,
            predictions=[],
            status=ATTRIBUTE_STATUS_DISABLED,
            source="body_type.classifier",
            reason=reason,
            model=None,
            adapter_active=None,
            task_prompt=BODY_TYPE_TASK_PROMPT,
            prompt_text=BODY_TYPE_PROMPT_TEXT,
        )

    def _infer_single_crop(self, evidence_item: Any) -> AttributePrediction:
        crop_path = Path(str(evidence_item.vehicle_crop_path)) if evidence_item.vehicle_crop_path else None
        if crop_path is None or not crop_path.exists():
            self._metrics["body_type_tracks_failed"] += 1
            return self._error_prediction(evidence_item, crop_path, "missing_crop_image", "Crop image does not exist.")
        image = cv2.imread(str(crop_path))
        if image is None or image.size == 0:
            self._metrics["body_type_tracks_failed"] += 1
            return self._error_prediction(evidence_item, crop_path, "invalid_crop_image", "Crop image could not be decoded.")
        prepared_image, padding_metadata = self._prepare_image_for_florence(image)
        self.logger.debug(
            "Body type Florence input original_crop_size=%sx%s resolution_tier=%s padded_size=%sx%s",
            evidence_item.original_crop_width,
            evidence_item.original_crop_height,
            evidence_item.resolution_tier,
            padding_metadata["padded_width"],
            padding_metadata["padded_height"],
        )
        response = self.backend.run_task(prepared_image, BODY_TYPE_TASK_PROMPT, BODY_TYPE_PROMPT_TEXT)
        if response["status"] != "completed":
            self._metrics["body_type_tracks_failed"] += 1
            return self._error_prediction(evidence_item, crop_path, "backend_error", str(response.get("reason")))
        payload = dict(response.get("payload") or {})
        raw_response = self._extract_body_type_text(payload)
        normalized_label, normalization_reason = self.normalize_label(raw_response)
        self.logger.debug(
            "Body type raw_response=%s parsed_answer=%s normalized_label=%s normalization_reason=%s",
            raw_response,
            payload.get("parsed_answer"),
            normalized_label,
            normalization_reason,
        )
        inference_duration_ms = float(payload.get("inference_duration_ms", 0.0))
        self._metrics["body_type_crop_inference_count"] += 1
        self._metrics["body_type_total_inference_duration_ms"] += inference_duration_ms
        original_width, original_height = self._original_dimensions(evidence_item)
        return AttributePrediction(
            attribute_name="vehicle_body_type",
            label=normalized_label,
            source_backend="florence2",
            source_model=str(payload.get("model_identifier", self.backend.model_identifier)),
            source_frame_number=evidence_item.frame_number,
            source_crop_path=str(crop_path),
            raw_response=raw_response,
            confidence=None,
            quality_weight=float(evidence_item.quality_score),
            evidence_role=evidence_item.evidence_role,
            adapter_active=bool(payload.get("adapter_active", self.backend.adapter_active)),
            inference_duration_ms=inference_duration_ms,
            original_crop_width=original_width,
            original_crop_height=original_height,
            resolution_tier=str(getattr(evidence_item, "resolution_tier", self.image_size_policy.florence.resolution_tier(original_width, original_height))),
            square_padding_applied=bool(padding_metadata["square_padding_applied"]),
            padded_width=int(padding_metadata["padded_width"]),
            padded_height=int(padding_metadata["padded_height"]),
            florence_input_width=int(padding_metadata["padded_width"]),
            florence_input_height=int(padding_metadata["padded_height"]),
            status="completed",
            reason=normalization_reason,
            error=None,
        )

    def _error_prediction(self, evidence_item: Any, crop_path: Path | None, reason: str, error: str) -> AttributePrediction:
        return AttributePrediction(
            attribute_name="vehicle_body_type",
            label=VEHICLE_BODY_TYPE_UNKNOWN,
            source_backend="florence2",
            source_model=self.backend.model_identifier,
            source_frame_number=evidence_item.frame_number,
            source_crop_path=str(crop_path) if crop_path else None,
            raw_response=None,
            confidence=None,
            quality_weight=float(evidence_item.quality_score),
            evidence_role=evidence_item.evidence_role,
            adapter_active=self.backend.adapter_active,
            inference_duration_ms=None,
            original_crop_width=self._original_dimensions(evidence_item)[0],
            original_crop_height=self._original_dimensions(evidence_item)[1],
            resolution_tier=str(getattr(evidence_item, "resolution_tier", "below_minimum")),
            status=ATTRIBUTE_STATUS_ERROR,
            reason=reason,
            error=error,
        )

    def _is_evidence_item_eligible(self, evidence_item: Any) -> bool:
        original_width, original_height = self._original_dimensions(evidence_item)
        vehicle_class = str(getattr(evidence_item, "vehicle_class", "unknown") or "unknown")
        tier = self.image_size_policy.florence.resolution_tier(original_width, original_height, vehicle_class)
        if tier == "below_minimum":
            self._metrics["body_type_crops_below_minimum"] += 1
            return False
        if getattr(evidence_item, "rejection_reasons", []):
            if any(reason in evidence_item.rejection_reasons for reason in ("crop_rejected_quality", "brightness_below_minimum", "brightness_above_maximum", "edge_truncation_above_maximum", "sharpness_below_minimum")):
                self._metrics["body_type_crops_rejected_quality"] += 1
                return False
        if tier == "acceptable":
            self._metrics["body_type_crops_acceptable"] += 1
        elif tier == "preferred":
            self._metrics["body_type_crops_preferred"] += 1
        return True

    def _select_final_evidence_items(self, eligible_evidence: list[Any]) -> list[Any]:
        ordered = sorted(
            eligible_evidence,
            key=lambda item: (
                1 if getattr(item, "resolution_tier", "") == "preferred" else 0,
                float(getattr(item, "quality_score", 0.0)),
                float(getattr(item, "sharpness_score", 0.0)),
                float(getattr(item, "original_crop_width", 0)),
                float(getattr(item, "original_crop_height", 0)),
                -float(getattr(item, "clipping_ratio", 0.0)),
            ),
            reverse=True,
        )
        selected: list[Any] = []
        for item in ordered:
            if len(selected) >= self.maximum_crops_per_track:
                break
            if self._is_temporally_too_close(selected, item):
                continue
            selected.append(item)
        if not selected:
            return ordered[: self.maximum_crops_per_track]
        return selected

    def _is_temporally_too_close(self, selected: list[Any], candidate: Any) -> bool:
        minimum_gap = 3
        for existing in selected:
            if int(existing.frame_number) == int(candidate.frame_number):
                continue
            if abs(int(existing.frame_number) - int(candidate.frame_number)) < minimum_gap:
                return True
        return False

    def _no_evidence_reason(self, evidence_items: list[Any]) -> str:
        if not evidence_items:
            return "no_track_evidence"
        if all(not self._is_dimension_eligible(item) for item in evidence_items):
            return "all_crops_below_minimum"
        if all("crop_rejected_quality" in getattr(item, "rejection_reasons", []) for item in evidence_items):
            return "all_crops_failed_quality"
        return "no_eligible_track_evidence"

    def _is_dimension_eligible(self, evidence_item: Any) -> bool:
        original_width, original_height = self._original_dimensions(evidence_item)
        vehicle_class = str(getattr(evidence_item, "vehicle_class", "unknown") or "unknown")
        return self.image_size_policy.florence.is_eligible(original_width, original_height, vehicle_class)

    @staticmethod
    def _original_dimensions(evidence_item: Any) -> tuple[int, int]:
        width = int(getattr(evidence_item, "original_crop_width", 0) or getattr(evidence_item, "crop_width", 0) or 0)
        height = int(getattr(evidence_item, "original_crop_height", 0) or getattr(evidence_item, "crop_height", 0) or 0)
        return width, height

    def _prepare_image_for_florence(self, image: Any) -> tuple[Any, dict[str, Any]]:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        padded = pil_image
        square_padding_applied = False
        if self.image_size_policy.florence.pad_to_square and pil_image.width != pil_image.height:
            padded = pad_to_square(
                pil_image,
                fill=(
                    self.image_size_policy.florence.square_padding_value,
                    self.image_size_policy.florence.square_padding_value,
                    self.image_size_policy.florence.square_padding_value,
                ),
            )
            square_padding_applied = True
        prepared_bgr = cv2.cvtColor(np.array(padded), cv2.COLOR_RGB2BGR)
        return prepared_bgr, {
            "square_padding_applied": square_padding_applied,
            "padded_width": int(padded.width),
            "padded_height": int(padded.height),
        }

    @staticmethod
    def _extract_body_type_text(payload: dict[str, Any]) -> str:
        parsed = payload.get("parsed_answer")
        if isinstance(parsed, dict):
            answer = parsed.get(BODY_TYPE_TASK_PROMPT)
            if isinstance(answer, str):
                return answer
            if isinstance(answer, dict):
                for value in answer.values():
                    if isinstance(value, str):
                        return value
            for value in parsed.values():
                if isinstance(value, str):
                    return value
                if isinstance(value, dict):
                    for nested in value.values():
                        if isinstance(nested, str):
                            return nested
        if isinstance(parsed, str):
            return parsed
        for key in (
            "decoded_generated_only_text_skip_special",
            "decoded_generated_only_text",
            "decoded_full_text_skip_special",
            "decoded_full_text",
            "generated_text",
        ):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value
        return str(payload.get("generated_text") or "")

    def normalize_label(self, raw_value: str) -> tuple[str, str]:
        return normalize_body_type_label(raw_value)

    def _aggregate_predictions(self, predictions: list[AttributePrediction]) -> tuple[str, str, float | None, float]:
        valid = [item for item in predictions if item.status == "completed" and item.label not in (None, VEHICLE_BODY_TYPE_UNKNOWN)]
        if not valid:
            return VEHICLE_BODY_TYPE_UNKNOWN, "no_valid_predictions", None, 0.0
        label_weights: dict[str, float] = {}
        for prediction in valid:
            label = str(prediction.label)
            label_weights[label] = label_weights.get(label, 0.0) + float(prediction.quality_weight or 0.0)
        ordered = sorted(label_weights.items(), key=lambda item: (item[1], item[0]), reverse=True)
        top_label, top_weight = ordered[0]
        total_weight = float(sum(label_weights.values()))
        agreement_score = float(top_weight / total_weight) if total_weight > 0.0 else None
        if len(ordered) == 1:
            if top_weight < 0.40:
                return VEHICLE_BODY_TYPE_UNKNOWN, "single_weak_prediction", agreement_score, total_weight
            return top_label, "weighted_agreement", agreement_score, total_weight
        _second_label, second_weight = ordered[1]
        if abs(top_weight - second_weight) <= max(0.05, 0.15 * max(top_weight, second_weight)):
            return VEHICLE_BODY_TYPE_UNKNOWN, "conflicting_high_quality_predictions", agreement_score, total_weight
        if agreement_score is not None and agreement_score >= 0.60:
            return top_label, "weighted_majority", agreement_score, total_weight
        return VEHICLE_BODY_TYPE_UNKNOWN, "insufficient_agreement", agreement_score, total_weight
