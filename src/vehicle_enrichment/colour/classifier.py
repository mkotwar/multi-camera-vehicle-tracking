from __future__ import annotations

import logging
from pathlib import Path
import re
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
    VEHICLE_COLOUR_UNKNOWN,
    VehicleColourResult,
)
from ..shared import FlorenceBackend


COLOUR_TASK_PROMPT = "<VQA>"
COLOUR_PROMPT_TEXT = (
    "What is the dominant exterior colour of this vehicle? "
    "Ignore windows, tyres, lights, number plates, shadows, reflections, "
    "road colour, and background. "
    "Answer with exactly one label: "
    "black, white, grey, silver, red, blue, green, yellow, "
    "orange, brown, beige, purple, or other."
)
COLOUR_FALLBACK_PROMPT_TEXT = (
    "Choose one vehicle colour only: black, white, grey, silver, red, blue, green, yellow, orange, brown, beige, purple, or other."
)

COLOUR_PROMPT_VARIANTS: dict[str, dict[str, str]] = {
    "prompt_a": {
        "id": "prompt_a",
        "task_prompt": "<VQA>",
        "prompt_text": COLOUR_PROMPT_TEXT,
        "description": "current prompt",
    },
    "prompt_b": {
        "id": "prompt_b",
        "task_prompt": "<VQA>",
        "prompt_text": "Vehicle body colour. Answer with one word only: black, white, grey, silver, red, blue, green, yellow, orange, brown, beige, purple, or other.",
        "description": "short direct prompt",
    },
    "prompt_c": {
        "id": "prompt_c",
        "task_prompt": "<VQA>",
        "prompt_text": "Look only at the painted exterior body panels of the vehicle. What is the main colour? Answer with exactly one word from: black, white, grey, silver, red, blue, green, yellow, orange, brown, beige, purple, other.",
        "description": "explicit object focus",
    },
    "prompt_d": {
        "id": "prompt_d",
        "task_prompt": "<VQA>",
        "prompt_text": "Classify the vehicle exterior colour into exactly one category: black, white, grey, silver, red, blue, green, yellow, orange, brown, beige, purple, or other.",
        "description": "classification wording",
    },
    "prompt_e": {
        "id": "prompt_e",
        "task_prompt": "<VQA>",
        "prompt_text": "What colour is the vehicle? One word only.",
        "description": "concise question",
    },
    "prompt_f": {
        "id": "prompt_f",
        "task_prompt": "",
        "prompt_text": "What is the dominant exterior colour of this vehicle? Ignore windows, tyres, lights, number plates, shadows, reflections, road colour, and background. Answer with exactly one label: black, white, grey, silver, red, blue, green, yellow, orange, brown, beige, purple, or other.",
        "description": "no task prefix fallback",
    },
}

DEFAULT_COLOUR_PROMPT_ID = "prompt_c"

COLOUR_LABEL_RULES: list[tuple[str, set[str]]] = [
    ("BLACK", {"black", "jet black", "matte black", "gloss black"}),
    ("WHITE", {"white", "pearl white", "off white", "off-white"}),
    ("GREY", {"grey", "gray", "dark grey", "dark gray", "light grey", "light gray", "metallic grey", "metallic gray"}),
    ("SILVER", {"silver", "metallic silver"}),
    ("RED", {"red", "dark red", "light red", "maroon"}),
    ("BLUE", {"blue", "dark blue", "light blue", "navy blue"}),
    ("GREEN", {"green", "dark green", "light green", "olive green"}),
    ("YELLOW", {"yellow"}),
    ("ORANGE", {"orange"}),
    ("BROWN", {"brown", "dark brown", "light brown"}),
    ("BEIGE", {"beige", "tan", "cream"}),
    ("PURPLE", {"purple", "violet"}),
    ("OTHER", {"other"}),
]

UNKNOWN_PHRASES = {
    "",
    "unknown",
    "unclear",
    "unanswerable",
    "not visible",
    "not sure",
    "cannot determine",
    "cant determine",
    "unable to determine",
    "cannot classify",
    "cannot tell",
    "cant tell",
    "not possible to determine",
}

GENERIC_INVALID_RESPONSES = {
    "qa",
    "q a",
    "yes",
    "no",
    "answer",
    "vehicle",
    "car",
    "colour",
    "color",
    "unanswerable",
    "cannot determine",
    "not visible",
}

UNCERTAIN_MARKERS = {"maybe", "possibly", "probably", "perhaps"}


def get_colour_prompt_variants(*, include_no_task_prefix_variant: bool = True) -> list[dict[str, str]]:
    prompt_ids = ["prompt_a", "prompt_b", "prompt_c", "prompt_d", "prompt_e"]
    if include_no_task_prefix_variant:
        prompt_ids.append("prompt_f")
    return [dict(COLOUR_PROMPT_VARIANTS[prompt_id]) for prompt_id in prompt_ids]


def get_default_colour_prompt_variant() -> dict[str, str]:
    return dict(COLOUR_PROMPT_VARIANTS[DEFAULT_COLOUR_PROMPT_ID])


def get_retry_colour_prompt_variant() -> dict[str, str]:
    return {
        "id": "retry_compact",
        "task_prompt": COLOUR_TASK_PROMPT,
        "prompt_text": COLOUR_FALLBACK_PROMPT_TEXT,
        "description": "single retry compact prompt",
    }


class VehicleColourClassifier:
    def __init__(self, config: dict[str, Any], *, backend: FlorenceBackend, image_size_policy: ImageSizePolicy | None = None, logger: logging.Logger) -> None:
        self.config = dict(config)
        self.backend = backend
        self.image_size_policy = image_size_policy or normalize_image_size_policy(
            {},
            fallback_body_type={"minimum_crop_width": 256, "minimum_crop_height": 192},
            fallback_colour=self.config,
            detection={},
        )
        self.logger = logger
        self.enabled = bool(self.config.get("enabled", False))
        self.allowed_labels = [str(item).strip().upper() for item in self.config.get("allowed_labels", []) if str(item).strip()]
        self.eligible_vehicle_classes = {
            str(item).strip().upper()
            for item in self.config.get("run_only_when_vehicle_class", ["3WHEELER", "BUS", "CAR", "MOTORCYCLE", "TRUCK"])
            if str(item).strip()
        }
        self.maximum_crops_per_track = max(1, int(self.config.get("maximum_crops_per_track", 2)))
        self.minimum_crop_width = int(self.image_size_policy.florence.minimum_original_width)
        self.minimum_crop_height = int(self.image_size_policy.florence.minimum_original_height)
        self.retry_on_invalid_response = bool(self.config.get("retry_on_invalid_response", False))
        self.maximum_prompt_attempts = max(1, min(2, int(self.config.get("maximum_prompt_attempts", 2))))
        self.primary_prompt_variant = self._resolve_prompt_variant(str(self.config.get("prompt_variant", DEFAULT_COLOUR_PROMPT_ID)).strip() or DEFAULT_COLOUR_PROMPT_ID)
        self.retry_prompt_variant = get_retry_colour_prompt_variant()
        self._metrics: dict[str, Any] = {
            "colour_eligible_tracks": 0,
            "colour_ineligible_tracks": 0,
            "colour_tracks_processed": 0,
            "colour_tracks_skipped_small_crop": 0,
            "colour_tracks_unknown": 0,
            "colour_tracks_failed": 0,
            "colour_crop_inference_count": 0,
            "colour_total_inference_duration_ms": 0.0,
            "colour_average_inference_duration_ms": 0.0,
            "colour_labels": {},
            "colour_unknown_reasons": {},
            "colour_conflicting_predictions": 0,
            "colour_no_valid_predictions": 0,
            "colour_prompt_attempts": 0,
            "colour_retry_count": 0,
            "colour_retry_success_count": 0,
            "colour_generic_response_count": 0,
            "colour_invalid_response_count": 0,
            "colour_prompt_variant": self.primary_prompt_variant["id"],
            "colour_raw_response_counts": {},
            "colour_crops_below_minimum": 0,
            "colour_crops_acceptable": 0,
            "colour_crops_preferred": 0,
            "colour_crops_rejected_quality": 0,
        }

    @property
    def metrics(self) -> dict[str, Any]:
        payload = dict(self._metrics)
        payload["colour_labels"] = dict(self._metrics["colour_labels"])
        payload["colour_unknown_reasons"] = dict(self._metrics["colour_unknown_reasons"])
        payload["colour_raw_response_counts"] = dict(self._metrics["colour_raw_response_counts"])
        if self._metrics["colour_crop_inference_count"] > 0:
            payload["colour_average_inference_duration_ms"] = float(
                self._metrics["colour_total_inference_duration_ms"] / self._metrics["colour_crop_inference_count"]
            )
        return payload

    def classify(self, request: TrackEnrichmentRequest, *args: Any, **kwargs: Any) -> VehicleColourResult:
        if not self.enabled:
            return self._disabled_result("Vehicle colour inference is disabled.")
        if str(request.vehicle_class).strip().upper() == "UNKNOWN":
            self._metrics["colour_ineligible_tracks"] += 1
            self.logger.info("Colour skipped: %s -> final class UNKNOWN", request.local_track_id)
            return self._skipped_result("vehicle_class_unknown")
        if str(request.vehicle_class).strip().upper() not in self.eligible_vehicle_classes:
            self._metrics["colour_ineligible_tracks"] += 1
            self.logger.info("Colour skipped: %s -> class %s", request.local_track_id, request.vehicle_class)
            return self._skipped_result("vehicle_class_not_eligible")

        self._metrics["colour_eligible_tracks"] += 1
        self.logger.info("Colour eligible track: %s", request.local_track_id)
        eligible_evidence = [
            item
            for item in request.evidence_items
            if self._is_evidence_item_eligible(item)
        ]
        if not eligible_evidence:
            self._metrics["colour_tracks_skipped_small_crop"] += 1
            self.logger.info("Colour skipped: %s -> crop below minimum size", request.local_track_id)
            skipped = self._skipped_result("no_eligible_crops")
            skipped.aggregation_reason = self._no_evidence_reason(request.evidence_items)
            return skipped

        eligible_evidence = self._select_final_evidence_items(eligible_evidence)
        predictions = [self._infer_single_crop(item) for item in eligible_evidence]
        final_label, aggregation_reason, agreement_score, accumulated_weight = self._aggregate_predictions(predictions)
        self._metrics["colour_tracks_processed"] += 1
        if final_label == VEHICLE_COLOUR_UNKNOWN:
            self._metrics["colour_tracks_unknown"] += 1
            self._metrics["colour_unknown_reasons"][aggregation_reason] = self._metrics["colour_unknown_reasons"].get(aggregation_reason, 0) + 1
            if aggregation_reason == "conflicting_high_quality_predictions":
                self._metrics["colour_conflicting_predictions"] += 1
            if aggregation_reason == "no_valid_predictions":
                self._metrics["colour_no_valid_predictions"] += 1
            self.logger.info("Colour result: %s -> UNKNOWN reason=%s", request.local_track_id, aggregation_reason)
        else:
            self._metrics["colour_labels"][final_label] = self._metrics["colour_labels"].get(final_label, 0) + 1
            self.logger.info("Colour result: %s -> %s", request.local_track_id, final_label)
        return VehicleColourResult(
            label=final_label,
            predictions=predictions,
            status="completed",
            source="florence2",
            reason=None if final_label != VEHICLE_COLOUR_UNKNOWN else aggregation_reason,
            model=self.backend.model_identifier,
            adapter_active=self.backend.adapter_active,
            aggregation_reason=aggregation_reason,
            agreement_score=agreement_score,
            accumulated_quality_weight=accumulated_weight,
            task_prompt=str(self.primary_prompt_variant["task_prompt"]),
            prompt_text=str(self.primary_prompt_variant["prompt_text"]),
        )

    def _disabled_result(self, reason: str) -> VehicleColourResult:
        return VehicleColourResult(
            label=VEHICLE_COLOUR_UNKNOWN,
            predictions=[],
            status=ATTRIBUTE_STATUS_DISABLED,
            source="colour.classifier",
            reason=reason,
            model=None,
            adapter_active=None,
            task_prompt=str(self.primary_prompt_variant["task_prompt"]),
            prompt_text=str(self.primary_prompt_variant["prompt_text"]),
        )

    def _skipped_result(self, reason: str) -> VehicleColourResult:
        return VehicleColourResult(
            label=VEHICLE_COLOUR_UNKNOWN,
            predictions=[],
            status="skipped",
            source="florence2",
            reason=reason,
            model=self.backend.model_identifier,
            adapter_active=self.backend.adapter_active,
            aggregation_reason=reason,
            task_prompt=str(self.primary_prompt_variant["task_prompt"]),
            prompt_text=str(self.primary_prompt_variant["prompt_text"]),
        )

    def _infer_single_crop(self, evidence_item: Any) -> AttributePrediction:
        crop_path = Path(str(evidence_item.vehicle_crop_path)) if evidence_item.vehicle_crop_path else None
        if crop_path is None or not crop_path.exists():
            self._metrics["colour_tracks_failed"] += 1
            return self._error_prediction(evidence_item, crop_path, "missing_crop_image", "Crop image does not exist.")
        image = cv2.imread(str(crop_path))
        if image is None or image.size == 0:
            self._metrics["colour_tracks_failed"] += 1
            return self._error_prediction(evidence_item, crop_path, "invalid_crop_image", "Crop image could not be decoded.")

        prepared_image, padding_metadata = self._prepare_image_for_florence(image)
        self.logger.debug(
            "Colour Florence input original_crop_size=%sx%s resolution_tier=%s padded_size=%sx%s",
            evidence_item.original_crop_width,
            evidence_item.original_crop_height,
            evidence_item.resolution_tier,
            padding_metadata["padded_width"],
            padding_metadata["padded_height"],
        )
        attempts = self._run_prompt_attempts(prepared_image)
        successful_attempts = [attempt for attempt in attempts if attempt["status"] == "completed"]
        if not successful_attempts:
            self._metrics["colour_tracks_failed"] += 1
            error_attempt = attempts[-1] if attempts else {"reason": "backend_error", "error": "Unknown Florence failure."}
            return self._error_prediction(
                evidence_item,
                crop_path,
                str(error_attempt.get("reason") or "backend_error"),
                str(error_attempt.get("error") or "Unknown Florence failure."),
            )

        selected_attempt = self._select_best_attempt(successful_attempts)
        if len(successful_attempts) > 1 and selected_attempt["normalized_label"] != VEHICLE_COLOUR_UNKNOWN:
            self._metrics["colour_retry_success_count"] += 1

        raw_response_payload: Any
        if len(successful_attempts) == 1:
            raw_response_payload = successful_attempts[0]["raw_response"]
        else:
            raw_response_payload = [
                {
                    "prompt_id": attempt["prompt_id"],
                    "task_prompt": attempt["task_prompt"],
                    "prompt_text": attempt["prompt_text"],
                    "raw_response": attempt["raw_response"],
                    "normalized_label": attempt["normalized_label"],
                    "normalization_reason": attempt["normalization_reason"],
                }
                for attempt in successful_attempts
            ]
        original_width, original_height = self._original_dimensions(evidence_item)

        return AttributePrediction(
            attribute_name="vehicle_colour",
            label=str(selected_attempt["normalized_label"]),
            source_backend="florence2",
            source_model=str(selected_attempt["source_model"]),
            source_frame_number=evidence_item.frame_number,
            source_crop_path=str(crop_path),
            raw_response=raw_response_payload,
            confidence=None,
            quality_weight=float(evidence_item.quality_score),
            evidence_role=evidence_item.evidence_role,
            adapter_active=bool(selected_attempt["adapter_active"]),
            inference_duration_ms=float(selected_attempt["inference_duration_ms"]),
            original_crop_width=original_width,
            original_crop_height=original_height,
            resolution_tier=str(getattr(evidence_item, "resolution_tier", self.image_size_policy.florence.resolution_tier(original_width, original_height))),
            square_padding_applied=bool(padding_metadata["square_padding_applied"]),
            padded_width=int(padding_metadata["padded_width"]),
            padded_height=int(padding_metadata["padded_height"]),
            florence_input_width=int(padding_metadata["padded_width"]),
            florence_input_height=int(padding_metadata["padded_height"]),
            status="completed",
            reason=str(selected_attempt["normalization_reason"]),
            error=None,
        )

    def _run_prompt_attempts(self, image: Any) -> list[dict[str, Any]]:
        attempts: list[dict[str, Any]] = []
        prompt_variants = [self.primary_prompt_variant]
        if self.retry_on_invalid_response and self.maximum_prompt_attempts > 1:
            prompt_variants.append(self.retry_prompt_variant)

        for index, prompt_variant in enumerate(prompt_variants):
            if index > 0:
                self._metrics["colour_retry_count"] += 1
            attempt = self._run_single_prompt_attempt(image, prompt_variant)
            attempts.append(attempt)
            if attempt["status"] != "completed":
                continue
            if not self._should_retry_attempt(attempt):
                break
            if index + 1 >= self.maximum_prompt_attempts:
                break
        return attempts

    def _run_single_prompt_attempt(self, image: Any, prompt_variant: dict[str, str]) -> dict[str, Any]:
        self._metrics["colour_prompt_attempts"] += 1
        response = self.backend.run_task(image, str(prompt_variant["task_prompt"]), str(prompt_variant["prompt_text"]))
        if response["status"] != "completed":
            return {
                "status": "error",
                "prompt_id": str(prompt_variant["id"]),
                "task_prompt": str(prompt_variant["task_prompt"]),
                "prompt_text": str(prompt_variant["prompt_text"]),
                "reason": "backend_error",
                "error": str(response.get("reason")),
            }
        payload = dict(response.get("payload") or {})
        raw_response = self._extract_colour_text(payload)
        normalized_label, normalization_reason = self.normalize_label(raw_response)
        response_kind = self.response_kind(raw_response, normalization_reason)
        cleaned_raw = self._clean_text(raw_response)
        if cleaned_raw:
            self._metrics["colour_raw_response_counts"][cleaned_raw] = self._metrics["colour_raw_response_counts"].get(cleaned_raw, 0) + 1
        if response_kind == "generic_invalid":
            self._metrics["colour_generic_response_count"] += 1
        if normalized_label == VEHICLE_COLOUR_UNKNOWN:
            self._metrics["colour_invalid_response_count"] += 1
        inference_duration_ms = float(payload.get("inference_duration_ms", 0.0))
        self._metrics["colour_crop_inference_count"] += 1
        self._metrics["colour_total_inference_duration_ms"] += inference_duration_ms
        self.logger.debug(
            "Colour prompt_id=%s raw_response=%s normalized_label=%s normalization_reason=%s",
            prompt_variant["id"],
            raw_response,
            normalized_label,
            normalization_reason,
        )
        return {
            "status": "completed",
            "prompt_id": str(prompt_variant["id"]),
            "task_prompt": str(prompt_variant["task_prompt"]),
            "prompt_text": str(prompt_variant["prompt_text"]),
            "raw_response": raw_response,
            "normalized_label": normalized_label,
            "normalization_reason": normalization_reason,
            "response_kind": response_kind,
            "source_model": str(payload.get("model_identifier", self.backend.model_identifier)),
            "adapter_active": bool(payload.get("adapter_active", self.backend.adapter_active)),
            "inference_duration_ms": inference_duration_ms,
            "payload": payload,
        }

    def _select_best_attempt(self, attempts: list[dict[str, Any]]) -> dict[str, Any]:
        valid = [attempt for attempt in attempts if attempt["normalized_label"] != VEHICLE_COLOUR_UNKNOWN]
        if valid:
            return valid[-1]
        return attempts[-1]

    def _should_retry_attempt(self, attempt: dict[str, Any]) -> bool:
        return attempt["status"] == "completed" and attempt["normalized_label"] == VEHICLE_COLOUR_UNKNOWN

    def _error_prediction(self, evidence_item: Any, crop_path: Path | None, reason: str, error: str) -> AttributePrediction:
        return AttributePrediction(
            attribute_name="vehicle_colour",
            label=VEHICLE_COLOUR_UNKNOWN,
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
        tier = self.image_size_policy.florence.resolution_tier(original_width, original_height)
        if tier == "below_minimum":
            self._metrics["colour_crops_below_minimum"] += 1
            return False
        if getattr(evidence_item, "rejection_reasons", []):
            if any(reason in evidence_item.rejection_reasons for reason in ("crop_rejected_quality", "brightness_below_minimum", "brightness_above_maximum", "edge_truncation_above_maximum", "sharpness_below_minimum")):
                self._metrics["colour_crops_rejected_quality"] += 1
                return False
        if tier == "acceptable":
            self._metrics["colour_crops_acceptable"] += 1
        elif tier == "preferred":
            self._metrics["colour_crops_preferred"] += 1
        return True

    def _select_final_evidence_items(self, eligible_evidence: list[Any]) -> list[Any]:
        def brightness_center_score(item: Any) -> float:
            brightness = float(getattr(item, "brightness_score", 0.0))
            return -abs(brightness - 140.0)

        ordered = sorted(
            eligible_evidence,
            key=lambda item: (
                1 if getattr(item, "resolution_tier", "") == "preferred" else 0,
                float(getattr(item, "quality_score", 0.0)),
                brightness_center_score(item),
                float(getattr(item, "sharpness_score", 0.0)),
                -float(getattr(item, "clipping_ratio", 0.0)),
                float(getattr(item, "original_crop_width", 0)),
                float(getattr(item, "original_crop_height", 0)),
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
        return self.image_size_policy.florence.is_eligible(original_width, original_height)

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
    def _extract_colour_text(payload: dict[str, Any]) -> str:
        parsed = payload.get("parsed_answer")
        prompt_keys = {COLOUR_TASK_PROMPT, "", None}
        if isinstance(parsed, dict):
            for key in prompt_keys:
                answer = parsed.get(key)
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

    @classmethod
    def normalize_label(cls, raw_value: str) -> tuple[str, str]:
        cleaned = cls._clean_text(raw_value)
        if cleaned in UNKNOWN_PHRASES:
            return VEHICLE_COLOUR_UNKNOWN, "unknown_phrase"
        if cleaned in GENERIC_INVALID_RESPONSES:
            return VEHICLE_COLOUR_UNKNOWN, "generic_invalid_response"
        if any(re.search(rf"\b{re.escape(marker)}\b", cleaned) for marker in UNCERTAIN_MARKERS):
            return VEHICLE_COLOUR_UNKNOWN, "uncertain_response"

        exact_matches = [label for label, phrases in COLOUR_LABEL_RULES if cleaned in phrases]
        unique_exact = sorted(set(exact_matches))
        if len(unique_exact) == 1:
            return unique_exact[0], "exact_phrase_match"
        if len(unique_exact) > 1:
            return VEHICLE_COLOUR_UNKNOWN, "ambiguous_multiple_labels"

        matches: list[str] = []
        for label, phrases in COLOUR_LABEL_RULES:
            for phrase in phrases:
                if re.search(rf"\b{re.escape(phrase)}\b", cleaned):
                    matches.append(label)
                    break
        unique_matches = sorted(set(matches))
        if len(unique_matches) > 1:
            return VEHICLE_COLOUR_UNKNOWN, "ambiguous_multiple_labels"
        if len(unique_matches) == 1:
            return unique_matches[0], "contained_phrase_match"
        return VEHICLE_COLOUR_UNKNOWN, "unexpected_output"

    @classmethod
    def response_kind(cls, raw_value: str, normalization_reason: str | None = None) -> str:
        cleaned = cls._clean_text(raw_value)
        if cleaned in GENERIC_INVALID_RESPONSES or normalization_reason == "generic_invalid_response":
            return "generic_invalid"
        if normalization_reason == "ambiguous_multiple_labels":
            return "ambiguous"
        if normalization_reason in {"exact_phrase_match", "contained_phrase_match"}:
            return "valid"
        if normalization_reason == "unknown_phrase":
            return "unknown_phrase"
        return "invalid"

    @staticmethod
    def _clean_text(raw_value: str) -> str:
        cleaned = " ".join(str(raw_value or "").strip().lower().replace("_", " ").split())
        cleaned = re.sub(r"[^\w\s-]", " ", cleaned)
        return " ".join(cleaned.split())

    def _resolve_prompt_variant(self, prompt_id: str) -> dict[str, str]:
        if prompt_id not in COLOUR_PROMPT_VARIANTS:
            return get_default_colour_prompt_variant()
        return dict(COLOUR_PROMPT_VARIANTS[prompt_id])

    def _aggregate_predictions(self, predictions: list[AttributePrediction]) -> tuple[str, str, float | None, float]:
        valid = [item for item in predictions if item.status == "completed" and item.label not in (None, VEHICLE_COLOUR_UNKNOWN)]
        if not valid:
            return VEHICLE_COLOUR_UNKNOWN, "no_valid_predictions", None, 0.0
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
                return VEHICLE_COLOUR_UNKNOWN, "single_weak_prediction", agreement_score, total_weight
            return top_label, "weighted_agreement", agreement_score, total_weight
        _second_label, second_weight = ordered[1]
        if abs(top_weight - second_weight) <= max(0.05, 0.15 * max(top_weight, second_weight)):
            return VEHICLE_COLOUR_UNKNOWN, "conflicting_high_quality_predictions", agreement_score, total_weight
        if agreement_score is not None and agreement_score >= 0.60:
            return top_label, "weighted_majority", agreement_score, total_weight
        return VEHICLE_COLOUR_UNKNOWN, "insufficient_agreement", agreement_score, total_weight
