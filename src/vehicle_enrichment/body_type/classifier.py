from __future__ import annotations

import logging
from pathlib import Path
import re
from typing import Any

import cv2

from ..schemas import (
    ATTRIBUTE_STATUS_DISABLED,
    ATTRIBUTE_STATUS_ERROR,
    AttributePrediction,
    TrackEnrichmentRequest,
    VEHICLE_BODY_TYPE_UNKNOWN,
    VehicleBodyTypeResult,
)
from ..shared import FlorenceBackend


BODY_TYPE_TASK_PROMPT = "<VQA>"
BODY_TYPE_PROMPT_TEXT = (
    "Which is closest: hatchback, sedan, suv, mpv, van, pickup, or other?"
)

UNKNOWN_PHRASES = {
    "",
    "unknown",
    "unclear",
    "not visible",
    "cannot determine",
    "cant determine",
    "unable to determine",
    "cannot classify",
    "not sure",
}

NORMALIZATION_RULES = [
    ("SUV", {"suv", "sport utility vehicle", "sports utility vehicle"}),
    ("SEDAN", {"sedan", "sedan car", "saloon"}),
    ("HATCHBACK", {"hatchback", "hatch back", "back"}),
    ("MPV", {"mpv", "muv", "multi purpose vehicle", "multi-purpose vehicle", "multi utility vehicle"}),
    ("VAN", {"van", "minivan"}),
    ("PICKUP", {"pickup", "pickup truck", "pick up", "pick-up"}),
    ("OTHER", {"other"}),
]


class VehicleBodyTypeClassifier:
    def __init__(self, config: dict[str, Any], *, backend: FlorenceBackend, logger: logging.Logger) -> None:
        self.config = dict(config)
        self.backend = backend
        self.logger = logger
        self.enabled = bool(self.config.get("enabled", False))
        self.allowed_labels = [str(item).strip().upper() for item in self.config.get("allowed_labels", []) if str(item).strip()]
        self.eligible_vehicle_classes = {
            str(item).strip().upper()
            for item in self.config.get("run_only_when_vehicle_class", ["CAR"])
            if str(item).strip()
        }
        self.maximum_crops_per_track = max(1, int(self.config.get("maximum_crops_per_track", 2)))
        self.minimum_crop_width = max(1, int(self.config.get("minimum_crop_width", 180)))
        self.minimum_crop_height = max(1, int(self.config.get("minimum_crop_height", 120)))
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
                reason="vehicle_class_not_eligible",
                model=self.backend.model_identifier,
                adapter_active=self.backend.adapter_active,
                aggregation_reason="vehicle_class_not_eligible",
                task_prompt=BODY_TYPE_TASK_PROMPT,
                prompt_text=BODY_TYPE_PROMPT_TEXT,
            )

        self._metrics["body_type_eligible_tracks"] += 1
        self.logger.info("Body type eligible track: %s", request.local_track_id)
        eligible_evidence = [
            item
            for item in request.evidence_items[: self.maximum_crops_per_track]
            if item.crop_width >= self.minimum_crop_width and item.crop_height >= self.minimum_crop_height
        ]
        if not eligible_evidence:
            self._metrics["body_type_tracks_skipped_small_crop"] += 1
            return VehicleBodyTypeResult(
                label=VEHICLE_BODY_TYPE_UNKNOWN,
                predictions=[],
                status="skipped",
                source="florence2",
                reason="no_eligible_crops",
                model=self.backend.model_identifier,
                adapter_active=self.backend.adapter_active,
                aggregation_reason="no_eligible_crops",
                task_prompt=BODY_TYPE_TASK_PROMPT,
                prompt_text=BODY_TYPE_PROMPT_TEXT,
            )

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
        response = self.backend.run_task(image, BODY_TYPE_TASK_PROMPT, BODY_TYPE_PROMPT_TEXT)
        if response["status"] != "completed":
            self._metrics["body_type_tracks_failed"] += 1
            return self._error_prediction(evidence_item, crop_path, "backend_error", str(response.get("reason")))
        payload = dict(response.get("payload") or {})
        raw_response = self._extract_body_type_text(payload)
        normalized_label, normalization_reason = self.normalize_label(raw_response)
        inference_duration_ms = float(payload.get("inference_duration_ms", 0.0))
        self._metrics["body_type_crop_inference_count"] += 1
        self._metrics["body_type_total_inference_duration_ms"] += inference_duration_ms
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
            status=ATTRIBUTE_STATUS_ERROR,
            reason=reason,
            error=error,
        )

    @staticmethod
    def _extract_body_type_text(payload: dict[str, Any]) -> str:
        parsed = payload.get("parsed_answer")
        if isinstance(parsed, dict):
            answer = parsed.get(BODY_TYPE_TASK_PROMPT)
            if isinstance(answer, str):
                return answer
        return str(payload.get("generated_text") or "")

    def normalize_label(self, raw_value: str) -> tuple[str, str]:
        cleaned = " ".join(str(raw_value or "").strip().lower().replace("_", " ").split())
        cleaned = re.sub(r"[^\w\s-]", " ", cleaned)
        cleaned = " ".join(cleaned.split())
        if cleaned in UNKNOWN_PHRASES:
            return VEHICLE_BODY_TYPE_UNKNOWN, "unknown_phrase"
        exact_matches = [label for label, phrases in NORMALIZATION_RULES if cleaned in phrases]
        if len(exact_matches) == 1:
            return exact_matches[0], "exact_phrase_match"
        tokenized = f" {cleaned} "
        matches: list[str] = []
        for label, phrases in NORMALIZATION_RULES:
            for phrase in phrases:
                if f" {phrase} " in tokenized:
                    matches.append(label)
                    break
        unique_matches = sorted(set(matches))
        if len(unique_matches) == 1:
            return unique_matches[0], "contained_phrase_match"
        if len(unique_matches) > 1:
            return VEHICLE_BODY_TYPE_UNKNOWN, "ambiguous_multiple_labels"
        return VEHICLE_BODY_TYPE_UNKNOWN, "unexpected_output"

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
