from __future__ import annotations

import logging
import shutil
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2

from src.models import LocalTrack, TrackEvidence
from src.output_writer import RunOutputManager

from .attribute_aggregator import AttributeAggregator
from .body_type import VehicleBodyTypeClassifier
from .colour import VehicleColourClassifier
from .evidence_adapter import EvidenceAdapter
from .evidence_quality import EvidenceQualityEvaluator, normalize_quality_config
from .make_model import VehicleMakeModelClassifier
from .plate import (
    PlateColourClassifier,
    PlateDetector,
    PlateOCREngine,
    PlateQualityValidator,
    PlateResultAggregator,
)
from .schemas import (
    ATTRIBUTE_STATUS_DISABLED,
    ENRICHMENT_STATUS_COMPLETED,
    ENRICHMENT_STATUS_DISABLED,
    ENRICHMENT_STATUS_ERROR,
    ENRICHMENT_STATUS_EVIDENCE_READY,
    ENRICHMENT_STATUS_NO_EVIDENCE,
    TrackEnrichmentRequest,
    TrackEnrichmentResult,
    VehicleBodyTypeResult,
    VehicleColourResult,
)
from .shared import FlorenceBackend, FlorenceBackendConfig


def normalize_vehicle_enrichment_config(raw_section: Any) -> dict[str, Any]:
    section = dict(raw_section or {})
    evidence = dict(section.get("evidence", {}) or {})
    scoring = dict(evidence.get("scoring", {}) or {})
    shared_florence = dict(section.get("shared_florence", {}) or {})
    body_type = dict(section.get("body_type", {}) or {})
    colour = dict(section.get("colour", {}) or {})
    make_model = dict(section.get("make_model", {}) or {})
    plate = dict(section.get("plate", {}) or {})
    ocr = dict(section.get("ocr", {}) or {})
    return {
        "enabled": bool(section.get("enabled", False)),
        "trigger": str(section.get("trigger", "track_completed")).strip() or "track_completed",
        "fail_open": bool(section.get("fail_open", True)),
        "best_crops_per_track": max(1, int(section.get("best_crops_per_track", 3))),
        "write_separate_output": bool(section.get("write_separate_output", True)),
        "extend_tracks_json": bool(section.get("extend_tracks_json", True)),
        "evidence": {
            "source": str(evidence.get("source", "existing_track_evidence")).strip() or "existing_track_evidence",
            "save_vehicle_crops": bool(evidence.get("save_vehicle_crops", True)),
            "minimum_crop_width": max(1, int(evidence.get("minimum_crop_width", 100))),
            "minimum_crop_height": max(1, int(evidence.get("minimum_crop_height", 70))),
            "minimum_sharpness": float(evidence.get("minimum_sharpness", 10.0)),
            "minimum_quality_score": float(evidence.get("minimum_quality_score", 0.20)),
            "border_margin_ratio": float(evidence.get("border_margin_ratio", 0.02)),
            "scoring": {
                "area_weight": float(scoring.get("area_weight", 0.25)),
                "sharpness_weight": float(scoring.get("sharpness_weight", 0.25)),
                "confidence_weight": float(scoring.get("confidence_weight", 0.20)),
                "role_weight": float(scoring.get("role_weight", 0.15)),
                "border_weight": float(scoring.get("border_weight", 0.05)),
                "clipping_weight": float(scoring.get("clipping_weight", 0.05)),
                "brightness_weight": float(scoring.get("brightness_weight", 0.05)),
            },
        },
        "shared_florence": {
            "enabled": bool(shared_florence.get("enabled", False)),
            "backend": str(shared_florence.get("backend", "florence2")).strip() or "florence2",
            "base_model_id": str(shared_florence.get("base_model_id", "microsoft/Florence-2-base-ft")).strip(),
            "adapter_path": str(shared_florence.get("adapter_path", "model_weights/florence/adaptor_florance_baseFT")).strip(),
            "adapter_enabled": bool(shared_florence.get("adapter_enabled", False)),
            "device": str(shared_florence.get("device", "auto")).strip() or "auto",
            "dtype": str(shared_florence.get("dtype", "auto")).strip() or "auto",
            "trust_remote_code": bool(shared_florence.get("trust_remote_code", True)),
            "attention_implementation": str(shared_florence.get("attention_implementation", "eager")).strip() or "eager",
            "max_new_tokens": int(shared_florence.get("max_new_tokens", 128)),
            "num_beams": int(shared_florence.get("num_beams", 3)),
            "use_cache": bool(shared_florence.get("use_cache", False)),
            "lazy_load": bool(shared_florence.get("lazy_load", True)),
        },
        "body_type": {
            "enabled": bool(body_type.get("enabled", False)),
            "backend": str(body_type.get("backend", "florence2")).strip() or "florence2",
            "run_only_when_vehicle_class": [
                str(item).strip().upper()
                for item in body_type.get("run_only_when_vehicle_class", ["CAR"])
                if str(item).strip()
            ],
            "maximum_crops_per_track": max(1, int(body_type.get("maximum_crops_per_track", 2))),
            "minimum_crop_width": max(1, int(body_type.get("minimum_crop_width", 180))),
            "minimum_crop_height": max(1, int(body_type.get("minimum_crop_height", 120))),
            "allowed_labels": [str(item).strip().upper() for item in body_type.get("allowed_labels", []) if str(item).strip()],
        },
        "colour": {"enabled": bool(colour.get("enabled", False))},
        "make_model": {"enabled": bool(make_model.get("enabled", False))},
        "plate": {
            "detection_enabled": bool(plate.get("detection_enabled", False)),
            "colour_enabled": bool(plate.get("colour_enabled", False)),
        },
        "ocr": {
            "enabled": bool(ocr.get("enabled", False)),
            "run_only_when_plate_detected": bool(ocr.get("run_only_when_plate_detected", True)),
        },
    }


class VehicleEnrichmentManager:
    def __init__(self, config: dict[str, Any], logger: logging.Logger, output_manager: RunOutputManager) -> None:
        self.logger = logger
        self.output_manager = output_manager
        self.config = normalize_vehicle_enrichment_config(config.get("vehicle_enrichment", {}))
        self.enabled = bool(self.config["enabled"])
        self.adapter = EvidenceAdapter(self.config, output_manager, logger)
        self.quality_evaluator = EvidenceQualityEvaluator(normalize_quality_config(self.config))
        self.attribute_aggregator = AttributeAggregator()
        self.florence_backend = FlorenceBackend(FlorenceBackendConfig(**self.config["shared_florence"]), logger=logger)
        self.body_type_classifier = VehicleBodyTypeClassifier(self.config["body_type"], backend=self.florence_backend, logger=logger)
        self.colour_classifier = VehicleColourClassifier(self.config["colour"])
        self.make_model_classifier = VehicleMakeModelClassifier(self.config["make_model"])
        self.plate_detector = PlateDetector(self.config["plate"])
        self.plate_quality_validator = PlateQualityValidator(self.config["plate"])
        self.plate_colour_classifier = PlateColourClassifier(self.config["plate"])
        self.plate_ocr_engine = PlateOCREngine(self.config["ocr"])
        self.plate_result_aggregator = PlateResultAggregator(self.config["plate"])
        self._results_by_track: dict[str, TrackEnrichmentResult] = {}
        self._metrics: dict[str, Any] = {
            "completed_tracks_received": 0,
            "tracks_with_existing_evidence": 0,
            "tracks_without_evidence": 0,
            "evidence_items_received": 0,
            "crop_candidates_evaluated": 0,
            "crop_candidates_rejected": 0,
            "crops_retained": 0,
            "crops_rejected_by_reason": {},
            "enrichment_results_written": 0,
            "enrichment_failures": 0,
            "cleanup_count": 0,
            "current_in_memory_tracks": 0,
        }

    @property
    def metrics(self) -> dict[str, Any]:
        payload = dict(self._metrics)
        payload["crops_rejected_by_reason"] = dict(self._metrics["crops_rejected_by_reason"])
        payload.update(self.florence_backend.metrics)
        payload.update(self.body_type_classifier.metrics)
        return payload

    @property
    def results_by_track(self) -> dict[str, TrackEnrichmentResult]:
        return dict(self._results_by_track)

    def enrich_completed_tracks(
        self,
        tracks: list[LocalTrack],
        finalized_evidence_records: list[TrackEvidence | dict[str, Any]],
    ) -> list[TrackEnrichmentResult]:
        grouped_records: dict[str, list[TrackEvidence | dict[str, Any]]] = defaultdict(list)
        for record in finalized_evidence_records:
            local_track_id = record["local_track_id"] if isinstance(record, dict) else record.local_track_id
            grouped_records[str(local_track_id)].append(record)

        results: list[TrackEnrichmentResult] = []
        for track in tracks:
            self._metrics["completed_tracks_received"] += 1
            self._metrics["current_in_memory_tracks"] += 1
            started_monotonic = time.perf_counter()
            started_iso = datetime.now(timezone.utc).isoformat()
            try:
                result = self._enrich_single_track(
                    track,
                    grouped_records.get(track.local_track_id, []),
                    started_iso,
                    started_monotonic,
                )
            except Exception as exc:
                self._metrics["enrichment_failures"] += 1
                if not self.config["fail_open"]:
                    raise
                result = self._build_error_result(track, started_iso, started_monotonic, exc)
            self._results_by_track[track.local_track_id] = result
            self._metrics["enrichment_results_written"] += 1
            results.append(result)
            self._cleanup_track(track.local_track_id)
        return results

    def _enrich_single_track(
        self,
        track: LocalTrack,
        evidence_records: list[TrackEvidence | dict[str, Any]],
        started_iso: str,
        started_monotonic: float,
    ) -> TrackEnrichmentResult:
        vehicle_class_confidence = self._compute_vehicle_class_confidence(track)
        if not self.enabled:
            return self._build_base_result(
                track=track,
                vehicle_class_confidence=vehicle_class_confidence,
                status=ENRICHMENT_STATUS_DISABLED,
                evidence_used=[],
                started_iso=started_iso,
                started_monotonic=started_monotonic,
                errors=[],
            )

        adapted = self.adapter.adapt_track(track, evidence_records)
        self._metrics["evidence_items_received"] += len(adapted)
        if adapted:
            self._metrics["tracks_with_existing_evidence"] += 1
        else:
            self._metrics["tracks_without_evidence"] += 1
            return self._build_base_result(
                track=track,
                vehicle_class_confidence=vehicle_class_confidence,
                status=ENRICHMENT_STATUS_NO_EVIDENCE,
                evidence_used=[],
                started_iso=started_iso,
                started_monotonic=started_monotonic,
                errors=[],
            )

        scored = [self.quality_evaluator.score_item(item) for item in adapted]
        self._metrics["crop_candidates_evaluated"] += len(scored)
        selected, rejected = self._select_best_evidence(scored)
        for rejected_item in rejected:
            self._metrics["crop_candidates_rejected"] += 1
            for reason in rejected_item.rejection_reasons:
                self._metrics["crops_rejected_by_reason"][reason] = self._metrics["crops_rejected_by_reason"].get(reason, 0) + 1
        self._metrics["crops_retained"] += len(selected)
        selected = [self._materialize_selected_crop(track, item) for item in selected]

        request = TrackEnrichmentRequest(
            local_track_id=track.local_track_id,
            camera_id=track.camera_id,
            native_tracker_id=track.native_tracker_id,
            vehicle_class=str(track.final_class or "UNKNOWN").upper(),
            vehicle_class_confidence=vehicle_class_confidence,
            track_status=track.status,
            completion_reason=track.completion_reason,
            started_at_seconds=float(track.first_timestamp_seconds),
            ended_at_seconds=float(track.last_timestamp_seconds),
            evidence_items=selected,
        )

        body_type_result = self.body_type_classifier.classify(request)
        colour_result = self.colour_classifier.classify(request)
        make_model_result = self.make_model_classifier.classify(request)
        plate_detection_result = self.plate_detector.detect(request)
        overall_status = ENRICHMENT_STATUS_COMPLETED if self.body_type_classifier.enabled else ENRICHMENT_STATUS_EVIDENCE_READY

        return self._build_base_result(
            track=track,
            vehicle_class_confidence=vehicle_class_confidence,
            status=overall_status if selected else ENRICHMENT_STATUS_NO_EVIDENCE,
            evidence_used=selected,
            started_iso=started_iso,
            started_monotonic=started_monotonic,
            errors=[],
            body_type_result=body_type_result,
            colour_result=colour_result,
            vehicle_make=make_model_result.make,
            vehicle_model=make_model_result.model,
            plate_detected=plate_detection_result.detected,
        )

    def _build_base_result(
        self,
        *,
        track: LocalTrack,
        vehicle_class_confidence: float | None,
        status: str,
        evidence_used: list[Any],
        started_iso: str,
        started_monotonic: float,
        errors: list[str],
        body_type_result: VehicleBodyTypeResult | None = None,
        colour_result: VehicleColourResult | None = None,
        vehicle_make: str | None = None,
        vehicle_model: str | None = None,
        plate_detected: bool = False,
    ) -> TrackEnrichmentResult:
        completed_iso = datetime.now(timezone.utc).isoformat()
        return TrackEnrichmentResult(
            local_track_id=track.local_track_id,
            camera_id=track.camera_id,
            vehicle_class=str(track.final_class or "UNKNOWN").upper(),
            vehicle_class_confidence=vehicle_class_confidence,
            vehicle_body_type=body_type_result
            or VehicleBodyTypeResult(
                label="UNKNOWN",
                predictions=[],
                status=ATTRIBUTE_STATUS_DISABLED,
                source="body_type.classifier",
                reason="Vehicle body type inference is disabled.",
            ),
            vehicle_colour=colour_result
            or VehicleColourResult(
                label="UNKNOWN",
                predictions=[],
                status=ATTRIBUTE_STATUS_DISABLED,
                source="colour.classifier",
                reason="Vehicle colour inference is disabled.",
            ),
            vehicle_make=vehicle_make,
            vehicle_model=vehicle_model,
            plate_detected=plate_detected,
            plate_colour=None,
            registration_category=None,
            plate_text=None,
            status=status,
            evidence_used=evidence_used,
            predictions=[],
            errors=errors,
            processing_started_at=started_iso,
            processing_completed_at=completed_iso,
            processing_duration_ms=float((time.perf_counter() - started_monotonic) * 1000.0),
        )

    def _build_error_result(
        self,
        track: LocalTrack,
        started_iso: str,
        started_monotonic: float,
        exc: Exception,
    ) -> TrackEnrichmentResult:
        self.logger.exception("vehicle enrichment failed for track=%s", track.local_track_id)
        return self._build_base_result(
            track=track,
            vehicle_class_confidence=self._compute_vehicle_class_confidence(track),
            status=ENRICHMENT_STATUS_ERROR,
            evidence_used=[],
            started_iso=started_iso,
            started_monotonic=started_monotonic,
            errors=[str(exc)],
        )

    def _select_best_evidence(self, items: list[Any]) -> tuple[list[Any], list[Any]]:
        best_count = int(self.config["best_crops_per_track"])
        accepted = [item for item in items if not item.rejection_reasons]
        rejected = [item for item in items if item.rejection_reasons]
        accepted.sort(key=lambda item: (item.quality_score, item.detection_confidence, item.frame_number), reverse=True)
        selected = accepted[:best_count]
        rejected.extend(accepted[best_count:])
        for item in accepted[best_count:]:
            item.rejection_reasons.append("best_crop_limit_exceeded")
        return selected, rejected

    def _materialize_selected_crop(self, track: LocalTrack, item: Any) -> Any:
        current_path = Path(str(item.vehicle_crop_path)) if item.vehicle_crop_path else None
        if current_path is None or not current_path.exists():
            return item
        target_path = self.output_manager.vehicle_enrichment_track_crop_path(
            track.local_track_id,
            item.frame_number,
            suffix=str(item.evidence_role).upper(),
        )
        target_path.parent.mkdir(parents=True, exist_ok=True)
        if current_path.resolve() != target_path.resolve():
            image = cv2.imread(str(current_path))
            if image is None or image.size == 0:
                shutil.copyfile(str(current_path), str(target_path))
            else:
                cv2.imwrite(str(target_path), image, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        item.vehicle_crop_path = str(target_path)
        return item

    @staticmethod
    def _compute_vehicle_class_confidence(track: LocalTrack) -> float | None:
        final_class = str(track.final_class or "").strip()
        if not final_class:
            return None
        count = max(0, int(track.class_counts.get(final_class.lower(), 0) or track.class_counts.get(final_class, 0)))
        confidence_sum = float(track.class_confidence_sums.get(final_class.lower(), 0.0) or track.class_confidence_sums.get(final_class, 0.0))
        if count <= 0:
            return None
        return max(0.0, min(1.0, confidence_sum / count))

    def _cleanup_track(self, local_track_id: str) -> None:
        self._metrics["cleanup_count"] += 1
        self._metrics["current_in_memory_tracks"] = max(0, self._metrics["current_in_memory_tracks"] - 1)
