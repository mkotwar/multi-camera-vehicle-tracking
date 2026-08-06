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
from .image_size_policy import ImageSizePolicy, normalize_image_size_policy
from .make_model import VehicleMakeModelClassifier
from .ocr_mukul import OCRMukulFlorenceFlow
from .vehicle_attribute_flow import BaseFlorenceVehicleAttributesFlow
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
    PlateDetectionResult,
    PlateOCRResult,
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
    image_size_policy = dict(section.get("image_size_policy", {}) or {})
    evidence_quality = dict(section.get("evidence_quality", {}) or {})
    track_evidence = dict(section.get("track_evidence", {}) or {})
    florence_comparison = dict(section.get("florence_comparison", {}) or {})
    ocr_mukul = dict(section.get("ocr_mukul", {}) or {})
    vehicle_attributes = dict(section.get("vehicle_attributes", {}) or {})
    vehicle_attribute_colour = dict(vehicle_attributes.get("colour", {}) or {})
    vehicle_attribute_body_type = dict(vehicle_attributes.get("body_type", {}) or {})
    return {
        "enabled": bool(section.get("enabled", False)),
        "trigger": str(section.get("trigger", "track_completed")).strip() or "track_completed",
        "florence_mode": str(section.get("florence_mode", "current")).strip() or "current",
        "fail_open": bool(section.get("fail_open", True)),
        "best_crops_per_track": max(1, int(section.get("best_crops_per_track", 3))),
        "write_separate_output": bool(section.get("write_separate_output", True)),
        "extend_tracks_json": bool(section.get("extend_tracks_json", True)),
        "florence_comparison": {
            "enabled": bool(florence_comparison.get("enabled", False)),
            "current_flow": bool(florence_comparison.get("current_flow", True)),
            "old_td_case2_flow": bool(florence_comparison.get("old_td_case2_flow", False)),
            "old_project_root": str(florence_comparison.get("old_project_root", "")).strip(),
            "output_directory": str(florence_comparison.get("output_directory", "")).strip(),
        },
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
            "processor_path": str(shared_florence.get("processor_path", "")).strip(),
            "adapter_path": str(shared_florence.get("adapter_path", "model_weights/florence/adaptor_florance_baseFT")).strip(),
            "adapter_enabled": bool(shared_florence.get("adapter_enabled", False)),
            "device": str(shared_florence.get("device", "auto")).strip() or "auto",
            "dtype": str(shared_florence.get("dtype", "auto")).strip() or "auto",
            "trust_remote_code": bool(shared_florence.get("trust_remote_code", True)),
            "attention_implementation": str(shared_florence.get("attention_implementation", "eager")).strip() or "eager",
            "max_new_tokens": int(shared_florence.get("max_new_tokens", 128)),
            "num_beams": int(shared_florence.get("num_beams", 3)),
            "use_cache": bool(shared_florence.get("use_cache", False)),
            "local_files_only": bool(shared_florence.get("local_files_only", False)),
            "lazy_load": bool(shared_florence.get("lazy_load", True)),
            "fail_if_adapter_missing": bool(shared_florence.get("fail_if_adapter_missing", False)),
        },
        "image_size_policy": image_size_policy,
        "evidence_quality": evidence_quality,
        "track_evidence": {
            "enabled": bool(track_evidence.get("enabled", True)),
            "maximum_candidates_per_track": max(1, int(track_evidence.get("maximum_candidates_per_track", 12))),
            "final_crops_per_attribute": max(1, int(track_evidence.get("final_crops_per_attribute", 3))),
            "minimum_frame_gap": max(0, int(track_evidence.get("minimum_frame_gap", 3))),
            "preferred_frame_gap": max(0, int(track_evidence.get("preferred_frame_gap", 8))),
            "deduplicate_similar_crops": bool(track_evidence.get("deduplicate_similar_crops", True)),
            "classify_at_track_completion": bool(track_evidence.get("classify_at_track_completion", True)),
            "early_classification": {
                "enabled": bool(dict(track_evidence.get("early_classification", {}) or {}).get("enabled", False)),
                "require_preferred_resolution": bool(dict(track_evidence.get("early_classification", {}) or {}).get("require_preferred_resolution", True)),
                "minimum_quality_score": float(dict(track_evidence.get("early_classification", {}) or {}).get("minimum_quality_score", 0.80)),
            },
        },
        "body_type": {
            "enabled": bool(body_type.get("enabled", False)),
            "backend": str(body_type.get("backend", "florence2")).strip() or "florence2",
            "run_only_when_vehicle_class": [
                str(item).strip().upper()
                for item in body_type.get("run_only_when_vehicle_class", ["CAR"])
                if str(item).strip()
            ],
            "maximum_crops_per_track": max(
                1,
                int(body_type.get("maximum_crops_per_track", track_evidence.get("final_crops_per_attribute", 3))),
            ),
            "minimum_crop_width": max(1, int(body_type.get("minimum_crop_width", 256))),
            "minimum_crop_height": max(1, int(body_type.get("minimum_crop_height", 192))),
            "allowed_labels": [str(item).strip().upper() for item in body_type.get("allowed_labels", []) if str(item).strip()],
        },
        "colour": {
            "enabled": bool(colour.get("enabled", False)),
            "backend": str(colour.get("backend", "florence2")).strip() or "florence2",
            "run_only_when_vehicle_class": [
                str(item).strip().upper()
                for item in colour.get("run_only_when_vehicle_class", ["3WHEELER", "BUS", "CAR", "MOTORCYCLE", "TRUCK"])
                if str(item).strip()
            ],
            "maximum_crops_per_track": max(
                1,
                int(colour.get("maximum_crops_per_track", track_evidence.get("final_crops_per_attribute", 3))),
            ),
            "minimum_crop_width": max(1, int(colour.get("minimum_crop_width", 256))),
            "minimum_crop_height": max(1, int(colour.get("minimum_crop_height", 192))),
            "allowed_labels": [str(item).strip().upper() for item in colour.get("allowed_labels", []) if str(item).strip()],
        },
        "ocr_mukul": {
            "enabled": bool(ocr_mukul.get("enabled", False)),
            "task_token": str(ocr_mukul.get("task_token", "<CAPTION>")).strip() or "<CAPTION>",
            "prompt": str(ocr_mukul.get("prompt", "") or ""),
            "reuse_caption_for_attributes": bool(ocr_mukul.get("reuse_caption_for_attributes", True)),
            "maximum_crops_per_track": max(1, int(ocr_mukul.get("maximum_crops_per_track", track_evidence.get("final_crops_per_attribute", 3)))),
            "body_type_vehicle_classes": [
                str(item).strip().upper()
                for item in ocr_mukul.get("body_type_vehicle_classes", body_type.get("eligible_vehicle_classes", body_type.get("run_only_when_vehicle_class", ["CAR"])))
                if str(item).strip()
            ],
            "colour_vehicle_classes": [
                str(item).strip().upper()
                for item in ocr_mukul.get("colour_vehicle_classes", colour.get("eligible_vehicle_classes", colour.get("run_only_when_vehicle_class", ["3WHEELER", "BUS", "CAR", "MOTORCYCLE", "TRUCK"])))
                if str(item).strip()
            ],
        },
        "vehicle_attributes": {
            "enabled": bool(vehicle_attributes.get("enabled", False)),
            "backend": str(vehicle_attributes.get("backend", "base_florence")).strip() or "base_florence",
            "maximum_crops_per_track": max(1, int(vehicle_attributes.get("maximum_crops_per_track", track_evidence.get("final_crops_per_attribute", 3)))),
            "reuse_single_response_for_attributes": bool(vehicle_attributes.get("reuse_single_response_for_attributes", True)),
            "task_token": str(vehicle_attributes.get("task_token", "<VQA>")).strip() or "<VQA>",
            "prompt": str(vehicle_attributes.get("prompt", "") or ""),
            "colour": {
                "enabled": bool(vehicle_attribute_colour.get("enabled", False)),
                "backend": str(vehicle_attribute_colour.get("backend", "base_florence")).strip() or "base_florence",
                "task_token": str(vehicle_attribute_colour.get("task_token", vehicle_attributes.get("task_token", "<VQA>"))).strip() or "<VQA>",
                "prompt": str(vehicle_attribute_colour.get("prompt", vehicle_attributes.get("prompt", "")) or ""),
                "generation": dict(vehicle_attribute_colour.get("generation", {}) or {}),
            },
            "body_type": {
                "enabled": bool(vehicle_attribute_body_type.get("enabled", True)),
                "backend": str(vehicle_attribute_body_type.get("backend", "base_florence")).strip() or "base_florence",
                "task_token": str(vehicle_attribute_body_type.get("task_token", vehicle_attributes.get("task_token", "<VQA>"))).strip() or "<VQA>",
                "prompt": str(vehicle_attribute_body_type.get("prompt", vehicle_attributes.get("prompt", "")) or ""),
                "generation": dict(vehicle_attribute_body_type.get("generation", {}) or {}),
            },
            "florence": dict(vehicle_attributes.get("florence", {}) or {}),
        },
        "make_model": {"enabled": bool(make_model.get("enabled", False))},
        "plate": {
            "detection_enabled": bool(plate.get("detection_enabled", False)),
            "colour_enabled": bool(plate.get("colour_enabled", False)),
            "recognition_enabled": bool(plate.get("recognition_enabled", False)),
            "detector": dict(plate.get("detector", {}) or {}),
            "ocr": dict(plate.get("ocr", {}) or {}),
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
        self.image_size_policy: ImageSizePolicy = normalize_image_size_policy(
            config.get("vehicle_enrichment", {}).get("image_size_policy", {}),
            fallback_body_type=self.config["body_type"],
            fallback_colour=self.config["colour"],
            detection=dict(config.get("detection", {}) or {}),
        )
        self.adapter = EvidenceAdapter(self.config, output_manager, logger)
        self.quality_evaluator = EvidenceQualityEvaluator(normalize_quality_config(self.config), self.image_size_policy)
        self.attribute_aggregator = AttributeAggregator()
        shared_florence_backend_config = {
            key: value
            for key, value in self.config["shared_florence"].items()
            if key in FlorenceBackendConfig.__annotations__
        }
        self.florence_backend = FlorenceBackend(FlorenceBackendConfig(**shared_florence_backend_config), logger=logger)
        vehicle_attribute_florence_config = dict(shared_florence_backend_config)
        vehicle_attribute_florence_config.update(
            {
                key: value
                for key, value in dict(self.config["vehicle_attributes"].get("florence", {}) or {}).items()
                if key in FlorenceBackendConfig.__annotations__
            }
        )
        vehicle_attribute_florence_config["adapter_enabled"] = False
        self.vehicle_attribute_backend = FlorenceBackend(
            FlorenceBackendConfig(**vehicle_attribute_florence_config),
            logger=logger,
            adapter_enabled_override=False,
        )
        ocr_shared_config = dict(shared_florence_backend_config)
        ocr_shared_config["adapter_enabled"] = True
        ocr_shared_config.update(
            {
                key: value
                for key, value in dict(self.config["plate"].get("ocr", {}).get("florence", {}) or {}).items()
                if key in FlorenceBackendConfig.__annotations__
            }
        )
        self.ocr_mukul_backend = FlorenceBackend(
            FlorenceBackendConfig(**ocr_shared_config),
            logger=logger,
            adapter_enabled_override=True,
        )
        self.body_type_classifier = VehicleBodyTypeClassifier(
            self.config["body_type"],
            backend=self.florence_backend,
            image_size_policy=self.image_size_policy,
            logger=logger,
        )
        self.colour_classifier = VehicleColourClassifier(
            self.config["colour"],
            backend=self.florence_backend,
            image_size_policy=self.image_size_policy,
            logger=logger,
        )
        self.make_model_classifier = VehicleMakeModelClassifier(self.config["make_model"])
        self.plate_detector = PlateDetector(self.config["plate"])
        self.plate_quality_validator = PlateQualityValidator(self.config["plate"])
        self.plate_colour_classifier = PlateColourClassifier(self.config["plate"])
        self.plate_ocr_engine = PlateOCREngine(self.config["plate"].get("ocr", self.config["ocr"]), backend=self.ocr_mukul_backend)
        self.plate_result_aggregator = PlateResultAggregator(self.config["plate"])
        self.vehicle_attribute_flow = BaseFlorenceVehicleAttributesFlow(
            self.config["vehicle_attributes"],
            backend=self.vehicle_attribute_backend,
            image_size_policy=self.image_size_policy,
            logger=logger,
        )
        self.ocr_mukul_flow = OCRMukulFlorenceFlow(
            {
                **self.config["ocr_mukul"],
                "enabled": bool(self.config["ocr_mukul"]["enabled"] or self.config["florence_mode"] in {"ocr_mukul", "comparison"}),
            },
            backend=self.ocr_mukul_backend,
            image_size_policy=self.image_size_policy,
            logger=logger,
        )
        self._results_by_track: dict[str, TrackEnrichmentResult] = {}
        self._metrics: dict[str, Any] = {
            "completed_tracks_received": 0,
            "tracks_with_existing_evidence": 0,
            "tracks_without_evidence": 0,
            "tracks_with_candidate_evidence": 0,
            "tracks_without_candidate_evidence": 0,
            "tracks_with_acceptable_crop": 0,
            "tracks_with_preferred_crop": 0,
            "tracks_with_no_florence_eligible_crop": 0,
            "evidence_items_received": 0,
            "crop_candidates_evaluated": 0,
            "crop_candidates_rejected": 0,
            "crops_retained": 0,
            "crops_rejected_by_reason": {},
            "enrichment_results_written": 0,
            "enrichment_failures": 0,
            "cleanup_count": 0,
            "current_in_memory_tracks": 0,
            "florence_crops_below_minimum": 0,
            "florence_crops_acceptable": 0,
            "florence_crops_preferred": 0,
            "florence_crops_rejected_width": 0,
            "florence_crops_rejected_height": 0,
            "florence_crops_rejected_quality": 0,
            "florence_crops_padded_to_square": 0,
            "florence_original_size_distribution": {},
            "body_type_selected_crop_count": 0,
            "colour_selected_crop_count": 0,
            "body_type_tracks_waited_for_completion": 0,
            "colour_tracks_waited_for_completion": 0,
            "early_classification_attempts": 0,
            "early_classification_successes": 0,
            "track_evidence_release_count": 0,
            "track_evidence_pending_count": 0,
            "plate_ocr_adapter_loads": 0,
            "plate_ocr_inference_calls": 0,
            "plate_ocr_skipped_no_plate": 0,
            "plate_ocr_skipped_low_quality": 0,
            "plate_ocr_attempts": 0,
            "gpu_memory_before_ocr_load_mb": 0.0,
            "gpu_memory_after_ocr_load_mb": 0.0,
        }

    @property
    def metrics(self) -> dict[str, Any]:
        payload = dict(self._metrics)
        payload["crops_rejected_by_reason"] = dict(self._metrics["crops_rejected_by_reason"])
        payload["florence_original_size_distribution"] = dict(self._metrics["florence_original_size_distribution"])
        payload.update(self.florence_backend.metrics)
        payload.update({f"ocr_backend_{key}": value for key, value in self.ocr_mukul_backend.metrics.items()})
        payload.update(self.body_type_classifier.metrics)
        payload.update(self.colour_classifier.metrics)
        payload.update(self.ocr_mukul_flow.metrics)
        payload.update(self.vehicle_attribute_flow.metrics)
        payload["plate_ocr_adapter_loads"] = int(self.ocr_mukul_backend.metrics.get("florence_load_successes", 0))
        payload["adapter_load_count"] = int(self.vehicle_attribute_backend.metrics.get("florence_adapter_load_successes", 0)) + int(self.ocr_mukul_backend.metrics.get("florence_adapter_load_successes", 0))
        payload["base_model_load_count"] = int(self.vehicle_attribute_backend.metrics.get("florence_load_successes", 0))
        payload["body_type_inference_calls"] = int(self.vehicle_attribute_flow.metrics.get("vehicle_attribute_body_inference_calls", 0))
        payload["colour_inference_calls"] = int(self.vehicle_attribute_flow.metrics.get("vehicle_attribute_colour_inference_calls", 0))
        payload["average_colour_inference_time_ms"] = float(self.vehicle_attribute_flow.metrics.get("vehicle_attribute_average_colour_inference_time_ms", 0.0) or 0.0)
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
        if scored:
            self._metrics["tracks_with_candidate_evidence"] += 1
        else:
            self._metrics["tracks_without_candidate_evidence"] += 1
        selected, rejected = self._select_best_evidence(scored)
        for rejected_item in rejected:
            self._metrics["crop_candidates_rejected"] += 1
            for reason in rejected_item.rejection_reasons:
                self._metrics["crops_rejected_by_reason"][reason] = self._metrics["crops_rejected_by_reason"].get(reason, 0) + 1
                if reason == "crop_width_below_florence_minimum":
                    self._metrics["florence_crops_rejected_width"] += 1
                if reason == "crop_height_below_florence_minimum":
                    self._metrics["florence_crops_rejected_height"] += 1
                if reason == "crop_rejected_quality":
                    self._metrics["florence_crops_rejected_quality"] += 1
        self._metrics["crops_retained"] += len(selected)
        selected = [self._materialize_selected_crop(track, item) for item in selected]
        self._record_selected_crop_metrics(selected)
        eligible_crop_count = len(
            [
                item
                for item in selected
                if getattr(item, "florence_eligible_for_body_type", False) or getattr(item, "florence_eligible_for_colour", False)
            ]
        )
        preferred_crop_count = len([item for item in selected if getattr(item, "resolution_tier", "") == "preferred"])
        if eligible_crop_count > 0:
            self._metrics["tracks_with_acceptable_crop"] += 1
        else:
            self._metrics["tracks_with_no_florence_eligible_crop"] += 1
        if preferred_crop_count > 0:
            self._metrics["tracks_with_preferred_crop"] += 1
        self._metrics["track_evidence_pending_count"] = max(0, self._metrics["current_in_memory_tracks"])

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

        florence_mode = str(self.config["florence_mode"]).strip().lower()
        comparison_payload: dict[str, Any] | None = None
        crop_level_captions: list[dict[str, Any]] = []
        crop_level_body_types: list[dict[str, Any]] = []
        crop_level_colours: list[dict[str, Any]] = []
        caption_inference_count = 0
        adapter_loaded: bool | None = None
        if self.config["vehicle_attributes"]["enabled"]:
            attribute_result = self.vehicle_attribute_flow.classify(request)
            body_type_result = attribute_result.body_type
            colour_result = attribute_result.colour
            caption_inference_count = attribute_result.inference_count
            adapter_loaded = attribute_result.adapter_loaded
            crop_level_captions = [
                {
                    "crop_path": row["vehicle_crop_path"],
                    "frame_index": row["frame_index"],
                    "caption": row["post_processed_response"],
                    "raw_response": row["raw_response"],
                    "post_processed_response": row["post_processed_response"],
                }
                for row in attribute_result.crop_level_rows
            ]
            crop_level_body_types = [
                {
                    "crop_path": row["vehicle_crop_path"],
                    "frame_index": row["frame_index"],
                    "raw_body_type_phrase": "",
                    "normalized_body_type": row["parsed_body_type"],
                    "status": row.get("body_type_status"),
                    "reason": row.get("body_type_reason"),
                }
                for row in attribute_result.crop_level_rows
            ]
            crop_level_colours = [
                {
                    "crop_path": row["vehicle_crop_path"],
                    "frame_index": row["frame_index"],
                    "raw_colour_phrase": "",
                    "normalized_colour": row["parsed_colour"],
                    "status": row.get("colour_status"),
                    "reason": row.get("colour_reason"),
                    "crop_source": row.get("crop_source"),
                    "crop_available": row.get("crop_available"),
                    "crop_skip_reason": row.get("crop_skip_reason"),
                }
                for row in attribute_result.crop_level_rows
            ]
        elif florence_mode == "ocr_mukul":
            ocr_result = self.ocr_mukul_flow.classify(request)
            body_type_result = ocr_result.body_type
            colour_result = ocr_result.colour
            caption_inference_count = ocr_result.caption_inference_count
            adapter_loaded = ocr_result.adapter_loaded
            crop_level_captions = [
                {"crop_path": row["crop_path"], "frame_index": row["frame_index"], "caption": row["caption"]}
                for row in ocr_result.crop_level_rows
            ]
            crop_level_body_types = [
                {"crop_path": row["crop_path"], "frame_index": row["frame_index"], "raw_body_type_phrase": row["raw_body_type_phrase"], "normalized_body_type": row["normalized_body_type"]}
                for row in ocr_result.crop_level_rows
            ]
            crop_level_colours = [
                {"crop_path": row["crop_path"], "frame_index": row["frame_index"], "raw_colour_phrase": row["raw_colour_phrase"], "normalized_colour": row["normalized_colour"]}
                for row in ocr_result.crop_level_rows
            ]
        elif florence_mode == "comparison":
            current_body_type_result = self.body_type_classifier.classify(request)
            current_colour_result = self.colour_classifier.classify(request)
            ocr_result = self.ocr_mukul_flow.classify(request)
            body_type_result = current_body_type_result
            colour_result = current_colour_result
            caption_inference_count = ocr_result.caption_inference_count
            adapter_loaded = ocr_result.adapter_loaded
            crop_level_captions = [
                {"crop_path": row["crop_path"], "frame_index": row["frame_index"], "caption": row["caption"]}
                for row in ocr_result.crop_level_rows
            ]
            crop_level_body_types = [
                {"crop_path": row["crop_path"], "frame_index": row["frame_index"], "raw_body_type_phrase": row["raw_body_type_phrase"], "normalized_body_type": row["normalized_body_type"]}
                for row in ocr_result.crop_level_rows
            ]
            crop_level_colours = [
                {"crop_path": row["crop_path"], "frame_index": row["frame_index"], "raw_colour_phrase": row["raw_colour_phrase"], "normalized_colour": row["normalized_colour"]}
                for row in ocr_result.crop_level_rows
            ]
            comparison_payload = {
                "current": {
                    "body_type_label": current_body_type_result.label,
                    "body_type_reason": current_body_type_result.aggregation_reason or current_body_type_result.reason,
                    "colour_label": current_colour_result.label,
                    "colour_reason": current_colour_result.aggregation_reason or current_colour_result.reason,
                    "body_type_raw_responses": [str(item.raw_response) for item in current_body_type_result.predictions if item.raw_response not in (None, "")],
                    "colour_raw_responses": [str(item.raw_response) for item in current_colour_result.predictions if item.raw_response not in (None, "")],
                },
                "ocr_mukul": {
                    "body_type_label": ocr_result.body_type.label,
                    "body_type_reason": ocr_result.body_type.aggregation_reason or ocr_result.body_type.reason,
                    "colour_label": ocr_result.colour.label,
                    "colour_reason": ocr_result.colour.aggregation_reason or ocr_result.colour.reason,
                    "captions": list(ocr_result.crop_level_rows),
                },
            }
        else:
            body_type_result = self.body_type_classifier.classify(request)
            colour_result = self.colour_classifier.classify(request)
            adapter_loaded = self.florence_backend.adapter_active
        self._apply_prediction_selection_metadata(selected, body_type_result.predictions, "body_type")
        self._apply_prediction_selection_metadata(selected, colour_result.predictions, "colour")
        self._metrics["body_type_selected_crop_count"] += len(body_type_result.predictions)
        self._metrics["colour_selected_crop_count"] += len(colour_result.predictions)
        self._metrics["body_type_tracks_waited_for_completion"] += 1
        self._metrics["colour_tracks_waited_for_completion"] += 1
        make_model_result = self.make_model_classifier.classify(request)
        plate_detection_result = self.plate_detector.detect(selected[0]) if selected else PlateDetectionResult(detected=False, predictions=[], status="skipped", source="plate.detector", reason="no_selected_vehicle_crop")
        plate_quality_result = self.plate_quality_validator.validate(None)
        if not getattr(plate_detection_result, "detected", False):
            self._metrics["plate_ocr_skipped_no_plate"] += 1
            plate_ocr_result = PlateOCRResult(text=None, predictions=[], status="skipped", source="plate.ocr_engine", reason="no_plate_detected")
        else:
            self._metrics["plate_ocr_attempts"] += 1
            self._metrics["gpu_memory_before_ocr_load_mb"] = float(self.ocr_mukul_backend.metrics.get("gpu_memory_allocated_mb") or 0.0)
            plate_ocr_result = self.plate_ocr_engine.recognize(None)
            self._metrics["gpu_memory_after_ocr_load_mb"] = float(self.ocr_mukul_backend.metrics.get("gpu_memory_allocated_mb") or 0.0)
            if plate_ocr_result.status == "completed":
                self._metrics["plate_ocr_inference_calls"] += 1
        plate_aggregate = self.plate_result_aggregator.aggregate(plate_detection_result, plate_quality_result, plate_ocr_result)
        overall_status = (
            ENRICHMENT_STATUS_COMPLETED
            if self.body_type_classifier.enabled or self.colour_classifier.enabled or self.config["vehicle_attributes"]["enabled"]
            else ENRICHMENT_STATUS_EVIDENCE_READY
        )

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
            plate_detected=plate_aggregate["plate_detected"],
            plate_text=plate_aggregate["plate_text"],
            candidate_crop_count=len(scored),
            eligible_crop_count=eligible_crop_count,
            preferred_crop_count=preferred_crop_count,
            selected_body_type_crop_paths=[str(item.source_crop_path) for item in body_type_result.predictions if item.source_crop_path],
            selected_colour_crop_paths=[str(item.source_crop_path) for item in colour_result.predictions if item.source_crop_path],
            florence_mode=florence_mode,
            adapter_loaded=adapter_loaded,
            selected_crop_paths=[str(getattr(item, "vehicle_crop_path", "")) for item in selected if getattr(item, "vehicle_crop_path", None)],
            crop_level_captions=crop_level_captions,
            crop_level_body_types=crop_level_body_types,
            crop_level_colours=crop_level_colours,
            final_body_type_reason=body_type_result.aggregation_reason or body_type_result.reason,
            final_colour_reason=colour_result.aggregation_reason or colour_result.reason,
            caption_inference_count=caption_inference_count,
            vehicle_attribute_raw_responses=attribute_result.raw_responses if self.config["vehicle_attributes"]["enabled"] else [],
            vehicle_attribute_selected_crop_paths=[str(item.get("vehicle_crop_path")) for item in (attribute_result.crop_level_rows if self.config["vehicle_attributes"]["enabled"] else [])],
            vehicle_attribute_inference_count=caption_inference_count if self.config["vehicle_attributes"]["enabled"] else 0,
            attribute_backend="base_florence" if self.config["vehicle_attributes"]["enabled"] else None,
            plate_ocr_backend="ocr_mukul_adapter" if bool(self.config["plate"].get("ocr", {}).get("enabled", False)) else None,
            plate_ocr_attempted=False,
            plate_ocr_raw_response=None,
            plate_ocr_reason=plate_aggregate["reason"],
            plate_quality_status=getattr(plate_quality_result, "reason", None),
            comparison_payload=comparison_payload,
            classification_trigger="track_completion",
            final_reason=body_type_result.aggregation_reason or body_type_result.reason or colour_result.aggregation_reason or colour_result.reason,
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
        plate_text: str | None = None,
        vehicle_attribute_raw_responses: list[str] | None = None,
        vehicle_attribute_selected_crop_paths: list[str] | None = None,
        vehicle_attribute_inference_count: int = 0,
        plate_detection_confidence: float | None = None,
        plate_bbox: list[float] | None = None,
        plate_crop_path: str | None = None,
        plate_ocr_attempted: bool = False,
        plate_ocr_raw_response: str | None = None,
        plate_text_confidence: float | None = None,
        plate_ocr_reason: str | None = None,
        attribute_backend: str | None = None,
        plate_ocr_backend: str | None = None,
        plate_quality_status: str | None = None,
        candidate_crop_count: int = 0,
        eligible_crop_count: int = 0,
        preferred_crop_count: int = 0,
        selected_body_type_crop_paths: list[str] | None = None,
        selected_colour_crop_paths: list[str] | None = None,
        florence_mode: str | None = None,
        adapter_loaded: bool | None = None,
        selected_crop_paths: list[str] | None = None,
        crop_level_captions: list[dict[str, Any]] | None = None,
        crop_level_body_types: list[dict[str, Any]] | None = None,
        crop_level_colours: list[dict[str, Any]] | None = None,
        final_body_type_reason: str | None = None,
        final_colour_reason: str | None = None,
        caption_inference_count: int = 0,
        comparison_payload: dict[str, Any] | None = None,
        classification_trigger: str | None = None,
        final_reason: str | None = None,
    ) -> TrackEnrichmentResult:
        completed_iso = datetime.now(timezone.utc).isoformat()
        return TrackEnrichmentResult(
            local_track_id=track.local_track_id,
            camera_id=track.camera_id,
            vehicle_class=str(track.final_class or "UNKNOWN").upper(),
            vehicle_class_confidence=vehicle_class_confidence,
            vehicle_body_type=body_type_result or self._default_body_type_result(status),
            vehicle_colour=colour_result or self._default_colour_result(status),
            vehicle_make=vehicle_make,
            vehicle_model=vehicle_model,
            plate_detected=plate_detected,
            plate_colour=None,
            registration_category=None,
            plate_text=plate_text,
            status=status,
            vehicle_attribute_raw_responses=list(vehicle_attribute_raw_responses or []),
            vehicle_attribute_selected_crop_paths=list(vehicle_attribute_selected_crop_paths or []),
            vehicle_attribute_inference_count=vehicle_attribute_inference_count,
            plate_detection_confidence=plate_detection_confidence,
            plate_bbox=plate_bbox,
            plate_crop_path=plate_crop_path,
            plate_ocr_attempted=plate_ocr_attempted,
            plate_ocr_raw_response=plate_ocr_raw_response,
            plate_text_confidence=plate_text_confidence,
            plate_ocr_reason=plate_ocr_reason,
            attribute_backend=attribute_backend,
            plate_ocr_backend=plate_ocr_backend,
            plate_quality_status=plate_quality_status,
            evidence_used=evidence_used,
            candidate_crop_count=candidate_crop_count,
            eligible_crop_count=eligible_crop_count,
            preferred_crop_count=preferred_crop_count,
            selected_body_type_crop_paths=list(selected_body_type_crop_paths or []),
            selected_colour_crop_paths=list(selected_colour_crop_paths or []),
            florence_mode=florence_mode,
            adapter_loaded=adapter_loaded,
            selected_crop_paths=list(selected_crop_paths or []),
            crop_level_captions=list(crop_level_captions or []),
            crop_level_body_types=list(crop_level_body_types or []),
            crop_level_colours=list(crop_level_colours or []),
            final_body_type_reason=final_body_type_reason,
            final_colour_reason=final_colour_reason,
            caption_inference_count=caption_inference_count,
            comparison_payload=dict(comparison_payload) if comparison_payload else None,
            classification_trigger=classification_trigger,
            final_reason=final_reason,
            predictions=[],
            errors=errors,
            processing_started_at=started_iso,
            processing_completed_at=completed_iso,
            processing_duration_ms=float((time.perf_counter() - started_monotonic) * 1000.0),
        )

    def _default_body_type_result(self, enrichment_status: str) -> VehicleBodyTypeResult:
        if self.body_type_classifier.enabled and enrichment_status == ENRICHMENT_STATUS_NO_EVIDENCE:
            return VehicleBodyTypeResult(
                label="UNKNOWN",
                predictions=[],
                status="skipped",
                source="body_type.classifier",
                reason="no_evidence",
            )
        return VehicleBodyTypeResult(
            label="UNKNOWN",
            predictions=[],
            status=ATTRIBUTE_STATUS_DISABLED,
            source="body_type.classifier",
            reason="Vehicle body type inference is disabled.",
        )

    def _default_colour_result(self, enrichment_status: str) -> VehicleColourResult:
        if self.colour_classifier.enabled and enrichment_status == ENRICHMENT_STATUS_NO_EVIDENCE:
            return VehicleColourResult(
                label="UNKNOWN",
                predictions=[],
                status="skipped",
                source="colour.classifier",
                reason="no_evidence",
            )
        return VehicleColourResult(
            label="UNKNOWN",
            predictions=[],
            status=ATTRIBUTE_STATUS_DISABLED,
            source="colour.classifier",
            reason="Vehicle colour inference is disabled.",
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
        accepted.sort(key=lambda item: (item.ranking_score, item.quality_score, item.detection_confidence, item.frame_number), reverse=True)
        selected = accepted[:best_count]
        rejected.extend(accepted[best_count:])
        for item in accepted[best_count:]:
            item.rejection_reasons.append("best_crop_limit_exceeded")
        return selected, rejected

    def _record_selected_crop_metrics(self, items: list[Any]) -> None:
        for item in items:
            tier = str(getattr(item, "resolution_tier", "below_minimum"))
            if tier == "below_minimum":
                self._metrics["florence_crops_below_minimum"] += 1
            elif tier == "acceptable":
                self._metrics["florence_crops_acceptable"] += 1
            elif tier == "preferred":
                self._metrics["florence_crops_preferred"] += 1
            key = f"{int(item.original_crop_width)}x{int(item.original_crop_height)}"
            self._metrics["florence_original_size_distribution"][key] = self._metrics["florence_original_size_distribution"].get(key, 0) + 1

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
        self._metrics["track_evidence_release_count"] += 1
        self._metrics["track_evidence_pending_count"] = max(0, self._metrics["current_in_memory_tracks"])

    def _apply_prediction_selection_metadata(self, items: list[Any], predictions: list[Any], attribute_name: str) -> None:
        by_path = {
            str(getattr(item, "vehicle_crop_path", "")): item
            for item in items
            if getattr(item, "vehicle_crop_path", None)
        }
        selected_frames = sorted(
            int(prediction.source_frame_number)
            for prediction in predictions
            if getattr(prediction, "source_frame_number", None) is not None
        )
        previous_frame: int | None = None
        for prediction in predictions:
            crop_path = str(getattr(prediction, "source_crop_path", "") or "")
            item = by_path.get(crop_path)
            if item is None:
                continue
            current_frame = int(getattr(prediction, "source_frame_number", item.frame_number))
            if previous_frame is None:
                frame_gap = None
            else:
                frame_gap = current_frame - previous_frame
            previous_frame = current_frame
            if attribute_name == "body_type":
                item.selected_for_body_type = True
                item.body_type_crop_result = str(prediction.label)
            else:
                item.selected_for_colour = True
                item.colour_crop_result = str(prediction.label)
            item.frame_gap_from_previous_selected = frame_gap
