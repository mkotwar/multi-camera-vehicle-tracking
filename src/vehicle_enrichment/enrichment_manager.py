from __future__ import annotations

import logging
import queue
import shutil
import threading
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2

from src.models import LocalTrack, TrackEvidence
from src.output_writer import RunOutputManager

from .attribute_aggregator import AttributeAggregator
from .body_type import VehicleBodyTypeClassifier
from .body_type.labels import BODY_TYPE_ALLOWED_LABELS, BODY_TYPE_PROMPT_TEXT, BODY_TYPE_TASK_PROMPT
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
    ATTRIBUTE_STATUS_READY,
    ENRICHMENT_STATUS_COMPLETED,
    ENRICHMENT_STATUS_DISABLED,
    ENRICHMENT_STATUS_ERROR,
    ENRICHMENT_STATUS_EVIDENCE_READY,
    ENRICHMENT_STATUS_NO_EVIDENCE,
    PlateDetectionResult,
    PlateOCRResult,
    PlateQualityResult,
    TrackEnrichmentRequest,
    TrackEnrichmentResult,
    VehicleMakeModelResult,
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
    async_colour = dict(section.get("async_colour", {}) or {})
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
        "async_colour": {
            "enabled": bool(async_colour.get("enabled", False)),
            "queue_size": max(1, int(async_colour.get("queue_size", 100))),
            "worker_count": max(1, int(async_colour.get("worker_count", 1))),
            "queue_put_timeout_seconds": float(async_colour.get("queue_put_timeout_seconds", 0.1)),
        },
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
            "class_specific_minimums": {
                str(class_name).strip().lower(): {
                    "minimum_crop_width": max(1, int(dict(payload or {}).get("minimum_crop_width", evidence.get("minimum_crop_width", 100)))),
                    "minimum_crop_height": max(1, int(dict(payload or {}).get("minimum_crop_height", evidence.get("minimum_crop_height", 70)))),
                }
                for class_name, payload in dict(evidence.get("class_specific_minimums", {}) or {}).items()
                if str(class_name).strip()
            },
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
                "task_token": str(vehicle_attribute_body_type.get("task_token", BODY_TYPE_TASK_PROMPT)).strip() or BODY_TYPE_TASK_PROMPT,
                "prompt": str(vehicle_attribute_body_type.get("prompt", BODY_TYPE_PROMPT_TEXT) or BODY_TYPE_PROMPT_TEXT),
                "generation": dict(vehicle_attribute_body_type.get("generation", {}) or {}),
                "car_only": bool(vehicle_attribute_body_type.get("car_only", True)),
                "run_only_when_vehicle_class": [
                    str(item).strip().upper()
                    for item in vehicle_attribute_body_type.get("run_only_when_vehicle_class", ["CAR"])
                    if str(item).strip()
                ],
                "maximum_crops_per_track": max(
                    1, int(vehicle_attribute_body_type.get("maximum_crops_per_track", vehicle_attributes.get("maximum_crops_per_track", track_evidence.get("final_crops_per_attribute", 3))))
                ),
                "minimum_original_width": max(1, int(vehicle_attribute_body_type.get("minimum_original_width", 120))),
                "minimum_original_height": max(1, int(vehicle_attribute_body_type.get("minimum_original_height", 100))),
                "preferred_original_width": max(1, int(vehicle_attribute_body_type.get("preferred_original_width", 240))),
                "preferred_original_height": max(1, int(vehicle_attribute_body_type.get("preferred_original_height", 160))),
                "allowed_labels": [
                    str(item).strip().upper()
                    for item in vehicle_attribute_body_type.get("allowed_labels", BODY_TYPE_ALLOWED_LABELS)
                    if str(item).strip()
                ],
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


@dataclass(slots=True, frozen=True)
class PreparedTrackEnrichment:
    track: LocalTrack
    request: TrackEnrichmentRequest
    vehicle_class_confidence: float | None
    started_iso: str
    started_monotonic: float
    evidence_source_name: str
    selected_items: tuple[Any, ...]
    candidate_crop_count: int
    eligible_crop_count: int
    preferred_crop_count: int
    readable_crop_count: int
    fallback_crop_count: int


@dataclass(slots=True, frozen=True)
class ColourEnrichmentJob:
    sequence_number: int
    local_track_id: str
    camera_id: str
    vehicle_class: str
    prepared: PreparedTrackEnrichment


@dataclass(slots=True, frozen=True)
class ColourEnrichmentWorkerResult:
    sequence_number: int
    local_track_id: str
    camera_id: str
    attribute_result: Any | None
    error_message: str | None
    worker_started_at: str
    worker_completed_at: str
    worker_duration_ms: float


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
        self.async_colour_config = dict(self.config.get("async_colour", {}) or {})
        self.async_colour_enabled = bool(
            self.async_colour_config.get("enabled", False)
            and self.config["vehicle_attributes"]["enabled"]
        )
        self.colour_queue_size = int(self.async_colour_config.get("queue_size", 100))
        self.colour_worker_count = int(self.async_colour_config.get("worker_count", 1))
        self.colour_queue_put_timeout_seconds = float(self.async_colour_config.get("queue_put_timeout_seconds", 0.1))
        if self.async_colour_enabled and self.colour_worker_count != 1:
            raise ValueError("vehicle_enrichment.async_colour.worker_count must be exactly 1 for this step.")
        self._colour_job_queue: queue.Queue[ColourEnrichmentJob | None] | None = None
        self._colour_result_queue: queue.SimpleQueue[ColourEnrichmentWorkerResult] | None = None
        self._colour_worker_thread: threading.Thread | None = None
        self._colour_job_sequence = 0
        self._pending_jobs_by_track: dict[str, PreparedTrackEnrichment] = {}
        self._enqueued_track_ids: set[str] = set()
        self._completed_track_ids: set[str] = set()
        if self.async_colour_enabled:
            self._colour_job_queue = queue.Queue(maxsize=self.colour_queue_size)
            self._colour_result_queue = queue.SimpleQueue()
            self._colour_worker_thread = threading.Thread(
                target=self._colour_worker_loop,
                name="colour-enrichment-worker",
                daemon=True,
            )
            self._colour_worker_thread.start()
        self._results_by_track: dict[str, TrackEnrichmentResult] = {}
        self._metrics: dict[str, Any] = {
            "completed_tracks_received": 0,
            "vehicle_tracks_finalized": 0,
            "vehicle_tracks_with_raw_crop": 0,
            "vehicle_tracks_with_preferred_crop": 0,
            "vehicle_tracks_using_low_resolution_fallback": 0,
            "vehicle_tracks_sent_to_florence": 0,
            "vehicle_tracks_with_zero_florence_calls": 0,
            "vehicle_tracks_with_raw_crop_but_zero_florence_calls": 0,
            "vehicle_tracks_with_valid_colour": 0,
            "vehicle_tracks_colour_unknown": 0,
            "tracks_with_existing_evidence": 0,
            "tracks_without_evidence": 0,
            "tracks_with_candidate_evidence": 0,
            "tracks_without_candidate_evidence": 0,
            "tracks_with_acceptable_crop": 0,
            "tracks_with_preferred_crop": 0,
            "tracks_with_no_florence_eligible_crop": 0,
            "evidence_items_received": 0,
            "capture_zone_crops_used_by_enrichment": 0,
            "capture_zone_fallback_to_existing_evidence": 0,
            "capture_zone_tracks_without_saved_evidence": 0,
            "capture_zone_motorcycle_florence_calls": 0,
            "capture_zone_motorcycle_valid_colours": 0,
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
            "car_tracks_total": 0,
            "car_tracks_with_body_type_crop": 0,
            "car_tracks_sent_to_body_type_florence": 0,
            "car_tracks_with_valid_body_type": 0,
            "car_tracks_body_type_unknown": 0,
            "body_type_label_distribution": {},
            "body_type_tracks_waited_for_completion": 0,
            "colour_tracks_waited_for_completion": 0,
            "early_classification_attempts": 0,
            "early_classification_successes": 0,
            "track_evidence_release_count": 0,
            "track_evidence_pending_count": 0,
            "colour_async_enabled": self.async_colour_enabled,
            "colour_worker_count": self.colour_worker_count if self.async_colour_enabled else 0,
            "colour_queue_count": 1 if self.async_colour_enabled else 0,
            "colour_queue_size": self.colour_queue_size if self.async_colour_enabled else 0,
            "colour_jobs_enqueued": 0,
            "colour_jobs_started": 0,
            "colour_jobs_completed": 0,
            "colour_jobs_failed": 0,
            "colour_jobs_duplicate_attempts": 0,
            "colour_jobs_lost": 0,
            "colour_queue_peak_depth": 0,
            "colour_queue_block_count": 0,
            "colour_queue_block_time_ms": 0.0,
            "colour_worker_busy_time_ms": 0.0,
            "colour_worker_shutdown_pending_jobs": 0,
            "plate_ocr_adapter_loads": 0,
            "plate_ocr_inference_calls": 0,
            "plate_ocr_skipped_no_plate": 0,
            "plate_ocr_skipped_low_quality": 0,
            "plate_ocr_attempts": 0,
            "gpu_memory_before_ocr_load_mb": 0.0,
            "gpu_memory_after_ocr_load_mb": 0.0,
            "vehicle_class_metrics": {},
        }

    @property
    def metrics(self) -> dict[str, Any]:
        payload = dict(self._metrics)
        payload["track_evidence_pending_count"] = len(self._pending_jobs_by_track)
        payload["crops_rejected_by_reason"] = dict(self._metrics["crops_rejected_by_reason"])
        payload["florence_original_size_distribution"] = dict(self._metrics["florence_original_size_distribution"])
        payload["vehicle_class_metrics"] = dict(self._metrics["vehicle_class_metrics"])
        payload["body_type_label_distribution"] = dict(self._metrics["body_type_label_distribution"])
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
        payload["body_type_average_inference_ms"] = float(self.vehicle_attribute_flow.metrics.get("vehicle_attribute_average_body_inference_time_ms", 0.0) or 0.0)
        payload["florence_model_instances"] = 1 if self.config["vehicle_attributes"]["enabled"] else 0
        payload["florence_concurrent_calls_possible"] = False if self.async_colour_enabled else False
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

        if not self.async_colour_enabled:
            results: list[TrackEnrichmentResult] = []
            for track in tracks:
                self._metrics["completed_tracks_received"] += 1
                self._metrics["vehicle_tracks_finalized"] += 1
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

        results: list[TrackEnrichmentResult] = []
        for track in tracks:
            self._metrics["completed_tracks_received"] += 1
            self._metrics["vehicle_tracks_finalized"] += 1
            self._metrics["current_in_memory_tracks"] += 1
            started_monotonic = time.perf_counter()
            started_iso = datetime.now(timezone.utc).isoformat()
            local_track_id = str(track.local_track_id)
            if local_track_id in self._enqueued_track_ids or local_track_id in self._completed_track_ids:
                self._metrics["colour_jobs_duplicate_attempts"] += 1
                continue
            try:
                prepared = self._prepare_track_for_async_colour(
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
                continue
            if isinstance(prepared, TrackEnrichmentResult):
                self._results_by_track[track.local_track_id] = prepared
                self._completed_track_ids.add(local_track_id)
                self._metrics["enrichment_results_written"] += 1
                results.append(prepared)
                self._cleanup_track(track.local_track_id)
                continue
            self._enqueue_colour_job(prepared)
        results.extend(self.drain_completed_results())
        return results

    def drain_completed_results(self) -> list[TrackEnrichmentResult]:
        if not self.async_colour_enabled or self._colour_result_queue is None:
            return []
        results: list[TrackEnrichmentResult] = []
        while True:
            try:
                worker_result = self._colour_result_queue.get_nowait()
            except queue.Empty:
                break
            results.append(self._finish_async_result(worker_result))
        return results

    def finalize_async_colour(self) -> list[TrackEnrichmentResult]:
        if not self.async_colour_enabled:
            return []
        if self._colour_job_queue is not None:
            self._colour_job_queue.join()
            self._colour_job_queue.put(None)
        if self._colour_worker_thread is not None:
            self._colour_worker_thread.join(timeout=30.0)
        results = self.drain_completed_results()
        self._metrics["colour_worker_shutdown_pending_jobs"] = len(self._pending_jobs_by_track)
        return results

    def _prepare_track_for_async_colour(
        self,
        track: LocalTrack,
        evidence_records: list[TrackEvidence | dict[str, Any]],
        started_iso: str,
        started_monotonic: float,
    ) -> PreparedTrackEnrichment | TrackEnrichmentResult:
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

        evidence_source_name, adapted = self._select_adapted_evidence(track, evidence_records)
        raw_track_fallback_items = self._load_raw_track_crop_fallback_items(track)
        adapted = self._merge_with_raw_track_crop_fallbacks(adapted, raw_track_fallback_items)
        self._metrics["evidence_items_received"] += len(adapted)
        if adapted:
            self._metrics["tracks_with_existing_evidence"] += 1
            if evidence_source_name == "capture_zone":
                self._metrics["capture_zone_crops_used_by_enrichment"] += len(adapted)
        else:
            self._metrics["tracks_without_evidence"] += 1
            if evidence_source_name == "capture_zone":
                self._metrics["capture_zone_tracks_without_saved_evidence"] += 1
            return self._build_base_result(
                track=track,
                vehicle_class_confidence=vehicle_class_confidence,
                status=ENRICHMENT_STATUS_NO_EVIDENCE,
                evidence_used=[],
                started_iso=started_iso,
                started_monotonic=started_monotonic,
                errors=[],
            )
        self.logger.info(
            "Evidence zone enrichment source camera=%s track=%s source=%s selected=%s",
            track.camera_id,
            track.local_track_id,
            evidence_source_name,
            len(adapted),
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
        readable_crop_count = len([item for item in selected if getattr(item, "readable_crop", False)])
        eligible_crop_count = len(
            [
                item
                for item in selected
                if getattr(item, "florence_eligible_for_body_type", False) or getattr(item, "florence_eligible_for_colour", False)
            ]
        )
        preferred_crop_count = len([item for item in selected if getattr(item, "resolution_tier", "") == "preferred"])
        fallback_crop_count = len([item for item in selected if str(getattr(item, "colour_selection_tier", "") or "") == "low_resolution_fallback"])
        if eligible_crop_count > 0:
            self._metrics["tracks_with_acceptable_crop"] += 1
        else:
            self._metrics["tracks_with_no_florence_eligible_crop"] += 1
        if preferred_crop_count > 0:
            self._metrics["tracks_with_preferred_crop"] += 1
            self._metrics["vehicle_tracks_with_preferred_crop"] += 1
        if scored:
            self._metrics["vehicle_tracks_with_raw_crop"] += 1
        if fallback_crop_count > 0:
            self._metrics["vehicle_tracks_using_low_resolution_fallback"] += 1

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
        return PreparedTrackEnrichment(
            track=track,
            request=request,
            vehicle_class_confidence=vehicle_class_confidence,
            started_iso=started_iso,
            started_monotonic=started_monotonic,
            evidence_source_name=evidence_source_name,
            selected_items=tuple(selected),
            candidate_crop_count=len(scored),
            eligible_crop_count=eligible_crop_count,
            preferred_crop_count=preferred_crop_count,
            readable_crop_count=readable_crop_count,
            fallback_crop_count=fallback_crop_count,
        )

    def _enqueue_colour_job(self, prepared: PreparedTrackEnrichment) -> None:
        if self._colour_job_queue is None:
            raise RuntimeError("Colour job queue is not initialized.")
        self._colour_job_sequence += 1
        local_track_id = str(prepared.track.local_track_id)
        job = ColourEnrichmentJob(
            sequence_number=self._colour_job_sequence,
            local_track_id=local_track_id,
            camera_id=str(prepared.track.camera_id),
            vehicle_class=str(prepared.track.final_class or "UNKNOWN").upper(),
            prepared=prepared,
        )
        queued = False
        blocked_started_at: float | None = None
        while not queued:
            try:
                self._colour_job_queue.put(job, timeout=self.colour_queue_put_timeout_seconds)
                queued = True
            except queue.Full:
                self._metrics["colour_queue_block_count"] += 1
                if blocked_started_at is None:
                    blocked_started_at = time.perf_counter()
        if blocked_started_at is not None:
            self._metrics["colour_queue_block_time_ms"] += float((time.perf_counter() - blocked_started_at) * 1000.0)
        self._pending_jobs_by_track[local_track_id] = prepared
        self._enqueued_track_ids.add(local_track_id)
        self._metrics["colour_jobs_enqueued"] += 1
        self._metrics["colour_queue_peak_depth"] = max(
            int(self._metrics["colour_queue_peak_depth"]),
            self._colour_job_queue.qsize(),
        )
        self.logger.info(
            "Colour job enqueued track=%s camera=%s queue_depth=%s",
            local_track_id,
            prepared.track.camera_id,
            self._colour_job_queue.qsize(),
        )

    def _colour_worker_loop(self) -> None:
        if self._colour_job_queue is None or self._colour_result_queue is None:
            return
        while True:
            job = self._colour_job_queue.get()
            if job is None:
                self._colour_job_queue.task_done()
                break
            worker_started_iso = datetime.now(timezone.utc).isoformat()
            started_monotonic = time.perf_counter()
            error_message: str | None = None
            attribute_result: Any | None = None
            try:
                self.logger.info("Colour worker started track=%s camera=%s", job.local_track_id, job.camera_id)
                attribute_result = self.vehicle_attribute_flow.classify(job.prepared.request)
            except Exception as exc:
                error_message = str(exc)
                self.logger.exception("Colour worker failed track=%s", job.local_track_id)
            completed_iso = datetime.now(timezone.utc).isoformat()
            duration_ms = float((time.perf_counter() - started_monotonic) * 1000.0)
            self.logger.info(
                "Colour worker completed track=%s camera=%s duration_ms=%.2f status=%s",
                job.local_track_id,
                job.camera_id,
                duration_ms,
                "failed" if error_message else "completed",
            )
            self._colour_result_queue.put(
                ColourEnrichmentWorkerResult(
                    sequence_number=job.sequence_number,
                    local_track_id=job.local_track_id,
                    camera_id=job.camera_id,
                    attribute_result=attribute_result,
                    error_message=error_message,
                    worker_started_at=worker_started_iso,
                    worker_completed_at=completed_iso,
                    worker_duration_ms=duration_ms,
                )
            )
            self._colour_job_queue.task_done()

    def _finish_async_result(self, worker_result: ColourEnrichmentWorkerResult) -> TrackEnrichmentResult:
        prepared = self._pending_jobs_by_track.pop(worker_result.local_track_id, None)
        self._enqueued_track_ids.discard(worker_result.local_track_id)
        if prepared is None:
            self._metrics["colour_jobs_lost"] += 1
            raise RuntimeError(f"Missing prepared async colour job for track {worker_result.local_track_id}")
        self._metrics["colour_jobs_started"] += 1
        self._metrics["colour_worker_busy_time_ms"] += float(worker_result.worker_duration_ms)
        if worker_result.error_message:
            self._metrics["colour_jobs_failed"] += 1
            self._metrics["enrichment_failures"] += 1
            if not self.config["fail_open"]:
                raise RuntimeError(worker_result.error_message)
            result = self._build_error_result(
                prepared.track,
                prepared.started_iso,
                prepared.started_monotonic,
                RuntimeError(worker_result.error_message),
            )
        else:
            self._metrics["colour_jobs_completed"] += 1
            result = self._build_async_vehicle_attribute_result(prepared, worker_result.attribute_result)
        self._results_by_track[result.local_track_id] = result
        self._completed_track_ids.add(result.local_track_id)
        self._metrics["enrichment_results_written"] += 1
        self._cleanup_track(result.local_track_id)
        return result

    def _build_async_vehicle_attribute_result(self, prepared: PreparedTrackEnrichment, attribute_result: Any) -> TrackEnrichmentResult:
        track = prepared.track
        selected = list(prepared.selected_items)
        body_type_result = attribute_result.body_type
        colour_result = attribute_result.colour
        crop_level_captions = [
            {
                "crop_path": row["vehicle_crop_path"],
                "frame_index": row["frame_index"],
                "caption": row.get("colour_post_processed_response") or row.get("body_type_post_processed_response") or "",
                "raw_response": row.get("colour_raw_response") or row.get("body_type_raw_response") or "",
                "post_processed_response": row.get("colour_post_processed_response") or row.get("body_type_post_processed_response") or "",
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
                "raw_response": row.get("body_type_raw_response"),
                "task_token": row.get("body_type_task_token"),
                "prompt": row.get("body_type_prompt"),
                "effective_processor_text": row.get("body_type_effective_processor_text"),
                "inference_time_ms": row.get("body_type_inference_time_ms"),
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
                "selection_tier": row.get("selection_tier"),
                "raw_response": row.get("colour_raw_response"),
                "task_token": row.get("colour_task_token"),
                "prompt": row.get("colour_prompt"),
                "effective_processor_text": row.get("colour_effective_processor_text"),
                "inference_time_ms": row.get("colour_inference_time_ms"),
            }
            for row in attribute_result.crop_level_rows
        ]
        self._apply_prediction_selection_metadata(selected, body_type_result.predictions, "body_type")
        self._apply_prediction_selection_metadata(selected, colour_result.predictions, "colour")
        self._metrics["body_type_selected_crop_count"] += len(body_type_result.predictions)
        self._metrics["colour_selected_crop_count"] += len(colour_result.predictions)
        if str(track.final_class or "").upper() == "CAR":
            self._metrics["car_tracks_total"] += 1
            self._metrics["car_tracks_with_body_type_crop"] += int(attribute_result.body_type_selected_crop_count > 0)
            self._metrics["car_tracks_sent_to_body_type_florence"] += int(attribute_result.body_type_florence_call_count > 0)
            self._metrics["car_tracks_with_valid_body_type"] += int(str(body_type_result.label or "").upper() not in {"", "UNKNOWN"})
            self._metrics["car_tracks_body_type_unknown"] += int(str(body_type_result.label or "").upper() in {"", "UNKNOWN"})
        label_key = str(body_type_result.label or "UNKNOWN").upper()
        self._metrics["body_type_label_distribution"][label_key] = int(self._metrics["body_type_label_distribution"].get(label_key, 0)) + 1
        self._metrics["body_type_tracks_waited_for_completion"] += 1
        self._metrics["colour_tracks_waited_for_completion"] += 1
        colour_selection_tier = self._resolve_colour_selection_tier(colour_result.predictions, selected)
        if colour_result.predictions:
            self._metrics["vehicle_tracks_sent_to_florence"] += 1
        else:
            self._metrics["vehicle_tracks_with_zero_florence_calls"] += 1
            if prepared.candidate_crop_count > 0 and prepared.readable_crop_count > 0:
                self._metrics["vehicle_tracks_with_raw_crop_but_zero_florence_calls"] += 1
        if str(colour_result.label or "").upper() not in {"", "UNKNOWN"}:
            self._metrics["vehicle_tracks_with_valid_colour"] += 1
        else:
            self._metrics["vehicle_tracks_colour_unknown"] += 1
        self._record_vehicle_class_metrics(
            vehicle_class=str(track.final_class or "UNKNOWN").upper(),
            has_raw_crop=prepared.candidate_crop_count > 0,
            sent_to_florence=bool(colour_result.predictions),
            used_fallback=prepared.fallback_crop_count > 0 or colour_selection_tier == "low_resolution_fallback",
            valid_colour=str(colour_result.label or "").upper() not in {"", "UNKNOWN"},
        )
        return self._build_base_result(
            track=track,
            vehicle_class_confidence=prepared.vehicle_class_confidence,
            status=ENRICHMENT_STATUS_COMPLETED if selected else ENRICHMENT_STATUS_NO_EVIDENCE,
            evidence_used=selected,
            started_iso=prepared.started_iso,
            started_monotonic=prepared.started_monotonic,
            errors=[],
            body_type_result=body_type_result,
            colour_result=colour_result,
            vehicle_make=None,
            vehicle_model=None,
            plate_detected=False,
            plate_text=None,
            candidate_crop_count=prepared.candidate_crop_count,
            eligible_crop_count=prepared.eligible_crop_count,
            preferred_crop_count=prepared.preferred_crop_count,
            readable_crop_count=prepared.readable_crop_count,
            fallback_crop_count=prepared.fallback_crop_count,
            selected_colour_crop_count=len(colour_result.predictions),
            colour_selection_tier=colour_selection_tier,
            selected_body_type_crop_paths=[str(item.source_crop_path) for item in body_type_result.predictions if item.source_crop_path],
            selected_colour_crop_paths=[str(item.source_crop_path) for item in colour_result.predictions if item.source_crop_path],
            body_type_eligible=attribute_result.body_type_eligible,
            body_type_candidate_crop_count=attribute_result.body_type_candidate_crop_count,
            body_type_selected_crop_count=attribute_result.body_type_selected_crop_count,
            body_type_florence_call_count=attribute_result.body_type_florence_call_count,
            body_type_valid_prediction_count=attribute_result.body_type_valid_prediction_count,
            body_type_failure_reason=attribute_result.body_type_failure_reason,
            florence_mode=str(self.config["florence_mode"]).strip().lower(),
            adapter_loaded=attribute_result.adapter_loaded,
            selected_crop_paths=[str(getattr(item, "vehicle_crop_path", "")) for item in selected if getattr(item, "vehicle_crop_path", None)],
            crop_level_captions=crop_level_captions,
            crop_level_body_types=crop_level_body_types,
            crop_level_colours=crop_level_colours,
            final_body_type_reason=body_type_result.aggregation_reason or body_type_result.reason,
            final_colour_reason=colour_result.aggregation_reason or colour_result.reason,
            caption_inference_count=attribute_result.inference_count,
            vehicle_attribute_raw_responses=attribute_result.raw_responses,
            vehicle_attribute_selected_crop_paths=[str(item.get("vehicle_crop_path")) for item in attribute_result.crop_level_rows],
            vehicle_attribute_inference_count=attribute_result.inference_count,
            attribute_backend="base_florence",
            plate_ocr_backend=None,
            plate_ocr_attempted=False,
            plate_ocr_raw_response=None,
            plate_ocr_reason="plate_scope_frozen",
            plate_quality_status="plate_scope_frozen",
            comparison_payload=None,
            classification_trigger="track_completion_async_colour",
            final_reason=body_type_result.aggregation_reason or body_type_result.reason or colour_result.aggregation_reason or colour_result.reason,
        )

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

        evidence_source_name, adapted = self._select_adapted_evidence(track, evidence_records)
        raw_track_fallback_items = self._load_raw_track_crop_fallback_items(track)
        adapted = self._merge_with_raw_track_crop_fallbacks(adapted, raw_track_fallback_items)
        self._metrics["evidence_items_received"] += len(adapted)
        if adapted:
            self._metrics["tracks_with_existing_evidence"] += 1
            if evidence_source_name == "capture_zone":
                self._metrics["capture_zone_crops_used_by_enrichment"] += len(adapted)
        else:
            self._metrics["tracks_without_evidence"] += 1
            if evidence_source_name == "capture_zone":
                self._metrics["capture_zone_tracks_without_saved_evidence"] += 1
            return self._build_base_result(
                track=track,
                vehicle_class_confidence=vehicle_class_confidence,
                status=ENRICHMENT_STATUS_NO_EVIDENCE,
                evidence_used=[],
                started_iso=started_iso,
                started_monotonic=started_monotonic,
                errors=[],
            )
        self.logger.info(
            "Evidence zone enrichment source camera=%s track=%s source=%s selected=%s",
            track.camera_id,
            track.local_track_id,
            evidence_source_name,
            len(adapted),
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
        readable_crop_count = len([item for item in selected if getattr(item, "readable_crop", False)])
        eligible_crop_count = len(
            [
                item
                for item in selected
                if getattr(item, "florence_eligible_for_body_type", False) or getattr(item, "florence_eligible_for_colour", False)
            ]
        )
        preferred_crop_count = len([item for item in selected if getattr(item, "resolution_tier", "") == "preferred"])
        fallback_crop_count = len([item for item in selected if str(getattr(item, "colour_selection_tier", "") or "") == "low_resolution_fallback"])
        if eligible_crop_count > 0:
            self._metrics["tracks_with_acceptable_crop"] += 1
        else:
            self._metrics["tracks_with_no_florence_eligible_crop"] += 1
        if preferred_crop_count > 0:
            self._metrics["tracks_with_preferred_crop"] += 1
            self._metrics["vehicle_tracks_with_preferred_crop"] += 1
        if scored:
            self._metrics["vehicle_tracks_with_raw_crop"] += 1
        if fallback_crop_count > 0:
            self._metrics["vehicle_tracks_using_low_resolution_fallback"] += 1
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
                    "caption": row.get("colour_post_processed_response") or row.get("body_type_post_processed_response") or "",
                    "raw_response": row.get("colour_raw_response") or row.get("body_type_raw_response") or "",
                    "post_processed_response": row.get("colour_post_processed_response") or row.get("body_type_post_processed_response") or "",
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
                    "raw_response": row.get("body_type_raw_response"),
                    "task_token": row.get("body_type_task_token"),
                    "prompt": row.get("body_type_prompt"),
                    "effective_processor_text": row.get("body_type_effective_processor_text"),
                    "inference_time_ms": row.get("body_type_inference_time_ms"),
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
                    "selection_tier": row.get("selection_tier"),
                    "raw_response": row.get("colour_raw_response"),
                    "task_token": row.get("colour_task_token"),
                    "prompt": row.get("colour_prompt"),
                    "effective_processor_text": row.get("colour_effective_processor_text"),
                    "inference_time_ms": row.get("colour_inference_time_ms"),
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
        if evidence_source_name == "capture_zone" and str(track.final_class or "").upper() == "MOTORCYCLE":
            if colour_result.predictions:
                self._metrics["capture_zone_motorcycle_florence_calls"] += 1
            if str(colour_result.label or "").upper() not in {"", "UNKNOWN"}:
                self._metrics["capture_zone_motorcycle_valid_colours"] += 1
        self._metrics["body_type_selected_crop_count"] += len(body_type_result.predictions)
        self._metrics["colour_selected_crop_count"] += len(colour_result.predictions)
        if str(track.final_class or "").upper() == "CAR":
            self._metrics["car_tracks_total"] += 1
            if self.config["vehicle_attributes"]["enabled"]:
                self._metrics["car_tracks_with_body_type_crop"] += int(attribute_result.body_type_selected_crop_count > 0)
                self._metrics["car_tracks_sent_to_body_type_florence"] += int(attribute_result.body_type_florence_call_count > 0)
                self._metrics["car_tracks_with_valid_body_type"] += int(str(body_type_result.label or "").upper() not in {"", "UNKNOWN"})
                self._metrics["car_tracks_body_type_unknown"] += int(str(body_type_result.label or "").upper() in {"", "UNKNOWN"})
        label_key = str(body_type_result.label or "UNKNOWN").upper()
        self._metrics["body_type_label_distribution"][label_key] = int(self._metrics["body_type_label_distribution"].get(label_key, 0)) + 1
        self._metrics["body_type_tracks_waited_for_completion"] += 1
        self._metrics["colour_tracks_waited_for_completion"] += 1
        colour_selection_tier = self._resolve_colour_selection_tier(colour_result.predictions, selected)
        if colour_result.predictions:
            self._metrics["vehicle_tracks_sent_to_florence"] += 1
        else:
            self._metrics["vehicle_tracks_with_zero_florence_calls"] += 1
            if scored and readable_crop_count > 0:
                self._metrics["vehicle_tracks_with_raw_crop_but_zero_florence_calls"] += 1
        if str(colour_result.label or "").upper() not in {"", "UNKNOWN"}:
            self._metrics["vehicle_tracks_with_valid_colour"] += 1
        else:
            self._metrics["vehicle_tracks_colour_unknown"] += 1
        self._record_vehicle_class_metrics(
            vehicle_class=str(track.final_class or "UNKNOWN").upper(),
            has_raw_crop=bool(scored),
            sent_to_florence=bool(colour_result.predictions),
            used_fallback=fallback_crop_count > 0 or colour_selection_tier == "low_resolution_fallback",
            valid_colour=str(colour_result.label or "").upper() not in {"", "UNKNOWN"},
        )
        make_model_enabled = bool(getattr(self.make_model_classifier, "enabled", False))
        plate_detection_enabled = bool(getattr(self.plate_detector, "enabled", False))
        plate_ocr_enabled = bool(getattr(self.plate_ocr_engine, "enabled", False))
        if make_model_enabled:
            make_model_result = self.make_model_classifier.classify(request)
        else:
            make_model_result = VehicleMakeModelResult(
                make=None,
                model=None,
                predictions=[],
                status=ATTRIBUTE_STATUS_DISABLED,
                source="make_model.classifier",
                reason="make_model_scope_frozen",
            )
        if plate_detection_enabled and selected:
            plate_detection_result = self.plate_detector.detect(selected[0])
            plate_quality_result = self.plate_quality_validator.validate(None)
        elif plate_detection_enabled:
            plate_detection_result = PlateDetectionResult(
                detected=False,
                predictions=[],
                status="skipped",
                source="plate.detector",
                reason="no_selected_vehicle_crop",
            )
            plate_quality_result = self.plate_quality_validator.validate(None)
        else:
            plate_detection_result = PlateDetectionResult(
                detected=False,
                predictions=[],
                status=ATTRIBUTE_STATUS_DISABLED,
                source="plate.detector",
                reason="plate_scope_frozen",
            )
            plate_quality_result = PlateQualityResult(
                acceptable=None,
                predictions=[],
                status=ATTRIBUTE_STATUS_DISABLED,
                source="plate.quality_validator",
                reason="plate_scope_frozen",
            )
        if plate_detection_enabled and not getattr(plate_detection_result, "detected", False):
            self._metrics["plate_ocr_skipped_no_plate"] += 1
            plate_ocr_result = PlateOCRResult(text=None, predictions=[], status="skipped", source="plate.ocr_engine", reason="no_plate_detected")
        elif plate_detection_enabled and plate_ocr_enabled:
            self._metrics["plate_ocr_attempts"] += 1
            self._metrics["gpu_memory_before_ocr_load_mb"] = float(self.ocr_mukul_backend.metrics.get("gpu_memory_allocated_mb") or 0.0)
            plate_ocr_result = self.plate_ocr_engine.recognize(None)
            self._metrics["gpu_memory_after_ocr_load_mb"] = float(self.ocr_mukul_backend.metrics.get("gpu_memory_allocated_mb") or 0.0)
            if plate_ocr_result.status == "completed":
                self._metrics["plate_ocr_inference_calls"] += 1
        elif plate_detection_enabled:
            plate_ocr_result = PlateOCRResult(text=None, predictions=[], status="disabled", source="plate.ocr_engine", reason="plate_ocr_disabled")
        else:
            plate_ocr_result = PlateOCRResult(text=None, predictions=[], status="disabled", source="plate.ocr_engine", reason="plate_scope_frozen")
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
            readable_crop_count=readable_crop_count,
            fallback_crop_count=fallback_crop_count,
            selected_colour_crop_count=len(colour_result.predictions),
            colour_selection_tier=colour_selection_tier,
            selected_body_type_crop_paths=[str(item.source_crop_path) for item in body_type_result.predictions if item.source_crop_path],
            selected_colour_crop_paths=[str(item.source_crop_path) for item in colour_result.predictions if item.source_crop_path],
            body_type_eligible=attribute_result.body_type_eligible if self.config["vehicle_attributes"]["enabled"] else str(track.final_class or "").upper() == "CAR",
            body_type_candidate_crop_count=attribute_result.body_type_candidate_crop_count if self.config["vehicle_attributes"]["enabled"] else 0,
            body_type_selected_crop_count=attribute_result.body_type_selected_crop_count if self.config["vehicle_attributes"]["enabled"] else len(body_type_result.predictions),
            body_type_florence_call_count=attribute_result.body_type_florence_call_count if self.config["vehicle_attributes"]["enabled"] else len(body_type_result.predictions),
            body_type_valid_prediction_count=attribute_result.body_type_valid_prediction_count if self.config["vehicle_attributes"]["enabled"] else sum(1 for item in body_type_result.predictions if str(item.label or "").upper() not in {"", "UNKNOWN"}),
            body_type_failure_reason=attribute_result.body_type_failure_reason if self.config["vehicle_attributes"]["enabled"] else (body_type_result.aggregation_reason or body_type_result.reason),
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

    def _select_adapted_evidence(
        self,
        track: LocalTrack,
        evidence_records: list[TrackEvidence | dict[str, Any]],
    ) -> tuple[str, list[Any]]:
        configured_source = str(self.config["evidence"].get("source", "existing_track_evidence")).strip() or "existing_track_evidence"
        grouped: dict[str, list[TrackEvidence | dict[str, Any]]] = defaultdict(list)
        for record in evidence_records:
            if isinstance(record, dict):
                source_name = str(record.get("evidence_source", "existing_track_evidence")).strip() or "existing_track_evidence"
            else:
                source_name = "existing_track_evidence"
            grouped[source_name].append(record)

        if configured_source == "capture_zone":
            return "capture_zone", self.adapter.adapt_track(track, grouped.get("capture_zone", []))
        if configured_source == "capture_zone_with_existing_fallback":
            capture_zone_items = self.adapter.adapt_track(track, grouped.get("capture_zone", []))
            if capture_zone_items:
                return "capture_zone", capture_zone_items
            self._metrics["capture_zone_fallback_to_existing_evidence"] += 1
            return "existing_track_evidence", self.adapter.adapt_track(track, grouped.get("existing_track_evidence", []))
        return "existing_track_evidence", self.adapter.adapt_track(track, grouped.get("existing_track_evidence", []))

    def _load_raw_track_crop_fallback_items(self, track: LocalTrack) -> list[Any]:
        track_name = str(track.local_track_id).split(":")[-1]
        track_directory = self.output_manager.track_crops_directory / track.camera_id / track_name
        if not track_directory.exists():
            return []
        items: list[Any] = []
        for crop_path in sorted(track_directory.glob("frame_*.jpg")):
            image = cv2.imread(str(crop_path))
            if image is None or image.size == 0:
                continue
            height, width = image.shape[:2]
            try:
                frame_number = int(crop_path.stem.split("_")[-1])
            except ValueError:
                frame_number = 0
            items.append(
                self.adapter._normalize_record(
                    track,
                    {
                        "local_track_id": track.local_track_id,
                        "camera_id": track.camera_id,
                        "native_tracker_id": track.native_tracker_id,
                        "tracker_namespace": track.tracker_namespace,
                        "role": "RAW_TRACK_CROP",
                        "frame_number": frame_number,
                        "timestamp_seconds": float(getattr(track, "first_timestamp_seconds", 0.0)),
                        "raw_class_name": track.final_class,
                        "final_class": track.final_class,
                        "confidence": 0.0,
                        "crop_path": str(crop_path),
                        "annotated_frame_path": str(crop_path),
                        "bbox_xyxy": [0.0, 0.0, float(width), float(height)],
                        "original_bbox": [0.0, 0.0, float(width), float(height)],
                        "expanded_crop_bbox": [0.0, 0.0, float(width), float(height)],
                        "context_padding_ratio": 0.0,
                        "source_frame_width": width,
                        "source_frame_height": height,
                        "original_crop_width": width,
                        "original_crop_height": height,
                        "sharpness_score": 0.0,
                        "best_overall_score": 0.0,
                        "evidence_source": "raw_track_crop_fallback",
                    },
                )
            )
        return [item for item in items if item is not None]

    @staticmethod
    def _merge_with_raw_track_crop_fallbacks(adapted: list[Any], raw_items: list[Any]) -> list[Any]:
        if not raw_items:
            return adapted
        seen = {
            (int(getattr(item, "frame_number", -1)), str(getattr(item, "vehicle_crop_path", "") or ""))
            for item in adapted
        }
        merged = list(adapted)
        for item in raw_items:
            key = (int(getattr(item, "frame_number", -1)), str(getattr(item, "vehicle_crop_path", "") or ""))
            if key in seen:
                continue
            seen.add(key)
            merged.append(item)
        return merged

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
        readable_crop_count: int = 0,
        fallback_crop_count: int = 0,
        selected_colour_crop_count: int = 0,
        colour_selection_tier: str | None = None,
        selected_body_type_crop_paths: list[str] | None = None,
        selected_colour_crop_paths: list[str] | None = None,
        body_type_eligible: bool | None = None,
        body_type_candidate_crop_count: int = 0,
        body_type_selected_crop_count: int = 0,
        body_type_florence_call_count: int = 0,
        body_type_valid_prediction_count: int = 0,
        body_type_failure_reason: str | None = None,
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
            readable_crop_count=readable_crop_count,
            fallback_crop_count=fallback_crop_count,
            selected_colour_crop_count=selected_colour_crop_count,
            colour_selection_tier=colour_selection_tier,
            selected_body_type_crop_paths=list(selected_body_type_crop_paths or []),
            selected_colour_crop_paths=list(selected_colour_crop_paths or []),
            body_type_eligible=body_type_eligible,
            body_type_candidate_crop_count=body_type_candidate_crop_count,
            body_type_selected_crop_count=body_type_selected_crop_count,
            body_type_florence_call_count=body_type_florence_call_count,
            body_type_valid_prediction_count=body_type_valid_prediction_count,
            body_type_failure_reason=body_type_failure_reason,
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
        preferred_candidates = [item for item in items if self._is_preferred_colour_candidate(item)]
        fallback_candidates = [item for item in items if self._is_readable_colour_candidate(item)]
        rejected = [item for item in items if not self._is_readable_colour_candidate(item)]
        ordered_preferred = self._rank_evidence_items(preferred_candidates)
        ordered_fallback = self._rank_evidence_items(fallback_candidates)

        if ordered_preferred:
            selected = ordered_preferred[:best_count]
            for item in selected:
                if not getattr(item, "colour_selection_tier", None):
                    item.colour_selection_tier = str(getattr(item, "resolution_tier", "") or "acceptable")
            rejected.extend(ordered_preferred[best_count:])
            for item in ordered_preferred[best_count:]:
                item.rejection_reasons.append("best_crop_limit_exceeded")
            rejected.extend([item for item in ordered_fallback if item not in selected and item not in ordered_preferred])
            return selected, rejected

        selected = ordered_fallback[:best_count]
        for item in selected:
            if str(getattr(item, "colour_selection_tier", "") or "") != "preferred":
                item.colour_selection_tier = "low_resolution_fallback"
        rejected.extend(ordered_fallback[best_count:])
        for item in ordered_fallback[best_count:]:
            item.rejection_reasons.append("best_crop_limit_exceeded")
        return selected, rejected

    @staticmethod
    def _rank_evidence_items(items: list[Any]) -> list[Any]:
        return sorted(
            items,
            key=lambda item: (
                1 if getattr(item, "evidence_source", "") == "capture_zone" else 0,
                1 if getattr(item, "resolution_tier", "") == "preferred" else 0,
                float(getattr(item, "ranking_score", 0.0)),
                float(getattr(item, "quality_score", 0.0)),
                float(getattr(item, "sharpness_score", 0.0)),
                float(getattr(item, "detection_confidence", 0.0)),
                -float(getattr(item, "clipping_ratio", 0.0)),
                float(getattr(item, "original_crop_area", 0.0) or (getattr(item, "original_crop_width", 0) or 0) * (getattr(item, "original_crop_height", 0) or 0)),
                float(getattr(item, "original_crop_height", 0)),
                float(getattr(item, "original_crop_width", 0)),
                -abs(float(getattr(item, "brightness_score", 0.0)) - 140.0),
                float(getattr(item, "frame_number", 0)),
            ),
            reverse=True,
        )

    @staticmethod
    def _is_readable_colour_candidate(item: Any) -> bool:
        crop_path = Path(str(getattr(item, "vehicle_crop_path", "") or ""))
        width = int(getattr(item, "original_crop_width", 0) or getattr(item, "crop_width", 0) or 0)
        height = int(getattr(item, "original_crop_height", 0) or getattr(item, "crop_height", 0) or 0)
        if not str(crop_path) or not crop_path.exists():
            return False
        if width <= 0 or height <= 0:
            return False
        if "invalid_bbox" in getattr(item, "rejection_reasons", []):
            return False
        if "missing_crop_image" in getattr(item, "rejection_reasons", []):
            return False
        if "empty_crop" in getattr(item, "rejection_reasons", []):
            return False
        return bool(getattr(item, "readable_crop", True))

    def _is_preferred_colour_candidate(self, item: Any) -> bool:
        return self._is_readable_colour_candidate(item) and str(getattr(item, "resolution_tier", "") or "") in {"acceptable", "preferred"}

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

    @staticmethod
    def _resolve_colour_selection_tier(predictions: list[Any], selected_items: list[Any]) -> str | None:
        by_path = {
            str(getattr(item, "vehicle_crop_path", "") or ""): str(getattr(item, "colour_selection_tier", "") or "")
            for item in selected_items
        }
        tiers = [by_path.get(str(getattr(prediction, "source_crop_path", "") or ""), "") for prediction in predictions]
        tiers = [tier for tier in tiers if tier]
        if not tiers:
            return None
        if "low_resolution_fallback" in tiers:
            return "low_resolution_fallback"
        if "acceptable" in tiers:
            return "acceptable"
        if "preferred" in tiers:
            return "preferred"
        return tiers[0]

    def _record_vehicle_class_metrics(self, *, vehicle_class: str, has_raw_crop: bool, sent_to_florence: bool, used_fallback: bool, valid_colour: bool) -> None:
        key = str(vehicle_class or "UNKNOWN").strip().lower()
        payload = dict(self._metrics["vehicle_class_metrics"].get(key, {}))
        payload["tracks_with_raw_crop"] = int(payload.get("tracks_with_raw_crop", 0)) + int(has_raw_crop)
        payload["tracks_sent_to_florence"] = int(payload.get("tracks_sent_to_florence", 0)) + int(sent_to_florence)
        payload["tracks_with_zero_florence_calls"] = int(payload.get("tracks_with_zero_florence_calls", 0)) + int(not sent_to_florence and has_raw_crop)
        payload["tracks_using_fallback"] = int(payload.get("tracks_using_fallback", 0)) + int(used_fallback)
        payload["tracks_with_valid_colour"] = int(payload.get("tracks_with_valid_colour", 0)) + int(valid_colour)
        payload["tracks_unknown"] = int(payload.get("tracks_unknown", 0)) + int(not valid_colour)
        self._metrics["vehicle_class_metrics"][key] = payload

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
