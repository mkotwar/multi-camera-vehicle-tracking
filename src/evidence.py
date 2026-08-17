from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .models import (
    ConfigurationError,
    EvidenceCandidate,
    FramePacket,
    LocalTrack,
    PipelineRuntimeError,
    TrackEvidence,
    TrackedDetection,
    TRACK_STATUS_COMPLETED,
    TRACK_STATUS_DISCARDED,
)
from .output_writer import RunOutputManager


EVIDENCE_ROLE_FIRST = "FIRST"
EVIDENCE_ROLE_MIDDLE = "MIDDLE"
EVIDENCE_ROLE_LAST = "LAST"
EVIDENCE_ROLE_HIGHEST_CONFIDENCE = "HIGHEST_CONFIDENCE"
EVIDENCE_ROLE_LARGEST = "LARGEST"
EVIDENCE_ROLE_SHARPEST = "SHARPEST"
EVIDENCE_ROLE_BEST_OVERALL = "BEST_OVERALL"
ALLOWED_EVIDENCE_ROLES = (
    EVIDENCE_ROLE_FIRST,
    EVIDENCE_ROLE_MIDDLE,
    EVIDENCE_ROLE_LAST,
    EVIDENCE_ROLE_HIGHEST_CONFIDENCE,
    EVIDENCE_ROLE_LARGEST,
    EVIDENCE_ROLE_SHARPEST,
    EVIDENCE_ROLE_BEST_OVERALL,
)


@dataclass(slots=True)
class _StoredCandidate:
    candidate: EvidenceCandidate
    frame_key: tuple[str, int]
    crop_bbox_xyxy: tuple[int, int, int, int]


@dataclass(slots=True)
class _CaptureZoneStoredCandidate:
    candidate: EvidenceCandidate
    crop_bbox_xyxy: tuple[int, int, int, int]
    crop_path: str
    trigger_x: float
    trigger_y: float
    zone_top: int
    zone_bottom: int


@dataclass(slots=True)
class _CaptureZoneTrackState:
    entered_zone: bool = False
    exited_zone: bool = False
    observation_count: int = 0
    last_position: str = "unknown"
    last_capture_frame: int | None = None
    retained_candidates: list[_CaptureZoneStoredCandidate] = field(default_factory=list)
    first_frame: int | None = None
    last_frame: int | None = None
    minimum_trigger_y: float | None = None
    maximum_trigger_y: float | None = None
    frame_of_max_trigger_y: int | None = None
    maximum_bbox_width: int = 0
    frame_of_max_bbox_width: int | None = None
    maximum_bbox_height: int = 0
    frame_of_max_bbox_height: int | None = None
    maximum_bbox_area: int = 0
    frame_of_max_bbox_area: int | None = None
    first_zone_entry_frame: int | None = None
    last_zone_frame: int | None = None
    zone_exit_frame: int | None = None
    zone_top_pixels: int | None = None
    zone_bottom_pixels: int | None = None
    capture_zone_candidate_count: int = 0
    capture_zone_retained_count: int = 0
    largest_saved_crop_width: int = 0
    largest_saved_crop_height: int = 0
    largest_saved_crop_frame: int | None = None
    class_counts: dict[str, int] = field(default_factory=dict)
    class_confidence_sums: dict[str, float] = field(default_factory=dict)
    stable_class_name: str = "unknown"
    source_frame_width: int = 0
    source_frame_height: int = 0


@dataclass(slots=True)
class _CaptureZoneGeometryRecord:
    camera_id: str
    local_track_id: str
    source_frame_width: int
    source_frame_height: int
    first_frame: int
    last_frame: int
    observation_count: int
    min_trigger_y: float
    max_trigger_y: float
    zone_top: int
    zone_bottom: int
    entered_zone: bool
    first_zone_entry_frame: int | None
    last_zone_frame: int | None
    zone_exit_frame: int | None
    max_bbox_width: int
    max_bbox_height: int
    max_bbox_area: int
    frame_of_max_trigger_y: int | None
    frame_of_max_bbox_width: int | None
    frame_of_max_bbox_height: int | None
    frame_of_max_bbox_area: int | None
    largest_saved_crop_width: int
    largest_saved_crop_height: int
    largest_saved_crop_frame: int | None
    capture_candidates: int
    retained_candidates: int
    geometry_status: str
    geometry_reason: str
    final_class: str
    stable_class_name: str
    completion_reason: str | None
    track_status: str
    evidence_eligible_zone_crop: bool
    florence_eligible_zone_crop: bool


class EvidenceCollector:
    def __init__(self, config: dict[str, Any], logger: Any, output_manager: RunOutputManager) -> None:
        self.logger = logger
        self.output_manager = output_manager
        self.config = self._validate_config(config.get("evidence", {}))
        self._confirmed_track_minimum_observations = int(dict(config.get("lifecycle", {}) or {}).get("minimum_observations", 3))
        track_class_config = dict(config.get("track_class", {}) or {})
        self._track_class_minimum_observations = int(track_class_config.get("minimum_observations", 3))
        self._track_class_minimum_winner_ratio = float(track_class_config.get("minimum_winner_ratio", 0.60))
        self._vehicle_enrichment_evidence_config = dict(dict(config.get("vehicle_enrichment", {}) or {}).get("evidence", {}) or {})
        self._vehicle_enrichment_florence_config = dict(dict(dict(config.get("vehicle_enrichment", {}) or {}).get("image_size_policy", {}) or {}).get("florence", {}) or {})
        debug_outputs = dict(config.get("debug_outputs", {}) or {})
        track_crops_debug = dict(debug_outputs.get("track_crops", {}) or {})
        self._debug_track_crops_enabled = bool(debug_outputs.get("enabled", False) and track_crops_debug.get("enabled", False))
        self._debug_track_crops_save_every_n_frames = max(1, int(track_crops_debug.get("save_every_n_frames", 3)))
        self._debug_track_crops_max_per_track = max(1, int(track_crops_debug.get("max_crops_per_track", 100)))
        self.enabled = bool(self.config["enabled"])
        self._track_candidates: dict[str, list[_StoredCandidate]] = {}
        self._capture_zone_candidates: dict[str, list[_CaptureZoneStoredCandidate]] = {}
        self._capture_zone_state: dict[str, _CaptureZoneTrackState] = {}
        self._capture_zone_index: list[dict[str, Any]] = []
        self._motorcycle_geometry_records: list[_CaptureZoneGeometryRecord] = []
        self._debug_track_crop_rows: list[dict[str, Any]] = []
        self._debug_track_crop_counts: dict[str, int] = {}
        self._frame_cache: dict[tuple[str, int], np.ndarray] = {}
        self._frame_ref_counts: dict[tuple[str, int], int] = {}
        self._evidence_index: list[dict[str, Any]] = []
        self._metrics = {
            "tracks_received": 0,
            "tracks_with_evidence": 0,
            "tracks_without_valid_evidence": 0,
            "candidate_observations": 0,
            "candidate_crops_seen": 0,
            "candidate_crops_retained": 0,
            "candidate_crops_rejected": 0,
            "candidate_crops_replaced": 0,
            "candidate_crops_deduplicated": 0,
            "candidate_pool_peak_per_track": 0,
            "candidate_pool_average_per_track": 0.0,
            "tracks_with_candidate_evidence": 0,
            "tracks_without_candidate_evidence": 0,
            "invalid_candidates": 0,
            "selected_evidence_records": 0,
            "unique_crop_files": 0,
            "unique_annotated_frame_files": 0,
            "role_counts": {},
            "tracks_by_camera": {},
            "saved_files_by_camera": {},
            "cache_peak_frames": 0,
            "evidence_cache_hits": 0,
            "evidence_cache_misses": 0,
            "evidence_cache_evictions": 0,
            "evidence_cache_eviction_skipped_referenced": 0,
            "cache_frames_released": 0,
            "cache_release_attempts": 0,
            "cache_release_deferred": 0,
            "missing_cache_frame_count": 0,
            "evidence_items_skipped_missing_frame": 0,
            "tracks_with_partial_evidence": 0,
            "pending_evidence_tracks_at_shutdown": 0,
            "pending_frame_reference_count": 0,
            "duplicate_frame_candidates_skipped": 0,
            "source_frame_resolution_counts": {},
            "evidence_crop_count": 0,
            "evidence_crop_width_min": 0,
            "evidence_crop_width_max": 0,
            "evidence_crop_width_average": 0.0,
            "evidence_crop_height_min": 0,
            "evidence_crop_height_max": 0,
            "evidence_crop_height_average": 0.0,
            "errors": [],
            "capture_zone_tracks_entered": 0,
            "capture_zone_candidate_attempts": 0,
            "capture_zone_candidates_saved": 0,
            "capture_zone_candidates_replaced": 0,
            "capture_zone_candidates_replaced_by_larger_crop": 0,
            "capture_zone_candidates_replaced_by_better_quality": 0,
            "capture_zone_tracks_with_saved_evidence": 0,
            "capture_zone_tracks_without_saved_evidence": 0,
            "capture_zone_invalid_bbox_count": 0,
            "capture_zone_too_small_count": 0,
            "capture_zone_candidates_too_small": 0,
            "capture_zone_candidates_rejected_by_class_threshold": 0,
            "capture_zone_duplicate_frame_suppressed": 0,
            "capture_zone_crops_used_by_enrichment": 0,
            "capture_zone_fallback_to_existing_evidence": 0,
            "capture_zone_missing_saved_crop": 0,
            "capture_zone_motorcycle_candidates": 0,
            "capture_zone_motorcycle_eligible_candidates": 0,
            "capture_zone_motorcycle_tracks_with_evidence": 0,
            "capture_zone_motorcycle_florence_calls": 0,
            "capture_zone_motorcycle_valid_colours": 0,
        }
        self._unique_crop_paths: set[str] = set()
        self._unique_annotated_paths: set[str] = set()
        self.logger.info(
            "EvidenceCollector initialized enabled=%s include_discarded_tracks=%s fail_pipeline_on_error=%s",
            self.enabled,
            self.config["include_discarded_tracks"],
            self.config["fail_pipeline_on_error"],
        )
        capture_zone = self.config["capture_zone"]
        self.logger.info(
            "Evidence capture zone initialized enabled=%s top_ratio=%.2f bottom_ratio=%.2f trigger_point=%s",
            capture_zone["enabled"],
            capture_zone["top_ratio"],
            capture_zone["bottom_ratio"],
            capture_zone["trigger_point"],
        )

    @property
    def evidence_index(self) -> list[dict[str, Any]]:
        return [dict(item) for item in self._evidence_index]

    @property
    def capture_zone_index(self) -> list[dict[str, Any]]:
        return [dict(item) for item in self._capture_zone_index]

    @property
    def metrics(self) -> dict[str, Any]:
        metrics = dict(self._metrics)
        metrics["role_counts"] = dict(self._metrics["role_counts"])
        metrics["tracks_by_camera"] = dict(self._metrics["tracks_by_camera"])
        metrics["saved_files_by_camera"] = dict(self._metrics["saved_files_by_camera"])
        metrics["source_frame_resolution_counts"] = dict(self._metrics["source_frame_resolution_counts"])
        metrics["errors"] = list(self._metrics["errors"])
        metrics["pending_evidence_tracks_at_shutdown"] = len(self._track_candidates)
        metrics["pending_frame_reference_count"] = int(sum(self._frame_ref_counts.values()))
        metrics["capture_zone_active_tracks"] = len(self._capture_zone_state)
        return metrics

    @property
    def motorcycle_geometry_records(self) -> list[dict[str, Any]]:
        return [asdict(item) for item in self._motorcycle_geometry_records]

    @property
    def debug_track_crop_rows(self) -> list[dict[str, Any]]:
        return [dict(item) for item in self._debug_track_crop_rows]

    def register_frame(self, frame_packet: FramePacket, tracked_detections: list[TrackedDetection]) -> None:
        if not self.enabled or not tracked_detections:
            return
        source_key = f"{int(frame_packet.source_frame_width)}x{int(frame_packet.source_frame_height)}"
        self._metrics["source_frame_resolution_counts"][source_key] = self._metrics["source_frame_resolution_counts"].get(source_key, 0) + 1
        frame_key = (frame_packet.camera_id, frame_packet.frame_number)
        frame_cached = False
        for tracked_detection in tracked_detections:
            self._metrics["candidate_observations"] += 1
            self._metrics["candidate_crops_seen"] += 1
            candidate_entry = self._build_candidate(frame_packet, tracked_detection)
            if candidate_entry is None:
                self._metrics["invalid_candidates"] += 1
                continue
            if not frame_cached:
                self._frame_cache[frame_key] = frame_packet.frame.copy()
                frame_cached = True
                self._metrics["cache_peak_frames"] = max(self._metrics["cache_peak_frames"], len(self._frame_cache))
            local_track_id = candidate_entry.candidate.local_track_id
            existing_candidates = self._track_candidates.setdefault(local_track_id, [])
            if any(item.candidate.frame_number == candidate_entry.candidate.frame_number for item in existing_candidates):
                self._metrics["duplicate_frame_candidates_skipped"] += 1
                self._metrics["candidate_crops_rejected"] += 1
                self.logger.debug(
                    "Evidence candidate skipped camera_id=%s local_track_id=%s frame_number=%s reason=duplicate_frame_candidate",
                    frame_packet.camera_id,
                    local_track_id,
                    frame_packet.frame_number,
                )
                continue
            retained = self._retain_candidate(local_track_id, candidate_entry)
            if retained:
                self._ensure_frame_cached(frame_key, frame_packet.frame)
                self._frame_ref_counts[frame_key] = self._frame_ref_counts.get(frame_key, 0) + 1
                self._metrics["candidate_crops_retained"] += 1
            else:
                self._metrics["candidate_crops_rejected"] += 1
                if frame_cached and self._frame_ref_counts.get(frame_key, 0) <= 0:
                    self._frame_cache.pop(frame_key, None)
        self._register_capture_zone_frame(frame_packet, tracked_detections)

    def _ensure_frame_cached(self, frame_key: tuple[str, int], frame: np.ndarray) -> None:
        if frame_key in self._frame_cache:
            return
        self._frame_cache[frame_key] = frame.copy()
        self._metrics["cache_peak_frames"] = max(self._metrics["cache_peak_frames"], len(self._frame_cache))

    def _retain_candidate(self, local_track_id: str, candidate_entry: _StoredCandidate) -> bool:
        existing_candidates = self._track_candidates.setdefault(local_track_id, [])
        candidate_entry.candidate.best_overall_score = self._instantaneous_candidate_score(candidate_entry.candidate)
        duplicate_index, duplicate_score = self._find_duplicate_candidate(existing_candidates, candidate_entry)
        if duplicate_index is not None:
            self._metrics["candidate_crops_deduplicated"] += 1
            candidate_entry.candidate.best_overall_score = max(candidate_entry.candidate.best_overall_score, duplicate_score)
            current = existing_candidates[duplicate_index]
            if candidate_entry.candidate.best_overall_score > current.candidate.best_overall_score:
                removed = existing_candidates.pop(duplicate_index)
                self._release_candidates([removed])
                self._metrics["candidate_crops_replaced"] += 1
                existing_candidates.append(candidate_entry)
                self._refresh_candidate_pool_metrics(existing_candidates)
                return True
            return False
        existing_candidates.append(candidate_entry)
        if len(existing_candidates) > int(self.config["maximum_candidates_per_track"]):
            weakest = min(existing_candidates, key=lambda item: (item.candidate.best_overall_score, item.candidate.frame_number))
            if weakest is candidate_entry:
                existing_candidates.remove(candidate_entry)
                return False
            existing_candidates.remove(weakest)
            self._release_candidates([weakest])
            self._metrics["candidate_crops_replaced"] += 1
        self._refresh_candidate_pool_metrics(existing_candidates)
        return True

    def _find_duplicate_candidate(self, existing_candidates: list[_StoredCandidate], candidate_entry: _StoredCandidate) -> tuple[int | None, float]:
        if not self.config["deduplicate_similar_crops"]:
            return None, 0.0
        minimum_frame_gap = int(self.config["minimum_frame_gap"])
        best_index: int | None = None
        best_score = 0.0
        for index, existing in enumerate(existing_candidates):
            frame_gap = abs(existing.candidate.frame_number - candidate_entry.candidate.frame_number)
            if frame_gap > minimum_frame_gap:
                continue
            score = self._bbox_iou(existing.candidate.expanded_crop_bbox_xyxy, candidate_entry.candidate.expanded_crop_bbox_xyxy)
            if score >= float(self.config["duplicate_iou_threshold"]) and score > best_score:
                best_index = index
                best_score = score
        return best_index, best_score

    @staticmethod
    def _bbox_iou(left: tuple[int, int, int, int], right: tuple[int, int, int, int]) -> float:
        x1 = max(left[0], right[0])
        y1 = max(left[1], right[1])
        x2 = min(left[2], right[2])
        y2 = min(left[3], right[3])
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        left_area = max(0, left[2] - left[0]) * max(0, left[3] - left[1])
        right_area = max(0, right[2] - right[0]) * max(0, right[3] - right[1])
        union = left_area + right_area - intersection
        if union <= 0:
            return 0.0
        return float(intersection / union)

    def _instantaneous_candidate_score(self, candidate: EvidenceCandidate) -> float:
        size_score = float(candidate.original_crop_width * candidate.original_crop_height)
        return (
            (float(candidate.confidence) * 1000.0)
            + (float(candidate.sharpness_score) * 0.5)
            + (size_score * 0.01)
            + (float(candidate.centeredness_score) * 100.0)
            + (float(candidate.edge_visibility_score) * 100.0)
        )

    def _refresh_candidate_pool_metrics(self, candidates: list[_StoredCandidate]) -> None:
        size = len(candidates)
        self._metrics["candidate_pool_peak_per_track"] = max(int(self._metrics["candidate_pool_peak_per_track"]), size)
        peak = int(self._metrics["candidate_pool_peak_per_track"])
        if peak <= 0:
            return
        retained = max(1, int(self._metrics["candidate_crops_retained"]))
        previous_average = float(self._metrics["candidate_pool_average_per_track"])
        self._metrics["candidate_pool_average_per_track"] = ((previous_average * (retained - 1)) + size) / retained

    def finalize_track(self, track: LocalTrack) -> list[TrackEvidence]:
        candidates = self._track_candidates.pop(track.local_track_id, [])
        zone_records = self._finalize_capture_zone_track(track)
        finalized = self._finalize_track_with_candidates(track, candidates, release_candidates_immediately=True)
        return finalized + zone_records

    def finalize_tracks(self, tracks: list[LocalTrack]) -> list[TrackEvidence]:
        evidence: list[TrackEvidence] = []
        candidates_to_release: list[_StoredCandidate] = []
        try:
            for track in tracks:
                candidates = self._track_candidates.pop(track.local_track_id, [])
                candidates_to_release.extend(candidates)
                evidence.extend(self._finalize_track_with_candidates(track, candidates, release_candidates_immediately=False))
                evidence.extend(self._finalize_capture_zone_track(track))
            return evidence
        finally:
            self._release_candidates(candidates_to_release)

    def _finalize_track_with_candidates(
        self,
        track: LocalTrack,
        candidates: list[_StoredCandidate],
        *,
        release_candidates_immediately: bool,
    ) -> list[TrackEvidence]:
        self._metrics["tracks_received"] += 1
        self._metrics["tracks_by_camera"][track.camera_id] = self._metrics["tracks_by_camera"].get(track.camera_id, 0) + 1
        try:
            if not self.enabled:
                return []
            if track.status == TRACK_STATUS_DISCARDED and not self.config["include_discarded_tracks"]:
                return []
            if track.status not in {TRACK_STATUS_COMPLETED, TRACK_STATUS_DISCARDED}:
                return []
            if not candidates:
                self._metrics["tracks_without_candidate_evidence"] += 1
                self._metrics["tracks_without_valid_evidence"] += 1
                return []
            self._metrics["tracks_with_candidate_evidence"] += 1

            self._score_candidates(candidates)
            selected_by_role = self._select_roles(candidates)
            if not selected_by_role:
                self._metrics["tracks_without_valid_evidence"] += 1
                return []

            self._metrics["tracks_with_evidence"] += 1
            track_folder = self._sanitize_track_id(track.local_track_id)
            frame_assets = self._save_selected_assets(track, track_folder, selected_by_role)
            records = self._build_evidence_records(track, selected_by_role, frame_assets)
            if any(record["crop_path"] is None or record["annotated_frame_path"] is None for record in records):
                self._metrics["tracks_with_partial_evidence"] += 1
            self.output_manager.save_track_evidence(track.camera_id, track_folder, records)
            self._evidence_index.extend(records)
            self._metrics["selected_evidence_records"] += len(records)
            for record in records:
                role = str(record["role"])
                self._metrics["role_counts"][role] = self._metrics["role_counts"].get(role, 0) + 1
            return [
                TrackEvidence(
                    local_track_id=record["local_track_id"],
                    camera_id=record["camera_id"],
                    native_tracker_id=int(record["native_tracker_id"]),
                    tracker_namespace=record["tracker_namespace"],
                    role=record["role"],
                    frame_number=int(record["frame_number"]),
                    timestamp_seconds=float(record["timestamp_seconds"]),
                    raw_class_name=record["raw_class_name"],
                    final_class=record["final_class"],
                    confidence=float(record["confidence"]),
                    crop_path=record["crop_path"],
                    annotated_frame_path=record["annotated_frame_path"],
                    bbox_xyxy=tuple(record["bbox_xyxy"]),
                    original_bbox_xyxy=tuple(record["original_bbox"]),
                    expanded_crop_bbox_xyxy=tuple(record["expanded_crop_bbox"]),
                    context_padding_ratio=float(record["context_padding_ratio"]),
                    source_frame_width=int(record["source_frame_width"]),
                    source_frame_height=int(record["source_frame_height"]),
                    original_crop_width=int(record["original_crop_width"]),
                    original_crop_height=int(record["original_crop_height"]),
                    sharpness_score=float(record["sharpness_score"]),
                    best_overall_score=float(record["best_overall_score"]),
                )
                for record in records
            ]
        finally:
            if release_candidates_immediately:
                self._release_candidates(candidates)

    def _validate_config(self, evidence: Any) -> dict[str, Any]:
        payload = dict(evidence or {})
        weights = dict(payload.get("best_overall_weights", {}) or {})
        capture_zone_payload = dict(payload.get("capture_zone", {}) or {})
        normalized = {
            "enabled": bool(payload.get("enabled", True)),
            "collect_first": bool(payload.get("collect_first", True)),
            "collect_middle": bool(payload.get("collect_middle", True)),
            "collect_last": bool(payload.get("collect_last", True)),
            "collect_highest_confidence": bool(payload.get("collect_highest_confidence", True)),
            "collect_largest": bool(payload.get("collect_largest", True)),
            "collect_sharpest": bool(payload.get("collect_sharpest", True)),
            "collect_best_overall": bool(payload.get("collect_best_overall", True)),
            "maximum_candidates_per_track": int(payload.get("maximum_candidates_per_track", 7)),
            "minimum_frame_gap": max(0, int(payload.get("minimum_frame_gap", 0))),
            "preferred_frame_gap": max(0, int(payload.get("preferred_frame_gap", 8))),
            "deduplicate_similar_crops": bool(payload.get("deduplicate_similar_crops", False)),
            "duplicate_iou_threshold": float(payload.get("duplicate_iou_threshold", 0.85)),
            "minimum_crop_width_pixels": int(payload.get("minimum_crop_width_pixels", 40)),
            "minimum_crop_height_pixels": int(payload.get("minimum_crop_height_pixels", 40)),
            "crop_padding_ratio_x": float(payload.get("crop_padding_ratio_x", 0.08)),
            "crop_padding_ratio_y": float(payload.get("crop_padding_ratio_y", 0.08)),
            "minimum_padding_pixels": int(payload.get("minimum_padding_pixels", 8)),
            "clamp_bbox_to_frame": bool(payload.get("clamp_bbox_to_frame", True)),
            "reject_invalid_bbox": bool(payload.get("reject_invalid_bbox", True)),
            "sharpness_enabled": bool(payload.get("sharpness_enabled", True)),
            "jpeg_quality": int(payload.get("jpeg_quality", 90)),
            "save_vehicle_crops": bool(payload.get("save_vehicle_crops", True)),
            "save_annotated_full_frames": bool(payload.get("save_annotated_full_frames", True)),
            "save_all_candidates": bool(payload.get("save_all_candidates", False)),
            "include_discarded_tracks": bool(payload.get("include_discarded_tracks", False)),
            "fail_pipeline_on_error": bool(payload.get("fail_pipeline_on_error", False)),
        }
        if normalized["maximum_candidates_per_track"] < 1:
            raise ConfigurationError("evidence.maximum_candidates_per_track must be at least 1.")
        if normalized["duplicate_iou_threshold"] < 0.0 or normalized["duplicate_iou_threshold"] > 1.0:
            raise ConfigurationError("evidence.duplicate_iou_threshold must be between 0 and 1.")
        if normalized["minimum_crop_width_pixels"] < 1 or normalized["minimum_crop_height_pixels"] < 1:
            raise ConfigurationError("evidence minimum crop dimensions must be at least 1 pixel.")
        if normalized["crop_padding_ratio_x"] < 0.0 or normalized["crop_padding_ratio_y"] < 0.0:
            raise ConfigurationError("evidence crop padding ratios must be at least 0.")
        if normalized["minimum_padding_pixels"] < 0:
            raise ConfigurationError("evidence.minimum_padding_pixels must be at least 0.")
        if not 0 <= normalized["jpeg_quality"] <= 100:
            raise ConfigurationError("evidence.jpeg_quality must be between 0 and 100.")
        raw_weights = {
            "confidence": float(weights.get("confidence", 0.35)),
            "sharpness": float(weights.get("sharpness", 0.25)),
            "bbox_area": float(weights.get("bbox_area", 0.20)),
            "centeredness": float(weights.get("centeredness", 0.10)),
            "edge_visibility": float(weights.get("edge_visibility", 0.10)),
        }
        if any(value < 0.0 for value in raw_weights.values()):
            raise ConfigurationError("evidence.best_overall_weights values must be at least 0.")
        weight_sum = sum(raw_weights.values())
        if weight_sum <= 0.0:
            raise ConfigurationError("evidence.best_overall_weights sum must be greater than zero.")
        normalized["best_overall_weights"] = {name: value / weight_sum for name, value in raw_weights.items()}
        normalized["capture_zone"] = self._normalize_capture_zone_config(
            capture_zone_payload,
            default_minimum_width=normalized["minimum_crop_width_pixels"],
            default_minimum_height=normalized["minimum_crop_height_pixels"],
        )
        capture_zone = normalized["capture_zone"]
        self.logger.info("EvidenceCollector normalized best-overall weights=%s", normalized["best_overall_weights"])
        return normalized

    def _normalize_capture_zone_config(
        self,
        payload: dict[str, Any],
        *,
        default_minimum_width: int,
        default_minimum_height: int,
    ) -> dict[str, Any]:
        def _normalize_profile(profile_payload: dict[str, Any], fallback: dict[str, Any], context: str) -> dict[str, Any]:
            profile = {
                "top_ratio": float(profile_payload.get("top_ratio", fallback["top_ratio"])),
                "bottom_ratio": float(profile_payload.get("bottom_ratio", fallback["bottom_ratio"])),
                "trigger_point": str(profile_payload.get("trigger_point", fallback["trigger_point"])).strip() or "bottom_center",
                "maximum_saved_candidates_per_track": int(profile_payload.get("maximum_saved_candidates_per_track", fallback["maximum_saved_candidates_per_track"])),
                "minimum_frame_gap": max(0, int(profile_payload.get("minimum_frame_gap", fallback["minimum_frame_gap"]))),
                "capture_policy": str(profile_payload.get("capture_policy", fallback["capture_policy"])).strip() or "best_quality",
                "save_immediately": bool(profile_payload.get("save_immediately", fallback["save_immediately"])),
                "require_confirmed_track": bool(profile_payload.get("require_confirmed_track", fallback["require_confirmed_track"])),
                "minimum_bbox_width_pixels": int(profile_payload.get("minimum_bbox_width_pixels", fallback["minimum_bbox_width_pixels"])),
                "minimum_bbox_height_pixels": int(profile_payload.get("minimum_bbox_height_pixels", fallback["minimum_bbox_height_pixels"])),
                "direction_mode": str(profile_payload.get("direction_mode", fallback["direction_mode"])).strip() or "any",
            }
            self._validate_capture_zone_profile(profile, context)
            return profile

        base_defaults = {
            "top_ratio": 0.55,
            "bottom_ratio": 0.72,
            "trigger_point": "bottom_center",
            "maximum_saved_candidates_per_track": 3,
            "minimum_frame_gap": 2,
            "capture_policy": "best_quality",
            "save_immediately": True,
            "require_confirmed_track": True,
            "minimum_bbox_width_pixels": int(default_minimum_width),
            "minimum_bbox_height_pixels": int(default_minimum_height),
            "direction_mode": "any",
        }
        default_source = {
            **dict(payload.get("default", {}) or {}),
            "trigger_point": payload.get("trigger_point", dict(payload.get("default", {}) or {}).get("trigger_point", base_defaults["trigger_point"])),
            "maximum_saved_candidates_per_track": payload.get("maximum_saved_candidates_per_track", dict(payload.get("default", {}) or {}).get("maximum_saved_candidates_per_track", base_defaults["maximum_saved_candidates_per_track"])),
            "minimum_frame_gap": payload.get("minimum_frame_gap", dict(payload.get("default", {}) or {}).get("minimum_frame_gap", base_defaults["minimum_frame_gap"])),
            "capture_policy": payload.get("capture_policy", dict(payload.get("default", {}) or {}).get("capture_policy", base_defaults["capture_policy"])),
            "save_immediately": payload.get("save_immediately", dict(payload.get("default", {}) or {}).get("save_immediately", base_defaults["save_immediately"])),
            "require_confirmed_track": payload.get("require_confirmed_track", dict(payload.get("default", {}) or {}).get("require_confirmed_track", base_defaults["require_confirmed_track"])),
            "minimum_bbox_width_pixels": payload.get("minimum_bbox_width_pixels", dict(payload.get("default", {}) or {}).get("minimum_bbox_width_pixels", base_defaults["minimum_bbox_width_pixels"])),
            "minimum_bbox_height_pixels": payload.get("minimum_bbox_height_pixels", dict(payload.get("default", {}) or {}).get("minimum_bbox_height_pixels", base_defaults["minimum_bbox_height_pixels"])),
            "direction_mode": payload.get("direction_mode", dict(payload.get("default", {}) or {}).get("direction_mode", base_defaults["direction_mode"])),
        }
        legacy_ratio_keys_present = "top_ratio" in payload or "bottom_ratio" in payload
        if legacy_ratio_keys_present or not default_source:
            default_source = {
                **default_source,
                "top_ratio": payload.get("top_ratio", default_source.get("top_ratio", base_defaults["top_ratio"])),
                "bottom_ratio": payload.get("bottom_ratio", default_source.get("bottom_ratio", base_defaults["bottom_ratio"])),
            }
        default_profile = _normalize_profile(default_source, base_defaults, "evidence.capture_zone.default")

        class_specific_profiles: dict[str, dict[str, Any]] = {}
        for class_name, class_payload in dict(payload.get("class_specific", {}) or {}).items():
            if not isinstance(class_payload, dict):
                raise ConfigurationError(f"evidence.capture_zone.class_specific.{class_name} must be a mapping.")
            class_specific_profiles[self._normalize_vehicle_class(str(class_name))] = _normalize_profile(
                dict(class_payload),
                default_profile,
                f"evidence.capture_zone.class_specific.{class_name}",
            )

        camera_profiles: dict[str, dict[str, Any]] = {}
        for camera_id, camera_payload in dict(payload.get("cameras", {}) or {}).items():
            if not isinstance(camera_payload, dict):
                raise ConfigurationError(f"evidence.capture_zone.cameras.{camera_id} must be a mapping.")
            camera_payload = dict(camera_payload)
            camera_default_source = {
                **dict(camera_payload.get("default", {}) or {}),
                "trigger_point": camera_payload.get("trigger_point", dict(camera_payload.get("default", {}) or {}).get("trigger_point", default_profile["trigger_point"])),
                "maximum_saved_candidates_per_track": camera_payload.get("maximum_saved_candidates_per_track", dict(camera_payload.get("default", {}) or {}).get("maximum_saved_candidates_per_track", default_profile["maximum_saved_candidates_per_track"])),
                "minimum_frame_gap": camera_payload.get("minimum_frame_gap", dict(camera_payload.get("default", {}) or {}).get("minimum_frame_gap", default_profile["minimum_frame_gap"])),
                "capture_policy": camera_payload.get("capture_policy", dict(camera_payload.get("default", {}) or {}).get("capture_policy", default_profile["capture_policy"])),
                "save_immediately": camera_payload.get("save_immediately", dict(camera_payload.get("default", {}) or {}).get("save_immediately", default_profile["save_immediately"])),
                "require_confirmed_track": camera_payload.get("require_confirmed_track", dict(camera_payload.get("default", {}) or {}).get("require_confirmed_track", default_profile["require_confirmed_track"])),
                "minimum_bbox_width_pixels": camera_payload.get("minimum_bbox_width_pixels", dict(camera_payload.get("default", {}) or {}).get("minimum_bbox_width_pixels", default_profile["minimum_bbox_width_pixels"])),
                "minimum_bbox_height_pixels": camera_payload.get("minimum_bbox_height_pixels", dict(camera_payload.get("default", {}) or {}).get("minimum_bbox_height_pixels", default_profile["minimum_bbox_height_pixels"])),
                "direction_mode": camera_payload.get("direction_mode", dict(camera_payload.get("default", {}) or {}).get("direction_mode", default_profile["direction_mode"])),
            }
            if "top_ratio" in camera_payload or "bottom_ratio" in camera_payload:
                camera_default_source = {
                    **camera_default_source,
                    "top_ratio": camera_payload.get("top_ratio", camera_default_source.get("top_ratio", default_profile["top_ratio"])),
                    "bottom_ratio": camera_payload.get("bottom_ratio", camera_default_source.get("bottom_ratio", default_profile["bottom_ratio"])),
                }
            camera_profile = {
                "enabled": bool(camera_payload.get("enabled", payload.get("enabled", False))),
                "default": _normalize_profile(
                    camera_default_source,
                    default_profile,
                    f"evidence.capture_zone.cameras.{camera_id}.default",
                ),
                "class_specific": {},
            }
            for class_name, class_payload in dict(camera_payload.get("class_specific", {}) or {}).items():
                if not isinstance(class_payload, dict):
                    raise ConfigurationError(f"evidence.capture_zone.cameras.{camera_id}.class_specific.{class_name} must be a mapping.")
                normalized_class = self._normalize_vehicle_class(str(class_name))
                fallback_profile = class_specific_profiles.get(normalized_class, camera_profile["default"])
                camera_profile["class_specific"][normalized_class] = _normalize_profile(
                    dict(class_payload),
                    fallback_profile,
                    f"evidence.capture_zone.cameras.{camera_id}.class_specific.{class_name}",
                )
            camera_profiles[str(camera_id).strip()] = camera_profile

        normalized_capture_zone = {
            "enabled": bool(payload.get("enabled", False)),
            "default": default_profile,
            "class_specific": class_specific_profiles,
            "cameras": camera_profiles,
            "top_ratio": default_profile["top_ratio"],
            "bottom_ratio": default_profile["bottom_ratio"],
            "trigger_point": default_profile["trigger_point"],
            "maximum_saved_candidates_per_track": default_profile["maximum_saved_candidates_per_track"],
            "minimum_frame_gap": default_profile["minimum_frame_gap"],
            "capture_policy": default_profile["capture_policy"],
            "save_immediately": default_profile["save_immediately"],
            "require_confirmed_track": default_profile["require_confirmed_track"],
            "minimum_bbox_width_pixels": default_profile["minimum_bbox_width_pixels"],
            "minimum_bbox_height_pixels": default_profile["minimum_bbox_height_pixels"],
            "direction_mode": default_profile["direction_mode"],
        }
        return normalized_capture_zone

    def _validate_capture_zone_profile(self, profile: dict[str, Any], context: str) -> None:
        if profile["trigger_point"] != "bottom_center":
            raise ConfigurationError(f"{context}.trigger_point must be bottom_center.")
        if not 0.0 <= profile["top_ratio"] < profile["bottom_ratio"] <= 1.0:
            raise ConfigurationError(f"{context} ratios must satisfy 0.0 <= top_ratio < bottom_ratio <= 1.0.")
        if profile["maximum_saved_candidates_per_track"] < 1:
            raise ConfigurationError(f"{context}.maximum_saved_candidates_per_track must be at least 1.")
        if profile["minimum_bbox_width_pixels"] < 1 or profile["minimum_bbox_height_pixels"] < 1:
            raise ConfigurationError(f"{context} minimum bbox dimensions must be at least 1.")

    def _capture_zone_config_for_camera(self, camera_id: str) -> dict[str, Any]:
        capture_zone = dict(self.config.get("capture_zone", {}) or {})
        camera_overrides = dict(capture_zone.get("cameras", {}) or {}).get(camera_id)
        if isinstance(camera_overrides, dict):
            return {
                "enabled": bool(camera_overrides.get("enabled", capture_zone.get("enabled", False))),
                "default": dict(camera_overrides.get("default", capture_zone.get("default", {})) or {}),
                "class_specific": dict(camera_overrides.get("class_specific", {}) or {}),
            }
        return {
            "enabled": bool(capture_zone.get("enabled", False)),
            "default": dict(capture_zone.get("default", {}) or {}),
            "class_specific": dict(capture_zone.get("class_specific", {}) or {}),
        }

    def _resolve_capture_zone_profile(self, camera_id: str, vehicle_class: str) -> dict[str, Any]:
        camera_capture_zone = self._capture_zone_config_for_camera(camera_id)
        normalized_class = self._normalize_vehicle_class(vehicle_class)
        return dict(
            dict(camera_capture_zone.get("class_specific", {}) or {}).get(
                normalized_class,
                camera_capture_zone.get("default", {}),
            )
            or {}
        )

    def _maybe_save_debug_track_crop(
        self,
        frame_packet: FramePacket,
        tracked_detection: TrackedDetection,
        *,
        zone_top: int,
        zone_bottom: int,
        inside_capture_zone: bool,
        vehicle_class: str,
    ) -> None:
        if not self._debug_track_crops_enabled:
            return
        local_track_id = self._logical_track_id(frame_packet.camera_id, tracked_detection)
        saved_count = int(self._debug_track_crop_counts.get(local_track_id, 0))
        if saved_count >= self._debug_track_crops_max_per_track:
            return
        if frame_packet.frame_number % self._debug_track_crops_save_every_n_frames != 0:
            return
        crop_bbox = self._apply_padding(
            tuple(float(item) for item in tracked_detection.bbox_xyxy),
            int(frame_packet.frame.shape[1]),
            int(frame_packet.frame.shape[0]),
        )
        x1, y1, x2, y2 = crop_bbox
        crop = frame_packet.frame[y1:y2, x1:x2]
        if crop.size == 0:
            return
        safe_track_id = self._sanitize_track_id(local_track_id)
        crop_path = self.output_manager.save_track_crop(
            frame_packet.camera_id,
            safe_track_id,
            frame_packet.frame_number,
            crop,
            jpeg_quality=int(self.config["jpeg_quality"]),
        )
        class_min_width, class_min_height = self._class_specific_evidence_thresholds(vehicle_class)
        florence_min_width, florence_min_height = self._class_specific_florence_thresholds(vehicle_class)
        crop_width = int(crop.shape[1])
        crop_height = int(crop.shape[0])
        evidence_eligible = crop_width >= class_min_width and crop_height >= class_min_height
        if evidence_eligible:
            evidence_rejection_reason = None
        elif crop_width < class_min_width:
            evidence_rejection_reason = self._class_reason("width_below", vehicle_class, "minimum")
        else:
            evidence_rejection_reason = self._class_reason("height_below", vehicle_class, "minimum")
        florence_eligible = crop_width >= florence_min_width and crop_height >= florence_min_height
        if florence_eligible:
            florence_rejection_reason = None
        elif crop_width < florence_min_width:
            florence_rejection_reason = self._class_reason("width_below", vehicle_class, "florence_minimum")
        else:
            florence_rejection_reason = self._class_reason("height_below", vehicle_class, "florence_minimum")
        self._debug_track_crop_rows.append(
            {
                "camera_id": frame_packet.camera_id,
                "local_track_id": local_track_id,
                "frame_number": int(frame_packet.frame_number),
                "timestamp_seconds": float(frame_packet.timestamp_seconds),
                "vehicle_class": vehicle_class,
                "confidence": float(tracked_detection.confidence),
                "bbox_x1": float(tracked_detection.bbox_xyxy[0]),
                "bbox_y1": float(tracked_detection.bbox_xyxy[1]),
                "bbox_x2": float(tracked_detection.bbox_xyxy[2]),
                "bbox_y2": float(tracked_detection.bbox_xyxy[3]),
                "crop_width": crop_width,
                "crop_height": crop_height,
                "crop_path": str(crop_path),
                "trigger_y": float(tracked_detection.bbox_xyxy[3]),
                "inside_capture_zone": bool(inside_capture_zone),
                "capture_zone_top": int(zone_top),
                "capture_zone_bottom": int(zone_bottom),
                "evidence_eligible": evidence_eligible,
                "evidence_rejection_reason": evidence_rejection_reason,
                "florence_eligible": florence_eligible,
                "florence_rejection_reason": florence_rejection_reason,
            }
        )
        self._debug_track_crop_counts[local_track_id] = saved_count + 1

    @staticmethod
    def _class_reason(prefix: str, vehicle_class: str, suffix: str) -> str:
        normalized = " ".join(str(vehicle_class or "").strip().lower().replace("_", " ").replace("-", " ").split())
        if normalized and normalized != "unknown":
            return f"{prefix}_{normalized}_{suffix}"
        return f"{prefix}_{suffix}"

    def _update_track_geometry(
        self,
        state: _CaptureZoneTrackState,
        *,
        frame_number: int,
        tracked_detection: TrackedDetection,
        trigger_y: float,
        source_frame_width: int,
        source_frame_height: int,
    ) -> None:
        bbox_width = max(0, int(round(tracked_detection.bbox_xyxy[2] - tracked_detection.bbox_xyxy[0])))
        bbox_height = max(0, int(round(tracked_detection.bbox_xyxy[3] - tracked_detection.bbox_xyxy[1])))
        bbox_area = bbox_width * bbox_height
        class_name = self._normalize_vehicle_class(tracked_detection.raw_class_name)
        state.first_frame = frame_number if state.first_frame is None else state.first_frame
        state.last_frame = frame_number
        state.minimum_trigger_y = trigger_y if state.minimum_trigger_y is None else min(state.minimum_trigger_y, trigger_y)
        if state.maximum_trigger_y is None or trigger_y >= state.maximum_trigger_y:
            state.maximum_trigger_y = trigger_y
            state.frame_of_max_trigger_y = frame_number
        if bbox_width >= state.maximum_bbox_width:
            state.maximum_bbox_width = bbox_width
            state.frame_of_max_bbox_width = frame_number
        if bbox_height >= state.maximum_bbox_height:
            state.maximum_bbox_height = bbox_height
            state.frame_of_max_bbox_height = frame_number
        if bbox_area >= state.maximum_bbox_area:
            state.maximum_bbox_area = bbox_area
            state.frame_of_max_bbox_area = frame_number
        state.class_counts[class_name] = state.class_counts.get(class_name, 0) + 1
        state.class_confidence_sums[class_name] = state.class_confidence_sums.get(class_name, 0.0) + float(tracked_detection.confidence)
        state.stable_class_name = self._stable_track_class_estimate(state)
        state.source_frame_width = int(source_frame_width)
        state.source_frame_height = int(source_frame_height)

    def _stable_track_class_estimate(self, state: _CaptureZoneTrackState) -> str:
        if not state.class_counts or state.observation_count < self._track_class_minimum_observations:
            return "unknown"
        winner_class, winner_confidence = max(
            state.class_confidence_sums.items(),
            key=lambda item: (item[1], item[0]),
        )
        winner_count = int(state.class_counts.get(winner_class, 0))
        winner_ratio = float(winner_count / max(1, sum(state.class_counts.values())))
        sorted_confidence = sorted(state.class_confidence_sums.items(), key=lambda item: (item[1], item[0]), reverse=True)
        runner_up_confidence = sorted_confidence[1][1] if len(sorted_confidence) > 1 else float("-inf")
        if winner_ratio < self._track_class_minimum_winner_ratio:
            return "unknown"
        if winner_confidence <= runner_up_confidence:
            return "unknown"
        return self._normalize_vehicle_class(winner_class)

    def _register_capture_zone_frame(self, frame_packet: FramePacket, tracked_detections: list[TrackedDetection]) -> None:
        if not tracked_detections:
            return
        camera_capture_zone = self._capture_zone_config_for_camera(frame_packet.camera_id)
        if not bool(camera_capture_zone.get("enabled", False)):
            return
        for tracked_detection in tracked_detections:
            local_track_id = self._logical_track_id(frame_packet.camera_id, tracked_detection)
            state = self._capture_zone_state.setdefault(local_track_id, _CaptureZoneTrackState())
            state.observation_count += 1
            trigger_x = float((tracked_detection.bbox_xyxy[0] + tracked_detection.bbox_xyxy[2]) / 2.0)
            trigger_y = float(tracked_detection.bbox_xyxy[3])
            self._update_track_geometry(
                state,
                frame_number=frame_packet.frame_number,
                tracked_detection=tracked_detection,
                trigger_y=trigger_y,
                source_frame_width=frame_packet.source_frame_width,
                source_frame_height=frame_packet.source_frame_height,
            )
            zone_profile = self._resolve_capture_zone_profile(frame_packet.camera_id, state.stable_class_name)
            zone_top = int(frame_packet.source_frame_height * float(zone_profile["top_ratio"]))
            zone_bottom = int(frame_packet.source_frame_height * float(zone_profile["bottom_ratio"]))
            state.zone_top_pixels = zone_top
            state.zone_bottom_pixels = zone_bottom
            position = self._zone_position(trigger_y, zone_top, zone_bottom)
            vehicle_class = self._normalize_vehicle_class(tracked_detection.raw_class_name)
            self._maybe_save_debug_track_crop(
                frame_packet,
                tracked_detection,
                zone_top=zone_top,
                zone_bottom=zone_bottom,
                inside_capture_zone=position == "inside",
                vehicle_class=vehicle_class,
            )
            self.logger.debug(
                "Motorcycle geometry track=%s frame=%s trigger_y=%.2f zone_top=%s zone_bottom=%s bbox=%sx%s state=%s stable_class=%s",
                local_track_id,
                frame_packet.frame_number,
                trigger_y,
                zone_top,
                zone_bottom,
                max(0, int(round(tracked_detection.bbox_xyxy[2] - tracked_detection.bbox_xyxy[0]))),
                max(0, int(round(tracked_detection.bbox_xyxy[3] - tracked_detection.bbox_xyxy[1]))),
                position.upper(),
                state.stable_class_name,
            )
            if position == "inside" and state.last_position != "inside":
                state.entered_zone = True
                state.first_zone_entry_frame = frame_packet.frame_number if state.first_zone_entry_frame is None else state.first_zone_entry_frame
                self._metrics["capture_zone_tracks_entered"] += 1
                self.logger.info(
                    "Evidence zone entered camera=%s track=%s frame=%s",
                    frame_packet.camera_id,
                    local_track_id,
                    frame_packet.frame_number,
                )
            if position == "inside":
                state.last_zone_frame = frame_packet.frame_number
            if position == "below" and state.last_position == "inside":
                state.exited_zone = True
                state.zone_exit_frame = frame_packet.frame_number
                self.logger.info(
                    "Evidence zone exited camera=%s track=%s frame=%s",
                    frame_packet.camera_id,
                    local_track_id,
                    frame_packet.frame_number,
                )
            should_consider = position == "inside"
            state.last_position = position
            if not should_consider:
                continue
            if bool(zone_profile.get("require_confirmed_track", True)) and state.observation_count < self._confirmed_track_minimum_observations:
                continue
            if state.last_capture_frame is not None and (frame_packet.frame_number - state.last_capture_frame) < int(zone_profile["minimum_frame_gap"]):
                self._metrics["capture_zone_duplicate_frame_suppressed"] += 1
                continue
            self._metrics["capture_zone_candidate_attempts"] += 1
            state.capture_zone_candidate_count += 1
            candidate_entry = self._build_candidate(frame_packet, tracked_detection)
            if candidate_entry is None:
                self._metrics["capture_zone_invalid_bbox_count"] += 1
                continue
            bbox_width = candidate_entry.crop_bbox_xyxy[2] - candidate_entry.crop_bbox_xyxy[0]
            bbox_height = candidate_entry.crop_bbox_xyxy[3] - candidate_entry.crop_bbox_xyxy[1]
            if bbox_width < int(zone_profile["minimum_bbox_width_pixels"]) or bbox_height < int(zone_profile["minimum_bbox_height_pixels"]):
                self._metrics["capture_zone_too_small_count"] += 1
                self._metrics["capture_zone_candidates_too_small"] += 1
                continue
            if vehicle_class == "motorcycle":
                self._metrics["capture_zone_motorcycle_candidates"] += 1
            crop = frame_packet.frame[candidate_entry.crop_bbox_xyxy[1]:candidate_entry.crop_bbox_xyxy[3], candidate_entry.crop_bbox_xyxy[0]:candidate_entry.crop_bbox_xyxy[2]]
            if crop.size == 0:
                self._metrics["capture_zone_invalid_bbox_count"] += 1
                continue
            class_min_width, class_min_height = self._class_specific_evidence_thresholds(vehicle_class)
            if candidate_entry.candidate.original_crop_width < class_min_width or candidate_entry.candidate.original_crop_height < class_min_height:
                self._metrics["capture_zone_candidates_rejected_by_class_threshold"] += 1
            elif vehicle_class == "motorcycle":
                self._metrics["capture_zone_motorcycle_eligible_candidates"] += 1
            safe_track_id = self._sanitize_track_id(local_track_id)
            crop_path = str(
                self.output_manager.save_capture_zone_crop(
                    frame_packet.camera_id,
                    safe_track_id,
                    frame_packet.frame_number,
                    crop,
                    jpeg_quality=int(self.config["jpeg_quality"]),
                )
            )
            stored = _CaptureZoneStoredCandidate(
                candidate=candidate_entry.candidate,
                crop_bbox_xyxy=candidate_entry.crop_bbox_xyxy,
                crop_path=crop_path,
                trigger_x=trigger_x,
                trigger_y=trigger_y,
                zone_top=zone_top,
                zone_bottom=zone_bottom,
            )
            kept = self._retain_capture_zone_candidate(local_track_id, stored, int(zone_profile["maximum_saved_candidates_per_track"]))
            if kept:
                state.last_capture_frame = frame_packet.frame_number
                state.capture_zone_retained_count += 1
                crop_width = int(candidate_entry.candidate.original_crop_width)
                crop_height = int(candidate_entry.candidate.original_crop_height)
                if (crop_width * crop_height) >= (state.largest_saved_crop_width * state.largest_saved_crop_height):
                    state.largest_saved_crop_width = crop_width
                    state.largest_saved_crop_height = crop_height
                    state.largest_saved_crop_frame = frame_packet.frame_number
                self._metrics["capture_zone_candidates_saved"] += 1
                self.logger.info(
                    "Evidence zone candidate captured camera=%s track=%s frame=%s quality=%.3f path=%s",
                    frame_packet.camera_id,
                    local_track_id,
                    frame_packet.frame_number,
                    candidate_entry.candidate.best_overall_score,
                    crop_path,
                )
            else:
                self._remove_file_if_exists(crop_path)

    def _retain_capture_zone_candidate(self, local_track_id: str, candidate_entry: _CaptureZoneStoredCandidate, maximum_candidates: int) -> bool:
        retained = self._capture_zone_candidates.setdefault(local_track_id, [])
        candidate_entry.candidate.best_overall_score = self._capture_zone_candidate_score(candidate_entry.candidate)
        same_frame = next((item for item in retained if item.candidate.frame_number == candidate_entry.candidate.frame_number), None)
        if same_frame is not None:
            self._metrics["capture_zone_duplicate_frame_suppressed"] += 1
            return False
        retained.append(candidate_entry)
        retained.sort(key=lambda item: (item.candidate.best_overall_score, item.candidate.frame_number), reverse=True)
        if len(retained) > maximum_candidates:
            removed = retained.pop()
            self._remove_file_if_exists(removed.crop_path)
            self._metrics["capture_zone_candidates_replaced"] += 1
            if (candidate_entry.candidate.original_crop_width * candidate_entry.candidate.original_crop_height) > (
                removed.candidate.original_crop_width * removed.candidate.original_crop_height
            ):
                self._metrics["capture_zone_candidates_replaced_by_larger_crop"] = self._metrics.get("capture_zone_candidates_replaced_by_larger_crop", 0) + 1
            else:
                self._metrics["capture_zone_candidates_replaced_by_better_quality"] = self._metrics.get("capture_zone_candidates_replaced_by_better_quality", 0) + 1
            self.logger.info(
                "Evidence zone candidate replaced camera=%s track=%s old_quality=%.3f new_quality=%.3f",
                candidate_entry.candidate.camera_id,
                local_track_id,
                removed.candidate.best_overall_score,
                candidate_entry.candidate.best_overall_score,
            )
        return candidate_entry in retained

    def _capture_zone_candidate_score(self, candidate: EvidenceCandidate) -> float:
        generic_score = self._instantaneous_candidate_score(candidate)
        vehicle_class = self._normalize_vehicle_class(candidate.raw_class_name)
        class_min_width, class_min_height = self._class_specific_evidence_thresholds(vehicle_class)
        florence_min_width, florence_min_height = self._class_specific_florence_thresholds(vehicle_class)
        width = max(1, int(candidate.original_crop_width))
        height = max(1, int(candidate.original_crop_height))
        evidence_width_ratio = min(2.0, width / max(1, class_min_width))
        evidence_height_ratio = min(2.0, height / max(1, class_min_height))
        florence_width_ratio = min(2.0, width / max(1, florence_min_width))
        florence_height_ratio = min(2.0, height / max(1, florence_min_height))
        evidence_bonus = 0.0
        if width >= class_min_width and height >= class_min_height:
            evidence_bonus += 350.0
        else:
            evidence_bonus -= 180.0
        if width >= florence_min_width and height >= florence_min_height:
            evidence_bonus += 500.0
        else:
            evidence_bonus -= 120.0
        size_bonus = ((evidence_width_ratio + evidence_height_ratio) * 90.0) + ((florence_width_ratio + florence_height_ratio) * 120.0)
        return generic_score + evidence_bonus + size_bonus

    @staticmethod
    def _normalize_vehicle_class(value: str | None) -> str:
        return " ".join(str(value or "").strip().lower().replace("_", " ").replace("-", " ").split()) or "unknown"

    def _class_specific_evidence_thresholds(self, vehicle_class: str) -> tuple[int, int]:
        defaults = (
            int(self._vehicle_enrichment_evidence_config.get("minimum_crop_width", 100)),
            int(self._vehicle_enrichment_evidence_config.get("minimum_crop_height", 70)),
        )
        payload = dict(dict(self._vehicle_enrichment_evidence_config.get("class_specific_minimums", {}) or {}).get(vehicle_class, {}) or {})
        return (
            int(payload.get("minimum_crop_width", defaults[0])),
            int(payload.get("minimum_crop_height", defaults[1])),
        )

    def _class_specific_florence_thresholds(self, vehicle_class: str) -> tuple[int, int]:
        florence = self._vehicle_enrichment_florence_config
        default_payload = dict(florence.get("default", {}) or {})
        class_specific = dict(florence.get("class_specific", {}) or {})
        payload = dict(class_specific.get(vehicle_class, {}) or {})
        default_width = int(florence.get("minimum_original_width", default_payload.get("minimum_original_width", 192)))
        default_height = int(florence.get("minimum_original_height", default_payload.get("minimum_original_height", 144)))
        return (
            int(payload.get("minimum_original_width", default_width)),
            int(payload.get("minimum_original_height", default_height)),
        )

    @staticmethod
    def _zone_position(trigger_y: float, zone_top: int, zone_bottom: int) -> str:
        if trigger_y < zone_top:
            return "above"
        if trigger_y > zone_bottom:
            return "below"
        return "inside"

    def _build_candidate(self, frame_packet: FramePacket, tracked_detection: TrackedDetection) -> _StoredCandidate | None:
        frame_height, frame_width = frame_packet.frame.shape[:2]
        x1, y1, x2, y2 = tracked_detection.bbox_xyxy
        if self.config["reject_invalid_bbox"] and (x2 <= x1 or y2 <= y1):
            return None
        padded_bbox = self._apply_padding((x1, y1, x2, y2), frame_width, frame_height)
        crop_x1, crop_y1, crop_x2, crop_y2 = padded_bbox
        crop_width = crop_x2 - crop_x1
        crop_height = crop_y2 - crop_y1
        if crop_width < self.config["minimum_crop_width_pixels"] or crop_height < self.config["minimum_crop_height_pixels"]:
            return None
        crop = frame_packet.frame[crop_y1:crop_y2, crop_x1:crop_x2]
        if crop.size == 0:
            return None
        bbox_area = max(0.0, float(x2 - x1)) * max(0.0, float(y2 - y1))
        sharpness_score = self._compute_sharpness(crop) if self.config["sharpness_enabled"] else 0.0
        local_track_id = self._logical_track_id(frame_packet.camera_id, tracked_detection)
        candidate = EvidenceCandidate(
            local_track_id=local_track_id,
            camera_id=frame_packet.camera_id,
            native_tracker_id=int(tracked_detection.tracker_id),
            tracker_namespace=tracked_detection.tracker_namespace,
            frame_number=frame_packet.frame_number,
            timestamp_seconds=frame_packet.timestamp_seconds,
            bbox_xyxy=tuple(float(item) for item in tracked_detection.bbox_xyxy),
            confidence=float(max(0.0, min(1.0, tracked_detection.confidence))),
            raw_class_name=tracked_detection.raw_class_name,
            final_class="UNKNOWN",
            role="",
            original_bbox_xyxy=tuple(float(item) for item in tracked_detection.bbox_xyxy),
            expanded_crop_bbox_xyxy=padded_bbox,
            context_padding_ratio=float(max(self.config["crop_padding_ratio_x"], self.config["crop_padding_ratio_y"])),
            source_frame_width=int(frame_packet.source_frame_width),
            source_frame_height=int(frame_packet.source_frame_height),
            original_crop_width=int(crop_width),
            original_crop_height=int(crop_height),
            bbox_area=bbox_area,
            sharpness_score=sharpness_score,
            centeredness_score=self._compute_centeredness_score(tracked_detection.bbox_xyxy, frame_width, frame_height),
            edge_visibility_score=self._compute_edge_visibility_score(tracked_detection.bbox_xyxy, frame_width, frame_height),
            best_overall_score=0.0,
        )
        return _StoredCandidate(
            candidate=candidate,
            frame_key=(frame_packet.camera_id, frame_packet.frame_number),
            crop_bbox_xyxy=padded_bbox,
        )

    def _apply_padding(
        self,
        bbox_xyxy: tuple[float, float, float, float],
        frame_width: int,
        frame_height: int,
    ) -> tuple[int, int, int, int]:
        x1, y1, x2, y2 = bbox_xyxy
        bbox_width = max(0.0, x2 - x1)
        bbox_height = max(0.0, y2 - y1)
        padding_x = max(bbox_width * self.config["crop_padding_ratio_x"], float(self.config["minimum_padding_pixels"]))
        padding_y = max(bbox_height * self.config["crop_padding_ratio_y"], float(self.config["minimum_padding_pixels"]))
        padded = (
            int(math.floor(x1 - padding_x)),
            int(math.floor(y1 - padding_y)),
            int(math.ceil(x2 + padding_x)),
            int(math.ceil(y2 + padding_y)),
        )
        if self.config["clamp_bbox_to_frame"]:
            return (
                max(0, min(frame_width, padded[0])),
                max(0, min(frame_height, padded[1])),
                max(0, min(frame_width, padded[2])),
                max(0, min(frame_height, padded[3])),
            )
        return padded

    def _compute_sharpness(self, crop: np.ndarray) -> float:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    def _compute_centeredness_score(
        self,
        bbox_xyxy: tuple[float, float, float, float],
        frame_width: int,
        frame_height: int,
    ) -> float:
        x1, y1, x2, y2 = bbox_xyxy
        bbox_center_x = (x1 + x2) / 2.0
        bbox_center_y = (y1 + y2) / 2.0
        frame_center_x = frame_width / 2.0
        frame_center_y = frame_height / 2.0
        distance = math.dist((bbox_center_x, bbox_center_y), (frame_center_x, frame_center_y))
        max_distance = math.dist((0.0, 0.0), (frame_center_x, frame_center_y))
        if max_distance <= 0.0:
            return 1.0
        return max(0.0, min(1.0, 1.0 - (distance / max_distance)))

    def _compute_edge_visibility_score(
        self,
        bbox_xyxy: tuple[float, float, float, float],
        frame_width: int,
        frame_height: int,
    ) -> float:
        x1, y1, x2, y2 = bbox_xyxy
        bbox_width = max(1.0, x2 - x1)
        bbox_height = max(1.0, y2 - y1)
        left_ratio = max(0.0, min(1.0, x1 / bbox_width))
        right_ratio = max(0.0, min(1.0, (frame_width - x2) / bbox_width))
        top_ratio = max(0.0, min(1.0, y1 / bbox_height))
        bottom_ratio = max(0.0, min(1.0, (frame_height - y2) / bbox_height))
        return max(0.0, min(1.0, min(left_ratio, right_ratio, top_ratio, bottom_ratio)))

    def _score_candidates(self, candidates: list[_StoredCandidate]) -> None:
        max_area = max(item.candidate.bbox_area for item in candidates) if candidates else 0.0
        max_sharpness = max(item.candidate.sharpness_score for item in candidates) if candidates else 0.0
        weights = self.config["best_overall_weights"]
        for item in candidates:
            candidate = item.candidate
            candidate.final_class = candidate.final_class or "UNKNOWN"
            normalized_area = 0.0 if max_area <= 0.0 else max(0.0, min(1.0, candidate.bbox_area / max_area))
            normalized_sharpness = 0.0 if max_sharpness <= 0.0 else max(0.0, min(1.0, candidate.sharpness_score / max_sharpness))
            candidate.best_overall_score = (
                weights["confidence"] * max(0.0, min(1.0, candidate.confidence))
                + weights["sharpness"] * normalized_sharpness
                + weights["bbox_area"] * normalized_area
                + weights["centeredness"] * max(0.0, min(1.0, candidate.centeredness_score))
                + weights["edge_visibility"] * max(0.0, min(1.0, candidate.edge_visibility_score))
            )

    def _select_roles(self, candidates: list[_StoredCandidate]) -> dict[str, _StoredCandidate]:
        selected: dict[str, _StoredCandidate] = {}
        ordered = sorted(candidates, key=lambda item: item.candidate.frame_number)
        if self.config["collect_first"]:
            selected[EVIDENCE_ROLE_FIRST] = ordered[0]
        if self.config["collect_middle"]:
            first_frame = ordered[0].candidate.frame_number
            last_frame = ordered[-1].candidate.frame_number
            midpoint = (first_frame + last_frame) / 2.0
            selected[EVIDENCE_ROLE_MIDDLE] = min(
                ordered,
                key=lambda item: (abs(item.candidate.frame_number - midpoint), item.candidate.frame_number),
            )
        if self.config["collect_last"]:
            selected[EVIDENCE_ROLE_LAST] = ordered[-1]
        if self.config["collect_highest_confidence"]:
            selected[EVIDENCE_ROLE_HIGHEST_CONFIDENCE] = max(
                ordered,
                key=lambda item: (item.candidate.confidence, -item.candidate.frame_number),
            )
        if self.config["collect_largest"]:
            selected[EVIDENCE_ROLE_LARGEST] = max(
                ordered,
                key=lambda item: (item.candidate.bbox_area, -item.candidate.frame_number),
            )
        if self.config["collect_sharpest"]:
            selected[EVIDENCE_ROLE_SHARPEST] = max(
                ordered,
                key=lambda item: (item.candidate.sharpness_score, -item.candidate.frame_number),
            )
        if self.config["collect_best_overall"]:
            selected[EVIDENCE_ROLE_BEST_OVERALL] = max(
                ordered,
                key=lambda item: (
                    item.candidate.best_overall_score,
                    item.candidate.confidence,
                    -item.candidate.frame_number,
                ),
            )
        return selected

    def _save_selected_assets(
        self,
        track: LocalTrack,
        safe_track_id: str,
        selected_by_role: dict[str, _StoredCandidate],
    ) -> dict[int, dict[str, str | None]]:
        grouped_roles: dict[int, list[str]] = {}
        for role, candidate in selected_by_role.items():
            grouped_roles.setdefault(candidate.candidate.frame_number, []).append(role)

        frame_assets: dict[int, dict[str, str | None]] = {}
        for frame_number, roles in grouped_roles.items():
            candidate_entry = next(
                item for item in selected_by_role.values() if item.candidate.frame_number == frame_number
            )
            frame = self._frame_cache.get(candidate_entry.frame_key)
            if frame is None:
                try:
                    raise RuntimeError(f"Evidence frame missing from cache for {track.local_track_id} frame={frame_number}")
                except RuntimeError as exc:
                    self._metrics["evidence_cache_misses"] += 1
                    self._metrics["missing_cache_frame_count"] += 1
                    self._metrics["evidence_items_skipped_missing_frame"] += len(roles)
                    self._handle_error(
                        exc,
                        camera_id=track.camera_id,
                        local_track_id=track.local_track_id,
                        frame_number=frame_number,
                        role=",".join(sorted(roles)),
                        cache_size=len(self._frame_cache),
                        error_type="missing_frame_from_cache",
                    )
                frame_assets[frame_number] = {"crop_path": None, "annotated_frame_path": None}
                continue
            self._metrics["evidence_cache_hits"] += 1
            crop_path: str | None = None
            annotated_path: str | None = None
            try:
                if self.config["save_vehicle_crops"]:
                    x1, y1, x2, y2 = candidate_entry.crop_bbox_xyxy
                    crop = frame[y1:y2, x1:x2]
                    crop_file = self.output_manager.save_evidence_crop(
                        track.camera_id,
                        safe_track_id,
                        frame_number,
                        crop,
                        jpeg_quality=self.config["jpeg_quality"],
                    )
                    crop_path = str(crop_file)
                    self._unique_crop_paths.add(crop_path)
                    self._metrics["saved_files_by_camera"][track.camera_id] = self._metrics["saved_files_by_camera"].get(track.camera_id, 0) + 1
                if self.config["save_annotated_full_frames"]:
                    annotated_frame = self._build_annotated_frame(track, candidate_entry, roles, frame)
                    annotated_file = self.output_manager.save_evidence_annotated_frame(
                        track.camera_id,
                        safe_track_id,
                        frame_number,
                        annotated_frame,
                        jpeg_quality=self.config["jpeg_quality"],
                    )
                    annotated_path = str(annotated_file)
                    self._unique_annotated_paths.add(annotated_path)
                    self._metrics["saved_files_by_camera"][track.camera_id] = self._metrics["saved_files_by_camera"].get(track.camera_id, 0) + 1
            except Exception as exc:
                self._handle_error(
                    exc,
                    camera_id=track.camera_id,
                    local_track_id=track.local_track_id,
                    frame_number=frame_number,
                    role=",".join(sorted(roles)),
                    cache_size=len(self._frame_cache),
                )
            frame_assets[frame_number] = {"crop_path": crop_path, "annotated_frame_path": annotated_path}
        self._metrics["unique_crop_files"] = len(self._unique_crop_paths)
        self._metrics["unique_annotated_frame_files"] = len(self._unique_annotated_paths)
        return frame_assets

    def _build_evidence_records(
        self,
        track: LocalTrack,
        selected_by_role: dict[str, _StoredCandidate],
        frame_assets: dict[int, dict[str, str | None]],
    ) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for role in ALLOWED_EVIDENCE_ROLES:
            candidate_entry = selected_by_role.get(role)
            if candidate_entry is None:
                continue
            candidate = candidate_entry.candidate
            candidate.final_class = track.final_class or "UNKNOWN"
            asset_paths = frame_assets.get(candidate.frame_number, {"crop_path": None, "annotated_frame_path": None})
            record = {
                "local_track_id": track.local_track_id,
                "camera_id": track.camera_id,
                "native_tracker_id": track.native_tracker_id,
                "tracker_namespace": track.tracker_namespace,
                "track_status": track.status,
                "final_class": track.final_class or "UNKNOWN",
                "role": role,
                "frame_number": candidate.frame_number,
                "timestamp_seconds": candidate.timestamp_seconds,
                "raw_class_name": candidate.raw_class_name,
                "confidence": candidate.confidence,
                "bbox_xyxy": list(candidate.bbox_xyxy),
                "original_bbox": list(candidate.original_bbox_xyxy),
                "expanded_crop_bbox": list(candidate.expanded_crop_bbox_xyxy),
                "context_padding_ratio": candidate.context_padding_ratio,
                "source_frame_width": candidate.source_frame_width,
                "source_frame_height": candidate.source_frame_height,
                "original_crop_width": candidate.original_crop_width,
                "original_crop_height": candidate.original_crop_height,
                "crop_path": asset_paths["crop_path"],
                "annotated_frame_path": asset_paths["annotated_frame_path"],
                "sharpness_score": candidate.sharpness_score,
                "centeredness_score": candidate.centeredness_score,
                "edge_visibility_score": candidate.edge_visibility_score,
                "best_overall_score": candidate.best_overall_score,
            }
            records.append(record)
            self._update_crop_metrics(candidate.original_crop_width, candidate.original_crop_height)
        return records

    def _finalize_capture_zone_track(self, track: LocalTrack) -> list[dict[str, Any]]:
        state = self._capture_zone_state.pop(track.local_track_id, None)
        candidates = self._capture_zone_candidates.pop(track.local_track_id, [])
        safe_track_id = self._sanitize_track_id(track.local_track_id)
        records: list[dict[str, Any]] = []
        motorcycle_track_with_evidence = False
        for candidate_entry in sorted(candidates, key=lambda item: item.candidate.best_overall_score, reverse=True):
            crop_path = candidate_entry.crop_path
            if not Path(crop_path).exists():
                self._metrics["capture_zone_missing_saved_crop"] += 1
                continue
            vehicle_class = self._normalize_vehicle_class(candidate_entry.candidate.raw_class_name)
            class_minimum_width, class_minimum_height = self._class_specific_evidence_thresholds(vehicle_class)
            florence_minimum_width, florence_minimum_height = self._class_specific_florence_thresholds(vehicle_class)
            evidence_eligible = (
                candidate_entry.candidate.original_crop_width >= class_minimum_width
                and candidate_entry.candidate.original_crop_height >= class_minimum_height
            )
            florence_eligible = (
                candidate_entry.candidate.original_crop_width >= florence_minimum_width
                and candidate_entry.candidate.original_crop_height >= florence_minimum_height
            )
            rejection_reason = None
            if not evidence_eligible:
                if candidate_entry.candidate.original_crop_width < class_minimum_width:
                    rejection_reason = f"crop_width_below_{vehicle_class}_minimum" if vehicle_class != "unknown" else "crop_width_below_minimum"
                elif candidate_entry.candidate.original_crop_height < class_minimum_height:
                    rejection_reason = f"crop_height_below_{vehicle_class}_minimum" if vehicle_class != "unknown" else "crop_height_below_minimum"
            elif not florence_eligible:
                if candidate_entry.candidate.original_crop_width < florence_minimum_width:
                    rejection_reason = f"crop_width_below_{vehicle_class}_florence_minimum" if vehicle_class != "unknown" else "crop_width_below_florence_minimum"
                elif candidate_entry.candidate.original_crop_height < florence_minimum_height:
                    rejection_reason = f"crop_height_below_{vehicle_class}_florence_minimum" if vehicle_class != "unknown" else "crop_height_below_florence_minimum"
            record = {
                "local_track_id": track.local_track_id,
                "camera_id": track.camera_id,
                "native_tracker_id": track.native_tracker_id,
                "tracker_namespace": track.tracker_namespace,
                "track_status": track.status,
                "final_class": track.final_class or "UNKNOWN",
                "role": "CAPTURE_ZONE",
                "frame_number": candidate_entry.candidate.frame_number,
                "timestamp_seconds": candidate_entry.candidate.timestamp_seconds,
                "raw_class_name": candidate_entry.candidate.raw_class_name,
                "confidence": candidate_entry.candidate.confidence,
                "bbox_xyxy": list(candidate_entry.candidate.bbox_xyxy),
                "original_bbox": list(candidate_entry.candidate.original_bbox_xyxy),
                "expanded_crop_bbox": list(candidate_entry.candidate.expanded_crop_bbox_xyxy),
                "context_padding_ratio": candidate_entry.candidate.context_padding_ratio,
                "source_frame_width": candidate_entry.candidate.source_frame_width,
                "source_frame_height": candidate_entry.candidate.source_frame_height,
                "original_crop_width": candidate_entry.candidate.original_crop_width,
                "original_crop_height": candidate_entry.candidate.original_crop_height,
                "crop_path": crop_path,
                "annotated_frame_path": None,
                "sharpness_score": candidate_entry.candidate.sharpness_score,
                "centeredness_score": candidate_entry.candidate.centeredness_score,
                "edge_visibility_score": candidate_entry.candidate.edge_visibility_score,
                "best_overall_score": candidate_entry.candidate.best_overall_score,
                "evidence_source": "capture_zone",
                "vehicle_class": vehicle_class,
                "trigger_x": candidate_entry.trigger_x,
                "trigger_y": candidate_entry.trigger_y,
                "zone_top": candidate_entry.zone_top,
                "zone_bottom": candidate_entry.zone_bottom,
                "class_minimum_width": class_minimum_width,
                "class_minimum_height": class_minimum_height,
                "florence_minimum_width": florence_minimum_width,
                "florence_minimum_height": florence_minimum_height,
                "evidence_eligible": evidence_eligible,
                "florence_eligible": florence_eligible,
                "rejection_reason": rejection_reason,
            }
            records.append(record)
            self._update_crop_metrics(candidate_entry.candidate.original_crop_width, candidate_entry.candidate.original_crop_height)
            if vehicle_class == "motorcycle":
                motorcycle_track_with_evidence = True
        if records:
            self.output_manager.save_capture_zone_track_evidence(track.camera_id, safe_track_id, records)
            self._capture_zone_index.extend(records)
            self._metrics["capture_zone_tracks_with_saved_evidence"] += 1
            if motorcycle_track_with_evidence:
                self._metrics["capture_zone_motorcycle_tracks_with_evidence"] += 1
        else:
            self._metrics["capture_zone_tracks_without_saved_evidence"] += 1
        self._record_motorcycle_geometry(track, state, records)
        if state is not None:
            state.retained_candidates.clear()
        return records

    def _record_motorcycle_geometry(
        self,
        track: LocalTrack,
        state: _CaptureZoneTrackState | None,
        records: list[dict[str, Any]],
    ) -> None:
        normalized_final_class = self._normalize_vehicle_class(track.final_class)
        observed_motorcycle = state is not None and state.class_counts.get("motorcycle", 0) > 0
        if normalized_final_class != "motorcycle" and not observed_motorcycle:
            return
        class_for_zone = normalized_final_class if normalized_final_class != "unknown" else (state.stable_class_name if state is not None else "unknown")
        zone_profile = self._resolve_capture_zone_profile(track.camera_id, class_for_zone)
        zone_top = int(state.zone_top_pixels if state is not None and state.zone_top_pixels is not None else 0)
        zone_bottom = int(state.zone_bottom_pixels if state is not None and state.zone_bottom_pixels is not None else 0)
        evidence_eligible_zone_crop = any(bool(item.get("evidence_eligible")) for item in records)
        florence_eligible_zone_crop = any(bool(item.get("florence_eligible")) for item in records)
        geometry_status, geometry_reason = self._geometry_status_for_track(
            track=track,
            state=state,
            has_records=bool(records),
            evidence_eligible_zone_crop=evidence_eligible_zone_crop,
            florence_eligible_zone_crop=florence_eligible_zone_crop,
        )
        record = _CaptureZoneGeometryRecord(
            camera_id=track.camera_id,
            local_track_id=track.local_track_id,
            source_frame_width=int(state.source_frame_width) if state is not None else 0,
            source_frame_height=int(state.source_frame_height) if state is not None else 0,
            first_frame=int(state.first_frame if state is not None and state.first_frame is not None else track.first_frame),
            last_frame=int(state.last_frame if state is not None and state.last_frame is not None else track.last_frame),
            observation_count=int(state.observation_count if state is not None else track.observation_count),
            min_trigger_y=float(state.minimum_trigger_y if state is not None and state.minimum_trigger_y is not None else 0.0),
            max_trigger_y=float(state.maximum_trigger_y if state is not None and state.maximum_trigger_y is not None else 0.0),
            zone_top=zone_top if zone_top > 0 else int(zone_profile["top_ratio"] * max(1, records[0]["source_frame_height"] if records else 1)),
            zone_bottom=zone_bottom if zone_bottom > 0 else int(zone_profile["bottom_ratio"] * max(1, records[0]["source_frame_height"] if records else 1)),
            entered_zone=bool(state.entered_zone) if state is not None else False,
            first_zone_entry_frame=state.first_zone_entry_frame if state is not None else None,
            last_zone_frame=state.last_zone_frame if state is not None else None,
            zone_exit_frame=state.zone_exit_frame if state is not None else None,
            max_bbox_width=int(state.maximum_bbox_width) if state is not None else 0,
            max_bbox_height=int(state.maximum_bbox_height) if state is not None else 0,
            max_bbox_area=int(state.maximum_bbox_area) if state is not None else 0,
            frame_of_max_trigger_y=state.frame_of_max_trigger_y if state is not None else None,
            frame_of_max_bbox_width=state.frame_of_max_bbox_width if state is not None else None,
            frame_of_max_bbox_height=state.frame_of_max_bbox_height if state is not None else None,
            frame_of_max_bbox_area=state.frame_of_max_bbox_area if state is not None else None,
            largest_saved_crop_width=int(state.largest_saved_crop_width) if state is not None else 0,
            largest_saved_crop_height=int(state.largest_saved_crop_height) if state is not None else 0,
            largest_saved_crop_frame=state.largest_saved_crop_frame if state is not None else None,
            capture_candidates=int(state.capture_zone_candidate_count) if state is not None else 0,
            retained_candidates=int(state.capture_zone_retained_count) if state is not None else 0,
            geometry_status=geometry_status,
            geometry_reason=geometry_reason,
            final_class=str(track.final_class or "UNKNOWN"),
            stable_class_name=state.stable_class_name if state is not None else "unknown",
            completion_reason=track.completion_reason,
            track_status=track.status,
            evidence_eligible_zone_crop=evidence_eligible_zone_crop,
            florence_eligible_zone_crop=florence_eligible_zone_crop,
        )
        self._motorcycle_geometry_records.append(record)

    def _geometry_status_for_track(
        self,
        *,
        track: LocalTrack,
        state: _CaptureZoneTrackState | None,
        has_records: bool,
        evidence_eligible_zone_crop: bool,
        florence_eligible_zone_crop: bool,
    ) -> tuple[str, str]:
        if state is None:
            return "REACHED_ZONE_NO_CAPTURE", "missing_capture_zone_state"
        max_trigger_y = float(state.maximum_trigger_y if state.maximum_trigger_y is not None else 0.0)
        zone_top = float(state.zone_top_pixels if state.zone_top_pixels is not None else 0.0)
        if max_trigger_y < zone_top:
            if str(track.completion_reason or "").upper() == "LOST_TIMEOUT":
                return "TRACK_ENDED_BEFORE_ZONE", "lost_timeout_before_zone"
            return "NEVER_REACHED_ZONE", "max_bottom_center_above_zone"
        if state.capture_zone_candidate_count == 0:
            return "REACHED_ZONE_NO_CAPTURE", "zone_reached_but_no_candidate_created"
        if not has_records:
            return "REACHED_ZONE_NO_CAPTURE", "candidate_not_retained"
        if florence_eligible_zone_crop or evidence_eligible_zone_crop:
            return "CAPTURED_ELIGIBLE", "eligible_zone_crop_retained"
        return "CAPTURED_TOO_SMALL", "captured_zone_crop_below_threshold"

    def _update_crop_metrics(self, width: int, height: int) -> None:
        self._metrics["evidence_crop_count"] += 1
        count = int(self._metrics["evidence_crop_count"])
        if count == 1:
            self._metrics["evidence_crop_width_min"] = width
            self._metrics["evidence_crop_width_max"] = width
            self._metrics["evidence_crop_height_min"] = height
            self._metrics["evidence_crop_height_max"] = height
            self._metrics["evidence_crop_width_average"] = float(width)
            self._metrics["evidence_crop_height_average"] = float(height)
            return
        self._metrics["evidence_crop_width_min"] = min(int(self._metrics["evidence_crop_width_min"]), width)
        self._metrics["evidence_crop_width_max"] = max(int(self._metrics["evidence_crop_width_max"]), width)
        self._metrics["evidence_crop_height_min"] = min(int(self._metrics["evidence_crop_height_min"]), height)
        self._metrics["evidence_crop_height_max"] = max(int(self._metrics["evidence_crop_height_max"]), height)
        self._metrics["evidence_crop_width_average"] = (
            (float(self._metrics["evidence_crop_width_average"]) * (count - 1)) + width
        ) / count
        self._metrics["evidence_crop_height_average"] = (
            (float(self._metrics["evidence_crop_height_average"]) * (count - 1)) + height
        ) / count

    def _build_annotated_frame(
        self,
        track: LocalTrack,
        candidate_entry: _StoredCandidate,
        roles: list[str],
        frame: np.ndarray,
    ) -> np.ndarray:
        annotated = frame.copy()
        x1, y1, x2, y2 = (int(round(value)) for value in candidate_entry.candidate.bbox_xyxy)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        lines = [
            f"{track.camera_id} | {track.local_track_id}",
            f"RAW {candidate_entry.candidate.raw_class_name.upper()} | FINAL {(track.final_class or 'UNKNOWN').upper()} | {candidate_entry.candidate.confidence:.2f}",
            f"native {track.native_tracker_id} | ns {track.tracker_namespace}",
            f"frame {candidate_entry.candidate.frame_number} | {candidate_entry.candidate.timestamp_seconds:.3f}s",
            f"roles: {', '.join(sorted(roles))}",
        ]
        origin_y = max(20, y1 - 10)
        for index, line in enumerate(lines):
            cv2.putText(
                annotated,
                line,
                (max(5, x1), origin_y + (index * 18)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )
        return annotated

    def _release_candidates(self, candidates: list[_StoredCandidate]) -> None:
        for item in candidates:
            frame_key = item.frame_key
            self._metrics["cache_release_attempts"] += 1
            remaining = self._frame_ref_counts.get(frame_key, 0) - 1
            if remaining <= 0:
                self._frame_ref_counts.pop(frame_key, None)
                if frame_key in self._frame_cache:
                    self._frame_cache.pop(frame_key, None)
                    self._metrics["evidence_cache_evictions"] += 1
                    self._metrics["cache_frames_released"] += 1
                    self.logger.debug(
                        "Frame cache release camera_id=%s frame_number=%s reason=no_remaining_references",
                        frame_key[0],
                        frame_key[1],
                    )
            else:
                self._frame_ref_counts[frame_key] = remaining
                self._metrics["evidence_cache_eviction_skipped_referenced"] += 1
                self._metrics["cache_release_deferred"] += 1
                self.logger.debug(
                    "Frame cache release deferred camera_id=%s frame_number=%s remaining_references=%s",
                    frame_key[0],
                    frame_key[1],
                    remaining,
                )

    @staticmethod
    def _remove_file_if_exists(path: str | None) -> None:
        if not path:
            return
        try:
            if os.path.exists(path):
                os.remove(path)
        except OSError:
            return

    def _handle_error(
        self,
        exc: Exception,
        *,
        camera_id: str | None = None,
        local_track_id: str | None = None,
        frame_number: int | None = None,
        role: str | None = None,
        cache_size: int | None = None,
        error_type: str | None = None,
    ) -> None:
        message = str(exc)
        structured_error = {
            "camera_id": camera_id,
            "local_track_id": local_track_id,
            "frame_number": frame_number,
            "role": role,
            "cache_size": cache_size,
            "error_type": error_type or exc.__class__.__name__,
            "error_class": exc.__class__.__name__,
            "message": message,
        }
        self.logger.error(
            "EvidenceCollector error camera_id=%s local_track_id=%s frame_number=%s role=%s cache_size=%s error_class=%s error=%s",
            camera_id,
            local_track_id,
            frame_number,
            role,
            cache_size,
            exc.__class__.__name__,
            message,
            exc_info=(type(exc), exc, exc.__traceback__) if exc.__traceback__ is not None else False,
        )
        self._metrics["errors"].append(structured_error)
        if self.config["fail_pipeline_on_error"]:
            raise PipelineRuntimeError(f"{exc.__class__.__name__}: {message}") from exc

    def _build_local_track_id(self, camera_id: str, tracker_namespace: str, native_tracker_id: int) -> str:
        normalized_namespace = str(tracker_namespace).strip()
        if normalized_namespace == "camera":
            return f"{camera_id}:TRACK_{native_tracker_id}"
        return f"{camera_id}:{normalized_namespace.upper()}:TRACK_{native_tracker_id}"

    def _logical_track_id(self, camera_id: str, tracked_detection: TrackedDetection) -> str:
        local_track_id = str(getattr(tracked_detection, "local_track_id", "") or "").strip()
        if local_track_id:
            return local_track_id
        return self._build_local_track_id(camera_id, tracked_detection.tracker_namespace, tracked_detection.tracker_id)

    def _sanitize_track_id(self, local_track_id: str) -> str:
        safe = "".join(character if character.isalnum() or character in {"_", "-"} else "_" for character in local_track_id)
        return safe.strip("_") or "track"
