from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
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


class EvidenceCollector:
    def __init__(self, config: dict[str, Any], logger: Any, output_manager: RunOutputManager) -> None:
        self.logger = logger
        self.output_manager = output_manager
        self.config = self._validate_config(config.get("evidence", {}))
        self.enabled = bool(self.config["enabled"])
        self._track_candidates: dict[str, list[_StoredCandidate]] = {}
        self._frame_cache: dict[tuple[str, int], np.ndarray] = {}
        self._frame_ref_counts: dict[tuple[str, int], int] = {}
        self._evidence_index: list[dict[str, Any]] = []
        self._metrics = {
            "tracks_received": 0,
            "tracks_with_evidence": 0,
            "tracks_without_valid_evidence": 0,
            "candidate_observations": 0,
            "invalid_candidates": 0,
            "selected_evidence_records": 0,
            "unique_crop_files": 0,
            "unique_annotated_frame_files": 0,
            "role_counts": {},
            "tracks_by_camera": {},
            "saved_files_by_camera": {},
            "cache_peak_frames": 0,
            "cache_frames_released": 0,
            "errors": [],
        }
        self._unique_crop_paths: set[str] = set()
        self._unique_annotated_paths: set[str] = set()
        self.logger.info(
            "EvidenceCollector initialized enabled=%s include_discarded_tracks=%s fail_pipeline_on_error=%s",
            self.enabled,
            self.config["include_discarded_tracks"],
            self.config["fail_pipeline_on_error"],
        )

    @property
    def evidence_index(self) -> list[dict[str, Any]]:
        return [dict(item) for item in self._evidence_index]

    @property
    def metrics(self) -> dict[str, Any]:
        metrics = dict(self._metrics)
        metrics["role_counts"] = dict(self._metrics["role_counts"])
        metrics["tracks_by_camera"] = dict(self._metrics["tracks_by_camera"])
        metrics["saved_files_by_camera"] = dict(self._metrics["saved_files_by_camera"])
        metrics["errors"] = list(self._metrics["errors"])
        return metrics

    def register_frame(self, frame_packet: FramePacket, tracked_detections: list[TrackedDetection]) -> None:
        if not self.enabled or not tracked_detections:
            return
        frame_key = (frame_packet.camera_id, frame_packet.frame_number)
        frame_cached = False
        for tracked_detection in tracked_detections:
            self._metrics["candidate_observations"] += 1
            candidate_entry = self._build_candidate(frame_packet, tracked_detection)
            if candidate_entry is None:
                self._metrics["invalid_candidates"] += 1
                continue
            if not frame_cached:
                self._frame_cache[frame_key] = frame_packet.frame.copy()
                frame_cached = True
                self._metrics["cache_peak_frames"] = max(self._metrics["cache_peak_frames"], len(self._frame_cache))
            local_track_id = candidate_entry.candidate.local_track_id
            self._track_candidates.setdefault(local_track_id, []).append(candidate_entry)
            self._frame_ref_counts[frame_key] = self._frame_ref_counts.get(frame_key, 0) + 1

    def finalize_track(self, track: LocalTrack) -> list[TrackEvidence]:
        self._metrics["tracks_received"] += 1
        self._metrics["tracks_by_camera"][track.camera_id] = self._metrics["tracks_by_camera"].get(track.camera_id, 0) + 1
        candidates = self._track_candidates.pop(track.local_track_id, [])
        try:
            if not self.enabled:
                self._release_candidates(candidates)
                return []
            if track.status == TRACK_STATUS_DISCARDED and not self.config["include_discarded_tracks"]:
                self._release_candidates(candidates)
                return []
            if track.status not in {TRACK_STATUS_COMPLETED, TRACK_STATUS_DISCARDED}:
                self._release_candidates(candidates)
                return []
            if not candidates:
                self._metrics["tracks_without_valid_evidence"] += 1
                return []

            self._score_candidates(candidates)
            selected_by_role = self._select_roles(candidates)
            if not selected_by_role:
                self._metrics["tracks_without_valid_evidence"] += 1
                return []

            self._metrics["tracks_with_evidence"] += 1
            track_folder = self._sanitize_track_id(track.local_track_id)
            frame_assets = self._save_selected_assets(track, track_folder, selected_by_role)
            records = self._build_evidence_records(track, selected_by_role, frame_assets)
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
                    sharpness_score=float(record["sharpness_score"]),
                    best_overall_score=float(record["best_overall_score"]),
                )
                for record in records
            ]
        finally:
            self._release_candidates(candidates)

    def finalize_tracks(self, tracks: list[LocalTrack]) -> list[TrackEvidence]:
        evidence: list[TrackEvidence] = []
        for track in tracks:
            evidence.extend(self.finalize_track(track))
        return evidence

    def _validate_config(self, evidence: Any) -> dict[str, Any]:
        payload = dict(evidence or {})
        weights = dict(payload.get("best_overall_weights", {}) or {})
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
        self.logger.info("EvidenceCollector normalized best-overall weights=%s", normalized["best_overall_weights"])
        return normalized

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
        local_track_id = self._build_local_track_id(frame_packet.camera_id, tracked_detection.tracker_namespace, tracked_detection.tracker_id)
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
                self._handle_error(RuntimeError(f"Evidence frame missing from cache for {track.local_track_id} frame={frame_number}"))
                frame_assets[frame_number] = {"crop_path": None, "annotated_frame_path": None}
                continue
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
                self._handle_error(exc)
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
                "crop_path": asset_paths["crop_path"],
                "annotated_frame_path": asset_paths["annotated_frame_path"],
                "sharpness_score": candidate.sharpness_score,
                "centeredness_score": candidate.centeredness_score,
                "edge_visibility_score": candidate.edge_visibility_score,
                "best_overall_score": candidate.best_overall_score,
            }
            records.append(record)
        return records

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
            remaining = self._frame_ref_counts.get(frame_key, 0) - 1
            if remaining <= 0:
                self._frame_ref_counts.pop(frame_key, None)
                if frame_key in self._frame_cache:
                    self._frame_cache.pop(frame_key, None)
                    self._metrics["cache_frames_released"] += 1
            else:
                self._frame_ref_counts[frame_key] = remaining

    def _handle_error(self, exc: Exception) -> None:
        message = f"{exc.__class__.__name__}: {exc}"
        self.logger.exception("EvidenceCollector error")
        self._metrics["errors"].append(message)
        if self.config["fail_pipeline_on_error"]:
            raise PipelineRuntimeError(message) from exc

    def _build_local_track_id(self, camera_id: str, tracker_namespace: str, native_tracker_id: int) -> str:
        normalized_namespace = str(tracker_namespace).strip()
        if normalized_namespace == "camera":
            return f"{camera_id}:TRACK_{native_tracker_id}"
        return f"{camera_id}:{normalized_namespace.upper()}:TRACK_{native_tracker_id}"

    def _sanitize_track_id(self, local_track_id: str) -> str:
        safe = "".join(character if character.isalnum() or character in {"_", "-"} else "_" for character in local_track_id)
        return safe.strip("_") or "track"
