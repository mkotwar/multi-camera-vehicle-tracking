from __future__ import annotations

import logging
from copy import deepcopy
from dataclasses import asdict
from typing import Any

from .models import (
    ConfigurationError,
    LocalTrack,
    TRACK_STATUS_ACTIVE,
    TRACK_STATUS_COMPLETED,
    TRACK_STATUS_DISCARDED,
    TRACK_STATUS_LOST,
    TRACK_STATUS_TENTATIVE,
    TrackObservation,
    TrackedDetection,
)


FINAL_CLASS_WEIGHTED_MAJORITY = "WEIGHTED_MAJORITY"
FINAL_CLASS_INSUFFICIENT_OBSERVATIONS = "INSUFFICIENT_OBSERVATIONS"
FINAL_CLASS_WINNER_RATIO_TOO_LOW = "WINNER_RATIO_TOO_LOW"
FINAL_CLASS_NO_CLEAR_WINNER = "NO_CLEAR_WINNER"
FINAL_CLASS_NO_CLASS_OBSERVATIONS = "NO_CLASS_OBSERVATIONS"
COMPLETION_REASON_LOST_TIMEOUT = "LOST_TIMEOUT"
COMPLETION_REASON_END_OF_STREAM = "END_OF_STREAM"
COMPLETION_REASON_INSUFFICIENT_OBSERVATIONS = "INSUFFICIENT_OBSERVATIONS"


class TrackManager:
    def __init__(self, config: dict[str, Any], logger: logging.Logger) -> None:
        self.config = dict(config)
        self.logger = logger
        lifecycle = dict(self.config.get("lifecycle", {}) or {})
        track_class = dict(self.config.get("track_class", {}) or {})
        self.minimum_observations = int(lifecycle.get("minimum_observations", 3))
        self.maximum_lost_frames = int(lifecycle.get("maximum_lost_frames", 30))
        self.keep_discarded_tracks = bool(lifecycle.get("keep_discarded_tracks", True))
        self.track_class_minimum_observations = int(track_class.get("minimum_observations", 3))
        self.minimum_winner_ratio = float(track_class.get("minimum_winner_ratio", 0.60))
        self.track_class_strategy = str(track_class.get("strategy", "confidence_weighted_majority")).strip()
        self.unknown_class_name = str(track_class.get("unknown_class_name", "UNKNOWN")).strip() or "UNKNOWN"
        if self.minimum_observations < 1:
            raise ConfigurationError("lifecycle.minimum_observations must be at least 1.")
        if self.maximum_lost_frames < 0:
            raise ConfigurationError("lifecycle.maximum_lost_frames must be at least 0.")
        if self.track_class_minimum_observations < 1:
            raise ConfigurationError("track_class.minimum_observations must be at least 1.")
        if not 0.0 <= self.minimum_winner_ratio <= 1.0:
            raise ConfigurationError("track_class.minimum_winner_ratio must be between 0 and 1.")
        if self.track_class_strategy != "confidence_weighted_majority":
            raise ConfigurationError("track_class.strategy must be confidence_weighted_majority.")
        self._tracks: dict[tuple[str, str, int], LocalTrack] = {}
        self._completed_tracks: list[LocalTrack] = []
        self._discarded_tracks: list[LocalTrack] = []
        self._last_frame_by_camera: dict[str, int] = {}
        self._mixed_class_logged_tracks: set[str] = set()
        self._metrics: dict[str, Any] = {
            "tracks_created_by_camera": {},
            "tracks_completed_by_camera": {},
            "tracks_discarded_by_camera": {},
            "observations_by_camera": {},
            "status_counts": {},
            "final_class_counts": {},
            "final_class_reason_counts": {},
            "lost_then_reactivated_count": 0,
            "end_of_stream_completed_count": 0,
            "lost_timeout_completed_count": 0,
            "duplicate_observation_count": 0,
            "out_of_order_frame_count": 0,
            "active_tracks_at_shutdown": 0,
            "minimum_observations": self.minimum_observations,
            "maximum_lost_frames": self.maximum_lost_frames,
            "minimum_winner_ratio": self.minimum_winner_ratio,
        }
        self.logger.info(
            "TrackManager initialized minimum_observations=%s maximum_lost_frames=%s minimum_winner_ratio=%.2f keep_discarded_tracks=%s",
            self.minimum_observations,
            self.maximum_lost_frames,
            self.minimum_winner_ratio,
            self.keep_discarded_tracks,
        )

    @property
    def completed_tracks(self) -> list[LocalTrack]:
        return [self._track_snapshot(track) for track in self._completed_tracks]

    @property
    def discarded_tracks(self) -> list[LocalTrack]:
        return [self._track_snapshot(track) for track in self._discarded_tracks]

    def get_all_output_tracks(self) -> list[LocalTrack]:
        output_tracks = list(self._completed_tracks)
        if self.keep_discarded_tracks:
            output_tracks.extend(self._discarded_tracks)
        return [self._track_snapshot(track) for track in output_tracks]

    def get_all_observations(self) -> list[TrackObservation]:
        observations: list[TrackObservation] = []
        for track in self.get_all_output_tracks():
            observations.extend(track.observations)
        return observations

    def get_metrics(self) -> dict[str, Any]:
        completed_tracks = [track for track in self._completed_tracks if track.status == TRACK_STATUS_COMPLETED]
        average_observations = (
            float(sum(track.observation_count for track in completed_tracks) / len(completed_tracks))
            if completed_tracks
            else 0.0
        )
        status_counts: dict[str, int] = {
            TRACK_STATUS_TENTATIVE: 0,
            TRACK_STATUS_ACTIVE: 0,
            TRACK_STATUS_LOST: 0,
            TRACK_STATUS_COMPLETED: len(self._completed_tracks),
            TRACK_STATUS_DISCARDED: len(self._discarded_tracks),
        }
        for track in self._tracks.values():
            status_counts[track.status] = status_counts.get(track.status, 0) + 1
        self._metrics["status_counts"] = status_counts
        self._metrics["active_tracks_at_shutdown"] = len(
            [track for track in self._tracks.values() if track.status in {TRACK_STATUS_TENTATIVE, TRACK_STATUS_ACTIVE, TRACK_STATUS_LOST}]
        )
        return {
            **self._metrics,
            "tracks_created_by_camera": dict(self._metrics["tracks_created_by_camera"]),
            "tracks_completed_by_camera": dict(self._metrics["tracks_completed_by_camera"]),
            "tracks_discarded_by_camera": dict(self._metrics["tracks_discarded_by_camera"]),
            "observations_by_camera": dict(self._metrics["observations_by_camera"]),
            "status_counts": dict(self._metrics["status_counts"]),
            "final_class_counts": dict(self._metrics["final_class_counts"]),
            "final_class_reason_counts": dict(self._metrics["final_class_reason_counts"]),
            "average_observations_per_completed_track": average_observations,
        }

    def update_frame(
        self,
        camera_id: str,
        frame_number: int,
        tracked_detections: list[TrackedDetection],
    ) -> list[LocalTrack]:
        self._validate_frame_order(camera_id, frame_number)
        present_track_keys: set[tuple[str, str, int]] = set()
        completed_now: list[LocalTrack] = []
        for detection in tracked_detections:
            self._validate_tracked_detection(camera_id, frame_number, detection)
            key = (camera_id, str(detection.tracker_namespace), int(detection.tracker_id))
            present_track_keys.add(key)
            track = self._tracks.get(key)
            if track is None:
                track = self._create_track_from_detection(detection)
                self._tracks[key] = track
            else:
                if track.status == TRACK_STATUS_LOST:
                    self._metrics["lost_then_reactivated_count"] += 1
                    self.logger.debug(
                        "track reactivated camera=%s frame=%s local_track_id=%s native_tracker_id=%s",
                        camera_id,
                        frame_number,
                        track.local_track_id,
                        detection.tracker_id,
                    )
                self._append_observation(track, detection)
        completed_now.extend(self._mark_missing_tracks(camera_id, frame_number, present_track_keys))
        return [self._track_snapshot(track) for track in completed_now]

    def flush_camera(self, camera_id: str, completion_reason: str = COMPLETION_REASON_END_OF_STREAM) -> list[LocalTrack]:
        finalized: list[LocalTrack] = []
        for key, track in list(self._tracks.items()):
            if track.camera_id != camera_id:
                continue
            finalized.append(self._finalize_track(key, completion_reason))
        self.logger.info("camera flushed camera_id=%s finalized_tracks=%s", camera_id, len(finalized))
        return [self._track_snapshot(track) for track in finalized]

    def flush_all(self) -> list[LocalTrack]:
        finalized: list[LocalTrack] = []
        for camera_id in sorted({track.camera_id for track in self._tracks.values()}):
            finalized.extend(self.flush_camera(camera_id))
        self.logger.info("all tracks flushed finalized_tracks=%s", len(finalized))
        return finalized

    def _validate_frame_order(self, camera_id: str, frame_number: int) -> None:
        previous = self._last_frame_by_camera.get(camera_id)
        if previous is not None and frame_number < previous:
            self._metrics["out_of_order_frame_count"] += 1
            self.logger.warning(
                "out-of-order frame received camera=%s previous_frame=%s current_frame=%s",
                camera_id,
                previous,
                frame_number,
            )
            raise ConfigurationError(f"Out-of-order frame for camera '{camera_id}': {frame_number} < {previous}")
        self._last_frame_by_camera[camera_id] = frame_number

    def _validate_tracked_detection(self, camera_id: str, frame_number: int, detection: TrackedDetection) -> None:
        if detection.camera_id != camera_id:
            raise ConfigurationError(
                f"TrackedDetection camera mismatch. update_frame camera={camera_id}; detection camera={detection.camera_id}"
            )
        if detection.frame_number != frame_number:
            raise ConfigurationError(
                f"TrackedDetection frame mismatch. update_frame frame={frame_number}; detection frame={detection.frame_number}"
            )
        if int(detection.tracker_id) < 0:
            raise ConfigurationError(f"TrackedDetection tracker_id must be a non-negative integer. Got: {detection.tracker_id}")
        if not str(detection.tracker_namespace).strip():
            raise ConfigurationError("TrackedDetection tracker_namespace must not be empty.")
        x1, y1, x2, y2 = detection.bbox_xyxy
        if not (x2 > x1 and y2 > y1):
            raise ConfigurationError(f"TrackedDetection bbox is invalid for {camera_id}: {detection.bbox_xyxy}")

    def _create_track_from_detection(self, detection: TrackedDetection) -> LocalTrack:
        local_track_id = self._build_local_track_id(detection.camera_id, detection.tracker_namespace, int(detection.tracker_id))
        observation = self._build_observation(local_track_id, detection)
        class_name = str(detection.raw_class_name).strip()
        track = LocalTrack(
            local_track_id=local_track_id,
            camera_id=detection.camera_id,
            tracker_namespace=str(detection.tracker_namespace),
            native_tracker_id=int(detection.tracker_id),
            status=TRACK_STATUS_TENTATIVE,
            first_frame=detection.frame_number,
            last_frame=detection.frame_number,
            first_timestamp_seconds=detection.timestamp_seconds,
            last_timestamp_seconds=detection.timestamp_seconds,
            observation_count=1,
            lost_frames=0,
            final_class=None,
            final_class_reason=None,
            class_counts={class_name: 1},
            class_confidence_sums={class_name: float(detection.confidence)},
            observations=[observation],
            completion_reason=None,
        )
        self._promote_if_ready(track)
        self._metrics["tracks_created_by_camera"][detection.camera_id] = self._metrics["tracks_created_by_camera"].get(detection.camera_id, 0) + 1
        self._metrics["observations_by_camera"][detection.camera_id] = self._metrics["observations_by_camera"].get(detection.camera_id, 0) + 1
        self.logger.debug(
            "track created camera=%s native_tracker_id=%s local_track_id=%s status=%s",
            detection.camera_id,
            detection.tracker_id,
            local_track_id,
            track.status,
        )
        return track

    def _append_observation(self, track: LocalTrack, detection: TrackedDetection) -> None:
        if detection.frame_number in {item.frame_number for item in track.observations}:
            self._metrics["duplicate_observation_count"] += 1
            self.logger.warning(
                "duplicate observation attempted camera=%s frame=%s local_track_id=%s",
                detection.camera_id,
                detection.frame_number,
                track.local_track_id,
            )
            raise ConfigurationError(
                f"Duplicate observation attempted for {track.local_track_id} frame={detection.frame_number}"
            )
        previous_status = track.status
        track.observations.append(self._build_observation(track.local_track_id, detection))
        track.last_frame = detection.frame_number
        track.last_timestamp_seconds = detection.timestamp_seconds
        track.observation_count += 1
        track.lost_frames = 0
        class_name = str(detection.raw_class_name).strip()
        track.class_counts[class_name] = track.class_counts.get(class_name, 0) + 1
        track.class_confidence_sums[class_name] = track.class_confidence_sums.get(class_name, 0.0) + float(detection.confidence)
        self._metrics["observations_by_camera"][detection.camera_id] = self._metrics["observations_by_camera"].get(detection.camera_id, 0) + 1
        if len(track.class_counts) > 1 and track.local_track_id not in self._mixed_class_logged_tracks:
            self._mixed_class_logged_tracks.add(track.local_track_id)
            self.logger.debug(
                "track mixed raw classes detected local_track_id=%s class_counts=%s",
                track.local_track_id,
                dict(sorted(track.class_counts.items())),
            )
        self._promote_if_ready(track)
        self.logger.debug(
            "track updated camera=%s frame=%s native_tracker_id=%s local_track_id=%s status=%s->%s lost_frames=%s observations=%s",
            detection.camera_id,
            detection.frame_number,
            detection.tracker_id,
            track.local_track_id,
            previous_status,
            track.status,
            track.lost_frames,
            track.observation_count,
        )

    def _mark_missing_tracks(self, camera_id: str, frame_number: int, present_track_keys: set[tuple[str, str, int]]) -> list[LocalTrack]:
        finalized: list[LocalTrack] = []
        for key, track in list(self._tracks.items()):
            if track.camera_id != camera_id:
                continue
            if key in present_track_keys:
                continue
            if track.status in {TRACK_STATUS_COMPLETED, TRACK_STATUS_DISCARDED}:
                continue
            previous_status = track.status
            track.lost_frames += 1
            track.status = TRACK_STATUS_LOST
            self.logger.debug(
                "track missing camera=%s frame=%s local_track_id=%s status=%s->%s lost_frames=%s",
                camera_id,
                frame_number,
                track.local_track_id,
                previous_status,
                track.status,
                track.lost_frames,
            )
            if track.lost_frames > self.maximum_lost_frames:
                finalized.append(self._finalize_track(key, COMPLETION_REASON_LOST_TIMEOUT))
        return finalized

    def _build_local_track_id(self, camera_id: str, tracker_namespace: str, native_tracker_id: int) -> str:
        normalized_namespace = str(tracker_namespace).strip()
        if normalized_namespace == "camera":
            return f"{camera_id}:TRACK_{native_tracker_id}"
        return f"{camera_id}:{normalized_namespace.upper()}:TRACK_{native_tracker_id}"

    def _promote_if_ready(self, track: LocalTrack) -> None:
        if track.observation_count >= self.minimum_observations:
            track.status = TRACK_STATUS_ACTIVE
        else:
            track.status = TRACK_STATUS_TENTATIVE

    def _finalize_track(self, key: tuple[str, int], completion_reason: str) -> LocalTrack:
        track = self._tracks.pop(key)
        final_class, final_reason = self._calculate_final_class(track)
        track.final_class = final_class
        track.final_class_reason = final_reason
        if track.observation_count >= self.minimum_observations:
            track.status = TRACK_STATUS_COMPLETED
            track.completion_reason = completion_reason
            self._completed_tracks.append(track)
            self._metrics["tracks_completed_by_camera"][track.camera_id] = self._metrics["tracks_completed_by_camera"].get(track.camera_id, 0) + 1
            if completion_reason == COMPLETION_REASON_END_OF_STREAM:
                self._metrics["end_of_stream_completed_count"] += 1
            if completion_reason == COMPLETION_REASON_LOST_TIMEOUT:
                self._metrics["lost_timeout_completed_count"] += 1
            self.logger.debug(
                "track completed camera=%s local_track_id=%s observations=%s final_class=%s reason=%s completion_reason=%s",
                track.camera_id,
                track.local_track_id,
                track.observation_count,
                track.final_class,
                track.final_class_reason,
                completion_reason,
            )
        else:
            track.status = TRACK_STATUS_DISCARDED
            track.completion_reason = COMPLETION_REASON_INSUFFICIENT_OBSERVATIONS if completion_reason != COMPLETION_REASON_END_OF_STREAM else completion_reason
            self._discarded_tracks.append(track)
            self._metrics["tracks_discarded_by_camera"][track.camera_id] = self._metrics["tracks_discarded_by_camera"].get(track.camera_id, 0) + 1
            self.logger.debug(
                "track discarded camera=%s local_track_id=%s observations=%s final_class=%s reason=%s completion_reason=%s",
                track.camera_id,
                track.local_track_id,
                track.observation_count,
                track.final_class,
                track.final_class_reason,
                track.completion_reason,
            )
        self._metrics["final_class_counts"][track.final_class] = self._metrics["final_class_counts"].get(track.final_class, 0) + 1
        self._metrics["final_class_reason_counts"][track.final_class_reason] = self._metrics["final_class_reason_counts"].get(track.final_class_reason, 0) + 1
        if len(track.class_counts) > 1:
            self.logger.debug(
                "track mixed raw classes summary local_track_id=%s class_counts=%s final_class=%s reason=%s",
                track.local_track_id,
                dict(sorted(track.class_counts.items())),
                track.final_class,
                track.final_class_reason,
            )
        if track.final_class == self.unknown_class_name:
            self.logger.debug(
                "final class returned UNKNOWN local_track_id=%s reason=%s",
                track.local_track_id,
                track.final_class_reason,
            )
        return track

    def _calculate_final_class(self, track: LocalTrack) -> tuple[str, str]:
        if not track.class_counts:
            self.logger.debug("final class calculation local_track_id=%s reason=%s", track.local_track_id, FINAL_CLASS_NO_CLASS_OBSERVATIONS)
            return self.unknown_class_name, FINAL_CLASS_NO_CLASS_OBSERVATIONS
        if track.observation_count < self.track_class_minimum_observations:
            self.logger.debug("final class calculation local_track_id=%s reason=%s", track.local_track_id, FINAL_CLASS_INSUFFICIENT_OBSERVATIONS)
            return self.unknown_class_name, FINAL_CLASS_INSUFFICIENT_OBSERVATIONS
        winner_class, winner_confidence_sum = max(
            track.class_confidence_sums.items(),
            key=lambda item: (item[1], item[0]),
        )
        sorted_confidence_sums = sorted(track.class_confidence_sums.items(), key=lambda item: (item[1], item[0]), reverse=True)
        runner_up_confidence_sum = sorted_confidence_sums[1][1] if len(sorted_confidence_sums) > 1 else float("-inf")
        winner_count = track.class_counts.get(winner_class, 0)
        winner_ratio = float(winner_count / sum(track.class_counts.values())) if track.class_counts else 0.0
        self.logger.debug(
            "final class calculation local_track_id=%s winner=%s winner_ratio=%.3f winner_confidence=%.3f runner_up_confidence=%.3f class_counts=%s",
            track.local_track_id,
            winner_class,
            winner_ratio,
            winner_confidence_sum,
            runner_up_confidence_sum if runner_up_confidence_sum != float("-inf") else -1.0,
            track.class_counts,
        )
        if winner_ratio < self.minimum_winner_ratio:
            return self.unknown_class_name, FINAL_CLASS_WINNER_RATIO_TOO_LOW
        if winner_confidence_sum <= runner_up_confidence_sum:
            return self.unknown_class_name, FINAL_CLASS_NO_CLEAR_WINNER
        return winner_class, FINAL_CLASS_WEIGHTED_MAJORITY

    def _build_observation(self, local_track_id: str, detection: TrackedDetection) -> TrackObservation:
        return TrackObservation(
            camera_id=detection.camera_id,
            tracker_namespace=str(detection.tracker_namespace),
            native_tracker_id=int(detection.tracker_id),
            local_track_id=local_track_id,
            frame_number=detection.frame_number,
            timestamp_seconds=detection.timestamp_seconds,
            bbox_xyxy=tuple(float(value) for value in detection.bbox_xyxy),
            confidence=float(detection.confidence),
            raw_class_id=int(detection.raw_class_id),
            raw_class_name=str(detection.raw_class_name),
        )

    def _track_snapshot(self, track: LocalTrack) -> LocalTrack:
        return LocalTrack(
            local_track_id=track.local_track_id,
            camera_id=track.camera_id,
            tracker_namespace=track.tracker_namespace,
            native_tracker_id=track.native_tracker_id,
            status=track.status,
            first_frame=track.first_frame,
            last_frame=track.last_frame,
            first_timestamp_seconds=track.first_timestamp_seconds,
            last_timestamp_seconds=track.last_timestamp_seconds,
            observation_count=track.observation_count,
            lost_frames=track.lost_frames,
            final_class=track.final_class,
            final_class_reason=track.final_class_reason,
            class_counts=dict(track.class_counts),
            class_confidence_sums=dict(track.class_confidence_sums),
            observations=[
                TrackObservation(
                    camera_id=item.camera_id,
                    tracker_namespace=item.tracker_namespace,
                    native_tracker_id=item.native_tracker_id,
                    local_track_id=item.local_track_id,
                    frame_number=item.frame_number,
                    timestamp_seconds=item.timestamp_seconds,
                    bbox_xyxy=tuple(item.bbox_xyxy),
                    confidence=item.confidence,
                    raw_class_id=item.raw_class_id,
                    raw_class_name=item.raw_class_name,
                )
                for item in track.observations
            ],
            completion_reason=track.completion_reason,
        )
