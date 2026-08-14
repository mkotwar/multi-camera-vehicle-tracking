from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import supervision as sv
import yaml
from supervision.tracker.byte_tracker import core as bt_core

from .detector_tracker import VehicleDetectorTracker
from .models import FramePacket


TARGET_IDS = [6, 12, 25, 27, 41]


class AssociationDebugByteTrack(sv.ByteTrack):
    def __init__(self, *args: Any, output_dir: Path, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.output_dir = output_dir
        self.matrix_dir = output_dir / "association_matrices"
        self.matrix_dir.mkdir(parents=True, exist_ok=True)
        self.events: list[dict[str, Any]] = []
        self.frame_outputs: list[dict[str, Any]] = []

    def update_with_tensors(self, tensors: np.ndarray) -> list[Any]:
        self.frame_id += 1
        frame_number = self.frame_id - 1
        activated_stracks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        scores = tensors[:, 4] if len(tensors) else np.array([], dtype=float)
        bboxes = tensors[:, :4] if len(tensors) else np.empty((0, 4), dtype=float)
        remain_inds = scores > self.track_activation_threshold
        inds_low = scores > 0.1
        inds_high = scores < self.track_activation_threshold
        inds_second = np.logical_and(inds_low, inds_high)
        dets_second = bboxes[inds_second]
        dets = bboxes[remain_inds]
        scores_keep = scores[remain_inds]
        scores_second = scores[inds_second]

        detections = [
            bt_core.STrack(
                bt_core.STrack.tlbr_to_tlwh(tlbr),
                score_keep,
                self.minimum_consecutive_frames,
                self.shared_kalman,
                self.internal_id_counter,
                self.external_id_counter,
            )
            for (tlbr, score_keep) in zip(dets, scores_keep)
        ]
        unconfirmed = []
        tracked_stracks = []
        for track in self.tracked_tracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        strack_pool = bt_core.joint_tracks(tracked_stracks, self.lost_tracks)
        bt_core.STrack.multi_predict(strack_pool, self.shared_kalman)
        iou_dists = bt_core.matching.iou_distance(strack_pool, detections)
        fused_dists = bt_core.matching.fuse_score(iou_dists.copy(), detections) if len(detections) else iou_dists
        matches, u_track, u_detection = bt_core.matching.linear_assignment(fused_dists, thresh=self.minimum_matching_threshold)
        self._write_matrix(frame_number, strack_pool, detections, iou_dists, fused_dists, matches, u_track, u_detection)

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            event_type = "first_pass_match" if track.state == bt_core.TrackState.Tracked else "first_pass_reactivation"
            self._event(frame_number, event_type, track=track, detection=det, detection_index=int(idet))
            if track.state == bt_core.TrackState.Tracked:
                track.update(detections[idet], self.frame_id)
                activated_stracks.append(track)
            else:
                track.re_activate(det, self.frame_id)
                refind_stracks.append(track)

        detections_second = [
            bt_core.STrack(
                bt_core.STrack.tlbr_to_tlwh(tlbr),
                score_second,
                self.minimum_consecutive_frames,
                self.shared_kalman,
                self.internal_id_counter,
                self.external_id_counter,
            )
            for (tlbr, score_second) in zip(dets_second, scores_second)
        ]
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == bt_core.TrackState.Tracked]
        second_dists = bt_core.matching.iou_distance(r_tracked_stracks, detections_second)
        second_matches, second_u_track, second_u_detection = bt_core.matching.linear_assignment(second_dists, thresh=0.5)
        self._event(
            frame_number,
            "second_pass_summary",
            payload={
                "eligible_track_ids": [int(t.external_track_id) for t in r_tracked_stracks],
                "low_conf_detection_count": len(detections_second),
                "matches": [[int(a), int(b)] for a, b in second_matches],
                "unmatched_track_indexes": [int(x) for x in second_u_track],
                "unmatched_detection_indexes": [int(x) for x in second_u_detection],
            },
        )
        for itracked, idet in second_matches:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            self._event(frame_number, "second_pass_match", track=track, detection=det, detection_index=int(idet))
            if track.state == bt_core.TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_stracks.append(track)
            else:
                track.re_activate(det, self.frame_id)
                refind_stracks.append(track)

        for it in second_u_track:
            track = r_tracked_stracks[it]
            if not track.state == bt_core.TrackState.Lost:
                self._event(frame_number, "marked_lost", track=track)
                track.state = bt_core.TrackState.Lost
                lost_stracks.append(track)

        detections = [detections[i] for i in u_detection]
        unconfirmed_dists = bt_core.matching.iou_distance(unconfirmed, detections)
        unconfirmed_dists = bt_core.matching.fuse_score(unconfirmed_dists, detections) if len(detections) else unconfirmed_dists
        unconfirmed_matches, u_unconfirmed, u_detection = bt_core.matching.linear_assignment(unconfirmed_dists, thresh=0.7)
        for itracked, idet in unconfirmed_matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_stracks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            self._event(frame_number, "removed_unconfirmed", track=track)
            track.state = bt_core.TrackState.Removed
            removed_stracks.append(track)

        for inew in u_detection:
            track = detections[inew]
            if track.score < self.det_thresh:
                self._event(frame_number, "new_track_rejected_det_thresh", detection=track, detection_index=int(inew))
                continue
            track.activate(self.kalman_filter, self.frame_id)
            self._event(frame_number, "new_track_created", track=track, detection=track, detection_index=int(inew))
            activated_stracks.append(track)
        for track in self.lost_tracks:
            if self.frame_id - track.frame_id > self.max_time_lost:
                self._event(frame_number, "removed_lost_timeout", track=track)
                track.state = bt_core.TrackState.Removed
                removed_stracks.append(track)

        self.tracked_tracks = [t for t in self.tracked_tracks if t.state == bt_core.TrackState.Tracked]
        self.tracked_tracks = bt_core.joint_tracks(self.tracked_tracks, activated_stracks)
        self.tracked_tracks = bt_core.joint_tracks(self.tracked_tracks, refind_stracks)
        self.lost_tracks = bt_core.sub_tracks(self.lost_tracks, self.tracked_tracks)
        self.lost_tracks.extend(lost_stracks)
        self.lost_tracks = bt_core.sub_tracks(self.lost_tracks, self.removed_tracks)
        self.removed_tracks = removed_stracks
        self.tracked_tracks, self.lost_tracks = bt_core.remove_duplicate_tracks(self.tracked_tracks, self.lost_tracks)
        output = [track for track in self.tracked_tracks if track.is_activated]
        self.frame_outputs.append(
            {
                "frame_number": frame_number,
                "output_ids": [int(track.external_track_id) for track in output],
                "tracked_ids": [int(track.external_track_id) for track in self.tracked_tracks],
                "lost_ids": [int(track.external_track_id) for track in self.lost_tracks],
                "removed_ids": [int(track.external_track_id) for track in self.removed_tracks],
            }
        )
        return output

    def flush(self) -> None:
        _write_jsonl(self.output_dir / "bytetrack_events.jsonl", self.events)
        _write_jsonl(self.output_dir / "bytetrack_frame_outputs.jsonl", self.frame_outputs)

    def _write_matrix(
        self,
        frame_number: int,
        tracks: list[Any],
        detections: list[Any],
        iou_dists: np.ndarray,
        fused_dists: np.ndarray,
        matches: np.ndarray,
        u_track: np.ndarray,
        u_detection: np.ndarray,
    ) -> None:
        rows = []
        matched_track_indexes = {int(a): int(b) for a, b in matches}
        for ti, track in enumerate(tracks):
            for di, det in enumerate(detections):
                rows.append(
                    {
                        "frame_number": frame_number,
                        "track_index": ti,
                        "track_id": int(track.external_track_id),
                        "track_state": str(track.state).split(".")[-1],
                        "track_bbox": json.dumps(_box(track.tlbr)),
                        "detection_index": di,
                        "detection_bbox": json.dumps(_box(det.tlbr)),
                        "detection_confidence": float(det.score),
                        "iou": float(1.0 - iou_dists[ti, di]),
                        "iou_distance": float(iou_dists[ti, di]),
                        "fused_cost": float(fused_dists[ti, di]),
                        "threshold": float(self.minimum_matching_threshold),
                        "hungarian_assigned": matched_track_indexes.get(ti) == di,
                        "accepted": matched_track_indexes.get(ti) == di and float(fused_dists[ti, di]) <= float(self.minimum_matching_threshold),
                        "track_unmatched": int(ti) in {int(x) for x in u_track},
                        "detection_unmatched": int(di) in {int(x) for x in u_detection},
                    }
                )
        _write_csv(self.matrix_dir / f"frame_{frame_number:05d}.csv", rows)

    def _event(
        self,
        frame_number: int,
        event_type: str,
        *,
        track: Any | None = None,
        detection: Any | None = None,
        detection_index: int | None = None,
        payload: dict[str, Any] | None = None,
    ) -> None:
        row = {"frame_number": frame_number, "event_type": event_type}
        if track is not None:
            row.update(
                {
                    "track_id": int(track.external_track_id),
                    "track_state": str(track.state).split(".")[-1],
                    "track_bbox": _box(track.tlbr),
                    "track_score": float(getattr(track, "score", 0.0)),
                    "track_frame_id": int(getattr(track, "frame_id", -1)),
                    "track_velocity": _velocity(track),
                }
            )
        if detection is not None:
            row.update(
                {
                    "detection_index": detection_index,
                    "detection_bbox": _box(detection.tlbr),
                    "detection_confidence": float(detection.score),
                }
            )
        if payload:
            row.update(payload)
        self.events.append(row)


def run_vehicle_association_debug(run_dir: str | Path) -> dict[str, Any]:
    run_path = Path(run_dir)
    output_dir = run_path / "vehicle_association_debug" / "yellow_plate_car"
    output_dir.mkdir(parents=True, exist_ok=True)
    matrix_dir = output_dir / "association_matrices"
    matrix_dir.mkdir(exist_ok=True)

    config = yaml.safe_load((run_path / "run_config.yaml").read_text(encoding="utf-8"))
    video_path = Path(config["input"]["cameras"][0]["source"])
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")
    source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    debug_trackers: list[AssociationDebugByteTrack] = []

    def tracker_factory(frame_rate: float) -> AssociationDebugByteTrack:
        tracker = AssociationDebugByteTrack(
            lost_track_buffer=int(config["tracking"]["lost_track_buffer"]),
            track_activation_threshold=float(config["tracking"]["track_activation_threshold"]),
            minimum_matching_threshold=float(config["tracking"]["minimum_matching_threshold"]),
            frame_rate=float(frame_rate),
            minimum_consecutive_frames=int(config["tracking"]["minimum_consecutive_frames"]),
            output_dir=output_dir,
        )
        debug_trackers.append(tracker)
        return tracker

    detector = VehicleDetectorTracker(config, _NullLogger(), tracker_factory=tracker_factory)
    rows: list[dict[str, Any]] = []
    frame_number = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        packet = FramePacket(
            camera_id="CAM_001",
            frame_number=frame_number,
            timestamp_seconds=frame_number / source_fps,
            source_fps=source_fps,
            frame=frame,
            source_frame_width=width,
            source_frame_height=height,
            worker_id=0,
            captured_at="",
            source_type="video",
        )
        result = detector.process_frame(packet)
        rows.append(
            {
                "frame_number": frame_number,
                "timestamp_seconds": packet.timestamp_seconds,
                "pretracker_detection_count": len(result.detections),
                "tracked_ids": json.dumps([item.tracker_id for item in result.tracked_detections]),
                "tracked_count": len(result.tracked_detections),
            }
        )
        frame_number += 1
    cap.release()
    for tracker in debug_trackers:
        tracker.flush()
    _write_csv(output_dir / "runtime_frame_summary.csv", rows)
    summary = {
        "run_id": run_path.name,
        "video_path": str(video_path),
        "source_fps": source_fps,
        "processed_frames": frame_number,
        "tracker_count": len(debug_trackers),
        "effective_config": {
            "tracking": config["tracking"],
            "tracking_roi": config["tracking_roi"],
        },
    }
    (output_dir / "runtime_debug_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


class _NullLogger:
    def debug(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def info(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def warning(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def error(self, *_args: Any, **_kwargs: Any) -> None:
        return None


def _box(value: Any) -> list[float]:
    return [float(item) for item in value]


def _velocity(track: Any) -> list[float]:
    mean = getattr(track, "mean", None)
    if mean is None or len(mean) < 8:
        return []
    return [float(mean[index]) for index in range(4, 8)]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
