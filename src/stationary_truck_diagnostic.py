from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import supervision as sv
import yaml
from supervision.tracker.byte_tracker import core as bt_core


TARGET_TRACK_IDS = [6, 14, 78, 100, 133, 143]
TARGET_LOCAL_IDS = [f"CAM_001:TRACK_{track_id}" for track_id in TARGET_TRACK_IDS]
TRANSITIONS = list(zip(TARGET_LOCAL_IDS[:-1], TARGET_LOCAL_IDS[1:]))
CLASS_IDS = {"car": 2, "motorcycle": 3, "truck": 4, "bus": 5, "3wheeler": 1}


@dataclass(frozen=True)
class ReplayFrame:
    frame_number: int
    output_ids: list[int]
    tracked_ids: list[int]
    lost_ids: list[int]
    removed_ids: list[int]


def run_stationary_truck_diagnostic(run_dir: str | Path, *, output_dir: str | Path | None = None) -> dict[str, Any]:
    run_path = Path(run_dir)
    debug_dir = Path(output_dir) if output_dir else run_path / "stationary_truck_debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    (debug_dir / "transitions").mkdir(exist_ok=True)

    config = yaml.safe_load((run_path / "run_config.yaml").read_text(encoding="utf-8"))
    metadata = json.loads((run_path / "run_metadata.json").read_text(encoding="utf-8"))
    tracks = json.loads((run_path / "tracks.json").read_text(encoding="utf-8"))
    bbox_metrics = json.loads((run_path / "bbox_quality_metrics.json").read_text(encoding="utf-8"))
    observations = _read_observations(run_path / "observations.csv")

    tracks_by_id = {str(track.get("local_track_id")): track for track in tracks}
    observations_by_id = {track_id: [row for row in observations if row["local_track_id"] == track_id] for track_id in TARGET_LOCAL_IDS}
    for rows in observations_by_id.values():
        rows.sort(key=lambda row: int(row["frame_number"]))

    video_path = Path(config["input"]["cameras"][0]["source"])
    video_info = _read_video_info(video_path)
    processed_fps = _estimate_processed_fps(observations)

    raw_detections = list(bbox_metrics.get("detections", []))
    frame_height = int(raw_detections[0].get("frame_height", video_info["height"])) if raw_detections else video_info["height"]
    roi_cfg = dict(config.get("tracking_roi", {}) or {})
    raw_by_frame: dict[int, list[dict[str, Any]]] = {}
    passed_by_frame: dict[int, list[dict[str, Any]]] = {}
    for row in raw_detections:
        frame_number = int(row["frame_number"])
        raw_by_frame.setdefault(frame_number, []).append(row)
        if bool(row.get("accepted_by_bbox_quality", True)) and _inside_roi(row, frame_height, roi_cfg):
            passed_by_frame.setdefault(frame_number, []).append(row)

    target_bbox = _median_target_bbox(observations_by_id)
    track_boundaries = _write_track_boundaries(debug_dir / "track_boundaries.csv", tracks_by_id, observations_by_id)

    replay, association_rows = _replay_bytetrack(config, passed_by_frame, int(metadata["processed_frames"]), TARGET_TRACK_IDS)
    _write_csv(debug_dir / "association_diagnostics.csv", association_rows)

    frame_rows, confidence_rows, iou_rows, transition_summaries = _build_frame_diagnostics(
        config=config,
        track_boundaries=track_boundaries,
        raw_by_frame=raw_by_frame,
        passed_by_frame=passed_by_frame,
        observations=observations,
        observations_by_id=observations_by_id,
        target_bbox=target_bbox,
        replay=replay,
    )
    _write_csv(debug_dir / "frame_diagnostics.csv", frame_rows)
    _write_csv(debug_dir / "confidence_timeline.csv", confidence_rows)
    _write_csv(debug_dir / "bbox_iou_timeline.csv", iou_rows)

    contact_sheet_path = debug_dir / "identity_contact_sheet.jpg"
    _make_identity_contact_sheet(run_path, tracks_by_id, contact_sheet_path)
    _make_transition_sheets(run_path, debug_dir, track_boundaries, raw_by_frame, observations)

    run_summary = _build_summary(
        run_path=run_path,
        config=config,
        metadata=metadata,
        tracks_by_id=tracks_by_id,
        observations_by_id=observations_by_id,
        track_boundaries=track_boundaries,
        transition_summaries=transition_summaries,
        association_rows=association_rows,
        video_info=video_info,
        processed_fps=processed_fps,
        replay=replay,
    )
    (debug_dir / "summary.json").write_text(json.dumps(run_summary, indent=2), encoding="utf-8")
    return run_summary


def _read_observations(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _read_video_info(video_path: Path) -> dict[str, Any]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {"path": str(video_path), "fps": None, "frame_count": None, "width": None, "height": None}
    try:
        return {
            "path": str(video_path),
            "fps": float(cap.get(cv2.CAP_PROP_FPS)),
            "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        }
    finally:
        cap.release()


def _estimate_processed_fps(observations: list[dict[str, Any]]) -> float | None:
    pairs = [(int(row["frame_number"]), float(row["timestamp_seconds"])) for row in observations if row.get("timestamp_seconds")]
    if len(pairs) < 2:
        return None
    first_frame, first_time = min(pairs)
    last_frame, last_time = max(pairs)
    if last_time <= first_time:
        return None
    return float((last_frame - first_frame) / (last_time - first_time))


def _inside_roi(row: dict[str, Any], frame_height: int, roi_cfg: dict[str, Any]) -> bool:
    if not bool(roi_cfg.get("enabled", False)):
        return True
    if str(roi_cfg.get("anchor", "bottom_center")) != "bottom_center":
        return True
    top = frame_height * float(roi_cfg.get("top_fraction", 0.0))
    bottom = frame_height * (1.0 - float(roi_cfg.get("bottom_fraction", 0.0)))
    y2 = float(row["bbox_xyxy"][3])
    return top <= y2 <= bottom


def _median_target_bbox(observations_by_id: dict[str, list[dict[str, Any]]]) -> list[float]:
    boxes = []
    for rows in observations_by_id.values():
        for row in rows:
            boxes.append([float(row["x1"]), float(row["y1"]), float(row["x2"]), float(row["y2"])])
    return [float(value) for value in np.median(np.asarray(boxes, dtype=float), axis=0)]


def _write_track_boundaries(path: Path, tracks_by_id: dict[str, dict[str, Any]], observations_by_id: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    rows = []
    for old_id, new_id in TRANSITIONS:
        old_rows = observations_by_id[old_id]
        new_rows = observations_by_id[new_id]
        old_last = old_rows[-1]
        new_first = new_rows[0]
        gap_frames = int(new_first["frame_number"]) - int(old_last["frame_number"])
        gap_seconds = float(new_first["timestamp_seconds"]) - float(old_last["timestamp_seconds"])
        rows.append(
            {
                "transition": f"{old_id.split(':')[-1]}_to_{new_id.split(':')[-1]}",
                "old_track": old_id,
                "new_track": new_id,
                "old_native_id": tracks_by_id[old_id]["native_tracker_id"],
                "new_native_id": tracks_by_id[new_id]["native_tracker_id"],
                "old_last_frame": int(old_last["frame_number"]),
                "old_last_timestamp_seconds": float(old_last["timestamp_seconds"]),
                "new_first_frame": int(new_first["frame_number"]),
                "new_first_timestamp_seconds": float(new_first["timestamp_seconds"]),
                "gap_frames": gap_frames,
                "gap_seconds": gap_seconds,
            }
        )
    _write_csv(path, rows)
    return rows


def _candidate_rows(rows: list[dict[str, Any]], target_bbox: list[float], *, min_iou: float = 0.08) -> list[dict[str, Any]]:
    candidates = []
    x1, y1, x2, y2 = target_bbox
    expanded = [x1 - 80, y1 - 80, x2 + 80, y2 + 80]
    for row in rows:
        name = str(row.get("normalized_class_name") or row.get("class_name") or "").lower()
        if name not in {"truck", "bus", "car"}:
            continue
        bbox = [float(value) for value in row["bbox_xyxy"]]
        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0
        overlap = _iou(bbox, target_bbox)
        if overlap >= min_iou or (expanded[0] <= cx <= expanded[2] and expanded[1] <= cy <= expanded[3]):
            enriched = dict(row)
            enriched["_target_iou"] = overlap
            candidates.append(enriched)
    return sorted(candidates, key=lambda item: (float(item.get("_target_iou", 0.0)), float(item.get("confidence", 0.0))), reverse=True)


def _replay_bytetrack(
    config: dict[str, Any],
    passed_by_frame: dict[int, list[dict[str, Any]]],
    frame_count: int,
    watch_ids: list[int],
) -> tuple[dict[int, ReplayFrame], list[dict[str, Any]]]:
    tracking = dict(config.get("tracking", {}) or {})
    tracker = DiagnosticByteTrack(
        watch_ids=set(watch_ids),
        lost_track_buffer=int(tracking.get("lost_track_buffer", 40)),
        track_activation_threshold=float(tracking.get("track_activation_threshold", 0.3)),
        minimum_matching_threshold=float(tracking.get("minimum_matching_threshold", 0.6)),
        minimum_consecutive_frames=int(tracking.get("minimum_consecutive_frames", 3)),
    )
    replay: dict[int, ReplayFrame] = {}
    for frame_number in range(frame_count):
        detections = _to_supervision(passed_by_frame.get(frame_number, []))
        tracked = tracker.update_with_detections(detections)
        replay[frame_number] = ReplayFrame(
            frame_number=frame_number,
            output_ids=[int(item) for item in getattr(tracked, "tracker_id", [])],
            tracked_ids=_state_ids(tracker.tracked_tracks),
            lost_ids=_state_ids(tracker.lost_tracks),
            removed_ids=_state_ids(tracker.removed_tracks),
        )
    return replay, tracker.diagnostics


def _to_supervision(rows: list[dict[str, Any]]) -> sv.Detections:
    if not rows:
        empty = sv.Detections.empty()
        empty.tracker_id = np.array([], dtype=int)
        return empty
    xyxy = np.asarray([row["bbox_xyxy"] for row in rows], dtype=np.float32)
    confidence = np.asarray([float(row["confidence"]) for row in rows], dtype=np.float32)
    class_id = np.asarray([CLASS_IDS.get(str(row.get("normalized_class_name") or row.get("class_name")).lower(), 0) for row in rows], dtype=np.int32)
    return sv.Detections(xyxy=xyxy, confidence=confidence, class_id=class_id)


def _state_ids(tracks: list[Any]) -> list[int]:
    return sorted(int(track.external_track_id) for track in tracks if hasattr(track, "external_track_id"))


class DiagnosticByteTrack(sv.ByteTrack):
    def __init__(self, *args: Any, watch_ids: set[int], **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.watch_ids = watch_ids
        self.diagnostics: list[dict[str, Any]] = []

    def update_with_tensors(self, tensors: np.ndarray) -> list[Any]:
        self.frame_id += 1
        frame_number = self.frame_id - 1
        activated_starcks = []
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
            bt_core.STrack(bt_core.STrack.tlbr_to_tlwh(tlbr), score_keep, self.minimum_consecutive_frames, self.shared_kalman, self.internal_id_counter, self.external_id_counter)
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
        dists = bt_core.matching.iou_distance(strack_pool, detections)
        fused_dists = bt_core.matching.fuse_score(dists.copy(), detections) if len(detections) else dists
        matches, u_track, u_detection = bt_core.matching.linear_assignment(fused_dists, thresh=self.minimum_matching_threshold)
        self._record_first_pass(frame_number, strack_pool, detections, fused_dists, matches, len(dets_second))

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == bt_core.TrackState.Tracked:
                track.update(detections[idet], self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id)
                refind_stracks.append(track)

        detections_second = [
            bt_core.STrack(bt_core.STrack.tlbr_to_tlwh(tlbr), score_second, self.minimum_consecutive_frames, self.shared_kalman, self.internal_id_counter, self.external_id_counter)
            for (tlbr, score_second) in zip(dets_second, scores_second)
        ]
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == bt_core.TrackState.Tracked]
        second_dists = bt_core.matching.iou_distance(r_tracked_stracks, detections_second)
        matches, u_track, _u_detection_second = bt_core.matching.linear_assignment(second_dists, thresh=0.5)
        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            if track.state == bt_core.TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id)
                refind_stracks.append(track)

        for it in u_track:
            track = r_tracked_stracks[it]
            if not track.state == bt_core.TrackState.Lost:
                track.state = bt_core.TrackState.Lost
                lost_stracks.append(track)

        detections = [detections[i] for i in u_detection]
        dists = bt_core.matching.iou_distance(unconfirmed, detections)
        dists = bt_core.matching.fuse_score(dists, detections) if len(detections) else dists
        matches, u_unconfirmed, u_detection = bt_core.matching.linear_assignment(dists, thresh=0.7)
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_starcks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.state = bt_core.TrackState.Removed
            removed_stracks.append(track)

        for inew in u_detection:
            track = detections[inew]
            if track.score < self.det_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            activated_starcks.append(track)
        for track in self.lost_tracks:
            if self.frame_id - track.frame_id > self.max_time_lost:
                track.state = bt_core.TrackState.Removed
                removed_stracks.append(track)

        self.tracked_tracks = [t for t in self.tracked_tracks if t.state == bt_core.TrackState.Tracked]
        self.tracked_tracks = bt_core.joint_tracks(self.tracked_tracks, activated_starcks)
        self.tracked_tracks = bt_core.joint_tracks(self.tracked_tracks, refind_stracks)
        self.lost_tracks = bt_core.sub_tracks(self.lost_tracks, self.tracked_tracks)
        self.lost_tracks.extend(lost_stracks)
        self.lost_tracks = bt_core.sub_tracks(self.lost_tracks, self.removed_tracks)
        self.removed_tracks = removed_stracks
        self.tracked_tracks, self.lost_tracks = bt_core.remove_duplicate_tracks(self.tracked_tracks, self.lost_tracks)
        return [track for track in self.tracked_tracks if track.is_activated]

    def _record_first_pass(self, frame_number: int, pool: list[Any], detections: list[Any], dists: np.ndarray, matches: np.ndarray, low_count: int) -> None:
        matched_track_indexes = {int(pair[0]): int(pair[1]) for pair in matches}
        for track_index, track in enumerate(pool):
            external_id = int(track.external_track_id)
            if external_id not in self.watch_ids:
                continue
            best_index = None
            best_distance = None
            best_iou = None
            if len(detections):
                distances = dists[track_index]
                best_index = int(np.argmin(distances))
                best_distance = float(distances[best_index])
                best_iou = float(_iou(track.tlbr, detections[best_index].tlbr))
            matched_detection = matched_track_indexes.get(track_index)
            self.diagnostics.append(
                {
                    "frame_number": frame_number,
                    "watched_native_id": external_id,
                    "track_state_before": str(track.state).split(".")[-1],
                    "predicted_bbox": _box_list(track.tlbr),
                    "high_conf_detection_count": len(detections),
                    "low_conf_detection_count": low_count,
                    "best_high_detection_index": best_index,
                    "best_high_detection_bbox": _box_list(detections[best_index].tlbr) if best_index is not None else None,
                    "best_high_detection_score": float(detections[best_index].score) if best_index is not None else None,
                    "best_high_detection_iou": best_iou,
                    "best_fused_distance": best_distance,
                    "matching_threshold": float(self.minimum_matching_threshold),
                    "first_pass_matched_detection_index": matched_detection,
                    "first_pass_match_accepted": matched_detection is not None,
                }
            )


def _build_frame_diagnostics(
    *,
    config: dict[str, Any],
    track_boundaries: list[dict[str, Any]],
    raw_by_frame: dict[int, list[dict[str, Any]]],
    passed_by_frame: dict[int, list[dict[str, Any]]],
    observations: list[dict[str, Any]],
    observations_by_id: dict[str, list[dict[str, Any]]],
    target_bbox: list[float],
    replay: dict[int, ReplayFrame],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    obs_by_frame: dict[int, list[dict[str, Any]]] = {}
    for row in observations:
        obs_by_frame.setdefault(int(row["frame_number"]), []).append(row)
    rows = []
    confidence_rows = []
    iou_rows = []
    transition_summaries = []
    activation = float(config["tracking"]["track_activation_threshold"])
    matching = float(config["tracking"]["minimum_matching_threshold"])
    for boundary in track_boundaries:
        old_id = boundary["old_track"]
        new_id = boundary["new_track"]
        window_start = max(0, int(boundary["old_last_frame"]) - 10)
        window_end = int(boundary["new_first_frame"]) + 10
        previous_primary = None
        primary_by_frame: dict[int, dict[str, Any] | None] = {}
        for frame_number in range(window_start, window_end + 1):
            raw_candidates = _candidate_rows(raw_by_frame.get(frame_number, []), target_bbox)
            passed_candidates = _candidate_rows(passed_by_frame.get(frame_number, []), target_bbox)
            primary = passed_candidates[0] if passed_candidates else (raw_candidates[0] if raw_candidates else None)
            primary_by_frame[frame_number] = primary
            tracked_rows = [row for row in obs_by_frame.get(frame_number, []) if row["local_track_id"] in {old_id, new_id}]
            tracked_any_target = [row for row in obs_by_frame.get(frame_number, []) if row["local_track_id"] in TARGET_LOCAL_IDS]
            replay_frame = replay.get(frame_number)
            row = {
                "transition": boundary["transition"],
                "frame_number": frame_number,
                "timestamp_seconds": _timestamp_for_frame(frame_number, observations_by_id),
                "raw_candidate_count": len(raw_candidates),
                "passed_candidate_count": len(passed_candidates),
                "truck_detected_by_yolo": bool(raw_candidates),
                "truck_detection_passed_to_bytetrack": bool(passed_candidates),
                "primary_class": primary.get("normalized_class_name") if primary else None,
                "primary_confidence": float(primary["confidence"]) if primary else None,
                "primary_confidence_band": _confidence_band(float(primary["confidence"]), activation) if primary else "missing",
                "primary_bbox": json.dumps(primary["bbox_xyxy"]) if primary else None,
                "primary_bbox_width": float(primary["bbox_width"]) if primary else None,
                "primary_bbox_height": float(primary["bbox_height"]) if primary else None,
                "primary_bbox_area": float(primary["bbox_area"]) if primary else None,
                "primary_center_x": _center(primary)[0] if primary else None,
                "primary_center_y": _center(primary)[1] if primary else None,
                "consecutive_primary_iou": _iou(primary["bbox_xyxy"], previous_primary["bbox_xyxy"]) if primary and previous_primary else None,
                "center_displacement": _center_distance(primary, previous_primary) if primary and previous_primary else None,
                "tracked_old_or_new_present": bool(tracked_rows),
                "target_track_outputs": ";".join(f"{r['local_track_id']}@{r['native_tracker_id']}" for r in tracked_any_target),
                "old_track_manager_state": _manager_state(old_id, frame_number, observations_by_id),
                "new_track_manager_state": _manager_state(new_id, frame_number, observations_by_id),
                "replay_output_ids": json.dumps(replay_frame.output_ids if replay_frame else []),
                "replay_tracked_ids": json.dumps(replay_frame.tracked_ids if replay_frame else []),
                "replay_lost_ids": json.dumps(replay_frame.lost_ids if replay_frame else []),
                "replay_removed_ids": json.dumps(replay_frame.removed_ids if replay_frame else []),
                "overlapping_non_target_detections": len(_overlapping_non_target(raw_by_frame.get(frame_number, []), primary["bbox_xyxy"] if primary else target_bbox)),
            }
            rows.append(row)
            if primary:
                confidence_rows.append(
                    {
                        "transition": boundary["transition"],
                        "frame_number": frame_number,
                        "confidence": float(primary["confidence"]),
                        "confidence_band": row["primary_confidence_band"],
                        "track_activation_threshold": activation,
                        "minimum_matching_threshold": matching,
                    }
                )
            if primary and previous_primary:
                iou_rows.append(
                    {
                        "transition": boundary["transition"],
                        "frame_number": frame_number,
                        "previous_frame": frame_number - 1,
                        "iou": row["consecutive_primary_iou"],
                        "center_displacement": row["center_displacement"],
                        "width_ratio": float(primary["bbox_width"]) / max(float(previous_primary["bbox_width"]), 1e-6),
                        "height_ratio": float(primary["bbox_height"]) / max(float(previous_primary["bbox_height"]), 1e-6),
                        "area_ratio": float(primary["bbox_area"]) / max(float(previous_primary["bbox_area"]), 1e-6),
                    }
                )
            previous_primary = primary or previous_primary
        gap_frames = range(int(boundary["old_last_frame"]) + 1, int(boundary["new_first_frame"]))
        gap_primaries = [primary_by_frame.get(frame) for frame in gap_frames]
        missing = [frame for frame, primary in zip(gap_frames, gap_primaries) if primary is None]
        low = [frame for frame, primary in zip(gap_frames, gap_primaries) if primary and float(primary["confidence"]) <= activation]
        ious = [float(row["iou"]) for row in iou_rows if row["transition"] == boundary["transition"] and row["iou"] is not None]
        overlaps = [int(row["overlapping_non_target_detections"]) for row in rows if row["transition"] == boundary["transition"]]
        transition_summaries.append(
            {
                "transition": boundary["transition"],
                "detector_dropout": bool(missing),
                "missing_frames": missing,
                "low_confidence_frames": low,
                "minimum_consecutive_iou": min(ious) if ious else None,
                "box_jump": min(ious) < 0.45 if ious else None,
                "maximum_overlapping_non_target_detections": max(overlaps) if overlaps else 0,
                "occlusion_suspected": max(overlaps) > 0 if overlaps else False,
            }
        )
    return rows, confidence_rows, iou_rows, transition_summaries


def _timestamp_for_frame(frame_number: int, observations_by_id: dict[str, list[dict[str, Any]]]) -> float:
    for rows in observations_by_id.values():
        for row in rows:
            if int(row["frame_number"]) == frame_number:
                return float(row["timestamp_seconds"])
    return frame_number / 29.970731707317075


def _manager_state(local_id: str, frame_number: int, observations_by_id: dict[str, list[dict[str, Any]]]) -> str:
    rows = observations_by_id.get(local_id, [])
    if not rows:
        return "UNKNOWN"
    first = int(rows[0]["frame_number"])
    last = int(rows[-1]["frame_number"])
    observed_frames = {int(row["frame_number"]) for row in rows}
    if frame_number in observed_frames:
        return "ACTIVE"
    if frame_number < first:
        return "NOT_CREATED"
    if frame_number <= last + 40:
        return "LOST"
    return "COMPLETED"


def _confidence_band(confidence: float, activation: float) -> str:
    if confidence > activation:
        return "high_confidence_path"
    if confidence > 0.1:
        return "low_confidence_path"
    return "below_bytetrack_low_confidence"


def _center(row: dict[str, Any]) -> tuple[float, float]:
    x1, y1, x2, y2 = [float(value) for value in row["bbox_xyxy"]]
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def _center_distance(left: dict[str, Any], right: dict[str, Any]) -> float:
    lx, ly = _center(left)
    rx, ry = _center(right)
    return float(math.hypot(lx - rx, ly - ry))


def _overlapping_non_target(rows: list[dict[str, Any]], bbox: list[float]) -> list[dict[str, Any]]:
    return [row for row in rows if _iou(row["bbox_xyxy"], bbox) > 0.05 and str(row.get("normalized_class_name", "")).lower() not in {"truck", "bus"}]


def _make_identity_contact_sheet(run_path: Path, tracks_by_id: dict[str, dict[str, Any]], output_path: Path) -> None:
    tile_w, tile_h = 220, 150
    roles = ["FIRST", "MIDDLE", "LAST", "BEST_OVERALL"]
    sheet = np.full((len(TARGET_LOCAL_IDS) * tile_h, len(roles) * tile_w, 3), 245, dtype=np.uint8)
    for row_index, local_id in enumerate(TARGET_LOCAL_IDS):
        records = _read_evidence_records(run_path, local_id)
        by_role = {str(record.get("role")): record for record in records}
        track = tracks_by_id[local_id]
        for col_index, role in enumerate(roles):
            record = by_role.get(role)
            tile = np.full((tile_h, tile_w, 3), 235, dtype=np.uint8)
            if record and Path(record["crop_path"]).exists():
                image = cv2.imread(str(record["crop_path"]))
                tile = _fit_image(image, tile_w, tile_h)
            label = f"{local_id.split(':')[-1]} {role} {track.get('final_class')} {((track.get('vehicle_enrichment') or {}).get('vehicle_colour') or {}).get('label')}"
            cv2.putText(tile, label[:34], (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
            y0, x0 = row_index * tile_h, col_index * tile_w
            sheet[y0 : y0 + tile_h, x0 : x0 + tile_w] = tile
    cv2.imwrite(str(output_path), sheet)


def _read_evidence_records(run_path: Path, local_id: str) -> list[dict[str, Any]]:
    safe = local_id.replace(":", "_")
    path = run_path / "evidence" / "CAM_001" / safe / "evidence.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else []


def _fit_image(image: np.ndarray | None, width: int, height: int) -> np.ndarray:
    tile = np.full((height, width, 3), 245, dtype=np.uint8)
    if image is None:
        return tile
    h, w = image.shape[:2]
    scale = min(width / max(w, 1), height / max(h, 1))
    resized = cv2.resize(image, (max(1, int(w * scale)), max(1, int(h * scale))))
    rh, rw = resized.shape[:2]
    y0 = (height - rh) // 2
    x0 = (width - rw) // 2
    tile[y0 : y0 + rh, x0 : x0 + rw] = resized
    return tile


def _make_transition_sheets(run_path: Path, debug_dir: Path, track_boundaries: list[dict[str, Any]], raw_by_frame: dict[int, list[dict[str, Any]]], observations: list[dict[str, Any]]) -> None:
    video_path = yaml.safe_load((run_path / "run_config.yaml").read_text(encoding="utf-8"))["input"]["cameras"][0]["source"]
    obs_by_frame: dict[int, list[dict[str, Any]]] = {}
    for row in observations:
        obs_by_frame.setdefault(int(row["frame_number"]), []).append(row)
    for boundary in track_boundaries:
        folder = debug_dir / "transitions" / boundary["transition"]
        folder.mkdir(parents=True, exist_ok=True)
        frames = sorted(set(np.linspace(max(0, int(boundary["old_last_frame"]) - 10), int(boundary["new_first_frame"]) + 10, num=12, dtype=int).tolist()))
        tiles = []
        for frame_number in frames:
            frame = _read_video_frame(video_path, frame_number)
            if frame is None:
                continue
            annotated = frame.copy()
            for row in raw_by_frame.get(frame_number, []):
                name = str(row.get("normalized_class_name") or row.get("class_name")).upper()
                if name not in {"TRUCK", "BUS", "CAR"}:
                    continue
                _draw_box(annotated, row["bbox_xyxy"], (0, 180, 255), f"YOLO {name} {float(row['confidence']):.2f}")
            for row in obs_by_frame.get(frame_number, []):
                if row["local_track_id"] in TARGET_LOCAL_IDS:
                    _draw_box(
                        annotated,
                        [float(row["x1"]), float(row["y1"]), float(row["x2"]), float(row["y2"])],
                        (80, 230, 80),
                        row["local_track_id"].split(":")[-1],
                    )
            cv2.putText(annotated, f"frame {frame_number}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 3)
            cv2.putText(annotated, f"frame {frame_number}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 1)
            tiles.append(_fit_image(annotated, 360, 210))
        if tiles:
            sheet = _grid(tiles, columns=3)
            cv2.imwrite(str(folder / "annotated_sequence_contact_sheet.jpg"), sheet)


def _read_video_frame(video_path: str, frame_number: int) -> np.ndarray | None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ok, frame = cap.read()
        return frame if ok else None
    finally:
        cap.release()


def _draw_box(image: np.ndarray, bbox: list[float], color: tuple[int, int, int], label: str) -> None:
    x1, y1, x2, y2 = [int(round(float(value))) for value in bbox]
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    cv2.putText(image, label, (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def _grid(tiles: list[np.ndarray], columns: int) -> np.ndarray:
    h, w = tiles[0].shape[:2]
    rows = int(math.ceil(len(tiles) / columns))
    sheet = np.full((rows * h, columns * w, 3), 245, dtype=np.uint8)
    for index, tile in enumerate(tiles):
        y = (index // columns) * h
        x = (index % columns) * w
        sheet[y : y + h, x : x + w] = tile
    return sheet


def _build_summary(
    *,
    run_path: Path,
    config: dict[str, Any],
    metadata: dict[str, Any],
    tracks_by_id: dict[str, dict[str, Any]],
    observations_by_id: dict[str, list[dict[str, Any]]],
    track_boundaries: list[dict[str, Any]],
    transition_summaries: list[dict[str, Any]],
    association_rows: list[dict[str, Any]],
    video_info: dict[str, Any],
    processed_fps: float | None,
    replay: dict[int, ReplayFrame],
) -> dict[str, Any]:
    replay_matches = _replay_match_summary(observations_by_id, replay)
    association_by_transition = {}
    for boundary in track_boundaries:
        old_native = int(boundary["old_native_id"])
        new_frame = int(boundary["new_first_frame"])
        near = [row for row in association_rows if int(row["watched_native_id"]) == old_native and new_frame - 3 <= int(row["frame_number"]) <= new_frame + 3]
        association_by_transition[boundary["transition"]] = near
    root_rows = []
    for transition in transition_summaries:
        root = _classify_root(transition, association_by_transition.get(transition["transition"], []))
        root_rows.append({**transition, "root_cause": root})
    return {
        "run": {
            "run_id": metadata.get("run_id"),
            "run_directory": str(run_path),
            "video_path": config["input"]["cameras"][0]["source"],
            "source_fps": video_info.get("fps"),
            "processed_fps_estimate": processed_fps,
            "processed_frame_count": metadata.get("processed_frames"),
            "tracking_backend": config["tracking"]["backend"],
            "track_activation_threshold": config["tracking"]["track_activation_threshold"],
            "lost_track_buffer": config["tracking"]["lost_track_buffer"],
            "minimum_matching_threshold": config["tracking"]["minimum_matching_threshold"],
            "minimum_consecutive_frames": config["tracking"]["minimum_consecutive_frames"],
            "maximum_lost_frames": config["lifecycle"]["maximum_lost_frames"],
            "yolo_confidence_threshold": config["detection"]["confidence_threshold"],
            "yolo_iou_threshold": config["detection"]["iou_threshold"],
            "yolo_agnostic_nms": config["detection"]["agnostic_nms"],
            "yolo_image_size": config["detection"]["image_size"],
        },
        "verified_fragments": [_track_identity_summary(tracks_by_id, observations_by_id, local_id) for local_id in TARGET_LOCAL_IDS],
        "track_boundaries": track_boundaries,
        "replay_match_summary": replay_matches,
        "root_cause_table": root_rows,
        "association_near_new_track_frames": association_by_transition,
        "primary_conclusion": _primary_conclusion(root_rows),
        "next_experiment": "Run a controlled ByteTrack replay A/B on the exact persisted bbox sequence with only association-related parameters varied, starting with lower minimum_matching_threshold, while keeping YOLO detections fixed.",
    }


def _track_identity_summary(tracks_by_id: dict[str, dict[str, Any]], observations_by_id: dict[str, list[dict[str, Any]]], local_id: str) -> dict[str, Any]:
    track = tracks_by_id[local_id]
    colour = ((track.get("vehicle_enrichment") or {}).get("vehicle_colour") or {}).get("label")
    rows = observations_by_id[local_id]
    centers = [_center({"bbox_xyxy": [float(row["x1"]), float(row["y1"]), float(row["x2"]), float(row["y2"])]}) for row in rows]
    return {
        "local_track_id": local_id,
        "native_tracker_id": track.get("native_tracker_id"),
        "final_class": track.get("final_class"),
        "colour": colour,
        "first_frame": track.get("first_frame"),
        "last_frame": track.get("last_frame"),
        "observation_count": track.get("observation_count"),
        "mean_center_x": float(np.mean([item[0] for item in centers])) if centers else None,
        "mean_center_y": float(np.mean([item[1] for item in centers])) if centers else None,
    }


def _replay_match_summary(observations_by_id: dict[str, list[dict[str, Any]]], replay: dict[int, ReplayFrame]) -> dict[str, Any]:
    total = 0
    matched = 0
    mismatches = []
    for local_id, rows in observations_by_id.items():
        native = int(local_id.split("_")[-1])
        for row in rows:
            total += 1
            frame = int(row["frame_number"])
            if native in replay.get(frame, ReplayFrame(frame, [], [], [], [])).output_ids:
                matched += 1
            elif len(mismatches) < 20:
                mismatches.append({"local_track_id": local_id, "frame_number": frame, "replay_output_ids": replay.get(frame).output_ids if replay.get(frame) else []})
    return {"target_observations": total, "matched_replay_outputs": matched, "match_ratio": matched / total if total else None, "sample_mismatches": mismatches}


def _classify_root(transition: dict[str, Any], association_rows: list[dict[str, Any]]) -> str:
    if transition["detector_dropout"]:
        return "YOLO DETECTION DROPOUT / ROI-PASSED DETECTION ABSENCE"
    if transition["low_confidence_frames"]:
        return "LOW-CONFIDENCE DETECTIONS ARE NOT BEING RECOVERED"
    if transition.get("box_jump"):
        return "YOLO BOUNDING-BOX INSTABILITY IS PRIMARY"
    rejected = [row for row in association_rows if row.get("first_pass_match_accepted") is False and row.get("best_fused_distance") is not None]
    if rejected:
        return "BYTETrack ASSOCIATION IS PRIMARY"
    return "MULTIPLE FACTORS ARE CONTRIBUTING"


def _primary_conclusion(rows: list[dict[str, Any]]) -> str:
    roots = [row["root_cause"] for row in rows]
    if roots and all(root == roots[0] for root in roots):
        return roots[0]
    if any("DROPOUT" in root for root in roots) and any("LOW-CONFIDENCE" in root for root in roots):
        return "MULTIPLE FACTORS ARE CONTRIBUTING: YOLO/ROI-passed truck detections drop out or fall into ByteTrack's low-confidence path, then ByteTrack emits a new native ID instead of reactivating the lost one."
    if any("LOW-CONFIDENCE" in root for root in roots):
        return "LOW-CONFIDENCE DETECTIONS ARE NOT BEING RECOVERED"
    return "MULTIPLE FACTORS ARE CONTRIBUTING"


def _box_list(box: Any) -> list[float]:
    return [float(value) for value in np.asarray(box, dtype=float).tolist()]


def _iou(left: Any, right: Any) -> float:
    lx1, ly1, lx2, ly2 = [float(value) for value in left]
    rx1, ry1, rx2, ry2 = [float(value) for value in right]
    ix1, iy1 = max(lx1, rx1), max(ly1, ry1)
    ix2, iy2 = min(lx2, rx2), min(ly2, ry2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_l = max(0.0, lx2 - lx1) * max(0.0, ly2 - ly1)
    area_r = max(0.0, rx2 - rx1) * max(0.0, ry2 - ry1)
    union = area_l + area_r - inter
    return float(inter / union) if union > 0.0 else 0.0


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
