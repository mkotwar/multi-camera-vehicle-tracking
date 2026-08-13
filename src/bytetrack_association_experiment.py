from __future__ import annotations

import csv
import json
import logging
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import supervision as sv
import yaml

from .detector_tracker import VehicleDetectorTracker
from .models import FramePacket
from .stationary_truck_diagnostic import DiagnosticByteTrack


TARGET_IDS = [6, 14, 78, 100, 133, 143]
TARGET_LOCAL_IDS = [f"CAM_001:TRACK_{item}" for item in TARGET_IDS]
THRESHOLDS = [0.60, 0.50, 0.40]
TRUCK_CLASSES = {"truck", "bus"}
FALLBACK_NEARBY_CLASSES = {"car"}


@dataclass(frozen=True)
class FrozenFrame:
    camera_id: str
    frame_number: int
    timestamp_seconds: float
    source_fps: float
    xyxy: list[list[float]]
    confidence: list[float]
    class_id: list[int]

    def to_json(self) -> dict[str, Any]:
        return {
            "camera_id": self.camera_id,
            "frame_number": self.frame_number,
            "timestamp_seconds": self.timestamp_seconds,
            "source_fps": self.source_fps,
            "xyxy": self.xyxy,
            "confidence": self.confidence,
            "class_id": self.class_id,
        }

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> "FrozenFrame":
        return cls(
            camera_id=str(payload["camera_id"]),
            frame_number=int(payload["frame_number"]),
            timestamp_seconds=float(payload["timestamp_seconds"]),
            source_fps=float(payload["source_fps"]),
            xyxy=[[float(value) for value in row] for row in payload.get("xyxy", [])],
            confidence=[float(value) for value in payload.get("confidence", [])],
            class_id=[int(value) for value in payload.get("class_id", [])],
        )

    def to_detections(self) -> sv.Detections:
        if not self.xyxy:
            empty = sv.Detections.empty()
            empty.tracker_id = np.array([], dtype=int)
            return empty
        return sv.Detections(
            xyxy=np.asarray(self.xyxy, dtype=np.float32),
            confidence=np.asarray(self.confidence, dtype=np.float32),
            class_id=np.asarray(self.class_id, dtype=np.int32),
        )


def run_association_threshold_experiment(run_dir: str | Path) -> dict[str, Any]:
    run_path = Path(run_dir)
    output_dir = run_path / "bytetrack_association_test"
    output_dir.mkdir(parents=True, exist_ok=True)

    config = yaml.safe_load((run_path / "run_config.yaml").read_text(encoding="utf-8"))
    original_tracks = json.loads((run_path / "tracks.json").read_text(encoding="utf-8"))
    original_observations = _read_csv(run_path / "observations.csv")
    target_bbox = _target_median_bbox(original_observations)

    frozen_path = output_dir / "frozen_detections.jsonl"
    if not frozen_path.exists():
        capture_frozen_detections(run_path, frozen_path)
    frozen_frames = read_frozen_detections(frozen_path)

    semantics = inspect_threshold_semantics()
    validation: dict[str, Any] = {}
    comparison: list[dict[str, Any]] = []
    variants: dict[str, Any] = {}
    baseline_valid = False
    for threshold in THRESHOLDS:
        key = _threshold_key(threshold)
        variant_dir = output_dir / key
        variant_dir.mkdir(parents=True, exist_ok=True)
        result = replay_threshold_variant(
            config=config,
            frames=frozen_frames,
            threshold=threshold,
            output_dir=variant_dir,
            target_bbox=target_bbox,
        )
        variants[key] = result
        comparison.append(_comparison_row(key, result))
        if threshold == 0.60:
            validation = validate_baseline_replay(original_tracks, original_observations, result)
            baseline_valid = bool(validation["valid"])
            (output_dir / "replay_validation.json").write_text(json.dumps(validation, indent=2), encoding="utf-8")
            if not baseline_valid:
                break

    _write_csv(output_dir / "comparison.csv", comparison)
    summary = {
        "frozen_detection_source": {
            "path": str(frozen_path),
            "note": "Captured once from the same video/config immediately before ByteTrack as sv.Detections-equivalent xyxy/confidence/class_id arrays. Threshold variants replay this frozen stream without rerunning YOLO.",
        },
        "threshold_semantics": semantics,
        "baseline_validation": validation,
        "variants": variants,
        "comparison": comparison,
        "decision": _decision(comparison, baseline_valid),
    }
    (output_dir / "comparison.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def capture_frozen_detections(run_dir: Path, output_path: Path) -> None:
    config = yaml.safe_load((run_dir / "run_config.yaml").read_text(encoding="utf-8"))
    camera = next(item for item in config["input"]["cameras"] if item.get("enabled", True))
    video_path = Path(camera["source"])
    logger = logging.getLogger("bytetrack_association_capture")
    logger.setLevel(logging.INFO)
    detector = VehicleDetectorTracker(config, logger)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    max_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        frame_number = 0
        while frame_number < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            packet = FramePacket(
                camera_id=str(camera["camera_id"]),
                frame_number=frame_number,
                timestamp_seconds=frame_number / fps if fps > 0 else 0.0,
                source_fps=fps,
                frame=frame,
                source_frame_width=width,
                source_frame_height=height,
                worker_id=0,
                captured_at=datetime.now(timezone.utc).isoformat(),
                source_type=str(camera.get("source_type", "video")),
            )
            raw_result = detector._infer_single_raw_result(packet)  # noqa: SLF001
            raw_detections, accepted, _diagnostics, _timings = detector._build_detection_payload(  # noqa: SLF001
                packet,
                raw_result,
                inference_wall_time_ms=0.0,
            )
            _ = raw_detections
            roi_eligible = detector.filter_detections_by_tracking_roi(packet, accepted)
            sv_detections = detector._to_ocr_mukul_supervision_detections(raw_result, roi_eligible)  # noqa: SLF001
            frozen = FrozenFrame(
                camera_id=packet.camera_id,
                frame_number=frame_number,
                timestamp_seconds=packet.timestamp_seconds,
                source_fps=fps,
                xyxy=np.asarray(sv_detections.xyxy, dtype=float).tolist(),
                confidence=np.asarray(sv_detections.confidence, dtype=float).tolist(),
                class_id=np.asarray(sv_detections.class_id, dtype=int).tolist(),
            )
            handle.write(json.dumps(frozen.to_json()) + "\n")
            frame_number += 1
    cap.release()


def read_frozen_detections(path: Path) -> list[FrozenFrame]:
    frames = [FrozenFrame.from_json(json.loads(line)) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert_frame_order(frames)
    return frames


def assert_frame_order(frames: list[FrozenFrame]) -> None:
    previous = -1
    for frame in frames:
        if frame.frame_number <= previous:
            raise ValueError("Frozen detections must be strictly increasing by frame_number.")
        previous = frame.frame_number


def replay_threshold_variant(
    *,
    config: dict[str, Any],
    frames: list[FrozenFrame],
    threshold: float,
    output_dir: Path,
    target_bbox: list[float],
) -> dict[str, Any]:
    tracking = dict(config.get("tracking", {}) or {})
    tracker = DiagnosticByteTrack(
        watch_ids=set(range(1, 400)),
        lost_track_buffer=int(tracking.get("lost_track_buffer", 40)),
        track_activation_threshold=float(tracking.get("track_activation_threshold", 0.3)),
        minimum_matching_threshold=float(threshold),
        minimum_consecutive_frames=int(tracking.get("minimum_consecutive_frames", 3)),
    )
    timeline: list[dict[str, Any]] = []
    observations: list[dict[str, Any]] = []
    for frozen in frames:
        tracked = tracker.update_with_detections(frozen.to_detections())
        frame_rows = _tracked_rows(frozen, tracked)
        observations.extend(frame_rows)
        target_rows = _target_rows(frame_rows, target_bbox)
        for row in target_rows:
            timeline.append(row)
    tracks = _build_tracks(observations)
    truck_tracks_all = _build_tracks(timeline)
    truck_tracks = [row for row in truck_tracks_all if int(row["observation_count"]) >= int(config.get("lifecycle", {}).get("minimum_observations", 3))]
    truck_blips = [row for row in truck_tracks_all if int(row["observation_count"]) < int(config.get("lifecycle", {}).get("minimum_observations", 3))]
    _write_csv(output_dir / "tracks.csv", tracks)
    _write_csv(output_dir / "truck_timeline.csv", timeline)
    _write_csv(output_dir / "association_diagnostics.csv", tracker.diagnostics)
    _make_visual_evidence(output_dir / "visual_evidence", config, timeline)
    return {
        "threshold": threshold,
        "total_scene_tracks": len(tracks),
        "completed_tracks": len(tracks),
        "short_tracks_lt_3": sum(1 for row in tracks if int(row["observation_count"]) < 3),
        "short_tracks_lt_10": sum(1 for row in tracks if int(row["observation_count"]) < 10),
        "mean_observations_per_track": float(np.mean([int(row["observation_count"]) for row in tracks])) if tracks else 0.0,
        "median_observations_per_track": float(np.median([int(row["observation_count"]) for row in tracks])) if tracks else 0.0,
        "reactivations": _count_reactivations(timeline),
        "stationary_truck_fragments": len(truck_tracks),
        "stationary_truck_transient_blips": len(truck_blips),
        "native_ids_used_by_truck": [int(row["native_tracker_id"]) for row in truck_tracks],
        "native_ids_used_by_truck_blips": [int(row["native_tracker_id"]) for row in truck_blips],
        "truck_tracks": truck_tracks,
        "truck_tracks_all_candidates": truck_tracks_all,
        "truck_blips": truck_blips,
        "transition_recovery": _transition_recovery(truck_tracks),
        "suspected_id_switches": _suspected_switches(timeline),
    }


def validate_baseline_replay(original_tracks: list[dict[str, Any]], original_observations: list[dict[str, Any]], replay: dict[str, Any]) -> dict[str, Any]:
    original_target_tracks = [track for track in original_tracks if str(track.get("local_track_id")) in TARGET_LOCAL_IDS]
    original_ids = [int(track["native_tracker_id"]) for track in original_target_tracks]
    original_spans = [(int(track["first_frame"]), int(track["last_frame"])) for track in original_target_tracks]
    replay_spans = [(int(track["first_frame"]), int(track["last_frame"])) for track in replay["truck_tracks"]]
    span_errors = []
    for original, candidate in zip(original_spans, replay_spans):
        span_errors.append(abs(original[0] - candidate[0]) + abs(original[1] - candidate[1]))
    valid = (
        replay["stationary_truck_fragments"] == len(original_target_tracks)
        and len(replay_spans) == len(original_spans)
        and (max(span_errors) if span_errors else 999) <= 12
    )
    return {
        "valid": valid,
        "reason": "Baseline replay reproduces stationary-truck fragment count and spans within tolerance." if valid else "Baseline replay did not reproduce stationary-truck fragmentation closely enough; threshold A/B is not authoritative.",
        "original_native_ids": original_ids,
        "original_spans": original_spans,
        "replay_native_ids": replay["native_ids_used_by_truck"],
        "replay_spans": replay_spans,
        "replay_transient_blips": replay.get("truck_blips", []),
        "span_error_sum_by_fragment": span_errors,
        "original_total_tracks": len(original_tracks),
        "replay_total_tracks": replay["total_scene_tracks"],
    }


def inspect_threshold_semantics() -> dict[str, Any]:
    import inspect

    source = inspect.getsource(sv.ByteTrack.update_with_tensors)
    return {
        "minimum_matching_threshold_used_as": "linear_assignment distance threshold after IoU distance is fused with detection score",
        "code_comparison": "matching.linear_assignment(dists, thresh=self.minimum_matching_threshold)",
        "interpretation": "Larger values are more permissive for the fused distance cost; lower values are stricter. Testing 0.50/0.40 is therefore a stricter-threshold test in this installed Supervision implementation, not a more permissive one.",
        "source_excerpt_present": "minimum_matching_threshold" in source and "linear_assignment" in source,
    }


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _target_median_bbox(observations: list[dict[str, Any]]) -> list[float]:
    boxes = [
        [float(row["x1"]), float(row["y1"]), float(row["x2"]), float(row["y2"])]
        for row in observations
        if row["local_track_id"] in TARGET_LOCAL_IDS
    ]
    return [float(value) for value in np.median(np.asarray(boxes, dtype=float), axis=0)]


def _tracked_rows(frozen: FrozenFrame, tracked: sv.Detections) -> list[dict[str, Any]]:
    ids = list(tracked.tracker_id) if getattr(tracked, "tracker_id", None) is not None else []
    rows = []
    for index, tracker_id in enumerate(ids):
        if int(tracker_id) < 0:
            continue
        bbox = [float(value) for value in tracked.xyxy[index]]
        confidence = float(tracked.confidence[index]) if getattr(tracked, "confidence", None) is not None else 0.0
        class_id = int(tracked.class_id[index]) if getattr(tracked, "class_id", None) is not None else -1
        rows.append(
            {
                "camera_id": frozen.camera_id,
                "frame_number": frozen.frame_number,
                "timestamp_seconds": frozen.timestamp_seconds,
                "native_tracker_id": int(tracker_id),
                "class_id": class_id,
                "class_name": _class_name(class_id),
                "confidence": confidence,
                "x1": bbox[0],
                "y1": bbox[1],
                "x2": bbox[2],
                "y2": bbox[3],
                "bbox": json.dumps(bbox),
            }
        )
    return rows


def _target_rows(rows: list[dict[str, Any]], target_bbox: list[float]) -> list[dict[str, Any]]:
    result = []
    fallback = []
    for row in rows:
        bbox = [float(row["x1"]), float(row["y1"]), float(row["x2"]), float(row["y2"])]
        overlap = _iou(bbox, target_bbox)
        cx, cy = (bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0
        near = target_bbox[0] - 90 <= cx <= target_bbox[2] + 90 and target_bbox[1] - 90 <= cy <= target_bbox[3] + 90
        class_name = str(row["class_name"])
        if class_name in TRUCK_CLASSES and (overlap >= 0.08 or near):
            enriched = dict(row)
            enriched["target_iou"] = overlap
            result.append(enriched)
        elif class_name in FALLBACK_NEARBY_CLASSES and overlap >= 0.35:
            enriched = dict(row)
            enriched["target_iou"] = overlap
            fallback.append(enriched)
    if not result:
        result = fallback
    if not result:
        return []
    return [max(result, key=lambda item: (float(item["target_iou"]), float(item["confidence"])))]


def _build_tracks(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(int(row["native_tracker_id"]), []).append(row)
    tracks = []
    for tracker_id, items in sorted(grouped.items()):
        items.sort(key=lambda row: int(row["frame_number"]))
        frames = [int(row["frame_number"]) for row in items]
        tracks.append(
            {
                "native_tracker_id": tracker_id,
                "first_frame": frames[0],
                "last_frame": frames[-1],
                "observation_count": len(items),
                "mean_confidence": float(np.mean([float(row["confidence"]) for row in items])),
                "class_counts": json.dumps(_counts(str(row["class_name"]) for row in items)),
            }
        )
    return tracks


def _transition_recovery(truck_tracks: list[dict[str, Any]]) -> dict[str, str]:
    statuses = {}
    for index, (old_id, new_id) in enumerate(zip(TARGET_IDS[:-1], TARGET_IDS[1:])):
        transition = f"{old_id}_to_{new_id}"
        if index + 1 >= len(truck_tracks):
            statuses[transition] = "RECOVERED"
            continue
        statuses[transition] = "RECOVERED" if truck_tracks[index]["native_tracker_id"] == truck_tracks[index + 1]["native_tracker_id"] else "STILL_SPLIT"
    return statuses


def _count_reactivations(timeline: list[dict[str, Any]]) -> int:
    by_id: dict[int, list[int]] = {}
    for row in timeline:
        by_id.setdefault(int(row["native_tracker_id"]), []).append(int(row["frame_number"]))
    return sum(1 for frames in by_id.values() if any((b - a) > 1 for a, b in zip(sorted(frames), sorted(frames)[1:])))


def _suspected_switches(timeline: list[dict[str, Any]]) -> int:
    switches = 0
    by_id: dict[int, list[dict[str, Any]]] = {}
    for row in timeline:
        by_id.setdefault(int(row["native_tracker_id"]), []).append(row)
    for rows in by_id.values():
        rows.sort(key=lambda row: int(row["frame_number"]))
        for left, right in zip(rows, rows[1:]):
            if int(right["frame_number"]) - int(left["frame_number"]) <= 1:
                continue
            left_box = [float(left["x1"]), float(left["y1"]), float(left["x2"]), float(left["y2"])]
            right_box = [float(right["x1"]), float(right["y1"]), float(right["x2"]), float(right["y2"])]
            if _iou(left_box, right_box) < 0.12:
                switches += 1
    return switches


def _make_visual_evidence(output_dir: Path, config: dict[str, Any], timeline: list[dict[str, Any]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if not timeline:
        return
    video_path = config["input"]["cameras"][0]["source"]
    frames = sorted({int(row["frame_number"]) for row in timeline})
    if len(frames) > 18:
        frames = sorted(set(np.linspace(min(frames), max(frames), 18, dtype=int).tolist()))
    by_frame = {int(row["frame_number"]): row for row in timeline}
    tiles = []
    for frame_number in frames:
        frame = _read_frame(video_path, frame_number)
        if frame is None:
            continue
        row = by_frame.get(frame_number)
        if row:
            bbox = [float(row["x1"]), float(row["y1"]), float(row["x2"]), float(row["y2"])]
            _draw_box(frame, bbox, f"ID {row['native_tracker_id']} conf {float(row['confidence']):.2f}")
        cv2.putText(frame, f"f{frame_number}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
        tiles.append(_fit(frame, 360, 210))
    if tiles:
        cv2.imwrite(str(output_dir / "truck_timeline_contact_sheet.jpg"), _grid(tiles, 3))


def _comparison_row(key: str, result: dict[str, Any]) -> dict[str, Any]:
    recovered = sum(1 for value in result["transition_recovery"].values() if value == "RECOVERED")
    return {
        "threshold": key,
        "stationary_truck_fragments": result["stationary_truck_fragments"],
        "stationary_truck_transient_blips": result["stationary_truck_transient_blips"],
        "native_ids_used_by_truck": " ".join(str(item) for item in result["native_ids_used_by_truck"]),
        "recovered_original_transitions": recovered,
        "total_scene_tracks": result["total_scene_tracks"],
        "short_tracks_lt_10": result["short_tracks_lt_10"],
        "reactivations": result["reactivations"],
        "suspected_id_switches": result["suspected_id_switches"],
    }


def _decision(comparison: list[dict[str, Any]], baseline_valid: bool) -> str:
    if not baseline_valid:
        return "STOP: baseline replay is not sufficiently faithful for threshold comparison."
    by_key = {row["threshold"]: row for row in comparison}
    baseline = by_key.get("threshold_060")
    if not baseline:
        return "STOP: missing baseline."
    for key in ["threshold_050", "threshold_040"]:
        row = by_key.get(key)
        if row and int(row["stationary_truck_fragments"]) < int(baseline["stationary_truck_fragments"]) and int(row["suspected_id_switches"]) == 0:
            return f"{key} is promising on this replay; validate on more videos."
    return "MATCHING THRESHOLD IS NOT SUFFICIENT under the tested 0.60/0.50/0.40 variants."


def _threshold_key(value: float) -> str:
    return f"threshold_{int(round(value * 100)):03d}"


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _class_name(class_id: int) -> str:
    return {1: "3wheeler", 2: "car", 3: "motorcycle", 4: "truck", 5: "bus"}.get(int(class_id), str(class_id))


def _counts(items: Iterable[str]) -> dict[str, int]:
    result: dict[str, int] = {}
    for item in items:
        result[item] = result.get(item, 0) + 1
    return result


def _iou(left: list[float], right: list[float]) -> float:
    lx1, ly1, lx2, ly2 = left
    rx1, ry1, rx2, ry2 = right
    ix1, iy1 = max(lx1, rx1), max(ly1, ry1)
    ix2, iy2 = min(lx2, rx2), min(ly2, ry2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_l = max(0.0, lx2 - lx1) * max(0.0, ly2 - ly1)
    area_r = max(0.0, rx2 - rx1) * max(0.0, ry2 - ry1)
    union = area_l + area_r - inter
    return float(inter / union) if union > 0 else 0.0


def _read_frame(video_path: str, frame_number: int) -> np.ndarray | None:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ok, frame = cap.read()
        return frame if ok else None
    finally:
        cap.release()


def _draw_box(frame: np.ndarray, bbox: list[float], label: str) -> None:
    x1, y1, x2, y2 = [int(round(value)) for value in bbox]
    cv2.rectangle(frame, (x1, y1), (x2, y2), (80, 230, 80), 2)
    cv2.putText(frame, label, (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80, 230, 80), 2)


def _fit(image: np.ndarray, width: int, height: int) -> np.ndarray:
    tile = np.full((height, width, 3), 245, dtype=np.uint8)
    h, w = image.shape[:2]
    scale = min(width / max(w, 1), height / max(h, 1))
    resized = cv2.resize(image, (max(1, int(w * scale)), max(1, int(h * scale))))
    y0 = (height - resized.shape[0]) // 2
    x0 = (width - resized.shape[1]) // 2
    tile[y0 : y0 + resized.shape[0], x0 : x0 + resized.shape[1]] = resized
    return tile


def _grid(tiles: list[np.ndarray], columns: int) -> np.ndarray:
    h, w = tiles[0].shape[:2]
    rows = int(math.ceil(len(tiles) / columns))
    sheet = np.full((rows * h, columns * w, 3), 245, dtype=np.uint8)
    for index, tile in enumerate(tiles):
        y = (index // columns) * h
        x = (index % columns) * w
        sheet[y : y + h, x : x + w] = tile
    return sheet
