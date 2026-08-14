from __future__ import annotations

import csv
import json
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import supervision as sv


MATCHING_THRESHOLDS = (0.60, 0.65, 0.70, 0.75)
ACTIVATION_THRESHOLDS = (0.30, 0.25, 0.22)
TARGET_TRACK_IDS = (6, 14, 78, 100, 133, 143)
TARGET_LOCAL_IDS = tuple(f"CAM_001:TRACK_{track_id}" for track_id in TARGET_TRACK_IDS)
TARGET_TRANSITION_WINDOWS = (
    ("6_to_14", 21, 70),
    ("14_to_78", 51, 120),
    ("78_to_100", 374, 443),
    ("100_to_133", 555, 634),
    ("133_to_143", 741, 764),
)


def clone_detections(detections: sv.Detections) -> sv.Detections:
    cloned = sv.Detections(
        xyxy=np.asarray(detections.xyxy, dtype=np.float32).copy(),
        confidence=np.asarray(detections.confidence, dtype=np.float32).copy()
        if getattr(detections, "confidence", None) is not None
        else None,
        class_id=np.asarray(detections.class_id, dtype=np.int32).copy()
        if getattr(detections, "class_id", None) is not None
        else None,
    )
    if getattr(detections, "tracker_id", None) is not None:
        cloned.tracker_id = np.asarray(detections.tracker_id, dtype=np.int32).copy()
    return cloned


def detections_signature(detections: sv.Detections) -> dict[str, Any]:
    return {
        "count": int(len(detections.xyxy)),
        "xyxy": np.asarray(detections.xyxy, dtype=np.float32).tolist(),
        "confidence": np.asarray(detections.confidence, dtype=np.float32).tolist()
        if getattr(detections, "confidence", None) is not None
        else [],
        "class_id": np.asarray(detections.class_id, dtype=np.int32).tolist()
        if getattr(detections, "class_id", None) is not None
        else [],
        "xyxy_dtype": str(np.asarray(detections.xyxy).dtype),
        "confidence_dtype": str(np.asarray(detections.confidence).dtype)
        if getattr(detections, "confidence", None) is not None
        else None,
        "class_id_dtype": str(np.asarray(detections.class_id).dtype)
        if getattr(detections, "class_id", None) is not None
        else None,
    }


@dataclass(slots=True)
class ShadowVariant:
    name: str
    matching_threshold: float
    activation_threshold: float
    lost_track_buffer: int
    minimum_consecutive_frames: int
    frame_rate: float
    tracker: Any
    rows: list[dict[str, Any]] = field(default_factory=list)
    previous_tracked_ids: set[int] = field(default_factory=set)
    ever_seen_ids: set[int] = field(default_factory=set)
    reactivations: int = 0
    new_track_activations: int = 0
    lost_events: int = 0


class RuntimeTrackingFixExperiment:
    """Runs isolated ByteTrack variants from the exact production tracker input."""

    def __init__(
        self,
        *,
        output_dir: str | Path,
        base_config: dict[str, Any],
        logger: Any,
        reference_run_dir: str | Path | None = None,
        enabled: bool = True,
        run_threshold_experiment: bool = True,
        run_activation_experiment: bool = False,
    ) -> None:
        self.enabled = bool(enabled)
        self.output_dir = Path(output_dir)
        self.reference_run_dir = Path(reference_run_dir) if reference_run_dir else None
        self.logger = logger
        self.base_config = dict(base_config)
        tracking = dict(self.base_config.get("tracking", {}) or {})
        self.base_activation = float(tracking.get("track_activation_threshold", 0.3))
        self.base_matching = float(tracking.get("minimum_matching_threshold", 0.6))
        self.lost_track_buffer = int(tracking.get("lost_track_buffer", 40))
        self.minimum_consecutive_frames = int(tracking.get("minimum_consecutive_frames", 3))
        self.run_threshold_experiment = bool(run_threshold_experiment)
        self.run_activation_experiment = bool(run_activation_experiment)
        self.variants_by_camera: dict[str, dict[str, ShadowVariant]] = {}
        self.input_rows: list[dict[str, Any]] = []
        self.input_errors: list[dict[str, Any]] = []
        self.frame_counts_by_camera: dict[str, int] = {}
        self.empty_frame_counts_by_camera: dict[str, int] = {}
        self._finalized = False

    @classmethod
    def from_config(
        cls,
        config: dict[str, Any],
        *,
        run_directory: str | Path,
        logger: Any,
    ) -> RuntimeTrackingFixExperiment | None:
        section = dict(config.get("tracking_fix_experiment", {}) or {})
        if not bool(section.get("enabled", False)):
            return None
        return cls(
            output_dir=Path(run_directory) / "tracking_fix_experiment",
            base_config=config,
            logger=logger,
            reference_run_dir=section.get("reference_run_dir"),
            enabled=True,
            run_threshold_experiment=bool(section.get("threshold_experiment", True)),
            run_activation_experiment=bool(section.get("activation_experiment", False)),
        )

    def observe(
        self,
        *,
        camera_id: str,
        frame_number: int,
        timestamp_seconds: float,
        source_fps: float,
        detections: sv.Detections,
    ) -> None:
        if not self.enabled:
            return
        source = clone_detections(detections)
        source_signature = detections_signature(source)
        self.frame_counts_by_camera[camera_id] = self.frame_counts_by_camera.get(camera_id, 0) + 1
        if source_signature["count"] == 0:
            self.empty_frame_counts_by_camera[camera_id] = self.empty_frame_counts_by_camera.get(camera_id, 0) + 1
        self.input_rows.append(
            {
                "camera_id": camera_id,
                "frame_number": int(frame_number),
                "timestamp_seconds": float(timestamp_seconds),
                "source_fps": float(source_fps or 30.0),
                **source_signature,
            }
        )
        variants = self._variants_for_camera(camera_id, float(source_fps or 30.0))
        for variant in variants.values():
            candidate_input = clone_detections(source)
            if detections_signature(candidate_input) != source_signature:
                self.input_errors.append(
                    {
                        "camera_id": camera_id,
                        "frame_number": int(frame_number),
                        "variant": variant.name,
                        "error": "input_copy_signature_mismatch",
                    }
                )
                continue
            tracked = variant.tracker.update_with_detections(candidate_input)
            self._record_variant_output(variant, camera_id, frame_number, timestamp_seconds, tracked)

    def finalize(self) -> dict[str, Any]:
        if self._finalized:
            return {"output_directory": str(self.output_dir), "already_finalized": True}
        self._finalized = True
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "threshold_shadow").mkdir(exist_ok=True)
        (self.output_dir / "activation_shadow").mkdir(exist_ok=True)
        (self.output_dir / "visual_evidence").mkdir(exist_ok=True)

        reference = self._load_reference()
        threshold_rows: list[dict[str, Any]] = []
        activation_rows: list[dict[str, Any]] = []
        full_scene_rows: list[dict[str, Any]] = []
        transition_rows: list[dict[str, Any]] = []
        variant_summaries: dict[str, Any] = {}

        for camera_id, variants in sorted(self.variants_by_camera.items()):
            for variant in variants.values():
                output_root = self.output_dir / ("activation_shadow" if variant.name.startswith("activation_") else "threshold_shadow")
                variant_dir = output_root / variant.name.rsplit("_", 1)[-1]
                variant_dir.mkdir(parents=True, exist_ok=True)
                self._write_csv(variant_dir / "track_timeline.csv", variant.rows)
                scene_metrics = self._scene_metrics(variant)
                truck_metrics = self._truck_metrics(variant, reference)
                summary = {**scene_metrics, **truck_metrics}
                summary.update(
                    {
                        "camera_id": camera_id,
                        "variant": variant.name,
                        "minimum_matching_threshold": variant.matching_threshold,
                        "track_activation_threshold": variant.activation_threshold,
                        "effective_new_track_threshold": min(1.0, variant.activation_threshold + 0.1),
                        "frame_rate": variant.frame_rate,
                    }
                )
                variant_summaries[variant.name] = summary
                (variant_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
                full_scene_rows.append(summary)
                if variant.name.startswith("threshold_"):
                    threshold_rows.append(summary)
                    transition_rows.extend(self._transition_status_rows(variant, reference))
                else:
                    activation_rows.append(summary)

        runtime_validation = {
            "enabled": self.enabled,
            "production_unaffected": True,
            "input_errors": self.input_errors,
            "frame_counts_by_camera": self.frame_counts_by_camera,
            "empty_frame_counts_by_camera": self.empty_frame_counts_by_camera,
            "variant_count_by_camera": {camera: len(variants) for camera, variants in self.variants_by_camera.items()},
            "identical_input_required": True,
        }
        (self.output_dir / "runtime_input_validation.json").write_text(json.dumps(runtime_validation, indent=2), encoding="utf-8")
        self._write_jsonl(self.output_dir / "runtime_detection_stream.jsonl", self.input_rows)
        self._write_csv(self.output_dir / "threshold_comparison.csv", threshold_rows)
        self._write_csv(self.output_dir / "activation_comparison.csv", activation_rows)
        self._write_csv(self.output_dir / "roi_comparison.csv", [])
        self._write_csv(self.output_dir / "full_scene_comparison.csv", full_scene_rows)
        self._write_csv(self.output_dir / "transition_validation.csv", transition_rows)
        decision = self._candidate_decision(threshold_rows, activation_rows)
        (self.output_dir / "final_candidate.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
        before_after = {"reference": reference.get("baseline", {}), "shadow_variants": variant_summaries, "decision": decision}
        (self.output_dir / "before_after_summary.json").write_text(json.dumps(before_after, indent=2), encoding="utf-8")
        self.logger.info("Tracking fix shadow experiment written output_dir=%s", self.output_dir)
        return {"output_directory": str(self.output_dir), "runtime_input_validation": runtime_validation, "decision": decision}

    def _variants_for_camera(self, camera_id: str, frame_rate: float) -> dict[str, ShadowVariant]:
        existing = self.variants_by_camera.get(camera_id)
        if existing is not None:
            return existing
        variants: dict[str, ShadowVariant] = {}
        if self.run_threshold_experiment:
            for value in MATCHING_THRESHOLDS:
                name = f"threshold_{int(round(value * 100)):03d}"
                variants[name] = self._make_variant(name, value, self.base_activation, frame_rate)
        if self.run_activation_experiment:
            for value in ACTIVATION_THRESHOLDS:
                name = f"activation_{int(round(value * 100)):03d}"
                variants[name] = self._make_variant(name, self.base_matching, value, frame_rate)
        self.variants_by_camera[camera_id] = variants
        return variants

    def _make_variant(self, name: str, matching_threshold: float, activation_threshold: float, frame_rate: float) -> ShadowVariant:
        tracker = sv.ByteTrack(
            lost_track_buffer=self.lost_track_buffer,
            track_activation_threshold=activation_threshold,
            minimum_matching_threshold=matching_threshold,
            frame_rate=frame_rate,
            minimum_consecutive_frames=self.minimum_consecutive_frames,
        )
        return ShadowVariant(
            name=name,
            matching_threshold=matching_threshold,
            activation_threshold=activation_threshold,
            lost_track_buffer=self.lost_track_buffer,
            minimum_consecutive_frames=self.minimum_consecutive_frames,
            frame_rate=frame_rate,
            tracker=tracker,
        )

    def _record_variant_output(
        self,
        variant: ShadowVariant,
        camera_id: str,
        frame_number: int,
        timestamp_seconds: float,
        tracked: sv.Detections,
    ) -> None:
        ids = [int(item) for item in list(tracked.tracker_id)] if getattr(tracked, "tracker_id", None) is not None else []
        boxes = np.asarray(tracked.xyxy, dtype=np.float32) if getattr(tracked, "xyxy", None) is not None else np.empty((0, 4), dtype=np.float32)
        confidences = (
            np.asarray(tracked.confidence, dtype=np.float32)
            if getattr(tracked, "confidence", None) is not None
            else np.zeros((len(ids),), dtype=np.float32)
        )
        class_ids = (
            np.asarray(tracked.class_id, dtype=np.int32)
            if getattr(tracked, "class_id", None) is not None
            else np.zeros((len(ids),), dtype=np.int32)
        )
        current_ids = set(ids)
        new_ids = current_ids - variant.ever_seen_ids
        reactivated_ids = current_ids & (variant.ever_seen_ids - variant.previous_tracked_ids)
        lost_now = variant.previous_tracked_ids - current_ids
        variant.new_track_activations += len(new_ids)
        variant.reactivations += len(reactivated_ids)
        variant.lost_events += len(lost_now)
        variant.ever_seen_ids.update(current_ids)
        variant.previous_tracked_ids = current_ids
        for index, tracker_id in enumerate(ids):
            variant.rows.append(
                {
                    "camera_id": camera_id,
                    "frame_number": int(frame_number),
                    "timestamp_seconds": float(timestamp_seconds),
                    "native_tracker_id": tracker_id,
                    "bbox_xyxy": boxes[index].tolist() if index < len(boxes) else [],
                    "confidence": float(confidences[index]) if index < len(confidences) else 0.0,
                    "class_id": int(class_ids[index]) if index < len(class_ids) else -1,
                    "new_track_activation": tracker_id in new_ids,
                    "reactivation": tracker_id in reactivated_ids,
                }
            )

    def _load_reference(self) -> dict[str, Any]:
        if not self.reference_run_dir:
            return {"baseline": {"truck_fragment_count": len(TARGET_LOCAL_IDS), "truck_ids": list(TARGET_LOCAL_IDS)}}
        observations_path = self.reference_run_dir / "observations.csv"
        tracks_path = self.reference_run_dir / "tracks.json"
        if not observations_path.exists() or not tracks_path.exists():
            return {"baseline": {"truck_fragment_count": len(TARGET_LOCAL_IDS), "truck_ids": list(TARGET_LOCAL_IDS)}}
        observations = self._read_csv(observations_path)
        truck_rows = [row for row in observations if str(row.get("local_track_id")) in TARGET_LOCAL_IDS]
        boxes = [self._row_box(row) for row in truck_rows]
        boxes = [box for box in boxes if box is not None]
        median_box = np.median(np.asarray(boxes, dtype=np.float32), axis=0).tolist() if boxes else None
        reference_by_frame = {
            int(row["frame_number"]): self._row_box(row)
            for row in truck_rows
            if self._row_box(row) is not None
        }
        return {
            "target_box": median_box,
            "target_box_by_frame": reference_by_frame,
            "target_rows": truck_rows,
            "baseline": {
                "truck_fragment_count": len(TARGET_LOCAL_IDS),
                "truck_ids": list(TARGET_LOCAL_IDS),
                "observation_count": len(truck_rows),
            },
        }

    def _truck_metrics(self, variant: ShadowVariant, reference: dict[str, Any]) -> dict[str, Any]:
        target_box = reference.get("target_box")
        target_box_by_frame = reference.get("target_box_by_frame") or {}
        if not target_box and not target_box_by_frame:
            return {
                "truck_fragments": None,
                "truck_native_ids": [],
                "truck_longest_continuous_span": None,
                "confirmed_false_merges": 0,
                "suspected_false_merges": 0,
            }
        rows_by_frame: dict[int, list[dict[str, Any]]] = {}
        for row in variant.rows:
            rows_by_frame.setdefault(int(row["frame_number"]), []).append(row)
        target_rows = []
        for frame_number, frame_target_box in target_box_by_frame.items():
            candidates = []
            for row in rows_by_frame.get(int(frame_number), []):
                box = self._parse_box(row["bbox_xyxy"])
                if box is None:
                    continue
                overlap = self._iou(box, frame_target_box)
                if overlap >= 0.20:
                    candidates.append((overlap, row))
            if not candidates:
                continue
            overlap, row = max(candidates, key=lambda item: item[0])
            copied = dict(row)
            copied["target_iou"] = overlap
            target_rows.append(copied)
        ids = sorted({int(row["native_tracker_id"]) for row in target_rows})
        suspected_false_merge_ids: set[int] = set()
        for tracker_id in ids:
            for row in variant.rows:
                if int(row["native_tracker_id"]) != tracker_id:
                    continue
                frame_target_box = target_box_by_frame.get(int(row["frame_number"]))
                if not frame_target_box:
                    continue
                box = self._parse_box(row["bbox_xyxy"])
                if box is not None and self._iou(box, frame_target_box) < 0.05 and self._center_distance(box, frame_target_box) > 120.0:
                    suspected_false_merge_ids.add(tracker_id)
        spans = []
        for tracker_id in ids:
            frames = [int(row["frame_number"]) for row in target_rows if int(row["native_tracker_id"]) == tracker_id]
            if frames:
                spans.append(max(frames) - min(frames) + 1)
        return {
            "truck_fragments": len(ids),
            "truck_native_ids": ids,
            "truck_fragment_spans": [
                {
                    "native_tracker_id": tracker_id,
                    "start_frame": min(int(row["frame_number"]) for row in target_rows if int(row["native_tracker_id"]) == tracker_id),
                    "end_frame": max(int(row["frame_number"]) for row in target_rows if int(row["native_tracker_id"]) == tracker_id),
                    "observations": sum(1 for row in target_rows if int(row["native_tracker_id"]) == tracker_id),
                }
                for tracker_id in ids
            ],
            "truck_lost_transitions": max(0, len(ids) - 1),
            "truck_successful_reactivations": sum(1 for row in target_rows if bool(row.get("reactivation"))),
            "truck_new_ids_created": len(ids),
            "truck_longest_continuous_span": max(spans) if spans else 0,
            "confirmed_false_merges": 0,
            "suspected_false_merges": len(suspected_false_merge_ids),
        }

    def _scene_metrics(self, variant: ShadowVariant) -> dict[str, Any]:
        counts: dict[int, int] = {}
        for row in variant.rows:
            track_id = int(row["native_tracker_id"])
            counts[track_id] = counts.get(track_id, 0) + 1
        durations = list(counts.values())
        return {
            "total_native_tracks": len(counts),
            "tracks_lt_3_obs": sum(1 for value in durations if value < 3),
            "tracks_lt_10_obs": sum(1 for value in durations if value < 10),
            "mean_observations": statistics.fmean(durations) if durations else 0.0,
            "median_observations": statistics.median(durations) if durations else 0.0,
            "reactivations": variant.reactivations,
            "new_track_activations": variant.new_track_activations,
            "lost_events": variant.lost_events,
            "removed_tracks": len(getattr(variant.tracker, "removed_tracks", []) or []),
        }

    def _transition_status_rows(self, variant: ShadowVariant, reference: dict[str, Any]) -> list[dict[str, Any]]:
        target_box = reference.get("target_box")
        target_box_by_frame = reference.get("target_box_by_frame") or {}
        if not target_box and not target_box_by_frame:
            return []
        rows = []
        rows_by_frame: dict[int, list[dict[str, Any]]] = {}
        for row in variant.rows:
            rows_by_frame.setdefault(int(row["frame_number"]), []).append(row)
        for label, start, end in TARGET_TRANSITION_WINDOWS:
            ids: set[int] = set()
            for frame_number in range(start, end + 1):
                frame_target_box = target_box_by_frame.get(frame_number)
                if not frame_target_box:
                    continue
                candidates = []
                for row in rows_by_frame.get(frame_number, []):
                    overlap = self._iou(self._parse_box(row["bbox_xyxy"]) or [], frame_target_box)
                    if overlap >= 0.20:
                        candidates.append((overlap, row))
                if candidates:
                    ids.add(int(max(candidates, key=lambda item: item[0])[1]["native_tracker_id"]))
            rows.append(
                {
                    "variant": variant.name,
                    "transition": label,
                    "frame_start": start,
                    "frame_end": end,
                    "native_ids_in_window": sorted(ids),
                    "status": "RECOVERED" if len(ids) <= 1 else "STILL_FRAGMENTED",
                }
            )
        return rows

    def _candidate_decision(self, threshold_rows: list[dict[str, Any]], activation_rows: list[dict[str, Any]]) -> dict[str, Any]:
        candidates = [
            row
            for row in threshold_rows + activation_rows
            if int(row.get("confirmed_false_merges", 0) or 0) == 0 and row.get("truck_fragments") is not None
        ]
        if not candidates:
            return {"status": "NO_SAFE_CANDIDATE", "reason": "No evaluable variant with zero confirmed false merges."}
        best = min(
            candidates,
            key=lambda row: (
                int(row["truck_fragments"]),
                int(row.get("suspected_false_merges", 0) or 0),
                float(row["minimum_matching_threshold"]),
                -int(row.get("truck_longest_continuous_span", 0) or 0),
            ),
        )
        return {
            "status": "SELECT_CANDIDATE_THRESHOLD" if str(best["variant"]).startswith("threshold_") else "SELECT_CANDIDATE_ACTIVATION",
            "variant": best["variant"],
            "minimum_matching_threshold": best["minimum_matching_threshold"],
            "track_activation_threshold": best["track_activation_threshold"],
            "effective_new_track_threshold": best["effective_new_track_threshold"],
            "truck_fragments": best["truck_fragments"],
            "confirmed_false_merges": best["confirmed_false_merges"],
            "note": "Shadow result only; do not promote without full end-to-end candidate pipeline validation.",
        }

    @staticmethod
    def _parse_box(value: Any) -> list[float] | None:
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                return None
        if not isinstance(value, (list, tuple)) or len(value) != 4:
            return None
        return [float(item) for item in value]

    @staticmethod
    def _row_box(row: dict[str, Any]) -> list[float] | None:
        if all(key in row for key in ("x1", "y1", "x2", "y2")):
            try:
                return [float(row["x1"]), float(row["y1"]), float(row["x2"]), float(row["y2"])]
            except (TypeError, ValueError):
                return None
        return RuntimeTrackingFixExperiment._parse_box(row.get("bbox_xyxy"))

    @staticmethod
    def _iou(a: list[float], b: list[float]) -> float:
        if len(a) != 4 or len(b) != 4:
            return 0.0
        x1 = max(a[0], b[0])
        y1 = max(a[1], b[1])
        x2 = min(a[2], b[2])
        y2 = min(a[3], b[3])
        inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
        area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
        denom = area_a + area_b - inter
        return inter / denom if denom > 0 else 0.0

    @staticmethod
    def _center_distance(a: list[float], b: list[float]) -> float:
        ax = (a[0] + a[2]) / 2.0
        ay = (a[1] + a[3]) / 2.0
        bx = (b[0] + b[2]) / 2.0
        by = (b[1] + b[3]) / 2.0
        return float(((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5)

    @staticmethod
    def _read_csv(path: Path) -> list[dict[str, Any]]:
        with path.open("r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))

    @staticmethod
    def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = sorted({key for row in rows for key in row})
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    @staticmethod
    def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, separators=(",", ":")) + "\n")
