from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

from .vehicle_enrichment.taxonomy import SUPPORTED_VEHICLE_CLASSES, SUPPORTED_VEHICLE_COLOUR_LABELS
from .vehicle_enrichment.schemas import VEHICLE_COLOUR_UNKNOWN


UNKNOWN_CLASS = "UNKNOWN"
DEFAULT_RECONCILIATION_CONFIG: dict[str, Any] = {
    "enabled": True,
    "max_gap_seconds": 2.0,
    "minimum_observations": 3,
    "history_points": 5,
    "future_points": 5,
    "minimum_motion_pixels": 8.0,
    "position_sigma_pixels": 120.0,
    "base_feasible_distance_pixels": 90.0,
    "speed_distance_multiplier": 2.5,
    "frame_diagonal_multiplier": 0.08,
    "behind_projection_tolerance_pixels": 35.0,
    "direction_reject_cosine": -0.35,
    "weights": {
        "motion": 0.20,
        "position": 0.35,
        "direction": 0.20,
        "colour": 0.15,
        "class": 0.10,
    },
    "match_threshold": 0.72,
    "minimum_margin": 0.12,
    "colour_compatible_pairs": [
        ["BLACK", "GREY"],
        ["WHITE", "SILVER"],
        ["GREY", "SILVER"],
    ],
    "debug_visuals": {
        "enabled": True,
        "source": "tracked_frames",
        "include_accepted": True,
        "include_ambiguous": True,
    },
}


@dataclass(frozen=True, slots=True)
class ObservationPoint:
    frame_number: int
    timestamp_seconds: float
    bbox_xyxy: tuple[float, float, float, float]
    confidence: float
    raw_class_name: str

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox_xyxy
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


@dataclass(slots=True)
class Tracklet:
    local_track_id: str
    camera_id: str
    status: str
    first_frame: int
    last_frame: int
    first_timestamp_seconds: float | None
    last_timestamp_seconds: float | None
    observation_count: int
    final_class: str
    colour: str
    track_payload: dict[str, Any]
    observations: list[ObservationPoint] = field(default_factory=list)

    @property
    def short_track_id(self) -> str:
        return self.local_track_id.split(":")[-1]

    @property
    def first_observation(self) -> ObservationPoint | None:
        return self.observations[0] if self.observations else None

    @property
    def last_observation(self) -> ObservationPoint | None:
        return self.observations[-1] if self.observations else None


@dataclass(frozen=True, slots=True)
class CandidateScore:
    old_track_id: str
    new_track_id: str
    vehicle_id: str | None
    rejected: bool
    rejection_reason: str | None
    score: float
    components: dict[str, float]
    reasons: list[str]
    time_gap_frames: int
    time_gap_seconds: float | None
    distance_pixels: float | None
    predicted_center: tuple[float, float] | None
    old_center: tuple[float, float] | None
    new_center: tuple[float, float] | None
    class_pair: tuple[str, str]
    colour_pair: tuple[str, str]


def run_track_reconciliation_experiment(
    run_dir: str | Path,
    *,
    config_path: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    run_path = Path(run_dir).expanduser().resolve()
    if not run_path.exists():
        raise FileNotFoundError(f"Run directory does not exist: {run_path}")
    config = load_reconciliation_config(config_path, run_path=run_path)
    experiment_dir = Path(output_dir).expanduser().resolve() if output_dir else run_path / "track_reconciliation_test"
    experiment_dir.mkdir(parents=True, exist_ok=True)

    tracks = load_tracklets(run_path)
    assignments, attempts, metrics = reconcile_tracklets(tracks, config)
    track_rows = _build_track_output_rows(tracks, assignments)
    associations = _build_association_rows(track_rows)
    result = {
        "source_run_directory": str(run_path),
        "config": config,
        "metrics": metrics,
        "tracks": track_rows,
        "accepted_associations": associations,
        "attempts": [_candidate_to_dict(item) for item in attempts],
        "manual_validation_template": str(experiment_dir / "manual_validation.csv"),
    }
    (experiment_dir / "track_reconciliation_test.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    _write_associations_csv(experiment_dir / "association_table.csv", associations)
    _write_manual_validation_csv(experiment_dir / "manual_validation.csv", associations)
    _write_report(experiment_dir / "report.md", metrics, associations, config)
    _generate_debug_visuals(run_path, experiment_dir, tracks, track_rows, attempts, config)
    return result


def load_reconciliation_config(config_path: str | Path | None, *, run_path: Path | None = None) -> dict[str, Any]:
    config = _deep_merge(DEFAULT_RECONCILIATION_CONFIG, {})
    candidate_paths: list[Path] = []
    if config_path is not None:
        candidate_paths.append(Path(config_path).expanduser())
    elif run_path is not None and (run_path / "run_config.yaml").exists():
        candidate_paths.append(run_path / "run_config.yaml")
    for path in candidate_paths:
        if not path.exists():
            raise FileNotFoundError(f"Reconciliation config path does not exist: {path}")
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(payload, dict):
            raise ValueError("Configuration root must be a mapping.")
        section = payload.get("track_reconciliation", payload)
        if not isinstance(section, dict):
            raise ValueError("track_reconciliation must be a mapping.")
        config = _deep_merge(config, section)
    config["weights"] = _normalize_weights(dict(config.get("weights", {}) or {}))
    return config


def load_tracklets(run_path: str | Path) -> list[Tracklet]:
    path = Path(run_path)
    tracks_payload = json.loads((path / "tracks.json").read_text(encoding="utf-8"))
    if not isinstance(tracks_payload, list):
        raise ValueError("tracks.json must contain a list.")
    observations_by_track = _load_observations(path / "observations.csv")
    tracklets: list[Tracklet] = []
    for track in tracks_payload:
        if not isinstance(track, dict):
            continue
        local_track_id = str(track.get("local_track_id") or "").strip()
        if not local_track_id:
            continue
        observations = sorted(observations_by_track.get(local_track_id, []), key=lambda item: item.frame_number)
        tracklets.append(
            Tracklet(
                local_track_id=local_track_id,
                camera_id=str(track.get("camera_id") or "").strip(),
                status=str(track.get("status") or "").strip().upper(),
                first_frame=_to_int(track.get("first_frame")),
                last_frame=_to_int(track.get("last_frame")),
                first_timestamp_seconds=_to_optional_float(track.get("first_timestamp_seconds")),
                last_timestamp_seconds=_to_optional_float(track.get("last_timestamp_seconds")),
                observation_count=_to_int(track.get("observation_count")),
                final_class=_normalize_class(track.get("final_class")),
                colour=_extract_track_colour(track),
                track_payload=dict(track),
                observations=observations,
            )
        )
    return tracklets


def reconcile_tracklets(tracklets: list[Tracklet], config: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], list[CandidateScore], dict[str, Any]]:
    completed = [track for track in tracklets if track.status == "COMPLETED"]
    ordered = sorted(completed, key=lambda item: (item.first_frame, item.last_frame, item.local_track_id))
    vehicle_by_track: dict[str, str] = {}
    origin_by_vehicle: dict[str, str] = {}
    assignment: dict[str, dict[str, Any]] = {}
    attempts: list[CandidateScore] = []
    vehicle_counter = 0
    accepted = 0
    ambiguous = 0
    rejected = 0
    no_candidate = 0

    for new_track in ordered:
        candidates = [old for old in ordered if old.local_track_id != new_track.local_track_id and old.first_frame < new_track.first_frame]
        scored = [score_candidate(old, new_track, config, vehicle_by_track.get(old.local_track_id)) for old in candidates]
        attempts.extend(scored)
        viable = sorted([item for item in scored if not item.rejected], key=lambda item: item.score, reverse=True)
        rejected += len([item for item in scored if item.rejected])
        best = viable[0] if viable else None
        second = viable[1] if len(viable) > 1 else None
        second_score = second.score if second is not None else 0.0

        if best is None:
            vehicle_counter += 1
            vehicle_id = _format_vehicle_id(vehicle_counter)
            vehicle_by_track[new_track.local_track_id] = vehicle_id
            origin_by_vehicle[vehicle_id] = new_track.local_track_id
            assignment[new_track.local_track_id] = _unmatched_assignment(vehicle_id, reason="no_recent_candidate")
            no_candidate += 1
            continue

        threshold = float(config["match_threshold"])
        margin = float(config["minimum_margin"])
        if best.score >= threshold and best.score - second_score >= margin and best.vehicle_id:
            vehicle_id = best.vehicle_id
            vehicle_by_track[new_track.local_track_id] = vehicle_id
            assignment[new_track.local_track_id] = {
                "vehicle_id": vehicle_id,
                "matched": True,
                "previous_track_id": best.old_track_id,
                "score": round(best.score, 6),
                "second_best_score": round(second_score, 6),
                "time_gap_frames": best.time_gap_frames,
                "time_gap_seconds": best.time_gap_seconds,
                "reasons": list(best.reasons),
                "components": dict(best.components),
                "result": "accepted",
            }
            accepted += 1
            continue

        vehicle_counter += 1
        vehicle_id = _format_vehicle_id(vehicle_counter)
        vehicle_by_track[new_track.local_track_id] = vehicle_id
        origin_by_vehicle[vehicle_id] = new_track.local_track_id
        if best.score >= threshold:
            reason = "ambiguous_candidate_margin"
            ambiguous += 1
        else:
            reason = "no_confident_candidate"
            no_candidate += 1
        assignment[new_track.local_track_id] = {
            "vehicle_id": vehicle_id,
            "matched": False,
            "reason": reason,
            "best_candidate_track_id": best.old_track_id,
            "best_score": round(best.score, 6),
            "second_best_score": round(second_score, 6),
            "minimum_margin": margin,
            "match_threshold": threshold,
            "result": "ambiguous" if reason == "ambiguous_candidate_margin" else "unmatched",
        }

    all_vehicle_ids = {item["vehicle_id"] for item in assignment.values()}
    metrics = {
        "raw_bytetrack_unique_tracks": len(completed),
        "reconciled_vehicle_identities": len(all_vehicle_ids),
        "potential_duplicate_tracks_removed": len(completed) - len(all_vehicle_ids),
        "reconciliation_attempts": len(attempts),
        "accepted_matches": accepted,
        "rejected_candidate_matches": rejected,
        "ambiguous_matches": ambiguous,
        "unmatched_tracks": no_candidate,
        "track_fragments_merged": accepted,
    }
    return assignment, attempts, metrics


def score_candidate(old: Tracklet, new: Tracklet, config: dict[str, Any], vehicle_id: str | None) -> CandidateScore:
    gap_frames = int(new.first_frame - old.last_frame)
    gap_seconds = _time_gap_seconds(old, new)
    old_obs = old.last_observation
    new_obs = new.first_observation
    old_center = old_obs.center if old_obs else None
    new_center = new_obs.center if new_obs else None
    class_pair = (old.final_class, new.final_class)
    colour_pair = (old.colour, new.colour)

    def reject(reason: str) -> CandidateScore:
        return CandidateScore(
            old_track_id=old.local_track_id,
            new_track_id=new.local_track_id,
            vehicle_id=vehicle_id,
            rejected=True,
            rejection_reason=reason,
            score=0.0,
            components={},
            reasons=[],
            time_gap_frames=gap_frames,
            time_gap_seconds=gap_seconds,
            distance_pixels=None,
            predicted_center=None,
            old_center=old_center,
            new_center=new_center,
            class_pair=class_pair,
            colour_pair=colour_pair,
        )

    if old.camera_id != new.camera_id:
        return reject("different_camera")
    if gap_frames <= 0:
        return reject("overlapping_or_non_sequential_track")
    max_gap_seconds = float(config["max_gap_seconds"])
    if gap_seconds is not None and gap_seconds > max_gap_seconds:
        return reject("time_gap_exceeds_window")
    if gap_seconds is None:
        return reject("missing_timestamps")
    if old.final_class != UNKNOWN_CLASS and new.final_class != UNKNOWN_CLASS and old.final_class != new.final_class:
        return reject("conflicting_vehicle_class")
    if old_center is None or new_center is None:
        return reject("missing_observations")

    old_velocity = _velocity(old.observations, tail=True, max_points=int(config["history_points"]))
    new_velocity = _velocity(new.observations, tail=False, max_points=int(config["future_points"]))
    predicted_center = (old_center[0] + old_velocity[0] * gap_frames, old_center[1] + old_velocity[1] * gap_frames)
    distance = _distance(predicted_center, new_center)
    observed_displacement = (new_center[0] - old_center[0], new_center[1] - old_center[1])
    feasible_distance = _feasible_distance(old, new, old_velocity, gap_frames, config)
    if distance > feasible_distance:
        return reject("impossible_spatial_displacement")
    old_direction = _direction_vector(old.observations, tail=True, max_points=int(config["history_points"]), min_motion=float(config["minimum_motion_pixels"]))
    new_direction = _direction_vector(new.observations, tail=False, max_points=int(config["future_points"]), min_motion=float(config["minimum_motion_pixels"]))
    if old_direction is not None:
        projected = _dot(observed_displacement, old_direction)
        if projected < -float(config["behind_projection_tolerance_pixels"]):
            return reject("appears_behind_old_trajectory")

    direction_cosine = None
    if old_direction is not None and new_direction is not None:
        direction_cosine = _dot(old_direction, new_direction)
        if direction_cosine < float(config["direction_reject_cosine"]):
            return reject("opposite_direction")

    position_score = max(0.0, math.exp(-distance / max(float(config["position_sigma_pixels"]), 1.0)))
    motion_score = _motion_score(old_velocity, new_velocity)
    direction_score = _direction_score(direction_cosine)
    class_score = 0.60 if old.final_class == UNKNOWN_CLASS or new.final_class == UNKNOWN_CLASS else 1.0
    colour_score, colour_reason = _colour_score(old.colour, new.colour, config)
    components = {
        "motion": motion_score,
        "position": position_score,
        "direction": direction_score,
        "colour": colour_score,
        "class": class_score,
    }
    weights = dict(config["weights"])
    score = sum(float(components[key]) * float(weights[key]) for key in weights)
    reasons = ["same_camera", "within_time_window", "spatially_feasible"]
    reasons.append("same_class" if class_score == 1.0 else "class_unknown_allowed")
    reasons.append(colour_reason)
    reasons.append("direction_consistent" if direction_score >= 0.70 else "direction_uncertain")
    reasons.append("motion_consistent" if motion_score >= 0.70 else "motion_uncertain")
    return CandidateScore(
        old_track_id=old.local_track_id,
        new_track_id=new.local_track_id,
        vehicle_id=vehicle_id,
        rejected=False,
        rejection_reason=None,
        score=score,
        components={key: round(value, 6) for key, value in components.items()},
        reasons=reasons,
        time_gap_frames=gap_frames,
        time_gap_seconds=gap_seconds,
        distance_pixels=distance,
        predicted_center=predicted_center,
        old_center=old_center,
        new_center=new_center,
        class_pair=class_pair,
        colour_pair=colour_pair,
    )


def _load_observations(path: Path) -> dict[str, list[ObservationPoint]]:
    if not path.exists():
        raise FileNotFoundError(f"observations.csv is required for trajectory reconciliation: {path}")
    grouped: dict[str, list[ObservationPoint]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            local_track_id = str(row.get("local_track_id") or "").strip()
            if not local_track_id:
                continue
            grouped.setdefault(local_track_id, []).append(
                ObservationPoint(
                    frame_number=_to_int(row.get("frame_number")),
                    timestamp_seconds=float(row.get("timestamp_seconds") or 0.0),
                    bbox_xyxy=(
                        float(row.get("x1") or 0.0),
                        float(row.get("y1") or 0.0),
                        float(row.get("x2") or 0.0),
                        float(row.get("y2") or 0.0),
                    ),
                    confidence=float(row.get("confidence") or 0.0),
                    raw_class_name=str(row.get("raw_class_name") or ""),
                )
            )
    return grouped


def _extract_track_colour(track: dict[str, Any]) -> str:
    enrichment = track.get("vehicle_enrichment")
    if isinstance(enrichment, dict):
        colour = enrichment.get("vehicle_colour")
        if isinstance(colour, dict):
            return _normalize_colour(colour.get("label"))
    return VEHICLE_COLOUR_UNKNOWN


def _normalize_class(value: Any) -> str:
    normalized = str(value or UNKNOWN_CLASS).strip().upper()
    return normalized if normalized in SUPPORTED_VEHICLE_CLASSES else UNKNOWN_CLASS


def _normalize_colour(value: Any) -> str:
    normalized = str(value or VEHICLE_COLOUR_UNKNOWN).strip().upper()
    return normalized if normalized in SUPPORTED_VEHICLE_COLOUR_LABELS else VEHICLE_COLOUR_UNKNOWN


def _time_gap_seconds(old: Tracklet, new: Tracklet) -> float | None:
    if old.last_timestamp_seconds is None or new.first_timestamp_seconds is None:
        return None
    return max(0.0, float(new.first_timestamp_seconds) - float(old.last_timestamp_seconds))


def _velocity(observations: list[ObservationPoint], *, tail: bool, max_points: int) -> tuple[float, float]:
    points = observations[-max_points:] if tail else observations[:max_points]
    if len(points) < 2:
        return (0.0, 0.0)
    first = points[0]
    last = points[-1]
    frame_delta = max(1, int(last.frame_number - first.frame_number))
    return ((last.center[0] - first.center[0]) / frame_delta, (last.center[1] - first.center[1]) / frame_delta)


def _direction_vector(observations: list[ObservationPoint], *, tail: bool, max_points: int, min_motion: float) -> tuple[float, float] | None:
    points = observations[-max_points:] if tail else observations[:max_points]
    if len(points) < 2:
        return None
    dx = points[-1].center[0] - points[0].center[0]
    dy = points[-1].center[1] - points[0].center[1]
    magnitude = math.hypot(dx, dy)
    if magnitude < min_motion:
        return None
    return (dx / magnitude, dy / magnitude)


def _feasible_distance(old: Tracklet, new: Tracklet, velocity: tuple[float, float], gap_frames: int, config: dict[str, Any]) -> float:
    centers = [item.center for item in old.observations + new.observations]
    max_x = max((point[0] for point in centers), default=0.0)
    max_y = max((point[1] for point in centers), default=0.0)
    diag = math.hypot(max_x, max_y)
    speed = math.hypot(*velocity)
    return (
        float(config["base_feasible_distance_pixels"])
        + speed * max(gap_frames, 1) * float(config["speed_distance_multiplier"])
        + diag * float(config["frame_diagonal_multiplier"])
    )


def _motion_score(old_velocity: tuple[float, float], new_velocity: tuple[float, float]) -> float:
    old_speed = math.hypot(*old_velocity)
    new_speed = math.hypot(*new_velocity)
    if old_speed < 1.0 or new_speed < 1.0:
        return 0.60
    ratio = min(old_speed, new_speed) / max(old_speed, new_speed)
    return max(0.0, min(1.0, ratio))


def _direction_score(cosine: float | None) -> float:
    if cosine is None:
        return 0.60
    return max(0.0, min(1.0, (cosine + 1.0) / 2.0))


def _colour_score(old_colour: str, new_colour: str, config: dict[str, Any]) -> tuple[float, str]:
    if old_colour == VEHICLE_COLOUR_UNKNOWN or new_colour == VEHICLE_COLOUR_UNKNOWN:
        return 0.50, "colour_unknown_allowed"
    if old_colour == new_colour:
        return 1.0, "same_colour"
    compatible = {tuple(sorted((_normalize_colour(a), _normalize_colour(b)))) for a, b in config.get("colour_compatible_pairs", [])}
    if tuple(sorted((old_colour, new_colour))) in compatible:
        return 0.70, "compatible_colour"
    return 0.0, "conflicting_colour_penalty"


def _build_track_output_rows(tracklets: list[Tracklet], assignments: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for track in sorted(tracklets, key=lambda item: (item.first_frame, item.local_track_id)):
        assigned = assignments.get(track.local_track_id)
        if assigned is None:
            continue
        payload = dict(track.track_payload)
        payload["track_id"] = track.short_track_id
        payload["vehicle_id"] = assigned["vehicle_id"]
        payload["reconciliation"] = {key: value for key, value in assigned.items() if key != "vehicle_id"}
        rows.append(payload)
    return rows


def _build_association_rows(track_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in track_rows:
        reconciliation = row.get("reconciliation")
        if not isinstance(reconciliation, dict) or not reconciliation.get("matched"):
            continue
        enrichment = row.get("vehicle_enrichment") if isinstance(row.get("vehicle_enrichment"), dict) else {}
        colour = VEHICLE_COLOUR_UNKNOWN
        colour_payload = enrichment.get("vehicle_colour") if isinstance(enrichment, dict) else None
        if isinstance(colour_payload, dict):
            colour = _normalize_colour(colour_payload.get("label"))
        rows.append(
            {
                "old_track": str(reconciliation.get("previous_track_id")),
                "new_track": str(row.get("local_track_id")),
                "vehicle_id": str(row.get("vehicle_id")),
                "gap_frames": reconciliation.get("time_gap_frames"),
                "gap_seconds": reconciliation.get("time_gap_seconds"),
                "score": reconciliation.get("score"),
                "second_best_score": reconciliation.get("second_best_score"),
                "colour": colour,
                "class": _normalize_class(row.get("final_class")),
                "result": "ACCEPTED",
            }
        )
    return rows


def _generate_debug_visuals(
    run_path: Path,
    experiment_dir: Path,
    tracklets: list[Tracklet],
    track_rows: list[dict[str, Any]],
    attempts: list[CandidateScore],
    config: dict[str, Any],
) -> None:
    visual_config = dict(config.get("debug_visuals", {}) or {})
    if not bool(visual_config.get("enabled", True)):
        return
    source_video_by_camera = _load_source_video_by_camera(run_path)
    by_track = {track.local_track_id: track for track in tracklets}
    rows_by_track = {str(row.get("local_track_id")): row for row in track_rows}
    accepted_pairs = [
        (str(row.get("reconciliation", {}).get("previous_track_id")), str(row.get("local_track_id")), "accepted")
        for row in track_rows
        if isinstance(row.get("reconciliation"), dict) and row.get("reconciliation", {}).get("matched")
    ]
    ambiguous_pairs: list[tuple[str, str, str]] = []
    if bool(visual_config.get("include_ambiguous", True)):
        for row in track_rows:
            reconciliation = row.get("reconciliation")
            if not isinstance(reconciliation, dict) or reconciliation.get("result") != "ambiguous":
                continue
            candidate = reconciliation.get("best_candidate_track_id")
            if candidate:
                ambiguous_pairs.append((str(candidate), str(row.get("local_track_id")), "ambiguous"))
    for old_id, new_id, result_type in accepted_pairs + ambiguous_pairs:
        old = by_track.get(old_id)
        new = by_track.get(new_id)
        new_row = rows_by_track.get(new_id, {})
        if old is None or new is None:
            continue
        pair_dir = experiment_dir / "visual_evidence" / result_type / f"{_safe_name(old_id)}__{_safe_name(new_id)}"
        before_dir = pair_dir / "before_occlusion"
        after_dir = pair_dir / "after_occlusion"
        before_dir.mkdir(parents=True, exist_ok=True)
        after_dir.mkdir(parents=True, exist_ok=True)
        vehicle_id = str(new_row.get("vehicle_id") or "")
        before = _draw_track_frame(
            run_path,
            old,
            vehicle_id=vehicle_id,
            label_suffix="before",
            source=str(visual_config.get("source", "tracked_frames")),
            source_video_by_camera=source_video_by_camera,
        )
        after = _draw_track_frame(
            run_path,
            new,
            vehicle_id=vehicle_id,
            label_suffix="after",
            source=str(visual_config.get("source", "tracked_frames")),
            source_video_by_camera=source_video_by_camera,
            use_first=True,
        )
        before_path = before_dir / f"{old.short_track_id}_last_frame_{old.last_frame:06d}.jpg"
        after_path = after_dir / f"{new.short_track_id}_first_frame_{new.first_frame:06d}.jpg"
        if before is not None:
            cv2.imwrite(str(before_path), before, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        if after is not None:
            cv2.imwrite(str(after_path), after, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        if before is not None and after is not None:
            contact = np.hstack([_resize_for_contact(before), _resize_for_contact(after)])
            cv2.imwrite(str(pair_dir / "before_after_contact_sheet.jpg"), contact, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        _write_pair_debug(pair_dir / "decision.json", old_id, new_id, attempts)


def _draw_track_frame(
    run_path: Path,
    track: Tracklet,
    *,
    vehicle_id: str,
    label_suffix: str,
    source: str,
    source_video_by_camera: dict[str, Path],
    use_first: bool = False,
) -> np.ndarray | None:
    obs = track.first_observation if use_first else track.last_observation
    if obs is None:
        return None
    frame_path = run_path / source / track.camera_id / f"frame_{obs.frame_number:06d}.jpg"
    if not frame_path.exists():
        fallback = run_path / "raw_frames" / track.camera_id / f"frame_{obs.frame_number:06d}.jpg"
        frame_path = fallback if fallback.exists() else frame_path
    frame = cv2.imread(str(frame_path)) if frame_path.exists() else None
    if frame is None:
        frame = _read_source_video_frame(source_video_by_camera.get(track.camera_id), obs.frame_number)
    if frame is None:
        return None
    x1, y1, x2, y2 = [int(round(value)) for value in obs.bbox_xyxy]
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 255), 2)
    label = f"T:{track.short_track_id} V:{vehicle_id} {label_suffix}"
    y_text = max(18, y1 - 8)
    cv2.putText(frame, label, (x1, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(frame, label, (x1, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 220, 255), 2, cv2.LINE_AA)
    return frame


def _load_source_video_by_camera(run_path: Path) -> dict[str, Path]:
    config_path = run_path / "run_config.yaml"
    if not config_path.exists():
        return {}
    try:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}
    cameras = payload.get("input", {}).get("cameras", []) if isinstance(payload, dict) else []
    result: dict[str, Path] = {}
    for camera in cameras:
        if not isinstance(camera, dict) or str(camera.get("source_type") or "").lower() != "video":
            continue
        source = Path(str(camera.get("source") or "")).expanduser()
        if source.exists():
            result[str(camera.get("camera_id") or "")] = source
    return result


def _read_source_video_frame(source_path: Path | None, frame_number: int) -> np.ndarray | None:
    if source_path is None or not source_path.exists():
        return None
    capture = cv2.VideoCapture(str(source_path))
    try:
        if not capture.isOpened():
            return None
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_number))
        ok, frame = capture.read()
        return frame if ok else None
    finally:
        capture.release()


def _resize_for_contact(frame: np.ndarray) -> np.ndarray:
    target_height = 480
    ratio = target_height / float(frame.shape[0])
    return cv2.resize(frame, (int(frame.shape[1] * ratio), target_height))


def _write_pair_debug(path: Path, old_id: str, new_id: str, attempts: list[CandidateScore]) -> None:
    rows = [item for item in attempts if item.old_track_id == old_id and item.new_track_id == new_id]
    path.write_text(json.dumps([_candidate_to_dict(item) for item in rows], indent=2), encoding="utf-8")


def _write_associations_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = ["old_track", "new_track", "vehicle_id", "gap_frames", "gap_seconds", "score", "second_best_score", "colour", "class", "result"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_manual_validation_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = ["old_track", "new_track", "vehicle_id", "score", "manual_label", "reviewer_notes"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "old_track": row["old_track"],
                    "new_track": row["new_track"],
                    "vehicle_id": row["vehicle_id"],
                    "score": row["score"],
                    "manual_label": "UNCERTAIN",
                    "reviewer_notes": "",
                }
            )


def _write_report(path: Path, metrics: dict[str, Any], associations: list[dict[str, Any]], config: dict[str, Any]) -> None:
    lines = [
        "# Track Reconciliation Test",
        "",
        f"- Raw ByteTrack unique tracks: `{metrics['raw_bytetrack_unique_tracks']}`",
        f"- Reconciled vehicle identities: `{metrics['reconciled_vehicle_identities']}`",
        f"- Potential duplicate tracks removed: `{metrics['potential_duplicate_tracks_removed']}`",
        f"- Reconciliation attempts: `{metrics['reconciliation_attempts']}`",
        f"- Accepted matches: `{metrics['accepted_matches']}`",
        f"- Rejected candidate matches: `{metrics['rejected_candidate_matches']}`",
        f"- Ambiguous matches: `{metrics['ambiguous_matches']}`",
        "",
        "## Configuration",
        "",
        "```yaml",
        yaml.safe_dump({"track_reconciliation": config}, sort_keys=False).strip(),
        "```",
        "",
        "## Accepted Associations",
        "",
        "| OLD TRACK | NEW TRACK | VEHICLE ID | GAP | SCORE | COLOUR | CLASS | RESULT |",
        "|---|---|---:|---:|---:|---|---|---|",
    ]
    if associations:
        for row in associations:
            lines.append(
                f"| {row['old_track']} | {row['new_track']} | {row['vehicle_id']} | {row['gap_frames']} | {row['score']} | {row['colour']} | {row['class']} | {row['result']} |"
            )
    else:
        lines.append("|  |  |  |  |  |  |  | no accepted recoveries |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _candidate_to_dict(item: CandidateScore) -> dict[str, Any]:
    return asdict(item)


def _unmatched_assignment(vehicle_id: str, *, reason: str) -> dict[str, Any]:
    return {
        "vehicle_id": vehicle_id,
        "matched": False,
        "reason": reason,
        "result": "unmatched",
    }


def _normalize_weights(weights: dict[str, Any]) -> dict[str, float]:
    required = ["motion", "position", "direction", "colour", "class"]
    normalized = {key: float(weights.get(key, DEFAULT_RECONCILIATION_CONFIG["weights"][key])) for key in required}
    total = sum(normalized.values())
    if total <= 0.0:
        raise ValueError("track_reconciliation.weights must sum to a positive value.")
    return {key: value / total for key, value in normalized.items()}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = {key: _deep_merge(value, {}) if isinstance(value, dict) else value for key, value in base.items()}
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _format_vehicle_id(index: int) -> str:
    return f"VEHICLE_{index:03d}"


def _distance(left: tuple[float, float], right: tuple[float, float]) -> float:
    return math.hypot(left[0] - right[0], left[1] - right[1])


def _dot(left: tuple[float, float], right: tuple[float, float]) -> float:
    return left[0] * right[0] + left[1] * right[1]


def _to_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _to_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_name(value: str) -> str:
    return str(value).replace(":", "_").replace("/", "_").replace("\\", "_")
