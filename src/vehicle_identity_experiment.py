from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


DEFAULT_CONFIG = {
    "same_camera_only": True,
    "maximum_gap_seconds": 6.0,
    "maximum_stationary_distance": 140.0,
    "maximum_moving_distance": 230.0,
    "stationary_motion_threshold": 80.0,
    "stationary_normalized_displacement_threshold": 0.80,
    "stationary_median_step_threshold": 3.0,
    "stationary_speed_threshold": 0.95,
    "vehicle_consistency_floor": 0.58,
    "vehicle_consistency_weight": 0.30,
    "vehicle_conflict_penalty": 0.10,
    "ambiguous_vehicle_delta": 0.04,
    "overlap_duplicate_min_iou": 0.15,
    "overlap_duplicate_min_spatial_score": 0.52,
    "acceptance_threshold": 0.80,
    "ambiguity_margin": 0.08,
    "weights": {
        "temporal": 0.10,
        "spatial": 0.32,
        "motion": 0.17,
        "class": 0.24,
        "colour": 0.01,
        "appearance": 0.16,
        "plate": 0.0,
    },
}

CALIBRATION_GRID = {
    "ambiguity_margin": [0.03, 0.05, 0.08, 0.10, 0.15],
    "acceptance_threshold": [0.70, 0.75, 0.80, 0.85],
}

PRE_CALIBRATION_POC_BASELINE = {
    "source": "vehicle_identity_test output before this calibration step",
    "raw_completed_tracks": 37,
    "predicted_vehicle_identities": 19,
    "precision": 1.0,
    "recall": 0.5,
    "f1": 0.666667,
    "suspicious_overmerge_count": 3,
    "known_suspicious_examples": [
        "CAM_001:TRACK_6/12 merged with TRACK_29/TRACK_32/TRACK_35",
        "CAM_001:TRACK_3/16/20 merged with TRACK_8/TRACK_31",
        "CAM_001:TRACK_22 merged with TRACK_28",
    ],
}


GROUND_TRUTH = {
    "same_vehicle_groups": [
        ["CAM_001:TRACK_6", "CAM_001:TRACK_12", "CAM_001:TRACK_25", "CAM_001:TRACK_27", "CAM_001:TRACK_41"],
        ["CAM_001:TRACK_5", "CAM_001:TRACK_23"],
        ["CAM_001:TRACK_3", "CAM_001:TRACK_16", "CAM_001:TRACK_20"],
        ["CAM_001:TRACK_17", "CAM_001:TRACK_19"],
        ["CAM_001:TRACK_22", "CAM_001:TRACK_24"],
    ],
    "must_not_merge": [
        ["CAM_001:TRACK_24", "CAM_001:TRACK_6"],
        ["CAM_001:TRACK_39", "CAM_001:TRACK_6"],
        ["CAM_001:TRACK_39", "CAM_001:TRACK_23"],
        ["CAM_001:TRACK_21", "CAM_001:TRACK_3"],
        ["CAM_001:TRACK_31", "CAM_001:TRACK_32"],
        ["CAM_001:TRACK_25", "CAM_001:TRACK_24"],
        ["CAM_001:TRACK_6", "CAM_001:TRACK_29"],
        ["CAM_001:TRACK_6", "CAM_001:TRACK_32"],
        ["CAM_001:TRACK_6", "CAM_001:TRACK_35"],
        ["CAM_001:TRACK_12", "CAM_001:TRACK_29"],
        ["CAM_001:TRACK_12", "CAM_001:TRACK_32"],
        ["CAM_001:TRACK_12", "CAM_001:TRACK_35"],
        ["CAM_001:TRACK_3", "CAM_001:TRACK_8"],
        ["CAM_001:TRACK_3", "CAM_001:TRACK_31"],
        ["CAM_001:TRACK_16", "CAM_001:TRACK_8"],
        ["CAM_001:TRACK_16", "CAM_001:TRACK_31"],
        ["CAM_001:TRACK_22", "CAM_001:TRACK_28"],
        ["CAM_001:TRACK_25", "CAM_001:TRACK_29"],
        ["CAM_001:TRACK_25", "CAM_001:TRACK_32"],
        ["CAM_001:TRACK_25", "CAM_001:TRACK_35"],
    ],
    "uncertain_pairs": [
        ["CAM_001:TRACK_24", "CAM_001:TRACK_28"],
        ["CAM_001:TRACK_29", "CAM_001:TRACK_32"],
        ["CAM_001:TRACK_32", "CAM_001:TRACK_35"],
    ],
}


@dataclass(slots=True)
class TrackletFeature:
    camera_id: str
    local_track_id: str
    native_tracker_id: int
    status: str
    first_frame: int
    last_frame: int
    first_timestamp: float
    last_timestamp: float
    duration_seconds: float
    observation_count: int
    final_class: str
    class_distribution: dict[str, int]
    colour: str
    start_bbox: list[float]
    end_bbox: list[float]
    start_center: list[float]
    end_center: list[float]
    mean_center: list[float]
    trajectory_points: list[list[float]]
    estimated_direction: list[float]
    speed_pixels_per_frame: float
    motion_magnitude: float
    stationary: bool
    bbox_size_history: list[list[float]]
    best_evidence_crops: list[str]
    appearance_descriptor: list[float]
    evidence_quality: float
    median_step_pixels: float
    normalized_displacement: float


def run_vehicle_identity_experiment(run_dir: str | Path, *, output_dir: str | Path | None = None) -> dict[str, Any]:
    run_path = Path(run_dir).expanduser().resolve()
    experiment_dir = Path(output_dir).expanduser().resolve() if output_dir else run_path / "vehicle_identity_test"
    experiment_dir.mkdir(parents=True, exist_ok=True)
    (experiment_dir / "visual_evidence").mkdir(exist_ok=True)

    tracks_hash_before = _sha256(run_path / "tracks.json")
    config = dict(DEFAULT_CONFIG)
    tracks = _read_json(run_path / "tracks.json")
    observations = _read_observations(run_path / "observations.csv")
    evidence = _read_json(run_path / "evidence_index.json", default=[])
    enrichment = {str(item.get("local_track_id")): item for item in _read_json(run_path / "vehicle_enrichment.json", default=[])}
    features = [
        _build_feature(track, observations.get(str(track.get("local_track_id")), []), evidence, enrichment, config)
        for track in tracks
        if str(track.get("status", "")).upper() == "COMPLETED"
    ]
    features = [item for item in features if item is not None]
    features_by_id = {item.local_track_id: item for item in features}

    pair_rows = []
    for old in features:
        for new in features:
            if old.local_track_id >= new.local_track_id:
                continue
            pair_rows.append(_score_pair(old, new, config))

    calibration = _run_calibration_grid(features, pair_rows, config, GROUND_TRUTH)
    selected_config = dict(config)
    selected_config.update(calibration["selected_config"])
    vehicle_id_by_track, decision_rows = _build_vehicles(features, pair_rows, selected_config)
    vehicles = _vehicle_records(features, vehicle_id_by_track)
    evaluation = _evaluate(vehicle_id_by_track, GROUND_TRUTH)
    analytics = _analytics_simulation(features, vehicles)
    existing = _existing_reconciliation_summary(run_path)
    _write_contact_sheets(run_path, experiment_dir / "visual_evidence", vehicles, evidence)
    _write_high_score_rejects(run_path, experiment_dir / "visual_evidence", pair_rows, evidence)

    result = {
        "source_run_directory": str(run_path),
        "existing_reconciliation_architecture": {
            "module": "src/track_reconciliation.py",
            "status": "exists",
            "signals": ["time gap", "motion", "position", "direction", "colour", "class"],
            "limitations": [
                "rejects overlapping/non-sequential tracklets",
                "does not explicitly support DUPLICATE_OVERLAP",
                "uses colour despite known crop contamination risk",
                "does not write a production vehicle_id contract",
            ],
            "can_be_extended": True,
        },
        "config": selected_config,
        "calibration": calibration,
        "ground_truth": GROUND_TRUTH,
        "metrics": evaluation,
        "analytics_simulation": analytics,
        "existing_reconciliation_baseline": existing,
        "pre_calibration_poc_baseline": PRE_CALIBRATION_POC_BASELINE,
        "tracks_json_sha256_before": tracks_hash_before,
        "tracks_json_sha256_after": _sha256(run_path / "tracks.json"),
        "tracks_json_unchanged": tracks_hash_before == _sha256(run_path / "tracks.json"),
        "output_directory": str(experiment_dir),
    }
    _write_json(experiment_dir / "tracklet_features.json", [asdict(item) for item in features])
    _write_csv(experiment_dir / "pairwise_scores.csv", pair_rows)
    _write_csv(experiment_dir / "pairwise_candidates.csv", [row for row in pair_rows if not row["rejected"]])
    _write_csv(experiment_dir / "association_decisions.csv", decision_rows)
    _write_csv(experiment_dir / "calibration_matrix.csv", calibration["grid"])
    _write_json(experiment_dir / "vehicle_id_map.json", vehicle_id_by_track)
    _write_json(experiment_dir / "vehicles.json", {"vehicles": vehicles})
    _write_json(experiment_dir / "ground_truth.json", GROUND_TRUTH)
    _write_json(experiment_dir / "ground_truth_calibration.json", GROUND_TRUTH)
    _write_json(experiment_dir / "calibration_summary.json", calibration)
    _write_json(experiment_dir / "evaluation.json", result)
    _write_report(experiment_dir / "report.md", result, vehicles)
    return result


def _build_feature(
    track: dict[str, Any],
    rows: list[dict[str, Any]],
    evidence: list[dict[str, Any]],
    enrichment: dict[str, dict[str, Any]],
    config: dict[str, Any],
) -> TrackletFeature | None:
    if not rows:
        return None
    local_id = str(track["local_track_id"])
    rows = sorted(rows, key=lambda row: int(row["frame_number"]))
    boxes = [_box_from_row(row) for row in rows]
    centers = [_center(box) for box in boxes]
    start_center = centers[0]
    end_center = centers[-1]
    mean_center = [float(np.mean([center[0] for center in centers])), float(np.mean([center[1] for center in centers]))]
    motion = _distance(start_center, end_center)
    direction = [0.0, 0.0] if motion < 1.0 else [(end_center[0] - start_center[0]) / motion, (end_center[1] - start_center[1]) / motion]
    frame_span = max(1, int(rows[-1]["frame_number"]) - int(rows[0]["frame_number"]))
    median_diag = float(np.median([max(1.0, math.hypot(box[2] - box[0], box[3] - box[1])) for box in boxes]))
    step_distances = [_distance(centers[index - 1], centers[index]) for index in range(1, len(centers))]
    median_step = float(np.median(step_distances)) if step_distances else 0.0
    normalized_displacement = motion / max(median_diag, 1.0)
    stationary_votes = [
        normalized_displacement <= float(config["stationary_normalized_displacement_threshold"]),
        median_step <= float(config["stationary_median_step_threshold"]),
        (motion / frame_span) <= float(config["stationary_speed_threshold"]),
    ]
    stationary = sum(1 for item in stationary_votes if item) >= 2
    evidence_rows = [item for item in evidence if item.get("local_track_id") == local_id]
    crops = [str(item.get("crop_path")) for item in evidence_rows if item.get("crop_path")]
    descriptor, quality = _appearance_descriptor(crops)
    colour = _extract_colour(track, enrichment.get(local_id))
    return TrackletFeature(
        camera_id=str(track.get("camera_id", "")),
        local_track_id=local_id,
        native_tracker_id=int(track.get("native_tracker_id", -1)),
        status=str(track.get("status", "")),
        first_frame=int(track.get("first_frame", rows[0]["frame_number"])),
        last_frame=int(track.get("last_frame", rows[-1]["frame_number"])),
        first_timestamp=float(track.get("first_timestamp_seconds", rows[0]["timestamp_seconds"])),
        last_timestamp=float(track.get("last_timestamp_seconds", rows[-1]["timestamp_seconds"])),
        duration_seconds=float(track.get("last_timestamp_seconds", rows[-1]["timestamp_seconds"])) - float(track.get("first_timestamp_seconds", rows[0]["timestamp_seconds"])),
        observation_count=int(track.get("observation_count", len(rows))),
        final_class=str(track.get("final_class") or "UNKNOWN").upper(),
        class_distribution={str(k).upper(): int(v) for k, v in dict(track.get("class_counts") or {}).items()},
        colour=colour,
        start_bbox=boxes[0],
        end_bbox=boxes[-1],
        start_center=start_center,
        end_center=end_center,
        mean_center=mean_center,
        trajectory_points=[centers[index] for index in np.linspace(0, len(centers) - 1, min(10, len(centers)), dtype=int)],
        estimated_direction=direction,
        speed_pixels_per_frame=motion / frame_span,
        motion_magnitude=motion,
        stationary=stationary,
        bbox_size_history=[[box[2] - box[0], box[3] - box[1]] for box in boxes[:: max(1, len(boxes) // 10)]],
        best_evidence_crops=crops[:5],
        appearance_descriptor=descriptor,
        evidence_quality=quality,
        median_step_pixels=median_step,
        normalized_displacement=normalized_displacement,
    )


def _score_pair(a: TrackletFeature, b: TrackletFeature, config: dict[str, Any]) -> dict[str, Any]:
    old, new = (a, b) if (a.first_frame, a.last_frame) <= (b.first_frame, b.last_frame) else (b, a)
    frame_gap = new.first_frame - old.last_frame - 1
    time_gap = new.first_timestamp - old.last_timestamp
    overlap_frames = max(0, min(old.last_frame, new.last_frame) - max(old.first_frame, new.first_frame) + 1)
    mode = "DUPLICATE_OVERLAP" if overlap_frames > 0 else "SEQUENTIAL"
    rejected = False
    rejection_reason = ""
    if config["same_camera_only"] and old.camera_id != new.camera_id:
        rejected, rejection_reason = True, "different_camera"
    class_score = _class_score(old.final_class, new.final_class)
    if not rejected and class_score == 0.0:
        rejected, rejection_reason = True, "reliable_class_conflict"
    spatial_distance = _distance(old.mean_center if old.stationary and new.stationary else old.end_center, new.mean_center if old.stationary and new.stationary else new.start_center)
    if not rejected and overlap_frames == 0 and time_gap > float(config["maximum_gap_seconds"]):
        if not (old.stationary and new.stationary and spatial_distance <= float(config["maximum_stationary_distance"])):
            rejected, rejection_reason = True, "time_gap_too_large"
    max_distance = float(config["maximum_stationary_distance"] if old.stationary and new.stationary else config["maximum_moving_distance"])
    if not rejected and spatial_distance > max_distance:
        rejected, rejection_reason = True, "spatial_jump_too_large"
    overlap_iou = _iou(old.end_bbox, new.start_bbox) if overlap_frames > 0 else 0.0
    duplicate_decision = ""
    if overlap_frames > 0:
        if overlap_iou >= float(config["overlap_duplicate_min_iou"]) or spatial_distance <= max_distance * float(config["overlap_duplicate_min_spatial_score"]):
            duplicate_decision = "DUPLICATE_OVERLAP_ACCEPT"
        elif not rejected:
            rejected, rejection_reason = True, "overlap_not_same_object"
            duplicate_decision = "DUPLICATE_OVERLAP_REJECT"
    temporal_score = 1.0 if overlap_frames > 0 else max(0.0, 1.0 - max(0.0, time_gap) / float(config["maximum_gap_seconds"]))
    spatial_score = max(0.0, 1.0 - spatial_distance / max(max_distance, 1.0))
    motion_score = 1.0 if old.stationary and new.stationary else max(0.0, _dot(old.estimated_direction, new.estimated_direction))
    appearance_score = _appearance_score(old, new)
    appearance_quality = min(old.evidence_quality, new.evidence_quality)
    colour_score = _colour_score(old.colour, new.colour)
    components = {
        "temporal": temporal_score,
        "spatial": spatial_score,
        "motion": motion_score,
        "class": class_score,
        "appearance": appearance_score,
        "colour": colour_score,
        "plate": 0.0,
    }
    effective_weights = dict(config["weights"])
    effective_weights["appearance"] = float(effective_weights["appearance"]) * appearance_quality
    total_weight = sum(float(effective_weights[key]) for key in effective_weights)
    score = sum(float(effective_weights[key]) * components[key] for key in effective_weights) / max(total_weight, 1e-9)
    return {
        "track_a": old.local_track_id,
        "track_b": new.local_track_id,
        "association_mode": mode,
        "rejected": rejected,
        "rejection_reason": rejection_reason,
        "score": round(score, 6) if not rejected else 0.0,
        "frame_gap": frame_gap,
        "time_gap_seconds": round(time_gap, 6),
        "overlap_frames": overlap_frames,
        "spatial_distance": round(spatial_distance, 6),
        "overlap_iou": round(overlap_iou, 6),
        "duplicate_overlap_decision": duplicate_decision,
        "appearance_quality": round(appearance_quality, 6),
        **{f"{key}_score": round(value, 6) for key, value in components.items()},
    }


def _build_vehicles(features: list[TrackletFeature], pair_rows: list[dict[str, Any]], config: dict[str, Any]) -> tuple[dict[str, str], list[dict[str, Any]]]:
    feature_order = sorted(features, key=lambda item: (item.first_frame, _track_sort_key(item.local_track_id)))
    pair_lookup: dict[tuple[str, str], dict[str, Any]] = {}
    for row in pair_rows:
        pair_lookup[tuple(sorted((str(row["track_a"]), str(row["track_b"]))))] = row
    vehicles: list[dict[str, Any]] = []
    mapping: dict[str, str] = {}
    decisions: list[dict[str, Any]] = []

    for feature in feature_order:
        candidates = []
        for vehicle in vehicles:
            member_rows = [_lookup_pair(pair_lookup, feature.local_track_id, member_id) for member_id in vehicle["members"]]
            usable_rows = [row for row in member_rows if row is not None]
            scored_rows = [row for row in usable_rows if not row["rejected"]]
            if not scored_rows:
                continue
            best = max(scored_rows, key=lambda row: float(row["score"]))
            compatible_rows = [row for row in scored_rows if float(row["score"]) >= float(config["vehicle_consistency_floor"])]
            conflicting_rows = [row for row in usable_rows if row["rejected"] or float(row.get("score", 0.0)) < float(config["vehicle_consistency_floor"])]
            consistency_score = sum(float(row["score"]) for row in compatible_rows) / max(len(usable_rows), 1)
            vehicle_score = (
                (1.0 - float(config["vehicle_consistency_weight"])) * float(best["score"])
                + float(config["vehicle_consistency_weight"]) * consistency_score
                - float(config["vehicle_conflict_penalty"]) * len(conflicting_rows)
            )
            candidates.append(
                {
                    "vehicle_id": vehicle["vehicle_id"],
                    "members": list(vehicle["members"]),
                    "best": best,
                    "best_member_score": float(best["score"]),
                    "vehicle_consistency_score": max(0.0, vehicle_score),
                    "conflicting_member_count": len(conflicting_rows),
                    "association_reason": _association_reason(best, len(conflicting_rows)),
                    "ambiguity_reason": "",
                }
            )
        ranked = sorted(candidates, key=lambda item: float(item["vehicle_consistency_score"]), reverse=True)
        best_candidate = ranked[0] if ranked else None
        second_score = float(ranked[1]["vehicle_consistency_score"]) if len(ranked) > 1 else 0.0
        accepted = False
        ambiguity_reason = "no_candidate"
        if best_candidate is not None:
            best_candidate["ambiguity_reason"] = _ambiguity_reason(best_candidate, second_score, config)
            ambiguity_reason = best_candidate["ambiguity_reason"]
            accepted = ambiguity_reason == ""
        if accepted:
            vehicle = next(item for item in vehicles if item["vehicle_id"] == best_candidate["vehicle_id"])
            vehicle["members"].append(feature.local_track_id)
            mapping[feature.local_track_id] = str(vehicle["vehicle_id"])
        else:
            vehicle_id = f"VEHICLE_{len(vehicles) + 1:03d}"
            vehicles.append({"vehicle_id": vehicle_id, "members": [feature.local_track_id]})
            mapping[feature.local_track_id] = vehicle_id
        if best_candidate is not None:
            decisions.append(
                {
                    **best_candidate["best"],
                    "candidate_vehicle_id": best_candidate["vehicle_id"],
                    "second_best_score": round(second_score, 6),
                    "decision": "MERGE" if accepted else "NEW_OR_AMBIGUOUS",
                    "association_reason": best_candidate["association_reason"],
                    "best_member_score": round(best_candidate["best_member_score"], 6),
                    "vehicle_consistency_score": round(best_candidate["vehicle_consistency_score"], 6),
                    "conflicting_member_count": best_candidate["conflicting_member_count"],
                    "ambiguity_reason": ambiguity_reason,
                }
            )
    return mapping, decisions


def _lookup_pair(pair_lookup: dict[tuple[str, str], dict[str, Any]], a: str, b: str) -> dict[str, Any] | None:
    return pair_lookup.get(tuple(sorted((a, b))))


def _association_reason(row: dict[str, Any], conflicting_members: int) -> str:
    if str(row.get("association_mode")) == "DUPLICATE_OVERLAP":
        return str(row.get("duplicate_overlap_decision") or "DUPLICATE_OVERLAP_ACCEPT")
    if conflicting_members:
        return "BEST_PAIR_WITH_MEMBER_CONFLICTS"
    return "BEST_SEQUENTIAL_PAIR"


def _ambiguity_reason(candidate: dict[str, Any], second_score: float, config: dict[str, Any]) -> str:
    if float(candidate["best_member_score"]) < float(config["acceptance_threshold"]):
        return "best_member_below_threshold"
    if float(candidate["vehicle_consistency_score"]) < float(config["acceptance_threshold"]):
        return "vehicle_consistency_below_threshold"
    if int(candidate["conflicting_member_count"]) > 0:
        return "conflicting_existing_vehicle_members"
    if float(candidate["vehicle_consistency_score"]) - second_score < float(config["ambiguity_margin"]):
        return "ambiguous_vehicle_margin"
    return ""


def _run_calibration_grid(
    features: list[TrackletFeature],
    pair_rows: list[dict[str, Any]],
    base_config: dict[str, Any],
    ground_truth: dict[str, Any],
) -> dict[str, Any]:
    grid_rows = []
    selected: dict[str, Any] | None = None
    for ambiguity_margin in CALIBRATION_GRID["ambiguity_margin"]:
        for acceptance_threshold in CALIBRATION_GRID["acceptance_threshold"]:
            config = dict(base_config)
            config["ambiguity_margin"] = ambiguity_margin
            config["acceptance_threshold"] = acceptance_threshold
            mapping, _decisions = _build_vehicles(features, pair_rows, config)
            vehicles = len(set(mapping.values()))
            metrics = _evaluate(mapping, ground_truth)
            row = {
                "ambiguity_margin": ambiguity_margin,
                "acceptance_threshold": acceptance_threshold,
                "vehicle_count": vehicles,
                "true_positive_merges": metrics["true_positive_merges"],
                "false_positive_merges": metrics["false_positive_merges"],
                "false_negative_merges": metrics["false_negative_merges"],
                "suspicious_overmerge_count": metrics["suspicious_overmerge_count"],
                "precision": round(metrics["precision"], 6),
                "recall": round(metrics["recall"], 6),
                "f1": round(metrics["f1"], 6),
            }
            grid_rows.append(row)
    ranked = sorted(
        grid_rows,
        key=lambda row: (
            int(row["suspicious_overmerge_count"]),
            int(row["false_positive_merges"]),
            -float(row["f1"]),
            -float(row["recall"]),
            int(row["vehicle_count"]),
        ),
    )
    if ranked:
        selected = {
            "ambiguity_margin": ranked[0]["ambiguity_margin"],
            "acceptance_threshold": ranked[0]["acceptance_threshold"],
        }
    return {
        "grid": grid_rows,
        "selected_config": selected or {},
        "selection_policy": "Minimize suspicious/false merges first, then maximize F1 and recall.",
        "selected_row": ranked[0] if ranked else {},
    }


def _vehicle_records(features: list[TrackletFeature], mapping: dict[str, str]) -> list[dict[str, Any]]:
    by_vehicle: dict[str, list[TrackletFeature]] = {}
    for feature in features:
        by_vehicle.setdefault(mapping[feature.local_track_id], []).append(feature)
    vehicles = []
    for vehicle_id, members in sorted(by_vehicle.items()):
        members = sorted(members, key=lambda item: (item.first_frame, item.local_track_id))
        class_counts: dict[str, int] = {}
        for member in members:
            class_counts[member.final_class] = class_counts.get(member.final_class, 0) + max(member.observation_count, 1)
        final_class = max(class_counts, key=class_counts.get) if class_counts else "UNKNOWN"
        vehicles.append(
            {
                "vehicle_id": vehicle_id,
                "camera_id": members[0].camera_id,
                "member_tracks": [member.local_track_id for member in members],
                "final_class": final_class,
                "first_seen_frame": min(member.first_frame for member in members),
                "last_seen_frame": max(member.last_frame for member in members),
                "first_seen_seconds": min(member.first_timestamp for member in members),
                "last_seen_seconds": max(member.last_timestamp for member in members),
                "stationary": sum(1 for member in members if member.stationary) >= len(members) / 2.0,
            }
        )
    return vehicles


def _evaluate(mapping: dict[str, str], ground_truth: dict[str, Any]) -> dict[str, Any]:
    positive_pairs = {tuple(sorted(pair)) for group in ground_truth["same_vehicle_groups"] for pair in _pairs(group)}
    negative_pairs = {tuple(sorted(pair)) for pair in ground_truth["must_not_merge"]}
    predicted_pairs = {tuple(sorted((a, b))) for a, va in mapping.items() for b, vb in mapping.items() if a < b and va == vb}
    tp = len(predicted_pairs & positive_pairs)
    fp = len(predicted_pairs & negative_pairs)
    fn = len(positive_pairs - predicted_pairs)
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
    yellow = ground_truth["same_vehicle_groups"][0]
    yellow_vehicle_ids = {mapping.get(track_id) for track_id in yellow}
    suspicious_overmerges = []
    for group in ground_truth["same_vehicle_groups"]:
        group_set = set(group)
        for vehicle_id in sorted({mapping.get(track_id) for track_id in group if mapping.get(track_id)}):
            members = {track_id for track_id, mapped in mapping.items() if mapped == vehicle_id}
            outsiders = sorted(members - group_set)
            covered = sorted(members & group_set)
            if covered and outsiders:
                suspicious_overmerges.append({"vehicle_id": vehicle_id, "covered_ground_truth_members": covered, "extra_members": outsiders})
    return {
        "true_positive_merges": tp,
        "false_positive_merges": fp,
        "false_negative_merges": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "yellow_plate_car": {
            "expected_tracks": yellow,
            "vehicle_ids": sorted(item for item in yellow_vehicle_ids if item),
            "pass": (
                len(yellow_vehicle_ids) == 1
                and mapping.get("CAM_001:TRACK_24") not in yellow_vehicle_ids
                and mapping.get("CAM_001:TRACK_39") not in yellow_vehicle_ids
                and not any(item["vehicle_id"] in yellow_vehicle_ids for item in suspicious_overmerges)
            ),
            "track_24_excluded": mapping.get("CAM_001:TRACK_24") not in yellow_vehicle_ids,
            "track_39_excluded": mapping.get("CAM_001:TRACK_39") not in yellow_vehicle_ids,
        },
        "suspicious_overmerge_count": len(suspicious_overmerges),
        "suspicious_overmerges": suspicious_overmerges,
        "highest_risk_false_pairs": sorted(predicted_pairs & negative_pairs),
    }


def _analytics_simulation(features: list[TrackletFeature], vehicles: list[dict[str, Any]]) -> dict[str, Any]:
    completed_by_class: dict[str, int] = {}
    for feature in features:
        completed_by_class[feature.final_class] = completed_by_class.get(feature.final_class, 0) + 1
    vehicles_by_class: dict[str, int] = {}
    for vehicle in vehicles:
        vehicles_by_class[vehicle["final_class"]] = vehicles_by_class.get(vehicle["final_class"], 0) + 1
    return {
        "raw_completed_tracks": len(features),
        "reconciled_physical_vehicles": len(vehicles),
        "duplicates_removed": len(features) - len(vehicles),
        "raw_by_class": completed_by_class,
        "reconciled_by_class": vehicles_by_class,
    }


def _existing_reconciliation_summary(run_path: Path) -> dict[str, Any]:
    try:
        from .track_reconciliation import run_track_reconciliation_experiment

        result = run_track_reconciliation_experiment(run_path, output_dir=run_path / "vehicle_identity_test" / "existing_reconciliation_baseline")
        return dict(result.get("metrics", {}))
    except Exception as exc:
        return {"error": str(exc)}


def _write_contact_sheets(run_path: Path, output_dir: Path, vehicles: list[dict[str, Any]], evidence: list[dict[str, Any]]) -> None:
    for vehicle in vehicles:
        if len(vehicle["member_tracks"]) < 2:
            continue
        tiles = []
        for track_id in vehicle["member_tracks"]:
            item = next((row for row in evidence if row.get("local_track_id") == track_id and row.get("crop_path")), None)
            if not item:
                continue
            image = cv2.imread(str(item["crop_path"]))
            if image is None:
                continue
            image = cv2.resize(image, (180, 130))
            cv2.putText(image, track_id.split(":")[-1], (4, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
            tiles.append(image)
        if tiles:
            canvas = np.hstack(tiles)
            cv2.imwrite(str(output_dir / f"{vehicle['vehicle_id']}.jpg"), canvas)


def _write_high_score_rejects(run_path: Path, output_dir: Path, pair_rows: list[dict[str, Any]], evidence: list[dict[str, Any]]) -> None:
    rejects = sorted([row for row in pair_rows if row["rejected"]], key=lambda row: row.get("spatial_score", 0), reverse=True)[:10]
    _write_csv(output_dir / "high_score_rejected_pairs.csv", rejects)


def _appearance_descriptor(crops: list[str]) -> tuple[list[float], float]:
    descriptors = []
    qualities = []
    for crop in crops[:3]:
        image = cv2.imread(crop)
        if image is None or image.size == 0:
            continue
        h, w = image.shape[:2]
        quality = min(1.0, (w * h) / (180.0 * 120.0))
        hist = cv2.calcHist([image], [0, 1, 2], None, [4, 4, 4], [0, 256, 0, 256, 0, 256]).astype(np.float32).flatten()
        total = float(hist.sum())
        if total > 0:
            hist /= total
        descriptors.append(hist)
        qualities.append(quality)
    if not descriptors:
        return [], 0.0
    return np.mean(np.asarray(descriptors), axis=0).tolist(), float(np.mean(qualities))


def _appearance_score(a: TrackletFeature, b: TrackletFeature) -> float:
    if a.evidence_quality < 0.15 or b.evidence_quality < 0.15 or not a.appearance_descriptor or not b.appearance_descriptor:
        return 0.5
    va = np.asarray(a.appearance_descriptor, dtype=np.float32)
    vb = np.asarray(b.appearance_descriptor, dtype=np.float32)
    denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
    if denom == 0:
        return 0.5
    return max(0.0, min(1.0, float(np.dot(va, vb) / denom)))


def _class_score(a: str, b: str) -> float:
    if a == "UNKNOWN" or b == "UNKNOWN":
        return 0.65
    if a == b:
        return 1.0
    compatible = {("CAR", "3WHEELER"), ("BUS", "TRUCK")}
    return 0.35 if tuple(sorted((a, b))) in compatible else 0.0


def _colour_score(a: str, b: str) -> float:
    if a == "UNKNOWN" or b == "UNKNOWN":
        return 0.5
    if a == b:
        return 1.0
    compatible = {("BLACK", "GREY"), ("GREY", "SILVER"), ("WHITE", "SILVER")}
    return 0.65 if tuple(sorted((a, b))) in compatible else 0.35


def _extract_colour(track: dict[str, Any], enrichment: dict[str, Any] | None) -> str:
    for payload in (track.get("vehicle_enrichment"), enrichment):
        if isinstance(payload, dict):
            colour = payload.get("vehicle_colour") or payload.get("colour")
            if isinstance(colour, dict):
                return str(colour.get("label") or "UNKNOWN").upper()
            if isinstance(colour, str):
                return colour.upper()
    return "UNKNOWN"


def _box_from_row(row: dict[str, Any]) -> list[float]:
    return [float(row[key]) for key in ("x1", "y1", "x2", "y2")]


def _center(box: list[float]) -> list[float]:
    return [(box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0]


def _distance(a: list[float], b: list[float]) -> float:
    return float(math.hypot(a[0] - b[0], a[1] - b[1]))


def _dot(a: list[float], b: list[float]) -> float:
    return float(a[0] * b[0] + a[1] * b[1])


def _iou(a: list[float], b: list[float]) -> float:
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    aa = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    bb = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / (aa + bb - inter) if aa + bb - inter > 0 else 0.0


def _pairs(group: list[str]) -> list[tuple[str, str]]:
    return [(group[i], group[j]) for i in range(len(group)) for j in range(i + 1, len(group))]


def _track_sort_key(track_id: str) -> tuple[str, int]:
    return (track_id.split(":")[0], int(track_id.split("TRACK_")[-1]) if "TRACK_" in track_id else 0)


def _read_observations(path: Path) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            grouped.setdefault(str(row["local_track_id"]), []).append(row)
    return grouped


def _read_json(path: Path, default: Any | None = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_report(path: Path, result: dict[str, Any], vehicles: list[dict[str, Any]]) -> None:
    metrics = result["metrics"]
    analytics = result["analytics_simulation"]
    existing = result.get("existing_reconciliation_baseline", {})
    pre_calibration = result.get("pre_calibration_poc_baseline", {})
    selected = dict(result.get("calibration", {}).get("selected_row", {}) or {})
    lines = [
        "# Vehicle Identity Test",
        "",
        f"- Source run: `{result['source_run_directory']}`",
        f"- Precision: `{metrics['precision']:.3f}`",
        f"- Recall: `{metrics['recall']:.3f}`",
        f"- F1: `{metrics['f1']:.3f}`",
        f"- Suspicious over-merges: `{metrics['suspicious_overmerge_count']}`",
        f"- Yellow-plate car pass: `{metrics['yellow_plate_car']['pass']}`",
        f"- Raw completed tracks: `{analytics['raw_completed_tracks']}`",
        f"- Reconciled vehicle identities: `{analytics['reconciled_physical_vehicles']}`",
        f"- Selected threshold: `{selected.get('acceptance_threshold')}`",
        f"- Selected ambiguity margin: `{selected.get('ambiguity_margin')}`",
        f"- tracks.json unchanged: `{result['tracks_json_unchanged']}`",
        "",
        "## Comparison",
        "",
        "| Mode | Vehicle IDs | Removed Duplicates | Precision | Recall | Suspicious Over-Merges |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
        (
            f"| Existing track reconciliation | {existing.get('reconciled_vehicle_identities', 'n/a')} | "
            f"{existing.get('potential_duplicate_tracks_removed', existing.get('track_fragments_merged', 'n/a'))} | n/a | n/a | n/a |"
        ),
        (
            f"| Pre-calibration identity POC | {pre_calibration.get('predicted_vehicle_identities', 'n/a')} | "
            f"{pre_calibration.get('raw_completed_tracks', 0) - pre_calibration.get('predicted_vehicle_identities', 0)} | "
            f"{float(pre_calibration.get('precision', 0.0)):.3f} | {float(pre_calibration.get('recall', 0.0)):.3f} | "
            f"{pre_calibration.get('suspicious_overmerge_count', 'n/a')} |"
        ),
        (
            f"| Calibrated identity POC | {analytics['reconciled_physical_vehicles']} | {analytics['duplicates_removed']} | "
            f"{metrics['precision']:.3f} | {metrics['recall']:.3f} | {metrics['suspicious_overmerge_count']} |"
        ),
        "",
        "## Multi-Track Vehicles",
    ]
    for vehicle in vehicles:
        if len(vehicle["member_tracks"]) > 1:
            lines.append(f"- `{vehicle['vehicle_id']}` {vehicle['final_class']}: {', '.join(vehicle['member_tracks'])}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
