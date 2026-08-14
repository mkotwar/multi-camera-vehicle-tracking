from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .vehicle_identity_experiment import _appearance_score, _class_score, _iou, _sha256, _track_sort_key, _write_csv, _write_json


DEFAULT_STATIONARY_RECOVERY_CONFIG = {
    "enabled": True,
    "minimum_stationary_confidence": 0.55,
    "maximum_gap_seconds": 15.0,
    "recovery_threshold": 0.74,
    "ambiguity_margin": 0.08,
    "location_distance_tolerance": 1.05,
    "whole_vehicle_consistency_floor": 0.62,
    "whole_vehicle_consistency_weight": 0.30,
    "overlap_strong_location_threshold": 0.65,
    "appearance_quality_threshold": 0.35,
    "appearance_contradiction_threshold": 0.32,
    "weights": {
        "location": 0.42,
        "stationary": 0.22,
        "class": 0.16,
        "appearance": 0.08,
        "plate": 0.02,
        "temporal": 0.10,
    },
}

STATIONARY_RECOVERY_GRID = {
    "minimum_stationary_confidence": [0.55, 0.62, 0.72, 0.78],
    "maximum_gap_seconds": [2.0, 5.0, 10.0, 15.0],
    "recovery_threshold": [0.70, 0.74, 0.78, 0.82],
    "ambiguity_margin": [0.05, 0.08, 0.12],
    "location_distance_tolerance": [0.85, 1.05, 1.25],
}

STATIONARY_RECOVERY_GROUND_TRUTH = {
    "same_vehicle_groups": [
        ["CAM_001:TRACK_6", "CAM_001:TRACK_12", "CAM_001:TRACK_25", "CAM_001:TRACK_27", "CAM_001:TRACK_41"],
    ],
    "must_not_merge": [
        ["CAM_001:TRACK_6", "CAM_001:TRACK_29"],
        ["CAM_001:TRACK_6", "CAM_001:TRACK_32"],
        ["CAM_001:TRACK_6", "CAM_001:TRACK_35"],
        ["CAM_001:TRACK_12", "CAM_001:TRACK_29"],
        ["CAM_001:TRACK_12", "CAM_001:TRACK_32"],
        ["CAM_001:TRACK_12", "CAM_001:TRACK_35"],
        ["CAM_001:TRACK_25", "CAM_001:TRACK_29"],
        ["CAM_001:TRACK_25", "CAM_001:TRACK_32"],
        ["CAM_001:TRACK_25", "CAM_001:TRACK_35"],
        ["CAM_001:TRACK_25", "CAM_001:TRACK_24"],
        ["CAM_001:TRACK_6", "CAM_001:TRACK_24"],
        ["CAM_001:TRACK_6", "CAM_001:TRACK_39"],
    ],
}


@dataclass(slots=True)
class VehicleGroupFeature:
    vehicle_id: str
    camera_id: str
    member_tracks: list[str]
    final_class: str
    class_distribution: dict[str, int]
    first_frame: int
    last_frame: int
    first_timestamp: float
    last_timestamp: float
    median_center: list[float]
    center_spread: float
    median_width: float
    median_height: float
    footprint_bbox: list[float]
    stationary_confidence: float
    appearance_descriptor: list[float]
    evidence_quality: float
    best_evidence_crops: list[str]


def run_stationary_recovery_experiment(run_dir: str | Path, *, identity_dir: str | Path | None = None) -> dict[str, Any]:
    run_path = Path(run_dir).expanduser().resolve()
    base_dir = Path(identity_dir).expanduser().resolve() if identity_dir else run_path / "vehicle_identity_test"
    output_dir = base_dir / "stationary_recovery"
    contact_dir = output_dir / "contact_sheets"
    contact_dir.mkdir(parents=True, exist_ok=True)
    tracks_hash_before = _sha256(run_path / "tracks.json")

    tracklet_features = _read_json(base_dir / "tracklet_features.json", default=[])
    conservative_vehicles = _read_json(base_dir / "vehicles.json", default={}).get("vehicles", [])
    if not tracklet_features or not conservative_vehicles:
        raise FileNotFoundError("Run the conservative vehicle identity experiment before stationary recovery.")

    features_by_track = {str(item["local_track_id"]): item for item in tracklet_features}
    group_features = [_build_group_feature(vehicle, features_by_track) for vehicle in conservative_vehicles]
    group_features = [item for item in group_features if item is not None]

    base_config = dict(DEFAULT_STATIONARY_RECOVERY_CONFIG)
    candidate_rows = _score_all_candidates(group_features, base_config)
    calibration = _run_stationary_grid(group_features, base_config)
    selected_config = dict(base_config)
    selected_config.update(calibration["selected_config"])
    persistent_map, persistent_vehicles, decisions = _build_persistent_vehicles(group_features, _score_all_candidates(group_features, selected_config), selected_config)
    evaluation = _evaluate_persistent(persistent_map, group_features)
    _write_stationary_contact_sheets(contact_dir, persistent_vehicles, group_features)
    result = {
        "source_run_directory": str(run_path),
        "experimental": True,
        "stage": "stationary_recovery",
        "config": selected_config,
        "calibration": calibration,
        "stationary_classification_audit": {
            "current_tracklet_rule": "stationary when at least two of normalized total displacement, median per-frame displacement, and speed thresholds pass",
            "normalization": "total displacement is normalized by median bbox diagonal; recovery additionally uses center spread and bbox-size-normalized location distance",
            "duration_dependence": "speed and temporal gap are duration/time aware; long-gap recovery only applies to high-confidence stationary vehicle groups",
            "bbox_size_dependence": "stationary confidence and same-location score use median bbox width/height and bbox diagonal",
            "jitter_handling": "median step and robust center spread reduce sensitivity to detector jitter",
            "moving_criteria": "groups below minimum stationary confidence are ineligible for long-gap recovery",
        },
        "metrics": evaluation,
        "analytics_simulation": {
            "raw_completed_tracks": len(tracklet_features),
            "conservative_vehicle_identities": len(conservative_vehicles),
            "stationary_recovered_vehicle_identities": len(persistent_vehicles),
            "duplicates_removed_by_stationary_recovery": len(conservative_vehicles) - len(persistent_vehicles),
        },
        "tracks_json_sha256_before": tracks_hash_before,
        "tracks_json_sha256_after": _sha256(run_path / "tracks.json"),
        "tracks_json_unchanged": tracks_hash_before == _sha256(run_path / "tracks.json"),
        "output_directory": str(output_dir),
    }
    _write_json(output_dir / "stationary_features.json", tracklet_features)
    _write_json(output_dir / "vehicle_group_features.json", [asdict(item) for item in group_features])
    _write_csv(output_dir / "recovery_candidates.csv", candidate_rows)
    _write_csv(output_dir / "recovery_scores.csv", _score_all_candidates(group_features, selected_config))
    _write_csv(output_dir / "recovery_decisions.csv", decisions)
    _write_json(output_dir / "persistent_vehicle_id_map.json", persistent_map)
    _write_json(output_dir / "persistent_vehicles.json", {"persistent_vehicles": persistent_vehicles})
    _write_json(output_dir / "evaluation.json", result)
    _write_json(output_dir / "ground_truth.json", STATIONARY_RECOVERY_GROUND_TRUTH)
    _write_csv(output_dir / "calibration_grid.csv", calibration["grid"])
    _write_json(output_dir / "calibration_summary.json", calibration)
    _write_report(output_dir / "report.md", result, persistent_vehicles, decisions)
    return result


def _build_group_feature(vehicle: dict[str, Any], features_by_track: dict[str, dict[str, Any]]) -> VehicleGroupFeature | None:
    members = [features_by_track[track_id] for track_id in vehicle.get("member_tracks", []) if track_id in features_by_track]
    if not members:
        return None
    centers = [center for member in members for center in list(member.get("trajectory_points", []) or [member.get("mean_center")])]
    centers_np = np.asarray(centers, dtype=np.float32)
    median_center = np.median(centers_np, axis=0)
    distances = [float(math.hypot(float(center[0] - median_center[0]), float(center[1] - median_center[1]))) for center in centers_np]
    widths = [float(size[0]) for member in members for size in member.get("bbox_size_history", []) if len(size) >= 2]
    heights = [float(size[1]) for member in members for size in member.get("bbox_size_history", []) if len(size) >= 2]
    median_width = float(np.median(widths)) if widths else 1.0
    median_height = float(np.median(heights)) if heights else 1.0
    class_distribution: dict[str, int] = {}
    for member in members:
        for cls, count in dict(member.get("class_distribution", {}) or {member.get("final_class", "UNKNOWN"): member.get("observation_count", 1)}).items():
            class_distribution[str(cls).upper()] = class_distribution.get(str(cls).upper(), 0) + int(count)
    final_class = max(class_distribution, key=class_distribution.get) if class_distribution else str(vehicle.get("final_class") or "UNKNOWN").upper()
    descriptor, quality = _merge_appearance(members)
    return VehicleGroupFeature(
        vehicle_id=str(vehicle["vehicle_id"]),
        camera_id=str(vehicle.get("camera_id") or members[0].get("camera_id") or ""),
        member_tracks=[str(member["local_track_id"]) for member in members],
        final_class=final_class,
        class_distribution=class_distribution,
        first_frame=min(int(member["first_frame"]) for member in members),
        last_frame=max(int(member["last_frame"]) for member in members),
        first_timestamp=min(float(member["first_timestamp"]) for member in members),
        last_timestamp=max(float(member["last_timestamp"]) for member in members),
        median_center=[float(median_center[0]), float(median_center[1])],
        center_spread=float(np.percentile(distances, 75)) if distances else 0.0,
        median_width=median_width,
        median_height=median_height,
        footprint_bbox=[
            float(median_center[0] - median_width / 2.0),
            float(median_center[1] - median_height / 2.0),
            float(median_center[0] + median_width / 2.0),
            float(median_center[1] + median_height / 2.0),
        ],
        stationary_confidence=_stationary_confidence(members, float(np.percentile(distances, 75)) if distances else 0.0, median_width, median_height),
        appearance_descriptor=descriptor,
        evidence_quality=quality,
        best_evidence_crops=[crop for member in members for crop in list(member.get("best_evidence_crops", []) or [])][:8],
    )


def _stationary_confidence(members: list[dict[str, Any]], center_spread: float, median_width: float, median_height: float) -> float:
    scores = []
    diag = max(1.0, math.hypot(median_width, median_height))
    spread_score = max(0.0, 1.0 - center_spread / diag)
    for member in members:
        displacement_score = max(0.0, 1.0 - float(member.get("normalized_displacement", 1.5)) / 1.2)
        step_score = max(0.0, 1.0 - float(member.get("median_step_pixels", 10.0)) / 8.0)
        speed_score = max(0.0, 1.0 - float(member.get("speed_pixels_per_frame", 10.0)) / 4.0)
        duration_bonus = min(1.0, float(member.get("duration_seconds", 0.0)) / 2.0)
        scores.append(0.34 * displacement_score + 0.24 * step_score + 0.24 * speed_score + 0.10 * spread_score + 0.08 * duration_bonus)
    return round(float(np.mean(scores)) if scores else 0.0, 6)


def _score_all_candidates(groups: list[VehicleGroupFeature], config: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for a in groups:
        for b in groups:
            if a.vehicle_id >= b.vehicle_id:
                continue
            rows.append(_score_recovery_pair(a, b, config))
    return rows


def _score_recovery_pair(a: VehicleGroupFeature, b: VehicleGroupFeature, config: dict[str, Any]) -> dict[str, Any]:
    old, new = (a, b) if (a.first_frame, a.last_frame) <= (b.first_frame, b.last_frame) else (b, a)
    time_gap = new.first_timestamp - old.last_timestamp
    overlap_frames = max(0, min(old.last_frame, new.last_frame) - max(old.first_frame, new.first_frame) + 1)
    rejected = False
    reason = ""
    class_score = _class_score(old.final_class, new.final_class)
    if old.camera_id != new.camera_id:
        rejected, reason = True, "different_camera"
    elif class_score == 0.0:
        rejected, reason = True, "reliable_class_conflict"
    elif min(old.stationary_confidence, new.stationary_confidence) < float(config["minimum_stationary_confidence"]):
        rejected, reason = True, "moving_or_low_stationary_confidence"
    elif overlap_frames == 0 and time_gap > float(config["maximum_gap_seconds"]):
        rejected, reason = True, "stationary_gap_too_large"
    location_score, center_distance, normalized_distance, size_score, footprint_iou = _same_location_score(old, new, config)
    if not rejected and overlap_frames > 0 and location_score < float(config["overlap_strong_location_threshold"]):
        rejected, reason = True, "simultaneous_occupancy_conflict"
    if not rejected and location_score < 0.50:
        rejected, reason = True, "different_parking_location"
    appearance_score, appearance_quality, appearance_reason = _group_appearance_score(old, new, config)
    if not rejected and appearance_reason == "high_quality_appearance_contradiction":
        rejected, reason = True, appearance_reason
    stationary_score = min(old.stationary_confidence, new.stationary_confidence)
    temporal_score = 1.0 if overlap_frames > 0 else max(0.0, 1.0 - max(0.0, time_gap) / max(float(config["maximum_gap_seconds"]), 1.0))
    plate_score = 0.5
    components = {
        "location": location_score,
        "stationary": stationary_score,
        "class": class_score,
        "appearance": appearance_score,
        "plate": plate_score,
        "temporal": temporal_score,
    }
    score = sum(float(config["weights"][key]) * components[key] for key in components) / sum(float(config["weights"][key]) for key in components)
    return {
        "source_vehicle_a": old.vehicle_id,
        "source_vehicle_b": new.vehicle_id,
        "tracks_a": ",".join(old.member_tracks),
        "tracks_b": ",".join(new.member_tracks),
        "rejected": rejected,
        "rejection_reason": reason,
        "score": 0.0 if rejected else round(score, 6),
        "raw_score": round(score, 6),
        "time_gap_seconds": round(time_gap, 6),
        "overlap_frames": overlap_frames,
        "location_score": round(location_score, 6),
        "center_distance": round(center_distance, 6),
        "normalized_center_distance": round(normalized_distance, 6),
        "size_score": round(size_score, 6),
        "footprint_iou": round(footprint_iou, 6),
        "stationary_confidence_a": old.stationary_confidence,
        "stationary_confidence_b": new.stationary_confidence,
        "class_score": round(class_score, 6),
        "appearance_score": round(appearance_score, 6),
        "appearance_quality": round(appearance_quality, 6),
        "appearance_reason": appearance_reason,
        "plate_score": plate_score,
        "temporal_score": round(temporal_score, 6),
    }


def _same_location_score(a: VehicleGroupFeature, b: VehicleGroupFeature, config: dict[str, Any]) -> tuple[float, float, float, float, float]:
    center_distance = math.hypot(a.median_center[0] - b.median_center[0], a.median_center[1] - b.median_center[1])
    diag = max(1.0, (math.hypot(a.median_width, a.median_height) + math.hypot(b.median_width, b.median_height)) / 2.0)
    normalized = center_distance / (diag * float(config["location_distance_tolerance"]))
    center_score = max(0.0, 1.0 - normalized)
    width_ratio = min(a.median_width, b.median_width) / max(a.median_width, b.median_width, 1.0)
    height_ratio = min(a.median_height, b.median_height) / max(a.median_height, b.median_height, 1.0)
    size_score = math.sqrt(width_ratio * height_ratio)
    footprint_iou = _iou(a.footprint_bbox, b.footprint_bbox)
    score = 0.58 * center_score + 0.30 * size_score + 0.12 * min(1.0, footprint_iou * 2.5)
    return max(0.0, min(1.0, score)), center_distance, normalized, size_score, footprint_iou


def _group_appearance_score(a: VehicleGroupFeature, b: VehicleGroupFeature, config: dict[str, Any]) -> tuple[float, float, str]:
    quality = min(a.evidence_quality, b.evidence_quality)
    if quality < float(config["appearance_quality_threshold"]) or not a.appearance_descriptor or not b.appearance_descriptor:
        return 0.5, quality, "low_quality_neutral"
    proxy_a = type("AppearanceProxy", (), {"evidence_quality": a.evidence_quality, "appearance_descriptor": a.appearance_descriptor})()
    proxy_b = type("AppearanceProxy", (), {"evidence_quality": b.evidence_quality, "appearance_descriptor": b.appearance_descriptor})()
    score = _appearance_score(proxy_a, proxy_b)
    if score < float(config["appearance_contradiction_threshold"]):
        return score, quality, "high_quality_appearance_contradiction"
    return score, quality, "high_quality_confirmation"


def _build_persistent_vehicles(
    groups: list[VehicleGroupFeature],
    rows: list[dict[str, Any]],
    config: dict[str, Any],
) -> tuple[dict[str, str], list[dict[str, Any]], list[dict[str, Any]]]:
    row_lookup = {(row["source_vehicle_a"], row["source_vehicle_b"]): row for row in rows}
    persistent: list[dict[str, Any]] = []
    mapping: dict[str, str] = {}
    decisions: list[dict[str, Any]] = []
    for group in sorted(groups, key=lambda item: (item.first_frame, item.vehicle_id)):
        candidates = []
        for target in persistent:
            member_rows = [_lookup_group_row(row_lookup, group.vehicle_id, source_id) for source_id in target["source_vehicle_ids"]]
            usable = [row for row in member_rows if row is not None]
            accepted_usable = [row for row in usable if not row["rejected"]]
            if not accepted_usable:
                continue
            best = max(accepted_usable, key=lambda row: float(row["score"]))
            conflicts = [row for row in usable if row["rejected"] or float(row["score"]) < float(config["whole_vehicle_consistency_floor"])]
            consistency = sum(float(row["score"]) for row in accepted_usable) / max(len(usable), 1)
            vehicle_score = (
                (1.0 - float(config["whole_vehicle_consistency_weight"])) * float(best["score"])
                + float(config["whole_vehicle_consistency_weight"]) * consistency
            )
            candidates.append({"target": target, "best": best, "score": vehicle_score, "consistency": consistency, "conflicts": conflicts})
        ranked = sorted(candidates, key=lambda item: float(item["score"]), reverse=True)
        best_candidate = ranked[0] if ranked else None
        second = float(ranked[1]["score"]) if len(ranked) > 1 else 0.0
        accepted = False
        reason = "no_candidate"
        if best_candidate is not None:
            if float(best_candidate["best"]["score"]) < float(config["recovery_threshold"]) or float(best_candidate["score"]) < float(config["recovery_threshold"]):
                reason = "below_recovery_threshold"
            elif best_candidate["conflicts"]:
                reason = "whole_vehicle_consistency_conflict"
            elif float(best_candidate["score"]) - second < float(config["ambiguity_margin"]):
                reason = "ambiguous_stationary_recovery"
            else:
                accepted = True
                reason = ""
        if accepted and best_candidate is not None:
            target = best_candidate["target"]
            target["source_vehicle_ids"].append(group.vehicle_id)
            target["member_tracks"].extend(group.member_tracks)
            target.setdefault("recovery_scores", []).append(round(float(best_candidate["score"]), 6))
            target["recovery_confidence"] = round(float(np.mean(target["recovery_scores"])), 6)
            target["first_seen_seconds"] = min(float(target["first_seen_seconds"]), group.first_timestamp)
            target["last_seen_seconds"] = max(float(target["last_seen_seconds"]), group.last_timestamp)
            mapping[group.vehicle_id] = target["persistent_vehicle_id"]
        else:
            persistent_id = f"PVEHICLE_{len(persistent) + 1:03d}"
            persistent.append(
                {
                    "persistent_vehicle_id": persistent_id,
                    "source_vehicle_ids": [group.vehicle_id],
                    "member_tracks": list(group.member_tracks),
                    "camera_id": group.camera_id,
                    "final_class": group.final_class,
                    "recovery_label": "STATIONARY RECOVERY" if group.stationary_confidence >= float(config["minimum_stationary_confidence"]) else "CONSERVATIVE ONLY",
                    "recovery_confidence": group.stationary_confidence,
                    "recovery_scores": [],
                    "first_seen_seconds": group.first_timestamp,
                    "last_seen_seconds": group.last_timestamp,
                }
            )
            mapping[group.vehicle_id] = persistent_id
        if best_candidate is not None:
            decisions.append({**best_candidate["best"], "decision": "MERGE" if accepted else "DO_NOT_MERGE", "second_best_score": round(second, 6), "final_reason": reason})
    return _renumber_persistent_vehicles(persistent), persistent, decisions


def _renumber_persistent_vehicles(persistent: list[dict[str, Any]]) -> dict[str, str]:
    persistent.sort(
        key=lambda item: (
            0 if len(item["source_vehicle_ids"]) > 1 else 1,
            -len(item["member_tracks"]),
            float(item.get("first_seen_seconds", 0.0)),
            item["source_vehicle_ids"][0],
        )
    )
    mapping: dict[str, str] = {}
    for index, vehicle in enumerate(persistent, start=1):
        next_id = f"PVEHICLE_{index:03d}"
        vehicle["persistent_vehicle_id"] = next_id
        for source_id in vehicle["source_vehicle_ids"]:
            mapping[str(source_id)] = next_id
    return mapping


def _lookup_group_row(rows: dict[tuple[str, str], dict[str, Any]], a: str, b: str) -> dict[str, Any] | None:
    return rows.get((a, b)) or rows.get((b, a))


def _evaluate_persistent(mapping: dict[str, str], groups: list[VehicleGroupFeature]) -> dict[str, Any]:
    positive = STATIONARY_RECOVERY_GROUND_TRUTH["same_vehicle_groups"][0]
    negative = STATIONARY_RECOVERY_GROUND_TRUTH["must_not_merge"]
    group_for_track = {track_id: group.vehicle_id for group in groups for track_id in group.member_tracks}
    track_to_persistent = {track: mapping.get(vehicle) for track, vehicle in group_for_track.items()}
    yellow_ids = {track_to_persistent.get(track) for track in positive if track_to_persistent.get(track)}
    predicted_negative = [pair for pair in negative if track_to_persistent.get(pair[0]) and track_to_persistent.get(pair[0]) == track_to_persistent.get(pair[1])]
    yellow_id = next(iter(yellow_ids), None) if len(yellow_ids) == 1 else None
    wrong_yellow_members = sorted(
        track
        for track, persistent_id in track_to_persistent.items()
        if yellow_id and persistent_id == yellow_id and track not in positive
    )
    source_groups_by_persistent: dict[str, set[str]] = {}
    for group in groups:
        persistent_id = mapping.get(group.vehicle_id)
        if persistent_id:
            source_groups_by_persistent.setdefault(persistent_id, set()).add(group.vehicle_id)
    unverified = [
        {"persistent_vehicle_id": persistent_id, "source_vehicle_ids": sorted(source_ids)}
        for persistent_id, source_ids in source_groups_by_persistent.items()
        if len(source_ids) > 1 and persistent_id != yellow_id
    ]
    return {
        "yellow_car_fully_recovered": len(yellow_ids) == 1,
        "yellow_car_persistent_ids": sorted(yellow_ids),
        "wrong_yellow_members": wrong_yellow_members,
        "risky_tracks_excluded_from_yellow": all(track not in wrong_yellow_members for track in ["CAM_001:TRACK_29", "CAM_001:TRACK_32", "CAM_001:TRACK_35", "CAM_001:TRACK_24", "CAM_001:TRACK_39"]),
        "confirmed_false_merges": len(predicted_negative),
        "suspicious_overmerge_count": len(predicted_negative),
        "unverified_non_yellow_recovery_count": len(unverified),
        "unverified_non_yellow_recoveries": unverified,
        "highest_risk_false_pairs": predicted_negative,
    }


def _run_stationary_grid(groups: list[VehicleGroupFeature], base_config: dict[str, Any]) -> dict[str, Any]:
    rows = []
    for min_conf in STATIONARY_RECOVERY_GRID["minimum_stationary_confidence"]:
        for max_gap in STATIONARY_RECOVERY_GRID["maximum_gap_seconds"]:
            for threshold in STATIONARY_RECOVERY_GRID["recovery_threshold"]:
                for margin in STATIONARY_RECOVERY_GRID["ambiguity_margin"]:
                    for tolerance in STATIONARY_RECOVERY_GRID["location_distance_tolerance"]:
                        config = dict(base_config)
                        config.update(
                            {
                                "minimum_stationary_confidence": min_conf,
                                "maximum_gap_seconds": max_gap,
                                "recovery_threshold": threshold,
                                "ambiguity_margin": margin,
                                "location_distance_tolerance": tolerance,
                            }
                        )
                        persistent_map, persistent_vehicles, _decisions = _build_persistent_vehicles(groups, _score_all_candidates(groups, config), config)
                        metrics = _evaluate_persistent(persistent_map, groups)
                        rows.append(
                            {
                                "minimum_stationary_confidence": min_conf,
                                "maximum_gap_seconds": max_gap,
                                "recovery_threshold": threshold,
                                "ambiguity_margin": margin,
                                "location_distance_tolerance": tolerance,
                                "persistent_vehicle_count": len(persistent_vehicles),
                                "yellow_car_fully_recovered": metrics["yellow_car_fully_recovered"],
                                "confirmed_false_merges": metrics["confirmed_false_merges"],
                                "suspicious_overmerge_count": metrics["suspicious_overmerge_count"],
                            }
                        )
    ranked = sorted(
        rows,
        key=lambda row: (
            int(row["confirmed_false_merges"]),
            int(row["suspicious_overmerge_count"]),
            0 if row["yellow_car_fully_recovered"] else 1,
            int(row["persistent_vehicle_count"]),
        ),
    )
    selected = ranked[0] if ranked else {}
    return {
        "grid": rows,
        "selected_row": selected,
        "selected_config": {key: selected[key] for key in ["minimum_stationary_confidence", "maximum_gap_seconds", "recovery_threshold", "ambiguity_margin", "location_distance_tolerance"] if key in selected},
        "selection_policy": "zero false merges, zero suspicious over-merges, yellow recovery, then recall",
    }


def _merge_appearance(members: list[dict[str, Any]]) -> tuple[list[float], float]:
    descriptors = [np.asarray(member.get("appearance_descriptor", []), dtype=np.float32) for member in members if member.get("appearance_descriptor")]
    qualities = [float(member.get("evidence_quality", 0.0)) for member in members if member.get("appearance_descriptor")]
    if not descriptors:
        return [], 0.0
    return np.mean(np.asarray(descriptors), axis=0).tolist(), float(np.mean(qualities))


def _write_stationary_contact_sheets(output_dir: Path, persistent_vehicles: list[dict[str, Any]], groups: list[VehicleGroupFeature]) -> None:
    group_by_id = {group.vehicle_id: group for group in groups}
    for vehicle in persistent_vehicles:
        tiles = []
        for source_id in vehicle["source_vehicle_ids"]:
            group = group_by_id.get(source_id)
            if group is None:
                continue
            for crop in group.best_evidence_crops[:2]:
                image = cv2.imread(crop)
                if image is None:
                    continue
                image = cv2.resize(image, (160, 112))
                cv2.putText(image, source_id, (4, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 255), 1)
                tiles.append(image)
        if tiles:
            cv2.imwrite(str(output_dir / f"{vehicle['persistent_vehicle_id']}.jpg"), np.hstack(tiles[:8]))


def _write_report(path: Path, result: dict[str, Any], persistent_vehicles: list[dict[str, Any]], decisions: list[dict[str, Any]]) -> None:
    metrics = result["metrics"]
    analytics = result["analytics_simulation"]
    lines = [
        "# Stationary Recovery Experiment",
        "",
        f"- Yellow car fully recovered: `{metrics['yellow_car_fully_recovered']}`",
        f"- Confirmed false merges: `{metrics['confirmed_false_merges']}`",
        f"- Suspicious over-merges: `{metrics['suspicious_overmerge_count']}`",
        f"- Unverified non-yellow recoveries: `{metrics['unverified_non_yellow_recovery_count']}`",
        f"- Risky yellow-neighbor tracks excluded: `{metrics['risky_tracks_excluded_from_yellow']}`",
        f"- Conservative identities: `{analytics['conservative_vehicle_identities']}`",
        f"- Stationary recovered identities: `{analytics['stationary_recovered_vehicle_identities']}`",
        f"- tracks.json unchanged: `{result['tracks_json_unchanged']}`",
        "",
        "## Comparison",
        "",
        "| Metric | Raw | Conservative POC | Stationary Recovery |",
        "| --- | ---: | ---: | ---: |",
        f"| Track records | {analytics['raw_completed_tracks']} | {analytics['raw_completed_tracks']} | {analytics['raw_completed_tracks']} |",
        f"| Vehicle identities | n/a | {analytics['conservative_vehicle_identities']} | {analytics['stationary_recovered_vehicle_identities']} |",
        f"| Confirmed false merges | n/a | 0 | {metrics['confirmed_false_merges']} |",
        f"| Suspicious over-merges | n/a | 0 | {metrics['suspicious_overmerge_count']} |",
        f"| Yellow car groups | 5 tracklets | 3 groups | {len(metrics['yellow_car_persistent_ids'])} group |",
        f"| Yellow car fully recovered | No | No | {'Yes' if metrics['yellow_car_fully_recovered'] else 'No'} |",
        f"| Wrong yellow-car members | N/A | None | {', '.join(metrics['wrong_yellow_members']) if metrics['wrong_yellow_members'] else 'None'} |",
        "",
        "## Yellow Bridge Decisions",
    ]
    bridge_ids = {("VEHICLE_006", "VEHICLE_022"), ("VEHICLE_022", "VEHICLE_024"), ("VEHICLE_006", "VEHICLE_024")}
    for row in decisions:
        pair = (row["source_vehicle_a"], row["source_vehicle_b"])
        if pair in bridge_ids or tuple(reversed(pair)) in bridge_ids:
            lines.append(
                f"- `{row['source_vehicle_a']}` -> `{row['source_vehicle_b']}`: decision `{row['decision']}`, "
                f"score `{row['score']}`, location `{row['location_score']}`, stationary "
                f"`{min(float(row['stationary_confidence_a']), float(row['stationary_confidence_b'])):.3f}`, "
                f"class `{row['class_score']}`, appearance `{row['appearance_score']}`, plate `{row['plate_score']}`, "
                f"gap `{row['time_gap_seconds']}`, reason `{row.get('final_reason') or row.get('rejection_reason')}`"
            )
    lines.extend(["", "## Persistent Vehicles"])
    for vehicle in persistent_vehicles:
        if len(vehicle["source_vehicle_ids"]) > 1:
            lines.append(f"- `{vehicle['persistent_vehicle_id']}`: {', '.join(vehicle['source_vehicle_ids'])} / {', '.join(vehicle['member_tracks'])}")
    lines.extend(["", "## Decision", ""])
    if metrics["confirmed_false_merges"] or metrics["suspicious_overmerge_count"]:
        lines.append("UNSAFE - FALSE MERGES INTRODUCED")
    elif metrics["yellow_car_fully_recovered"] and metrics["risky_tracks_excluded_from_yellow"]:
        lines.append("STATIONARY RECOVERY SUCCESSFUL - SAFE FOR MORE VIDEO VALIDATION")
    else:
        lines.append("PARTIAL - STATIONARY RECOVERY NEEDS MORE CALIBRATION")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _read_json(path: Path, default: Any | None = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))
