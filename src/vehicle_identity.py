from __future__ import annotations

import csv
import json
import math
import re
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import yaml

from .vehicle_identity_experiment import (
    DEFAULT_CONFIG as CONSERVATIVE_DEFAULT_CONFIG,
    TrackletFeature,
    _build_feature,
    _build_vehicles,
    _read_observations,
    _score_pair,
    _track_sort_key,
)


DEFAULT_VEHICLE_IDENTITY_CONFIG = {
    "enabled": True,
    "output": {
        "physical_vehicles": "physical_vehicles.json",
        "vehicle_identity_map": "vehicle_identity_map.json",
        "identity_decisions": "identity_decisions.json",
    },
    "conservative": {
        "enabled": True,
        "acceptance_threshold": 0.70,
        "ambiguity_margin": 0.03,
        "vehicle_consistency_floor": 0.58,
    },
    "plate_assistance": {
        "enabled": True,
        "require_high_quality_for_exact_override": True,
        "contradiction_veto": True,
        "plate_weight": 0.26,
        "exact_match_bonus": 0.34,
        "partial_match_bonus": 0.18,
        "contradiction_penalty": 0.40,
        "exact_match_override_threshold": 0.64,
        "partial_match_threshold": 0.86,
        "clear_contradiction_literal_threshold": 0.62,
        "clear_contradiction_confusion_threshold": 0.72,
        "high_score_threshold": 0.72,
        "medium_score_threshold": 0.55,
        "minimum_detector_confidence_high": 0.70,
        "minimum_ocr_confidence_high": 0.70,
        "minimum_text_length": 6,
        "minimum_crop_width": 40,
        "minimum_crop_height": 16,
    },
    "stationary_recovery": {
        "enabled": False,
    },
}


@dataclass(frozen=True, slots=True)
class PhysicalIdentityResult:
    run_id: str
    physical_vehicles: list[dict[str, Any]]
    vehicle_identity_map: dict[str, str]
    identity_decisions: list[dict[str, Any]]
    metrics: dict[str, Any]
    config: dict[str, Any]
    paths: dict[str, str]


@dataclass(slots=True)
class PlateConsensus:
    local_track_id: str
    plate_detected: bool
    ocr_attempted: bool
    raw_plate_text: str | None
    normalized_plate_text: str | None
    plate_detection_confidence: float | None
    plate_text_confidence: float | None
    plate_crop_path: str | None
    plate_bbox: list[Any] | None
    plate_ocr_raw_response: str | None
    plate_ocr_reason: str | None
    plate_quality_status: str | None
    reliability_score: float
    reliability_label: str


def normalize_vehicle_identity_config(raw_config: Any) -> dict[str, Any]:
    raw = dict(raw_config or {}) if isinstance(raw_config, dict) else {}
    normalized = _deep_merge(DEFAULT_VEHICLE_IDENTITY_CONFIG, raw)
    conservative = dict(normalized.get("conservative", {}) or {})
    plate = dict(normalized.get("plate_assistance", {}) or {})
    stationary = dict(normalized.get("stationary_recovery", {}) or {})
    output = dict(normalized.get("output", {}) or {})
    normalized["enabled"] = bool(normalized.get("enabled", True))
    conservative["enabled"] = bool(conservative.get("enabled", True))
    conservative["acceptance_threshold"] = float(conservative.get("acceptance_threshold", 0.70))
    conservative["ambiguity_margin"] = float(conservative.get("ambiguity_margin", 0.03))
    conservative["vehicle_consistency_floor"] = float(conservative.get("vehicle_consistency_floor", 0.58))
    plate["enabled"] = bool(plate.get("enabled", True))
    plate["require_high_quality_for_exact_override"] = bool(plate.get("require_high_quality_for_exact_override", True))
    plate["contradiction_veto"] = bool(plate.get("contradiction_veto", True))
    stationary["enabled"] = bool(stationary.get("enabled", False))
    normalized["conservative"] = conservative
    normalized["plate_assistance"] = plate
    normalized["stationary_recovery"] = stationary
    normalized["output"] = output
    return normalized


def build_physical_vehicle_identity_for_run(run_dir: str | Path, config: dict[str, Any] | None = None) -> PhysicalIdentityResult:
    run_path = Path(run_dir).expanduser().resolve()
    identity_config = normalize_vehicle_identity_config(config)
    paths = {
        "physical_vehicles": str(run_path / str(identity_config["output"]["physical_vehicles"])),
        "vehicle_identity_map": str(run_path / str(identity_config["output"]["vehicle_identity_map"])),
        "identity_decisions": str(run_path / str(identity_config["output"]["identity_decisions"])),
    }
    if not bool(identity_config.get("enabled", True)):
        result = PhysicalIdentityResult(
            run_id=run_path.name,
            physical_vehicles=[],
            vehicle_identity_map={},
            identity_decisions=[],
            metrics={"enabled": False, "raw_completed_tracks": 0, "physical_vehicle_count": 0, "duplicates_removed": 0},
            config=identity_config,
            paths=paths,
        )
        _write_outputs(result)
        return result

    tracks = _read_json(run_path / "tracks.json", default=[])
    observations = _read_observations(run_path / "observations.csv")
    evidence = _read_json(run_path / "evidence_index.json", default=[])
    enrichment_rows = _read_json(run_path / "vehicle_enrichment.json", default=[])
    enrichment = {str(item.get("local_track_id")): item for item in enrichment_rows if isinstance(item, dict)}
    completed_tracks = [track for track in tracks if isinstance(track, dict) and str(track.get("status", "")).upper() == "COMPLETED"]
    conservative_config = _conservative_config(identity_config)
    features = [
        _build_feature(track, observations.get(str(track.get("local_track_id")), []), evidence, enrichment, conservative_config)
        for track in completed_tracks
    ]
    features = [item for item in features if item is not None]
    base_pair_rows = _score_pairs(features, conservative_config)
    pair_rows = list(base_pair_rows)
    plate_consensus = _plate_consensus_for_tracks(completed_tracks, enrichment, identity_config)
    if bool(identity_config["plate_assistance"]["enabled"]):
        pair_rows = [_apply_plate_to_pair(row, plate_consensus, identity_config) for row in pair_rows]
        conservative_config["weights"] = dict(conservative_config["weights"])
        conservative_config["weights"]["plate"] = float(identity_config["plate_assistance"].get("plate_weight", 0.26))
    mapping, decisions = _build_vehicles(features, pair_rows, conservative_config)
    decisions = [_normalize_decision(row, plate_consensus) for row in decisions]
    vehicles = _physical_vehicle_records(
        run_id=run_path.name,
        features=features,
        mapping=mapping,
        plate_consensus=plate_consensus,
        decisions=decisions,
    )
    metrics = _identity_metrics(
        completed_tracks=completed_tracks,
        vehicles=vehicles,
        decisions=decisions,
        plate_consensus=plate_consensus,
        stationary_enabled=bool(identity_config["stationary_recovery"]["enabled"]),
    )
    result = PhysicalIdentityResult(
        run_id=run_path.name,
        physical_vehicles=vehicles,
        vehicle_identity_map=mapping,
        identity_decisions=decisions,
        metrics=metrics,
        config=identity_config,
        paths=paths,
    )
    _write_outputs(result)
    return result


def load_physical_vehicle_identity(run_dir: str | Path) -> dict[str, Any] | None:
    run_path = Path(run_dir).expanduser().resolve()
    vehicles = _read_json(run_path / "physical_vehicles.json", default=None)
    identity_map = _read_json(run_path / "vehicle_identity_map.json", default=None)
    decisions = _read_json(run_path / "identity_decisions.json", default=None)
    if not isinstance(vehicles, dict) or not isinstance(identity_map, dict) or not isinstance(decisions, list):
        return None
    return {
        "run_id": run_path.name,
        "available": True,
        "production": True,
        "physical_vehicles": list(vehicles.get("physical_vehicles", []) or []),
        "vehicle_identity_map": identity_map,
        "identity_decisions": decisions,
        "metrics": dict(vehicles.get("metrics", {}) or {}),
        "config": dict(vehicles.get("config", {}) or {}),
        "paths": {
            "physical_vehicles": str(run_path / "physical_vehicles.json"),
            "vehicle_identity_map": str(run_path / "vehicle_identity_map.json"),
            "identity_decisions": str(run_path / "identity_decisions.json"),
        },
    }


def normalize_plate_text(value: Any) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(value or "").upper())


def _conservative_config(identity_config: dict[str, Any]) -> dict[str, Any]:
    config = dict(CONSERVATIVE_DEFAULT_CONFIG)
    config["weights"] = dict(config["weights"])
    section = dict(identity_config.get("conservative", {}) or {})
    config["acceptance_threshold"] = float(section.get("acceptance_threshold", 0.85))
    config["ambiguity_margin"] = float(section.get("ambiguity_margin", 0.15))
    return config


def _score_pairs(features: list[TrackletFeature], config: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for old in features:
        for new in features:
            if old.local_track_id >= new.local_track_id:
                continue
            rows.append(_score_pair(old, new, config))
    return rows


def _plate_consensus_for_tracks(
    tracks: list[dict[str, Any]],
    enrichment: dict[str, dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, PlateConsensus]:
    return {
        str(track.get("local_track_id")): _build_plate_consensus(str(track.get("local_track_id")), enrichment.get(str(track.get("local_track_id")), {}), config)
        for track in sorted(tracks, key=lambda item: _track_sort_key(str(item.get("local_track_id"))))
        if track.get("local_track_id")
    }


def _build_plate_consensus(track_id: str, enrichment: dict[str, Any], config: dict[str, Any]) -> PlateConsensus:
    raw_text = enrichment.get("plate_text")
    normalized = normalize_plate_text(raw_text)
    detected = bool(enrichment.get("plate_detected"))
    attempted = bool(enrichment.get("plate_ocr_attempted"))
    det_conf = _float_or_none(enrichment.get("plate_detection_confidence"))
    ocr_conf = _float_or_none(enrichment.get("plate_text_confidence"))
    reliability_score, reliability_label = _plate_reliability(
        normalized=normalized,
        detected=detected,
        det_conf=det_conf,
        ocr_conf=ocr_conf,
        quality_status=str(enrichment.get("plate_quality_status") or ""),
        reason=str(enrichment.get("plate_ocr_reason") or ""),
        config=dict(config.get("plate_assistance", {}) or {}),
    )
    return PlateConsensus(
        local_track_id=track_id,
        plate_detected=detected,
        ocr_attempted=attempted,
        raw_plate_text=str(raw_text) if raw_text else None,
        normalized_plate_text=normalized or None,
        plate_detection_confidence=det_conf,
        plate_text_confidence=ocr_conf,
        plate_crop_path=str(enrichment.get("plate_crop_path") or "") or None,
        plate_bbox=enrichment.get("plate_bbox"),
        plate_ocr_raw_response=enrichment.get("plate_ocr_raw_response"),
        plate_ocr_reason=enrichment.get("plate_ocr_reason"),
        plate_quality_status=enrichment.get("plate_quality_status"),
        reliability_score=round(reliability_score, 6),
        reliability_label=reliability_label,
    )


def _plate_reliability(
    *,
    normalized: str,
    detected: bool,
    det_conf: float | None,
    ocr_conf: float | None,
    quality_status: str,
    reason: str,
    config: dict[str, Any],
) -> tuple[float, str]:
    if not detected or not normalized or len(normalized) < int(config.get("minimum_text_length", 6)):
        return 0.0, "UNUSABLE"
    normalized_reason = str(reason or "").strip().lower()
    if normalized_reason and normalized_reason not in {"ocr_completed", "completed"} and not normalized_reason.startswith("validated_"):
        return 0.15, "LOW"
    det = float(det_conf or 0.0)
    ocr = float(ocr_conf if ocr_conf is not None else det)
    accepted_bonus = 1.0 if quality_status == "plate_quality_accepted" else 0.35
    score = 0.42 * det + 0.42 * ocr + 0.16 * accepted_bonus
    if score >= float(config.get("high_score_threshold", 0.72)) and det >= float(config.get("minimum_detector_confidence_high", 0.70)) and ocr >= float(config.get("minimum_ocr_confidence_high", 0.70)):
        return score, "HIGH"
    if score >= float(config.get("medium_score_threshold", 0.55)):
        return score, "MEDIUM"
    return score, "LOW"


def _apply_plate_to_pair(row: dict[str, Any], consensus: dict[str, PlateConsensus], config: dict[str, Any]) -> dict[str, Any]:
    plate_config = dict(config.get("plate_assistance", {}) or {})
    out = dict(row)
    a = consensus.get(str(row.get("track_a")))
    b = consensus.get(str(row.get("track_b")))
    relation = _plate_relation(a, b, plate_config)
    base_score = float(out.get("score", 0.0) or 0.0)
    evidence = relation["plate_evidence"]
    blocked_override = _plate_override_blocked(out)
    if evidence == "STRONG_POSITIVE" and blocked_override:
        relation = dict(relation)
        relation["plate_evidence"] = "GEOMETRY_BLOCKED_PLATE_MATCH"
        relation["plate_contribution"] = 0.0
        relation["plate_reason_code"] = "GEOMETRY_BLOCKED_PLATE_MATCH"
    if evidence == "STRONG_NEGATIVE" and bool(plate_config.get("contradiction_veto", True)):
        out["rejected"] = True
        out["rejection_reason"] = "REJECTED_BY_PLATE_CONTRADICTION"
        out["score"] = 0.0
    elif evidence == "STRONG_POSITIVE" and not blocked_override:
        out["rejected"] = False
        out["rejection_reason"] = ""
        out["score"] = round(max(base_score, float(plate_config.get("exact_match_override_threshold", 0.64))) + float(plate_config.get("exact_match_bonus", 0.34)), 6)
    elif evidence == "PARTIAL_POSITIVE" and not _bool(out.get("rejected")):
        out["score"] = round(min(1.0, base_score + float(plate_config.get("partial_match_bonus", 0.18))), 6)
    elif evidence == "WEAK_NEGATIVE" and not _bool(out.get("rejected")):
        out["score"] = round(max(0.0, base_score - float(plate_config.get("contradiction_penalty", 0.40)) / 2.0), 6)
    else:
        out["score"] = round(base_score, 6)
    out.update(relation)
    out["track_a_plate"] = a.normalized_plate_text if a else None
    out["track_b_plate"] = b.normalized_plate_text if b else None
    out["track_a_plate_quality"] = a.reliability_label if a else "UNUSABLE"
    out["track_b_plate_quality"] = b.reliability_label if b else "UNUSABLE"
    out["plate_score"] = round(max(0.0, min(1.0, (float(relation["plate_contribution"]) + 1.0) / 2.0)), 6)
    return out


def _plate_override_blocked(row: dict[str, Any]) -> bool:
    if bool(row.get("impossible_geometry")):
        return True
    if not _bool(row.get("rejected")):
        return False
    reason = str(row.get("rejection_reason") or "")
    return reason in {
        "different_camera",
        "reliable_class_conflict",
        "simultaneous_occupancy_conflict",
        "overlap_not_same_object",
    }


def _plate_relation(a: PlateConsensus | None, b: PlateConsensus | None, config: dict[str, Any]) -> dict[str, Any]:
    a_text = a.normalized_plate_text if a else None
    b_text = b.normalized_plate_text if b else None
    if not a_text or not b_text:
        return _relation("NEUTRAL", 0.0, "PLATE_MISSING", a_text, b_text)
    literal = _string_similarity(a_text, b_text)
    confusion = _confusion_similarity(a_text, b_text)
    edit = _edit_distance(a_text, b_text)
    high = a.reliability_label == "HIGH" and b.reliability_label == "HIGH"
    medium_or_high = a.reliability_label in {"HIGH", "MEDIUM"} and b.reliability_label in {"HIGH", "MEDIUM"}
    if high and a_text == b_text:
        return _relation("STRONG_POSITIVE", 1.0, "PLATE_EXACT_MATCH", a_text, b_text)
    if high and literal < float(config.get("clear_contradiction_literal_threshold", 0.62)) and confusion < float(config.get("clear_contradiction_confusion_threshold", 0.72)):
        return _relation("STRONG_NEGATIVE", -1.0, "PLATE_CONTRADICTION", a_text, b_text)
    if medium_or_high and (edit <= 1 or confusion >= float(config.get("partial_match_threshold", 0.86))):
        return _relation("PARTIAL_POSITIVE", 0.72, "PLATE_PARTIAL_MATCH", a_text, b_text)
    if high:
        return _relation("WEAK_NEGATIVE", -0.25, "PLATE_LOW_SIMILARITY", a_text, b_text)
    return _relation("NEUTRAL", 0.0, "PLATE_LOW_CONFIDENCE", a_text, b_text)


def _relation(evidence: str, contribution: float, reason: str, a_text: str | None, b_text: str | None) -> dict[str, Any]:
    return {
        "plate_evidence": evidence,
        "plate_contribution": contribution,
        "plate_reason_code": reason,
        "literal_similarity": round(_string_similarity(a_text or "", b_text or ""), 6) if a_text and b_text else 0.0,
        "confusion_similarity": round(_confusion_similarity(a_text or "", b_text or ""), 6) if a_text and b_text else 0.0,
        "edit_distance": _edit_distance(a_text or "", b_text or "") if a_text and b_text else None,
    }


def _normalize_decision(row: dict[str, Any], consensus: dict[str, PlateConsensus]) -> dict[str, Any]:
    decision = dict(row)
    decision["source_track_id"] = str(decision.get("track_a") or "")
    decision["target_track_id"] = str(decision.get("track_b") or "")
    decision["decision"] = str(decision.get("decision") or "NEW_OR_AMBIGUOUS")
    reason_parts = [
        str(decision.get("plate_reason_code") or ""),
        str(decision.get("association_reason") or ""),
        str(decision.get("rejection_reason") or ""),
        str(decision.get("ambiguity_reason") or ""),
    ]
    decision["reason"] = " | ".join(part for part in reason_parts if part)
    decision["identity_method"] = "plate_assisted" if decision.get("plate_reason_code") else "conservative"
    decision["final_score"] = _float_or_none(decision.get("score"))
    for key in ("temporal_score", "spatial_score", "motion_score", "appearance_score", "colour_score", "plate_score"):
        decision[key] = _float_or_none(decision.get(key))
    return decision


def _physical_vehicle_records(
    *,
    run_id: str,
    features: list[TrackletFeature],
    mapping: dict[str, str],
    plate_consensus: dict[str, PlateConsensus],
    decisions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_vehicle: dict[str, list[TrackletFeature]] = {}
    for feature in features:
        by_vehicle.setdefault(mapping[feature.local_track_id], []).append(feature)
    decision_lookup = {
        tuple(sorted((str(row.get("source_track_id")), str(row.get("target_track_id"))))): row
        for row in decisions
        if row.get("source_track_id") and row.get("target_track_id")
    }
    vehicles = []
    for vehicle_id, members in sorted(by_vehicle.items()):
        members = sorted(members, key=lambda item: (item.first_frame, item.local_track_id))
        member_ids = [member.local_track_id for member in members]
        plates = [plate_consensus.get(track_id) for track_id in member_ids if plate_consensus.get(track_id)]
        high_plates = [item for item in plates if item and item.reliability_label == "HIGH" and item.normalized_plate_text]
        class_counts: dict[str, int] = {}
        colour_counts: dict[str, int] = {}
        for member in members:
            class_counts[member.final_class] = class_counts.get(member.final_class, 0) + max(member.observation_count, 1)
            colour_counts[member.colour] = colour_counts.get(member.colour, 0) + max(member.observation_count, 1)
        merge_decisions = [
            decision_lookup[pair]
            for pair in _pairs(member_ids)
            if pair in decision_lookup and str(decision_lookup[pair].get("decision")) == "MERGE"
        ]
        vehicle = {
            "vehicle_id": vehicle_id,
            "vehicle_key": vehicle_id,
            "run_id": run_id,
            "camera_ids": sorted({member.camera_id for member in members}),
            "primary_camera_id": members[0].camera_id,
            "member_track_ids": member_ids,
            "member_track_count": len(member_ids),
            "vehicle_class": _top_value(class_counts),
            "vehicle_colour": _top_value(colour_counts),
            "first_seen_seconds": min(member.first_timestamp for member in members),
            "last_seen_seconds": max(member.last_timestamp for member in members),
            "first_frame": min(member.first_frame for member in members),
            "last_frame": max(member.last_frame for member in members),
            "identity_confidence": round(max([float(row.get("final_score") or 0.0) for row in merge_decisions] or [1.0]), 6),
            "identity_method": "plate_assisted" if any(row.get("plate_reason_code") for row in merge_decisions) else ("single_track" if len(members) == 1 else "conservative"),
            "identity_status": "MERGED" if len(members) > 1 else "SINGLE_TRACK",
            "consensus_plate_text": _top_value({str(item.normalized_plate_text): 1 for item in high_plates if item and item.normalized_plate_text}),
            "plate_confidence": round(max([float(item.plate_text_confidence or item.plate_detection_confidence or 0.0) for item in high_plates] or [0.0]), 6),
            "plate_quality": "HIGH" if high_plates else _best_quality([item.reliability_label for item in plates if item]),
            "plate_evidence": [asdict(item) for item in plates],
            "representative_evidence": _representative_evidence(members, plate_consensus),
            "association_decisions": merge_decisions,
        }
        vehicles.append(vehicle)
    return vehicles


def _identity_metrics(
    *,
    completed_tracks: list[dict[str, Any]],
    vehicles: list[dict[str, Any]],
    decisions: list[dict[str, Any]],
    plate_consensus: dict[str, PlateConsensus],
    stationary_enabled: bool,
) -> dict[str, Any]:
    return {
        "enabled": True,
        "raw_completed_tracks": len(completed_tracks),
        "physical_vehicle_count": len(vehicles),
        "duplicates_removed": len(completed_tracks) - len(vehicles),
        "multi_track_physical_vehicles": sum(1 for item in vehicles if int(item.get("member_track_count", 0) or 0) > 1),
        "merge_decisions": sum(1 for row in decisions if str(row.get("decision")) == "MERGE"),
        "plate_exact_merges": sum(1 for row in decisions if str(row.get("decision")) == "MERGE" and row.get("plate_reason_code") == "PLATE_EXACT_MATCH"),
        "plate_contradiction_rejections": sum(1 for row in decisions if row.get("plate_reason_code") == "PLATE_CONTRADICTION"),
        "tracks_with_plate_detected": sum(1 for item in plate_consensus.values() if item.plate_detected),
        "tracks_with_readable_plate": sum(1 for item in plate_consensus.values() if item.normalized_plate_text),
        "tracks_with_high_quality_plate": sum(1 for item in plate_consensus.values() if item.reliability_label == "HIGH"),
        "stationary_recovery_enabled": stationary_enabled,
    }


def _write_outputs(result: PhysicalIdentityResult) -> None:
    Path(result.paths["physical_vehicles"]).write_text(
        json.dumps(
            {
                "run_id": result.run_id,
                "production": True,
                "physical_vehicles": result.physical_vehicles,
                "metrics": result.metrics,
                "config": result.config,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    Path(result.paths["vehicle_identity_map"]).write_text(json.dumps(result.vehicle_identity_map, indent=2), encoding="utf-8")
    Path(result.paths["identity_decisions"]).write_text(json.dumps(result.identity_decisions, indent=2), encoding="utf-8")


def _read_json(path: Path, *, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _string_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def _confusion_similarity(a: str, b: str) -> float:
    normalize = str.maketrans({"0": "O", "1": "I", "5": "S", "8": "B"})
    return _string_similarity(a.translate(normalize), b.translate(normalize))


def _edit_distance(a: str, b: str) -> int:
    if not a:
        return len(b)
    if not b:
        return len(a)
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            current.append(min(current[j - 1] + 1, previous[j] + 1, previous[j - 1] + (0 if ca == cb else 1)))
        previous = current
    return previous[-1]


def _pairs(items: list[str]) -> list[tuple[str, str]]:
    return [tuple(sorted((items[i], items[j]))) for i in range(len(items)) for j in range(i + 1, len(items))]


def _representative_evidence(members: list[TrackletFeature], plate_consensus: dict[str, PlateConsensus]) -> list[dict[str, Any]]:
    rows = []
    for member in members:
        plate = plate_consensus.get(member.local_track_id)
        rows.append(
            {
                "local_track_id": member.local_track_id,
                "camera_id": member.camera_id,
                "vehicle_crop_path": member.best_evidence_crops[0] if member.best_evidence_crops else None,
                "plate_crop_path": plate.plate_crop_path if plate else None,
                "plate_text": plate.normalized_plate_text if plate else None,
                "plate_confidence": plate.plate_text_confidence if plate else None,
                "plate_quality": plate.reliability_label if plate else "UNUSABLE",
            }
        )
    return rows


def _top_value(counts: dict[str, int]) -> str | None:
    if not counts:
        return None
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]


def _best_quality(values: list[str]) -> str:
    order = {"HIGH": 3, "MEDIUM": 2, "LOW": 1, "UNUSABLE": 0}
    return max(values or ["UNUSABLE"], key=lambda item: order.get(str(item), 0))


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes"}


def _float_or_none(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        if isinstance(value, float) and math.isnan(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(dict(merged[key]), value)
        else:
            merged[key] = value
    return merged
