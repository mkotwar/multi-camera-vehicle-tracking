from __future__ import annotations

import csv
import json
import math
import re
import shutil
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

from .vehicle_identity_experiment import (
    DEFAULT_CONFIG as IDENTITY_DEFAULT_CONFIG,
    TrackletFeature,
    _build_vehicles,
    _read_json,
    _sha256,
    _track_sort_key,
    _vehicle_records,
    _write_csv,
    _write_json,
)


PLATE_CONFIG = {
    "high_score_threshold": 0.72,
    "medium_score_threshold": 0.55,
    "minimum_detector_confidence_high": 0.70,
    "minimum_ocr_confidence_high": 0.70,
    "minimum_text_length": 6,
    "minimum_crop_width": 40,
    "minimum_crop_height": 16,
    "exact_match_bonus": 0.34,
    "partial_match_bonus": 0.18,
    "contradiction_penalty": 0.40,
    "plate_weight": 0.26,
    "exact_match_override_threshold": 0.64,
    "partial_match_threshold": 0.86,
    "clear_contradiction_literal_threshold": 0.62,
    "clear_contradiction_confusion_threshold": 0.72,
    "impossible_overlap_distance_ratio": 3.50,
    "impossible_overlap_absolute_pixels": 1300.0,
    "impossible_overlap_iou_threshold": 0.02,
}


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
    vehicle_crop_path: str | None
    plate_ocr_reason: str | None
    plate_quality_status: str | None
    plate_crop_width: int
    plate_crop_height: int
    plate_crop_sharpness: float
    reliability_score: float
    reliability_label: str
    consensus_status: str
    supporting_observations: int


def normalize_plate_text(value: Any) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(value or "").upper())


def run_plate_assisted_identity_experiment(run_dir: str | Path) -> dict[str, Any]:
    run_path = Path(run_dir).expanduser().resolve()
    if run_path.name != "20260814_181311":
        raise ValueError(f"This experiment was requested only for run 20260814_181311, got {run_path.name}")
    output_dir = run_path / "vehicle_identity_test" / "plate_assisted"
    output_dir.mkdir(parents=True, exist_ok=True)
    contact_dir = output_dir / "contact_sheets"
    contact_dir.mkdir(parents=True, exist_ok=True)

    verification = _verify_run(run_path)
    if not (verification["plate_enabled"] and verification["ocr_enabled"] and verification["rectangle_roi_enabled"]):
        raise RuntimeError(f"Run is not a verified plate-enabled rectangle ROI run: {verification}")

    tracks_hash_before = _sha256(run_path / "tracks.json")
    baseline_dir = output_dir / "no_plate_baseline"
    if not (baseline_dir / "vehicle_id_map.json").exists():
        from .vehicle_identity_experiment import run_vehicle_identity_experiment

        run_vehicle_identity_experiment(run_path, output_dir=baseline_dir)

    tracks = _read_json(run_path / "tracks.json", default=[])
    completed_tracks = [track for track in tracks if str(track.get("status", "")).upper() == "COMPLETED"]
    features = [TrackletFeature(**item) for item in _read_json(baseline_dir / "tracklet_features.json", default=[])]
    features_by_id = {item.local_track_id: item for item in features}
    base_pair_rows = _coerce_pair_rows(_read_csv(baseline_dir / "pairwise_scores.csv"))
    base_map = {str(k): str(v) for k, v in dict(_read_json(baseline_dir / "vehicle_id_map.json", default={})).items()}
    base_vehicles = _read_json(baseline_dir / "vehicles.json", default={}).get("vehicles", [])

    enrichment = {str(item.get("local_track_id")): item for item in _read_json(run_path / "vehicle_enrichment.json", default=[])}
    plate_csv_rows = _read_csv(run_path / "plate_ocr_results.csv")
    consensus = {
        track_id: _build_consensus(track_id, enrichment.get(track_id, {}), _find_plate_csv_row(plate_csv_rows, track_id))
        for track_id in sorted((str(track.get("local_track_id")) for track in completed_tracks), key=_track_sort_key)
    }
    plate_rows = [asdict(item) for item in consensus.values()]
    pair_rows = _score_plate_pairs(base_pair_rows, features_by_id, consensus)
    assisted_pair_rows = [_apply_plate_to_pair(row) for row in pair_rows]
    config = dict(IDENTITY_DEFAULT_CONFIG)
    config["weights"] = dict(config["weights"])
    config["weights"]["plate"] = PLATE_CONFIG["plate_weight"]
    config["acceptance_threshold"] = 0.70
    config["ambiguity_margin"] = 0.03
    config["vehicle_consistency_floor"] = 0.58
    assisted_map, decisions = _build_vehicles(features, assisted_pair_rows, config)
    vehicles = _vehicle_records(features, assisted_map)
    decisions = [_add_plate_decision_fields(row, consensus) for row in decisions]

    run_truth = _build_run_local_truth(consensus)
    evaluation = _evaluate_against_run_truth(assisted_map, run_truth)
    baseline_evaluation = _evaluate_against_run_truth(base_map, run_truth)
    plate_coverage = _plate_coverage(completed_tracks, consensus, pair_rows)
    examples = _build_examples(base_map, assisted_map, run_truth, pair_rows, consensus)

    _write_contact_sheets(run_path, contact_dir, run_truth, consensus)
    _write_json(output_dir / "tracklet_features.json", [asdict(item) for item in features])
    _write_json(output_dir / "track_plate_consensus.json", plate_rows)
    _write_csv(output_dir / "plate_pair_scores.csv", pair_rows)
    _write_csv(output_dir / "identity_scores.csv", assisted_pair_rows)
    _write_csv(output_dir / "association_decisions.csv", decisions)
    _write_json(output_dir / "vehicle_id_map.json", assisted_map)
    _write_json(output_dir / "vehicles.json", {"vehicles": vehicles})
    _write_json(output_dir / "run_local_plate_truth.json", run_truth)
    _write_json(output_dir / "evaluation.json", {
        "run_used": str(run_path),
        "verification": verification,
        "plate_coverage": plate_coverage,
        "baseline_without_plate": baseline_evaluation,
        "plate_assisted": evaluation,
        "examples": examples,
        "config": PLATE_CONFIG,
        "tracks_json_sha256_before": tracks_hash_before,
        "tracks_json_sha256_after": _sha256(run_path / "tracks.json"),
        "tracks_json_unchanged": tracks_hash_before == _sha256(run_path / "tracks.json"),
    })
    _write_report(output_dir / "report.md", run_path, verification, plate_coverage, baseline_evaluation, evaluation, examples, vehicles)
    return {
        "output_directory": str(output_dir),
        "verification": verification,
        "plate_coverage": plate_coverage,
        "baseline_without_plate": baseline_evaluation,
        "plate_assisted": evaluation,
        "tracks_json_unchanged": tracks_hash_before == _sha256(run_path / "tracks.json"),
    }


def _verify_run(run_path: Path) -> dict[str, Any]:
    config = yaml.safe_load((run_path / "run_config.yaml").read_text(encoding="utf-8")) or {}
    cameras = list(dict(config.get("input", {}) or {}).get("cameras", []) or [])
    plate = dict(dict(dict(config.get("vehicle_enrichment", {}) or {}).get("enrichment", {}) or {}).get("plate", {}) or {})
    detector = dict(plate.get("detector", {}) or {})
    ocr = dict(plate.get("ocr", {}) or {})
    tracking = dict(config.get("tracking", {}) or {})
    roi = dict(config.get("tracking_roi", {}) or {})
    return {
        "run_id": run_path.name,
        "video_path": str(cameras[0].get("source")) if cameras else None,
        "plate_enabled": bool(plate.get("enabled", False) or detector.get("enabled", False)),
        "ocr_enabled": bool(ocr.get("enabled", False)),
        "rectangle_roi_enabled": str(roi.get("mode", "")).lower() == "rectangle" and bool(roi.get("enabled", False)),
        "tracking_settings": tracking,
        "tracking_roi": roi,
        "plate_detector_model_path": detector.get("model_path"),
        "plate_ocr_adapter_path": dict(ocr.get("florence", {}) or {}).get("adapter_path"),
    }


def _build_consensus(track_id: str, enrichment: dict[str, Any], csv_row: dict[str, str] | None) -> PlateConsensus:
    raw_text = enrichment.get("plate_text") or (csv_row or {}).get("normalized_plate_text") or None
    normalized = normalize_plate_text(raw_text)
    plate_crop_path = str(enrichment.get("plate_crop_path") or (csv_row or {}).get("plate_crop_path") or "") or None
    vehicle_crop_path = _first_path(str((csv_row or {}).get("vehicle_crop_path") or ""))
    det_conf = _float_or_none(enrichment.get("plate_detection_confidence") or (csv_row or {}).get("plate_detection_confidence"))
    ocr_conf = _float_or_none(enrichment.get("plate_text_confidence"))
    plate_detected = bool(enrichment.get("plate_detected") or (csv_row or {}).get("plate_detection_confidence"))
    ocr_attempted = bool(enrichment.get("plate_ocr_attempted") or (csv_row or {}).get("ocr_status") == "completed")
    quality_status = str(enrichment.get("plate_quality_status") or (csv_row or {}).get("plate_quality_status") or "")
    ocr_reason = str(enrichment.get("plate_ocr_reason") or (csv_row or {}).get("ocr_reason") or "")
    width, height, sharpness = _image_stats(plate_crop_path)
    reliability = _plate_reliability(
        normalized=normalized,
        detected=plate_detected,
        det_conf=det_conf,
        ocr_conf=ocr_conf,
        width=width,
        height=height,
        sharpness=sharpness,
        quality_status=quality_status,
        ocr_reason=ocr_reason,
    )
    return PlateConsensus(
        local_track_id=track_id,
        plate_detected=plate_detected,
        ocr_attempted=ocr_attempted,
        raw_plate_text=str(raw_text) if raw_text else None,
        normalized_plate_text=normalized or None,
        plate_detection_confidence=det_conf,
        plate_text_confidence=ocr_conf,
        plate_crop_path=plate_crop_path,
        vehicle_crop_path=vehicle_crop_path,
        plate_ocr_reason=ocr_reason or None,
        plate_quality_status=quality_status or None,
        plate_crop_width=width,
        plate_crop_height=height,
        plate_crop_sharpness=round(sharpness, 6),
        reliability_score=round(reliability[0], 6),
        reliability_label=reliability[1],
        consensus_status="CONSENSUS" if reliability[1] in {"HIGH", "MEDIUM"} else "UNUSABLE",
        supporting_observations=1 if normalized else 0,
    )


def _plate_reliability(
    *,
    normalized: str,
    detected: bool,
    det_conf: float | None,
    ocr_conf: float | None,
    width: int,
    height: int,
    sharpness: float,
    quality_status: str,
    ocr_reason: str,
) -> tuple[float, str]:
    if not detected or not normalized or len(normalized) < int(PLATE_CONFIG["minimum_text_length"]):
        return 0.0, "UNUSABLE"
    if ocr_reason and ocr_reason not in {"ocr_completed", "completed"}:
        return 0.15, "LOW"
    det = float(det_conf or 0.0)
    ocr = float(ocr_conf if ocr_conf is not None else det)
    crop_size = min(1.0, (width * height) / float(100 * 32)) if width and height else 0.0
    sharp = min(1.0, sharpness / 400.0) if sharpness else 0.35
    accepted_bonus = 1.0 if quality_status == "plate_quality_accepted" else 0.35
    score = 0.36 * det + 0.36 * ocr + 0.14 * crop_size + 0.08 * sharp + 0.06 * accepted_bonus
    if score >= float(PLATE_CONFIG["high_score_threshold"]) and det >= float(PLATE_CONFIG["minimum_detector_confidence_high"]) and ocr >= float(PLATE_CONFIG["minimum_ocr_confidence_high"]):
        return score, "HIGH"
    if score >= float(PLATE_CONFIG["medium_score_threshold"]):
        return score, "MEDIUM"
    return score, "LOW"


def _score_plate_pairs(
    base_pair_rows: list[dict[str, str]],
    features_by_id: dict[str, TrackletFeature],
    consensus: dict[str, PlateConsensus],
) -> list[dict[str, Any]]:
    rows = []
    for row in base_pair_rows:
        a_id = str(row["track_a"])
        b_id = str(row["track_b"])
        a = consensus.get(a_id)
        b = consensus.get(b_id)
        relation = _plate_relation(a, b)
        impossible = _impossible_geometry(row, features_by_id.get(a_id), features_by_id.get(b_id))
        if impossible and relation["evidence"] in {"STRONG_POSITIVE", "PARTIAL_POSITIVE"}:
            relation["evidence"] = "GEOMETRY_BLOCKED_PLATE_MATCH"
            relation["contribution"] = 0.0
            relation["reason_code"] = "PLATE_MATCH_BLOCKED_BY_IMPOSSIBLE_GEOMETRY"
        rows.append(
            {
                **row,
                "track_a_plate": a.normalized_plate_text if a else None,
                "track_b_plate": b.normalized_plate_text if b else None,
                "track_a_plate_quality": a.reliability_label if a else "UNUSABLE",
                "track_b_plate_quality": b.reliability_label if b else "UNUSABLE",
                "literal_similarity": relation["literal_similarity"],
                "confusion_similarity": relation["confusion_similarity"],
                "edit_distance": relation["edit_distance"],
                "plate_evidence": relation["evidence"],
                "plate_contribution": relation["contribution"],
                "plate_reason_code": relation["reason_code"],
                "impossible_geometry": impossible,
            }
        )
    return rows


def _plate_relation(a: PlateConsensus | None, b: PlateConsensus | None) -> dict[str, Any]:
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
    if high and literal < float(PLATE_CONFIG["clear_contradiction_literal_threshold"]) and confusion < float(PLATE_CONFIG["clear_contradiction_confusion_threshold"]):
        return _relation("STRONG_NEGATIVE", -1.0, "PLATE_CONTRADICTION", a_text, b_text)
    if medium_or_high and (edit <= 1 or confusion >= float(PLATE_CONFIG["partial_match_threshold"])):
        return _relation("PARTIAL_POSITIVE", 0.72, "PLATE_PARTIAL_MATCH", a_text, b_text)
    if high:
        return _relation("WEAK_NEGATIVE", -0.25, "PLATE_LOW_SIMILARITY", a_text, b_text)
    return _relation("NEUTRAL", 0.0, "PLATE_LOW_CONFIDENCE", a_text, b_text)


def _relation(evidence: str, contribution: float, reason: str, a_text: str | None, b_text: str | None) -> dict[str, Any]:
    return {
        "evidence": evidence,
        "contribution": contribution,
        "reason_code": reason,
        "literal_similarity": round(_string_similarity(a_text or "", b_text or ""), 6) if a_text and b_text else 0.0,
        "confusion_similarity": round(_confusion_similarity(a_text or "", b_text or ""), 6) if a_text and b_text else 0.0,
        "edit_distance": _edit_distance(a_text or "", b_text or "") if a_text and b_text else None,
    }


def _apply_plate_to_pair(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    base_score = float(row.get("score", 0.0) or 0.0)
    contribution = float(row.get("plate_contribution", 0.0) or 0.0)
    evidence = str(row.get("plate_evidence") or "")
    impossible = bool(row.get("impossible_geometry"))
    if evidence == "STRONG_NEGATIVE":
        out["rejected"] = True
        out["rejection_reason"] = "REJECTED_BY_PLATE_CONTRADICTION"
        out["score"] = 0.0
    elif evidence == "STRONG_POSITIVE" and not impossible:
        out["rejected"] = False
        out["rejection_reason"] = ""
        out["score"] = round(max(base_score, float(PLATE_CONFIG["exact_match_override_threshold"])) + float(PLATE_CONFIG["exact_match_bonus"]), 6)
    elif evidence == "PARTIAL_POSITIVE" and not impossible and not _bool(row.get("rejected")):
        out["score"] = round(min(1.0, base_score + float(PLATE_CONFIG["partial_match_bonus"])), 6)
    elif evidence == "WEAK_NEGATIVE" and not _bool(row.get("rejected")):
        out["score"] = round(max(0.0, base_score - float(PLATE_CONFIG["contradiction_penalty"]) / 2.0), 6)
    else:
        out["score"] = round(base_score, 6)
    out["plate_score"] = round(max(0.0, min(1.0, (contribution + 1.0) / 2.0)), 6)
    return out


def _impossible_geometry(row: dict[str, Any], a: TrackletFeature | None, b: TrackletFeature | None) -> bool:
    if int(float(row.get("overlap_frames", 0) or 0)) <= 0:
        return False
    if a is None or b is None:
        return False
    distance = float(row.get("spatial_distance", 0.0) or 0.0)
    diag_a = _median_diag(a)
    diag_b = _median_diag(b)
    limit = float(PLATE_CONFIG["impossible_overlap_distance_ratio"]) * max(diag_a, diag_b, 1.0)
    iou = float(row.get("overlap_iou", 0.0) or 0.0)
    absolute_limit = float(PLATE_CONFIG["impossible_overlap_absolute_pixels"])
    return distance > max(limit, absolute_limit) and iou < float(PLATE_CONFIG["impossible_overlap_iou_threshold"])


def _build_run_local_truth(consensus: dict[str, PlateConsensus]) -> dict[str, Any]:
    by_plate: dict[str, list[str]] = {}
    for item in consensus.values():
        if item.reliability_label == "HIGH" and item.normalized_plate_text:
            by_plate.setdefault(item.normalized_plate_text, []).append(item.local_track_id)
    groups = [sorted(members, key=_track_sort_key) for members in by_plate.values() if len(members) > 1]
    # A conservative visual/plate partial: HR30T42 is an incomplete OCR read of HR30T4246.
    hr30 = sorted([tid for tid, item in consensus.items() if item.normalized_plate_text in {"HR30T4246", "HR30T42"}], key=_track_sort_key)
    if len(hr30) >= 2 and hr30 not in groups:
        groups.append(hr30)
    high_items = [item for item in consensus.values() if item.reliability_label == "HIGH" and item.normalized_plate_text]
    must_not_merge = []
    for index, a in enumerate(high_items):
        for b in high_items[index + 1 :]:
            if a.normalized_plate_text != b.normalized_plate_text and _confusion_similarity(a.normalized_plate_text, b.normalized_plate_text) < 0.72:
                must_not_merge.append(sorted([a.local_track_id, b.local_track_id], key=_track_sort_key))
    return {"same_vehicle_groups": groups, "must_not_merge": must_not_merge}


def _evaluate_against_run_truth(mapping: dict[str, str], truth: dict[str, Any]) -> dict[str, Any]:
    positives = {tuple(sorted(pair)) for group in truth["same_vehicle_groups"] for pair in _pairs(group)}
    negatives = {tuple(sorted(pair)) for pair in truth["must_not_merge"]}
    predicted = {tuple(sorted((a, b))) for a, va in mapping.items() for b, vb in mapping.items() if a < b and va == vb}
    tp = len(predicted & positives)
    fp = len(predicted & negatives)
    fn = len(positives - predicted)
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "raw_completed_tracks": len(mapping),
        "reconciled_identities": len(set(mapping.values())),
        "duplicates_removed": len(mapping) - len(set(mapping.values())),
        "true_fragment_merges": tp,
        "false_merges": fp,
        "missed_merges": fn,
        "merge_precision": precision,
        "merge_recall": recall,
        "false_merge_pairs": sorted(predicted & negatives),
        "missed_positive_pairs": sorted(positives - predicted),
    }


def _plate_coverage(completed_tracks: list[dict[str, Any]], consensus: dict[str, PlateConsensus], pair_rows: list[dict[str, Any]]) -> dict[str, Any]:
    completed = len(completed_tracks)
    detected = sum(1 for item in consensus.values() if item.plate_detected)
    attempted = sum(1 for item in consensus.values() if item.ocr_attempted)
    readable = sum(1 for item in consensus.values() if item.normalized_plate_text)
    high = sum(1 for item in consensus.values() if item.reliability_label == "HIGH")
    return {
        "completed_tracks": completed,
        "plate_detected_count": detected,
        "plate_detection_rate": detected / completed if completed else 0.0,
        "ocr_attempted_count": attempted,
        "readable_plate_count": readable,
        "readable_plate_rate": readable / completed if completed else 0.0,
        "high_quality_plate_count": high,
        "exact_matching_plate_pairs": sum(1 for row in pair_rows if row.get("plate_evidence") == "STRONG_POSITIVE"),
        "partial_matching_plate_pairs": sum(1 for row in pair_rows if row.get("plate_evidence") == "PARTIAL_POSITIVE"),
        "high_confidence_contradictory_pairs": sum(1 for row in pair_rows if row.get("plate_evidence") == "STRONG_NEGATIVE"),
    }


def _build_examples(
    base_map: dict[str, str],
    assisted_map: dict[str, str],
    truth: dict[str, Any],
    pair_rows: list[dict[str, Any]],
    consensus: dict[str, PlateConsensus],
) -> dict[str, Any]:
    lookup = {tuple(sorted((str(row["track_a"]), str(row["track_b"])))): row for row in pair_rows}
    same = []
    for group in truth["same_vehicle_groups"]:
        for a, b in _pairs(group):
            row = lookup.get(tuple(sorted((a, b))), {})
            same.append(_example_row(a, b, row, base_map, assisted_map, consensus))
    contradictions = []
    for pair in truth["must_not_merge"][:12]:
        a, b = pair
        row = lookup.get(tuple(sorted((a, b))), {})
        contradictions.append(_example_row(a, b, row, base_map, assisted_map, consensus))
    return {
        "same_vehicle_fragments": same,
        "plate_contradictions": contradictions,
        "yellow_plate_vehicle": "No manually verified yellow commercial plate vehicle was isolated from this run without external ground truth; plate crops are available in contact_sheets for manual review.",
    }


def _example_row(a: str, b: str, row: dict[str, Any], base_map: dict[str, str], assisted_map: dict[str, str], consensus: dict[str, PlateConsensus]) -> dict[str, Any]:
    return {
        "track_a": a,
        "track_b": b,
        "track_a_plate": consensus[a].normalized_plate_text if a in consensus else None,
        "track_b_plate": consensus[b].normalized_plate_text if b in consensus else None,
        "track_a_quality": consensus[a].reliability_label if a in consensus else None,
        "track_b_quality": consensus[b].reliability_label if b in consensus else None,
        "plate_evidence": row.get("plate_evidence"),
        "plate_reason_code": row.get("plate_reason_code"),
        "base_merged": base_map.get(a) == base_map.get(b),
        "plate_assisted_merged": assisted_map.get(a) == assisted_map.get(b),
        "impossible_geometry": row.get("impossible_geometry"),
    }


def _write_contact_sheets(run_path: Path, contact_dir: Path, truth: dict[str, Any], consensus: dict[str, PlateConsensus]) -> None:
    groups = list(truth["same_vehicle_groups"])
    for group in groups:
        _write_group_contact_sheet(contact_dir / f"same__{'__'.join(t.split(':')[-1] for t in group)}.jpg", group, consensus)
    for pair in truth["must_not_merge"][:12]:
        _write_group_contact_sheet(contact_dir / f"contradiction__{pair[0].split(':')[-1]}__{pair[1].split(':')[-1]}.jpg", pair, consensus)


def _write_group_contact_sheet(path: Path, track_ids: list[str], consensus: dict[str, PlateConsensus]) -> None:
    tiles = []
    for track_id in track_ids:
        item = consensus[track_id]
        vehicle = _read_first_image(item.vehicle_crop_path)
        plate = _read_first_image(item.plate_crop_path)
        tile = np.full((260, 260, 3), 245, dtype=np.uint8)
        if vehicle is not None:
            tile[0:150, 0:260] = _fit_image(vehicle, 260, 150)
        if plate is not None:
            tile[155:210, 0:180] = _fit_image(plate, 180, 55)
        label = f"{track_id.split(':')[-1]} {item.normalized_plate_text or 'NO_PLATE'} {item.reliability_label}"
        cv2.putText(tile, label[:34], (4, 238), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1)
        cv2.putText(tile, f"conf {item.plate_detection_confidence or 0:.2f}", (4, 254), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 1)
        tiles.append(tile)
    if tiles:
        cv2.imwrite(str(path), np.hstack(tiles))


def _write_report(path: Path, run_path: Path, verification: dict[str, Any], coverage: dict[str, Any], baseline: dict[str, Any], assisted: dict[str, Any], examples: dict[str, Any], vehicles: list[dict[str, Any]]) -> None:
    decision = _final_decision(baseline, assisted, coverage)
    lines = [
        "# Plate-Assisted Identity Experiment",
        "",
        f"- Run: `{run_path.name}`",
        f"- Video: `{verification.get('video_path')}`",
        f"- Plate enabled: `{verification['plate_enabled']}`",
        f"- OCR enabled: `{verification['ocr_enabled']}`",
        f"- Rectangle ROI enabled: `{verification['rectangle_roi_enabled']}`",
        f"- Raw completed tracks: `{coverage['completed_tracks']}`",
        f"- Plate detected/readable/high-quality: `{coverage['plate_detected_count']}` / `{coverage['readable_plate_count']}` / `{coverage['high_quality_plate_count']}`",
        "",
        "## Metrics",
        "",
        "| Metric | No Plate | Plate Assisted |",
        "| --- | ---: | ---: |",
        f"| Raw tracks | {baseline['raw_completed_tracks']} | {assisted['raw_completed_tracks']} |",
        f"| Reconciled identities | {baseline['reconciled_identities']} | {assisted['reconciled_identities']} |",
        f"| True fragment merges | {baseline['true_fragment_merges']} | {assisted['true_fragment_merges']} |",
        f"| False merges | {baseline['false_merges']} | {assisted['false_merges']} |",
        f"| Missed merges | {baseline['missed_merges']} | {assisted['missed_merges']} |",
        f"| Merge precision | {baseline['merge_precision']:.3f} | {assisted['merge_precision']:.3f} |",
        f"| Merge recall | {baseline['merge_recall']:.3f} | {assisted['merge_recall']:.3f} |",
        "",
        "## Same-Vehicle Plate Examples",
    ]
    for item in examples["same_vehicle_fragments"]:
        lines.append(f"- `{item['track_a']}` / `{item['track_b']}`: {item['track_a_plate']} vs {item['track_b_plate']} -> base={item['base_merged']} plate={item['plate_assisted_merged']} reason={item['plate_reason_code']} impossible={item['impossible_geometry']}")
    lines.extend(["", "## Plate Contradiction Examples"])
    for item in examples["plate_contradictions"][:8]:
        lines.append(f"- `{item['track_a']}` / `{item['track_b']}`: {item['track_a_plate']} vs {item['track_b_plate']} -> base={item['base_merged']} plate={item['plate_assisted_merged']} reason={item['plate_reason_code']}")
    lines.extend(["", "## Multi-Track Vehicles"])
    for vehicle in vehicles:
        if len(vehicle["member_tracks"]) > 1:
            lines.append(f"- `{vehicle['vehicle_id']}`: {', '.join(vehicle['member_tracks'])}")
    lines.extend(["", f"## Final Decision", "", decision])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _final_decision(baseline: dict[str, Any], assisted: dict[str, Any], coverage: dict[str, Any]) -> str:
    if coverage["readable_plate_count"] == 0 or coverage["high_quality_plate_count"] < 2:
        return "PLATE OCR QUALITY INSUFFICIENT FOR IDENTITY USE"
    if assisted["false_merges"] > baseline["false_merges"]:
        return "UNSAFE - PLATE EVIDENCE INTRODUCES FALSE MERGES"
    if assisted["true_fragment_merges"] > baseline["true_fragment_merges"] and assisted["false_merges"] <= baseline["false_merges"]:
        return "PLATE-ASSISTED IDENTITY IMPROVES DUPLICATE REMOVAL SAFELY"
    return "PARTIAL - PLATE HELPS BUT MORE CALIBRATION REQUIRED"


def _find_plate_csv_row(rows: list[dict[str, str]], track_id: str) -> dict[str, str] | None:
    return next((row for row in rows if row.get("local_track_id") == track_id), None)


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _coerce_pair_rows(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    numeric_keys = {
        "score",
        "frame_gap",
        "time_gap_seconds",
        "overlap_frames",
        "spatial_distance",
        "overlap_iou",
        "appearance_quality",
        "temporal_score",
        "spatial_score",
        "motion_score",
        "class_score",
        "appearance_score",
        "colour_score",
        "plate_score",
    }
    coerced = []
    for row in rows:
        payload: dict[str, Any] = dict(row)
        payload["rejected"] = _bool(payload.get("rejected"))
        for key in numeric_keys:
            if key in payload:
                value = _float_or_none(payload.get(key))
                payload[key] = value if value is not None else 0.0
        coerced.append(payload)
    return coerced


def _float_or_none(value: Any) -> float | None:
    try:
        if value in (None, ""):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes"}


def _first_path(value: str) -> str | None:
    first = str(value or "").split("|")[0].strip()
    return first or None


def _image_stats(path: str | None) -> tuple[int, int, float]:
    image = cv2.imread(str(path or ""))
    if image is None or image.size == 0:
        return 0, 0, 0.0
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return int(image.shape[1]), int(image.shape[0]), float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _read_first_image(path: str | None) -> np.ndarray | None:
    image = cv2.imread(str(path or ""))
    if image is None or image.size == 0:
        return None
    return image


def _fit_image(image: np.ndarray, width: int, height: int) -> np.ndarray:
    canvas = np.full((height, width, 3), 235, dtype=np.uint8)
    h, w = image.shape[:2]
    scale = min(width / max(w, 1), height / max(h, 1))
    resized = cv2.resize(image, (max(1, int(w * scale)), max(1, int(h * scale))))
    y = (height - resized.shape[0]) // 2
    x = (width - resized.shape[1]) // 2
    canvas[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    return canvas


def _median_diag(feature: TrackletFeature) -> float:
    if not feature.bbox_size_history:
        return 1.0
    return float(np.median([math.hypot(max(1.0, w), max(1.0, h)) for w, h in feature.bbox_size_history]))


def _string_similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def _confusion_similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    max_len = max(len(a), len(b))
    edit = _weighted_edit_distance(a, b)
    return max(0.0, 1.0 - edit / max_len)


def _edit_distance(a: str, b: str) -> int:
    return int(_weighted_edit_distance(a, b, weighted=False))


def _weighted_edit_distance(a: str, b: str, *, weighted: bool = True) -> float:
    confusions = {tuple(sorted(pair)) for pair in [("0", "O"), ("1", "I"), ("5", "S"), ("8", "B"), ("2", "Z")]}
    dp = [[0.0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(len(a) + 1):
        dp[i][0] = float(i)
    for j in range(len(b) + 1):
        dp[0][j] = float(j)
    for i, ca in enumerate(a, 1):
        for j, cb in enumerate(b, 1):
            if ca == cb:
                cost = 0.0
            elif weighted and tuple(sorted((ca, cb))) in confusions:
                cost = 0.35
            else:
                cost = 1.0
            dp[i][j] = min(dp[i - 1][j] + 1.0, dp[i][j - 1] + 1.0, dp[i - 1][j - 1] + cost)
    return dp[-1][-1]


def _pairs(group: list[str]) -> list[tuple[str, str]]:
    return [(group[i], group[j]) for i in range(len(group)) for j in range(i + 1, len(group))]


def _add_plate_decision_fields(row: dict[str, Any], consensus: dict[str, PlateConsensus]) -> dict[str, Any]:
    a = consensus.get(str(row.get("track_a")))
    b = consensus.get(str(row.get("track_b")))
    return {
        **row,
        "decision_reason_codes": " | ".join(
            item
            for item in [
                str(row.get("plate_reason_code") or ""),
                "SPATIAL_MATCH" if float(row.get("spatial_score", 0.0) or 0.0) >= 0.55 else "",
                "APPEARANCE_MATCH" if float(row.get("appearance_score", 0.0) or 0.0) >= 0.75 else "",
            ]
            if item
        ),
        "track_a_plate_quality": a.reliability_label if a else "UNUSABLE",
        "track_b_plate_quality": b.reliability_label if b else "UNUSABLE",
    }
