from __future__ import annotations

import json
from pathlib import Path

from src.plate_assisted_identity_experiment import (
    PlateConsensus,
    _apply_plate_to_pair,
    _evaluate_against_run_truth,
    _plate_relation,
    _sha256,
    normalize_plate_text,
)


def _plate(track_id: str, text: str | None, *, quality: str = "HIGH", score: float = 0.82) -> PlateConsensus:
    return PlateConsensus(
        local_track_id=track_id,
        plate_detected=bool(text),
        ocr_attempted=bool(text),
        raw_plate_text=text,
        normalized_plate_text=normalize_plate_text(text),
        plate_detection_confidence=0.82 if text else None,
        plate_text_confidence=0.82 if text else None,
        plate_crop_path="plate.jpg" if text else None,
        vehicle_crop_path="vehicle.jpg",
        plate_ocr_reason="ocr_completed" if text else "no_plate_detected",
        plate_quality_status="plate_quality_accepted" if text else "no_plate_detection",
        plate_crop_width=100 if text else 0,
        plate_crop_height=32 if text else 0,
        plate_crop_sharpness=200.0 if text else 0.0,
        reliability_score=score if text else 0.0,
        reliability_label=quality if text else "UNUSABLE",
        consensus_status="CONSENSUS" if text else "UNUSABLE",
        supporting_observations=1 if text else 0,
    )


def test_normalize_plate_text_is_deterministic_without_confusion_rewrites() -> None:
    assert normalize_plate_text(" mp-09 aB 12.34 ") == "MP09AB1234"
    assert normalize_plate_text("BOI58Z") == "BOI58Z"


def test_exact_high_confidence_plate_match_is_strong_positive() -> None:
    relation = _plate_relation(_plate("A", "MP09AB1234"), _plate("B", "MP09AB1234"))

    assert relation["evidence"] == "STRONG_POSITIVE"
    assert relation["reason_code"] == "PLATE_EXACT_MATCH"


def test_high_confidence_plate_contradiction_is_strong_negative() -> None:
    relation = _plate_relation(_plate("A", "MP09AB1234"), _plate("B", "DL01XY6789"))

    assert relation["evidence"] == "STRONG_NEGATIVE"
    assert relation["reason_code"] == "PLATE_CONTRADICTION"


def test_missing_plate_is_neutral() -> None:
    relation = _plate_relation(_plate("A", None), _plate("B", "DL01XY6789"))

    assert relation["evidence"] == "NEUTRAL"
    assert relation["reason_code"] == "PLATE_MISSING"


def test_low_confidence_ocr_mismatch_is_neutral() -> None:
    relation = _plate_relation(_plate("A", "MP09AB1234", quality="LOW", score=0.3), _plate("B", "DL01XY6789"))

    assert relation["evidence"] == "NEUTRAL"
    assert relation["reason_code"] == "PLATE_LOW_CONFIDENCE"


def test_one_character_ocr_difference_is_partial_positive() -> None:
    relation = _plate_relation(_plate("A", "MP09AB1234"), _plate("B", "MP09A81234"))

    assert relation["evidence"] == "PARTIAL_POSITIVE"
    assert relation["reason_code"] == "PLATE_PARTIAL_MATCH"
    assert relation["confusion_similarity"] > relation["literal_similarity"]


def test_plate_contradiction_rejects_candidate_pair() -> None:
    row = {
        "score": 0.92,
        "rejected": False,
        "rejection_reason": "",
        "plate_evidence": "STRONG_NEGATIVE",
        "plate_contribution": -1.0,
    }

    updated = _apply_plate_to_pair(row)

    assert updated["rejected"] is True
    assert updated["rejection_reason"] == "REJECTED_BY_PLATE_CONTRADICTION"
    assert updated["score"] == 0.0


def test_same_plate_cannot_override_impossible_geometry() -> None:
    row = {
        "score": 0.2,
        "rejected": True,
        "rejection_reason": "overlap_not_same_object",
        "plate_evidence": "GEOMETRY_BLOCKED_PLATE_MATCH",
        "plate_contribution": 0.0,
        "impossible_geometry": True,
    }

    updated = _apply_plate_to_pair(row)

    assert updated["rejected"] is True
    assert updated["rejection_reason"] == "overlap_not_same_object"


def test_plate_assisted_merge_provenance_and_raw_track_ids_preserved() -> None:
    mapping = {"CAM_001:TRACK_4": "VEHICLE_001", "CAM_001:TRACK_14": "VEHICLE_001"}
    truth = {"same_vehicle_groups": [["CAM_001:TRACK_4", "CAM_001:TRACK_14"]], "must_not_merge": []}

    metrics = _evaluate_against_run_truth(mapping, truth)

    assert set(mapping) == {"CAM_001:TRACK_4", "CAM_001:TRACK_14"}
    assert metrics["true_fragment_merges"] == 1
    assert metrics["merge_precision"] == 1.0


def test_tracks_json_hash_is_unchanged_when_only_experimental_output_is_written(tmp_path: Path) -> None:
    tracks_path = tmp_path / "tracks.json"
    tracks_path.write_text(json.dumps([{"local_track_id": "CAM_001:TRACK_1"}]), encoding="utf-8")
    before = _sha256(tracks_path)
    (tmp_path / "vehicle_identity_test" / "plate_assisted").mkdir(parents=True)
    (tmp_path / "vehicle_identity_test" / "plate_assisted" / "evaluation.json").write_text("{}", encoding="utf-8")

    assert _sha256(tracks_path) == before
