from __future__ import annotations

from pathlib import Path

import pytest

from src.vehicle_identity import _build_plate_consensus, build_physical_vehicle_identity_for_run, normalize_vehicle_identity_config


def test_physical_vehicle_identity_matches_validated_plate_run() -> None:
    run_dir = Path("outputs/runs/20260814_181311")
    if not run_dir.exists():
        pytest.skip("Validated plate-enabled rectangle ROI run is not available.")

    result = build_physical_vehicle_identity_for_run(run_dir)

    assert result.metrics["raw_completed_tracks"] == 23
    assert result.metrics["physical_vehicle_count"] == 14
    assert result.metrics["duplicates_removed"] == 9
    assert result.metrics["plate_exact_merges"] == 2
    assert result.metrics["plate_contradiction_rejections"] == 0

    for left, right in (
        ("CAM_001:TRACK_11", "CAM_001:TRACK_13"),
        ("CAM_001:TRACK_15", "CAM_001:TRACK_17"),
    ):
        assert result.vehicle_identity_map[left] == result.vehicle_identity_map[right]
    assert result.vehicle_identity_map["CAM_001:TRACK_4"] != result.vehicle_identity_map["CAM_001:TRACK_14"]


def test_validated_plate_reason_counts_as_high_quality_consensus() -> None:
    config = normalize_vehicle_identity_config(None)

    consensus = _build_plate_consensus(
        "CAM_001:TRACK_14",
        {
            "plate_detected": True,
            "plate_ocr_attempted": True,
            "plate_text": "HR38AD4296",
            "plate_detection_confidence": 0.7553320527076721,
            "plate_text_confidence": 0.7553320527076721,
            "plate_crop_path": "plate.jpg",
            "plate_bbox": [150.0, 257.0, 263.0, 293.0],
            "plate_ocr_raw_response": "HR38AD4296",
            "plate_ocr_reason": "validated_standard_state_registration",
            "plate_quality_status": "plate_quality_accepted",
        },
        config,
    )

    assert consensus.normalized_plate_text == "HR38AD4296"
    assert consensus.reliability_label == "HIGH"
    assert consensus.reliability_score == pytest.approx(0.794479, rel=1e-6)
