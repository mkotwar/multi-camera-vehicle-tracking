from __future__ import annotations

from pathlib import Path

import pytest

from src.vehicle_identity import build_physical_vehicle_identity_for_run


def test_physical_vehicle_identity_matches_validated_plate_run() -> None:
    run_dir = Path("outputs/runs/20260814_181311")
    if not run_dir.exists():
        pytest.skip("Validated plate-enabled rectangle ROI run is not available.")

    result = build_physical_vehicle_identity_for_run(run_dir)

    assert result.metrics["raw_completed_tracks"] == 23
    assert result.metrics["physical_vehicle_count"] == 13
    assert result.metrics["duplicates_removed"] == 10
    assert result.metrics["plate_exact_merges"] == 3
    assert result.metrics["plate_contradiction_rejections"] == 0

    for left, right in (
        ("CAM_001:TRACK_4", "CAM_001:TRACK_14"),
        ("CAM_001:TRACK_11", "CAM_001:TRACK_13"),
        ("CAM_001:TRACK_15", "CAM_001:TRACK_17"),
    ):
        assert result.vehicle_identity_map[left] == result.vehicle_identity_map[right]
