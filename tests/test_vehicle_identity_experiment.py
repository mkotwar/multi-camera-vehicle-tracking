from __future__ import annotations

from src.stationary_recovery_experiment import (
    DEFAULT_STATIONARY_RECOVERY_CONFIG,
    VehicleGroupFeature,
    _build_persistent_vehicles,
    _group_appearance_score,
    _same_location_score,
    _score_recovery_pair,
    _stationary_confidence,
)
from src.vehicle_identity_experiment import DEFAULT_CONFIG, TrackletFeature, _build_vehicles, _score_pair


def _feature(track_id: str, *, camera: str = "CAM_001", cls: str = "CAR", first: int = 0, last: int = 10, center=(100.0, 100.0), motion=5.0) -> TrackletFeature:
    return TrackletFeature(
        camera_id=camera,
        local_track_id=track_id,
        native_tracker_id=int(track_id.split("TRACK_")[-1]),
        status="COMPLETED",
        first_frame=first,
        last_frame=last,
        first_timestamp=first / 30.0,
        last_timestamp=last / 30.0,
        duration_seconds=(last - first) / 30.0,
        observation_count=last - first + 1,
        final_class=cls,
        class_distribution={cls: last - first + 1},
        colour="UNKNOWN",
        start_bbox=[center[0] - 20, center[1] - 10, center[0] + 20, center[1] + 10],
        end_bbox=[center[0] - 20 + motion, center[1] - 10, center[0] + 20 + motion, center[1] + 10],
        start_center=[center[0], center[1]],
        end_center=[center[0] + motion, center[1]],
        mean_center=[center[0] + motion / 2.0, center[1]],
        trajectory_points=[[center[0], center[1]], [center[0] + motion, center[1]]],
        estimated_direction=[1.0, 0.0],
        speed_pixels_per_frame=motion / max(1, last - first),
        motion_magnitude=motion,
        stationary=motion <= DEFAULT_CONFIG["stationary_motion_threshold"],
        bbox_size_history=[[40.0, 20.0]],
        best_evidence_crops=[],
        appearance_descriptor=[],
        evidence_quality=0.0,
        median_step_pixels=0.0,
        normalized_displacement=0.0,
    )


def test_same_camera_gate_rejects_cross_camera_pair() -> None:
    row = _score_pair(_feature("CAM_001:TRACK_1"), _feature("CAM_002:TRACK_2", camera="CAM_002", first=12, last=20), DEFAULT_CONFIG)
    assert row["rejected"] is True
    assert row["rejection_reason"] == "different_camera"


def test_class_conflict_gate_rejects_reliable_incompatible_pair() -> None:
    row = _score_pair(_feature("CAM_001:TRACK_1", cls="CAR"), _feature("CAM_001:TRACK_2", cls="MOTORCYCLE", first=12, last=20), DEFAULT_CONFIG)
    assert row["rejected"] is True
    assert row["rejection_reason"] == "reliable_class_conflict"


def test_overlap_duplicate_pair_can_be_scored_not_rejected() -> None:
    row = _score_pair(_feature("CAM_001:TRACK_1", first=10, last=30), _feature("CAM_001:TRACK_2", first=20, last=40), DEFAULT_CONFIG)
    assert row["association_mode"] == "DUPLICATE_OVERLAP"
    assert row["rejected"] is False


def test_ambiguity_margin_prevents_forced_merge() -> None:
    a = _feature("CAM_001:TRACK_1", first=0, last=10)
    c = _feature("CAM_001:TRACK_3", first=0, last=10)
    b = _feature("CAM_001:TRACK_2", first=12, last=20)
    pairs = [_score_pair(a, b, DEFAULT_CONFIG), _score_pair(c, b, DEFAULT_CONFIG)]
    mapping, decisions = _build_vehicles([a, b, c], pairs, DEFAULT_CONFIG)
    assert decisions[0]["decision"] == "NEW_OR_AMBIGUOUS"
    assert len(set(mapping.values())) == 3


def test_deterministic_vehicle_ids_for_accepted_pair() -> None:
    a = _feature("CAM_001:TRACK_1", first=0, last=10)
    b = _feature("CAM_001:TRACK_2", first=12, last=20)
    pairs = [_score_pair(a, b, DEFAULT_CONFIG)]
    mapping, _decisions = _build_vehicles([b, a], pairs, DEFAULT_CONFIG)
    assert mapping["CAM_001:TRACK_1"] == "VEHICLE_001"
    assert mapping["CAM_001:TRACK_2"] == "VEHICLE_001"


def _group(vehicle_id: str, *, first: int = 0, last: int = 30, center=(100.0, 100.0), size=(120.0, 80.0), stationary=0.9, cls="CAR") -> VehicleGroupFeature:
    return VehicleGroupFeature(
        vehicle_id=vehicle_id,
        camera_id="CAM_001",
        member_tracks=[f"CAM_001:TRACK_{int(vehicle_id.split('_')[-1])}"],
        final_class=cls,
        class_distribution={cls: 30},
        first_frame=first,
        last_frame=last,
        first_timestamp=first / 30.0,
        last_timestamp=last / 30.0,
        median_center=[center[0], center[1]],
        center_spread=4.0,
        median_width=size[0],
        median_height=size[1],
        footprint_bbox=[center[0] - size[0] / 2, center[1] - size[1] / 2, center[0] + size[0] / 2, center[1] + size[1] / 2],
        stationary_confidence=stationary,
        appearance_descriptor=[],
        evidence_quality=0.0,
        best_evidence_crops=[],
    )


def test_stationary_confidence_rewards_jittery_parked_track() -> None:
    parked = [{"normalized_displacement": 0.25, "median_step_pixels": 2.0, "speed_pixels_per_frame": 0.8, "duration_seconds": 4.0}]
    moving = [{"normalized_displacement": 1.6, "median_step_pixels": 8.5, "speed_pixels_per_frame": 5.0, "duration_seconds": 4.0}]
    assert _stationary_confidence(parked, 5.0, 120.0, 80.0) > 0.75
    assert _stationary_confidence(moving, 80.0, 120.0, 80.0) < 0.35


def test_stationary_location_score_uses_bbox_normalized_distance_and_size() -> None:
    same = _same_location_score(_group("VEHICLE_001"), _group("VEHICLE_002", center=(112.0, 104.0)), DEFAULT_STATIONARY_RECOVERY_CONFIG)
    nearby_small = _same_location_score(_group("VEHICLE_001"), _group("VEHICLE_003", center=(112.0, 104.0), size=(35.0, 42.0)), DEFAULT_STATIONARY_RECOVERY_CONFIG)
    assert same[0] > 0.85
    assert nearby_small[0] < same[0]


def test_stationary_recovery_excludes_moving_and_class_conflicts() -> None:
    moving = _group("VEHICLE_001", stationary=0.2)
    parked = _group("VEHICLE_002", first=60, last=80)
    class_conflict = _group("VEHICLE_003", first=90, last=110, cls="BUS")
    assert _score_recovery_pair(moving, parked, DEFAULT_STATIONARY_RECOVERY_CONFIG)["rejection_reason"] == "moving_or_low_stationary_confidence"
    assert _score_recovery_pair(parked, class_conflict, DEFAULT_STATIONARY_RECOVERY_CONFIG)["rejection_reason"] == "reliable_class_conflict"


def test_stationary_recovery_rejects_simultaneous_different_occupancy() -> None:
    a = _group("VEHICLE_001", first=0, last=100)
    b = _group("VEHICLE_002", first=50, last=120, center=(240.0, 100.0))
    row = _score_recovery_pair(a, b, DEFAULT_STATIONARY_RECOVERY_CONFIG)
    assert row["rejection_reason"] == "simultaneous_occupancy_conflict"


def test_appearance_missing_or_low_quality_is_neutral() -> None:
    score, quality, reason = _group_appearance_score(_group("VEHICLE_001"), _group("VEHICLE_002"), DEFAULT_STATIONARY_RECOVERY_CONFIG)
    assert score == 0.5
    assert quality == 0.0
    assert reason == "low_quality_neutral"


def test_stationary_recovery_builds_deterministic_persistent_ids_without_unsafe_chaining() -> None:
    a = _group("VEHICLE_001", first=0, last=30)
    b = _group("VEHICLE_002", first=120, last=150, center=(150.0, 100.0))
    c = _group("VEHICLE_003", first=240, last=270, center=(220.0, 100.0))
    rows = [_score_recovery_pair(a, b, DEFAULT_STATIONARY_RECOVERY_CONFIG), _score_recovery_pair(b, c, DEFAULT_STATIONARY_RECOVERY_CONFIG), _score_recovery_pair(a, c, DEFAULT_STATIONARY_RECOVERY_CONFIG)]
    mapping, persistent, decisions = _build_persistent_vehicles([a, b, c], rows, DEFAULT_STATIONARY_RECOVERY_CONFIG)
    assert mapping["VEHICLE_001"] == "PVEHICLE_001"
    assert mapping["VEHICLE_002"] == "PVEHICLE_001"
    assert mapping["VEHICLE_003"] != "PVEHICLE_001"
    assert any(decision["final_reason"] in {"whole_vehicle_consistency_conflict", "below_recovery_threshold"} for decision in decisions)
