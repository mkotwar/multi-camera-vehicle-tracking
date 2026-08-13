from __future__ import annotations

import csv
import json
from pathlib import Path

from src.track_reconciliation import run_track_reconciliation_experiment


def _track(track_id: int, *, first: int, last: int, vehicle_class: str = "CAR", colour: str = "BLACK") -> dict:
    local_track_id = f"CAM_001:TRACK_{track_id}"
    return {
        "local_track_id": local_track_id,
        "camera_id": "CAM_001",
        "tracker_namespace": "camera",
        "native_tracker_id": track_id,
        "status": "COMPLETED",
        "first_frame": first,
        "last_frame": last,
        "first_timestamp_seconds": first / 10.0,
        "last_timestamp_seconds": last / 10.0,
        "observation_count": last - first + 1,
        "lost_frames": 31,
        "final_class": vehicle_class,
        "final_class_reason": "WEIGHTED_MAJORITY",
        "class_counts": {vehicle_class.lower(): last - first + 1},
        "class_confidence_sums": {vehicle_class.lower(): float(last - first + 1) * 0.8},
        "completion_reason": "LOST_TIMEOUT",
        "vehicle_enrichment": {
            "vehicle_colour": {
                "label": colour,
                "status": "completed" if colour != "UNKNOWN" else "skipped",
            }
        },
    }


def _obs(local_track_id: str, frame: int, center_x: float, center_y: float = 100.0, raw_class: str = "car") -> dict:
    return {
        "local_track_id": local_track_id,
        "camera_id": "CAM_001",
        "tracker_namespace": "camera",
        "native_tracker_id": local_track_id.rsplit("_", 1)[-1],
        "frame_number": frame,
        "timestamp_seconds": frame / 10.0,
        "x1": center_x - 10.0,
        "y1": center_y - 5.0,
        "x2": center_x + 10.0,
        "y2": center_y + 5.0,
        "confidence": 0.8,
        "raw_class_id": 0,
        "raw_class_name": raw_class,
    }


def _write_run(tmp_path: Path, tracks: list[dict], observations: list[dict]) -> Path:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "tracks.json").write_text(json.dumps(tracks, indent=2), encoding="utf-8")
    with (run_dir / "observations.csv").open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "local_track_id",
            "camera_id",
            "tracker_namespace",
            "native_tracker_id",
            "frame_number",
            "timestamp_seconds",
            "x1",
            "y1",
            "x2",
            "y2",
            "confidence",
            "raw_class_id",
            "raw_class_name",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(observations)
    return run_dir


def _observations_for(track_id: int, frames_and_centers: list[tuple[int, float]], *, raw_class: str = "car") -> list[dict]:
    local_track_id = f"CAM_001:TRACK_{track_id}"
    return [_obs(local_track_id, frame, center, raw_class=raw_class) for frame, center in frames_and_centers]


def test_genuine_occlusion_recovery_assigns_same_vehicle_id(tmp_path: Path) -> None:
    tracks = [_track(12, first=0, last=2), _track(27, first=5, last=7)]
    observations = _observations_for(12, [(0, 10), (1, 20), (2, 30)]) + _observations_for(27, [(5, 60), (6, 70), (7, 80)])
    result = run_track_reconciliation_experiment(_write_run(tmp_path, tracks, observations))

    rows = {row["local_track_id"]: row for row in result["tracks"]}
    assert rows["CAM_001:TRACK_12"]["vehicle_id"] == rows["CAM_001:TRACK_27"]["vehicle_id"]
    assert rows["CAM_001:TRACK_27"]["reconciliation"]["matched"] is True
    assert result["metrics"]["accepted_matches"] == 1


def test_similar_colour_is_not_sufficient_for_merge(tmp_path: Path) -> None:
    tracks = [_track(12, first=0, last=2, colour="BLACK"), _track(27, first=5, last=7, colour="BLACK")]
    observations = _observations_for(12, [(0, 10), (1, 20), (2, 30)]) + _observations_for(27, [(5, 260), (6, 270), (7, 280)])
    result = run_track_reconciliation_experiment(_write_run(tmp_path, tracks, observations))

    assert result["metrics"]["accepted_matches"] == 0
    assert result["metrics"]["reconciled_vehicle_identities"] == 2


def test_different_class_is_hard_rejected(tmp_path: Path) -> None:
    tracks = [_track(12, first=0, last=2, vehicle_class="CAR"), _track(27, first=5, last=7, vehicle_class="MOTORCYCLE")]
    observations = _observations_for(12, [(0, 10), (1, 20), (2, 30)]) + _observations_for(27, [(5, 60), (6, 70), (7, 80)], raw_class="motorcycle")
    result = run_track_reconciliation_experiment(_write_run(tmp_path, tracks, observations))

    rejected = [item for item in result["attempts"] if item["rejection_reason"] == "conflicting_vehicle_class"]
    assert rejected
    assert result["metrics"]["accepted_matches"] == 0


def test_large_temporal_gap_is_rejected(tmp_path: Path) -> None:
    tracks = [_track(12, first=0, last=2), _track(27, first=35, last=37)]
    observations = _observations_for(12, [(0, 10), (1, 20), (2, 30)]) + _observations_for(27, [(35, 60), (36, 70), (37, 80)])
    result = run_track_reconciliation_experiment(_write_run(tmp_path, tracks, observations))

    assert any(item["rejection_reason"] == "time_gap_exceeds_window" for item in result["attempts"])
    assert result["metrics"]["accepted_matches"] == 0


def test_ambiguous_candidates_do_not_auto_merge(tmp_path: Path) -> None:
    tracks = [_track(12, first=0, last=2), _track(18, first=0, last=2), _track(27, first=5, last=7)]
    observations = (
        _observations_for(12, [(0, 10), (1, 20), (2, 30)])
        + _observations_for(18, [(0, 12), (1, 22), (2, 32)])
        + _observations_for(27, [(5, 60), (6, 70), (7, 80)])
    )
    result = run_track_reconciliation_experiment(_write_run(tmp_path, tracks, observations))

    rows = {row["local_track_id"]: row for row in result["tracks"]}
    assert rows["CAM_001:TRACK_27"]["reconciliation"]["matched"] is False
    assert rows["CAM_001:TRACK_27"]["reconciliation"]["reason"] == "ambiguous_candidate_margin"
    assert result["metrics"]["ambiguous_matches"] == 1


def test_colour_unavailable_still_allows_motion_based_recovery(tmp_path: Path) -> None:
    tracks = [_track(12, first=0, last=2, colour="UNKNOWN"), _track(27, first=5, last=7, colour="UNKNOWN")]
    observations = _observations_for(12, [(0, 10), (1, 20), (2, 30)]) + _observations_for(27, [(5, 60), (6, 70), (7, 80)])
    result = run_track_reconciliation_experiment(_write_run(tmp_path, tracks, observations))

    assert result["metrics"]["accepted_matches"] == 1
