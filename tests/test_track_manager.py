from __future__ import annotations

import csv
from pathlib import Path

import pytest

from src.models import (
    ConfigurationError,
    TRACK_STATUS_ACTIVE,
    TRACK_STATUS_DISCARDED,
    TRACK_STATUS_LOST,
    TRACK_STATUS_TENTATIVE,
    TrackedDetection,
)
from src.output_writer import RunOutputManager
from src.track_manager import (
    COMPLETION_REASON_END_OF_STREAM,
    COMPLETION_REASON_LOST_TIMEOUT,
    FINAL_CLASS_INSUFFICIENT_OBSERVATIONS,
    FINAL_CLASS_NO_CLEAR_WINNER,
    FINAL_CLASS_WEIGHTED_MAJORITY,
    FINAL_CLASS_WINNER_RATIO_TOO_LOW,
    TrackManager,
)


def _logger():
    import logging

    logger = logging.getLogger("track-manager-test")
    logger.handlers.clear()
    logger.propagate = False
    logger.addHandler(logging.NullHandler())
    return logger


def _config(*, minimum_observations: int = 3, maximum_lost_frames: int = 2) -> dict:
    return {
        "lifecycle": {
            "minimum_observations": minimum_observations,
            "maximum_lost_frames": maximum_lost_frames,
            "keep_discarded_tracks": True,
        },
        "track_class": {
            "minimum_observations": 3,
            "minimum_winner_ratio": 0.60,
            "strategy": "confidence_weighted_majority",
            "unknown_class_name": "UNKNOWN",
        },
    }


def _tracked(
    *,
    camera_id: str = "CAM_001",
    frame_number: int = 0,
    tracker_id: int = 1,
    confidence: float = 0.80,
    raw_class_name: str = "car",
    raw_class_id: int = 0,
    bbox_xyxy: tuple[float, float, float, float] = (1.0, 2.0, 10.0, 12.0),
) -> TrackedDetection:
    return TrackedDetection(
        camera_id=camera_id,
        tracker_namespace="camera",
        frame_number=frame_number,
        timestamp_seconds=frame_number / 10.0,
        tracker_id=tracker_id,
        bbox_xyxy=bbox_xyxy,
        confidence=confidence,
        raw_class_id=raw_class_id,
        raw_class_name=raw_class_name,
    )


def test_new_native_id_creates_tentative_track() -> None:
    manager = TrackManager(_config(minimum_observations=3), _logger())
    completed = manager.update_frame("CAM_001", 0, [_tracked(frame_number=0)])
    assert completed == []
    track = manager._tracks[("CAM_001", "camera", 1)]
    assert track.status == TRACK_STATUS_TENTATIVE
    assert track.observation_count == 1


def test_third_observation_activates_track_and_identity_includes_camera() -> None:
    manager = TrackManager(_config(minimum_observations=3), _logger())
    manager.update_frame("CAM_001", 0, [_tracked(frame_number=0)])
    manager.update_frame("CAM_001", 1, [_tracked(frame_number=1)])
    manager.update_frame("CAM_001", 2, [_tracked(frame_number=2)])
    track = manager._tracks[("CAM_001", "camera", 1)]
    assert track.status == TRACK_STATUS_ACTIVE
    assert track.local_track_id == "CAM_001:TRACK_1"


def test_same_native_id_in_different_cameras_creates_different_tracks() -> None:
    manager = TrackManager(_config(), _logger())
    manager.update_frame("CAM_001", 0, [_tracked(camera_id="CAM_001", frame_number=0, tracker_id=7)])
    manager.update_frame("CAM_002", 0, [_tracked(camera_id="CAM_002", frame_number=0, tracker_id=7)])
    assert ("CAM_001", "camera", 7) in manager._tracks
    assert ("CAM_002", "camera", 7) in manager._tracks
    assert manager._tracks[("CAM_001", "camera", 7)].local_track_id != manager._tracks[("CAM_002", "camera", 7)].local_track_id


def test_existing_track_appends_observations_and_count_is_correct() -> None:
    manager = TrackManager(_config(), _logger())
    manager.update_frame("CAM_001", 0, [_tracked(frame_number=0)])
    manager.update_frame("CAM_001", 1, [_tracked(frame_number=1)])
    track = manager._tracks[("CAM_001", "camera", 1)]
    assert len(track.observations) == 2
    assert track.observation_count == 2


def test_missing_track_becomes_lost_and_lost_count_increments_once_per_frame() -> None:
    manager = TrackManager(_config(), _logger())
    manager.update_frame("CAM_001", 0, [_tracked(frame_number=0)])
    manager.update_frame("CAM_001", 1, [])
    track = manager._tracks[("CAM_001", "camera", 1)]
    assert track.status == TRACK_STATUS_LOST
    assert track.lost_frames == 1
    manager.update_frame("CAM_001", 2, [])
    assert track.lost_frames == 2


def test_reappearing_track_resets_lost_count_and_becomes_active() -> None:
    manager = TrackManager(_config(), _logger())
    manager.update_frame("CAM_001", 0, [_tracked(frame_number=0)])
    manager.update_frame("CAM_001", 1, [_tracked(frame_number=1)])
    manager.update_frame("CAM_001", 2, [_tracked(frame_number=2)])
    manager.update_frame("CAM_001", 3, [])
    manager.update_frame("CAM_001", 4, [_tracked(frame_number=4)])
    track = manager._tracks[("CAM_001", "camera", 1)]
    assert track.status == TRACK_STATUS_ACTIVE
    assert track.lost_frames == 0


def test_track_completes_after_exact_lost_frame_boundary() -> None:
    manager = TrackManager(_config(minimum_observations=3, maximum_lost_frames=2), _logger())
    manager.update_frame("CAM_001", 0, [_tracked(frame_number=0)])
    manager.update_frame("CAM_001", 1, [_tracked(frame_number=1)])
    manager.update_frame("CAM_001", 2, [_tracked(frame_number=2)])
    assert manager.update_frame("CAM_001", 3, []) == []
    assert manager.update_frame("CAM_001", 4, []) == []
    completed = manager.update_frame("CAM_001", 5, [])
    assert completed[0].status == "COMPLETED"
    assert completed[0].completion_reason == COMPLETION_REASON_LOST_TIMEOUT


def test_short_track_becomes_discarded_and_flush_one_camera_isolated() -> None:
    manager = TrackManager(_config(minimum_observations=3), _logger())
    manager.update_frame("CAM_001", 0, [_tracked(camera_id="CAM_001", frame_number=0)])
    manager.update_frame("CAM_002", 0, [_tracked(camera_id="CAM_002", frame_number=0)])
    finalized = manager.flush_camera("CAM_001")
    assert finalized[0].status == TRACK_STATUS_DISCARDED
    assert ("CAM_002", "camera", 1) in manager._tracks


def test_flush_all_leaves_zero_active_tracks() -> None:
    manager = TrackManager(_config(), _logger())
    manager.update_frame("CAM_001", 0, [_tracked(frame_number=0)])
    manager.update_frame("CAM_001", 1, [_tracked(frame_number=1)])
    manager.update_frame("CAM_001", 2, [_tracked(frame_number=2)])
    manager.flush_all()
    assert manager.get_metrics()["active_tracks_at_shutdown"] == 0


def test_confidence_sums_class_counts_and_weighted_winner_are_correct() -> None:
    manager = TrackManager(_config(), _logger())
    manager.update_frame("CAM_001", 0, [_tracked(frame_number=0, raw_class_name="car", confidence=0.80)])
    manager.update_frame("CAM_001", 1, [_tracked(frame_number=1, raw_class_name="car", confidence=0.75)])
    manager.update_frame("CAM_001", 2, [_tracked(frame_number=2, raw_class_name="truck", confidence=0.52)])
    manager.update_frame("CAM_001", 3, [_tracked(frame_number=3, raw_class_name="car", confidence=0.70)])
    finalized = manager.flush_camera("CAM_001")
    track = finalized[0]
    assert track.class_counts == {"car": 3, "truck": 1}
    assert track.class_confidence_sums["car"] == pytest.approx(2.25)
    assert track.final_class == "car"
    assert track.final_class_reason == FINAL_CLASS_WEIGHTED_MAJORITY


def test_low_winner_ratio_insufficient_observations_and_tied_result_return_unknown() -> None:
    low_ratio = TrackManager(_config(), _logger())
    low_ratio.update_frame("CAM_001", 0, [_tracked(frame_number=0, raw_class_name="car", confidence=0.9)])
    low_ratio.update_frame("CAM_001", 1, [_tracked(frame_number=1, raw_class_name="car", confidence=0.8)])
    low_ratio.update_frame("CAM_001", 2, [_tracked(frame_number=2, raw_class_name="truck", confidence=0.7)])
    low_ratio.update_frame("CAM_001", 3, [_tracked(frame_number=3, raw_class_name="truck", confidence=0.6)])
    track_low_ratio = low_ratio.flush_camera("CAM_001")[0]
    assert track_low_ratio.final_class == "UNKNOWN"
    assert track_low_ratio.final_class_reason == FINAL_CLASS_WINNER_RATIO_TOO_LOW

    insufficient = TrackManager(_config(), _logger())
    insufficient.update_frame("CAM_001", 0, [_tracked(frame_number=0, raw_class_name="car", confidence=0.8)])
    insufficient.update_frame("CAM_001", 1, [_tracked(frame_number=1, raw_class_name="truck", confidence=0.82)])
    track_insufficient = insufficient.flush_camera("CAM_001")[0]
    assert track_insufficient.final_class == "UNKNOWN"
    assert track_insufficient.final_class_reason == FINAL_CLASS_INSUFFICIENT_OBSERVATIONS

    tied = TrackManager(_config(), _logger())
    tied.update_frame("CAM_001", 0, [_tracked(frame_number=0, raw_class_name="car", confidence=0.9)])
    tied.update_frame("CAM_001", 1, [_tracked(frame_number=1, raw_class_name="truck", confidence=0.95)])
    tied.update_frame("CAM_001", 2, [_tracked(frame_number=2, raw_class_name="car", confidence=0.2)])
    tied.update_frame("CAM_001", 3, [_tracked(frame_number=3, raw_class_name="truck", confidence=0.15)])
    track_tied = tied.flush_camera("CAM_001")[0]
    assert track_tied.final_class == "UNKNOWN"
    assert track_tied.final_class_reason == FINAL_CLASS_WINNER_RATIO_TOO_LOW


def test_raw_classes_remain_unchanged_and_duplicate_and_out_of_order_are_rejected() -> None:
    manager = TrackManager(_config(), _logger())
    manager.update_frame("CAM_001", 0, [_tracked(frame_number=0, raw_class_name="bus")])
    with pytest.raises(ConfigurationError):
        manager.update_frame("CAM_001", 0, [_tracked(frame_number=0, raw_class_name="bus")])
    with pytest.raises(ConfigurationError):
        manager.update_frame("CAM_001", -1, [])
    track = manager._tracks[("CAM_001", "camera", 1)]
    assert track.observations[0].raw_class_name == "bus"


def test_tracks_json_excludes_embedded_observations_and_observations_csv_contains_each_once(tmp_path: Path) -> None:
    manager = TrackManager(_config(), _logger())
    manager.update_frame("CAM_001", 0, [_tracked(frame_number=0)])
    manager.update_frame("CAM_001", 1, [_tracked(frame_number=1)])
    manager.update_frame("CAM_001", 2, [_tracked(frame_number=2)])
    manager.flush_camera("CAM_001")
    output = RunOutputManager(tmp_path)
    tracks = manager.get_all_output_tracks()
    observations = manager.get_all_observations()
    output.save_tracks(
        [
            {
                "local_track_id": track.local_track_id,
                "camera_id": track.camera_id,
                "native_tracker_id": track.native_tracker_id,
                "status": track.status,
                "first_frame": track.first_frame,
                "last_frame": track.last_frame,
                "first_timestamp_seconds": track.first_timestamp_seconds,
                "last_timestamp_seconds": track.last_timestamp_seconds,
                "observation_count": track.observation_count,
                "final_class": track.final_class,
                "final_class_reason": track.final_class_reason,
                "class_counts": track.class_counts,
                "class_confidence_sums": track.class_confidence_sums,
                "completion_reason": track.completion_reason,
            }
            for track in tracks
        ]
    )
    output.save_observations(
        [
            {
                "local_track_id": item.local_track_id,
                "camera_id": item.camera_id,
                "native_tracker_id": item.native_tracker_id,
                "frame_number": item.frame_number,
                "timestamp_seconds": item.timestamp_seconds,
                "x1": item.bbox_xyxy[0],
                "y1": item.bbox_xyxy[1],
                "x2": item.bbox_xyxy[2],
                "y2": item.bbox_xyxy[3],
                "confidence": item.confidence,
                "raw_class_id": item.raw_class_id,
                "raw_class_name": item.raw_class_name,
            }
            for item in observations
        ]
    )
    tracks_payload = Path(output.run_directory / "tracks.json").read_text(encoding="utf-8")
    assert '"observations"' not in tracks_payload
    with (output.run_directory / "observations.csv").open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 3
    assert len({(row["local_track_id"], row["frame_number"]) for row in rows}) == 3


def test_metrics_json_payload_is_created() -> None:
    manager = TrackManager(_config(), _logger())
    manager.update_frame("CAM_001", 0, [_tracked(frame_number=0)])
    manager.flush_camera("CAM_001")
    metrics = manager.get_metrics()
    assert "tracks_created_by_camera" in metrics
    assert "duplicate_observation_count" in metrics
