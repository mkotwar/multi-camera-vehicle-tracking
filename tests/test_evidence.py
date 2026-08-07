from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.evidence import (
    EVIDENCE_ROLE_BEST_OVERALL,
    EVIDENCE_ROLE_FIRST,
    EVIDENCE_ROLE_HIGHEST_CONFIDENCE,
    EVIDENCE_ROLE_LARGEST,
    EVIDENCE_ROLE_LAST,
    EVIDENCE_ROLE_MIDDLE,
    EVIDENCE_ROLE_SHARPEST,
    EvidenceCollector,
)
from src.logging_setup import setup_logging
from src.models import (
    FramePacket,
    LocalTrack,
    PipelineRuntimeError,
    TrackObservation,
    TrackedDetection,
)
from src.output_writer import RunOutputManager


def _build_config(**evidence_overrides):
    evidence = {
        "enabled": True,
        "collect_first": True,
        "collect_middle": True,
        "collect_last": True,
        "collect_highest_confidence": True,
        "collect_largest": True,
        "collect_sharpest": True,
        "collect_best_overall": True,
        "maximum_candidates_per_track": 7,
        "minimum_crop_width_pixels": 10,
        "minimum_crop_height_pixels": 10,
        "crop_padding_ratio_x": 0.0,
        "crop_padding_ratio_y": 0.0,
        "minimum_padding_pixels": 0,
        "clamp_bbox_to_frame": True,
        "reject_invalid_bbox": True,
        "sharpness_enabled": True,
        "best_overall_weights": {
            "confidence": 0.35,
            "sharpness": 0.25,
            "bbox_area": 0.20,
            "centeredness": 0.10,
            "edge_visibility": 0.10,
        },
        "jpeg_quality": 90,
        "save_vehicle_crops": True,
        "save_annotated_full_frames": True,
        "save_all_candidates": False,
        "include_discarded_tracks": False,
        "fail_pipeline_on_error": False,
    }
    evidence.update(evidence_overrides)
    return {
        "evidence": evidence,
        "lifecycle": {"minimum_observations": 3},
    }


def _make_frame(width: int = 120, height: int = 80, *, sharp: bool = False, fill: int = 80) -> np.ndarray:
    frame = np.full((height, width, 3), fill, dtype=np.uint8)
    if sharp:
        for x in range(0, width, 4):
            frame[:, x : x + 2] = 255
    return frame


def _packet(frame_number: int, frame: np.ndarray, *, camera_id: str = "CAM_001") -> FramePacket:
    return FramePacket(
        camera_id=camera_id,
        frame_number=frame_number,
        timestamp_seconds=frame_number / 10.0,
        source_fps=10.0,
        frame=frame,
        source_frame_width=int(frame.shape[1]),
        source_frame_height=int(frame.shape[0]),
        worker_id=0,
        captured_at="2026-07-30T00:00:00+00:00",
        source_type="video",
    )


def _tracked(
    frame_number: int,
    *,
    bbox_xyxy=(20.0, 20.0, 60.0, 60.0),
    confidence: float = 0.8,
    raw_class_name: str = "car",
    tracker_id: int = 1,
    tracker_namespace: str = "camera",
    camera_id: str = "CAM_001",
) -> TrackedDetection:
    return TrackedDetection(
        camera_id=camera_id,
        tracker_namespace=tracker_namespace,
        frame_number=frame_number,
        timestamp_seconds=frame_number / 10.0,
        tracker_id=tracker_id,
        bbox_xyxy=bbox_xyxy,
        confidence=confidence,
        raw_class_id=0,
        raw_class_name=raw_class_name,
    )


def _observation(
    frame_number: int,
    *,
    bbox_xyxy=(20.0, 20.0, 60.0, 60.0),
    confidence: float = 0.8,
    raw_class_name: str = "car",
    tracker_namespace: str = "camera",
    camera_id: str = "CAM_001",
    tracker_id: int = 1,
    local_track_id: str = "CAM_001:TRACK_1",
) -> TrackObservation:
    return TrackObservation(
        camera_id=camera_id,
        tracker_namespace=tracker_namespace,
        native_tracker_id=tracker_id,
        local_track_id=local_track_id,
        frame_number=frame_number,
        timestamp_seconds=frame_number / 10.0,
        bbox_xyxy=bbox_xyxy,
        confidence=confidence,
        raw_class_id=0,
        raw_class_name=raw_class_name,
    )


def _track(
    observations: list[TrackObservation],
    *,
    status: str = "COMPLETED",
    local_track_id: str = "CAM_001:TRACK_1",
    tracker_namespace: str = "camera",
    camera_id: str = "CAM_001",
    tracker_id: int = 1,
    final_class: str = "car",
    completion_reason: str = "END_OF_STREAM",
) -> LocalTrack:
    return LocalTrack(
        local_track_id=local_track_id,
        camera_id=camera_id,
        tracker_namespace=tracker_namespace,
        native_tracker_id=tracker_id,
        status=status,
        first_frame=observations[0].frame_number,
        last_frame=observations[-1].frame_number,
        first_timestamp_seconds=observations[0].timestamp_seconds,
        last_timestamp_seconds=observations[-1].timestamp_seconds,
        observation_count=len(observations),
        lost_frames=0,
        final_class=final_class,
        final_class_reason="WEIGHTED_MAJORITY",
        class_counts={final_class: len(observations)},
        class_confidence_sums={final_class: sum(item.confidence for item in observations)},
        observations=observations,
        completion_reason=completion_reason,
    )


def _collector(tmp_path: Path, **evidence_overrides) -> tuple[EvidenceCollector, RunOutputManager]:
    output_manager = RunOutputManager(tmp_path)
    logger = setup_logging(output_manager.run_directory, log_level="INFO")
    collector = EvidenceCollector(_build_config(**evidence_overrides), logger, output_manager)
    return collector, output_manager


def test_register_and_finalize_track_creates_evidence_files_and_reuses_paths(tmp_path: Path) -> None:
    collector, output_manager = _collector(tmp_path)
    frame = _make_frame(sharp=True)
    packet = _packet(0, frame)
    tracked = _tracked(0, confidence=0.92, raw_class_name="motorcycle")
    collector.register_frame(packet, [tracked])
    track = _track([_observation(0, confidence=0.92, raw_class_name="motorcycle")], final_class="motorcycle")

    evidence = collector.finalize_track(track)

    assert {item.role for item in evidence} == {
        EVIDENCE_ROLE_FIRST,
        EVIDENCE_ROLE_MIDDLE,
        EVIDENCE_ROLE_LAST,
        EVIDENCE_ROLE_HIGHEST_CONFIDENCE,
        EVIDENCE_ROLE_LARGEST,
        EVIDENCE_ROLE_SHARPEST,
        EVIDENCE_ROLE_BEST_OVERALL,
    }
    crop_paths = {item.crop_path for item in evidence}
    annotated_paths = {item.annotated_frame_path for item in evidence}
    assert len(crop_paths) == 1
    assert len(annotated_paths) == 1
    evidence_json = output_manager.evidence_directory / "CAM_001" / "CAM_001_TRACK_1" / "evidence.json"
    payload = json.loads(evidence_json.read_text(encoding="utf-8"))
    assert payload[0]["local_track_id"] == "CAM_001:TRACK_1"
    assert payload[0]["tracker_namespace"] == "camera"
    assert payload[0]["raw_class_name"] == "motorcycle"
    assert payload[0]["final_class"] == "motorcycle"
    assert collector.metrics["cache_frames_released"] == 1
    assert collector._frame_cache == {}
    assert collector.metrics["cache_release_attempts"] >= 1


def test_middle_highest_largest_sharpest_and_best_overall_selection_are_correct(tmp_path: Path) -> None:
    collector, _output_manager = _collector(
        tmp_path,
        best_overall_weights={
            "confidence": 0.0,
            "sharpness": 2.0,
            "bbox_area": 0.0,
            "centeredness": 0.0,
            "edge_visibility": 0.0,
        },
    )
    frames = {
        0: _make_frame(fill=30),
        4: _make_frame(fill=60),
        9: _make_frame(sharp=True),
    }
    detections = {
        0: _tracked(0, bbox_xyxy=(10.0, 10.0, 40.0, 40.0), confidence=0.55),
        4: _tracked(4, bbox_xyxy=(20.0, 20.0, 75.0, 75.0), confidence=0.95),
        9: _tracked(9, bbox_xyxy=(25.0, 25.0, 50.0, 50.0), confidence=0.70),
    }
    observations = []
    for frame_number in (0, 4, 9):
        collector.register_frame(_packet(frame_number, frames[frame_number]), [detections[frame_number]])
        observations.append(
            _observation(
                frame_number,
                bbox_xyxy=detections[frame_number].bbox_xyxy,
                confidence=detections[frame_number].confidence,
            )
        )
    evidence = collector.finalize_track(_track(observations))
    by_role = {item.role: item for item in evidence}

    assert by_role[EVIDENCE_ROLE_FIRST].frame_number == 0
    assert by_role[EVIDENCE_ROLE_MIDDLE].frame_number == 4
    assert by_role[EVIDENCE_ROLE_LAST].frame_number == 9
    assert by_role[EVIDENCE_ROLE_HIGHEST_CONFIDENCE].frame_number == 4
    assert by_role[EVIDENCE_ROLE_LARGEST].frame_number == 4
    assert by_role[EVIDENCE_ROLE_SHARPEST].frame_number == 9
    assert by_role[EVIDENCE_ROLE_BEST_OVERALL].frame_number == 9


def test_invalid_bbox_small_crop_and_empty_crop_are_rejected_without_crashing(tmp_path: Path) -> None:
    collector, _output_manager = _collector(
        tmp_path,
        minimum_crop_width_pixels=20,
        minimum_crop_height_pixels=20,
    )
    frame = _make_frame(width=40, height=40)
    collector.register_frame(_packet(0, frame), [_tracked(0, bbox_xyxy=(15.0, 15.0, 10.0, 25.0))])


def test_capture_zone_captures_when_bottom_center_enters_zone(tmp_path: Path) -> None:
    collector, output_manager = _collector(
        tmp_path,
        capture_zone={
            "enabled": True,
            "top_ratio": 0.50,
            "bottom_ratio": 0.80,
            "trigger_point": "bottom_center",
            "maximum_saved_candidates_per_track": 3,
            "minimum_frame_gap": 1,
            "save_immediately": True,
            "require_confirmed_track": False,
            "minimum_bbox_width_pixels": 10,
            "minimum_bbox_height_pixels": 10,
        },
    )
    frame = _make_frame(width=120, height=100, sharp=True)
    collector.register_frame(_packet(0, frame), [_tracked(0, bbox_xyxy=(20.0, 10.0, 60.0, 40.0))])
    collector.register_frame(_packet(1, frame), [_tracked(1, bbox_xyxy=(20.0, 30.0, 60.0, 65.0))])
    track = _track([_observation(0, bbox_xyxy=(20.0, 10.0, 60.0, 40.0)), _observation(1, bbox_xyxy=(20.0, 30.0, 60.0, 65.0))])

    evidence = collector.finalize_track(track)

    zone_records = [item for item in evidence if isinstance(item, dict) and item.get("evidence_source") == "capture_zone"]
    assert len(zone_records) == 1
    assert Path(zone_records[0]["crop_path"]).is_file()
    assert zone_records[0]["trigger_y"] == pytest.approx(65.0)
    assert (output_manager.evidence_capture_zone_directory / "CAM_001" / "CAM_001_TRACK_1" / "frame_000001.jpg").exists()


def test_capture_zone_respects_minimum_frame_gap_and_maximum_candidates(tmp_path: Path) -> None:
    collector, _output_manager = _collector(
        tmp_path,
        capture_zone={
            "enabled": True,
            "top_ratio": 0.40,
            "bottom_ratio": 0.90,
            "trigger_point": "bottom_center",
            "maximum_saved_candidates_per_track": 2,
            "minimum_frame_gap": 2,
            "save_immediately": True,
            "require_confirmed_track": False,
            "minimum_bbox_width_pixels": 10,
            "minimum_bbox_height_pixels": 10,
        },
    )
    frame = _make_frame(width=120, height=100, sharp=True)
    for frame_number in range(5):
        collector.register_frame(_packet(frame_number, frame), [_tracked(frame_number, bbox_xyxy=(20.0, 30.0, 70.0, 75.0), confidence=0.6 + (frame_number * 0.05))])
    track = _track([_observation(frame_number, bbox_xyxy=(20.0, 30.0, 70.0, 75.0), confidence=0.6 + (frame_number * 0.05)) for frame_number in range(5)])

    evidence = collector.finalize_track(track)

    zone_records = [item for item in evidence if isinstance(item, dict) and item.get("evidence_source") == "capture_zone"]
    assert len(zone_records) <= 2
    assert collector.metrics["capture_zone_duplicate_frame_suppressed"] >= 1


def test_capture_zone_saved_crop_survives_without_frame_cache(tmp_path: Path) -> None:
    collector, _output_manager = _collector(
        tmp_path,
        capture_zone={
            "enabled": True,
            "top_ratio": 0.40,
            "bottom_ratio": 0.90,
            "trigger_point": "bottom_center",
            "maximum_saved_candidates_per_track": 3,
            "minimum_frame_gap": 1,
            "save_immediately": True,
            "require_confirmed_track": False,
            "minimum_bbox_width_pixels": 10,
            "minimum_bbox_height_pixels": 10,
        },
    )
    frame = _make_frame(width=120, height=100, sharp=True)
    collector.register_frame(_packet(0, frame), [_tracked(0, bbox_xyxy=(20.0, 30.0, 70.0, 75.0))])
    collector._frame_cache.clear()

    evidence = collector.finalize_track(_track([_observation(0, bbox_xyxy=(20.0, 30.0, 70.0, 75.0))]))

    zone_records = [item for item in evidence if isinstance(item, dict) and item.get("evidence_source") == "capture_zone"]
    assert zone_records
    assert Path(zone_records[0]["crop_path"]).is_file()


def test_capture_zone_cleanup_removes_track_state(tmp_path: Path) -> None:
    collector, _output_manager = _collector(
        tmp_path,
        capture_zone={
            "enabled": True,
            "top_ratio": 0.40,
            "bottom_ratio": 0.90,
            "trigger_point": "bottom_center",
            "maximum_saved_candidates_per_track": 3,
            "minimum_frame_gap": 1,
            "save_immediately": True,
            "require_confirmed_track": False,
            "minimum_bbox_width_pixels": 10,
            "minimum_bbox_height_pixels": 10,
        },
    )
    frame = _make_frame(width=120, height=100, sharp=True)
    collector.register_frame(_packet(0, frame), [_tracked(0, bbox_xyxy=(20.0, 30.0, 70.0, 75.0))])

    collector.finalize_track(_track([_observation(0, bbox_xyxy=(20.0, 30.0, 70.0, 75.0))]))

    assert collector.metrics["capture_zone_active_tracks"] == 0
    collector.register_frame(_packet(1, frame), [_tracked(1, bbox_xyxy=(1.0, 1.0, 8.0, 8.0))])
    collector.register_frame(
        _packet(2, frame),
        [_tracked(2, bbox_xyxy=(0.0, 0.0, 0.0, 0.0), confidence=0.7)],
    )

    assert collector.metrics["invalid_candidates"] == 2
    assert collector.finalize_track(_track([_observation(0)], final_class="car")) == []


def test_capture_zone_prefers_later_larger_motorcycle_crop(tmp_path: Path) -> None:
    collector, _output_manager = _collector(
        tmp_path,
        capture_zone={
            "enabled": True,
            "top_ratio": 0.40,
            "bottom_ratio": 0.90,
            "trigger_point": "bottom_center",
            "maximum_saved_candidates_per_track": 2,
            "minimum_frame_gap": 1,
            "save_immediately": True,
            "require_confirmed_track": False,
            "minimum_bbox_width_pixels": 10,
            "minimum_bbox_height_pixels": 10,
        },
    )
    collector._vehicle_enrichment_evidence_config = {
        "minimum_crop_width": 100,
        "minimum_crop_height": 70,
        "class_specific_minimums": {"motorcycle": {"minimum_crop_width": 120, "minimum_crop_height": 120}},
    }
    collector._vehicle_enrichment_florence_config = {
        "default": {"minimum_original_width": 192, "minimum_original_height": 144},
        "class_specific": {"motorcycle": {"minimum_original_width": 120, "minimum_original_height": 120}},
    }
    frame = _make_frame(width=300, height=220, sharp=True)
    collector.register_frame(_packet(0, frame), [_tracked(0, bbox_xyxy=(20.0, 70.0, 100.0, 173.0), raw_class_name="motorcycle")])
    collector.register_frame(_packet(1, frame), [_tracked(1, bbox_xyxy=(20.0, 40.0, 150.0, 185.0), raw_class_name="motorcycle")])

    evidence = collector.finalize_track(
        _track(
            [
                _observation(0, bbox_xyxy=(20.0, 70.0, 100.0, 173.0), raw_class_name="motorcycle"),
                _observation(1, bbox_xyxy=(20.0, 40.0, 150.0, 185.0), raw_class_name="motorcycle"),
            ],
            final_class="motorcycle",
        )
    )

    zone_records = [item for item in evidence if isinstance(item, dict) and item.get("evidence_source") == "capture_zone"]
    assert zone_records
    assert max(record["original_crop_width"] for record in zone_records) >= 130
    assert any(record["evidence_eligible"] is True for record in zone_records)


def test_motorcycle_geometry_reports_never_reached_zone(tmp_path: Path) -> None:
    collector, _output_manager = _collector(
        tmp_path,
        capture_zone={
            "enabled": True,
            "top_ratio": 0.60,
            "bottom_ratio": 0.85,
            "require_confirmed_track": False,
        },
    )
    frame = _make_frame(width=200, height=200, sharp=True)
    collector.register_frame(_packet(0, frame), [_tracked(0, bbox_xyxy=(20.0, 20.0, 70.0, 90.0), raw_class_name="motorcycle")])
    collector.finalize_track(_track([_observation(0, bbox_xyxy=(20.0, 20.0, 70.0, 90.0), raw_class_name="motorcycle")], final_class="motorcycle"))

    row = collector.motorcycle_geometry_records[0]
    assert row["max_trigger_y"] == pytest.approx(90.0)
    assert row["zone_top"] == 120
    assert row["geometry_status"] == "NEVER_REACHED_ZONE"
    assert row["geometry_reason"] == "max_bottom_center_above_zone"


def test_motorcycle_geometry_reports_lost_before_zone(tmp_path: Path) -> None:
    collector, _output_manager = _collector(
        tmp_path,
        capture_zone={
            "enabled": True,
            "top_ratio": 0.70,
            "bottom_ratio": 0.90,
            "require_confirmed_track": False,
        },
    )
    frame = _make_frame(width=200, height=200, sharp=True)
    collector.register_frame(_packet(0, frame), [_tracked(0, bbox_xyxy=(20.0, 30.0, 70.0, 100.0), raw_class_name="motorcycle")])
    collector.finalize_track(
        _track(
            [_observation(0, bbox_xyxy=(20.0, 30.0, 70.0, 100.0), raw_class_name="motorcycle")],
            final_class="motorcycle",
            completion_reason="LOST_TIMEOUT",
        )
    )

    row = collector.motorcycle_geometry_records[0]
    assert row["geometry_status"] == "TRACK_ENDED_BEFORE_ZONE"
    assert row["geometry_reason"] == "lost_timeout_before_zone"


def test_motorcycle_specific_zone_is_used_after_class_becomes_stable(tmp_path: Path) -> None:
    collector, _output_manager = _collector(
        tmp_path,
        capture_zone={
            "enabled": True,
            "default": {"top_ratio": 0.70, "bottom_ratio": 0.90, "require_confirmed_track": False},
            "class_specific": {
                "motorcycle": {"top_ratio": 0.50, "bottom_ratio": 0.80, "require_confirmed_track": False},
            },
        },
    )
    frame = _make_frame(width=200, height=200, sharp=True)
    collector.register_frame(_packet(0, frame), [_tracked(0, bbox_xyxy=(20.0, 20.0, 70.0, 110.0), raw_class_name="motorcycle")])
    collector.register_frame(_packet(1, frame), [_tracked(1, bbox_xyxy=(20.0, 20.0, 70.0, 130.0), raw_class_name="motorcycle")])
    collector.register_frame(_packet(2, frame), [_tracked(2, bbox_xyxy=(20.0, 20.0, 70.0, 145.0), raw_class_name="motorcycle")])
    collector.finalize_track(
        _track(
            [
                _observation(0, bbox_xyxy=(20.0, 20.0, 70.0, 110.0), raw_class_name="motorcycle"),
                _observation(1, bbox_xyxy=(20.0, 20.0, 70.0, 130.0), raw_class_name="motorcycle"),
                _observation(2, bbox_xyxy=(20.0, 20.0, 70.0, 145.0), raw_class_name="motorcycle"),
            ],
            final_class="motorcycle",
        )
    )

    row = collector.motorcycle_geometry_records[0]
    assert row["first_zone_entry_frame"] == 2
    assert row["zone_top"] == 100
    assert row["entered_zone"] is True


def test_capture_zone_geometry_report_is_saved_by_output_manager(tmp_path: Path) -> None:
    collector, output_manager = _collector(
        tmp_path,
        capture_zone={
            "enabled": True,
            "top_ratio": 0.50,
            "bottom_ratio": 0.80,
            "require_confirmed_track": False,
        },
    )
    frame = _make_frame(width=160, height=160, sharp=True)
    collector.register_frame(_packet(0, frame), [_tracked(0, bbox_xyxy=(20.0, 20.0, 80.0, 120.0), raw_class_name="motorcycle")])
    collector.finalize_track(_track([_observation(0, bbox_xyxy=(20.0, 20.0, 80.0, 120.0), raw_class_name="motorcycle")], final_class="motorcycle"))

    path = output_manager.save_motorcycle_geometry_report(collector.motorcycle_geometry_records)

    payload = path.read_text(encoding="utf-8")
    assert path.name == "motorcycle_geometry_report.csv"
    assert "local_track_id" in payload
    assert "CAM_001:TRACK_1" in payload


def test_track_192_style_case_reports_never_reached_zone(tmp_path: Path) -> None:
    collector, _output_manager = _collector(
        tmp_path,
        capture_zone={
            "enabled": True,
            "default": {"top_ratio": 0.68, "bottom_ratio": 0.85, "require_confirmed_track": False},
            "class_specific": {
                "motorcycle": {"top_ratio": 0.68, "bottom_ratio": 0.85, "require_confirmed_track": False},
            },
        },
    )
    frame = _make_frame(width=640, height=720, sharp=True)
    observations = []
    for frame_number, y2 in enumerate((420.625, 426.875, 433.125, 443.75, 444.0625), start=1152):
        bbox = (200.0, float(y2 - 100.0), 280.0, float(y2))
        collector.register_frame(_packet(frame_number, frame), [_tracked(frame_number, bbox_xyxy=bbox, raw_class_name="motorcycle", tracker_id=192)])
        observations.append(_observation(frame_number, bbox_xyxy=bbox, raw_class_name="motorcycle", local_track_id="CAM_001:TRACK_192", tracker_id=192))
    collector.finalize_track(
        _track(
            observations,
            local_track_id="CAM_001:TRACK_192",
            tracker_id=192,
            final_class="motorcycle",
            completion_reason="LOST_TIMEOUT",
        )
    )

    row = collector.motorcycle_geometry_records[0]
    assert row["local_track_id"] == "CAM_001:TRACK_192"
    assert row["max_trigger_y"] == pytest.approx(444.0625)
    assert row["zone_top"] == 489
    assert row["entered_zone"] is False
    assert row["geometry_status"] == "TRACK_ENDED_BEFORE_ZONE"


def test_padding_is_clamped_to_frame_and_paths_are_separated_by_camera_and_track(tmp_path: Path) -> None:
    collector, output_manager = _collector(
        tmp_path,
        crop_padding_ratio_x=0.5,
        crop_padding_ratio_y=0.5,
        minimum_padding_pixels=10,
    )
    frame = _make_frame(width=50, height=50, sharp=True)
    collector.register_frame(_packet(0, frame, camera_id="CAM_001"), [_tracked(0, bbox_xyxy=(0.0, 0.0, 20.0, 20.0), camera_id="CAM_001")])
    collector.register_frame(
        _packet(0, frame, camera_id="CAM_002"),
        [_tracked(0, bbox_xyxy=(5.0, 5.0, 25.0, 25.0), camera_id="CAM_002", tracker_namespace="car", tracker_id=9)],
    )

    track_one = _track([_observation(0)], camera_id="CAM_001", local_track_id="CAM_001:TRACK_1")
    track_two = _track(
        [_observation(0, camera_id="CAM_002", tracker_namespace="car", tracker_id=9, local_track_id="CAM_002:CAR:TRACK_9")],
        camera_id="CAM_002",
        tracker_namespace="car",
        tracker_id=9,
        local_track_id="CAM_002:CAR:TRACK_9",
    )

    evidence_one = collector.finalize_track(track_one)
    evidence_two = collector.finalize_track(track_two)

    assert Path(evidence_one[0].crop_path).is_file()
    assert Path(evidence_two[0].crop_path).is_file()
    assert "CAM_001_TRACK_1" in evidence_one[0].crop_path
    assert "CAM_002_CAR_TRACK_9" in evidence_two[0].crop_path
    assert (output_manager.evidence_directory / "CAM_001" / "CAM_001_TRACK_1").exists()
    assert (output_manager.evidence_directory / "CAM_002" / "CAM_002_CAR_TRACK_9").exists()


def test_discarded_tracks_are_excluded_by_default_and_can_be_included(tmp_path: Path) -> None:
    collector, _output_manager = _collector(tmp_path)
    frame = _make_frame()
    collector.register_frame(_packet(0, frame), [_tracked(0)])
    discarded_track = _track([_observation(0)], status="DISCARDED", final_class="UNKNOWN")

    assert collector.finalize_track(discarded_track) == []

    collector_included, _output_manager_included = _collector(tmp_path / "included", include_discarded_tracks=True)
    collector_included.register_frame(_packet(0, frame), [_tracked(0)])
    evidence = collector_included.finalize_track(discarded_track)

    assert evidence
    assert collector_included.metrics["tracks_with_evidence"] == 1


def test_write_errors_are_logged_when_non_strict_and_raised_when_strict(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    collector, output_manager = _collector(tmp_path, fail_pipeline_on_error=False)
    frame = _make_frame()
    collector.register_frame(_packet(0, frame), [_tracked(0)])
    track = _track([_observation(0)])

    def _fail_crop(*args, **kwargs):
        raise OSError("crop write failed")

    monkeypatch.setattr(output_manager, "save_evidence_crop", _fail_crop)
    evidence = collector.finalize_track(track)
    assert evidence
    assert collector.metrics["errors"]
    assert collector.metrics["errors"][0]["error_class"] == "OSError"
    assert collector.metrics["errors"][0]["camera_id"] == "CAM_001"

    strict_collector, strict_output_manager = _collector(tmp_path / "strict", fail_pipeline_on_error=True)
    strict_collector.register_frame(_packet(0, frame), [_tracked(0)])
    monkeypatch.setattr(strict_output_manager, "save_evidence_crop", _fail_crop)
    with pytest.raises(PipelineRuntimeError):
        strict_collector.finalize_track(track)


def test_exception_logging_captures_active_traceback_context(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    collector, output_manager = _collector(tmp_path, fail_pipeline_on_error=False)
    frame = _make_frame()
    collector.register_frame(_packet(0, frame), [_tracked(0)])
    track = _track([_observation(0)])
    logged = {}

    def _capture(message, *args, **kwargs):
        logged["message"] = message
        logged["args"] = args
        logged["exc_info"] = kwargs.get("exc_info")

    def _fail_crop(*args, **kwargs):
        raise OSError("crop write failed")

    monkeypatch.setattr(output_manager, "save_evidence_crop", _fail_crop)
    monkeypatch.setattr(collector.logger, "error", _capture)

    collector.finalize_track(track)

    assert "EvidenceCollector error" in logged["message"]
    exc_info = logged["exc_info"]
    assert exc_info is not None and exc_info is not False
    assert exc_info[0] is OSError
    assert str(exc_info[1]) == "crop write failed"
    assert exc_info[2] is not None


def test_frame_referenced_by_pending_other_track_is_not_released_early(tmp_path: Path) -> None:
    collector, _output_manager = _collector(tmp_path)
    frame = _make_frame(sharp=True)
    packet = _packet(5, frame)
    collector.register_frame(
        packet,
        [
            _tracked(5, tracker_id=1),
            _tracked(5, tracker_id=2),
        ],
    )
    track_one = _track([_observation(5, tracker_id=1, local_track_id="CAM_001:TRACK_1")], local_track_id="CAM_001:TRACK_1", tracker_id=1)
    track_two = _track([_observation(5, tracker_id=2, local_track_id="CAM_001:TRACK_2")], local_track_id="CAM_001:TRACK_2", tracker_id=2)

    collector.finalize_track(track_one)

    assert ("CAM_001", 5) in collector._frame_cache
    assert collector.metrics["cache_release_deferred"] >= 1

    collector.finalize_track(track_two)

    assert ("CAM_001", 5) not in collector._frame_cache


def test_finalize_tracks_batch_keeps_shared_frame_until_batch_completes(tmp_path: Path) -> None:
    collector, _output_manager = _collector(tmp_path)
    frame = _make_frame(sharp=True)
    packet = _packet(12, frame)
    collector.register_frame(packet, [_tracked(12, tracker_id=1), _tracked(12, tracker_id=2)])
    track_one = _track([_observation(12, tracker_id=1, local_track_id="CAM_001:TRACK_1")], local_track_id="CAM_001:TRACK_1", tracker_id=1)
    track_two = _track([_observation(12, tracker_id=2, local_track_id="CAM_001:TRACK_2")], local_track_id="CAM_001:TRACK_2", tracker_id=2)
    observed_cache_presence: list[bool] = []
    original_save = collector._save_selected_assets

    def _wrapped_save(*args, **kwargs):
        observed_cache_presence.append(("CAM_001", 12) in collector._frame_cache)
        return original_save(*args, **kwargs)

    collector._save_selected_assets = _wrapped_save  # type: ignore[method-assign]

    evidence = collector.finalize_tracks([track_one, track_two])

    assert evidence
    assert observed_cache_presence == [True, True]
    assert ("CAM_001", 12) not in collector._frame_cache


def test_missing_frame_skips_only_affected_evidence_item_and_records_metrics(tmp_path: Path) -> None:
    collector, _output_manager = _collector(tmp_path, fail_pipeline_on_error=False)
    collector.register_frame(_packet(0, _make_frame(fill=40)), [_tracked(0, confidence=0.95)])
    collector.register_frame(_packet(1, _make_frame(sharp=True, fill=90)), [_tracked(1, confidence=0.70)])
    track = _track([
        _observation(0, confidence=0.95),
        _observation(1, confidence=0.70),
    ])
    collector._frame_cache.pop(("CAM_001", 0), None)

    evidence = collector.finalize_track(track)

    assert evidence
    assert any(item.crop_path is None for item in evidence if item.frame_number == 0)
    assert any(item.crop_path is not None for item in evidence if item.frame_number == 1)
    assert collector.metrics["missing_cache_frame_count"] >= 1
    assert collector.metrics["evidence_items_skipped_missing_frame"] >= 1
    assert collector.metrics["tracks_with_partial_evidence"] >= 1
    assert collector.metrics["errors"][0]["error_type"] == "missing_frame_from_cache"


def test_pending_evidence_tracks_metric_reflects_shutdown_state(tmp_path: Path) -> None:
    collector, _output_manager = _collector(tmp_path)
    collector.register_frame(_packet(0, _make_frame()), [_tracked(0)])

    assert collector.metrics["pending_evidence_tracks_at_shutdown"] == 1
    assert collector.metrics["pending_frame_reference_count"] == 1


def test_duplicate_same_frame_candidate_is_skipped_without_double_counting(tmp_path: Path) -> None:
    collector, _output_manager = _collector(tmp_path)
    packet = _packet(94, _make_frame())
    detection = _tracked(94, tracker_id=30)

    collector.register_frame(packet, [detection])
    collector.register_frame(packet, [detection])

    assert collector.metrics["duplicate_frame_candidates_skipped"] == 1
    assert collector._frame_ref_counts[("CAM_001", 94)] == 1
    assert len(collector._track_candidates["CAM_001:TRACK_30"]) == 1


def test_same_frame_replacement_keeps_cache_and_final_evidence_succeeds(tmp_path: Path) -> None:
    collector, _output_manager = _collector(
        tmp_path,
        deduplicate_similar_crops=True,
        duplicate_iou_threshold=0.80,
        minimum_frame_gap=2,
    )
    frame = _make_frame(width=120, height=80, sharp=True)
    packet = _packet(120, frame)
    original = collector._build_candidate(packet, _tracked(120, bbox_xyxy=(20.0, 20.0, 60.0, 60.0), confidence=0.60))
    replacement = collector._build_candidate(packet, _tracked(120, bbox_xyxy=(20.0, 20.0, 62.0, 62.0), confidence=0.95))

    assert original is not None and replacement is not None

    track_id = original.candidate.local_track_id
    collector._frame_cache[("CAM_001", 120)] = frame.copy()
    collector._frame_ref_counts[("CAM_001", 120)] = 1
    collector._track_candidates[track_id] = [original]

    retained = collector._retain_candidate(track_id, replacement)
    assert retained is True
    collector._ensure_frame_cached(("CAM_001", 120), frame)
    collector._frame_ref_counts[("CAM_001", 120)] = collector._frame_ref_counts.get(("CAM_001", 120), 0) + 1

    track = _track([_observation(120)], local_track_id=track_id)
    evidence = collector.finalize_track(track)

    assert evidence
    assert any(item.frame_number == 120 and item.crop_path is not None for item in evidence)
    assert collector.metrics["missing_cache_frame_count"] == 0
    assert ("CAM_001", 120) not in collector._frame_cache


def test_different_frame_replacement_evicts_unreferenced_old_frame(tmp_path: Path) -> None:
    collector, _output_manager = _collector(
        tmp_path,
        deduplicate_similar_crops=True,
        duplicate_iou_threshold=0.80,
        minimum_frame_gap=30,
    )
    frame_old = _make_frame(width=120, height=80, fill=40)
    frame_new = _make_frame(width=120, height=80, sharp=True, fill=90)
    packet_old = _packet(120, frame_old)
    packet_new = _packet(121, frame_new)
    original = collector._build_candidate(packet_old, _tracked(120, bbox_xyxy=(20.0, 20.0, 60.0, 60.0), confidence=0.60))
    replacement = collector._build_candidate(packet_new, _tracked(121, bbox_xyxy=(20.0, 20.0, 62.0, 62.0), confidence=0.95))

    assert original is not None and replacement is not None

    track_id = original.candidate.local_track_id
    collector._frame_cache[("CAM_001", 120)] = frame_old.copy()
    collector._frame_ref_counts[("CAM_001", 120)] = 1
    collector._track_candidates[track_id] = [original]

    retained = collector._retain_candidate(track_id, replacement)
    assert retained is True
    collector._ensure_frame_cached(("CAM_001", 121), frame_new)
    collector._frame_ref_counts[("CAM_001", 121)] = collector._frame_ref_counts.get(("CAM_001", 121), 0) + 1

    assert ("CAM_001", 120) not in collector._frame_cache
    assert ("CAM_001", 121) in collector._frame_cache
    assert collector.metrics["evidence_cache_evictions"] >= 1


def test_same_frame_number_across_cameras_remains_isolated_on_release(tmp_path: Path) -> None:
    collector, _output_manager = _collector(tmp_path)
    frame = _make_frame(sharp=True)
    collector.register_frame(_packet(120, frame, camera_id="CAM_001"), [_tracked(120, camera_id="CAM_001", tracker_id=1)])
    collector.register_frame(_packet(120, frame, camera_id="CAM_002"), [_tracked(120, camera_id="CAM_002", tracker_id=1)])

    track_one = _track([_observation(120, camera_id="CAM_001", local_track_id="CAM_001:TRACK_1")], camera_id="CAM_001", local_track_id="CAM_001:TRACK_1")
    track_two = _track([_observation(120, camera_id="CAM_002", local_track_id="CAM_002:TRACK_1")], camera_id="CAM_002", local_track_id="CAM_002:TRACK_1")

    collector.finalize_track(track_one)

    assert ("CAM_001", 120) not in collector._frame_cache
    assert ("CAM_002", 120) in collector._frame_cache

    collector.finalize_track(track_two)
    assert ("CAM_002", 120) not in collector._frame_cache


def test_missing_frame_fail_closed_raises_pipeline_runtime_error(tmp_path: Path) -> None:
    collector, _output_manager = _collector(tmp_path, fail_pipeline_on_error=True)
    collector.register_frame(_packet(0, _make_frame(fill=40)), [_tracked(0, confidence=0.95)])
    collector.register_frame(_packet(1, _make_frame(sharp=True, fill=90)), [_tracked(1, confidence=0.70)])
    track = _track([
        _observation(0, confidence=0.95),
        _observation(1, confidence=0.70),
    ])
    collector._frame_cache.pop(("CAM_001", 0), None)

    with pytest.raises(PipelineRuntimeError, match="Evidence frame missing from cache"):
        collector.finalize_track(track)


def test_evidence_index_and_metrics_files_can_be_written_from_collector_results(tmp_path: Path) -> None:
    collector, output_manager = _collector(tmp_path)
    frame = _make_frame(sharp=True)
    collector.register_frame(_packet(0, frame), [_tracked(0)])
    evidence = collector.finalize_track(_track([_observation(0)]))

    evidence_index_path = output_manager.save_evidence_index(collector.evidence_index)
    evidence_metrics_path = output_manager.save_evidence_metrics(collector.metrics)

    assert evidence
    assert json.loads(evidence_index_path.read_text(encoding="utf-8"))[0]["local_track_id"] == "CAM_001:TRACK_1"
    assert json.loads(evidence_metrics_path.read_text(encoding="utf-8"))["tracks_with_evidence"] == 1
