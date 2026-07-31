from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from src.logging_setup import setup_logging
from src.models import LocalTrack, TrackEvidence
from src.output_writer import RunOutputManager
from src.vehicle_enrichment.enrichment_manager import VehicleEnrichmentManager


def _track() -> LocalTrack:
    return LocalTrack(
        local_track_id="CAM_001:TRACK_1",
        camera_id="CAM_001",
        tracker_namespace="camera",
        native_tracker_id=1,
        status="COMPLETED",
        first_frame=0,
        last_frame=3,
        first_timestamp_seconds=0.0,
        last_timestamp_seconds=0.3,
        observation_count=3,
        lost_frames=0,
        final_class="car",
        final_class_reason="WEIGHTED_MAJORITY",
        class_counts={"car": 3},
        class_confidence_sums={"car": 2.7},
        observations=[],
        completion_reason="END_OF_STREAM",
    )


def _record(path: str) -> TrackEvidence:
    return TrackEvidence(
        local_track_id="CAM_001:TRACK_1",
        camera_id="CAM_001",
        native_tracker_id=1,
        tracker_namespace="camera",
        role="BEST_OVERALL",
        frame_number=3,
        timestamp_seconds=0.3,
        raw_class_name="car",
        final_class="car",
        confidence=0.9,
        crop_path=path,
        annotated_frame_path=path,
        bbox_xyxy=(1.0, 1.0, 20.0, 20.0),
        sharpness_score=10.0,
        best_overall_score=0.8,
    )


def _manager(tmp_path: Path, *, enabled: bool = True) -> tuple[VehicleEnrichmentManager, RunOutputManager]:
    output_manager = RunOutputManager(tmp_path)
    logger = setup_logging(output_manager.run_directory, log_level="INFO")
    config = {
        "vehicle_enrichment": {
            "enabled": enabled,
            "fail_open": True,
            "best_crops_per_track": 3,
            "extend_tracks_json": True,
            "write_separate_output": True,
            "evidence": {
                "source": "existing_track_evidence",
                "save_vehicle_crops": True,
                "minimum_crop_width": 10,
                "minimum_crop_height": 10,
                "minimum_sharpness": 1.0,
                "minimum_quality_score": 0.0,
                "border_margin_ratio": 0.02,
                "scoring": {
                    "area_weight": 0.25,
                    "sharpness_weight": 0.25,
                    "confidence_weight": 0.20,
                    "role_weight": 0.15,
                    "border_weight": 0.05,
                    "clipping_weight": 0.05,
                    "brightness_weight": 0.05,
                },
            },
            "shared_florence": {"enabled": False},
            "body_type": {"enabled": False},
            "colour": {"enabled": False},
            "make_model": {"enabled": False},
            "plate": {"detection_enabled": False, "colour_enabled": False},
            "ocr": {"enabled": False, "run_only_when_plate_detected": True},
        }
    }
    return VehicleEnrichmentManager(config, logger, output_manager), output_manager


def test_manager_returns_disabled_result_when_enrichment_disabled(tmp_path: Path) -> None:
    manager, _output = _manager(tmp_path, enabled=False)
    result = manager.enrich_completed_tracks([_track()], [])

    assert result[0].status == "disabled"
    assert result[0].vehicle_body_type.status == "disabled"


def test_manager_returns_evidence_ready_result_with_disabled_modules(tmp_path: Path) -> None:
    manager, _output = _manager(tmp_path, enabled=True)
    image = np.full((30, 30, 3), 130, dtype=np.uint8)
    image[:, ::2] = 255
    crop_path = tmp_path / "crop.jpg"
    cv2.imwrite(str(crop_path), image)

    result = manager.enrich_completed_tracks([_track()], [_record(str(crop_path))])

    assert result[0].status == "evidence_ready"
    assert result[0].vehicle_class == "CAR"
    assert len(result[0].evidence_used) == 1
    assert Path(str(result[0].evidence_used[0].vehicle_crop_path)).exists()
    assert manager.metrics["current_in_memory_tracks"] == 0


def test_manager_handles_missing_evidence_and_fail_open(tmp_path: Path, monkeypatch) -> None:
    manager, _output = _manager(tmp_path, enabled=True)

    def _boom(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(manager.adapter, "adapt_track", _boom)
    result = manager.enrich_completed_tracks([_track()], [])

    assert result[0].status == "error"
    assert "boom" in result[0].errors[0]
    assert manager.metrics["enrichment_failures"] == 1
