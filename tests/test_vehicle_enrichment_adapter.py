from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from src.logging_setup import setup_logging
from src.models import LocalTrack, TrackEvidence
from src.output_writer import RunOutputManager
from src.vehicle_enrichment.evidence_adapter import EvidenceAdapter


def _track() -> LocalTrack:
    return LocalTrack(
        local_track_id="CAM_001:TRACK_1",
        camera_id="CAM_001",
        tracker_namespace="camera",
        native_tracker_id=1,
        status="COMPLETED",
        first_frame=1,
        last_frame=3,
        first_timestamp_seconds=0.1,
        last_timestamp_seconds=0.3,
        observation_count=3,
        lost_frames=0,
        final_class="car",
        final_class_reason="WEIGHTED_MAJORITY",
        class_counts={"car": 3},
        class_confidence_sums={"car": 2.4},
        observations=[],
        completion_reason="END_OF_STREAM",
    )


def _record(crop_path: str | None, annotated_path: str | None, *, bbox=(10.0, 10.0, 40.0, 35.0), frame_number: int = 1) -> TrackEvidence:
    return TrackEvidence(
        local_track_id="CAM_001:TRACK_1",
        camera_id="CAM_001",
        native_tracker_id=1,
        tracker_namespace="camera",
        role="BEST_OVERALL",
        frame_number=frame_number,
        timestamp_seconds=frame_number / 10.0,
        raw_class_name="car",
        final_class="car",
        confidence=0.88,
        crop_path=crop_path,
        annotated_frame_path=annotated_path,
        bbox_xyxy=bbox,
        sharpness_score=11.0,
        best_overall_score=0.75,
    )


def _write(path: Path, image: np.ndarray) -> str:
    cv2.imwrite(str(path), image)
    return str(path)


def _adapter(tmp_path: Path) -> EvidenceAdapter:
    manager = RunOutputManager(tmp_path)
    logger = setup_logging(manager.run_directory, log_level="INFO")
    config = {"evidence": {"save_vehicle_crops": True, "border_margin_ratio": 0.02}}
    return EvidenceAdapter(config, manager, logger)


def test_adapter_reuses_existing_crop_and_preserves_ids(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)
    crop = _write(tmp_path / "crop.jpg", np.full((20, 30, 3), 100, dtype=np.uint8))
    annotated = _write(tmp_path / "annotated.jpg", np.full((80, 120, 3), 140, dtype=np.uint8))

    items = adapter.adapt_track(_track(), [_record(crop, annotated)])

    assert len(items) == 1
    assert items[0].camera_id == "CAM_001"
    assert items[0].local_track_id == "CAM_001:TRACK_1"
    assert items[0].vehicle_crop_path == str(Path(crop).resolve())


def test_adapter_extracts_fallback_crop_when_existing_crop_is_missing(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)
    annotated = _write(tmp_path / "annotated.jpg", np.full((80, 120, 3), 140, dtype=np.uint8))

    items = adapter.adapt_track(_track(), [_record(None, annotated)])

    assert len(items) == 1
    assert items[0].vehicle_crop_path is not None
    assert Path(str(items[0].vehicle_crop_path)).exists()


def test_adapter_marks_invalid_bbox_and_missing_image(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)
    items = adapter.adapt_track(_track(), [_record(None, str(tmp_path / "missing.jpg"), bbox=(10.0, 10.0, 5.0, 5.0))])

    assert len(items) == 1
    assert "invalid_bbox" in items[0].rejection_reasons


def test_adapter_deduplicates_same_frame_and_bbox(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)
    crop = _write(tmp_path / "crop.jpg", np.full((20, 30, 3), 100, dtype=np.uint8))
    annotated = _write(tmp_path / "annotated.jpg", np.full((80, 120, 3), 140, dtype=np.uint8))

    items = adapter.adapt_track(_track(), [_record(crop, annotated), _record(crop, annotated)])

    assert len(items) == 1
