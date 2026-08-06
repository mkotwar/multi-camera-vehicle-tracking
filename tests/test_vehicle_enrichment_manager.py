from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from src.logging_setup import setup_logging
from src.models import LocalTrack, TrackEvidence
from src.output_writer import RunOutputManager
from src.vehicle_enrichment.enrichment_manager import VehicleEnrichmentManager, normalize_vehicle_enrichment_config
from src.vehicle_enrichment.schemas import VehicleBodyTypeResult, VehicleColourResult


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
        original_bbox_xyxy=(1.0, 1.0, 20.0, 20.0),
        expanded_crop_bbox_xyxy=(1, 1, 20, 20),
        context_padding_ratio=0.0,
        source_frame_width=30,
        source_frame_height=30,
        original_crop_width=19,
        original_crop_height=19,
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


def test_manager_marks_enabled_body_type_and_colour_as_skipped_when_no_evidence(tmp_path: Path) -> None:
    output_manager = RunOutputManager(tmp_path)
    logger = setup_logging(output_manager.run_directory, log_level="INFO")
    config = {
        "vehicle_enrichment": {
            "enabled": True,
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
            "shared_florence": {"enabled": True},
            "body_type": {"enabled": True},
            "colour": {"enabled": True},
            "make_model": {"enabled": False},
            "plate": {"detection_enabled": False, "colour_enabled": False},
            "ocr": {"enabled": False, "run_only_when_plate_detected": True},
        }
    }
    manager = VehicleEnrichmentManager(config, logger, output_manager)

    result = manager.enrich_completed_tracks([_track()], [])[0]

    assert result.status == "no_evidence"
    assert result.vehicle_body_type.status == "skipped"
    assert result.vehicle_body_type.reason == "no_evidence"
    assert result.vehicle_colour.status == "skipped"
    assert result.vehicle_colour.reason == "no_evidence"


def test_manager_handles_missing_evidence_and_fail_open(tmp_path: Path, monkeypatch) -> None:
    manager, _output = _manager(tmp_path, enabled=True)

    def _boom(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(manager.adapter, "adapt_track", _boom)
    result = manager.enrich_completed_tracks([_track()], [])

    assert result[0].status == "error"
    assert "boom" in result[0].errors[0]
    assert manager.metrics["enrichment_failures"] == 1


def test_manager_uses_shared_backend_and_writes_body_type_and_colour(tmp_path: Path, monkeypatch) -> None:
    output_manager = RunOutputManager(tmp_path)
    logger = setup_logging(output_manager.run_directory, log_level="INFO")
    config = {
        "vehicle_enrichment": {
            "enabled": True,
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
            "shared_florence": {"enabled": True},
            "body_type": {"enabled": True, "run_only_when_vehicle_class": ["CAR"]},
            "colour": {
                "enabled": True,
                "run_only_when_vehicle_class": ["3WHEELER", "BUS", "CAR", "MOTORCYCLE", "TRUCK"],
            },
            "make_model": {"enabled": False},
            "plate": {"detection_enabled": False, "colour_enabled": False},
            "ocr": {"enabled": False, "run_only_when_plate_detected": True},
        }
    }
    manager = VehicleEnrichmentManager(config, logger, output_manager)

    assert manager.body_type_classifier.backend is manager.florence_backend
    assert manager.colour_classifier.backend is manager.florence_backend

    image = np.full((30, 30, 3), 130, dtype=np.uint8)
    image[:, ::2] = 255
    crop_path = tmp_path / "crop_shared.jpg"
    cv2.imwrite(str(crop_path), image)

    monkeypatch.setattr(
        manager.body_type_classifier,
        "classify",
        lambda request: VehicleBodyTypeResult(label="SEDAN", status="completed", source="florence2"),
    )
    monkeypatch.setattr(
        manager.colour_classifier,
        "classify",
        lambda request: VehicleColourResult(label="WHITE", status="completed", source="florence2"),
    )

    result = manager.enrich_completed_tracks([_track()], [_record(str(crop_path))])[0]

    assert result.status == "completed"
    assert result.vehicle_body_type.label == "SEDAN"
    assert result.vehicle_body_type.status == "completed"
    assert result.vehicle_colour.label == "WHITE"
    assert result.vehicle_colour.status == "completed"
    assert result.vehicle_make is None
    assert result.vehicle_model is None
    assert result.plate_detected is False


def test_normalize_vehicle_enrichment_config_supports_colour_only_vehicle_attributes() -> None:
    normalized = normalize_vehicle_enrichment_config(
        {
            "enabled": True,
            "shared_florence": {"enabled": True},
            "vehicle_attributes": {
                "enabled": True,
                "reuse_single_response_for_attributes": False,
                "colour": {
                    "enabled": True,
                    "task_token": "<VQA>",
                    "prompt": "What colour is the vehicle?",
                    "generation": {"max_new_tokens": 16, "num_beams": 1, "use_cache": True, "early_stopping": False},
                },
                "body_type": {"enabled": False},
            },
            "plate": {"detection_enabled": False, "recognition_enabled": False},
        }
    )

    assert normalized["vehicle_attributes"]["enabled"] is True
    assert normalized["vehicle_attributes"]["reuse_single_response_for_attributes"] is False
    assert normalized["vehicle_attributes"]["colour"]["enabled"] is True
    assert normalized["vehicle_attributes"]["colour"]["prompt"] == "What colour is the vehicle?"
    assert normalized["vehicle_attributes"]["body_type"]["enabled"] is False
