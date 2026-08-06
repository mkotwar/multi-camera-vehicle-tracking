from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from src.logging_setup import setup_logging
from src.models import LocalTrack, TrackEvidence
from src.output_writer import RunOutputManager
from src.vehicle_enrichment.enrichment_manager import VehicleEnrichmentManager
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
        source_frame_width=400,
        source_frame_height=300,
        original_crop_width=220,
        original_crop_height=200,
        sharpness_score=10.0,
        best_overall_score=0.8,
    )


def _manager(tmp_path: Path, *, florence_mode: str) -> VehicleEnrichmentManager:
    output_manager = RunOutputManager(tmp_path)
    logger = setup_logging(output_manager.run_directory, log_level="INFO")
    config = {
        "vehicle_enrichment": {
            "enabled": True,
            "florence_mode": florence_mode,
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
            "shared_florence": {"enabled": True, "adapter_path": "D:/project/models/OCR_MUKUL/OCR_MUKUL/adaptor_florance_baseFT", "local_files_only": True},
            "body_type": {"enabled": True, "run_only_when_vehicle_class": ["CAR"]},
            "colour": {"enabled": True, "run_only_when_vehicle_class": ["CAR"]},
            "ocr_mukul": {"enabled": True, "task_token": "<CAPTION>", "maximum_crops_per_track": 3},
            "make_model": {"enabled": False},
            "plate": {"detection_enabled": False, "colour_enabled": False},
            "ocr": {"enabled": False, "run_only_when_plate_detected": True},
        }
    }
    return VehicleEnrichmentManager(config, logger, output_manager)


def test_manager_ocr_mukul_mode_uses_ocr_flow(tmp_path: Path, monkeypatch) -> None:
    manager = _manager(tmp_path, florence_mode="ocr_mukul")
    image = np.full((220, 220, 3), 120, dtype=np.uint8)
    crop_path = tmp_path / "ocr_crop.jpg"
    cv2.imwrite(str(crop_path), image)

    monkeypatch.setattr(manager.body_type_classifier, "classify", lambda request: (_ for _ in ()).throw(AssertionError("current body flow should not run")))
    monkeypatch.setattr(manager.colour_classifier, "classify", lambda request: (_ for _ in ()).throw(AssertionError("current colour flow should not run")))
    monkeypatch.setattr(
        manager.ocr_mukul_flow,
        "classify",
        lambda request: type(
            "Result",
            (),
            {
                "body_type": VehicleBodyTypeResult(label="SEDAN", status="completed", source="ocr_mukul"),
                "colour": VehicleColourResult(label="BLACK", status="completed", source="ocr_mukul"),
                "crop_level_rows": [{"crop_path": str(crop_path), "frame_index": 3, "caption": "A black sedan driving on the road.", "raw_body_type_phrase": "sedan", "normalized_body_type": "SEDAN", "raw_colour_phrase": "black", "normalized_colour": "BLACK"}],
                "caption_inference_count": 1,
                "adapter_loaded": True,
            },
        )(),
    )

    result = manager.enrich_completed_tracks([_track()], [_record(str(crop_path))])[0]
    assert result.florence_mode == "ocr_mukul"
    assert result.vehicle_body_type.label == "SEDAN"
    assert result.vehicle_colour.label == "BLACK"
    assert result.caption_inference_count == 1
    assert result.crop_level_captions


def test_manager_comparison_mode_runs_both_paths(tmp_path: Path, monkeypatch) -> None:
    manager = _manager(tmp_path, florence_mode="comparison")
    image = np.full((220, 220, 3), 120, dtype=np.uint8)
    crop_path = tmp_path / "comparison_crop.jpg"
    cv2.imwrite(str(crop_path), image)
    monkeypatch.setattr(manager.body_type_classifier, "classify", lambda request: VehicleBodyTypeResult(label="UNKNOWN", status="completed", source="florence2"))
    monkeypatch.setattr(manager.colour_classifier, "classify", lambda request: VehicleColourResult(label="UNKNOWN", status="completed", source="florence2"))
    monkeypatch.setattr(
        manager.ocr_mukul_flow,
        "classify",
        lambda request: type(
            "Result",
            (),
            {
                "body_type": VehicleBodyTypeResult(label="SEDAN", status="completed", source="ocr_mukul"),
                "colour": VehicleColourResult(label="BLACK", status="completed", source="ocr_mukul"),
                "crop_level_rows": [{"crop_path": str(crop_path), "frame_index": 3, "caption": "A black sedan driving on the road.", "raw_body_type_phrase": "sedan", "normalized_body_type": "SEDAN", "raw_colour_phrase": "black", "normalized_colour": "BLACK"}],
                "caption_inference_count": 1,
                "adapter_loaded": True,
            },
        )(),
    )
    result = manager.enrich_completed_tracks([_track()], [_record(str(crop_path))])[0]
    assert result.florence_mode == "comparison"
    assert result.comparison_payload is not None
    assert result.comparison_payload["ocr_mukul"]["body_type_label"] == "SEDAN"
    assert result.crop_level_captions
