from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import time

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


def _record_for_track(path: str, *, local_track_id: str, camera_id: str, native_tracker_id: int) -> TrackEvidence:
    record = _record(path)
    record.local_track_id = local_track_id
    record.camera_id = camera_id
    record.native_tracker_id = native_tracker_id
    return record


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


def _async_manager(tmp_path: Path) -> tuple[VehicleEnrichmentManager, RunOutputManager]:
    output_manager = RunOutputManager(tmp_path)
    logger = setup_logging(output_manager.run_directory, log_level="INFO")
    config = {
        "vehicle_enrichment": {
            "enabled": True,
            "fail_open": True,
            "best_crops_per_track": 3,
            "extend_tracks_json": True,
            "write_separate_output": True,
            "async_colour": {
                "enabled": True,
                "queue_size": 2,
                "worker_count": 1,
                "queue_put_timeout_seconds": 0.01,
            },
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
            "vehicle_attributes": {
                "enabled": True,
                "backend": "base_florence",
                "maximum_crops_per_track": 3,
                "reuse_single_response_for_attributes": False,
                "task_token": "<VQA>",
                "prompt": "What colour is the vehicle?",
                "colour": {
                    "enabled": True,
                    "backend": "base_florence",
                    "task_token": "<VQA>",
                    "prompt": "What colour is the vehicle?",
                    "generation": {"max_new_tokens": 16, "num_beams": 1, "use_cache": True, "early_stopping": False},
                },
                "body_type": {"enabled": False},
            },
            "body_type": {"enabled": False},
            "colour": {"enabled": False},
            "make_model": {"enabled": False},
            "plate": {"detection_enabled": False, "colour_enabled": False},
            "ocr": {"enabled": False, "run_only_when_plate_detected": True},
        }
    }
    return VehicleEnrichmentManager(config, logger, output_manager), output_manager


def _attribute_result(*, colour_label: str = "WHITE", body_type_label: str = "UNKNOWN", crop_path: str = "crop.jpg"):
    return SimpleNamespace(
        body_type=VehicleBodyTypeResult(label=body_type_label, predictions=[], status="disabled", source="base_florence"),
        colour=VehicleColourResult(label=colour_label, predictions=[], status="completed", source="base_florence", aggregation_reason="weighted_agreement"),
        crop_level_rows=[
            {
                "vehicle_crop_path": crop_path,
                "frame_index": 3,
                "parsed_body_type": body_type_label,
                "body_type_status": "disabled",
                "body_type_reason": "disabled",
                "body_type_raw_response": "",
                "body_type_task_token": "<VQA>",
                "body_type_prompt": "",
                "body_type_effective_processor_text": "",
                "body_type_inference_time_ms": 0.0,
                "parsed_colour": colour_label,
                "colour_status": "completed",
                "colour_reason": "weighted_agreement",
                "crop_source": "saved_vehicle_crop",
                "crop_available": True,
                "crop_skip_reason": None,
                "selection_tier": "acceptable",
                "colour_raw_response": colour_label.lower(),
                "colour_task_token": "<VQA>",
                "colour_prompt": "What colour is the vehicle?",
                "colour_effective_processor_text": "<VQA>What colour is the vehicle?",
                "colour_inference_time_ms": 12.0,
                "colour_post_processed_response": colour_label,
            }
        ],
        inference_count=1,
        adapter_loaded=False,
        raw_responses=[colour_label.lower()],
        body_type_eligible=False,
        body_type_candidate_crop_count=0,
        body_type_selected_crop_count=0,
        body_type_florence_call_count=0,
        body_type_valid_prediction_count=0,
        body_type_failure_reason="disabled",
    )


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


def test_manager_skips_frozen_make_model_and_plate_runtime_paths(tmp_path: Path, monkeypatch) -> None:
    manager, _output = _manager(tmp_path, enabled=True)
    image = np.full((30, 30, 3), 130, dtype=np.uint8)
    crop_path = tmp_path / "crop_frozen.jpg"
    cv2.imwrite(str(crop_path), image)

    make_model_calls = 0
    plate_detect_calls = 0
    plate_validate_calls = 0
    plate_ocr_calls = 0

    def _make_model_probe(*args, **kwargs):
        nonlocal make_model_calls
        make_model_calls += 1
        raise AssertionError("make/model should stay frozen")

    def _plate_detect_probe(*args, **kwargs):
        nonlocal plate_detect_calls
        plate_detect_calls += 1
        raise AssertionError("plate detector should stay frozen")

    def _plate_validate_probe(*args, **kwargs):
        nonlocal plate_validate_calls
        plate_validate_calls += 1
        raise AssertionError("plate validator should stay frozen")

    def _plate_ocr_probe(*args, **kwargs):
        nonlocal plate_ocr_calls
        plate_ocr_calls += 1
        raise AssertionError("plate OCR should stay frozen")

    monkeypatch.setattr(manager.make_model_classifier, "classify", _make_model_probe)
    monkeypatch.setattr(manager.plate_detector, "detect", _plate_detect_probe)
    monkeypatch.setattr(manager.plate_quality_validator, "validate", _plate_validate_probe)
    monkeypatch.setattr(manager.plate_ocr_engine, "recognize", _plate_ocr_probe)

    result = manager.enrich_completed_tracks([_track()], [_record(str(crop_path))])[0]

    assert result.status == "evidence_ready"
    assert make_model_calls == 0
    assert plate_detect_calls == 0
    assert plate_validate_calls == 0
    assert plate_ocr_calls == 0
    assert result.plate_detected is False
    assert result.plate_text is None


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

    assert normalized["enrichment"]["enabled"] is True
    assert normalized["enrichment"]["reuse_single_response_for_attributes"] is False
    assert normalized["enrichment"]["colour"]["enabled"] is True
    assert normalized["enrichment"]["colour"]["prompt"] == "What colour is the vehicle?"
    assert normalized["enrichment"]["body_type"]["enabled"] is False


def test_normalize_vehicle_enrichment_config_supports_canonical_schema() -> None:
    normalized = normalize_vehicle_enrichment_config(
        {
            "enabled": True,
            "florence": {
                "enabled": True,
                "backend": "florence2",
                "model_id": "florence-model",
                "processor_path": "florence-processor",
                "adapter": {"enabled": True, "path": "adapter-path", "fail_if_missing": True},
                "device": "cpu",
                "dtype": "float16",
                "local_files_only": True,
                "lazy_load": True,
                "trust_remote_code": True,
                "attention_implementation": "eager",
            },
            "enrichment": {
                "enabled": True,
                "colour": {
                    "enabled": True,
                    "backend": "base_florence",
                    "task_token": "<VQA>",
                    "prompt": "What colour is the vehicle?",
                    "generation": {"max_new_tokens": 16, "num_beams": 1, "use_cache": True},
                    "async": {"enabled": True, "queue_size": 9, "worker_count": 1, "queue_put_timeout_seconds": 0.2},
                },
                "body_type": {
                    "enabled": True,
                    "backend": "florence2",
                    "run_only_when_vehicle_class": ["CAR"],
                    "allowed_labels": ["SEDAN", "UNKNOWN"],
                },
                "make_model": {"enabled": False},
                "plate": {
                    "enabled": True,
                    "detector": {"enabled": True, "model_path": "plate.pt"},
                    "colour": {"enabled": False},
                    "ocr": {"enabled": True, "backend": "ocr_mukul_adapter", "task_token": "<OCR>", "prompt": ""},
                },
            },
        }
    )

    assert normalized["florence"]["backend"] == "florence"
    assert normalized["florence"]["model_id"] == "florence-model"
    assert normalized["florence"]["adapter"]["enabled"] is True
    assert normalized["florence"]["adapter"]["path"] == "adapter-path"
    assert normalized["enrichment"]["colour"]["backend"] == "florence"
    assert normalized["enrichment"]["body_type"]["backend"] == "florence"
    assert normalized["enrichment"]["plate"]["detector"]["enabled"] is True
    assert normalized["enrichment"]["plate"]["ocr"]["enabled"] is True
    assert normalized["enrichment"]["colour"]["async"]["enabled"] is True


def test_new_canonical_keys_win_over_legacy_keys() -> None:
    normalized = normalize_vehicle_enrichment_config(
        {
            "shared_florence": {"enabled": True, "backend": "florence2", "base_model_id": "legacy-model"},
            "vehicle_attributes": {
                "enabled": True,
                "colour": {"enabled": True, "prompt": "legacy colour prompt"},
                "body_type": {"enabled": True, "prompt": "legacy body prompt"},
            },
            "plate": {"detection_enabled": False, "recognition_enabled": False},
            "ocr": {"enabled": False, "run_only_when_plate_detected": True},
            "florence": {"enabled": True, "backend": "base_florence", "model_id": "new-model"},
            "enrichment": {
                "enabled": True,
                "colour": {"enabled": True, "prompt": "new colour prompt"},
                "body_type": {"enabled": True, "prompt": "new body prompt"},
                "plate": {"ocr": {"enabled": True, "task_token": "<OCR>"}, "detector": {"enabled": True, "model_path": "plate.pt"}},
            },
        }
    )

    assert normalized["florence"]["model_id"] == "new-model"
    assert normalized["enrichment"]["colour"]["prompt"] == "new colour prompt"
    assert normalized["enrichment"]["body_type"]["prompt"] == "new body prompt"
    assert normalized["enrichment"]["plate"]["detector"]["enabled"] is True
    assert normalized["enrichment"]["plate"]["ocr"]["enabled"] is True


def test_legacy_translation_normalizes_backend_aliases_and_async_colour() -> None:
    normalized = normalize_vehicle_enrichment_config(
        {
            "shared_florence": {"enabled": True, "backend": "florence2"},
            "vehicle_attributes": {
                "enabled": True,
                "backend": "base_florence",
                "colour": {"enabled": True, "backend": "base_florence"},
                "body_type": {"enabled": True, "backend": "base_florence"},
            },
            "async_colour": {"enabled": True, "queue_size": 7, "worker_count": 1, "queue_put_timeout_seconds": 0.05},
        }
    )

    assert normalized["florence"]["backend"] == "florence"
    assert normalized["enrichment"]["colour"]["backend"] == "florence"
    assert normalized["enrichment"]["body_type"]["backend"] == "florence"
    assert normalized["enrichment"]["colour"]["async"]["enabled"] is True
    assert normalized["enrichment"]["colour"]["async"]["queue_size"] == 7


def test_legacy_plate_and_execution_mode_translate_to_canonical_shape(tmp_path: Path) -> None:
    manager, _output = _manager(tmp_path, enabled=True)
    normalized = normalize_vehicle_enrichment_config(
        {
            "shared_florence": {"enabled": True},
            "plate": {"detection_enabled": True, "recognition_enabled": True, "colour_enabled": True, "detector": {"model_path": "plate.pt"}},
            "ocr": {"enabled": False, "run_only_when_plate_detected": True},
            "florence_mode": "ocr_mukul",
        }
    )

    assert normalized["execution_mode"] == "ocr_mukul"
    assert normalized["enrichment"]["plate"]["enabled"] is True
    assert normalized["enrichment"]["plate"]["detector"]["enabled"] is True
    assert normalized["enrichment"]["plate"]["ocr"]["enabled"] is True
    assert normalized["enrichment"]["plate"]["colour"]["enabled"] is True
    assert manager.config["execution_mode"] == "current"


def test_manager_prefers_capture_zone_with_existing_fallback(tmp_path: Path) -> None:
    manager, _output = _manager(tmp_path, enabled=True)
    manager.config["evidence"]["source"] = "capture_zone_with_existing_fallback"
    image = np.full((30, 30, 3), 130, dtype=np.uint8)
    image[:, ::2] = 255
    crop_path = tmp_path / "capture_zone_crop.jpg"
    cv2.imwrite(str(crop_path), image)
    record = {
        "local_track_id": "CAM_001:TRACK_1",
        "camera_id": "CAM_001",
        "native_tracker_id": 1,
        "tracker_namespace": "camera",
        "role": "CAPTURE_ZONE",
        "frame_number": 3,
        "timestamp_seconds": 0.3,
        "raw_class_name": "car",
        "final_class": "car",
        "confidence": 0.9,
        "crop_path": str(crop_path),
        "annotated_frame_path": None,
        "bbox_xyxy": [1.0, 1.0, 20.0, 20.0],
        "original_bbox": [1.0, 1.0, 20.0, 20.0],
        "expanded_crop_bbox": [1, 1, 20, 20],
        "context_padding_ratio": 0.0,
        "source_frame_width": 30,
        "source_frame_height": 30,
        "original_crop_width": 19,
        "original_crop_height": 19,
        "sharpness_score": 10.0,
        "best_overall_score": 0.9,
        "evidence_source": "capture_zone",
    }

    result = manager.enrich_completed_tracks([_track()], [record, _record(str(crop_path))])[0]

    assert result.status == "evidence_ready"
    assert manager.metrics["capture_zone_crops_used_by_enrichment"] >= 1
    assert result.evidence_used[0].evidence_source == "capture_zone"


def test_manager_falls_back_to_existing_evidence_when_capture_zone_missing(tmp_path: Path) -> None:
    manager, _output = _manager(tmp_path, enabled=True)
    manager.config["evidence"]["source"] = "capture_zone_with_existing_fallback"
    image = np.full((30, 30, 3), 130, dtype=np.uint8)
    image[:, ::2] = 255
    crop_path = tmp_path / "fallback_crop.jpg"
    cv2.imwrite(str(crop_path), image)

    result = manager.enrich_completed_tracks([_track()], [_record(str(crop_path))])[0]

    assert result.status == "evidence_ready"
    assert manager.metrics["capture_zone_fallback_to_existing_evidence"] == 1
    assert result.evidence_used[0].evidence_source == "existing_track_evidence"


def test_manager_uses_raw_track_crop_fallback_when_finalized_evidence_missing(tmp_path: Path, monkeypatch) -> None:
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
            "shared_florence": {"enabled": False},
            "body_type": {"enabled": False},
            "colour": {"enabled": True},
            "make_model": {"enabled": False},
            "plate": {"detection_enabled": False, "colour_enabled": False},
            "ocr": {"enabled": False, "run_only_when_plate_detected": True},
        }
    }
    manager = VehicleEnrichmentManager(config, logger, output_manager)
    image = np.full((101, 79, 3), 130, dtype=np.uint8)
    image[:, ::2] = 255
    raw_track_dir = output_manager.track_crops_directory / "CAM_001" / "TRACK_1"
    raw_track_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(raw_track_dir / "frame_001152.jpg"), image)
    monkeypatch.setattr(
        manager.colour_classifier,
        "classify",
        lambda request: VehicleColourResult(
            label="BLACK",
            predictions=[],
            status="completed",
            source="florence2",
            aggregation_reason="weighted_agreement",
        ),
    )

    result = manager.enrich_completed_tracks([_track()], [])[0]

    assert result.status == "completed"
    assert result.candidate_crop_count >= 1
    assert result.readable_crop_count >= 1
    assert result.fallback_crop_count >= 1
    assert result.evidence_used[0].evidence_source == "raw_track_crop_fallback"


def test_async_colour_track_enqueues_exactly_once_and_drains(tmp_path: Path, monkeypatch) -> None:
    manager, _output = _async_manager(tmp_path)
    image = np.full((30, 30, 3), 130, dtype=np.uint8)
    crop_path = tmp_path / "async_crop.jpg"
    cv2.imwrite(str(crop_path), image)
    calls: list[str] = []

    def _classify(request):
        calls.append(request.local_track_id)
        return _attribute_result(colour_label="WHITE", crop_path=str(crop_path))

    monkeypatch.setattr(manager.vehicle_attribute_flow, "classify", _classify)
    try:
        first = manager.enrich_completed_tracks([_track()], [_record(str(crop_path))])
        duplicate = manager.enrich_completed_tracks([_track()], [_record(str(crop_path))])
        drained = manager.finalize_async_colour()
    finally:
        manager.finalize_async_colour()

    assert first == []
    assert duplicate == []
    assert len(drained) == 1
    assert drained[0].local_track_id == "CAM_001:TRACK_1"
    assert calls == ["CAM_001:TRACK_1"]
    assert manager.metrics["colour_jobs_enqueued"] == 1
    assert manager.metrics["colour_jobs_duplicate_attempts"] == 1
    assert manager.metrics["track_evidence_pending_count"] == 0


def test_async_colour_preserves_camera_and_track_identity(tmp_path: Path, monkeypatch) -> None:
    manager, _output = _async_manager(tmp_path)
    image = np.full((30, 30, 3), 130, dtype=np.uint8)
    crop_path = tmp_path / "async_identity.jpg"
    cv2.imwrite(str(crop_path), image)
    seen: list[tuple[str, str]] = []

    def _classify(request):
        seen.append((request.camera_id, request.local_track_id))
        return _attribute_result(colour_label="BLUE", crop_path=str(crop_path))

    monkeypatch.setattr(manager.vehicle_attribute_flow, "classify", _classify)
    track_two = _track()
    track_two.local_track_id = "CAM_002:TRACK_1"
    track_two.camera_id = "CAM_002"
    try:
        immediate_from_enqueue = manager.enrich_completed_tracks(
            [_track(), track_two],
            [
                _record_for_track(str(crop_path), local_track_id="CAM_001:TRACK_1", camera_id="CAM_001", native_tracker_id=1),
                _record_for_track(str(crop_path), local_track_id="CAM_002:TRACK_1", camera_id="CAM_002", native_tracker_id=1),
            ],
        )
        immediate = manager.drain_completed_results()
        drained = immediate_from_enqueue + immediate + manager.finalize_async_colour()
    finally:
        manager.finalize_async_colour()

    assert ("CAM_001", "CAM_001:TRACK_1") in seen
    assert ("CAM_002", "CAM_002:TRACK_1") in seen
    assert len(drained) == 2
    assert {result.local_track_id for result in drained} == {"CAM_001:TRACK_1", "CAM_002:TRACK_1"}


def test_async_colour_returns_before_worker_finishes(tmp_path: Path, monkeypatch) -> None:
    manager, _output = _async_manager(tmp_path)
    image = np.full((30, 30, 3), 130, dtype=np.uint8)
    crop_path = tmp_path / "async_nonblocking.jpg"
    cv2.imwrite(str(crop_path), image)

    def _classify(request):
        time.sleep(0.1)
        return _attribute_result(colour_label="RED", crop_path=str(crop_path))

    monkeypatch.setattr(manager.vehicle_attribute_flow, "classify", _classify)
    started = time.perf_counter()
    try:
        immediate = manager.enrich_completed_tracks([_track()], [_record(str(crop_path))])
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        drained = immediate + manager.finalize_async_colour()
    finally:
        manager.finalize_async_colour()

    assert immediate == []
    assert elapsed_ms < 100.0
    assert len(drained) == 1
    assert drained[0].vehicle_colour.label == "RED"


def test_async_colour_worker_survives_failed_job(tmp_path: Path, monkeypatch) -> None:
    manager, _output = _async_manager(tmp_path)
    image = np.full((30, 30, 3), 130, dtype=np.uint8)
    crop_path = tmp_path / "async_failure.jpg"
    cv2.imwrite(str(crop_path), image)
    call_count = 0

    def _classify(request):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise RuntimeError("mock florence failure")
        return _attribute_result(colour_label="GREEN", crop_path=str(crop_path))

    monkeypatch.setattr(manager.vehicle_attribute_flow, "classify", _classify)
    second_track = _track()
    second_track.local_track_id = "CAM_001:TRACK_2"
    second_track.native_tracker_id = 2
    try:
        immediate = manager.enrich_completed_tracks(
            [_track(), second_track],
            [
                _record_for_track(str(crop_path), local_track_id="CAM_001:TRACK_1", camera_id="CAM_001", native_tracker_id=1),
                _record_for_track(str(crop_path), local_track_id="CAM_001:TRACK_2", camera_id="CAM_001", native_tracker_id=2),
            ],
        )
        drained = immediate + manager.finalize_async_colour()
    finally:
        manager.finalize_async_colour()

    by_track = {item.local_track_id: item for item in drained}
    assert by_track["CAM_001:TRACK_1"].status == "error"
    assert by_track["CAM_001:TRACK_2"].vehicle_colour.label == "GREEN"
    assert manager.metrics["colour_jobs_failed"] == 1
    assert manager.metrics["colour_jobs_completed"] == 1


def test_async_colour_matches_sync_result_for_mocked_input(tmp_path: Path, monkeypatch) -> None:
    sync_manager, _ = _manager(tmp_path / "sync", enabled=True)
    async_manager, _ = _async_manager(tmp_path / "async")
    image = np.full((30, 30, 3), 130, dtype=np.uint8)
    crop_path = tmp_path / "async_match.jpg"
    cv2.imwrite(str(crop_path), image)

    monkeypatch.setattr(
        sync_manager.colour_classifier,
        "classify",
        lambda request: VehicleColourResult(label="WHITE", predictions=[], status="completed", source="florence2", aggregation_reason="weighted_agreement"),
    )
    monkeypatch.setattr(
        async_manager.vehicle_attribute_flow,
        "classify",
        lambda request: _attribute_result(colour_label="WHITE", crop_path=str(crop_path)),
    )
    try:
        sync_result = sync_manager.enrich_completed_tracks([_track()], [_record(str(crop_path))])[0]
        async_manager.enrich_completed_tracks([_track()], [_record(str(crop_path))])
        async_result = async_manager.finalize_async_colour()[0]
    finally:
        async_manager.finalize_async_colour()

    assert sync_result.vehicle_colour.label == async_result.vehicle_colour.label
    assert async_result.vehicle_colour.label == "WHITE"
