from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from src.vehicle_enrichment.image_size_policy import normalize_image_size_policy
from src.vehicle_enrichment.schemas import EnrichmentEvidenceItem, TrackEnrichmentRequest
from src.vehicle_enrichment.vehicle_attribute_flow import BaseFlorenceVehicleAttributesFlow


class _FakeBackend:
    def __init__(self) -> None:
        self.adapter_active = False
        self.model_identifier = "base-model"
        self.load_calls = 0
        self.metrics = {"gpu_memory_allocated_mb": 0.0}

    def load(self) -> None:
        self.load_calls += 1

    def run_task(self, image, task_prompt, text_input=None, *, adapter_active=None, generation_overrides=None):
        return {
            "status": "completed",
            "payload": {
                "generated_text": "COLOUR: BLACK; BODY_TYPE: MPV",
                "parsed_answer": "COLOUR: BLACK; BODY_TYPE: MPV",
                "inference_duration_ms": 11.0,
            },
        }


def _request(tmp_path: Path) -> TrackEnrichmentRequest:
    image = np.full((220, 220, 3), 127, dtype=np.uint8)
    crop_path = tmp_path / "crop.jpg"
    assert cv2.imwrite(str(crop_path), image)
    item = EnrichmentEvidenceItem(
        local_track_id="TRACK_1",
        camera_id="CAM_001",
        native_tracker_id=1,
        frame_number=1,
        timestamp_seconds=0.0,
        source_image_path=str(crop_path),
        vehicle_crop_path=str(crop_path),
        annotated_frame_path=str(crop_path),
        bbox_xyxy=(0.0, 0.0, 20.0, 20.0),
        evidence_role="BEST",
        detection_confidence=0.9,
        crop_width=220,
        crop_height=220,
        crop_area=48400,
        sharpness_score=10.0,
        brightness_score=100.0,
        border_penalty=0.0,
        clipping_ratio=0.0,
        quality_score=0.9,
        source_frame_width=1920,
        source_frame_height=1080,
        context_padding_ratio=0.0,
        original_crop_width=220,
        original_crop_height=220,
        resolution_tier="acceptable",
    )
    return TrackEnrichmentRequest(
        local_track_id="TRACK_1",
        camera_id="CAM_001",
        native_tracker_id=1,
        vehicle_class="CAR",
        vehicle_class_confidence=0.9,
        track_status="COMPLETED",
        completion_reason="END",
        started_at_seconds=0.0,
        ended_at_seconds=0.1,
        evidence_items=[item],
    )


def test_vehicle_attribute_flow_uses_base_backend_and_one_call(tmp_path: Path) -> None:
    backend = _FakeBackend()
    flow = BaseFlorenceVehicleAttributesFlow(
        {
            "enabled": True,
            "maximum_crops_per_track": 3,
            "task_token": "<VQA>",
            "prompt": "Identify colour and body type.",
        },
        backend=backend,
        image_size_policy=normalize_image_size_policy(
            {"florence": {"minimum_original_width": 100, "minimum_original_height": 80, "preferred_original_width": 320, "preferred_original_height": 240, "pad_to_square": True}},
            fallback_body_type={"minimum_crop_width": 100, "minimum_crop_height": 80},
            fallback_colour={"minimum_crop_width": 100, "minimum_crop_height": 80},
            detection={},
        ),
        logger=__import__("logging").getLogger(__name__),
    )
    result = flow.classify(_request(tmp_path))
    assert backend.load_calls == 1
    assert result.adapter_loaded is False
    assert result.inference_count == 1
    assert result.body_type.label == "MPV"
    assert result.colour.label == "BLACK"


def test_vehicle_attribute_flow_supports_colour_only_mode(tmp_path: Path) -> None:
    backend = _FakeBackend()
    flow = BaseFlorenceVehicleAttributesFlow(
        {
            "enabled": True,
            "maximum_crops_per_track": 3,
            "reuse_single_response_for_attributes": False,
            "colour": {
                "enabled": True,
                "task_token": "<VQA>",
                "prompt": "What colour is the vehicle?",
                "generation": {"max_new_tokens": 16, "num_beams": 1, "use_cache": True},
            },
            "body_type": {"enabled": False},
        },
        backend=backend,
        image_size_policy=normalize_image_size_policy(
            {"florence": {"minimum_original_width": 100, "minimum_original_height": 80, "preferred_original_width": 320, "preferred_original_height": 240, "pad_to_square": True}},
            fallback_body_type={"minimum_crop_width": 100, "minimum_crop_height": 80},
            fallback_colour={"minimum_crop_width": 100, "minimum_crop_height": 80},
            detection={},
        ),
        logger=__import__("logging").getLogger(__name__),
    )
    result = flow.classify(_request(tmp_path))

    assert backend.load_calls == 1
    assert result.inference_count == 1
    assert result.colour.label == "BLACK"
    assert result.body_type.status == "disabled"
    assert flow.metrics["vehicle_attribute_colour_inference_calls"] == 1
    assert flow.metrics["vehicle_attribute_body_inference_calls"] == 0
    assert result.crop_level_rows[0]["prompt"] == "What colour is the vehicle?"
    assert result.crop_level_rows[0]["colour_status"] == "valid"


def test_vehicle_attribute_flow_skips_missing_crop_safely(tmp_path: Path) -> None:
    backend = _FakeBackend()
    request = _request(tmp_path)
    request.evidence_items[0].vehicle_crop_path = str(tmp_path / "missing.jpg")
    flow = BaseFlorenceVehicleAttributesFlow(
        {
            "enabled": True,
            "maximum_crops_per_track": 3,
            "colour": {"enabled": True, "task_token": "<VQA>", "prompt": "What colour is the vehicle?"},
            "body_type": {"enabled": False},
        },
        backend=backend,
        image_size_policy=normalize_image_size_policy(
            {"florence": {"minimum_original_width": 100, "minimum_original_height": 80, "preferred_original_width": 320, "preferred_original_height": 240, "pad_to_square": True}},
            fallback_body_type={"minimum_crop_width": 100, "minimum_crop_height": 80},
            fallback_colour={"minimum_crop_width": 100, "minimum_crop_height": 80},
            detection={},
        ),
        logger=__import__("logging").getLogger(__name__),
    )

    result = flow.classify(request)

    assert result.colour.label == "UNKNOWN"
    assert flow.metrics["vehicle_attribute_colour_inference_calls"] == 0
    assert flow.metrics["vehicle_attribute_skipped_missing_crop"] == 1
    assert result.crop_level_rows[0]["crop_available"] is False
    assert result.crop_level_rows[0]["crop_skip_reason"] == "missing_crop"


def test_vehicle_attribute_flow_uses_low_resolution_fallback_crop(tmp_path: Path) -> None:
    backend = _FakeBackend()
    request = _request(tmp_path)
    request.evidence_items[0].original_crop_width = 79
    request.evidence_items[0].original_crop_height = 101
    request.evidence_items[0].resolution_tier = "below_minimum"
    request.evidence_items[0].colour_selection_tier = "low_resolution_fallback"
    flow = BaseFlorenceVehicleAttributesFlow(
        {
            "enabled": True,
            "maximum_crops_per_track": 3,
            "colour": {"enabled": True, "task_token": "<VQA>", "prompt": "What colour is the vehicle?"},
            "body_type": {"enabled": False},
        },
        backend=backend,
        image_size_policy=normalize_image_size_policy(
            {"florence": {"minimum_original_width": 100, "minimum_original_height": 80, "preferred_original_width": 320, "preferred_original_height": 240, "pad_to_square": True}},
            fallback_body_type={"minimum_crop_width": 100, "minimum_crop_height": 80},
            fallback_colour={"minimum_crop_width": 100, "minimum_crop_height": 80},
            detection={},
        ),
        logger=__import__("logging").getLogger(__name__),
    )

    result = flow.classify(request)

    assert result.colour.label == "BLACK"
    assert flow.metrics["vehicle_attribute_colour_inference_calls"] == 1
    assert result.crop_level_rows[0]["selection_tier"] == "low_resolution_fallback"
