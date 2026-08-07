from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from src.vehicle_enrichment.image_size_policy import normalize_image_size_policy
from src.vehicle_enrichment.ocr_mukul.attribute_parser import ATTRIBUTE_REASON_PLATE_LIKE, parse_caption_attributes
from src.vehicle_enrichment.ocr_mukul.backend import OCRMukulFlorenceFlow
from src.vehicle_enrichment.ocr_mukul.image_preprocessor import OCRMukulImagePreprocessor
from src.vehicle_enrichment.schemas import EnrichmentEvidenceItem, TrackEnrichmentRequest


class _FakeBackend:
    def __init__(self, caption: str, *, adapter_active: bool = True) -> None:
        self.caption = caption
        self.adapter_active = adapter_active
        self.model_identifier = "adapter-model"
        self.processor_identifier = "base-processor"
        self.load_calls = 0
        self.run_calls = 0

    def load(self) -> None:
        self.load_calls += 1

    def run_task(self, image, task_prompt, text_input=None, *, adapter_active=None):
        self.run_calls += 1
        return {
            "status": "completed",
            "reason": None,
            "payload": {
                "generated_text": self.caption,
                "parsed_answer": self.caption,
                "inference_duration_ms": 12.5,
                "pixel_values_shape": [1, 3, 128, 256],
                "model_identifier": self.model_identifier,
                "processor_identifier": self.processor_identifier,
            },
        }


def _evidence(tmp_path: Path) -> EnrichmentEvidenceItem:
    image = np.full((120, 180, 3), 127, dtype=np.uint8)
    crop_path = tmp_path / "crop.jpg"
    assert cv2.imwrite(str(crop_path), image)
    return EnrichmentEvidenceItem(
        local_track_id="CAM_001:TRACK_1",
        camera_id="CAM_001",
        native_tracker_id=1,
        frame_number=1,
        timestamp_seconds=0.1,
        source_image_path=str(crop_path),
        vehicle_crop_path=str(crop_path),
        annotated_frame_path=str(crop_path),
        bbox_xyxy=(0.0, 0.0, 10.0, 10.0),
        evidence_role="BEST",
        detection_confidence=0.9,
        crop_width=180,
        crop_height=120,
        crop_area=21600,
        sharpness_score=40.0,
        brightness_score=100.0,
        border_penalty=0.0,
        clipping_ratio=0.0,
        quality_score=0.9,
        source_frame_width=1920,
        source_frame_height=1080,
        context_padding_ratio=0.08,
        original_crop_width=180,
        original_crop_height=120,
        resolution_tier="acceptable",
    )


def _request(tmp_path: Path, *, vehicle_class: str = "CAR") -> TrackEnrichmentRequest:
    return TrackEnrichmentRequest(
        local_track_id="CAM_001:TRACK_1",
        camera_id="CAM_001",
        native_tracker_id=1,
        vehicle_class=vehicle_class,
        vehicle_class_confidence=0.9,
        track_status="COMPLETED",
        completion_reason="END_OF_STREAM",
        started_at_seconds=0.0,
        ended_at_seconds=0.1,
        evidence_items=[_evidence(tmp_path)],
    )


def test_ocr_mukul_preprocessor_resizes_small_images() -> None:
    preprocessor = OCRMukulImagePreprocessor()
    image = np.zeros((100, 120, 3), dtype=np.uint8)
    prepared = preprocessor.prepare(image)
    assert prepared.original_width == 120
    assert prepared.original_height == 100
    assert prepared.preprocessed_width >= 200 or prepared.preprocessed_height >= 150
    assert prepared.square_padding_applied is False


def test_caption_parsing_black_sedan_and_white_suv_and_generic() -> None:
    sedan = parse_caption_attributes("A black sedan driving on the road.")
    suv = parse_caption_attributes("A white sport utility vehicle parked near a building.")
    generic = parse_caption_attributes("A vehicle travelling on a road.")
    assert sedan.normalized_colour == "BLACK"
    assert sedan.normalized_body_type == "SEDAN"
    assert suv.normalized_colour == "WHITE"
    assert suv.normalized_body_type == "SUV"
    assert generic.normalized_colour == "UNKNOWN"
    assert generic.normalized_body_type == "UNKNOWN"


def test_word_boundaries_avoid_false_matches() -> None:
    parsed = parse_caption_attributes("An advantage is visible in infrared lighting.")
    assert parsed.normalized_body_type == "UNKNOWN"
    assert parsed.normalized_colour == "UNKNOWN"


def test_combined_format_and_natural_language_parse_both_attributes() -> None:
    combined = parse_caption_attributes("COLOUR: BLACK; BODY_TYPE: MPV")
    natural = parse_caption_attributes("A dark grey sport utility vehicle")
    pink = parse_caption_attributes("A pink hatchback parked by the road")
    assert combined.normalized_colour == "BLACK"
    assert combined.normalized_body_type == "MPV"
    assert natural.normalized_colour == "GREY"
    assert natural.normalized_body_type == "SUV"
    assert pink.normalized_colour == "PINK"
    assert pink.normalized_body_type == "HATCHBACK"


def test_plate_like_response_returns_unknown_attributes() -> None:
    parsed = parse_caption_attributes("DL4BC2038")
    assert parsed.normalized_colour == "UNKNOWN"
    assert parsed.normalized_body_type == "UNKNOWN"
    assert parsed.colour_reason == ATTRIBUTE_REASON_PLATE_LIKE
    assert parsed.body_type_reason == ATTRIBUTE_REASON_PLATE_LIKE


def test_ocr_mukul_flow_reuses_one_caption_for_both_attributes(tmp_path: Path) -> None:
    backend = _FakeBackend("A black sedan driving on the road.")
    image_size_policy = normalize_image_size_policy(
        {"florence": {"minimum_original_width": 100, "minimum_original_height": 80, "preferred_original_width": 320, "preferred_original_height": 240, "pad_to_square": True}},
        fallback_body_type={"minimum_crop_width": 100, "minimum_crop_height": 80},
        fallback_colour={"minimum_crop_width": 100, "minimum_crop_height": 80},
        detection={},
    )
    flow = OCRMukulFlorenceFlow(
        {"enabled": True, "task_token": "<CAPTION>", "reuse_caption_for_attributes": True, "maximum_crops_per_track": 3},
        backend=backend,
        image_size_policy=image_size_policy,
        logger=__import__("logging").getLogger(__name__),
    )
    result = flow.classify(_request(tmp_path))
    assert backend.load_calls == 1
    assert backend.run_calls == 1
    assert result.body_type.label == "SEDAN"
    assert result.colour.label == "BLACK"
    assert result.caption_inference_count == 1
    assert flow.metrics["ocr_mukul_caption_reused_for_body_type"] == 1
    assert flow.metrics["ocr_mukul_caption_reused_for_colour"] == 1


def test_ocr_mukul_flow_fails_closed_without_adapter(tmp_path: Path) -> None:
    backend = _FakeBackend("A black sedan driving on the road.", adapter_active=False)
    image_size_policy = normalize_image_size_policy(
        {"florence": {"minimum_original_width": 100, "minimum_original_height": 80, "preferred_original_width": 320, "preferred_original_height": 240, "pad_to_square": True}},
        fallback_body_type={"minimum_crop_width": 100, "minimum_crop_height": 80},
        fallback_colour={"minimum_crop_width": 100, "minimum_crop_height": 80},
        detection={},
    )
    flow = OCRMukulFlorenceFlow(
        {"enabled": True, "task_token": "<CAPTION>", "reuse_caption_for_attributes": True, "maximum_crops_per_track": 3},
        backend=backend,
        image_size_policy=image_size_policy,
        logger=__import__("logging").getLogger(__name__),
    )
    try:
        flow.classify(_request(tmp_path))
    except RuntimeError as exc:
        assert "requires the Florence adapter" in str(exc)
    else:
        raise AssertionError("Expected adapter-missing OCR_MUKUL flow to fail.")
