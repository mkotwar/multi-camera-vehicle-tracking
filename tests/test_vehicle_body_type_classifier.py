from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from src.vehicle_enrichment.body_type.classifier import BODY_TYPE_PROMPT_TEXT, BODY_TYPE_TASK_PROMPT, VehicleBodyTypeClassifier
from src.vehicle_enrichment.schemas import EnrichmentEvidenceItem, TrackEnrichmentRequest


class FakeBackend:
    def __init__(self, responses=None):
        self.responses = list(responses or [])
        self.calls = []
        self.model_identifier = "microsoft/Florence-2-base-ft"
        self.adapter_active = False

    def run_task(self, image, task_prompt, text_input=None):
        self.calls.append({"shape": image.shape, "task_prompt": task_prompt, "text_input": text_input})
        response = self.responses.pop(0) if self.responses else "UNKNOWN"
        return {
            "status": "completed",
            "reason": None,
            "payload": {
                "generated_text": response,
                "decoded_generated_only_text": response,
                "decoded_generated_only_text_skip_special": response,
                "parsed_answer": {task_prompt: response},
                "model_identifier": self.model_identifier,
                "adapter_active": self.adapter_active,
                "inference_duration_ms": 10.0,
            },
        }


class FakeLogger:
    def info(self, *args, **kwargs):
        return None

    def debug(self, *args, **kwargs):
        return None


def _write_image(path: Path, size=(240, 320)) -> str:
    image = np.full((size[0], size[1], 3), 120, dtype=np.uint8)
    cv2.imwrite(str(path), image)
    return str(path)


def _item(tmp_path: Path, name: str, *, width=320, height=240, quality=0.9, role="BEST_OVERALL") -> EnrichmentEvidenceItem:
    crop_path = _write_image(tmp_path / f"{name}.jpg", size=(height, width))
    return EnrichmentEvidenceItem(
        local_track_id="CAM_001:TRACK_1",
        camera_id="CAM_001",
        native_tracker_id=1,
        frame_number=1,
        timestamp_seconds=0.1,
        source_image_path=crop_path,
        vehicle_crop_path=crop_path,
        annotated_frame_path=None,
        bbox_xyxy=(0.0, 0.0, float(width), float(height)),
        evidence_role=role,
        detection_confidence=0.9,
        crop_width=width,
        crop_height=height,
        crop_area=width * height,
        sharpness_score=100.0,
        brightness_score=120.0,
        border_penalty=0.0,
        clipping_ratio=0.0,
        quality_score=quality,
        rejection_reasons=[],
    )


def _request(tmp_path: Path, *, vehicle_class="CAR", items=None) -> TrackEnrichmentRequest:
    return TrackEnrichmentRequest(
        local_track_id="CAM_001:TRACK_1",
        camera_id="CAM_001",
        native_tracker_id=1,
        vehicle_class=vehicle_class,
        vehicle_class_confidence=0.9,
        track_status="COMPLETED",
        completion_reason="END_OF_STREAM",
        started_at_seconds=0.0,
        ended_at_seconds=1.0,
        evidence_items=items or [_item(tmp_path, "crop1")],
    )


def _classifier(backend: FakeBackend) -> VehicleBodyTypeClassifier:
    return VehicleBodyTypeClassifier(
        {
            "enabled": True,
            "backend": "florence2",
            "run_only_when_vehicle_class": ["CAR"],
            "maximum_crops_per_track": 2,
            "minimum_crop_width": 256,
            "minimum_crop_height": 192,
            "allowed_labels": ["SUV", "SEDAN", "HATCHBACK", "MPV", "VAN", "PICKUP", "COUPE", "CONVERTIBLE", "WAGON", "UNKNOWN"],
        },
        backend=backend,
        logger=FakeLogger(),
    )


def test_only_car_is_eligible(tmp_path: Path) -> None:
    classifier = _classifier(FakeBackend())
    for vehicle_class in ["TRUCK", "MOTORCYCLE", "3WHEELER", "BUS", "UNKNOWN"]:
        result = classifier.classify(_request(tmp_path, vehicle_class=vehicle_class))
        assert result.status == "skipped"
        assert result.reason == "non_car_vehicle"


def test_crop_dimension_eligibility_and_maximum_crop_count(tmp_path: Path) -> None:
    backend = FakeBackend(["SUV", "SUV", "SUV"])
    classifier = _classifier(backend)
    items = [
        _item(tmp_path, "big1", width=320, height=240, quality=0.9),
        _item(tmp_path, "big2", width=256, height=192, quality=0.8),
        _item(tmp_path, "small", width=100, height=80, quality=0.95),
    ]
    result = classifier.classify(_request(tmp_path, items=items))
    assert result.label == "SUV"
    assert len(backend.calls) == 2


def test_exact_minimum_florence_size_is_accepted(tmp_path: Path) -> None:
    backend = FakeBackend(["SEDAN"])
    classifier = _classifier(backend)
    result = classifier.classify(_request(tmp_path, items=[_item(tmp_path, "exact_min", width=256, height=192)]))
    assert result.status == "completed"
    assert result.label == "SEDAN"


def test_below_minimum_florence_size_is_rejected(tmp_path: Path) -> None:
    classifier = _classifier(FakeBackend())
    result = classifier.classify(_request(tmp_path, items=[_item(tmp_path, "too_narrow", width=255, height=192)]))
    assert result.status == "skipped"
    assert result.reason == "no_body_type_usable_crop"


def test_prompt_construction_uses_vqa_and_constrained_text(tmp_path: Path) -> None:
    backend = FakeBackend(["SUV"])
    classifier = _classifier(backend)
    classifier.classify(_request(tmp_path))
    assert backend.calls[0]["task_prompt"] == BODY_TYPE_TASK_PROMPT
    assert backend.calls[0]["text_input"] == BODY_TYPE_PROMPT_TEXT


def test_synonym_and_ambiguous_normalization_rules(tmp_path: Path) -> None:
    classifier = _classifier(FakeBackend())
    assert classifier.normalize_label("sport utility vehicle") == ("SUV", "exact_phrase_match")
    assert classifier.normalize_label("saloon") == ("SEDAN", "exact_phrase_match")
    assert classifier.normalize_label("hatch back") == ("HATCHBACK", "exact_phrase_match")
    assert classifier.normalize_label("pickup truck") == ("PICKUP", "exact_phrase_match")
    assert classifier.normalize_label("station wagon") == ("WAGON", "exact_phrase_match")
    assert classifier.normalize_label("cabriolet") == ("CONVERTIBLE", "exact_phrase_match")
    assert classifier.normalize_label("coupé") == ("COUPE", "exact_phrase_match")
    assert classifier.normalize_label("van sedan") == ("UNKNOWN", "ambiguous_multiple_labels")
    assert classifier.normalize_label("sedan") == ("SEDAN", "exact_phrase_match")
    assert classifier.normalize_label("The closest body type is SUV.") == ("SUV", "contained_phrase_match")
    assert classifier.normalize_label("car") == ("UNKNOWN", "unknown_phrase")
    assert classifier.normalize_label("unanswerable") == ("UNKNOWN", "unknown_phrase")
    assert classifier.normalize_label("qA") == ("UNKNOWN", "unexpected_output")


def test_weighted_agreement_and_conflict_rules(tmp_path: Path) -> None:
    agree_backend = FakeBackend(["SUV", "SUV"])
    agree_classifier = _classifier(agree_backend)
    agree_result = agree_classifier.classify(
        _request(tmp_path, items=[_item(tmp_path, "a1", quality=0.9), _item(tmp_path, "a2", quality=0.8)])
    )
    assert agree_result.label == "SUV"
    assert agree_result.aggregation_reason == "weighted_agreement"

    conflict_backend = FakeBackend(["SEDAN", "HATCHBACK"])
    conflict_classifier = _classifier(conflict_backend)
    conflict_result = conflict_classifier.classify(
        _request(tmp_path, items=[_item(tmp_path, "c1", quality=0.84), _item(tmp_path, "c2", quality=0.82)])
    )
    assert conflict_result.label == "UNKNOWN"
    assert conflict_result.aggregation_reason == "conflicting_high_quality_predictions"


def test_failed_crop_does_not_fail_complete_track_and_confidence_is_not_invented(tmp_path: Path) -> None:
    class ErrorBackend(FakeBackend):
        def run_task(self, image, task_prompt, text_input=None):
            self.calls.append({"shape": image.shape, "task_prompt": task_prompt, "text_input": text_input})
            return {"status": "error", "reason": "boom", "payload": None}

    backend = ErrorBackend()
    classifier = _classifier(backend)
    result = classifier.classify(_request(tmp_path))
    assert result.label == "UNKNOWN"
    assert result.predictions[0].status == "error"
    assert result.predictions[0].confidence is None


def test_parsed_answer_dictionary_values_are_handled(tmp_path: Path) -> None:
    class ParsedBackend(FakeBackend):
        def run_task(self, image, task_prompt, text_input=None):
            self.calls.append({"shape": image.shape, "task_prompt": task_prompt, "text_input": text_input})
            return {
                "status": "completed",
                "reason": None,
                "payload": {
                    "generated_text": "qA",
                    "decoded_generated_only_text": "qA",
                    "parsed_answer": {task_prompt: {"answer": "sedan"}},
                    "model_identifier": self.model_identifier,
                    "adapter_active": self.adapter_active,
                    "inference_duration_ms": 10.0,
                },
            }

    classifier = _classifier(ParsedBackend())
    result = classifier.classify(_request(tmp_path))

    assert result.label == "SEDAN"
