from __future__ import annotations

import importlib.util
from pathlib import Path

import cv2
import numpy as np

from src.vehicle_enrichment.body_type.classifier import VehicleBodyTypeClassifier
from src.vehicle_enrichment.colour.classifier import (
    COLOUR_PROMPT_TEXT,
    COLOUR_TASK_PROMPT,
    VehicleColourClassifier,
    get_colour_prompt_variants,
)
from src.vehicle_enrichment.colour.search_aliases import expand_colour_search_labels
from src.vehicle_enrichment.schemas import EnrichmentEvidenceItem, TrackEnrichmentRequest


def _load_benchmark_module():
    module_path = Path("scripts/benchmark_florence_vehicle_colour_prompts.py").resolve()
    spec = importlib.util.spec_from_file_location("benchmark_florence_vehicle_colour_prompts", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeBackend:
    def __init__(self, responses=None):
        self.responses = list(responses or [])
        self.calls = []
        self.model_identifier = "microsoft/Florence-2-base-ft"
        self.adapter_active = False
        self.loaded = False
        self.load_attempts = 0
        self.metrics = {"florence_load_attempts": 0, "florence_load_successes": 0}

    def load(self):
        self.load_attempts += 1
        self.loaded = True
        self.metrics["florence_load_attempts"] += 1
        self.metrics["florence_load_successes"] += 1

    def run_task(self, image, task_prompt, text_input=None):
        if not self.loaded:
            self.load()
        self.calls.append({"shape": image.shape, "task_prompt": task_prompt, "text_input": text_input})
        response = self.responses.pop(0) if self.responses else "UNKNOWN"
        if isinstance(response, Exception):
            return {"status": "error", "reason": str(response), "payload": None}
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
                "inference_duration_ms": 12.5,
            },
        }


class FakeLogger:
    def info(self, *args, **kwargs):
        return None

    def debug(self, *args, **kwargs):
        return None


def _write_image(path: Path, *, size=(220, 320), value=160) -> str:
    image = np.full((size[0], size[1], 3), value, dtype=np.uint8)
    cv2.imwrite(str(path), image)
    return str(path)


def _item(tmp_path: Path, name: str, *, width=320, height=220, quality=0.9, role="BEST_OVERALL") -> EnrichmentEvidenceItem:
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


def _classifier(backend: FakeBackend, *, enabled=True, **overrides) -> VehicleColourClassifier:
    config = {
        "enabled": enabled,
        "backend": "florence2",
        "run_only_when_vehicle_class": ["3WHEELER", "BUS", "CAR", "MOTORCYCLE", "TRUCK"],
        "maximum_crops_per_track": 2,
        "minimum_crop_width": 256,
        "minimum_crop_height": 192,
        "allowed_labels": [
            "BLACK",
            "WHITE",
            "GREY",
            "SILVER",
            "RED",
            "PINK",
            "BLUE",
            "GREEN",
            "YELLOW",
            "ORANGE",
            "BROWN",
            "BEIGE",
            "PURPLE",
            "OTHER",
            "UNKNOWN",
        ],
        "retry_on_invalid_response": True,
        "maximum_prompt_attempts": 2,
    }
    config.update(overrides)
    return VehicleColourClassifier(config, backend=backend, logger=FakeLogger())


def test_colour_disabled_returns_disabled_status(tmp_path: Path) -> None:
    classifier = _classifier(FakeBackend(), enabled=False)
    result = classifier.classify(_request(tmp_path))
    assert result.status == "disabled"
    assert result.label == "UNKNOWN"


def test_all_prompt_variants_execute_through_same_backend(tmp_path: Path) -> None:
    backend = FakeBackend(["white"] * len(get_colour_prompt_variants(include_no_task_prefix_variant=True)))
    classifier = _classifier(backend)
    image_path = Path(_item(tmp_path, "prompt_variants").vehicle_crop_path)
    image = cv2.imread(str(image_path))

    for variant in get_colour_prompt_variants(include_no_task_prefix_variant=True):
        attempt = classifier._run_single_prompt_attempt(image, variant)
        assert attempt["status"] == "completed"

    assert backend.load_attempts == 1
    assert len(backend.calls) == len(get_colour_prompt_variants(include_no_task_prefix_variant=True))


def test_enabled_colour_classifier_calls_shared_backend(tmp_path: Path) -> None:
    backend = FakeBackend(["white"])
    classifier = _classifier(backend)
    result = classifier.classify(_request(tmp_path))
    assert result.status == "completed"
    assert result.label == "WHITE"
    assert backend.calls[0]["task_prompt"] == COLOUR_TASK_PROMPT
    assert backend.calls[0]["text_input"] == classifier.primary_prompt_variant["prompt_text"]
    assert classifier.primary_prompt_variant["prompt_text"] != COLOUR_PROMPT_TEXT or backend.calls[0]["text_input"] == COLOUR_PROMPT_TEXT


def test_body_type_and_colour_share_same_backend_and_load_once(tmp_path: Path) -> None:
    backend = FakeBackend(["sedan", "white"])
    body_type_classifier = VehicleBodyTypeClassifier(
        {
            "enabled": True,
            "backend": "florence2",
            "run_only_when_vehicle_class": ["CAR"],
            "maximum_crops_per_track": 2,
            "minimum_crop_width": 100,
            "minimum_crop_height": 80,
            "allowed_labels": ["SUV", "SEDAN", "HATCHBACK", "MPV", "VAN", "PICKUP", "COUPE", "CONVERTIBLE", "WAGON", "UNKNOWN"],
        },
        backend=backend,
        logger=FakeLogger(),
    )
    colour_classifier = _classifier(backend)
    request = _request(tmp_path)

    body_result = body_type_classifier.classify(request)
    colour_result = colour_classifier.classify(request)

    assert body_type_classifier.backend is colour_classifier.backend
    assert body_result.label == "SEDAN"
    assert colour_result.label == "WHITE"
    assert backend.load_attempts == 1


def test_valid_responses_normalize_correctly() -> None:
    classifier = _classifier(FakeBackend())
    assert classifier.normalize_label("white vehicle") == ("WHITE", "contained_phrase_match")
    assert classifier.normalize_label("the vehicle is white") == ("WHITE", "contained_phrase_match")
    assert classifier.normalize_label("main colour is blue") == ("BLUE", "contained_phrase_match")
    assert classifier.normalize_label("WHITE") == ("WHITE", "exact_phrase_match")


def test_gray_normalizes_to_grey() -> None:
    classifier = _classifier(FakeBackend())
    assert classifier.normalize_label("gray") == ("GREY", "exact_phrase_match")
    assert classifier.normalize_label("metallic gray") == ("GREY", "exact_phrase_match")
    assert classifier.normalize_label("dark gray") == ("GREY", "exact_phrase_match")


def test_pink_normalizes_to_pink() -> None:
    classifier = _classifier(FakeBackend())
    assert classifier.normalize_label("pink") == ("PINK", "exact_phrase_match")
    assert classifier.normalize_label("PINK") == ("PINK", "exact_phrase_match")
    assert classifier.normalize_label("Pink vehicle") == ("PINK", "contained_phrase_match")


def test_generic_responses_remain_unknown() -> None:
    classifier = _classifier(FakeBackend())
    assert classifier.normalize_label("qA") == ("UNKNOWN", "generic_invalid_response")
    assert classifier.normalize_label("yes") == ("UNKNOWN", "generic_invalid_response")
    assert classifier.normalize_label("no") == ("UNKNOWN", "generic_invalid_response")


def test_unexpected_and_unknown_outputs_return_unknown() -> None:
    classifier = _classifier(FakeBackend())
    assert classifier.normalize_label("unanswerable") == ("UNKNOWN", "unknown_phrase")
    assert classifier.normalize_label("hyundai") == ("UNKNOWN", "unexpected_output")
    assert classifier.normalize_label("sedan") == ("UNKNOWN", "unexpected_output")
    assert classifier.normalize_label("vehicle") == ("UNKNOWN", "generic_invalid_response")


def test_word_boundary_parsing_prevents_false_positives() -> None:
    classifier = _classifier(FakeBackend())
    assert classifier.normalize_label("bluetooth") == ("UNKNOWN", "unexpected_output")
    assert classifier.normalize_label("greenish") == ("UNKNOWN", "unexpected_output")


def test_multiple_colour_labels_and_uncertain_responses_return_unknown() -> None:
    classifier = _classifier(FakeBackend())
    assert classifier.normalize_label("black or blue") == ("UNKNOWN", "ambiguous_multiple_labels")
    assert classifier.normalize_label("white and silver") == ("UNKNOWN", "ambiguous_multiple_labels")
    assert classifier.normalize_label("maybe red") == ("UNKNOWN", "uncertain_response")


def test_small_readable_crops_are_sent_to_florence_as_fallback(tmp_path: Path) -> None:
    classifier = _classifier(FakeBackend(["white"]))
    result = classifier.classify(_request(tmp_path, items=[_item(tmp_path, "small", width=60, height=50)]))
    assert result.status == "completed"
    assert result.label == "WHITE"


def test_exact_minimum_florence_size_is_accepted(tmp_path: Path) -> None:
    classifier = _classifier(FakeBackend(["white"]))
    result = classifier.classify(_request(tmp_path, items=[_item(tmp_path, "exact_min", width=256, height=192)]))
    assert result.status == "completed"
    assert result.label == "WHITE"


def test_below_minimum_florence_size_is_used_as_fallback(tmp_path: Path) -> None:
    classifier = _classifier(FakeBackend(["white"]))
    result = classifier.classify(_request(tmp_path, items=[_item(tmp_path, "too_short", width=256, height=191)]))
    assert result.status == "completed"
    assert result.label == "WHITE"


def test_only_readable_fallback_crops_selects_best_available(tmp_path: Path) -> None:
    backend = FakeBackend(["white", "white"])
    classifier = _classifier(backend, maximum_crops_per_track=2)
    low = _item(tmp_path, "low", width=56, height=123, quality=0.35)
    better = _item(tmp_path, "better", width=79, height=101, quality=0.85)
    result = classifier.classify(_request(tmp_path, items=[low, better]))
    assert result.status == "completed"
    assert result.label == "WHITE"
    assert len(result.predictions) == 2


def test_missing_crops_are_handled_safely(tmp_path: Path) -> None:
    classifier = _classifier(FakeBackend())
    item = _item(tmp_path, "missing")
    item.vehicle_crop_path = str(tmp_path / "does_not_exist.jpg")
    result = classifier.classify(_request(tmp_path, items=[item]))
    assert result.label == "UNKNOWN"
    assert result.predictions[0].status == "error"
    assert result.predictions[0].reason == "missing_crop_image"


def test_colour_runs_for_all_supported_vehicle_classes(tmp_path: Path) -> None:
    for vehicle_class in ["3WHEELER", "BUS", "CAR", "MOTORCYCLE", "TRUCK"]:
        backend = FakeBackend(["white"])
        classifier = _classifier(backend)
        result = classifier.classify(_request(tmp_path, vehicle_class=vehicle_class))
        assert result.status == "completed"
        assert result.label == "WHITE"


def test_final_unknown_vehicle_class_is_skipped(tmp_path: Path) -> None:
    classifier = _classifier(FakeBackend())
    result = classifier.classify(_request(tmp_path, vehicle_class="UNKNOWN"))
    assert result.status == "skipped"
    assert result.reason == "vehicle_class_unknown"


def test_retry_happens_only_for_invalid_responses_and_records_both_attempts(tmp_path: Path) -> None:
    backend = FakeBackend(["qA", "white"])
    classifier = _classifier(backend)
    result = classifier.classify(_request(tmp_path))
    assert result.label == "WHITE"
    assert len(backend.calls) == 2
    assert isinstance(result.predictions[0].raw_response, list)
    assert result.predictions[0].raw_response[0]["raw_response"] == "qA"
    assert result.predictions[0].raw_response[1]["raw_response"] == "white"
    assert classifier.metrics["colour_retry_count"] == 1
    assert classifier.metrics["colour_retry_success_count"] == 1


def test_retry_does_not_happen_for_valid_responses(tmp_path: Path) -> None:
    backend = FakeBackend(["white"])
    classifier = _classifier(backend)
    result = classifier.classify(_request(tmp_path))
    assert result.label == "WHITE"
    assert len(backend.calls) == 1
    assert classifier.metrics["colour_retry_count"] == 0


def test_maximum_attempts_are_respected(tmp_path: Path) -> None:
    backend = FakeBackend(["qA", "yes", "white"])
    classifier = _classifier(backend, maximum_prompt_attempts=2)
    result = classifier.classify(_request(tmp_path))
    assert result.label == "UNKNOWN"
    assert len(backend.calls) == 2


def test_matching_conflicting_and_mixed_predictions_aggregate_correctly(tmp_path: Path) -> None:
    matching = _classifier(FakeBackend(["white", "white"]))
    matching_result = matching.classify(
        _request(tmp_path, items=[_item(tmp_path, "m1", quality=0.9), _item(tmp_path, "m2", quality=0.8)])
    )
    assert matching_result.label == "WHITE"
    assert matching_result.aggregation_reason == "weighted_agreement"

    conflicting = _classifier(FakeBackend(["white", "silver"]))
    conflicting_result = conflicting.classify(
        _request(tmp_path, items=[_item(tmp_path, "c1", quality=0.84), _item(tmp_path, "c2", quality=0.82)])
    )
    assert conflicting_result.label == "UNKNOWN"
    assert conflicting_result.aggregation_reason == "conflicting_high_quality_predictions"

    mixed = _classifier(FakeBackend(["white", "qA"]))
    mixed_result = mixed.classify(
        _request(tmp_path, items=[_item(tmp_path, "x1", quality=0.9), _item(tmp_path, "x2", quality=0.2)])
    )
    assert mixed_result.label == "WHITE"


def test_all_pink_predictions_aggregate_to_pink(tmp_path: Path) -> None:
    classifier = _classifier(FakeBackend(["pink", "pink", "pink"]), maximum_crops_per_track=3)
    result = classifier.classify(
        _request(
            tmp_path,
            items=[
                _item(tmp_path, "p1", quality=0.9),
                _item(tmp_path, "p2", quality=0.8),
                _item(tmp_path, "p3", quality=0.7),
            ],
        )
    )
    assert result.label == "PINK"


def test_colour_search_aliases_expand_red_to_red_and_pink_only() -> None:
    assert expand_colour_search_labels("RED") == ("RED", "PINK")
    assert expand_colour_search_labels("PINK") == ("PINK",)
    assert expand_colour_search_labels("BLUE") == ("BLUE",)


def test_raw_responses_and_crop_paths_are_preserved(tmp_path: Path) -> None:
    backend = FakeBackend(["the colour is white"])
    classifier = _classifier(backend)
    item = _item(tmp_path, "crop_preserved")
    result = classifier.classify(_request(tmp_path, items=[item]))
    assert result.predictions[0].raw_response == "the colour is white"
    assert result.predictions[0].source_crop_path == item.vehicle_crop_path
    assert result.predictions[0].evidence_role == item.evidence_role


def test_colour_metrics_are_recorded(tmp_path: Path) -> None:
    backend = FakeBackend(["white", "black or blue", "qA", "white"])
    classifier = _classifier(backend)
    classifier.classify(_request(tmp_path, items=[_item(tmp_path, "ok1")]))
    classifier.classify(_request(tmp_path, items=[_item(tmp_path, "bad1")]))
    metrics = classifier.metrics
    assert metrics["colour_eligible_tracks"] == 2
    assert metrics["colour_tracks_processed"] == 2
    assert metrics["colour_prompt_attempts"] == 3
    assert metrics["colour_retry_count"] == 1
    assert metrics["colour_generic_response_count"] == 1
    assert metrics["colour_invalid_response_count"] >= 2
    assert metrics["colour_labels"]["WHITE"] == 1
    assert metrics["colour_tracks_unknown"] == 1
    assert metrics["colour_unknown_reasons"]["no_valid_predictions"] == 1
    assert metrics["colour_raw_response_counts"]["white"] == 1


def test_prompt_benchmark_writes_csv_json_and_review_fields(tmp_path: Path) -> None:
    module = _load_benchmark_module()
    crop_dir = tmp_path / "crops"
    crop_dir.mkdir()
    _write_image(crop_dir / "image1.jpg")
    backend = FakeBackend(["white"] * 6)
    output_dir = tmp_path / "benchmark_out"

    result = module.run_benchmark(input_path=crop_dir, output_dir=output_dir, device="cpu", backend=backend)

    assert Path(result["csv_path"]).exists()
    assert Path(result["json_path"]).exists()
    assert Path(result["manual_review_path"]).exists()
    assert Path(result["comparison_path"]).exists()
    manual_review_text = Path(result["manual_review_path"]).read_text(encoding="utf-8")
    assert "manual_colour" in manual_review_text
    assert "review_notes" in manual_review_text
    assert backend.load_attempts == 1
