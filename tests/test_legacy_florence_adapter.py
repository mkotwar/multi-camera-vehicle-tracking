from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from src.vehicle_enrichment.legacy_florence import (
    OldTdCase2CropPreprocessor,
    OldTdCase2Adapter,
    inspect_old_reference_project,
    parse_old_td_case2_caption,
)
from src.vehicle_enrichment.schemas import EnrichmentEvidenceItem


class _FakeProcessor:
    def __call__(self, *, text, images, return_tensors):
        self.last_text = text
        self.last_image_shape = list(images.shape)
        return {"input_ids": _FakeTensor([1, 2]), "pixel_values": _FakeTensor([1, 3, images.shape[0], images.shape[1]])}

    def batch_decode(self, generated_ids, skip_special_tokens):
        return ["a white sedan vehicle"]

    def post_process_generation(self, decoded, task, image_size):
        return {task: decoded}


class _FakeModel:
    def generate(self, **kwargs):
        return _FakeTensor([1, 2, 3])


class _FakeRuntimeDevice:
    device = "cpu"
    dtype = None


class _FakeTensor(list):
    @property
    def shape(self):
        return tuple(self)

    def to(self, *args, **kwargs):
        return self


class _FakeBackend:
    def __init__(self):
        self._processor = _FakeProcessor()
        self._model = _FakeModel()
        self._runtime_device = _FakeRuntimeDevice()
        self.config = type("Config", (), {"max_new_tokens": 64, "num_beams": 1})()

    def load(self):
        return None


def test_parse_old_caption_uses_caption_keywords() -> None:
    parsed = parse_old_td_case2_caption("a white sedan vehicle on the road")
    assert parsed.normalized_body_type == "SEDAN"
    assert parsed.normalized_colour == "WHITE"


def test_old_crop_preprocessor_keeps_original_dimensions(tmp_path: Path) -> None:
    image_path = tmp_path / "crop.jpg"
    image = np.full((48, 96, 3), 120, dtype=np.uint8)
    cv2.imwrite(str(image_path), image)

    result = OldTdCase2CropPreprocessor().prepare_from_crop_path(image_path)
    assert result.original_width == 96
    assert result.original_height == 48
    assert result.manual_resize_applied is False
    assert result.manual_square_padding_applied is False


def test_old_adapter_selection_prefers_spread_out_scored_crops() -> None:
    adapter = OldTdCase2Adapter(_FakeBackend())
    items = []
    for frame_number, confidence, area in [(10, 0.8, 2000), (11, 0.95, 2200), (30, 0.7, 1800)]:
        items.append(
            EnrichmentEvidenceItem(
                local_track_id="CAM_001:TRACK_1",
                camera_id="CAM_001",
                native_tracker_id=1,
                frame_number=frame_number,
                timestamp_seconds=float(frame_number),
                source_image_path=None,
                vehicle_crop_path=f"crop_{frame_number}.jpg",
                annotated_frame_path=None,
                bbox_xyxy=(0.0, 0.0, 10.0, 10.0),
                evidence_role="BEST",
                detection_confidence=confidence,
                crop_width=50,
                crop_height=40,
                crop_area=area,
                sharpness_score=0.0,
                brightness_score=0.0,
                border_penalty=0.1,
                clipping_ratio=0.0,
                quality_score=0.0,
            )
        )
    result = adapter.select_track_evidence(items, maximum_crops_per_track=2, minimum_frame_gap=3)
    assert result.selected_frame_numbers == [11, 30]


def test_old_adapter_uses_caption_prompt_and_same_crop(tmp_path: Path, monkeypatch) -> None:
    image_path = tmp_path / "crop.jpg"
    image = np.full((32, 64, 3), 80, dtype=np.uint8)
    cv2.imwrite(str(image_path), image)
    monkeypatch.setattr("src.vehicle_enrichment.legacy_florence.old_td_case2_adapter.move_batch_to_device", lambda batch, device, dtype: batch)
    monkeypatch.setitem(__import__("sys").modules, "torch", type("TorchStub", (), {})())

    result = OldTdCase2Adapter(_FakeBackend()).run_caption_inference(image_path)
    assert result.task_prompt == "<CAPTION>"
    assert result.crop_path == str(image_path)
    assert result.body_type_label == "SEDAN"
    assert result.colour_label == "WHITE"


def test_old_reference_inspection_reports_missing_root(tmp_path: Path) -> None:
    missing = tmp_path / "missing_old_project"
    try:
        inspect_old_reference_project(missing)
    except FileNotFoundError as exc:
        assert str(missing) in str(exc)
    else:
        raise AssertionError("Expected FileNotFoundError for missing old project root")
