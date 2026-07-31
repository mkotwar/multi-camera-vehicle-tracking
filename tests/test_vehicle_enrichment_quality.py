from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from src.vehicle_enrichment.evidence_quality import EvidenceQualityEvaluator, normalize_quality_config
from src.vehicle_enrichment.schemas import EnrichmentEvidenceItem


def _write_image(path: Path, image: np.ndarray) -> Path:
    cv2.imwrite(str(path), image)
    return path


def _item(path: Path, *, role: str = "BEST_OVERALL", border_penalty: float = 0.0, clipping_ratio: float = 0.0) -> EnrichmentEvidenceItem:
    return EnrichmentEvidenceItem(
        local_track_id="CAM_001:TRACK_1",
        camera_id="CAM_001",
        native_tracker_id=1,
        frame_number=1,
        timestamp_seconds=0.1,
        source_image_path=str(path),
        vehicle_crop_path=str(path),
        annotated_frame_path=None,
        bbox_xyxy=(0.0, 0.0, 40.0, 30.0),
        evidence_role=role,
        detection_confidence=0.9,
        crop_width=0,
        crop_height=0,
        crop_area=0,
        sharpness_score=0.0,
        brightness_score=0.0,
        border_penalty=border_penalty,
        clipping_ratio=clipping_ratio,
        quality_score=0.0,
        rejection_reasons=[],
    )


def test_quality_scoring_is_between_zero_and_one_and_rewards_sharp_images(tmp_path: Path) -> None:
    config = normalize_quality_config({"evidence": {"minimum_crop_width": 20, "minimum_crop_height": 20, "minimum_sharpness": 5.0}})
    evaluator = EvidenceQualityEvaluator(config)
    blurred = np.full((40, 40, 3), 120, dtype=np.uint8)
    sharp = blurred.copy()
    sharp[:, ::2] = 255
    blurred_path = _write_image(tmp_path / "blurred.jpg", blurred)
    sharp_path = _write_image(tmp_path / "sharp.jpg", sharp)

    blurred_item = evaluator.score_item(_item(blurred_path))
    sharp_item = evaluator.score_item(_item(sharp_path))

    assert 0.0 <= blurred_item.quality_score <= 1.0
    assert 0.0 <= sharp_item.quality_score <= 1.0
    assert sharp_item.sharpness_score > blurred_item.sharpness_score
    assert sharp_item.quality_score > blurred_item.quality_score


def test_dark_and_bright_crops_receive_penalties(tmp_path: Path) -> None:
    config = normalize_quality_config({"evidence": {"minimum_crop_width": 10, "minimum_crop_height": 10}})
    evaluator = EvidenceQualityEvaluator(config)
    dark_item = evaluator.score_item(_item(_write_image(tmp_path / "dark.jpg", np.full((20, 20, 3), 5, dtype=np.uint8))))
    bright_item = evaluator.score_item(_item(_write_image(tmp_path / "bright.jpg", np.full((20, 20, 3), 250, dtype=np.uint8))))
    normal_item = evaluator.score_item(_item(_write_image(tmp_path / "normal.jpg", np.full((20, 20, 3), 120, dtype=np.uint8))))

    assert dark_item.brightness_score < 40.0
    assert bright_item.brightness_score > 215.0
    assert normal_item.quality_score > dark_item.quality_score
    assert normal_item.quality_score > bright_item.quality_score


def test_border_penalty_clipping_penalty_and_role_bonus_affect_score(tmp_path: Path) -> None:
    config = normalize_quality_config({"evidence": {"minimum_crop_width": 10, "minimum_crop_height": 10}})
    evaluator = EvidenceQualityEvaluator(config)
    image_path = _write_image(tmp_path / "base.jpg", np.full((30, 30, 3), 120, dtype=np.uint8))

    best = evaluator.score_item(_item(image_path, role="BEST_OVERALL", border_penalty=0.0, clipping_ratio=0.0))
    weaker = evaluator.score_item(_item(image_path, role="FIRST", border_penalty=0.8, clipping_ratio=0.5))

    assert best.quality_score > weaker.quality_score
    assert weaker.border_penalty == 0.8
    assert weaker.clipping_ratio == 0.5
