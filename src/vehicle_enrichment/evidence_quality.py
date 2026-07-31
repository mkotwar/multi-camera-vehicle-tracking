from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .schemas import EnrichmentEvidenceItem


ROLE_BONUS_SCORES = {
    "BEST_OVERALL": 1.0,
    "SHARPEST": 0.95,
    "LARGEST": 0.90,
    "HIGHEST_CONFIDENCE": 0.85,
    "MIDDLE": 0.75,
    "FIRST": 0.65,
    "LAST": 0.65,
}


@dataclass(slots=True, frozen=True)
class EvidenceQualityConfig:
    minimum_crop_width: int
    minimum_crop_height: int
    minimum_sharpness: float
    minimum_quality_score: float
    border_margin_ratio: float
    area_weight: float
    sharpness_weight: float
    confidence_weight: float
    role_weight: float
    border_weight: float
    clipping_weight: float
    brightness_weight: float


def normalize_quality_config(raw_config: dict[str, Any]) -> EvidenceQualityConfig:
    evidence = dict(raw_config.get("evidence", {}) or {})
    scoring = dict(evidence.get("scoring", {}) or {})
    return EvidenceQualityConfig(
        minimum_crop_width=int(evidence.get("minimum_crop_width", 100)),
        minimum_crop_height=int(evidence.get("minimum_crop_height", 70)),
        minimum_sharpness=float(evidence.get("minimum_sharpness", 10.0)),
        minimum_quality_score=float(evidence.get("minimum_quality_score", 0.20)),
        border_margin_ratio=float(evidence.get("border_margin_ratio", 0.02)),
        area_weight=float(scoring.get("area_weight", 0.25)),
        sharpness_weight=float(scoring.get("sharpness_weight", 0.25)),
        confidence_weight=float(scoring.get("confidence_weight", 0.20)),
        role_weight=float(scoring.get("role_weight", 0.15)),
        border_weight=float(scoring.get("border_weight", 0.05)),
        clipping_weight=float(scoring.get("clipping_weight", 0.05)),
        brightness_weight=float(scoring.get("brightness_weight", 0.05)),
    )


class EvidenceQualityEvaluator:
    """Deterministic enrichment-crop scoring.

    Formula:
        score =
            area_weight * normalized_area_score
          + sharpness_weight * normalized_sharpness_score
          + confidence_weight * detection_confidence
          + role_weight * evidence_role_score
          - border_weight * border_penalty
          - clipping_weight * clipping_ratio
          - brightness_weight * brightness_penalty

    Where:
    - normalized_area_score is capped to [0, 1] relative to the configured minimum area.
    - normalized_sharpness_score is capped to [0, 1] relative to 4x the configured minimum sharpness.
    - brightness_penalty is 0 inside the valid band [40, 215] and rises toward 1 outside it.
    - final score is normalized to [0, 1].
    """

    def __init__(self, config: EvidenceQualityConfig) -> None:
        self.config = config

    def score_item(self, item: EnrichmentEvidenceItem) -> EnrichmentEvidenceItem:
        reasons = list(item.rejection_reasons)
        crop_path = Path(str(item.vehicle_crop_path)) if item.vehicle_crop_path else None
        image = self._load_crop_image(crop_path)
        if image is None:
            reasons.append("missing_crop_image")
            item.rejection_reasons = self._dedupe_reasons(reasons)
            item.quality_score = 0.0
            return item

        crop_height, crop_width = image.shape[:2]
        crop_area = int(crop_width * crop_height)
        sharpness = self.compute_sharpness(image)
        brightness = self.compute_brightness(image)
        border_penalty = max(0.0, min(1.0, float(item.border_penalty)))
        clipping_ratio = max(0.0, min(1.0, float(item.clipping_ratio)))
        aspect_ratio = float(crop_width / crop_height) if crop_height > 0 else 0.0

        if crop_width < self.config.minimum_crop_width:
            reasons.append("crop_width_below_minimum")
        if crop_height < self.config.minimum_crop_height:
            reasons.append("crop_height_below_minimum")
        if sharpness < self.config.minimum_sharpness:
            reasons.append("sharpness_below_minimum")
        if aspect_ratio <= 0.0 or aspect_ratio > 6.0 or aspect_ratio < 0.2:
            reasons.append("aspect_ratio_out_of_range")

        normalized_area_score = self._normalize_area(crop_area)
        normalized_sharpness_score = self._normalize_sharpness(sharpness)
        brightness_penalty = self._brightness_penalty(brightness)
        role_score = ROLE_BONUS_SCORES.get(str(item.evidence_role).upper(), 0.50)
        positive_weight_sum = max(
            1e-9,
            self.config.area_weight + self.config.sharpness_weight + self.config.confidence_weight + self.config.role_weight,
        )
        negative_weight_sum = max(
            1e-9,
            self.config.border_weight + self.config.clipping_weight + self.config.brightness_weight,
        )
        positive = (
            self.config.area_weight * normalized_area_score
            + self.config.sharpness_weight * normalized_sharpness_score
            + self.config.confidence_weight * max(0.0, min(1.0, float(item.detection_confidence)))
            + self.config.role_weight * role_score
        ) / positive_weight_sum
        negative = (
            self.config.border_weight * border_penalty
            + self.config.clipping_weight * clipping_ratio
            + self.config.brightness_weight * brightness_penalty
        ) / negative_weight_sum
        score = max(0.0, min(1.0, positive - (negative * 0.35)))

        if score < self.config.minimum_quality_score:
            reasons.append("quality_score_below_minimum")

        item.crop_width = crop_width
        item.crop_height = crop_height
        item.crop_area = crop_area
        item.sharpness_score = sharpness
        item.brightness_score = brightness
        item.quality_score = score
        item.rejection_reasons = self._dedupe_reasons(reasons)
        return item

    @staticmethod
    def compute_sharpness(image: np.ndarray) -> float:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    @staticmethod
    def compute_brightness(image: np.ndarray) -> float:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        return float(np.mean(gray))

    def _normalize_area(self, crop_area: int) -> float:
        minimum_area = max(1, self.config.minimum_crop_width * self.config.minimum_crop_height)
        target_area = max(minimum_area, minimum_area * 4)
        score = float((crop_area - minimum_area) / max(1, target_area - minimum_area))
        return max(0.0, min(1.0, score))

    def _normalize_sharpness(self, sharpness: float) -> float:
        target = max(1.0, self.config.minimum_sharpness * 4.0)
        return max(0.0, min(1.0, sharpness / target))

    @staticmethod
    def _brightness_penalty(brightness: float) -> float:
        if 40.0 <= brightness <= 215.0:
            return 0.0
        if brightness < 40.0:
            return max(0.0, min(1.0, (40.0 - brightness) / 40.0))
        return max(0.0, min(1.0, (brightness - 215.0) / 40.0))

    @staticmethod
    def _dedupe_reasons(reasons: list[str]) -> list[str]:
        seen: set[str] = set()
        ordered: list[str] = []
        for reason in reasons:
            normalized = str(reason).strip()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            ordered.append(normalized)
        return ordered

    @staticmethod
    def _load_crop_image(crop_path: Path | None) -> np.ndarray | None:
        if crop_path is None or not crop_path.exists():
            return None
        image = cv2.imread(str(crop_path))
        return image if image is not None and image.size > 0 else None
