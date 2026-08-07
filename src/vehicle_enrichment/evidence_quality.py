from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .image_size_policy import ImageSizePolicy, RESOLUTION_TIER_ACCEPTABLE, RESOLUTION_TIER_PREFERRED, normalize_image_size_policy
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
    minimum_brightness: float
    maximum_brightness: float
    maximum_edge_truncation_ratio: float
    area_weight: float
    sharpness_weight: float
    confidence_weight: float
    role_weight: float
    border_weight: float
    clipping_weight: float
    brightness_weight: float
    class_specific_minimums: dict[str, dict[str, int]]


def normalize_quality_config(raw_config: dict[str, Any]) -> EvidenceQualityConfig:
    evidence = dict(raw_config.get("evidence", {}) or {})
    scoring = dict(evidence.get("scoring", {}) or {})
    quality = dict(raw_config.get("evidence_quality", {}) or {})
    return EvidenceQualityConfig(
        minimum_crop_width=int(evidence.get("minimum_crop_width", 100)),
        minimum_crop_height=int(evidence.get("minimum_crop_height", 70)),
        minimum_sharpness=float(evidence.get("minimum_sharpness", 10.0)),
        minimum_quality_score=float(evidence.get("minimum_quality_score", 0.20)),
        border_margin_ratio=float(evidence.get("border_margin_ratio", 0.02)),
        minimum_brightness=float(quality.get("minimum_brightness", 35.0)),
        maximum_brightness=float(quality.get("maximum_brightness", 225.0)),
        maximum_edge_truncation_ratio=float(quality.get("maximum_edge_truncation_ratio", 0.15)),
        area_weight=float(scoring.get("area_weight", 0.25)),
        sharpness_weight=float(scoring.get("sharpness_weight", 0.25)),
        confidence_weight=float(scoring.get("confidence_weight", 0.20)),
        role_weight=float(scoring.get("role_weight", 0.15)),
        border_weight=float(scoring.get("border_weight", 0.05)),
        clipping_weight=float(scoring.get("clipping_weight", 0.05)),
        brightness_weight=float(scoring.get("brightness_weight", 0.05)),
        class_specific_minimums={
            str(class_name).strip().lower(): {
                "minimum_crop_width": int(dict(payload or {}).get("minimum_crop_width", evidence.get("minimum_crop_width", 100))),
                "minimum_crop_height": int(dict(payload or {}).get("minimum_crop_height", evidence.get("minimum_crop_height", 70))),
            }
            for class_name, payload in dict(evidence.get("class_specific_minimums", {}) or {}).items()
            if str(class_name).strip()
        },
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

    def __init__(self, config: EvidenceQualityConfig, image_size_policy: ImageSizePolicy | None = None) -> None:
        self.config = config
        self.image_size_policy = image_size_policy or normalize_image_size_policy(
            {},
            fallback_body_type={"minimum_crop_width": 256, "minimum_crop_height": 192},
            fallback_colour={"minimum_crop_width": 256, "minimum_crop_height": 192},
            detection={},
        )

    def score_item(self, item: EnrichmentEvidenceItem) -> EnrichmentEvidenceItem:
        reasons = list(item.rejection_reasons)
        crop_path = Path(str(item.vehicle_crop_path)) if item.vehicle_crop_path else None
        image = self._load_crop_image(crop_path)
        if image is None:
            reasons.append("missing_crop_image")
            item.rejection_reasons = self._dedupe_reasons(reasons)
            item.quality_score = 0.0
            item.readable_crop = False
            return item

        crop_height, crop_width = image.shape[:2]
        item.readable_crop = bool(crop_width > 0 and crop_height > 0)
        crop_area = int(crop_width * crop_height)
        sharpness = self.compute_sharpness(image)
        brightness = self.compute_brightness(image)
        border_penalty = max(0.0, min(1.0, float(item.border_penalty)))
        clipping_ratio = max(0.0, min(1.0, float(item.clipping_ratio)))
        aspect_ratio = float(crop_width / crop_height) if crop_height > 0 else 0.0
        vehicle_class = str(getattr(item, "vehicle_class", "unknown") or "unknown")
        class_minimum_width, class_minimum_height = self._class_minimums(vehicle_class)
        florence_thresholds = self.image_size_policy.florence.thresholds_for(vehicle_class)
        tier = self.image_size_policy.florence.resolution_tier(int(item.original_crop_width), int(item.original_crop_height), vehicle_class)
        skip_reason = self.image_size_policy.florence.eligibility_reason(int(item.original_crop_width), int(item.original_crop_height), vehicle_class)

        original_width = int(item.original_crop_width or crop_width)
        original_height = int(item.original_crop_height or crop_height)
        if original_width < class_minimum_width:
            reasons.append(self._class_reason("crop_width_below", vehicle_class, "minimum"))
        if original_height < class_minimum_height:
            reasons.append(self._class_reason("crop_height_below", vehicle_class, "minimum"))
        if sharpness < self.config.minimum_sharpness:
            reasons.append("sharpness_below_minimum")
        if brightness < self.config.minimum_brightness:
            reasons.append("brightness_below_minimum")
        if brightness > self.config.maximum_brightness:
            reasons.append("brightness_above_maximum")
        if clipping_ratio > self.config.maximum_edge_truncation_ratio:
            reasons.append("edge_truncation_above_maximum")
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
        if any(
            reason in reasons
            for reason in (
                "brightness_below_minimum",
                "brightness_above_maximum",
                "edge_truncation_above_maximum",
                "sharpness_below_minimum",
            )
        ):
            reasons.append("crop_rejected_quality")

        item.crop_width = crop_width
        item.crop_height = crop_height
        item.crop_area = crop_area
        item.original_crop_width = original_width
        item.original_crop_height = original_height
        item.resolution_tier = tier
        item.vehicle_class = vehicle_class
        item.class_minimum_width = class_minimum_width
        item.class_minimum_height = class_minimum_height
        item.florence_minimum_width = florence_thresholds.minimum_original_width
        item.florence_minimum_height = florence_thresholds.minimum_original_height
        item.evidence_eligible = original_width >= class_minimum_width and original_height >= class_minimum_height
        item.florence_eligible_for_body_type = tier != "below_minimum"
        item.florence_eligible_for_colour = item.readable_crop
        item.florence_body_type_skip_reason = skip_reason
        item.florence_colour_skip_reason = None if item.readable_crop else skip_reason
        item.sharpness_score = sharpness
        item.brightness_score = brightness
        item.quality_score = score
        item.edge_truncated = clipping_ratio > self.config.maximum_edge_truncation_ratio
        tier_bonus = 0.12 if tier == RESOLUTION_TIER_PREFERRED else 0.05 if tier == RESOLUTION_TIER_ACCEPTABLE else -0.25
        evidence_size_bonus = self._evidence_size_bonus(crop_width, crop_height, class_minimum_width, class_minimum_height)
        florence_size_bonus = self._florence_size_bonus(
            original_width,
            original_height,
            florence_thresholds.minimum_original_width,
            florence_thresholds.minimum_original_height,
        )
        item.ranking_score = float(
            score
            + tier_bonus
            + evidence_size_bonus
            + florence_size_bonus
            - (0.20 if "crop_rejected_quality" in reasons else 0.0)
        )
        if not item.readable_crop:
            item.colour_selection_tier = None
        elif tier == RESOLUTION_TIER_PREFERRED:
            item.colour_selection_tier = RESOLUTION_TIER_PREFERRED
        elif tier == RESOLUTION_TIER_ACCEPTABLE:
            item.colour_selection_tier = RESOLUTION_TIER_ACCEPTABLE
        else:
            item.colour_selection_tier = "low_resolution_fallback"
        item.rejection_reasons = self._dedupe_reasons(reasons)
        return item

    def _class_minimums(self, vehicle_class: str) -> tuple[int, int]:
        payload = self.config.class_specific_minimums.get(str(vehicle_class).strip().lower(), {})
        return (
            int(payload.get("minimum_crop_width", self.config.minimum_crop_width)),
            int(payload.get("minimum_crop_height", self.config.minimum_crop_height)),
        )

    @staticmethod
    def _class_reason(prefix: str, vehicle_class: str, suffix: str) -> str:
        normalized = " ".join(str(vehicle_class or "").strip().lower().replace("_", " ").replace("-", " ").split())
        if normalized and normalized != "unknown":
            return f"{prefix}_{normalized}_{suffix}"
        return f"{prefix}_{suffix}"

    @staticmethod
    def _evidence_size_bonus(width: int, height: int, minimum_width: int, minimum_height: int) -> float:
        width_score = min(1.0, max(0.0, width / max(1, minimum_width)))
        height_score = min(1.0, max(0.0, height / max(1, minimum_height)))
        return (width_score + height_score) * 0.08

    @staticmethod
    def _florence_size_bonus(width: int, height: int, minimum_width: int, minimum_height: int) -> float:
        if minimum_width <= 0 or minimum_height <= 0:
            return 0.0
        width_score = min(1.0, max(0.0, width / minimum_width))
        height_score = min(1.0, max(0.0, height / minimum_height))
        return (width_score + height_score) * 0.10

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
