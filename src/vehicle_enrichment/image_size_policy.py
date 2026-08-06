from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from PIL import Image, ImageOps


RESOLUTION_TIER_BELOW_MINIMUM = "below_minimum"
RESOLUTION_TIER_ACCEPTABLE = "acceptable"
RESOLUTION_TIER_PREFERRED = "preferred"


@dataclass(slots=True, frozen=True)
class FlorenceImageSizePolicy:
    minimum_original_width: int
    minimum_original_height: int
    preferred_original_width: int
    preferred_original_height: int
    eligibility_dimensions_source: str
    preserve_aspect_ratio: bool
    pad_to_square: bool
    square_padding_value: int
    reject_upscaled_eligibility: bool

    def resolution_tier(self, width: int, height: int) -> str:
        if width < self.minimum_original_width or height < self.minimum_original_height:
            return RESOLUTION_TIER_BELOW_MINIMUM
        if width >= self.preferred_original_width and height >= self.preferred_original_height:
            return RESOLUTION_TIER_PREFERRED
        return RESOLUTION_TIER_ACCEPTABLE

    def eligibility_reason(self, width: int, height: int) -> str | None:
        if width < self.minimum_original_width and height < self.minimum_original_height:
            return "crop_below_minimum_original_resolution"
        if width < self.minimum_original_width:
            return "crop_width_below_florence_minimum"
        if height < self.minimum_original_height:
            return "crop_height_below_florence_minimum"
        return None

    def is_eligible(self, width: int, height: int) -> bool:
        return self.resolution_tier(width, height) != RESOLUTION_TIER_BELOW_MINIMUM


@dataclass(slots=True, frozen=True)
class ImageSizePolicy:
    ingestion_preserve_source_resolution: bool
    detection_inference_size: int
    detection_preserve_aspect_ratio: bool
    detection_resize_mode: str
    evidence_preserve_original_crop: bool
    evidence_context_padding_ratio: float
    evidence_clamp_to_frame: bool
    evidence_store_original_dimensions: bool
    florence: FlorenceImageSizePolicy


def normalize_image_size_policy(raw_section: Any, *, fallback_body_type: dict[str, Any], fallback_colour: dict[str, Any], detection: dict[str, Any]) -> ImageSizePolicy:
    section = dict(raw_section or {})
    ingestion = dict(section.get("ingestion", {}) or {})
    detection_section = dict(section.get("detection", {}) or {})
    evidence_crop = dict(section.get("evidence_crop", {}) or {})
    florence = dict(section.get("florence", {}) or {})
    fallback_minimum_width = max(
        int(fallback_body_type.get("minimum_crop_width", 256)),
        int(fallback_colour.get("minimum_crop_width", 256)),
    )
    fallback_minimum_height = max(
        int(fallback_body_type.get("minimum_crop_height", 192)),
        int(fallback_colour.get("minimum_crop_height", 192)),
    )
    return ImageSizePolicy(
        ingestion_preserve_source_resolution=bool(ingestion.get("preserve_source_resolution", True)),
        detection_inference_size=int(detection_section.get("inference_size", detection.get("image_size", 1024))),
        detection_preserve_aspect_ratio=bool(detection_section.get("preserve_aspect_ratio", True)),
        detection_resize_mode=str(detection_section.get("resize_mode", "letterbox")).strip() or "letterbox",
        evidence_preserve_original_crop=bool(evidence_crop.get("preserve_original_crop", True)),
        evidence_context_padding_ratio=float(evidence_crop.get("context_padding_ratio", 0.08)),
        evidence_clamp_to_frame=bool(evidence_crop.get("clamp_to_frame", True)),
        evidence_store_original_dimensions=bool(evidence_crop.get("store_original_dimensions", True)),
        florence=FlorenceImageSizePolicy(
            minimum_original_width=int(florence.get("minimum_original_width", fallback_minimum_width)),
            minimum_original_height=int(florence.get("minimum_original_height", fallback_minimum_height)),
            preferred_original_width=int(florence.get("preferred_original_width", 320)),
            preferred_original_height=int(florence.get("preferred_original_height", 240)),
            eligibility_dimensions_source=str(florence.get("eligibility_dimensions_source", "original_crop")).strip() or "original_crop",
            preserve_aspect_ratio=bool(florence.get("preserve_aspect_ratio", True)),
            pad_to_square=bool(florence.get("pad_to_square", True)),
            square_padding_value=int(florence.get("square_padding_value", 114)),
            reject_upscaled_eligibility=bool(florence.get("reject_upscaled_eligibility", True)),
        ),
    )


def pad_to_square(image: Image.Image, fill: tuple[int, int, int] | int = (114, 114, 114)) -> Image.Image:
    image = image.convert("RGB")
    width, height = image.size
    if width == height:
        return image
    side = max(width, height)
    left = (side - width) // 2
    right = side - width - left
    top = (side - height) // 2
    bottom = side - height - top
    return ImageOps.expand(image, border=(left, top, right, bottom), fill=fill)
