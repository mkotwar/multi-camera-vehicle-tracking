from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np


@dataclass(slots=True, frozen=True)
class OCRMukulPreparedImage:
    image_bgr: np.ndarray
    original_width: int
    original_height: int
    preprocessed_width: int
    preprocessed_height: int
    manual_resize_applied: bool
    square_padding_applied: bool
    interpolation: str


def resize_proportionally_if_needed(image: np.ndarray, target_width: int = 200, target_height: int = 150) -> np.ndarray:
    # Adapted from:
    # D:\project\models\OCR_MUKUL\OCR_MUKUL\anpr_frog_speed.py
    # function: resize_proportionally_if_needed
    height, width = image.shape[:2]
    if height == 0 or width == 0:
        return image
    if width < target_width or height < target_height:
        scale_factor = max(target_width / width, target_height / height)
        return cv2.resize(image, (int(width * scale_factor), int(height * scale_factor)), interpolation=cv2.INTER_LINEAR)
    return image


class OCRMukulImagePreprocessor:
    """
    Adapted from:
    D:\\project\\models\\OCR_MUKUL\\OCR_MUKUL\\anpr_frog_speed.py
    functions: resize_proportionally_if_needed, run_florence_inference
    """

    def prepare(self, image_bgr: np.ndarray) -> OCRMukulPreparedImage:
        original_height, original_width = image_bgr.shape[:2]
        resized = resize_proportionally_if_needed(image_bgr.copy())
        preprocessed_height, preprocessed_width = resized.shape[:2]
        return OCRMukulPreparedImage(
            image_bgr=resized,
            original_width=int(original_width),
            original_height=int(original_height),
            preprocessed_width=int(preprocessed_width),
            preprocessed_height=int(preprocessed_height),
            manual_resize_applied=bool((preprocessed_width, preprocessed_height) != (original_width, original_height)),
            square_padding_applied=False,
            interpolation="INTER_LINEAR",
        )
