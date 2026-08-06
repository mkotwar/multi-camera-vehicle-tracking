from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(slots=True, frozen=True)
class LegacyCropPreparationResult:
    image_bgr: np.ndarray
    image_rgb: np.ndarray
    original_width: int
    original_height: int
    preprocessed_width: int
    preprocessed_height: int
    manual_resize_applied: bool
    manual_square_padding_applied: bool
    source: str


class OldTdCase2CropPreprocessor:
    """
    Reproduced from:
    - D:\old_files\reference_pro\Final_vedio_Ai_system\tests\td_case2\step_04a_florence_model_audit.py
      function: run_florence_generation
    - D:\old_files\reference_pro\Final_vedio_Ai_system\tests\td_case2\streaming_tracking_pipeline\florence_inference.py
      function: _generate
    """

    def prepare_from_crop_path(self, crop_path: str | Path) -> LegacyCropPreparationResult:
        path = Path(crop_path)
        image_bgr = cv2.imread(str(path))
        if image_bgr is None or image_bgr.size == 0:
            raise FileNotFoundError(f"Legacy Florence input crop could not be read: {path}")
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        height, width = image_rgb.shape[:2]
        return LegacyCropPreparationResult(
            image_bgr=image_bgr,
            image_rgb=image_rgb,
            original_width=width,
            original_height=height,
            preprocessed_width=width,
            preprocessed_height=height,
            manual_resize_applied=False,
            manual_square_padding_applied=False,
            source="selected_vehicle_crop",
        )
