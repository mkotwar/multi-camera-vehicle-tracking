from __future__ import annotations

from typing import Any

from ..schemas import ATTRIBUTE_STATUS_DISABLED, PlateOCRResult


class PlateOCREngine:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = dict(config)
        self.enabled = bool(self.config.get("enabled", False))

    def recognize(self, *args: Any, **kwargs: Any) -> PlateOCRResult:
        return PlateOCRResult(
            text=None,
            predictions=[],
            status=ATTRIBUTE_STATUS_DISABLED if not self.enabled else "not_run",
            source="plate.ocr_engine",
            reason="Plate OCR is disabled in Step 2.",
        )
