from __future__ import annotations

from typing import Any

from ..schemas import ATTRIBUTE_STATUS_DISABLED, PlateDetectionResult


class PlateDetector:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = dict(config)
        self.enabled = bool(self.config.get("detection_enabled", False))

    def detect(self, *args: Any, **kwargs: Any) -> PlateDetectionResult:
        return PlateDetectionResult(
            detected=False,
            predictions=[],
            status=ATTRIBUTE_STATUS_DISABLED if not self.enabled else "not_run",
            source="plate.detector",
            reason="Plate detection is disabled in Step 2.",
        )
