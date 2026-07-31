from __future__ import annotations

from typing import Any

from ..schemas import ATTRIBUTE_STATUS_DISABLED, PlateQualityResult


class PlateQualityValidator:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = dict(config)
        self.enabled = bool(self.config.get("detection_enabled", False))

    def validate(self, *args: Any, **kwargs: Any) -> PlateQualityResult:
        return PlateQualityResult(
            acceptable=None,
            predictions=[],
            status=ATTRIBUTE_STATUS_DISABLED if not self.enabled else "not_run",
            source="plate.quality_validator",
            reason="Plate quality validation is disabled in Step 2.",
        )
