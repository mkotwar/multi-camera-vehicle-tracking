from __future__ import annotations

from typing import Any

from ..schemas import ATTRIBUTE_STATUS_DISABLED, PlateColourResult


class PlateColourClassifier:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = dict(config)
        self.enabled = bool(self.config.get("colour_enabled", False))

    def classify(self, *args: Any, **kwargs: Any) -> PlateColourResult:
        return PlateColourResult(
            label=None,
            predictions=[],
            status=ATTRIBUTE_STATUS_DISABLED if not self.enabled else "not_run",
            source="plate.colour_classifier",
            reason="Plate colour classification is disabled in Step 2.",
        )
