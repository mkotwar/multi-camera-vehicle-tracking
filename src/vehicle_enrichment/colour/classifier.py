from __future__ import annotations

from typing import Any

from ..schemas import ATTRIBUTE_STATUS_DISABLED, VEHICLE_COLOUR_UNKNOWN, VehicleColourResult


class VehicleColourClassifier:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = dict(config)
        self.enabled = bool(self.config.get("enabled", False))

    def classify(self, *args: Any, **kwargs: Any) -> VehicleColourResult:
        return VehicleColourResult(
            label=VEHICLE_COLOUR_UNKNOWN,
            predictions=[],
            status=ATTRIBUTE_STATUS_DISABLED if not self.enabled else "not_run",
            source="colour.classifier",
            reason="Vehicle colour inference is disabled in Step 2.",
        )
