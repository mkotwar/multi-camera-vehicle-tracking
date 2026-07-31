from __future__ import annotations

from typing import Any

from ..schemas import ATTRIBUTE_STATUS_DISABLED, VehicleMakeModelResult


class VehicleMakeModelClassifier:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = dict(config)
        self.enabled = bool(self.config.get("enabled", False))

    def classify(self, *args: Any, **kwargs: Any) -> VehicleMakeModelResult:
        return VehicleMakeModelResult(
            make=None,
            model=None,
            predictions=[],
            status=ATTRIBUTE_STATUS_DISABLED if not self.enabled else "not_run",
            source="make_model.classifier",
            reason="Vehicle make/model inference is disabled in Step 2.",
        )
