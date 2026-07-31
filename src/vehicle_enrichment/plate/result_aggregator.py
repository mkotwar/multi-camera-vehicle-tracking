from __future__ import annotations

from typing import Any


class PlateResultAggregator:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = dict(config)

    def aggregate(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return {
            "status": "disabled",
            "reason": "Plate aggregation is disabled in Step 2.",
        }
