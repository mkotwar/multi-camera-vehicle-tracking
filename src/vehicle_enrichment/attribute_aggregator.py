from __future__ import annotations

from typing import Iterable

from .schemas import AttributePrediction


class AttributeAggregator:
    def choose_label(
        self,
        predictions: Iterable[AttributePrediction],
        *,
        default_label: str | None,
    ) -> tuple[str | None, list[AttributePrediction]]:
        ordered = [item for item in predictions]
        valid = [item for item in ordered if item.label not in (None, "", "UNKNOWN") and item.status == "ready"]
        if not valid:
            return default_label, ordered
        best = max(
            valid,
            key=lambda item: (
                item.confidence if item.confidence is not None else -1.0,
                item.quality_weight if item.quality_weight is not None else -1.0,
            ),
        )
        return best.label, ordered
