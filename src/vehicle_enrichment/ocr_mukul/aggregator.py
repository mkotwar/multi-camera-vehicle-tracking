from __future__ import annotations

from typing import Any

from ..schemas import ATTRIBUTE_STATUS_NOT_RUN, AttributePrediction


def aggregate_predictions(predictions: list[AttributePrediction], *, unknown_label: str, conflict_reason: str) -> tuple[str, str, float | None, float]:
    valid = [item for item in predictions if item.status == "completed" and item.label not in (None, unknown_label)]
    if not valid:
        return unknown_label, "no_valid_predictions", None, 0.0
    label_weights: dict[str, float] = {}
    for prediction in valid:
        weight = float(prediction.quality_weight or 0.0)
        label = str(prediction.label)
        label_weights[label] = label_weights.get(label, 0.0) + weight
    ordered = sorted(label_weights.items(), key=lambda item: (item[1], item[0]), reverse=True)
    top_label, top_weight = ordered[0]
    total_weight = float(sum(label_weights.values()))
    agreement_score = float(top_weight / total_weight) if total_weight > 0.0 else None
    if len(ordered) == 1:
        return top_label, "weighted_agreement", agreement_score, total_weight
    second_weight = ordered[1][1]
    if abs(top_weight - second_weight) <= max(0.05, 0.15 * max(top_weight, second_weight)):
        return unknown_label, conflict_reason, agreement_score, total_weight
    if agreement_score is not None and agreement_score >= 0.60:
        return top_label, "weighted_majority", agreement_score, total_weight
    return unknown_label, conflict_reason, agreement_score, total_weight
