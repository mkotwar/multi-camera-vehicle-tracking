from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .vehicle_analytics import VehicleRecord, count_by_class, count_by_colour, find_vehicle_class_comparison_intervals
from .video_chat_plan import AnalyticsPlan, ComparisonDefinition, FilterCondition, FilterExpression, MetricOperand, and_filter


@dataclass(frozen=True, slots=True)
class CompiledAnalyticsQuery:
    plan: AnalyticsPlan
    repository_source: str = "repository"

    def to_dict(self) -> dict[str, Any]:
        return {
            "repository_source": self.repository_source,
            "plan": self.plan.to_dict(),
        }


def compile_analytics_plan(plan: AnalyticsPlan) -> CompiledAnalyticsQuery:
    return CompiledAnalyticsQuery(plan=plan)


def execute_analytics_plan(records: list[VehicleRecord], plan: AnalyticsPlan, *, previous_vehicle_ids: list[str] | None = None) -> dict[str, Any]:
    matched = _apply_plan_filters(records, plan, previous_vehicle_ids=previous_vehicle_ids)
    if plan.result_shape == "unsupported_capability":
        return {
            "status": "unsupported",
            "detail": "That information is not available in the current vehicle analytics data.",
            "vehicle_ids": [],
        }
    if plan.legacy_intent == "FIND_INTERVALS" and plan.comparison is not None:
        return _interval_result(matched, plan=plan)
    if plan.legacy_intent == "UNIQUE_CLASSES":
        counts = count_by_class(matched)
        return {"vehicle_classes_present": [label for label, count in counts.items() if count > 0], "vehicle_ids": [_result_vehicle_id(record, plan) for record in matched]}
    if plan.legacy_intent == "UNIQUE_COLOURS":
        counts = count_by_colour(matched)
        return {"colours_present": [label for label, count in counts.items() if count > 0], "vehicle_ids": [_result_vehicle_id(record, plan) for record in matched]}
    if plan.result_shape == "summary":
        if plan.group_by:
            return _grouped_summary(matched, plan=plan)
        return _summary(matched)
    if plan.result_shape == "plate_lookup":
        return {"total": len(matched), "vehicle_ids": [_result_vehicle_id(record, plan) for record in matched]}
    if plan.result_shape == "comparison" and plan.comparison is not None:
        return _comparison_result(records, plan, matched)
    if plan.group_by:
        return _grouped_result(matched, plan=plan)
    metric_value = _metric_value(matched, plan.metric)
    return {
        "total": len(matched),
        "metric_value": metric_value,
        "by_class": count_by_class(matched),
        "by_colour": count_by_colour(matched),
        "vehicle_ids": [_result_vehicle_id(record, plan) for record in matched],
    }


def _apply_plan_filters(records: list[VehicleRecord], plan: AnalyticsPlan, *, previous_vehicle_ids: list[str] | None = None) -> list[VehicleRecord]:
    previous_ids = {str(item) for item in list(previous_vehicle_ids or [])}
    return [
        record
        for record in records
        if (not previous_ids or record.vehicle_id in previous_ids or _scoped_vehicle_id(record) in previous_ids)
        and (not plan.selected_run_ids or not getattr(record, "run_id", None) or str(getattr(record, "run_id", None)) in set(plan.selected_run_ids))
        and (not plan.include_camera_ids or record.camera_id in set(plan.include_camera_ids))
        and (not plan.exclude_camera_ids or record.camera_id not in set(plan.exclude_camera_ids))
        and _matches_expression(record, plan.filters)
        and _matches_time(record, plan)
    ]


def _matches_expression(record: VehicleRecord, expression: FilterExpression | None) -> bool:
    if expression is None:
        return True
    if expression.kind == "condition":
        return _matches_condition(record, expression.condition)
    if expression.kind == "and":
        return all(_matches_expression(record, item) for item in expression.conditions)
    if expression.kind == "or":
        return any(_matches_expression(record, item) for item in expression.conditions)
    if expression.kind == "not":
        return not any(_matches_expression(record, item) for item in expression.conditions)
    return True


def _matches_condition(record: VehicleRecord, condition: FilterCondition | None) -> bool:
    if condition is None:
        return True
    value = condition.value
    field_value = _field_value(record, condition.field)
    operator = condition.operator
    if operator == "eq":
        return field_value == value
    if operator == "neq":
        return field_value != value
    if operator == "in":
        return field_value in set(value or [])
    if operator == "not_in":
        return field_value not in set(value or [])
    if operator == "starts_with":
        return isinstance(field_value, str) and isinstance(value, str) and field_value.startswith(value)
    if operator == "ends_with":
        return isinstance(field_value, str) and isinstance(value, str) and field_value.endswith(value)
    if operator == "contains":
        return isinstance(field_value, str) and isinstance(value, str) and value in field_value
    if operator == "exists":
        return field_value not in {None, "", False}
    if operator == "not_exists":
        return field_value in {None, "", False}
    if operator == "gt":
        return field_value is not None and value is not None and field_value > value
    if operator == "gte":
        return field_value is not None and value is not None and field_value >= value
    if operator == "lt":
        return field_value is not None and value is not None and field_value < value
    if operator == "lte":
        return field_value is not None and value is not None and field_value <= value
    if operator == "between":
        if not isinstance(value, (list, tuple)) or len(value) != 2 or field_value is None:
            return False
        return value[0] <= field_value <= value[1]
    return False


def _field_value(record: VehicleRecord, field: str) -> Any:
    if field == "class":
        return record.vehicle_class
    if field == "colour":
        return record.colour
    if field == "camera":
        return record.camera_id
    if field == "run":
        return str(record.run_id) if record.run_id else None
    if field == "plate_text":
        return record.plate_text
    if field == "plate_detected":
        return bool(record.plate_detected) if record.plate_detected is not None else False
    if field == "plate_readable":
        return bool(record.plate_text)
    if field == "plate_presence":
        if record.plate_text:
            return "readable"
        if record.plate_detected:
            return "detected"
        return "missing"
    if field == "start_time":
        return record.first_seen_seconds
    if field == "end_time":
        return record.last_seen_seconds
    if field == "vehicle_id":
        return record.vehicle_id
    return None


def _matches_time(record: VehicleRecord, plan: AnalyticsPlan) -> bool:
    if plan.time is None:
        return True
    if plan.time.start_seconds is not None and record.last_seen_seconds is not None and record.last_seen_seconds < plan.time.start_seconds:
        return False
    if plan.time.end_seconds is not None and record.first_seen_seconds is not None and record.first_seen_seconds > plan.time.end_seconds:
        return False
    return True


def _summary(records: list[VehicleRecord]) -> dict[str, Any]:
    first_seen = [record.first_seen_seconds for record in records if record.first_seen_seconds is not None]
    last_seen = [record.last_seen_seconds for record in records if record.last_seen_seconds is not None]
    return {
        "total_unique_vehicles": len(records),
        "vehicle_classes": count_by_class(records),
        "colours": count_by_colour(records),
        "first_seen_seconds": min(first_seen) if first_seen else None,
        "last_seen_seconds": max(last_seen) if last_seen else None,
        "vehicle_ids": [record.vehicle_id for record in records],
    }


def _grouped_summary(records: list[VehicleRecord], *, plan: AnalyticsPlan) -> dict[str, Any]:
    grouped = _group_records(records, plan.group_by[0])
    return {
        "total_unique_vehicles": len(records),
        "group_by": plan.group_by[0],
        "groups": {key: _summary(value) for key, value in grouped.items()},
        "vehicle_ids": [_result_vehicle_id(record, plan) for record in records],
    }


def _grouped_result(records: list[VehicleRecord], *, plan: AnalyticsPlan) -> dict[str, Any]:
    group_by = plan.group_by[0]
    grouped = _group_records(records, group_by)
    counts = {key: len(items) for key, items in grouped.items()}
    result_key = {
        "class": "by_class",
        "colour": "by_colour",
        "camera": "by_camera",
        "run": "by_run",
        "run_camera": "by_run_camera",
    }.get(group_by, "by_group")
    payload: dict[str, Any] = {
        "total": len(records),
        result_key: counts,
        "vehicle_ids": [_result_vehicle_id(record, plan) for record in records],
    }
    ranking = _ranking_result(counts, group_by=group_by, plan=plan)
    if ranking is not None:
        payload["ranking_result"] = ranking
    top = _top_count(counts)
    if top is not None:
        payload[f"top_{group_by if group_by != 'run_camera' else 'camera'}"] = top
    return payload


def _group_records(records: list[VehicleRecord], group_by: str) -> dict[str, list[VehicleRecord]]:
    grouped: dict[str, list[VehicleRecord]] = {}
    for record in records:
        key = _group_key(record, group_by)
        grouped.setdefault(key, []).append(record)
    return dict(sorted(grouped.items()))


def _group_key(record: VehicleRecord, group_by: str) -> str:
    if group_by == "class":
        return record.vehicle_class
    if group_by == "colour":
        return record.colour
    if group_by == "camera":
        return record.camera_id or "UNKNOWN"
    if group_by == "run":
        return str(record.run_id or "UNKNOWN")
    if group_by == "run_camera":
        return f"{str(record.run_id or 'UNKNOWN')} / {record.camera_id or 'UNKNOWN'}"
    return "UNKNOWN"


def _ranking_result(counts: dict[str, int], *, group_by: str, plan: AnalyticsPlan) -> dict[str, Any] | None:
    if not plan.order_by:
        return None
    direction = plan.order_by[0].direction
    reverse = direction == "desc"
    nonzero = {key: int(value) for key, value in counts.items() if int(value) > 0}
    entries = [
        {
            "label": key,
            "count": value,
            **({"run_id": key.split(" / ", 1)[0], "camera_id": key.split(" / ", 1)[1]} if group_by == "run_camera" and " / " in key else {}),
            **({"run_id": key} if group_by == "run" else {}),
            **({"camera_id": key} if group_by == "camera" else {}),
        }
        for key, value in sorted(nonzero.items(), key=lambda item: ((-item[1], item[0]) if reverse else (item[1], item[0])))
    ]
    if not entries:
        return {"group_by": group_by, "sort_by": f"count_{direction}", "entries": [], "winners": [], "is_tie": False}
    winners = [entry for entry in entries if entry["count"] == entries[0]["count"]]
    limit = max(plan.limit or 3, len(winners), 3)
    return {
        "group_by": group_by,
        "sort_by": f"count_{direction}",
        "entries": entries[:limit],
        "winners": winners,
        "is_tie": len(winners) > 1,
    }


def _comparison_result(all_records: list[VehicleRecord], plan: AnalyticsPlan, base_records: list[VehicleRecord]) -> dict[str, Any]:
    comparison = plan.comparison
    if comparison is None:
        return {"total": len(base_records), "vehicle_ids": [_result_vehicle_id(record, plan) for record in base_records]}
    left_records = _operand_records(all_records, plan, comparison.left)
    right_records = _operand_records(all_records, plan, comparison.right)
    left_total = len(left_records)
    right_total = len(right_records)
    operation = comparison.operation
    answer = "YES" if left_total > right_total else "NO"
    result: dict[str, Any] = {
        "left": _operand_label(comparison.left),
        "right": _operand_label(comparison.right),
        "left_total": left_total,
        "right_total": right_total,
        "difference": abs(left_total - right_total),
        "ratio": round(left_total / right_total, 6) if right_total else None,
        "percentage": round((left_total / right_total) * 100.0, 3) if right_total else None,
        "winner": _operand_label(comparison.left) if left_total > right_total else _operand_label(comparison.right) if right_total > left_total else None,
        "is_tie": left_total == right_total,
        "mode": operation,
        "metric_label": "vehicle_count",
        "answer": answer,
        "vehicle_ids": [_result_vehicle_id(record, plan) for record in _dedupe_records(left_records + right_records)],
    }
    return result


def _interval_result(records: list[VehicleRecord], *, plan: AnalyticsPlan) -> dict[str, Any]:
    comparison = plan.comparison
    if comparison is None:
        return {"left_class": "", "right_class": "", "operator": ">", "intervals": [], "window_seconds": 5.0, "vehicle_ids": []}
    left_class = _operand_label(comparison.left)
    right_class = _operand_label(comparison.right)
    operator = comparison.operation if comparison.operation in {">", "<", "="} else ">"
    window_seconds = float(plan.time.bucket_seconds) if plan.time is not None and plan.time.bucket_seconds is not None else 5.0
    result = find_vehicle_class_comparison_intervals(
        records,
        left_class=left_class,
        operator=operator,
        right_class=right_class,
        window_seconds=window_seconds,
    )
    result["vehicle_ids"] = [_result_vehicle_id(record, plan) for record in records]
    return result


def _operand_records(records: list[VehicleRecord], plan: AnalyticsPlan, operand: MetricOperand) -> list[VehicleRecord]:
    combined_filters = and_filter(plan.filters, operand.filters)
    scoped_plan = AnalyticsPlan(
        entity=plan.entity,
        filters=combined_filters,
        group_by=(),
        metric=plan.metric,
        comparison=None,
        order_by=(),
        limit=None,
        time=plan.time,
        selected_run_ids=plan.selected_run_ids,
        include_camera_ids=plan.include_camera_ids,
        exclude_camera_ids=plan.exclude_camera_ids,
        show_evidence=False,
        result_shape="scalar",
        legacy_intent=plan.legacy_intent,
        context_reference=plan.context_reference,
        context_resolution=plan.context_resolution,
        evidence_navigation=plan.evidence_navigation,
        provenance=plan.provenance,
    )
    return _apply_plan_filters(records, scoped_plan)


def _operand_label(operand: MetricOperand) -> str:
    if operand.filters is None or operand.filters.condition is None:
        return "ALL"
    field = operand.filters.condition.field
    value = operand.filters.condition.value
    if not isinstance(value, str):
        return str(value)
    if field == "colour":
        return value.title()
    if field == "camera":
        return value
    return value.upper()


def _metric_value(records: list[VehicleRecord], metric: Any) -> Any:
    if metric is None:
        return len(records)
    metric_type = getattr(metric, "type", None)
    if metric_type in {None, "vehicle_count", "count_distinct"}:
        return len(records)
    if metric_type == "difference":
        left = len(_metric_operand_records(records, metric.left))
        right = len(_metric_operand_records(records, metric.right))
        return left - right
    if metric_type == "ratio":
        left = len(_metric_operand_records(records, metric.left))
        right = len(_metric_operand_records(records, metric.right))
        return None if right == 0 else left / right
    if metric_type == "percentage":
        numerator = len(_metric_operand_records(records, metric.numerator))
        denominator = len(_metric_operand_records(records, metric.denominator))
        return None if denominator == 0 else (numerator / denominator) * 100.0
    return len(records)


def _metric_operand_records(records: list[VehicleRecord], operand: MetricOperand | None) -> list[VehicleRecord]:
    if operand is None or operand.filters is None:
        return records
    return [record for record in records if _matches_expression(record, operand.filters)]


def _top_count(counts: dict[str, int]) -> dict[str, Any] | None:
    nonzero = {key: int(value) for key, value in counts.items() if int(value) > 0}
    if not nonzero:
        return None
    label, count = max(nonzero.items(), key=lambda item: item[1])
    return {"label": label, "count": count}


def _dedupe_records(records: list[VehicleRecord]) -> list[VehicleRecord]:
    seen: set[str] = set()
    result: list[VehicleRecord] = []
    for record in records:
        if record.vehicle_id in seen:
            continue
        seen.add(record.vehicle_id)
        result.append(record)
    return result


def _result_vehicle_id(record: VehicleRecord, plan: AnalyticsPlan) -> str:
    if len(plan.selected_run_ids) > 1 and record.run_id:
        return _scoped_vehicle_id(record)
    return record.vehicle_id


def _scoped_vehicle_id(record: VehicleRecord) -> str:
    run_id = getattr(record, "run_id", None)
    return f"{run_id}::{record.vehicle_id}" if run_id else record.vehicle_id
