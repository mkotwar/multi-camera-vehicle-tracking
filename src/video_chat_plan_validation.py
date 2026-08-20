from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .indian_plate_validator import VALID_STATE_CODES, validate_indian_plate
from .video_chat_plan import (
    FIELD_OPERATORS,
    FILTERABLE_FIELDS,
    GROUPABLE_FIELDS,
    SUPPORTED_COMPARISON_OPERATIONS,
    SUPPORTED_ENTITIES,
    SUPPORTED_METRICS,
    SUPPORTED_RESULT_SHAPES,
    AnalyticsPlan,
    ComparisonDefinition,
    FilterCondition,
    FilterExpression,
    MetricDefinition,
    MetricOperand,
    OrderBy,
    TimeRange,
)


@dataclass(frozen=True, slots=True)
class PlanValidationIssue:
    code: str
    detail: str

    def to_dict(self) -> dict[str, str]:
        return {"code": self.code, "detail": self.detail}


def normalize_analytics_plan_payload(payload: dict[str, Any], *, selected_run_ids: list[str] | None = None) -> tuple[dict[str, Any], list[str]]:
    normalized: dict[str, Any] = {
        "entity": _normalize_entity(payload.get("entity")),
        "filters": _normalize_filter_expression(payload.get("filters")),
        "group_by": _normalize_group_by(payload.get("group_by")),
        "metric": _normalize_metric(payload.get("metric")),
        "comparison": _normalize_comparison(payload.get("comparison")),
        "order_by": _normalize_order_by(payload.get("order_by")),
        "limit": _normalize_limit(payload.get("limit")),
        "time": _normalize_time(payload.get("time")),
        "include_camera_ids": _normalize_camera_ids(payload.get("include_camera_ids")),
        "exclude_camera_ids": _normalize_camera_ids(payload.get("exclude_camera_ids")),
        "show_evidence": bool(payload.get("show_evidence")),
        "result_shape": _normalize_result_shape(payload.get("result_shape")),
        "context_reference": _normalize_context_reference(payload.get("context_reference")),
        "context_resolution": _normalize_context_resolution(payload.get("context_resolution")),
    }
    actions: list[str] = []
    if normalized["group_by"] == ["camera"] and len(list(selected_run_ids or [])) > 1:
        normalized["group_by"] = ["run_camera"]
        actions.append("group_by camera -> run_camera for multi-run scope")
    return normalized, actions


def validate_analytics_plan_schema(payload: dict[str, Any]) -> list[PlanValidationIssue]:
    issues: list[PlanValidationIssue] = []
    if not isinstance(payload, dict):
        return [PlanValidationIssue("payload_not_object", "AnalyticsPlan payload must be a JSON object.")]
    entity = payload.get("entity")
    if entity not in SUPPORTED_ENTITIES:
        issues.append(PlanValidationIssue("invalid_entity", f"Unsupported entity: {entity!r}"))
    issues.extend(_validate_filter_schema(payload.get("filters"), path="filters"))
    group_by = payload.get("group_by")
    if not isinstance(group_by, list):
        issues.append(PlanValidationIssue("invalid_group_by_type", "group_by must be a list."))
    else:
        for value in group_by:
            if value not in GROUPABLE_FIELDS:
                issues.append(PlanValidationIssue("invalid_group_by", f"Unsupported group_by value: {value!r}"))
    issues.extend(_validate_metric_schema(payload.get("metric"), path="metric"))
    comparison = payload.get("comparison")
    if comparison is not None:
        issues.extend(_validate_comparison_schema(comparison, path="comparison"))
    order_by = payload.get("order_by")
    if not isinstance(order_by, list):
        issues.append(PlanValidationIssue("invalid_order_by_type", "order_by must be a list."))
    else:
        for index, item in enumerate(order_by):
            if not isinstance(item, dict):
                issues.append(PlanValidationIssue("invalid_order_by_item", f"order_by[{index}] must be an object."))
                continue
            if item.get("field") not in {"metric", *GROUPABLE_FIELDS}:
                issues.append(PlanValidationIssue("invalid_order_by_field", f"Unsupported order_by field: {item.get('field')!r}"))
            if item.get("direction") not in {"asc", "desc"}:
                issues.append(PlanValidationIssue("invalid_order_by_direction", f"Unsupported order direction: {item.get('direction')!r}"))
    limit = payload.get("limit")
    if limit is not None and (not isinstance(limit, int) or limit < 1):
        issues.append(PlanValidationIssue("invalid_limit", "limit must be an integer >= 1 or null."))
    issues.extend(_validate_time_schema(payload.get("time")))
    for key in ("include_camera_ids", "exclude_camera_ids"):
        value = payload.get(key)
        if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
            issues.append(PlanValidationIssue(f"invalid_{key}", f"{key} must be a list of strings."))
    if not isinstance(payload.get("show_evidence"), bool):
        issues.append(PlanValidationIssue("invalid_show_evidence", "show_evidence must be a boolean."))
    if payload.get("result_shape") not in SUPPORTED_RESULT_SHAPES:
        issues.append(PlanValidationIssue("invalid_result_shape", f"Unsupported result_shape: {payload.get('result_shape')!r}"))
    context_reference = payload.get("context_reference")
    if context_reference not in {None, "previous_results"}:
        issues.append(PlanValidationIssue("invalid_context_reference", f"Unsupported context_reference: {context_reference!r}"))
    context_resolution = payload.get("context_resolution")
    if context_resolution not in {None, "single", "multiple"}:
        issues.append(PlanValidationIssue("invalid_context_resolution", f"Unsupported context_resolution: {context_resolution!r}"))
    return issues


def validate_analytics_plan_semantics(plan: AnalyticsPlan) -> list[PlanValidationIssue]:
    issues: list[PlanValidationIssue] = []
    if plan.entity != "vehicle":
        issues.append(PlanValidationIssue("unsupported_entity", f"Entity {plan.entity!r} is not executable in this phase."))
    if plan.result_shape == "ranking":
        if not plan.group_by:
            issues.append(PlanValidationIssue("ranking_requires_group_by", "Ranking queries require group_by."))
        if not plan.order_by:
            issues.append(PlanValidationIssue("ranking_requires_order", "Ranking queries require order_by."))
        if plan.metric is None:
            issues.append(PlanValidationIssue("ranking_requires_metric", "Ranking queries require a metric."))
    if plan.result_shape == "grouped" and not plan.group_by:
        issues.append(PlanValidationIssue("grouped_requires_group_by", "Grouped results require group_by."))
    if plan.result_shape == "comparison" and plan.comparison is None:
        issues.append(PlanValidationIssue("comparison_requires_operands", "Comparison results require left and right operands."))
    if plan.metric is not None:
        issues.extend(_validate_metric_semantics(plan.metric))
    if plan.comparison is not None and plan.comparison.operation not in SUPPORTED_COMPARISON_OPERATIONS:
        issues.append(PlanValidationIssue("invalid_comparison_operation", f"Unsupported comparison operation: {plan.comparison.operation!r}"))
    if plan.time is not None and plan.time.start_seconds is not None and plan.time.end_seconds is not None and plan.time.start_seconds > plan.time.end_seconds:
        issues.append(PlanValidationIssue("invalid_time_range", "time.start_seconds must be <= time.end_seconds."))
    if len(plan.selected_run_ids) > 1 and "camera" in plan.group_by:
        issues.append(PlanValidationIssue("unsafe_multi_run_camera_grouping", "Use run_camera when grouping cameras across multiple runs."))
    exact_plate_values = _collect_plate_filter_values(plan.filters, operator="eq")
    for value in exact_plate_values:
        validation = validate_indian_plate(str(value))
        if not validation.valid:
            issues.append(PlanValidationIssue("invalid_exact_plate", f"Exact plate filter requires a valid canonical plate: {value!r}"))
    prefix_values = _collect_plate_filter_values(plan.filters, operator="starts_with")
    for value in prefix_values:
        if _validate_prefix_fragment(str(value)) is None:
            issues.append(PlanValidationIssue("invalid_plate_prefix", f"Invalid plate prefix fragment: {value!r}"))
    suffix_values = _collect_plate_filter_values(plan.filters, operator="ends_with")
    for value in suffix_values:
        if not str(value).isdigit():
            issues.append(PlanValidationIssue("invalid_plate_suffix", f"Invalid plate suffix fragment: {value!r}"))
    contains_values = _collect_plate_filter_values(plan.filters, operator="contains")
    for value in contains_values:
        if _validate_contains_fragment(str(value)) is None:
            issues.append(PlanValidationIssue("invalid_plate_contains", f"Invalid plate contains fragment: {value!r}"))
    return issues


def analytics_plan_from_payload(payload: dict[str, Any], *, provenance: str) -> AnalyticsPlan:
    return AnalyticsPlan(
        entity=str(payload.get("entity") or "vehicle"),
        filters=_expression_from_payload(payload.get("filters")),
        group_by=tuple(str(item) for item in list(payload.get("group_by") or [])),
        metric=_metric_from_payload(payload.get("metric")),
        comparison=_comparison_from_payload(payload.get("comparison")),
        order_by=tuple(OrderBy(field=str(item["field"]), direction=str(item["direction"])) for item in list(payload.get("order_by") or [])),
        limit=payload.get("limit"),
        time=_time_from_payload(payload.get("time")),
        include_camera_ids=tuple(str(item) for item in list(payload.get("include_camera_ids") or [])),
        exclude_camera_ids=tuple(str(item) for item in list(payload.get("exclude_camera_ids") or [])),
        show_evidence=bool(payload.get("show_evidence")),
        result_shape=str(payload.get("result_shape") or "scalar"),
        legacy_intent=None,
        context_reference=payload.get("context_reference"),
        context_resolution=payload.get("context_resolution"),
        provenance=provenance,
    )


def _validate_filter_schema(payload: Any, *, path: str) -> list[PlanValidationIssue]:
    if payload is None:
        return []
    if not isinstance(payload, dict):
        return [PlanValidationIssue("invalid_filter_expression", f"{path} must be an object or null.")]
    issues: list[PlanValidationIssue] = []
    kind = payload.get("kind")
    if kind not in {"condition", "and", "or", "not"}:
        issues.append(PlanValidationIssue("invalid_filter_kind", f"{path}.kind {kind!r} is not supported."))
        return issues
    if kind == "condition":
        condition = payload.get("condition")
        if not isinstance(condition, dict):
            issues.append(PlanValidationIssue("missing_condition", f"{path}.condition must be present for condition filters."))
            return issues
        field = condition.get("field")
        operator = condition.get("operator")
        if field not in FILTERABLE_FIELDS:
            issues.append(PlanValidationIssue("invalid_filter_field", f"Unsupported filter field: {field!r}"))
        elif operator not in FIELD_OPERATORS.get(str(field), ()):
            issues.append(PlanValidationIssue("invalid_filter_operator", f"Operator {operator!r} is not allowed for field {field!r}."))
        return issues
    conditions = payload.get("conditions")
    if not isinstance(conditions, list) or not conditions:
        issues.append(PlanValidationIssue("missing_nested_conditions", f"{path}.conditions must be a non-empty list."))
        return issues
    for index, item in enumerate(conditions):
        issues.extend(_validate_filter_schema(item, path=f"{path}.conditions[{index}]"))
    return issues


def _validate_metric_schema(payload: Any, *, path: str) -> list[PlanValidationIssue]:
    if payload is None:
        return []
    if not isinstance(payload, dict):
        return [PlanValidationIssue("invalid_metric", f"{path} must be an object or null.")]
    issues: list[PlanValidationIssue] = []
    metric_type = payload.get("type")
    if metric_type not in SUPPORTED_METRICS:
        issues.append(PlanValidationIssue("invalid_metric_type", f"Unsupported metric type: {metric_type!r}"))
    for key in ("operand", "numerator", "denominator", "left", "right"):
        value = payload.get(key)
        if value is not None:
            issues.extend(_validate_metric_operand_schema(value, path=f"{path}.{key}"))
    return issues


def _validate_metric_operand_schema(payload: Any, *, path: str) -> list[PlanValidationIssue]:
    if not isinstance(payload, dict):
        return [PlanValidationIssue("invalid_metric_operand", f"{path} must be an object.")]
    issues: list[PlanValidationIssue] = []
    if payload.get("metric") not in SUPPORTED_METRICS:
        issues.append(PlanValidationIssue("invalid_metric_operand_type", f"Unsupported metric operand: {payload.get('metric')!r}"))
    issues.extend(_validate_filter_schema(payload.get("filters"), path=f"{path}.filters"))
    return issues


def _validate_comparison_schema(payload: Any, *, path: str) -> list[PlanValidationIssue]:
    if not isinstance(payload, dict):
        return [PlanValidationIssue("invalid_comparison", f"{path} must be an object.")]
    issues: list[PlanValidationIssue] = []
    if payload.get("operation") not in SUPPORTED_COMPARISON_OPERATIONS:
        issues.append(PlanValidationIssue("invalid_comparison_operation", f"Unsupported comparison operation: {payload.get('operation')!r}"))
    issues.extend(_validate_metric_operand_schema(payload.get("left"), path=f"{path}.left"))
    issues.extend(_validate_metric_operand_schema(payload.get("right"), path=f"{path}.right"))
    return issues


def _validate_time_schema(payload: Any) -> list[PlanValidationIssue]:
    if payload is None:
        return []
    if not isinstance(payload, dict):
        return [PlanValidationIssue("invalid_time", "time must be an object or null.")]
    issues: list[PlanValidationIssue] = []
    for key in ("start_seconds", "end_seconds", "bucket_seconds"):
        value = payload.get(key)
        if value is not None and not isinstance(value, (int, float)):
            issues.append(PlanValidationIssue("invalid_time_value", f"time.{key} must be numeric or null."))
    return issues


def _validate_metric_semantics(metric: MetricDefinition) -> list[PlanValidationIssue]:
    issues: list[PlanValidationIssue] = []
    if metric.type in {"vehicle_count", "count_distinct"} and metric.operand is None:
        issues.append(PlanValidationIssue("metric_operand_required", f"{metric.type} requires operand."))
    if metric.type == "percentage" and (metric.numerator is None or metric.denominator is None):
        issues.append(PlanValidationIssue("percentage_requires_denominator", "percentage requires numerator and denominator."))
    if metric.type == "ratio" and (metric.left is None or metric.right is None):
        issues.append(PlanValidationIssue("ratio_requires_operands", "ratio requires left and right operands."))
    if metric.type == "difference" and (metric.left is None or metric.right is None):
        issues.append(PlanValidationIssue("difference_requires_operands", "difference requires left and right operands."))
    return issues


def _normalize_entity(value: Any) -> str:
    return str(value or "vehicle").strip().lower()


def _normalize_group_by(value: Any) -> list[str]:
    normalized: list[str] = []
    aliases = {
        "vehicle_class": "class",
        "vehicle_type": "class",
        "type": "class",
        "category": "class",
        "vehicle_colour": "colour",
        "vehicle_color": "colour",
        "color": "colour",
        "camera_id": "camera",
        "run_id": "run",
        "run+camera": "run_camera",
    }
    for item in list(value or []):
        label = str(item).strip().lower().replace(" ", "_")
        normalized.append(aliases.get(label, label))
    return normalized


def _normalize_filter_expression(value: Any) -> dict[str, Any] | None:
    if value is None or not isinstance(value, dict):
        return None if value is None else value
    kind = str(value.get("kind") or "").strip().lower()
    if kind == "condition":
        condition = dict(value.get("condition") or {})
        field = str(condition.get("field") or "").strip().lower().replace(" ", "_")
        operator_aliases = {"prefix": "starts_with", "suffix": "ends_with", "equals": "eq", "not_equals": "neq", "exists_true": "exists"}
        value_field_aliases = {
            "vehicle_class": "class",
            "vehicle_type": "class",
            "type": "class",
            "category": "class",
            "vehicle_colour": "colour",
            "vehicle_color": "colour",
            "color": "colour",
            "camera_id": "camera",
            "run_id": "run",
        }
        operator = str(condition.get("operator") or "").strip().lower()
        normalized_value = _normalize_filter_value(value_field_aliases.get(field, field), condition.get("value"))
        return {
            "kind": "condition",
            "condition": {
                "field": value_field_aliases.get(field, field),
                "operator": operator_aliases.get(operator, operator),
                "value": normalized_value,
            },
        }
    return {
        "kind": kind,
        "conditions": [_normalize_filter_expression(item) for item in list(value.get("conditions") or [])],
    }


def _normalize_filter_value(field: str, value: Any) -> Any:
    if field == "class":
        return _normalize_token(value)
    if field == "colour":
        colour = _normalize_token(value)
        return "GREY" if colour == "GRAY" else colour
    if field == "plate_text":
        return _normalize_token(value)
    if field in {"camera", "run"} and isinstance(value, list):
        return [str(item).strip().upper() for item in value]
    if field in {"camera", "run"} and value is not None:
        return str(value).strip().upper()
    if isinstance(value, list):
        return [_normalize_filter_value(field, item) for item in value]
    return value


def _normalize_metric(value: Any) -> dict[str, Any] | None:
    if value is None or not isinstance(value, dict):
        return None if value is None else value
    normalized = {"type": str(value.get("type") or "").strip().lower()}
    for key in ("operand", "numerator", "denominator", "left", "right"):
        operand = value.get(key)
        if operand is not None:
            normalized[key] = _normalize_metric_operand(operand)
    return normalized


def _normalize_metric_operand(value: Any) -> dict[str, Any]:
    payload = dict(value or {})
    return {
        "metric": str(payload.get("metric") or "vehicle_count").strip().lower(),
        "filters": _normalize_filter_expression(payload.get("filters")),
    }


def _normalize_comparison(value: Any) -> dict[str, Any] | None:
    if value is None or not isinstance(value, dict):
        return None if value is None else value
    return {
        "operation": str(value.get("operation") or "").strip().lower(),
        "left": _normalize_metric_operand(value.get("left")),
        "right": _normalize_metric_operand(value.get("right")),
    }


def _normalize_order_by(value: Any) -> list[dict[str, str]]:
    aliases = {"count": "metric", "vehicle_count": "metric"}
    normalized: list[dict[str, str]] = []
    for item in list(value or []):
        if not isinstance(item, dict):
            normalized.append(item)
            continue
        field = str(item.get("field") or "").strip().lower().replace(" ", "_")
        direction = str(item.get("direction") or "").strip().lower()
        normalized.append({"field": aliases.get(field, field), "direction": direction})
    return normalized


def _normalize_limit(value: Any) -> Any:
    if value in {None, ""}:
        return None
    if isinstance(value, bool):
        return value
    try:
        return int(value)
    except (TypeError, ValueError):
        return value


def _normalize_time(value: Any) -> dict[str, Any] | None:
    if value is None or not isinstance(value, dict):
        return None if value is None else value
    return {
        "start_seconds": _normalize_float(value.get("start_seconds")),
        "end_seconds": _normalize_float(value.get("end_seconds")),
        "bucket_seconds": _normalize_float(value.get("bucket_seconds")),
    }


def _normalize_camera_ids(value: Any) -> list[str]:
    return [str(item).strip().upper() for item in list(value or []) if str(item).strip()]


def _normalize_result_shape(value: Any) -> str:
    return str(value or "scalar").strip().lower()


def _normalize_context_reference(value: Any) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip().lower()
    return "previous_results" if normalized in {"previous_result", "previous_results"} else normalized


def _normalize_context_resolution(value: Any) -> str | None:
    if value is None:
        return None
    return str(value).strip().lower()


def _expression_from_payload(payload: Any) -> FilterExpression | None:
    if payload is None:
        return None
    if not isinstance(payload, dict):
        return None
    kind = str(payload.get("kind") or "")
    if kind == "condition":
        condition = dict(payload.get("condition") or {})
        return FilterExpression(
            kind="condition",
            condition=FilterCondition(
                field=str(condition.get("field") or ""),
                operator=str(condition.get("operator") or ""),
                value=condition.get("value"),
            ),
        )
    return FilterExpression(kind=kind, conditions=tuple(_expression_from_payload(item) for item in list(payload.get("conditions") or []) if _expression_from_payload(item) is not None))


def _metric_from_payload(payload: Any) -> MetricDefinition | None:
    if payload is None or not isinstance(payload, dict):
        return None
    return MetricDefinition(
        type=str(payload.get("type") or ""),
        operand=_metric_operand_from_payload(payload.get("operand")),
        numerator=_metric_operand_from_payload(payload.get("numerator")),
        denominator=_metric_operand_from_payload(payload.get("denominator")),
        left=_metric_operand_from_payload(payload.get("left")),
        right=_metric_operand_from_payload(payload.get("right")),
    )


def _metric_operand_from_payload(payload: Any) -> MetricOperand | None:
    if payload is None or not isinstance(payload, dict):
        return None
    return MetricOperand(metric=str(payload.get("metric") or "vehicle_count"), filters=_expression_from_payload(payload.get("filters")))


def _comparison_from_payload(payload: Any) -> ComparisonDefinition | None:
    if payload is None or not isinstance(payload, dict):
        return None
    left = _metric_operand_from_payload(payload.get("left"))
    right = _metric_operand_from_payload(payload.get("right"))
    if left is None or right is None:
        return None
    return ComparisonDefinition(operation=str(payload.get("operation") or ""), left=left, right=right)


def _time_from_payload(payload: Any) -> TimeRange | None:
    if payload is None or not isinstance(payload, dict):
        return None
    return TimeRange(
        start_seconds=_normalize_float(payload.get("start_seconds")),
        end_seconds=_normalize_float(payload.get("end_seconds")),
        bucket_seconds=_normalize_float(payload.get("bucket_seconds")),
    )


def _normalize_float(value: Any) -> Any:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return value


def _normalize_token(value: Any) -> Any:
    if isinstance(value, list):
        return [_normalize_token(item) for item in value]
    if value is None:
        return None
    return str(value).strip().upper().replace(" ", "_")


def _collect_plate_filter_values(expression: FilterExpression | None, *, operator: str) -> list[str]:
    if expression is None:
        return []
    if expression.kind == "condition" and expression.condition is not None:
        if expression.condition.field == "plate_text" and expression.condition.operator == operator:
            return [str(expression.condition.value)]
        return []
    values: list[str] = []
    for item in expression.conditions:
        values.extend(_collect_plate_filter_values(item, operator=operator))
    return values


def _validate_prefix_fragment(value: str) -> str | None:
    token = value.strip().upper()
    if len(token) == 2 and token in VALID_STATE_CODES:
        return token
    if len(token) >= 3:
        return token
    return None


def _validate_contains_fragment(value: str) -> str | None:
    token = value.strip().upper()
    if len(token) < 2 or len(token) > 6:
        return None
    return token if any(character.isdigit() for character in token) else None
