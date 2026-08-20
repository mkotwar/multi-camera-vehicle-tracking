from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


SUPPORTED_ENTITIES = frozenset({"vehicle", "camera", "run"})
ENTITY_VEHICLE = "vehicle"
GROUPABLE_FIELDS = frozenset({"class", "colour", "camera", "run", "run_camera", "plate_presence", "time_bucket"})
FILTERABLE_FIELDS = frozenset(
    {
        "class",
        "colour",
        "camera",
        "run",
        "plate_text",
        "plate_detected",
        "plate_readable",
        "plate_presence",
        "start_time",
        "end_time",
        "vehicle_id",
    }
)
FIELD_OPERATORS: dict[str, tuple[str, ...]] = {
    "class": ("eq", "neq", "in", "not_in"),
    "colour": ("eq", "neq", "in", "not_in"),
    "camera": ("eq", "neq", "in", "not_in"),
    "run": ("eq", "neq", "in", "not_in"),
    "plate_text": ("eq", "starts_with", "ends_with", "contains", "exists", "not_exists"),
    "plate_detected": ("eq",),
    "plate_readable": ("eq",),
    "plate_presence": ("eq", "neq"),
    "start_time": ("gt", "gte", "lt", "lte", "between"),
    "end_time": ("gt", "gte", "lt", "lte", "between"),
    "vehicle_id": ("eq", "in"),
}
SUPPORTED_METRICS = frozenset({"vehicle_count", "count_distinct", "difference", "ratio", "percentage"})
SUPPORTED_COMPARISON_OPERATIONS = frozenset({"winner", "difference", "ratio", "percentage"})
SUPPORTED_RESULT_SHAPES = frozenset(
    {
        "scalar",
        "list",
        "grouped",
        "ranking",
        "comparison",
        "summary",
        "plate_lookup",
        "unsupported_capability",
    }
)


@dataclass(frozen=True, slots=True)
class CapabilityCatalogue:
    entity: str = ENTITY_VEHICLE
    supported_entities: tuple[str, ...] = tuple(sorted(SUPPORTED_ENTITIES))
    filterable_fields: tuple[str, ...] = tuple(sorted(FILTERABLE_FIELDS))
    groupable_fields: tuple[str, ...] = tuple(sorted(GROUPABLE_FIELDS))
    supported_metrics: tuple[str, ...] = tuple(sorted(SUPPORTED_METRICS))
    supported_comparison_operations: tuple[str, ...] = tuple(sorted(SUPPORTED_COMPARISON_OPERATIONS))
    supported_result_shapes: tuple[str, ...] = tuple(sorted(SUPPORTED_RESULT_SHAPES))
    operators_by_field: dict[str, tuple[str, ...]] = field(default_factory=lambda: dict(FIELD_OPERATORS))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


VIDEO_CHAT_CAPABILITY_CATALOGUE = CapabilityCatalogue()


@dataclass(frozen=True, slots=True)
class FilterCondition:
    field: str
    operator: str
    value: Any = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class FilterExpression:
    kind: Literal["condition", "and", "or", "not"]
    condition: FilterCondition | None = None
    conditions: tuple["FilterExpression", ...] = ()

    def to_dict(self) -> dict[str, Any]:
        payload = {"kind": self.kind}
        if self.condition is not None:
            payload["condition"] = self.condition.to_dict()
        if self.conditions:
            payload["conditions"] = [item.to_dict() for item in self.conditions]
        return payload


@dataclass(frozen=True, slots=True)
class MetricOperand:
    metric: str = "vehicle_count"
    filters: FilterExpression | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = {"metric": self.metric}
        if self.filters is not None:
            payload["filters"] = self.filters.to_dict()
        return payload


@dataclass(frozen=True, slots=True)
class MetricDefinition:
    type: str
    operand: MetricOperand | None = None
    numerator: MetricOperand | None = None
    denominator: MetricOperand | None = None
    left: MetricOperand | None = None
    right: MetricOperand | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = {"type": self.type}
        for key in ("operand", "numerator", "denominator", "left", "right"):
            value = getattr(self, key)
            if value is not None:
                payload[key] = value.to_dict()
        return payload


@dataclass(frozen=True, slots=True)
class ComparisonDefinition:
    operation: str
    left: MetricOperand
    right: MetricOperand

    def to_dict(self) -> dict[str, Any]:
        return {
            "operation": self.operation,
            "left": self.left.to_dict(),
            "right": self.right.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class OrderBy:
    field: str
    direction: Literal["asc", "desc"]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class TimeRange:
    start_seconds: float | None = None
    end_seconds: float | None = None
    bucket_seconds: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class AnalyticsPlan:
    entity: str = ENTITY_VEHICLE
    filters: FilterExpression | None = None
    group_by: tuple[str, ...] = ()
    metric: MetricDefinition | None = None
    comparison: ComparisonDefinition | None = None
    order_by: tuple[OrderBy, ...] = ()
    limit: int | None = None
    time: TimeRange | None = None
    selected_run_ids: tuple[str, ...] = ()
    include_camera_ids: tuple[str, ...] = ()
    exclude_camera_ids: tuple[str, ...] = ()
    show_evidence: bool = False
    result_shape: str = "scalar"
    legacy_intent: str | None = None
    context_reference: str | None = None
    context_resolution: str | None = None
    evidence_navigation: str | None = None
    provenance: str = "rule_based_fallback"

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "entity": self.entity,
            "group_by": list(self.group_by),
            "order_by": [item.to_dict() for item in self.order_by],
            "limit": self.limit,
            "selected_run_ids": list(self.selected_run_ids),
            "include_camera_ids": list(self.include_camera_ids),
            "exclude_camera_ids": list(self.exclude_camera_ids),
            "show_evidence": self.show_evidence,
            "result_shape": self.result_shape,
            "legacy_intent": self.legacy_intent,
            "context_reference": self.context_reference,
            "context_resolution": self.context_resolution,
            "evidence_navigation": self.evidence_navigation,
            "provenance": self.provenance,
        }
        if self.filters is not None:
            payload["filters"] = self.filters.to_dict()
        if self.metric is not None:
            payload["metric"] = self.metric.to_dict()
        if self.comparison is not None:
            payload["comparison"] = self.comparison.to_dict()
        if self.time is not None:
            payload["time"] = self.time.to_dict()
        return payload


def condition(field: str, operator: str, value: Any = None) -> FilterExpression:
    return FilterExpression(kind="condition", condition=FilterCondition(field=field, operator=operator, value=value))


def and_filter(*items: FilterExpression) -> FilterExpression | None:
    filtered = tuple(item for item in items if item is not None)
    if not filtered:
        return None
    if len(filtered) == 1:
        return filtered[0]
    return FilterExpression(kind="and", conditions=filtered)


def or_filter(*items: FilterExpression) -> FilterExpression | None:
    filtered = tuple(item for item in items if item is not None)
    if not filtered:
        return None
    if len(filtered) == 1:
        return filtered[0]
    return FilterExpression(kind="or", conditions=filtered)


def not_filter(item: FilterExpression | None) -> FilterExpression | None:
    if item is None:
        return None
    return FilterExpression(kind="not", conditions=(item,))


def analytics_plan_json_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "$defs": {
            "condition": {
                "type": "object",
                "additionalProperties": False,
                "required": ["field", "operator", "value"],
                "properties": {
                    "field": {"type": "string", "enum": sorted(FILTERABLE_FIELDS)},
                    "operator": {
                        "type": "string",
                        "enum": sorted({operator for values in FIELD_OPERATORS.values() for operator in values}),
                    },
                    "value": {
                        "anyOf": [
                            {"type": "string"},
                            {"type": "number"},
                            {"type": "boolean"},
                            {"type": "null"},
                            {"type": "array", "items": {"anyOf": [{"type": "string"}, {"type": "number"}]}},
                        ]
                    },
                },
            },
            "filter_expression": {
                "type": "object",
                "additionalProperties": False,
                "required": ["kind"],
                "properties": {
                    "kind": {"type": "string", "enum": ["condition", "and", "or", "not"]},
                    "condition": {"$ref": "#/$defs/condition"},
                    "conditions": {"type": "array", "items": {"$ref": "#/$defs/filter_expression"}},
                },
            },
            "metric_operand": {
                "type": "object",
                "additionalProperties": False,
                "required": ["metric"],
                "properties": {
                    "metric": {"type": "string", "enum": sorted(SUPPORTED_METRICS)},
                    "filters": {"anyOf": [{"$ref": "#/$defs/filter_expression"}, {"type": "null"}]},
                },
            },
            "metric_definition": {
                "type": "object",
                "additionalProperties": False,
                "required": ["type"],
                "properties": {
                    "type": {"type": "string", "enum": sorted(SUPPORTED_METRICS)},
                    "operand": {"anyOf": [{"$ref": "#/$defs/metric_operand"}, {"type": "null"}]},
                    "numerator": {"anyOf": [{"$ref": "#/$defs/metric_operand"}, {"type": "null"}]},
                    "denominator": {"anyOf": [{"$ref": "#/$defs/metric_operand"}, {"type": "null"}]},
                    "left": {"anyOf": [{"$ref": "#/$defs/metric_operand"}, {"type": "null"}]},
                    "right": {"anyOf": [{"$ref": "#/$defs/metric_operand"}, {"type": "null"}]},
                },
            },
            "comparison_definition": {
                "type": "object",
                "additionalProperties": False,
                "required": ["operation", "left", "right"],
                "properties": {
                    "operation": {"type": "string", "enum": sorted(SUPPORTED_COMPARISON_OPERATIONS)},
                    "left": {"$ref": "#/$defs/metric_operand"},
                    "right": {"$ref": "#/$defs/metric_operand"},
                },
            },
            "order_by": {
                "type": "object",
                "additionalProperties": False,
                "required": ["field", "direction"],
                "properties": {
                    "field": {"type": "string", "enum": ["metric", *sorted(GROUPABLE_FIELDS)]},
                    "direction": {"type": "string", "enum": ["asc", "desc"]},
                },
            },
            "time_range": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "start_seconds": {"anyOf": [{"type": "number"}, {"type": "null"}]},
                    "end_seconds": {"anyOf": [{"type": "number"}, {"type": "null"}]},
                    "bucket_seconds": {"anyOf": [{"type": "number"}, {"type": "null"}]},
                },
            },
        },
        "required": [
            "entity",
        ],
        "properties": {
            "entity": {"type": "string", "enum": sorted(SUPPORTED_ENTITIES)},
            "filters": {"anyOf": [{"$ref": "#/$defs/filter_expression"}, {"type": "null"}]},
            "group_by": {"type": "array", "items": {"type": "string", "enum": sorted(GROUPABLE_FIELDS)}},
            "metric": {"anyOf": [{"$ref": "#/$defs/metric_definition"}, {"type": "null"}]},
            "comparison": {"anyOf": [{"$ref": "#/$defs/comparison_definition"}, {"type": "null"}]},
            "order_by": {"type": "array", "items": {"$ref": "#/$defs/order_by"}},
            "limit": {"anyOf": [{"type": "integer", "minimum": 1}, {"type": "null"}]},
            "time": {"anyOf": [{"$ref": "#/$defs/time_range"}, {"type": "null"}]},
            "include_camera_ids": {"type": "array", "items": {"type": "string"}},
            "exclude_camera_ids": {"type": "array", "items": {"type": "string"}},
            "show_evidence": {"type": "boolean"},
            "result_shape": {"type": "string", "enum": sorted(SUPPORTED_RESULT_SHAPES)},
            "context_reference": {"anyOf": [{"type": "string", "enum": ["previous_results"]}, {"type": "null"}]},
            "context_resolution": {"anyOf": [{"type": "string", "enum": ["single", "multiple"]}, {"type": "null"}]},
        },
    }
