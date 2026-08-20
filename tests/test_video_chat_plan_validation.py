from __future__ import annotations

from src.video_chat_plan_validation import (
    analytics_plan_from_payload,
    normalize_analytics_plan_payload,
    validate_analytics_plan_schema,
    validate_analytics_plan_semantics,
)


def _base_payload(**overrides):
    payload = {
        "entity": "vehicle",
        "filters": None,
        "group_by": [],
        "metric": {"type": "vehicle_count", "operand": {"metric": "vehicle_count", "filters": None}},
        "comparison": None,
        "order_by": [],
        "limit": None,
        "time": None,
        "include_camera_ids": [],
        "exclude_camera_ids": [],
        "show_evidence": False,
        "result_shape": "scalar",
        "context_reference": None,
        "context_resolution": None,
    }
    payload.update(overrides)
    return payload


def test_plan_schema_validation_rejects_invalid_group_by_field_and_operator() -> None:
    payload = _base_payload(
        group_by=["banana"],
        filters={"kind": "condition", "condition": {"field": "plate_text", "operator": "around", "value": "HR"}},
    )
    issues = validate_analytics_plan_schema(payload)
    codes = {issue.code for issue in issues}
    assert "invalid_group_by" in codes
    assert "invalid_filter_operator" in codes


def test_plan_semantic_validation_rejects_comparison_without_operands_and_invalid_time_range() -> None:
    payload = _base_payload(
        result_shape="comparison",
        comparison=None,
        time={"start_seconds": 10.0, "end_seconds": 5.0, "bucket_seconds": None},
    )
    plan = analytics_plan_from_payload(payload, provenance="qwen")
    issues = validate_analytics_plan_semantics(plan)
    codes = {issue.code for issue in issues}
    assert "comparison_requires_operands" in codes
    assert "invalid_time_range" in codes


def test_plan_semantic_validation_rejects_ranking_without_metric_and_derived_metric_operands() -> None:
    ranking_payload = _base_payload(result_shape="ranking", group_by=["class"], metric=None, order_by=[{"field": "metric", "direction": "desc"}], limit=1)
    ratio_payload = _base_payload(metric={"type": "ratio", "left": {"metric": "vehicle_count", "filters": None}}, result_shape="scalar")
    percentage_payload = _base_payload(metric={"type": "percentage", "numerator": {"metric": "vehicle_count", "filters": None}}, result_shape="scalar")

    ranking_issues = validate_analytics_plan_semantics(analytics_plan_from_payload(ranking_payload, provenance="qwen"))
    ratio_issues = validate_analytics_plan_semantics(analytics_plan_from_payload(ratio_payload, provenance="qwen"))
    percentage_issues = validate_analytics_plan_semantics(analytics_plan_from_payload(percentage_payload, provenance="qwen"))

    assert {issue.code for issue in ranking_issues} >= {"ranking_requires_metric"}
    assert {issue.code for issue in ratio_issues} >= {"ratio_requires_operands"}
    assert {issue.code for issue in percentage_issues} >= {"percentage_requires_denominator"}


def test_plan_semantic_validation_checks_plate_exact_and_allows_partial_fragments() -> None:
    exact_payload = _base_payload(filters={"kind": "condition", "condition": {"field": "plate_text", "operator": "eq", "value": "THEVEHICLESWHOSE"}})
    prefix_payload = _base_payload(filters={"kind": "condition", "condition": {"field": "plate_text", "operator": "starts_with", "value": "HR"}})
    suffix_payload = _base_payload(filters={"kind": "condition", "condition": {"field": "plate_text", "operator": "ends_with", "value": "62"}})
    contains_payload = _base_payload(filters={"kind": "condition", "condition": {"field": "plate_text", "operator": "contains", "value": "590"}})

    exact_issues = validate_analytics_plan_semantics(analytics_plan_from_payload(exact_payload, provenance="qwen"))
    prefix_issues = validate_analytics_plan_semantics(analytics_plan_from_payload(prefix_payload, provenance="qwen"))
    suffix_issues = validate_analytics_plan_semantics(analytics_plan_from_payload(suffix_payload, provenance="qwen"))
    contains_issues = validate_analytics_plan_semantics(analytics_plan_from_payload(contains_payload, provenance="qwen"))

    assert {issue.code for issue in exact_issues} >= {"invalid_exact_plate"}
    assert prefix_issues == []
    assert suffix_issues == []
    assert contains_issues == []


def test_plan_normalization_canonicalizes_aliases_and_safe_multi_run_camera_grouping() -> None:
    payload = _base_payload(
        filters={"kind": "condition", "condition": {"field": "vehicle_color", "operator": "eq", "value": "gray"}},
        group_by=["camera"],
    )
    normalized, actions = normalize_analytics_plan_payload(payload, selected_run_ids=["RUN_A", "RUN_B"])

    assert normalized["filters"]["condition"]["field"] == "colour"
    assert normalized["filters"]["condition"]["value"] == "GREY"
    assert normalized["group_by"] == ["run_camera"]
    assert actions == ["group_by camera -> run_camera for multi-run scope"]
