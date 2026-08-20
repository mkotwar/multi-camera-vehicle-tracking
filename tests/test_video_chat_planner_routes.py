from __future__ import annotations

import pytest

from src.vehicle_analytics import VehicleRecord
from src.vehicle_nlp import VehicleQueryParseError
from src.video_chat import ChatVehicleQuery, handle_video_chat


class _ProviderSequence:
    def __init__(self, payloads):
        self.payloads = list(payloads)
        self.calls = 0
        self.contexts = []

    def parse(self, message, context):
        self.calls += 1
        self.contexts.append(context)
        payload = self.payloads[min(self.calls - 1, len(self.payloads) - 1)]
        if isinstance(payload, Exception):
            raise payload
        return payload


class _CameraScopeRepository:
    def __init__(self, cameras_by_run):
        self.cameras_by_run = cameras_by_run

    def list_cameras(self, *, run_id=None):
        return [{"camera_id": camera_id} for camera_id in self.cameras_by_run.get(run_id, [])]

    def get_physical_vehicle(self, *, vehicle_id, run_id):
        return None

    def get_track(self, *, camera_id, track_id, run_id):
        return {
            "run_id": run_id,
            "camera_id": camera_id,
            "track_id": track_id,
            "local_track_id": f"{camera_id}:{track_id}",
            "vehicle_class": "CAR",
            "colour": "WHITE",
            "plate_text": None,
            "first_seen_seconds": 1.0,
            "last_seen_seconds": 2.0,
            "best_crop_parts": None,
        }

    def resolve_media_path(self, *, run_id, category, relative_parts):
        return None


def _record(run_id: str, camera_id: str, track_id: str, vehicle_class: str = "CAR", colour: str = "WHITE") -> VehicleRecord:
    return VehicleRecord(
        run_id=run_id,
        vehicle_id=f"{camera_id}:{track_id}",
        local_track_id=f"{camera_id}:{track_id}",
        camera_id=camera_id,
        vehicle_class=vehicle_class,
        colour=colour,
        first_seen_seconds=1.0,
        last_seen_seconds=2.0,
        observation_count=4,
        status="COMPLETED",
    )


def _analytics_payload(**overrides):
    payload = {
        "entity": "vehicle",
        "filters": {"kind": "condition", "condition": {"field": "class", "operator": "in", "value": ["CAR"]}},
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


def test_planner_route_qwen_direct() -> None:
    response = handle_video_chat(
        message="how many cars are there",
        run_ids=["RUN_A"],
        records=[_record("RUN_A", "CAM_001", "TRACK_1"), _record("RUN_A", "CAM_001", "TRACK_2", "BUS", "BLUE")],
        repository=_CameraScopeRepository({"RUN_A": ["CAM_001"]}),  # type: ignore[arg-type]
        llm_provider=_ProviderSequence([_analytics_payload()]),
    )

    assert response["parser_used"] in {"qwen", "qwen_repaired"}


def test_planner_route_qwen_normalized() -> None:
    response = handle_video_chat(
        message="show gray cars",
        run_ids=["RUN_A"],
        records=[_record("RUN_A", "CAM_001", "TRACK_1", "CAR", "GREY")],
        repository=_CameraScopeRepository({"RUN_A": ["CAM_001"]}),  # type: ignore[arg-type]
        llm_provider=_ProviderSequence([
            _analytics_payload(
                filters={
                    "kind": "and",
                    "conditions": [
                        {"kind": "condition", "condition": {"field": "class", "operator": "in", "value": ["CAR"]}},
                        {"kind": "condition", "condition": {"field": "vehicle_color", "operator": "eq", "value": "gray"}},
                    ],
                },
                result_shape="list",
                show_evidence=True,
            )
        ]),
    )

    assert response["parser_used"] in {"qwen_normalized", "qwen_repaired"}


def test_planner_route_qwen_repaired() -> None:
    response = handle_video_chat(
        message="count vehicles camera wise",
        run_ids=["RUN_A"],
        records=[_record("RUN_A", "CAM_001", "TRACK_1"), _record("RUN_A", "CAM_002", "TRACK_2", "BUS", "BLUE")],
        repository=_CameraScopeRepository({"RUN_A": ["CAM_001", "CAM_002"]}),  # type: ignore[arg-type]
        llm_provider=_ProviderSequence([_analytics_payload(filters=None, result_shape="scalar")]),
    )

    assert response["parser_used"] == "qwen_repaired"
    assert response["parsed_query"]["group_by"] == "camera"


def test_planner_route_qwen_retry() -> None:
    provider = _ProviderSequence([
        _analytics_payload(group_by=["banana"]),
        _analytics_payload(group_by=["class"], result_shape="ranking", order_by=[{"field": "metric", "direction": "desc"}], limit=1),
    ])
    response = handle_video_chat(
        message="which vehicle class have more vehicles",
        run_ids=["RUN_A"],
        records=[_record("RUN_A", "CAM_001", "TRACK_1"), _record("RUN_A", "CAM_001", "TRACK_2", "BUS", "BLUE")],
        repository=_CameraScopeRepository({"RUN_A": ["CAM_001"]}),  # type: ignore[arg-type]
        llm_provider=provider,
    )

    assert response["parser_used"] == "qwen_retry"
    assert provider.calls == 2


def test_planner_route_rule_based_fallback_on_double_invalid_qwen() -> None:
    response = handle_video_chat(
        message="how many cars are there",
        run_ids=["RUN_A"],
        records=[_record("RUN_A", "CAM_001", "TRACK_1"), _record("RUN_A", "CAM_001", "TRACK_2", "BUS", "BLUE")],
        repository=_CameraScopeRepository({"RUN_A": ["CAM_001"]}),  # type: ignore[arg-type]
        llm_provider=_ProviderSequence([_analytics_payload(group_by=["banana"]), _analytics_payload(group_by=["banana"])]),
    )

    assert response["parser_used"] == "rule_based_fallback"


def test_planner_route_controlled_failure_when_retry_and_fallback_both_fail() -> None:
    with pytest.raises(VehicleQueryParseError):
        handle_video_chat(
            message="please locate passenger vehicles that look pale",
            run_ids=["RUN_A"],
            records=[_record("RUN_A", "CAM_001", "TRACK_1")],
            repository=_CameraScopeRepository({"RUN_A": ["CAM_001"]}),  # type: ignore[arg-type]
            llm_provider=_ProviderSequence([_analytics_payload(group_by=["banana"]), _analytics_payload(group_by=["banana"])]),
        )


def test_valid_qwen_plan_never_calls_legacy_rule_parser(monkeypatch) -> None:
    def fail_legacy(*args, **kwargs):
        raise AssertionError("legacy parser must not run for valid Qwen plans")

    generic_calls = []

    def track_execute(records, plan, **kwargs):
        generic_calls.append(plan.provenance)
        return {"total": 1, "metric_value": 1, "by_class": {"CAR": 1}, "by_colour": {"WHITE": 1}, "vehicle_ids": ["CAM_001:TRACK_1"]}

    monkeypatch.setattr("src.video_chat._parse_rule_chat_query", fail_legacy)
    monkeypatch.setattr("src.video_chat.execute_analytics_plan", track_execute)

    response = handle_video_chat(
        message="how many cars are there",
        run_ids=["RUN_A"],
        records=[_record("RUN_A", "CAM_001", "TRACK_1")],
        repository=_CameraScopeRepository({"RUN_A": ["CAM_001"]}),  # type: ignore[arg-type]
        llm_provider=_ProviderSequence([_analytics_payload()]),
    )

    assert response["parser_used"] in {"qwen", "qwen_repaired"}
    assert response["planner_trace"]["final_provenance"] == "qwen"
    assert generic_calls == ["qwen"]


def test_repaired_qwen_plan_never_calls_legacy_rule_parser(monkeypatch) -> None:
    def fail_legacy(*args, **kwargs):
        raise AssertionError("legacy parser must not run for repaired Qwen plans")

    monkeypatch.setattr("src.video_chat._parse_rule_chat_query", fail_legacy)

    response = handle_video_chat(
        message="which vehicle class has the most vehicles",
        run_ids=["RUN_A"],
        records=[_record("RUN_A", "CAM_001", "TRACK_1"), _record("RUN_A", "CAM_002", "TRACK_2", "BUS", "BLUE")],
        repository=_CameraScopeRepository({"RUN_A": ["CAM_001", "CAM_002"]}),  # type: ignore[arg-type]
        llm_provider=_ProviderSequence([
            _analytics_payload(group_by=["class"], result_shape="grouped", order_by=[], limit=None),
        ]),
    )

    assert response["parser_used"] == "qwen_repaired"
    assert response["planner_trace"]["repairs"]


def test_retry_valid_second_attempt_never_calls_legacy_rule_parser(monkeypatch) -> None:
    provider = _ProviderSequence([
        _analytics_payload(group_by=["banana"]),
        _analytics_payload(group_by=["class"], result_shape="ranking", order_by=[{"field": "metric", "direction": "desc"}], limit=1),
    ])

    def fail_legacy(*args, **kwargs):
        raise AssertionError("legacy parser must not run when bounded retry succeeds")

    monkeypatch.setattr("src.video_chat._parse_rule_chat_query", fail_legacy)

    response = handle_video_chat(
        message="which vehicle class have more vehicles",
        run_ids=["RUN_A"],
        records=[_record("RUN_A", "CAM_001", "TRACK_1"), _record("RUN_A", "CAM_001", "TRACK_2", "BUS", "BLUE")],
        repository=_CameraScopeRepository({"RUN_A": ["CAM_001"]}),  # type: ignore[arg-type]
        llm_provider=provider,
    )

    assert response["parser_used"] == "qwen_retry"
    assert provider.calls == 2
    assert response["planner_trace"]["qwen_retry_used"] is True


def test_retry_is_bounded_to_one_and_fallback_calls_legacy_once(monkeypatch) -> None:
    provider = _ProviderSequence([_analytics_payload(group_by=["banana"]), _analytics_payload(group_by=["banana"])])
    legacy_calls = []

    def legacy_parser(text, context):
        legacy_calls.append((text, bool(context)))
        return ChatVehicleQuery(intent="COUNT", include_classes=["CAR"])

    monkeypatch.setattr("src.video_chat._parse_rule_chat_query", legacy_parser)

    response = handle_video_chat(
        message="how many cars are there",
        run_ids=["RUN_A"],
        records=[_record("RUN_A", "CAM_001", "TRACK_1"), _record("RUN_A", "CAM_001", "TRACK_2", "BUS", "BLUE")],
        repository=_CameraScopeRepository({"RUN_A": ["CAM_001"]}),  # type: ignore[arg-type]
        llm_provider=provider,
    )

    assert response["parser_used"] == "rule_based_fallback"
    assert provider.calls == 2
    assert len(legacy_calls) == 1
    assert response["planner_trace"]["fallback_used"] is True


def test_qwen_timeout_uses_rule_based_fallback(monkeypatch) -> None:
    provider = _ProviderSequence([TimeoutError("timed out")])
    legacy_calls = []

    def legacy_parser(text, context):
        legacy_calls.append(text)
        return ChatVehicleQuery(intent="COUNT", include_classes=["CAR"])

    monkeypatch.setattr("src.video_chat._parse_rule_chat_query", legacy_parser)

    response = handle_video_chat(
        message="how many cars are there",
        run_ids=["RUN_A"],
        records=[_record("RUN_A", "CAM_001", "TRACK_1")],
        repository=_CameraScopeRepository({"RUN_A": ["CAM_001"]}),  # type: ignore[arg-type]
        llm_provider=provider,
    )

    assert response["parser_used"] == "rule_based_fallback"
    assert response["fallback_reason"] == "qwen_timeout"
    assert provider.calls == 1
    assert len(legacy_calls) == 1


def test_qwen_unavailable_uses_rule_based_fallback(monkeypatch) -> None:
    provider = _ProviderSequence([RuntimeError("Ollama unavailable")])
    legacy_calls = []

    def legacy_parser(text, context):
        legacy_calls.append(text)
        return ChatVehicleQuery(intent="COUNT", include_classes=["CAR"])

    monkeypatch.setattr("src.video_chat._parse_rule_chat_query", legacy_parser)

    response = handle_video_chat(
        message="how many cars are there",
        run_ids=["RUN_A"],
        records=[_record("RUN_A", "CAM_001", "TRACK_1")],
        repository=_CameraScopeRepository({"RUN_A": ["CAM_001"]}),  # type: ignore[arg-type]
        llm_provider=provider,
    )

    assert response["parser_used"] == "rule_based_fallback"
    assert response["fallback_reason"] == "qwen_unavailable"
    assert provider.calls == 1
    assert len(legacy_calls) == 1


def test_qwen_malformed_output_retries_once_then_falls_back(monkeypatch) -> None:
    provider = _ProviderSequence([RuntimeError("Ollama message.content was not valid JSON"), RuntimeError("Ollama message.content was not valid JSON")])
    legacy_calls = []

    def legacy_parser(text, context):
        legacy_calls.append(text)
        return ChatVehicleQuery(intent="COUNT", include_classes=["CAR"])

    monkeypatch.setattr("src.video_chat._parse_rule_chat_query", legacy_parser)

    response = handle_video_chat(
        message="how many cars are there",
        run_ids=["RUN_A"],
        records=[_record("RUN_A", "CAM_001", "TRACK_1")],
        repository=_CameraScopeRepository({"RUN_A": ["CAM_001"]}),  # type: ignore[arg-type]
        llm_provider=provider,
    )

    assert response["parser_used"] == "rule_based_fallback"
    assert response["fallback_reason"] == "qwen_malformed_output"
    assert provider.calls == 2
    assert len(legacy_calls) == 1


def test_unsupported_capability_is_not_parse_failure() -> None:
    response = handle_video_chat(
        message="what is the driver's name?",
        run_ids=["RUN_A"],
        records=[_record("RUN_A", "CAM_001", "TRACK_1")],
        repository=_CameraScopeRepository({"RUN_A": ["CAM_001"]}),  # type: ignore[arg-type]
        llm_provider=_ProviderSequence([
            _analytics_payload(result_shape="unsupported_capability", filters=None),
        ]),
    )

    assert response["parser_used"] in {"qwen", "qwen_repaired"}
    assert response["analytics_result"]["status"] == "unsupported"
    assert response["result_status"] == "unsupported"
    assert response["answer"] == "That information is not available in the current vehicle analytics data."


def test_no_data_is_not_parse_failure() -> None:
    response = handle_video_chat(
        message="show CAM_999",
        run_ids=["RUN_A"],
        records=[_record("RUN_A", "CAM_001", "TRACK_1")],
        repository=_CameraScopeRepository({"RUN_A": ["CAM_001"]}),  # type: ignore[arg-type]
        llm_provider=_ProviderSequence([
            _analytics_payload(
                filters={"kind": "condition", "condition": {"field": "camera", "operator": "eq", "value": "CAM_999"}},
                result_shape="list",
                show_evidence=True,
            ),
        ]),
    )

    assert response["parser_used"] in {"qwen", "qwen_repaired"}
    assert response["result_status"] == "no_data"
    assert response["matching_vehicle_ids"] == []
    assert response["analytics_result"]["vehicle_ids"] == []


def test_planner_trace_contains_routing_details() -> None:
    response = handle_video_chat(
        message="how many cars are there",
        run_ids=["RUN_A"],
        records=[_record("RUN_A", "CAM_001", "TRACK_1")],
        repository=_CameraScopeRepository({"RUN_A": ["CAM_001"]}),  # type: ignore[arg-type]
        llm_provider=_ProviderSequence([_analytics_payload()]),
    )

    trace = response["planner_trace"]
    assert trace["query_id"]
    assert trace["raw_query"] == "how many cars are there"
    assert trace["selected_runs"] == ["RUN_A"]
    assert trace["final_provenance"] == "qwen"
    assert trace["compiled_execution_plan"]["repository_source"] == "repository"
    assert trace["execution_duration_ms"] is not None
    assert trace["total_duration_ms"] is not None


def test_valid_qwen_plate_query_uses_primary_path(monkeypatch) -> None:
    def fail_legacy(*args, **kwargs):
        raise AssertionError("legacy parser must not run for valid plate plans")

    monkeypatch.setattr("src.video_chat._parse_rule_chat_query", fail_legacy)

    response = handle_video_chat(
        message="find UP84AT5908",
        run_ids=["RUN_A"],
        records=[_record("RUN_A", "CAM_001", "TRACK_1")],
        repository=_CameraScopeRepository({"RUN_A": ["CAM_001"]}),  # type: ignore[arg-type]
        llm_provider=_ProviderSequence([
            _analytics_payload(
                filters={"kind": "condition", "condition": {"field": "plate_text", "operator": "eq", "value": "UP84AT5908"}},
                result_shape="list",
                show_evidence=True,
            ),
        ]),
    )

    assert response["parser_used"] in {"qwen", "qwen_repaired"}
    assert response["analytics_plan"]["filters"]["conditions"][-1]["condition"]["field"] == "plate_text"


def test_valid_qwen_comparison_query_uses_generic_executor(monkeypatch) -> None:
    def fail_legacy(*args, **kwargs):
        raise AssertionError("legacy parser must not run for valid comparison plans")

    monkeypatch.setattr("src.video_chat._parse_rule_chat_query", fail_legacy)

    response = handle_video_chat(
        message="are there more cars or motorcycles",
        run_ids=["RUN_A"],
        records=[
            _record("RUN_A", "CAM_001", "TRACK_1", "CAR"),
            _record("RUN_A", "CAM_001", "TRACK_2", "CAR"),
            _record("RUN_A", "CAM_001", "TRACK_3", "MOTORCYCLE"),
        ],
        repository=_CameraScopeRepository({"RUN_A": ["CAM_001"]}),  # type: ignore[arg-type]
        llm_provider=_ProviderSequence([
                _analytics_payload(
                    filters=None,
                    comparison={
                        "operation": "winner",
                        "left": {"metric": "vehicle_count", "filters": {"kind": "condition", "condition": {"field": "class", "operator": "eq", "value": "CAR"}}},
                        "right": {"metric": "vehicle_count", "filters": {"kind": "condition", "condition": {"field": "class", "operator": "eq", "value": "MOTORCYCLE"}}},
                    },
                result_shape="comparison",
            ),
        ]),
    )

    assert response["parser_used"] in {"qwen", "qwen_repaired"}
    assert response["analytics_result"]["left_total"] == 2
    assert response["analytics_result"]["right_total"] == 1


def test_valid_qwen_time_query_uses_primary_path(monkeypatch) -> None:
    def fail_legacy(*args, **kwargs):
        raise AssertionError("legacy parser must not run for valid time-filter plans")

    monkeypatch.setattr("src.video_chat._parse_rule_chat_query", fail_legacy)

    response = handle_video_chat(
        message="show vehicles between 5 and 10 seconds",
        run_ids=["RUN_A"],
        records=[
            _record("RUN_A", "CAM_001", "TRACK_1", "CAR"),
            _record("RUN_A", "CAM_001", "TRACK_2", "BUS"),
        ],
        repository=_CameraScopeRepository({"RUN_A": ["CAM_001"]}),  # type: ignore[arg-type]
        llm_provider=_ProviderSequence([
            _analytics_payload(
                filters=None,
                result_shape="list",
                show_evidence=True,
                time={"start_seconds": 5.0, "end_seconds": 10.0, "bucket_seconds": None},
            ),
        ]),
    )

    assert response["parser_used"] == "qwen"
    assert response["analytics_plan"]["time"]["start_seconds"] == 5.0
    assert response["analytics_plan"]["time"]["end_seconds"] == 10.0


def test_qwen_analytics_plan_repairs_missing_plate_filter_and_invented_context(monkeypatch) -> None:
    def fail_legacy(*args, **kwargs):
        raise AssertionError("legacy parser must not run for repaired plate plans")

    monkeypatch.setattr("src.video_chat._parse_rule_chat_query", fail_legacy)

    response = handle_video_chat(
        message="find UP84AT5908",
        run_ids=["RUN_A"],
        records=[_record("RUN_A", "CAM_001", "TRACK_1")],
        repository=_CameraScopeRepository({"RUN_A": ["CAM_001"]}),  # type: ignore[arg-type]
        llm_provider=_ProviderSequence([
            _analytics_payload(filters=None, result_shape="list", show_evidence=False, context_reference="previous_results", context_resolution="single"),
        ]),
    )

    assert response["parser_used"] == "qwen_repaired"
    plate_conditions = response["analytics_plan"]["filters"]["conditions"]
    assert [item["condition"]["field"] for item in plate_conditions] == ["plate_readable", "plate_detected", "plate_text"]
    assert plate_conditions[-1]["condition"]["operator"] == "eq"
    assert plate_conditions[-1]["condition"]["value"] == "UP84AT5908"
    assert response["analytics_plan"]["context_reference"] is None
    assert response["analytics_plan"]["show_evidence"] is True


def test_qwen_analytics_plan_repairs_missing_time_range(monkeypatch) -> None:
    def fail_legacy(*args, **kwargs):
        raise AssertionError("legacy parser must not run for repaired time plans")

    monkeypatch.setattr("src.video_chat._parse_rule_chat_query", fail_legacy)

    response = handle_video_chat(
        message="show vehicles between 5 and 10 seconds",
        run_ids=["RUN_A"],
        records=[_record("RUN_A", "CAM_001", "TRACK_1"), _record("RUN_A", "CAM_001", "TRACK_2", "BUS", "BLUE")],
        repository=_CameraScopeRepository({"RUN_A": ["CAM_001"]}),  # type: ignore[arg-type]
        llm_provider=_ProviderSequence([
            _analytics_payload(filters=None, result_shape="list", show_evidence=False, time=None, context_reference="previous_results", context_resolution="single"),
        ]),
    )

    assert response["parser_used"] == "qwen_repaired"
    assert response["analytics_plan"]["time"]["start_seconds"] == 5.0
    assert response["analytics_plan"]["time"]["end_seconds"] == 10.0
    assert response["analytics_plan"]["context_reference"] is None
    assert response["analytics_plan"]["show_evidence"] is True


def test_qwen_analytics_plan_repairs_ranking_metric_and_invented_class_filters(monkeypatch) -> None:
    def fail_legacy(*args, **kwargs):
        raise AssertionError("legacy parser must not run for repaired ranking plans")

    monkeypatch.setattr("src.video_chat._parse_rule_chat_query", fail_legacy)

    response = handle_video_chat(
        message="which colour has the most vehicles",
        run_ids=["RUN_A"],
        records=[
            _record("RUN_A", "CAM_001", "TRACK_1", "CAR", "WHITE"),
            _record("RUN_A", "CAM_001", "TRACK_2", "CAR", "BLACK"),
            _record("RUN_A", "CAM_001", "TRACK_3", "MOTORCYCLE", "BLACK"),
        ],
        repository=_CameraScopeRepository({"RUN_A": ["CAM_001"]}),  # type: ignore[arg-type]
        llm_provider=_ProviderSequence([
            _analytics_payload(
                filters={"kind": "condition", "condition": {"field": "class", "operator": "eq", "value": "MOTORCYCLE"}},
                group_by=["colour"],
                metric=None,
                result_shape="ranking",
                order_by=[{"field": "metric", "direction": "desc"}],
                limit=1,
                context_reference="previous_results",
                context_resolution="single",
            ),
        ]),
    )

    assert response["parser_used"] == "qwen_repaired"
    assert response["analytics_plan"]["metric"]["type"] == "vehicle_count"
    assert "filters" not in response["analytics_plan"]
    assert response["analytics_plan"]["context_reference"] is None


def test_multi_run_camera_grouping_is_deterministically_normalized() -> None:
    response = handle_video_chat(
        message="which camera has the most vehicles",
        run_ids=["RUN_A", "RUN_B"],
        records=[
            _record("RUN_A", "CAM_001", "TRACK_1"),
            _record("RUN_B", "CAM_001", "TRACK_2"),
            _record("RUN_B", "CAM_002", "TRACK_3"),
        ],
        repository=_CameraScopeRepository({"RUN_A": ["CAM_001"], "RUN_B": ["CAM_001", "CAM_002"]}),  # type: ignore[arg-type]
        llm_provider=_ProviderSequence([
            _analytics_payload(group_by=["camera"], result_shape="ranking", order_by=[{"field": "metric", "direction": "desc"}], limit=1),
        ]),
    )

    assert response["parser_used"] in {"qwen_normalized", "qwen_repaired"}
    assert response["analytics_plan"]["group_by"] == ["run_camera"]


def test_qwen_and_rule_fallback_converge_on_same_executor_result(monkeypatch) -> None:
    records = [
        _record("RUN_A", "CAM_001", "TRACK_1", "CAR", "WHITE"),
        _record("RUN_A", "CAM_001", "TRACK_2", "CAR", "WHITE"),
        _record("RUN_A", "CAM_001", "TRACK_3", "BUS", "BLUE"),
    ]
    repository = _CameraScopeRepository({"RUN_A": ["CAM_001"]})  # type: ignore[arg-type]

    qwen_response = handle_video_chat(
        message="show white cars",
        run_ids=["RUN_A"],
        records=records,
        repository=repository,
        llm_provider=_ProviderSequence([
            _analytics_payload(
                filters={
                    "kind": "and",
                    "conditions": [
                        {"kind": "condition", "condition": {"field": "class", "operator": "in", "value": ["CAR"]}},
                        {"kind": "condition", "condition": {"field": "colour", "operator": "in", "value": ["WHITE"]}},
                    ],
                },
                result_shape="list",
                show_evidence=True,
            ),
        ]),
    )

    fallback_response = handle_video_chat(
        message="show white cars",
        run_ids=["RUN_A"],
        records=records,
        repository=repository,
        llm_provider=_ProviderSequence([RuntimeError("Ollama unavailable")]),
    )

    assert qwen_response["matching_vehicle_ids"] == fallback_response["matching_vehicle_ids"]
    assert qwen_response["analytics_result"]["vehicle_ids"] == fallback_response["analytics_result"]["vehicle_ids"]
