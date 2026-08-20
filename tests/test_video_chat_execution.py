from __future__ import annotations

from src.vehicle_analytics import VehicleRecord
from src.video_chat import ChatVehicleQuery, analytics_plan_from_chat_query
from src.video_chat_execution import compile_analytics_plan, execute_analytics_plan


def _record(run_id: str, camera_id: str, track_id: str, vehicle_class: str, colour: str, *, plate_text: str | None = None) -> VehicleRecord:
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
        plate_text=plate_text,
        plate_detected=plate_text is not None,
    )


def test_legacy_chat_query_compiles_into_generic_analytics_plan_and_executes_ranking() -> None:
    parsed = ChatVehicleQuery(
        intent="GROUP",
        include_classes=["CAR"],
        include_colours=["WHITE"],
        selected_run_ids=["RUN_A", "RUN_B"],
        group_by="run_camera",
        sort_by="count_desc",
        limit=1,
    )
    plan = analytics_plan_from_chat_query(parsed, provenance="rule_based_fallback")
    compiled = compile_analytics_plan(plan)
    records = [
        _record("RUN_A", "CAM_001", "TRACK_1", "CAR", "WHITE"),
        _record("RUN_A", "CAM_001", "TRACK_2", "CAR", "WHITE"),
        _record("RUN_B", "CAM_002", "TRACK_3", "CAR", "WHITE"),
        _record("RUN_B", "CAM_003", "TRACK_4", "CAR", "WHITE"),
        _record("RUN_B", "CAM_003", "TRACK_5", "CAR", "WHITE"),
        _record("RUN_B", "CAM_003", "TRACK_7", "CAR", "WHITE"),
        _record("RUN_B", "CAM_003", "TRACK_6", "BUS", "WHITE"),
    ]

    result = execute_analytics_plan(records, plan)

    assert compiled.to_dict()["plan"]["group_by"] == ["run_camera"]
    assert compiled.to_dict()["plan"]["provenance"] == "rule_based_fallback"
    assert result["by_run_camera"]["RUN_B / CAM_003"] == 3
    assert result["ranking_result"]["winners"][0]["run_id"] == "RUN_B"
    assert result["ranking_result"]["winners"][0]["camera_id"] == "CAM_003"
