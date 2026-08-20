from __future__ import annotations

import json
from pathlib import Path
from urllib.error import URLError

import pytest

from src.ollama_qwen_provider import DEFAULT_OLLAMA_NUM_PREDICT, DEFAULT_OLLAMA_TEMPERATURE, OllamaQwenChatLLMProvider, build_chat_llm_provider_from_env
from src.run_repository import RunRepository
from src.vehicle_analytics import VehicleRecord, load_vehicle_records_from_tracks_json, vehicle_records_from_physical_vehicles
from src.vehicle_nlp import VehicleQueryParseError
from src.video_chat import _parse_plate_text_query, execute_chat_vehicle_query, handle_video_chat, parse_chat_vehicle_query, parse_chat_vehicle_query_detailed


class _FakeProvider:
    def __init__(self, payload=None, error: Exception | None = None):
        self.payload = payload
        self.error = error
        self.context = None

    def parse(self, message, context):
        self.context = context
        if self.error is not None:
            raise self.error
        return self.payload


class _FakeHTTPResponse:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self):
        return json.dumps(self.payload).encode("utf-8")


class _PhysicalEvidenceRepository:
    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        self.member_crop = run_dir / "05_florence_selected_crops" / "CAM_001" / "TRACK_22" / "frame_000022_MIDDLE.jpg"
        self.member_crop.parent.mkdir(parents=True, exist_ok=True)
        self.member_crop.write_bytes(b"member crop")

    def get_physical_vehicle(self, *, vehicle_id: str, run_id: str) -> dict | None:
        if vehicle_id == "VEHICLE_001":
            return {
                "run_id": run_id,
                "vehicle_id": "VEHICLE_001",
                "vehicle_class": "CAR",
                "vehicle_colour": "WHITE",
                "primary_camera_id": "CAM_001",
                "consensus_plate_text": "MP09AB1234",
                "first_seen_seconds": 1.0,
                "last_seen_seconds": 9.0,
                "member_track_ids": ["CAM_001:TRACK_11", "CAM_001:TRACK_22"],
                "representative_evidence": [
                    {
                        "local_track_id": "CAM_001:TRACK_11",
                        "vehicle_crop_path": str(self.run_dir / "evidence" / "CAM_001" / "CAM_001_TRACK_11" / "crops" / "missing.jpg"),
                    }
                ],
            }
        if vehicle_id == "VEHICLE_002":
            return {
                "run_id": run_id,
                "vehicle_id": "VEHICLE_002",
                "vehicle_class": "MOTORCYCLE",
                "vehicle_colour": "BLACK",
                "primary_camera_id": "CAM_001",
                "first_seen_seconds": 10.0,
                "last_seen_seconds": 12.0,
                "member_track_ids": ["CAM_001:TRACK_33"],
                "representative_evidence": [
                    {
                        "local_track_id": "CAM_001:TRACK_33",
                        "vehicle_crop_path": str(self.run_dir / "evidence" / "CAM_001" / "CAM_001_TRACK_33" / "crops" / "frame_000033.jpg"),
                    }
                ],
            }
        return None

    def get_track(self, *, camera_id: str, track_id: str, run_id: str) -> dict | None:
        if camera_id == "CAM_001" and track_id == "TRACK_22":
            return {
                "run_id": run_id,
                "camera_id": camera_id,
                "track_id": track_id,
                "local_track_id": "CAM_001:TRACK_22",
                "vehicle_class": "CAR",
                "colour": "WHITE",
                "plate_text": "DL8CAF5030",
                "first_seen_seconds": 2.0,
                "last_seen_seconds": 8.0,
                "best_crop_parts": {
                    "category": "florence_selected_crops",
                    "run_id": run_id,
                    "parts": ["CAM_001", "TRACK_22", "frame_000022_MIDDLE.jpg"],
                },
            }
        if camera_id == "CAM_001" and track_id == "TRACK_33":
            return {
                "run_id": run_id,
                "camera_id": camera_id,
                "track_id": track_id,
                "local_track_id": "CAM_001:TRACK_33",
                "vehicle_class": "MOTORCYCLE",
                "colour": "BLACK",
                "plate_text": "MH12XY9876",
                "first_seen_seconds": 10.0,
                "last_seen_seconds": 12.0,
                "best_crop_parts": None,
            }
        return None

    def resolve_media_path(self, *, run_id: str, category: str, relative_parts: list[str]) -> Path | None:
        base = {
            "florence_selected_crops": self.run_dir / "05_florence_selected_crops",
            "evidence": self.run_dir / "evidence",
        }.get(category)
        if base is None:
            return None
        candidate = base.joinpath(*relative_parts)
        return candidate if candidate.exists() else None


def _valid_payload(**overrides):
    payload = {
        "intent": "LIST",
        "subject": "vehicles",
        "run_filter": None,
        "class_include": ["CAR"],
        "class_exclude": [],
        "colour_include": ["WHITE"],
        "colour_exclude": [],
        "plate_presence": None,
        "plate_detected": None,
        "plate_readable": None,
        "plate_text": None,
        "plate_match_mode": None,
        "start_time": None,
        "end_time": None,
        "group_by": None,
        "operator": None,
        "show_evidence": True,
        "context_reference": None,
    }
    payload.update(overrides)
    return payload


def _normalized_payload(**overrides):
    payload = {
        "intent": "LIST",
        "subject": "vehicles",
        "run_filter": None,
        "classes": ["CAR"],
        "exclude_classes": [],
        "colours": ["WHITE"],
        "exclude_colours": [],
        "plate_presence": None,
        "plate_detected": None,
        "plate_readable": None,
        "plate_text": None,
        "plate_match_mode": None,
        "start_time": None,
        "end_time": None,
        "group_by": None,
        "operator": None,
        "sort_by": None,
        "limit": None,
        "show_evidence": True,
        "context_reference": None,
    }
    payload.update(overrides)
    return payload


def _analytics_payload(**overrides):
    payload = {
        "entity": "vehicle",
        "filters": {
            "kind": "and",
            "conditions": [
                {"kind": "condition", "condition": {"field": "class", "operator": "in", "value": ["CAR"]}},
                {"kind": "condition", "condition": {"field": "colour", "operator": "in", "value": ["WHITE"]}},
            ],
        },
        "group_by": [],
        "metric": {"type": "vehicle_count", "operand": {"metric": "vehicle_count", "filters": None}},
        "comparison": None,
        "order_by": [],
        "limit": None,
        "time": None,
        "include_camera_ids": [],
        "exclude_camera_ids": [],
        "show_evidence": True,
        "result_shape": "list",
        "context_reference": None,
        "context_resolution": None,
    }
    payload.update(overrides)
    return payload


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _build_semantic_run(tmp_path: Path) -> tuple[str, Path]:
    run_id = "20260812_113742"
    run_dir = tmp_path / run_id
    _write_json(run_dir / "summary.json", {"run_id": run_id, "status": "COMPLETED", "processed_frames": 600})
    _write_json(run_dir / "run_metadata.json", {"status": "COMPLETED", "camera_count": 1})
    car_ids = {
        "TRACK_1": "BLACK",
        "TRACK_2": "SILVER",
        "TRACK_3": "BLACK",
        "TRACK_4": "BLACK",
        "TRACK_5": "BLACK",
        "TRACK_11": "BLACK",
        "TRACK_13": "WHITE",
        "TRACK_19": "WHITE",
        "TRACK_26": "WHITE",
        "TRACK_28": "WHITE",
        "TRACK_30": "BLACK",
        "TRACK_33": "WHITE",
        "TRACK_34": "BLACK",
        "TRACK_35": "BLACK",
        "TRACK_42": "WHITE",
        "TRACK_43": "BLACK",
        "TRACK_46": "BLACK",
    }
    motorcycle_ids = {
        "TRACK_7": "WHITE",
        "TRACK_6": "BLACK",
        "TRACK_10": "RED",
        "TRACK_9": "RED",
        "TRACK_14": "BLACK",
        "TRACK_18": "BLACK",
        "TRACK_17": "BLACK",
        "TRACK_24": "RED",
        "TRACK_25": "BLACK",
        "TRACK_23": "BLACK",
        "TRACK_27": "BLACK",
        "TRACK_31": "BLACK",
        "TRACK_32": "BLACK",
        "TRACK_38": "BLACK",
        "TRACK_39": "BLUE",
        "TRACK_40": "RED",
        "TRACK_36": "BLACK",
        "TRACK_45": "BLACK",
    }
    tracks = []
    enrichments = []

    def add(track_id: str, vehicle_class: str, colour: str) -> None:
        crop = run_dir / "05_florence_selected_crops" / "CAM_001" / track_id / "frame_000006_MIDDLE.jpg"
        crop.parent.mkdir(parents=True, exist_ok=True)
        crop.write_bytes(f"{track_id} crop".encode("utf-8"))
        tracks.append(
            {
                "local_track_id": f"CAM_001:{track_id}",
                "camera_id": "CAM_001",
                "status": "COMPLETED",
                "final_class": vehicle_class,
                "first_timestamp_seconds": 6.0 if track_id in {"TRACK_14", "TRACK_16", "TRACK_15", "TRACK_18", "TRACK_17", "TRACK_19", "TRACK_24", "TRACK_25", "TRACK_23"} else 20.0,
                "last_timestamp_seconds": 8.0 if track_id in {"TRACK_14", "TRACK_16", "TRACK_15", "TRACK_18", "TRACK_17", "TRACK_19", "TRACK_24", "TRACK_25", "TRACK_23"} else 25.0,
                "observation_count": 10,
                "vehicle_enrichment": {"vehicle_colour": {"label": colour, "status": "completed" if colour != "UNKNOWN" else "skipped"}},
            }
        )
        enrichments.append(
            {
                "local_track_id": f"CAM_001:{track_id}",
                "camera_id": "CAM_001",
                "vehicle_class": vehicle_class.upper(),
                "vehicle_colour": {"label": colour, "status": "completed"},
                "evidence_used": [{"frame_number": 6, "timestamp_seconds": 6.0, "vehicle_crop_path": str(crop), "evidence_role": "MIDDLE", "selected_for_colour": True}],
                "selected_crop_paths": [str(crop)],
                "status": "completed",
            }
        )

    for track_id, colour in car_ids.items():
        add(track_id, "car", colour)
    for track_id, colour in motorcycle_ids.items():
        add(track_id, "motorcycle", colour)
    add("TRACK_16", "truck", "WHITE")
    add("TRACK_15", "unknown", "UNKNOWN")
    for track_id in ["TRACK_29", "TRACK_37", "TRACK_41", "TRACK_44"]:
        add(track_id, "3wheeler", "GREEN")
    _write_json(run_dir / "tracks.json", tracks)
    _write_json(run_dir / "vehicle_enrichment.json", enrichments)
    return run_id, run_dir


def test_video_chat_qwen_provider_structured_payload_is_accepted() -> None:
    parsed, parser_used, diagnostics = parse_chat_vehicle_query_detailed(
        message="Show white cars",
        context={},
        llm_provider=_FakeProvider(_valid_payload()),
    )

    assert parser_used == "qwen"
    assert diagnostics["llm_attempted"] is True
    assert diagnostics["llm_accepted"] is True
    assert diagnostics["llm_raw_structured_output"] == _valid_payload()
    assert diagnostics["normalized_llm_output"] == _normalized_payload()
    assert parsed.intent == "LIST"
    assert parsed.include_classes == ["CAR"]
    assert parsed.include_colours == ["WHITE"]
    assert parsed.show_evidence is True


def test_video_chat_qwen_multi_class_synonyms_exclusion_group_summary_evidence_and_context() -> None:
    cases = [
        (_valid_payload(intent="COUNT", class_include=["CAR", "MOTORCYCLE"], colour_include=["WHITE"], show_evidence=False), ["CAR", "MOTORCYCLE"], []),
        (_valid_payload(intent="GROUP", class_include=[], class_exclude=["MOTORCYCLE"], colour_include=["WHITE"], group_by="vehicle_class", show_evidence=False), [], ["MOTORCYCLE"]),
        (_valid_payload(intent="SUMMARY", class_include=[], colour_include=[], show_evidence=False), [], []),
        (_valid_payload(intent="LIST", class_include=["3WHEELER"], colour_include=["GREEN"], show_evidence=True), ["3WHEELER"], []),
        (_valid_payload(intent="LIST", class_include=[], colour_include=["WHITE"], context_reference="previous_result", show_evidence=True), [], []),
    ]
    for payload, expected_classes, expected_exclusions in cases:
        context = {"previous_vehicle_ids": ["CAM_001:TRACK_1"]} if payload.get("context_reference") == "previous_result" else {}
        message = "Which of those were white?" if payload.get("context_reference") == "previous_result" else "semantic query"
        parsed, parser_used = parse_chat_vehicle_query(message=message, context=context, llm_provider=_FakeProvider(payload))
        assert parser_used == "qwen"
        assert parsed.include_classes == expected_classes
        assert parsed.exclude_classes == expected_exclusions


def test_video_chat_qwen_receives_small_structured_context() -> None:
    provider = _FakeProvider(_valid_payload(intent="COUNT", class_include=["MOTORCYCLE"], colour_include=["BLACK"], show_evidence=False, context_reference="previous_result"))
    parse_chat_vehicle_query(
        message="How many of those were black?",
        context={
            "previous_filters": {"include_classes": ["MOTORCYCLE"]},
            "previous_vehicle_ids": [f"CAM_001:TRACK_{index}" for index in range(30)],
            "unbounded_history": ["do not send"],
        },
        llm_provider=provider,
    )

    assert provider.context["previous_filters"] == {"include_classes": ["MOTORCYCLE"]}
    assert len(provider.context["previous_vehicle_ids"]) == 20
    assert "unbounded_history" not in provider.context


def test_video_chat_qwen_invalid_schema_falls_back_to_rule_parser() -> None:
    parsed, parser_used, diagnostics = parse_chat_vehicle_query_detailed(
        message="How many cars were there?",
        context={},
        llm_provider=_FakeProvider({"intent": "COUNT", "classes": ["SPACESHIP"]}),
    )

    assert parser_used == "rule_based_fallback"
    assert parsed.include_classes == ["CAR"]
    assert diagnostics["llm_attempted"] is True
    assert diagnostics["llm_accepted"] is False
    assert diagnostics["llm_rejection_reason"] == "unsupported_class:SPACESHIP"
    assert diagnostics["fallback_reason"] == "qwen_schema_validation_failed"


def test_video_chat_qwen_unavailable_and_timeout_fall_back_to_rule_parser() -> None:
    expected = {
        "Ollama unavailable": "qwen_unavailable",
        "timed out": "qwen_timeout",
    }
    for message, fallback_reason in expected.items():
        parsed, parser_used, diagnostics = parse_chat_vehicle_query_detailed(
            message="How many motorcycles were there?",
            context={},
            llm_provider=_FakeProvider(error=RuntimeError(message) if fallback_reason == "qwen_unavailable" else TimeoutError(message)),
        )
        assert parser_used == "rule_based_fallback"
        assert parsed.include_classes == ["MOTORCYCLE"]
        assert diagnostics["fallback_reason"] == fallback_reason


def test_video_chat_qwen_invalid_schema_without_rule_fallback_returns_rule_error() -> None:
    with pytest.raises(VehicleQueryParseError, match="Could not determine chat query intent"):
        parse_chat_vehicle_query(
            message="please locate passenger vehicles that look pale",
            context={},
            llm_provider=_FakeProvider({"intent": "COUNT", "classes": ["SPACESHIP"]}),
        )


@pytest.mark.parametrize("message", ["hello", "hi", "thanks"])
def test_video_chat_general_chat_bypasses_qwen_and_has_no_filters(message: str) -> None:
    provider = _FakeProvider(_valid_payload(intent="COUNT", class_include=[], colour_include=[], show_evidence=False))
    parsed, parser_used, diagnostics = parse_chat_vehicle_query_detailed(
        message=message,
        context={"previous_filters": {"include_colours": ["BLACK"]}, "previous_vehicle_ids": ["CAM_001:TRACK_1"]},
        llm_provider=provider,
    )

    assert parser_used == "rule_based"
    assert diagnostics["message_type"] == "GENERAL_CHAT"
    assert diagnostics["llm_attempted"] is False
    assert provider.context is None
    assert parsed.intent == "GENERAL_CHAT"
    assert parsed.include_colours == []


def test_video_chat_standalone_group_queries_do_not_inherit_llm_context() -> None:
    context = {
        "previous_filters": {"include_classes": ["MOTORCYCLE"], "include_colours": ["BLACK"]},
        "previous_vehicle_ids": ["CAM_001:TRACK_6"],
    }
    parsed, parser_used, diagnostics = parse_chat_vehicle_query_detailed(
        message="give the numbers class wise",
        context=context,
        llm_provider=_FakeProvider(_valid_payload(intent="GROUP", class_include=["MOTORCYCLE"], colour_include=["BLACK"], group_by="vehicle_class", show_evidence=False, context_reference="previous_result")),
    )

    assert parser_used == "rule_based_fallback"
    assert diagnostics["message_type"] == "NEW_ANALYTICS_QUERY"
    assert diagnostics["llm_rejection_reason"] == "incorrect_group_by:expected_class"
    assert parsed.intent == "GROUP"
    assert parsed.group_by == "class"
    assert parsed.include_classes == []
    assert parsed.include_colours == []
    assert parsed.context_reference is None


def test_video_chat_qwen_group_colour_must_preserve_explicit_class() -> None:
    parsed, parser_used, diagnostics = parse_chat_vehicle_query_detailed(
        message="give me the colours of motorcycles",
        context={},
        llm_provider=_FakeProvider(_valid_payload(intent="UNIQUE_COLOURS", class_include=[], colour_include=[], show_evidence=False)),
    )

    assert parser_used == "qwen_repaired"
    assert diagnostics["semantic_repair_applied"] is True
    assert parsed.intent == "GROUP"
    assert parsed.group_by == "colour"
    assert parsed.include_classes == ["MOTORCYCLE"]


def test_video_chat_qwen_group_class_must_preserve_explicit_colour() -> None:
    parsed, parser_used, diagnostics = parse_chat_vehicle_query_detailed(
        message="what vehicle classes were black?",
        context={},
        llm_provider=_FakeProvider(_valid_payload(intent="GROUP", class_include=[], colour_include=[], group_by="vehicle_class", show_evidence=False)),
    )

    assert parser_used == "qwen_repaired"
    assert diagnostics["semantic_repair_applied"] is True
    assert parsed.intent == "GROUP"
    assert parsed.group_by == "class"
    assert parsed.include_colours == ["BLACK"]


def test_video_chat_explicit_follow_up_uses_previous_context() -> None:
    context = {
        "previous_filters": {"include_classes": ["MOTORCYCLE"]},
        "previous_vehicle_ids": ["CAM_001:TRACK_6"],
    }
    parsed, parser_used = parse_chat_vehicle_query(
        message="how many of those were red?",
        context=context,
        llm_provider=None,
    )

    assert parser_used == "rule_based"
    assert parsed.intent == "COUNT"
    assert parsed.include_classes == ["MOTORCYCLE"]
    assert parsed.include_colours == ["RED"]
    assert parsed.context_reference == "previous_results"


@pytest.mark.parametrize(
    ("message", "expected_plate", "expected_show_evidence"),
    [
        ("find DL6CQ1126", "DL6CQ1126", True),
        ("show vehicle DL6C Q1126", "DL6CQ1126", True),
        ("find plate DL6C-Q1126", "DL6CQ1126", True),
    ],
)
def test_video_chat_direct_plate_queries_normalize_registration_like_tokens(
    message: str,
    expected_plate: str,
    expected_show_evidence: bool,
) -> None:
    parsed, parser_used = parse_chat_vehicle_query(message=message, context={}, llm_provider=None)

    assert parser_used == "rule_based"
    assert parsed.intent == "LIST"
    assert parsed.plate_text == expected_plate
    assert parsed.show_evidence is expected_show_evidence


def test_video_chat_contextual_plate_lookup_uses_internal_plate_intent() -> None:
    parsed, parser_used = parse_chat_vehicle_query(
        message="what is its number plate",
        context={
            "previous_vehicle_ids": ["VEHICLE_001"],
            "previous_filters": {"selected_run_ids": ["RUN_A"]},
        },
        llm_provider=None,
    )

    assert parser_used == "rule_based"
    assert parsed.intent == "PLATE_LOOKUP"
    assert parsed.context_reference == "previous_results"
    assert parsed.context_resolution == "single"


@pytest.mark.parametrize("message", ["unknown vehicle", "Show unknown vehicles"])
def test_video_chat_unknown_vehicle_is_explicit_class_filter(message: str) -> None:
    parsed, parser_used, diagnostics = parse_chat_vehicle_query_detailed(
        message=message,
        context={},
        llm_provider=_FakeProvider(_valid_payload(intent="LIST", class_include=[], colour_include=[], show_evidence=True)),
    )

    assert parser_used == "qwen_repaired"
    assert diagnostics["semantic_repair_applied"] is True
    assert parsed.intent == "LIST"
    assert parsed.include_classes == ["UNKNOWN"]
    assert parsed.include_colours == []
    assert parsed.show_evidence is True


def test_video_chat_qwen_unknown_vehicle_payload_is_accepted() -> None:
    parsed, parser_used = parse_chat_vehicle_query(
        message="Show unknown vehicles",
        context={},
        llm_provider=_FakeProvider(_valid_payload(intent="LIST", class_include=["UNKNOWN"], colour_include=[], show_evidence=True)),
    )

    assert parser_used == "qwen"
    assert parsed.intent == "LIST"
    assert parsed.include_classes == ["UNKNOWN"]


def test_video_chat_qwen_alias_normalization_and_harmless_fields() -> None:
    parsed, parser_used, diagnostics = parse_chat_vehicle_query_detailed(
        message="Show gray autos",
        context={},
        llm_provider=_FakeProvider(
            _valid_payload(
                intent="LIST",
                class_include=["AUTO"],
                colour_include=["GRAY"],
                comparison=None,
                sort_by=None,
                limit=None,
                show_evidence=True,
            )
        ),
    )

    assert parser_used == "qwen"
    assert parsed.include_classes == ["3WHEELER"]
    assert parsed.include_colours == ["GREY"]
    assert diagnostics["normalized_llm_output"]["classes"] == ["3WHEELER"]
    assert diagnostics["normalized_llm_output"]["colours"] == ["GREY"]


def test_video_chat_qwen_negation_is_repaired_against_explicit_language() -> None:
    parsed, parser_used, diagnostics = parse_chat_vehicle_query_detailed(
        message="show black vehicles except motorcycles",
        context={},
        llm_provider=_FakeProvider(_valid_payload(intent="LIST", class_include=["MOTORCYCLE"], class_exclude=[], colour_include=["BLACK"], show_evidence=True)),
    )

    assert parser_used == "qwen_repaired"
    assert parsed.include_classes == []
    assert parsed.exclude_classes == ["MOTORCYCLE"]
    assert parsed.include_colours == ["BLACK"]
    assert diagnostics["semantic_repair_applied"] is True
    assert diagnostics["semantic_validation_result"] == "repaired"


@pytest.mark.parametrize(
    "phrase",
    [
        "except bikes",
        "other than bikes",
        "without bikes",
        "excluding motorcycles",
        "apart from motorcycles",
        "anything but bikes",
        "all except motorcycles",
        "not motorcycles",
    ],
)
def test_video_chat_class_negation_variants_map_to_same_exclusion(phrase: str) -> None:
    parsed, _ = parse_chat_vehicle_query(message=f"show vehicles {phrase}", context={}, llm_provider=None)

    assert parsed.intent == "LIST"
    assert parsed.include_classes == []
    assert parsed.exclude_classes == ["MOTORCYCLE"]


@pytest.mark.parametrize("phrase", ["except black", "other than black", "without black", "excluding black", "anything but black"])
def test_video_chat_colour_negation_variants_map_to_same_exclusion(phrase: str) -> None:
    parsed, _ = parse_chat_vehicle_query(message=f"what vehicle colours are present {phrase}", context={}, llm_provider=None)

    assert parsed.intent in {"GROUP", "UNIQUE_COLOURS"}
    assert parsed.group_by in {"colour", None}
    assert parsed.exclude_colours == ["BLACK"]


@pytest.mark.parametrize(
    ("message", "classes", "exclude_classes", "colours", "exclude_colours", "time_range"),
    [
        ("black cars except white", ["CAR"], [], ["BLACK"], ["WHITE"], (None, None)),
        ("red or blue vehicles except motorcycles", [], ["MOTORCYCLE"], ["RED", "BLUE"], [], (None, None)),
        ("cars and trucks except black ones", ["CAR", "TRUCK"], [], [], ["BLACK"], (None, None)),
        ("vehicles except motorcycles between 5 and 10 seconds", [], ["MOTORCYCLE"], [], [], (5.0, 10.0)),
    ],
)
def test_video_chat_composition_include_exclude_and_time(message: str, classes, exclude_classes, colours, exclude_colours, time_range) -> None:
    parsed, _ = parse_chat_vehicle_query(message=f"show {message}", context={}, llm_provider=None)

    assert parsed.include_classes == classes
    assert parsed.exclude_classes == exclude_classes
    assert parsed.include_colours == colours
    assert parsed.exclude_colours == exclude_colours
    assert (parsed.start_time, parsed.end_time) == time_range


def test_video_chat_exact_negation_failures_a_b_c(tmp_path: Path) -> None:
    run_id, run_dir = _build_semantic_run(tmp_path)
    records = load_vehicle_records_from_tracks_json(run_dir / "tracks.json")

    parsed_a, _ = parse_chat_vehicle_query(message="show black vehicles except motorcycles", context={}, llm_provider=None)
    result_a = execute_chat_vehicle_query(records, parsed_a)
    assert parsed_a.include_colours == ["BLACK"]
    assert parsed_a.exclude_classes == ["MOTORCYCLE"]
    assert result_a["total"] == 10
    assert all(record.colour == "BLACK" and record.vehicle_class != "MOTORCYCLE" for record in records if record.vehicle_id in result_a["vehicle_ids"])

    parsed_b, _ = parse_chat_vehicle_query(message="i want to see the vehicles other than car and bike", context={}, llm_provider=None)
    result_b = execute_chat_vehicle_query(records, parsed_b)
    assert parsed_b.exclude_classes == ["CAR", "MOTORCYCLE"]
    assert result_b["total"] == 6
    assert result_b["by_class"]["3WHEELER"] == 4
    assert result_b["by_class"]["TRUCK"] == 1
    assert result_b["by_class"]["UNKNOWN"] == 1

    parsed_c, _ = parse_chat_vehicle_query(message="what vehicle colours are present except black?", context={}, llm_provider=None)
    result_c = execute_chat_vehicle_query(records, parsed_c)
    assert parsed_c.exclude_colours == ["BLACK"]
    assert result_c["colours_present"] == ["WHITE", "SILVER", "RED", "BLUE", "GREEN", "UNKNOWN"]
    assert "BLACK" not in result_c["colours_present"]

    response_a = handle_video_chat(
        message="show black vehicles except motorcycles",
        run_id=run_id,
        tracks_path=str(run_dir / "tracks.json"),
        repository=RunRepository(tmp_path),
        session_context={},
        llm_provider=None,
    )
    assert response_a["evidence"]
    assert response_a["evidence_validation_removed_count"] == 0
    assert all(item["colour"] == "BLACK" and item["vehicle_class"] != "MOTORCYCLE" for item in response_a["evidence"])


def test_video_chat_physical_vehicle_show_them_resolves_member_track_evidence(tmp_path: Path) -> None:
    run_id = "20260815_170454"
    repository = _PhysicalEvidenceRepository(tmp_path / run_id)
    records = [
        type("Record", (), {
            "vehicle_id": "VEHICLE_001",
            "vehicle_class": "CAR",
            "colour": "WHITE",
            "camera_id": "CAM_001",
            "first_seen_seconds": 1.0,
            "last_seen_seconds": 9.0,
        })(),
        type("Record", (), {
            "vehicle_id": "VEHICLE_002",
            "vehicle_class": "MOTORCYCLE",
            "colour": "BLACK",
            "camera_id": "CAM_001",
            "first_seen_seconds": 10.0,
            "last_seen_seconds": 12.0,
        })(),
    ]

    response = handle_video_chat(
        message="Show them",
        run_id=run_id,
        records=records,
        repository=repository,  # type: ignore[arg-type]
        session_context={"previous_vehicle_ids": ["VEHICLE_001", "VEHICLE_002"]},
        llm_provider=None,
    )

    assert response["analytics_result"]["total"] == 2
    assert response["matching_vehicle_ids"] == ["VEHICLE_001", "VEHICLE_002"]
    assert response["evidence_page"]["matching_total"] == 2
    assert response["evidence_page"]["evidence_returned_count"] == 2
    assert response["answer"] == "2 vehicles were observed. Showing 2 of 2."
    assert response["evidence"][0]["vehicle_id"] == "VEHICLE_001"
    assert response["evidence"][0]["member_track_ids"] == ["CAM_001:TRACK_11", "CAM_001:TRACK_22"]
    assert response["evidence"][0]["plate_text"] == "MP09AB1234"
    assert response["evidence"][0]["best_crop_url"] == f"/api/media/florence_selected_crops/{run_id}/CAM_001/TRACK_22/frame_000022_MIDDLE.jpg"
    assert response["evidence"][0]["image_url"] == response["evidence"][0]["best_crop_url"]
    assert response["evidence"][1]["vehicle_id"] == "VEHICLE_002"
    assert response["evidence"][1]["plate_text"] == "MH12XY9876"


def test_video_chat_contextual_plate_lookup_returns_single_readable_plate(tmp_path: Path) -> None:
    run_id = "20260815_170454"
    repository = _PhysicalEvidenceRepository(tmp_path / run_id)
    records = [
        type("Record", (), {
            "vehicle_id": "VEHICLE_001",
            "vehicle_class": "CAR",
            "colour": "WHITE",
            "camera_id": "CAM_001",
            "first_seen_seconds": 1.0,
            "last_seen_seconds": 9.0,
        })(),
    ]

    response = handle_video_chat(
        message="what is its number plate",
        run_id=run_id,
        records=records,
        repository=repository,  # type: ignore[arg-type]
        session_context={"previous_vehicle_ids": ["VEHICLE_001"]},
        llm_provider=None,
    )

    assert response["parsed_query"]["intent"] == "PLATE_LOOKUP"
    assert response["analytics_result"]["readable_count"] == 1
    assert response["analytics_result"]["plate_rows"][0]["plate_text"] == "MP09AB1234"
    assert response["answer"] == "The number plate is MP09AB1234."


def test_video_chat_contextual_plate_lookup_returns_multiple_plate_states(tmp_path: Path) -> None:
    run_id = "20260815_170454"
    repository = _PhysicalEvidenceRepository(tmp_path / run_id)
    records = [
        type("Record", (), {
            "vehicle_id": "VEHICLE_001",
            "vehicle_class": "CAR",
            "colour": "WHITE",
            "camera_id": "CAM_001",
            "first_seen_seconds": 1.0,
            "last_seen_seconds": 9.0,
        })(),
        type("Record", (), {
            "vehicle_id": "VEHICLE_002",
            "vehicle_class": "MOTORCYCLE",
            "colour": "BLACK",
            "camera_id": "CAM_001",
            "first_seen_seconds": 10.0,
            "last_seen_seconds": 12.0,
        })(),
    ]

    response = handle_video_chat(
        message="what are their number plates",
        run_id=run_id,
        records=records,
        repository=repository,  # type: ignore[arg-type]
        session_context={"previous_vehicle_ids": ["VEHICLE_001", "VEHICLE_002"]},
        llm_provider=None,
    )

    assert response["parsed_query"]["intent"] == "PLATE_LOOKUP"
    assert response["parsed_query"]["context_resolution"] == "multiple"
    assert response["analytics_result"]["readable_count"] == 2
    assert response["analytics_result"]["detected_unreadable_count"] == 0
    assert response["answer"] == "2 of the 2 matched vehicles have readable number plates."


def test_video_chat_contextual_plate_lookup_requests_clarification_for_ambiguous_single_reference(tmp_path: Path) -> None:
    run_id = "20260815_170454"
    repository = _PhysicalEvidenceRepository(tmp_path / run_id)
    records = [
        type("Record", (), {
            "vehicle_id": "VEHICLE_001",
            "vehicle_class": "CAR",
            "colour": "WHITE",
            "camera_id": "CAM_001",
            "first_seen_seconds": 1.0,
            "last_seen_seconds": 9.0,
        })(),
        type("Record", (), {
            "vehicle_id": "VEHICLE_002",
            "vehicle_class": "MOTORCYCLE",
            "colour": "BLACK",
            "camera_id": "CAM_001",
            "first_seen_seconds": 10.0,
            "last_seen_seconds": 12.0,
        })(),
    ]

    response = handle_video_chat(
        message="what is its number plate",
        run_id=run_id,
        records=records,
        repository=repository,  # type: ignore[arg-type]
        session_context={"previous_vehicle_ids": ["VEHICLE_001", "VEHICLE_002"]},
        llm_provider=None,
    )

    assert response["analytics_result"]["ambiguous"] is True
    assert response["answer"].startswith("There are multiple vehicles in the current result.")


def test_video_chat_contextual_plate_lookup_handles_missing_plate() -> None:
    repository = _CameraScopeRepository({"RUN_A": ["CAM_001"]})
    records = [_record("RUN_A", "CAM_001", "TRACK_1", "CAR", "WHITE")]

    response = handle_video_chat(
        message="what is its number plate",
        run_id="RUN_A",
        records=records,
        repository=repository,  # type: ignore[arg-type]
        session_context={"previous_vehicle_ids": ["CAM_001:TRACK_1"]},
        llm_provider=None,
    )

    assert response["parsed_query"]["intent"] == "PLATE_LOOKUP"
    assert response["analytics_result"]["no_plate_count"] == 1
    assert response["answer"] == "No number plate was detected for this vehicle."


def test_video_chat_contextual_plate_lookup_handles_detected_unreadable_plate() -> None:
    class _UnreadablePlateRepository(_CameraScopeRepository):
        def get_track(self, *, camera_id: str, track_id: str, run_id: str) -> dict | None:
            payload = super().get_track(camera_id=camera_id, track_id=track_id, run_id=run_id) or {}
            payload.update(
                {
                    "plate_detected": True,
                    "plate_readable": False,
                    "plate_text": None,
                    "plate_raw_text": "LIGAJ7519",
                }
            )
            return payload

    repository = _UnreadablePlateRepository({"RUN_A": ["CAM_001"]})
    records = [_record("RUN_A", "CAM_001", "TRACK_1", "CAR", "WHITE")]

    response = handle_video_chat(
        message="what is its number plate",
        run_id="RUN_A",
        records=records,
        repository=repository,  # type: ignore[arg-type]
        session_context={"previous_vehicle_ids": ["CAM_001:TRACK_1"]},
        llm_provider=None,
    )

    assert response["parsed_query"]["intent"] == "PLATE_LOOKUP"
    assert response["analytics_result"]["detected_unreadable_count"] == 1
    assert response["answer"] == "A plate was detected on this vehicle, but no readable number plate is available."


def test_video_chat_accepts_common_summary_typo() -> None:
    parsed, parser_used = parse_chat_vehicle_query(message="summry of the video", context={}, llm_provider=None)

    assert parser_used == "rule_based"
    assert parsed.intent == "SUMMARY"


class _CameraScopeRepository:
    def __init__(self, cameras_by_run: dict[str, list[str]]) -> None:
        self.cameras_by_run = cameras_by_run

    def list_cameras(self, *, run_id: str | None = None) -> list[dict]:
        return [{"run_id": run_id, "camera_id": camera_id} for camera_id in self.cameras_by_run.get(str(run_id), [])]

    def get_physical_vehicle(self, *, vehicle_id: str, run_id: str) -> dict | None:
        return None

    def get_track(self, *, camera_id: str, track_id: str, run_id: str) -> dict | None:
        return {
            "run_id": run_id,
            "camera_id": camera_id,
            "track_id": track_id,
            "local_track_id": f"{camera_id}:{track_id}",
            "vehicle_class": "CAR",
            "colour": "WHITE",
            "first_seen_seconds": 1.0,
            "last_seen_seconds": 2.0,
            "best_crop_parts": None,
        }

    def resolve_media_path(self, *, run_id: str, category: str, relative_parts: list[str]) -> Path | None:
        return None


def _record(run_id: str, camera_id: str, track_id: str, vehicle_class: str = "CAR", colour: str = "WHITE") -> VehicleRecord:
    local_track_id = f"{camera_id}:{track_id}"
    return VehicleRecord(
        run_id=run_id,
        vehicle_id=local_track_id,
        local_track_id=local_track_id,
        camera_id=camera_id,
        vehicle_class=vehicle_class,
        colour=colour,
        first_seen_seconds=1.0,
        last_seen_seconds=2.0,
        observation_count=3,
        status="COMPLETED",
    )


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("show cars from CAM_001", ["CAM_001"]),
        ("show cars from cam_001", ["CAM_001"]),
        ("show cars from camera 1", ["CAM_001"]),
        ("show cars from cam 1", ["CAM_001"]),
        ("show cars from first camera", ["CAM_001"]),
        ("show cars from camera one", ["CAM_001"]),
        ("show bikes from second camera", ["CAM_002"]),
        ("show vehicles from cameras 1 and 3", ["CAM_001", "CAM_003"]),
        ("show vehicles from all cameras", []),
    ],
)
def test_video_chat_camera_reference_normalization(message: str, expected: list[str]) -> None:
    repository = _CameraScopeRepository({"RUN_A": ["CAM_001", "CAM_002", "CAM_003"]})
    response = handle_video_chat(
        message=message,
        run_id="RUN_A",
        records=[
            _record("RUN_A", "CAM_001", "TRACK_1", "CAR"),
            _record("RUN_A", "CAM_002", "TRACK_2", "MOTORCYCLE"),
            _record("RUN_A", "CAM_003", "TRACK_3", "CAR"),
        ],
        repository=repository,  # type: ignore[arg-type]
        llm_provider=None,
    )

    assert response["parsed_query"]["include_camera_ids"] == expected


def test_video_chat_arbitrary_camera_and_invalid_camera_scope() -> None:
    repository = _CameraScopeRepository({"RUN_A": ["CAM_001", "CAM_005"]})
    response = handle_video_chat(
        message="show vehicles from camera 5",
        run_id="RUN_A",
        records=[_record("RUN_A", "CAM_005", "TRACK_5")],
        repository=repository,  # type: ignore[arg-type]
        llm_provider=None,
    )

    assert response["parsed_query"]["include_camera_ids"] == ["CAM_005"]
    assert response["analytics_result"]["total"] == 1
    with pytest.raises(VehicleQueryParseError, match="CAM_004 is not present in this run"):
        handle_video_chat(
            message="show vehicles from camera 4",
            run_id="RUN_A",
            records=[_record("RUN_A", "CAM_005", "TRACK_5")],
            repository=repository,  # type: ignore[arg-type]
            llm_provider=None,
        )


@pytest.mark.parametrize(
    ("phrase", "expected"),
    [
        ("2 wheeler", "MOTORCYCLE"),
        ("2-wheeler", "MOTORCYCLE"),
        ("two wheeler", "MOTORCYCLE"),
        ("bike", "MOTORCYCLE"),
        ("motorbike", "MOTORCYCLE"),
        ("3 wheeler", "3WHEELER"),
        ("auto rickshaw", "3WHEELER"),
    ],
)
def test_video_chat_class_aliases_normalize_to_canonical_classes(phrase: str, expected: str) -> None:
    parsed, _ = parse_chat_vehicle_query(message=f"show {phrase}", context={}, llm_provider=None)

    assert parsed.include_classes == [expected]


def test_video_chat_negative_bike_alias_normalizes_to_exclusion() -> None:
    parsed, _ = parse_chat_vehicle_query(message="show everything except bikes", context={}, llm_provider=None)

    assert parsed.exclude_classes == ["MOTORCYCLE"]


def test_video_chat_multi_run_keeps_same_track_id_distinct_and_groups() -> None:
    repository = _CameraScopeRepository({"RUN_A": ["CAM_001", "CAM_002"], "RUN_B": ["CAM_001", "CAM_003"]})
    records = [
        _record("RUN_A", "CAM_001", "TRACK_1", "CAR", "WHITE"),
        _record("RUN_B", "CAM_001", "TRACK_1", "CAR", "WHITE"),
        _record("RUN_B", "CAM_003", "TRACK_7", "MOTORCYCLE", "BLACK"),
    ]

    all_runs = handle_video_chat(
        message="show white cars across all selected runs",
        run_ids=["RUN_A", "RUN_B"],
        records=records,
        repository=repository,  # type: ignore[arg-type]
        llm_provider=None,
    )
    by_camera = handle_video_chat(
        message="count vehicles by camera",
        run_ids=["RUN_A", "RUN_B"],
        records=records,
        repository=repository,  # type: ignore[arg-type]
        llm_provider=None,
    )
    second_run = handle_video_chat(
        message="show vehicles from the second selected run",
        run_ids=["RUN_A", "RUN_B"],
        records=records,
        repository=repository,  # type: ignore[arg-type]
        llm_provider=None,
    )

    assert all_runs["analytics_result"]["total"] == 2
    assert set(all_runs["matching_vehicle_ids"]) == {"RUN_A::CAM_001:TRACK_1", "RUN_B::CAM_001:TRACK_1"}
    assert by_camera["parsed_query"]["group_by"] == "camera"
    assert by_camera["analytics_result"]["by_camera"] == {"CAM_001": 2, "CAM_003": 1}
    assert second_run["parsed_query"]["selected_run_ids"] == ["RUN_B"]
    assert second_run["analytics_result"]["total"] == 2


def test_video_chat_multi_run_camera_scope_allows_camera_where_present() -> None:
    repository = _CameraScopeRepository({"RUN_A": ["CAM_001", "CAM_002"], "RUN_B": ["CAM_001", "CAM_003"]})
    response = handle_video_chat(
        message="show vehicles from camera 3",
        run_ids=["RUN_A", "RUN_B"],
        records=[_record("RUN_A", "CAM_001", "TRACK_1"), _record("RUN_B", "CAM_003", "TRACK_3")],
        repository=repository,  # type: ignore[arg-type]
        llm_provider=None,
    )

    assert response["parsed_query"]["include_camera_ids"] == ["CAM_003"]
    assert response["analytics_result"]["total"] == 1


def test_video_chat_follow_up_preserves_camera_and_run_scope() -> None:
    repository = _CameraScopeRepository({"RUN_A": ["CAM_001", "CAM_002"], "RUN_B": ["CAM_001", "CAM_002"]})
    records = [
        _record("RUN_A", "CAM_002", "TRACK_1", "CAR", "WHITE"),
        _record("RUN_B", "CAM_002", "TRACK_2", "CAR", "BLACK"),
        _record("RUN_B", "CAM_001", "TRACK_3", "CAR", "BLACK"),
    ]
    first = handle_video_chat(
        message="show white cars from camera 2",
        run_ids=["RUN_A", "RUN_B"],
        records=records,
        repository=repository,  # type: ignore[arg-type]
        llm_provider=None,
    )
    follow_up = handle_video_chat(
        message="only the black ones",
        run_ids=["RUN_A", "RUN_B"],
        records=records,
        repository=repository,  # type: ignore[arg-type]
        session_context=first["next_context"],
        llm_provider=None,
    )

    assert follow_up["parsed_query"]["include_camera_ids"] == ["CAM_002"]
    assert follow_up["parsed_query"]["selected_run_ids"] == ["RUN_A", "RUN_B"]


@pytest.mark.parametrize(
    ("message", "expected_group_by", "expected_classes"),
    [
        ("count vehicles camera wise", "camera", []),
        ("count vehicles camera-wise", "camera", []),
        ("count vehicles by camera", "camera", []),
        ("count vehicles per camera", "camera", []),
        ("how many vehicles in each camera", "camera", []),
        ("count 2 wheelers per camera", "camera", ["MOTORCYCLE"]),
        ("count vehicles by run", "run", []),
        ("count vehicles by run and camera", "run_camera", []),
    ],
)
def test_video_chat_scope_grouping_phrases_map_to_multi_camera_grouping(message: str, expected_group_by: str, expected_classes: list[str]) -> None:
    parsed, _ = parse_chat_vehicle_query(message=message, context={}, llm_provider=None)

    assert parsed.intent == "GROUP"
    assert parsed.group_by == expected_group_by
    assert parsed.include_classes == expected_classes


def test_video_chat_follow_up_preserves_grouping_scope_and_filters() -> None:
    repository = _CameraScopeRepository({"RUN_A": ["CAM_001", "CAM_002"], "RUN_B": ["CAM_001", "CAM_002"]})
    records = [
        _record("RUN_A", "CAM_001", "TRACK_1", "CAR", "WHITE"),
        _record("RUN_A", "CAM_002", "TRACK_2", "CAR", "BLACK"),
        _record("RUN_B", "CAM_001", "TRACK_3", "CAR", "BLACK"),
    ]
    first = handle_video_chat(
        message="count cars camera wise",
        run_ids=["RUN_A", "RUN_B"],
        records=records,
        repository=repository,  # type: ignore[arg-type]
        llm_provider=None,
    )
    follow_up = handle_video_chat(
        message="only black",
        run_ids=["RUN_A", "RUN_B"],
        records=records,
        repository=repository,  # type: ignore[arg-type]
        session_context=first["next_context"],
        llm_provider=None,
    )

    assert follow_up["parsed_query"]["intent"] == "GROUP"
    assert follow_up["parsed_query"]["group_by"] == "camera"
    assert follow_up["parsed_query"]["selected_run_ids"] == ["RUN_A", "RUN_B"]
    assert follow_up["parsed_query"]["include_colours"] == ["BLACK"]
    assert follow_up["analytics_result"]["by_camera"] == {"CAM_001": 1, "CAM_002": 1}


def test_video_chat_rejects_llm_that_flattens_camera_group_query() -> None:
    records = [
        _record("RUN_A", "CAM_001", "TRACK_1", "CAR", "WHITE"),
        _record("RUN_A", "CAM_002", "TRACK_2", "MOTORCYCLE", "BLACK"),
    ]
    response = handle_video_chat(
        message="count vehicles camera wise",
        run_ids=["RUN_A"],
        records=records,
        repository=_CameraScopeRepository({"RUN_A": ["CAM_001", "CAM_002"]}),  # type: ignore[arg-type]
        llm_provider=_FakeProvider(_valid_payload(intent="COUNT", class_include=[], colour_include=[], show_evidence=False)),
    )

    assert response["parsed_query"]["intent"] == "GROUP"
    assert response["parsed_query"]["group_by"] == "camera"
    assert response["parser_used"] == "qwen_repaired"
    assert response["llm_rejection_reason"] is None


def test_video_chat_qwen_repairs_camera_ranking_query() -> None:
    parsed, parser_used, diagnostics = parse_chat_vehicle_query_detailed(
        message="which camera out of all have more vehicles",
        context={},
        llm_provider=_FakeProvider(
            _valid_payload(
                intent="LIST",
                class_include=[],
                colour_include=[],
                group_by="run_camera",
                run_filter="multiple_cameras",
                show_evidence=False,
            )
        ),
    )

    assert parser_used == "qwen_repaired"
    assert diagnostics["qwen_raw_plan"]["intent"] == "LIST"
    assert diagnostics["normalized_plan"]["intent"] == "GROUP"
    assert diagnostics["normalized_plan"]["group_by"] == "camera"
    assert diagnostics["normalized_plan"]["sort_by"] == "count_desc"
    assert diagnostics["normalized_plan"]["limit"] == 1
    assert diagnostics["semantic_repair_applied"] is True
    assert parsed.intent == "GROUP"
    assert parsed.group_by == "camera"
    assert parsed.sort_by == "count_desc"
    assert parsed.limit == 1


def test_video_chat_camera_ranking_uses_run_camera_identity_for_multi_run_scope() -> None:
    repository = _CameraScopeRepository({"RUN_A": ["CAM_001", "CAM_002"], "RUN_B": ["CAM_001", "CAM_003"]})
    records = [
        _record("RUN_A", "CAM_001", "TRACK_1", "CAR", "WHITE"),
        _record("RUN_A", "CAM_001", "TRACK_2", "MOTORCYCLE", "BLACK"),
        _record("RUN_B", "CAM_001", "TRACK_3", "CAR", "WHITE"),
        _record("RUN_B", "CAM_003", "TRACK_4", "CAR", "WHITE"),
        _record("RUN_B", "CAM_003", "TRACK_5", "MOTORCYCLE", "BLACK"),
        _record("RUN_B", "CAM_003", "TRACK_6", "BUS", "BLUE"),
    ]
    response = handle_video_chat(
        message="which camera out of all have more vehicles",
        run_ids=["RUN_A", "RUN_B"],
        records=records,
        repository=repository,  # type: ignore[arg-type]
        llm_provider=_FakeProvider(
            _valid_payload(
                intent="LIST",
                class_include=[],
                colour_include=[],
                group_by="run_camera",
                run_filter="multiple_cameras",
                show_evidence=False,
            )
        ),
    )

    assert response["parser_used"] == "qwen_repaired"
    assert response["parsed_query"]["group_by"] == "run_camera"
    assert response["parsed_query"]["sort_by"] == "count_desc"
    assert response["parsed_query"]["limit"] == 1
    assert response["analytics_result"]["ranking_result"]["winners"][0]["run_id"] == "RUN_B"
    assert response["analytics_result"]["ranking_result"]["winners"][0]["camera_id"] == "CAM_003"
    assert response["answer"] == "CAM_003 in run RUN_B has the highest vehicle count with 3 vehicles."


def test_video_chat_camera_ranking_variants_and_filters() -> None:
    repository = _CameraScopeRepository({"RUN_A": ["CAM_001", "CAM_002"], "RUN_B": ["CAM_001", "CAM_003"]})
    records = [
        _record("RUN_A", "CAM_001", "TRACK_1", "CAR", "WHITE"),
        _record("RUN_A", "CAM_002", "TRACK_2", "MOTORCYCLE", "BLACK"),
        _record("RUN_A", "CAM_002", "TRACK_3", "MOTORCYCLE", "BLACK"),
        _record("RUN_B", "CAM_001", "TRACK_4", "CAR", "WHITE"),
        _record("RUN_B", "CAM_003", "TRACK_5", "MOTORCYCLE", "BLACK"),
        _record("RUN_B", "CAM_003", "TRACK_6", "MOTORCYCLE", "BLACK"),
        _record("RUN_B", "CAM_003", "TRACK_7", "CAR", "WHITE"),
        _record("RUN_B", "CAM_003", "TRACK_8", "CAR", "WHITE"),
        _record("RUN_B", "CAM_003", "TRACK_9", "MOTORCYCLE", "BLACK"),
    ]

    busiest = handle_video_chat(
        message="busiest camera",
        run_ids=["RUN_A", "RUN_B"],
        records=records,
        repository=repository,  # type: ignore[arg-type]
        llm_provider=None,
    )
    motorcycles = handle_video_chat(
        message="which camera has the most motorcycles",
        run_ids=["RUN_A", "RUN_B"],
        records=records,
        repository=repository,  # type: ignore[arg-type]
        llm_provider=None,
    )
    white_cars = handle_video_chat(
        message="which camera has the most white cars",
        run_ids=["RUN_A", "RUN_B"],
        records=records,
        repository=repository,  # type: ignore[arg-type]
        llm_provider=None,
    )
    least = handle_video_chat(
        message="which camera has the least vehicles",
        run_ids=["RUN_A", "RUN_B"],
        records=records,
        repository=repository,  # type: ignore[arg-type]
        llm_provider=None,
    )

    assert busiest["answer"] == "CAM_003 in run RUN_B has the highest vehicle count with 5 vehicles."
    assert motorcycles["parsed_query"]["include_classes"] == ["MOTORCYCLE"]
    assert motorcycles["answer"] == "CAM_003 in run RUN_B has the highest vehicle count with 3 vehicles."
    assert white_cars["parsed_query"]["include_classes"] == ["CAR"]
    assert white_cars["parsed_query"]["include_colours"] == ["WHITE"]
    assert white_cars["answer"] == "CAM_003 in run RUN_B has the highest vehicle count with 2 vehicles."
    assert least["parsed_query"]["sort_by"] == "count_asc"
    assert least["answer"] == "CAM_001 in run RUN_A and CAM_001 in run RUN_B are tied for the lowest vehicle count at 1 vehicle each."


def test_video_chat_run_ranking_and_ties() -> None:
    repository = _CameraScopeRepository({"RUN_A": ["CAM_001"], "RUN_B": ["CAM_001", "CAM_002"], "RUN_C": ["CAM_003"]})
    records = [
        _record("RUN_A", "CAM_001", "TRACK_1", "CAR", "WHITE"),
        _record("RUN_A", "CAM_001", "TRACK_2", "CAR", "BLACK"),
        _record("RUN_B", "CAM_001", "TRACK_3", "CAR", "WHITE"),
        _record("RUN_B", "CAM_002", "TRACK_4", "CAR", "WHITE"),
        _record("RUN_C", "CAM_003", "TRACK_5", "MOTORCYCLE", "BLACK"),
    ]
    response = handle_video_chat(
        message="which run has the most vehicles",
        run_ids=["RUN_A", "RUN_B", "RUN_C"],
        records=records,
        repository=repository,  # type: ignore[arg-type]
        llm_provider=None,
    )
    tie_response = handle_video_chat(
        message="top run by vehicle count",
        run_ids=["RUN_A", "RUN_B"],
        records=records[:4],
        repository=repository,  # type: ignore[arg-type]
        llm_provider=None,
    )

    assert response["parsed_query"]["group_by"] == "run"
    assert response["answer"] == "RUN_A and RUN_B are tied for the highest vehicle count at 2 vehicles each."
    assert tie_response["answer"] == "RUN_A and RUN_B are tied for the highest vehicle count at 2 vehicles each."


def test_video_chat_qwen_repairs_vehicle_class_ranking_query() -> None:
    parsed, parser_used, diagnostics = parse_chat_vehicle_query_detailed(
        message="which vehicle class have more vehicles",
        context={},
        llm_provider=_FakeProvider(
            _valid_payload(
                intent="COMPARE",
                class_include=[],
                colour_include=[],
                group_by="vehicle_class",
                show_evidence=False,
                operator=">",
            )
        ),
    )

    assert parser_used == "qwen_repaired"
    assert diagnostics["qwen_raw_plan"]["intent"] == "COMPARE"
    assert diagnostics["normalized_plan"]["intent"] == "GROUP"
    assert diagnostics["normalized_plan"]["group_by"] == "vehicle_class"
    assert diagnostics["normalized_plan"]["sort_by"] == "count_desc"
    assert diagnostics["normalized_plan"]["limit"] == 1
    assert parsed.intent == "GROUP"
    assert parsed.group_by == "class"
    assert parsed.sort_by == "count_desc"


def test_video_chat_class_and_colour_rankings_and_top_k() -> None:
    repository = _CameraScopeRepository({"RUN_A": ["CAM_001"], "RUN_B": ["CAM_002"]})
    records = [
        _record("RUN_A", "CAM_001", "TRACK_1", "CAR", "BLACK"),
        _record("RUN_A", "CAM_001", "TRACK_2", "CAR", "WHITE"),
        _record("RUN_A", "CAM_001", "TRACK_3", "MOTORCYCLE", "BLACK"),
        _record("RUN_A", "CAM_001", "TRACK_4", "BUS", "BLACK"),
        _record("RUN_B", "CAM_002", "TRACK_5", "CAR", "BLACK"),
        _record("RUN_B", "CAM_002", "TRACK_6", "CAR", "WHITE"),
        _record("RUN_B", "CAM_002", "TRACK_7", "TRUCK", "WHITE"),
        _record("RUN_B", "CAM_002", "TRACK_8", "MOTORCYCLE", "BLACK"),
    ]

    top_class = handle_video_chat(
        message="which vehicle class have more vehicles",
        run_ids=["RUN_A", "RUN_B"],
        records=records,
        repository=repository,  # type: ignore[arg-type]
        llm_provider=None,
    )
    most_common = handle_video_chat(
        message="which class is most common",
        run_ids=["RUN_A", "RUN_B"],
        records=records,
        repository=repository,  # type: ignore[arg-type]
        llm_provider=None,
    )
    top_three = handle_video_chat(
        message="top 3 vehicle classes",
        run_ids=["RUN_A", "RUN_B"],
        records=records,
        repository=repository,  # type: ignore[arg-type]
        llm_provider=None,
    )
    top_colour = handle_video_chat(
        message="which colour is most common among cars",
        run_ids=["RUN_A", "RUN_B"],
        records=records,
        repository=repository,  # type: ignore[arg-type]
        llm_provider=None,
    )
    black_class = handle_video_chat(
        message="which vehicle class has the most black vehicles",
        run_ids=["RUN_A", "RUN_B"],
        records=records,
        repository=repository,  # type: ignore[arg-type]
        llm_provider=None,
    )

    assert top_class["parsed_query"]["group_by"] == "class"
    assert top_class["answer"] == "Cars has the highest vehicle count with 4 vehicles."
    assert most_common["answer"] == "Cars has the highest vehicle count with 4 vehicles."
    assert top_three["parsed_query"]["limit"] == 3
    assert "Top 3 vehicle class groups" in top_three["answer"]
    assert "1. Cars - 4" in top_three["answer"]
    assert top_colour["parsed_query"]["include_classes"] == ["CAR"]
    assert top_colour["answer"] == "Black and White are tied for the highest vehicle count at 2 vehicles each."
    assert black_class["parsed_query"]["include_colours"] == ["BLACK"]
    assert black_class["answer"] == "Cars and Motorcycles are tied for the highest vehicle count at 2 vehicles each."


def test_video_chat_generic_comparisons() -> None:
    repository = _CameraScopeRepository({"RUN_A": ["CAM_001", "CAM_002", "CAM_003"]})
    records = [
        _record("RUN_A", "CAM_001", "TRACK_1", "CAR", "BLACK"),
        _record("RUN_A", "CAM_001", "TRACK_2", "CAR", "WHITE"),
        _record("RUN_A", "CAM_002", "TRACK_3", "MOTORCYCLE", "BLACK"),
        _record("RUN_A", "CAM_002", "TRACK_4", "MOTORCYCLE", "BLACK"),
        _record("RUN_A", "CAM_003", "TRACK_5", "CAR", "BLACK"),
        _record("RUN_A", "CAM_003", "TRACK_6", "BUS", "WHITE"),
    ]

    class_compare = handle_video_chat(
        message="are there more cars or motorcycles",
        run_ids=["RUN_A"],
        records=records,
        repository=repository,  # type: ignore[arg-type]
        llm_provider=None,
    )
    class_difference = handle_video_chat(
        message="how many more cars than motorcycles",
        run_ids=["RUN_A"],
        records=records,
        repository=repository,  # type: ignore[arg-type]
        llm_provider=None,
    )
    colour_compare = handle_video_chat(
        message="are black vehicles more common than white vehicles",
        run_ids=["RUN_A"],
        records=records,
        repository=repository,  # type: ignore[arg-type]
        llm_provider=None,
    )
    camera_compare = handle_video_chat(
        message="which camera has more cars, CAM_001 or CAM_003",
        run_ids=["RUN_A"],
        records=records,
        repository=repository,  # type: ignore[arg-type]
        llm_provider=None,
    )

    assert class_compare["parsed_query"]["intent"] == "COMPARE"
    assert class_compare["answer"] == "CAR: 3. MOTORCYCLE: 2. Yes. CAR has more vehicles than MOTORCYCLE."
    assert class_difference["answer"] == "CAR: 3. MOTORCYCLE: 2. CAR has 1 more vehicles than MOTORCYCLE."
    assert colour_compare["answer"] == "Black: 4. White: 2. Yes. Black has more vehicles than White."
    assert camera_compare["answer"] == "CAM_001: 2. CAM_003: 1. Yes. CAM_001 has more vehicles than CAM_003."


def test_video_chat_counts_detected_number_plates() -> None:
    records = [
        VehicleRecord(
            run_id="RUN_A",
            vehicle_id="CAM_001:TRACK_1",
            local_track_id="CAM_001:TRACK_1",
            camera_id="CAM_001",
            vehicle_class="CAR",
            colour="WHITE",
            first_seen_seconds=1.0,
            last_seen_seconds=2.0,
            observation_count=3,
            status="COMPLETED",
            plate_detected=True,
        ),
        VehicleRecord(
            run_id="RUN_A",
            vehicle_id="CAM_001:TRACK_2",
            local_track_id="CAM_001:TRACK_2",
            camera_id="CAM_001",
            vehicle_class="CAR",
            colour="BLACK",
            first_seen_seconds=3.0,
            last_seen_seconds=4.0,
            observation_count=3,
            status="COMPLETED",
            plate_detected=False,
        ),
        VehicleRecord(
            run_id="RUN_A",
            vehicle_id="CAM_001:TRACK_3",
            local_track_id="CAM_001:TRACK_3",
            camera_id="CAM_001",
            vehicle_class="BUS",
            colour="BLUE",
            first_seen_seconds=5.0,
            last_seen_seconds=6.0,
            observation_count=3,
            status="COMPLETED",
            plate_text="DL01AB1234",
        ),
    ]

    response = handle_video_chat(
        message="how many cars with number plate",
        run_ids=["RUN_A"],
        records=records,
        repository=_CameraScopeRepository({"RUN_A": ["CAM_001"]}),  # type: ignore[arg-type]
        llm_provider=None,
    )

    assert response["parsed_query"]["plate_presence"] == "detected"
    assert response["analytics_result"]["total"] == 1
    assert response["answer"] == "There is 1 car with number plates."


def test_vehicle_records_from_physical_vehicles_promotes_valid_plate_evidence_when_consensus_missing() -> None:
    records = vehicle_records_from_physical_vehicles(
        [
            {
                "run_id": "RUN_A",
                "vehicle_id": "VEHICLE_006",
                "primary_camera_id": "CAM_002",
                "vehicle_class": "CAR",
                "vehicle_colour": "WHITE",
                "identity_status": "SINGLE_TRACK",
                "consensus_plate_text": None,
                "member_track_ids": ["CAM_002:TRACK_2"],
                "plate_evidence": [
                    {
                        "local_track_id": "CAM_002:TRACK_2",
                        "normalized_plate_text": "UP84AT5908",
                        "plate_detected": True,
                    }
                ],
                "representative_evidence": [
                    {
                        "local_track_id": "CAM_002:TRACK_2",
                        "plate_text": "UP84AT5908",
                    }
                ],
            }
        ]
    )

    assert len(records) == 1
    assert records[0].vehicle_id == "VEHICLE_006"
    assert records[0].plate_text == "UP84AT5908"
    assert records[0].plate_detected is True


def test_video_chat_plate_search_supports_exact_prefix_suffix_and_contains() -> None:
    repository = _CameraScopeRepository({"RUN_A": ["CAM_001", "CAM_002", "CAM_003"]})
    records = [
        VehicleRecord(run_id="RUN_A", vehicle_id="VEHICLE_006", local_track_id="CAM_002:TRACK_2", camera_id="CAM_002", vehicle_class="CAR", colour="WHITE", first_seen_seconds=1.0, last_seen_seconds=2.0, observation_count=3, status="COMPLETED", plate_text="UP84AT5908", plate_detected=True),
        VehicleRecord(run_id="RUN_A", vehicle_id="VEHICLE_007", local_track_id="CAM_001:TRACK_3", camera_id="CAM_001", vehicle_class="CAR", colour="BLACK", first_seen_seconds=3.0, last_seen_seconds=4.0, observation_count=3, status="COMPLETED", plate_text="HR26DK8337", plate_detected=True),
        VehicleRecord(run_id="RUN_A", vehicle_id="VEHICLE_008", local_track_id="CAM_003:TRACK_4", camera_id="CAM_003", vehicle_class="MOTORCYCLE", colour="RED", first_seen_seconds=5.0, last_seen_seconds=6.0, observation_count=3, status="COMPLETED", plate_text="DL8CAF5062", plate_detected=True),
    ]

    exact = handle_video_chat(message="find UP84AT5908", run_ids=["RUN_A"], records=records, repository=repository)  # type: ignore[arg-type]
    prefix_alias = handle_video_chat(message="find all HR number plates", run_ids=["RUN_A"], records=records, repository=repository)  # type: ignore[arg-type]
    natural_prefix = handle_video_chat(message="find all the vehicles whose number plate have HR", run_ids=["RUN_A"], records=records, repository=repository)  # type: ignore[arg-type]
    natural_prefix_with = handle_video_chat(message="find all vehicles with plate HR", run_ids=["RUN_A"], records=records, repository=repository)  # type: ignore[arg-type]
    prefix = handle_video_chat(message="find plates starting with HR", run_ids=["RUN_A"], records=records, repository=repository)  # type: ignore[arg-type]
    prefix_start_from = handle_video_chat(message="show me the vehicles where the number plate start from HR", run_ids=["RUN_A"], records=records, repository=repository)  # type: ignore[arg-type]
    suffix = handle_video_chat(message="find all number plates ending with 62", run_ids=["RUN_A"], records=records, repository=repository)  # type: ignore[arg-type]
    contains = handle_video_chat(message="find plates containing 84AT", run_ids=["RUN_A"], records=records, repository=repository)  # type: ignore[arg-type]
    contains_digits = handle_video_chat(message="show vehicles whose number plate contains 590", run_ids=["RUN_A"], records=records, repository=repository)  # type: ignore[arg-type]
    count = handle_video_chat(message="how many HR plates are there", run_ids=["RUN_A"], records=records, repository=repository)  # type: ignore[arg-type]

    assert exact["parsed_query"]["plate_match_mode"] == "exact"
    assert exact["matching_vehicle_ids"] == ["VEHICLE_006"]
    assert exact["answer"].startswith("1 vehicle with plate UP84AT5908 was observed.")
    assert prefix_alias["parsed_query"]["plate_match_mode"] == "prefix"
    assert prefix_alias["parsed_query"]["plate_text"] == "HR"
    assert prefix_alias["matching_vehicle_ids"] == ["VEHICLE_007"]
    assert natural_prefix["parsed_query"]["plate_match_mode"] == "prefix"
    assert natural_prefix["parsed_query"]["plate_text"] == "HR"
    assert natural_prefix["matching_vehicle_ids"] == ["VEHICLE_007"]
    assert natural_prefix_with["parsed_query"]["plate_match_mode"] == "prefix"
    assert natural_prefix_with["parsed_query"]["plate_text"] == "HR"
    assert natural_prefix_with["matching_vehicle_ids"] == ["VEHICLE_007"]
    assert prefix["parsed_query"]["plate_match_mode"] == "prefix"
    assert prefix["matching_vehicle_ids"] == ["VEHICLE_007"]
    assert prefix["answer"].startswith("1 vehicle with plates starting with HR was observed.")
    assert prefix_start_from["parsed_query"]["plate_match_mode"] == "prefix"
    assert prefix_start_from["parsed_query"]["plate_text"] == "HR"
    assert prefix_start_from["matching_vehicle_ids"] == ["VEHICLE_007"]
    assert suffix["parsed_query"]["plate_match_mode"] == "suffix"
    assert suffix["matching_vehicle_ids"] == ["VEHICLE_008"]
    assert suffix["answer"].startswith("1 vehicle with plates ending with 62 was observed.")
    assert contains["parsed_query"]["plate_match_mode"] == "contains"
    assert contains["matching_vehicle_ids"] == ["VEHICLE_006"]
    assert contains["answer"].startswith("1 vehicle with plates containing 84AT was observed.")
    assert contains_digits["parsed_query"]["plate_match_mode"] == "contains"
    assert contains_digits["parsed_query"]["plate_text"] == "590"
    assert contains_digits["matching_vehicle_ids"] == ["VEHICLE_006"]
    assert count["parsed_query"]["plate_match_mode"] == "prefix"
    assert count["analytics_result"]["total"] == 1
    assert count["answer"] == "There is 1 vehicle with plates starting with HR."


@pytest.mark.parametrize("message", ["compare CAM_001 or CAM_003", "show TRACK_2", "find VEHICLE_006", "run 20260819_180937"])
def test_video_chat_plate_search_does_not_capture_non_plate_tokens(message: str) -> None:
    assert _parse_plate_text_query(message.lower()) is None


@pytest.mark.parametrize(
    ("value", "match_mode", "expected"),
    [
        ("HR", "prefix", "HR"),
        ("UP84", "prefix", "UP84"),
        ("62", "suffix", "62"),
        ("590", "contains", "590"),
        ("84AT", "contains", "84AT"),
        ("the vehicles whose", "prefix", None),
        ("number plate", "prefix", None),
        ("show vehicles", "contains", None),
        ("plate have", "prefix", None),
    ],
)
def test_validated_plate_search_fragment_rejects_natural_language(value: str, match_mode: str, expected: str | None) -> None:
    from src.video_chat import _validated_plate_search_fragment

    assert _validated_plate_search_fragment(value, match_mode=match_mode) == expected


def test_video_chat_qwen_repairs_missing_plate_match_mode_for_exact_lookup() -> None:
    parsed, parser_used, diagnostics = parse_chat_vehicle_query_detailed(
        message="find UP84AT5908",
        context={},
        llm_provider=_FakeProvider(
            _valid_payload(
                intent="LIST",
                class_include=[],
                colour_include=[],
                plate_presence="readable",
                plate_detected=True,
                plate_readable=True,
                plate_text="UP84AT5908",
                plate_match_mode=None,
                show_evidence=True,
            )
        ),
    )

    assert parser_used == "qwen_repaired"
    assert diagnostics["normalized_plan"]["plate_match_mode"] == "exact"
    assert parsed.plate_text == "UP84AT5908"
    assert parsed.plate_match_mode == "exact"


def test_video_chat_counts_runs_in_current_selection() -> None:
    response = handle_video_chat(
        message="how many runs are there in this search",
        run_ids=["RUN_A", "RUN_B", "RUN_C"],
        records=[_record("RUN_A", "CAM_001", "TRACK_1")],
        repository=_CameraScopeRepository({"RUN_A": ["CAM_001"], "RUN_B": ["CAM_001", "CAM_002"], "RUN_C": ["CAM_003"]}),  # type: ignore[arg-type]
        llm_provider=None,
    )

    assert response["parsed_query"]["subject"] == "runs"
    assert response["analytics_result"]["total"] == 3
    assert response["answer"] == "There are 3 runs in the current selection."


def test_video_chat_lists_runs_with_multiple_cameras() -> None:
    response = handle_video_chat(
        message="which run have multiple cameras",
        run_ids=["RUN_A", "RUN_B", "RUN_C"],
        records=[_record("RUN_A", "CAM_001", "TRACK_1")],
        repository=_CameraScopeRepository({"RUN_A": ["CAM_001"], "RUN_B": ["CAM_001", "CAM_002"], "RUN_C": ["CAM_003", "CAM_004", "CAM_005"]}),  # type: ignore[arg-type]
        llm_provider=None,
    )

    assert response["parsed_query"]["subject"] == "runs"
    assert response["parsed_query"]["run_filter"] == "multiple_cameras"
    assert response["analytics_result"]["run_ids"] == ["RUN_B", "RUN_C"]
    assert "RUN_B (2 cameras)" in response["answer"]
    assert "RUN_C (3 cameras)" in response["answer"]


def test_video_chat_run_scope_prefers_participating_camera_metadata_over_configured_count() -> None:
    class _RunScopeRepository(_CameraScopeRepository):
        def get_run(self, run_id: str) -> dict | None:
            if run_id == "RUN_A":
                return {
                    "run_id": run_id,
                    "camera_count": 1,
                    "summary": {"configured_camera_count": 3, "enabled_camera_count": 1},
                    "metadata": {"camera_count": 1},
                }
            if run_id == "RUN_B":
                return {
                    "run_id": run_id,
                    "camera_count": 2,
                    "summary": {"configured_camera_count": 2, "enabled_camera_count": 2},
                    "metadata": {"camera_count": 2},
                }
            return None

    response = handle_video_chat(
        message="which run have multiple cameras",
        run_ids=["RUN_A", "RUN_B"],
        records=[_record("RUN_A", "CAM_001", "TRACK_1")],
        repository=_RunScopeRepository({"RUN_A": ["CAM_001"], "RUN_B": ["CAM_001", "CAM_002"]}),  # type: ignore[arg-type]
        llm_provider=None,
    )

    assert response["parsed_query"]["subject"] == "runs"
    assert response["analytics_result"]["run_ids"] == ["RUN_B"]
    assert "RUN_B (2 cameras)" in response["answer"]


def test_video_chat_summary_run_and_camera_wise_returns_grouped_summary() -> None:
    records = [
        _record("RUN_A", "CAM_001", "TRACK_1", "CAR", "WHITE"),
        _record("RUN_A", "CAM_002", "TRACK_2", "MOTORCYCLE", "BLACK"),
        _record("RUN_B", "CAM_001", "TRACK_3", "CAR", "WHITE"),
    ]

    response = handle_video_chat(
        message="give me the summary run and camera wise",
        run_ids=["RUN_A", "RUN_B"],
        records=records,
        repository=_CameraScopeRepository({"RUN_A": ["CAM_001", "CAM_002"], "RUN_B": ["CAM_001"]}),  # type: ignore[arg-type]
        llm_provider=None,
    )

    assert response["parsed_query"]["intent"] == "SUMMARY"
    assert response["parsed_query"]["group_by"] == "run_camera"
    assert response["analytics_result"]["groups"]["RUN_A / CAM_001"]["total_unique_vehicles"] == 1
    assert response["analytics_result"]["groups"]["RUN_A / CAM_002"]["total_unique_vehicles"] == 1
    assert response["analytics_result"]["groups"]["RUN_B / CAM_001"]["total_unique_vehicles"] == 1


def test_semantic_query_evaluation_fixture_final_plans() -> None:
    fixtures = json.loads(Path("tests/fixtures/semantic_query_evaluation.json").read_text(encoding="utf-8"))
    assert len(fixtures) >= 50
    context = {
        "previous_filters": {"include_classes": ["MOTORCYCLE"], "exclude_classes": [], "include_colours": [], "exclude_colours": []},
        "previous_vehicle_ids": ["CAM_001:TRACK_6"],
    }
    incorrect: list[str] = []
    for item in fixtures:
        parsed, _ = parse_chat_vehicle_query(
            message=item["text"],
            context=context if item.get("context") else {},
            llm_provider=None,
        )
        plan = parsed.to_dict()
        for key, expected_value in dict(item["expected"]).items():
            if plan.get(key) != expected_value:
                incorrect.append(f"{item['text']} expected {key}={expected_value!r}, got {plan.get(key)!r}")
    assert incorrect == []


def test_ollama_provider_posts_schema_think_false_and_ignores_thinking(monkeypatch) -> None:
    captured = {}

    def fake_urlopen(request, timeout):
        captured["timeout"] = timeout
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        return _FakeHTTPResponse(
            {
                "message": {
                    "thinking": "hidden chain of thought",
                    "content": json.dumps(
                        _analytics_payload(
                            filters={"kind": "condition", "condition": {"field": "class", "operator": "in", "value": ["CAR"]}},
                            result_shape="scalar",
                            show_evidence=False,
                        )
                    ),
                }
            }
        )

    monkeypatch.setattr("src.ollama_qwen_provider.urlopen", fake_urlopen)
    provider = OllamaQwenChatLLMProvider(base_url="http://127.0.0.1:11434", model="qwen3:1.7b", timeout_seconds=3)

    parsed = provider.parse("How many cars?", {"previous_filters": {}})

    assert parsed["entity"] == "vehicle"
    assert captured["payload"]["model"] == "qwen3:1.7b"
    assert captured["payload"]["stream"] is False
    assert captured["payload"]["think"] is False
    assert captured["payload"]["format"]["type"] == "object"
    assert "entity" in captured["payload"]["format"]["required"]
    assert "filters" not in captured["payload"]["format"]["required"]
    assert "class_include" not in captured["payload"]["format"]["properties"]
    assert captured["payload"]["options"]["temperature"] == DEFAULT_OLLAMA_TEMPERATURE
    assert captured["payload"]["options"]["num_predict"] == DEFAULT_OLLAMA_NUM_PREDICT
    assert captured["payload"]["keep_alive"] == "10m"
    assert "hidden chain of thought" not in json.dumps(parsed)
    assert provider.last_metadata["model"] == "qwen3:1.7b"
    assert provider.last_metadata["num_predict"] == DEFAULT_OLLAMA_NUM_PREDICT
    assert provider.last_metadata["temperature"] == DEFAULT_OLLAMA_TEMPERATURE
    assert provider.last_metadata["structured_output"] is True


def test_ollama_provider_default_timeout_allows_local_qwen_warmup() -> None:
    provider = OllamaQwenChatLLMProvider()

    assert provider.timeout_seconds == 45.0
    assert provider.model == "qwen3:1.7b"
    assert provider.num_predict == DEFAULT_OLLAMA_NUM_PREDICT
    assert provider.temperature == DEFAULT_OLLAMA_TEMPERATURE


def test_build_chat_llm_provider_from_env_prefers_explicit_qwen_settings(monkeypatch) -> None:
    monkeypatch.setenv("VIDEO_CHAT_LLM_PROVIDER", "ollama")
    monkeypatch.setenv("VIDEO_CHAT_OLLAMA_MODEL", "qwen3:4b")
    monkeypatch.setenv("VIDEO_CHAT_QWEN_MODEL", "qwen3:1.7b")
    monkeypatch.setenv("VIDEO_CHAT_QWEN_KEEP_ALIVE", "5m")
    monkeypatch.setenv("VIDEO_CHAT_QWEN_TIMEOUT_SECONDS", "30")
    monkeypatch.setenv("VIDEO_CHAT_QWEN_NUM_PREDICT", "88")
    monkeypatch.setenv("VIDEO_CHAT_QWEN_TEMPERATURE", "0")

    provider = build_chat_llm_provider_from_env()

    assert provider is not None
    assert provider.model == "qwen3:1.7b"
    assert provider.keep_alive == "5m"
    assert provider.timeout_seconds == 30.0
    assert provider.num_predict == 88
    assert provider.temperature == 0.0


def test_ollama_provider_malformed_json_and_unavailable_raise_runtime_error(monkeypatch) -> None:
    def bad_json_urlopen(request, timeout):
        return _FakeHTTPResponse({"message": {"content": "{not-json"}})

    monkeypatch.setattr("src.ollama_qwen_provider.urlopen", bad_json_urlopen)
    provider = OllamaQwenChatLLMProvider()
    with pytest.raises(RuntimeError, match="reason=qwen_invalid_json"):
        provider.parse("message", {})

    def fenced_json_urlopen(request, timeout):
        return _FakeHTTPResponse({"message": {"content": "```json\n{\"entity\":\"vehicle\"}\n```"}, "done_reason": "stop"})

    monkeypatch.setattr("src.ollama_qwen_provider.urlopen", fenced_json_urlopen)
    assert provider.parse("message", {}) == {"entity": "vehicle"}

    def dict_content_urlopen(request, timeout):
        return _FakeHTTPResponse({"message": {"content": {"entity": "vehicle"}}, "done_reason": "stop"})

    monkeypatch.setattr("src.ollama_qwen_provider.urlopen", dict_content_urlopen)
    assert provider.parse("message", {}) == {"entity": "vehicle"}

    def truncated_json_urlopen(request, timeout):
        return _FakeHTTPResponse({"message": {"content": "{\"entity\":\"vehicle\","}, "done_reason": "length", "eval_count": 128})

    monkeypatch.setattr("src.ollama_qwen_provider.urlopen", truncated_json_urlopen)
    with pytest.raises(RuntimeError, match="reason=qwen_output_truncated"):
        provider.parse("message", {})

    def empty_content_urlopen(request, timeout):
        return _FakeHTTPResponse({"message": {"content": ""}, "done_reason": "stop"})

    monkeypatch.setattr("src.ollama_qwen_provider.urlopen", empty_content_urlopen)
    with pytest.raises(RuntimeError, match="reason=qwen_empty_content"):
        provider.parse("message", {})

    def unavailable_urlopen(request, timeout):
        raise URLError("connection refused")

    monkeypatch.setattr("src.ollama_qwen_provider.urlopen", unavailable_urlopen)
    with pytest.raises(RuntimeError, match="provider=ollama.*model=qwen3:1.7b.*timeout_seconds=45.*exception=URLError"):
        provider.parse("message", {})
