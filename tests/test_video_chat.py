from __future__ import annotations

import json
from pathlib import Path
from urllib.error import URLError

import pytest

from src.ollama_qwen_provider import OllamaQwenChatLLMProvider
from src.run_repository import RunRepository
from src.vehicle_analytics import load_vehicle_records_from_tracks_json
from src.vehicle_nlp import VehicleQueryParseError
from src.video_chat import execute_chat_vehicle_query, handle_video_chat, parse_chat_vehicle_query, parse_chat_vehicle_query_detailed


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
                "first_seen_seconds": 2.0,
                "last_seen_seconds": 8.0,
                "best_crop_parts": {
                    "category": "florence_selected_crops",
                    "run_id": run_id,
                    "parts": ["CAM_001", "TRACK_22", "frame_000022_MIDDLE.jpg"],
                },
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
        "classes": ["CAR"],
        "exclude_classes": [],
        "colours": ["WHITE"],
        "exclude_colours": [],
        "start_time": None,
        "end_time": None,
        "group_by": None,
        "operator": None,
        "show_evidence": True,
        "context_reference": None,
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
    assert diagnostics["normalized_llm_output"] == _valid_payload()
    assert parsed.intent == "LIST"
    assert parsed.include_classes == ["CAR"]
    assert parsed.include_colours == ["WHITE"]
    assert parsed.show_evidence is True


def test_video_chat_qwen_multi_class_synonyms_exclusion_group_summary_evidence_and_context() -> None:
    cases = [
        (_valid_payload(intent="COUNT", classes=["CAR", "MOTORCYCLE"], show_evidence=False), ["CAR", "MOTORCYCLE"], []),
        (_valid_payload(intent="GROUP", classes=[], exclude_classes=["MOTORCYCLE"], group_by="vehicle_class", show_evidence=False), [], ["MOTORCYCLE"]),
        (_valid_payload(intent="SUMMARY", classes=[], colours=[], show_evidence=False), [], []),
        (_valid_payload(intent="LIST", classes=["3WHEELER"], colours=["GREEN"], show_evidence=True), ["3WHEELER"], []),
        (_valid_payload(intent="LIST", classes=[], colours=["WHITE"], context_reference="previous_result", show_evidence=True), [], []),
    ]
    for payload, expected_classes, expected_exclusions in cases:
        context = {"previous_vehicle_ids": ["CAM_001:TRACK_1"]} if payload.get("context_reference") == "previous_result" else {}
        message = "Which of those were white?" if payload.get("context_reference") == "previous_result" else "semantic query"
        parsed, parser_used = parse_chat_vehicle_query(message=message, context=context, llm_provider=_FakeProvider(payload))
        assert parser_used == "qwen"
        assert parsed.include_classes == expected_classes
        assert parsed.exclude_classes == expected_exclusions


def test_video_chat_qwen_receives_small_structured_context() -> None:
    provider = _FakeProvider(_valid_payload(intent="COUNT", classes=["MOTORCYCLE"], colours=["BLACK"], show_evidence=False, context_reference="previous_result"))
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

    assert parser_used == "rule_based"
    assert parsed.include_classes == ["CAR"]
    assert diagnostics["llm_attempted"] is True
    assert diagnostics["llm_accepted"] is False
    assert diagnostics["llm_rejection_reason"] == "unsupported_class:SPACESHIP"


def test_video_chat_qwen_unavailable_and_timeout_fall_back_to_rule_parser() -> None:
    for error in [RuntimeError("Ollama unavailable"), TimeoutError("timed out")]:
        parsed, parser_used = parse_chat_vehicle_query(
            message="How many motorcycles were there?",
            context={},
            llm_provider=_FakeProvider(error=error),
        )
        assert parser_used == "rule_based"
        assert parsed.include_classes == ["MOTORCYCLE"]


def test_video_chat_qwen_invalid_schema_without_rule_fallback_returns_rule_error() -> None:
    with pytest.raises(VehicleQueryParseError, match="Could not determine chat query intent"):
        parse_chat_vehicle_query(
            message="please locate passenger vehicles that look pale",
            context={},
            llm_provider=_FakeProvider({"intent": "COUNT", "classes": ["SPACESHIP"]}),
        )


@pytest.mark.parametrize("message", ["hello", "hi", "thanks"])
def test_video_chat_general_chat_bypasses_qwen_and_has_no_filters(message: str) -> None:
    provider = _FakeProvider(_valid_payload(intent="COUNT", classes=[], colours=[], show_evidence=False))
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
        llm_provider=_FakeProvider(_valid_payload(intent="GROUP", classes=["MOTORCYCLE"], colours=["BLACK"], group_by="vehicle_class", show_evidence=False, context_reference="previous_result")),
    )

    assert parser_used == "rule_based"
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
        llm_provider=_FakeProvider(_valid_payload(intent="UNIQUE_COLOURS", classes=[], colours=[], show_evidence=False)),
    )

    assert parser_used == "qwen"
    assert diagnostics["semantic_repair_applied"] is True
    assert parsed.intent == "GROUP"
    assert parsed.group_by == "colour"
    assert parsed.include_classes == ["MOTORCYCLE"]


def test_video_chat_qwen_group_class_must_preserve_explicit_colour() -> None:
    parsed, parser_used, diagnostics = parse_chat_vehicle_query_detailed(
        message="what vehicle classes were black?",
        context={},
        llm_provider=_FakeProvider(_valid_payload(intent="GROUP", classes=[], colours=[], group_by="vehicle_class", show_evidence=False)),
    )

    assert parser_used == "qwen"
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


@pytest.mark.parametrize("message", ["unknown vehicle", "Show unknown vehicles"])
def test_video_chat_unknown_vehicle_is_explicit_class_filter(message: str) -> None:
    parsed, parser_used, diagnostics = parse_chat_vehicle_query_detailed(
        message=message,
        context={},
        llm_provider=_FakeProvider(_valid_payload(intent="LIST", classes=[], colours=[], show_evidence=True)),
    )

    assert parser_used == "qwen"
    assert diagnostics["semantic_repair_applied"] is True
    assert parsed.intent == "LIST"
    assert parsed.include_classes == ["UNKNOWN"]
    assert parsed.include_colours == []
    assert parsed.show_evidence is True


def test_video_chat_qwen_unknown_vehicle_payload_is_accepted() -> None:
    parsed, parser_used = parse_chat_vehicle_query(
        message="Show unknown vehicles",
        context={},
        llm_provider=_FakeProvider(_valid_payload(intent="LIST", classes=["UNKNOWN"], colours=[], show_evidence=True)),
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
                classes=["AUTO"],
                colours=["GRAY"],
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
        llm_provider=_FakeProvider(_valid_payload(intent="LIST", classes=["MOTORCYCLE"], exclude_classes=[], colours=["BLACK"], show_evidence=True)),
    )

    assert parser_used == "qwen"
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
    assert response["evidence"][0]["best_crop_url"] == f"/api/media/florence_selected_crops/{run_id}/CAM_001/TRACK_22/frame_000022_MIDDLE.jpg"
    assert response["evidence"][0]["image_url"] == response["evidence"][0]["best_crop_url"]


def test_video_chat_accepts_common_summary_typo() -> None:
    parsed, parser_used = parse_chat_vehicle_query(message="summry of the video", context={}, llm_provider=None)

    assert parser_used == "rule_based"
    assert parsed.intent == "SUMMARY"


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
                    "content": json.dumps(_valid_payload(intent="COUNT", show_evidence=False)),
                }
            }
        )

    monkeypatch.setattr("src.ollama_qwen_provider.urlopen", fake_urlopen)
    provider = OllamaQwenChatLLMProvider(base_url="http://127.0.0.1:11434", model="qwen3:1.7b", timeout_seconds=3)

    parsed = provider.parse("How many cars?", {"previous_filters": {}})

    assert parsed["intent"] == "COUNT"
    assert captured["payload"]["model"] == "qwen3:1.7b"
    assert captured["payload"]["stream"] is False
    assert captured["payload"]["think"] is False
    assert captured["payload"]["format"]["type"] == "object"
    assert "class_include" in captured["payload"]["format"]["required"]
    assert "classes" not in captured["payload"]["format"]["properties"]
    assert captured["payload"]["options"]["temperature"] == 0
    assert captured["payload"]["keep_alive"] == "10m"
    assert "hidden chain of thought" not in json.dumps(parsed)


def test_ollama_provider_default_timeout_allows_local_qwen_warmup() -> None:
    provider = OllamaQwenChatLLMProvider()

    assert provider.timeout_seconds == 45.0


def test_ollama_provider_malformed_json_and_unavailable_raise_runtime_error(monkeypatch) -> None:
    def bad_json_urlopen(request, timeout):
        return _FakeHTTPResponse({"message": {"content": "{not-json"}})

    monkeypatch.setattr("src.ollama_qwen_provider.urlopen", bad_json_urlopen)
    provider = OllamaQwenChatLLMProvider()
    with pytest.raises(RuntimeError, match="not valid JSON"):
        provider.parse("message", {})

    def unavailable_urlopen(request, timeout):
        raise URLError("connection refused")

    monkeypatch.setattr("src.ollama_qwen_provider.urlopen", unavailable_urlopen)
    with pytest.raises(RuntimeError, match="provider=ollama.*model=qwen3:1.7b.*timeout_seconds=45.*exception=URLError"):
        provider.parse("message", {})
