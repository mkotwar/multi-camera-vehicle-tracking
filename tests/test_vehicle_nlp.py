from __future__ import annotations

import pytest

from src.vehicle_analytics import vehicle_records_from_tracks
from src.vehicle_nlp import (
    VehicleQueryParseError,
    execute_vehicle_query,
    parse_time_range,
    parse_time_value,
    parse_vehicle_query,
    search_vehicle_data,
)


def _track(
    track_id: str,
    *,
    vehicle_class: str = "car",
    colour: str = "BLACK",
    first_seen: float = 0.0,
    last_seen: float = 1.0,
) -> dict:
    return {
        "local_track_id": f"CAM_001:{track_id}",
        "camera_id": "CAM_001",
        "status": "COMPLETED",
        "final_class": vehicle_class,
        "first_timestamp_seconds": first_seen,
        "last_timestamp_seconds": last_seen,
        "observation_count": 5,
        "vehicle_enrichment": {"vehicle_colour": {"label": colour, "status": "completed"}},
    }


def _records():
    return vehicle_records_from_tracks(
        [
            _track("TRACK_1", vehicle_class="car", colour="WHITE", first_seen=0.0, last_seen=8.0),
            _track("TRACK_2", vehicle_class="motorcycle", colour="BLACK", first_seen=5.0, last_seen=10.0),
            _track("TRACK_3", vehicle_class="3wheeler", colour="GREEN", first_seen=20.0, last_seen=25.0),
            _track("TRACK_4", vehicle_class="unknown", colour="UNKNOWN", first_seen=30.0, last_seen=35.0),
        ]
    )


def test_count_intent_parses_class_query() -> None:
    parsed = parse_vehicle_query("How many cars are there?")

    assert parsed.intent == "COUNT"
    assert parsed.vehicle_class == "CAR"
    assert parsed.colour is None


def test_list_intent_parses_class_colour_time_query() -> None:
    parsed = parse_vehicle_query("Show white cars between 5 and 10 seconds.")

    assert parsed.intent == "LIST"
    assert parsed.vehicle_class == "CAR"
    assert parsed.colour == "WHITE"
    assert parsed.start_time == 5.0
    assert parsed.end_time == 10.0


def test_unique_classes_and_colours_intents() -> None:
    assert parse_vehicle_query("What vehicle types are present?").intent == "UNIQUE_CLASSES"
    assert parse_vehicle_query("What colours are present?").intent == "UNIQUE_COLOURS"


@pytest.mark.parametrize("term", ["bike", "motorbike", "motorcycle"])
def test_motorcycle_synonyms(term: str) -> None:
    parsed = parse_vehicle_query(f"How many {term}s are there?")

    assert parsed.vehicle_class == "MOTORCYCLE"


@pytest.mark.parametrize("term", ["auto", "rickshaw", "auto-rickshaw", "three wheeler"])
def test_three_wheeler_synonyms(term: str) -> None:
    parsed = parse_vehicle_query(f"Show {term}s")

    assert parsed.vehicle_class == "3WHEELER"


def test_unknown_vehicle_class_synonym() -> None:
    parsed = parse_vehicle_query("Show unknown vehicles")

    assert parsed.intent == "LIST"
    assert parsed.vehicle_class == "UNKNOWN"


@pytest.mark.parametrize("term", ["gray", "grey"])
def test_grey_synonyms(term: str) -> None:
    parsed = parse_vehicle_query(f"How many {term} vehicles are there?")

    assert parsed.colour == "GREY"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("5 seconds", 5.0),
        ("10 seconds", 10.0),
        ("1 minute", 60.0),
        ("1:30", 90.0),
        ("00:05", 5.0),
        ("00:10", 10.0),
    ],
)
def test_time_value_parsing(raw: str, expected: float) -> None:
    assert parse_time_value(raw) == expected


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("between 5 and 10 seconds", (5.0, 10.0)),
        ("from 00:05 to 00:10", (5.0, 10.0)),
        ("in the first 10 seconds", (0.0, 10.0)),
        ("in the first minute", (0.0, 60.0)),
        ("after 5 seconds", (5.0, None)),
        ("before 10 seconds", (None, 10.0)),
        ("from 1:00 to 1:30", (60.0, 90.0)),
    ],
)
def test_time_range_parsing(query: str, expected: tuple[float | None, float | None]) -> None:
    assert parse_time_range(query) == expected


def test_invalid_time_range_fails_validation() -> None:
    with pytest.raises(VehicleQueryParseError):
        parse_vehicle_query("Show vehicles between 10 and 5 seconds")


def test_ambiguous_query_requires_clarification() -> None:
    with pytest.raises(VehicleQueryParseError):
        parse_vehicle_query("show vehicles around 5")


def test_unknown_vehicle_class_fails() -> None:
    with pytest.raises(VehicleQueryParseError):
        parse_vehicle_query("How many vans are there?")


def test_unknown_colour_fails() -> None:
    with pytest.raises(VehicleQueryParseError):
        parse_vehicle_query("Show teal vehicles")


def test_dark_vehicle_query_does_not_silently_become_black() -> None:
    with pytest.raises(VehicleQueryParseError):
        parse_vehicle_query("Show dark vehicles")


def test_executor_uses_vehicle_analytics_for_count() -> None:
    parsed = parse_vehicle_query("How many white cars are there?")
    result = execute_vehicle_query(_records(), parsed)

    assert result["total"] == 1
    assert result["vehicle_ids"] == ["CAM_001:TRACK_1"]


def test_executor_uses_vehicle_analytics_for_time_filtered_list() -> None:
    parsed = parse_vehicle_query("Show black motorcycles between 5 and 10 seconds")
    result = execute_vehicle_query(_records(), parsed)

    assert result["total"] == 1
    assert result["vehicle_ids"] == ["CAM_001:TRACK_2"]


def test_unique_class_and_colour_execution() -> None:
    classes = execute_vehicle_query(_records(), parse_vehicle_query("What types of vehicles are present?"))
    colours = execute_vehicle_query(_records(), parse_vehicle_query("What colours are present?"))

    assert classes["vehicle_classes_present"] == ["3WHEELER", "CAR", "MOTORCYCLE", "UNKNOWN"]
    assert colours["colours_present"] == ["BLACK", "WHITE", "GREEN", "UNKNOWN"]


def test_traceable_search_result(tmp_path) -> None:
    tracks_path = tmp_path / "tracks.json"
    tracks_path.write_text(
        __import__("json").dumps([_track("TRACK_1", vehicle_class="car", colour="WHITE")]),
        encoding="utf-8",
    )

    result = search_vehicle_data("How many white cars are there?", tracks_path)

    assert result["original_query"] == "How many white cars are there?"
    assert result["parsed_query"]["vehicle_class"] == "CAR"
    assert result["analytics_result"]["total"] == 1
    assert "1 white car" in result["response"]
