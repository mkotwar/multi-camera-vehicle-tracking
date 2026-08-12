from __future__ import annotations

from src.vehicle_analytics import (
    VehicleRecord,
    build_vehicle_analytics,
    filter_vehicles,
    get_vehicle_statistics,
    vehicle_records_from_tracks,
)


def _track(
    track_id: str,
    *,
    status: str = "COMPLETED",
    vehicle_class: str = "car",
    colour: str = "BLACK",
    first_seen: float = 0.0,
    last_seen: float = 1.0,
    observation_count: int = 5,
) -> dict:
    return {
        "local_track_id": f"CAM_001:{track_id}",
        "camera_id": "CAM_001",
        "status": status,
        "final_class": vehicle_class,
        "first_timestamp_seconds": first_seen,
        "last_timestamp_seconds": last_seen,
        "observation_count": observation_count,
        "vehicle_enrichment": {
            "vehicle_colour": {
                "label": colour,
                "status": "completed" if colour != "UNKNOWN" else "skipped",
            }
        },
    }


def _records() -> list[VehicleRecord]:
    return vehicle_records_from_tracks(
        [
            _track("TRACK_1", vehicle_class="car", colour="BLACK", first_seen=0.0, last_seen=4.0),
            _track("TRACK_2", vehicle_class="motorcycle", colour="BLACK", first_seen=4.0, last_seen=8.0),
            _track("TRACK_3", vehicle_class="car", colour="WHITE", first_seen=8.0, last_seen=12.0),
            _track("TRACK_4", vehicle_class="3wheeler", colour="GREEN", first_seen=20.0, last_seen=25.0),
            _track("TRACK_5", vehicle_class="unknown", colour="UNKNOWN", first_seen=1.0, last_seen=2.0),
            _track("TRACK_6", status="DISCARDED", vehicle_class="car", colour="RED", first_seen=0.0, last_seen=30.0),
        ]
    )


def test_completed_tracks_are_counted_and_discarded_tracks_are_excluded() -> None:
    records = _records()
    analytics = build_vehicle_analytics(records)

    assert analytics["total_unique_vehicles"] == 5
    assert analytics["vehicle_classes"]["CAR"] == 2
    assert analytics["vehicle_classes"]["MOTORCYCLE"] == 1
    assert analytics["vehicle_classes"]["3WHEELER"] == 1
    assert analytics["vehicle_classes"]["UNKNOWN"] == 1
    assert "CAM_001:TRACK_6" not in analytics["vehicle_ids"]


def test_class_filtering() -> None:
    matches = filter_vehicles(_records(), vehicle_class="car")

    assert [item.vehicle_id for item in matches] == ["CAM_001:TRACK_1", "CAM_001:TRACK_3"]


def test_colour_filtering() -> None:
    matches = filter_vehicles(_records(), colour="black")

    assert [item.vehicle_id for item in matches] == ["CAM_001:TRACK_1", "CAM_001:TRACK_2"]


def test_time_overlap_filtering() -> None:
    matches = filter_vehicles(_records(), start_time=5.0, end_time=10.0)

    assert [item.vehicle_id for item in matches] == ["CAM_001:TRACK_2", "CAM_001:TRACK_3"]


def test_combined_class_and_colour_filtering() -> None:
    matches = filter_vehicles(_records(), vehicle_class="car", colour="white")

    assert [item.vehicle_id for item in matches] == ["CAM_001:TRACK_3"]


def test_general_include_exclude_filtering() -> None:
    matches = filter_vehicles(
        _records(),
        include_colours=["BLACK"],
        exclude_classes=["MOTORCYCLE"],
    )

    assert [item.vehicle_id for item in matches] == ["CAM_001:TRACK_1"]


def test_statistics_uses_general_include_exclude_filtering() -> None:
    result = get_vehicle_statistics(_records(), exclude_classes=["CAR", "MOTORCYCLE"])

    assert result["total"] == 2
    assert result["by_class"]["3WHEELER"] == 1
    assert result["by_class"]["UNKNOWN"] == 1
    assert result["vehicle_ids"] == ["CAM_001:TRACK_4", "CAM_001:TRACK_5"]


def test_combined_class_and_time_filtering() -> None:
    matches = filter_vehicles(_records(), vehicle_class="car", start_time=5.0, end_time=10.0)

    assert [item.vehicle_id for item in matches] == ["CAM_001:TRACK_3"]


def test_class_colour_and_time_filtering() -> None:
    matches = filter_vehicles(_records(), vehicle_class="motorcycle", colour="black", start_time=5.0, end_time=10.0)

    assert [item.vehicle_id for item in matches] == ["CAM_001:TRACK_2"]


def test_zero_result_query() -> None:
    matches = filter_vehicles(_records(), vehicle_class="bus", colour="yellow")

    assert matches == []


def test_boundary_last_seen_equal_query_start_counts_as_overlapping() -> None:
    matches = filter_vehicles(_records(), start_time=4.0, end_time=4.0)

    assert [item.vehicle_id for item in matches] == ["CAM_001:TRACK_1", "CAM_001:TRACK_2"]


def test_boundary_first_seen_equal_query_end_counts_as_overlapping() -> None:
    matches = filter_vehicles(_records(), start_time=12.0, end_time=20.0)

    assert [item.vehicle_id for item in matches] == ["CAM_001:TRACK_3", "CAM_001:TRACK_4"]


def test_unknown_class_and_colour_handling() -> None:
    matches = filter_vehicles(_records(), vehicle_class="unknown", colour="unknown")

    assert [item.vehicle_id for item in matches] == ["CAM_001:TRACK_5"]


def test_statistics_returns_breakdowns_and_matching_ids() -> None:
    result = get_vehicle_statistics(_records(), start_time=5.0, end_time=10.0)

    assert result["total"] == 2
    assert result["by_class"]["CAR"] == 1
    assert result["by_class"]["MOTORCYCLE"] == 1
    assert result["by_colour"]["BLACK"] == 1
    assert result["by_colour"]["WHITE"] == 1
    assert result["vehicle_ids"] == ["CAM_001:TRACK_2", "CAM_001:TRACK_3"]
