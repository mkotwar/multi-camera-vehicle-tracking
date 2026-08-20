from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

from .indian_plate_validator import validate_indian_plate
from .plate_text import normalize_plate_text
from .vehicle_enrichment.taxonomy import SUPPORTED_VEHICLE_CLASSES, SUPPORTED_VEHICLE_COLOUR_LABELS
from .vehicle_enrichment.schemas import VEHICLE_COLOUR_UNKNOWN


UNKNOWN_CLASS = "UNKNOWN"
CLASS_COUNT_KEYS: tuple[str, ...] = (*SUPPORTED_VEHICLE_CLASSES, UNKNOWN_CLASS)
COLOUR_COUNT_KEYS: tuple[str, ...] = SUPPORTED_VEHICLE_COLOUR_LABELS


@dataclass(frozen=True, slots=True)
class VehicleRecord:
    vehicle_id: str
    local_track_id: str
    camera_id: str
    vehicle_class: str
    colour: str
    first_seen_seconds: float | None
    last_seen_seconds: float | None
    observation_count: int
    status: str
    member_track_ids: tuple[str, ...] = ()
    plate_text: str | None = None
    plate_detected: bool | None = None
    plate_detection_confidence: float | None = None
    plate_text_confidence: float | None = None
    plate_quality_status: str | None = None
    plate_ocr_reason: str | None = None
    run_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_vehicle_records_from_tracks_json(path: str | Path) -> list[VehicleRecord]:
    tracks_path = Path(path)
    payload = json.loads(tracks_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("tracks.json must contain a list of tracks.")
    return vehicle_records_from_tracks(payload)


def vehicle_records_from_tracks(tracks: list[dict[str, Any]]) -> list[VehicleRecord]:
    records: list[VehicleRecord] = []
    seen_completed_ids: set[str] = set()
    for track in tracks:
        if not isinstance(track, dict):
            continue
        if str(track.get("status", "")).strip().upper() != "COMPLETED":
            continue
        local_track_id = str(track.get("local_track_id") or "").strip()
        if not local_track_id:
            continue
        run_id = str(track.get("run_id") or "").strip() or None
        unique_key = f"{run_id or ''}\0{local_track_id}"
        if unique_key in seen_completed_ids:
            raise ValueError(f"Duplicate completed local_track_id: {local_track_id}")
        seen_completed_ids.add(unique_key)
        records.append(
            VehicleRecord(
                vehicle_id=local_track_id,
                local_track_id=local_track_id,
                camera_id=str(track.get("camera_id") or "").strip(),
                vehicle_class=_normalize_vehicle_class(track.get("final_class")),
                colour=_extract_track_colour(track),
                first_seen_seconds=_coerce_float(track.get("first_timestamp_seconds")),
                last_seen_seconds=_coerce_float(track.get("last_timestamp_seconds")),
                observation_count=_coerce_int(track.get("observation_count")),
                status="COMPLETED",
                plate_text=_extract_track_plate_text(track),
                plate_detected=_coerce_bool(_extract_track_enrichment_value(track, "plate_detected")),
                plate_detection_confidence=_coerce_float(_extract_track_enrichment_value(track, "plate_detection_confidence")),
                plate_text_confidence=_coerce_float(_extract_track_enrichment_value(track, "plate_text_confidence")),
                plate_quality_status=_coerce_str_or_none(_extract_track_enrichment_value(track, "plate_quality_status")),
                plate_ocr_reason=_coerce_str_or_none(_extract_track_enrichment_value(track, "plate_ocr_reason")),
                run_id=run_id,
            )
        )
    return records


def vehicle_records_from_repository_tracks(tracks: list[dict[str, Any]]) -> list[VehicleRecord]:
    records: list[VehicleRecord] = []
    seen_completed_ids: set[str] = set()
    for track in tracks:
        if not isinstance(track, dict):
            continue
        if str(track.get("status") or "").strip().upper() != "COMPLETED":
            continue
        local_track_id = str(track.get("local_track_id") or "").strip()
        if not local_track_id:
            continue
        run_id = str(track.get("run_id") or "").strip() or None
        unique_key = f"{run_id or ''}\0{local_track_id}"
        if unique_key in seen_completed_ids:
            raise ValueError(f"Duplicate completed local_track_id: {local_track_id}")
        seen_completed_ids.add(unique_key)
        records.append(
            VehicleRecord(
                vehicle_id=local_track_id,
                local_track_id=local_track_id,
                camera_id=str(track.get("camera_id") or "").strip(),
                vehicle_class=_normalize_vehicle_class(track.get("vehicle_class")),
                colour=_normalize_colour(track.get("colour")),
                first_seen_seconds=_coerce_float(track.get("first_seen_seconds") or track.get("first_seen")),
                last_seen_seconds=_coerce_float(track.get("last_seen_seconds") or track.get("last_seen")),
                observation_count=_coerce_int(track.get("observation_count")),
                status="COMPLETED",
                plate_text=_canonical_track_plate_text(track),
                plate_detected=_coerce_bool(track.get("plate_detected")),
                plate_detection_confidence=_coerce_float(track.get("plate_detection_confidence")),
                plate_text_confidence=_coerce_float(track.get("plate_text_confidence")),
                plate_quality_status=_coerce_str_or_none(track.get("plate_quality_status")),
                plate_ocr_reason=_coerce_str_or_none(track.get("plate_ocr_reason")),
                run_id=run_id,
            )
        )
    return records


def vehicle_records_from_physical_vehicles(vehicles: list[dict[str, Any]]) -> list[VehicleRecord]:
    records: list[VehicleRecord] = []
    seen_ids: set[str] = set()
    for vehicle in vehicles:
        if not isinstance(vehicle, dict):
            continue
        vehicle_id = str(vehicle.get("vehicle_id") or vehicle.get("vehicle_key") or "").strip()
        run_id = str(vehicle.get("run_id") or "").strip() or None
        unique_key = f"{run_id or ''}\0{vehicle_id}"
        if not vehicle_id or unique_key in seen_ids:
            continue
        seen_ids.add(unique_key)
        member_track_ids = tuple(str(item) for item in list(vehicle.get("member_track_ids") or vehicle.get("member_tracks") or []) if item)
        records.append(
            VehicleRecord(
                vehicle_id=vehicle_id,
                local_track_id=member_track_ids[0] if member_track_ids else vehicle_id,
                camera_id=str(vehicle.get("primary_camera_id") or (list(vehicle.get("camera_ids") or []) or [""])[0] or "").strip(),
                vehicle_class=_normalize_vehicle_class(vehicle.get("vehicle_class") or vehicle.get("final_class")),
                colour=_normalize_colour(vehicle.get("vehicle_colour") or vehicle.get("colour")),
                first_seen_seconds=_coerce_float(vehicle.get("first_seen_seconds") or vehicle.get("first_timestamp_seconds")),
                last_seen_seconds=_coerce_float(vehicle.get("last_seen_seconds") or vehicle.get("last_timestamp_seconds")),
                observation_count=_coerce_int(vehicle.get("member_track_count") or len(member_track_ids) or 1),
                status=str(vehicle.get("identity_status") or "PHYSICAL_VEHICLE"),
                member_track_ids=member_track_ids,
                plate_text=_canonical_physical_vehicle_plate_text(vehicle),
                plate_detected=_coerce_bool(
                    vehicle.get("plate_detected")
                    or (vehicle.get("plate") or {}).get("detected")
                    or bool(_canonical_physical_vehicle_plate_text(vehicle))
                ),
                plate_detection_confidence=_coerce_float(
                    vehicle.get("plate_detection_confidence")
                    or (vehicle.get("plate") or {}).get("detection_confidence")
                ),
                plate_text_confidence=_coerce_float(
                    vehicle.get("plate_text_confidence")
                    or (vehicle.get("plate") or {}).get("text_confidence")
                ),
                plate_quality_status=_coerce_str_or_none(
                    vehicle.get("plate_quality_status")
                    or (vehicle.get("plate") or {}).get("quality")
                ),
                plate_ocr_reason=_coerce_str_or_none(
                    vehicle.get("plate_ocr_reason")
                    or (vehicle.get("plate") or {}).get("status")
                ),
                run_id=run_id,
            )
        )
    return records


def filter_vehicles(
    records: list[VehicleRecord],
    *,
    vehicle_class: str | None = None,
    include_classes: list[str] | None = None,
    exclude_classes: list[str] | None = None,
    colour: str | None = None,
    include_colours: list[str] | None = None,
    exclude_colours: list[str] | None = None,
    start_time: float | None = None,
    end_time: float | None = None,
    camera_id: str | None = None,
) -> list[VehicleRecord]:
    query_classes = {_normalize_vehicle_class(item) for item in include_classes or []}
    query_colours = {_normalize_colour(item) for item in include_colours or []}
    if vehicle_class is not None:
        query_classes.add(_normalize_vehicle_class(vehicle_class))
    if colour is not None:
        query_colours.add(_normalize_colour(colour))
    excluded_classes = {_normalize_vehicle_class(item) for item in exclude_classes or []}
    excluded_colours = {_normalize_colour(item) for item in exclude_colours or []}
    return [
        record
        for record in records
        if (camera_id is None or record.camera_id == str(camera_id))
        and (not query_classes or record.vehicle_class in query_classes)
        and record.vehicle_class not in excluded_classes
        and (not query_colours or record.colour in query_colours)
        and record.colour not in excluded_colours
        and _overlaps_time_range(record, start_time=start_time, end_time=end_time)
    ]


def get_vehicle_statistics(
    records: list[VehicleRecord],
    *,
    vehicle_class: str | None = None,
    include_classes: list[str] | None = None,
    exclude_classes: list[str] | None = None,
    colour: str | None = None,
    include_colours: list[str] | None = None,
    exclude_colours: list[str] | None = None,
    start_time: float | None = None,
    end_time: float | None = None,
    camera_id: str | None = None,
) -> dict[str, Any]:
    matched = filter_vehicles(
        records,
        vehicle_class=vehicle_class,
        include_classes=include_classes,
        exclude_classes=exclude_classes,
        colour=colour,
        include_colours=include_colours,
        exclude_colours=exclude_colours,
        start_time=start_time,
        end_time=end_time,
        camera_id=camera_id,
    )
    return {
        "total": len(matched),
        "by_class": count_by_class(matched),
        "by_colour": count_by_colour(matched),
        "vehicle_ids": [record.vehicle_id for record in matched],
        "filters": {
            "vehicle_class": _normalize_vehicle_class(vehicle_class) if vehicle_class is not None else None,
            "include_classes": [_normalize_vehicle_class(item) for item in include_classes or []],
            "exclude_classes": [_normalize_vehicle_class(item) for item in exclude_classes or []],
            "colour": _normalize_colour(colour) if colour is not None else None,
            "include_colours": [_normalize_colour(item) for item in include_colours or []],
            "exclude_colours": [_normalize_colour(item) for item in exclude_colours or []],
            "start_time": start_time,
            "end_time": end_time,
            "camera_id": camera_id,
            "time_semantics": "overlap_inclusive",
        },
    }


def find_vehicle_class_comparison_intervals(
    records: list[VehicleRecord],
    *,
    left_class: str,
    operator: str,
    right_class: str,
    window_seconds: float = 5.0,
) -> dict[str, Any]:
    if window_seconds <= 0:
        raise ValueError("window_seconds must be > 0.")
    left = _normalize_vehicle_class(left_class)
    right = _normalize_vehicle_class(right_class)
    if operator not in {">", "<", "="}:
        raise ValueError(f"Unsupported interval comparison operator: {operator}")
    timestamps = [
        value
        for record in records
        for value in (record.first_seen_seconds, record.last_seen_seconds)
        if value is not None
    ]
    if not timestamps:
        return {
            "left_class": left,
            "right_class": right,
            "operator": operator,
            "window_seconds": window_seconds,
            "time_semantics": "visibility overlap inclusive: first_seen_seconds <= window_end and last_seen_seconds >= window_start",
            "intervals": [],
        }
    start = 0.0
    end = max(timestamps)
    intervals: list[dict[str, Any]] = []
    cursor = start
    while cursor <= end:
        window_end = cursor + window_seconds
        left_count = len([record for record in records if record.vehicle_class == left and _overlaps_time_range(record, start_time=cursor, end_time=window_end)])
        right_count = len([record for record in records if record.vehicle_class == right and _overlaps_time_range(record, start_time=cursor, end_time=window_end)])
        if _compare_counts(left_count, operator, right_count):
            intervals.append(
                {
                    "start_time": cursor,
                    "end_time": window_end,
                    left: left_count,
                    right: right_count,
                }
            )
        cursor += window_seconds
    return {
        "left_class": left,
        "right_class": right,
        "operator": operator,
        "window_seconds": window_seconds,
        "time_semantics": "vehicles observed during interval using visibility overlap inclusive semantics",
        "intervals": _merge_adjacent_intervals(intervals, left_class=left, right_class=right),
    }


def build_vehicle_analytics(records: list[VehicleRecord]) -> dict[str, Any]:
    class_counts = count_by_class(records)
    colour_counts = count_by_colour(records)
    return {
        "canonical_source": "physical_vehicles.json",
        "vehicle_record_rule": "one production PhysicalVehicle = one unique vehicle; raw LocalTracks remain member_track_ids",
        "time_filter_semantics": "overlap inclusive: first_seen_seconds <= query_end and last_seen_seconds >= query_start",
        "total_unique_vehicles": len(records),
        "vehicle_classes": class_counts,
        "colours": colour_counts,
        "vehicle_classes_present": [key for key in CLASS_COUNT_KEYS if class_counts.get(key, 0) > 0],
        "colours_present": [key for key in COLOUR_COUNT_KEYS if colour_counts.get(key, 0) > 0],
        "vehicle_ids": [record.vehicle_id for record in records],
        "records": [record.to_dict() for record in records],
    }


def build_vehicle_analytics_for_run(run_dir: str | Path) -> dict[str, Any]:
    run_path = Path(run_dir)
    records = load_vehicle_records_from_tracks_json(run_path / "tracks.json")
    return build_vehicle_analytics(records)


def write_vehicle_analytics_for_run(run_dir: str | Path) -> Path:
    run_path = Path(run_dir)
    analytics = build_vehicle_analytics_for_run(run_path)
    output_path = run_path / "vehicle_analytics.json"
    output_path.write_text(json.dumps(analytics, indent=2), encoding="utf-8")
    return output_path


def count_by_class(records: list[VehicleRecord]) -> dict[str, int]:
    counts = {key: 0 for key in CLASS_COUNT_KEYS}
    for record in records:
        key = _normalize_vehicle_class(record.vehicle_class)
        counts[key] = int(counts.get(key, 0)) + 1
    return counts


def count_by_colour(records: list[VehicleRecord]) -> dict[str, int]:
    counts = {key: 0 for key in COLOUR_COUNT_KEYS}
    for record in records:
        key = _normalize_colour(record.colour)
        counts[key] = int(counts.get(key, 0)) + 1
    return counts


def _extract_track_colour(track: dict[str, Any]) -> str:
    enrichment = track.get("vehicle_enrichment")
    if isinstance(enrichment, dict):
        colour = enrichment.get("vehicle_colour")
        if isinstance(colour, dict):
            return _normalize_colour(colour.get("label"))
    return VEHICLE_COLOUR_UNKNOWN


def _extract_track_plate_text(track: dict[str, Any]) -> str | None:
    enrichment = track.get("vehicle_enrichment")
    value = enrichment.get("plate_text") if isinstance(enrichment, dict) else track.get("plate_text")
    text = str(value or "").strip().upper()
    return text or None


def _canonical_track_plate_text(track: dict[str, Any]) -> str | None:
    candidates = [
        _extract_track_plate_text(track),
        _extract_track_enrichment_value(track, "plate_normalized_text"),
        _extract_track_enrichment_value(track, "plate_raw_text"),
        _extract_track_enrichment_value(track, "plate_ocr_raw_response"),
    ]
    return _first_valid_canonical_plate(candidates)


def _canonical_physical_vehicle_plate_text(vehicle: dict[str, Any]) -> str | None:
    candidates: list[Any] = [
        vehicle.get("consensus_plate_text"),
        vehicle.get("plate_text"),
        vehicle.get("plate_normalized_text"),
        vehicle.get("normalized_plate_text"),
        (vehicle.get("plate") or {}).get("consensus_text") if isinstance(vehicle.get("plate"), dict) else None,
    ]
    for item in list(vehicle.get("representative_evidence") or []):
        if isinstance(item, dict):
            candidates.extend([
                item.get("plate_text"),
                item.get("raw_plate_text"),
                item.get("normalized_plate_text"),
                item.get("plate_ocr_raw_response"),
            ])
    for item in list(vehicle.get("plate_evidence") or []):
        if isinstance(item, dict):
            candidates.extend([
                item.get("plate_text"),
                item.get("raw_plate_text"),
                item.get("normalized_plate_text"),
                item.get("plate_ocr_raw_response"),
            ])
    for item in list(vehicle.get("member_tracks") or []):
        if isinstance(item, dict):
            candidates.extend([
                item.get("plate_text"),
                item.get("raw_plate_text"),
                item.get("normalized_plate_text"),
                item.get("plate_ocr_raw_response"),
            ])
    return _first_valid_canonical_plate(candidates)


def _first_valid_canonical_plate(candidates: list[Any]) -> str | None:
    for candidate in candidates:
        normalized = normalize_plate_text(candidate)
        if not normalized:
            continue
        validation = validate_indian_plate(normalized)
        if validation.valid and validation.canonical_text:
            return validation.canonical_text
    return None


def _extract_track_enrichment_value(track: dict[str, Any], key: str) -> Any:
    enrichment = track.get("vehicle_enrichment")
    if isinstance(enrichment, dict) and key in enrichment:
        return enrichment.get(key)
    return track.get(key)


def _normalize_vehicle_class(value: Any) -> str:
    normalized = str(value or UNKNOWN_CLASS).strip().upper()
    return normalized if normalized in SUPPORTED_VEHICLE_CLASSES else UNKNOWN_CLASS


def _normalize_colour(value: Any) -> str:
    normalized = str(value or VEHICLE_COLOUR_UNKNOWN).strip().upper()
    return normalized if normalized in SUPPORTED_VEHICLE_COLOUR_LABELS else VEHICLE_COLOUR_UNKNOWN


def _overlaps_time_range(record: VehicleRecord, *, start_time: float | None, end_time: float | None) -> bool:
    if start_time is not None and record.last_seen_seconds is not None and record.last_seen_seconds < float(start_time):
        return False
    if end_time is not None and record.first_seen_seconds is not None and record.first_seen_seconds > float(end_time):
        return False
    return True


def _compare_counts(left: int, operator: str, right: int) -> bool:
    if operator == ">":
        return left > right
    if operator == "<":
        return left < right
    return left == right


def _merge_adjacent_intervals(intervals: list[dict[str, Any]], *, left_class: str, right_class: str) -> list[dict[str, Any]]:
    if not intervals:
        return []
    merged: list[dict[str, Any]] = [dict(intervals[0])]
    for interval in intervals[1:]:
        last = merged[-1]
        if (
            float(last["end_time"]) == float(interval["start_time"])
            and int(last.get(left_class, 0)) == int(interval.get(left_class, 0))
            and int(last.get(right_class, 0)) == int(interval.get(right_class, 0))
        ):
            last["end_time"] = interval["end_time"]
            continue
        merged.append(dict(interval))
    return merged


def _coerce_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _coerce_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    return bool(value)


def _coerce_str_or_none(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None
