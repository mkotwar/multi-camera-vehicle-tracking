from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from pathlib import Path
from typing import Any

from .vehicle_analytics import (
    VehicleRecord,
    build_vehicle_analytics,
    get_vehicle_statistics,
    load_vehicle_records_from_tracks_json,
)


SUPPORTED_INTENTS = {"COUNT", "LIST", "SUMMARY", "UNIQUE_CLASSES", "UNIQUE_COLOURS"}

CLASS_SYNONYMS: dict[str, tuple[str, ...]] = {
    "MOTORCYCLE": ("motorcycle", "motorcycles", "motor cycle", "motor cycles", "motorbike", "motorbikes", "bike", "bikes", "two wheeler", "two wheelers", "two-wheeler", "two-wheelers", "2 wheeler", "2 wheelers", "2-wheeler", "2-wheelers", "2wheeler", "2wheelers"),
    "3WHEELER": ("three wheeler", "three wheelers", "three-wheeler", "three-wheelers", "3 wheeler", "3 wheelers", "3-wheeler", "3-wheelers", "3wheeler", "3wheelers", "auto-rickshaw", "auto-rickshaws", "auto rickshaw", "auto rickshaws", "autorickshaw", "autorickshaws", "rickshaw", "rickshaws", "auto", "autos"),
    "CAR": ("car", "cars", "automobile", "automobiles", "sedan", "sedans"),
    "TRUCK": ("truck", "trucks", "lorry", "lorries"),
    "BUS": ("bus", "buses"),
    "UNKNOWN": ("unknown vehicle", "unknown vehicles", "unknown class", "unknown classes", "unclassified vehicle", "unclassified vehicles"),
}

COLOUR_SYNONYMS: dict[str, tuple[str, ...]] = {
    "BLACK": ("black",),
    "WHITE": ("white",),
    "SILVER": ("silver",),
    "GREY": ("grey", "gray"),
    "RED": ("red",),
    "BLUE": ("blue",),
    "GREEN": ("green",),
    "YELLOW": ("yellow",),
    "ORANGE": ("orange",),
    "BROWN": ("brown",),
    "BEIGE": ("beige",),
    "PURPLE": ("purple",),
}

UNSUPPORTED_CLASS_TERMS = {"van", "vans", "scooter", "scooters", "tempo", "taxi", "taxis"}
UNSUPPORTED_COLOUR_TERMS = {"dark", "light", "teal", "cyan", "gold", "maroon", "cream", "tan"}


class VehicleQueryParseError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class VehicleQuery:
    vehicle_class: str | None = None
    colour: str | None = None
    start_time: float | None = None
    end_time: float | None = None
    camera_id: str | None = None
    intent: str = "COUNT"

    def __post_init__(self) -> None:
        if self.intent not in SUPPORTED_INTENTS:
            raise VehicleQueryParseError(f"Unsupported intent: {self.intent}")
        if self.start_time is not None and self.end_time is not None and float(self.start_time) > float(self.end_time):
            raise VehicleQueryParseError("Invalid time range: start_time must be <= end_time.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def parse_vehicle_query(query: str) -> VehicleQuery:
    original = str(query or "").strip()
    if not original:
        raise VehicleQueryParseError("Query is empty.")
    normalized = _normalize_query_text(original)
    if re.search(r"\baround\s+\d+", normalized):
        raise VehicleQueryParseError("Ambiguous time expression: use 'between', 'from/to', 'after', or 'before' with seconds.")
    if "frame" in normalized or "frames" in normalized:
        raise VehicleQueryParseError("Frame-based time queries are not supported yet; use video-relative seconds.")

    intent = _parse_intent(normalized)
    vehicle_class = _parse_vehicle_class(normalized)
    colour = _parse_colour(normalized)
    start_time, end_time = parse_time_range(normalized)
    camera_id = _parse_camera_id(normalized)
    return VehicleQuery(
        vehicle_class=vehicle_class,
        colour=colour,
        start_time=start_time,
        end_time=end_time,
        camera_id=camera_id,
        intent=intent,
    )


def parse_time_value(value: str, default_unit: str | None = None) -> float:
    raw = str(value or "").strip().lower()
    if not raw:
        raise VehicleQueryParseError("Missing time value.")
    clock = re.fullmatch(r"(\d{1,2})(?::(\d{2}))(?::(\d{2}))?", raw)
    if clock:
        first = int(clock.group(1))
        second = int(clock.group(2))
        third = clock.group(3)
        if third is None:
            return float((first * 60) + second)
        return float((first * 3600) + (second * 60) + int(third))
    match = re.fullmatch(r"(\d+(?:\.\d+)?)\s*(seconds?|secs?|s|minutes?|mins?|m)?", raw)
    if not match:
        raise VehicleQueryParseError(f"Unsupported time value: {value}")
    amount = float(match.group(1))
    unit = str(match.group(2) or default_unit or "").lower()
    if unit in {"minute", "minutes", "min", "mins", "m"}:
        return amount * 60.0
    if unit in {"second", "seconds", "sec", "secs", "s"}:
        return amount
    raise VehicleQueryParseError(f"Missing time unit for value: {value}")


def parse_time_range(query: str) -> tuple[float | None, float | None]:
    text = _normalize_query_text(query)
    between = re.search(r"\bbetween\s+([0-9:.]+)\s+and\s+([0-9:.]+)\s*(seconds?|secs?|s|minutes?|mins?|m)?\b", text)
    if between:
        unit = between.group(3)
        return _validated_range(parse_time_value(between.group(1), unit), parse_time_value(between.group(2), unit))
    from_to = re.search(r"\bfrom\s+([0-9:.]+)\s+to\s+([0-9:.]+)\s*(seconds?|secs?|s|minutes?|mins?|m)?\b", text)
    if from_to:
        unit = from_to.group(3)
        return _validated_range(parse_time_value(from_to.group(1), unit), parse_time_value(from_to.group(2), unit))
    first = re.search(r"\b(?:in\s+the\s+)?first\s+([0-9:.]+)\s*(seconds?|secs?|s|minutes?|mins?|m)\b", text)
    if first:
        return _validated_range(0.0, parse_time_value(first.group(1), first.group(2)))
    if re.search(r"\b(?:in\s+the\s+)?first\s+minute\b", text):
        return 0.0, 60.0
    after = re.search(r"\bafter\s+([0-9:.]+)\s*(seconds?|secs?|s|minutes?|mins?|m)\b", text)
    before = re.search(r"\bbefore\s+([0-9:.]+)\s*(seconds?|secs?|s|minutes?|mins?|m)\b", text)
    start_time = parse_time_value(after.group(1), after.group(2)) if after else None
    end_time = parse_time_value(before.group(1), before.group(2)) if before else None
    if start_time is not None or end_time is not None:
        return _validated_range(start_time, end_time)
    return None, None


def execute_vehicle_query(records: list[VehicleRecord], parsed_query: VehicleQuery) -> dict[str, Any]:
    analytics = build_vehicle_analytics(records)
    if parsed_query.intent == "UNIQUE_CLASSES":
        return {"vehicle_classes_present": analytics["vehicle_classes_present"]}
    if parsed_query.intent == "UNIQUE_COLOURS":
        return {"colours_present": analytics["colours_present"]}
    if parsed_query.intent == "SUMMARY":
        return analytics
    stats = get_vehicle_statistics(
        records,
        vehicle_class=parsed_query.vehicle_class,
        colour=parsed_query.colour,
        start_time=parsed_query.start_time,
        end_time=parsed_query.end_time,
        camera_id=parsed_query.camera_id,
    )
    if parsed_query.intent == "COUNT":
        return {"total": stats["total"], "by_class": stats["by_class"], "by_colour": stats["by_colour"], "vehicle_ids": stats["vehicle_ids"]}
    return stats


def format_vehicle_query_response(parsed_query: VehicleQuery, analytics_result: dict[str, Any]) -> str:
    if parsed_query.intent == "UNIQUE_CLASSES":
        return "Vehicle classes present: " + ", ".join(analytics_result["vehicle_classes_present"]) + "."
    if parsed_query.intent == "UNIQUE_COLOURS":
        return "Vehicle colours present: " + ", ".join(analytics_result["colours_present"]) + "."
    if parsed_query.intent == "SUMMARY":
        return f"There are {analytics_result['total_unique_vehicles']} completed unique vehicle records in the video."
    total = int(analytics_result.get("total", 0))
    description = _query_description(parsed_query, total=total)
    if parsed_query.intent == "COUNT":
        verb = "is" if total == 1 else "are"
        return f"There {verb} {total} {description}."
    ids = list(analytics_result.get("vehicle_ids", []) or [])
    verb = "was" if total == 1 else "were"
    lines = [f"{total} {description} {verb} observed."]
    if ids:
        lines.append("")
        lines.append("Vehicle IDs:")
        lines.extend(str(item) for item in ids)
    return "\n".join(lines)


def search_vehicle_data(query: str, tracks_path: str | Path) -> dict[str, Any]:
    records = load_vehicle_records_from_tracks_json(tracks_path)
    return search_vehicle_records(query=query, records=records)


def search_vehicle_records(query: str, records: list[VehicleRecord]) -> dict[str, Any]:
    parsed = parse_vehicle_query(query)
    result = execute_vehicle_query(records, parsed)
    response = format_vehicle_query_response(parsed, result)
    return {
        "original_query": query,
        "parsed_query": parsed.to_dict(),
        "analytics_result": result,
        "response": response,
    }


def _parse_intent(text: str) -> str:
    if re.search(r"\b(types?|classes?|kinds?)\s+of\s+vehicles?\s+(?:are\s+)?present\b", text) or "vehicle types are present" in text:
        return "UNIQUE_CLASSES"
    if re.search(r"\bcolou?rs?\s+(?:are\s+)?present\b", text):
        return "UNIQUE_COLOURS"
    if re.search(r"\b(summ?ary|summ?ry|summarize|summarise|overview|breakdown)\b", text):
        return "SUMMARY"
    if re.search(r"\b(show|list|find|display)\b", text):
        return "LIST"
    if re.search(r"\b(how many|count|number of|total)\b", text):
        return "COUNT"
    raise VehicleQueryParseError("Could not determine query intent.")


def _parse_vehicle_class(text: str) -> str | None:
    matches = _find_synonym_matches(text, CLASS_SYNONYMS)
    if len(matches) > 1:
        raise VehicleQueryParseError(f"Ambiguous vehicle class terms: {sorted(matches)}")
    unsupported = sorted(term for term in UNSUPPORTED_CLASS_TERMS if _contains_phrase(text, term))
    if unsupported:
        raise VehicleQueryParseError(f"Unsupported vehicle class term: {unsupported[0]}")
    return next(iter(matches)) if matches else None


def _parse_colour(text: str) -> str | None:
    matches = _find_synonym_matches(text, COLOUR_SYNONYMS)
    if len(matches) > 1:
        raise VehicleQueryParseError(f"Ambiguous colour terms: {sorted(matches)}")
    unsupported = sorted(term for term in UNSUPPORTED_COLOUR_TERMS if _contains_phrase(text, term))
    if unsupported:
        raise VehicleQueryParseError(f"Unsupported colour term: {unsupported[0]}")
    return next(iter(matches)) if matches else None


def _parse_camera_id(text: str) -> str | None:
    match = re.search(r"\b(cam_\d+|camera\s+\d+)\b", text)
    if not match:
        return None
    raw = match.group(1).replace("camera", "cam").replace(" ", "_").upper()
    return raw


def _find_synonym_matches(text: str, synonyms: dict[str, tuple[str, ...]]) -> set[str]:
    matches: set[str] = set()
    for label, phrases in synonyms.items():
        if any(_contains_phrase(text, phrase) for phrase in phrases):
            matches.add(label)
    return matches


def _contains_phrase(text: str, phrase: str) -> bool:
    normalized_phrase = re.escape(_normalize_query_text(phrase)).replace(r"\ ", r"[\s-]+")
    return bool(re.search(rf"(?<!\w){normalized_phrase}(?!\w)", text))


def _validated_range(start_time: float | None, end_time: float | None) -> tuple[float | None, float | None]:
    if start_time is not None and end_time is not None and start_time > end_time:
        raise VehicleQueryParseError("Invalid time range: start_time must be <= end_time.")
    return start_time, end_time


def _normalize_query_text(query: str) -> str:
    text = str(query or "").strip().lower()
    text = text.replace("colour", "color")
    text = re.sub(r"[?,.!]", " ", text)
    return " ".join(text.split())


def _query_description(query: VehicleQuery, *, total: int) -> str:
    parts: list[str] = []
    if query.colour:
        parts.append(query.colour.lower())
    if query.vehicle_class:
        parts.append(_class_noun(query.vehicle_class, total=total))
    else:
        parts.append("vehicles")
    description = " ".join(parts)
    if query.start_time is not None and query.end_time is not None:
        description += f" between {query.start_time:.1f} and {query.end_time:.1f} seconds"
    elif query.start_time is not None:
        description += f" after {query.start_time:.1f} seconds"
    elif query.end_time is not None:
        description += f" before {query.end_time:.1f} seconds"
    return description


def _class_noun(vehicle_class: str, *, total: int) -> str:
    singular = {
        "CAR": "car",
        "MOTORCYCLE": "motorcycle",
        "BUS": "bus",
        "TRUCK": "truck",
        "3WHEELER": "3wheeler",
        "UNKNOWN": "unknown vehicle",
    }.get(vehicle_class, vehicle_class.lower())
    if total == 1:
        return singular
    plural = {
        "bus": "buses",
        "3wheeler": "3wheelers",
        "unknown vehicle": "unknown vehicles",
    }.get(singular)
    return plural or f"{singular}s"
