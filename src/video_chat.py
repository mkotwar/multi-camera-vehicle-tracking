from __future__ import annotations

from dataclasses import asdict, dataclass, field
import re
from typing import Any, Protocol

from .run_repository import RunRepository
from .vehicle_analytics import (
    CLASS_COUNT_KEYS,
    COLOUR_COUNT_KEYS,
    VehicleRecord,
    count_by_class,
    count_by_colour,
    find_vehicle_class_comparison_intervals,
    load_vehicle_records_from_tracks_json,
)
from .vehicle_enrichment.taxonomy import SUPPORTED_VEHICLE_CLASSES, SUPPORTED_VEHICLE_COLOUR_LABELS
from .vehicle_nlp import (
    CLASS_SYNONYMS,
    COLOUR_SYNONYMS,
    UNSUPPORTED_CLASS_TERMS,
    UNSUPPORTED_COLOUR_TERMS,
    VehicleQueryParseError,
    parse_time_range,
)


SUPPORTED_CHAT_INTENTS = {
    "GENERAL_CHAT",
    "COUNT",
    "LIST",
    "SUMMARY",
    "GROUP",
    "COMPARE",
    "FIND_INTERVALS",
    "UNIQUE_CLASSES",
    "UNIQUE_COLOURS",
}
EVIDENCE_LIMIT = 6
ANALYTICS_INTENTS = SUPPORTED_CHAT_INTENTS - {"GENERAL_CHAT"}

LLM_CLASS_ALIASES = {
    "BIKE": "MOTORCYCLE",
    "BIKES": "MOTORCYCLE",
    "MOTORBIKE": "MOTORCYCLE",
    "MOTORBIKES": "MOTORCYCLE",
    "TWO_WHEELER": "MOTORCYCLE",
    "TWO-WHEELER": "MOTORCYCLE",
    "TWO_WHEELERS": "MOTORCYCLE",
    "TWO-WHEELERS": "MOTORCYCLE",
    "AUTO": "3WHEELER",
    "AUTOS": "3WHEELER",
    "AUTO_RICKSHAW": "3WHEELER",
    "AUTO-RICKSHAW": "3WHEELER",
    "RICKSHAW": "3WHEELER",
    "RICKSHAWS": "3WHEELER",
    "THREE_WHEELER": "3WHEELER",
    "THREE-WHEELER": "3WHEELER",
    "THREE_WHEELERS": "3WHEELER",
    "THREE-WHEELERS": "3WHEELER",
}
LLM_COLOUR_ALIASES = {"GRAY": "GREY"}
EXCLUSION_PHRASE_PATTERN = r"\b(?:except|other\s+than|apart\s+from|excluding|exclude|without|not\s+including|not|anything\s+but|everything\s+but|all\s+but|all\s+except|but\s+not)\b"


class ChatLLMProvider(Protocol):
    def parse(self, message: str, context: dict[str, Any]) -> dict[str, Any]:
        ...


@dataclass(frozen=True, slots=True)
class ChatVehicleQuery:
    intent: str
    include_classes: list[str] = field(default_factory=list)
    exclude_classes: list[str] = field(default_factory=list)
    include_colours: list[str] = field(default_factory=list)
    exclude_colours: list[str] = field(default_factory=list)
    start_time: float | None = None
    end_time: float | None = None
    camera_id: str | None = None
    group_by: str | None = None
    comparison: dict[str, Any] | None = None
    sort_by: str | None = None
    limit: int | None = None
    show_evidence: bool = False
    context_reference: str | None = None
    evidence_navigation: str | None = None

    def __post_init__(self) -> None:
        if self.intent not in SUPPORTED_CHAT_INTENTS:
            raise VehicleQueryParseError(f"Unsupported chat intent: {self.intent}")
        for label in self.include_classes + self.exclude_classes:
            if label not in (*SUPPORTED_VEHICLE_CLASSES, "UNKNOWN"):
                raise VehicleQueryParseError(f"Unsupported vehicle class from parser: {label}")
        for label in self.include_colours + self.exclude_colours:
            if label not in SUPPORTED_VEHICLE_COLOUR_LABELS:
                raise VehicleQueryParseError(f"Unsupported colour from parser: {label}")
        if self.group_by not in {None, "class", "colour"}:
            raise VehicleQueryParseError(f"Unsupported group_by: {self.group_by}")
        if self.evidence_navigation not in {None, "next", "restart"}:
            raise VehicleQueryParseError(f"Unsupported evidence_navigation: {self.evidence_navigation}")
        if self.limit is not None and int(self.limit) < 1:
            raise VehicleQueryParseError("Invalid limit: must be >= 1.")
        if self.start_time is not None and self.end_time is not None and self.start_time > self.end_time:
            raise VehicleQueryParseError("Invalid time range: start_time must be <= end_time.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ExplicitFilterMentions:
    positive_classes: list[str] = field(default_factory=list)
    negative_classes: list[str] = field(default_factory=list)
    positive_colours: list[str] = field(default_factory=list)
    negative_colours: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def handle_video_chat(
    *,
    message: str,
    run_id: str,
    repository: RunRepository,
    tracks_path: str | None = None,
    records: list[VehicleRecord] | None = None,
    session_context: dict[str, Any] | None = None,
    llm_provider: ChatLLMProvider | None = None,
) -> dict[str, Any]:
    context = dict(session_context or {})
    if records is None:
        if tracks_path is None:
            raise ValueError("Either tracks_path or records is required.")
        records = load_vehicle_records_from_tracks_json(tracks_path)
    parsed, parser_used, diagnostics = parse_chat_vehicle_query_detailed(message=message, context=context, llm_provider=llm_provider)
    analytics_result = execute_chat_vehicle_query(records, parsed, context=context) if parsed.intent in ANALYTICS_INTENTS else {}
    matching_vehicle_ids = list(analytics_result.get("vehicle_ids", []) or [])
    previous_ids = list(context.get("previous_vehicle_ids", []) or [])
    is_evidence_navigation = parsed.evidence_navigation is not None
    if is_evidence_navigation and previous_ids:
        matching_vehicle_ids = [str(item) for item in previous_ids]
        analytics_result = {**analytics_result, "total": len(matching_vehicle_ids), "vehicle_ids": matching_vehicle_ids}
    evidence_offset = _evidence_offset_for_query(parsed, context=context)
    evidence = resolve_vehicle_evidence(
        repository=repository,
        run_id=run_id,
        vehicle_ids=matching_vehicle_ids,
        limit=EVIDENCE_LIMIT if parsed.show_evidence else 0,
        offset=evidence_offset,
    )
    evidence_before_validation = len(evidence)
    evidence = [item for item in evidence if _evidence_matches_query(item, parsed)]
    evidence_page = _evidence_page_metadata(
        matching_total=len(matching_vehicle_ids),
        returned_count=len(evidence),
        offset=evidence_offset if parsed.show_evidence else 0,
        page_size=EVIDENCE_LIMIT,
    )
    answer = format_chat_answer(parsed, analytics_result, evidence_count=len(evidence), evidence_page=evidence_page)
    next_evidence_offset = evidence_page["next_offset"] if parsed.show_evidence else 0
    if not parsed.show_evidence and not is_evidence_navigation:
        next_evidence_offset = 0
    next_context = dict(context) if parsed.intent == "GENERAL_CHAT" else {
        "previous_query": parsed.to_dict(),
        "previous_filters": {
            "include_classes": parsed.include_classes,
            "exclude_classes": parsed.exclude_classes,
            "include_colours": parsed.include_colours,
            "exclude_colours": parsed.exclude_colours,
            "start_time": parsed.start_time,
            "end_time": parsed.end_time,
            "camera_id": parsed.camera_id,
        },
        "previous_vehicle_ids": matching_vehicle_ids,
        "evidence_offset": next_evidence_offset,
        "evidence_page_size": EVIDENCE_LIMIT,
        "evidence_shown_ids": _next_shown_ids(context if is_evidence_navigation else {}, evidence),
    }
    filters_after_context = _query_filters(parsed)
    diagnostics = {
        **diagnostics,
        "context_was_available": bool(context.get("previous_query") or context.get("previous_vehicle_ids")),
        "context_reference": parsed.context_reference,
        "filters_after_context": filters_after_context,
        "explicit_filters_detected": _explicit_filters_detected(parsed),
        "filters_before_validation": diagnostics.get("filters_before_context"),
        "filters_after_validation": filters_after_context,
        "group_by": parsed.group_by,
        "matching_vehicle_ids_count": len(matching_vehicle_ids),
        "matching_count": len(matching_vehicle_ids),
        "evidence_validation_removed_count": evidence_before_validation - len(evidence),
        "context_saved_vehicle_ids_count": len(next_context.get("previous_vehicle_ids", []) or []),
    }
    diagnostics.setdefault("filters_before_context", filters_after_context)
    return {
        "answer": answer,
        "original_query": message,
        "parser_used": parser_used,
        **diagnostics,
        "parsed_query": parsed.to_dict(),
        "analytics_result": analytics_result,
        "matching_vehicle_ids": matching_vehicle_ids,
        "evidence": evidence,
        "evidence_page": evidence_page,
        "context_used": parsed.context_reference == "previous_results",
        "next_context": next_context,
    }


def parse_chat_vehicle_query(
    *,
    message: str,
    context: dict[str, Any] | None = None,
    llm_provider: ChatLLMProvider | None = None,
) -> tuple[ChatVehicleQuery, str]:
    parsed, parser_used, _ = parse_chat_vehicle_query_detailed(message=message, context=context, llm_provider=llm_provider)
    return parsed, parser_used


def parse_chat_vehicle_query_detailed(
    *,
    message: str,
    context: dict[str, Any] | None = None,
    llm_provider: ChatLLMProvider | None = None,
) -> tuple[ChatVehicleQuery, str, dict[str, Any]]:
    text = _normalize(str(message or ""))
    if not text:
        raise VehicleQueryParseError("Query is empty.")
    diagnostics: dict[str, Any] = {
        "llm_attempted": llm_provider is not None,
        "llm_accepted": False,
        "llm_rejection_reason": None,
        "llm_raw_structured_output": None,
        "qwen_raw_plan": None,
        "normalized_llm_output": None,
        "normalized_plan": None,
        "semantic_validation_result": "not_run",
        "semantic_repair_applied": False,
        "final_query_plan": None,
        "message_type": _classify_message_type(text, context or {}),
        "filters_before_context": None,
    }
    explicit_mentions = _extract_explicit_filter_mentions(text)
    diagnostics.update(
        {
            "explicit_positive_classes": explicit_mentions.positive_classes,
            "explicit_negative_classes": explicit_mentions.negative_classes,
            "explicit_positive_colours": explicit_mentions.positive_colours,
            "explicit_negative_colours": explicit_mentions.negative_colours,
        }
    )
    general_chat = _parse_general_chat_query(text)
    if general_chat is not None:
        diagnostics["llm_attempted"] = False
        diagnostics["filters_before_context"] = _query_filters(general_chat)
        diagnostics["final_query_plan"] = general_chat.to_dict()
        return general_chat, "rule_based", diagnostics
    navigation = _parse_evidence_navigation_query(text, context or {})
    if navigation is not None:
        diagnostics["filters_before_context"] = _query_filters(navigation)
        diagnostics["final_query_plan"] = navigation.to_dict()
        return navigation, "rule_based", diagnostics
    if llm_provider is not None:
        try:
            payload = llm_provider.parse(message, _llm_context(context or {}))
            diagnostics["llm_raw_structured_output"] = dict(payload) if isinstance(payload, dict) else payload
            diagnostics["qwen_raw_plan"] = diagnostics["llm_raw_structured_output"]
            normalized = normalize_llm_vehicle_query(payload)
            repaired, repair_applied, validation_result = _repair_llm_plan_with_explicit_mentions(normalized, explicit_mentions)
            diagnostics["normalized_llm_output"] = normalized
            diagnostics["normalized_plan"] = repaired
            diagnostics["semantic_validation_result"] = validation_result
            diagnostics["semantic_repair_applied"] = repair_applied
            diagnostics["filters_before_context"] = _payload_filters(repaired)
            parsed = chat_query_from_llm_vehicle_query(repaired, text=text, context=context or {}, explicit_mentions=explicit_mentions)
            diagnostics["final_query_plan"] = parsed.to_dict()
            diagnostics["llm_accepted"] = True
            return parsed, "qwen", diagnostics
        except Exception as exc:
            diagnostics["llm_rejection_reason"] = _diagnostic_reason(exc)
    try:
        parsed = _parse_rule_chat_query(text, context or {})
        diagnostics["filters_before_context"] = _query_filters(parsed)
        diagnostics["final_query_plan"] = parsed.to_dict()
        return parsed, "rule_based", diagnostics
    except VehicleQueryParseError:
        raise


def execute_chat_vehicle_query(records: list[VehicleRecord], parsed: ChatVehicleQuery, *, context: dict[str, Any] | None = None) -> dict[str, Any]:
    base_ids = None
    if parsed.context_reference == "previous_results":
        base_ids = set(str(item) for item in list((context or {}).get("previous_vehicle_ids", []) or []))
    matched = _filter_records(
        records,
        include_classes=parsed.include_classes,
        exclude_classes=parsed.exclude_classes,
        include_colours=parsed.include_colours,
        exclude_colours=parsed.exclude_colours,
        start_time=parsed.start_time,
        end_time=parsed.end_time,
        camera_id=parsed.camera_id,
        base_ids=base_ids,
    )
    if parsed.intent == "SUMMARY":
        return _summary_payload(matched)
    if parsed.intent == "UNIQUE_CLASSES":
        counts = count_by_class(matched)
        return {"vehicle_classes_present": [label for label, count in counts.items() if count > 0], "vehicle_ids": [record.vehicle_id for record in matched]}
    if parsed.intent == "UNIQUE_COLOURS":
        counts = count_by_colour(matched)
        return {"colours_present": [label for label, count in counts.items() if count > 0], "vehicle_ids": [record.vehicle_id for record in matched]}
    if parsed.intent == "COMPARE":
        left = str((parsed.comparison or {}).get("left") or "").upper()
        right = str((parsed.comparison or {}).get("right") or "").upper()
        comparison_base = _filter_records(
            records,
            exclude_classes=parsed.exclude_classes,
            include_colours=parsed.include_colours,
            exclude_colours=parsed.exclude_colours,
            start_time=parsed.start_time,
            end_time=parsed.end_time,
            camera_id=parsed.camera_id,
            base_ids=base_ids,
        )
        left_records = _filter_records(comparison_base, include_classes=[left] if left else [])
        right_records = _filter_records(comparison_base, include_classes=[right] if right else [])
        return {
            "left": left,
            "right": right,
            "left_total": len(left_records),
            "right_total": len(right_records),
            "answer": "YES" if len(left_records) > len(right_records) else "NO",
            "vehicle_ids": [record.vehicle_id for record in left_records + right_records],
        }
    if parsed.intent == "FIND_INTERVALS":
        comparison = parsed.comparison or {}
        interval_base = _filter_records(
            records,
            exclude_classes=parsed.exclude_classes,
            include_colours=parsed.include_colours,
            exclude_colours=parsed.exclude_colours,
            start_time=parsed.start_time,
            end_time=parsed.end_time,
            camera_id=parsed.camera_id,
            base_ids=base_ids,
        )
        return find_vehicle_class_comparison_intervals(
            interval_base,
            left_class=str(comparison.get("left") or ""),
            operator=str(comparison.get("operator") or ">"),
            right_class=str(comparison.get("right") or ""),
            window_seconds=float(comparison.get("window_seconds") or 5.0),
        )
    if parsed.intent == "GROUP" and parsed.group_by == "colour":
        counts = count_by_colour(matched)
        return {"total": len(matched), "by_colour": counts, "top_colour": _top_count(counts), "vehicle_ids": [record.vehicle_id for record in matched]}
    if parsed.intent == "GROUP" and parsed.group_by == "class":
        counts = count_by_class(matched)
        return {"total": len(matched), "by_class": counts, "top_class": _top_count(counts), "vehicle_ids": [record.vehicle_id for record in matched]}
    return {
        "total": len(matched),
        "by_class": count_by_class(matched),
        "by_colour": count_by_colour(matched),
        "vehicle_ids": [record.vehicle_id for record in matched],
    }


def resolve_vehicle_evidence(
    *,
    repository: RunRepository,
    run_id: str,
    vehicle_ids: list[str],
    limit: int = EVIDENCE_LIMIT,
    offset: int = 0,
) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    if limit <= 0:
        return evidence
    for vehicle_id in vehicle_ids[offset:offset + limit]:
        physical_vehicle = _get_physical_vehicle(repository, vehicle_id=vehicle_id, run_id=run_id)
        if physical_vehicle is not None:
            evidence.append(_physical_vehicle_evidence(repository=repository, run_id=run_id, vehicle=physical_vehicle))
            continue
        camera_id, track_id = _split_vehicle_id(vehicle_id)
        if not camera_id or not track_id:
            continue
        track = repository.get_track(camera_id=camera_id, track_id=track_id, run_id=run_id)
        if track is None:
            continue
        media = track.get("best_crop_parts")
        image_url = _media_url(media)
        if image_url is not None and repository.resolve_media_path(
            run_id=run_id,
            category=str(media.get("category")),
            relative_parts=[str(item) for item in list(media.get("parts", []) or [])],
        ) is None:
            image_url = None
        evidence.append(
            {
                "vehicle_id": vehicle_id,
                "camera_id": str(track.get("camera_id") or camera_id),
                "track_id": str(track.get("track_id") or track_id),
                "member_track_ids": [str(track.get("local_track_id") or vehicle_id)],
                "vehicle_class": str(track.get("vehicle_class") or "UNKNOWN").upper(),
                "colour": str(track.get("colour") or "UNKNOWN").upper(),
                "plate_text": _clean_plate_text(track.get("plate_text")),
                "first_seen_seconds": track.get("first_seen_seconds"),
                "last_seen_seconds": track.get("last_seen_seconds"),
                "best_crop_url": image_url,
                "image_url": image_url,
                "track_detail_url": f"/tracks/{camera_id}/{track_id}?run_id={run_id}",
            }
        )
    return evidence


def _get_physical_vehicle(repository: RunRepository, *, vehicle_id: str, run_id: str) -> dict[str, Any] | None:
    getter = getattr(repository, "get_physical_vehicle", None)
    if not callable(getter):
        return None
    vehicle = getter(vehicle_id=vehicle_id, run_id=run_id)
    return dict(vehicle) if isinstance(vehicle, dict) else None


def _physical_vehicle_evidence(*, repository: RunRepository, run_id: str, vehicle: dict[str, Any]) -> dict[str, Any]:
    vehicle_id = str(vehicle.get("vehicle_id") or vehicle.get("vehicle_key") or "")
    member_track_ids = [str(item) for item in list(vehicle.get("member_track_ids") or vehicle.get("member_tracks") or []) if item]
    primary_track_id = member_track_ids[0] if member_track_ids else ""
    camera_id = str(vehicle.get("primary_camera_id") or (list(vehicle.get("camera_ids") or []) or [""])[0] or _split_vehicle_id(primary_track_id)[0] or "")
    track_id = str(_split_vehicle_id(primary_track_id)[1] or primary_track_id or vehicle_id)
    image_url = _physical_vehicle_image_url(repository=repository, run_id=run_id, vehicle=vehicle, member_track_ids=member_track_ids)
    plate_text = _physical_vehicle_plate_text(repository=repository, run_id=run_id, vehicle=vehicle, member_track_ids=member_track_ids)
    return {
        "vehicle_id": vehicle_id,
        "camera_id": camera_id,
        "track_id": track_id,
        "member_track_ids": member_track_ids,
        "vehicle_class": str(vehicle.get("vehicle_class") or vehicle.get("final_class") or "UNKNOWN").upper(),
        "colour": str(vehicle.get("vehicle_colour") or vehicle.get("colour") or "UNKNOWN").upper(),
        "plate_text": plate_text,
        "first_seen_seconds": vehicle.get("first_seen_seconds") or vehicle.get("first_timestamp_seconds"),
        "last_seen_seconds": vehicle.get("last_seen_seconds") or vehicle.get("last_timestamp_seconds"),
        "best_crop_url": image_url,
        "image_url": image_url,
        "track_detail_url": f"/tracks/{camera_id}/{track_id}?run_id={run_id}" if camera_id and track_id else "",
    }


def _physical_vehicle_plate_text(
    *,
    repository: RunRepository,
    run_id: str,
    vehicle: dict[str, Any],
    member_track_ids: list[str],
) -> str | None:
    plate_payload = vehicle.get("plate")
    consensus = _clean_plate_text(
        vehicle.get("consensus_plate_text")
        or (plate_payload.get("consensus_text") if isinstance(plate_payload, dict) else None)
    )
    if consensus:
        return consensus
    for item in list(vehicle.get("representative_evidence") or []):
        if not isinstance(item, dict):
            continue
        plate_text = _clean_plate_text(item.get("plate_text") or item.get("normalized_plate_text"))
        if plate_text:
            return plate_text
    for local_track_id in member_track_ids:
        camera_id, track_id = _split_vehicle_id(local_track_id)
        if not camera_id or not track_id:
            continue
        track = repository.get_track(camera_id=camera_id, track_id=track_id, run_id=run_id)
        if not track:
            continue
        plate_text = _clean_plate_text(track.get("plate_text"))
        if plate_text:
            return plate_text
    return None


def _clean_plate_text(value: Any) -> str | None:
    text = str(value or "").strip().upper()
    return text or None


def _physical_vehicle_image_url(
    *,
    repository: RunRepository,
    run_id: str,
    vehicle: dict[str, Any],
    member_track_ids: list[str],
) -> str | None:
    direct_url = str(vehicle.get("best_crop_url") or "").strip()
    if direct_url:
        return direct_url
    for item in list(vehicle.get("representative_evidence") or []):
        if not isinstance(item, dict):
            continue
        url = str(item.get("vehicle_crop_url") or "").strip()
        if url:
            return url
        url = _media_url_from_path(repository=repository, run_id=run_id, path=item.get("vehicle_crop_path"))
        if url:
            return url
    for local_track_id in member_track_ids:
        camera_id, track_id = _split_vehicle_id(local_track_id)
        if not camera_id or not track_id:
            continue
        track = repository.get_track(camera_id=camera_id, track_id=track_id, run_id=run_id)
        if not track:
            continue
        url = _valid_media_url(repository=repository, media=track.get("best_crop_parts"))
        if url:
            return url
    return None


def _media_url_from_path(*, repository: RunRepository, run_id: str, path: Any) -> str | None:
    if not path:
        return None
    raw = str(path).replace("\\", "/")
    marker = f"/outputs/runs/{run_id}/"
    if marker in raw:
        raw = raw.split(marker, 1)[1]
    elif raw.startswith(f"outputs/runs/{run_id}/"):
        raw = raw[len(f"outputs/runs/{run_id}/") :]
    elif f"/{run_id}/" in raw:
        raw = raw.split(f"/{run_id}/", 1)[1]
    parts = [part for part in raw.split("/") if part]
    if not parts:
        return None
    category_map = {
        "evidence": "evidence",
        "05_florence_selected_crops": "florence_selected_crops",
        "04_track_crops": "track_crops",
        "07_body_type_selected_crops": "body_type_selected_crops",
        "tracked_frames": "tracked_frames",
        "detected_frames": "detected_frames",
        "raw_frames": "raw_frames",
    }
    category = category_map.get(parts[0])
    if category is None:
        return None
    media_parts = parts[1:]
    if repository.resolve_media_path(run_id=run_id, category=category, relative_parts=media_parts) is None:
        return None
    return f"/api/media/{category}/{run_id}/{'/'.join(media_parts)}"


def _valid_media_url(*, repository: RunRepository, media: Any) -> str | None:
    image_url = _media_url(media)
    if image_url is None or not isinstance(media, dict):
        return image_url
    if repository.resolve_media_path(
        run_id=str(media.get("run_id") or ""),
        category=str(media.get("category") or ""),
        relative_parts=[str(item) for item in list(media.get("parts", []) or [])],
    ) is None:
        return None
    return image_url


def _evidence_offset_for_query(parsed: ChatVehicleQuery, *, context: dict[str, Any]) -> int:
    if not parsed.show_evidence:
        return 0
    if parsed.evidence_navigation == "restart":
        return 0
    if parsed.evidence_navigation == "next":
        return int(context.get("evidence_offset", 0) or 0)
    return 0


def _evidence_page_metadata(*, matching_total: int, returned_count: int, offset: int, page_size: int) -> dict[str, Any]:
    next_offset = min(matching_total, offset + returned_count)
    return {
        "matching_total": matching_total,
        "evidence_returned_count": returned_count,
        "evidence_offset": offset,
        "evidence_page_size": page_size,
        "evidence_remaining_count": max(0, matching_total - next_offset),
        "shown_count": next_offset,
        "next_offset": next_offset,
    }


def _next_shown_ids(context: dict[str, Any], evidence: list[dict[str, Any]]) -> list[str]:
    shown = [str(item) for item in list(context.get("evidence_shown_ids", []) or [])]
    for item in evidence:
        vehicle_id = str(item.get("vehicle_id") or "")
        if vehicle_id and vehicle_id not in shown:
            shown.append(vehicle_id)
    return shown


def _evidence_matches_query(item: dict[str, Any], parsed: ChatVehicleQuery) -> bool:
    vehicle_class = str(item.get("vehicle_class") or "UNKNOWN").upper()
    colour = str(item.get("colour") or "UNKNOWN").upper()
    if parsed.include_classes and vehicle_class not in set(parsed.include_classes):
        return False
    if vehicle_class in set(parsed.exclude_classes):
        return False
    if parsed.include_colours and colour not in set(parsed.include_colours):
        return False
    if colour in set(parsed.exclude_colours):
        return False
    return True


def format_chat_answer(
    parsed: ChatVehicleQuery,
    analytics_result: dict[str, Any],
    *,
    evidence_count: int = 0,
    evidence_page: dict[str, Any] | None = None,
) -> str:
    if parsed.intent == "GENERAL_CHAT":
        return "Hello. I can answer questions about this processed traffic video, such as vehicle counts, classes, colours, time ranges, comparisons, and evidence."
    if parsed.intent == "SUMMARY":
        classes = _nonzero_counts(analytics_result.get("vehicle_classes", {}))
        colours = _nonzero_counts(analytics_result.get("colours", {}))
        return (
            f"Traffic summary: {analytics_result['total_unique_vehicles']} completed unique vehicle records. "
            f"Classes: {_format_counts(classes)}. Colours: {_format_counts(colours)}."
        )
    if parsed.intent == "UNIQUE_CLASSES":
        return "Vehicle classes present: " + ", ".join(analytics_result.get("vehicle_classes_present", [])) + "."
    if parsed.intent == "UNIQUE_COLOURS":
        return "Vehicle colours present: " + ", ".join(analytics_result.get("colours_present", [])) + "."
    if parsed.intent == "COMPARE":
        answer = "Yes." if analytics_result.get("answer") == "YES" else "No."
        return f"{answer} {analytics_result.get('left')} = {analytics_result.get('left_total')}; {analytics_result.get('right')} = {analytics_result.get('right_total')}."
    if parsed.intent == "FIND_INTERVALS":
        return _format_interval_answer(analytics_result)
    if parsed.intent == "GROUP":
        key = "by_colour" if parsed.group_by == "colour" else "by_class"
        top_key = "top_colour" if parsed.group_by == "colour" else "top_class"
        top = analytics_result.get(top_key)
        natural = _format_natural_group_answer(parsed, analytics_result)
        if natural:
            return natural
        if parsed.limit == 1 and isinstance(top, dict):
            return f"I found {analytics_result.get('total', 0)} matching vehicles. Most common {parsed.group_by}: {top.get('label')} ({top.get('count')})."
        return f"I found {analytics_result.get('total', 0)} matching vehicles. {parsed.group_by.title()} breakdown: {_format_counts(_nonzero_counts(analytics_result.get(key, {})))}."
    total = int(analytics_result.get("total", 0) or 0)
    description = _query_description(parsed, total)
    if parsed.intent == "COUNT":
        verb = "is" if total == 1 else "are"
        return f"There {verb} {total} {description}."
    shown = _format_evidence_page_text(parsed, evidence_page, evidence_count=evidence_count)
    verb = "was" if total == 1 else "were"
    return f"{total} {description} {verb} observed.{shown}"


def _format_interval_answer(analytics_result: dict[str, Any]) -> str:
    left = str(analytics_result.get("left_class") or "")
    right = str(analytics_result.get("right_class") or "")
    operator = str(analytics_result.get("operator") or ">")
    intervals = list(analytics_result.get("intervals", []) or [])
    if not intervals:
        return f"No {left} {operator} {right} intervals were found using {analytics_result.get('window_seconds')} second visibility windows."
    pieces = [
        f"{float(item['start_time']):.1f}-{float(item['end_time']):.1f}s ({left} {item.get(left, 0)}, {right} {item.get(right, 0)})"
        for item in intervals
    ]
    return f"{left} {operator} {right} during: " + "; ".join(pieces) + ". These are vehicles observed during each interval."


def _format_natural_group_answer(parsed: ChatVehicleQuery, analytics_result: dict[str, Any]) -> str | None:
    total = int(analytics_result.get("total", 0) or 0)
    if parsed.group_by == "colour" and parsed.include_classes:
        counts = _nonzero_counts(analytics_result.get("by_colour", {}))
        if not counts:
            return None
        class_label = _class_noun(parsed.include_classes[0], total=total)
        rows = "\n".join(f"{label.title()}: {count}" for label, count in _sorted_count_items(counts))
        top = analytics_result.get("top_colour")
        suffix = ""
        if isinstance(top, dict):
            suffix = f" {str(top.get('label')).title()} was the most common {class_label.rstrip('s')} colour."
        return f"The {total} {class_label} were:\n\n{rows}\n\n{suffix.strip()}"
    if parsed.group_by == "class":
        counts = _nonzero_counts(analytics_result.get("by_class", {}))
        if not counts:
            return None
        rows = "\n".join(f"{_class_label(label, count)}: {count}" for label, count in _sorted_count_items(counts))
        return f"Vehicle class breakdown:\n\n{rows}\n\nTotal: {total} vehicles."
    if parsed.group_by == "colour":
        counts = _nonzero_counts(analytics_result.get("by_colour", {}))
        if not counts:
            return None
        rows = "\n".join(f"{label.title()}: {count}" for label, count in _sorted_count_items(counts))
        return f"Vehicle colour breakdown:\n\n{rows}\n\nTotal: {total} vehicles."
    return None


def _format_natural_counts(counts: dict[str, int]) -> str:
    label_order = {label: index for index, label in enumerate((*COLOUR_COUNT_KEYS, *CLASS_COUNT_KEYS))}
    pieces = [
        f"{value} {key.lower()}"
        for key, value in sorted(counts.items(), key=lambda item: (-item[1], label_order.get(item[0], len(label_order)), item[0]))
    ]
    if len(pieces) <= 1:
        return pieces[0] if pieces else "none"
    return ", ".join(pieces[:-1]) + f", and {pieces[-1]}"


def _sorted_count_items(counts: dict[str, int]) -> list[tuple[str, int]]:
    if any(label in counts for label in ("MOTORCYCLE", "CAR", "3WHEELER", "TRUCK", "BUS")):
        display_order = ("MOTORCYCLE", "CAR", "3WHEELER", "TRUCK", "UNKNOWN", "BUS")
    else:
        display_order = (
            "BLACK",
            "WHITE",
            "GREEN",
            "RED",
            "SILVER",
            "BLUE",
            "GREY",
            "PINK",
            "YELLOW",
            "ORANGE",
            "BROWN",
            "BEIGE",
            "PURPLE",
            "OTHER",
            "UNKNOWN",
        )
    label_order = {label: index for index, label in enumerate(display_order)}
    return sorted(counts.items(), key=lambda item: (-item[1], label_order.get(item[0], len(label_order)), item[0]))


def _class_label(vehicle_class: str, count: int) -> str:
    labels = {
        "CAR": "Car" if count == 1 else "Cars",
        "MOTORCYCLE": "Motorcycle" if count == 1 else "Motorcycles",
        "3WHEELER": "Three-wheeler" if count == 1 else "Three-wheelers",
        "TRUCK": "Truck" if count == 1 else "Trucks",
        "BUS": "Bus" if count == 1 else "Buses",
        "UNKNOWN": "Unknown",
    }
    return labels.get(vehicle_class, vehicle_class.title())


def _format_evidence_page_text(parsed: ChatVehicleQuery, evidence_page: dict[str, Any] | None, *, evidence_count: int) -> str:
    if not parsed.show_evidence or not evidence_page:
        return ""
    total = int(evidence_page.get("matching_total", 0) or 0)
    shown = int(evidence_page.get("shown_count", 0) or 0)
    remaining = int(evidence_page.get("evidence_remaining_count", 0) or 0)
    returned = int(evidence_page.get("evidence_returned_count", evidence_count) or 0)
    offset = int(evidence_page.get("evidence_offset", 0) or 0)
    if total == 0:
        return ""
    if returned == 0 and remaining == 0 and offset >= total:
        return f" All {total} matching vehicles have already been shown."
    if offset > 0 and remaining == 0:
        return f" Showing {returned} more - {shown} of {total} shown. All {total} matching vehicles have now been shown."
    return f" Showing {returned} of {total}." if offset == 0 else f" Showing {returned} more - {shown} of {total} shown."


def _parse_rule_chat_query(text: str, context: dict[str, Any]) -> ChatVehicleQuery:
    unsupported_class = sorted(term for term in UNSUPPORTED_CLASS_TERMS if _contains_phrase(text, term))
    if unsupported_class:
        raise VehicleQueryParseError(f"Unsupported vehicle class term: {unsupported_class[0]}")
    unsupported_colour = sorted(term for term in UNSUPPORTED_COLOUR_TERMS if _contains_phrase(text, term))
    if unsupported_colour:
        raise VehicleQueryParseError(f"Unsupported colour term: {unsupported_colour[0]}")
    context_reference = _context_reference(text)
    if context_reference == "previous_results" and not context.get("previous_vehicle_ids"):
        context_reference = None
    previous_filters = dict(context.get("previous_filters", {}) or {}) if context_reference else {}
    explicit_mentions = _extract_explicit_filter_mentions(text)
    show_evidence = bool(re.search(r"\b(show|show me|find|which ones|let me see|want to see|see|evidence|display|show them)\b", text))
    include_classes = explicit_mentions.positive_classes or list(previous_filters.get("include_classes", []) or [])
    include_colours = explicit_mentions.positive_colours or list(previous_filters.get("include_colours", []) or [])
    exclude_classes = _dedupe(list(previous_filters.get("exclude_classes", []) or []) + explicit_mentions.negative_classes)
    exclude_colours = _dedupe(list(previous_filters.get("exclude_colours", []) or []) + explicit_mentions.negative_colours)
    start_time, end_time = parse_time_range(text)
    if context_reference and start_time is None and end_time is None:
        start_time = previous_filters.get("start_time")
        end_time = previous_filters.get("end_time")
    camera_id = _parse_camera_id(text) or previous_filters.get("camera_id")

    if _is_show_previous(text) and context.get("previous_vehicle_ids"):
        return ChatVehicleQuery(intent="LIST", show_evidence=True, context_reference="previous_results", evidence_navigation="next")
    if re.search(r"\bwhich\s+of\s+(those|them|these|ones)\b", text):
        return ChatVehicleQuery(
            intent="LIST",
            include_classes=include_classes,
            exclude_classes=exclude_classes,
            include_colours=include_colours,
            exclude_colours=exclude_colours,
            start_time=start_time,
            end_time=end_time,
            camera_id=camera_id,
            show_evidence=show_evidence,
            context_reference=context_reference,
        )
    if (include_classes or exclude_classes) and _is_colour_group_query(text):
        return ChatVehicleQuery(intent="GROUP", include_classes=include_classes, exclude_classes=exclude_classes, include_colours=include_colours, exclude_colours=exclude_colours, group_by="colour", sort_by="count_desc", context_reference=context_reference)
    if (include_colours or exclude_colours) and _is_class_group_query(text):
        return ChatVehicleQuery(intent="GROUP", include_classes=include_classes, exclude_classes=exclude_classes, include_colours=include_colours, exclude_colours=exclude_colours, group_by="class", sort_by="count_desc", context_reference=context_reference)
    if _is_class_wise_query(text):
        return ChatVehicleQuery(intent="GROUP", include_classes=include_classes, exclude_classes=exclude_classes, include_colours=include_colours, exclude_colours=exclude_colours, group_by="class")
    if _is_colour_wise_query(text):
        return ChatVehicleQuery(intent="GROUP", include_classes=include_classes, exclude_classes=exclude_classes, include_colours=include_colours, exclude_colours=exclude_colours, group_by="colour")
    if "most" in text and re.search(r"\b(vehicle\s+)?(class|type|category)\b", text):
        return ChatVehicleQuery(intent="GROUP", include_classes=include_classes, exclude_classes=exclude_classes, include_colours=include_colours, exclude_colours=exclude_colours, group_by="class", sort_by="count_desc", limit=1)
    if _is_summary_query(text):
        return ChatVehicleQuery(intent="SUMMARY", include_classes=include_classes, exclude_classes=exclude_classes, include_colours=include_colours, exclude_colours=exclude_colours, start_time=start_time, end_time=end_time, camera_id=camera_id, context_reference=context_reference)
    interval_query = _parse_interval_comparison_query(text)
    if interval_query is not None:
        return interval_query
    if re.search(r"\bmore\s+\w+\s+than\b", text) or "more cars than motorcycles" in text:
        classes = _find_labels(text, CLASS_SYNONYMS)
        if len(classes) < 2:
            raise VehicleQueryParseError("Compare queries need two vehicle classes.")
        return ChatVehicleQuery(intent="COMPARE", comparison={"left": classes[0], "right": classes[1]})
    if re.search(r"\b(types?|classes?|kinds?)\s+of\s+vehicles?\s+(?:are\s+)?present\b", text):
        return ChatVehicleQuery(intent="UNIQUE_CLASSES", include_classes=include_classes, exclude_classes=exclude_classes, include_colours=include_colours, exclude_colours=exclude_colours, start_time=start_time, end_time=end_time, camera_id=camera_id, context_reference=context_reference)
    if _is_colour_group_query(text) or ("most" in text and re.search(r"\bcolou?r\b", text)):
        if include_classes:
            return ChatVehicleQuery(intent="GROUP", include_classes=include_classes, exclude_classes=exclude_classes, include_colours=include_colours, exclude_colours=exclude_colours, group_by="colour", sort_by="count_desc", limit=1 if "most" in text else None)
        return ChatVehicleQuery(intent="UNIQUE_COLOURS", include_classes=include_classes, exclude_classes=exclude_classes, include_colours=include_colours, exclude_colours=exclude_colours, start_time=start_time, end_time=end_time, camera_id=camera_id, context_reference=context_reference)
    if re.search(r"\b(group|breakdown)\s+by\s+class\b", text):
        return ChatVehicleQuery(intent="GROUP", include_classes=include_classes, exclude_classes=exclude_classes, include_colours=include_colours, exclude_colours=exclude_colours, group_by="class")
    if re.search(r"\b(group|breakdown)\s+by\s+colou?r\b", text):
        return ChatVehicleQuery(intent="GROUP", include_classes=include_classes, exclude_classes=exclude_classes, include_colours=include_colours, exclude_colours=exclude_colours, group_by="colour")
    if include_classes and re.fullmatch(r"(unknown|unclassified)\s+vehicles?", text):
        return ChatVehicleQuery(intent="LIST", include_classes=include_classes, exclude_classes=exclude_classes, include_colours=include_colours, exclude_colours=exclude_colours, show_evidence=True)
    if show_evidence or re.search(r"\b(which vehicles|appeared|list|display|want to see)\b", text):
        return ChatVehicleQuery(
            intent="LIST",
            include_classes=include_classes,
            exclude_classes=exclude_classes,
            include_colours=include_colours,
            exclude_colours=exclude_colours,
            start_time=start_time,
            end_time=end_time,
            camera_id=camera_id,
            show_evidence=True,
            sort_by=None,
            limit=None,
            context_reference=context_reference,
        )
    if re.search(r"\b(how many|count|number of|total)\b", text):
        return ChatVehicleQuery(
            intent="COUNT",
            include_classes=include_classes,
            exclude_classes=exclude_classes,
            include_colours=include_colours,
            exclude_colours=exclude_colours,
            start_time=start_time,
            end_time=end_time,
            camera_id=camera_id,
            context_reference=context_reference,
        )
    if include_classes or exclude_classes or include_colours or exclude_colours:
        return ChatVehicleQuery(
            intent="LIST",
            include_classes=include_classes,
            exclude_classes=exclude_classes,
            include_colours=include_colours,
            exclude_colours=exclude_colours,
            start_time=start_time,
            end_time=end_time,
            camera_id=camera_id,
            show_evidence=True,
            context_reference=context_reference,
        )
    raise VehicleQueryParseError("Could not determine chat query intent.")


def _parse_general_chat_query(text: str) -> ChatVehicleQuery | None:
    if re.fullmatch(r"(hello|hi|hey|good\s+(morning|afternoon|evening)|thanks|thank\s+you|who\s+are\s+you\??|what\s+can\s+you\s+do\??)", text):
        return ChatVehicleQuery(intent="GENERAL_CHAT")
    return None


def _is_summary_query(text: str) -> bool:
    return bool(re.search(r"\b(summ?ary|summ?ry|summarize|summarise|overview|breakdown)\b", text))


def _classify_message_type(text: str, context: dict[str, Any]) -> str:
    if _parse_general_chat_query(text) is not None:
        return "GENERAL_CHAT"
    if _parse_evidence_navigation_query(text, context) is not None:
        return "NAVIGATION"
    if _context_reference(text):
        return "FOLLOW_UP"
    return "NEW_ANALYTICS_QUERY"


def _is_class_wise_query(text: str) -> bool:
    return bool(
        re.search(r"\bclass[\s-]*wise\b", text)
        or re.search(r"\b(vehicle\s+)?(class|category|type)s?\s+counts?\b", text)
        or re.search(r"\b(all\s+)?vehicles?\s+(class|category|type)[\s-]*wise\b", text)
        or re.search(r"\bbreak\s*down\s+vehicles?\s+by\s+(class|category|type)\b", text)
        or re.search(r"\bwhat\s+(types|kinds|classes)\s+of\s+vehicles?\s+(are\s+there|were\s+there).*\b(how\s+many|count|numbers?)\b", text)
    )


def _is_colour_wise_query(text: str) -> bool:
    return bool(
        re.search(r"\bcolou?r[\s-]*wise\b", text)
        or re.search(r"\b(vehicle\s+)?colou?rs?\s+breakdown\b", text)
        or re.search(r"\b(vehicle\s+)?colou?r\s+counts?\b", text)
        or re.search(r"\bbreak\s*down\s+vehicles?\s+by\s+colou?r\b", text)
    )


def _is_colour_group_query(text: str) -> bool:
    return bool(
        re.search(r"\b(colou?rs?|colou?r)\s+(?:of|were|was|are|present)\b", text)
        or re.search(r"\b(colou?r|colou?rs?)\s+breakdown\b", text)
        or re.search(r"\bshow\s+.+\s+colou?rs?\b", text)
    )


def _is_class_group_query(text: str) -> bool:
    return bool(
        re.search(r"\b(vehicle\s+)?(classes|class|types|type|categories|category)\s+(?:of|were|was|are)\b", text)
        or re.search(r"\b(vehicle\s+)?(classes|class|types|type|categories|category)\s+breakdown\b", text)
        or re.search(r"\bshow\s+.+\s+(classes|types|categories)\b", text)
    )


def _explicit_filters_detected(parsed: ChatVehicleQuery) -> dict[str, bool]:
    return {
        "classes": bool(parsed.include_classes or parsed.exclude_classes),
        "colours": bool(parsed.include_colours or parsed.exclude_colours),
        "time": parsed.start_time is not None or parsed.end_time is not None,
        "camera": parsed.camera_id is not None,
    }


def _parse_evidence_navigation_query(text: str, context: dict[str, Any]) -> ChatVehicleQuery | None:
    if not context.get("previous_vehicle_ids"):
        return None
    previous_filters = dict(context.get("previous_filters", {}) or {})
    classes = list(previous_filters.get("include_classes", []) or [])
    exclude_classes = list(previous_filters.get("exclude_classes", []) or [])
    colours = list(previous_filters.get("include_colours", []) or [])
    exclude_colours = list(previous_filters.get("exclude_colours", []) or [])
    if re.fullmatch(r"(show|display|list)\s+(from\s+the\s+beginning|again|all)", text):
        return ChatVehicleQuery(intent="LIST", include_classes=classes, exclude_classes=exclude_classes, include_colours=colours, exclude_colours=exclude_colours, show_evidence=True, context_reference="previous_results", evidence_navigation="restart")
    if re.fullmatch(r"(show\s+)?(more|next|next\s+\d+|remaining|the\s+rest|rest|the\s+other\s+\d+|other\s+\d+)", text):
        return ChatVehicleQuery(intent="LIST", include_classes=classes, exclude_classes=exclude_classes, include_colours=colours, exclude_colours=exclude_colours, show_evidence=True, context_reference="previous_results", evidence_navigation="next")
    if re.fullmatch(r"(show|display|list)\s+(them|those|these|ones|evidence)", text):
        return ChatVehicleQuery(intent="LIST", include_classes=classes, exclude_classes=exclude_classes, include_colours=colours, exclude_colours=exclude_colours, show_evidence=True, context_reference="previous_results", evidence_navigation="next")
    if re.fullmatch(r"show\s+me\s+(the\s+)?(other\s+\d+|remaining|rest|evidence|them|those)", text):
        return ChatVehicleQuery(intent="LIST", include_classes=classes, exclude_classes=exclude_classes, include_colours=colours, exclude_colours=exclude_colours, show_evidence=True, context_reference="previous_results", evidence_navigation="next")
    return None


def _parse_interval_comparison_query(text: str) -> ChatVehicleQuery | None:
    if not re.search(r"\b(when|time|times|duration|period|interval)\b", text):
        return None
    classes = _find_labels(text, CLASS_SYNONYMS)
    if len(classes) < 2:
        return None
    operator = ">"
    if re.search(r"\b(equal|same)\b", text):
        operator = "="
    elif re.search(r"\bless\s+than|fewer\s+than\b", text):
        operator = "<"
    elif re.search(r"\bmore\s+than|more\s+common\s+than\b", text):
        operator = ">"
    if "cars are more than bikes" in text or "cars more than bikes" in text or "cars were more than" in text:
        left, right = "CAR", "MOTORCYCLE"
    elif "bikes are more than cars" in text or "bikes more than cars" in text or "motorcycles more than cars" in text:
        left, right = "MOTORCYCLE", "CAR"
    else:
        left, right = classes[0], classes[1]
    return ChatVehicleQuery(
        intent="FIND_INTERVALS",
        comparison={"left": left, "operator": operator, "right": right, "window_seconds": 5.0},
    )


def normalize_llm_vehicle_query(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise VehicleQueryParseError("LLM returned an invalid schema.")
    intent = str(payload.get("intent") or "").upper()
    if not intent:
        raise VehicleQueryParseError("missing_required:intent")
    normalized = {
        "intent": intent,
        "classes": _normalize_llm_class_list(payload.get("class_include", payload.get("classes", payload.get("include_classes", [])))),
        "exclude_classes": _normalize_llm_class_list(payload.get("class_exclude", payload.get("exclude_classes", []))),
        "colours": _normalize_llm_colour_list(payload.get("colour_include", payload.get("color_include", payload.get("colours", payload.get("include_colours", []))))),
        "exclude_colours": _normalize_llm_colour_list(payload.get("colour_exclude", payload.get("color_exclude", payload.get("exclude_colours", [])))),
        "start_time": _optional_float(payload.get("start_time")),
        "end_time": _optional_float(payload.get("end_time")),
        "group_by": _normalize_llm_group_by(payload.get("group_by")),
        "operator": _normalize_operator(payload.get("operator")),
        "show_evidence": bool(payload.get("show_evidence")),
        "context_reference": _normalize_llm_context_reference(payload.get("context_reference")),
    }
    return normalized


def _repair_llm_plan_with_explicit_mentions(payload: dict[str, Any], mentions: ExplicitFilterMentions) -> tuple[dict[str, Any], bool, str]:
    repaired = {**payload}
    class_include = list(repaired.get("classes", []) or [])
    class_exclude = list(repaired.get("exclude_classes", []) or [])
    colour_include = list(repaired.get("colours", []) or [])
    colour_exclude = list(repaired.get("exclude_colours", []) or [])
    repair_applied = False

    for label in mentions.negative_classes:
        if label in class_include:
            class_include = [item for item in class_include if item != label]
            repair_applied = True
        if label not in class_exclude:
            class_exclude.append(label)
            repair_applied = True
    for label in mentions.negative_colours:
        if label in colour_include:
            colour_include = [item for item in colour_include if item != label]
            repair_applied = True
        if label not in colour_exclude:
            colour_exclude.append(label)
            repair_applied = True
    for label in mentions.positive_classes:
        if label not in class_include and label not in class_exclude:
            class_include.append(label)
            repair_applied = True
    for label in mentions.positive_colours:
        if label not in colour_include and label not in colour_exclude:
            colour_include.append(label)
            repair_applied = True

    intent = str(repaired.get("intent") or "").upper()
    if intent == "UNIQUE_COLOURS" and (class_include or class_exclude):
        repaired["intent"] = "GROUP"
        repaired["group_by"] = "colour"
        repair_applied = True
    if intent == "UNIQUE_CLASSES" and (colour_include or colour_exclude):
        repaired["intent"] = "GROUP"
        repaired["group_by"] = "vehicle_class"
        repair_applied = True

    repaired["classes"] = _dedupe(class_include)
    repaired["exclude_classes"] = _dedupe(class_exclude)
    repaired["colours"] = _dedupe(colour_include)
    repaired["exclude_colours"] = _dedupe(colour_exclude)
    return repaired, repair_applied, "repaired" if repair_applied else "accepted"


def chat_query_from_llm_vehicle_query(payload: dict[str, Any], *, text: str, context: dict[str, Any], explicit_mentions: ExplicitFilterMentions | None = None) -> ChatVehicleQuery:
    _reject_suspicious_llm_payload(payload, text, context, explicit_mentions=explicit_mentions)
    classes = list(payload.get("classes", []) or [])
    colours = list(payload.get("colours", []) or [])
    context_reference = payload.get("context_reference")
    if context_reference is not None and not _context_reference(text):
        context_reference = None
        payload = {**payload, "context_reference": None}
    if context_reference == "previous_filters":
        context_reference = None
        payload = {**payload, "context_reference": None}
    if context_reference == "previous_result" and not context.get("previous_vehicle_ids"):
        context_reference = None
        payload = {**payload, "context_reference": None}
    if context_reference == "previous_result":
        previous_filters = dict(context.get("previous_filters", {}) or {})
        if not classes:
            classes = list(previous_filters.get("include_classes", []) or [])
        if not colours:
            colours = list(previous_filters.get("include_colours", []) or [])
    internal_context_reference = "previous_results" if context_reference == "previous_result" else context_reference
    comparison = _comparison_from_llm(payload)
    if str(payload["intent"]).upper() in {"COMPARE", "FIND_INTERVALS"} and comparison is None:
        raise VehicleQueryParseError("invalid_comparison:need_two_classes")
    sort_by = "count_desc" if payload.get("intent") == "GROUP" and "most" in text else None
    limit = 1 if sort_by else None
    return ChatVehicleQuery(
        intent=str(payload["intent"]).upper(),
        include_classes=classes,
        exclude_classes=list(payload.get("exclude_classes", []) or []),
        include_colours=colours,
        exclude_colours=list(payload.get("exclude_colours", []) or []),
        start_time=payload.get("start_time"),
        end_time=payload.get("end_time"),
        camera_id=None,
        group_by=_normalize_group_by(payload.get("group_by")),
        comparison=comparison,
        sort_by=sort_by,
        limit=limit,
        show_evidence=bool(payload.get("show_evidence")),
        context_reference=internal_context_reference,
    )


def _filter_records(
    records: list[VehicleRecord],
    *,
    include_classes: list[str] | None = None,
    exclude_classes: list[str] | None = None,
    include_colours: list[str] | None = None,
    exclude_colours: list[str] | None = None,
    start_time: float | None = None,
    end_time: float | None = None,
    camera_id: str | None = None,
    base_ids: set[str] | None = None,
) -> list[VehicleRecord]:
    include_classes_set = {item.upper() for item in include_classes or []}
    exclude_classes_set = {item.upper() for item in exclude_classes or []}
    include_colours_set = {item.upper() for item in include_colours or []}
    exclude_colours_set = {item.upper() for item in exclude_colours or []}
    return [
        record
        for record in records
        if (base_ids is None or record.vehicle_id in base_ids)
        and (not include_classes_set or record.vehicle_class in include_classes_set)
        and record.vehicle_class not in exclude_classes_set
        and (not include_colours_set or record.colour in include_colours_set)
        and record.colour not in exclude_colours_set
        and (camera_id is None or record.camera_id == str(camera_id))
        and not (start_time is not None and record.last_seen_seconds is not None and record.last_seen_seconds < start_time)
        and not (end_time is not None and record.first_seen_seconds is not None and record.first_seen_seconds > end_time)
    ]


def _summary_payload(records: list[VehicleRecord]) -> dict[str, Any]:
    first_values = [record.first_seen_seconds for record in records if record.first_seen_seconds is not None]
    last_values = [record.last_seen_seconds for record in records if record.last_seen_seconds is not None]
    return {
        "total_unique_vehicles": len(records),
        "vehicle_classes": count_by_class(records),
        "colours": count_by_colour(records),
        "first_seen_seconds": min(first_values) if first_values else None,
        "last_seen_seconds": max(last_values) if last_values else None,
        "vehicle_ids": [record.vehicle_id for record in records],
    }


def _query_filters(parsed: ChatVehicleQuery) -> dict[str, Any]:
    return {
        "include_classes": list(parsed.include_classes),
        "exclude_classes": list(parsed.exclude_classes),
        "include_colours": list(parsed.include_colours),
        "exclude_colours": list(parsed.exclude_colours),
        "start_time": parsed.start_time,
        "end_time": parsed.end_time,
        "camera_id": parsed.camera_id,
    }


def _payload_filters(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "include_classes": list(payload.get("classes", []) or []),
        "exclude_classes": list(payload.get("exclude_classes", []) or []),
        "include_colours": list(payload.get("colours", []) or []),
        "exclude_colours": list(payload.get("exclude_colours", []) or []),
        "start_time": payload.get("start_time"),
        "end_time": payload.get("end_time"),
        "camera_id": None,
    }


def _reject_suspicious_llm_payload(payload: dict[str, Any], text: str, context: dict[str, Any], *, explicit_mentions: ExplicitFilterMentions | None = None) -> None:
    if not isinstance(payload, dict):
        raise VehicleQueryParseError("LLM returned an invalid schema.")
    intent = str(payload.get("intent") or "").upper()
    if intent not in SUPPORTED_CHAT_INTENTS:
        raise VehicleQueryParseError(f"unsupported_intent:{intent}")
    if _parse_general_chat_query(text) is not None and intent != "GENERAL_CHAT":
        raise VehicleQueryParseError("incorrect_intent:expected_GENERAL_CHAT")
    explicit_mentions = explicit_mentions or _extract_explicit_filter_mentions(text)
    mentioned_classes = _dedupe(explicit_mentions.positive_classes + explicit_mentions.negative_classes)
    mentioned_colours = _dedupe(explicit_mentions.positive_colours + explicit_mentions.negative_colours)
    if _is_class_wise_query(text):
        if intent != "GROUP" or _normalize_group_by(payload.get("group_by")) != "class" or (not mentioned_colours and _upper_list(payload.get("colours"))):
            raise VehicleQueryParseError("incorrect_group_by:expected_class")
    if _is_colour_wise_query(text):
        if intent != "GROUP" or _normalize_group_by(payload.get("group_by")) != "colour" or (not mentioned_classes and _upper_list(payload.get("classes"))):
            raise VehicleQueryParseError("incorrect_group_by:expected_colour")
    if payload.get("context_reference") is not None and not _context_reference(text):
        raise VehicleQueryParseError("invented_context_reference")
    if (payload.get("start_time") == 0 or payload.get("end_time") == 0) and not re.search(r"\b(0|zero|first|between|after|before|from|to|\d)\b", text):
        raise VehicleQueryParseError("invalid_time:invented_zero")
    if re.search(r"\bhow many\b", text) and not re.search(r"\bmore\s+\w+\s+than\b|more common than", text) and intent != "COUNT":
        raise VehicleQueryParseError("incorrect_intent:expected_COUNT")
    is_interval_text = bool(re.search(r"\b(when|time|times|duration|period|interval)\b", text))
    if re.search(r"\bmore common than\b|\bmore\s+\w+\s+than\b", text) and not is_interval_text and intent != "COMPARE":
        raise VehicleQueryParseError("incorrect_intent:expected_COMPARE")
    if is_interval_text and re.search(r"\bmore\s+than|more\s+common\s+than|equal|same\b", text) and intent != "FIND_INTERVALS":
        raise VehicleQueryParseError("incorrect_intent:expected_FIND_INTERVALS")
    if re.search(r"\b(auto|autos|rickshaw|rickshaws|three wheeler|three wheelers)\b", text):
        include_classes = _upper_list(payload.get("classes"))
        exclude_classes = _upper_list(payload.get("exclude_classes"))
        if "3WHEELER" not in include_classes or "3WHEELER" in exclude_classes:
            raise VehicleQueryParseError("wrong_synonym:auto_to_3WHEELER")
    if "most" in text and re.search(r"\bcolou?r\b", text):
        if intent != "GROUP" or _normalize_group_by(payload.get("group_by")) != "colour":
            raise VehicleQueryParseError("incorrect_group_by:expected_colour")
    if re.search(EXCLUSION_PHRASE_PATTERN, text):
        if (explicit_mentions.negative_classes and not _upper_list(payload.get("exclude_classes"))) or (explicit_mentions.negative_colours and not _upper_list(payload.get("exclude_colours"))):
            raise VehicleQueryParseError("missing_exclusion")
    include_classes = _upper_list(payload.get("classes"))
    exclude_classes = _upper_list(payload.get("exclude_classes"))
    if re.search(r"\bvehicles?\b", text) and not mentioned_classes and include_classes:
        raise VehicleQueryParseError("invented_class_for_generic_vehicle_query")
    for label in explicit_mentions.positive_classes:
        if label not in include_classes:
            raise VehicleQueryParseError(f"missed_class:{label}")
    for label in explicit_mentions.negative_classes:
        if label not in exclude_classes:
            raise VehicleQueryParseError(f"missed_class:{label}")
    include_colours = _upper_list(payload.get("colours"))
    exclude_colours = _upper_list(payload.get("exclude_colours"))
    for label in explicit_mentions.positive_colours:
        if label not in include_colours:
            raise VehicleQueryParseError(f"missed_colour:{label}")
    for label in explicit_mentions.negative_colours:
        if label not in exclude_colours:
            raise VehicleQueryParseError(f"missed_colour:{label}")


def _top_count(counts: dict[str, int]) -> dict[str, Any] | None:
    nonzero = _nonzero_counts(counts)
    if not nonzero:
        return None
    label, count = max(nonzero.items(), key=lambda item: item[1])
    return {"label": label, "count": count}


def _normalize_llm_class_list(value: Any) -> list[str]:
    labels: list[str] = []
    for item in list(value or []):
        raw = str(item).strip().upper().replace(" ", "_")
        normalized = LLM_CLASS_ALIASES.get(raw, raw)
        if normalized not in (*SUPPORTED_VEHICLE_CLASSES, "UNKNOWN"):
            raise VehicleQueryParseError(f"unsupported_class:{raw}")
        labels.append(normalized)
    return _dedupe(labels)


def _normalize_llm_colour_list(value: Any) -> list[str]:
    labels: list[str] = []
    for item in list(value or []):
        raw = str(item).strip().upper().replace(" ", "_")
        normalized = LLM_COLOUR_ALIASES.get(raw, raw)
        if normalized not in SUPPORTED_VEHICLE_COLOUR_LABELS:
            raise VehicleQueryParseError(f"unsupported_colour:{raw}")
        labels.append(normalized)
    return _dedupe(labels)


def _normalize_llm_group_by(value: Any) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip().lower()
    if normalized in {"vehicle_class", "class"}:
        return "vehicle_class"
    if normalized in {"colour", "color", "vehicle_colour", "vehicle_color"}:
        return "colour"
    raise VehicleQueryParseError(f"invalid_group_by:{value}")


def _normalize_llm_context_reference(value: Any) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip().lower()
    if normalized in {"previous_result", "previous_results"}:
        return "previous_result"
    if normalized == "previous_filters":
        return "previous_filters"
    raise VehicleQueryParseError(f"invalid_context_reference:{value}")


def _normalize_operator(value: Any) -> str | None:
    if value is None:
        return None
    operator = str(value).strip()
    if operator not in {">", "<", "="}:
        raise VehicleQueryParseError(f"invalid_operator:{value}")
    return operator


def _comparison_from_llm(payload: dict[str, Any]) -> dict[str, Any] | None:
    intent = str(payload.get("intent") or "").upper()
    if intent not in {"COMPARE", "FIND_INTERVALS"}:
        return None
    classes = list(payload.get("classes", []) or [])
    if len(classes) >= 2:
        comparison = {"left": classes[0], "right": classes[1]}
        if intent == "FIND_INTERVALS":
            comparison["operator"] = payload.get("operator") or ">"
            comparison["window_seconds"] = 5.0
        return comparison
    return None


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    results: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        results.append(value)
    return results


def _llm_context(context: dict[str, Any]) -> dict[str, Any]:
    return {
        "previous_query": context.get("previous_query"),
        "previous_filters": context.get("previous_filters"),
        "previous_vehicle_ids": list(context.get("previous_vehicle_ids", []) or [])[:20],
    }


def _diagnostic_reason(exc: Exception) -> str:
    return str(exc) or exc.__class__.__name__


def _extract_explicit_filter_mentions(text: str) -> ExplicitFilterMentions:
    mentions = _find_label_mentions(text)
    positive_classes: list[str] = []
    negative_classes: list[str] = []
    positive_colours: list[str] = []
    negative_colours: list[str] = []
    for mention in mentions:
        label = str(mention["label"])
        target = (
            negative_classes
            if mention["dimension"] == "class" and _is_negative_mention(text, int(mention["start"]))
            else positive_classes
            if mention["dimension"] == "class"
            else negative_colours
            if _is_negative_mention(text, int(mention["start"]))
            else positive_colours
        )
        if label not in target:
            target.append(label)
    return ExplicitFilterMentions(
        positive_classes=positive_classes,
        negative_classes=negative_classes,
        positive_colours=positive_colours,
        negative_colours=negative_colours,
    )


def _find_label_mentions(text: str) -> list[dict[str, Any]]:
    mentions: list[dict[str, Any]] = []
    for label, phrases in CLASS_SYNONYMS.items():
        for phrase in phrases:
            match = _phrase_match(text, phrase)
            if match is not None:
                mentions.append({"start": match.start(), "end": match.end(), "label": label, "dimension": "class"})
                break
    for label, phrases in COLOUR_SYNONYMS.items():
        for phrase in phrases:
            match = _phrase_match(text, phrase)
            if match is not None:
                mentions.append({"start": match.start(), "end": match.end(), "label": label, "dimension": "colour"})
                break
    mentions.sort(key=lambda item: int(item["start"]))
    return mentions


def _phrase_match(text: str, phrase: str) -> re.Match[str] | None:
    normalized_phrase = re.escape(_normalize(phrase)).replace(r"\ ", r"[\s-]+")
    return re.search(rf"(?<!\w){normalized_phrase}(?!\w)", text)


def _is_negative_mention(text: str, mention_start: int) -> bool:
    prefix = text[:mention_start]
    negations = list(re.finditer(EXCLUSION_PHRASE_PATTERN, prefix))
    return bool(negations)


def _find_labels(text: str, synonyms: dict[str, tuple[str, ...]]) -> list[str]:
    matches: list[tuple[int, str]] = []
    for label, phrases in synonyms.items():
        positions = [text.find(_normalize(phrase)) for phrase in phrases if _contains_phrase(text, phrase)]
        if positions:
            matches.append((min(position for position in positions if position >= 0), label))
    return [label for _, label in sorted(matches)]


def _contains_phrase(text: str, phrase: str) -> bool:
    normalized_phrase = re.escape(_normalize(phrase)).replace(r"\ ", r"[\s-]+")
    return bool(re.search(rf"(?<!\w){normalized_phrase}(?!\w)", text))


def _context_reference(text: str) -> str | None:
    if re.search(r"\b(those|them|that|these|ones)\b", text):
        return "previous_results"
    return None


def _is_show_previous(text: str) -> bool:
    return bool(re.fullmatch(r"(show|list|display)\s+(them|those|these|ones)", text))


def _query_description(parsed: ChatVehicleQuery, total: int) -> str:
    parts: list[str] = []
    if parsed.include_colours:
        parts.append(" or ".join(colour.lower() for colour in parsed.include_colours))
    if parsed.include_classes:
        nouns = [_class_noun(label, total=total) for label in parsed.include_classes]
        parts.append(" or ".join(nouns))
    else:
        parts.append("vehicles")
    description = " ".join(parts)
    exclusions: list[str] = []
    if parsed.exclude_classes:
        exclusions.append("excluding " + " or ".join(_class_noun(label, total=2) for label in parsed.exclude_classes))
    if parsed.exclude_colours:
        exclusions.append("excluding " + " or ".join(colour.lower() for colour in parsed.exclude_colours))
    if exclusions:
        description += " " + " and ".join(exclusions)
    if parsed.start_time is not None and parsed.end_time is not None:
        description += f" between {parsed.start_time:.1f} and {parsed.end_time:.1f} seconds"
    elif parsed.start_time is not None:
        description += f" after {parsed.start_time:.1f} seconds"
    elif parsed.end_time is not None:
        description += f" before {parsed.end_time:.1f} seconds"
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
    return {"bus": "buses", "3wheeler": "3wheelers", "unknown vehicle": "unknown vehicles"}.get(singular, f"{singular}s")


def _format_counts(counts: dict[str, int]) -> str:
    return ", ".join(f"{key} {value}" for key, value in counts.items()) if counts else "none"


def _nonzero_counts(counts: Any) -> dict[str, int]:
    return {str(key): int(value) for key, value in dict(counts or {}).items() if int(value or 0) > 0}


def _split_vehicle_id(vehicle_id: str) -> tuple[str | None, str | None]:
    parts = str(vehicle_id).split(":", 1)
    if len(parts) != 2:
        return None, None
    return parts[0], parts[1]


def _media_url(media: Any) -> str | None:
    if not isinstance(media, dict):
        return None
    category = str(media.get("category") or "").strip()
    run_id = str(media.get("run_id") or "").strip()
    parts = [str(item).strip() for item in list(media.get("parts", []) or []) if str(item).strip()]
    if not category or not run_id or not parts:
        return None
    return f"/api/media/{category}/{run_id}/{'/'.join(parts)}"


def _parse_camera_id(text: str) -> str | None:
    match = re.search(r"\b(cam_\d+|camera\s+\d+)\b", text)
    if not match:
        return None
    return match.group(1).replace("camera", "cam").replace(" ", "_").upper()


def _upper_list(value: Any) -> list[str]:
    return [str(item).upper() for item in list(value or [])]


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _normalize_group_by(value: Any) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip().lower()
    if normalized in {"vehicle_class", "class", "classes"}:
        return "class"
    if normalized in {"vehicle_colour", "vehicle_color", "colour", "color", "colours", "colors"}:
        return "colour"
    return normalized


def _normalize(value: str) -> str:
    text = str(value or "").strip().lower().replace("colour", "color")
    text = re.sub(r"[?,.!]", " ", text)
    return " ".join(text.split())
