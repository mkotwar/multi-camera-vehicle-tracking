from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import yaml

from src.importers.models import (
    ColourPredictionRow,
    DryRunReport,
    DryRunRows,
    LogicalTrackRef,
    MediaAssetRow,
    IdentityDecisionRow,
    PlateDetectionRow,
    PlateReadingRow,
    PhysicalVehicleRow,
    PhysicalVehicleTrackRow,
    ProcessingRunRow,
    RunCameraRow,
    TrackEvidenceRow,
    TrackObservationRow,
    ValidationIssue,
    VehicleAttributePredictionRow,
    VehicleTrackRow,
)
from src.importers.validation import count_by_severity, verdict_from_issues


RUN_TYPED_FIELDS = {
    "run_id",
    "status",
    "project_name",
    "started_at",
    "completed_at",
    "config_path",
    "detection_backend",
    "tracking_backend",
    "vehicle_enrichment_enabled",
    "processed_frames",
    "raw_yolo_detections",
    "roi_filtered_detections",
    "completed_tracks",
    "error_count",
}
TRACK_TYPED_FIELDS = {
    "local_track_id",
    "camera_id",
    "tracker_namespace",
    "native_tracker_id",
    "status",
    "completion_reason",
    "first_frame",
    "last_frame",
    "first_timestamp_seconds",
    "last_timestamp_seconds",
    "observation_count",
    "lost_frames",
    "final_class",
    "class_counts",
    "class_confidence_sums",
}
OBSERVATION_TYPED_FIELDS = {
    "local_track_id",
    "camera_id",
    "tracker_namespace",
    "native_tracker_id",
    "frame_number",
    "timestamp_seconds",
    "x1",
    "y1",
    "x2",
    "y2",
    "confidence",
    "raw_class_id",
    "raw_class_name",
}
EVIDENCE_TYPED_FIELDS = {
    "local_track_id",
    "camera_id",
    "role",
    "evidence_role",
    "frame_number",
    "timestamp_seconds",
    "bbox_xyxy",
    "original_bbox",
    "original_bbox_xyxy",
    "expanded_crop_bbox",
    "expanded_crop_bbox_xyxy",
    "confidence",
    "detection_confidence",
    "best_overall_score",
    "quality_score",
    "sharpness_score",
    "brightness_score",
    "crop_width",
    "crop_height",
    "original_crop_width",
    "original_crop_height",
    "resolution_tier",
    "selected_for_colour",
    "selected_for_body_type",
    "evidence_source",
    "candidate_rank",
    "crop_path",
    "vehicle_crop_path",
    "source_image_path",
    "source_frame_path",
    "annotated_frame_path",
}
ENRICHMENT_FINAL_FIELDS = {
    "local_track_id",
    "camera_id",
    "vehicle_class",
    "vehicle_class_confidence",
    "vehicle_colour",
    "vehicle_body_type",
    "plate_detected",
    "plate_text",
    "plate_colour",
    "registration_category",
}


def load_run_files(run_dir: Path) -> dict[str, Any]:
    return {
        "metadata": _read_json(run_dir / "run_metadata.json", default={}),
        "summary": _read_json(run_dir / "summary.json", default={}),
        "config": _read_yaml(run_dir / "run_config.yaml"),
        "tracks": _read_json(run_dir / "tracks.json", default=[]),
        "observations": _read_csv(run_dir / "observations.csv"),
        "evidence": _read_json(run_dir / "evidence_index.json", default=[]),
        "enrichment": _read_json(run_dir / "vehicle_enrichment.json", default=[]),
        "physical_vehicles": _read_json(run_dir / "physical_vehicles.json", default={}),
        "identity_decisions": _read_json(run_dir / "identity_decisions.json", default=[]),
    }


def build_dry_run(run_dir: str | Path) -> DryRunReport:
    run_path = Path(run_dir).resolve()
    rows = DryRunRows()
    issues: list[ValidationIssue] = []
    normalizations: list[dict[str, Any]] = []
    media_seen: set[tuple[str, str | None, str | None]] = set()
    source = _load_run_files_for_report(run_path, issues)
    run_key = str(source["metadata"].get("run_id") or source["summary"].get("run_id") or run_path.name)

    rows.processing_runs.append(_map_processing_run(run_key, source, run_path))
    rows.run_cameras.extend(_map_run_cameras(run_key, source))

    known_refs: dict[str, LogicalTrackRef] = {}
    track_candidates: dict[tuple[str, str], list[tuple[LogicalTrackRef, int | None, int | None, str | None]]] = defaultdict(list)
    track_occurrences = Counter((_str_or_none(track.get("camera_id")) or "UNKNOWN_CAMERA", _str_or_none(track.get("local_track_id")) or "UNKNOWN_TRACK") for track in source["tracks"])
    completed_occurrences = Counter(
        (_str_or_none(track.get("camera_id")) or "UNKNOWN_CAMERA", _str_or_none(track.get("local_track_id")) or "UNKNOWN_TRACK")
        for track in source["tracks"]
        if str(track.get("status") or "").upper() == "COMPLETED"
    )
    completed_seen: set[tuple[str, str]] = set()
    track_refs: list[tuple[dict[str, Any], LogicalTrackRef]] = []
    duplicate_tracks = 0
    for index, track in enumerate(source["tracks"]):
        camera_key = _str_or_none(track.get("camera_id")) or "UNKNOWN_CAMERA"
        original_local = _str_or_none(track.get("local_track_id")) or "UNKNOWN_TRACK"
        identity = (camera_key, original_local)
        status = str(track.get("status") or "").upper()
        local_for_import = original_local
        if track_occurrences[identity] > 1:
            if status == "COMPLETED" and identity not in completed_seen and completed_occurrences[identity] == 1:
                completed_seen.add(identity)
            else:
                duplicate_tracks += 1
                if status == "COMPLETED":
                    issues.append(
                        ValidationIssue(
                            "ERROR",
                            "duplicate_completed_logical_track_identity",
                            "Multiple completed tracks share one logical track identity.",
                            {
                                "rule": "vehicle_tracks_unique_completed_logical_identity",
                                "table": "vehicle_tracks",
                                "entity": "vehicle_track",
                                "track_key": f"{run_key}|{camera_key}|{original_local}",
                                "expected": "at most one COMPLETED row for a run/camera/local_track_id",
                                "actual": f"{completed_occurrences[identity]} COMPLETED rows",
                            },
                        )
                    )
                local_for_import = f"{original_local}__DUPLICATE_{index + 1}_{status or 'UNKNOWN'}"
                issues.append(
                    ValidationIssue(
                        "INFO",
                        "duplicate_noncanonical_track_identity_remapped",
                        "A duplicate noncanonical raw track was assigned an import-only logical ID.",
                        {
                            "rule": "vehicle_tracks_unique_logical_identity",
                            "table": "vehicle_tracks",
                            "entity": "vehicle_track",
                            "original_track_key": f"{run_key}|{camera_key}|{original_local}",
                            "import_track_key": f"{run_key}|{camera_key}|{local_for_import}",
                            "expected": "unique run/camera/local_track_id for DB import",
                            "actual": f"{track_occurrences[identity]} raw rows share the original ID",
                            "status": status,
                        },
                    )
                )
        ref = _track_ref(run_key, camera_key, local_for_import)
        known_refs[ref.key] = ref
        track_candidates[identity].append((ref, _int_or_none(track.get("first_frame")), _int_or_none(track.get("last_frame")), status))
        track_refs.append((track, ref))

    enrichment_by_track = _index_by_track(source["enrichment"], run_key, track_candidates)
    for track, ref in track_refs:
        rows.vehicle_tracks.append(_map_vehicle_track(ref, track, enrichment_by_track.get(ref.key), normalizations))

    _map_observations(run_key, source["observations"], known_refs, rows, issues, track_candidates)
    _map_evidence(run_key, source["evidence"], known_refs, rows, issues, run_path, media_seen, track_candidates)
    _map_enrichment(run_key, source["enrichment"], known_refs, rows, issues, normalizations, run_path, media_seen, track_candidates)
    _map_physical_identity(run_key, source, known_refs, rows, issues)
    _add_pipeline_artifact_media(run_key, run_path, rows, media_seen)

    _validate_media(rows.media_assets, issues)

    track_status_counts = Counter(row.track_status for row in rows.vehicle_tracks)
    evidence_roles = Counter(row.evidence_role for row in rows.track_evidence)
    track_role_counts = Counter((row.ref.key, row.evidence_role) for row in rows.track_evidence)
    duplicate_track_roles = sum(1 for count in track_role_counts.values() if count > 1)
    obs_track_frame_counts = Counter((row.ref.key, row.frame_number) for row in rows.track_observations)
    duplicate_obs_track_frames = sum(count - 1 for count in obs_track_frame_counts.values() if count > 1)

    media_checks = {
        "references": len(rows.media_assets),
        "existing": sum(1 for row in rows.media_assets if row.exists),
        "missing": sum(1 for row in rows.media_assets if not row.exists and not row.invalid_path),
        "invalid_path": sum(1 for row in rows.media_assets if row.invalid_path),
        "outside_run_directory": sum(1 for row in rows.media_assets if row.outside_run_directory),
    }
    colour_final_counts = Counter(row.vehicle_colour for row in rows.vehicle_tracks)
    body_final_counts = Counter(row.body_type for row in rows.vehicle_tracks)
    counts = {
        "source": {
            "tracks": len(source["tracks"]),
            "observations": len(source["observations"]),
            "evidence": len(source["evidence"]),
            "enrichment": len(source["enrichment"]),
        },
        "proposed_table_counts": {
            "processing_runs": len(rows.processing_runs),
            "run_cameras": len(rows.run_cameras),
            "vehicle_tracks": len(rows.vehicle_tracks),
            "track_observations": len(rows.track_observations),
            "track_evidence": len(rows.track_evidence),
            "media_assets": len(rows.media_assets),
            "colour_predictions": len(rows.colour_predictions),
            "vehicle_attribute_predictions": len(rows.vehicle_attribute_predictions),
            "plate_detections": len(rows.plate_detections),
            "plate_readings": len(rows.plate_readings),
            "physical_vehicles": len(rows.physical_vehicles),
            "physical_vehicle_tracks": len(rows.physical_vehicle_tracks),
            "identity_decisions": len(rows.identity_decisions),
        },
        "tracks": {
            "mapped": len(rows.vehicle_tracks),
            "completed": track_status_counts["COMPLETED"],
            "discarded": track_status_counts["DISCARDED"],
            "other_statuses": {str(k): v for k, v in track_status_counts.items() if k not in {"COMPLETED", "DISCARDED"}},
            "duplicates": duplicate_tracks,
            "invalid": sum(1 for issue in issues if issue.code.startswith("invalid_track")),
            "searchable_by_default": sum(1 for row in rows.vehicle_tracks if row.searchable_by_default),
        },
        "observations": {
            "mapped": len(rows.track_observations),
            "orphans": sum(1 for issue in issues if issue.code == "orphan_observation"),
            "duplicates": duplicate_obs_track_frames,
            "invalid": sum(1 for issue in issues if issue.code.startswith("invalid_observation")),
            "unique_track_frame_safe": duplicate_obs_track_frames == 0,
        },
        "evidence": {
            "mapped": len(rows.track_evidence),
            "orphans": sum(1 for issue in issues if issue.code == "orphan_evidence"),
            "roles": dict(sorted(evidence_roles.items())),
            "duplicate_track_role_pairs": duplicate_track_roles,
        },
        "colour": {
            "prediction_rows": len(rows.colour_predictions),
            "tracks_with_colour": sum(1 for row in rows.vehicle_tracks if row.vehicle_colour and row.vehicle_colour != "UNKNOWN"),
            "tracks_without_colour": sum(1 for row in rows.vehicle_tracks if row.vehicle_colour is None),
            "tracks_resolved_unknown": colour_final_counts["UNKNOWN"],
            "tracks_not_processed": sum(1 for row in rows.vehicle_tracks if row.vehicle_colour_status in {None, "skipped"}),
            "final_values": dict(sorted((str(k), v) for k, v in colour_final_counts.items())),
        },
        "attributes": {
            "rows": len(rows.vehicle_attribute_predictions),
            "attribute_types": dict(Counter(row.attribute_type for row in rows.vehicle_attribute_predictions)),
            "body_type_final_values": dict(sorted((str(k), v) for k, v in body_final_counts.items())),
        },
        "plates": {
            "detections": len(rows.plate_detections),
            "readings": len(rows.plate_readings),
            "tracks_plate_detected": sum(1 for row in rows.vehicle_tracks if row.plate_detected),
        },
        "physical_identity": {
            "physical_vehicles": len(rows.physical_vehicles),
            "physical_vehicle_tracks": len(rows.physical_vehicle_tracks),
            "identity_decisions": len(rows.identity_decisions),
        },
        "issues": count_by_severity(issues),
    }
    field_mapping = _build_field_mapping(source)
    verdict = verdict_from_issues(issues)
    return DryRunReport(
        run_dir=run_path,
        run_key=run_key,
        rows=rows,
        issues=issues,
        counts=counts,
        field_mapping=field_mapping,
        media_checks=media_checks,
        normalizations=normalizations,
        verdict=verdict,
    )


def report_to_dict(report: DryRunReport, include_rows: bool = False) -> dict[str, Any]:
    payload = {
        "run_dir": str(report.run_dir),
        "run_key": report.run_key,
        "counts": report.counts,
        "media_checks": report.media_checks,
        "validation_issues": [asdict(issue) for issue in report.issues],
        "field_mapping": report.field_mapping,
        "normalizations": report.normalizations,
        "verdict": report.verdict,
        "database_calls": {"postgres_reads": "NO", "postgres_writes": "NO", "supabase_rest_writes": "NO"},
    }
    if include_rows:
        payload["rows"] = _jsonable(report.rows)
    return payload


def format_console_report(report: DryRunReport) -> str:
    c = report.counts
    lines = [
        "DRY RUN",
        "NO DATABASE WRITES",
        "",
        "RUN",
        f"run key: {report.run_key}",
        f"processing_runs rows: {c['proposed_table_counts']['processing_runs']}",
        "",
        "CAMERAS",
        f"run_cameras rows: {c['proposed_table_counts']['run_cameras']}",
        "",
        "TRACKS",
        f"source: {c['source']['tracks']}",
        f"mapped: {c['tracks']['mapped']}",
        f"completed: {c['tracks']['completed']}",
        f"discarded: {c['tracks']['discarded']}",
        f"duplicates: {c['tracks']['duplicates']}",
        f"invalid: {c['tracks']['invalid']}",
        f"searchable_by_default: {c['tracks']['searchable_by_default']}",
        "",
        "OBSERVATIONS",
        f"source: {c['source']['observations']}",
        f"mapped: {c['observations']['mapped']}",
        f"orphans: {c['observations']['orphans']}",
        f"duplicates: {c['observations']['duplicates']}",
        f"invalid: {c['observations']['invalid']}",
        f"unique_track_frame_safe: {c['observations']['unique_track_frame_safe']}",
        "",
        "EVIDENCE",
        f"source: {c['source']['evidence']}",
        f"mapped: {c['evidence']['mapped']}",
        f"orphans: {c['evidence']['orphans']}",
        f"roles: {c['evidence']['roles']}",
        f"duplicate track+role pairs: {c['evidence']['duplicate_track_role_pairs']}",
        "",
        "MEDIA",
        f"references: {report.media_checks['references']}",
        f"existing: {report.media_checks['existing']}",
        f"missing: {report.media_checks['missing']}",
        f"invalid path: {report.media_checks['invalid_path']}",
        f"outside run directory: {report.media_checks['outside_run_directory']}",
        "",
        "COLOUR",
        f"prediction rows: {c['colour']['prediction_rows']}",
        f"tracks with colour: {c['colour']['tracks_with_colour']}",
        f"tracks without colour: {c['colour']['tracks_without_colour']}",
        f"tracks resolved UNKNOWN: {c['colour']['tracks_resolved_unknown']}",
        f"tracks not processed: {c['colour']['tracks_not_processed']}",
        f"final values: {c['colour']['final_values']}",
        "",
        "ATTRIBUTES",
        f"rows: {c['attributes']['rows']}",
        f"attribute types: {c['attributes']['attribute_types']}",
        "",
        "PLATES",
        f"detections: {c['plates']['detections']}",
        f"readings: {c['plates']['readings']}",
        "",
        "UNMAPPED",
        f"typed: {len(report.field_mapping['typed'])}",
        f"jsonb: {len(report.field_mapping['jsonb'])}",
        f"derived: {len(report.field_mapping['derived'])}",
        f"ignored intentionally: {len(report.field_mapping['ignored_intentionally'])}",
        f"unresolved: {len(report.field_mapping['unresolved'])}",
        "",
        "VALIDATION",
        f"errors: {c['issues']['ERROR']}",
        f"warnings: {c['issues']['WARNING']}",
        f"info: {c['issues']['INFO']}",
        "",
        "VERDICT",
        "DRY RUN PASSED" if c["issues"]["ERROR"] == 0 else report.verdict,
    ]
    return "\n".join(lines)


def _load_run_files_for_report(run_path: Path, issues: list[ValidationIssue]) -> dict[str, Any]:
    source = {
        "metadata": _safe_read_json(run_path / "run_metadata.json", default={}, issues=issues, required=False),
        "summary": _safe_read_json(run_path / "summary.json", default={}, issues=issues, required=False),
        "config": _safe_read_yaml(run_path / "run_config.yaml", issues=issues),
        "tracks": _safe_read_json(run_path / "tracks.json", default=[], issues=issues, required=True),
        "observations": _safe_read_csv(run_path / "observations.csv", issues=issues),
        "evidence": _safe_read_json(run_path / "evidence_index.json", default=[], issues=issues, required=False),
        "enrichment": _safe_read_json(run_path / "vehicle_enrichment.json", default=[], issues=issues, required=False),
        "physical_vehicles": _safe_read_json(run_path / "physical_vehicles.json", default={}, issues=issues, required=False),
        "identity_decisions": _safe_read_json(run_path / "identity_decisions.json", default=[], issues=issues, required=False),
    }
    for key in ("tracks", "evidence", "enrichment"):
        if not isinstance(source[key], list):
            issues.append(ValidationIssue("ERROR", f"invalid_{key}_json_root", f"{key} source must contain a list.", {"run_dir": str(run_path)}))
            source[key] = []
    if not isinstance(source["physical_vehicles"], dict):
        issues.append(ValidationIssue("ERROR", "invalid_physical_vehicles_json_root", "physical_vehicles source must contain a mapping.", {"run_dir": str(run_path)}))
        source["physical_vehicles"] = {}
    if not isinstance(source["identity_decisions"], list):
        issues.append(ValidationIssue("ERROR", "invalid_identity_decisions_json_root", "identity_decisions source must contain a list.", {"run_dir": str(run_path)}))
        source["identity_decisions"] = []
    return source


def _map_processing_run(run_key: str, source: dict[str, Any], run_path: Path) -> ProcessingRunRow:
    metadata = source["metadata"]
    summary = source["summary"]
    discarded = sum((summary.get("tracks_discarded_by_camera") or {}).values()) if isinstance(summary.get("tracks_discarded_by_camera"), dict) else None
    metrics = {k: v for k, v in summary.items() if k not in RUN_TYPED_FIELDS}
    metadata_jsonb = {k: v for k, v in metadata.items() if k not in RUN_TYPED_FIELDS and k != "run_id"}
    return ProcessingRunRow(
        run_key=run_key,
        status=metadata.get("status") or summary.get("status"),
        project_name=metadata.get("project_name") or summary.get("project_name"),
        started_at=metadata.get("started_at"),
        completed_at=metadata.get("completed_at"),
        output_directory=str(run_path),
        config_path=metadata.get("config_path"),
        config_snapshot=source["config"],
        summary=summary,
        detection_backend=summary.get("detection_backend"),
        tracking_backend=summary.get("tracking_backend"),
        enrichment_enabled=summary.get("vehicle_enrichment_enabled"),
        processed_frames=_int_or_none(metadata.get("processed_frames") or summary.get("processed_frames")),
        raw_yolo_detections=_int_or_none(summary.get("raw_yolo_detections")),
        roi_filtered_detections=_int_or_none(summary.get("roi_filtered_detections")),
        completed_tracks=_int_or_none(metadata.get("completed_tracks") or summary.get("completed_tracks")),
        discarded_tracks=_int_or_none(discarded),
        error_count=_int_or_none(metadata.get("error_count")),
        metrics=metrics,
        metadata=metadata_jsonb,
    )


def _map_run_cameras(run_key: str, source: dict[str, Any]) -> list[RunCameraRow]:
    config = source["config"]
    summary = source["summary"]
    cameras = (((config.get("input") or {}).get("cameras")) or []) if isinstance(config, dict) else []
    fps = _float_or_none((config.get("ingestion") or {}).get("target_read_fps")) if isinstance(config, dict) else None
    frames = summary.get("frames_by_camera") or summary.get("frames_consumed_by_camera") or {}
    scheduled_frames = summary.get("frames_scheduled_by_camera") or frames
    detections = summary.get("detections_by_camera") or {}
    rows = []
    seen = set()
    for camera in cameras:
        camera_key = camera.get("camera_id")
        if not camera_key:
            continue
        enabled = camera.get("enabled")
        has_runtime_presence = (
            (isinstance(scheduled_frames, dict) and scheduled_frames.get(camera_key) not in {None, 0, "0"})
            or (isinstance(frames, dict) and frames.get(camera_key) not in {None, 0, "0"})
            or (isinstance(detections, dict) and detections.get(camera_key) not in {None, 0, "0"})
        )
        if enabled is False and not has_runtime_presence:
            continue
        seen.add(camera_key)
        rows.append(
            RunCameraRow(
                run_key=run_key,
                camera_key=camera_key,
                source=camera.get("source"),
                source_type=camera.get("source_type"),
                enabled=enabled,
                fps=fps,
                width=None,
                height=None,
                total_frames=_int_or_none(scheduled_frames.get(camera_key) if isinstance(scheduled_frames, dict) else None),
                frames_processed=_int_or_none(frames.get(camera_key)),
                detections_count=_int_or_none(detections.get(camera_key)),
                metadata={k: v for k, v in camera.items() if k not in {"camera_id", "source", "source_type", "enabled"}},
            )
        )
    for camera_key in sorted(set(frames) - seen):
        rows.append(
            RunCameraRow(
                run_key=run_key,
                camera_key=camera_key,
                source=None,
                source_type=None,
                enabled=None,
                fps=fps,
                width=None,
                height=None,
                total_frames=_int_or_none(scheduled_frames.get(camera_key) if isinstance(scheduled_frames, dict) else None),
                frames_processed=_int_or_none(frames.get(camera_key)),
                detections_count=_int_or_none(detections.get(camera_key)),
                metadata={},
            )
        )
    return rows


def _map_vehicle_track(ref: LogicalTrackRef, track: dict[str, Any], enrichment: dict[str, Any] | None, normalizations: list[dict[str, Any]]) -> VehicleTrackRow:
    colour_obj = (enrichment or {}).get("vehicle_colour") or {}
    body_obj = (enrichment or {}).get("vehicle_body_type") or {}
    final_colour = _normalize_label(colour_obj.get("label") if isinstance(colour_obj, dict) else None, "vehicle_colour", ref, normalizations, attempted=bool(colour_obj))
    body_type = _normalize_label(body_obj.get("label") if isinstance(body_obj, dict) else None, "body_type", ref, normalizations, attempted=bool(body_obj))
    plate_detected = (enrichment or {}).get("plate_detected")
    plate_readable = (enrichment or {}).get("plate_readable")
    plate_text = _empty_to_none((enrichment or {}).get("plate_text"))
    enrichment_summary = {k: v for k, v in (enrichment or {}).items() if k not in ENRICHMENT_FINAL_FIELDS}
    raw_track = {k: v for k, v in track.items() if k not in TRACK_TYPED_FIELDS and k != "vehicle_enrichment"}
    original_local_track_id = _str_or_none(track.get("local_track_id"))
    if original_local_track_id and original_local_track_id != ref.local_track_id:
        raw_track["original_local_track_id"] = original_local_track_id
        raw_track["import_local_track_id"] = ref.local_track_id
    return VehicleTrackRow(
        ref=ref,
        tracker_namespace=track.get("tracker_namespace"),
        native_tracker_id=_str_or_none(track.get("native_tracker_id")),
        track_status=track.get("status"),
        searchable_by_default=track.get("status") == "COMPLETED",
        completion_reason=track.get("completion_reason"),
        first_frame=_int_or_none(track.get("first_frame")),
        last_frame=_int_or_none(track.get("last_frame")),
        first_seen_seconds=_float_or_none(track.get("first_timestamp_seconds")),
        last_seen_seconds=_float_or_none(track.get("last_timestamp_seconds")),
        observation_count=_int_or_none(track.get("observation_count")),
        lost_frames=_int_or_none(track.get("lost_frames")),
        vehicle_class=_normalize_label(track.get("final_class"), "vehicle_class", ref, normalizations, attempted=True),
        final_class_reason=_empty_to_none(track.get("final_class_reason")),
        vehicle_class_confidence=_float_or_none((enrichment or {}).get("vehicle_class_confidence")),
        vehicle_colour=final_colour,
        vehicle_colour_status=colour_obj.get("status") if isinstance(colour_obj, dict) else None,
        body_type=body_type,
        body_type_status=body_obj.get("status") if isinstance(body_obj, dict) else None,
        plate_text=plate_text if plate_detected else None,
        plate_detected=bool(plate_detected) if plate_detected is not None else None,
        registration_category=_empty_to_none((enrichment or {}).get("registration_category")),
        plate_colour=_empty_to_none((enrichment or {}).get("plate_colour")),
        class_counts=track.get("class_counts") or {},
        class_confidence_sums=track.get("class_confidence_sums") or {},
        evidence_record_count=_int_or_none(track.get("evidence_record_count")),
        raw_track=raw_track,
        enrichment_summary={
            **enrichment_summary,
            "plate_readable": plate_readable,
            "plate_raw_text": _empty_to_none((enrichment or {}).get("plate_raw_text")),
            "plate_normalized_text": _empty_to_none((enrichment or {}).get("plate_normalized_text")),
            "plate_validation_status": _empty_to_none((enrichment or {}).get("plate_validation_status")),
            "plate_validation_reason": _empty_to_none((enrichment or {}).get("plate_validation_reason")),
            "plate_format_type": _empty_to_none((enrichment or {}).get("plate_format_type")),
            "plate_correction_applied": (enrichment or {}).get("plate_correction_applied"),
        },
    )


def _map_observations(
    run_key: str,
    observations: list[dict[str, Any]],
    known_refs: dict[str, LogicalTrackRef],
    rows: DryRunRows,
    issues: list[ValidationIssue],
    track_candidates: dict[tuple[str, str], list[tuple[LogicalTrackRef, int | None, int | None, str | None]]],
) -> None:
    for index, obs in enumerate(observations):
        ref = _resolve_track_ref(run_key, obs.get("camera_id"), obs.get("local_track_id"), obs.get("frame_number"), track_candidates)
        if ref.key not in known_refs:
            issues.append(ValidationIssue("ERROR", "orphan_observation", "Observation references missing track.", {"row": index, "track_key": ref.key}))
        frame_number = _int_or_none(obs.get("frame_number"))
        timestamp = _float_or_none(obs.get("timestamp_seconds"))
        if frame_number is None or timestamp is None:
            issues.append(ValidationIssue("ERROR", "invalid_observation_time", "Observation has invalid required frame/timestamp.", {"row": index, "track_key": ref.key}))
        rows.track_observations.append(
            TrackObservationRow(
                ref=ref,
                tracker_namespace=obs.get("tracker_namespace"),
                native_tracker_id=_str_or_none(obs.get("native_tracker_id")),
                frame_number=frame_number,
                timestamp_seconds=timestamp,
                bbox_x1=_float_or_none(obs.get("x1")),
                bbox_y1=_float_or_none(obs.get("y1")),
                bbox_x2=_float_or_none(obs.get("x2")),
                bbox_y2=_float_or_none(obs.get("y2")),
                detection_confidence=_float_or_none(obs.get("confidence")),
                raw_class_id=_int_or_none(obs.get("raw_class_id")),
                raw_class_name=obs.get("raw_class_name"),
                metadata={k: v for k, v in obs.items() if k not in OBSERVATION_TYPED_FIELDS},
            )
        )


def _map_evidence(
    run_key: str,
    evidence_rows: list[dict[str, Any]],
    known_refs: dict[str, LogicalTrackRef],
    rows: DryRunRows,
    issues: list[ValidationIssue],
    run_path: Path,
    media_seen: set[tuple[str, str | None, str | None]],
    track_candidates: dict[tuple[str, str], list[tuple[LogicalTrackRef, int | None, int | None, str | None]]],
) -> None:
    for index, evidence in enumerate(evidence_rows):
        ref = _resolve_track_ref(run_key, evidence.get("camera_id"), evidence.get("local_track_id"), evidence.get("frame_number"), track_candidates)
        if ref.key not in known_refs:
            issues.append(ValidationIssue("ERROR", "orphan_evidence", "Evidence references missing track.", {"row": index, "track_key": ref.key}))
        crop_path = evidence.get("crop_path") or evidence.get("vehicle_crop_path")
        annotated_path = evidence.get("annotated_frame_path")
        source_path = evidence.get("source_image_path") or evidence.get("source_frame_path")
        crop_media = _add_media(run_key, run_path, rows, media_seen, ref, "crop", crop_path, evidence.get("frame_number"), evidence.get("timestamp_seconds"), evidence.get("original_crop_width") or evidence.get("crop_width"), evidence.get("original_crop_height") or evidence.get("crop_height"), {"source": "evidence"})
        source_media = _add_media(run_key, run_path, rows, media_seen, ref, "source_full_frame", source_path, evidence.get("frame_number"), evidence.get("timestamp_seconds"), evidence.get("source_frame_width"), evidence.get("source_frame_height"), {"source": "evidence"}) if source_path else None
        annotated_media = _add_media(run_key, run_path, rows, media_seen, ref, "annotated_frame", annotated_path, evidence.get("frame_number"), evidence.get("timestamp_seconds"), evidence.get("source_frame_width"), evidence.get("source_frame_height"), {"source": "evidence"}) if annotated_path else None
        rows.track_evidence.append(
            TrackEvidenceRow(
                ref=ref,
                evidence_role=evidence.get("evidence_role") or evidence.get("role"),
                frame_number=_int_or_none(evidence.get("frame_number")),
                timestamp_seconds=_float_or_none(evidence.get("timestamp_seconds")),
                bbox_xyxy=evidence.get("bbox_xyxy"),
                original_bbox_xyxy=evidence.get("original_bbox_xyxy") or evidence.get("original_bbox"),
                expanded_crop_bbox_xyxy=evidence.get("expanded_crop_bbox_xyxy") or evidence.get("expanded_crop_bbox"),
                detection_confidence=_float_or_none(evidence.get("detection_confidence") or evidence.get("confidence")),
                quality_score=_float_or_none(evidence.get("quality_score") or evidence.get("best_overall_score")),
                sharpness_score=_float_or_none(evidence.get("sharpness_score")),
                centeredness_score=_float_or_none(evidence.get("centeredness_score")),
                edge_visibility_score=_float_or_none(evidence.get("edge_visibility_score")),
                brightness_score=_float_or_none(evidence.get("brightness_score")),
                crop_width=_int_or_none(evidence.get("crop_width") or evidence.get("original_crop_width")),
                crop_height=_int_or_none(evidence.get("crop_height") or evidence.get("original_crop_height")),
                resolution_tier=evidence.get("resolution_tier"),
                selected_for_colour=evidence.get("selected_for_colour"),
                selected_for_body_type=evidence.get("selected_for_body_type"),
                evidence_source=evidence.get("evidence_source"),
                candidate_rank=_int_or_none(evidence.get("candidate_rank")),
                crop_relative_path=crop_media.relative_path if crop_media else None,
                source_frame_relative_path=source_media.relative_path if source_media else None,
                annotated_frame_relative_path=annotated_media.relative_path if annotated_media else None,
                metadata={k: v for k, v in evidence.items() if k not in EVIDENCE_TYPED_FIELDS},
            )
        )


def _map_enrichment(
    run_key: str,
    enrichment_rows: list[dict[str, Any]],
    known_refs: dict[str, LogicalTrackRef],
    rows: DryRunRows,
    issues: list[ValidationIssue],
    normalizations: list[dict[str, Any]],
    run_path: Path,
    media_seen: set[tuple[str, str | None, str | None]],
    track_candidates: dict[tuple[str, str], list[tuple[LogicalTrackRef, int | None, int | None, str | None]]],
) -> None:
    for index, enrichment in enumerate(enrichment_rows):
        ref = _resolve_track_ref(run_key, enrichment.get("camera_id"), enrichment.get("local_track_id"), enrichment.get("source_frame_number"), track_candidates)
        if ref.key not in known_refs:
            issues.append(ValidationIssue("ERROR", "orphan_enrichment", "Enrichment references missing track.", {"row": index, "track_key": ref.key}))
        colour_obj = enrichment.get("vehicle_colour") or {}
        for pred in colour_obj.get("predictions") or []:
            rows.colour_predictions.append(_colour_prediction_from_model(ref, pred, colour_obj, run_path, run_key, rows, media_seen))
        for pred in enrichment.get("crop_level_colours") or []:
            rows.colour_predictions.append(_colour_prediction_from_crop_level(ref, pred, colour_obj, run_path, run_key, rows, media_seen))
        body_obj = enrichment.get("vehicle_body_type") or {}
        body_predictions = body_obj.get("predictions") or []
        if body_predictions:
            for pred in body_predictions:
                rows.vehicle_attribute_predictions.append(_attribute_prediction(ref, "body_type", pred, body_obj, run_path, run_key, rows, media_seen))
        elif body_obj:
            rows.vehicle_attribute_predictions.append(
                VehicleAttributePredictionRow(
                    ref=ref,
                    attribute_type="body_type",
                    attribute_value=_normalize_label(body_obj.get("label"), "body_type", ref, normalizations, attempted=True),
                    status=body_obj.get("status"),
                    confidence=None,
                    source_backend=body_obj.get("source"),
                    source_model=body_obj.get("model"),
                    raw_response=None,
                    evidence_relative_path=None,
                    evidence_frame_number=None,
                    evidence_timestamp_seconds=None,
                    metadata={k: v for k, v in body_obj.items() if k not in {"label", "status", "source", "model"}},
                )
            )
        for path in enrichment.get("selected_crop_paths") or []:
            _add_media(run_key, run_path, rows, media_seen, ref, "selected_enrichment_crop", path, None, None, None, None, {"source": "vehicle_enrichment.selected_crop_paths"})
        for path in enrichment.get("selected_colour_crop_paths") or []:
            _add_media(run_key, run_path, rows, media_seen, ref, "selected_colour_crop", path, None, None, None, None, {"source": "vehicle_enrichment.selected_colour_crop_paths"})
        for path in enrichment.get("selected_body_type_crop_paths") or []:
            _add_media(run_key, run_path, rows, media_seen, ref, "selected_body_type_crop", path, None, None, None, None, {"source": "vehicle_enrichment.selected_body_type_crop_paths"})
        if enrichment.get("plate_crop_path"):
            plate_media = _add_media(run_key, run_path, rows, media_seen, ref, "plate_crop", enrichment.get("plate_crop_path"), None, None, None, None, {"source": "vehicle_enrichment.plate_crop_path"})
        else:
            plate_media = None
        if enrichment.get("plate_detected"):
            rows.plate_detections.append(
                PlateDetectionRow(
                    ref=ref,
                    plate_bbox=enrichment.get("plate_bbox"),
                    confidence=_float_or_none(enrichment.get("plate_detection_confidence")),
                    crop_relative_path=plate_media.relative_path if plate_media else None,
                    frame_number=None,
                    timestamp_seconds=None,
                    source_model=enrichment.get("plate_ocr_backend") or enrichment.get("attribute_backend"),
                    quality_status=enrichment.get("plate_quality_status"),
                    metadata={},
                )
            )
        if enrichment.get("plate_text"):
            rows.plate_readings.append(
                PlateReadingRow(
                    ref=ref,
                    plate_text=enrichment.get("plate_text"),
                    confidence=_float_or_none(enrichment.get("plate_text_confidence")),
                    plate_colour=enrichment.get("plate_colour"),
                    registration_category=enrichment.get("registration_category"),
                    ocr_backend=enrichment.get("plate_ocr_backend"),
                    raw_response=enrichment.get("plate_ocr_raw_response"),
                    reason=enrichment.get("plate_ocr_reason"),
                    is_selected=True,
                    metadata={
                        "plate_readable": enrichment.get("plate_readable"),
                        "plate_raw_text": enrichment.get("plate_raw_text"),
                        "plate_normalized_text": enrichment.get("plate_normalized_text"),
                        "plate_validation_status": enrichment.get("plate_validation_status"),
                        "plate_validation_reason": enrichment.get("plate_validation_reason"),
                        "plate_format_type": enrichment.get("plate_format_type"),
                        "plate_correction_applied": enrichment.get("plate_correction_applied"),
                    },
                )
            )


def _map_physical_identity(
    run_key: str,
    source: dict[str, Any],
    known_refs: dict[str, LogicalTrackRef],
    rows: DryRunRows,
    issues: list[ValidationIssue],
) -> None:
    payload = dict(source.get("physical_vehicles") or {})
    vehicles = list(payload.get("physical_vehicles", []) or [])
    for vehicle in vehicles:
        if not isinstance(vehicle, dict):
            continue
        vehicle_key = _str_or_none(vehicle.get("vehicle_key") or vehicle.get("vehicle_id"))
        if not vehicle_key:
            issues.append(ValidationIssue("WARNING", "invalid_physical_vehicle", "Physical vehicle row has no vehicle key.", {"vehicle": vehicle}))
            continue
        rows.physical_vehicles.append(
            PhysicalVehicleRow(
                run_key=run_key,
                vehicle_key=vehicle_key,
                vehicle_class=_str_or_none(vehicle.get("vehicle_class") or vehicle.get("final_class")),
                vehicle_colour=_str_or_none(vehicle.get("vehicle_colour") or vehicle.get("colour")),
                first_timestamp_seconds=_float_or_none(vehicle.get("first_seen_seconds") or vehicle.get("first_timestamp_seconds")),
                last_timestamp_seconds=_float_or_none(vehicle.get("last_seen_seconds") or vehicle.get("last_timestamp_seconds")),
                identity_confidence=_float_or_none(vehicle.get("identity_confidence")),
                identity_method=_str_or_none(vehicle.get("identity_method")),
                identity_status=_str_or_none(vehicle.get("identity_status")),
                consensus_plate_text=_str_or_none(vehicle.get("consensus_plate_text") or dict(vehicle.get("plate", {}) or {}).get("consensus_text")),
                plate_confidence=_float_or_none(vehicle.get("plate_confidence")),
                metadata={
                    k: v
                    for k, v in vehicle.items()
                    if k
                    not in {
                        "vehicle_key",
                        "vehicle_id",
                        "vehicle_class",
                        "final_class",
                        "vehicle_colour",
                        "colour",
                        "first_seen_seconds",
                        "first_timestamp_seconds",
                        "last_seen_seconds",
                        "last_timestamp_seconds",
                        "identity_confidence",
                        "identity_method",
                        "identity_status",
                        "consensus_plate_text",
                        "plate_confidence",
                    }
                },
            )
        )
        for local_track_id in list(vehicle.get("member_track_ids") or vehicle.get("member_tracks") or []):
            ref = _known_ref_for_local_track(run_key, _str_or_none(local_track_id), known_refs)
            if ref is None:
                issues.append(ValidationIssue("WARNING", "orphan_physical_vehicle_track", "Physical vehicle member track does not exist in raw vehicle_tracks.", {"vehicle_key": vehicle_key, "local_track_id": local_track_id}))
                continue
            rows.physical_vehicle_tracks.append(
                PhysicalVehicleTrackRow(
                    run_key=run_key,
                    vehicle_key=vehicle_key,
                    ref=ref,
                    association_score=_float_or_none(vehicle.get("identity_confidence")),
                    association_method=_str_or_none(vehicle.get("identity_method")),
                    association_reason=_str_or_none(vehicle.get("identity_status")),
                    metadata={},
                )
            )
    for item in list(source.get("identity_decisions") or []):
        if not isinstance(item, dict):
            continue
        rows.identity_decisions.append(
            IdentityDecisionRow(
                run_key=run_key,
                source_ref=_known_ref_for_local_track(run_key, _str_or_none(item.get("source_track_id") or item.get("track_a")), known_refs),
                target_ref=_known_ref_for_local_track(run_key, _str_or_none(item.get("target_track_id") or item.get("track_b")), known_refs),
                decision=_str_or_none(item.get("decision")) or "UNKNOWN",
                final_score=_float_or_none(item.get("final_score") or item.get("score")),
                plate_score=_float_or_none(item.get("plate_score")),
                spatial_score=_float_or_none(item.get("spatial_score")),
                temporal_score=_float_or_none(item.get("temporal_score")),
                motion_score=_float_or_none(item.get("motion_score")),
                appearance_score=_float_or_none(item.get("appearance_score")),
                colour_score=_float_or_none(item.get("colour_score")),
                reason=_str_or_none(item.get("reason") or item.get("decision_reason_codes") or item.get("plate_reason_code")),
                metadata=item,
            )
        )


def _colour_prediction_from_model(
    ref: LogicalTrackRef,
    pred: dict[str, Any],
    colour_obj: dict[str, Any],
    run_path: Path,
    run_key: str,
    rows: DryRunRows,
    media_seen: set[tuple[str, str | None, str | None]],
) -> ColourPredictionRow:
    media = _add_media(run_key, run_path, rows, media_seen, ref, "selected_colour_crop", pred.get("source_crop_path"), pred.get("source_frame_number"), None, pred.get("original_crop_width"), pred.get("original_crop_height"), {"source": "colour_prediction"}) if pred.get("source_crop_path") else None
    return ColourPredictionRow(
        ref=ref,
        evidence_relative_path=media.relative_path if media else None,
        predicted_colour=pred.get("label"),
        normalized_colour=pred.get("label"),
        confidence=_float_or_none(pred.get("confidence")),
        status=pred.get("status"),
        source_model=pred.get("source_model") or colour_obj.get("model"),
        source_backend=pred.get("source_backend") or colour_obj.get("source"),
        prompt=colour_obj.get("prompt_text") or colour_obj.get("task_prompt"),
        raw_response=pred.get("raw_response"),
        inference_duration_ms=_float_or_none(pred.get("inference_duration_ms")),
        evidence_frame_number=_int_or_none(pred.get("source_frame_number")),
        evidence_timestamp_seconds=None,
        metadata={k: v for k, v in pred.items() if k not in {"label", "confidence", "status", "source_model", "source_backend", "source_crop_path", "source_frame_number", "raw_response", "inference_duration_ms"}},
    )


def _colour_prediction_from_crop_level(
    ref: LogicalTrackRef,
    pred: dict[str, Any],
    colour_obj: dict[str, Any],
    run_path: Path,
    run_key: str,
    rows: DryRunRows,
    media_seen: set[tuple[str, str | None, str | None]],
) -> ColourPredictionRow:
    media = _add_media(run_key, run_path, rows, media_seen, ref, "selected_colour_crop", pred.get("crop_path"), pred.get("frame_index"), None, None, None, {"source": "crop_level_colours"}) if pred.get("crop_path") else None
    return ColourPredictionRow(
        ref=ref,
        evidence_relative_path=media.relative_path if media else None,
        predicted_colour=pred.get("raw_colour_phrase") or pred.get("normalized_colour"),
        normalized_colour=pred.get("normalized_colour"),
        confidence=None,
        status=pred.get("status"),
        source_model=colour_obj.get("model"),
        source_backend=colour_obj.get("source"),
        prompt=pred.get("prompt") or colour_obj.get("prompt_text"),
        raw_response=pred.get("raw_response"),
        inference_duration_ms=_float_or_none(pred.get("inference_time_ms")),
        evidence_frame_number=_int_or_none(pred.get("frame_index")),
        evidence_timestamp_seconds=None,
        metadata={k: v for k, v in pred.items() if k not in {"raw_colour_phrase", "normalized_colour", "status", "crop_path", "frame_index", "prompt", "raw_response", "inference_time_ms"}},
    )


def _attribute_prediction(
    ref: LogicalTrackRef,
    attribute_type: str,
    pred: dict[str, Any],
    source_obj: dict[str, Any],
    run_path: Path,
    run_key: str,
    rows: DryRunRows,
    media_seen: set[tuple[str, str | None, str | None]],
) -> VehicleAttributePredictionRow:
    media = _add_media(run_key, run_path, rows, media_seen, ref, f"selected_{attribute_type}_crop", pred.get("source_crop_path"), pred.get("source_frame_number"), None, pred.get("original_crop_width"), pred.get("original_crop_height"), {"source": f"{attribute_type}_prediction"}) if pred.get("source_crop_path") else None
    return VehicleAttributePredictionRow(
        ref=ref,
        attribute_type=attribute_type,
        attribute_value=pred.get("label"),
        status=pred.get("status"),
        confidence=_float_or_none(pred.get("confidence")),
        source_backend=pred.get("source_backend") or source_obj.get("source"),
        source_model=pred.get("source_model") or source_obj.get("model"),
        raw_response=pred.get("raw_response"),
        evidence_relative_path=media.relative_path if media else None,
        evidence_frame_number=_int_or_none(pred.get("source_frame_number")),
        evidence_timestamp_seconds=None,
        metadata={k: v for k, v in pred.items() if k not in {"label", "status", "confidence", "source_backend", "source_model", "raw_response", "source_crop_path", "source_frame_number"}},
    )


def _add_pipeline_artifact_media(run_key: str, run_path: Path, rows: DryRunRows, media_seen: set[tuple[str, str | None, str | None]]) -> None:
    for folder, media_type in [
        ("01_extracted_frames", "raw_frame"),
        ("02_yolo_detected_frames", "detected_frame"),
        ("03_tracked_frames", "tracked_frame"),
    ]:
        path = run_path / folder
        if not path.exists():
            continue
        for image_path in path.rglob("*.jpg"):
            camera_key = image_path.parent.name if image_path.parent != path else None
            _add_media(run_key, run_path, rows, media_seen, None, media_type, str(image_path), None, None, None, None, {"source": folder, "camera_key": camera_key})


def _add_media(
    run_key: str,
    run_path: Path,
    rows: DryRunRows,
    media_seen: set[tuple[str, str | None, str | None]],
    ref: LogicalTrackRef | None,
    media_type: str,
    original_path: Any,
    frame_number: Any,
    timestamp_seconds: Any,
    width: Any,
    height: Any,
    metadata: dict[str, Any],
) -> MediaAssetRow | None:
    normalized = _normalize_media_path(original_path, run_path)
    if not normalized["original_path"]:
        return None
    key = (media_type, normalized["relative_path"], normalized["original_path"])
    if key in media_seen:
        for row in rows.media_assets:
            if (row.media_type, row.relative_path, row.original_path) == key:
                return row
        return None
    media_seen.add(key)
    row = MediaAssetRow(
        run_key=run_key,
        camera_key=ref.camera_key if ref else metadata.get("camera_key"),
        track_local_id=ref.local_track_id if ref else None,
        media_type=media_type,
        relative_path=normalized["relative_path"],
        original_path=normalized["original_path"],
        frame_number=_int_or_none(frame_number),
        timestamp_seconds=_float_or_none(timestamp_seconds),
        width=_int_or_none(width),
        height=_int_or_none(height),
        exists=normalized["exists"],
        invalid_path=normalized["invalid_path"],
        outside_run_directory=normalized["outside_run_directory"],
        metadata={**metadata, **normalized["metadata"]},
    )
    rows.media_assets.append(row)
    return row


def _normalize_media_path(value: Any, run_path: Path) -> dict[str, Any]:
    original = _str_or_none(value)
    if not original:
        return {"original_path": None, "relative_path": None, "exists": False, "invalid_path": True, "outside_run_directory": False, "metadata": {}}
    try:
        raw = Path(original)
        candidate = raw if raw.is_absolute() else (run_path / raw)
        resolved = candidate.resolve()
        run_resolved = run_path.resolve()
        outside = not _is_relative_to(resolved, run_resolved)
        relative = str(resolved.relative_to(run_resolved)).replace("\\", "/") if not outside else None
        return {
            "original_path": original,
            "relative_path": relative,
            "exists": resolved.exists(),
            "invalid_path": False,
            "outside_run_directory": outside,
            "metadata": {"absolute_path_fallback": original if outside else None},
        }
    except (OSError, RuntimeError, ValueError):
        return {"original_path": original, "relative_path": None, "exists": False, "invalid_path": True, "outside_run_directory": False, "metadata": {}}


def _validate_media(media_rows: list[MediaAssetRow], issues: list[ValidationIssue]) -> None:
    for row in media_rows:
        if row.invalid_path:
            issues.append(ValidationIssue("WARNING", "invalid_media_path", "Media path could not be parsed.", {"path": row.original_path}))
        elif row.outside_run_directory:
            issues.append(ValidationIssue("WARNING", "media_outside_run_directory", "Media path is outside the run directory; only debug metadata keeps the original path.", {"path": row.original_path}))
        elif not row.exists:
            issues.append(ValidationIssue("WARNING", "missing_media_file", "Referenced media file is missing.", {"path": row.relative_path or row.original_path, "media_type": row.media_type}))


def _build_field_mapping(source: dict[str, Any]) -> dict[str, Any]:
    track_keys = _all_keys(source["tracks"])
    obs_keys = _all_keys(source["observations"])
    evidence_keys = _all_keys(source["evidence"])
    enrichment_keys = _all_keys(source["enrichment"])
    typed = sorted(
        {f"tracks.{k}" for k in track_keys & TRACK_TYPED_FIELDS}
        | {f"observations.{k}" for k in obs_keys & OBSERVATION_TYPED_FIELDS}
        | {f"evidence.{k}" for k in evidence_keys & EVIDENCE_TYPED_FIELDS}
        | {f"enrichment.{k}" for k in enrichment_keys & ENRICHMENT_FINAL_FIELDS}
        | {f"run.{k}" for k in RUN_TYPED_FIELDS}
    )
    jsonb = sorted(
        {f"tracks.{k}" for k in track_keys - TRACK_TYPED_FIELDS - {"vehicle_enrichment"}}
        | {f"observations.{k}" for k in obs_keys - OBSERVATION_TYPED_FIELDS}
        | {f"evidence.{k}" for k in evidence_keys - EVIDENCE_TYPED_FIELDS}
        | {f"enrichment.{k}" for k in enrichment_keys - ENRICHMENT_FINAL_FIELDS}
    )
    return {
        "typed": typed,
        "jsonb": jsonb,
        "derived": ["vehicle_tracks.searchable_by_default", "vehicle_tracks.duration_seconds", "track_observations.unique_track_frame_safe"],
        "ignored_intentionally": ["raw image bytes", "future global vehicle identity"],
        "unresolved": [],
    }


def _index_by_track(
    items: list[dict[str, Any]],
    run_key: str,
    track_candidates: dict[tuple[str, str], list[tuple[LogicalTrackRef, int | None, int | None, str | None]]] | None = None,
) -> dict[str, dict[str, Any]]:
    result = {}
    for item in items:
        ref = _resolve_track_ref(run_key, item.get("camera_id"), item.get("local_track_id"), item.get("source_frame_number"), track_candidates or {})
        result[ref.key] = item
    return result


def _track_ref(run_key: str, camera_id: Any, local_track_id: Any) -> LogicalTrackRef:
    camera_key = _str_or_none(camera_id) or "UNKNOWN_CAMERA"
    local = _str_or_none(local_track_id) or "UNKNOWN_TRACK"
    return LogicalTrackRef(run_key=run_key, camera_key=camera_key, local_track_id=local)


def _resolve_track_ref(
    run_key: str,
    camera_id: Any,
    local_track_id: Any,
    frame_number: Any,
    track_candidates: dict[tuple[str, str], list[tuple[LogicalTrackRef, int | None, int | None, str | None]]],
) -> LogicalTrackRef:
    camera_key = _str_or_none(camera_id) or "UNKNOWN_CAMERA"
    local = _str_or_none(local_track_id) or "UNKNOWN_TRACK"
    candidates = list(track_candidates.get((camera_key, local), []) or [])
    if not candidates:
        return _track_ref(run_key, camera_key, local)
    if len(candidates) == 1:
        return candidates[0][0]
    frame = _int_or_none(frame_number)
    if frame is not None:
        in_range = [candidate for candidate in candidates if candidate[1] is not None and candidate[2] is not None and candidate[1] <= frame <= candidate[2]]
        if len(in_range) == 1:
            return in_range[0][0]
    completed = [candidate for candidate in candidates if candidate[3] == "COMPLETED"]
    if len(completed) == 1:
        return completed[0][0]
    return candidates[0][0]


def _known_ref_for_local_track(run_key: str, local_track_id: str | None, known_refs: dict[str, LogicalTrackRef]) -> LogicalTrackRef | None:
    if not local_track_id:
        return None
    for ref in known_refs.values():
        if ref.run_key == run_key and ref.local_track_id == local_track_id:
            return ref
    return None


def _safe_read_json(path: Path, *, default: Any, issues: list[ValidationIssue], required: bool) -> Any:
    if not path.exists():
        if required:
            issues.append(ValidationIssue("ERROR", "missing_required_file", "Required run file is missing.", {"path": str(path)}))
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        issues.append(ValidationIssue("ERROR", "malformed_json", "JSON file could not be parsed.", {"path": str(path), "error": str(exc)}))
        return default


def _safe_read_yaml(path: Path, issues: list[ValidationIssue]) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return payload if isinstance(payload, dict) else {}
    except Exception as exc:
        issues.append(ValidationIssue("ERROR", "malformed_yaml", "YAML file could not be parsed.", {"path": str(path), "error": str(exc)}))
        return {}


def _safe_read_csv(path: Path, issues: list[ValidationIssue]) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))
    except Exception as exc:
        issues.append(ValidationIssue("ERROR", "malformed_csv", "CSV file could not be parsed.", {"path": str(path), "error": str(exc)}))
        return []


def _normalize_label(value: Any, field_name: str, ref: LogicalTrackRef, normalizations: list[dict[str, Any]], attempted: bool) -> str | None:
    text = _empty_to_none(value)
    if text is None:
        return "UNKNOWN" if attempted and field_name in {"vehicle_class"} else None
    normalized = str(text).upper()
    if normalized != text:
        normalizations.append({"field": field_name, "track_key": ref.key, "from": text, "to": normalized, "rule": "uppercase canonical label"})
    return normalized


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return payload if isinstance(payload, dict) else {}


def _read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _all_keys(items: list[dict[str, Any]]) -> set[str]:
    keys: set[str] = set()
    for item in items:
        keys.update(item.keys())
    return keys


def _empty_to_none(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _str_or_none(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _int_or_none(value: Any) -> int | None:
    if value in {None, ""}:
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _float_or_none(value: Any) -> float | None:
    if value in {None, ""}:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return {k: _jsonable(v) for k, v in asdict(value).items()}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Dry-run map a file-based run into canonical DB row models.")
    parser.add_argument("--run-dir", required=True, help="Path to outputs/runs/<run_id>.")
    parser.add_argument("--dry-run", action="store_true", help="Required safety flag. No database writes are implemented.")
    parser.add_argument("--report-json", help="Optional path to write a machine-readable dry-run report.")
    parser.add_argument("--include-rows", action="store_true", help="Include proposed row payloads in the JSON report.")
    args = parser.parse_args(argv)
    if not args.dry_run:
        parser.error("--dry-run is required; this importer has no live DB writer.")
    report = build_dry_run(args.run_dir)
    print(format_console_report(report))
    if args.report_json:
        report_path = Path(args.report_json)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report_to_dict(report, include_rows=args.include_rows), indent=2, sort_keys=True), encoding="utf-8")
        print(f"\nWrote machine-readable dry-run report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
