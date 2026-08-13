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
    PlateDetectionRow,
    PlateReadingRow,
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
    }


def build_dry_run(run_dir: str | Path) -> DryRunReport:
    run_path = Path(run_dir).resolve()
    source = load_run_files(run_path)
    run_key = str(source["metadata"].get("run_id") or source["summary"].get("run_id") or run_path.name)
    rows = DryRunRows()
    issues: list[ValidationIssue] = []
    normalizations: list[dict[str, Any]] = []
    media_seen: set[tuple[str, str | None, str | None]] = set()

    enrichment_by_track = _index_by_track(source["enrichment"], run_key)
    rows.processing_runs.append(_map_processing_run(run_key, source))
    rows.run_cameras.extend(_map_run_cameras(run_key, source))

    known_refs: dict[str, LogicalTrackRef] = {}
    duplicate_tracks = 0
    for track in source["tracks"]:
        ref = _track_ref(run_key, track.get("camera_id"), track.get("local_track_id"))
        if ref.key in known_refs:
            duplicate_tracks += 1
            issues.append(ValidationIssue("ERROR", "duplicate_logical_track_identity", "Duplicate logical track identity.", {"track_key": ref.key}))
        known_refs[ref.key] = ref
        rows.vehicle_tracks.append(_map_vehicle_track(ref, track, enrichment_by_track.get(ref.key), normalizations))

    _map_observations(run_key, source["observations"], known_refs, rows, issues)
    _map_evidence(run_key, source["evidence"], known_refs, rows, issues, run_path, media_seen)
    _map_enrichment(run_key, source["enrichment"], known_refs, rows, issues, normalizations, run_path, media_seen)
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
        "database_calls": {"supabase_reads": "NO", "supabase_writes": "NO"},
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
        report.verdict,
    ]
    return "\n".join(lines)


def _map_processing_run(run_key: str, source: dict[str, Any]) -> ProcessingRunRow:
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
        config_path=metadata.get("config_path"),
        config_snapshot=source["config"],
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
    detections = summary.get("detections_by_camera") or {}
    rows = []
    seen = set()
    for camera in cameras:
        camera_key = camera.get("camera_id")
        if not camera_key:
            continue
        seen.add(camera_key)
        rows.append(
            RunCameraRow(
                run_key=run_key,
                camera_key=camera_key,
                source=camera.get("source"),
                source_type=camera.get("source_type"),
                enabled=camera.get("enabled"),
                fps=fps,
                width=None,
                height=None,
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
    plate_text = _empty_to_none((enrichment or {}).get("plate_text"))
    enrichment_summary = {k: v for k, v in (enrichment or {}).items() if k not in ENRICHMENT_FINAL_FIELDS}
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
        vehicle_class_confidence=_float_or_none((enrichment or {}).get("vehicle_class_confidence")),
        vehicle_colour=final_colour,
        vehicle_colour_status=colour_obj.get("status") if isinstance(colour_obj, dict) else None,
        body_type=body_type,
        body_type_status=body_obj.get("status") if isinstance(body_obj, dict) else None,
        plate_text=plate_text if plate_detected else None,
        plate_detected=bool(plate_detected) if plate_detected is not None else None,
        plate_colour=_empty_to_none((enrichment or {}).get("plate_colour")),
        registration_category=_empty_to_none((enrichment or {}).get("registration_category")),
        class_counts=track.get("class_counts") or {},
        class_confidence_sums=track.get("class_confidence_sums") or {},
        raw_track={k: v for k, v in track.items() if k not in TRACK_TYPED_FIELDS and k != "vehicle_enrichment"},
        enrichment_summary=enrichment_summary,
    )


def _map_observations(run_key: str, observations: list[dict[str, Any]], known_refs: dict[str, LogicalTrackRef], rows: DryRunRows, issues: list[ValidationIssue]) -> None:
    for index, obs in enumerate(observations):
        ref = _track_ref(run_key, obs.get("camera_id"), obs.get("local_track_id"))
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
) -> None:
    for index, evidence in enumerate(evidence_rows):
        ref = _track_ref(run_key, evidence.get("camera_id"), evidence.get("local_track_id"))
        if ref.key not in known_refs:
            issues.append(ValidationIssue("ERROR", "orphan_evidence", "Evidence references missing track.", {"row": index, "track_key": ref.key}))
        crop_path = evidence.get("crop_path") or evidence.get("vehicle_crop_path")
        annotated_path = evidence.get("annotated_frame_path")
        source_path = evidence.get("source_image_path") or evidence.get("source_frame_path")
        crop_media = _add_media(run_key, run_path, rows, media_seen, ref, "crop", crop_path, evidence.get("frame_number"), evidence.get("original_crop_width") or evidence.get("crop_width"), evidence.get("original_crop_height") or evidence.get("crop_height"), {"source": "evidence"})
        source_media = _add_media(run_key, run_path, rows, media_seen, ref, "source_full_frame", source_path, evidence.get("frame_number"), evidence.get("source_frame_width"), evidence.get("source_frame_height"), {"source": "evidence"}) if source_path else None
        annotated_media = _add_media(run_key, run_path, rows, media_seen, ref, "annotated_frame", annotated_path, evidence.get("frame_number"), evidence.get("source_frame_width"), evidence.get("source_frame_height"), {"source": "evidence"}) if annotated_path else None
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
) -> None:
    for index, enrichment in enumerate(enrichment_rows):
        ref = _track_ref(run_key, enrichment.get("camera_id"), enrichment.get("local_track_id"))
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
                    label=_normalize_label(body_obj.get("label"), "body_type", ref, normalizations, attempted=True),
                    status=body_obj.get("status"),
                    confidence=None,
                    source_backend=body_obj.get("source"),
                    source_model=body_obj.get("model"),
                    raw_response=None,
                    evidence_relative_path=None,
                    metadata={k: v for k, v in body_obj.items() if k not in {"label", "status", "source", "model"}},
                )
            )
        for path in enrichment.get("selected_crop_paths") or []:
            _add_media(run_key, run_path, rows, media_seen, ref, "selected_enrichment_crop", path, None, None, None, {"source": "vehicle_enrichment.selected_crop_paths"})
        for path in enrichment.get("selected_colour_crop_paths") or []:
            _add_media(run_key, run_path, rows, media_seen, ref, "selected_colour_crop", path, None, None, None, {"source": "vehicle_enrichment.selected_colour_crop_paths"})
        for path in enrichment.get("selected_body_type_crop_paths") or []:
            _add_media(run_key, run_path, rows, media_seen, ref, "selected_body_type_crop", path, None, None, None, {"source": "vehicle_enrichment.selected_body_type_crop_paths"})
        if enrichment.get("plate_crop_path"):
            plate_media = _add_media(run_key, run_path, rows, media_seen, ref, "plate_crop", enrichment.get("plate_crop_path"), None, None, None, {"source": "vehicle_enrichment.plate_crop_path"})
        else:
            plate_media = None
        if enrichment.get("plate_detected"):
            rows.plate_detections.append(
                PlateDetectionRow(
                    ref=ref,
                    plate_bbox=enrichment.get("plate_bbox"),
                    confidence=_float_or_none(enrichment.get("plate_detection_confidence")),
                    crop_relative_path=plate_media.relative_path if plate_media else None,
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
                    metadata={},
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
    media = _add_media(run_key, run_path, rows, media_seen, ref, "selected_colour_crop", pred.get("source_crop_path"), pred.get("source_frame_number"), pred.get("original_crop_width"), pred.get("original_crop_height"), {"source": "colour_prediction"}) if pred.get("source_crop_path") else None
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
    media = _add_media(run_key, run_path, rows, media_seen, ref, "selected_colour_crop", pred.get("crop_path"), pred.get("frame_index"), None, None, {"source": "crop_level_colours"}) if pred.get("crop_path") else None
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
    media = _add_media(run_key, run_path, rows, media_seen, ref, f"selected_{attribute_type}_crop", pred.get("source_crop_path"), pred.get("source_frame_number"), pred.get("original_crop_width"), pred.get("original_crop_height"), {"source": f"{attribute_type}_prediction"}) if pred.get("source_crop_path") else None
    return VehicleAttributePredictionRow(
        ref=ref,
        attribute_type=attribute_type,
        label=pred.get("label"),
        status=pred.get("status"),
        confidence=_float_or_none(pred.get("confidence")),
        source_backend=pred.get("source_backend") or source_obj.get("source"),
        source_model=pred.get("source_model") or source_obj.get("model"),
        raw_response=pred.get("raw_response"),
        evidence_relative_path=media.relative_path if media else None,
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
            _add_media(run_key, run_path, rows, media_seen, None, media_type, str(image_path), None, None, None, {"source": folder, "camera_key": camera_key})


def _add_media(
    run_key: str,
    run_path: Path,
    rows: DryRunRows,
    media_seen: set[tuple[str, str | None, str | None]],
    ref: LogicalTrackRef | None,
    media_type: str,
    original_path: Any,
    frame_number: Any,
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


def _index_by_track(items: list[dict[str, Any]], run_key: str) -> dict[str, dict[str, Any]]:
    result = {}
    for item in items:
        ref = _track_ref(run_key, item.get("camera_id"), item.get("local_track_id"))
        result[ref.key] = item
    return result


def _track_ref(run_key: str, camera_id: Any, local_track_id: Any) -> LogicalTrackRef:
    camera_key = _str_or_none(camera_id) or "UNKNOWN_CAMERA"
    local = _str_or_none(local_track_id) or "UNKNOWN_TRACK"
    short = local.split(":", 1)[1] if ":" in local else local
    return LogicalTrackRef(run_key=run_key, camera_key=camera_key, local_track_id=short)


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
