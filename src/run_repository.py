from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from .vehicle_enrichment.taxonomy import SUPPORTED_VEHICLE_CLASSES, SUPPORTED_VEHICLE_COLOUR_LABELS
from .vehicle_analytics import vehicle_records_from_physical_vehicles, vehicle_records_from_repository_tracks


class RunRepository:
    def __init__(self, outputs_root: str | Path) -> None:
        self.outputs_root = Path(outputs_root).expanduser().resolve()

    def latest_run_id(self) -> str | None:
        runs = self.list_runs()
        if not runs:
            return None
        return str(runs[0]["run_id"])

    def resolve_run_id(self, run_id: str | None = None) -> str | None:
        if run_id is None or str(run_id).strip() == "" or str(run_id).strip().lower() == "latest":
            return self.latest_run_id()
        candidate = str(run_id).strip()
        return candidate if self._resolve_run_directory(candidate) is not None else None

    def tracks_json_path(self, run_id: str) -> Path | None:
        run_dir = self._resolve_run_directory(run_id)
        if run_dir is None:
            return None
        return run_dir / "tracks.json"

    def list_runs(self) -> list[dict[str, Any]]:
        runs: list[dict[str, Any]] = []
        for run_dir in self._iter_run_directories():
            summary = self._read_json(run_dir / "summary.json", default={})
            metadata = self._read_json(run_dir / "run_metadata.json", default={})
            run_config = self._read_yaml_text(run_dir / "run_config.yaml")
            tracks = self._read_json(run_dir / "tracks.json", default=[])
            raw_track_count = len([item for item in tracks if isinstance(item, dict)])
            completed_track_count = len([item for item in tracks if isinstance(item, dict) and str(item.get("status") or "").upper() == "COMPLETED"])
            physical_vehicle_count = self._physical_vehicle_count(run_dir)
            camera_count = summary.get("configured_camera_count")
            if camera_count is None:
                camera_count = metadata.get("camera_count")
            runs.append(
                {
                    "run_id": run_dir.name,
                    "status": summary.get("status") or metadata.get("status") or "UNKNOWN",
                    "start_time": metadata.get("started_at"),
                    "completed_at": metadata.get("completed_at"),
                    "camera_count": camera_count,
                    "processed_frames": summary.get("processed_frames"),
                    "overall_pipeline_runtime_ms": summary.get("overall_pipeline_runtime_ms"),
                    "duration_seconds": self._duration_seconds_from_summary(summary),
                    "track_count": physical_vehicle_count or raw_track_count,
                    "physical_vehicle_count": physical_vehicle_count,
                    "raw_track_count": raw_track_count,
                    "completed_track_count": completed_track_count,
                    "frames_by_camera": summary.get("frames_by_camera", {}),
                    "run_directory": str(run_dir),
                    "has_run_config": run_config is not None,
                }
            )
        runs.sort(key=lambda item: str(item["run_id"]), reverse=True)
        return runs

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        run_dir = self._resolve_run_directory(run_id)
        if run_dir is None:
            return None
        summary = self._read_json(run_dir / "summary.json", default={})
        metadata = self._read_json(run_dir / "run_metadata.json", default={})
        detection = self._read_json(run_dir / "detection_tracking_metrics.json", default={})
        ingestion = self._read_json(run_dir / "ingestion_metrics.json", default={})
        evidence = self._read_json(run_dir / "evidence_metrics.json", default={})
        enrichment = self._read_json(run_dir / "vehicle_enrichment_metrics.json", default={})
        tracks = self._read_json(run_dir / "tracks.json", default=[])
        raw_track_count = len([item for item in tracks if isinstance(item, dict)])
        completed_track_count = len([item for item in tracks if isinstance(item, dict) and str(item.get("status") or "").upper() == "COMPLETED"])
        physical_vehicle_count = self._physical_vehicle_count(run_dir)
        return {
            "run_id": run_id,
            "summary": summary,
            "metadata": metadata,
            "detection_tracking_metrics": detection,
            "ingestion_metrics": ingestion,
            "evidence_metrics": evidence,
            "vehicle_enrichment_metrics": enrichment,
            "track_count": physical_vehicle_count or raw_track_count,
            "physical_vehicle_count": physical_vehicle_count,
            "raw_track_count": raw_track_count,
            "completed_track_count": completed_track_count,
            "paths": {
                "tracks": str(run_dir / "tracks.json"),
                "vehicle_enrichment": str(run_dir / "vehicle_enrichment.json"),
                "evidence_index": str(run_dir / "evidence_index.json"),
                "track_crop_manifest": str(run_dir / "04_track_crops" / "track_crop_manifest.csv"),
                "detected_frames": str(run_dir / "detected_frames"),
                "tracked_frames": str(run_dir / "tracked_frames"),
            },
        }

    def get_track_reconciliation(self, run_id: str) -> dict[str, Any] | None:
        run_dir = self._resolve_run_directory(run_id)
        if run_dir is None:
            return None
        experiment_dir = run_dir / "track_reconciliation_test"
        result = self._read_json(experiment_dir / "track_reconciliation_test.json", default=None)
        if not isinstance(result, dict):
            return {
                "run_id": run_id,
                "available": False,
                "message": "Reconciliation test has not been run for this run.",
                "metrics": {},
                "config": {},
                "tracks": [],
                "accepted_associations": [],
                "manual_validation": [],
                "visual_evidence": [],
                "paths": {},
            }
        associations = self._read_csv_rows(experiment_dir / "association_table.csv")
        manual_validation = self._read_csv_rows(experiment_dir / "manual_validation.csv")
        visual_evidence = self._build_reconciliation_visual_evidence(run_id=run_id, experiment_dir=experiment_dir)
        return {
            "run_id": run_id,
            "available": True,
            "message": None,
            "metrics": result.get("metrics", {}),
            "config": result.get("config", {}),
            "tracks": result.get("tracks", []),
            "accepted_associations": result.get("accepted_associations", associations),
            "manual_validation": manual_validation,
            "visual_evidence": visual_evidence,
            "paths": {
                "result_json": str(experiment_dir / "track_reconciliation_test.json"),
                "association_table": str(experiment_dir / "association_table.csv"),
                "manual_validation": str(experiment_dir / "manual_validation.csv"),
                "report": str(experiment_dir / "report.md"),
            },
        }

    def get_vehicle_identity_experiment(self, run_id: str) -> dict[str, Any] | None:
        run_dir = self._resolve_run_directory(run_id)
        if run_dir is None:
            return None
        experiment_dir = run_dir / "vehicle_identity_test"
        vehicles_payload = self._read_json(experiment_dir / "vehicles.json", default=None)
        identity_map = self._read_json(experiment_dir / "vehicle_id_map.json", default={})
        evaluation = self._read_json(experiment_dir / "evaluation.json", default={})
        decisions = self._read_csv_rows(experiment_dir / "association_decisions.csv")
        if not isinstance(vehicles_payload, dict):
            return {
                "run_id": run_id,
                "experimental": True,
                "available": False,
                "message": "Persistent vehicle identity experiment has not been run for this run.",
                "metrics": {},
                "analytics_simulation": {},
                "config": {},
                "calibration": {},
                "vehicles": [],
                "vehicle_id_map": {},
                "association_decisions": [],
                "paths": {},
            }
        vehicles = list(vehicles_payload.get("vehicles", []) or [])
        for vehicle in vehicles:
            if not isinstance(vehicle, dict):
                continue
            vehicle["contact_sheet_url"] = self._vehicle_identity_contact_sheet_url(run_id, str(vehicle.get("vehicle_id") or ""))
        return {
            "run_id": run_id,
            "experimental": True,
            "available": True,
            "message": None,
            "metrics": evaluation.get("metrics", {}),
            "analytics_simulation": evaluation.get("analytics_simulation", {}),
            "existing_reconciliation_baseline": evaluation.get("existing_reconciliation_baseline", {}),
            "config": evaluation.get("config", {}),
            "calibration": evaluation.get("calibration", {}),
            "vehicles": vehicles,
            "vehicle_id_map": identity_map if isinstance(identity_map, dict) else {},
            "association_decisions": decisions,
            "paths": {
                "vehicles": str(experiment_dir / "vehicles.json"),
                "vehicle_id_map": str(experiment_dir / "vehicle_id_map.json"),
                "evaluation": str(experiment_dir / "evaluation.json"),
                "calibration_summary": str(experiment_dir / "calibration_summary.json"),
                "association_decisions": str(experiment_dir / "association_decisions.csv"),
                "report": str(experiment_dir / "report.md"),
            },
        }

    def get_vehicle_identity_summary(self, run_id: str) -> dict[str, Any] | None:
        payload = self.get_vehicle_identity_experiment(run_id)
        if payload is None:
            return None
        return {
            "run_id": payload["run_id"],
            "experimental": True,
            "available": payload["available"],
            "message": payload["message"],
            "metrics": payload["metrics"],
            "analytics_simulation": payload["analytics_simulation"],
            "existing_reconciliation_baseline": payload.get("existing_reconciliation_baseline", {}),
            "calibration": payload.get("calibration", {}),
        }

    def get_stationary_recovery_experiment(self, run_id: str) -> dict[str, Any] | None:
        run_dir = self._resolve_run_directory(run_id)
        if run_dir is None:
            return None
        output_dir = run_dir / "vehicle_identity_test" / "stationary_recovery"
        vehicles_payload = self._read_json(output_dir / "persistent_vehicles.json", default=None)
        persistent_map = self._read_json(output_dir / "persistent_vehicle_id_map.json", default={})
        evaluation = self._read_json(output_dir / "evaluation.json", default={})
        decisions = self._read_csv_rows(output_dir / "recovery_decisions.csv")
        scores = self._read_csv_rows(output_dir / "recovery_scores.csv")
        if not isinstance(vehicles_payload, dict):
            return {
                "run_id": run_id,
                "experimental": True,
                "stage": "stationary_recovery",
                "available": False,
                "message": "Stationary recovery experiment has not been run for this run.",
                "metrics": {},
                "analytics_simulation": {},
                "config": {},
                "calibration": {},
                "persistent_vehicles": [],
                "persistent_vehicle_id_map": {},
                "recovery_decisions": [],
                "recovery_scores": [],
                "paths": {},
            }
        vehicles = list(vehicles_payload.get("persistent_vehicles", []) or [])
        for vehicle in vehicles:
            if not isinstance(vehicle, dict):
                continue
            vehicle["contact_sheet_url"] = self._stationary_recovery_contact_sheet_url(run_id, str(vehicle.get("persistent_vehicle_id") or ""))
        return {
            "run_id": run_id,
            "experimental": True,
            "stage": "stationary_recovery",
            "available": True,
            "message": None,
            "metrics": evaluation.get("metrics", {}),
            "analytics_simulation": evaluation.get("analytics_simulation", {}),
            "config": evaluation.get("config", {}),
            "calibration": evaluation.get("calibration", {}),
            "persistent_vehicles": vehicles,
            "persistent_vehicle_id_map": persistent_map if isinstance(persistent_map, dict) else {},
            "recovery_decisions": decisions,
            "recovery_scores": scores,
            "paths": {
                "persistent_vehicles": str(output_dir / "persistent_vehicles.json"),
                "persistent_vehicle_id_map": str(output_dir / "persistent_vehicle_id_map.json"),
                "evaluation": str(output_dir / "evaluation.json"),
                "recovery_decisions": str(output_dir / "recovery_decisions.csv"),
                "recovery_scores": str(output_dir / "recovery_scores.csv"),
                "report": str(output_dir / "report.md"),
            },
        }

    def get_plate_assisted_identity_experiment(self, run_id: str) -> dict[str, Any] | None:
        run_dir = self._resolve_run_directory(run_id)
        if run_dir is None:
            return None
        output_dir = run_dir / "vehicle_identity_test" / "plate_assisted"
        vehicles_payload = self._read_json(output_dir / "vehicles.json", default=None)
        vehicle_id_map = self._read_json(output_dir / "vehicle_id_map.json", default={})
        evaluation = self._read_json(output_dir / "evaluation.json", default={})
        consensus_rows = self._read_json(output_dir / "track_plate_consensus.json", default=[])
        decisions = self._read_csv_rows(output_dir / "association_decisions.csv")
        pair_scores = self._read_csv_rows(output_dir / "plate_pair_scores.csv")
        identity_scores = self._read_csv_rows(output_dir / "identity_scores.csv")
        if not isinstance(vehicles_payload, dict):
            return {
                "run_id": run_id,
                "experimental": True,
                "stage": "plate_assisted_identity",
                "available": False,
                "message": "Plate-assisted identity experiment has not been run for this run.",
                "verification": {},
                "plate_coverage": {},
                "baseline_without_plate": {},
                "plate_assisted": {},
                "vehicles": [],
                "vehicle_id_map": {},
                "track_plate_consensus": [],
                "association_decisions": [],
                "plate_pair_scores": [],
                "identity_scores": [],
                "paths": {},
            }
        consensus_by_track = {
            str(item.get("local_track_id")): dict(item)
            for item in consensus_rows
            if isinstance(item, dict) and item.get("local_track_id")
        }
        decisions_by_pair = self._rows_by_track_pair(decisions)
        vehicles = list(vehicles_payload.get("vehicles", []) or [])
        for vehicle in vehicles:
            if not isinstance(vehicle, dict):
                continue
            member_tracks = [str(item) for item in list(vehicle.get("member_tracks", []) or [])]
            member_plate_rows = [self._serialize_plate_member(run_id, consensus_by_track.get(track_id, {})) for track_id in member_tracks]
            accepted_decisions = [
                decisions_by_pair[key]
                for key in self._pair_keys(member_tracks)
                if key in decisions_by_pair and str(decisions_by_pair[key].get("decision")) == "MERGE"
            ]
            plate_texts = [
                str(item.get("normalized_plate_text"))
                for item in member_plate_rows
                if item.get("normalized_plate_text") and str(item.get("plate_evidence_status")) != "NO READABLE PLATE"
            ]
            consensus_text = self._most_common_value(plate_texts)
            qualities = [str(item.get("quality") or "UNUSABLE") for item in member_plate_rows]
            vehicle["member_track_ids"] = member_tracks
            vehicle["plate"] = {
                "consensus_text": consensus_text,
                "quality": self._best_plate_quality(qualities),
                "status": self._plate_vehicle_status(member_plate_rows, accepted_decisions),
                "member_plates": member_plate_rows,
            }
            vehicle["representative_evidence"] = [
                {
                    "track_id": item.get("local_track_id"),
                    "vehicle_crop_url": item.get("vehicle_crop_url"),
                    "plate_crop_url": item.get("plate_crop_url"),
                    "plate_text": item.get("normalized_plate_text"),
                    "confidence": item.get("plate_text_confidence") or item.get("plate_detection_confidence"),
                    "quality": item.get("quality"),
                }
                for item in member_plate_rows
            ]
            vehicle["association_reasons"] = sorted(
                {
                    str(decision.get("decision_reason_codes") or decision.get("plate_reason_code") or decision.get("association_reason"))
                    for decision in accepted_decisions
                    if decision.get("decision_reason_codes") or decision.get("plate_reason_code") or decision.get("association_reason")
                }
            )
            vehicle["contact_sheet_url"] = self._plate_assisted_contact_sheet_url(run_id, member_tracks)
        return {
            "run_id": run_id,
            "experimental": True,
            "stage": "plate_assisted_identity",
            "available": True,
            "message": None,
            "verification": evaluation.get("verification", {}),
            "plate_coverage": evaluation.get("plate_coverage", {}),
            "baseline_without_plate": evaluation.get("baseline_without_plate", {}),
            "plate_assisted": evaluation.get("plate_assisted", {}),
            "examples": evaluation.get("examples", {}),
            "vehicles": vehicles,
            "vehicle_id_map": vehicle_id_map if isinstance(vehicle_id_map, dict) else {},
            "track_plate_consensus": consensus_rows if isinstance(consensus_rows, list) else [],
            "association_decisions": decisions,
            "plate_pair_scores": pair_scores,
            "identity_scores": identity_scores,
            "paths": {
                "vehicles": str(output_dir / "vehicles.json"),
                "vehicle_id_map": str(output_dir / "vehicle_id_map.json"),
                "track_plate_consensus": str(output_dir / "track_plate_consensus.json"),
                "association_decisions": str(output_dir / "association_decisions.csv"),
                "evaluation": str(output_dir / "evaluation.json"),
                "report": str(output_dir / "report.md"),
            },
        }

    def list_tracks(
        self,
        *,
        run_id: str | None = None,
        camera_id: str | None = None,
        vehicle_class: str | None = None,
        colour: str | None = None,
        track_id: str | None = None,
        from_time: float | None = None,
        to_time: float | None = None,
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        run_ids = self._resolve_run_ids(run_id)
        for candidate_run_id in run_ids:
            rows.extend(
                self._load_run_tracks(
                    run_id=candidate_run_id,
                    camera_id=camera_id,
                    vehicle_class=vehicle_class,
                    colour=colour,
                    track_id=track_id,
                    from_time=from_time,
                    to_time=to_time,
                    include_evidence=False,
                )
            )
        rows.sort(key=lambda item: (str(item.get("run_id", "")), str(item.get("camera_id", "")), str(item.get("track_id", ""))), reverse=True)
        return rows

    def get_track(self, *, camera_id: str, track_id: str, run_id: str | None = None) -> dict[str, Any] | None:
        run_ids = self._resolve_run_ids(run_id)
        for candidate_run_id in run_ids:
            for item in self._load_run_tracks(run_id=candidate_run_id, include_evidence=True):
                if str(item.get("camera_id")) != camera_id:
                    continue
                if str(item.get("track_id")) == track_id or str(item.get("local_track_id")) == track_id:
                    return item
        return None

    def get_track_evidence(self, *, camera_id: str, track_id: str, run_id: str | None = None) -> list[dict[str, Any]]:
        run_ids = self._resolve_run_ids(run_id)
        for candidate_run_id in run_ids:
            track = self.get_track(camera_id=camera_id, track_id=track_id, run_id=candidate_run_id)
            if track is not None:
                return list(track.get("evidence", []) or [])
        return []

    def list_cameras(self, *, run_id: str | None = None) -> list[dict[str, Any]]:
        resolved_run_id = self._resolve_single_run_id(run_id)
        if resolved_run_id is None:
            return []
        tracks = self._load_run_tracks(run_id=resolved_run_id, include_evidence=False)
        by_camera: dict[str, dict[str, Any]] = {}
        for item in tracks:
            camera_id = str(item.get("camera_id", "") or "")
            if not camera_id:
                continue
            camera = by_camera.setdefault(
                camera_id,
                {
                    "camera_id": camera_id,
                    "name": camera_id,
                    "status": "completed",
                    "frame_number": None,
                    "timestamp_seconds": None,
                    "processed_fps": 0.0,
                    "input_fps": None,
                    "active_vehicle_count": 0,
                    "active_track_ids": [],
                    "detections": [],
                    "last_update": None,
                    "run_id": resolved_run_id,
                    "latest_frame_url_parts": None,
                    "source_type": "saved_run",
                    "source": str(self._resolve_run_directory(resolved_run_id) or ""),
                },
            )
            camera["active_vehicle_count"] = int(camera["active_vehicle_count"]) + 1
            camera["active_track_ids"].append(str(item.get("track_id") or item.get("local_track_id") or ""))
            last_seen = item.get("last_seen_seconds")
            last_frame = item.get("last_frame")
            if last_seen is not None and (camera["timestamp_seconds"] is None or float(last_seen) >= float(camera["timestamp_seconds"])):
                camera["timestamp_seconds"] = float(last_seen)
                camera["frame_number"] = int(last_frame) if last_frame is not None else None
                camera["last_update"] = float(last_seen)
                camera["latest_frame_url_parts"] = self._tracked_frame_relative_parts(
                    run_id=resolved_run_id,
                    camera_id=camera_id,
                    frame_number=last_frame,
                )
        return sorted(by_camera.values(), key=lambda item: str(item["camera_id"]))

    def get_filter_options(self, *, run_id: str | None = None) -> dict[str, list[str]]:
        tracks = self.list_tracks(run_id=run_id)
        cameras = sorted({str(item.get("camera_id")) for item in tracks if item.get("camera_id")})
        runs = [str(item["run_id"]) for item in self.list_runs()]
        return {
            "cameras": cameras,
            "vehicle_classes": list(SUPPORTED_VEHICLE_CLASSES),
            "colours": list(SUPPORTED_VEHICLE_COLOUR_LABELS),
            "runs": runs,
        }

    def list_vehicle_records(self, *, run_id: str | None = None) -> list[Any]:
        vehicles: list[dict[str, Any]] = []
        for candidate_run_id in self._resolve_run_ids(run_id):
            vehicles.extend(self.list_physical_vehicles(run_id=candidate_run_id))
        if vehicles:
            return vehicle_records_from_physical_vehicles(vehicles)
        tracks: list[dict[str, Any]] = []
        for candidate_run_id in self._resolve_run_ids(run_id):
            tracks.extend(self._load_run_tracks(run_id=candidate_run_id, include_evidence=False))
        return vehicle_records_from_repository_tracks(tracks)

    def list_physical_vehicles(
        self,
        *,
        run_id: str | None = None,
        vehicle_class: str | None = None,
        colour: str | None = None,
        plate_text: str | None = None,
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for candidate_run_id in self._resolve_run_ids(run_id):
            run_dir = self._resolve_run_directory(candidate_run_id)
            if run_dir is None:
                continue
            payload = self._read_json(run_dir / "physical_vehicles.json", default={})
            vehicles = list(dict(payload or {}).get("physical_vehicles", []) or []) if isinstance(payload, dict) else []
            for vehicle in vehicles:
                if not isinstance(vehicle, dict):
                    continue
                record = dict(vehicle)
                record["run_id"] = candidate_run_id
                record["vehicle_id"] = str(record.get("vehicle_id") or record.get("vehicle_key") or "")
                if vehicle_class and str(record.get("vehicle_class", "")).upper() != vehicle_class.upper():
                    continue
                if colour and str(record.get("vehicle_colour", "")).upper() != colour.upper():
                    continue
                if plate_text and str(record.get("consensus_plate_text", "")).upper() != str(plate_text).upper():
                    continue
                rows.append(record)
        rows.sort(key=lambda item: (str(item.get("run_id", "")), float(item.get("last_seen_seconds") or 0.0)), reverse=True)
        return rows

    def _physical_vehicle_count(self, run_dir: Path) -> int:
        payload = self._read_json(run_dir / "physical_vehicles.json", default={})
        if not isinstance(payload, dict):
            return 0
        return len([item for item in list(payload.get("physical_vehicles", []) or []) if isinstance(item, dict)])

    def get_physical_vehicle(self, *, vehicle_id: str, run_id: str | None = None) -> dict[str, Any] | None:
        for vehicle in self.list_physical_vehicles(run_id=run_id):
            if str(vehicle.get("vehicle_id")) == vehicle_id or str(vehicle.get("vehicle_key")) == vehicle_id:
                return vehicle
        return None

    def resolve_media_path(self, *, run_id: str, category: str, relative_parts: list[str]) -> Path | None:
        run_dir = self._resolve_run_directory(run_id)
        if run_dir is None:
            return None
        base = self._category_base_directory(run_dir, category)
        if base is None:
            return None
        candidate = base
        for part in relative_parts:
            cleaned = str(part).strip()
            if cleaned in {"", ".", ".."} or "/" in cleaned or "\\" in cleaned:
                return None
            candidate = candidate / cleaned
        try:
            resolved = candidate.resolve()
        except FileNotFoundError:
            resolved = candidate
        if not self._is_relative_to(resolved, base.resolve()):
            return None
        if not resolved.exists() or not resolved.is_file():
            return None
        return resolved

    def _load_run_tracks(
        self,
        *,
        run_id: str,
        camera_id: str | None = None,
        vehicle_class: str | None = None,
        colour: str | None = None,
        track_id: str | None = None,
        from_time: float | None = None,
        to_time: float | None = None,
        include_evidence: bool = False,
    ) -> list[dict[str, Any]]:
        run_dir = self._resolve_run_directory(run_id)
        if run_dir is None:
            return []
        tracks = self._read_json(run_dir / "tracks.json", default=[])
        enrichments = self._read_json(run_dir / "vehicle_enrichment.json", default=[])
        enrichment_by_track = {
            str(item.get("local_track_id")): item
            for item in enrichments
            if isinstance(item, dict) and item.get("local_track_id")
        }
        results: list[dict[str, Any]] = []
        for item in tracks:
            if not isinstance(item, dict):
                continue
            local_track_id = str(item.get("local_track_id", ""))
            if not local_track_id:
                continue
            short_track_id = self._short_track_id(local_track_id)
            enrichment = enrichment_by_track.get(local_track_id, {})
            colour_payload = dict(enrichment.get("vehicle_colour", {}) or {})
            colour_label = colour_payload.get("label")
            plate_crop_path = enrichment.get("plate_crop_path")
            vehicle_class_value = enrichment.get("vehicle_class") or item.get("final_class")
            first_seen_seconds = self._coerce_float(item.get("first_timestamp_seconds"))
            last_seen_seconds = self._coerce_float(item.get("last_timestamp_seconds"))
            evidence_rows = (
                [self._normalize_evidence_item(run_id=run_id, payload=row) for row in list(enrichment.get("evidence_used", []) or [])]
                if include_evidence
                else []
            )
            record = {
                "run_id": run_id,
                "camera_id": item.get("camera_id"),
                "track_id": short_track_id,
                "local_track_id": local_track_id,
                "status": item.get("status"),
                "vehicle_class": vehicle_class_value,
                "colour": colour_label,
                "colour_status": colour_payload.get("status"),
                "plate_text": enrichment.get("plate_text"),
                "plate_detected": enrichment.get("plate_detected"),
                "plate_colour": enrichment.get("plate_colour"),
                "registration_category": enrichment.get("registration_category"),
                "plate_detection_confidence": enrichment.get("plate_detection_confidence"),
                "plate_text_confidence": enrichment.get("plate_text_confidence"),
                "plate_quality_status": enrichment.get("plate_quality_status"),
                "plate_ocr_reason": enrichment.get("plate_ocr_reason"),
                "plate_crop_path": plate_crop_path,
                "plate_crop_parts": self._path_to_media_reference(run_id=run_id, path_value=plate_crop_path),
                "first_seen": first_seen_seconds,
                "last_seen": last_seen_seconds,
                "first_seen_seconds": first_seen_seconds,
                "last_seen_seconds": last_seen_seconds,
                "duration_seconds": self._derive_duration_seconds(first_seen_seconds, last_seen_seconds),
                "first_frame": item.get("first_frame"),
                "last_frame": item.get("last_frame"),
                "observation_count": item.get("observation_count"),
                "completion_reason": item.get("completion_reason"),
                "vehicle_enrichment_status": enrichment.get("status"),
                "evidence": evidence_rows,
                "best_crop": self._best_crop_from_enrichment(enrichment),
                "best_crop_parts": self._path_to_media_reference(run_id=run_id, path_value=self._best_crop_from_enrichment(enrichment)),
                "available_crop_paths": list(enrichment.get("selected_crop_paths", []) or []),
                "colour_resolution": self._build_colour_resolution(colour_payload),
            }
            if camera_id and str(record["camera_id"]) != camera_id:
                continue
            if vehicle_class and str(record["vehicle_class"]).upper() != str(vehicle_class).upper():
                continue
            if colour and str(record["colour"]).upper() != str(colour).upper():
                continue
            if track_id and str(record["track_id"]) != track_id and str(record["local_track_id"]) != track_id:
                continue
            if from_time is not None and record["last_seen_seconds"] is not None and float(record["last_seen_seconds"]) < float(from_time):
                continue
            if to_time is not None and record["first_seen_seconds"] is not None and float(record["first_seen_seconds"]) > float(to_time):
                continue
            results.append(record)
        return results

    def _best_crop_from_enrichment(self, enrichment: dict[str, Any]) -> str | None:
        evidence = list(enrichment.get("evidence_used", []) or [])
        for item in evidence:
            if item.get("selected_for_colour") and item.get("vehicle_crop_path"):
                return str(item["vehicle_crop_path"])
        crop_paths = list(enrichment.get("selected_crop_paths", []) or [])
        return str(crop_paths[0]) if crop_paths else None

    def _iter_run_directories(self) -> list[Path]:
        if not self.outputs_root.exists():
            return []
        return [item for item in self.outputs_root.iterdir() if item.is_dir()]

    def _resolve_run_ids(self, run_id: str | None) -> list[str]:
        if run_id is None or str(run_id).strip() == "" or str(run_id).strip().lower() == "all":
            return [str(item["run_id"]) for item in self.list_runs()]
        if str(run_id).strip().lower() == "latest":
            latest = self.latest_run_id()
            return [latest] if latest else []
        return [str(run_id)]

    def _resolve_single_run_id(self, run_id: str | None) -> str | None:
        run_ids = self._resolve_run_ids(run_id)
        return run_ids[0] if run_ids else None

    def _resolve_run_directory(self, run_id: str) -> Path | None:
        candidate = (self.outputs_root / str(run_id)).resolve()
        if not candidate.exists() or not candidate.is_dir():
            return None
        if not self._is_relative_to(candidate, self.outputs_root.resolve()):
            return None
        return candidate

    def _category_base_directory(self, run_dir: Path, category: str) -> Path | None:
        mapping = {
            "evidence": run_dir / "evidence",
            "florence_selected_crops": run_dir / "05_florence_selected_crops",
            "track_crops": run_dir / "04_track_crops",
            "body_type_selected_crops": run_dir / "07_body_type_selected_crops",
            "tracked_frames": run_dir / "tracked_frames",
            "detected_frames": run_dir / "detected_frames",
            "raw_frames": run_dir / "raw_frames",
            "track_reconciliation_visual": run_dir / "track_reconciliation_test" / "visual_evidence",
            "vehicle_identity_visual": run_dir / "vehicle_identity_test" / "visual_evidence",
            "stationary_recovery_contact_sheets": run_dir / "vehicle_identity_test" / "stationary_recovery" / "contact_sheets",
            "plate_assisted_contact_sheets": run_dir / "vehicle_identity_test" / "plate_assisted" / "contact_sheets",
        }
        return mapping.get(category)

    def _vehicle_identity_contact_sheet_url(self, run_id: str, vehicle_id: str) -> str | None:
        if not vehicle_id:
            return None
        path = self.resolve_media_path(run_id=run_id, category="vehicle_identity_visual", relative_parts=[f"{vehicle_id}.jpg"])
        if path is None:
            return None
        return f"/api/media/vehicle_identity_visual/{run_id}/{vehicle_id}.jpg"

    def _stationary_recovery_contact_sheet_url(self, run_id: str, persistent_vehicle_id: str) -> str | None:
        if not persistent_vehicle_id:
            return None
        path = self.resolve_media_path(run_id=run_id, category="stationary_recovery_contact_sheets", relative_parts=[f"{persistent_vehicle_id}.jpg"])
        if path is None:
            return None
        return f"/api/media/stationary_recovery_contact_sheets/{run_id}/{persistent_vehicle_id}.jpg"

    def _plate_assisted_contact_sheet_url(self, run_id: str, member_tracks: list[str]) -> str | None:
        run_dir = self._resolve_run_directory(run_id)
        if run_dir is None or len(member_tracks) < 2:
            return None
        contact_dir = run_dir / "vehicle_identity_test" / "plate_assisted" / "contact_sheets"
        if not contact_dir.exists():
            return None
        short_ids = [self._short_track_id(track_id) for track_id in member_tracks]
        for path in sorted(contact_dir.glob("*.jpg")):
            name = path.stem
            if name.startswith("same__") and all(short_id in name for short_id in short_ids):
                return f"/api/media/plate_assisted_contact_sheets/{run_id}/{path.name}"
        return None

    def _read_json(self, path: Path, *, default: Any) -> Any:
        if not path.exists():
            return default
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return default

    def _read_csv_rows(self, path: Path) -> list[dict[str, str]]:
        if not path.exists():
            return []
        try:
            with path.open("r", encoding="utf-8", newline="") as handle:
                return list(csv.DictReader(handle))
        except Exception:
            return []

    def _read_yaml_text(self, path: Path) -> str | None:
        if not path.exists():
            return None
        try:
            return path.read_text(encoding="utf-8")
        except Exception:
            return None

    def _is_relative_to(self, path: Path, root: Path) -> bool:
        try:
            path.relative_to(root)
            return True
        except ValueError:
            return False

    def _short_track_id(self, local_track_id: str) -> str:
        parts = str(local_track_id).split(":")
        return parts[-1] if parts else local_track_id

    def _duration_seconds_from_summary(self, summary: dict[str, Any]) -> float | None:
        runtime_ms = summary.get("overall_pipeline_runtime_ms")
        if runtime_ms is None:
            return None
        try:
            return float(runtime_ms) / 1000.0
        except (TypeError, ValueError):
            return None

    def _coerce_float(self, value: Any) -> float | None:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _derive_duration_seconds(self, first_seen: float | None, last_seen: float | None) -> float | None:
        if first_seen is None or last_seen is None:
            return None
        return max(0.0, float(last_seen) - float(first_seen))

    def _build_colour_resolution(self, colour_payload: dict[str, Any]) -> list[dict[str, Any]]:
        predictions = list(colour_payload.get("predictions", []) or [])
        rows: list[dict[str, Any]] = []
        for index, item in enumerate(predictions, start=1):
            if not isinstance(item, dict):
                continue
            rows.append(
                {
                    "index": index,
                    "label": item.get("label"),
                    "frame_number": item.get("source_frame_number"),
                    "evidence_role": item.get("evidence_role"),
                    "quality_weight": item.get("quality_weight"),
                    "status": item.get("status"),
                    "reason": item.get("reason"),
                    "crop_path": item.get("source_crop_path"),
                }
            )
        return rows

    def _normalize_evidence_item(self, *, run_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        item = dict(payload)
        item["timestamp_seconds"] = self._coerce_float(item.get("timestamp_seconds"))
        item["crop_media"] = self._path_to_media_reference(run_id=run_id, path_value=item.get("vehicle_crop_path"))
        item["full_frame_media"] = self._path_to_media_reference(run_id=run_id, path_value=item.get("annotated_frame_path") or item.get("source_image_path"))
        return item

    def _path_to_media_reference(self, *, run_id: str, path_value: Any) -> dict[str, Any] | None:
        if not path_value:
            return None
        run_dir = self._resolve_run_directory(run_id)
        if run_dir is None:
            return None
        try:
            path = Path(str(path_value)).resolve()
        except Exception:
            return None
        category_roots = {
            "evidence": run_dir / "evidence",
            "florence_selected_crops": run_dir / "05_florence_selected_crops",
            "track_crops": run_dir / "04_track_crops",
            "body_type_selected_crops": run_dir / "07_body_type_selected_crops",
            "tracked_frames": run_dir / "tracked_frames",
            "detected_frames": run_dir / "detected_frames",
            "raw_frames": run_dir / "raw_frames",
            "track_reconciliation_visual": run_dir / "track_reconciliation_test" / "visual_evidence",
            "plate_assisted_contact_sheets": run_dir / "vehicle_identity_test" / "plate_assisted" / "contact_sheets",
        }
        for category, root in category_roots.items():
            try:
                relative = path.relative_to(root.resolve())
            except ValueError:
                continue
            parts = [item for item in relative.parts if item]
            return {
                "category": category,
                "run_id": run_id,
                "parts": parts,
                "filename": parts[-1] if parts else None,
            }
        return None

    def _tracked_frame_relative_parts(self, *, run_id: str, camera_id: str, frame_number: Any) -> list[str] | None:
        if frame_number is None:
            return None
        try:
            frame_index = int(frame_number)
        except (TypeError, ValueError):
            return None
        run_dir = self._resolve_run_directory(run_id)
        if run_dir is None:
            return None
        candidate = run_dir / "tracked_frames" / camera_id / f"frame_{frame_index:06d}.jpg"
        if candidate.exists():
            return [camera_id, candidate.name]
        return None

    def _serialize_plate_member(self, run_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        item = dict(payload)
        item["quality"] = item.get("reliability_label") or "UNUSABLE"
        item["plate_evidence_status"] = self._plate_member_status(item)
        item["vehicle_crop_media"] = self._path_to_media_reference(run_id=run_id, path_value=item.get("vehicle_crop_path"))
        item["plate_crop_media"] = self._path_to_media_reference(run_id=run_id, path_value=item.get("plate_crop_path"))
        item["vehicle_crop_url"] = self._media_reference_url(item.get("vehicle_crop_media"))
        item["plate_crop_url"] = self._media_reference_url(item.get("plate_crop_media"))
        return item

    def _media_reference_url(self, media: dict[str, Any] | None) -> str | None:
        if not media:
            return None
        category = str(media.get("category") or "")
        run_id = str(media.get("run_id") or "")
        parts = [str(item) for item in list(media.get("parts", []) or []) if str(item)]
        if not category or not run_id or not parts:
            return None
        return f"/api/media/{category}/{run_id}/{'/'.join(parts)}"

    def _plate_member_status(self, item: dict[str, Any]) -> str:
        text = str(item.get("normalized_plate_text") or "")
        if not text:
            return "NO READABLE PLATE"
        quality = str(item.get("reliability_label") or "").upper()
        if quality == "HIGH" and len(text) >= 9:
            return "EXACT / HIGH QUALITY"
        if quality in {"HIGH", "MEDIUM"}:
            return "PARTIAL"
        return "LOW CONFIDENCE"

    def _plate_vehicle_status(self, members: list[dict[str, Any]], decisions: list[dict[str, str]]) -> str:
        reason_text = " ".join(str(item.get("decision_reason_codes") or item.get("plate_reason_code") or "") for item in decisions)
        if "PLATE_EXACT_MATCH" in reason_text:
            return "EXACT / HIGH QUALITY"
        if "PLATE_PARTIAL_MATCH" in reason_text:
            return "PARTIAL"
        if any(str(item.get("plate_evidence_status")) == "EXACT / HIGH QUALITY" for item in members):
            return "EXACT / HIGH QUALITY"
        if any(str(item.get("plate_evidence_status")) == "PARTIAL" for item in members):
            return "PARTIAL"
        return "NO READABLE PLATE"

    def _best_plate_quality(self, qualities: list[str]) -> str:
        order = {"HIGH": 3, "MEDIUM": 2, "LOW": 1, "UNUSABLE": 0}
        if not qualities:
            return "UNUSABLE"
        return max(qualities, key=lambda item: order.get(str(item).upper(), 0))

    def _most_common_value(self, values: list[str]) -> str | None:
        if not values:
            return None
        counts: dict[str, int] = {}
        for value in values:
            counts[value] = counts.get(value, 0) + 1
        return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]

    def _rows_by_track_pair(self, rows: list[dict[str, str]]) -> dict[tuple[str, str], dict[str, str]]:
        return {
            tuple(sorted((str(row.get("track_a")), str(row.get("track_b"))))): row
            for row in rows
            if row.get("track_a") and row.get("track_b")
        }

    def _pair_keys(self, track_ids: list[str]) -> list[tuple[str, str]]:
        return [
            tuple(sorted((track_ids[index], track_ids[other_index])))
            for index in range(len(track_ids))
            for other_index in range(index + 1, len(track_ids))
        ]

    def _build_reconciliation_visual_evidence(self, *, run_id: str, experiment_dir: Path) -> list[dict[str, Any]]:
        root = experiment_dir / "visual_evidence"
        if not root.exists():
            return []
        rows: list[dict[str, Any]] = []
        for result_dir in sorted([item for item in root.iterdir() if item.is_dir()]):
            for pair_dir in sorted([item for item in result_dir.iterdir() if item.is_dir()]):
                contact = pair_dir / "before_after_contact_sheet.jpg"
                before = next((item for item in (pair_dir / "before_occlusion").glob("*.jpg")), None) if (pair_dir / "before_occlusion").exists() else None
                after = next((item for item in (pair_dir / "after_occlusion").glob("*.jpg")), None) if (pair_dir / "after_occlusion").exists() else None
                rows.append(
                    {
                        "result": result_dir.name,
                        "pair_key": pair_dir.name,
                        "contact_sheet_url": self._reconciliation_media_url(run_id, root, contact),
                        "before_url": self._reconciliation_media_url(run_id, root, before),
                        "after_url": self._reconciliation_media_url(run_id, root, after),
                    }
                )
        return rows

    def _reconciliation_media_url(self, run_id: str, root: Path, path: Path | None) -> str | None:
        if path is None or not path.exists():
            return None
        try:
            relative = path.relative_to(root.resolve())
        except ValueError:
            return None
        parts = "/".join(relative.parts)
        return f"/api/media/track_reconciliation_visual/{run_id}/{parts}"


def read_csv_rows(path: str | Path) -> list[dict[str, str]]:
    csv_path = Path(path)
    if not csv_path.exists():
        return []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))
