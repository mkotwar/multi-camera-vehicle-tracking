from __future__ import annotations

import csv
import json
from pathlib import Path

from src.importers.run_file_importer import build_dry_run, main


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else ["local_track_id"])
        writer.writeheader()
        writer.writerows(rows)


def _base_run(tmp_path: Path) -> Path:
    run_dir = tmp_path / "20260812_192758"
    crop = run_dir / "evidence" / "CAM_001" / "TRACK_1" / "crop.jpg"
    crop.parent.mkdir(parents=True, exist_ok=True)
    crop.write_bytes(b"image")
    _write_json(
        run_dir / "run_metadata.json",
        {
            "run_id": "20260812_192758",
            "project_name": "multicamera_vehicle_tracking",
            "started_at": "2026-08-12T13:57:58+00:00",
            "completed_at": "2026-08-12T14:03:22+00:00",
            "status": "COMPLETED",
            "processed_frames": 10,
            "completed_tracks": 1,
            "error_count": 0,
            "config_path": "config.yaml",
        },
    )
    _write_json(
        run_dir / "summary.json",
        {
            "run_id": "20260812_192758",
            "status": "COMPLETED",
            "project_name": "multicamera_vehicle_tracking",
            "detection_backend": "ocr_mukul",
            "tracking_backend": "bytetrack",
            "vehicle_enrichment_enabled": True,
            "frames_by_camera": {"CAM_001": 10},
            "detections_by_camera": {"CAM_001": 3},
            "tracks_discarded_by_camera": {"CAM_001": 1},
        },
    )
    (run_dir / "run_config.yaml").write_text(
        "input:\n  cameras:\n    - camera_id: CAM_001\n      source_type: video\n      source: video.mp4\n      enabled: true\ningestion:\n  target_read_fps: 10\n",
        encoding="utf-8",
    )
    _write_json(
        run_dir / "tracks.json",
        [
            {
                "local_track_id": "CAM_001:TRACK_1",
                "camera_id": "CAM_001",
                "tracker_namespace": "camera",
                "native_tracker_id": 1,
                "status": "COMPLETED",
                "completion_reason": "END_OF_STREAM",
                "first_frame": 1,
                "last_frame": 5,
                "first_timestamp_seconds": 0.1,
                "last_timestamp_seconds": 0.5,
                "observation_count": 1,
                "lost_frames": 0,
                "final_class": "car",
                "class_counts": {"car": 1},
                "class_confidence_sums": {"car": 0.9},
                "new_debug_field": "kept",
            },
            {
                "local_track_id": "CAM_001:TRACK_2",
                "camera_id": "CAM_001",
                "status": "DISCARDED",
                "first_frame": 2,
                "last_frame": 2,
                "first_timestamp_seconds": 0.2,
                "last_timestamp_seconds": 0.2,
                "observation_count": 0,
                "final_class": "",
                "class_counts": {},
                "class_confidence_sums": {},
            },
        ],
    )
    _write_csv(
        run_dir / "observations.csv",
        [
            {
                "local_track_id": "CAM_001:TRACK_1",
                "camera_id": "CAM_001",
                "tracker_namespace": "camera",
                "native_tracker_id": 1,
                "frame_number": 1,
                "timestamp_seconds": 0.1,
                "x1": 1,
                "y1": 2,
                "x2": 3,
                "y2": 4,
                "confidence": 0.9,
                "raw_class_id": 2,
                "raw_class_name": "car",
            }
        ],
    )
    _write_json(
        run_dir / "evidence_index.json",
        [
            {
                "local_track_id": "CAM_001:TRACK_1",
                "camera_id": "CAM_001",
                "role": "FIRST",
                "frame_number": 1,
                "timestamp_seconds": 0.1,
                "bbox_xyxy": [1, 2, 3, 4],
                "confidence": 0.9,
                "best_overall_score": 0.7,
                "crop_path": str(crop),
                "original_crop_width": 100,
                "original_crop_height": 80,
            }
        ],
    )
    _write_json(
        run_dir / "vehicle_enrichment.json",
        [
            {
                "local_track_id": "CAM_001:TRACK_1",
                "camera_id": "CAM_001",
                "vehicle_class": "CAR",
                "vehicle_class_confidence": 0.9,
                "vehicle_colour": {
                    "label": "white",
                    "status": "completed",
                    "model": "florence",
                    "source": "adapter",
                    "predictions": [
                        {
                            "label": "WHITE",
                            "status": "completed",
                            "confidence": 0.8,
                            "source_crop_path": str(crop),
                            "raw_response": "white",
                        }
                    ],
                },
                "vehicle_body_type": {"label": "UNKNOWN", "status": "disabled", "model": "florence"},
                "plate_detected": False,
                "plate_text": None,
                "selected_crop_paths": [str(crop)],
                "status": "completed",
            }
        ],
    )
    return run_dir


def test_parses_valid_track(tmp_path: Path) -> None:
    run_dir = _base_run(tmp_path)
    report = build_dry_run(run_dir)
    track = report.rows.vehicle_tracks[0]
    assert track.ref.local_track_id == "CAM_001:TRACK_1"
    assert track.vehicle_class == "CAR"
    assert track.raw_track["new_debug_field"] == "kept"


def test_preserves_completed_and_discarded_status(tmp_path: Path) -> None:
    report = build_dry_run(_base_run(tmp_path))
    assert report.counts["tracks"]["completed"] == 1
    assert report.counts["tracks"]["discarded"] == 1
    assert report.counts["tracks"]["searchable_by_default"] == 1


def test_observation_maps_to_correct_logical_track(tmp_path: Path) -> None:
    report = build_dry_run(_base_run(tmp_path))
    assert report.rows.track_observations[0].ref.key == "20260812_192758|CAM_001|CAM_001:TRACK_1"


def test_orphan_observation_is_reported(tmp_path: Path) -> None:
    run_dir = _base_run(tmp_path)
    _write_csv(
        run_dir / "observations.csv",
        [
            {
                "local_track_id": "CAM_001:TRACK_99",
                "camera_id": "CAM_001",
                "tracker_namespace": "camera",
                "native_tracker_id": 99,
                "frame_number": 1,
                "timestamp_seconds": 0.1,
                "x1": 1,
                "y1": 2,
                "x2": 3,
                "y2": 4,
                "confidence": 0.9,
                "raw_class_id": 2,
                "raw_class_name": "car",
            }
        ],
    )
    report = build_dry_run(run_dir)
    assert any(issue.code == "orphan_observation" for issue in report.issues)


def test_evidence_maps_to_correct_logical_track(tmp_path: Path) -> None:
    report = build_dry_run(_base_run(tmp_path))
    assert report.rows.track_evidence[0].ref.key == "20260812_192758|CAM_001|CAM_001:TRACK_1"
    assert report.rows.track_evidence[0].evidence_role == "FIRST"


def test_media_absolute_path_normalizes_to_relative_path(tmp_path: Path) -> None:
    report = build_dry_run(_base_run(tmp_path))
    crop_media = next(row for row in report.rows.media_assets if row.media_type == "crop")
    assert crop_media.relative_path == "evidence/CAM_001/TRACK_1/crop.jpg"
    assert crop_media.exists is True


def test_missing_media_warns_not_crashes(tmp_path: Path) -> None:
    run_dir = _base_run(tmp_path)
    evidence = json.loads((run_dir / "evidence_index.json").read_text(encoding="utf-8"))
    evidence[0]["crop_path"] = str(run_dir / "missing.jpg")
    _write_json(run_dir / "evidence_index.json", evidence)
    report = build_dry_run(run_dir)
    assert any(issue.code == "missing_media_file" and issue.severity == "WARNING" for issue in report.issues)


def test_null_vs_unknown_semantics_are_preserved(tmp_path: Path) -> None:
    report = build_dry_run(_base_run(tmp_path))
    completed = report.rows.vehicle_tracks[0]
    discarded = report.rows.vehicle_tracks[1]
    assert completed.body_type == "UNKNOWN"
    assert completed.plate_text is None
    assert discarded.vehicle_colour is None


def test_colour_enrichment_produces_final_value_and_prediction(tmp_path: Path) -> None:
    report = build_dry_run(_base_run(tmp_path))
    assert report.rows.vehicle_tracks[0].vehicle_colour == "WHITE"
    assert len(report.rows.colour_predictions) == 1
    assert report.rows.colour_predictions[0].normalized_colour == "WHITE"


def test_unknown_source_fields_are_not_silently_lost(tmp_path: Path) -> None:
    report = build_dry_run(_base_run(tmp_path))
    assert "tracks.new_debug_field" in report.field_mapping["jsonb"]
    assert report.field_mapping["unresolved"] == []


def test_duplicate_completed_logical_track_identity_is_detected(tmp_path: Path) -> None:
    run_dir = _base_run(tmp_path)
    tracks = json.loads((run_dir / "tracks.json").read_text(encoding="utf-8"))
    tracks.append(tracks[0])
    _write_json(run_dir / "tracks.json", tracks)
    report = build_dry_run(run_dir)
    assert any(issue.code == "duplicate_completed_logical_track_identity" for issue in report.issues)


def test_reused_native_tracker_id_with_distinct_logical_ids_passes_integrity(tmp_path: Path) -> None:
    run_dir = _base_run(tmp_path)
    tracks = json.loads((run_dir / "tracks.json").read_text(encoding="utf-8"))
    tracks[0]["native_tracker_id"] = 91
    tracks.append(
        {
            **tracks[0],
            "local_track_id": "CAM_001:TRACK_2",
            "native_tracker_id": 91,
            "first_frame": 20,
            "last_frame": 24,
            "first_timestamp_seconds": 2.0,
            "last_timestamp_seconds": 2.4,
        }
    )
    tracks[1] = {
        **tracks[1],
        "local_track_id": "CAM_001:TRACK_3",
    }
    _write_json(run_dir / "tracks.json", tracks)
    _write_csv(
        run_dir / "observations.csv",
        [
            {
                "local_track_id": "CAM_001:TRACK_1",
                "camera_id": "CAM_001",
                "tracker_namespace": "camera",
                "native_tracker_id": 91,
                "frame_number": 1,
                "timestamp_seconds": 0.1,
                "x1": 1,
                "y1": 2,
                "x2": 3,
                "y2": 4,
                "confidence": 0.9,
                "raw_class_id": 2,
                "raw_class_name": "car",
            },
            {
                "local_track_id": "CAM_001:TRACK_2",
                "camera_id": "CAM_001",
                "tracker_namespace": "camera",
                "native_tracker_id": 91,
                "frame_number": 20,
                "timestamp_seconds": 2.0,
                "x1": 5,
                "y1": 6,
                "x2": 7,
                "y2": 8,
                "confidence": 0.8,
                "raw_class_id": 2,
                "raw_class_name": "car",
            },
        ],
    )

    report = build_dry_run(run_dir)

    assert report.counts["issues"]["ERROR"] == 0
    assert not any(issue.code == "duplicate_completed_logical_track_identity" for issue in report.issues)
    assert [row.ref.local_track_id for row in report.rows.vehicle_tracks if row.track_status == "COMPLETED"] == [
        "CAM_001:TRACK_1",
        "CAM_001:TRACK_2",
    ]


def test_duplicate_discarded_track_identity_is_remapped_for_import(tmp_path: Path) -> None:
    run_dir = _base_run(tmp_path)
    tracks = json.loads((run_dir / "tracks.json").read_text(encoding="utf-8"))
    tracks.append(
        {
            **tracks[0],
            "status": "DISCARDED",
            "first_frame": 9,
            "last_frame": 9,
            "observation_count": 1,
        }
    )
    _write_json(run_dir / "tracks.json", tracks)
    _write_csv(
        run_dir / "observations.csv",
        [
            {
                "local_track_id": "CAM_001:TRACK_1",
                "camera_id": "CAM_001",
                "tracker_namespace": "camera",
                "native_tracker_id": 1,
                "frame_number": 1,
                "timestamp_seconds": 0.1,
                "x1": 1,
                "y1": 2,
                "x2": 3,
                "y2": 4,
                "confidence": 0.9,
                "raw_class_id": 2,
                "raw_class_name": "car",
            },
            {
                "local_track_id": "CAM_001:TRACK_1",
                "camera_id": "CAM_001",
                "tracker_namespace": "camera",
                "native_tracker_id": 1,
                "frame_number": 9,
                "timestamp_seconds": 0.9,
                "x1": 1,
                "y1": 2,
                "x2": 3,
                "y2": 4,
                "confidence": 0.9,
                "raw_class_id": 2,
                "raw_class_name": "car",
            },
        ],
    )

    report = build_dry_run(run_dir)

    assert report.counts["issues"]["ERROR"] == 0
    assert any(issue.code == "duplicate_noncanonical_track_identity_remapped" for issue in report.issues)
    remapped = next(row for row in report.rows.vehicle_tracks if row.ref.local_track_id.startswith("CAM_001:TRACK_1__DUPLICATE"))
    assert remapped.track_status == "DISCARDED"
    assert remapped.raw_track["original_local_track_id"] == "CAM_001:TRACK_1"
    assert any(row.ref.local_track_id == remapped.ref.local_track_id and row.frame_number == 9 for row in report.rows.track_observations)


def test_missing_tracks_json_is_reported(tmp_path: Path) -> None:
    run_dir = _base_run(tmp_path)
    (run_dir / "tracks.json").unlink()
    report = build_dry_run(run_dir)
    assert any(issue.code == "missing_required_file" for issue in report.issues)


def test_malformed_json_is_reported(tmp_path: Path) -> None:
    run_dir = _base_run(tmp_path)
    (run_dir / "tracks.json").write_text("{not-json", encoding="utf-8")
    report = build_dry_run(run_dir)
    assert any(issue.code == "malformed_json" for issue in report.issues)


def test_invalid_csv_row_is_reported(tmp_path: Path) -> None:
    run_dir = _base_run(tmp_path)
    _write_csv(
        run_dir / "observations.csv",
        [
            {
                "local_track_id": "CAM_001:TRACK_1",
                "camera_id": "CAM_001",
                "tracker_namespace": "camera",
                "native_tracker_id": 1,
                "frame_number": "not-a-number",
                "timestamp_seconds": 0.1,
                "x1": 1,
                "y1": 2,
                "x2": 3,
                "y2": 4,
                "confidence": 0.9,
                "raw_class_id": 2,
                "raw_class_name": "car",
            }
        ],
    )
    report = build_dry_run(run_dir)
    assert any(issue.code == "invalid_observation_time" for issue in report.issues)


def test_orphan_evidence_is_reported(tmp_path: Path) -> None:
    run_dir = _base_run(tmp_path)
    evidence = json.loads((run_dir / "evidence_index.json").read_text(encoding="utf-8"))
    evidence[0]["local_track_id"] = "CAM_001:TRACK_99"
    _write_json(run_dir / "evidence_index.json", evidence)
    report = build_dry_run(run_dir)
    assert any(issue.code == "orphan_evidence" for issue in report.issues)


def test_dry_run_executes_without_supabase_client_call(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.delenv("SUPABASE_KEY", raising=False)
    run_dir = _base_run(tmp_path)
    assert main(["--run-dir", str(run_dir), "--dry-run"]) == 0
