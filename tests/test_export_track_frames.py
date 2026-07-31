from __future__ import annotations

import csv
import json
from pathlib import Path

from scripts.export_track_frames import export_track_frames, find_latest_run_directory


def test_export_track_frames_copies_saved_frames_and_records_missing_ones(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "20260730_140000"
    tracked_dir = run_dir / "tracked_frames" / "CAM_001"
    tracked_dir.mkdir(parents=True)
    observations_path = run_dir / "observations.csv"
    observations_path.write_text(
        "\n".join(
            [
                "local_track_id,camera_id,tracker_namespace,native_tracker_id,frame_number,timestamp_seconds,x1,y1,x2,y2,confidence,raw_class_id,raw_class_name",
                "CAM_001:TRACK_2,CAM_001,camera,2,0,0.0,1,2,3,4,0.9,3,motorcycle",
                "CAM_001:TRACK_2,CAM_001,camera,2,2,0.2,1,2,3,4,0.8,3,motorcycle",
                "CAM_001:TRACK_2,CAM_001,camera,2,4,0.4,1,2,3,4,0.7,3,motorcycle",
                "CAM_001:TRACK_9,CAM_001,camera,9,6,0.6,1,2,3,4,0.6,0,car",
            ]
        ),
        encoding="utf-8",
    )
    (tracked_dir / "frame_000000.jpg").write_bytes(b"a")
    (tracked_dir / "frame_000004.jpg").write_bytes(b"b")

    summary = export_track_frames(run_dir=run_dir, local_track_id="CAM_001:TRACK_2")

    export_dir = Path(summary["export_directory"])
    assert summary["observation_count"] == 3
    assert summary["copied_frame_count"] == 2
    assert summary["missing_frame_count"] == 1
    assert (export_dir / "frame_000000.jpg").exists()
    assert not (export_dir / "frame_000002.jpg").exists()
    assert (export_dir / "frame_000004.jpg").exists()

    manifest_path = export_dir / "manifest.csv"
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [int(row["frame_number"]) for row in rows] == [0, 2, 4]
    assert [row["frame_saved"] for row in rows] == ["True", "False", "True"]

    summary_json = json.loads((export_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary_json["local_track_id"] == "CAM_001:TRACK_2"
    assert summary_json["first_frame"] == 0
    assert summary_json["last_frame"] == 4


def test_find_latest_run_directory_returns_most_recent_folder(tmp_path: Path) -> None:
    runs_root = tmp_path / "outputs" / "runs"
    runs_root.mkdir(parents=True)
    older = runs_root / "20260730_120000"
    newer = runs_root / "20260730_130000"
    older.mkdir()
    newer.mkdir()
    (older / "marker.txt").write_text("older", encoding="utf-8")
    (newer / "marker.txt").write_text("newer", encoding="utf-8")

    latest = find_latest_run_directory(runs_root)

    assert latest == newer.resolve()
