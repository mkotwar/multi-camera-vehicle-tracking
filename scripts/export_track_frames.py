from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export saved frames for one local_track_id from a pipeline run.")
    parser.add_argument("--local-track-id", required=True, help="Track ID to export, for example CAM_001:TRACK_2.")
    parser.add_argument(
        "--run-dir",
        help="Run directory that contains observations.csv and tracked_frames/. Defaults to the latest outputs/runs/<run_id> folder.",
    )
    parser.add_argument(
        "--frame-source",
        default="tracked_frames",
        choices=["tracked_frames", "detected_frames", "raw_frames"],
        help="Which saved frame directory to export from.",
    )
    parser.add_argument(
        "--output-dir",
        help="Optional destination directory. Defaults to <run_dir>/track_exports/<sanitized_track_id>/.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve() if args.run_dir else find_latest_run_directory(Path("outputs") / "runs")
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else None
    summary = export_track_frames(
        run_dir=run_dir,
        local_track_id=args.local_track_id,
        frame_source=args.frame_source,
        output_dir=output_dir,
    )
    print(json.dumps(summary, indent=2))
    return 0


def export_track_frames(
    *,
    run_dir: Path,
    local_track_id: str,
    frame_source: str = "tracked_frames",
    output_dir: Path | None = None,
) -> dict[str, Any]:
    resolved_run_dir = run_dir.expanduser().resolve()
    if not resolved_run_dir.exists():
        raise FileNotFoundError(f"Run directory does not exist: {resolved_run_dir}")

    observations_path = resolved_run_dir / "observations.csv"
    if not observations_path.exists():
        raise FileNotFoundError(f"observations.csv was not found in: {resolved_run_dir}")

    rows = _load_track_rows(observations_path, local_track_id)
    if not rows:
        raise ValueError(f"Track ID was not found in observations.csv: {local_track_id}")

    cameras = sorted({row["camera_id"] for row in rows})
    if len(cameras) != 1:
        raise ValueError(f"Track ID must resolve to one camera, found cameras={cameras} for {local_track_id}")
    camera_id = cameras[0]

    frame_root = resolved_run_dir / frame_source / camera_id
    if not frame_root.exists():
        raise FileNotFoundError(f"Frame source directory does not exist: {frame_root}")

    export_dir = (output_dir.expanduser().resolve() if output_dir else resolved_run_dir / "track_exports" / _sanitize_track_id(local_track_id))
    export_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, Any]] = []
    copied_count = 0
    missing_count = 0

    for row in rows:
        frame_number = int(row["frame_number"])
        frame_name = f"frame_{frame_number:06d}.jpg"
        source_path = frame_root / frame_name
        destination_path = export_dir / frame_name
        frame_saved = source_path.exists()
        if frame_saved:
            shutil.copy2(source_path, destination_path)
            copied_count += 1
        else:
            missing_count += 1
        manifest_rows.append(
            {
                "local_track_id": local_track_id,
                "camera_id": camera_id,
                "tracker_namespace": row["tracker_namespace"],
                "native_tracker_id": row["native_tracker_id"],
                "frame_number": frame_number,
                "timestamp_seconds": row["timestamp_seconds"],
                "frame_source": frame_source,
                "source_path": str(source_path),
                "exported_path": str(destination_path) if frame_saved else "",
                "frame_saved": frame_saved,
                "raw_class_name": row["raw_class_name"],
                "confidence": row["confidence"],
            }
        )

    _write_manifest(export_dir / "manifest.csv", manifest_rows)
    summary = {
        "run_directory": str(resolved_run_dir),
        "local_track_id": local_track_id,
        "camera_id": camera_id,
        "frame_source": frame_source,
        "export_directory": str(export_dir),
        "observation_count": len(rows),
        "copied_frame_count": copied_count,
        "missing_frame_count": missing_count,
        "first_frame": min(int(row["frame_number"]) for row in rows),
        "last_frame": max(int(row["frame_number"]) for row in rows),
    }
    (export_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def find_latest_run_directory(runs_root: Path) -> Path:
    resolved_root = runs_root.expanduser().resolve()
    if not resolved_root.exists():
        raise FileNotFoundError(f"Runs root does not exist: {resolved_root}")
    candidates = [path for path in resolved_root.iterdir() if path.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No run directories were found under: {resolved_root}")
    return max(candidates, key=lambda path: path.name)


def _load_track_rows(observations_path: Path, local_track_id: str) -> list[dict[str, str]]:
    with observations_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = [row for row in reader if row["local_track_id"] == local_track_id]
    rows.sort(key=lambda row: int(row["frame_number"]))
    return rows


def _write_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "local_track_id",
        "camera_id",
        "tracker_namespace",
        "native_tracker_id",
        "frame_number",
        "timestamp_seconds",
        "frame_source",
        "source_path",
        "exported_path",
        "frame_saved",
        "raw_class_name",
        "confidence",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _sanitize_track_id(local_track_id: str) -> str:
    return local_track_id.replace(":", "_").replace("/", "_").replace("\\", "_")


if __name__ == "__main__":
    raise SystemExit(main())
