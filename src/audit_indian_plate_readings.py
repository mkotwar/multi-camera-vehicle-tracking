from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from src.env_loader import load_env_file
from src.importers.db_writer import DEFAULT_DB_SCHEMA
from src.indian_plate_validator import validate_indian_plate


def _database_config(env_path: str | Path = ".env") -> tuple[str, str]:
    values = load_env_file(env_path)
    dsn = values.get("DATABASE_URL") or values.get("SUPABASE_DB_URL") or values.get("POSTGRES_URL")
    if not dsn:
        raise RuntimeError("Set DATABASE_URL, SUPABASE_DB_URL, or POSTGRES_URL before running the audit.")
    schema = values.get("DB_SCHEMA") or DEFAULT_DB_SCHEMA
    return dsn, schema


def audit_postgres_plate_readings(*, env_path: str | Path = ".env", sample_limit: int = 20) -> dict[str, Any]:
    try:
        import psycopg
        from psycopg.rows import dict_row
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("The PostgreSQL audit requires psycopg in the active environment.") from exc

    dsn, schema = _database_config(env_path)
    query = f"""
        select
            pr.id::text as plate_reading_id,
            pr.track_id::text as track_id,
            pr.raw_text,
            pr.normalized_text,
            pr.status,
            pr.is_selected,
            vt.local_track_id,
            vt.camera_id::text as camera_id,
            run.run_key
        from {schema}.plate_readings pr
        left join {schema}.vehicle_tracks vt on vt.id = pr.track_id
        left join {schema}.processing_runs run on run.id = vt.run_id
        order by run.run_key nulls last, vt.local_track_id nulls last, pr.id
    """
    rows: list[dict[str, Any]] = []
    with psycopg.connect(dsn, row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            cur.execute(query)
            rows = [dict(item) for item in cur.fetchall()]

    audit_rows: list[dict[str, Any]] = []
    invalid_rows: list[dict[str, Any]] = []
    format_counts: Counter[str] = Counter()
    affected_runs: set[str] = set()
    affected_tracks: set[str] = set()

    for row in rows:
        candidate_text = row.get("normalized_text") or row.get("raw_text")
        validation = validate_indian_plate(candidate_text)
        audit_row = {
            "plate_reading_id": row.get("plate_reading_id"),
            "run_key": row.get("run_key"),
            "track_id": row.get("track_id"),
            "local_track_id": row.get("local_track_id"),
            "camera_id": row.get("camera_id"),
            "raw_text": row.get("raw_text"),
            "normalized_text": row.get("normalized_text"),
            "status": row.get("status"),
            "is_selected": row.get("is_selected"),
            "valid": validation.valid,
            "canonical_text": validation.canonical_text,
            "format_type": validation.format_type,
            "reason": validation.reason,
            "correction_applied": validation.correction_applied,
        }
        audit_rows.append(audit_row)
        if validation.valid:
            format_counts[str(validation.format_type or "unknown")] += 1
            continue
        invalid_rows.append(audit_row)
        if row.get("run_key"):
            affected_runs.add(str(row["run_key"]))
        if row.get("local_track_id"):
            affected_tracks.add(str(row["local_track_id"]))

    total = len(audit_rows)
    valid_total = total - len(invalid_rows)
    invalid_total = len(invalid_rows)
    invalid_percentage = round((invalid_total / total) * 100.0, 2) if total else 0.0
    samples = [
        {
            "run_key": row.get("run_key"),
            "local_track_id": row.get("local_track_id"),
            "raw_text": row.get("raw_text"),
            "normalized_text": row.get("normalized_text"),
            "reason": row.get("reason"),
        }
        for row in invalid_rows[:sample_limit]
    ]
    return {
        "mode": "postgres_plate_readings_audit",
        "read_only": True,
        "schema": schema,
        "total_readings": total,
        "valid_readings": valid_total,
        "invalid_readings": invalid_total,
        "invalid_percentage": invalid_percentage,
        "valid_format_counts": dict(sorted(format_counts.items())),
        "sample_invalid_values": samples,
        "affected_run_ids": sorted(affected_runs),
        "affected_vehicle_ids_if_available": sorted(affected_tracks),
    }


def audit_run_file(run_dir: str | Path, *, sample_limit: int = 20) -> dict[str, Any]:
    run_path = Path(run_dir)
    enrichment_path = run_path / "vehicle_enrichment.json"
    if not enrichment_path.exists():
        raise RuntimeError(f"Run enrichment file not found: {enrichment_path}")
    items = json.loads(enrichment_path.read_text(encoding="utf-8"))
    invalid_rows: list[dict[str, Any]] = []
    valid_total = 0
    for item in items:
        if not isinstance(item, dict):
            continue
        raw_text = item.get("plate_raw_text") or item.get("plate_ocr_raw_response") or item.get("plate_text")
        if not raw_text:
            continue
        validation = validate_indian_plate(str(raw_text))
        if validation.valid:
            valid_total += 1
            continue
        invalid_rows.append(
            {
                "local_track_id": item.get("local_track_id"),
                "camera_id": item.get("camera_id"),
                "raw_text": raw_text,
                "normalized_text": validation.normalized_text,
                "reason": validation.reason,
            }
        )
    return {
        "mode": "run_file_plate_audit",
        "read_only": True,
        "run_dir": str(run_path),
        "invalid_examples": invalid_rows[:sample_limit],
        "invalid_count": len(invalid_rows),
        "valid_count": valid_total,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Read-only audit for Indian plate validity.")
    parser.add_argument("--mode", choices=("postgres", "run-file"), required=True)
    parser.add_argument("--env-path", default=".env")
    parser.add_argument("--run-dir")
    parser.add_argument("--sample-limit", type=int, default=20)
    parser.add_argument("--report-json")
    args = parser.parse_args(argv)

    if args.mode == "postgres":
        report = audit_postgres_plate_readings(env_path=args.env_path, sample_limit=args.sample_limit)
    else:
        if not args.run_dir:
            parser.error("--run-dir is required when --mode=run-file")
        report = audit_run_file(args.run_dir, sample_limit=args.sample_limit)

    print(json.dumps(report, indent=2, sort_keys=True))
    if args.report_json:
        report_path = Path(args.report_json)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
