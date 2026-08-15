from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from src.importers.db_writer import (
    DatabaseImportError,
    DatabaseRunWriter,
    DatabaseWriteConfig,
    DatabaseWriteConfigurationError,
    build_payload,
)
from src.importers.run_file_importer import build_dry_run


DEFAULT_MIGRATION = Path("database/migrations/202608150002_align_vehicle_analytics_schema.sql")


def import_completed_run(
    run_dir: str | Path,
    *,
    apply_migration: bool = False,
    migration_file: str | Path = DEFAULT_MIGRATION,
    replace: bool = False,
    db_schema: str | None = None,
    observation_batch_size: int | None = None,
    confirm_target_host: str | None = None,
) -> tuple[Any, dict[str, Any]]:
    report = build_dry_run(run_dir)
    payload = build_payload(report)
    if report.counts["issues"]["ERROR"]:
        details = "\n".join(_format_validation_issue(index, issue) for index, issue in enumerate(report.issues, start=1) if issue.severity == "ERROR")
        raise DatabaseImportError(f"DRY RUN FAILED\n{details}\nNo database writes performed.")

    config = DatabaseWriteConfig.from_env()
    if db_schema or observation_batch_size:
        config = DatabaseWriteConfig(
            dsn=config.dsn,
            schema=db_schema or config.schema,
            observation_batch_size=observation_batch_size or config.observation_batch_size,
            supabase_url=config.supabase_url,
        )
    if config.supabase_host:
        if not confirm_target_host:
            raise DatabaseWriteConfigurationError("--confirm-target-host is required when SUPABASE_URL is configured.")
        if confirm_target_host != config.supabase_host:
            raise DatabaseWriteConfigurationError(
                f"Refusing to write: confirmed host {confirm_target_host!r} does not match configured host {config.supabase_host!r}."
            )

    writer = DatabaseRunWriter(config)
    if apply_migration:
        writer.apply_migration(migration_file)
    return writer.insert_reference_run(payload, replace=replace), payload.counts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Controlled one-run canonical DB import.")
    parser.add_argument("--run-dir", required=True, help="Path to outputs/runs/<run_id>.")
    parser.add_argument("--write-db", action="store_true", help="Explicitly insert the run into PostgreSQL/Supabase.")
    parser.add_argument("--apply-migration", action="store_true", help="Apply the canonical v1 migration before inserting.")
    parser.add_argument("--migration-file", default=str(DEFAULT_MIGRATION), help="SQL migration file to apply when --apply-migration is set.")
    parser.add_argument("--replace", action="store_true", help="Delete an existing matching run_key before importing.")
    parser.add_argument("--db-schema", help="PostgreSQL schema to write to. Defaults to DB_SCHEMA or vehicle_analytics.")
    parser.add_argument("--observation-batch-size", type=int, help="Batch size for track_observations upserts.")
    parser.add_argument("--confirm-target-host", help="Required for DB writes when SUPABASE_URL is set; prevents writing to the wrong project.")
    args = parser.parse_args(argv)

    report = build_dry_run(args.run_dir)
    payload = build_payload(report)
    print("CANONICAL DB IMPORT")
    print(f"run key: {payload.run_key}")
    print(f"write intent: {'YES' if args.write_db else 'NO'}")
    print(f"apply migration: {'YES' if args.apply_migration else 'NO'}")
    print(f"payload counts: {payload.counts}")
    if report.counts["issues"]["ERROR"]:
        print("DRY RUN FAILED")
        for index, issue in enumerate((item for item in report.issues if item.severity == "ERROR"), start=1):
            print(_format_validation_issue(index, issue))
        print("No database writes performed.")
        return 2
    if not args.write_db and not args.apply_migration:
        print("NO DATABASE WRITES")
        return 0

    try:
        if args.write_db:
            result, _counts = import_completed_run(
                args.run_dir,
                apply_migration=args.apply_migration,
                migration_file=args.migration_file,
                replace=args.replace,
                db_schema=args.db_schema,
                observation_batch_size=args.observation_batch_size,
                confirm_target_host=args.confirm_target_host,
            )
            if args.apply_migration:
                print(f"applied migration: {args.migration_file}")
            print(f"import result: {result.to_dict()}")
        elif args.apply_migration:
            config = DatabaseWriteConfig.from_env()
            if config.supabase_host:
                print(f"target Supabase host: {config.supabase_host}")
                if not args.confirm_target_host:
                    raise DatabaseWriteConfigurationError("--confirm-target-host is required when SUPABASE_URL is configured.")
                if args.confirm_target_host != config.supabase_host:
                    raise DatabaseWriteConfigurationError(
                        f"Refusing to write: confirmed host {args.confirm_target_host!r} does not match configured host {config.supabase_host!r}."
                    )
            writer = DatabaseRunWriter(config)
            writer.apply_migration(args.migration_file)
            print(f"applied migration: {args.migration_file}")
        return 0
    except (DatabaseWriteConfigurationError, DatabaseImportError) as exc:
        print(f"ERROR: {exc}")
        return 2
    except Exception as exc:
        print(f"ERROR: {exc}")
        return 2


def _format_validation_issue(index: int, issue: Any) -> str:
    context = dict(getattr(issue, "context", {}) or {})
    key = context.get("track_key") or context.get("import_track_key") or context.get("vehicle_key") or context.get("path") or context.get("row") or "unknown"
    parts = [
        f"[{index}] {getattr(issue, 'code', 'unknown_validation_error')}: {getattr(issue, 'message', '')}",
        f"rule={context.get('rule', getattr(issue, 'code', 'unknown'))}",
        f"table/entity={context.get('table', 'unknown')}/{context.get('entity', 'unknown')}",
        f"id/key={key}",
        f"expected={context.get('expected', 'not specified')}",
        f"actual={context.get('actual', 'not specified')}",
    ]
    extras = {k: v for k, v in context.items() if k not in {"rule", "table", "entity", "track_key", "import_track_key", "vehicle_key", "path", "row", "expected", "actual"}}
    if extras:
        parts.append(f"context={extras}")
    return " | ".join(parts)


if __name__ == "__main__":
    raise SystemExit(main())
