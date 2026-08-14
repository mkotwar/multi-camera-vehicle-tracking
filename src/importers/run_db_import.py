from __future__ import annotations

import argparse
from pathlib import Path

from src.importers.db_writer import (
    DatabaseRunWriter,
    DatabaseWriteConfig,
    DatabaseWriteConfigurationError,
    DuplicateRunError,
    build_payload,
)
from src.importers.run_file_importer import build_dry_run


DEFAULT_MIGRATION = Path("supabase/migrations/202608140001_create_vehicle_analytics_v1.sql")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Controlled one-run canonical DB import.")
    parser.add_argument("--run-dir", required=True, help="Path to outputs/runs/<run_id>.")
    parser.add_argument("--write-db", action="store_true", help="Explicitly insert the run into PostgreSQL/Supabase.")
    parser.add_argument("--apply-migration", action="store_true", help="Apply the canonical v1 migration before inserting.")
    parser.add_argument("--migration-file", default=str(DEFAULT_MIGRATION), help="SQL migration file to apply when --apply-migration is set.")
    parser.add_argument("--replace", action="store_true", help="Delete an existing matching run_key before importing.")
    parser.add_argument("--confirm-target-host", help="Required for DB writes when SUPABASE_URL is set; prevents writing to the wrong project.")
    args = parser.parse_args(argv)

    report = build_dry_run(args.run_dir)
    payload = build_payload(report)
    print("CANONICAL DB IMPORT")
    print(f"run key: {payload.run_key}")
    print(f"write intent: {'YES' if args.write_db else 'NO'}")
    print(f"apply migration: {'YES' if args.apply_migration else 'NO'}")
    print(f"payload counts: {payload.counts}")
    if not args.write_db and not args.apply_migration:
        print("NO DATABASE WRITES")
        return 0

    try:
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
        if args.apply_migration:
            writer.apply_migration(args.migration_file)
            print(f"applied migration: {args.migration_file}")
        if args.write_db:
            counts = writer.insert_reference_run(payload, replace=args.replace)
            print(f"inserted counts: {counts}")
        return 0
    except (DatabaseWriteConfigurationError, DuplicateRunError) as exc:
        print(f"ERROR: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
