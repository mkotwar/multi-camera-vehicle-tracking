from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.track_reconciliation import run_track_reconciliation_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run isolated ByteTrack track reconciliation experiment on an existing run directory.")
    parser.add_argument("--run-dir", required=True, help="Existing output run directory containing tracks.json and observations.csv.")
    parser.add_argument("--config", default=None, help="Optional YAML file or run_config.yaml containing track_reconciliation settings.")
    parser.add_argument("--output-dir", default=None, help="Optional output directory. Defaults to <run-dir>/track_reconciliation_test.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run_track_reconciliation_experiment(args.run_dir, config_path=args.config, output_dir=args.output_dir)
    metrics = result["metrics"]
    print(f"Raw ByteTrack unique tracks: {metrics['raw_bytetrack_unique_tracks']}")
    print(f"Reconciled vehicle identities: {metrics['reconciled_vehicle_identities']}")
    print(f"Potential duplicate tracks removed: {metrics['potential_duplicate_tracks_removed']}")
    print(f"Reconciliation attempts: {metrics['reconciliation_attempts']}")
    print(f"Accepted matches: {metrics['accepted_matches']}")
    print(f"Rejected candidate matches: {metrics['rejected_candidate_matches']}")
    print(f"Ambiguous matches: {metrics['ambiguous_matches']}")
    print(json.dumps(result["accepted_associations"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
