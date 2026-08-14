from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.vehicle_identity_experiment import run_vehicle_identity_experiment


def main() -> int:
    parser = argparse.ArgumentParser(description="Run isolated persistent vehicle identity experiment on a completed run.")
    parser.add_argument("--run-dir", required=True, help="Run directory containing tracks.json and observations.csv.")
    parser.add_argument("--output-dir", default=None, help="Optional output directory. Defaults to <run-dir>/vehicle_identity_test.")
    args = parser.parse_args()
    result = run_vehicle_identity_experiment(args.run_dir, output_dir=args.output_dir)
    print(json.dumps({"output_directory": result["output_directory"], "metrics": result["metrics"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
