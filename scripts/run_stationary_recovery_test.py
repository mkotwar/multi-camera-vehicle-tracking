from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.stationary_recovery_experiment import run_stationary_recovery_experiment


def main() -> int:
    parser = argparse.ArgumentParser(description="Run isolated stationary long-gap vehicle identity recovery on conservative identity outputs.")
    parser.add_argument("--run-dir", required=True, help="Run directory containing vehicle_identity_test outputs.")
    parser.add_argument("--identity-dir", default=None, help="Optional vehicle_identity_test directory. Defaults to <run-dir>/vehicle_identity_test.")
    args = parser.parse_args()
    result = run_stationary_recovery_experiment(args.run_dir, identity_dir=args.identity_dir)
    print(json.dumps({"output_directory": result["output_directory"], "metrics": result["metrics"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
