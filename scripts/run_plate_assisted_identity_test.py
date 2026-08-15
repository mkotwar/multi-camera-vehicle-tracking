from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.plate_assisted_identity_experiment import run_plate_assisted_identity_experiment


def main() -> int:
    parser = argparse.ArgumentParser(description="Run experimental plate-assisted vehicle identity analysis.")
    parser.add_argument("--run-dir", required=True, help="Exact completed run directory.")
    args = parser.parse_args()
    result = run_plate_assisted_identity_experiment(args.run_dir)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
