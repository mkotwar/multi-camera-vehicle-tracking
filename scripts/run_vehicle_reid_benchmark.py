from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.vehicle_reid_benchmark import run_vehicle_reid_benchmark


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run isolated dedicated Vehicle ReID benchmark.")
    parser.add_argument("--run-dir", required=True, help="Existing pipeline run directory.")
    parser.add_argument("--output-dir", default=None, help="Defaults to <run-dir>/vehicle_reid_benchmark.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run_vehicle_reid_benchmark(args.run_dir, output_dir=args.output_dir)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
