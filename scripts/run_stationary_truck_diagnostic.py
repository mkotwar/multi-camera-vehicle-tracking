from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.stationary_truck_diagnostic import run_stationary_truck_diagnostic


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run isolated stationary truck fragmentation diagnostic on an existing run.")
    parser.add_argument("--run-dir", default="outputs/runs/20260813_182311", help="Existing output run directory.")
    parser.add_argument("--output-dir", default=None, help="Optional diagnostic output directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_stationary_truck_diagnostic(Path(args.run_dir), output_dir=args.output_dir)
    print(json.dumps(summary["primary_conclusion"], indent=2))
    print(f"Wrote diagnostics to {Path(args.output_dir) if args.output_dir else Path(args.run_dir) / 'stationary_truck_debug'}")


if __name__ == "__main__":
    main()
