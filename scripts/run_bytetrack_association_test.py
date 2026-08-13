from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.bytetrack_association_experiment import run_association_threshold_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run controlled ByteTrack matching-threshold replay experiment.")
    parser.add_argument("--run-dir", default="outputs/runs/20260813_182311")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_association_threshold_experiment(Path(args.run_dir))
    print(json.dumps(summary["decision"], indent=2))
    print(Path(args.run_dir) / "bytetrack_association_test")


if __name__ == "__main__":
    main()
