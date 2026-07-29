from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.pipeline import run_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Step 1 multicamera vehicle tracking foundation pipeline.")
    parser.add_argument("--config", default="config.yaml", help="Path to the YAML configuration file.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    exit_code, run_id, run_directory = run_pipeline(args.config)
    if exit_code == 0:
        print(f"Run completed: {run_id}")
        print(f"Output: {run_directory}")
        return 0
    log_path = Path(run_directory) / "pipeline.log"
    print(f"Run failed. Check: {log_path}")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
