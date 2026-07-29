from __future__ import annotations

import argparse
import json
import sys

from src.bbox_tuning import run_bbox_tuning


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the per-class bbox tuning experiment.")
    parser.add_argument("--config", default="config.yaml", help="Path to the YAML configuration file.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run_bbox_tuning(args.config)
    print(json.dumps(
        {
            "run_id": result["run_id"],
            "run_directory": result["run_directory"],
            "bbox_tuning_directory": result["bbox_tuning_directory"],
            "selected_profiles": result["selected_profiles"],
            "selected_edge_mode": result["selected_edge_mode"],
        },
        indent=2,
    ))
    return 0


if __name__ == "__main__":
    sys.exit(main())
