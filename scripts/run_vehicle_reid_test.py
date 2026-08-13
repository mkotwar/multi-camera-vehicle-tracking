from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.vehicle_reid_experiment import run_vehicle_reid_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run isolated vehicle ReID / appearance-similarity diagnostic.")
    parser.add_argument("--run-dir", required=True, help="Existing pipeline run directory.")
    parser.add_argument("--output-dir", default=None, help="Defaults to <run-dir>/vehicle_reid_test.")
    parser.add_argument(
        "--embedder",
        default="florence",
        choices=["florence", "color_texture"],
        help="Embedding backend. Florence is the generic appearance baseline using the local Florence vision encoder.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    embedder = None
    if args.embedder == "color_texture":
        from src.vehicle_reid_experiment import ColorTextureAppearanceEmbedder

        embedder = ColorTextureAppearanceEmbedder()
    result = run_vehicle_reid_experiment(args.run_dir, output_dir=args.output_dir, embedder=embedder)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
