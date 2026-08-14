from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.vehicle_association_debug import run_vehicle_association_debug


def main() -> int:
    parser = argparse.ArgumentParser(description="Run runtime ByteTrack association diagnostics for a known vehicle.")
    parser.add_argument("run_dir", help="Existing run directory whose persisted config/video should be replayed through the real detector/tracker path.")
    args = parser.parse_args()
    result = run_vehicle_association_debug(Path(args.run_dir))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
