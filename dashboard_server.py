from __future__ import annotations

import argparse
import threading

from src.api_app import create_app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the FastAPI dashboard server for the vehicle-tracking pipeline.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--outputs-root", default="outputs/runs")
    parser.add_argument("--config", default=None, help="Optional pipeline config to launch alongside the API.")
    parser.add_argument("--run-pipeline", action="store_true", help="Start the current pipeline in a background thread.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        import uvicorn
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise RuntimeError("uvicorn is not installed. Install fastapi and uvicorn first.") from exc

    if args.run_pipeline:
        if not args.config:
            raise RuntimeError("--config is required when --run-pipeline is used.")
        from src.pipeline import run_pipeline

        thread = threading.Thread(target=run_pipeline, args=(args.config,), daemon=True, name="pipeline-thread")
        thread.start()

    app = create_app(outputs_root=args.outputs_root)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
