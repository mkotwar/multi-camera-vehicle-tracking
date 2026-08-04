from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
import sys

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.vehicle_enrichment.shared.florence_backend import FlorenceBackend, FlorenceBackendConfig


DEFAULT_CROP_PATH = PROJECT_ROOT / "outputs" / "runs" / "20260804_112659" / "vehicle_enrichment" / "crops" / "CAM_001_TRACK_1" / "frame_000004_HIGHEST_CONFIDENCE.jpg"
DEFAULT_MODEL_PATH = Path(r"C:\Mukul K\vinfo1\video-search-engine\models\florence\Florence-2-base-ft")
DEFAULT_PROCESSOR_PATH = DEFAULT_MODEL_PATH
DEFAULT_ADAPTER_PATH = Path(r"C:\Mukul K\OCR_MUKUL\adaptor_florance_baseFT")

CAPTION_PROMPT = "<CAPTION>"
VQA_PROMPT = "<VQA>"
VQA_TEXT = "What is the primary color of the vehicle?"
BODY_TYPE_TEXT = "Which is closest: hatchback, sedan, suv, mpv, van, pickup, or other?"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the old working Florence reference path on one saved crop.")
    parser.add_argument("--crop", default=str(DEFAULT_CROP_PATH))
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--processor-path", default=str(DEFAULT_PROCESSOR_PATH))
    parser.add_argument("--adapter-path", default=str(DEFAULT_ADAPTER_PATH))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--caption-prompt", default=CAPTION_PROMPT)
    parser.add_argument("--vqa-prompt", default=VQA_PROMPT)
    parser.add_argument("--vqa-text", default=VQA_TEXT)
    parser.add_argument("--body-type-prompt", default=VQA_PROMPT)
    parser.add_argument("--body-type-text", default=BODY_TYPE_TEXT)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--num-beams", type=int, default=3)
    parser.add_argument("--local-files-only", action="store_true", default=True)
    parser.add_argument("--adapter-on", action="store_true")
    return parser.parse_args()


def build_backend_config(args: argparse.Namespace, *, adapter_enabled: bool) -> FlorenceBackendConfig:
    return FlorenceBackendConfig(
        enabled=True,
        backend="florence2",
        base_model_id=str(args.model_path),
        processor_path=str(args.processor_path),
        adapter_path=str(args.adapter_path),
        adapter_enabled=adapter_enabled,
        device=str(args.device),
        dtype=str(args.dtype),
        trust_remote_code=True,
        attention_implementation="eager",
        max_new_tokens=int(args.max_new_tokens),
        num_beams=int(args.num_beams),
        use_cache=False,
        local_files_only=bool(args.local_files_only),
        lazy_load=False,
    )


def run_task(backend: FlorenceBackend, image, task_prompt: str, text_input: str | None) -> dict[str, object]:
    result = backend.run_task(image, task_prompt, text_input)
    payload = dict(result.get("payload") or {})
    return {
        "status": result.get("status"),
        "reason": result.get("reason"),
        "task_prompt": task_prompt,
        "text_input": text_input,
        "generated_ids": payload.get("generated_ids"),
        "decoded_raw_output": payload.get("generated_text"),
        "post_processed_output": payload.get("parsed_answer"),
        "inference_duration_ms": payload.get("inference_duration_ms"),
        "adapter_active": payload.get("adapter_active"),
    }


def main() -> int:
    args = parse_args()
    crop_path = Path(args.crop).expanduser().resolve()
    if not crop_path.exists():
        raise FileNotFoundError(f"Crop image does not exist: {crop_path}")
    image = cv2.imread(str(crop_path))
    if image is None or image.size == 0:
        raise RuntimeError(f"Failed to decode crop image: {crop_path}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = PROJECT_ROOT / "outputs" / "florence_reference_diagnostics" / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    report: dict[str, object] = {
        "crop_path": str(crop_path),
        "adapter_requested": bool(args.adapter_on),
        "runs": [],
    }

    for mode_name, adapter_enabled in (("adapter_off", False), ("adapter_on", True)):
        if mode_name == "adapter_on" and not args.adapter_on:
            continue
        backend = FlorenceBackend(build_backend_config(args, adapter_enabled=adapter_enabled))
        try:
            backend.load()
            report[f"{mode_name}_runtime"] = {
                "model_class": backend.metrics.get("florence_model_class"),
                "processor_class": backend.metrics.get("florence_processor_class"),
                "model_source": backend.model_identifier,
                "processor_source": backend.processor_identifier,
                "device": backend.resolved_device,
                "dtype": backend.resolved_dtype,
                "loading_info": backend.loading_info,
                "critical_missing_keys": backend.metrics.get("florence_critical_missing_keys"),
            }
            report["runs"].append({
                "mode": mode_name,
                "caption": run_task(backend, image, args.caption_prompt, None),
                "vqa": run_task(backend, image, args.vqa_prompt, args.vqa_text),
                "body_type": run_task(backend, image, args.body_type_prompt, args.body_type_text),
            })
        finally:
            backend.close()

    report["transformers_version"] = _safe_import_version("transformers")
    report["torch_version"] = _safe_import_version("torch")
    report["report_path"] = str(output_dir / "report.json")
    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


def _safe_import_version(module_name: str) -> str:
    module = __import__(module_name)
    return str(getattr(module, "__version__", "unknown"))


if __name__ == "__main__":
    raise SystemExit(main())
