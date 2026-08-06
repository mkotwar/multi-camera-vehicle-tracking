from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys

import cv2

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.vehicle_enrichment.body_type.classifier import BODY_TYPE_PROMPT_TEXT, VehicleBodyTypeClassifier
from src.vehicle_enrichment.shared import FlorenceBackend, FlorenceBackendConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a single Florence-2 body-type inference diagnostic.")
    parser.add_argument("--image", required=True, help="Path to the cropped vehicle image.")
    parser.add_argument("--model", required=True, help="Florence-2 model path or identifier.")
    parser.add_argument("--processor-path", default="", help="Optional Florence processor path.")
    parser.add_argument("--device", default="auto", help="Runtime device: auto, cpu, cuda, or cuda:<index>.")
    parser.add_argument("--dtype", default="auto", help="Runtime dtype: auto, float16, or float32.")
    parser.add_argument("--max-new-tokens", type=int, default=128, help="Maximum generated tokens.")
    parser.add_argument("--num-beams", type=int, default=3, help="Beam count.")
    parser.add_argument("--use-cache", action="store_true", help="Enable model generation cache.")
    parser.add_argument("--local-files-only", action="store_true", help="Restrict loading to local files only.")
    parser.add_argument("--trust-remote-code", action="store_true", default=True, help="Trust remote code.")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging.")
    return parser.parse_args()


def build_backend(args: argparse.Namespace) -> FlorenceBackend:
    config = FlorenceBackendConfig(
        enabled=True,
        backend="florence2",
        base_model_id=str(args.model),
        processor_path=str(args.processor_path),
        adapter_path="",
        adapter_enabled=False,
        device=str(args.device),
        dtype=str(args.dtype),
        trust_remote_code=bool(args.trust_remote_code),
        attention_implementation="eager",
        max_new_tokens=int(args.max_new_tokens),
        num_beams=int(args.num_beams),
        use_cache=bool(args.use_cache),
        local_files_only=bool(args.local_files_only),
        lazy_load=False,
    )
    return FlorenceBackend(config, logger=logging.getLogger("florence-body-type-diagnostic"))


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    image_path = Path(args.image).expanduser().resolve()
    if not image_path.exists():
        raise FileNotFoundError(f"Image does not exist: {image_path}")
    image = cv2.imread(str(image_path))
    if image is None or image.size == 0:
        raise RuntimeError(f"Unable to decode image: {image_path}")

    backend = build_backend(args)
    backend.load()
    result = backend.run_task(image, "<VQA>", BODY_TYPE_PROMPT_TEXT)
    print("image:", str(image_path))
    print("status:", result["status"])
    print("reason:", result.get("reason"))
    payload = dict(result.get("payload") or {})
    print("final prompt:", payload.get("final_processor_text"))
    print("input tensor shapes and dtypes:", json.dumps({
        "input_ids_shape": payload.get("input_ids_shape"),
        "pixel_values_shape": payload.get("pixel_values_shape"),
        "device": payload.get("device"),
        "dtype": payload.get("dtype"),
    }))
    print("generated token IDs:", payload.get("generated_ids"))
    print("decoded full output:", payload.get("decoded_full_text"))
    print("decoded generated-only output:", payload.get("decoded_generated_only_text"))
    print("post-processed output:", json.dumps(payload.get("parsed_answer"), ensure_ascii=True))
    classifier = VehicleBodyTypeClassifier(
        {
            "enabled": True,
            "backend": "florence2",
            "run_only_when_vehicle_class": ["CAR"],
            "maximum_crops_per_track": 1,
            "minimum_crop_width": 1,
            "minimum_crop_height": 1,
            "allowed_labels": ["SUV", "SEDAN", "HATCHBACK", "MPV", "VAN", "PICKUP", "OTHER", "UNKNOWN"],
        },
        backend=backend,
        logger=logging.getLogger("florence-body-type-diagnostic"),
    )
    extracted = classifier._extract_body_type_text(payload)  # noqa: SLF001
    normalized_label, normalization_reason = classifier.normalize_label(extracted)
    print("normalized label:", normalized_label)
    print("normalization reason:", normalization_reason)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
