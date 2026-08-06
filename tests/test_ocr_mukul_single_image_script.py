from __future__ import annotations

import importlib.util
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import pytest


_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "test_ocr_mukul_single_image.py"
_SPEC = importlib.util.spec_from_file_location("test_ocr_mukul_single_image_script_module", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
run_single_image_test = _MODULE.run_single_image_test
run_separate_vehicle_attribute_test = _MODULE.run_separate_vehicle_attribute_test


class _FakeBackend:
    def __init__(self, *, adapter_active: bool = True) -> None:
        self.adapter_active = adapter_active
        self.resolved_device = "cuda:0"
        self.resolved_dtype = "float16"
        self.model_identifier = "fake-model"
        self.processor_identifier = "fake-processor"
        self.load_calls = 0
        self.run_calls = 0
        self.close_calls = 0

    def load(self) -> None:
        self.load_calls += 1

    @property
    def metrics(self) -> dict[str, float]:
        return {
            "gpu_memory_allocated_mb": 256.0,
            "gpu_memory_reserved_mb": 512.0,
        }

    def run_task(self, image, task_prompt, text_input=None, *, adapter_active=None, generation_overrides=None):
        self.run_calls += 1
        assert task_prompt == "<VQA>"
        assert text_input == "What is the exterior colour and body type?"
        return {
            "status": "completed",
            "reason": None,
            "payload": {
                "generated_text": "A black sedan.",
                "parsed_answer": "A black sedan.",
                "inference_duration_ms": 10.5,
                "pixel_values_shape": [1, 3, 224, 224],
                "model_identifier": self.model_identifier,
                "processor_identifier": self.processor_identifier,
            },
        }

    def close(self) -> None:
        self.close_calls += 1


class _BackendFactory:
    def __init__(self) -> None:
        self.created: list[_FakeBackend] = []

    def __call__(self, *args, **kwargs):
        backend = _FakeBackend(adapter_active=bool(kwargs.get("adapter_enabled_override")))
        self.created.append(backend)
        return backend


class _SeparateBackend:
    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.adapter_active = False
        self.resolved_device = "cuda:0"
        self.resolved_dtype = "float16"
        self.model_identifier = "fake-model"
        self.processor_identifier = "fake-processor"
        self.load_calls = 0
        self.run_calls = 0
        self.close_calls = 0
        self.images_seen: list[tuple[int, int, int]] = []
        self.generation_overrides_seen: list[dict[str, object]] = []
        self.prompts_seen: list[tuple[str, str]] = []

    def load(self) -> None:
        self.load_calls += 1

    @property
    def metrics(self) -> dict[str, object]:
        return {
            "gpu_memory_allocated_mb": 256.0,
            "gpu_memory_reserved_mb": 512.0,
            "florence_model_class": "FakeModel",
            "florence_processor_class": "FakeProcessor",
            "florence_load_duration_ms": 12.5,
            "florence_load_successes": self.load_calls,
        }

    def run_task(self, image, task_prompt, text_input=None, *, adapter_active=None, generation_overrides=None):
        self.run_calls += 1
        self.images_seen.append(tuple(image.shape))
        self.generation_overrides_seen.append(dict(generation_overrides or {}))
        self.prompts_seen.append((task_prompt, text_input or ""))
        raw_response = self.responses[self.run_calls - 1]
        return {
            "status": "completed",
            "reason": None,
            "payload": {
                "generated_text": raw_response,
                "parsed_answer": {task_prompt: raw_response},
                "inference_duration_ms": 5.0 + self.run_calls,
                "pixel_values_shape": [1, 3, 224, 224],
                "model_identifier": self.model_identifier,
                "processor_identifier": self.processor_identifier,
                "final_processor_text": f"{task_prompt}{text_input or ''}",
            },
        }

    def close(self) -> None:
        self.close_calls += 1


def _config_text() -> str:
    return "\n".join(
        [
            "vehicle_enrichment:",
            "  shared_florence:",
            "    enabled: true",
            "    backend: florence2",
            "    base_model_id: D:/project/models/Florence-2-base-ft",
            "    processor_path: D:/project/models/Florence-2-base-ft",
            "    adapter_path: D:/project/models/OCR_MUKUL/OCR_MUKUL/adaptor_florance_baseFT",
            "    adapter_enabled: true",
            "    device: auto",
            "    dtype: auto",
            "    trust_remote_code: true",
            "    attention_implementation: eager",
            "    max_new_tokens: 64",
            "    num_beams: 1",
            "    use_cache: false",
            "    local_files_only: true",
            "    lazy_load: true",
            "  ocr_mukul:",
            "    task_token: \"<VQA>\"",
            "    prompt: \"What is the exterior colour and body type?\"",
            "  vehicle_attributes:",
            "    enabled: true",
            "    task_token: \"<VQA>\"",
            "    prompt: \"What is the exterior colour and body type?\"",
            "    florence:",
            "      adapter_enabled: false",
            "  diagnostic_separate_attributes:",
            "    enabled: false",
            "    colour:",
            "      task_token: \"<VQA>\"",
            "      prompt: \"What colour is the vehicle?\"",
            "      generation:",
            "        max_new_tokens: 16",
            "        num_beams: 1",
            "        do_sample: false",
            "        use_cache: true",
            "        early_stopping: false",
            "    body_vqa:",
            "      task_token: \"<VQA>\"",
            "      prompt: \"What body type is the vehicle?\"",
            "      generation:",
            "        max_new_tokens: 16",
            "        num_beams: 1",
            "        do_sample: false",
            "        use_cache: true",
            "        early_stopping: false",
            "    body_detailed_caption:",
            "      task_token: \"<MORE_DETAILED_CAPTION>\"",
            "      prompt: \"\"",
            "      generation:",
            "        max_new_tokens: 64",
            "        num_beams: 3",
            "        do_sample: false",
            "        use_cache: true",
            "        early_stopping: true",
            "  plate:",
            "    ocr:",
            "      enabled: true",
            "      task_token: \"<OCR>\"",
            "      prompt: \"\"",
            "      florence:",
            "        adapter_enabled: true",
        ]
    )


def test_run_single_image_test_with_mocked_backend(tmp_path: Path) -> None:
    image_path = tmp_path / "vehicle.jpg"
    image = np.full((120, 180, 3), 127, dtype=np.uint8)
    assert cv2.imwrite(str(image_path), image)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(_config_text(), encoding="utf-8")

    backend = _FakeBackend()
    result, output_dir = run_single_image_test(
        image_path=image_path,
        config_path=config_path,
        task_token=None,
        prompt=None,
        output_root=tmp_path / "outputs",
        repeat=3,
        command_text="python scripts/test_ocr_mukul_single_image.py --repeat 3",
        model_mode="adapter",
        backend=backend,
    )

    assert backend.load_calls == 1
    assert backend.run_calls == 3
    assert result["parsed_body_type"] == "SEDAN"
    assert result["parsed_colour"] == "BLACK"
    assert result["effective_processor_text"] == "<VQA>What is the exterior colour and body type?"
    assert result["adapter_loaded"] is True
    assert len(result["repetitions"]) == 3
    assert (output_dir / "adapter" / "result.json").exists()
    assert (output_dir / "adapter" / "result.txt").exists()
    assert (output_dir / "input_image.jpg").exists()
    assert (output_dir / "preprocessed_image.jpg").exists()

    persisted = json.loads((output_dir / "adapter" / "result.json").read_text(encoding="utf-8"))
    assert persisted["raw_generated_text"] == "A black sedan."
    assert persisted["device"] == "cuda:0"


def test_base_mode_disables_adapter_even_if_yaml_enables_it(tmp_path: Path, monkeypatch) -> None:
    image_path = tmp_path / "vehicle.jpg"
    assert cv2.imwrite(str(image_path), np.full((120, 180, 3), 127, dtype=np.uint8))
    config_path = tmp_path / "config.yaml"
    config_path.write_text(_config_text(), encoding="utf-8")
    factory = _BackendFactory()
    monkeypatch.setattr(_MODULE, "FlorenceBackend", factory)

    result, _ = run_single_image_test(
        image_path=image_path,
        config_path=config_path,
        task_token=None,
        prompt=None,
        output_root=tmp_path / "outputs",
        repeat=2,
        command_text="python script.py --model-mode base",
        model_mode="base",
    )

    assert result["adapter_requested"] is False
    assert result["adapter_loaded"] is False
    assert result["effective_model_type"] == "base_model"
    assert factory.created[0].load_calls == 1
    assert factory.created[0].run_calls == 2
    assert factory.created[0].close_calls == 1


def test_adapter_mode_requires_adapter(tmp_path: Path) -> None:
    image_path = tmp_path / "vehicle.jpg"
    assert cv2.imwrite(str(image_path), np.full((120, 180, 3), 127, dtype=np.uint8))
    config_path = tmp_path / "config.yaml"
    config_path.write_text(_config_text(), encoding="utf-8")

    try:
        run_single_image_test(
            image_path=image_path,
            config_path=config_path,
            task_token=None,
            prompt=None,
            output_root=tmp_path / "outputs",
            repeat=1,
            command_text="python script.py --model-mode adapter",
            model_mode="adapter",
            backend=_FakeBackend(adapter_active=False),
        )
    except RuntimeError as exc:
        assert "requires the Florence adapter" in str(exc)
    else:
        raise AssertionError("Expected adapter mode to fail when adapter is inactive.")


def test_compare_mode_runs_base_then_adapter_and_creates_artifacts(tmp_path: Path, monkeypatch) -> None:
    image_path = tmp_path / "vehicle.jpg"
    assert cv2.imwrite(str(image_path), np.full((120, 180, 3), 127, dtype=np.uint8))
    config_path = tmp_path / "config.yaml"
    config_path.write_text(_config_text(), encoding="utf-8")
    factory = _BackendFactory()
    monkeypatch.setattr(_MODULE, "FlorenceBackend", factory)

    result, output_dir = run_single_image_test(
        image_path=image_path,
        config_path=config_path,
        task_token=None,
        prompt=None,
        output_root=tmp_path / "outputs",
        repeat=1,
        command_text="python script.py --model-mode compare",
        model_mode="compare",
    )

    assert len(factory.created) == 2
    assert factory.created[0].adapter_active is False
    assert factory.created[1].adapter_active is True
    assert factory.created[0].close_calls == 1
    assert factory.created[1].close_calls == 1
    assert result["comparison"]["same_preprocessing"] is True
    assert result["comparison"]["same_prompt"] is True
    assert (output_dir / "base" / "result.json").exists()
    assert (output_dir / "adapter" / "result.json").exists()
    assert (output_dir / "comparison.json").exists()
    assert (output_dir / "comparison.txt").exists()


def test_missing_image_fails_clearly(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(_config_text(), encoding="utf-8")

    try:
        run_single_image_test(
            image_path=tmp_path / "missing.jpg",
            config_path=config_path,
            task_token=None,
            prompt=None,
            output_root=tmp_path / "outputs",
            repeat=1,
            command_text="python script.py",
            model_mode="base",
        )
    except FileNotFoundError as exc:
        assert "Image does not exist" in str(exc)
    else:
        raise AssertionError("Expected missing image failure.")


def test_separate_vehicle_attributes_requires_base_mode(tmp_path: Path) -> None:
    image_path = tmp_path / "vehicle.jpg"
    assert cv2.imwrite(str(image_path), np.full((120, 180, 3), 127, dtype=np.uint8))
    config_path = tmp_path / "config.yaml"
    config_path.write_text(_config_text(), encoding="utf-8")

    parser = _MODULE.build_arg_parser()
    args = parser.parse_args(
        [
            "--image",
            str(image_path),
            "--config",
            str(config_path),
            "--task",
            "separate-vehicle-attributes",
            "--model-mode",
            "adapter",
        ]
    )
    with pytest.raises(ValueError, match="requires --model-mode base"):
        if args.task == "separate-vehicle-attributes" and args.model_mode != "base":
            raise ValueError("--task separate-vehicle-attributes requires --model-mode base.")


def test_separate_vehicle_attributes_runs_independent_prompts_and_writes_artifacts(tmp_path: Path) -> None:
    image_path = tmp_path / "vehicle.jpg"
    image = np.full((120, 180, 3), 127, dtype=np.uint8)
    assert cv2.imwrite(str(image_path), image)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(_config_text(), encoding="utf-8")
    backend = _SeparateBackend(
        [
            "black",
            "black",
            "black",
            "suv",
            "suv",
            "vehicle",
            "A black sport utility vehicle.",
            "A black sport utility vehicle.",
            "DL4BC2038",
        ]
    )

    result, output_dir = run_separate_vehicle_attribute_test(
        image_path=image_path,
        config_path=config_path,
        output_root=tmp_path / "outputs",
        repeat=3,
        command_text="python scripts/test_ocr_mukul_single_image.py --task separate-vehicle-attributes --repeat 3",
        body_strategy="both",
        backend=backend,
    )

    assert backend.load_calls == 1
    assert backend.run_calls == 9
    assert backend.close_calls == 1
    assert len({shape for shape in backend.images_seen}) == 1
    assert backend.prompts_seen[:3] == [("<VQA>", "What colour is the vehicle?")] * 3
    assert backend.prompts_seen[3:6] == [("<VQA>", "What body type is the vehicle?")] * 3
    assert backend.prompts_seen[6:] == [("<MORE_DETAILED_CAPTION>", "")] * 3
    assert all(item == {"max_new_tokens": 16, "num_beams": 1, "do_sample": False, "use_cache": True, "early_stopping": False} for item in backend.generation_overrides_seen[:6])
    assert all(item == {"max_new_tokens": 64, "num_beams": 3, "do_sample": False, "use_cache": True, "early_stopping": True} for item in backend.generation_overrides_seen[6:])
    assert result["model"]["adapter_loaded"] is False
    assert result["model"]["load_count"] == 1
    assert result["colour"]["consensus_colour"] == "BLACK"
    assert result["colour"]["stable"] is True
    assert result["body_vqa"]["consensus_body_type"] == "SUV"
    assert result["body_vqa"]["stable"] is True
    assert result["body_detailed_caption"]["consensus_body_type"] == "SUV"
    assert result["body_detailed_caption"]["stable"] is True
    assert result["body_detailed_caption"]["repetitions"][2]["response_reason"] == "plate_like_response"
    assert (output_dir / "input_image.jpg").exists()
    assert (output_dir / "preprocessed_image.jpg").exists()
    assert (output_dir / "result.json").exists()
    assert (output_dir / "result.txt").exists()
    assert (output_dir / "results.csv").exists()

    rows = list(csv.DictReader((output_dir / "results.csv").open(encoding="utf-8")))
    assert len(rows) == 9
    assert {row["model_mode"] for row in rows} == {"base"}
    assert {row["adapter_loaded"] for row in rows} == {"False"}


def test_separate_vehicle_attributes_body_strategy_vqa_only(tmp_path: Path) -> None:
    image_path = tmp_path / "vehicle.jpg"
    image = np.full((120, 180, 3), 127, dtype=np.uint8)
    assert cv2.imwrite(str(image_path), image)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(_config_text(), encoding="utf-8")
    backend = _SeparateBackend(["grey", "sedan"])

    result, _ = run_separate_vehicle_attribute_test(
        image_path=image_path,
        config_path=config_path,
        output_root=tmp_path / "outputs",
        repeat=1,
        command_text="python",
        body_strategy="vqa",
        backend=backend,
    )

    assert result["body_vqa"]["enabled"] is True
    assert result["body_detailed_caption"]["enabled"] is False
    assert result["body_detailed_caption"]["repetitions"] == []


def test_repeat_consensus_conflicts_return_unknown() -> None:
    consensus = _MODULE.calculate_repeat_consensus(["BLACK", "WHITE", "GREY"])
    assert consensus["consensus_label"] == "UNKNOWN"
    assert consensus["stable"] is False
    assert consensus["reason"] == "conflicting_repeated_predictions"
