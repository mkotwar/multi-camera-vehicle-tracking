from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

import src.vehicle_enrichment.shared.florence_backend as florence_backend_module
from src.runtime_device import RuntimeDevice
from src.vehicle_enrichment.shared.florence_backend import FlorenceBackend, FlorenceBackendConfig


class FakeTensor:
    def __init__(self, value):
        self.value = value
        self.device = None
        self.dtype = None

    def to(self, device=None, dtype=None):
        self.device = device
        self.dtype = dtype
        return self


class FakeProcessor:
    def __init__(self):
        self.calls = []

    def __call__(self, text, images, return_tensors):
        self.calls.append({"text": text, "size": images.size, "return_tensors": return_tensors})
        return {
            "input_ids": torch.tensor([[1, 2, 3]], dtype=torch.int64),
            "pixel_values": torch.zeros((1, 3, 4, 4), dtype=torch.float32),
            "attention_mask": torch.tensor([[1, 1, 1]], dtype=torch.int64),
        }

    def batch_decode(self, generated_ids, skip_special_tokens=False):
        return ["SUV"]

    def post_process_generation(self, generated_text, task, image_size):
        return {task: generated_text}


class FakeModel:
    def __init__(self):
        self.to_device = None
        self.generate_calls = []
        self.eval_called = False

    def to(self, device):
        self.to_device = device
        return self

    def eval(self):
        self.eval_called = True

    def generate(self, **kwargs):
        self.generate_calls.append(kwargs)
        return torch.tensor([[1, 2, 3]], dtype=torch.int64)


def _config(**overrides) -> FlorenceBackendConfig:
    payload = {
        "enabled": True,
        "backend": "florence2",
        "base_model_id": "microsoft/Florence-2-base-ft",
        "processor_path": "",
        "adapter_path": "",
        "adapter_enabled": False,
        "device": "cpu",
        "dtype": "auto",
        "trust_remote_code": True,
        "attention_implementation": "eager",
        "max_new_tokens": 128,
        "num_beams": 3,
        "use_cache": False,
        "local_files_only": False,
        "lazy_load": True,
    }
    payload.update(overrides)
    return FlorenceBackendConfig(**payload)


def test_lazy_load_and_repeated_run_task_reuse_model() -> None:
    model = FakeModel()
    processor = FakeProcessor()
    load_calls = {"model": 0, "processor": 0}

    def model_loader(*args, **kwargs):
        load_calls["model"] += 1
        return model

    def processor_loader(*args, **kwargs):
        load_calls["processor"] += 1
        return processor

    backend = FlorenceBackend(_config(), model_loader=model_loader, processor_loader=processor_loader)
    image = np.zeros((16, 16, 3), dtype=np.uint8)

    result_one = backend.run_task(image, "<VQA>", "Question")
    result_two = backend.run_task(image, "<VQA>", "Question")

    assert result_one["status"] == "completed"
    assert result_two["status"] == "completed"
    assert backend.is_loaded is True
    assert load_calls["model"] == 1
    assert load_calls["processor"] == 1
    assert len(model.generate_calls) == 2
    assert processor.calls[0]["text"] == "<VQA>Question"
    assert isinstance(processor.calls[0]["text"], str)


def test_non_square_images_are_padded_before_processor_call() -> None:
    backend = FlorenceBackend(
        _config(),
        model_loader=lambda *a, **k: FakeModel(),
        processor_loader=lambda *a, **k: FakeProcessor(),
    )
    image = np.zeros((20, 40, 3), dtype=np.uint8)

    result = backend.run_task(image, "<VQA>", "Question")

    assert result["status"] == "completed"
    assert backend._processor.calls[0]["size"] == (40, 20)


def test_prompt_is_combined_and_not_passed_as_batch_elements() -> None:
    processor = FakeProcessor()
    backend = FlorenceBackend(
        _config(),
        model_loader=lambda *a, **k: FakeModel(),
        processor_loader=lambda *a, **k: processor,
    )

    result = backend.run_task(np.zeros((16, 16, 3), dtype=np.uint8), "<VQA>", "Which body type?")

    assert result["status"] == "completed"
    assert processor.calls[0]["text"] == "<VQA>Which body type?"
    assert not isinstance(processor.calls[0]["text"], list)


def test_no_task_prefix_prompt_text_is_passed_safely() -> None:
    processor = FakeProcessor()
    backend = FlorenceBackend(
        _config(),
        model_loader=lambda *a, **k: FakeModel(),
        processor_loader=lambda *a, **k: processor,
    )

    result = backend.run_task(np.zeros((16, 16, 3), dtype=np.uint8), "", "What colour is the vehicle?")

    assert result["status"] == "completed"
    assert processor.calls[0]["text"] == "What colour is the vehicle?"


def test_backend_loads_once_across_multiple_prompt_variants() -> None:
    model = FakeModel()
    processor = FakeProcessor()
    load_calls = {"model": 0, "processor": 0}

    def model_loader(*args, **kwargs):
        load_calls["model"] += 1
        return model

    def processor_loader(*args, **kwargs):
        load_calls["processor"] += 1
        return processor

    backend = FlorenceBackend(_config(), model_loader=model_loader, processor_loader=processor_loader)
    image = np.zeros((16, 16, 3), dtype=np.uint8)

    backend.run_task(image, "<VQA>", "Prompt A")
    backend.run_task(image, "<VQA>", "Prompt B")
    backend.run_task(image, "", "Prompt C")

    assert backend.metrics["florence_load_attempts"] == 1
    assert backend.metrics["florence_load_successes"] == 1
    assert load_calls["model"] == 1
    assert load_calls["processor"] == 1


def test_device_and_dtype_resolution_use_cpu_safe_defaults() -> None:
    backend = FlorenceBackend(_config(device="cpu", dtype="auto"), model_loader=lambda *a, **k: FakeModel(), processor_loader=lambda *a, **k: FakeProcessor())
    backend.load()
    assert backend.resolved_device == "cpu"
    assert backend.resolved_dtype == "float32"


def test_adapter_enabled_uses_adapter_loader(tmp_path: Path) -> None:
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    model = FakeModel()
    processor = FakeProcessor()
    adapter_calls = {"count": 0}

    def adapter_loader(base_model, adapter_path):
        adapter_calls["count"] += 1
        return base_model

    backend = FlorenceBackend(
        _config(adapter_path=str(adapter_dir), adapter_enabled=True),
        model_loader=lambda *a, **k: model,
        processor_loader=lambda *a, **k: processor,
        adapter_loader=adapter_loader,
    )
    backend.load()

    assert backend.adapter_active is True
    assert adapter_calls["count"] == 1
    assert backend.metrics["florence_adapter_requested"] is True
    assert backend.metrics["florence_adapter_load_attempts"] == 1
    assert backend.metrics["florence_adapter_load_successes"] == 1
    assert backend.metrics["florence_adapter_loaded"] is True
    assert backend.metrics["florence_effective_model_type"] == "peft_adapter"


def test_missing_adapter_fails_clearly(tmp_path: Path) -> None:
    backend = FlorenceBackend(
        _config(adapter_path=str(tmp_path / "missing_adapter"), adapter_enabled=True),
        model_loader=lambda *a, **k: FakeModel(),
        processor_loader=lambda *a, **k: FakeProcessor(),
    )
    with pytest.raises(FileNotFoundError):
        backend.load()
    assert backend.metrics["florence_adapter_requested"] is True
    assert backend.metrics["florence_adapter_load_attempts"] == 1
    assert backend.metrics["florence_adapter_loaded"] is False
    assert "Configured Florence adapter path does not exist" in str(backend.metrics["florence_adapter_load_error"])


def test_model_load_error_is_recorded() -> None:
    def _boom(*args, **kwargs):
        raise RuntimeError("load failed")

    backend = FlorenceBackend(_config(), model_loader=_boom, processor_loader=lambda *a, **k: FakeProcessor())
    with pytest.raises(RuntimeError, match="load failed"):
        backend.load()
    assert backend.metrics["florence_load_failures"] == 1


def test_critical_missing_language_keys_fail_clearly() -> None:
    backend = FlorenceBackend(
        _config(),
        model_loader=lambda *a, **k: (
            FakeModel(),
            {"missing_keys": ["language_model.lm_head.weight"], "unexpected_keys": [], "mismatched_keys": [], "error_msgs": []},
        ),
        processor_loader=lambda *a, **k: FakeProcessor(),
    )
    with pytest.raises(RuntimeError, match="critical language weights"):
        backend.load()


def test_explicit_processor_source_is_recorded() -> None:
    backend = FlorenceBackend(
        _config(processor_path="custom_processor"),
        model_loader=lambda *a, **k: FakeModel(),
        processor_loader=lambda source, **k: FakeProcessor(),
    )
    backend.load()
    assert backend.metrics["florence_processor_path"] == "custom_processor"
    assert backend.metrics["florence_base_model_path"] == "microsoft/Florence-2-base-ft"


def test_generation_error_returns_error_status() -> None:
    class BrokenModel(FakeModel):
        def generate(self, **kwargs):
            raise RuntimeError("generation failed")

    backend = FlorenceBackend(
        _config(),
        model_loader=lambda *a, **k: BrokenModel(),
        processor_loader=lambda *a, **k: FakeProcessor(),
    )
    result = backend.run_task(np.zeros((16, 16, 3), dtype=np.uint8), "<VQA>", "Question")
    assert result["status"] == "error"
    assert "generation failed" in result["reason"]


def test_generated_token_slicing_uses_token_count_not_character_count() -> None:
    class SlicingProcessor(FakeProcessor):
        def batch_decode(self, generated_ids, skip_special_tokens=False):
            token_count = int(generated_ids.shape[1])
            if token_count == 5:
                return ["qA"]
            if skip_special_tokens:
                return ["sedan"]
            return ["sedan"]

        def post_process_generation(self, generated_text, task, image_size):
            return {task: generated_text}

    class SlicingModel(FakeModel):
        def generate(self, **kwargs):
            self.generate_calls.append(kwargs)
            return torch.tensor([[11, 12, 13, 21, 22]], dtype=torch.int64)

    backend = FlorenceBackend(
        _config(),
        model_loader=lambda *a, **k: SlicingModel(),
        processor_loader=lambda *a, **k: SlicingProcessor(),
    )

    result = backend.run_task(np.zeros((16, 16, 3), dtype=np.uint8), "<VQA>", "Question")

    assert result["status"] == "completed"
    assert result["payload"]["decoded_full_text"] == "qA"
    assert result["payload"]["decoded_generated_only_text"] == "sedan"
    assert result["payload"]["generated_text"] == "sedan"


def test_explicit_cpu_with_float16_is_rejected() -> None:
    backend = FlorenceBackend(
        _config(device="cpu", dtype="float16"),
        model_loader=lambda *a, **k: FakeModel(),
        processor_loader=lambda *a, **k: FakeProcessor(),
    )
    with pytest.raises(Exception, match="float16 requires a CUDA device"):
        backend.load()


def test_model_loader_receives_resolved_torch_dtype(monkeypatch) -> None:
    captured_kwargs = {}

    def model_loader(*args, **kwargs):
        captured_kwargs.update(kwargs)
        return FakeModel()

    runtime_device = RuntimeDevice(
        configured_device="auto",
        configured_dtype="auto",
        device=torch.device("cuda:0"),
        dtype=torch.float16,
        cuda_available=True,
        cuda_device_count=1,
        device_name="GPU-0",
        reason="CUDA is available and selected.",
        torch_version="test",
        torch_cuda_version="12.1",
    )
    monkeypatch.setattr(florence_backend_module, "resolve_runtime_device", lambda *args, **kwargs: runtime_device)

    backend = FlorenceBackend(
        _config(device="auto", dtype="auto"),
        model_loader=model_loader,
        processor_loader=lambda *a, **k: FakeProcessor(),
    )
    backend.load()

    assert captured_kwargs["torch_dtype"] == torch.float16
    assert backend._model.to_device == torch.device("cuda:0")


def test_adapter_disabled_loading_remains_unchanged() -> None:
    backend = FlorenceBackend(
        _config(adapter_enabled=False),
        model_loader=lambda *a, **k: FakeModel(),
        processor_loader=lambda *a, **k: FakeProcessor(),
    )

    backend.load()

    assert backend.adapter_active is False
    assert backend.metrics["florence_adapter_active"] is False


def test_cuda_and_dtype_behavior_remain_unchanged(monkeypatch) -> None:
    runtime_device = RuntimeDevice(
        configured_device="auto",
        configured_dtype="auto",
        device=torch.device("cuda:0"),
        dtype=torch.float16,
        cuda_available=True,
        cuda_device_count=1,
        device_name="GPU-0",
        reason="CUDA is available and selected.",
        torch_version="test",
        torch_cuda_version="12.1",
    )
    monkeypatch.setattr(florence_backend_module, "resolve_runtime_device", lambda *args, **kwargs: runtime_device)

    backend = FlorenceBackend(
        _config(device="auto", dtype="auto"),
        model_loader=lambda *a, **k: FakeModel(),
        processor_loader=lambda *a, **k: FakeProcessor(),
    )
    backend.load()

    assert backend.resolved_device == "cuda:0"
    assert backend.resolved_dtype == "float16"


def test_close_resets_loaded_state() -> None:
    backend = FlorenceBackend(
        _config(),
        model_loader=lambda *a, **k: FakeModel(),
        processor_loader=lambda *a, **k: FakeProcessor(),
    )
    backend.load()
    backend.close()
    assert backend.is_loaded is False
    assert backend.metrics["florence_loaded"] is False
