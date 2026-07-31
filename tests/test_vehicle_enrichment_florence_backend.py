from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.vehicle_enrichment.shared.florence_backend import FlorenceBackend, FlorenceBackendConfig


class FakeTensor:
    def __init__(self, value):
        self.value = value
        self.device = None

    def to(self, device):
        self.device = device
        return self


class FakeProcessor:
    def __init__(self):
        self.calls = []

    def __call__(self, text, images, return_tensors):
        self.calls.append({"text": text, "size": images.size, "return_tensors": return_tensors})
        return {
            "input_ids": FakeTensor([1, 2, 3]),
            "pixel_values": FakeTensor([4, 5, 6]),
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
        return [[1, 2, 3]]


def _config(**overrides) -> FlorenceBackendConfig:
    payload = {
        "enabled": True,
        "backend": "florence2",
        "base_model_id": "microsoft/Florence-2-base-ft",
        "adapter_path": "",
        "adapter_enabled": False,
        "device": "cpu",
        "dtype": "auto",
        "trust_remote_code": True,
        "attention_implementation": "eager",
        "max_new_tokens": 128,
        "num_beams": 3,
        "use_cache": False,
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


def test_non_square_images_are_padded_before_processor_call() -> None:
    backend = FlorenceBackend(
        _config(),
        model_loader=lambda *a, **k: FakeModel(),
        processor_loader=lambda *a, **k: FakeProcessor(),
    )
    image = np.zeros((20, 40, 3), dtype=np.uint8)

    result = backend.run_task(image, "<VQA>", "Question")

    assert result["status"] == "completed"
    assert backend._processor.calls[0]["size"] == (40, 40)


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


def test_missing_adapter_fails_clearly(tmp_path: Path) -> None:
    backend = FlorenceBackend(
        _config(adapter_path=str(tmp_path / "missing_adapter"), adapter_enabled=True),
        model_loader=lambda *a, **k: FakeModel(),
        processor_loader=lambda *a, **k: FakeProcessor(),
    )
    with pytest.raises(FileNotFoundError):
        backend.load()


def test_model_load_error_is_recorded() -> None:
    def _boom(*args, **kwargs):
        raise RuntimeError("load failed")

    backend = FlorenceBackend(_config(), model_loader=_boom, processor_loader=lambda *a, **k: FakeProcessor())
    with pytest.raises(RuntimeError, match="load failed"):
        backend.load()
    assert backend.metrics["florence_load_failures"] == 1


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
