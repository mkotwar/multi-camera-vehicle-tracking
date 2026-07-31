from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
import time
from typing import Any


@dataclass(slots=True, frozen=True)
class FlorenceBackendConfig:
    enabled: bool
    backend: str
    base_model_id: str
    adapter_path: str
    adapter_enabled: bool
    device: str
    dtype: str
    trust_remote_code: bool
    attention_implementation: str
    max_new_tokens: int
    num_beams: int
    use_cache: bool
    lazy_load: bool


class FlorenceBackend:
    def __init__(
        self,
        config: FlorenceBackendConfig,
        *,
        logger: logging.Logger | None = None,
        model_loader: Any | None = None,
        processor_loader: Any | None = None,
        adapter_loader: Any | None = None,
    ) -> None:
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self._repo_root = Path(__file__).resolve().parents[3]
        self._loaded = False
        self._model: Any | None = None
        self._processor: Any | None = None
        self._resolved_device = "cpu"
        self._resolved_dtype = "float32"
        self._model_identifier = str(config.base_model_id)
        self._adapter_active = False
        self._model_loader = model_loader
        self._processor_loader = processor_loader
        self._adapter_loader = adapter_loader
        self._metrics: dict[str, Any] = {
            "florence_load_attempts": 0,
            "florence_load_successes": 0,
            "florence_load_failures": 0,
            "florence_loaded": False,
            "florence_model_id": str(config.base_model_id),
            "florence_adapter_path": self._sanitize_path(config.adapter_path),
            "florence_adapter_active": bool(config.adapter_enabled),
            "florence_device": None,
            "florence_dtype": None,
            "florence_load_duration_ms": 0.0,
            "gpu_memory_allocated_mb": None,
            "gpu_memory_reserved_mb": None,
        }

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def model_identifier(self) -> str:
        return self._model_identifier

    @property
    def adapter_active(self) -> bool:
        return self._adapter_active

    @property
    def resolved_device(self) -> str:
        return self._resolved_device

    @property
    def resolved_dtype(self) -> str:
        return self._resolved_dtype

    @property
    def metrics(self) -> dict[str, Any]:
        return dict(self._metrics)

    def load(self) -> None:
        if self._loaded or not self.config.enabled:
            return
        self._metrics["florence_load_attempts"] += 1
        started_at = time.perf_counter()
        self.logger.info("Florence lazy loading started")
        try:
            import torch
            from peft import PeftModel
            from transformers.configuration_utils import PretrainedConfig
            from transformers.tokenization_utils_base import PreTrainedTokenizerBase
            from transformers import AutoModelForCausalLM, AutoProcessor

            # Florence-2 remote config expects this attribute during dynamic-config
            # construction, but newer transformers builds may not define it.
            if not hasattr(PretrainedConfig, "forced_bos_token_id"):
                PretrainedConfig.forced_bos_token_id = None
            if not hasattr(PreTrainedTokenizerBase, "additional_special_tokens"):
                PreTrainedTokenizerBase.additional_special_tokens = property(
                    lambda self: list(getattr(self, "extra_special_tokens", []) or [])
                )
            if not hasattr(PreTrainedTokenizerBase, "additional_special_tokens_ids"):
                PreTrainedTokenizerBase.additional_special_tokens_ids = property(
                    lambda self: list(getattr(self, "extra_special_tokens_ids", []) or [])
                )

            self._resolved_device = self._select_device(torch)
            torch_dtype = self._select_torch_dtype(torch, self._resolved_device)
            self._resolved_dtype = self._dtype_name(torch_dtype)
            model_source = self._resolve_model_source(self.config.base_model_id)
            self._model_identifier = str(model_source)
            model_loader = self._model_loader or AutoModelForCausalLM.from_pretrained
            processor_loader = self._processor_loader or AutoProcessor.from_pretrained
            self._model = model_loader(
                model_source,
                trust_remote_code=self.config.trust_remote_code,
                attn_implementation=self.config.attention_implementation,
                torch_dtype=torch_dtype,
            )
            if hasattr(self._model, "to"):
                self._model = self._model.to(self._resolved_device)
            self._processor = processor_loader(
                model_source,
                trust_remote_code=self.config.trust_remote_code,
            )
            self._adapter_active = False
            if self.config.adapter_enabled:
                adapter_path = self._resolve_optional_path(self.config.adapter_path)
                if adapter_path is None:
                    raise FileNotFoundError(f"Configured Florence adapter path does not exist: {self.config.adapter_path}")
                adapter_loader = self._adapter_loader or PeftModel.from_pretrained
                self._model = adapter_loader(self._model, str(adapter_path))
                self._adapter_active = True
            if hasattr(self._model, "eval"):
                self._model.eval()
            self._loaded = True
            self._metrics["florence_load_successes"] += 1
            self._metrics["florence_loaded"] = True
            self._metrics["florence_model_id"] = self._model_identifier
            self._metrics["florence_adapter_active"] = self._adapter_active
            self._metrics["florence_device"] = self._resolved_device
            self._metrics["florence_dtype"] = self._resolved_dtype
            self._metrics["florence_load_duration_ms"] = float((time.perf_counter() - started_at) * 1000.0)
            self._update_gpu_metrics(torch)
            self.logger.info("Florence model: %s", self._model_identifier)
            self.logger.info("Florence adapter configured: %s", bool(self.config.adapter_path))
            self.logger.info("Florence adapter active for body type: %s", self._adapter_active)
            self.logger.info("Florence device: %s", self._resolved_device)
            self.logger.info("Florence dtype: %s", self._resolved_dtype)
            self.logger.info("Florence loaded once in %.3f seconds", self._metrics["florence_load_duration_ms"] / 1000.0)
        except Exception:
            self._loaded = False
            self._metrics["florence_load_failures"] += 1
            self._metrics["florence_loaded"] = False
            raise

    def run_task(
        self,
        image: Any,
        task_prompt: str,
        text_input: str | None = None,
        *,
        adapter_active: bool | None = None,
    ) -> dict[str, Any]:
        if not self.config.enabled:
            return {"status": "disabled", "reason": "Florence backend disabled.", "payload": None}
        if not self._loaded and self.config.lazy_load:
            try:
                self.load()
            except Exception as exc:
                return {"status": "error", "reason": str(exc), "payload": None}
        if not self._loaded or self._model is None or self._processor is None:
            return {"status": "error", "reason": "Florence model is not loaded.", "payload": None}
        if adapter_active is not None and adapter_active != self._adapter_active:
            return {
                "status": "error",
                "reason": "Requested adapter state does not match the loaded Florence backend.",
                "payload": None,
            }
        try:
            import cv2
            import numpy as np
            from PIL import Image
            import torch

            if image is None or getattr(image, "size", 0) == 0:
                raise ValueError("Input image is empty.")
            started_at = time.perf_counter()
            prepared_image = self._pad_to_square(image, np)
            image_pil = Image.fromarray(cv2.cvtColor(prepared_image, cv2.COLOR_BGR2RGB))
            prompt = task_prompt + text_input if text_input else task_prompt
            inputs = self._processor(text=prompt, images=image_pil, return_tensors="pt")
            prepared_inputs = {
                key: value.to(self._resolved_device) if hasattr(value, "to") else value
                for key, value in dict(inputs).items()
            }
            with torch.no_grad():
                generated_ids = self._model.generate(
                    input_ids=prepared_inputs["input_ids"],
                    pixel_values=prepared_inputs["pixel_values"],
                    max_new_tokens=self.config.max_new_tokens,
                    do_sample=False,
                    num_beams=self.config.num_beams,
                    use_cache=self.config.use_cache,
                )
            generated_text = self._processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
            parsed_answer = self._processor.post_process_generation(
                generated_text,
                task=task_prompt,
                image_size=(image_pil.width, image_pil.height),
            )
            inference_duration_ms = float((time.perf_counter() - started_at) * 1000.0)
            self._update_gpu_metrics(torch)
            return {
                "status": "completed",
                "reason": None,
                "payload": {
                    "task_prompt": task_prompt,
                    "prompt_text": text_input,
                    "generated_text": generated_text,
                    "parsed_answer": parsed_answer,
                    "adapter_active": self._adapter_active,
                    "model_identifier": self._model_identifier,
                    "device": self._resolved_device,
                    "dtype": self._resolved_dtype,
                    "inference_duration_ms": inference_duration_ms,
                },
            }
        except Exception as exc:
            return {"status": "error", "reason": str(exc), "payload": None}

    def close(self) -> None:
        try:
            import torch
        except Exception:
            torch = None
        self._model = None
        self._processor = None
        self._loaded = False
        self._metrics["florence_loaded"] = False
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _resolve_model_source(self, raw_value: str) -> str:
        resolved = self._resolve_optional_path(raw_value)
        return str(resolved) if resolved is not None else str(raw_value)

    def _resolve_optional_path(self, raw_value: str) -> Path | None:
        if raw_value in ("", None):
            return None
        candidate = Path(str(raw_value)).expanduser()
        if not candidate.is_absolute():
            candidate = (self._repo_root / candidate).resolve()
        else:
            candidate = candidate.resolve()
        return candidate if candidate.exists() else None

    def _select_device(self, torch_module: Any) -> str:
        requested = str(self.config.device or "auto").strip().lower()
        if requested == "auto":
            return "cuda" if torch_module.cuda.is_available() else "cpu"
        if requested == "cuda" and not torch_module.cuda.is_available():
            raise RuntimeError("CUDA requested for Florence, but CUDA is unavailable.")
        if requested not in {"cpu", "cuda"}:
            raise RuntimeError(f"Unsupported Florence device value: {self.config.device}")
        return requested

    def _select_torch_dtype(self, torch_module: Any, device: str) -> Any:
        normalized = str(self.config.dtype or "auto").strip().lower()
        if normalized == "auto":
            return torch_module.float16 if device == "cuda" else torch_module.float32
        mapping = {
            "float16": torch_module.float16,
            "fp16": torch_module.float16,
            "float32": torch_module.float32,
            "fp32": torch_module.float32,
            "bfloat16": torch_module.bfloat16,
            "bf16": torch_module.bfloat16,
        }
        return mapping.get(normalized, torch_module.float32)

    @staticmethod
    def _dtype_name(value: Any) -> str:
        raw = str(value)
        return raw.split(".")[-1] if "." in raw else raw

    @staticmethod
    def _sanitize_path(raw_value: str) -> str | None:
        if raw_value in ("", None):
            return None
        path = Path(str(raw_value))
        return path.name if path.is_absolute() else str(path).replace("\\", "/")

    def _update_gpu_metrics(self, torch_module: Any) -> None:
        if torch_module.cuda.is_available():
            self._metrics["gpu_memory_allocated_mb"] = round(float(torch_module.cuda.memory_allocated() / (1024 * 1024)), 3)
            self._metrics["gpu_memory_reserved_mb"] = round(float(torch_module.cuda.memory_reserved() / (1024 * 1024)), 3)

    @staticmethod
    def _pad_to_square(image: Any, np_module: Any) -> Any:
        height, width = image.shape[:2]
        if height == width:
            return image
        size = max(height, width)
        padded = np_module.zeros((size, size, image.shape[2]), dtype=image.dtype)
        top = (size - height) // 2
        left = (size - width) // 2
        padded[top : top + height, left : left + width] = image
        return padded
