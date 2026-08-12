from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
import time
from typing import Any

from src.runtime_device import RuntimeDevice, move_batch_to_device, resolve_runtime_device


CRITICAL_LANGUAGE_KEYS = {
    "language_model.model.decoder.embed_tokens.weight",
    "language_model.model.encoder.embed_tokens.weight",
    "language_model.lm_head.weight",
}


@dataclass(slots=True, frozen=True)
class FlorenceBackendConfig:
    enabled: bool
    backend: str
    base_model_id: str
    processor_path: str
    adapter_path: str
    adapter_enabled: bool
    device: str
    dtype: str
    trust_remote_code: bool
    attention_implementation: str
    max_new_tokens: int
    num_beams: int
    use_cache: bool
    local_files_only: bool
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
        adapter_enabled_override: bool | None = None,
    ) -> None:
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self._repo_root = Path(__file__).resolve().parents[3]
        self._loaded = False
        self._model: Any | None = None
        self._processor: Any | None = None
        self._resolved_device = "cpu"
        self._resolved_dtype = "float32"
        self._runtime_device: RuntimeDevice | None = None
        self._model_identifier = str(config.base_model_id)
        self._processor_identifier = str(config.processor_path or config.base_model_id)
        self._adapter_active = False
        self._adapter_requested = bool(
            config.adapter_enabled if adapter_enabled_override is None else adapter_enabled_override
        )
        self._adapter_path_resolved: str | None = None
        self._effective_model_type = "base_model"
        self._model_loader = model_loader
        self._processor_loader = processor_loader
        self._adapter_loader = adapter_loader
        self._adapter_enabled_override = adapter_enabled_override
        self._loading_info: dict[str, Any] = {
            "missing_keys": [],
            "unexpected_keys": [],
            "mismatched_keys": [],
            "error_msgs": [],
        }
        self._metrics: dict[str, Any] = {
            "florence_load_attempts": 0,
            "florence_load_successes": 0,
            "florence_load_failures": 0,
            "florence_loaded": False,
            "florence_model_id": str(config.base_model_id),
            "florence_base_model_path": self._sanitize_path(config.base_model_id),
            "florence_processor_path": self._sanitize_path(config.processor_path),
            "florence_adapter_requested": bool(config.adapter_enabled),
            "florence_adapter_path": self._sanitize_path(config.adapter_path),
            "florence_adapter_active": bool(config.adapter_enabled),
            "florence_adapter_load_attempts": 0,
            "florence_adapter_load_successes": 0,
            "florence_adapter_loaded": False,
            "florence_adapter_load_error": None,
            "florence_effective_model_type": "base_model",
            "florence_device": None,
            "florence_dtype": None,
            "florence_model_class": None,
            "florence_processor_class": None,
            "florence_load_duration_ms": 0.0,
            "florence_missing_keys": [],
            "florence_unexpected_keys": [],
            "florence_mismatched_keys": [],
            "florence_critical_missing_keys": [],
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
    def processor_identifier(self) -> str:
        return self._processor_identifier

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
    def loading_info(self) -> dict[str, Any]:
        return dict(self._loading_info)

    @property
    def metrics(self) -> dict[str, Any]:
        payload = dict(self._metrics)
        payload["florence_missing_keys"] = list(self._metrics["florence_missing_keys"])
        payload["florence_unexpected_keys"] = list(self._metrics["florence_unexpected_keys"])
        payload["florence_mismatched_keys"] = list(self._metrics["florence_mismatched_keys"])
        payload["florence_critical_missing_keys"] = list(self._metrics["florence_critical_missing_keys"])
        return payload

    def load(self) -> None:
        if self._loaded or not self.config.enabled:
            return
        self._metrics["florence_load_attempts"] += 1
        started_at = time.perf_counter()
        self.logger.info("Florence lazy loading started")
        try:
            import torch
            from peft import PeftModel
            from transformers import AutoModelForCausalLM, AutoProcessor

            runtime_device = resolve_runtime_device(self.config.device, self.config.dtype)
            self._runtime_device = runtime_device
            self._resolved_device = runtime_device.resolved_device
            self._resolved_dtype = runtime_device.resolved_dtype
            model_source = self._resolve_required_source(self.config.base_model_id, label="model")
            processor_source = self._resolve_processor_source(model_source)
            self._model_identifier = str(model_source)
            self._processor_identifier = str(processor_source)
            self._metrics["florence_base_model_path"] = self._sanitize_path(self._model_identifier)
            self._metrics["florence_processor_path"] = self._sanitize_path(self._processor_identifier)
            self._metrics["florence_adapter_requested"] = self._adapter_requested
            self._metrics["florence_adapter_path"] = self._sanitize_path(self.config.adapter_path)
            self._metrics["florence_adapter_load_error"] = None

            model_kwargs = {
                "trust_remote_code": self.config.trust_remote_code,
                "local_files_only": self.config.local_files_only,
                "attn_implementation": self.config.attention_implementation,
                "torch_dtype": runtime_device.dtype,
            }
            processor_kwargs = {
                "trust_remote_code": self.config.trust_remote_code,
                "local_files_only": self.config.local_files_only,
            }

            if self._model_loader is None:
                self._model, loading_info = AutoModelForCausalLM.from_pretrained(
                    str(model_source),
                    output_loading_info=True,
                    **model_kwargs,
                )
            else:
                loaded = self._model_loader(str(model_source), **model_kwargs)
                if isinstance(loaded, tuple) and len(loaded) == 2:
                    self._model, loading_info = loaded
                else:
                    self._model = loaded
                    loading_info = {}
            processor_loader = self._processor_loader or AutoProcessor.from_pretrained
            self._processor = processor_loader(str(processor_source), **processor_kwargs)
            self._loading_info = self._normalize_loading_info(loading_info)
            self._validate_loading_info()

            self._adapter_active = False
            self._adapter_path_resolved = None
            self._effective_model_type = "base_model"
            if self._adapter_requested:
                self._metrics["florence_adapter_load_attempts"] += 1
                adapter_path = self._resolve_optional_path(self.config.adapter_path)
                if adapter_path is None:
                    self._metrics["florence_adapter_load_error"] = f"Configured Florence adapter path does not exist: {self.config.adapter_path}"
                    raise FileNotFoundError(f"Configured Florence adapter path does not exist: {self.config.adapter_path}")
                adapter_loader = self._adapter_loader or PeftModel.from_pretrained
                try:
                    if self._adapter_loader is None:
                        self._model = adapter_loader(
                            self._model,
                            str(adapter_path),
                            local_files_only=self.config.local_files_only,
                        )
                    else:
                        self._model = adapter_loader(self._model, str(adapter_path))
                    self._adapter_active = True
                    self._adapter_path_resolved = str(adapter_path)
                    self._effective_model_type = "peft_adapter"
                    self._metrics["florence_adapter_load_successes"] += 1
                    self._metrics["florence_adapter_loaded"] = True
                except Exception as exc:
                    self._metrics["florence_adapter_load_error"] = str(exc)
                    self._metrics["florence_adapter_loaded"] = False
                    raise
            if hasattr(self._model, "to"):
                self._model = self._model.to(runtime_device.device)
            if hasattr(self._model, "eval"):
                self._model.eval()

            self._loaded = True
            self._metrics["florence_load_successes"] += 1
            self._metrics["florence_loaded"] = True
            self._metrics["florence_model_id"] = self._model_identifier
            self._metrics["florence_base_model_path"] = self._sanitize_path(self._model_identifier)
            self._metrics["florence_processor_path"] = self._sanitize_path(self._processor_identifier)
            self._metrics["florence_adapter_active"] = self._adapter_active
            self._metrics["florence_adapter_loaded"] = self._adapter_active
            self._metrics["florence_effective_model_type"] = self._effective_model_type
            self._metrics["florence_device"] = self._resolved_device
            self._metrics["florence_dtype"] = self._resolved_dtype
            self._metrics["florence_model_class"] = type(self._model).__name__
            self._metrics["florence_processor_class"] = type(self._processor).__name__
            self._metrics["florence_load_duration_ms"] = float((time.perf_counter() - started_at) * 1000.0)
            self._metrics["florence_missing_keys"] = list(self._loading_info["missing_keys"])
            self._metrics["florence_unexpected_keys"] = list(self._loading_info["unexpected_keys"])
            self._metrics["florence_mismatched_keys"] = list(self._loading_info["mismatched_keys"])
            self._metrics["florence_critical_missing_keys"] = [
                key for key in self._loading_info["missing_keys"] if key in CRITICAL_LANGUAGE_KEYS
            ]
            self._update_gpu_metrics(torch)
            self.logger.info("Florence model: %s", self._model_identifier)
            self.logger.debug("Florence processor: %s", self._processor_identifier)
            self.logger.debug("Florence adapter requested: %s", self._adapter_requested)
            self.logger.debug("Florence adapter configured path: %s", self._sanitize_path(self.config.adapter_path))
            self.logger.debug("Florence adapter resolved path: %s", self._sanitize_path(self._adapter_path_resolved))
            self.logger.info("Florence adapter active: %s", self._adapter_active)
            self.logger.debug("Florence effective model type: %s", self._effective_model_type)
            self.logger.debug("Florence model class: %s", self._metrics["florence_model_class"])
            self.logger.debug("Florence processor class: %s", self._metrics["florence_processor_class"])
            self.logger.info("Florence device: %s", self._resolved_device)
            self.logger.info("Florence dtype: %s", self._resolved_dtype)
            self.logger.info("Florence model loaded device=%s dtype=%s", self._resolved_device, self._resolved_dtype)
            if runtime_device.device.type == "cuda":
                self.logger.info(
                    "Florence GPU memory allocated_mib=%.3f reserved_mib=%.3f",
                    self._metrics["gpu_memory_allocated_mb"] or 0.0,
                    self._metrics["gpu_memory_reserved_mb"] or 0.0,
                )
            self.logger.info("Florence loaded once in %.3f seconds", self._metrics["florence_load_duration_ms"] / 1000.0)
        except Exception:
            self._loaded = False
            self._metrics["florence_load_failures"] += 1
            self._metrics["florence_loaded"] = False
            if self._adapter_requested and self._metrics["florence_adapter_load_error"] is None:
                self._metrics["florence_adapter_load_error"] = "Adapter load failed before completion."
            raise

    def run_task(
        self,
        image: Any,
        task_prompt: str,
        text_input: str | None = None,
        *,
        adapter_active: bool | None = None,
        generation_overrides: dict[str, Any] | None = None,
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
            from PIL import Image
            import torch

            if image is None or getattr(image, "size", 0) == 0:
                raise ValueError("Input image is empty.")
            started_at = time.perf_counter()
            image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            processor_text = self._build_processor_text(task_prompt, text_input)
            self.logger.debug("Florence task_prompt=%s", task_prompt)
            self.logger.debug("Florence prompt_text=%s", text_input)
            self.logger.debug("Florence final_processor_text=%s", processor_text)
            inputs = self._processor(text=processor_text, images=image_pil, return_tensors="pt")
            if self._runtime_device is None:
                raise RuntimeError("Florence runtime device was not resolved.")
            inputs = move_batch_to_device(
                dict(inputs),
                device=self._runtime_device.device,
                dtype=self._runtime_device.dtype,
            )
            input_ids = inputs["input_ids"]
            pixel_values = inputs["pixel_values"]
            attention_mask = inputs.get("attention_mask")
            self.logger.debug("Florence input_ids shape=%s dtype=%s", self._shape_of(input_ids), self._dtype_of(input_ids))
            self.logger.debug("Florence pixel_values shape=%s dtype=%s", self._shape_of(pixel_values), self._dtype_of(pixel_values))
            generation = self._resolve_generation_settings(generation_overrides)
            with torch.inference_mode():
                if self._runtime_device.device.type == "cuda":
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        generated_ids = self._model.generate(
                            input_ids=input_ids,
                            pixel_values=pixel_values,
                            attention_mask=attention_mask,
                            max_new_tokens=generation["max_new_tokens"],
                            do_sample=generation["do_sample"],
                            num_beams=generation["num_beams"],
                            use_cache=generation["use_cache"],
                            early_stopping=generation["early_stopping"],
                        )
                else:
                    generated_ids = self._model.generate(
                        input_ids=input_ids,
                        pixel_values=pixel_values,
                        attention_mask=attention_mask,
                        max_new_tokens=generation["max_new_tokens"],
                        do_sample=generation["do_sample"],
                        num_beams=generation["num_beams"],
                        use_cache=generation["use_cache"],
                        early_stopping=generation["early_stopping"],
                    )
            prompt_token_count = self._token_count(input_ids)
            generated_only_ids = self._slice_generated_only_ids(generated_ids, prompt_token_count)
            decoded_full = self._processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
            decoded_full_skip_special = self._processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            decoded_generated_only = self._decode_generated_only(generated_only_ids, skip_special_tokens=False)
            decoded_generated_only_skip_special = self._decode_generated_only(generated_only_ids, skip_special_tokens=True)
            generated_text = decoded_generated_only or decoded_full
            self.logger.debug("Florence generated token IDs=%s", self._serialize_generated_ids(generated_ids))
            self.logger.debug("Florence decoded raw text full=%s", decoded_full)
            self.logger.debug("Florence decoded raw text full_skip_special=%s", decoded_full_skip_special)
            self.logger.debug("Florence decoded raw text generated_only=%s", decoded_generated_only)
            self.logger.debug("Florence decoded raw text generated_only_skip_special=%s", decoded_generated_only_skip_special)
            parsed_answer = None
            if str(task_prompt).startswith("<") and hasattr(self._processor, "post_process_generation"):
                parsed_answer = self._processor.post_process_generation(
                    generated_text,
                    task=task_prompt,
                    image_size=(image_pil.width, image_pil.height),
                )
                self.logger.debug("Florence post_processed output=%s", parsed_answer)
            inference_duration_ms = float((time.perf_counter() - started_at) * 1000.0)
            self._update_gpu_metrics(torch)
            return {
                "status": "completed",
                "reason": None,
                "payload": {
                    "task_prompt": task_prompt,
                    "prompt_text": text_input,
                    "final_processor_text": processor_text,
                    "input_ids_shape": self._shape_of(input_ids),
                    "pixel_values_shape": self._shape_of(pixel_values),
                    "generated_ids": self._serialize_generated_ids(generated_ids),
                    "generated_only_ids": self._serialize_generated_ids(generated_only_ids),
                    "generated_text": generated_text,
                    "decoded_full_text": decoded_full,
                    "decoded_full_text_skip_special": decoded_full_skip_special,
                    "decoded_generated_only_text": decoded_generated_only,
                    "decoded_generated_only_text_skip_special": decoded_generated_only_skip_special,
                    "parsed_answer": parsed_answer,
                    "adapter_active": self._adapter_active,
                    "model_identifier": self._model_identifier,
                    "processor_identifier": self._processor_identifier,
                    "device": self._resolved_device,
                    "dtype": self._resolved_dtype,
                    "inference_duration_ms": inference_duration_ms,
                    "generation_settings": generation,
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

    def _resolve_required_source(self, raw_value: str, *, label: str) -> str:
        resolved = self._resolve_optional_path(raw_value)
        if resolved is not None:
            return str(resolved)
        if self.config.local_files_only:
            raise FileNotFoundError(f"Configured Florence {label} path does not exist: {raw_value}")
        return str(raw_value)

    def _resolve_processor_source(self, model_source: str) -> str:
        if self.config.processor_path not in ("", None):
            explicit = self._resolve_optional_path(self.config.processor_path)
            if explicit is not None:
                return str(explicit)
            if not self.config.local_files_only:
                return str(self.config.processor_path)
        adapter = self._resolve_optional_path(self.config.adapter_path)
        if adapter is not None:
            return str(adapter)
        return str(model_source)

    def _resolve_optional_path(self, raw_value: str) -> Path | None:
        if raw_value in ("", None):
            return None
        candidate = Path(str(raw_value)).expanduser()
        if not candidate.is_absolute():
            candidate = (self._repo_root / candidate).resolve()
        else:
            candidate = candidate.resolve()
        return candidate if candidate.exists() else None

    def _validate_loading_info(self) -> None:
        missing_keys = set(self._loading_info.get("missing_keys", []))
        critical_missing = sorted(key for key in missing_keys if key in CRITICAL_LANGUAGE_KEYS)
        if critical_missing:
            raise RuntimeError(
                "Florence checkpoint is missing critical language weights: " + ", ".join(critical_missing)
            )

    @staticmethod
    def _normalize_loading_info(loading_info: Any) -> dict[str, Any]:
        payload = dict(loading_info or {})
        return {
            "missing_keys": list(payload.get("missing_keys", [])),
            "unexpected_keys": list(payload.get("unexpected_keys", [])),
            "mismatched_keys": list(payload.get("mismatched_keys", [])),
            "error_msgs": list(payload.get("error_msgs", [])),
        }

    @staticmethod
    def _sanitize_path(raw_value: str) -> str | None:
        if raw_value in ("", None):
            return None
        path = Path(str(raw_value))
        return path.name if path.is_absolute() else str(path).replace("\\", "/")

    def _update_gpu_metrics(self, torch_module: Any) -> None:
        if self._runtime_device is not None and self._runtime_device.device.type == "cuda" and torch_module.cuda.is_available():
            self._metrics["gpu_memory_allocated_mb"] = round(float(torch_module.cuda.memory_allocated() / (1024 * 1024)), 3)
            self._metrics["gpu_memory_reserved_mb"] = round(float(torch_module.cuda.memory_reserved() / (1024 * 1024)), 3)

    @staticmethod
    def _serialize_generated_ids(generated_ids: Any) -> list[int]:
        if generated_ids is None or len(generated_ids) == 0:
            return []
        first = generated_ids[0]
        if hasattr(first, "tolist"):
            return list(first.tolist())
        return list(first)

    @staticmethod
    def _shape_of(value: Any) -> list[int] | None:
        shape = getattr(value, "shape", None)
        if shape is None:
            return None
        return [int(item) for item in shape]

    @staticmethod
    def _dtype_of(value: Any) -> str | None:
        dtype = getattr(value, "dtype", None)
        return None if dtype is None else str(dtype)

    @staticmethod
    def _token_count(input_ids: Any) -> int:
        shape = getattr(input_ids, "shape", None)
        if shape is not None and len(shape) >= 2:
            return int(shape[1])
        first = input_ids[0] if hasattr(input_ids, "__getitem__") else []
        return len(first)

    @staticmethod
    def _slice_generated_only_ids(generated_ids: Any, prompt_token_count: int) -> Any:
        if generated_ids is None:
            return generated_ids
        try:
            return generated_ids[:, prompt_token_count:]
        except Exception:
            first = generated_ids[0] if len(generated_ids) > 0 else []
            if hasattr(first, "__getitem__"):
                return [first[prompt_token_count:]]
            return generated_ids

    def _decode_generated_only(self, generated_only_ids: Any, *, skip_special_tokens: bool) -> str:
        try:
            decoded = self._processor.batch_decode(generated_only_ids, skip_special_tokens=skip_special_tokens)
        except Exception:
            return ""
        if not decoded:
            return ""
        return str(decoded[0])

    @staticmethod
    def _build_processor_text(task_prompt: str, text_input: str | None) -> str:
        if task_prompt and text_input:
            return f"{task_prompt}{text_input}"
        if task_prompt:
            return str(task_prompt)
        if text_input:
            return str(text_input)
        return ""

    def _resolve_generation_settings(self, generation_overrides: dict[str, Any] | None) -> dict[str, Any]:
        payload = dict(generation_overrides or {})
        num_beams = int(payload.get("num_beams", self.config.num_beams))
        do_sample = bool(payload.get("do_sample", False))
        early_stopping_default = bool(num_beams > 1)
        early_stopping = bool(payload.get("early_stopping", early_stopping_default))
        if num_beams <= 1:
            early_stopping = False
        return {
            "max_new_tokens": int(payload.get("max_new_tokens", self.config.max_new_tokens)),
            "num_beams": num_beams,
            "do_sample": do_sample,
            "use_cache": bool(payload.get("use_cache", self.config.use_cache)),
            "early_stopping": early_stopping,
        }
