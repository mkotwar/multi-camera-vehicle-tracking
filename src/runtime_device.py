from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from .models import ConfigurationError


CPU_ONLY_BUILD_REASON = "CUDA unavailable because installed PyTorch build is CPU-only."
CUDA_RUNTIME_UNAVAILABLE_REASON = "CUDA unavailable because torch.cuda.is_available() is False."


@dataclass(slots=True, frozen=True)
class RuntimeDevice:
    configured_device: str
    configured_dtype: str
    device: torch.device
    dtype: torch.dtype
    cuda_available: bool
    cuda_device_count: int
    device_name: str | None
    reason: str
    torch_version: str
    torch_cuda_version: str | None

    @property
    def resolved_device(self) -> str:
        if self.device.index is None:
            return self.device.type
        return f"{self.device.type}:{self.device.index}"

    @property
    def resolved_dtype(self) -> str:
        return dtype_name(self.dtype)

    @property
    def cuda_device_name(self) -> str | None:
        return self.device_name

    @property
    def yolo_device(self) -> int | str:
        return self.device.index if self.device.type == "cuda" else "cpu"

    @property
    def yolo_half(self) -> bool:
        return self.device.type == "cuda" and self.dtype == torch.float16

    @property
    def yolo_quantize(self) -> int | None:
        return 16 if self.yolo_half else None


def dtype_name(value: Any) -> str:
    raw = str(value)
    return raw.split(".")[-1] if "." in raw else raw


def move_batch_to_device(
    inputs: dict[str, Any],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in inputs.items():
        if not hasattr(value, "to"):
            moved[key] = value
            continue
        target_dtype = None
        try:
            if torch.is_floating_point(value):
                target_dtype = dtype
            else:
                target_dtype = getattr(value, "dtype", None)
        except TypeError:
            target_dtype = getattr(value, "dtype", None)
        try:
            if target_dtype is None:
                moved[key] = value.to(device=device)
            else:
                moved[key] = value.to(device=device, dtype=target_dtype)
        except TypeError:
            moved[key] = value.to(str(device))
    return moved


def resolve_runtime_device(
    configured_device: Any = "auto",
    configured_dtype: Any = "auto",
) -> RuntimeDevice:
    normalized_device = str(configured_device or "auto").strip().lower()
    normalized_dtype = str(configured_dtype or "auto").strip().lower()
    cuda_available = bool(torch.cuda.is_available())
    cuda_device_count = int(torch.cuda.device_count()) if cuda_available else 0
    torch_version = str(torch.__version__)
    torch_cuda_version = str(torch.version.cuda) if torch.version.cuda is not None else None
    cuda_build_is_available = torch.version.cuda is not None

    device = _resolve_device(
        normalized_device=normalized_device,
        cuda_available=cuda_available,
        cuda_device_count=cuda_device_count,
        cuda_build_is_available=cuda_build_is_available,
    )
    dtype = _resolve_dtype(normalized_dtype=normalized_dtype, device=device)
    device_name = None
    if device.type == "cuda":
        device_name = str(torch.cuda.get_device_name(device.index or 0))

    if device.type == "cuda":
        reason = "CUDA is available and selected."
    elif not cuda_build_is_available:
        reason = CPU_ONLY_BUILD_REASON
    else:
        reason = CUDA_RUNTIME_UNAVAILABLE_REASON

    return RuntimeDevice(
        configured_device=normalized_device,
        configured_dtype=normalized_dtype,
        device=device,
        dtype=dtype,
        cuda_available=cuda_available,
        cuda_device_count=cuda_device_count,
        device_name=device_name,
        reason=reason,
        torch_version=torch_version,
        torch_cuda_version=torch_cuda_version,
    )


def _resolve_device(
    *,
    normalized_device: str,
    cuda_available: bool,
    cuda_device_count: int,
    cuda_build_is_available: bool,
) -> torch.device:
    if normalized_device == "auto":
        return torch.device("cuda:0" if cuda_available else "cpu")
    if normalized_device == "cpu":
        return torch.device("cpu")
    if normalized_device == "cuda":
        _ensure_cuda_available(cuda_available=cuda_available, cuda_build_is_available=cuda_build_is_available)
        return torch.device("cuda:0")
    if normalized_device.startswith("cuda:"):
        _ensure_cuda_available(cuda_available=cuda_available, cuda_build_is_available=cuda_build_is_available)
        raw_index = normalized_device.split(":", 1)[1]
        try:
            device_index = int(raw_index)
        except ValueError as exc:
            raise ConfigurationError(
                f"Unsupported device value: {normalized_device}. Expected auto, cpu, cuda, or cuda:<index>."
            ) from exc
        if device_index < 0 or device_index >= cuda_device_count:
            raise ConfigurationError(
                f"Requested CUDA device {normalized_device} is invalid. Available CUDA device count: {cuda_device_count}."
            )
        return torch.device(f"cuda:{device_index}")
    raise ConfigurationError(
        f"Unsupported device value: {normalized_device}. Expected auto, cpu, cuda, or cuda:<index>."
    )


def _resolve_dtype(*, normalized_dtype: str, device: torch.device) -> torch.dtype:
    if normalized_dtype == "auto":
        return torch.float16 if device.type == "cuda" else torch.float32
    if normalized_dtype in {"float16", "fp16"}:
        if device.type != "cuda":
            raise ConfigurationError("Invalid dtype configuration: float16 requires a CUDA device.")
        return torch.float16
    if normalized_dtype in {"float32", "fp32"}:
        return torch.float32
    raise ConfigurationError("Unsupported dtype value. Expected auto, float16, or float32.")


def _ensure_cuda_available(*, cuda_available: bool, cuda_build_is_available: bool) -> None:
    if cuda_available:
        return
    if not cuda_build_is_available:
        raise ConfigurationError(
            "CUDA was explicitly requested, but the installed PyTorch build is CPU-only (torch.version.cuda is None)."
        )
    raise ConfigurationError("CUDA was explicitly requested, but torch.cuda.is_available() is False.")
