from __future__ import annotations

import json
import os
from pathlib import Path
import time
from typing import Any
from urllib.parse import urlparse
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .vehicle_enrichment.taxonomy import SUPPORTED_VEHICLE_CLASSES, SUPPORTED_VEHICLE_COLOUR_LABELS


DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"
DEFAULT_OLLAMA_MODEL = "qwen3:1.7b"
DEFAULT_OLLAMA_KEEP_ALIVE = "10m"
DEFAULT_OLLAMA_TIMEOUT_SECONDS = 45.0
DEFAULT_OLLAMA_TEMPERATURE = 0.0
DEFAULT_OLLAMA_NUM_PREDICT = 128


class OllamaQwenChatLLMProvider:
    def __init__(
        self,
        *,
        base_url: str = DEFAULT_OLLAMA_URL,
        model: str = DEFAULT_OLLAMA_MODEL,
        keep_alive: str = DEFAULT_OLLAMA_KEEP_ALIVE,
        timeout_seconds: float = DEFAULT_OLLAMA_TIMEOUT_SECONDS,
        temperature: float = DEFAULT_OLLAMA_TEMPERATURE,
        num_predict: int = DEFAULT_OLLAMA_NUM_PREDICT,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.keep_alive = keep_alive
        self.timeout_seconds = float(timeout_seconds)
        self.temperature = float(temperature)
        self.num_predict = int(num_predict)
        self.last_metadata: dict[str, Any] = {}

    def parse(self, message: str, context: dict[str, Any]) -> dict[str, Any]:
        started = time.perf_counter()
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": _system_prompt()},
                {"role": "user", "content": json.dumps({"message": message, "context": context}, ensure_ascii=True)},
            ],
            "stream": False,
            "think": False,
            "format": chat_vehicle_query_json_schema(),
            "options": {
                "temperature": self.temperature,
                "num_predict": self.num_predict,
            },
            "keep_alive": self.keep_alive,
        }
        request = Request(
            f"{self.base_url}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                raw = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            raise RuntimeError(f"Ollama chat request failed: {_safe_ollama_error_details(self, exc, started, http_status=exc.code)}") from exc
        except (URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Ollama chat request failed: {_safe_ollama_error_details(self, exc, started)}") from exc
        content = str(dict(raw.get("message") or {}).get("content") or "").strip()
        if not content:
            raise RuntimeError("Ollama response did not include message.content")
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError as exc:
            raise RuntimeError("Ollama message.content was not valid JSON") from exc
        if not isinstance(parsed, dict):
            raise RuntimeError("Ollama structured output was not a JSON object")
        self.last_metadata = {
            "parser": "qwen",
            "provider": "ollama",
            "model": self.model,
            "base_url": self.base_url,
            "keep_alive": self.keep_alive,
            "timeout_seconds": self.timeout_seconds,
            "temperature": self.temperature,
            "num_predict": self.num_predict,
            "think": False,
            "structured_output": True,
            "wall_time_ms": round((time.perf_counter() - started) * 1000, 3),
            "total_duration_ms": _duration_ms(raw.get("total_duration")),
            "load_duration_ms": _duration_ms(raw.get("load_duration")),
            "prompt_eval_duration_ms": _duration_ms(raw.get("prompt_eval_duration")),
            "eval_duration_ms": _duration_ms(raw.get("eval_duration")),
            "prompt_eval_count": raw.get("prompt_eval_count"),
            "eval_count": raw.get("eval_count"),
            "done_reason": raw.get("done_reason"),
        }
        return parsed


def _safe_ollama_error_details(provider: OllamaQwenChatLLMProvider, exc: Exception, started: float, *, http_status: int | None = None) -> str:
    host = urlparse(provider.base_url).netloc or provider.base_url
    elapsed = time.perf_counter() - started
    parts = [
        f"provider=ollama",
        f"model={provider.model}",
        f"host={host}",
        f"timeout_seconds={provider.timeout_seconds:g}",
        f"elapsed_seconds={elapsed:.3f}",
        f"exception={exc.__class__.__name__}",
    ]
    if http_status is not None:
        parts.append(f"http_status={http_status}")
    parts.append(f"detail={exc}")
    return " ".join(parts)


def build_chat_llm_provider_from_env() -> OllamaQwenChatLLMProvider | None:
    env = _video_chat_env()
    provider = env.get("VIDEO_CHAT_LLM_PROVIDER", "").strip().strip('"').lower()
    if provider not in {"ollama", "qwen", "qwen3"}:
        return None
    return OllamaQwenChatLLMProvider(
        base_url=_env_first(env, ["VIDEO_CHAT_QWEN_URL", "VIDEO_CHAT_OLLAMA_URL"], DEFAULT_OLLAMA_URL),
        model=_env_first(env, ["VIDEO_CHAT_QWEN_MODEL", "VIDEO_CHAT_OLLAMA_MODEL"], DEFAULT_OLLAMA_MODEL),
        keep_alive=_env_first(env, ["VIDEO_CHAT_QWEN_KEEP_ALIVE", "VIDEO_CHAT_OLLAMA_KEEP_ALIVE"], DEFAULT_OLLAMA_KEEP_ALIVE),
        timeout_seconds=float(_env_first(env, ["VIDEO_CHAT_QWEN_TIMEOUT_SECONDS", "VIDEO_CHAT_OLLAMA_TIMEOUT_SECONDS"], str(DEFAULT_OLLAMA_TIMEOUT_SECONDS))),
        temperature=float(_env_first(env, ["VIDEO_CHAT_QWEN_TEMPERATURE", "VIDEO_CHAT_OLLAMA_TEMPERATURE"], str(DEFAULT_OLLAMA_TEMPERATURE))),
        num_predict=int(_env_first(env, ["VIDEO_CHAT_QWEN_NUM_PREDICT", "VIDEO_CHAT_OLLAMA_NUM_PREDICT"], str(DEFAULT_OLLAMA_NUM_PREDICT))),
    )


def chat_vehicle_query_json_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "intent",
            "subject",
            "run_filter",
            "class_include",
            "class_exclude",
            "colour_include",
            "colour_exclude",
            "plate_presence",
            "plate_detected",
            "plate_readable",
            "plate_text",
            "start_time",
            "end_time",
            "group_by",
            "operator",
            "show_evidence",
            "context_reference",
        ],
        "properties": {
            "intent": {"type": "string", "enum": ["GENERAL_CHAT", "COUNT", "LIST", "SUMMARY", "GROUP", "COMPARE", "FIND_INTERVALS", "UNIQUE_CLASSES", "UNIQUE_COLOURS"]},
            "subject": {"type": "string", "enum": ["vehicles", "runs"]},
            "run_filter": {"anyOf": [{"type": "string", "enum": ["multiple_cameras"]}, {"type": "null"}]},
            "class_include": {"type": "array", "items": {"type": "string", "enum": [*SUPPORTED_VEHICLE_CLASSES, "UNKNOWN"]}},
            "class_exclude": {"type": "array", "items": {"type": "string", "enum": [*SUPPORTED_VEHICLE_CLASSES, "UNKNOWN"]}},
            "colour_include": {"type": "array", "items": {"type": "string", "enum": list(SUPPORTED_VEHICLE_COLOUR_LABELS)}},
            "colour_exclude": {"type": "array", "items": {"type": "string", "enum": list(SUPPORTED_VEHICLE_COLOUR_LABELS)}},
            "plate_presence": {"anyOf": [{"type": "string", "enum": ["detected", "readable"]}, {"type": "null"}]},
            "plate_detected": {"anyOf": [{"type": "boolean"}, {"type": "null"}]},
            "plate_readable": {"anyOf": [{"type": "boolean"}, {"type": "null"}]},
            "plate_text": {"anyOf": [{"type": "string"}, {"type": "null"}]},
            "start_time": {"anyOf": [{"type": "number"}, {"type": "null"}]},
            "end_time": {"anyOf": [{"type": "number"}, {"type": "null"}]},
            "group_by": {"anyOf": [{"type": "string", "enum": ["vehicle_class", "colour", "camera", "run", "run_camera"]}, {"type": "null"}]},
            "operator": {"anyOf": [{"type": "string", "enum": [">", "<", "="]}, {"type": "null"}]},
            "show_evidence": {"type": "boolean"},
            "context_reference": {"anyOf": [{"type": "string", "enum": ["previous_result", "previous_filters"]}, {"type": "null"}]},
        },
    }


def _system_prompt() -> str:
    return """Convert the user's traffic-video question into the supplied JSON schema.
Return exactly one compact JSON object on a single line.

Never answer the question. Never provide counts. Never invent values.
Only populate fields clearly required by the user. Leave unrelated fields empty or null.

Normalize synonyms:
- bike, bikes, motorbike, two-wheeler, two-wheelers -> MOTORCYCLE
- auto, autos, auto-rickshaw, rickshaw, three wheeler -> 3WHEELER
- unknown vehicle, unknown vehicles, unclassified vehicle -> UNKNOWN
- gray -> GREY

Rules:
- greetings, thanks, who are you, what can you do -> GENERAL_CHAT with no filters and no context_reference
- default subject is vehicles unless the user is explicitly asking about runs
- run questions must use subject runs
- use run_filter multiple_cameras only for questions about runs with multiple cameras
- how many -> COUNT
- show, show me, find, which ones, let me see -> LIST and show_evidence true
- overview, summary -> SUMMARY
- class-wise, vehicle category counts, vehicle type counts, breakdown by class -> GROUP with group_by vehicle_class and no filters unless explicitly mentioned
- colour-wise, vehicle colour counts, breakdown by colour -> GROUP with group_by colour and no filters unless explicitly mentioned
- camera-wise, by camera, per camera, camera breakdown, in each camera -> GROUP with group_by camera
- run-wise, by run, per run, run breakdown, in each run -> GROUP with group_by run
- by run and camera, per run and camera, compare cameras across runs -> GROUP with group_by run_camera
- with detected number plates -> plate_presence detected, plate_detected true
- with readable number plates -> plate_presence readable, plate_detected true, plate_readable true
- without readable number plates -> plate_detected true if detection is implied, plate_readable false
- without number plates / no number plates -> plate_detected false, plate_readable false
- exact plate lookup -> set plate_text and also plate_detected true plus plate_readable true
- colours of motorcycles/cars/three-wheelers/bikes -> GROUP with group_by colour and preserve the named class
- vehicle classes/types were black/white/red/etc -> GROUP with group_by vehicle_class and preserve the named colour
- what kinds/types -> GROUP with group_by vehicle_class
- what colour / most common colour -> GROUP with group_by colour
- more common than / more than -> COMPARE with classes in the compared order
- at what time / when / during which period one class is more than another -> FIND_INTERVALS with classes in compared order and operator >, <, or =
- except / other than / apart from / excluding / exclude / without / not / not including / anything but / everything but / all but / but not -> put the mentioned class or colour in the matching exclude field
- those, them, these, ones -> context_reference previous_result
- Do not reference previous context unless the current message explicitly says those, them, these, previous, remaining, other ones, or of those.
- If the user asks a complete standalone analytics question, set context_reference null.

Examples:
User: hello
Output: {"intent":"GENERAL_CHAT","subject":"vehicles","run_filter":null,"class_include":[],"class_exclude":[],"colour_include":[],"colour_exclude":[],"plate_presence":null,"plate_detected":null,"plate_readable":null,"plate_text":null,"start_time":null,"end_time":null,"group_by":null,"operator":null,"show_evidence":false,"context_reference":null}

User: How many cars are there?
Output: {"intent":"COUNT","subject":"vehicles","run_filter":null,"class_include":["CAR"],"class_exclude":[],"colour_include":[],"colour_exclude":[],"plate_presence":null,"plate_detected":null,"plate_readable":null,"plate_text":null,"start_time":null,"end_time":null,"group_by":null,"operator":null,"show_evidence":false,"context_reference":null}

User: cars with detected number plates
Output: {"intent":"LIST","subject":"vehicles","run_filter":null,"class_include":["CAR"],"class_exclude":[],"colour_include":[],"colour_exclude":[],"plate_presence":"detected","plate_detected":true,"plate_readable":null,"plate_text":null,"start_time":null,"end_time":null,"group_by":null,"operator":null,"show_evidence":true,"context_reference":null}

User: cars without readable number plates
Output: {"intent":"LIST","subject":"vehicles","run_filter":null,"class_include":["CAR"],"class_exclude":[],"colour_include":[],"colour_exclude":[],"plate_presence":null,"plate_detected":true,"plate_readable":false,"plate_text":null,"start_time":null,"end_time":null,"group_by":null,"operator":null,"show_evidence":true,"context_reference":null}

User: show plate DL01AB1234
Output: {"intent":"LIST","subject":"vehicles","run_filter":null,"class_include":[],"class_exclude":[],"colour_include":[],"colour_exclude":[],"plate_presence":"readable","plate_detected":true,"plate_readable":true,"plate_text":"DL01AB1234","start_time":null,"end_time":null,"group_by":null,"operator":null,"show_evidence":true,"context_reference":null}

User: which runs have multiple cameras
Output: {"intent":"LIST","subject":"runs","run_filter":"multiple_cameras","class_include":[],"class_exclude":[],"colour_include":[],"colour_exclude":[],"plate_presence":null,"plate_detected":null,"plate_readable":null,"plate_text":null,"start_time":null,"end_time":null,"group_by":null,"operator":null,"show_evidence":false,"context_reference":null}

User: how many runs are there in this search
Output: {"intent":"COUNT","subject":"runs","run_filter":null,"class_include":[],"class_exclude":[],"colour_include":[],"colour_exclude":[],"plate_presence":null,"plate_detected":null,"plate_readable":null,"plate_text":null,"start_time":null,"end_time":null,"group_by":null,"operator":null,"show_evidence":false,"context_reference":null}

User: give me the summary run and camera wise
Output: {"intent":"SUMMARY","subject":"vehicles","run_filter":null,"class_include":[],"class_exclude":[],"colour_include":[],"colour_exclude":[],"plate_presence":null,"plate_detected":null,"plate_readable":null,"plate_text":null,"start_time":null,"end_time":null,"group_by":"run_camera","operator":null,"show_evidence":false,"context_reference":null}

User: Show black vehicles except motorcycles.
Output: {"intent":"LIST","subject":"vehicles","run_filter":null,"class_include":[],"class_exclude":["MOTORCYCLE"],"colour_include":["BLACK"],"colour_exclude":[],"plate_presence":null,"plate_detected":null,"plate_readable":null,"plate_text":null,"start_time":null,"end_time":null,"group_by":null,"operator":null,"show_evidence":true,"context_reference":null}

User: Which of those were black?
Output: {"intent":"LIST","subject":"vehicles","run_filter":null,"class_include":[],"class_exclude":[],"colour_include":["BLACK"],"colour_exclude":[],"plate_presence":null,"plate_detected":null,"plate_readable":null,"plate_text":null,"start_time":null,"end_time":null,"group_by":null,"operator":null,"show_evidence":false,"context_reference":"previous_result"}

Return only JSON."""


def _video_chat_env() -> dict[str, str]:
    values: dict[str, str] = {}
    env_path = Path(".env")
    skip_dotenv = bool(os.getenv("PYTEST_CURRENT_TEST")) and "VIDEO_CHAT_LLM_PROVIDER" not in os.environ
    if env_path.exists() and not skip_dotenv:
        for line in env_path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key, value = stripped.split("=", 1)
            key = key.strip()
            if key.startswith("VIDEO_CHAT_"):
                values[key] = value.strip()
    for key, value in os.environ.items():
        if key.startswith("VIDEO_CHAT_"):
            values[key] = value
    return values


def _env_first(values: dict[str, str], keys: list[str], default: str) -> str:
    for key in keys:
        value = values.get(key)
        if value is not None and str(value).strip():
            return str(value).strip().strip('"')
    return default


def _duration_ms(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return round(float(value) / 1_000_000, 3)
    except (TypeError, ValueError):
        return None
