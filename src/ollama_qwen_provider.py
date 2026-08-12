from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .vehicle_enrichment.taxonomy import SUPPORTED_VEHICLE_CLASSES, SUPPORTED_VEHICLE_COLOUR_LABELS


DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"
DEFAULT_OLLAMA_MODEL = "qwen3:1.7b"
DEFAULT_OLLAMA_KEEP_ALIVE = "10m"
DEFAULT_OLLAMA_TIMEOUT_SECONDS = 8.0


class OllamaQwenChatLLMProvider:
    def __init__(
        self,
        *,
        base_url: str = DEFAULT_OLLAMA_URL,
        model: str = DEFAULT_OLLAMA_MODEL,
        keep_alive: str = DEFAULT_OLLAMA_KEEP_ALIVE,
        timeout_seconds: float = DEFAULT_OLLAMA_TIMEOUT_SECONDS,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.keep_alive = keep_alive
        self.timeout_seconds = float(timeout_seconds)

    def parse(self, message: str, context: dict[str, Any]) -> dict[str, Any]:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": _system_prompt()},
                {"role": "user", "content": json.dumps({"message": message, "context": context}, ensure_ascii=True)},
            ],
            "stream": False,
            "think": False,
            "format": chat_vehicle_query_json_schema(),
            "options": {"temperature": 0},
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
        except (HTTPError, URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Ollama chat request failed: {exc}") from exc
        content = str(dict(raw.get("message") or {}).get("content") or "").strip()
        if not content:
            raise RuntimeError("Ollama response did not include message.content")
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError as exc:
            raise RuntimeError("Ollama message.content was not valid JSON") from exc
        if not isinstance(parsed, dict):
            raise RuntimeError("Ollama structured output was not a JSON object")
        return parsed


def build_chat_llm_provider_from_env() -> OllamaQwenChatLLMProvider | None:
    env = _video_chat_env()
    provider = env.get("VIDEO_CHAT_LLM_PROVIDER", "").strip().strip('"').lower()
    if provider not in {"ollama", "qwen", "qwen3"}:
        return None
    return OllamaQwenChatLLMProvider(
        base_url=env.get("VIDEO_CHAT_OLLAMA_URL", DEFAULT_OLLAMA_URL).strip('"'),
        model=env.get("VIDEO_CHAT_OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL).strip('"'),
        keep_alive=env.get("VIDEO_CHAT_OLLAMA_KEEP_ALIVE", DEFAULT_OLLAMA_KEEP_ALIVE).strip('"'),
        timeout_seconds=float(env.get("VIDEO_CHAT_OLLAMA_TIMEOUT_SECONDS", str(DEFAULT_OLLAMA_TIMEOUT_SECONDS)).strip('"')),
    )


def chat_vehicle_query_json_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "intent",
            "class_include",
            "class_exclude",
            "colour_include",
            "colour_exclude",
            "start_time",
            "end_time",
            "group_by",
            "operator",
            "show_evidence",
            "context_reference",
        ],
        "properties": {
            "intent": {"type": "string", "enum": ["GENERAL_CHAT", "COUNT", "LIST", "SUMMARY", "GROUP", "COMPARE", "FIND_INTERVALS", "UNIQUE_CLASSES", "UNIQUE_COLOURS"]},
            "class_include": {"type": "array", "items": {"type": "string", "enum": [*SUPPORTED_VEHICLE_CLASSES, "UNKNOWN"]}},
            "class_exclude": {"type": "array", "items": {"type": "string", "enum": [*SUPPORTED_VEHICLE_CLASSES, "UNKNOWN"]}},
            "colour_include": {"type": "array", "items": {"type": "string", "enum": list(SUPPORTED_VEHICLE_COLOUR_LABELS)}},
            "colour_exclude": {"type": "array", "items": {"type": "string", "enum": list(SUPPORTED_VEHICLE_COLOUR_LABELS)}},
            "start_time": {"anyOf": [{"type": "number"}, {"type": "null"}]},
            "end_time": {"anyOf": [{"type": "number"}, {"type": "null"}]},
            "group_by": {"anyOf": [{"type": "string", "enum": ["vehicle_class", "colour"]}, {"type": "null"}]},
            "operator": {"anyOf": [{"type": "string", "enum": [">", "<", "="]}, {"type": "null"}]},
            "show_evidence": {"type": "boolean"},
            "context_reference": {"anyOf": [{"type": "string", "enum": ["previous_result", "previous_filters"]}, {"type": "null"}]},
        },
    }


def _system_prompt() -> str:
    return """Convert the user's traffic-video question into the supplied JSON schema.

Never answer the question. Never provide counts. Never invent values.
Only populate fields clearly required by the user. Leave unrelated fields empty or null.

Normalize synonyms:
- bike, bikes, motorbike, two-wheeler, two-wheelers -> MOTORCYCLE
- auto, autos, auto-rickshaw, rickshaw, three wheeler -> 3WHEELER
- unknown vehicle, unknown vehicles, unclassified vehicle -> UNKNOWN
- gray -> GREY

Rules:
- greetings, thanks, who are you, what can you do -> GENERAL_CHAT with no filters and no context_reference
- how many -> COUNT
- show, show me, find, which ones, let me see -> LIST and show_evidence true
- overview, summary -> SUMMARY
- class-wise, vehicle category counts, vehicle type counts, breakdown by class -> GROUP with group_by vehicle_class and no filters unless explicitly mentioned
- colour-wise, vehicle colour counts, breakdown by colour -> GROUP with group_by colour and no filters unless explicitly mentioned
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
Output: {"intent":"GENERAL_CHAT","class_include":[],"class_exclude":[],"colour_include":[],"colour_exclude":[],"start_time":null,"end_time":null,"group_by":null,"operator":null,"show_evidence":false,"context_reference":null}

User: How many cars are there?
Output: {"intent":"COUNT","class_include":["CAR"],"class_exclude":[],"colour_include":[],"colour_exclude":[],"start_time":null,"end_time":null,"group_by":null,"operator":null,"show_evidence":false,"context_reference":null}

User: How many cars and bikes did we see?
Output: {"intent":"COUNT","class_include":["CAR","MOTORCYCLE"],"class_exclude":[],"colour_include":[],"colour_exclude":[],"start_time":null,"end_time":null,"group_by":null,"operator":null,"show_evidence":false,"context_reference":null}

User: Apart from bikes, what kinds of vehicles were there?
Output: {"intent":"GROUP","class_include":[],"class_exclude":["MOTORCYCLE"],"colour_include":[],"colour_exclude":[],"start_time":null,"end_time":null,"group_by":"vehicle_class","operator":null,"show_evidence":false,"context_reference":null}

User: Show black vehicles except motorcycles.
Output: {"intent":"LIST","class_include":[],"class_exclude":["MOTORCYCLE"],"colour_include":["BLACK"],"colour_exclude":[],"start_time":null,"end_time":null,"group_by":null,"operator":null,"show_evidence":true,"context_reference":null}

User: I want to see the vehicles other than car and bike.
Output: {"intent":"LIST","class_include":[],"class_exclude":["CAR","MOTORCYCLE"],"colour_include":[],"colour_exclude":[],"start_time":null,"end_time":null,"group_by":null,"operator":null,"show_evidence":true,"context_reference":null}

User: What vehicle colours are present except black?
Output: {"intent":"GROUP","class_include":[],"class_exclude":[],"colour_include":[],"colour_exclude":["BLACK"],"start_time":null,"end_time":null,"group_by":"colour","operator":null,"show_evidence":false,"context_reference":null}

Previous context has black motorcycle results. User: Give all vehicles class wise.
Output: {"intent":"GROUP","class_include":[],"class_exclude":[],"colour_include":[],"colour_exclude":[],"start_time":null,"end_time":null,"group_by":"vehicle_class","operator":null,"show_evidence":false,"context_reference":null}

User: Show colour-wise counts.
Output: {"intent":"GROUP","class_include":[],"class_exclude":[],"colour_include":[],"colour_exclude":[],"start_time":null,"end_time":null,"group_by":"colour","operator":null,"show_evidence":false,"context_reference":null}

User: Give me the colours of motorcycles.
Output: {"intent":"GROUP","class_include":["MOTORCYCLE"],"class_exclude":[],"colour_include":[],"colour_exclude":[],"start_time":null,"end_time":null,"group_by":"colour","operator":null,"show_evidence":false,"context_reference":null}

User: What vehicle classes were black?
Output: {"intent":"GROUP","class_include":[],"class_exclude":[],"colour_include":["BLACK"],"colour_exclude":[],"start_time":null,"end_time":null,"group_by":"vehicle_class","operator":null,"show_evidence":false,"context_reference":null}

User: Were two-wheelers more common than cars?
Output: {"intent":"COMPARE","class_include":["MOTORCYCLE","CAR"],"class_exclude":[],"colour_include":[],"colour_exclude":[],"start_time":null,"end_time":null,"group_by":null,"operator":null,"show_evidence":false,"context_reference":null}

User: At what time were bikes more than cars?
Output: {"intent":"FIND_INTERVALS","class_include":["MOTORCYCLE","CAR"],"class_exclude":[],"colour_include":[],"colour_exclude":[],"start_time":null,"end_time":null,"group_by":null,"operator":">","show_evidence":false,"context_reference":null}

User: Show me the green autos.
Output: {"intent":"LIST","class_include":["3WHEELER"],"class_exclude":[],"colour_include":["GREEN"],"colour_exclude":[],"start_time":null,"end_time":null,"group_by":null,"operator":null,"show_evidence":true,"context_reference":null}

User: Show unknown vehicles.
Output: {"intent":"LIST","class_include":["UNKNOWN"],"class_exclude":[],"colour_include":[],"colour_exclude":[],"start_time":null,"end_time":null,"group_by":null,"operator":null,"show_evidence":true,"context_reference":null}

User: What colour were most of the cars?
Output: {"intent":"GROUP","class_include":["CAR"],"class_exclude":[],"colour_include":[],"colour_exclude":[],"start_time":null,"end_time":null,"group_by":"colour","operator":null,"show_evidence":false,"context_reference":null}

User: Give me a quick overview of this traffic video.
Output: {"intent":"SUMMARY","class_include":[],"class_exclude":[],"colour_include":[],"colour_exclude":[],"start_time":null,"end_time":null,"group_by":null,"operator":null,"show_evidence":false,"context_reference":null}

Previous context has MOTORCYCLE results. User: Which of those were black?
Output: {"intent":"LIST","class_include":[],"class_exclude":[],"colour_include":["BLACK"],"colour_exclude":[],"start_time":null,"end_time":null,"group_by":null,"operator":null,"show_evidence":false,"context_reference":"previous_result"}

Previous context has black motorcycle results. User: Show them.
Output: {"intent":"LIST","class_include":[],"class_exclude":[],"colour_include":[],"colour_exclude":[],"start_time":null,"end_time":null,"group_by":null,"operator":null,"show_evidence":true,"context_reference":"previous_result"}

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
