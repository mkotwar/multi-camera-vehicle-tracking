from __future__ import annotations

import json
import os
from pathlib import Path
import time
from typing import Any
from urllib.parse import urlparse
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .video_chat_plan import VIDEO_CHAT_CAPABILITY_CATALOGUE, analytics_plan_json_schema


DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"
DEFAULT_OLLAMA_MODEL = "qwen3:1.7b"
DEFAULT_OLLAMA_KEEP_ALIVE = "10m"
DEFAULT_OLLAMA_TIMEOUT_SECONDS = 45.0
DEFAULT_OLLAMA_TEMPERATURE = 0.0
DEFAULT_OLLAMA_NUM_PREDICT = 384
MIN_OLLAMA_NUM_PREDICT_FOR_ANALYTICS_PLAN = 384


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
        retry_feedback = dict(context.get("planner_retry") or {}) if isinstance(context, dict) else {}
        effective_num_predict = max(self.num_predict, MIN_OLLAMA_NUM_PREDICT_FOR_ANALYTICS_PLAN)
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": _system_prompt(retry_feedback=retry_feedback)},
                {"role": "user", "content": json.dumps({"message": message, "context": context, "capabilities": VIDEO_CHAT_CAPABILITY_CATALOGUE.to_dict()}, ensure_ascii=True)},
            ],
            "stream": False,
            "think": False,
            "format": analytics_plan_json_schema(),
            "options": {
                "temperature": self.temperature,
                "num_predict": effective_num_predict,
            },
            "keep_alive": self.keep_alive,
        }
        request = Request(
            f"{self.base_url}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        raw: dict[str, Any] | None = None
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                raw = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            raise RuntimeError(f"Ollama chat request failed: {_safe_ollama_error_details(self, exc, started, http_status=exc.code)}") from exc
        except (URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Ollama chat request failed: {_safe_ollama_error_details(self, exc, started)}") from exc
        message_payload = dict(raw.get("message") or {})
        content_payload = message_payload.get("content")
        self.last_metadata = _metadata_from_raw(self, raw, started, effective_num_predict=effective_num_predict)
        self.last_metadata["response_preview"] = _safe_preview(content_payload)
        self.last_metadata["message_thinking_preview"] = _safe_preview(message_payload.get("thinking"))
        if isinstance(content_payload, dict):
            return content_payload
        content = _normalize_transport_content(content_payload)
        if not content:
            raise RuntimeError(_provider_failure_message("qwen_empty_content", raw=raw))
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError as exc:
            failure_reason = "qwen_output_truncated" if str(raw.get("done_reason")) == "length" else "qwen_invalid_json"
            raise RuntimeError(_provider_failure_message(failure_reason, raw=raw, content=content, parse_error=exc)) from exc
        if not isinstance(parsed, dict):
            raise RuntimeError(_provider_failure_message("qwen_invalid_json", raw=raw, content=content))
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
    return analytics_plan_json_schema()


def _system_prompt(*, retry_feedback: dict[str, Any] | None = None) -> str:
    retry_text = ""
    if retry_feedback:
        retry_text = f"""

Retry guidance:
- The previous AnalyticsPlan was invalid.
- Original query: {json.dumps(retry_feedback.get("original_query"), ensure_ascii=True)}
- Previous invalid plan: {json.dumps(retry_feedback.get("previous_plan"), ensure_ascii=True)}
- Validation errors: {json.dumps(retry_feedback.get("validation_errors"), ensure_ascii=True)}
- Return a corrected AnalyticsPlan JSON object only.
"""
    return f"""Convert the user's traffic-video analytics question into the supplied AnalyticsPlan JSON schema.
Return exactly one compact JSON object on a single line.

AnalyticsPlan is the canonical semantic representation.
Never answer the question. Never provide counts. Never invent facts or unsupported fields.

Capability summary:
- Entity: vehicle
- Groupable fields: class, colour, camera, run, run_camera, plate_presence, time_bucket
- Filterable fields: class, colour, camera, run, plate_text, plate_detected, plate_readable, plate_presence, start_time, end_time, vehicle_id
- Operators: eq, neq, in, not_in, starts_with, ends_with, contains, exists, not_exists, gt, gte, lt, lte, between
- Metrics: vehicle_count, count_distinct, difference, ratio, percentage
- Result shapes: scalar, list, grouped, ranking, comparison, summary, plate_lookup, unsupported_capability

Planning rules:
- Use entity=vehicle.
- Omit optional fields that are not needed for the user's request. Do not invent values just to fill the schema.
- For simple counts, use result_shape scalar and metric vehicle_count.
- For simple counts like "how many cars are there", include the class filter only. Do not add group_by, order_by, comparison, time, or context_reference unless the user explicitly asked for them.
- For list/show/find queries, use result_shape list and show_evidence true when the user wants matching vehicles or evidence.
- For grouped breakdowns, use result_shape grouped, group_by, and metric vehicle_count.
- For rankings, use result_shape ranking, group_by, metric vehicle_count, order_by on metric, and limit.
- For direct comparisons, use result_shape comparison and comparison.operation winner or difference.
- Do not emit comparison unless the user explicitly asks to compare two things.
- Do not emit group_by or order_by unless the user explicitly asks for a breakdown, top/bottom ranking, or grouped answer.
- Do not emit time unless the user asked for a time range or interval behavior.
- Do not emit context_reference or context_resolution unless the user explicitly refers to previous results.
- Set show_evidence=true only when the user asks to show, list, find, display, or inspect matching vehicles/evidence.
- Preserve explicit classes, colours, cameras, runs, time ranges, and plate constraints.
- Use class values like CAR, MOTORCYCLE, BUS, TRUCK, 3WHEELER, UNKNOWN.
- Use colour values like BLACK, WHITE, RED, BLUE, GREY, SILVER, GREEN when present.
- Normalize bike, bikes, motorbike, two wheeler -> MOTORCYCLE.
- Normalize auto, auto-rickshaw, rickshaw -> 3WHEELER.
- Normalize gray -> GREY.
- Exact plate search uses field plate_text with operator eq and a full canonical plate.
- Prefix/suffix/contains plate search uses plate_text with starts_with, ends_with, or contains.
- Readable plates use plate_readable eq true.
- Detected plates use plate_detected eq true.
- If the message explicitly refers to previous results using those, them, these, ones, set context_reference to previous_results.
- Do not invent context_reference for standalone analytics questions.
- Across multiple selected runs, camera grouping should use run_camera unless the user explicitly asks for cross-run aggregation by same camera name.

Representative examples:
- "how many cars are there"
  -> filters class=CAR, result_shape scalar, metric vehicle_count, no group_by, no comparison, no order_by, show_evidence false
- "show white cars"
  -> filters class=CAR and colour=WHITE, result_shape list, show_evidence true
- "which vehicle class has the most vehicles"
  -> group_by class, metric vehicle_count, order_by metric desc, limit 1, result_shape ranking
- "top 3 colours"
  -> group_by colour, metric vehicle_count, order_by metric desc, limit 3, result_shape ranking
- "are there more cars or motorcycles"
  -> comparison winner between class=CAR and class=MOTORCYCLE, result_shape comparison
- "find UP84AT5908"
  -> filter plate_text eq UP84AT5908, result_shape list, show_evidence true
- "find all HR number plates"
  -> filter plate_text starts_with HR, result_shape list, show_evidence true
- "find plates ending with 62"
  -> filter plate_text ends_with 62, result_shape list, show_evidence true
- "which colour is most common among motorcycles"
  -> filter class=MOTORCYCLE, group_by colour, metric vehicle_count, order_by metric desc, limit 1, result_shape ranking

Unsupported data:
- If the user asks for unavailable data such as driver's name or vehicle owner, use result_shape unsupported_capability.

{retry_text}

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


def _metadata_from_raw(
    provider: OllamaQwenChatLLMProvider,
    raw: dict[str, Any],
    started: float,
    *,
    effective_num_predict: int,
) -> dict[str, Any]:
    return {
        "parser": "qwen",
        "provider": "ollama",
        "model": provider.model,
        "base_url": provider.base_url,
        "keep_alive": provider.keep_alive,
        "timeout_seconds": provider.timeout_seconds,
        "temperature": provider.temperature,
        "configured_num_predict": provider.num_predict,
        "num_predict": effective_num_predict,
        "think": False,
        "structured_output": True,
        "wall_time_ms": round((time.perf_counter() - started) * 1000, 3),
        "total_duration_ms": _duration_ms(raw.get("total_duration")),
        "load_duration_ms": _duration_ms(raw.get("load_duration")),
        "prompt_eval_duration_ms": _duration_ms(raw.get("prompt_eval_duration")),
        "eval_duration_ms": _duration_ms(raw.get("eval_duration")),
        "prompt_eval_count": raw.get("prompt_eval_count"),
        "eval_count": raw.get("eval_count"),
        "done": raw.get("done"),
        "done_reason": raw.get("done_reason"),
    }


def _safe_preview(value: Any, *, limit: int = 400) -> str | None:
    if value is None:
        return None
    if isinstance(value, (dict, list)):
        text = json.dumps(value, ensure_ascii=True)
    else:
        text = str(value)
    return text[:limit]


def _normalize_transport_content(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.startswith("```json") and text.endswith("```"):
        inner = text[len("```json") : -3]
        return inner.strip()
    if text.startswith("```") and text.endswith("```"):
        inner = text[3:-3]
        return inner.strip()
    return text


def _provider_failure_message(
    reason: str,
    *,
    raw: dict[str, Any],
    content: str | None = None,
    parse_error: Exception | None = None,
) -> str:
    parts = [
        f"reason={reason}",
        f"done={raw.get('done')}",
        f"done_reason={raw.get('done_reason')}",
        f"eval_count={raw.get('eval_count')}",
        f"prompt_eval_count={raw.get('prompt_eval_count')}",
        f"content_preview={json.dumps(_safe_preview(content if content is not None else dict(raw.get('message') or {}).get('content')), ensure_ascii=True)}",
    ]
    if parse_error is not None:
        parts.append(f"parse_error={parse_error}")
    return "Ollama structured output invalid: " + " ".join(parts)
