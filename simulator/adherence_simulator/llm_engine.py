"""LLM backends for the adherence simulator."""

from __future__ import annotations

import abc
import hashlib
import json
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable

from dotenv import load_dotenv

from .config import SimulatorConfig
from .models import StepResponse

logger = logging.getLogger(__name__)

_SCHEMA_PATH = Path(__file__).resolve().parent.parent / "schemas" / "step_response.schema.json"
NEBULA_BASE_URL = "https://nebula.cs.vu.nl/api/"
OPENAI_BASE_URL = "https://api.openai.com/v1/"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"

# DeepSeek pricing per 1M tokens: (cache_hit_input, cache_miss_input, output).
# Pro values use the current 75%-off promo rate. Cache-hit input is ~50-120x
# cheaper than cache-miss, so realized cost depends heavily on the hit rate
# (DeepSeek reports prompt_cache_hit_tokens / prompt_cache_miss_tokens).
DEEPSEEK_PRICING = {
    "deepseek-v4-pro": (0.003625, 0.435, 0.87),
    "deepseek-v4-flash": (0.0028, 0.14, 0.28),
    "deepseek-reasoner": (0.003625, 0.435, 0.87),
    "deepseek-chat": (0.0028, 0.14, 0.28),
}

# USD per 1M tokens (input, output). Output includes reasoning tokens for
# reasoning models. Used only for the display cost estimate; update as needed.
# Note: OpenAI auto-caches prompt prefixes and bills them at ~10% input rate,
# so realized input cost is typically below this uncached estimate.
OPENAI_PRICING = {
    "gpt-5.5": (5.0, 30.0),
    "gpt-5.4-mini": (0.75, 4.5),
    "gpt-5.4": (2.5, 15.0),
    "gpt-5-mini": (0.25, 2.0),
    "gpt-5-nano": (0.05, 0.40),
    "gpt-5": (1.25, 10.0),
    "gpt-4.1": (2.0, 8.0),
    "gpt-4o": (2.5, 10.0),
    "o3": (2.0, 8.0),
    "o4-mini": (1.1, 4.4),
}

# Exception/message patterns that mean "the server is temporarily down" rather
# than "the request was bad". Used to decide whether to keep retrying during a
# backend outage (e.g. Nebula 2-3am maintenance) instead of failing fast.
_TRANSIENT_EXC_NAMES = {
    "ConnectionError",
    "ConnectTimeout",
    "ReadTimeout",
    "Timeout",
    "ConnectionResetError",
    "RemoteDisconnected",
    "APIConnectionError",
    "APITimeoutError",
    "InternalServerError",
    "ServiceUnavailableError",
    "BadGatewayError",
    "GatewayTimeoutError",
    "RateLimitError",
    "ProtocolError",
    "IncompleteRead",
}
_TRANSIENT_MSG_FRAGMENTS = (
    "timed out",
    "timeout",
    "connection reset",
    "connection refused",
    "connection aborted",
    "remote end closed",
    "temporarily unavailable",
    "service unavailable",
    "bad gateway",
    "gateway timeout",
    "internal server error",
    " 502",
    " 503",
    " 504",
    " 408",
    " 429",
    # Nebula maintenance fingerprints: when the inference route is briefly
    # removed, the front proxy returns a plain 404 / 502 mix rather than a
    # clean 503. Observed 2026-04-24 02:00 window.
    "404 page not found",
    "no healthy upstream",
    "upstream connect error",
)

# Extra signatures only honoured once outage-retry mode has already been
# triggered. The intent is stickiness: if we've already seen a clean transient
# error (e.g. 502), subsequent calls on the same outage window may return
# different-looking failures (404 routing, connection-refused from a specific
# replica). Widening the transient filter mid-outage avoids falsely exiting
# the 30-min budget because Nebula's failure mode shifted.
_OUTAGE_STICKY_FRAGMENTS = (
    "not found",
    "no route",
    "no backend",
    "backend not found",
)
_OUTAGE_STICKY_STATUS_CODES = frozenset({404, 502, 503, 504, 408, 429})


def _is_transient_failure(exc: BaseException, *, sticky: bool = False) -> bool:
    """Heuristically detect a transient backend failure worth retrying long.

    ``sticky=True`` is used once outage-retry mode is already active. In that
    mode we widen the net (404, "not found" fragments, any 5xx): during a
    Nebula maintenance window the backend often cycles through several
    different error shapes as proxies come up and down, and we don't want to
    bail out of the 30-min budget just because the signature changed.
    """
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if type(current).__name__ in _TRANSIENT_EXC_NAMES:
            return True
        status_code = getattr(current, "status_code", None)
        if status_code is None:
            response = getattr(current, "response", None)
            status_code = getattr(response, "status_code", None) if response is not None else None
        if isinstance(status_code, int):
            if status_code >= 500 or status_code in (408, 429):
                return True
            if sticky and status_code in _OUTAGE_STICKY_STATUS_CODES:
                return True
        msg = str(current).lower()
        if any(fragment in msg for fragment in _TRANSIENT_MSG_FRAGMENTS):
            return True
        if sticky and any(fragment in msg for fragment in _OUTAGE_STICKY_FRAGMENTS):
            return True
        current = current.__cause__ or current.__context__
    return False


def _load_response_schema() -> dict[str, Any]:
    with open(_SCHEMA_PATH, encoding="utf-8") as handle:
        return json.load(handle)


_RESPONSE_SCHEMA = _load_response_schema()


def _make_lite_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Strip ``reason_code`` from the schema to reduce output tokens."""
    lite = json.loads(json.dumps(schema))
    lite["properties"].pop("reason_code", None)
    if "reason_code" in lite.get("required", []):
        lite["required"] = [field for field in lite["required"] if field != "reason_code"]
    return lite


_RESPONSE_SCHEMA_LITE = _make_lite_schema(_RESPONSE_SCHEMA)


def _make_lookahead_schema(schema: dict[str, Any], lookahead_window: int) -> dict[str, Any]:
    """Wrap the per-day response schema in a fixed-length JSON array when needed."""
    window = max(1, int(lookahead_window or 1))
    schema_copy = json.loads(json.dumps(schema))
    if window <= 1:
        return schema_copy
    return {
        "type": "array",
        "minItems": window,
        "maxItems": window,
        "items": schema_copy,
    }


def _supports_reasoning_effort_param(model: str) -> bool:
    """Return whether the OpenAI-compatible backend supports ``reasoning_effort``.

    Nebula sits behind a LiteLLM/OpenAI-compatible server. In practice,
    ``reasoning_effort`` is accepted for the gpt-oss family, while Qwen-family
    models reject it with a 400. Keep this conservative and only opt in where
    we've seen it work.
    """
    normalized = model.strip().lower()
    return "gpt-oss" in normalized


def _reasoning_effort_from_budget(*, thinking: bool, budget: int) -> str:
    if not thinking:
        return "low"
    if budget <= 256:
        return "low"
    if budget <= 2048:
        return "medium"
    return "high"


def _build_openai_reasoning_extra_body(model: str, *, thinking: bool, budget: int) -> dict[str, Any]:
    """Build backend-specific reasoning controls for OpenAI-compatible APIs."""
    extra_body: dict[str, Any] = {
        "chat_template_kwargs": {"enable_thinking": thinking},
    }
    if _supports_reasoning_effort_param(model):
        extra_body["reasoning_effort"] = _reasoning_effort_from_budget(
            thinking=thinking,
            budget=budget,
        )
    return extra_body


_VALID_REASON_CODES = {
    # Shared behavioral-outcome codes.
    "routine",
    "intervention_response",
    "side_effects",
    "low_motivation",
    "fatigue",
    "no_change",
    # Dataset-profile codes used by simulator.adherence_simulator.dataset_prompts.
    "context_favorable",
    "suggestion_response",
    "baseline_activity",
    "context_unfavorable",
    "in_transit",
    "sedentary_setting",
    "weather_barrier",
    "low_engagement",
    "time_slot_effect",
    "habit_walk",
    "routine_dosing",
    "regimen_confusion",
    "anticipated_exposure",
    "no_perceived_risk",
    "travel_disruption",
    "pill_fatigue",
    "stigma_concern",
    "ran_out",
    "study_engagement",
    "declining_motivation",
    "partner_disclosure_issue",
    "support_boost",
    "responded_to_suggestion",
    "context_match",
    "context_mismatch",
    "habituated",
    "disengaged",
    "no_suggestion_sent",
    "spontaneous_activity",
}
_DEFAULT_STATE_UPDATE = {
    "stress_level": None,
    "routine_disrupted": None,
    "motivation": None,
    "side_effects": None,
    "social_support_change": None,
    "life_event": None,
}


def _resolve_env_api_key(name: str, explicit_api_key: str | None = None) -> str | None:
    if explicit_api_key:
        return explicit_api_key
    load_dotenv()
    return os.getenv(name)


def _parse_response(text: str, lookahead_window: int = 1) -> StepResponse:
    """Parse and validate a JSON step response.

    When ``lookahead_window > 1`` the response is expected to be a JSON array
    of per-day objects; only the first element is consumed by the simulator.
    """
    try:
        return _parse_response_inner(text, lookahead_window=lookahead_window)
    except Exception as exc:
        preview = text or ""
        logger.error("Failed to parse LLM response (%s): %s", exc, preview)
        raise


def _parse_response_inner(text: str, lookahead_window: int = 1) -> StepResponse:
    if not text:
        raise ValueError("Empty response from LLM")

    import re as regex

    stripped = text.strip()
    stripped = regex.sub(r"<think>.*?</think>", "", stripped, flags=regex.DOTALL).strip()
    stripped = regex.sub(r"<\|[^|]*\|>", "", stripped).strip()
    if not stripped:
        raise ValueError("Empty response from LLM after stripping tags")

    if stripped.startswith("```"):
        lines = stripped.splitlines()
        stripped = "\n".join(lines[1:-1] if lines and lines[-1].strip() == "```" else lines[1:])

    stripped = regex.sub(r":\s*\+(\d)", r": \1", stripped)

    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        # Lookahead asks for an array; try array regex first, then fall back
        # to the object regex.
        if int(lookahead_window or 1) > 1:
            match = regex.search(r"\[.*\]", stripped, regex.DOTALL)
            if match:
                try:
                    parsed = json.loads(match.group())
                except json.JSONDecodeError:
                    parsed = None
            else:
                parsed = None
            if parsed is None:
                obj_match = regex.search(r"\{.*\}", stripped, regex.DOTALL)
                if not obj_match:
                    raise ValueError(f"No JSON found in response: {stripped[:200]}")
                parsed = json.loads(obj_match.group())
        else:
            match = regex.search(r"\{.*\}", stripped, regex.DOTALL)
            if not match:
                raise ValueError(f"No JSON object found in response: {stripped[:200]}")
            parsed = json.loads(match.group())

    # Lookahead generation: collapse to day 1 here so the rest of the engine
    # never sees the array. If the model returned an object even though we
    # asked for an array, keep parsing it directly (graceful fallback).
    if isinstance(parsed, list):
        if not parsed:
            raise ValueError("LLM returned empty JSON array for lookahead response")
        first = parsed[0]
        if not isinstance(first, dict):
            raise ValueError(
                f"Lookahead day 1 is not a JSON object: {type(first).__name__}"
            )
        parsed = first

    adherence = parsed.get("adherence")
    if adherence is None or not isinstance(adherence, (int, float)):
        raise ValueError(f"Missing or invalid adherence: {adherence}")

    certainty = parsed.get("adherence_certainty")
    if not isinstance(certainty, (int, float)):
        certainty = 0.5

    reason_code = parsed.get("reason_code", "no_change")
    if reason_code not in _VALID_REASON_CODES:
        logger.debug("Unknown reason_code '%s', defaulting to 'no_change'", reason_code)
        reason_code = "no_change"

    state_update = parsed.get("state_update")
    if not isinstance(state_update, dict):
        state_update = dict(_DEFAULT_STATE_UPDATE)
    else:
        state_update = {**_DEFAULT_STATE_UPDATE, **state_update}

    return StepResponse(
        adherence=max(0.0, min(1.0, float(adherence))),
        adherence_certainty=max(0.0, min(1.0, float(certainty))),
        reason_code=reason_code,
        state_update=state_update,
        activity_bin=parsed.get("activity_bin") if isinstance(parsed.get("activity_bin"), str) else None,
    )


class BaseEngine(abc.ABC):
    """Shared ensemble sampling, retry logic, and token tracking."""

    def __init__(self, config: SimulatorConfig):
        self.config = config
        self.model = config.model
        per_day_schema = _RESPONSE_SCHEMA_LITE if config.no_reasoning else _RESPONSE_SCHEMA
        self._schema = _make_lookahead_schema(per_day_schema, self._lookahead_window)
        self._system_instruction = ""
        self._call_count = 0
        self._prompt_token_count = 0
        self._output_token_count = 0
        self._thinking_token_count = 0
        self._retry_count = 0
        self._fail_count = 0
        self._lock = threading.Lock()

    def _record_tokens(self, prompt_tokens: int, output_tokens: int, thinking_tokens: int = 0) -> None:
        with self._lock:
            self._call_count += 1
            self._prompt_token_count += prompt_tokens
            self._output_token_count += output_tokens
            self._thinking_token_count += thinking_tokens

    @property
    def _lookahead_window(self) -> int:
        """Look at config lazily so the engines stay free of feature-specific args."""
        return max(1, int(getattr(self.config, "lookahead_window", 1) or 1))

    @abc.abstractmethod
    def _call_prompt_once(
        self,
        user_prompt: str,
        system_instruction: str,
        seed: int | None = None,
    ) -> StepResponse:
        """Make a single LLM call and return the parsed response."""

    def _call_with_retries(
        self,
        call_fn: Callable[[int | None], StepResponse],
        call_index: int,
        ensemble_size: int,
        seed: int | None = None,
    ) -> StepResponse | None:
        """Single ensemble call with retry logic.

        Two retry regimes:
          * Normal errors (bad JSON, schema violation, etc.): ``max_retries``
            attempts spaced by ``retry_delay``.
          * Transient backend errors (connection reset, 5xx, 429, timeout):
            exponential backoff capped at 60s between attempts, up to a total
            of ``outage_retry_seconds`` elapsed. This is what carries the
            simulator through a Nebula maintenance window instead of killing
            the whole run.
        """
        outage_deadline = (
            float(self.config.outage_retry_seconds)
            if getattr(self.config, "outage_retry_enabled", True)
            else 0.0
        )
        outage_elapsed = 0.0
        non_transient_attempts = 0
        attempt = 0
        outage_logged = False
        while True:
            try:
                retry_seed = (seed + attempt) if (seed is not None and attempt > 0) else seed
                result = call_fn(retry_seed)
                if outage_logged:
                    logger.info(
                        "LLM call %d/%d recovered after %.1fs of outage backoff.",
                        call_index + 1, ensemble_size, outage_elapsed,
                    )
                return result
            except Exception as exc:
                with self._lock:
                    self._retry_count += 1
                # Once we're already in outage-retry mode, widen the transient
                # filter: Nebula maintenance cycles through several failure
                # shapes (502 -> 404 -> connection-refused) as proxies bounce.
                transient = _is_transient_failure(exc, sticky=outage_logged)
                if transient and outage_elapsed < outage_deadline:
                    # Exponential backoff during a suspected backend outage.
                    step = min(outage_elapsed // 60, 6)  # ramp up, then cap
                    delay = min(60.0, float(self.config.retry_delay) * (2 ** step))
                    if not outage_logged:
                        logger.warning(
                            "LLM call %d/%d attempt %d hit transient backend error, "
                            "entering outage-retry mode (budget %.0fs): %s",
                            call_index + 1, ensemble_size, attempt + 1, outage_deadline, exc,
                        )
                        outage_logged = True
                    else:
                        logger.info(
                            "LLM call %d/%d still retrying (elapsed %.0fs / %.0fs, next in %.1fs): %s",
                            call_index + 1, ensemble_size, outage_elapsed, outage_deadline, delay, exc,
                        )
                    time.sleep(delay)
                    outage_elapsed += delay
                    attempt += 1
                    continue
                # Non-transient path: keep the short-retry budget.
                non_transient_attempts += 1
                logger.warning(
                    "LLM call %d/%d attempt %d failed: %s",
                    call_index + 1, ensemble_size, attempt + 1, exc,
                )
                if non_transient_attempts < self.config.max_retries:
                    time.sleep(self.config.retry_delay)
                    attempt += 1
                    continue
                break
        with self._lock:
            self._fail_count += 1
        logger.error(
            "LLM call %d/%d exhausted retries (outage elapsed %.0fs).",
            call_index + 1, ensemble_size, outage_elapsed,
        )
        return None

    def _sample_responses(
        self,
        call_fn: Callable[[int | None], StepResponse],
        *,
        k: int | None = None,
        base_seed: int | None = None,
    ) -> list[StepResponse]:
        """Run ensemble sampling for an arbitrary prompt-producing callable."""
        ensemble_size = k or self.config.ensemble_size
        seeds = [
            (base_seed + index) if base_seed is not None else None
            for index in range(ensemble_size)
        ]

        if ensemble_size == 1 or not self.config.parallel:
            results: list[StepResponse] = []
            for index in range(ensemble_size):
                response = self._call_with_retries(call_fn, index, ensemble_size, seeds[index])
                if response is not None:
                    results.append(response)
        else:
            results = []
            with ThreadPoolExecutor(max_workers=ensemble_size) as executor:
                futures = {
                    executor.submit(
                        self._call_with_retries,
                        call_fn,
                        index,
                        ensemble_size,
                        seeds[index],
                    ): index
                    for index in range(ensemble_size)
                }
                for future in as_completed(futures):
                    response = future.result()
                    if response is not None:
                        results.append(response)

        if not results:
            raise RuntimeError(f"All {ensemble_size} ensemble calls failed. Cannot proceed.")
        return results

    def sample_prompt(
        self,
        user_prompt: str,
        *,
        system_instruction: str | None = None,
        k: int | None = None,
        base_seed: int | None = None,
    ) -> list[StepResponse]:
        """Call the LLM ``k`` times for an arbitrary prompt and return parsed responses."""
        resolved_system_instruction = system_instruction or self._system_instruction
        return self._sample_responses(
            lambda seed: self._call_prompt_once(
                user_prompt,
                resolved_system_instruction,
                seed=seed,
            ),
            k=k,
            base_seed=base_seed,
        )

    @property
    def total_calls(self) -> int:
        return self._call_count

    @property
    def prompt_token_count(self) -> int:
        return self._prompt_token_count

    @property
    def output_token_count(self) -> int:
        return self._output_token_count

    @property
    def thinking_token_count(self) -> int:
        return self._thinking_token_count

    @property
    def estimated_cost_usd(self) -> float | None:
        return None


class NebulaEngine(BaseEngine):
    """OpenAI-compatible engine for VU Amsterdam's Nebula cluster."""

    def __init__(self, config: SimulatorConfig, api_key: str | None = None):
        super().__init__(config)
        from openai import OpenAI

        self._openai_cls = OpenAI
        self._api_key = _resolve_env_api_key("NEBULA_API_KEY", api_key)
        if not self._api_key:
            raise RuntimeError("NEBULA_API_KEY is not set. Pass --api-key or define it in the environment.")
        logger.info("Nebula engine ready (model=%s)", self.model)

    def _make_client(self):
        return self._openai_cls(base_url=NEBULA_BASE_URL, api_key=self._api_key)

    def _call_prompt_once(
        self,
        user_prompt: str,
        system_instruction: str,
        seed: int | None = None,
    ) -> StepResponse:
        budget = int(self.config.thinking_budget)
        thinking = budget != 0
        # no_reasoning forces the LLM to skip thinking entirely. Without this the
        # flag only trimmed the JSON schema; gpt-oss on Nebula kept reasoning anyway.
        if getattr(self.config, "no_reasoning", False):
            thinking = False

        # Qwen-family models on Nebula accept ``chat_template_kwargs`` but may
        # reject ``reasoning_effort`` with a hard 400. Only send that parameter
        # for model families that actually support it (currently gpt-oss).
        extra_body = _build_openai_reasoning_extra_body(
            self.model,
            thinking=thinking,
            budget=budget,
        )

        # Qwen3 via vLLM has no native thinking-budget param — only on/off.
        # Map the integer budget to a soft steering hint appended to the system
        # prompt. Buckets: low <=256, medium <=2048, else high (unconstrained).
        if thinking:
            if budget <= 256:
                hint = (
                    "Think briefly before answering: keep internal reasoning to "
                    "2-3 short sentences maximum, then produce the JSON answer."
                )
            elif budget <= 2048:
                hint = (
                    "Think concisely before answering: keep internal reasoning "
                    "under ~10 sentences, then produce the JSON answer."
                )
            else:
                hint = None
            if hint is not None:
                system_instruction = f"{system_instruction}\n\n{hint}"

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.config.temperature,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "adherence_response",
                    "schema": self._schema,
                    "strict": False,
                },
            },
            "extra_body": extra_body,
        }
        if seed is not None:
            kwargs["seed"] = seed
        if self.config.top_p is not None:
            kwargs["top_p"] = self.config.top_p

        client = self._make_client()
        response = client.chat.completions.create(**kwargs)

        usage = response.usage
        prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
        generation_tokens = getattr(usage, "completion_tokens", 0) or 0
        details = getattr(usage, "completion_tokens_details", None)
        thinking_tokens = int(getattr(details, "reasoning_tokens", 0) or 0)

        message = response.choices[0].message
        if thinking_tokens == 0:
            reasoning = getattr(message, "reasoning_content", None) or ""
            thinking_tokens = len(reasoning.split()) if reasoning else 0

        logger.debug(
            "Nebula call: prompt_tokens=%d, gen_tokens=%d, thinking_tokens=%d",
            prompt_tokens,
            generation_tokens,
            thinking_tokens,
        )
        self._record_tokens(prompt_tokens, generation_tokens, thinking_tokens)

        return _parse_response(message.content or "", lookahead_window=self._lookahead_window)


def _is_openai_reasoning_model(model: str) -> bool:
    """GPT-5 family and o-series are reasoning models with restricted params."""
    m = model.strip().lower()
    return m.startswith(("o1", "o3", "o4", "o5")) or "gpt-5" in m


class OpenAIEngine(BaseEngine):
    """Engine for OpenAI's official API (api.openai.com).

    Unlike Nebula (vLLM/LiteLLM behind an OpenAI-compatible shim), the
    real OpenAI API rejects ``chat_template_kwargs`` and, for reasoning models
    (gpt-5 family, o-series), rejects non-default ``temperature``/``top_p``.
    Reasoning is controlled with the top-level ``reasoning_effort`` parameter.
    """

    def __init__(self, config: SimulatorConfig, api_key: str | None = None):
        super().__init__(config)
        from openai import OpenAI

        self._openai_cls = OpenAI
        self._api_key = _resolve_env_api_key("OPENAI_API_KEY", api_key)
        if not self._api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Pass --api-key or define it in the environment."
            )
        self._is_reasoning = _is_openai_reasoning_model(self.model)
        # Running total of input tokens that were served from OpenAI's automatic
        # prompt cache (billed at ~10% of the input rate).
        self._cached_prompt_token_count = 0
        logger.info(
            "OpenAI engine ready (model=%s, reasoning=%s)", self.model, self._is_reasoning
        )

    def _make_client(self):
        return self._openai_cls(base_url=OPENAI_BASE_URL, api_key=self._api_key)

    def _reasoning_effort(self) -> str:
        budget = int(self.config.thinking_budget)
        thinking = budget != 0 and not getattr(self.config, "no_reasoning", False)
        m = self.model.strip().lower()
        if not thinking:
            # Lowest-cost reasoning setting; the accepted value differs by family:
            # gpt-5.1+ uses "none", original gpt-5 uses "minimal", o-series has
            # neither so floor at "low".
            if any(m.startswith(f"gpt-5.{d}") for d in "123456789"):
                return "none"
            if "gpt-5" in m:
                return "minimal"
            return "low"
        if budget <= 256:
            return "low"
        if budget <= 2048:
            return "medium"
        return "high"

    def _call_prompt_once(
        self,
        user_prompt: str,
        system_instruction: str,
        seed: int | None = None,
    ) -> StepResponse:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": user_prompt},
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "adherence_response",
                    "schema": self._schema,
                    "strict": False,
                },
            },
        }
        if self._is_reasoning:
            # Reasoning models reject temperature/top_p != default; use effort instead.
            kwargs["reasoning_effort"] = self._reasoning_effort()
        else:
            kwargs["temperature"] = self.config.temperature
            if self.config.top_p is not None:
                kwargs["top_p"] = self.config.top_p
        if seed is not None:
            kwargs["seed"] = seed
        # Improve prompt-cache hit rate: route all calls sharing the same static
        # system prompt to the same cache. OpenAI auto-caches prefixes >1024
        # tokens; this key just steers identical-prefix requests together.
        kwargs["prompt_cache_key"] = "hs-" + hashlib.sha256(
            system_instruction.encode("utf-8")
        ).hexdigest()[:16]

        client = self._make_client()
        response = client.chat.completions.create(**kwargs)

        usage = response.usage
        prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
        generation_tokens = getattr(usage, "completion_tokens", 0) or 0
        details = getattr(usage, "completion_tokens_details", None)
        thinking_tokens = int(getattr(details, "reasoning_tokens", 0) or 0)
        prompt_details = getattr(usage, "prompt_tokens_details", None)
        cached_tokens = int(getattr(prompt_details, "cached_tokens", 0) or 0)
        with self._lock:
            self._cached_prompt_token_count += cached_tokens

        message = response.choices[0].message
        logger.debug(
            "OpenAI call: prompt_tokens=%d (cached=%d), gen_tokens=%d, thinking_tokens=%d",
            prompt_tokens,
            cached_tokens,
            generation_tokens,
            thinking_tokens,
        )
        self._record_tokens(prompt_tokens, generation_tokens, thinking_tokens)

        return _parse_response(message.content or "", lookahead_window=self._lookahead_window)

    @property
    def estimated_cost_usd(self) -> float | None:
        m = self.model.strip().lower()
        price = None
        for key, val in OPENAI_PRICING.items():
            if m == key or m.startswith(key):
                price = val
                break
        if price is None:
            return None
        in_price, out_price = price
        cached = min(self._cached_prompt_token_count, self._prompt_token_count)
        uncached_in = self._prompt_token_count - cached
        # OpenAI bills cached input at ~10% of the standard input rate.
        return (
            uncached_in / 1_000_000 * in_price
            + cached / 1_000_000 * in_price * 0.1
            + self._output_token_count / 1_000_000 * out_price
        )


class DeepSeekEngine(BaseEngine):
    """Engine for DeepSeek's official API (api.deepseek.com, OpenAI format).

    Differences from OpenAIEngine handled here:
      - DeepSeek supports ``response_format={"type": "json_object"}`` but not
        ``json_schema``; when json_object is requested the prompt must contain
        the word "json", so a guard appends a short instruction if missing.
      - Usage reports ``prompt_cache_hit_tokens`` / ``prompt_cache_miss_tokens``
        (automatic prompt caching, no key needed); cost bills the hit portion at
        the much cheaper cache-hit rate.
      - Thinking vs non-thinking is selected by the model id (e.g. a non-thinking
        chat model) rather than a request flag. ``reasoning_tokens`` are captured
        so a smoke test can confirm thinking is actually off before a full run.
    """

    def __init__(self, config: SimulatorConfig, api_key: str | None = None):
        super().__init__(config)
        from openai import OpenAI

        self._openai_cls = OpenAI
        self._api_key = _resolve_env_api_key("DEEPSEEK_API_KEY", api_key)
        if not self._api_key:
            raise RuntimeError(
                "DEEPSEEK_API_KEY is not set. Pass --api-key or define it in the environment."
            )
        self._cached_prompt_token_count = 0
        logger.info("DeepSeek engine ready (model=%s)", self.model)

    def _make_client(self):
        return self._openai_cls(base_url=DEEPSEEK_BASE_URL, api_key=self._api_key)

    def _call_prompt_once(
        self,
        user_prompt: str,
        system_instruction: str,
        seed: int | None = None,
    ) -> StepResponse:
        # json_object mode requires the literal token "json" somewhere in the
        # messages; guarantee it without disturbing the baseline prompt content.
        if "json" not in (system_instruction + user_prompt).lower():
            user_prompt = f"{user_prompt}\n\nRespond with a single JSON object."

        # DeepSeek V4 defaults to thinking mode (expensive). Disable it unless a
        # thinking budget is explicitly requested. Verified mechanism:
        # extra_body={"thinking": {"type": "disabled"|"enabled"}}.
        thinking = int(self.config.thinking_budget) != 0 and not getattr(
            self.config, "no_reasoning", False
        )
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": user_prompt},
            ],
            "response_format": {"type": "json_object"},
            "temperature": self.config.temperature,
            "extra_body": {
                "thinking": {"type": "enabled" if thinking else "disabled"}
            },
        }
        if self.config.top_p is not None:
            kwargs["top_p"] = self.config.top_p
        if seed is not None:
            kwargs["seed"] = seed

        client = self._make_client()
        response = client.chat.completions.create(**kwargs)

        usage = response.usage
        prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
        generation_tokens = getattr(usage, "completion_tokens", 0) or 0
        details = getattr(usage, "completion_tokens_details", None)
        thinking_tokens = int(getattr(details, "reasoning_tokens", 0) or 0)
        cached_tokens = int(getattr(usage, "prompt_cache_hit_tokens", 0) or 0)
        with self._lock:
            self._cached_prompt_token_count += cached_tokens

        message = response.choices[0].message
        if thinking_tokens == 0:
            reasoning = getattr(message, "reasoning_content", None) or ""
            thinking_tokens = len(reasoning.split()) if reasoning else 0

        logger.debug(
            "DeepSeek call: prompt_tokens=%d (cache_hit=%d), gen_tokens=%d, thinking_tokens=%d",
            prompt_tokens,
            cached_tokens,
            generation_tokens,
            thinking_tokens,
        )
        self._record_tokens(prompt_tokens, generation_tokens, thinking_tokens)

        return _parse_response(message.content or "", lookahead_window=self._lookahead_window)

    @property
    def estimated_cost_usd(self) -> float | None:
        m = self.model.strip().lower()
        price = None
        for key, val in DEEPSEEK_PRICING.items():
            if m == key or m.startswith(key):
                price = val
                break
        if price is None:
            return None
        hit_price, miss_price, out_price = price
        cached = min(self._cached_prompt_token_count, self._prompt_token_count)
        uncached_in = self._prompt_token_count - cached
        return (
            uncached_in / 1_000_000 * miss_price
            + cached / 1_000_000 * hit_price
            + self._output_token_count / 1_000_000 * out_price
        )


def create_engine(config: SimulatorConfig, api_key: str | None = None) -> BaseEngine:
    """Create the backend used by the dataset validation experiments."""
    if config.backend == "nebula":
        return NebulaEngine(config, api_key=api_key)
    if config.backend == "openai":
        return OpenAIEngine(config, api_key=api_key)
    if config.backend == "deepseek":
        return DeepSeekEngine(config, api_key=api_key)
    raise ValueError(f"Unsupported backend: {config.backend}")
