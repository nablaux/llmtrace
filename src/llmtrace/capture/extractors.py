"""Provider-specific data extraction from LLM API responses."""

from __future__ import annotations

import contextlib
import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

from llmtrace._logging import get_logger
from llmtrace.models import ErrorTrace, TokenUsage, ToolCallTrace

logger = get_logger()


@dataclass
class ExtractedData:
    """Internal container for data extracted from an LLM provider response.

    Not a Pydantic model — this is an internal data transfer object used
    between the capture and transform stages.
    """

    provider: str
    model: str
    request_payload: dict[str, Any]
    response_payload: dict[str, Any]
    token_usage: TokenUsage | None = None
    tool_calls: list[ToolCallTrace] = field(default_factory=list)
    error: ErrorTrace | None = None


_SENSITIVE_KEYS: frozenset[str] = frozenset(
    {
        "api_key",
        "apikey",
        "api-key",
        "authorization",
        "x-api-key",
        "secret",
        "token",
        "password",
        "credential",
    }
)


def _deep_redact(data: dict[str, Any], keys: frozenset[str]) -> dict[str, Any]:
    """Recursively redact sensitive keys from a dict (case-insensitive)."""
    result: dict[str, Any] = {}
    for k, v in data.items():
        if k.lower() in keys:
            result[k] = "[REDACTED]"
        elif isinstance(v, dict):
            result[k] = _deep_redact(v, keys)
        elif isinstance(v, list):
            result[k] = [_deep_redact(item, keys) if isinstance(item, dict) else item for item in v]
        else:
            result[k] = v
    return result


def _sanitize_request(request_kwargs: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of request kwargs with sensitive fields deeply redacted."""
    return _deep_redact(request_kwargs, _SENSITIVE_KEYS)


def _dump_response(response: Any) -> dict[str, Any]:
    """Attempt to serialize a response object to a dict."""
    if hasattr(response, "model_dump"):
        try:
            result: dict[str, Any] = response.model_dump()
            return result
        except Exception:
            logger.debug("model_dump() failed, falling back to str()")
    return {"raw": str(response)}


def extract_anthropic(request_kwargs: dict[str, Any], response: Any) -> ExtractedData:
    """Extract trace data from an Anthropic Message response.

    Handles the Anthropic SDK response shape without importing anthropic
    at module level. Uses defensive attribute access throughout.
    """
    model = "unknown"
    with contextlib.suppress(AttributeError):
        model = response.model

    token_usage: TokenUsage | None = None
    try:
        usage = response.usage
        input_t: int = usage.input_tokens
        output_t: int = usage.output_tokens
        token_usage = TokenUsage(
            prompt_tokens=input_t,
            completion_tokens=output_t,
            total_tokens=input_t + output_t,
        )
    except (AttributeError, TypeError, Exception) as exc:
        logger.debug("Could not extract Anthropic token usage: %s", exc)

    tool_calls: list[ToolCallTrace] = []
    try:
        for block in response.content:
            if getattr(block, "type", None) == "tool_use":
                tool_calls.append(
                    ToolCallTrace(
                        tool_name=block.name,
                        arguments=block.input if isinstance(block.input, dict) else {},
                    )
                )
    except (AttributeError, TypeError) as exc:
        logger.debug("Could not extract Anthropic tool calls: %s", exc)

    return ExtractedData(
        provider="anthropic",
        model=model,
        request_payload=_sanitize_request(request_kwargs),
        response_payload=_dump_response(response),
        token_usage=token_usage,
        tool_calls=tool_calls,
    )


def extract_openai(request_kwargs: dict[str, Any], response: Any) -> ExtractedData:
    """Extract trace data from an OpenAI ChatCompletion response.

    Handles the OpenAI SDK response shape without importing openai
    at module level. Uses defensive attribute access throughout.
    """
    model = "unknown"
    with contextlib.suppress(AttributeError):
        model = response.model

    token_usage: TokenUsage | None = None
    try:
        usage = response.usage
        prompt_t: int = usage.prompt_tokens
        compl_t: int = usage.completion_tokens
        total_t: int = getattr(usage, "total_tokens", prompt_t + compl_t)
        token_usage = TokenUsage(
            prompt_tokens=prompt_t,
            completion_tokens=compl_t,
            total_tokens=total_t,
        )
    except (AttributeError, TypeError, Exception) as exc:
        logger.debug("Could not extract OpenAI token usage: %s", exc)

    tool_calls: list[ToolCallTrace] = []
    try:
        choices = response.choices
        if choices:
            message = choices[0].message
            raw_tool_calls = getattr(message, "tool_calls", None)
            if raw_tool_calls:
                for tc in raw_tool_calls:
                    arguments: dict[str, Any] = {}
                    try:
                        arguments = json.loads(tc.function.arguments)
                    except (json.JSONDecodeError, AttributeError, TypeError):
                        logger.debug("Could not parse tool call arguments")
                    tool_calls.append(
                        ToolCallTrace(
                            tool_name=tc.function.name,
                            arguments=arguments,
                        )
                    )
    except (AttributeError, TypeError, IndexError) as exc:
        logger.debug("Could not extract OpenAI tool calls: %s", exc)

    return ExtractedData(
        provider="openai",
        model=model,
        request_payload=_sanitize_request(request_kwargs),
        response_payload=_dump_response(response),
        token_usage=token_usage,
        tool_calls=tool_calls,
    )


def extract_generic(provider: str, request_kwargs: dict[str, Any], response: Any) -> ExtractedData:
    """Fallback extractor for unknown LLM providers.

    Attempts to extract common fields using duck-typed attribute access.
    """
    model = "unknown"
    with contextlib.suppress(AttributeError):
        model = response.model

    # Try to serialize response
    response_payload: dict[str, Any]
    if hasattr(response, "model_dump"):
        try:
            response_payload = response.model_dump()
        except Exception:
            response_payload = {"raw": str(response)}
    else:
        try:
            response_payload = dict(vars(response))
        except TypeError:
            response_payload = {"raw": str(response)}

    # Attempt token usage extraction from common field names
    token_usage: TokenUsage | None = None
    try:
        usage = response.usage
        prompt = getattr(usage, "prompt_tokens", None) or getattr(usage, "input_tokens", None)
        completion = getattr(usage, "completion_tokens", None) or getattr(
            usage, "output_tokens", None
        )
        if prompt is not None and completion is not None:
            p = int(prompt)
            c = int(completion)
            token_usage = TokenUsage(
                prompt_tokens=p,
                completion_tokens=c,
                total_tokens=p + c,
            )
    except (AttributeError, TypeError, ValueError) as exc:
        logger.debug("Could not extract generic token usage: %s", exc)

    return ExtractedData(
        provider=provider,
        model=model,
        request_payload=_sanitize_request(request_kwargs),
        response_payload=response_payload,
        token_usage=token_usage,
    )


class ExtractorRegistry:
    """Registry mapping provider names to extraction functions.

    Provides built-in extractors for Anthropic and OpenAI, with a fallback
    to the generic extractor for unknown providers.
    """

    def __init__(self) -> None:
        self._extractors: dict[str, Callable[..., ExtractedData]] = {
            "anthropic": lambda kwargs, resp: extract_anthropic(kwargs, resp),
            "openai": lambda kwargs, resp: extract_openai(kwargs, resp),
        }

    def get(self, provider: str) -> Callable[..., ExtractedData]:
        """Return the extractor for the given provider.

        Falls back to extract_generic (partially applied with provider name)
        if no specific extractor is registered.
        """
        if provider in self._extractors:
            return self._extractors[provider]
        return lambda kwargs, resp: extract_generic(provider, kwargs, resp)

    def register(self, provider: str, fn: Callable[..., ExtractedData]) -> None:
        """Register a custom extractor for a provider."""
        self._extractors[provider] = fn
