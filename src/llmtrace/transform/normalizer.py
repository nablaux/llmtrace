"""Event normalization and enrichment pipeline."""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from uuid import UUID

    from llmtrace.capture.extractors import ExtractedData
    from llmtrace.config import LLMTraceConfig

from llmtrace.models import Cost, ErrorTrace, TokenUsage, TraceEvent

SENSITIVE_REQUEST_KEYS: frozenset[str] = frozenset(
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


def _deep_redact_keys(data: dict[str, Any], keys: frozenset[str]) -> dict[str, Any]:
    """Recursively redact sensitive keys from a dict (case-insensitive).

    Returns a new dict — never mutates the input.
    """
    result: dict[str, Any] = {}
    for k, v in data.items():
        if k.lower() in keys:
            result[k] = "[REDACTED]"
        elif isinstance(v, dict):
            result[k] = _deep_redact_keys(v, keys)
        elif isinstance(v, list):
            result[k] = [
                _deep_redact_keys(item, keys) if isinstance(item, dict) else item for item in v
            ]
        else:
            result[k] = v
    return result


def _estimate_byte_size(data: dict[str, Any]) -> int:
    """Return approximate byte size of the dict when serialized to JSON."""
    return len(json.dumps(data, default=str))


def _compute_throughput(token_usage: TokenUsage | None, latency_ms: float) -> dict[str, float]:
    """Compute token throughput metrics.

    Returns empty dict if token_usage is None or latency_ms <= 0.
    """
    if token_usage is None or latency_ms <= 0:
        return {}
    seconds = latency_ms / 1000
    return {
        "tokens_per_second": token_usage.total_tokens / seconds,
        "input_tokens_per_second": token_usage.prompt_tokens / seconds,
        "output_tokens_per_second": token_usage.completion_tokens / seconds,
    }


ERROR_TAXONOMY: dict[str, str] = {
    # auth — before invalid_request since "invalid api key" contains "invalid"
    r"authentication": "auth",
    r"\b401\b": "auth",
    r"\b403\b": "auth",
    r"invalid.api.key": "auth",
    # rate_limit
    r"ratelimit": "rate_limit",
    r"rate.limit": "rate_limit",
    r"\b429\b": "rate_limit",
    r"too many requests": "rate_limit",
    # timeout
    r"timeout": "timeout",
    r"timed.out": "timeout",
    r"deadline.exceeded": "timeout",
    # context_length
    r"context.length": "context_length",
    r"token.limit": "context_length",
    r"maximum.*tokens": "context_length",
    # content_filter
    r"content.filter": "content_filter",
    r"safety": "content_filter",
    r"blocked": "content_filter",
    # server
    r"\b500\b": "server",
    r"\b502\b": "server",
    r"\b503\b": "server",
    r"internal.server.error": "server",
    # invalid_request — last because "invalid" is generic
    r"\b400\b": "invalid_request",
    r"malformed": "invalid_request",
    r"invalid": "invalid_request",
}


def _classify_error(error: ErrorTrace) -> str:
    """Classify an error into a normalized category.

    Checks error_type, message, and provider_error_code against ERROR_TAXONOMY.
    Returns the first matching category, or "unknown".
    """
    combined = f"{error.error_type} {error.message} {error.provider_error_code or ''}"
    for pattern, category in ERROR_TAXONOMY.items():
        if re.search(pattern, combined, re.IGNORECASE):
            return category
    return "unknown"


def normalize(
    extracted: ExtractedData,
    latency_ms: float,
    config: LLMTraceConfig,
    *,
    parent_id: UUID | None = None,
    span_id: UUID | None = None,
    extra_tags: dict[str, str] | None = None,
    extra_metadata: dict[str, Any] | None = None,
    redact_sensitive_keys: bool = True,
) -> TraceEvent:
    """Normalize extracted data into a fully enriched TraceEvent."""
    # a) Redact sensitive keys from request payload
    request_payload = extracted.request_payload
    if redact_sensitive_keys:
        request_payload = _deep_redact_keys(request_payload, SENSITIVE_REQUEST_KEYS)

    # b) Apply capture_request / capture_response filters
    request = request_payload if config.capture_request else {}
    response = extracted.response_payload if config.capture_response else {}

    # c) Compute cost via pricing registry
    cost: Cost | None = None
    if extracted.token_usage is not None:
        cost = config.pricing_registry.compute_cost(
            extracted.provider, extracted.model, extracted.token_usage
        )

    # d) Compute throughput metrics
    throughput = _compute_throughput(extracted.token_usage, latency_ms)

    # e) Classify error
    error_metadata: dict[str, Any] = {}
    if extracted.error is not None:
        error_metadata["error_category"] = _classify_error(extracted.error)

    # f) Estimate request/response byte sizes
    byte_sizes: dict[str, Any] = {
        "request_byte_size": _estimate_byte_size(request),
        "response_byte_size": _estimate_byte_size(response),
    }

    # g) Merge tags: config defaults + extra (extra overrides)
    merged_tags = {**config.default_tags, **(extra_tags or {})}

    # h) Merge metadata: config defaults + throughput + byte_sizes + error + extra
    merged_metadata: dict[str, Any] = {
        **config.default_metadata,
        **throughput,
        **byte_sizes,
        **error_metadata,
        **(extra_metadata or {}),
    }

    span_kw: dict[str, Any] = {}
    if span_id is not None:
        span_kw["span_id"] = span_id

    return TraceEvent(
        parent_id=parent_id,
        **span_kw,
        provider=extracted.provider,
        model=extracted.model,
        request=request,
        response=response,
        token_usage=extracted.token_usage,
        cost=cost,
        latency_ms=latency_ms,
        tool_calls=extracted.tool_calls,
        error=extracted.error,
        tags=merged_tags,
        metadata=merged_metadata,
    )
