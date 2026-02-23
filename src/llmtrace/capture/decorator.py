"""Decorator-based tracing for LLM calls."""

from __future__ import annotations

import asyncio
import functools
import inspect
import random
import threading
import time
import traceback
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any
from uuid import uuid4

if TYPE_CHECKING:
    from uuid import UUID

from llmtrace._logging import get_logger
from llmtrace.capture.extractors import _SENSITIVE_KEYS, ExtractorRegistry, _deep_redact
from llmtrace.config import LLMTraceConfig, get_config
from llmtrace.models import ErrorTrace, TraceEvent

logger = get_logger()

_current_trace_id: ContextVar[UUID | None] = ContextVar("_current_trace_id", default=None)
_current_parent_id: ContextVar[UUID | None] = ContextVar("_current_parent_id", default=None)
_current_span_id: ContextVar[UUID | None] = ContextVar("_current_span_id", default=None)

_registry = ExtractorRegistry()

_background_tasks: set[asyncio.Task[None]] = set()

_RETRYABLE_PATTERNS = frozenset(
    {
        "ratelimit",
        "rate_limit",
        "rate limit",
        "timeout",
        "timed out",
        "429",
        "503",
        "overloaded",
    }
)


def trace(
    *,
    provider: str | None = None,
    tags: dict[str, str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> Any:
    """Decorator that traces LLM calls and emits TraceEvents."""

    def decorator(func: Any) -> Any:
        if inspect.iscoroutinefunction(func):
            return _wrap_async(func, provider, tags, metadata)
        return _wrap_sync(func, provider, tags, metadata)

    return decorator


def _wrap_async(
    func: Any,
    provider: str | None,
    tags: dict[str, str] | None,
    metadata: dict[str, Any] | None,
) -> Any:
    @functools.wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        config = get_config()

        if config.sample_rate < 1.0 and random.random() > config.sample_rate:
            return await func(*args, **kwargs)

        span_id = uuid4()
        existing_trace_id = _current_trace_id.get()
        trace_id = existing_trace_id or uuid4()
        saved_parent = _current_parent_id.get()

        trace_token = _current_trace_id.set(trace_id)
        parent_token = _current_parent_id.set(span_id)
        span_token = _current_span_id.set(span_id)

        start = time.perf_counter()
        error_trace: ErrorTrace | None = None
        result: Any = None
        try:
            result = await func(*args, **kwargs)
            return result
        except Exception as exc:
            error_trace = ErrorTrace(
                error_type=type(exc).__name__,
                message=str(exc),
                is_retryable=_is_retryable(exc),
                stack_trace=traceback.format_exc(),
            )
            raise
        finally:
            latency_ms = (time.perf_counter() - start) * 1000
            _current_span_id.reset(span_token)
            _current_parent_id.reset(parent_token)
            _current_trace_id.reset(trace_token)
            await _emit_trace_async(
                config,
                result,
                kwargs,
                provider,
                tags,
                metadata,
                latency_ms,
                error_trace,
                trace_id=trace_id,
                span_id=span_id,
                parent_id=saved_parent,
            )

    return wrapper


def _wrap_sync(
    func: Any,
    provider: str | None,
    tags: dict[str, str] | None,
    metadata: dict[str, Any] | None,
) -> Any:
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        config = get_config()

        if config.sample_rate < 1.0 and random.random() > config.sample_rate:
            return func(*args, **kwargs)

        span_id = uuid4()
        existing_trace_id = _current_trace_id.get()
        trace_id = existing_trace_id or uuid4()
        saved_parent = _current_parent_id.get()

        trace_token = _current_trace_id.set(trace_id)
        parent_token = _current_parent_id.set(span_id)
        span_token = _current_span_id.set(span_id)

        start = time.perf_counter()
        error_trace: ErrorTrace | None = None
        result: Any = None
        try:
            result = func(*args, **kwargs)
            return result
        except Exception as exc:
            error_trace = ErrorTrace(
                error_type=type(exc).__name__,
                message=str(exc),
                is_retryable=_is_retryable(exc),
                stack_trace=traceback.format_exc(),
            )
            raise
        finally:
            latency_ms = (time.perf_counter() - start) * 1000
            _current_span_id.reset(span_token)
            _current_parent_id.reset(parent_token)
            _current_trace_id.reset(trace_token)
            _emit_trace_sync(
                config,
                result,
                kwargs,
                provider,
                tags,
                metadata,
                latency_ms,
                error_trace,
                trace_id=trace_id,
                span_id=span_id,
                parent_id=saved_parent,
            )

    return wrapper


def _detect_provider(result: Any) -> str | None:
    """Auto-detect provider from the result object's type hierarchy."""
    if result is None:
        return None
    type_name = type(result).__module__
    if "anthropic" in type_name:
        return "anthropic"
    if "openai" in type_name:
        return "openai"
    if "google" in type_name:
        return "google"
    return None


_UNSET: Any = object()


def _build_trace_event(
    config: LLMTraceConfig,
    result: Any,
    kwargs: dict[str, Any],
    provider: str | None,
    tags: dict[str, str] | None,
    metadata: dict[str, Any] | None,
    latency_ms: float,
    error_trace: ErrorTrace | None,
    *,
    trace_id: UUID | None = None,
    span_id: UUID | None = None,
    parent_id: Any = _UNSET,
) -> TraceEvent:
    """Build a TraceEvent from captured data."""
    resolved_provider = provider or _detect_provider(result) or "unknown"

    extractor = _registry.get(resolved_provider)
    extracted = extractor(kwargs, result)

    merged_tags = {**config.default_tags, **(tags or {})}
    merged_metadata = {**config.default_metadata, **(metadata or {})}

    cost = None
    if extracted.token_usage is not None:
        cost = config.pricing_registry.compute_cost(
            extracted.provider, extracted.model, extracted.token_usage
        )

    request_payload = extracted.request_payload if config.capture_request else {}
    if config.redact_sensitive_keys and request_payload:
        request_payload = _deep_redact(request_payload, _SENSITIVE_KEYS)
    response_payload = extracted.response_payload if config.capture_response else {}

    resolved_parent = parent_id if parent_id is not _UNSET else _current_parent_id.get()

    trace_kw: dict[str, Any] = {}
    if trace_id is not None:
        trace_kw["trace_id"] = trace_id
    if span_id is not None:
        trace_kw["span_id"] = span_id

    event = TraceEvent(
        **trace_kw,
        parent_id=resolved_parent,
        provider=extracted.provider,
        model=extracted.model,
        request=request_payload,
        response=response_payload,
        token_usage=extracted.token_usage,
        cost=cost,
        latency_ms=latency_ms,
        tool_calls=extracted.tool_calls,
        error=error_trace or extracted.error,
        tags=merged_tags,
        metadata=merged_metadata,
    )

    for enricher in config.enrichers:
        try:
            event = enricher(event)
        except Exception as exc:
            logger.warning("Enricher failed: %s", exc)

    return event


async def _emit_trace_async(
    config: LLMTraceConfig,
    result: Any,
    kwargs: dict[str, Any],
    provider: str | None,
    tags: dict[str, str] | None,
    metadata: dict[str, Any] | None,
    latency_ms: float,
    error_trace: ErrorTrace | None,
    *,
    trace_id: UUID | None = None,
    span_id: UUID | None = None,
    parent_id: Any = _UNSET,
) -> None:
    """Build and emit a trace event (async path)."""
    try:
        event = _build_trace_event(
            config,
            result,
            kwargs,
            provider,
            tags,
            metadata,
            latency_ms,
            error_trace,
            trace_id=trace_id,
            span_id=span_id,
            parent_id=parent_id,
        )
        if config.sink is not None:
            await config.sink.write(event)
    except Exception as exc:
        logger.warning("Failed to emit trace event: %s", exc)


def _emit_trace_sync(
    config: LLMTraceConfig,
    result: Any,
    kwargs: dict[str, Any],
    provider: str | None,
    tags: dict[str, str] | None,
    metadata: dict[str, Any] | None,
    latency_ms: float,
    error_trace: ErrorTrace | None,
    *,
    trace_id: UUID | None = None,
    span_id: UUID | None = None,
    parent_id: Any = _UNSET,
) -> None:
    """Build and emit a trace event (sync path)."""
    try:
        event = _build_trace_event(
            config,
            result,
            kwargs,
            provider,
            tags,
            metadata,
            latency_ms,
            error_trace,
            trace_id=trace_id,
            span_id=span_id,
            parent_id=parent_id,
        )
        if config.sink is None:
            return
        # Check if sink.write is a coroutine function
        if inspect.iscoroutinefunction(config.sink.write):
            try:
                loop = asyncio.get_running_loop()
                task = loop.create_task(config.sink.write(event))
                _background_tasks.add(task)
                task.add_done_callback(_background_tasks.discard)
            except RuntimeError:
                # No running loop — fire and forget in a thread
                thread = threading.Thread(
                    target=asyncio.run,
                    args=(config.sink.write(event),),
                    daemon=True,
                )
                thread.start()
                thread.join(timeout=5.0)
        else:
            config.sink.write(event)
    except Exception as exc:
        logger.warning("Failed to emit trace event: %s", exc)


def _is_retryable(exc: Exception) -> bool:
    """Check if an exception is retryable (rate limit, timeout, etc.)."""
    exc_str = f"{type(exc).__name__} {exc}".lower()
    return any(pattern in exc_str for pattern in _RETRYABLE_PATTERNS)
