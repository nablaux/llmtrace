"""Base instrumentor for monkey-patching LLM provider methods."""

from __future__ import annotations

import asyncio
import functools
import inspect
import random
import threading
import time
import traceback
from abc import ABC, abstractmethod
from typing import Any
from uuid import uuid4

from llmtrace._logging import get_logger
from llmtrace.capture.decorator import _current_parent_id, _current_trace_id
from llmtrace.capture.extractors import _SENSITIVE_KEYS, ExtractorRegistry, _deep_redact
from llmtrace.config import LLMTraceConfig, get_config
from llmtrace.models import ErrorTrace, TraceEvent

logger = get_logger()

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


def _is_retryable(exc: Exception) -> bool:
    exc_str = f"{type(exc).__name__} {exc}".lower()
    return any(pattern in exc_str for pattern in _RETRYABLE_PATTERNS)


def _build_event(
    config: LLMTraceConfig,
    provider: str,
    result: Any,
    kwargs: dict[str, Any],
    latency_ms: float,
    error_trace: ErrorTrace | None,
) -> TraceEvent:
    extractor = _registry.get(provider)
    extracted = extractor(kwargs, result)

    cost = None
    if extracted.token_usage is not None:
        cost = config.pricing_registry.compute_cost(
            extracted.provider, extracted.model, extracted.token_usage
        )

    request_payload = extracted.request_payload if config.capture_request else {}
    if config.redact_sensitive_keys and request_payload:
        request_payload = _deep_redact(request_payload, _SENSITIVE_KEYS)
    response_payload = extracted.response_payload if config.capture_response else {}

    event = TraceEvent(
        trace_id=_current_trace_id.get() or uuid4(),
        parent_id=_current_parent_id.get(),
        span_id=uuid4(),
        provider=extracted.provider,
        model=extracted.model,
        request=request_payload,
        response=response_payload,
        token_usage=extracted.token_usage,
        cost=cost,
        latency_ms=latency_ms,
        tool_calls=extracted.tool_calls,
        error=error_trace or extracted.error,
        tags=dict(config.default_tags),
        metadata=dict(config.default_metadata),
    )

    for enricher in config.enrichers:
        try:
            event = enricher(event)
        except Exception as exc:
            logger.warning("Enricher failed: %s", exc)

    return event


async def _trace_coroutine(
    config: LLMTraceConfig,
    provider: str,
    coro: Any,
    kwargs: dict[str, Any],
    start: float,
) -> Any:
    """Await a coroutine result and emit trace data after completion.

    Handles the case where a method is not detected as a coroutine function
    by inspect.iscoroutinefunction (e.g. due to SDK decorators) but still
    returns a coroutine at runtime.
    """
    error_trace: ErrorTrace | None = None
    result: Any = None
    try:
        result = await coro
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
        try:
            event = _build_event(config, provider, result, kwargs, latency_ms, error_trace)
            if config.sink is not None:
                await config.sink.write(event)
        except Exception as emit_exc:
            logger.warning("Failed to emit trace event: %s", emit_exc)


def _emit_sync(config: LLMTraceConfig, event: TraceEvent) -> None:
    if config.sink is None:
        return
    if inspect.iscoroutinefunction(config.sink.write):
        try:
            loop = asyncio.get_running_loop()
            task = loop.create_task(config.sink.write(event))
            _background_tasks.add(task)
            task.add_done_callback(_background_tasks.discard)
        except RuntimeError:
            thread = threading.Thread(
                target=asyncio.run,
                args=(config.sink.write(event),),
                daemon=True,
            )
            thread.start()
            thread.join(timeout=5.0)
    else:
        config.sink.write(event)


class BaseInstrumentor(ABC):
    """Abstract base for provider instrumentors.

    Concrete subclasses implement provider_name and _get_targets().
    This base handles monkey-patching lifecycle and wrapper creation.
    """

    def __init__(self) -> None:
        self._original_methods: dict[str, Any] = {}
        self._is_instrumented: bool = False

    @property
    @abstractmethod
    def provider_name(self) -> str: ...

    @abstractmethod
    def _get_targets(self) -> list[tuple[Any, str]]: ...

    def instrument(self) -> None:
        if self._is_instrumented:
            return
        for obj, method_name in self._get_targets():
            key = f"{id(obj)}:{method_name}"
            original = getattr(obj, method_name)
            self._original_methods[key] = original
            wrapper = self._create_wrapper(original, obj, method_name)
            setattr(obj, method_name, wrapper)
        self._is_instrumented = True

    def uninstrument(self) -> None:
        if not self._is_instrumented:
            return
        for obj, method_name in self._get_targets():
            key = f"{id(obj)}:{method_name}"
            if key in self._original_methods:
                setattr(obj, method_name, self._original_methods[key])
        self._original_methods.clear()
        self._is_instrumented = False

    def _create_wrapper(self, original: Any, obj: Any, method_name: str) -> Any:
        provider = self.provider_name

        if inspect.iscoroutinefunction(original):

            @functools.wraps(original)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                config = get_config()
                if config.sample_rate < 1.0 and random.random() > config.sample_rate:
                    return await original(*args, **kwargs)

                start = time.perf_counter()
                error_trace: ErrorTrace | None = None
                result: Any = None
                try:
                    result = await original(*args, **kwargs)
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
                    try:
                        event = _build_event(
                            config, provider, result, kwargs, latency_ms, error_trace
                        )
                        if config.sink is not None:
                            await config.sink.write(event)
                    except Exception as emit_exc:
                        logger.warning("Failed to emit trace event: %s", emit_exc)

            return async_wrapper

        @functools.wraps(original)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            config = get_config()
            if config.sample_rate < 1.0 and random.random() > config.sample_rate:
                return original(*args, **kwargs)

            start = time.perf_counter()
            error_trace: ErrorTrace | None = None
            result: Any = None
            is_coro = False
            try:
                result = original(*args, **kwargs)
                if asyncio.iscoroutine(result):
                    is_coro = True
                    return _trace_coroutine(config, provider, result, kwargs, start)
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
                if not is_coro:
                    latency_ms = (time.perf_counter() - start) * 1000
                    try:
                        event = _build_event(config, provider, result, kwargs, latency_ms, error_trace)
                        _emit_sync(config, event)
                    except Exception as emit_exc:
                        logger.warning("Failed to emit trace event: %s", emit_exc)

        return sync_wrapper
