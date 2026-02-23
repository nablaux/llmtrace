"""Decorator and utilities for tracing tool execution."""

from __future__ import annotations

import asyncio
import functools
import inspect
import threading
import time
import traceback
from typing import Any
from uuid import uuid4

from llmtrace._logging import get_logger
from llmtrace.capture.decorator import _current_parent_id, _current_trace_id
from llmtrace.config import LLMTraceConfig, get_config
from llmtrace.models import ErrorTrace, ToolCallTrace, TraceEvent

logger = get_logger()

_background_tasks: set[asyncio.Task[None]] = set()


def _safe_serialize(value: Any) -> Any:
    """Best-effort serialization of a value for trace storage."""
    if value is None:
        return None
    if isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, dict):
        return {str(k): _safe_serialize(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_safe_serialize(v) for v in value]
    if hasattr(value, "model_dump"):
        try:
            return value.model_dump()
        except Exception:
            pass
    try:
        return str(value)
    except Exception:
        return "<unserializable>"


def _capture_arguments(func: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> dict[str, Any]:
    """Capture function arguments as a serializable dict."""
    try:
        sig = inspect.signature(func)
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()
        return {k: _safe_serialize(v) for k, v in bound.arguments.items()}
    except Exception:
        return {"args": _safe_serialize(args), "kwargs": _safe_serialize(kwargs)}


def _build_tool_event(
    config: LLMTraceConfig,
    tool_name: str,
    arguments: dict[str, Any],
    result: Any,
    latency_ms: float,
    error_trace: ErrorTrace | None,
    tags: dict[str, str] | None,
    metadata: dict[str, Any] | None,
) -> TraceEvent:
    """Build a TraceEvent for a tool execution."""
    merged_tags = {**config.default_tags, **(tags or {})}
    merged_metadata = {**config.default_metadata, **(metadata or {})}

    success = error_trace is None
    serialized_result = _safe_serialize(result) if success else None

    tool_call = ToolCallTrace(
        tool_name=tool_name,
        arguments=arguments,
        result=serialized_result,
        latency_ms=latency_ms,
        success=success,
        error_message=error_trace.message if error_trace is not None else None,
    )

    event = TraceEvent(
        trace_id=_current_trace_id.get() or uuid4(),
        parent_id=_current_parent_id.get(),
        span_id=uuid4(),
        provider="tool",
        model=tool_name,
        request=arguments,
        response={"result": serialized_result} if success else {},
        latency_ms=latency_ms,
        tool_calls=[tool_call],
        error=error_trace,
        tags=merged_tags,
        metadata=merged_metadata,
    )

    for enricher in config.enrichers:
        try:
            event = enricher(event)
        except Exception as exc:
            logger.warning("Enricher failed: %s", exc)

    return event


async def _emit_tool_event_async(config: LLMTraceConfig, event: TraceEvent) -> None:
    """Emit a tool trace event (async path)."""
    try:
        if config.sink is not None:
            await config.sink.write(event)
    except Exception as exc:
        logger.warning("Failed to emit tool trace event: %s", exc)


def _emit_tool_event_sync(config: LLMTraceConfig, event: TraceEvent) -> None:
    """Emit a tool trace event (sync path)."""
    try:
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
    except Exception as exc:
        logger.warning("Failed to emit tool trace event: %s", exc)


def trace_tool(
    *,
    name: str | None = None,
    tags: dict[str, str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> Any:
    """Decorator that traces tool function execution.

    Wraps sync or async tool functions, capturing arguments, return value,
    latency, and errors as a TraceEvent with ``provider="tool"``.

    Args:
        name: Override the tool name (defaults to the function name).
        tags: Extra tags to attach to the trace event.
        metadata: Extra metadata to attach to the trace event.
    """

    def decorator(func: Any) -> Any:
        tool_name = name or func.__name__

        if inspect.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                config = get_config()
                arguments = _capture_arguments(func, args, kwargs)
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
                        is_retryable=False,
                        stack_trace=traceback.format_exc(),
                    )
                    raise
                finally:
                    latency_ms = (time.perf_counter() - start) * 1000
                    try:
                        event = _build_tool_event(
                            config,
                            tool_name,
                            arguments,
                            result,
                            latency_ms,
                            error_trace,
                            tags,
                            metadata,
                        )
                        await _emit_tool_event_async(config, event)
                    except Exception as emit_exc:
                        logger.warning("Failed to emit tool trace event: %s", emit_exc)

            return async_wrapper

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            config = get_config()
            arguments = _capture_arguments(func, args, kwargs)
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
                    is_retryable=False,
                    stack_trace=traceback.format_exc(),
                )
                raise
            finally:
                latency_ms = (time.perf_counter() - start) * 1000
                try:
                    event = _build_tool_event(
                        config,
                        tool_name,
                        arguments,
                        result,
                        latency_ms,
                        error_trace,
                        tags,
                        metadata,
                    )
                    _emit_tool_event_sync(config, event)
                except Exception as emit_exc:
                    logger.warning("Failed to emit tool trace event: %s", emit_exc)

        return sync_wrapper

    return decorator


def instrument_tools(
    tools: dict[str, Any],
    *,
    tags: dict[str, str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Wrap every callable in a tool mapping with ``@trace_tool``.

    Args:
        tools: Mapping of ``{name: callable}`` for tool functions.
        tags: Extra tags applied to every tool trace.
        metadata: Extra metadata applied to every tool trace.

    Returns:
        A new dict with each callable wrapped by ``trace_tool``.
    """
    wrapped: dict[str, Any] = {}
    for tool_name, func in tools.items():
        wrapped[tool_name] = trace_tool(name=tool_name, tags=tags, metadata=metadata)(func)
    return wrapped
