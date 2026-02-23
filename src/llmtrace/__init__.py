"""llmtrace — lightweight structured tracing for LLM applications."""

from __future__ import annotations

from typing import TYPE_CHECKING

from llmtrace._version import __version__
from llmtrace.capture.context import span, span_sync
from llmtrace.capture.decorator import trace
from llmtrace.capture.tool_decorator import instrument_tools, trace_tool
from llmtrace.config import configure, get_config, reset
from llmtrace.instruments import get_instrumentor
from llmtrace.models import Cost, ErrorTrace, SpanContext, TokenUsage, ToolCallTrace, TraceEvent
from llmtrace.protocols import Enricher, Instrumentor, Sink, SinkSync

if TYPE_CHECKING:
    import asyncio

    from llmtrace.instruments._base import BaseInstrumentor

_active_instrumentors: dict[str, BaseInstrumentor] = {}
_background_tasks: set[asyncio.Task[None]] = set()


def instrument(*providers: str) -> None:
    """Activate tracing for the specified providers."""
    for provider in providers:
        if provider in _active_instrumentors:
            continue
        inst = get_instrumentor(provider)
        inst.instrument()
        _active_instrumentors[provider] = inst


def uninstrument(*providers: str) -> None:
    """Deactivate tracing for the specified providers."""
    targets = list(providers) if providers else list(_active_instrumentors.keys())
    for provider in targets:
        inst = _active_instrumentors.pop(provider, None)
        if inst is not None:
            inst.uninstrument()


def emit(event: TraceEvent) -> None:
    """Manually emit a TraceEvent to the configured sink."""
    import asyncio

    config = get_config()
    if config.sink is not None:
        try:
            loop = asyncio.get_running_loop()
            task = loop.create_task(config.sink.write(event))
            _background_tasks.add(task)
            task.add_done_callback(_background_tasks.discard)
        except RuntimeError:
            asyncio.run(config.sink.write(event))


__all__ = [
    "Cost",
    "Enricher",
    "ErrorTrace",
    "Instrumentor",
    "Sink",
    "SinkSync",
    "SpanContext",
    "TokenUsage",
    "ToolCallTrace",
    "TraceEvent",
    "__version__",
    "configure",
    "emit",
    "get_config",
    "instrument",
    "instrument_tools",
    "reset",
    "span",
    "span_sync",
    "trace",
    "trace_tool",
    "uninstrument",
]
