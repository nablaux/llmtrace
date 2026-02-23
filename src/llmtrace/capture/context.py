"""Span context managers for grouping trace events."""

from __future__ import annotations

import asyncio
import inspect
import threading
from contextlib import asynccontextmanager, contextmanager
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from uuid import uuid4

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterator
    from uuid import UUID

from llmtrace._logging import get_logger
from llmtrace.capture.decorator import _current_parent_id, _current_span_id, _current_trace_id
from llmtrace.config import get_config
from llmtrace.models import SpanContext, TraceEvent

logger = get_logger()

_background_tasks: set[asyncio.Task[None]] = set()


class SpanHandle:
    """Handle for interacting with an active async span.

    Provides methods to annotate the span, add events, and create
    child spans within an async context.
    """

    def __init__(self, context: SpanContext, parent_span_id: UUID | None) -> None:
        """Initialize the span handle.

        Args:
            context: The span context this handle operates on.
            parent_span_id: The parent span's ID, if any.
        """
        self._context = context
        self._parent_span_id = parent_span_id

    @property
    def context(self) -> SpanContext:
        """Return the underlying span context."""
        return self._context

    def annotate(self, **kwargs: Any) -> None:
        """Add annotations to the span context.

        Args:
            **kwargs: Key-value pairs to add to the span's annotations.
        """
        self._context.annotations.update(kwargs)

    def add_event(self, event: TraceEvent) -> None:
        """Add a trace event to the span.

        Args:
            event: The trace event to add.
        """
        self._context.events.append(event)

    @asynccontextmanager
    async def child(
        self,
        name: str,
        *,
        tags: dict[str, str] | None = None,
    ) -> AsyncIterator[SpanHandle]:
        """Create a child span nested under this span.

        Args:
            name: Name of the child span.
            tags: Optional tags for the child span.

        Yields:
            A SpanHandle for the child span.
        """
        async with span(name, tags=tags) as handle:
            self._context.children.append(handle.context)
            yield handle


class SpanHandleSync:
    """Handle for interacting with an active synchronous span.

    Provides the same interface as SpanHandle but for synchronous code paths.
    """

    def __init__(self, context: SpanContext, parent_span_id: UUID | None) -> None:
        """Initialize the sync span handle.

        Args:
            context: The span context this handle operates on.
            parent_span_id: The parent span's ID, if any.
        """
        self._context = context
        self._parent_span_id = parent_span_id

    @property
    def context(self) -> SpanContext:
        """Return the underlying span context."""
        return self._context

    def annotate(self, **kwargs: Any) -> None:
        """Add annotations to the span context.

        Args:
            **kwargs: Key-value pairs to add to the span's annotations.
        """
        self._context.annotations.update(kwargs)

    def add_event(self, event: TraceEvent) -> None:
        """Add a trace event to the span.

        Args:
            event: The trace event to add.
        """
        self._context.events.append(event)

    @contextmanager
    def child(
        self,
        name: str,
        *,
        tags: dict[str, str] | None = None,
    ) -> Iterator[SpanHandleSync]:
        """Create a child span nested under this span.

        Args:
            name: Name of the child span.
            tags: Optional tags for the child span.

        Yields:
            A SpanHandleSync for the child span.
        """
        with span_sync(name, tags=tags) as handle:
            self._context.children.append(handle.context)
            yield handle


@asynccontextmanager
async def span(
    name: str,
    *,
    tags: dict[str, str] | None = None,
) -> AsyncIterator[SpanHandle]:
    """Create an async span context for grouping trace events.

    Sets the current span_id so that @trace calls inside the span
    are automatically associated with this span. Also propagates
    trace_id and parent_id for correct span tree construction.

    Args:
        name: Name of the span.
        tags: Optional tags for the span.

    Yields:
        A SpanHandle for interacting with the span.
    """
    config = get_config()
    parent_span_id = _current_span_id.get(None)

    existing_trace = _current_trace_id.get()
    trace_id = existing_trace or uuid4()

    context = SpanContext(
        name=name,
        started_at=datetime.now(UTC),
        tags={**config.default_tags, **(tags or {})},
        parent_span_id=parent_span_id,
    )

    trace_token = _current_trace_id.set(trace_id)
    span_token = _current_span_id.set(context.span_id)
    parent_token = _current_parent_id.set(context.span_id)

    handle = SpanHandle(context, parent_span_id)
    try:
        yield handle
    finally:
        context.ended_at = datetime.now(UTC)
        if config.sink is not None:
            for event in context.events:
                try:
                    await config.sink.write(event)
                except Exception as exc:
                    logger.warning("Failed to emit span event: %s", exc)
        _current_span_id.reset(span_token)
        _current_parent_id.reset(parent_token)
        _current_trace_id.reset(trace_token)


@contextmanager
def span_sync(
    name: str,
    *,
    tags: dict[str, str] | None = None,
) -> Iterator[SpanHandleSync]:
    """Create a synchronous span context for grouping trace events.

    Sets the current span_id so that @trace calls inside the span
    are automatically associated with this span. Also propagates
    trace_id and parent_id for correct span tree construction.

    Args:
        name: Name of the span.
        tags: Optional tags for the span.

    Yields:
        A SpanHandleSync for interacting with the span.
    """
    config = get_config()
    parent_span_id = _current_span_id.get(None)

    existing_trace = _current_trace_id.get()
    trace_id = existing_trace or uuid4()

    context = SpanContext(
        name=name,
        started_at=datetime.now(UTC),
        tags={**config.default_tags, **(tags or {})},
        parent_span_id=parent_span_id,
    )

    trace_token = _current_trace_id.set(trace_id)
    span_token = _current_span_id.set(context.span_id)
    parent_token = _current_parent_id.set(context.span_id)

    handle = SpanHandleSync(context, parent_span_id)
    try:
        yield handle
    finally:
        context.ended_at = datetime.now(UTC)
        if config.sink is not None:
            for event in context.events:
                try:
                    _emit_event_sync(config.sink, event)
                except Exception as exc:
                    logger.warning("Failed to emit span event: %s", exc)
        _current_span_id.reset(span_token)
        _current_parent_id.reset(parent_token)
        _current_trace_id.reset(trace_token)


def _emit_event_sync(sink: Any, event: TraceEvent) -> None:
    """Emit a single event from a synchronous context to a possibly-async sink."""
    if inspect.iscoroutinefunction(sink.write):
        try:
            loop = asyncio.get_running_loop()
            task = loop.create_task(sink.write(event))
            _background_tasks.add(task)
            task.add_done_callback(_background_tasks.discard)
        except RuntimeError:
            thread = threading.Thread(
                target=asyncio.run,
                args=(sink.write(event),),
                daemon=True,
            )
            thread.start()
            thread.join(timeout=5.0)
    else:
        sink.write(event)
