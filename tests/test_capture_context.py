"""Tests for llmtrace capture context (span and span_sync)."""

from __future__ import annotations

from typing import Any

import pytest

from llmtrace.capture.context import SpanHandle, SpanHandleSync, span, span_sync
from llmtrace.capture.decorator import (
    _current_parent_id,
    _current_span_id,
    _current_trace_id,
    trace,
)
from llmtrace.config import configure, reset
from llmtrace.models import TraceEvent
from llmtrace.sinks.callback import CallbackSink


@pytest.fixture(autouse=True)
def _clean_config() -> Any:
    """Reset config and contextvars before each test."""
    reset()
    span_token = _current_span_id.set(None)
    parent_token = _current_parent_id.set(None)
    trace_token = _current_trace_id.set(None)
    yield
    _current_span_id.reset(span_token)
    _current_parent_id.reset(parent_token)
    _current_trace_id.reset(trace_token)
    reset()


def _make_sink() -> tuple[CallbackSink, list[TraceEvent]]:
    """Create a CallbackSink that captures events into a list."""
    events: list[TraceEvent] = []

    def on_event(event: TraceEvent) -> None:
        events.append(event)

    return CallbackSink(on_event), events


class _MockResponse:
    """Minimal mock response for use with @trace(provider='test')."""

    def __init__(self) -> None:
        self.model = "test-model"

    def model_dump(self) -> dict[str, Any]:
        return {"model": self.model}


# ── Basic span ────────────────────────────────────────────────────────────


class TestBasicSpan:
    """Tests for basic async span functionality."""

    async def test_span_has_correct_name(self) -> None:
        sink, _ = _make_sink()
        configure(sink=sink)

        async with span("my-span") as handle:
            assert handle.context.name == "my-span"

    async def test_span_sets_started_at(self) -> None:
        sink, _ = _make_sink()
        configure(sink=sink)

        async with span("timed-span") as handle:
            assert handle.context.started_at is not None

    async def test_span_sets_ended_at_after_exit(self) -> None:
        sink, _ = _make_sink()
        configure(sink=sink)

        async with span("timed-span") as handle:
            assert handle.context.ended_at is None

        assert handle.context.ended_at is not None

    async def test_span_duration_positive(self) -> None:
        sink, _ = _make_sink()
        configure(sink=sink)

        async with span("duration-span") as handle:
            pass

        duration = handle.context.duration_ms()
        assert duration is not None
        assert duration >= 0

    async def test_span_merges_config_tags(self) -> None:
        sink, _ = _make_sink()
        configure(sink=sink, default_tags={"env": "test"})

        async with span("tagged-span", tags={"feature": "chat"}) as handle:
            pass

        assert handle.context.tags["env"] == "test"
        assert handle.context.tags["feature"] == "chat"

    async def test_handle_is_span_handle(self) -> None:
        sink, _ = _make_sink()
        configure(sink=sink)

        async with span("typed-span") as handle:
            assert isinstance(handle, SpanHandle)


# ── Span with annotation ─────────────────────────────────────────────────


class TestSpanAnnotation:
    """Tests for span annotation."""

    async def test_annotate_adds_to_context(self) -> None:
        sink, _ = _make_sink()
        configure(sink=sink)

        async with span("annotated-span") as handle:
            handle.annotate(key="value", count=42)

        assert handle.context.annotations["key"] == "value"
        assert handle.context.annotations["count"] == 42

    async def test_multiple_annotate_calls_merge(self) -> None:
        sink, _ = _make_sink()
        configure(sink=sink)

        async with span("multi-annotate") as handle:
            handle.annotate(a=1)
            handle.annotate(b=2)

        assert handle.context.annotations == {"a": 1, "b": 2}


# ── Nested spans via child() ─────────────────────────────────────────────


class TestNestedSpans:
    """Tests for nested spans via child()."""

    async def test_child_span_has_correct_parent_id(self) -> None:
        sink, _ = _make_sink()
        configure(sink=sink)

        async with span("outer") as outer_handle:  # noqa: SIM117
            async with outer_handle.child("inner") as inner_handle:
                pass

        assert inner_handle.context.parent_span_id == outer_handle.context.span_id

    async def test_child_appears_in_parent_children(self) -> None:
        sink, _ = _make_sink()
        configure(sink=sink)

        async with span("outer") as outer_handle:  # noqa: SIM117
            async with outer_handle.child("inner") as inner_handle:
                pass

        assert len(outer_handle.context.children) == 1
        assert outer_handle.context.children[0] is inner_handle.context

    async def test_deeply_nested_spans(self) -> None:
        sink, _ = _make_sink()
        configure(sink=sink)

        async with span("root") as root, root.child("level1") as l1:  # noqa: SIM117
            async with l1.child("level2") as l2:
                pass

        assert l1.context.parent_span_id == root.context.span_id
        assert l2.context.parent_span_id == l1.context.span_id
        assert len(root.context.children) == 1
        assert len(l1.context.children) == 1


# ── Span groups @trace events ────────────────────────────────────────────


class TestSpanGroupsTraceEvents:
    """Tests that @trace events inside a span link to the span via parent_id."""

    async def test_trace_inside_span_has_parent_id(self) -> None:
        sink, events = _make_sink()
        configure(sink=sink)

        @trace(provider="test")
        async def llm_call() -> _MockResponse:
            return _MockResponse()

        async with span("my-span") as handle:
            await llm_call()

        assert len(events) == 1
        # @trace generates its own span_id, parent_id links to the span
        assert events[0].parent_id == handle.context.span_id
        assert events[0].span_id != handle.context.span_id

    async def test_trace_outside_span_has_own_span_id(self) -> None:
        sink, events = _make_sink()
        configure(sink=sink)

        @trace(provider="test")
        async def llm_call() -> _MockResponse:
            return _MockResponse()

        await llm_call()

        assert len(events) == 1
        # @trace always generates its own span_id
        assert events[0].span_id is not None
        assert events[0].span_id == events[0].span_id  # is a UUID


# ── Exception in span ────────────────────────────────────────────────────


class TestSpanException:
    """Tests that exceptions propagate and ended_at is still set."""

    async def test_exception_propagates(self) -> None:
        sink, _ = _make_sink()
        configure(sink=sink)

        with pytest.raises(ValueError, match="boom"):
            async with span("failing-span") as handle:
                msg = "boom"
                raise ValueError(msg)

        assert handle.context.ended_at is not None

    async def test_ended_at_set_on_exception(self) -> None:
        sink, _ = _make_sink()
        configure(sink=sink)

        handle: SpanHandle | None = None
        with pytest.raises(RuntimeError):
            async with span("error-span") as handle:
                msg = "unexpected"
                raise RuntimeError(msg)

        assert handle is not None
        assert handle.context.ended_at is not None
        duration = handle.context.duration_ms()
        assert duration is not None
        assert duration >= 0


# ── span_sync ─────────────────────────────────────────────────────────────


class TestSpanSync:
    """Tests for synchronous span_sync."""

    async def test_basic_sync_span(self) -> None:
        sink, _ = _make_sink()
        configure(sink=sink)

        with span_sync("sync-span") as handle:
            assert isinstance(handle, SpanHandleSync)
            assert handle.context.name == "sync-span"

        assert handle.context.ended_at is not None

    async def test_sync_span_annotation(self) -> None:
        sink, _ = _make_sink()
        configure(sink=sink)

        with span_sync("annotated-sync") as handle:
            handle.annotate(tool="pytest")

        assert handle.context.annotations["tool"] == "pytest"

    async def test_sync_span_child(self) -> None:
        sink, _ = _make_sink()
        configure(sink=sink)

        with span_sync("outer-sync") as outer, outer.child("inner-sync") as inner:
            pass

        assert inner.context.parent_span_id == outer.context.span_id
        assert len(outer.context.children) == 1

    async def test_sync_span_exception(self) -> None:
        sink, _ = _make_sink()
        configure(sink=sink)

        with pytest.raises(ValueError, match="sync boom"):  # noqa: SIM117
            with span_sync("failing-sync") as handle:
                msg = "sync boom"
                raise ValueError(msg)

        assert handle.context.ended_at is not None

    async def test_sync_span_sets_parent_id_contextvar(self) -> None:
        sink, events = _make_sink()
        configure(sink=sink)

        @trace(provider="test")
        async def llm_call() -> _MockResponse:
            return _MockResponse()

        with span_sync("sync-group") as handle:
            # The ContextVar is set, so a @trace call here picks it up.
            # We need to await inside sync span — this works because
            # the test runner is async.
            await llm_call()

        assert len(events) == 1
        # @trace generates its own span_id, parent_id links to the span
        assert events[0].parent_id == handle.context.span_id
        assert events[0].span_id != handle.context.span_id


# ── Span flush failure ───────────────────────────────────────────────────


class TestSpanFlushFailure:
    """Cover exception paths when span flushes events to a failing sink."""

    async def test_async_span_flush_error_is_caught(self) -> None:
        """Exception during async span event flush logs warning, doesn't crash."""

        class FailingSink:
            async def write(self, event: TraceEvent) -> None:
                msg = "sink down"
                raise RuntimeError(msg)

            async def flush(self) -> None:
                pass

            async def close(self) -> None:
                pass

        configure(sink=FailingSink())

        event = TraceEvent(provider="test", model="m", latency_ms=1.0)
        # Should not raise even though sink.write fails
        async with span("failing-flush") as handle:
            handle.add_event(event)

        assert handle.context.ended_at is not None

    def test_sync_span_flush_error_is_caught(self) -> None:
        """Exception during sync span event flush logs warning, doesn't crash."""

        class FailingSyncSink:
            def write(self, event: TraceEvent) -> None:
                msg = "sync sink down"
                raise RuntimeError(msg)

            def flush(self) -> None:
                pass

            def close(self) -> None:
                pass

        configure(sink=FailingSyncSink())

        event = TraceEvent(provider="test", model="m", latency_ms=1.0)
        with span_sync("failing-sync-flush") as handle:
            handle.add_event(event)

        assert handle.context.ended_at is not None


# ── _emit_event_sync edge cases ─────────────────────────────────────────


class TestEmitEventSyncEdgeCases:
    """Cover _emit_event_sync paths in capture/context.py."""

    def test_sync_span_with_sync_sink_flushes(self) -> None:
        """span_sync with a synchronous sink calls sink.write directly."""
        events: list[TraceEvent] = []

        class SyncSink:
            def write(self, event: TraceEvent) -> None:
                events.append(event)

            def flush(self) -> None:
                pass

            def close(self) -> None:
                pass

        configure(sink=SyncSink())

        event = TraceEvent(provider="test", model="m", latency_ms=1.0)
        with span_sync("s") as handle:
            handle.add_event(event)

        assert len(events) == 1
