"""Tests for llmtrace tool tracing decorator."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from llmtrace.capture.decorator import _current_parent_id, _current_span_id, _current_trace_id
from llmtrace.capture.tool_decorator import instrument_tools, trace_tool
from llmtrace.config import configure, reset
from llmtrace.models import TraceEvent
from llmtrace.sinks.callback import CallbackSink


@pytest.fixture(autouse=True)
def _clean_config() -> Any:
    """Reset config and contextvars before each test."""
    reset()
    parent_token = _current_parent_id.set(None)
    span_token = _current_span_id.set(None)
    trace_token = _current_trace_id.set(None)
    yield
    _current_parent_id.reset(parent_token)
    _current_span_id.reset(span_token)
    _current_trace_id.reset(trace_token)
    reset()


def _make_sink() -> tuple[CallbackSink, list[TraceEvent]]:
    """Create a CallbackSink that captures events into a list."""
    events: list[TraceEvent] = []

    def on_event(event: TraceEvent) -> None:
        events.append(event)

    return CallbackSink(on_event), events


# ── Async tool tests ────────────────────────────────────────────────────


class TestAsyncTraceTool:
    """Tests for tracing async tool functions."""

    async def test_emits_trace_event(self) -> None:
        sink, events = _make_sink()
        configure(sink=sink)

        @trace_tool()
        async def get_weather(city: str) -> str:
            return f"Sunny in {city}"

        result = await get_weather("Paris")
        assert result == "Sunny in Paris"
        assert len(events) == 1
        assert events[0].provider == "tool"
        assert events[0].model == "get_weather"

    async def test_captures_args_and_result(self) -> None:
        sink, events = _make_sink()
        configure(sink=sink)

        @trace_tool()
        async def search(query: str, limit: int = 10) -> list[str]:
            return ["result1", "result2"]

        await search("test", limit=5)
        assert len(events) == 1
        event = events[0]
        assert event.request["query"] == "test"
        assert event.request["limit"] == 5
        assert event.response["result"] == ["result1", "result2"]
        assert len(event.tool_calls) == 1
        assert event.tool_calls[0].tool_name == "search"
        assert event.tool_calls[0].success is True

    async def test_latency_measured(self) -> None:
        sink, events = _make_sink()
        configure(sink=sink)

        @trace_tool()
        async def slow_tool() -> str:
            await asyncio.sleep(0.01)
            return "done"

        await slow_tool()
        assert len(events) == 1
        assert events[0].latency_ms >= 10

    async def test_preserves_return_value(self) -> None:
        sink, _ = _make_sink()
        configure(sink=sink)

        sentinel = object()

        @trace_tool()
        async def my_tool() -> Any:
            return sentinel

        result = await my_tool()
        assert result is sentinel

    async def test_custom_name(self) -> None:
        sink, events = _make_sink()
        configure(sink=sink)

        @trace_tool(name="custom_tool_name")
        async def my_tool() -> str:
            return "ok"

        await my_tool()
        assert events[0].model == "custom_tool_name"
        assert events[0].tool_calls[0].tool_name == "custom_tool_name"

    async def test_error_handling(self) -> None:
        sink, events = _make_sink()
        configure(sink=sink)

        @trace_tool()
        async def failing_tool() -> None:
            msg = "tool broke"
            raise ValueError(msg)

        with pytest.raises(ValueError, match="tool broke"):
            await failing_tool()

        assert len(events) == 1
        assert events[0].error is not None
        assert events[0].error.error_type == "ValueError"
        assert events[0].error.message == "tool broke"
        assert events[0].error.stack_trace is not None
        assert events[0].tool_calls[0].success is False
        assert events[0].tool_calls[0].error_message == "tool broke"

    async def test_span_propagation(self) -> None:
        sink, events = _make_sink()
        configure(sink=sink)

        parent_id = uuid4()
        trace_id = uuid4()
        _current_parent_id.set(parent_id)
        _current_trace_id.set(trace_id)

        @trace_tool()
        async def my_tool() -> str:
            return "ok"

        await my_tool()
        assert events[0].parent_id == parent_id
        assert events[0].trace_id == trace_id
        # Tool events get their own unique span_id
        assert events[0].span_id != parent_id

    async def test_tags_and_metadata(self) -> None:
        sink, events = _make_sink()
        configure(sink=sink, default_tags={"env": "test"})

        @trace_tool(tags={"category": "search"}, metadata={"version": "1.0"})
        async def my_tool() -> str:
            return "ok"

        await my_tool()
        assert events[0].tags["env"] == "test"
        assert events[0].tags["category"] == "search"
        assert events[0].metadata["version"] == "1.0"


# ── Sync tool tests ─────────────────────────────────────────────────────


class TestSyncTraceTool:
    """Tests for tracing sync tool functions."""

    async def test_emits_trace_event(self) -> None:
        sink, events = _make_sink()
        configure(sink=sink)

        @trace_tool()
        def get_weather(city: str) -> str:
            return f"Sunny in {city}"

        result = get_weather("Tokyo")
        assert result == "Sunny in Tokyo"
        # Sync path with async sink fires in background; give it time
        await asyncio.sleep(0.1)
        assert len(events) == 1
        assert events[0].provider == "tool"
        assert events[0].model == "get_weather"

    async def test_preserves_return_value(self) -> None:
        sink, _ = _make_sink()
        configure(sink=sink)

        sentinel = {"key": "value"}

        @trace_tool()
        def my_tool() -> dict[str, str]:
            return sentinel

        result = my_tool()
        assert result is sentinel

    async def test_error_handling(self) -> None:
        sink, events = _make_sink()
        configure(sink=sink)

        @trace_tool()
        def failing_tool() -> None:
            msg = "sync tool failure"
            raise RuntimeError(msg)

        with pytest.raises(RuntimeError, match="sync tool failure"):
            failing_tool()

        await asyncio.sleep(0.1)
        assert len(events) == 1
        assert events[0].error is not None
        assert events[0].error.error_type == "RuntimeError"


# ── instrument_tools tests ──────────────────────────────────────────────


class TestInstrumentTools:
    """Tests for the instrument_tools() convenience function."""

    async def test_wraps_all_tools(self) -> None:
        sink, events = _make_sink()
        configure(sink=sink)

        async def get_weather(city: str) -> str:
            return f"Sunny in {city}"

        async def search(query: str) -> list[str]:
            return ["a", "b"]

        wrapped = instrument_tools({"get_weather": get_weather, "search": search})

        assert await wrapped["get_weather"]("Paris") == "Sunny in Paris"
        assert await wrapped["search"]("test") == ["a", "b"]
        assert len(events) == 2

    async def test_correct_tool_names(self) -> None:
        sink, events = _make_sink()
        configure(sink=sink)

        async def my_func() -> str:
            return "ok"

        wrapped = instrument_tools({"custom_name": my_func})
        await wrapped["custom_name"]()

        assert events[0].model == "custom_name"
        assert events[0].tool_calls[0].tool_name == "custom_name"

    async def test_tags_and_metadata_applied(self) -> None:
        sink, events = _make_sink()
        configure(sink=sink)

        async def my_func() -> str:
            return "ok"

        wrapped = instrument_tools(
            {"my_tool": my_func},
            tags={"group": "utils"},
            metadata={"source": "test"},
        )
        await wrapped["my_tool"]()

        assert events[0].tags["group"] == "utils"
        assert events[0].metadata["source"] == "test"


# ── Sink error safety ────────────────────────────────────────────────────


class TestToolSinkErrorSafety:
    """Tests that sink errors never crash the user's tool code."""

    async def test_broken_sink_doesnt_crash_async(self) -> None:
        broken_sink = MagicMock()
        broken_sink.write = MagicMock(side_effect=RuntimeError("sink broke"))

        configure(sink=broken_sink)

        @trace_tool()
        async def my_tool() -> str:
            return "ok"

        result = await my_tool()
        assert result == "ok"

    async def test_broken_sink_doesnt_crash_sync(self) -> None:
        broken_sink = MagicMock()
        broken_sink.write = MagicMock(side_effect=RuntimeError("sink broke"))

        configure(sink=broken_sink)

        @trace_tool()
        def my_tool() -> str:
            return "ok"

        result = my_tool()
        assert result == "ok"
