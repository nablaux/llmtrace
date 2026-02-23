"""Tests for llmtrace capture decorator."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from llmtrace.capture.decorator import _current_parent_id, _current_trace_id, _is_retryable, trace
from llmtrace.config import configure, reset
from llmtrace.models import TraceEvent
from llmtrace.sinks.callback import CallbackSink


@pytest.fixture(autouse=True)
def _clean_config() -> Any:
    """Reset config and contextvars before each test."""
    reset()
    parent_token = _current_parent_id.set(None)
    trace_token = _current_trace_id.set(None)
    yield
    _current_parent_id.reset(parent_token)
    _current_trace_id.reset(trace_token)
    reset()


def _make_sink() -> tuple[CallbackSink, list[TraceEvent]]:
    """Create a CallbackSink that captures events into a list."""
    events: list[TraceEvent] = []

    def on_event(event: TraceEvent) -> None:
        events.append(event)

    return CallbackSink(on_event), events


class MockAnthropicUsage:
    def __init__(self, input_tokens: int = 100, output_tokens: int = 50) -> None:
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


class MockAnthropicMessage:
    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        usage: MockAnthropicUsage | None = None,
    ) -> None:
        self.model = model
        self.usage = usage or MockAnthropicUsage()
        self.content: list[Any] = []

    def model_dump(self) -> dict[str, Any]:
        return {"model": self.model, "content": []}


# ── Async function tests ─────────────────────────────────────────────────


class TestAsyncTrace:
    """Tests for tracing async functions."""

    async def test_emits_trace_event(self) -> None:
        sink, events = _make_sink()
        configure(sink=sink)

        @trace()
        async def my_llm_call(model: str = "test") -> MockAnthropicMessage:
            return MockAnthropicMessage()

        result = await my_llm_call(model="test")
        assert isinstance(result, MockAnthropicMessage)
        assert len(events) == 1
        assert events[0].latency_ms > 0

    async def test_latency_measured(self) -> None:
        sink, events = _make_sink()
        configure(sink=sink)

        @trace()
        async def slow_call() -> MockAnthropicMessage:
            import asyncio

            await asyncio.sleep(0.01)
            return MockAnthropicMessage()

        await slow_call()
        assert len(events) == 1
        assert events[0].latency_ms >= 10

    async def test_return_value_preserved(self) -> None:
        sink, _ = _make_sink()
        configure(sink=sink)

        msg = MockAnthropicMessage(model="custom-model")

        @trace()
        async def my_call() -> MockAnthropicMessage:
            return msg

        result = await my_call()
        assert result is msg


# ── Sync function tests ──────────────────────────────────────────────────


class TestSyncTrace:
    """Tests for tracing sync functions."""

    async def test_emits_trace_event(self) -> None:
        sink, events = _make_sink()
        configure(sink=sink)

        @trace()
        def my_sync_call(model: str = "test") -> MockAnthropicMessage:
            return MockAnthropicMessage()

        result = my_sync_call(model="test")
        assert isinstance(result, MockAnthropicMessage)
        # Sync path with async sink fires in background; give it time
        import asyncio

        await asyncio.sleep(0.1)
        assert len(events) == 1
        assert events[0].latency_ms > 0

    async def test_return_value_preserved(self) -> None:
        sink, _ = _make_sink()
        configure(sink=sink)

        msg = MockAnthropicMessage(model="sync-model")

        @trace()
        def my_call() -> MockAnthropicMessage:
            return msg

        result = my_call()
        assert result is msg


# ── Exception handling ───────────────────────────────────────────────────


class TestExceptionHandling:
    """Tests that exceptions are captured and re-raised."""

    async def test_async_exception_captured(self) -> None:
        sink, events = _make_sink()
        configure(sink=sink)

        @trace()
        async def failing_call() -> None:
            msg = "API error"
            raise ValueError(msg)

        with pytest.raises(ValueError, match="API error"):
            await failing_call()

        assert len(events) == 1
        assert events[0].error is not None
        assert events[0].error.error_type == "ValueError"
        assert events[0].error.message == "API error"
        assert events[0].error.stack_trace is not None

    async def test_sync_exception_captured(self) -> None:
        sink, events = _make_sink()
        configure(sink=sink)

        @trace()
        def failing_sync() -> None:
            msg = "sync failure"
            raise RuntimeError(msg)

        with pytest.raises(RuntimeError, match="sync failure"):
            failing_sync()

        import asyncio

        await asyncio.sleep(0.1)
        assert len(events) == 1
        assert events[0].error is not None
        assert events[0].error.error_type == "RuntimeError"


# ── Provider auto-detection ──────────────────────────────────────────────


class TestProviderAutoDetection:
    """Tests for auto-detecting provider from response type."""

    async def test_anthropic_detection(self) -> None:
        sink, events = _make_sink()
        configure(sink=sink)

        # Create a mock that looks like it came from the anthropic module
        mock_response = MockAnthropicMessage()
        mock_response.__class__ = type(
            "Message", (), {"__module__": "anthropic.types", **vars(mock_response)}
        )
        # Override model_dump since we changed the class
        mock_response.model_dump = lambda: {"model": "claude-sonnet-4-20250514", "content": []}  # type: ignore[attr-defined]

        @trace()
        async def call_anthropic() -> Any:
            return mock_response

        await call_anthropic()
        assert len(events) == 1
        assert events[0].provider == "anthropic"

    async def test_explicit_provider_override(self) -> None:
        sink, events = _make_sink()
        configure(sink=sink)

        @trace(provider="custom-provider")
        async def call_custom() -> MockAnthropicMessage:
            return MockAnthropicMessage()

        await call_custom()
        assert len(events) == 1
        assert events[0].provider == "custom-provider"


# ── Tag and metadata merging ─────────────────────────────────────────────


class TestTagMerging:
    """Tests for merging config and decorator tags."""

    async def test_config_tags_applied(self) -> None:
        sink, events = _make_sink()
        configure(sink=sink, default_tags={"env": "test", "team": "ml"})

        @trace()
        async def my_call() -> MockAnthropicMessage:
            return MockAnthropicMessage()

        await my_call()
        assert events[0].tags["env"] == "test"
        assert events[0].tags["team"] == "ml"

    async def test_decorator_tags_override_config(self) -> None:
        sink, events = _make_sink()
        configure(sink=sink, default_tags={"env": "test"})

        @trace(tags={"env": "prod", "feature": "chat"})
        async def my_call() -> MockAnthropicMessage:
            return MockAnthropicMessage()

        await my_call()
        assert events[0].tags["env"] == "prod"
        assert events[0].tags["feature"] == "chat"

    async def test_metadata_merging(self) -> None:
        sink, events = _make_sink()
        configure(sink=sink, default_metadata={"version": "1.0"})

        @trace(metadata={"request_id": "abc-123"})
        async def my_call() -> MockAnthropicMessage:
            return MockAnthropicMessage()

        await my_call()
        assert events[0].metadata["version"] == "1.0"
        assert events[0].metadata["request_id"] == "abc-123"


# ── Parent ID propagation ────────────────────────────────────────────────


class TestParentIdPropagation:
    """Tests for nested trace parent_id propagation (OTel conventions)."""

    async def test_nested_traces_set_parent_id(self) -> None:
        sink, events = _make_sink()
        configure(sink=sink)

        @trace()
        async def inner_call() -> MockAnthropicMessage:
            return MockAnthropicMessage()

        @trace()
        async def outer_call() -> MockAnthropicMessage:
            await inner_call()
            return MockAnthropicMessage()

        await outer_call()
        assert len(events) == 2
        inner_event = events[0]
        outer_event = events[1]
        # OTel conventions:
        # - All events in a trace share the same trace_id
        # - Each event has a unique span_id
        # - Child events' parent_id references the parent's span_id
        # - Root span has parent_id = None
        assert inner_event.trace_id == outer_event.trace_id
        assert inner_event.span_id != outer_event.span_id
        assert inner_event.parent_id == outer_event.span_id
        assert outer_event.parent_id is None


# ── Sample rate ──────────────────────────────────────────────────────────


class TestSampleRate:
    """Tests for sample rate filtering."""

    async def test_sample_rate_zero_skips_all(self) -> None:
        sink, events = _make_sink()
        configure(sink=sink, sample_rate=0.0)

        @trace()
        async def my_call() -> MockAnthropicMessage:
            return MockAnthropicMessage()

        for _ in range(10):
            await my_call()

        assert len(events) == 0

    async def test_sample_rate_one_captures_all(self) -> None:
        sink, events = _make_sink()
        configure(sink=sink, sample_rate=1.0)

        @trace()
        async def my_call() -> MockAnthropicMessage:
            return MockAnthropicMessage()

        for _ in range(5):
            await my_call()

        assert len(events) == 5


# ── Capture request/response redaction ───────────────────────────────────


class TestCaptureRedaction:
    """Tests for capture_request and capture_response config."""

    async def test_capture_request_false(self) -> None:
        sink, events = _make_sink()
        configure(sink=sink, capture_request=False)

        @trace()
        async def my_call(model: str = "test") -> MockAnthropicMessage:
            return MockAnthropicMessage()

        await my_call(model="test")
        assert len(events) == 1
        assert events[0].request == {}

    async def test_capture_response_false(self) -> None:
        sink, events = _make_sink()
        configure(sink=sink, capture_response=False)

        @trace()
        async def my_call() -> MockAnthropicMessage:
            return MockAnthropicMessage()

        await my_call()
        assert len(events) == 1
        assert events[0].response == {}


# ── _is_retryable ────────────────────────────────────────────────────────


class TestIsRetryable:
    """Tests for _is_retryable helper."""

    def test_rate_limit_error(self) -> None:
        exc = Exception("RateLimit exceeded")
        assert _is_retryable(exc) is True

    def test_timeout_error(self) -> None:
        exc = TimeoutError("connection timed out")
        assert _is_retryable(exc) is True

    def test_normal_error_not_retryable(self) -> None:
        exc = ValueError("bad value")
        assert _is_retryable(exc) is False


# ── Sink error safety ────────────────────────────────────────────────────


class TestSinkErrorSafety:
    """Tests that sink errors never crash the user's code."""

    async def test_broken_sink_doesnt_crash(self) -> None:
        broken_sink = MagicMock()
        broken_sink.write = MagicMock(side_effect=RuntimeError("sink broke"))

        configure(sink=broken_sink)

        @trace()
        async def my_call() -> MockAnthropicMessage:
            return MockAnthropicMessage()

        # Should not raise
        result = await my_call()
        assert isinstance(result, MockAnthropicMessage)
