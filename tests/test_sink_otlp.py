"""Tests for OTLPSink."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from decimal import Decimal
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import (
    SimpleSpanProcessor,
    SpanExporter,
    SpanExportResult,
)
from opentelemetry.trace import SpanKind, StatusCode

from llmtrace.models import Cost, ErrorTrace, TokenUsage, ToolCallTrace, TraceEvent
from llmtrace.sinks.otlp import OTLPSink, _uuid_to_span_id, _uuid_to_trace_id

if TYPE_CHECKING:
    from collections.abc import Sequence

    import pytest

# ---------------------------------------------------------------------------
# In-memory exporter (OTel SDK may not ship one in all versions)
# ---------------------------------------------------------------------------


class _InMemorySpanExporter(SpanExporter):
    """Minimal in-memory span exporter for testing."""

    def __init__(self) -> None:
        self._spans: list[ReadableSpan] = []
        self._stopped = False

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        self._spans.extend(spans)
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        self._stopped = True

    def get_finished_spans(self) -> list[ReadableSpan]:
        return list(self._spans)

    def clear(self) -> None:
        self._spans.clear()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_sink(*, capture_content: bool = False) -> tuple[OTLPSink, _InMemorySpanExporter]:
    """Create an OTLPSink wired to an in-memory exporter for assertions."""
    sink = OTLPSink(
        endpoint="http://localhost:4318",
        service_name="test-service",
        capture_content=capture_content,
    )

    # Replace internals with test-friendly components
    exporter = _InMemorySpanExporter()
    resource = Resource.create({"service.name": "test-service"})
    provider = TracerProvider(resource=resource)
    processor = SimpleSpanProcessor(exporter)
    provider.add_span_processor(processor)
    tracer = provider.get_tracer("llmtrace")

    sink._provider = provider
    sink._processor = processor  # type: ignore[assignment]  # SimpleSpanProcessor != BatchSpanProcessor
    sink._tracer = tracer
    sink._exporter = exporter

    return sink, exporter


def _make_event(**kwargs: object) -> TraceEvent:
    """Create a TraceEvent with sensible defaults for testing."""
    defaults: dict[str, object] = {
        "trace_id": uuid4(),
        "span_id": uuid4(),
        "provider": "openai",
        "model": "gpt-4",
        "latency_ms": 250.0,
        "timestamp": datetime(2024, 6, 15, 14, 30, 45, tzinfo=UTC),
    }
    defaults.update(kwargs)
    return TraceEvent(**defaults)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Tests: UUID conversion helpers
# ---------------------------------------------------------------------------


class TestUuidConversion:
    """Tests for UUID-to-OTel ID conversion functions."""

    def test_uuid_to_trace_id_is_128_bit_int(self) -> None:
        uid = UUID("12345678-1234-5678-1234-567812345678")
        trace_id = _uuid_to_trace_id(uid)
        assert trace_id == uid.int
        assert isinstance(trace_id, int)
        assert 0 < trace_id < (1 << 128)

    def test_uuid_to_span_id_is_lower_64_bits(self) -> None:
        uid = UUID("12345678-1234-5678-1234-567812345678")
        span_id = _uuid_to_span_id(uid)
        expected = uid.int & ((1 << 64) - 1)
        assert span_id == expected
        assert isinstance(span_id, int)
        assert 0 < span_id < (1 << 64)

    def test_different_uuids_produce_different_ids(self) -> None:
        uid1 = uuid4()
        uid2 = uuid4()
        assert _uuid_to_trace_id(uid1) != _uuid_to_trace_id(uid2)

    def test_span_id_nonzero(self) -> None:
        """Span IDs must be non-zero for OTel validity."""
        # Use a UUID where the lower 64 bits are definitely nonzero
        uid = UUID("ffffffff-ffff-ffff-ffff-ffffffffffff")
        assert _uuid_to_span_id(uid) != 0


# ---------------------------------------------------------------------------
# Tests: Span creation and attributes
# ---------------------------------------------------------------------------


class TestOTLPSinkSpanCreation:
    """Tests for basic span creation and gen_ai.* attributes."""

    async def test_span_name_format(self) -> None:
        sink, exporter = _make_sink()
        event = _make_event(model="gpt-4o")

        await sink.write(event)

        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].name == "chat gpt-4o"
        await sink.close()

    async def test_span_kind_is_client(self) -> None:
        sink, exporter = _make_sink()
        await sink.write(_make_event())

        spans = exporter.get_finished_spans()
        assert spans[0].kind == SpanKind.CLIENT
        await sink.close()

    async def test_gen_ai_attributes(self) -> None:
        sink, exporter = _make_sink()
        event = _make_event(provider="anthropic", model="claude-3-opus")

        await sink.write(event)

        spans = exporter.get_finished_spans()
        attrs = dict(spans[0].attributes or {})
        assert attrs["gen_ai.provider.name"] == "anthropic"
        assert attrs["gen_ai.request.model"] == "claude-3-opus"
        assert attrs["gen_ai.operation.name"] == "chat"
        await sink.close()

    async def test_token_usage_attributes(self) -> None:
        sink, exporter = _make_sink()
        event = _make_event(
            token_usage=TokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150),
        )

        await sink.write(event)

        spans = exporter.get_finished_spans()
        attrs = dict(spans[0].attributes or {})
        assert attrs["gen_ai.usage.input_tokens"] == 100
        assert attrs["gen_ai.usage.output_tokens"] == 50
        await sink.close()

    async def test_no_token_usage_omits_attributes(self) -> None:
        sink, exporter = _make_sink()
        event = _make_event(token_usage=None)

        await sink.write(event)

        spans = exporter.get_finished_spans()
        attrs = dict(spans[0].attributes or {})
        assert "gen_ai.usage.input_tokens" not in attrs
        assert "gen_ai.usage.output_tokens" not in attrs
        await sink.close()


# ---------------------------------------------------------------------------
# Tests: Cost attributes
# ---------------------------------------------------------------------------


class TestOTLPSinkCostAttributes:
    """Tests for cost-related span attributes."""

    async def test_cost_attributes_present(self) -> None:
        sink, exporter = _make_sink()
        event = _make_event(
            cost=Cost(
                input_cost=Decimal("0.003"),
                output_cost=Decimal("0.006"),
                total_cost=Decimal("0.009"),
                currency="USD",
            ),
        )

        await sink.write(event)

        spans = exporter.get_finished_spans()
        attrs = dict(spans[0].attributes or {})
        assert attrs["llmtrace.cost.total"] == "0.009"
        assert attrs["llmtrace.cost.input"] == "0.003"
        assert attrs["llmtrace.cost.output"] == "0.006"
        assert attrs["llmtrace.cost.currency"] == "USD"
        await sink.close()

    async def test_no_cost_omits_attributes(self) -> None:
        sink, exporter = _make_sink()
        event = _make_event(cost=None)

        await sink.write(event)

        spans = exporter.get_finished_spans()
        attrs = dict(spans[0].attributes or {})
        assert "llmtrace.cost.total" not in attrs
        assert "llmtrace.cost.input" not in attrs
        assert "llmtrace.cost.output" not in attrs
        assert "llmtrace.cost.currency" not in attrs
        await sink.close()


# ---------------------------------------------------------------------------
# Tests: Error handling
# ---------------------------------------------------------------------------


class TestOTLPSinkErrors:
    """Tests for error trace handling."""

    async def test_error_sets_span_status(self) -> None:
        sink, exporter = _make_sink()
        event = _make_event(
            error=ErrorTrace(
                error_type="RateLimitError",
                message="Rate limit exceeded",
            ),
        )

        await sink.write(event)

        spans = exporter.get_finished_spans()
        span = spans[0]
        assert span.status.status_code == StatusCode.ERROR
        assert span.status.description == "Rate limit exceeded"
        attrs = dict(span.attributes or {})
        assert attrs["error.type"] == "RateLimitError"
        await sink.close()

    async def test_no_error_span_status_ok(self) -> None:
        sink, exporter = _make_sink()
        event = _make_event(error=None)

        await sink.write(event)

        spans = exporter.get_finished_spans()
        # Unset status is the default for non-error spans
        assert spans[0].status.status_code != StatusCode.ERROR
        await sink.close()


# ---------------------------------------------------------------------------
# Tests: Tool calls as child spans
# ---------------------------------------------------------------------------


class TestOTLPSinkToolCalls:
    """Tests for tool call child spans."""

    async def test_tool_calls_create_child_spans(self) -> None:
        sink, exporter = _make_sink()
        event = _make_event(
            tool_calls=[
                ToolCallTrace(
                    tool_name="get_weather",
                    arguments={"city": "London"},
                    result={"temp": 15},
                    latency_ms=50.0,
                    success=True,
                ),
                ToolCallTrace(
                    tool_name="search",
                    arguments={"q": "test"},
                    result=None,
                    latency_ms=30.0,
                    success=True,
                ),
            ],
        )

        await sink.write(event)

        spans = exporter.get_finished_spans()
        # 1 parent + 2 tool call children
        assert len(spans) == 3

        tool_spans = [s for s in spans if s.name.startswith("execute_tool")]
        assert len(tool_spans) == 2

        names = {s.name for s in tool_spans}
        assert "execute_tool get_weather" in names
        assert "execute_tool search" in names
        await sink.close()

    async def test_tool_call_span_attributes(self) -> None:
        sink, exporter = _make_sink()
        event = _make_event(
            tool_calls=[
                ToolCallTrace(
                    tool_name="get_weather",
                    arguments={"city": "London"},
                    result={"temp": 15},
                    latency_ms=50.0,
                    success=True,
                ),
            ],
        )

        await sink.write(event)

        spans = exporter.get_finished_spans()
        tool_spans = [s for s in spans if s.name.startswith("execute_tool")]
        assert len(tool_spans) == 1
        attrs = dict(tool_spans[0].attributes or {})
        assert attrs["gen_ai.operation.name"] == "execute_tool"
        assert attrs["gen_ai.tool.name"] == "get_weather"
        assert attrs["gen_ai.tool.type"] == "function"
        await sink.close()

    async def test_tool_call_failure_sets_error_status(self) -> None:
        sink, exporter = _make_sink()
        event = _make_event(
            tool_calls=[
                ToolCallTrace(
                    tool_name="failing_tool",
                    arguments={},
                    success=False,
                    error_message="Something went wrong",
                ),
            ],
        )

        await sink.write(event)

        spans = exporter.get_finished_spans()
        tool_spans = [s for s in spans if s.name.startswith("execute_tool")]
        assert tool_spans[0].status.status_code == StatusCode.ERROR
        assert tool_spans[0].status.description == "Something went wrong"
        await sink.close()

    async def test_tool_call_content_omitted_by_default(self) -> None:
        sink, exporter = _make_sink(capture_content=False)
        event = _make_event(
            tool_calls=[
                ToolCallTrace(
                    tool_name="get_weather",
                    arguments={"city": "London"},
                    result={"temp": 15},
                ),
            ],
        )

        await sink.write(event)

        spans = exporter.get_finished_spans()
        tool_spans = [s for s in spans if s.name.startswith("execute_tool")]
        attrs = dict(tool_spans[0].attributes or {})
        assert "gen_ai.tool.arguments" not in attrs
        assert "gen_ai.tool.result" not in attrs
        await sink.close()

    async def test_tool_call_content_included_when_capture_content(self) -> None:
        sink, exporter = _make_sink(capture_content=True)
        event = _make_event(
            tool_calls=[
                ToolCallTrace(
                    tool_name="get_weather",
                    arguments={"city": "London"},
                    result={"temp": 15},
                ),
            ],
        )

        await sink.write(event)

        spans = exporter.get_finished_spans()
        tool_spans = [s for s in spans if s.name.startswith("execute_tool")]
        attrs = dict(tool_spans[0].attributes or {})
        assert json.loads(attrs["gen_ai.tool.arguments"]) == {"city": "London"}  # type: ignore[arg-type]  # OTel attr is str at runtime
        assert json.loads(attrs["gen_ai.tool.result"]) == {"temp": 15}  # type: ignore[arg-type]  # OTel attr is str at runtime
        await sink.close()


# ---------------------------------------------------------------------------
# Tests: Content capture
# ---------------------------------------------------------------------------


class TestOTLPSinkContentCapture:
    """Tests for capture_content flag controlling message attributes."""

    async def test_capture_content_false_omits_messages(self) -> None:
        sink, exporter = _make_sink(capture_content=False)
        event = _make_event(
            request={"messages": [{"role": "user", "content": "Hello"}]},
            response={"content": "Hi there"},
        )

        await sink.write(event)

        spans = exporter.get_finished_spans()
        attrs = dict(spans[0].attributes or {})
        assert "gen_ai.input.messages" not in attrs
        assert "gen_ai.output.messages" not in attrs
        await sink.close()

    async def test_capture_content_true_includes_messages(self) -> None:
        sink, exporter = _make_sink(capture_content=True)
        request = {"messages": [{"role": "user", "content": "Hello"}]}
        response = {"content": "Hi there"}
        event = _make_event(request=request, response=response)

        await sink.write(event)

        spans = exporter.get_finished_spans()
        attrs = dict(spans[0].attributes or {})
        assert json.loads(attrs["gen_ai.input.messages"]) == request  # type: ignore[arg-type]  # OTel attr is str at runtime
        assert json.loads(attrs["gen_ai.output.messages"]) == response  # type: ignore[arg-type]  # OTel attr is str at runtime
        await sink.close()


# ---------------------------------------------------------------------------
# Tests: Tags and metadata
# ---------------------------------------------------------------------------


class TestOTLPSinkTagsAndMetadata:
    """Tests for tag and metadata prefixed attributes."""

    async def test_tags_as_prefixed_attributes(self) -> None:
        sink, exporter = _make_sink()
        event = _make_event(tags={"env": "production", "team": "ml"})

        await sink.write(event)

        spans = exporter.get_finished_spans()
        attrs = dict(spans[0].attributes or {})
        assert attrs["llmtrace.tag.env"] == "production"
        assert attrs["llmtrace.tag.team"] == "ml"
        await sink.close()

    async def test_metadata_as_prefixed_attributes(self) -> None:
        sink, exporter = _make_sink()
        event = _make_event(metadata={"user_id": "abc123", "session": 42})

        await sink.write(event)

        spans = exporter.get_finished_spans()
        attrs = dict(spans[0].attributes or {})
        assert attrs["llmtrace.meta.user_id"] == "abc123"
        assert attrs["llmtrace.meta.session"] == "42"
        await sink.close()

    async def test_empty_tags_and_metadata_no_attributes(self) -> None:
        sink, exporter = _make_sink()
        event = _make_event(tags={}, metadata={})

        await sink.write(event)

        spans = exporter.get_finished_spans()
        attrs = dict(spans[0].attributes or {})
        tag_keys = [k for k in attrs if k.startswith("llmtrace.tag.")]
        meta_keys = [k for k in attrs if k.startswith("llmtrace.meta.")]
        assert tag_keys == []
        assert meta_keys == []
        await sink.close()


# ---------------------------------------------------------------------------
# Tests: Lifecycle methods
# ---------------------------------------------------------------------------


class TestOTLPSinkLifecycle:
    """Tests for flush and close lifecycle."""

    async def test_flush_calls_force_flush(self) -> None:
        sink, _exporter = _make_sink()
        flushed = False
        original = sink._processor.force_flush

        def _mock_force_flush(timeout_millis: int = 30000) -> bool:
            nonlocal flushed
            flushed = True
            return original(timeout_millis)

        sink._processor.force_flush = _mock_force_flush  # type: ignore[assignment]
        await sink.flush()
        assert flushed
        await sink.close()

    async def test_close_calls_shutdown(self) -> None:
        sink, _exporter = _make_sink()
        shutdown_called = False
        original = sink._provider.shutdown

        def _mock_shutdown() -> None:
            nonlocal shutdown_called
            shutdown_called = True
            original()

        sink._provider.shutdown = _mock_shutdown  # type: ignore[method-assign]
        await sink.close()
        assert shutdown_called


# ---------------------------------------------------------------------------
# Tests: Error safety
# ---------------------------------------------------------------------------


class TestOTLPSinkErrorSafety:
    """Tests that write() never raises, even on internal error."""

    async def test_write_error_does_not_raise(self, caplog: pytest.LogCaptureFixture) -> None:
        sink, _exporter = _make_sink()

        # Sabotage the tracer to force an error
        sink._tracer = None  # type: ignore[assignment]

        with caplog.at_level(logging.WARNING, logger="llmtrace.sinks.otlp"):
            # Must NOT raise
            await sink.write(_make_event())

        assert any("Sink error" in rec.message for rec in caplog.records)
        await sink.close()


# ---------------------------------------------------------------------------
# Tests: Span timing
# ---------------------------------------------------------------------------


class TestOTLPSinkTiming:
    """Tests for span start and end time computation."""

    async def test_span_start_and_end_times(self) -> None:
        sink, exporter = _make_sink()
        ts = datetime(2024, 6, 15, 14, 30, 45, tzinfo=UTC)
        event = _make_event(timestamp=ts, latency_ms=500.0)

        await sink.write(event)

        spans = exporter.get_finished_spans()
        span = spans[0]

        # OTel uses nanoseconds since epoch
        expected_start_ns = int(ts.timestamp() * 1_000_000_000)
        expected_end_ns = expected_start_ns + int(500.0 * 1_000_000)

        assert span.start_time == expected_start_ns
        assert span.end_time == expected_end_ns
        await sink.close()
