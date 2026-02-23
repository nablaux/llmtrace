"""OTLP sink that exports TraceEvents as OpenTelemetry spans with gen_ai.* attributes."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Literal

from llmtrace._logging import get_logger
from llmtrace.sinks.base import BaseSink

if TYPE_CHECKING:
    from uuid import UUID

    from opentelemetry.context import Context as OTelContext
    from opentelemetry.sdk.trace.export import SpanExporter
    from opentelemetry.trace import Tracer as OTelTracer

    from llmtrace.models import ToolCallTrace, TraceEvent

logger = get_logger()


def _uuid_to_trace_id(uid: UUID) -> int:
    """Convert a UUID to a 128-bit OpenTelemetry trace ID.

    Args:
        uid: The UUID to convert.

    Returns:
        The UUID's integer value as a 128-bit trace ID.
    """
    return uid.int


def _uuid_to_span_id(uid: UUID) -> int:
    """Convert a UUID to a 64-bit OpenTelemetry span ID.

    Uses the lower 64 bits of the UUID's integer representation.

    Args:
        uid: The UUID to convert.

    Returns:
        The lower 64 bits of the UUID's integer value.
    """
    return uid.int & ((1 << 64) - 1)


def _create_exporter(
    protocol: Literal["http/protobuf", "grpc"],
    endpoint: str,
    headers: dict[str, str] | None,
) -> SpanExporter:
    """Create an OTLP span exporter for the given protocol.

    Args:
        protocol: The OTLP transport protocol to use.
        endpoint: The collector endpoint URL.
        headers: Optional HTTP headers for authentication.

    Returns:
        A configured SpanExporter instance.

    Raises:
        ImportError: If the required exporter package is not installed.
    """
    if protocol == "grpc":
        try:
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
                OTLPSpanExporter as GrpcExporter,
            )
        except ImportError as exc:
            raise ImportError(
                "gRPC OTLP exporter requires opentelemetry-exporter-otlp-proto-grpc. "
                "Install with: pip install llmtrace[otlp-grpc]"
            ) from exc
        return GrpcExporter(endpoint=endpoint, headers=headers)
    else:
        try:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
                OTLPSpanExporter as HttpExporter,
            )
        except ImportError as exc:
            raise ImportError(
                "HTTP OTLP exporter requires opentelemetry-exporter-otlp-proto-http. "
                "Install with: pip install llmtrace[otlp]"
            ) from exc
        return HttpExporter(endpoint=endpoint, headers=headers)


class OTLPSink(BaseSink):
    """Sink that exports TraceEvents as OpenTelemetry spans via OTLP.

    Converts TraceEvent objects into OTel spans following the gen_ai.*
    semantic conventions and exports them through a BatchSpanProcessor.
    Tool calls are represented as child spans of the main LLM call span.
    """

    def __init__(
        self,
        endpoint: str = "http://localhost:4318",
        *,
        headers: dict[str, str] | None = None,
        protocol: Literal["http/protobuf", "grpc"] = "http/protobuf",
        service_name: str = "llmtrace",
        resource_attributes: dict[str, str] | None = None,
        capture_content: bool = False,
    ) -> None:
        """Initialize the OTLP sink.

        Args:
            endpoint: The OTLP collector endpoint URL.
            headers: Optional HTTP headers for authentication.
            protocol: The OTLP transport protocol to use.
            service_name: The service name to set in the OTel resource.
            resource_attributes: Additional resource attributes.
            capture_content: Whether to include request/response content in spans.

        Raises:
            ImportError: If opentelemetry-sdk is not installed.
        """
        try:
            from opentelemetry.sdk.resources import Resource
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import BatchSpanProcessor
        except ImportError as exc:
            raise ImportError(
                "OTLPSink requires opentelemetry-sdk. Install with: pip install llmtrace[otlp]"
            ) from exc

        self._capture_content = capture_content

        attrs: dict[str, str] = {"service.name": service_name}
        if resource_attributes:
            attrs.update(resource_attributes)
        resource = Resource.create(attrs)

        self._exporter: SpanExporter = _create_exporter(protocol, endpoint, headers)
        self._processor = BatchSpanProcessor(self._exporter)
        self._provider = TracerProvider(resource=resource)
        self._provider.add_span_processor(self._processor)
        self._tracer: OTelTracer = self._provider.get_tracer("llmtrace")

    async def write(self, event: TraceEvent) -> None:
        """Convert a TraceEvent to OTel span(s) and export.

        Creates a parent span for the LLM call with gen_ai.* attributes
        and optional child spans for tool calls. Never raises; errors are
        logged via _log_error.
        """
        try:
            self._write_event(event)
        except Exception as exc:
            self._log_error(exc, "OTLPSink.write")

    def _write_event(self, event: TraceEvent) -> None:
        """Internal method that performs the actual span creation.

        Args:
            event: The TraceEvent to convert and export.
        """
        from opentelemetry.trace import (
            NonRecordingSpan,
            SpanContext,
            SpanKind,
            StatusCode,
            TraceFlags,
            set_span_in_context,
        )
        from opentelemetry.trace.status import Status

        # Compute timing in nanoseconds
        start_time_ns = int(event.timestamp.timestamp() * 1_000_000_000)
        end_time_ns = start_time_ns + int(event.latency_ms * 1_000_000)

        # Build the OTel context for correct trace/parent linking
        trace_id = _uuid_to_trace_id(event.trace_id)

        if event.parent_id is not None:
            # Child span: create parent context from parent_id
            parent_span_ctx = SpanContext(
                trace_id=trace_id,
                span_id=_uuid_to_span_id(event.parent_id),
                is_remote=True,
                trace_flags=TraceFlags(0x01),
            )
            otel_ctx = set_span_in_context(NonRecordingSpan(parent_span_ctx))
        else:
            # Root span: create context with just trace_id
            root_ctx = SpanContext(
                trace_id=trace_id,
                span_id=_uuid_to_span_id(event.span_id),
                is_remote=True,
                trace_flags=TraceFlags(0x01),
            )
            otel_ctx = set_span_in_context(NonRecordingSpan(root_ctx))

        # Create the main LLM call span
        span = self._tracer.start_span(
            name=f"chat {event.model}",
            kind=SpanKind.CLIENT,
            context=otel_ctx,
            start_time=start_time_ns,
        )

        # Set gen_ai.* attributes
        span.set_attribute("gen_ai.provider.name", event.provider)
        span.set_attribute("gen_ai.request.model", event.model)
        span.set_attribute("gen_ai.operation.name", "chat")

        # Token usage
        if event.token_usage is not None:
            span.set_attribute("gen_ai.usage.input_tokens", event.token_usage.prompt_tokens)
            span.set_attribute("gen_ai.usage.output_tokens", event.token_usage.completion_tokens)

        # Cost
        if event.cost is not None:
            span.set_attribute("llmtrace.cost.total", str(event.cost.total_cost))
            span.set_attribute("llmtrace.cost.input", str(event.cost.input_cost))
            span.set_attribute("llmtrace.cost.output", str(event.cost.output_cost))
            span.set_attribute("llmtrace.cost.currency", event.cost.currency)

        # Tags
        for key, value in event.tags.items():
            span.set_attribute(f"llmtrace.tag.{key}", value)

        # Metadata
        for key, value in event.metadata.items():
            span.set_attribute(f"llmtrace.meta.{key}", str(value))

        # Content capture
        if self._capture_content:
            if event.request:
                span.set_attribute("gen_ai.input.messages", json.dumps(event.request))
            if event.response:
                span.set_attribute("gen_ai.output.messages", json.dumps(event.response))

        # Error status
        if event.error is not None:
            span.set_status(Status(StatusCode.ERROR, event.error.message))
            span.set_attribute("error.type", event.error.error_type)

        span.end(end_time=end_time_ns)

        # Create child spans for tool calls
        # Use the main span as parent for tool call spans
        main_span_ctx = span.get_span_context()
        tool_parent_ctx = set_span_in_context(NonRecordingSpan(main_span_ctx))

        for tool_call in event.tool_calls:
            self._write_tool_call_span(tool_call, tool_parent_ctx, start_time_ns)

    def _write_tool_call_span(
        self,
        tool_call: ToolCallTrace,
        parent_context: OTelContext,
        parent_start_ns: int,
    ) -> None:
        """Create a child span for a tool call.

        Args:
            tool_call: The ToolCallTrace to convert.
            parent_context: The OTel context containing the parent span.
            parent_start_ns: The parent span's start time in nanoseconds.
        """
        from opentelemetry.trace import SpanKind, StatusCode
        from opentelemetry.trace.status import Status

        tool_start_ns = parent_start_ns
        tool_end_ns = tool_start_ns
        if tool_call.latency_ms is not None:
            tool_end_ns = tool_start_ns + int(tool_call.latency_ms * 1_000_000)

        tool_span = self._tracer.start_span(
            name=f"execute_tool {tool_call.tool_name}",
            kind=SpanKind.CLIENT,
            context=parent_context,
            start_time=tool_start_ns,
        )

        tool_span.set_attribute("gen_ai.operation.name", "execute_tool")
        tool_span.set_attribute("gen_ai.tool.name", tool_call.tool_name)
        tool_span.set_attribute("gen_ai.tool.type", "function")

        if self._capture_content:
            tool_span.set_attribute("gen_ai.tool.arguments", json.dumps(tool_call.arguments))
            if tool_call.result is not None:
                tool_span.set_attribute("gen_ai.tool.result", json.dumps(tool_call.result))

        if not tool_call.success:
            tool_span.set_status(
                Status(StatusCode.ERROR, tool_call.error_message or "Tool call failed")
            )

        tool_span.end(end_time=tool_end_ns)

    async def flush(self) -> None:
        """Flush all pending spans through the processor."""
        self._processor.force_flush()

    async def close(self) -> None:
        """Shut down the TracerProvider and release resources."""
        self._provider.shutdown()
