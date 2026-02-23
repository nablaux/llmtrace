"""Tests for llmtrace models."""

import json
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from uuid import UUID

import pytest
from pydantic import ValidationError

from llmtrace.models import (
    Cost,
    ErrorTrace,
    SpanContext,
    TokenUsage,
    ToolCallTrace,
    TraceEvent,
)


class TestTokenUsage:
    """Tests for TokenUsage model."""

    def test_construction(self) -> None:
        usage = TokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150)
        assert usage.prompt_tokens == 100
        assert usage.completion_tokens == 50
        assert usage.total_tokens == 150

    def test_auto_computed_total_tokens(self) -> None:
        usage = TokenUsage(prompt_tokens=100, completion_tokens=50)
        assert usage.total_tokens == 150

    def test_explicit_total_tokens_honored(self) -> None:
        usage = TokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=200)
        assert usage.total_tokens == 200

    def test_frozen_immutability(self) -> None:
        usage = TokenUsage(prompt_tokens=100, completion_tokens=50)
        with pytest.raises(ValidationError):
            usage.prompt_tokens = 200  # type: ignore[misc]

    def test_cache_fields_optional(self) -> None:
        usage = TokenUsage(prompt_tokens=100, completion_tokens=50)
        assert usage.cache_read_tokens is None
        assert usage.cache_write_tokens is None

    def test_cache_fields_set(self) -> None:
        usage = TokenUsage(
            prompt_tokens=100,
            completion_tokens=50,
            cache_read_tokens=30,
            cache_write_tokens=10,
        )
        assert usage.cache_read_tokens == 30
        assert usage.cache_write_tokens == 10

    def test_negative_prompt_tokens_rejected(self) -> None:
        with pytest.raises(ValidationError):
            TokenUsage(prompt_tokens=-1, completion_tokens=50)

    def test_negative_completion_tokens_rejected(self) -> None:
        with pytest.raises(ValidationError):
            TokenUsage(prompt_tokens=0, completion_tokens=-1)


class TestCost:
    """Tests for Cost model."""

    def test_construction_with_decimal(self) -> None:
        cost = Cost(
            input_cost=Decimal("0.001"),
            output_cost=Decimal("0.002"),
            total_cost=Decimal("0.003"),
        )
        assert cost.input_cost == Decimal("0.001")
        assert cost.output_cost == Decimal("0.002")
        assert cost.total_cost == Decimal("0.003")

    def test_auto_computed_total_cost(self) -> None:
        cost = Cost(input_cost=Decimal("0.001"), output_cost=Decimal("0.002"))
        assert cost.total_cost == Decimal("0.003")

    def test_explicit_total_cost_honored(self) -> None:
        cost = Cost(
            input_cost=Decimal("0.001"),
            output_cost=Decimal("0.002"),
            total_cost=Decimal("0.010"),
        )
        assert cost.total_cost == Decimal("0.010")

    def test_currency_default(self) -> None:
        cost = Cost(input_cost=Decimal("0.001"), output_cost=Decimal("0.002"))
        assert cost.currency == "USD"

    def test_currency_custom(self) -> None:
        cost = Cost(
            input_cost=Decimal("0.001"),
            output_cost=Decimal("0.002"),
            currency="EUR",
        )
        assert cost.currency == "EUR"

    def test_frozen_immutability(self) -> None:
        cost = Cost(input_cost=Decimal("0.001"), output_cost=Decimal("0.002"))
        with pytest.raises(ValidationError):
            cost.input_cost = Decimal("0.999")  # type: ignore[misc]

    def test_string_input_coerced_to_decimal(self) -> None:
        cost = Cost(input_cost="0.001", output_cost="0.002")  # type: ignore[arg-type]
        assert cost.input_cost == Decimal("0.001")
        assert cost.total_cost == Decimal("0.003")


class TestToolCallTrace:
    """Tests for ToolCallTrace model."""

    def test_construction(self) -> None:
        tool = ToolCallTrace(tool_name="search", arguments={"query": "test"})
        assert tool.tool_name == "search"
        assert tool.arguments == {"query": "test"}

    def test_defaults(self) -> None:
        tool = ToolCallTrace(tool_name="search", arguments={})
        assert tool.result is None
        assert tool.latency_ms is None
        assert tool.success is True
        assert tool.error_message is None

    def test_all_fields(self) -> None:
        tool = ToolCallTrace(
            tool_name="search",
            arguments={"query": "test"},
            result={"items": [1, 2, 3]},
            latency_ms=150.5,
            success=False,
            error_message="timeout",
        )
        assert tool.result == {"items": [1, 2, 3]}
        assert tool.latency_ms == 150.5
        assert tool.success is False
        assert tool.error_message == "timeout"

    def test_frozen_immutability(self) -> None:
        tool = ToolCallTrace(tool_name="search", arguments={})
        with pytest.raises(ValidationError):
            tool.tool_name = "other"  # type: ignore[misc]


class TestErrorTrace:
    """Tests for ErrorTrace model."""

    def test_construction(self) -> None:
        error = ErrorTrace(error_type="RateLimitError", message="Too many requests")
        assert error.error_type == "RateLimitError"
        assert error.message == "Too many requests"

    def test_defaults(self) -> None:
        error = ErrorTrace(error_type="RateLimitError", message="Too many requests")
        assert error.provider_error_code is None
        assert error.is_retryable is False
        assert error.stack_trace is None

    def test_all_fields(self) -> None:
        error = ErrorTrace(
            error_type="RateLimitError",
            message="Too many requests",
            provider_error_code="429",
            is_retryable=True,
            stack_trace="Traceback...",
        )
        assert error.provider_error_code == "429"
        assert error.is_retryable is True
        assert error.stack_trace == "Traceback..."

    def test_frozen_immutability(self) -> None:
        error = ErrorTrace(error_type="RateLimitError", message="Too many requests")
        with pytest.raises(ValidationError):
            error.message = "changed"  # type: ignore[misc]


class TestTraceEvent:
    """Tests for TraceEvent model."""

    def test_minimal_construction(self) -> None:
        event = TraceEvent(provider="openai", model="gpt-4", latency_ms=100.0)
        assert event.provider == "openai"
        assert event.model == "gpt-4"
        assert event.latency_ms == 100.0

    def test_default_uuid_generation(self) -> None:
        event = TraceEvent(provider="openai", model="gpt-4", latency_ms=100.0)
        assert isinstance(event.trace_id, UUID)
        assert event.parent_id is None
        assert isinstance(event.span_id, UUID)

    def test_unique_trace_ids(self) -> None:
        e1 = TraceEvent(provider="openai", model="gpt-4", latency_ms=100.0)
        e2 = TraceEvent(provider="openai", model="gpt-4", latency_ms=100.0)
        assert e1.trace_id != e2.trace_id

    def test_default_timestamp_is_utc(self) -> None:
        event = TraceEvent(provider="openai", model="gpt-4", latency_ms=100.0)
        assert event.timestamp.tzinfo is not None

    def test_default_collections(self) -> None:
        event = TraceEvent(provider="openai", model="gpt-4", latency_ms=100.0)
        assert event.request == {}
        assert event.response == {}
        assert event.tool_calls == []
        assert event.tags == {}
        assert event.metadata == {}

    def test_mutable_fields(self) -> None:
        event = TraceEvent(provider="openai", model="gpt-4", latency_ms=100.0)
        event.tags["env"] = "test"
        assert event.tags == {"env": "test"}

    def test_negative_latency_rejected(self) -> None:
        with pytest.raises(ValidationError, match="latency_ms"):
            TraceEvent(provider="openai", model="gpt-4", latency_ms=-1.0)

    def test_to_json_produces_valid_json(self) -> None:
        event = TraceEvent(provider="openai", model="gpt-4", latency_ms=100.0)
        json_str = event.to_json()
        parsed = json.loads(json_str)
        assert parsed["provider"] == "openai"
        assert parsed["model"] == "gpt-4"
        assert parsed["latency_ms"] == 100.0

    def test_to_json_round_trip(self) -> None:
        event = TraceEvent(
            provider="anthropic",
            model="claude-3",
            latency_ms=200.0,
            token_usage=TokenUsage(prompt_tokens=10, completion_tokens=20),
            cost=Cost(input_cost=Decimal("0.001"), output_cost=Decimal("0.002")),
        )
        json_str = event.to_json()
        parsed = json.loads(json_str)
        assert parsed["token_usage"]["total_tokens"] == 30
        assert parsed["cost"]["total_cost"] == "0.003"

    def test_to_dict_json_safe_types(self) -> None:
        event = TraceEvent(provider="openai", model="gpt-4", latency_ms=100.0)
        d = event.to_dict()
        assert isinstance(d["trace_id"], str)
        assert isinstance(d["timestamp"], str)
        # Must be JSON-serializable without error
        json.dumps(d)

    def test_to_dict_with_cost_decimal(self) -> None:
        event = TraceEvent(
            provider="openai",
            model="gpt-4",
            latency_ms=100.0,
            cost=Cost(input_cost=Decimal("0.001"), output_cost=Decimal("0.002")),
        )
        d = event.to_dict()
        assert isinstance(d["cost"]["total_cost"], str)
        json.dumps(d)


class TestSpanContext:
    """Tests for SpanContext model."""

    def test_construction(self) -> None:
        now = datetime.now(UTC)
        span = SpanContext(name="test-span", started_at=now)
        assert span.name == "test-span"
        assert span.started_at == now
        assert isinstance(span.span_id, UUID)

    def test_duration_ms_calculation(self) -> None:
        start = datetime(2024, 1, 1, 0, 0, 0, tzinfo=UTC)
        end = start + timedelta(milliseconds=1500)
        span = SpanContext(name="test-span", started_at=start, ended_at=end)
        assert span.duration_ms() == 1500.0

    def test_duration_ms_none_when_not_ended(self) -> None:
        now = datetime.now(UTC)
        span = SpanContext(name="test-span", started_at=now)
        assert span.duration_ms() is None

    def test_total_cost_single_event(self) -> None:
        now = datetime.now(UTC)
        event = TraceEvent(
            provider="openai",
            model="gpt-4",
            latency_ms=100.0,
            cost=Cost(input_cost=Decimal("0.01"), output_cost=Decimal("0.02")),
        )
        span = SpanContext(name="test-span", started_at=now, events=[event])
        assert span.total_cost() == Decimal("0.03")

    def test_total_cost_nested_children(self) -> None:
        now = datetime.now(UTC)
        event1 = TraceEvent(
            provider="openai",
            model="gpt-4",
            latency_ms=100.0,
            cost=Cost(input_cost=Decimal("0.01"), output_cost=Decimal("0.02")),
        )
        event2 = TraceEvent(
            provider="anthropic",
            model="claude-3",
            latency_ms=200.0,
            cost=Cost(input_cost=Decimal("0.03"), output_cost=Decimal("0.04")),
        )
        child = SpanContext(name="child-span", started_at=now, events=[event2])
        parent = SpanContext(name="parent-span", started_at=now, events=[event1], children=[child])
        # parent event: 0.03, child event: 0.07 = 0.10
        assert parent.total_cost() == Decimal("0.10")

    def test_total_cost_no_costs(self) -> None:
        now = datetime.now(UTC)
        span = SpanContext(name="test-span", started_at=now)
        assert span.total_cost() == Decimal("0")

    def test_total_tokens_aggregation(self) -> None:
        now = datetime.now(UTC)
        event1 = TraceEvent(
            provider="openai",
            model="gpt-4",
            latency_ms=100.0,
            token_usage=TokenUsage(prompt_tokens=100, completion_tokens=50),
        )
        event2 = TraceEvent(
            provider="anthropic",
            model="claude-3",
            latency_ms=200.0,
            token_usage=TokenUsage(prompt_tokens=200, completion_tokens=100),
        )
        child = SpanContext(name="child-span", started_at=now, events=[event2])
        parent = SpanContext(name="parent-span", started_at=now, events=[event1], children=[child])
        # event1: 150, event2: 300 = 450
        assert parent.total_tokens() == 450

    def test_total_tokens_no_usage(self) -> None:
        now = datetime.now(UTC)
        span = SpanContext(name="test-span", started_at=now)
        assert span.total_tokens() == 0


class TestValidationErrors:
    """Tests for validation error handling."""

    def test_negative_latency_rejected(self) -> None:
        with pytest.raises(ValidationError, match="latency_ms"):
            TraceEvent(provider="openai", model="gpt-4", latency_ms=-1.0)

    def test_negative_prompt_tokens_rejected(self) -> None:
        with pytest.raises(ValidationError, match="prompt_tokens"):
            TokenUsage(prompt_tokens=-1, completion_tokens=0)

    def test_invalid_type_for_provider(self) -> None:
        with pytest.raises(ValidationError):
            TraceEvent(provider=123, model="gpt-4", latency_ms=100.0)  # type: ignore[arg-type]


class TestSerialization:
    """Tests for serialization methods."""

    def test_to_json_valid_json_with_nested_models(self) -> None:
        event = TraceEvent(
            provider="openai",
            model="gpt-4",
            latency_ms=100.0,
            token_usage=TokenUsage(prompt_tokens=10, completion_tokens=20),
            cost=Cost(input_cost=Decimal("0.001"), output_cost=Decimal("0.002")),
            tool_calls=[ToolCallTrace(tool_name="search", arguments={"q": "test"})],
            error=ErrorTrace(error_type="TestError", message="test"),
        )
        json_str = event.to_json()
        parsed = json.loads(json_str)
        assert isinstance(parsed, dict)
        assert parsed["provider"] == "openai"
        assert parsed["token_usage"]["prompt_tokens"] == 10
        assert parsed["tool_calls"][0]["tool_name"] == "search"

    def test_to_dict_no_raw_uuid(self) -> None:
        event = TraceEvent(provider="openai", model="gpt-4", latency_ms=100.0)
        d = event.to_dict()
        assert isinstance(d["trace_id"], str)
        # Verify it parses back to a valid UUID
        UUID(d["trace_id"])

    def test_to_dict_no_raw_datetime(self) -> None:
        event = TraceEvent(provider="openai", model="gpt-4", latency_ms=100.0)
        d = event.to_dict()
        assert isinstance(d["timestamp"], str)

    def test_to_dict_fully_json_serializable(self) -> None:
        event = TraceEvent(
            provider="openai",
            model="gpt-4",
            latency_ms=100.0,
            token_usage=TokenUsage(prompt_tokens=10, completion_tokens=20),
            cost=Cost(input_cost=Decimal("0.001"), output_cost=Decimal("0.002")),
        )
        d = event.to_dict()
        result = json.dumps(d)
        assert isinstance(result, str)
