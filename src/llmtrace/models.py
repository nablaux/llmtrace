"""Pydantic v2 models for llmtrace structured tracing."""

from datetime import UTC, datetime
from decimal import Decimal
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator


class TokenUsage(BaseModel):
    """Token usage statistics for an LLM call.

    Tracks prompt, completion, and cache token counts. Total tokens
    are auto-computed from prompt + completion if not explicitly provided.
    """

    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=0)
    cache_read_tokens: int | None = None
    cache_write_tokens: int | None = None

    model_config = ConfigDict(frozen=True)

    @model_validator(mode="before")
    @classmethod
    def _compute_total_tokens(cls, data: Any) -> Any:
        """Compute total_tokens as sum of prompt and completion if not provided."""
        if isinstance(data, dict) and data.get("total_tokens") is None:
            prompt = data.get("prompt_tokens", 0)
            completion = data.get("completion_tokens", 0)
            data["total_tokens"] = prompt + completion
        return data


class Cost(BaseModel):
    """Cost breakdown for an LLM call.

    All monetary values use Decimal for exact arithmetic.
    Total cost is auto-computed from input + output if not provided.
    """

    input_cost: Decimal
    output_cost: Decimal
    total_cost: Decimal
    currency: str = "USD"

    model_config = ConfigDict(frozen=True)

    @model_validator(mode="before")
    @classmethod
    def _compute_total_cost(cls, data: Any) -> Any:
        """Compute total_cost as sum of input and output cost if not provided."""
        if isinstance(data, dict) and data.get("total_cost") is None:
            input_cost = Decimal(str(data.get("input_cost", 0)))
            output_cost = Decimal(str(data.get("output_cost", 0)))
            data["total_cost"] = input_cost + output_cost
        return data


class ToolCallTrace(BaseModel):
    """Trace data for a single tool call within an LLM response."""

    tool_name: str
    arguments: dict[str, Any]
    result: Any | None = None
    latency_ms: float | None = None
    success: bool = True
    error_message: str | None = None

    model_config = ConfigDict(frozen=True)


class ErrorTrace(BaseModel):
    """Error information captured during an LLM call."""

    error_type: str
    message: str
    provider_error_code: str | None = None
    is_retryable: bool = False
    stack_trace: str | None = None

    model_config = ConfigDict(frozen=True)


class TraceEvent(BaseModel):
    """A single trace event representing one LLM API call.

    TraceEvent is mutable because enrichers may modify it after creation.
    Provides serialization methods for JSON output.
    """

    trace_id: UUID = Field(default_factory=uuid4)
    parent_id: UUID | None = None
    span_id: UUID = Field(default_factory=uuid4)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    provider: str
    model: str
    request: dict[str, Any] = Field(default_factory=dict)
    response: dict[str, Any] = Field(default_factory=dict)
    token_usage: TokenUsage | None = None
    cost: Cost | None = None
    latency_ms: float = Field(ge=0)
    tool_calls: list[ToolCallTrace] = Field(default_factory=list)
    error: ErrorTrace | None = None
    tags: dict[str, str] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(frozen=False)

    def to_json(self) -> str:
        """Serialize to a JSON string.

        Returns JSON with 2-space indent, serializing UUID and datetime values.
        """
        return self.model_dump_json(indent=2)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dictionary.

        Returns a dict with all values converted to JSON-compatible types
        (UUIDs as strings, datetimes as ISO 8601 strings, Decimals as strings).
        """
        return self.model_dump(mode="json")


class SpanContext(BaseModel):
    """A span representing a logical grouping of trace events.

    Spans can be nested to form a tree structure for complex operations.
    """

    span_id: UUID = Field(default_factory=uuid4)
    parent_span_id: UUID | None = None
    name: str
    started_at: datetime
    ended_at: datetime | None = None
    events: list[TraceEvent] = Field(default_factory=list)
    children: list["SpanContext"] = Field(default_factory=list)
    tags: dict[str, str] = Field(default_factory=dict)
    annotations: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(frozen=False)

    def duration_ms(self) -> float | None:
        """Calculate span duration in milliseconds.

        Returns None if the span has not ended yet.
        """
        if self.ended_at is None:
            return None
        delta = self.ended_at - self.started_at
        return delta.total_seconds() * 1000

    def total_cost(self) -> Decimal:
        """Sum of all event costs, recursive into children.

        Returns Decimal("0") if no events have cost information.
        """
        total = Decimal("0")
        for event in self.events:
            if event.cost is not None:
                total += event.cost.total_cost
        for child in self.children:
            total += child.total_cost()
        return total

    def total_tokens(self) -> int:
        """Sum of all event total_tokens, recursive into children.

        Returns 0 if no events have token usage information.
        """
        total = 0
        for event in self.events:
            if event.token_usage is not None:
                total += event.token_usage.total_tokens
        for child in self.children:
            total += child.total_tokens()
        return total
