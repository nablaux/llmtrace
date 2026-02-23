"""Protocol definitions for llmtrace extensibility points."""

from typing import Any, Protocol

from llmtrace.models import TraceEvent

TagDict = dict[str, str]
MetadataDict = dict[str, Any]


class Sink(Protocol):
    """Async destination for trace events.

    Implementations receive trace events and persist them to a backend
    (e.g., file, database, HTTP endpoint). All methods are async to
    support non-blocking I/O.
    """

    async def write(self, event: TraceEvent) -> None:
        """Write a single trace event to the sink."""
        ...

    async def flush(self) -> None:
        """Flush any buffered events to the underlying storage."""
        ...

    async def close(self) -> None:
        """Release resources and close the sink."""
        ...


class SinkSync(Protocol):
    """Synchronous destination for trace events.

    Blocking variant of Sink for use in synchronous code paths.
    """

    def write(self, event: TraceEvent) -> None:
        """Write a single trace event to the sink."""
        ...

    def flush(self) -> None:
        """Flush any buffered events to the underlying storage."""
        ...

    def close(self) -> None:
        """Release resources and close the sink."""
        ...


class Instrumentor(Protocol):
    """Hooks into an LLM provider's client to capture trace events.

    Each instrumentor targets a specific provider (e.g., OpenAI, Anthropic)
    and monkey-patches or wraps the provider's API calls to emit TraceEvents.
    """

    @property
    def provider_name(self) -> str:
        """Return the name of the LLM provider this instrumentor targets."""
        ...

    def instrument(self) -> None:
        """Activate instrumentation for the target provider."""
        ...

    def uninstrument(self) -> None:
        """Deactivate instrumentation and restore original behavior."""
        ...


class Enricher(Protocol):
    """Callable that augments a TraceEvent with additional data.

    Enrichers are applied to events before they reach sinks. They can
    add tags, metadata, or modify fields (e.g., attach cost estimates,
    environment info, or user context).
    """

    def __call__(self, event: TraceEvent) -> TraceEvent:
        """Enrich a trace event and return the modified event."""
        ...
