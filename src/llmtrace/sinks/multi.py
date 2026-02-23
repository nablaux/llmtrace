"""MultiSink that dispatches trace events to multiple child sinks concurrently."""

import asyncio

from llmtrace.models import TraceEvent
from llmtrace.sinks.base import BaseSink


class MultiSink(BaseSink):
    """Fan-out sink that dispatches events to multiple child sinks.

    All operations (write, flush, close) are dispatched concurrently via
    asyncio.gather. Errors in individual sinks are logged but never propagated,
    ensuring one failing sink cannot break others.
    """

    def __init__(self, sinks: list[BaseSink]) -> None:
        """Initialize the multi-sink.

        Args:
            sinks: List of child sinks to dispatch events to.
        """
        self._sinks = sinks

    async def _dispatch(self, method: str, *args: object) -> None:
        """Call a method on all child sinks concurrently, logging any errors."""
        results = await asyncio.gather(
            *(getattr(sink, method)(*args) for sink in self._sinks),
            return_exceptions=True,
        )
        for result in results:
            if isinstance(result, Exception):
                self._log_error(result, f"MultiSink.{method}")

    async def write(self, event: TraceEvent) -> None:
        """Write an event to all child sinks concurrently."""
        await self._dispatch("write", event)

    async def flush(self) -> None:
        """Flush all child sinks concurrently."""
        await self._dispatch("flush")

    async def close(self) -> None:
        """Close all child sinks concurrently."""
        await self._dispatch("close")
