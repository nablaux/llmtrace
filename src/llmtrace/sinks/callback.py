"""CallbackSink that invokes a user-provided function for each trace event."""

import inspect
from collections.abc import Awaitable, Callable

from llmtrace.models import TraceEvent
from llmtrace.sinks.base import BaseSink


class CallbackSink(BaseSink):
    """Sink that invokes a callback function for each trace event.

    Supports both synchronous and asynchronous callbacks. The callback type
    is detected at init time using inspect.iscoroutinefunction.
    """

    def __init__(
        self,
        callback: Callable[[TraceEvent], None] | Callable[[TraceEvent], Awaitable[None]],
    ) -> None:
        """Initialize the callback sink.

        Args:
            callback: Function to call for each event. Can be sync or async.
        """
        self._callback = callback
        self._is_async = inspect.iscoroutinefunction(callback)

    async def write(self, event: TraceEvent) -> None:
        """Invoke the callback with the given event."""
        try:
            if self._is_async:
                await self._callback(event)  # type: ignore[misc]  # union callback narrowed by _is_async
            else:
                self._callback(event)
        except Exception as exc:
            self._log_error(exc, "CallbackSink.write")

    async def flush(self) -> None:
        """No-op — callbacks have no buffer to flush."""

    async def close(self) -> None:
        """No-op — callbacks have no resources to release."""
