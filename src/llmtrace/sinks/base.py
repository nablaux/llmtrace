"""Abstract base class for llmtrace sinks."""

from abc import ABC, abstractmethod
from types import TracebackType

from llmtrace._logging import get_logger
from llmtrace.models import TraceEvent

logger = get_logger()


class BaseSink(ABC):
    """Base class for all sink implementations.

    Provides async context manager support and shared error logging.
    Subclasses must implement write, flush, and close.
    """

    async def __aenter__(self) -> "BaseSink":
        """Enter the async context manager."""
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Exit the async context manager by closing the sink."""
        await self.close()

    @abstractmethod
    async def write(self, event: TraceEvent) -> None:
        """Write a single trace event to the sink."""
        ...

    @abstractmethod
    async def flush(self) -> None:
        """Flush any buffered events to the underlying storage."""
        ...

    @abstractmethod
    async def close(self) -> None:
        """Release resources and close the sink."""
        ...

    def _log_error(self, error: Exception, context: str) -> None:
        """Log a warning for sink errors without crashing the caller.

        Args:
            error: The exception that occurred.
            context: Description of what operation failed.
        """
        logger.warning("Sink error during %s: %s", context, error)
