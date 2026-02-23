"""Webhook sink that POSTs trace events as JSON batches to an HTTP endpoint."""

from __future__ import annotations

import asyncio
import contextlib
from typing import TYPE_CHECKING, Any

try:
    import httpx
except ImportError as _import_err:
    raise ImportError(
        "WebhookSink requires httpx. Install with: pip install llmtrace[webhook]"
    ) from _import_err

from llmtrace._logging import get_logger
from llmtrace.sinks.base import BaseSink

if TYPE_CHECKING:
    from llmtrace.models import TraceEvent

logger = get_logger()


class WebhookSink(BaseSink):
    """Sink that POSTs trace events as JSON batches to a webhook endpoint.

    Events are buffered and flushed either when the buffer reaches batch_size
    or every flush_interval_s seconds, whichever comes first.
    """

    def __init__(
        self,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        batch_size: int = 50,
        flush_interval_s: float = 10.0,
        max_retries: int = 3,
        timeout_s: float = 30.0,
    ) -> None:
        """Initialize the webhook sink.

        Args:
            url: Endpoint to POST trace event batches to.
            headers: Custom HTTP headers (e.g., Authorization).
            batch_size: Flush when the buffer reaches this count.
            flush_interval_s: Flush every N seconds regardless of batch size.
            max_retries: Number of retries on transient failure.
            timeout_s: HTTP request timeout in seconds.
        """
        self._url = url
        self._headers = headers or {}
        self._batch_size = batch_size
        self._flush_interval_s = flush_interval_s
        self._max_retries = max_retries
        self._timeout_s = timeout_s
        self._buffer: list[dict[str, Any]] = []
        self._lock = asyncio.Lock()
        self._client: httpx.AsyncClient | None = None
        self._flush_task: asyncio.Task[None] | None = None

    async def write(self, event: TraceEvent) -> None:
        """Append an event to the buffer and flush if batch_size is reached.

        Starts the periodic flush background task on the first call.
        """
        async with self._lock:
            self._buffer.append(event.to_dict())
            should_flush = len(self._buffer) >= self._batch_size

        if self._flush_task is None:
            await self._start_periodic_flush()

        if should_flush:
            await self.flush()

    async def flush(self) -> None:
        """Flush all buffered events to the webhook endpoint."""
        async with self._lock:
            if not self._buffer:
                return
            events = self._buffer
            self._buffer = []

        await self._send_batch(events)

    async def _send_batch(self, events: list[dict[str, Any]]) -> None:
        """POST a batch of events to the webhook endpoint with retries.

        Retries on 5xx HTTP errors and network errors with exponential backoff.
        On persistent failure, logs a warning and drops the batch.
        """
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self._timeout_s)

        delays = [2**i for i in range(self._max_retries)]
        last_error: Exception | None = None

        for attempt, delay in enumerate(delays):
            try:
                response = await self._client.post(
                    self._url,
                    json=events,
                    headers=self._headers,
                )
                response.raise_for_status()
                return
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code < 500:
                    self._log_error(exc, f"send batch ({len(events)} events)")
                    return
                last_error = exc
            except httpx.TransportError as exc:
                last_error = exc

            if attempt < len(delays) - 1:
                await asyncio.sleep(delay)

        if last_error is not None:
            self._log_error(last_error, f"send batch ({len(events)} events, all retries exhausted)")

    async def _start_periodic_flush(self) -> None:
        """Start a background task that flushes the buffer periodically."""

        async def _periodic() -> None:
            while True:
                await asyncio.sleep(self._flush_interval_s)
                try:
                    await self.flush()
                except Exception as exc:
                    self._log_error(exc, "periodic flush")

        self._flush_task = asyncio.create_task(_periodic())

    async def close(self) -> None:
        """Cancel periodic flush, flush remaining events, and close the HTTP client."""
        if self._flush_task is not None:
            self._flush_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._flush_task
            self._flush_task = None

        await self.flush()

        if self._client is not None:
            await self._client.aclose()
            self._client = None
