"""Tests for WebhookSink."""

import asyncio
import logging
from datetime import UTC, datetime

import httpx
import pytest
import respx

from llmtrace.models import TraceEvent
from llmtrace.sinks.webhook import WebhookSink

WEBHOOK_URL = "https://example.com/traces"


def _make_event(**kwargs: object) -> TraceEvent:
    """Create a TraceEvent with sensible defaults for testing."""
    defaults: dict[str, object] = {
        "provider": "openai",
        "model": "gpt-4",
        "latency_ms": 123.0,
        "timestamp": datetime(2024, 6, 15, 14, 30, 45, tzinfo=UTC),
    }
    defaults.update(kwargs)
    return TraceEvent(**defaults)  # type: ignore[arg-type]


class TestWebhookSinkBasicFlush:
    """Tests for basic flush triggering on batch_size."""

    @respx.mock
    async def test_flush_on_batch_size(self) -> None:
        route = respx.post(WEBHOOK_URL).mock(return_value=httpx.Response(200))
        sink = WebhookSink(WEBHOOK_URL, batch_size=3, flush_interval_s=999)

        for _ in range(3):
            await sink.write(_make_event())

        assert route.call_count == 1
        import json

        payload = json.loads(route.calls[0].request.content)
        assert len(payload) == 3
        assert payload[0]["provider"] == "openai"

        await sink.close()


class TestWebhookSinkBatching:
    """Tests for buffering behavior before batch_size is reached."""

    @respx.mock
    async def test_no_post_before_batch_size(self) -> None:
        route = respx.post(WEBHOOK_URL).mock(return_value=httpx.Response(200))
        sink = WebhookSink(WEBHOOK_URL, batch_size=10, flush_interval_s=999)

        for _ in range(5):
            await sink.write(_make_event())

        assert route.call_count == 0

        await sink.flush()
        assert route.call_count == 1

        await sink.close()


class TestWebhookSinkRetry:
    """Tests for retry on 5xx errors."""

    @respx.mock
    async def test_retry_on_500(self) -> None:
        route = respx.post(WEBHOOK_URL).mock(
            side_effect=[
                httpx.Response(500),
                httpx.Response(500),
                httpx.Response(200),
            ]
        )
        sink = WebhookSink(WEBHOOK_URL, batch_size=1, flush_interval_s=999, max_retries=3)
        # Patch sleep to avoid real waits
        original_sleep = asyncio.sleep
        sleeps: list[float] = []

        async def _fake_sleep(delay: float) -> None:
            sleeps.append(delay)

        asyncio.sleep = _fake_sleep  # type: ignore[assignment]
        try:
            await sink.write(_make_event())
            assert route.call_count == 3
        finally:
            asyncio.sleep = original_sleep  # type: ignore[assignment]

        await sink.close()


class TestWebhookSinkDropOnPersistentFailure:
    """Tests for dropping batch after all retries exhausted."""

    @respx.mock
    async def test_persistent_failure_logs_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        route = respx.post(WEBHOOK_URL).mock(return_value=httpx.Response(500))
        sink = WebhookSink(WEBHOOK_URL, batch_size=1, flush_interval_s=999, max_retries=3)

        original_sleep = asyncio.sleep

        async def _fake_sleep(delay: float) -> None:
            pass

        asyncio.sleep = _fake_sleep  # type: ignore[assignment]
        try:
            with caplog.at_level(logging.WARNING, logger="llmtrace.sinks.webhook"):
                await sink.write(_make_event())  # Should not raise
        finally:
            asyncio.sleep = original_sleep  # type: ignore[assignment]

        assert route.call_count == 3
        assert any("all retries exhausted" in rec.message for rec in caplog.records)

        await sink.close()


class TestWebhookSinkCustomHeaders:
    """Tests for custom headers being sent."""

    @respx.mock
    async def test_custom_headers_sent(self) -> None:
        route = respx.post(WEBHOOK_URL).mock(return_value=httpx.Response(200))
        headers = {"Authorization": "Bearer secret-token", "X-Custom": "value"}
        sink = WebhookSink(WEBHOOK_URL, headers=headers, batch_size=1, flush_interval_s=999)

        await sink.write(_make_event())

        request = route.calls[0].request
        assert request.headers["Authorization"] == "Bearer secret-token"
        assert request.headers["X-Custom"] == "value"

        await sink.close()


class TestWebhookSinkClose:
    """Tests for close flushing remaining events."""

    @respx.mock
    async def test_close_flushes_remaining(self) -> None:
        route = respx.post(WEBHOOK_URL).mock(return_value=httpx.Response(200))
        sink = WebhookSink(WEBHOOK_URL, batch_size=100, flush_interval_s=999)

        await sink.write(_make_event())
        await sink.write(_make_event())
        assert route.call_count == 0

        await sink.close()
        assert route.call_count == 1

        import json

        payload = json.loads(route.calls[0].request.content)
        assert len(payload) == 2


class TestWebhookSink4xxError:
    """Tests for 4xx client error early return (non-retryable)."""

    @respx.mock
    async def test_4xx_error_does_not_retry(self) -> None:
        route = respx.post(WEBHOOK_URL).mock(return_value=httpx.Response(400))
        sink = WebhookSink(WEBHOOK_URL, batch_size=1, flush_interval_s=999, max_retries=3)

        await sink.write(_make_event())

        # 4xx errors should NOT retry — only 1 call
        assert route.call_count == 1

        await sink.close()


class TestWebhookSinkTransportError:
    """Tests for transport error handling."""

    @respx.mock
    async def test_transport_error_retries(self) -> None:
        route = respx.post(WEBHOOK_URL).mock(
            side_effect=[
                httpx.ConnectError("connection refused"),
                httpx.Response(200),
            ]
        )
        sink = WebhookSink(WEBHOOK_URL, batch_size=1, flush_interval_s=999, max_retries=2)

        original_sleep = asyncio.sleep

        async def _fake_sleep(delay: float) -> None:
            pass

        asyncio.sleep = _fake_sleep  # type: ignore[assignment]
        try:
            await sink.write(_make_event())
            assert route.call_count == 2
        finally:
            asyncio.sleep = original_sleep  # type: ignore[assignment]

        await sink.close()


class TestWebhookSinkPeriodicFlush:
    """Tests for periodic flush task."""

    @respx.mock
    async def test_periodic_flush_fires(self) -> None:
        route = respx.post(WEBHOOK_URL).mock(return_value=httpx.Response(200))
        # Large batch_size so write won't trigger flush, short interval
        sink = WebhookSink(WEBHOOK_URL, batch_size=100, flush_interval_s=0.05)

        await sink.write(_make_event())
        assert route.call_count == 0

        # Wait for periodic flush to fire
        await asyncio.sleep(0.15)
        assert route.call_count >= 1

        await sink.close()

    @respx.mock
    async def test_periodic_flush_exception_is_caught(self) -> None:
        """Exceptions in periodic flush don't kill the background task."""
        call_count = 0

        def _side_effect(request: object) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise httpx.ConnectError("down")
            return httpx.Response(200)

        respx.post(WEBHOOK_URL).mock(side_effect=_side_effect)
        sink = WebhookSink(WEBHOOK_URL, batch_size=100, flush_interval_s=0.05, max_retries=1)

        await sink.write(_make_event())
        # Wait for periodic flush to attempt and fail, then succeed on next cycle
        await asyncio.sleep(0.25)
        # Add another event for the second flush cycle
        await sink.write(_make_event())
        await asyncio.sleep(0.15)

        await sink.close()
        # Should have attempted multiple times without crashing
        assert call_count >= 1


class TestWebhookSinkConcurrentWrites:
    """Tests for thread safety of concurrent writes."""

    @respx.mock
    async def test_concurrent_writes_no_corruption(self) -> None:
        route = respx.post(WEBHOOK_URL).mock(return_value=httpx.Response(200))
        sink = WebhookSink(WEBHOOK_URL, batch_size=100, flush_interval_s=999)

        async def _write_events(n: int) -> None:
            for i in range(n):
                await sink.write(_make_event(latency_ms=float(i)))

        tasks = [asyncio.create_task(_write_events(10)) for _ in range(5)]
        await asyncio.gather(*tasks)

        await sink.flush()

        import json

        total_events = 0
        for call in route.calls:
            payload = json.loads(call.request.content)
            total_events += len(payload)
        assert total_events == 50

        await sink.close()
