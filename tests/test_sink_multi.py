"""Tests for MultiSink and CallbackSink."""

from datetime import UTC, datetime

from llmtrace.models import TraceEvent
from llmtrace.sinks.base import BaseSink
from llmtrace.sinks.callback import CallbackSink
from llmtrace.sinks.multi import MultiSink


def _make_event(**kwargs: object) -> TraceEvent:
    """Create a TraceEvent with sensible defaults for testing."""
    defaults: dict[str, object] = {
        "provider": "openai",
        "model": "gpt-4",
        "latency_ms": 100.0,
        "timestamp": datetime(2024, 6, 15, 14, 30, 45, tzinfo=UTC),
    }
    defaults.update(kwargs)
    return TraceEvent(**defaults)  # type: ignore[arg-type]


class TestMultiSinkDispatch:
    """Tests for MultiSink event dispatch."""

    async def test_dispatches_to_all_child_sinks(self) -> None:
        received: list[list[TraceEvent]] = [[], [], []]
        sinks = [CallbackSink(callback=r.append) for r in received]
        multi = MultiSink(sinks)
        event = _make_event()

        await multi.write(event)

        for r in received:
            assert len(r) == 1
            assert r[0] is event

    async def test_continues_if_one_sink_raises(self) -> None:
        received: list[TraceEvent] = []
        sinks: list[BaseSink] = [
            _RaisingSink(),
            CallbackSink(callback=received.append),
            CallbackSink(callback=received.append),
        ]
        multi = MultiSink(sinks)
        event = _make_event()

        await multi.write(event)

        assert len(received) == 2

    async def test_flush_propagates_to_all_children(self) -> None:
        flushed: list[bool] = []
        sinks = [CallbackSink(callback=lambda e: None) for _ in range(3)]
        # Monkey-patch flush to track calls
        for sink in sinks:
            sink.flush = _make_tracking_flush(flushed)  # type: ignore[assignment]
        multi = MultiSink(sinks)

        await multi.flush()

        assert len(flushed) == 3

    async def test_close_propagates_to_all_children(self) -> None:
        closed: list[bool] = []
        sinks = [CallbackSink(callback=lambda e: None) for _ in range(3)]
        for sink in sinks:
            sink.close = _make_tracking_close(closed)  # type: ignore[assignment]
        multi = MultiSink(sinks)

        await multi.close()

        assert len(closed) == 3


class TestCallbackSinkSync:
    """Tests for CallbackSink with synchronous callbacks."""

    async def test_sync_callback_receives_event(self) -> None:
        received: list[TraceEvent] = []
        sink = CallbackSink(callback=received.append)
        event = _make_event()

        await sink.write(event)

        assert received == [event]

    async def test_sync_callback_exception_does_not_crash(self) -> None:
        def exploding(event: TraceEvent) -> None:
            raise ValueError("kaboom")

        sink = CallbackSink(callback=exploding)
        event = _make_event()

        await sink.write(event)  # Should not raise


class TestCallbackSinkAsync:
    """Tests for CallbackSink with asynchronous callbacks."""

    async def test_async_callback_receives_event(self) -> None:
        received: list[TraceEvent] = []

        async def async_cb(event: TraceEvent) -> None:
            received.append(event)

        sink = CallbackSink(callback=async_cb)
        event = _make_event()

        await sink.write(event)

        assert received == [event]

    async def test_async_callback_exception_does_not_crash(self) -> None:
        async def exploding(event: TraceEvent) -> None:
            raise RuntimeError("async boom")

        sink = CallbackSink(callback=exploding)
        event = _make_event()

        await sink.write(event)  # Should not raise


class TestProtocolsImport:
    """Ensure protocols module is importable and covers its definitions."""

    def test_protocols_define_expected_names(self) -> None:
        from llmtrace import protocols

        assert hasattr(protocols, "Sink")
        assert hasattr(protocols, "SinkSync")
        assert hasattr(protocols, "Instrumentor")
        assert hasattr(protocols, "Enricher")


# ── Helpers ──────────────────────────────────────────────────────────


class _RaisingSink(BaseSink):
    """Test sink that raises on every write."""

    async def write(self, event: TraceEvent) -> None:
        raise RuntimeError("boom")

    async def flush(self) -> None:
        pass

    async def close(self) -> None:
        pass


def _make_tracking_flush(tracker: list[bool]):  # type: ignore[type-arg]
    """Create an async flush that appends True to the tracker."""

    async def _flush() -> None:
        tracker.append(True)

    return _flush


def _make_tracking_close(tracker: list[bool]):  # type: ignore[type-arg]
    """Create an async close that appends True to the tracker."""

    async def _close() -> None:
        tracker.append(True)

    return _close
