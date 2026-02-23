"""Integration tests for the llmtrace public API."""

from __future__ import annotations

import asyncio
import sys
import types
from typing import Any

import pytest

import llmtrace
from llmtrace import configure, emit, get_config, instrument, reset, uninstrument
from llmtrace.capture.context import span, span_sync
from llmtrace.capture.decorator import trace
from llmtrace.config import _resolve_sink
from llmtrace.models import TraceEvent
from llmtrace.sinks.callback import CallbackSink

# ── Mock Anthropic SDK ───────────────────────────────────────────────────


class MockUsage:
    def __init__(self) -> None:
        self.input_tokens = 100
        self.output_tokens = 50


class MockResponse:
    def __init__(self) -> None:
        self.model = "claude-3-sonnet-20240229"
        self.usage = MockUsage()
        self.content: list[Any] = []

    def model_dump(self) -> dict[str, Any]:
        return {"model": self.model, "content": []}


class MockMessages:
    def create(self, **kwargs: Any) -> MockResponse:
        return MockResponse()


class MockAsyncMessages:
    async def create(self, **kwargs: Any) -> MockResponse:
        return MockResponse()


# ── Fixtures ─────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _clean_state() -> Any:
    """Reset config and clear active instrumentors before/after each test."""
    reset()
    llmtrace._active_instrumentors.clear()
    yield
    llmtrace._active_instrumentors.clear()
    reset()


@pytest.fixture()
def mock_anthropic() -> Any:
    """Install a fake anthropic package into sys.modules."""
    original_sync = MockMessages.create
    original_async = MockAsyncMessages.create

    saved: dict[str, types.ModuleType] = {}
    for key in list(sys.modules):
        if key == "anthropic" or key.startswith("anthropic."):
            saved[key] = sys.modules.pop(key)
    sys.modules.pop("llmtrace.instruments.anthropic", None)

    mod = types.ModuleType("anthropic")
    resources = types.ModuleType("anthropic.resources")
    resources.Messages = MockMessages  # type: ignore[attr-defined]
    resources.AsyncMessages = MockAsyncMessages  # type: ignore[attr-defined]
    mod.resources = resources  # type: ignore[attr-defined]

    sys.modules["anthropic"] = mod
    sys.modules["anthropic.resources"] = resources

    yield mod

    MockMessages.create = original_sync  # type: ignore[assignment]
    MockAsyncMessages.create = original_async  # type: ignore[assignment]

    sys.modules.pop("anthropic", None)
    sys.modules.pop("anthropic.resources", None)
    sys.modules.pop("llmtrace.instruments.anthropic", None)
    sys.modules.update(saved)


@pytest.fixture()
def captured_events() -> list[TraceEvent]:
    """Configure a CallbackSink that collects events into a list."""
    events: list[TraceEvent] = []
    configure(sink=CallbackSink(lambda e: events.append(e)))
    return events


# ── Tests: Full Flow ─────────────────────────────────────────────────────


class TestFullFlow:
    """Full end-to-end: configure → instrument → call → verify events."""

    def test_configure_instrument_call_captures_event(
        self, mock_anthropic: Any, captured_events: list[TraceEvent]
    ) -> None:
        instrument("anthropic")

        messages = MockMessages()
        messages.create(model="claude-3-sonnet", max_tokens=100)

        assert len(captured_events) == 1
        assert captured_events[0].provider == "anthropic"

        uninstrument("anthropic")


# ── Tests: instrument / uninstrument ─────────────────────────────────────


class TestInstrumentUninstrument:
    """Lifecycle tests for instrument() and uninstrument()."""

    def test_instrument_and_uninstrument_cycle(
        self, mock_anthropic: Any, captured_events: list[TraceEvent]
    ) -> None:
        instrument("anthropic")

        messages = MockMessages()
        messages.create(model="claude-3-sonnet", max_tokens=100)
        assert len(captured_events) == 1

        uninstrument("anthropic")

        messages.create(model="claude-3-sonnet", max_tokens=100)
        assert len(captured_events) == 1  # no new event after uninstrument

    def test_unknown_provider_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown provider"):
            instrument("nonexistent")

    def test_double_instrument_is_idempotent(
        self, mock_anthropic: Any, captured_events: list[TraceEvent]
    ) -> None:
        instrument("anthropic")
        instrument("anthropic")  # second call is a no-op

        messages = MockMessages()
        messages.create(model="claude-3-sonnet", max_tokens=100)

        assert len(captured_events) == 1  # not doubled

    def test_uninstrument_without_args_clears_all(
        self, mock_anthropic: Any, captured_events: list[TraceEvent]
    ) -> None:
        instrument("anthropic")
        assert len(llmtrace._active_instrumentors) == 1

        uninstrument()  # no args → clears all

        assert len(llmtrace._active_instrumentors) == 0

    def test_uninstrument_nonexistent_provider_is_safe(self) -> None:
        """uninstrument for a provider not in _active_instrumentors is a no-op."""
        uninstrument("not_active")  # pop returns None, no error


# ── Tests: emit ──────────────────────────────────────────────────────────


class TestEmit:
    """Tests for manually emitting TraceEvents."""

    def test_emit_sends_event_to_sink(self) -> None:
        events: list[TraceEvent] = []
        configure(sink=CallbackSink(lambda e: events.append(e)))

        event = TraceEvent(provider="manual", model="test-model", latency_ms=42.0)
        emit(event)

        assert len(events) == 1
        assert events[0].provider == "manual"
        assert events[0].model == "test-model"

    def test_emit_with_no_sink_is_noop(self) -> None:
        """emit() does nothing when sink is None."""
        configure(sink=None)
        event = TraceEvent(provider="test", model="m", latency_ms=1.0)
        emit(event)  # should not raise

    async def test_emit_in_async_context(self) -> None:
        """emit() uses create_task when an event loop is running."""
        events: list[TraceEvent] = []
        configure(sink=CallbackSink(lambda e: events.append(e)))

        event = TraceEvent(provider="async-test", model="m", latency_ms=1.0)
        emit(event)

        # Give the task a chance to run
        await asyncio.sleep(0.05)

        assert len(events) == 1
        assert events[0].provider == "async-test"


# ── Tests: __all__ exports ───────────────────────────────────────────────


class TestExports:
    """Verify that all public exports are importable."""

    def test_all_exports_are_accessible(self) -> None:
        for name in llmtrace.__all__:
            obj = getattr(llmtrace, name)
            assert obj is not None, f"llmtrace.{name} resolved to None"


# ── Tests: config sink resolution ────────────────────────────────────────


class TestConfigSinkResolution:
    """Cover _resolve_sink for jsonfile and webhook string formats."""

    def test_resolve_jsonfile_sink(self, tmp_path: Any) -> None:
        path = str(tmp_path / "trace.jsonl")
        configure(sink=f"jsonfile:{path}")
        config = get_config()
        assert config.sink is not None

    def test_resolve_webhook_sink(self) -> None:
        configure(sink="webhook:https://example.com/hook")
        config = get_config()
        assert config.sink is not None

    def test_resolve_unknown_sink_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown sink format"):
            _resolve_sink("unknown:foo")

    def test_env_llmtrace_sink(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """LLMTRACE_SINK env var is used as fallback."""
        monkeypatch.setenv("LLMTRACE_SINK", "console")
        configure()  # no sink kwarg — falls back to env
        config = get_config()
        assert config.sink is not None


# ── Tests: span event flushing ───────────────────────────────────────────


class TestSpanEventFlushing:
    """Cover add_event and span flush-on-exit paths in capture/context.py."""

    async def test_async_span_flushes_events(self) -> None:
        """Events added via add_event are flushed to sink when span exits."""
        events: list[TraceEvent] = []
        configure(sink=CallbackSink(lambda e: events.append(e)))

        event = TraceEvent(provider="test", model="m", latency_ms=1.0)
        async with span("test-span") as handle:
            handle.add_event(event)

        assert len(events) == 1
        assert events[0].provider == "test"

    def test_sync_span_flushes_events(self) -> None:
        """Events added via add_event are flushed for sync spans."""
        events: list[TraceEvent] = []
        configure(sink=CallbackSink(lambda e: events.append(e)))

        event = TraceEvent(provider="test", model="m", latency_ms=1.0)
        with span_sync("test-span") as handle:
            handle.add_event(event)

        assert len(events) == 1

    def test_sync_span_add_event_on_handle(self) -> None:
        """SpanHandleSync.add_event appends to context.events."""
        configure(sink=None)
        with span_sync("s") as handle:
            event = TraceEvent(provider="p", model="m", latency_ms=0.0)
            handle.add_event(event)
            assert len(handle.context.events) == 1


# ── Tests: decorator enricher failure + provider detection ───────────────


class TestDecoratorEdgeCases:
    """Cover enricher failure and auto-detection in capture/decorator.py."""

    def test_enricher_failure_does_not_crash(self) -> None:
        """A broken enricher logs a warning but doesn't break tracing."""
        events: list[TraceEvent] = []

        def bad_enricher(event: TraceEvent) -> TraceEvent:
            msg = "enricher boom"
            raise RuntimeError(msg)

        configure(
            sink=CallbackSink(lambda e: events.append(e)),
            enrichers=[bad_enricher],
        )

        @trace(provider="test")
        def call_llm(**kwargs: Any) -> MockResponse:
            return MockResponse()

        call_llm(model="m")
        assert len(events) == 1

    def test_detect_openai_provider(self) -> None:
        """Auto-detect 'openai' from result module name."""
        events: list[TraceEvent] = []
        configure(sink=CallbackSink(lambda e: events.append(e)))

        # Create an object whose type has 'openai' in __module__
        openai_mod = types.ModuleType("openai.types.chat")
        openai_mod.__name__ = "openai.types.chat"  # type: ignore[attr-defined]
        fake_result_cls = type("ChatCompletion", (), {"__module__": "openai.types.chat"})
        result_obj = fake_result_cls()
        result_obj.model = "gpt-4"  # type: ignore[attr-defined]
        result_obj.usage = None  # type: ignore[attr-defined]

        @trace()  # no provider — auto-detect
        def call_openai(**kwargs: Any) -> Any:
            return result_obj

        call_openai(model="gpt-4")
        assert len(events) == 1
        assert events[0].provider == "openai"

    def test_detect_google_provider(self) -> None:
        """Auto-detect 'google' from result module name."""
        events: list[TraceEvent] = []
        configure(sink=CallbackSink(lambda e: events.append(e)))

        fake_result_cls = type("GenerateResponse", (), {"__module__": "google.genai.types"})
        result_obj = fake_result_cls()
        result_obj.model = "gemini-pro"  # type: ignore[attr-defined]
        result_obj.usage = None  # type: ignore[attr-defined]

        @trace()
        def call_google(**kwargs: Any) -> Any:
            return result_obj

        call_google(model="gemini-pro")
        assert len(events) == 1
        assert events[0].provider == "google"

    def test_sync_trace_sample_rate_skip(self) -> None:
        """sync @trace respects sample_rate=0 to skip all events."""
        events: list[TraceEvent] = []
        configure(
            sink=CallbackSink(lambda e: events.append(e)),
            sample_rate=0.0,
        )

        @trace(provider="test")
        def call_llm(**kwargs: Any) -> str:
            return "ok"

        result = call_llm(model="m")
        assert result == "ok"
        assert len(events) == 0

    def test_sync_trace_with_sync_sink(self) -> None:
        """@trace with a synchronous sink calls sink.write directly."""
        events: list[TraceEvent] = []

        class SyncSink:
            def write(self, event: TraceEvent) -> None:
                events.append(event)

            def flush(self) -> None:
                pass

            def close(self) -> None:
                pass

        configure(sink=SyncSink())

        @trace(provider="test")
        def call_llm(**kwargs: Any) -> MockResponse:
            return MockResponse()

        call_llm(model="m")
        assert len(events) == 1


# ── Tests: BaseInstrumentor edge cases ───────────────────────────────────


class TestBaseInstrumentorEdgeCases:
    """Cover _base.py: enricher failure, _emit_sync sync sink, uninstrument not-instrumented."""

    def test_uninstrument_when_not_instrumented(self, mock_anthropic: Any) -> None:
        """Calling uninstrument on a fresh instrumentor is a no-op."""
        from llmtrace.instruments.anthropic import AnthropicInstrumentor

        instr = AnthropicInstrumentor()
        instr.uninstrument()  # not instrumented yet — should not raise

    def test_sync_wrapper_with_sync_sink(self, mock_anthropic: Any) -> None:
        """Sync wrapper with a SinkSync writes events directly."""
        events: list[TraceEvent] = []

        class SyncSink:
            def write(self, event: TraceEvent) -> None:
                events.append(event)

            def flush(self) -> None:
                pass

            def close(self) -> None:
                pass

        configure(sink=SyncSink())

        from llmtrace.instruments.anthropic import AnthropicInstrumentor

        instr = AnthropicInstrumentor()
        instr.instrument()

        messages = MockMessages()
        messages.create(model="claude-3-sonnet", max_tokens=100)

        assert len(events) == 1

        instr.uninstrument()

    def test_enricher_failure_in_base_instrumentor(self, mock_anthropic: Any) -> None:
        """Enricher exception in _base._build_event is caught."""

        def bad_enricher(event: TraceEvent) -> TraceEvent:
            msg = "boom"
            raise RuntimeError(msg)

        events: list[TraceEvent] = []
        configure(
            sink=CallbackSink(lambda e: events.append(e)),
            enrichers=[bad_enricher],
        )

        from llmtrace.instruments.anthropic import AnthropicInstrumentor

        instr = AnthropicInstrumentor()
        instr.instrument()

        messages = MockMessages()
        messages.create(model="claude-3-sonnet", max_tokens=100)

        # Event is still emitted despite enricher failure
        assert len(events) == 1

        instr.uninstrument()

    def test_sync_wrapper_exception_captures_error(self, mock_anthropic: Any) -> None:
        """Sync wrapper captures ErrorTrace when the call raises."""
        events: list[TraceEvent] = []
        configure(sink=CallbackSink(lambda e: events.append(e)))

        from llmtrace.instruments.anthropic import AnthropicInstrumentor

        # Patch create to raise
        original_create = MockMessages.create

        def raising_create(self: Any, **kwargs: Any) -> Any:
            msg = "API error"
            raise RuntimeError(msg)

        MockMessages.create = raising_create  # type: ignore[assignment]

        instr = AnthropicInstrumentor()
        instr.instrument()

        messages = MockMessages()
        with pytest.raises(RuntimeError, match="API error"):
            messages.create(model="claude-3-sonnet", max_tokens=100)

        assert len(events) == 1
        assert events[0].error is not None
        assert events[0].error.error_type == "RuntimeError"

        instr.uninstrument()
        MockMessages.create = original_create  # type: ignore[assignment]

    async def test_async_wrapper_exception_captures_error(self, mock_anthropic: Any) -> None:
        """Async wrapper captures ErrorTrace when the call raises."""
        events: list[TraceEvent] = []
        configure(sink=CallbackSink(lambda e: events.append(e)))

        from llmtrace.instruments.anthropic import AnthropicInstrumentor

        original_create = MockAsyncMessages.create

        async def raising_create(self: Any, **kwargs: Any) -> Any:
            msg = "Async API error"
            raise RuntimeError(msg)

        MockAsyncMessages.create = raising_create  # type: ignore[assignment]

        instr = AnthropicInstrumentor()
        instr.instrument()

        messages = MockAsyncMessages()
        with pytest.raises(RuntimeError, match="Async API error"):
            await messages.create(model="claude-3-sonnet", max_tokens=100)

        assert len(events) == 1
        assert events[0].error is not None
        assert events[0].error.error_type == "RuntimeError"

        instr.uninstrument()
        MockAsyncMessages.create = original_create  # type: ignore[assignment]

    def test_sync_wrapper_sample_rate_skip(self, mock_anthropic: Any) -> None:
        """Sync wrapper with sample_rate=0 skips tracing."""
        events: list[TraceEvent] = []
        configure(
            sink=CallbackSink(lambda e: events.append(e)),
            sample_rate=0.0,
        )

        from llmtrace.instruments.anthropic import AnthropicInstrumentor

        instr = AnthropicInstrumentor()
        instr.instrument()

        messages = MockMessages()
        result = messages.create(model="claude-3-sonnet", max_tokens=100)

        assert isinstance(result, MockResponse)
        assert len(events) == 0

        instr.uninstrument()


# ── Tests: _emit_event_sync branches ─────────────────────────────────────


class TestEmitEventSync:
    """Cover _emit_event_sync in capture/context.py for sync sink path."""

    def test_sync_span_with_sync_sink_flushes(self) -> None:
        """span_sync with a synchronous sink calls sink.write directly."""
        events: list[TraceEvent] = []

        class SyncSink:
            def write(self, event: TraceEvent) -> None:
                events.append(event)

            def flush(self) -> None:
                pass

            def close(self) -> None:
                pass

        configure(sink=SyncSink())

        event = TraceEvent(provider="test", model="m", latency_ms=1.0)
        with span_sync("s") as handle:
            handle.add_event(event)

        assert len(events) == 1


# ── Tests: _emit_sync no-sink early return in _base.py ───────────────────


class TestEmitSyncNoSink:
    """Cover _emit_sync returning early when sink is None."""

    def test_base_emit_sync_no_sink(self, mock_anthropic: Any) -> None:
        """_emit_sync returns immediately when config.sink is None."""
        configure(sink=None)

        from llmtrace.instruments.anthropic import AnthropicInstrumentor

        instr = AnthropicInstrumentor()
        instr.instrument()

        messages = MockMessages()
        # Should not raise even with no sink
        messages.create(model="claude-3-sonnet", max_tokens=100)

        instr.uninstrument()


# ── Tests: sync wrapper emit exception in _base.py ───────────────────────


class TestSyncWrapperEmitException:
    """Cover sync wrapper catching emit exceptions in _base.py."""

    def test_sync_emit_exception_is_caught(self, mock_anthropic: Any) -> None:
        """If _build_event or sink.write raises, sync wrapper catches it."""

        class ExplodingSink:
            def write(self, event: TraceEvent) -> None:
                msg = "sink exploded"
                raise RuntimeError(msg)

            def flush(self) -> None:
                pass

            def close(self) -> None:
                pass

        configure(sink=ExplodingSink())

        from llmtrace.instruments.anthropic import AnthropicInstrumentor

        instr = AnthropicInstrumentor()
        instr.instrument()

        messages = MockMessages()
        # Should not raise — emit exception is caught
        result = messages.create(model="claude-3-sonnet", max_tokens=100)
        assert isinstance(result, MockResponse)

        instr.uninstrument()


# ── Tests: async wrapper emit exception in _base.py ──────────────────────


class TestAsyncWrapperEmitException:
    """Cover async wrapper catching emit exceptions in _base.py."""

    async def test_async_emit_exception_is_caught(self, mock_anthropic: Any) -> None:
        """If sink.write raises in async wrapper, it's caught."""

        class ExplodingAsyncSink:
            async def write(self, event: TraceEvent) -> None:
                msg = "async sink exploded"
                raise RuntimeError(msg)

            async def flush(self) -> None:
                pass

            async def close(self) -> None:
                pass

        configure(sink=ExplodingAsyncSink())

        from llmtrace.instruments.anthropic import AnthropicInstrumentor

        instr = AnthropicInstrumentor()
        instr.instrument()

        messages = MockAsyncMessages()
        # Should not raise — emit exception is caught
        result = await messages.create(model="claude-3-sonnet", max_tokens=100)
        assert isinstance(result, MockResponse)

        instr.uninstrument()


# ── Tests: decorator sync emit failure ───────────────────────────────────


class TestDecoratorSyncEmitFailure:
    """Cover decorator.py sync emit exception path (lines 258-259)."""

    def test_sync_trace_emit_failure_is_caught(self) -> None:
        """If sink.write raises in sync @trace, the exception is caught."""

        class ExplodingSink:
            def write(self, event: TraceEvent) -> None:
                msg = "sync emit exploded"
                raise RuntimeError(msg)

            def flush(self) -> None:
                pass

            def close(self) -> None:
                pass

        configure(sink=ExplodingSink())

        @trace(provider="test")
        def call_llm(**kwargs: Any) -> MockResponse:
            return MockResponse()

        # Should not raise
        result = call_llm(model="m")
        assert isinstance(result, MockResponse)


# ── Tests: _dump_response fallback in extractors.py ──────────────────────


class TestDumpResponseFallback:
    """Cover extractors.py _dump_response when model_dump() raises."""

    def test_model_dump_exception_falls_back_to_str(self) -> None:
        from llmtrace.capture.extractors import _dump_response

        class BadModelDump:
            def model_dump(self) -> dict[str, Any]:
                msg = "serialization failed"
                raise TypeError(msg)

        result = _dump_response(BadModelDump())
        assert "raw" in result
        assert "BadModelDump" in result["raw"]
